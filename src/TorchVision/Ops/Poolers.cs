// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/ops/poolers.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Google.Protobuf.Collections;
using ICSharpCode.SharpZipLib.GZip;
using TorchSharp.TorchVision.Ops;
using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            // TODO: (eellison) T54974082 https://github.com/pytorch/pytorch/issues/26744/pytorch/issues/26744
            public static LevelMapper initLevelMapper(
                long k_min,
                long k_max,
                long canonical_scale = 224,
                long canonical_level = 4,
                float eps = 1e-6f
            )
            {
                return new LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps);
            }

            internal static Tensor _convert_to_roi_format(IList<Tensor> boxes)
            {
                var concat_boxes = torch.cat(boxes, dim: 0);
                var (device, dtype) = (concat_boxes.device, concat_boxes.dtype);
                var ids = new List<Tensor>();
                for (int i = 0; i < boxes.Count; i++)
                    ids.Add(torch.full_like(boxes[i][TensorIndex.Colon, TensorIndex.Slice(stop: 1)], i, dtype: dtype, device: device));

                var id = torch.cat(ids, dim: 0);
                var rois = torch.cat(new List<Tensor> { id, concat_boxes }, dim: 1);
                return rois;
            }

            internal static float _infer_scale(Tensor feature, List<long> original_size)
            {
                // assumption: the scale is of the form 2 ** (-k), with k integer
                var size = new long[] { feature.shape[feature.shape.Length - 2], feature.shape[feature.shape.Length - 1] };
                List<float> possible_scales = new List<float>();
                for (int i = 0; i < size.Length; i++) {
                    var s1 = size[i];
                    var s2 = original_size[i];
                    var approx_scale = (s1) / (s2);
                    var scale = (float)Math.Pow(2, torch.tensor(approx_scale).log2().round().item<float>());
                    possible_scales.Add(scale);
                }

                return possible_scales[0];
            }

            internal static (List<float>, LevelMapper) _setup_scales(List<Tensor> features, List<long[]> image_shapes,
                long canonical_scale, long canonical_level)
            {
                if (image_shapes == null || image_shapes.Count == 0)
                    throw new ArgumentException("images list should not be empty");
                long max_x = 0;
                long max_y = 0;
                foreach (var shape in image_shapes) {
                    max_x = Math.Max(shape[0], max_x);
                    max_y = Math.Max(shape[1], max_y);
                }
                var original_input_shape = new List<long> { max_x, max_y };

                var scales = features.Select(feat => _infer_scale(feat, original_input_shape)).ToList();
                //# get the levels in the feature map by leveraging the fact that the network always
                //# downsamples by a factor of 2 at each level.
                var lvl_min = -torch.log2(torch.tensor(scales[0], dtype: torch.float32)).item<float>();
                var lvl_max = -torch.log2(torch.tensor(scales[scales.Count - 1], dtype: torch.float32)).item<float>();

                var map_levels = initLevelMapper(
                    (long)(lvl_min),
                    (long)(lvl_max),
                    canonical_scale: canonical_scale,
                    canonical_level: canonical_level
                );
                return (scales, map_levels);
            }

            internal static List<Tensor> _filter_input(Dictionary<string, Tensor> x, List<string> featmap_names)
            {
                var x_filtered = new List<Tensor>();
                foreach (var pair in x)
                    if (featmap_names.Contains(pair.Key))
                        x_filtered.Add(pair.Value);
                return x_filtered;
            }

            /// <summary>
            /// Multiscale roi align.
            /// </summary>
            /// <param name="x_filtered">List of input tensors.</param>
            /// <param name="boxes">boxes to be used to perform the pooling operation, in
            ///            (x1, y1, x2, y2) format and in the image reference size, not the feature map
            ///            reference. The coordinate must satisfy ``0 $lt;= x1 $lt; x2`` and ``0 $lt;= y1 $lt; y2``.</param>
            /// <param name="output_size">size of the output</param>
            /// <param name="sampling_ratio">sampling ratio for ROIAlign</param>
            /// <param name="scales">If None, scales will be automatically infered. Default value is None.</param>
            /// <param name="mapper">If none, mapper will be automatically infered. Default value is None.</param>
            /// <returns></returns>
            internal static Tensor _multiscale_roi_align(IList<Tensor> x_filtered,
                IList<Tensor> boxes, IList<long> output_size, long sampling_ratio,
                IList<float> scales, LevelMapper mapper)
            {
                if (scales is null || mapper is null)
                    throw new ArgumentException("scales and mapper should not be None");

                var num_levels = x_filtered.Count;
                var rois = _convert_to_roi_format(boxes);

                if (num_levels == 1)
                    return roi_align(
                        x_filtered[0],
                        rois,
                        output_size: output_size,
                        spatial_scale: scales[0],
                        sampling_ratio: sampling_ratio
                    );

                var levels = mapper.__call__(boxes);

                var num_rois = rois.shape[0];
                var num_channels = x_filtered[0].shape[1];

                var (dtype, device) = (x_filtered[0].dtype, x_filtered[0].device);
                List<long> result_size = new List<long>();
                result_size.Add(num_rois);
                result_size.Add(num_channels);
                result_size.AddRange(output_size);
                var result = torch.zeros(
                    result_size.ToArray(),
                    dtype: dtype,
                    device: device
                );

                var tracing_results = new List<Tensor>();
                for (int level = 0; level < x_filtered.Count; level++) {
                    var per_level_feature = x_filtered[level];
                    var scale = scales[level];

                    var idx_in_level = torch.where(levels == level)[0];
                    var rois_per_level = rois[idx_in_level];

                    var result_idx_in_level = roi_align(
                        per_level_feature,
                        rois_per_level,
                        output_size: output_size,
                        spatial_scale: scale,
                        sampling_ratio: sampling_ratio
                    );

                    // result and result_idx_in_level's dtypes are based on dtypes of different
                    // elements in x_filtered.  x_filtered contains tensors output by different
                    // layers.  When autocast is active, it may choose different dtypes for
                    // different layers' outputs.  Therefore, we defensively match result's dtype
                    // before copying elements from result_idx_in_level in the following op.
                    // We need to cast manually (can't rely on autocast to cast for us) because
                    // the op acts on result in-place, and autocast only affects out-of-place ops.
                    result[idx_in_level] = result_idx_in_level.to(result.dtype);
                }

                return result;
            }
        }
    }

    namespace TorchVision.Ops
    {
        /// <summary>
        /// Determine which FPN level each RoI in a set of RoIs should map to based
        /// on the heuristic in the FPN paper.
        /// </summary>
        public class LevelMapper
        {
            private long k_min;
            private long k_max;
            private long s0;
            private long lvl0;
            private float eps;

            public LevelMapper(long k_min, long k_max, long canonical_scale = 224, long canonical_level = 4, float eps = 1e-6f)
            {
                this.k_min = k_min;
                this.k_max = k_max;
                this.s0 = canonical_scale;
                this.lvl0 = canonical_level;
                this.eps = eps;
            }

            public Tensor __call__(IList<Tensor> boxlists)
            {
                // Compute level ids
                var s = torch.sqrt(torch.cat(boxlists.Select(boxlist => torchvision.ops.box_area(boxlist)).ToArray()));

                // Eqn.(1) in FPN paper
                var target_lvls = torch.floor(this.lvl0 + torch.log2(s / this.s0) + torch.tensor(this.eps, dtype: s.dtype));
                target_lvls = torch.clamp(target_lvls, min: this.k_min, max: this.k_max);
                return (target_lvls.to(torch.int64) - this.k_min).to(torch.int64);
            }
        }

        /// <summary>
        /// Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.
        /// It infers the scale of the pooling via the heuristics specified in eq. 1
        /// of the `Feature Pyramid Network paper &lt;https://arxiv.org/abs/1612.03144>`_.
        /// They keyword-only parameters ``canonical_scale`` and ``canonical_level``
        /// correspond respectively to ``224`` and ``k0=4`` in eq. 1, and
        /// have the following meaning: ``canonical_level`` is the target level of the pyramid from
        /// which to pool a region of interest with ``w x h = canonical_scale x canonical_scale``.
        /// Examples::
        ///    >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
        ///    >>> i = OrderedDict()
        ///    >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        ///    >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        ///    >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        ///    >>> # create some random bounding boxes
        ///    >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        ///    >>> # original image size, before computing the feature maps
        ///    >>> image_sizes = [(512, 512)]
        ///    >>> output = m(i, [boxes], image_sizes)
        ///    >>> print(output.shape)
        ///    >>> torch.Size([6, 5, 3, 3])
        /// </summary>
        public class MultiScaleRoIAlign : nn.Module<Dictionary<string, Tensor>, List<Tensor>, List<long[]>, Tensor>
        {
            private List<string> featmap_names;
            private long sampling_ratio;
            private long[] output_size;
            private List<float> scales;
            private LevelMapper map_levels;
            private long canonical_scale;
            private long canonical_level;

            public long[] Output_size { get => output_size; set => output_size = value; }

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="featmap_names">the names of the feature maps that will be used for the pooling.</param>
            /// <param name="output_size">output size for the pooled region</param>
            /// <param name="sampling_ratio">sampling ratio for ROIAlign</param>
            /// <param name="canonical_scale">canonical_scale for LevelMapper</param>
            /// <param name="canonical_level">canonical_level for LevelMapper</param>
            public MultiScaleRoIAlign(string name,
                List<string> featmap_names,
                List<long> output_size,
                long sampling_ratio,
                long canonical_scale = 224,
                long canonical_level = 4) : base(name)
            {
                this.featmap_names = featmap_names;
                this.sampling_ratio = sampling_ratio;
                this.Output_size = output_size.ToArray();
                this.scales = null;
                this.map_levels = null;
                this.canonical_scale = canonical_scale;
                this.canonical_level = canonical_level;
            }

            public Tensor convert_to_roi_format(List<Tensor> boxes)
            {
                return torchvision.ops._convert_to_roi_format(boxes);
            }

            public float infer_scale(Tensor feature, List<long> original_size)
            {
                return torchvision.ops._infer_scale(feature, original_size);
            }

            public void setup_setup_scales(List<Tensor> features, List<long[]> image_shapes)
            {
                (this.scales, this.map_levels) = torchvision.ops._setup_scales(features, image_shapes, this.canonical_scale, this.canonical_level);
            }

            /// <summary>
            /// Forward method.
            /// </summary>
            /// <param name="x">feature maps for each level. They are assumed to have
            ///            all the same number of channels, but they can have different sizes.</param>
            /// <param name="boxes">boxes to be used to perform the pooling operation, in
            ///            (x1, y1, x2, y2) format and in the image reference size, not the feature map
            ///            reference. The coordinate must satisfy ``0 &lt;= x1 &lt; x2`` and ``0 &lt;= y1 &lt; y2``.</param>
            /// <param name="image_shapes">the sizes of each image before they
            ///            have been fed to a CNN to obtain feature maps. This allows us to infer the
            ///            scale factor for each one of the levels to be pooled.</param>
            /// <returns></returns>
            public override Tensor forward(Dictionary<string, Tensor> x, List<Tensor> boxes, List<long[]> image_shapes)
            {
                var x_filtered = torchvision.ops._filter_input(x, this.featmap_names);
                if (this.scales is null || this.map_levels is null)
                    (this.scales, this.map_levels) = torchvision.ops._setup_scales(
                        x_filtered, image_shapes, this.canonical_scale, this.canonical_level
                    );

                return torchvision.ops._multiscale_roi_align(
                    x_filtered,
                    boxes,
                    this.Output_size,
                    this.sampling_ratio,
                    this.scales,
                    this.map_levels
                );
            }
        }
    }
}
