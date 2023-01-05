// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/anchor_utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using static TorchSharp.torch;
using TorchSharp.Utils;

namespace TorchSharp
{
    namespace Modules.Detection
    {
        /// <summary>
        /// //Module that generates anchors for a set of feature maps and
        /// image sizes.
        /// 
        /// The module support computing anchors at multiple sizes and aspect ratios
        /// per feature map. This module assumes aspect ratio = height / width for
        /// each anchor.
        /// 
        /// sizes and aspect_ratios should have the same number of elements, and it should
        /// correspond to the number of feature maps.
        /// 
        /// sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
        /// and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
        /// per spatial location for feature map i.
        /// </summary>
        public class AnchorGenerator : nn.Module<ImageList, List<Tensor>, List<Tensor>>
        {
            private Dictionary<string, List<Tensor>> __annotations__;
            private List<List<int>> sizes;
            private List<List<double>> aspect_ratios;
            private List<Tensor> cell_anchors;

            /// <summary>
            /// Constructor.
            /// </summary>
            public AnchorGenerator(
                string name,
                List<List<int>> sizes = null,
                List<List<double>> aspect_ratios = null
            ) : base(name)
            {
                __annotations__ = new Dictionary<string, List<Tensor>>();
                __annotations__.Add("cell_anchors", new List<Tensor>());

                if (sizes == null)
                    sizes = new List<List<int>> { new List<int> { 128, 256, 512 } };
                if (aspect_ratios == null)
                    aspect_ratios = new List<List<double>> { new List<double> { 0.5, 1.0, 2.0 } };

                this.sizes = sizes;
                this.aspect_ratios = aspect_ratios;
                List<Tensor> anchors = new List<Tensor>();
                for (int i = 0; i < sizes.Count; i++) {
                    anchors.Add(this.generate_anchors(sizes[i], aspect_ratios[i]));
                }
                this.cell_anchors = anchors;
            }

            //# TODO: https://github.com/pytorch/pytorch/issues/26792
            //# For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
            //# (scales, aspect_ratios) are usually an element of zip(this.scales, this.aspect_ratios)
            //# This method assumes aspect ratio = height / width for an anchor.
            public Tensor generate_anchors(
                List<int> scales,
                List<double> aspect_ratios,
                torch.ScalarType dtype = torch.ScalarType.Float32,
                torch.Device device = null
            )
            {
                if (device == null)
                    device = torch.CPU;

                var scales_tensor = torch.as_tensor(scales, dtype: dtype, device: device);
                var aspect_ratios_tensor = torch.as_tensor(aspect_ratios, dtype: dtype, device: device);
                var h_ratios = torch.sqrt(aspect_ratios_tensor);
                var w_ratios = 1 / h_ratios;

                var ws = (w_ratios[torch.TensorIndex.Colon, torch.TensorIndex.None] * scales_tensor[torch.TensorIndex.None, torch.TensorIndex.Colon]).view(-1);
                var hs = (h_ratios[torch.TensorIndex.Colon, torch.TensorIndex.None] * scales_tensor[torch.TensorIndex.None, torch.TensorIndex.Colon]).view(-1);

                var base_anchors = torch.stack(new List<Tensor> { -ws, -hs, ws, hs }, dim: 1) / 2;
                return base_anchors.round();
            }

            public void set_cell_anchors(torch.ScalarType dtype, torch.Device device)
            {
                List<Tensor> cells = new List<Tensor>();
                foreach (var cell_anchor in this.cell_anchors)
                    cells.Add(cell_anchor.to(dtype, device));
                this.cell_anchors = cells;
            }

            public List<int> num_anchors_per_location()
            {
                List<int> num_anchors = new List<int>();
                for (int i = 0; i < this.sizes.Count; i++)
                    num_anchors.Add(sizes[i].Count * aspect_ratios[i].Count);
                return num_anchors;
            }

            //# For every combination of (a, (g, s), i) in (this.cell_anchors, zip(grid_sizes, strides), 0:2),
            //# output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
            public List<Tensor> grid_anchors(List<List<long>> grid_sizes, List<List<Tensor>> strides)
            {
                var anchors = new List<Tensor>();
                var cell_anchors = this.cell_anchors;
                Debug.Assert(cell_anchors != null, "cell_anchors should not be None");
                Debug.Assert(
                    grid_sizes.Count == strides.Count && strides.Count == cell_anchors.Count,
                    "Anchors should be Tuple[Tuple[int]] because each feature " +
                    "map could potentially have different sizes and aspect ratios. " +
                    "There needs to be a match between the number of " +
                    "feature maps passed and the number of sizes / aspect ratios specified."
                );

                for (int i = 0; i < grid_sizes.Count; i++) {
                    var size = grid_sizes[i];
                    var stride = strides[i];
                    var base_anchors = cell_anchors[i];

                    var (grid_height, grid_width, _) = size;
                    var (stride_height, stride_width, _) = stride;

                    var device = base_anchors.device;

                    //# For output anchor, compute [x_center, y_center, x_center, y_center]
                    var shifts_x = torch.arange(0, grid_width, dtype: torch.ScalarType.Int32, device: device) * stride_width;
                    var shifts_y = torch.arange(0, grid_height, dtype: torch.ScalarType.Int32, device: device) * stride_height;

                    Tensor[] shifts = new Tensor[] { shifts_y, shifts_x };

                    var (shift_x, shift_y, _) = torch.meshgrid(shifts, indexing: "ij");
                    shift_x = shift_x.reshape(-1);
                    shift_y = shift_y.reshape(-1);
                    var shiftsResult = torch.stack(new Tensor[] { shift_x, shift_y, shift_x, shift_y }, dim: 1);

                    //            # For every (base anchor, output anchor) pair,
                    //# offset each zero-centered base anchor by the center of the output anchor.
                    anchors.Add((shiftsResult.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4));
                }

                return anchors;
            }

            public override List<Tensor> forward(ImageList image_list, List<Tensor> feature_maps)
            {
                List<List<long>> grid_sizes = new List<List<long>>();
                foreach (var feature_map in feature_maps) {
                    List<long> grid = new List<long>();
                    grid.Add(feature_map.shape[feature_map.shape.Length - 2]);
                    grid.Add(feature_map.shape[feature_map.shape.Length - 1]);
                    grid_sizes.Add(grid);
                }

                List<long> image_size = new List<long>();
                image_size.Add(image_list.tensors.shape[image_list.tensors.shape.Length - 2]);
                image_size.Add(image_list.tensors.shape[image_list.tensors.shape.Length - 1]);

                var (dtype, device) = (feature_maps[0].dtype, feature_maps[0].device);

                var strides = new List<List<Tensor>>();
                foreach (var g in grid_sizes) {
                    List<Tensor> item = new List<Tensor>();
                    item.Add(torch.empty(new long[0], dtype: torch.int64, device: device).fill_(image_size[0] / g[0]));
                    item.Add(torch.empty(new long[0], dtype: torch.int64, device: device).fill_(image_size[1] / g[1]));
                    strides.Add(item);
                }

                this.set_cell_anchors(dtype, device);
                var anchors_over_all_feature_maps = this.grid_anchors(grid_sizes, strides);
                List<List<Tensor>> anchors = new List<List<Tensor>>();
                for (int i = 0; i < image_list.image_sizes.Count; i++) {
                    List<Tensor> anchors_in_image = new List<Tensor>();
                    foreach (var anchors_per_feature_map in anchors_over_all_feature_maps)
                        anchors_in_image.Add(anchors_per_feature_map);
                    anchors.Add(anchors_in_image);
                }
                List<Tensor> result = new List<Tensor>();
                foreach (var anchors_per_image in anchors)
                    result.Add(torch.cat(anchors_per_image));
                return result;
            }
        }

        /// <summary>
        /// This module generates the default boxes of SSD for a set of feature maps and image sizes.
        /// </summary>
        public class DefaultBoxGenerator : nn.Module<ImageList, List<Tensor>, List<Tensor>>
        {
            public List<List<int>> aspect_ratios { get; set; }
            public List<int> steps { get; set; }
            public bool clip { get; set; }
            public List<float> scales { get; set; }
            public List<Tensor> _wh_pairs { get; set; }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="aspect_ratios">A list with all the aspect ratios used in each feature map.</param>
            /// <param name="min_ratio">The minimum scale :math:`\text{s}_{\text{min}}` of the default boxes used in the estimation
            ///        of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.</param>
            /// <param name="max_ratio">The maximum scale :math:`\text{s}_{\text{max}}`  of the default boxes used in the estimation
            ///        of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.</param>
            /// <param name="scales">The scales of the default boxes. If not provided it will be estimated using
            ///        the ``min_ratio`` and ``max_ratio`` parameters.</param>
            /// <param name="steps">It's a hyper-parameter that affects the tiling of defalt boxes. If not provided
            ///        it will be estimated from the data.</param>
            /// <param name="clip">Whether the standardized values of default boxes should be clipped between 0 and 1. The clipping
            ///        is applied while the boxes are encoded in format ``(cx, cy, w, h)``.</param>
            /// <exception cref="ArgumentException"></exception>
            public DefaultBoxGenerator(string name,
                List<List<int>> aspect_ratios,
                float min_ratio = 0.15f,
                float max_ratio = 0.9f,
                List<float> scales = null,
                List<int> steps = null,
                bool clip = true) : base(name)
            {
                if (steps != null && aspect_ratios.Count != steps.Count)
                    throw new ArgumentException("aspect_ratios and steps should have the same length");
                this.aspect_ratios = aspect_ratios;
                this.steps = steps;
                this.clip = clip;
                var num_outputs = aspect_ratios.Count;

                //# Estimation of default boxes scales
                if (scales == null) {
                    if (num_outputs > 1) {
                        var range_ratio = max_ratio - min_ratio;
                        this.scales = new List<float>();
                        for (int k = 0; k < num_outputs; k++)
                            scales.Add((float)(min_ratio + range_ratio * k / (num_outputs - 1.0)));
                        this.scales.Add(1.0f);
                    } else
                        this.scales = new List<float> { min_ratio, max_ratio };
                } else
                    this.scales = scales;

                this._wh_pairs = this._generate_wh_pairs(num_outputs);
            }

            private List<Tensor> _generate_wh_pairs(
                int num_outputs, torch.ScalarType dtype = torch.ScalarType.Float32,
                torch.Device device = null
            )
            {
                if (device == null)
                    device = torch.CPU;

                _wh_pairs = new List<Tensor>();
                for (int k = 0; k < num_outputs; k++) {
                    //# Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
                    var s_k = this.scales[k];
                    float s_prime_k = (float)Math.Sqrt(this.scales[k] * this.scales[k + 1]);
                    var wh_pairs = new List<(float, float)> { ( s_k, s_k ),
                    ( s_prime_k, s_prime_k ) };

                    //# Adding 2 pairs for each aspect ratio of the feature map k
                    foreach (var ar in this.aspect_ratios[k]) {
                        float sq_ar = (float)Math.Sqrt(ar);
                        var w = this.scales[k] * sq_ar;
                        var h = this.scales[k] / sq_ar;
                        wh_pairs.Add((w, h));
                        wh_pairs.Add((h, w));
                    }

                    _wh_pairs.Add(torch.as_tensor(wh_pairs, dtype: dtype, device: device));
                }
                return _wh_pairs;
            }

            public List<int> num_anchors_per_location()
            {
                //# Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feaure map.
                List<int> result = new List<int>();
                foreach (var r in this.aspect_ratios)
                    result.Add(2 + 2 * r.Count);
                return result;
            }

            //# Default Boxes calculation based on page 6 of SSD paper
            private Tensor _grid_default_boxes(
                List<List<long>> grid_sizes, List<long> image_size,
                torch.ScalarType dtype = torch.ScalarType.Float32)
            {
                var default_boxes = new List<Tensor>();
                for (int k = 0; k < grid_sizes.Count; k++) {
                    var f_k = grid_sizes[k];
                    //# Now add the default boxes for each width-height pair
                    long x_f_k = f_k[1], y_f_k = f_k[0];
                    if (this.steps != null) {
                        x_f_k = image_size[0] / this.steps[k];
                        y_f_k = image_size[1] / this.steps[k];
                    }

                    var shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(dtype);
                    var shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(dtype);
                    var array = new Tensor[] { shifts_y, shifts_x };
                    array = torch.meshgrid(array, indexing: "ij");
                    var shift_x = array[1].reshape(-1);
                    var shift_y = array[0].reshape(-1);

                    List<Tensor> shiftArray = new List<Tensor>();
                    for (int i = 0; i < this._wh_pairs[k].shape[0]; i++) {
                        shiftArray.Add(shift_x);
                        shiftArray.Add(shift_y);
                    }
                    var shifts = torch.stack(shiftArray, dim: -1).reshape(-1, 2);
                    //# Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
                    var _wh_pair = this.clip ? this._wh_pairs[k].clamp(min: 0, max: 1) : this._wh_pairs[k];
                    var wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1);

                    var default_box = torch.cat(new List<Tensor> { shifts, wh_pairs }, dim: 1);

                    default_boxes.Add(default_box);
                }

                return torch.cat(default_boxes, dim: 0);
            }

            public override string ToString()
            {
                string s = string.Format("{0},aspect_ratios={1},clip={2},scales={3},steps={4}",
                    "DefaultBoxGenerator", this.aspect_ratios, this.clip, this.scales, this.steps);

                return s;
            }

            public override List<Tensor> forward(ImageList image_list, List<Tensor> feature_maps)
            {
                var grid_sizes = new List<List<long>>();
                foreach (var feature_map in feature_maps) {
                    List<long> grid_size = new List<long>();
                    grid_size.Add(feature_map.shape[feature_map.shape.Length - 2]);
                    grid_size.Add(feature_map.shape[feature_map.shape.Length - 1]);
                    grid_sizes.Add(grid_size);
                }
                var image_size = new List<long>();
                image_size.Add(image_list.tensors.shape[image_list.tensors.shape.Length - 2]);
                image_size.Add(image_list.tensors.shape[image_list.tensors.shape.Length - 1]);

                var dtype = feature_maps[0].dtype;
                var device = feature_maps[0].device;

                var default_boxes = this._grid_default_boxes(grid_sizes, image_size, dtype: dtype);
                default_boxes = default_boxes.to(device);
                var dboxes = new List<Tensor>();

                var x_y_size = torch.tensor(new long[] { image_size[1], image_size[0] }, device: default_boxes.device);
                for (int i = 0; i < image_list.image_sizes.Count; i++) {
                    var dboxes_in_image = default_boxes;

                    dboxes_in_image = torch.cat(
                        new List<Tensor> {
                        (dboxes_in_image[torch.TensorIndex.Colon, torch.TensorIndex.Slice(null, 2)] -
                        0.5 * dboxes_in_image[torch.TensorIndex.Colon, torch.TensorIndex.Slice(2)]) * x_y_size,
                        (dboxes_in_image[torch.TensorIndex.Colon, torch.TensorIndex.Slice(null, 2)] +
                        0.5 * dboxes_in_image[torch.TensorIndex.Colon, torch.TensorIndex.Slice(2)]) * x_y_size,
                        },
                            -1
                        );

                    dboxes.Add(dboxes_in_image);
                }
                return dboxes;
            }
        }
    }
}
