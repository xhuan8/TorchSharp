// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/f56e6f63aa1d37e648b0c4cb951ce26292238c53/torchvision/ops/roi_align.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

using static TorchSharp.torch;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            /// <summary>
            /// Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.
            /// </summary>
            /// <param name="input">(Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            /// contains ``C`` feature maps of dimensions ``H x W``.
            /// If the tensor is quantized, we expect a batch size of ``N == 1``.</param>
            /// <param name="boxes">
            ///    (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            ///    format where the regions will be taken from.
            ///    The coordinate must satisfy ``0 &lt;= x1 &lt; x2`` and ``0 &lt;= y1 &lt; y2``.
            ///    If a single Tensor is passed, then the first column should
            ///    contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            ///    If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            ///    in the batch.
            /// </param>
            /// <param name="output_size">(int or Tuple[int, int]): the size of the output (in bins or pixels) after the pooling
            /// is performed, as (height, width).</param>
            /// <param name="spatial_scale">a scaling factor that maps the box coordinates to
            ///    the input coordinates. For example, if your boxes are defined on the scale
            ///    of a 224x224 image and your input is a 112x112 feature map (resulting from a 0.5x scaling of
            ///    the original image), you'll want to set this to 0.5. Default: 1.0</param>
            /// <param name="sampling_ratio">
            /// number of sampling points in the interpolation grid
            /// used to compute the output value of each pooled output bin. If > 0,
            /// then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            /// &lt;= 0, then an adaptive number of grid points are used (computed as
            /// ``ceil(roi_width / output_width)``, and likewise for height). Default: -1
            /// </param>
            /// <param name="aligned">
            /// If False, use the legacy implementation.
            ///    If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two
            ///    neighboring pixel indices. This version is used in Detectron2
            /// </param>
            /// <returns>Tensor[K, C, output_size[0], output_size[1]]: The pooled RoIs.</returns>
            public static Tensor roi_align(
                    Tensor input,
                    object boxes,
                    object output_size,
                    float spatial_scale = 1.0f,
                    long sampling_ratio = -1,
                    bool aligned = false
                )
            {
                check_roi_boxes_shape(boxes);
                object rois = boxes;
                var output_size_list = Modules.ModulesUtils._pair<int>(output_size).ToArray();
                if (rois is List<Tensor> list)
                    rois = convert_boxes_to_roi_format(list);
                var roisTensor = rois as Tensor;
                var res = LibTorchSharp.THSVision_roi_align(
                    input.Handle, roisTensor.Handle, spatial_scale, output_size_list[0], output_size_list[1], sampling_ratio, aligned);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    namespace TorchVision.Ops
    {
        /// <summary>
        /// see roi_align.
        /// </summary>
        public class RoIAlign : nn.Module
        {
            private object output_size;
            private float spatial_scale;
            private int sampling_ratio;
            private bool aligned;

            public RoIAlign(object output_size, float spatial_scale, int sampling_ratio, bool aligned = false)
                : base(string.Empty)
            {
                this.output_size = output_size;
                this.spatial_scale = spatial_scale;
                this.sampling_ratio = sampling_ratio;
                this.aligned = aligned;
            }

            public Tensor forward(Tensor input, object rois)
            {
                return torchvision.ops.roi_align(input, rois, this.output_size, this.spatial_scale, this.sampling_ratio, this.aligned);
            }

            public override string ToString()
            {
                return string.Format("{0}, output_size={1}, spatial_scale={2}, sampling_ratio={3}, aligned={4}",
                    this.GetType().Name, this.output_size, this.spatial_scale, this.sampling_ratio, this.aligned);
            }
        }
    }
}
