// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/f56e6f63aa1d37e648b0c4cb951ce26292238c53/torchvision/ops/giou_loss.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using static System.Net.WebRequestMethods;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Utils;
using System;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            /// <summary>
            /// Gradient-friendly IoU loss with an additional penalty that is non-zero when the
            /// boxes do not overlap and scales with the size of their smallest enclosing box.
            /// This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
            /// Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
            /// ``0 &lt;= x1 &lt; x2`` and ``0 &lt;= y1 &lt; y2``, and The two boxes should have the
            /// same dimensions.
            /// </summary>
            /// <param name="boxes1">first set of boxes</param>
            /// <param name="boxes2">second set of boxes</param>
            /// <param name="reduction">
            /// Specifies the reduction to apply to the output:
            /// ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: No reduction will be
            /// applied to the output. ``'mean'``: The output will be averaged.
            /// ``'sum'``: The output will be summed.Default: ``'none'``
            /// </param>
            /// <param name="eps">small number to prevent division by zero. Default: 1e-7</param>
            /// <returns>Loss tensor with the reduction option applied.</returns>
            public static Tensor generalized_box_iou_loss(Tensor boxes1, Tensor boxes2, Reduction reduction = Reduction.None,
                float eps = 1e-7f)
            {
                //Hamid Rezatofighi et. al: Generalized Intersection over Union:
                //A Metric and A Loss for Bounding Box Regression:
                //https://arxiv.org/abs/1902.09630
                // Original implementation from https://github.com/facebookresearch/fvcore/blob/bfff2ef/fvcore/nn/giou_loss.py

                boxes1 = _upcast_non_float(boxes1);
                boxes2 = _upcast_non_float(boxes2);
                var (intsctk, unionk) = _loss_inter_union(boxes1, boxes2);
                var iouk = intsctk / (unionk + eps);

                var (x1, y1, x2, y2, _) = boxes1.unbind(dimension: -1);
                var (x1g, y1g, x2g, y2g, _) = boxes2.unbind(dimension: -1);

                // smallest enclosing box
                var xc1 = torch.min(x1, x1g);
                var yc1 = torch.min(y1, y1g);
                var xc2 = torch.max(x2, x2g);
                var yc2 = torch.max(y2, y2g);

                var area_c = (xc2 - xc1) * (yc2 - yc1);
                var miouk = iouk - ((area_c - unionk) / (area_c + eps));

                var loss = 1 - miouk;

                if (reduction == Reduction.Mean)
                    loss = loss.numel() > 0 ? loss.mean() : 0.0 * loss.sum();
                else if (reduction == Reduction.Sum)
                    loss = loss.sum();

                return loss;
            }
        }
    }
}
