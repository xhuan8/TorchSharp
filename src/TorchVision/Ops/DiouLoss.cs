// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/f56e6f63aa1d37e648b0c4cb951ce26292238c53/torchvision/ops/diou_loss.py
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
            /// distance between boxes' centers isn't zero.Indeed, for two exactly overlapping
            /// boxes, the distance IoU is the same as the IoU loss.
            /// This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
            /// 
            /// Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
            /// ``0 &lt;= x1&lt;x2`` and ``0 &lt;= y1&lt;y2``, and The two boxes should have the
            /// same dimensions.
            /// </summary>
            /// <param name="boxes1">first set of boxes</param>
            /// <param name="boxes2">second set of boxes</param>
            /// <param name="reduction">Specifies the reduction to apply to the output:
            /// ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: No reduction will be
            /// applied to the output. ``'mean'``: The output will be averaged.
            /// ``'sum'``: The output will be summed.Default: ``'none'``</param>
            /// <param name="eps">small number to prevent division by zero. Default: 1e-7</param>
            /// <returns>Loss tensor with the reduction option applied.</returns>
            public static Tensor distance_box_iou_loss(Tensor boxes1, Tensor boxes2, Reduction reduction = Reduction.None,
                float eps = 1e-7f)
            {
                //Reference:
                //    Zhaohui Zheng et. al: Distance Intersection over Union Loss:
                //    https://arxiv.org/abs/1911.08287
                //Original Implementation from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/losses.py

                boxes1 = _upcast_non_float(boxes1);
                boxes2 = _upcast_non_float(boxes2);

                var (loss, _) = _diou_iou_loss(boxes1, boxes2, eps);

                if (reduction == Reduction.Mean)
                    loss = loss.numel() > 0 ? loss.mean() : 0.0 * loss.sum();
                else if (reduction == Reduction.Sum)
                    loss = loss.sum();
                return loss;
            }

            internal static (Tensor, Tensor) _diou_iou_loss(Tensor boxes1, Tensor boxes2, float eps = 1e-7f)
            {
                var (intsct, union) = _loss_inter_union(boxes1, boxes2);
                var iou = intsct / (union + eps);
                //# smallest enclosing box
                var (x1, y1, x2, y2, _) = boxes1.unbind(dimension: -1);
                var (x1g, y1g, x2g, y2g, _) = boxes2.unbind(dimension: -1);
                var xc1 = torch.min(x1, x1g);
                var yc1 = torch.min(y1, y1g);
                var xc2 = torch.max(x2, x2g);
                var yc2 = torch.max(y2, y2g);
                //# The diagonal distance of the smallest enclosing box squared
                var diagonal_distance_squared = (torch.pow((xc2 - xc1), 2)) + (torch.pow((yc2 - yc1), 2)) + eps;
                //# centers of boxes
                var x_p = (x2 + x1) / 2;
                var y_p = (y2 + y1) / 2;
                var x_g = (x1g + x2g) / 2;
                var y_g = (y1g + y2g) / 2;
                //# The distance between boxes' centers squared.
                var centers_distance_squared = (torch.pow((x_p - x_g), 2)) + (torch.pow((y_p - y_g), 2));
                //# The distance IoU is the IoU penalized by a normalized
                //# distance between boxes' centers squared.
                var loss = 1 - iou + (centers_distance_squared / diagonal_distance_squared);
                return (loss, iou);
            }
        }
    }
}
