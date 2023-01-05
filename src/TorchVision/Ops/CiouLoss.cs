// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/f56e6f63aa1d37e648b0c4cb951ce26292238c53/torchvision/ops/ciou_loss.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System.Drawing;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchvision;
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
            /// boxes do not overlap.This loss function considers important geometrical
            /// factors such as overlap area, normalized central point distance and aspect ratio.
            /// This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.
            /// Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
            /// ``0 &lt;= x1&lt;x2`` and ``0 &lt;= y1&lt;y2``, and The two boxes should have the
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
            public static Tensor complete_box_iou_loss(Tensor boxes1, Tensor boxes2, Reduction reduction = Reduction.None,
                float eps = 1e-7f)
            {
                //   Zhaohui Zheng et al.: Complete Intersection over Union Loss:
                //   https://arxiv.org/abs/1911.08287
                //   Original Implementation from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/losses.py

                boxes1 = _upcast_non_float(boxes1);
                boxes2 = _upcast_non_float(boxes2);

                var (diou_loss, iou) = _diou_iou_loss(boxes1, boxes2);

                var (x1, y1, x2, y2, _) = boxes1.unbind(dimension: -1);
                var (x1g, y1g, x2g, y2g, _) = boxes2.unbind(dimension: -1);

                //# width and height of boxes
                var w_pred = x2 - x1;
                var h_pred = y2 - y1;
                var w_gt = x2g - x1g;
                var h_gt = y2g - y1g;
                var v = (4 / (torch.pow(Math.PI, 2))) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2);
                Tensor alpha;
                using (torch.no_grad())
                   alpha = v / (1 - iou + v + eps);

               var loss = diou_loss + alpha * v;
                if (reduction == Reduction.Mean)
                    loss = loss.numel() > 0 ? loss.mean() : 0.0 * loss.sum();
                else if (reduction == Reduction.Sum)
                    loss = loss.sum();

                return loss;
            }
        }
    }
}
