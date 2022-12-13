// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/f56e6f63aa1d37e648b0c4cb951ce26292238c53/torchvision/ops/_utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using TorchSharp.Utils;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            /// <summary>
            /// Efficient version of torch.cat that avoids a copy if there is only a single element in a list
            /// </summary>
            /// <param name="tensors"></param>
            /// <param name="dim"></param>
            /// <returns></returns>
            public static Tensor _cat(List<Tensor> tensors, long dim = 0)
            {
                if (tensors.Count == 1)
                    return tensors[0];
                return torch.cat(tensors, dim);
            }

            /// <summary>
            /// Converts list of Tensor to roi format.
            /// </summary>
            /// <param name="boxes"></param>
            /// <returns></returns>
            public static Tensor convert_boxes_to_roi_format(List<Tensor> boxes)
            {
                var concat_boxes = _cat(boxes, dim: 0);
                var temp = new List<Tensor>();
                for (int i = 0; i < boxes.Count; i++)
                    temp.Add(torch.full_like(boxes[i][TensorIndex.Colon, TensorIndex.Slice(stop: 1)], i));

                var ids = _cat(temp, dim: 0);

                var rois = torch.cat(new List<Tensor> { ids, concat_boxes }, dim: 1);
                return rois;
            }

            /// <summary>
            /// Checks if format of boxes is correct.
            /// </summary>
            /// <param name="boxes"></param>
            public static void check_roi_boxes_shape(object boxes)
            {
                if (boxes is List<Tensor> boxesList)
                    foreach (var _tensor in boxesList)
                        Debug.Assert(
                            _tensor.size(1) == 4, "The shape of the tensor in the boxes list is not correct as List[Tensor[L, 4]]"
                        );
                else if (boxes is Tensor tensor)
                    Debug.Assert(tensor.size(1) == 5, "The boxes tensor shape is not correct as Tensor[K, 5]");
                else
                    Debug.Assert(false, "boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]");
            }

            /// <summary>
            /// Splits normalization parameters.
            /// </summary>
            /// <param name="model"></param>
            /// <param name="norm_classes"></param>
            /// <returns>Tuple of normalization parameters and othere parameters.</returns>
            /// <exception cref="ArgumentException"></exception>
            public static (List<Tensor>, List<Tensor>) split_normalization_params(
                nn.Module model, List<Type>? norm_classes = null
            )
            {
                if (norm_classes == null)
                    norm_classes = new List<Type> {
                        //nn.modules.batchnorm._BatchNorm,
                        typeof(LayerNorm),
                        typeof(GroupNorm),
                        //nn.modules.instancenorm._InstanceNorm,
                        typeof(LocalResponseNorm),
                    };

                foreach (var t in norm_classes)
                    if (!(t.IsSubclassOf(typeof(nn.Module))))
                        throw new ArgumentException(string.Format("Class {0} is not a subclass of nn.Module.", t));

                var classes = norm_classes;

                var norm_params = new List<Tensor>();
                var other_params = new List<Tensor>();
                foreach (var named_module in model.named_modules()) {
                    var module = named_module.module;
                    var named_children = module.named_children().GetEnumerator();
                    if (named_children.MoveNext()) {
                        foreach (var p in module.named_parameters(false))
                            if (p.parameter.requires_grad)
                                other_params.Add(p.parameter);

                    } else if (classes.Contains(module.GetType())) {
                        foreach (var p in module.named_parameters(false))
                            if (p.parameter.requires_grad)
                                norm_params.Add(p.parameter);
                    } else {
                        foreach (var p in module.named_parameters(false))
                            if (p.parameter.requires_grad)
                                other_params.Add(p.parameter);
                    }
                }
                return (norm_params, other_params);
            }

            /// <summary>
            /// Protects from numerical overflows in multiplications by upcasting to the equivalent higher type.
            /// </summary>
            public static Tensor _upcast(Tensor t)
            {
                if (t.is_floating_point())
                    return t.dtype == torch.float32 || t.dtype == torch.float64 ? t : t.@float();
                else
                    return t.dtype == torch.int32 || t.dtype == torch.int64 ? t : t.@int();
            }

            /// <summary>
            /// Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
            /// </summary>
            /// <param name="t"></param>
            /// <returns></returns>
            public static Tensor _upcast_non_float(Tensor t)
            {
                if (t.dtype != torch.float32 && t.dtype != torch.float64)
                    return t.@float();
                return t;
            }


            public static (Tensor, Tensor) _loss_inter_union(
                Tensor boxes1,
                Tensor boxes2
            )
            {
                var (x1, y1, x2, y2, _) = boxes1.unbind(dimension: -1);
                var (x1g, y1g, x2g, y2g, __) = boxes2.unbind(dimension: -1);

                //# Intersection keypoints
                var xkis1 = torch.max(x1, x1g);
                var ykis1 = torch.max(y1, y1g);
                var xkis2 = torch.min(x2, x2g);
                var ykis2 = torch.min(y2, y2g);

                var intsctk = torch.zeros_like(x1);
                var mask = (ykis2 > ykis1) & (xkis2 > xkis1);
                intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask]);
                var unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk;

                return (intsctk, unionk);
            }
        }
    }
}
