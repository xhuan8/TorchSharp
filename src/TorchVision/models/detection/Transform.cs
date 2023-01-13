// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/transform.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System.Collections.Generic;
using System.Diagnostics;
using System;
using static TorchSharp.torch;
using TorchSharp.Utils;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            public static partial class detection
            {
                internal static (Tensor, Dictionary<string, Tensor>) _resize_image_and_masks(
                    Tensor image,
                    float self_min_size,
                    float self_max_size,
                    Dictionary<string, Tensor> target = null,
                    long[] fixed_size = null)
                {
                    var im_shape = torch.tensor(new long[] { image.shape[image.shape.Length - 2],
                        image.shape[image.shape.Length - 1] });

                    List<long> size = null;
                    float? scale_factor = null;
                    bool? recompute_scale_factor = null;
                    if (fixed_size != null)
                        size = new List<long> { fixed_size[1], fixed_size[0] };
                    else {
                        var min_size = torch.min(im_shape).to(torch.ScalarType.Float32);
                        var max_size = torch.max(im_shape).to(torch.ScalarType.Float32);
                        var scale = torch.minimum(self_min_size / min_size, self_max_size / max_size);

                        scale_factor = scale.item<float>();
                        recompute_scale_factor = true;
                    }

                    var dim = image.dim() - 1;
                    var scale_factor_list = new List<double>();
                    if (scale_factor.HasValue)
                        for (int i = 0; i < dim; i++)
                            scale_factor_list.Add(scale_factor.Value);
                    image = torch.nn.functional.interpolate(
                        image[torch.TensorIndex.None],
                        size: size != null ? size.ToArray() : null,
                        scale_factor: scale_factor.HasValue ? scale_factor_list.ToArray() : null,
                        mode: InterpolationMode.Bilinear,
                        recompute_scale_factor: recompute_scale_factor == true,
                        align_corners: false
                    )[0];

                    if (target == null)
                        return (image, target);

                    if (target.ContainsKey("masks")) {
                        var mask = target["masks"];
                        mask = torch.nn.functional.interpolate(
                                mask[torch.TensorIndex.Colon, torch.TensorIndex.None].@float(),
                                size: size != null ? size.ToArray() : null,
                                scale_factor: scale_factor.HasValue ? new double[] { scale_factor.Value } : null,
                                recompute_scale_factor: recompute_scale_factor == true
                            )[torch.TensorIndex.Colon, 0].@byte();
                        target["masks"] = mask;
                    }
                    return (image, target);
                }

                internal static Tensor resize_keypoints(Tensor keypoints, List<long> original_size, List<long> new_size)
                {
                    var ratios = new List<Tensor>();
                    for (int i = 0; i < new_size.Count; i++) {
                        var s = new_size[i];
                        var s_orig = original_size[i];
                        ratios.Add(torch.tensor(s, dtype: torch.float32, device: keypoints.device)
                                / torch.tensor(s_orig, dtype: torch.float32, device: keypoints.device));
                    }

                    var ratio_h = ratios[0];
                    var ratio_w = ratios[1];
                    var resized_data = keypoints.clone();
                    resized_data[torch.TensorIndex.Ellipsis, 0] *= ratio_w;
                    resized_data[torch.TensorIndex.Ellipsis, 1] *= ratio_h;
                    return resized_data;
                }

                internal static Tensor resize_boxes(Tensor boxes, List<long> original_size, List<long> new_size)
                {
                    var ratios = new List<Tensor>();
                    for (int i = 0; i < new_size.Count; i++) {
                        var s = new_size[i];
                        var s_orig = original_size[i];
                        ratios.Add(torch.tensor(s, dtype: torch.float32, device: boxes.device)
                                / torch.tensor(s_orig, dtype: torch.float32, device: boxes.device));
                    }

                    var ratio_height = ratios[0];
                    var ratio_width = ratios[1];
                    var (xmin, ymin, xmax, ymax, _) = boxes.unbind(1);

                    xmin = xmin * ratio_width;
                    xmax = xmax * ratio_width;
                    ymin = ymin * ratio_height;
                    ymax = ymax * ratio_height;
                    return torch.stack(new Tensor[] { xmin, ymin, xmax, ymax }, 1);
                }
            }
        }
    }

    namespace Modules.Detection
    {
        /// <summary>
        /// Performs input / target transformation before feeding the data to a GeneralizedRCNN
        /// model.
        /// The transformations it perform are:
        ///    - input normalization (mean subtraction and std division)
        ///    - input / target resizing to match min_size / max_size
        /// It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
        /// </summary>
        public class GeneralizedRCNNTransform : nn.Module<List<Tensor>, List<Dictionary<string, Tensor>>,
            (ImageList, List<Dictionary<string, Tensor>>)>
        {
            private bool _skip_resize;
            private List<long> min_size;
            private long max_size;
            private float[] image_mean;
            private float[] image_std;
            private long size_divisible;
            private long[] fixed_size;

            public GeneralizedRCNNTransform(
                string name,
                long min_size,
                long max_size,
                float[] image_mean,
                float[] image_std,
                long size_divisible = 32,
                long[] fixed_size = null,
                Dictionary<string, object> kwargs = null
            ) : base(name)
            {
                if (kwargs != null) {
                    foreach (var key in kwargs.Keys) {
                        switch (key) {
                        case "min_size":
                            min_size = (long)kwargs[key];
                            break;
                        case "max_size":
                            max_size = (long)kwargs[key];
                            break;
                        case "image_mean":
                            image_mean = (float[])kwargs[key];
                            break;
                        case "image_std":
                            image_std = (float[])kwargs[key];
                            break;
                        case "size_divisible":
                            size_divisible = (long)kwargs[key];
                            break;
                        case "fixed_size":
                            fixed_size = (long[])kwargs[key];
                            break;
                        case "_skip_resize":
                            _skip_resize = (bool)kwargs[key];
                            break;
                        }
                    }
                }
                this.min_size = new List<long> { min_size };
                this.max_size = max_size;
                this.image_mean = image_mean;
                this.image_std = image_std;
                this.size_divisible = size_divisible;
                this.fixed_size = fixed_size;
            }

            public override (ImageList, List<Dictionary<string, Tensor>>) forward(
               List<Tensor> images, List<Dictionary<string, Tensor>> targets = null)
            {
                images = new List<Tensor>(images);
                if (targets != null) {
                    // make a copy of targets to avoid modifying it in-place
                    // once torchscript supports dict comprehension
                    // this can be simplified as follows
                    // targets = [{k: v for k,v in t.items()} for t in targets]
                    var targets_copy = new List<Dictionary<string, Tensor>>();
                    foreach (var t in targets) {
                        var data = new Dictionary<string, Tensor>();
                        foreach (var kv in t)
                            data[kv.Key] = kv.Value;
                        targets_copy.Add(data);
                    }

                    targets = targets_copy;
                }
                for (int i = 0; i < images.Count; i++) {
                    var image = images[i];
                    var target_index = targets != null ? targets[i] : null;

                    if (image.dim() != 3)
                        throw new ArgumentException(string.Format("images is expected to be a list of 3d tensors of shape [C, H, W], got {0}", image.shape));
                    image = this.normalize(image);
                    (image, target_index) = this.resize(image, target_index);
                    images[i] = image;
                    if (targets != null && target_index != null)
                        targets[i] = target_index;
                }
                var image_sizes = new List<List<long>>();
                foreach (var img in images)
                    image_sizes.Add(new List<long> { img.shape[img.shape.Length - 2], img.shape[img.shape.Length - 1] });

                var imagesBatch = this.batch_images(images, size_divisible: this.size_divisible);

                List<long[]> image_sizes_list = new List<long[]>();
                foreach (var image_size in image_sizes) {
                    Debug.Assert(
                        image_size.Count == 2,
                        string.Format("Input tensors expected to have in the last two elements H and W, instead got {0}", image_size.Count));
                    image_sizes_list.Add(new long[] { image_size[0], image_size[1] });
                }

                var image_list = new ImageList(imagesBatch, image_sizes_list);
                return (image_list, targets);
            }

            public Tensor normalize(Tensor image)
            {
                if (!image.is_floating_point())
                    throw new ArgumentException("Expected input images to be of floating type (in range [0, 1])");
                var (dtype, device) = (image.dtype, image.device);
                var mean = torch.as_tensor(this.image_mean, dtype: dtype, device: device);
                var std = torch.as_tensor(this.image_std, dtype: dtype, device: device);
                return (image - mean[torch.TensorIndex.Colon, torch.TensorIndex.None, torch.TensorIndex.None]) /
                    std[torch.TensorIndex.Colon, torch.TensorIndex.None, torch.TensorIndex.None];
            }

            /// <summary>
            /// Implements `random.choice` via torch ops so it can be compiled with
            /// TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
            /// is fixed.
            /// </summary>
            /// <param name="k"></param>
            /// <returns></returns>
            public long torch_choice(List<long> k)
            {
                var index = (int)(torch.empty(1).uniform_(0.0, (float)(k.Count)).item<float>());
                return k[index];
            }

            public (Tensor, Dictionary<string, Tensor>) resize(
                Tensor image,
                Dictionary<string, Tensor> target = null)
            {
                var h = image.shape[image.shape.Length - 2];
                var w = image.shape[image.shape.Length - 1];
                float size;
                if (this.training) {
                    if (this._skip_resize)
                        return (image, target);
                    size = (float)(this.torch_choice(this.min_size));
                } else {
                    //# FIXME assume for now that testing uses the largest scale
                    size = (float)(this.min_size[this.min_size.Count - 1]);
                }
                (image, target) = torchvision.models.detection._resize_image_and_masks(image, size, (float)(this.max_size), target, this.fixed_size);

                if (target == null)
                    return (image, target);

                var bbox = target["boxes"];
                bbox = torchvision.models.detection.resize_boxes(bbox, new List<long> { h, w },
                    new List<long> { image.shape[image.shape.Length - 2], image.shape[image.shape.Length - 1] });
                target["boxes"] = bbox;

                if (target.ContainsKey("keypoints")) {
                    var keypoints = target["keypoints"];
                    keypoints = torchvision.models.detection.resize_keypoints(keypoints, new List<long> { h, w },
                        new List<long> { image.shape[image.shape.Length - 2], image.shape[image.shape.Length - 1] });
                    target["keypoints"] = keypoints;
                }
                return (image, target);
            }

            public List<long> max_by_axis(List<List<long>> the_list)
            {
                var maxes = the_list[0];
                for (int i = 1; i < the_list.Count; i++) {
                    var sublist = the_list[i];
                    for (int j = 0; j < sublist.Count; j++) {
                        var item = sublist[j];
                        maxes[j] = Math.Max(maxes[j], item);
                    }
                }
                return maxes;
            }

            public Tensor batch_images(List<Tensor> images, long size_divisible = 32)
            {
                List<List<long>> shapes = new List<List<long>>();
                foreach (var img in images)
                    shapes.Add(new List<long>(img.shape));
                var max_size = this.max_by_axis(shapes);
                var stride = (float)(size_divisible);
                max_size[1] = (int)(Math.Ceiling((float)(max_size[1]) / stride) * stride);
                max_size[2] = (int)(Math.Ceiling((float)(max_size[2]) / stride) * stride);

                var batch_shape = new List<long>();
                batch_shape.Add(images.Count);
                batch_shape.AddRange(max_size);
                var batched_imgs = images[0].new_full(batch_shape.ToArray(), 0);
                for (int i = 0; i < batched_imgs.shape[0]; i++) {
                    var img = images[i];
                    batched_imgs[i, TensorIndex.Slice(stop: img.shape[0]), TensorIndex.Slice(stop: img.shape[1]),
                        TensorIndex.Slice(stop: img.shape[2])].copy_(img);
                }

                return batched_imgs;
            }

            public List<Dictionary<string, Tensor>> postprocess(
                List<Dictionary<string, Tensor>> result,
                List<long[]> image_shapes,
                List<long[]> original_image_sizes
            )
            {
                if (this.training)
                    return result;
                for (int i = 0; i < image_shapes.Count; i++) {
                    var pred = result[i];
                    var im_s = new List<long>(image_shapes[i]);
                    var o_im_s = new List<long>(original_image_sizes[i]);

                    var boxes = pred["boxes"];
                    boxes = torchvision.models.detection.resize_boxes(boxes, im_s, o_im_s);
                    result[i]["boxes"] = boxes;
                    if (pred.ContainsKey("masks")) {
                        var masks = pred["masks"];
                        masks = torchvision.models.detection.paste_masks_in_image(masks, boxes, o_im_s.ToArray());
                        result[i]["masks"] = masks;
                    }
                    if (pred.ContainsKey("keypoints")) {
                        var keypoints = pred["keypoints"];
                        keypoints = torchvision.models.detection.resize_keypoints(keypoints, im_s, o_im_s);
                        result[i]["keypoints"] = keypoints;
                    }
                }

                return result;
            }

            public override string ToString()
            {
                string format_string = "{this.__class__.__name__}(";
                string _indent = "\n    ";
                format_string += "{0}Normalize(mean={1}, std={2})";
                format_string += "{0}Resize(min_size={3}, max_size={4}, mode='bilinear')";
                format_string += "\n)";

                string.Format(format_string, _indent, this.image_mean, this.image_std, this.min_size, this.max_size);
                return format_string;
            }
        }
    }
}
