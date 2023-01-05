// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/_utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Text;
using System.Xml.Linq;
using static TorchSharp.torch;
using TorchSharp.Utils;
using System.Diagnostics;
using System.Buffers.Text;
using System.Drawing;
using System.Reflection;
using System.Text.RegularExpressions;
using TorchSharp.Ops;
using TorchSharp.Modules;
using System.Linq;
using TorchSharp.Modules.Detection;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            public static partial class detection
            {
                /// <summary>
                /// Encode a set of proposals with respect to some reference boxes.
                /// </summary>
                /// <param name="reference_boxes">reference boxes</param>
                /// <param name="proposals">boxes to be encoded</param>
                /// <param name="weights">the weights for ``(x, y, w, h)``</param>
                /// <returns></returns>
                public static Tensor encode_boxes(Tensor reference_boxes, Tensor proposals, Tensor weights)
                {
                    //# perform some unpacking to make it JIT-fusion friendly
                    var wx = weights[0];
                    var wy = weights[1];
                    var ww = weights[2];
                    var wh = weights[3];

                    var proposals_x1 = proposals[TensorIndex.Colon, 0].unsqueeze(1);
                    var proposals_y1 = proposals[TensorIndex.Colon, 1].unsqueeze(1);
                    var proposals_x2 = proposals[TensorIndex.Colon, 2].unsqueeze(1);
                    var proposals_y2 = proposals[TensorIndex.Colon, 3].unsqueeze(1);

                    var reference_boxes_x1 = reference_boxes[TensorIndex.Colon, 0].unsqueeze(1);
                    var reference_boxes_y1 = reference_boxes[TensorIndex.Colon, 1].unsqueeze(1);
                    var reference_boxes_x2 = reference_boxes[TensorIndex.Colon, 2].unsqueeze(1);
                    var reference_boxes_y2 = reference_boxes[TensorIndex.Colon, 3].unsqueeze(1);

                    //# implementation starts here
                    var ex_widths = proposals_x2 - proposals_x1;
                    var ex_heights = proposals_y2 - proposals_y1;
                    var ex_ctr_x = proposals_x1 + 0.5 * ex_widths;
                    var ex_ctr_y = proposals_y1 + 0.5 * ex_heights;

                    var gt_widths = reference_boxes_x2 - reference_boxes_x1;
                    var gt_heights = reference_boxes_y2 - reference_boxes_y1;
                    var gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths;
                    var gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights;

                    var targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths;
                    var targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights;
                    var targets_dw = ww * torch.log(gt_widths / ex_widths);
                    var targets_dh = wh * torch.log(gt_heights / ex_heights);

                    var targets = torch.cat(new List<Tensor> { targets_dx, targets_dy, targets_dw, targets_dh }, dim: 1);
                    return targets;
                }

                /// <summary>
                /// This method overwrites the default eps values of all the
                ///    FrozenBatchNorm2d layers of the model with the provided value.
                ///    This is necessary to address the BC-breaking change introduced
                ///    by the bug-fix at pytorch/vision#2933. The overwrite is applied
                ///    only when the pretrained weights are loaded to maintain compatibility
                ///    with previous versions.
                /// </summary>
                /// <param name="model">The model on which we perform the overwrite.</param>
                /// <param name="eps">The new value of eps.</param>
                public static void overwrite_eps(nn.Module model, float eps)
                {
                    foreach (var module in model.modules())
                        if (module is FrozenBatchNorm2d frozen)
                            frozen.Eps = eps;
                }

                /// <summary>
                /// This method retrieves the number of output channels of a specific model.
                /// </summary>
                /// <param name="model">The model for which we estimate the out_channels.
                ///            It should return a single Tensor or an OrderedDict[Tensor].</param>
                /// <param name="size">The size (wxh) of the input.</param>
                /// <returns>A list of the output channels of the model.</returns>
                public static List<long> retrieve_out_channels(nn.Module<Tensor, Tensor> model, (long, long) size)
                {
                    var in_training = model.training;
                    model.eval();
                    var out_channels = new List<long>();
                    using (torch.no_grad()) {
                        //# Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
                        var device = model.parameters().First().device;
                        var tmp_img = torch.zeros(new long[] { 1, 3, size.Item2, size.Item1 }, device: device);
                        var features = model.forward(tmp_img);
                        var features_dict = new OrderedDict<string, Tensor>();
                        if (features is torch.Tensor)
                            features_dict.Add("0", features);

                        foreach (var x in features_dict.values())
                            out_channels.Add(x.size(1));
                    }

                    if (in_training)
                        model.train();

                    return out_channels;
                }

                /// <summary>
                /// ONNX spec requires the k-value to be less than or equal to the number of inputs along
                ///    provided dim. Certain models use the number of elements along a particular axis instead of K
                ///    if K exceeds the number of elements along that axis. Previously, python's min() function was
                ///    used to determine whether to use the provided k-value or the specified dim axis value.
                ///    However in cases where the model is being exported in tracing mode, python min() is
                ///    static causing the model to be traced incorrectly and eventually fail at the topk node.
                ///    In order to avoid this situation, in tracing mode, torch.min() is used instead.
                /// </summary>
                /// <param name="input">The orignal input tensor.</param>
                /// <param name="orig_kval">The provided k-value.</param>
                /// <param name="axis">Axis along which we retreive the input size.</param>
                /// <returns>Appropriately selected k-value.</returns>
                internal static long _topk_min(Tensor input, long orig_kval, int axis)
                {
                    return Math.Min(orig_kval, input.size(axis));
                }

                internal static Tensor _box_loss(string type, BoxCoder box_coder, Tensor anchors_per_image,
                    Tensor matched_gt_boxes_per_image, Tensor bbox_regression_per_image,
                    Dictionary<string, float> cnf = null)
                {
                    Debug.Assert(new string[] { "l1", "smooth_l1", "ciou", "diou", "giou" }.Contains(type), string.Format("Unsupported loss: {0}", type));

                    if (type == "l1") {
                        var target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image);
                        return functional.l1_loss(bbox_regression_per_image, target_regression, reduction: Reduction.Sum);
                    } else if (type == "smooth_l1") {
                        var target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image);
                        float beta = 1.0f;
                        if (cnf != null && cnf.ContainsKey("beta"))
                            beta = cnf["beta"];
                        return functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction: Reduction.Sum, beta: beta);
                    } else {
                        var bbox_per_image = box_coder.decode_single(bbox_regression_per_image, anchors_per_image);
                        float eps = 1e-7f;
                        if (cnf != null && cnf.ContainsKey("eps"))
                            eps = cnf["eps"];
                        if (type == "ciou")
                            return torchvision.ops.complete_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction: Reduction.Sum, eps: eps);
                        if (type == "diou")
                            return torchvision.ops.distance_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction: Reduction.Sum, eps: eps);
                        //# otherwise giou
                        return torchvision.ops.generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction: Reduction.Sum, eps: eps);
                    }
                }
            }
        }
    }

    namespace Modules.Detection
    {
        /// <summary>
        /// This class samples batches, ensuring that they contain a fixed proportion of positives.
        /// </summary>
        public class BalancedPositiveNegativeSampler
        {
            private long batch_size_per_image;
            private float positive_fraction;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="batch_size_per_image">number of elements to be selected per image</param>
            /// <param name="positive_fraction">percentage of positive elements per batch</param>
            public BalancedPositiveNegativeSampler(long batch_size_per_image, float positive_fraction)
            {
                this.batch_size_per_image = batch_size_per_image;
                this.positive_fraction = positive_fraction;
            }

            /// <summary>
            /// Samples batches.
            /// </summary>
            /// <param name="matched_idxs">list of tensors containing -1, 0 or positive values.
            ///        Each tensor corresponds to a specific image.
            ///        -1 values are ignored, 0 are considered as negatives and > 0 as
            ///        positives.</param>
            /// <returns>Returns two lists of binary masks for each image.
            /// The first list contains the positive elements that were selected,
            /// and the second list the negative example.</returns>
            public (List<Tensor>, List<Tensor>) __call__(List<Tensor> matched_idxs)
            {
                var pos_idx = new List<Tensor>();
                var neg_idx = new List<Tensor>();
                foreach (var matched_idxs_per_image in matched_idxs) {
                    var positive = torch.nonzero(matched_idxs_per_image >= 1)[0];
                    var negative = torch.nonzero(matched_idxs_per_image == 0)[0];

                    var num_pos = (long)(this.batch_size_per_image * this.positive_fraction);
                    //# protect against not enough positive examples
                    num_pos = Math.Min(positive.numel(), num_pos);
                    var num_neg = this.batch_size_per_image - num_pos;
                    //# protect against not enough negative examples
                    num_neg = Math.Min(negative.numel(), num_neg);

                    //# randomly select positive and negative examples
                    var perm1 = torch.randperm(positive.numel(), device: positive.device)[TensorIndex.Slice(stop: num_pos)];
                    var perm2 = torch.randperm(negative.numel(), device: negative.device)[TensorIndex.Slice(stop: num_neg)];

                    var pos_idx_per_image = positive[perm1];
                    var neg_idx_per_image = negative[perm2];
                    //# create binary mask from indices
                    var pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype: torch.uint8);
                    var neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype: torch.uint8);
                    pos_idx_per_image_mask[pos_idx_per_image] = 1;
                    neg_idx_per_image_mask[neg_idx_per_image] = 1;

                    pos_idx.Add(pos_idx_per_image_mask);
                    neg_idx.Add(neg_idx_per_image_mask);
                }
                return (pos_idx, neg_idx);
            }
        }

        /// <summary>
        /// This class encodes and decodes a set of bounding boxes into
        /// the representation used for training the regressors.
        /// </summary>
        public class BoxCoder
        {
            private float[] weights;
            private float bbox_xform_clip;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="weights">(4-element tuple)</param>
            /// <param name="bbox_xform_clip">bbox_xform_clip</param>
            public BoxCoder(
                 float[] weights, float? bbox_xform_clip = null)
            {
                if (bbox_xform_clip == null)
                    bbox_xform_clip = (float)Math.Log(1000.0 / 16);
                this.weights = weights;
                this.bbox_xform_clip = bbox_xform_clip.Value;
            }

            public List<Tensor> encode(List<Tensor> reference_boxes, List<Tensor> proposals)
            {
                var boxes_per_image = new List<long>();
                foreach (var b in reference_boxes)
                    boxes_per_image.Add(b.shape[0]);
                var reference_boxes_t = torch.cat(reference_boxes, dim: 0);
                var proposals_t = torch.cat(proposals, dim: 0);

                var targets = this.encode_single(reference_boxes_t, proposals_t);
                return new List<Tensor>(targets.split(boxes_per_image.ToArray(), 0));
            }

            /// <summary>
            /// Encode a set of proposals with respect to some reference boxes.
            /// </summary>
            /// <param name="reference_boxes">reference boxes</param>
            /// <param name="proposals">boxes to be encoded</param>
            /// <returns></returns>
            public Tensor encode_single(Tensor reference_boxes, Tensor proposals)
            {
                var dtype = reference_boxes.dtype;
                var device = reference_boxes.device;
                var weights = torch.as_tensor(this.weights, dtype: dtype, device: device);
                var targets = torchvision.models.detection.encode_boxes(reference_boxes, proposals, weights);

                return targets;
            }

            public Tensor decode(Tensor rel_codes, List<Tensor> boxes)
            {
                var boxes_per_image = new List<long>();
                foreach (var b in boxes)
                    boxes_per_image.Add(b.size(0));

                var concat_boxes = torch.cat(boxes, dim: 0);

                var box_sum = 0L;
                foreach (var val in boxes_per_image)
                    box_sum += val;
                if (box_sum > 0)
                    rel_codes = rel_codes.reshape(box_sum, -1);
                var pred_boxes = this.decode_single(rel_codes, concat_boxes);
                if (box_sum > 0)
                    pred_boxes = pred_boxes.reshape(box_sum, -1, 4);
                return pred_boxes;
            }

            /// <summary>
            /// From a set of original boxes and encoded relative box offsets, get the decoded boxes.
            /// </summary>
            /// <param name="rel_codes">encoded boxes</param>
            /// <param name="boxes">reference boxes</param>
            /// <returns></returns>
            public Tensor decode_single(Tensor rel_codes, Tensor boxes)
            {
                boxes = boxes.to(rel_codes.dtype);

                var widths = boxes[TensorIndex.Colon, 2] - boxes[TensorIndex.Colon, 0];
                var heights = boxes[TensorIndex.Colon, 3] - boxes[TensorIndex.Colon, 1];
                var ctr_x = boxes[TensorIndex.Colon, 0] + 0.5 * widths;
                var ctr_y = boxes[TensorIndex.Colon, 1] + 0.5 * heights;

                var (wx, wy, ww, wh, _) = this.weights;
                var dx = rel_codes[TensorIndex.Colon, TensorIndex.Slice(0, null, 4)] / wx;
                var dy = rel_codes[TensorIndex.Colon, TensorIndex.Slice(1, null, 4)] / wy;
                var dw = rel_codes[TensorIndex.Colon, TensorIndex.Slice(2, null, 4)] / ww;
                var dh = rel_codes[TensorIndex.Colon, TensorIndex.Slice(3, null, 4)] / wh;

                //# Prevent sending too large values into torch.exp()
                dw = torch.clamp(dw, max: this.bbox_xform_clip);
                dh = torch.clamp(dh, max: this.bbox_xform_clip);

                var pred_ctr_x = dx * widths[TensorIndex.Colon, TensorIndex.None] + ctr_x[TensorIndex.Colon, TensorIndex.None];
                var pred_ctr_y = dy * heights[TensorIndex.Colon, TensorIndex.None] + ctr_y[TensorIndex.Colon, TensorIndex.None];
                var pred_w = torch.exp(dw) * widths[TensorIndex.Colon, TensorIndex.None];
                var pred_h = torch.exp(dh) * heights[TensorIndex.Colon, TensorIndex.None];

                //# Distance from center to box's corner.
                var c_to_c_h = torch.tensor(0.5, dtype: pred_ctr_y.dtype, device: pred_h.device) * pred_h;
                var c_to_c_w = torch.tensor(0.5, dtype: pred_ctr_x.dtype, device: pred_w.device) * pred_w;

                var pred_boxes1 = pred_ctr_x - c_to_c_w;
                var pred_boxes2 = pred_ctr_y - c_to_c_h;
                var pred_boxes3 = pred_ctr_x + c_to_c_w;
                var pred_boxes4 = pred_ctr_y + c_to_c_h;

                var pred_boxes = torch.stack(new Tensor[] { pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4 }, dim: 2).flatten(1);
                return pred_boxes;
            }
        }

        /// <summary>
        /// The linear box-to-box transform defined in FCOS. The transformation is parameterized
        /// by the distance from the center of (square) src box to 4 edges of the target box.
        /// </summary>
        public class BoxLinearCoder
        {
            private bool normalize_by_size;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="normalize_by_size">normalize deltas by the size of src (anchor) boxes.</param>
            public BoxLinearCoder(bool normalize_by_size = true)
            {
                this.normalize_by_size = normalize_by_size;
            }

            /// <summary>
            /// Encode a set of proposals with respect to some reference boxes.
            /// </summary>
            /// <param name="reference_boxes">reference boxes</param>
            /// <param name="proposals">boxes to be encoded</param>
            /// <returns>the encoded relative box offsets that can be used to decode the boxes.</returns>
            public Tensor encode_single(Tensor reference_boxes, Tensor proposals)
            {
                //# get the center of reference_boxes
                var reference_boxes_ctr_x = 0.5 * (reference_boxes[TensorIndex.Colon, 0] + reference_boxes[TensorIndex.Colon, 2]);
                var reference_boxes_ctr_y = 0.5 * (reference_boxes[TensorIndex.Colon, 1] + reference_boxes[TensorIndex.Colon, 3]);

                //# get box regression transformation deltas
                var target_l = reference_boxes_ctr_x - proposals[TensorIndex.Colon, 0];
                var target_t = reference_boxes_ctr_y - proposals[TensorIndex.Colon, 1];
                var target_r = proposals[TensorIndex.Colon, 2] - reference_boxes_ctr_x;
                var target_b = proposals[TensorIndex.Colon, 3] - reference_boxes_ctr_y;

                var targets = torch.stack(new Tensor[] { target_l, target_t, target_r, target_b }, dim: 1);
                if (this.normalize_by_size) {
                    var reference_boxes_w = reference_boxes[TensorIndex.Colon, 2] - reference_boxes[TensorIndex.Colon, 0];
                    var reference_boxes_h = reference_boxes[TensorIndex.Colon, 3] - reference_boxes[TensorIndex.Colon, 1];
                    var reference_boxes_size = torch.stack(
                         new Tensor[] { reference_boxes_w, reference_boxes_h, reference_boxes_w, reference_boxes_h }, dim: 1
                     );
                    targets = targets / reference_boxes_size;
                }

                return targets;
            }

            /// <summary>
            /// From a set of original boxes and encoded relative box offsets, get the decoded boxes.
            /// </summary>
            /// <param name="rel_codes">encoded boxes</param>
            /// <param name="boxes">reference boxes.</param>
            /// <returns>the predicted boxes with the encoded relative box offsets.</returns>
            public Tensor decode_single(Tensor rel_codes, Tensor boxes)
            {
                boxes = boxes.to(rel_codes.dtype);

                var ctr_x = 0.5 * (boxes[TensorIndex.Colon, 0] + boxes[TensorIndex.Colon, 2]);
                var ctr_y = 0.5 * (boxes[TensorIndex.Colon, 1] + boxes[TensorIndex.Colon, 3]);
                if (this.normalize_by_size) {
                    var boxes_w = boxes[TensorIndex.Colon, 2] - boxes[TensorIndex.Colon, 0];
                    var boxes_h = boxes[TensorIndex.Colon, 3] - boxes[TensorIndex.Colon, 1];
                    var boxes_size = torch.stack(new Tensor[] { boxes_w, boxes_h, boxes_w, boxes_h }, dim: -1);
                    rel_codes = rel_codes * boxes_size;
                }

                var pred_boxes1 = ctr_x - rel_codes[TensorIndex.Colon, 0];
                var pred_boxes2 = ctr_y - rel_codes[TensorIndex.Colon, 1];
                var pred_boxes3 = ctr_x + rel_codes[TensorIndex.Colon, 2];
                var pred_boxes4 = ctr_y + rel_codes[TensorIndex.Colon, 3];
                var pred_boxes = torch.stack(new Tensor[] { pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4 }, dim: -1);
                return pred_boxes;
            }
        }

        /// <summary>
        /// This class assigns to each predicted "element" (e.g., a box) a ground-truth
        /// element.Each predicted element will have exactly zero or one matches; each
        /// ground-truth element may be assigned to zero or more predicted elements.
        /// 
        /// Matching is based on the MxN match_quality_matrix, that characterizes how well
        /// each(ground-truth, predicted)-pair match.For example, if the elements are
        /// boxes, the matrix may contain box IoU overlap values.
        /// 
        /// The matcher returns a tensor of size N containing the index of the ground-truth
        /// element m that matches to prediction n. If there is no match, a negative value
        /// is returned.
        /// </summary>
        public class Matcher
        {
            public int BELOW_LOW_THRESHOLD = -1;
            public int BETWEEN_THRESHOLDS = -2;

            private float high_threshold;
            private float low_threshold;
            private bool allow_low_quality_matches;

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="high_threshold">quality values greater than or equal to this value are candidate matches.</param>
            /// <param name="low_threshold">a lower quality threshold used to stratify
            ///         matches into three levels:
            ///         1) matches >= high_threshold
            ///         2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
            ///         3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)</param>
            /// <param name="allow_low_quality_matches">if True, produce additional matches
            ///         for predictions that have only low-quality match candidates. See
            ///         set_low_quality_matches_ for more details.</param>
            public Matcher(float high_threshold, float low_threshold, bool allow_low_quality_matches = false)
            {
                this.BELOW_LOW_THRESHOLD = -1;
                this.BETWEEN_THRESHOLDS = -2;
                Debug.Assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold");
                this.high_threshold = high_threshold;
                this.low_threshold = low_threshold;
                this.allow_low_quality_matches = allow_low_quality_matches;
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="match_quality_matrix">an MxN tensor, containing the
            /// pairwise quality between M ground-truth elements and N predicted elements.</param>
            /// <returns>an N tensor where N[i] is a matched gt in
            ///    [0, M - 1] or a negative value indicating that prediction i could not
            ///    be matched.</returns>
            /// <exception cref="ArgumentException"></exception>
            public virtual Tensor __call__(Tensor match_quality_matrix)
            {
                if (match_quality_matrix.numel() == 0) {
                    //# empty targets or proposals not supported during training
                    if (match_quality_matrix.shape[0] == 0)
                        throw new ArgumentException("No ground-truth boxes available for one of the images during training");
                    else
                        throw new ArgumentException("No proposal boxes available for one of the images during training");
                }

                //# match_quality_matrix is M (gt) x N (predicted)
                //# Max over gt elements (dim 0) to find best gt candidate for each prediction
                var (matched_vals, matches) = match_quality_matrix.max(0);
                Tensor all_matches = null;
                if (this.allow_low_quality_matches)
                    all_matches = matches.clone();
                else
                    all_matches = null;  //# type: ignore[assignment]

                //# Assign candidate matches with low quality to negative (unassigned) values
                var below_low_threshold = matched_vals < this.low_threshold;
                var between_thresholds = (matched_vals >= this.low_threshold) & (matched_vals < this.high_threshold);
                matches[below_low_threshold] = this.BELOW_LOW_THRESHOLD;
                matches[between_thresholds] = this.BETWEEN_THRESHOLDS;

                if (this.allow_low_quality_matches) {
                    if ((object)all_matches == null)
                        Debug.Assert(false, "all_matches should not be None");
                    else
                        this.set_low_quality_matches_(matches, all_matches, match_quality_matrix);
                }

                return matches;
            }

            /// <summary>
            /// Produce additional matches for predictions that have only low-quality matches.
            /// Specifically, for each ground-truth find the set of predictions that have
            /// maximum overlap with it (including ties); for each prediction in that set, if
            /// it is unmatched, then match it to the ground-truth with which it has the highest
            /// quality value.
            /// </summary>
            public void set_low_quality_matches_(Tensor matches, Tensor all_matches, Tensor match_quality_matrix)
            {
                //# For each gt, find the prediction with which it has highest quality
                var (highest_quality_foreach_gt, _) = match_quality_matrix.max(1);
                //# Find highest quality match available, even if it is low, including ties
                var gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[TensorIndex.Colon, TensorIndex.None]);
                //# Example gt_pred_pairs_of_highest_quality:
                //#   tensor([[    0, 39796],
                //#           [    1, 32055],
                //#           [    1, 32070],
                //#           [    2, 39190],
                //#           [    2, 40255],
                //#           [    3, 40390],
                //#           [    3, 41455],
                //#           [    4, 45470],
                //#           [    5, 45325],
                //#           [    5, 46390]])
                //# Each row is a (gt index, prediction index)
                //# Note how gt items 1, 2, 3, and 5 each have two ties

                var pred_inds_to_update = gt_pred_pairs_of_highest_quality[1];
                matches[pred_inds_to_update] = all_matches[pred_inds_to_update];
            }
        }

        class SSDMatcher : Matcher
        {
            public SSDMatcher(float threshold) :
                base(threshold, threshold, allow_low_quality_matches: false)
            {

            }

            public override Tensor __call__(Tensor match_quality_matrix)
            {
                var matches = base.__call__(match_quality_matrix);

                //# For each gt, find the prediction with which it has the highest quality
                var (_, highest_quality_pred_foreach_gt) = match_quality_matrix.max(1);
                matches[highest_quality_pred_foreach_gt] = torch.arange(
                    highest_quality_pred_foreach_gt.size(0), dtype: torch.int64, device: highest_quality_pred_foreach_gt.device
                );

                return matches;
            }
        }
    }
}
