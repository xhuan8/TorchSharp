// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/roi_heads.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System.Collections.Generic;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Utils;
using System.Linq;
using System;
using System.Security;
using TorchSharp.TorchVision.Ops;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            public static partial class detection
            {
                /// <summary>
                /// Computes the loss for Faster R-CNN.
                /// </summary>
                public static (Tensor, Tensor) fastrcnn_loss(Tensor class_logits, Tensor box_regression, List<Tensor> labels, List<Tensor> regression_targets)
                {
                    var labels_t = torch.cat(labels, 0);
                    var regression_targets_t = torch.cat(regression_targets, 0);

                    var classification_loss = functional.cross_entropy(class_logits, labels_t);

                    // get indices that correspond to the regression targets for
                    // the corresponding ground truth labels, to be used with
                    // advanced indexing
                    var sampled_pos_inds_subset = torch.where(labels_t > 0)[0];
                    var labels_pos = labels_t[sampled_pos_inds_subset];
                    var (N, num_classes, _) = class_logits.shape;
                    box_regression = box_regression.reshape(N, box_regression.size(-1) / 4, 4);

                    var box_loss = functional.smooth_l1_loss(
                        box_regression[sampled_pos_inds_subset, labels_pos],
                        regression_targets[(int)sampled_pos_inds_subset],
                        Reduction.Sum,
                        1.0 / 9
                    );
                    box_loss = box_loss / labels_t.numel();

                    return (classification_loss, box_loss);
                }

                /// <summary>
                /// From the results of the CNN, post process the masks
                /// by taking the mask corresponding to the class with max
                /// probability (which are of fixed size and directly output
                /// by the CNN) and return the masks in the mask field of the BoxList.
                /// </summary>
                /// <param name="x">the mask logits</param>
                /// <param name="labels">bounding boxes that are used as
                ///        reference, one for ech image</param>
                /// <returns>one BoxList for each image, containing the extra field mask</returns>
                public static IList<Tensor> maskrcnn_inference(Tensor x, List<Tensor> labels)
                {
                    var mask_prob = x.sigmoid();

                    // select masks corresponding to the predicted classes
                    var num_masks = x.shape[0];
                    var boxes_per_image = labels.Select(label => label.shape[0]);
                    var labels_t = torch.cat(labels);
                    var index = torch.arange(num_masks, device: labels_t.device);

                    mask_prob = mask_prob[index, labels_t][TensorIndex.Colon, TensorIndex.None];
                    var mask_prob_t = mask_prob.split(boxes_per_image.ToArray(), dim: 0);

                    return mask_prob_t;
                }

                /// <summary>
                /// Given segmentation masks and the bounding boxes corresponding
                ///     to the location of the masks in the image, this function
                ///     crops and resizes the masks in the position defined by the
                ///     boxes. This prepares the masks for them to be fed to the
                ///     loss computation as the targets.
                /// </summary>
                public static Tensor project_masks_on_boxes(Tensor gt_masks, Tensor boxes, Tensor matched_idxs, long M)
                {
                    matched_idxs = matched_idxs.to(boxes);
                    List<Tensor> input = new List<Tensor>();
                    input.Add(matched_idxs[TensorIndex.Colon, TensorIndex.None]);
                    input.Add(boxes);
                    var rois = torch.cat(input, dim: 1);
                    gt_masks = gt_masks[TensorIndex.Colon, TensorIndex.None].to(rois);
                    return torchvision.ops.roi_align(gt_masks, rois, (M, M), 1.0f)[TensorIndex.Colon, 0];
                }

                public static Tensor maskrcnn_loss(Tensor mask_logits, List<Tensor> proposals, List<Tensor> gt_masks, List<Tensor> gt_labels, List<Tensor> mask_matched_idxs)
                {
                    var discretization_size = mask_logits.shape[mask_logits.shape.Length - 1];
                    var labels = new List<Tensor>();
                    for (int i = 0; i < gt_labels.Count; i++)
                        labels.Add(gt_labels[i][mask_matched_idxs[i]]);
                    var mask_targets = new List<Tensor>();
                    for (int i = 0; i < gt_masks.Count; i++)
                        mask_targets.Add(project_masks_on_boxes(gt_masks[i], proposals[i], mask_matched_idxs[i], discretization_size));

                    var labels_t = torch.cat(labels, 0);
                    var mask_targets_t = torch.cat(mask_targets, 0);

                    // torch.mean (in binary_cross_entropy_with_logits) doesn't
                    // accept empty tensors, so handle it separately
                    if (mask_targets_t.numel() == 0)
                        return mask_logits.sum() * 0;

                    var mask_loss = functional.binary_cross_entropy_with_logits(
                        mask_logits[torch.arange(labels_t.shape[0], device: labels_t.device), labels_t], mask_targets_t
                    );
                    return mask_loss;
                }

                public static (Tensor, Tensor) keypoints_to_heatmap(Tensor keypoints, Tensor rois, long heatmap_size)
                {
                    var offset_x = rois[TensorIndex.Colon, 0];
                    var offset_y = rois[TensorIndex.Colon, 1];
                    var scale_x = heatmap_size / (rois[TensorIndex.Colon, 2] - rois[TensorIndex.Colon, 0]);
                    var scale_y = heatmap_size / (rois[TensorIndex.Colon, 3] - rois[TensorIndex.Colon, 1]);

                    offset_x = offset_x[TensorIndex.Colon, TensorIndex.None];
                    offset_y = offset_y[TensorIndex.Colon, TensorIndex.None];
                    scale_x = scale_x[TensorIndex.Colon, TensorIndex.None];
                    scale_y = scale_y[TensorIndex.Colon, TensorIndex.None];

                    var x = keypoints[TensorIndex.Ellipsis, 0];
                    var y = keypoints[TensorIndex.Ellipsis, 1];

                    var x_boundary_inds = x == rois[TensorIndex.Colon, 2][TensorIndex.Colon, TensorIndex.None];
                    var y_boundary_inds = y == rois[TensorIndex.Colon, 3][TensorIndex.Colon, TensorIndex.None];

                    x = (x - offset_x) * scale_x;
                    x = x.floor().@long();
                    y = (y - offset_y) * scale_y;
                    y = y.floor().@long();

                    x[x_boundary_inds] = heatmap_size - 1;
                    y[y_boundary_inds] = heatmap_size - 1;

                    var valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size);
                    var vis = keypoints[TensorIndex.Ellipsis, 2] > 0;
                    var valid = (valid_loc & vis).@long();

                    var lin_ind = y * heatmap_size + x;
                    var heatmaps = lin_ind * valid;

                    return (heatmaps, valid);
                }

                /// <summary>
                /// Extract predicted keypoint locations from heatmaps. Output has shape
                ///    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
                ///    for each keypoint.
                ///    This function converts a discrete image coordinate in a HEATMAP_SIZE x
                ///    HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
                ///    consistency with keypoints_to_heatmap_labels by using the conversion from
                ///    Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
                ///    continuous coordinate.
                /// </summary>
                public static (Tensor, Tensor) heatmaps_to_keypoints(Tensor maps, Tensor rois)
                {
                    var offset_x = rois[TensorIndex.Colon, 0];
                    var offset_y = rois[TensorIndex.Colon, 1];

                    var widths = rois[TensorIndex.Colon, 2] - rois[TensorIndex.Colon, 0];
                    var heights = rois[TensorIndex.Colon, 3] - rois[TensorIndex.Colon, 1];
                    widths = widths.clamp(min: 1);
                    heights = heights.clamp(min: 1);
                    var widths_ceil = widths.ceil();
                    var heights_ceil = heights.ceil();

                    var num_keypoints = maps.shape[1];

                    var xy_preds = torch.zeros(new long[] { rois.shape[0], 3, num_keypoints }, dtype: torch.float32, device: maps.device);
                    var end_scores = torch.zeros(new long[] { rois.shape[0], num_keypoints }, dtype: torch.float32, device: maps.device);
                    for (int i = 0; i < rois.shape[0]; i++) {
                        var roi_map_width = (widths_ceil[i].item<long>());
                        var roi_map_height = (heights_ceil[i].item<long>());
                        var width_correction = widths[i] / roi_map_width;
                        var height_correction = heights[i] / roi_map_height;
                        var roi_map = functional.interpolate(
                                maps[i][TensorIndex.Colon, TensorIndex.None],
                                size: new long[] { roi_map_height, roi_map_width }, mode: InterpolationMode.Bicubic,
                                align_corners: false
                                )[TensorIndex.Colon, 0];
                        // roi_map_probs = scores_to_probs(roi_map.copy())
                        var w = roi_map.shape[2];
                        var pos = roi_map.reshape(num_keypoints, -1).argmax(dim: 1);

                        var x_int = pos % w;
                        var y_int = torch.div(pos - x_int, w, rounding_mode: RoundingMode.floor);
                        // assert (roi_map_probs[k, y_int, x_int] ==
                        // roi_map_probs[k, :, :].max());
                        var x = (x_int.@float() + 0.5) * width_correction;
                        var y = (y_int.@float() + 0.5) * height_correction;
                        xy_preds[i, 0, TensorIndex.Colon] = x + offset_x[i];
                        xy_preds[i, 1, TensorIndex.Colon] = y + offset_y[i];
                        xy_preds[i, 2, TensorIndex.Colon] = 1;
                        end_scores[i, TensorIndex.Colon] = roi_map[torch.arange(num_keypoints, device: roi_map.device), y_int, x_int];
                    }

                    return (xy_preds.permute(0, 2, 1), end_scores);
                }

                public static Tensor keypointrcnn_loss(Tensor keypoint_logits, List<Tensor> proposals, List<Tensor> gt_keypoints, List<Tensor> keypoint_matched_idxs)
                {
                    var (N, K, H, W, _) = keypoint_logits.shape;
                    if (H != W)
                        throw new ArgumentException(string.Format("keypoint_logits height and width " +
                            "(last two elements of shape) should be equal. Instead got H = {0} and W = {1}", H, W));
                    var discretization_size = H;
                    var heatmaps = new List<Tensor>();
                    var valid = new List<Tensor>();
                    for (int i = 0; i < proposals.Count; i++) {
                        var proposals_per_image = proposals[i];
                        var gt_kp_in_image = gt_keypoints[i];
                        var midx = keypoint_matched_idxs[i];

                        var kp = gt_kp_in_image[midx];
                        var (heatmaps_per_image, valid_per_image) = keypoints_to_heatmap(kp, proposals_per_image, discretization_size);
                        heatmaps.Add(heatmaps_per_image.view(-1));
                        valid.Add(valid_per_image.view(-1));
                    }

                    var keypoint_targets = torch.cat(heatmaps, dim: 0);
                    var valid_t = torch.cat(valid, dim: 0).to(torch.uint8);
                    valid_t = torch.where(valid_t)[0];

                    // torch.mean (in binary_cross_entropy_with_logits) does'nt
                    // accept empty tensors, so handle it sepaartely
                    if (keypoint_targets.numel() == 0 || valid_t.shape[0] == 0)
                        return keypoint_logits.sum() * 0;

                    keypoint_logits = keypoint_logits.view(N * K, H * W);

                    var keypoint_loss = functional.cross_entropy(keypoint_logits[valid_t], keypoint_targets[valid_t]);
                    return keypoint_loss;
                }

                public static (IList<Tensor>, IList<Tensor>) keypointrcnn_inference(Tensor x, List<Tensor> boxes)
                {
                    var kp_probs = new List<Tensor>();
                    var kp_scores = new List<Tensor>();

                    var boxes_per_image = boxes.Select(box => box.size(0));
                    var x2 = x.split(boxes_per_image.ToArray(), dim: 0);

                    for (int i = 0; i < x2.Length; i++) {
                        var xx = x2[i];
                        var bb = boxes[i];

                        var (kp_prob, scores) = heatmaps_to_keypoints(xx, bb);
                        kp_probs.Add(kp_prob);
                        kp_scores.Add(scores);
                    }

                    return (kp_probs, kp_scores);
                }

                /// <summary>
                /// the next two functions should be merged inside Masker
                /// but are kept here for the moment while we need them
                /// temporarily for paste_mask_in_image
                /// </summary>
                public static Tensor expand_boxes(Tensor boxes, float scale)
                {
                    var w_half = (boxes[TensorIndex.Colon, 2] - boxes[TensorIndex.Colon, 0]) * 0.5;
                    var h_half = (boxes[TensorIndex.Colon, 3] - boxes[TensorIndex.Colon, 1]) * 0.5;
                    var x_c = (boxes[TensorIndex.Colon, 2] + boxes[TensorIndex.Colon, 0]) * 0.5;
                    var y_c = (boxes[TensorIndex.Colon, 3] + boxes[TensorIndex.Colon, 1]) * 0.5;

                    w_half *= scale;
                    h_half *= scale;

                    var boxes_exp = torch.zeros_like(boxes);
                    boxes_exp[TensorIndex.Colon, 0] = x_c - w_half;
                    boxes_exp[TensorIndex.Colon, 2] = x_c + w_half;
                    boxes_exp[TensorIndex.Colon, 1] = y_c - h_half;
                    boxes_exp[TensorIndex.Colon, 3] = y_c + h_half;
                    return boxes_exp;
                }

                public static (Tensor, float) expand_masks(Tensor mask, long padding)
                {
                    var M = mask.shape[mask.shape.Length - 1];
                    var scale = (float)(M + 2 * padding) / M;
                    var padded_mask = functional.pad(mask, new long[] { padding, padding, padding, padding });
                    return (padded_mask, scale);
                }

                public static Tensor paste_mask_in_image(Tensor mask, Tensor box, long im_h, long im_w)
                {
                    var TO_REMOVE = 1;
                    var w = (long)(box[2] - box[0] + TO_REMOVE);
                    var h = (long)(box[3] - box[1] + TO_REMOVE);
                    w = Math.Max(w, 1);
                    h = Math.Max(h, 1);

                    // Set shape to [batchxCxHxW]
                    mask = mask.expand((1, 1, -1, -1));

                    // Resize mask
                    mask = functional.interpolate(mask, size: new long[] { h, w }, mode: InterpolationMode.Bilinear, align_corners: false);
                    mask = mask[0][0];

                    var im_mask = torch.zeros(new long[] { im_h, im_w }, dtype: mask.dtype, device: mask.device);
                    var x_0 = Math.Max(box[0].item<long>(), 0);
                    var x_1 = Math.Max(box[2].item<long>() + 1, im_w);
                    var y_0 = Math.Max(box[1].item<long>(), 0);
                    var y_1 = Math.Max(box[3].item<long>() + 1, im_h);

                    im_mask[TensorIndex.Slice(y_0, y_1), TensorIndex.Slice(x_0, x_1)] =
                        mask[TensorIndex.Slice((y_0 - box[1].item<long>()), (y_1 - box[1].item<long>())),
                        TensorIndex.Slice((x_0 - box[0].item<long>()), (x_1 - box[0].item<long>()))];
                    return im_mask;
                }

                public static Tensor paste_masks_in_image(Tensor masks, Tensor boxes, long[] img_shape, long padding = 1)
                {
                    var (masks_e, scale) = expand_masks(masks, padding: padding);
                    boxes = expand_boxes(boxes, scale).to(torch.int64);
                    var (im_h, im_w, _) = img_shape;

                    var res = new List<Tensor>();
                    for (int i = 0; i < masks_e.shape[0]; i++) {
                        res.Add(paste_mask_in_image(masks_e[i][0], boxes[i], im_h, im_w));
                    }
                    Tensor ret = null;
                    if (res.Count > 0)
                        ret = torch.stack(res, dim: 0)[TensorIndex.Colon, TensorIndex.None];
                    else
                        ret = masks_e.new_empty(new long[] { 0, 1, im_h, im_w });
                    return ret;
                }
            }
        }
    }

    namespace Modules.Detection
    {
        public class RoIHeads : nn.Module<OrderedDict<string, Tensor>, List<Tensor>, List<long[]>, List<Dictionary<string, Tensor>>,
            (List<Dictionary<string, Tensor>>, Dictionary<string, Tensor>)>
        {
            private Func<Tensor, Tensor, Tensor> box_similarity;
            private Matcher proposal_matcher;
            private BalancedPositiveNegativeSampler fg_bg_sampler;
            private BoxCoder box_coder;
            private MultiScaleRoIAlign box_roi_pool;
            private Module<Tensor, Tensor> box_head;
            private Module<Tensor, (Tensor, Tensor)> box_predictor;
            private float score_thresh;
            private float nms_thresh;
            private long detections_per_img;
            private MultiScaleRoIAlign mask_roi_pool;
            private Module<Tensor, Tensor> mask_head;
            private Module<Tensor, Tensor> mask_predictor;
            private MultiScaleRoIAlign keypoint_roi_pool;
            private Module<Tensor, Tensor> keypoint_head;
            private Module<Tensor, Tensor> keypoint_predictor;

            public RoIHeads(
                string name,
                MultiScaleRoIAlign box_roi_pool,
                nn.Module<Tensor, Tensor> box_head,
                nn.Module<Tensor, (Tensor, Tensor)> box_predictor,
                ////# Faster R-CNN training
                float fg_iou_thresh,
                float bg_iou_thresh,
                long batch_size_per_image,
                float positive_fraction,
                float[] bbox_reg_weights,
                // Faster R-CNN inference
                float score_thresh,
                float nms_thresh,
                long detections_per_img,
                // Mask
                MultiScaleRoIAlign mask_roi_pool = null,
                nn.Module<Tensor, Tensor> mask_head = null,
                nn.Module<Tensor, Tensor> mask_predictor = null,
                MultiScaleRoIAlign keypoint_roi_pool = null,
                nn.Module<Tensor, Tensor> keypoint_head = null,
                nn.Module<Tensor, Tensor> keypoint_predictor = null
                        ) : base(name)
            {
                this.box_similarity = torchvision.ops.box_iou;
                // assign ground-truth boxes for each proposal
                this.proposal_matcher = new Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches: false);
                this.fg_bg_sampler = new BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction);

                if (bbox_reg_weights == null)
                    bbox_reg_weights = new float[] { 10.0f, 10.0f, 5.0f, 5.0f };
                this.box_coder = new BoxCoder(bbox_reg_weights);

                this.box_roi_pool = box_roi_pool;
                this.box_head = box_head;
                this.box_predictor = box_predictor;

                this.score_thresh = score_thresh;
                this.nms_thresh = nms_thresh;
                this.detections_per_img = detections_per_img;

                this.mask_roi_pool = mask_roi_pool;
                this.mask_head = mask_head;
                this.mask_predictor = mask_predictor;

                this.keypoint_roi_pool = keypoint_roi_pool;
                this.keypoint_head = keypoint_head;
                this.keypoint_predictor = keypoint_predictor;
            }

            public bool has_mask()
            {
                if (this.mask_roi_pool == null)
                    return false;
                if (this.mask_head == null)
                    return false;
                if (this.mask_predictor == null)
                    return false;
                return true;
            }

            public bool has_keypoint()
            {
                if (this.keypoint_roi_pool == null)
                    return false;
                if (this.keypoint_head == null)
                    return false;
                if (this.keypoint_predictor == null)
                    return false;
                return true;
            }

            public (List<Tensor>, List<Tensor>) assign_targets_to_proposals(List<Tensor> proposals, List<Tensor> gt_boxes, List<Tensor> gt_labels)
            {
                var matched_idxs = new List<Tensor>();
                var labels = new List<Tensor>();
                for (int i = 0; i < proposals.Count; i++) {
                    var proposals_in_image = proposals[i];
                    var gt_boxes_in_image = gt_boxes[i];
                    var gt_labels_in_image = gt_labels[i];

                    Tensor labels_in_image = null;
                    Tensor clamped_matched_idxs_in_image = null;
                    if (gt_boxes_in_image.numel() == 0) {
                        // Background image
                        var device = proposals_in_image.device;
                        clamped_matched_idxs_in_image = torch.zeros(
                             new long[] { proposals_in_image.shape[0] }, dtype: torch.int64, device: device
                         );
                        labels_in_image = torch.zeros(new long[] { proposals_in_image.shape[0] },
                            dtype: torch.int64, device: device);
                    } else {
                        // set to this.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                        Tensor match_quality_matrix = torchvision.ops.box_iou(gt_boxes_in_image, proposals_in_image);
                        var matched_idxs_in_image = this.proposal_matcher.__call__(match_quality_matrix);

                        clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min: 0);

                        labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image];
                        labels_in_image = labels_in_image.to(torch.int64);

                        // Label background (below the low threshold)
                        var bg_inds = matched_idxs_in_image == this.proposal_matcher.BELOW_LOW_THRESHOLD;
                        labels_in_image[bg_inds] = 0;

                        // Label ignore proposals (between low and high thresholds)
                        var ignore_inds = matched_idxs_in_image == this.proposal_matcher.BETWEEN_THRESHOLDS;
                        labels_in_image[ignore_inds] = -1;  // -1 is ignored by sampler
                    }
                    matched_idxs.Add(clamped_matched_idxs_in_image);
                    labels.Add(labels_in_image);
                }
                return (matched_idxs, labels);
            }

            public List<Tensor> subsample(List<Tensor> labels)
            {
                var (sampled_pos_inds, sampled_neg_inds) = this.fg_bg_sampler.__call__(labels);
                var sampled_inds = new List<Tensor>();
                for (int i = 0; i < sampled_pos_inds.Count; i++) {
                    var pos_inds_img = sampled_pos_inds[i];
                    var neg_inds_img = sampled_neg_inds[i];
                    var img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0];
                    sampled_inds.Add(img_sampled_inds);
                }
                return sampled_inds;
            }

            public List<Tensor> add_gt_proposals(List<Tensor> proposals, List<Tensor> gt_boxes)
            {
                var result = new List<Tensor>();
                for (int i = 0; i < proposals.Count; i++) {
                    var proposal = proposals[i];
                    var gt_box = gt_boxes[i];
                    result.Add(torch.cat(new List<Tensor> { proposal, gt_box }));
                }
                return result;
            }

            public void check_targets(List<Dictionary<string, Tensor>> targets = null)
            {
                if (targets == null)
                    throw new ArgumentException("targets should not be None");
                foreach (var target in targets)
                    if (!target.ContainsKey("boxes"))
                        throw new ArgumentException("Every element of targets should have a boxes key");
                foreach (var target in targets)
                    if (!target.ContainsKey("labels"))
                        throw new ArgumentException("Every element of targets should have a labels key");
                if (this.has_mask())
                    foreach (var target in targets)
                        if (!target.ContainsKey("masks"))
                            throw new ArgumentException("Every element of targets should have a masks key");
            }

            public (List<Tensor>, List<Tensor>, List<Tensor>, List<Tensor>) select_training_samples(
                List<Tensor> proposals,
                List<Dictionary<string, Tensor>> targets)
            {
                this.check_targets(targets);
                if (targets == null)
                    throw new ArgumentException("targets should not be None");
                var dtype = proposals[0].dtype;
                var device = proposals[0].device;

                var gt_boxes = new List<Tensor>(targets.Select(t => t["boxes"].to(dtype)));
                var gt_labels = new List<Tensor>(targets.Select(t => t["labels"]));

                // append ground-truth bboxes to propos
                proposals = this.add_gt_proposals(proposals, gt_boxes);

                // get matching gt indices for each proposal
                var (matched_idxs, labels) = this.assign_targets_to_proposals(proposals, gt_boxes, gt_labels);
                // sample a fixed proportion of positive-negative proposals
                var sampled_inds = this.subsample(labels);
                var matched_gt_boxes = new List<Tensor>();
                var num_images = proposals.Count;
                for (int img_id = 0; img_id < num_images; img_id++) {
                    var img_sampled_inds = sampled_inds[img_id];
                    proposals[img_id] = proposals[img_id][img_sampled_inds];
                    labels[img_id] = labels[img_id][img_sampled_inds];
                    matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds];

                    var gt_boxes_in_image = gt_boxes[img_id];
                    if (gt_boxes_in_image.numel() == 0)
                        gt_boxes_in_image = torch.zeros(new long[] { 1, 4 }, dtype: dtype, device: device);
                    matched_gt_boxes.Add(gt_boxes_in_image[matched_idxs[img_id]]);
                }

                var regression_targets = this.box_coder.encode(matched_gt_boxes, proposals);
                return (proposals, matched_idxs, labels, regression_targets);
            }

            public (List<Tensor>, List<Tensor>, List<Tensor>) postprocess_detections(
                Tensor class_logits,
                Tensor box_regression,
                List<Tensor> proposals,
                List<long[]> image_shapes
                )
            {
                var device = class_logits.device;
                var num_classes = class_logits.shape[class_logits.shape.Length - 1];

                var boxes_per_image = new List<long>(proposals.Select(boxes_in_image => boxes_in_image.shape[0]));
                var pred_boxes = this.box_coder.decode(box_regression, proposals);

                var pred_scores = functional.softmax(class_logits, -1);

                var pred_boxes_list = pred_boxes.split(boxes_per_image.ToArray(), 0);
                var pred_scores_list = pred_scores.split(boxes_per_image.ToArray(), 0);

                var all_boxes = new List<Tensor>();
                var all_scores = new List<Tensor>();
                var all_labels = new List<Tensor>();
                for (int i = 0; i < image_shapes.Count; i++) {
                    var boxes = pred_boxes_list[i];
                    var scores = pred_scores_list[i];
                    var image_shape = image_shapes[i];
                    boxes = torchvision.ops.clip_boxes_to_image(boxes, image_shape);

                    // create labels for each prediction
                    var labels = torch.arange(num_classes, device: device);
                    labels = labels.view(1, -1).expand_as(scores);

                    // remove predictions with the background label
                    boxes = boxes[TensorIndex.Colon, TensorIndex.Slice(1)];
                    scores = scores[TensorIndex.Colon, TensorIndex.Slice(1)];
                    labels = labels[TensorIndex.Colon, TensorIndex.Slice(1)];

                    // batch everything, by making every class prediction be a separate instance
                    boxes = boxes.reshape(-1, 4);
                    scores = scores.reshape(-1);
                    labels = labels.reshape(-1);

                    // remove low scoring boxes
                    var inds = torch.where(scores > this.score_thresh)[0];
                    boxes = boxes[inds];
                    scores = scores[inds];
                    labels = labels[inds];

                    // remove empty boxes
                    Tensor keep = torchvision.ops.remove_small_boxes(boxes, min_size: 1e-2);
                    boxes = boxes[keep];
                    scores = scores[keep];
                    labels = labels[keep];

                    // non-maximum suppression, independently done per class
                    keep = torchvision.ops.batched_nms(boxes, scores, labels, this.nms_thresh);
                    // keep only topk scoring predictions
                    keep = keep[TensorIndex.Slice(stop: this.detections_per_img)];
                    boxes = boxes[keep];
                    scores = scores[keep];
                    labels = labels[keep];

                    all_boxes.Add(boxes);
                    all_scores.Add(scores);
                    all_labels.Add(labels);
                }

                return (all_boxes, all_scores, all_labels);
            }

            public override (List<Dictionary<string, Tensor>>, Dictionary<string, Tensor>) forward(
                OrderedDict<string, Tensor> features,
               List<Tensor> proposals,
               List<long[]> image_shapes,
                List<Dictionary<string, Tensor>> targets = null
            )
            {
                if (targets != null) {
                    foreach (var t in targets) {
                        //# TODO: https://github.com/pytorch/pytorch/issues/26731
                        var floating_point_types = new List<ScalarType> { torch.@float, torch.@double, torch.half };
                        if (!floating_point_types.Contains(t["boxes"].dtype))
                            throw new ArgumentException("target boxes must of float type, instead got {0}", t["boxes"].dtype.ToString());
                        if (t["labels"].dtype != torch.int64)
                            throw new ArgumentException("target labels must of int64 type, instead got {0}", t["labels"].dtype.ToString());
                        if (this.has_keypoint())
                            if (t["keypoints"].dtype != torch.float32)
                                throw new ArgumentException("target keypoints must of float type, instead got {0}", t["keypoints"].dtype.ToString());
                    }
                }

                List<Tensor> labels = null;
                List<Tensor> regression_targets = null;
                List<Tensor> matched_idxs = null;
                List<Tensor> boxes = null;
                List<Tensor> scores = null;
                if (this.training) {
                    (proposals, matched_idxs, labels, regression_targets) = this.select_training_samples(proposals, targets);
                }

                Tensor box_features = this.box_roi_pool.forward(features, proposals, image_shapes);
                box_features = this.box_head.forward(box_features);
                var (class_logits, box_regression) = this.box_predictor.forward(box_features);

                var result = new List<Dictionary<string, Tensor>>();
                var losses = new Dictionary<string, Tensor>();
                if (this.training) {
                    if (labels == null)
                        throw new ArgumentException("labels cannot be None");
                    if (regression_targets == null)
                        throw new ArgumentException("regression_targets cannot be None");
                    var (loss_classifier, loss_box_reg) = torchvision.models.detection.fastrcnn_loss(class_logits, box_regression, labels, regression_targets);
                    losses.Add("loss_classifier", loss_classifier);
                    losses.Add("loss_box_reg", loss_box_reg);
                } else {
                    (boxes, scores, labels) = this.postprocess_detections(class_logits, box_regression, proposals, image_shapes);
                    var num_images = boxes.Count;
                    for (int i = 0; i < num_images; i++) {
                        Dictionary<string, Tensor> dict = new Dictionary<string, Tensor>();
                        dict.Add("boxes", boxes[i]);
                        dict.Add("labels", labels[i]);
                        dict.Add("scores", scores[i]);
                        result.Add(dict);
                    }
                }

                if (this.has_mask()) {
                    var mask_proposals = new List<Tensor>(result.Select(p => p["boxes"]));
                    List<Tensor> pos_matched_idxs = null;
                    if (this.training) {
                        if (matched_idxs == null)
                            throw new ArgumentException("if in trainning, matched_idxs should not be None");

                        // during training, only focus on positive boxes
                        var num_images = proposals.Count;
                        mask_proposals = new List<Tensor>();
                        pos_matched_idxs = new List<Tensor>();
                        for (int img_id = 0; img_id < num_images; img_id++) {
                            var pos = torch.where(labels[img_id] > 0)[0];
                            mask_proposals.Add(proposals[img_id][pos]);
                            pos_matched_idxs.Add(matched_idxs[img_id][pos]);
                        }
                    }

                    Tensor mask_logits = null;
                    if (this.mask_roi_pool != null) {
                        Tensor mask_features = this.mask_roi_pool.forward(features, mask_proposals, image_shapes);
                        mask_features = this.mask_head.forward(mask_features);
                        mask_logits = this.mask_predictor.forward(mask_features);
                    } else
                        throw new ArgumentException("Expected mask_roi_pool to be not None");

                    var loss_mask = new Dictionary<string, Tensor>();
                    if (this.training) {
                        if (targets == null || pos_matched_idxs == null || (object)mask_logits == null)
                            throw new ArgumentException("targets, pos_matched_idxs, mask_logits cannot be None when training");

                        var gt_masks = new List<Tensor>(targets.Select(t => t["masks"]));
                        var gt_labels = new List<Tensor>(targets.Select(t => t["labels"]));
                        var rcnn_loss_mask = torchvision.models.detection.maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs);

                        loss_mask.Add("loss_mask", rcnn_loss_mask);
                    } else {
                        labels = new List<Tensor>(result.Select(r => r["labels"]));
                        var masks_probs = torchvision.models.detection.maskrcnn_inference(mask_logits, labels);
                        for (int i = 0; i < masks_probs.Count; i++)
                            result[i]["masks"] = masks_probs[i];
                    }

                    foreach (var key in loss_mask.Keys) {
                        if (losses.ContainsKey(key))
                            losses[key] = loss_mask[key];
                        else
                            losses.Add(key, loss_mask[key]);
                    }
                }

                // keep none checks in if conditional so torchscript will conditionally
                // compile each branch
                if (
                    this.keypoint_roi_pool != null &&
                    this.keypoint_head != null &&
                    this.keypoint_predictor != null) {
                    List<Tensor> pos_matched_idxs = null;
                    var keypoint_proposals = new List<Tensor>(result.Select(p => p["boxes"]));
                    if (this.training) {
                        // during training, only focus on positive boxes
                        var num_images = (proposals).Count;
                        keypoint_proposals = new List<Tensor>();
                        pos_matched_idxs = new List<Tensor>();
                        if (matched_idxs == null)
                            throw new ArgumentException("if in trainning, matched_idxs should not be None");

                        for (int img_id = 0; img_id < num_images; img_id++) {
                            var pos = torch.where(labels[img_id] > 0)[0];
                            keypoint_proposals.Add(proposals[img_id][pos]);
                            pos_matched_idxs.Add(matched_idxs[img_id][pos]);
                        }
                    }

                    Tensor keypoint_features = this.keypoint_roi_pool.forward(features, keypoint_proposals, image_shapes);
                    keypoint_features = this.keypoint_head.forward(keypoint_features);
                    Tensor keypoint_logits = this.keypoint_predictor.forward(keypoint_features);

                    var loss_keypoint = new Dictionary<string, Tensor>();
                    if (this.training) {
                        if (targets == null || pos_matched_idxs == null)
                            throw new ArgumentException("both targets and pos_matched_idxs should not be None when in training mode");

                        var gt_keypoints = new List<Tensor>(targets.Select(t => t["keypoints"]));
                        var rcnn_loss_keypoint = torchvision.models.detection.keypointrcnn_loss(
                            keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                        );
                        loss_keypoint.Add("loss_keypoint", rcnn_loss_keypoint);
                    } else {
                        if (keypoint_logits is null || keypoint_proposals is null)
                            throw new ArgumentException(
                                "both keypoint_logits and keypoint_proposals should not be None when not in training mode");
                        var (keypoints_probs, kp_scores) = torchvision.models.detection.keypointrcnn_inference(keypoint_logits, keypoint_proposals);
                        for (int i = 0; i < keypoints_probs.Count; i++) {
                            result[i]["keypoints"] = keypoints_probs[i];
                            result[i]["keypoints_scores"] = kp_scores[i];
                        }
                    }

                    foreach (var key in loss_keypoint.Keys) {
                        if (losses.ContainsKey(key))
                            losses[key] = loss_keypoint[key];
                        else
                            losses.Add(key, loss_keypoint[key]);
                    }
                }

                return (result, losses);
            }
        }
    }
}
