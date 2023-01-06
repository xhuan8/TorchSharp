// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/rpn.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using static TorchSharp.torch;
using TorchSharp.Utils;
using static Google.Protobuf.Reflection.FieldDescriptorProto.Types;
using static TorchSharp.torchvision;
using System.Runtime.InteropServices;
using Tensorboard;
using System.ComponentModel;
using static TorchSharp.torch.nn;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            public static partial class detection
            {
                public static Tensor permute_and_flatten(Tensor layer, long N, long A, long C, long H, long W)
                {
                    layer = layer.view(N, -1, C, H, W);
                    layer = layer.permute(0, 3, 4, 1, 2);
                    layer = layer.reshape(N, -1, C);
                    return layer;
                }

                public static (Tensor, Tensor) concat_box_prediction_layers(List<Tensor> box_cls, List<Tensor> box_regression)
                {
                    var box_cls_flattened = new List<Tensor>();
                    var box_regression_flattened = new List<Tensor>();
                    //# for each feature level, permute the outputs to make them be in the
                    //# same format as the labels. Note that the labels are computed for
                    //# all feature levels concatenated, so we keep the same representation
                    //# for the objectness and the box_regression
                    for (int i = 0; i < box_cls.Count; i++) {
                        var box_cls_per_level = box_cls[i];
                        var box_regression_per_level = box_regression[i];

                        var (N, AxC, H, W, _) = box_cls_per_level.shape;
                        var Ax4 = box_regression_per_level.shape[1];
                        var A = Ax4 / 4;
                        var C = AxC / A;
                        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W);
                        box_cls_flattened.Add(box_cls_per_level);

                        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W);
                        box_regression_flattened.Add(box_regression_per_level);
                    }

                    //# concatenate on the first dimension (representing the feature levels), to
                    //# take into account the way the labels were generated (with all feature maps
                    //# being concatenated as well)
                    var box_cls_output = torch.cat(box_cls_flattened, dim: 1).flatten(0, -2);
                    var box_regression_output = torch.cat(box_regression_flattened, dim: 1).reshape(-1, 4);
                    return (box_cls_output, box_regression_output);
                }
            }
        }
    }

    namespace Modules.Detection
    {
        /// <summary>
        /// Adds a simple RPN Head with classification and regression heads
        /// </summary>
        public class RPNHead : nn.Module<List<Tensor>, (List<Tensor>, List<Tensor>)>
        {
            internal int? _version = 2;
            private Sequential conv;
            private Conv2d cls_logits;
            private Conv2d bbox_pred;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="in_channels">number of channels of the input feature</param>
            /// <param name="num_anchors">number of anchors to be predicted</param>
            /// <param name="conv_depth">number of convolutions</param>
            public RPNHead(string name, long in_channels, long num_anchors, long conv_depth = 1)
                : base(name)
            {
                var convs = new List<nn.Module<Tensor, Tensor>>();
                for (int i = 0; i < conv_depth; i++)
                    convs.Add(torchvision.ops.Conv2dNormActivation(in_channels, in_channels, kernel_size: 3, norm_layer: null));
                this.conv = nn.Sequential(convs);
                this.cls_logits = nn.Conv2d(in_channels, num_anchors, kernelSize: 1, stride: 1);
                this.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernelSize: 1, stride: 1);

                foreach (var layer in this.modules()) {
                    if (layer is Conv2d conv2D) {
                        torch.nn.init.normal_(conv2D.weight, std: 0.01);  //# type: ignore[arg-type]
                        if (conv2D.bias is not null)
                            torch.nn.init.constant_(conv2D.bias, 0);  //# type: ignore[arg-type]
                    }
                }
            }

            public override (IList<string> missing_keys, IList<string> unexpected_keyes, IList<string> error_msgs)
                _load_from_state_dict(Dictionary<string, Tensor> state_dict, string prefix,
                Dictionary<string, object> local_metadata, bool strict)
            {
                if (_version == null || _version.Value < 2) {
                    foreach (var type in new string[] { "weight", "bias" }) {
                        var old_key = string.Format("{0}conv.{1}", prefix, type);
                        var new_key = string.Format("{0}conv.0.0.{1}", prefix, type);
                        if (state_dict.ContainsKey(old_key)) {
                            state_dict[new_key] = state_dict[old_key];
                            state_dict.Remove(old_key);
                        }
                    }
                }
                return base._load_from_state_dict(state_dict, prefix, local_metadata, strict);
            }

            public override (List<Tensor>, List<Tensor>) forward(List<Tensor> x)
            {
                var logits = new List<Tensor>();
                var bbox_reg = new List<Tensor>();
                foreach (var feature in x) {
                    var t = this.conv.forward(feature);
                    logits.Add(this.cls_logits.forward(t));
                    bbox_reg.Add(this.bbox_pred.forward(t));
                }
                return (logits, bbox_reg);
            }
        }

        /// <summary>
        /// Implements Region Proposal Network (RPN).
        /// </summary>
        public class RegionProposalNetwork : nn.Module<ImageList, Dictionary<string, Tensor>, List<Dictionary<string, Tensor>>, (List<Tensor>, Dictionary<string, Tensor>)>
        {
            private AnchorGenerator anchor_generator;
            private nn.Module<List<Tensor>, (List<Tensor>, List<Tensor>)> head;
            private BoxCoder box_coder;
            private Func<Tensor, Tensor, Tensor> box_similarity;
            private Matcher proposal_matcher;
            private BalancedPositiveNegativeSampler fg_bg_sampler;
            private Dictionary<string, long> _pre_nms_top_n;
            private Dictionary<string, long> _post_nms_top_n;
            private float nms_thresh;
            private float score_thresh;
            private float min_size;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="anchor_generator">module that generates the anchors for a set of feature maps.</param>
            /// <param name="head">module that computes the objectness and regression deltas</param>
            /// <param name="fg_iou_thresh">minimum IoU between the anchor and the GT box so that they can be
            /// considered as positive during training of the RPN.</param>
            /// <param name="bg_iou_thresh">maximum IoU between the anchor and the GT box so that they can be
            /// considered as negative during training of the RPN.</param>
            /// <param name="batch_size_per_image">number of anchors that are sampled during training of the RPN
            /// for computing the loss</param>
            /// <param name="positive_fraction">proportion of positive anchors in a mini-batch during training
            /// of the RPN</param>
            /// <param name="pre_nms_top_n">number of proposals to keep before applying NMS. It should
            /// contain two fields: training and testing, to allow for different values depending
            /// on training or evaluation</param>
            /// <param name="post_nms_top_n">number of proposals to keep after applying NMS. It should
            /// contain two fields: training and testing, to allow for different values depending
            /// on training or evaluation</param>
            /// <param name="nms_thresh">NMS threshold used for postprocessing the RPN proposals</param>
            /// <param name="score_thresh">score_thresh</param>
            public RegionProposalNetwork(string name,
                AnchorGenerator anchor_generator,
                nn.Module<List<Tensor>, (List<Tensor>, List<Tensor>)> head,
                // Faster-RCNN Training
                float fg_iou_thresh,
                float bg_iou_thresh,
                long batch_size_per_image,
                float positive_fraction,
                // Faster-RCNN Inference,
                Dictionary<string, long> pre_nms_top_n,
                Dictionary<string, long> post_nms_top_n,
                float nms_thresh,
                float score_thresh = 0.0f) : base(name)
            {
                this.anchor_generator = anchor_generator;
                this.head = head;
                this.box_coder = new BoxCoder(weights: new float[] { 1.0f, 1.0f, 1.0f, 1.0f });

                //# used during training
                this.box_similarity = torchvision.ops.box_iou;

                this.proposal_matcher = new Matcher(
                    fg_iou_thresh,
                    bg_iou_thresh,
                    allow_low_quality_matches: true
                );

                this.fg_bg_sampler = new BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction);
                //# used during testing
                this._pre_nms_top_n = pre_nms_top_n;
                this._post_nms_top_n = post_nms_top_n;
                this.nms_thresh = nms_thresh;
                this.score_thresh = score_thresh;
                this.min_size = 1e-3f;
            }

            public long pre_nms_top_n()
            {
                if (this.training)
                    return this._pre_nms_top_n["training"];
                return this._pre_nms_top_n["testing"];
            }

            public long post_nms_top_n()
            {
                if (this.training)
                    return this._post_nms_top_n["training"];
                return this._post_nms_top_n["testing"];
            }

            public (List<Tensor>, List<Tensor>) assign_targets_to_anchors(List<Tensor> anchors,
                List<Dictionary<string, Tensor>> targets)
            {
                var labels = new List<Tensor>();
                var matched_gt_boxes = new List<Tensor>();
                for (int i = 0; i < anchors.Count; i++) {
                    var anchors_per_image = anchors[i];
                    var targets_per_image = targets[i];

                    var gt_boxes = targets_per_image["boxes"];

                    Tensor matched_gt_boxes_per_image = null;
                    Tensor labels_per_image = null;
                    if (gt_boxes.numel() == 0) {
                        //# Background image (negative example)
                        var device = anchors_per_image.device;
                        matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype: torch.float32, device: device);
                        labels_per_image = torch.zeros(new long[] { anchors_per_image.shape[0] }, dtype: torch.float32, device: device);
                    } else {
                        var match_quality_matrix = this.box_similarity(gt_boxes, anchors_per_image);
                        var matched_idxs = this.proposal_matcher.__call__(match_quality_matrix);
                        //# get the targets corresponding GT for each proposal
                        //# NB: need to clamp the indices because we can have a single
                        //# GT in the image, and matched_idxs can be -2, which goes
                        //# out of bounds
                        matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min: 0)];

                        labels_per_image = matched_idxs >= 0;
                        labels_per_image = labels_per_image.to(type: ScalarType.Float32);

                        //# Background (negative examples)
                        var bg_indices = matched_idxs == this.proposal_matcher.BELOW_LOW_THRESHOLD;
                        labels_per_image[bg_indices] = 0.0;

                        //# discard indices that are between thresholds
                        var inds_to_discard = matched_idxs == this.proposal_matcher.BETWEEN_THRESHOLDS;
                        labels_per_image[inds_to_discard] = -1.0;
                    }

                    labels.Add(labels_per_image);
                    matched_gt_boxes.Add(matched_gt_boxes_per_image);
                }

                return (labels, matched_gt_boxes);
            }

            internal Tensor _get_top_n_idx(Tensor objectness, List<long> num_anchors_per_level)
            {
                var r = new List<Tensor>();
                long offset = 0;
                foreach (var ob in objectness.split(num_anchors_per_level.ToArray(), 1)) {
                    var num_anchors = ob.shape[1];
                    var pre_nms_top_n = torchvision.models.detection._topk_min(ob, this.pre_nms_top_n(), 1);
                    var (_, top_n_idx) = ob.topk((int)pre_nms_top_n, dim: 1);
                    r.Add(top_n_idx + offset);
                    offset += num_anchors;
                }
                return torch.cat(r, dim: 1);
            }

            public (List<Tensor>, List<Tensor>) filter_proposals(Tensor proposals, Tensor objectness,
                List<(long, long)> image_shapes, List<long> num_anchors_per_level)
            {
                var num_images = proposals.shape[0];
                var device = proposals.device;
                //# do not backprop through objectness
                objectness = objectness.detach();
                objectness = objectness.reshape(num_images, -1);
                var levels = new List<Tensor>();
                for (int i = 0; i < num_anchors_per_level.Count; i++) {
                    levels.Add(torch.full(new long[] { num_anchors_per_level[i] }, i, dtype: torch.int64, device: device));
                }
                var levels_tensor = torch.cat(levels, 0);
                levels_tensor = levels_tensor.reshape(1, -1).expand_as(objectness);

                //# select top_n boxes independently per level before applying nms
                var top_n_idx = this._get_top_n_idx(objectness, num_anchors_per_level);

                var image_range = torch.arange(num_images, device: device);
                var batch_idx = image_range[TensorIndex.Colon, TensorIndex.None];

                objectness = objectness[batch_idx, top_n_idx];
                levels_tensor = levels_tensor[batch_idx, top_n_idx];
                proposals = proposals[batch_idx, top_n_idx];

                var objectness_prob = torch.sigmoid(objectness);

                var final_boxes = new List<Tensor>();
                var final_scores = new List<Tensor>();
                for (int i = 0; i < image_shapes.Count; i++) {
                    var boxes = proposals[i];
                    var scores = objectness_prob[i];
                    var lvl = levels_tensor[i];
                    var img_shape = image_shapes[i];

                    boxes = torchvision.ops.clip_boxes_to_image(boxes, new long[] { img_shape.Item1, img_shape.Item2 });

                    //# remove small boxes
                    var keep = torchvision.ops.remove_small_boxes(boxes, this.min_size);
                    (boxes, scores, lvl) = (boxes[keep], scores[keep], lvl[keep]);

                    //# remove low scoring boxes
                    //# use >= for Backwards compatibility
                    keep = torch.where(scores >= this.score_thresh)[0];
                    (boxes, scores, lvl) = (boxes[keep], scores[keep], lvl[keep]);

                    //# non-maximum suppression, independently done per level
                    keep = torchvision.ops.batched_nms(boxes, scores, lvl, this.nms_thresh);

                    //# keep only topk scoring predictions
                    keep = keep[TensorIndex.Slice(stop: this.post_nms_top_n())];
                    (boxes, scores) = (boxes[keep], scores[keep]);

                    final_boxes.Add(boxes);
                    final_scores.Add(scores);
                }
                return (final_boxes, final_scores);
            }

            public (Tensor, Tensor) compute_loss(Tensor objectness, Tensor pred_bbox_deltas,
                List<Tensor> labels, List<Tensor> regression_targets)
            {
                var (sampled_pos_inds, sampled_neg_inds) = this.fg_bg_sampler.__call__(labels);
                var sampled_pos_inds_arr = torch.where(torch.cat(sampled_pos_inds, dim: 0))[0];
                var sampled_neg_inds_arr = torch.where(torch.cat(sampled_neg_inds, dim: 0))[0];

                var sampled_inds = torch.cat(new List<Tensor> { sampled_pos_inds_arr, sampled_neg_inds_arr }, dim: 0);

                objectness = objectness.flatten();

                var labels_tensor = torch.cat(labels, dim: 0);
                var regression_targets_tensor = torch.cat(regression_targets, dim: 0);

                var box_loss = functional.smooth_l1_loss(
                    pred_bbox_deltas[sampled_pos_inds_arr],
                    regression_targets_tensor[sampled_pos_inds_arr],
                    beta: 1.0 / 9,
                    reduction: Reduction.Sum
                ) / (sampled_inds.numel());

                var objectness_loss = functional.binary_cross_entropy_with_logits(objectness[sampled_inds], labels_tensor[sampled_inds]);

                return (objectness_loss, box_loss);
            }

            /// <summary>
            /// Forward method.
            /// </summary>
            /// <param name="images">images for which we want to compute the predictions</param>
            /// <param name="features">features computed from the images that are
            ///                used for computing the predictions. Each tensor in the list
            ///                correspond to different feature levels</param>
            /// <param name="targets">ground-truth boxes present in the image (optional).
            ///                If provided, each element in the dict should contain a field `boxes`,
            ///                with the locations of the ground-truth boxes.</param>
            /// <returns>boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
            ///                image.
            ///            losses (Dict[str, Tensor]): the losses for the model during training. During
            ///                testing, it is an empty dict.</returns>
            public override (List<Tensor>, Dictionary<string, Tensor>) forward(ImageList images,
                Dictionary<string, Tensor> features, List<Dictionary<string, Tensor>> targets)
            {
                //# RPN uses all feature maps that are available
                var features_list = features.Values.ToList();
                var (objectness, pred_bbox_deltas) = this.head.forward(features_list);
                var anchors = this.anchor_generator.forward(images, features_list);

                var num_images = anchors.Count;
                var num_anchors_per_level_shape_tensors = new List<long[]>();
                foreach (var o in objectness)
                    num_anchors_per_level_shape_tensors.Add(o[0].shape);
                var num_anchors_per_level = new List<long>();
                foreach (var s in num_anchors_per_level_shape_tensors)
                    num_anchors_per_level.Add(s[0] * s[1] * s[2]);

                var (objectness_tensor, pred_bbox_deltas_tensor) = torchvision.models.detection.concat_box_prediction_layers(objectness, pred_bbox_deltas);
                //# apply pred_bbox_deltas to anchors to obtain the decoded proposals
                //# note that we detach the deltas because Faster R-CNN do not backprop through
                //# the proposals
                var proposals = this.box_coder.decode(pred_bbox_deltas_tensor.detach(), anchors);
                proposals = proposals.view(num_images, -1, 4);

                var (boxes, scores) = this.filter_proposals(proposals, objectness_tensor,
                    images.image_sizes, num_anchors_per_level);

                var losses = new Dictionary<string, Tensor>();
                if (this.training) {
                    if (targets is null)
                        throw new ArgumentException("targets should not be None");
                    var (labels, matched_gt_boxes) = this.assign_targets_to_anchors(anchors, targets);
                    var regression_targets = this.box_coder.encode(matched_gt_boxes, anchors);
                    var (loss_objectness, loss_rpn_box_reg) = this.compute_loss(
                         objectness_tensor, pred_bbox_deltas_tensor, labels, regression_targets
                     );
                    losses["loss_objectness"] = loss_objectness;
                    losses["loss_rpn_box_reg"] = loss_rpn_box_reg;
                }
                return (boxes, losses);
            }
        }
    }
}
