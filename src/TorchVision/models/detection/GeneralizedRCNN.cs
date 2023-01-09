// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/generalized_rcnn.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System.Collections.Generic;
using System.Collections;
using System.Diagnostics;
using System;
using static TorchSharp.torch;

namespace TorchSharp
{
    namespace Modules.Detection
    {
        /// <summary>
        /// Main class for Generalized R-CNN.
        /// </summary>
        public class GeneralizedRCNN : nn.Module<List<Tensor>, List<Dictionary<string, Tensor>>,
            (Dictionary<string, Tensor>, List<Dictionary<string, Tensor>>)>
        {
            private GeneralizedRCNNTransform transform;
            private nn.Module<Tensor, Tensor> backbone;
            private RegionProposalNetwork rpn;
            private RoIHeads roi_heads;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="backbone">backbone</param>
            /// <param name="rpn">rpn</param>
            /// <param name="roi_heads">takes the features + the proposals from the RPN and computes
            ///        detections / masks from it.</param>
            /// <param name="transform">performs the data transformation from the inputs to feed into the model</param>
            public GeneralizedRCNN(
                string name,
                nn.Module<Tensor, Tensor> backbone,
                RegionProposalNetwork rpn,
                RoIHeads roi_heads, GeneralizedRCNNTransform transform)
                : base(name)
            {
                this.transform = transform;
                this.backbone = backbone;
                this.rpn = rpn;
                this.roi_heads = roi_heads;
            }

            /// <summary>
            /// Forward method.
            /// </summary>
            /// <param name="images">images to be processed</param>
            /// <param name="targets">ground-truth boxes present in the image</param>
            /// <returns>the output from the model.
            ///        During training, it returns a dict[Tensor] which contains the losses.
            ///        During testing, it returns list[BoxList] contains additional fields
            ///        like `scores`, `labels` and `mask` (for Mask R-CNN models).</returns>
            public override (Dictionary<string, Tensor>, List<Dictionary<string, Tensor>>)
                forward(List<Tensor> images, List<Dictionary<string, Tensor>> targets = null)
            {
                if (this.training) {
                    if (targets == null)
                        Debug.Assert(false, "targets should not be none when in training mode");
                    else {
                        foreach (var target in targets) {
                            var boxes = target["boxes"];
                            if (boxes is torch.Tensor)
                                Debug.Assert(
                                    boxes.shape.Length == 2 && boxes.shape[boxes.shape.Length - 1] == 4,
                                    string.Format("Expected target boxes to be a tensor of shape [N, 4], got {0}.", boxes.shape)
                                );
                            else
                                Debug.Assert(false, string.Format("Expected target boxes to be of type Tensor, got {0}.", boxes.GetType()));
                        }
                    }
                }

                var original_image_sizes = new List<long[]>();
                foreach (var img in images) {
                    var val = new long[] { img.shape[img.shape.Length - 2], img.shape[img.shape.Length - 1] };

                    Debug.Assert(
                        val.Length == 2,
                        "expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}"
                    );

                    original_image_sizes.Add(new long[] { val[0], val[1] });
                }

                var (imageList, targets_transformed) = this.transform.forward(images, targets);

                // Check for degenerate boxes
                // TODO: Move this to a function
                if (targets_transformed != null) {
                    for (int target_idx = 0; target_idx < targets_transformed.Count; target_idx++) {
                        var target = targets_transformed[target_idx];
                        var boxes = target["boxes"];

                        var degenerate_boxes = boxes[TensorIndex.Colon, TensorIndex.Slice(start: 2)]
                            <= boxes[TensorIndex.Colon, TensorIndex.Slice(stop: 2)];
                        if (degenerate_boxes.any().item<bool>()) {
                            // print the first degenerate box
                            var bb_idx = torch.nonzero(degenerate_boxes.any(dim: 1))[0][0];

                            ArrayList degen_bb = boxes[bb_idx].tolist() as ArrayList;
                            Debug.Assert(
                                false,
                                "All bounding boxes should have positive height and width." +
                                String.Format(" Found invalid box {0} for target at index {1}.", degen_bb, target_idx)
                            );
                        }
                    }
                }

                var features = this.backbone.forward(imageList.tensors);
                Dictionary<string, Tensor> featuresDic = new Dictionary<string, Tensor>();
                if (features is Tensor)
                    featuresDic.Add("0", features);

                var (proposals, proposal_losses) = this.rpn.forward(imageList, featuresDic, targets_transformed);
                var (detections, detector_losses) = this.roi_heads.forward(featuresDic, proposals, imageList.image_sizes, targets_transformed);
                detections = this.transform.postprocess(detections, imageList.image_sizes, original_image_sizes);  // type: ignore[operator]

                var losses = new Dictionary<string, Tensor>();
                foreach (var key in detector_losses.Keys)
                    losses[key] = detector_losses[key];
                foreach (var key in proposal_losses.Keys)
                    losses[key] = proposal_losses[key];

                return (losses, detections);
            }
        }
    }
}
