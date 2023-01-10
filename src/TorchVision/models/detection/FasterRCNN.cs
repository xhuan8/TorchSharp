// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/faster_rcnn.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System.Collections.Generic;
using System;
using TorchSharp.Modules.Detection;
using static TorchSharp.torch;
using TorchSharp.TorchVision.Ops;
using static TorchSharp.torch.nn;
using TorchSharp.Utils;
using System.Diagnostics;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            public static partial class detection
            {
                internal static AnchorGenerator _default_anchorgen()
                {
                    var anchor_sizes = new List<List<int>> { new List<int> { 32 }, new List<int> { 64 },
                        new List<int>{ 128 }, new List<int>{ 256 }, new List<int>{ 512 } };
                    var aspect_ratios = new List<List<double>>();
                    for (int i = 0; i < anchor_sizes.Count; i++) {
                        List<double> item = new List<double> { 0.5, 1.0, 2.0 };
                        aspect_ratios.Add(item);
                    }
                    return new AnchorGenerator(string.Empty, anchor_sizes, aspect_ratios);
                }
            }
        }
    }

    namespace Modules.Detection
    {
        /// <summary>
        /// Implements Faster R-CNN.
        ///
        /// The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
        /// image, and should be in 0-1 range. Different images can have different sizes.
        ///
        /// The behavior of the model changes depending if it is in training or evaluation mode.
        ///
        /// During training, the model expects both the input tensors, as well as a targets (list of dictionary),
        /// containing:
        ///     - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
        ///       ``0 &lt;= x1 &lt; x2 &lt;= W`` and ``0 &lt;= y1 &lt; y2 &lt;= H``.
        ///     - labels (Int64Tensor[N]): the class label for each ground-truth box
        ///
        /// The model returns a Dict[Tensor] during training, containing the classification and regression
        /// losses for both the RPN and the R-CNN.
        ///
        /// During inference, the model requires only the input tensors, and returns the post-processed
        /// predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
        /// follows:
        ///     - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
        ///       ``0 &lt;= x1 &lt; x2 &lt;= W`` and ``0 &lt;= y1 &lt; y2 &lt;= H``.
        ///     - labels (Int64Tensor[N]): the predicted labels for each image
        ///     - scores (Tensor[N]): the scores or each prediction
        /// Example::
        ///
        ///    >>> import torch
        ///    >>> import torchvision
        ///    >>> from torchvision.models.detection import FasterRCNN
        ///    >>> from torchvision.models.detection.rpn import AnchorGenerator
        ///    >>> //# load a pre-trained model for classification and return
        ///    >>> //# only the features
        ///    >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        ///    >>> //# FasterRCNN needs to know the number of
        ///    >>> //# output channels in a backbone. For mobilenet_v2, it's 1280
        ///    >>> //# so we need to add it here
        ///    >>> backbone.out_channels = 1280
        ///    >>>
        ///    >>> //# let's make the RPN generate 5 x 3 anchors per spatial
        ///    >>> //# location, with 5 different sizes and 3 different aspect
        ///    >>> //# ratios. We have a Tuple[Tuple[int]] because each feature
        ///    >>> //# map could potentially have different sizes and
        ///    >>> //# aspect ratios
        ///    >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        ///    >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        ///    >>>
        ///    >>> //# let's define what are the feature maps that we will
        ///    >>> //# use to perform the region of interest cropping, as well as
        ///    >>> //# the size of the crop after rescaling.
        ///    >>> //# if your backbone returns a Tensor, featmap_names is expected to
        ///    >>> //# be ['0']. More generally, the backbone should return an
        ///    >>> //# OrderedDict[Tensor], and in featmap_names you can choose which
        ///    >>> //# feature maps to use.
        ///    >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        ///    >>>                                                 output_size=7,
        ///    >>>                                                 sampling_ratio=2)
        ///    >>>
        ///    >>> //# put the pieces together inside a FasterRCNN model
        ///    >>> model = FasterRCNN(backbone,
        ///    >>>                    num_classes=2,
        ///    >>>                    rpn_anchor_generator=anchor_generator,
        ///    >>>                    box_roi_pool=roi_pooler)
        ///    >>> model.eval()
        ///    >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        ///    >>> predictions = model(x)
        /// </summary>
        public class FasterRCNN : GeneralizedRCNN
        {
            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="backbone">the network used to compute the features for the model.
            ///        It should contain a out_channels attribute, which indicates the number of output
            ///        channels that each feature map has (and it should be the same for all feature maps).
            ///        The backbone should return a single Tensor or and OrderedDict[Tensor].</param>
            /// <param name="num_classes">number of output classes of the model (including the background).
            ///        If box_predictor is specified, num_classes should be null.</param>
            /// <param name="min_size">minimum size of the image to be rescaled before feeding it to the backbone</param>
            /// <param name="max_size">maximum size of the image to be rescaled before feeding it to the backbone</param>
            /// <param name="image_mean">mean values used for input normalization.
            ///        They are generally the mean values of the dataset on which the backbone has been trained
            ///        on</param>
            /// <param name="image_std">std values used for input normalization.
            ///        They are generally the std values of the dataset on which the backbone has been trained on</param>
            /// <param name="rpn_anchor_generator">module that generates the anchors for a set of feature
            ///        maps.</param>
            /// <param name="rpn_head">module that computes the objectness and regression deltas from the RPN</param>
            /// <param name="rpn_pre_nms_top_n_train">number of proposals to keep before applying NMS during training</param>
            /// <param name="rpn_pre_nms_top_n_test">number of proposals to keep before applying NMS during testing</param>
            /// <param name="rpn_post_nms_top_n_train">number of proposals to keep after applying NMS during training</param>
            /// <param name="rpn_post_nms_top_n_test">number of proposals to keep after applying NMS during testing</param>
            /// <param name="rpn_nms_thresh">NMS threshold used for postprocessing the RPN proposals</param>
            /// <param name="rpn_fg_iou_thresh">minimum IoU between the anchor and the GT box so that they can be
            ///        considered as positive during training of the RPN.</param>
            /// <param name="rpn_bg_iou_thresh">maximum IoU between the anchor and the GT box so that they can be
            ///        considered as negative during training of the RPN.</param>
            /// <param name="rpn_batch_size_per_image">number of anchors that are sampled during training of the RPN
            ///        for computing the loss</param>
            /// <param name="rpn_positive_fraction">proportion of positive anchors in a mini-batch during training
            ///        of the RPN</param>
            /// <param name="rpn_score_thresh">during inference, only return proposals with a classification score
            ///        greater than rpn_score_thresh</param>
            /// <param name="box_roi_pool">the module which crops and resizes the feature maps in
            ///        the locations indicated by the bounding boxes</param>
            /// <param name="box_head">module that takes the cropped feature maps as input</param>
            /// <param name="box_predictor">module that takes the output of box_head and returns the
            ///        classification logits and box regression deltas.</param>
            /// <param name="box_score_thresh">during inference, only return proposals with a classification score
            ///        greater than box_score_thresh</param>
            /// <param name="box_nms_thresh">NMS threshold for the prediction head. Used during inference</param>
            /// <param name="box_detections_per_img">maximum number of detections per image, for all classes.</param>
            /// <param name="box_fg_iou_thresh">minimum IoU between the proposals and the GT box so that they can be
            ///        considered as positive during training of the classification head</param>
            /// <param name="box_bg_iou_thresh">maximum IoU between the proposals and the GT box so that they can be
            ///        considered as negative during training of the classification head</param>
            /// <param name="box_batch_size_per_image">number of proposals that are sampled during training
            /// of the classification head</param>
            /// <param name="box_positive_fraction">proportion of positive proposals in a mini-batch during
            /// training of the classification head</param>
            /// <param name="bbox_reg_weights">weights for the encoding/decoding of the bounding boxes</param>
            /// <param name="kwargs">kwargs</param>
            /// <exception cref="ArgumentException"></exception>
            public FasterRCNN(
                string name,
                BackboneWithFPN backbone,
                long? num_classes = null,
                //# transform parameters
                long min_size = 800,
                long max_size = 1333,
                float[] image_mean = null,
                float[] image_std = null,
                //# RPN parameters
                AnchorGenerator rpn_anchor_generator = null,
                nn.Module<List<Tensor>, (List<Tensor>, List<Tensor>)> rpn_head = null,
                long rpn_pre_nms_top_n_train = 2000,
                long rpn_pre_nms_top_n_test = 1000,
                long rpn_post_nms_top_n_train = 2000,
                long rpn_post_nms_top_n_test = 1000,
                float rpn_nms_thresh = 0.7f,
                float rpn_fg_iou_thresh = 0.7f,
                float rpn_bg_iou_thresh = 0.3f,
                long rpn_batch_size_per_image = 256,
                float rpn_positive_fraction = 0.5f,
                float rpn_score_thresh = 0.0f,
                //# Box parameters
                MultiScaleRoIAlign box_roi_pool = null,
                nn.Module<Tensor, Tensor> box_head = null,
                nn.Module<Tensor, (Tensor, Tensor)> box_predictor = null,
                float box_score_thresh = 0.05f,
                float box_nms_thresh = 0.5f,
                long box_detections_per_img = 100,
                float box_fg_iou_thresh = 0.5f,
                float box_bg_iou_thresh = 0.5f,
                long box_batch_size_per_image = 512,
                float box_positive_fraction = 0.25f,
                float[] bbox_reg_weights = null,
                Dictionary<string, object> kwargs = null
                ) : base(name, backbone,
                    GenerateRpn(name, rpn_anchor_generator, rpn_head, rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                        rpn_batch_size_per_image, rpn_positive_fraction,
                        rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
                        rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
                        rpn_nms_thresh, rpn_score_thresh, backbone.Out_channels, kwargs),
                    GenerateRoIHeads(name, box_roi_pool, box_head, box_predictor, box_fg_iou_thresh, box_bg_iou_thresh,
                        box_batch_size_per_image, box_positive_fraction, bbox_reg_weights, box_score_thresh,
                        box_nms_thresh, box_detections_per_img, backbone.Out_channels, num_classes, kwargs),
                    GenerateGeneralizedRCNNTransform(name, min_size, max_size, image_mean != null ? image_mean : new float[] { 0.485f, 0.456f, 0.406f },
                        image_std != null ? image_std : new float[] { 0.229f, 0.224f, 0.225f }, kwargs: kwargs))
            {
                
            }

            private static GeneralizedRCNNTransform GenerateGeneralizedRCNNTransform(string name, long min_size, long max_size,
                float[] image_mean, float[] image_std, Dictionary<string, object> kwargs)
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
                        }
                    }
                }

                return new GeneralizedRCNNTransform(name, min_size, max_size, image_mean != null ? image_mean : new float[] { 0.485f, 0.456f, 0.406f },
                        image_std != null ? image_std : new float[] { 0.229f, 0.224f, 0.225f }, kwargs: kwargs);
            }

            private static RoIHeads GenerateRoIHeads(string name, MultiScaleRoIAlign box_roi_pool,
                nn.Module<Tensor, Tensor> box_head, nn.Module<Tensor, (Tensor, Tensor)> box_predictor,
                float box_fg_iou_thresh, float box_bg_iou_thresh, long box_batch_size_per_image,
                float box_positive_fraction, float[] bbox_reg_weights, float box_score_thresh,
                float box_nms_thresh, long box_detections_per_img, long out_channels, long? num_classes,
                Dictionary<string, object> kwargs)
            {
                if (kwargs != null) {
                    foreach (var key in kwargs.Keys) {
                        switch (key) {
                        case "num_classes":
                            num_classes = (long)kwargs[key];
                            break;
                        case "box_roi_pool":
                            box_roi_pool = (MultiScaleRoIAlign)kwargs[key];
                            break;
                        case "box_head":
                            box_head = (nn.Module<Tensor, Tensor>)kwargs[key];
                            break;
                        case "box_predictor":
                            box_predictor = (nn.Module<Tensor, (Tensor, Tensor)>)kwargs[key];
                            break;
                        case "box_score_thresh":
                            box_score_thresh = (float)kwargs[key];
                            break;
                        case "box_nms_thresh":
                            box_nms_thresh = (float)kwargs[key];
                            break;
                        case "box_detections_per_img":
                            box_detections_per_img = (long)kwargs[key];
                            break;
                        case "box_fg_iou_thresh":
                            box_fg_iou_thresh = (float)kwargs[key];
                            break;
                        case "box_bg_iou_thresh":
                            box_bg_iou_thresh = (float)kwargs[key];
                            break;
                        case "box_batch_size_per_image":
                            box_batch_size_per_image = (long)kwargs[key];
                            break;
                        case "box_positive_fraction":
                            box_positive_fraction = (float)kwargs[key];
                            break;
                        case "bbox_reg_weights":
                            bbox_reg_weights = (float[])kwargs[key];
                            break;
                        }
                    }
                }

                if (box_roi_pool == null)
                    box_roi_pool = new MultiScaleRoIAlign(name, new List<string> { "0", "1", "2", "3" },
                        new List<long> { 7 }, 2);

                if (box_head == null) {
                    var resolution = box_roi_pool.Output_size[0];
                    var representation_size = 1024;
                    box_head = new TwoMLPHead(name, out_channels * (long)Math.Pow(resolution, 2), representation_size);
                }

                if (box_predictor == null) {
                    var representation_size = 1024;
                    box_predictor = new FastRCNNPredictor(name, representation_size, num_classes.Value);
                }

                return new RoIHeads(name, box_roi_pool, box_head, box_predictor, box_fg_iou_thresh, box_bg_iou_thresh,
                        box_batch_size_per_image, box_positive_fraction, bbox_reg_weights, box_score_thresh,
                        box_nms_thresh, box_detections_per_img);
            }

            private static RegionProposalNetwork GenerateRpn(string name,
                AnchorGenerator rpn_anchor_generator,
                nn.Module<List<Tensor>, (List<Tensor>, List<Tensor>)> rpn_head,
                float rpn_fg_iou_thresh, float rpn_bg_iou_thresh,
                long rpn_batch_size_per_image, float rpn_positive_fraction,
                long rpn_pre_nms_top_n_train, long rpn_pre_nms_top_n_test,
                long rpn_post_nms_top_n_train, long rpn_post_nms_top_n_test,
                float rpn_nms_thresh, float rpn_score_thresh, long channels, Dictionary<string, object> kwargs)
            {
                if (kwargs != null) {
                    foreach (var key in kwargs.Keys) {
                        switch (key) {
                        case "rpn_anchor_generator":
                            rpn_anchor_generator = (AnchorGenerator)kwargs[key];
                            break;
                        case "rpn_head":
                            rpn_head = (nn.Module<List<Tensor>, (List<Tensor>, List<Tensor>)>)kwargs[key];
                            break;
                        case "rpn_pre_nms_top_n_train":
                            rpn_pre_nms_top_n_train = (long)kwargs[key];
                            break;
                        case "rpn_pre_nms_top_n_test":
                            rpn_pre_nms_top_n_test = (long)kwargs[key];
                            break;
                        case "rpn_post_nms_top_n_train":
                            rpn_post_nms_top_n_train = (long)kwargs[key];
                            break;
                        case "rpn_post_nms_top_n_test":
                            rpn_post_nms_top_n_test = (long)kwargs[key];
                            break;
                        case "rpn_nms_thresh":
                            rpn_nms_thresh = (float)kwargs[key];
                            break;
                        case "rpn_fg_iou_thresh":
                            rpn_fg_iou_thresh = (float)kwargs[key];
                            break;
                        case "rpn_bg_iou_thresh":
                            rpn_bg_iou_thresh = (float)kwargs[key];
                            break;
                        case "rpn_batch_size_per_image":
                            rpn_batch_size_per_image = (long)kwargs[key];
                            break;
                        case "rpn_positive_fraction":
                            rpn_positive_fraction = (float)kwargs[key];
                            break;
                        case "rpn_score_thresh":
                            rpn_score_thresh = (float)kwargs[key];
                            break;
                        }
                    }
                }

                if (rpn_anchor_generator == null)
                    rpn_anchor_generator = torchvision.models.detection._default_anchorgen();
                if (rpn_head == null)
                    rpn_head = new RPNHead(name, channels, rpn_anchor_generator.num_anchors_per_location()[0]);

                return new RegionProposalNetwork(name, rpn_anchor_generator, rpn_head, rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                        rpn_batch_size_per_image, rpn_positive_fraction,
                        new Dictionary<string, long>() { { "training", rpn_pre_nms_top_n_train }, { "testing", rpn_pre_nms_top_n_test } },
                        new Dictionary<string, long>() { { "training", rpn_post_nms_top_n_train }, { "testing", rpn_post_nms_top_n_test } },
                        rpn_nms_thresh, rpn_score_thresh);
            }
        }

        /// <summary>
        /// Standard heads for FPN-based models
        /// </summary>
        public class TwoMLPHead : nn.Module<Tensor, Tensor>
        {
            private Linear fc6;
            private Linear fc7;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="in_channels">number of input channels</param>
            /// <param name="representation_size">size of the intermediate representation</param>
            public TwoMLPHead(string name, long in_channels, long representation_size) : base(name)
            {
                this.fc6 = nn.Linear(in_channels, representation_size);
                this.fc7 = nn.Linear(representation_size, representation_size);
            }

            public override Tensor forward(Tensor x)
            {
                x = x.flatten(start_dim: 1);

                x = functional.relu(this.fc6.forward(x));
                x = functional.relu(this.fc7.forward(x));

                return x;
            }
        }

        public class FastRCNNConvFCHead : Sequential
        {
            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="input_size">the input size in CHW format.</param>
            /// <param name="conv_layers">feature dimensions of each Convolution layer</param>
            /// <param name="fc_layers">feature dimensions of each FCN layer</param>
            /// <param name="norm_layer">Module specifying the normalization layer to use. Default: null</param>
            public FastRCNNConvFCHead(
                string name,
                long[] input_size,
                List<long> conv_layers,
                List<long> fc_layers,
                Func<long, nn.Module<Tensor, Tensor>> norm_layer = null
            ) : base(GenerateModules(name, input_size, conv_layers, fc_layers, norm_layer))
            {
                foreach (var layer in this.modules()) {
                    if (layer is Conv2d conv2d) {
                        nn.init.kaiming_normal_(conv2d.weight, mode: init.FanInOut.FanOut, nonlinearity: init.NonlinearityType.ReLU);
                        if (conv2d.bias is not null)
                            nn.init.zeros_(conv2d.bias);
                    }
                }
            }

            private static Module<Tensor, Tensor>[] GenerateModules(string name, long[] input_size,
                List<long> conv_layers, List<long> fc_layers, Func<long, nn.Module<Tensor, Tensor>> norm_layer)
            {
                var (in_channels, in_height, in_width, _) = input_size;

                var blocks = new List<nn.Module<Tensor, Tensor>>();
                var previous_channels = in_channels;
                foreach (var current_channels in conv_layers) {
                    blocks.Add(torchvision.ops.Conv2dNormActivation(previous_channels, current_channels, norm_layer: norm_layer));
                    previous_channels = current_channels;
                }
                blocks.Add(nn.Flatten());
                previous_channels = previous_channels * in_height * in_width;
                foreach (var current_channels in fc_layers) {
                    blocks.Add(nn.Linear(previous_channels, current_channels));
                    blocks.Add(nn.ReLU(inplace: true));
                    previous_channels = current_channels;
                }

                return blocks.ToArray();
            }
        }

        /// <summary>
        /// Standard classification + bounding box regression layers for Fast R-CNN.
        /// </summary>
        public class FastRCNNPredictor : nn.Module<Tensor, (Tensor, Tensor)>
        {
            private Linear cls_score;
            private Linear bbox_pred;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="in_channels">number of input channels</param>
            /// <param name="num_classes">number of output classes (including background)</param>
            public FastRCNNPredictor(string name, long in_channels, long num_classes) : base(name)
            {
                this.cls_score = nn.Linear(in_channels, num_classes);
                this.bbox_pred = nn.Linear(in_channels, num_classes * 4);
            }

            public override (Tensor, Tensor) forward(Tensor x)
            {
                if (x.dim() == 4)
                    Debug.Assert(
                        x.shape[2] == 1 && x.shape[3] == 1,
                        string.Format(
                            "x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {0} {1}", x.shape[2], x.shape[3])
                    );
                x = x.flatten(start_dim: 1);
                var scores = this.cls_score.forward(x);
                var bbox_deltas = this.bbox_pred.forward(x);

                return (scores, bbox_deltas);
            }
        }
    }
}
