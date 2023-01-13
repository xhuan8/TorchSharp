// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/backbone_utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using static TorchSharp.torchvision;
using System.Xml.Linq;
using static TorchSharp.torch;
using static TorchSharp.torchvision.ops;
using TorchSharp.Utils;
using TorchSharp.Modules;
using System.Linq;
using TorchSharp.Modules.Detection;
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
                ///    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.
                ///    Examples::
                ///        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
                ///        >>> backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=3)
                ///        >>> # get some dummy image
                ///        >>> x = torch.rand(1,3,64,64)
                ///        >>> # compute the output
                ///        >>> output = backbone(x)
                ///        >>> print([(k, v.shape) for k, v in output.items()])
                ///        >>> # returns
                ///        >>>   [('0', torch.Size([1, 256, 16, 16])),
                ///        >>>    ('1', torch.Size([1, 256, 8, 8])),
                ///        >>>    ('2', torch.Size([1, 256, 4, 4])),
                ///        >>>    ('3', torch.Size([1, 256, 2, 2])),
                ///        >>>    ('pool', torch.Size([1, 256, 1, 1]))]
                /// </summary>
                /// <param name="backbone_name">resnet architecture. Possible values are 'resnet18', 'resnet34', 'resnet50',
                ///             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'</param>
                /// <param name="weights_file">Path of the pretrained weights for the model</param>
                /// <param name="norm_layer">it is recommended to use the default value. For details visit:
                ///            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)</param>
                /// <param name="trainable_layers">number of trainable (not frozen) layers starting from final block.
                ///            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.</param>
                /// <param name="returned_layers">The layers of the network to return. Each entry must be in ``[1, 4]``.
                ///            By default all layers are returned.</param>
                /// <param name="extra_blocks">if provided, extra operations will
                ///            be performed. It is expected to take the fpn features, the original
                ///            features and the names of the original features as input, and returns
                ///            a new list of feature maps and their corresponding names. By
                ///            default a ``LastLevelMaxPool`` is used.</param>
                /// <returns></returns>
                public static BackboneWithFPN resnet_fpn_backbone(string backbone_name, string weights_file,
                    Func<long, nn.Module<Tensor, Tensor>> norm_layer,
                    int trainable_layers = 3,
                    List<int> returned_layers = null,
                    ExtraFPNBlock extra_blocks = null)
                {
                    var backbone = torchvision.models.get_resnet(backbone_name, weights_file: weights_file,
                        norm_layer: norm_layer);
                    return _resnet_fpn_extractor(backbone, trainable_layers, returned_layers, extra_blocks);
                }

                public static BackboneWithFPN _resnet_fpn_extractor(ResNet backbone, int trainable_layers, List<int> returned_layers = null,
                    ExtraFPNBlock extra_blocks = null, Func<long, nn.Module<Tensor, Tensor>> norm_layer = null)
                {
                    //    # select layers that wont be frozen
                    if (trainable_layers < 0 || trainable_layers > 5)
                        throw new ArgumentException(string.Format("Trainable layers should be in the range [0,5], got {0}", trainable_layers));
                    List<string> layers_to_train = new List<string> { "layer4", "layer3", "layer2", "layer1", "conv1" };
                    for (int i = 4; i >= trainable_layers; i--)
                        layers_to_train.RemoveAt(i);
                    if (trainable_layers == 5)
                        layers_to_train.Add("bn1");
                    foreach (var (name, parameter) in backbone.named_parameters()) {
                        bool needTrain = false;
                        foreach (var layer in layers_to_train) {
                            if (name.StartsWith(layer)) {
                                needTrain = true;
                                break;
                            }
                        }
                        if (!needTrain)
                            parameter.requires_grad_(false);
                    }

                    if (extra_blocks is null)
                        extra_blocks = new LastLevelMaxPool(string.Empty);

                    if (returned_layers is null)
                        returned_layers = new List<int> { 1, 2, 3, 4 };
                    if (returned_layers.Min() <= 0 || returned_layers.Max() >= 5)
                        throw new ArgumentException(string.Format("Each returned layer should be in the range [1,4]. Got {0}", returned_layers));
                    Dictionary<string, string> return_layers = new Dictionary<string, string>();
                    for (int v = 0; v < returned_layers.Count; v++) {
                        var k = returned_layers[v];
                        return_layers.Add("layer" + k.ToString(), v.ToString());
                    }

                    var in_channels_stage2 = backbone.In_planes / 8;
                    var in_channels_list = new List<long>();
                    foreach (var i in returned_layers)
                        in_channels_list.Add((long)Math.Pow(in_channels_stage2 * 2, (i - 1)));
                    var out_channels = 256;
                    return new BackboneWithFPN(string.Empty,
                        backbone, return_layers, in_channels_list, out_channels, extra_blocks: extra_blocks, norm_layer: norm_layer
                    );
                }

                public static int _validate_trainable_layers(bool is_trained, int? trainable_backbone_layers, int max_value, int default_value)
                {
                    //    # don't freeze any layers if pretrained model or backbone is not used
                    if (!is_trained) {
                        if (trainable_backbone_layers is not null) {
                            //warnings.warn(
                            //    "Changing trainable_backbone_layers has not effect if "
                            //    "neither pretrained nor pretrained_backbone have been set to True, "
                            //    f"falling back to trainable_backbone_layers={max_value} so that all layers are trainable"
                            //)
                            trainable_backbone_layers = max_value;
                        }
                    }

                    //    # by default freeze first blocks
                    if (trainable_backbone_layers is null)
                        trainable_backbone_layers = default_value;
                    if (trainable_backbone_layers < 0 || trainable_backbone_layers > max_value)
                        throw new ArgumentException(
                            string.Format("Trainable backbone layers should be in the range [0,{0}], got {1} ",
                            max_value, trainable_backbone_layers)
                        );
                    return trainable_backbone_layers.Value;
                }

                public static nn.Module mobilenet_backbone(
                    string backbone_name,
                    string weights,
                    bool fpn,
                    Func<long, nn.Module<Tensor, Tensor>> norm_layer,
                    int trainable_layers = 2,
                    List<int> returned_layers = null,
                    ExtraFPNBlock extra_blocks = null
                )
                {
                    if (norm_layer == null)
                        norm_layer = torchvision.ops.FrozenBatchNorm2d;
                    nn.Module<Tensor, Tensor> backbone = null;
                    if (backbone_name == "MobileNetV2") {
                        backbone = new MobileNetV2(string.Empty, norm_layer: norm_layer);
                    } else if (backbone_name == "MobileNetV3") {
                        backbone = mobilenet_v3_large(norm_layer);
                    }
                    backbone.load(weights);
                    return _mobilenet_extractor(backbone, fpn, trainable_layers, returned_layers, extra_blocks, norm_layer);
                }

                internal static nn.Module _mobilenet_extractor(nn.Module<Tensor, Tensor> backbone,
                    bool fpn, long trainable_layers, List<int> returned_layers, ExtraFPNBlock extra_blocks,
                    Func<long, nn.Module<Tensor, Tensor>> norm_layer)
                {
                    nn.Module<Tensor, Tensor> backbone_features = null;
                    if (backbone is MobileNetV2 v2)
                        backbone_features = v2.Features;
                    else if (backbone is MobileNetV3 v3)
                        backbone_features = v3.Features;

                    //# Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
                    //# The first and last blocks are always included because they are the C0 (conv1) and Cn.
                    List<int> stage_indices = new List<int>();
                    stage_indices.Add(0);
                    int index = 0;
                    foreach (var (name, feature) in backbone_features.named_modules()) {
                        if (feature.GetType().GetField("_is_cn", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.GetField) == null)
                            stage_indices.Add(index);
                        index++;
                    }
                    stage_indices.Add(index - 1);
                    var num_stages = stage_indices.Count;

                    //# find the index of the layer from which we wont freeze
                    if (trainable_layers < 0 || trainable_layers > num_stages)
                        throw new ArgumentException(string.Format("Trainable layers should be in the range [0,{0}], got {1} ", num_stages, trainable_layers));
                    var freeze_before = trainable_layers == 0 ? index : stage_indices[(int)(num_stages - trainable_layers)];

                    int layerIndex = 0;
                    foreach (var (_, b) in backbone_features.named_modules()) {
                        if (layerIndex >= freeze_before)
                            break;

                        foreach (var parameter in b.parameters())
                            parameter.requires_grad_(false);
                        layerIndex++;
                    }
                    var out_channels = 256;
                    if (fpn) {
                        if (extra_blocks is null)
                            extra_blocks = new LastLevelMaxPool(string.Empty);

                        if (returned_layers is null)
                            returned_layers = new List<int> { num_stages - 2, num_stages - 1 };
                        if (returned_layers.Min() < 0 || returned_layers.Max() >= num_stages)
                            throw new ArgumentException(string.Format("Each returned layer should be in the range [0,{0}], got {1} ", num_stages - 1, returned_layers));
                        var return_layers = new Dictionary<string, string>();
                        for (int v = 0; v < returned_layers.Count; v++) {
                            var k = returned_layers[v];
                            return_layers.Add(stage_indices[k].ToString(), v.ToString());
                        }
                        var in_channels_list = new List<long>();

                        foreach (var i in returned_layers) {
                            int j = stage_indices[i];
                            layerIndex = 0;
                            foreach (var (_, b) in backbone_features.named_modules()) {
                                if (layerIndex == j) {
                                    if (b is MobileNetV2.InvertedResidual ir2)
                                        in_channels_list.Add(ir2.Out_channels);
                                    else if (b is MobileNetV3.InvertedResidual ir3)
                                        in_channels_list.Add(ir3.Out_channels);
                                }
                                layerIndex++;
                            }
                        }
                        return new BackboneWithFPN(string.Empty,
                            backbone_features, return_layers, in_channels_list, out_channels, extra_blocks: extra_blocks, norm_layer: norm_layer);
                    } else {
                        var (_, last_module) = backbone_features.named_modules().ToArray()[backbone_features.named_modules().ToArray().Length - 1];
                        long out_channels_backbone = 0;
                        if (last_module is MobileNetV2.InvertedResidual ir2)
                            out_channels_backbone = ir2.Out_channels;
                        else if (last_module is MobileNetV3.InvertedResidual ir3)
                            out_channels_backbone = ir3.Out_channels;
                        var m = nn.Sequential(
                            backbone_features,
                            //# depthwise linear combination of channels to reduce their size
                            nn.Conv2d(out_channels_backbone, out_channels, 1)
                        );
                        //m.out_channels = out_channels;  //# type: ignore[assignment]
                        return m;
                    }
                }
            }
        }
    }

    namespace Modules.Detection
    {
        /// <summary>
        /// Adds a FPN on top of a model.
        /// Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
        /// extract a submodel that returns the feature maps specified in return_layers.
        /// The same limitations of IntermediateLayerGetter apply here.
        /// </summary>
        public class BackboneWithFPN : nn.Module<Tensor, OrderedDict<string, Tensor>>
        {
            private IntermediateLayerGetter body;
            private FeaturePyramidNetwork fpn;
            private long out_channels;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="name">name</param>
            /// <param name="backbone">backbone</param>
            /// <param name="return_layers">
            /// a dict containing the names
            ///        of the modules for which the activations will be returned as
            ///        the key of the dict, and the value of the dict is the name
            ///        of the returned activation (which the user can specify).
            /// </param>
            /// <param name="in_channels_list">number of channels for each feature map
            /// that is returned, in the order they are present in the OrderedDict</param>
            /// <param name="out_channels">number of channels in the FPN.</param>
            /// <param name="extra_blocks">extra blocks</param>
            /// <param name="norm_layer">Module specifying the normalization layer to use. Default: None</param>
            public BackboneWithFPN(string name, nn.Module<Tensor, Tensor> backbone, Dictionary<string, string> return_layers,
                List<long> in_channels_list, long out_channels, ExtraFPNBlock extra_blocks, Func<long, nn.Module<Tensor, Tensor>> norm_layer)
                : base(name)
            {
                if (extra_blocks == null) {
                    extra_blocks = new LastLevelMaxPool(name);
                }

                this.body = new IntermediateLayerGetter(backbone, return_layers: return_layers);
                this.fpn = new FeaturePyramidNetwork(name, in_channels_list: in_channels_list,
                    out_channels: out_channels, extra_blocks: extra_blocks, norm_layer: norm_layer);
                this.Out_channels = out_channels;
            }

            public long Out_channels { get => out_channels; set => out_channels = value; }

            /// <summary>
            /// Forward.
            /// </summary>
            public override OrderedDict<string, Tensor> forward(Tensor x)
            {
                var result = this.body.forward(x);
                result = this.fpn.forward(result);
                return result;
            }
        }
    }
}
