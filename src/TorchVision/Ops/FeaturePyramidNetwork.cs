// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/ops/feature_pyramid_network.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class ops
        {
            /// <summary>
            /// Base class for the extra block in the FPN.
            /// </summary>
            public class ExtraFPNBlock : nn.Module
            {
                //Args:
                //    results (List[Tensor]): the result of the FPN
                //    x (List[Tensor]): the original feature maps
                //    names (List[str]): the names for each one of the
                //        original feature maps

                //Returns:
                //    results (List[Tensor]): the extended set of results
                //        of the FPN
                //    names (List[str]): the extended set of names for the results
                //"""
                public ExtraFPNBlock(string name) : base(name) { }

                /// <summary>
                /// Processes extra block.
                /// </summary>
                /// <param name="results">the result of the FPN</param>
                /// <param name="x">the original feature maps</param>
                /// <param name="names">the names for each one of the original feature maps</param>
                /// <returns>
                /// (List[Tensor]): the extended set of results of the FPN
                /// (List[str]): the extended set of names for the results
                /// </returns>
                /// <exception cref="NotImplementedException"></exception>
                public virtual (List<Tensor>, List<string>) forward(
                    List<Tensor> results,
                    List<Tensor> x,
                    List<string> names
                )
                {
                    throw new NotImplementedException();
                }
            }

            /// <summary>
            /// Module that adds a FPN from on top of a set of feature maps. This is based on
            /// `"Feature Pyramid Network for Object Detection" &lt;https://arxiv.org/abs/1612.03144>`_.
            /// The feature maps are currently supposed to be in increasing depth
            /// order.
            /// The input to the model is expected to be an OrderedDict[Tensor], containing
            /// the feature maps on top of which the FPN will be added.
            /// </summary>
            public class FeaturePyramidNetwork : nn.Module<Utils.OrderedDict<string, Tensor>, Utils.OrderedDict<string, Tensor>>
            {
                //Examples::

                //    >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
                //    >>> # get some dummy data
                //    >>> x = OrderedDict()
                //    >>> x['feat0'] = torch.rand(1, 10, 64, 64)
                //    >>> x['feat2'] = torch.rand(1, 20, 16, 16)
                //    >>> x['feat3'] = torch.rand(1, 30, 8, 8)
                //    >>> # compute the FPN on top of x
                //    >>> output = m(x)
                //    >>> print([(k, v.shape) for k, v in output.items()])
                //    >>> # returns
                //    >>>   [('feat0', torch.Size([1, 5, 64, 64])),
                //    >>>    ('feat2', torch.Size([1, 5, 16, 16])),
                //    >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

                //private static int _version = 2;
                private ModuleList<nn.Module<Tensor, Tensor>> inner_blocks;
                private ModuleList<nn.Module<Tensor, Tensor>> layer_blocks;
                private ExtraFPNBlock extra_blocks;

                /// <summary>
                /// Constructor.
                /// </summary>
                /// <param name="name"></param>
                /// <param name="in_channels_list">number of channels for each feature map that is passed to the module</param>
                /// <param name="out_channels">number of channels of the FPN representation</param>
                /// <param name="extra_blocks">if provided, extra operations will
                ///         be performed. It is expected to take the fpn features, the original
                ///         features and the names of the original features as input, and returns
                ///         a new list of feature maps and their corresponding names</param>
                /// <param name="norm_layer">Module specifying the normalization layer to use. Default: None</param>
                /// <exception cref="ArgumentException"></exception>
                public FeaturePyramidNetwork(
                    string name,
                    List<int> in_channels_list,
                    int out_channels,
                    ExtraFPNBlock extra_blocks = null,
                    Func<long, nn.Module<Tensor, Tensor>> norm_layer = null
                ) : base(name)
                {
                    this.inner_blocks = nn.ModuleList<nn.Module<Tensor, Tensor>>();
                    this.layer_blocks = nn.ModuleList<nn.Module<Tensor, Tensor>>();
                    foreach (var in_channels in in_channels_list) {
                        if (in_channels == 0)
                            throw new ArgumentException("in_channels=0 is currently not supported");
                        var inner_block_module = torchvision.ops.Conv2dNormActivation(
                            in_channels, out_channels, kernel_size: 1, padding: 0, norm_layer: norm_layer, activation_layer: null
                        );
                        var layer_block_module = Conv2dNormActivation(
                            out_channels, out_channels, kernel_size: 3, norm_layer: norm_layer, activation_layer: null
                        );
                        this.inner_blocks.append(inner_block_module);
                        this.layer_blocks.append(layer_block_module);
                    }

                    //# initialize parameters now to avoid modifying the initialization of top_blocks
                    foreach (var m in this.modules()) {
                        if (m is Conv2d conv2D) {
                            nn.init.kaiming_uniform_(conv2D.weight, a: 1);
                            if (conv2D.bias is not null)
                                nn.init.constant_(conv2D.bias, 0);
                        }
                    }

                    if (extra_blocks is not null) {
                        if (!(extra_blocks is ExtraFPNBlock))
                            throw new ArgumentException(string.Format("extra_blocks should be of type ExtraFPNBlock not {0}", extra_blocks.GetType()));
                    }
                    this.extra_blocks = extra_blocks;
                }

                public override (IList<string> missing_keys, IList<string> unexpected_keyes) load_state_dict(
                    Dictionary<string, Tensor> source, bool strict = true, IList<string> skip = null)
                {
                    return base.load_state_dict(source, strict, skip);
                }

                public override (IList<string> missing_keys, IList<string> unexpected_keyes, IList<string> error_msgs) _load_from_state_dict(Dictionary<string, Tensor> state_dict, string prefix, Dictionary<string, object> local_metadata, bool strict)
                {
                    if (!local_metadata.ContainsKey("version") || (int)local_metadata["version"] < 2) {
                        var num_blocks = this.inner_blocks.Count;
                        foreach (var block in new List<string> { "inner_blocks", "layer_blocks" }) {
                            for (int i = 0; i < num_blocks; i++) {
                                foreach (var type in new List<string> { "weight", "bias" }) {
                                    var old_key = string.Format("{0}{1}.{2}.{3}", prefix, block, i, type);
                                    var new_key = string.Format("{0}{1}.{2}.0.{3}", prefix, block, i, type);
                                    if (state_dict.ContainsKey(old_key)) {
                                        state_dict[new_key] = state_dict[old_key];
                                        state_dict.Remove(old_key);
                                    }
                                }
                            }
                        }
                    }
                    return base._load_from_state_dict(state_dict, prefix, local_metadata, strict);
                }

                /// <summary>
                /// This is equivalent to this.inner_blocks[idx](x),
                /// but torchscript doesn't support this yet
                /// </summary>
                /// <param name="x"></param>
                /// <param name="idx"></param>
                /// <returns></returns>
                public Tensor get_result_from_inner_blocks(Tensor x, int idx)
                {
                    var num_blocks = this.inner_blocks.Count;
                    if (idx < 0)
                        idx += num_blocks;
                    var @out = x;
                    for (int i = 0; i < this.inner_blocks.Count; i++) {
                        if (i == idx)
                            @out = this.inner_blocks[i].forward(x);
                    }
                    return @out;
                }

                /// <summary>
                /// This is equivalent to this.layer_blocks[idx](x),
                /// but torchscript doesn't support this yet
                /// </summary>
                /// <param name="x"></param>
                /// <param name="idx"></param>
                /// <returns></returns>
                public Tensor get_result_from_layer_blocks(Tensor x, int idx)
                {
                    var num_blocks = this.layer_blocks.Count;
                    if (idx < 0)
                        idx += num_blocks;
                    var @out = x;
                    for (int i = 0; i < this.layer_blocks.Count; i++) {
                        if (i == idx)
                            @out = this.layer_blocks[i].forward(x);
                    }
                    return @out;
                }

                /// <summary>
                /// Computes the FPN for a set of feature maps.
                /// </summary>
                /// <param name="x">feature maps for each feature level.</param>
                /// <returns>feature maps after FPN layers. They are ordered from highest resolution first.</returns>
                public override Utils.OrderedDict<string, Tensor> forward(Utils.OrderedDict<string, Tensor> x)
                {
                    //# unpack OrderedDict into two lists for easier handling
                    var names = x.Keys.ToList();
                    var values = x.Values.ToList();

                    var last_inner = this.get_result_from_inner_blocks(values[values.Count - 1], -1);
                    var results = new List<Tensor>();
                    results.Add(this.get_result_from_layer_blocks(last_inner, -1));

                    for (int idx = values.Count - 2; idx >= 0; idx--) {
                        var inner_lateral = this.get_result_from_inner_blocks(values[idx], idx);
                        var feat_shape = new long[2];
                        feat_shape[0] = inner_lateral.shape[inner_lateral.shape.Length - 2];
                        feat_shape[1] = inner_lateral.shape[inner_lateral.shape.Length - 1];
                        var inner_top_down = nn.functional.interpolate(last_inner, size: feat_shape, mode: InterpolationMode.Nearest);
                        last_inner = inner_lateral + inner_top_down;
                        results.Insert(0, this.get_result_from_layer_blocks(last_inner, idx));
                    }

                    if (this.extra_blocks is not null)
                        (results, names) = this.extra_blocks.forward(results, values, names);

                    //# make it back an OrderedDict
                    var @out = new Utils.OrderedDict<string, Tensor>();
                    for (int i = 0; i < results.Count; i++) {
                        @out.Add(names[i], results[i]);
                    }
                    return @out;
                }
            }

            /// <summary>
            /// Applies a max_pool2d on top of the last feature map.
            /// </summary>
            public class LastLevelMaxPool : ExtraFPNBlock
            {
                public LastLevelMaxPool(string name) : base(name) { }

                public override (List<Tensor>, List<string>) forward(List<Tensor> x, List<Tensor> y, List<string> names)
                {
                    names.Add("pool");
                    x.Add(nn.functional.max_pool2d(x[-1], 1, 2, 0));
                    return (x, names);
                }
            }

            /// <summary>
            /// This module is used in RetinaNet to generate extra layers, P6 and P7.
            /// </summary>
            public class LastLevelP6P7 : ExtraFPNBlock
            {
                private Conv2d p6;
                private Conv2d p7;
                private bool use_P5;

                public LastLevelP6P7(string name, int in_channels, int out_channels) : base(name)
                {
                    this.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1);
                    this.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1);
                    foreach (var module in new List<Conv2d> { this.p6, this.p7 }) {
                        nn.init.kaiming_uniform_(module.weight, a: 1);
                        nn.init.constant_(module.bias, 0);
                    }
                    this.use_P5 = in_channels == out_channels;
                }

                public override (List<Tensor>, List<string>) forward(List<Tensor> p, List<Tensor> c, List<string> names)
                {
                    var (p5, c5) = (p[p.Count - 1], c[p.Count - 1]);
                    var x = this.use_P5 ? p5 : c5;
                    var p6_result = this.p6.forward(x);
                    var p7_result = this.p7.forward(nn.functional.relu(p6_result));
                    p.Add(p6_result);
                    p.Add(p7_result);
                    names.Add("p6");
                    names.Add("p7");
                    return (p, names);
                }
            }
        }
    }
}
