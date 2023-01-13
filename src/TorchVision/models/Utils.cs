// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ebbdfe8a6b47e1f6d6164b0c86ac48839281602/torchvision/models/_utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using TorchSharp.Modules;
using TorchSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchvision;

#nullable enable
namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class models
        {
            internal static partial class _utils
            {
                /// <summary>
                /// This function is taken from the original tf repo.
                /// It ensures that all layers have a channel number that is divisible by 8
                /// It can be seen here:
                /// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
                /// </summary>
                internal static long _make_divisible(double v, long divisor, long? min_value = null)
                {
                    if (!min_value.HasValue) {
                        min_value = divisor;
                    }
                    var new_v = Math.Max(min_value.Value, (long)(v + divisor / 2.0) / divisor * divisor);
                    // Make sure that round down does not go down by more than 10%.
                    if (new_v < 0.9 * v) {
                        new_v += divisor;
                    }
                    return new_v;
                }

                internal static T _ovewrite_value_param<T>(T? input, T new_value) where T : IComparable
                {
                    if (input is not null)
                        if (input.CompareTo(new_value) != 0)
                            throw new ArgumentException(string.Format(
                                "The parameter '{0}' expected value {1} but got {0} instead.", input, new_value));
                    return new_value;
                }
            }
        }
    }

    namespace Modules
    {
        /// <summary>
        /// Module wrapper that returns intermediate layers from a model
        /// It has a strong assumption that the modules have been registered
        /// into the model in the same order as they are used.
        /// This means that one should **not** reuse the same nn.Module
        /// twice in the forward if you want this to work.
        /// Additionally, it is only able to query submodules that are directly
        /// assigned to the model. So if `model` is passed, `model.feature1` can
        /// be returned, but not `model.feature1.layer2`.
        ///
        ///     >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        ///     >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        ///     >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        ///     >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        ///     >>> out = new_m(torch.rand(1, 3, 224, 224))
        ///     >>> print([(k, v.shape) for k, v in out.items()])
        ///     >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        ///     >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
        /// </summary>
        internal class IntermediateLayerGetter : ModuleDict<Module>
        {
            private Dictionary<string, string> return_layers;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="model">model on which we will extract the features</param>
            /// <param name="return_layers">
            /// a dict containing the names
            ///         of the modules for which the activations will be returned as
            ///         the key of the dict, and the value of the dict is the name
            ///         of the returned activation (which the user can specify).
            /// </param>
            public IntermediateLayerGetter(nn.Module model, Dictionary<string, string> return_layers)
            {
                foreach (var key in return_layers.Keys) {
                    bool exists = false;
                    foreach (var (name, _) in model.named_children()) {
                        if (name == key) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) {
                        throw new ArgumentException("return_layers are not present in model");
                    }
                }
                var orig_return_layers = new Dictionary<string, string>();
                foreach (var pair in return_layers)
                    orig_return_layers[pair.Key] = pair.Value;
                var layers = new OrderedDict<string, nn.Module>();
                foreach (var (name, module) in model.named_children()) {
                    layers[name] = module;
                    if (return_layers.ContainsKey(name)) {
                        return_layers.Remove(name);
                    }
                    if (return_layers.Count == 0)
                        break;
                }
                foreach (var pair in layers)
                    base.Add(pair);
                this.return_layers = orig_return_layers;
            }

            public OrderedDict<string, Tensor> forward(Tensor x)
            {
                OrderedDict<string, Tensor> @out = new OrderedDict<string, Tensor>();
                foreach (var (moduleName, module) in this.items()) {
                    if (module is nn.Module<Tensor, Tensor> common)
                        x = common.forward(x);
                    if (this.return_layers.ContainsKey(moduleName)) {
                        string out_name = this.return_layers[moduleName];
                        @out[out_name] = x;
                    }
                }
                return @out;
            }
        }
    }
}