// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using static TorchSharp.torch;
using static TorchSharp.PInvoke.LibTorchSharp;

namespace TorchSharp
{
    using Modules;

    namespace Modules
    {
        /// <summary>
        /// This class is used to represent a Dropout2d module.
        /// </summary>
        public sealed class Dropout1d : torch.nn.Module<Tensor, Tensor>
        {
            internal Dropout1d(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

            public override Tensor forward(Tensor tensor)
            {
                if (tensor.ndim != 3)
                    throw new ArgumentException("tensor passed to Dropout1d must be of the shape (N,C,L");
                var res = THSNN_Dropout1d_forward(handle, tensor.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }
        }
    }

    public static partial class torch
    {
        public static partial class nn
        {
            /// <summary>
            /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor).
            /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
            /// </summary>
            /// <param name="p">Probability of an element to be zeroed. Default: 0.5</param>
            /// <param name="inplace">If set to true, will do this operation in-place. Default: false</param>
            /// <returns></returns>
            public static Dropout1d Dropout1d(double p = 0.5, bool inplace = false)
            {
                var handle = THSNN_Dropout1d_ctor(p, inplace, out var boxedHandle);
                if (handle == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Dropout1d(handle, boxedHandle);
            }

            public static partial class functional
            {

                /// <summary>
                /// Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor).
                /// Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.
                /// </summary>
                /// <returns></returns>
                public static Tensor dropout1d(Tensor input, double p = 0.5, bool training = true, bool inplace = false)
                {
                    return dropout2d(input.unsqueeze(-1), p, training, inplace).squeeze(-1);
                }
            }
        }
    }
}
