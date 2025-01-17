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
        /// This class is used to represent an unflattening operation.
        /// </summary>
        public sealed class Unflatten : torch.nn.Module<Tensor, Tensor>
        {
            internal Unflatten(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle)
            {
            }

            public override Tensor forward(Tensor tensor)
            {
                var res = THSNN_Unflatten_forward(handle, tensor.Handle);
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
            /// Unflattens a tensor dim expanding it to a desired shape. For use with Sequential.
            /// </summary>
            /// <param name="dim">Dimension to be unflattened</param>
            /// <param name="unflattenedSize">New shape of the unflattened dimension</param>
            /// <returns></returns>
            public static Unflatten Unflatten(long dim, long[] unflattenedSize)
            {
                unsafe {
                    fixed (long* pUnflattenedSize = unflattenedSize) {
                        var handle = THSNN_Unflatten_ctor(dim, (IntPtr)pUnflattenedSize, unflattenedSize.Length, out var boxedHandle);
                        if (handle == IntPtr.Zero) { CheckForErrors(); }
                        return new Unflatten(handle, boxedHandle);
                    }
                }
            }
        }
    }
}
