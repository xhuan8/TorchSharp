// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/pytorch/blob/3a02873183e81ed0af76ab46b01c3829b8dc1d35/torch/nn/modules/utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp
{
    using System.Linq;
    using System.Security.Cryptography;
    using System.Xml.Linq;
    using Modules;

    namespace Modules
    {
        public static class ModulesUtils
        {
            public class _ntuple<T>
            {
                private int n;
                private string name;

                public _ntuple(int n, string name)
                {
                    this.n = n;
                    this.name = name;
                }

                public IEnumerable<T> parse(object x)
                {
                    if (x is IEnumerable<T> list)
                        return list;
                    return Enumerable.Repeat((T)x, n);
                }
            }

            public static IEnumerable<T> _single<T>(object x)
            {
                return new _ntuple<T>(1, "_single").parse(x);
            }
            public static IEnumerable<T> _pair<T>(object x)
            {
                return new _ntuple<T>(2, "_pair").parse(x);
            }
            public static IEnumerable<T> _triple<T>(object x)
            {
                return new _ntuple<T>(3, "_triple").parse(x);
            }
            public static IEnumerable<T> _quadruple<T>(object x)
            {
                return new _ntuple<T>(4, "_quadruple").parse(x);
            }
        }
    }
}
