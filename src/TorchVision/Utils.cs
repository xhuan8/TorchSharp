// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/_utils.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace TorchSharp
{
    public static partial class torchvision
    {
        public static partial class utils
        {
            public static string sequence_to_str(Array seq, string separate_last = "")
            {
                if (seq == null)
                    return "";
                if (seq.Length == 1)
                    return $"'{seq.GetValue(0)}'";

                string head = "'";
                for (int i = 0; i < seq.Length; i++) {
                    head += seq.GetValue(i).ToString();

                    if (i == seq.Length - 1)
                        continue;
                    if (i == seq.Length - 2) {
                        if (!string.IsNullOrEmpty(separate_last) && seq.Length == 2) {

                        } else
                            head += "','";
                        head += separate_last;
                    }

                    head += "', '";
                }
                head = head.Substring(0, head.Length - 4);
                head += "'";
                return head;
            }
        }
    }
}
