// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

// A number of implementation details in this file have been translated from the Python version of torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/blob/6ca9c76adb6daf2695d603ad623a9cf1c4f4806f/torchvision/models/detection/image_list.py
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/main/LICENSE
//

using System;
using System.Collections.Generic;
using System.Text;
using static TorchSharp.torch;

namespace TorchSharp
{
    namespace Modules.Detection
    {
        /// <summary>
        /// Structure that holds a list of images (of possibly varying sizes) as a single tensor.
        /// This works by padding the images to the same size, and storing in a field the original sizes of each image.
        /// </summary>
        public class ImageList
        {
            public Tensor tensors { get; set; }
            public List<(int, int)> image_sizes { get; set; }

            /// <summary>
            /// Constructor
            /// </summary>
            /// <param name="tensors">Tensor containing images.</param>
            /// <param name="image_sizes">List of Tuples each containing size of images.</param>
            public ImageList(Tensor tensors, List<(int, int)> image_sizes)
            {
                this.tensors = tensors;
                this.image_sizes = image_sizes;
            }

            public ImageList to(torch.Device device)
            {
                var cast_tensor = this.tensors.to(device);
                return new ImageList(cast_tensor, this.image_sizes);
            }
        }
    }
}
