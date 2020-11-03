// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

#nullable enable
namespace TorchSharp.Tensor
{
    public sealed class TorchTensor : IDisposable
    {
        internal IntPtr handle;

        internal TorchTensor(IntPtr handle)
        {
            this.handle = handle;
        }

        public override bool Equals(object? obj)
        {
            return (obj is TorchTensor) && this.Equal((obj as TorchTensor)!);

        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        /// <summary>
        ///   Finalize the tensor. Releases the tensor and its associated data.
        /// </summary>
        ~TorchTensor() => Dispose(false);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        [DllImport("LibTorchSharp")]
        extern static void THSTensor_dispose(IntPtr handle);

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        void Dispose(bool disposing)
        {
            if (handle != IntPtr.Zero) {
                THSTensor_dispose(handle);
                handle = IntPtr.Zero;
            }
        }

        public IntPtr Handle => handle;

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_ndimension(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public long Dimensions => THSTensor_ndimension(handle);

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_element_size(IntPtr handle);

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_numel(IntPtr handle);

        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumberOfElements => THSTensor_numel(handle);

        /// <summary>
        ///  Get the size of each element in the tensor.
        /// </summary>
        public long ElementSize => THSTensor_element_size(handle);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_data(IntPtr handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<T> Data<T>()
        {
            if (NumberOfElements > int.MaxValue)
            {
                throw new ArgumentException("Span only supports up to int.MaxValue elements.");
            }
            unsafe
            {
                var res = THSTensor_data(handle);
                Torch.CheckForErrors();
                // NOTE: there is no safety here.
                return new Span<T>((void*)res, (int)NumberOfElements);
            }
        }

        public T DataItem<T>()
        {
            if (NumberOfElements != 1) throw new ArgumentException("Number of elements in the tensor must be 1");

            return Data<T>()[0];
        }

        [DllImport("LibTorchSharp")]
        private static extern float THSTensor_data_idx_half(IntPtr handle, long i);

        public float ReadHalf(long i)
        {
            if (i >= NumberOfElements) {
                throw new IndexOutOfRangeException("The index is greater than the number of elements in the tensor");
            }
            return THSTensor_data_idx_half(handle, i);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_item(IntPtr handle);

        public TorchScalar Item()
        {
            var sptr = THSTensor_item(Handle);
            Torch.CheckForErrors();
            return new TorchScalar(sptr);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_fill_(IntPtr handle, IntPtr value);

        public TorchTensor FillInPlace(TorchScalar value)
        {
            var res = THSTensor_fill_(handle, value.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get1(IntPtr handle, long i1);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set1(IntPtr handle, long i1, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1]
        {
            get
            {
                var res = THSTensor_get1(handle, i1);
                Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set
            {
                THSTensor_set1(handle, i1, value.Item().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set2(IntPtr handle, long i1, long i2, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2]
        {
            get
            {
                var res = THSTensor_get2(handle, i1, i2);
                Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set
            {
                THSTensor_set2(handle, i1, i2, value.Item().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set3(IntPtr handle, long i1, long i2, long i3, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3]
        {
            get
            {
                var res = THSTensor_get3(handle, i1, i2, i3);
                Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set
            {
                THSTensor_set3(handle, i1, i2, i3, value.Item().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get4(IntPtr handle, long i1, long i2, long i3, long i4);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set4(IntPtr handle, long i1, long i2, long i3, long i4, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4]
        {
            get
            {
                var res = THSTensor_get4(handle, i1, i2, i3, i4);
                Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set
            {
                THSTensor_set4(handle, i1, i2, i3, i4, value.Item().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get5(IntPtr handle, long i1, long i2, long i3, long i4, long i5);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set5(IntPtr handle, long i1, long i2, long i3, long i4, long i5, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4, long i5] {
            get {
                var res = THSTensor_get5(handle, i1, i2, i3, i4, i5);
                Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set {
                THSTensor_set5(handle, i1, i2, i3, i4, i5, value.Item().Handle);
                Torch.CheckForErrors();
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4, long i5, long i6] {
            get {
                var res = THSTensor_get6(handle, i1, i2, i3, i4, i5, i6);
                Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set {
                THSTensor_set6(handle, i1, i2, i3, i4, i5, i6, value.Item().Handle);
                Torch.CheckForErrors();
            }
        }
        [DllImport("LibTorchSharp")]
        private static extern sbyte THSTensor_type(IntPtr handle);

        public ScalarType Type => (ScalarType)THSTensor_type(handle);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        private static extern string THSTensor_device_str(IntPtr handle);

        public string DeviceString
        {
            get
            {
                var res = THSTensor_device_str(handle);
                Torch.CheckForErrors();
                return res;
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern int THSTensor_device_index(IntPtr handle);

        public int DeviceIndex {
            get {
                var res = THSTensor_device_index(handle);
                Torch.CheckForErrors();
                return res;
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern int THSTensor_device_type(IntPtr handle);

        public DeviceType DeviceType {
            get {
                var res = THSTensor_device_type(handle);
                Torch.CheckForErrors();
                return (DeviceType)res;
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_is_sparse(IntPtr handle);

        public bool IsSparse
        {
            get
            {
                var res = THSTensor_is_sparse(handle);
                Torch.CheckForErrors();
                return res;
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);

        public TorchTensor ToType(ScalarType type)
        {
            var res = THSTensor_to_type(handle, (sbyte)type);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_load([MarshalAs(UnmanagedType.LPStr)] string location);

        public static TorchTensor Load(string location)
        {
            var res = THSTensor_load(location);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_save(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string location);

        public void Save(string location)
        {
            THSTensor_save(handle, location);
            Torch.CheckForErrors();
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_requires_grad(IntPtr handle);

        public bool IsGradRequired => THSTensor_requires_grad(handle);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set_requires_grad(IntPtr handle, bool requires_grad);

        public TorchTensor RequiresGrad(bool requiresGrad)
        {
            var res = THSTensor_set_requires_grad(handle, requiresGrad);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cpu(IntPtr handle);

        public TorchTensor Cpu()
        {
            var res = THSTensor_cpu(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cuda(IntPtr handle);

        public TorchTensor Cuda()
        {
            Torch.InitializeDeviceType(DeviceType.CUDA);
            var res = THSTensor_cuda(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_device(IntPtr handle, int device_type, int device_index);

        public TorchTensor ToDevice(DeviceType deviceType, int deviceIndex = 0)
        {
            Torch.InitializeDeviceType(deviceType);
            var res = THSTensor_to_device(handle, (int)deviceType, deviceIndex);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_size(IntPtr handle, long dimension);

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var res = THSTensor_size(handle, dim);
            Torch.CheckForErrors();
            return res;
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long[] Shape
        {
            get
            {
                var dims = new long[Dimensions];
                for (var i = 0; i < dims.Length; i++)
                    dims[i] = GetTensorDimension(i);

                return dims;
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_indices(IntPtr handle);

        public TorchTensor SparseIndices {
            get {
                var res = THSTensor_indices(handle);
                Torch.CheckForErrors();
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_values(IntPtr handle);

        public TorchTensor SparseValues {
            get {
                var res = THSTensor_values(handle);
                Torch.CheckForErrors();
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_stride(IntPtr handle, long dimension);

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride(int dim)
        {
            var res = THSTensor_stride(handle, dim);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
            Torch.CheckForErrors();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_dense(IntPtr handle);

        public TorchTensor ToDense()
        {
            var res = THSTensor_to_dense(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_clone(IntPtr handle);

        public TorchTensor Clone()
        {
            var res = THSTensor_clone(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_contiguous(IntPtr handle);

        public TorchTensor Contiguous()
        {
            var res = THSTensor_contiguous(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_grad(IntPtr handle);

        public TorchTensor Grad()
        {
            var res = THSTensor_grad(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_index_select(IntPtr src, long dimension, IntPtr index);

        public TorchTensor IndexSelect(long dimension, TorchTensor index)
        {
            var res = THSTensor_index_select(handle, dimension, index.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_reshape(IntPtr src, IntPtr shape, int length);

        public TorchTensor Reshape(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    var res = THSTensor_reshape(handle, (IntPtr)pshape, shape.Length);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_squeeze(IntPtr src, long dimension);

        public TorchTensor Squeeze(long dimension)
        {
            var res = THSTensor_squeeze(handle, dimension);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_t(IntPtr src);

        public TorchTensor T()
        {
            var res = THSTensor_t(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_transpose(IntPtr src, long dim1, long dim2);

        public TorchTensor Transpose(long dimension1, long dimension2)
        {
            var res = THSTensor_transpose(handle, dimension1, dimension2);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_transpose_(IntPtr src, long dim1, long dim2);

        public TorchTensor TransposeInPlace(long dimension1, long dimension2)
        {
            return new TorchTensor(THSTensor_transpose_(handle, dimension1, dimension2));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_view(IntPtr src, IntPtr shape, int length);

        public TorchTensor View(params long[] shape)
        {
            unsafe
            {
                fixed (long* pshape = shape)
                {
                    var res = THSTensor_view(handle, (IntPtr)pshape, shape.Length);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add(IntPtr src, IntPtr trg, IntPtr alpha);

        public TorchTensor Add(TorchTensor target)
        {
            return Add(target, 1);
        }

        public TorchTensor Add(TorchTensor target, TorchScalar alpha)
        {
            var res = THSTensor_add(handle, target.Handle, alpha.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add_scalar(IntPtr src, IntPtr trg, IntPtr alpha);

        public TorchTensor AddScalar(TorchScalar scalar)
        {
            return AddScalar(scalar, 1);
        }
        public TorchTensor AddScalar(TorchScalar scalar, TorchScalar alpha)
        {
            return new TorchTensor(THSTensor_add_scalar(handle, scalar.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add_(IntPtr src, IntPtr trg, IntPtr alpha);

        public TorchTensor AddInPlace(TorchTensor target)
        {
            return AddInPlace(target, 1);
        }

        public TorchTensor AddInPlace(TorchTensor target, TorchScalar alpha)
        {
            return new TorchTensor(THSTensor_add_(handle, target.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add_scalar_(IntPtr src, IntPtr trg, IntPtr alpha);

        public TorchTensor AddInPlace(TorchScalar scalar)
        {
            return AddScalarInPlace(scalar, 1);
        }

        public TorchTensor AddScalarInPlace(TorchScalar scalar, TorchScalar alpha)
        {
            var res = THSTensor_add_scalar_(handle, scalar.Handle, alpha.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public TorchTensor Addbmm(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addbmm_(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public TorchTensor AddbmmInPlace(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm_(handle, batch1.Handle, batch2.Handle, beta, alpha);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addcdiv(IntPtr src, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        public TorchTensor Addcdiv(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcdiv(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addcdiv_(IntPtr src, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        public TorchTensor AddcdivInPlace(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcdiv_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addcmul(IntPtr src, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        public TorchTensor Addcmul(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcmul(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addcmul_(IntPtr src, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        public TorchTensor AddcmulInPlace(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcmul_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmm(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        public TorchTensor Addmm(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            var res = THSTensor_addmm(handle, mat1.Handle, mat2.Handle, beta, alpha);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmm_(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        public TorchTensor AddmmInPlace(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            var res = THSTensor_addmm_(handle, mat1.Handle, mat2.Handle, beta, alpha);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmv(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        public TorchTensor Addmv(TorchTensor mat1, TorchTensor vec2, float beta, float alpha)
        {
            var res = THSTensor_addmv(handle, mat1.Handle, vec2.Handle, beta, alpha);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmv_(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        public TorchTensor AddmvInPlace(TorchTensor mat1, TorchTensor vec2, float beta, float alpha)
        {
            var res = THSTensor_addmv_(handle, mat1.Handle, vec2.Handle, beta, alpha);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_all(IntPtr src);

        public TorchTensor All()
        {
            var res = THSTensor_all(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_all_along_dimension(IntPtr src, long dimension, bool keep_dim);

        public TorchTensor All(long dimension, bool keepDim = false)
        {
            var res = THSTensor_all_along_dimension(handle, dimension, keepDim);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_any(IntPtr src);

        public TorchTensor Any()
        {
            var res = THSTensor_any(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_any_along_dimension(IntPtr src, long dimension, bool keep_dim);

        public TorchTensor Any(long dimension, bool keepDim = false)
        {
            var res = THSTensor_any_along_dimension(handle, dimension, keepDim);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmax(IntPtr src);

        public TorchTensor Argmax()
        {
            var res = THSTensor_argmax(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmax_along_dimension(IntPtr src, long dimension, bool keep_dim);

        public TorchTensor Argmax(long dimension, bool keepDim = false)
        {
            var res = THSTensor_argmax_along_dimension(handle, dimension, keepDim);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmin(IntPtr src);

        public TorchTensor Argmin()
        {
            var res = THSTensor_argmin(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmin_along_dimension(IntPtr src, long dimension, bool keep_dim);

        public TorchTensor Argmin(long dimension, bool keepDim = false)
        {
            var res = THSTensor_argmin_along_dimension(handle, dimension, keepDim);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cos(IntPtr src);

        public TorchTensor Cos()
        {
            var res = THSTensor_cos(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cos_(IntPtr src);

        public TorchTensor CosInPlace()
        {
            var res = THSTensor_cos_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sin(IntPtr src);

        public TorchTensor Sin()
        {
            var res = THSTensor_sin(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sin_(IntPtr src);

        public TorchTensor SinInPlace()
        {
            var res = THSTensor_sin_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tan(IntPtr src);

        public TorchTensor Tan()
        {
            var res = THSTensor_tan(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tan_(IntPtr src);

        public TorchTensor TanInPlace()
        {
            var res = THSTensor_tan_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_angle(IntPtr src);

        public TorchTensor Angle()
        {
            return new TorchTensor(THSTensor_angle(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_asin(IntPtr src);

        public TorchTensor Asin()
        {
            return new TorchTensor(THSTensor_asin(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_asin_(IntPtr src);

        public TorchTensor AsinInPlace()
        {
            var res = THSTensor_asin_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_acos(IntPtr src);

        public TorchTensor Acos()
        {
            var res = THSTensor_acos(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_acos_(IntPtr src);

        public TorchTensor AcosInPlace()
        {
            var res = THSTensor_acos_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan(IntPtr src);

        public TorchTensor Atan()
        {
            var res = THSTensor_atan(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan_(IntPtr src);

        public TorchTensor AtanInPlace()
        {
            var res = THSTensor_atan_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan2(IntPtr src, IntPtr other);

        public TorchTensor Atan2(TorchTensor other)
        {
            var res = THSTensor_atan2(handle, other.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan2_(IntPtr src, IntPtr other);

        public TorchTensor Atan2InPlace(TorchTensor other)
        {
            var res = THSTensor_atan2_(handle, other.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sinh(IntPtr src);

        public TorchTensor Sinh()
        {
            var res = THSTensor_sinh(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sinh_(IntPtr src);

        public TorchTensor SinhInPlace()
        {
            var res = THSTensor_sinh_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cosh(IntPtr src);

        public TorchTensor Cosh()
        {
            var res = THSTensor_cosh(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cosh_(IntPtr src);

        public TorchTensor CosInPlaceh()
        {
            var res = THSTensor_cosh_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tanh(IntPtr src);

        public TorchTensor Tanh()
        {
            var res = THSTensor_tanh(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tanh_(IntPtr src);

        public TorchTensor TanhInPlace()
        {
            var res = THSTensor_tanh_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_floor(IntPtr src);

        public TorchTensor Floor()
        {
            var res = THSTensor_floor(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_floor_(IntPtr src);

        public TorchTensor FloorInPlace()
        {
            var res = THSTensor_floor_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ceil(IntPtr src);

        public TorchTensor Ceil()
        {
            var res = THSTensor_ceil(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ceil_(IntPtr src);

        public TorchTensor CeilInPlace()
        {
            var res = THSTensor_ceil_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sign(IntPtr src);

        public TorchTensor Sign()
        {
            var res = THSTensor_sign(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sign_(IntPtr src);

        public TorchTensor SignInPlace()
        {
            var res = THSTensor_sign_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_softplus(IntPtr src);

        public TorchTensor Softplus()
        {
            var res = THSTensor_softplus(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_relu(IntPtr src);

        public TorchTensor Relu()
        {
            var res = THSTensor_relu(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_relu_(IntPtr src);

        public TorchTensor ReluInPlace()
        {
            var res = THSTensor_relu_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_round(IntPtr src);

        public TorchTensor Round()
        {
            var res = THSTensor_round(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_round_(IntPtr src);

        public TorchTensor RoundInPlace()
        {
            var res = THSTensor_round_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_abs(IntPtr src);

        public TorchTensor Abs()
        {
            var res = THSTensor_abs(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_abs_(IntPtr src);

        public TorchTensor AbsInPlace()
        {
            var res = THSTensor_abs_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log10(IntPtr src);

        public TorchTensor Log10()
        {
            var res = THSTensor_log10(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log10_(IntPtr src);

        public TorchTensor Log10InPlace()
        {
            var res = THSTensor_log10_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sqrt(IntPtr src);

        public TorchTensor Sqrt()
        {
            var res = THSTensor_sqrt(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sqrt_(IntPtr src);

        public TorchTensor SqrtInPlace()
        {
            var res = THSTensor_sqrt_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        public static TorchTensor operator -(TorchTensor tensor)
        {
            return tensor.Neg();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_neg(IntPtr src);

        public TorchTensor Neg()
        {
            var res = THSTensor_neg(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_neg_(IntPtr src);

        public TorchTensor NegInPlace()
        {
            var res = THSTensor_neg_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta,
            float alpha);

        public TorchTensor Baddbmm(TorchTensor batch2, TorchTensor mat, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_baddbmm(handle, batch2.Handle, mat.Handle, beta, alpha);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

        public TorchTensor Bmm(TorchTensor batch2)
        {
            var res = THSTensor_bmm(handle, batch2.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bincount(IntPtr tensor, IntPtr weights, long minlength);

        /// <summary>
        /// Count the frequency of each value in an array of non-negative ints.
        /// </summary>
        public TorchTensor Bincount(TorchTensor? weights, long minlength = 0)
        {
            var weightsHandle = (weights is null ? IntPtr.Zero : weights.Handle);
            var res = THSTensor_bincount(handle, weightsHandle, minlength);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_and(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseAnd(TorchTensor other)
        {
            var res = THSTensor_bitwise_and(handle, other.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_and_(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseAndInPlace(TorchTensor other)
        {
            var res = THSTensor_bitwise_and_(handle, other.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_not(IntPtr tensor);

        public TorchTensor BitwiseNot()
        {
            var res = THSTensor_bitwise_not(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_not_(IntPtr tensor);

        public TorchTensor BitwiseNotInPlace(TorchTensor other)
        {
            var res = THSTensor_bitwise_not_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_or(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseOr(TorchTensor other)
        {
            var res = THSTensor_bitwise_or(handle, other.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_or_(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseOrInPlace(TorchTensor other)
        {
            var res = THSTensor_bitwise_or_(handle, other.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_xor(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseXor(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor(handle, other.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_xor_(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseXorInPlace(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor_(handle, other.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cholesky(IntPtr input, bool upper);

        public TorchTensor Cholesky(bool upper = false)
        {
            var res = THSTensor_cholesky(handle, upper);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cholesky_inverse(IntPtr input, bool upper);

        public TorchTensor CholeskyInverse(bool upper = false)
        {
            var res = THSTensor_cholesky_inverse(handle, upper);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cholesky_solve(IntPtr input, IntPtr input2, bool upper);

        public TorchTensor CholeskySolve(TorchTensor input2, bool upper = false)
        {
            var res = THSTensor_cholesky_solve(handle, input2.Handle, upper);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor Clamp(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_clamp(handle, min.Handle, max.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor ClampInPlace(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_clamp_(handle, min.Handle, max.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_max(IntPtr input, IntPtr max);

        public TorchTensor ClampMax(TorchScalar max)
        {
            var res = THSTensor_clamp_max(handle, max.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_max_(IntPtr input, IntPtr max);

        public TorchTensor ClampMaxInPlace(TorchScalar max)
        {
            var res = THSTensor_clamp_max_(handle, max.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_min(IntPtr input, IntPtr min);

        public TorchTensor ClampMin(TorchScalar min)
        {
            var res = THSTensor_clamp_min(handle, min.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_min_(IntPtr input, IntPtr min);

        public TorchTensor ClampMinInPlace(TorchScalar min)
        {
            var res = THSTensor_clamp_min_(handle, min.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cross(IntPtr input, IntPtr other, long dim);

        /// <summary>
        /// Returns the cross product of vectors in dimension dim of input and other.
        /// input and other must have the same size, and the size of their dim dimension should be 3.
        /// </summary>
        public TorchTensor Cross(TorchScalar other, long dim)
        {
            var res = THSTensor_cross(handle, other.Handle, dim);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div(IntPtr src, IntPtr trg);

        public TorchTensor Div(TorchTensor target)
        {
            var res = THSTensor_div(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div_scalar(IntPtr src, IntPtr trg);

        public TorchTensor DivScalar(TorchScalar target)
        {
            var res = THSTensor_div_scalar(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div_(IntPtr src, IntPtr trg);

        public TorchTensor DivInPlace(TorchTensor target)
        {
            var res = THSTensor_div_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div_scalar_(IntPtr src, IntPtr trg);

        public TorchTensor DivScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_div_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erf(IntPtr src);

        public TorchTensor Erf()
        {
            var res = THSTensor_erf(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erf_(IntPtr src);

        public TorchTensor ErfInPlace()
        {
            var res = THSTensor_erf_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq(IntPtr src, IntPtr trg);

        public TorchTensor Eq(TorchTensor target)
        {
            var res = THSTensor_eq(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq_(IntPtr src, IntPtr trg);

        public TorchTensor EqInPlace(TorchTensor target)
        {
            var res = THSTensor_eq_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq_scalar(IntPtr src, IntPtr trg);

        public TorchTensor EqScalar(TorchScalar target)
        {
            var res = THSTensor_eq_scalar(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq_scalar_(IntPtr src, IntPtr trg);

        public TorchTensor EqScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_eq_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_equal(IntPtr src, IntPtr trg);

        public bool Equal(TorchTensor target)
        {
            var res = THSTensor_equal(handle, target.Handle);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_allclose(IntPtr src, IntPtr trg, double rtol, double atol, bool equal_nan);

        public bool AllClose(TorchTensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
        {
            var res = THSTensor_allclose(handle, target.Handle, rtol, atol, equal_nan);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_exp(IntPtr src);

        public TorchTensor Exp()
        {
            var res = THSTensor_exp(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge(IntPtr src, IntPtr trg);

        public TorchTensor Ge(TorchTensor target)
        {
            var res = THSTensor_ge(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge_(IntPtr src, IntPtr trg);

        public TorchTensor GeInPlace(TorchTensor target)
        {
            var res = THSTensor_ge_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge_scalar(IntPtr src, IntPtr trg);

        public TorchTensor GeScalar(TorchScalar target)
        {
            var res = THSTensor_ge_scalar(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge_scalar_(IntPtr src, IntPtr trg);

        public TorchTensor GeScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_ge_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt(IntPtr src, IntPtr trg);

        public TorchTensor Gt(TorchTensor target)
        {
            var res = THSTensor_gt(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt_(IntPtr src, IntPtr trg);

        public TorchTensor GtInPlace(TorchTensor target)
        {
            var res = THSTensor_gt_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt_scalar(IntPtr src, IntPtr trg);

        public TorchTensor GtScalar(TorchScalar target)
        {
            var res = THSTensor_gt_scalar(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt_scalar_(IntPtr src, IntPtr trg);

        public TorchTensor GtScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_gt_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le(IntPtr src, IntPtr trg);

        public TorchTensor Le(TorchTensor target)
        {
            var res = THSTensor_le(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le_(IntPtr src, IntPtr trg);

        public TorchTensor LeInPlace(TorchTensor target)
        {
            var res = THSTensor_le_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le_scalar(IntPtr src, IntPtr trg);

        public TorchTensor LeScalar(TorchScalar target)
        {
            var res = THSTensor_le_scalar(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le_scalar_(IntPtr src, IntPtr trg);

        public TorchTensor LeScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_le_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log(IntPtr src);

        public TorchTensor Log()
        {
            var res = THSTensor_log(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log_(IntPtr src);

        public TorchTensor LogInPlace()
        {
            var res = THSTensor_log_(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt(IntPtr src, IntPtr trg);

        public TorchTensor Lt(TorchTensor target)
        {
            var res = THSTensor_lt(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt_(IntPtr src, IntPtr trg);

        public TorchTensor LtInPlace(TorchTensor target)
        {
            var res = THSTensor_lt_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt_scalar(IntPtr src, IntPtr trg);

        public TorchTensor LtScalar(TorchScalar target)
        {
            var res = THSTensor_lt_scalar(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt_scalar_(IntPtr src, IntPtr trg);

        public TorchTensor LtScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_lt_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_matmul(IntPtr src, IntPtr target);

        public TorchTensor MatMul(TorchTensor target)
        {
            var res = THSTensor_matmul(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_topk(IntPtr src, AllocatePinnedArray allocator, int k,
            long dimension, bool largest, bool sorted);

        public (TorchTensor values, TorchTensor indexes) TopK(int k, int dimension = -1, bool largest = true, bool sorted = true)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_topk(handle, pa.CreateArray, k, dimension, largest, sorted);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_unbind(IntPtr src, AllocatePinnedArray allocator, long dimension);

        public TorchTensor[] Unbind(int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_unbind(handle, pa.CreateArray, dimension);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }


        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_split_with_sizes(IntPtr src, AllocatePinnedArray allocator, IntPtr psizes, int length, long dimension);

        public TorchTensor[] SplitWithSizes(long[] sizes, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                unsafe
                {
                    fixed (long* psizes = sizes)
                    {
                        THSTensor_split_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_max(IntPtr src, AllocatePinnedArray allocator, long dimension,
            bool keep_dim);

        public (TorchTensor values, TorchTensor indexes) Max(long dimension, bool keepDim = false)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>())
            {
                THSTensor_max(handle, pa.CreateArray, dimension, keepDim);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mean(IntPtr src);

        public TorchTensor Mean()
        {
            var res = THSTensor_mean(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mean_along_dimensions(IntPtr src, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

        public TorchTensor Mean(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
        {
            unsafe {
                fixed (long* pdims = dimensions) {
                    var res = THSTensor_mean_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_median(IntPtr src);

        public TorchTensor Median()
        {
            var res = THSTensor_median(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mm(IntPtr src, IntPtr target);

        public TorchTensor Mm(TorchTensor target)
        {
            var res = THSTensor_mm(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul(IntPtr src, IntPtr target);

        public TorchTensor Mul(TorchTensor target)
        {
            var res = THSTensor_mul(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul_scalar(IntPtr src, IntPtr scalar);

        public TorchTensor MulScalar(TorchScalar scalar)
        {
            var res = THSTensor_mul_scalar(handle, scalar.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul_(IntPtr src, IntPtr target);

        public TorchTensor MulInPlace(TorchTensor target)
        {
            var res = THSTensor_mul_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul_scalar_(IntPtr src, IntPtr target);

        public TorchTensor MulScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_mul_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne(IntPtr src, IntPtr trg);

        public TorchTensor Ne(TorchTensor target)
        {
            var res = THSTensor_ne(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne_(IntPtr src, IntPtr trg);

        public TorchTensor NeInPlace(TorchTensor target)
        {
            var res = THSTensor_ne_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne_scalar(IntPtr src, IntPtr trg);

        public TorchTensor NeScalar(TorchScalar target)
        {
            var res = THSTensor_ne_scalar(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne_scalar_(IntPtr src, IntPtr trg);

        public TorchTensor NeScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_ne_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_norm(IntPtr src, float p);

        public TorchTensor Norm(float p = 2.0f)
        {
            var res = THSTensor_norm(handle, p);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_norm_along_dimension(IntPtr src, int dimension, bool keepdim, float p);

        public TorchTensor Norm(int dimension, bool keepdim = false, float p = 2.0f)
        {
            var res = THSTensor_norm_along_dimension(handle, dimension, keepdim, p);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow(IntPtr src, IntPtr exponent);

        public TorchTensor Pow(TorchTensor exponent)
        {
            var res = THSTensor_pow(handle, exponent.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow_(IntPtr src, IntPtr exponent);

        public TorchTensor PowInPlace(TorchTensor exponent)
        {
            var res = THSTensor_pow_(handle, exponent.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow_scalar(IntPtr src, IntPtr scalar);

        public TorchTensor PowScalar(TorchScalar scalar)
        {
            var res = THSTensor_pow_scalar(handle, scalar.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow_scalar_(IntPtr src, IntPtr scalar);

        public TorchTensor PowScalarInPlace(TorchScalar scalar)
        {
            var res = THSTensor_pow_scalar_(handle, scalar.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder(IntPtr src, IntPtr trg);

        public TorchTensor Remainder(TorchTensor target)
        {
            var res = THSTensor_remainder(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder_(IntPtr src, IntPtr trg);

        public TorchTensor RemainderInPlace(TorchTensor target)
        {
            var res = THSTensor_remainder_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder_scalar(IntPtr src, IntPtr scalar);

        public TorchTensor RemainderScalar(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar(handle, scalar.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainderS_(IntPtr src, IntPtr scalar);

        public TorchTensor RemainderInPlace(TorchScalar scalar)
        {
            var res = THSTensor_remainderS_(handle, scalar.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sigmoid(IntPtr src);

        public TorchTensor Sigmoid()
        {
            var res = THSTensor_sigmoid(handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub(IntPtr src, IntPtr trg);

        public TorchTensor Sub(TorchTensor target)
        {
            var res = THSTensor_sub(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub_scalar(IntPtr src, IntPtr trg);

        public TorchTensor SubScalar(TorchScalar target)
        {
            var res = THSTensor_sub_scalar(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub_(IntPtr src, IntPtr trg);

        public TorchTensor SubInPlace(TorchTensor target)
        {
            var res = THSTensor_sub_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub_scalar_(IntPtr src, IntPtr trg);

        public TorchTensor SubScalarInPlace(TorchScalar target)
        {
            var res = THSTensor_sub_scalar_(handle, target.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sum(IntPtr src, bool has_type, sbyte scalar_type);

        /// <summary>
        /// Returns the sum of all elements in the :attr:`input` tensor.
        /// </summary>
        public TorchTensor Sum(ScalarType? type = null)
        {
            var res = THSTensor_sum(handle, type.HasValue, (sbyte)type.GetValueOrDefault());
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sum_along_dimensions(IntPtr src, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

        /// <summary>
        ///  Returns the sum of each row of the input tensor in the given dimensions.
        /// </summary>
        public TorchTensor Sum(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
        {
            unsafe
            {
                fixed (long* pdims = dimensions)
                {
                    var res = THSTensor_sum_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bernoulli(IntPtr src, double p);

        public TorchTensor Bernoulli(double p)
        {
            var res = THSTensor_bernoulli(handle, p);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bernoulli_(IntPtr src, double p);

        public TorchTensor BernoulliInPlace(double p)
        {
            var res = THSTensor_bernoulli_(handle, p);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cauchy_(IntPtr src, double median, double sigma);

        public TorchTensor CauchyInPlace(double median = 0.0, double sigma = 1.0)
        {
            var res = THSTensor_cauchy_(handle, median, sigma);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_exponential_(IntPtr src, double lambd);

        public TorchTensor ExponentialInPlace(double lambd = 1.0)
        {
            var res = THSTensor_exponential_(handle, lambd);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_geometric_(IntPtr src, double p);

        public TorchTensor GeometricInPlace(double p)
        {
            var res = THSTensor_geometric_(handle, p);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log_normal_(IntPtr src, double mean, double std);

        public TorchTensor LogNormalInPlace(double mean = 1.0, double std = 2.0)
        {
            var res = THSTensor_log_normal_(handle, mean, std);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_normal_(IntPtr src, double mean, double std);

        public TorchTensor NormalInPlace(double mean = 0.0, double std = 1.0)
        {
            var res = THSTensor_normal_(handle, mean, std);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_uniform_(IntPtr src, double from, double to);

        public TorchTensor UniformInPlace(double from = 0.0, double to = 1.0)
        {
            var res = THSTensor_uniform_(handle, from, to);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_multinomial(IntPtr src, double num_samples, bool replacement);

        public TorchTensor Multinomial(double num_samples, bool replacement = false)
        {
            var res = THSTensor_multinomial(handle, num_samples, replacement);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_expand(IntPtr src, IntPtr psizes, int length, bool isImplicit);

        /// <summary>
        ///  Returns a new view of the tensor with singleton dimensions expanded to a larger size.
        /// </summary>
        public TorchTensor Expand(long[] sizes, bool isImplicit = false)
        {
            unsafe
            {
                fixed (long* psizes = sizes)
                {
                    var res = THSTensor_expand(handle, (IntPtr)psizes, sizes.Length, isImplicit);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_scatter(IntPtr tensor, long dimension, IntPtr index, IntPtr source);

        /// <summary>
        ///  Writes all values from the tensor src into self at the indices specified in the index tensor. For each
        ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
        ///  corresponding value in index for dimension = dim.
        /// </summary>
        public TorchTensor Scatter(long dimension, TorchTensor index, TorchTensor src)
        {
            var res = THSTensor_scatter(handle, dimension, index.Handle, src.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_gather(IntPtr tensor, long dimension, IntPtr index);

        /// <summary>
        /// Gathers values along an axis specified by dim.
        /// </summary>
        public TorchTensor Gather(long dimension, TorchTensor index)
        {
            var res = THSTensor_gather(handle, dimension, index.Handle);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_flip(IntPtr src, IntPtr psizes, int length);

        /// <summary>
        ///  Reverse the order of a n-D tensor along given axis in dims.
        /// </summary>
        public TorchTensor Flip(long[] sizes)
        {
            unsafe
            {
                fixed (long* psizes = sizes)
                {
                    var res = THSTensor_flip(handle, (IntPtr)psizes, sizes.Length);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_narrow(IntPtr src, long dimension, long start, long length);

        /// <summary>
        ///  Returns a new tensor that is a narrowed version of the input along one dimension. The
        /// dimension is input from start to start + length. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor Narrow(long dimension, long start, long length)
        {
            var res = THSTensor_narrow(handle, dimension, start, length);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_slice(IntPtr src, long dimension, long start, long length, long step);

        /// <summary>
        ///  Returns a new tensor that is a sliced version of the input along one dimension. The
        /// dimension is input from start to finish-1. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor Slice(long dimension, long start, long finish, long step)
        {
            var res = THSTensor_slice(handle, dimension, start, finish, step);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_unsqueeze(IntPtr src, long dimension);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv1D(TorchTensor weight, TorchTensor? bias = null, 
            long? stride = null,
            long? padding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var paddingArray = new long[] { padding ?? 0 }; 
            var dilationArray = new long[] { dilation ?? 1 };
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = paddingArray, pdilation = dilationArray)
                {
                    var res =
                        THSTensor_conv1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddingArray.Length,
                            (IntPtr)pdilation, dilationArray.Length,
                            groups);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv2D(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_conv2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv3D(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_conv3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length, 
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose1D(TorchTensor weight, TorchTensor? bias = null,
            long? stride = null,
            long? padding = null,
            long? outputPadding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var outputPaddings = new long[] { outputPadding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = paddings, poutputPadding = outputPaddings, pdilation = dilations)
                {
                    var res =
                        THSTensor_conv_transpose1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)poutputPadding, outputPaddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            groups);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose2D(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? outputPadding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation)
                {
                    var res =
                        THSTensor_conv_transpose2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose3D(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? outputPadding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1, 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe
            {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation)
                {
                    var res =
                        THSTensor_conv_transpose3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxpool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool1D(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            unsafe
            {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations)
                {
                    var res =
                        THSTensor_maxpool1d(handle, 
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            ceil_mode);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_maxpool1d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) MaxPool1DWithIndices(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                        THSTensor_maxpool1d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxpool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool2D(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            unsafe
            {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_maxpool2d(handle,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_maxpool2d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) MaxPool2DWithIndices(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                        THSTensor_maxpool2d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxpool3d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool3D(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            unsafe
            {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation)
                {
                    var res =
                        THSTensor_maxpool3d(handle,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_maxpool3d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) MaxPool3DWithIndices(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                        THSTensor_maxpool3d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxunpool2d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength);

        public TorchTensor MaxUnpool2D(TorchTensor indices, long[] outputSize)
        {
            unsafe {
                fixed (long* poutputSize = outputSize) {
                    var res = THSTensor_maxunpool2d(handle, indices.Handle,
                        (IntPtr)poutputSize, outputSize.Length);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxunpool3d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength, IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength);

        public TorchTensor MaxUnpool3D(TorchTensor indices, long[] outputSize, long[] strides, long[] padding)
        {
            unsafe {
                fixed (long* poutputSize = outputSize, pstrides = strides, ppadding = padding) {
                    var res = THSTensor_maxunpool3d(handle, indices.Handle,
                        (IntPtr)poutputSize, outputSize.Length,
                        (IntPtr)pstrides, strides.Length,
                        (IntPtr)ppadding, padding.Length);
                    Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        /// <summary>
        ///  Returns a new tensor with a dimension of size one inserted at the specified position.
        ///  The returned tensor shares the same underlying data with this tensor.
        /// </summary>

        public TorchTensor Unsqueeze(long dimension)
        {
            var res = THSTensor_unsqueeze(handle, dimension);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        // Operators overloading

        public static TorchTensor operator ==(TorchTensor left, TorchTensor right)
        {
            return left.Eq(right);
        }

        public static TorchTensor operator ==(TorchTensor left, TorchScalar right)
        {
            return left.EqScalar(right);
        }

        public static TorchTensor operator ==(TorchScalar left, TorchTensor right)
        {
            return right.EqScalar(left);
        }

        public static TorchTensor operator !=(TorchTensor left, TorchTensor right)
        {
            return left.Ne(right);
        }

        public static TorchTensor operator !=(TorchTensor left, TorchScalar right)
        {
            return left.NeScalar(right);
        }

        public static TorchTensor operator !=(TorchScalar left, TorchTensor right)
        {
            return right.NeScalar(left);
        }

        public static TorchTensor operator +(TorchTensor left, TorchTensor right)
        {
            return left.Add(right);
        }

        public static TorchTensor operator +(TorchTensor left, TorchScalar right)
        {
            return left.AddScalar(right);
        }

        public static TorchTensor operator +(TorchScalar left, TorchTensor right)
        {
            return right.AddScalar(left);
        }

        public static TorchTensor operator *(TorchTensor left, TorchTensor right)
        {
            return left.Mul(right);
        }

        public static TorchTensor operator *(TorchTensor left, TorchScalar right)
        {
            return left.MulScalar(right);
        }

        public static TorchTensor operator *(TorchScalar left, TorchTensor right)
        {
            return right.MulScalar(left);
        }

        public static TorchTensor operator -(TorchTensor left, TorchTensor right)
        {
            return left.Sub(right);
        }

        public static TorchTensor operator -(TorchTensor left, TorchScalar right)
        {
            return new TorchTensor(THSTensor_sub_scalar(left.Handle, right.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_scalar_sub(IntPtr src, IntPtr trg);

        public static TorchTensor operator -(TorchScalar left, TorchTensor right)
        {
            return new TorchTensor(THSTensor_scalar_sub(left.Handle, right.Handle));
        }

        public static TorchTensor operator /(TorchTensor left, TorchTensor right)
        {
            return left.Div(right);
        }

        public static TorchTensor operator /(TorchTensor left, TorchScalar right)
        {
            return left.DivScalar(right);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_scalar_div(IntPtr left, IntPtr right);

        public static TorchTensor operator /(TorchScalar left, TorchTensor right)
        {
            return new TorchTensor(THSTensor_scalar_div(left.Handle, right.Handle));
        }

        public static TorchTensor operator %(TorchTensor left, TorchTensor right)
        {
            return left.Remainder(right);
        }

        public static TorchTensor operator %(TorchTensor left, TorchScalar right)
        {
            return left.RemainderScalar(right);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainderS2(IntPtr left, IntPtr right);

        public static TorchTensor operator %(TorchScalar left, TorchTensor right)
        {
            return new TorchTensor(THSTensor_remainderS2(left.Handle, right.Handle));
        }

        public static TorchTensor operator <(TorchTensor left, TorchTensor right)
        {
            return left.Lt(right);
        }

        public static TorchTensor operator <(TorchTensor left, TorchScalar right)
        {
            return left.LtScalar(right);
        }

        public static TorchTensor operator <(TorchScalar left, TorchTensor right)
        {
            return right.GtScalar(left);
        }

        public static TorchTensor operator <=(TorchTensor left, TorchTensor right)
        {
            return left.Le(right);
        }

        public static TorchTensor operator <=(TorchTensor left, TorchScalar right)
        {
            return left.LeScalar(right);
        }

        public static TorchTensor operator <=(TorchScalar left, TorchTensor right)
        {
            return right.GeScalar(left);
        }

        public static TorchTensor operator >(TorchTensor left, TorchTensor right)
        {
            return left.Gt(right);
        }

        public static TorchTensor operator >(TorchTensor left, TorchScalar right)
        {
            return left.GtScalar(right);
        }

        public static TorchTensor operator >(TorchScalar left, TorchTensor right)
        {
            return right.LtScalar(left);
        }

        public static TorchTensor operator >=(TorchTensor left, TorchTensor right)
        {
            return left.Ge(right);
        }

        public static TorchTensor operator >=(TorchTensor left, TorchScalar right)
        {
            return left.GeScalar(right);
        }

        public static TorchTensor operator >=(TorchScalar left, TorchTensor right)
        {
            return right.LeScalar(left);
        }

        /// <summary>
        ///   Get a string representation of the tensor.
        /// </summary>
        public override string ToString()
        {
            if (Handle == IntPtr.Zero) return "";

            var n = Dimensions;
            if (n == 0)
                return "[]";

            var sb = new StringBuilder("[");
            for (var i = 0; i < n; i++)
            {
                sb.Append(GetTensorDimension(i));
                if (i + 1 < n)
                    sb.Append("x");
            }

            sb.Append("]");
            sb.Append($", device = {DeviceString}");
            return sb.ToString();
        }

    }

    public enum ScalarType : sbyte
    {
        Byte = 0,
        SByte = 1,
        Short = 2,
        Int = 3,
        Long = 4,
        Half = 5,
        Float = 6,
        Double = 7,
        //ComplexHalf = 8,
        ComplexFloat = 9,
        ComplexDouble = 10,
        Bool = 11,
        //QInt8 = 12,
        //QUInt8 = 13,
        //QUInt32 = 14,
        //BHalf = 15
    }

    public static class TensorExtensionMethods
    {
        public static TorchTensor ToTorchTensor<T>(this T[] rawArray, long[] dimensions, bool doCopy = false, bool requiresGrad = false)
        {
            var array = doCopy ? (T[])rawArray.Clone() : rawArray;

            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                    {
                        return ByteTensor.From(array as byte[], dimensions, requiresGrad); ;
                    }
                case bool _ when typeof(T) == typeof(sbyte):
                    {
                        return SByteTensor.From(array as sbyte[], dimensions, requiresGrad); ;
                    }
                case bool _ when typeof(T) == typeof(short):
                    {
                        return ShortTensor.From(array as short[], dimensions, requiresGrad); ;
                    }
                case bool _ when typeof(T) == typeof(int):
                    {
                        return IntTensor.From(array as int[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(long):
                    {
                        return LongTensor.From(array as long[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(double):
                    {
                        return DoubleTensor.From(array as double[], dimensions, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(float):
                    {
                        return FloatTensor.From(array as float[], dimensions, requiresGrad);
                    }
                //case bool _ when typeof(T) == typeof(Half):
                //    {
                //        return HalfTensor.From(array as Half[], dimensions, requiresGrad);
                //    }
                case bool _ when typeof(T) == typeof(bool):
                    {
                        return BoolTensor.From(array as bool[], dimensions, requiresGrad);
                    }
                //case bool _ when typeof(T) == typeof(System.Numerics.Complex):
                //    {
                //        return ComplexDoubleTensor.From(array as System.Numerics.Complex[], dimensions, requiresGrad);
                //    }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        public static TorchTensor ToTorchTensor<T>(this T scalar, bool requiresGrad = false) where T : struct
        {
            if (requiresGrad && typeof(T) != typeof(float) && typeof(T) != typeof(double))
            {
                throw new ArgumentException(nameof(requiresGrad), "Only floating point types support gradients.");
            }

            switch (true)
            {
                case bool _ when typeof(T) == typeof(byte):
                    {
                        return ByteTensor.From((byte)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(sbyte):
                    {
                        return SByteTensor.From((sbyte)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(short):
                    {
                        return ShortTensor.From((short)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(int):
                    {
                        return IntTensor.From((int)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(long):
                    {
                        return LongTensor.From((long)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(double):
                    {
                        return DoubleTensor.From((double)(object)scalar, requiresGrad);
                    }
                case bool _ when typeof(T) == typeof(float):
                    {
                        return FloatTensor.From((float)(object)scalar, requiresGrad);
                    }
                default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_cat(IntPtr src, int len, long dim);

        public static TorchTensor Cat(this TorchTensor[] tensors, long dimension)
        {
            if (tensors.Length == 0)
            {
                throw new ArgumentException(nameof(tensors));
            }
            if (tensors.Length == 1)
            {
                return tensors[0];
            }

            var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            return new TorchTensor(THSTensor_cat(tensorsRef, parray.Array.Length, dimension));
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_stack(IntPtr src, int len, long dim);

        public static TorchTensor Stack(this TorchTensor[] tensors, long dimension)
        {
            var parray = new PinnedArray<IntPtr>();
            IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

            var res = THSTensor_stack(tensorsRef, parray.Array.Length, dimension);
            Torch.CheckForErrors();
            return new TorchTensor(res);
        }

    }
}