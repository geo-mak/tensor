use crate::assertions::assert_not_zst;
use crate::core::alloc::UnsafeBufferPointer;
use crate::metadata::TensorMetaData;
use crate::Tensor;

impl<T, const R: usize> Tensor<T, R> {
    /// Creates a new tensor with the specified dimensions and initializes all elements to a
    /// given value.
    ///
    /// For creating `Tensor` declaratively, consider using `tensor!` macro.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: An array specifying the size of each dimension of the tensor.
    /// - `value`: The value to initialize all elements of the tensor to.
    ///
    /// # Panics
    /// This function will panic if any dimension has `0` value.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor = Tensor::new_set([2, 3], 0);
    ///
    /// assert_eq!(tensor.shape(), &[2, 3]);
    ///
    /// assert_eq!(tensor.get(&[0, 0]), &0);
    /// ```
    pub fn new_set(dimensions: [usize; R], value: T) -> Self
    where
        T: Clone,
    {
        assert_not_zst::<T>();

        let metadata = TensorMetaData::new(dimensions);
        unsafe {
            Self {
                metadata,
                data: UnsafeBufferPointer::new_allocate_memset(metadata.size(), value),
            }
        }
    }

    /// Creates a new tensor with the specified dimensions and initializes all elements to
    /// default value of `T`.
    ///
    /// For creating `Tensor` declaratively, consider using `tensor!` macro.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: An array specifying the size of each dimension of the tensor.
    ///
    /// # Panics
    /// This function will panic if any dimension has `0` value.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor: Tensor<u8, 2> = Tensor::new_default([2, 3]);
    ///
    /// assert_eq!(tensor.shape(), &[2, 3]);
    ///
    /// assert_eq!(tensor.get(&[0, 0]), &0);
    /// ```
    pub fn new_default(dimensions: [usize; R]) -> Self
    where
        T: Default,
    {
        assert_not_zst::<T>();

        let metadata = TensorMetaData::new(dimensions);
        unsafe {
            Self {
                metadata,
                data: UnsafeBufferPointer::new_allocate_default(metadata.size()),
            }
        }
    }

    /// Creates a new tensor with the specified values and dimensions.
    ///
    /// For creating `Tensor` declaratively, consider using `tensor!` macro.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: An array specifying the size of each dimension of the tensor.
    /// - `values`: A slice specifying the values in the tensor.
    ///
    /// # Panics
    /// This function will panic if the slice is empty, or if the size of dimensions doesn't match
    /// the number of provided elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor = Tensor::from_slice([2, 3], &[1,2,3,4,5,6]);
    ///
    /// assert_eq!(tensor.shape(), &[2, 3]);
    ///
    /// assert_eq!(tensor.get(&[0,0]), &1);
    /// assert_eq!(tensor.get(&[1,2]), &6);
    /// ```
    pub fn from_slice(dimensions: [usize; R], values: &[T]) -> Self
    where
        T: Copy,
    {
        assert_not_zst::<T>();

        unsafe {
            Self {
                metadata: TensorMetaData::new_cmp_eq(values.len(), dimensions),
                data: UnsafeBufferPointer::from_slice(values),
            }
        }
    }

    /// Creates a new tensor from vector with the specified dimensions.
    ///
    /// For creating `Tensor` declaratively, consider using `tensor!` macro.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: An array specifying the size of each dimension of the tensor.
    /// - `values`: A vector specifying the values in the tensor.
    ///
    /// # Panics
    /// This function will panic if the vector is empty, or if the size of dimensions doesn't match
    /// the number of provided elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec([2, 3], vec![1,2,3,4,5,6]);
    ///
    /// assert_eq!(tensor.shape(), &[2, 3]);
    ///
    /// assert_eq!(tensor.get(&[0,0]), &1);
    /// assert_eq!(tensor.get(&[1,2]), &6);
    /// ```
    pub fn from_vec(dimensions: [usize; R], values: Vec<T>) -> Self {
        assert_not_zst::<T>();

        unsafe {
            Self {
                metadata: TensorMetaData::new_cmp_eq(values.len(), dimensions),
                data: UnsafeBufferPointer::from_vec(values),
            }
        }
    }
}

#[cfg(test)]
mod instance_tests {
    use super::*;

    #[test]
    fn test_new_set() {
        let tensor = Tensor::new_set([2, 3], 0);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get(&[0, 0]), &0);
        assert_eq!(tensor.get(&[1, 2]), &0);
    }

    #[test]
    fn test_new_set_zero_rank() {
        let tensor = Tensor::new_set([], 3);
        assert_eq!(tensor.shape(), &[]);
        assert_eq!(tensor.get(&[]), &3);
    }

    #[test]
    #[should_panic]
    fn test_new_set_zst() {
        Tensor::new_set([2, 3], ());
    }

    #[test]
    #[should_panic]
    fn test_new_set_invalid_dimensions() {
        Tensor::new_set([2, 0], 0);
    }

    #[test]
    fn test_new_default() {
        let tensor: Tensor<u8, 2> = Tensor::new_default([2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get(&[0, 0]), &0);
        assert_eq!(tensor.get(&[1, 2]), &0);
    }

    #[test]
    fn test_new_default_zero_rank() {
        let tensor: Tensor<u8, 0> = Tensor::new_default([]);
        assert_eq!(tensor.shape(), &[]);
        assert_eq!(tensor.get(&[]), &0);
    }

    #[test]
    #[should_panic]
    fn test_new_default_zst() {
        Tensor::<(), 2>::new_default([2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_new_default_invalid_dimensions() {
        Tensor::<u8, 2>::new_default([2, 0]);
    }

    #[test]
    fn test_from_slice() {
        let tensor = Tensor::from_slice([2, 3], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get(&[0, 0]), &1);
        assert_eq!(tensor.get(&[1, 2]), &6);
    }

    #[test]
    fn test_from_slice_zero_rank() {
        let tensor = Tensor::from_slice([], &[1]);
        assert_eq!(tensor.shape(), &[]);
        assert_eq!(tensor.get(&[]), &1);
    }

    #[test]
    #[should_panic]
    fn test_from_slice_zst() {
        Tensor::from_slice([1], &[()]);
    }

    #[test]
    #[should_panic]
    fn test_from_slice_invalid_shape() {
        Tensor::from_slice([2, 3], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_from_vec() {
        let tensor = Tensor::from_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get(&[0, 0]), &1);
        assert_eq!(tensor.get(&[1, 2]), &6);
    }

    #[test]
    fn test_from_vec_zero_rank() {
        let tensor = Tensor::from_vec([], vec![1]);
        assert_eq!(tensor.shape(), &[]);
        assert_eq!(tensor.get(&[]), &1);
    }

    #[test]
    #[should_panic]
    fn test_from_vec_zst() {
        Tensor::from_vec([1], vec![()]);
    }

    #[test]
    #[should_panic]
    fn test_from_vec_invalid_shape() {
        Tensor::from_vec([2, 3], vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_tensor_declarative_constructor() {
        use crate::tensor;

        let tensor = tensor![[
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-7.0, -8.0, -9.0], [-10.0, -11.0, -12.0]]
        ]];

        assert_eq!(tensor.shape(), &[2, 2, 3]);

        assert_eq!(tensor.get(&[0, 0, 0]), &-1.0);
        assert_eq!(tensor.get(&[0, 0, 1]), &-2.0);
        assert_eq!(tensor.get(&[0, 0, 2]), &-3.0);
        assert_eq!(tensor.get(&[0, 1, 0]), &-4.0);
        assert_eq!(tensor.get(&[0, 1, 1]), &-5.0);
        assert_eq!(tensor.get(&[0, 1, 2]), &-6.0);
        assert_eq!(tensor.get(&[1, 0, 0]), &-7.0);
        assert_eq!(tensor.get(&[1, 0, 1]), &-8.0);
        assert_eq!(tensor.get(&[1, 0, 2]), &-9.0);
        assert_eq!(tensor.get(&[1, 1, 0]), &-10.0);
        assert_eq!(tensor.get(&[1, 1, 1]), &-11.0);
        assert_eq!(tensor.get(&[1, 1, 2]), &-12.0);
    }
}
