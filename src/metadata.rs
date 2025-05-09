use crate::assertions::{assert_non_zero_size, assert_same_size};
use core::fmt::Debug;

/// `TensorMetaData` stores information about dimensions and size of the tensor, and it is
/// responsible for indexing the values in the data buffer.
///
/// `TensorMetaData` uses C-style "row-major" memory ordering for indexing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TensorMetaData<const R: usize> {
    dims: [usize; R],
    strides: [usize; R],
    size: usize,
}

impl<const R: usize> TensorMetaData<R> {
    /// Creates new instance of `TensorMetaData`.
    ///
    /// This function computes the size of the data buffer from the product of the provided
    /// dimensions.
    ///
    /// This function will panic if the size of the provided dimensions is `0`.
    #[must_use]
    #[inline]
    pub(crate) const fn new(dims: [usize; R]) -> Self {
        let (size, strides) = Self::compute(dims.as_ptr());
        assert_non_zero_size(size);

        TensorMetaData {
            dims,
            strides,
            size,
        }
    }

    /// Creates new instance of `TensorMetaData`.
    ///
    /// This function compares the size of the data buffer with the size of the provided
    /// dimensions.
    ///
    /// This function will panic if the size of the provided dimensions and the provided size `n`
    /// don't match.
    #[must_use]
    #[inline]
    pub(crate) const fn new_cmp_eq(n: usize, dims: [usize; R]) -> Self {
        let (new_size, strides) = Self::compute(dims.as_ptr());
        assert_same_size(n, new_size);

        TensorMetaData {
            dims,
            strides,
            size: n,
        }
    }

    /// Returns the pointer of the dimensions' array.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn as_ptr(&self) -> *const usize {
        self.dims.as_ptr()
    }

    /// Returns the length of the tensor.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn size(&self) -> usize {
        self.size
    }

    /// Returns the dimensions of the tensor.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn shape(&self) -> &[usize] {
        &self.dims
    }

    /// Sets the new dimensions and computes their strides.
    ///
    /// This method will panic if current size does not match the product of the dimensions.
    #[inline]
    pub(crate) const fn reshape(&mut self, dims: [usize; R]) {
        let (size, strides) = Self::compute(dims.as_ptr());

        assert_same_size(self.size, size);

        // Update
        self.size = size;
        self.strides = strides;
        self.dims = dims;
    }

    /// Computes the strides and size of the provided dimensions and returns them.
    ///
    /// Returns `1` as size if `R` is `0`.
    #[must_use]
    const fn compute(dims: *const usize) -> (usize, [usize; R]) {
        let mut strides = [0; R];
        let strides_ptr = strides.as_mut_ptr();

        unsafe {
            let mut size = 1;
            let mut stride = 1;
            let mut i = R;
            while i != 0 {
                i -= 1;
                strides_ptr.add(i).write(stride);
                let dim = *dims.add(i);
                stride *= dim;
                size *= dim;
            }
            (size, strides)
        }
    }

    /// Computes and returns the linear index of an item in the data buffer.
    ///
    /// This method will panic if the index is out of bounds.
    #[must_use]
    #[inline]
    pub(crate) const fn offset(&self, index: *const usize) -> usize {
        let dims_ptr = self.dims.as_ptr();
        let strides_ptr = self.strides.as_ptr();

        unsafe {
            let mut offset = 0;
            let mut i = R;
            while i != 0 {
                i -= 1;
                let dim = *dims_ptr.add(i);
                let idx = *index.add(i);
                if idx < dim {
                    offset += idx * *strides_ptr.add(i);
                    continue;
                };
                panic!("Index out of bounds");
            }
            offset
        }
    }

    /// Compares the `dimensions` of two instances element-wise from right to left.
    #[must_use]
    pub(crate) const fn cmp_dims_eq(&self, other: &Self) -> bool {
        let self_dims = self.dims.as_ptr();
        let other_dims = other.dims.as_ptr();

        unsafe {
            let mut i = R;
            while i != 0 {
                i -= 1;
                if *self_dims.add(i) != *other_dims.add(i) {
                    return false;
                }
            }
            true
        }
    }

    /// Compares two instances by comparing the size first, if it is the same, it compares the
    /// dimensions element-wise from right to left.
    #[must_use]
    #[inline]
    pub(crate) const fn cmp_eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        self.cmp_dims_eq(other)
    }
}
