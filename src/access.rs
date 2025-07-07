use core::ops::{Index, IndexMut};
use core::slice::{Iter, IterMut};

use crate::Tensor;

impl<T, const R: usize> Tensor<T, R> {
    /// Sets the value at the specified multidimensional indices.
    ///
    /// # Parameters
    ///
    /// - `index`: A coordinates' slice specifying the position in each dimension.
    /// - `value`: The value to set at the specified indices.
    ///
    /// # Panics
    /// This method will panic if any of the indices are out of bounds
    #[inline]
    pub const fn set(&mut self, index: &[usize; R], value: T) {
        let offset = self.metadata.offset(index.as_ptr());
        unsafe {
            self.data.store(offset, value);
        };
    }

    /// Returns a reference to the value at the specified multidimensional index.
    ///
    /// # Parameters
    ///
    /// - `index`: A coordinates' slice specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if any of the indices are out of bounds.
    #[must_use]
    #[inline]
    pub const fn get(&self, index: &[usize; R]) -> &T {
        let offset = self.metadata.offset(index.as_ptr());
        unsafe { self.data.access(offset) }
    }

    /// Returns the shape (dimensions) of the tensor.
    #[inline]
    pub const fn shape(&self) -> &[usize] {
        self.metadata.shape()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor = Tensor::new_set([2, 3, 4], 0.0);
    ///
    /// assert_eq!(tensor.size(), 24);
    /// ```
    #[inline]
    pub const fn size(&self) -> usize {
        self.metadata.size()
    }

    /// Returns the number of elements along a specific dimension.
    ///
    /// # Parameters
    ///
    /// - `index`: The index of the dimension (0-based) for which to get the size.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    /// let tensor = Tensor::new_set([2, 3], 0.0);
    ///
    /// assert_eq!(tensor.dim_size(0), Some(2));
    /// assert_eq!(tensor.dim_size(1), Some(3));
    /// assert_eq!(tensor.dim_size(2), None);
    /// ```
    #[inline]
    pub const fn dim_size(&self, index: usize) -> Option<usize> {
        if index < R {
            return Some(unsafe { *self.metadata.as_ptr().add(index) });
        }
        None
    }

    /// Returns an immutable flattened slice of the values in the tensor.
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { self.data.as_slice(self.metadata.size()) }
    }

    /// Returns a mutable flattened slice of the values in the tensor.
    #[inline]
    pub const fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { self.data.as_slice_mut(self.metadata.size()) }
    }

    /// Returns an iterator over a flattened slice of the values of the tensor.
    #[inline]
    pub fn iter(&self) -> Iter<T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over a flattened slice of the values of the tensor.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.as_slice_mut().iter_mut()
    }
}

impl<T, const R: usize> Index<&[usize; R]> for Tensor<T, R> {
    type Output = T;

    /// Returns a reference to the value at the specified multidimensional index.
    ///
    /// # Parameters
    ///
    /// - `index`: A coordinates' slice specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    #[inline]
    fn index(&self, index: &[usize; R]) -> &Self::Output {
        let offset = self.metadata.offset(index.as_ptr());
        unsafe { self.data.access(offset) }
    }
}

impl<T, const R: usize> IndexMut<&[usize; R]> for Tensor<T, R> {
    /// Returns a mutable reference to the value at the specified multidimensional index.
    ///
    /// # Parameters
    ///
    /// - `index`: A coordinates' slice specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if index is out of bounds.
    #[inline]
    fn index_mut(&mut self, index: &[usize; R]) -> &mut Self::Output {
        let offset = self.metadata.offset(index.as_ptr());
        unsafe { self.data.access_mut(offset) }
    }
}

#[cfg(test)]
mod access_tests {
    use super::*;

    #[test]
    fn test_set_get() {
        let mut tensor = Tensor::new_set([2, 3], 0);
        tensor.set(&[0, 2], 100);
        assert_eq!(tensor.get(&[0, 2]), &100);
    }

    #[test]
    #[should_panic]
    fn test_get_out_of_bounds() {
        let tensor = Tensor::new_set([2, 3], 0);
        let _ = tensor.get(&[2, 2]);
    }

    #[test]
    fn test_size() {
        let tensor = Tensor::new_set([2, 3, 4], 0.0);
        assert_eq!(tensor.size(), 24);
    }

    #[test]
    fn test_dim_size() {
        let tensor = Tensor::new_set([2, 3], 0.0);
        assert_eq!(tensor.dim_size(0), Some(2));
        assert_eq!(tensor.dim_size(1), Some(3));
        assert_eq!(tensor.dim_size(2), None);
    }

    #[test]
    fn test_tensor_access_index() {
        let mut tensor = Tensor::new_set([2, 3], 0);
        tensor[&[0, 2]] = 100;
        assert_eq!(tensor[&[0, 2]], 100);
    }
}
