use core::ops::Add;

use crate::assertions::assert_same_shape;
use crate::core::alloc::MemorySpace;
use crate::Tensor;

/// Adds `n` values of `a` to `b` and writes result to `r`.
#[inline(always)]
unsafe fn add<T>(n: usize, a: *const T, b: *const T, r: *mut T)
where
    T: Copy + Add<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        let b_i = *b.add(i);
        r.add(i).write(a_i + b_i);
        i += 1;
    }
}

impl<T, const R: usize> Add<Self> for &Tensor<T, R>
where
    T: Copy + Add<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs element-wise addition between `self` and `other` tensor and returns new
    /// `Tensor<T, R>` as a result of the addition without consuming `self` or `other`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor1 = Tensor::new_set([2, 3], 1);
    /// let tensor2 = Tensor::new_set([2, 3], 2);
    ///
    /// let result = &tensor1 + &tensor2;
    ///
    /// assert_eq!(result.get(&[0, 0]), &3);
    /// assert_eq!(result.get(&[1, 2]), &3);
    /// ```
    fn add(self, other: Self) -> Tensor<T, R> {
        // Note: Broadcasting can be cheap to add here, because it writes to new buffer anyway.
        assert_same_shape(self, other);

        // len is assumed to be > 0.
        let len = self.metadata.size();
        let a = self.data.ptr();
        let b = other.data.ptr();

        unsafe {
            let result = MemorySpace::new_allocate(len);

            add(len, a, b, result.ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Add<&Tensor<T, R>> for &mut Tensor<T, R>
where
    T: Copy + Add<Output = T>,
{
    type Output = ();

    /// Performs in-place element-wise addition of another tensor to `self`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set([2, 3], 1);
    /// let tensor2 = Tensor::new_set([2, 3], 2);
    ///
    /// &mut tensor1 + &tensor2;
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &3);
    /// assert_eq!(tensor1.get(&[1, 2]), &3);
    fn add(self, other: &Tensor<T, R>) {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = self.data.ptr_mut();
        let b = other.data.ptr();

        unsafe {
            add(len, a, b, a);
        }
    }
}

/// Adds `n` count of `v` to `a`, and writes result to `r`.
#[inline(always)]
unsafe fn add_value<T>(n: usize, a: *const T, v: T, r: *mut T)
where
    T: Copy + Add<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        r.add(i).write(a_i + v);
        i += 1;
    }
}

impl<T, const R: usize> Add<T> for &Tensor<T, R>
where
    T: Copy + Add<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs element-wise addition of `value` to `self` and returns result as new
    /// `Tensor<T, R>`, without affecting the original instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 3], 1);
    ///
    /// let result = &tensor + 2;
    ///
    /// assert_eq!(result.get(&[0, 0]), &3);
    /// assert_eq!(result.get(&[1, 2]), &3);
    fn add(self, value: T) -> Tensor<T, R> {
        let len = self.metadata.size();
        let a = self.data.ptr();

        unsafe {
            let result = MemorySpace::new_allocate(len);

            add_value(len, a, value, result.ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Add<T> for &mut Tensor<T, R>
where
    T: Copy + Add<Output = T>,
{
    type Output = ();

    /// Performs in-place element-wise addition of `value` to `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 3], 1);
    ///
    /// &mut tensor + 2;
    ///
    /// assert_eq!(tensor.get(&[0, 0]), &3);
    /// assert_eq!(tensor.get(&[1, 2]), &3);
    fn add(self, value: T) {
        let len = self.metadata.size();
        let a = self.data.ptr_mut();

        unsafe {
            add_value(len, a, value, a);
        }
    }
}

#[cfg(test)]
mod add_tests {
    use super::*;

    #[test]
    fn test_add_new() {
        let tensor1 = Tensor::new_set([2, 2], 1);
        let tensor2 = Tensor::new_set([2, 2], 2);

        let result = &tensor1 + &tensor2;

        assert_eq!(result.get(&[0, 0]), &3);
        assert_eq!(result.get(&[0, 1]), &3);
        assert_eq!(result.get(&[1, 0]), &3);
        assert_eq!(result.get(&[1, 1]), &3);
    }

    #[test]
    fn test_add_mutate() {
        let mut tensor1 = Tensor::new_set([2, 2], 1);
        let tensor2 = Tensor::new_set([2, 2], 2);

        &mut tensor1 + &tensor2;

        assert_eq!(tensor1.get(&[0, 0]), &3);
        assert_eq!(tensor1.get(&[0, 1]), &3);
        assert_eq!(tensor1.get(&[1, 0]), &3);
        assert_eq!(tensor1.get(&[1, 1]), &3);
    }

    #[test]
    fn test_add_value() {
        let tensor = Tensor::new_set([2, 2], 1);

        let result = &tensor + 2;

        assert_eq!(result.get(&[0, 0]), &3);
        assert_eq!(result.get(&[0, 1]), &3);
        assert_eq!(result.get(&[1, 0]), &3);
        assert_eq!(result.get(&[1, 1]), &3);
    }

    #[test]
    fn test_add_value_mut() {
        let mut tensor = Tensor::new_set([2, 2], 1);

        &mut tensor + 2;

        assert_eq!(tensor.get(&[0, 0]), &3);
        assert_eq!(tensor.get(&[0, 1]), &3);
        assert_eq!(tensor.get(&[1, 0]), &3);
        assert_eq!(tensor.get(&[1, 1]), &3);
    }
}
