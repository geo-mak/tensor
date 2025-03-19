use core::ops::Sub;

use crate::assertions::assert_same_shape;
use crate::core::alloc::UnsafeBufferPointer;
use crate::Tensor;

/// Subtracts `n` values of `b` from `a` and writes result to `r`.
#[inline(always)]
unsafe fn sub<T>(n: usize, a: *const T, b: *const T, r: *mut T)
where
    T: Copy + Sub<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        let b_i = *b.add(i);
        r.add(i).write(a_i - b_i);
        i += 1;
    }
}

impl<T, const R: usize> Sub<Self> for &Tensor<T, R>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs element-wise subtraction between `self` and `other` tensor and returns new
    /// `Tensor<T, R>` as a result of the subtraction without consuming either.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor1 = Tensor::new_set([2, 3], 5);
    /// let tensor2 = Tensor::new_set([2, 3], 3);
    ///
    /// let result = &tensor1 - &tensor2;
    ///
    /// assert_eq!(result.get(&[0, 0]), &2);
    /// assert_eq!(result.get(&[1, 2]), &2);
    /// ```
    fn sub(self, other: Self) -> Tensor<T, R> {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = self.data.raw_ptr();
        let b = other.data.raw_ptr();

        unsafe {
            let result = UnsafeBufferPointer::new_allocate(len);

            sub(len, a, b, result.raw_ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Sub<&Tensor<T, R>> for &mut Tensor<T, R>
where
    T: Copy + Sub<Output = T>,
{
    type Output = ();
    /// Performs in-place element-wise subtraction of another tensor from `self`.
    ///
    /// # Panics
    /// This method panics if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set([2, 3], 5);
    /// let tensor2 = Tensor::new_set([2, 3], 3);
    ///
    /// &mut tensor1 - &tensor2;
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &2);
    /// assert_eq!(tensor1.get(&[1, 2]), &2);
    /// ```
    fn sub(self, other: &Tensor<T, R>) {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = self.data.raw_ptr_mut();
        let b = other.data.raw_ptr();

        unsafe {
            sub(len, a, b, a);
        }
    }
}

/// Subtracts `n` count of `v` from `a` and writes result to `r`.
#[inline(always)]
unsafe fn sub_value<T>(n: usize, a: *const T, v: T, r: *mut T)
where
    T: Copy + Sub<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        r.add(i).write(a_i - v);
        i += 1;
    }
}

impl<T, const R: usize> Sub<T> for &Tensor<T, R>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs in-place element-wise subtraction `value` from `self`, and returns result as new
    /// `Tensor<T, R>` without affecting the original instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 3], 5);
    ///
    /// let result = &tensor - 3;
    ///
    /// assert_eq!(result.get(&[0, 0]), &2);
    /// assert_eq!(result.get(&[1, 2]), &2);
    fn sub(self, value: T) -> Tensor<T, R> {
        // len is assumed to be > 0.
        let len = self.metadata.size();
        let a = self.data.raw_ptr();

        unsafe {
            let result = UnsafeBufferPointer::new_allocate(len);

            sub_value(len, a, value, result.raw_ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Sub<T> for &mut Tensor<T, R>
where
    T: Copy + Sub<Output = T>,
{
    type Output = ();

    /// Performs in-place element-wise subtraction of `value` from `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 3], 5);
    ///
    /// &mut tensor - 3;
    ///
    /// assert_eq!(tensor.get(&[0, 0]), &2);
    /// assert_eq!(tensor.get(&[1, 2]), &2);
    fn sub(self, value: T) {
        // len is assumed to be > 0.
        let len = self.metadata.size();
        let a = self.data.raw_ptr_mut();

        unsafe { sub_value(len, a, value, a) }
    }
}

#[cfg(test)]
mod sub_tests {
    use super::*;

    #[test]
    fn test_sub_new() {
        let tensor1 = Tensor::new_set([2, 2], 5);
        let tensor2 = Tensor::new_set([2, 2], 3);

        let result = &tensor1 - &tensor2;

        assert_eq!(result.get(&[0, 0]), &2);
        assert_eq!(result.get(&[0, 1]), &2);
        assert_eq!(result.get(&[1, 0]), &2);
        assert_eq!(result.get(&[1, 1]), &2);
    }

    #[test]
    fn test_sub_mutate() {
        let mut tensor1 = Tensor::new_set([2, 2], 5);
        let tensor2 = Tensor::new_set([2, 2], 3);

        &mut tensor1 - &tensor2;

        assert_eq!(tensor1.as_slice(), &[2, 2, 2, 2]);
    }

    #[test]
    fn test_sub_value() {
        let mut tensor = Tensor::new_set([2, 2], 5);

        &mut tensor - 3;

        assert_eq!(tensor.get(&[0, 0]), &2);
        assert_eq!(tensor.get(&[0, 1]), &2);
        assert_eq!(tensor.get(&[1, 0]), &2);
        assert_eq!(tensor.get(&[1, 1]), &2);
    }

    #[test]
    fn test_sub_value_mut() {
        let tensor = Tensor::new_set([2, 2], 5);

        let result = &tensor - 3;

        assert_eq!(result.get(&[0, 0]), &2);
        assert_eq!(result.get(&[0, 1]), &2);
        assert_eq!(result.get(&[1, 0]), &2);
        assert_eq!(result.get(&[1, 1]), &2);
    }
}
