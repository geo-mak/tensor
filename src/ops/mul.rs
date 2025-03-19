use core::ops::Mul;

use crate::assertions::assert_same_shape;
use crate::core::alloc::UnsafeBufferPointer;
use crate::Tensor;

/// Multiplies `n` values of `a` with `b` and writes result to `r`.
#[inline(always)]
unsafe fn mul<T>(n: usize, a: *const T, b: *const T, r: *mut T)
where
    T: Copy + Mul<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        let b_i = *b.add(i);
        r.add(i).write(a_i * b_i);
        i += 1;
    }
}

impl<T, const R: usize> Mul<Self> for &Tensor<T, R>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs element-wise multiplication between `self` and `other` tensor and returns new
    /// `Tensor<T, R>` as a result of the multiplication without consuming `self` or `other`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set([2, 3], 2);
    /// let tensor2 = Tensor::new_set([2, 3], 3);
    ///
    /// let result = &tensor1 * &tensor2;
    ///
    /// assert_eq!(result.get(&[0, 0]), &6);
    /// assert_eq!(result.get(&[1, 2]), &6);
    /// ```
    fn mul(self, other: Self) -> Tensor<T, R> {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = self.data.raw_ptr();
        let b = other.data.raw_ptr();

        unsafe {
            let result = UnsafeBufferPointer::new_allocate(len);

            mul(len, a, b, result.raw_ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Mul<&Tensor<T, R>> for &mut Tensor<T, R>
where
    T: Copy + Mul<Output = T>,
{
    type Output = ();
    /// Performs in-place element-wise multiplication of another tensor with `self`.
    ///
    /// # Panics
    /// This method panics if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set([2, 3], 2);
    /// let tensor2 = Tensor::new_set([2, 3], 3);
    ///
    /// &mut tensor1 * &tensor2;
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &6);
    /// assert_eq!(tensor1.get(&[1, 2]), &6);
    /// ```
    fn mul(self, other: &Tensor<T, R>) {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = self.data.raw_ptr_mut();
        let b = other.data.raw_ptr();

        unsafe {
            mul(len, a, b, a);
        }
    }
}

/// Multiplies `n` count of `a` by `v`, and writes result to `r`.
#[inline(always)]
unsafe fn mul_value<T>(n: usize, a: *const T, v: T, r: *mut T)
where
    T: Copy + Mul<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        r.add(i).write(a_i * v);
        i += 1;
    }
}

impl<T, const R: usize> Mul<T> for &Tensor<T, R>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs in-place element-wise multiplication of `self` by `value`, and returns result as
    /// new `Tensor<T, R>` without affecting the original instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 3], 2);
    ///
    /// &mut tensor * 2;
    ///
    /// assert_eq!(tensor.get(&[0, 0]), &4);
    /// assert_eq!(tensor.get(&[1, 2]), &4);
    /// ```
    fn mul(self, value: T) -> Tensor<T, R> {
        // len is assumed to be > 0.
        let len = self.metadata.size();
        let a = self.data.raw_ptr();

        unsafe {
            let result = UnsafeBufferPointer::new_allocate(len);

            mul_value(len, a, value, result.raw_ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Mul<T> for &mut Tensor<T, R>
where
    T: Copy + Mul<Output = T>,
{
    type Output = ();

    /// Performs in-place element-wise multiplication of `self` by `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 3], 2);
    ///
    /// let result = &tensor * 2;
    ///
    /// assert_eq!(result.get(&[0, 0]), &4);
    /// assert_eq!(result.get(&[1, 2]), &4);
    /// ```
    fn mul(self, value: T) -> Self::Output {
        // len is assumed to be > 0.
        let len = self.metadata.size();
        let a = self.data.raw_ptr_mut();

        unsafe {
            mul_value(len, a, value, a);
        }
    }
}

#[cfg(test)]
mod mul_tests {
    use super::*;

    #[test]
    fn test_mul_new() {
        let tensor1 = Tensor::new_set([2, 2], 2);
        let tensor2 = Tensor::new_set([2, 2], 3);

        let result = &tensor1 * &tensor2;

        assert_eq!(result.get(&[0, 0]), &6);
        assert_eq!(result.get(&[0, 1]), &6);
        assert_eq!(result.get(&[1, 0]), &6);
        assert_eq!(result.get(&[1, 1]), &6);
    }

    #[test]
    fn test_mul_mutate() {
        let mut tensor1 = Tensor::new_set([2, 2], 2);
        let tensor2 = Tensor::new_set([2, 2], 3);

        &mut tensor1 * &tensor2;

        assert_eq!(tensor1.get(&[0, 0]), &6);
        assert_eq!(tensor1.get(&[0, 1]), &6);
        assert_eq!(tensor1.get(&[1, 0]), &6);
        assert_eq!(tensor1.get(&[1, 1]), &6);
    }

    #[test]
    fn test_mul_value() {
        let mut tensor = Tensor::new_set([2, 2], 2);

        &mut tensor * 3;

        assert_eq!(tensor.get(&[0, 0]), &6);
        assert_eq!(tensor.get(&[0, 1]), &6);
        assert_eq!(tensor.get(&[1, 0]), &6);
        assert_eq!(tensor.get(&[1, 1]), &6);
    }

    #[test]
    fn test_mul_value_mut() {
        let tensor = Tensor::new_set([2, 2], 2);

        let result = &tensor * 3;

        assert_eq!(result.get(&[0, 0]), &6);
        assert_eq!(result.get(&[0, 1]), &6);
        assert_eq!(result.get(&[1, 0]), &6);
        assert_eq!(result.get(&[1, 1]), &6);
    }
}
