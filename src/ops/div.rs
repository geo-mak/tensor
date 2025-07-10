use core::ops::Div;

use crate::assertions::assert_same_shape;
use crate::core::alloc::MemorySpace;
use crate::Tensor;

/// Divides `n` values of `a` by `b` and writes result to `r`.
#[inline(always)]
unsafe fn div<T>(n: usize, a: *const T, b: *const T, r: *mut T)
where
    T: Copy + Div<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        let b_i = *b.add(i);
        r.add(i).write(a_i / b_i);
        i += 1;
    }
}

impl<T, const R: usize> Div<Self> for &Tensor<T, R>
where
    T: Copy + Div<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs element-wise division between `self` and `other` tensor and returns new
    /// `Tensor<T, R>` as a result of the division without consuming `self` or `other`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match. Additionally,
    /// it will panic if any element of `other` is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor1 = Tensor::new_set([2, 3], 6);
    /// let tensor2 = Tensor::new_set([2, 3], 2);
    ///
    /// let result = &tensor1 / &tensor2;
    ///
    /// assert_eq!(result.get(&[0, 0]), &3);
    /// assert_eq!(result.get(&[1, 2]), &3);
    /// ```
    fn div(self, other: Self) -> Tensor<T, R> {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = self.data.ptr();
        let b = other.data.ptr();

        unsafe {
            let result = MemorySpace::new_allocate(len);

            div(len, a, b, result.ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Div<&Tensor<T, R>> for &mut Tensor<T, R>
where
    T: Copy + Div<Output = T>,
{
    type Output = ();

    /// Performs in-place element-wise division of `self` by another tensor.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match. Additionally,
    /// it will panic if any element of `other` is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set([2, 3], 6);
    /// let tensor2 = Tensor::new_set([2, 3], 3);
    ///
    /// &mut tensor1 / &tensor2;
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &2);
    /// assert_eq!(tensor1.get(&[1, 2]), &2);
    /// ```
    fn div(self, other: &Tensor<T, R>) -> Self::Output {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = self.data.ptr_mut();
        let b = other.data.ptr();

        unsafe {
            div(len, a, b, a);
        }
    }
}

/// Divides `n` count of `a` by `v`, and writes result to `r`.
#[inline(always)]
unsafe fn div_value<T>(n: usize, a: *const T, v: T, r: *mut T)
where
    T: Copy + Div<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        r.add(i).write(a_i / v);
        i += 1;
    }
}

impl<T, const R: usize> Div<T> for &Tensor<T, R>
where
    T: Copy + Div<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs in-place element-wise division of `self` by `value`, and returns result as new
    /// `Tensor<T, R>` without affecting the original instance.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 3], 6);
    ///
    /// let result =  &tensor / 2;
    ///
    /// assert_eq!(result.get(&[0, 0]), &3);
    /// assert_eq!(result.get(&[1, 2]), &3);
    /// ```
    fn div(self, value: T) -> Tensor<T, R> {
        // len is assumed to be > 0.
        let len = self.metadata.size();
        let a = self.data.ptr();

        unsafe {
            let result = MemorySpace::new_allocate(len);

            div_value(len, a, value, result.ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Div<T> for &mut Tensor<T, R>
where
    T: Copy + Div<Output = T>,
{
    type Output = ();

    /// Performs in-place element-wise division of `self` by `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 3], 6);
    ///
    /// &mut tensor / 2;
    ///
    /// assert_eq!(tensor.get(&[0, 0]), &3);
    /// assert_eq!(tensor.get(&[1, 2]), &3);
    /// ```
    fn div(self, value: T) {
        let len = self.metadata.size();
        let a = self.data.ptr_mut();

        unsafe {
            div_value(len, a, value, a);
        }
    }
}

#[cfg(test)]
mod div_tests {
    use super::*;

    #[test]
    fn test_div_new() {
        let tensor1 = Tensor::new_set([2, 2], 6);
        let tensor2 = Tensor::new_set([2, 2], 3);

        let result = &tensor1 / &tensor2;

        assert_eq!(result.get(&[0, 0]), &2);
        assert_eq!(result.get(&[0, 1]), &2);
        assert_eq!(result.get(&[1, 0]), &2);
        assert_eq!(result.get(&[1, 1]), &2);
    }

    #[test]
    fn test_div_mutate() {
        let mut tensor1 = Tensor::new_set([2, 2], 8);
        let tensor2 = Tensor::new_set([2, 2], 4);

        &mut tensor1 / &tensor2;

        assert_eq!(tensor1.as_slice(), &[2, 2, 2, 2]);
    }

    #[test]
    fn test_div_value() {
        let mut tensor = Tensor::new_set([2, 2], 6);

        &mut tensor / 3;

        assert_eq!(tensor.get(&[0, 0]), &2);
        assert_eq!(tensor.get(&[0, 1]), &2);
        assert_eq!(tensor.get(&[1, 0]), &2);
        assert_eq!(tensor.get(&[1, 1]), &2);
    }

    #[test]
    fn test_div_value_mut() {
        let tensor = Tensor::new_set([2, 2], 6);

        let result = &tensor / 3;

        assert_eq!(result.get(&[0, 0]), &2);
        assert_eq!(result.get(&[0, 1]), &2);
        assert_eq!(result.get(&[1, 0]), &2);
        assert_eq!(result.get(&[1, 1]), &2);
    }
}
