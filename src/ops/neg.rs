use core::ops::Neg;

use crate::core::alloc::UnsafeBufferPointer;
use crate::Tensor;

/// Negates `n` values of `a`, and writes result to `r`.
#[inline(always)]
unsafe fn neg<T>(n: usize, a: *const T, r: *mut T)
where
    T: Copy + Neg<Output = T>,
{
    let mut i = 0;
    while i < n {
        let a_i = *a.add(i);
        r.add(i).write(-a_i);
        i += 1;
    }
}

impl<T, const R: usize> Neg for &Tensor<T, R>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Tensor<T, R>;

    /// Performs element-wise negation of the tensor and returns new `Tensor<T, R>` as a result of
    /// negation without consuming `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor = Tensor::new_set([2, 3], 1);
    ///
    /// let result = -&tensor;
    ///
    /// assert_eq!(result.get(&[0, 0]), &-1);
    /// assert_eq!(result.get(&[1, 2]), &-1);
    /// ```
    fn neg(self) -> Self::Output {
        // len is assumed to be > 0.
        let len = self.metadata.size();
        let a = self.data.raw_ptr();
        unsafe {
            let result = UnsafeBufferPointer::new_allocate(len);

            neg(len, a, result.raw_ptr_mut());

            Tensor {
                metadata: self.metadata,
                data: result,
            }
        }
    }
}

impl<T, const R: usize> Neg for &mut Tensor<T, R>
where
    T: Copy + Neg<Output = T>,
{
    type Output = ();

    /// Performs in-place negation of each element in the tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set([2, 2], 5);
    ///
    /// -&mut tensor;
    ///
    /// assert_eq!(tensor.get(&[0, 0]), &-5);
    /// assert_eq!(tensor.get(&[1, 1]), &-5);
    /// ```
    fn neg(self) {
        let len = self.metadata.size();
        let a = self.data.raw_ptr_mut();
        unsafe { neg(len, a, a) }
    }
}

#[cfg(test)]
mod neg_tests {
    use super::*;

    #[test]
    fn test_neg_new() {
        let tensor = Tensor::new_set([2, 2], 5);

        let result = -&tensor;

        assert_eq!(result.as_slice(), &[-5, -5, -5, -5]);
    }

    #[test]
    fn test_neg_mutate() {
        let mut tensor = Tensor::new_set([2, 2], 5);

        -&mut tensor;

        assert_eq!(tensor.as_slice(), &[-5, -5, -5, -5]);
    }
}
