use core::ops::{AddAssign, Mul, Sub};

use crate::assertions::assert_same_shape;
use crate::tensor::Tensor;

impl<T, const R: usize> Tensor<T, R> {
    /// Returns the dot product of two tensors.
    pub fn dot_product(&self, other: &Self) -> T
    where
        T: Copy + Default + AddAssign<T> + Mul<Output = T>,
    {
        assert_same_shape(self, other);

        // len is assumed to be > 0.
        let len = self.metadata.size();
        let a = &self.data;
        let b = &other.data;

        let mut product = T::default();

        unsafe {
            let mut i = 0;
            while i < len {
                let a_i = *a.load(i);
                let b_i = *b.load(i);
                product += a_i * b_i;
                i += 1;
            }
        }

        product
    }

    /// Returns the cosine similarity between two tensors.
    pub fn cosine_similarity(&self, other: &Self) -> f64
    where
        T: Copy + Default + Into<f64> + AddAssign<T> + Mul<Output = T>,
    {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = &self.data;
        let b = &other.data;

        let mut product_a_b: T = T::default();
        let mut sum_exp_a: T = T::default();
        let mut sum_exp_b: T = T::default();

        unsafe {
            let mut i = 0;
            while i < len {
                let a_i = *a.load(i);
                let b_i = *b.load(i);
                product_a_b += a_i * b_i;
                sum_exp_a += a_i * a_i;
                sum_exp_b += b_i * b_i;
                i += 1;
            }
        }

        let e_norm_a: f64 = sum_exp_a.into().sqrt();
        let e_norm_b: f64 = sum_exp_b.into().sqrt();

        if e_norm_a == 0.0 || e_norm_b == 0.0 {
            0.0
        } else {
            // Negative values are NaN for square root, and all ops with NaN return NaN.
            product_a_b.into() / (e_norm_a * e_norm_b)
        }
    }

    /// Returns the Euclidean distance between two tensors.
    pub fn euclidean_distance(&self, other: &Self) -> f64
    where
        T: Copy + Default + Into<f64> + AddAssign<T> + Mul<Output = T> + Sub<Output = T>,
    {
        assert_same_shape(self, other);

        let len = self.metadata.size();
        let a = &self.data;
        let b = &other.data;

        let mut sum: T = T::default();

        unsafe {
            let mut i = 0;
            while i < len {
                let a_i = *a.load(i);
                let b_i = *b.load(i);
                let delta = a_i - b_i;
                sum += delta * delta;
                i += 1;
            }
        }

        sum.into().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_similar() {
        let tensor1 = Tensor::from_slice([1, 3], &[1.0, 1.0, 0.0]);
        let tensor2 = Tensor::from_slice([1, 3], &[1.0, 1.0, 0.0]);

        let result = tensor1.cosine_similarity(&tensor2);

        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different() {
        let tensor1 = Tensor::from_slice([1, 3], &[1.0, 1.0, 0.0]);
        let tensor2 = Tensor::from_slice([1, 3], &[1.0, 0.0, 0.0]);

        let result = tensor1.cosine_similarity(&tensor2);

        assert!((result - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_same() {
        let tensor1 = Tensor::new_set([1, 3], 3.0);
        let tensor2 = Tensor::new_set([1, 3], 3.0);

        let result = tensor1.euclidean_distance(&tensor2);

        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_euclidean_distance_different() {
        let tensor1 = Tensor::new_set([1, 3], 3.0);
        let tensor2 = Tensor::new_set([1, 3], 2.0);

        let result = tensor1.euclidean_distance(&tensor2);

        assert!((result - 1.7320508075688772).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let tensor1 = Tensor::new_set([1, 3], 3.0);
        let tensor2 = Tensor::new_set([1, 3], 2.0);

        let result = tensor1.dot_product(&tensor2);

        assert_eq!(result, 18.0);
    }
}
