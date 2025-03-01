use crate::assertions::assert_same_dimensions;
use crate::tensor::Tensor;
use core::ops::{Add, Mul, Sub};

// -------------------------------------------------------------------
// This module contains the implementation of similarity metrics.
// The following similarity metrics are implemented:
// - dot_product: Dot product of two tensors.
// - cosine_similarity: Cosine similarity between two tensors.
// - euclidean_distance: Euclidean distance between two tensors.
// -------------------------------------------------------------------

impl<T> Tensor<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default,
{
    /// Computes the dot product of two tensors.
    ///
    /// # Formula
    /// The dot product of two vectors `a` and `b` is defined as:
    /// `a · b = a1 * b1 + a2 * b2 + ... + an * bn`
    /// where `a1, a2, ..., an` and `b1, b2, ..., bn` are the elements of the two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to another tensor.
    ///
    /// # Returns
    ///
    /// The dot product as a value of type `T`.
    pub fn dot_product(&self, other: &Tensor<T>) -> T {
        assert_same_dimensions(self, other);
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a * *b)
            .fold(T::default(), |acc, x| acc + x)
    }
}

impl<T> Tensor<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Into<f64>,
{
    /// Computes the cosine similarity between two tensors.
    ///
    /// # Formula
    /// The cosine similarity between two vectors `a` and `b` is defined as:
    /// `cosine_similarity = (a · b) / (||a|| * ||b||)`
    /// where `a · b` is the dot product of the two vectors and `||a||` and `||b||` are the magnitudes of the two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to another tensor.
    ///
    /// # Returns
    ///
    /// The cosine similarity as a `f64` value.
    pub fn cosine_similarity(&self, other: &Tensor<T>) -> f64 {
        assert_same_dimensions(self, other);
        // Dot product of the two tensors
        let dot_product: f64 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (*a * *b).into())
            .sum();

        // Euclidean norm of the first tensor
        let magnitude1: f64 = self
            .data
            .iter()
            .map(|a| (*a * *a).into())
            .sum::<f64>()
            .sqrt();

        // Euclidean norm of the second tensor
        let magnitude2: f64 = other
            .data
            .iter()
            .map(|a| (*a * *a).into())
            .sum::<f64>()
            .sqrt();

        // Avoiding division by zero
        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            0.0
        } else {
            dot_product / (magnitude1 * magnitude2)
        }
    }
}

impl<T> Tensor<T>
where
    T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy + Into<f64>,
{
    /// Computes the Euclidean distance between two tensors.
    ///
    /// # Formula
    /// The Euclidean distance between two vectors `a` and `b` is defined as:
    /// `euclidean_distance = sqrt((a1 - b1)^2 + (a2 - b2)^2 + ... + (an - bn)^2)`
    /// where `a1, a2, ..., an` and `b1, b2, ..., bn` are the elements of the two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - A reference to another tensor.
    ///
    /// # Returns
    ///
    /// The Euclidean distance as a `f64` value.
    pub fn euclidean_distance(&self, other: &Tensor<T>) -> f64 {
        assert_same_dimensions(self, other);
        // Squared difference between the two tensors
        let squared_diff: f64 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| ((*a - *b) * (*a - *b)).into())
            .sum();

        // Square root of the sum of squared differences
        squared_diff.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_same() {
        let tensor1 = Tensor::with_values(vec![1.0, 1.0, 0.0], vec![1, 3]);
        let tensor2 = Tensor::with_values(vec![1.0, 1.0, 0.0], vec![1, 3]);
        let result = tensor1.cosine_similarity(&tensor2);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_different() {
        let tensor1 = Tensor::with_values(vec![1.0, 1.0, 0.0], vec![1, 3]);
        let tensor2 = Tensor::with_values(vec![1.0, 0.0, 0.0], vec![1, 3]);
        let result = tensor1.cosine_similarity(&tensor2);
        assert!((result - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_same() {
        let tensor1 = Tensor::new_set(vec![1, 3], 3.0);
        let tensor2 = Tensor::new_set(vec![1, 3], 3.0);
        let result = tensor1.euclidean_distance(&tensor2);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_euclidean_distance_different() {
        let tensor1 = Tensor::new_set(vec![1, 3], 3.0);
        let tensor2 = Tensor::new_set(vec![1, 3], 2.0);
        let result = tensor1.euclidean_distance(&tensor2);
        assert!((result - 1.7320508075688772).abs() < 1e-6);
    }
    
    #[test]
    fn test_dot_product() {
        let tensor1 = Tensor::new_set(vec![1, 3], 3.0);
        let tensor2 = Tensor::new_set(vec![1, 3], 2.0);
        let result = tensor1.dot_product(&tensor2);
        assert_eq!(result , 18.0);
    }
}
