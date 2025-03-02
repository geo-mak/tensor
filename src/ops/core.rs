use crate::assertions::assert_same_dimensions;
use crate::Tensor;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

impl<T> Tensor<T>
where
    T: Copy + Add<Output = T>,
{
    /// Performs element-wise addition of two tensors.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Returns
    /// New `Tensor<T>` that contains the result of the element-wise addition.
    /// This new tensor will have the same dimensions and strides as `self` and `other`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor1 = Tensor::new_set(vec![2, 3], 1);
    /// let tensor2 = Tensor::new_set(vec![2, 3], 2);
    ///
    /// let result = tensor1.add(&tensor2);
    ///
    /// assert_eq!(result.get(&[0, 0]), &3);
    /// assert_eq!(result.get(&[1, 2]), &3);
    /// ```
    pub fn add(&self, other: &Tensor<T>) -> Self {
        assert_same_dimensions(self, other);
        let data: Vec<T> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a + *b)
            .collect();

        Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + Sub<Output = T>,
{
    /// Performs element-wise subtraction of two tensors.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Returns
    /// New `Tensor<T>` that contains the result of the element-wise subtraction.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor1 = Tensor::new_set(vec![2, 3], 5);
    /// let tensor2 = Tensor::new_set(vec![2, 3], 3);
    ///
    /// let result = tensor1.sub(&tensor2);
    ///
    /// assert_eq!(result.get(&[0, 0]), &2);
    /// assert_eq!(result.get(&[1, 2]), &2);
    /// ```
    pub fn sub(&self, other: &Tensor<T>) -> Tensor<T> {
        assert_same_dimensions(self, other);
        let data: Vec<T> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a - *b)
            .collect();

        Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T>,
{
    /// Performs element-wise multiplication of two tensors.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Returns
    /// New `Tensor<T>` that contains the result of the element-wise multiplication.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor1 = Tensor::new_set(vec![2, 3], 2);
    /// let tensor2 = Tensor::new_set(vec![2, 3], 3);
    ///
    /// let result = tensor1.mul(&tensor2);
    ///
    /// assert_eq!(result.get(&[0, 0]), &6);
    /// assert_eq!(result.get(&[1, 2]), &6);
    /// ```
    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T> {
        assert_same_dimensions(self, other);
        let data: Vec<T> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a * *b)
            .collect();

        Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + Div<Output = T>,
{
    /// Performs element-wise division of two tensors.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match. Additionally,
    /// it will panic if any element of `other` is zero, as division by zero is not allowed.
    ///
    /// # Returns
    /// New `Tensor<T>` that contains the result of the element-wise division.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor1 = Tensor::new_set(vec![2, 3], 6);
    /// let tensor2 = Tensor::new_set(vec![2, 3], 2);
    ///
    /// let result = tensor1.div(&tensor2);
    ///
    /// assert_eq!(result.get(&[0, 0]), &3);
    /// assert_eq!(result.get(&[1, 2]), &3);
    /// ```
    pub fn div(&self, other: &Tensor<T>) -> Tensor<T> {
        assert_same_dimensions(self, other);
        let data: Vec<T> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a / *b)
            .collect();

        Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + Neg<Output = T>,
{
    /// Performs element-wise negation of the tensor.
    ///
    /// # Returns
    /// New `Tensor<T>` where each element is the negated value of the corresponding element
    /// in `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor = Tensor::new_set(vec![2, 3], 1);
    ///
    /// let result = tensor.neg();
    ///
    /// assert_eq!(result.get(&[0, 0]), &-1);
    /// assert_eq!(result.get(&[1, 2]), &-1);
    /// ```
    pub fn neg(&self) -> Tensor<T> {
        let data: Vec<T> = self.data.iter().map(|a| -*a).collect();

        Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + AddAssign,
{
    /// Performs in-place element-wise addition of another tensor to `self`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///
    /// # Panics
    /// This method panics if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set(vec![2, 3], 1);
    /// let tensor2 = Tensor::new_set(vec![2, 3], 2);
    ///
    /// tensor1.add_mutate(&tensor2);
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &3);
    /// assert_eq!(tensor1.get(&[1, 2]), &3);
    /// ```
    pub fn add_mutate(&mut self, other: &Tensor<T>) {
        assert_same_dimensions(self, other);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += *b;
        }
    }
}

// Implement mutable subtraction for `Tensor`
impl<T> Tensor<T>
where
    T: Copy + SubAssign,
{
    /// Performs in-place element-wise subtraction of another tensor from `self`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///
    /// # Panics
    /// This method panics if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set(vec![2, 3], 5);
    /// let tensor2 = Tensor::new_set(vec![2, 3], 3);
    ///
    /// tensor1.sub_mutate(&tensor2);
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &2);
    /// assert_eq!(tensor1.get(&[1, 2]), &2);
    /// ```
    pub fn sub_mutate(&mut self, other: &Tensor<T>) {
        assert_same_dimensions(self, other);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a -= *b;
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + MulAssign,
{
    /// Performs in-place element-wise multiplication of another tensor with `self`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///
    /// # Panics
    /// This method panics if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set(vec![2, 3], 2);
    /// let tensor2 = Tensor::new_set(vec![2, 3], 3);
    ///
    /// tensor1.mul_mutate(&tensor2);
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &6);
    /// assert_eq!(tensor1.get(&[1, 2]), &6);
    /// ```
    pub fn mul_mutate(&mut self, other: &Tensor<T>) {
        assert_same_dimensions(self, other);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a *= *b;
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + DivAssign,
{
    /// Performs in-place element-wise division of `self` by another tensor.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match. Additionally,
    /// it will panic if any element of `other` is zero, as division by zero is not allowed.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor1 = Tensor::new_set(vec![2, 3], 6);
    /// let tensor2 = Tensor::new_set(vec![2, 3], 3);
    ///
    /// tensor1.div_mutate(&tensor2);
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &2);
    /// assert_eq!(tensor1.get(&[1, 2]), &2);
    /// ```
    pub fn div_mutate(&mut self, other: &Tensor<T>) {
        assert_same_dimensions(self, other);
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a /= *b;
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + Neg<Output = T>,
{
    /// Performs in-place negation of each element in the tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let mut tensor = Tensor::new_set(vec![2, 2], 5);
    ///
    /// tensor.neg_mutate();
    ///
    /// assert_eq!(tensor.get(&[0, 0]), &-5);
    /// assert_eq!(tensor.get(&[1, 1]), &-5);
    /// ```
    pub fn neg_mutate(&mut self) {
        for a in self.data.iter_mut() {
            *a = -*a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let tensor1 = Tensor::new_set(vec![2, 2], 1);
        let tensor2 = Tensor::new_set(vec![2, 2], 2);
        let result = tensor1.add(&tensor2);

        assert_eq!(result.get(&[0, 0]), &3);
        assert_eq!(result.get(&[0, 1]), &3);
        assert_eq!(result.get(&[1, 0]), &3);
        assert_eq!(result.get(&[1, 1]), &3);
    }

    #[test]
    fn test_sub() {
        let tensor1 = Tensor::new_set(vec![2, 2], 5);
        let tensor2 = Tensor::new_set(vec![2, 2], 3);
        let result = tensor1.sub(&tensor2);

        assert_eq!(result.get(&[0, 0]), &2);
        assert_eq!(result.get(&[0, 1]), &2);
        assert_eq!(result.get(&[1, 0]), &2);
        assert_eq!(result.get(&[1, 1]), &2);
    }

    #[test]
    fn test_mul() {
        let tensor1 = Tensor::new_set(vec![2, 2], 2);
        let tensor2 = Tensor::new_set(vec![2, 2], 3);
        let result = tensor1.mul(&tensor2);

        assert_eq!(result.get(&[0, 0]), &6);
        assert_eq!(result.get(&[0, 1]), &6);
        assert_eq!(result.get(&[1, 0]), &6);
        assert_eq!(result.get(&[1, 1]), &6);
    }

    #[test]
    fn test_div() {
        let tensor1 = Tensor::new_set(vec![2, 2], 6);
        let tensor2 = Tensor::new_set(vec![2, 2], 3);
        let result = tensor1.div(&tensor2);

        assert_eq!(result.get(&[0, 0]), &2);
        assert_eq!(result.get(&[0, 1]), &2);
        assert_eq!(result.get(&[1, 0]), &2);
        assert_eq!(result.get(&[1, 1]), &2);
    }

    #[test]
    fn test_neg() {
        let tensor = Tensor::new_set(vec![2, 2], 5);
        let result = tensor.neg();
        assert_eq!(result.data, vec![-5, -5, -5, -5]);
    }

    #[test]
    fn test_add_mutate() {
        let mut tensor1 = Tensor::new_set(vec![2, 2], 1);
        let tensor2 = Tensor::new_set(vec![2, 2], 2);
        tensor1.add_mutate(&tensor2);
        assert_eq!(tensor1.data, vec![3, 3, 3, 3]);
    }

    #[test]
    fn test_sub_mutate() {
        let mut tensor1 = Tensor::new_set(vec![2, 2], 5);
        let tensor2 = Tensor::new_set(vec![2, 2], 3);
        tensor1.sub_mutate(&tensor2);
        assert_eq!(tensor1.data, vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_mul_mutate() {
        let mut tensor1 = Tensor::new_set(vec![2, 2], 3);
        let tensor2 = Tensor::new_set(vec![2, 2], 4);
        tensor1.mul_mutate(&tensor2);
        assert_eq!(tensor1.data, vec![12, 12, 12, 12]);
    }

    #[test]
    fn test_div_mutate() {
        let mut tensor1 = Tensor::new_set(vec![2, 2], 8);
        let tensor2 = Tensor::new_set(vec![2, 2], 4);
        tensor1.div_mutate(&tensor2);
        assert_eq!(tensor1.data, vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_neg_mutate() {
        let mut tensor = Tensor::new_set(vec![2, 2], 5);
        tensor.neg_mutate();
        assert_eq!(tensor.data, vec![-5, -5, -5, -5]);
    }
}
