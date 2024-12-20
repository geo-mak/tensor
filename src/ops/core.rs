use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::Tensor;

//////////////////////////////////////////////////////////////////////
// Immutable operations for `Tensor`
//////////////////////////////////////////////////////////////////////

// Implement addition for Tensor
impl<T> Tensor<T>
where
    T: Copy + Add<Output = T>,
{
    /// Performs element-wise addition of two tensors.
    ///
    /// This method adds two tensors of the same dimensions, element by element.
    /// The resulting tensor will have the same dimensions and strides as the original tensors,
    /// with each element being the sum of the corresponding elements from the two tensors.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///   The `other` tensor will be added to `self`.
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
    /// // Create two tensors with dimensions 2x3, initialized with values 1 and 2 respectively.
    /// let tensor1 = Tensor::new(vec![2, 3], 1);
    /// let tensor2 = Tensor::new(vec![2, 3], 2);
    ///
    /// // Perform element-wise addition of tensor1 and tensor2.
    /// let result = tensor1.add(&tensor2);
    ///
    /// assert_eq!(result.get(&[0, 0]), &3); // 1 + 2
    /// assert_eq!(result.get(&[1, 2]), &3); // 1 + 2
    /// ```
    pub fn add(&self, other: &Tensor<T>) -> Self {
        if self.dimensions != other.dimensions {
            panic!("Tensors must have the same dimensions for addition");
        }

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

// Implement subtraction for `Tensor`
impl<T> Tensor<T>
where
    T: Copy + Sub<Output = T>,
{
    /// Performs element-wise subtraction of two tensors.
    ///
    /// This method subtracts the elements of one tensor from the corresponding elements of another
    /// tensor. Both tensors must have the same dimensions for the subtraction to be performed.
    /// The resulting tensor will have the same dimensions and strides as the original tensors,
    /// with each element being the result of subtracting the corresponding elements of the `other`
    /// tensor from `self`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///   The elements of `self` will be subtracted by the elements of `other`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Returns
    /// New `Tensor<T>` that contains the result of the element-wise subtraction.
    /// The new tensor will have the same dimensions and strides as `self` and `other`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create two tensors with dimensions 2x3, initialized with values 5 and 3 respectively.
    /// let tensor1 = Tensor::new(vec![2, 3], 5);
    /// let tensor2 = Tensor::new(vec![2, 3], 3);
    ///
    /// // Perform element-wise subtraction of tensor2 from tensor1.
    /// let result = tensor1.sub(&tensor2);
    ///
    /// assert_eq!(result.get(&[0, 0]), &2); // 5 - 3
    /// assert_eq!(result.get(&[1, 2]), &2); // 5 - 3
    /// ```
    pub fn sub(&self, other: &Tensor<T>) -> Tensor<T> {
        if self.dimensions != other.dimensions {
            panic!("Tensors must have the same dimensions for subtraction");
        }

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

// Implement multiplication for `Tensor`
impl<T> Tensor<T>
where
    T: Copy + Mul<Output = T>,
{
    /// Performs element-wise multiplication of two tensors.
    ///
    /// This method multiplies the elements of one tensor with the corresponding elements of
    /// another tensor. Both tensors must have the same dimensions for the multiplication to be
    /// performed. The resulting tensor will have the same dimensions and strides as the original
    /// tensors, with each element being the result of multiplying the corresponding elements of
    /// `self` and `other`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///   The elements of `self` will be multiplied by the elements of `other`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match.
    ///
    /// # Returns
    /// New `Tensor<T>` that contains the result of the element-wise multiplication.
    /// The new tensor will have the same dimensions and strides as `self` and `other`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create two tensors with dimensions 2x3, initialized with values 2 and 3 respectively.
    /// let tensor1 = Tensor::new(vec![2, 3], 2);
    /// let tensor2 = Tensor::new(vec![2, 3], 3);
    ///
    /// // Perform element-wise multiplication of tensor1 and tensor2.
    /// let result = tensor1.mul(&tensor2);
    ///
    /// assert_eq!(result.get(&[0, 0]), &6); // 2 * 3
    /// assert_eq!(result.get(&[1, 2]), &6); // 2 * 3
    /// ```
    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T> {
        if self.dimensions != other.dimensions {
            panic!("Tensors must have the same dimensions for multiplication");
        }

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

// Implement division for `Tensor`
impl<T> Tensor<T>
where
    T: Copy + Default + PartialEq + Div<Output = T>,
{
    /// Performs element-wise division of two tensors.
    ///
    /// This method divides the elements of one tensor by the corresponding elements of another
    /// tensor. Both tensors must have the same dimensions for the division to be performed.
    /// The resulting tensor will have the same dimensions and strides as the original tensors,
    /// with each element being the result of dividing the elements of `self` by the elements of
    /// `other`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///   The elements of `self` will be divided by the elements of `other`.
    ///
    /// # Panics
    /// This method will panic if the dimensions of `self` and `other` do not match. Additionally,
    /// it will panic if any element of `other` is zero, as division by zero is not allowed.
    ///
    /// # Returns
    /// New `Tensor<T>` that contains the result of the element-wise division.
    /// The new tensor will have the same dimensions and strides as `self` and `other`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create two tensors with dimensions 2x3, initialized with values 6 and 2 respectively.
    /// let tensor1 = Tensor::new(vec![2, 3], 6);
    /// let tensor2 = Tensor::new(vec![2, 3], 2);
    ///
    /// // Perform element-wise division of tensor1 by tensor2.
    /// let result = tensor1.div(&tensor2);
    ///
    /// assert_eq!(result.get(&[0, 0]), &3); // 6 / 2
    /// assert_eq!(result.get(&[1, 2]), &3); // 6 / 2
    /// ```
    pub fn div(&self, other: &Tensor<T>) -> Tensor<T> {
        if self.dimensions != other.dimensions {
            panic!("Tensors must have the same dimensions for division");
        }

        let default_value = T::default();

        let data: Vec<T> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| {
                if *b == default_value {
                    panic!("Division by zero");
                }
                *a / *b
            })
            .collect();

        Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        }
    }
}

// Implement negation for `Tensor`
impl<T> Tensor<T>
where
    T: Copy + Neg<Output = T>,
{
    /// Performs element-wise negation of the tensor.
    ///
    /// This method negates each element of the tensor, resulting in a new tensor where each
    /// element is the negation of the corresponding element in the original tensor. The resulting
    /// tensor will have the same dimensions and strides as the original tensor.
    ///
    /// # Returns
    /// New `Tensor<T>` where each element is the negated value of the corresponding element
    /// in `self`. The new tensor will have the same dimensions and strides as `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create a tensor with dimensions 2x3, initialized with values 1 through 6.
    /// let tensor = Tensor::new(vec![2, 3], 1);
    ///
    /// // Perform element-wise negation of the tensor.
    /// let result = tensor.neg();
    ///
    /// assert_eq!(result.get(&[0, 0]), &-1); // Negation of 1
    /// assert_eq!(result.get(&[1, 2]), &-1); // Negation of 1
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

//////////////////////////////////////////////////////////////////////
// Mutable operations for `Tensor`
//////////////////////////////////////////////////////////////////////

// Implement mutable addition for `Tensor`
impl<T> Tensor<T>
where
    T: Copy + AddAssign,
{
    /// Performs in-place element-wise addition of another tensor to `self`.
    ///
    /// This method adds the elements of the provided tensor to the corresponding elements of the
    /// tensor on which it is called. The operation is performed in-place, modifying `self`
    /// directly. Both tensors must have the same dimensions for the addition to be valid.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///   This tensor will be added to the elements of `self`.
    ///
    /// # Panics
    /// This method panics if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create two tensors with dimensions 2x3, initialized with values.
    /// let mut tensor1 = Tensor::new(vec![2, 3], 1);
    /// let tensor2 = Tensor::new(vec![2, 3], 2);
    ///
    /// // Perform in-place element-wise addition.
    /// tensor1.add_mutate(&tensor2);
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &3); // 1 + 2
    /// assert_eq!(tensor1.get(&[1, 2]), &3); // 1 + 2
    /// ```
    pub fn add_mutate(&mut self, other: &Tensor<T>) {
        if self.dimensions != other.dimensions {
            panic!("Tensors must have the same dimensions for addition");
        }

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
    /// This method subtracts the elements of the provided tensor from the corresponding elements
    /// of the tensor on which it is called. The operation is performed in-place, modifying `self`
    /// directly. Both tensors must have the same dimensions for the subtraction to be valid.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///   The elements of `other` will be subtracted from the elements of `self`.
    ///
    /// # Panics
    /// This method panics if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create two tensors with dimensions 2x3, initialized with values.
    /// let mut tensor1 = Tensor::new(vec![2, 3], 5);
    /// let tensor2 = Tensor::new(vec![2, 3], 3);
    ///
    /// // Perform in-place element-wise subtraction.
    /// tensor1.sub_mutate(&tensor2);
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &2); // 5 - 3
    /// assert_eq!(tensor1.get(&[1, 2]), &2); // 5 - 3
    /// ```
    pub fn sub_mutate(&mut self, other: &Tensor<T>) {
        if self.dimensions != other.dimensions {
            panic!("Tensors must have the same dimensions for subtraction");
        }

        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a -= *b;
        }
    }
}

// Implement mutable multiplication for `Tensor`
impl<T> Tensor<T>
where
    T: Copy + MulAssign,
{
    /// Performs in-place element-wise multiplication of another tensor with `self`.
    ///
    /// This method multiplies each element of the provided tensor with the corresponding element
    /// of the tensor on which it is called. The operation is performed in-place, meaning `self` is
    /// directly modified to store the results. Both tensors must have the same dimensions for the
    /// multiplication to be valid.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///   The elements of `other` will be multiplied with the elements of `self`.
    ///
    /// # Panics
    /// This method panics if the dimensions of `self` and `other` do not match.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create two tensors with dimensions 2x3, initialized with values.
    /// let mut tensor1 = Tensor::new(vec![2, 3], 2);
    /// let tensor2 = Tensor::new(vec![2, 3], 3);
    ///
    /// // Perform in-place element-wise multiplication.
    /// tensor1.mul_mutate(&tensor2);
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &6); // 2 * 3
    /// assert_eq!(tensor1.get(&[1, 2]), &6); // 2 * 3
    /// ```
    pub fn mul_mutate(&mut self, other: &Tensor<T>) {
        if self.dimensions != other.dimensions {
            panic!("Tensors must have the same dimensions for multiplication");
        }

        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a *= *b;
        }
    }
}

// Implement mutable division for Tensor
impl<T> Tensor<T>
where
    T: Copy + Default + PartialEq + DivAssign,
{
    /// Performs in-place element-wise division of `self` by another tensor.
    ///
    /// This method divides each element of `self` by the corresponding element of the provided
    /// tensor, modifying `self` directly with the results. The operation is performed in-place,
    /// meaning that `self` will be updated to contain the results of the division. Both tensors
    /// must have the same dimensions for the division to be valid.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///   The elements of `self` will be divided by the corresponding elements of `other`.
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
    /// // Create two tensors with dimensions 2x3, initialized with values.
    /// let mut tensor1 = Tensor::new(vec![2, 3], 6);
    /// let tensor2 = Tensor::new(vec![2, 3], 3);
    ///
    /// // Perform in-place element-wise division.
    /// tensor1.div_mutate(&tensor2);
    ///
    /// assert_eq!(tensor1.get(&[0, 0]), &2); // 6 / 3
    /// assert_eq!(tensor1.get(&[1, 2]), &2); // 6 / 3
    /// ```
    pub fn div_mutate(&mut self, other: &Tensor<T>) {
        if self.dimensions != other.dimensions {
            panic!("Tensors must have the same dimensions for division");
        }

        let default_value = T::default();

        for (a, b) in self.data.iter_mut().zip(&other.data) {
            if *b == default_value {
                panic!("Division by zero");
            }
            *a /= *b;
        }
    }
}

// Implement mutable negation for `Tensor`
impl<T> Tensor<T>
where
    T: Copy + Neg<Output = T>,
{
    /// Performs in-place negation of each element in the tensor.
    ///
    /// This method negates each element of `self` in-place, modifying `self` directly with
    /// the results. The negation is applied to every element in the tensor, meaning that `self`
    /// will be updated to contain the negated values of its original elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create a tensor with dimensions 2x2, initialized with positive values.
    /// let mut tensor = Tensor::new(vec![2, 2], 5);
    ///
    /// // Perform in-place negation.
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

    // Test tensor addition
    #[test]
    fn test_add() {
        let tensor1 = Tensor::new(vec![2, 2], 1);
        let tensor2 = Tensor::new(vec![2, 2], 2);
        let result = tensor1.add(&tensor2);

        assert_eq!(result.get(&[0, 0]), &3);
        assert_eq!(result.get(&[0, 1]), &3);
        assert_eq!(result.get(&[1, 0]), &3);
        assert_eq!(result.get(&[1, 1]), &3);
    }

    // Test tensor subtraction
    #[test]
    fn test_sub() {
        let tensor1 = Tensor::new(vec![2, 2], 5);
        let tensor2 = Tensor::new(vec![2, 2], 3);
        let result = tensor1.sub(&tensor2);

        assert_eq!(result.get(&[0, 0]), &2);
        assert_eq!(result.get(&[0, 1]), &2);
        assert_eq!(result.get(&[1, 0]), &2);
        assert_eq!(result.get(&[1, 1]), &2);
    }

    // Test tensor multiplication
    #[test]
    fn test_mul() {
        let tensor1 = Tensor::new(vec![2, 2], 2);
        let tensor2 = Tensor::new(vec![2, 2], 3);
        let result = tensor1.mul(&tensor2);

        assert_eq!(result.get(&[0, 0]), &6);
        assert_eq!(result.get(&[0, 1]), &6);
        assert_eq!(result.get(&[1, 0]), &6);
        assert_eq!(result.get(&[1, 1]), &6);
    }

    // Test tensor division
    #[test]
    fn test_div() {
        let tensor1 = Tensor::new(vec![2, 2], 6);
        let tensor2 = Tensor::new(vec![2, 2], 3);
        let result = tensor1.div(&tensor2);

        assert_eq!(result.get(&[0, 0]), &2);
        assert_eq!(result.get(&[0, 1]), &2);
        assert_eq!(result.get(&[1, 0]), &2);
        assert_eq!(result.get(&[1, 1]), &2);
    }

    // Test tensor division by zero
    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_div_by_zero() {
        let tensor1 = Tensor::new(vec![2, 2], 6);
        let tensor2 = Tensor::new(vec![2, 2], 0);
        tensor1.div(&tensor2);
    }

    #[test]
    fn test_neg() {
        let tensor = Tensor::new(vec![2, 2], 5);
        let result = tensor.neg();
        assert_eq!(result.data, vec![-5, -5, -5, -5]);
    }

    #[test]
    fn test_add_mutate() {
        let mut tensor1 = Tensor::new(vec![2, 2], 1);
        let tensor2 = Tensor::new(vec![2, 2], 2);
        tensor1.add_mutate(&tensor2);
        assert_eq!(tensor1.data, vec![3, 3, 3, 3]);
    }

    #[test]
    fn test_sub_mutate() {
        let mut tensor1 = Tensor::new(vec![2, 2], 5);
        let tensor2 = Tensor::new(vec![2, 2], 3);
        tensor1.sub_mutate(&tensor2);
        assert_eq!(tensor1.data, vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_mul_mutate() {
        let mut tensor1 = Tensor::new(vec![2, 2], 3);
        let tensor2 = Tensor::new(vec![2, 2], 4);
        tensor1.mul_mutate(&tensor2);
        assert_eq!(tensor1.data, vec![12, 12, 12, 12]);
    }

    #[test]
    fn test_div_mutate() {
        let mut tensor1 = Tensor::new(vec![2, 2], 8);
        let tensor2 = Tensor::new(vec![2, 2], 4);
        tensor1.div_mutate(&tensor2);
        assert_eq!(tensor1.data, vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_neg_mutate() {
        let mut tensor = Tensor::new(vec![2, 2], 5);
        tensor.neg_mutate();
        assert_eq!(tensor.data, vec![-5, -5, -5, -5]);
    }
}