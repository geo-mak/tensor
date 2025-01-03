// Public modules
pub mod cast;

// Private modules
mod ops;
mod similarity;

// Private imports
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};

use crate::cast::error::CastError;
use crate::cast::traits::TryCast;

/// A multidimensional tensor data structure.
#[derive(Clone, PartialEq)]
pub struct Tensor<T> {
    data: Vec<T>,
    dimensions: Vec<usize>,
    strides: Vec<usize>,
}

//////////////////////////////////////////////////////////////////////
// Core implementation for `Tensor`
//////////////////////////////////////////////////////////////////////

impl<T> Tensor<T>
where
    T: Copy,
{
    /// Creates a new tensor with the specified dimensions and initializes all elements to a
    /// given value.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: A vector specifying the size of each dimension of the tensor.
    /// - `initial_value`: The value to initialize all elements of the tensor.
    ///
    /// # Returns
    /// An instance of `Tensor<T>`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    /// let tensor = Tensor::new(vec![2, 3], 0);
    /// ```
    pub fn new(dimensions: Vec<usize>, initial_value: T) -> Self {
        // Calculate total number of elements in the tensor.
        let size = dimensions.iter().product();
        // Create a vector filled with the initial value.
        let data = vec![initial_value; size];
        // Compute strides
        let strides = Self::compute_strides(&dimensions);
        Tensor {
            data,
            dimensions,
            strides,
        }
    }

    /// Computes the strides for each dimension based on the provided dimensions.
    fn compute_strides(dimensions: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dimensions.len()];
        for i in (0..dimensions.len() - 1).rev() {
            strides[i] = strides[i + 1] * dimensions[i + 1];
        }
        strides
    }

    /// Calculates the linear index in the flattened data vector from the multidimensional indices.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of `usize` values representing the indices in each dimension of the
    ///   tensor.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor.
    ///
    /// # Returns
    /// A `usize` value which is the computed linear index in the flattened data vector.
    #[inline]
    fn linear_index(&self, indices: &[usize]) -> usize {
        if indices.len() != self.dimensions.len() {
            panic!("Incorrect number of indices");
        }

        indices
            .iter()
            .zip(&self.strides)
            .map(|(&ind, &stride)| ind * stride)
            .sum()
    }

    /// Retrieves a reference to the value at the specified multidimensional indices.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if any of the indices are out of bounds.
    ///
    /// # Returns
    /// A reference to the value at the specified indices.
    #[inline]
    pub fn get(&self, indices: &[usize]) -> &T {
        &self.data[self.linear_index(indices)]
    }

    /// Sets the value at the specified multidimensional indices.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices specifying the position in each dimension.
    /// - `value`: The value to set at the specified indices.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if any of the indices are out of bounds.
    #[inline]
    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = self.linear_index(indices);
        self.data[idx] = value;
    }

    /// Returns the shape (dimensions) of the tensor.
    ///
    /// # Returns
    /// A slice of `usize` representing the size of each dimension of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.dimensions
    }

    /// Reshapes the tensor to new dimensions.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: A vector specifying the new size of each dimension.
    ///
    /// # Panics
    /// This method will panic if the new dimensions do not match the number of elements in the
    /// tensor.
    pub fn reshape(&mut self, dimensions: &[usize]) {
        // Calculate total number of elements for new dimensions.
        let new_size: usize = dimensions.iter().product();
        if new_size != self.data.len() {
            panic!("New dimensions must have the same number of elements");
        }
        self.strides = Self::compute_strides(dimensions);
        // Update dimensions to new values.
        self.dimensions.clear();
        self.dimensions.extend_from_slice(dimensions);
    }

    /// Checks if the tensor is empty.
    ///
    /// # Returns
    /// `true` if the tensor has no elements, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// This method calculates the total number of elements by multiplying the sizes
    /// of all dimensions. For example, a tensor with dimensions `[2, 3, 4]` will have
    /// `2 * 3 * 4 = 24` elements.
    ///
    /// # Returns
    /// The total number of elements as a `usize`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// // Create a tensor with dimensions 2x3x4
    /// let tensor = Tensor::new(vec![2, 3, 4], 0.0);
    ///
    /// // Get the total number of elements
    /// let total_elements = tensor.len();
    ///
    /// // The total number of elements is 2 * 3 * 4 = 24
    /// assert_eq!(total_elements, 24);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.dimensions.iter().product()
    }

    /// Returns the number of elements along a specific dimension.
    ///
    /// # Parameters
    ///
    /// - `dim_index`: The index of the dimension (0-based) for which to get the size.
    ///
    /// # Returns
    /// The number of elements along the specified dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    /// let tensor = Tensor::new(vec![2, 3], 0.0);
    ///
    /// assert_eq!(tensor.len(), 6);
    /// assert_eq!(tensor.dim_len(0), 2);
    /// assert_eq!(tensor.dim_len(1), 3);
    /// ```
    #[inline]
    pub fn dim_len(&self, dim_index: usize) -> usize {
        match self.dimensions.get(dim_index) {
            Some(&len) => len,
            None => 0,
        }
    }

    /// Returns an iterator over the elements of the tensor.
    #[inline]
    pub fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    /// Returns a mutable iterator over the elements of the tensor.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Attempts to cast the tensor into a tensor of a different type without consuming
    /// the original tensor.
    ///
    /// # Returns
    /// - `Ok(Tensor<U>)`: if the conversion is successful.
    /// - `Err(CastError)`: if the conversion fails.
    pub fn try_cast<U>(&self) -> Result<Tensor<U>, CastError>
    where
        T: TryCast<U>,
    {
        let data: Vec<U> = self
            .data
            .iter()
            .map(|value| value.try_cast())
            .collect::<Result<Vec<U>, _>>()?;

        Ok(Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        })
    }
}

// Implement the Index trait for `Tensor`
impl<T> Index<&[usize]> for Tensor<T>
where
    T: Copy,
{
    type Output = T;

    /// Retrieves a reference to the value at the specified multidimensional indices using
    /// indexing syntax.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if any of the indices are out of bounds.
    ///
    /// # Returns
    /// A reference to the value at the specified indices.
    #[inline]
    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.data[self.linear_index(indices)]
    }
}

// Implement the IndexMut trait for Tensor
impl<T> IndexMut<&[usize]> for Tensor<T>
where
    T: Copy,
{
    /// Retrieves a mutable reference to the value at the specified multidimensional indices
    /// using indexing syntax.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if any of the indices are out of bounds.
    ///
    /// # Returns
    /// A mutable reference to the value at the specified indices.
    #[inline]
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let idx = self.linear_index(indices);
        &mut self.data[idx]
    }
}

//////////////////////////////////////////////////////////////////////
// Display and Debug implementations for `Tensor`
//////////////////////////////////////////////////////////////////////

// Implement Debug for `Tensor`
impl<T> Debug for Tensor<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data)
            .field("dimensions", &self.dimensions)
            .field("strides", &self.strides)
            .finish()
    }
}

// Implement Display for `Tensor`
impl<T> Display for Tensor<T>
where
    T: Copy + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Get the number of dimensions of the tensor
        let dim_len = self.dimensions.len();

        // Tracking vector to keep track of the current indices
        // Initialize with zeros: [0, 0, ..., 0]
        let mut indices = vec![0; dim_len];

        // Initialize the ordinal number of the element: 0, 1, 2, ...
        let mut ord = 0;

        loop {
            // Get the value at the current indices
            let value = self.get(&indices);

            // Write according to the format: `ord: [indices] -> value`
            writeln!(f, "{}: {:?} -> {}", ord, indices, value)?;

            // Move to the next index
            'inner: for i in (0..dim_len).rev() {
                // Check if the current index can be incremented without exceeding the dimension size
                if indices[i] + 1 < self.dimensions[i] {
                    // Increment the current index
                    indices[i] += 1;
                    break 'inner;
                } else if i == 0 {
                    // If the first index has reached its maximum value, we are done
                    return Ok(());
                } else {
                    // Reset the current index to 0 and continue to the next index
                    indices[i] = 0;
                }
            }
            ord += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test setting and getting tensor values
    #[test]
    fn test_set_get() {
        let mut tensor = Tensor::new(vec![2, 3], 0);
        tensor.set(&[0, 0], 1);
        tensor.set(&[0, 1], 2);
        tensor.set(&[0, 2], 3);
        tensor.set(&[1, 0], 4);
        tensor.set(&[1, 1], 5);
        tensor.set(&[1, 2], 6);

        assert_eq!(tensor.get(&[0, 0]), &1);
        assert_eq!(tensor.get(&[0, 1]), &2);
        assert_eq!(tensor.get(&[0, 2]), &3);
        assert_eq!(tensor.get(&[1, 0]), &4);
        assert_eq!(tensor.get(&[1, 1]), &5);
        assert_eq!(tensor.get(&[1, 2]), &6);
    }

    // Test direct indexing with the Index and IndexMut traits
    #[test]
    fn test_direct_indexing() {
        let mut tensor = Tensor::new(vec![2, 3], 0);
        tensor[&[0, 2]] = 100;
        assert_eq!(tensor[&[0, 2]], 100);
    }

    // Test reshaping the tensor
    #[test]
    fn test_reshape() {
        let mut tensor = Tensor::new(vec![2, 3], 0);
        tensor.set(&[0, 0], 1);
        tensor.set(&[0, 1], 2);
        tensor.set(&[0, 2], 3);
        tensor.set(&[1, 0], 4);
        tensor.set(&[1, 1], 5);
        tensor.set(&[1, 2], 6);

        tensor.reshape(&[3, 2]);
        assert_eq!(tensor.shape(), &[3, 2]);

        assert_eq!(tensor.get(&[0, 0]), &1);
        assert_eq!(tensor.get(&[0, 1]), &2);
        assert_eq!(tensor.get(&[1, 0]), &3);
        assert_eq!(tensor.get(&[1, 1]), &4);
        assert_eq!(tensor.get(&[2, 0]), &5);
        assert_eq!(tensor.get(&[2, 1]), &6);
    }

    #[test]
    fn test_ops_with_casting() {
        // Create a tensor with floating-point numbers
        let tensor1 = Tensor::new(vec![2, 2], 1.0);

        // Create a tensor with integer numbers and attempt to cast to floating-point
        let tensor2 = Tensor::new(vec![2, 2], 2);

        // Attempt to cast the tensor to f64
        let tensor2_f64 = tensor2.try_cast::<f64>().unwrap();

        // Perform tensor operations
        let result_add = tensor1.add(&tensor2_f64);
        let result_sub = tensor1.sub(&tensor2_f64);
        let result_mul = tensor1.mul(&tensor2_f64);
        let result_div = tensor1.div(&tensor2_f64);

        // Define expected results for operations
        let expected_add = Tensor::new(vec![2, 2], 3.0);
        let expected_sub = Tensor::new(vec![2, 2], -1.0);
        let expected_mul = Tensor::new(vec![2, 2], 2.0);
        let expected_div = Tensor::new(vec![2, 2], 0.5);

        // Verify results for addition
        assert_eq!(result_add, expected_add);

        // Verify results for subtraction
        assert_eq!(result_sub, expected_sub);

        // Verify results for multiplication
        assert_eq!(result_mul, expected_mul);

        // Verify results for division
        assert_eq!(result_div, expected_div);
    }

    #[test]
    fn test_cast_overflow() {
        let tensor = Tensor::<u16>::new(vec![2, 2], 256);

        // Attempt to cast the tensor into a tensor of u8
        let result: Result<Tensor<u8>, CastError> = tensor.try_cast();

        // Result must be an error due to overflow
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CastError::Overflow);
    }

    #[test]
    fn test_cast_precision_loss() {
        let tensor = Tensor::<f32>::new(vec![2, 2], 3.14);

        // Attempt to cast the tensor into a tensor of i32
        let result: Result<Tensor<i32>, CastError> = tensor.try_cast();

        // Result must be an error due to precision loss
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CastError::PrecisionLoss);
    }

    #[test]
    fn test_debug_tensor() {
        let tensor = Tensor::new(vec![2, 2], 1);
        let expected_output = "Tensor { data: [1, 1, 1, 1], dimensions: [2, 2], strides: [2, 1] }";
        let debug_output = format!("{:?}", tensor);
        assert_eq!(debug_output, expected_output);
    }

    #[test]
    fn test_display_tensor_2d() {
        let mut tensor = Tensor::new(vec![2, 2], 0);
        tensor.set(&[0, 0], 1);
        tensor.set(&[0, 1], 2);
        tensor.set(&[1, 0], 3);
        tensor.set(&[1, 1], 4);

        let expected_output = r#"0: [0, 0] -> 1
1: [0, 1] -> 2
2: [1, 0] -> 3
3: [1, 1] -> 4
"#;

        let output = format!("{}", tensor);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_display_tensor_3d() {
        let mut tensor = Tensor::new(vec![2, 2, 2], 0);
        tensor.set(&[0, 0, 0], 1);
        tensor.set(&[0, 0, 1], 2);
        tensor.set(&[0, 1, 0], 3);
        tensor.set(&[0, 1, 1], 4);
        tensor.set(&[1, 0, 0], 5);
        tensor.set(&[1, 0, 1], 6);
        tensor.set(&[1, 1, 0], 7);
        tensor.set(&[1, 1, 1], 8);

        let expected_output = r#"0: [0, 0, 0] -> 1
1: [0, 0, 1] -> 2
2: [0, 1, 0] -> 3
3: [0, 1, 1] -> 4
4: [1, 0, 0] -> 5
5: [1, 0, 1] -> 6
6: [1, 1, 0] -> 7
7: [1, 1, 1] -> 8
"#;

        let output = format!("{}", tensor);
        assert_eq!(output, expected_output);
    }
}
