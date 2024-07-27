pub mod casting;

use crate::casting::Cast;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
use std::slice::{Iter, IterMut};

/// A multidimensional tensor data structure.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    data: Vec<T>,           // Stores the elements of the tensor in a flattened vector.
    dimensions: Vec<usize>, // Stores the size of each dimension of the tensor.
    strides: Vec<usize>,    // Stores the strides for each dimension.
}

impl<T> Tensor<T>
where
    T: Default + Clone + Debug + PartialEq,
{
    /// Creates a new tensor with the specified dimensions and initializes all elements to a given value.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: A vector specifying the size of each dimension of the tensor.
    /// - `initial_value`: The value to initialize all elements of the tensor.
    ///
    /// # Returns
    ///
    /// Returns an instance of `Tensor<T>`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    /// let tensor = Tensor::new(vec![2, 3], 0);
    /// ```
    pub fn new(dimensions: Vec<usize>, initial_value: T) -> Self {
        let size = dimensions.iter().product(); // Calculate total number of elements in the tensor.
        let data = vec![initial_value; size]; // Create a vector filled with the initial value.
        let strides = Self::compute_strides(&dimensions); // Compute strides
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
    /// - `indices`: A slice of `usize` values representing the indices in each dimension of the tensor.
    ///
    /// # Returns
    ///
    /// Returns a `usize` value which is the computed linear index in the flattened data vector.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of indices provided does not match the number of dimensions of the tensor.
    /// It will also panic if any of the indices are out of bounds.
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
    ///
    /// This function will panic if the number of indices provided does not match the number of dimensions of the tensor.
    /// It will also panic if any of the indices are out of bounds.
    ///
    /// # Returns
    ///
    /// Returns a reference to the value at the specified indices.
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
    ///
    /// This function will panic if the number of indices provided does not match the number of dimensions of the tensor.
    /// It will also panic if any of the indices are out of bounds.
    #[inline]
    pub fn set(&mut self, indices: &[usize], value: T) {
        let idx = self.linear_index(indices);
        self.data[idx] = value;
    }

    /// Returns the shape (dimensions) of the tensor.
    ///
    /// # Returns
    ///
    /// Returns a slice of `usize` representing the size of each dimension of the tensor.
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
    ///
    /// This method will panic if the new dimensions do not match the number of elements in the tensor.
    pub fn reshape(&mut self, dimensions: Vec<usize>) {
        let new_size: usize = dimensions.iter().product(); // Calculate total number of elements for new dimensions.
        if new_size != self.data.len() {
            panic!("New dimensions must have the same number of elements")
        }
        self.strides = Self::compute_strides(&dimensions);
        self.dimensions = dimensions; // Update dimensions to new values.
    }

    /// Checks if the tensor is empty.
    ///
    /// # Returns
    ///
    /// Returns `true` if the tensor has no elements, `false` otherwise.
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
    ///
    /// Returns the total number of elements as a `usize`.
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
    ///
    /// Returns the number of elements along the specified dimension.
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
        self.dimensions.get(dim_index).copied().unwrap_or(0)
    }

    #[inline]
    pub fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Consumes the tensor and converts it into a `Tensor<U>`.
    ///
    /// # Panics
    ///
    /// This method will panic if elements cannot be cast safely to the target type `U`.
    pub fn into_cast<U: Cast<T>>(self) -> Tensor<U> {
        let data: Vec<U> = self.data.into_iter().map(|value| U::cast(value)).collect();

        Tensor {
            data,
            dimensions: self.dimensions,
            strides: self.strides,
        }
    }

    /// Converts a reference to the current tensor into a new `Tensor<U>`.
    ///
    /// # Panics
    ///
    /// This method will panic if elements cannot be cast safely to the target type `U`.
    pub fn cast<U: Cast<T>>(&self) -> Tensor<U> {
        let data: Vec<U> = self
            .data
            .iter()
            .cloned()
            .map(|value| U::cast(value))
            .collect();

        Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        }
    }
}

// Implement the Index trait for Tensor
impl<T> Index<&[usize]> for Tensor<T>
where
    T: Default + Clone + Debug + PartialEq,
{
    type Output = T;

    /// Retrieves a reference to the value at the specified multidimensional indices using indexing syntax.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices specifying the position in each dimension.
    ///
    /// # Returns
    ///
    /// Returns a reference to the value at the specified indices.
    #[inline]
    fn index(&self, indices: &[usize]) -> &Self::Output {
        &self.data[self.linear_index(indices)]
    }
}

// Implement the IndexMut trait for Tensor
impl<T> IndexMut<&[usize]> for Tensor<T>
where
    T: Default + Clone + Debug + PartialEq,
{
    /// Retrieves a mutable reference to the value at the specified multidimensional indices using indexing syntax.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices specifying the position in each dimension.
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to the value at the specified indices.
    #[inline]
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let idx = self.linear_index(indices);
        &mut self.data[idx]
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: Default + Debug + Clone + PartialEq,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tensor Data:")?;

        let mut indices = vec![0; self.dimensions.len()];
        let mut ord = 1;

        loop {
            // Retrieve the value from the tensor
            let value = self.get(&indices);

            // Formatting indices as comma-separated values
            let index_str: String = indices
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<String>>()
                .join(",");

            // Use the retrieved value in the formatting string
            writeln!(f, "{}: [{}] -> {:?}", ord, index_str, value)?;

            // Move to the next index.
            for i in (0..self.dimensions.len()).rev() {
                if indices[i] + 1 < self.dimensions[i] {
                    indices[i] += 1;
                    break;
                } else if i == 0 {
                    return Ok(());
                } else {
                    indices[i] = 0;
                }
            }
            ord += 1;
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + Debug + Add<Output = T>,
{
    /// Performs element-wise addition of two tensors.
    ///
    /// This method adds two tensors of the same dimensions, element by element.
    /// The resulting tensor will have the same dimensions and strides as the original tensors, with each element being the sum of the corresponding elements from the two tensors.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`. The `other` tensor will be added to `self`.
    ///
    /// # Returns
    ///
    /// Returns a new `Tensor<T>` that contains the result of the element-wise addition.
    /// This new tensor will have the same dimensions and strides as `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `other` do not match.
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
            .map(|(a, b)| a.clone() + b.clone())
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
    T: Clone + Debug + Sub<Output = T>,
{
    /// Performs element-wise subtraction of two tensors.
    ///
    /// This method subtracts the elements of one tensor from the corresponding elements of another tensor.
    /// Both tensors must have the same dimensions for the subtraction to be performed. The resulting tensor will have the same dimensions and strides as the original tensors, with each element being the result of subtracting the corresponding elements of the `other` tensor from `self`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///   The elements of `self` will be subtracted by the elements of `other`.
    ///
    /// # Returns
    ///
    /// Returns a new `Tensor<T>` that contains the result of the element-wise subtraction.
    /// The new tensor will have the same dimensions and strides as `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `other` do not match.
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
            .map(|(a, b)| a.clone() - b.clone())
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
    T: Clone + Debug + Mul<Output = T>,
{
    /// Performs element-wise multiplication of two tensors.
    ///
    /// This method multiplies the elements of one tensor with the corresponding elements of another tensor.
    /// Both tensors must have the same dimensions for the multiplication to be performed.
    /// The resulting tensor will have the same dimensions and strides as the original tensors, with each element being the result of multiplying the corresponding elements of `self` and `other`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`. The elements of `self` will be multiplied by the elements of `other`.
    ///
    /// # Returns
    ///
    /// Returns a new `Tensor<T>` that contains the result of the element-wise multiplication.
    /// The new tensor will have the same dimensions and strides as `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `other` do not match.
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
            .map(|(a, b)| a.clone() * b.clone())
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
    T: Clone + Debug + PartialEq + Div<Output = T> + Default,
{
    /// Performs element-wise division of two tensors.
    ///
    /// This method divides the elements of one tensor by the corresponding elements of another tensor.
    /// Both tensors must have the same dimensions for the division to be performed.
    /// The resulting tensor will have the same dimensions and strides as the original tensors, with each element being the result of dividing the elements of `self` by the elements of `other`.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` that has the same dimensions as `self`.
    ///   The elements of `self` will be divided by the elements of `other`.
    ///
    /// # Returns
    ///
    /// Returns a new `Tensor<T>` that contains the result of the element-wise division.
    /// The new tensor will have the same dimensions and strides as `self` and `other`.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `other` do not match.
    /// Additionally, it will panic if any element of `other` is zero, as division by zero is not allowed.
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

        let data: Vec<T> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| {
                if b.clone() == T::default() {
                    panic!("Division by zero");
                }
                a.clone() / b.clone()
            })
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
    T: Clone + Debug + Neg<Output = T>,
{
    /// Performs element-wise negation of the tensor.
    ///
    /// This method negates each element of the tensor, resulting in a new tensor where each element is the negation of the corresponding element in the original tensor.
    /// The resulting tensor will have the same dimensions and strides as the original tensor.
    ///
    /// # Returns
    ///
    /// Returns a new `Tensor<T>` where each element is the negated value of the corresponding element in `self`.
    /// The new tensor will have the same dimensions and strides as `self`.
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
        let data: Vec<T> = self.data.iter().map(|a| -a.clone()).collect();

        Tensor {
            data,
            dimensions: self.dimensions.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + Debug + Add<Output = T> + AddAssign,
{
    /// Performs in-place element-wise addition of another tensor to `self`.
    ///
    /// This method adds the elements of the provided tensor to the corresponding elements of the tensor on which it is called.
    /// The operation is performed in-place, modifying `self` directly.
    /// Both tensors must have the same dimensions for the addition to be valid.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///   This tensor will be added to the elements of `self`.
    ///
    /// # Returns
    ///
    /// This method does not return any value. The result of the addition is stored in `self`, modifying it directly.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `other` do not match.
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
            *a += b.clone();
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + Debug + Sub<Output = T> + SubAssign,
{
    /// Performs in-place element-wise subtraction of another tensor from `self`.
    ///
    /// This method subtracts the elements of the provided tensor from the corresponding elements of the tensor on which it is called.
    /// The operation is performed in-place, modifying `self` directly.
    /// Both tensors must have the same dimensions for the subtraction to be valid.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///   The elements of `other` will be subtracted from the elements of `self`.
    ///
    /// # Returns
    ///
    /// This method does not return any value. The result of the subtraction is stored in `self`, modifying it directly.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `other` do not match.
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
            *a -= b.clone();
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + Debug + Mul<Output = T> + MulAssign,
{
    /// Performs in-place element-wise multiplication of another tensor with `self`.
    ///
    /// This method multiplies each element of the provided tensor with the corresponding element of the tensor on which it is called.
    /// The operation is performed in-place, meaning `self` is directly modified to store the results.
    /// Both tensors must have the same dimensions for the multiplication to be valid.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`. The elements of `other` will be multiplied with the elements of `self`.
    ///
    /// # Returns
    ///
    /// This method does not return any value. The result of the multiplication is stored in `self`, modifying it directly.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions of `self` and `other` do not match.
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
            *a *= b.clone();
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + Debug + PartialEq + Default + Div<Output = T> + DivAssign,
{
    /// Performs in-place element-wise division of `self` by another tensor.
    ///
    /// This method divides each element of `self` by the corresponding element of the provided tensor, modifying `self` directly with the results.
    /// The operation is performed in-place, meaning that `self` will be updated to contain the results of the division.
    /// Both tensors must have the same dimensions for the division to be valid.
    ///
    /// # Parameters
    ///
    /// - `other`: A reference to another `Tensor<T>` with the same dimensions as `self`.
    ///   The elements of `self` will be divided by the corresponding elements of `other`.
    ///
    /// # Returns
    ///
    /// This method does not return any value. The result of the division is stored in `self`, modifying it directly.
    ///
    /// # Panics
    ///
    /// - Panics if the dimensions of `self` and `other` do not match. Both tensors must have the same dimensions for the division to be performed.
    /// - Panics if any element of `other` is zero, as division by zero is not allowed.
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

        for (a, b) in self.data.iter_mut().zip(&other.data) {
            if b.clone() == T::default() {
                panic!("Division by zero");
            }
            *a /= b.clone();
        }
    }
}

impl<T> Tensor<T>
where
    T: Clone + Debug + Neg<Output = T>,
{
    /// Performs in-place negation of each element in the tensor.
    ///
    /// This method negates each element of `self` in-place, modifying `self` directly with the results.
    /// The negation is applied to every element in the tensor, meaning that `self` will be updated to contain the negated values of its original elements.
    ///
    /// # Returns
    ///
    /// This method does not return any value. The result of the negation is stored in `self`, modifying it directly.
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
            *a = -a.clone();
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

        tensor.reshape(vec![3, 2]);
        assert_eq!(tensor.shape(), &[3, 2]);

        assert_eq!(tensor.get(&[0, 0]), &1);
        assert_eq!(tensor.get(&[0, 1]), &2);
        assert_eq!(tensor.get(&[1, 0]), &3);
        assert_eq!(tensor.get(&[1, 1]), &4);
        assert_eq!(tensor.get(&[2, 0]), &5);
        assert_eq!(tensor.get(&[2, 1]), &6);
    }

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

    #[test]
    fn test_ops_with_casting() {
        // Create a tensor with floating-point numbers
        let tensor1 = Tensor::new(vec![2, 2], 1.0);

        // Create a tensor with integer numbers and attempt to cast to floating-point
        let tensor2 = Tensor::new(vec![2, 2], 2);
        let tensor2_f64 = tensor2.cast::<f64>();

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

    // Test display output
    #[test]
    fn test_display_tensor_2d() {
        let mut tensor = Tensor::new(vec![2, 2], 0);
        tensor.set(&[0, 0], 1);
        tensor.set(&[0, 1], 2);
        tensor.set(&[1, 0], 3);
        tensor.set(&[1, 1], 4);

        let expected_output = r#"Tensor Data:
1: [0,0] -> 1
2: [0,1] -> 2
3: [1,0] -> 3
4: [1,1] -> 4
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

        let expected_output = r#"Tensor Data:
1: [0,0,0] -> 1
2: [0,0,1] -> 2
3: [0,1,0] -> 3
4: [0,1,1] -> 4
5: [1,0,0] -> 5
6: [1,0,1] -> 6
7: [1,1,0] -> 7
8: [1,1,1] -> 8
"#;

        let output = format!("{}", tensor);
        assert_eq!(output, expected_output);
    }
}
