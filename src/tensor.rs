use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::ops::{Index, IndexMut};
use core::slice::{Iter, IterMut};

use crate::{CastError, TryCast};

/// A multidimensional tensor data structure.
#[derive(Clone, PartialEq)]
pub struct Tensor<T> {
    pub(crate) data: Vec<T>,
    pub(crate) dimensions: Vec<usize>,
    pub(crate) strides: Vec<usize>,
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
    /// For creating `Tensor` declaratively, consider using `tensor!` macro.
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
}

impl<T> Tensor<T> {
    /// Creates a new tensor with the specified data and dimensions.
    ///
    /// For creating `Tensor` declaratively, consider using `tensor!` macro.
    ///
    /// # Parameters
    ///
    /// - `data`: A vector specifying the data in the tensor.
    /// - `dimensions`: A vector specifying the size of each dimension of the tensor.
    ///
    /// # Returns
    /// An instance of `Tensor<T>`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    /// let tensor = Tensor::with_data(vec![1,2,3,4,5,6], vec![2, 3]);
    ///
    /// assert_eq!(tensor.shape(), &[2, 3]);
    ///
    /// assert_eq!(tensor.get(&[0,0]), &1);
    /// assert_eq!(tensor.get(&[0,1]), &2);
    /// assert_eq!(tensor.get(&[0,2]), &3);
    ///
    /// assert_eq!(tensor.get(&[1,0]), &4);
    /// assert_eq!(tensor.get(&[1,1]), &5);
    /// assert_eq!(tensor.get(&[1,2]), &6);
    /// ```
    pub fn with_data(data: Vec<T>, dimensions: Vec<usize>) -> Self {
        Self {
            data,
            strides: Self::compute_strides(&dimensions),
            dimensions,
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
{
    type Output = T;

    /// Retrieves a reference to the value at the specified multidimensional indices using
    /// indexing preprocessing.
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
{
    /// Retrieves a mutable reference to the value at the specified multidimensional indices
    /// using indexing preprocessing.
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
    T: Display,
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
