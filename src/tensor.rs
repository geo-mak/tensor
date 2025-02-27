use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::ops::{Index, IndexMut};
use core::slice::{Iter, IterMut};

use crate::{CastError, TryCast};

use crate::assertions::{assert_valid_dimensions, assert_valid_shape};

/// A multidimensional tensor data structure.
#[derive(Clone, PartialEq)]
pub struct Tensor<T> {
    pub(crate) data: Vec<T>,
    pub(crate) dimensions: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T> Tensor<T> {
    /// Creates a new tensor with the specified dimensions and initializes all elements to a
    /// given value.
    ///
    /// For creating `Tensor` declaratively, consider using `tensor!` macro.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: A vector specifying the size of each dimension of the tensor.
    /// - `value`: The value to initialize all elements of the tensor.
    ///
    /// # Panics
    /// This method will panic if dimensions are invalid e.g. empty or an axis is `0`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor = Tensor::new_set(vec![2, 3], 0);
    /// ```
    pub fn new_set(dimensions: Vec<usize>, value: T) -> Self
    where
        T: Clone,
    {
        let size = dimensions.iter().product();
        assert_valid_dimensions(&dimensions, size);
        let data = vec![value; size];
        let strides = Self::compute_strides(&dimensions);
        Tensor {
            data,
            dimensions,
            strides,
        }
    }

    /// Creates a new tensor with the specified values and dimensions.
    ///
    /// For creating `Tensor` declaratively, consider using `tensor!` macro.
    ///
    /// # Parameters
    ///
    /// - `values`: A vector specifying the values in the tensor.
    /// - `dimensions`: A vector specifying the size of each dimension of the tensor.
    ///
    /// # Panics
    /// This method will panic if the dimensions do not match the number of elements in the
    /// tensor or if an axis is `0`
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    /// let tensor = Tensor::with_values(vec![1,2,3,4,5,6], vec![2, 3]);
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
    pub fn with_values(values: Vec<T>, dimensions: Vec<usize>) -> Self {
        assert_valid_shape(values.len(), &dimensions);
        Self {
            data: values,
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

    /// Computes and returns the linear index of an item in the data vector.
    ///
    /// # Parameters
    ///
    /// - `index`: A slice of `usize` values representing the indices in each dimension of the
    ///   tensor.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor.
    #[inline]
    fn compute_index(&self, index: &[usize]) -> usize {
        if index.len() != self.dimensions.len() {
            panic!("Incorrect number of indices");
        }

        index
            .iter()
            .zip(&self.strides)
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Sets the value at the specified multidimensional indices.
    ///
    /// # Parameters
    ///
    /// - `index`: A slice of indices specifying the position in each dimension.
    /// - `value`: The value to set at the specified indices.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if any of the indices are out of bounds.
    #[inline]
    pub fn set(&mut self, index: &[usize], value: T) {
        let idx = self.compute_index(index);
        self.data[idx] = value;
    }

    /// Returns a reference to the value at the specified multidimensional index.
    ///
    /// # Parameters
    ///
    /// - `indices`: A slice of indices specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if any of the indices are out of bounds.
    #[inline]
    pub fn get(&self, index: &[usize]) -> &T {
        &self.data[self.compute_index(index)]
    }

    /// Returns the shape (dimensions) of the tensor.
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
    /// This method will panic if the new dimensions' size can't represent the elements in the
    /// tensor.
    pub fn reshape(&mut self, dimensions: &[usize]) {
        assert_valid_shape(self.data.len(), dimensions);
        self.strides = Self::compute_strides(dimensions);
        self.dimensions.clear();
        self.dimensions.extend_from_slice(dimensions);
    }

    /// Returns the total number of elements the tensor can store with the current dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    ///
    /// let tensor = Tensor::new_set(vec![2, 3, 4], 0.0);
    ///
    /// // The total number of elements is 2 * 3 * 4 = 24
    /// assert_eq!(tensor.size(), 24);
    /// ```
    #[inline]
    pub fn size(&self) -> usize {
        self.dimensions.iter().product()
    }

    /// Returns the number of elements along a specific dimension.
    ///
    /// # Parameters
    ///
    /// - `index`: The index of the dimension (0-based) for which to get the size.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor::Tensor;
    /// let tensor = Tensor::new_set(vec![2, 3], 0.0);
    ///
    /// assert_eq!(tensor.size(), 6);
    /// assert_eq!(tensor.dim_size(0), Some(2));
    /// assert_eq!(tensor.dim_size(1), Some(3));
    /// ```
    #[inline]
    pub fn dim_size(&self, index: usize) -> Option<usize> {
        if let Some(&len) = self.dimensions.get(index) {
            return Some(len);
        }
        None
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

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    /// Returns a reference to the value at the specified multidimensional index.
    ///
    /// # Parameters
    ///
    /// - `index`: A slice of indices specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    #[inline]
    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.data[self.compute_index(index)]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    /// Returns a mutable reference to the value at the specified multidimensional index.
    ///
    /// # Parameters
    ///
    /// - `index`: A slice of indices specifying the position in each dimension.
    ///
    /// # Panics
    /// This method will panic if the number of indices provided does not match the number of
    /// dimensions of the tensor. It will panic also if index is out of bounds.
    #[inline]
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let idx = self.compute_index(index);
        &mut self.data[idx]
    }
}

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

impl<T> Display for Tensor<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let dim_len = self.dimensions.len();
        let mut index = vec![0; dim_len];
        let mut ord = 0;

        writeln!(f, "Shape: {:?}", self.dimensions)?;
        writeln!(f, "Data:")?;

        loop {
            let value = self.get(&index);

            writeln!(f, "{}: {:?} -> {}", ord, index, value)?;

            'inner: for i in (0..dim_len).rev() {
                if index[i] + 1 < self.dimensions[i] {
                    index[i] += 1;
                    break 'inner;
                } else if i == 0 {
                    return Ok(());
                } else {
                    index[i] = 0;
                }
            }
            ord += 1;
        }
    }
}
