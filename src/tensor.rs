use core::fmt;
use core::fmt::{Debug, Display, Formatter};

use crate::core::alloc::UnsafeBufferPointer;
use crate::metadata::TensorMetaData;

/// `Tensor` is a generic data structure that logically arranges its data as multidimensional
/// points.
///
/// Each dimension is represented as a contiguous vector, where each value corresponds to a
/// coordinate.
///
/// Each value stored in the tensor has a number of coordinates corresponds to the number
/// of dimensions of the tensor.
///
/// The number of dimensions, known as `Rank`, is denoted by the character `R`.
/// A scalar has rank `0`, a vector has rank `1`, a matrix is rank `2`.
///
/// `Tensor` has two generic parameters:
///
/// - `T`: The `type` of the stored data.  
/// - `R`: A `value` specifying the rank of the tensor.
///
/// These two parameters are part of the type definition, and they remain unchanged throughout the
/// instance's lifetime.
pub struct Tensor<T, const R: usize> {
    pub(crate) metadata: TensorMetaData<R>,
    pub(crate) data: UnsafeBufferPointer<T>,
}

impl<T, const R: usize> Drop for Tensor<T, R> {
    fn drop(&mut self) {
        // len is assumed to be > 0, an invariant that must be upheld by all constructors.
        let len = self.metadata.size();
        unsafe {
            self.data.drop_initialized(len);
            self.data.deallocate(len);
        }
    }
}

impl<T, const R: usize> Clone for Tensor<T, R>
where
    T: Clone,
{
    /// Creates new `Tensor` by cloning the data from the current instance.
    fn clone(&self) -> Self {
        unsafe {
            let metadata = self.metadata;
            Tensor {
                metadata,
                data: self.data.make_clone(metadata.size()),
            }
        }
    }
}

impl<T, const R: usize> Tensor<T, R>
where
    T: Copy,
{
    /// Creates new `Tensor` by copying _bitwise_ the data from the current instance.
    pub fn copy(&self) -> Self {
        unsafe {
            let metadata = self.metadata;
            Tensor {
                metadata,
                data: self.data.make_copy(metadata.size()),
            }
        }
    }
}

impl<T, const R: usize> PartialEq for Tensor<T, R>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        // cmp order:
        // 1: cmp size
        // 2: cmp dims
        // 3: cmp data
        if !self.metadata.cmp_eq(&other.metadata) {
            return false;
        }
        unsafe { self.data.cmp_eq(&other.data, self.metadata.size()) }
    }
}

impl<T, const R: usize> Debug for Tensor<T, R>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("metadata", &self.metadata)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T, const R: usize> Display for Tensor<T, R>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let len =  self.metadata.size();
        let shape = self.metadata.shape();
        
        writeln!(f, "Shape: {:?}", shape)?;
        writeln!(f, "Data:")?;
        
        let mut index = [0; R];
        let mut num = 0;
        
        while num < len {
            let value = self.get(&index);

            writeln!(f, "{}: {:?} -> {}", num, index, value)?;
            
            // Only reachable if R > 0.
            let mut i = R;
            'idx: while i != 0 {
                i -= 1;
                // Try incrementing within bounds.
                if index[i] + 1 < shape[i] {
                    index[i] += 1;
                    break 'idx
                } else if i == 0 {
                    // All dimensions have been traversed.
                    return Ok(());
                } else {
                    index[i] = 0;
                }
            }

            num += 1;
        }
        
        // Only reachable if R == 0.
        Ok(())
    }
}

#[cfg(test)]
mod core_ops_tests {
    use super::*;

    #[test]
    fn test_tensor_copy() {
        let mut tensor = Tensor::from_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        let copied_tensor = tensor.copy();

        assert_eq!(tensor.shape(), copied_tensor.shape());
        assert_eq!(tensor.size(), copied_tensor.size());
        assert_eq!(tensor.as_slice(), copied_tensor.as_slice());

        tensor.set(&[1, 2], 10);
        assert_eq!(copied_tensor.get(&[1, 2]), &6);
    }

    #[test]
    fn test_clone() {
        let mut tensor = Tensor::from_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        let cloned_tensor = tensor.clone();

        assert_eq!(tensor.shape(), cloned_tensor.shape());
        assert_eq!(tensor.size(), cloned_tensor.size());
        assert_eq!(tensor.as_slice(), cloned_tensor.as_slice());

        tensor.set(&[1, 2], 10);
        assert_eq!(cloned_tensor.get(&[1, 2]), &6);
    }

    #[test]
    fn test_partial_eq() {
        let tensor1 = Tensor::from_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        let tensor2 = Tensor::from_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        let tensor3 = Tensor::from_vec([2, 3], vec![6, 5, 4, 3, 2, 1]);

        assert_eq!(tensor1, tensor2);
        assert_ne!(tensor1, tensor3);
    }

    #[test]
    fn test_tensor_display() {
        let tensor = Tensor::from_slice([2, 2, 2], &[1, 2, 3, 4, 5, 6, 7, 8]);

        let expected = "\
Shape: [2, 2, 2]
Data:
0: [0, 0, 0] -> 1
1: [0, 0, 1] -> 2
2: [0, 1, 0] -> 3
3: [0, 1, 1] -> 4
4: [1, 0, 0] -> 5
5: [1, 0, 1] -> 6
6: [1, 1, 0] -> 7
7: [1, 1, 1] -> 8
";

        let fmt_output = format!("{}", tensor);
        assert_eq!(fmt_output, expected);
    }

    #[test]
    fn test_tensor_display_zero_rank() {
        let tensor = Tensor::from_slice([], &[1]);

        let expected = "\
Shape: []
Data:
0: [] -> 1
";

        let fmt_output = format!("{}", tensor);
        assert_eq!(fmt_output, expected);
    }
}
