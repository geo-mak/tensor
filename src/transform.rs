use core::mem::ManuallyDrop;

use crate::metadata::TensorMetaData;
use crate::Tensor;

impl<T, const R: usize> Tensor<T, R> {
    /// Reshapes the tensor to new dimensions.
    ///
    /// This method doesn't reorder values, only the index is changed and the memory layout is
    /// maintained.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: An array specifying the new size of each dimension.
    ///
    /// # Panics
    /// This method will panic if number the elements in the tensor does not match the product of
    /// the new dimensions.
    #[inline]
    pub const fn reshape(&mut self, dimensions: [usize; R]) {
        self.metadata.reshape(dimensions);
    }

    /// Transforms the shape of this tensor with the specified dimensions without reordering values.
    ///
    /// This method allows upgrading or downgrading the rank of the tensor `R` with a new rank `N`.
    ///
    /// This method doesn't reorder values, only the index is changed and the memory layout is
    /// maintained.
    ///
    /// # Parameters
    ///
    /// - `dimensions`: An array specifying the new size of each dimension.
    ///
    /// # Panics
    /// This method will panic if number the elements in the tensor does not match the product of
    /// the new dimensions.
    #[inline]
    pub fn change_rank<const N: usize>(self, dimensions: [usize; N]) -> Tensor<T, N> {
        Tensor {
            metadata: TensorMetaData::new_cmp_eq(self.metadata.size(), dimensions),
            data: unsafe { ManuallyDrop::new(self).data.duplicate() },
        }
    }
}

#[cfg(test)]
mod transformations_tests {
    use super::*;

    #[test]
    fn test_tensor_reshape() {
        let mut tensor = Tensor::from_slice([2, 3], &[1, 2, 3, 4, 5, 6]);

        tensor.reshape([3, 2]);

        assert_eq!(tensor.shape(), &[3, 2]);

        assert_eq!(tensor.get(&[0, 0]), &1);
        assert_eq!(tensor.get(&[0, 1]), &2);
        assert_eq!(tensor.get(&[1, 0]), &3);
        assert_eq!(tensor.get(&[1, 1]), &4);
        assert_eq!(tensor.get(&[2, 0]), &5);
        assert_eq!(tensor.get(&[2, 1]), &6);
    }

    #[test]
    #[should_panic]
    fn test_tensor_reshape_invalid_shape() {
        let mut tensor = Tensor::from_slice([2, 3], &[1, 2, 3, 4, 5, 6]);
        tensor.reshape([3, 1]);
    }

    #[test]
    fn test_change_rank_upgrade() {
        let tensor = Tensor::from_slice([2, 3], &[1, 2, 3, 4, 5, 6]);
        let altered_tensor = tensor.change_rank([2, 3, 1]);

        assert_eq!(altered_tensor.shape(), &[2, 3, 1]);
        assert_eq!(altered_tensor.size(), 6);
        assert_eq!(altered_tensor.get(&[0, 0, 0]), &1);
        assert_eq!(altered_tensor.get(&[1, 2, 0]), &6);
    }

    #[test]
    fn test_change_rank_downgrade() {
        let tensor = Tensor::from_slice([2, 3, 1], &[1, 2, 3, 4, 5, 6]);
        let altered_tensor = tensor.change_rank([2, 3]);

        assert_eq!(altered_tensor.shape(), &[2, 3]);
        assert_eq!(altered_tensor.size(), 6);
        assert_eq!(altered_tensor.get(&[0, 0]), &1);
        assert_eq!(altered_tensor.get(&[1, 2]), &6);
    }

    #[test]
    fn test_change_rank_zero_rank() {
        let tensor = Tensor::from_slice([], &[1]);
        assert_eq!(tensor.shape(), &[]);
        let altered_tensor = tensor.change_rank([1]);
        assert_eq!(altered_tensor.shape(), &[1]);
    }

    #[test]
    #[should_panic]
    fn test_change_rank_invalid_shape() {
        let tensor = Tensor::from_slice([2, 3], &[1, 2, 3, 4, 5, 6]);
        tensor.change_rank([2, 2]);
    }
}
