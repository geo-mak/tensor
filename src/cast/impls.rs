use crate::core::alloc::UnsafeBufferPointer;
use crate::{CastError, Tensor, TryCast};

impl<T, const R: usize> Tensor<T, R> {
    /// Attempts to cast the tensor into a tensor of a different type without consuming
    /// the original tensor.
    pub fn try_cast<U>(&self) -> Result<Tensor<U, R>, CastError>
    where
        T: TryCast<U>,
    {
        // Note: Current length is assumed to be greater than 0.
        let len = self.metadata.size();
        let data = &self.data;

        unsafe {
            let mut result = UnsafeBufferPointer::new_allocate(len);

            let mut i = 0;
            while i < len {
                match data.load(i).try_cast() {
                    Ok(u_i) => {
                        result.store(i, u_i);
                    }
                    Err(err) => {
                        // Cleanup.
                        result.drop_initialized(i);
                        result.deallocate(len);
                        return Err(err);
                    }
                }
                i += 1;
            }

            let instance = Tensor {
                metadata: self.metadata,
                data: result,
            };

            Ok(instance)
        }
    }
}

#[cfg(test)]
mod casting_tests {
    use crate::{CastError, Tensor};

    #[test]
    fn test_tensor_ops_with_casting() {
        let tensor_int = Tensor::new_set([2, 2], 2);

        // Attempt to cast the tensor to f64
        let tensor_f64 = tensor_int.try_cast::<f64>().unwrap();

        assert_eq!(tensor_int.shape(), tensor_f64.shape());
        assert_eq!(tensor_int.size(), tensor_f64.size());
        assert_eq!(tensor_f64.get(&[0, 0]), &2.0);
    }

    #[test]
    fn test_tensor_cast_overflow() {
        let tensor = Tensor::<u16, 2>::new_set([2, 2], 256);
        let result: Result<Tensor<u8, 2>, CastError> = tensor.try_cast();
        assert_eq!(result.unwrap_err(), CastError::Overflow);
    }

    #[test]
    fn test_tensor_cast_precision_loss() {
        let tensor = Tensor::<f32, 2>::new_set([2, 2], 3.14);
        let result: Result<Tensor<i32, 2>, CastError> = tensor.try_cast();
        assert_eq!(result.unwrap_err(), CastError::PrecisionLoss);
    }
}
