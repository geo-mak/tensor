use core::hint::unreachable_unchecked;

use crate::mem::error::OnError;
use crate::mem::pointers::UnmanagedPointer;

use crate::{CastError, Tensor, TryCast};

impl<T, U, const R: usize> TryCast<Tensor<U, R>> for Tensor<T, R>
where
    T: TryCast<U>,
{
    /// Attempts to cast the tensor into a tensor of a different type without consuming
    /// the original tensor.
    fn try_cast(&self) -> Result<Tensor<U, R>, CastError> {
        // Safety: Length is assumed to be greater than 0.
        let len = self.metadata.size();
        let data = &self.data;

        let mut output = UnmanagedPointer::<U>::new();

        // Safety: Layout must be checked because the size of U * len might overflow.
        let layout = match unsafe { output.layout_of(len, OnError::Panic) } {
            Ok(layout) => layout,
            Err(_) => unsafe { unreachable_unchecked() },
        };

        match unsafe { output.acquire(layout, OnError::Panic) } {
            Ok(_) => (),
            Err(_) => unsafe { unreachable_unchecked() },
        };

        let mut i = 0;

        while i < len {
            let ref_i = unsafe { data.reference(i) };

            match ref_i.try_cast() {
                Ok(u_i) => unsafe { output.store(i, u_i) },
                Err(err) => {
                    // Cleanup.
                    unsafe {
                        output.drop_initialized(i);
                        output.release(layout);
                    }

                    return Err(err);
                }
            }

            i += 1;
        }

        let instance = Tensor {
            metadata: self.metadata,
            data: output,
        };

        Ok(instance)
    }
}

#[cfg(test)]
mod casting_tests {
    use super::*;

    #[test]
    fn test_tensor_ops_with_casting() {
        let tensor_int = Tensor::new_set([2, 2], 2);

        let tensor_f64: Tensor<f64, 2> = tensor_int.try_cast().unwrap();

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
