#[cfg(test)]
mod tests {
    use crate::cast::CastError;
    use crate::tensor::Tensor;

    #[test]
    fn test_tensor_new_set_get() {
        let mut tensor = Tensor::new_set(vec![2, 3], 0);

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

    #[test]
    fn test_tensor_new_with_values() {
        let tensor = Tensor::with_values(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4]);
        assert_eq!(tensor.shape(), &[2, 4]);
        assert_eq!(tensor.get(&[1, 2]), &7);
    }

    #[test]
    #[should_panic = "Invalid shape: Values' count is 8, but dimensions' size is 10"]
    fn test_tensor_new_with_values_error() {
        let _ = Tensor::with_values(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 5]);
    }

    #[test]
    fn test_tensor_access_index() {
        let mut tensor = Tensor::new_set(vec![2, 3], 0);
        tensor[&[0, 2]] = 100;
        assert_eq!(tensor[&[0, 2]], 100);
    }

    #[test]
    fn test_tensor_reshape() {
        let mut tensor = Tensor::new_set(vec![2, 3], 0);
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
    fn test_tensor_ops_with_casting() {
        let tensor1 = Tensor::new_set(vec![2, 2], 1.0);
        let tensor2 = Tensor::new_set(vec![2, 2], 2);

        // Attempt to cast the tensor to f64
        let tensor2_f64 = tensor2.try_cast::<f64>().unwrap();

        // Core tensor operations
        let result_add = tensor1.add(&tensor2_f64);
        let result_sub = tensor1.sub(&tensor2_f64);
        let result_mul = tensor1.mul(&tensor2_f64);
        let result_div = tensor1.div(&tensor2_f64);

        // Expected results
        let expected_add = Tensor::new_set(vec![2, 2], 3.0);
        let expected_sub = Tensor::new_set(vec![2, 2], -1.0);
        let expected_mul = Tensor::new_set(vec![2, 2], 2.0);
        let expected_div = Tensor::new_set(vec![2, 2], 0.5);

        assert_eq!(result_add, expected_add);
        assert_eq!(result_sub, expected_sub);
        assert_eq!(result_mul, expected_mul);
        assert_eq!(result_div, expected_div);
    }

    #[test]
    fn test_tensor_cast_overflow() {
        let tensor = Tensor::<u16>::new_set(vec![2, 2], 256);
        let result: Result<Tensor<u8>, CastError> = tensor.try_cast();
        assert_eq!(result.unwrap_err(), CastError::Overflow);
    }

    #[test]
    fn test_tensor_cast_precision_loss() {
        let tensor = Tensor::<f32>::new_set(vec![2, 2], 3.14);
        let result: Result<Tensor<i32>, CastError> = tensor.try_cast();
        assert_eq!(result.unwrap_err(), CastError::PrecisionLoss);
    }

    #[test]
    fn test_tensor_display() {
        let mut tensor = Tensor::new_set(vec![2, 2, 2], 0);
        tensor.set(&[0, 0, 0], 1);
        tensor.set(&[0, 0, 1], 2);
        tensor.set(&[0, 1, 0], 3);
        tensor.set(&[0, 1, 1], 4);
        tensor.set(&[1, 0, 0], 5);
        tensor.set(&[1, 0, 1], 6);
        tensor.set(&[1, 1, 0], 7);
        tensor.set(&[1, 1, 1], 8);

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
}
