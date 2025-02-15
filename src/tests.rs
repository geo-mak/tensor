#[cfg(test)]
mod tests {
    use crate::cast::CastError;
    use crate::tensor::Tensor;

    // Test setting and getting tensor values
    #[test]
    fn test_tensor_new_set_get() {
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

    #[test]
    fn test_tensor_new_with_values() {
        let tensor = Tensor::with_values(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4]);
        assert_eq!(tensor.shape(), &[2, 4]);
        assert_eq!(tensor.get(&[1, 2]), &7);
    }

    #[test]
    #[should_panic = "Dimensions's size doesn't match values' count"]
    fn test_tensor_new_with_values_error() {
        let _ = Tensor::with_values(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 5]);
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
