/// A builder macro that creates `Tensor` from nested arrays.
///
/// # Examples
///
/// ```
/// use tensor::tensor;
///
/// fn main() {
///  //                     1D: |---------0----------|  |----------1------------|
///  //                     2D: |----0----||----1----|  |----0---|  |-----1-----|
///  //                     3D: ||0--1--2|  |0--1--2||  ||0--1--2|  |0----1---2||
///  let tensor = tensor![i32: [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]];
///
///  assert_eq!(tensor.shape(), &[2, 2, 3]);
/// }
/// ```
#[macro_export]
macro_rules! tensor {
    ($ty:ty : $array:expr) => {{
        use $crate::tensor_builder;
        #[allow(unused_imports)]
        use $crate::Tensor;
        let tensor: $crate::Tensor<$ty> = tensor_builder!($array);
        tensor
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_tensor_builder_macro() {
        let tensor = tensor![f32: [
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-7.0, -8.0, -9.0], [-10.0, -11.0, -12.0]]
        ]];

        assert_eq!(tensor.shape(), &[2, 2, 3]);

        assert_eq!(tensor.get(&[0, 0, 0]), &-1.0);
        assert_eq!(tensor.get(&[0, 0, 1]), &-2.0);
        assert_eq!(tensor.get(&[0, 0, 2]), &-3.0);
        assert_eq!(tensor.get(&[0, 1, 0]), &-4.0);
        assert_eq!(tensor.get(&[0, 1, 1]), &-5.0);
        assert_eq!(tensor.get(&[0, 1, 2]), &-6.0);
        assert_eq!(tensor.get(&[1, 0, 0]), &-7.0);
        assert_eq!(tensor.get(&[1, 0, 1]), &-8.0);
        assert_eq!(tensor.get(&[1, 0, 2]), &-9.0);
        assert_eq!(tensor.get(&[1, 1, 0]), &-10.0);
        assert_eq!(tensor.get(&[1, 1, 1]), &-11.0);
        assert_eq!(tensor.get(&[1, 1, 2]), &-12.0);
    }
}
