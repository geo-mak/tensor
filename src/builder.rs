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
    ($t_type:ty : $array:expr) => {
        {
            #[allow(unused_imports)]
            use $crate::Tensor;
            use $crate::tensor_builder;
            tensor_builder![$t_type : $array]
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_tensor_builder_macro() {
        let tensor = tensor![i32: [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]];

        assert_eq!(tensor.shape(), &[2, 2, 3]);

        assert_eq!(tensor.get(&[0, 0, 0]), &1);
        assert_eq!(tensor.get(&[0, 0, 1]), &2);
        assert_eq!(tensor.get(&[0, 0, 2]), &3);
        assert_eq!(tensor.get(&[0, 1, 0]), &4);
        assert_eq!(tensor.get(&[0, 1, 1]), &5);
        assert_eq!(tensor.get(&[0, 1, 2]), &6);
        assert_eq!(tensor.get(&[1, 0, 0]), &7);
        assert_eq!(tensor.get(&[1, 0, 1]), &8);
        assert_eq!(tensor.get(&[1, 0, 2]), &9);
        assert_eq!(tensor.get(&[1, 1, 0]), &10);
        assert_eq!(tensor.get(&[1, 1, 1]), &11);
        assert_eq!(tensor.get(&[1, 1, 2]), &12);
    }
}
