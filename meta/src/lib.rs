use proc_macro::TokenStream;

mod parsing;

/// A declarative constructor that creates an instance of `Tensor` from nested arrays.
/// 
/// Type and rank are inferred from the input, but explicit annotation is deterministic regarding
/// the tensor's type and its memory usage.
/// 
/// # Examples
///
/// ```text
/// use tensor::tensor;
///
/// fn main() {
///  //                1D: |---------0----------|  |----------1------------|
///  //                2D: |----0----||----1----|  |----0---|  |-----1-----|
///  //                3D: ||0--1--2|  |0--1--2||  ||0--1--2|  |0----1---2||
///  let tensor = tensor![[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]];
///
///  assert_eq!(tensor.shape(), &[2, 2, 3]);
/// }
/// ```
#[proc_macro]
pub fn tensor(input: TokenStream) -> TokenStream {
    parsing::TensorParser::parse(input)
}