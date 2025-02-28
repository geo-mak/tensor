use proc_macro::TokenStream;

mod parsing;

/// Compile-time tensor builder.
///
/// ## Input Patterns
/// `[Nested Arrays]`: `[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]`
#[proc_macro]
pub fn tensor_builder(input: TokenStream) -> TokenStream {
    parsing::TensorParser::parse(input)
}