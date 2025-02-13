use proc_macro::TokenStream;

mod syntax;

/// Compile-time tensor builder.
///
/// ## Input Patterns
/// `[Numeric Type: Nested Arrays]`: `[i32: [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]`
#[proc_macro]
pub fn tensor_builder(input: TokenStream) -> TokenStream {
    syntax::transform(input)
}