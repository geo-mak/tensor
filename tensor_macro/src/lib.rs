use proc_macro::TokenStream;
use syn::{parse_macro_input, Expr, ExprArray, Type, Token};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use quote::quote;

// Input.
struct TensorMacroInput {
    ty: Type,
    array: ExprArray,
}

impl Parse for TensorMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ty: Type = input.parse()?;
        input.parse::<Token![:]>()?;
        let array: ExprArray = input.parse()?;
        Ok(TensorMacroInput { ty, array })
    }
}

// Intermediate types.
type ItemsSequence = Punctuated<Expr, Token![,]>;
type Values = Vec<proc_macro2::TokenStream>;
type Dimensions = Vec<usize>;

// Result.
struct ParsedTensor {
    data: Values,
    dimensions: Dimensions,
}

fn parse_nested_array(items: &ItemsSequence) -> ParsedTensor {
    let mut data = Vec::new();
    let mut dimensions = Vec::new();
    parse_array(items, &mut data, &mut dimensions, 0);
    ParsedTensor {
        data, dimensions,
    }
}

/// Recursively parses nested arrays.
fn parse_array(
    items: &ItemsSequence,
    values: &mut Values,
    dimensions: &mut Dimensions,
    level: usize, 
) {
    if level >= dimensions.len() {
        dimensions.push(items.len());
    } else if dimensions[level] < items.len() {
        dimensions[level] = items.len();
    }
    
    for item in items {
        match item {
            Expr::Array(inner_array) => {
                parse_array(&inner_array.elems, values, dimensions, level + 1);
            },
            Expr::Lit(value_expr) => {
                let value = match &value_expr.lit {
                    syn::Lit::Int(int) => int.token(),
                    syn::Lit::Float(float) => float.token(),
                    _ => unimplemented!("Unsupported value type"),
                };
                values.push(quote!(#value));
            }
            _ => {}
        }
    }
}

/// Compile-time tensor builder.
/// 
/// ## Input Patterns
/// `[Numeric Type: Nested Arrays]`: `[i32: [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]`
#[proc_macro]
pub fn tensor_builder(input: TokenStream) -> TokenStream {
    let TensorMacroInput { ty, array } = parse_macro_input!(input as TensorMacroInput);
    let ParsedTensor {data, dimensions} = parse_nested_array(&array.elems);
    let instance_tokens = quote! {
        Tensor::<#ty>::with_data(vec![#(#data),*], vec![#(#dimensions),*])
    };
    TokenStream::from(instance_tokens)
}