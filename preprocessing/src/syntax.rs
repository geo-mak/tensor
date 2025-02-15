use proc_macro::TokenStream;
use proc_macro2::Literal;
use quote::quote;
use syn::{Expr, ExprArray, Type, Token, parse_macro_input};
use syn::Lit::{Float, Int};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;

pub(crate) fn transform(input: TokenStream) -> TokenStream {
    let NestedArraysPattern { ty, array } = parse_macro_input!(input as NestedArraysPattern);
    let ParsedTensor {data, dimensions} = parse_nested_array(&array.elems);
    let instance_tokens = quote! {
        Tensor::<#ty>::with_values(vec![#(#data),*], vec![#(#dimensions),*])
    };
    TokenStream::from(instance_tokens)
}

// Input.
struct NestedArraysPattern { 
    ty: Type, 
    array: ExprArray,
}

impl Parse for NestedArraysPattern {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ty: Type = input.parse()?;
        input.parse::<Token![:]>()?;
        let array: ExprArray = input.parse()?;
        Ok(NestedArraysPattern { ty, array })
    }
}

type ExprSequence = Punctuated<Expr, Token![,]>;
type Values = Vec<Literal>;
type Dimensions = Vec<usize>;

// Result.
struct ParsedTensor {
    data: Values,
    dimensions: Dimensions,
}

fn parse_nested_array(items: &ExprSequence) -> ParsedTensor {
    let mut data = Vec::new();
    let mut dimensions = Vec::new();
    parse_array(items, &mut data, &mut dimensions, 0);
    ParsedTensor {
        data, dimensions,
    }
}

// Recursively parses nested arrays.
fn parse_array(
    expressions: &ExprSequence,
    values: &mut Values,
    dimensions: &mut Dimensions,
    level: usize,
) {
    let level_span = expressions.span();
    let level_len = expressions.len();

    if level >= dimensions.len() {
        dimensions.push(level_len);
    } else if dimensions[level] != level_len {
        panic!(
            "Syntax error at {}:{}: expected length {} at level {}, but found {}",
            level_span.start().line,
            level_span.start().column,
            dimensions[level],
            level,
            level_len
        );
    }

    let mut is_array = false;
    let mut is_literal = false;

    for expr in expressions {
        match expr {
            Expr::Array(array_expr) => {
                is_array = true;
                parse_array(&array_expr.elems, values, dimensions, level + 1);
            }
            Expr::Lit(literal_expr) => {
                is_literal = true;
                let literal = match &literal_expr.lit {
                    Int(int) => int.token(),
                    Float(float) => float.token(),
                    _ => panic!(
                        "Value error at {}:{}: unsupported value type",
                        level_span.start().line,
                        level_span.start().column
                    ),
                };
                values.push(literal);
            }
            _ => {}
        }
    }

    if is_array && is_literal {
        panic!(
            "Syntax error at {}:{}: found both arrays and scalars at level {}",
            level_span.start().line,
            level_span.start().column,
            level
        );
    }
}