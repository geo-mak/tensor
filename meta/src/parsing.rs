use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};
use proc_macro::token_stream::IntoIter;

/// Syntax:
/// ```text
/// Array       <- '[' Elements ']'
/// Elements    <- '[' ElementList ']'
/// ElementList <- Element (',' Element)+
/// Element     <- (!Elements Number) / (!Number Elements)
/// Number      <- '-'? Digit+ ('.' Digit+)?
/// Digit       <- [0-9]
/// ```
pub(crate) struct TensorParser;

impl TensorParser {
    pub(crate) fn parse(input: TokenStream) -> TokenStream {
        let mut stream = input.into_iter();
        let group = Self::match_input(&mut stream);
        let mut values = Values(Vec::new());
        let mut dimensions = Dimensions(Vec::new());
        Self::parse_group(0, &group, &mut values, &mut dimensions);
        Self::generate(values, dimensions)
    }

    fn match_input(stream: &mut IntoIter) -> Group {
        if let Some(TokenTree::Group(group)) = stream.next() {
            if group.delimiter() == Delimiter::None {
                let mut stream = group.stream().into_iter();
                if let Some(TokenTree::Group(inner)) = stream.next() {
                    if inner.delimiter() == Delimiter::Bracket {
                        return inner;
                    }
                }
            } else if group.delimiter() == Delimiter::Bracket {
                return group
            }
        }
        diagnostics::expected_array_expr()
    }

    /// Recursively parses nested arrays.
    fn parse_group(
        level: usize,
        group: &Group,
        values: &mut Values,
        dimensions: &mut Dimensions,
    ) {
        // Type checking of literals is left to the type system, no enforcement here,
        // because literals' attributes are not accessible, so parsing literal as string is the
        // only way to validate types, and this is just not worth it, at least for now.
        let span= &group.span();
        let mut state = GroupState::new();
        let mut stream = group.stream().into_iter();
        while let Some(member) = stream.next() {
            match member {
                TokenTree::Group(ref g) => {
                    if g.delimiter() != Delimiter::Bracket {
                        diagnostics::invalid_delimiter(&g.span())
                    }
                    Self::parse_group(level + 1, g, values, dimensions);
                    state.add(2)
                }
                TokenTree::Literal(lit) => {
                    values.0.push(Value::Literal(lit));
                    state.add(1)
                }
                TokenTree::Punct(ref p)
                if p.as_char() == '-' => {
                    if let Some(TokenTree::Literal(lit)) = stream.next() {
                        values.0.push(Value::SignedLiteral(lit));
                        state.add(1)
                    } else {
                        diagnostics::unexpected_token(&member, span)
                    }
                }
                _ => diagnostics::unexpected_token(&member, span)
            }
            Self::match_sep(span, &mut stream);
        };

        match state.kind {
            0 => diagnostics::empty_array(level + 1),
            3 => diagnostics::expected_scalars(span),
            _ => Self::update_dimensions(level, span, state, dimensions)
        }
    }

    fn match_sep(span: &Span, stream: &mut IntoIter) {
        match stream.next() {
            Some(TokenTree::Punct(p)) if p.as_char() == ',' => { /* Go ahead */ },
            None => { /* End of tokens */ },
            _ => diagnostics::missing_sep(span)
        }
    }

    fn update_dimensions(level: usize, span: &Span, state: GroupState, dims: &mut Dimensions) {
        if level >= dims.0.len() {
            // Allocate for all discovered dimensions at once.
            dims.0.resize_with(level + 1, GroupState::new)
        }
        let prev_state = &mut dims.0[level];
        // Update undecided state once per dimension with the first valid observed state.
        if prev_state.kind == 0 {
            *prev_state = state;
        } else if prev_state.ne(&state) {
            diagnostics::inhomogeneous_shape(level + 1, prev_state, &state, span);
        }
    }

    fn generate(values: Values, dimensions: Dimensions) -> TokenStream {
        let tokens = [
            TokenTree::Ident(Ident::new("Tensor", Span::call_site())),
            TokenTree::Punct(Punct::new(':', Spacing::Joint)),
            TokenTree::Punct(Punct::new(':', Spacing::Joint)),
            TokenTree::Ident(Ident::new("from_slice", Span::call_site())),
            TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                {
                    let param_tokens = [
                        TokenTree::Group(Group::new(Delimiter::Bracket, dimensions.into_stream())), 
                        TokenTree::Punct(Punct::new(',', Spacing::Alone)),
                        TokenTree::Punct(Punct::new('&', Spacing::Joint)),
                        TokenTree::Group(Group::new(Delimiter::Bracket, values.into_stream())),
                    ];
                    TokenStream::from_iter(param_tokens)
                },
            )),
        ];
        TokenStream::from_iter(tokens)
    }
}

struct GroupState {
    len: usize,
    kind: u8,
}

impl GroupState {
    fn new() -> Self{
        Self {
            len: 0,
            // Undecided.
            kind: 0b00
        }
    }

    fn add(&mut self, kind: u8) {
        self.kind |= kind;
        self.len += 1;
    }

    fn ne(&self, other: &GroupState) -> bool {
        self.len != other.len || self.kind != other.kind
    }

    fn describe(&self) -> &'static str {
        match self.kind {
            1 => "scalars",
            2 => "arrays",
            _ => unreachable!()
        }
    }
}

enum Value {
    Literal(Literal),
    SignedLiteral(Literal)
}

struct Values(Vec<Value>);

impl Values {
    /// Consumes `Values` and generates the resulting values as `TokenStream`.
    fn into_stream(self) -> TokenStream {
        let mut stream = TokenStream::new();
        for (i, value) in self.0.into_iter().enumerate() {
            if i > 0 {
                stream.extend([
                    TokenTree::Punct(Punct::new(',', Spacing::Alone))
                ]);
            }
            match value {
                Value::Literal(literal) => stream.extend([
                    TokenTree::Literal(literal)
                ]),
                Value::SignedLiteral(literal) => {
                    stream.extend([
                        TokenTree::Punct(Punct::new('-', Spacing::Joint)),
                        TokenTree::Literal(literal),
                    ]);
                }
            }
        }
        stream
    }
}

struct Dimensions(Vec<GroupState>);

impl Dimensions {
    /// Consumes `Dimensions` and generates the resulting dimensions as `TokenStream`.
    fn into_stream(self) -> TokenStream {
        let mut stream = TokenStream::new();
        for (i, dimension) in self.0.into_iter().enumerate() {
            if i > 0 {
                stream.extend([
                    TokenTree::Punct(Punct::new(',', Spacing::Alone))
                ]);
            }
            stream.extend([
                TokenTree::Literal(Literal::usize_unsuffixed(dimension.len))
            ]);
        }
        stream
    }
}

mod diagnostics {
    use super::*;

    fn parse_span(span: &Span) -> String {
        // Start line and column are unstable APIs in `proc_macro::Span` at the moment.
        if let Some(str) = span.source_text() {
            return str
        }
        String::new()
    }

    pub(super) fn expected_array_expr() -> ! {
        panic!("Syntax error: expected array expression '[...]'")
    }

    pub(super) fn invalid_delimiter(span: &Span) -> ! {
        panic!(
            "Syntax error: arrays must be delimited with square brackets '[' ']'.\nLocation: {}",
            parse_span(span)
        )
    }

    pub(super) fn missing_sep(span: &Span) -> ! {
        panic!(
            "Syntax error: missing separator ','.\nLocation: {}",
            parse_span(span)
        )
    }

    pub(super) fn unexpected_token(token: &TokenTree, span: &Span) -> ! {
        panic!(
            "Syntax error: unexpected token `{}`.\nLocation: {}",
            token,
            parse_span(span)
        )
    }

    pub(super) fn expected_scalars(span: &Span) -> ! {
        panic!(
            "Invalid array: expected scalars as elements but found array.\nLocation: {}",
            parse_span(span)
        );
    }

    pub(super) fn empty_array(dim: usize) -> ! {
        panic!("Invalid array: an array in dimension {} is empty", dim)
    }

    pub(super) fn inhomogeneous_shape(
        dim: usize, expected: &GroupState, found: &GroupState, span: &Span
    ) -> ! {
        panic!(
            "Inhomogeneous tensor: expected {} {} in dimension {}, but found {} {}.\nLocation: {}",
            expected.len,
            expected.describe(),
            dim,
            found.len,
            found.describe(),
            parse_span(span)
        );
    }
}