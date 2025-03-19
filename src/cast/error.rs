/// Error type for casting operations.
/// This enum is used to represent the different types of errors that can occur during casting.
/// The following errors are defined:
/// - `Overflow`: The result of the casting operation is too large to be represented by the target type.
/// - `PrecisionLoss`: The casting operation results in a loss of precision.
#[derive(Debug, PartialEq)]
pub enum CastError {
    Overflow,
    PrecisionLoss,
}
