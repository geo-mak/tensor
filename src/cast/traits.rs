use crate::cast::error::CastError;

/// Trait for casting self into another type `T`.
/// Types that implement this trait must ensure that casting does not result in precision loss,
/// overflow, or undefined behavior.
pub trait TryCast<T> {
    /// Attempt to cast self into type `T`.
    ///
    /// # Returns
    ///
    /// - `Ok(T)`: If casting is successful.
    /// - `Err(CastError)`: If casting is unsuccessful.
    fn try_cast(&self) -> Result<T, CastError>;
}
