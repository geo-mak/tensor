use crate::cast::error::CastError;
use crate::cast::traits::TryCast;

//////////////////////////////////////////////////////////////////////
// Cast implementations for f64
//////////////////////////////////////////////////////////////////////

impl TryCast<u8> for f64 {
    fn try_cast(&self) -> Result<u8, CastError> {
        if *self < 0.0 || *self > u8::MAX as f64 {
            return Err(CastError::Overflow);
        }
        if *self != self.trunc() {
            return Err(CastError::PrecisionLoss);
        }
        Ok(*self as u8)
    }
}

impl TryCast<i8> for f64 {
    fn try_cast(&self) -> Result<i8, CastError> {
        if *self < i8::MIN as f64 || *self > i8::MAX as f64 {
            return Err(CastError::Overflow);
        }
        if *self != self.trunc() {
            return Err(CastError::PrecisionLoss);
        }
        Ok(*self as i8)
    }
}

impl TryCast<i16> for f64 {
    fn try_cast(&self) -> Result<i16, CastError> {
        if *self < i16::MIN as f64 || *self > i16::MAX as f64 {
            return Err(CastError::Overflow);
        }
        if *self != self.trunc() {
            return Err(CastError::PrecisionLoss);
        }
        Ok(*self as i16)
    }
}

impl TryCast<u16> for f64 {
    fn try_cast(&self) -> Result<u16, CastError> {
        if *self < 0.0 || *self > u16::MAX as f64 {
            return Err(CastError::Overflow);
        }
        if *self != self.trunc() {
            return Err(CastError::PrecisionLoss);
        }
        Ok(*self as u16)
    }
}

impl TryCast<i32> for f64 {
    fn try_cast(&self) -> Result<i32, CastError> {
        if *self < i32::MIN as f64 || *self > i32::MAX as f64 {
            return Err(CastError::Overflow);
        }
        if *self != self.trunc() {
            return Err(CastError::PrecisionLoss);
        }
        Ok(*self as i32)
    }
}

impl TryCast<u32> for f64 {
    fn try_cast(&self) -> Result<u32, CastError> {
        if *self < 0.0 || *self > u32::MAX as f64 {
            return Err(CastError::Overflow);
        }
        if *self != self.trunc() {
            return Err(CastError::PrecisionLoss);
        }
        Ok(*self as u32)
    }
}

impl TryCast<i64> for f64 {
    fn try_cast(&self) -> Result<i64, CastError> {
        if *self != self.trunc() {
            return Err(CastError::PrecisionLoss);
        }
        Ok(*self as i64)
    }
}

impl TryCast<u64> for f64 {
    fn try_cast(&self) -> Result<u64, CastError> {
        if *self < 0.0 {
            return Err(CastError::Overflow);
        }
        if *self != self.trunc() {
            return Err(CastError::PrecisionLoss);
        }
        Ok(*self as u64)
    }
}

impl TryCast<f32> for f64 {
    fn try_cast(&self) -> Result<f32, CastError> {
        Ok(*self as f32)
    }
}
