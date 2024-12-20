use crate::cast::error::CastError;
use crate::cast::traits::TryCast;

//////////////////////////////////////////////////////////////////////
// Cast implementations for i16
//////////////////////////////////////////////////////////////////////

impl TryCast<u8> for i16 {
    fn try_cast(&self) -> Result<u8, CastError> {
        if *self < 0 || *self > u8::MAX as i16 {
            return Err(CastError::Overflow);
        }
        Ok(*self as u8)
    }
}

impl TryCast<i8> for i16 {
    fn try_cast(&self) -> Result<i8, CastError> {
        if *self < i8::MIN as i16 || *self > i8::MAX as i16 {
            return Err(CastError::Overflow);
        }
        Ok(*self as i8)
    }
}

impl TryCast<u16> for i16 {
    fn try_cast(&self) -> Result<u16, CastError> {
        if *self < 0 {
            return Err(CastError::Overflow);
        }
        Ok(*self as u16)
    }
}

impl TryCast<i32> for i16 {
    fn try_cast(&self) -> Result<i32, CastError> {
        Ok(*self as i32)
    }
}

impl TryCast<u32> for i16 {
    fn try_cast(&self) -> Result<u32, CastError> {
        if *self < 0 {
            return Err(CastError::Overflow);
        }
        Ok(*self as u32)
    }
}

impl TryCast<i64> for i16 {
    fn try_cast(&self) -> Result<i64, CastError> {
        Ok(*self as i64)
    }
}

impl TryCast<u64> for i16 {
    fn try_cast(&self) -> Result<u64, CastError> {
        if *self < 0 {
            return Err(CastError::Overflow);
        }
        Ok(*self as u64)
    }
}

impl TryCast<f32> for i16 {
    fn try_cast(&self) -> Result<f32, CastError> {
        Ok(*self as f32)
    }
}

impl TryCast<f64> for i16 {
    fn try_cast(&self) -> Result<f64, CastError> {
        Ok(*self as f64)
    }
}

//////////////////////////////////////////////////////////////////////
// Cast implementations for u16
//////////////////////////////////////////////////////////////////////

impl TryCast<u8> for u16 {
    fn try_cast(&self) -> Result<u8, CastError> {
        if *self > u8::MAX as u16 {
            return Err(CastError::Overflow);
        }
        Ok(*self as u8)
    }
}

impl TryCast<i8> for u16 {
    fn try_cast(&self) -> Result<i8, CastError> {
        if *self > i8::MAX as u16 {
            return Err(CastError::Overflow);
        }
        Ok(*self as i8)
    }
}

impl TryCast<i16> for u16 {
    fn try_cast(&self) -> Result<i16, CastError> {
        if *self > i16::MAX as u16 {
            return Err(CastError::Overflow);
        }
        Ok(*self as i16)
    }
}

impl TryCast<i32> for u16 {
    fn try_cast(&self) -> Result<i32, CastError> {
        Ok(*self as i32)
    }
}

impl TryCast<u32> for u16 {
    fn try_cast(&self) -> Result<u32, CastError> {
        Ok(*self as u32)
    }
}

impl TryCast<i64> for u16 {
    fn try_cast(&self) -> Result<i64, CastError> {
        Ok(*self as i64)
    }
}

impl TryCast<u64> for u16 {
    fn try_cast(&self) -> Result<u64, CastError> {
        Ok(*self as u64)
    }
}

impl TryCast<f32> for u16 {
    fn try_cast(&self) -> Result<f32, CastError> {
        Ok(*self as f32)
    }
}

impl TryCast<f64> for u16 {
    fn try_cast(&self) -> Result<f64, CastError> {
        Ok(*self as f64)
    }
}
