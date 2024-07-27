/// Trait for casting from another type into self
/// Types that implement this trait must ensure that casting does not result in precision loss, overflow, or undefined behavior.
pub trait Cast<T> {
    fn cast(value: T) -> Self
    where
        Self: Sized;
}

// Implementations for casting from i32 to other types
impl Cast<i32> for f64 {
    fn cast(value: i32) -> f64 {
        value as f64
    }
}

impl Cast<i32> for f32 {
    fn cast(value: i32) -> f32 {
        value as f32
    }
}

impl Cast<i32> for i64 {
    fn cast(value: i32) -> i64 {
        value as i64
    }
}

// Implementations for casting from f64 to integer types
impl Cast<f64> for i32 {
    fn cast(value: f64) -> i32 {
        if value.fract() != 0.0 {
            panic!("Cannot cast f64 to i32: fractional part is non-zero");
        }
        if value < i32::MIN as f64 || value > i32::MAX as f64 {
            panic!("Cannot cast f64 to i32: out of bounds");
        }
        value as i32
    }
}

impl Cast<f64> for i64 {
    fn cast(value: f64) -> i64 {
        if value.fract() != 0.0 {
            panic!("Cannot cast f64 to i64: fractional part is non-zero");
        }
        if value < i64::MIN as f64 || value > i64::MAX as f64 {
            panic!("Cannot cast f64 to i64: out of bounds");
        }
        value as i64
    }
}

// Implementations for casting from f64 to unsigned integer types
impl Cast<f64> for u32 {
    fn cast(value: f64) -> u32 {
        if value < 0.0 {
            panic!("Cannot cast negative f64 to unsigned u32");
        }
        if value.fract() != 0.0 {
            panic!("Cannot cast fractional f64 to unsigned u32");
        }
        if value > u32::MAX as f64 {
            panic!("Cannot cast f64 to u32: out of bounds");
        }
        value as u32
    }
}

impl Cast<f64> for u64 {
    fn cast(value: f64) -> u64 {
        if value < 0.0 {
            panic!("Cannot cast negative f64 to unsigned u64");
        }
        if value.fract() != 0.0 {
            panic!("Cannot cast fractional f64 to unsigned u64");
        }
        if value > u64::MAX as f64 {
            panic!("Cannot cast f64 to u64: out of bounds");
        }
        value as u64
    }
}

// Implementations for casting from f32 to integer types
impl Cast<f32> for i32 {
    fn cast(value: f32) -> i32 {
        if value.fract() != 0.0 {
            panic!("Cannot cast f32 to i32: fractional part is non-zero");
        }
        if value < i32::MIN as f32 || value > i32::MAX as f32 {
            panic!("Cannot cast f32 to i32: out of bounds");
        }
        value as i32
    }
}

impl Cast<f32> for i64 {
    fn cast(value: f32) -> i64 {
        if value.fract() != 0.0 {
            panic!("Cannot cast f32 to i64: fractional part is non-zero");
        }
        if value < i64::MIN as f32 || value > i64::MAX as f32 {
            panic!("Cannot cast f32 to i64: out of bounds");
        }
        value as i64
    }
}

// Implementations for casting from f32 to unsigned integer types
impl Cast<f32> for u32 {
    fn cast(value: f32) -> u32 {
        if value < 0.0 {
            panic!("Cannot cast negative f32 to unsigned u32");
        }
        if value.fract() != 0.0 {
            panic!("Cannot cast fractional f32 to unsigned u32");
        }
        if value > u32::MAX as f32 {
            panic!("Cannot cast f32 to u32: out of bounds");
        }
        value as u32
    }
}

impl Cast<f32> for u64 {
    fn cast(value: f32) -> u64 {
        if value < 0.0 {
            panic!("Cannot cast negative f32 to unsigned u64");
        }
        if value.fract() != 0.0 {
            panic!("Cannot cast fractional f32 to unsigned u64");
        }
        if value > u64::MAX as f32 {
            panic!("Cannot cast f32 to u64: out of bounds");
        }
        value as u64
    }
}

// Implementations for casting from i64 to other types
impl Cast<i64> for i32 {
    fn cast(value: i64) -> i32 {
        if value < i32::MIN as i64 || value > i32::MAX as i64 {
            panic!("Cannot cast i64 to i32: out of bounds");
        }
        value as i32
    }
}

impl Cast<i64> for f32 {
    fn cast(value: i64) -> f32 {
        value as f32
    }
}

impl Cast<i64> for f64 {
    fn cast(value: i64) -> f64 {
        value as f64
    }
}

impl Cast<i64> for u32 {
    fn cast(value: i64) -> u32 {
        if value < 0 {
            panic!("Cannot cast negative i64 to unsigned u32");
        }
        if value > u32::MAX as i64 {
            panic!("Cannot cast i64 to u32: out of bounds");
        }
        value as u32
    }
}

impl Cast<i64> for u64 {
    fn cast(value: i64) -> u64 {
        if value < 0 {
            panic!("Cannot cast negative i64 to unsigned u64");
        }
        if value > u64::MAX as i64 {
            panic!("Cannot cast i64 to u64: out of bounds");
        }
        value as u64
    }
}

// Implementations for casting from u32 to other types
impl Cast<u32> for i32 {
    fn cast(value: u32) -> i32 {
        if value > i32::MAX as u32 {
            panic!("Cannot cast u32 to i32: out of bounds");
        }
        value as i32
    }
}

impl Cast<u32> for f32 {
    fn cast(value: u32) -> f32 {
        value as f32
    }
}

impl Cast<u32> for f64 {
    fn cast(value: u32) -> f64 {
        value as f64
    }
}

impl Cast<u32> for i64 {
    fn cast(value: u32) -> i64 {
        value as i64
    }
}

impl Cast<u32> for u64 {
    fn cast(value: u32) -> u64 {
        value as u64
    }
}

// Implementations for casting from u64 to other types
impl Cast<u64> for i32 {
    fn cast(value: u64) -> i32 {
        if value > i32::MAX as u64 {
            panic!("Cannot cast u64 to i32: out of bounds");
        }
        value as i32
    }
}

impl Cast<u64> for f32 {
    fn cast(value: u64) -> f32 {
        value as f32
    }
}

impl Cast<u64> for f64 {
    fn cast(value: u64) -> f64 {
        value as f64
    }
}

impl Cast<u64> for i64 {
    fn cast(value: u64) -> i64 {
        value as i64
    }
}

impl Cast<u64> for u32 {
    fn cast(value: u64) -> u32 {
        if value > u32::MAX as u64 {
            panic!("Cannot cast u64 to u32: out of bounds");
        }
        value as u32
    }
}
