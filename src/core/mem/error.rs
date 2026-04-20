use core::alloc::Layout;

use std::alloc::handle_alloc_error;

#[derive(Clone, Copy, Debug)]
pub enum MemoryError {
    LayoutErr,
    AllocatorErr,
}

#[derive(Clone, Copy)]
pub enum OnError {
    Panic,
    ReturnErr,
}

impl OnError {
    /// Handles `LayoutErr` error according to the current variant.
    #[inline]
    pub const fn layout_err(&self) -> MemoryError {
        match self {
            OnError::Panic => panic!("layout error"),
            OnError::ReturnErr => MemoryError::LayoutErr,
        }
    }

    /// Handles `ServerErr` according to the current variant.
    #[inline]
    pub fn server_err(&self, layout: Layout) -> MemoryError {
        match self {
            OnError::Panic => handle_alloc_error(layout),
            OnError::ReturnErr => MemoryError::AllocatorErr,
        }
    }
}
