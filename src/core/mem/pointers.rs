use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::Range;
use core::ptr;

use std::alloc::{self, alloc};

use crate::core::mem::error::{MemoryError, OnError};
use crate::core::opt::OnDrop;
use crate::core::opt::branch_hints::likely;

/// Debug-mode check for the valid alignment.
/// This function is only available in debug builds.
///
/// Conditions:
///
/// - `align` of `T` must not be zero.
///
/// - `align` of `T` must be a power of two.
#[cfg(debug_assertions)]
const fn debug_assert_valid_alignment(align: usize) {
    assert!(align.is_power_of_two(), "Alignment must be a power of two");
}

/// Debug-mode check for the valid allocation size.
/// This function is only available in debug builds.
///
/// Conditions:
///
/// - `size` must be greater than `0`.
#[cfg(debug_assertions)]
const fn debug_assert_non_zero_size(size: usize) {
    assert!(size > 0, "Allocation size must be greater than 0");
}

/// Debug-mode check for the layout size and alignment.
/// This function is only available in debug builds.
///
/// Conditions:
///
/// - `align` of `T` must not be zero.
///
/// - `align` of `T` must be a power of two.
///
/// - `size` must be greater than `0`.
///
/// - `size`, when rounded up to the nearest multiple of `align`, must be less than or
///   equal to `isize::MAX`.
///
#[cfg(debug_assertions)]
const fn debug_assert_valid_layout(size: usize, align: usize) {
    debug_assert_valid_alignment(align);
    debug_assert_non_zero_size(size);
    assert!(
        ((isize::MAX as usize + 1) - align) >= size,
        "Allocation size exceeds maximum limit on this platform"
    );
}

/// Debug-mode check to check the allocation state.
/// This function is only available in debug builds.
///
/// Conditions:
///
/// - The pointer must not be null.
///
#[cfg(debug_assertions)]
const fn debug_assert_not_null<T>(instance: &UnmanagedPointer<T>) {
    assert!(!instance.ptr.is_null(), "Pointer must not be null");
}

/// Debug-mode check to check the allocation state.
/// This function is only available in debug builds.
///
/// Conditions:
///
/// - The pointer must be null.
///
#[cfg(debug_assertions)]
const fn debug_assert_is_null<T>(instance: &UnmanagedPointer<T>) {
    assert!(instance.ptr.is_null(), "Pointer must be null");
}

/// An indirect reference to _one or more_ values of type `T` consecutively in memory,
/// with set of methods for accessing and managing memory directly.
///
/// It extends the common pointer operations with more operations that simplify the development of custom data structures.
///  
/// It doesn't store any metadata about its allocated memory, such as the size of
/// the allocated memory and the number of initialized elements, therefore it doesn't provide
/// checked operations or automatic memory management.
///
/// Limited checks for invariants are done in debug mode only.
///
/// It uses the registered `#[global_allocator]` to allocate memory.
///
/// Using custom allocators will be supported in the future.
///
/// **Note**: Unwind-safety is limited to specific functions. Not all functions are unwind-safe.
pub struct UnmanagedPointer<T> {
    ptr: *mut T,
    _t: PhantomData<T>,
}

impl<T> UnmanagedPointer<T> {
    pub const T_SIZE: usize = size_of::<T>();
    pub const T_ALIGN: usize = align_of::<T>();
    pub const MAX_LAYOUT_SIZE: usize = (isize::MAX as usize + 1) - Self::T_ALIGN;

    /// Creates a new pointer set to `null`.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        UnmanagedPointer {
            ptr: ptr::null_mut(),
            _t: PhantomData,
        }
    }

    /// Creates a new instance from slice.
    ///
    /// # Safety
    ///
    /// - The length of `slice` must be greater than `0`.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is length of the slice.
    #[must_use]
    #[inline]
    pub unsafe fn from_slice(slice: &[T], on_err: OnError) -> Result<Self, MemoryError>
    where
        T: Copy,
    {
        let len = slice.len();

        let mut instance = UnmanagedPointer::<T>::new();

        unsafe {
            let layout = instance.layout_unchecked_of(len);

            instance.acquire(layout, on_err)?;

            instance.ptr.copy_from_nonoverlapping(slice.as_ptr(), len);
        }

        Ok(instance)
    }

    /// Creates a new instance from vector.
    ///
    /// # Safety
    ///
    /// The allocation size of `vec` must be greater than `0`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    #[must_use]
    #[inline(always)]
    pub unsafe fn from_vec(vec: Vec<T>) -> Self {
        #[cfg(debug_assertions)]
        debug_assert_valid_layout(vec.capacity() * Self::T_SIZE, Self::T_ALIGN);

        UnmanagedPointer {
            ptr: ManuallyDrop::new(vec).as_mut_ptr(),
            _t: PhantomData,
        }
    }

    /// Checks if the pointer is `null`.
    #[must_use]
    #[inline(always)]
    pub const fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    /// Returns an instance with copy of the base pointer.
    ///
    /// # Safety
    ///
    /// The returned instance might be `null`.
    ///
    #[must_use]
    #[inline(always)]
    pub const unsafe fn duplicate(&mut self) -> UnmanagedPointer<T> {
        UnmanagedPointer {
            ptr: self.ptr,
            _t: PhantomData,
        }
    }

    /// Constructs the memory layout for the specified `count` of type `T` in unchecked-mode.
    ///
    /// This method doesn't check for overflow and checks for valid size and alignment in debug
    /// mode only.
    ///
    /// The _resulted size_ must be greater than `0` for whatever reason, this implies that
    /// `T` can't be `ZST`, and the alignment must be power of 2, which implies it can't be zero
    /// also.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn layout_unchecked_of(&self, count: usize) -> Layout {
        unsafe {
            // Checked in debug-mode for overflow as part of Rust's assert_unsafe_precondition.
            let size = count.unchecked_mul(Self::T_SIZE);

            // More constrained size check.
            #[cfg(debug_assertions)]
            debug_assert_valid_layout(size, Self::T_ALIGN);

            // Also checked in debug-mode by assert_unsafe_precondition.
            Layout::from_size_align_unchecked(size, Self::T_ALIGN)
        }
    }

    /// Constructs the memory layout for the specified `count` of type `T` in checked-mode.
    ///
    /// This method checks for **overflow** and valid layout **size** in release-mode, and for
    /// _non-zero_ size and valid alignment in debug-mode.
    ///
    /// The _resulted size_ must be greater than `0` for whatever reason, this implies that
    /// `T` can't be `ZST`, and the alignment must be power of 2, which implies it can't be zero
    /// also.
    #[inline(always)]
    pub const unsafe fn layout_of(
        &self,
        count: usize,
        on_err: OnError,
    ) -> Result<Layout, MemoryError> {
        #[cfg(debug_assertions)]
        debug_assert_valid_alignment(Self::T_ALIGN);

        if let Some(size) = count.checked_mul(Self::T_SIZE) {
            #[cfg(debug_assertions)]
            debug_assert_non_zero_size(size);

            if size > Self::MAX_LAYOUT_SIZE {
                return Err(on_err.layout_err());
            };

            let layout = unsafe { Layout::from_size_align_unchecked(size, Self::T_ALIGN) };
            return Ok(layout);
        }

        Err(on_err.layout_err())
    }

    /// Tries to allocate memory space according to the provided `layout`.
    ///
    /// This method handles allocation error according to the error handling context `on_err`.
    ///
    /// Note that the process may be terminated even if the allocation was successful, because
    /// detecting memory allocation failures at the process-level is platform-specific.
    ///
    /// For instance, on some systems overcommit is allowed by default, which means
    /// that the kernel will map virtual memory to the process regardless of the available memory.
    ///
    /// On such systems, allocation is always reported to be successful, but the process may become a target
    /// for termination later.
    ///
    /// For better safety, consult the platform-specific documentation regarding out-of-memory
    /// (OOM) behavior.
    ///
    /// # Safety
    ///
    /// - Pointer must be `null` before calling this method.
    ///   This method doesn't deallocate the allocated memory space pointed to by this pointer.
    ///   Calling this method with a non-null pointer causes memory leaks, as access to the
    ///   allocated memory space will be lost without freeing it.
    ///
    /// - `align` must be a power of 2.
    ///
    /// - `size` must be greater than `0`, this implies that `T` can't be `ZST`.
    ///
    /// - `size` in bytes, when rounded up to the nearest multiple of `align`, must be less than
    ///   or equal to `isize::MAX` bytes.
    pub unsafe fn acquire(&mut self, layout: Layout, on_err: OnError) -> Result<(), MemoryError> {
        #[cfg(debug_assertions)]
        debug_assert_is_null(self);

        #[cfg(debug_assertions)]
        debug_assert_valid_layout(layout.size(), layout.align());

        let ptr = unsafe { alloc(layout) as *mut T };

        if likely(!ptr.is_null()) {
            self.ptr = ptr;
            return Ok(());
        }

        Err(on_err.allocator_err(layout))
    }

    /// Releases the memory space pointed to by the pointer according to the provided `layout`.
    ///
    /// This method doesn't call `drop` on the initialized elements.
    ///
    /// The pointer is set to `null` after deallocation.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - Initialized elements will not be dropped before deallocating memory.
    ///   This might cause memory leaks if `T` is not of trivial type, or if the elements are not
    ///   dropped properly before calling this method.
    ///
    /// - `layout` must be the same layout used to allocate the memory space.
    pub unsafe fn release(&mut self, layout: Layout) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        #[cfg(debug_assertions)]
        debug_assert_valid_layout(layout.size(), layout.align());

        unsafe { alloc::dealloc(self.ptr as *mut u8, layout) };

        self.ptr = ptr::null_mut();
    }

    /// Returns the base pointer.
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        self.ptr
    }

    /// Returns the base pointer as mutable pointer.
    #[must_use]
    #[inline(always)]
    pub const fn as_ptr_mut(&self) -> *mut T {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        self.ptr
    }

    /// Returns the base pointer as a pointer of type `C`.
    #[must_use]
    #[inline(always)]
    pub const fn cast<C>(&self) -> *const C {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        self.ptr.cast::<C>()
    }

    /// Returns the base pointer as a mutable pointer of type `C`.
    #[must_use]
    #[inline(always)]
    pub const fn cast_mut<C>(&mut self) -> *mut C {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        self.ptr.cast::<C>()
    }

    /// Sets the base pointer at current offset plus `t_offset` of the strides of `T`.
    #[inline(always)]
    pub const unsafe fn set_plus(&mut self, t_offset: usize) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { self.ptr = self.ptr.add(t_offset) };
    }

    /// Sets the base pointer at current offset minus `t_offset` of the strides of `T`.
    #[inline(always)]
    pub const unsafe fn set_minus(&mut self, t_offset: usize) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { self.ptr = self.ptr.sub(t_offset) };
    }

    /// Sets all elements in the allocated memory space to the specified value of type `T`.
    ///
    /// Offset is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the writing range `[0, count - 1]`.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `count` must be within the bounds of the allocated memory space.
    ///
    /// - Initialized values will be overwritten **without** calling `drop`.
    ///   This might cause memory leaks if `T` is not of trivial type, or if the values are not
    ///   dropped properly before calling this method.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the `count` of values of type `T` to be written.
    ///
    #[inline(always)]
    pub unsafe fn memset(&mut self, count: usize, value: T)
    where
        T: Copy,
    {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe {
            let mut i = 0;

            while i < count {
                let value_ptr = self.ptr.add(i);
                value_ptr.write(value.clone());
                i += 1;
            }
        }
    }

    /// Sets all elements in the allocated memory space to the default value of `T`.
    ///
    /// Offset is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the writing range `[0, count - 1]`.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `count` must be within the bounds of the allocated memory space.
    ///
    /// - Initialized values will be overwritten **without** calling `drop`.
    ///   This might cause memory leaks if `T` is not of trivial type, or if the values are not
    ///   dropped properly before calling this method.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the `count` of values of type `T` to be written.
    ///
    #[inline(always)]
    pub unsafe fn memset_default(&mut self, count: usize)
    where
        T: Copy,
        T: Default,
    {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe {
            let mut i = 0;

            while i < count {
                let value_ptr = self.ptr.add(i);
                value_ptr.write(T::default());
                i += 1;
            }
        }
    }

    /// Writes `0` bytes to `count` values with the size of `T` in the allocated memory space
    /// starting from the offset `0`.
    ///
    /// Indexing is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the writing range `[0, count - 1]`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `count` must be within the bounds of the allocated memory space.
    ///
    /// - Initialized elements will be overwritten **without** calling `drop`.
    ///   This might cause memory leaks if `T` is not of trivial type, or if the elements are not
    ///   dropped properly before calling this method.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the `count` of type `T`.
    #[inline(always)]
    pub const unsafe fn memset_zero(&mut self, count: usize) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { self.ptr.write_bytes(0, count) };
    }

    /// Stores a value at the specified offset `at`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `offset` must be within the bounds of the allocated memory space.
    ///
    /// - If the offset has already been initialized, the value will be overwritten **without**
    ///   calling `drop`. This might cause memory leaks if the element is not of trivial type,
    ///   or not dropped properly before overwriting.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[inline(always)]
    pub const unsafe fn store(&mut self, offset: usize, value: T) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { self.ptr.add(offset).write(value) };
    }

    /// Returns a reference to an element at the specified `offset`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - The value at the current address must be an initialized value of type T.
    ///   Accessing an uninitialized element as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    #[must_use]
    #[inline(always)]
    pub const unsafe fn reference(&self, offset: usize) -> &T {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { &*self.ptr.add(offset) }
    }

    /// Returns a mutable reference to an element at the specified `offset`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - The value at the current address must be an initialized value of type T.
    ///   Accessing an uninitialized element as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[must_use]
    #[inline(always)]
    pub const unsafe fn reference_mut(&mut self, offset: usize) -> &mut T {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { &mut *(self.ptr).add(offset) }
    }

    /// Returns a reference to the element where the current pointer is.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`..
    ///
    /// - The value at the current address must be an initialized value of type T.
    ///   Accessing an uninitialized element as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[must_use]
    #[inline(always)]
    pub const unsafe fn reference_first(&self) -> &T {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { &*self.ptr }
    }

    /// Returns an immutable slice of the initialized elements starting from the offset `0`.
    ///
    /// Indexing is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the slice range `[0, count - 1]`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `count` must be within the bounds of the initialized elements.
    ///   Loading an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[inline(always)]
    pub const unsafe fn as_slice(&self, count: usize) -> &[T] {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { &*ptr::slice_from_raw_parts(self.ptr, count) }
    }

    /// Returns a mutable slice over `count` initialized elements starting from the offset `0`.
    ///
    /// Indexing is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the slice range `[0, count - 1]`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `count` must be within the bounds of the initialized elements.
    ///   Accessing an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[inline(always)]
    pub const unsafe fn as_slice_mut(&mut self, count: usize) -> &mut [T] {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { &mut *ptr::slice_from_raw_parts_mut(self.ptr, count) }
    }

    /// Reads and returns the value at the specified `offset`.
    ///
    /// This method creates a bitwise copy of `T` with `move` semantics.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `offset` must be within the bounds of the initialized elements.
    ///   Loading an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// - If `T` is not a trivial type, the value at this offset can be in an invalid state after
    ///   calling this method, because it might have been dropped by the caller.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    #[inline(always)]
    pub const unsafe fn read_for_ownership(&mut self, offset: usize) -> T {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { self.ptr.add(offset).read() }
    }

    /// Shifts `count` number of values after the provided `offset` to the left,
    /// overwriting the value at that `offset`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment.
    ///
    /// - `offset + count` must be within the bounds of the allocated memory.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the number (`count`) of the elements to be shifted.
    #[inline(always)]
    pub const unsafe fn shift_left(&mut self, offset: usize, count: usize) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe {
            let dst = self.ptr.add(offset);
            let src = dst.add(1);
            dst.copy_from(src, count);
        }
    }

    /// Copies the value at the offset `src` to the offset `dst`, overwriting the value at `dst`
    /// and leaving the value at `src` unaffected.
    ///
    /// This operation is internally untyped, the initialization state is operationally irrelevant.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment.
    ///
    /// - `src` and `dst` must be within the bounds of the allocated memory-segment.
    ///
    /// - If the value at offset `dst` has been initialized already, the value will be overwritten **without**
    ///   calling `drop`. This might cause memory leaks if the element is not of trivial type,
    ///   or not dropped properly before overwriting.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    #[inline(always)]
    pub const unsafe fn memmove_one(&mut self, src_offset: usize, dst_offset: usize) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe {
            let src = self.ptr.add(src_offset);
            let dst = self.ptr.add(dst_offset);
            dst.copy_from(src, 1);
        }
    }

    /// Copies values of type `T` from the memory space pointed to by the source pointer `source`.
    ///
    /// Indexing is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the copy range `[0, count - 1]`.
    ///
    /// This method is no-op if `count` is `0`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `source` pointer must be allocated.
    #[inline(always)]
    pub unsafe fn copy_disjoint_from(&mut self, source: *const T, count: usize)
    where
        T: Copy,
    {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);
        debug_assert!(!source.is_null());

        unsafe { self.ptr.copy_from_nonoverlapping(source, count) };
    }

    /// Clones values of type `T` from the memory space pointed to by the source pointer `source`.
    ///
    /// Indexing is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the copy range `[0, count - 1]`.
    ///
    /// This method is unwind-safe. It will call drop on the cloned elements when unwinding
    /// starts.
    ///
    /// This method is no-op if `count` is `0`.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `clone_count` must be within the bounds of the initialized elements.
    ///   Cloning an uninitialized elements as `T` is `undefined behavior`.
    #[inline(always)]
    pub unsafe fn clone_from(&mut self, source: *const T, clone_count: usize)
    where
        T: Clone,
    {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);
        debug_assert!(!source.is_null());

        let self_ptr = self.ptr;

        let cloned = 0;

        let mut drop_guard =
            OnDrop::set_on(cloned, |cloned| unsafe { self.drop_initialized(*cloned) });

        unsafe {
            for i in 0..clone_count {
                let src_ptr = source.add(i);
                let dst_ptr = self_ptr.add(i);
                dst_ptr.write((*src_ptr).clone());
                drop_guard.arg += 1;
            }
        }

        // Cloned successfully (If any).
        drop_guard.set_off();
    }

    /// Creates new instance and copies values from the current memory space to the new memory space.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `count` must be within the bounds of the allocated memory space.
    ///   Copying more elements than the allocated count will cause termination with `SIGSEGV`.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the number (`count`) of values to be copied.
    #[must_use]
    #[inline]
    pub unsafe fn make_copy(&self, count: usize, on_err: OnError) -> Result<Self, MemoryError>
    where
        T: Copy,
    {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe {
            let layout = self.layout_unchecked_of(count);

            let mut instance = Self::new();

            instance.acquire(layout, on_err)?;

            instance.copy_disjoint_from(self.ptr, count);

            Ok(instance)
        }
    }

    /// Creates new instance and clones values from the current memory space
    /// to the new memory space.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `count` must be within the bounds of the initialized elements.
    ///   Cloning an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the number (`count`) of values to be cloned.
    #[must_use]
    pub unsafe fn make_clone(&self, count: usize, on_err: OnError) -> Result<Self, MemoryError>
    where
        T: Clone,
    {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe {
            let layout = self.layout_unchecked_of(count);

            let mut instance = Self::new();

            instance.acquire(layout, on_err)?;

            // Unwind-safe.
            instance.clone_from(self.ptr, count);

            Ok(instance)
        }
    }

    /// Compares the values in the memory space pointed to by this pointer with the values in a
    /// memory space pointed to by other pointer.
    ///
    /// Offset is zero-based, i.e., the last value is at offset `count - 1`, this will make
    /// the comparing range `[0, count - 1]`.
    ///
    /// Returns `true` if `count` is `0`.
    ///
    /// # Safety
    ///
    /// - Both pointers must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `count` must be within the bounds of the initialized elements.
    ///   Accessing an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the number (`count`) of values to be compared.
    #[must_use]
    #[inline(always)]
    pub unsafe fn compare_partial_eq(&self, other: &Self, count: usize) -> bool
    where
        T: PartialEq,
    {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        #[cfg(debug_assertions)]
        debug_assert_not_null(other);

        unsafe {
            let mut i = 0;

            while i < count {
                if *self.ptr.add(i) != *other.ptr.add(i) {
                    return false;
                }
                i += 1;
            }

            true
        }
    }

    /// Calls `drop` on the initialized elements with the specified `count` starting from the
    /// offset `0`.
    ///
    /// Indexing is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the drop range `[0, count - 1]`.
    ///
    /// This method is no-op when `count` is `0` or when `T` is of trivial type.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `count` must be within the bounds of the **initialized** elements.
    ///   Calling `drop` on uninitialized elements is `undefined behavior`.
    ///
    /// - If `T` is not of trivial type, using dropped values after calling this method is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the number (`count`) of the elements to be dropped.
    ///
    #[inline(always)]
    pub unsafe fn drop_initialized(&mut self, count: usize) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);

        unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr, count)) };
    }

    /// Calls `drop` on the initialized elements in the specified range.
    ///
    /// The total drop `count` equals `end - start - 1`.
    ///
    /// This method is no-op if `T` is of a trivial type.
    ///
    /// # Safety
    ///
    /// - Pointer must point to an already allocated memory-segment aligned to the alignment of `T`.
    ///
    /// - `range` must not be empty.
    ///
    /// - `range` must be within the bounds of the **initialized** elements.
    ///   Calling `drop` on uninitialized elements is `undefined behavior`.
    ///
    /// - If `T` is not of trivial type, using dropped values after calling this method is `undefined behavior`.
    ///
    /// These invariants are checked in debug mode only.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the number (`count`) of the elements to be dropped.
    ///
    #[inline(always)]
    pub unsafe fn drop_range(&mut self, range: Range<usize>) {
        #[cfg(debug_assertions)]
        debug_assert_not_null(self);
        debug_assert!(!range.is_empty(), "Drop range must not be empty");

        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.ptr.add(range.start),
                range.end - range.start,
            ))
        };
    }
}

#[cfg(test)]
mod tests_unmanaged_ptr {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_unmanaged_ptr_new() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        assert!(unmanaged_ptr.is_null());
    }

    #[test]
    fn test_unmanaged_ptr_make_layout_unchecked_ok() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        unsafe {
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            assert_eq!(layout.size(), 3);
            assert_eq!(layout.align(), UnmanagedPointer::<u8>::T_ALIGN);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size must be greater than 0")]
    fn test_unmanaged_ptr_make_layout_unchecked_zero_size() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        unsafe {
            let _ = unmanaged_ptr.layout_unchecked_of(0);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size exceeds maximum limit on this platform")]
    fn test_unmanaged_ptr_make_layout_unchecked_invalid_size() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        unsafe {
            let _ = unmanaged_ptr.layout_unchecked_of(isize::MAX as usize + 1);
        }
    }

    #[test]
    fn test_unmanaged_ptr_layout_ok() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        unsafe {
            let layout = unmanaged_ptr.layout_of(3, OnError::Panic).unwrap();
            assert_eq!(layout.size(), 3);
            assert_eq!(layout.align(), UnmanagedPointer::<u8>::T_ALIGN);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size must be greater than 0")]
    fn test_unmanaged_ptr_layout_zero_size() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        unsafe {
            let _ = unmanaged_ptr.layout_of(0, OnError::Panic);
        }
    }

    #[test]
    #[should_panic(expected = "layout error")]
    fn test_unmanaged_ptr_layout_panic() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        unsafe {
            let _ = unmanaged_ptr.layout_of(usize::MAX, OnError::Panic);
        }
    }

    #[test]
    fn test_unmanaged_ptr_layout_with_err() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        unsafe {
            let result = unmanaged_ptr.layout_of(usize::MAX, OnError::ReturnErr);
            assert!(result.is_err());
            assert!(matches!(result, Err(MemoryError::LayoutErr)));
        }
    }

    #[test]
    fn test_unmanaged_ptr_acquire() {
        let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();

        unsafe {
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let result = unmanaged_ptr.acquire(layout, OnError::Panic);

            assert!(result.is_ok());
            assert!(!unmanaged_ptr.is_null());

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pointer must be null")]
    #[cfg_attr(miri, ignore)]
    fn test_unmanaged_ptr_acquire_non_null() {
        let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        unsafe {
            let layout = unmanaged_ptr.layout_unchecked_of(1);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            assert!(!unmanaged_ptr.is_null());

            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);
        }
    }

    #[test]
    fn test_unmanaged_ptr_acquire_release() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();

            let layout = unmanaged_ptr.layout_unchecked_of(3);

            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            assert!(!unmanaged_ptr.is_null());

            unmanaged_ptr.release(layout);

            assert!(unmanaged_ptr.is_null());
        }
    }

    #[test]
    fn test_unmanaged_ptr_memset_zero() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            for i in 0..3 {
                unmanaged_ptr.store(i, i as u8 + 1);
            }

            for i in 0..3 {
                assert_ne!(*unmanaged_ptr.reference(i), 0);
            }

            unmanaged_ptr.memset_zero(3);

            for i in 0..3 {
                assert_eq!(*unmanaged_ptr.reference(i), 0);
            }

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_store_access() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            for i in 0..3 {
                unmanaged_ptr.store(i, i as u8 + 1);
            }

            assert_eq!(*unmanaged_ptr.reference(0), 1);
            assert_eq!(*unmanaged_ptr.reference(1), 2);
            assert_eq!(*unmanaged_ptr.reference(2), 3);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_access_mut() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            unmanaged_ptr.store(0, 1);
            unmanaged_ptr.store(1, 2);

            *unmanaged_ptr.reference_mut(0) = 10;

            assert_eq!(*unmanaged_ptr.reference(0), 10);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_access_first() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            unmanaged_ptr.store(0, 1);
            unmanaged_ptr.store(1, 2);

            assert_eq!(unmanaged_ptr.reference_first(), &1);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_rfo() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            unmanaged_ptr.store(0, 1);
            unmanaged_ptr.store(1, 2);

            assert_eq!(unmanaged_ptr.read_for_ownership(0), 1);

            assert_eq!(*unmanaged_ptr.reference(1), 2);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_shift_left() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(5);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            for i in 0..5 {
                unmanaged_ptr.store(i, i as u8 + 1);
            }

            unmanaged_ptr.shift_left(2, 2);

            assert_eq!(*unmanaged_ptr.reference(0), 1);
            assert_eq!(*unmanaged_ptr.reference(1), 2);
            assert_eq!(*unmanaged_ptr.reference(2), 4);
            assert_eq!(*unmanaged_ptr.reference(3), 5);
            assert_eq!(*unmanaged_ptr.reference(4), 5);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_move_one() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            unmanaged_ptr.store(0, 10);
            unmanaged_ptr.store(1, 20);
            unmanaged_ptr.store(2, 30);

            unmanaged_ptr.memmove_one(0, 2);

            assert_eq!(*unmanaged_ptr.reference(0), 10);
            assert_eq!(*unmanaged_ptr.reference(1), 20);
            assert_eq!(*unmanaged_ptr.reference(2), 10);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pointer must not be null")]
    fn test_unmanaged_ptr_as_slice_null() {
        let unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        let _ = unsafe { unmanaged_ptr.as_slice(0) };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pointer must not be null")]
    fn test_unmanaged_ptr_as_slice_mut_null() {
        let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
        let _ = unsafe { unmanaged_ptr.as_slice_mut(0) };
    }

    #[test]
    fn test_unmanaged_ptr_as_slice_empty() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            let slice = unmanaged_ptr.as_slice(0);
            assert_eq!(slice, &[]);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_as_slice() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            for i in 0..3 {
                unmanaged_ptr.store(i, i as u8 + 1);
            }

            let slice = unmanaged_ptr.as_slice(3);
            assert_eq!(slice, &[1, 2, 3]);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_as_slice_mut_empty() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            let slice = unmanaged_ptr.as_slice_mut(0);
            assert_eq!(slice, &[]);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_as_slice_mut() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            for i in 0..3 {
                unmanaged_ptr.store(i, i as u8 + 1);
            }

            let slice = unmanaged_ptr.as_slice_mut(3);
            assert_eq!(slice, &mut [1, 2, 3]);

            unmanaged_ptr.release(layout);
        }
    }

    #[derive(Debug, Clone)]
    struct DropCounter {
        count: Rc<RefCell<usize>>,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            *self.count.borrow_mut() += 1;
        }
    }

    #[test]
    fn test_unmanaged_ptr_drop_init() {
        let drop_count = Rc::new(RefCell::new(0));

        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<DropCounter> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(3);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            for i in 0..3 {
                unmanaged_ptr.store(
                    i,
                    DropCounter {
                        count: Rc::clone(&drop_count),
                    },
                );
            }

            unmanaged_ptr.drop_initialized(0);
            assert_eq!(*drop_count.borrow(), 0);

            unmanaged_ptr.drop_initialized(3);

            assert_eq!(*drop_count.borrow(), 3);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Drop range must not be empty")]
    #[cfg_attr(miri, ignore)]
    fn test_unmanaged_ptr_drop_range_invalid() {
        unsafe {
            let mut unmanaged_ptr: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = unmanaged_ptr.layout_unchecked_of(5);
            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);
            unmanaged_ptr.drop_range(0..0);
        }
    }

    #[test]
    fn test_unmanaged_ptr_drop_range() {
        let drop_count = DropCounter {
            count: Rc::new(RefCell::new(0)),
        };

        let mut unmanaged_ptr = UnmanagedPointer::new();

        unsafe {
            let layout = unmanaged_ptr.layout_unchecked_of(5);

            let _ = unmanaged_ptr.acquire(layout, OnError::Panic);

            for i in 0..3 {
                unmanaged_ptr.store(i, drop_count.clone())
            }

            unmanaged_ptr.drop_range(0..3);

            assert_eq!(*drop_count.count.borrow(), 3);

            unmanaged_ptr.release(layout);
        }
    }

    #[test]
    fn test_unmanaged_ptr_clone_from() {
        unsafe {
            let mut source: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let layout = source.layout_unchecked_of(3);
            let _ = source.acquire(layout, OnError::Panic);

            for i in 0..3 {
                source.store(i, i as u8 + 1);
            }

            let mut target: UnmanagedPointer<u8> = UnmanagedPointer::new();
            let _ = target.acquire(layout, OnError::Panic);

            target.clone_from(source.ptr, 3);

            for i in 0..3 {
                assert_eq!(*source.reference(i), *target.reference(i));
            }

            source.release(layout);
            target.release(layout);
        }
    }

    struct PanicOnClone {
        id: usize,
        panic_on: usize,
        dropped: Rc<RefCell<usize>>,
    }

    impl Clone for PanicOnClone {
        fn clone(&self) -> Self {
            if self.id == self.panic_on {
                panic!("A clone with id {} panicked", self.id);
            }
            Self {
                id: self.id,
                panic_on: self.panic_on,
                dropped: Rc::clone(&self.dropped),
            }
        }
    }

    impl Drop for PanicOnClone {
        fn drop(&mut self) {
            *self.dropped.borrow_mut() += 1;
        }
    }

    #[test]
    fn test_unmanaged_ptr_clone_from_unwind() {
        let mut source: UnmanagedPointer<PanicOnClone> = UnmanagedPointer::new();
        let mut target: UnmanagedPointer<PanicOnClone> = UnmanagedPointer::new();
        unsafe {
            let layout = source.layout_unchecked_of(10);
            let _ = source.acquire(layout, OnError::Panic);
            let _ = target.acquire(layout, OnError::Panic);

            let drop_counter = Rc::new(RefCell::new(0));
            for i in 0..10 {
                let value = PanicOnClone {
                    id: i,
                    panic_on: 5,
                    dropped: Rc::clone(&drop_counter),
                };
                source.store(i, value);
            }

            let source_ptr = source.ptr as *const ();
            let target_ptr = &mut target as *mut UnmanagedPointer<PanicOnClone> as *mut ();

            let result = std::panic::catch_unwind(move || {
                let target = &mut *(target_ptr as *mut UnmanagedPointer<PanicOnClone>);
                let source = source_ptr as *const PanicOnClone;
                target.clone_from(source, 10);
            });

            assert!(result.is_err());
            assert_eq!(*drop_counter.borrow(), 5);

            source.drop_initialized(10);
            source.release(layout);
            target.release(layout);
        }
    }
}
