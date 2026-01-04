use core::alloc::Layout;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::Range;
use core::ptr;

use crate::core::opt::branch_prediction;
use std::alloc::{self, alloc};

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
const fn debug_layout_size_align(size: usize, align: usize) {
    assert!(align.is_power_of_two(), "Alignment must be a power of two");

    assert!(size != 0, "Allocation size must be greater than 0");

    let max_size = (isize::MAX as usize + 1) - align;
    assert!(
        max_size > size,
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
const fn debug_assert_allocated<T>(instance: &AllocationPointer<T>) {
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
const fn debug_assert_not_allocated<T>(instance: &AllocationPointer<T>) {
    assert!(instance.ptr.is_null(), "Pointer must be null");
}

/// An indirect reference to _one or more_ values of type `T` consecutively in memory,
/// with methods for managing the underlying memory directly.
///
/// It guarantees proper `size` and `alignment` of `T`, when storing or accessing
/// values, but it doesn't guarantee safe operations with measures such as null pointer checks or
/// bounds checking.
///
/// Moreover, it doesn't store any metadata about its allocated memory, such as the size of
/// the allocated memory and the number of initialized elements, therefore it doesn't offer
/// automatic memory management.
///
/// If `T` is not of trivial type, `drop` must be called on the elements to release resources
/// before deallocating the allocated memory.
///
/// Limited checks for invariants are done in debug mode only.
///
/// It uses the registered `#[global_allocator]` to allocate memory.
///
/// Using custom allocators will be supported in the future.
pub(crate) struct AllocationPointer<T> {
    ptr: *mut T,
    _marker: PhantomData<T>,
}

impl<T> AllocationPointer<T> {
    pub(crate) const T_SIZE: usize = size_of::<T>();
    pub(crate) const T_ALIGN: usize = align_of::<T>();

    /// Creates a new `MemorySpace` without allocating memory.
    ///
    /// The pointer is set to `null`.
    ///
    #[must_use]
    #[inline]
    pub(crate) const fn new() -> Self {
        AllocationPointer {
            ptr: ptr::null_mut(),
            _marker: PhantomData,
        }
    }

    /// Creates a new `MemorySpace` with the specified `count`.
    ///
    /// Memory is allocated for the specified `count` of type `T`.
    ///
    /// # Safety
    ///
    /// - `count` must be greater than `0`.
    ///
    /// - The total size of the allocated memory when rounded up to the nearest multiple of `align`,
    ///   must be less than or equal to `isize::MAX`.
    ///
    ///   If the total size exceeds `isize::MAX` bytes, the memory allocation will fail.
    ///
    #[must_use]
    #[inline]
    pub(crate) unsafe fn new_allocate(count: usize) -> Self {
        let mut instance = Self::new();
        instance.allocate(count);
        instance
    }

    /// Creates a new `MemorySpace` with the specified `count` of type `T` and populates
    /// it with the default value of `T`.
    ///
    /// # Safety
    ///
    /// - `count` must be greater than `0`.
    ///
    /// - The total size of the allocated memory when rounded up to the nearest multiple of `align`,
    ///   must be less than or equal to `isize::MAX`.
    ///
    ///   If the total size exceeds `isize::MAX` bytes, the memory allocation will fail.
    ///
    #[must_use]
    #[inline]
    pub(crate) unsafe fn new_allocate_default(count: usize) -> Self
    where
        T: Default,
    {
        let mut instance = Self::new();
        instance.allocate(count);
        instance.memset_default(count);
        instance
    }

    /// Creates a new `MemorySpace` with the specified `count` of type `T` and populates
    /// it with the provided value.
    ///
    /// # Safety
    ///
    /// - `count` must be greater than `0`.
    ///
    /// - The total size of the allocated memory when rounded up to the nearest multiple of `align`,
    ///   must be less than or equal to `isize::MAX`.
    ///
    ///   If the total size exceeds `isize::MAX` bytes, the memory allocation will fail.
    ///
    #[must_use]
    #[inline]
    pub(crate) unsafe fn new_allocate_memset(count: usize, value: T) -> Self
    where
        T: Clone,
    {
        let mut instance = Self::new();
        instance.allocate(count);
        instance.memset(count, value);
        instance
    }

    /// Creates a new `MemorySpace` from slice.
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
    pub(crate) unsafe fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        let len = slice.len();
        unsafe {
            let instance = AllocationPointer::new_allocate(len);
            ptr::copy_nonoverlapping(slice.as_ptr(), instance.ptr, len);
            instance
        }
    }

    /// Creates a new `MemorySpace` from vector.
    ///
    /// Allocator must be `Global`. Custom allocators are not supported.
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
    pub(crate) unsafe fn from_vec(vec: Vec<T>) -> Self {
        #[cfg(debug_assertions)]
        debug_layout_size_align(vec.capacity() * Self::T_SIZE, Self::T_ALIGN);

        AllocationPointer {
            ptr: ManuallyDrop::new(vec).as_mut_ptr(),
            _marker: PhantomData,
        }
    }

    /// Returns the base pointer of the buffer as raw pointer.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn base(&self) -> *const T {
        self.ptr
    }

    /// Returns the base pointer of the buffer as mutable raw pointer.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn base_mut(&self) -> *mut T {
        self.ptr
    }

    /// Returns an instance with copy of the base pointer.
    ///
    /// # Safety
    ///
    /// This method doesn't provide any guarantees about the state of the returned pointer and its
    /// allocated memory space.
    ///
    #[must_use]
    #[inline(always)]
    pub(crate) const unsafe fn duplicate(&mut self) -> AllocationPointer<T> {
        AllocationPointer {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    /// Creates a new layout for the specified `count` of type `T`.
    ///
    /// This method checks for valid size and alignment in debug mode only.
    ///
    #[must_use]
    #[inline(always)]
    const unsafe fn make_layout(&self, count: usize) -> Layout {
        let size = count.unchecked_mul(Self::T_SIZE);

        #[cfg(debug_assertions)]
        debug_layout_size_align(size, Self::T_ALIGN);

        Layout::from_size_align_unchecked(size, Self::T_ALIGN)
    }

    /// Allocates memory space for the specified `count` of type `T`.
    ///
    /// # Safety
    ///
    /// - Pointer must be `null` before calling this method.
    ///   This method doesn't deallocate the allocated memory space pointed by the pointer.
    ///   Calling this method with a non-null pointer will cause memory leaks, as access to the
    ///   allocated memory space will be lost.
    ///
    /// - `count` must be greater than `0`.
    ///
    /// - `count` in bytes, when rounded up to the nearest multiple of `align`, must be less than
    ///   or equal to `isize::MAX` bytes.
    ///
    pub(crate) unsafe fn allocate(&mut self, count: usize) {
        #[cfg(debug_assertions)]
        debug_assert_not_allocated(self);

        let new_layout = self.make_layout(count);

        let ptr = alloc(new_layout) as *mut T;

        // Success branch.
        if branch_prediction::likely(!ptr.is_null()) {
            self.ptr = ptr;
            return;
        }

        alloc::handle_alloc_error(new_layout);
    }

    /// Deallocates the memory space pointed by the pointer.
    ///
    /// This method doesn't call `drop` on the initialized elements.
    ///
    /// The pointer is set to `null` after deallocation.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - Initialized values will not be dropped before deallocating memory.
    ///   This might cause memory leaks if `T` is not of trivial type, or if the values are not
    ///   dropped properly before calling this method.
    ///
    /// - `allocated_count` must be the same as the actual allocated count of type `T`, which
    ///   implies it can't be `0` also.
    ///   If the count is not the same, the result is `undefined behavior`.
    ///
    pub(crate) unsafe fn deallocate(&mut self, allocated_count: usize) {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        let current_layout = self.make_layout(allocated_count);

        alloc::dealloc(self.ptr as *mut u8, current_layout);

        self.ptr = ptr::null_mut();
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
    pub(crate) unsafe fn memset_default(&mut self, count: usize)
    where
        T: Default,
    {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        let mut i = 0;
        while i < count {
            ptr::write((self.ptr).add(i), T::default());
            i += 1;
        }
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
    pub(crate) unsafe fn memset(&mut self, count: usize, value: T)
    where
        T: Clone,
    {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        let mut i = 0;
        while i < count {
            ptr::write((self.ptr).add(i), value.clone());
            i += 1;
        }
    }

    /// Stores a value at the specified offset `at`.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `at` must be within the bounds of the allocated memory space.
    ///
    /// - If the offset has already been initialized, the value will be overwritten **without**
    ///   calling `drop`. This might cause memory leaks if `T` is not of trivial type, or the value
    ///   is not dropped properly before overwriting.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[inline(always)]
    pub(crate) const unsafe fn store(&mut self, at: usize, value: T) {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        ptr::write((self.ptr).add(at), value);
    }

    /// Returns a reference to an initialized element at the specified offset `at`.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGSEGV`.
    ///
    /// - `at` must be within the bounds of the initialized elements.
    ///   Accessing an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[must_use]
    #[inline(always)]
    pub(crate) const unsafe fn reference(&self, at: usize) -> &T {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        &*self.ptr.add(at)
    }

    /// Returns a mutable reference to an initialized element at the specified offset `at`.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGSEGV`.
    ///
    /// - `at` must be within the bounds of the initialized elements.
    ///   Accessing an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[must_use]
    #[inline(always)]
    pub(crate) const unsafe fn reference_mut(&mut self, at: usize) -> &mut T {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        &mut *(self.ptr).add(at)
    }

    /// Reads and returns the value at the specified offset `at`.
    ///
    /// This method creates a bitwise copy of `T` with `move` semantics.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null ptr will cause termination with `SIGABRT`.
    ///
    /// - `at` must be within the bounds of the initialized elements.
    ///   Loading an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// - If `T` is not a trivial type, the value at this offset can be in an invalid state after
    ///   calling this method, because it might have been dropped by the caller.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) const unsafe fn read_for_ownership(&mut self, at: usize) -> T {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        ptr::read((self.ptr).add(at))
    }

    /// Calls `drop` on the initialized elements with the specified `count` starting from the
    /// offset `0`.
    ///
    /// Offset is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the drop range `[0, count - 1]`.
    ///
    /// This method is no-op when `count` is `0` or when `T` is of trivial type.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `count` must be within the bounds of the **initialized** elements.
    ///   Calling `drop` on uninitialized elements is `undefined behavior`.
    ///
    /// - If `T` is not of trivial type, using dropped values after calling this method can cause
    ///   `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the number (`count`) of the elements to be dropped.
    ///
    #[inline(always)]
    pub(crate) unsafe fn drop_initialized(&mut self, count: usize) {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr, count));
    }

    /// Calls `drop` on the initialized elements in the specified range.
    ///
    /// The total drop `count` equals `end - start - 1`.
    ///
    /// This method is no-op when `T` is of trivial type.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `range` must not be empty.
    ///
    /// - `range` must be within the bounds of the **initialized** elements.
    ///   Calling `drop` on uninitialized elements is `undefined behavior`.
    ///
    /// - If `T` is not of trivial type, using dropped values after calling this method is
    ///   `undefined behavior`.
    ///
    /// These invariants are checked in debug mode only.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the number (`count`) of the elements to be dropped.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) unsafe fn drop_range(&mut self, range: Range<usize>) {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        debug_assert!(!range.is_empty(), "Drop range must not be empty");

        ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
            self.ptr.add(range.start),
            range.end - range.start,
        ));
    }

    /// Returns an immutable slice of the initialized elements starting from the offset `0`.
    ///
    /// Offset is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the slice range `[0, count - 1]`.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `count` must be within the bounds of the initialized elements.
    ///   Accessing an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[inline(always)]
    pub(crate) const unsafe fn as_slice(&self, count: usize) -> &[T] {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        &*ptr::slice_from_raw_parts(self.ptr, count)
    }

    /// Returns a mutable slice over `count` initialized elements starting from the offset `0`.
    ///
    /// Offset is zero-based, i.e., the last element is at offset `count - 1`, this will make
    /// the slice range `[0, count - 1]`.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - `count` must be within the bounds of the initialized elements.
    ///   Accessing an uninitialized elements as `T` is `undefined behavior`.
    ///
    /// # Time Complexity
    ///
    /// _O_(1).
    ///
    #[inline(always)]
    pub(crate) const unsafe fn as_slice_mut(&mut self, count: usize) -> &mut [T] {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        &mut *ptr::slice_from_raw_parts_mut(self.ptr, count)
    }

    /// Copies _bitwise_ values of type `T` from slice to the allocated memory.
    ///
    /// This method is no-op if `count` is `0`.
    ///
    /// # Safety
    ///
    /// - Pointer must be allocated before calling this method.
    ///   Calling this method with a null pointer will cause termination with `SIGABRT`.
    ///
    /// - Memory regions must not overlap.
    ///
    /// - Slice's length must be within the bounds of the allocated memory space.
    ///   Copying more than the allocated count will cause termination with `SIGSEGV`.
    ///
    /// # Time Complexity
    ///
    /// _O_(n) where `n` is the length of the slice.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) const unsafe fn copy_from_slice(&mut self, slice: &[T])
    where
        T: Copy,
    {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr, slice.len());
    }

    /// Creates new `MemorySpace` and clones values from the current memory space
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
    pub(crate) unsafe fn make_clone(&self, count: usize) -> Self
    where
        T: Clone,
    {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        let instance = Self::new_allocate(count);

        let mut i = 0;
        while i < count {
            let src = self.ptr.add(i);
            let dst = instance.ptr.add(i);
            ptr::write(dst, (*src).clone());
            i += 1;
        }

        instance
    }

    /// Creates new `MemorySpace` and copies _bitwise_ values from the current memory space
    /// to the new memory space.
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
    pub(crate) unsafe fn make_copy(&self, count: usize) -> Self
    where
        T: Copy,
    {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        let instance = AllocationPointer::new_allocate(count);

        ptr::copy_nonoverlapping(self.ptr, instance.ptr, count);

        instance
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
    pub(crate) unsafe fn cmp_eq(&self, other: &Self, count: usize) -> bool
    where
        T: PartialEq,
    {
        #[cfg(debug_assertions)]
        debug_assert_allocated(self);

        #[cfg(debug_assertions)]
        debug_assert_allocated(other);

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

#[cfg(test)]
mod alloc_ptr_tests {
    use super::*;

    #[test]
    fn test_alloc_ptr_new() {
        let alloc_ptr: AllocationPointer<u8> = AllocationPointer::new();
        assert!(alloc_ptr.ptr.is_null());
    }

    #[test]
    fn test_alloc_ptr_new_allocate() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(3);

            // Memory space should have been allocated.
            assert!(!alloc_ptr.ptr.is_null());

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size must be greater than 0")]
    fn test_alloc_ptr_new_allocate_zero_size() {
        // count is 0, should panic.
        let _: AllocationPointer<u8> = unsafe { AllocationPointer::new_allocate(0) };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size exceeds maximum limit on this platform")]
    fn test_alloc_ptr_new_allocate_overflow() {
        let _: AllocationPointer<u8> =
            unsafe { AllocationPointer::new_allocate(isize::MAX as usize + 1) };
    }

    #[test]
    fn test_alloc_ptr_allocate() {
        let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new();

        unsafe {
            alloc_ptr.allocate(3);

            assert!(!alloc_ptr.ptr.is_null());

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size must be greater than 0")]
    fn test_alloc_ptr_allocate_zero_size() {
        let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new();

        unsafe {
            alloc_ptr.allocate(0);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size exceeds maximum limit on this platform")]
    fn test_alloc_ptr_allocate_overflow() {
        let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new();

        unsafe {
            alloc_ptr.allocate(isize::MAX as usize + 1);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pointer must be null")]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_ptr_allocate_allocated() {
        let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new();
        unsafe {
            alloc_ptr.allocate(1);

            assert!(!alloc_ptr.ptr.is_null());

            alloc_ptr.allocate(2);
        }
    }

    #[derive(PartialEq, Debug)]
    struct TestDefault {
        data: String,
    }

    impl Default for TestDefault {
        fn default() -> Self {
            Self {
                data: String::from("default"),
            }
        }
    }

    #[test]
    fn test_alloc_ptr_memset_default() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<TestDefault> = AllocationPointer::new_allocate(3);

            let t_default = TestDefault::default();

            alloc_ptr.memset_default(3);

            for i in 0..3 {
                assert_eq!(*alloc_ptr.ptr.add(i), t_default)
            }

            alloc_ptr.drop_initialized(3);
            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    fn test_alloc_ptr_new_allocate_default() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<TestDefault> =
                AllocationPointer::new_allocate_default(3);

            let t_default = TestDefault::default();

            assert!(!alloc_ptr.ptr.is_null());

            for i in 0..3 {
                assert_eq!(*alloc_ptr.ptr.add(i), t_default)
            }

            alloc_ptr.drop_initialized(3);
            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size must be greater than 0")]
    fn test_alloc_ptr_new_allocate_default_zero_count() {
        let _: AllocationPointer<u8> = unsafe { AllocationPointer::new_allocate_default(0) };
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size exceeds maximum limit on this platform")]
    fn test_alloc_ptr_new_allocate_default_overflow() {
        let _: AllocationPointer<u8> =
            unsafe { AllocationPointer::new_allocate_default(isize::MAX as usize + 1) };
    }

    #[test]
    fn test_alloc_ptr_store_access() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(3);

            for i in 0..3 {
                alloc_ptr.store(i, i as u8 + 1);
            }

            assert_eq!(*alloc_ptr.reference(0), 1);
            assert_eq!(*alloc_ptr.reference(1), 2);
            assert_eq!(*alloc_ptr.reference(2), 3);

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    fn test_alloc_ptr_access_mut() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(3);

            alloc_ptr.store(0, 1);
            alloc_ptr.store(1, 2);

            *alloc_ptr.reference_mut(0) = 10;

            assert_eq!(*alloc_ptr.reference(0), 10);

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    fn test_alloc_ptr_take() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(3);

            alloc_ptr.store(0, 1);
            alloc_ptr.store(1, 2);

            assert_eq!(alloc_ptr.read_for_ownership(0), 1);

            assert_eq!(*alloc_ptr.reference(1), 2);

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pointer must not be null")]
    fn test_alloc_ptr_as_slice_null_ptr() {
        let alloc_ptr: AllocationPointer<u8> = AllocationPointer::new();
        let slice = unsafe { alloc_ptr.as_slice(0) };
        assert_eq!(slice, &[]);
    }

    #[test]
    fn test_alloc_ptr_as_slice_empty() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(3);
            let slice = alloc_ptr.as_slice(0);
            assert_eq!(slice, &[]);

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    fn test_alloc_ptr_as_slice() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(3);

            for i in 0..3 {
                alloc_ptr.store(i, i as u8 + 1);
            }

            let slice = alloc_ptr.as_slice(3);
            assert_eq!(slice, &[1, 2, 3]);

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pointer must not be null")]
    fn test_alloc_ptr_as_slice_mut_null_ptr() {
        let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new();
        let slice = unsafe { alloc_ptr.as_slice_mut(0) };
        assert_eq!(slice, &mut []);
    }

    #[test]
    fn test_alloc_ptr_as_slice_mut_empty() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(3);
            let slice = alloc_ptr.as_slice_mut(0);
            assert_eq!(slice, &[]);

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    fn test_alloc_ptr_as_slice_mut() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(3);

            for i in 0..3 {
                alloc_ptr.store(i, i as u8 + 1);
            }

            let slice = alloc_ptr.as_slice_mut(3);
            assert_eq!(slice, &mut [1, 2, 3]);

            alloc_ptr.deallocate(3);
        }
    }

    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Debug)]
    struct DropCounter {
        count: Rc<RefCell<usize>>,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            *self.count.borrow_mut() += 1;
        }
    }

    #[test]
    fn test_alloc_ptr_drop_init() {
        let drop_count = Rc::new(RefCell::new(0));

        unsafe {
            let mut alloc_ptr: AllocationPointer<DropCounter> = AllocationPointer::new_allocate(3);

            for i in 0..3 {
                alloc_ptr.store(
                    i,
                    DropCounter {
                        count: Rc::clone(&drop_count),
                    },
                );
            }

            alloc_ptr.drop_initialized(0);
            assert_eq!(*drop_count.borrow(), 0);

            alloc_ptr.drop_initialized(3);

            assert_eq!(*drop_count.borrow(), 3);

            alloc_ptr.deallocate(3);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Drop range must not be empty")]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_ptr_drop_range_invalid() {
        unsafe {
            let mut alloc_ptr: AllocationPointer<u8> = AllocationPointer::new_allocate(5);
            alloc_ptr.drop_range(0..0);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_ptr_drop_range() {
        let drop_count = Rc::new(RefCell::new(0));

        unsafe {
            let mut alloc_ptr: AllocationPointer<DropCounter> = AllocationPointer::new_allocate(5);

            for i in 0..5 {
                alloc_ptr.store(
                    i,
                    DropCounter {
                        count: Rc::clone(&drop_count),
                    },
                );
            }

            alloc_ptr.drop_range(0..3);

            assert_eq!(*drop_count.borrow(), 3);

            alloc_ptr.deallocate(5);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pointer must not be null")]
    fn test_alloc_ptr_clone_empty() {
        let original: AllocationPointer<u8> = AllocationPointer::new();

        let _ = unsafe { original.make_clone(0) };
    }

    #[test]
    fn test_alloc_ptr_make_copy() {
        unsafe {
            let mut original: AllocationPointer<u8> = AllocationPointer::new_allocate(3);

            for i in 0..3 {
                original.store(i, i as u8 + 1);
            }

            let mut copied = original.make_copy(3);

            assert_ne!(copied.ptr.addr(), original.ptr.addr());

            for i in 0..3 {
                assert_eq!(*copied.reference(i), *original.reference(i));
            }

            *original.reference_mut(0) = 10;
            assert_eq!(*original.reference(0), 10);
            assert_eq!(*copied.reference(0), 1);

            *copied.reference_mut(0) = 11;
            assert_eq!(*copied.reference(0), 11);
            assert_eq!(*original.reference(0), 10);

            original.deallocate(3);
            copied.deallocate(3);
        }
    }

    #[test]
    fn test_alloc_ptr_make_clone() {
        unsafe {
            let mut original: AllocationPointer<String> = AllocationPointer::new_allocate(3);

            for i in 0..3 {
                original.store(i, (i + 1).to_string());
            }

            let mut cloned = original.make_clone(3);

            assert_ne!(cloned.ptr.addr(), original.ptr.addr());

            for i in 0..3 {
                assert_eq!(**cloned.reference(i), **original.reference(i));
            }

            original.reference_mut(0).push('0');
            assert_eq!(original.reference(0), "10");
            assert_eq!(cloned.reference(0), "1");

            cloned.reference_mut(0).push('1');
            assert_eq!(cloned.reference(0), "11");
            assert_eq!(original.reference(0), "10");

            original.drop_initialized(3);
            cloned.drop_initialized(3);

            original.deallocate(3);
            cloned.deallocate(3);
        }
    }

    #[test]
    fn test_cmp_eq() {
        unsafe {
            let mut alloc_ptr1: AllocationPointer<u8> = AllocationPointer::from_slice(&[1, 2, 3]);
            let mut alloc_ptr2: AllocationPointer<u8> = AllocationPointer::from_slice(&[1, 2, 3]);
            let mut alloc_ptr3: AllocationPointer<u8> = AllocationPointer::from_slice(&[3, 2, 1]);

            assert!(alloc_ptr1.cmp_eq(&alloc_ptr2, 3));
            assert!(!alloc_ptr1.cmp_eq(&alloc_ptr3, 3));

            alloc_ptr1.deallocate(3);
            alloc_ptr2.deallocate(3);
            alloc_ptr3.deallocate(3);
        }
    }

    #[test]
    fn test_from_slice() {
        let values = [1, 2, 3];
        unsafe {
            let mut alloc_ptr = AllocationPointer::from_slice(&values);
            for (i, &value) in values.iter().enumerate() {
                assert_eq!(*alloc_ptr.reference(i), value);
            }
            alloc_ptr.deallocate(values.len());
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size must be greater than 0")]
    fn test_from_slice_zero_len() {
        let values: [u8; 0] = [];
        unsafe {
            let _ = AllocationPointer::from_slice(&values);
        }
    }

    #[test]
    fn test_from_vec() {
        let values = vec![1, 2, 3];
        unsafe {
            let mut alloc_ptr = AllocationPointer::from_vec(values.clone());
            for (i, &value) in values.iter().enumerate() {
                assert_eq!(*alloc_ptr.reference(i), value);
            }
            alloc_ptr.deallocate(values.len());
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Allocation size must be greater than 0")]
    fn test_from_vec_not_allocated() {
        unsafe {
            let _ = AllocationPointer::from_vec(vec![()]);
        }
    }
}
