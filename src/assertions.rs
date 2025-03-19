use crate::Tensor;

/// Condition: Type `T` is not `ZST`.
pub(crate) const fn assert_not_zst<T>() {
    assert!(size_of::<T>() != 0, "Zero-sized types are not allowed")
}

/// Condition: The dimensions' size is not `0`.
pub(crate) const fn assert_non_zero_size(dims_size: usize) {
    assert!(
        dims_size != 0,
        "Invalid dimensions: dimensions' size must be greater than `0`"
    );
}

/// Condition: The values' count matches the dimensions' size (product).
pub(crate) const fn assert_same_size(data_len: usize, dims_size: usize) {
    assert!(
        data_len == dims_size,
        "Invalid shape: values' count doesn't match dimensions' size"
    )
}

/// Condition: Tensors of rank `R` must have the same value in each dimension.
pub(crate) const fn assert_same_shape<T, const R: usize>(a: &Tensor<T, R>, b: &Tensor<T, R>) {
    assert!(
        a.metadata.cmp_dims_eq(&b.metadata),
        "Tensors must have the same shape"
    );
}
