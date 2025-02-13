use crate::Tensor;

#[inline(always)]
pub(crate) fn assert_same_dimensions<T>(a: &Tensor<T>, b: &Tensor<T>) {
    assert_eq!(
        a.dimensions, b.dimensions,
        "Tensors must have the same dimensions"
    );
}
