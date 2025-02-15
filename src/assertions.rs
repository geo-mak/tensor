use crate::Tensor;

#[inline(always)]
pub(crate) fn assert_same_dimensions<T>(a: &Tensor<T>, b: &Tensor<T>) {
    assert_eq!(
        a.dimensions, b.dimensions,
        "Tensors must have the same dimensions"
    );
}

#[inline(always)]
pub(crate) fn assert_valid_dimensions<T>(data: &[T], dimensions: &[usize]) {
    let data_count = data.len();
    let dim_count = dimensions.iter().product();
    assert_eq!(
        data_count, dim_count,
        "Dimensions's size doesn't match values' count"
    );
}
