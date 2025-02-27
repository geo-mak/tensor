use crate::Tensor;

pub(crate) fn assert_same_dimensions<T>(a: &Tensor<T>, b: &Tensor<T>) {
    assert_eq!(
        a.dimensions, b.dimensions,
        "Tensors must have the same dimensions"
    );
}

pub(crate) fn assert_valid_dimensions(dims: &[usize], size: usize) {
    // size alone is unsound because [] will return 1 as product
    assert!(dims.len() > 0 && size > 0, "Invalid dimensions: {:?}", dims);
}

pub(crate) fn assert_valid_shape(len: usize, dimensions: &[usize]) {
    let size: usize = dimensions.iter().product();
    assert_valid_dimensions(dimensions, size);
    assert_eq!(
        len, size,
        "Invalid shape: Values' count is {}, but dimensions' size is {}",
        len, size
    );
}
