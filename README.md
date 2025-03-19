# Tensor
[![CI](https://github.com/geo-mak/tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/geo-mak/tensor/actions/workflows/ci.yml)

Multidimensional tensor implementation in Rust with focus on simplicity in terms of usage and implementation.

## Features

- Relatively simple and lightweight implementation that relies mostly on `core` intrinsics.
- Variety of operations with great attention to usability and performance.
- Explicit and transparent regarding its operational semantics.

## Examples

###  Creating new tensor

Tensors can be created in multiple ways and from multiple data structures.

Creating new tensor with initial value:

```rust
use tensor::Tensor;

fn main() {
    // Creates tensor of rank 2 with `0` as initial value.
    let mut tensor = Tensor::new_set([2, 3], 0);
    
    assert_eq!(tensor.get(&[0, 0]), &0);
}
```

Creating new tensor declaratively:

```rust
use tensor::tensor;

fn main() {
    let tensor = tensor![[
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-7.0, -8.0, -9.0], [-10.0, -11.0, -12.0]]
        ]];

    assert_eq!(tensor.shape(), &[2, 2, 3]);
}
```

###  Reading and mutating values.

Tensor is a `collection` type and like many collection types, values can be access and mutated individually.

```rust
use tensor::Tensor;

fn main() {
    let mut tensor = Tensor::new_set([2, 3], 0);
    tensor[&[0, 2]] = 100;
    assert_eq!(tensor[&[0, 2]], 100);
}
```

## Operations

Current operations have `eager` execution. Defining lazy graph execution is not yet supported.

Many operations have multiple variants with the same result but with different effects.

### Addition

The state of the binding affects the operational semantics regarding the memory of the input tensors.

Using an immutable reference returns a new tensor as a result without affecting the input tensors.

```rust
use tensor::Tensor;

fn main() {
    let tensor1 = Tensor::new_set([2, 2], 1);
    let tensor2 = Tensor::new_set([2, 2], 2);

    // Addition as immutable references.
    let tensor3 = &tensor1 + &tensor2;

    assert_eq!(tensor3.get(&[0, 0]), &3);

    // Original `lhs` is unaffected.
    assert_eq!(tensor1.get(&[0, 0]), &1);

    // `rhs` can be a scalar also.
    let result = &tensor3 + 2;

    assert_eq!(result.get(&[0, 0]), &5);

    // Original `lhs` is unaffected.
    assert_eq!(tensor3.get(&[0, 0]), &3);
}
```

Using a mutable reference performs addition in-place by mutating `lhs` without consuming the input tensors.

```rust
use tensor::Tensor;

fn main() {
    let mut tensor1 = Tensor::new_set([2, 2], 1);
    let tensor2 = Tensor::new_set([2, 2], 2);
    
    // In-place addition, tensor2 remains accessible in the scope until consumed or dropped later.
    &mut tensor1 + &tensor2;
    
    assert_eq!(tensor1.get(&[0, 0]), &3);

    // `rhs` as scalar.
    &mut tensor1 + 2;

    assert_eq!(tensor1.get(&[0, 0]), &5);
}
```

###  Reshaping

Reshaping doesn't reorder values, only the index is changed and the memory layout is always maintained.

Reshaping without changing the rank `R` of the tensor:

```rust
use tensor::tensor;

fn main() {
    let mut tensor_r2 = tensor![i32: [[1,2, 3], [4,5,6]]];

    // Current shape.
    assert_eq!(tensor_r2.shape(), &[2, 3]);

    assert_eq!(tensor_r2.get(&[0, 0]), &1);
    assert_eq!(tensor_r2.get(&[0, 1]), &2);
    assert_eq!(tensor_r2.get(&[0, 2]), &3);
    assert_eq!(tensor_r2.get(&[1, 0]), &4);
    assert_eq!(tensor_r2.get(&[1, 1]), &5);
    assert_eq!(tensor_r2.get(&[1, 2]), &6);

    tensor_r2.reshape(&[3, 2]);

    assert_eq!(tensor_r2.shape(), &[3, 2]);

    assert_eq!(tensor_r2.get(&[0, 0]), &1);
    assert_eq!(tensor_r2.get(&[0, 1]), &2);
    assert_eq!(tensor_r2.get(&[1, 0]), &3);
    assert_eq!(tensor_r2.get(&[1, 1]), &4);
    assert_eq!(tensor_r2.get(&[2, 0]), &5);
    assert_eq!(tensor_r2.get(&[2, 1]), &6);
}
```

Reshaping with different rank:

The rank `R` of the tensor is part of its static type definition, so reshaping to different rank `N` requires <br>
new (type) instance. This transformation is very cheap, but will consume the current instance.

```rust
use tensor::{Tensor, tensor};

fn main() {
    let mut tensor_r2 = tensor![[[1,2, 3], [4,5,6]]];

    // Current shape.
    assert_eq!(tensor_r2.shape(), &[2, 3]);
    
    let tensor_r3 = tensor_r2.change_rank([2, 3, 1]);
    
    assert_eq!(tensor_r3.shape(), &[2, 3, 1]);
    
    assert_eq!(tensor_r3.get(&[0, 0, 0]), &1);
    assert_eq!(tensor_r3.get(&[1, 2, 0]), &6);
}
```

### Dot product

```rust
use tensor::Tensor;

fn main() {
    let tensor1 = Tensor::new_set([2, 2, 2, 2], 1.0);
    let tensor2 = Tensor::new_set([2, 2, 2, 2], 2.0);
    let result = tensor1.dot_product(&tensor2);
    let expected: f64 = 32.0;
    assert!((result - expected).abs() < 1e-6);
}
```

### Euclidean distance

```rust
use tensor::Tensor;

fn main() {
    let tensor1 = Tensor::new_set([2, 2, 2, 2], 1.0);
    let tensor2 = Tensor::new_set([2, 2, 2, 2], 2.0);
    let result = tensor1.euclidean_distance(&tensor2);
    let expected: f64 = 4.0;
    assert!((result - expected).abs() < 1e-6);
}
```

### Cosine similarity

```rust
use tensor::Tensor;

fn main() {
    let tensor1 = Tensor::new_set([2, 2, 2, 2], 1.0);
    let tensor2 = Tensor::new_set([2, 2, 2, 2], 2.0);
    let result = tensor1.cosine_similarity(&tensor2);
    let expected: f64 = 1.0;
    assert!((result - expected).abs() < 1e-6);
}
```

### Casting
Currently, casting relies on the `TryCast` trait, which the source type must implement.
Casting using `TryCast` trait can fail if the casting process results in precision loss or overflow.
A blanket implementation of `TryCast` is provided for all numeric types that __could__ be cast to each other.

```rust
use tensor::Tensor;

fn main() {
    let tensor_int = Tensor::new_set([2, 2], 2);

    // Attempt to cast the tensor to f64
    let tensor_f64 = tensor_int.try_cast::<f64>().unwrap();

    assert_eq!(tensor_f64.get(&[0, 0]), &2.0);
}
```

### Console display 
```rust
use tensor::tensor;

fn main() {
    let mut tensor = tensor![[[[1, 2], [3, 4]], [[5, 6],[7, 8]]]];
    
    // print tensor to console
    println!("{}", tensor);
}
```
```text
Output:
Shape: [2, 2, 2]
Data:
0: [0, 0, 0] -> 1
1: [0, 0, 1] -> 2
2: [0, 1, 0] -> 3
3: [0, 1, 1] -> 4
4: [1, 0, 0] -> 5
5: [1, 0, 1] -> 6
6: [1, 1, 0] -> 7
7: [1, 1, 1] -> 8
```
