# Tensor
[![CI](https://github.com/geo-mak/tensor/actions/workflows/ci.yml/badge.svg)](https://github.com/geo-mak/tensor/actions/workflows/ci.yml)

Multidimensional tensor implementation in Rust with focus on simplicity in terms of usage and implementation.

Created as a test environment for numerical stuff in Rust and their hardware-based acceleration and optimization techniques.

Currently, only CPU-based operations are supported without fancy optimizations and accelerations.

## Features

- Element-wise operations: addition, subtraction, multiplication, division
- Reshaping tensors
- Casting to different types
- In-place modifications
- Direct and safe indexing
- Display formatted tensor output
- Similarity measures: Dot product, Euclidean distance, Cosine similarity

## Examples

###  Creating tensor with set and get

```rust
use tensor::Tensor;

fn main(){
    let mut tensor = Tensor::new(vec![2, 3], 0);
    tensor.set(&[0, 0], 1);
    tensor.set(&[0, 1], 2);
    tensor.set(&[0, 2], 3);
    tensor.set(&[1, 0], 4);
    tensor.set(&[1, 1], 5);
    tensor.set(&[1, 2], 6);

    assert_eq!(tensor.get(&[0, 0]), &1);
    assert_eq!(tensor.get(&[0, 1]), &2);
    assert_eq!(tensor.get(&[0, 2]), &3);
    assert_eq!(tensor.get(&[1, 0]), &4);
    assert_eq!(tensor.get(&[1, 1]), &5);
    assert_eq!(tensor.get(&[1, 2]), &6);
}
```

###  Creating tensor using the declarative syntax
This example requires feature `builder`.

```rust
use tensor::tensor;

fn main(){
    let tensor = tensor![i32: [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]];

    assert_eq!(tensor.shape(), &[2, 2, 3]);

    assert_eq!(tensor.get(&[0, 0, 0]), &1);
    assert_eq!(tensor.get(&[0, 0, 1]), &2);
    assert_eq!(tensor.get(&[0, 0, 2]), &3);
    assert_eq!(tensor.get(&[0, 1, 0]), &4);
    assert_eq!(tensor.get(&[0, 1, 1]), &5);
    assert_eq!(tensor.get(&[0, 1, 2]), &6);
    assert_eq!(tensor.get(&[1, 0, 0]), &7);
    assert_eq!(tensor.get(&[1, 0, 1]), &8);
    assert_eq!(tensor.get(&[1, 0, 2]), &9);
    assert_eq!(tensor.get(&[1, 1, 0]), &10);
    assert_eq!(tensor.get(&[1, 1, 1]), &11);
    assert_eq!(tensor.get(&[1, 1, 2]), &12);
}
```

###  Set and get by index

```rust
use tensor::Tensor;

fn main(){
    let mut tensor = Tensor::new(vec![2, 3], 0);
    tensor[&[0, 2]] = 100;
    assert_eq!(tensor[&[0, 2]], 100);
}
```

###  Reshaping the tensor

```rust
use tensor::tensor;

fn main(){
    let mut tensor = tensor![i32: [[1,2, 3], [4,5,6]]];

    // Current shape.
    assert_eq!(tensor.shape(), &[2, 3]);

    assert_eq!(tensor.get(&[0, 0]), &1);
    assert_eq!(tensor.get(&[0, 1]), &2);
    assert_eq!(tensor.get(&[0, 2]), &3);
    assert_eq!(tensor.get(&[1, 0]), &4);
    assert_eq!(tensor.get(&[1, 1]), &5);
    assert_eq!(tensor.get(&[1, 2]), &6);

    tensor.reshape(&[3, 2]);

    assert_eq!(tensor.shape(), &[3, 2]);

    assert_eq!(tensor.get(&[0, 0]), &1);
    assert_eq!(tensor.get(&[0, 1]), &2);
    assert_eq!(tensor.get(&[1, 0]), &3);
    assert_eq!(tensor.get(&[1, 1]), &4);
    assert_eq!(tensor.get(&[2, 0]), &5);
    assert_eq!(tensor.get(&[2, 1]), &6);
}
```

### Tensor addition

```rust
use tensor::Tensor;

fn main(){
    let tensor1 = Tensor::new(vec![2, 2], 1);
    let tensor2 = Tensor::new(vec![2, 2], 2);
    let result = tensor1.add(&tensor2);

    assert_eq!(result.get(&[0, 0]), &3);
    assert_eq!(result.get(&[0, 1]), &3);
    assert_eq!(result.get(&[1, 0]), &3);
    assert_eq!(result.get(&[1, 1]), &3);
}
```

### Tensor subtraction

```rust
use tensor::Tensor;

fn main(){
    let tensor1 = Tensor::new(vec![2, 2], 5);
    let tensor2 = Tensor::new(vec![2, 2], 3);
    let result = tensor1.sub(&tensor2);

    assert_eq!(result.get(&[0, 0]), &2);
    assert_eq!(result.get(&[0, 1]), &2);
    assert_eq!(result.get(&[1, 0]), &2);
    assert_eq!(result.get(&[1, 1]), &2);
}
```

### Tensor multiplication

```rust
use tensor::Tensor;

fn main(){
    let tensor1 = Tensor::new(vec![2, 2], 2);
    let tensor2 = Tensor::new(vec![2, 2], 3);
    let result = tensor1.mul(&tensor2);

    assert_eq!(result.get(&[0, 0]), &6);
    assert_eq!(result.get(&[0, 1]), &6);
    assert_eq!(result.get(&[1, 0]), &6);
    assert_eq!(result.get(&[1, 1]), &6);
}
```

### Tensor division

```rust
use tensor::Tensor;

fn main(){
    let tensor1 = Tensor::new(vec![2, 2], 6);
    let tensor2 = Tensor::new(vec![2, 2], 3);
    let result = tensor1.div(&tensor2);

    assert_eq!(result.get(&[0, 0]), &2);
    assert_eq!(result.get(&[0, 1]), &2);
    assert_eq!(result.get(&[1, 0]), &2);
    assert_eq!(result.get(&[1, 1]), &2);
}
```

### Tensor addition in-place

```rust
use tensor::Tensor;

fn main(){
    let mut tensor1 = Tensor::new(vec![2, 2], 1);
    let tensor2 = Tensor::new(vec![2, 2], 2);
    tensor1.add_mutate(&tensor2);
    assert_eq!(tensor1.data, vec![3, 3, 3, 3]);
}
```

### Tensor subtraction in-place

```rust
use tensor::Tensor;

fn main(){
    let mut tensor1 = Tensor::new(vec![2, 2], 5);
    let tensor2 = Tensor::new(vec![2, 2], 3);
    tensor1.sub_mutate(&tensor2);
    assert_eq!(tensor1.data, vec![2, 2, 2, 2]);
}
```

### Tensor multiplication in-place

```rust
use tensor::Tensor;

fn main(){
    let mut tensor1 = Tensor::new(vec![2, 2], 3);
    let tensor2 = Tensor::new(vec![2, 2], 4);
    tensor1.mul_mutate(&tensor2);
    assert_eq!(tensor1.data, vec![12, 12, 12, 12]);
}
```

### Tensor division in-place

```rust
use tensor::Tensor;

fn main(){
    let mut tensor1 = Tensor::new(vec![2, 2], 8);
    let tensor2 = Tensor::new(vec![2, 2], 4);
    tensor1.div_mutate(&tensor2);
    assert_eq!(tensor1.data, vec![2, 2, 2, 2]);
}
```

### Tensor negative conversion in-place

```rust
use tensor::Tensor;

fn main(){
    let mut tensor = Tensor::new(vec![2, 2], 5);
    tensor.neg_mutate();
    assert_eq!(tensor.data, vec![-5, -5, -5, -5]);
}
```

### Dot product

```rust
use tensor::Tensor;

fn main(){
    let tensor1 = Tensor::new(vec![2, 2, 2, 2], 1.0);
    let tensor2 = Tensor::new(vec![2, 2, 2, 2], 2.0);
    let result = tensor1.dot_product(&tensor2);
    let expected: f64 = 32.0;
    assert!((result - expected).abs() < 1e-6);
}
```

### Euclidean distance

```rust
use tensor::Tensor;

fn main(){
    let tensor1 = Tensor::new(vec![2, 2, 2, 2], 1.0);
    let tensor2 = Tensor::new(vec![2, 2, 2, 2], 2.0);
    let result = tensor1.euclidean_distance(&tensor2);
    let expected: f64 = 4.0;
    assert!((result - expected).abs() < 1e-6);
}
```

### Cosine similarity

```rust
use tensor::Tensor;

fn main(){
    let tensor1 = Tensor::new(vec![2, 2, 2, 2], 1.0);
    let tensor2 = Tensor::new(vec![2, 2, 2, 2], 2.0);
    let result = tensor1.cosine_similarity(&tensor2);
    let expected: f64 = 1.0;
    assert!((result - expected).abs() < 1e-6);
}
```

### Tensor casting
Casting relies on the 'TryCast' trait, which the source type must implement.
Casting using 'TryCast' trait can fail if the casting process results in precision loss, overflow.
A blanket implementation of 'TryCast' is provided for all numeric types that __could__ be cast to each other.

```rust
use tensor::Tensor;

fn main(){
    // Create a tensor with floating-point numbers
    let tensor1 = Tensor::new(vec![2, 2], 1.0);

    // Create a tensor with integer numbers and cast to floating-point
    let tensor2 = Tensor::new(vec![2, 2], 2);
    let tensor2_float = tensor2.try_cast::<f64>().unwrap();

    // Perform tensor operations
    let result_add = tensor1.add(&tensor2_float);
    let result_sub = tensor1.sub(&tensor2_float);
    let result_mul = tensor1.mul(&tensor2_float);
    let result_div = tensor1.div(&tensor2_float);

    // Verify results for addition
    let expected_add = Tensor::new(vec![2, 2], 3.0);
    assert_eq!(result_add, expected_add);

    // Verify results for subtraction
    let expected_sub = Tensor::new(vec![2, 2], -1.0);
    assert_eq!(result_sub, expected_sub);

    // Verify results for multiplication
    let expected_mul = Tensor::new(vec![2, 2], 2.0);
    assert_eq!(result_mul, expected_mul);

    // Verify results for division
    let expected_div = Tensor::new(vec![2, 2], 0.5);
    assert_eq!(result_div, expected_div);
}
```

### Tensor display in console
```rust
use tensor::tensor;

fn main(){
    let mut tensor = tensor![i32: [[[1,2],[3,4]], [[5,6],[7, 8]]]];
    
    // print tensor to console
    println!("{}", tensor);
}
```
```text
Output:
0: [0, 0, 0] -> 1
1: [0, 0, 1] -> 2
2: [0, 1, 0] -> 3
3: [0, 1, 1] -> 4
4: [1, 0, 0] -> 5
5: [1, 0, 1] -> 6
6: [1, 1, 0] -> 7
7: [1, 1, 1] -> 8
```
