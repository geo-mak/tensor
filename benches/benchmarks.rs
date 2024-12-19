use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tensor::Tensor;

// Small tensor benchmarks
fn benchmark_add_small(c: &mut Criterion) {
    let tensor1 = Tensor::<i64>::new(vec![100, 100], 1);
    let tensor2 = Tensor::<i64>::new(vec![100, 100], 2);
    c.bench_function("tensor_addition_small", |b| {
        b.iter(|| {
            black_box(tensor1.add(&tensor2));
        })
    });
}

fn benchmark_add_mut_small(c: &mut Criterion) {
    let mut tensor1 = Tensor::<i64>::new(vec![100, 100], 1);
    let tensor2 = Tensor::<i64>::new(vec![100, 100], 2);
    c.bench_function("tensor_add_mutation_small", |b| {
        b.iter(|| {
            black_box(tensor1.add_mutate(&tensor2));
        })
    });
}

fn benchmark_sub_small(c: &mut Criterion) {
    let tensor1 = Tensor::<i64>::new(vec![100, 100], 5);
    let tensor2 = Tensor::<i64>::new(vec![100, 100], 3);
    c.bench_function("tensor_subtraction_small", |b| {
        b.iter(|| {
            black_box(tensor1.sub(&tensor2));
        })
    });
}

fn benchmark_sub_mut_small(c: &mut Criterion) {
    let mut tensor1 = Tensor::<i64>::new(vec![100, 100], 5);
    let tensor2 = Tensor::<i64>::new(vec![100, 100], 3);
    c.bench_function("tensor_subtraction_mutation_small", |b| {
        b.iter(|| {
            black_box(tensor1.sub_mutate(&tensor2));
        })
    });
}

fn benchmark_mul_small(c: &mut Criterion) {
    let tensor1 = Tensor::<i64>::new(vec![100, 100], 2);
    let tensor2 = Tensor::<i64>::new(vec![100, 100], 3);
    c.bench_function("tensor_multiplication_small", |b| {
        b.iter(|| {
            black_box(tensor1.mul(&tensor2));
        })
    });
}

fn benchmark_mul_mut_small(c: &mut Criterion) {
    let mut tensor1 = Tensor::<i64>::new(vec![100, 100], 2);
    let tensor2 = Tensor::<i64>::new(vec![100, 100], 3);
    c.bench_function("tensor_multiplication_mutation_small", |b| {
        b.iter(|| {
            black_box(tensor1.mul_mutate(&tensor2));
        })
    });
}

fn benchmark_div_small(c: &mut Criterion) {
    let tensor1 = Tensor::<i64>::new(vec![100, 100], 6);
    let tensor2 = Tensor::<i64>::new(vec![100, 100], 3);
    c.bench_function("tensor_division_small", |b| {
        b.iter(|| {
            black_box(tensor1.div(&tensor2));
        })
    });
}

fn benchmark_div_mut_small(c: &mut Criterion) {
    let mut tensor1 = Tensor::<i64>::new(vec![100, 100], 6);
    let tensor2 = Tensor::<i64>::new(vec![100, 100], 3);
    c.bench_function("tensor_division_mutation_small", |b| {
        b.iter(|| {
            black_box(tensor1.div_mutate(&tensor2));
        })
    });
}

fn benchmark_neg_small(c: &mut Criterion) {
    let tensor = Tensor::<i64>::new(vec![100, 100], 1);
    c.bench_function("tensor_negation_small", |b| {
        b.iter(|| {
            black_box(tensor.neg());
        })
    });
}

fn benchmark_neg_mut_small(c: &mut Criterion) {
    let mut tensor = Tensor::<i64>::new(vec![100, 100], 1);
    c.bench_function("tensor_negation_mutation_small", |b| {
        b.iter(|| {
            black_box(tensor.neg_mutate());
        })
    });
}

// Large tensor benchmarks
fn benchmark_add_large(c: &mut Criterion) {
    let tensor1 = Tensor::<i64>::new(vec![10_000, 10_000], 1);
    let tensor2 = Tensor::<i64>::new(vec![10_000, 10_000], 2);
    c.bench_function("tensor_addition_large", |b| {
        b.iter(|| {
            black_box(tensor1.add(&tensor2));
        })
    });
}

fn benchmark_add_mut_large(c: &mut Criterion) {
    let mut tensor1 = Tensor::<i64>::new(vec![10_000, 10_000], 1);
    let tensor2 = Tensor::<i64>::new(vec![10_000, 10_000], 2);
    c.bench_function("tensor_add_mutation_large", |b| {
        b.iter(|| {
            black_box(tensor1.add_mutate(&tensor2));
        })
    });
}

fn benchmark_sub_large(c: &mut Criterion) {
    let tensor1 = Tensor::<i64>::new(vec![10_000, 10_000], 5);
    let tensor2 = Tensor::<i64>::new(vec![10_000, 10_000], 3);
    c.bench_function("tensor_subtraction_large", |b| {
        b.iter(|| {
            black_box(tensor1.sub(&tensor2));
        })
    });
}

fn benchmark_sub_mut_large(c: &mut Criterion) {
    let mut tensor1 = Tensor::<i64>::new(vec![10_000, 10_000], 5);
    let tensor2 = Tensor::<i64>::new(vec![10_000, 10_000], 3);
    c.bench_function("tensor_subtraction_mutation_large", |b| {
        b.iter(|| {
            black_box(tensor1.sub_mutate(&tensor2));
        })
    });
}

fn benchmark_mul_large(c: &mut Criterion) {
    let tensor1 = Tensor::<i64>::new(vec![10_000, 10_000], 2);
    let tensor2 = Tensor::<i64>::new(vec![10_000, 10_000], 3);
    c.bench_function("tensor_multiplication_large", |b| {
        b.iter(|| {
            black_box(tensor1.mul(&tensor2));
        })
    });
}

fn benchmark_mul_mut_large(c: &mut Criterion) {
    let mut tensor1 = Tensor::<i64>::new(vec![10_000, 10_000], 2);
    let tensor2 = Tensor::<i64>::new(vec![10_000, 10_000], 3);
    c.bench_function("tensor_multiplication_mutation_large", |b| {
        b.iter(|| {
            black_box(tensor1.mul_mutate(&tensor2));
        })
    });
}

fn benchmark_div_large(c: &mut Criterion) {
    let tensor1 = Tensor::<i64>::new(vec![10_000, 10_000], 6);
    let tensor2 = Tensor::<i64>::new(vec![10_000, 10_000], 3);
    c.bench_function("tensor_division_large", |b| {
        b.iter(|| {
            black_box(tensor1.div(&tensor2));
        })
    });
}

fn benchmark_div_mut_large(c: &mut Criterion) {
    let mut tensor1 = Tensor::<i64>::new(vec![10_000, 10_000], 6);
    let tensor2 = Tensor::<i64>::new(vec![10_000, 10_000], 3);
    c.bench_function("tensor_division_mutation_large", |b| {
        b.iter(|| {
            black_box(tensor1.div_mutate(&tensor2));
        })
    });
}

fn benchmark_neg_large(c: &mut Criterion) {
    let tensor = Tensor::<i64>::new(vec![10_000, 10_000], 1);
    c.bench_function("tensor_negation_large", |b| {
        b.iter(|| {
            black_box(tensor.neg());
        })
    });
}

fn benchmark_neg_mut_large(c: &mut Criterion) {
    let mut tensor = Tensor::<i64>::new(vec![10_000, 10_000], 1);
    c.bench_function("tensor_negation_mutation_large", |b| {
        b.iter(|| {
            black_box(tensor.neg_mutate());
        })
    });
}

// Small tensor benchmarks for casting
fn benchmark_casting_small(c: &mut Criterion) {
    let tensor_int = Tensor::<i64>::new(vec![100, 100], 1);
    c.bench_function("tensor_casting_small", |b| {
        b.iter(|| {
            black_box(tensor_int.try_cast::<f64>().unwrap());
        })
    });
}

// Large tensor benchmarks for casting
fn benchmark_casting_large(c: &mut Criterion) {
    let tensor_int = Tensor::<i64>::new(vec![10_000, 10_000], 1);
    c.bench_function("tensor_casting_large", |b| {
        b.iter(|| {
            black_box(tensor_int.try_cast::<f64>().unwrap());
        })
    });
}

// similarity measures benchmarks for 4D tensors
fn benchmark_dot_product_4d(c: &mut Criterion) {
    let tensor1 = Tensor::<f64>::new(vec![10, 10, 10, 10], 1.0);
    let tensor2 = Tensor::<f64>::new(vec![10, 10, 10, 10], 2.0);
    c.bench_function("tensor_dot_product_4d", |b| {
        b.iter(|| {
            black_box(tensor1.dot_product(&tensor2));
        })
    });
}

fn benchmark_cosine_similarity_4d(c: &mut Criterion) {
    let tensor1 = Tensor::<f64>::new(vec![10, 10, 10, 10], 1.0);
    let tensor2 = Tensor::<f64>::new(vec![10, 10, 10, 10], 2.0);
    c.bench_function("tensor_cosine_similarity_4d", |b| {
        b.iter(|| {
            black_box(tensor1.cosine_similarity(&tensor2));
        })
    });
}

fn benchmark_euclidean_distance_4d(c: &mut Criterion) {
    let tensor1 = Tensor::<f64>::new(vec![10, 10, 10, 10], 1.0);
    let tensor2 = Tensor::<f64>::new(vec![10, 10, 10, 10], 2.0);
    c.bench_function("tensor_euclidean_distance_4d", |b| {
        b.iter(|| {
            black_box(tensor1.euclidean_distance(&tensor2));
        })
    });
}

criterion_group!(
    small_benches,
    benchmark_add_small,
    benchmark_add_mut_small,
    benchmark_sub_small,
    benchmark_sub_mut_small,
    benchmark_mul_small,
    benchmark_mul_mut_small,
    benchmark_div_small,
    benchmark_div_mut_small,
    benchmark_neg_small,
    benchmark_neg_mut_small,
    benchmark_casting_small
);

criterion_group!(
    large_benches,
    benchmark_add_large,
    benchmark_add_mut_large,
    benchmark_sub_large,
    benchmark_sub_mut_large,
    benchmark_mul_large,
    benchmark_mul_mut_large,
    benchmark_div_large,
    benchmark_div_mut_large,
    benchmark_neg_large,
    benchmark_neg_mut_large,
    benchmark_casting_large
);

criterion_group!(
    similarity_benches_4d,
    benchmark_dot_product_4d,
    benchmark_cosine_similarity_4d,
    benchmark_euclidean_distance_4d
);

criterion_main!(small_benches, large_benches, similarity_benches_4d);
