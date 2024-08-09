use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tensor::Tensor;

// Small tensor benchmarks
fn benchmark_addition_small(c: &mut Criterion) {
    c.bench_function("tensor_addition_small", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![100, 100], 1);
            let tensor2 = Tensor::new(vec![100, 100], 2);
            black_box(tensor1.add(&tensor2));
        })
    });
}

fn benchmark_add_mutation_small(c: &mut Criterion) {
    c.bench_function("tensor_add_mutation_small", |b| {
        b.iter(|| {
            let mut tensor1 = Tensor::new(vec![100, 100], 1);
            let tensor2 = Tensor::new(vec![100, 100], 2);
            black_box(tensor1.add_mutate(&tensor2));
        })
    });
}

fn benchmark_subtraction_small(c: &mut Criterion) {
    c.bench_function("tensor_subtraction_small", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![100, 100], 5);
            let tensor2 = Tensor::new(vec![100, 100], 3);
            black_box(tensor1.sub(&tensor2));
        })
    });
}

fn benchmark_mul_small(c: &mut Criterion) {
    c.bench_function("tensor_multiplication_small", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![100, 100], 2);
            let tensor2 = Tensor::new(vec![100, 100], 3);
            black_box(tensor1.mul(&tensor2));
        })
    });
}

fn benchmark_div_small(c: &mut Criterion) {
    c.bench_function("tensor_division_small", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![100, 100], 6);
            let tensor2 = Tensor::new(vec![100, 100], 3);
            black_box(tensor1.div(&tensor2));
        })
    });
}

// Large tensor benchmarks
fn benchmark_addition_large(c: &mut Criterion) {
    c.bench_function("tensor_addition_large", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10_000, 10_000], 1);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 2);
            black_box(tensor1.add(&tensor2));
        })
    });
}

fn benchmark_add_mutation_large(c: &mut Criterion) {
    c.bench_function("tensor_add_mutation_large", |b| {
        b.iter(|| {
            let mut tensor1 = Tensor::new(vec![10_000, 10_000], 1);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 2);
            black_box(tensor1.add_mutate(&tensor2));
        })
    });
}

fn benchmark_subtraction_large(c: &mut Criterion) {
    c.bench_function("tensor_subtraction_large", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10_000, 10_000], 5);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 3);
            black_box(tensor1.sub(&tensor2));
        })
    });
}

fn benchmark_mul_large(c: &mut Criterion) {
    c.bench_function("tensor_multiplication_large", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10_000, 10_000], 2);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 3);
            black_box(tensor1.mul(&tensor2));
        })
    });
}

fn benchmark_div_large(c: &mut Criterion) {
    c.bench_function("tensor_division_large", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10_000, 10_000], 6);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 3);
            black_box(tensor1.div(&tensor2));
        })
    });
}

// Small tensor benchmarks for casting
fn benchmark_casting_small(c: &mut Criterion) {
    c.bench_function("tensor_casting_small", |b| {
        b.iter(|| {
            let tensor_int = Tensor::new(vec![100, 100], 1);
            black_box(tensor_int.cast::<f64>());
        })
    });
}

// Large tensor benchmarks for casting
fn benchmark_casting_large(c: &mut Criterion) {
    c.bench_function("tensor_casting_large", |b| {
        b.iter(|| {
            let tensor_int = Tensor::new(vec![10_000, 10_000], 1);
            black_box(tensor_int.cast::<f64>());
        })
    });
}

// similarity measures benchmarks for 4D tensors
fn benchmark_dot_product_4d(c: &mut Criterion) {
    c.bench_function("tensor_dot_product_4d", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10, 10, 10, 10], 1.0);
            let tensor2 = Tensor::new(vec![10, 10, 10, 10], 2.0);
            black_box(tensor1.dot_product(&tensor2));
        })
    });
}

fn benchmark_cosine_similarity_4d(c: &mut Criterion) {
    c.bench_function("tensor_cosine_similarity_4d", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10, 10, 10, 10], 1.0);
            let tensor2 = Tensor::new(vec![10, 10, 10, 10], 2.0);
            black_box(tensor1.cosine_similarity(&tensor2));
        })
    });
}

fn benchmark_euclidean_distance_4d(c: &mut Criterion) {
    c.bench_function("tensor_euclidean_distance_4d", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10, 10, 10, 10], 1.0);
            let tensor2 = Tensor::new(vec![10, 10, 10, 10], 2.0);
            black_box(tensor1.euclidean_distance(&tensor2));
        })
    });
}

criterion_group!(
    small_benches,
    benchmark_addition_small,
    benchmark_add_mutation_small,
    benchmark_subtraction_small,
    benchmark_mul_small,
    benchmark_div_small,
    benchmark_casting_small
);

criterion_group!(
    large_benches,
    benchmark_addition_large,
    benchmark_add_mutation_large,
    benchmark_subtraction_large,
    benchmark_mul_large,
    benchmark_div_large,
    benchmark_casting_large
);

criterion_group!(
    similarity_benches_4d,
    benchmark_dot_product_4d,
    benchmark_cosine_similarity_4d,
    benchmark_euclidean_distance_4d
);

criterion_main!(small_benches, large_benches, similarity_benches_4d);
