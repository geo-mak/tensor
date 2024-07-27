use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tensor::Tensor;

// Small tensor benchmarks
fn benchmark_addition_small(c: &mut Criterion) {
    c.bench_function("tensor_addition_small", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![100, 100], 1);
            let tensor2 = Tensor::new(vec![100, 100], 2);
            let result = tensor1.add(&tensor2);
            black_box(result);
        })
    });
}

fn benchmark_add_mutation_small(c: &mut Criterion) {
    c.bench_function("tensor_add_mutation_small", |b| {
        b.iter(|| {
            let mut tensor1 = Tensor::new(vec![100, 100], 1);
            let tensor2 = Tensor::new(vec![100, 100], 2);
            tensor1.add_mutate(&tensor2);
            black_box(&tensor1);
        })
    });
}

fn benchmark_subtraction_small(c: &mut Criterion) {
    c.bench_function("tensor_subtraction_small", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![100, 100], 5);
            let tensor2 = Tensor::new(vec![100, 100], 3);
            let result = tensor1.sub(&tensor2);
            black_box(result);
        })
    });
}

fn benchmark_mul_small(c: &mut Criterion) {
    c.bench_function("tensor_multiplication_small", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![100, 100], 2);
            let tensor2 = Tensor::new(vec![100, 100], 3);
            let result = tensor1.mul(&tensor2);
            black_box(result);
        })
    });
}

fn benchmark_div_small(c: &mut Criterion) {
    c.bench_function("tensor_division_small", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![100, 100], 6);
            let tensor2 = Tensor::new(vec![100, 100], 3);
            let result = tensor1.div(&tensor2);
            black_box(result);
        })
    });
}

// Large tensor benchmarks
fn benchmark_addition_large(c: &mut Criterion) {
    c.bench_function("tensor_addition_large", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10_000, 10_000], 1);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 2);
            let result = tensor1.add(&tensor2);
            black_box(result);
        })
    });
}

fn benchmark_add_mutation_large(c: &mut Criterion) {
    c.bench_function("tensor_add_mutation_large", |b| {
        b.iter(|| {
            let mut tensor1 = Tensor::new(vec![10_000, 10_000], 1);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 2);
            tensor1.add_mutate(&tensor2);
            black_box(&tensor1); // Ensure the mutation is not optimized away
        })
    });
}

fn benchmark_subtraction_large(c: &mut Criterion) {
    c.bench_function("tensor_subtraction_large", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10_000, 10_000], 5);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 3);
            let result = tensor1.sub(&tensor2);
            black_box(result);
        })
    });
}

fn benchmark_mul_large(c: &mut Criterion) {
    c.bench_function("tensor_multiplication_large", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10_000, 10_000], 2);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 3);
            let result = tensor1.mul(&tensor2);
            black_box(result);
        })
    });
}

fn benchmark_div_large(c: &mut Criterion) {
    c.bench_function("tensor_division_large", |b| {
        b.iter(|| {
            let tensor1 = Tensor::new(vec![10_000, 10_000], 6);
            let tensor2 = Tensor::new(vec![10_000, 10_000], 3);
            let result = tensor1.div(&tensor2);
            black_box(result);
        })
    });
}

// Small tensor benchmarks for casting
fn benchmark_casting_small(c: &mut Criterion) {
    c.bench_function("tensor_casting_small", |b| {
        b.iter(|| {
            let tensor_int = Tensor::new(vec![100, 100], 1);
            let result = tensor_int.cast::<f64>();
            black_box(result)
        })
    });
}

// Large tensor benchmarks for casting
fn benchmark_casting_large(c: &mut Criterion) {
    c.bench_function("tensor_casting_large", |b| {
        b.iter(|| {
            let tensor_int = Tensor::new(vec![10_000, 10_000], 1);
            let result = tensor_int.cast::<f64>();
            black_box(result)
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

criterion_main!(small_benches, large_benches);
