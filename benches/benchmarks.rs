use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tensor::Tensor;

fn bench_set(c: &mut Criterion) {
    let mut t = Tensor::<u8, 3>::new_set([10, 10, 10], 3);
    c.bench_function("tensor, u8, set, R=3, N=1e3", |b| {
        b.iter(|| {
            black_box(t.set(&[5, 5, 5], 5));
        })
    });
}

fn bench_get(c: &mut Criterion) {
    let t = Tensor::<u8, 3>::new_set([10, 10, 10], 3);
    c.bench_function("tensor, u8, get, R=3, N=1e3", |b| {
        b.iter(|| {
            black_box(t.get(&[5, 5, 5]));
        })
    });
}

fn bench_reshape(c: &mut Criterion) {
    let mut t = Tensor::<u8, 3>::new_set([10, 10, 10], 3);
    c.bench_function("tensor, u8, reshape, R=3, N=1e3", |b| {
        b.iter(|| {
            black_box(t.reshape([2, 5, 100]));
        })
    });
}

fn bench_add_new_1e6(c: &mut Criterion) {
    let t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, add_new, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(&t1 + &t2);
        })
    });
}

fn bench_add_mut_1e6(c: &mut Criterion) {
    let mut t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, add_mut, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(&mut t1 + &t2);
        })
    });
}

fn bench_sub_1e6(c: &mut Criterion) {
    let t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, sub, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(&t1 - &t2);
        })
    });
}

fn bench_sub_mut_1e6(c: &mut Criterion) {
    let mut t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, sub_mut, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(&mut t1 - &t2);
        })
    });
}

fn bench_mul_1e6(c: &mut Criterion) {
    let t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, mul, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(&t1 * &t2);
        })
    });
}

fn bench_mul_mut_1e6(c: &mut Criterion) {
    let mut t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, mul_mut, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(&mut t1 * &t2);
        })
    });
}

fn bench_div_1e6(c: &mut Criterion) {
    let t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, div, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(&t1 / &t2);
        })
    });
}

fn bench_div_mut_1e6(c: &mut Criterion) {
    let mut t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, div_mut, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(&mut t1 / &t2);
        })
    });
}

fn bench_neg_1e6(c: &mut Criterion) {
    let t = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, neg, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(-&t);
        })
    });
}

fn bench_neg_mut_1e6(c: &mut Criterion) {
    let mut t = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, neg_mut, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(-&mut t);
        })
    });
}

fn bench_cast_1e6(c: &mut Criterion) {
    let t = Tensor::<i8, 1>::new_set([1_000_000], -3);
    c.bench_function("tensor, i8, cast -> f64, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(t.try_cast::<f64>().unwrap());
        })
    });
}

fn bench_dot_product_1e6(c: &mut Criterion) {
    let t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, dot product, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(t1.dot_product(&t2));
        })
    });
}

fn bench_cosine_similarity_1e6(c: &mut Criterion) {
    let t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, cosine similarity, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(t1.cosine_similarity(&t2));
        })
    });
}

fn bench_euclidean_distance_1e6(c: &mut Criterion) {
    let t1 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    let t2 = Tensor::<f64, 1>::new_set([1_000_000], 3.0);
    c.bench_function("tensor, f64, euclidean distance, R=1, N=1e6", |b| {
        b.iter(|| {
            black_box(t1.euclidean_distance(&t2));
        })
    });
}

criterion_group!(benches_core_ops, bench_set, bench_get, bench_reshape,);

criterion_group!(
    benches_special_ops,
    bench_add_new_1e6,
    bench_add_mut_1e6,
    bench_sub_1e6,
    bench_sub_mut_1e6,
    bench_mul_1e6,
    bench_mul_mut_1e6,
    bench_div_1e6,
    bench_div_mut_1e6,
    bench_neg_1e6,
    bench_neg_mut_1e6,
    bench_cast_1e6,
    bench_dot_product_1e6,
    bench_cosine_similarity_1e6,
    bench_euclidean_distance_1e6
);

criterion_main!(benches_core_ops, benches_special_ops);
