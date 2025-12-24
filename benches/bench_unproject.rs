use camera_intrinsic_model::{GenericModel, OpenCVModel5, model_from_json};
use diol::prelude::*;
use nalgebra as na;

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register_many(
        "unproject_one",
        list![
            bench_eucm_unproject_one.with_name("eucm"),
            bench_eucmt_unproject_one.with_name("eucmt"),
            bench_fov_unproject_one.with_name("fov"),
            bench_ftheta_unproject_one.with_name("ftheta"),
            bench_kb4_unproject_one.with_name("kb4"),
            bench_opencv5_unproject_one.with_name("opencv5"),
            bench_ucm_unproject_one.with_name("ucm"),
        ],
        [None],
    );
    bench.register_many(
        "unproject_2000",
        list![
            bench_eucm_unproject_2000.with_name("eucm"),
            bench_eucmt_unproject_2000.with_name("eucmt"),
            bench_fov_unproject_2000.with_name("fov"),
            bench_ftheta_unproject_2000.with_name("ftheta"),
            bench_kb4_unproject_2000.with_name("kb4"),
            bench_opencv5_unproject_2000.with_name("opencv5"),
            bench_ucm_unproject_2000.with_name("ucm"),
        ],
        [None],
    );
    bench.run()?;
    Ok(())
}

fn bench_eucm_unproject_one(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/eucm0.json").cast::<f32>();
    let p2d = na::Vector2::new(100.0, 100.0);
    bencher.bench(|| {
        let _ = model.unproject_one(&p2d);
    });
}

fn bench_eucmt_unproject_one(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/eucmt.json").cast::<f32>();
    let p2d = na::Vector2::new(100.0, 100.0);
    bencher.bench(|| {
        let _ = model.unproject_one(&p2d);
    });
}

fn bench_fov_unproject_one(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/fov_tum_mono.json").cast::<f32>();
    let p2d = na::Vector2::new(100.0, 100.0);
    bencher.bench(|| {
        let _ = model.unproject_one(&p2d);
    });
}

fn bench_ftheta_unproject_one(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/ftheta.json").cast::<f32>();
    let p2d = na::Vector2::new(100.0, 100.0);
    bencher.bench(|| {
        let _ = model.unproject_one(&p2d);
    });
}

fn bench_kb4_unproject_one(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/kb4.json").cast::<f32>();
    let p2d = na::Vector2::new(100.0, 100.0);
    bencher.bench(|| {
        let _ = model.unproject_one(&p2d);
    });
}

fn bench_opencv5_unproject_one(bencher: Bencher, _: Option<bool>) {
    let params = na::DVector::from_vec(vec![
        500.0, 500.0, 320.0, 240.0, // fx, fy, cx, cy
        0.1, 0.01, 0.001, 0.001, 0.0, // k1, k2, p1, p2, k3
    ]);
    let model = GenericModel::OpenCVModel5(OpenCVModel5::new(&params, 640, 480));
    let model = model.cast::<f32>();
    let p2d = na::Vector2::new(100.0, 100.0);
    bencher.bench(|| {
        let _ = model.unproject_one(&p2d);
    });
}

fn bench_ucm_unproject_one(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/ucm.json").cast::<f32>();
    let p2d = na::Vector2::new(100.0, 100.0);
    bencher.bench(|| {
        let _ = model.unproject_one(&p2d);
    });
}

fn bench_eucm_unproject_2000(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/eucm0.json").cast::<f32>();
    let p2ds = vec![na::Vector2::new(100.0, 100.0); 2000];
    bencher.bench(|| {
        let _ = model.unproject(&p2ds);
    });
}

fn bench_eucmt_unproject_2000(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/eucmt.json").cast::<f32>();
    let p2ds = vec![na::Vector2::new(100.0, 100.0); 2000];
    bencher.bench(|| {
        let _ = model.unproject(&p2ds);
    });
}

fn bench_fov_unproject_2000(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/fov_tum_mono.json").cast::<f32>();
    let p2ds = vec![na::Vector2::new(100.0, 100.0); 2000];
    bencher.bench(|| {
        let _ = model.unproject(&p2ds);
    });
}

fn bench_ftheta_unproject_2000(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/ftheta.json").cast::<f32>();
    let p2ds = vec![na::Vector2::new(100.0, 100.0); 2000];
    bencher.bench(|| {
        let _ = model.unproject(&p2ds);
    });
}

fn bench_kb4_unproject_2000(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/kb4.json").cast::<f32>();
    let p2ds = vec![na::Vector2::new(100.0, 100.0); 2000];
    bencher.bench(|| {
        let _ = model.unproject(&p2ds);
    });
}

fn bench_opencv5_unproject_2000(bencher: Bencher, _: Option<bool>) {
    let params = na::DVector::from_vec(vec![
        500.0, 500.0, 320.0, 240.0, // fx, fy, cx, cy
        0.1, 0.01, 0.001, 0.001, 0.0, // k1, k2, p1, p2, k3
    ]);
    let model = GenericModel::OpenCVModel5(OpenCVModel5::new(&params, 640, 480));
    let model = model.cast::<f32>();
    let p2ds = vec![na::Vector2::new(100.0, 100.0); 2000];
    bencher.bench(|| {
        let _ = model.unproject(&p2ds);
    });
}

fn bench_ucm_unproject_2000(bencher: Bencher, _: Option<bool>) {
    let model = model_from_json("data/ucm.json").cast::<f32>();
    let p2ds = vec![na::Vector2::new(100.0, 100.0); 2000];
    bencher.bench(|| {
        let _ = model.unproject(&p2ds);
    });
}
