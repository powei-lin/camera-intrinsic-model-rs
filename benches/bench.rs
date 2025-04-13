use camera_intrinsic_model::{model_from_json, remap};
use diol::prelude::*;
use image::ImageReader;

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register("remap", bench_remap, [true]);
    bench.run()?;
    Ok(())
}

fn bench_remap(bencher: Bencher, _dummy: bool) {
    let model1 = model_from_json("data/eucm0.json");
    let new_w_h = 1024;
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    let p = model1.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = model1.init_undistort_map(&p, (new_w_h, new_w_h), None);
    bencher.bench(|| {
        let remaped = remap(&img, &xmap, &ymap);
    });
}
