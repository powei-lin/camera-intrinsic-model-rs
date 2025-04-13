use camera_intrinsic_model::{compute_for_fast_remap, fast_remap, model_from_json, remap};
use diol::prelude::*;
use image::{DynamicImage, ImageReader};

fn main() -> eyre::Result<()> {
    let bench = Bench::from_args()?;
    bench.register_many(
        "remap",
        list![
            bench_remap.with_name("mono8 normal"),
            bench_remap_fast.with_name("mono8 fast"),
            bench_remap_rgb.with_name("rgb8 normal"),
            bench_remap_fast_rgb.with_name("rgb8 fast"),
        ],
        [None],
    );
    bench.run()?;
    Ok(())
}

fn bench_remap(bencher: Bencher, _dummy: Option<bool>) {
    let model1 = model_from_json("data/eucm0.json");
    let new_w_h = 1024;
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    let p = model1.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = model1.init_undistort_map(&p, (new_w_h, new_w_h), None);
    let img_l8 = DynamicImage::ImageLuma8(img.to_luma8());
    bencher.bench(|| {
        let _ = remap(&img_l8, &xmap, &ymap);
    });
}

fn bench_remap_rgb(bencher: Bencher, _dummy: Option<bool>) {
    let model1 = model_from_json("data/eucm0.json");
    let new_w_h = 1024;
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    let p = model1.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = model1.init_undistort_map(&p, (new_w_h, new_w_h), None);
    let img_rgb8 = DynamicImage::ImageRgb8(img.to_rgb8());
    bencher.bench(|| {
        let _ = remap(&img_rgb8, &xmap, &ymap);
    });
}

fn bench_remap_fast(bencher: Bencher, _dummy: Option<bool>) {
    let model1 = model_from_json("data/eucm0.json");
    let new_w_h = 1024;
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    let p = model1.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = model1.init_undistort_map(&p, (new_w_h, new_w_h), None);
    let img_l8 = DynamicImage::ImageLuma8(img.to_luma8());
    let xy_pos_weight = compute_for_fast_remap(&xmap, &ymap);
    bencher.bench(|| {
        let _ = fast_remap(&img_l8, (new_w_h, new_w_h), &xy_pos_weight);
    });
}

fn bench_remap_fast_rgb(bencher: Bencher, _dummy: Option<bool>) {
    let model1 = model_from_json("data/eucm0.json");
    let new_w_h = 1024;
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    let p = model1.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = model1.init_undistort_map(&p, (new_w_h, new_w_h), None);
    let img_rgb = DynamicImage::ImageRgb8(img.to_rgb8());
    let xy_pos_weight = compute_for_fast_remap(&xmap, &ymap);
    bencher.bench(|| {
        let _ = fast_remap(&img_rgb, (new_w_h, new_w_h), &xy_pos_weight);
    });
}
