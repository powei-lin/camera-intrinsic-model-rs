use std::arch::aarch64::*;

use camera_intrinsic_model::{compute_for_fast_remap, fast_remap, model_from_json, remap};
use diol::prelude::*;
use image::{DynamicImage, ImageReader};
use nalgebra::{self as na};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

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
    bench.register_many(
        "project points",
        list![
            bench_eucm_project_points.with_name("eucm"),
            bench_eucm_simd.with_name("eucm simd"),
        ],
        [None],
    );
    bench.run()?;
    Ok(())
}

// #[target_feature(enable = "neon")]
unsafe fn bench_eucm_simd_impl(
    params: &na::DVector<f32>,
    p3ds: &[na::Vector3<f32>],
) -> Vec<na::Vector2<f32>> {
    let fx = params[0];
    let fy = params[1];
    let cx = params[2];
    let cy = params[3];
    let alpha = params[4];
    let beta = params[5];

    let fx = unsafe { vdupq_n_f32(fx) };
    let fy = unsafe { vdupq_n_f32(fy) };
    let cx = unsafe { vdupq_n_f32(cx) };
    let cy = unsafe { vdupq_n_f32(cy) };
    let alpha = unsafe { vdupq_n_f32(alpha) };
    let beta = unsafe { vdupq_n_f32(beta) };
    let one = unsafe { vdupq_n_f32(1.0) };
    let one_alpha = unsafe { vsubq_f32(one, alpha) };

    let mut p2ds = vec![na::Vector2::<f32>::zeros(); p3ds.len()];
    let chunk_size = 4;
    p3ds.par_chunks(chunk_size)
        .zip(p2ds.par_chunks_mut(chunk_size))
        .for_each(|(p3dss, p2dss)| {
            let x = [p3dss[0].x, p3dss[1].x, p3dss[2].x, p3dss[3].x];
            let y = [p3dss[0].y, p3dss[1].y, p3dss[2].y, p3dss[3].y];
            let z = [p3dss[0].z, p3dss[1].z, p3dss[2].z, p3dss[3].z];
            let mut outx = [0.0f32; 4];
            let mut outy = [0.0f32; 4];
            unsafe {
                let x = vld1q_f32(x.as_ptr());
                let y = vld1q_f32(y.as_ptr());
                let z = vld1q_f32(z.as_ptr());

                let xx = vmulq_f32(x, x);
                let yy = vmulq_f32(y, y);

                let rho2 = vfmaq_f32(vmulq_f32(z, z), beta, vaddq_f32(xx, yy));
                let rho = vsqrtq_f32(rho2);

                let ar = vmulq_f32(alpha, rho);
                let norm = vfmaq_f32(ar, one_alpha, z);
                let mx = vdivq_f32(x, norm);
                let my = vdivq_f32(y, norm);

                let px = vfmaq_f32(cx, fx, mx);
                let py = vfmaq_f32(cy, fy, my);
                vst1q_f32(outx.as_mut_ptr(), px);
                vst1q_f32(outy.as_mut_ptr(), py);
                for i in 0..4 {
                    p2dss[i].x = outx[i];
                    p2dss[i].y = outy[i];
                }
            }
        });
    p2ds
}
fn bench_eucm_simd(bencher: Bencher, _dummy: Option<bool>) {
    let model1 = model_from_json("data/eucm0.json");
    let params = model1.params().cast::<f32>();
    let p3ds = vec![na::Vector3::new(0.5, 0.5, 2.0); 3856 * 176];
    bencher.bench(|| {
        let _p2ds = unsafe { bench_eucm_simd_impl(&params, &p3ds) };
        // std::hint::black_box(p2ds);
    });
}

fn bench_eucm_project_points(bencher: Bencher, _dummy: Option<bool>) {
    let model1 = model_from_json("data/eucm0.json").cast::<f32>();
    let p3ds = vec![na::Vector3::new(0.5, 0.5, 2.0); 3856 * 176];
    bencher.bench(|| {
        let _p2ds = model1.project(&p3ds);
    });
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
    let xy_pos_weight = compute_for_fast_remap(&xmap, &ymap, img_l8.width() as usize);
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
    let xy_pos_weight = compute_for_fast_remap(&xmap, &ymap, img_rgb.width() as usize);
    bencher.bench(|| {
        let _ = fast_remap(&img_rgb, (new_w_h, new_w_h), &xy_pos_weight);
    });
}
