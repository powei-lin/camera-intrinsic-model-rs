use camera_intrinsic_model::model_from_json;
use nalgebra as na;
use rand::random;
use rayon::prelude::*;
use std::arch::aarch64::*;

#[inline(always)]
fn bench_eucm_simd_impl(
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
                // let mx = vdivq_f32(x, norm);
                // let my = vdivq_f32(y, norm);
                let norm_recip = vrecpeq_f32(norm); // 初步估算 reciprocal
                let norm_recip = vmulq_f32(vrecpsq_f32(norm, norm_recip), norm_recip); // refine 1
                let mx = vmulq_f32(x, norm_recip);
                let my = vmulq_f32(y, norm_recip);

                let px = vfmaq_f32(cx, fx, mx);
                let py = vfmaq_f32(cy, fy, my);
                vst1q_f32(outx.as_mut_ptr(), px);
                vst1q_f32(outy.as_mut_ptr(), py);
                p2dss[0].x = outx[0];
                p2dss[0].y = outy[0];
                p2dss[1].x = outx[1];
                p2dss[1].y = outy[1];
                p2dss[2].x = outx[2];
                p2dss[2].y = outy[2];
                p2dss[3].x = outx[3];
                p2dss[3].y = outy[3];
            }
        });
    p2ds
}

fn main() {
    let model1 = model_from_json("data/eucm0.json").cast();
    let params = model1.params().cast::<f32>();

    let mut p3ds = Vec::new();
    for _ in 0..1000 {
        p3ds.push(na::Vector3::new(rand::random(), random(), 1.0));
    }
    let p2ds0 = bench_eucm_simd_impl(&params, &p3ds);
    let p2ds1 = model1.project(&p3ds);
    p2ds0.iter().zip(p2ds1).for_each(|(p0, p1)| {
        if let Some(p1) = p1 {
            println!("p0\t{}\t{}", p0.x, p0.y);
            println!("p1\t{}\t{}\n", p0.x, p1.y);
        }
    });
}
