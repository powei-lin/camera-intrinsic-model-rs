use camera_intrinsic_model::{compute_for_fast_remap, fast_remap, remap};
use image::{DynamicImage, GrayImage};
use nalgebra as na;

#[test]
fn test_fast_remap_matches_remap_mono() {
    let w = 100;
    let h = 100;
    // Create a source image slightly larger to avoid "fast_remap" OOB at edges
    // fast_remap reads at floor(x)+1. If x=99.5, floor=99, reads 100. Need width 101.
    let src_w = w + 2;
    let src_h = h + 2;
    let mut img_buf = GrayImage::new(src_w, src_h);
    for y in 0..src_h {
        for x in 0..src_w {
            img_buf.put_pixel(x, y, image::Luma([(x + y) as u8]));
        }
    }
    let img = DynamicImage::ImageLuma8(img_buf);

    // Create identity map with slight offset/warp
    let mut xmap = na::DMatrix::<f32>::zeros(h as usize, w as usize);
    let mut ymap = na::DMatrix::<f32>::zeros(h as usize, w as usize);
    for y in 0..h {
        for x in 0..w {
            xmap[(y as usize, x as usize)] = x as f32 + 0.5;
            ymap[(y as usize, x as usize)] = y as f32 + 0.5;
        }
    }

    let remap_res = remap(&img, &xmap, &ymap);
    let xy_pos = compute_for_fast_remap(&xmap, &ymap, img.width() as usize);
    // Target size is w, h
    let fast_res = fast_remap(&img, (w, h), &xy_pos);

    let remap_buf = remap_res.to_luma8();
    let fast_buf = fast_res.to_luma8();

    let mut diff_sum = 0u64;
    for (p1, p2) in remap_buf.pixels().zip(fast_buf.pixels()) {
        let v1 = p1.0[0] as i32;
        let v2 = p2.0[0] as i32;
        diff_sum += (v1 - v2).abs() as u64;
    }
    let avg_diff = diff_sum as f64 / (w * h) as f64;
    println!("Average diff: {}", avg_diff);
    // Allow small difference due to precision
    assert!(
        avg_diff < 2.0,
        "Difference between remap and fast_remap is too large: {}",
        avg_diff
    );
}
