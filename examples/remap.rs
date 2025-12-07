use camera_intrinsic_model::*;
use image::{DynamicImage, ImageReader};
use nalgebra as na;

fn main() {
    let img = ImageReader::open("data/tum_vi_with_chart.png")
        .unwrap()
        .decode()
        .unwrap();
    // let img = image::DynamicImage::ImageLuma8(img.to_luma8());
    let params = na::dvector![
        190.89618687183938,
        190.87022285882367,
        254.9375370481962,
        256.86414483060787,
        0.6283550447635853,
        1.0458678747533083
    ];
    let model0 = eucm::EUCM::new(&params, 512, 512);
    model_to_json("eucm.json", &GenericModel::EUCM(model0));
    let model1 = model_from_json("data/eucm0.json");
    let new_w_h = 1024;
    let p = model1.estimate_new_camera_matrix_for_undistort(0.0, Some((new_w_h, new_w_h)));
    let (xmap, ymap) = model1.init_undistort_map(&p, (new_w_h, new_w_h), None);
    let img_l8 = DynamicImage::ImageLuma8(img.to_luma8());
    let remaped = remap(&img_l8, &xmap, &ymap);
    remaped.save("remaped0.png").unwrap();

    // let xy_pos_weight = compute_for_fast_remap(&xmap, &ymap);
    let xy_pos = compute_for_fast_remap(&xmap, &ymap, img.width() as usize);
    // Target size is w, h
    let remaped1 = fast_remap(&img, (new_w_h, new_w_h), &xy_pos);
    remaped1.save("remaped1.png").unwrap();

    let img_rgb8 = DynamicImage::ImageRgb8(img.to_rgb8());
    let remaped1 = fast_remap(&img_rgb8, (new_w_h, new_w_h), &xy_pos);
    remaped1.save("remaped2.png").unwrap();
}
