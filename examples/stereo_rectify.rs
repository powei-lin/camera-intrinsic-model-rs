use camera_intrinsic_model::*;
use image::{ImageReader, Rgb, Rgba};
use nalgebra as na;

fn main() {
    let img0 = ImageReader::open("data/cam0.png")
        .unwrap()
        .decode()
        .unwrap();
    let img1 = ImageReader::open("data/cam1.png")
        .unwrap()
        .decode()
        .unwrap();
    // let img0 = image::DynamicImage::ImageRgb8(img0.to_rgb8());
    // let img1 = image::DynamicImage::ImageRgb8(img1.to_rgb8());
    let model0 = model_from_json("data/eucm0.json");
    let model1 = model_from_json("data/eucm1.json");
    let tvec = na::Vector3::new(
        -0.10098947190325333,
        -0.0020811599784744455,
        -0.0012888359197775197,
    );
    let quat = na::Quaternion::new(
        0.9997158799903332,
        0.02382966001551074,
        -0.00032454324393309654,
        0.00044863167728445325,
    );
    let rvec = na::UnitQuaternion::from_quaternion(quat).scaled_axis();
    let (r0, r1, p) = stereo_rectify(&model0, &model1, &rvec, &tvec, None);
    let image_w_h = (
        model0.width().round() as u32,
        model0.height().round() as u32,
    );
    let (xmap0, ymap0) = model0.init_undistort_map(&p, image_w_h, Some(r0));
    let remaped0 = remap(&img0, &xmap0, &ymap0);
    let (xmap1, ymap1) = model1.init_undistort_map(&p, image_w_h, Some(r1));
    let remaped1 = remap(&img1, &xmap1, &ymap1);

    let remaped0 = remaped0.rotate90().to_rgb8();
    let remaped1 = remaped1.rotate90().to_rgb8();
    let width = remaped0.width();
    let height = remaped0.height() * 2;
    let mut c0 = remaped0.to_vec();
    let mut c1 = remaped1.to_vec();
    c0.append(&mut c1);
    let cur = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width, height, c0).unwrap();
    let mut cur = image::DynamicImage::ImageRgb8(cur).rotate270();

    for row in (10..width).step_by(20) {
        imageproc::drawing::draw_line_segment_mut(
            &mut cur,
            (0.0, row as f32),
            (height as f32, row as f32),
            Rgba::<u8>([rand::random(), rand::random(), rand::random(), 255]),
        );
    }
    cur.save("undistort_rectified.png").unwrap()
}
