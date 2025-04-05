use std::str::FromStr;

use super::generic_functions::*;
use super::{FovCamera, Ftheta, KannalaBrandt4, OpenCVModel5, EUCM, EUCMT, UCM};
use image::DynamicImage;
use nalgebra as na;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum GenericModel<T: na::RealField> {
    EUCM(EUCM<T>),
    UCM(UCM<T>),
    OpenCVModel5(OpenCVModel5<T>),
    KannalaBrandt4(KannalaBrandt4<T>),
    EUCMT(EUCMT<T>),
    Ftheta(Ftheta<T>),
    FovCamera(FovCamera<T>),
}
macro_rules! generic_impl {
    ($fn_name:tt, $out:ty, $($v:tt: $t:ty),+) => {
        pub fn $fn_name(&self, $($v: $t),+) -> $out{
            match self {
                GenericModel::EUCM(eucm) => $fn_name(eucm, $($v),+),
                GenericModel::UCM(ucm) => $fn_name(ucm, $($v),+),
                GenericModel::OpenCVModel5(open_cvmodel5) => $fn_name(open_cvmodel5, $($v),+),
                GenericModel::KannalaBrandt4(kannala_brandt4) => $fn_name(kannala_brandt4, $($v),+),
                GenericModel::EUCMT(eucmt) => $fn_name(eucmt, $($v),+),
                GenericModel::Ftheta(ftheta) => $fn_name(ftheta, $($v),+),
                GenericModel::FovCamera(fov_camera) => $fn_name(fov_camera, $($v),+),
            }
        }
    };
}
macro_rules! generic_impl_self {
    ($fn_name:tt -> $out:ty) => {
        pub fn $fn_name(&self) -> $out{
            match self {
                GenericModel::EUCM(eucm) => eucm.$fn_name(),
                GenericModel::UCM(ucm) => ucm.$fn_name(),
                GenericModel::OpenCVModel5(open_cvmodel5) => open_cvmodel5.$fn_name(),
                GenericModel::KannalaBrandt4(kannala_brandt4) => kannala_brandt4.$fn_name(),
                GenericModel::EUCMT(eucmt) => eucmt.$fn_name(),
                GenericModel::Ftheta(ftheta) => ftheta.$fn_name(),
                GenericModel::FovCamera(fov_camera) => fov_camera.$fn_name(),
            }
        }
    };
    ($fn_name:tt, $out:ty, $($v:tt: $t:ty),+) => {
        pub fn $fn_name(&self, $($v: $t),+) -> $out{
            match self {
                GenericModel::EUCM(eucm) => eucm.$fn_name($($v),+),
                GenericModel::UCM(ucm) => ucm.$fn_name($($v),+),
                GenericModel::OpenCVModel5(open_cvmodel5) => open_cvmodel5.$fn_name($($v),+),
                GenericModel::KannalaBrandt4(kannala_brandt4) => kannala_brandt4.$fn_name($($v),+),
                GenericModel::EUCMT(eucmt) => eucmt.$fn_name($($v),+),
                GenericModel::Ftheta(ftheta) => ftheta.$fn_name($($v),+),
                GenericModel::FovCamera(fov_camera) => fov_camera.$fn_name($($v),+),
            }
        }
    };
    ($fn_name:tt, $($v:tt: $t:ty),+) => {
        pub fn $fn_name(&mut self, $($v: $t),+){
            match self {
                GenericModel::EUCM(eucm) => eucm.$fn_name($($v),+),
                GenericModel::UCM(ucm) => ucm.$fn_name($($v),+),
                GenericModel::OpenCVModel5(open_cvmodel5) => open_cvmodel5.$fn_name($($v),+),
                GenericModel::KannalaBrandt4(kannala_brandt4) => kannala_brandt4.$fn_name($($v),+),
                GenericModel::EUCMT(eucmt) => eucmt.$fn_name($($v),+),
                GenericModel::Ftheta(ftheta) => ftheta.$fn_name($($v),+),
                GenericModel::FovCamera(fov_camera) => fov_camera.$fn_name($($v),+),
            }
        }
    };
}
impl GenericModel<f64> {
    generic_impl!(init_undistort_map, (na::DMatrix<f32>, na::DMatrix<f32>), projection_mat: &na::Matrix3<f64>, new_w_h: (u32, u32), rotation: Option<na::Rotation3<f64>>);
    generic_impl!(estimate_new_camera_matrix_for_undistort, na::Matrix3<f64>, balance: f64, new_image_w_h: Option<(u32, u32)>);
}

impl FromStr for GenericModel<f64> {
    type Err = std::fmt::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ucm" | "UCM" => Ok(GenericModel::UCM(UCM::zeros())),
            "eucm" | "EUCM" => Ok(GenericModel::EUCM(EUCM::zeros())),
            "kb4" | "KB4" => Ok(GenericModel::KannalaBrandt4(KannalaBrandt4::zeros())),
            "opencv5" | "OPENCV5" => Ok(GenericModel::OpenCVModel5(OpenCVModel5::zeros())),
            "eucmt" | "EUCMT" => Ok(GenericModel::EUCMT(EUCMT::zeros())),
            "ftheta" | "FTHETA" => Ok(GenericModel::Ftheta(Ftheta::zeros())),
            "fov_camera" | "FOV_CAMERA" => Ok(GenericModel::FovCamera(FovCamera::zeros())),
            _ => Err(std::fmt::Error),
        }
    }
}

impl<T: na::RealField + Clone> GenericModel<T> {
    generic_impl_self!(width -> T);
    generic_impl_self!(height -> T);
    generic_impl_self!(params -> na::DVector<T>);
    generic_impl_self!(set_params, params: &na::DVector<T>);
    generic_impl_self!(set_w_h, w: u32, h: u32);
    generic_impl_self!(camera_params -> na::DVector<T>);
    generic_impl_self!(distortion_params -> na::DVector<T>);
    generic_impl_self!(distortion_params_bound -> Vec<(usize, (f64, f64))>);
    generic_impl_self!(project_one, na::Vector2<T>, pt: &na::Vector3<T>);
    generic_impl_self!(unproject_one, na::Vector3<T>, pt: &na::Vector2<T>);
    generic_impl_self!(project, Vec<Option<na::Vector2<T>>>, p3d: &[na::Vector3<T>]);
    generic_impl_self!(unproject, Vec<Option<na::Vector3<T>>>, p2d: &[na::Vector2<T>]);
    pub fn cast<U: na::RealField + Clone>(&self) -> GenericModel<U> {
        match self {
            GenericModel::EUCM(eucm) => GenericModel::EUCM(EUCM::from(eucm)),
            GenericModel::UCM(ucm) => GenericModel::UCM(UCM::from(ucm)),
            GenericModel::OpenCVModel5(open_cvmodel5) => {
                GenericModel::OpenCVModel5(OpenCVModel5::from(open_cvmodel5))
            }
            GenericModel::KannalaBrandt4(kannala_brandt4) => {
                GenericModel::KannalaBrandt4(KannalaBrandt4::from(kannala_brandt4))
            }
            GenericModel::EUCMT(eucmt) => GenericModel::EUCMT(EUCMT::from(eucmt)),
            GenericModel::Ftheta(ftheta) => GenericModel::Ftheta(Ftheta::from(ftheta)),
            GenericModel::FovCamera(fov_camera) => {
                GenericModel::FovCamera(FovCamera::from(fov_camera))
            }
        }
    }
    pub fn new_from_params(&self, params: &na::DVector<T>) -> GenericModel<T> {
        match self {
            GenericModel::EUCM(m) => GenericModel::EUCM(EUCM::new(params, m.width, m.height)),
            GenericModel::UCM(m) => GenericModel::UCM(UCM::new(params, m.width, m.height)),
            GenericModel::OpenCVModel5(m) => {
                GenericModel::OpenCVModel5(OpenCVModel5::new(params, m.width, m.height))
            }
            GenericModel::KannalaBrandt4(m) => {
                GenericModel::KannalaBrandt4(KannalaBrandt4::new(params, m.width, m.height))
            }
            GenericModel::EUCMT(m) => GenericModel::EUCMT(EUCMT::new(params, m.width, m.height)),
            GenericModel::Ftheta(m) => GenericModel::Ftheta(Ftheta::new(params, m.width, m.height)),
            GenericModel::FovCamera(m) => {
                GenericModel::FovCamera(FovCamera::new(params, m.width, m.height))
            }
        }
    }
}

macro_rules! remap_impl {
    ($src:expr, $map0:expr, $map1:expr, $($img_type:ident => ($inner_type:ident, $default_value:expr)),*) => {
        match $src {
            $(
                DynamicImage::$img_type(img) => {
                    let (r, c) = $map0.shape();
                    let out_img = image::ImageBuffer::from_par_fn(c as u32, r as u32, |x, y| {
                        let idx = y as usize * c + x as usize;
                        let (x_cor, y_cor) = unsafe { ($map0.get_unchecked(idx), $map1.get_unchecked(idx)) };
                        if x_cor.is_nan() || y_cor.is_nan() {
                            return image::$inner_type($default_value);
                        }
                        image::imageops::interpolate_bilinear(img, *x_cor, *y_cor)
                            .unwrap_or(image::$inner_type($default_value))
                    });
                    DynamicImage::$img_type(out_img)
                }
            )*
            _ => {
                panic!("Not support this image type.");
            }
        }
    };
}

pub fn remap(src: &DynamicImage, map0: &na::DMatrix<f32>, map1: &na::DMatrix<f32>) -> DynamicImage {
    remap_impl!(src, map0, map1,
        ImageLuma8 => (Luma, [0]),
        ImageLumaA8 => (LumaA, [0, 0]),
        ImageLuma16 => (Luma, [0]),
        ImageLumaA16 => (LumaA, [0, 0]),
        ImageRgb8 => (Rgb, [0, 0, 0]),
        ImageRgba8 => (Rgba, [0, 0, 0, 0]),
        ImageRgb16 => (Rgb, [0, 0, 0]),
        ImageRgba16 => (Rgba, [0, 0, 0, 0]),
        ImageRgb32F => (Rgb, [0.0, 0.0, 0.0]),
        ImageRgba32F => (Rgba, [0.0, 0.0, 0.0, 0.0])
    )
}

pub trait ModelCast<T: na::RealField + Clone>: CameraModel<T> {
    fn cast<U: na::RealField>(&self) -> na::DVector<U> {
        let v: Vec<_> = self
            .params()
            .iter()
            .map(|i| U::from_f64(i.to_subset().unwrap()).unwrap())
            .collect();
        na::DVector::from_vec(v)
    }
}

pub trait CameraModel<T: na::RealField + Clone>
where
    Self: Sync,
{
    fn set_params(&mut self, params: &na::DVector<T>);
    fn params(&self) -> na::DVector<T>;
    fn camera_params(&self) -> na::DVector<T>;
    fn distortion_params(&self) -> na::DVector<T>;
    fn width(&self) -> T;
    fn height(&self) -> T;
    fn set_w_h(&mut self, w: u32, h: u32);
    fn project_one(&self, pt: &na::Vector3<T>) -> na::Vector2<T>;
    fn distortion_params_bound(&self) -> Vec<(usize, (f64, f64))>;
    fn project(&self, p3d: &[na::Vector3<T>]) -> Vec<Option<na::Vector2<T>>> {
        p3d.par_iter()
            .map(|pt| {
                let p2d = self.project_one(pt);
                if p2d[0] < T::from_f64(0.0).unwrap()
                    || p2d[0] > self.width()
                    || p2d[1] < T::from_f64(0.0).unwrap()
                    || p2d[1] > self.height()
                {
                    None
                } else {
                    Some(p2d)
                }
            })
            .collect()
    }
    fn unproject_one(&self, pt: &na::Vector2<T>) -> na::Vector3<T>;
    fn unproject(&self, p2d: &[na::Vector2<T>]) -> Vec<Option<na::Vector3<T>>> {
        p2d.par_iter()
            .map(|pt| {
                if pt[0] < T::from_f64(0.0).unwrap()
                    || pt[0] > self.width() - T::from_f64(1.0).unwrap()
                    || pt[1] < T::from_f64(0.0).unwrap()
                    || pt[1] > self.height() - T::from_f64(1.0).unwrap()
                {
                    None
                } else {
                    let p3d = self.unproject_one(pt);
                    Some(p3d)
                }
            })
            .collect()
    }
}
