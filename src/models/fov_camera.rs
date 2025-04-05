use crate::generic_model::{CameraModel, ModelCast};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct FovCamera<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub w: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> FovCamera<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> FovCamera<T> {
        if params.shape() != (5, 1) {
            panic!("the length of the vector should be 5");
        }
        FovCamera {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            w: params[4].clone(),
            width,
            height,
        }
    }
    pub fn from<U: na::RealField + Clone>(m: &FovCamera<U>) -> FovCamera<T> {
        FovCamera::new(&m.cast(), m.width, m.height)
    }
    pub fn zeros() -> FovCamera<T> {
        FovCamera {
            fx: T::zero(),
            fy: T::zero(),
            cx: T::zero(),
            cy: T::zero(),
            w: T::from_f64(0.5).unwrap(),
            width: 0,
            height: 0,
        }
    }
}

impl<T: na::RealField + Clone> ModelCast<T> for FovCamera<T> {}

impl<T: na::RealField + Clone> CameraModel<T> for FovCamera<T> {
    fn set_params(&mut self, params: &nalgebra::DVector<T>) {
        if params.shape() != self.params().shape() {
            panic!("params has wrong shape.")
        }
        self.fx = params[0].clone();
        self.fy = params[1].clone();
        self.cx = params[2].clone();
        self.cy = params[3].clone();
        self.w = params[4].clone();
    }
    #[inline]
    fn params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            self.w.clone(),
        ]
    }
    fn width(&self) -> T {
        T::from_u32(self.width).unwrap()
    }

    fn height(&self) -> T {
        T::from_u32(self.height).unwrap()
    }

    fn project_one(&self, pt: &nalgebra::Vector3<T>) -> nalgebra::Vector2<T> {
        let params = self.params();
        let fx = &params[0];
        let fy = &params[1];
        let cx = &params[2];
        let cy = &params[3];
        let w = &params[4];

        let x = pt[0].clone();
        let y = pt[1].clone();
        let z = pt[2].clone();

        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let r = r2.clone().sqrt();

        let two = T::from_f64(2.0).unwrap();

        let tanwhalf = w.clone() / two.clone();
        let atan_wrd = (two.clone() * tanwhalf.clone() * r.clone()).atan2(z);

        let eps_sqrt = T::from_f64(f64::EPSILON).unwrap().sqrt();

        let rd = if r2 < eps_sqrt {
            two.clone() * tanwhalf / w.clone()
        } else {
            atan_wrd / (r * w.clone())
        };

        let mx = x * rd.clone();
        let my = y * rd.clone();

        na::Vector2::new(fx.clone() * mx + cx.clone(), fy.clone() * my + cy.clone())
    }

    fn unproject_one(&self, pt: &nalgebra::Vector2<T>) -> nalgebra::Vector3<T> {
        let params = self.params();
        let fx = &params[0];
        let fy = &params[1];
        let cx = &params[2];
        let cy = &params[3];
        let w = &params[4];

        let one = T::from_f64(1.0).unwrap();
        let two = T::from_f64(2.0).unwrap();

        let tan_w_2 = (w.clone() / two.clone()).tan();
        let mul2tanwby2 = tan_w_2 * two;

        let mx = (pt[0].clone() - cx.clone()) / fx.clone();
        let my = (pt[1].clone() - cy.clone()) / fy.clone();

        let r2 = mx.clone() * mx.clone() + my.clone() * my.clone();
        let rd = r2.sqrt();

        let eps_sqrt = T::from_f64(f64::EPSILON).unwrap().sqrt();

        if mul2tanwby2 > eps_sqrt.clone() && rd > eps_sqrt {
            let sin_rd_w = (rd.clone() * w.clone()).sin();
            let cos_rd_w = (rd.clone() * w.clone()).cos();
            let ru = sin_rd_w / (rd * mul2tanwby2);

            na::Vector3::new(mx * ru.clone() / cos_rd_w.clone(), my * ru / cos_rd_w, one)
        } else {
            na::Vector3::new(mx, my, one)
        }
    }

    fn camera_params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone()
        ]
    }

    fn distortion_params(&self) -> nalgebra::DVector<T> {
        na::dvector![self.w.clone()]
    }

    fn set_w_h(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
    }

    fn distortion_params_bound(&self) -> Vec<(usize, (f64, f64))> {
        // w [eps, 3]
        vec![(4, (f64::EPSILON, 3.0))]
    }
}
