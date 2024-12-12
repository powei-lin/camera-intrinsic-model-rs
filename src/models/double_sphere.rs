use crate::generic_model::{CameraModel, ModelCast};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct DoubleSphere<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub xi: T,
    pub alpha: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> DoubleSphere<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> DoubleSphere<T> {
        if params.shape() != (6, 1) {
            panic!("the length of the vector should be 6");
        }
        DoubleSphere {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            xi: params[4].clone(),
            alpha: params[5].clone(),
            width,
            height,
        }
    }
    pub fn from<U: na::RealField + Clone>(m: &DoubleSphere<U>) -> DoubleSphere<T> {
        DoubleSphere::new(&m.cast(), m.width, m.height)
    }
    pub fn zeros() -> DoubleSphere<T> {
        DoubleSphere {
            fx: T::zero(),
            fy: T::zero(),
            cx: T::zero(),
            cy: T::zero(),
            xi: T::from_f64(0.001).unwrap(),
            alpha: T::from_f64(0.5).unwrap(),
            width: 0,
            height: 0,
        }
    }
}

impl<T: na::RealField + Clone> ModelCast<T> for DoubleSphere<T> {}

impl<T: na::RealField + Clone> CameraModel<T> for DoubleSphere<T> {
    fn set_params(&mut self, params: &nalgebra::DVector<T>) {
        if params.shape() != self.params().shape() {
            panic!("params has wrong shape.")
        }
        self.fx = params[0].clone();
        self.fy = params[1].clone();
        self.cx = params[2].clone();
        self.cy = params[3].clone();
        self.xi = params[4].clone();
        self.alpha = params[5].clone();
    }
    #[inline]
    fn params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            self.xi.clone(),
            self.alpha.clone(),
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
        let xi = &params[4];
        let alpha = &params[5];

        let x = pt[0].clone();
        let y = pt[1].clone();
        let z = pt[2].clone();

        let xx = x.clone() * x.clone();
        let yy = y.clone() * y.clone();
        let zz = z.clone() * z.clone();
    
        let r2 = xx + yy;
    
        let d1_2 = r2.clone() + zz;
        let d1 = d1_2.sqrt();
    
        let k = xi.clone() * d1 + z;
        let kk = k.clone() * k.clone();
    
        let d2_2 = r2 + kk;
        let d2 = d2_2.sqrt();
    
        let norm = alpha.clone() * d2 + (T::one() - alpha.clone()) * k;
    
        let mx = x.clone() / norm.clone();
        let my = y.clone() / norm;

        na::Vector2::new(fx.clone() * mx + cx.clone(), fy.clone() * my + cy.clone())
    }

    fn unproject_one(&self, pt: &nalgebra::Vector2<T>) -> nalgebra::Vector3<T> {
        let params = self.params();
        let fx = &params[0];
        let fy = &params[1];
        let cx = &params[2];
        let cy = &params[3];
        let xi = &params[4];
        let alpha = &params[5];

        let mx = (pt[0].clone() - cx.clone()) / fx.clone();
        let my = (pt[1].clone() - cy.clone()) / fy.clone();

        let r2 = mx.clone() * mx.clone() + my.clone() * my.clone();

        let xi2_2 = alpha.clone() * alpha.clone();
        let xi1_2 = xi.clone() * xi.clone();
        let sqrt2 = (T::one() - (T::from_f64(2.0).unwrap() * alpha.clone() - T::one()) * r2.clone()).sqrt();
        let norm2 = alpha.clone() * sqrt2 + T::one() - alpha.clone();
        let mz = (T::one() - xi2_2 * r2.clone()) / norm2;
        let mz2 = mz.clone() * mz.clone();
        let norm1 = mz2.clone() + r2.clone();
        let sqrt1 = (mz2 + (T::one() - xi1_2) * r2).sqrt();
        let k = (mz.clone() * xi.clone() + sqrt1) / norm1;
    
        let z = k.clone() * mz - xi.clone();

        na::Vector3::new(mx * k.clone() / z.clone(), my * k / z, T::one())
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
        na::dvector![self.xi.clone(), self.alpha.clone()]
    }

    fn set_w_h(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
    }

    fn distortion_params_bound(&self) -> Vec<(usize, (f64, f64))> {
        // alpha [0, 1], beta > 0
        vec![(4, (-1.0, 1.0)), (5, (0.0, 1.0))]
    }
}
