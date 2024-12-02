use crate::generic::{CameraModel, ModelCast};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug)]
pub struct Ftheta6<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    pub k1: T,
    pub k2: T,
    pub k3: T,
    pub k4: T,
    pub k5: T,
    pub k6: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> ModelCast<T> for Ftheta6<T> {}
impl<T: na::RealField + Clone> Ftheta6<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> Ftheta6<T> {
        Ftheta6 {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            k1: params[4].clone(),
            k2: params[5].clone(),
            k3: params[6].clone(),
            k4: params[7].clone(),
            k5: params[8].clone(),
            k6: params[9].clone(),
            width,
            height,
        }
    }
    pub fn zeros() -> Ftheta6<T> {
        Ftheta6 {
            fx: T::zero(),
            fy: T::zero(),
            cx: T::zero(),
            cy: T::zero(),
            k1: T::one(),
            k2: T::zero(),
            k3: T::zero(),
            k4: T::zero(),
            k5: T::zero(),
            k6: T::zero(),
            width: 0,
            height: 0,
        }
    }
    fn f_theta(k2: &T, k3: &T, k4: &T, k5: &T, k6: &T, theta: &T) -> T {
        let theta2 = theta.clone() * theta.clone();
        let theta3 = theta2.clone() * theta.clone();
        let theta4 = theta3.clone() * theta.clone();
        let theta5 = theta4.clone() * theta.clone();
        let theta6 = theta5.clone() * theta.clone();

        theta.clone()
            + k2.clone() * theta2
            + k3.clone() * theta3
            + k4.clone() * theta4
            + k5.clone() * theta5
            + k6.clone() * theta6
    }
    fn df_dtheta(k2: &T, k3: &T, k4: &T, k5: &T, k6: &T, theta: &T) -> T {
        let theta2 = theta.clone() * theta.clone();
        let theta3 = theta2.clone() * theta.clone();
        let theta4 = theta3.clone() * theta.clone();
        let theta5 = theta4.clone() * theta.clone();

        T::from_f64(1.0).unwrap()
            + T::from_f64(2.0).unwrap() * k2.clone() * theta.clone()
            + T::from_f64(3.0).unwrap() * k3.clone() * theta2.clone()
            + T::from_f64(4.0).unwrap() * k4.clone() * theta3.clone()
            + T::from_f64(5.0).unwrap() * k5.clone() * theta4.clone()
            + T::from_f64(6.0).unwrap() * k6.clone() * theta5.clone()
    }
    pub fn from<U: na::RealField + Clone>(m: &Ftheta6<U>) -> Ftheta6<T> {
        Ftheta6::new(&m.cast(), m.width, m.height)
    }
}

impl<T: na::RealField + Clone> CameraModel<T> for Ftheta6<T> {
    fn set_params(&mut self, params: &nalgebra::DVector<T>) {
        if params.shape() != self.params().shape() {
            panic!("params has wrong shape.")
        }
        self.fx = params[0].clone();
        self.fy = params[1].clone();
        self.cx = params[2].clone();
        self.cy = params[3].clone();
        self.k1 = params[4].clone();
        self.k2 = params[5].clone();
        self.k3 = params[6].clone();
        self.k4 = params[7].clone();
        self.k5 = params[8].clone();
        self.k6 = params[9].clone();
    }
    fn params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            self.k1.clone(),
            self.k2.clone(),
            self.k3.clone(),
            self.k4.clone(),
            self.k5.clone(),
            self.k6.clone(),
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
        let xn = pt[0].clone() / pt[2].clone();
        let yn = pt[1].clone() / pt[2].clone();
        let r2 = xn.clone() * xn.clone() + yn.clone() * yn.clone();
        let r = r2.sqrt();
        let theta = r.clone().atan();

        let fx = &params[0];
        let fy = &params[1];
        let cx = &params[2];
        let cy = &params[3];
        let k2 = &params[5];
        let k3 = &params[6];
        let k4 = &params[7];
        let k5 = &params[8];
        let k6 = &params[9];

        let theta_d = Self::f_theta(k2, k3, k4, k5, k6, &theta);
        let d = theta_d / r.clone();
        let px = fx.clone() * (xn * d.clone()) + cx.clone();
        let py = fy.clone() * (yn * d) + cy.clone();
        na::Vector2::new(px, py)
    }

    fn unproject_one(&self, pt: &nalgebra::Vector2<T>) -> nalgebra::Vector3<T> {
        let xd = (pt[0].clone() - self.cx.clone()) / self.fx.clone();
        let yd = (pt[1].clone() - self.cy.clone()) / self.fy.clone();

        let rd2 = xd.clone() * xd.clone() + yd.clone() * yd.clone();
        let rd = rd2.sqrt();
        let mut theta = rd.clone();
        let theta_threshold = T::from_f64(1e-6).unwrap();
        let one = T::from_f64(1.0).unwrap();
        let zero = T::from_f64(0.0).unwrap();
        if theta > theta_threshold {
            for _ in 0..5 {
                let theta_next = theta.clone()
                    - (Self::f_theta(
                        &self.k2,
                        &self.k3,
                        &self.k4,
                        &self.k5,
                        &self.k6,
                        &theta.clone(),
                    ) - rd.clone())
                        / Self::df_dtheta(&self.k2, &self.k3, &self.k4, &self.k5, &self.k6, &theta);
                if (theta_next.clone() - theta).abs() < theta_threshold {
                    theta = theta_next.clone();
                    break;
                }
                theta = theta_next;
            }
            let r = theta.clone().tan();
            let f_theta = Self::f_theta(&self.k2, &self.k3, &self.k4, &self.k5, &self.k6, &theta);
            na::Vector3::new(xd * r.clone() / f_theta.clone(), yd * r / f_theta, one)
        } else {
            na::Vector3::new(zero.clone(), zero, one)
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
        na::dvector![
            self.k1.clone(),
            self.k2.clone(),
            self.k3.clone(),
            self.k4.clone(),
            self.k5.clone(),
            self.k6.clone(),
        ]
    }
    fn set_w_h(&mut self, w: u32, h: u32) {
        self.width = w;
        self.height = h;
    }
    fn distortion_params_bound(&self) -> Vec<(usize, (f64, f64))> {
        // k1 is always one, other k [-1, 1]
        vec![
            (4, (1.0, 1.0)),
            (5, (-1.0, 1.0)),
            (6, (-1.0, 1.0)),
            (7, (-1.0, 1.0)),
            (8, (-1.0, 1.0)),
            (9, (-1.0, 1.0)),
        ]
    }
}
