use crate::generic_model::{CameraModel, ModelCast};
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug)]
pub struct Ftheta<T: na::RealField + Clone> {
    pub fx: T,
    pub fy: T,
    pub cx: T,
    pub cy: T,
    // pub k1: T, By definition it's always one.
    pub k2: T,
    pub k3: T,
    pub k4: T,
    pub k5: T,
    pub k6: T,
    pub width: u32,
    pub height: u32,
}
impl<T: na::RealField + Clone> ModelCast<T> for Ftheta<T> {}
impl<T: na::RealField + Clone> Ftheta<T> {
    pub fn new(params: &na::DVector<T>, width: u32, height: u32) -> Ftheta<T> {
        Ftheta {
            fx: params[0].clone(),
            fy: params[1].clone(),
            cx: params[2].clone(),
            cy: params[3].clone(),
            // k1: params[4].clone(),
            k2: params[4].clone(),
            k3: params[5].clone(),
            k4: params[6].clone(),
            k5: params[7].clone(),
            k6: params[8].clone(),
            width,
            height,
        }
    }
    pub fn zeros() -> Ftheta<T> {
        Ftheta {
            fx: T::zero(),
            fy: T::zero(),
            cx: T::zero(),
            cy: T::zero(),
            // k1: T::one(),
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
    // fn df_dtheta removed as it is not used by bisection

    pub fn from<U: na::RealField + Clone>(m: &Ftheta<U>) -> Ftheta<T> {
        Ftheta::new(&m.cast(), m.width, m.height)
    }
}

impl<T: na::RealField + Clone> CameraModel<T> for Ftheta<T> {
    fn set_params(&mut self, params: &nalgebra::DVector<T>) {
        if params.shape() != self.params().shape() {
            panic!("params has wrong shape.")
        }
        self.fx = params[0].clone();
        self.fy = params[1].clone();
        self.cx = params[2].clone();
        self.cy = params[3].clone();
        // self.k1 = params[4].clone();
        self.k2 = params[4].clone();
        self.k3 = params[5].clone();
        self.k4 = params[6].clone();
        self.k5 = params[7].clone();
        self.k6 = params[8].clone();
    }
    fn params(&self) -> nalgebra::DVector<T> {
        na::dvector![
            self.fx.clone(),
            self.fy.clone(),
            self.cx.clone(),
            self.cy.clone(),
            // self.k1.clone(),
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
        let x = pt[0].clone();
        let y = pt[1].clone();
        let z = pt[2].clone();
        let r_xy = (x.clone() * x.clone() + y.clone() * y.clone()).sqrt();
        let theta = r_xy.clone().atan2(z);

        let fx = &params[0];
        let fy = &params[1];
        let cx = &params[2];
        let cy = &params[3];
        let k2 = &params[4];
        let k3 = &params[5];
        let k4 = &params[6];
        let k5 = &params[7];
        let k6 = &params[8];

        let theta_d = Self::f_theta(k2, k3, k4, k5, k6, &theta);

        // x_img = theta_d * (x / r_xy)
        // If r_xy is small, x/r_xy is undefined, but x, y -> 0 so x_img -> 0.
        // However, if theta -> 0, theta_d -> 0. Limit is finite.
        // But if theta -> pi (r_xy -> 0, z < 0), then theta_d is large, but x/r_xy is undefined?
        // Actually if r_xy=0, x=0, y=0.
        let (mx, my) = if r_xy > T::from_f64(1e-6).unwrap() {
            let d = theta_d / r_xy;
            (x * d.clone(), y * d)
        } else {
            // r_xy -> 0.
            // If z > 0, theta -> 0, theta_d -> 0. mx, my -> 0.
            // If z < 0, theta -> pi. theta_d -> f(pi).
            // But x, y are 0. So projection should be 0?
            // In pinhole, (0,0, -1) projects to ??
            // Fish-eye (0,0,-1) usually depends on max FOV.
            // But mathematically, singularity at poles?
            // If strictly on axis, x=0, y=0. So result is cx, cy.
            (T::zero(), T::zero())
        };

        let px = fx.clone() * mx + cx.clone();
        let py = fy.clone() * my + cy.clone();
        na::Vector2::new(px, py)
    }

    fn unproject_one(&self, pt: &nalgebra::Vector2<T>) -> nalgebra::Vector3<T> {
        let xd = (pt[0].clone() - self.cx.clone()) / self.fx.clone();
        let yd = (pt[1].clone() - self.cy.clone()) / self.fy.clone();

        let rd2 = xd.clone() * xd.clone() + yd.clone() * yd.clone();
        let rd = rd2.sqrt();
        let theta_threshold = T::from_f64(1e-6).unwrap();
        let one = T::from_f64(1.0).unwrap();
        let zero = T::from_f64(0.0).unwrap();
        if rd > theta_threshold {
            let mut lower = T::zero();
            let mut upper = T::pi();

            for _ in 0..50 {
                let mid = (lower.clone() + upper.clone()) / T::from_f64(2.0).unwrap();
                let f_val = Self::f_theta(&self.k2, &self.k3, &self.k4, &self.k5, &self.k6, &mid);
                if f_val < rd {
                    lower = mid;
                } else {
                    upper = mid;
                }
            }
            let theta = (lower + upper) / T::from_f64(2.0).unwrap();

            let sin_theta = theta.clone().sin();
            let cos_theta = theta.cos();
            na::Vector3::new(
                xd * sin_theta.clone() / rd.clone(),
                yd * sin_theta / rd,
                cos_theta,
            )
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
            // self.k1.clone(),
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
            // (4, (1.0, 1.0)),
            (4, (-1.0, 1.0)),
            (5, (-1.0, 1.0)),
            (6, (-1.0, 1.0)),
            (7, (-1.0, 1.0)),
            (8, (-1.0, 1.0)),
        ]
    }
}
