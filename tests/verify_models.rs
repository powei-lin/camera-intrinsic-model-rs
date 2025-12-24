use camera_intrinsic_model::*;
use nalgebra as na;
use rand::Rng;

fn test_model_roundtrip(model: &GenericModel<f64>, samples: usize) {
    let width = model.width();
    let height = model.height();
    let mut rng = rand::rng();

    for _ in 0..samples {
        let u = rng.random_range(0.0..width);
        let v = rng.random_range(0.0..height);

        let cams = model.camera_params();
        let cx = cams[2];
        let cy = cams[3];
        let r2 = (u - cx).powi(2) + (v - cy).powi(2);
        let max_r = width.min(height) / 2.0;
        if r2 > max_r.powi(2) {
            continue;
        }

        let p2d = na::Vector2::new(u, v);

        let p3d = model.unproject_one(&p2d);
        let p2d_reprojected = model.project_one(&p3d);

        let dist = (p2d - p2d_reprojected).norm();

        assert!(
            dist < 1e-3,
            "Roundtrip failed for point {:?}. \n3D: {:?} \nReprojected: {:?}, Dist: {}",
            p2d,
            p3d,
            p2d_reprojected,
            dist
        );
    }
}

#[test]
fn test_eucm_roundtrip() {
    let model = model_from_json("data/eucm0.json");
    // Ensure it is EUCM
    match model {
        GenericModel::EUCM(_) => (),
        _ => panic!("Expected EUCM model"),
    }
    test_model_roundtrip(&model, 1000);
}

#[test]
fn test_fov_roundtrip() {
    let model = model_from_json("data/fov_tum_mono.json");
    // Ensure it is FovCamera
    match model {
        GenericModel::FovCamera(_) => (),
        _ => panic!("Expected FovCamera model"),
    }
    test_model_roundtrip(&model, 1000);
}

#[test]
fn test_ftheta_roundtrip() {
    let model = model_from_json("data/ftheta.json");
    match model {
        GenericModel::Ftheta(_) => (),
        _ => panic!("Expected Ftheta model"),
    }
    test_model_roundtrip(&model, 1000);
}

#[test]
fn test_eucmt_roundtrip() {
    let model = model_from_json("data/eucmt.json");
    match model {
        GenericModel::EUCMT(_) => (),
        _ => panic!("Expected EUCMT model"),
    }
    test_model_roundtrip(&model, 1000);
}

#[test]
fn test_kb4_roundtrip() {
    let model = model_from_json("data/kb4.json");
    match model {
        GenericModel::KannalaBrandt4(_) => (),
        _ => panic!("Expected KannalaBrandt4 model"),
    }
    test_model_roundtrip(&model, 1000);
}

#[test]
fn test_ucm_roundtrip() {
    let model = model_from_json("data/ucm.json");
    match model {
        GenericModel::UCM(_) => (),
        _ => panic!("Expected UCM model"),
    }
    test_model_roundtrip(&model, 1000);
}
