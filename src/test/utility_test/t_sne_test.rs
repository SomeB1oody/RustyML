use super::*;

#[test]
fn test_tsne_default() {
    let tsne = TSNE::default();

    assert_eq!(tsne.get_n_components(), 2);
    assert_abs_diff_eq!(tsne.get_perplexity(), 30.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tsne.get_learning_rate(), 200.0, epsilon = 1e-12);
    assert_eq!(tsne.get_n_iter(), 1000);
    assert!(tsne.get_random_state().is_none());
}

#[test]
fn test_tsne_new_getters() {
    let tsne = TSNE::new(3, 10.0, 50.0, 250, Some(7)).unwrap();

    assert_eq!(tsne.get_n_components(), 3);
    assert_abs_diff_eq!(tsne.get_perplexity(), 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tsne.get_learning_rate(), 50.0, epsilon = 1e-12);
    assert_eq!(tsne.get_n_iter(), 250);
    assert_eq!(tsne.get_random_state(), Some(7));
}

#[test]
fn test_tsne_new_validation() {
    assert!(matches!(
        TSNE::new(0, 30.0, 200.0, 100, None),
        Err(ModelError::InputValidationError(_))
    ));
    assert!(matches!(
        TSNE::new(2, 0.0, 200.0, 100, None),
        Err(ModelError::InputValidationError(_))
    ));
    assert!(matches!(
        TSNE::new(2, 30.0, 0.0, 100, None),
        Err(ModelError::InputValidationError(_))
    ));
    assert!(matches!(
        TSNE::new(2, 30.0, 200.0, 0, None),
        Err(ModelError::InputValidationError(_))
    ));
}

#[test]
fn test_tsne_fit_transform_shape() {
    let tsne = TSNE::new(2, 3.0, 100.0, 50, Some(42)).unwrap();
    let data = arr2(&[
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0],
        [7.0, 8.0, 9.0],
        [8.0, 9.0, 10.0],
    ]);

    let embedded = tsne.fit_transform(&data.view()).unwrap();

    assert_eq!(embedded.shape(), &[8, 2]);
    assert!(embedded.iter().all(|v| v.is_finite()));
}

#[test]
fn test_tsne_fit_transform_shape_alternate_seed() {
    let tsne = TSNE::new(2, 3.0, 100.0, 50, Some(7)).unwrap();
    let data = arr2(&[
        [1.5, 2.5, 3.5],
        [2.5, 3.5, 4.5],
        [3.5, 4.5, 5.5],
        [4.5, 5.5, 6.5],
        [5.5, 6.5, 7.5],
        [6.5, 7.5, 8.5],
        [7.5, 8.5, 9.5],
        [8.5, 9.5, 10.5],
    ]);

    let embedded = tsne.fit_transform(&data.view()).unwrap();

    assert_eq!(embedded.shape(), &[8, 2]);
    assert!(embedded.iter().all(|v| v.is_finite()));
}

#[test]
fn test_tsne_fit_transform_invalid_input() {
    let tsne = TSNE::new(2, 2.0, 100.0, 10, None).unwrap();
    let single_sample = Array2::<f64>::zeros((1, 2));
    assert!(matches!(
        tsne.fit_transform(&single_sample.view()),
        Err(ModelError::InputValidationError(_))
    ));

    let mut data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    data[[0, 1]] = f64::NAN;
    assert!(matches!(
        tsne.fit_transform(&data.view()),
        Err(ModelError::InputValidationError(_))
    ));
}

#[test]
fn test_tsne_perplexity_validation() {
    let tsne = TSNE::new(2, 5.0, 100.0, 10, None).unwrap();
    let data = Array2::<f64>::zeros((5, 2));
    assert!(matches!(
        tsne.fit_transform(&data.view()),
        Err(ModelError::InputValidationError(_))
    ));
}

#[test]
fn test_tsne_fit_transform_is_deterministic_and_centered() {
    let tsne = TSNE::new(2, 3.0, 100.0, 350, Some(7)).unwrap();
    let data = arr2(&[
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        [5.0, 5.0, 5.0],
        [5.1, 5.0, 5.0],
        [5.0, 5.1, 5.0],
        [5.0, 5.0, 5.1],
    ]);

    let y_first = tsne.fit_transform(&data.view()).unwrap();
    let y_second = tsne.fit_transform(&data.view()).unwrap();

    for (first, second) in y_first.iter().zip(y_second.iter()) {
        assert_abs_diff_eq!(first, second, epsilon = 1e-10);
    }

    let mean = y_first
        .mean_axis(Axis(0))
        .expect("embedding has at least one row");
    for value in mean.iter() {
        assert!(value.abs() < 1e-8);
    }
}
