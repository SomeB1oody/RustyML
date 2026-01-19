use super::*;

#[test]
fn test_gaussian_noise_new() {
    let noise_layer = GaussianNoise::new(0.1, vec![32, 128]).unwrap();
    assert_eq!(noise_layer.layer_type(), "GaussianNoise");
    assert_eq!(noise_layer.output_shape(), "[32, 128]");
}

#[test]
#[should_panic(expected = "Standard deviation cannot be negative")]
fn test_gaussian_noise_negative_stddev() {
    GaussianNoise::new(-0.1, vec![32, 128]).unwrap();
}

#[test]
fn test_gaussian_noise_forward_training() {
    let mut noise_layer = GaussianNoise::new(0.5, vec![4, 4]).unwrap();
    let input = Array2::ones((4, 4)).into_dyn();

    noise_layer.set_training(true);
    let output = noise_layer.forward(&input).unwrap();

    // Output shape should match input shape
    assert_eq!(output.shape(), input.shape());

    // With high stddev, output should be different from input
    // (extremely unlikely to be exactly the same with noise added)
    let diff = (&output - &input).mapv(|x| x.abs()).sum();
    assert!(diff > 0.0, "Expected noise to be added during training");
}

#[test]
fn test_gaussian_noise_forward_inference() {
    let mut noise_layer = GaussianNoise::new(0.5, vec![4, 4]).unwrap();
    let input = Array2::ones((4, 4)).into_dyn();

    noise_layer.set_training(false);
    let output = noise_layer.forward(&input).unwrap();

    // During inference, output should be identical to input
    assert_eq!(output, input);
}

#[test]
fn test_gaussian_noise_zero_stddev() {
    let mut noise_layer = GaussianNoise::new(0.0, vec![4, 4]).unwrap();
    let input = Array2::ones((4, 4)).into_dyn();

    noise_layer.set_training(true);
    let output = noise_layer.forward(&input).unwrap();

    // With stddev=0, no noise should be added
    assert_eq!(output, input);
}

#[test]
fn test_gaussian_noise_backward() {
    let mut noise_layer = GaussianNoise::new(0.1, vec![4, 4]).unwrap();
    let grad_output = Array2::from_elem((4, 4), 2.0).into_dyn();

    let grad_input = noise_layer.backward(&grad_output).unwrap();

    // Gradient should pass through unchanged
    assert_eq!(grad_input, grad_output);
}

#[test]
fn test_gaussian_noise_shape_validation() {
    let mut noise_layer = GaussianNoise::new(0.1, vec![4, 4]).unwrap();
    let wrong_input = Array2::ones((3, 3)).into_dyn();

    let result = noise_layer.forward(&wrong_input);
    assert!(result.is_err());
}
