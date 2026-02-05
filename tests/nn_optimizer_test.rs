use ndarray::array;
use rustyml::neural_network::optimizer::ada_grad::AdaGradStates;
use rustyml::neural_network::optimizer::adam::{Adam, AdamStates};
use rustyml::neural_network::optimizer::rms_prop::{RMSprop, RMSpropCache};
use rustyml::neural_network::optimizer::{AdaGrad, SGD};

#[test]
fn test_adam_new() {
    // Valid parameters
    let adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
    assert!(adam.is_ok());

    // Invalid learning rate (negative)
    let adam = Adam::new(-0.001, 0.9, 0.999, 1e-8);
    assert!(adam.is_err());

    // Invalid learning rate (zero)
    let adam = Adam::new(0.0, 0.9, 0.999, 1e-8);
    assert!(adam.is_err());

    // Invalid beta1 (negative)
    let adam = Adam::new(0.001, -0.1, 0.999, 1e-8);
    assert!(adam.is_err());

    // Invalid beta1 (greater than 1)
    let adam = Adam::new(0.001, 1.1, 0.999, 1e-8);
    assert!(adam.is_err());

    // Invalid beta2 (negative)
    let adam = Adam::new(0.001, 0.9, -0.999, 1e-8);
    assert!(adam.is_err());

    // Invalid beta2 (greater than 1)
    let adam = Adam::new(0.001, 0.9, 1.1, 1e-8);
    assert!(adam.is_err());

    // Invalid epsilon (negative)
    let adam = Adam::new(0.001, 0.9, 0.999, -1e-8);
    assert!(adam.is_err());

    // Invalid epsilon (zero)
    let adam = Adam::new(0.001, 0.9, 0.999, 0.0);
    assert!(adam.is_err());

    // Invalid learning rate (infinite)
    let adam = Adam::new(f32::INFINITY, 0.9, 0.999, 1e-8);
    assert!(adam.is_err());

    // Invalid epsilon (infinite)
    let adam = Adam::new(0.001, 0.9, 0.999, f32::INFINITY);
    assert!(adam.is_err());
}

#[test]
fn test_adam_states_new() {
    let states = AdamStates::new((2, 3), Some((3, 3)), (1, 3));

    // Check main parameter states
    assert_eq!(states.m.shape(), &[2, 3]);
    assert_eq!(states.v.shape(), &[2, 3]);

    // Check recurrent parameter states
    assert!(states.m_recurrent.is_some());
    assert!(states.v_recurrent.is_some());
    assert_eq!(states.m_recurrent.as_ref().unwrap().shape(), &[3, 3]);
    assert_eq!(states.v_recurrent.as_ref().unwrap().shape(), &[3, 3]);

    // Check bias states
    assert_eq!(states.m_bias.shape(), &[1, 3]);
    assert_eq!(states.v_bias.shape(), &[1, 3]);

    // Check that all values are initialized to zero
    assert!(states.m.iter().all(|&x| x == 0.0));
    assert!(states.v.iter().all(|&x| x == 0.0));
    assert!(
        states
            .m_recurrent
            .as_ref()
            .unwrap()
            .iter()
            .all(|&x| x == 0.0)
    );
    assert!(
        states
            .v_recurrent
            .as_ref()
            .unwrap()
            .iter()
            .all(|&x| x == 0.0)
    );
    assert!(states.m_bias.iter().all(|&x| x == 0.0));
    assert!(states.v_bias.iter().all(|&x| x == 0.0));
}

#[test]
fn test_adam_states_update_parameter() {
    let mut states = AdamStates::new((1, 2), None, (1, 2));
    let grad_param = array![[0.1, 0.1]];
    let grad_bias = array![[0.01, 0.01]];
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    let lr = 0.001;
    let t = 1;

    let (param_update, recurrent_update, bias_update) =
        states.update_parameter(&grad_param, None, &grad_bias, beta1, beta2, epsilon, t, lr);

    // Check that first moment estimates have been updated
    // m = beta1 * m + (1 - beta1) * grad
    assert!((states.m[[0, 0]] - 0.01).abs() < 1e-6); // 0.9 * 0 + 0.1 * 0.1 = 0.01
    assert!((states.m[[0, 1]] - 0.01).abs() < 1e-6);
    assert!((states.m_bias[[0, 0]] - 0.001).abs() < 1e-6); // 0.9 * 0 + 0.1 * 0.01 = 0.001
    assert!((states.m_bias[[0, 1]] - 0.001).abs() < 1e-6);

    // Check that second moment estimates have been updated
    // v = beta2 * v + (1 - beta2) * grad^2
    assert!((states.v[[0, 0]] - 0.00001).abs() < 1e-6); // 0.999 * 0 + 0.001 * 0.01 = 0.00001
    assert!((states.v[[0, 1]] - 0.00001).abs() < 1e-6);
    assert!((states.v_bias[[0, 0]] - 0.0000001).abs() < 1e-6); // 0.999 * 0 + 0.001 * 0.0001 = 0.0000001
    assert!((states.v_bias[[0, 1]] - 0.0000001).abs() < 1e-6);

    // Check that updates are returned
    assert_eq!(param_update.shape(), &[1, 2]);
    assert!(recurrent_update.is_none());
    assert_eq!(bias_update.shape(), &[1, 2]);

    // Verify update values are reasonable (bias-corrected)
    // m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
    // update = lr * m_hat / (sqrt(v_hat) + epsilon)
    assert!(param_update[[0, 0]].abs() > 0.0);
    assert!(bias_update[[0, 0]].abs() > 0.0);
}

#[test]
fn test_adam_states_update_parameter_with_recurrent() {
    let mut states = AdamStates::new((2, 2), Some((2, 2)), (1, 2));
    let grad_param = array![[0.1, 0.2], [0.3, 0.4]];
    let grad_recurrent = array![[0.05, 0.06], [0.07, 0.08]];
    let grad_bias = array![[0.01, 0.02]];
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    let lr = 0.001;
    let t = 1;

    let (param_update, recurrent_update, bias_update) = states.update_parameter(
        &grad_param,
        Some(&grad_recurrent),
        &grad_bias,
        beta1,
        beta2,
        epsilon,
        t,
        lr,
    );

    // Check that main parameter states have been updated
    assert!((states.m[[0, 0]] - 0.01).abs() < 1e-6); // (1 - 0.9) * 0.1 = 0.01
    assert!((states.m[[0, 1]] - 0.02).abs() < 1e-6); // (1 - 0.9) * 0.2 = 0.02
    assert!((states.m[[1, 0]] - 0.03).abs() < 1e-6); // (1 - 0.9) * 0.3 = 0.03
    assert!((states.m[[1, 1]] - 0.04).abs() < 1e-6); // (1 - 0.9) * 0.4 = 0.04

    // Check that recurrent parameter states have been updated
    assert!(states.m_recurrent.is_some());
    assert!(states.v_recurrent.is_some());
    let m_rec = states.m_recurrent.as_ref().unwrap();
    assert!((m_rec[[0, 0]] - 0.005).abs() < 1e-6); // (1 - 0.9) * 0.05 = 0.005
    assert!((m_rec[[0, 1]] - 0.006).abs() < 1e-6); // (1 - 0.9) * 0.06 = 0.006
    assert!((m_rec[[1, 0]] - 0.007).abs() < 1e-6); // (1 - 0.9) * 0.07 = 0.007
    assert!((m_rec[[1, 1]] - 0.008).abs() < 1e-6); // (1 - 0.9) * 0.08 = 0.008

    // Check that updates are returned
    assert_eq!(param_update.shape(), &[2, 2]);
    assert!(recurrent_update.is_some());
    assert_eq!(recurrent_update.as_ref().unwrap().shape(), &[2, 2]);
    assert_eq!(bias_update.shape(), &[1, 2]);
}

#[test]
fn test_adam_states_bias_correction() {
    let mut states = AdamStates::new((1, 2), None, (1, 2));
    let grad_param = array![[0.1, 0.1]];
    let grad_bias = array![[0.01, 0.01]];
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    let lr = 0.001;

    // First step
    let (update1, _, _) =
        states.update_parameter(&grad_param, None, &grad_bias, beta1, beta2, epsilon, 1, lr);

    // Second step with same gradients
    let (update2, _, _) =
        states.update_parameter(&grad_param, None, &grad_bias, beta1, beta2, epsilon, 2, lr);

    // Updates should be different due to bias correction changing with timestep
    // The difference is small but should be non-zero (approximately 3.4e-9)
    assert!((update1[[0, 0]] - update2[[0, 0]]).abs() > 1e-10);
}

#[test]
fn test_rmsprop_new() {
    // Valid parameters
    let rmsprop = RMSprop::new(0.001, 0.9, 1e-8);
    assert!(rmsprop.is_ok());

    // Invalid learning rate (negative)
    let rmsprop = RMSprop::new(-0.001, 0.9, 1e-8);
    assert!(rmsprop.is_err());

    // Invalid learning rate (zero)
    let rmsprop = RMSprop::new(0.0, 0.9, 1e-8);
    assert!(rmsprop.is_err());

    // Invalid rho (negative)
    let rmsprop = RMSprop::new(0.001, -0.1, 1e-8);
    assert!(rmsprop.is_err());

    // Invalid rho (greater than 1)
    let rmsprop = RMSprop::new(0.001, 1.1, 1e-8);
    assert!(rmsprop.is_err());

    // Invalid epsilon (negative)
    let rmsprop = RMSprop::new(0.001, 0.9, -1e-8);
    assert!(rmsprop.is_err());

    // Invalid epsilon (zero)
    let rmsprop = RMSprop::new(0.001, 0.9, 0.0);
    assert!(rmsprop.is_err());

    // Invalid learning rate (infinite)
    let rmsprop = RMSprop::new(f32::INFINITY, 0.9, 1e-8);
    assert!(rmsprop.is_err());

    // Invalid epsilon (infinite)
    let rmsprop = RMSprop::new(0.001, 0.9, f32::INFINITY);
    assert!(rmsprop.is_err());
}

#[test]
fn test_rmsprop_cache_new() {
    let cache = RMSpropCache::new((2, 3), Some((3, 3)), (1, 3));

    assert_eq!(cache.cache.shape(), &[2, 3]);
    assert!(cache.cache_recurrent.is_some());
    assert_eq!(cache.cache_recurrent.as_ref().unwrap().shape(), &[3, 3]);
    assert_eq!(cache.bias.shape(), &[1, 3]);

    // Check that all values are initialized to zero
    assert!(cache.cache.iter().all(|&x| x == 0.0));
    assert!(
        cache
            .cache_recurrent
            .as_ref()
            .unwrap()
            .iter()
            .all(|&x| x == 0.0)
    );
    assert!(cache.bias.iter().all(|&x| x == 0.0));
}

#[test]
fn test_rmsprop_cache_update_param() {
    let mut cache = array![[0.0, 0.0]];
    let mut param = array![[1.0, 2.0]];
    let grad = array![[0.1, 0.2]];
    let rho = 0.9;
    let lr = 0.001;
    let epsilon = 1e-8;

    RMSpropCache::update_param(&mut param, &grad, &mut cache, rho, lr, epsilon);

    // Check that cache has been updated
    // cache = rho * cache + (1 - rho) * grad^2
    assert!((cache[[0, 0]] - 0.001).abs() < 1e-6); // 0.9 * 0 + 0.1 * 0.01 = 0.001
    assert!((cache[[0, 1]] - 0.004).abs() < 1e-6); // 0.9 * 0 + 0.1 * 0.04 = 0.004

    // Check that parameters have been updated
    // param = param - lr * grad / (sqrt(cache) + epsilon)
    let expected_param_0 = 1.0 - lr * 0.1 / (0.001_f32.sqrt() + epsilon);
    let expected_param_1 = 2.0 - lr * 0.2 / (0.004_f32.sqrt() + epsilon);
    assert!((param[[0, 0]] - expected_param_0).abs() < 1e-6);
    assert!((param[[0, 1]] - expected_param_1).abs() < 1e-6);
}

#[test]
fn test_rmsprop_cache_update_parameters() {
    let mut cache = RMSpropCache::new((2, 2), Some((2, 2)), (1, 2));
    let mut param = array![[1.0, 2.0], [3.0, 4.0]];
    let mut recurrent_param = array![[0.5, 0.6], [0.7, 0.8]];
    let mut bias_param = array![[0.1, 0.2]];
    let grad = array![[0.1, 0.2], [0.3, 0.4]];
    let recurrent_grad = array![[0.05, 0.06], [0.07, 0.08]];
    let bias_grad = array![[0.01, 0.02]];
    let rho = 0.9;
    let lr = 0.001;
    let epsilon = 1e-8;

    cache.update_parameters(
        &mut param,
        Some(&mut recurrent_param),
        &mut bias_param,
        &grad,
        Some(&recurrent_grad),
        &bias_grad,
        rho,
        lr,
        epsilon,
    );

    // Check that all caches have been updated
    assert!((cache.cache[[0, 0]] - 0.001).abs() < 1e-6); // 0.1 * 0.01 = 0.001
    assert!((cache.cache[[0, 1]] - 0.004).abs() < 1e-6); // 0.1 * 0.04 = 0.004
    assert!((cache.cache[[1, 0]] - 0.009).abs() < 1e-6); // 0.1 * 0.09 = 0.009
    assert!((cache.cache[[1, 1]] - 0.016).abs() < 1e-6); // 0.1 * 0.16 = 0.016

    assert!(cache.cache_recurrent.is_some());
    let cache_rec = cache.cache_recurrent.as_ref().unwrap();
    assert!((cache_rec[[0, 0]] - 0.00025).abs() < 1e-6); // 0.1 * 0.0025 = 0.00025
    assert!((cache_rec[[0, 1]] - 0.00036).abs() < 1e-6); // 0.1 * 0.0036 = 0.00036

    assert!((cache.bias[[0, 0]] - 0.00001).abs() < 1e-8); // 0.1 * 0.0001 = 0.00001
    assert!((cache.bias[[0, 1]] - 0.00004).abs() < 1e-8); // 0.1 * 0.0004 = 0.00004

    // Check that parameters have been updated (values should be different from initial)
    assert!((param[[0, 0]] - 1.0).abs() > 1e-6);
    assert!((recurrent_param[[0, 0]] - 0.5).abs() > 1e-6);
    assert!((bias_param[[0, 0]] - 0.1).abs() > 1e-6);
}

#[test]
fn test_rmsprop_cache_accumulation_over_multiple_steps() {
    let mut cache = RMSpropCache::new((1, 2), None, (1, 2));
    let mut param = array![[1.0, 2.0]];
    let mut bias_param = array![[0.1, 0.2]];
    let grad = array![[0.1, 0.2]];
    let bias_grad = array![[0.01, 0.02]];
    let rho = 0.9;
    let lr = 0.001;
    let epsilon = 1e-8;

    // First update
    cache.update_parameters(
        &mut param,
        None,
        &mut bias_param,
        &grad,
        None,
        &bias_grad,
        rho,
        lr,
        epsilon,
    );

    let cache_after_first = cache.cache.clone();
    let bias_cache_after_first = cache.bias.clone();

    // Second update with the same gradients
    cache.update_parameters(
        &mut param,
        None,
        &mut bias_param,
        &grad,
        None,
        &bias_grad,
        rho,
        lr,
        epsilon,
    );

    // Check that caches have accumulated (not replaced)
    // cache_new = rho * cache_old + (1 - rho) * grad^2
    let expected_cache_0 = rho * cache_after_first[[0, 0]] + (1.0 - rho) * 0.01;
    let expected_cache_1 = rho * cache_after_first[[0, 1]] + (1.0 - rho) * 0.04;
    assert!((cache.cache[[0, 0]] - expected_cache_0).abs() < 1e-6);
    assert!((cache.cache[[0, 1]] - expected_cache_1).abs() < 1e-6);

    let expected_bias_cache_0 = rho * bias_cache_after_first[[0, 0]] + (1.0 - rho) * 0.0001;
    let expected_bias_cache_1 = rho * bias_cache_after_first[[0, 1]] + (1.0 - rho) * 0.0004;
    assert!((cache.bias[[0, 0]] - expected_bias_cache_0).abs() < 1e-8);
    assert!((cache.bias[[0, 1]] - expected_bias_cache_1).abs() < 1e-8);
}

#[test]
fn test_sgd_new() {
    // Valid learning rate
    let sgd = SGD::new(0.01);
    assert!(sgd.is_ok());

    // Invalid learning rate (negative)
    let sgd = SGD::new(-0.01);
    assert!(sgd.is_err());

    // Invalid learning rate (zero)
    let sgd = SGD::new(0.0);
    assert!(sgd.is_err());

    // Invalid learning rate (infinite)
    let sgd = SGD::new(f32::INFINITY);
    assert!(sgd.is_err());
}

#[test]
fn test_sgd_update_parameters() {
    let mut weights = vec![1.0, 2.0, 3.0, 4.0];
    let weight_grads = vec![0.1, 0.2, 0.3, 0.4];
    let mut bias = vec![0.5, 1.0];
    let bias_grads = vec![0.05, 0.1];
    let lr = 0.1;

    SGD::update_sgd_parameters(&mut weights, &weight_grads, &mut bias, &bias_grads, lr);

    // Check if parameters are updated correctly: param = param - lr * grad
    assert!((weights[0] - 0.99).abs() < 1e-6); // 1.0 - 0.1 * 0.1 = 0.99
    assert!((weights[1] - 1.98).abs() < 1e-6); // 2.0 - 0.1 * 0.2 = 1.98
    assert!((weights[2] - 2.97).abs() < 1e-6); // 3.0 - 0.1 * 0.3 = 2.97
    assert!((weights[3] - 3.96).abs() < 1e-6); // 4.0 - 0.1 * 0.4 = 3.96
    assert!((bias[0] - 0.495).abs() < 1e-6); // 0.5 - 0.1 * 0.05 = 0.495
    assert!((bias[1] - 0.99).abs() < 1e-6); // 1.0 - 0.1 * 0.1 = 0.99
}

#[test]
fn test_sgd_update_parameters_rnn() {
    let mut kernel = array![[1.0, 2.0], [3.0, 4.0]];
    let grad_kernel = array![[0.1, 0.2], [0.3, 0.4]];
    let mut recurrent_kernel = array![[0.5, 0.6], [0.7, 0.8]];
    let grad_recurrent_kernel = array![[0.05, 0.06], [0.07, 0.08]];
    let mut bias = array![[0.1, 0.2]];
    let grad_bias = array![[0.01, 0.02]];
    let lr = 0.1;

    SGD::update_sgd_parameters_rnn(
        &mut kernel,
        &grad_kernel,
        &mut recurrent_kernel,
        &grad_recurrent_kernel,
        &mut bias,
        &grad_bias,
        lr,
    );

    // Check kernel updates
    assert!((kernel[[0, 0]] - 0.99).abs() < 1e-6); // 1.0 - 0.1 * 0.1 = 0.99
    assert!((kernel[[0, 1]] - 1.98).abs() < 1e-6); // 2.0 - 0.1 * 0.2 = 1.98
    assert!((kernel[[1, 0]] - 2.97).abs() < 1e-6); // 3.0 - 0.1 * 0.3 = 2.97
    assert!((kernel[[1, 1]] - 3.96).abs() < 1e-6); // 4.0 - 0.1 * 0.4 = 3.96

    // Check recurrent kernel updates
    assert!((recurrent_kernel[[0, 0]] - 0.495).abs() < 1e-6); // 0.5 - 0.1 * 0.05 = 0.495
    assert!((recurrent_kernel[[0, 1]] - 0.594).abs() < 1e-6); // 0.6 - 0.1 * 0.06 = 0.594
    assert!((recurrent_kernel[[1, 0]] - 0.693).abs() < 1e-6); // 0.7 - 0.1 * 0.07 = 0.693
    assert!((recurrent_kernel[[1, 1]] - 0.792).abs() < 1e-6); // 0.8 - 0.1 * 0.08 = 0.792

    // Check bias updates
    assert!((bias[[0, 0]] - 0.099).abs() < 1e-6); // 0.1 - 0.1 * 0.01 = 0.099
    assert!((bias[[0, 1]] - 0.198).abs() < 1e-6); // 0.2 - 0.1 * 0.02 = 0.198
}

#[test]
fn test_ada_grad_new() {
    // Valid parameters
    let ada_grad = AdaGrad::new(0.01, 1e-8);
    assert!(ada_grad.is_ok());

    // Invalid learning rate (negative)
    let ada_grad = AdaGrad::new(-0.01, 1e-8);
    assert!(ada_grad.is_err());

    // Invalid learning rate (zero)
    let ada_grad = AdaGrad::new(0.0, 1e-8);
    assert!(ada_grad.is_err());

    // Invalid epsilon (negative)
    let ada_grad = AdaGrad::new(0.01, -1e-8);
    assert!(ada_grad.is_err());

    // Invalid epsilon (zero)
    let ada_grad = AdaGrad::new(0.01, 0.0);
    assert!(ada_grad.is_err());

    // Invalid learning rate (infinite)
    let ada_grad = AdaGrad::new(f32::INFINITY, 1e-8);
    assert!(ada_grad.is_err());

    // Invalid epsilon (infinite)
    let ada_grad = AdaGrad::new(0.01, f32::INFINITY);
    assert!(ada_grad.is_err());
}

#[test]
fn test_ada_grad_states_new() {
    let states = AdaGradStates::new((2, 3), Some((3, 3)), (1, 3));

    assert_eq!(states.accumulator.shape(), &[2, 3]);
    assert!(states.accumulator_recurrent.is_some());
    assert_eq!(
        states.accumulator_recurrent.as_ref().unwrap().shape(),
        &[3, 3]
    );
    assert_eq!(states.accumulator_bias.shape(), &[1, 3]);

    // Check that all values are initialized to zero
    assert!(states.accumulator.iter().all(|&x| x == 0.0));
    assert!(
        states
            .accumulator_recurrent
            .as_ref()
            .unwrap()
            .iter()
            .all(|&x| x == 0.0)
    );
    assert!(states.accumulator_bias.iter().all(|&x| x == 0.0));
}

#[test]
fn test_ada_grad_states_update_parameter() {
    let mut states = AdaGradStates::new((1, 2), None, (1, 2));
    let grad_param = array![[0.1, 0.1]];
    let grad_bias = array![[0.01, 0.01]];
    let epsilon = 1e-8;
    let lr = 0.01;

    let (param_update, recurrent_update, bias_update) =
        states.update_parameter(&grad_param, None, &grad_bias, epsilon, lr);

    // Check that accumulators have been updated
    assert!((states.accumulator[[0, 0]] - 0.01).abs() < 1e-6); // 0.1^2 = 0.01
    assert!((states.accumulator[[0, 1]] - 0.01).abs() < 1e-6);
    assert!((states.accumulator_bias[[0, 0]] - 0.0001).abs() < 1e-6); // 0.01^2 = 0.0001
    assert!((states.accumulator_bias[[0, 1]] - 0.0001).abs() < 1e-6);

    // Check that updates are calculated correctly
    // update = lr * grad / (sqrt(accumulator) + epsilon)
    let expected_param_update = lr * 0.1 / (0.01_f32.sqrt() + epsilon);
    let expected_bias_update = lr * 0.01 / (0.0001_f32.sqrt() + epsilon);

    assert!((param_update[[0, 0]] - expected_param_update).abs() < 1e-6);
    assert!((param_update[[0, 1]] - expected_param_update).abs() < 1e-6);
    assert!((bias_update[[0, 0]] - expected_bias_update).abs() < 1e-6);
    assert!((bias_update[[0, 1]] - expected_bias_update).abs() < 1e-6);
    assert!(recurrent_update.is_none());
}

#[test]
fn test_ada_grad_states_update_parameter_with_recurrent() {
    let mut states = AdaGradStates::new((2, 2), Some((2, 2)), (1, 2));
    let grad_param = array![[0.1, 0.2], [0.3, 0.4]];
    let grad_recurrent = array![[0.05, 0.06], [0.07, 0.08]];
    let grad_bias = array![[0.01, 0.02]];
    let epsilon = 1e-8;
    let lr = 0.01;

    let (param_update, recurrent_update, bias_update) =
        states.update_parameter(&grad_param, Some(&grad_recurrent), &grad_bias, epsilon, lr);

    // Check that all accumulators have been updated
    assert!((states.accumulator[[0, 0]] - 0.01).abs() < 1e-6); // 0.1^2 = 0.01
    assert!((states.accumulator[[0, 1]] - 0.04).abs() < 1e-6); // 0.2^2 = 0.04
    assert!((states.accumulator[[1, 0]] - 0.09).abs() < 1e-6); // 0.3^2 = 0.09
    assert!((states.accumulator[[1, 1]] - 0.16).abs() < 1e-6); // 0.4^2 = 0.16

    assert!(states.accumulator_recurrent.is_some());
    let acc_recurrent = states.accumulator_recurrent.as_ref().unwrap();
    assert!((acc_recurrent[[0, 0]] - 0.0025).abs() < 1e-6); // 0.05^2 = 0.0025
    assert!((acc_recurrent[[0, 1]] - 0.0036).abs() < 1e-6); // 0.06^2 = 0.0036
    assert!((acc_recurrent[[1, 0]] - 0.0049).abs() < 1e-6); // 0.07^2 = 0.0049
    assert!((acc_recurrent[[1, 1]] - 0.0064).abs() < 1e-6); // 0.08^2 = 0.0064

    assert!((states.accumulator_bias[[0, 0]] - 0.0001).abs() < 1e-6); // 0.01^2 = 0.0001
    assert!((states.accumulator_bias[[0, 1]] - 0.0004).abs() < 1e-6); // 0.02^2 = 0.0004

    // Check that updates are returned
    assert_eq!(param_update.shape(), &[2, 2]);
    assert!(recurrent_update.is_some());
    assert_eq!(recurrent_update.as_ref().unwrap().shape(), &[2, 2]);
    assert_eq!(bias_update.shape(), &[1, 2]);
}

#[test]
fn test_ada_grad_states_accumulation_over_multiple_steps() {
    let mut states = AdaGradStates::new((1, 2), None, (1, 2));
    let grad_param = array![[0.1, 0.2]];
    let grad_bias = array![[0.01, 0.02]];
    let epsilon = 1e-8;
    let lr = 0.01;

    // First update
    states.update_parameter(&grad_param, None, &grad_bias, epsilon, lr);

    let acc_after_first = states.accumulator.clone();
    let acc_bias_after_first = states.accumulator_bias.clone();

    // Second update with the same gradients
    states.update_parameter(&grad_param, None, &grad_bias, epsilon, lr);

    // Check that accumulators have accumulated (not replaced)
    assert!((states.accumulator[[0, 0]] - (acc_after_first[[0, 0]] + 0.01)).abs() < 1e-6);
    assert!((states.accumulator[[0, 1]] - (acc_after_first[[0, 1]] + 0.04)).abs() < 1e-6);
    assert!(
        (states.accumulator_bias[[0, 0]] - (acc_bias_after_first[[0, 0]] + 0.0001)).abs() < 1e-6
    );
    assert!(
        (states.accumulator_bias[[0, 1]] - (acc_bias_after_first[[0, 1]] + 0.0004)).abs() < 1e-6
    );
}
