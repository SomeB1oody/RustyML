use super::*;

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layer: &mut dyn Layer) {
        // 直接调用层的参数更新方法即可
        layer.update_parameters_sgd(self.learning_rate);
    }
}

/// Adam 优化器实现：内部保存 beta1、beta2、epsilon 及全局步数 t
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: u64,
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, layer: &mut dyn Layer) {
        self.t += 1; // 每次更新时增加步数
        layer.update_parameters_adam(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.t);
    }
}