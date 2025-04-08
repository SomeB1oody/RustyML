use super::*;
use ndarray::s;

pub struct MeanSquaredError;

impl MeanSquaredError {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for MeanSquaredError {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // 计算预测值和真实值之间的差异
        let diff = y_pred - y_true;

        // 计算差异的平方
        let squared_diff = &diff.mapv(|x| x * x);

        // 计算平均值（总和除以元素数量）
        let n = squared_diff.len() as f32;
        squared_diff.sum() / n
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // 计算预测值和真实值之间的差异
        let diff = y_pred - y_true;

        // 梯度是差异的2倍除以样本数量
        let n = diff.len() as f32;
        diff.mapv(|x| 2.0 * x / n)
    }
}

pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for BinaryCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // 确保预测值在 (0,1) 范围内，避免数值问题
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // 二元交叉熵公式: -1/n * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        let losses = y_true.mapv(|y_t| y_t).to_owned() * &y_pred_clipped.mapv(|y_p| y_p.ln())
            + (1.0 - y_true).mapv(|y_t| y_t) * &(1.0 - &y_pred_clipped).mapv(|y_p| y_p.ln());

        // 计算平均损失（负号在这里加上）
        let n = losses.len() as f32;
        -losses.sum() / n
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // 确保预测值在 (0,1) 范围内，避免数值问题
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // 二元交叉熵的梯度: -y_true/y_pred + (1-y_true)/(1-y_pred)
        let grad = -y_true / &y_pred_clipped + (1.0 - y_true) / (1.0 - &y_pred_clipped);

        // 除以样本数量以获得平均梯度
        let n = grad.len() as f32;
        grad / n
    }
}

pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for CategoricalCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // 确保预测值在数值稳定的范围内，避免log(0)的问题
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // 计算多类别交叉熵：-Σ[y_true * log(y_pred)]
        // 这里y_true必须是one-hot编码
        let losses = y_true * &y_pred_clipped.mapv(|y_p| y_p.ln());

        // 计算平均损失（负号）
        let n = y_true.shape()[0] as f32; // 假设第一维是样本数
        -losses.sum() / n
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // 确保预测值在数值稳定的范围内
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // 多类别交叉熵的梯度是 -y_true / y_pred
        let grad = -y_true / &y_pred_clipped;

        // 除以样本数量以获得平均梯度
        let n = y_true.shape()[0] as f32; // 假设第一维是样本数
        grad / n
    }
}

pub struct SparseCategoricalCrossEntropy;

impl SparseCategoricalCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for SparseCategoricalCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // 确保预测值在数值稳定的范围内，避免log(0)的问题
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // 假设y_true包含类别索引（整数值）
        // 我们需要从每个样本的预测中获取对应真实类别的概率
        let mut total_loss = 0.0;
        let batch_size = y_true.shape()[0];

        for i in 0..batch_size {
            // 获取当前样本的真实类别索引
            let class_idx = y_true.slice(s![i, ..]).iter().next().unwrap().round() as usize;

            // 先保存切片视图，然后再从中提取值
            let slice = y_pred_clipped.slice(s![i, class_idx]);
            let predicted_prob = slice.iter().next().unwrap();

            // 累加交叉熵损失: -log(predicted_prob)
            total_loss -= predicted_prob.ln();
        }

        // 返回平均损失
        total_loss / batch_size as f32
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // 确保预测值在数值稳定的范围内
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // 创建一个与y_pred形状相同的梯度张量，初始值为0
        let mut grad = y_pred.clone();
        grad.fill(0.0);

        let batch_size = y_true.shape()[0];

        for i in 0..batch_size {
            // 获取当前样本的真实类别索引
            let class_idx = y_true.slice(s![i, ..]).iter().next().unwrap().round() as usize;

            // 先保存切片视图，然后再从中提取值
            let slice = y_pred_clipped.slice(s![i, class_idx]);
            let predicted_prob = slice.iter().next().unwrap();

            // 修改对应位置的梯度值
            let mut view = grad.slice_mut(s![i, class_idx]);
            *view.iter_mut().next().unwrap() = -1.0 / predicted_prob;
        }

        // 返回平均梯度
        grad / batch_size as f32
    }
}