use super::*;

/// Sequential 模型：模拟 Keras 的接口风格，支持链式调用 add、compile、fit 方法
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Option<Box<dyn Optimizer>>,
    loss: Option<Box<dyn LossFunction>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: None,
            loss: None,
        }
    }

    // 添加层，支持链式调用
    pub fn add<L: 'static + Layer>(&mut self, layer: L) -> &mut Self {
        self.layers.push(Box::new(layer));
        self
    }

    // 配置优化器和损失函数
    pub fn compile<O, LFunc>(&mut self, optimizer: O, loss: LFunc) -> &mut Self
    where
        O: 'static + Optimizer,
        LFunc: 'static + LossFunction,
    {
        self.optimizer = Some(Box::new(optimizer));
        self.loss = Some(Box::new(loss));
        self
    }

    // 简化的训练过程，依次执行前向传播、损失计算、反向传播与参数更新
    pub fn fit(&mut self, x: &Tensor, y: &Tensor, epochs: u32) {
        for epoch in 0..epochs {
            println!("Epoch {}", epoch + 1);
            // 前向传播
            let mut output = x.clone();
            for layer in &mut self.layers {
                output = layer.forward(&output);
            }
            // 计算损失
            let loss_value = self.loss.as_ref().unwrap().compute_loss(y, &output);
            println!("Loss: {}", loss_value);
            // 计算损失函数对输出的梯度
            let mut grad = self.loss.as_ref().unwrap().compute_grad(y, &output);
            // 反向传播和参数更新（逆序遍历各层）
            for layer in self.layers.iter_mut().rev() {
                grad = layer.backward(&grad);
                if let Some(ref mut optimizer) = self.optimizer {
                    optimizer.update(&mut **layer);
                }
            }
        }
    }

    // predict 方法：仅执行前向传播，返回预测结果
    pub fn predict(&mut self, x: &Tensor) -> Tensor {
        let mut output = x.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    // summary 方法：打印模型中每一层的基本信息
    // summary 方法：以表格形式打印模型各层信息及参数统计
    // summary 方法：以表格形式打印模型各层信息及参数统计
    pub fn summary(&self) {
        let col1_width = 33;
        let col2_width = 24;
        let col3_width = 15;
        println!("Model: \"sequential\"");
        println!(
            "┏{}┳{}┳{}┓",
            "━".repeat(col1_width),
            "━".repeat(col2_width),
            "━".repeat(col3_width)
        );
        println!(
            "┃ {:<31} ┃ {:<22} ┃ {:>13} ┃",
            "Layer (type)", "Output Shape", "Param #"
        );
        println!(
            "┡{}╇{}╇{}┩",
            "━".repeat(col1_width),
            "━".repeat(col2_width),
            "━".repeat(col3_width)
        );
        let mut total_params = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            // 为每一层生成名称，第一层命名为 "dense"，之后依次 "dense_1", "dense_2", ...
            let layer_name = if i == 0 {
                "dense".to_string()
            } else {
                format!("dense_{}", i)
            };
            let out_shape = layer.output_shape();
            let param_count = layer.param_count();
            total_params += param_count;
            println!(
                "│ {:<31} │ {:<22} │ {:>13} │",
                format!("{} ({})", layer_name, layer.layer_type()),
                out_shape,
                param_count
            );
        }
        println!(
            "└{}┴{}┴{}┘",
            "─".repeat(col1_width),
            "─".repeat(col2_width),
            "─".repeat(col3_width)
        );
        println!(
            " Total params: {} ({} B)",
            total_params,
            total_params * 4
        ); // 使用f32，每个参数 4 字节
        println!(
            " Trainable params: {} ({} B)",
            total_params,
            total_params * 4
        );
        println!(" Non-trainable params: 0 (0 B)");
    }
}