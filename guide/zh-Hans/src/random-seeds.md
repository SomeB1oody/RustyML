# 随机种子

种子控制操作中的随机选择。
API 支持局部种子时，优先显式指定局部种子。
例如，`train_test_split` 的最后一个参数可以使用 `Some(42)`。

## 使用全局种子

`set_global_seed` 为当前线程设置种子流。
未指定局部种子的操作可以使用这个种子流。
重复同一组随机操作前，重新设置种子。

```rust
use ndarray::array;
use rustyml::error::RustymlResult;
use rustyml::utils::train_test_split::train_test_split;
use rustyml::{clear_global_seed, set_global_seed};

fn main() -> RustymlResult<()> {
    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    set_global_seed(42);
    let first = train_test_split(x.clone(), y.clone(), Some(0.33), None)?;
    set_global_seed(42);
    let second = train_test_split(x, y, Some(0.33), None)?;
    println!("same test rows: {}", first.1 == second.1);
    clear_global_seed();
    Ok(())
}
```

输出：

```text
same test rows: true
```

显式局部种子的优先级高于全局种子。
局部种子不会推进全局种子流。
两种种子都未设置时，需要随机性的操作可以使用系统熵源。

全局种子不会影响已经创建的随机数生成器。
其他线程有各自的种子流。
操作顺序改变时，未指定局部种子的操作可能获得不同种子。
后续操作不应再使用该种子流时，调用 `clear_global_seed`。

## 神经网络和求解器

构建神经网络前，可使用 `SequentialBuilder::new_with_seed(42)`。
它控制 batch 的打乱种子。
构建前，通过 `with_random_state` 设置随机层的种子。
使用 `model.set_seed(42)` 控制后续随机训练操作。
该方法不会重置现有权重。

某些求解器有单独的种子设置。
例如，`SVDSolver::Randomized` 接收显式种子。
全局种子不会替代这个设置。

数据、配置和操作顺序相同时，种子有助于重复实验。
不同平台或不同版本的库不保证产生完全相同的浮点结果。
