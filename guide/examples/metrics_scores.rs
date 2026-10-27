use ndarray::array;
use rustyml::metrics::{
    Average, ConfusionMatrix, MulticlassConfusionMatrix, accuracy, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc, root_mean_squared_error,
};

fn main() {
    let actual = array![1.0, 2.0, 3.0];
    let predicted = array![1.0, 2.0, 4.0];
    println!("MSE: {:.3}", mean_squared_error(&actual, &predicted));
    println!("RMSE: {:.3}", root_mean_squared_error(&actual, &predicted));
    println!("MAE: {:.3}", mean_absolute_error(&actual, &predicted));
    println!("R2: {:.3}", r2_score(&actual, &predicted));
    let actual = array![0.0, 0.0, 1.0, 1.0];
    let predicted = array![0.0, 1.0, 1.0, 1.0];
    let matrix = ConfusionMatrix::new(&actual, &predicted);
    println!("accuracy: {:.3}", accuracy(&actual, &predicted));
    println!("TP, FP, TN, FN: {:?}", matrix.get_counts());
    println!(
        "precision: {:.3}, recall: {:.3}, F1: {:.3}",
        matrix.precision(),
        matrix.recall(),
        matrix.f1_score()
    );
    let labels = array![false, false, true, true];
    let scores = array![0.1, 0.4, 0.35, 0.8];
    println!("ROC AUC: {:.3}", roc_auc(&labels, &scores));
    let actual = array![0_usize, 1, 2, 2];
    let predicted = array![0_usize, 2, 2, 2];
    let matrix = MulticlassConfusionMatrix::new(&actual, &predicted);
    println!("macro F1: {:.3}", matrix.f1(Average::Macro));
}
