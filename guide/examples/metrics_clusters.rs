use ndarray::array;
use rustyml::math::distance::DistanceCalculationMetric;
use rustyml::metrics::{adjusted_rand_index, normalized_mutual_info, silhouette_score};

fn main() {
    let x = array![[0.0], [1.0], [10.0], [11.0]];
    let reference = array![0_isize, 0, 1, 1];
    let clusters = array![7_isize, 7, 3, 3];
    println!("ARI: {:.3}", adjusted_rand_index(&reference, &clusters));
    println!("NMI: {:.3}", normalized_mutual_info(&reference, &clusters));
    println!(
        "silhouette: {:.3}",
        silhouette_score(&x, &clusters, DistanceCalculationMetric::Euclidean)
    );
}
