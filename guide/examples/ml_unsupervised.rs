use ndarray::array;
use rustyml::prelude::machine_learning::{DBSCAN, KMeans, PCA};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]];
    let mut kmeans = KMeans::new(2, 100, 1e-6)?.with_random_state(42);
    let labels = kmeans.fit_predict(&x)?;
    println!("same first pair: {}", labels[0] == labels[1]);
    println!("different groups: {}", labels[0] != labels[2]);
    let queries = array![[0.05, 0.05], [5.05, 5.05]];
    println!("KMeans predictions: {}", kmeans.predict(&queries)?.len());

    let mut dbscan = DBSCAN::new(0.3, 2)?;
    let labels = dbscan.fit_predict(&x)?;
    println!("DBSCAN labels: {:?}", labels.to_vec());
    let noise = dbscan.predict(&array![[20.0, 20.0]])?;
    println!("DBSCAN noise: {}", noise[0]);

    let mut pca = PCA::new(1)?;
    let reduced = pca.fit_transform(&x)?;
    println!("PCA shape: {:?}", reduced.dim());
    println!("PCA query shape: {:?}", pca.transform(&queries)?.dim());
    Ok(())
}
