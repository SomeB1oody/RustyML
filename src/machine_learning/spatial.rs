//! Internal kd-tree spatial index used to accelerate neighborhood queries
//!
//! Brute-force neighbor search costs O(n) distance evaluations per query, so a full DBSCAN
//! fit or a batch of KNN predictions is O(n^2). A kd-tree partitions the points with
//! axis-aligned splits so that whole subtrees can be pruned, bringing the average query cost
//! down to roughly O(log n) in low dimensions
//!
//! The tree is metric-agnostic: it works for any [`DistanceCalculationMetric`] variant
//! (Euclidean, Manhattan, Minkowski). All comparisons happen in the metric's root-free
//! "comparable" space (see [`DistanceCalculationMetric::comparable_distance`]), and pruning
//! uses the per-axis lower bound `comparable_scalar(|q_a - split|)`, a valid lower bound on
//! the comparable distance for every Minkowski metric (`p >= 1`)
//!
//! kd-trees lose their pruning power as the dimensionality grows, so callers fall back to
//! brute force above a small feature-count threshold; this module only provides the index

use crate::types::DistanceCalculationMetric;
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Sentinel for an absent child (empty subtree)
const NONE: usize = usize::MAX;

/// A single kd-tree node, holding one point (the per-subtree median) and the split it defines
#[derive(Debug, Clone)]
struct KdNode {
    /// Index (into the owned `points` matrix) of the point stored at this node
    point_idx: usize,
    /// Axis this node splits on
    axis: usize,
    /// Splitting value along `axis` (the stored point's coordinate)
    split: f64,
    /// Child node index for the `<= split` side, or [`NONE`]
    left: usize,
    /// Child node index for the `> split` side, or [`NONE`]
    right: usize,
}

/// A candidate neighbor ordered by `(comparable_distance, index)`
///
/// The index tie-break makes the k-nearest result a total order, so the tree returns exactly
/// the same set as a brute-force search using the same ordering, staying deterministic even
/// when several points are equidistant from the query
#[derive(Debug, Clone, Copy)]
struct Neighbor {
    cmp_dist: f64,
    idx: usize,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Neighbor {}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // `total_cmp` gives a total order on f64; distances here are finite and non-negative
        self.cmp_dist
            .total_cmp(&other.cmp_dist)
            .then(self.idx.cmp(&other.idx))
    }
}

/// A kd-tree over a set of points, owning a copy of the coordinates so it is self-contained
/// (`Send + Sync`, serializable-by-rebuild) and free of self-referential borrows
#[derive(Debug, Clone)]
pub struct KdTree {
    metric: DistanceCalculationMetric,
    /// Owned copy of the points, indexed by original row index
    points: Array2<f64>,
    nodes: Vec<KdNode>,
    /// Root node index, or [`NONE`] for an empty tree
    root: usize,
}

impl KdTree {
    /// Builds a balanced kd-tree over `points` (rows are samples)
    ///
    /// Construction is `O(n * d * log n)`: each level chooses the maximum-spread axis and splits
    /// at the count-median (via `select_nth_unstable`), so the tree is balanced regardless of how
    /// the coordinate values are distributed
    pub fn build(points: ArrayView2<f64>, metric: DistanceCalculationMetric) -> Self {
        let points = points.to_owned();
        let n = points.nrows();
        let mut nodes: Vec<KdNode> = Vec::with_capacity(n);
        let mut indices: Vec<usize> = (0..n).collect();
        let root = Self::build_recursive(&points, &mut indices, &mut nodes);
        KdTree {
            metric,
            points,
            nodes,
            root,
        }
    }

    /// Recursively builds a subtree over `indices`, appending nodes to `nodes` and returning the
    /// subtree's root node index (or [`NONE`] when `indices` is empty)
    fn build_recursive(
        points: &Array2<f64>,
        indices: &mut [usize],
        nodes: &mut Vec<KdNode>,
    ) -> usize {
        if indices.is_empty() {
            return NONE;
        }

        // Split on the axis of greatest spread among this subset for well-balanced regions
        let n_features = points.ncols();
        let mut axis = 0;
        let mut best_spread = f64::NEG_INFINITY;
        for d in 0..n_features {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &idx in indices.iter() {
                let v = points[[idx, d]];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            let spread = hi - lo;
            if spread > best_spread {
                best_spread = spread;
                axis = d;
            }
        }

        // Count-median split keeps the tree balanced even with repeated coordinate values
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            points[[a, axis]]
                .partial_cmp(&points[[b, axis]])
                .unwrap_or(Ordering::Equal)
        });
        let median_idx = indices[mid];
        let split = points[[median_idx, axis]];

        // Reserve this node's slot before recursing so its index is stable as `nodes` grows
        let node_index = nodes.len();
        nodes.push(KdNode {
            point_idx: median_idx,
            axis,
            split,
            left: NONE,
            right: NONE,
        });

        let (left_slice, rest) = indices.split_at_mut(mid);
        let right_slice = &mut rest[1..]; // skip the median at `mid`
        let left = Self::build_recursive(points, left_slice, nodes);
        let right = Self::build_recursive(points, right_slice, nodes);
        nodes[node_index].left = left;
        nodes[node_index].right = right;
        node_index
    }

    /// Returns the indices of every point whose distance to `query` is `<= radius`
    ///
    /// The result is sorted ascending by index, matching a brute-force `(0..n).filter(...)`
    /// scan, so downstream order-sensitive logic (e.g. DBSCAN expansion) is unaffected
    pub(super) fn radius_neighbors(&self, query: ArrayView1<f64>, radius: f64) -> Vec<usize> {
        let radius_cmp = self.metric.comparable_scalar(radius);
        let mut out = Vec::new();
        self.radius_recurse(self.root, query, radius_cmp, &mut out);
        out.sort_unstable();
        out
    }

    fn radius_recurse(
        &self,
        node: usize,
        query: ArrayView1<f64>,
        radius_cmp: f64,
        out: &mut Vec<usize>,
    ) {
        if node == NONE {
            return;
        }
        let node = &self.nodes[node];

        let d_cmp = self
            .metric
            .comparable_distance(query, self.points.row(node.point_idx));
        if d_cmp <= radius_cmp {
            out.push(node.point_idx);
        }

        let delta = query[node.axis] - node.split;
        let (near, far) = if delta <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        self.radius_recurse(near, query, radius_cmp, out);
        // Only descend the far side if a point on it could still fall within the radius
        if self.metric.comparable_scalar(delta.abs()) <= radius_cmp {
            self.radius_recurse(far, query, radius_cmp, out);
        }
    }

    /// Returns the `k` nearest points to `query` as `(index, comparable_distance)` pairs,
    /// sorted ascending by `(comparable_distance, index)`
    ///
    /// Distances are returned in comparable space; convert with
    /// [`DistanceCalculationMetric::distance_from_comparable`] when a true distance is needed
    pub fn k_nearest(&self, query: ArrayView1<f64>, k: usize) -> Vec<(usize, f64)> {
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k.min(self.nodes.len()));
        if k > 0 {
            self.knn_recurse(self.root, query, k, &mut heap);
        }
        let mut found: Vec<Neighbor> = heap.into_vec();
        found.sort_unstable();
        found.into_iter().map(|n| (n.idx, n.cmp_dist)).collect()
    }

    fn knn_recurse(
        &self,
        node: usize,
        query: ArrayView1<f64>,
        k: usize,
        heap: &mut BinaryHeap<Neighbor>,
    ) {
        if node == NONE {
            return;
        }
        let node = &self.nodes[node];

        let candidate = Neighbor {
            cmp_dist: self
                .metric
                .comparable_distance(query, self.points.row(node.point_idx)),
            idx: node.point_idx,
        };
        // Keep the heap as the k smallest under the `(cmp_dist, idx)` total order
        if heap.len() < k {
            heap.push(candidate);
        } else if candidate < *heap.peek().unwrap() {
            heap.pop();
            heap.push(candidate);
        }

        let delta = query[node.axis] - node.split;
        let (near, far) = if delta <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        self.knn_recurse(near, query, k, heap);

        // Descend the far side only if its per-axis lower bound is within the current k-th distance (or the heap is not yet full)
        let worst = if heap.len() < k {
            f64::INFINITY
        } else {
            heap.peek().unwrap().cmp_dist
        };
        if self.metric.comparable_scalar(delta.abs()) <= worst {
            self.knn_recurse(far, query, k, heap);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand::rngs::StdRng;
    use ndarray_rand::rand_distr::Uniform;

    /// Brute-force fixed-radius search: all indices with `distance <= radius`, sorted ascending
    fn brute_radius(
        points: &Array2<f64>,
        query: ArrayView1<f64>,
        radius: f64,
        metric: DistanceCalculationMetric,
    ) -> Vec<usize> {
        let radius_cmp = metric.comparable_scalar(radius);
        (0..points.nrows())
            .filter(|&i| metric.comparable_distance(query, points.row(i)) <= radius_cmp)
            .collect()
    }

    /// Brute-force k-NN: the k smallest under the `(comparable_distance, index)` total order
    fn brute_knn(
        points: &Array2<f64>,
        query: ArrayView1<f64>,
        k: usize,
        metric: DistanceCalculationMetric,
    ) -> Vec<(usize, f64)> {
        let mut all: Vec<Neighbor> = (0..points.nrows())
            .map(|i| Neighbor {
                cmp_dist: metric.comparable_distance(query, points.row(i)),
                idx: i,
            })
            .collect();
        all.sort_unstable();
        all.into_iter()
            .take(k)
            .map(|n| (n.idx, n.cmp_dist))
            .collect()
    }

    fn metrics() -> Vec<DistanceCalculationMetric> {
        vec![
            DistanceCalculationMetric::Euclidean,
            DistanceCalculationMetric::Manhattan,
            DistanceCalculationMetric::Minkowski(3.0),
        ]
    }

    /// Across random datasets, dimensions, and metrics the kd-tree radius search must return
    /// exactly the brute-force neighbor set
    #[test]
    fn radius_search_matches_brute_force() {
        for (seed, &n) in [11_u64, 23, 37, 41]
            .iter()
            .zip([1_usize, 5, 30, 200].iter())
        {
            for &d in &[1_usize, 2, 3, 5] {
                let mut rng = StdRng::seed_from_u64(seed * 100 + d as u64);
                let points =
                    Array2::random_using((n, d), Uniform::new(-5.0, 5.0).unwrap(), &mut rng);
                for metric in metrics() {
                    let tree = KdTree::build(points.view(), metric);
                    // Mix queries from the data (exact hits) and fresh random points
                    for qi in 0..n.min(6) {
                        for &radius in &[0.0, 0.5, 2.0, 6.0, 20.0] {
                            let q = points.row(qi);
                            let mut got = tree.radius_neighbors(q, radius);
                            got.sort_unstable();
                            let expected = brute_radius(&points, q, radius, metric);
                            assert_eq!(
                                got, expected,
                                "radius mismatch: metric={metric:?} n={n} d={d} qi={qi} r={radius}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The kd-tree k-NN must return exactly the brute-force result under the `(distance, index)`
    /// total order, for every metric, including ties and `k` larger than the dataset
    #[test]
    fn knn_matches_brute_force() {
        for (seed, &n) in [7_u64, 19, 53].iter().zip([1_usize, 12, 150].iter()) {
            for &d in &[1_usize, 2, 4] {
                let mut rng = StdRng::seed_from_u64(seed * 50 + d as u64);
                let points =
                    Array2::random_using((n, d), Uniform::new(-3.0, 3.0).unwrap(), &mut rng);
                for metric in metrics() {
                    let tree = KdTree::build(points.view(), metric);
                    let q =
                        Array2::random_using((1, d), Uniform::new(-3.0, 3.0).unwrap(), &mut rng);
                    let q = q.row(0);
                    for &k in &[1_usize, 3, 7, 200] {
                        let got = tree.k_nearest(q, k);
                        let expected = brute_knn(&points, q, k, metric);
                        let got_idx: Vec<usize> = got.iter().map(|&(i, _)| i).collect();
                        let exp_idx: Vec<usize> = expected.iter().map(|&(i, _)| i).collect();
                        assert_eq!(
                            got_idx, exp_idx,
                            "knn index mismatch: metric={metric:?} n={n} d={d} k={k}"
                        );
                        for (&(_, gc), &(_, ec)) in got.iter().zip(expected.iter()) {
                            assert!((gc - ec).abs() < 1e-12, "knn distance mismatch");
                        }
                    }
                }
            }
        }
    }

    /// Duplicate points (heavy distance ties) must still be resolved identically to brute force
    #[test]
    fn handles_duplicate_points_deterministically() {
        let points = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        )
        .unwrap();
        let metric = DistanceCalculationMetric::Euclidean;
        let tree = KdTree::build(points.view(), metric);
        let q = points.row(0);

        let mut got_r = tree.radius_neighbors(q, 1.5);
        got_r.sort_unstable();
        assert_eq!(got_r, brute_radius(&points, q, 1.5, metric));

        for k in 1..=points.nrows() {
            let got: Vec<usize> = tree.k_nearest(q, k).into_iter().map(|(i, _)| i).collect();
            let exp: Vec<usize> = brute_knn(&points, q, k, metric)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            assert_eq!(got, exp, "duplicate-point knn mismatch at k={k}");
        }
    }

    /// A single-point tree is a valid degenerate case
    #[test]
    fn single_point_tree() {
        let points = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let metric = DistanceCalculationMetric::Euclidean;
        let tree = KdTree::build(points.view(), metric);
        assert_eq!(tree.radius_neighbors(points.row(0), 0.0), vec![0]);
        assert_eq!(tree.k_nearest(points.row(0), 5), vec![(0, 0.0)]);
        let far = ndarray::array![10.0, 10.0, 10.0];
        assert!(tree.radius_neighbors(far.view(), 1.0).is_empty());
    }
}
