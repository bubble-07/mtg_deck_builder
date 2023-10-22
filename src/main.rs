use rand::prelude::SliceRandom;
use rand::Rng;
use rand_distr::WeightedIndex;
use rand_distr::Distribution;
use std::env;
use std::io::BufRead;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::AddAssign;
use serde_derive::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use ndarray::*;
use ndarray_linalg::UPLO;
use stats::OnlineStats;
use std::str::FromStr;
use nalgebra_sparse::coo::CooMatrix;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_linalg::qr::QR;
use ndarray_linalg::Eigh;
use ndarray_linalg::Inverse;

extern crate openblas_src;

// TODO: We currently don't understand the "partner" mechanic -
// could we fix that?

fn print_usage() -> ! {
    println!("Usage:");
    println!("cargo run card_incidence_stats mtgtop8 [mtg_top_8_data_base_directory]");
    println!("cargo run card_incidence_stats protour [mtg_pro_tour_csv]");
    println!("cargo run card_incidence_stats deckstobeat [deckstobeat_base_directory]");
    println!("cargo run card_incidence_stats edhrec [edhrec_json_base_directory]");
    println!("cargo run merge_incidence_stats [incidence_csv_one] [incidence_csv_two]");
    println!("cargo run combine_incidence_stats_with_metadata [incidence_csv] [metadata_csv]");
    println!("cargo run card_metadata [scryfall_oracle_cards_db_file]");
    println!("cargo run card_names_from_metadata [metadata_csv]");
    println!("cargo run preprocess [combined_data_csv] [destination]");
    println!("cargo run complete [method] [preprocessed_data] [destination]");
    println!("cargo run merge_preprocessed_trusted_with_untrusted [trusted_preprocessed_data] [untrusted_preprocessed_data] [destination]");
    println!("cargo run filter_preprocessed [preprocessed_data] [card_list] [destination]");
    println!("cargo run suggest_card_purchases [preprocessed_data] [card_list]");
    println!("cargo run rank_commanders [preprocessed_data]");
    println!("cargo run build_commander_deck [preprocessed_data] [constraints_file] [commander_name]");
    panic!();
}

// TODO: should define a file-format for "deck-building rules"

/// Parses a (quoted) card name, including converting escaped characters
/// to ordinary ones.
fn parse_card_name(quoted_card_name: &str) -> String {
    let whitespace_trimmed = quoted_card_name.trim();
    let quotes_removed = if whitespace_trimmed.starts_with("\"") {
        let after_quotes = whitespace_trimmed.strip_prefix('\"').unwrap();
        let inside_quotes = after_quotes.strip_suffix('\"').unwrap();
        inside_quotes
    } else {
        whitespace_trimmed
    };
    let quotes_unescaped = quotes_removed.replace("\"\"", "\"");
    standardize_card_name(&quotes_unescaped)
}

/// Standardizes a card name to use common formatting
fn standardize_card_name(card_name: &str) -> String {
    // Replace single-slash "double-faced card" notation
    // with scryfall's double-slash notation.
    let double_slashed = card_name.trim().replace(" / ", " // ");
    double_slashed
}

/// Formats a (raw) card name to have quotes around it, and to escape
/// any internal quotes.
fn format_card_name(raw_card_name: &str) -> String {
    let escaped_card_name = raw_card_name.replace("\"", "\"\"");
    format!("\"{}\"", escaped_card_name)
}

fn sparse_times_dense(sparse: &CsrMatrix<f64>, dense: ArrayView2<f32>) 
    -> Array2<f32> {
    let (l, m) = (sparse.nrows(), sparse.ncols());
    let n = dense.shape()[1];
    let mut result = Array::zeros((l, n));
    for (i, j, sparse_value) in sparse.triplet_iter() {
        for k in 0..n {
            result[[i, k]] += (*sparse_value as f32) * dense[[j, k]];
        }
    }
    result
}

struct Constraint {
    filter: CardFilter,
    count_lower_bound_inclusive: usize,
    count_upper_bound_inclusive: usize,
}

impl Constraint {
    fn parse(text: &str) -> Self {
        let (quantities, remainder) = text.split_once(' ').unwrap();
        let filter = CardFilter::parse(remainder);
        if quantities.contains('-') {
            let (lower, upper) = quantities.split_once('-').unwrap();
            let lower = str::parse::<usize>(lower).unwrap();
            let upper = str::parse::<usize>(upper).unwrap();
            Self {
                filter,
                count_lower_bound_inclusive: lower,
                count_upper_bound_inclusive: upper,
            }
        } else {
            let count = str::parse::<usize>(quantities).unwrap();
            Self {
                filter,
                count_lower_bound_inclusive: count,
                count_upper_bound_inclusive: count,
            }
        }
    }
    fn count_distance(&self, actual_count: usize) -> usize {
        if actual_count < self.count_lower_bound_inclusive {
            self.count_lower_bound_inclusive - actual_count
        } else if actual_count > self.count_upper_bound_inclusive {
            actual_count - self.count_upper_bound_inclusive
        } else {
            0
        }
    }
}

struct Constraints {
    constraints: Vec<Constraint>,
}

impl Constraints {
    fn load_from_file<R: std::io::Read>(file: R) -> anyhow::Result<Self> {
        let reader = std::io::BufReader::new(file);
        let mut constraints = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            let constraint = Constraint::parse(line);
            constraints.push(constraint);
        }
        Ok(Self {
            constraints,
        })
    }
    fn get_actual_counts(&self, active_set_indices: &[usize],
                         all_names: &[String],
                         all_metadata: &[CardMetadata]) -> Vec<usize> {
        let mut result = Vec::new();
        for constraint in self.constraints.iter() {
            let mut count = 0;
            for index in active_set_indices {
                let name = &all_names[*index];
                let metadata = &all_metadata[*index];
                if constraint.filter.matches(name, metadata) {
                    count += 1;
                }
            }
            result.push(count);
        }
        result
    }
    fn count_distance(&self, actual_counts: &[usize]) -> usize {
        let mut result = 0;
        for (actual_count, constraint) in actual_counts.iter()
            .zip(self.constraints.iter()) {
            result += constraint.count_distance(*actual_count);
        }
        result
    }
    fn get_updated_counts_for_swap(&self,
        actual_counts: &[usize],
        removed_index: usize, added_index: usize,
        all_names: &[String],
        all_metadata: &[CardMetadata]) -> Vec<usize> {

        // Copy the previous collection of counts
        let mut result: Vec<usize> = actual_counts.iter().copied().collect();

        let removed_name = &all_names[removed_index];
        let removed_metadata = &all_metadata[removed_index];

        let added_name = &all_names[added_index];
        let added_metadata = &all_metadata[added_index];

        for (i, constraint) in self.constraints.iter().enumerate() {
            if constraint.filter.matches(removed_name, removed_metadata) {
                result[i] -= 1;
            }
            if constraint.filter.matches(added_name, added_metadata) {
                result[i] += 1;
            }
        }
        result
    }
}

/// An iterator over (active_set_index_to_swap_out,
/// full_set_index) pairs which is eventually guaranteed
/// to generate all such pairs, with random ordering.
/// Some pairs may be visited more than once.
struct SwapIndexIterator<'a> {
    weighted_index: &'a WeightedIndex<f32>,
    iter_count: usize,
    full_set_size: usize,
    random_choice_iteration_bound: usize,
    all_possible_results_random_order: Option<Vec<(usize, usize)>>,
}

impl<'a> SwapIndexIterator<'a> {
    pub fn new(weighted_index: &'a WeightedIndex<f32>,
        full_set_size: usize, random_choice_iteration_bound: usize) -> Self {
        Self {
            weighted_index,
            iter_count: 0,
            full_set_size,
            random_choice_iteration_bound,
            all_possible_results_random_order: None,
        }
    }
}

impl<'a> Iterator for SwapIndexIterator<'a> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(all_possible) = &self.all_possible_results_random_order {
            // We're in the second phase, index into the "all possible results"
            if self.iter_count >= all_possible.len() {
                None
            } else {
                let result = all_possible[self.iter_count];
                self.iter_count += 1;
                Some(result)
            }
        } else {
            // We're in the first phase; choose random elements
            let mut rng = rand::thread_rng();

            // Deliberately avoids swapping out the commander (index 0)
            let active_set_index_to_remove = (rng.gen::<usize>() % 99) + 1;
            let index_added = self.weighted_index.sample(&mut rng);

            let result = (active_set_index_to_remove, index_added);

            self.iter_count += 1;
            if self.iter_count >= self.random_choice_iteration_bound {
                // Prepare to switch over to the second phase
                let mut all_possible = Vec::new();
                for active_set_index_to_remove in 1..100 {
                    for index_added in 0..self.full_set_size {
                        all_possible.push((active_set_index_to_remove, index_added));
                    }
                }
                // Randomize the order of all possible options
                all_possible.shuffle(&mut rng);
                // Store the result in the iterator, and reset the iter count,
                // since we're on phase 2 and will use it for indexing.
                self.all_possible_results_random_order = Some(all_possible);
                self.iter_count = 0;
            }
            Some(result)
        }
    }
}

const BASIC_LAND_NAMES: [&str; 11] = [
    "Plains",
    "Mountain",
    "Swamp",
    "Island",
    "Forest",
    "Wastes",
    "Snow-Covered Plains",
    "Snow-Covered Mountain",
    "Snow-Covered Swamp",
    "Snow-Covered Island",
    "Snow-Covered Forest",
];

//TODO: this is stupid, is there a better way?
const VALID_PLANESWALKER_COMMANDERS: [&str; 29] = [
    "Commodore Guff",
    "Freyalise, Llanowar's Fury",
    "Daretti, Scrap Servant",
    "Ob Nixilis of the Black Oath",
    "Teferi, Temporal Archmage",
    "Nahiri, the Lithomancer",
    "Sivitri, Dragon Master",
    "Jared Carthalion",
    "Dihada, Binder of Wills",
    "Aminatou, the Fateshifter",
    "Rowan Kenrith",
    "Will Kenrith",
    "Tasha, the Witch Queen",
    "Minsc & Boo, Timeless Heroes",
    "Elminster",
    "Liliana, Heretical Healer // Liliana, Defiant Necromancer",
    "Grist, the Hunger Tide",
    "Mila, Crafty Companion // Lukka, Wayward Bonder",
    "Valki, God of Lies // Tibalt, Cosmic Impostor",
    "Jeska, Thrice Reborn",
    "Tevesh Szat, Doom of Fools",
    "Saheeli, the Gifted",
    "Lord Windgrace",
    "Estrid, the Masked",
    "Nicol Bolas, the Ravager // Nicol Bolas, the Arisen",
    "Nissa, Vastwood Seer // Nissa, Sage Animist",
    "Chandra, Fire of Kaladesh // Chandra, Roaring Flame",
    "Jace, Vryn's Prodigy // Jace, Telepath Unbound",
    "Kytheon, Hero of Akros // Gideon, Battle-Forged",
];

fn as_sparse(dense: &Array2<f32>) -> (CsrMatrix<f64>, usize) {
    let mut nnz = 0;
    let (rows, cols) = (dense.shape()[0], dense.shape()[1]);
    let mut result = CooMatrix::new(rows, cols);
    for ((i, j), value) in dense.indexed_iter() {
        let value = *value as f64;
        if value != 0.0 {
            result.push(i, j, value);
            nnz += 1;
        }
    }
    (CsrMatrix::from(&result), nnz)
}

fn to_sparse(dense: Array2<f32>) -> (CsrMatrix<f64>, usize) {
    as_sparse(&dense)
}

fn low_rank_entry(U: ArrayView2<f32>, S: ArrayView1<f32>, 
                  i: usize, j: usize) -> f32 {
    let mut result = 0.0;
    for k in 0..U.shape()[1] {
        // We do this directly to avoid extraneous allocations
        // in a tight loop.
        result += U[[i, k]] * U[[j, k]] * S[[k]];
    }
    result
}
fn low_rank_multiply(U: ArrayView2<f32>, S: ArrayView1<f32>, 
                     A: ArrayView2<f32>) -> Array2<f32> {
    let mut transformed = U.t().dot(&A);
    for ((i, j), value) in transformed.indexed_iter_mut() {
        *value *= S[[i,]];
    }
    U.dot(&transformed)
}
fn low_rank_sketch(U: ArrayView2<f32>, S: ArrayView1<f32>,
                   Q: ArrayView2<f32>) -> Array2<f32> {
    let sketch_rhs = U.t().dot(&Q); // dims (s x s)
    let mut sketch_rhs_scaled = sketch_rhs.clone();
    for ((i, j), value) in sketch_rhs_scaled.indexed_iter_mut() {
        *value *= S[[i,]];
    }
    sketch_rhs.t().dot(&sketch_rhs_scaled) // dims (s x s)
}

fn flat_view(X: &CsrMatrix<f64>) -> ArrayView1<f64> {
    ArrayView::from_shape((X.nnz(),), X.values()).unwrap()
}

// Combined data which has been pre-processed to bring
// it into a form which is immediately usable by the
// rank_commanders and build_commander_deck commands.
#[derive(Debug, Serialize, Deserialize)]
struct PreprocessedData {
    card_names: Vec<String>, 
    metadata: Vec<CardMetadata>,
    score_matrix: Array2<f32>,
}

impl PreprocessedData {
    fn write_to_path(&self, path: &str) -> anyhow::Result<()> {
        let bytes: Vec<u8> = postcard::to_allocvec(&self)?;
        std::fs::write(path, &bytes)?;
        Ok(())
    }
    fn load_from_path(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let result: PreprocessedData = postcard::from_bytes(&bytes)?;
        Ok(result)
    }
    fn complete_rank_one_pursuit(mut self) -> Self {
        // Uses the "Economic" version in https://arxiv.org/pdf/1404.1377.pdf,
        // but generalized to deal with `k` singular values at every iteration.
        let n = self.metadata.len();

        let k = 8;

        let (sparse_score_matrix, nnz) = to_sparse(self.score_matrix);
        let epsilon = 0.03;

        let mut original_matrix_frobenius_norm = 0.0;
        for (_, _, value) in sparse_score_matrix.triplet_iter() {
            original_matrix_frobenius_norm += *value * *value;
        }
        let original_matrix_frobenius_norm = original_matrix_frobenius_norm.sqrt();
        
        // Iteration matrix
        let mut X = sparse_score_matrix.clone();
        for (_, _, value) in X.triplet_iter_mut() {
            *value = 0.0;
        }

        // Coefficient vector (weights of low-rank components)
        let mut theta = Vec::new();

        // The low-rank factors (expressed as a single unit-norm vector,
        // we'll outer-product each of them with themselves to derive the M matrices)
        let mut low_rank_factors = Vec::new();

        loop {
            let mut current_residual_frob_norm = 0.0;
            let mut residual = sparse_score_matrix.clone();
            for (residual_value, x_value) in
                residual.values_mut().iter_mut().zip(X.values().iter()) {
                *residual_value -= x_value;
                current_residual_frob_norm += *residual_value * *residual_value;
            }
            let residual = residual;

            let current_residual_frob_norm = current_residual_frob_norm.sqrt();
            println!("Current residual frob norm: {}", current_residual_frob_norm);

            if (current_residual_frob_norm / original_matrix_frobenius_norm) <=
                epsilon {
                // We're actually done, time to reconstruct
                break;
            }

            // Pick out the top k singular vectors
            let residual_svd_result = svdlibrs::svd_dim(&residual, k).unwrap();

            // Construct M_{\omega}_l's
            let mut M_omegas = Vec::new();
            for l in 0..k {
                let u = residual_svd_result.ut.row(l);
                let mut M_omega = sparse_score_matrix.clone();
                for (i, j, value) in M_omega.triplet_iter_mut() {
                    *value = u[[i,]] * u[[j,]];
                }
                M_omegas.push(M_omega);
            }
            let M_omegas = M_omegas;

            // We're done with the singular vectors for this iteration,
            // store them for later.
            for l in 0..k {
                let u = residual_svd_result.ut.row(l);
                low_rank_factors.push(u.into_owned());
            }

            // Now, find the coefficients for each low-rank component
            // in the overall mixture.
            let y_flat = flat_view(&sparse_score_matrix);

            let mut Z_t = Array::zeros((k + 1, nnz));
            Z_t.row_mut(0).assign(&flat_view(&X));
            for (l, M_omega) in (0..k).zip(M_omegas.iter()) {
                Z_t.row_mut(l + 1).assign(&flat_view(&M_omega));
            }

            let right_factor = Z_t.dot(&y_flat);
            let normal_matrix = Z_t.dot(&Z_t.t());

            // Compute the pseudoinverse of the normal matrix
            let (mut normal_s, normal_u) = normal_matrix.eigh(UPLO::Upper).unwrap();
            for value in normal_s.iter_mut() {
                if *value != 0.0 {
                    *value = 1.0 / *value;
                }
            }
            let normal_matrix_inv = normal_u.dot(&Array2::from_diag(&normal_s))
                                            .dot(&normal_u.t());
            let coefs = normal_matrix_inv.dot(&right_factor);

            // Update theta
            for old_theta_value in theta.iter_mut() {
                // All of the entries which were rolled up into
                // X get multiplied by the first coefficient
                *old_theta_value *= coefs[[0,]];
            }
            // All of the other coefficients get appended raw
            for l in 0..k {
                theta.push(coefs[[l + 1,]]);
            }

            // Update X
            for x_value in X.values_mut() {
                *x_value *= coefs[[0,]];
            }
            for l in 0..k {
                let M_omega = &M_omegas[l];
                let coef = coefs[[l + 1,]];
                for (x_value, m_value) in
                    X.values_mut().iter_mut().zip(M_omega.values().iter()) {
                    *x_value += coef * m_value;
                }
            }
        }
        println!("Compacting low rank factors");
        let k = low_rank_factors.len();
        let mut us = Array::zeros((k, n));
        for (i, low_rank_factor) in low_rank_factors.into_iter().enumerate() {
            us.row_mut(i).assign(&low_rank_factor.mapv(|x| x as f32));
        }
        let us = us;

        println!("Step 1 of score matrix reconstruction");

        // Reconstruct the result
        self.score_matrix = Array::zeros((n, n));

        // Step 1 of reconstruction: Reconstruct low-rank component
        for l in 0..k {
            let u = us.row(l);
            let coef = theta[l] as f32;
            for i in 0..n {
                let scaled_row = (coef * u[[i,]]) * &u;
                self.score_matrix.row_mut(i).assign(&scaled_row);
           }
        }

        println!("Step 2 of score matrix reconstruction");

        // Step 2 of reconstruction: Fill known entries back in,
        // since we don't want to erase those
        for (i, j, value) in sparse_score_matrix.triplet_iter() {
            self.score_matrix[[i, j]] = *value as f32;
        }
        self
    }
    fn complete_soft_impute(mut self) -> Self {
        // Uses a simplified version of https://arxiv.org/pdf/1703.05487.pdf
        // employing a randomized SVD https://gregorygundersen.com/blog/2019/01/17/randomized-svd/
        //
        // The first algorithm is in turn an accelerated version of:
        // https://www.jmlr.org/papers/volume11/mazumder10a/mazumder10a.pdf
        let n = self.metadata.len();
        // The initial number of singular values to compute
        let mut num_singular_values = 2;
        let num_singular_values_growth_rate = 2.0;

        let (sparse_score_matrix, nnz) = to_sparse(self.score_matrix);

        //TODO: Allow customization!
        let singular_value_threshold = 0.001;
        println!("singular value threshold: {}", singular_value_threshold);        

        let step_size = 1.0;
        println!("Step size: {}", step_size);

        let epsilon = 0.01 * 0.01;

        let mut original_matrix_frobenius_norm = 0.0;
        for (_, _, value) in sparse_score_matrix.triplet_iter() {
            original_matrix_frobenius_norm += *value * *value;
        }
        let original_matrix_frobenius_norm = original_matrix_frobenius_norm.sqrt();

        // c is a counter which determines how much sequence
        // acceleration to apply
        let mut c: usize = 1;

        // From iteration to iteration, we maintain X = U S U^T as the
        // current best-estimate, but we also need to maintain
        // an X_old in order to apply the acceleration strategy.
        let mut U_old = Array::zeros((n, num_singular_values));
        let mut S_old = Array::zeros((num_singular_values,));
        let mut U = Array::zeros((n, num_singular_values));
        let mut S = Array::zeros((num_singular_values,));

        let mut objective_value = f32::MAX;
        let mut old_objective_value = f32::MAX;

        loop {
            // Find the value of theta, which determines how "aggressive"
            // we want to be with respect to acceleration.
            let theta = ((c - 1) as f32) / ((c + 2) as f32);
            let one_plus_theta = 1.0 + theta;

            // Now, we have the logical assignment
            // Y = (1 + theta) X - theta X_old

            // Now, we'll be setting Z = Y + A,
            // where A is a sparse matrix given by
            // step_size * (P_{\omega}(O) - P_{\omega}(Y)).
            
            // First, compute the sparse component A
            let mut sparse_component = sparse_score_matrix.clone();
            for (i, j, value) in sparse_component.triplet_iter_mut() {
                // Compute the (i, j) entry of X right here
                let x_value = low_rank_entry(U.view(), S.view(), i, j);
                // Compute the (i, j) entry of X_old
                let x_old_value = low_rank_entry(U_old.view(), S_old.view(), i, j);
                *value -= (one_plus_theta * x_value - theta * x_old_value) as f64;
                *value *= step_size;
            }
            // Now we have everything we need for Z, so time to compute
            // the SVT of that using a randomized SVD.
            // First, we're gonna approximate the range of Z
            let random_vectors = Array::random((n, num_singular_values), StandardNormal);

            let mut sparse_transformed = sparse_times_dense(&sparse_component, random_vectors.view());
            let y_transformed = one_plus_theta * 
                low_rank_multiply(U.view(), S.view(), random_vectors.view()) - 
            theta *
                low_rank_multiply(U_old.view(), S_old.view(), random_vectors.view());
                              
            let transformed = sparse_transformed + &y_transformed;
            // Q now contains an orthonormal approxmation of the
            // range of Z - this is of shape (n, s) [s number of singular vals]
            let (Q, _) = transformed.qr().unwrap();

            
            // Now, use the derived orthogonal projection to sketch Z

            // Sketch Y
            let y_sketched = one_plus_theta * 
                low_rank_sketch(U.view(), S.view(), Q.view()) -
            theta *
                low_rank_sketch(U_old.view(), S_old.view(), Q.view());

            // Sketch the sparse part
            let sparse_sketched = sparse_times_dense(&sparse_component, Q.view());
            let sparse_sketched = Q.t().dot(&sparse_sketched);

            // Combine into one sketch
            let sketched = y_sketched + &sparse_sketched;

            
            // Compute eigh of the sketch
            let (sketch_s, sketch_u) = sketched.eigh(UPLO::Upper).unwrap();


            // Persist the old values for S and U, since we're about to
            // update them.
            S_old = S;
            U_old = U;

            // Scale the eigh of the sketch up to get new values for S and U,
            // but where S hasn't yet been singular-value-truncated.
            S = sketch_s;
            U = Q.dot(&sketch_u);

            // Check to see whether/not we need to increase the number
            // of singular values to compute for the next iteration.
            let mut have_enough_singular_values = false;
            let mut x_nuclear_norm = 0.0;
            for value in S.iter() {
                let abs_value = value.abs();
                x_nuclear_norm += abs_value;
                if abs_value <= singular_value_threshold {
                    // We could threshold, so we have enough.
                    have_enough_singular_values = true;
                }
            }
            // Perform the singular value truncation.
            // This completes our update to X.
            for singular_value in S.iter_mut() {
                if singular_value.abs() <= singular_value_threshold {
                    *singular_value = 0.0;
                } else if *singular_value > 0.0 {
                    *singular_value -= singular_value_threshold;
                } else if *singular_value < 0.0 {
                    *singular_value += singular_value_threshold;
                }
            }
            
            // Adjust the number of singular values to compute if
            // we need to.
            if !have_enough_singular_values {
                num_singular_values = 
                    ((num_singular_values as f64) * 
                     num_singular_values_growth_rate) as usize;
                num_singular_values = num_singular_values.min(n);
                println!("Number of singular values increased: {}", 
                         num_singular_values);
            }

            // Now, determine what the new objective value is, and
            // adjust our parameters if it's a significant
            // regression from previous iterations.
            let mut diff_frobenius_norm_sq = 0.0;
            for (i, j, value) in sparse_score_matrix.triplet_iter() {
                let x_value = low_rank_entry(U.view(), S.view(), i, j);
                let diff = (*value as f32) - x_value;
                diff_frobenius_norm_sq += diff * diff;
            }
            old_objective_value = objective_value;
            objective_value = 0.5 * diff_frobenius_norm_sq + 
                              singular_value_threshold * x_nuclear_norm;

            println!("Objective value: {}, Frob norm: {}", objective_value,
                     diff_frobenius_norm_sq.sqrt());
            if objective_value > old_objective_value {
                // Regression, lay off the acceleration
                c = 1;
            } else {
                // Go faster!
                c += 1;
            }
        }
    }
    fn complete_svt(mut self) -> Self {
        let n = self.metadata.len();
        // The initial number of singular values to compute (less than
        // the total number of singular values) - gets multiplied by
        // a constant every time that more are needed.
        let mut num_singular_values = 2;
        // The growth rate for the number of singular values
        let num_singular_values_growth_rate = 2.0;

        // Uses https://arxiv.org/pdf/0810.3286.pdf
        let (sparse_score_matrix, nnz) = to_sparse(self.score_matrix);

        // Pick the singular value threshold to be the largest singular
        // value in the original matrix, so that it's roughly proportional.
        let initial_svd_result = svdlibrs::svd_dim(&sparse_score_matrix, 2).unwrap();
        let singular_value_threshold = initial_svd_result.s[[0,]];
        println!("singular value threshold: {}", singular_value_threshold);

        // Per the paper, pick the step size between 0 and 2
        let step_size = 2.0;
        println!("step size: {}", step_size);

        // The relative error at which we'll accept the reconstruction
        // For our problem, we pick this so that we'd expect that
        // the synergy scores of two valid commander decks don't
        // change their synergy ranking pre-and-post-completion,
        // assuming that all pairwise synergies were originally defined.
        let epsilon = 0.01 * 0.01;

        let mut X: Array2<f32> = Array::zeros((n, n));
        let mut Y: CsrMatrix<f64> = sparse_score_matrix.clone();

        let mut original_matrix_frobenius_norm = 0.0;
        for (_, _, value) in sparse_score_matrix.triplet_iter() {
            original_matrix_frobenius_norm += *value * *value;
        }
        let original_matrix_frobenius_norm = original_matrix_frobenius_norm.sqrt();

        loop {
            // First, update X by computing the singular value shrinkage
            // operator on the previous value of Y
            let mut svd_result = svdlibrs::svd_dim(&Y, num_singular_values).unwrap();
            // Check to see whether/not we need to increase the number
            // of singular values to compute for the next iteration.
            let mut have_enough_singular_values = false;
            for value in svd_result.s.iter() {
                if *value <= singular_value_threshold {
                    // We could threshold, so we have enough.
                    have_enough_singular_values = true;
                }
            }

            if !have_enough_singular_values {
                num_singular_values = 
                    ((num_singular_values as f64) * 
                     num_singular_values_growth_rate) as usize;
                num_singular_values = num_singular_values.min(n);
                println!("Number of singular values increased: {}", 
                         num_singular_values);
            }

            // Shrinkage operator on all of the s values
            for singular_value in svd_result.s.iter_mut() {
                *singular_value -= singular_value_threshold;
                if *singular_value < 0.0 {
                    *singular_value = 0.0;
                }
            }
            // Now, reconstruct the X matrix 
            let mut s_vt = svd_result.vt;
            for ((i, j), value) in s_vt.indexed_iter_mut() {
                *value *= svd_result.s[[i,]];
            }
            let s_vt = s_vt.mapv(|x| x as f32);
            let u = svd_result.ut.t().mapv(|x| x as f32);
            X = u.dot(&s_vt);

            let mut diff_frobenius_norm = 0.0;

            // X matrix reconstructed, now update Y
            for ((i, j, y_value), (_, _, m_value)) in 
                Y.triplet_iter_mut().zip(sparse_score_matrix.triplet_iter()) {
                let x_value = X[[i, j]] as f64;
                let diff = m_value - x_value;
                let step = step_size * diff;
                *y_value += step;
                diff_frobenius_norm += diff * diff;
            }
            let diff_frobenius_norm = diff_frobenius_norm.sqrt();
            println!("Diff Frobenius norm this iteration: {}",
                diff_frobenius_norm);
            if diff_frobenius_norm <= original_matrix_frobenius_norm * epsilon {
                break;
            }
        }
        self.score_matrix = X;
        self
    }
    fn suggest_card_purchases(self, card_list: DeckList) {
        let mut name_to_id_map = HashMap::new();
        for (id, card_name) in self.card_names.iter().enumerate() {
            name_to_id_map.insert(card_name.to_string(), id); 
        }
        // TODO: Should make this handle the case where the
        // deck list actually just has single-sides of double-faced cards
        let mut my_cards_vec = Array::zeros((self.metadata.len(),));
        for card in card_list.cards.into_iter() {
            if let Some(my_card_id) = name_to_id_map.get(&card) {
                my_cards_vec[[*my_card_id]] = 1.0;
            }
        }
        // Tallies the scores of other cards against my cards
        let active_score = self.score_matrix.dot(&my_cards_vec);
        let mut result = Vec::new();
        for (id, card_name) in self.card_names.into_iter().enumerate() {
            if my_cards_vec[[id]] > 0.0 {
                // We already have it
                continue;
            }
            let type_line = self.metadata[id].card_types;
            let price_cents = self.metadata[id].price_cents;
            if price_cents == 0 {
                // Price is actually undefined, skip 
                continue;
            }
            let score = active_score[[id]];
            let score_per_cent = score / (price_cents as f32);
            let price = (price_cents as f32) / 100.0;
            result.push((card_name, type_line, score, score_per_cent, price));
        }
        result.sort_by(|(_, _, a, _, _), (_, _, b, _, _)| b.partial_cmp(&a).unwrap());

        for (card_name, type_line, score, score_per_cent, price) in result.into_iter() {
            println!("{}, {}, {}, {}, {}", format_card_name(&card_name), 
                     type_line.as_characters(), score, score_per_cent, price);
        }
    }
    fn merge_with_untrusted(self, mut untrusted: PreprocessedData) -> Self {
        // First, determine the empirical multiplier for untrusted elements
        // in order to adjust them into the range of trusted values
        let mut my_card_name_to_id = HashMap::new();
        for (i, card_name) in self.card_names.iter().enumerate() {
            my_card_name_to_id.insert(card_name.to_string(), i);
        }

        let mut other_id_to_my_id = HashMap::new();
        for (other_id, card_name) in untrusted.card_names.iter().enumerate() {
            if let Some(my_id) = my_card_name_to_id.get(card_name) {
                other_id_to_my_id.insert(other_id, *my_id);
            }
        }

        let mut empirical_multiplier_stats = OnlineStats::new();

        for i_other in 0..untrusted.metadata.len() {
            if let Some(i_mine) = other_id_to_my_id.get(&i_other) {
                for j_other in 0..untrusted.metadata.len() {
                    if let Some(j_mine) = other_id_to_my_id.get(&j_other) {
                        let my_value = self.score_matrix[[*i_mine, *j_mine]];
                        let other_value = untrusted.score_matrix[[i_other, j_other]];
                        if my_value != 0.0 && other_value != 0.0 {
                            let empirical_multiplier = (my_value as f64) / (other_value as f64);
                            empirical_multiplier_stats.add(empirical_multiplier);
                        }
                    }
                }
            }
        }
        // Return the empirical multiplier mean minus the standard
        // deviation, which is guaranteed to be greater than zero,
        // but if the multipliers were normally distributed, it would
        // encompass a >80% CI.
        let adjusted_multiplier: f32 = (empirical_multiplier_stats.mean()
            - empirical_multiplier_stats.stddev()) as f32;

        // Now, mutate the untrusted values
        // NOTE: We assume that the "untrusted" data
        // every card in it!
        for ((i_other, j_other), value) in untrusted.score_matrix.indexed_iter_mut() {
            // First, adjust by the requested multiplier
            *value *= adjusted_multiplier;
            if let Some(i_mine) = other_id_to_my_id.get(&i_other) {
                if let Some(j_mine) = other_id_to_my_id.get(&j_other) {
                    // If the trusted matrix has a nonzero entry, substitute it.
                    let trusted_value = self.score_matrix[[*i_mine, *j_mine]];
                    if trusted_value != 0.0 {
                        *value = trusted_value;
                    }
                }
            }
        }
        untrusted
    }
    fn filter(self, criteria: impl Fn(&str, &CardMetadata) -> bool) -> Self {
        // First, scan through and assign new contiguous indices to all matches
        // We can straightforwardly update the names and metadata
        // for this in one go.
        let mut new_id_to_old_id = Vec::new();

        let mut metadata = Vec::new();
        let mut card_names = Vec::new();

        for (old_id, card_name) in self.card_names.into_iter().enumerate() {
            let card_metadata = &self.metadata[old_id];
            if !criteria(&card_name, card_metadata) {
                continue;
            }
            new_id_to_old_id.push(old_id);
            metadata.push(card_metadata.clone());
            card_names.push(card_name);
        }
        
        let metadata = metadata;
        let card_names = card_names;

        // Now, using the old-id-to-new-ID map, update the score matrix
        let mut score_matrix = Array::zeros((metadata.len(), metadata.len()));
        for ((new_id_one, new_id_two), value) in score_matrix.indexed_iter_mut() {
            let old_id_one = new_id_to_old_id[new_id_one];
            let old_id_two = new_id_to_old_id[new_id_two];
            *value = self.score_matrix[[old_id_one, old_id_two]];
        }
        let score_matrix = score_matrix;
        Self {
            metadata,
            card_names,
            score_matrix    
        }
    }
    fn get_id_from_name(&self, name: &str) -> Option<usize> {
        for (id, card_name) in self.card_names.iter().enumerate() {
            if card_name == name {
                return Some(id);
            }
        }
        None
    }
    pub fn build_commander_deck(self, constraints: Constraints, commander_name: &str) {
        // SETUP

        // Early bail if we can't get the ID
        let commander_id = self.get_id_from_name(commander_name)
            .expect("No commander was found by that name");

        // Early bail if there's not 100 or more cards
        // in the IDs
        if self.card_names.len() < 100 {
            panic!("Not enough cards to build a commander deck");
        }

        // Filter counts, ids, and metadata to only those
        // cards which are within the commander's color identity
        let commander_metadata = self.metadata[commander_id].clone();
        let commander_color_identity = commander_metadata.color_identity;

        let filtered_data = self.filter(move |_, metadata| {
            metadata.color_identity.fits_within(commander_color_identity)
        });

        // After filtering, the commander's ID will be different, but no biggie.
        let commander_id = filtered_data.get_id_from_name(commander_name).unwrap();

        // Get all counts for the data-set, and all ids-to-names
        let mut score_matrix = filtered_data.score_matrix;
        let reverse_id_map = filtered_data.card_names;
        let metadata = filtered_data.metadata;

        let num_cards_in_pool = metadata.len();

        // Now, adjust the scores of the basic lands so that they're
        // smaller than the scores of all other cards
        let mut basic_land_names = HashSet::new();
        for basic_land_name in BASIC_LAND_NAMES {
            basic_land_names.insert(basic_land_name.to_string());
        }
        let mut basic_land_ids = HashSet::new();
        for (id, name) in reverse_id_map.iter().enumerate() {
            if basic_land_names.contains(name) {
                basic_land_ids.insert(id);
            }
        }
        let basic_land_ids = basic_land_ids;

        for basic_land_id in basic_land_ids.iter() {
            let land_color = metadata[*basic_land_id].color_identity;
            for i in 0..num_cards_in_pool {
                let other_color = metadata[i].color_identity;
                if land_color.fits_within(other_color) {
                    let value = 0.0000001;
                    score_matrix[[i, *basic_land_id]] = value;
                    score_matrix[[*basic_land_id, i]] = value;
                }
            }
        }

        // Adjust the incidence scores of the commander to incorporate
        // a multiplier for just how much more frequently we expect
        // to have the commander co-occur with cards than the ones
        // in the opening drawn hand, compared against each other.

        // Adjust the incidence scores of the commander to incorporate
        // a multiplier for just how much more frequently the
        // commander co-occurs with other cards throughout the entire
        // game than the collection of cards which are already
        // potentially-in-play, assuming one card drawn per turn.
        // This is the integral of the gain over the incidence
        // for all card-sets of any given size, from 7 to 99
        // - the incidence gain is given by (100 / (n - 1)) - 1,
        // which was computed via combinatorial considerations.
        let commander_incidence_multiplier = 187.32;

        for i in 0..num_cards_in_pool {
            if i == commander_id {
                score_matrix[[commander_id, commander_id]] *=
                    commander_incidence_multiplier;
            } else {
                score_matrix[[commander_id, i]] *=
                    commander_incidence_multiplier;
                score_matrix[[i, commander_id]] *=
                    commander_incidence_multiplier;
            }
        }

        // Adjust the incidence scores of lands downward,
        // to account for the fact that only one may be cast
        // per turn (unlike other spells, where this is not necessarily
        // the case). We guesstimate that up to 3 spells may be cast
        // in a given turn, which is a little pessimistic toward
        // lands, but good lands are also expensive, so this isn't
        // actually that bad of an idea for the common-case.
        for i in 0..num_cards_in_pool {
            for j in 0..num_cards_in_pool {
                if metadata[i].card_types.has_type(CardType::Land) ||
                   metadata[j].card_types.has_type(CardType::Land) {
                    score_matrix[[i, j]] /= 3.0;
                }
            }
        }

        // Finally, zero the diagonal in the score matrix, with the
        // rough rationale being that there should be no isolated
        // "good stuff" cards which don't synergize with the rest
        // of the deck. Another line of reasoning is that the
        // diagonal doesn't represent true card pairings, and
        // so it should be excluded.
        for i in 0..num_cards_in_pool {
            score_matrix[[i, i]] = 0.0;
        }

        let score_matrix = score_matrix;

        // Number of optimization attempts
        // TODO: allow customization?
        let num_optimization_attempts = 5;
 
        // Rationale: ~1/2 the total number of possible choices
        // before we resort to sampling without replacement (more expensive,
        // O(num_cards_in_pool)) to verify that we've truly hit
        // a local maximum.
        let random_choice_iteration_bound = num_cards_in_pool * 10;

        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();
        // Construct the initial set of weighted probabilities which
        // we'll always use to pick the initial set of cards to draw from
        let mut total_commander_synergy = 0.0;
        for commander_synergy in score_matrix.row(commander_id).iter() {
            total_commander_synergy += commander_synergy;
        }
        for commander_synergy in score_matrix.row(commander_id).iter() {
            let commander_synergy_as_probability = commander_synergy / total_commander_synergy;
            // Take that, and average with a uniform distribution
            let uniform_probability = 1.0 / (num_cards_in_pool as f32);
            // TODO: allow configuring this value
            let alpha = 0.1;
            let weight = alpha * uniform_probability + (1.0 - alpha) * commander_synergy_as_probability;
            weights.push(weight);
        }
        let weighted_index = WeightedIndex::new(weights).unwrap();

        // OPTIMIZATION
        
        let mut best_objective_value = 0.0;
        let mut best_active_set_indices = Vec::new();
        let mut best_active_set = Array::zeros((num_cards_in_pool,));
        
        for _ in 0..num_optimization_attempts {
            // Now that we're filtered, pick an initial, random set of 100 cards
            // which includes the commander.
            let mut active_set_indices = Vec::new();
            let mut active_set = Array::zeros((num_cards_in_pool,));

            active_set_indices.push(commander_id);
            active_set[[commander_id,]] = 1.0f32;

            while active_set_indices.len() < 100 {
                let random_index = weighted_index.sample(&mut rng);
                if active_set[random_index] > 0.5 && 
                   !basic_land_ids.contains(&random_index) {
                    continue;
                }
                active_set_indices.push(random_index);
                active_set[[random_index,]] += 1.0;
            }


            // Determine the counts of constraint-filter-satisfying cards
            // in the current active set.
            let mut constraint_counts = constraints.get_actual_counts(&active_set_indices, &reverse_id_map, &metadata);
            let mut constraint_count_distance = constraints.count_distance(&constraint_counts);

            // Determine the current objective value
            let mut objective_value = active_set.dot(&score_matrix).dot(&active_set);

            // Determine the active score, which is going to be
            // a vector that we'll update frequently
            let mut active_score = score_matrix.dot(&active_set);
           
            let mut still_working = true;
            // Kick off the main optimization loop
            while still_working {
                // We're going to temporarily set this to false,
                // because we're not sure that this iteration
                // will yield any kind of improvement.
                still_working = false;
                for (active_set_index_to_remove, index_added) in
                    SwapIndexIterator::new(&weighted_index, num_cards_in_pool, 
                                           random_choice_iteration_bound) {
                    // Make sure that we're not going to be adding an index
                    // which is already in the set, if the index does
                    // not correspond to a basic land.
                    if active_set[[index_added,]] > 0.5
                    && !basic_land_ids.contains(&index_added) {
                        continue;
                    }

                    let index_removed = active_set_indices[active_set_index_to_remove];

                    if index_added == index_removed {
                        continue;
                    }

                    // Determine what the updated constraint counts would be
                    let updated_constraint_counts = constraints.get_updated_counts_for_swap(
                        &constraint_counts, index_removed, index_added, &reverse_id_map, &metadata);
                    let updated_constraint_count_distance = constraints.count_distance(&updated_constraint_counts);

                    if constraint_count_distance < updated_constraint_count_distance {
                        // It makes constraint violation worse; bail.
                        continue;
                    }

                    let is_feasible = updated_constraint_count_distance == 0;
                    
                    // Determine what the updated objective value would be.
                    let diffing_value = 
                        score_matrix[[index_added, index_added]]
                        - 2.0 * score_matrix[[index_added, index_removed]]
                        + score_matrix[[index_removed, index_removed]];

                    let updated_objective_value = objective_value
                        + 2.0 * (active_score[[index_added,]] - active_score[[index_removed,]])
                        + diffing_value;

                    if is_feasible && objective_value >= updated_objective_value {
                        // Still feasible, but doesn't make the objective
                        // value any better!
                        continue;
                    }

                    let updated_active_score = &active_score
                        + &score_matrix.row(index_added)
                        - &score_matrix.row(index_removed);

                    // Great, let's pick it. Update all state variables appropriately.
                    active_set[[index_added,]] += 1.0;
                    active_set[[index_removed,]] -= 1.0;
                    active_set_indices[active_set_index_to_remove] = index_added;
                    active_score = updated_active_score;
                    objective_value = updated_objective_value;
                    constraint_count_distance = updated_constraint_count_distance;
                    constraint_counts = updated_constraint_counts;
                    
                    // Indicate that we've completed this iteration, but
                    // we're still working on optimization.
                    still_working = true;
                    break;
                }
            }
            // Completed an optimization, check if the solution is better
            // than the other ones.
            if objective_value > best_objective_value {
                best_objective_value = objective_value;
                best_active_set_indices = active_set_indices;
                best_active_set = active_set;
            }
        }

        let active_score = score_matrix.dot(&best_active_set);

        // Order `best_active_set_indices` by decreasing order
        // of contribution to the total overall score.
        let mut sorted_indices = Vec::new();
        // Always put the commander at the top of the list
        sorted_indices.push((best_active_set_indices[0], f32::MAX));
        for i in 1..best_active_set_indices.len() {
            sorted_indices.push((best_active_set_indices[i], active_score[[i,]]));
        }
        sorted_indices.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let mut non_land_color_count = ColorCounts::new();
        // Print out the solution we found
        // Loop 1: Print out non-lands (required)
        // Loop 2: Print out lands (suggestions, could be swapped for basics)
        for (index, _) in sorted_indices.iter() {
            if !metadata[*index].card_types.has_type(CardType::Land) {
                non_land_color_count.add(metadata[*index].color_identity);
                println!("1 {}", reverse_id_map[*index]);
            }
        }
        println!("");
        // For the lands, we're going to print out all the non-basics,
        // but the basic lands will be saved for last
        let mut basic_land_ids_to_counts: HashMap<_, usize> = HashMap::new();
        let mut nonbasic_land_color_count = ColorCounts::new();
        for (index, _) in sorted_indices.iter() {
            if metadata[*index].card_types.has_type(CardType::Land) {
                if basic_land_ids.contains(index) {
                    if basic_land_ids_to_counts.contains_key(index) {
                        basic_land_ids_to_counts.get_mut(index).unwrap()
                                                .add_assign(1);
                    } else {
                        basic_land_ids_to_counts.insert(*index, 1);
                    }
                } else {
                    println!("1 {}", reverse_id_map[*index]);
                    nonbasic_land_color_count.add(metadata[*index].color_identity);
                }
            }
        }
        // Now, re-adjust the basic land counts to better-match the color
        // identities of the non-land cards in play. This needs to be done
        // because the optimization routine assigns a constant, very small
        // value to each basic land to prevent flooding.
        let mut total_basic_lands_count = 0;
        for (_, count) in basic_land_ids_to_counts.iter() {
            total_basic_lands_count += *count;
        }
        match non_land_color_count.get_colors_required_for_basics(
            nonbasic_land_color_count, total_basic_lands_count) {
            Some(recommended_color_counts) => {
                recommended_color_counts.print_basic_lands();
            },
            None => {
                // Something failed with that routine, instead just print
                // the basic land counts, unmodified
                for (id, count) in basic_land_ids_to_counts.drain() {
                    println!("{} {}", count, reverse_id_map[id]);
                }
            },
        }
    }

    pub fn rank_commanders(self) {
        let reverse_id_map = self.card_names;
        let score_matrix = self.score_matrix;

        let mut names_and_scores_by_number_of_colors = vec![Vec::new(); 6];

        for (commander_index, score_matrix_row) in score_matrix.outer_iter().enumerate() {
            let commander_metadata = &self.metadata[commander_index];
            if !commander_metadata.card_types.is_possibly_valid_commander() {
                continue;
            }
            // If the card types contains "planeswalker", make sure it's a valid
            // planeswalker-commander.
            if commander_metadata.card_types.has_type(CardType::Planeswalker)
               && !VALID_PLANESWALKER_COMMANDERS.contains(
                   &reverse_id_map[commander_index].as_str()) {
                continue;
            }
            let color_identity = commander_metadata.color_identity;
            let number_of_colors = color_identity.number_of_colors();
            // Collect the scores of all cards in the commander's color identity
            let mut valid_color_identity_scores = Vec::new();
            for (other_index, score) in score_matrix_row.indexed_iter() {
                let other_metadata = &self.metadata[other_index];
                let other_color_identity = other_metadata.color_identity;
                if other_color_identity.fits_within(color_identity) {
                    valid_color_identity_scores.push(*score);
                }
            }
            // Sort the scores and extract just the top 100
            // (the rationale being that it's the number of cards
            // in a commander deck, including the commander.)
            valid_color_identity_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
            valid_color_identity_scores.truncate(100);

            let total_score: f32 = valid_color_identity_scores.iter().sum();
            let name = &reverse_id_map[commander_index];

            names_and_scores_by_number_of_colors[number_of_colors].push((name, total_score));
        }
        for (number_of_colors, mut names_and_scores) in
            names_and_scores_by_number_of_colors.drain(..)
            .enumerate() {
            println!("{}-Color Commanders:", number_of_colors);
            // Sort by descending order of score
            names_and_scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            for (name, score) in names_and_scores.into_iter() {
                println!("{}, {}", format_card_name(&name), score);
            }
            println!("");
        }
    }
}

struct CombinedData {
    ids: HashMap<String, usize>,
    // Indexed by ID
    metadata: Vec<CardMetadata>,
    counts: HashMap<(usize, usize), usize>,
}

impl CombinedData {
    pub fn preprocess(self) -> PreprocessedData {
        let card_names = self.get_reverse_id_map();
        let score_matrix = self.get_score_matrix();
        PreprocessedData {
            card_names,
            score_matrix,
            metadata: self.metadata,
        }
    }
    pub fn get_reverse_id_map(&self) -> Vec<String> {
        let mut result = vec![None; self.metadata.len()];
        for (name, id) in self.ids.iter() {
            result[*id] = Some(name.to_string());
        }
        let result = result.into_iter()
            .map(|x| x.unwrap())
            .collect();
        result
    }
    pub fn get_score_matrix(&self) -> Array2<f32> {
        // We take the raw count matrix
        // and turn it into scores by taking
        // the natural logarithm of 1 + the entries.
        // 
        // The rough rationale for this is that we first do Laplace
        // smoothing (1+), and then take the occurrence
        // statistics and assume that while they're roughly
        // de facto different by powers of 10 (see the dataset
        // yourself!), their actual power levels likely lie on
        // a linear scale, and the reason they're so skewed
        // in top-ranked tournament play has to do with the
        // fact that there's an implicit smooth maximization
        // taking place there.
        let mut result = self.get_counts_matrix();
        for ((i, j), value) in result.indexed_iter_mut() {
            let mut updated_value = value.ln_1p();
            if i != j {
                // Halve off-diagonal values so that when
                // using them in the quadratic program, they sum
                // to the strength scoring for the pair.
                updated_value *= 0.5;
            }
            // Little bit of normalization to bring ourselves to a reasonable range
            updated_value *= 0.01;
            *value = updated_value;
        }
        result
    }
    pub fn get_counts_matrix(&self) -> Array2<f32> {
        let dim = self.metadata.len();
        // Pack the incidence counts into an array
        let mut similarity_array = Array::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                if let Some(count) = self.counts.get(&(i, j)) {
                    similarity_array[[i, j]] = *count as f32;
                } else if let Some(count) = self.counts.get(&(j, i)) {
                    similarity_array[[i, j]] = *count as f32;
                }
            }
        }
        similarity_array
    }
    pub fn from_combination(
        card_incidence_stats: CardIncidenceStats,
        cards_metadata: CardsMetadata,
    ) -> Self {
        let double_faced_cards = cards_metadata.double_faced_cards;

        // Set of incidence-stats IDs which do not have entries in metadata,
        // likely because they're not legal in the Commander format.
        let mut blacklisted_ids = HashSet::new();

        let mut old_id_to_new_id = HashMap::new();
        let mut next_id = 0;

        // First step: standardize the names used in the IDs mapping
        let mut modified_ids = HashMap::new();
        for (old_name, old_id) in card_incidence_stats.ids.into_iter() {
            let standardized_card_name = match double_faced_cards.get(&old_name) {
                Some(combined_card_name) => combined_card_name.to_string(),
                None => old_name,
            };
            if !cards_metadata.cards.contains_key(&standardized_card_name) {
                // Card wasn't in the metadata DB, so not a commander
                // card that we recognize.
                blacklisted_ids.insert(old_id);
                continue;
            }
            // Check whether/not the standardized name already
            // exists in the target map
            if let Some(existing_new_id) = modified_ids.get(&standardized_card_name) {
                // Previously existed, so we'll have to instead
                // merge all of the data for the entry we're currently
                // iterating over into this pre-existing entry.
                //
                // We actually do this outside of this loop, for now,
                // just record the fact that the old id maps here.
                old_id_to_new_id.insert(old_id, *existing_new_id);
                continue;
            } else {
                // Didn't exist before, go ahead and insert
                modified_ids.insert(standardized_card_name, next_id);
            }
            // By default, assume that we inserted, and update the state
            old_id_to_new_id.insert(old_id, next_id);
            next_id += 1;
        }
        // The ID map is finalized
        let ids = modified_ids;
        let blacklisted_ids = blacklisted_ids;

        // Now, fix up the metadata mapping to be index-based
        // instead of being name-based.
        let mut modified_metadata = vec![None; next_id];
        for (name, id) in ids.iter() {
            // We already verified that they'd be here, otherwise
            // they'd be in the blacklist, so this is okay.
            let metadata = cards_metadata.cards.get(name).unwrap();
            modified_metadata[*id] = Some(metadata.clone());
        }
        let metadata = modified_metadata.into_iter()
            .map(|x| x.unwrap()).collect();

        // Finally, fix up the counts
        let mut modified_counts = HashMap::<(usize, usize), usize>::new();
        for ((old_id_one, old_id_two), count) in card_incidence_stats.counts.into_iter() {
            if blacklisted_ids.contains(&old_id_one) ||
               blacklisted_ids.contains(&old_id_two) {
                continue;
            }
            let new_id_one = *old_id_to_new_id.get(&old_id_one).unwrap();
            let new_id_two = *old_id_to_new_id.get(&old_id_two).unwrap();
            let pair = (new_id_one, new_id_two);
            if let Some(total_count) = modified_counts.get_mut(&pair) {
                total_count.add_assign(count);
            } else {
                modified_counts.insert(pair, count);
            }
        }
        let counts = modified_counts;

        Self {
            ids,
            metadata,
            counts,
        }
    }

    fn load_from_csv<R: std::io::Read>(file: R) -> anyhow::Result<Self> {
        let mut ids = HashMap::new();
        let mut metadata_map = HashMap::new();
        let mut counts = HashMap::new();

        for parts in iter_csv_rows(file) {
            if parts.len() == 6 {
                let card_name = parse_card_name(&parts[0]);
                let card_id = parts[1].trim().parse::<usize>().unwrap();
                let card_types = CardTypes::parse_from_characters(parts[2].trim());
                let color_identity = 
                    ColorIdentity::parse_from_characters(parts[3].trim());
                let cmc = parts[4].trim().parse::<usize>().unwrap() as u8;
                let price_cents = parts[5].trim().parse::<usize>().unwrap() as u16;
                let metadata = CardMetadata {
                    card_types,
                    color_identity,
                    cmc,
                    price_cents,
                };
                ids.insert(card_name.to_string(), card_id);
                metadata_map.insert(card_id, metadata);
            } else if parts.len() == 3 {
                let parts: Vec<usize> 
                           = parts.into_iter()
                             .map(|x| str::parse::<usize>(x.trim()).unwrap())
                             .collect();
                let id_one = parts[0];
                let id_two = parts[1];
                let count = parts[2];
                counts.insert((id_one, id_two), count);
            }
        }
        let mut metadata = vec![None; metadata_map.len()];
        // Transform the metadata hash-map into a plain old vector.
        for (id, m) in metadata_map.drain() {
            metadata[id] = Some(m);
        }
        let metadata = metadata.into_iter()
                               .map(|x| x.unwrap())
                               .collect();
        Ok(Self {
            metadata,
            ids,
            counts,
        })
    }
    fn print(&self) {
        for (name, card_id) in self.ids.iter() {
            let metadata = self.metadata[*card_id];
            let card_types = metadata.card_types.as_characters();
            let color_identity = metadata.color_identity.as_characters();
            let cmc = metadata.cmc;
            let price_cents = metadata.price_cents;
            println!("{}, {}, {}, {}, {}, {}", 
                    format_card_name(name),
                    card_id,
                    card_types,
                    color_identity,
                    cmc,
                    price_cents);
        }
        for ((id_one, id_two), count) in self.counts.iter() {
            println!("{}, {}, {}", id_one, id_two, count);
        }
    }
}

struct CardIncidenceStats {
    ids: HashMap<String, usize>,
    next_id: usize,
    // Mapping is from two card name-ids to the count of their
    // co-occurrences among decks, where the card names are
    // sorted lexicographically relative to each other.
    // Note that the diagonal contains the absolute count
    // of the number of decks which contain a particular card.
    //
    // This is deliberately sparse instead of dense.
    counts: HashMap<(usize, usize), usize>,
}

struct CardsMetadata {
    // For double-faced cards, the card types in the metadata
    // will actually be the union of all of the archetypes
    // on each face.
    cards: HashMap<String, CardMetadata>,
    // Map which relates names of individual faces of
    // modal-double-faced cards to their combined
    // name. Needed because raw incidence stats
    // may only wind up mentioning one side
    // of any given card.
    double_faced_cards: HashMap<String, String>,
}

impl CardsMetadata {
    fn load_from_csv<R: std::io::Read>(file: R) -> anyhow::Result<Self> {
        let mut cards = HashMap::new();
        let mut double_faced_cards = HashMap::new();
        for parts in iter_csv_rows(file) {
            if parts.len() == 5 {
                let card_name = parse_card_name(&parts[0]);
                let card_types = CardTypes::parse_from_characters(parts[1].trim());
                let color_identity = 
                    ColorIdentity::parse_from_characters(parts[2].trim());
                let cmc = parts[3].trim().parse::<usize>().unwrap() as u8;
                let price_cents = parts[4].trim().parse::<usize>().unwrap() as u16;
                let metadata = CardMetadata {
                    card_types,
                    color_identity,
                    cmc,
                    price_cents,
                };
                cards.insert(card_name.to_string(), metadata);
            } else if parts.len() == 2 {
                let face_name = parse_card_name(&parts[0]);
                let card_name = parse_card_name(&parts[1]);
                double_faced_cards.insert(face_name, card_name);
            }
        }
        Ok(CardsMetadata {
            cards,
            double_faced_cards,
        })
    }
    fn print_card_names(&self) {
        for (card_name, _) in self.cards.iter() {
            println!("{}", card_name);
        }
        for (face_name, _) in self.double_faced_cards.iter() {
            println!("{}", face_name);
        }
    }
    fn print(&self) {
        for (card_name, metadata) in self.cards.iter() {
            let card_types = metadata.card_types.as_characters();
            let color_identity = metadata.color_identity.as_characters();
            let cmc = metadata.cmc;
            let price_cents = metadata.price_cents;
            println!("{}, {}, {}, {}, {}", 
            format_card_name(card_name), card_types, color_identity, cmc,
                             price_cents);
        }
        for (face_name, card_name) in self.double_faced_cards.iter() {
            println!("{}, {}",
            format_card_name(face_name), format_card_name(card_name));
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, EnumIter, Eq, PartialEq, Hash, Debug)]
enum CardType {
    Legendary = 1,
    Artifact = 2,
    Creature = 4,
    Enchantment = 8,
    Instant = 16,
    Land = 32,
    Planeswalker = 64,
    Sorcery = 128,
}

impl CardType {
    pub fn as_character(&self) -> char {
        match self {
            Self::Legendary => 'L',
            Self::Artifact => 'a',
            Self::Creature => 'c',
            Self::Enchantment => 'e',
            Self::Instant => 'i',
            Self::Land => 'l',
            Self::Planeswalker => 'p',
            Self::Sorcery => 's',
        }
    }
    pub fn parse_from_character(character: char) -> Option<Self> {
        match character {
            'L' => Some(Self::Legendary),
            'a' => Some(Self::Artifact),
            'c' => Some(Self::Creature),
            'e' => Some(Self::Enchantment),
            'i' => Some(Self::Instant),
            'l' => Some(Self::Land),
            'p' => Some(Self::Planeswalker),
            's' => Some(Self::Sorcery),
            _ => None,
        }
    }
}

#[derive(Default)]
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
struct CardTypes {
    bitfield: u8,
}

impl CardTypes {
    pub fn is_possibly_valid_commander(&self) -> bool {
        self.has_type(CardType::Legendary) &&
        (
            self.has_type(CardType::Planeswalker) ||
            self.has_type(CardType::Creature)
        )
    }
    pub fn has_type(&self, card_type: CardType) -> bool {
        let bitmask: u8 = card_type as u8;
        (self.bitfield & bitmask) > 0
    }
    pub fn fits_within(&self, other: CardTypes) -> bool {
        (self.bitfield | other.bitfield) == other.bitfield
    }
    pub fn add_type(&mut self, card_type: CardType) {
        let bitmask: u8 = card_type as u8;
        self.bitfield |= bitmask;
    }
    pub fn parse_from_characters(text: &str) -> Self {
        let mut result = CardTypes::default();
        for character in text.chars() {
            if let Some(card_type) = CardType::parse_from_character(character) {
                result.add_type(card_type);
            }
        }
        result
    }
    pub fn as_characters(&self) -> String {
        let mut result = "".to_string();
        for card_type in CardType::iter() {
            if self.has_type(card_type) {
                result.push(card_type.as_character());
            }
        }
        result
    }
    pub fn parse_from_type_line(type_line: &str) -> Self {
        let mut result = CardTypes::default();
        for type_string in type_line.trim().split_whitespace() {
            match type_string {
                "Legendary" => result.add_type(CardType::Legendary),
                "Artifact" => result.add_type(CardType::Artifact),
                "Creature" => result.add_type(CardType::Creature),
                "Enchantment" => result.add_type(CardType::Enchantment),
                "Instant" => result.add_type(CardType::Instant),
                "Land" => result.add_type(CardType::Land),
                "Planeswalker" => result.add_type(CardType::Planeswalker),
                "Sorcery" => result.add_type(CardType::Sorcery),
                _ => {
                    // TODO: Should we log this?
                },
            };
        }
        result
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Eq, EnumIter, PartialEq, Hash, Debug)]
enum Color {
    Black = 1,
    Blue = 2,
    Green = 4,
    Red = 8,
    White = 16,
}

impl Color {
    pub fn as_character(&self) -> char {
        match self {
            Self::Black => 'B',
            Self::Blue => 'U',
            Self::Green => 'G',
            Self::Red => 'R',
            Self::White => 'W',
        }
    }
    pub fn parse_from_character(text: char) -> Option<Self> {
        match text {
            'B' => Some(Self::Black),
            'U' => Some(Self::Blue),
            'G' => Some(Self::Green),
            'R' => Some(Self::Red),
            'W' => Some(Self::White),
            _ => None,
        }
    }
}

struct ColorCounts {
    black: usize,
    blue: usize,
    green: usize,
    red: usize,
    white: usize,
}

impl ColorCounts {
    fn new() -> Self {
        Self {
            black: 0,
            blue: 0,
            green: 0,
            red: 0,
            white: 0,
        }
    }
    fn add(&mut self, color_identity: ColorIdentity) {
        self.black += ((color_identity.bitfield & 1u8) > 0) as usize;
        self.blue += ((color_identity.bitfield & 2u8) > 0) as usize;
        self.green += ((color_identity.bitfield & 4u8) > 0) as usize;
        self.red += ((color_identity.bitfield & 8u8) > 0) as usize;
        self.white += ((color_identity.bitfield & 16u8) > 0) as usize;
    }
    fn print_basic_lands(&self) {
        if self.black > 0 {
            println!("{} Swamp", self.black);
        }
        if self.blue > 0 {
            println!("{} Island", self.blue);
        }
        if self.green > 0 {
            println!("{} Forest", self.green);
        }
        if self.red > 0 {
            println!("{} Mountain", self.red);
        }
        if self.white > 0 {
            println!("{} Plains", self.white);
        }
    }
    // Assumes that this is the color-counts of non-land-cards,
    // and that the second argument is the color-counts of lands
    fn get_colors_required_for_basics(&self, other_lands_counts: ColorCounts,
                                      num_basic_lands: usize) 
            -> Option<ColorCounts> {
        let black = self.black.saturating_sub(other_lands_counts.black);
        let blue = self.blue.saturating_sub(other_lands_counts.blue);
        let green = self.green.saturating_sub(other_lands_counts.green);
        let red = self.red.saturating_sub(other_lands_counts.red);
        let white = self.white.saturating_sub(other_lands_counts.white);
        let total = black + blue + green + red + white;
        if total == 0 {
            return None;
        }
        let total = total as f64;
        
        let scale = |value: usize| {
            (((value as f64) / total) * (num_basic_lands as f64)).ceil() as usize
        };
        let mut black = scale(black);
        let mut blue = scale(blue);
        let mut green = scale(green);
        let mut red = scale(red);
        let mut white = scale(white);

        let mut result_total = black + blue + green + red + white;
        if result_total == 0 {
            // Punt 
            return None;
        }
        
        // Get to `num_basic_lands` if we went over
        while result_total > num_basic_lands {
            if black > 0 {
                black = black.saturating_sub(1);
            } else if blue > 0 {
                blue = blue.saturating_sub(1);
            } else if green > 0 {
                green = green.saturating_sub(1);
            } else if red > 0 {
                red = red.saturating_sub(1);
            } else if white > 0 {
                white = white.saturating_sub(1);
            }
            result_total -= 1;
        }

        Some(Self {
            black,
            blue,
            green,
            red,
            white,
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
struct ColorIdentity {
    bitfield: u8,
}

impl ColorIdentity {
    pub fn new() -> Self {
        Self {
            bitfield: 0u8,
        }
    }
    pub fn number_of_colors(&self) -> usize {
        let mut result = 0;
        for color in Color::iter() {
            if self.has_color(color) {
                result += 1;
            }
        }
        result
    }
    pub fn has_color(&self, color: Color) -> bool {
        let bitmask: u8 = color as u8;
        (self.bitfield & bitmask) > 0
    }
    pub fn add_color(&mut self, color: Color) {
        let bitmask: u8 = color as u8;
        self.bitfield |= bitmask;
    }
    pub fn fits_within(&self, other: ColorIdentity) -> bool {
        (self.bitfield | other.bitfield) == other.bitfield
    }
    pub fn as_characters(&self) -> String {
        let mut result = "".to_string();
        for color in Color::iter() {
            if self.has_color(color) {
                result.push(color.as_character());
            }
        }
        result
    }
    pub fn parse_from_characters(text: &str) -> Self {
        let mut result = ColorIdentity::new();
        for character in text.chars() {
            if let Some(color) = Color::parse_from_character(character) {
                result.add_color(color);
            }
        }
        result
    }
}

enum CardFilter {
    TypeContains(CardTypes),
    // Units: cents
    InPriceRange(u16, u16),
    CmcInRange(u8, u8),
    Named(String),
}

impl CardFilter {
    pub fn parse(text: &str) -> Self {
        // First, split off the operator from the arguments
        let (operator, remainder) = text.split_once(" ").unwrap();
        match operator {
            "named" => {
                let name = parse_card_name(remainder);
                Self::Named(name)
            },
            "type_contains" => {
                let card_types = CardTypes::parse_from_characters(remainder);
                Self::TypeContains(card_types)
            },
            "cmc_in_range" => {
                let (lower_bound, upper_bound) = remainder.split_once('-').unwrap();     
                let lower_bound = str::parse::<usize>(lower_bound).unwrap() as u8;
                let upper_bound = str::parse::<usize>(upper_bound).unwrap() as u8;
                Self::CmcInRange(lower_bound, upper_bound) 
            },
            "in_price_range" => {
                let (lower_bound, upper_bound) = remainder.split_once('-').unwrap();
                let lower_bound = str::parse::<f64>(lower_bound).unwrap();
                let upper_bound = str::parse::<f64>(upper_bound).unwrap();
                let lower_bound = (lower_bound * 100.0) as u16;
                let upper_bound = (upper_bound * 100.0) as u16;
                Self::InPriceRange(lower_bound, upper_bound)
            },
            _ => {
                panic!("Unknown operator: {}", operator);
            },
        }
    }
    pub fn matches(&self, name: &str, metadata: &CardMetadata) -> bool {
        match self {
            Self::TypeContains(t) => {
                t.fits_within(metadata.card_types)
            },
            Self::InPriceRange(l, h) => {
                *l <= metadata.price_cents &&
                metadata.price_cents <= *h
            },
            Self::CmcInRange(l, h) => {
                *l <= metadata.cmc &&
                metadata.cmc <= *h
            },
            Self::Named(s) => {
                s == name
            }
        }
    }
    pub fn land() -> CardFilter {
        Self::TypeContains(CardTypes::parse_from_characters("l"))
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct CardMetadata {
    card_types: CardTypes,
    color_identity: ColorIdentity,
    cmc: u8,
    price_cents: u16,
}

struct CsvRowIterator<R: std::io::Read> {
    reader: std::io::Lines<std::io::BufReader<R>>,
}

impl<R: std::io::Read> Iterator for CsvRowIterator<R> {
    type Item = Vec<String>;
    fn next(&mut self) -> Option<Self::Item> {
        // Scan the line to find all places which have non-quote-enclosed
        // comma characters.
        let mut parts: Vec<String> = Vec::new();
        let mut current_part: Vec<char> = Vec::new();
        let mut inside_quotes = false;

        while let Some(line) = self.reader.next() {
            let line = line.unwrap();
            for current_char in line.chars() {
                if current_char == ',' && !inside_quotes {
                    let part_string = current_part.iter().collect();
                    parts.push(part_string);
                    current_part.clear();
                } else {
                    if current_char == '\"' {
                        inside_quotes = !inside_quotes;
                    }
                    current_part.push(current_char);
                }
            }
            // If, by the end of the line, we're not inside quotes, return
            if !inside_quotes {
                let part_string = current_part.iter().collect();
                parts.push(part_string);
                current_part.clear();
                return Some(parts);
            } else {
                // Still inside quotes, continue, but make sure to add in
                // the newline
                current_part.push('\n');
            }
        }
        //If we're out of lines, yield none
        None
    }
}

fn iter_csv_rows<R: std::io::Read>(file: R) -> impl Iterator<Item = Vec<String>> {
    let reader = std::io::BufReader::new(file);
    CsvRowIterator {
        reader: reader.lines(),
    }
}

fn card_listing_from_file<R: std::io::Read>(file: R) -> anyhow::Result<DeckList> {
    let reader = std::io::BufReader::new(file);
    let mut result = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        result.push(line);
    }
    Ok(DeckList::new(result))
}

impl CardIncidenceStats {
    fn new() -> Self {
        Self {
            ids: HashMap::new(),
            next_id: 0,
            counts: HashMap::new(),
        }
    }
    fn load_from_csv<R: std::io::Read>(file: R) -> anyhow::Result<Self> {
        let mut ids = HashMap::new();
        let mut counts = HashMap::new();
        let mut next_id = 0;

        for parts in iter_csv_rows(file) {
            // Now we have all of the comma-separated parts, dispatch
            // on the parts
            if parts.len() == 2 {
                let name = parse_card_name(&parts[0]);
                let id = str::parse::<usize>(parts[1].trim())?;
                next_id = std::cmp::max(next_id, id + 1);
                ids.insert(name.to_owned(), id);
            } else if parts.len() == 3 {
                let parts: Vec<usize> 
                               = parts.into_iter()
                                 .map(|x| str::parse::<usize>(x.trim()).unwrap())
                                 .collect();
                let id_one = parts[0];
                let id_two = parts[1];
                let count = parts[2];
                counts.insert((id_one, id_two), count);
            } else {
                eprintln!("Odd number of parts: {}", parts.len());
            }
        }
        Ok(Self {
            ids,
            next_id,
            counts,
        })
    }

    fn merge(&mut self, mut other: CardIncidenceStats) {
        // We need to convert the other's ids into our ids
        let mut other_id_to_my_id = HashMap::new();
        for (card_name, other_card_id) in &other.ids {
            let my_card_id = self.get_card_id(card_name.clone());
            other_id_to_my_id.insert(other_card_id, my_card_id);
        }
        for ((other_id_one, other_id_two), other_count) in other.counts.drain() {
            let my_id_one = *other_id_to_my_id.get(&other_id_one).unwrap();
            let my_id_two = *other_id_to_my_id.get(&other_id_two).unwrap();
            self.add_to_id_pair(my_id_one, my_id_two, other_count);
        }
    }
    fn get_card_id(&mut self, name: String) -> usize {
        if self.ids.contains_key(&name) {
            *self.ids.get(&name).unwrap()
        } else {
            let id = self.next_id;
            self.ids.insert(name, id);
            self.next_id += 1;
            id
        }
    }
    fn set_id_pair(&mut self, first_card_id: usize, second_card_id: usize,
        increment: usize) {
        let card_pair = (first_card_id, second_card_id);
        self.counts.insert(card_pair, increment);
    }
    fn add_to_id_pair(&mut self, first_card_id: usize, second_card_id: usize,
                      increment: usize) {
        let card_pair = (first_card_id, second_card_id);
        if self.counts.contains_key(&card_pair) {
            let previous_count = *self.counts.get(&card_pair).unwrap();
            self.counts.insert(card_pair, previous_count + increment);
        } else {
            self.counts.insert(card_pair, increment);
        }
    }

    fn add_deck_list(&mut self, deck_list: DeckList) {
        let num_cards = deck_list.cards.len();
        for i in 0..num_cards {
            let first_card = deck_list.cards[i].clone();
            let first_card_id = self.get_card_id(first_card);

            for j in i..num_cards {
                let second_card = deck_list.cards[j].clone();
                let second_card_id = self.get_card_id(second_card);
                self.add_to_id_pair(first_card_id, second_card_id, 1);
            }
        }
    }
    fn print(&self) {
        for (name, id) in &self.ids {
            println!("{}, {}", format_card_name(name), id);
        }
        for ((first_card_id, second_card_id), incidence_count) in &self.counts {
            println!("{}, {}, {}", first_card_id, second_card_id, incidence_count);
        }
    }
}

struct DeckList {
    // Cards are sorted lexicographically by name
    cards: Vec<String>,
}

impl DeckList {
    fn new(mut cards: Vec<String>) -> Self {
        cards.sort();
        Self {
            cards,
        }
    }
    fn into_card_name_set(self) -> HashSet<String> {
        let mut result = HashSet::new();
        for card in self.cards.into_iter() {
            result.insert(card);
        }
        result
    }
    /// Update single-face card names to their double-faced names
    fn normalize_against_preprocessed(self, preprocessed: &PreprocessedData)
        -> Self {
        let mut card_face_to_double_faced = HashMap::new();
        for card_name in preprocessed.card_names.iter() {
            if card_name.contains("//") {
                let (side_a, side_b) = card_name.split_once("//").unwrap();
                let side_a = side_a.trim();
                let side_b = side_b.trim();
                card_face_to_double_faced.insert(side_a.to_string(), card_name.to_string());
                card_face_to_double_faced.insert(side_b.to_string(), card_name.to_string());
            }
        }
        let mut cards = Vec::new();
        for name in self.cards.into_iter() {
            if let Some(double_faced_name) = card_face_to_double_faced.get(&name) {
                cards.push(double_faced_name.to_string());
            } else {
                cards.push(name);
            }
        }
        Self {
            cards,
        }
    }
}

fn card_incidence_stats_from_deckstobeat(folder_path: &str) ->
    anyhow::Result<CardIncidenceStats> {
    let mut card_incidence_stats = CardIncidenceStats::new();

    for filename in ["historic decks.csv", "legacy decks.csv",
                     "modern decks.csv", "pioneer decks.csv",
                     "standard decks.csv"] {
        let file_path_buf = std::path::Path::new(folder_path);
        let file_path_buf = file_path_buf.join(filename);
        let file_path = file_path_buf.as_path();
        
        let file = std::fs::File::open(file_path)?;
        // Skip the first row of headers
        for csv_row in iter_csv_rows(file).skip(1) {
            let deck_list_string = csv_row[6].trim().trim_matches('\"');
            let main_deck_list_string = match deck_list_string.split_once("Sideboard") {
                Some((a, _)) => a,
                None => {
                    // Some formats don't have sideboards
                    deck_list_string
                },
            };
            let mut deck_list = Vec::new();
            for card_line in main_deck_list_string.lines() {
                let card_line = card_line.trim();
                let (_, card_name) = card_line.split_once(" ").unwrap();
                let card_name = standardize_card_name(card_name);
                deck_list.push(card_name);
            }
            let deck_list = DeckList::new(deck_list);
            card_incidence_stats.add_deck_list(deck_list);
        }
    }
    Ok(card_incidence_stats)
}

fn card_incidence_stats_from_protour(filename: &str) ->
    anyhow::Result<CardIncidenceStats> {
    let file = std::fs::File::open(filename)?;

    let mut card_incidence_stats = CardIncidenceStats::new();

    let mut current_pilot = "Pilot 1".to_string();
    let mut deck_list = Vec::new();

    // We skip the first row of headers
    for csv_row in iter_csv_rows(file).skip(1) {
        let card = parse_card_name(&csv_row[0]);
        let pilot = csv_row[2].trim();
        let mainboard = csv_row[6].trim();
        if pilot != current_pilot {
            // The pilot has changed since the last one,
            // so we're dealing with a different deck now.
            // Dump the current card contents.
            let mut previous_deck_list = Vec::new();
            std::mem::swap(&mut deck_list, &mut previous_deck_list);
            let previous_deck_list = DeckList::new(previous_deck_list);
            card_incidence_stats.add_deck_list(previous_deck_list);
            current_pilot = pilot.to_owned();
        }
        if mainboard == "Mainboard" {
            deck_list.push(card.to_owned());
        }
    }
    Ok(card_incidence_stats)
}

fn card_incidence_stats_from_mtgtop8(base_directory_name: &str) -> 
    anyhow::Result<CardIncidenceStats> {
    let events_directory_name = format!("{}/events", base_directory_name);
    let events_directory = std::path::Path::new(&events_directory_name);

    let mut card_incidence_stats = CardIncidenceStats::new();

    for event_directory in std::fs::read_dir(events_directory)? {
        let event_directory = event_directory?;
        let mut decks_directory = event_directory.path();
        // Check whether/not there's a "players_decks" subdirectory
        decks_directory.push("players_decks");
        let maybe_decks_directory_contents = std::fs::read_dir(decks_directory);
        if let Ok(decks_directory_contents) = maybe_decks_directory_contents {
            for deck_json_path in decks_directory_contents {
                let deck_json_path = deck_json_path?;
                let deck_json_path = deck_json_path.path();
                let deck_json = std::fs::read_to_string(deck_json_path)?;
                if let Ok(mut deck_json) = json::parse(&deck_json) {
                    let mut deck_list = Vec::new();
                    for deck_component in ["main_deck"] {
                        if deck_json.has_key(deck_component) {
                            let deck_array = deck_json.remove(deck_component);
                            if let json::JsonValue::Array(deck_array) = deck_array {
                                for mut card_entry in deck_array {
                                    let _card_count = card_entry.pop();
                                    let mut card_name = card_entry.pop();
                                    let card_name = card_name.take_string().unwrap();
                                    let card_name = standardize_card_name(&card_name);
                                    deck_list.push(card_name);
                                }
                            }
                        }
                    }
                    let deck_list = DeckList::new(deck_list);
                    card_incidence_stats.add_deck_list(deck_list);
                }
            }
        }
    }
    Ok(card_incidence_stats)
}

fn card_incidence_stats_from_edhrec(base_directory_name: &str) ->
    anyhow::Result<CardIncidenceStats> {
    let cards_directory = std::path::Path::new(base_directory_name);

    let mut card_incidence_stats = CardIncidenceStats::new();

    for card_json_file in std::fs::read_dir(cards_directory)? {
        let card_json_file = card_json_file?;
        let card_json_file = card_json_file.path();
        let card_json = std::fs::read_to_string(card_json_file.clone())?;
        if let Ok(mut card_json) = json::parse(&card_json) {
            let mut header = card_json.remove("header");
            let header = header.take_string().unwrap();
            let card_name = header.replace(" (Card)", "");
            let card_list = card_json.remove("cardlist");

            let card_id = card_incidence_stats.get_card_id(card_name.clone());
            
            if let json::JsonValue::Array(mut card_list) = card_list {
                let mut total_decks_card_is_in = 0;
                for mut other_card_json in card_list.drain(..) {
                    let mut other_card_name = other_card_json.remove("name");
                    let other_card_name = other_card_name.take_string().unwrap();

                    let other_card_id = card_incidence_stats.get_card_id(other_card_name.clone());

                    let num_decks = other_card_json.remove("num_decks");
                    let num_decks = num_decks.as_usize().unwrap();

                    let potential_decks = other_card_json.remove("potential_decks");
                    let potential_decks = potential_decks.as_usize().unwrap();

                    // Inevitably, there will be some card which is colorless
                    // in this list. 
                    total_decks_card_is_in = total_decks_card_is_in.max(potential_decks);

                    // Add what we learned to the incidence stats
                    if card_name <= other_card_name {
                        card_incidence_stats.set_id_pair(card_id, other_card_id, num_decks);
                    } else {
                        card_incidence_stats.set_id_pair(other_card_id, card_id, num_decks);
                    }
                }
                card_incidence_stats.set_id_pair(card_id, card_id, total_decks_card_is_in);
            }
        }
    }
    Ok(card_incidence_stats)
}

fn cards_metadata_from_scryfall(oracle_cards_json_filename: &str) ->
    anyhow::Result<CardsMetadata> {
    let mut double_faced_cards = HashMap::new();
    let mut result = HashMap::new();
    let oracle_cards_json_path = std::path::Path::new(&oracle_cards_json_filename);
    let oracle_cards_json = std::fs::read_to_string(oracle_cards_json_path)?;
    let oracle_cards_json = json::parse(&oracle_cards_json)?;
    if let json::JsonValue::Array(listing) = oracle_cards_json {
        for object in listing.into_iter() {
            if let json::JsonValue::Object(mut object) = object {
                let maybe_name = object.remove("name");
                let maybe_type_line = object.remove("type_line");
                let maybe_color_identity = object.remove("color_identity");
                let maybe_cmc = object.remove("cmc");
                // Determine whether/not the card is legal in commander.
                // If not, throw it out.
                // Should be a json object
                let maybe_legalities = object.remove("legalities");
                if maybe_name.is_none() ||
                    maybe_legalities.is_none() ||
                    maybe_type_line.is_none() {
                    continue;
                }
                let legalities = maybe_legalities.unwrap();
                if let json::JsonValue::Object(mut legalities) = legalities {
                    let maybe_commander_legality = legalities.remove("commander");
                    if maybe_commander_legality.is_none() {
                        continue;
                    }
                    let mut commander_legality = maybe_commander_legality.unwrap();
                    let commander_legality = commander_legality.take_string().unwrap();
                    if commander_legality == "not_legal" {
                        continue;
                    }
                } else {
                    panic!("Legalities is not an object?");
                }
                let name = maybe_name.unwrap();
                let name = name.as_str().unwrap();
                let type_line = maybe_type_line.unwrap();
                let type_line = type_line.as_str().unwrap();

                let card_types = CardTypes::parse_from_type_line(type_line);

                let mut color_identity = ColorIdentity::new();
                if let Some(color_identity_array) = maybe_color_identity {
                    if let json::JsonValue::Array(color_identity_array) = color_identity_array {
                        for color_object in color_identity_array.into_iter() {
                            let color_string = color_object.as_str().unwrap();
                            let color_char = color_string.chars().next().unwrap();
                            let color = Color::parse_from_character(color_char).unwrap();
                            color_identity.add_color(color);
                        }
                    } else {
                        panic!("Color identity is not an array?");
                    }
                }
                let mut cmc = 0u8;
                if let Some(cmc_float) = maybe_cmc {
                    cmc = cmc_float.as_f32().unwrap() as u8;
                }
                // Attempt to obtain the price in cents - default is 0,
                // indicating not defined.
                let mut price_cents: u16 = 0;
                if let Some(json::JsonValue::Object(mut prices_object)) = 
                    object.remove("prices") {
                    let price_usd = prices_object.remove("usd").unwrap();
                    if !price_usd.is_null() {
                        let price_string = price_usd.as_str().unwrap();
                        let price_float = f64::from_str(&price_string).unwrap();
                        let price_float = price_float * 100.0;
                        // Saturating cast to a u16 range
                        price_cents = price_float as u16;
                    }
                }


                let metadata = CardMetadata {
                    card_types,
                    color_identity,
                    cmc,
                    price_cents,
                };
                result.insert(name.to_string(), metadata);

                // Now, check if it's a double-faced card. If so,
                // add data about its faces' names.
                if let Some(json::JsonValue::Array(faces)) 
                    = object.remove("card_faces") {
                    for face in faces {
                        if let json::JsonValue::Object(mut face) = face {
                            if let Some(face_name) = face.remove("name") {
                                let face_name = face_name.as_str().unwrap().to_string();
                                double_faced_cards.insert(face_name, name.to_string());
                            }
                        }
                    }
                }
            }
        }
    } else {
        panic!("Expected an array of card data");
    }
    Ok(CardsMetadata {
        cards: result,
        double_faced_cards,
    })
}


fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        print_usage();
    }
    match args[1].as_str() {
        "card_incidence_stats" => {
            let incidence_stats = match args[2].as_str() {
                "mtgtop8" => {
                    card_incidence_stats_from_mtgtop8(&args[3])?                
                },
                "protour" => {
                    card_incidence_stats_from_protour(&args[3])?
                },
                "deckstobeat" => {
                    card_incidence_stats_from_deckstobeat(&args[3])?
                },
                "edhrec" => {
                    card_incidence_stats_from_edhrec(&args[3])?
                },
                _ => {
                    print_usage();
                },
            };
            incidence_stats.print();
        }
        "merge_incidence_stats" => {
            let filename_one = &args[2];
            let filename_two = &args[3];
            let file_one = std::fs::File::open(filename_one)?;
            let file_two = std::fs::File::open(filename_two)?;
            let mut incidence_stats_one = CardIncidenceStats::load_from_csv(file_one)?;
            let incidence_stats_two = CardIncidenceStats::load_from_csv(file_two)?;
            incidence_stats_one.merge(incidence_stats_two);
            incidence_stats_one.print();
        },
        "preprocess" => {
            let combined_data_csv = &args[2];
            let destination = &args[3];
            let combined_data_csv = std::fs::File::open(combined_data_csv)?;
            let combined_data = CombinedData::load_from_csv(combined_data_csv)?;
            let preprocessed_data = combined_data.preprocess();
            preprocessed_data.write_to_path(destination)?;
        },
        "merge_preprocessed_trusted_with_untrusted" => {
            let trusted_filename = &args[2];
            let untrusted_filename = &args[3];
            let destination = &args[4];
            let trusted_preprocessed = PreprocessedData::load_from_path(trusted_filename)?;
            let untrusted_preprocessed = PreprocessedData::load_from_path(untrusted_filename)?;
            let merged = trusted_preprocessed.merge_with_untrusted(untrusted_preprocessed);
            merged.write_to_path(destination)?;
        },
        "filter_preprocessed" => {
            let preprocessed_data = &args[2];
            let card_list = &args[3];
            let destination = &args[4];
            let card_list = std::fs::File::open(card_list)?;
            let preprocessed_data = PreprocessedData::load_from_path(preprocessed_data)?;
            let card_list = card_listing_from_file(card_list)?;
            let card_list = card_list.normalize_against_preprocessed(&preprocessed_data);
            let card_name_set = card_list.into_card_name_set();
            let preprocessed_data = preprocessed_data.filter(|card_name, _| {
                card_name_set.contains(card_name)
            });
            preprocessed_data.write_to_path(destination)?;
        },
        "combine_incidence_stats_with_metadata" => {
            let incidence_stats_csv = &args[2];
            let metadata_csv = &args[3];
            let incidence_stats_csv = std::fs::File::open(incidence_stats_csv)?;
            let metadata_csv = std::fs::File::open(metadata_csv)?;
            let incidence_stats_csv = CardIncidenceStats::load_from_csv(incidence_stats_csv)?;
            let metadata_csv = CardsMetadata::load_from_csv(metadata_csv)?;
            let combined = CombinedData::from_combination(incidence_stats_csv, metadata_csv);
            combined.print();
        },
        "suggest_card_purchases" => {
            let preprocessed_data = &args[2];
            let preprocessed_data = PreprocessedData::load_from_path(preprocessed_data)?;
            let card_list = &args[3];
            let card_list = std::fs::File::open(card_list)?;
            let card_list = card_listing_from_file(card_list)?;
            let card_list = card_list.normalize_against_preprocessed(&preprocessed_data);
            preprocessed_data.suggest_card_purchases(card_list);
        },
        "card_metadata" => {
            let scryfall_oracle_cards_file = &args[2];
            let cards_metadata = cards_metadata_from_scryfall(scryfall_oracle_cards_file)?;
            cards_metadata.print();
        },
        "card_names_from_metadata" => {
            let metadata_csv = &args[2];
            let metadata_csv = std::fs::File::open(metadata_csv)?;
            let metadata_csv = CardsMetadata::load_from_csv(metadata_csv)?;
            metadata_csv.print_card_names();
        },
        "complete" => {
            let method = args[2].as_str();
            let preprocessed_data = &args[3];
            let destination = &args[4];
            let preprocessed_data = PreprocessedData::load_from_path(preprocessed_data)?;
            let preprocessed_data = match method {
                "svt" => {
                    preprocessed_data.complete_svt()
                },
                "soft-impute" => {
                    preprocessed_data.complete_soft_impute()
                },
                "rank-one-pursuit" => {
                    preprocessed_data.complete_rank_one_pursuit()
                },
                _ => panic!("Unrecognized completion method - pick svt or alm"),
            };
            preprocessed_data.write_to_path(destination)?;
        },
        "rank_commanders" => {
            let preprocessed_data = &args[2];
            let preprocessed_data = PreprocessedData::load_from_path(preprocessed_data)?;
            preprocessed_data.rank_commanders();
        },
        "build_commander_deck" => {
            let preprocessed_data = &args[2];
            let preprocessed_data = PreprocessedData::load_from_path(preprocessed_data)?;
            let constraints_file = &args[3];
            let constraints_file = std::fs::File::open(constraints_file)?;
            let constraints = Constraints::load_from_file(constraints_file)?;
            let commander_name = &args[4];
            preprocessed_data.build_commander_deck(constraints, commander_name);
        },
        _ => {
            print_usage();
        }
    }
    Ok(())
}
