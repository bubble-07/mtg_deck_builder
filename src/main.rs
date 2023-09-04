use rand::prelude::SliceRandom;
use rand::Rng;
use rand_distr::WeightedIndex;
use rand_distr::Distribution;
use std::env;
use std::ops::AddAssign;
use std::ops::MulAssign;
use std::io::BufRead;
use std::collections::HashMap;
use std::collections::HashSet;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use anyhow::bail;
use ndarray::*;

// TODO: We currently don't understand the "partner" mechanic -
// could we fix that?


// TODO: Parsing the CSVs and computing the derived data
// for combined_data seems like they're the slowest steps.

fn print_usage() -> ! {
    println!("Usage:");
    println!("cargo run card_incidence_stats mtgtop8 [mtg_top_8_data_base_directory]");
    println!("cargo run card_incidence_stats protour [mtg_pro_tour_csv]");
    println!("cargo run card_incidence_stats deckstobeat [deckstobeat_base_directory]");
    println!("cargo run merge_incidence_stats [incidence_csv_one] [incidence_csv_two]");
    println!("cargo run combine_incidence_stats_with_metadata [incidence_csv] [metadata_csv]");
    println!("cargo run card_metadata [scryfall_oracle_cards_db_file]");
    println!("cargo run rank_commanders [combined_data_csv]");
    println!("cargo run build_commander_deck [combined_data_csv] [commander_name]");
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

struct CombinedData {
    ids: HashMap<String, usize>,
    // Indexed by ID
    metadata: Vec<CardMetadata>,
    counts: HashMap<(usize, usize), usize>,
}

struct Constraint {
    filter: MetadataFilter,
    count_lower_bound_inclusive: usize,
    count_upper_bound_inclusive: usize,
}

impl Constraint {
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
    fn default() -> Self {
        let mut constraints = Vec::new();
        
        let land_constraint = Constraint {
            filter: MetadataFilter::land(),
            count_lower_bound_inclusive: 33,
            count_upper_bound_inclusive: 40,
        };

        constraints.push(land_constraint);
        Self {
            constraints
        }
    }
    fn get_actual_counts(&self, active_set_indices: &[usize],
                         all_metadata: &[CardMetadata]) -> Vec<usize> {
        let mut result = Vec::new();
        for constraint in self.constraints.iter() {
            let mut count = 0;
            for index in active_set_indices {
                let metadata = &all_metadata[*index];
                if constraint.filter.matches(metadata) {
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
        all_metadata: &[CardMetadata]) -> Vec<usize> {

        // Copy the previous collection of counts
        let mut result: Vec<usize> = actual_counts.iter().copied().collect();

        let removed_metadata = &all_metadata[removed_index];
        let added_metadata = &all_metadata[added_index];

        for (i, constraint) in self.constraints.iter().enumerate() {
            if constraint.filter.matches(removed_metadata) {
                result[i] -= 1;
            }
            if constraint.filter.matches(added_metadata) {
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
    weighted_index: &'a WeightedIndex<f64>,
    iter_count: usize,
    full_set_size: usize,
    random_choice_iteration_bound: usize,
    all_possible_results_random_order: Option<Vec<(usize, usize)>>,
}

impl<'a> SwapIndexIterator<'a> {
    pub fn new(weighted_index: &'a WeightedIndex<f64>,
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

impl CombinedData {
    pub fn build_commander_deck(self, commander_name: &str) {
        // SETUP

        // Early bail if we can't get the ID
        let commander_id = *self.ids.get(commander_name)
            .expect("No commander was found by that name");

        // Early bail if there's not 100 or more cards
        // in the IDs
        if self.ids.len() < 100 {
            panic!("Not enough cards to build a commander deck");
        }

        // Filter counts, ids, and metadata to only those
        // cards which are within the commander's color identity
        let commander_metadata = self.metadata[commander_id].clone();
        let commander_color_identity = commander_metadata.color_identity;

        let filtered_data = self.filter(move |_, metadata| {
            metadata.color_identity.fits_within(commander_color_identity)
        });

        // Get all counts for the data-set, and all ids-to-names
        let mut score_matrix = filtered_data.get_score_matrix();
        let reverse_id_map = filtered_data.get_reverse_id_map();
        let metadata = filtered_data.metadata;

        let num_cards_in_pool = metadata.len();

        // Now, adjust the scores of the basic lands so that they're
        // smaller than the scores of all other cards
        let mut basic_land_ids = HashSet::new();
        for basic_land_name in BASIC_LAND_NAMES {
            if let Some(id) = filtered_data.ids.get(basic_land_name) {
                basic_land_ids.insert(*id);
            }
        }
        let basic_land_ids = basic_land_ids;

        for basic_land_id in basic_land_ids.iter() {
            let land_color = metadata[*basic_land_id].color_identity;
            for i in 0..num_cards_in_pool {
                let other_color = metadata[i].color_identity;
                let value = if land_color.fits_within(other_color) {
                    0.0000001
                } else {
                    0.0
                };
                score_matrix[[i, *basic_land_id]] = value;
                score_matrix[[*basic_land_id, i]] = value;
            }
        }
       
        // After filtering, the commander's ID will be different, but no biggie.
        let commander_id = *filtered_data.ids.get(commander_name).unwrap();

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

        let score_matrix = score_matrix;

        // For now, use the default constraints
        // TODO: allow customization
        let constraints = Constraints::default();

        // Number of optimization attempts
        // TODO: allow customization?
        let num_optimization_attempts = 5;
 
        // Rationale: ~1/2 the total number of possible choices
        // before we resort to sampling without replacement (more expensive,
        // O(num_cards_in_pool)) to verify that we've truly hit
        // a local maximum.
        let random_choice_iteration_bound = num_cards_in_pool * 10;

        let mut rng = rand::thread_rng();

        // This is something which will ease in computing changes to the objective
        // function quickly
        let mut diffing_matrix = Array::zeros((num_cards_in_pool, num_cards_in_pool));
        for index_removed in 0..num_cards_in_pool {
            for index_added in 0..num_cards_in_pool {
                diffing_matrix[[index_removed, index_added]] =
                        score_matrix[[index_added, index_added]]
                        - 2.0 * score_matrix[[index_added, index_removed]]
                        + score_matrix[[index_removed, index_removed]];
            }
        }
        let diffing_matrix = diffing_matrix;

        let mut weights = Vec::new();
        // Construct the initial set of weighted probabilities which
        // we'll always use to pick the initial set of cards to draw from
        let mut total_commander_synergy = 0.0;
        for i in 0..num_cards_in_pool {
            total_commander_synergy += score_matrix[[commander_id, i]];
        }
        for i in 0..num_cards_in_pool {
            let commander_synergy = score_matrix[[commander_id, i]];
            let commander_synergy_as_probability = commander_synergy / total_commander_synergy;
            // Take that, and average with a uniform distribution
            let uniform_probability = 1.0 / (num_cards_in_pool as f64);
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
            active_set[[commander_id,]] = 1.0f64;

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
            let mut constraint_counts = constraints.get_actual_counts(&active_set_indices, &metadata);
            let mut constraint_count_distance = constraints.count_distance(&constraint_counts);

            // Determine the current objective value
            let mut objective_value = active_set.dot(&score_matrix).dot(&active_set);
            // Determine the active score, which is going to be
            // a vector that we'll update frequently
            let mut active_score = active_set.dot(&score_matrix);
           
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
                        &constraint_counts, index_removed, index_added, &metadata);
                    let updated_constraint_count_distance = constraints.count_distance(&updated_constraint_counts);

                    if constraint_count_distance < updated_constraint_count_distance {
                        // It makes constraint violation worse; bail.
                        continue;
                    }

                    let is_feasible = updated_constraint_count_distance == 0;
                    
                    // Determine what the updated objective value would be.

                    let updated_objective_value = objective_value
                        + 2.0 * (active_score[[index_added,]] - active_score[[index_removed,]])
                        + diffing_matrix[[index_removed, index_added]];

                    if is_feasible && objective_value >= updated_objective_value {
                        // Still feasible, but doesn't make the objective
                        // value any better!
                        continue;
                    }

                    let updated_active_score = &active_score
                        + &score_matrix.row(index_added) - &score_matrix.row(index_removed);

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

        let active_score = best_active_set.dot(&score_matrix);

        // Order `best_active_set_indices` by decreasing order
        // of contribution to the total overall score.
        let mut sorted_indices = Vec::new();
        // Always put the commander at the top of the list
        sorted_indices.push((best_active_set_indices[0], f64::MAX));
        for i in 1..best_active_set_indices.len() {
            sorted_indices.push((best_active_set_indices[i], active_score[[i,]]));
        }
        sorted_indices.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            
        // Print out the solution we found
        // Loop 1: Print out non-lands (required)
        // Loop 2: Print out lands (suggestions, could be swapped for basics)
        println!("Non-Land Cards:");
        for (index, _) in sorted_indices.iter() {
            if !metadata[*index].card_types.has_type(CardType::Land) {
                println!("1 {}", reverse_id_map[*index]);
            }
        }
        println!("Land Cards:");
        for (index, _) in sorted_indices.iter() {
            if metadata[*index].card_types.has_type(CardType::Land) {
                println!("1 {}", reverse_id_map[*index]);
            }
        }
    }

    pub fn rank_commanders(&self) {
        let reverse_id_map = self.get_reverse_id_map();
        let score_matrix = self.get_score_matrix();

        let mut names_and_scores_by_number_of_colors = vec![Vec::new(); 6];

        let n = self.metadata.len();
        for commander_index in 0..n {
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
            for other_index in 0..n {
                let other_metadata = &self.metadata[other_index];
                let other_color_identity = other_metadata.color_identity;
                if other_color_identity.fits_within(color_identity) {
                    let score = score_matrix[[commander_index, other_index]];
                    valid_color_identity_scores.push(score);
                }
            }
            // Sort the scores and extract just the top 100
            // (the rationale being that it's the number of cards
            // in a commander deck, including the commander.)
            valid_color_identity_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
            valid_color_identity_scores.truncate(100);

            let total_score: f64 = valid_color_identity_scores.iter().sum();
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
    pub fn get_score_matrix(&self) -> Array2<f64> {
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
        let dense_counts = self.get_dense_counts();
        let mut dense_scores = dense_counts.mapv(|x| x.ln_1p());
        for ((i, j), value) in dense_scores.indexed_iter_mut() {
            if i != j {
                // Halve off-diagonal values so that when
                // using them in the quadratic program, they sum
                // to the strength scoring for the pair.
                value.mul_assign(0.5);
            }
        }
        // Little bit of normalization to bring ourselves to a reasonable range
        dense_scores * 0.01
    }
    pub fn get_dense_counts(&self) -> Array2<f64> {
        let dim = self.metadata.len();
        // Pack the incidence counts into an array
        let mut similarity_array = Array::<f64, _>::zeros((dim, dim));
        for ((i, j), value) in similarity_array.indexed_iter_mut() {
            if let Some(count) = self.counts.get(&(i, j)) {
                value.add_assign(*count as f64);
            } else if let Some(count) = self.counts.get(&(j, i)) {
                value.add_assign(*count as f64);
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

    fn filter(self, criteria: impl Fn(&str, &CardMetadata) -> bool) -> Self {
        // First, scan through and assign new contiguous indices to all matches
        // We can straightforwardly update the ID mapping and metadata
        // for this in one go.
        let mut next_id = 0;
        let mut old_id_to_new_id = HashMap::new();

        let mut metadata = Vec::new();
        let mut ids = HashMap::new();

        for (card_name, old_id) in self.ids.into_iter() {
            let card_metadata = &self.metadata[old_id];
            if !criteria(&card_name, card_metadata) {
                continue;
            }
            old_id_to_new_id.insert(old_id, next_id);
            metadata.push(card_metadata.clone());
            ids.insert(card_name.to_string(), next_id);
            next_id += 1;
        }
        
        let metadata = metadata;
        let ids = ids;

        // Now, using the old-to-new ID map, update the counts
        let mut counts = HashMap::new();
        for ((old_id_one, old_id_two), count) in self.counts.into_iter() {
            if let Some(new_id_one) = old_id_to_new_id.get(&old_id_one) {
                if let Some(new_id_two) = old_id_to_new_id.get(&old_id_two) {
                    counts.insert((*new_id_one, *new_id_two), count);
                }
            }
        }
        let counts = counts;
        Self {
            metadata,
            ids,
            counts,
        }
    }

    fn load_from_csv<R: std::io::Read>(file: R) -> anyhow::Result<Self> {
        let mut ids = HashMap::new();
        let mut metadata_map = HashMap::new();
        let mut counts = HashMap::new();

        for parts in iter_csv_rows(file) {
            if parts.len() == 5 {
                let card_name = parse_card_name(&parts[0]);
                let card_id = parts[1].trim().parse::<usize>().unwrap();
                let card_types = CardTypes::parse_from_characters(parts[2].trim());
                let color_identity = 
                    ColorIdentity::parse_from_characters(parts[3].trim());
                let cmc = parts[4].trim().parse::<usize>().unwrap() as u8;
                let metadata = CardMetadata {
                    card_types,
                    color_identity,
                    cmc,
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
            println!("{}, {}, {}, {}, {}", 
                    format_card_name(name),
                    card_id,
                    card_types,
                    color_identity,
                    cmc);
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
            if parts.len() == 4 {
                let card_name = parse_card_name(&parts[0]);
                let card_types = CardTypes::parse_from_characters(parts[1].trim());
                let color_identity = 
                    ColorIdentity::parse_from_characters(parts[2].trim());
                let cmc = parts[3].trim().parse::<usize>().unwrap() as u8;
                let metadata = CardMetadata {
                    card_types,
                    color_identity,
                    cmc,
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
    fn print(&self) {
        for (card_name, metadata) in self.cards.iter() {
            let card_types = metadata.card_types.as_characters();
            let color_identity = metadata.color_identity.as_characters();
            let cmc = metadata.cmc;
            println!("{}, {}, {}, {}", 
            format_card_name(card_name), card_types, color_identity, cmc);
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
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
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

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
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

enum MetadataFilter {
    CardTypeContains(CardType),
}

impl MetadataFilter {
    pub fn matches(&self, metadata: &CardMetadata) -> bool {
        match self {
            Self::CardTypeContains(t) => {
                metadata.card_types.has_type(t.clone())
            }
        }
    }
    pub fn land() -> MetadataFilter {
        Self::CardTypeContains(CardType::Land)
    }
}

#[derive(Copy, Clone)]
struct CardMetadata {
    card_types: CardTypes,
    color_identity: ColorIdentity,
    cmc: u8,
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
                    cmc = cmc_float.as_f64().unwrap() as u8;
                }
                let metadata = CardMetadata {
                    card_types,
                    color_identity,
                    cmc,
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
        "filter_combined_data" => {
            let combined_data_csv = &args[2];
            let card_list = &args[3];
            let combined_data_csv = std::fs::File::open(combined_data_csv)?;
            let card_list = std::fs::File::open(card_list)?;
            let combined_data_csv = CombinedData::load_from_csv(combined_data_csv)?;
            let card_list = card_listing_from_file(card_list)?;
            let card_name_set = card_list.into_card_name_set();
            let combined_data_csv = combined_data_csv.filter(|card_name, _| {
                card_name_set.contains(card_name)
            });
            combined_data_csv.print();
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
        "card_metadata" => {
            let scryfall_oracle_cards_file = &args[2];
            let cards_metadata = cards_metadata_from_scryfall(scryfall_oracle_cards_file)?;
            cards_metadata.print();
        },
        "rank_commanders" => {
            let combined_data = &args[2];
            let combined_data = std::fs::File::open(combined_data)?;
            let combined_data = CombinedData::load_from_csv(combined_data)?;
            combined_data.rank_commanders();
        },
        "build_commander_deck" => {
            let combined_data = &args[2];
            let combined_data = std::fs::File::open(combined_data)?;
            let combined_data = CombinedData::load_from_csv(combined_data)?;
            let commander_name = &args[3];
            combined_data.build_commander_deck(commander_name);
        },
        _ => {
            print_usage();
        }
    }
    Ok(())
}
