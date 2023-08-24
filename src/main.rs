use std::env;
use std::ops::AddAssign;
use std::io::BufRead;
use std::collections::HashMap;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use anyhow::bail;
use ndarray::*;

fn print_usage() -> ! {
    println!("Usage:");
    println!("cargo run card_incidence_stats mtgtop8 [mtg_top_8_data_base_directory]");
    println!("cargo run card_incidence_stats protour [mtg_pro_tour_csv]");
    println!("cargo run card_incidence_stats deckstobeat [deckstobeat_base_directory]");
    println!("cargo run merge_incidence_stats [incidence_csv_one] [incidence_csv_two]");
    println!("cargo run filter_incidence_stats [incidence_stats_csv] [card_list_file]");
    println!("cargo run card_metadata [scryfall_oracle_cards_db_file]");
    println!("cargo run build_commander_deck [incidence_stats_csv]");
    panic!();
}

// TODO: Should make a combined format which includes both card metadata
// and also incidence stats. Remove `filter_incidence_stats`, and instead
// allow this combined format to be filtered. Also create a command which
// allows for merging incidence stats with some metadata.

// TODO: should make a command which takes combined stats and yields
// an ordered listing of all commanders by their synergy totals
// among cards within their color identity.

// TODO: internal functionality to filter cards to be within specific
// color identity.

// TODO: Make a command which takes in a commander's name, fuzzy-matches
// that to the closest name among included cards, filters the card
// DB to its color identity, and then does a trial-solve of a constraint-based
// program for the entire deck.

// TODO: should define a query language for filtering cards according to criteria

// TODO: should define a file-format for "deck-building rules"

/// Parses a (quoted) card name, including converting escaped characters
/// to ordinary ones.
fn parse_card_name(quoted_card_name: &str) -> String {
    let whitespace_trimmed = quoted_card_name.trim();
    let after_quotes = whitespace_trimmed.strip_prefix('\"').unwrap();
    let inside_quotes = after_quotes.strip_suffix('\"').unwrap();
    inside_quotes.replace("\"\"", "\"")
}

/// Formats a (raw) card name to have quotes around it, and to escape
/// any internal quotes.
fn format_card_name(raw_card_name: &str) -> String {
    let escaped_card_name = raw_card_name.replace("\"", "\"\"");
    format!("\"{}\"", escaped_card_name)
}

struct CardIncidenceStats {
    ids: HashMap<String, usize>,
    next_id: usize,
    // Mapping is from two card name-ids to the count of their
    // co-occurrences among decks, where the card names are
    // sorted lexicographically relative to each other.
    // Note that the diagonal contains the absolute count
    // of the number of decks which contain a particular card.
    counts: HashMap<(usize, usize), usize>,
}

struct CardsMetadata {
    cards: HashMap<String, CardMetadata>,
}

impl CardsMetadata {
    fn load_from_csv<R: std::io::Read>(file: R) -> anyhow::Result<Self> {
        let mut cards = HashMap::new();
        for parts in iter_csv_rows(file) {
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
        }
        Ok(CardsMetadata {
            cards,
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
    pub fn is_valid_commander(&self) -> bool {
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

fn build_commander_deck(card_incidence_stats: CardIncidenceStats) -> anyhow::Result<DeckList> {
    if card_incidence_stats.ids.len() < 60 {
        bail!("There are not enough cards");
    }
    let dim = card_incidence_stats.next_id;
    let mut diagonal_total = 0usize;
    for i in 0..dim {
        let count = card_incidence_stats.counts.get(&(i, i)).unwrap();
        diagonal_total += count;
    }
    // Pack the (normalized) incidence stats into an array.
    // Normalize by the diagonal total, so that the diagonal
    // represents rough probabilities for inclusion of a single
    // card in a random deck.
    let mut similarity_array = Array::<f64, _>::zeros((dim, dim));
    for ((i, j), value) in similarity_array.indexed_iter_mut() {
        if i == j {
            // Diagonal element
            let count = *card_incidence_stats.counts.get(&(i, i)).unwrap();
            value.add_assign((count as f64) / (diagonal_total as f64));
        } else {
            // Non-diagonal element, need to ensure that the matrix is symmetric
            // PSD.
            let mut count = 0usize;
            if let Some(x) = card_incidence_stats.counts.get(&(i, j)) {
                count = *x;
            }
            if let Some(x) = card_incidence_stats.counts.get(&(j, i)) {
                count = *x;
            }
            // Halve the value, since the other side of the diagonal
            // will incorporate the other half 
            value.add_assign(((count as f64) / (diagonal_total as f64)) / 2.0f64);
        }
    }
    panic!();
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

    fn filter(mut self, card_list: &DeckList) -> Self {
        // First, remove all irrelevant entries
        let mut to_new_id = HashMap::new();
        let mut next_id = 0;
        for card in &card_list.cards {
            if let Some(id) = self.ids.get(card) {
                to_new_id.insert(*id, next_id);
                next_id += 1;
            }
        } 
        self.ids.retain(|_, id| {
            to_new_id.contains_key(id)
        });
        self.ids.shrink_to_fit();
        self.counts.retain(|(id_one, id_two), _| {
            to_new_id.contains_key(id_one) &&
            to_new_id.contains_key(id_two)
        });
        self.counts.shrink_to_fit();
        
        //Now, compact the maps, assigning new id's as needed
        let mut ids = HashMap::new();
        for (name, old_id) in self.ids.drain() {
            ids.insert(name, *to_new_id.get(&old_id).unwrap());
        }
        let mut counts = HashMap::new();
        for ((id_one, id_two), count) in self.counts.drain() {
            counts.insert((*to_new_id.get(&id_one).unwrap(),
                           *to_new_id.get(&id_two).unwrap()),
                           count);
        }

        Self {
            ids,
            next_id,
            counts,
        }
    }

    fn merge(&mut self, mut other: CardIncidenceStats) {
        // We need to convert the other's ids into our ids
        let mut other_id_to_my_id = HashMap::new();
        for (card_name, other_card_id) in &other.ids {
            let my_card_id = match self.ids.get(card_name) {
                Some(my_card_id) => {
                   *my_card_id 
                },
                None => {
                    self.get_card_id(card_name.clone())
                },
            };
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
                let card_name = card_name.trim();
                deck_list.push(card_name.to_string());
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
            }
        }
    } else {
        panic!("Expected an array of card data");
    }
    Ok(CardsMetadata {
        cards: result,
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
        "filter_incidence_stats" => {
            let incidence_stats_csv = &args[2];
            let card_list = &args[3];
            let incidence_stats_csv = std::fs::File::open(incidence_stats_csv)?;
            let card_list = std::fs::File::open(card_list)?;
            let incidence_stats_csv = CardIncidenceStats::load_from_csv(incidence_stats_csv)?;
            let card_list = card_listing_from_file(card_list)?;
            let incidence_stats_csv = incidence_stats_csv.filter(&card_list);
            incidence_stats_csv.print();
        },
        "card_metadata" => {
            let scryfall_oracle_cards_file = &args[2];
            let cards_metadata = cards_metadata_from_scryfall(scryfall_oracle_cards_file)?;
            cards_metadata.print();
        },
        "build_commander_deck" => {
            let incidence_stats_csv = &args[2];
            let incidence_stats_csv = std::fs::File::open(incidence_stats_csv)?;
            let incidence_stats_csv = CardIncidenceStats::load_from_csv(incidence_stats_csv)?;
            let deck_list = build_commander_deck(incidence_stats_csv);
        },
        _ => {
            print_usage();
        }
    }
    Ok(())
}
