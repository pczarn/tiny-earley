#![feature(stmt_expr_attributes)]
#![feature(test)]

extern crate test;

mod forest;

use test::Bencher;

use std::{collections::BinaryHeap, ops::Index};
// #[cfg(feature = "debug")]
use std::collections::{BTreeMap, BTreeSet};

use self::forest::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct Symbol(u32);

#[derive(Clone)]
pub struct Grammar<const S: usize> {
    rules: Vec<Rule>,
    start_symbol: Symbol,
    symbol_names: [&'static str; S],
    gen_symbols: u32,
    rule_id: usize,
    lhs: Symbol,
    nulling: BTreeMap<Symbol, usize>,
    rhs_nulling: Vec<NullingEliminated>,
}

#[derive(Clone)]
enum Side {
    Left,
    Right,
    Both,
}

#[derive(Clone)]
struct NullingEliminated {
    rule_id: Option<usize>,
    sym: Symbol,
    side: Side,
}

#[derive(Clone, Copy, Debug)]
struct Rule {
    lhs: Symbol,
    rhs0: Symbol,
    rhs1: Option<Symbol>,
    id: Option<usize>,
}

#[derive(Clone)]
struct Tables<const S: usize> {
    prediction_matrix: [[bool; S]; S],
    start_symbol: Symbol,
    rules: Vec<Rule>,
    rules_by_rhs0: Vec<Rule>,
    completions: Vec<Vec<PredictionTransition>>,
    symbol_names: [&'static str; S],
    gen_completions: Vec<PredictionTransition>,
}

#[derive(Copy, Clone, Debug, Default)]
struct PredictionTransition {
    symbol: Symbol,
    top: Symbol,
    dot: usize,
    is_unary: bool,
}

// Recognizer

#[derive(Clone)]
pub struct Recognizer<const S: usize> {
    tables: Tables<S>,
    earley_chart: EarleyChart<S>,
    complete: BinaryHeap<CompletedItem>,
    pub forest: Forest,
    pub finished_node: Option<NodeHandle>,
}

#[derive(Clone)]
struct EarleyChart<const S: usize> {
    predicted: Vec<[bool; S]>,
    indices: Vec<usize>,
    items: Vec<Item>,
}

impl<const S: usize> EarleyChart<S> {
    fn next_set(&mut self, predicted: Option<[bool; S]>) {
        self.predicted.push(predicted.unwrap_or([false; S]));
        self.indices.push(self.items.len());
    }

    fn new() -> Self {
        EarleyChart {
            predicted: vec![],
            indices: vec![],
            items: vec![],
        }
    }

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn predicted(&self, index: usize) -> &[bool] {
        &self.predicted[index][..]
    }

    fn medial(&mut self) -> &mut Vec<Item> {
        &mut self.items
    }

    fn next_medial(&self) -> &[Item] {
        &self.items[self.indices[self.indices.len() - 1]..]
    }

    fn last(&self) -> EarleySetRef {
        let items = &self.items[self.indices[self.indices.len() - 1]..];
        let predicted = &self.predicted.last().unwrap()[..];
        EarleySetRef { items, predicted }
    }

    fn next_to_last(&self) -> EarleySetRef {
        let items =
            &self.items[self.indices[self.indices.len() - 2]..self.indices[self.indices.len() - 1]];
        let predicted = &self.predicted[self.predicted.len() - 2][..];
        EarleySetRef { items, predicted }
    }

    fn last_mut(&mut self) -> EarleySetMut {
        EarleySetMut {
            items: &mut self.items[self.indices[self.indices.len() - 1]..],
            predicted: &mut self.predicted.last_mut().unwrap()[..],
        }
    }
}

struct EarleySetMut<'a> {
    items: &'a mut [Item],
    predicted: &'a mut [bool],
}

struct EarleySetRef<'a> {
    items: &'a [Item],
    predicted: &'a [bool],
}

impl<const S: usize> Index<usize> for EarleyChart<S> {
    type Output = [Item];
    fn index(&self, index: usize) -> &Self::Output {
        &self.items[self.indices[index]..self.indices[index + 1]]
    }
}

#[derive(Clone)]
struct EarleySet<const S: usize> {
    predicted: [bool; S],
    medial: Vec<Item>,
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Clone)]
struct Item {
    postdot: Symbol,
    dot: usize,
    origin: usize,
    node: NodeHandle,
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
struct CompletedItem {
    origin: usize,
    dot: usize,
    left_node: NodeHandle,
    right_node: Option<NodeHandle>,
}

trait UnionWith {
    fn union_with(&mut self, other: &[bool]);
}

impl<const S: usize> UnionWith for [bool; S] {
    fn union_with(&mut self, other: &[bool]) {
        for (dst, &src) in self.iter_mut().zip(other.iter()) {
            *dst |= src;
        }
    }
}

impl<'a> UnionWith for &'a mut [bool] {
    fn union_with(&mut self, other: &[bool]) {
        for (dst, &src) in self.iter_mut().zip(other.iter()) {
            *dst |= src;
        }
    }
}

impl Symbol {
    fn usize(self) -> usize {
        self.0 as usize
    }
}

impl<const S: usize> EarleySet<S> {
    fn new() -> Self {
        EarleySet {
            predicted: [false; S],
            medial: vec![],
        }
    }
}

impl<const S: usize> Grammar<S> {
    pub fn new(symbol_names: [&'static str; S], start_symbol: usize) -> Self {
        Self {
            rules: vec![],
            start_symbol: Symbol(start_symbol as u32),
            symbol_names,
            gen_symbols: 0,
            rule_id: 2,
            lhs: Symbol(0),
            nulling: BTreeMap::new(),
            rhs_nulling: vec![],
        }
    }

    pub fn symbols(&self) -> [Symbol; S] {
        let mut result = [Symbol(0); S];
        for (i, elem) in result.iter_mut().enumerate() {
            *elem = Symbol(i as u32);
        }
        result
    }

    pub fn rule<const N: usize>(&mut self, lhs: Symbol, rhs: [Symbol; N]) -> &mut Self {
        if N == 0 {
            self.nulling.insert(lhs, self.rule_id);
            self.rule_id += 1;
            self.lhs = lhs;
            return self;
        }
        let mut cur_rhs0 = rhs[0];
        for i in 1 .. N - 1 {
            let gensym = Symbol(self.gen_symbols + S as u32);
            self.gen_symbols += 1;
            self.rules.push(Rule {
                lhs: gensym,
                rhs0: cur_rhs0,
                rhs1: Some(rhs[i]),
                id: None,
            });
            cur_rhs0 = gensym;
        }
        self.rules.push(Rule {
            lhs,
            rhs0: cur_rhs0,
            rhs1: if N == 1 { None } else { Some(rhs[N - 1]) },
            id: Some(self.rule_id),
        });
        self.rule_id += 1;
        self.lhs = lhs;
        self
    }

    fn rhs<const N: usize>(&mut self, rhs: [Symbol; N]) -> &mut Self {
        self.rule(self.lhs, rhs);
        self
    }

    pub fn sort_rules(&mut self) {
        self.rules.sort_by(|a, b| a.lhs.cmp(&b.lhs));
    }

    fn eliminate_nulling(&mut self) {
        let mut new_rules = vec![];
        let mut rules = self.rules.clone();
        let mut change = true;
        while change {
            change = false;
            for rule in &self.rules {
                if let Some(_id) = self.nulling.get(&rule.rhs0) {
                    if let Some(rhs1) = rule.rhs1 {
                        new_rules.push(Rule {
                            lhs: rule.lhs,
                            rhs0: rhs1,
                            rhs1: None,
                            id: rule.id,
                        });
                        if let Some(_id) = self.nulling.get(&rhs1) {
                            self.rhs_nulling.push(NullingEliminated {
                                rule_id: rule.id,
                                side: Side::Both,
                                sym: rule.rhs0,
                            });
                            change |= self
                                .nulling
                                .insert(rule.lhs, rule.id.unwrap_or(0))
                                .is_none();
                        } else {
                            self.rhs_nulling.push(NullingEliminated {
                                rule_id: rule.id,
                                side: Side::Left,
                                sym: rule.rhs0,
                            });
                        }
                    } else {
                        // TODO rule.id
                        self.rhs_nulling.push(NullingEliminated {
                            rule_id: rule.id,
                            side: Side::Left,
                            sym: rule.rhs0,
                        });
                        change |= self
                            .nulling
                            .insert(rule.lhs, rule.id.unwrap_or(0))
                            .is_none();
                    }
                }
                if let Some(rhs1) = rule.rhs1 {
                    if let Some(_id) = self.nulling.get(&rhs1) {
                        self.rhs_nulling.push(NullingEliminated {
                            rule_id: rule.id,
                            side: Side::Right,
                            sym: rhs1,
                        });
                        new_rules.push(Rule {
                            lhs: rule.lhs,
                            rhs0: rule.rhs0,
                            rhs1: None,
                            id: rule.id,
                        });
                    }
                }
            }
        }
        self.rules.extend(new_rules);
    }

    pub fn stringify_to_bnf(&self) -> String {
        use std::fmt::Write;
        let mut result = String::new();
        for (i, rule) in self.rules.iter().enumerate() {
            let tostr = |sym: Symbol| if sym.usize() >= S { format!("g{}({})", sym.usize() - S, sym.usize()) } else { format!("{}({})", self.symbol_names[sym.usize()], sym.usize()) };
            let lhs = tostr(rule.lhs);
            let rhs0 = tostr(rule.rhs0);
            let rhs1 = if let Some(rhs1) = rule.rhs1 {
                format!(" {}", tostr(rhs1))
            } else {
                "".to_string()
            };
            writeln!(&mut result, "{}: {} ::= {}{};", i, lhs, rhs0, rhs1).unwrap();
        }
        result
    }
}

// Implementation for the recognizer.
//
// The recognizer has a chart of earley sets (Vec<EarleySet>) as well as the last set (next_set).
//
// A typical loop that utilizes the recognizer:
//
// - for character in string {
// 1.   recognizer.begin_earleme();
// 2.   recognizer.scan(token_to_symbol(character), values());
//        2a. complete
// 3.   recognizer.end_earleme();
//        3a. self.complete_all_sums_entirely();
//        3b. self.sort_medial_items();
//        3c. self.prediction_pass();
// - }
//
impl<const S: usize> Recognizer<S> {
    pub fn new(grammar: &Grammar<S>) -> Self {
        let mut result = Self {
            tables: Tables::new(grammar),
            earley_chart: EarleyChart::new(),
            forest: Forest::new(grammar),
            // complete: BinaryHeap::new_by_key(Box::new(|completed_item| (completed_item.origin, completed_item.dot))),
            complete: BinaryHeap::with_capacity(64),
            finished_node: None,
        };
        result.initialize();
        result
    }

    fn initialize(&mut self) {
        self.earley_chart.next_set(Some(
            self.tables.prediction_matrix[self.tables.start_symbol.usize()],
        ));
        self.earley_chart.next_set(None);
    }

    pub fn scan(&mut self, terminal: Symbol, values: u32) {
        let earleme = self.earley_chart.len() - 2;
        let node = self.forest.leaf(terminal, earleme + 1, values);
        self.complete(earleme, terminal, node);
    }

    pub fn end_earleme(&mut self) -> bool {
        if self.is_exhausted() {
            false
        } else {
            // Completion pass, which saves successful parses.
            self.finished_node = None;
            self.complete_all_sums_entirely();
            // Do the rest.
            self.sort_medial_items();
            self.prediction_pass();
            self.earley_chart.next_set(None);
            true
        }
    }

    pub fn is_exhausted(&self) -> bool {
        self.earley_chart.next_medial().len() == 0 && self.complete.is_empty()
    }

    fn complete_all_sums_entirely(&mut self) {
        while let Some(&ei) = self.complete.peek() {
            let lhs_sym = self.tables.get_lhs(ei.dot);
            let mut result_node = None;
            while let Some(&ei2) = self.complete.peek() {
                if ei.origin == ei2.origin && lhs_sym == self.tables.get_lhs(ei2.dot) {
                    result_node = Some(self.forest.push_summand(ei2));
                    self.complete.pop();
                } else {
                    break;
                }
            }
            if ei.origin == 0 && lhs_sym == self.tables.start_symbol {
                self.finished_node = Some(result_node.unwrap());
            }
            self.complete(ei.origin, lhs_sym, result_node.unwrap());
        }
    }

    /// Sorts medial items with deduplication.
    fn sort_medial_items(&mut self) {
        // Build index by postdot
        // These medial positions themselves are sorted by postdot symbol.
        self.earley_chart.last_mut().items.sort_unstable();
    }

    fn prediction_pass(&mut self) {
        // Iterate through medial items in the current set.
        let mut last = self.earley_chart.last_mut();
        let iter = last.items.iter();
        // For each medial item in the current set, predict its postdot symbol.
        for ei in iter {
            if let Some(postdot) = self
                .tables
                .get_rhs1(ei.dot)
                .filter(|postdot| !last.predicted[postdot.usize()])
            {
                // Prediction happens here. We would prefer to call `self.predict`, but we can't,
                // because `self.medial` is borrowed by `iter`.
                let source = &self.tables.prediction_matrix[postdot.usize()][..];
                last.predicted.union_with(source);
            }
        }
    }

    fn complete(&mut self, earleme: usize, symbol: Symbol, node: NodeHandle) {
        if symbol.usize() >= S {
            self.complete_binary_predictions(earleme, symbol, node);
        } else if self.earley_chart.predicted(earleme)[symbol.usize()] {
            self.complete_medial_items(earleme, symbol, node);
            self.complete_predictions(earleme, symbol, node);
        }
    }

    fn complete_medial_items(&mut self, earleme: usize, symbol: Symbol, right_node: NodeHandle) {
        let inner_start = {
            // we use binary search to narrow down the range of items.
            let set_idx = self.earley_chart[earleme]
                .binary_search_by(|ei| (self.tables.get_rhs1(ei.dot), 1).cmp(&(Some(symbol), 0)));
            match set_idx {
                Ok(idx) | Err(idx) => idx,
            }
        };

        let rhs1_eq = |ei: &&Item| self.tables.get_rhs1(ei.dot) == Some(symbol);
        for item in self.earley_chart[earleme][inner_start..]
            .iter()
            .take_while(rhs1_eq)
        {
            self.complete.push(CompletedItem {
                dot: item.dot,
                origin: item.origin,
                left_node: item.node,
                right_node: Some(right_node),
            });
        }
    }

    fn complete_predictions(&mut self, earleme: usize, symbol: Symbol, node: NodeHandle) {
        // println!("{:?}", slice);
        for trans in &self.tables.completions[symbol.usize()] {
            if self.earley_chart.predicted(earleme)[trans.top.usize()] {
                if trans.is_unary {
                    self.complete.push(CompletedItem {
                        origin: earleme,
                        dot: trans.dot,
                        left_node: node,
                        right_node: None,
                    });
                } else {
                    self.earley_chart.medial().push(Item {
                        origin: earleme,
                        dot: trans.dot,
                        node: node,
                        postdot: self.tables.get_rhs1(trans.dot).unwrap(),
                    });
                }
            }
        }
    }

    fn complete_binary_predictions(&mut self, earleme: usize, symbol: Symbol, node: NodeHandle) {
        let trans = self.tables.gen_completions[symbol.usize() - S];
        if self.earley_chart.predicted(earleme)[trans.top.usize()] {
            self.earley_chart.medial().push(Item {
                origin: earleme,
                dot: trans.dot,
                node,
                postdot: self.tables.get_rhs1(trans.dot).unwrap(),
            });
            if trans.is_unary {
                self.complete.push(CompletedItem {
                    origin: earleme,
                    dot: trans.dot,
                    left_node: node,
                    right_node: None,
                });
            }
        }
    }

    #[cfg(feature = "debug")]
    pub fn log_last_earley_set(&self) {
        let dots = self.dots_for_log(self.earley_chart.last());
        for (rule_id, dots) in dots {
            print!(
                "{} ::= ",
                self.tables.symbol_names[self.tables.get_lhs(rule_id).usize()]
            );
            if let Some(origins) = dots.get(&0) {
                print!("{:?}", origins);
            }
            print!(
                " {} ",
                self.tables.symbol_names[self.tables.rules[rule_id].rhs0.usize()]
            );
            if let Some(origins) = dots.get(&1) {
                print!("{:?}", origins);
            }
            if let Some(rhs1) = self.tables.get_rhs1(rule_id) {
                print!(" {} ", self.tables.symbol_names[rhs1.usize()]);
            }
            println!();
        }
        println!();
    }

    #[cfg(feature = "debug")]
    pub fn log_earley_set_diff(&self) {
        use std::collections::{BTreeMap, BTreeSet};
        let dots_last_by_id = self.dots_for_log(self.earley_chart.next_to_last());
        let mut dots_next_by_id = self.dots_for_log(self.earley_chart.last());
        let mut rule_ids: BTreeSet<usize> = BTreeSet::new();
        rule_ids.extend(dots_last_by_id.keys());
        rule_ids.extend(dots_next_by_id.keys());
        for item in self.complete.iter() {
            let position = if self.tables.get_rhs1(item.dot).is_some() {
                2
            } else {
                1
            };
            dots_next_by_id
                .entry(item.dot)
                .or_insert(BTreeMap::new())
                .entry(position)
                .or_insert(BTreeSet::new())
                .insert(item.origin);
        }
        let mut empty_diff = true;
        for rule_id in rule_ids {
            let dots_last = dots_last_by_id.get(&rule_id);
            let dots_next = dots_next_by_id.get(&rule_id);
            if dots_last == dots_next {
                continue;
            }
            empty_diff = false;
            print!(
                "from {} ::= ",
                self.tables.symbol_names[self.tables.get_top_lhs(rule_id).usize()]
            );
            if let Some(origins) = dots_last.and_then(|d| d.get(&0)) {
                print!("{:?}", origins);
            }
            print!(
                " {} ",
                self.tables.symbol_names[self.tables.get_top_rhs0(rule_id).usize()]
            );
            if let Some(origins) = dots_last.and_then(|d| d.get(&1)) {
                print!("{:?}", origins);
            }
            if let Some(rhs1) = self.tables.get_rhs1(rule_id) {
                print!(" {} ", self.tables.symbol_names[rhs1.usize()]);
            }
            println!();
            print!(
                "to   {} ::= ",
                self.tables.symbol_names[self.tables.get_top_lhs(rule_id).usize()]
            );
            if let Some(origins) = dots_next.and_then(|d| d.get(&0)) {
                print!("{:?}", origins);
            }
            print!(
                " {} ",
                self.tables.symbol_names[self.tables.get_top_rhs0(rule_id).usize()]
            );
            if let Some(origins) = dots_next.and_then(|d| d.get(&1)) {
                print!("{:?}", origins);
            }
            if let Some(rhs1) = self.tables.get_rhs1(rule_id) {
                print!(" {} ", self.tables.symbol_names[rhs1.usize()]);
            }
            if let Some(origins) = dots_next.and_then(|d| d.get(&2)) {
                print!("{:?}", origins);
            }
            println!();
        }
        if empty_diff {
            println!("no diff");
            println!();
        } else {
            println!();
        }
    }

    #[cfg(feature = "debug")]
    fn dots_for_log(&self, es: EarleySetRef) -> BTreeMap<usize, BTreeMap<usize, BTreeSet<usize>>> {
        let mut dots = BTreeMap::new();
        for (i, rule) in self.tables.rules.iter().enumerate() {
            if es.predicted[self.tables.get_top_lhs(i).usize()] {
                dots.entry(i)
                    .or_insert(BTreeMap::new())
                    .entry(0)
                    .or_insert(BTreeSet::new())
                    .insert(self.earley_chart.len() - 1);
            }
        }
        for item in es.items {
            dots.entry(item.dot)
                .or_insert(BTreeMap::new())
                .entry(1)
                .or_insert(BTreeSet::new())
                .insert(item.origin);
        }
        dots
    }
}

impl<const S: usize> Tables<S> {
    fn new(grammar: &Grammar<S>) -> Self {
        let mut result = Self {
            prediction_matrix: [[false; S]; S],
            start_symbol: grammar.start_symbol,
            rules: vec![],
            rules_by_rhs0: vec![],
            completions: vec![],
            symbol_names: grammar.symbol_names,
            gen_completions: vec![Default::default(); grammar.gen_symbols as usize],
        };
        result.populate(grammar);
        result
    }

    fn populate(&mut self, grammar: &Grammar<S>) {
        self.populate_rules(grammar);
        self.populate_prediction_matrix(grammar);
        self.populate_completions(grammar);
    }

    fn populate_prediction_matrix(&mut self, grammar: &Grammar<S>) {
        for rule in &grammar.rules {
            if rule.rhs0.usize() < S {
                let mut top = rule.lhs;
                while top.usize() >= S {
                    // appears on only one rhs0
                    let idx = self
                        .rules_by_rhs0
                        .binary_search_by_key(&top, |elem| elem.rhs0)
                        .expect("lhs not found");
                    top = self.rules_by_rhs0[idx].lhs;
                }
                self.prediction_matrix[top.usize()][rule.rhs0.usize()] = true;
            }
        }
        self.reflexive_closure();
        self.transitive_closure();
    }

    fn reflexive_closure(&mut self) {
        for i in 0..S {
            self.prediction_matrix[i][i] = true;
        }
    }

    fn transitive_closure(&mut self) {
        for pos in 0..S {
            let (rows0, rows1) = self.prediction_matrix.split_at_mut(pos);
            let (rows1, rows2) = rows1.split_at_mut(1);
            for dst_row in rows0.iter_mut().chain(rows2.iter_mut()) {
                if dst_row[pos] {
                    dst_row.union_with(&rows1[0]);
                }
            }
        }
    }

    fn populate_rules(&mut self, grammar: &Grammar<S>) {
        self.rules = grammar.rules.clone();
        self.rules_by_rhs0 = self.rules.clone();
        self.rules_by_rhs0.sort_by_key(|rule| rule.rhs0);
    }

    fn populate_completions(&mut self, grammar: &Grammar<S>) {
        self.completions
            .resize(S, vec![]);
        for (i, rule) in grammar.rules.iter().enumerate() {
            let rhs0 = rule.rhs0.usize();
            let mut top = rule.lhs;
            while top.usize() >= S {
                // appears on only one rhs0
                let idx = self
                    .rules_by_rhs0
                    .binary_search_by_key(&top, |elem| elem.rhs0)
                    .expect("lhs not found");
                top = self.rules_by_rhs0[idx].lhs;
            }
            let transition = PredictionTransition {
                symbol: rule.lhs,
                top,
                dot: i,
                is_unary: rule.rhs1.is_none(),
            };
            if rhs0 >= S {
                if rule.rhs1.is_some() {
                    self.gen_completions[rhs0 - S] = transition;
                } else {
                    self.gen_completions[rhs0 - S].is_unary = true;
                }
            } else {
                self.completions[rhs0].push(transition);
            }
        }
    }

    fn get_rhs1(&self, n: usize) -> Option<Symbol> {
        self.rules.get(n).and_then(|rule| rule.rhs1)
    }

    fn get_lhs(&self, n: usize) -> Symbol {
        self.rules[n].lhs
    }

    #[cfg(feature = "debug")]
    fn get_top_lhs(&self, dot: usize) -> Symbol {
        let mut top = self.rules[dot].lhs;
        while top.usize() >= S {
            // appears on only one rhs0
            let idx = self
                .rules_by_rhs0
                .binary_search_by_key(&top, |elem| elem.rhs0)
                .expect("lhs not found");
            top = self.rules_by_rhs0[idx].lhs;
        }
        top
    }

    #[cfg(feature = "debug")]
    fn get_top_rhs0(&self, dot: usize) -> Symbol {
        let mut top = self.rules[dot].rhs0;
        while top.usize() >= S {
            // appears on only one rhs0
            let idx = self
                .rules_by_rhs0
                .binary_search_by_key(&top, |elem| elem.rhs0)
                .expect("lhs not found");
            top = self.rules_by_rhs0[idx].lhs;
        }
        top
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    Digits(String),
    Float(f64),
    None,
}

#[derive(Clone)]
pub struct CalcRecognizer {
    grammar: Grammar<13>,
    recognizer: Recognizer<13>,
}

pub fn calc_recognizer() -> CalcRecognizer {
    let mut grammar = Grammar::new(
        [
            "sum", "factor", "op_mul", "op_div", "lparen", "rparen", "expr_sym", "op_minus",
            "op_plus", "number", "whole", "digit", "dot",
        ],
        0,
    );
    let [sum, factor, op_mul, op_div, lparen, rparen, expr_sym, op_minus, op_plus, number, whole, digit, dot] =
        grammar.symbols();
    // sum ::= sum [+-] factor
    // sum ::= factor
    // factor ::= factor [*/] expr
    // factor ::= expr
    // expr ::= '(' sum ')' | '-' expr | number
    // number ::= whole | whole '.' whole
    // whole ::= whole [0-9] | [0-9]
    grammar.rule(sum, [sum, op_plus, factor]);
    grammar.rule(sum, [sum, op_minus, factor]);
    grammar.rule(sum, [factor]);
    grammar.rule(factor, [factor, op_mul, expr_sym]);
    grammar.rule(factor, [factor, op_div, expr_sym]);
    grammar.rule(factor, [expr_sym]);

    grammar.rule(expr_sym, [lparen, sum, rparen]);
    grammar.rule(expr_sym, [op_minus, expr_sym]);
    grammar.rule(expr_sym, [number]);
    grammar.rule(number, [whole]);
    grammar.rule(number, [whole, dot, whole]);
    grammar.rule(whole, [whole, digit]);
    grammar.rule(whole, [digit]);
    grammar.sort_rules();
    let recognizer = Recognizer::new(&grammar);
    CalcRecognizer {
        recognizer,
        grammar,
    }
}

struct E {
    symbols: [Symbol; 13],
}

impl self::forest::Eval for E {
    type Elem = Value;

    fn leaf(&self, terminal: Symbol, values: u32) -> Self::Elem {
        let [sum, factor, op_mul, op_div, lparen, rparen, _expr_sym, op_minus, op_plus, _number, _whole, digit, dot] =
            self.symbols;
        if terminal == digit {
            Value::Digits((values as u8 as char).to_string())
        } else {
            Value::None
        }
    }

    fn product(&self, action: u32, args: Vec<Self::Elem>) -> Self::Elem {
        let [sum, factor, op_mul, op_div, lparen, rparen, _expr_sym, op_minus, op_plus, _number, _whole, digit, dot] =
            self.symbols;
        // let mut iter = args.into_iter();
        match (
            action,
            args.get(0).cloned().unwrap_or(Value::None),
            args.get(1).cloned().unwrap_or(Value::None),
            args.get(2).cloned().unwrap_or(Value::None),
        ) {
            (2, Value::Float(left), _, Value::Float(right)) => Value::Float(left + right),
            (3, Value::Float(left), _, Value::Float(right)) => Value::Float(left - right),
            (4, val, Value::None, Value::None) => val,
            (5, Value::Float(left), _, Value::Float(right)) => Value::Float(left * right),
            (6, Value::Float(left), _, Value::Float(right)) => Value::Float(left / right),
            (7, val, Value::None, Value::None) => val,
            (8, _, val, _) => val,
            (9, _, Value::Float(num), Value::None) => Value::Float(-num),
            (10, Value::Digits(digits), Value::None, Value::None) => {
                Value::Float(digits.parse::<f64>().unwrap())
            }
            (11, val @ Value::Digits(..), _, _) => val,
            (12, Value::Digits(before_dot), _, Value::Digits(after_dot)) => {
                let mut digits = before_dot;
                digits.push('.');
                digits.push_str(&after_dot[..]);
                Value::Digits(digits)
            }
            (13, Value::Digits(mut num), Value::Digits(digit), _) => {
                num.push_str(&digit[..]);
                Value::Digits(num)
            }
            (14, val @ Value::Digits(..), _, _) => val,
            args => panic!("unknown rule id {:?} or args {:?}", action, args),
        }
    }
}

impl CalcRecognizer {
    pub fn parse(&mut self, expr: &str) -> f64 {
        let [sum, factor, op_mul, op_div, lparen, rparen, _expr_sym, op_minus, op_plus, _number, _whole, digit, dot] =
            self.grammar.symbols();

        let symbols = self.grammar.symbols();

        for (i, ch) in expr.chars().enumerate() {
            let terminal = match ch {
                '-' => op_minus,
                '.' => dot,
                '0'..='9' => digit,
                '(' => lparen,
                ')' => rparen,
                '*' => op_mul,
                '/' => op_div,
                '+' => op_plus,
                ' ' => continue,
                other => panic!("invalid character {}", other),
            };
            self.recognizer.scan(terminal, ch as u32);
            let success = self.recognizer.end_earleme();
            #[cfg(feature = "debug")]
            if !success {
                self.recognizer.log_earley_set_diff();
            }
            assert!(success, "parse failed at character {}", i);
        }
        let finished_node = self.recognizer.finished_node.expect("parse failed");
        let result = self
            .recognizer
            .forest
            .evaluator(E { symbols })
            .evaluate(finished_node);
        if let Value::Float(num) = result {
            num
        } else {
            panic!("evaluation failed {:?}", result)
        }
    }
}

pub fn calc(expr: &str) -> f64 {
    let mut recognizer = calc_recognizer();
    recognizer.parse(expr)
}

#[test]
fn test_parse() {
    assert_eq!(calc("1.0 + 2.0"), 3.0);
}

#[test]
fn test_parse_big() {
    assert_eq!(calc("1.0 + (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 \
        * 3.01234234 + (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 * 3.01234234 + \
        (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 * 3.01234234"), 79.07405404);
}

#[bench]
fn bench_parser(bench: &mut Bencher) {
    let recognizer = calc_recognizer();
    bench.bytes = 9;
    bench.iter(|| {
        let mut parser = recognizer.clone();
        parser.parse("1.0 + 2.0")
    });
}

#[bench]
fn bench_parser2(bench: &mut Bencher) {
    let recognizer = calc_recognizer();
    bench.bytes = 76;
    bench.iter(|| {
        let mut parser = recognizer.clone();
        parser.parse("1.0 + 2.0 * 3.0 + 1.0 + 2.0 * 3.0 + 1.0 + 2.0 * 3.0 / 1.0 + 2.0 * 3.01234234")
    });
}

#[bench]
fn bench_parser3(bench: &mut Bencher) {
    let recognizer = calc_recognizer();
    bench.bytes = 68 + 92 + 74;
    bench.iter(|| {
        let mut parser = recognizer.clone();
        parser.parse("1.0 + (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 \
        * 3.01234234 + (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 * 3.01234234 + \
        (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 * 3.01234234")
    });
}

mod bench_c;
