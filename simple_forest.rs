// Forest

use super::*;

#[derive(Clone)]
pub struct Forest {
    graph: Vec<Node>,
    eval: Vec<Option<usize>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct NodeHandle(usize);

#[derive(Clone)]
enum Node {
    Product {
        action: u32,
        left_factor: NodeHandle,
        right_factor: Option<NodeHandle>,
    },
    Leaf {
        terminal: Symbol,
        values: u32,
    },
}

const NULL_ACTION: u32 = !0;

impl Forest {
    pub fn new<const S: usize>(grammar: &Grammar<S>) -> Self {
        Self {
            graph: vec![],
            eval: grammar.rules.iter().map(|rule| rule.id).collect(),
        }
    }

    pub fn leaf(&mut self, terminal: Symbol, _x: usize, values: u32) -> NodeHandle {
        let handle = NodeHandle(self.graph.len());
        self.graph.push(Node::Leaf { terminal, values });
        handle
    }

    pub fn push_summand(&mut self, item: CompletedItem) -> NodeHandle {
        let handle = NodeHandle(self.graph.len());
        let eval = self.eval[item.dot].map(|id| id as u32);
        self.graph.push(Node::Product {
            action: eval.unwrap_or(NULL_ACTION),
            left_factor: item.left_node,
            right_factor: item.right_node,
        });
        handle
    }
}

pub struct Evaluator<F, G> {
    eval_product: F,
    eval_leaf: G,
}

struct Rec<T>(T, Option<Box<Rec<T>>>);

impl<T, F, G> Evaluator<F, G>
where
    F: Fn(u32, &[T]) -> T + Copy,
    G: Fn(Symbol, u32) -> T + Copy,
    T: Clone + ::std::fmt::Debug,
{
    pub fn new(eval_product: F, eval_leaf: G) -> Self {
        Self {
            eval_product,
            eval_leaf,
        }
    }

    pub fn evaluate(&mut self, forest: &mut Forest, finished_node: NodeHandle) -> T {
        self.evaluate_rec(forest, finished_node)[0].clone()
    }

    fn evaluate_rec(&mut self, forest: &mut Forest, handle: NodeHandle) -> Vec<T> {
        match forest.graph[handle.0] {
            Node::Product {
                left_factor,
                right_factor,
                action,
            } => {
                let mut result = self.evaluate_rec(forest, left_factor);
                if let Some(factor) = right_factor {
                    result.extend(self.evaluate_rec(forest, factor));
                }
                if action != NULL_ACTION {
                    vec![(self.eval_product)(action as u32, &result)]
                } else {
                    result
                }
            }
            Node::Leaf { terminal, values } => {
                vec![(self.eval_leaf)(terminal, values)]
            }
        }
    }
}