// Forest

use std::collections::LinkedList;

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
    pub fn memory_use(&self) -> usize {
        self.graph.len() * std::mem::size_of::<Node>()        
    }

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

    pub fn evaluator<T: Eval>(&mut self, eval: T) -> Evaluator<T> {
        Evaluator {
            forest: self,
            eval,
        }
    }
}

pub trait Eval {
    type Elem;
    fn leaf(&mut self, terminal: Symbol, values: u32) -> Self::Elem;
    fn product(&mut self, action: u32, list: &mut LinkedList<Self::Elem>) -> Self::Elem;
}

pub struct Evaluator<'a, T> {
    eval: T,
    forest: &'a mut Forest,
}

impl<'a, T: Eval> Evaluator<'a, T> {
    pub fn evaluate(&mut self, finished_node: NodeHandle) -> T::Elem {
        self.evaluate_rec(finished_node).into_iter().next().unwrap()
    }

    fn evaluate_rec(&mut self, handle: NodeHandle) -> LinkedList<T::Elem> {
        match self.forest.graph[handle.0] {
            Node::Product {
                left_factor,
                right_factor,
                action,
            } => {
                let mut evald = self.evaluate_rec(left_factor);
                if let Some(factor) = right_factor {
                    evald.append(&mut self.evaluate_rec(factor));
                }
                if action != NULL_ACTION {
                    let mut list = LinkedList::new();
                    list.push_back(self.eval.product(action as u32, &mut evald));
                    list
                } else {
                    evald
                }
            }
            Node::Leaf { terminal, values } => {
                let mut list = LinkedList::new();
                list.push_back(self.eval.leaf(terminal, values));
                list
            }
        }
    }
}