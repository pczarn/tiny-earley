// Forest

use std::{collections::LinkedList, iter};

use super::*;

#[derive(Clone)]
pub struct Forest {
    graph: Vec<u8>,
    eval: Vec<Option<usize>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct NodeHandle(usize, usize);

impl NodeHandle {
    fn get(self) -> usize {
        self.1 - self.0
    }
}

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

static NODE_SIZES: &'static [(u8, u8, u8, bool)] = &[
    (8, 8, 0, false),
    (16, 16, 0, false),
    (32, 32, 0, false),
    (32, 64, 0, false),
    (8, 8, 0, true),
    (8, 8, 8, true),
    (8, 16, 8, true),
    (16, 16, 8, true),
    (8, 16, 16, true),
    (8, 16, 0, true),
    (8, 24, 24, true),
    (16, 24, 24, true),
    (16, 32, 32, true),
    (16, 32, 0, true),
    (32, 64, 64, true),
    (32, 64, 0, true),
    (32, 64, 16, true),
];

const NULL_ACTION: u32 = 0;

impl Forest {
    fn push_node(&mut self, node: Node) {
        let size = |mut x: u64| {
            let mut result = 0;
            while x != 0 {
                x = x >> 8;
                result += 8;
            }
            result
        };
        let tup = match node {
            Node::Product { action, left_factor, right_factor } => {
                (action as u32, (self.graph.len() - left_factor.get()) as u64, right_factor.map_or(0, |f| (self.graph.len() - f.get()) as u64), true)
            }
            Node::Leaf { terminal, values } => {
                (terminal.0 as u32, values as u64, 0, false)
            }
        };
        let s = (size(tup.0 as u64), size(tup.1), size(tup.2));
        let (idx, size) = NODE_SIZES.iter().enumerate().find(|(_i, sizes)| s.0 <= sizes.0 && s.1 <= sizes.1 && s.2 <= sizes.2 && tup.3 == sizes.3).unwrap();
        // if idx.is_none() {
        //     panic!("wrong size for {:?} {:?}", tup, size(tup.1));
        // }
        // let idx = idx.unwrap();
        // let size = NODE_SIZES[idx];

        let mut result = [0u8; 24];
        result[0 .. size.0 as usize / 8].copy_from_slice(&u32::to_le_bytes(tup.0)[0 .. size.0 as usize / 8]);
        result[size.0 as usize / 8 .. size.0 as usize / 8 + size.1 as usize / 8].copy_from_slice(&u64::to_le_bytes(tup.1)[0 .. size.1 as usize / 8]);
        result[size.0 as usize / 8 + size.1 as usize / 8 .. size.0 as usize / 8 + size.1 as usize / 8 + size.2 as usize / 8].copy_from_slice(&u64::to_le_bytes(tup.2)[0 .. size.2 as usize / 8]);
        self.graph.push(idx as u8);
        self.graph.extend(result[0 .. size.0 as usize / 8 + size.1 as usize / 8 + size.2 as usize / 8].into_iter().cloned());
    }
}

impl Forest {
    pub fn memory_use(&self) -> usize {
        self.graph.len()
    }

    pub fn new<const S: usize>(grammar: &Grammar<S>) -> Self {
        Self {
            graph: vec![0],
            eval: grammar.rules.iter().map(|rule| rule.id).collect(),
        }
    }

    pub fn leaf(&mut self, terminal: Symbol, _x: usize, values: u32) -> NodeHandle {
        let handle = NodeHandle(0, self.graph.len());
        self.push_node(Node::Leaf { terminal, values });
        handle
    }

    pub fn push_summand(&mut self, item: CompletedItem) -> NodeHandle {
        let handle = NodeHandle(0, self.graph.len());
        let eval = self.eval[item.dot].map(|id| id as u32);
        self.push_node(Node::Product {
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

    fn get(&self, handle: NodeHandle) -> Node {
        let slice = &self.graph[handle.get() ..];
        let size = NODE_SIZES[slice[0] as usize];
        let all = &slice[1 .. size.0 as usize / 8 + size.1 as usize / 8 + size.2 as usize / 8 + 1];
        let (first, second) = all.split_at(size.0 as usize / 8);
        let (second, third) = second.split_at(size.1 as usize / 8);
        let mut a = [0; 4];
        a[0 .. first.len()].copy_from_slice(first);
        let mut b = [0; 8];
        b[0 .. second.len()].copy_from_slice(second);
        let mut c = [0; 8];
        c[0 .. third.len()].copy_from_slice(third);
        if size.3 {
            Node::Product {
                action: u32::from_le_bytes(a),
                left_factor: NodeHandle(u64::from_le_bytes(b) as usize, handle.get()),
                right_factor: if size.2 == 0 { None } else { Some(NodeHandle(u64::from_le_bytes(c) as usize, handle.get())) },
            }
        } else {
            Node::Leaf {
                terminal: Symbol(u32::from_le_bytes(a)),
                values: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            }
        }
    }
}

pub trait Eval {
    type Elem: Send;
    fn leaf(&self, terminal: Symbol, values: u32) -> Self::Elem;
    fn product(&self, action: u32, list: &mut LinkedList<Self::Elem>) -> Self::Elem;
}

pub struct Evaluator<'a, T> {
    eval: T,
    forest: &'a mut Forest,
}

impl<'a, T: Eval + Send + Sync> Evaluator<'a, T> {
    pub fn evaluate(&mut self, finished_node: NodeHandle) -> T::Elem {
        self.evaluate_rec(finished_node).into_iter().next().unwrap()
    }

    fn evaluate_rec(&self, handle: NodeHandle) -> LinkedList<T::Elem> {
        match self.forest.get(handle) {
            Node::Product {
                left_factor,
                right_factor,
                action,
            } => {
                // add parallel
                // let mut evald = self.evaluate_rec(left_factor);
                let mut evald;
                // if handle.get() < limit + 1 {
                    // println!("non-parallel");
                //     if let Some(factor) = right_factor {
                //         let mut a = self.evaluate_rec(left_factor, limit);
                //         a.append(&mut self.evaluate_rec(factor));
                //         evald = a;
                //     } else {
                //         evald = self.evaluate_rec(left_factor);
                //     }
                // } else {
                    // println!("parallel");
                    if let Some(factor) = right_factor {
                        let (mut a, mut b) = rayon::join(|| self.evaluate_rec(left_factor), || self.evaluate_rec(factor));
                        a.append(&mut b);
                        evald = a;
                    } else {
                        evald = self.evaluate_rec(left_factor);
                    }
                // }
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