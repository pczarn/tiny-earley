// Forest

// use std::{simd::{u64x4, Simd}};

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

#[derive(Clone, Copy)]
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

// const fn gen() -> [(u8, u8, u8, bool); 8 * 8 * 8] {
//     let mut result = [(0, 0, 0, false); 512];
//     for
//     result
// }

// static NODE_SIZES: [(u8, u8, u8, bool); 512] = gen();

const NULL_ACTION: u32 = 1;

impl Forest {
    fn push_node(&mut self, node: Node) {
        let size = |mut x: u64| {
            let mut result = 0;
            while x != 0 {
                x = x >> 8;
                result += 1;
            }
            result
        };
        let tup = match node {
            Node::Product {
                action,
                left_factor,
                right_factor,
            } => (
                action as u32,
                (self.graph.len() - left_factor.get()) as u64,
                right_factor.map_or(0, |f| (self.graph.len() - f.get()) as u64),
            ),
            Node::Leaf { terminal, values } => (0, terminal.0 as u64, values as u64),
        };
        let s = (size(tup.0 as u64), size(tup.1), size(tup.2));
        let idx = s.0 + s.1 * 4 + s.2 * 4 * 8;
        self.graph.push(idx);
        self.graph.extend(&u32::to_le_bytes(tup.0)[0..s.0 as usize]);
        self.graph.extend(&u64::to_le_bytes(tup.1)[0..s.1 as usize]);
        self.graph.extend(&u64::to_le_bytes(tup.2)[0..s.2 as usize]);
        // let (idx, size) = NODE_SIZES.iter().enumerate().find(|(_i, sizes)| s.0 <= sizes.0 && s.1 <= sizes.1 && s.2 <= sizes.2 && tup.3 == sizes.3).unwrap();
        // // if idx.is_none() {
        // //     panic!("wrong size for {:?} {:?}", tup, size(tup.1));
        // // }
        // // let idx = idx.unwrap();
        // // let size = NODE_SIZES[idx];

        // let mut result = [0u8; 24];
        // let val = u64x4::from(0);

        // let zeros = Simd::from_array([0, 0, 0, 0, 0, 0, 0, 0]);

        // result[0 .. size.0 as usize / 8].copy_from_slice(&u32::to_le_bytes(tup.0)[0 .. size.0 as usize / 8]);
        // result[size.0 as usize / 8 .. size.0 as usize / 8 + size.1 as usize / 8].copy_from_slice(&u64::to_le_bytes(tup.1)[0 .. size.1 as usize / 8]);
        // result[size.0 as usize / 8 + size.1 as usize / 8 .. size.0 as usize / 8 + size.1 as usize / 8 + size.2 as usize / 8].copy_from_slice(&u64::to_le_bytes(tup.2)[0 .. size.2 as usize / 8]);
        // self.graph.push(idx as u8);
        // self.graph.extend(result[0 .. size.0 as usize / 8 + size.1 as usize / 8 + size.2 as usize / 8].into_iter().cloned());
    }
}

impl Forest {
    pub fn memory_use(&self) -> usize {
        self.graph.len()
    }

    pub fn new<const S: usize>(grammar: &Grammar<S>) -> Self {
        Self {
            graph: vec![0, 0],
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
        Evaluator { forest: self, eval }
    }

    fn get(&self, handle: NodeHandle) -> Node {
        let slice = &self.graph[handle.get()..];
        let size = slice[0];
        let s = (size % 4, (size / 4) % 8, size / 4 / 8);
        let all = &slice[1..(s.0 + s.1 + s.2) as usize + 1];
        let (first, second) = all.split_at(s.0 as usize);
        let (second, third) = second.split_at(s.1 as usize);
        let mut a = [0; 4];
        a[0..first.len()].copy_from_slice(first);
        let mut b = [0; 8];
        b[0..second.len()].copy_from_slice(second);
        let mut c = [0; 8];
        c[0..third.len()].copy_from_slice(third);
        if s.0 != 0 {
            Node::Product {
                action: u32::from_le_bytes(a),
                left_factor: NodeHandle(u64::from_le_bytes(b) as usize, handle.get()),
                right_factor: if s.2 == 0 {
                    None
                } else {
                    Some(NodeHandle(u64::from_le_bytes(c) as usize, handle.get()))
                },
            }
        } else {
            Node::Leaf {
                terminal: Symbol(u64::from_le_bytes(b) as u32),
                values: u64::from_le_bytes(c) as u32,
            }
        }
    }
}

pub trait Eval {
    type Elem: Send;
    fn leaf(&self, terminal: Symbol, values: u32) -> Self::Elem;
    fn product(&self, action: u32, list: Vec<Self::Elem>) -> Self::Elem;
}

pub struct Evaluator<'a, T> {
    eval: T,
    forest: &'a mut Forest,
}

struct Work<Elem> {
    node: Node,
    progress: u32,
    parent: u32,
    result: Vec<Elem>,
}

impl<'a, T: Eval + Send + Sync> Evaluator<'a, T>
where
    T::Elem: ::std::fmt::Debug,
{
    pub fn evaluate(&self, finished_node: NodeHandle) -> T::Elem {
        let mut stack = vec![
            Work {
                node: Node::Leaf {
                    terminal: Symbol(0),
                    values: 0,
                },
                progress: 0,
                parent: 0,
                result: vec![],
            },
            Work {
                node: self.forest.get(finished_node),
                progress: 0,
                parent: 0,
                result: vec![],
            },
        ];
        while stack.len() > 1 {
            let mut work = stack.pop().expect("stack too small");
            match (work.node, work.progress) {
                (Node::Product {
                    left_factor,
                    right_factor,
                    action,
                }, 0) => {
                    work.progress = 1;
                    let new = Work {
                        node: self.forest.get(left_factor),
                        progress: 0,
                        parent: if action == NULL_ACTION { work.parent } else { stack.len() as u32 },
                        result: vec![],
                    };
                    stack.push(work);
                    stack.push(new);
                }
                (Node::Product {
                    left_factor,
                    right_factor: Some(right),
                    action,
                }, 1) => {
                    work.progress = 2;
                    let new = Work {
                        node: self.forest.get(right),
                        progress: 0,
                        parent: if action == NULL_ACTION { work.parent } else { stack.len() as u32 },
                        result: vec![],
                    };
                    stack.push(work);
                    stack.push(new);
                }
                (Node::Product {
                    left_factor,
                    right_factor,
                    action: NULL_ACTION,
                }, _) => {}
                (Node::Product {
                    left_factor,
                    right_factor,
                    action,
                }, _) => {
                    let evaluated = self.eval.product(action, work.result);
                    stack[work.parent as usize].result.push(evaluated);
                }
                (Node::Leaf { terminal, values }, _) => {
                    let evaluated = self.eval.leaf(terminal, values);
                    stack[work.parent as usize].result.push(evaluated);
                }
            }
        }
        stack
            .into_iter()
            .next()
            .unwrap()
            .result
            .into_iter()
            .next()
            .unwrap()
    }
}

// test bench_c::bench_parse_c ... bench:  95,337,687.00 ns/iter (+/- 3,388,675.44) = 3 MB/s
// test bench_parser           ... bench:       3,560.12 ns/iter (+/- 601.29) = 2 MB/s
// test bench_parser2          ... bench:      19,440.32 ns/iter (+/- 2,518.07) = 3 MB/s
// test bench_parser3          ... bench:      57,736.32 ns/iter (+/- 1,987.76) = 4 MB/s
