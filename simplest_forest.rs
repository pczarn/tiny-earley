// Forest

use std::iter::Product;

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
    Evaluated {
        index: usize,
    }
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
    stack: Vec<NodeHandle>,
}

impl<T, F, G> Evaluator<F, G>
where
    F: Fn(u32, &[T], &[usize]) -> T + Copy,
    G: Fn(Symbol, u32) -> T + Copy,
    T: Clone + ::std::fmt::Debug,
{
    pub fn new(eval_product: F, eval_leaf: G) -> Self {
        Self {
            eval_product,
            eval_leaf,
            stack: vec![],
        }
    }

    pub fn evaluate(&mut self, forest: &mut Forest, finished_node: NodeHandle) -> T {
        let mut liveness = ::std::iter::repeat(false).take(forest.graph.len()).collect::<Vec<_>>();
        let mut work = vec![finished_node];
        let mut result = vec![];
        let mut refs = vec![];
        while let Some(node) = work.pop() {
            if !liveness[node.0] {
                match forest.graph[node.0] {
                    Node::Product { left_factor, right_factor, action } => {
                        liveness[node.0] = action != NULL_ACTION;
                        if let Some(right) = right_factor {
                            work.push(right);
                        }
                        work.push(left_factor);
                    }
                    Node::Leaf { .. } => {
                        liveness[node.0] = true;
                    }
                    Node::Evaluated { .. } => unreachable!()
                }
            }
        }
        for (i, alive) in (0 .. forest.graph.len()).zip(liveness.iter().cloned()) {
            if alive {
                self.evaluate_node(forest, NodeHandle(i), &mut result, &mut refs);
            }
        }
        match forest.graph[finished_node.0] {
            Node::Evaluated { index } => {
                result[index].clone()
            }
            _ => unreachable!()
        }
    }
    
    fn evaluate_node<'a>(&mut self, forest: &mut Forest, node: NodeHandle, result: &'a mut Vec<T>, refs: &'a mut Vec<usize>) {
        #[derive(Eq, PartialEq)]
        enum Action {
            Leaf(Symbol, u32),
            Product(u32),
        }
        self.stack.push(node);
        let mut first_action = Action::Product(NULL_ACTION);
        while let Some(factor) = self.stack.pop() {
            match &forest.graph[factor.0] {
                &Node::Product { action, left_factor, right_factor } => {
                    if action != NULL_ACTION {
                        first_action = Action::Product(action);
                    }
                    if let Some(rfactor) = right_factor {
                        self.stack.push(rfactor);
                    }
                    self.stack.push(left_factor);
                }
                &Node::Evaluated { index } => {
                    refs.push(index);
                }
                &Node::Leaf { terminal, values } => {
                    first_action = Action::Leaf(terminal, values)
                }
            }
        }
        let finished = match first_action {
            Action::Leaf(terminal, values) => {
                (self.eval_leaf)(terminal, values)
            }
            Action::Product(action) => {
                (self.eval_product)(action, &result[..], &refs[..])
            }
        };
        refs.clear();
        self.stack.clear();
        forest.graph[node.0] = Node::Evaluated { index: result.len() };
        result.push(finished);
    }
}