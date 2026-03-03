use std::{
    cell::{Ref, RefCell, RefMut},
    collections::VecDeque,
    iter::zip,
    rc::{Rc, Weak},
};

use std::cell::Cell;

use crate::common::{GenericTensor, Shape, Tensor};

// This can let every thread control autograd
thread_local! {
    static AUTOGRAD_ENABLED: Cell<bool> = Cell::new(true);
}

pub fn is_grad_enabled() -> bool {
    AUTOGRAD_ENABLED.with(|g| g.get())
}

pub struct GradGuard {
    prev: bool,
}

impl GradGuard {
    pub fn new(state: bool) -> Self {
        let prev = AUTOGRAD_ENABLED.with(|g| {
            let old = g.get();
            g.set(state);
            old
        });

        Self { prev }
    }
}

impl Drop for GradGuard {
    fn drop(&mut self) {
        AUTOGRAD_ENABLED.with(|g| {
            g.set(self.prev);
        });
    }
}

#[macro_export]
macro_rules! no_grad {
    ($block: expr) => {{
        let _guard = $crate::common::autograd::GradGuard::new(false);
        $block
    }};
}

#[macro_export]
macro_rules! enable_grad {
    ($block: expr) => {{
        let _guard = $crate::common::autograd::GradGuard::new(true);
        $block
    }};
}

pub trait OpGrad {
    fn forward(&mut self, inputs: &[GenericTensor]) -> Vec<GenericTensor>;
    fn backward(&self, grad_inputs: &[TensorGrad]) -> Vec<TensorGrad>;
}

#[derive(Clone)]
pub struct GraphNode(Rc<RefCell<GraphNodeBase>>);

pub struct GraphNodeBase {
    op: Box<dyn OpGrad>,
    next_nodes: Vec<(GraphNode, usize)>,
    input_grad: Vec<Option<TensorGrad>>,
    output_shapes: Vec<Shape>,
    cnt: usize,
}

impl GraphNode {
    pub fn new(
        op: Box<dyn OpGrad>,
        inputs: &[TensorGrad],
        output_shapes: Vec<Shape>,
        num_outputs: usize,
    ) -> Self {
        let next_nodes = inputs
            .iter()
            .filter_map(|t| {
                if let Some(next_node) = &t.0.borrow().node {
                    Some((next_node.clone(), t.0.borrow().output_nr))
                } else {
                    None
                }
            })
            .collect::<Vec<(GraphNode, usize)>>();

        let base = GraphNodeBase {
            op,
            next_nodes,
            input_grad: vec![None; num_outputs],
            output_shapes,
            cnt: 0,
        };

        GraphNode(Rc::new(RefCell::new(base)))
    }
}
// pub fn leaf(
//     tensor: GenericTensor,
//     update_fn: Option<Box<dyn FnOnce(&mut GenericTensor, &GenericTensor)>>,
// ) -> Self {
//     GraphNode::new(GraphNodeBase::new(tensor, Vec::new(), None, update_fn))
// }

// pub fn borrow(&self) -> Ref<'_, GraphNodeBase> {
//     self.0.borrow()
// }

// pub fn borrow_mut(&self) -> RefMut<'_, GraphNodeBase> {
//     self.0.borrow_mut()
// }

//pub fn backward(&self, init_grad: Option<&[&GraphNode]>) {
// let mut node_base = RefCell::into_inner(
//     Rc::into_inner(self.0).expect("Unable to own the graph node exclusively."),
// );

// node_base.grad = Some(if let Some(init_grad) = init_grad {
//     assert!(
//         init_grad.len() == node_base.tensor.len(),
//         "The expected number of init grad is {}, but got {}.",
//         node_base.tensor.len(),
//         init_grad.len()
//     );

//     // Use init grad
//     init_grad
//         .iter()
//         .map(|t| (*t).clone())
//         .collect::<Vec<GenericTensor>>()
// } else {
//     // Default: use ones grad
//     node_base
//         .tensor
//         .iter()
//         .map(|t| GenericTensor::like_ones(t))
//         .collect::<Vec<GenericTensor>>()
// });

// node_base.backward();
//}

// pub fn backward(&mut self) {
//     let mut nodes_queue: VecDeque<GraphNode> = VecDeque::new();
//     nodes_queue.push_back(self.clone());

//     while let Some(node) = nodes_queue.pop_front() {
//         let grad_fn = &node.0.borrow().grad_fn;
//         let grad = &node.0.borrow().grad;

//         // Backward
//         if let Some(grad_fn) = grad_fn {
//             let next_grad = grad_fn.backward(
//                 grad.as_ref()
//                     .expect("Invalid computational graph.")
//                     .as_slice()
//             );
//             let next_nodes = &node.0.borrow().inputs;

//             // Zip
//             for (next_node, next_grad) in zip(next_nodes, next_grads) {
//             //     // Update grad
//             //     let mut next_node_b = next_node.borrow_mut();

//             //     if next_node_b.grad.is_none() {
//             //         // Create a new grad
//             //         next_node_b.grad = Some(GenericTensor::like_zeros(&next_grad));
//             //     }

//             //     let mut new_grad = next_node_b.grad.take().unwrap();
//             //     new_grad += &next_grad;
//             //     next_node_b.grad = Some(new_grad);

//             //     // Push back
//             //     nodes_queue.push_back(next_node.clone());
//             }
//         }
//     }
// }
//}

#[derive(Clone)]
pub struct TensorGrad(Rc<RefCell<TensorGradBase>>);

struct TensorGradBase {
    tensor: GenericTensor,
    grad: Option<TensorGrad>,
    node: Option<GraphNode>,
    output_nr: usize,
}

impl TensorGrad {
    fn new(base: TensorGradBase) -> Self {
        TensorGrad(Rc::new(RefCell::new(base)))
    }

    pub fn leaf(tensor: GenericTensor, requires_grad: bool) -> Self {
        // TODO: requires_grad
        TensorGrad::new(TensorGradBase {
            tensor,
            grad: None,
            node: None,
            output_nr: 0,
        })
    }

    pub fn intermediate(tensor: GenericTensor, node: GraphNode, output_nr: usize) -> Self {
        TensorGrad::new(TensorGradBase {
            tensor,
            grad: None,
            node: if is_grad_enabled() { Some(node) } else { None }, // Is in autograd?
            output_nr,
        })
    }

    pub fn borrow(&self) -> Ref<'_, GenericTensor> {
        Ref::map(self.0.borrow(), |b| &b.tensor)
    }

    pub fn backward(&mut self, init_grad: TensorGrad) {
        let mut nodes_queue: VecDeque<GraphNode> = VecDeque::new();

        // Step 1: Create a sub-DAG
        if let Some(node) = &self.0.borrow().node {
            nodes_queue.push_back(node.clone());
        }

        while let Some(node) = nodes_queue.pop_front() {
            for (next_node, _) in &node.0.borrow().next_nodes {
                let cnt = &mut next_node.0.borrow_mut().cnt;
                if *cnt == 0 {
                    // Hasn't visited
                    nodes_queue.push_back(next_node.clone());
                }
                *cnt += 1; // Increase in-deg

                // TODO: Init grad
            }
        }

        // Step 2: Topo-sort
        if let Some(node) = &self.0.borrow().node {
            // Init grad
            node.0.borrow_mut().input_grad[self.0.borrow().output_nr] = Some(init_grad);

            nodes_queue.push_back(node.clone());
        }

        while let Some(node) = nodes_queue.pop_front() {
            let op = &node.0.borrow().op;
            // Take grad from the node
            let grad = node
                .0
                .borrow_mut()
                .input_grad
                .iter_mut()
                .map(|g| g.take().unwrap())
                .collect::<Vec<TensorGrad>>();
            // Backward
            let output_grad = op.backward(&grad);

            assert!(
                output_grad.len() == node.0.borrow().next_nodes.len(),
                "Invalid computational graph."
            );

            // TODO: Check grad shape

            for ((next_node, pos), grad) in zip(&node.0.borrow().next_nodes, output_grad) {
                let mut base = next_node.0.borrow_mut();

                // Acc
                base.input_grad[*pos] = Some(
                    base.input_grad[*pos]
                        .as_ref()
                        .expect("Invalid computational graph.")
                        + &grad,
                );

                // Topo
                base.cnt -= 1;

                if base.cnt == 0 {
                    nodes_queue.push_back(next_node.clone());
                }
            }
        }
    }
}

impl Tensor for TensorGrad {
    fn shape(&self) -> super::Shape {
        self.0.borrow().tensor.shape()
    }

    fn device(&self) -> super::Device {
        self.0.borrow().tensor.device()
    }

    fn dtype(&self) -> super::DType {
        self.0.borrow().tensor.dtype()
    }
}
