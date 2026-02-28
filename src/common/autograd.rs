use std::{
    cell::{Ref, RefCell, RefMut},
    collections::VecDeque,
    iter::zip,
    rc::Rc,
};

use std::cell::Cell;

use crate::common::GenericTensor;

// This can let every thread control autograd
thread_local! {
    static AUTOGRAD_ENABLED: Cell<bool> = Cell::new(true);
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
    ($block: expr) => {
        {
            let _guard = $crate::common::autograd::GradGuard::new(false);
            $block
        }
    };
}

#[macro_export]
macro_rules! enable_grad {
    ($block: expr) => {
        {
            let _guard = $crate::common::autograd::GradGuard::new(true);
            $block
        }
    };
}


pub trait OpGrad {
    fn forward(&mut self, inputs: &[&GenericTensor]) -> Vec<GenericTensor>;
    fn backward(&self, grad_inputs: &[&GraphNode]) -> Vec<GraphNode>;
}

#[derive(Clone)]
pub struct GraphNode(Rc<RefCell<GraphNodeBase>>);

impl GraphNode {
    pub fn new(base: GraphNodeBase) -> Self {
        GraphNode(Rc::new(RefCell::new(base)))
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

    pub fn backward(&self, init_grad: Option<&[&GraphNode]>) {
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
    }
}

pub struct GraphNodeBase {
    pub tensor: Vec<GenericTensor>,
    pub grad: Option<Vec<GraphNode>>,
    pub inputs: Vec<GraphNode>,
    pub grad_fn: Option<Box<dyn OpGrad>>,
    // pub update_fn: Option<Box<dyn FnOnce(&[&mut GenericTensor], &[&GenericTensor])>>,
}

// impl GraphNodeBase {
//     // pub fn new(
//     //     tensor: GenericTensor,
//     //     inputs: Vec<GraphNode>,
//     //     grad_fn: Option<Box<dyn OpGrad>>,
//     //     update_fn: Option<Box<dyn FnOnce(&mut CPUGenericTensor, &CPUGenericTensor)>>,
//     // ) -> Self {
//     //     GraphNodeBase {
//     //         tensor,
//     //         grad: None,
//     //         inputs,
//     //         grad_fn: grad_fn,
//     //         update_fn: update_fn,
//     //     }
//     // }

//     pub fn backward(self) {
//         let mut nodes_queue: VecDeque<GraphNodeBase> = VecDeque::new();
//         nodes_queue.push_back(self);

//         while let Some(node) = nodes_queue.pop_front() {

//             let update_fn = node.update_fn.take();
//             let grad_fn = node.grad_fn.take();
//             let grad = node.grad.take();

//             // Update callback
//             if let Some(update_fn) = update_fn {
//                 if let Some(grad) = &grad {
//                     update_fn(&mut node.tensor, grad);
//                 }
//             }

//             // Backward
//             if let Some(mut grad_fn) = grad_fn {
//                 let next_grads = grad_fn.backward(&grad.expect("Invalid graph!"));
//                 let next_nodes = &mut node.inputs;

//                 // Zip
//                 for (next_node, next_grad) in zip(next_nodes, next_grads) {
//                     // Update grad
//                     let mut next_node_b = next_node.borrow_mut();

//                     if next_node_b.grad.is_none() {
//                         // Create a new grad
//                         next_node_b.grad = Some(GenericTensor::like_zeros(&next_grad));
//                     }

//                     let mut new_grad = next_node_b.grad.take().unwrap();
//                     new_grad += &next_grad;
//                     next_node_b.grad = Some(new_grad);

//                     // Push back
//                     nodes_queue.push_back(next_node.clone());
//                 }
//             }
//         }
//     }
// }
