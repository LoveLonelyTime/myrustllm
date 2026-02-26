// use std::{
//     cell::{Ref, RefCell, RefMut},
//     collections::VecDeque,
//     iter::zip,
//     rc::Rc,
// };

// use crate::cpu::tensor::CPUGenericTensor;

// pub trait CPUOpGrad {
//     fn forward(&mut self, inputs: &[CPUGenericTensor]) -> CPUGenericTensor;
//     fn backward(&mut self, grad_inputs: &CPUGenericTensor) -> Vec<CPUGenericTensor>;
// }

// pub struct CPUGraphNode(Rc<RefCell<CPUGraphNodeBase>>);

// impl CPUGraphNode {
//     pub fn new(base: CPUGraphNodeBase) -> Self {
//         CPUGraphNode(Rc::new(RefCell::new(base)))
//     }

//     pub fn borrow(&self) -> Ref<'_, CPUGraphNodeBase> {
//         self.0.borrow()
//     }

//     pub fn borrow_mut(&self) -> RefMut<'_, CPUGraphNodeBase> {
//         self.0.borrow_mut()
//     }
// }

// impl Clone for CPUGraphNode {
//     fn clone(&self) -> Self {
//         CPUGraphNode(self.0.clone())
//     }
// }

// pub struct CPUGraphNodeBase {
//     tensor: CPUGenericTensor,
//     grad: Option<CPUGenericTensor>,
//     inputs: Vec<CPUGraphNode>,
//     grad_fn: Option<Box<dyn CPUOpGrad>>,
//     update_fn: Option<Box<dyn FnOnce(&mut CPUGenericTensor, &CPUGenericTensor)>>,
// }

// impl CPUGraphNodeBase {

//     pub fn backward(self) {
//         let mut nodes_queue = VecDeque::new();
//         nodes_queue.push_back(CPUGraphNode::new(self));

//         while let Some(node) = nodes_queue.pop_front() {
//             let mut node = node.borrow_mut();
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
//                         next_node_b.grad = Some(CPUGenericTensor::like_zeros(&next_grad));
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
