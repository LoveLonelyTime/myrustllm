use std::{
    cell::{Ref, RefCell},
    collections::VecDeque,
    iter::zip,
    rc::{Rc, Weak},
};

use crate::{
    autograd::impls::{Autograd, AutogradPrototype},
    common::{
        Shape, Tensor,
        dtype::DTypeImpl,
        impls::Impl,
        init::TensorZerosInit,
        ops::{binary_ops::TensorAdd, reduce_ops::TensorAddReduce},
        tensor::{TensorMetadata, TensorPrototype},
    },
};

pub struct GradSlot<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
>(Rc<RefCell<GradSlotBase<I, GI>>>);

impl<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> Clone for GradSlot<I, GI>
{
    fn clone(&self) -> Self {
        GradSlot(self.0.clone())
    }
}

impl<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> GradSlot<I, GI>
{
    pub fn new() -> Self {
        GradSlot(Rc::new(RefCell::new(GradSlotBase::new())))
    }

    pub fn catch_grad(&mut self) {
        self.0.borrow_mut().catch_grad();
    }

    pub fn require_grad(&self) -> bool {
        self.0.borrow().require_grad()
    }

    pub fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>) {
        self.0.borrow_mut().acc_grad(grad);
    }

    pub fn grad(&self) -> Option<Ref<'_, Tensor<Autograd<I, GI>, GI>>> {
        Ref::filter_map(self.0.borrow(), |s| s.slot.as_ref()).ok()
    }
}

pub struct GradSlotBase<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> {
    slot: Option<Tensor<Autograd<I, GI>, GI>>,
    require_grad: bool,
}

impl<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> GradSlotBase<I, GI>
{
    pub fn new() -> Self {
        GradSlotBase {
            slot: None,
            require_grad: false,
        }
    }

    pub fn catch_grad(&mut self) {
        self.require_grad = true;
    }

    pub fn require_grad(&self) -> bool {
        self.require_grad
    }

    pub fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>) {
        if self.slot.is_none() {
            self.slot = Some(Tensor::<Autograd<I, GI>, GI>::zeros(
                &Shape::scalar(),
                &grad.device(),
            ));
        }

        self.slot = Some(&self.slot.take().unwrap() + grad);
    }
}

/// The graph node on DAG for autograd.
pub struct GraphNode<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
>(Rc<RefCell<GraphNodeBase<I, GI>>>);

impl<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> Clone for GraphNode<I, GI>
{
    fn clone(&self) -> Self {
        GraphNode(self.0.clone())
    }
}

impl<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> GraphNode<I, GI>
{
    pub fn new() -> Self {
        GraphNode(Rc::new(RefCell::new(GraphNodeBase::new())))
    }

    pub fn add_input<TI: DTypeImpl<I>>(self, tensor: &AutogradPrototype<I, TI, GI>) -> Self {
        self.0
            .borrow_mut()
            .next_nodes
            .push((tensor.node(), tensor.output_nr()));
        self
    }

    pub fn set_backward_op(
        self,
        backward_op: impl FnMut(&[Tensor<Autograd<I, GI>, GI>]) -> Vec<Tensor<Autograd<I, GI>, GI>>
        + 'static,
    ) -> Self {
        self.0.borrow_mut().backward_op = Some(Box::new(backward_op));
        self
    }

    pub fn attch_output<TI: DTypeImpl<I>>(
        &self,
        tensor: Tensor<I, TI>,
        output_nr: usize,
    ) -> AutogradPrototype<I, TI, GI> {
        self.0
            .borrow_mut()
            .output_metas
            .push((tensor.shape(), tensor.dtype(), tensor.device()));

        self.0.borrow_mut().input_grad.push(None);
        let output = AutogradPrototype::immediate(tensor, self.clone(), output_nr);
        self.0
            .borrow_mut()
            .grad_slot
            .push(Rc::downgrade(&output.grad().0));
        output
    }

    pub fn init_grad(&mut self) {
        self.0.borrow_mut().init_grad();
    }

    pub fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>, pos: usize) {
        self.0.borrow_mut().acc_grad(grad, pos);
    }

    pub fn backward(&mut self, retain_graph: bool) -> Vec<Tensor<Autograd<I, GI>, GI>> {
        self.0.borrow_mut().backward(retain_graph)
    }

    fn dispatch_grad(&self) {
        self.0.borrow().dispatch_grad();
    }
}

/// The base struct for the graph node. It contains the gradient data and edges for the graph node.
pub struct GraphNodeBase<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> {
    // The backward function
    backward_op:
        Option<Box<dyn FnMut(&[Tensor<Autograd<I, GI>, GI>]) -> Vec<Tensor<Autograd<I, GI>, GI>>>>,
    // Next nodes on the backward path, with the output index of the input tensors on the forawrd path. This is used to store the edges of the graph node.
    // The length of this vector is the same as the number of inputs of the node on the forward path.
    next_nodes: Vec<(Option<GraphNode<I, GI>>, usize)>,
    // Input gradients on the backward path. This is used to store the intermediate gradients during the backward pass.
    // The length of this vector is the same as the number of outputs of the node on the forward path.
    input_grad: Vec<Option<Tensor<Autograd<I, GI>, GI>>>,
    // The metadata of the outputs. This is used to check the shape and dtype of the gradients during the backward pass.
    output_metas: Vec<TensorMetadata<I>>,
    // The grad slots of the outputs on the forward path.
    // The length of this vector is the same as the number of outputs of the node on the forward path.
    grad_slot: Vec<Weak<RefCell<GradSlotBase<I, GI>>>>,
    // The in-deg of the node on the backward path. This is used to determine when to push the node into the queue during the backward pass.
    cnt: usize,
}

impl<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> GraphNodeBase<I, GI>
{
    pub fn new() -> Self {
        GraphNodeBase {
            backward_op: None,
            next_nodes: Vec::new(),
            input_grad: Vec::new(),
            output_metas: Vec::new(),
            grad_slot: Vec::new(),
            cnt: 0,
        }
    }
    /// Init `input_grad` with zero grad.
    pub fn init_grad(&mut self) {
        // TODO: replace fill_zeros with broadcast_to
        for ((shape, _dtype, device), grad) in zip(&self.output_metas, &mut self.input_grad) {
            *grad = Some(Tensor::<Autograd<I, GI>, GI>::zeros(shape, device));
        }
    }

    /// Accumulate `grad` into `input_grad[pos]`.
    pub fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>, pos: usize) {
        assert!(
            grad.shape() == self.output_metas[pos].0,
            "Invalid computational graph."
        );

        let acc_grad = self.input_grad[pos].as_mut().unwrap();
        *acc_grad = &*acc_grad + grad;
    }

    /// Backward the node.
    pub fn backward(&mut self, retain_graph: bool) -> Vec<Tensor<Autograd<I, GI>, GI>> {
        // Take all input grads
        let input_grad = self
            .input_grad
            .iter_mut()
            .map(|o| o.take().unwrap())
            .collect::<Vec<Tensor<Autograd<I, GI>, GI>>>();

        // Backward the node
        let output_grad = self
            .backward_op
            .as_mut()
            .expect("Invalid computational graph.")(&input_grad);

        // Drop backward op if not retain graph
        if !retain_graph {
            self.backward_op.take();
        }

        assert!(
            output_grad.len() == self.next_nodes.len(),
            "Invalid computational graph."
        );

        output_grad
    }

    // Add a weak ref of output.
    //
    // This can help node dispatch grad to tensor.
    // pub fn put_grad_slot(&mut self, output: &Tensor<Autograd<I, GI>, TI>) {
    //     self.grad_slot[output.prototype.output_nr()] = Rc::downgrade(output.prototype.grad());
    // }

    /// Dispatch grad to tensor.
    pub fn dispatch_grad(&self) {
        for (output_nr, slot) in self.grad_slot.iter().enumerate() {
            if let Some(slot) = slot.upgrade()
                && slot.borrow().require_grad()
            {
                slot.borrow_mut()
                    .acc_grad(self.input_grad[output_nr].as_ref().unwrap());
            }
        }
    }
}

// The implementation of backward for the autograd tensor. It will create a sub-DAG for the tensor, and then do a topo-sort to backward the graph.
impl<
    I: Impl,
    TI: DTypeImpl<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> Tensor<Autograd<I, GI>, TI>
{
    /// Backward from the tensor.
    pub fn backward(&mut self, init_grad: &Tensor<Autograd<I, GI>, GI>, retain_graph: bool) {
        let mut nodes_queue: VecDeque<GraphNode<I, GI>> = VecDeque::new();

        // Step 1: Create a sub-DAG
        if let Some(node) = self.prototype.node() {
            nodes_queue.push_back(node);
        }

        while let Some(mut node) = nodes_queue.pop_front() {
            // Init grad
            node.init_grad();

            for (next_node, _) in &node.0.borrow().next_nodes {
                if let Some(next_node) = next_node {
                    let cnt = &mut next_node.0.borrow_mut().cnt;
                    if *cnt == 0 {
                        // Hasn't visited
                        nodes_queue.push_back(next_node.clone());
                    }
                    *cnt += 1; // Increase in-deg
                }
            }
        }

        // Step 2: Topo-sort
        if let Some(mut node) = self.prototype.node() {
            // Init grad
            node.acc_grad(init_grad, self.prototype.output_nr());

            nodes_queue.push_back(node);
        }

        while let Some(mut node) = nodes_queue.pop_front() {
            // Dispatch
            node.dispatch_grad();

            // Backward
            let output_grad = node.backward(retain_graph);

            for ((next_node, pos), grad) in zip(&node.0.borrow().next_nodes, output_grad) {
                if let Some(next_node) = next_node {
                    let mut base = next_node.0.borrow_mut();

                    // Acc
                    base.acc_grad(&grad, *pos);

                    // Topo
                    base.cnt -= 1;

                    if base.cnt == 0 {
                        nodes_queue.push_back(next_node.clone());
                    }
                }
            }
        }
    }
}
