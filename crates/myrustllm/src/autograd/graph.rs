use std::{
    cell::{Ref, RefCell},
    collections::VecDeque,
    iter::zip,
    rc::{Rc, Weak},
};

use crate::{
    autograd::impls::Autograd,
    common::{
        Tensor,
        dtype::DTypeImpl,
        impls::Impl,
        init::TensorZerosInit,
        ops::binary_ops::TensorAdd,
        tensor::{TensorMeta, TensorPrototype},
    },
};

pub trait OpGrad<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> {
    fn forward(&self, inputs: &[&Tensor<I, TI>]) -> Vec<Tensor<I, TI>>;
    fn backward(
        &self,
        grad_inputs: &[&Tensor<Autograd<I, GI>, GI>],
    ) -> Vec<Tensor<Autograd<I, GI>, GI>>;
}

pub struct GradSlot<I: Impl, GI: DTypeImpl<I>> {
    slot: Option<Tensor<Autograd<I, GI>, GI>>,
    require_grad: bool,
}

impl<I: Impl, GI: DTypeImpl<I>> GradSlot<I, GI> {
    pub fn new(require_grad: bool) -> Self {
        GradSlot {
            slot: None,
            require_grad,
        }
    }

    pub fn require_grad(&self) -> bool {
        self.require_grad
    }

    pub fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>) {}
}

pub struct GraphNode<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>>(
    Rc<RefCell<GraphNodeBase<I, TI, GI>>>,
);

impl<
    I: Impl,
    TI: DTypeImpl<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI>,
> GraphNode<I, TI, GI>
{
    fn init_grad(&mut self) {
        self.0.borrow_mut().init_grad();
    }

    fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>, pos: usize) {
        self.0.borrow_mut().acc_grad(grad, pos);
    }

    fn backward(&mut self, retain_graph: bool) -> Vec<Tensor<Autograd<I, GI>, GI>> {
        self.0.borrow_mut().backward(retain_graph)
    }

    fn add_output_weak(&mut self, output: &Tensor<Autograd<I, GI>, TI>) {
        self.0.borrow_mut().put_grad_slot(output);
    }

    fn dispatch_grad(&self) {
        self.0.borrow().dispatch_grad();
    }
}

impl<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> Clone for GraphNode<I, TI, GI> {
    fn clone(&self) -> Self {
        GraphNode(self.0.clone())
    }
}

struct GraphNodeBase<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> {
    op: Option<Box<dyn OpGrad<I, TI, GI>>>,
    next_nodes: Vec<(Option<GraphNode<I, TI, GI>>, usize)>,
    input_grad: Vec<Option<Tensor<Autograd<I, GI>, GI>>>,
    output_metas: Vec<TensorMeta<I>>,
    grad_slot: Vec<Weak<RefCell<GradSlot<I, GI>>>>,
    cnt: usize,
}

impl<
    I: Impl,
    TI: DTypeImpl<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI>,
> GraphNodeBase<I, TI, GI>
{
    /// Init `input_grad` with zero grad.
    fn init_grad(&mut self) {
        // TODO: replace fill_zeros with broadcast_to
        for ((shape, _dtype, device), grad) in zip(&self.output_metas, &mut self.input_grad) {
            *grad = Some(Tensor::<Autograd<I, GI>, GI>::zeros(shape, device));
        }
    }

    /// Accumulate `grad` into `input_grad[pos]`.
    fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>, pos: usize) {
        assert!(
            grad.shape() == self.output_metas[pos].0,
            "Invalid computational graph."
        );

        let acc_grad = self.input_grad[pos].as_mut().unwrap();
        *acc_grad = &*acc_grad + grad;
    }

    /// Backward the node.
    fn backward(&mut self, retain_graph: bool) -> Vec<Tensor<Autograd<I, GI>, GI>> {
        let input_grad = self
            .input_grad
            .iter_mut()
            .map(|o| o.take().unwrap())
            .collect::<Vec<Tensor<Autograd<I, GI>, GI>>>();

        let output_grad = self
            .op
            .as_ref()
            .expect("Invalid computational graph.")
            .backward(
                &input_grad
                    .iter()
                    .collect::<Vec<&Tensor<Autograd<I, GI>, GI>>>(),
            );

        if !retain_graph {
            // Drop op
            self.op = None
        }

        assert!(
            output_grad.len() == self.next_nodes.len(),
            "Invalid computational graph."
        );

        output_grad
    }

    /// Add a weak ref of output.
    ///
    /// This can help node dispatch grad to tensor.
    fn put_grad_slot(&mut self, output: &Tensor<Autograd<I, GI>, TI>) {
        self.grad_slot[output.prototype.output_nr()] = Rc::downgrade(output.prototype.grad());
    }

    /// Dispatch grad to tensor.
    fn dispatch_grad(&self) {
        for (output_nr, output) in self.grad_slot.iter().enumerate() {
            if let Some(output) = output.upgrade() {
                let mut slot = output.borrow_mut();
                slot.acc_grad(self.input_grad[output_nr].as_ref().unwrap());
            }
        }
    }
}

impl<
    I: Impl,
    TI: DTypeImpl<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI>,
> Tensor<Autograd<I, GI>, TI>
{
    pub fn backward(&mut self, init_grad: &Tensor<Autograd<I, GI>, GI>, retain_graph: bool) {
        let mut nodes_queue: VecDeque<GraphNode<I, TI, GI>> = VecDeque::new();

        // Step 1: Create a sub-DAG
        if let Some(node) = self.prototype.node() {
            nodes_queue.push_back(node.clone());
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
        if let Some(node) = self.prototype.node() {
            let mut node = node.clone();
            // Init grad
            node.acc_grad(init_grad, self.prototype.output_nr());

            nodes_queue.push_back(node.clone());
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
