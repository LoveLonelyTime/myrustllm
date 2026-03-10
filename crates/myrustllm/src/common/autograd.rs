use crate::common::{DType, Device, GenericTensor, Shape, Tensor, TensorMeta};
use std::cell::{Cell, Ref, RefCell};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::zip;
use std::rc::{Rc, Weak};

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

pub trait OpGrad: Debug {
    fn forward(&self, inputs: &[&GenericTensor]) -> Vec<GenericTensor>;
    fn backward(&self, grad_inputs: &[&TensorGrad]) -> Vec<TensorGrad>;
}

#[derive(Clone, Debug)]
pub struct GraphNode(Rc<RefCell<GraphNodeBase>>);

#[derive(Debug)]
struct GraphNodeBase {
    op: Option<Box<dyn OpGrad>>,
    next_nodes: Vec<(Option<GraphNode>, usize)>,
    input_grad: Vec<Option<TensorGrad>>,
    output_metas: Vec<TensorMeta>,
    output_weak: Vec<Weak<TensorGradBase>>,
    cnt: usize,
}

impl GraphNodeBase {
    /// Init `input_grad` with zero grad.
    fn init_grad(&mut self) {
        // TODO: replace fill_zeros with broadcast_to
        for ((shape, dtype, device), grad) in zip(&self.output_metas, &mut self.input_grad) {
            *grad = Some(TensorGrad::leaf(
                GenericTensor::zeros(shape, *dtype, *device),
                false,
            ));
        }
    }

    /// Accumulate `grad` into `input_grad[pos]`.
    fn acc_grad(&mut self, grad: &TensorGrad, pos: usize) {
        assert!(
            grad.tensor().shape() == self.output_metas[pos].0,
            "Invalid computational graph."
        );

        let acc_grad = self.input_grad[pos].as_mut().unwrap();
        *acc_grad = &*acc_grad + grad;
    }

    /// Backward the node.
    fn backward(&mut self, retain_graph: bool) -> Vec<TensorGrad> {
        let input_grad = self
            .input_grad
            .iter_mut()
            .map(|o| o.take().unwrap())
            .collect::<Vec<TensorGrad>>();

        let output_grad = self
            .op
            .as_ref()
            .expect("Invalid computational graph.")
            .backward(&input_grad.iter().collect::<Vec<&TensorGrad>>());

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
    fn add_output_weak(&mut self, output: &TensorGrad) {
        self.output_weak.push(Rc::downgrade(&output.0));
    }

    /// Dispatch grad to tensor.
    fn dispatch_grad(&self) {
        for output in &self.output_weak {
            if let Some(output) = output.upgrade()
                && output.retains_grad.get()
            {
                let mut old_grad = output.grad.borrow_mut();
                let acc = if let Some(t) = &*old_grad {
                    t + self.input_grad[output.output_nr].as_ref().unwrap()
                } else {
                    self.input_grad[output.output_nr].as_ref().unwrap().clone()
                };
                *old_grad = Some(acc);
            }
        }
    }
}
#[derive(Debug)]
struct LeafGrad;

impl OpGrad for LeafGrad {
    fn forward(&self, _inputs: &[&GenericTensor]) -> Vec<GenericTensor> {
        Vec::new()
    }

    fn backward(&self, _grad_inputs: &[&TensorGrad]) -> Vec<TensorGrad> {
        Vec::new()
    }
}
impl GraphNode {
    /// Create a node on DAG.
    ///
    /// This function will apply `inputs` with `op` and create a node with connections on DAG.
    /// This function will return (node, outputs). If a node isn't needed on DAG, it will return `None`.
    pub fn forward(
        op: impl OpGrad + 'static,
        inputs: &[&TensorGrad],
    ) -> (Option<Self>, Vec<GenericTensor>) {
        // Execute op
        let outputs = op.forward(
            &inputs
                .iter()
                .map(|t| t.tensor())
                .collect::<Vec<&GenericTensor>>(),
        );

        // Create the node
        let node = if inputs.iter().any(|t| t.requires_grad()) {
            // Do we require grad?
            // Connect edges
            let next_nodes = inputs
                .iter()
                .map(|t| (t.node().map(GraphNode::clone), t.output_nr()))
                .collect::<Vec<(Option<GraphNode>, usize)>>();

            // Collect output tensor metas
            let output_metas = outputs
                .iter()
                .map(|t| (t.shape().clone(), t.dtype(), t.device()))
                .collect::<Vec<TensorMeta>>();

            let base = GraphNodeBase {
                op: Some(Box::new(op)),
                next_nodes,
                input_grad: vec![None; outputs.len()],
                output_metas,
                output_weak: Vec::new(),
                cnt: 0,
            };

            Some(GraphNode(Rc::new(RefCell::new(base))))
        } else {
            None
        };

        (node, outputs)
    }

    pub fn leaf(tensor: &GenericTensor) -> Self {
        let base = GraphNodeBase {
            op: Some(Box::new(LeafGrad)),
            next_nodes: Vec::new(),
            input_grad: vec![None; 1],
            output_metas: vec![(tensor.shape().clone(), tensor.dtype(), tensor.device())],
            output_weak: Vec::new(),
            cnt: 0,
        };
        GraphNode(Rc::new(RefCell::new(base)))
    }

    fn init_grad(&mut self) {
        self.0.borrow_mut().init_grad();
    }

    fn acc_grad(&mut self, grad: &TensorGrad, pos: usize) {
        self.0.borrow_mut().acc_grad(grad, pos);
    }

    fn backward(&mut self, retain_graph: bool) -> Vec<TensorGrad> {
        self.0.borrow_mut().backward(retain_graph)
    }

    fn add_output_weak(&mut self, output: &TensorGrad) {
        self.0.borrow_mut().add_output_weak(output);
    }

    fn dispatch_grad(&self) {
        self.0.borrow().dispatch_grad();
    }
}

#[derive(Clone, Debug)]
pub struct TensorGrad(Rc<TensorGradBase>);

#[derive(Debug)]
struct TensorGradBase {
    tensor: GenericTensor,
    grad: RefCell<Option<TensorGrad>>,
    node: Option<GraphNode>,
    output_nr: usize,
    retains_grad: Cell<bool>,
}

impl Tensor for TensorGrad {
    fn shape(&self) -> Shape {
        self.0.tensor.shape()
    }

    fn device(&self) -> Device {
        self.0.tensor.device()
    }

    fn dtype(&self) -> DType {
        self.0.tensor.dtype()
    }
}

impl TensorGrad {
    /// Create a leaf tensor.
    ///
    /// If a leaf tensor doesn't need grad, you can set `requires_grad` to `false`.
    pub fn leaf(tensor: GenericTensor, requires_grad: bool) -> Self {
        if requires_grad {
            let mut node = GraphNode::leaf(&tensor);
            let tensor = TensorGrad(Rc::new(TensorGradBase {
                tensor,
                grad: RefCell::new(None),
                node: Some(node.clone()),
                output_nr: 0,
                retains_grad: Cell::new(requires_grad),
            }));
            node.add_output_weak(&tensor);
            tensor
        } else {
            TensorGrad(Rc::new(TensorGradBase {
                tensor,
                grad: RefCell::new(None),
                node: None,
                output_nr: 0,
                retains_grad: Cell::new(requires_grad),
            }))
        }
    }

    /// Create a intermediate tensor.
    ///
    /// A intermediate tensor is created by a `impl OpGrad`, which is represented by a `GraphNode` on DAG.
    pub fn intermediate(tensor: GenericTensor, node: Option<GraphNode>, output_nr: usize) -> Self {
        let output = TensorGrad(Rc::new(TensorGradBase {
            tensor,
            grad: RefCell::new(None),
            node: if is_grad_enabled() { node } else { None }, // Is in autograd?,
            output_nr,
            retains_grad: Cell::new(false),
        }));

        if let Some(mut node) = output.node().map(GraphNode::clone) {
            node.add_output_weak(&output);
        }

        output
    }

    /// Return the ref of the inner tensor.
    pub fn tensor(&self) -> &GenericTensor {
        &self.0.tensor
    }

    /// Return the ref of the inner node.
    fn node(&self) -> Option<&GraphNode> {
        self.0.node.as_ref()
    }

    /// Return the output location of the inner node.
    pub fn output_nr(&self) -> usize {
        self.0.output_nr
    }

    /// Does the node require grad?
    pub fn requires_grad(&self) -> bool {
        self.0.node.is_some()
    }

    // Does the node retain grad?
    pub fn retains_grad(&self) -> bool {
        self.0.retains_grad.get()
    }

    pub fn retain_grad(&mut self) {
        self.0.retains_grad.set(true);
    }

    /// Return the ref of the inner grad.
    pub fn grad(&self) -> Option<TensorGrad> {
        self.0.grad.borrow().clone()
    }

    pub fn zero_grad(&mut self) {
        self.0.grad.borrow_mut().take();
    }

    /// Backward from the node.
    pub fn backward(&mut self, init_grad: &TensorGrad, retain_graph: bool) {
        let mut nodes_queue: VecDeque<GraphNode> = VecDeque::new();

        // Step 1: Create a sub-DAG
        if let Some(node) = self.0.node.clone() {
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
        if let Some(mut node) = self.0.node.clone() {
            // Init grad
            node.acc_grad(init_grad, self.0.output_nr);

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
