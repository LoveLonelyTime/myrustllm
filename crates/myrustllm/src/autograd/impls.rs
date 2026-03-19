use crate::autograd::autograd_guard::is_autograd_enabled;
use crate::autograd::graph::{GradSlot, GraphNode};
use crate::common::init::TensorZerosInit;
use crate::common::ops::binary_ops::TensorAdd;
use crate::common::ops::reduce_ops::TensorAddReduce;
use crate::common::{DType, DTypeImpl, Impl, Shape, Tensor, TensorPrototype};
use std::marker::PhantomData;

//pub type GradSlot = Rc<RefCell<Option<>>>;

/// Autograd implementation.
/// This is a wrapper around the actual tensor implementation, which adds the autograd functionality.
///
/// Type parameters:
/// - `I`: The actual implementation of the tensor operations (e.g., CPU, CUDA, etc.).
/// - `GI`: The data type implementation for the gradient (e.g., F32, F64, etc.).
#[derive(Debug)]
pub struct Autograd<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> {
    _marker_i: PhantomData<I>,
    _marker_gi: PhantomData<GI>,
}

impl<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> Impl for Autograd<I, GI>
{
    type Device = I::Device;
}

impl<
    I: Impl,
    TI: DTypeImpl<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> DTypeImpl<Autograd<I, GI>> for TI
{
    type Prototype = AutogradPrototype<I, TI, GI>;
}

/// The prototype for the autograd tensor. It contains the actual tensor, the gradient slot, and the graph node.
pub struct AutogradPrototype<
    I: Impl,
    TI: DTypeImpl<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> {
    tensor: Tensor<I, TI>,
    grad: GradSlot<I, GI>,
    node: Option<GraphNode<I, GI>>,
    output_nr: usize,
}

impl<
    I: Impl,
    TI: DTypeImpl<I, Prototype: Clone>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> Clone for AutogradPrototype<I, TI, GI>
{
    fn clone(&self) -> Self {
        AutogradPrototype {
            tensor: self.tensor.clone(),
            grad: self.grad.clone(),
            node: self.node.clone(),
            output_nr: self.output_nr,
        }
    }
}

impl<
    I: Impl,
    TI: DTypeImpl<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> TensorPrototype<Autograd<I, GI>> for AutogradPrototype<I, TI, GI>
{
    fn shape(&self) -> Shape {
        self.tensor.shape()
    }

    fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    fn device(&self) -> <I as Impl>::Device {
        self.tensor.device()
    }
}

impl<
    I: Impl,
    TI: DTypeImpl<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> AutogradPrototype<I, TI, GI>
{
    /// Create a leaf tensor on DAG.
    pub fn leaf(tensor: Tensor<I, TI>) -> Self {
        AutogradPrototype {
            tensor,
            grad: GradSlot::new(),
            node: None,
            output_nr: 0,
            // retains_grad: Cell::new(requires_grad),
        }
    }

    /// Create a immediate tensor.
    /// Generally, a immediate tensor is created by a autograd-operation, which creates (node, output_nr).
    pub fn immediate(tensor: Tensor<I, TI>, node: GraphNode<I, GI>, output_nr: usize) -> Self {
        AutogradPrototype {
            tensor,
            grad: GradSlot::new(),
            node: if is_autograd_enabled() {
                Some(node)
            } else {
                None
            }, // If autograd is closed, it doesn't need to attach a node.
            output_nr,
        }
    }

    /// Return the reference of its tensor.
    pub fn tensor(&self) -> &Tensor<I, TI> {
        &self.tensor
    }

    /// Return the grad slot.
    pub fn grad(&self) -> GradSlot<I, GI> {
        self.grad.clone()
    }

    /// Return the location of its tensor on its node.
    pub fn output_nr(&self) -> usize {
        self.output_nr
    }

    /// Return its node on DAG.
    pub fn node(&self) -> Option<GraphNode<I, GI>> {
        self.node.clone()
    }

    /// Require the tensor to catch grad.
    pub fn catch_grad(&mut self) {
        self.grad.catch_grad();
    }
}
