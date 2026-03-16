use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{
    autograd::graph::{GradSlot, GraphNode},
    common::{DType, Shape, Tensor, dtype::DTypeImpl, impls::Impl, tensor::TensorPrototype},
};

//pub type GradSlot = Rc<RefCell<Option<>>>;

pub struct Autograd<I: Impl, GI: DTypeImpl<I>> {
    _marker: PhantomData<I>,
    _marker2: PhantomData<GI>,
}

impl<I: Impl, GI: DTypeImpl<I>> Impl for Autograd<I, GI> {
    type Device = I::Device;
}

impl<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> DTypeImpl<Autograd<I, GI>> for TI {
    type Prototype = AutoGradPrototype<I, TI, GI>;
}

pub struct AutoGradPrototype<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> {
    tensor: Tensor<I, TI>,
    grad: Rc<RefCell<GradSlot<I, GI>>>,
    node: Option<GraphNode<I, TI, GI>>,
    output_nr: usize,
}

impl<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> TensorPrototype<Autograd<I, GI>>
    for AutoGradPrototype<I, TI, GI>
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

impl<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> AutoGradPrototype<I, TI, GI> {
    pub fn leaf(tensor: Tensor<I, TI>) -> Self {
        AutoGradPrototype {
            tensor,
            grad: Rc::new(RefCell::new(GradSlot::new(false))),
            node: None,
            output_nr: 0,
            // retains_grad: Cell::new(requires_grad),
        }
    }

    pub fn output_nr(&self) -> usize {
        self.output_nr
    }

    pub fn grad(&self) -> &Rc<RefCell<GradSlot<I, GI>>> {
        &self.grad
    }

    pub fn node(&self) -> Option<&GraphNode<I, TI, GI>> {
        self.node.as_ref()
    }
}
