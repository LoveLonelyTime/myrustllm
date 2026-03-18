#![feature(prelude_import)]
extern crate std;
#[prelude_import]
use std::prelude::rust_2024::*;
pub mod autograd {
    pub mod impls {
        use std::{cell::RefCell, marker::PhantomData, rc::Rc};
        use crate::{
            autograd::graph::{GradSlot, GraphNode},
            common::{
                DType, Shape, Tensor, dtype::DTypeImpl, impls::Impl,
                tensor::TensorPrototype,
            },
        };
        pub struct Autograd<I: Impl, GI: DTypeImpl<I>> {
            _marker: PhantomData<I>,
            _marker2: PhantomData<GI>,
        }
        impl<I: Impl, GI: DTypeImpl<I>> Impl for Autograd<I, GI> {
            type Device = I::Device;
        }
        impl<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> DTypeImpl<Autograd<I, GI>>
        for TI {
            type Prototype = AutoGradPrototype<I, TI, GI>;
        }
        pub struct AutoGradPrototype<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> {
            tensor: Tensor<I, TI>,
            grad: Rc<RefCell<GradSlot<I, GI>>>,
            node: Option<GraphNode<I, TI, GI>>,
            output_nr: usize,
        }
        impl<
            I: Impl,
            TI: DTypeImpl<I>,
            GI: DTypeImpl<I>,
        > TensorPrototype<Autograd<I, GI>> for AutoGradPrototype<I, TI, GI> {
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
    }
    pub mod graph {
        use std::{
            cell::{Ref, RefCell},
            collections::VecDeque, iter::zip, rc::{Rc, Weak},
        };
        use crate::{
            autograd::impls::Autograd,
            common::{
                Tensor, dtype::DTypeImpl, impls::Impl, init::TensorZerosInit,
                ops::binary_ops::TensorAdd, tensor::{TensorMetadata, TensorPrototype},
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
        > GraphNode<I, TI, GI> {
            fn init_grad(&mut self) {
                self.0.borrow_mut().init_grad();
            }
            fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>, pos: usize) {
                self.0.borrow_mut().acc_grad(grad, pos);
            }
            fn backward(
                &mut self,
                retain_graph: bool,
            ) -> Vec<Tensor<Autograd<I, GI>, GI>> {
                self.0.borrow_mut().backward(retain_graph)
            }
            fn add_output_weak(&mut self, output: &Tensor<Autograd<I, GI>, TI>) {
                self.0.borrow_mut().put_grad_slot(output);
            }
            fn dispatch_grad(&self) {
                self.0.borrow().dispatch_grad();
            }
        }
        impl<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> Clone
        for GraphNode<I, TI, GI> {
            fn clone(&self) -> Self {
                GraphNode(self.0.clone())
            }
        }
        struct GraphNodeBase<I: Impl, TI: DTypeImpl<I>, GI: DTypeImpl<I>> {
            op: Option<Box<dyn OpGrad<I, TI, GI>>>,
            next_nodes: Vec<(Option<GraphNode<I, TI, GI>>, usize)>,
            input_grad: Vec<Option<Tensor<Autograd<I, GI>, GI>>>,
            output_metas: Vec<TensorMetadata<I>>,
            grad_slot: Vec<Weak<RefCell<GradSlot<I, GI>>>>,
            cnt: usize,
        }
        impl<
            I: Impl,
            TI: DTypeImpl<I>,
            GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI>,
        > GraphNodeBase<I, TI, GI> {
            /// Init `input_grad` with zero grad.
            fn init_grad(&mut self) {
                for ((shape, _dtype, device), grad) in zip(
                    &self.output_metas,
                    &mut self.input_grad,
                ) {
                    *grad = Some(Tensor::<Autograd<I, GI>, GI>::zeros(shape, device));
                }
            }
            /// Accumulate `grad` into `input_grad[pos]`.
            fn acc_grad(&mut self, grad: &Tensor<Autograd<I, GI>, GI>, pos: usize) {
                if !(grad.shape() == self.output_metas[pos].0) {
                    {
                        ::core::panicking::panic_fmt(
                            format_args!("Invalid computational graph."),
                        );
                    }
                }
                let acc_grad = self.input_grad[pos].as_mut().unwrap();
                *acc_grad = &*acc_grad + grad;
            }
            /// Backward the node.
            fn backward(
                &mut self,
                retain_graph: bool,
            ) -> Vec<Tensor<Autograd<I, GI>, GI>> {
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
                        &input_grad.iter().collect::<Vec<&Tensor<Autograd<I, GI>, GI>>>(),
                    );
                if !retain_graph {
                    self.op = None;
                }
                if !(output_grad.len() == self.next_nodes.len()) {
                    {
                        ::core::panicking::panic_fmt(
                            format_args!("Invalid computational graph."),
                        );
                    }
                }
                output_grad
            }
            /// Add a weak ref of output.
            ///
            /// This can help node dispatch grad to tensor.
            fn put_grad_slot(&mut self, output: &Tensor<Autograd<I, GI>, TI>) {
                self.grad_slot[output.prototype.output_nr()] = Rc::downgrade(
                    output.prototype.grad(),
                );
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
        > Tensor<Autograd<I, GI>, TI> {
            pub fn backward(
                &mut self,
                init_grad: &Tensor<Autograd<I, GI>, GI>,
                retain_graph: bool,
            ) {
                let mut nodes_queue: VecDeque<GraphNode<I, TI, GI>> = VecDeque::new();
                if let Some(node) = self.prototype.node() {
                    nodes_queue.push_back(node.clone());
                }
                while let Some(mut node) = nodes_queue.pop_front() {
                    node.init_grad();
                    for (next_node, _) in &node.0.borrow().next_nodes {
                        if let Some(next_node) = next_node {
                            let cnt = &mut next_node.0.borrow_mut().cnt;
                            if *cnt == 0 {
                                nodes_queue.push_back(next_node.clone());
                            }
                            *cnt += 1;
                        }
                    }
                }
                if let Some(node) = self.prototype.node() {
                    let mut node = node.clone();
                    node.acc_grad(init_grad, self.prototype.output_nr());
                    nodes_queue.push_back(node.clone());
                }
                while let Some(mut node) = nodes_queue.pop_front() {
                    node.dispatch_grad();
                    let output_grad = node.backward(retain_graph);
                    for ((next_node, pos), grad) in zip(
                        &node.0.borrow().next_nodes,
                        output_grad,
                    ) {
                        if let Some(next_node) = next_node {
                            let mut base = next_node.0.borrow_mut();
                            base.acc_grad(&grad, *pos);
                            base.cnt -= 1;
                            if base.cnt == 0 {
                                nodes_queue.push_back(next_node.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    pub mod init {
        use crate::{
            autograd::impls::Autograd,
            common::{
                Shape, Tensor, dtype::DTypeImpl, impls::Impl,
                init::{TensorAllocInit, TensorZerosInit},
            },
        };
        impl<
            I: Impl,
            TI: DTypeImpl<I> + TensorAllocInit<I>,
            GI: DTypeImpl<I>,
        > TensorAllocInit<Autograd<I, GI>> for TI {
            fn init(
                shape: &Shape,
                device: &<Autograd<I, GI> as Impl>::Device,
            ) -> Self::Prototype {
                Self::Prototype::leaf(Tensor::new(TI::init(shape, device)))
            }
        }
        impl<
            I: Impl,
            TI: DTypeImpl<I> + TensorZerosInit<I>,
            GI: DTypeImpl<I>,
        > TensorZerosInit<Autograd<I, GI>> for TI {
            fn init(
                shape: &Shape,
                device: &<Autograd<I, GI> as Impl>::Device,
            ) -> Self::Prototype {
                Self::Prototype::leaf(Tensor::new(TI::init(shape, device)))
            }
        }
    }
    pub mod ops {
        pub mod binary_ops {
            use crate::{
                autograd::impls::Autograd,
                common::{dtype::DTypeImpl, impls::Impl, ops::binary_ops::TensorAdd},
            };
            impl<
                I: Impl,
                GI: DTypeImpl<I>,
                TI: DTypeImpl<I> + TensorAdd<I, Rhs>,
                Rhs: DTypeImpl<I>,
            > TensorAdd<Autograd<I, GI>, Rhs> for TI {
                type Output = TI::Output;
                fn add(
                    lhs: &Self::Prototype,
                    rhs: &<Rhs as DTypeImpl<Autograd<I, GI>>>::Prototype,
                ) -> <Self::Output as DTypeImpl<Autograd<I, GI>>>::Prototype {
                    ::core::panicking::panic("not yet implemented")
                }
            }
        }
    }
}
pub mod common {
    pub mod dtype {
        //! MyRustLLM has two type system:
        //! - Dynamic types: These types are defined by DType (u8). Some dispatching system can identify dynamic types from DType ID.
        //! - Generic param types: These types are defined by generic params, which are used in rust-lang.
        use crate::common::{Impl, TensorPrototype};
        /// DType ID (u8).
        pub type DType = u8;
        pub const DTYPE_F32: DType = 0;
        pub const DTYPE_F64: DType = 1;
        pub const DTYPE_I32: DType = 2;
        pub const DTYPE_I64: DType = 3;
        /// Any type implements trait `DTypeOf` can derive a dynamic type.
        pub trait DTypeOf {
            const DTYPE: DType;
        }
        pub struct F32;
        #[automatically_derived]
        impl ::core::fmt::Debug for F32 {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::write_str(f, "F32")
            }
        }
        #[automatically_derived]
        #[doc(hidden)]
        unsafe impl ::core::clone::TrivialClone for F32 {}
        #[automatically_derived]
        impl ::core::clone::Clone for F32 {
            #[inline]
            fn clone(&self) -> F32 {
                *self
            }
        }
        #[automatically_derived]
        impl ::core::marker::Copy for F32 {}
        impl DTypeOf for F32 {
            const DTYPE: DType = DTYPE_F32;
        }
        pub struct F64;
        #[automatically_derived]
        impl ::core::fmt::Debug for F64 {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::write_str(f, "F64")
            }
        }
        #[automatically_derived]
        #[doc(hidden)]
        unsafe impl ::core::clone::TrivialClone for F64 {}
        #[automatically_derived]
        impl ::core::clone::Clone for F64 {
            #[inline]
            fn clone(&self) -> F64 {
                *self
            }
        }
        #[automatically_derived]
        impl ::core::marker::Copy for F64 {}
        impl DTypeOf for F64 {
            const DTYPE: DType = DTYPE_F64;
        }
        pub struct I32;
        #[automatically_derived]
        impl ::core::fmt::Debug for I32 {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::write_str(f, "I32")
            }
        }
        #[automatically_derived]
        #[doc(hidden)]
        unsafe impl ::core::clone::TrivialClone for I32 {}
        #[automatically_derived]
        impl ::core::clone::Clone for I32 {
            #[inline]
            fn clone(&self) -> I32 {
                *self
            }
        }
        #[automatically_derived]
        impl ::core::marker::Copy for I32 {}
        impl DTypeOf for I32 {
            const DTYPE: DType = DTYPE_I32;
        }
        pub struct I64;
        #[automatically_derived]
        impl ::core::fmt::Debug for I64 {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::write_str(f, "I64")
            }
        }
        #[automatically_derived]
        #[doc(hidden)]
        unsafe impl ::core::clone::TrivialClone for I64 {}
        #[automatically_derived]
        impl ::core::clone::Clone for I64 {
            #[inline]
            fn clone(&self) -> I64 {
                *self
            }
        }
        #[automatically_derived]
        impl ::core::marker::Copy for I64 {}
        impl DTypeOf for I64 {
            const DTYPE: DType = DTYPE_I64;
        }
        pub struct Any;
        #[automatically_derived]
        impl ::core::fmt::Debug for Any {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::write_str(f, "Any")
            }
        }
        #[automatically_derived]
        #[doc(hidden)]
        unsafe impl ::core::clone::TrivialClone for Any {}
        #[automatically_derived]
        impl ::core::clone::Clone for Any {
            #[inline]
            fn clone(&self) -> Any {
                *self
            }
        }
        #[automatically_derived]
        impl ::core::marker::Copy for Any {}
        /// Trait `DTypeImpl` associate a pair (Impl, Generic param type) with a tensor prototype.
        pub trait DTypeImpl<I: Impl> {
            type Prototype: TensorPrototype<I>;
        }
    }
    pub mod impls {
        pub trait Impl {
            type Device: Default;
        }
    }
    pub mod init {
        //! This mod (init) defines how to init a tensor.
        //!
        //! List:
        //! - TensorAllocInit: alloc
        //! - TensorZerosInit: zeros
        //! - TensorOnesInit: ones
        //! - TensorRawDataInit: from_raw.
        use crate::common::io::{Literal, TensorRawData};
        use crate::common::{DTypeImpl, Impl, Shape, Tensor};
        /// Tensor alloc init implementation.
        pub trait TensorAllocInit<I: Impl>: DTypeImpl<I> {
            fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
        }
        impl<I: Impl, TI: DTypeImpl<I> + TensorAllocInit<I>> Tensor<I, TI> {
            /// Allocate a tensor without any initial data.
            ///
            /// Alloc just allocates a memory for a tensor, which may contain dirty data.
            pub fn alloc(shape: &Shape, device: &I::Device) -> Self {
                Tensor::new(TI::init(shape, device))
            }
        }
        /// Tensor zeros init implementation.
        pub trait TensorZerosInit<I: Impl>: DTypeImpl<I> {
            fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
        }
        impl<I: Impl, TI: DTypeImpl<I> + TensorZerosInit<I>> Tensor<I, TI> {
            /// Create a tensor with zero data.
            pub fn zeros(shape: &Shape, device: &I::Device) -> Self {
                Tensor::new(TI::init(shape, device))
            }
        }
        /// Tensor ones init implementation.
        pub trait TensorOnesInit<I: Impl>: DTypeImpl<I> {
            fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
        }
        impl<I: Impl, TI: DTypeImpl<I> + TensorOnesInit<I>> Tensor<I, TI> {
            /// Create a tensor with one data.
            pub fn ones(shape: &Shape, device: &I::Device) -> Self {
                Tensor::new(TI::init(shape, device))
            }
        }
        /// Tensor raw data init implementation.
        pub trait TensorRawDataInit<I: Impl>: DTypeImpl<I> {
            fn init(
                data: impl Into<TensorRawData>,
                device: &I::Device,
            ) -> Self::Prototype;
        }
        impl<I: Impl, TI: DTypeImpl<I> + TensorRawDataInit<I>> Tensor<I, TI> {
            /// Create a tensor from raw data.
            pub fn from_raw(data: impl Into<TensorRawData>, device: &I::Device) -> Self {
                Tensor::new(TI::init(data, device))
            }
            /// Create a tensor from literal.
            pub fn from_literal(literal: impl Literal, device: &I::Device) -> Self {
                Self::from_raw(literal, device)
            }
        }
    }
    pub mod io {
        use crate::common::dtype::{F32, F64, I32, I64};
        use crate::common::{DType, DTypeOf, Shape};
        use std::io::Read;
        /// `TensorRawData` hold a reference of raw data (byte source), which can be used to init a tensor.
        ///
        /// The byte source must follow the rules declared by `DType`, e.g. endian, size, and alignment.
        pub struct TensorRawData {
            pub source: Box<dyn Read>,
            pub shape: Shape,
            pub dtype: DType,
        }
        /// Rust scalars and rust arrays with scalars can be considered as a literal.
        pub trait Literal {
            const DTYPE: DType;
            fn shape(&self) -> Shape;
            fn data(&self) -> Vec<u8>;
        }
        impl<T: Literal> From<T> for TensorRawData {
            fn from(value: T) -> Self {
                let shape = value.shape();
                let data = value.data();
                TensorRawData {
                    source: Box::new(std::io::Cursor::new(data)),
                    shape,
                    dtype: T::DTYPE,
                }
            }
        }
        impl Literal for f32 {
            const DTYPE: DType = <F32 as DTypeOf>::DTYPE;
            fn shape(&self) -> Shape {
                Shape::scalar()
            }
            fn data(&self) -> Vec<u8> {
                Vec::from(self.to_le_bytes())
            }
        }
        impl Literal for f64 {
            const DTYPE: DType = <F64 as DTypeOf>::DTYPE;
            fn shape(&self) -> Shape {
                Shape::scalar()
            }
            fn data(&self) -> Vec<u8> {
                Vec::from(self.to_le_bytes())
            }
        }
        impl Literal for i32 {
            const DTYPE: DType = <I32 as DTypeOf>::DTYPE;
            fn shape(&self) -> Shape {
                Shape::scalar()
            }
            fn data(&self) -> Vec<u8> {
                Vec::from(self.to_le_bytes())
            }
        }
        impl Literal for i64 {
            const DTYPE: DType = <I64 as DTypeOf>::DTYPE;
            fn shape(&self) -> Shape {
                Shape::scalar()
            }
            fn data(&self) -> Vec<u8> {
                Vec::from(self.to_le_bytes())
            }
        }
        impl<T: Literal, const N: usize> Literal for [T; N] {
            const DTYPE: DType = T::DTYPE;
            fn shape(&self) -> Shape {
                if !(N > 0) {
                    {
                        ::core::panicking::panic_fmt(
                            format_args!("Empty literal isn\'t supported."),
                        );
                    }
                }
                let mut shape_v = <[_]>::into_vec(::alloc::boxed::box_new([N]));
                shape_v.extend(self[0].shape().iter());
                Shape::from(shape_v)
            }
            fn data(&self) -> Vec<u8> {
                let mut vec = Vec::new();
                for item in self {
                    vec.extend(item.data());
                }
                vec
            }
        }
    }
    pub mod ops {
        pub mod binary_ops {
            //! This mod (binary ops) defines binary operations.
            //!
            //! List:
            //! - TensorAdd: +
            //! - TensorSub: -
            //! - TensorMul: *
            //! - TensorDiv: /
            use crate::common::{DTypeImpl, Impl, Tensor};
            /// Tensor add implementation.
            pub trait TensorAdd<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
                type Output: DTypeImpl<I>;
                fn add(
                    lhs: &Self::Prototype,
                    rhs: &Rhs::Prototype,
                ) -> <Self::Output as DTypeImpl<I>>::Prototype;
            }
            impl<
                I: Impl,
                Lhs: DTypeImpl<I> + TensorAdd<I, Rhs>,
                Rhs: DTypeImpl<I>,
            > std::ops::Add<&Tensor<I, Rhs>> for &Tensor<I, Lhs> {
                type Output = Tensor<I, Lhs::Output>;
                fn add(self, rhs: &Tensor<I, Rhs>) -> Self::Output {
                    Tensor::new(Lhs::add(&self.prototype, &rhs.prototype))
                }
            }
            /// Tensor sub implementation.
            pub trait TensorSub<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
                type Output: DTypeImpl<I>;
                fn sub(
                    lhs: &Self::Prototype,
                    rhs: &Rhs::Prototype,
                ) -> <Self::Output as DTypeImpl<I>>::Prototype;
            }
            impl<
                I: Impl,
                Lhs: DTypeImpl<I> + TensorSub<I, Rhs>,
                Rhs: DTypeImpl<I>,
            > std::ops::Sub<&Tensor<I, Rhs>> for &Tensor<I, Lhs> {
                type Output = Tensor<I, Lhs::Output>;
                fn sub(self, rhs: &Tensor<I, Rhs>) -> Self::Output {
                    Tensor::new(Lhs::sub(&self.prototype, &rhs.prototype))
                }
            }
            /// Tensor mul implementation.
            pub trait TensorMul<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
                type Output: DTypeImpl<I>;
                fn mul(
                    lhs: &Self::Prototype,
                    rhs: &Rhs::Prototype,
                ) -> <Self::Output as DTypeImpl<I>>::Prototype;
            }
            impl<
                I: Impl,
                Lhs: DTypeImpl<I> + TensorMul<I, Rhs>,
                Rhs: DTypeImpl<I>,
            > std::ops::Mul<&Tensor<I, Rhs>> for &Tensor<I, Lhs> {
                type Output = Tensor<I, Lhs::Output>;
                fn mul(self, rhs: &Tensor<I, Rhs>) -> Self::Output {
                    Tensor::new(Lhs::mul(&self.prototype, &rhs.prototype))
                }
            }
            /// Tensor div implementation.
            pub trait TensorDiv<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
                type Output: DTypeImpl<I>;
                fn div(
                    lhs: &Self::Prototype,
                    rhs: &Rhs::Prototype,
                ) -> <Self::Output as DTypeImpl<I>>::Prototype;
            }
            impl<
                I: Impl,
                Lhs: DTypeImpl<I> + TensorDiv<I, Rhs>,
                Rhs: DTypeImpl<I>,
            > std::ops::Div<&Tensor<I, Rhs>> for &Tensor<I, Lhs> {
                type Output = Tensor<I, Lhs::Output>;
                fn div(self, rhs: &Tensor<I, Rhs>) -> Self::Output {
                    Tensor::new(Lhs::div(&self.prototype, &rhs.prototype))
                }
            }
        }
        pub mod cast {
            //! This mod (cast ops) defines operations with cast & copy.
            //!
            //! All operations may copy data.
            //!
            //! List:
            //! - TensorCast: cast
            //! - TensorCopy: copy
            //! - TensorReshape: reshape
            use crate::common::{DTypeImpl, Impl, Shape, Tensor};
            /// Tensor cast implementation.
            pub trait TensorCast<I: Impl, Dst: DTypeImpl<I>>: DTypeImpl<I> {
                fn cast(src: &Self::Prototype) -> Dst::Prototype;
            }
            impl<I: Impl, Src: DTypeImpl<I>> Tensor<I, Src> {
                /// Return a tensor with type `Dst` from the tensor with type `Src`.
                ///
                /// If `Dst` is not equal to `Src`, it may create a new tensor.
                pub fn cast<Dst: DTypeImpl<I>>(&self) -> Tensor<I, Dst>
                where
                    Src: TensorCast<I, Dst>,
                {
                    Tensor::new(Src::cast(&self.prototype))
                }
            }
            /// Tensor copy implementation.
            pub trait TensorCopy<I: Impl, Src: DTypeImpl<I>>: DTypeImpl<I> {
                fn copy(dst: &mut Self::Prototype, src: &Src::Prototype);
            }
            impl<I: Impl, Dst: DTypeImpl<I>> Tensor<I, Dst> {
                /// Copy data from the tensor `src`.
                pub fn copy<Src: DTypeImpl<I>>(&mut self, src: &Src::Prototype)
                where
                    Dst: TensorCopy<I, Src>,
                {
                    Dst::copy(&mut self.prototype, src);
                }
            }
            /// Tensor reshape implementation.
            pub trait TensorReshape<I: Impl>: DTypeImpl<I> {
                fn reshape(src: &Self::Prototype, new_shape: &Shape) -> Self::Prototype;
            }
            impl<I: Impl, Src: DTypeImpl<I> + TensorReshape<I>> Tensor<I, Src> {
                /// Return a tensor with the same data copyed from its original data, but with the specified shape `new_shape`.
                ///
                /// If it can be viewed, it may not create a new tensor.
                /// The number of elements must be equal.
                pub fn reshape(&self, new_shape: &Shape) -> Self {
                    Tensor::new(Src::reshape(&self.prototype, new_shape))
                }
            }
        }
        pub mod promote {
            //! This mod (promote) declares how two types (lhs, rhs) to promote type.
            use crate::common::dtype::{F32, F64, I32, I64};
            /// Trait `Promote` declares that `Self` and `Rhs` should promote to `Promote<Rhs>::Output` when they are calculating.
            pub trait Promote<Rhs> {
                type Output;
            }
            impl Promote<F32> for F32 {
                type Output = F32;
            }
            impl Promote<F64> for F32 {
                type Output = F64;
            }
            impl Promote<I32> for F32 {
                type Output = F32;
            }
            impl Promote<I64> for F32 {
                type Output = F32;
            }
            impl Promote<F32> for F64 {
                type Output = F64;
            }
            impl Promote<F64> for F64 {
                type Output = F64;
            }
            impl Promote<I32> for F64 {
                type Output = F64;
            }
            impl Promote<I64> for F64 {
                type Output = F64;
            }
            impl Promote<F32> for I32 {
                type Output = F32;
            }
            impl Promote<F64> for I32 {
                type Output = F64;
            }
            impl Promote<I32> for I32 {
                type Output = I32;
            }
            impl Promote<I64> for I32 {
                type Output = I64;
            }
            impl Promote<F32> for I64 {
                type Output = F32;
            }
            impl Promote<F64> for I64 {
                type Output = F64;
            }
            impl Promote<I32> for I64 {
                type Output = I64;
            }
            impl Promote<I64> for I64 {
                type Output = I64;
            }
        }
        pub mod view {
            //! This mod (view ops) defines operations with view.
            //!
            //! All operations will not copy data, but just reinterpret the original data.
            //!
            //! List:
            //! - TensorView: view
            //! - TensorSlice: slice
            //! - TensorBroadcast: broadcast_to
            use crate::common::shape::broadcast_shape;
            use crate::common::{DTypeImpl, Impl, Shape, Tensor, TensorPrototype};
            use std::ops::{
                Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
            };
            /// Slice type of tensor.
            pub enum TensorIndex {
                /// i
                Index(isize),
                /// start..end
                Range(isize, isize),
                /// start..
                RangeFrom(isize),
                /// ..end
                RangeTo(isize),
                /// ..
                RangeFull,
                /// start..=end
                RangeInclusive(isize, isize),
                /// ..=end
                RangeToInclusive(isize),
                /// expand a new dim.
                Expand,
                /// ...
                Full,
            }
            #[automatically_derived]
            impl ::core::fmt::Debug for TensorIndex {
                #[inline]
                fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                    match self {
                        TensorIndex::Index(__self_0) => {
                            ::core::fmt::Formatter::debug_tuple_field1_finish(
                                f,
                                "Index",
                                &__self_0,
                            )
                        }
                        TensorIndex::Range(__self_0, __self_1) => {
                            ::core::fmt::Formatter::debug_tuple_field2_finish(
                                f,
                                "Range",
                                __self_0,
                                &__self_1,
                            )
                        }
                        TensorIndex::RangeFrom(__self_0) => {
                            ::core::fmt::Formatter::debug_tuple_field1_finish(
                                f,
                                "RangeFrom",
                                &__self_0,
                            )
                        }
                        TensorIndex::RangeTo(__self_0) => {
                            ::core::fmt::Formatter::debug_tuple_field1_finish(
                                f,
                                "RangeTo",
                                &__self_0,
                            )
                        }
                        TensorIndex::RangeFull => {
                            ::core::fmt::Formatter::write_str(f, "RangeFull")
                        }
                        TensorIndex::RangeInclusive(__self_0, __self_1) => {
                            ::core::fmt::Formatter::debug_tuple_field2_finish(
                                f,
                                "RangeInclusive",
                                __self_0,
                                &__self_1,
                            )
                        }
                        TensorIndex::RangeToInclusive(__self_0) => {
                            ::core::fmt::Formatter::debug_tuple_field1_finish(
                                f,
                                "RangeToInclusive",
                                &__self_0,
                            )
                        }
                        TensorIndex::Expand => {
                            ::core::fmt::Formatter::write_str(f, "Expand")
                        }
                        TensorIndex::Full => ::core::fmt::Formatter::write_str(f, "Full"),
                    }
                }
            }
            #[automatically_derived]
            #[doc(hidden)]
            unsafe impl ::core::clone::TrivialClone for TensorIndex {}
            #[automatically_derived]
            impl ::core::clone::Clone for TensorIndex {
                #[inline]
                fn clone(&self) -> TensorIndex {
                    let _: ::core::clone::AssertParamIsClone<isize>;
                    *self
                }
            }
            #[automatically_derived]
            impl ::core::marker::Copy for TensorIndex {}
            #[automatically_derived]
            impl ::core::marker::StructuralPartialEq for TensorIndex {}
            #[automatically_derived]
            impl ::core::cmp::PartialEq for TensorIndex {
                #[inline]
                fn eq(&self, other: &TensorIndex) -> bool {
                    let __self_discr = ::core::intrinsics::discriminant_value(self);
                    let __arg1_discr = ::core::intrinsics::discriminant_value(other);
                    __self_discr == __arg1_discr
                        && match (self, other) {
                            (
                                TensorIndex::Index(__self_0),
                                TensorIndex::Index(__arg1_0),
                            ) => __self_0 == __arg1_0,
                            (
                                TensorIndex::Range(__self_0, __self_1),
                                TensorIndex::Range(__arg1_0, __arg1_1),
                            ) => __self_0 == __arg1_0 && __self_1 == __arg1_1,
                            (
                                TensorIndex::RangeFrom(__self_0),
                                TensorIndex::RangeFrom(__arg1_0),
                            ) => __self_0 == __arg1_0,
                            (
                                TensorIndex::RangeTo(__self_0),
                                TensorIndex::RangeTo(__arg1_0),
                            ) => __self_0 == __arg1_0,
                            (
                                TensorIndex::RangeInclusive(__self_0, __self_1),
                                TensorIndex::RangeInclusive(__arg1_0, __arg1_1),
                            ) => __self_0 == __arg1_0 && __self_1 == __arg1_1,
                            (
                                TensorIndex::RangeToInclusive(__self_0),
                                TensorIndex::RangeToInclusive(__arg1_0),
                            ) => __self_0 == __arg1_0,
                            _ => true,
                        }
                }
            }
            #[automatically_derived]
            impl ::core::cmp::Eq for TensorIndex {
                #[inline]
                #[doc(hidden)]
                #[coverage(off)]
                fn assert_receiver_is_total_eq(&self) -> () {
                    let _: ::core::cmp::AssertParamIsEq<isize>;
                }
            }
            impl From<isize> for TensorIndex {
                fn from(value: isize) -> Self {
                    TensorIndex::Index(value)
                }
            }
            impl From<Range<isize>> for TensorIndex {
                fn from(value: Range<isize>) -> Self {
                    TensorIndex::Range(value.start, value.end)
                }
            }
            impl From<RangeFrom<isize>> for TensorIndex {
                fn from(value: RangeFrom<isize>) -> Self {
                    TensorIndex::RangeFrom(value.start)
                }
            }
            impl From<RangeTo<isize>> for TensorIndex {
                fn from(value: RangeTo<isize>) -> Self {
                    TensorIndex::RangeTo(value.end)
                }
            }
            impl From<RangeFull> for TensorIndex {
                fn from(_: RangeFull) -> Self {
                    TensorIndex::RangeFull
                }
            }
            impl From<RangeInclusive<isize>> for TensorIndex {
                fn from(value: RangeInclusive<isize>) -> Self {
                    TensorIndex::RangeInclusive(*value.start(), *value.end())
                }
            }
            impl From<RangeToInclusive<isize>> for TensorIndex {
                fn from(value: RangeToInclusive<isize>) -> Self {
                    TensorIndex::RangeToInclusive(value.end)
                }
            }
            /// Tensor view implementation.
            pub trait TensorView<I: Impl>: DTypeImpl<I> {
                fn view(
                    src: &Self::Prototype,
                    new_shape: &Shape,
                ) -> Option<Self::Prototype>;
            }
            impl<I: Impl, Src: DTypeImpl<I> + TensorView<I>> Tensor<I, Src> {
                /// Return a new tensor with the same data as the tensor `&self` but of a different shape `new_shape`.
                ///
                /// For a tensor to be viewed, the new view size must be compatible with its original size and stride.
                /// In other words, the new view size must completely merge and split the contiguous subspaces derived from the tensor `&self`.
                /// If `new_shape` is not compatible with its original size and stride, it will return `None`.
                ///
                /// # Exmaples
                ///
                /// ## Contiguous tensors
                ///
                /// ```
                /// use myrustllm::cpu::tensor::CPUTensor;
                /// use myrustllm::cpu::shape::Shape;
                ///
                /// let tensor = CPUTensor::<f32>::from_shape(&Shape::new(vec![4, 5]));
                /// let viewed_tensor = tensor.view(&Shape::new(vec![2, 2, 5])).unwrap();
                /// ```
                ///
                /// ## Auto-inferred dim
                ///
                /// `0` in `new_shape` refers to an auto-inferred dim.
                ///
                /// ```
                /// use myrustllm::cpu::tensor::{Tensor, CPUTensor};
                /// use myrustllm::cpu::shape::Shape;
                ///
                /// let tensor = CPUTensor::<f32>::from_shape(&Shape::new(vec![2, 2, 5]));
                /// let viewed_tensor = tensor.view(&Shape::new(vec![0, 5]));
                ///
                /// assert_eq!(viewed_tensor.shape(), Shape::new(vec![4, 5]));
                /// ```
                pub fn view(&self, new_shape: &Shape) -> Option<Self> {
                    Some(Tensor::new(Src::view(&self.prototype, new_shape)?))
                }
            }
            /// Tensor slice implementation.
            pub trait TensorSlice<I: Impl>: DTypeImpl<I> {
                fn slice(
                    src: &Self::Prototype,
                    indices: &[TensorIndex],
                ) -> Self::Prototype;
            }
            impl<I: Impl, Src: DTypeImpl<I> + TensorSlice<I>> Tensor<I, Src> {
                /// Extract a new slice(view) derived from `&self`.
                ///
                /// The returned CPU tensor will share the same memory with `&self`.
                ///
                /// # Examples
                ///
                /// ```
                /// use myrustllm::cpu::tensor::CPUTensor;
                /// use myrustllm::cpu::slice::TensorIndex;
                /// use myrustllm::idx;
                ///
                /// let tensor = CPUTensor::from_literal([
                ///     [1.0, 2.0],
                ///     [3.0, 4.0]
                /// ]);
                ///
                /// let sliced_tensor = tensor.slice(&idx!(.., 1));
                /// let a = sliced_tensor.slice(&idx!(0));
                /// let b = sliced_tensor.slice(&idx!(-1));
                ///
                /// assert_eq!(a.into_scalar(), 2.0);
                /// assert_eq!(b.into_scalar(), 4.0);
                /// ```
                pub fn slice(&self, indices: &[TensorIndex]) -> Self {
                    Tensor::new(Src::slice(&self.prototype, indices))
                }
            }
            /// Tensor broadcast implementation.
            pub trait TensorBroadcast<I: Impl>: DTypeImpl<I> {
                fn broadcast_to(
                    src: &Self::Prototype,
                    target_shape: &Shape,
                ) -> Option<Self::Prototype>;
            }
            impl<I: Impl, Src: DTypeImpl<I> + TensorBroadcast<I>> Tensor<I, Src> {
                /// Broadcast `&self` to the shape `target_shape`.
                ///
                /// Broadcasting doesn't mean allocating new memory, but only creates a new view on the existing tensor.
                /// If the shape `target_shape` is not compatible with the shape of its original shape, it will return `None`.
                ///
                /// # Examples
                ///
                /// ```
                /// use myrustllm::cpu::tensor::CPUTensor;
                /// use myrustllm::cpu::shape::Shape;
                ///
                /// // 1 -> 5
                /// let tensor = CPUTensor::<f32>::from_shape(&Shape::new(vec![2,1,2,3]));
                /// let broadcast_tensor = tensor.broadcast_to(&Shape::new(vec![2,5,2,3])).unwrap();
                /// ```
                pub fn broadcast_to(&self, target_shape: &Shape) -> Option<Self> {
                    Some(Tensor::new(Src::broadcast_to(&self.prototype, target_shape)?))
                }
            }
            /// Try broadcast two tensor prototypes mutually.
            pub fn broadcast_prot<
                I: Impl,
                Lhs: DTypeImpl<I> + TensorBroadcast<I>,
                Rhs: DTypeImpl<I> + TensorBroadcast<I>,
            >(
                lhs: &Lhs::Prototype,
                rhs: &Rhs::Prototype,
            ) -> Option<(Lhs::Prototype, Rhs::Prototype)> {
                let target_shape: Shape = broadcast_shape(&lhs.shape(), &rhs.shape())?;
                Some((
                    Lhs::broadcast_to(lhs, &target_shape)?,
                    Rhs::broadcast_to(rhs, &target_shape)?,
                ))
            }
            /// Try broadcast two tensors mutually.
            pub fn broadcast<
                I: Impl,
                Lhs: DTypeImpl<I> + TensorBroadcast<I>,
                Rhs: DTypeImpl<I> + TensorBroadcast<I>,
            >(
                lhs: &Tensor<I, Lhs>,
                rhs: &Tensor<I, Rhs>,
            ) -> Option<(Tensor<I, Lhs>, Tensor<I, Rhs>)> {
                let (lhs, rhs) = broadcast_prot::<
                    I,
                    Lhs,
                    Rhs,
                >(&lhs.prototype, &rhs.prototype)?;
                Some((Tensor::new(lhs), Tensor::new(rhs)))
            }
        }
    }
    pub mod shape {
        use std::ops::Deref;
        use std::rc::Rc;
        /// Metadata for shape and stride.
        pub struct Shape(Rc<[usize]>);
        #[automatically_derived]
        impl ::core::fmt::Debug for Shape {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Shape", &&self.0)
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for Shape {
            #[inline]
            fn clone(&self) -> Shape {
                Shape(::core::clone::Clone::clone(&self.0))
            }
        }
        #[automatically_derived]
        impl ::core::marker::StructuralPartialEq for Shape {}
        #[automatically_derived]
        impl ::core::cmp::PartialEq for Shape {
            #[inline]
            fn eq(&self, other: &Shape) -> bool {
                self.0 == other.0
            }
        }
        #[automatically_derived]
        impl ::core::cmp::Eq for Shape {
            #[inline]
            #[doc(hidden)]
            #[coverage(off)]
            fn assert_receiver_is_total_eq(&self) -> () {
                let _: ::core::cmp::AssertParamIsEq<Rc<[usize]>>;
            }
        }
        impl Shape {
            /// Create a new shape for scalar.
            pub fn scalar() -> Self {
                Shape(Rc::new([]))
            }
            /// Return the number of elements along the shape.
            pub fn numel(&self) -> usize {
                self.iter().product()
            }
            /// Is the shape for scalar?
            pub fn is_scalar(&self) -> bool {
                self.is_empty()
            }
        }
        impl Deref for Shape {
            type Target = [usize];
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl<T: IntoIterator<Item = usize>> From<T> for Shape {
            fn from(value: T) -> Self {
                Shape(value.into_iter().collect())
            }
        }
        /// Create a contiguous stride along the shape.
        ///
        /// # Example
        ///
        /// ```
        /// use myrustllm::cpu::shape::{Shape, create_contiguous_stride};
        ///
        /// let shape = Shape::new(vec![2, 3, 4]);
        /// let contiguous_stride = create_contiguous_stride(&shape);
        /// assert_eq!(contiguous_stride, Shape::new(vec![12, 4, 1]));
        /// ```
        ///
        /// For the above example, 1 = 1, 4 = 1 * 4, 12 = 3 * 4.
        pub fn create_contiguous_stride(shape: &Shape) -> Shape {
            let mut stride_v = ::alloc::vec::from_elem(1, shape.len());
            if !shape.is_scalar() {
                for i in (0..shape.len() - 1).rev() {
                    stride_v[i] = stride_v[i + 1] * shape[i + 1];
                }
            }
            stride_v.into()
        }
        pub fn broadcast_shape(shape_a: &Shape, shape_b: &Shape) -> Option<Shape> {
            let max_dims = std::cmp::max(shape_a.len(), shape_b.len());
            let mut result_shape_v = Vec::with_capacity(max_dims);
            for i in 0..max_dims {
                let d_a = if i < shape_a.len() {
                    shape_a[shape_a.len() - 1 - i]
                } else {
                    1
                };
                let d_b = if i < shape_b.len() {
                    shape_b[shape_b.len() - 1 - i]
                } else {
                    1
                };
                if d_a != d_b && d_a != 1 && d_b != 1 {
                    return None;
                }
                result_shape_v.push(std::cmp::max(d_a, d_b));
            }
            result_shape_v.reverse();
            return Some(result_shape_v.into());
        }
    }
    pub mod tensor {
        use crate::common::dtype::F32;
        use crate::common::{DType, DTypeImpl, Impl, Shape};
        use crate::cpu::impls::CPU;
        /// Tensor is the basic data type in MyRustLLM.
        ///
        /// A Tensor is implemented by a pair(Impl, DTypeImpl<Impl>), where `Impl` is the backend implementation and `DTypeImpl<Impl>` is the data type of the tensor.
        pub struct Tensor<I: Impl = CPU, TI: DTypeImpl<I> = F32> {
            pub prototype: TI::Prototype,
        }
        impl<I: Impl, TI: DTypeImpl<I>> Tensor<I, TI> {
            /// Create a tensor from a prototype.
            ///
            /// If you aren't a lib developer, you should use e.g. alloc, ones, ... to create a tensor instead of `new`.
            pub fn new(prototype: TI::Prototype) -> Self {
                Self { prototype }
            }
        }
        impl<I: Impl, TI: DTypeImpl<I>> TensorPrototype<I> for Tensor<I, TI> {
            fn shape(&self) -> Shape {
                self.prototype.shape()
            }
            fn dtype(&self) -> DType {
                self.prototype.dtype()
            }
            fn device(&self) -> I::Device {
                self.prototype.device()
            }
        }
        /// Trait `TensorPrototype` defines three basic functions for tensor prototypes:
        /// - shape(): Return the shape of the tensor.
        /// - dtype(): Return the dtype of the tensor.
        /// - device(): Return the device of the tensor.
        ///
        /// Any tensor prototype should implement `TensorPrototype`.
        pub trait TensorPrototype<I: Impl> {
            /// Return the shape of `&self`.
            fn shape(&self) -> Shape;
            /// Return the dtype of the tensor.
            fn dtype(&self) -> DType;
            /// Return the dtype of the tensor.
            fn device(&self) -> I::Device;
            /// Return the dimension of `&self`.
            fn dims(&self) -> usize {
                self.shape().len()
            }
            /// Is `&self` a scalar?
            fn is_scalar(&self) -> bool {
                self.shape().is_scalar()
            }
            /// Return the number of elements.
            fn numel(&self) -> usize {
                self.shape().numel()
            }
        }
        /// Tensor metadata is a tuple of (Shape, DType, Device).
        pub type TensorMetadata<I> = (Shape, DType, <I as Impl>::Device);
    }
    pub use dtype::DType;
    pub use dtype::DTypeOf;
    pub use dtype::DTypeImpl;
    pub use impls::Impl;
    pub use shape::Shape;
    pub use tensor::Tensor;
    pub use tensor::TensorPrototype;
}
pub mod cpu {
    pub mod impls {
        use crate::common::dtype::{Any, F32, F64, I32, I64};
        use crate::common::shape::create_contiguous_stride;
        use crate::common::tensor::TensorPrototype;
        use crate::common::{DType, DTypeImpl, DTypeOf, Impl, Shape};
        use crate::cpu::interface;
        use crate::cpu::interface::IntoInterface;
        use crate::cpu::mem::{CPUMemory, SharedCPUMemory};
        use std::cell::RefCell;
        use std::rc::Rc;
        /// CPU is one of the basic implementations in MyRustLLM, which provides tensor operations on CPU.
        /// All Tensors using CPU implementation will be allocated in the main memory and processed by the CPU.
        /// Tensors on CPU use shared memory to store their data, so it can be easily cloned with few costs.
        pub struct CPU {}
        impl Impl for CPU {
            /// CPU does not require any special device context, so we can use an empty tuple as the device type.
            type Device = ();
        }
        /// `CPUTensorPrototype` is the prototype of a tensor on CPU. It contains shared memory for the tensor data, as well as the shape, stride, and offset information.
        pub struct CPUTensorPrototype<T> {
            data: SharedCPUMemory<T>,
            shape: Shape,
            stride: Shape,
            offset: usize,
        }
        #[automatically_derived]
        impl<T: ::core::clone::Clone> ::core::clone::Clone for CPUTensorPrototype<T> {
            #[inline]
            fn clone(&self) -> CPUTensorPrototype<T> {
                CPUTensorPrototype {
                    data: ::core::clone::Clone::clone(&self.data),
                    shape: ::core::clone::Clone::clone(&self.shape),
                    stride: ::core::clone::Clone::clone(&self.stride),
                    offset: ::core::clone::Clone::clone(&self.offset),
                }
            }
        }
        #[automatically_derived]
        impl<T: ::core::fmt::Debug> ::core::fmt::Debug for CPUTensorPrototype<T> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field4_finish(
                    f,
                    "CPUTensorPrototype",
                    "data",
                    &self.data,
                    "shape",
                    &self.shape,
                    "stride",
                    &self.stride,
                    "offset",
                    &&self.offset,
                )
            }
        }
        impl<T> CPUTensorPrototype<T> {
            /// Create a new CPU tensor prototype with the given shared data, shape, stride, and offset.
            pub fn new(
                data: SharedCPUMemory<T>,
                shape: &Shape,
                stride: &Shape,
                offset: usize,
            ) -> Self {
                CPUTensorPrototype {
                    data,
                    shape: shape.clone(),
                    stride: stride.clone(),
                    offset,
                }
            }
            /// Allocate a new CPU tensor prototype with the given shape. The data won't be initialized, and the caller should fill it with valid data before using it.
            pub fn alloc(shape: &Shape) -> Self {
                CPUTensorPrototype {
                    data: Rc::new(RefCell::new(CPUMemory::new(shape.numel()))),
                    shape: shape.clone(),
                    stride: create_contiguous_stride(shape),
                    offset: 0,
                }
            }
            /// Return the stride of the tensor.
            pub fn stride(&self) -> Shape {
                self.stride.clone()
            }
            /// Return the offset of the tensor.
            pub fn offset(&self) -> usize {
                self.offset
            }
            /// Return a new reference to the shared memory of the tensor data.
            pub fn data(&self) -> SharedCPUMemory<T> {
                self.data.clone()
            }
        }
        impl TensorPrototype<CPU> for CPUTensorPrototype<f32> {
            fn shape(&self) -> Shape {
                self.shape.clone()
            }
            fn dtype(&self) -> DType {
                <F32 as DTypeOf>::DTYPE
            }
            fn device(&self) -> <CPU as Impl>::Device {
                Default::default()
            }
        }
        impl DTypeImpl<CPU> for F32 {
            type Prototype = CPUTensorPrototype<f32>;
        }
        impl TensorPrototype<CPU> for CPUTensorPrototype<f64> {
            fn shape(&self) -> Shape {
                self.shape.clone()
            }
            fn dtype(&self) -> DType {
                <F64 as DTypeOf>::DTYPE
            }
            fn device(&self) -> <CPU as Impl>::Device {
                Default::default()
            }
        }
        impl DTypeImpl<CPU> for F64 {
            type Prototype = CPUTensorPrototype<f64>;
        }
        impl TensorPrototype<CPU> for CPUTensorPrototype<i32> {
            fn shape(&self) -> Shape {
                self.shape.clone()
            }
            fn dtype(&self) -> DType {
                <I32 as DTypeOf>::DTYPE
            }
            fn device(&self) -> <CPU as Impl>::Device {
                Default::default()
            }
        }
        impl DTypeImpl<CPU> for I32 {
            type Prototype = CPUTensorPrototype<i32>;
        }
        impl TensorPrototype<CPU> for CPUTensorPrototype<i64> {
            fn shape(&self) -> Shape {
                self.shape.clone()
            }
            fn dtype(&self) -> DType {
                <I64 as DTypeOf>::DTYPE
            }
            fn device(&self) -> <CPU as Impl>::Device {
                Default::default()
            }
        }
        impl DTypeImpl<CPU> for I64 {
            type Prototype = CPUTensorPrototype<i64>;
        }
        impl<T> IntoInterface for CPUTensorPrototype<T>
        where
            CPUTensorPrototype<T>: TensorPrototype<CPU>,
        {
            unsafe fn into_interface(&self) -> interface::CPUTensor {
                interface::CPUTensor {
                    data: unsafe {
                        self.data.borrow_mut().as_mut_ptr().add(self.offset) as *mut u8
                    },
                    shape: self.shape.as_ptr(),
                    stride: self.stride.as_ptr(),
                    dims: self.dims(),
                    dtype: self.dtype(),
                }
            }
        }
        /// `CPUTensorAnyPrototype` is a type-erased version of `CPUTensorPrototype<T>`, which can hold any data type.
        pub struct CPUTensorAnyPrototype {
            inner: CPUTensorPrototype<u8>,
            dtype: DType,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for CPUTensorAnyPrototype {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field2_finish(
                    f,
                    "CPUTensorAnyPrototype",
                    "inner",
                    &self.inner,
                    "dtype",
                    &&self.dtype,
                )
            }
        }
        #[automatically_derived]
        impl ::core::clone::Clone for CPUTensorAnyPrototype {
            #[inline]
            fn clone(&self) -> CPUTensorAnyPrototype {
                CPUTensorAnyPrototype {
                    inner: ::core::clone::Clone::clone(&self.inner),
                    dtype: ::core::clone::Clone::clone(&self.dtype),
                }
            }
        }
        impl<T> From<CPUTensorPrototype<T>> for CPUTensorAnyPrototype
        where
            CPUTensorPrototype<T>: TensorPrototype<CPU>,
        {
            fn from(value: CPUTensorPrototype<T>) -> Self {
                let dtype = value.dtype();
                CPUTensorAnyPrototype {
                    inner: unsafe { std::mem::transmute(value) },
                    dtype,
                }
            }
        }
        impl TensorPrototype<CPU> for CPUTensorAnyPrototype {
            fn shape(&self) -> Shape {
                self.inner.shape.clone()
            }
            fn dtype(&self) -> DType {
                self.dtype
            }
            fn device(&self) -> <CPU as Impl>::Device {
                Default::default()
            }
        }
        impl DTypeImpl<CPU> for Any {
            type Prototype = CPUTensorAnyPrototype;
        }
    }
    pub mod init {
        use crate::common::dtype::{Any, DTYPE_F32};
        use crate::common::init::{
            TensorAllocInit, TensorOnesInit, TensorRawDataInit, TensorZerosInit,
        };
        use crate::common::io::TensorRawData;
        use crate::common::ops::cast::TensorCast;
        use crate::common::shape::create_contiguous_stride;
        use crate::common::{DTypeImpl, Impl, Shape};
        use crate::cpu::impls::{CPU, CPUTensorAnyPrototype, CPUTensorPrototype};
        use crate::cpu::interface::IntoInterface;
        use crate::cpu::mem::CPUMemory;
        use num_traits::{One, Zero};
        impl<
            T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>,
            U,
        > TensorAllocInit<CPU> for T {
            fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
                CPUTensorPrototype::alloc(shape)
            }
        }
        impl<T> CPUTensorPrototype<T> {
            pub fn fill(value: T, shape: &Shape) -> Self {
                ::core::panicking::panic("not yet implemented")
            }
        }
        impl<
            T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>,
            U: Zero,
        > TensorZerosInit<CPU> for T {
            fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
                CPUTensorPrototype::fill(U::zero(), shape)
            }
        }
        impl<
            T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>,
            U: One,
        > TensorOnesInit<CPU> for T {
            fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
                CPUTensorPrototype::fill(U::one(), shape)
            }
        }
        impl TensorRawDataInit<CPU> for Any {
            fn init(
                data: impl Into<TensorRawData>,
                _device: &<CPU as Impl>::Device,
            ) -> Self::Prototype {
                let mut data = data.into();
                let shape = data.shape;
                match data.dtype {
                    DTYPE_F32 => {
                        let mut mem: CPUMemory<f32> = CPUMemory::new(shape.numel());
                        if let Err(e) = data
                            .source
                            .read(unsafe {
                                std::slice::from_raw_parts_mut(
                                    mem.as_mut_ptr() as *mut u8,
                                    mem.size() * size_of::<f32>(),
                                )
                            })
                        {
                            {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "Cannot create a tensor from raw data: {0}.",
                                        e,
                                    ),
                                );
                            };
                        }
                        CPUTensorAnyPrototype::from(
                            CPUTensorPrototype::new(
                                mem.into(),
                                &shape,
                                &create_contiguous_stride(&shape),
                                0,
                            ),
                        )
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!("DType {0} is not supported.", data.dtype),
                        );
                    }
                }
            }
        }
    }
    pub mod interface {
        pub type DType = u8;
        #[repr(C)]
        pub struct CPUTensor {
            pub data: *mut u8,
            pub shape: *const usize,
            pub stride: *const usize,
            pub dims: usize,
            pub dtype: DType,
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for CPUTensor {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field5_finish(
                    f,
                    "CPUTensor",
                    "data",
                    &self.data,
                    "shape",
                    &self.shape,
                    "stride",
                    &self.stride,
                    "dims",
                    &self.dims,
                    "dtype",
                    &&self.dtype,
                )
            }
        }
        #[automatically_derived]
        impl ::core::marker::Copy for CPUTensor {}
        #[automatically_derived]
        #[doc(hidden)]
        unsafe impl ::core::clone::TrivialClone for CPUTensor {}
        #[automatically_derived]
        impl ::core::clone::Clone for CPUTensor {
            #[inline]
            fn clone(&self) -> CPUTensor {
                let _: ::core::clone::AssertParamIsClone<*mut u8>;
                let _: ::core::clone::AssertParamIsClone<*const usize>;
                let _: ::core::clone::AssertParamIsClone<*const usize>;
                let _: ::core::clone::AssertParamIsClone<usize>;
                let _: ::core::clone::AssertParamIsClone<DType>;
                *self
            }
        }
        #[allow(clippy::unnecessary_operation, clippy::identity_op)]
        const _: () = {
            ["Size of CPUTensor"][::std::mem::size_of::<CPUTensor>() - 40usize];
            ["Alignment of CPUTensor"][::std::mem::align_of::<CPUTensor>() - 8usize];
            [
                "Offset of field: CPUTensor::data",
            ][const { builtin # offset_of(CPUTensor, data) } - 0usize];
            [
                "Offset of field: CPUTensor::shape",
            ][const { builtin # offset_of(CPUTensor, shape) } - 8usize];
            [
                "Offset of field: CPUTensor::stride",
            ][const { builtin # offset_of(CPUTensor, stride) } - 16usize];
            [
                "Offset of field: CPUTensor::dims",
            ][const { builtin # offset_of(CPUTensor, dims) } - 24usize];
            [
                "Offset of field: CPUTensor::dtype",
            ][const { builtin # offset_of(CPUTensor, dtype) } - 32usize];
        };
        unsafe extern "C" {
            pub fn cpu_tensor_cast(out: CPUTensor, lhs: CPUTensor);
        }
        unsafe extern "C" {
            pub fn cpu_tensor_copy(dst: CPUTensor, src: CPUTensor);
        }
        unsafe extern "C" {
            pub fn cpu_tensor_add(out: CPUTensor, lhs: CPUTensor, rhs: CPUTensor);
        }
        /// A trait for converting a CPU tensor prototype into the CPUTensor interface used by the CPU operations. This is necessary because the CPU operations are implemented in C and expect a specific interface for the tensor data.
        pub trait IntoInterface {
            unsafe fn into_interface(&self) -> CPUTensor;
        }
    }
    pub mod mem {
        use std::cell::RefCell;
        use std::rc::Rc;
        /// Tensor's representation in CPU memory.
        pub struct CPUMemory<T> {
            ptr: *mut T,
            size: usize,
            layout: std::alloc::Layout,
        }
        #[automatically_derived]
        impl<T: ::core::fmt::Debug> ::core::fmt::Debug for CPUMemory<T> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field3_finish(
                    f,
                    "CPUMemory",
                    "ptr",
                    &self.ptr,
                    "size",
                    &self.size,
                    "layout",
                    &&self.layout,
                )
            }
        }
        impl<T> CPUMemory<T> {
            /// Allocate a CPU memory area
            ///
            /// `size` is the number of elements
            pub fn new(size: usize) -> Self {
                let layout = std::alloc::Layout::array::<T>(size).unwrap();
                let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
                CPUMemory { ptr, size, layout }
            }
            pub fn size(&self) -> usize {
                self.size
            }
            pub fn as_ptr(&self) -> *const T {
                self.ptr
            }
            pub fn as_mut_ptr(&mut self) -> *mut T {
                self.ptr
            }
        }
        impl<T> Drop for CPUMemory<T> {
            fn drop(&mut self) {
                unsafe {
                    std::alloc::dealloc(self.ptr as *mut u8, self.layout);
                };
            }
        }
        impl<T> From<CPUMemory<T>> for SharedCPUMemory<T> {
            fn from(value: CPUMemory<T>) -> Self {
                Rc::new(RefCell::new(value))
            }
        }
        pub type SharedCPUMemory<T> = Rc<RefCell<CPUMemory<T>>>;
    }
    pub mod ops {
        pub mod binary_ops {
            use crate::common::init::TensorAllocInit;
            use crate::common::ops::{
                binary_ops::TensorAdd, cast::TensorCast, promote::Promote,
                view::{TensorBroadcast, broadcast_prot},
            };
            use crate::common::{DTypeImpl, DTypeOf, TensorPrototype};
            use crate::cpu::impls::CPU;
            use crate::cpu::interface;
            use crate::cpu::interface::IntoInterface;
            impl<Lhs, Rhs> TensorAdd<CPU, Rhs> for Lhs
            where
                Lhs: Promote<Rhs>,
                Lhs: DTypeImpl<CPU> + TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
                Rhs: DTypeImpl<CPU> + TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
                <Lhs as Promote<
                    Rhs,
                >>::Output: DTypeImpl<CPU, Prototype: IntoInterface>
                    + TensorBroadcast<CPU> + TensorAllocInit<CPU> + DTypeOf,
            {
                type Output = <Lhs as Promote<Rhs>>::Output;
                fn add(
                    lhs: &Self::Prototype,
                    rhs: &Rhs::Prototype,
                ) -> <Self::Output as DTypeImpl<CPU>>::Prototype {
                    let (lhs, rhs) = broadcast_prot::<
                        CPU,
                        Self::Output,
                        Self::Output,
                    >(
                            &<Lhs as TensorCast<CPU, Self::Output>>::cast(lhs),
                            &<Rhs as TensorCast<CPU, Self::Output>>::cast(rhs),
                        )
                        .expect(
                            &::alloc::__export::must_use({
                                ::alloc::fmt::format(
                                    format_args!(
                                        "Two tensors with shapes ({0:?}, {1:?}) cannot be broadcast.",
                                        lhs.shape(),
                                        rhs.shape(),
                                    ),
                                )
                            }),
                        );
                    let out = <Self::Output as TensorAllocInit<
                        CPU,
                    >>::init(&lhs.shape(), &Default::default());
                    unsafe {
                        interface::cpu_tensor_add(
                            out.into_interface(),
                            lhs.into_interface(),
                            rhs.into_interface(),
                        );
                    };
                    out
                }
            }
        }
        pub mod cast {
            use crate::common::dtype::Any;
            use crate::common::init::TensorAllocInit;
            use crate::common::ops::{
                cast::{TensorCast, TensorCopy, TensorReshape},
                view::{TensorBroadcast, TensorView},
            };
            use crate::common::{DTypeImpl, DTypeOf, Shape, TensorPrototype};
            use crate::cpu::impls::{CPU, CPUTensorAnyPrototype, CPUTensorPrototype};
            use crate::cpu::interface;
            use crate::cpu::interface::IntoInterface;
            impl<
                M,
                N,
                T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<M>> + DTypeOf,
                Dst: DTypeImpl<CPU, Prototype = CPUTensorPrototype<N>>
                    + TensorAllocInit<CPU> + DTypeOf,
            > TensorCast<CPU, Dst> for T
            where
                T::Prototype: TensorPrototype<CPU> + Clone,
                Dst::Prototype: TensorPrototype<CPU>,
            {
                fn cast(src: &Self::Prototype) -> <Dst as DTypeImpl<CPU>>::Prototype {
                    if <T as DTypeOf>::DTYPE == <Dst as DTypeOf>::DTYPE {
                        return unsafe { std::mem::transmute(src.clone()) };
                    }
                    let out = <Dst as TensorAllocInit<
                        CPU,
                    >>::init(&src.shape(), &Default::default());
                    unsafe {
                        interface::cpu_tensor_cast(
                            out.into_interface(),
                            src.into_interface(),
                        );
                    }
                    out
                }
            }
            impl<
                Dst: DTypeImpl<CPU, Prototype: IntoInterface> + TensorBroadcast<CPU>
                    + DTypeOf,
                Src: DTypeImpl<CPU> + TensorCast<CPU, Dst>,
            > TensorCopy<CPU, Src> for Dst {
                fn copy(
                    dst: &mut Self::Prototype,
                    src: &<Src as DTypeImpl<CPU>>::Prototype,
                ) {
                    let src = Dst::broadcast_to(&Src::cast(src), &dst.shape())
                        .expect(
                            &::alloc::__export::must_use({
                                ::alloc::fmt::format(
                                    format_args!(
                                        "Src with shape {0:?} cannot broadcast to shape {1:?} of dst.",
                                        src.shape(),
                                        dst.shape(),
                                    ),
                                )
                            }),
                        );
                    unsafe {
                        interface::cpu_tensor_copy(
                            dst.into_interface(),
                            src.into_interface(),
                        );
                    }
                }
            }
            impl<
                Src: DTypeImpl<CPU> + TensorView<CPU> + TensorAllocInit<CPU>
                    + TensorCopy<CPU, Src>,
            > TensorReshape<CPU> for Src {
                fn reshape(src: &Self::Prototype, new_shape: &Shape) -> Self::Prototype {
                    if let Some(t) = Src::view(src, new_shape) {
                        return t;
                    }
                    let mut new_shape_v: Vec<usize> = new_shape.as_ref().into();
                    let mut inferred_dim = None;
                    let mut known_size = 1;
                    for (i, &dim) in new_shape_v.iter().enumerate() {
                        if dim == 0 {
                            if !inferred_dim.is_none() {
                                {
                                    ::core::panicking::panic_fmt(
                                        format_args!(
                                            "The number of auto-inferred dims is greater than 1.",
                                        ),
                                    );
                                }
                            }
                            inferred_dim = Some(i);
                        } else {
                            known_size *= dim;
                        }
                    }
                    if let Some(i) = inferred_dim {
                        if !(src.shape().numel() % known_size == 0) {
                            {
                                ::core::panicking::panic_fmt(
                                    format_args!("Cannot infer the auto-inferred dim."),
                                );
                            }
                        }
                        new_shape_v[i] = src.shape().numel() / known_size;
                        known_size *= new_shape_v[i];
                    }
                    if !(src.shape().numel() == known_size) {
                        {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "The new number {0} of elements is not equal to the original number {1}.",
                                    known_size,
                                    src.shape().numel(),
                                ),
                            );
                        }
                    }
                    let mut out = <Src as TensorAllocInit<
                        CPU,
                    >>::init(&new_shape_v.into(), &());
                    <Src as TensorCopy<CPU, Src>>::copy(&mut out, src);
                    out
                }
            }
        }
        pub mod view {
            use crate::common::ops::view::{
                TensorBroadcast, TensorIndex, TensorSlice, TensorView,
            };
            use crate::common::{DTypeImpl, Shape, TensorPrototype};
            use crate::cpu::impls::{CPU, CPUTensorPrototype};
            impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U> TensorView<CPU>
            for T
            where
                CPUTensorPrototype<U>: TensorPrototype<CPU>,
            {
                fn view(
                    src: &Self::Prototype,
                    new_shape: &Shape,
                ) -> Option<Self::Prototype> {
                    let mut new_shape_v: Vec<usize> = new_shape.as_ref().into();
                    let mut inferred_dim = None;
                    let mut known_size = 1;
                    for (i, &dim) in new_shape.iter().enumerate() {
                        if dim == 0 {
                            if inferred_dim.is_some() {
                                return None;
                            }
                            inferred_dim = Some(i);
                        } else {
                            known_size *= dim;
                        }
                    }
                    if let Some(i) = inferred_dim {
                        if src.shape().numel() % known_size != 0 {
                            return None;
                        }
                        new_shape_v[i] = src.shape().numel() / known_size;
                        known_size *= new_shape_v[i];
                    }
                    if src.shape().numel() != known_size {
                        return None;
                    }
                    let mut merged_shape_v = Vec::new();
                    let mut merged_stride_v = Vec::new();
                    let mut i = src.shape().len();
                    if i == 0 {
                        merged_shape_v.push(1);
                        merged_stride_v.push(1);
                    }
                    while i > 0 {
                        let mut block_dim = src.shape()[i - 1];
                        let mut block_stride = src.stride()[i - 1];
                        i -= 1;
                        while i > 0
                            && src.stride()[i - 1] == src.stride()[i] * src.shape()[i]
                        {
                            block_dim *= src.shape()[i - 1];
                            block_stride = src.stride()[i];
                            i -= 1;
                        }
                        merged_shape_v.push(block_dim);
                        merged_stride_v.push(block_stride);
                    }
                    merged_shape_v.reverse();
                    merged_stride_v.reverse();
                    let mut new_stride_v = ::alloc::vec::from_elem(0, new_shape_v.len());
                    let mut block_i = merged_shape_v.len();
                    let mut new_i = new_shape_v.len();
                    while block_i > 0 {
                        let block_dim = merged_shape_v[block_i - 1];
                        let block_stride = merged_stride_v[block_i - 1];
                        block_i -= 1;
                        let mut acc = 1;
                        let mut dims = Vec::new();
                        while new_i > 0 && acc <= block_dim {
                            acc *= new_shape_v[new_i - 1];
                            dims.push(new_i - 1);
                            new_i -= 1;
                        }
                        if acc != block_dim {
                            return None;
                        }
                        let mut running_stride = block_stride;
                        for dim in dims {
                            new_stride_v[dim] = running_stride;
                            running_stride *= new_shape_v[dim];
                        }
                    }
                    if new_i != 0 {
                        return None;
                    }
                    Some(
                        CPUTensorPrototype::new(
                            src.data(),
                            &new_shape_v.into(),
                            &new_stride_v.into(),
                            src.offset(),
                        ),
                    )
                }
            }
            impl<
                T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>,
                U,
            > TensorSlice<CPU> for T
            where
                CPUTensorPrototype<U>: TensorPrototype<CPU>,
            {
                fn slice(
                    src: &Self::Prototype,
                    indices: &[crate::common::ops::view::TensorIndex],
                ) -> Self::Prototype {
                    let mut new_shape_v = Vec::new();
                    let mut new_stride_v = Vec::new();
                    let mut new_offset = src.offset();
                    fn _neg_index_to_pos_index(index: isize, dim: usize) -> usize {
                        if index >= 0 {
                            index as usize
                        } else {
                            dim - ((-index) as usize)
                        }
                    }
                    let (indices, mut indices_i, mut dim_i, incr) = if !indices
                        .is_empty()
                    {
                        if indices[0] == TensorIndex::Full {
                            (&indices[1..], indices.len() - 1, src.dims(), false)
                        } else if indices[indices.len() - 1] == TensorIndex::Full {
                            (&indices[..indices.len() - 1], 1, 1, true)
                        } else {
                            (indices, 1, 1, true)
                        }
                    } else {
                        (indices, 1, 1, true)
                    };
                    while indices_i > 0 && indices_i <= indices.len() && dim_i > 0
                        && dim_i <= src.dims()
                    {
                        let dim_size = src.shape()[dim_i - 1];
                        let dim_stride = src.stride()[dim_i - 1];
                        match indices[indices_i - 1] {
                            TensorIndex::Index(_i) => {
                                let i = _neg_index_to_pos_index(_i, dim_size);
                                if !(i < dim_size) {
                                    {
                                        ::core::panicking::panic_fmt(
                                            format_args!(
                                                "Index {0} out of bounds of dimension {1} with size {2}.",
                                                _i,
                                                dim_i - 1,
                                                dim_size,
                                            ),
                                        );
                                    }
                                }
                                new_offset += i * dim_stride;
                                if incr {
                                    dim_i = dim_i + 1;
                                } else {
                                    dim_i = dim_i - 1;
                                };
                            }
                            TensorIndex::Range(_start, _end) => {
                                let (start, end) = (
                                    _neg_index_to_pos_index(_start, dim_size),
                                    _neg_index_to_pos_index(_end, dim_size),
                                );
                                if !(start < dim_size && end <= dim_size && start < end) {
                                    {
                                        ::core::panicking::panic_fmt(
                                            format_args!(
                                                "Range {0:?} out of bounds of dimension {1} with size {2}.",
                                                TensorIndex::Range(_start, _end),
                                                dim_i - 1,
                                                dim_size,
                                            ),
                                        );
                                    }
                                }
                                new_offset += start * dim_stride;
                                new_shape_v.push(end - start);
                                new_stride_v.push(dim_stride);
                                if incr {
                                    dim_i = dim_i + 1;
                                } else {
                                    dim_i = dim_i - 1;
                                };
                            }
                            TensorIndex::RangeFrom(_start) => {
                                let start = _neg_index_to_pos_index(_start, dim_size);
                                if !(start < dim_size) {
                                    {
                                        ::core::panicking::panic_fmt(
                                            format_args!(
                                                "Range {0:?} out of bounds of dimension {1} with size {2}.",
                                                TensorIndex::RangeFrom(_start),
                                                dim_i - 1,
                                                dim_size,
                                            ),
                                        );
                                    }
                                }
                                new_offset += start * dim_stride;
                                new_shape_v.push(dim_size - start);
                                new_stride_v.push(dim_stride);
                                if incr {
                                    dim_i = dim_i + 1;
                                } else {
                                    dim_i = dim_i - 1;
                                };
                            }
                            TensorIndex::RangeTo(_end) => {
                                let end = _neg_index_to_pos_index(_end, dim_size);
                                if !(end > 0 && end <= dim_size) {
                                    {
                                        ::core::panicking::panic_fmt(
                                            format_args!(
                                                "Range {0:?} out of bounds of dimension {1} with size {2}.",
                                                TensorIndex::RangeTo(_end),
                                                dim_i - 1,
                                                dim_size,
                                            ),
                                        );
                                    }
                                }
                                new_shape_v.push(end);
                                new_stride_v.push(dim_stride);
                                if incr {
                                    dim_i = dim_i + 1;
                                } else {
                                    dim_i = dim_i - 1;
                                };
                            }
                            TensorIndex::RangeFull => {
                                new_shape_v.push(dim_size);
                                new_stride_v.push(dim_stride);
                                if incr {
                                    dim_i = dim_i + 1;
                                } else {
                                    dim_i = dim_i - 1;
                                };
                            }
                            TensorIndex::RangeInclusive(_start, _end) => {
                                let (start, end) = (
                                    _neg_index_to_pos_index(_start, dim_size),
                                    _neg_index_to_pos_index(_end, dim_size),
                                );
                                if !(start < dim_size && end < dim_size && start <= end) {
                                    {
                                        ::core::panicking::panic_fmt(
                                            format_args!(
                                                "Range {0:?} out of bounds of dimension {1} with size {2}.",
                                                TensorIndex::RangeInclusive(_start, _end),
                                                dim_i - 1,
                                                dim_size,
                                            ),
                                        );
                                    }
                                }
                                new_offset += start * dim_stride;
                                new_shape_v.push(end - start + 1);
                                new_stride_v.push(dim_stride);
                                if incr {
                                    dim_i = dim_i + 1;
                                } else {
                                    dim_i = dim_i - 1;
                                };
                            }
                            TensorIndex::RangeToInclusive(_end) => {
                                let end = _neg_index_to_pos_index(_end, dim_size);
                                if !(end < dim_size) {
                                    {
                                        ::core::panicking::panic_fmt(
                                            format_args!(
                                                "Range {0:?} out of bounds of dimension {1} with size {2}.",
                                                TensorIndex::RangeToInclusive(_end),
                                                dim_i - 1,
                                                dim_size,
                                            ),
                                        );
                                    }
                                }
                                new_shape_v.push(end + 1);
                                new_stride_v.push(dim_stride);
                                if incr {
                                    dim_i = dim_i + 1;
                                } else {
                                    dim_i = dim_i - 1;
                                };
                            }
                            TensorIndex::Expand => {
                                new_shape_v.push(1);
                                new_stride_v.push(0);
                            }
                            TensorIndex::Full => {
                                {
                                    ::core::panicking::panic_fmt(
                                        format_args!("Inner `Full` is not supported."),
                                    );
                                };
                            }
                        }
                        if incr {
                            indices_i = indices_i + 1;
                        } else {
                            indices_i = indices_i - 1;
                        };
                    }
                    if !((incr && indices_i == indices.len() + 1)
                        || (!incr && indices_i == 0))
                    {
                        {
                            ::core::panicking::panic_fmt(
                                format_args!("Indices aren\'t exhausted."),
                            );
                        }
                    }
                    while dim_i > 0 && dim_i <= src.dims() {
                        new_shape_v.push(src.shape()[dim_i - 1]);
                        new_stride_v.push(src.stride()[dim_i - 1]);
                        if incr {
                            dim_i = dim_i + 1;
                        } else {
                            dim_i = dim_i - 1;
                        };
                    }
                    if !incr {
                        new_shape_v.reverse();
                        new_stride_v.reverse();
                    }
                    Self::Prototype::new(
                        src.data(),
                        &new_shape_v.into(),
                        &new_stride_v.into(),
                        new_offset,
                    )
                }
            }
            impl<
                T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>,
                U,
            > TensorBroadcast<CPU> for T
            where
                CPUTensorPrototype<U>: TensorPrototype<CPU>,
            {
                fn broadcast_to(
                    src: &Self::Prototype,
                    target_shape: &Shape,
                ) -> Option<Self::Prototype> {
                    if target_shape.len() < src.dims() {
                        return None;
                    }
                    let mut new_stride_v = Vec::with_capacity(target_shape.len());
                    let diff = target_shape.len() - src.dims();
                    for _ in 0..diff {
                        new_stride_v.push(0);
                    }
                    for i in 0..src.dims() {
                        if src.shape()[i] != 1
                            && src.shape()[i] != target_shape[i + diff]
                        {
                            return None;
                        }
                        if src.shape()[i] == 1 && target_shape[i + diff] != 1 {
                            new_stride_v.push(0);
                        } else {
                            new_stride_v.push(src.stride()[i]);
                        }
                    }
                    Some(
                        Self::Prototype::new(
                            src.data(),
                            &target_shape.clone(),
                            &new_stride_v.into(),
                            src.offset(),
                        ),
                    )
                }
            }
        }
    }
}
