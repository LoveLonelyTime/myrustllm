use crate::common::{
    GenericTensor, Shape, Tensor,
    autograd::{GraphNode, OpGrad, TensorGrad},
};

pub trait TensorCopy<Rhs = Self> {
    fn copy_from(&mut self, rhs: Rhs);
}

impl TensorCopy for &GenericTensor {
    fn copy_from(&mut self, rhs: Self) {
        match (self, rhs) {
            _ => todo!(),
        }
    }
}

impl std::ops::Add for &GenericTensor {
    type Output = GenericTensor;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (GenericTensor::CPUTensor(lhs), GenericTensor::CPUTensor(rhs)) => {
                GenericTensor::CPUTensor(lhs + rhs)
            }
            _ => todo!(),
        }
    }
}

impl TensorAddReduce for &GenericTensor {
    type Output = GenericTensor;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output {
        match self {
            GenericTensor::CPUTensor(t) => GenericTensor::CPUTensor(t.add_reduce(dims, keep_dim)),
            _ => todo!(),
        }
    }
}

pub struct AddReduceGrad {
    dims: Option<Vec<usize>>,
    keep_dim: bool,
}

impl AddReduceGrad {
    pub fn new(dims: Option<&[usize]>, keep_dim: bool) -> Self {
        AddReduceGrad {
            dims: dims.map(|r| r.iter().map(|v| *v).collect::<Vec<usize>>()),
            keep_dim,
        }
    }
}

impl OpGrad for AddReduceGrad {
    fn forward(&mut self, inputs: &[GenericTensor]) -> Vec<GenericTensor> {
        vec![inputs[0].add_reduce(self.dims.as_ref().map(|v| v.as_slice()), self.keep_dim)]
    }

    fn backward(&self, grad_inputs: &[TensorGrad]) -> Vec<TensorGrad> {
        todo!()
    }
}

impl TensorAddReduce for &TensorGrad {
    type Output = TensorGrad;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output {
        let mut op = AddReduceGrad::new(dims, keep_dim);
        let inputs = vec![self.borrow().clone()];
        let output = op.forward(&inputs)[0].clone();
        let output_shapes = vec![output.shape()];
        TensorGrad::intermediate(
            output,
            GraphNode::new(Box::new(op), &vec![self.clone()], output_shapes, 1),
            0,
        )
    }
}

pub struct AddGrad {
    a_shape: Option<Shape>,
    b_shape: Option<Shape>,
}

impl AddGrad {
    pub fn new() -> Self {
        AddGrad {
            a_shape: None,
            b_shape: None,
        }
    }

    pub fn reduce_grad(grad: &TensorGrad, target_shape: &Shape) -> TensorGrad {
        let mut reduce_dims = Vec::new();
        let diff = grad.dims() - target_shape.len();

        for dim in 0..diff {
            reduce_dims.push(dim);
        }

        let grad = grad.add_reduce(Some(&reduce_dims), false);
        reduce_dims.clear();

        for dim in 0..target_shape.len() {
            if target_shape[dim] == 1 && grad.shape()[dim] != 1 {
                reduce_dims.push(dim);
            }
        }

        grad.add_reduce(Some(&reduce_dims), true)
    }
}

impl OpGrad for AddGrad {
    fn forward(&mut self, inputs: &[GenericTensor]) -> Vec<GenericTensor> {
        assert!(
            inputs.len() == 2,
            "AddGrad requires two operands, but got {}.",
            inputs.len()
        );
        self.a_shape = Some(inputs[0].shape());
        self.b_shape = Some(inputs[1].shape());

        vec![&inputs[0] + &inputs[1]]
    }

    fn backward(&self, grad_inputs: &[TensorGrad]) -> Vec<TensorGrad> {
        let a_shape = self
            .a_shape
            .as_ref()
            .expect("AddGrad forward hasn't been called.");
        let b_shape = self
            .b_shape
            .as_ref()
            .expect("AddGrad forward hasn't been called.");

        return vec![
            AddGrad::reduce_grad(&grad_inputs[0], a_shape),
            AddGrad::reduce_grad(&grad_inputs[0], b_shape),
        ];
    }
}

pub trait TensorAddReduce {
    type Output;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output;
}

impl std::ops::Add for &TensorGrad {
    type Output = TensorGrad;
    fn add(self, rhs: Self) -> Self::Output {
        let mut op = AddGrad::new();
        let inputs = vec![self.borrow().clone(), rhs.borrow().clone()];
        let output = op.forward(&inputs)[0].clone();
        let output_shapes = vec![output.shape()];

        TensorGrad::intermediate(
            output,
            GraphNode::new(
                Box::new(op),
                &vec![self.clone(), rhs.clone()],
                output_shapes,
                1,
            ),
            0,
        )
    }
}
