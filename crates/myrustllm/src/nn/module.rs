use std::marker::PhantomData;

use myrustllm_derive::Module;

use crate::nn::parameter::Parameter;

pub trait ModuleVisitor<'a> {
    fn visit_parameter(&mut self, p: &'a mut Parameter);
}

pub trait Forward<Input> {
    type Output;
    fn forward(&self, input: Input) -> Self::Output;
}

pub trait Module {
    fn visit<'a>(&'a mut self, visitor: &mut dyn ModuleVisitor<'a>);
}

impl<T: Module> Module for Option<T> {
    fn visit<'a>(&'a mut self, visitor: &mut dyn ModuleVisitor<'a>) {
        if let Some(m) = self {
            m.visit(visitor);
        }
    }
}

#[derive(Module)]
pub struct CascadeModule2<M1: Module + Forward<I1>, M2: Module + Forward<M1::Output>, I1> {
    #[module]
    m1: M1,
    #[module]
    m2: M2,
    _marker: PhantomData<I1>,
}

impl<M1: Module + Forward<I1>, M2: Module + Forward<M1::Output>, I1> CascadeModule2<M1, M2, I1> {
    pub fn cascade(m1: M1, m2: M2) -> Self {
        CascadeModule2 {
            m1,
            m2,
            _marker: PhantomData,
        }
    }
}

impl<M1: Module + Forward<I1>, M2: Module + Forward<M1::Output>, I1> Forward<I1>
    for CascadeModule2<M1, M2, I1>
{
    type Output = M2::Output;
    fn forward(&self, input: I1) -> Self::Output {
        self.m2.forward(self.m1.forward(input))
    }
}

#[macro_export]
macro_rules! cascade {
    ($m:expr) => {
        $m
    };

    ($m1:expr, $($rest:expr),+) => {
        $crate::nn::module::CascadeModule2::cascade($m1, cascade!($($rest),+))
    };
}

pub struct ParamCollector<'a> {
    pub params: Vec<&'a mut Parameter>,
}

impl<'a> ParamCollector<'a> {
    pub fn new() -> Self {
        ParamCollector { params: Vec::new() }
    }
}

impl<'a> ModuleVisitor<'a> for ParamCollector<'a> {
    fn visit_parameter(&mut self, p: &'a mut Parameter) {
        self.params.push(p);
    }
}

impl<'a> Into<Vec<&'a mut Parameter>> for ParamCollector<'a> {
    fn into(self) -> Vec<&'a mut Parameter> {
        self.params
    }
}
