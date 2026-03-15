// This file is used to include myrustllm-cpu

include!(concat!(env!("OUT_DIR"), "/myrustllm-cpu-bindings.rs"));

pub trait IntoInterface {
    unsafe fn into_interface(&self) -> CPUTensor;
}