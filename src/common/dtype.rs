/// DType struct.
/// 
/// e.g.: f32, i32, ...
pub struct DType {
    name: String
}

impl DType {
    pub fn new(name: &str) -> Self{
        DType {
            name: String::from(name)
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}