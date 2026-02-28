/// Device struct.
/// 
/// e.g.: <cpu, 0>, <gpu, 0>
pub struct Device {
    name: String,
    id: usize
}

impl Device {
    pub fn new(name: &str, id: usize) -> Self {
        Device { name: String::from(name), id }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> usize {
        self.id
    }
}