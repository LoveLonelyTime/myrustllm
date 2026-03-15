use crate::common::{DType, Shape, dtype::StdDType};

pub trait ByteSource {
    fn len(&self) -> usize;
    fn read(&self, offset: usize, buf: &mut [u8]);
}

pub struct MemorySource {
    data: Vec<u8>,
}

impl MemorySource {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }
}

impl ByteSource for MemorySource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn read(&self, offset: usize, buf: &mut [u8]) {
        buf.copy_from_slice(&self.data[offset..offset + buf.len()]);
    }
}

pub struct ConcatSource {
    sources: Vec<Box<dyn ByteSource>>,
    offsets: Vec<usize>,
    len: usize,
}

impl ConcatSource {
    pub fn new(sources: Vec<Box<dyn ByteSource>>) -> Self {
        let mut offsets = Vec::new();
        let mut cur = 0;

        for s in &sources {
            offsets.push(cur);
            cur += s.len();
        }

        Self {
            sources,
            offsets,
            len: cur,
        }
    }
}

impl ByteSource for ConcatSource {
    fn len(&self) -> usize {
        self.len
    }

    fn read(&self, offset: usize, buf: &mut [u8]) {
        let mut remain = buf.len();
        let mut dst = 0;
        let mut off = offset;

        for (i, src) in self.sources.iter().enumerate().rev() {
            if off >= self.offsets[i] {
                let local = off - self.offsets[i];
                let max = src.len() - local;
                let n = remain.min(max);

                src.read(local, &mut buf[dst..dst + n]);

                remain -= n;
                dst += n;
                off += n;

                if remain == 0 {
                    return;
                }
            }
        }
    }
}

pub struct TensorRawData {
    pub source: Box<dyn ByteSource>,
    pub shape: Shape,
    pub dtype: DType,
}

pub trait Literal {
    type Type;
    fn shape(&self) -> Shape;
    fn data(&self) -> Vec<Self::Type>;
}

// impl<T: StdDType> Literal for T::RType {
//     fn shape(&self) -> Shape {
//         todo!()
//     }

//     fn data(&self) -> Vec<T::RType> {
//         todo!()
//     }
// }

// impl<T: StdDType<RType: bytemuck::Pod>> From<T::RType> for TensorRawData {
//     fn from(value: T::RType) -> Self {
//         let bytes = bytemuck::bytes_of(&value).to_vec();

//         TensorRawData {
//             source: Box::new(MemorySource::new(bytes)),
//             shape: Shape::scalar(),
//             dtype: T::DTYPE,
//         }
//     }
// }

// impl<T: Into<TensorRawData>, const N: usize> From<[T; N]> for TensorRawData {
//     fn from(arr: [T; N]) -> Self {
//         assert!(N > 0);

//         let mut sources = Vec::with_capacity(N);
//         let mut shape = None;
//         let mut dtype = None;

//         for item in arr {
//             let raw_data = item.into();
//             sources.push(raw_data.source);
//             assert!(*shape.get_or_insert(raw_data.shape.clone()) == raw_data.shape);
//             assert!(*dtype.get_or_insert(raw_data.dtype) == raw_data.dtype);
//         }

//         let mut shape_v = vec![N];
//         shape_v.extend(shape.unwrap().iter());

//         TensorRawData {
//             source: Box::new(ConcatSource::new(sources)),
//             shape: Shape::from(shape_v),
//             dtype: dtype.unwrap(),
//         }
//     }
// }
