use crate::common::ops::view::{TensorBroadcast, TensorIndex, TensorSlice, TensorView};
use crate::common::{DTypeImpl, Shape, TensorPrototype};
use crate::cpu::impls::{CPU, CPUTensorPrototype};

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U> TensorView<CPU> for T
where
    CPUTensorPrototype<U>: TensorPrototype<CPU>,
{
    fn view(src: &Self::Prototype, new_shape: &Shape) -> Option<Self::Prototype> {
        let mut new_shape_v: Vec<usize> = new_shape.as_ref().into();

        // Check inferred dim
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

        // Merge
        let mut merged_shape_v = Vec::new();
        let mut merged_stride_v = Vec::new();

        let mut i = src.shape().len();

        if i == 0 {
            // Scalar
            merged_shape_v.push(1);
            merged_stride_v.push(1);
        }

        while i > 0 {
            // Tensor
            let mut block_dim = src.shape()[i - 1];
            let mut block_stride = src.stride()[i - 1];
            i -= 1;

            while i > 0 && src.stride()[i - 1] == src.stride()[i] * src.shape()[i] {
                block_dim *= src.shape()[i - 1];
                block_stride = src.stride()[i];
                i -= 1;
            }
            merged_shape_v.push(block_dim);
            merged_stride_v.push(block_stride);
        }

        merged_shape_v.reverse();
        merged_stride_v.reverse();

        // Split
        let mut new_stride_v = vec![0; new_shape_v.len()];
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

        Some(CPUTensorPrototype::new(
            src.data(),
            &new_shape_v.into(),
            &new_stride_v.into(),
            src.offset(),
        ))
    }
}

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U> TensorSlice<CPU> for T
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

        // Step 1: Check `Full`
        let (indices, mut indices_i, mut dim_i, incr) = if !indices.is_empty() {
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

        // Step 2: Match
        while indices_i > 0 && indices_i <= indices.len() && dim_i > 0 && dim_i <= src.dims() {
            let dim_size = src.shape()[dim_i - 1];
            let dim_stride = src.stride()[dim_i - 1];

            match indices[indices_i - 1] {
                // Index
                TensorIndex::Index(_i) => {
                    let i = _neg_index_to_pos_index(_i, dim_size);
                    assert!(
                        i < dim_size,
                        "Index {} out of bounds of dimension {} with size {}.",
                        _i,
                        dim_i - 1,
                        dim_size
                    );
                    new_offset += i * dim_stride;
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // Range
                TensorIndex::Range(_start, _end) => {
                    let (start, end) = (
                        _neg_index_to_pos_index(_start, dim_size),
                        _neg_index_to_pos_index(_end, dim_size),
                    );
                    assert!(
                        start < dim_size && end <= dim_size && start < end,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::Range(_start, _end),
                        dim_i - 1,
                        dim_size
                    );
                    new_offset += start * dim_stride;
                    new_shape_v.push(end - start);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeFrom
                TensorIndex::RangeFrom(_start) => {
                    let start = _neg_index_to_pos_index(_start, dim_size);
                    assert!(
                        start < dim_size,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::RangeFrom(_start),
                        dim_i - 1,
                        dim_size
                    );
                    new_offset += start * dim_stride;
                    new_shape_v.push(dim_size - start);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeTo
                TensorIndex::RangeTo(_end) => {
                    let end = _neg_index_to_pos_index(_end, dim_size);
                    assert!(
                        end > 0 && end <= dim_size,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::RangeTo(_end),
                        dim_i - 1,
                        dim_size
                    );
                    new_shape_v.push(end);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeFull
                TensorIndex::RangeFull => {
                    new_shape_v.push(dim_size);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeInclusive
                TensorIndex::RangeInclusive(_start, _end) => {
                    let (start, end) = (
                        _neg_index_to_pos_index(_start, dim_size),
                        _neg_index_to_pos_index(_end, dim_size),
                    );
                    assert!(
                        start < dim_size && end < dim_size && start <= end,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::RangeInclusive(_start, _end),
                        dim_i - 1,
                        dim_size
                    );
                    new_offset += start * dim_stride;
                    new_shape_v.push(end - start + 1);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeToInclusive
                TensorIndex::RangeToInclusive(_end) => {
                    let end = _neg_index_to_pos_index(_end, dim_size);
                    assert!(
                        end < dim_size,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::RangeToInclusive(_end),
                        dim_i - 1,
                        dim_size
                    );
                    new_shape_v.push(end + 1);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // Expand
                TensorIndex::Expand => {
                    new_shape_v.push(1);
                    new_stride_v.push(0);
                }

                TensorIndex::Full => {
                    panic!("Inner `Full` is not supported.");
                }
            }
            if incr {
                indices_i = indices_i + 1;
            } else {
                indices_i = indices_i - 1;
            };
        }

        assert!(
            (incr && indices_i == indices.len() + 1) || (!incr && indices_i == 0),
            "Indices aren't exhausted."
        );

        // Step 3: Handle remained dimensions
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

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U> TensorBroadcast<CPU> for T
where
    CPUTensorPrototype<U>: TensorPrototype<CPU>,
{
    fn broadcast_to(src: &Self::Prototype, target_shape: &Shape) -> Option<Self::Prototype> {
        if target_shape.len() < src.dims() {
            return None;
        }

        let mut new_stride_v = Vec::with_capacity(target_shape.len());

        let diff = target_shape.len() - src.dims();
        for _ in 0..diff {
            new_stride_v.push(0);
        }

        for i in 0..src.dims() {
            if src.shape()[i] != 1 && src.shape()[i] != target_shape[i + diff] {
                return None;
            }

            if src.shape()[i] == 1 && target_shape[i + diff] != 1 {
                new_stride_v.push(0);
            } else {
                new_stride_v.push(src.stride()[i]);
            }
        }

        Some(Self::Prototype::new(
            src.data(),
            &target_shape.clone(),
            &new_stride_v.into(),
            src.offset(),
        ))
    }
}
