use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// Slice type of tensor
pub enum TensorSlice {
    Index(usize),
    Range(Range<usize>),
    RangeFrom(RangeFrom<usize>),
    RangeTo(RangeTo<usize>),
    RangeFull(RangeFull),
}

impl From<usize> for TensorSlice {
    fn from(value: usize) -> Self {
        TensorSlice::Index(value)
    }
}

impl From<Range<usize>> for TensorSlice {
    fn from(value: Range<usize>) -> Self {
        TensorSlice::Range(value)
    }
}

impl From<RangeFrom<usize>> for TensorSlice {
    fn from(value: RangeFrom<usize>) -> Self {
        TensorSlice::RangeFrom(value)
    }
}

impl From<RangeTo<usize>> for TensorSlice {
    fn from(value: RangeTo<usize>) -> Self {
        TensorSlice::RangeTo(value)
    }
}

impl From<RangeFull> for TensorSlice {
    fn from(value: RangeFull) -> Self {
        TensorSlice::RangeFull(value)
    }
}

#[macro_export]
macro_rules! idx {
    ($($slice:expr),* $(,)?) => {
        [ $( idx!(@parse $slice) ),* ]
    };

    (@parse $slice:expr) => {
        TensorSlice::from($slice);
    }
}
