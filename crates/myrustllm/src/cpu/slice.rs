use std::ops::{Range, RangeFrom, RangeFull, RangeTo, RangeInclusive, RangeToInclusive};

/// Slice type of tensor
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorIndex {
    Index(isize),
    Range(isize, isize),
    RangeFrom(isize),
    RangeTo(isize),
    RangeFull,
    RangeInclusive(isize, isize),
    RangeToInclusive(isize),
    Expand,
    Full,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Expand;

impl From<Expand> for TensorIndex {
    fn from(_: Expand) -> Self {
        TensorIndex::Expand
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Full;

impl From<Full> for TensorIndex {
    fn from(_: Full) -> Self {
        TensorIndex::Full
    }
}

#[macro_export]
macro_rules! idx {
    ($($slice:expr),* $(,)?) => {
        [ $( idx!(@parse $slice) ),* ]
    };

    (@parse $slice:expr) => {
        TensorIndex::from($slice)
    };
}
