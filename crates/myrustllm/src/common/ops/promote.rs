//! This mod (promote) declares how two types (lhs, rhs) to promote type.

use crate::common::dtype::{F32, F64, I32, I64};

/// Trait `Promote` declares that `Self` and `Rhs` should promote to `Promote<Rhs>::Output` when they are calculating.
pub trait Promote<Rhs> {
    type Output;
}

macro_rules! promote {
    ($lhs: ty, $rhs: ty, $out: ty) => {
        impl Promote<$rhs> for $lhs {
            type Output = $out;
        }
    };
}

// F32
promote!(F32, F32, F32);
promote!(F32, F64, F64);
promote!(F32, I32, F32);
promote!(F32, I64, F32);

// F64
promote!(F64, F32, F64);
promote!(F64, F64, F64);
promote!(F64, I32, F64);
promote!(F64, I64, F64);

// I32
promote!(I32, F32, F32);
promote!(I32, F64, F64);
promote!(I32, I32, I32);
promote!(I32, I64, I64);

// I64
promote!(I64, F32, F32);
promote!(I64, F64, F64);
promote!(I64, I32, I64);
promote!(I64, I64, I64);
