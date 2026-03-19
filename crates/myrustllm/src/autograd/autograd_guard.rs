use std::cell::Cell;

thread_local! {
    static AUTOGRAD_ENABLED: Cell<bool> = Cell::new(true);
}

/// Return the status of autograd within the current thread.
pub fn is_autograd_enabled() -> bool {
    AUTOGRAD_ENABLED.with(|g| g.get())
}

pub struct AutogradGuard {
    prev: bool,
}

impl AutogradGuard {
    pub fn new(state: bool) -> Self {
        let prev = AUTOGRAD_ENABLED.with(|g| {
            let old = g.get();
            g.set(state);
            old
        });

        Self { prev }
    }
}

impl Drop for AutogradGuard {
    fn drop(&mut self) {
        AUTOGRAD_ENABLED.with(|g| {
            g.set(self.prev);
        });
    }
}

/// Close autograd within its scope.
#[macro_export]
macro_rules! no_grad {
    ($block: expr) => {{
        let _guard = $crate:::autograd::autograd_guard::AutogradGuard::new(false);
        $block
    }};
}

/// Open autograd within its scope.
#[macro_export]
macro_rules! enable_grad {
    ($block: expr) => {{
        let _guard = $crate::autograd::autograd_guard::AutogradGuard::new(true);
        $block
    }};
}
