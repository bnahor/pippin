//! Worker-thread setup for Apple Silicon's heterogeneous cores.
//!
//! Physics workers are latency-critical (the async pipeline waits on them), so
//! on macOS they run at `QOS_CLASS_USER_INITIATED`. That keeps the scheduler
//! from parking them on slower cores or demoting them under load. Set
//! `PIPPIN_QOS=default` to opt out.

use std::sync::Once;

static INIT: Once = Once::new();

#[cfg(target_os = "macos")]
fn raise_qos() {
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INITIATED, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn raise_qos() {}

/// Configure the global rayon pool once (no-op if the host already built it).
pub fn init() {
    INIT.call_once(|| {
        if std::env::var("PIPPIN_QOS").as_deref() == Ok("default") {
            return;
        }
        let _ = rayon::ThreadPoolBuilder::new().start_handler(|_| raise_qos()).build_global();
    });
}
