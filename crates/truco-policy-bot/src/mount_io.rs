//! Concurrency for policy-artifact reads.
//!
//! Policy profiles are memory-mapped files. On a local disk, reading them is
//! effectively free and everything here is irrelevant. In production they are
//! a network-backed mount (Cloud Run's Cloud Storage FUSE volume), where every
//! page fault and every file open costs a network round trip. Two cold paths
//! are dominated by that latency:
//!
//! - loading the store, which opens one file per solved profile; and
//! - posterior seeding, which binary-searches a profile once per candidate
//!   hand.
//!
//! Both are embarrassingly parallel and neither is CPU-bound, so they share a
//! pool whose width reflects how many reads we want in flight rather than how
//! many cores exist. That distinction matters: the production runtime has a
//! single vCPU, so Rayon's global pool would be one thread and would overlap
//! nothing. A thread blocked on a page fault consumes no CPU.

use std::sync::OnceLock;

/// Reads to keep in flight. Chosen for network round-trip overlap, not cores.
const MOUNT_IO_THREADS: usize = 32;

/// Shared pool for artifact reads, or `None` if one could not be built, in
/// which case callers do the same work serially.
pub(crate) fn mount_io_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(MOUNT_IO_THREADS)
            .thread_name(|index| format!("policy-io-{index}"))
            .build()
        {
            Ok(pool) => Some(pool),
            Err(error) => {
                eprintln!("policy artifacts: no thread pool ({error}); reading serially");
                None
            }
        }
    })
    .as_ref()
}
