# spinlock_atomic.mojo — mutual exclusion built from std.atomic.Atomic
#
# Companion to MOJO_SPINLOCK_FROM_ATOMIC.md. Mojo's stdlib has no Mutex; when you
# truly need a critical section, you build one from the primitive that IS
# provided: Atomic. A spinlock is a single atomic flag that threads flip 0->1 via
# compare-and-swap (CAS), busy-waiting until they win.
#
# The demo has each of `workers` threads do `iters` non-atomic increments of one
# shared counter, guarded by the lock. Correct total = workers * iters. Remove the
# lock and the same program loses ~85% of its updates to races — the whole point.
#
# Reach for a spinlock RARELY: only for very short critical sections at low
# contention. For a plain counter use Atomic.fetch_add (lock-free); for a
# reduction use private partials (see scalar_sum.mojo). See the .md for details.

from std.atomic import Atomic
from std.algorithm import parallelize
from std.sys import num_physical_cores
from std.collections import List


struct SpinLock:
    """A minimal test-and-CAS spinlock over a single atomic flag."""
    var flag: Atomic[DType.int64]            # 0 = free, 1 = held

    def __init__(out self):
        self.flag = Atomic[DType.int64](0)

    def lock(mut self):
        while True:
            var expected: Int64 = 0          # reset each attempt: CAS overwrites it on failure
            if self.flag.compare_exchange(expected, 1):   # writes 1 only if flag was still 0
                return

    def unlock(mut self):
        self.flag.store(0)                   # only the holder calls this; a plain store suffices


# RAII guard so you can't forget to unlock. Holds an origin-tracked Pointer to the
# ONE shared lock (never a copy). `with LockGuard(lock): ...` unlocks on block exit.
struct LockGuard[o: MutOrigin]:
    var lk: Pointer[SpinLock, Self.o]

    def __init__(out self, ref [Self.o]lk: SpinLock):
        self.lk = Pointer(to=lk)

    def __enter__(mut self):
        self.lk[].lock()

    def __exit__(mut self):
        self.lk[].unlock()


def main() raises:
    var workers = num_physical_cores()
    var iters = 100_000

    # Shared counter behind a stable heap address (List storage) so every worker
    # mutates the SAME memory; the lock serializes the non-atomic += .
    var counter = List[Int64](length=1, fill=0)
    var cptr = counter.unsafe_ptr()
    var lock = SpinLock()
    var lptr = UnsafePointer(to=lock)

    @parameter
    def work(w: Int):
        for _ in range(iters):
            with LockGuard(lptr[]):           # acquire; auto-release at block end
                cptr[0] = cptr[0] + 1         # non-atomic read-modify-write, protected

    parallelize[work](workers, workers)

    var expected = Int64(workers) * Int64(iters)
    print("spinlock_atomic — mutual exclusion via CAS")
    print("  workers  :", workers)
    print("  counter  :", cptr[0])
    print("  expected :", expected)
    print("  RESULT   :", "PASS" if cptr[0] == expected else "FAIL (lost updates)")
