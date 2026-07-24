# 🔒 A Spinlock Built from `Atomic` in Mojo

## What this is

Mojo's stdlib has **no `Mutex`/`Lock`**. When you genuinely need mutual
exclusion — a critical section only one worker may enter at a time — you build it
from the one primitive that *is* provided: `Atomic` (in `std.atomic`). A
**spinlock** is the simplest such lock: a single atomic flag that threads
repeatedly try to flip from "free" to "held" via compare-and-swap (CAS),
*spinning* (busy-waiting) until they win.

> **Dialect:** all code here is **Mojo 1.0** (`mojo>=1.0.0b2`) and was compiled
> and run against that toolchain, including the concurrent test below.
>
> **Runnable companion:** `spinlock_atomic.mojo` — the `SpinLock`, the
> `LockGuard`, and the concurrent PASS/FAIL demo in one file. Run it with
> `mojo run spinlock_atomic.mojo`.

## When to use it — and when NOT to

A spinlock is the right tool **only** when the critical section is extremely
short and contention is low, because a waiting thread burns a full CPU core doing
nothing. Specifically:

- ✅ Guarding a couple of non-atomic writes that must happen together.
- ✅ Very short sections where the wait is expected to be nanoseconds.
- ❌ **Anything long** (I/O, allocation, a big computation) — you waste cores.
- ❌ **Oversubscription** (more spinning threads than physical cores) — spinners
  never yield, so they can starve the very thread holding the lock. Catastrophic.
- ❌ **A shared counter** — use `flag.fetch_add(1)` directly; it's lock-free and
  strictly faster than locking. (See the note at the bottom.)
- ❌ **A parallel reduction** (like `scalar_sum`) — use private per-worker
  accumulators + a serial combine. A lock there serializes every add and erases
  all parallelism.

In short: reach for a spinlock rarely. Most "I need a lock" moments in Mojo are
better solved by an atomic op or by *not sharing mutable state* in the first place.

---

## The spinlock

```mojo
from std.atomic import Atomic

struct SpinLock:
    """A minimal test-and-CAS spinlock over a single atomic flag."""
    var flag: Atomic[DType.int64]   # 0 = free, 1 = held

    def __init__(out self):
        self.flag = Atomic[DType.int64](0)

    def lock(mut self):
        # Spin until we successfully flip the flag 0 -> 1.
        while True:
            var expected: Int64 = 0                     # must reset every attempt (see below)
            if self.flag.compare_exchange(expected, 1):  # CAS: writes 1 only if flag was 0
                return
            # (optional) CPU-friendly backoff could go here — see caveats.

    def unlock(mut self):
        # Release: publish the free state. A plain atomic store is enough because
        # only the current holder ever calls unlock().
        self.flag.store(0)
```

**Why CAS and not "read, then write"?** The whole point is atomicity: between a
separate read and write, another thread could slip in. `compare_exchange` does
*observe-and-swap as one indivisible step* — it only writes `1` if the flag was
still `0`, and returns whether it won. That's what makes it a lock.

**Why reset `expected` each loop?** `compare_exchange(mut expected, desired)`
takes `expected` by `mut`: on failure it overwrites `expected` with the value it
actually found (here, `1`). If you didn't reset it back to `0`, the next attempt
would try to CAS `1 -> 1` and never acquire.

> Mojo's `Atomic.compare_exchange` is the *weak* form (may fail spuriously) — the
> correct choice inside a spin loop that retries anyway, and cheaper on some
> hardware.

---

## Optional: a scoped guard (RAII / `with`)

So you can't forget to `unlock()`, wrap acquire/release in a context manager. The
guard holds an origin-tracked `Pointer` to the *one* shared lock (never a copy):

```mojo
struct LockGuard[o: MutOrigin]:
    var lk: Pointer[SpinLock, Self.o]

    def __init__(out self, ref [Self.o]lk: SpinLock):
        self.lk = Pointer(to=lk)

    def __enter__(mut self):
        self.lk[].lock()

    def __exit__(mut self):
        self.lk[].unlock()

# usage:
#   with LockGuard(my_lock):
#       # critical section — unlock() runs automatically on block exit
#       shared_thing += 1
```

Note the `Self.o` qualification: inside a struct body, parameters must be
`Self`-qualified. The `ref [Self.o]lk` argument binds the guard's origin to the
lock it refers to, so the atomic op always hits the same memory.

---

## Using it across `parallelize` workers

`parallelize` gives you no synchronization of its own, so this is where a lock
comes in. Every worker increments one shared `Int64` (non-atomic RMW), guarded by
the spinlock. Both the counter and the lock are shared by **stable address** —
the counter via `List` storage, the lock via a pointer — so all workers touch the
same memory. (`parallelize[work]` consumes the worker as a comptime parameter, so
it's `@parameter def` with implicit capture — no capture-list braces.)

```mojo
from std.atomic import Atomic
from std.algorithm import parallelize
from std.sys import num_physical_cores
from std.collections import List

def main() raises:
    var workers = num_physical_cores()
    var iters = 100_000

    var counter = List[Int64](length=1, fill=0)   # shared, stable heap address
    var cptr = counter.unsafe_ptr()
    var lock = SpinLock()
    var lptr = UnsafePointer(to=lock)

    @parameter
    def work(w: Int):
        for _ in range(iters):
            lptr[].lock()
            cptr[0] = cptr[0] + 1        # non-atomic RMW, protected by the lock
            lptr[].unlock()

    parallelize[work](workers, workers)

    var expected = Int64(workers) * Int64(iters)
    print("counter :", cptr[0], " expected :", expected)
    print("RESULT  :", "PASS" if cptr[0] == expected else "FAIL (lost updates)")
```

**Measured (M4 Max, 16 cores):** with the lock, `counter == 1,600,000` every
run (PASS). Comment out the `lock()`/`unlock()` calls and the same program loses
~85% of its updates (e.g. `234,523`) — proof the lock is actually enforcing
mutual exclusion, not passing by luck.

---

## Caveats & sharp edges

- **Busy-waiting burns a core.** A spinning worker does no useful work and never
  yields. Keep the critical section to a handful of instructions.
- **Backoff helps under contention.** Real spinlocks insert a CPU "pause"/relax
  hint (or exponential backoff) in the spin body to cut cache-line ping-pong and
  power draw. The bare loop above is correct but not contention-optimized.
- **Not reentrant.** If the same worker calls `lock()` twice without unlocking,
  it deadlocks against itself. Add an owner-id + recursion count if you need that.
- **No fairness.** There's no queue; a thread can be repeatedly beaten to the CAS
  and starve. Fine for low contention, bad for hot locks (use a ticket lock then).
- **Memory ordering.** Correctness needs *acquire* on `lock()` and *release* on
  `unlock()` so the critical section can't be reordered out of the lock. Mojo's
  `Atomic` defaults to sequentially-consistent ordering (the strongest, safe
  here); only weaken it if you've thought it through.
- **Share the same instance.** Mutual exclusion relies on the atomic op hitting
  the *same* memory — capture the lock by pointer/reference, never a copy.

---

## Prefer atomics or partitioning when you can

For the two most common "I need a lock" cases, skip the lock entirely:

```mojo
# Shared counter — lock-free, faster, simpler:
var counter = Atomic[DType.int64](0)
# inside a worker:
_ = counter.fetch_add(1)          # atomic increment, returns previous value

# Shared accumulation (sums, histograms) — private partials + serial combine,
# exactly as in MOJO_SCALAR_SUM.md §4/§6. No lock, no contention.
```

Build the spinlock only when you must guard *multiple* non-atomic operations as
one indivisible unit and no single atomic op covers it.

---

## ✅ Verification status

Compiled and run on the curriculum toolchain (`mojo 1.0.0b2`, Apple M4 Max,
16 cores): the `SpinLock`, the `LockGuard` context manager, and the concurrent
`parallelize` test all behave as described — 16 workers × 100k increments sum to
exactly 1,600,000 with the lock, and lose updates without it.

**Verified API (Mojo 1.0):** `from std.atomic import Atomic`; methods
`load()`, `store(v)`, `fetch_add(v)` (returns previous), and
`compare_exchange(mut expected, desired) -> Bool` (weak CAS). These names/paths
have drifted across past Mojo releases — the durable part is the *logic* (reset
`expected`, CAS `0 -> 1`, store `0` to release); re-probe the syntax if a future
toolchain rejects it.
