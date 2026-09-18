//===----------------------------------------------------------------------===//
// mojo_level_zero_rt.cpp — AURORA PATCH (G5 step 4d)
//
// A Level Zero implementation of the `AsyncRT_*` C entry points that Mojo's open
// `max.gpu.host.DeviceContext` calls, so unmodified Mojo GPU programs run on
// Aurora's Intel Max (Ponte Vecchio) GPUs.
//
// Scope: the 18 functions the unchanged MOJO_CURRICULUM programs 02_vecadd_gpu,
// 03c_matmul_coarse and 04b_train_mlp_gpu reference (measured from their host
// assembly). Anything else is deliberately absent: a missing symbol is a link
// error naming exactly what to add next.
//
// Design notes, from the G1-G4 groundwork:
//  * A Level Zero immediate command list is NOT in-order (the G4 lesson), so a
//    barrier is appended after every launch, copy and fill. `DeviceContext`
//    presents an in-order stream, and that ordering is the runtime's job.
//  * Kernel arguments arrive as one host pointer per argument (`args[i]` ->
//    the argument's packed device bytes). Our `spirv64` backend passes every
//    kernel argument by pointer, so each argument is copied into a device-visible
//    cell and the cell's address is bound with `zeKernelSetArgumentValue`. Cells
//    come from a bump arena that is reset on `synchronize`, so they outlive the
//    launches that read them.
//  * `DeviceContext` passes no argument sizes on this path, so the sizes are read
//    from the SPIR-V module itself: each kernel parameter is a pointer, and the
//    size of its pointee is what must be copied (see `spirvKernelArgSizes`).
//  * Buffers reached through a holder are indirect accesses, so every kernel gets
//    `zeKernelSetIndirectAccess(HOST|DEVICE|SHARED)` (the G2 lesson).
//
// Build on Aurora (uan-0007):  bash build_runtime.sh
//===----------------------------------------------------------------------===//

#include <level_zero/ze_api.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

//===----------------------------------------------------------------------===//
// Error reporting: every fallible entry point returns a C string (null = ok)
// that the caller frees with AsyncRT_DeviceContext_strfree.
//===----------------------------------------------------------------------===//

const char *makeError(const std::string &message) {
  char *copy = static_cast<char *>(std::malloc(message.size() + 1));
  if (!copy)
    return "out of memory building an error message";
  std::memcpy(copy, message.c_str(), message.size() + 1);
  return copy;
}

const char *zeError(const char *what, ze_result_t result) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "0x%x", static_cast<unsigned>(result));
  return makeError(std::string(what) + " failed with Level Zero result " +
                   buffer);
}

#define ZE_TRY(expr)                                                           \
  do {                                                                         \
    ze_result_t zeTryResult = (expr);                                          \
    if (zeTryResult != ZE_RESULT_SUCCESS)                                      \
      return zeError(#expr, zeTryResult);                                      \
  } while (0)

//===----------------------------------------------------------------------===//
// IGC build options
//
// zeModuleCreate hands pBuildFlags straight to Intel's GPU compiler. We shipped
// an empty string through G5, which leaves IGC on its defaults: 128 GRF and a
// sub-group width of its own choosing. On PVC the curriculum's coarse matmul
// holds 64 accumulators plus two 8-wide staging vectors per thread, and IGC
// reported "compiled SIMD32 allocated 128 regs and spilled around 247" -- i.e.
// it ran in half the register file the hardware has.
//
// Useful values for MOJO_LZ_BUILD_FLAGS:
//   -ze-opt-large-register-file    256 GRF per thread instead of 128
//   -ze-opt-level=2                IGC optimization level
//   -ze-intel-256-GRF-per-thread   older spelling of the large-GRF switch
// Sub-group width is a kernel property (SPIR-V SubgroupSize execution mode),
// not a build flag, so it is swept from the kernel side instead.
//
// Read once: a process runs one configuration, which is what a sweep wants.
//===----------------------------------------------------------------------===//

const char *igcBuildFlags() {
  static const char *flags = [] {
    const char *env = std::getenv("MOJO_LZ_BUILD_FLAGS");
    return env ? env : "";
  }();
  return flags;
}

//===----------------------------------------------------------------------===//
// Launch ordering (MOJO_LZ_ORDERING)
//
// `DeviceContext` presents an in-order stream. A Level Zero IMMEDIATE command list
// is not in-order by default, which is what silently corrupted G4 (dependent
// kernels overlapped; missing writes, no wrong values). We fixed it the blunt way:
// zeCommandListAppendBarrier after every launch.
//
// G6 job 8836110 measured what that costs. With a trivial kernel, per-launch time
// went 3.9 us at 1 workgroup -> 38.2 us at 256 -> 38.4 us at 4096, while measured
// HOST cost stayed flat at ~3.7 us. The rise is the barrier draining the device
// after every kernel, and it is most of the 40.3-vs-6.3 us/pass vecadd gap against
// our own harness (which launched independent kernels with no barrier at all).
//
// Level Zero can give the same ordering in the driver, without a full drain, via an
// in-order command list. Modes:
//   barrier  (default) - what G5 shipped; correct, conservative, slow
//   inorder            - ZE_COMMAND_QUEUE_FLAG_IN_ORDER, no explicit barrier
//   none               - DIAGNOSTIC ONLY. No ordering. Dependent kernels WILL read
//                        stale data; 04b's loss curve is expected to go wrong. It
//                        exists to prove the barrier is what costs the time.
//===----------------------------------------------------------------------===//

enum class Ordering { Barrier, InOrder, None };

Ordering orderingMode() {
  static const Ordering mode = [] {
    const char *env = std::getenv("MOJO_LZ_ORDERING");
    if (!env || !*env)
      return Ordering::Barrier;
    if (std::strcmp(env, "inorder") == 0)
      return Ordering::InOrder;
    if (std::strcmp(env, "none") == 0)
      return Ordering::None;
    if (std::strcmp(env, "barrier") != 0)
      std::fprintf(stderr,
                   "[mojo-lz] unknown MOJO_LZ_ORDERING='%s', using 'barrier'\n",
                   env);
    return Ordering::Barrier;
  }();
  return mode;
}

const char *orderingName() {
  switch (orderingMode()) {
  case Ordering::InOrder: return "inorder";
  case Ordering::None:    return "none";
  default:                return "barrier";
  }
}

//===----------------------------------------------------------------------===//
// Optional profiling (MOJO_LZ_PROFILE=1)
//
// `DeviceContext.enqueue_function` constructs a fresh DeviceFunction on every
// call, so AsyncRT_DeviceContext_loadFunction runs once per launch, not once per
// kernel. The compiled-kernel cache added in G5 removed the IGC recompile, but
// the cache LOOKUP itself builds a key holding the whole SPIR-V module and hashes
// it -- 86 KB for the coarse matmul, on every launch. This measures that instead
// of assuming it, and splits the per-launch cost into key / lookup / build /
// argument encoding. Off by default and behind one predicted branch when off.
//===----------------------------------------------------------------------===//

struct ProfileStats {
  std::atomic<uint64_t> loadCalls{0}, loadNs{0}, keyNs{0}, lookupNs{0};
  std::atomic<uint64_t> buildCalls{0}, buildNs{0}, keyBytes{0};
  std::atomic<uint64_t> enqCalls{0}, enqNs{0}, argNs{0};
};

ProfileStats &profileStats() {
  static ProfileStats stats;
  return stats;
}

bool profileEnabled() {
  static const bool on = [] {
    const char *env = std::getenv("MOJO_LZ_PROFILE");
    return env && *env && std::strcmp(env, "0") != 0;
  }();
  return on;
}

inline uint64_t nowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void profileReport() {
  ProfileStats &p = profileStats();
  uint64_t loads = p.loadCalls.load();
  uint64_t enqs = p.enqCalls.load();
  if (!loads && !enqs)
    return;
  auto us = [](uint64_t ns, uint64_t n) {
    return n ? (double)ns / (double)n / 1000.0 : 0.0;
  };
  std::fprintf(stderr, "\n[mojo-lz] per-launch host cost (MOJO_LZ_PROFILE)\n");
  std::fprintf(stderr, "  loadFunction calls : %llu\n",
               (unsigned long long)loads);
  std::fprintf(stderr, "    total            : %8.2f us/call\n",
               us(p.loadNs.load(), loads));
  std::fprintf(stderr, "    cache-key build  : %8.2f us/call  (%llu bytes copied per call)\n",
               us(p.keyNs.load(), loads),
               (unsigned long long)(loads ? p.keyBytes.load() / loads : 0));
  std::fprintf(stderr, "    cache lookup     : %8.2f us/call\n",
               us(p.lookupNs.load(), loads));
  std::fprintf(stderr, "    module build     : %8.2f us/call  (%llu misses = IGC compiles)\n",
               us(p.buildNs.load(), loads),
               (unsigned long long)p.buildCalls.load());
  std::fprintf(stderr, "  enqueueFunction calls: %llu\n",
               (unsigned long long)enqs);
  std::fprintf(stderr, "    total            : %8.2f us/call\n",
               us(p.enqNs.load(), enqs));
  std::fprintf(stderr, "    argument encoding: %8.2f us/call\n",
               us(p.argNs.load(), enqs));
  std::fprintf(stderr, "  NOTE: loadFunction runs once per LAUNCH, not once per kernel --\n");
  std::fprintf(stderr, "        DeviceContext builds a fresh DeviceFunction on every call.\n");
}

// Close out a loadFunction that missed the cache and built a module.
void profileRecordLoad(uint64_t loadT0, uint64_t buildT0) {
  ProfileStats &p = profileStats();
  uint64_t end = nowNs();
  p.buildNs.fetch_add(end - buildT0, std::memory_order_relaxed);
  p.buildCalls.fetch_add(1, std::memory_order_relaxed);
  p.loadNs.fetch_add(end - loadT0, std::memory_order_relaxed);
  p.loadCalls.fetch_add(1, std::memory_order_relaxed);
}

struct ProfileReporter {
  ~ProfileReporter() {
    if (profileEnabled())
      profileReport();
  }
};
ProfileReporter profileReporter;

bool igcLogWanted() {
  static const bool wanted = [] {
    const char *env = std::getenv("MOJO_LZ_BUILD_LOG");
    return env && *env && std::strcmp(env, "0") != 0;
  }();
  return wanted;
}

//===----------------------------------------------------------------------===//
// SPIR-V: kernel argument sizes
//
// A minimal type walker over the module's type instructions. For each parameter
// of the entry point (always a pointer under our ABI) it records the size of the
// pointee, which is how many bytes of the caller's argument slot to copy.
//===----------------------------------------------------------------------===//

struct SpirvTypes {
  // Sizes and alignments of every type id defined in the module.
  std::unordered_map<uint32_t, uint64_t> size, align;
  std::unordered_map<uint32_t, uint64_t> constant;   // OpConstant integer values
  std::unordered_map<uint32_t, uint32_t> pointee;    // OpTypePointer -> type id
  std::unordered_map<uint32_t, std::vector<uint32_t>> functionParams;
  std::unordered_map<std::string, uint32_t> entryPoints; // name -> function id
  std::unordered_map<uint32_t, uint32_t> functionType;   // function id -> type id
};

uint64_t roundUp(uint64_t value, uint64_t alignment) {
  return alignment ? (value + alignment - 1) / alignment * alignment : value;
}

/// Parses `words` far enough to answer "how big is each kernel argument".
bool parseSpirv(const uint32_t *words, size_t count, SpirvTypes &types) {
  if (count < 5 || words[0] != 0x07230203u)
    return false;
  size_t i = 5;
  while (i < count) {
    uint32_t opcode = words[i] & 0xFFFFu;
    uint32_t length = words[i] >> 16;
    if (length == 0 || i + length > count)
      return false;
    const uint32_t *op = words + i;
    switch (opcode) {
    case 15: { // OpEntryPoint: model, function, name...
      const char *name = reinterpret_cast<const char *>(op + 3);
      types.entryPoints[std::string(name)] = op[2];
      break;
    }
    case 19: // OpTypeVoid
      types.size[op[1]] = 0, types.align[op[1]] = 1;
      break;
    case 20: // OpTypeBool
      types.size[op[1]] = 1, types.align[op[1]] = 1;
      break;
    case 21: // OpTypeInt: result, width, signedness
    case 22: { // OpTypeFloat: result, width
      uint64_t bytes = op[2] / 8;
      types.size[op[1]] = bytes, types.align[op[1]] = bytes;
      break;
    }
    case 23: { // OpTypeVector: result, component, count
      uint64_t component = types.size.count(op[2]) ? types.size[op[2]] : 0;
      uint64_t bytes = component * op[3];
      types.size[op[1]] = bytes;
      types.align[op[1]] = bytes; // LLVM aligns vectors to their full width
      break;
    }
    case 28: { // OpTypeArray: result, element, length id
      uint64_t element = types.size.count(op[2]) ? types.size[op[2]] : 0;
      uint64_t length = types.constant.count(op[3]) ? types.constant[op[3]] : 0;
      types.size[op[1]] = element * length;
      types.align[op[1]] = types.align.count(op[2]) ? types.align[op[2]] : 1;
      break;
    }
    case 30: { // OpTypeStruct: result, members...
      uint64_t offset = 0, structAlign = 1;
      for (uint32_t m = 2; m < length; ++m) {
        uint32_t member = op[m];
        uint64_t memberSize = types.size.count(member) ? types.size[member] : 0;
        uint64_t memberAlign =
            types.align.count(member) ? types.align[member] : 1;
        offset = roundUp(offset, memberAlign) + memberSize;
        if (memberAlign > structAlign)
          structAlign = memberAlign;
      }
      types.size[op[1]] = roundUp(offset, structAlign);
      types.align[op[1]] = structAlign;
      break;
    }
    case 32: { // OpTypePointer: result, storage class, type
      types.size[op[1]] = 8, types.align[op[1]] = 8;
      types.pointee[op[1]] = op[3];
      break;
    }
    case 33: { // OpTypeFunction: result, return type, parameters...
      std::vector<uint32_t> params;
      for (uint32_t p = 3; p < length; ++p)
        params.push_back(op[p]);
      types.functionParams[op[1]] = std::move(params);
      break;
    }
    case 43: { // OpConstant: type, result, value...
      types.constant[op[2]] = length >= 5 ? (static_cast<uint64_t>(op[4]) << 32 |
                                             op[3])
                                          : op[3];
      break;
    }
    case 54: // OpFunction: result type, result, control, function type
      types.functionType[op[2]] = op[4];
      break;
    default:
      break;
    }
    i += length;
  }
  return true;
}

/// Byte size of each argument of `kernelName`, or an error string.
const char *spirvKernelArgSizes(const void *spirv, size_t bytes,
                                const char *kernelName,
                                std::vector<size_t> &sizes) {
  if (bytes % 4)
    return makeError("SPIR-V module size is not a multiple of 4");
  SpirvTypes types;
  if (!parseSpirv(static_cast<const uint32_t *>(spirv), bytes / 4, types))
    return makeError("not a SPIR-V module (bad magic or truncated)");
  auto entry = types.entryPoints.find(kernelName);
  if (entry == types.entryPoints.end())
    return makeError(std::string("kernel '") + kernelName +
                     "' is not an entry point of the SPIR-V module");
  auto fnType = types.functionType.find(entry->second);
  if (fnType == types.functionType.end())
    return makeError("entry point has no OpFunction");
  auto params = types.functionParams.find(fnType->second);
  if (params == types.functionParams.end())
    return makeError("entry point has no OpTypeFunction");
  for (uint32_t param : params->second) {
    auto pointee = types.pointee.find(param);
    if (pointee == types.pointee.end())
      return makeError("kernel argument is not a pointer: this runtime expects "
                       "the spirv64 backend's by-pointer argument ABI");
    auto size = types.size.find(pointee->second);
    sizes.push_back(size == types.size.end() ? 8 : size->second);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Driver state
//===----------------------------------------------------------------------===//

struct Driver {
  ze_driver_handle_t handle = nullptr;
  std::vector<ze_device_handle_t> devices;
  std::string error;
};

Driver &driver() {
  static Driver state = [] {
    Driver d;
    if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) {
      d.error = "zeInit(GPU_ONLY) failed: no Level Zero GPU driver";
      return d;
    }
    uint32_t driverCount = 0;
    if (zeDriverGet(&driverCount, nullptr) != ZE_RESULT_SUCCESS ||
        driverCount == 0) {
      d.error = "no Level Zero drivers found";
      return d;
    }
    std::vector<ze_driver_handle_t> drivers(driverCount);
    zeDriverGet(&driverCount, drivers.data());
    for (ze_driver_handle_t candidate : drivers) {
      uint32_t deviceCount = 0;
      if (zeDeviceGet(candidate, &deviceCount, nullptr) != ZE_RESULT_SUCCESS ||
          deviceCount == 0)
        continue;
      std::vector<ze_device_handle_t> devices(deviceCount);
      zeDeviceGet(candidate, &deviceCount, devices.data());
      for (ze_device_handle_t device : devices) {
        ze_device_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (zeDeviceGetProperties(device, &props) == ZE_RESULT_SUCCESS &&
            props.type == ZE_DEVICE_TYPE_GPU)
          d.devices.push_back(device);
      }
      if (!d.devices.empty()) {
        d.handle = candidate;
        break;
      }
    }
    if (d.devices.empty())
      d.error = "no Level Zero GPU devices found";
    return d;
  }();
  return state;
}

//===----------------------------------------------------------------------===//
// Handles
//===----------------------------------------------------------------------===//

struct Arena; // below

struct Context {
  std::atomic<int> refs{1};
  int64_t id = 0;
  ze_context_handle_t zeContext = nullptr;
  ze_device_handle_t device = nullptr;
  ze_command_list_handle_t commandList = nullptr; // immediate, async
  std::mutex mutex;
  std::vector<void *> chunks; // argument-cell arena (host USM)
  size_t chunkSize = 0, chunkUsed = 0;
  size_t chunkIndex = 0;
  std::string api = "level_zero";
  std::string name;
  // Compiled kernels, keyed by entry-point name + module bytes. `DeviceContext`
  // builds a fresh `DeviceFunction` on *every* `enqueue_function`, so without
  // this the SPIR-V would be recompiled by IGC per launch (measured: ~15 ms for
  // vecadd, ~285 ms for the coarse matmul). The cache holds one reference to
  // each function; they are destroyed with the context.
  std::unordered_map<std::string, struct Function *> functionCache;
};

struct Buffer {
  std::atomic<int> refs{1};
  Context *context = nullptr;
  void *pointer = nullptr;
  size_t bytes = 0;
  bool host = false;
};

struct Function {
  std::atomic<int> refs{1}; // one reference is held by the context's cache
  Context *context = nullptr;
  ze_module_handle_t module = nullptr;
  ze_kernel_handle_t kernel = nullptr;
  std::vector<size_t> argSizes;
};

constexpr size_t kArenaChunk = 1u << 20; // 1 MiB of argument cells per chunk

/// Bump-allocates `size` bytes of device-visible host memory. Cells stay valid
/// until the next `synchronize`, which is after every launch that reads them.
void *arenaAlloc(Context *context, size_t size) {
  size_t aligned = roundUp(size < 8 ? 8 : size, 8);
  if (context->chunkIndex < context->chunks.size() &&
      context->chunkUsed + aligned <= context->chunkSize) {
    void *cell = static_cast<char *>(context->chunks[context->chunkIndex]) +
                 context->chunkUsed;
    context->chunkUsed += aligned;
    return cell;
  }
  size_t wanted = aligned > kArenaChunk ? aligned : kArenaChunk;
  if (context->chunkIndex + 1 < context->chunks.size() &&
      wanted <= context->chunkSize) {
    // Reuse an already-allocated chunk after a reset.
    ++context->chunkIndex;
    context->chunkUsed = aligned;
    return context->chunks[context->chunkIndex];
  }
  ze_host_mem_alloc_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  void *chunk = nullptr;
  if (zeMemAllocHost(context->zeContext, &desc, wanted, 64, &chunk) !=
      ZE_RESULT_SUCCESS)
    return nullptr;
  context->chunks.push_back(chunk);
  context->chunkIndex = context->chunks.size() - 1;
  context->chunkSize = wanted;
  context->chunkUsed = aligned;
  return chunk;
}

void arenaReset(Context *context) {
  context->chunkIndex = 0;
  context->chunkUsed = 0;
}

/// Waits for every enqueued command, then frees the argument cells.
const char *synchronizeLocked(Context *context) {
  ZE_TRY(zeCommandListHostSynchronize(context->commandList, UINT64_MAX));
  arenaReset(context);
  return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// AsyncRT entry points
//===----------------------------------------------------------------------===//

extern "C" {

void AsyncRT_DeviceContext_strfree(const char *pointer) {
  std::free(const_cast<char *>(pointer));
}

const char *AsyncRT_DeviceContext_create(Context **result, const char *api,
                                         int32_t id) {
  Driver &d = driver();
  if (!d.error.empty())
    return makeError(d.error);
  if (api && *api && std::string(api) != "level_zero")
    return makeError(std::string("this runtime provides the 'level_zero' "
                                 "device API, not '") +
                     api + "'");
  if (id < 0 || static_cast<size_t>(id) >= d.devices.size())
    return makeError("device id " + std::to_string(id) + " out of range (" +
                     std::to_string(d.devices.size()) + " Level Zero GPUs; "
                     "set ZE_AFFINITY_MASK to pick a tile)");

  auto *context = new Context();
  context->id = id;
  context->device = d.devices[id];

  ze_context_desc_t contextDesc{};
  contextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  if (ze_result_t r =
          zeContextCreate(d.handle, &contextDesc, &context->zeContext);
      r != ZE_RESULT_SUCCESS) {
    delete context;
    return zeError("zeContextCreate", r);
  }

  // Pick a compute queue group for the immediate command list.
  uint32_t groupCount = 0;
  zeDeviceGetCommandQueueGroupProperties(context->device, &groupCount, nullptr);
  std::vector<ze_command_queue_group_properties_t> groups(groupCount);
  for (auto &group : groups)
    group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  zeDeviceGetCommandQueueGroupProperties(context->device, &groupCount,
                                         groups.data());
  uint32_t ordinal = 0;
  for (uint32_t g = 0; g < groupCount; ++g)
    if (groups[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ordinal = g;
      break;
    }

  ze_command_queue_desc_t queueDesc{};
  queueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queueDesc.ordinal = ordinal;
  queueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  queueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  // ZE_COMMAND_QUEUE_FLAG_IN_ORDER is an ENUMERATOR, not a macro, so it cannot be
  // probed with #ifdef -- that test is always false and would silently disable this
  // path. Used unconditionally: a header too old to have it fails the build loudly,
  // which is what we want. Level Zero 1.9+ (oneAPI 2024+); Aurora is on 2026.1.
  if (orderingMode() == Ordering::InOrder)
    queueDesc.flags |= ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
  if (std::getenv("MOJO_LZ_PROFILE") || std::getenv("MOJO_LZ_BUILD_LOG"))
    std::fprintf(stderr, "[mojo-lz] launch ordering: %s\n", orderingName());
  if (ze_result_t r = zeCommandListCreateImmediate(
          context->zeContext, context->device, &queueDesc,
          &context->commandList);
      r != ZE_RESULT_SUCCESS) {
    zeContextDestroy(context->zeContext);
    delete context;
    return zeError("zeCommandListCreateImmediate", r);
  }

  ze_device_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  if (zeDeviceGetProperties(context->device, &props) == ZE_RESULT_SUCCESS)
    context->name = props.name;

  *result = context;
  return nullptr;
}

void AsyncRT_DeviceContext_retain(Context *context) {
  context->refs.fetch_add(1, std::memory_order_relaxed);
}

void AsyncRT_DeviceContext_release(Context *context) {
  if (context->refs.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  zeCommandListHostSynchronize(context->commandList, UINT64_MAX);
  for (auto &entry : context->functionCache) {
    zeKernelDestroy(entry.second->kernel);
    zeModuleDestroy(entry.second->module);
    delete entry.second;
  }
  for (void *chunk : context->chunks)
    zeMemFree(context->zeContext, chunk);
  zeCommandListDestroy(context->commandList);
  zeContextDestroy(context->zeContext);
  delete context;
}

int64_t AsyncRT_DeviceContext_id(Context *context) { return context->id; }

/// Writes a `StringRef`-shaped {pointer, length} pair (Mojo's `StaticString`).
void AsyncRT_DeviceContext_deviceApi(void **result, Context *context) {
  result[0] = const_cast<char *>(context->api.c_str());
  result[1] = reinterpret_cast<void *>(context->api.size());
}

const char *AsyncRT_DeviceContext_synchronize(Context *context) {
  std::lock_guard<std::mutex> lock(context->mutex);
  return synchronizeLocked(context);
}

const char *AsyncRT_DeviceContext_createBuffer_async(Buffer **result,
                                                     void **devicePointer,
                                                     Context *context,
                                                     size_t length,
                                                     size_t elementSize) {
  size_t bytes = length * elementSize;
  ze_device_mem_alloc_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void *pointer = nullptr;
  ZE_TRY(zeMemAllocDevice(context->zeContext, &desc, bytes ? bytes : 1, 64,
                          context->device, &pointer));
  auto *buffer = new Buffer{{1}, context, pointer, bytes, /*host=*/false};
  AsyncRT_DeviceContext_retain(context);
  *devicePointer = pointer;
  *result = buffer;
  return nullptr;
}

const char *AsyncRT_DeviceContext_createHostBuffer(Buffer **result,
                                                   void **hostPointer,
                                                   Context *context,
                                                   size_t length,
                                                   size_t elementSize) {
  size_t bytes = length * elementSize;
  ze_host_mem_alloc_desc_t desc{};
  desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  void *pointer = nullptr;
  ZE_TRY(zeMemAllocHost(context->zeContext, &desc, bytes ? bytes : 1, 64,
                        &pointer));
  auto *buffer = new Buffer{{1}, context, pointer, bytes, /*host=*/true};
  AsyncRT_DeviceContext_retain(context);
  *hostPointer = pointer;
  *result = buffer;
  return nullptr;
}

int64_t AsyncRT_DeviceBuffer_bytesize(Buffer *buffer) {
  return static_cast<int64_t>(buffer->bytes);
}

Context *AsyncRT_DeviceBuffer_context(Buffer *buffer) {
  AsyncRT_DeviceContext_retain(buffer->context);
  return buffer->context;
}

void AsyncRT_DeviceBuffer_retain(Buffer *buffer) {
  buffer->refs.fetch_add(1, std::memory_order_relaxed);
}

void AsyncRT_DeviceBuffer_release(Buffer *buffer) {
  if (buffer->refs.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  Context *context = buffer->context;
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    // Commands in flight may still read this allocation.
    zeCommandListHostSynchronize(context->commandList, UINT64_MAX);
    zeMemFree(context->zeContext, buffer->pointer);
  }
  delete buffer;
  AsyncRT_DeviceContext_release(context);
}

/// Copies between any two buffers of this runtime (device or host memory);
/// `DeviceContext` routes `enqueue_copy_to` and `map_to_host` through it.
const char *AsyncRT_DeviceContext_DtoD_async(Context *context, Buffer *dst,
                                             Buffer *src) {
  size_t bytes = dst->bytes < src->bytes ? dst->bytes : src->bytes;
  std::lock_guard<std::mutex> lock(context->mutex);
  ZE_TRY(zeCommandListAppendMemoryCopy(context->commandList, dst->pointer,
                                       src->pointer, bytes, nullptr, 0,
                                       nullptr));
  ZE_TRY(zeCommandListAppendBarrier(context->commandList, nullptr, 0, nullptr));
  return nullptr;
}

const char *AsyncRT_DeviceContext_setMemory_async(Context *context, Buffer *dst,
                                                  uint64_t value,
                                                  size_t valueSize) {
  std::lock_guard<std::mutex> lock(context->mutex);
  // The fill pattern must outlive the append, so it lives in the arena.
  void *pattern = arenaAlloc(context, valueSize ? valueSize : 1);
  if (!pattern)
    return makeError("out of device-visible memory for a fill pattern");
  std::memcpy(pattern, &value, valueSize);
  ZE_TRY(zeCommandListAppendMemoryFill(context->commandList, dst->pointer,
                                       pattern, valueSize, dst->bytes, nullptr,
                                       0, nullptr));
  ZE_TRY(zeCommandListAppendBarrier(context->commandList, nullptr, 0, nullptr));
  return nullptr;
}

const char *AsyncRT_DeviceContext_loadFunction(
    Function **result, Context *context, const char *moduleName,
    const char *functionName, const char *data, size_t dataLength,
    int32_t maxDynamicSharedBytes, const char *debugLevel,
    int32_t optimizationLevel) {
  (void)moduleName;
  (void)maxDynamicSharedBytes;
  (void)debugLevel;
  (void)optimizationLevel;

  const bool prof = profileEnabled();
  const uint64_t loadT0 = prof ? nowNs() : 0;

  // NOTE: this key holds the ENTIRE SPIR-V module, and it is rebuilt and rehashed
  // on every launch because DeviceContext calls loadFunction per launch.
  std::string cacheKey(functionName);
  cacheKey.push_back('\0');
  cacheKey.append(data, dataLength);
  const uint64_t keyT1 = prof ? nowNs() : 0;
  if (prof) {
    ProfileStats &p = profileStats();
    p.keyNs.fetch_add(keyT1 - loadT0, std::memory_order_relaxed);
    p.keyBytes.fetch_add(cacheKey.size(), std::memory_order_relaxed);
  }
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    auto cached = context->functionCache.find(cacheKey);
    if (cached != context->functionCache.end()) {
      cached->second->refs.fetch_add(1, std::memory_order_relaxed);
      AsyncRT_DeviceContext_retain(context);
      *result = cached->second;
      if (prof) {
        ProfileStats &p = profileStats();
        uint64_t end = nowNs();
        p.lookupNs.fetch_add(end - keyT1, std::memory_order_relaxed);
        p.loadNs.fetch_add(end - loadT0, std::memory_order_relaxed);
        p.loadCalls.fetch_add(1, std::memory_order_relaxed);
      }
      return nullptr;
    }
  }
  const uint64_t buildT0 = prof ? nowNs() : 0;
  if (prof)
    profileStats().lookupNs.fetch_add(buildT0 - keyT1,
                                      std::memory_order_relaxed);

  ze_module_desc_t moduleDesc{};
  moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  moduleDesc.inputSize = dataLength;
  moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(data);
  moduleDesc.pBuildFlags = igcBuildFlags();

  ze_module_handle_t module = nullptr;
  ze_module_build_log_handle_t buildLog = nullptr;
  if (ze_result_t r = zeModuleCreate(context->zeContext, context->device,
                                     &moduleDesc, &module, &buildLog);
      r != ZE_RESULT_SUCCESS) {
    std::string message = "zeModuleCreate failed";
    if (buildLog) {
      size_t logSize = 0;
      zeModuleBuildLogGetString(buildLog, &logSize, nullptr);
      std::string log(logSize, '\0');
      zeModuleBuildLogGetString(buildLog, &logSize, log.data());
      zeModuleBuildLogDestroy(buildLog);
      message += ": " + log;
    }
    return makeError(message);
  }
  // IGC reports register allocation and spills here ("compiled SIMD32 allocated
  // 128 regs and spilled around 247"). It is the only place those numbers are
  // visible at run time, so surface them on request instead of discarding them.
  if (buildLog) {
    if (igcLogWanted()) {
      size_t logSize = 0;
      zeModuleBuildLogGetString(buildLog, &logSize, nullptr);
      if (logSize > 1) {
        std::string log(logSize, '\0');
        zeModuleBuildLogGetString(buildLog, &logSize, log.data());
        std::fprintf(stderr, "[mojo-lz] IGC build log for '%s':\n%s\n",
                     functionName, log.c_str());
      }
    }
    zeModuleBuildLogDestroy(buildLog);
  }

  ze_kernel_desc_t kernelDesc{};
  kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDesc.pKernelName = functionName;
  ze_kernel_handle_t kernel = nullptr;
  if (ze_result_t r = zeKernelCreate(module, &kernelDesc, &kernel);
      r != ZE_RESULT_SUCCESS) {
    zeModuleDestroy(module);
    return makeError(std::string("zeKernelCreate('") + functionName +
                     "') failed");
  }

  auto *function = new Function();
  function->context = context;
  function->module = module;
  function->kernel = kernel;
  if (const char *error = spirvKernelArgSizes(data, dataLength, functionName,
                                              function->argSizes)) {
    zeKernelDestroy(kernel);
    zeModuleDestroy(module);
    delete function;
    return error;
  }
  // Buffers are reached through the argument holders, so the kernel needs
  // indirect access to every kind of allocation (the G2 lesson).
  zeKernelSetIndirectAccess(kernel,
                            ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
                                ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
                                ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED);
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    auto cached = context->functionCache.find(cacheKey);
    if (cached != context->functionCache.end()) {
      // Another thread compiled the same kernel first; keep that one.
      zeKernelDestroy(function->kernel);
      zeModuleDestroy(function->module);
      delete function;
      cached->second->refs.fetch_add(1, std::memory_order_relaxed);
      AsyncRT_DeviceContext_retain(context);
      *result = cached->second;
      if (prof)
        profileRecordLoad(loadT0, buildT0);
      return nullptr;
    }
    function->refs.store(2, std::memory_order_relaxed); // cache + caller
    context->functionCache.emplace(cacheKey, function);
  }
  AsyncRT_DeviceContext_retain(context);
  *result = function;
  if (prof)
    profileRecordLoad(loadT0, buildT0);
  return nullptr;
}

void AsyncRT_DeviceFunction_release(Function *function) {
  // The context's cache holds the last reference, so the module stays compiled
  // for the next launch; everything is destroyed with the context.
  function->refs.fetch_sub(1, std::memory_order_acq_rel);
  AsyncRT_DeviceContext_release(function->context);
}

const char *AsyncRT_DeviceContext_enqueueFunctionDirect(
    Context *context, Function *function, uint32_t gridX, uint32_t gridY,
    uint32_t gridZ, uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t sharedMemoryBytes, void *attributes, uint32_t attributeCount,
    void **args, uint32_t argCount, uint64_t *argSizes) {
  (void)sharedMemoryBytes; // shared memory is a module-scope Workgroup variable
  (void)attributes;
  (void)attributeCount;

  if (argCount != function->argSizes.size())
    return makeError("kernel expects " +
                     std::to_string(function->argSizes.size()) +
                     " arguments but the launch supplied " +
                     std::to_string(argCount));

  const bool prof = profileEnabled();
  const uint64_t enqT0 = prof ? nowNs() : 0;

  std::lock_guard<std::mutex> lock(context->mutex);
  for (uint32_t i = 0; i < argCount; ++i) {
    size_t size = argSizes ? static_cast<size_t>(argSizes[i])
                           : function->argSizes[i];
    void *cell = arenaAlloc(context, size);
    if (!cell)
      return makeError("out of device-visible memory for kernel arguments");
    std::memcpy(cell, args[i], size);
    ZE_TRY(zeKernelSetArgumentValue(function->kernel, i, sizeof(void *),
                                    &cell));
  }
  if (prof)
    profileStats().argNs.fetch_add(nowNs() - enqT0, std::memory_order_relaxed);
  ZE_TRY(zeKernelSetGroupSize(function->kernel, blockX, blockY, blockZ));
  ze_group_count_t groups{gridX, gridY, gridZ};
  ZE_TRY(zeCommandListAppendLaunchKernel(context->commandList, function->kernel,
                                         &groups, nullptr, 0, nullptr));
  // `DeviceContext` promises an in-order stream. Under Ordering::Barrier we supply
  // that with a full drain after every launch (correct, and measurably expensive --
  // see the MOJO_LZ_ORDERING note). Under Ordering::InOrder the driver supplies it.
  if (orderingMode() == Ordering::Barrier)
    ZE_TRY(zeCommandListAppendBarrier(context->commandList, nullptr, 0, nullptr));
  if (prof) {
    ProfileStats &p = profileStats();
    p.enqNs.fetch_add(nowNs() - enqT0, std::memory_order_relaxed);
    p.enqCalls.fetch_add(1, std::memory_order_relaxed);
  }
  return nullptr;
}

} // extern "C"
