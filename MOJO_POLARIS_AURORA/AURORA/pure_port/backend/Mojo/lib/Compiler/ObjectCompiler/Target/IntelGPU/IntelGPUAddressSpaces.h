//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// AURORA PATCH (G5): address-space legalization for Intel GPU kernels.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_TARGET_INTELGPU_INTELGPUADDRESSSPACES_H
#define KGEN_COMPILER_TARGET_INTELGPU_INTELGPUADDRESSSPACES_H

#include "Support/ErrorOr.h"

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN {

/// Moves every pointer in Mojo's generic address space (0) to the OpenCL global
/// address space (1) throughout `module`: function signatures, instruction
/// results, aggregate fields, globals and type attributes.
///
/// Mojo lowers `AddressSpace.GENERIC` pointers (e.g. a `TileTensor`'s data
/// pointer) to LLVM address space 0, which the SPIR-V backend emits as
/// `Function` storage: thread-private memory, through which IGC is free to drop
/// stores. Device buffers must be `CrossWorkgroup` (address space 1).
///
/// Stack allocations must stay in address space 0; any `alloca` that survives
/// optimization is kept there and bridged to its uses with an `addrspacecast`
/// (which SPIR-V consumers may reject: kernels normally optimize allocas away).
/// Returns an error if the rewritten module fails LLVM verification.
ErrorOrSuccess moveGenericPointersToGlobalAddressSpace(llvm::Module &module);

} // namespace M::KGEN

#endif // KGEN_COMPILER_TARGET_INTELGPU_INTELGPUADDRESSSPACES_H
