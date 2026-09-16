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
// AURORA PATCH (G5): MLIR-lowering policy for Intel GPU (`spirv64`) targets.
// Exported offload functions become OpenCL kernels (`spir_kernel`).
//
// Kernel argument ABI: every argument is passed by pointer, in address space 1
// (OpenCL global / SPIR-V CrossWorkgroup). `DeviceContext` packs each argument's
// device-type bytes into its own slot; the Level Zero runtime copies each slot
// into device-visible memory and binds that pointer. The runtime therefore never
// needs a kernel's argument layout, and SPIR-V never sees an aggregate passed by
// value (illegal when it holds pointers or empty structs, as `TileTensor` does).
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENTypes.h"
#include "Target/IntelGPU/IntelGPUTraits.h"
#include "Target/TargetLowering.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace M::KGEN {
namespace {

/// OpenCL global address space (SPIR-V CrossWorkgroup storage).
constexpr unsigned kKernelArgAddressSpace = 1;

class IntelGPULowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override {
    return &IntelGPUTraits::get();
  }

  void markExportedKernel(mlir::Operation *func) const override {
    if (auto llvmFunc = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(func))
      llvmFunc.setCConv(mlir::LLVM::CConv::SPIR_KERNEL);
  }

  bool isExportedKernel(mlir::Operation *func) const override {
    auto llvmFunc = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(func);
    return llvmFunc && llvmFunc.getCConv() == mlir::LLVM::CConv::SPIR_KERNEL;
  }

  /// Memory-form kernel arguments are pointers into device memory the runtime
  /// fills, not copies: `byref` records the pointee type (which the SPIR-V
  /// backend uses to type opaque pointers) without `byval`'s copy semantics.
  llvm::StringRef getKernelByValArgAttrName() const override {
    return "llvm.byref";
  }

  /// Scalars and raw pointers are passed by pointer too, so every kernel
  /// argument has the same shape for the runtime.
  mlir::Type
  getKernelArgIndirectionType(mlir::Type type,
                              ArgConvention convention) const override {
    if (convention != ArgConvention::ImmReg &&
        convention != ArgConvention::OwnedReg)
      return {};
    if (!llvm::isa<KGEN::SIMDType, KGEN::PointerType>(type))
      return {};
    return KGEN::PointerType::get(type, kKernelArgAddressSpace);
  }

  mlir::Type lowerKernelArgToMemory(mlir::Type type) const override {
    // Aggregates (e.g. `TileTensor`) stay in memory form. Parameter packs are
    // expanded into their members first, so they are left alone here.
    auto structType = llvm::dyn_cast<KGEN::StructType>(type);
    if (!structType || structType.getIsParamPack())
      return {};
    return KGEN::PointerType::get(type, kKernelArgAddressSpace);
  }

protected:
  // Open-source target: not gated on MAX being installed.
  bool isBaseTarget() const override { return true; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<IntelGPULowering> registerIntelGPULowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
