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
// AURORA PATCH (G5): target metadata for Intel GPUs reached through SPIR-V
// (OpenCL-flavoured `spirv64` kernels, launched via Level Zero).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TARGET_INTELGPU_INTELGPUTRAITS_H
#define KGEN_TARGET_INTELGPU_INTELGPUTRAITS_H

#include "Target/TargetTraits.h"

#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

struct IntelGPUTraits final : TargetTraits {
  llvm::StringRef name() const override { return "intel"; }
  bool matches(const llvm::Triple &triple) const override {
    return triple.getArch() == llvm::Triple::spirv64;
  }
  bool isGPU() const override { return true; }

  llvm::StringRef getAsmExtension() const override { return ".spvasm"; }
  llvm::StringRef getLLVMExtension() const override { return ".spirv.ll"; }
  llvm::StringRef getObjectExtension() const override { return ".spv"; }
  llvm::StringRef getBitcodeExtension() const override { return ".spirv.bc"; }

  llvm::ArrayRef<EmissionKind> supportedEmissionKinds() const override {
    return commonEmissionKinds();
  }

  llvm::StringRef acceleratorSectionTitle() const override {
    return "Intel GPU (SPIR-V / Level Zero)";
  }
  llvm::ArrayRef<AcceleratorArch> supportedAcceleratorArchs() const override;

  /// Shared stateless instance for the lowering and backend `traits()`.
  static const IntelGPUTraits &get();

protected:
  // Open-source target: not gated on MAX being installed.
  bool isBaseTarget() const override { return true; }
};

} // namespace M::KGEN

#endif // KGEN_TARGET_INTELGPU_INTELGPUTRAITS_H
