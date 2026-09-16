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
// AURORA PATCH (G5): TargetBackend for Intel GPUs. Each offload kernel is
// codegen'd by LLVM's in-tree SPIR-V backend: `asm` is SPIR-V text, `object`
// is a SPIR-V binary module that Level Zero loads directly (no link step).
//
//===----------------------------------------------------------------------===//

#include "IntelGPUAddressSpaces.h"
#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Target/IntelGPU/IntelGPUTraits.h"

#include "llvm/IR/Module.h"

namespace M::KGEN {
namespace {

class IntelGPUBackend final : public TargetBackend {
public:
  const TargetTraits *traits() const override {
    return &IntelGPUTraits::get();
  }

  SplitStrategy
  splitStrategy(const CompilationOptions &options) const override {
    return SplitStrategy::PerExported;
  }

  bool isOffload() const override { return true; }

  /// The accelerator arch (e.g. `intel-pvc`) names the device for the stdlib's
  /// GPU tables; LLVM's SPIR-V TargetMachine has no CPU names or features, so
  /// drop them rather than have it warn and ignore them.
  CompilationOptions
  adjustOptionsForTargetMachine(const CompilationOptions &options,
                                llvm::StringRef moduleTriple) const override {
    CompilationOptions adjusted = options;
    adjusted.targetCpu.clear();
    adjusted.targetFeatures.clear();
    return adjusted;
  }

  ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                  EmitContext &ctx) const override {
    if (ErrorOrSuccess error = moveGenericPointersToGlobalAddressSpace(module))
      return Error(error.getError());
    WriteableBufferRef buf = WriteableBuffer::get();
    if (ErrorOrSuccess error =
            ctx.runLlc(module, *buf, /*createObjectFile=*/false))
      return Error(Twine(error.getError()) +
                   ", SPIR-V codegen to assembly failed");
    return buf;
  }

  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override {
    if (ErrorOrSuccess error = moveGenericPointersToGlobalAddressSpace(module))
      return Error(error.getError());
    WriteableBufferRef buf = WriteableBuffer::get();
    if (ErrorOrSuccess error =
            ctx.runLlc(module, *buf, /*createObjectFile=*/true))
      return Error(Twine(error.getError()) +
                   ", SPIR-V codegen to a binary module failed");
    return buf;
  }

  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef moduleName,
                                   EmitContext &ctx) const override {
    return Error("IntelGPUBackend::createArchive is not supported: kernels "
                 "are emitted as one SPIR-V module each");
  }

protected:
  /// OpenCL `__local` / SPIR-V Workgroup storage, where `AddressSpace.SHARED`
  /// (`stack_allocation[SHARED]`) lives.
  std::optional<unsigned> sharedMemoryAddressSpace() const override {
    return 3;
  }

  // Open-source target: not gated on MAX being installed.
  bool isBaseTarget() const override { return true; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<IntelGPUBackend> registerIntelGPUBackend;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
