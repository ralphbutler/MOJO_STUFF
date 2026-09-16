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

#include "Target/IntelGPU/IntelGPUTraits.h"

namespace M::KGEN {

const IntelGPUTraits &IntelGPUTraits::get() {
  static const IntelGPUTraits instance;
  return instance;
}

llvm::ArrayRef<TargetTraits::AcceleratorArch>
IntelGPUTraits::supportedAcceleratorArchs() const {
  static const AcceleratorArch archs[] = {
      {"intel-pvc", "Intel Data Center GPU Max (Ponte Vecchio)"},
  };
  return archs;
}

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetTraits<IntelGPUTraits> registerIntelGPUTraits;
#pragma GCC diagnostic pop
} // namespace

} // namespace M::KGEN
