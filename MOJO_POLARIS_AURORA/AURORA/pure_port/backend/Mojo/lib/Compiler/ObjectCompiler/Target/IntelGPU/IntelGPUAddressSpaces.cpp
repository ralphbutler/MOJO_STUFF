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
// AURORA PATCH (G5): see IntelGPUAddressSpaces.h.
//
// Opaque pointers make a pointer's address space part of its type, so the
// module is rebuilt with LLVM's cloning utilities and a type remapper: every
// global and function whose type mentions an address-space-0 pointer gets a
// copy with remapped types, bodies are cloned through the remapper, then the
// originals are deleted and the copies take their names.
//
//===----------------------------------------------------------------------===//

#include "IntelGPUAddressSpaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <optional>
#include <string>
#include <utility>

namespace M::KGEN {
namespace {

constexpr unsigned kGenericAddressSpace = 0;
constexpr unsigned kGlobalAddressSpace = 1;

/// Maps `ptr` (address space 0) to `ptr addrspace(1)`, recursively through
/// aggregate, vector and function types.
class GenericToGlobalTypeRemapper final : public llvm::ValueMapTypeRemapper {
public:
  llvm::Type *remapType(llvm::Type *type) override {
    auto it = cache.find(type);
    if (it != cache.end())
      return it->second;
    llvm::Type *result = compute(type);
    cache[type] = result;
    return result;
  }

private:
  llvm::Type *compute(llvm::Type *type) {
    llvm::LLVMContext &ctx = type->getContext();
    if (auto *ptr = llvm::dyn_cast<llvm::PointerType>(type))
      return ptr->getAddressSpace() == kGenericAddressSpace
                 ? llvm::PointerType::get(ctx, kGlobalAddressSpace)
                 : type;
    if (auto *array = llvm::dyn_cast<llvm::ArrayType>(type))
      return llvm::ArrayType::get(remapType(array->getElementType()),
                                  array->getNumElements());
    if (auto *vector = llvm::dyn_cast<llvm::VectorType>(type))
      return llvm::VectorType::get(remapType(vector->getElementType()),
                                   vector->getElementCount());
    if (auto *fn = llvm::dyn_cast<llvm::FunctionType>(type)) {
      llvm::SmallVector<llvm::Type *> params;
      for (llvm::Type *param : fn->params())
        params.push_back(remapType(param));
      return llvm::FunctionType::get(remapType(fn->getReturnType()), params,
                                     fn->isVarArg());
    }
    if (auto *st = llvm::dyn_cast<llvm::StructType>(type)) {
      if (st->isOpaque())
        return type;
      llvm::SmallVector<llvm::Type *> elements;
      bool changed = false;
      for (llvm::Type *element : st->elements()) {
        llvm::Type *mapped = remapType(element);
        changed |= mapped != element;
        elements.push_back(mapped);
      }
      if (!changed)
        return type;
      if (st->isLiteral())
        return llvm::StructType::get(ctx, elements, st->isPacked());
      return llvm::StructType::create(ctx, elements,
                                      (st->getName() + ".global").str(),
                                      st->isPacked());
    }
    return type;
  }

  llvm::DenseMap<llvm::Type *, llvm::Type *> cache;
};

/// Remaps the types carried by type attributes (`byref`, `byval`, `sret`, ...)
/// on a function's parameters and return value.
void remapAttributeTypes(llvm::Function &fn,
                         GenericToGlobalTypeRemapper &remapper) {
  llvm::LLVMContext &ctx = fn.getContext();
  llvm::AttributeList attrs = fn.getAttributes();
  for (unsigned i = 0; i < attrs.getNumAttrSets(); ++i) {
    for (int kind = llvm::Attribute::FirstTypeAttr;
         kind <= llvm::Attribute::LastTypeAttr; ++kind) {
      auto typedAttr = static_cast<llvm::Attribute::AttrKind>(kind);
      if (llvm::Type *ty =
              attrs.getAttributeAtIndex(i, typedAttr).getValueAsType())
        attrs = attrs.replaceAttributeTypeAtIndex(ctx, i, typedAttr,
                                                  remapper.remapType(ty));
    }
  }
  fn.setAttributes(attrs);
}

} // namespace

ErrorOrSuccess moveGenericPointersToGlobalAddressSpace(llvm::Module &module) {
  GenericToGlobalTypeRemapper remapper;
  llvm::ValueToValueMapTy vmap;
  llvm::LLVMContext &ctx = module.getContext();

  // 1. New globals for those whose value type or own address space changes.
  llvm::SmallVector<std::pair<llvm::GlobalVariable *, llvm::GlobalVariable *>>
      globals;
  for (llvm::GlobalVariable *gv : llvm::make_pointer_range(module.globals())) {
    llvm::Type *valueType = remapper.remapType(gv->getValueType());
    unsigned addressSpace = gv->getAddressSpace() == kGenericAddressSpace
                                ? kGlobalAddressSpace
                                : gv->getAddressSpace();
    if (valueType == gv->getValueType() &&
        addressSpace == gv->getAddressSpace())
      continue;
    auto *copy = new llvm::GlobalVariable(
        module, valueType, gv->isConstant(), gv->getLinkage(),
        /*Initializer=*/nullptr, gv->getName() + ".global",
        /*InsertBefore=*/nullptr, gv->getThreadLocalMode(), addressSpace,
        gv->isExternallyInitialized());
    copy->copyAttributesFrom(gv);
    vmap[gv] = copy;
    globals.emplace_back(gv, copy);
  }

  // 2. New functions: every definition (bodies may use generic pointers) and
  //    every declaration whose signature changes.
  llvm::SmallVector<llvm::Function *> originals;
  for (llvm::Function &fn : module)
    originals.push_back(&fn);
  llvm::SmallVector<std::pair<llvm::Function *, llvm::Function *>> functions;
  for (llvm::Function *fn : originals) {
    auto *fnType = llvm::cast<llvm::FunctionType>(
        remapper.remapType(fn->getFunctionType()));
    if (fnType == fn->getFunctionType() && fn->isDeclaration())
      continue;
    llvm::Function *copy =
        llvm::Function::Create(fnType, fn->getLinkage(), fn->getAddressSpace(),
                               fn->getName() + ".global", &module);
    vmap[fn] = copy;
    for (auto [oldArg, newArg] : llvm::zip(fn->args(), copy->args())) {
      newArg.setName(oldArg.getName());
      vmap[&oldArg] = &newArg;
    }
    functions.emplace_back(fn, copy);
  }

  // 3. Initializers and bodies, through the remapper.
  for (auto [gv, copy] : globals)
    if (gv->hasInitializer())
      copy->setInitializer(llvm::MapValue(gv->getInitializer(), vmap,
                                          llvm::RF_None, &remapper));
  for (auto [fn, copy] : functions) {
    if (fn->isDeclaration()) {
      copy->copyAttributesFrom(fn);
    } else {
      llvm::SmallVector<llvm::ReturnInst *> returns;
      llvm::CloneFunctionInto(copy, fn, vmap,
                              llvm::CloneFunctionChangeType::GlobalChanges,
                              returns, "", nullptr, &remapper);
    }
    remapAttributeTypes(*copy, remapper);
  }

  // 4. Stack allocations stay in the data layout's alloca address space; a use
  //    that now expects a global pointer goes through an addrspacecast.
  unsigned allocaAddressSpace = module.getDataLayout().getAllocaAddrSpace();
  for (auto [fn, copy] : functions) {
    for (llvm::Instruction &inst :
         llvm::make_early_inc_range(llvm::instructions(*copy))) {
      auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(&inst);
      if (!alloca || alloca->getAddressSpace() == allocaAddressSpace)
        continue;
      llvm::Type *globalPtr = alloca->getType();
      alloca->mutateType(llvm::PointerType::get(ctx, allocaAddressSpace));
      auto *cast = new llvm::AddrSpaceCastInst(
          alloca, globalPtr, alloca->getName() + ".global",
          std::next(alloca->getIterator()));
      alloca->replaceUsesWithIf(
          cast, [cast](llvm::Use &use) { return use.getUser() != cast; });
    }
  }

  // 5. Retire the originals; the copies take their names.
  for (auto [gv, copy] : globals)
    gv->dropAllReferences();
  for (auto [fn, copy] : functions)
    fn->dropAllReferences();
  for (auto [gv, copy] : globals) {
    if (!gv->use_empty())
      return Error("Intel GPU address-space legalization: global '" +
                   gv->getName().str() + "' is still referenced");
    copy->takeName(gv);
    gv->eraseFromParent();
  }
  for (auto [fn, copy] : functions) {
    if (!fn->use_empty())
      return Error("Intel GPU address-space legalization: function '" +
                   fn->getName().str() + "' is still referenced");
    copy->takeName(fn);
    fn->eraseFromParent();
  }

  // 6. Intrinsic declarations overloaded on pointer types (e.g. llvm.memcpy)
  //    need their names re-mangled for the new address space.
  for (llvm::Function &fn : llvm::make_early_inc_range(module)) {
    if (!fn.isIntrinsic())
      continue;
    if (std::optional<llvm::Function *> remangled =
            llvm::Intrinsic::remangleIntrinsicFunction(&fn)) {
      fn.replaceAllUsesWith(*remangled);
      fn.eraseFromParent();
    }
  }

  std::string message;
  llvm::raw_string_ostream os(message);
  if (llvm::verifyModule(module, &os))
    return Error("Intel GPU address-space legalization produced invalid IR: " +
                 message);
  return {};
}

} // namespace M::KGEN
