// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace IREE {
namespace VM {

template <typename ConcreteType>
class DebugOnly : public Trait::TraitBase<ConcreteType, DebugOnly> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify debug-only.
    return success();
  }
};

template <typename ConcreteType>
class FullBarrier : public Trait::TraitBase<ConcreteType, FullBarrier> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify full barrier.
    return success();
  }
};

template <typename ConcreteType>
class PseudoOp : public Trait::TraitBase<ConcreteType, PseudoOp> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify pseudo op (not serializable?).
    return success();
  }
};

template <typename ConcreteType>
class ExtI64 : public Trait::TraitBase<ConcreteType, ExtI64> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify i64 ext is supported.
    return success();
  }
};

template <typename ConcreteType>
class ExtF32 : public Trait::TraitBase<ConcreteType, ExtF32> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify f32 ext is supported.
    return success();
  }
};

template <typename ConcreteType>
class ExtF64 : public Trait::TraitBase<ConcreteType, ExtF64> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    // TODO(benvanik): verify f64 ext is supported.
    return success();
  }
};

}  // namespace VM
}  // namespace IREE
}  // namespace OpTrait
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_IR_VMTRAITS_H_
