/* Copyright 2026 The llzk-to-shlo Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_SIMPLIFYSUBCOMPONENTSINTERNAL_H_
#define LLZK_TO_SHLO_CONVERSION_LLZKTOSTABLEHLO_SIMPLIFYSUBCOMPONENTSINTERNAL_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

namespace mlir::llzk_to_shlo {

// Pod-carrier dispatch shapes
// ---------------------------
// The five circom pod-dispatch IR shapes this pass family materializes, each
// paired with its entry point. This is a catalogue, not a C++ enum: nothing
// dispatches on a shape tag, so an enum would be dead infrastructure.
//
// - ArrayOfPods_CrossBlock: an array-of-pods carrier read in one block and
//   written in another (sibling scf.while bodies); bridged via a parallel
//   felt array. → materializePodArrayCompField
// - ArrayOfPods_InputWriteThrough: a dispatch *input* pod whose writer value
//   is forwarded straight through the pod cell to its sibling firing-site
//   read. → materializePodArrayInputPodField
// - ArrayOfPods_ScalarCountCountdown: a scalar pod array dispatched by a
//   decrementing `@count` countdown rather than a constant slot index. →
//   materializeScalarPodCompField
// - StructOfPods_Uniform: a `!pod<[@idx_0..@idx_K-1: T]>` with a single
//   uniform inner type T, rewritten directly to `!array<K x T>`. →
//   convertStructOfPodsToArrayOfPods
// - StructOfPods_NonUniform: a struct-of-pods whose `@idx_*` cells resolve to
//   distinct per-slot classes (e.g. webb Poseidon's Ark cascade); completed
//   by materializing one parallel felt array per referenced `@F` member. →
//   materializeStructOfPodsCompField

/// Walk up from `funcBlock` past any nested `builtin.module` wrappers to the
/// top-level module (LLZK v2's `createEmptyTemplateRemoval` wraps each
/// component in its own `builtin.module`).
ModuleOp getTopLevelModule(Block &funcBlock);

/// Resolve a struct type's defining `struct.def` by exact SymbolRefAttr match.
/// Falls back to the leaf symbol only when the type carries no nested scope.
llzk::component::StructDefOp findStructDefByExactSymbol(
    ModuleOp module, llzk::component::StructType structTy);

/// Build `array<destDims + innerDims x leafFelt>` when `innerFeltTy` is a felt
/// array, or `array<destDims x innerFeltTy>` when it is a scalar `!felt`.
llzk::array::ArrayType
combineDispatchAndInnerFeltDims(Type innerFeltTy, ArrayRef<int64_t> destDims);

/// Check if all results of an operation are unused.
bool isAllResultsUnused(Operation &op);

/// True iff `v` is defined inside one of `op`'s regions.
bool isValueDefinedInside(Value v, Operation &op);

/// Recursively clone the defining-op chain of `v` BEFORE `insertBefore`.
/// Returns cloned value or null Value() on failure.
Value cloneDefiningOpBefore(Value v, Operation *insertBefore,
                            Operation &guardOp,
                            llvm::DenseMap<Value, Value> &cloneCache,
                            unsigned depth = 8);

/// Create an llzk.nondet operation producing an uninitialized value.
Value createNondet(OpBuilder &builder, Location loc, Type type);

/// Emit an `arith.constant <v> : index`.
Value emitConstIndex(OpBuilder &builder, Location loc, int64_t v);

/// Store `value` into `carrier[indices...]`: `array.insert` when `value` is
/// array-typed (a sub-array slice), else `array.write` (a single element).
void emitCarrierWrite(OpBuilder &builder, Location loc, Value carrier,
                      ValueRange indices, Value value);

/// Read `carrier[indices...]` as `resultTy`: `array.extract` when `resultTy`
/// is array-typed (a sub-array slice), else `array.read` (a single element).
Value emitCarrierRead(OpBuilder &builder, Location loc, Value carrier,
                      ValueRange indices, Type resultTy);

/// True for types that participate in pod-array per-field flattening:
/// `!felt.type` or `!array.type<... x !felt.type>`.
bool isFlattenableFelt(Type ty);

/// Index operands of an LLZK `array.read` / `array.write` op. The first
/// operand is the array; everything after is the index list.
SmallVector<Value> arrayAccessIndices(Operation *arrayAccess);

/// Populate the module-scope cache of struct members read outside @constrain.
/// Must be called before any phase erases readm ops.
void populateExternallyLiveMembers(ModuleOp module);

} // namespace mlir::llzk_to_shlo

#endif // NOLINT(build/header_guard): guard exceeds the 80-col line limit
