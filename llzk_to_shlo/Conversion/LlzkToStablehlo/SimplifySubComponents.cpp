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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/InputPodElimination.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/LlzkUpstreamArtifacts.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodArrayMaterialize.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodArrayReads.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodArrayWhileCarry.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodDispatchPhases.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodFinalization.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodInvariants.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponentsInternal.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/StructOfPodsConversion.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::llzk_to_shlo {

#define GEN_PASS_DEF_SIMPLIFYSUBCOMPONENTS
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponents.h.inc"

// ===----------------------------------------------------------------------===
// Shared helper functions (external linkage — declared in
// SimplifySubComponentsInternal.h, used by both this file and
// PodDispatchPhases.cpp).
// ===----------------------------------------------------------------------===

/// Check if all results of an operation are unused.
bool isAllResultsUnused(Operation &op) {
  for (auto result : op.getResults())
    if (!result.use_empty())
      return false;
  return true;
}

/// True iff `v` is defined inside one of `op`'s regions (either as an
/// OpResult of a nested op, or as a BlockArgument of a nested block).
/// `Value::getParentRegion()` returns the enclosing region for both
/// flavors; `Region::isAncestor` handles the rest.
bool isValueDefinedInside(Value v, Operation &op) {
  Region *def = v.getParentRegion();
  if (!def)
    return false;
  for (Region &r : op.getRegions())
    if (r.isAncestor(def))
      return true;
  return false;
}

/// Create an llzk.nondet operation producing an uninitialized value.
Value createNondet(OpBuilder &builder, Location loc, Type type) {
  OperationState state(loc, "llzk.nondet");
  state.addTypes({type});
  return builder.create(state)->getResult(0);
}

Value emitConstIndex(OpBuilder &builder, Location loc, int64_t v) {
  OperationState state(loc, "arith.constant");
  state.addAttribute("value", builder.getIndexAttr(v));
  state.addTypes({builder.getIndexType()});
  return builder.create(state)->getResult(0);
}

void emitCarrierWrite(OpBuilder &builder, Location loc, Value carrier,
                      ValueRange indices, Value value) {
  StringRef opName = isa<llzk::array::ArrayType>(value.getType())
                         ? "array.insert"
                         : "array.write";
  OperationState state(loc, opName);
  SmallVector<Value> operands{carrier};
  llvm::append_range(operands, indices);
  operands.push_back(value);
  state.addOperands(operands);
  builder.create(state);
}

Value emitCarrierRead(OpBuilder &builder, Location loc, Value carrier,
                      ValueRange indices, Type resultTy) {
  StringRef opName =
      isa<llzk::array::ArrayType>(resultTy) ? "array.extract" : "array.read";
  OperationState state(loc, opName);
  SmallVector<Value> operands{carrier};
  llvm::append_range(operands, indices);
  state.addOperands(operands);
  state.addTypes({resultTy});
  return builder.create(state)->getResult(0);
}

// `cloneDefiningOpBefore` has external linkage — it is declared in
// SimplifySubComponentsInternal.h and called by PodDispatchPhases.cpp.
// `isSafeToCloneBefore` is its file-private helper (`static`): no other
// translation unit needs it, so it keeps internal linkage.

/// True iff cloning `def` before `guardOp` is safe — i.e. the clone reads
/// the same value its original location would read. Two categories qualify:
/// (1) ops without memory effects (`Pure` trait or empty
/// `MemoryEffectOpInterface`), and (2) LLZK's read-only array/pod access ops
/// (`array.read` / `array.extract` / `array.len` / `pod.read`). These read
/// mutable LLZK records so they are NOT `Pure`, but hoisting them out of
/// `guardOp` is nevertheless safe because the clone reads the record operand
/// BEFORE `guardOp` executes — any writes that `guardOp`'s body would perform
/// haven't happened yet at the clone's new position. `pod.read` matters for
/// the webb Poseidon shape: writer-side inputs come from a struct-of-pods
/// dispatch chain (`pod.read %cell[@in]` ← `array.read %carrier[%cK]`) that
/// `materializeStructOfPodsCompField` must clone past the count-guard scf.if
/// to materialize 68-arm Ark cascade writers — without the pod.read entry
/// every writer's hoist clone fails and the carrier stays unwritten.
static bool isSafeToCloneBefore(Operation *def) {
  if (mlir::isMemoryEffectFree(def))
    return true;
  return isa<llzk::array::ReadArrayOp, llzk::array::ExtractArrayOp,
             llzk::array::ArrayLengthOp, llzk::pod::ReadPodOp>(def);
}

static bool canCloneDefiningOpBeforeImpl(Value v, Operation &guardOp,
                                         llvm::DenseSet<Value> &visiting,
                                         unsigned depth) {
  if (!isValueDefinedInside(v, guardOp))
    return true;
  if (depth == 0)
    return false;
  if (!visiting.insert(v).second)
    return true;

  auto eraseOnExit = llvm::make_scope_exit([&] { visiting.erase(v); });

  Operation *def = v.getDefiningOp();
  if (!def)
    return false; // block-argument inside guardOp — not clonable
  if (!isSafeToCloneBefore(def))
    return false;
  for (Value operand : def->getOperands()) {
    if (!canCloneDefiningOpBeforeImpl(operand, guardOp, visiting, depth - 1))
      return false;
  }
  return true;
}

/// Recursively clone the defining-op chain of `v` BEFORE `insertBefore` so
/// the clone's result dominates `insertBefore`. Returns the cloned value,
/// or a null `Value()` on failure (chain reaches a block argument inside
/// `guardOp`, an unsafe op, or `depth` exhausted).
///
/// `guardOp` is the scope we are hoisting OUT of (typically the enclosing
/// `scf.if`). Values defined outside `guardOp` are returned as-is and
/// terminate the recursion.
///
/// `cloneCache` dedupes when multiple args share a sub-chain. The depth
/// cap is a convergence guard against adversarial chains.
Value cloneDefiningOpBefore(Value v, Operation *insertBefore,
                            Operation &guardOp,
                            llvm::DenseMap<Value, Value> &cloneCache,
                            unsigned depth) {
  if (!isValueDefinedInside(v, guardOp))
    return v;
  if (depth == 0)
    return Value();
  if (auto it = cloneCache.find(v); it != cloneCache.end())
    return it->second;
  Operation *def = v.getDefiningOp();
  if (!def)
    return Value(); // block-argument inside guardOp — not clonable
  if (!isSafeToCloneBefore(def))
    return Value();

  SmallVector<Value> newOperands;
  newOperands.reserve(def->getNumOperands());
  for (Value o : def->getOperands()) {
    Value cloned =
        cloneDefiningOpBefore(o, insertBefore, guardOp, cloneCache, depth - 1);
    if (!cloned)
      return Value();
    newOperands.push_back(cloned);
  }

  OpBuilder builder(insertBefore);
  Operation *cloned = def->clone();
  for (auto [i, o] : llvm::enumerate(newOperands))
    cloned->setOperand(i, o);
  builder.insert(cloned);

  unsigned resultIdx = 0;
  for (unsigned i = 0, ni = def->getNumResults(); i < ni; ++i) {
    if (def->getResult(i) == v) {
      resultIdx = i;
      break;
    }
  }
  Value result = cloned->getResult(resultIdx);
  cloneCache[v] = result;
  return result;
}

bool canCloneDefiningOpBefore(Value v, Operation &guardOp, unsigned depth) {
  llvm::DenseSet<Value> visiting;
  return canCloneDefiningOpBeforeImpl(v, guardOp, visiting, depth);
}

/// Walk up from `funcBlock` past any nested `builtin.module` wrappers to
/// the top-level module. LLZK v2's `createEmptyTemplateRemoval` wraps each
/// component in its own `builtin.module`, so a SymbolTable lookup that
/// must reach a sibling component must start at the outermost module.
ModuleOp getTopLevelModule(Block &funcBlock) {
  ModuleOp moduleOp = funcBlock.getParentOp()->getParentOfType<ModuleOp>();
  while (moduleOp) {
    ModuleOp outer = moduleOp->getParentOfType<ModuleOp>();
    if (!outer)
      break;
    moduleOp = outer;
  }
  return moduleOp;
}

/// Build `array<destDims + innerDims x leafFelt>` when `innerFeltTy` is a
/// felt array (`!array<K x !felt>`), or `array<destDims x innerFeltTy>` when
/// it is a scalar `!felt`. Used by writers + readers materializing a
/// dispatch-sized parallel destination for a sub-component's `@out` member.
llzk::array::ArrayType
combineDispatchAndInnerFeltDims(Type innerFeltTy, ArrayRef<int64_t> destDims) {
  if (auto innerArr = dyn_cast<llzk::array::ArrayType>(innerFeltTy)) {
    auto innerDims = getArrayDimensions(innerArr);
    SmallVector<int64_t> combined(destDims.begin(), destDims.end());
    combined.append(innerDims.begin(), innerDims.end());
    return llzk::array::ArrayType::get(innerArr.getElementType(), combined);
  }
  return llzk::array::ArrayType::get(innerFeltTy, destDims);
}

/// True for types that participate in pod-array per-field flattening:
/// `!felt.type` or `!array.type<... x !felt.type>`.
bool isFlattenableFelt(Type ty) {
  if (isa<llzk::felt::FeltType>(ty))
    return true;
  if (auto at = dyn_cast<llzk::array::ArrayType>(ty))
    return isa<llzk::felt::FeltType>(at.getElementType());
  return false;
}

/// Index operands of an LLZK `array.read` / `array.write` op. The first
/// operand is the array; everything after is the index list.
SmallVector<Value> arrayAccessIndices(Operation *arrayAccess) {
  return llvm::to_vector(llvm::drop_begin(arrayAccess->getOperands()));
}

namespace {

using BlockPhase = bool (*)(Block &);
using ModulePhase = bool (*)(ModuleOp);

// Single source of truth for SSC's per-block phase entry points. Consumed by
// `runSingleTestPhase` (test-phase=<name> lookup) AND by `runOnOperation` via
// `runBlockPhase` so a phase invocation site without a registration entry
// cannot drift past the assertion at first run. Leaky-pointer pattern (no
// exit-time destructor registered via `atexit`) per LLVM's coding standards
// "Do not use Static Constructors" / non-trivial-destructor guidance.
const llvm::StringMap<BlockPhase> &getBlockPhases() {
  static const auto *m = new llvm::StringMap<BlockPhase>{
      {"flattenPodArrayWhileCarry", flattenPodArrayWhileCarry},
      {"flattenPodArrayScfIfResults", flattenPodArrayScfIfResults},
      {"unpackPodWhileCarry", unpackPodWhileCarry},
      {"convertStructOfPodsToArrayOfPods", convertStructOfPodsToArrayOfPods},
      {"materializeStructOfPodsCompField", materializeStructOfPodsCompField},
      {"materializePodArrayCompField", materializePodArrayCompField},
      {"materializePodArrayInputPodField", materializePodArrayInputPodField},
      {"materializeScalarPodCompField", materializeScalarPodCompField},
      {"eraseDeadPodAndCountOps", eraseDeadPodAndCountOps},
      {"replaceRemainingPodOps", replaceRemainingPodOps},
      {"eliminatePodDispatch", eliminatePodDispatch},
      {"resolveArrayPodCompReads", resolveArrayPodCompReads},
      {"rewriteArrayPodCountCompInReads", rewriteArrayPodCountCompInReads},
  };
  return *m;
}

// Module-scope phase registry. Void-returning phases are wrapped in a `+[]`
// trampoline so the map has one uniform `bool(ModuleOp)` signature; the
// trampoline returns `true` (changed-flag value is meaningless for those
// phases). `erasePodTypedCarrierSlots` already returns `bool` natively — its
// real return is used by `runOnOperation`'s cleanup fixed point.
const llvm::StringMap<ModulePhase> &getModulePhases() {
  static const auto *m = new llvm::StringMap<ModulePhase>{
      {"flattenSingleEntityWrapperModules",
       +[](ModuleOp m) -> bool {
         flattenSingleEntityWrapperModules(m);
         return true;
       }},
      {"stripEmptyStructParams",
       +[](ModuleOp m) -> bool {
         stripEmptyStructParams(m);
         return true;
       }},
      {"eliminateInputPods",
       +[](ModuleOp m) -> bool {
         eliminateInputPods(m);
         return true;
       }},
      {"inlineInputPodCarries",
       +[](ModuleOp m) -> bool {
         inlineInputPodCarries(m);
         return true;
       }},
      {"erasePodTypedCarrierSlots", erasePodTypedCarrierSlots},
  };
  return *m;
}

bool runBlockPhase(StringRef name, Block &block) {
  auto fn = getBlockPhases().lookup(name);
  assert(fn && "phase missing from getBlockPhases()");
  return fn(block);
}

bool runModulePhase(StringRef name, ModuleOp module) {
  auto fn = getModulePhases().lookup(name);
  assert(fn && "phase missing from getModulePhases()");
  return fn(module);
}

/// Test-only driver: run exactly ONE named phase entry point on `module` and
/// return, instead of the full fixed-point pipeline. Lets lit exercise an
/// individual phase against its documented pre/postcondition.
///
/// For block-scope phases, the named phase is invoked on every
/// `function.def` body block in the module — mirroring the
/// granularity at which the real driver invokes these phases (it walks
/// `struct.def → function.def @compute → region → block`). The test driver is
/// deliberately broader than the real driver's `@compute`-only gating so a
/// fixture need not be named `@compute`; this is acceptable because the only
/// goal is to drive ONE phase against a hand-written precondition.
///
/// For module-scope phases, the named phase is invoked once on `module`.
///
/// `extractCallsFromScfIf` / `replacePodReads` are intentionally NOT supported:
/// they take an extra `trackedPodValues` map and cannot be invoked standalone.
///
/// Returns false (and the caller signals pass failure) on an unknown or
/// unsupported phase name.
bool runSingleTestPhase(StringRef name, ModuleOp module) {
  if (auto fn = getBlockPhases().lookup(name)) {
    // Collect every function-body block once, so phases that mutate the IR
    // don't perturb a live walk.
    SmallVector<Block *> blocks;
    module.walk([&](Operation *op) {
      if (!isa<llzk::function::FuncDefOp>(op))
        return;
      for (Region &region : op->getRegions())
        for (Block &block : region)
          blocks.push_back(&block);
    });
    for (Block *block : blocks)
      fn(*block);
    return true;
  }
  if (auto fn = getModulePhases().lookup(name)) {
    fn(module);
    return true;
  }
  return false;
}

struct SimplifySubComponents
    : impl::SimplifySubComponentsBase<SimplifySubComponents> {
  using SimplifySubComponentsBase::SimplifySubComponentsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Test-only: run a single named phase in isolation and return. The empty
    // default leaves the full fixed-point pipeline below unchanged.
    if (!testPhase.empty()) {
      if (!runSingleTestPhase(testPhase, module)) {
        module.emitError("simplify-sub-components: unknown or unsupported "
                         "test-phase '")
            << testPhase << "'";
        signalPassFailure();
      }
      return;
    }

    // Snapshot the set of struct members read outside @constrain — Phase 4
    // (`eraseDeadPodAndCountOps`) consults this to decide whether an
    // scf.if-wrapped `struct.writem %self[@F]` is preserving a live wire or
    // dead bookkeeping. Must be computed BEFORE any phase erases readm ops.
    populateExternallyLiveMembers(module);

    // Idempotence: SSC runs from both the CLI and from inside
    // `--llzk-to-stablehlo`. The second invocation finds a pod-free,
    // template-free module; probe once and skip the v2 prereqs if no
    // pod or `$inputs` member remains.
    bool needsV2Prereqs = false;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "pod" ||
          (isa<llzk::component::MemberDefOp>(op) &&
           op->getAttrOfType<StringAttr>("sym_name") &&
           op->getAttrOfType<StringAttr>("sym_name")
               .getValue()
               .ends_with("$inputs"))) {
        needsV2Prereqs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // Pre-step: strip `@X$inputs` pod traffic from `@compute`/`@constrain`
    // so the later template-removal `applyFullConversion` does not trip on
    // residual `pod.read` ops from the LLZK v2 constrain-side channel.
    if (needsV2Prereqs)
      runModulePhase("eliminateInputPods", module);

    // Run to fixed point for multi-level sub-component chains.
    bool changed = true;
    while (changed) {
      changed = false;
      module.walk([&](Operation *structDef) {
        if (!isa<llzk::component::StructDefOp>(structDef))
          return;

        structDef->walk([&](Operation *funcDef) {
          if (!isa<llzk::function::FuncDefOp>(funcDef))
            return;
          auto symName = funcDef->getAttrOfType<StringAttr>("sym_name");
          if (!symName || symName.getValue() != "compute")
            return;

          for (Region &region : funcDef->getRegions()) {
            for (Block &block : region) {
              bool hasPod = false;
              block.walk([&](Operation *op) {
                if (op->getName().getDialectNamespace() == "pod")
                  hasPod = true;
              });
              if (hasPod) {
                changed |= runBlockPhase("materializePodArrayCompField", block);
                // Run BEFORE `materializePodArrayInputPodField` so any
                // struct-of-pods carrier (`!pod<[@idx_N..]>` with uniform
                // inner type) is rewritten to array-of-pods first. The
                // `@in` read-modify-write pattern downstream then sees
                // `pod.read` on an `array.read`-produced cell — the exact
                // shape `materializePodArrayInputPodField` and
                // `flattenPodArrayWhileCarry` already handle. One call per
                // outer iter is sufficient because the rewrite is
                // idempotent — after success the carrier's type is array,
                // no longer matching the seed predicate.
                changed |=
                    runBlockPhase("convertStructOfPodsToArrayOfPods", block);
                // Non-uniform-inner struct-of-pods carriers
                // (`convertStructOfPodsToArrayOfPods` no-op) leave the
                // dispatched calls hoisted by `extractCallsFromScfIf` with
                // no consumer for their `!struct<@Sub_K>` results — the
                // reader cascade in a sibling scf.while body reads
                // `struct.readm [@F]` from a pre-existing
                // `llzk.nondet : !struct<@Sub_K>` instead. Bridge the
                // writer↔reader link by materializing a parallel felt
                // carrier per `@F`. Idempotent.
                changed |=
                    runBlockPhase("materializeStructOfPodsCompField", block);
                // Run BEFORE `flattenPodArrayWhileCarry` so the
                // writer-side `pod.write %cell[@in] = %src` and the
                // firing-site `pod.read %cell[@in]` are still SSA-paired
                // through `%cell`. After flatten, `%cell` is severed from
                // the per-field carry across nested scf.whiles and the
                // pairing is unrecoverable.
                changed |=
                    runBlockPhase("materializePodArrayInputPodField", block);
                changed |= runBlockPhase("flattenPodArrayWhileCarry", block);
                // Drive `unpackPodWhileCarry` to its own fixed point before
                // materializing tail calls. The unpacker processes one
                // while per call (it erases chained `scf.while` users
                // inline, invalidating SmallVector pointers, so it returns
                // early); without this loop the outer fixed point would
                // run `eliminatePodDispatch` after only one writer-while's
                // pod-carry was unpacked, and Phase 5
                // (`replaceRemainingPodOps`) would `llzk.nondet` the
                // sibling pods' cross-block readers before
                // `materializeScalarPodCompField` could see them.
                while (runBlockPhase("unpackPodWhileCarry", block))
                  changed = true;
                changed |=
                    runBlockPhase("materializeScalarPodCompField", block);
                changed |= runBlockPhase("eliminatePodDispatch", block);
                // Recursively process nested while body blocks.
                std::function<void(Block &)> processNested;
                processNested = [&](Block &parent) {
                  for (Operation &op : parent) {
                    if (!isa<scf::WhileOp>(op))
                      continue;
                    for (Region &r : op.getRegions()) {
                      for (Block &b : r) {
                        changed |=
                            runBlockPhase("flattenPodArrayWhileCarry", b);
                        changed |= runBlockPhase("unpackPodWhileCarry", b);
                        bool hasArrayOfPods = false;
                        for (Operation &bop : b)
                          if (isa<llzk::array::ReadArrayOp>(bop) &&
                              bop.getNumResults() > 0 &&
                              isa<llzk::pod::PodType>(
                                  bop.getResult(0).getType()))
                            hasArrayOfPods = true;
                        // Skip eliminatePodDispatch when this scf.while
                        // body still has pod-typed block args. Phase 5
                        // (`replaceRemainingPodOps`) would nondet every
                        // `pod.read` in the block — including
                        // `pod.read %arg[@field]` reads that
                        // `unpackPodWhileCarry` needs to discover field
                        // types on the next outer fixed-point iteration.
                        // Once the carry is unpacked, the block args
                        // become non-pod and dispatch elimination can
                        // proceed normally.
                        bool hasPodBlockArg = false;
                        for (BlockArgument arg : b.getArguments())
                          if (isa<llzk::pod::PodType>(arg.getType()))
                            hasPodBlockArg = true;
                        if (!hasArrayOfPods && !hasPodBlockArg) {
                          changed |= runBlockPhase("eliminatePodDispatch", b);
                        } else if (hasArrayOfPods && !hasPodBlockArg) {
                          // Post-Option-B carrier: block has
                          // `array.read %carrier[%i] : !pod` at top
                          // level but no pod-typed block args. Phase 5's
                          // nondet would still clobber the `array.read`
                          // chain's pod field discovery, so skip the
                          // full eliminatePodDispatch. But Phase 1
                          // (`extractCallsFromScfIf`) is benign — it
                          // hoists buried `function.call`s out of
                          // dispatch-firing scf.if's via clone-hoist.
                          // Without this targeted call, deep dispatch
                          // chains (canonical case: webb Poseidon's
                          // 68-round Ark→Mix→Sigma) sit in
                          // statically-false scf.if's that
                          // `--llzk-to-stablehlo` later DCEs.
                          //
                          // Phase 2 (`replacePodReads`) is also safe
                          // here: the carrier is now array-of-pods, so
                          // pod-typed block args have already been
                          // unpacked by an earlier outer iter; no live
                          // pod-read-back patterns through block args
                          // remain. Phase 2 only RAUWs pod.reads whose
                          // source is tracked via the local Phase 1
                          // scan — narrow enough not to clobber
                          // unrelated pod traffic.
                          //
                          // Phase 2 is intentionally NOT run when
                          // `hasPodBlockArg` is true: pod block args
                          // host `pod.read %arg[@field]` read-back
                          // patterns that the broader RAUW would tear
                          // (canonical regression:
                          // `unpack_pod_while_carry_block_arg_*.mlir`).
                          llvm::DenseMap<Value, llvm::StringMap<Value>>
                              localTrackedPodValues;
                          changed |=
                              extractCallsFromScfIf(b, localTrackedPodValues);
                          changed |= replacePodReads(b, localTrackedPodValues);
                        } else if (hasPodBlockArg) {
                          // Surviving struct-of-pods carrier on a pod
                          // block-arg with NON-uniform inner types (each
                          // `@idx_K` resolves to a different `@comp:
                          // !struct<@Sub_K>`).
                          // `convertStructOfPodsToArrayOfPods` cannot rewrite
                          // the carrier — there's no single array element type
                          // that admits the K distinct struct classes. But the
                          // buried `function.call @Sub_K::@compute(%input)` is
                          // already concrete in the IR (circom emits
                          // the literal `@Sub_K` symbol per cascade arm).
                          // Phase 1 (`extractCallsFromScfIf`) can still
                          // clone-hoist these calls when their operand
                          // chain is pure / non-pod (e.g.
                          // `array.extract %outer_iter_arg[%c_K]` on
                          // a felt-array sibling carrier). Phase 2's
                          // broad RAUW is intentionally skipped: it
                          // would tear `pod.read %arg[@field]` read-back
                          // patterns that `unpackPodWhileCarry` later
                          // depends on (canonical regression:
                          // `unpack_pod_while_carry_block_arg_*.mlir`).
                          // Canonical case: webb Poseidon's 68-round
                          // Ark cascade where each `@idx_K` resolves to
                          // a distinct `!struct<@Ark_K>` round constant
                          // — there's no uniform-inner shape Option B
                          // can target.
                          llvm::DenseMap<Value, llvm::StringMap<Value>>
                              localTrackedPodValues;
                          changed |=
                              extractCallsFromScfIf(b, localTrackedPodValues);
                        }
                        // No remaining branch — every (hasArrayOfPods,
                        // hasPodBlockArg) combination is dispatched
                        // above.
                        changed |= runBlockPhase("resolveArrayPodCompReads", b);
                        // Fold residual `array.read → pod.read
                        // @count/@comp/@in` chains that
                        // `resolveArrayPodCompReads` can't redirect
                        // (the dispatched call's SSA value is local to
                        // the dispatch-firing block; circom v2 emits
                        // post-loop read-back loops that re-walk the
                        // dispatch-pod array).
                        if (hasArrayOfPods) {
                          changed |= runBlockPhase(
                              "rewriteArrayPodCountCompInReads", b);
                          changed |=
                              runBlockPhase("eraseDeadPodAndCountOps", b);
                        }
                        processNested(b);
                      }
                    }
                  }
                };
                processNested(block);
              }
            }
          }
        });
      });
    }

    // Straggler flatten: `processNested` only recurses into scf.while bodies,
    // so a pod-array-carrying scf.while buried inside an scf.if branch is
    // never visited. AES `@AES256Encrypt_6::compute` has its xor_2 / xor_3
    // input-pod carries threaded through `scf.if` branches at depth 5, and
    // the main loop above leaves them as raw `<D x !pod<[@a, @b]>>`. Here we
    // walk the module looking for any remaining pod-array iter-arg and
    // re-invoke `flattenPodArrayWhileCarry` on the containing block.
    // Combined with the field-type fallback above, this closes the chain
    // regardless of nesting.
    {
      bool stragglerChanged = true;
      while (stragglerChanged) {
        stragglerChanged = false;
        llvm::SmallSetVector<Block *, 8> blocksToFlatten;
        module.walk([&](scf::WhileOp w) {
          for (unsigned i = 0; i < w.getNumResults(); ++i) {
            Type ty = w.getResult(i).getType();
            // NOLINTNEXTLINE(readability/braces)
            if (auto at = dyn_cast<llzk::array::ArrayType>(ty))
              if (isa<llzk::pod::PodType>(at.getElementType())) {
                blocksToFlatten.insert(w->getBlock());
                break;
              }
          }
        });
        for (Block *b : blocksToFlatten)
          stragglerChanged |= runBlockPhase("flattenPodArrayWhileCarry", *b);
      }
    }

    // Straggler scf.if + scf.while pod-array flattening, run as a single
    // fixed-point. scf.ifs whose result type list still contains
    // `<D x !pod<[@a, @b, ...]>>` survive the main while-carry flatten because
    // `flattenPodArrayWhileCarry` only walks scf.while iter-args. AES rounds-
    // loop has scf.ifs nested inside an already-flattened outer scf.while
    // whose pod-array result slots blocked the per-field carry from threading
    // through to outer scf.yields. After the scf.if rewrite, the new per-field
    // felt-array result slots become promotable carries
    // (`isPromotableCarryType` excludes pod-element arrays but accepts felt-
    // element arrays); LlzkToStablehlo's `extendResultBearingScfIfArrayChain`
    // then walks each branch, finds the inner scf.while's per-field SSA
    // carries, and rewrites the branch yields to use the latest values.
    //
    // The two ops are folded into one walk so a single iteration also catches
    // scf.while iter-args newly exposed when an scf.if's pod-array result fed
    // an outer scf.yield → outer scf.while iter-arg chain (and vice versa,
    // for completeness).
    {
      bool changed = true;
      while (changed) {
        changed = false;
        llvm::SmallSetVector<Block *, 8> blocksToFlattenIf;
        llvm::SmallSetVector<Block *, 8> blocksToFlattenWhile;
        module.walk([&](Operation *op) {
          if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
              auto at =
                  dyn_cast<llzk::array::ArrayType>(ifOp.getResult(i).getType());
              if (at && isa<llzk::pod::PodType>(at.getElementType())) {
                blocksToFlattenIf.insert(ifOp->getBlock());
                break;
              }
            }
          } else if (auto w = dyn_cast<scf::WhileOp>(op)) {
            for (unsigned i = 0; i < w.getNumResults(); ++i) {
              auto at =
                  dyn_cast<llzk::array::ArrayType>(w.getResult(i).getType());
              if (at && isa<llzk::pod::PodType>(at.getElementType())) {
                blocksToFlattenWhile.insert(w->getBlock());
                break;
              }
            }
          }
        });
        for (Block *b : blocksToFlattenIf)
          changed |= runBlockPhase("flattenPodArrayScfIfResults", *b);
        for (Block *b : blocksToFlattenWhile)
          changed |= runBlockPhase("flattenPodArrayWhileCarry", *b);
      }
    }

    // Post-flatten rewire: connect each scf.while's per-field block args to
    // nested whiles' per-field nondet inits. The inner rewire inside
    // `expandPodArrayWhile` (`rewritePodArrayUsesInBlock`) only fires when
    // the nested while has *already* been flattened by a prior
    // outer-fixed-point iteration; when the parent is flattened first, the
    // nested still carries the OLD pod-array type at that moment, so the
    // inner rewire's "contiguous run of llzk.nondet of per-field types"
    // match fails. Once the outer fixed point has settled and `processNested`
    // has flattened the nested whiles too, both sides have per-field
    // felt-typed carries, but the nested's inits are LOCAL nondets (created
    // by the nested's own `expandPodArrayWhile`) and never rewired back to
    // the parent's per-field block args. This module-level pass closes that
    // chain.
    //
    // Algorithm: for every scf.while parent and every nested scf.while
    // inside parent's body region, find each contiguous run of `llzk.nondet`
    // operands of flattenable felt / felt-array types in the nested's
    // init list, then find an unclaimed contiguous run of parent block args
    // with the same type sequence and rewire `nested.setOperand` to point
    // at parent block args. Same start+N pattern as the inner rewire; the
    // claimed-position set disambiguates when parent has multiple per-field
    // groups of identical types (e.g. xor_a (@a, @b) followed by xor_b
    // (@a, @b) of matching shape).
    //
    // Convergence: this only `setOperand`s on existing scf.whiles — no pod
    // ops added or removed. `eliminatePodDispatch` would not reach a
    // different fixed point even if re-run, so we run AFTER the main loop
    // has settled and don't need to revisit dispatch elimination.
    // Visit each scf.while as a `parent` and rewire ONLY its immediate child
    // scf.whiles' nondet inits to parent's per-field block args. Iterate the
    // block non-recursively — a deeper-nested while is the immediate child
    // of its own parent in module.walk's later visit, not of an outer
    // ancestor. No fixed point needed: this only setOperands existing ops,
    // and rewiring a nested to its parent's block args doesn't expose new
    // nondet operands elsewhere.
    module.walk([&](scf::WhileOp parent) {
      for (Region &r : parent->getRegions()) {
        if (r.empty())
          continue;
        Block &blk = r.front();
        // Collect direct-child scf.whiles plus those reachable through
        // scf.if/scf.for branches without crossing a deeper scf.while
        // boundary. AES rounds-loop has per-field carrier-bearing
        // scf.whiles at depth 4 (rounds-loop body → scf.if branch → scf.if
        // branch → scf.while), introduced by `flattenPodArrayScfIfResults`
        // after each enclosing scf.if's pod-array result slot was
        // rewritten to per-field. Without recursion the rewire would
        // never see the depth-4 scf.while's per-field nondet inits,
        // leaving them disconnected from `parent`'s per-field block args.
        // Each inner scf.while is itself a parent in its own
        // module.walk visit, so we stop descent at scf.while boundaries
        // to avoid double-rewire (the nondet-operand check at the matcher
        // also defends against this). `Block::walk` over `blk` never visits
        // `parent` itself — parent contains blk's region rather than living
        // inside it — so no equality guard is needed.
        SmallVector<scf::WhileOp, 8> nestedWhiles;
        blk.walk([&](scf::WhileOp w) {
          nestedWhiles.push_back(w);
          return WalkResult::skip();
        });
        for (scf::WhileOp nested : nestedWhiles) {
          llvm::DenseSet<unsigned> claimedBaPositions;

          auto isFlattenableNondet = [](Value v) {
            Operation *def = v.getDefiningOp();
            return def && isa<llzk::NonDetOp>(def) &&
                   isFlattenableFelt(v.getType());
          };

          // Try to commit a contiguous run of `nested` operands at
          // `[start, start+runLen)` to a same-typed unclaimed run of
          // parent block args. Returns true on commit.
          auto tryClaimRun = [&](unsigned start, unsigned runLen) -> bool {
            for (unsigned baStart = 0;
                 baStart + runLen <= blk.getNumArguments(); ++baStart) {
              bool ok = true;
              for (unsigned k = 0; k < runLen && ok; ++k) {
                ok = !claimedBaPositions.count(baStart + k) &&
                     blk.getArgument(baStart + k).getType() ==
                         nested->getOperand(start + k).getType();
              }
              if (!ok)
                continue;
              for (unsigned k = 0; k < runLen; ++k) {
                nested->setOperand(start + k, blk.getArgument(baStart + k));
                claimedBaPositions.insert(baStart + k);
              }
              return true;
            }
            return false;
          };

          // First pass — full contiguous mixed-type match. A single
          // per-pod-array group whose per-field carries occupy adjacent
          // positions in BOTH inner's operand list AND parent's block
          // args (after `flattenPodArrayWhileCarry`'s record-order
          // resort) commits as one run. Canonical case: maci_splicer's
          // QuinSelector_6 @in fill loop nested in @main — inner's
          // `[5x5 felt, 5 felt]` for `[@in, @index]` matches parent's
          // adjacent same sequence. Without this pass the homogeneous
          // fallback below picks the first unclaimed same-type arg,
          // which for chips with multiple 5-felt per-field carries
          // (mux's @s) is the wrong group.
          for (unsigned start = 0; start < nested->getNumOperands();) {
            unsigned end = start;
            while (end < nested->getNumOperands() &&
                   isFlattenableNondet(nested->getOperand(end)))
              ++end;
            if (end - start < 2) {
              start = end == start ? start + 1 : end;
              continue;
            }
            start = tryClaimRun(start, end - start) ? end : start + 1;
          }

          // Second pass — homogeneous-type sub-runs. AES rounds-loop is
          // the canonical case: 4 adjacent nondets typed
          // <13,4,3,32>×2 + <13,4,32>×2 must be matched as two
          // independent type-homogeneous runs against parent body args
          // at non-contiguous positions [0,1] + [8,9].
          for (unsigned start = 0; start < nested->getNumOperands();) {
            unsigned end = start;
            Type runType;
            while (end < nested->getNumOperands() &&
                   isFlattenableNondet(nested->getOperand(end))) {
              Type ty = nested->getOperand(end).getType();
              if (end == start)
                runType = ty;
              else if (ty != runType)
                break;
              ++end;
            }
            if (end == start) {
              ++start;
              continue;
            }
            tryClaimRun(start, end - start);
            start = end;
          }
        }
      }
    });

    // Post-step: inline single-field input pods used as `scf.while` carry
    // (e.g. `!pod.type<[@claim: !array<8 x felt>]>`) to the raw inner
    // type. These are orthogonal to dispatch pods (no `@count` / `@comp`
    // fields) and are not caught by the dispatch rewrite, but they still
    // need to be gone before template removal runs `applyFullConversion`.
    if (needsV2Prereqs)
      runModulePhase("inlineInputPodCarries", module);

    // Post-step: scrub residual pod ops module-wide. `eliminatePodDispatch`
    // tracks pod-field values per-block and per-pod-SSA-value; circom v2
    // emits scalar dispatch pods (`pod.new { @count = N }`) at the
    // `@compute` body level whose `pod.read`s live many regions deep
    // (inside `scf.while` / `scf.for` bodies that iterate the dispatch),
    // so the per-block tracker never sees them and Phase 5's
    // non-recursive walk leaves them behind. After
    // `inlineInputPodCarries` has already inlined the input-pod carries
    // it can match (single-field, while-threaded), anything still
    // referencing a `pod.*` op is structural bookkeeping whose witness
    // value is don't-care for our lowering — replace `pod.read` with
    // `llzk.nondet` of the result type, then DCE `pod.write`/`pod.new`
    // chains. Runs AFTER `inlineInputPodCarries` so that pass can still
    // discover the inner type from a live `pod.read`.
    if (needsV2Prereqs) {
      SmallVector<Operation *> podReadsToErase;
      SmallVector<Operation *> podWritesToErase;
      module.walk([&](Operation *op) {
        if (isa<llzk::pod::WritePodOp>(op)) {
          podWritesToErase.push_back(op);
          return;
        }
        if (!isa<llzk::pod::ReadPodOp>(op) || op->getNumResults() == 0)
          return;
        OpBuilder b(op);
        Type rty = op->getResult(0).getType();
        Value replacement;
        if (rty.isIndex()) {
          // `llzk.nondet : index` is illegal in the dialect-conversion
          // target. Substitute `arith.constant 0` for index-typed
          // dispatch-pod `@count` reads (same justification as
          // `rewriteArrayPodCountCompInReads` for the scf.while case).
          OperationState state(op->getLoc(), "arith.constant");
          state.addAttribute("value", b.getIndexAttr(0));
          state.addTypes({rty});
          replacement = b.create(state)->getResult(0);
        } else {
          replacement = createNondet(b, op->getLoc(), rty);
        }
        op->getResult(0).replaceAllUsesWith(replacement);
        podReadsToErase.push_back(op);
      });
      for (Operation *op : podReadsToErase)
        op->erase();
      for (Operation *op : podWritesToErase)
        op->erase();

      // `erasePodTypedCarrierSlots` rebuilds carrier ops to drop pod-typed
      // iter/result slots whose only consumers are other pod.new chains
      // or surviving carrier-forwarding terminators. The standalone
      // pod.new DCE that follows catches any orphan it missed (e.g. a
      // pod.new whose pre-cleanup result use was a now-erased pod.read).
      // The pod-typed `llzk.nondet` DCE clears orphan placeholders the
      // pod.read substitution above left behind once the carrier slot
      // drop severed all their downstream consumers — `llzk.nondet : !pod`
      // is dialect-conversion-illegal at LlzkToStablehlo so any survivor
      // would trip the next pass. Iterate the trio to a fixed point:
      // dropping a carrier slot may unlock a new pod.new for DCE, and
      // erasing a pod.new chain may unlock new carrier slots or new
      // orphan nondets for the next round.
      bool changedCleanup = true;
      while (changedCleanup) {
        changedCleanup = false;
        changedCleanup |= runModulePhase("erasePodTypedCarrierSlots", module);
        bool dcePodNew = true;
        while (dcePodNew) {
          dcePodNew = false;
          SmallVector<Operation *> deadOrphans;
          module.walk([&](Operation *op) {
            bool isCandidate =
                isa<llzk::pod::NewPodOp>(op) ||
                (isa<llzk::NonDetOp>(op) && op->getNumResults() == 1 &&
                 isa<llzk::pod::PodType>(op->getResult(0).getType()));
            if (isCandidate && isAllResultsUnused(*op))
              deadOrphans.push_back(op);
          });
          for (Operation *op : deadOrphans) {
            op->erase();
            dcePodNew = true;
            changedCleanup = true;
          }
        }
      }
    }

    // Late lift: hoist single-instance dispatch calls out of their
    // writerWhile bodies. By this point all pod.read chains have
    // settled into `array.extract %ba[const]` patterns the lift's
    // operand resolver can recognize — the same call inside an N-iter
    // scf.while runs N times structurally, but writes its result to a
    // const-index destArr cell where last-write-wins makes the first
    // N-1 iters dead. The lift collapses to one post-while call.
    // Drives a small fixed point because lifting one chain may expose
    // another in an outer while (rare but correct under conservative
    // gating).
    {
      bool liftChanged = true;
      while (liftChanged)
        liftChanged = liftConstIndexPodArrayCallPostWhile(module);
    }

    // Post-step: unwrap LLZK v2 `poly.template` shells and collapse empty
    // parameter lists on struct type refs (`@X<[]>` → `@X`). Runs AFTER
    // dispatch cleanup and while-carry inlining so the `pod.new`/
    // `pod.read`/`pod.write` ops that circom v2 emits have already been
    // DCE'd — the upstream pass's `applyFullConversion` has no target
    // pattern for pod ops and would bail out otherwise. Skip when no
    // `poly.template` is present (hand-written LIT fixtures bypass the
    // template wrapping).
    {
      bool hasTemplate = false;
      module.walk([&](llzk::polymorphic::TemplateOp) {
        hasTemplate = true;
        return WalkResult::interrupt();
      });
      if (hasTemplate) {
        // Pre-strip `<[]>` from `llzk.nondet` result types before template
        // removal. The upstream pass walks only the ops in its
        // `OpClassesWithStructTypes` tuple; `llzk.nondet` is not in that
        // list, so any SSA value SSC's dispatch cleanup synthesized above
        // with a `!struct.type<@X::@X<[]>>` result would keep its
        // pre-strip form, produce an unrealized `conversion_cast` against
        // the stripped form expected by downstream `struct.readm`, and
        // bail the pass's `applyFullConversion`.
        runModulePhase("stripEmptyStructParams", module);

        OpPassManager pm("builtin.module");
        pm.addPass(llzk::polymorphic::createEmptyTemplateRemoval());
        if (failed(runPipeline(pm, module))) {
          signalPassFailure();
          return;
        }

        // Post-step: project-llzk/circom PR #378 wraps every emitted
        // function.def / struct.def in a same-named `poly.template` so
        // upstream passes can track polymorphic typing. After
        // EmptyTemplateRemoval converts those to `builtin.module @X
        // { function.def @X }` (or `struct.def @X`), the inner symbol
        // collides with the wrapping module on the next pass that walks
        // the parent's symbol table (LlzkToStablehlo trips with
        // "redefinition of symbol named '<X>'"). Hoist each child out of
        // its single-purpose wrapper, then erase the wrapper. Symbol
        // refs still resolve because the inner @X kept its name and the
        // wrapper had no semantically-load-bearing identity.
        runModulePhase("flattenSingleEntityWrapperModules", module);
      }
    }

    // NOTE: constrain function body clearing causes crashes in circuits
    // with sub-component function.call chains (multimimc7, mimcsponge_wrap).
    // Left as future work — constrain clearing needs careful handling of
    // cross-function references and verification constraints.

    // End-of-pass structural post-condition (debug builds only, no-op under
    // NDEBUG). The while-carry flatten + unpack phases above exist precisely
    // to drain pod-typed scf.while carries to zero; a reorder or accidental
    // removal of one would leave a pod-typed carry here and silently
    // miscompile. Assert at the true end-of-pass — after the outer fixed
    // point AND the straggler flatten passes — because intermediate IR
    // legitimately carries pod-typed whiles mid-elimination.
    assertNoPodTypedWhileCarry(module);
  }
};

} // namespace

} // namespace mlir::llzk_to_shlo
