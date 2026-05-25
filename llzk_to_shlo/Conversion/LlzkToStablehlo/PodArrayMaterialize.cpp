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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/PodArrayMaterialize.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/SimplifySubComponentsInternal.h"
#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::llzk_to_shlo {

/// Materialize a per-struct-field felt array for every function-scope
/// `array.new : <N x !pod<[..., @comp: !struct, ...]>>` whose `@comp`
/// slots are written by a hoisted `function.call` result in one block and
/// read back in a *different* block via `pod.read [@comp]; struct.readm
/// [@F] : !felt`.
///
/// Circomlib's `gates.circom` `XorArray*` / `AndArray*` family (used in
/// keccak chi/iota/rhopi/round/squeeze/theta + AES + iden3 + SHA-256)
/// emits the writer/reader split as two sibling `scf.while` ops over the
/// same dispatch-pod array. `extractCallsFromScfIf`'s tracker is keyed by
/// pod SSA value, and the writer's `%dp = array.read %arr[%i]` produces a
/// different SSA value than the reader's `%dp2 = array.read %arr[%j]`,
/// so the call result is not forwarded across the array slot. Without
/// intervention `rewriteArrayPodCountCompInReads` (and Phase 5 as a
/// backstop) substitutes the reader's `pod.read %dp2[@comp]` with
/// `llzk.nondet`, severing the data flow and silently emitting zeros.
///
/// This pass allocates a per-struct-field `%arr_F = array.new : <N x
/// !felt>` parallel to `%arr` for each field `F` consumed by a
/// `struct.readm` reader, hoists the dispatch `function.call` out of the
/// writer's `scf.if`, inserts `array.write %arr_F[%i] = struct.readm
/// %callResult[@F]` after the scf.if at body level, and rewrites every
/// reader-side `struct.readm [@F]` into `array.read %arr_F[%j]`. The
/// original pod array's `pod.write @comp` / `pod.read @comp` / `array.write`
/// traffic is left in place — Phase 4 DCEs the dispatch-firing scf.if (its
/// only effect was the dead pod.write/array.write pair) along with the
/// reader-side pod.read once its struct.readm consumer is gone.
///
/// We materialize at the felt level (not at the !struct level) because
/// `convertWhileBodyArgsToSSA` (in `LlzkToStablehlo.cpp`) promotes any
/// captured non-pod-element array used inside `scf.while` to an iter-arg
/// and converts each `array.write` to a result-bearing form. `array.write`
/// is `ZeroResults`, so the result-bearing rewrite only verifies once
/// `ArrayWritePattern` rewrites it to `stablehlo.dynamic_update_slice`.
/// `ArrayWritePattern` is gated on `involvesPod`, which treats `!struct`
/// element arrays as pod-involved (legal, *not* converted) — so a
/// `<N x !struct>` array.write inside an scf.while body is left
/// result-bearing and trips the verifier. The felt element type sidesteps
/// that path entirely.
///
/// The transform fires only when at least one writer (call-result-paired
/// `pod.write [@comp]` directly preceding `array.write %arr[*]`) and at
/// least one cross-block reader whose `pod.read [@comp]` is consumed by a
/// `struct.readm [@F]` exist. Same-block readers are gated out — those
/// are already forwarded by `resolveArrayPodCompReads` via the
/// per-block call-result-by-type map. Readers whose `pod.read [@comp]`
/// is consumed by something other than `struct.readm` (e.g. an
/// `array.write %array_3[%i] = %comp` extracting the struct into a
/// `struct.member @aux` array) are also skipped — those represent
/// don't-care bookkeeping that the existing nondet path handles
/// correctly. Idempotent: subsequent invocations find no readers and
/// skip.

// Candidate function-scope pod-array whose `@comp` field is a struct type
// (the only case where the cross-while bug currently bites — `gates.circom`
// helpers all dispatch a struct-returning component).
struct PacfCandidate {
  Operation *arrNew;
  Type compType;
};

struct PacfWriter {
  SmallVector<Value> outerIndices; // body-scope indices used to read %arr.
  Operation *insertAfter;          // body-block ancestor of the writer.
  Value callResult;    // function.call result fed into pod.write[@comp].
  Operation *podWrite; // pod.write %cell[@comp] = %callResult, sited
                       // in writer's scf.if body. Captured here so the
                       // single-instance fold below can erase it
                       // directly without re-scanning the call's uses.
};

// Cross-block reader: `%dp2 = array.read %arr[%j]; %comp = pod.read
// %dp2[@comp]; struct.readm %comp[@F] : !felt`. We materialize one
// per-field felt array per distinct F.
struct PacfReader {
  Operation *arrayRead;   // %arr[%j] — its block + operand 1 (index)
                          // are the same-block guard key + the
                          // body-scope index for the rewritten read.
  Operation *structReadm; // the felt-yielding consumer to rewrite.
  StringRef field;        // struct field name, key into per-field arrays.
};

// Cross-block drain reader: `%dp2 = array.read %arr[%j]; %comp =
// pod.read %dp2[@comp]; array.write %destArr[%j] = %comp` where
// `%destArr : array<D x !struct>` flows into `struct.writem
// %self[@F'] = %destArr` on the parent struct. The parent member's
// witness slot is sized by `getMemberFlatSize` as the array's dim
// count (one felt per cell), so we materialize a parallel felt
// array, populate it from the writer-side `function.call` results
// by extracting the inner struct's single felt-typed member (e.g.
// `@out`), and redirect the parent `struct.writem` to consume the
// felt array. The original struct-array drain stays in place; its
// `pod.read [@comp]` is nondet'd by Phase 5 as before, and the now-
// unused `%destArr = array.new` flows into a writem that no longer
// uses it (DCE'd later).
struct PacfDrainReader {
  Operation *arrayRead;     // %arr[%j] (the dispatch-pod array read).
  Operation *arrayWriteDst; // array.write %destArr[*] = %comp.
  Value destArr;            // %destArr (struct-element array).
  Operation *writem;        // struct.writem %self[@F'] = %destArr.
  StringRef parentField;    // @F' name (key into parent struct.def).
};

struct PacfPubFelt {
  StringRef field;
  Type ty;
};

// K=0 path: inner struct has zero pub felt members but at least one
// writem-targeted, non-pod, non-zero-flat member. The inner contribution
// is the concat of each writem-targeted member's recursive flat felts
// (e.g. webb's `@ManyMerkleProof_275 → @hasher : <30 x !felt>` (30 felts)
// + `@switcher : <30, 2 x !felt>` (60 felts) = 90 per instance). Member
// shapes are *heterogeneous*, so each member's natural per-element
// layout (row-major across its declared dims) is unrolled at writer
// sites into per-element `array.write` into the flat destFelt at offset
// `offsetWithinInstance + linear_idx`.
struct PacfWritemMember {
  StringRef field;
  Type ty;
  int64_t flatSize;
  int64_t offsetWithinInstance; // cumulative offset of this member
                                // within one inner-struct instance
};

struct PacfDrainPlan {
  Value destFelt;       // Parallel destination — type follows
                        // `combineDispatchAndInnerFeltDims(combinedInnerTy)`
                        // for K=1/K>1 paths, or `<destDims..., totalFlat x
                        // !felt>` for the K=0 recursive flatten path.
  Type combinedInnerTy; // K=1: `pubFelts[0].ty` (preserves single-pub
                        // byte-layout). K>1: `!array<K x ...inner>`
                        // (one extra K dim prepended). K=0:
                        // `!array<totalFlat x !felt>` (flat per-instance
                        // concat of writem-targeted member contents).
  SmallVector<PacfPubFelt, 2> pubFelts; // Pub members in declaration order.
                                        // Empty for the K=0 recursive path.
  SmallVector<Value, 2> kIndices;       // K=1: empty. K>1: K shared
                                        // `arith.constant <j> : index` Values
                                        // emitted once at destFelt allocation
                                        // (function-block-dominant) and reused
                                        // across every writer site + @compute
                                        // reader. Single-instance fold keeps
                                        // the surviving writer's `insertAfter`
                                        // dominated by these.
  SmallVector<PacfWritemMember, 2> recursiveMembers; // K=0 path:
                                                     // writem-targeted members.
                                                     // Empty otherwise.
  int64_t totalFlat = 0; // K=0 path: sum of `recursiveMembers[*].flatSize`.
                         // Zero on K>=1 paths (unused).
  Type structArrTy;      // Original array<D x !struct.type<@Sub>>.
  // True when destFelt was reused from `perFieldArrays[innerField]`
  // (Loop A's reader-side allocation). In that case Loop A already
  // emits the `array.write %perFieldArrays[innerField][i] = struct.readm
  // %callResult[@innerField]` per writer site, and Loop B (drain
  // emission) must skip its `array.write` to avoid duplicating into the
  // same array at the same indices. The duplicate-write pair was the
  // smoking gun behind the AES `aes_256_encrypt` [0,128) ciphertext
  // residual — see memory/aes-encrypt-mod16-residual-followup.md.
  // Reuse only fires for K=1 — K>1's combined `<D, K x ...>` shape
  // never matches a per-field `<D x ...>` allocation by Loop A.
  bool reusedFromPerField = false;
};

// Index-typed bit width used to normalize felt/index constant APInts before
// comparison in the single-instance fold (see fold site for rationale).
constexpr unsigned kIndexBitWidth = 64;

// Populates `out` with the pub felt members of `structTy` in declaration
// order when its struct.def has at least one `{llzk.pub}` member whose type
// is `!felt` or `!array<... x !felt>`. When K>1, all members must share the
// same type (uniform-shape constraint — see PacfDrainPlan comment).
static bool findInnerFeltMembers(Block &funcBlock,
                                 llzk::component::StructType structTy,
                                 SmallVectorImpl<PacfPubFelt> &out) {
  out.clear();
  ModuleOp moduleOp = getTopLevelModule(funcBlock);
  if (!moduleOp)
    return false;
  llzk::component::StructDefOp foundDef =
      findStructDefByExactSymbol(moduleOp, structTy);
  if (!foundDef)
    return false;
  // Public is the canonical "this is the witness output" marker —
  // structs like `MiMC7_0` expose felt-or-felt-array internals
  // (`@t2/@t4/@t6/@t7`) alongside the pub `@out`; without the pub
  // filter, widening acceptance to felt-array would make the count
  // ambiguous and silently drop the case.
  foundDef->walk([&](Operation *m) {
    if (!isa<llzk::component::MemberDefOp>(m))
      return;
    auto memTy = m->getAttrOfType<TypeAttr>("type");
    if (!memTy || !isFlattenableFelt(memTy.getValue()))
      return;
    if (!m->hasAttr(llzk::PublicAttr::name))
      return;
    auto sym = m->getAttrOfType<StringAttr>("sym_name");
    if (!sym)
      return;
    out.push_back({sym.getValue(), memTy.getValue()});
  });
  if (out.empty())
    return false;
  // Mixed-shape multi-pub would need a flat-felt concat path
  // (`<D, totalFlat x !felt>`). No bucket-1 chip exhibits that
  // shape today — Switcher is all-scalar and BitElementMulAny is
  // all-`<2 x !felt>` — so reject and let the caller fall back to
  // the existing nondet path until evidence demands otherwise.
  if (out.size() > 1) {
    Type first = out.front().ty;
    for (const PacfPubFelt &pf : out)
      if (pf.ty != first) {
        out.clear();
        return false;
      }
  }
  return true;
}

// K=0 path helper. Collects writem-targeted, non-pod members of `structTy`
// in declaration order with their recursive flat sizes, and (if `promote` is
// true) promotes those members to `{llzk.pub}` on the inner struct.def so
// subsequent `struct.readm` from outside the inner struct is legal under
// LLZK's MemberReadOp verifier (which rejects external reads of private
// members).
//
// Returns true when at least one writem-targeted member contributes
// non-zero recursive flat (i.e. the K=0 path has real content to
// drain). The "no pub" filter is intentionally *omitted* — the
// caller decides which path to take based on `findInnerFeltMembers`
// first; this fallback applies only when that path rejected.
//
// For Phase 3 single-level support, struct-typed writem-targeted
// members with non-zero recursive flat are *rejected* — multi-level
// recursion (emitting per-level struct.readm chains at writer sites)
// is not implemented today. Zero-flat struct-typed members (e.g.
// webb's `@indexBits` / `@set` whose inner Num2Bits_205 /
// ForceSetMembershipIfEnabled_274 contribute zero felts in MMP's
// writem-target set) are silently skipped.
static bool collectAndPromoteRecursiveWritemMembers(
    Block &funcBlock, llzk::component::StructType structTy,
    SmallVectorImpl<PacfWritemMember> &out, bool promote) {
  out.clear();
  ModuleOp moduleOp = getTopLevelModule(funcBlock);
  if (!moduleOp)
    return false;
  llzk::component::StructDefOp foundDef =
      findStructDefByExactSymbol(moduleOp, structTy);
  if (!foundDef)
    return false;
  llvm::DenseSet<StringAttr> writemSet = collectWritemTargets(foundDef);
  int64_t offset = 0;
  bool rejected = false;
  SmallVector<Operation *, 2> memberOps;
  foundDef->walk([&](Operation *m) {
    if (rejected || !isa<llzk::component::MemberDefOp>(m))
      return;
    auto sym = m->getAttrOfType<StringAttr>("sym_name");
    if (!sym || !writemSet.count(sym))
      return;
    auto memTy = m->getAttrOfType<TypeAttr>("type");
    if (!memTy)
      return;
    Type ty = memTy.getValue();
    // Skip pod-typed members at any ShapedType depth (mirrors
    // `isPodMemberType` in TypeConversion.cpp).
    Type leafTy = ty;
    while (auto shaped = dyn_cast<ShapedType>(leafTy))
      leafTy = shaped.getElementType();
    if (isa<llzk::pod::PodType>(leafTy))
      return;
    // Struct-typed writem-targets with non-zero recursive flat require
    // multi-level recursion — Phase 3 single-level scope rejects.
    if (LlzkToStablehloTypeConverter::isStructType(ty)) {
      int64_t flat = getMemberFlatSize(ty, moduleOp);
      if (flat > 0) {
        rejected = true;
        return;
      }
      // Zero-flat struct member contributes nothing — skip.
      return;
    }
    int64_t flat = getMemberFlatSize(ty, moduleOp);
    if (flat == 0)
      return;
    out.push_back({sym.getValue(), ty, flat, offset});
    offset += flat;
    memberOps.push_back(m);
  });
  if (rejected) {
    out.clear();
    return false;
  }
  if (out.empty())
    return false;
  // LLZK's `MemberReadOp::verifySymbolUses` rejects external reads of
  // private members (see llzk struct dialect Ops.cpp). The writer-emit
  // step below issues `struct.readm %callResult[@<field>]` from inside
  // the *parent* struct's @compute, which is "outside" by that
  // verifier's notion. Promote each collected member to `{llzk.pub}`
  // on the inner struct.def so the read is legal.
  //
  // Semantic argument: WLA already counts these slots in the parent's
  // witness-layout footprint via `getMemberFlatSize`'s recursive walk
  // (PR #99). Promoting them to pub aligns the LLZK struct visibility
  // contract with the WLA exposure already in effect — the witness
  // layer treats them as observable; pub makes the LLZK verifier
  // agree.
  if (promote) {
    for (Operation *m : memberOps) {
      if (!m->hasAttr(llzk::PublicAttr::name))
        m->setAttr(llzk::PublicAttr::name, UnitAttr::get(m->getContext()));
    }
  }
  return true;
}

// Resolve each index Value in `indices` to a constant APInt, looking through
// `cast.toindex` chains. Returns nullopt if any index isn't a felt/arith
// constant. Felt APInts come out of `FeltConstAttr` at the literal's
// minimum-bits-needed width; arith.constant-index APInts are 64-bit.
// Comparing them directly trips `APInt::operator==`'s "equal bit widths"
// assertion, so normalize to `kIndexBitWidth` via `zextOrTrunc` (loss-free
// for valid array offsets, same precedent as `TypeConversion.cpp`).
static std::optional<SmallVector<llvm::APInt>>
outerIndexConstValues(ArrayRef<Value> indices) {
  SmallVector<llvm::APInt> out;
  for (Value idx : indices) {
    Operation *def = idx.getDefiningOp();
    while (def && isa<llzk::cast::FeltToIndexOp>(def) &&
           def->getNumOperands() >= 1)
      def = def->getOperand(0).getDefiningOp();
    if (!def)
      return std::nullopt;
    // `felt.const` carries a custom `FeltConstAttr` (APInt + FeltType),
    // not a plain `IntegerAttr`. Standard `arith.constant` paths use
    // `IntegerAttr` on integer-typed results — accept either so the
    // fold keeps working if someone canonicalizes a `cast.toindex`
    // chain into a plain `arith.constant index` upstream.
    if (isa<llzk::felt::FeltConstantOp>(def)) {
      auto attr = def->getAttr("value");
      if (auto feltConst = dyn_cast_or_null<llzk::felt::FeltConstAttr>(attr)) {
        out.push_back(feltConst.getValue().zextOrTrunc(kIndexBitWidth));
        continue;
      }
      return std::nullopt;
    }
    if (isa<arith::ConstantOp>(def)) {
      auto intAttr = def->getAttrOfType<IntegerAttr>("value");
      if (!intAttr)
        return std::nullopt;
      out.push_back(intAttr.getValue().zextOrTrunc(kIndexBitWidth));
      continue;
    }
    return std::nullopt;
  }
  return out;
}

// True iff `afterOp` strictly follows `beforeOp` in their smallest common
// enclosing block. Returns false when no common block exists (different
// scf.if regions etc.) — caller treats that as "ordering unprovable" and
// bails out of the fold.
static bool strictlyAfter(Operation *afterOp, Operation *beforeOp) {
  if (!afterOp || !beforeOp)
    return false;
  for (Block *blk = afterOp->getBlock(); blk;
       blk = blk->getParentOp() ? blk->getParentOp()->getBlock() : nullptr) {
    Operation *afterAnc = blk->findAncestorOpInBlock(*afterOp);
    Operation *beforeAnc = blk->findAncestorOpInBlock(*beforeOp);
    if (afterAnc && beforeAnc)
      return beforeAnc->isBeforeInBlock(afterAnc);
  }
  return false;
}

// Collect transitive operand defs of `v` that live under `insertAfter` in
// operands-before-uses order, into `ordered` (using `seen` for dedup).
// Building the order during traversal (recurse first, then append) avoids a
// second pass walking every op in the ancestor's regions just to find which
// are `needed` — ~6× fewer op visits across keccak's 48 dispatch fires.
static void collectHoistDefs(Operation *insertAfter, Value v,
                             llvm::DenseSet<Operation *> &seen,
                             SmallVectorImpl<Operation *> &ordered) {
  Operation *def = v.getDefiningOp();
  if (!def || !seen.insert(def).second)
    return;
  // Leave external defs in `seen` so subsequent visits early-return
  // at the `insert` check instead of re-running `isAncestor` and
  // recursing into their (possibly large) operand chains.
  if (!insertAfter->isAncestor(def))
    return;
  for (Value operand : def->getOperands())
    collectHoistDefs(insertAfter, operand, seen, ordered);
  ordered.push_back(def);
}

// Collect candidate function-scope pod-arrays whose `@comp` field is a
// struct type (the only case where the cross-while bug currently bites
// — `gates.circom` helpers all dispatch a struct-returning component).
static SmallVector<PacfCandidate> collectCandidates(Block &funcBlock) {
  SmallVector<PacfCandidate> candidates;
  for (Operation &op : funcBlock) {
    if (!isa<llzk::array::CreateArrayOp>(op) || op.getNumResults() == 0)
      continue;
    auto arrTy = dyn_cast<llzk::array::ArrayType>(op.getResult(0).getType());
    if (!arrTy)
      continue;
    auto podTy = dyn_cast<llzk::pod::PodType>(arrTy.getElementType());
    if (!podTy)
      continue;
    Type compType;
    for (auto rec : podTy.getRecords())
      if (rec.getName() == "comp") {
        compType = rec.getType();
        break;
      }
    if (compType && isa<llzk::component::StructType>(compType))
      candidates.push_back({&op, compType});
  }
  return candidates;
}

// Classify uses of the dispatch pod-array `arr`: collect writers (the
// call-result-paired `pod.write[@comp]` directly preceding `array.write
// %arr[*]`), cross-block struct.readm readers (one per distinct field),
// and cross-block struct.writem drain readers. A trailing same-block prune
// drops drain readers nested in a writer body (forwarded same-block by
// `eliminatePodDispatch`); reader same-block pruning is intentionally NOT
// applied — see the prune comment.
static void classifyArrUses(Value arr, SmallVectorImpl<PacfWriter> &writers,
                            SmallVectorImpl<PacfReader> &readers,
                            llvm::StringMap<Type> &fieldFeltTypes,
                            SmallVectorImpl<PacfDrainReader> &drainReaders) {
  for (OpOperand &use : arr.getUses()) {
    Operation *user = use.getOwner();
    if (use.getOperandNumber() != 0)
      continue;
    if (isa<llzk::array::WriteArrayOp>(user) && user->getNumOperands() >= 3) {
      // For multi-dim arrays the value is always the last operand,
      // preceded by 1+ index operands. Hard-coding `getOperand(2)`
      // would silently grab an index for any rank > 1.
      Value writtenPod = user->getOperand(user->getNumOperands() - 1);
      Operation *podWrite = nullptr;
      for (Operation *prev = user->getPrevNode(); prev;
           prev = prev->getPrevNode()) {
        if (!isa<llzk::pod::WritePodOp>(prev) || prev->getNumOperands() < 2 ||
            prev->getOperand(0) != writtenPod)
          continue;
        auto rn = prev->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (rn && rn.getValue() == "comp") {
          podWrite = prev;
          break;
        }
      }
      if (!podWrite)
        continue;
      Operation *callDef = podWrite->getOperand(1).getDefiningOp();
      if (!callDef || !isa<llzk::function::CallOp>(callDef))
        continue;

      // %writtenPod must be defined by an outer `array.read %arr[idx...]`.
      Operation *podDef = writtenPod.getDefiningOp();
      if (!podDef || !isa<llzk::array::ReadArrayOp>(podDef) ||
          podDef->getNumOperands() < 2 || podDef->getOperand(0) != arr)
        continue;
      Block *bodyBlock = podDef->getBlock();
      SmallVector<Value> outerIndices = arrayAccessIndices(podDef);

      // Walk up from `user` to the immediate ancestor that lives in
      // bodyBlock; that's where the writer materialization will go.
      Operation *ancestor = user;
      while (ancestor && ancestor->getBlock() != bodyBlock)
        ancestor = ancestor->getParentOp();
      if (!ancestor)
        continue;

      writers.push_back({std::move(outerIndices), ancestor,
                         podWrite->getOperand(1), podWrite});
      continue;
    }

    if (isa<llzk::array::ReadArrayOp>(user) && user->getNumResults() > 0 &&
        user->getNumOperands() >= 2) {
      Value podVal = user->getResult(0);
      for (OpOperand &subUse : podVal.getUses()) {
        Operation *subUser = subUse.getOwner();
        if (!isa<llzk::pod::ReadPodOp>(subUser) ||
            subUser->getNumResults() == 0)
          continue;
        auto rn = subUser->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (!rn || rn.getValue() != "comp")
          continue;
        // Collect every felt-yielding `struct.readm [@F]` consumer of
        // this `pod.read [@comp]`. Multi-field structs (e.g. a gate
        // with both @out and @aux fields read in loop2) need every
        // field materialized; non-`struct.readm` consumers (e.g. the
        // `array.write %array_3[%i] = %comp` pattern that drains the
        // struct into a `struct.member @aux` array) are @constrain
        // bookkeeping and stay don't-care — the existing
        // `rewriteArrayPodCountCompInReads` nondets the surviving
        // pod.read correctly for them.
        Value compVal = subUser->getResult(0);
        for (OpOperand &compUse : compVal.getUses()) {
          Operation *consumer = compUse.getOwner();

          if (isa<llzk::component::MemberReadOp>(consumer) &&
              consumer->getNumResults() > 0) {
            auto memberAttr =
                consumer->getAttrOfType<FlatSymbolRefAttr>("member_name");
            if (!memberAttr)
              continue;
            Type feltTy = consumer->getResult(0).getType();
            // Accept scalar `!felt` OR `!array<... x !felt>`. The array
            // case (e.g. `@Num2Bits_2.@out : !array<32 x !felt>`) needs
            // the per-field array's dims to combine the dispatch dims
            // with the field's inner dims, and writer/reader emit
            // `array.insert` / `array.extract` to move whole sub-arrays
            // — see writer/reader sites below for the dispatch logic.
            bool isFelt = isa<llzk::felt::FeltType>(feltTy);
            bool isFeltArr = false;
            if (auto arrFieldTy = dyn_cast<llzk::array::ArrayType>(feltTy))
              isFeltArr =
                  isa<llzk::felt::FeltType>(arrFieldTy.getElementType());
            if (!isFelt && !isFeltArr)
              continue;
            StringRef fieldName = memberAttr.getValue();
            // All readers of the same field must agree on the field type.
            auto it = fieldFeltTypes.find(fieldName);
            if (it == fieldFeltTypes.end())
              fieldFeltTypes[fieldName] = feltTy;
            else if (it->second != feltTy)
              continue;
            readers.push_back({user, consumer, fieldName});
            continue;
          }

          // Drain consumer: `array.write %destArr[*] = %comp` where the
          // value is `%comp` (last operand) and the destination's
          // element type is a struct. The destination must flow into
          // a single `struct.writem %self[@F'] = %destArr` on the
          // parent struct — only then is there a witness slot to
          // populate.
          if (isa<llzk::array::WriteArrayOp>(consumer) &&
              consumer->getNumOperands() >= 3 &&
              consumer->getOperand(consumer->getNumOperands() - 1) == compVal) {
            Value destArr = consumer->getOperand(0);
            auto destArrTy =
                dyn_cast<llzk::array::ArrayType>(destArr.getType());
            if (!destArrTy)
              continue;
            if (!isa<llzk::component::StructType>(destArrTy.getElementType()))
              continue;
            Operation *writem = nullptr;
            StringRef parentField;
            for (OpOperand &dstUse : destArr.getUses()) {
              Operation *dstUser = dstUse.getOwner();
              if (!isa<llzk::component::MemberWriteOp>(dstUser) ||
                  dstUse.getOperandNumber() != 1)
                continue;
              auto memberAttr =
                  dstUser->getAttrOfType<FlatSymbolRefAttr>("member_name");
              if (!memberAttr)
                continue;
              if (writem) {
                // Multiple struct.writems on the same destArr — bail
                // (we'd need a single redirect target).
                writem = nullptr;
                break;
              }
              writem = dstUser;
              parentField = memberAttr.getValue();
            }
            if (!writem)
              continue;
            drainReaders.push_back(
                {user, consumer, destArr, writem, parentField});
          }
        }
      }
    }
  }

  // Materialize struct.readm consumers in any block, including the
  // writer's body. The previous same-block prune assumed
  // `resolveArrayPodCompReads`'s call-result-by-type-per-block fallback
  // would forward same-block reads safely; that fallback misforwards
  // chained-call patterns where every chain link returns the same
  // struct type (chained-XOR @b-input is the canonical case — the
  // fallback routes every prior-link reader to the LAST call's @out).
  // The per-field array path indexes by the reader's own array.read
  // indices, so a prior-link read at index `i-1` correctly consumes
  // the iter-`i-1` writer's @out, not the latest writer's.
  //
  // Drain readers (struct.writem destinations) do not exhibit the
  // chained-call shape — each iteration drains its own dispatch-pod
  // cell into its own destArr slot — so the same-block prune still
  // applies to them.
  llvm::DenseSet<Block *> writerBodyBlocks;
  for (auto &w : writers)
    writerBodyBlocks.insert(w.insertAfter->getBlock());
  drainReaders.erase(llvm::remove_if(drainReaders,
                                     [&](const PacfDrainReader &r) {
                                       return writerBodyBlocks.count(
                                           r.arrayRead->getBlock());
                                     }),
                     drainReaders.end());
}

// Materialize one `%arr_F = array.new : <N x feltType>` per distinct
// field name seen in cross-block readers, just after %arr. When the
// field type is itself a felt array (e.g. `@out : !array<32 x !felt>`),
// combine the dispatch dims with the field's inner dims so a single
// `array.insert`/`array.extract` at outer indices stores/loads the
// whole sub-array slice — same shape as `flattenPodArrayWhileCarry`'s
// per-field array construction at line ~1843.
static llvm::StringMap<Value>
allocPerFieldArrays(Operation *arrNew, llzk::array::ArrayType arrTy,
                    ArrayRef<PacfReader> readers,
                    const llvm::StringMap<Type> &fieldFeltTypes) {
  OpBuilder builder(arrNew);
  builder.setInsertionPointAfter(arrNew);
  auto dims = getArrayDimensions(arrTy);
  llvm::StringMap<Value> perFieldArrays;
  for (auto &entry : fieldFeltTypes) {
    // Skip fields no reader actually targets after same-block pruning.
    bool stillUsed = false;
    for (auto &r : readers)
      if (r.field == entry.first()) {
        stillUsed = true;
        break;
      }
    if (!stillUsed)
      continue;
    auto perFieldArrTy = combineDispatchAndInnerFeltDims(entry.second, dims);
    Value arrField = builder.create<llzk::array::CreateArrayOp>(
        arrNew->getLoc(), perFieldArrTy);
    perFieldArrays[entry.first()] = arrField;
  }
  return perFieldArrays;
}

// For each unique drain destination, plan to populate a parallel
// felt array at the writer sites (extracting the inner struct's
// single felt member from each call result), then redirect the
// parent `struct.writem` to consume the felt array and flip the
// `struct.member @F'` TypeAttr from `array<D x !struct>` to
// `array<D x !felt>`. Direct struct-element `array.write` inside
// `scf.while` bodies hits a known capture-then-lift hazard
// (`processBlockForArrayMutations` lifts the write to SSA form,
// but `ArrayWritePattern` is gated to skip pod/struct-element
// arrays — leaving an SSA-form `array.write` that fails the LLZK
// op def's zero-results constraint). Felt writes don't trip that
// path: they get lifted to SSA, then `ArrayWritePattern` rewrites
// to `stablehlo.dynamic_update_slice` cleanly. Flipping `@F'`'s
// type cascades to `@constrain` whose `struct.readm @F' →
// array.read → struct.readm @out` chain must be repaired in the
// same pass; a downstream `function.call @Sub::@constrain(...)`
// whose operand type would now mismatch the callee signature is
// erased (`@constrain` itself is unreachable at witness-gen time
// — `ConstrainFunctionErasePattern` deletes it during the main
// conversion).
//
// The inner struct must expose at least one `{llzk.pub}` member,
// each typed as scalar `!felt` (`@XOR_0`, `@Bits2Num_1`,
// `@Switcher_206.@outL/@outR`) or `!array<M x !felt>` (AES
// `@Num2Bits_5.@out`, `@BitElementMulAny_6.@dblOut/@addOut`). When
// K=1, the parent's flat witness slot stays `<D x innerTy>` (sister
// K=1 chips are byte-equal post-flip). When K>1, the K pub felt
// members must share a single inner type (uniform-shape constraint
// — both Switcher's all-scalar and BitElementMulAny's all-`<2 x
// !felt>` shapes satisfy this); the parent member widens to
// `<D, K x innerTy>` with one slot per pub field in declaration
// order, matching circom's `.wtns` flat felt layout.
static llvm::DenseMap<Value, PacfDrainPlan>
planDrains(Block &funcBlock, Operation *arrNew,
           ArrayRef<PacfDrainReader> drainReaders,
           const llvm::StringMap<Value> &perFieldArrays) {
  llvm::DenseMap<Value, PacfDrainPlan> drainPlans;
  for (auto &dr : drainReaders) {
    if (drainPlans.count(dr.destArr))
      continue;
    auto destArrTy = dyn_cast<llzk::array::ArrayType>(dr.destArr.getType());
    if (!destArrTy)
      continue;
    auto innerStruct =
        dyn_cast<llzk::component::StructType>(destArrTy.getElementType());
    if (!innerStruct)
      continue;
    SmallVector<PacfPubFelt, 2> pubFelts;
    SmallVector<PacfWritemMember, 2> recursiveMembers;
    bool useRecursive = false;
    if (!findInnerFeltMembers(funcBlock, innerStruct, pubFelts)) {
      // K=0 (or mixed-shape K>1 rejected by uniform-shape gate)
      // fall-through: try recursive writem-target flatten. Promote
      // collected members to `{llzk.pub}` so the writer-side
      // `struct.readm %callResult[@<field>]` is legal under LLZK's
      // MemberReadOp visibility verifier.
      if (!collectAndPromoteRecursiveWritemMembers(
              funcBlock, innerStruct, recursiveMembers, /*promote=*/true))
        continue;
      useRecursive = true;
    }

    // K=1 keeps the existing combined shape (`<D x innerTy>`) so AES
    // sister chips stay byte-identical. K>1 prepends a K dim so each
    // pub field gets its own slot at outer index `j`; declaration
    // order is the canonical pub-field ordering and matches circom's
    // .wtns flat felt layout per chip iteration. `combineDispatchAnd
    // InnerFeltDims` already does the scalar-vs-array branch with the
    // exact "prepend leading dim(s)" semantics — reuse it with `{K}`
    // as the leading-dim list.
    //
    // K=0 (recursive) widens the parent slot to `<D × totalFlat × !felt>`
    // — `totalFlat` is the sum of writem-targeted member flats per
    // instance (e.g. MMP_275 → 30 hasher + 60 switcher = 90), with
    // per-instance row-major layout matching circom's `.wtns` for the
    // chained sub-component output (writem-targeted members in
    // declaration order, each member's elements walked in declared
    // dim order).
    int64_t totalFlat = 0;
    Type combinedInnerTy;
    if (useRecursive) {
      for (const PacfWritemMember &wm : recursiveMembers)
        totalFlat += wm.flatSize;
      // Inner felt-type for the recursive flatten path. All members
      // contribute to the *same* `!felt` element type today (we reject
      // anything that's not felt-or-felt-array via the writem walk's
      // pod / struct gates). Pick the leaf felt type from the first
      // member — for `<N x !felt>` it's the array's element type, for
      // scalar `!felt` it's the member type itself.
      Type leafFelt = recursiveMembers.front().ty;
      while (auto shaped = dyn_cast<ShapedType>(leafFelt))
        leafFelt = shaped.getElementType();
      combinedInnerTy = llzk::array::ArrayType::get(leafFelt, {totalFlat});
    } else {
      combinedInnerTy =
          pubFelts.size() == 1
              ? pubFelts[0].ty
              : Type(combineDispatchAndInnerFeltDims(
                    pubFelts[0].ty, {static_cast<int64_t>(pubFelts.size())}));
    }

    // Allocate the parallel felt array right after the dispatch pod
    // array `%arr` so it dominates every writer site. The drain
    // destination is typically declared near the bottom of the
    // @compute body (after every writer); felt writes will go into
    // the parallel array at writer sites and the original destArr
    // will be replaced by an `unrealized_conversion_cast` of the
    // felt array on the consumer side.
    //
    // Reuse `perFieldArrays[innerField]` when Loop A already allocated
    // a felt array of the same element type and shape for the same
    // field name. Without this guard, Loop A and Loop B both emit one
    // `array.write` per writer site at identical outer indices —
    // ciphertext = same XOR result splattered into both per-field
    // carriers (`@a` reader-driven AND `@b` drain-driven), clobbering
    // a sibling `pod.write [@b] = round-key-bit` from
    // `rewritePodArrayUsesInBlock`. AES `aes_256_encrypt` is the
    // canonical case (60/128 ciphertext bytes wrong on round-0 XOR);
    // the post-conversion `dynamic_update_slice` pair at offsets
    // matching `iterArg_192`/`iterArg_193` collapses on identical RHS
    // and the legitimate round-key-bit emission is dropped. K>1
    // never reuses — its `<D, K x ...>` combined shape never matches
    // a Loop A per-field `<D x ...>` allocation.
    auto destDims = getArrayDimensions(destArrTy);
    auto feltArrTy = combineDispatchAndInnerFeltDims(combinedInnerTy, destDims);
    Value destFelt;
    bool reused = false;
    // K=0 path never reuses a `perFieldArrays` allocation — its
    // `<D × totalFlat × !felt>` shape never matches a Loop A
    // per-field `<D × ...>` carrier (no pub felts means no Loop A
    // reader-side allocation for the inner struct's members).
    if (!useRecursive && pubFelts.size() == 1) {
      auto reuseIt = perFieldArrays.find(pubFelts[0].field);
      if (reuseIt != perFieldArrays.end() &&
          reuseIt->second.getType() == feltArrTy) {
        destFelt = reuseIt->second;
        reused = true;
      }
    }
    if (!destFelt) {
      OpBuilder db(arrNew);
      db.setInsertionPointAfter(arrNew);
      destFelt =
          db.create<llzk::array::CreateArrayOp>(arrNew->getLoc(), feltArrTy);
    }
    // K>1: emit the K shared index constants once near `arrNew`
    // (function-block scope, dominates every writer site). Without
    // this hoist, the writer loop would emit one fresh constant per
    // (writer, j) = W·K constants per chip — D=30 K=2 → 60 surplus
    // `arith.constant <0..1>` ops that CSE folds downstream but
    // bloat the IR every intermediate pass walks.
    SmallVector<Value, 2> kIndices;
    if (pubFelts.size() > 1) {
      OpBuilder ib(arrNew);
      ib.setInsertionPointAfter(arrNew);
      for (size_t j = 0; j < pubFelts.size(); ++j)
        kIndices.push_back(emitConstIndex(ib, arrNew->getLoc(), j));
    }
    drainPlans[dr.destArr] = {destFelt,
                              combinedInnerTy,
                              SmallVector<PacfPubFelt, 2>(pubFelts),
                              std::move(kIndices),
                              std::move(recursiveMembers),
                              totalFlat,
                              dr.destArr.getType(),
                              reused};
  }
  return drainPlans;
}

// Single-instance pod-array dispatch fast path: when every writer
// targets the same destArr cell (outerIndex tuples compare equal as
// `felt.const` values, looked through `cast.toindex`) AND every cross-
// block reader/drainReader follows the last writer in funcBlock
// source order, drop all but the last writer. Earlier writers'
// results are last-write-wins-clobbered on the destArr cell, so
// emitting their `function.call`s + `array.insert`s only inflates
// GPU work (each redundant call lowers to its own kernel that
// produces a value the next writer immediately overwrites).
//
// Empirical: `aes_256_ctr` has 2 writers sharing `cast.toindex
// %felt_const_0` over a `!array<1 x !pod>` dispatch pod, so
// pre-fold it emits 2 hoisted calls × {128, 1920} body iters =
// 2048 calls at runtime; post-fold, 1 call × 128 iters = 128.
// The `materializeScalarPodCompField` sister already handles the
// analogous `pod.new` shape; this is the array-of-pods variant.
static void foldSingleInstanceWriters(SmallVectorImpl<PacfWriter> &writers,
                                      ArrayRef<PacfReader> readers,
                                      ArrayRef<PacfDrainReader> drainReaders) {
  if (writers.size() > 1) {
    bool allSameOuterIndex = true;
    auto firstVals = outerIndexConstValues(writers.front().outerIndices);
    if (!firstVals) {
      allSameOuterIndex = false;
    } else {
      for (size_t i = 1; i < writers.size() && allSameOuterIndex; ++i) {
        auto v = outerIndexConstValues(writers[i].outerIndices);
        if (!v || v->size() != firstVals->size()) {
          allSameOuterIndex = false;
          break;
        }
        for (size_t j = 0; j < v->size(); ++j)
          if ((*v)[j] != (*firstVals)[j]) {
            allSameOuterIndex = false;
            break;
          }
      }
    }

    if (allSameOuterIndex) {
      // Pick the source-order-latest writer.
      size_t lastIdx = 0;
      for (size_t i = 1; i < writers.size(); ++i) {
        Operation *cur = writers[i].callResult.getDefiningOp();
        Operation *best = writers[lastIdx].callResult.getDefiningOp();
        if (cur && best && strictlyAfter(cur, best))
          lastIdx = i;
      }

      Operation *lastCall = writers[lastIdx].callResult.getDefiningOp();
      bool consumersAllAfter = lastCall != nullptr;
      for (auto &r : readers)
        if (!strictlyAfter(r.structReadm, lastCall)) {
          consumersAllAfter = false;
          break;
        }
      // NOLINTNEXTLINE(readability/braces)
      if (consumersAllAfter)
        for (auto &dr : drainReaders)
          if (!strictlyAfter(dr.writem, lastCall)) {
            consumersAllAfter = false;
            break;
          }

      if (consumersAllAfter) {
        // Erase the dropped writers' `function.call` + `pod.write
        // [@comp] = %callResult` so `extractCallsFromScfIf` (Phase 1)
        // doesn't re-emit them as orphan calls before the enclosing
        // scf.if. Pod.write [@comp] is the ONLY consumer of the
        // call's struct result (verified at writer-collection time:
        // we walked back from the array.write site to find this
        // pair). The post-while lift to ONE total call (vs. one per
        // surviving writerWhile body iter) happens later in
        // `liftConstIndexPodArrayCallPostWhile`, after all pod.read
        // operands have been resolved to their staged array values.
        for (size_t i = 0; i < writers.size(); ++i) {
          if (i == lastIdx)
            continue;
          Operation *callOp = writers[i].callResult.getDefiningOp();
          if (writers[i].podWrite)
            writers[i].podWrite->erase();
          if (callOp && isAllResultsUnused(*callOp))
            callOp->erase();
        }

        PacfWriter keep = std::move(writers[lastIdx]);
        writers.clear();
        writers.push_back(std::move(keep));
      }
    }
  }
}

// For each writer: hoist the dispatch `function.call` (and its
// transitive operand defs) out of `w.insertAfter`'s scf.if so its
// result is visible at body level, then for each per-field array
// emit `%felt = struct.readm %callResult[@F]; array.write
// %arr_F[%outerIndex] = %felt` after `w.insertAfter`.
//
// Per-writer hoisting is required because Phase 1's tracker is
// overwritten on each scf.if extraction (count-countdown helpers
// emit one fire per pending input — keccak XorArray fires twice,
// arity-3 gates would fire three times). Forwarding via Phase 2
// would resolve all hoisted reads to the *last* call, breaking
// dominance for earlier writes. The single-instance fold above
// sidesteps this concern by collapsing same-cell writers up front
// when no consumer is sandwiched between them.
static void
emitWriters(ArrayRef<PacfWriter> writers,
            const llvm::StringMap<Value> &perFieldArrays,
            const llvm::StringMap<Type> &fieldFeltTypes,
            const llvm::DenseMap<Value, PacfDrainPlan> &drainPlans) {
  for (auto &w : writers) {
    Operation *callOp = w.callResult.getDefiningOp();
    if (!callOp)
      continue;

    // Collect transitive operand defs that live under `w.insertAfter`
    // in operands-before-uses order, then move each before
    // `w.insertAfter`.
    llvm::DenseSet<Operation *> seen;
    SmallVector<Operation *> ordered;
    for (Value operand : callOp->getOperands())
      collectHoistDefs(w.insertAfter, operand, seen, ordered);
    ordered.push_back(callOp);

    for (Operation *op : ordered)
      op->moveBefore(w.insertAfter);

    OpBuilder b(w.insertAfter);
    b.setInsertionPointAfter(w.insertAfter);
    for (auto &kv : perFieldArrays) {
      StringRef fieldName = kv.first();
      Value arrField = kv.second;
      Type feltTy = fieldFeltTypes.lookup(fieldName);
      // MemberReadOp carries `AttrSizedOperandSegments`; use the
      // typed builder so the segment-sizes attribute is populated.
      Value feltVal = b.create<llzk::component::MemberReadOp>(
          w.insertAfter->getLoc(), feltTy, w.callResult,
          b.getStringAttr(fieldName));

      emitCarrierWrite(b, w.insertAfter->getLoc(), arrField, w.outerIndices,
                       feltVal);
    }
    // Drain side: extract each pub felt member from %callResult and
    // write it into the parallel destFelt array. K=1 emits one write
    // per writer at outerIndices (existing single-pub shape). K>1
    // emits K writes per writer with a constant K-dim index `j`
    // appended to outerIndices, one per pub field in declaration
    // order. Either way, the destArr's struct-element layout is
    // discarded — the lowered tensor for the parent's `@F'` member
    // becomes `tensor<D[*K[*M]] x F>`, exactly the felt-flat witness
    // contribution the canonical circom `.wtns` expects per cell.
    for (auto &kv : drainPlans) {
      const PacfDrainPlan &plan = kv.second;
      // K=0 recursive flatten path: emit one struct.readm per
      // writem-targeted member, then walk each member's natural
      // dim shape row-major and emit a per-element `array.write` into
      // `destFelt[outerIndices..., %off]`. The flat offset
      // `offsetWithinInstance + linear_idx` is materialized as
      // `arith.constant : index`. Each writer site contributes
      // `sum(memberFlats)` writes (e.g. 90 for MMP_275: 30 hasher +
      // 60 switcher). Sister K=1/K>1 chips never reach this branch
      // because they take the pub-felt path above.
      if (!plan.recursiveMembers.empty()) {
        for (const PacfWritemMember &wm : plan.recursiveMembers) {
          Value memVal = b.create<llzk::component::MemberReadOp>(
              w.insertAfter->getLoc(), wm.ty, w.callResult,
              b.getStringAttr(wm.field));
          SmallVector<int64_t> memDims;
          if (auto arrTy = dyn_cast<llzk::array::ArrayType>(wm.ty))
            memDims = getArrayDimensions(arrTy);
          // Scalar `!felt` member: single write at offsetWithinInstance.
          // Multi-dim array member: row-major walk of all elements.
          int64_t numElements = wm.flatSize;
          for (int64_t linear = 0; linear < numElements; ++linear) {
            // Decompose `linear` into multi-dim coords via row-major
            // (last dim varies fastest). For scalar members memDims is
            // empty and the loop reduces to a no-op (no array.read /
            // memVal IS the felt directly).
            SmallVector<Value> coordVals;
            if (!memDims.empty()) {
              int64_t rem = linear;
              SmallVector<int64_t> coords(memDims.size(), 0);
              for (int dim = static_cast<int>(memDims.size()) - 1; dim >= 0;
                   --dim) {
                coords[dim] = rem % memDims[dim];
                rem /= memDims[dim];
              }
              for (int64_t c : coords)
                coordVals.push_back(
                    emitConstIndex(b, w.insertAfter->getLoc(), c));
            }
            Value scalar;
            if (memDims.empty()) {
              scalar = memVal;
            } else {
              // Scalar leaf type: peel any nested ShapedType wrappers
              // off `wm.ty` to land on the felt type itself.
              Type leafTy = wm.ty;
              while (auto shaped = dyn_cast<ShapedType>(leafTy))
                leafTy = shaped.getElementType();
              OperationState readState(w.insertAfter->getLoc(), "array.read");
              SmallVector<Value> readOperands;
              readOperands.push_back(memVal);
              readOperands.append(coordVals.begin(), coordVals.end());
              readState.addOperands(readOperands);
              readState.addTypes({leafTy});
              scalar = b.create(readState)->getResult(0);
            }
            // Flat inner offset constant for destFelt's last dim.
            Value offVal = emitConstIndex(b, w.insertAfter->getLoc(),
                                          wm.offsetWithinInstance + linear);
            OperationState writeState(w.insertAfter->getLoc(), "array.write");
            SmallVector<Value> writeOperands;
            writeOperands.push_back(plan.destFelt);
            writeOperands.append(w.outerIndices.begin(), w.outerIndices.end());
            writeOperands.push_back(offVal);
            writeOperands.push_back(scalar);
            writeState.addOperands(writeOperands);
            b.create(writeState);
          }
        }
        continue;
      }
      // When the drain target shares storage with Loop A's
      // `perFieldArrays[plan.pubFelts[0].field]`, Loop A above
      // already emitted `array.write %perFieldArrays[innerField][i]
      // = struct.readm %callResult[@innerField]` for this writer.
      // Re-emitting the same write here would produce a duplicate
      // pair at identical indices — the post-conversion
      // `dynamic_update_slice` collapse on identical RHS clobbers
      // sibling per-field writes elsewhere (the round-key-bit
      // emission for AES `aes_256_encrypt`).
      if (plan.reusedFromPerField)
        continue;
      for (size_t j = 0; j < plan.pubFelts.size(); ++j) {
        Value feltVal = b.create<llzk::component::MemberReadOp>(
            w.insertAfter->getLoc(), plan.pubFelts[j].ty, w.callResult,
            b.getStringAttr(plan.pubFelts[j].field));
        // K>1 appends the shared K-dim index emitted at destFelt-allocation
        // time, keeping K=1's IR shape byte-identical to the AES single-pub
        // path.
        SmallVector<Value> writeIndices(w.outerIndices.begin(),
                                        w.outerIndices.end());
        if (!plan.kIndices.empty())
          writeIndices.push_back(plan.kIndices[j]);
        emitCarrierWrite(b, w.insertAfter->getLoc(), plan.destFelt,
                         writeIndices, feltVal);
      }
    }
  }
}

static void rewriteReaders(ArrayRef<PacfReader> readers,
                           const llvm::StringMap<Value> &perFieldArrays,
                           const llvm::StringMap<Type> &fieldFeltTypes) {
  // The intervening `pod.read %dp2[@comp]` becomes use-empty; Phase 5
  // / `rewriteArrayPodCountCompInReads` nondets it on the next driver
  // iteration, then Phase 4 DCEs the dead substitute.
  SmallVector<Operation *> toErase;
  for (auto &r : readers) {
    auto fieldIt = perFieldArrays.find(r.field);
    if (fieldIt == perFieldArrays.end())
      continue;
    Value arrField = fieldIt->second;
    Type feltTy = fieldFeltTypes.lookup(r.field);
    OpBuilder b(r.structReadm);
    Value newReadVal = emitCarrierRead(b, r.structReadm->getLoc(), arrField,
                                       arrayAccessIndices(r.arrayRead), feltTy);
    r.structReadm->getResult(0).replaceAllUsesWith(newReadVal);
    toErase.push_back(r.structReadm);
  }
  for (Operation *op : toErase)
    op->erase();
}

// For each unique drain destination: redirect the parent
// `struct.writem` operand from `%destArr` (struct array, populated
// with `Phase 5`-nondet'd zeros) to the parallel felt array. This
// requires also flipping the parent `struct.member @F'`'s declared
// TypeAttr from `array<D x !struct>` to `array<D x !felt>` so the
// writem's operand type matches the member type. The
// `@constrain` chain that reads `@F'` is repaired to match —
// struct.readm result type is retyped, the inner `array.read` is
// retyped, and the inner `struct.readm @out` is erased
// (replaced by the array.read result). Function calls in
// `@constrain` whose operand types now mismatch (e.g.
// `function.call @Sub::@constrain(%cell, ...)`) are erased
// wholesale — `@constrain` is unreachable from witness generation
// (`ConstrainFunctionErasePattern` deletes it during the main
// conversion) so this is safe.
static void
redirectDrains(ArrayRef<PacfDrainReader> drainReaders,
               const llvm::DenseMap<Value, PacfDrainPlan> &drainPlans) {
  llvm::DenseSet<Value> processedDest;
  for (auto &dr : drainReaders) {
    auto planIt = drainPlans.find(dr.destArr);
    if (planIt == drainPlans.end())
      continue;

    // Erase the drain reader's `array.write %destArr[idx] = %comp`
    // (Phase 5 already nondets %comp to a zero struct, which would
    // otherwise overwrite the felt-array slot if the writem still
    // pointed at %destArr).
    dr.arrayWriteDst->erase();

    if (!processedDest.insert(dr.destArr).second)
      continue;

    const PacfDrainPlan &plan = planIt->second;
    bool isRecursive = !plan.recursiveMembers.empty();
    auto destDims =
        getArrayDimensions(cast<llzk::array::ArrayType>(plan.structArrTy));
    // Sister chips with scalar inner @out (K=1) are byte-equal post-flip;
    // chips with `!array<M x !felt>` inner members inflate the parent's
    // flat witness slot from D to D*M; multi-pub (K>1) inflates by an
    // additional factor of K (matches `getMemberFlatSize` over the
    // lowered `tensor<D[*K[*M]]>` shape). K=0 recursive flatten
    // produces `<D × totalFlat × !felt>` directly from `combinedInnerTy
    // = <totalFlat × !felt>`.
    auto newMemberArrTy =
        combineDispatchAndInnerFeltDims(plan.combinedInnerTy, destDims);

    // Redirect the parent struct.writem.
    dr.writem->setOperand(1, plan.destFelt);

    // Flip the parent struct.member @F' TypeAttr.
    Operation *parentStructDef = dr.writem->getParentOp();
    while (parentStructDef &&
           !isa<llzk::component::StructDefOp>(parentStructDef))
      parentStructDef = parentStructDef->getParentOp();
    if (!parentStructDef)
      continue;
    Operation *memberOp = nullptr;
    for (Region &region : parentStructDef->getRegions())
      for (Block &block : region)
        for (Operation &nested : block) {
          if (!isa<llzk::component::MemberDefOp>(nested))
            continue;
          auto sym = nested.getAttrOfType<StringAttr>("sym_name");
          if (sym && sym.getValue() == dr.parentField) {
            memberOp = &nested;
            break;
          }
        }
    if (!memberOp)
      continue;
    memberOp->setAttr("type", TypeAttr::get(newMemberArrTy));

    // Repair @constrain: retype struct.readm @F' result and the
    // immediate `array.read %readm[..]` result; erase the inner
    // `struct.readm @<f_j>` (K=1: replace with array.read result;
    // K>1: replace with `array.read|extract %slice[%c_j]` per pub
    // field's index in declaration order; K=0 recursive: replace
    // each inner struct.readm with an `llzk.nondet` of the original
    // member type — its only consumer is the now-dead sibling
    // `function.call ::@constrain`, so the placeholder DCEs in
    // Phase 4); erase any function.call consumer whose operand
    // type now mismatches.
    bool isMultiPub = plan.pubFelts.size() > 1;
    bool innerIsArray =
        !isRecursive && isa<llzk::array::ArrayType>(plan.pubFelts[0].ty);
    llvm::DenseMap<StringRef, size_t> fieldIdx;
    for (size_t j = 0; j < plan.pubFelts.size(); ++j)
      fieldIdx[plan.pubFelts[j].field] = j;
    // K=0: map inner member name to original type so each
    // `struct.readm @<member>` in @constrain can be replaced by a
    // shape-matched llzk.nondet placeholder (the slice's true value is
    // not available as a felt-typed projection because heterogeneous
    // member shapes can't be re-extracted from a flat `<totalFlat ×
    // !felt>` row).
    llvm::DenseMap<StringRef, Type> recursiveMemberTys;
    if (isRecursive) {
      for (const PacfWritemMember &wm : plan.recursiveMembers)
        recursiveMemberTys[wm.field] = wm.ty;
    }
    parentStructDef->walk([&](Operation *funcOp) {
      if (!isa<llzk::function::FuncDefOp>(funcOp))
        return;
      auto sym = funcOp->getAttrOfType<StringAttr>("sym_name");
      if (!sym || sym.getValue() != "constrain")
        return;
      SmallVector<Operation *> readms;
      funcOp->walk([&](Operation *rm) {
        if (!isa<llzk::component::MemberReadOp>(rm) || rm->getNumResults() == 0)
          return;
        auto memberAttr = rm->getAttrOfType<FlatSymbolRefAttr>("member_name");
        if (!memberAttr || memberAttr.getValue() != dr.parentField)
          return;
        readms.push_back(rm);
      });
      // K=1 scalar inner: in-place retype keeps the array.read
      // intact (one fewer op + matches the byte-stable single-pub
      // shape). All other shapes (K=1 array, K>1 anything, K=0
      // recursive) shave one dim off the parent and need an
      // `array.extract` to produce the per-cell slice.
      bool sliceViaExtract = isRecursive || isMultiPub || innerIsArray;
      // Defer erases until after the whole for-rm loop: an inner-arUser
      // struct.readm handled below can match a *later* @F'-readm in
      // `readms` (in deeply nested recursive members the inner walk picks
      // up readms across nesting levels, and the inner level's @F'-readm
      // can be a downstream user of the outer level's extracted slice).
      // Erasing per-rm would free that later `rm` before its own iteration
      // accesses it. Accumulating toErase across rms keeps every `rm`
      // alive through the for-rm loop; by the time the deeply-nested rm's
      // turn comes its uses have already been replaced with an
      // `llzk.nondet` placeholder, so its for-user body is a no-op and
      // its erase falls out of the post-loop cleanup. SetVector dedupes
      // because the cross-rm reach now lets the same arUser surface via
      // two different rms' slices — a duplicate `push_back` followed by
      // a per-element `op->erase()` would double-free.
      llvm::SetVector<Operation *> toErase;
      for (Operation *rm : readms) {
        rm->getResult(0).setType(newMemberArrTy);
        // @compute's `kIndices` live in a different function — emit
        // a parallel set lazily inside @constrain so K>1 readers in
        // this function have an SSA value to index into the K-dim
        // slice with.
        SmallVector<Value, 2> constrainKIndices;
        // Snapshot users before iterating: the body creates a new
        // `array.extract` whose operand list mirrors the current `user`'s
        // (so `rm->getResult(0)` picks up a fresh use), which mutates the
        // use-list the range-for would otherwise be walking.
        auto rmUsers = llvm::to_vector(rm->getResult(0).getUsers());
        for (Operation *user : rmUsers) {
          if (!isa<llzk::array::ReadArrayOp>(user) ||
              user->getNumResults() == 0)
            continue;
          Value newReadResult;
          if (sliceViaExtract) {
            OpBuilder rb(user);
            OperationState extractState(user->getLoc(), "array.extract");
            extractState.addOperands(user->getOperands());
            extractState.addTypes({plan.combinedInnerTy});
            Operation *extractOp = rb.create(extractState);
            newReadResult = extractOp->getResult(0);
            user->getResult(0).replaceAllUsesWith(newReadResult);
            toErase.insert(user);
          } else {
            user->getResult(0).setType(plan.combinedInnerTy);
            newReadResult = user->getResult(0);
          }
          for (OpOperand &arUse :
               llvm::make_early_inc_range(newReadResult.getUses())) {
            Operation *arUser = arUse.getOwner();
            // function.call to sibling `Sub::@constrain` now has a
            // felt[-array] operand where it expected `!struct<@Sub>`
            // — erase. @constrain is dead from a witness perspective
            // (`ConstrainFunctionErasePattern`) so unreferenced
            // operands DCE in Phase 4.
            if (isa<llzk::function::CallOp>(arUser)) {
              toErase.insert(arUser);
              continue;
            }
            if (!isa<llzk::component::MemberReadOp>(arUser) ||
                arUser->getNumResults() == 0)
              continue;
            auto innerMember =
                arUser->getAttrOfType<FlatSymbolRefAttr>("member_name");
            if (!innerMember)
              continue;
            // K=0 recursive: heterogeneous member shapes can't be
            // re-projected from a flat felt slice. Replace the inner
            // `struct.readm @<member>` with a typed `llzk.nondet`
            // placeholder — its only downstream consumer is the
            // sibling `function.call ::@constrain` (queued for erase
            // above), so the placeholder DCEs cleanly in Phase 4.
            if (isRecursive) {
              auto rmIt = recursiveMemberTys.find(innerMember.getValue());
              if (rmIt == recursiveMemberTys.end())
                continue;
              OpBuilder ib(arUser);
              Value placeholder =
                  createNondet(ib, arUser->getLoc(), rmIt->second);
              arUser->getResult(0).replaceAllUsesWith(placeholder);
              toErase.insert(arUser);
              continue;
            }
            auto fIt = fieldIdx.find(innerMember.getValue());
            if (fIt == fieldIdx.end())
              continue;
            if (!isMultiPub) {
              // K=1: the slice IS the field value.
              arUser->getResult(0).replaceAllUsesWith(newReadResult);
              toErase.insert(arUser);
              continue;
            }
            // K>1: index into the K-dim slice with the field's
            // declaration-order position. Array pub field produces
            // a sub-slice via `array.extract`; scalar pub field
            // produces a scalar via `array.read`.
            if (constrainKIndices.empty()) {
              OpBuilder ib(rm);
              ib.setInsertionPointAfter(rm);
              for (size_t k = 0; k < plan.pubFelts.size(); ++k)
                constrainKIndices.push_back(
                    emitConstIndex(ib, rm->getLoc(), k));
            }
            OpBuilder ib(arUser);
            StringRef readOpName =
                innerIsArray ? "array.extract" : "array.read";
            OperationState readState(arUser->getLoc(), readOpName);
            readState.addOperands(
                {newReadResult, constrainKIndices[fIt->second]});
            readState.addTypes({plan.pubFelts[fIt->second].ty});
            Operation *readOp = ib.create(readState);
            arUser->getResult(0).replaceAllUsesWith(readOp->getResult(0));
            toErase.insert(arUser);
          }
        }
      }
      for (Operation *op : toErase)
        op->erase();
    });
  }
}

bool materializePodArrayCompField(Block &funcBlock) {
  bool changed = false;

  SmallVector<PacfCandidate> candidates = collectCandidates(funcBlock);

  for (auto &cand : candidates) {
    Value arr = cand.arrNew->getResult(0);
    auto arrTy = cast<llzk::array::ArrayType>(arr.getType());

    SmallVector<PacfWriter> writers;
    SmallVector<PacfReader> readers;
    llvm::StringMap<Type> fieldFeltTypes;
    SmallVector<PacfDrainReader> drainReaders;

    classifyArrUses(arr, writers, readers, fieldFeltTypes, drainReaders);

    if (writers.empty() || (readers.empty() && drainReaders.empty()))
      continue;

    llvm::StringMap<Value> perFieldArrays =
        allocPerFieldArrays(cand.arrNew, arrTy, readers, fieldFeltTypes);

    llvm::DenseMap<Value, PacfDrainPlan> drainPlans =
        planDrains(funcBlock, cand.arrNew, drainReaders, perFieldArrays);

    if (perFieldArrays.empty() && drainPlans.empty())
      continue;

    foldSingleInstanceWriters(writers, readers, drainReaders);

    emitWriters(writers, perFieldArrays, fieldFeltTypes, drainPlans);

    rewriteReaders(readers, perFieldArrays, fieldFeltTypes);

    redirectDrains(drainReaders, drainPlans);

    if (!perFieldArrays.empty() || !drainPlans.empty())
      changed = true;
  }

  return changed;
}

/// Close the input-pod data-flow gap that survives `flattenPodArrayWhileCarry`.
///
/// Circom emits a per-instance input-pod array `array.new : <D x !pod<[@in:
/// !felt]>>` to stage the operand for a deferred sub-component dispatch. The
/// writer body inside an scf.while iteration writes `%src` into the cell's
/// `@in` field; the firing scf.if (count countdown == 0) reads it back via
/// `pod.read %cell[@in]` and feeds the dispatched call. SSA-wise, `%src` and
/// the firing-site read share the same pod cell.
///
/// `flattenPodArrayWhileCarry` is supposed to rebuild that data flow at the
/// felt-array level when the carry crosses scf.while levels, but at 3+ levels
/// of nesting (AES `@AES256Encrypt_6::@compute` is the canonical case) the
/// per-level rewire leaves the firing-site read disconnected. Phase 5
/// (`rewriteArrayPodCountCompInReads`) then nondets it, the dispatched call
/// gets fed const-zero, and the parent witness sees the sub-component's
/// `Bits(0) = zeros` output.
///
/// This pass runs BEFORE `flattenPodArrayWhileCarry` while writer and reader
/// are still SSA-paired through `%cell`: it replaces every sibling `pod.read
/// %cell[@in]` result with `%src` directly and erases the pod.read. The
/// pod.write becomes use-empty and is DCE'd by Phase 4. Dominance holds
/// because `%src` and `%cell` are defined at the writer body block level
/// and the firing-site read lives in a child region of that same block.
///
/// Convergence with `eliminatePodDispatch`: this pass only does
/// `replaceAllUsesWith` + `erase` on existing pod.read ops, so Phase 5
/// finds nothing extra and Phase 1 routes the call's now-felt operand
/// through its `directArgs` branch. Idempotent: subsequent invocations
/// find no `pod.read [@in]` left.

bool materializePodArrayInputPodField(Block &funcBlock) {
  SmallVector<Operation *> toErase;

  funcBlock.walk([&](Operation *op) {
    if (!isa<llzk::pod::WritePodOp>(op) || op->getNumOperands() < 2)
      return;
    auto rn = op->getAttrOfType<FlatSymbolRefAttr>("record_name");
    if (!rn || rn.getValue() != "in")
      return;

    Value cell = op->getOperand(0);
    Value src = op->getOperand(1);

    // Array-element only. Scalar input pods (`pod.new : <[@in:...]>`) are
    // handled by `inlineInputPodCarries`; firing on a scalar risks
    // RAUW-ing a self-referential read-modify-write (`%v = pod.read
    // %p[@in]; pod.write %p[@in] = %v`) where erasing the pod.read leaves
    // dangling references in the function.call we just rewired.
    Operation *cellDef = cell.getDefiningOp();
    if (!cellDef || !isa<llzk::array::ReadArrayOp>(cellDef))
      return;
    auto podTy = dyn_cast<llzk::pod::PodType>(cell.getType());
    if (!podTy)
      return;
    auto recs = podTy.getRecords();
    if (recs.size() != 1 || recs[0].getName() != "in")
      return;

    // Read-modify-write guard: an array-typed `@in: !array<...>` cell uses
    // the same pattern at the inner array (`%v = pod.read %cell[@in];
    // array.write %v[i] = %x; pod.write %cell[@in] = %v`). Defer to
    // `eliminatePodDispatch`'s tracker (mirrors `extractCallsFromScfIf`
    // line 287-303).
    if (auto *srcDef = src.getDefiningOp()) {
      if (isa<llzk::pod::ReadPodOp>(srcDef) && srcDef->getOperand(0) == cell) {
        auto srcRn = srcDef->getAttrOfType<FlatSymbolRefAttr>("record_name");
        if (srcRn && srcRn.getValue() == "in")
          return;
      }
    }

    for (OpOperand &use : llvm::make_early_inc_range(cell.getUses())) {
      Operation *user = use.getOwner();
      if (user == op || use.getOperandNumber() != 0)
        continue;
      if (!isa<llzk::pod::ReadPodOp>(user) || user->getNumResults() == 0)
        continue;
      auto rn2 = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (!rn2 || rn2.getValue() != "in")
        continue;
      user->getResult(0).replaceAllUsesWith(src);
      toErase.push_back(user);
    }
  });

  for (Operation *op : toErase)
    op->erase();
  return !toErase.empty();
}

/// Materialize a tail `function.call` after the writer-while for cross-block
/// readbacks of a function-scope SCALAR `pod.new : <[..., @comp: !struct,
/// ...]>`.
///
/// Distinct from `materializePodArrayCompField` (which handles `array.new : <N
/// x !pod<...>>`): this variant fires when the dispatch storage is a single
/// pod whose `@comp` is written under `scf.if` in one `scf.while` body and
/// read back from a different block. The per-block trackers in
/// `eliminatePodDispatch` cannot bridge this — Phase 5 nondets the cross-
/// block reader and silently zeroes the dispatch result. Symptom on circomlib
/// `gates.circom` users with a *scalar* outer dispatch (keccak iota3/iota10's
/// `XorArray_2` invocation): @main never calls `<gate>_compute` and the
/// reader-loop's lane reads `dense<0>` for the dispatch's output range.
///
/// The fix exploits the count-countdown invariant of circomlib's dispatch
/// pattern: the substantively-firing call's operands match the writer-while's
/// post-loop iter-arg state. After `unpackPodWhileCarry` runs, the writer-
/// while's body-block args carry one felt-array per dispatch input, mutated
/// in-place by `array.write` per iteration. The LAST writer's call (in source
/// order) names these block args directly — its post-while projection is
/// exactly `whileOp.getResult(i)` for each block-arg operand.
///
/// Multi-while: Mux2/Mux3 templates emit two sequential scf.whiles (one per
/// dispatch input field — e.g. an @c-loop and an @s-loop) sharing the same
/// dispatch pod. The count countdown fires only on the SECOND loop's last
/// iteration, so the LAST writer is in the LAST while in funcBlock source
/// order; its enclosing while is the projection target. The per-while body
/// tracking otherwise generalizes unchanged.
///
/// We emit `%postCall = function.call @F(<post-while operand values>)` after
/// the writer-while terminator and replace each cross-block `pod.read
/// %pod[@comp]` with `%postCall`. Existing Phase 4/5 cleanup DCEs the
/// (orphaned) writer scf.if + pod.write + pod.new traffic.
///
/// Driver order: AFTER `unpackPodWhileCarry` (so writer-while iter-args are
/// felt/array, not pod — call operands are direct block args), BEFORE
/// `eliminatePodDispatch` (so writer scf.ifs + pod.writes are still intact).
bool materializeScalarPodCompField(Block &funcBlock) {
  bool changed = false;

  // 1. Function-scope candidates: `pod.new` or `llzk.nondet` producing a
  //    dispatch pod whose @comp field is a struct. Post-circom-llzk PR #390
  //    (2026-04-30) the dispatch pod's pod-creation site is emitted as
  //    `llzk.nondet : !pod.type<[@count, @comp, @params]>` instead of
  //    `pod.new {@count = const_N}`; the cross-block @comp readback shape is
  //    otherwise identical, so the same materialization is sound.
  //    (Array-of-pods is `materializePodArrayCompField`'s concern.)
  struct Candidate {
    Operation *podDef;
    llzk::component::StructType compTy;
  };
  SmallVector<Candidate> candidates;
  for (Operation &op : funcBlock) {
    if (!isa<llzk::pod::NewPodOp, llzk::NonDetOp>(op) ||
        op.getNumResults() == 0)
      continue;
    auto podTy = dyn_cast<llzk::pod::PodType>(op.getResult(0).getType());
    if (!podTy)
      continue;
    for (auto rec : podTy.getRecords()) {
      if (rec.getName() != "comp")
        continue;
      auto compTy = dyn_cast<llzk::component::StructType>(rec.getType());
      if (!compTy)
        break;
      candidates.push_back({&op, compTy});
      break;
    }
  }

  for (auto &cand : candidates) {
    Operation *podDef = cand.podDef;
    Value pod = podDef->getResult(0);

    struct Writer {
      Operation *callOp; // function.call result fed into pod.write[@comp].
      scf::WhileOp writerWhile; // enclosing scf.while, or null if writer is at
                                // function scope (e.g. inside an scf.if hanging
                                // directly off funcBlock — getValueByIndex's
                                // @main pattern).
    };
    SmallVector<Writer> writers;
    SmallVector<Operation *> readers;

    // 2. Walk pod uses, classify writers and readers.
    for (OpOperand &use : pod.getUses()) {
      Operation *user = use.getOwner();
      if (use.getOperandNumber() != 0)
        continue;
      auto field = user->getAttrOfType<FlatSymbolRefAttr>("record_name");
      if (!field || field.getValue() != "comp")
        continue;

      if (isa<llzk::pod::WritePodOp>(user) && user->getNumOperands() >= 2) {
        Operation *callOp = user->getOperand(1).getDefiningOp();
        if (!callOp || !isa<llzk::function::CallOp>(callOp))
          continue;
        scf::WhileOp w = user->getParentOfType<scf::WhileOp>();
        if (w) {
          Block &writerBody = w.getAfter().front();
          if (!writerBody.findAncestorOpInBlock(*user))
            continue;
        }
        writers.push_back({callOp, w});
      } else if (isa<llzk::pod::ReadPodOp>(user) && user->getNumResults() > 0) {
        readers.push_back(user);
      }
    }

    if (readers.empty())
      continue;

    // 2b. No-writer path: only readers exist (constant-table sub-component
    //     dispatch, e.g. keccak's RC_0). Synthesize a zero-arg call at
    //     function scope and rebind readers. Gate on the @compute method
    //     actually being zero-arg (looked up via the module SymbolTable)
    //     so we never invent operands for a sub-component that needs them.
    if (writers.empty()) {
      llzk::component::StructType compTy = cand.compTy;
      SymbolRefAttr structRef = compTy.getNameRef();
      MLIRContext *ctx = structRef.getContext();
      SmallVector<FlatSymbolRefAttr> nested(structRef.getNestedReferences());
      nested.push_back(FlatSymbolRefAttr::get(ctx, "compute"));
      auto callee = SymbolRefAttr::get(structRef.getRootReference(), nested);
      ModuleOp module = getTopLevelModule(funcBlock);
      auto fnDef = dyn_cast_or_null<llzk::function::FuncDefOp>(
          SymbolTable::lookupSymbolIn(module, callee));
      if (!fnDef || fnDef.getFunctionType().getNumInputs() != 0)
        continue;

      OpBuilder builder(podDef);
      builder.setInsertionPointAfter(podDef);
      Operation *postCall = builder.create<llzk::function::CallOp>(
          podDef->getLoc(), TypeRange{compTy}, callee, ValueRange{});
      Value postCallResult = postCall->getResult(0);
      for (Operation *r : readers) {
        r->getResult(0).replaceAllUsesWith(postCallResult);
        r->erase();
      }
      changed = true;
      continue;
    }

    // 3. Pick the LAST writer in funcBlock source order; its enclosing
    //    scf.while (or, for function-scope writers, its funcBlock-level
    //    anchor) is the projection target. Single-writer inputs collapse to
    //    the original behavior. Per-writer source-order is well-defined at
    //    funcBlock scope across all shapes (in-while + function-scope).
    auto funcAnchorOf = [&](Operation *op) -> Operation * {
      return funcBlock.findAncestorOpInBlock(*op);
    };
    Writer *lastWriter = &writers.front();
    Operation *lastFuncAnchor = funcAnchorOf(lastWriter->callOp);
    if (!lastFuncAnchor)
      continue;
    for (auto &w : writers) {
      Operation *fa = funcAnchorOf(w.callOp);
      if (!fa)
        continue;
      if (lastFuncAnchor->isBeforeInBlock(fa)) {
        lastWriter = &w;
        lastFuncAnchor = fa;
        continue;
      }
      // Same funcBlock anchor (e.g. two writers in one scf.while body, or
      // two writers in one funcBlock-level scf.if): isBeforeInBlock is
      // ill-defined at funcBlock level. Disambiguate at the smallest block
      // that contains both call ops by walking their parent chains until
      // they share a common block, then comparing direct-children-of-that-
      // block ancestors. Skip pairs whose call ops share no common block
      // (e.g. different scf.if regions) — source order between disjoint
      // execution regions has no meaningful answer; keep the earlier-seen
      // writer.
      if (fa != lastFuncAnchor)
        continue;
      for (Block *b = w.callOp->getBlock(); b;
           b = b->getParentOp() ? b->getParentOp()->getBlock() : nullptr) {
        Operation *aAnc = b->findAncestorOpInBlock(*lastWriter->callOp);
        Operation *bAnc = b->findAncestorOpInBlock(*w.callOp);
        if (aAnc && bAnc) {
          if (aAnc->isBeforeInBlock(bAnc))
            lastWriter = &w;
          break;
        }
      }
    }
    scf::WhileOp writerWhile = lastWriter->writerWhile;
    // The op `def` must dominate to be usable at funcBlock scope.
    Operation *dominanceScope =
        writerWhile ? writerWhile.getOperation() : lastFuncAnchor;

    // 4. Cross-block guard: drop readers nested under any writer-while body
    //    (those are forwarded same-block by `eliminatePodDispatch`'s tracker).
    //    Function-scope writers contribute no body — they're already at
    //    funcBlock level.
    llvm::DenseSet<Block *> writerBodies;
    for (auto &w : writers)
      if (w.writerWhile)
        writerBodies.insert(&w.writerWhile.getAfter().front());
    SmallVector<Operation *> crossBlockReaders;
    for (auto *r : readers) {
      bool inAnyWriterBody = false;
      for (Block *wb : writerBodies)
        if (wb->findAncestorOpInBlock(*r)) {
          inAnyWriterBody = true;
          break;
        }
      if (!inAnyWriterBody)
        crossBlockReaders.push_back(r);
    }
    if (crossBlockReaders.empty())
      continue;

    // 5. Dominance gate: every cross-block reader must follow lastFuncAnchor
    //    so RAUW to the tail call won't create use-before-def for readers
    //    sandwiched between writers.
    bool readersAfter = llvm::all_of(crossBlockReaders, [&](Operation *r) {
      Operation *rAnchor = funcAnchorOf(r);
      return rAnchor && lastFuncAnchor->isBeforeInBlock(rAnchor);
    });
    if (!readersAfter)
      continue;

    // 6. Resolve call operands at funcBlock scope. Body-block args of an
    //    in-while writer project to `whileOp.getResult(i)`; everything else
    //    must be defined outside `dominanceScope` (the writerWhile, or for
    //    function-scope writers, lastFuncAnchor) so it dominates the post-
    //    anchor insertion point. After `unpackPodWhileCarry` runs, function-
    //    scope `pod.read %carry[@field]` operands have already been replaced
    //    by unpacked while results — those satisfy the dominance check
    //    automatically.
    SmallVector<Value> resolvedOperands;
    bool resolveOK = true;
    for (Value operand : lastWriter->callOp->getOperands()) {
      if (auto ba = dyn_cast<BlockArgument>(operand)) {
        if (writerWhile && ba.getOwner() == &writerWhile.getAfter().front()) {
          resolvedOperands.push_back(writerWhile.getResult(ba.getArgNumber()));
          continue;
        }
        if (ba.getOwner() == &funcBlock) {
          resolvedOperands.push_back(operand);
          continue;
        }
        resolveOK = false;
        break;
      }
      Operation *def = operand.getDefiningOp();
      if (!def || dominanceScope->isAncestor(def)) {
        resolveOK = false;
        break;
      }
      resolvedOperands.push_back(operand);
    }
    if (!resolveOK)
      continue;

    // 7. Emit tail call after `dominanceScope` (the writerWhile, or the
    //    funcBlock-level scf.if/etc holding the last function-scope writer).
    Operation *insertAfter = dominanceScope;
    OpBuilder builder(insertAfter);
    builder.setInsertionPointAfter(insertAfter);
    auto callee = lastWriter->callOp->getAttrOfType<SymbolRefAttr>("callee");
    if (!callee)
      continue;
    Operation *postCall = builder.create<llzk::function::CallOp>(
        lastWriter->callOp->getLoc(), lastWriter->callOp->getResultTypes(),
        callee, resolvedOperands);
    Value postCallResult = postCall->getResult(0);

    // 8. Replace cross-block reader pod.read results with the tail-call
    //    result. The reader's chained `struct.readm [@F]` consumers now
    //    read from the materialized struct directly.
    for (Operation *r : crossBlockReaders) {
      r->getResult(0).replaceAllUsesWith(postCallResult);
      r->erase();
    }

    changed = true;
  }

  return changed;
}

} // namespace mlir::llzk_to_shlo
