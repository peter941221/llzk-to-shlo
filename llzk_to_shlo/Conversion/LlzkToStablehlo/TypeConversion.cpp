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

#include "llzk_to_shlo/Conversion/LlzkToStablehlo/TypeConversion.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::llzk_to_shlo {

// ===----------------------------------------------------------------------===
// Type checking helpers
// ===----------------------------------------------------------------------===

namespace {
StringRef getDialectNamespace(Type type) {
  if (auto opaqueType = dyn_cast<OpaqueType>(type))
    return opaqueType.getDialectNamespace();
  return type.getDialect().getNamespace();
}
} // namespace

bool LlzkToStablehloTypeConverter::isFeltType(Type type) {
  return getDialectNamespace(type) == "felt";
}

bool LlzkToStablehloTypeConverter::isArrayType(Type type) {
  return getDialectNamespace(type) == "array";
}

bool LlzkToStablehloTypeConverter::isStructType(Type type) {
  StringRef ns = getDialectNamespace(type);
  // The LLZK struct dialect is registered as "component" in the namespace
  return ns == "struct" || ns == "component";
}

// ===----------------------------------------------------------------------===
// Shared utility functions
// ===----------------------------------------------------------------------===

SmallVector<int64_t> getArrayDimensions(Type arrayType) {
  // LLZK ArrayType implements ShapedType::Trait and provides getShape().
  if (auto shaped = dyn_cast<ShapedType>(arrayType)) {
    auto shape = shaped.getShape();
    return SmallVector<int64_t>(shape.begin(), shape.end());
  }
  return {ShapedType::kDynamic};
}

namespace {

// Filter: skip pod-typed members (`*$inputs` dispatch scaffolding) at any
// `ShapedType` depth. Mirrors `isPodMember` in
// `WitnessLayoutAnchor.cpp` so the same exclusion applies on the
// recursive descent.
bool isPodMemberType(Type ty) {
  while (auto shaped = dyn_cast<ShapedType>(ty))
    ty = shaped.getElementType();
  return isa<llzk::pod::PodType>(ty);
}

llzk::component::StructDefOp findStructDefByExactSymbol(
    ModuleOp module, llzk::component::StructType structTy) {
  if (!module || !structTy)
    return {};
  if (auto def = dyn_cast_or_null<llzk::component::StructDefOp>(
          SymbolTable::lookupSymbolIn(module, structTy.getNameRef()))) {
    return def;
  }
  if (!structTy.getNameRef().getNestedReferences().empty())
    return {};
  return dyn_cast_or_null<llzk::component::StructDefOp>(
      SymbolTable::lookupSymbolIn(module,
                                  structTy.getNameRef().getLeafReference()));
}

int64_t getMemberFlatSizeImpl(Type memberType, ModuleOp module,
                              DenseSet<StringRef> &visited);

// Recursive flat-size for a struct, summing each writem-targeted,
// non-pod member's flat size. The writem-targeted filter matches the
// offset-map invariant in `registerStructFieldOffsets`
// (LlzkToStablehlo.cpp): members the lowering never writes to occupy
// zero cells, so they must not advance the running offset here either.
// Pub-ness is irrelevant for inner-struct sizing — pubs of an inner
// struct are persisted just like any other writem-targeted member.
int64_t getStructFlatSize(llzk::component::StructDefOp def, ModuleOp module,
                          DenseSet<StringRef> &visited) {
  // Cycle guard. LLZK struct definitions are tree-shaped today, but a
  // defensive check costs nothing and produces a finite answer if a
  // future LLZK frontend ever emits a self-referential struct.
  StringRef leaf = def.getSymName();
  if (!visited.insert(leaf).second)
    return 0;
  auto writem = collectWritemTargets(def);
  int64_t total = 0;
  for (auto m : def.getMemberDefs()) {
    Type ty = m.getType();
    if (isPodMemberType(ty))
      continue;
    if (!writem.contains(m.getSymNameAttr()))
      continue;
    total += getMemberFlatSizeImpl(ty, module, visited);
  }
  visited.erase(leaf);
  return total;
}

int64_t getMemberFlatSizeImpl(Type memberType, ModuleOp module,
                              DenseSet<StringRef> &visited) {
  if (LlzkToStablehloTypeConverter::isArrayType(memberType)) {
    auto dims = getArrayDimensions(memberType);
    int64_t product = 1;
    for (int64_t d : dims)
      product *= (d == ShapedType::kDynamic) ? 1 : d;
    if (auto arr = dyn_cast<llzk::array::ArrayType>(memberType))
      return product *
             getMemberFlatSizeImpl(arr.getElementType(), module, visited);
    return product;
  }
  if (LlzkToStablehloTypeConverter::isStructType(memberType)) {
    auto sty = dyn_cast<llzk::component::StructType>(memberType);
    if (!sty || !module)
      return 1;
    if (auto def = findStructDefByExactSymbol(module, sty))
      return getStructFlatSize(def, module, visited);
    return 1;
  }
  return 1;
}

} // namespace

int64_t getMemberFlatSize(Type memberType, ModuleOp module) {
  DenseSet<StringRef> visited;
  return getMemberFlatSizeImpl(memberType, module, visited);
}

DenseSet<StringAttr> collectWritemTargets(Operation *structDef) {
  DenseSet<StringAttr> targets;
  structDef->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "struct.writem")
      return;
    if (auto memberRef = op->getAttrOfType<FlatSymbolRefAttr>("member_name"))
      targets.insert(memberRef.getAttr());
  });
  return targets;
}

int64_t getStaticShapeProduct(RankedTensorType t) {
  int64_t size = 1;
  for (int64_t d : t.getShape())
    if (d != ShapedType::kDynamic)
      size *= d;
  return size;
}

bool isZeroSplatConstant(Value v) {
  auto cst = v.getDefiningOp<stablehlo::ConstantOp>();
  if (!cst)
    return false;
  auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
  return attr && attr.isSplat() && attr.getElementType().isInteger() &&
         attr.getSplatValue<APInt>().isZero();
}

Value lookThroughCast(Value v) {
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    if (castOp.getNumOperands() == 1)
      return castOp.getOperand(0);
  return v;
}

Value ensureTensorType(OpBuilder &b, Value v, Type originalType,
                       const LlzkToStablehloTypeConverter &tc, Location loc) {
  v = lookThroughCast(v);
  if (isa<RankedTensorType>(v.getType()))
    return v;
  Type converted = tc.convertType(originalType);
  if (!converted)
    return v;
  return b.create<UnrealizedConversionCastOp>(loc, converted, v).getResult(0);
}

Value convertToIndexTensor(OpBuilder &b, Value idx,
                           const LlzkToStablehloTypeConverter &tc,
                           Location loc) {
  auto i32TensorType = RankedTensorType::get({}, b.getI32Type());

  // If already a 0-d i32 tensor, return as-is.
  if (auto tt = dyn_cast<RankedTensorType>(idx.getType()))
    if (tt.getRank() == 0 && tt.getElementType().isInteger(32))
      return idx;

  // Look through unrealized_conversion_cast chain.
  Value v = idx;
  for (int i = 0; i < 3; ++i) {
    if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      v = castOp.getInputs()[0];
      if (auto tt = dyn_cast<RankedTensorType>(v.getType()))
        if (tt.getRank() == 0 && tt.getElementType().isInteger(32))
          return v;
    } else {
      break;
    }
  }

  // Look through cast.toindex. First try a `felt.const` operand for a
  // statically known index — this lets the resulting `dense<N>` constant
  // be folded through later passes.
  if (auto *defOp = idx.getDefiningOp()) {
    if (defOp->getName().getStringRef() == "cast.toindex" &&
        defOp->getNumOperands() > 0) {
      if (auto *feltOp = defOp->getOperand(0).getDefiningOp()) {
        if (feltOp->getName().getStringRef() == "felt.const") {
          if (auto valAttr = feltOp->getAttr("value")) {
            std::optional<APInt> ap;
            if (auto feltConst = dyn_cast<llzk::felt::FeltConstAttr>(valAttr))
              ap = feltConst.getValue();
            else if (auto intAttr = dyn_cast<IntegerAttr>(valAttr))
              ap = intAttr.getValue();
            if (ap) {
              // A field constant whose value doesn't fit in a signed
              // 64 bits can't be a valid array index — bail instead of
              // asserting inside APInt::getSExtValue. Field elements
              // can be 254 bits; real indices are bounded by array.len.
              if (ap->getSignificantBits() > 64)
                return Value();
              int64_t val = ap->getSExtValue();
              return b.create<stablehlo::ConstantOp>(
                  loc, DenseElementsAttr::get(i32TensorType,
                                              b.getI32IntegerAttr(val)));
            }
          }
        }
      }
      // Otherwise convert the felt operand to tensor<i32>. The operand
      // may still be in its original `!felt.type` form (e.g. an scf.while
      // iter-arg whose SCF structural conversion has not yet rebuilt the
      // block); materialize a target conversion so the slice index is
      // tied to the runtime loop counter rather than silently folded to 0.
      Value feltOperand = defOp->getOperand(0);
      Value feltVal = lookThroughCast(feltOperand);
      if (!isa<RankedTensorType>(feltVal.getType()))
        feltVal =
            ensureTensorType(b, feltOperand, feltOperand.getType(), tc, loc);
      if (isa<RankedTensorType>(feltVal.getType()))
        return b.create<stablehlo::ConvertOp>(loc, i32TensorType, feltVal);
    }
  }

  // If it's a 0-d tensor of another integer type, convert via stablehlo.
  if (auto tt = dyn_cast<RankedTensorType>(v.getType()))
    if (tt.getRank() == 0 && tt.getElementType().isIntOrIndex())
      return b.create<stablehlo::ConvertOp>(loc, i32TensorType, v);

  // Bare integer scalar (not wrapped in a tensor) backed by arith.constant.
  if (auto *defOp = v.getDefiningOp()) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t val = intAttr.getInt();
        return b.create<stablehlo::ConstantOp>(
            loc,
            DenseElementsAttr::get(i32TensorType, b.getI32IntegerAttr(val)));
      }
    }
  }
  // Bare integer/index scalar produced by another op (e.g. arith.subi from
  // `array.len - 1`). Cast to i32 if needed and wrap in a 0-d tensor.
  if (v.getType().isIntOrIndex()) {
    Value asI32 = v;
    if (!v.getType().isInteger(32))
      asI32 = b.create<arith::IndexCastOp>(loc, b.getI32Type(), v);
    return b.create<tensor::FromElementsOp>(loc, i32TensorType,
                                            ValueRange{asI32});
  }
  // No conversion path matched. A constant-0 fallback would silently pin
  // every slice to index 0; return a null Value instead so the caller's
  // pattern fails matchAndRewrite and the driver retries once operand
  // dependencies settle.
  return Value();
}

Value createIndexConstant(OpBuilder &b, Location loc, int64_t value) {
  auto type = RankedTensorType::get({}, b.getI32Type());
  auto attr = DenseElementsAttr::get(type, b.getI32IntegerAttr(value));
  return b.create<stablehlo::ConstantOp>(loc, attr);
}

std::optional<int64_t> parseBoolCmpPredicate(Attribute predicateAttr) {
  // Try direct integer value first
  if (auto intAttr = dyn_cast<IntegerAttr>(predicateAttr))
    return intAttr.getInt();

  // Fall back to string parsing: enum printed as #bool<cmp lt> etc.
  std::string s;
  llvm::raw_string_ostream os(s);
  predicateAttr.print(os);
  StringRef spelled = StringRef(s).trim();

  // Predicate enum: eq=0, ne=1, lt=2, le=3, gt=4, ge=5
  return llvm::StringSwitch<std::optional<int64_t>>(spelled)
      .Case("#bool<cmp eq>", 0)
      .Case("#bool<cmp ne>", 1)
      .Case("#bool<cmp lt>", 2)
      .Case("#bool<cmp le>", 3)
      .Case("#bool<cmp gt>", 4)
      .Case("#bool<cmp ge>", 5)
      .Default(std::nullopt);
}

ArrayAttr getPodInitializedRecordsAttr(Operation *podNewOp) {
  auto newPod = dyn_cast<llzk::pod::NewPodOp>(podNewOp);
  if (!newPod)
    return {};
  return newPod.getInitializedRecordsAttr();
}

SmallVector<std::string> getPodInitializedRecords(Operation *podNewOp) {
  SmallVector<std::string> fieldNames;
  auto initAttr = getPodInitializedRecordsAttr(podNewOp);
  if (!initAttr)
    return fieldNames;

  fieldNames.reserve(initAttr.size());
  for (Attribute attr : initAttr) {
    auto nameAttr = dyn_cast<StringAttr>(attr);
    if (!nameAttr)
      return {};
    fieldNames.push_back(nameAttr.getValue().str());
  }
  return fieldNames;
}

// ===----------------------------------------------------------------------===
// Type converter implementation
// ===----------------------------------------------------------------------===

LlzkToStablehloTypeConverter::LlzkToStablehloTypeConverter(
    MLIRContext *ctx, const llvm::APInt &prime, unsigned storageBitWidth,
    bool usePrimeFieldType) {
  auto intType = IntegerType::get(ctx, storageBitWidth);
  auto modulusAttr = IntegerAttr::get(intType, prime);
  primeFieldType = prime_ir::field::PrimeFieldType::get(ctx, modulusAttr);

  this->usePrimeFieldType = usePrimeFieldType;
  this->elementType = usePrimeFieldType ? Type(primeFieldType) : Type(intType);

  // Identity conversion for types we don't need to convert
  addConversion([](Type type) { return type; });

  // Convert bare i1 → tensor<i1>.
  // StableHLO requires all values to be tensors. Bare i1 appears from
  // bool.cmp results and llzk.nondet : i1 used as scf.while carry.
  // SCF structural type conversion uses this to wrap while carry / if results.
  addConversion([](IntegerType type) -> std::optional<Type> {
    if (type.getWidth() != 1)
      return std::nullopt;
    return RankedTensorType::get({}, type);
  });

  // Convert felt.type → tensor<!pf>
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isFeltType(type))
      return std::nullopt;
    return RankedTensorType::get({}, elementType);
  });

  // Convert !array.type<N x felt> → tensor<N x !pf>
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isArrayType(type))
      return std::nullopt;
    return RankedTensorType::get(getArrayDimensions(type), elementType);
  });

  // Convert !struct.type<@Name> → tensor<M x !pf> (flattened)
  addConversion([this](Type type) -> std::optional<Type> {
    if (!isStructType(type))
      return std::nullopt;
    auto flatSize = getStructFlattenedSize(type);
    if (!flatSize)
      // Default to dynamic if struct not registered
      return RankedTensorType::get({ShapedType::kDynamic}, elementType);
    return RankedTensorType::get({*flatSize}, elementType);
  });

  // Convert function types recursively
  addConversion([this](FunctionType type) -> Type {
    SmallVector<Type> inputs, results;
    for (Type t : type.getInputs())
      inputs.push_back(convertType(t));
    for (Type t : type.getResults())
      results.push_back(convertType(t));
    return FunctionType::get(type.getContext(), inputs, results);
  });

  // Source materialization: convert from target type back to source type
  addSourceMaterialization([](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  // Target materialization: convert from source type to target type
  addTargetMaterialization([](OpBuilder &builder, Type resultType,
                              ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

std::optional<int64_t>
LlzkToStablehloTypeConverter::getStructFlattenedSize(Type structType) const {
  auto it = structFlattenedSizes.find(structType);
  if (it == structFlattenedSizes.end())
    return std::nullopt;
  return it->second;
}

void LlzkToStablehloTypeConverter::registerStructFlattenedSize(Type structType,
                                                               int64_t size) {
  structFlattenedSizes[structType] = size;
}

std::optional<int64_t>
LlzkToStablehloTypeConverter::getFieldOffset(Type structType,
                                             StringRef fieldName) const {
  auto *ctx = structType.getContext();
  auto key = std::make_pair(structType, StringAttr::get(ctx, fieldName));
  auto it = fieldOffsets.find(key);
  if (it == fieldOffsets.end())
    return std::nullopt;
  return it->second;
}

void LlzkToStablehloTypeConverter::registerFieldOffset(Type structType,
                                                       StringRef fieldName,
                                                       int64_t offset) {
  auto *ctx = structType.getContext();
  auto key = std::make_pair(structType, StringAttr::get(ctx, fieldName));
  fieldOffsets[key] = offset;
}

IntegerType LlzkToStablehloTypeConverter::getStorageType() const {
  if (usePrimeFieldType)
    return primeFieldType.getStorageType();
  return cast<IntegerType>(elementType);
}

DenseElementsAttr LlzkToStablehloTypeConverter::createConstantAttr(
    RankedTensorType tensorType, int64_t value, OpBuilder &rewriter) const {
  auto storageType = getStorageType();
  auto storageTensorType =
      RankedTensorType::get(tensorType.getShape(), storageType);
  return DenseElementsAttr::get(storageTensorType,
                                rewriter.getIntegerAttr(storageType, value));
}

DenseElementsAttr
LlzkToStablehloTypeConverter::createConstantAttr(RankedTensorType tensorType,
                                                 const llvm::APInt &value,
                                                 OpBuilder &rewriter) const {
  auto storageType = getStorageType();
  // Zero-extend (not sign-extend): field elements are unsigned by definition,
  // ranged in [0, p) with p < 2^254 < 2^256.
  llvm::APInt resized = value.zextOrTrunc(storageType.getWidth());
  auto storageTensorType =
      RankedTensorType::get(tensorType.getShape(), storageType);
  return DenseElementsAttr::get(storageTensorType,
                                IntegerAttr::get(storageType, resized));
}

} // namespace mlir::llzk_to_shlo
