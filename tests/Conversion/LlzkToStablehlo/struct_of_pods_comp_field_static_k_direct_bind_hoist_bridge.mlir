// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Static-K direct-binding regression for hoist-bridged writers.
//
// The writer call sits under a count-guard `scf.if` inside an
// `scf.execute_region`, so `materializeStructOfPodsCompField` records
// `hoistAbove != nullptr` and re-emits the call before the outermost
// count-guard. Prior to this regression, the static-K fast path rejected
// every such writer up front and fell back to the per-field carrier even
// though the hoisted clone still dominates the reader. The fast path should
// now treat hoistable static-K writers the same as already-hoisted writers
// and wire the cloned writer-side `struct.readm` directly into the reader,
// without allocating a per-field carrier array.
//
// CHECK-LABEL: function.def @compute(
// CHECK-NOT: "mSoPCF.carrier-for"
// CHECK-DAG: %[[CALL:.*]] = function.call @Sub_A::@compute(
// CHECK-DAG: %[[OUT:.*]] = struct.readm %[[CALL]][@out]
// CHECK: scf.yield %[[OUT]] : !felt.type

module attributes {llzk.lang, llzk.main = !struct.type<@Main_0<[]>>} {
  struct.def @Sub_A {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%arg0: !array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Sub_A<[]>>
      %c0 = arith.constant 0 : index
      %v = array.read %arg0[%c0] : <1 x !felt.type>, !felt.type
      struct.writem %self[@out] = %v : <@Sub_A<[]>>, !felt.type
      function.return %self : !struct.type<@Sub_A<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Sub_A<[]>>, %arg1: !array.type<1 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }

  struct.def @Main_0 {
    struct.member @out : !array.type<1 x !felt.type> {llzk.pub}
    function.def @compute(%arg0: !array.type<1, 1 x !felt.type>) -> !struct.type<@Main_0<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main_0<[]>>
      %nondet_out = llzk.nondet : !array.type<1 x !felt.type>
      %c0 = arith.constant 0 : index

      %dummy = pod.new : <[@d: index]>
      %dummy_read = pod.read %dummy[@d] : <[@d: index]>, index

      %count = arith.constant 1 : index
      %bridged = scf.execute_region -> !felt.type {
        %ext = array.extract %arg0[%c0] : <1, 1 x !felt.type>
        %sub = arith.subi %c0, %count : index
        %fire = arith.cmpi eq, %sub, %c0 : index
        scf.if %fire {
          %call = function.call @Sub_A::@compute(%ext) : (!array.type<1 x !felt.type>) -> !struct.type<@Sub_A<[]>>
        }
        %reader_src = llzk.nondet : !struct.type<@Sub_A<[]>>
        %read = struct.readm %reader_src[@out] : <@Sub_A<[]>>, !felt.type
        scf.yield %read : !felt.type
      }
      array.write %nondet_out[%c0] = %bridged : <1 x !felt.type>, !felt.type

      struct.writem %self[@out] = %nondet_out : <@Main_0<[]>>, !array.type<1 x !felt.type>
      function.return %self : !struct.type<@Main_0<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main_0<[]>>, %arg1: !array.type<1, 1 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
