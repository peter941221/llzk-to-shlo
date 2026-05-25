// RUN: llzk-to-shlo-opt --simplify-sub-components %s | FileCheck %s

// Regression: `materializePodArrayCompField` must resolve inner `struct.def`
// by the FULL `SymbolRefAttr`, not just the leaf symbol.
//
// Shape:
//   1. Two sibling wrapper modules each define a leaf-identical `@Sub`.
//   2. `@ChipA::@Sub` is the REAL dispatch target and exposes
//      `@out : !felt`.
//   3. `@ChipB::@Sub` is a decoy with the same leaf symbol but
//      `@out : !array<2 x !felt>`.
//   4. The parent dispatch pod array stores `!struct<@ChipA::@Sub>`.
//
// Before the fix, `findInnerFeltMembers` / recursive drain lookup matched the
// first leaf-identical `struct.def @Sub` they walked to, so the decoy `ChipB`
// could mis-size the materialized felt array and crash or silently miswire the
// reader/drain rewrite. Exact-symbol lookup must bind to `@ChipA::@Sub`.
//
// CHECK-LABEL: struct.def @Main
// The drain member must be retyped to the REAL inner flat shape: scalar
// `@ChipA::@Sub.@out` means `<2 x !felt>`, not the decoy `<2,2 x !felt>`.
// CHECK: struct.member @drain : !array.type<2 x !felt.type>
// CHECK-LABEL: function.def @compute
// Materialized felt carrier also stays 1-D over dispatch slots.
// CHECK: array.new {{.*}}: <2 x !felt.type>
// CHECK-NOT: array.new {{.*}}: <2,2 x !felt.type>

module attributes {llzk.lang, llzk.main = !struct.type<@Main<[]>>} {
  builtin.module @ChipB {
    struct.def @Sub {
      struct.member @out : !array.type<2 x !felt.type> {llzk.pub}
      function.def @compute(%arg0: !felt.type) -> !struct.type<@ChipB::@Sub<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
        %self = struct.new : <@ChipB::@Sub<[]>>
        %nondet = llzk.nondet : !array.type<2 x !felt.type>
        struct.writem %self[@out] = %nondet : <@ChipB::@Sub<[]>>, !array.type<2 x !felt.type>
        function.return %self : !struct.type<@ChipB::@Sub<[]>>
      }
      function.def @constrain(%arg0: !struct.type<@ChipB::@Sub<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
        function.return
      }
    }
  }

  builtin.module @ChipA {
    struct.def @Sub {
      struct.member @out : !felt.type {llzk.pub}
      function.def @compute(%arg0: !felt.type) -> !struct.type<@ChipA::@Sub<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
        %self = struct.new : <@ChipA::@Sub<[]>>
        struct.writem %self[@out] = %arg0 : <@ChipA::@Sub<[]>>, !felt.type
        function.return %self : !struct.type<@ChipA::@Sub<[]>>
      }
      function.def @constrain(%arg0: !struct.type<@ChipA::@Sub<[]>>, %arg1: !felt.type) attributes {function.allow_constraint} {
        function.return
      }
    }
  }

  struct.def @Main {
    struct.member @drain$inputs : !array.type<2 x !pod.type<[@in: !felt.type]>>
    struct.member @drain : !array.type<2 x !struct.type<@ChipA::@Sub<[]>>>
    function.def @compute(%arg0: !array.type<2 x !felt.type>) -> !struct.type<@Main<[]>> attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@Main<[]>>
      %array = array.new : <2 x !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>>
      %inputs = array.new : <2 x !pod.type<[@in: !felt.type]>>
      %destStruct = array.new : <2 x !struct.type<@ChipA::@Sub<[]>>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      scf.for %i = %c0 to %c2 step %c1 {
        %p = array.read %array[%i] : <2 x !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>
        pod.write %p[@count] = %c1 : <[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>, index
        array.write %array[%i] = %p : <2 x !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>
      }
      %felt0 = felt.const 0
      %0 = scf.while (%iter = %felt0) : (!felt.type) -> !felt.type {
        %felt2 = felt.const 2
        %cond = bool.cmp lt(%iter, %felt2)
        scf.condition(%cond) %iter : !felt.type
      } do {
      ^bb0(%iter: !felt.type):
        %idx = cast.toindex %iter
        %a = array.read %arg0[%idx] : <2 x !felt.type>, !felt.type
        %ip = array.read %inputs[%idx] : <2 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        pod.write %ip[@in] = %a : <[@in: !felt.type]>, !felt.type
        array.write %inputs[%idx] = %ip : <2 x !pod.type<[@in: !felt.type]>>, !pod.type<[@in: !felt.type]>
        %dp = array.read %array[%idx] : <2 x !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>
        %cnt = pod.read %dp[@count] : <[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>, index
        %cnt2 = arith.subi %cnt, %c1 : index
        pod.write %dp[@count] = %cnt2 : <[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>, index
        %fire = arith.cmpi eq, %cnt2, %c0 : index
        scf.if %fire {
          %res = function.call @ChipA::@Sub::@compute(%a) : (!felt.type) -> !struct.type<@ChipA::@Sub<[]>>
          pod.write %dp[@comp] = %res : <[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>, !struct.type<@ChipA::@Sub<[]>>
          array.write %array[%idx] = %dp : <2 x !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>
        }
        %felt1 = felt.const 1
        %next = felt.add %iter, %felt1 : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      scf.for %j = %c0 to %c2 step %c1 {
        %dp2 = array.read %array[%j] : <2 x !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>>, !pod.type<[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>
        %comp = pod.read %dp2[@comp] : <[@count: index, @comp: !struct.type<@ChipA::@Sub<[]>>, @params: !pod.type<[]>]>, !struct.type<@ChipA::@Sub<[]>>
        array.write %destStruct[%j] = %comp : <2 x !struct.type<@ChipA::@Sub<[]>>>, !struct.type<@ChipA::@Sub<[]>>
      }
      struct.writem %self[@drain$inputs] = %inputs : <@Main<[]>>, !array.type<2 x !pod.type<[@in: !felt.type]>>
      struct.writem %self[@drain] = %destStruct : <@Main<[]>>, !array.type<2 x !struct.type<@ChipA::@Sub<[]>>>
      function.return %self : !struct.type<@Main<[]>>
    }
    function.def @constrain(%arg0: !struct.type<@Main<[]>>, %arg1: !array.type<2 x !felt.type>) attributes {function.allow_constraint} {
      function.return
    }
  }
}
