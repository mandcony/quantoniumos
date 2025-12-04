// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_phi_rft_core.h for the primary calling header

#include "Vtb_phi_rft_core__pch.h"
#include "Vtb_phi_rft_core___024root.h"

VL_ATTR_COLD void Vtb_phi_rft_core___024root___eval_initial__TOP(Vtb_phi_rft_core___024root* vlSelf);
VlCoroutine Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__0(Vtb_phi_rft_core___024root* vlSelf);
VlCoroutine Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__1(Vtb_phi_rft_core___024root* vlSelf);
VlCoroutine Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__2(Vtb_phi_rft_core___024root* vlSelf);

void Vtb_phi_rft_core___024root___eval_initial(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_initial\n"); );
    // Body
    Vtb_phi_rft_core___024root___eval_initial__TOP(vlSelf);
    vlSelf->__Vm_traceActivity[1U] = 1U;
    Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__1(vlSelf);
    Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__2(vlSelf);
    vlSelf->__Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__clk__0 
        = vlSelf->tb_phi_rft_core__DOT__clk;
    vlSelf->__Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__rst_n__0 
        = vlSelf->tb_phi_rft_core__DOT__rst_n;
}

VL_INLINE_OPT VlCoroutine Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__0(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__0\n"); );
    // Init
    IData/*31:0*/ __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 0;
    std::string __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name;
    IData/*31:0*/ __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle = 0;
    IData/*31:0*/ __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency = 0;
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[__Vi0] = 0;
    }
    IData/*31:0*/ __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0;
    IData/*31:0*/ __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0;
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[__Vi0] = 0;
    }
    IData/*31:0*/ __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout = 0;
    IData/*31:0*/ __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count = 0;
    VlUnpacked<SData/*15:0*/, 8> __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[__Vi0] = 0;
    }
    // Body
    VL_WRITEF("\n######################################################\n# phi_rft_core Standalone Testbench                  #\n# Verifying \316\246-RFT Transform + SIS Digest             #\n######################################################\n\n");
    vlSelf->tb_phi_rft_core__DOT__rst_n = 0U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    vlSelf->tb_phi_rft_core__DOT__samples[0U] = 0U;
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = 0U;
    vlSelf->tb_phi_rft_core__DOT__samples[2U] = 0U;
    vlSelf->tb_phi_rft_core__DOT__samples[3U] = 0U;
    vlSelf->tb_phi_rft_core__DOT__mode = 0U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       220);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__rst_n = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       222);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       222);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       222);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       222);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       222);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    VL_WRITEF("Reset complete, starting tests...\n");
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name 
        = VL_CVT_PACK_STR_NN(vlSelf->tb_phi_rft_core__DOT__test_names
                             [0U]);
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 0U;
    VL_WRITEF("\n--- Test %0d: %@ ---\n",32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx,
              -1,&(__Vtask_tb_phi_rft_core__DOT__run_single_test__0__name));
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    if (((((((((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx) 
               | (1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
              | (2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
             | (3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
            | (4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
           | (5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
          | (6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
         | (7U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx))) {
        if ((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                   [6U]);
        } else if ((1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                   [6U]);
        } else if ((2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                   [6U]);
        } else if ((3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                   [6U]);
        } else if ((4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                   [6U]);
        } else if ((5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                   [6U]);
        } else if ((6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                   [6U]);
        } else {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                   [6U]);
        }
    } else {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[0U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [0U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[1U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [1U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[2U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [2U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[3U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [3U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[4U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [4U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[5U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [5U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[6U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [6U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[7U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [7U];
        vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [0U]))));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [2U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [1U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [0U]))) 
                        >> 0x20U)));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [5U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [4U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [3U])))) 
                << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[2U] = 
            (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                          [5U] << 0x10U) 
                                         | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [4U]))) << 0x10U) 
                       | (QData)((IData)(__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [3U])))) >> 0x10U) 
             | ((IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                             [5U] << 0x10U) 
                                            | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [4U]))) 
                           << 0x10U) | (QData)((IData)(
                                                       __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                       [3U]))) 
                         >> 0x20U)) << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[3U] = 
            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
              [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
             [6U]);
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       174);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle 
        = vlSelf->tb_phi_rft_core__DOT__cycle_count;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       177);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency 
        = (vlSelf->tb_phi_rft_core__DOT__cycle_count 
           - __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle);
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if (vlSelf->tb_phi_rft_core__DOT__digest_valid) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Digest computed in %0d cycles\n       Digest[255:192] = %016x\n       Digest[191:128] = %016x\n       Digest[127:64]  = %016x\n       Digest[63:0]    = %016x\n       Resonance = %b\n",
                  32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency,
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[7U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[6U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[5U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[4U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[3U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[2U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[1U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[0U]))),
                  1,(IData)(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] No digest_valid received\n");
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           200);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name 
        = VL_CVT_PACK_STR_NN(vlSelf->tb_phi_rft_core__DOT__test_names
                             [1U]);
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 1U;
    VL_WRITEF("\n--- Test %0d: %@ ---\n",32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx,
              -1,&(__Vtask_tb_phi_rft_core__DOT__run_single_test__0__name));
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    if (((((((((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx) 
               | (1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
              | (2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
             | (3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
            | (4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
           | (5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
          | (6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
         | (7U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx))) {
        if ((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                   [6U]);
        } else if ((1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                   [6U]);
        } else if ((2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                   [6U]);
        } else if ((3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                   [6U]);
        } else if ((4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                   [6U]);
        } else if ((5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                   [6U]);
        } else if ((6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                   [6U]);
        } else {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                   [6U]);
        }
    } else {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[0U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [0U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[1U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [1U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[2U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [2U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[3U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [3U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[4U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [4U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[5U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [5U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[6U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [6U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[7U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [7U];
        vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [0U]))));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [2U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [1U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [0U]))) 
                        >> 0x20U)));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [5U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [4U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [3U])))) 
                << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[2U] = 
            (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                          [5U] << 0x10U) 
                                         | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [4U]))) << 0x10U) 
                       | (QData)((IData)(__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [3U])))) >> 0x10U) 
             | ((IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                             [5U] << 0x10U) 
                                            | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [4U]))) 
                           << 0x10U) | (QData)((IData)(
                                                       __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                       [3U]))) 
                         >> 0x20U)) << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[3U] = 
            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
              [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
             [6U]);
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       174);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle 
        = vlSelf->tb_phi_rft_core__DOT__cycle_count;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       177);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency 
        = (vlSelf->tb_phi_rft_core__DOT__cycle_count 
           - __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle);
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if (vlSelf->tb_phi_rft_core__DOT__digest_valid) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Digest computed in %0d cycles\n       Digest[255:192] = %016x\n       Digest[191:128] = %016x\n       Digest[127:64]  = %016x\n       Digest[63:0]    = %016x\n       Resonance = %b\n",
                  32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency,
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[7U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[6U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[5U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[4U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[3U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[2U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[1U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[0U]))),
                  1,(IData)(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] No digest_valid received\n");
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           200);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t = 2U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name 
        = VL_CVT_PACK_STR_NN(vlSelf->tb_phi_rft_core__DOT__test_names
                             [2U]);
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 2U;
    VL_WRITEF("\n--- Test %0d: %@ ---\n",32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx,
              -1,&(__Vtask_tb_phi_rft_core__DOT__run_single_test__0__name));
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    if (((((((((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx) 
               | (1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
              | (2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
             | (3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
            | (4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
           | (5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
          | (6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
         | (7U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx))) {
        if ((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                   [6U]);
        } else if ((1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                   [6U]);
        } else if ((2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                   [6U]);
        } else if ((3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                   [6U]);
        } else if ((4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                   [6U]);
        } else if ((5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                   [6U]);
        } else if ((6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                   [6U]);
        } else {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                   [6U]);
        }
    } else {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[0U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [0U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[1U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [1U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[2U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [2U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[3U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [3U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[4U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [4U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[5U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [5U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[6U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [6U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[7U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [7U];
        vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [0U]))));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [2U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [1U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [0U]))) 
                        >> 0x20U)));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [5U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [4U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [3U])))) 
                << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[2U] = 
            (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                          [5U] << 0x10U) 
                                         | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [4U]))) << 0x10U) 
                       | (QData)((IData)(__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [3U])))) >> 0x10U) 
             | ((IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                             [5U] << 0x10U) 
                                            | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [4U]))) 
                           << 0x10U) | (QData)((IData)(
                                                       __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                       [3U]))) 
                         >> 0x20U)) << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[3U] = 
            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
              [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
             [6U]);
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       174);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle 
        = vlSelf->tb_phi_rft_core__DOT__cycle_count;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       177);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency 
        = (vlSelf->tb_phi_rft_core__DOT__cycle_count 
           - __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle);
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if (vlSelf->tb_phi_rft_core__DOT__digest_valid) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Digest computed in %0d cycles\n       Digest[255:192] = %016x\n       Digest[191:128] = %016x\n       Digest[127:64]  = %016x\n       Digest[63:0]    = %016x\n       Resonance = %b\n",
                  32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency,
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[7U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[6U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[5U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[4U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[3U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[2U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[1U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[0U]))),
                  1,(IData)(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] No digest_valid received\n");
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           200);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t = 3U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name 
        = VL_CVT_PACK_STR_NN(vlSelf->tb_phi_rft_core__DOT__test_names
                             [3U]);
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 3U;
    VL_WRITEF("\n--- Test %0d: %@ ---\n",32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx,
              -1,&(__Vtask_tb_phi_rft_core__DOT__run_single_test__0__name));
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    if (((((((((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx) 
               | (1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
              | (2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
             | (3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
            | (4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
           | (5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
          | (6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
         | (7U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx))) {
        if ((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                   [6U]);
        } else if ((1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                   [6U]);
        } else if ((2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                   [6U]);
        } else if ((3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                   [6U]);
        } else if ((4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                   [6U]);
        } else if ((5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                   [6U]);
        } else if ((6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                   [6U]);
        } else {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                   [6U]);
        }
    } else {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[0U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [0U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[1U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [1U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[2U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [2U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[3U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [3U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[4U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [4U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[5U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [5U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[6U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [6U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[7U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [7U];
        vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [0U]))));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [2U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [1U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [0U]))) 
                        >> 0x20U)));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [5U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [4U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [3U])))) 
                << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[2U] = 
            (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                          [5U] << 0x10U) 
                                         | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [4U]))) << 0x10U) 
                       | (QData)((IData)(__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [3U])))) >> 0x10U) 
             | ((IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                             [5U] << 0x10U) 
                                            | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [4U]))) 
                           << 0x10U) | (QData)((IData)(
                                                       __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                       [3U]))) 
                         >> 0x20U)) << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[3U] = 
            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
              [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
             [6U]);
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       174);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle 
        = vlSelf->tb_phi_rft_core__DOT__cycle_count;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       177);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency 
        = (vlSelf->tb_phi_rft_core__DOT__cycle_count 
           - __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle);
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if (vlSelf->tb_phi_rft_core__DOT__digest_valid) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Digest computed in %0d cycles\n       Digest[255:192] = %016x\n       Digest[191:128] = %016x\n       Digest[127:64]  = %016x\n       Digest[63:0]    = %016x\n       Resonance = %b\n",
                  32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency,
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[7U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[6U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[5U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[4U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[3U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[2U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[1U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[0U]))),
                  1,(IData)(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] No digest_valid received\n");
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           200);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t = 4U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name 
        = VL_CVT_PACK_STR_NN(vlSelf->tb_phi_rft_core__DOT__test_names
                             [4U]);
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 4U;
    VL_WRITEF("\n--- Test %0d: %@ ---\n",32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx,
              -1,&(__Vtask_tb_phi_rft_core__DOT__run_single_test__0__name));
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    if (((((((((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx) 
               | (1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
              | (2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
             | (3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
            | (4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
           | (5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
          | (6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
         | (7U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx))) {
        if ((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                   [6U]);
        } else if ((1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                   [6U]);
        } else if ((2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                   [6U]);
        } else if ((3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                   [6U]);
        } else if ((4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                   [6U]);
        } else if ((5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                   [6U]);
        } else if ((6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                   [6U]);
        } else {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                   [6U]);
        }
    } else {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[0U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [0U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[1U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [1U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[2U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [2U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[3U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [3U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[4U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [4U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[5U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [5U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[6U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [6U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[7U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [7U];
        vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [0U]))));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [2U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [1U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [0U]))) 
                        >> 0x20U)));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [5U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [4U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [3U])))) 
                << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[2U] = 
            (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                          [5U] << 0x10U) 
                                         | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [4U]))) << 0x10U) 
                       | (QData)((IData)(__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [3U])))) >> 0x10U) 
             | ((IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                             [5U] << 0x10U) 
                                            | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [4U]))) 
                           << 0x10U) | (QData)((IData)(
                                                       __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                       [3U]))) 
                         >> 0x20U)) << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[3U] = 
            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
              [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
             [6U]);
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       174);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle 
        = vlSelf->tb_phi_rft_core__DOT__cycle_count;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       177);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency 
        = (vlSelf->tb_phi_rft_core__DOT__cycle_count 
           - __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle);
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if (vlSelf->tb_phi_rft_core__DOT__digest_valid) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Digest computed in %0d cycles\n       Digest[255:192] = %016x\n       Digest[191:128] = %016x\n       Digest[127:64]  = %016x\n       Digest[63:0]    = %016x\n       Resonance = %b\n",
                  32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency,
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[7U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[6U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[5U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[4U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[3U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[2U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[1U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[0U]))),
                  1,(IData)(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] No digest_valid received\n");
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           200);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t = 5U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name 
        = VL_CVT_PACK_STR_NN(vlSelf->tb_phi_rft_core__DOT__test_names
                             [5U]);
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 5U;
    VL_WRITEF("\n--- Test %0d: %@ ---\n",32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx,
              -1,&(__Vtask_tb_phi_rft_core__DOT__run_single_test__0__name));
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    if (((((((((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx) 
               | (1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
              | (2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
             | (3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
            | (4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
           | (5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
          | (6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
         | (7U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx))) {
        if ((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                   [6U]);
        } else if ((1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                   [6U]);
        } else if ((2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                   [6U]);
        } else if ((3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                   [6U]);
        } else if ((4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                   [6U]);
        } else if ((5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                   [6U]);
        } else if ((6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                   [6U]);
        } else {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                   [6U]);
        }
    } else {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[0U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [0U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[1U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [1U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[2U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [2U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[3U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [3U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[4U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [4U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[5U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [5U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[6U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [6U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[7U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [7U];
        vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [0U]))));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [2U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [1U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [0U]))) 
                        >> 0x20U)));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [5U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [4U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [3U])))) 
                << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[2U] = 
            (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                          [5U] << 0x10U) 
                                         | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [4U]))) << 0x10U) 
                       | (QData)((IData)(__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [3U])))) >> 0x10U) 
             | ((IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                             [5U] << 0x10U) 
                                            | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [4U]))) 
                           << 0x10U) | (QData)((IData)(
                                                       __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                       [3U]))) 
                         >> 0x20U)) << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[3U] = 
            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
              [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
             [6U]);
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       174);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle 
        = vlSelf->tb_phi_rft_core__DOT__cycle_count;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       177);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency 
        = (vlSelf->tb_phi_rft_core__DOT__cycle_count 
           - __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle);
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if (vlSelf->tb_phi_rft_core__DOT__digest_valid) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Digest computed in %0d cycles\n       Digest[255:192] = %016x\n       Digest[191:128] = %016x\n       Digest[127:64]  = %016x\n       Digest[63:0]    = %016x\n       Resonance = %b\n",
                  32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency,
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[7U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[6U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[5U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[4U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[3U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[2U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[1U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[0U]))),
                  1,(IData)(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] No digest_valid received\n");
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           200);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t = 6U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name 
        = VL_CVT_PACK_STR_NN(vlSelf->tb_phi_rft_core__DOT__test_names
                             [6U]);
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 6U;
    VL_WRITEF("\n--- Test %0d: %@ ---\n",32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx,
              -1,&(__Vtask_tb_phi_rft_core__DOT__run_single_test__0__name));
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    if (((((((((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx) 
               | (1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
              | (2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
             | (3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
            | (4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
           | (5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
          | (6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
         | (7U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx))) {
        if ((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                   [6U]);
        } else if ((1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                   [6U]);
        } else if ((2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                   [6U]);
        } else if ((3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                   [6U]);
        } else if ((4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                   [6U]);
        } else if ((5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                   [6U]);
        } else if ((6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                   [6U]);
        } else {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                   [6U]);
        }
    } else {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[0U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [0U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[1U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [1U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[2U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [2U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[3U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [3U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[4U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [4U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[5U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [5U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[6U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [6U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[7U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [7U];
        vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [0U]))));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [2U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [1U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [0U]))) 
                        >> 0x20U)));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [5U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [4U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [3U])))) 
                << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[2U] = 
            (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                          [5U] << 0x10U) 
                                         | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [4U]))) << 0x10U) 
                       | (QData)((IData)(__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [3U])))) >> 0x10U) 
             | ((IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                             [5U] << 0x10U) 
                                            | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [4U]))) 
                           << 0x10U) | (QData)((IData)(
                                                       __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                       [3U]))) 
                         >> 0x20U)) << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[3U] = 
            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
              [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
             [6U]);
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       174);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle 
        = vlSelf->tb_phi_rft_core__DOT__cycle_count;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       177);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency 
        = (vlSelf->tb_phi_rft_core__DOT__cycle_count 
           - __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle);
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if (vlSelf->tb_phi_rft_core__DOT__digest_valid) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Digest computed in %0d cycles\n       Digest[255:192] = %016x\n       Digest[191:128] = %016x\n       Digest[127:64]  = %016x\n       Digest[63:0]    = %016x\n       Resonance = %b\n",
                  32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency,
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[7U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[6U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[5U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[4U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[3U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[2U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[1U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[0U]))),
                  1,(IData)(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] No digest_valid received\n");
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           200);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t = 7U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__name 
        = VL_CVT_PACK_STR_NN(vlSelf->tb_phi_rft_core__DOT__test_names
                             [7U]);
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx = 7U;
    VL_WRITEF("\n--- Test %0d: %@ ---\n",32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx,
              -1,&(__Vtask_tb_phi_rft_core__DOT__run_single_test__0__name));
    if (((((((((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx) 
               | (1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
              | (2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
             | (3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
            | (4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
           | (5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
          | (6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) 
         | (7U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx))) {
        if ((0U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_0
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__1__arr
                   [6U]);
        } else if ((1U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_1
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__2__arr
                   [6U]);
        } else if ((2U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_2
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__3__arr
                   [6U]);
        } else if ((3U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_3
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__4__arr
                   [6U]);
        } else if ((4U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_4
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__5__arr
                   [6U]);
        } else if ((5U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_5
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__6__arr
                   [6U]);
        } else if ((6U == __Vtask_tb_phi_rft_core__DOT__run_single_test__0__test_idx)) {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_6
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__7__arr
                   [6U]);
        } else {
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[0U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [0U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[1U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [1U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[2U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [2U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[3U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [3U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[4U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [4U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[5U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [5U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[6U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [6U];
            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr[7U] 
                = vlSelf->tb_phi_rft_core__DOT__test_input_7
                [7U];
            vlSelf->tb_phi_rft_core__DOT__samples[0U] 
                = (IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                              [2U] 
                                              << 0x10U) 
                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                             [1U]))) 
                            << 0x10U) | (QData)((IData)(
                                                        __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                        [0U]))));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [2U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [1U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [0U]))) 
                              >> 0x20U)));
            vlSelf->tb_phi_rft_core__DOT__samples[1U] 
                = ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                   | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                  [5U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                 [4U]))) 
                                << 0x10U) | (QData)((IData)(
                                                            __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                            [3U])))) 
                      << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[2U] 
                = (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                [5U] 
                                                << 0x10U) 
                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                               [4U]))) 
                              << 0x10U) | (QData)((IData)(
                                                          __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                          [3U])))) 
                    >> 0x10U) | ((IData)(((((QData)((IData)(
                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                              [5U] 
                                                              << 0x10U) 
                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [4U]))) 
                                            << 0x10U) 
                                           | (QData)((IData)(
                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                                                             [3U]))) 
                                          >> 0x20U)) 
                                 << 0x10U));
            vlSelf->tb_phi_rft_core__DOT__samples[3U] 
                = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                    [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__8__arr
                   [6U]);
        }
    } else {
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[0U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [0U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[1U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [1U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[2U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [2U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[3U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [3U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[4U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [4U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[5U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [5U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[6U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [6U];
        __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr[7U] 
            = vlSelf->tb_phi_rft_core__DOT__test_input_0
            [7U];
        vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                                               [0U]))));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffff0000U & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | (IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [2U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [1U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [0U]))) 
                        >> 0x20U)));
        vlSelf->tb_phi_rft_core__DOT__samples[1U] = 
            ((0xffffU & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
             | ((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [5U] << 0x10U) 
                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                           [4U]))) 
                          << 0x10U) | (QData)((IData)(
                                                      __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                      [3U])))) 
                << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[2U] = 
            (((IData)((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                          [5U] << 0x10U) 
                                         | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [4U]))) << 0x10U) 
                       | (QData)((IData)(__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                         [3U])))) >> 0x10U) 
             | ((IData)(((((QData)((IData)(((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                             [5U] << 0x10U) 
                                            | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                            [4U]))) 
                           << 0x10U) | (QData)((IData)(
                                                       __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
                                                       [3U]))) 
                         >> 0x20U)) << 0x10U));
        vlSelf->tb_phi_rft_core__DOT__samples[3U] = 
            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
              [7U] << 0x10U) | __Vtask_tb_phi_rft_core__DOT__pack_samples__9__arr
             [6U]);
    }
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       174);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle 
        = vlSelf->tb_phi_rft_core__DOT__cycle_count;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       177);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__10__count);
    }
    __Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency 
        = (vlSelf->tb_phi_rft_core__DOT__cycle_count 
           - __Vtask_tb_phi_rft_core__DOT__run_single_test__0__start_cycle);
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if (vlSelf->tb_phi_rft_core__DOT__digest_valid) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Digest computed in %0d cycles\n       Digest[255:192] = %016x\n       Digest[191:128] = %016x\n       Digest[127:64]  = %016x\n       Digest[63:0]    = %016x\n       Resonance = %b\n",
                  32,__Vtask_tb_phi_rft_core__DOT__run_single_test__0__latency,
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[7U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[6U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[5U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[4U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[3U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[2U]))),
                  64,(((QData)((IData)(vlSelf->tb_phi_rft_core__DOT__digest[1U])) 
                       << 0x20U) | (QData)((IData)(
                                                   vlSelf->tb_phi_rft_core__DOT__digest[0U]))),
                  1,(IData)(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] No digest_valid received\n");
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           200);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       201);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t = 8U;
    VL_WRITEF("\n--- Back-to-back test ---\n");
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[0U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [0U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[1U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [1U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[2U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [2U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[3U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [3U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[4U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [4U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[5U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [5U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[6U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [6U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[7U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [7U];
    vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                        (((QData)((IData)(
                                                                          ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                            [2U] 
                                                                            << 0x10U) 
                                                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                           [1U]))) 
                                                          << 0x10U) 
                                                         | (QData)((IData)(
                                                                           __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                           [0U]))));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffff0000U 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | (IData)(
                                                           ((((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [0U]))) 
                                                            >> 0x20U)));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffffU 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | ((IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [4U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [3U])))) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[2U] = (((IData)(
                                                          (((QData)((IData)(
                                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                              [5U] 
                                                                              << 0x10U) 
                                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                             [4U]))) 
                                                            << 0x10U) 
                                                           | (QData)((IData)(
                                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                             [3U])))) 
                                                  >> 0x10U) 
                                                 | ((IData)(
                                                            ((((QData)((IData)(
                                                                               ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                                | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [4U]))) 
                                                               << 0x10U) 
                                                              | (QData)((IData)(
                                                                                __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [3U]))) 
                                                             >> 0x20U)) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[3U] = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                  [7U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                 [6U]);
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       236);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       238);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count);
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           241);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    vlSelf->tb_phi_rft_core__DOT__unnamedblk3__DOT__i = 1U;
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[0U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [0U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[1U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [1U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[2U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [2U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[3U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [3U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[4U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [4U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[5U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [5U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[6U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [6U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[7U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [7U];
    vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                        (((QData)((IData)(
                                                                          ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                            [2U] 
                                                                            << 0x10U) 
                                                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                           [1U]))) 
                                                          << 0x10U) 
                                                         | (QData)((IData)(
                                                                           __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                           [0U]))));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffff0000U 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | (IData)(
                                                           ((((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [0U]))) 
                                                            >> 0x20U)));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffffU 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | ((IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [4U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [3U])))) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[2U] = (((IData)(
                                                          (((QData)((IData)(
                                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                              [5U] 
                                                                              << 0x10U) 
                                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                             [4U]))) 
                                                            << 0x10U) 
                                                           | (QData)((IData)(
                                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                             [3U])))) 
                                                  >> 0x10U) 
                                                 | ((IData)(
                                                            ((((QData)((IData)(
                                                                               ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                                | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [4U]))) 
                                                               << 0x10U) 
                                                              | (QData)((IData)(
                                                                                __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [3U]))) 
                                                             >> 0x20U)) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[3U] = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                  [7U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                 [6U]);
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       236);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       238);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count);
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           241);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    vlSelf->tb_phi_rft_core__DOT__unnamedblk3__DOT__i = 2U;
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[0U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [0U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[1U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [1U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[2U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [2U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[3U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [3U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[4U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [4U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[5U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [5U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[6U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [6U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[7U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [7U];
    vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                        (((QData)((IData)(
                                                                          ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                            [2U] 
                                                                            << 0x10U) 
                                                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                           [1U]))) 
                                                          << 0x10U) 
                                                         | (QData)((IData)(
                                                                           __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                           [0U]))));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffff0000U 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | (IData)(
                                                           ((((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [0U]))) 
                                                            >> 0x20U)));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffffU 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | ((IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [4U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [3U])))) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[2U] = (((IData)(
                                                          (((QData)((IData)(
                                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                              [5U] 
                                                                              << 0x10U) 
                                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                             [4U]))) 
                                                            << 0x10U) 
                                                           | (QData)((IData)(
                                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                             [3U])))) 
                                                  >> 0x10U) 
                                                 | ((IData)(
                                                            ((((QData)((IData)(
                                                                               ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                                | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [4U]))) 
                                                               << 0x10U) 
                                                              | (QData)((IData)(
                                                                                __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [3U]))) 
                                                             >> 0x20U)) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[3U] = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                  [7U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                 [6U]);
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       236);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       238);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count);
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           241);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    vlSelf->tb_phi_rft_core__DOT__unnamedblk3__DOT__i = 3U;
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[0U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [0U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[1U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [1U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[2U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [2U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[3U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [3U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[4U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [4U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[5U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [5U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[6U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [6U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr[7U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_0
        [7U];
    vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                        (((QData)((IData)(
                                                                          ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                            [2U] 
                                                                            << 0x10U) 
                                                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                           [1U]))) 
                                                          << 0x10U) 
                                                         | (QData)((IData)(
                                                                           __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                           [0U]))));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffff0000U 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | (IData)(
                                                           ((((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [0U]))) 
                                                            >> 0x20U)));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffffU 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | ((IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [4U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                               [3U])))) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[2U] = (((IData)(
                                                          (((QData)((IData)(
                                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                              [5U] 
                                                                              << 0x10U) 
                                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                             [4U]))) 
                                                            << 0x10U) 
                                                           | (QData)((IData)(
                                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                             [3U])))) 
                                                  >> 0x10U) 
                                                 | ((IData)(
                                                            ((((QData)((IData)(
                                                                               ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                                | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [4U]))) 
                                                               << 0x10U) 
                                                              | (QData)((IData)(
                                                                                __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                                                [3U]))) 
                                                             >> 0x20U)) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[3U] = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                  [7U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__11__arr
                                                 [6U]);
    vlSelf->tb_phi_rft_core__DOT__mode = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       236);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       238);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout = 0x16U;
    __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count = 0U;
    while (((~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid)) 
            & VL_LTS_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout))) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           141);
        vlSelf->__Vm_traceActivity[2U] = 1U;
        __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count 
            = ((IData)(1U) + __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count);
    }
    if (VL_UNLIKELY(VL_GTES_III(32, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count, __Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__timeout))) {
        VL_WRITEF("[ERROR] Timeout waiting for digest_valid after %0d cycles\n",
                  32,__Vtask_tb_phi_rft_core__DOT__wait_digest_valid__12__count);
    }
    while (vlSelf->tb_phi_rft_core__DOT__busy) {
        co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                           nullptr, 
                                                           "@(posedge tb_phi_rft_core.clk)", 
                                                           "tb/tb_phi_rft_core.sv", 
                                                           241);
        vlSelf->__Vm_traceActivity[2U] = 1U;
    }
    vlSelf->tb_phi_rft_core__DOT__unnamedblk3__DOT__i = 4U;
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    vlSelf->tb_phi_rft_core__DOT__pass_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__pass_count);
    VL_WRITEF("[PASS] 4 back-to-back transforms completed\n\n--- Reset recovery test ---\n");
    __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[0U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_1
        [0U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[1U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_1
        [1U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[2U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_1
        [2U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[3U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_1
        [3U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[4U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_1
        [4U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[5U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_1
        [5U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[6U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_1
        [6U];
    __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr[7U] 
        = vlSelf->tb_phi_rft_core__DOT__test_input_1
        [7U];
    vlSelf->tb_phi_rft_core__DOT__samples[0U] = (IData)(
                                                        (((QData)((IData)(
                                                                          ((__Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                            [2U] 
                                                                            << 0x10U) 
                                                                           | __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                           [1U]))) 
                                                          << 0x10U) 
                                                         | (QData)((IData)(
                                                                           __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                           [0U]))));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffff0000U 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | (IData)(
                                                           ((((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                                [2U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                               [1U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                               [0U]))) 
                                                            >> 0x20U)));
    vlSelf->tb_phi_rft_core__DOT__samples[1U] = ((0xffffU 
                                                  & vlSelf->tb_phi_rft_core__DOT__samples[1U]) 
                                                 | ((IData)(
                                                            (((QData)((IData)(
                                                                              ((__Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                               | __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                               [4U]))) 
                                                              << 0x10U) 
                                                             | (QData)((IData)(
                                                                               __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                               [3U])))) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[2U] = (((IData)(
                                                          (((QData)((IData)(
                                                                            ((__Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                              [5U] 
                                                                              << 0x10U) 
                                                                             | __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                             [4U]))) 
                                                            << 0x10U) 
                                                           | (QData)((IData)(
                                                                             __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                             [3U])))) 
                                                  >> 0x10U) 
                                                 | ((IData)(
                                                            ((((QData)((IData)(
                                                                               ((__Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                                [5U] 
                                                                                << 0x10U) 
                                                                                | __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                                [4U]))) 
                                                               << 0x10U) 
                                                              | (QData)((IData)(
                                                                                __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                                                [3U]))) 
                                                             >> 0x20U)) 
                                                    << 0x10U));
    vlSelf->tb_phi_rft_core__DOT__samples[3U] = ((__Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                  [7U] 
                                                  << 0x10U) 
                                                 | __Vtask_tb_phi_rft_core__DOT__pack_samples__13__arr
                                                 [6U]);
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       250);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       252);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__start = 0U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       254);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       254);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       254);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__rst_n = 0U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       256);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       256);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       256);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       256);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       256);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__rst_n = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    co_await vlSelf->__VtrigSched_hfdb76b99__0.trigger(0U, 
                                                       nullptr, 
                                                       "@(posedge tb_phi_rft_core.clk)", 
                                                       "tb/tb_phi_rft_core.sv", 
                                                       258);
    vlSelf->__Vm_traceActivity[2U] = 1U;
    vlSelf->tb_phi_rft_core__DOT__test_count = ((IData)(1U) 
                                                + vlSelf->tb_phi_rft_core__DOT__test_count);
    if ((1U & ((~ (IData)(vlSelf->tb_phi_rft_core__DOT__busy)) 
               & (~ (IData)(vlSelf->tb_phi_rft_core__DOT__digest_valid))))) {
        vlSelf->tb_phi_rft_core__DOT__pass_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__pass_count);
        VL_WRITEF("[PASS] Reset recovery - core returned to idle\n");
    } else {
        vlSelf->tb_phi_rft_core__DOT__fail_count = 
            ((IData)(1U) + vlSelf->tb_phi_rft_core__DOT__fail_count);
        VL_WRITEF("[FAIL] Reset recovery - unexpected state\n");
    }
    VL_WRITEF("\n######################################################\n# TEST SUMMARY                                       #\n######################################################\n  Total Tests: %0d\n  Passed:      %0d\n  Failed:      %0d\n  Cycles:      %0d\n######################################################\n\n",
              32,vlSelf->tb_phi_rft_core__DOT__test_count,
              32,vlSelf->tb_phi_rft_core__DOT__pass_count,
              32,vlSelf->tb_phi_rft_core__DOT__fail_count,
              32,vlSelf->tb_phi_rft_core__DOT__cycle_count);
    if ((0U == vlSelf->tb_phi_rft_core__DOT__fail_count)) {
        VL_WRITEF("*** ALL TESTS PASSED ***\n\n");
        VL_FINISH_MT("tb/tb_phi_rft_core.sv", 280, "");
    } else {
        VL_WRITEF("*** SOME TESTS FAILED ***\n\n");
        VL_FINISH_MT("tb/tb_phi_rft_core.sv", 283, "");
    }
    vlSelf->__Vm_traceActivity[2U] = 1U;
}

VL_INLINE_OPT VlCoroutine Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__1(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__1\n"); );
    // Body
    co_await vlSelf->__VdlySched.delay(0x3b9aca00ULL, 
                                       nullptr, "tb/tb_phi_rft_core.sv", 
                                       291);
    VL_WRITEF("\n[ERROR] Global timeout - simulation exceeded limit\n");
    VL_FINISH_MT("tb/tb_phi_rft_core.sv", 293, "");
}

VL_INLINE_OPT VlCoroutine Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__2(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_initial__TOP__Vtiming__2\n"); );
    // Body
    while (1U) {
        co_await vlSelf->__VdlySched.delay(0x1388ULL, 
                                           nullptr, 
                                           "tb/tb_phi_rft_core.sv", 
                                           105);
        vlSelf->tb_phi_rft_core__DOT__clk = (1U & (~ (IData)(vlSelf->tb_phi_rft_core__DOT__clk)));
    }
}

void Vtb_phi_rft_core___024root___eval_act(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_act\n"); );
}

extern const VlWide<8>/*255:0*/ Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0;

VL_INLINE_OPT void Vtb_phi_rft_core___024root___nba_sequent__TOP__0(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___nba_sequent__TOP__0\n"); );
    // Init
    VlWide<9>/*256:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout;
    VL_ZERO_W(257, __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout);
    VlWide<4>/*127:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector;
    VL_ZERO_W(128, __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector);
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k = 0;
    VlWide<9>/*256:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result;
    VL_ZERO_W(257, __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result);
    VlUnpacked<SData/*15:0*/, 8> __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[__Vi0] = 0;
    }
    VlUnpacked<IData/*31:0*/, 8> __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[__Vi0] = 0;
    }
    VlUnpacked<IData/*31:0*/, 8> __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__total_energy;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__total_energy = 0;
    VlUnpacked<SData/*15:0*/, 8> __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[__Vi0] = 0;
    }
    VlUnpacked<CData/*7:0*/, 8> __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[__Vi0] = 0;
    }
    VlUnpacked<SData/*15:0*/, 8> __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat;
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[__Vi0] = 0;
    }
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real = 0;
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag = 0;
    SData/*15:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r = 0;
    SData/*15:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i = 0;
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real = 0;
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag = 0;
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mag_real;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mag_real = 0;
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mag_imag;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mag_imag = 0;
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amplitude;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amplitude = 0;
    SData/*15:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amp_scaled;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amp_scaled = 0;
    IData/*31:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sq_term;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sq_term = 0;
    VlWide<4>/*127:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo;
    VL_ZERO_W(128, __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo);
    VlWide<4>/*127:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi;
    VL_ZERO_W(128, __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi);
    SData/*15:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout = 0;
    CData/*2:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k = 0;
    CData/*5:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx = 0;
    SData/*15:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout = 0;
    CData/*2:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k = 0;
    CData/*5:0*/ __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx;
    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx = 0;
    CData/*0:0*/ __Vdly__tb_phi_rft_core__DOT__dut__DOT__processing;
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__processing = 0;
    CData/*3:0*/ __Vdly__tb_phi_rft_core__DOT__dut__DOT__latency_cnt;
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__latency_cnt = 0;
    VlWide<9>/*256:0*/ __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result;
    VL_ZERO_W(257, __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result);
    // Body
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[0U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[0U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[1U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[1U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[2U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[2U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[3U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[3U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[4U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[4U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[5U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[5U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[6U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[6U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[7U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[7U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[8U] 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[8U];
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__latency_cnt 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__latency_cnt;
    __Vdly__tb_phi_rft_core__DOT__dut__DOT__processing 
        = vlSelf->tb_phi_rft_core__DOT__dut__DOT__processing;
    if (vlSelf->tb_phi_rft_core__DOT__rst_n) {
        vlSelf->tb_phi_rft_core__DOT__digest_valid = 0U;
        if (((IData)(vlSelf->tb_phi_rft_core__DOT__start) 
             & (~ (IData)(vlSelf->tb_phi_rft_core__DOT__dut__DOT__processing)))) {
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[0U] 
                = vlSelf->tb_phi_rft_core__DOT__samples[0U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[1U] 
                = vlSelf->tb_phi_rft_core__DOT__samples[1U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[2U] 
                = vlSelf->tb_phi_rft_core__DOT__samples[2U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[3U] 
                = vlSelf->tb_phi_rft_core__DOT__samples[3U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__total_energy = 0U;
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__processing = 1U;
            vlSelf->tb_phi_rft_core__DOT__busy = 1U;
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__latency_cnt = 0xbU;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[0U] 
                = (0xffffU & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[0U]);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[0U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[0U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[1U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[0U] 
                   >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[1U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[1U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[2U] 
                = (0xffffU & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[1U]);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[2U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[2U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[3U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[1U] 
                   >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[3U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[3U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[4U] 
                = (0xffffU & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[2U]);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[4U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[4U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[5U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[2U] 
                   >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[5U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[5U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[6U] 
                = (0xffffU & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[3U]);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[6U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[6U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr[7U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__vector[3U] 
                   >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[7U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[7U] = 0U;
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k = 0U;
            while (VL_GTS_III(32, 8U, __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k)) {
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx 
                    = ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k) 
                       << 3U);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2d15U
                                            : 0x22a3U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xf441U : 0xd44dU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xbbfU : 0x2bb3U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf82dU
                                            : 0x250dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd36eU
                                            : 0x19fbU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2c92U
                                            : 0xe605U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? 0x2c8fU : 0xd371U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd449U
                                            : 0x272eU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf44fU
                                            : 0xe95cU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xfc0cU : 0x2d14U)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0x3f4U : 0xd2ecU)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf78U
                                            : 0x2902U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2a86U
                                            : 0x1321U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf088U
                                            : 0xd6feU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)))
                                : 0x2d40U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx 
                    = ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k) 
                       << 3U);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2d15U
                                            : 0x22a3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd44dU : 0xbbfU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2bb3U : 0xf441U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2c92U
                                            : 0xe605U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf82dU
                                            : 0x250dU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd36eU
                                            : 0x19fbU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? 0xf81fU : 0x7e1U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd449U
                                            : 0x272eU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf44fU
                                            : 0xe95cU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd2ecU : 0xfc0cU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2d14U : 0x3f4U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf78U
                                            : 0x2902U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2a86U
                                            : 0x1321U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf088U
                                            : 0xd6feU)))
                                : 0U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [0U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [0U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx 
                    = (1U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2d15U
                                            : 0x22a3U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xf441U : 0xd44dU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xbbfU : 0x2bb3U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf82dU
                                            : 0x250dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd36eU
                                            : 0x19fbU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2c92U
                                            : 0xe605U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? 0x2c8fU : 0xd371U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd449U
                                            : 0x272eU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf44fU
                                            : 0xe95cU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xfc0cU : 0x2d14U)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0x3f4U : 0xd2ecU)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf78U
                                            : 0x2902U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2a86U
                                            : 0x1321U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf088U
                                            : 0xd6feU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)))
                                : 0x2d40U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx 
                    = (1U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2d15U
                                            : 0x22a3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd44dU : 0xbbfU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2bb3U : 0xf441U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2c92U
                                            : 0xe605U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf82dU
                                            : 0x250dU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd36eU
                                            : 0x19fbU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? 0xf81fU : 0x7e1U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd449U
                                            : 0x272eU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf44fU
                                            : 0xe95cU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd2ecU : 0xfc0cU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2d14U : 0x3f4U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf78U
                                            : 0x2902U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2a86U
                                            : 0x1321U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf088U
                                            : 0xd6feU)))
                                : 0U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [1U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [1U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx 
                    = (2U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2d15U
                                            : 0x22a3U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xf441U : 0xd44dU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xbbfU : 0x2bb3U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf82dU
                                            : 0x250dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd36eU
                                            : 0x19fbU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2c92U
                                            : 0xe605U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? 0x2c8fU : 0xd371U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd449U
                                            : 0x272eU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf44fU
                                            : 0xe95cU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xfc0cU : 0x2d14U)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0x3f4U : 0xd2ecU)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf78U
                                            : 0x2902U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2a86U
                                            : 0x1321U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf088U
                                            : 0xd6feU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)))
                                : 0x2d40U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx 
                    = (2U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2d15U
                                            : 0x22a3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd44dU : 0xbbfU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2bb3U : 0xf441U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2c92U
                                            : 0xe605U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf82dU
                                            : 0x250dU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd36eU
                                            : 0x19fbU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? 0xf81fU : 0x7e1U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd449U
                                            : 0x272eU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf44fU
                                            : 0xe95cU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd2ecU : 0xfc0cU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2d14U : 0x3f4U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf78U
                                            : 0x2902U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2a86U
                                            : 0x1321U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf088U
                                            : 0xd6feU)))
                                : 0U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [2U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [2U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx 
                    = (3U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2d15U
                                            : 0x22a3U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xf441U : 0xd44dU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xbbfU : 0x2bb3U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf82dU
                                            : 0x250dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd36eU
                                            : 0x19fbU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2c92U
                                            : 0xe605U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? 0x2c8fU : 0xd371U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd449U
                                            : 0x272eU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf44fU
                                            : 0xe95cU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xfc0cU : 0x2d14U)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0x3f4U : 0xd2ecU)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf78U
                                            : 0x2902U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2a86U
                                            : 0x1321U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf088U
                                            : 0xd6feU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)))
                                : 0x2d40U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx 
                    = (3U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2d15U
                                            : 0x22a3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd44dU : 0xbbfU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2bb3U : 0xf441U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2c92U
                                            : 0xe605U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf82dU
                                            : 0x250dU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd36eU
                                            : 0x19fbU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? 0xf81fU : 0x7e1U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd449U
                                            : 0x272eU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf44fU
                                            : 0xe95cU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd2ecU : 0xfc0cU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2d14U : 0x3f4U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf78U
                                            : 0x2902U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2a86U
                                            : 0x1321U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf088U
                                            : 0xd6feU)))
                                : 0U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [3U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [3U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx 
                    = (4U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2d15U
                                            : 0x22a3U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xf441U : 0xd44dU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xbbfU : 0x2bb3U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf82dU
                                            : 0x250dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd36eU
                                            : 0x19fbU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2c92U
                                            : 0xe605U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? 0x2c8fU : 0xd371U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd449U
                                            : 0x272eU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf44fU
                                            : 0xe95cU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xfc0cU : 0x2d14U)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0x3f4U : 0xd2ecU)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf78U
                                            : 0x2902U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2a86U
                                            : 0x1321U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf088U
                                            : 0xd6feU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)))
                                : 0x2d40U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx 
                    = (4U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2d15U
                                            : 0x22a3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd44dU : 0xbbfU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2bb3U : 0xf441U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2c92U
                                            : 0xe605U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf82dU
                                            : 0x250dU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd36eU
                                            : 0x19fbU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? 0xf81fU : 0x7e1U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd449U
                                            : 0x272eU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf44fU
                                            : 0xe95cU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd2ecU : 0xfc0cU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2d14U : 0x3f4U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf78U
                                            : 0x2902U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2a86U
                                            : 0x1321U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf088U
                                            : 0xd6feU)))
                                : 0U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [4U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [4U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx 
                    = (5U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2d15U
                                            : 0x22a3U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xf441U : 0xd44dU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xbbfU : 0x2bb3U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf82dU
                                            : 0x250dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd36eU
                                            : 0x19fbU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2c92U
                                            : 0xe605U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? 0x2c8fU : 0xd371U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd449U
                                            : 0x272eU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf44fU
                                            : 0xe95cU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xfc0cU : 0x2d14U)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0x3f4U : 0xd2ecU)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf78U
                                            : 0x2902U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2a86U
                                            : 0x1321U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf088U
                                            : 0xd6feU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)))
                                : 0x2d40U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx 
                    = (5U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2d15U
                                            : 0x22a3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd44dU : 0xbbfU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2bb3U : 0xf441U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2c92U
                                            : 0xe605U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf82dU
                                            : 0x250dU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd36eU
                                            : 0x19fbU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? 0xf81fU : 0x7e1U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd449U
                                            : 0x272eU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf44fU
                                            : 0xe95cU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd2ecU : 0xfc0cU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2d14U : 0x3f4U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf78U
                                            : 0x2902U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2a86U
                                            : 0x1321U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf088U
                                            : 0xd6feU)))
                                : 0U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [5U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [5U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx 
                    = (6U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2d15U
                                            : 0x22a3U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xf441U : 0xd44dU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xbbfU : 0x2bb3U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf82dU
                                            : 0x250dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd36eU
                                            : 0x19fbU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2c92U
                                            : 0xe605U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? 0x2c8fU : 0xd371U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd449U
                                            : 0x272eU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf44fU
                                            : 0xe95cU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xfc0cU : 0x2d14U)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0x3f4U : 0xd2ecU)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf78U
                                            : 0x2902U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2a86U
                                            : 0x1321U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf088U
                                            : 0xd6feU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)))
                                : 0x2d40U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx 
                    = (6U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2d15U
                                            : 0x22a3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd44dU : 0xbbfU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2bb3U : 0xf441U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2c92U
                                            : 0xe605U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf82dU
                                            : 0x250dU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd36eU
                                            : 0x19fbU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? 0xf81fU : 0x7e1U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd449U
                                            : 0x272eU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf44fU
                                            : 0xe95cU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd2ecU : 0xfc0cU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2d14U : 0x3f4U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf78U
                                            : 0x2902U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2a86U
                                            : 0x1321U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf088U
                                            : 0xd6feU)))
                                : 0U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [6U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [6U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx 
                    = (7U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2d15U
                                            : 0x22a3U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xf441U : 0xd44dU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xbbfU : 0x2bb3U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf82dU
                                            : 0x250dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd36eU
                                            : 0x19fbU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2c92U
                                            : 0xe605U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? 0x2c8fU : 0xd371U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd449U
                                            : 0x272eU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf44fU
                                            : 0xe95cU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0xfc0cU : 0x2d14U)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? 0x3f4U : 0xd2ecU)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf78U
                                            : 0x2902U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0x2a86U
                                            : 0x1321U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xf088U
                                            : 0xd6feU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)))
                                : 0x2d40U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_real__15__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k 
                    = (7U & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx 
                    = (7U | ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__k) 
                             << 3U));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout 
                    = ((0x20U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                        ? ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2d15U
                                            : 0x22a3U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x3e6U
                                            : 0xe2e1U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd2ebU
                                            : 0xdd5dU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xfc1aU
                                            : 0x1d1fU)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd44dU : 0xbbfU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2bb3U : 0xf441U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2c92U
                                            : 0xe605U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf82dU
                                            : 0x250dU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd36eU
                                            : 0x19fbU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x7d3U
                                            : 0xdaf3U)))
                                : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? 0xf81fU : 0x7e1U)))
                        : ((0x10U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                            ? ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd449U
                                            : 0x272eU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf44fU
                                            : 0xe95cU))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2bb7U
                                            : 0xd8d2U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xbb1U
                                            : 0x16a4U)))
                                : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0xd2ecU : 0xfc0cU)
                                    : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? 0x2d14U : 0x3f4U)))
                            : ((8U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                ? ((4U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                    ? ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xd57aU
                                            : 0xecdfU)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf78U
                                            : 0x2902U))
                                    : ((2U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                        ? ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0x2a86U
                                            : 0x1321U)
                                        : ((1U & (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__idx))
                                            ? 0xf088U
                                            : 0xd6feU)))
                                : 0U)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__kernel_imag__16__Vfuncout;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [7U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_r)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag 
                    = VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                    __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sample_arr
                                                    [7U]), 
                                  VL_EXTENDS_II(32,16, (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__coeff_i)));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_real);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mult_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real[(7U 
                                                                                & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k)] 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag[(7U 
                                                                                & __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k)] 
                    = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag;
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mag_real 
                    = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real 
                        >> 0x1fU) ? (- __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real)
                        : __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_real);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mag_imag 
                    = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag 
                        >> 0x1fU) ? (- __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag)
                        : __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__acc_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amplitude 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mag_real 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__mag_imag);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amp_scaled 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amplitude 
                       >> 0x10U);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sq_term 
                    = ((IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amp_scaled) 
                       * (IData)(__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__amp_scaled));
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__total_energy 
                    = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__total_energy 
                       + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sq_term);
                __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k 
                    = ((IData)(1U) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__unnamedblk2__DOT__k);
            }
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[0U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real
                   [0U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[0U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag
                   [0U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[0U] 
                = (0xffU & (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real
                            [0U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag
                            [0U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[1U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real
                   [1U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[1U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag
                   [1U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[1U] 
                = (0xffU & (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real
                            [1U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag
                            [1U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[2U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real
                   [2U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[2U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag
                   [2U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[2U] 
                = (0xffU & (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real
                            [2U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag
                            [2U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[3U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real
                   [3U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[3U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag
                   [3U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[3U] 
                = (0xffU & (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real
                            [3U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag
                            [3U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[4U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real
                   [4U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[4U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag
                   [4U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[4U] 
                = (0xffU & (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real
                            [4U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag
                            [4U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[5U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real
                   [5U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[5U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag
                   [5U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[5U] 
                = (0xffU & (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real
                            [5U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag
                            [5U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[6U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real
                   [6U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[6U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag
                   [6U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[6U] 
                = (0xffU & (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real
                            [6U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag
                            [6U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real[7U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_real
                   [7U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag[7U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__rft_imag
                   [7U] >> 0x10U);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals[7U] 
                = (0xffU & (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_real
                            [7U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__q_imag
                            [7U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[0U] 
                = (0xffffU & (((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [0U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [1U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                               [2U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                              [3U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[1U] 
                = (0xffffU & (((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [1U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [2U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                               [3U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                              [4U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[2U] 
                = (0xffffU & (((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [2U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [3U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                               [4U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                              [5U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[3U] 
                = (0xffffU & (((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [3U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [4U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                               [5U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                              [6U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[4U] 
                = (0xffffU & (((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [4U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [5U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                               [6U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                              [7U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[5U] 
                = (0xffffU & (((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [5U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [6U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                               [7U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                              [0U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[6U] 
                = (0xffffU & (((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [6U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [7U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                               [0U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                              [1U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat[7U] 
                = (0xffffU & (((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [7U] + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                                [0U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                               [1U]) + __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__s_vals
                              [2U]));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[0U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                    [5U] << 0x10U) | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                   [4U]);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[1U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                    [7U] << 0x10U) | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                   [6U]);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[2U] 
                = (IData)((((QData)((IData)(((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                              [3U] 
                                              << 0x10U) 
                                             | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                             [2U]))) 
                            << 0x20U) | (QData)((IData)(
                                                        ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                                          [1U] 
                                                          << 0x10U) 
                                                         | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                                         [0U])))));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[3U] 
                = (IData)(((((QData)((IData)(((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                               [3U] 
                                               << 0x10U) 
                                              | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                              [2U]))) 
                             << 0x20U) | (QData)((IData)(
                                                         ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                                           [1U] 
                                                           << 0x10U) 
                                                          | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                                          [0U])))) 
                           >> 0x20U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[0U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                    [5U] << 0x10U) | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                   [4U]);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[1U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                    [7U] << 0x10U) | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                   [6U]);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[2U] 
                = (IData)((((QData)((IData)(((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                              [3U] 
                                              << 0x10U) 
                                             | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                             [2U]))) 
                            << 0x20U) | (QData)((IData)(
                                                        ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                                          [1U] 
                                                          << 0x10U) 
                                                         | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                                         [0U])))));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[3U] 
                = (IData)(((((QData)((IData)(((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                               [3U] 
                                               << 0x10U) 
                                              | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                              [2U]))) 
                             << 0x20U) | (QData)((IData)(
                                                         ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                                           [1U] 
                                                           << 0x10U) 
                                                          | __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__lat
                                                          [0U])))) 
                           >> 0x20U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[0U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[0U] 
                    << 1U) | (0x3e8U < __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__total_energy));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[1U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[0U] 
                    >> 0x1fU) | (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[1U] 
                                 << 1U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[2U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[1U] 
                    >> 0x1fU) | (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[2U] 
                                 << 1U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[3U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[2U] 
                    >> 0x1fU) | (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[3U] 
                                 << 1U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[4U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_lo[3U] 
                    >> 0x1fU) | (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[0U] 
                                 << 1U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[5U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[0U] 
                    >> 0x1fU) | (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[1U] 
                                 << 1U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[6U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[1U] 
                    >> 0x1fU) | (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[2U] 
                                 << 1U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[7U] 
                = ((__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[2U] 
                    >> 0x1fU) | (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[3U] 
                                 << 1U));
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[8U] 
                = (__Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__sis_hi[3U] 
                   >> 0x1fU);
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[0U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[0U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[1U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[1U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[2U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[2U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[3U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[3U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[4U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[4U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[5U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[5U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[6U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[6U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[7U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[7U];
            __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[8U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__result[8U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[0U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[0U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[1U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[1U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[2U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[2U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[3U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[3U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[4U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[4U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[5U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[5U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[6U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[6U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[7U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[7U];
            __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[8U] 
                = __Vfunc_tb_phi_rft_core__DOT__dut__DOT__compute_block__14__Vfuncout[8U];
        } else if (vlSelf->tb_phi_rft_core__DOT__dut__DOT__processing) {
            if ((0U == (IData)(vlSelf->tb_phi_rft_core__DOT__dut__DOT__latency_cnt))) {
                vlSelf->tb_phi_rft_core__DOT__digest[0U] 
                    = ((vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[1U] 
                        << 0x1fU) | (vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[0U] 
                                     >> 1U));
                vlSelf->tb_phi_rft_core__DOT__digest[1U] 
                    = ((vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[2U] 
                        << 0x1fU) | (vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[1U] 
                                     >> 1U));
                vlSelf->tb_phi_rft_core__DOT__digest[2U] 
                    = ((vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[3U] 
                        << 0x1fU) | (vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[2U] 
                                     >> 1U));
                vlSelf->tb_phi_rft_core__DOT__digest[3U] 
                    = ((vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[4U] 
                        << 0x1fU) | (vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[3U] 
                                     >> 1U));
                vlSelf->tb_phi_rft_core__DOT__digest[4U] 
                    = ((vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[5U] 
                        << 0x1fU) | (vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[4U] 
                                     >> 1U));
                vlSelf->tb_phi_rft_core__DOT__digest[5U] 
                    = ((vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[6U] 
                        << 0x1fU) | (vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[5U] 
                                     >> 1U));
                vlSelf->tb_phi_rft_core__DOT__digest[6U] 
                    = ((vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[7U] 
                        << 0x1fU) | (vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[6U] 
                                     >> 1U));
                vlSelf->tb_phi_rft_core__DOT__digest[7U] 
                    = ((vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[8U] 
                        << 0x1fU) | (vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[7U] 
                                     >> 1U));
                vlSelf->tb_phi_rft_core__DOT__digest_valid = 1U;
                __Vdly__tb_phi_rft_core__DOT__dut__DOT__processing = 0U;
                vlSelf->tb_phi_rft_core__DOT__busy = 0U;
                vlSelf->tb_phi_rft_core__DOT__resonance_flag 
                    = (1U & vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[0U]);
            } else {
                __Vdly__tb_phi_rft_core__DOT__dut__DOT__latency_cnt 
                    = (0xfU & ((IData)(vlSelf->tb_phi_rft_core__DOT__dut__DOT__latency_cnt) 
                               - (IData)(1U)));
            }
        }
    } else {
        vlSelf->tb_phi_rft_core__DOT__busy = 0U;
        vlSelf->tb_phi_rft_core__DOT__digest_valid = 0U;
        __Vdly__tb_phi_rft_core__DOT__dut__DOT__processing = 0U;
        __Vdly__tb_phi_rft_core__DOT__dut__DOT__latency_cnt = 0U;
        vlSelf->tb_phi_rft_core__DOT__digest[0U] = 
            Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0[0U];
        vlSelf->tb_phi_rft_core__DOT__digest[1U] = 
            Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0[1U];
        vlSelf->tb_phi_rft_core__DOT__digest[2U] = 
            Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0[2U];
        vlSelf->tb_phi_rft_core__DOT__digest[3U] = 
            Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0[3U];
        vlSelf->tb_phi_rft_core__DOT__digest[4U] = 
            Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0[4U];
        vlSelf->tb_phi_rft_core__DOT__digest[5U] = 
            Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0[5U];
        vlSelf->tb_phi_rft_core__DOT__digest[6U] = 
            Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0[6U];
        vlSelf->tb_phi_rft_core__DOT__digest[7U] = 
            Vtb_phi_rft_core__ConstPool__CONST_h9e67c271_0[7U];
        vlSelf->tb_phi_rft_core__DOT__resonance_flag = 0U;
    }
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__processing 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__processing;
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__latency_cnt 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__latency_cnt;
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[0U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[0U];
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[1U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[1U];
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[2U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[2U];
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[3U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[3U];
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[4U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[4U];
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[5U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[5U];
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[6U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[6U];
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[7U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[7U];
    vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result[8U] 
        = __Vdly__tb_phi_rft_core__DOT__dut__DOT__pending_result[8U];
}
