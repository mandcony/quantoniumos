// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_phi_rft_core.h for the primary calling header

#include "Vtb_phi_rft_core__pch.h"
#include "Vtb_phi_rft_core___024root.h"

VL_INLINE_OPT void Vtb_phi_rft_core___024root___nba_sequent__TOP__1(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___nba_sequent__TOP__1\n"); );
    // Body
    vlSelf->tb_phi_rft_core__DOT__cycle_count = ((IData)(1U) 
                                                 + vlSelf->tb_phi_rft_core__DOT__cycle_count);
}

void Vtb_phi_rft_core___024root___nba_sequent__TOP__0(Vtb_phi_rft_core___024root* vlSelf);

void Vtb_phi_rft_core___024root___eval_nba(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_nba\n"); );
    // Body
    if ((2ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vtb_phi_rft_core___024root___nba_sequent__TOP__0(vlSelf);
        vlSelf->__Vm_traceActivity[3U] = 1U;
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        Vtb_phi_rft_core___024root___nba_sequent__TOP__1(vlSelf);
    }
}

void Vtb_phi_rft_core___024root___timing_resume(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___timing_resume\n"); );
    // Body
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        vlSelf->__VtrigSched_hfdb76b99__0.resume("@(posedge tb_phi_rft_core.clk)");
    }
    if ((4ULL & vlSelf->__VactTriggered.word(0U))) {
        vlSelf->__VdlySched.resume();
    }
}

void Vtb_phi_rft_core___024root___timing_commit(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___timing_commit\n"); );
    // Body
    if ((! (1ULL & vlSelf->__VactTriggered.word(0U)))) {
        vlSelf->__VtrigSched_hfdb76b99__0.commit("@(posedge tb_phi_rft_core.clk)");
    }
}

void Vtb_phi_rft_core___024root___eval_triggers__act(Vtb_phi_rft_core___024root* vlSelf);
void Vtb_phi_rft_core___024root___eval_act(Vtb_phi_rft_core___024root* vlSelf);

bool Vtb_phi_rft_core___024root___eval_phase__act(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<3> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtb_phi_rft_core___024root___eval_triggers__act(vlSelf);
    Vtb_phi_rft_core___024root___timing_commit(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        Vtb_phi_rft_core___024root___timing_resume(vlSelf);
        Vtb_phi_rft_core___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtb_phi_rft_core___024root___eval_phase__nba(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtb_phi_rft_core___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_phi_rft_core___024root___dump_triggers__nba(Vtb_phi_rft_core___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_phi_rft_core___024root___dump_triggers__act(Vtb_phi_rft_core___024root* vlSelf);
#endif  // VL_DEBUG

void Vtb_phi_rft_core___024root___eval(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            Vtb_phi_rft_core___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("tb/tb_phi_rft_core.sv", 12, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                Vtb_phi_rft_core___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("tb/tb_phi_rft_core.sv", 12, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (Vtb_phi_rft_core___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (Vtb_phi_rft_core___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtb_phi_rft_core___024root___eval_debug_assertions(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_debug_assertions\n"); );
}
#endif  // VL_DEBUG
