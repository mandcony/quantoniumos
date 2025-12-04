// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_phi_rft_core.h for the primary calling header

#include "Vtb_phi_rft_core__pch.h"
#include "Vtb_phi_rft_core__Syms.h"
#include "Vtb_phi_rft_core___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_phi_rft_core___024root___dump_triggers__act(Vtb_phi_rft_core___024root* vlSelf);
#endif  // VL_DEBUG

void Vtb_phi_rft_core___024root___eval_triggers__act(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_triggers__act\n"); );
    // Body
    vlSelf->__VactTriggered.set(0U, ((IData)(vlSelf->tb_phi_rft_core__DOT__clk) 
                                     & (~ (IData)(vlSelf->__Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__clk__0))));
    vlSelf->__VactTriggered.set(1U, (((IData)(vlSelf->tb_phi_rft_core__DOT__clk) 
                                      & (~ (IData)(vlSelf->__Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__clk__0))) 
                                     | ((~ (IData)(vlSelf->tb_phi_rft_core__DOT__rst_n)) 
                                        & (IData)(vlSelf->__Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__rst_n__0))));
    vlSelf->__VactTriggered.set(2U, vlSelf->__VdlySched.awaitingCurrentTime());
    vlSelf->__Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__clk__0 
        = vlSelf->tb_phi_rft_core__DOT__clk;
    vlSelf->__Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__rst_n__0 
        = vlSelf->tb_phi_rft_core__DOT__rst_n;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_phi_rft_core___024root___dump_triggers__act(vlSelf);
    }
#endif
}
