// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_phi_rft_core.h for the primary calling header

#include "Vtb_phi_rft_core__pch.h"
#include "Vtb_phi_rft_core__Syms.h"
#include "Vtb_phi_rft_core___024root.h"

VL_ATTR_COLD void Vtb_phi_rft_core___024root___eval_initial__TOP(Vtb_phi_rft_core___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root___eval_initial__TOP\n"); );
    // Init
    VlWide<5>/*159:0*/ __Vtemp_1;
    // Body
    __Vtemp_1[0U] = 0x2e766364U;
    __Vtemp_1[1U] = 0x636f7265U;
    __Vtemp_1[2U] = 0x7266745fU;
    __Vtemp_1[3U] = 0x7068695fU;
    __Vtemp_1[4U] = 0x74625fU;
    vlSymsp->_vm_contextp__->dumpfile(VL_CVT_PACK_STR_NW(5, __Vtemp_1));
    vlSymsp->_traceDumpOpen();
}
