// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtb_phi_rft_core__Syms.h"


void Vtb_phi_rft_core___024root__trace_chg_0_sub_0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vtb_phi_rft_core___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_chg_0\n"); );
    // Init
    Vtb_phi_rft_core___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtb_phi_rft_core___024root*>(voidSelf);
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtb_phi_rft_core___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vtb_phi_rft_core___024root__trace_chg_0_sub_0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_chg_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[0U])) {
        bufp->chgSData(oldp+0,(vlSelf->tb_phi_rft_core__DOT__test_input_0[0]),16);
        bufp->chgSData(oldp+1,(vlSelf->tb_phi_rft_core__DOT__test_input_0[1]),16);
        bufp->chgSData(oldp+2,(vlSelf->tb_phi_rft_core__DOT__test_input_0[2]),16);
        bufp->chgSData(oldp+3,(vlSelf->tb_phi_rft_core__DOT__test_input_0[3]),16);
        bufp->chgSData(oldp+4,(vlSelf->tb_phi_rft_core__DOT__test_input_0[4]),16);
        bufp->chgSData(oldp+5,(vlSelf->tb_phi_rft_core__DOT__test_input_0[5]),16);
        bufp->chgSData(oldp+6,(vlSelf->tb_phi_rft_core__DOT__test_input_0[6]),16);
        bufp->chgSData(oldp+7,(vlSelf->tb_phi_rft_core__DOT__test_input_0[7]),16);
        bufp->chgSData(oldp+8,(vlSelf->tb_phi_rft_core__DOT__test_input_1[0]),16);
        bufp->chgSData(oldp+9,(vlSelf->tb_phi_rft_core__DOT__test_input_1[1]),16);
        bufp->chgSData(oldp+10,(vlSelf->tb_phi_rft_core__DOT__test_input_1[2]),16);
        bufp->chgSData(oldp+11,(vlSelf->tb_phi_rft_core__DOT__test_input_1[3]),16);
        bufp->chgSData(oldp+12,(vlSelf->tb_phi_rft_core__DOT__test_input_1[4]),16);
        bufp->chgSData(oldp+13,(vlSelf->tb_phi_rft_core__DOT__test_input_1[5]),16);
        bufp->chgSData(oldp+14,(vlSelf->tb_phi_rft_core__DOT__test_input_1[6]),16);
        bufp->chgSData(oldp+15,(vlSelf->tb_phi_rft_core__DOT__test_input_1[7]),16);
        bufp->chgSData(oldp+16,(vlSelf->tb_phi_rft_core__DOT__test_input_2[0]),16);
        bufp->chgSData(oldp+17,(vlSelf->tb_phi_rft_core__DOT__test_input_2[1]),16);
        bufp->chgSData(oldp+18,(vlSelf->tb_phi_rft_core__DOT__test_input_2[2]),16);
        bufp->chgSData(oldp+19,(vlSelf->tb_phi_rft_core__DOT__test_input_2[3]),16);
        bufp->chgSData(oldp+20,(vlSelf->tb_phi_rft_core__DOT__test_input_2[4]),16);
        bufp->chgSData(oldp+21,(vlSelf->tb_phi_rft_core__DOT__test_input_2[5]),16);
        bufp->chgSData(oldp+22,(vlSelf->tb_phi_rft_core__DOT__test_input_2[6]),16);
        bufp->chgSData(oldp+23,(vlSelf->tb_phi_rft_core__DOT__test_input_2[7]),16);
        bufp->chgSData(oldp+24,(vlSelf->tb_phi_rft_core__DOT__test_input_3[0]),16);
        bufp->chgSData(oldp+25,(vlSelf->tb_phi_rft_core__DOT__test_input_3[1]),16);
        bufp->chgSData(oldp+26,(vlSelf->tb_phi_rft_core__DOT__test_input_3[2]),16);
        bufp->chgSData(oldp+27,(vlSelf->tb_phi_rft_core__DOT__test_input_3[3]),16);
        bufp->chgSData(oldp+28,(vlSelf->tb_phi_rft_core__DOT__test_input_3[4]),16);
        bufp->chgSData(oldp+29,(vlSelf->tb_phi_rft_core__DOT__test_input_3[5]),16);
        bufp->chgSData(oldp+30,(vlSelf->tb_phi_rft_core__DOT__test_input_3[6]),16);
        bufp->chgSData(oldp+31,(vlSelf->tb_phi_rft_core__DOT__test_input_3[7]),16);
        bufp->chgSData(oldp+32,(vlSelf->tb_phi_rft_core__DOT__test_input_4[0]),16);
        bufp->chgSData(oldp+33,(vlSelf->tb_phi_rft_core__DOT__test_input_4[1]),16);
        bufp->chgSData(oldp+34,(vlSelf->tb_phi_rft_core__DOT__test_input_4[2]),16);
        bufp->chgSData(oldp+35,(vlSelf->tb_phi_rft_core__DOT__test_input_4[3]),16);
        bufp->chgSData(oldp+36,(vlSelf->tb_phi_rft_core__DOT__test_input_4[4]),16);
        bufp->chgSData(oldp+37,(vlSelf->tb_phi_rft_core__DOT__test_input_4[5]),16);
        bufp->chgSData(oldp+38,(vlSelf->tb_phi_rft_core__DOT__test_input_4[6]),16);
        bufp->chgSData(oldp+39,(vlSelf->tb_phi_rft_core__DOT__test_input_4[7]),16);
        bufp->chgSData(oldp+40,(vlSelf->tb_phi_rft_core__DOT__test_input_5[0]),16);
        bufp->chgSData(oldp+41,(vlSelf->tb_phi_rft_core__DOT__test_input_5[1]),16);
        bufp->chgSData(oldp+42,(vlSelf->tb_phi_rft_core__DOT__test_input_5[2]),16);
        bufp->chgSData(oldp+43,(vlSelf->tb_phi_rft_core__DOT__test_input_5[3]),16);
        bufp->chgSData(oldp+44,(vlSelf->tb_phi_rft_core__DOT__test_input_5[4]),16);
        bufp->chgSData(oldp+45,(vlSelf->tb_phi_rft_core__DOT__test_input_5[5]),16);
        bufp->chgSData(oldp+46,(vlSelf->tb_phi_rft_core__DOT__test_input_5[6]),16);
        bufp->chgSData(oldp+47,(vlSelf->tb_phi_rft_core__DOT__test_input_5[7]),16);
        bufp->chgSData(oldp+48,(vlSelf->tb_phi_rft_core__DOT__test_input_6[0]),16);
        bufp->chgSData(oldp+49,(vlSelf->tb_phi_rft_core__DOT__test_input_6[1]),16);
        bufp->chgSData(oldp+50,(vlSelf->tb_phi_rft_core__DOT__test_input_6[2]),16);
        bufp->chgSData(oldp+51,(vlSelf->tb_phi_rft_core__DOT__test_input_6[3]),16);
        bufp->chgSData(oldp+52,(vlSelf->tb_phi_rft_core__DOT__test_input_6[4]),16);
        bufp->chgSData(oldp+53,(vlSelf->tb_phi_rft_core__DOT__test_input_6[5]),16);
        bufp->chgSData(oldp+54,(vlSelf->tb_phi_rft_core__DOT__test_input_6[6]),16);
        bufp->chgSData(oldp+55,(vlSelf->tb_phi_rft_core__DOT__test_input_6[7]),16);
        bufp->chgSData(oldp+56,(vlSelf->tb_phi_rft_core__DOT__test_input_7[0]),16);
        bufp->chgSData(oldp+57,(vlSelf->tb_phi_rft_core__DOT__test_input_7[1]),16);
        bufp->chgSData(oldp+58,(vlSelf->tb_phi_rft_core__DOT__test_input_7[2]),16);
        bufp->chgSData(oldp+59,(vlSelf->tb_phi_rft_core__DOT__test_input_7[3]),16);
        bufp->chgSData(oldp+60,(vlSelf->tb_phi_rft_core__DOT__test_input_7[4]),16);
        bufp->chgSData(oldp+61,(vlSelf->tb_phi_rft_core__DOT__test_input_7[5]),16);
        bufp->chgSData(oldp+62,(vlSelf->tb_phi_rft_core__DOT__test_input_7[6]),16);
        bufp->chgSData(oldp+63,(vlSelf->tb_phi_rft_core__DOT__test_input_7[7]),16);
    }
    if (VL_UNLIKELY((vlSelf->__Vm_traceActivity[1U] 
                     | vlSelf->__Vm_traceActivity[2U]))) {
        bufp->chgBit(oldp+64,(vlSelf->tb_phi_rft_core__DOT__rst_n));
        bufp->chgBit(oldp+65,(vlSelf->tb_phi_rft_core__DOT__start));
        bufp->chgWData(oldp+66,(vlSelf->tb_phi_rft_core__DOT__samples),128);
        bufp->chgCData(oldp+70,(vlSelf->tb_phi_rft_core__DOT__mode),4);
        bufp->chgIData(oldp+71,(vlSelf->tb_phi_rft_core__DOT__test_count),32);
        bufp->chgIData(oldp+72,(vlSelf->tb_phi_rft_core__DOT__pass_count),32);
        bufp->chgIData(oldp+73,(vlSelf->tb_phi_rft_core__DOT__fail_count),32);
        bufp->chgIData(oldp+74,(vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t),32);
        bufp->chgIData(oldp+75,(vlSelf->tb_phi_rft_core__DOT__unnamedblk3__DOT__i),32);
    }
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[3U])) {
        bufp->chgBit(oldp+76,(vlSelf->tb_phi_rft_core__DOT__busy));
        bufp->chgBit(oldp+77,(vlSelf->tb_phi_rft_core__DOT__digest_valid));
        bufp->chgWData(oldp+78,(vlSelf->tb_phi_rft_core__DOT__digest),256);
        bufp->chgBit(oldp+86,(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
        bufp->chgCData(oldp+87,(vlSelf->tb_phi_rft_core__DOT__dut__DOT__latency_cnt),4);
        bufp->chgBit(oldp+88,(vlSelf->tb_phi_rft_core__DOT__dut__DOT__processing));
        bufp->chgWData(oldp+89,(vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result),257);
    }
    bufp->chgBit(oldp+98,(vlSelf->tb_phi_rft_core__DOT__clk));
    bufp->chgIData(oldp+99,(vlSelf->tb_phi_rft_core__DOT__cycle_count),32);
}

void Vtb_phi_rft_core___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_cleanup\n"); );
    // Init
    Vtb_phi_rft_core___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtb_phi_rft_core___024root*>(voidSelf);
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[2U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[3U] = 0U;
}
