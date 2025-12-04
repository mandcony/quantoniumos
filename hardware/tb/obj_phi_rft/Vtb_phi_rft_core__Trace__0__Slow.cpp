// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtb_phi_rft_core__Syms.h"


VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_init_sub__TOP__rpu_pkg__0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd* tracep);

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_init_sub__TOP__0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_init_sub__TOP__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->pushPrefix("rpu_pkg", VerilatedTracePrefixType::SCOPE_MODULE);
    Vtb_phi_rft_core___024root__trace_init_sub__TOP__rpu_pkg__0(vlSelf, tracep);
    tracep->popPrefix();
    tracep->pushPrefix("tb_phi_rft_core", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+101,0,"CLK_PERIOD",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+102,0,"SAMPLE_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+103,0,"BLOCK_SAMPLES",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+104,0,"DIGEST_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+105,0,"CORE_LATENCY",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+103,0,"NUM_TESTS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->pushPrefix("test_input_0", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("test_input_1", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+9+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("test_input_2", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+17+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("test_input_3", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+25+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("test_input_4", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+33+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("test_input_5", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+41+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("test_input_6", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+49+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("test_input_7", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 8; ++i) {
        tracep->declBus(c+57+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBit(c+99,0,"clk",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+65,0,"rst_n",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+66,0,"start",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+67,0,"samples",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 127,0);
    tracep->declBus(c+71,0,"mode",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBit(c+77,0,"busy",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+78,0,"digest_valid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+79,0,"digest",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 255,0);
    tracep->declBit(c+87,0,"resonance_flag",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+72,0,"test_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+73,0,"pass_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+74,0,"fail_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+100,0,"cycle_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+106,0,"current_test",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->pushPrefix("dut", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+102,0,"SAMPLE_WIDTH_P",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+103,0,"BLOCK_SAMPLES_P",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+104,0,"DIGEST_WIDTH_P",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+105,0,"CORE_LATENCY",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBit(c+99,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+65,0,"rst_n",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+66,0,"start",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+67,0,"samples",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 127,0);
    tracep->declBus(c+71,0,"mode",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBit(c+77,0,"busy",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+78,0,"digest_valid",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+79,0,"digest",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 255,0);
    tracep->declBit(c+87,0,"resonance_flag",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+107,0,"LAT_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+88,0,"latency_cnt",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 3,0);
    tracep->declBit(c+89,0,"processing",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declArray(c+90,0,"pending_result",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 256,0);
    tracep->popPrefix();
    tracep->pushPrefix("unnamedblk2", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+75,0,"t",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("unnamedblk3", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+76,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_init_sub__TOP__rpu_pkg__0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_init_sub__TOP__rpu_pkg__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBus(c+102,0,"SAMPLE_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+103,0,"BLOCK_SAMPLES",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+104,0,"DIGEST_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+108,0,"SAMPLE_FRAME_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+103,0,"TILE_DIM",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+109,0,"TILE_COUNT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+109,0,"SCRATCH_DEPTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+109,0,"TOPO_MEM_DEPTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+108,0,"CTRL_PAYLOAD_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+110,0,"DMA_TILE_BITS",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+111,0,"DMA_PAYLOAD_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+109,0,"MAX_INFLIGHT",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+112,0,"HOP_LATENCY",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::INT, false,-1, 31,0);
}

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_init_top(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_init_top\n"); );
    // Body
    Vtb_phi_rft_core___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtb_phi_rft_core___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtb_phi_rft_core___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_register(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_register\n"); );
    // Body
    tracep->addConstCb(&Vtb_phi_rft_core___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vtb_phi_rft_core___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vtb_phi_rft_core___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vtb_phi_rft_core___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_const_0_sub_0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_const_0\n"); );
    // Init
    Vtb_phi_rft_core___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtb_phi_rft_core___024root*>(voidSelf);
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtb_phi_rft_core___024root__trace_const_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_const_0_sub_0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_const_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullIData(oldp+101,(0xaU),32);
    bufp->fullIData(oldp+102,(0x10U),32);
    bufp->fullIData(oldp+103,(8U),32);
    bufp->fullIData(oldp+104,(0x100U),32);
    bufp->fullIData(oldp+105,(0xcU),32);
    bufp->fullIData(oldp+106,(0U),32);
    bufp->fullIData(oldp+107,(4U),32);
    bufp->fullIData(oldp+108,(0x80U),32);
    bufp->fullIData(oldp+109,(0x40U),32);
    bufp->fullIData(oldp+110,(7U),32);
    bufp->fullIData(oldp+111,(0x97U),32);
    bufp->fullIData(oldp+112,(2U),32);
}

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_full_0_sub_0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_full_0\n"); );
    // Init
    Vtb_phi_rft_core___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtb_phi_rft_core___024root*>(voidSelf);
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtb_phi_rft_core___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtb_phi_rft_core___024root__trace_full_0_sub_0(Vtb_phi_rft_core___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vtb_phi_rft_core__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_phi_rft_core___024root__trace_full_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullSData(oldp+1,(vlSelf->tb_phi_rft_core__DOT__test_input_0[0]),16);
    bufp->fullSData(oldp+2,(vlSelf->tb_phi_rft_core__DOT__test_input_0[1]),16);
    bufp->fullSData(oldp+3,(vlSelf->tb_phi_rft_core__DOT__test_input_0[2]),16);
    bufp->fullSData(oldp+4,(vlSelf->tb_phi_rft_core__DOT__test_input_0[3]),16);
    bufp->fullSData(oldp+5,(vlSelf->tb_phi_rft_core__DOT__test_input_0[4]),16);
    bufp->fullSData(oldp+6,(vlSelf->tb_phi_rft_core__DOT__test_input_0[5]),16);
    bufp->fullSData(oldp+7,(vlSelf->tb_phi_rft_core__DOT__test_input_0[6]),16);
    bufp->fullSData(oldp+8,(vlSelf->tb_phi_rft_core__DOT__test_input_0[7]),16);
    bufp->fullSData(oldp+9,(vlSelf->tb_phi_rft_core__DOT__test_input_1[0]),16);
    bufp->fullSData(oldp+10,(vlSelf->tb_phi_rft_core__DOT__test_input_1[1]),16);
    bufp->fullSData(oldp+11,(vlSelf->tb_phi_rft_core__DOT__test_input_1[2]),16);
    bufp->fullSData(oldp+12,(vlSelf->tb_phi_rft_core__DOT__test_input_1[3]),16);
    bufp->fullSData(oldp+13,(vlSelf->tb_phi_rft_core__DOT__test_input_1[4]),16);
    bufp->fullSData(oldp+14,(vlSelf->tb_phi_rft_core__DOT__test_input_1[5]),16);
    bufp->fullSData(oldp+15,(vlSelf->tb_phi_rft_core__DOT__test_input_1[6]),16);
    bufp->fullSData(oldp+16,(vlSelf->tb_phi_rft_core__DOT__test_input_1[7]),16);
    bufp->fullSData(oldp+17,(vlSelf->tb_phi_rft_core__DOT__test_input_2[0]),16);
    bufp->fullSData(oldp+18,(vlSelf->tb_phi_rft_core__DOT__test_input_2[1]),16);
    bufp->fullSData(oldp+19,(vlSelf->tb_phi_rft_core__DOT__test_input_2[2]),16);
    bufp->fullSData(oldp+20,(vlSelf->tb_phi_rft_core__DOT__test_input_2[3]),16);
    bufp->fullSData(oldp+21,(vlSelf->tb_phi_rft_core__DOT__test_input_2[4]),16);
    bufp->fullSData(oldp+22,(vlSelf->tb_phi_rft_core__DOT__test_input_2[5]),16);
    bufp->fullSData(oldp+23,(vlSelf->tb_phi_rft_core__DOT__test_input_2[6]),16);
    bufp->fullSData(oldp+24,(vlSelf->tb_phi_rft_core__DOT__test_input_2[7]),16);
    bufp->fullSData(oldp+25,(vlSelf->tb_phi_rft_core__DOT__test_input_3[0]),16);
    bufp->fullSData(oldp+26,(vlSelf->tb_phi_rft_core__DOT__test_input_3[1]),16);
    bufp->fullSData(oldp+27,(vlSelf->tb_phi_rft_core__DOT__test_input_3[2]),16);
    bufp->fullSData(oldp+28,(vlSelf->tb_phi_rft_core__DOT__test_input_3[3]),16);
    bufp->fullSData(oldp+29,(vlSelf->tb_phi_rft_core__DOT__test_input_3[4]),16);
    bufp->fullSData(oldp+30,(vlSelf->tb_phi_rft_core__DOT__test_input_3[5]),16);
    bufp->fullSData(oldp+31,(vlSelf->tb_phi_rft_core__DOT__test_input_3[6]),16);
    bufp->fullSData(oldp+32,(vlSelf->tb_phi_rft_core__DOT__test_input_3[7]),16);
    bufp->fullSData(oldp+33,(vlSelf->tb_phi_rft_core__DOT__test_input_4[0]),16);
    bufp->fullSData(oldp+34,(vlSelf->tb_phi_rft_core__DOT__test_input_4[1]),16);
    bufp->fullSData(oldp+35,(vlSelf->tb_phi_rft_core__DOT__test_input_4[2]),16);
    bufp->fullSData(oldp+36,(vlSelf->tb_phi_rft_core__DOT__test_input_4[3]),16);
    bufp->fullSData(oldp+37,(vlSelf->tb_phi_rft_core__DOT__test_input_4[4]),16);
    bufp->fullSData(oldp+38,(vlSelf->tb_phi_rft_core__DOT__test_input_4[5]),16);
    bufp->fullSData(oldp+39,(vlSelf->tb_phi_rft_core__DOT__test_input_4[6]),16);
    bufp->fullSData(oldp+40,(vlSelf->tb_phi_rft_core__DOT__test_input_4[7]),16);
    bufp->fullSData(oldp+41,(vlSelf->tb_phi_rft_core__DOT__test_input_5[0]),16);
    bufp->fullSData(oldp+42,(vlSelf->tb_phi_rft_core__DOT__test_input_5[1]),16);
    bufp->fullSData(oldp+43,(vlSelf->tb_phi_rft_core__DOT__test_input_5[2]),16);
    bufp->fullSData(oldp+44,(vlSelf->tb_phi_rft_core__DOT__test_input_5[3]),16);
    bufp->fullSData(oldp+45,(vlSelf->tb_phi_rft_core__DOT__test_input_5[4]),16);
    bufp->fullSData(oldp+46,(vlSelf->tb_phi_rft_core__DOT__test_input_5[5]),16);
    bufp->fullSData(oldp+47,(vlSelf->tb_phi_rft_core__DOT__test_input_5[6]),16);
    bufp->fullSData(oldp+48,(vlSelf->tb_phi_rft_core__DOT__test_input_5[7]),16);
    bufp->fullSData(oldp+49,(vlSelf->tb_phi_rft_core__DOT__test_input_6[0]),16);
    bufp->fullSData(oldp+50,(vlSelf->tb_phi_rft_core__DOT__test_input_6[1]),16);
    bufp->fullSData(oldp+51,(vlSelf->tb_phi_rft_core__DOT__test_input_6[2]),16);
    bufp->fullSData(oldp+52,(vlSelf->tb_phi_rft_core__DOT__test_input_6[3]),16);
    bufp->fullSData(oldp+53,(vlSelf->tb_phi_rft_core__DOT__test_input_6[4]),16);
    bufp->fullSData(oldp+54,(vlSelf->tb_phi_rft_core__DOT__test_input_6[5]),16);
    bufp->fullSData(oldp+55,(vlSelf->tb_phi_rft_core__DOT__test_input_6[6]),16);
    bufp->fullSData(oldp+56,(vlSelf->tb_phi_rft_core__DOT__test_input_6[7]),16);
    bufp->fullSData(oldp+57,(vlSelf->tb_phi_rft_core__DOT__test_input_7[0]),16);
    bufp->fullSData(oldp+58,(vlSelf->tb_phi_rft_core__DOT__test_input_7[1]),16);
    bufp->fullSData(oldp+59,(vlSelf->tb_phi_rft_core__DOT__test_input_7[2]),16);
    bufp->fullSData(oldp+60,(vlSelf->tb_phi_rft_core__DOT__test_input_7[3]),16);
    bufp->fullSData(oldp+61,(vlSelf->tb_phi_rft_core__DOT__test_input_7[4]),16);
    bufp->fullSData(oldp+62,(vlSelf->tb_phi_rft_core__DOT__test_input_7[5]),16);
    bufp->fullSData(oldp+63,(vlSelf->tb_phi_rft_core__DOT__test_input_7[6]),16);
    bufp->fullSData(oldp+64,(vlSelf->tb_phi_rft_core__DOT__test_input_7[7]),16);
    bufp->fullBit(oldp+65,(vlSelf->tb_phi_rft_core__DOT__rst_n));
    bufp->fullBit(oldp+66,(vlSelf->tb_phi_rft_core__DOT__start));
    bufp->fullWData(oldp+67,(vlSelf->tb_phi_rft_core__DOT__samples),128);
    bufp->fullCData(oldp+71,(vlSelf->tb_phi_rft_core__DOT__mode),4);
    bufp->fullIData(oldp+72,(vlSelf->tb_phi_rft_core__DOT__test_count),32);
    bufp->fullIData(oldp+73,(vlSelf->tb_phi_rft_core__DOT__pass_count),32);
    bufp->fullIData(oldp+74,(vlSelf->tb_phi_rft_core__DOT__fail_count),32);
    bufp->fullIData(oldp+75,(vlSelf->tb_phi_rft_core__DOT__unnamedblk2__DOT__t),32);
    bufp->fullIData(oldp+76,(vlSelf->tb_phi_rft_core__DOT__unnamedblk3__DOT__i),32);
    bufp->fullBit(oldp+77,(vlSelf->tb_phi_rft_core__DOT__busy));
    bufp->fullBit(oldp+78,(vlSelf->tb_phi_rft_core__DOT__digest_valid));
    bufp->fullWData(oldp+79,(vlSelf->tb_phi_rft_core__DOT__digest),256);
    bufp->fullBit(oldp+87,(vlSelf->tb_phi_rft_core__DOT__resonance_flag));
    bufp->fullCData(oldp+88,(vlSelf->tb_phi_rft_core__DOT__dut__DOT__latency_cnt),4);
    bufp->fullBit(oldp+89,(vlSelf->tb_phi_rft_core__DOT__dut__DOT__processing));
    bufp->fullWData(oldp+90,(vlSelf->tb_phi_rft_core__DOT__dut__DOT__pending_result),257);
    bufp->fullBit(oldp+99,(vlSelf->tb_phi_rft_core__DOT__clk));
    bufp->fullIData(oldp+100,(vlSelf->tb_phi_rft_core__DOT__cycle_count),32);
}
