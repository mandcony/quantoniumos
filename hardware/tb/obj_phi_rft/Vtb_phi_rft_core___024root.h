// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_phi_rft_core.h for the primary calling header

#ifndef VERILATED_VTB_PHI_RFT_CORE___024ROOT_H_
#define VERILATED_VTB_PHI_RFT_CORE___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_phi_rft_core__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_phi_rft_core___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ tb_phi_rft_core__DOT__clk;
    CData/*0:0*/ tb_phi_rft_core__DOT__rst_n;
    CData/*0:0*/ tb_phi_rft_core__DOT__start;
    CData/*3:0*/ tb_phi_rft_core__DOT__mode;
    CData/*0:0*/ tb_phi_rft_core__DOT__busy;
    CData/*0:0*/ tb_phi_rft_core__DOT__digest_valid;
    CData/*0:0*/ tb_phi_rft_core__DOT__resonance_flag;
    CData/*3:0*/ tb_phi_rft_core__DOT__dut__DOT__latency_cnt;
    CData/*0:0*/ tb_phi_rft_core__DOT__dut__DOT__processing;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_phi_rft_core__DOT__rst_n__0;
    CData/*0:0*/ __VactContinue;
    VlWide<4>/*127:0*/ tb_phi_rft_core__DOT__samples;
    VlWide<8>/*255:0*/ tb_phi_rft_core__DOT__digest;
    IData/*31:0*/ tb_phi_rft_core__DOT__test_count;
    IData/*31:0*/ tb_phi_rft_core__DOT__pass_count;
    IData/*31:0*/ tb_phi_rft_core__DOT__fail_count;
    IData/*31:0*/ tb_phi_rft_core__DOT__cycle_count;
    IData/*31:0*/ tb_phi_rft_core__DOT__unnamedblk2__DOT__t;
    IData/*31:0*/ tb_phi_rft_core__DOT__unnamedblk3__DOT__i;
    IData/*31:0*/ __VactIterCount;
    VlWide<9>/*256:0*/ tb_phi_rft_core__DOT__dut__DOT__pending_result;
    VlUnpacked<SData/*15:0*/, 8> tb_phi_rft_core__DOT__test_input_0;
    VlUnpacked<SData/*15:0*/, 8> tb_phi_rft_core__DOT__test_input_1;
    VlUnpacked<SData/*15:0*/, 8> tb_phi_rft_core__DOT__test_input_2;
    VlUnpacked<SData/*15:0*/, 8> tb_phi_rft_core__DOT__test_input_3;
    VlUnpacked<SData/*15:0*/, 8> tb_phi_rft_core__DOT__test_input_4;
    VlUnpacked<SData/*15:0*/, 8> tb_phi_rft_core__DOT__test_input_5;
    VlUnpacked<SData/*15:0*/, 8> tb_phi_rft_core__DOT__test_input_6;
    VlUnpacked<SData/*15:0*/, 8> tb_phi_rft_core__DOT__test_input_7;
    VlUnpacked<CData/*0:0*/, 4> __Vm_traceActivity;
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_hfdb76b99__0;
    VlTriggerVec<3> __VactTriggered;
    VlTriggerVec<3> __VnbaTriggered;
    VlUnpacked<std::string, 8> tb_phi_rft_core__DOT__test_names;

    // INTERNAL VARIABLES
    Vtb_phi_rft_core__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtb_phi_rft_core___024root(Vtb_phi_rft_core__Syms* symsp, const char* v__name);
    ~Vtb_phi_rft_core___024root();
    VL_UNCOPYABLE(Vtb_phi_rft_core___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
