// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
// 
// RFTPU 4x4 Variant for Physical Design
// ======================================
// Reduced from 8x8 (64 tiles) to 4x4 (16 tiles) for:
// - Faster OpenLane place & route
// - Realistic chip layout viewing
// - Physical design validation
//
// Target: SkyWater SKY130 PDK
// Area estimate: ~2.25 mm² @ 130nm

`timescale 1ns/1ps

package rftpu_pkg;
   localparam int SAMPLE_WIDTH        = 16;
   localparam int BLOCK_SAMPLES       = 8;
   localparam int DIGEST_WIDTH        = 256;
   localparam int SAMPLE_FRAME_WIDTH  = SAMPLE_WIDTH * BLOCK_SAMPLES;
   localparam int TILE_DIM            = 4;  // Reduced for physical design
   localparam int TILE_COUNT          = TILE_DIM * TILE_DIM;  // 4x4 = 16 tiles
   localparam int SCRATCH_DEPTH       = 32;  // Reduced for area
   localparam int TOPO_MEM_DEPTH      = 32;  // Reduced for area
   localparam int CTRL_PAYLOAD_WIDTH  = 128;
   localparam int DMA_TILE_BITS       = 4;  // log2(16) = 4
   localparam int DMA_PAYLOAD_WIDTH   = SAMPLE_FRAME_WIDTH + 16 + DMA_TILE_BITS;
   localparam int MAX_INFLIGHT        = 16;  // Reduced for 4x4
   localparam int HOP_LATENCY         = 2;

   typedef struct packed {
      logic        start;
      logic [3:0]  mode;
      logic [15:0] length;
      logic [31:0] in_addr;
      logic [31:0] out_addr;
      logic        cascade_enable;
      logic [2:0]  cascade_dest_x;
      logic [2:0]  cascade_dest_y;
      logic        h3_enable;
      logic [2:0]  h3_slot0_dest_x;
      logic [2:0]  h3_slot0_dest_y;
      logic [5:0]  h3_slot0_vertex;
      logic [2:0]  h3_slot1_dest_x;
      logic [2:0]  h3_slot1_dest_y;
      logic [5:0]  h3_slot1_vertex;
      logic [7:0]  vertex_base_id;
   } tile_ctrl_frame_t;

   typedef struct packed {
      logic busy;
      logic done;
      logic error;
      logic resonance_active;
      logic [15:0] remaining;
      logic [3:0]  mode;
      logic [7:0]  last_vertex_id;
   } tile_status_t;

   typedef struct packed {
      logic [DIGEST_WIDTH-1:0] digest;
      logic [5:0]  vertex_id;
      logic [3:0]  mode;
      logic [2:0]  src_x;
      logic [2:0]  src_y;
      logic [2:0]  dest_x;
      logic [2:0]  dest_y;
      logic [1:0]  pkt_type;   // 0=digest, 1=data, 2=control
   } noc_payload_t;

   typedef struct packed {
      logic [SAMPLE_FRAME_WIDTH-1:0] samples;
      logic [15:0]                   block_idx;
      logic [DMA_TILE_BITS-1:0]      tile_id;
   } dma_frame_t;

   function automatic logic [CTRL_PAYLOAD_WIDTH-1:0]
      make_ctrl_payload(
         input logic        start,
         input logic [3:0]  mode,
         input logic [15:0] length,
         input logic [31:0] in_addr,
         input logic [31:0] out_addr,
         input logic        cascade_enable,
         input logic [2:0]  cascade_dest_x,
         input logic [2:0]  cascade_dest_y,
         input logic        h3_enable,
         input logic [2:0]  h3_slot0_dest_x,
         input logic [2:0]  h3_slot0_dest_y,
         input logic [5:0]  h3_slot0_vertex,
         input logic [2:0]  h3_slot1_dest_x,
         input logic [2:0]  h3_slot1_dest_y,
         input logic [5:0]  h3_slot1_vertex,
         input logic [7:0]  vertex_base_id
      );
      logic [CTRL_PAYLOAD_WIDTH-1:0] payload;
      payload              = '0;
      payload[0]           = start;
      payload[4:1]         = mode;
      payload[20:5]        = length;
      payload[52:21]       = in_addr;
      payload[84:53]       = out_addr;
      payload[85]          = cascade_enable;
      payload[88:86]       = cascade_dest_x;
      payload[91:89]       = cascade_dest_y;
      payload[92]          = h3_enable;
      payload[95:93]       = h3_slot0_dest_x;
      payload[98:96]       = h3_slot0_dest_y;
      payload[104:99]      = h3_slot0_vertex;
      payload[107:105]     = h3_slot1_dest_x;
      payload[110:108]     = h3_slot1_dest_y;
      payload[116:111]     = h3_slot1_vertex;
      payload[124:117]     = vertex_base_id;
      return payload;
   endfunction

   function automatic logic [SAMPLE_FRAME_WIDTH-1:0]
      make_sample_frame(input logic [15:0] seed);
      logic [SAMPLE_FRAME_WIDTH-1:0] frame;
      frame = '0;
      for (int i = 0; i < BLOCK_SAMPLES; i++) begin
         frame[i*SAMPLE_WIDTH +: SAMPLE_WIDTH] = seed + i[15:0];
      end
      return frame;
   endfunction

   function automatic logic [DMA_PAYLOAD_WIDTH-1:0]
      make_dma_payload(
         input logic [SAMPLE_FRAME_WIDTH-1:0] samples,
         input logic [15:0]                   block_idx,
         input logic [DMA_TILE_BITS-1:0]      tile_id
      );
      return {samples, block_idx, tile_id};
   endfunction
endpackage : rftpu_pkg;

module phi_rft_core #(
   parameter int SAMPLE_WIDTH_P  = rftpu_pkg::SAMPLE_WIDTH,
   parameter int BLOCK_SAMPLES_P = rftpu_pkg::BLOCK_SAMPLES,
   parameter int DIGEST_WIDTH_P  = rftpu_pkg::DIGEST_WIDTH,
   parameter int CORE_LATENCY    = 12
   ) (
      input  logic                               clk,
      input  logic                               rst_n,
      input  logic                               start,
      input  logic [BLOCK_SAMPLES_P*SAMPLE_WIDTH_P-1:0] samples,
      input  logic [3:0]                         mode,
      output logic                               busy,
      output logic                               digest_valid,
      output logic [DIGEST_WIDTH_P-1:0]          digest,
      output logic                               resonance_flag
   );

   localparam int LAT_BITS = (CORE_LATENCY > 1) ? $clog2(CORE_LATENCY) : 1;

   typedef struct packed {
      logic [DIGEST_WIDTH_P-1:0] digest;
      logic                      resonance;
   } core_result_t;

   logic [LAT_BITS-1:0]  latency_cnt;
   logic                 processing;
   core_result_t         pending_result;

   function automatic logic signed [15:0]
      kernel_real(input logic [2:0] k, input logic [2:0] n);
      logic [5:0] idx;
      idx = {k, n};
      unique case (idx)
         6'b000000,6'b000001,6'b000010,6'b000011,
         6'b000100,6'b000101,6'b000110,6'b000111: kernel_real = 16'sh2D40;
         6'b001000: kernel_real = 16'shECDF;
         6'b001001: kernel_real = 16'shD57A;
         6'b001010: kernel_real = 16'shD6FE;
         6'b001011: kernel_real = 16'shF088;
         6'b001100: kernel_real = 16'sh1321;
         6'b001101: kernel_real = 16'sh2A86;
         6'b001110: kernel_real = 16'sh2902;
         6'b001111: kernel_real = 16'sh0F78;
         6'b010000,6'b010100: kernel_real = 16'shD2EC;
         6'b010001,6'b010101: kernel_real = 16'sh03F4;
         6'b010010,6'b010110: kernel_real = 16'sh2D14;
         6'b010011,6'b010111: kernel_real = 16'shFC0C;
         6'b011000: kernel_real = 16'shD8D2;
         6'b011001: kernel_real = 16'sh2BB7;
         6'b011010: kernel_real = 16'shE95C;
         6'b011011: kernel_real = 16'shF44F;
         6'b011100: kernel_real = 16'sh272E;
         6'b011101: kernel_real = 16'shD449;
         6'b011110: kernel_real = 16'sh16A4;
         6'b011111: kernel_real = 16'sh0BB1;
         6'b100000,6'b100010,6'b100100,6'b100110: kernel_real = 16'shD371;
         6'b100001,6'b100011,6'b100101,6'b100111: kernel_real = 16'sh2C8F;
         6'b101000: kernel_real = 16'shE605;
         6'b101001: kernel_real = 16'sh2C92;
         6'b101010: kernel_real = 16'shDAF3;
         6'b101011: kernel_real = 16'sh07D3;
         6'b101100: kernel_real = 16'sh19FB;
         6'b101101: kernel_real = 16'shD36E;
         6'b101110: kernel_real = 16'sh250D;
         6'b101111: kernel_real = 16'shF82D;
         6'b110000,6'b110100: kernel_real = 16'sh2BB3;
         6'b110001,6'b110101: kernel_real = 16'sh0BBF;
         6'b110010,6'b110110: kernel_real = 16'shD44D;
         6'b110011,6'b110111: kernel_real = 16'shF441;
         6'b111000: kernel_real = 16'shDD5D;
         6'b111001: kernel_real = 16'shD2EB;
         6'b111010: kernel_real = 16'shE2E1;
         6'b111011: kernel_real = 16'sh03E6;
         6'b111100: kernel_real = 16'sh22A3;
         6'b111101: kernel_real = 16'sh2D15;
         6'b111110: kernel_real = 16'sh1D1F;
         6'b111111: kernel_real = 16'shFC1A;
         default:  kernel_real = 16'sh0000;
      endcase
   endfunction

   function automatic logic signed [15:0]
      kernel_imag(input logic [2:0] k, input logic [2:0] n);
      logic [5:0] idx;
      idx = {k, n};
      unique case (idx)
         6'b001000: kernel_imag = 16'shD6FE;
         6'b001001: kernel_imag = 16'shF088;
         6'b001010: kernel_imag = 16'sh1321;
         6'b001011: kernel_imag = 16'sh2A86;
         6'b001100: kernel_imag = 16'sh2902;
         6'b001101: kernel_imag = 16'sh0F78;
         6'b001110: kernel_imag = 16'shECDF;
         6'b001111: kernel_imag = 16'shD57A;
         6'b010000,6'b010100: kernel_imag = 16'sh03F4;
         6'b010001,6'b010101: kernel_imag = 16'sh2D14;
         6'b010010,6'b010110: kernel_imag = 16'shFC0C;
         6'b010011,6'b010111: kernel_imag = 16'shD2EC;
         6'b011000: kernel_imag = 16'sh16A4;
         6'b011001: kernel_imag = 16'sh0BB1;
         6'b011010: kernel_imag = 16'shD8D2;
         6'b011011: kernel_imag = 16'sh2BB7;
         6'b011100: kernel_imag = 16'shE95C;
         6'b011101: kernel_imag = 16'shF44F;
         6'b011110: kernel_imag = 16'sh272E;
         6'b011111: kernel_imag = 16'shD449;
         6'b100000,6'b100010,6'b100100,6'b100110: kernel_imag = 16'sh07E1;
         6'b100001,6'b100011,6'b100101,6'b100111: kernel_imag = 16'shF81F;
         6'b101000: kernel_imag = 16'shDAF3;
         6'b101001: kernel_imag = 16'sh07D3;
         6'b101010: kernel_imag = 16'sh19FB;
         6'b101011: kernel_imag = 16'shD36E;
         6'b101100: kernel_imag = 16'sh250D;
         6'b101101: kernel_imag = 16'shF82D;
         6'b101110: kernel_imag = 16'shE605;
         6'b101111: kernel_imag = 16'sh2C92;
         6'b110000,6'b110100: kernel_imag = 16'shF441;
         6'b110001,6'b110101: kernel_imag = 16'sh2BB3;
         6'b110010,6'b110110: kernel_imag = 16'sh0BBF;
         6'b110011,6'b110111: kernel_imag = 16'shD44D;
         6'b111000: kernel_imag = 16'sh1D1F;
         6'b111001: kernel_imag = 16'shFC1A;
         6'b111010: kernel_imag = 16'shDD5D;
         6'b111011: kernel_imag = 16'shD2EB;
         6'b111100: kernel_imag = 16'shE2E1;
         6'b111101: kernel_imag = 16'sh03E6;
         6'b111110: kernel_imag = 16'sh22A3;
         6'b111111: kernel_imag = 16'sh2D15;
         default:  kernel_imag = 16'sh0000;
      endcase
   endfunction

   function automatic core_result_t
      compute_block(
         input logic [BLOCK_SAMPLES_P*SAMPLE_WIDTH_P-1:0] vector,
         input logic [3:0]                                variant
      );
      core_result_t                             result;
      logic signed [SAMPLE_WIDTH_P-1:0]         sample_arr [0:BLOCK_SAMPLES_P-1];
      logic signed [31:0]                       rft_real   [0:BLOCK_SAMPLES_P-1];
      logic signed [31:0]                       rft_imag   [0:BLOCK_SAMPLES_P-1];
      logic [31:0]                               total_energy;
      logic [15:0]                               q_real    [0:BLOCK_SAMPLES_P-1];
      logic [15:0]                               q_imag    [0:BLOCK_SAMPLES_P-1];
      logic [7:0]                                s_vals    [0:BLOCK_SAMPLES_P-1];
      logic [15:0]                               lat       [0:BLOCK_SAMPLES_P-1];
      // Move all loop-scoped variables to function scope for Verilator compatibility
      logic signed [31:0] acc_real;
      logic signed [31:0] acc_imag;
      logic signed [15:0] coeff_r;
      logic signed [15:0] coeff_i;
      logic signed [31:0] mult_real;
      logic signed [31:0] mult_imag;
      logic [31:0] mag_real;
      logic [31:0] mag_imag;
      logic [31:0] amplitude;
      logic [15:0] amp_scaled;
      logic [31:0] sq_term;
      logic [127:0] sis_lo;
      logic [127:0] sis_hi;

      total_energy = 32'd0;
      /* verilator lint_off UNUSED */
      if (variant == 4'd0) begin end // variant reserved for extended kernel families
      /* verilator lint_on UNUSED */

      for (int i = 0; i < BLOCK_SAMPLES_P; i++) begin
         sample_arr[i] = vector[i*SAMPLE_WIDTH_P +: SAMPLE_WIDTH_P];
         rft_real[i]   = 32'sd0;
         rft_imag[i]   = 32'sd0;
      end

      for (int k = 0; k < BLOCK_SAMPLES_P; k++) begin
         acc_real = 32'sd0;
         acc_imag = 32'sd0;
         for (int n = 0; n < BLOCK_SAMPLES_P; n++) begin
            coeff_r   = kernel_real(k[2:0], n[2:0]);
            coeff_i   = kernel_imag(k[2:0], n[2:0]);
            mult_real = sample_arr[n] * coeff_r;
            mult_imag = sample_arr[n] * coeff_i;
            acc_real += mult_real;
            acc_imag += mult_imag;
         end
         rft_real[k] = acc_real;
         rft_imag[k] = acc_imag;

         mag_real   = acc_real[31] ? (32'h0 - acc_real) : acc_real;
         mag_imag   = acc_imag[31] ? (32'h0 - acc_imag) : acc_imag;
         amplitude  = mag_real + mag_imag;
         amp_scaled = amplitude[31:16];
         sq_term    = amp_scaled * amp_scaled;
         total_energy += sq_term;
      end

      for (int m = 0; m < BLOCK_SAMPLES_P; m++) begin
         q_real[m] = rft_real[m][31:16];
         q_imag[m] = rft_imag[m][31:16];
         s_vals[m] = 8'(q_real[m] + q_imag[m]);
      end

      lat[0] = 16'(s_vals[0]) + 16'(s_vals[1]) + 16'(s_vals[2]) + 16'(s_vals[3]);
      lat[1] = 16'(s_vals[1]) + 16'(s_vals[2]) + 16'(s_vals[3]) + 16'(s_vals[4]);
      lat[2] = 16'(s_vals[2]) + 16'(s_vals[3]) + 16'(s_vals[4]) + 16'(s_vals[5]);
      lat[3] = 16'(s_vals[3]) + 16'(s_vals[4]) + 16'(s_vals[5]) + 16'(s_vals[6]);
      lat[4] = 16'(s_vals[4]) + 16'(s_vals[5]) + 16'(s_vals[6]) + 16'(s_vals[7]);
      lat[5] = 16'(s_vals[5]) + 16'(s_vals[6]) + 16'(s_vals[7]) + 16'(s_vals[0]);
      lat[6] = 16'(s_vals[6]) + 16'(s_vals[7]) + 16'(s_vals[0]) + 16'(s_vals[1]);
      lat[7] = 16'(s_vals[7]) + 16'(s_vals[0]) + 16'(s_vals[1]) + 16'(s_vals[2]);

      sis_lo = {lat[3], lat[2], lat[1], lat[0], lat[7], lat[6], lat[5], lat[4]};
      sis_hi = {lat[3], lat[2], lat[1], lat[0], lat[7], lat[6], lat[5], lat[4]};

      result.digest    = {sis_hi, sis_lo};
      result.resonance = (total_energy > 32'd1000);
      return result;
   endfunction

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         busy            <= 1'b0;
         digest_valid    <= 1'b0;
         processing      <= 1'b0;
         latency_cnt     <= '0;
         digest          <= '0;
         resonance_flag  <= 1'b0;
      end else begin
         digest_valid <= 1'b0;
         if (start && !processing) begin
            pending_result <= compute_block(samples, mode);
            processing     <= 1'b1;
            busy           <= 1'b1;
            latency_cnt    <= LAT_BITS'((CORE_LATENCY > 0) ? CORE_LATENCY - 1 : 0);
         end else if (processing) begin
            if (latency_cnt == {LAT_BITS{1'b0}}) begin
               digest         <= pending_result.digest;
               resonance_flag <= pending_result.resonance;
               digest_valid   <= 1'b1;
               processing     <= 1'b0;
               busy           <= 1'b0;
            end else begin
               latency_cnt <= latency_cnt - 1'b1;
            end
         end
      end
   end
endmodule : phi_rft_core;

module rftpu_dma_ingress #(
      parameter int TILE_COUNT_P = rftpu_pkg::TILE_COUNT
   ) (
      input  logic                                    dma_valid,
      output logic                                    dma_ready,
      input  logic [rftpu_pkg::DMA_PAYLOAD_WIDTH-1:0]   dma_payload,
      output logic [TILE_COUNT_P-1:0]                 sample_valid,
      input  logic [TILE_COUNT_P-1:0]                 sample_ready,
      output logic [rftpu_pkg::SAMPLE_FRAME_WIDTH-1:0]  sample_data     [TILE_COUNT_P-1:0],
      output logic [15:0]                             sample_block_idx[TILE_COUNT_P-1:0]
   );

   import rftpu_pkg::*;

   dma_frame_t frame;
   assign frame = dma_frame_t'(dma_payload);

   logic [DMA_TILE_BITS-1:0] tile_index;
   assign tile_index = frame.tile_id;

   logic target_valid;
   assign target_valid = (tile_index < 7'(TILE_COUNT_P));

   logic tile_ready_mux;
   always_comb begin
      tile_ready_mux = 1'b0;
      if (target_valid) begin
         tile_ready_mux = sample_ready[tile_index[5:0]];
      end
   end

   assign dma_ready = tile_ready_mux;

   always_comb begin
      for (int t = 0; t < TILE_COUNT_P; t++) begin
         sample_valid[t]      = 1'b0;
         sample_data[t]       = '0;
         sample_block_idx[t]  = '0;
      end
      if (dma_valid && dma_ready && target_valid) begin
         sample_valid[tile_index[5:0]]     = 1'b1;
         sample_data[tile_index[5:0]]      = frame.samples;
         sample_block_idx[tile_index[5:0]] = frame.block_idx;
      end
   end
endmodule : rftpu_dma_ingress;

module rftpu_tile_shell #(
      parameter int X_COORD    = 0,
      parameter int Y_COORD    = 0,
      parameter int TILE_DIM_P = rftpu_pkg::TILE_DIM
   ) (
      input  logic                    clk,
      input  logic                    rst_n,
      input  rftpu_pkg::tile_ctrl_frame_t ctrl_frame,
      input  logic                    ctrl_valid,
      output logic                    ctrl_ready,
      output rftpu_pkg::tile_status_t   status,
      output logic                    irq_done,
      output logic                    irq_error,
      output logic                    noc_req_valid,
      input  logic                    noc_req_ready,
      output rftpu_pkg::noc_payload_t   noc_req_payload,
      input  logic                    noc_rsp_valid,
      output logic                    noc_rsp_ready,
      input  rftpu_pkg::noc_payload_t   noc_rsp_payload,
      input  logic                    sample_valid,
      output logic                    sample_ready,
      input  logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] sample_payload,
      input  logic [15:0]             sample_block_idx
   );

   import rftpu_pkg::*;

   localparam int SCRATCH_AW = $clog2(SCRATCH_DEPTH);
   localparam int TOPO_AW    = $clog2(TOPO_MEM_DEPTH);

   typedef enum logic [1:0] {SEND_IDLE, SEND_PRIMARY, SEND_H3A, SEND_H3B} send_state_t;

   tile_ctrl_frame_t         active_ctrl;
   logic                     run_active;
   logic                     run_active_d;
   logic [15:0]              blocks_remaining;
   logic [15:0]              blocks_completed;
   logic [SCRATCH_AW-1:0]    scratch_wr_ptr;
   logic [DIGEST_WIDTH-1:0]  scratchpad [SCRATCH_DEPTH-1:0];
   logic [DIGEST_WIDTH-1:0]  topo_mem   [TOPO_MEM_DEPTH-1:0];
   logic                     resonance_latched;
   logic                     error_flag;

   logic                     core_start;
   logic [BLOCK_SAMPLES*SAMPLE_WIDTH-1:0] sample_bus;
   logic                     core_busy;
   logic                     core_digest_valid;
   logic [DIGEST_WIDTH-1:0]  core_digest;
   logic                     core_resonance_flag;

   logic                     incoming_pending;
   noc_payload_t             incoming_payload;

   logic [DIGEST_WIDTH-1:0]  send_digest;
   logic [5:0]               send_vertex_id;
   logic [5:0]               last_vertex_id_reg;
   send_state_t              send_state;
   noc_payload_t             req_payload_reg;
   logic                     req_active;
   logic [15:0]              inflight_block_idx;
   logic [5:0]               inflight_vertex_idx;

   tile_status_t             status_reg;
   logic                     irq_done_reg;
   logic                     irq_error_reg;

   // Module-scope signals for procedural blocks
   logic [TOPO_AW-1:0]       topo_slot;
   logic [DIGEST_WIDTH-1:0]  topo_new_value;

   function automatic logic [TOPO_AW-1:0]
      map_vertex_index(
         input logic [7:0] base_id,
         input logic [5:0] vertex_id,
         input logic [3:0] mode
      );
      logic [9:0] folded;
      folded = 10'({mode[1:0], base_id[5:0]}) + 10'(vertex_id);
      return TOPO_AW'(folded % 32'(TOPO_MEM_DEPTH));
   endfunction

   function automatic logic [DIGEST_WIDTH-1:0]
      small_matrix_mix(input logic [DIGEST_WIDTH-1:0] data_in);
      logic [DIGEST_WIDTH-1:0] mix;
      mix = data_in;
      for (int k = 0; k < DIGEST_WIDTH/16; k += 4) begin
         for (int c = 0; c < 4; c++) begin
            mix[(k+c)*16 +: 16] = data_in[(k+c)*16 +: 16] ^ data_in[(k+((c+1)%4))*16 +: 16];
         end
      end
      return mix;
   endfunction

   function automatic noc_payload_t
      compose_payload(
         input logic [2:0] dest_x,
         input logic [2:0] dest_y,
         input logic [DIGEST_WIDTH-1:0] digest_value,
         input logic [5:0] vertex_id,
         input logic [3:0] mode_value,
         input logic [1:0] pkt_type_sel
      );
      noc_payload_t payload;
      payload.digest    = digest_value;
      payload.vertex_id = vertex_id;
      payload.mode      = mode_value;
      payload.src_x     = X_COORD[2:0];
      payload.src_y     = Y_COORD[2:0];
      payload.dest_x    = dest_x;
      payload.dest_y    = dest_y;
      payload.pkt_type  = pkt_type_sel;
      return payload;
   endfunction

   phi_rft_core core_inst (
      .clk          (clk),
      .rst_n        (rst_n),
      .start        (core_start),
      .samples      (sample_bus),
      .mode         (active_ctrl.mode),
      .busy         (core_busy),
      .digest_valid (core_digest_valid),
      .digest       (core_digest),
      .resonance_flag(core_resonance_flag)
   );

   assign ctrl_ready = !run_active;
   assign sample_ready = run_active && (blocks_remaining != 0) && !core_busy;
   assign inflight_vertex_idx = active_ctrl.vertex_base_id[5:0] + inflight_block_idx[5:0];

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         active_ctrl       <= '0;
         run_active        <= 1'b0;
         run_active_d      <= 1'b0;
         blocks_remaining  <= 16'd0;
         blocks_completed  <= 16'd0;
         scratch_wr_ptr    <= '0;
         resonance_latched <= 1'b0;
         error_flag        <= 1'b0;
      end else begin
         run_active_d <= run_active;
         if (ctrl_valid && ctrl_ready) begin
            active_ctrl      <= ctrl_frame;
            run_active       <= ctrl_frame.start;
            blocks_remaining <= ctrl_frame.length;
            blocks_completed <= 16'd0;
            scratch_wr_ptr   <= '0;
            resonance_latched<= 1'b0;
            error_flag       <= (ctrl_frame.length == 16'd0);
         end else if (ctrl_valid && !ctrl_ready) begin
            error_flag <= 1'b1;
         end
         if (run_active && core_start) begin
            blocks_remaining <= blocks_remaining - 1;
         end
         if (!run_active && ctrl_valid && ctrl_frame.start && !ctrl_ready)
            error_flag <= 1'b1;
         if (run_active && blocks_remaining == 0 && !core_busy && send_state == SEND_IDLE)
            run_active <= 1'b0;
      end
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         sample_bus  <= '0;
         core_start  <= 1'b0;
         inflight_block_idx <= 16'd0;
      end else begin
         core_start <= 1'b0;
         if (sample_valid && sample_ready) begin
            sample_bus         <= sample_payload;
            inflight_block_idx <= sample_block_idx;
            core_start         <= 1'b1;
         end
      end
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         send_digest    <= '0;
         send_vertex_id <= 6'd0;
         last_vertex_id_reg <= 6'd0;
         send_state     <= SEND_IDLE;
         req_active     <= 1'b0;
         req_payload_reg<= '0;
      end else begin
         if (core_digest_valid) begin
            send_digest        <= core_digest;
            send_vertex_id     <= inflight_vertex_idx;
            last_vertex_id_reg <= inflight_vertex_idx;
            blocks_completed   <= blocks_completed + 1;
            if (active_ctrl.cascade_enable) begin
               send_state <= SEND_PRIMARY;
            end else if (active_ctrl.h3_enable) begin
               send_state <= SEND_H3A;
            end else begin
               send_state <= SEND_IDLE;
               req_active <= 1'b0;
            end
         end

         case (send_state)
            SEND_IDLE: begin
               req_active <= 1'b0;
            end
            SEND_PRIMARY: begin
               req_payload_reg <= compose_payload(
                                     active_ctrl.cascade_dest_x,
                                     active_ctrl.cascade_dest_y,
                                     send_digest,
                                     send_vertex_id,
                                     active_ctrl.mode,
                                     2'd0
                                  );
               req_active <= 1'b1;
               if (noc_req_ready) begin
                  send_state <= active_ctrl.h3_enable ? SEND_H3A : SEND_IDLE;
                  if (!active_ctrl.h3_enable)
                     req_active <= 1'b0;
               end
            end
            SEND_H3A: begin
               req_payload_reg <= compose_payload(
                                     active_ctrl.h3_slot0_dest_x,
                                     active_ctrl.h3_slot0_dest_y,
                                     send_digest,
                                     send_vertex_id,
                                     active_ctrl.mode,
                                     2'd0
                                  );
               req_active <= 1'b1;
               if (noc_req_ready) begin
                  send_state <= (active_ctrl.h3_slot1_dest_x != 3'd0 || active_ctrl.h3_slot1_dest_y != 3'd0)
                                ? SEND_H3B : SEND_IDLE;
                  if (!(active_ctrl.h3_slot1_dest_x != 3'd0 || active_ctrl.h3_slot1_dest_y != 3'd0))
                     req_active <= 1'b0;
               end
            end
            SEND_H3B: begin
               req_payload_reg <= compose_payload(
                                     active_ctrl.h3_slot1_dest_x,
                                     active_ctrl.h3_slot1_dest_y,
                                     send_digest,
                                     send_vertex_id,
                                     active_ctrl.mode,
                                     2'd0
                                  );
               req_active <= 1'b1;
               if (noc_req_ready) begin
                  send_state <= SEND_IDLE;
                  req_active <= 1'b0;
               end
            end
         endcase
      end
   end

   assign noc_req_valid   = req_active;
   assign noc_req_payload = req_payload_reg;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         incoming_pending <= 1'b0;
         incoming_payload <= '0;
      end else begin
         if (!incoming_pending && noc_rsp_valid) begin
            incoming_pending <= 1'b1;
            incoming_payload <= noc_rsp_payload;
         end else if (incoming_pending) begin
            topo_slot = map_vertex_index(active_ctrl.vertex_base_id,
                                         incoming_payload.vertex_id,
                                         incoming_payload.mode);
            topo_new_value = topo_mem[topo_slot] ^ incoming_payload.digest;
            topo_mem[topo_slot] <= small_matrix_mix(topo_new_value);
            incoming_pending <= 1'b0;
         end
      end
   end

   assign noc_rsp_ready = !incoming_pending;

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         status_reg            <= '0;
         irq_done_reg          <= 1'b0;
         irq_error_reg         <= 1'b0;
      end else begin
         status_reg.busy             <= run_active;
         status_reg.done             <= !run_active;
         status_reg.error            <= error_flag;
         status_reg.resonance_active <= resonance_latched;
         status_reg.remaining        <= blocks_remaining;
         status_reg.mode             <= active_ctrl.mode;
         status_reg.last_vertex_id   <= {2'b0, last_vertex_id_reg};
         irq_done_reg                <= run_active_d && !run_active;
         irq_error_reg               <= error_flag;
      end
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         resonance_latched <= 1'b0;
      end else if (core_digest_valid) begin
         resonance_latched <= resonance_latched | core_resonance_flag;
         scratchpad[scratch_wr_ptr] <= core_digest;
         scratch_wr_ptr             <= scratch_wr_ptr + 1'b1;
         topo_slot = map_vertex_index(active_ctrl.vertex_base_id,
                                 inflight_vertex_idx,
                                 active_ctrl.mode);
         topo_mem[topo_slot] <= core_digest;
      end
   end

   assign status   = status_reg;
   assign irq_done = irq_done_reg;
   assign irq_error= irq_error_reg;
endmodule : rftpu_tile_shell;

module rftpu_noc_fabric #(
      parameter int TILE_DIM_P   = rftpu_pkg::TILE_DIM,
      parameter int MAX_INFLIGHT = rftpu_pkg::MAX_INFLIGHT,
      parameter int HOP_LATENCY  = rftpu_pkg::HOP_LATENCY
   ) (
      input  logic                         clk,
      input  logic                         rst_n,
      input  logic [TILE_DIM_P*TILE_DIM_P-1:0] req_valid,
      output logic [TILE_DIM_P*TILE_DIM_P-1:0] req_ready,
      input  rftpu_pkg::noc_payload_t        req_payload   [TILE_DIM_P*TILE_DIM_P],
      output logic [TILE_DIM_P*TILE_DIM_P-1:0] rsp_valid,
      input  logic [TILE_DIM_P*TILE_DIM_P-1:0] rsp_ready,
      output rftpu_pkg::noc_payload_t        rsp_payload   [TILE_DIM_P*TILE_DIM_P]
   );

   import rftpu_pkg::*;

   localparam int TILE_COUNT_P  = TILE_DIM_P * TILE_DIM_P;
   localparam int LATENCY_WIDTH = $clog2(HOP_LATENCY * (2*TILE_DIM_P) + 2);

   typedef struct packed {
      logic                     valid;
      noc_payload_t             payload;
      logic [LATENCY_WIDTH-1:0] latency;
   } inflight_t;

   inflight_t                  inflight   [MAX_INFLIGHT-1:0];
   logic [TILE_COUNT_P-1:0]    rsp_mailbox_valid;
   noc_payload_t               rsp_mailbox_payload[TILE_COUNT_P-1:0];
   logic [$clog2(TILE_COUNT_P)-1:0] arb_ptr;
   logic [$clog2(MAX_INFLIGHT)-1:0] free_slot_idx;
   logic                        free_slot_found;
   logic [$clog2(TILE_COUNT_P)-1:0] grant_idx;
   logic                        grant_found;

   function automatic logic [LATENCY_WIDTH-1:0]
      compute_latency(
         input noc_payload_t payload
      );
      logic [LATENCY_WIDTH-1:0] dx;
      logic [LATENCY_WIDTH-1:0] dy;
      logic [LATENCY_WIDTH-1:0] hops;
      dx   = (payload.dest_x > payload.src_x) ? LATENCY_WIDTH'(payload.dest_x - payload.src_x)
                                              : LATENCY_WIDTH'(payload.src_x - payload.dest_x);
      dy   = (payload.dest_y > payload.src_y) ? LATENCY_WIDTH'(payload.dest_y - payload.src_y)
                                              : LATENCY_WIDTH'(payload.src_y - payload.dest_y);
      hops = LATENCY_WIDTH'(HOP_LATENCY) * (dx + dy);
      if (hops == 0)
         hops = LATENCY_WIDTH'(HOP_LATENCY);
      return hops;
   endfunction

   always_comb begin
      free_slot_found = 1'b0;
      free_slot_idx   = '0;
      for (int i = 0; i < MAX_INFLIGHT; i++) begin
         if (!inflight[i].valid && !free_slot_found) begin
            free_slot_found = 1'b1;
            free_slot_idx   = i[$clog2(MAX_INFLIGHT)-1:0];
         end
      end
   end

   always_comb begin
      grant_found = 1'b0;
      grant_idx   = arb_ptr;
      for (int j = 0; j < TILE_COUNT_P; j++) begin
         logic [$clog2(TILE_COUNT_P)-1:0] candidate;
         candidate = $clog2(TILE_COUNT_P)'(($clog2(TILE_COUNT_P)'(arb_ptr) + $clog2(TILE_COUNT_P)'(j)) % TILE_COUNT_P);
         if (req_valid[candidate] && !grant_found) begin
            grant_found = 1'b1;
            grant_idx   = candidate;
         end
      end
      for (int k = 0; k < TILE_COUNT_P; k++) begin
         req_ready[k] = grant_found && free_slot_found && (grant_idx == $clog2(TILE_COUNT_P)'(k));
      end
      for (int t = 0; t < TILE_COUNT_P; t++) begin
         rsp_valid[t]   = rsp_mailbox_valid[t];
         rsp_payload[t] = rsp_mailbox_payload[t];
      end
   end

   always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
         for (int i = 0; i < MAX_INFLIGHT; i++) begin
            inflight[i].valid   <= 1'b0;
            inflight[i].payload <= '0;
            inflight[i].latency <= '0;
         end
         for (int j = 0; j < TILE_COUNT_P; j++) begin
            rsp_mailbox_valid[j]   <= 1'b0;
            rsp_mailbox_payload[j] <= '0;
         end
         arb_ptr <= '0;
      end else begin
         if (grant_found && free_slot_found && req_ready[grant_idx]) begin
            inflight[free_slot_idx].valid   <= 1'b1;
            inflight[free_slot_idx].payload <= req_payload[grant_idx];
            inflight[free_slot_idx].latency <= compute_latency(req_payload[grant_idx]);
            arb_ptr <= grant_idx + 1;
         end
         for (int m = 0; m < MAX_INFLIGHT; m++) begin
            if (inflight[m].valid) begin
               if (inflight[m].latency != 0) begin
                  inflight[m].latency <= inflight[m].latency - 1;
               end else begin
                  logic [$clog2(TILE_COUNT_P)-1:0] dest_tile;
                  dest_tile = $clog2(TILE_COUNT_P)'(inflight[m].payload.dest_y) * $clog2(TILE_COUNT_P)'(TILE_DIM_P) + $clog2(TILE_COUNT_P)'(inflight[m].payload.dest_x);
                  if (!rsp_mailbox_valid[dest_tile]) begin
                     rsp_mailbox_valid[dest_tile]   <= 1'b1;
                     rsp_mailbox_payload[dest_tile] <= inflight[m].payload;
                     inflight[m].valid              <= 1'b0;
                  end
               end
            end
         end
         for (int d = 0; d < TILE_COUNT_P; d++) begin
            if (rsp_mailbox_valid[d] && rsp_ready[d]) begin
               rsp_mailbox_valid[d] <= 1'b0;
            end
         end
      end
   end
endmodule : rftpu_noc_fabric;

module rftpu_accelerator #(
      parameter int TILE_DIM_P = rftpu_pkg::TILE_DIM
   ) (
      input  logic                                 clk,
      input  logic                                 rst_n,
      input  logic                                 cfg_valid,
      output logic                                 cfg_ready,
      input  logic [6:0]                           cfg_tile_id,
      input  logic [rftpu_pkg::CTRL_PAYLOAD_WIDTH-1:0] cfg_payload,
      input  logic                                 dma_valid,
      output logic                                 dma_ready,
      input  logic [rftpu_pkg::DMA_PAYLOAD_WIDTH-1:0] dma_payload,
      output logic [TILE_DIM_P*TILE_DIM_P-1:0]     tile_done_bitmap,
      output logic [TILE_DIM_P*TILE_DIM_P-1:0]     tile_busy_bitmap,
      output logic                                 global_irq_done,
      output logic                                 global_irq_error
   );

   import rftpu_pkg::*;

   localparam int TILE_COUNT_P = TILE_DIM_P * TILE_DIM_P;

   tile_ctrl_frame_t  broadcast_frame;
   logic [TILE_COUNT_P-1:0] tile_ctrl_valid;
   logic [TILE_COUNT_P-1:0] tile_ctrl_ready;
   tile_status_t            tile_status   [TILE_COUNT_P-1:0];
   logic [TILE_COUNT_P-1:0] tile_irq_done;
   logic [TILE_COUNT_P-1:0] tile_irq_error;

   logic [TILE_COUNT_P-1:0] noc_req_valid;
   noc_payload_t            noc_req_payload[TILE_COUNT_P-1:0];
   logic [TILE_COUNT_P-1:0] noc_req_ready;
   logic [TILE_COUNT_P-1:0] noc_rsp_valid;
   noc_payload_t            noc_rsp_payload[TILE_COUNT_P-1:0];
   logic [TILE_COUNT_P-1:0] noc_rsp_ready;
   logic [TILE_COUNT_P-1:0] sample_valid_bus;
   logic [TILE_COUNT_P-1:0] sample_ready_bus;
   logic [SAMPLE_FRAME_WIDTH-1:0] sample_data_bus[TILE_COUNT_P-1:0];
   logic [15:0]                 sample_block_idx_bus[TILE_COUNT_P-1:0];

   function automatic tile_ctrl_frame_t
      decode_ctrl_frame(input logic [CTRL_PAYLOAD_WIDTH-1:0] payload);
      tile_ctrl_frame_t frame;
      frame.start             = payload[0];
      frame.mode              = payload[4:1];
      frame.length            = payload[20:5];
      frame.in_addr           = payload[52:21];
      frame.out_addr          = payload[84:53];
      frame.cascade_enable    = payload[85];
      frame.cascade_dest_x    = payload[88:86];
      frame.cascade_dest_y    = payload[91:89];
      frame.h3_enable         = payload[92];
      frame.h3_slot0_dest_x   = payload[95:93];
      frame.h3_slot0_dest_y   = payload[98:96];
      frame.h3_slot0_vertex   = payload[104:99];
      frame.h3_slot1_dest_x   = payload[107:105];
      frame.h3_slot1_dest_y   = payload[110:108];
      frame.h3_slot1_vertex   = payload[116:111];
      frame.vertex_base_id    = payload[124:117];
      return frame;
   endfunction

   assign broadcast_frame = decode_ctrl_frame(cfg_payload);

   rftpu_dma_ingress #(
      .TILE_COUNT_P (TILE_COUNT_P)
   ) dma_ingress_inst (
      .dma_valid        (dma_valid),
      .dma_ready        (dma_ready),
      .dma_payload      (dma_payload),
      .sample_valid     (sample_valid_bus),
      .sample_ready     (sample_ready_bus),
      .sample_data      (sample_data_bus),
      .sample_block_idx (sample_block_idx_bus)
   );

   generate
      genvar row;
      genvar col;
      for (row = 0; row < TILE_DIM_P; row++) begin : gen_rows
         for (col = 0; col < TILE_DIM_P; col++) begin : gen_cols
            localparam int TILE_ID = row*TILE_DIM_P + col;
            localparam logic [6:0] TILE_IDX = TILE_ID[6:0];
            assign tile_ctrl_valid[TILE_ID] = cfg_valid && (cfg_tile_id == TILE_IDX);
            rftpu_tile_shell #(
               .X_COORD    (col),
               .Y_COORD    (row),
               .TILE_DIM_P (TILE_DIM_P)
            ) tile_inst (
               .clk            (clk),
               .rst_n          (rst_n),
               .ctrl_frame     (broadcast_frame),
               .ctrl_valid     (tile_ctrl_valid[TILE_ID]),
               .ctrl_ready     (tile_ctrl_ready[TILE_ID]),
               .status         (tile_status[TILE_ID]),
               .irq_done       (tile_irq_done[TILE_ID]),
               .irq_error      (tile_irq_error[TILE_ID]),
               .noc_req_valid  (noc_req_valid[TILE_ID]),
               .noc_req_ready  (noc_req_ready[TILE_ID]),
               .noc_req_payload(noc_req_payload[TILE_ID]),
               .noc_rsp_valid  (noc_rsp_valid[TILE_ID]),
               .noc_rsp_ready  (noc_rsp_ready[TILE_ID]),
               .noc_rsp_payload(noc_rsp_payload[TILE_ID]),
               .sample_valid   (sample_valid_bus[TILE_ID]),
               .sample_ready   (sample_ready_bus[TILE_ID]),
               .sample_payload (sample_data_bus[TILE_ID]),
               .sample_block_idx(sample_block_idx_bus[TILE_ID])
            );
         end
      end
   endgenerate

   rftpu_noc_fabric #(
      .TILE_DIM_P   (TILE_DIM_P),
      .MAX_INFLIGHT (rftpu_pkg::MAX_INFLIGHT),
      .HOP_LATENCY  (rftpu_pkg::HOP_LATENCY)
   ) noc_inst (
      .clk         (clk),
      .rst_n       (rst_n),
      .req_valid   (noc_req_valid),
      .req_ready   (noc_req_ready),
      .req_payload (noc_req_payload),
      .rsp_valid   (noc_rsp_valid),
      .rsp_ready   (noc_rsp_ready),
      .rsp_payload (noc_rsp_payload)
   );

   function automatic logic mux_ready(
      input logic [TILE_COUNT_P-1:0] ready_vec,
      input logic [6:0]              tile_id
   );
      mux_ready = (tile_id < 7'(TILE_COUNT_P)) ? ready_vec[tile_id[5:0]] : 1'b0;
   endfunction

   assign cfg_ready        = mux_ready(tile_ctrl_ready, cfg_tile_id);
   assign global_irq_done  = |tile_irq_done;
   assign global_irq_error = |tile_irq_error;

   always_comb begin
      for (int k = 0; k < TILE_COUNT_P; k++) begin
         tile_done_bitmap[k] = tile_status[k].done;
         tile_busy_bitmap[k] = tile_status[k].busy;
      end
   end
endmodule : rftpu_accelerator;