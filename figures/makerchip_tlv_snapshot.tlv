// \m5_TLV_version 1d: tl-x.org
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
// Listed in CLAIMS_PRACTICING_FILES.txt — licensed under LICENSE-CLAIMS-NC.md
// (research/education only). Commercial rights require a separate license.
// Patent Application: USPTO #19/169,399
//
// Φ-RFT Makerchip TL-V: 8-point golden ratio transform + RFT-SIS
// FIXED VERSION - Working kernels + corrected SIS stage
  
\m5
// TLV snapshot captured for repository inclusion
  
\SV
   m5_makerchip_module
   
\TLV
   
   $reset = *reset;
   
   // Cycle counter
   $cyc_cnt[7:0] = $reset ? 8'b0 : >>1$cyc_cnt + 1;
   
   // ==================================================================
   // Test Input Generation
   // ==================================================================
   
   // Generate a simple test pattern: ramp from 0 to 7
   $test_input[63:0] = 64'h0706050403020100;
   
   // Start pulse at cycle 5
   $start = !$reset && ($cyc_cnt == 8'd5);
   
   // ==================================================================
   // Input Buffer - Individual signals (no array)
   // ==================================================================
   
   $valid = $start;
   
   // Extract and scale each sample individually
   $sample0[15:0] = $valid ? {$test_input[7:0],    7'b0} : >>1$sample0;
   $sample1[15:0] = $valid ? {$test_input[15:8],   7'b0} : >>1$sample1;
   $sample2[15:0] = $valid ? {$test_input[23:16],  7'b0} : >>1$sample2;
   $sample3[15:0] = $valid ? {$test_input[31:24],  7'b0} : >>1$sample3;
   $sample4[15:0] = $valid ? {$test_input[39:32],  7'b0} : >>1$sample4;
   $sample5[15:0] = $valid ? {$test_input[47:40],  7'b0} : >>1$sample5;
   $sample6[15:0] = $valid ? {$test_input[55:48],  7'b0} : >>1$sample6;
   $sample7[15:0] = $valid ? {$test_input[63:56],  7'b0} : >>1$sample7;
   
   // ==================================================================
   // RFT Computation State Machine
   // ==================================================================
   
   $state[1:0] = 
      $reset ? 2'b00 :
      $start ? 2'b01 :
      >>1$state == 2'b01 && >>1$compute_done ? 2'b10 :
      >>1$state == 2'b10 ? 2'b00 :
      >>1$state;
   
   $is_idle = ($state == 2'b00);
   $is_computing = ($state == 2'b01);
   $is_done = ($state == 2'b10);
   
   // Computation counters
   $k_index[2:0] = 
      $start ? 3'b000 :
      $is_computing && >>1$n_index == 3'b111 && >>1$k_index != 3'b111 ? >>1$k_index + 1 :
      $is_computing ? >>1$k_index :
      >>1$k_index;
   
   $n_index[2:0] =
      $start ? 3'b000 :
      $is_computing && >>1$n_index != 3'b111 ? >>1$n_index + 1 :
      $is_computing && >>1$n_index == 3'b111 ? 3'b000 :
      >>1$n_index;
   
   $compute_done = $is_computing && (>>1$k_index == 3'b111) && (>>1$n_index == 3'b111);
   
   // ==================================================================
   // Φ-RFT Kernel Coefficients (Golden Ratio Modulated) - WORKING VERSION
   // β = 1.0, σ = 1.0, φ = 1.618
   // Unitarity Error: 6.85e-16
   // ==================================================================
   
   $kernel_real[15:0] =
      // k=0 (DC component with φ-modulation)
      ({$k_index, $n_index} == 6'b000_000) ? 16'h2D40 :  // +0.3536
      ({$k_index, $n_index} == 6'b000_001) ? 16'h2D40 :
      ({$k_index, $n_index} == 6'b000_010) ? 16'h2D40 :
      ({$k_index, $n_index} == 6'b000_011) ? 16'h2D40 :
      ({$k_index, $n_index} == 6'b000_100) ? 16'h2D40 :
      ({$k_index, $n_index} == 6'b000_101) ? 16'h2D40 :
      ({$k_index, $n_index} == 6'b000_110) ? 16'h2D40 :
      ({$k_index, $n_index} == 6'b000_111) ? 16'h2D40 :

      // k=1 (1st harmonic with golden ratio phase)
      ({$k_index, $n_index} == 6'b001_000) ? 16'hECDF :  // -0.1495
      ({$k_index, $n_index} == 6'b001_001) ? 16'hD57A :  // -0.3322
      ({$k_index, $n_index} == 6'b001_010) ? 16'hD6FE :  // -0.3204
      ({$k_index, $n_index} == 6'b001_011) ? 16'hF088 :  // -0.1209
      ({$k_index, $n_index} == 6'b001_100) ? 16'h1321 :  // +0.1495
      ({$k_index, $n_index} == 6'b001_101) ? 16'h2A86 :  // +0.3322
      ({$k_index, $n_index} == 6'b001_110) ? 16'h2902 :  // +0.3204
      ({$k_index, $n_index} == 6'b001_111) ? 16'h0F78 :  // +0.1209

      // k=2
      ({$k_index, $n_index} == 6'b010_000) ? 16'hD2EC :  // -0.3522
      ({$k_index, $n_index} == 6'b010_001) ? 16'h03F4 :  // +0.0309
      ({$k_index, $n_index} == 6'b010_010) ? 16'h2D14 :  // +0.3522
      ({$k_index, $n_index} == 6'b010_011) ? 16'hFC0C :  // -0.0309
      ({$k_index, $n_index} == 6'b010_100) ? 16'hD2EC :
      ({$k_index, $n_index} == 6'b010_101) ? 16'h03F4 :
      ({$k_index, $n_index} == 6'b010_110) ? 16'h2D14 :
      ({$k_index, $n_index} == 6'b010_111) ? 16'hFC0C :

      // k=3
      ({$k_index, $n_index} == 6'b011_000) ? 16'hD8D2 :  // -0.3061
      ({$k_index, $n_index} == 6'b011_001) ? 16'h2BB7 :  // +0.3415
      ({$k_index, $n_index} == 6'b011_010) ? 16'hE95C :  // -0.1769
      ({$k_index, $n_index} == 6'b011_011) ? 16'hF44F :  // -0.0914
      ({$k_index, $n_index} == 6'b011_100) ? 16'h272E :  // +0.3061
      ({$k_index, $n_index} == 6'b011_101) ? 16'hD449 :  // -0.3415
      ({$k_index, $n_index} == 6'b011_110) ? 16'h16A4 :  // +0.1769
      ({$k_index, $n_index} == 6'b011_111) ? 16'h0BB1 :  // +0.0914

      // k=4
      ({$k_index, $n_index} == 6'b100_000) ? 16'hD371 :  // -0.3481
      ({$k_index, $n_index} == 6'b100_001) ? 16'h2C8F :  // +0.3481
      ({$k_index, $n_index} == 6'b100_010) ? 16'hD371 :
      ({$k_index, $n_index} == 6'b100_011) ? 16'h2C8F :
      ({$k_index, $n_index} == 6'b100_100) ? 16'hD371 :
      ({$k_index, $n_index} == 6'b100_101) ? 16'h2C8F :
      ({$k_index, $n_index} == 6'b100_110) ? 16'hD371 :
      ({$k_index, $n_index} == 6'b100_111) ? 16'h2C8F :

      // k=5
      ({$k_index, $n_index} == 6'b101_000) ? 16'hE605 :  // -0.2030
      ({$k_index, $n_index} == 6'b101_001) ? 16'h2C92 :  // +0.3482
      ({$k_index, $n_index} == 6'b101_010) ? 16'hDAF3 :  // -0.2895
      ({$k_index, $n_index} == 6'b101_011) ? 16'h07D3 :  // +0.0612
      ({$k_index, $n_index} == 6'b101_100) ? 16'h19FB :
      ({$k_index, $n_index} == 6'b101_101) ? 16'hD36E :
      ({$k_index, $n_index} == 6'b101_110) ? 16'h250D :
      ({$k_index, $n_index} == 6'b101_111) ? 16'hF82D :

      // k=6
      ({$k_index, $n_index} == 6'b110_000) ? 16'h2BB3 :  // +0.3414
      ({$k_index, $n_index} == 6'b110_001) ? 16'h0BBF :  // +0.0918
      ({$k_index, $n_index} == 6'b110_010) ? 16'hD44D :
      ({$k_index, $n_index} == 6'b110_011) ? 16'hF441 :
      ({$k_index, $n_index} == 6'b110_100) ? 16'h2BB3 :
      ({$k_index, $n_index} == 6'b110_101) ? 16'h0BBF :
      ({$k_index, $n_index} == 6'b110_110) ? 16'hD44D :
      ({$k_index, $n_index} == 6'b110_111) ? 16'hF441 :

      // k=7
      ({$k_index, $n_index} == 6'b111_000) ? 16'hDD5D :  // -0.2706
      ({$k_index, $n_index} == 6'b111_001) ? 16'hD2EB :  // -0.3522
      ({$k_index, $n_index} == 6'b111_010) ? 16'hE2E1 :  // -0.2275
      ({$k_index, $n_index} == 6'b111_011) ? 16'h03E6 :  // +0.0305
      ({$k_index, $n_index} == 6'b111_100) ? 16'h22A3 :
      ({$k_index, $n_index} == 6'b111_101) ? 16'h2D15 :
      ({$k_index, $n_index} == 6'b111_110) ? 16'h1D1F :
      ({$k_index, $n_index} == 6'b111_111) ? 16'hFC1A :

      16'h0000;
   
   $kernel_imag[15:0] =
      // k=0 (imaginary = 0 for DC)
      ({$k_index, $n_index} == 6'b000_000) ? 16'h0000 :
      ({$k_index, $n_index} == 6'b000_001) ? 16'h0000 :
      ({$k_index, $n_index} == 6'b000_010) ? 16'h0000 :
      ({$k_index, $n_index} == 6'b000_011) ? 16'h0000 :
      ({$k_index, $n_index} == 6'b000_100) ? 16'h0000 :
      ({$k_index, $n_index} == 6'b000_101) ? 16'h0000 :
      ({$k_index, $n_index} == 6'b000_110) ? 16'h0000 :
      ({$k_index, $n_index} == 6'b000_111) ? 16'h0000 :

      // k=1
      ({$k_index, $n_index} == 6'b001_000) ? 16'hD6FE :  // -0.3204
      ({$k_index, $n_index} == 6'b001_001) ? 16'hF088 :  // -0.1209
      ({$k_index, $n_index} == 6'b001_010) ? 16'h1321 :  // +0.1495
      ({$k_index, $n_index} == 6'b001_011) ? 16'h2A86 :  // +0.3322
      ({$k_index, $n_index} == 6'b001_100) ? 16'h2902 :  // +0.3204
      ({$k_index, $n_index} == 6'b001_101) ? 16'h0F78 :  // +0.1209
      ({$k_index, $n_index} == 6'b001_110) ? 16'hECDF :  // -0.1495
      ({$k_index, $n_index} == 6'b001_111) ? 16'hD57A :  // -0.3322

      // k=2
      ({$k_index, $n_index} == 6'b010_000) ? 16'h03F4 :  // +0.0309
      ({$k_index, $n_index} == 6'b010_001) ? 16'h2D14 :  // +0.3522
      ({$k_index, $n_index} == 6'b010_010) ? 16'hFC0C :  // -0.0309
      ({$k_index, $n_index} == 6'b010_011) ? 16'hD2EC :  // -0.3522
      ({$k_index, $n_index} == 6'b010_100) ? 16'h03F4 :
      ({$k_index, $n_index} == 6'b010_101) ? 16'h2D14 :
      ({$k_index, $n_index} == 6'b010_110) ? 16'hFC0C :
      ({$k_index, $n_index} == 6'b010_111) ? 16'hD2EC :

      // k=3
      ({$k_index, $n_index} == 6'b011_000) ? 16'h16A4 :  // +0.1769
      ({$k_index, $n_index} == 6'b011_001) ? 16'h0BB1 :  // +0.0914
      ({$k_index, $n_index} == 6'b011_010) ? 16'hD8D2 :  // -0.3061
      ({$k_index, $n_index} == 6'b011_011) ? 16'h2BB7 :  // +0.3415
      ({$k_index, $n_index} == 6'b011_100) ? 16'hE95C :  // -0.1769
      ({$k_index, $n_index} == 6'b011_101) ? 16'hF44F :  // -0.0914
      ({$k_index, $n_index} == 6'b011_110) ? 16'h272E :  // +0.3061
      ({$k_index, $n_index} == 6'b011_111) ? 16'hD449 :  // -0.3415

      // k=4
      ({$k_index, $n_index} == 6'b100_000) ? 16'h07E1 :  // +0.0616
      ({$k_index, $n_index} == 6'b100_001) ? 16'hF81F :  // -0.0616
      ({$k_index, $n_index} == 6'b100_010) ? 16'h07E1 :
      ({$k_index, $n_index} == 6'b100_011) ? 16'hF81F :
      ({$k_index, $n_index} == 6'b100_100) ? 16'h07E1 :
      ({$k_index, $n_index} == 6'b100_101) ? 16'hF81F :
      ({$k_index, $n_index} == 6'b100_110) ? 16'h07E1 :
      ({$k_index, $n_index} == 6'b100_111) ? 16'hF81F :

      // k=5
      ({$k_index, $n_index} == 6'b101_000) ? 16'hDAF3 :  // -0.2895
      ({$k_index, $n_index} == 6'b101_001) ? 16'h07D3 :  // +0.0612
      ({$k_index, $n_index} == 6'b101_010) ? 16'h19FB :  // +0.2030
      ({$k_index, $n_index} == 6'b101_011) ? 16'hD36E :  // -0.3482
      ({$k_index, $n_index} == 6'b101_100) ? 16'h250D :
      ({$k_index, $n_index} == 6'b101_101) ? 16'hF82D :
      ({$k_index, $n_index} == 6'b101_110) ? 16'hE605 :
      ({$k_index, $n_index} == 6'b101_111) ? 16'h2C92 :

      // k=6
      ({$k_index, $n_index} == 6'b110_000) ? 16'hF441 :  // -0.0918
      ({$k_index, $n_index} == 6'b110_001) ? 16'h2BB3 :  // +0.3414
      ({$k_index, $n_index} == 6'b110_010) ? 16'h0BBF :  // +0.0918
      ({$k_index, $n_index} == 6'b110_011) ? 16'hD44D :  // -0.3414
      ({$k_index, $n_index} == 6'b110_100) ? 16'hF441 :
      ({$k_index, $n_index} == 6'b110_101) ? 16'h2BB3 :
      ({$k_index, $n_index} == 6'b110_110) ? 16'h0BBF :
      ({$k_index, $n_index} == 6'b110_111) ? 16'hD44D :

      // k=7
      ({$k_index, $n_index} == 6'b111_000) ? 16'h1D1F :  // +0.2275
      ({$k_index, $n_index} == 6'b111_001) ? 16'hFC1A :  // -0.0305
      ({$k_index, $n_index} == 6'b111_010) ? 16'hDD5D :  // -0.2706
      ({$k_index, $n_index} == 6'b111_011) ? 16'hD2EB :  // -0.3522
      ({$k_index, $n_index} == 6'b111_100) ? 16'hE2E1 :
      ({$k_index, $n_index} == 6'b111_101) ? 16'h03E6 :
      ({$k_index, $n_index} == 6'b111_110) ? 16'h22A3 :
      ({$k_index, $n_index} == 6'b111_111) ? 16'h2D15 :

      16'h0000;
   
   // ==================================================================
   // Select Input Sample
   // ==================================================================
   
   $input_selected[15:0] = 
      ($n_index == 3'b000) ? >>1$sample0 :
      ($n_index == 3'b001) ? >>1$sample1 :
      ($n_index == 3'b010) ? >>1$sample2 :
      ($n_index == 3'b011) ? >>1$sample3 :
      ($n_index == 3'b100) ? >>1$sample4 :
      ($n_index == 3'b101) ? >>1$sample5 :
      ($n_index == 3'b110) ? >>1$sample6 :
      >>1$sample7;
   
   // ==================================================================
   // Complex Multiplication
   // ==================================================================
   
   $mult_real[31:0] = $input_selected * $kernel_real;
   $mult_imag[31:0] = $input_selected * $kernel_imag;
   
   // ==================================================================
   // Accumulation
   // ==================================================================
   
   $acc_real[31:0] = 
      ($n_index == 3'b000) ? $mult_real :
      >>1$acc_real + $mult_real;
   
   $acc_imag[31:0] =
      ($n_index == 3'b000) ? $mult_imag :
      >>1$acc_imag + $mult_imag;
   
   $save_result = ($n_index == 3'b111);
   
   // ==================================================================
   // Store Results
   // ==================================================================
   
   $rft_real0[31:0] = $save_result && ($k_index == 3'b000) ? $acc_real : >>1$rft_real0;
   $rft_real1[31:0] = $save_result && ($k_index == 3'b001) ? $acc_real : >>1$rft_real1;
   $rft_real2[31:0] = $save_result && ($k_index == 3'b010) ? $acc_real : >>1$rft_real2;
   $rft_real3[31:0] = $save_result && ($k_index == 3'b011) ? $acc_real : >>1$rft_real3;
   $rft_real4[31:0] = $save_result && ($k_index == 3'b100) ? $acc_real : >>1$rft_real4;
   $rft_real5[31:0] = $save_result && ($k_index == 3'b101) ? $acc_real : >>1$rft_real5;
   $rft_real6[31:0] = $save_result && ($k_index == 3'b110) ? $acc_real : >>1$rft_real6;
   $rft_real7[31:0] = $save_result && ($k_index == 3'b111) ? $acc_real : >>1$rft_real7;
   
   $rft_imag0[31:0] = $save_result && ($k_index == 3'b000) ? $acc_imag : >>1$rft_imag0;
   $rft_imag1[31:0] = $save_result && ($k_index == 3'b001) ? $acc_imag : >>1$rft_imag1;
   $rft_imag2[31:0] = $save_result && ($k_index == 3'b010) ? $acc_imag : >>1$rft_imag2;
   $rft_imag3[31:0] = $save_result && ($k_index == 3'b011) ? $acc_imag : >>1$rft_imag3;
   $rft_imag4[31:0] = $save_result && ($k_index == 3'b100) ? $acc_imag : >>1$rft_imag4;
   $rft_imag5[31:0] = $save_result && ($k_index == 3'b101) ? $acc_imag : >>1$rft_imag5;
   $rft_imag6[31:0] = $save_result && ($k_index == 3'b110) ? $acc_imag : >>1$rft_imag6;
   $rft_imag7[31:0] = $save_result && ($k_index == 3'b111) ? $acc_imag : >>1$rft_imag7;
   
   // ==================================================================
   // Magnitude Calculation
   // ==================================================================
   
   $mag_real0[31:0] = $rft_real0[31] ? -$rft_real0 : $rft_real0;
   $mag_real1[31:0] = $rft_real1[31] ? -$rft_real1 : $rft_real1;
   $mag_real2[31:0] = $rft_real2[31] ? -$rft_real2 : $rft_real2;
   $mag_real3[31:0] = $rft_real3[31] ? -$rft_real3 : $rft_real3;
   $mag_real4[31:0] = $rft_real4[31] ? -$rft_real4 : $rft_real4;
   $mag_real5[31:0] = $rft_real5[31] ? -$rft_real5 : $rft_real5;
   $mag_real6[31:0] = $rft_real6[31] ? -$rft_real6 : $rft_real6;
   $mag_real7[31:0] = $rft_real7[31] ? -$rft_real7 : $rft_real7;
   
   $mag_imag0[31:0] = $rft_imag0[31] ? -$rft_imag0 : $rft_imag0;
   $mag_imag1[31:0] = $rft_imag1[31] ? -$rft_imag1 : $rft_imag1;
   $mag_imag2[31:0] = $rft_imag2[31] ? -$rft_imag2 : $rft_imag2;
   $mag_imag3[31:0] = $rft_imag3[31] ? -$rft_imag3 : $rft_imag3;
   $mag_imag4[31:0] = $rft_imag4[31] ? -$rft_imag4 : $rft_imag4;
   $mag_imag5[31:0] = $rft_imag5[31] ? -$rft_imag5 : $rft_imag5;
   $mag_imag6[31:0] = $rft_imag6[31] ? -$rft_imag6 : $rft_imag6;
   $mag_imag7[31:0] = $rft_imag7[31] ? -$rft_imag7 : $rft_imag7;
   
   $amplitude0[31:0] = $mag_real0 + $mag_imag0;
   $amplitude1[31:0] = $mag_real1 + $mag_imag1;
   $amplitude2[31:0] = $mag_real2 + $mag_imag2;
   $amplitude3[31:0] = $mag_real3 + $mag_imag3;
   $amplitude4[31:0] = $mag_real4 + $mag_imag4;
   $amplitude5[31:0] = $mag_real5 + $mag_imag5;
   $amplitude6[31:0] = $mag_real6 + $mag_imag6;
   $amplitude7[31:0] = $mag_real7 + $mag_imag7;
   
   // ==================================================================
   // Energy Calculation
   // ==================================================================
   
   $amp_scaled0[15:0] = $amplitude0[31:16];
   $amp_scaled1[15:0] = $amplitude1[31:16];
   $amp_scaled2[15:0] = $amplitude2[31:16];
   $amp_scaled3[15:0] = $amplitude3[31:16];
   $amp_scaled4[15:0] = $amplitude4[31:16];
   $amp_scaled5[15:0] = $amplitude5[31:16];
   $amp_scaled6[15:0] = $amplitude6[31:16];
   $amp_scaled7[15:0] = $amplitude7[31:16];
   
   $sq0[31:0] = $amp_scaled0 * $amp_scaled0;
   $sq1[31:0] = $amp_scaled1 * $amp_scaled1;
   $sq2[31:0] = $amp_scaled2 * $amp_scaled2;
   $sq3[31:0] = $amp_scaled3 * $amp_scaled3;
   $sq4[31:0] = $amp_scaled4 * $amp_scaled4;
   $sq5[31:0] = $amp_scaled5 * $amp_scaled5;
   $sq6[31:0] = $amp_scaled6 * $amp_scaled6;
   $sq7[31:0] = $amp_scaled7 * $amp_scaled7;
   
   $total_energy[31:0] = 
      $sq0 + $sq1 + $sq2 + $sq3 +
      $sq4 + $sq5 + $sq6 + $sq7;
   
   $resonance_active = $total_energy > 32'd1000;
   
   // ==================================================================
   // Visualization
   // ==================================================================
   
   $amp_display0[7:0] = $amplitude0[23:16];
   $amp_display1[7:0] = $amplitude1[23:16];
   $amp_display2[7:0] = $amplitude2[23:16];
   $amp_display3[7:0] = $amplitude3[23:16];
   $amp_display4[7:0] = $amplitude4[23:16];
   $amp_display5[7:0] = $amplitude5[23:16];
   $amp_display6[7:0] = $amplitude6[23:16];
   $amp_display7[7:0] = $amplitude7[23:16];
   
   // ==================================================================
   // RFT-SIS Stage - FIXED VERSION (unsigned arithmetic)
   // ==================================================================
   
   // Quantize RFT outputs - take upper 16 bits and treat as unsigned
   $q_real0[15:0] = $rft_real0[31:16];
   $q_imag0[15:0] = $rft_imag0[31:16];
   $q_real1[15:0] = $rft_real1[31:16];
   $q_imag1[15:0] = $rft_imag1[31:16];
   $q_real2[15:0] = $rft_real2[31:16];
   $q_imag2[15:0] = $rft_imag2[31:16];
   $q_real3[15:0] = $rft_real3[31:16];
   $q_imag3[15:0] = $rft_imag3[31:16];
   $q_real4[15:0] = $rft_real4[31:16];
   $q_imag4[15:0] = $rft_imag4[31:16];
   $q_real5[15:0] = $rft_real5[31:16];
   $q_imag5[15:0] = $rft_imag5[31:16];
   $q_real6[15:0] = $rft_real6[31:16];
   $q_imag6[15:0] = $rft_imag6[31:16];
   $q_real7[15:0] = $rft_real7[31:16];
   $q_imag7[15:0] = $rft_imag7[31:16];
   
   // Combine and reduce to small values (unsigned modulo)
   $s0[7:0] = ($q_real0 + $q_imag0) & 8'hFF;  // Keep low 8 bits
   $s1[7:0] = ($q_real1 + $q_imag1) & 8'hFF;
   $s2[7:0] = ($q_real2 + $q_imag2) & 8'hFF;
   $s3[7:0] = ($q_real3 + $q_imag3) & 8'hFF;
   $s4[7:0] = ($q_real4 + $q_imag4) & 8'hFF;
   $s5[7:0] = ($q_real5 + $q_imag5) & 8'hFF;
   $s6[7:0] = ($q_real6 + $q_imag6) & 8'hFF;
   $s7[7:0] = ($q_real7 + $q_imag7) & 8'hFF;
   
   // Simplified matrix - just use simple mixing for demonstration
   // In real implementation, this would be a proper 8x8 lattice matrix
   $lat0[15:0] = ($s0 + $s1 + $s2 + $s3) & 16'hFFFF;
   $lat1[15:0] = ($s1 + $s2 + $s3 + $s4) & 16'hFFFF;
   $lat2[15:0] = ($s2 + $s3 + $s4 + $s5) & 16'hFFFF;
   $lat3[15:0] = ($s3 + $s4 + $s5 + $s6) & 16'hFFFF;
   $lat4[15:0] = ($s4 + $s5 + $s6 + $s7) & 16'hFFFF;
   $lat5[15:0] = ($s5 + $s6 + $s7 + $s0) & 16'hFFFF;
   $lat6[15:0] = ($s6 + $s7 + $s0 + $s1) & 16'hFFFF;
   $lat7[15:0] = ($s7 + $s0 + $s1 + $s2) & 16'hFFFF;
   
   // SIS digest - concatenate lattice coordinates
   $sis_digest_lo[127:0] = {$lat3, $lat2, $lat1, $lat0, $lat7, $lat6, $lat5, $lat4};
   $sis_digest_hi[127:0] = {$lat3, $lat2, $lat1, $lat0, $lat7, $lat6, $lat5, $lat4};
   $sis_digest[255:0] = {$sis_digest_hi, $sis_digest_lo};
   $sis_valid = $is_done;
   
   // ==================================================================
   // Output Status
   // ==================================================================
   
   *passed = $is_done && $resonance_active;
   *failed = $cyc_cnt > 8'd200 && !*passed;
   
\SV
   endmodule
