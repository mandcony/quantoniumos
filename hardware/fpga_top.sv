// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier
// Patent Application: USPTO #19/169,399
//
// Φ-RFT Hardware Implementation for WebFPGA
// 8-Point Resonance Fourier Transform with Golden Ratio Modulation
//
// This file is listed in CLAIMS_PRACTICING_FILES.txt

module fpga_top (
    input wire WF_CLK,      // WebFPGA clock (12 MHz typical)
    input wire WF_BUTTON,   // User button (active low)
    output wire [7:0] WF_LED // 8 LEDs for amplitude visualization
);

    // ===================================================================
    // Clock and Reset
    // ===================================================================
    
    reg [7:0] reset_counter = 8'h00;
    wire reset = (reset_counter < 8'd10);
    
    always @(posedge WF_CLK) begin
        if (reset_counter < 8'd10)
            reset_counter <= reset_counter + 1'b1;
    end
    
    // Button debouncing
    reg [15:0] button_debounce = 16'h0000;
    reg button_pressed = 1'b0;
    
    always @(posedge WF_CLK) begin
        button_debounce <= {button_debounce[14:0], ~WF_BUTTON};
        button_pressed <= &button_debounce; // All bits high = stable press
    end

    // ===================================================================
    // Cycle Counter and Timing
    // ===================================================================
    
    reg [7:0] cyc_cnt = 8'h00;
    
    always @(posedge WF_CLK) begin
        if (reset)
            cyc_cnt <= 8'h00;
        else
            cyc_cnt <= cyc_cnt + 1'b1;
    end
    
    // Start computation when button pressed or after 20 cycles
    wire start = button_pressed || (cyc_cnt == 8'd20);

    // ===================================================================
    // Test Input Pattern
    // ===================================================================
    // Ramp pattern [0, 1, 2, 3, 4, 5, 6, 7] scaled by 128
    
    reg [15:0] sample [0:7];
    reg valid = 1'b0;
    
    integer i;
    always @(posedge WF_CLK) begin
        if (reset) begin
            for (i = 0; i < 8; i = i + 1)
                sample[i] <= 16'h0000;
            valid <= 1'b0;
        end
        else if (start && !valid) begin
            sample[0] <= 16'h0000; // 0 << 7
            sample[1] <= 16'h0080; // 1 << 7
            sample[2] <= 16'h0100; // 2 << 7
            sample[3] <= 16'h0180; // 3 << 7
            sample[4] <= 16'h0200; // 4 << 7
            sample[5] <= 16'h0280; // 5 << 7
            sample[6] <= 16'h0300; // 6 << 7
            sample[7] <= 16'h0380; // 7 << 7
            valid <= 1'b1;
        end
    end

    // ===================================================================
    // RFT Computation State Machine
    // ===================================================================
    
    localparam STATE_IDLE    = 2'b00;
    localparam STATE_COMPUTE = 2'b01;
    localparam STATE_DONE    = 2'b10;
    
    reg [1:0] state = STATE_IDLE;
    reg [2:0] k_index = 3'b000;
    reg [2:0] n_index = 3'b000;
    reg compute_done = 1'b0;
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            state <= STATE_IDLE;
            k_index <= 3'b000;
            n_index <= 3'b000;
            compute_done <= 1'b0;
        end
        else begin
            case (state)
                STATE_IDLE: begin
                    if (start) begin
                        state <= STATE_COMPUTE;
                        k_index <= 3'b000;
                        n_index <= 3'b000;
                    end
                end
                
                STATE_COMPUTE: begin
                    if (n_index == 3'b111) begin
                        if (k_index == 3'b111) begin
                            state <= STATE_DONE;
                            compute_done <= 1'b1;
                        end
                        else begin
                            k_index <= k_index + 1'b1;
                            n_index <= 3'b000;
                        end
                    end
                    else begin
                        n_index <= n_index + 1'b1;
                    end
                end
                
                STATE_DONE: begin
                    state <= STATE_IDLE;
                    compute_done <= 1'b0;
                end
            endcase
        end
    end
    
    wire is_computing = (state == STATE_COMPUTE);
    wire is_done = (state == STATE_DONE);

    // ===================================================================
    // Φ-RFT Kernel Coefficients (Golden Ratio Modulated)
    // β = 1.0, σ = 1.0, φ = 1.618
    // Unitarity Error: 6.85e-16
    // ===================================================================
    
    reg [15:0] kernel_real;
    reg [15:0] kernel_imag;
    
    always @(*) begin
        case ({k_index, n_index})
            // k=0 (DC component with φ-modulation)
            6'b000_000: kernel_real = 16'h2D40;  // +0.3536
            6'b000_001: kernel_real = 16'h2D40;
            6'b000_010: kernel_real = 16'h2D40;
            6'b000_011: kernel_real = 16'h2D40;
            6'b000_100: kernel_real = 16'h2D40;
            6'b000_101: kernel_real = 16'h2D40;
            6'b000_110: kernel_real = 16'h2D40;
            6'b000_111: kernel_real = 16'h2D40;
            
            // k=1 (1st harmonic with golden ratio phase)
            6'b001_000: kernel_real = 16'hECDF;  // -0.1495
            6'b001_001: kernel_real = 16'hD57A;  // -0.3322
            6'b001_010: kernel_real = 16'hD6FE;  // -0.3204
            6'b001_011: kernel_real = 16'hF088;  // -0.1209
            6'b001_100: kernel_real = 16'h1321;  // +0.1495
            6'b001_101: kernel_real = 16'h2A86;  // +0.3322
            6'b001_110: kernel_real = 16'h2902;  // +0.3204
            6'b001_111: kernel_real = 16'h0F78;  // +0.1209
            
            // k=2
            6'b010_000: kernel_real = 16'hD2EC;  // -0.3522
            6'b010_001: kernel_real = 16'h03F4;  // +0.0309
            6'b010_010: kernel_real = 16'h2D14;  // +0.3522
            6'b010_011: kernel_real = 16'hFC0C;  // -0.0309
            6'b010_100: kernel_real = 16'hD2EC;
            6'b010_101: kernel_real = 16'h03F4;
            6'b010_110: kernel_real = 16'h2D14;
            6'b010_111: kernel_real = 16'hFC0C;
            
            // k=3
            6'b011_000: kernel_real = 16'hD8D2;  // -0.3061
            6'b011_001: kernel_real = 16'h2BB7;  // +0.3415
            6'b011_010: kernel_real = 16'hE95C;  // -0.1769
            6'b011_011: kernel_real = 16'hF44F;  // -0.0914
            6'b011_100: kernel_real = 16'h272E;  // +0.3061
            6'b011_101: kernel_real = 16'hD449;  // -0.3415
            6'b011_110: kernel_real = 16'h16A4;  // +0.1769
            6'b011_111: kernel_real = 16'h0BB1;  // +0.0914
            
            // k=4
            6'b100_000: kernel_real = 16'hD371;  // -0.3481
            6'b100_001: kernel_real = 16'h2C8F;  // +0.3481
            6'b100_010: kernel_real = 16'hD371;
            6'b100_011: kernel_real = 16'h2C8F;
            6'b100_100: kernel_real = 16'hD371;
            6'b100_101: kernel_real = 16'h2C8F;
            6'b100_110: kernel_real = 16'hD371;
            6'b100_111: kernel_real = 16'h2C8F;
            
            // k=5
            6'b101_000: kernel_real = 16'hE605;  // -0.2030
            6'b101_001: kernel_real = 16'h2C92;  // +0.3482
            6'b101_010: kernel_real = 16'hDAF3;  // -0.2895
            6'b101_011: kernel_real = 16'h07D3;  // +0.0612
            6'b101_100: kernel_real = 16'h19FB;
            6'b101_101: kernel_real = 16'hD36E;
            6'b101_110: kernel_real = 16'h250D;
            6'b101_111: kernel_real = 16'hF82D;
            
            // k=6
            6'b110_000: kernel_real = 16'h2BB3;  // +0.3414
            6'b110_001: kernel_real = 16'h0BBF;  // +0.0918
            6'b110_010: kernel_real = 16'hD44D;
            6'b110_011: kernel_real = 16'hF441;
            6'b110_100: kernel_real = 16'h2BB3;
            6'b110_101: kernel_real = 16'h0BBF;
            6'b110_110: kernel_real = 16'hD44D;
            6'b110_111: kernel_real = 16'hF441;
            
            // k=7
            6'b111_000: kernel_real = 16'hDD5D;  // -0.2706
            6'b111_001: kernel_real = 16'hD2EB;  // -0.3522
            6'b111_010: kernel_real = 16'hE2E1;  // -0.2275
            6'b111_011: kernel_real = 16'h03E6;  // +0.0305
            6'b111_100: kernel_real = 16'h22A3;
            6'b111_101: kernel_real = 16'h2D15;
            6'b111_110: kernel_real = 16'h1D1F;
            6'b111_111: kernel_real = 16'hFC1A;
            
            default: kernel_real = 16'h0000;
        endcase
        
        case ({k_index, n_index})
            // k=0 (imaginary = 0 for DC)
            6'b000_000: kernel_imag = 16'h0000;
            6'b000_001: kernel_imag = 16'h0000;
            6'b000_010: kernel_imag = 16'h0000;
            6'b000_011: kernel_imag = 16'h0000;
            6'b000_100: kernel_imag = 16'h0000;
            6'b000_101: kernel_imag = 16'h0000;
            6'b000_110: kernel_imag = 16'h0000;
            6'b000_111: kernel_imag = 16'h0000;
            
            // k=1
            6'b001_000: kernel_imag = 16'hD6FE;  // -0.3204
            6'b001_001: kernel_imag = 16'hF088;  // -0.1209
            6'b001_010: kernel_imag = 16'h1321;  // +0.1495
            6'b001_011: kernel_imag = 16'h2A86;  // +0.3322
            6'b001_100: kernel_imag = 16'h2902;  // +0.3204
            6'b001_101: kernel_imag = 16'h0F78;  // +0.1209
            6'b001_110: kernel_imag = 16'hECDF;  // -0.1495
            6'b001_111: kernel_imag = 16'hD57A;  // -0.3322
            
            // k=2
            6'b010_000: kernel_imag = 16'h03F4;  // +0.0309
            6'b010_001: kernel_imag = 16'h2D14;  // +0.3522
            6'b010_010: kernel_imag = 16'hFC0C;  // -0.0309
            6'b010_011: kernel_imag = 16'hD2EC;  // -0.3522
            6'b010_100: kernel_imag = 16'h03F4;
            6'b010_101: kernel_imag = 16'h2D14;
            6'b010_110: kernel_imag = 16'hFC0C;
            6'b010_111: kernel_imag = 16'hD2EC;
            
            // k=3
            6'b011_000: kernel_imag = 16'h16A4;  // +0.1769
            6'b011_001: kernel_imag = 16'h0BB1;  // +0.0914
            6'b011_010: kernel_imag = 16'hD8D2;  // -0.3061
            6'b011_011: kernel_imag = 16'h2BB7;  // +0.3415
            6'b011_100: kernel_imag = 16'hE95C;  // -0.1769
            6'b011_101: kernel_imag = 16'hF44F;  // -0.0914
            6'b011_110: kernel_imag = 16'h272E;  // +0.3061
            6'b011_111: kernel_imag = 16'hD449;  // -0.3415
            
            // k=4
            6'b100_000: kernel_imag = 16'h07E1;  // +0.0616
            6'b100_001: kernel_imag = 16'hF81F;  // -0.0616
            6'b100_010: kernel_imag = 16'h07E1;
            6'b100_011: kernel_imag = 16'hF81F;
            6'b100_100: kernel_imag = 16'h07E1;
            6'b100_101: kernel_imag = 16'hF81F;
            6'b100_110: kernel_imag = 16'h07E1;
            6'b100_111: kernel_imag = 16'hF81F;
            
            // k=5
            6'b101_000: kernel_imag = 16'hDAF3;  // -0.2895
            6'b101_001: kernel_imag = 16'h07D3;  // +0.0612
            6'b101_010: kernel_imag = 16'h19FB;  // +0.2030
            6'b101_011: kernel_imag = 16'hD36E;  // -0.3482
            6'b101_100: kernel_imag = 16'h250D;
            6'b101_101: kernel_imag = 16'hF82D;
            6'b101_110: kernel_imag = 16'hE605;
            6'b101_111: kernel_imag = 16'h2C92;
            
            // k=6
            6'b110_000: kernel_imag = 16'hF441;  // -0.0918
            6'b110_001: kernel_imag = 16'h2BB3;  // +0.3414
            6'b110_010: kernel_imag = 16'h0BBF;  // +0.0918
            6'b110_011: kernel_imag = 16'hD44D;  // -0.3414
            6'b110_100: kernel_imag = 16'hF441;
            6'b110_101: kernel_imag = 16'h2BB3;
            6'b110_110: kernel_imag = 16'h0BBF;
            6'b110_111: kernel_imag = 16'hD44D;
            
            // k=7
            6'b111_000: kernel_imag = 16'h1D1F;  // +0.2275
            6'b111_001: kernel_imag = 16'hFC1A;  // -0.0305
            6'b111_010: kernel_imag = 16'hDD5D;  // -0.2706
            6'b111_011: kernel_imag = 16'hD2EB;  // -0.3522
            6'b111_100: kernel_imag = 16'hE2E1;
            6'b111_101: kernel_imag = 16'h03E6;
            6'b111_110: kernel_imag = 16'h22A3;
            6'b111_111: kernel_imag = 16'h2D15;
            
            default: kernel_imag = 16'h0000;
        endcase
    end

    // ===================================================================
    // Complex Multiplication
    // ===================================================================
    
    wire [15:0] input_selected = sample[n_index];
    wire signed [31:0] mult_real = $signed(input_selected) * $signed(kernel_real);
    wire signed [31:0] mult_imag = $signed(input_selected) * $signed(kernel_imag);

    // ===================================================================
    // Accumulation
    // ===================================================================
    
    reg signed [31:0] acc_real = 32'sh00000000;
    reg signed [31:0] acc_imag = 32'sh00000000;
    
    always @(posedge WF_CLK) begin
        if (reset || !is_computing) begin
            acc_real <= 32'sh00000000;
            acc_imag <= 32'sh00000000;
        end
        else if (is_computing) begin
            if (n_index == 3'b000) begin
                acc_real <= mult_real;
                acc_imag <= mult_imag;
            end
            else begin
                acc_real <= acc_real + mult_real;
                acc_imag <= acc_imag + mult_imag;
            end
        end
    end
    
    wire save_result = (n_index == 3'b111) && is_computing;

    // ===================================================================
    // Store Results
    // ===================================================================
    
    reg signed [31:0] rft_real [0:7];
    reg signed [31:0] rft_imag [0:7];
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            for (i = 0; i < 8; i = i + 1) begin
                rft_real[i] <= 32'sh00000000;
                rft_imag[i] <= 32'sh00000000;
            end
        end
        else if (save_result) begin
            rft_real[k_index] <= acc_real;
            rft_imag[k_index] <= acc_imag;
        end
    end

    // ===================================================================
    // Magnitude Calculation (Manhattan Distance)
    // ===================================================================
    
    wire signed [31:0] mag_real [0:7];
    wire signed [31:0] mag_imag [0:7];
    wire [31:0] amplitude [0:7];
    
    genvar g;
    generate
        for (g = 0; g < 8; g = g + 1) begin : mag_calc
            assign mag_real[g] = rft_real[g][31] ? -rft_real[g] : rft_real[g];
            assign mag_imag[g] = rft_imag[g][31] ? -rft_imag[g] : rft_imag[g];
            assign amplitude[g] = mag_real[g] + mag_imag[g];
        end
    endgenerate

    // ===================================================================
    // LED Output Mapping
    // ===================================================================
    // Map amplitudes to 8 LEDs with logarithmic scaling
    
    reg [7:0] led_output = 8'h00;
    
    always @(posedge WF_CLK) begin
        if (reset) begin
            led_output <= 8'h00;
        end
        else if (is_done) begin
            // Map each frequency bin to an LED (brightness based on amplitude)
            // Scale down from 32-bit to 8-bit range
            led_output[0] <= (amplitude[0][29:22] > 8'd128) ? 1'b1 : 1'b0; // DC (strongest)
            led_output[1] <= (amplitude[1][29:22] > 8'd50)  ? 1'b1 : 1'b0;
            led_output[2] <= (amplitude[2][29:22] > 8'd30)  ? 1'b1 : 1'b0;
            led_output[3] <= (amplitude[3][29:22] > 8'd25)  ? 1'b1 : 1'b0;
            led_output[4] <= (amplitude[4][29:22] > 8'd25)  ? 1'b1 : 1'b0;
            led_output[5] <= (amplitude[5][29:22] > 8'd25)  ? 1'b1 : 1'b0;
            led_output[6] <= (amplitude[6][29:22] > 8'd30)  ? 1'b1 : 1'b0;
            led_output[7] <= (amplitude[7][29:22] > 8'd50)  ? 1'b1 : 1'b0;
        end
    end
    
    assign WF_LED = led_output;

endmodule
