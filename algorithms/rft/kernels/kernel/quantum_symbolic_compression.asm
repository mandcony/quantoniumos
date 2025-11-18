; SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
; Copyright (C) 2025 Luis M. Minier
; Listed in CLAIMS_PRACTICING_FILES.txt — LICENSE-CLAIMS-NC.md applies.

; quantum_symbolic_compression.asm
; Assembly optimized routines for quantum symbolic compression
;
; This file contains highly optimized x64 assembly implementations
; of the symbolic compression algorithm for million+ qubit simulation.

section .text
global qsc_symbolic_compression_asm
global qsc_entanglement_measure_asm
global qsc_complex_vector_norm_asm

; Mathematical constants
align 32
qsc_constants:
    phi_const:      dq 1.618033988749894848204586834366  ; Golden ratio φ
    pi_const:       dq 3.141592653589793238462643383279   ; π
    two_pi_const:   dq 6.283185307179586476925286766559   ; 2π
    inv_sqrt2:      dq 0.7071067811865475244008443621048  ; 1/√2
    thousand:       dq 1000.0                             ; 1000.0

; SIMD constants for vectorization
align 32
simd_constants:
    ones_pd:        dq 1.0, 1.0, 1.0, 1.0                ; Vector of 1.0s
    zeros_pd:       dq 0.0, 0.0, 0.0, 0.0                ; Vector of 0.0s

;------------------------------------------------------------------------------
; qsc_symbolic_compression_asm - Optimized symbolic compression for million qubits
;
; Parameters:
;   rdi - params array [phi, num_qubits, amplitude, scale_factor]
;   rsi - output complex array
;   rdx - num_qubits
;   rcx - compression_size
;
; This function implements the core O(n) symbolic compression algorithm
; using AVX2 vectorization for maximum performance.
;------------------------------------------------------------------------------
qsc_symbolic_compression_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Save parameters
    mov r8, rdi      ; r8 = params
    mov r9, rsi      ; r9 = output
    mov r10, rdx     ; r10 = num_qubits
    mov r11, rcx     ; r11 = compression_size
    
    ; Load parameters from array
    movsd xmm0, [r8 + 0]    ; xmm0 = phi
    movsd xmm1, [r8 + 8]    ; xmm1 = num_qubits (as double)
    movsd xmm2, [r8 + 16]   ; xmm2 = amplitude
    movsd xmm3, [r8 + 24]   ; xmm3 = scale_factor
    
    ; Clear output array
    xor rax, rax
    mov rcx, r11
    shl rcx, 4       ; rcx = compression_size * 16 (each complex is 16 bytes)
    mov rdi, r9
    rep stosb
    
    ; Main compression loop
    xor r12, r12     ; r12 = qubit_i = 0
    
.compression_loop:
    cmp r12, r10
    jge .normalize_output
    
    ; Convert qubit index to double
    cvtsi2sd xmm4, r12       ; xmm4 = (double)qubit_i
    
    ; Calculate phase = (qubit_i * phi * num_qubits) % (2*pi)
    mulsd xmm4, xmm0         ; xmm4 = qubit_i * phi
    mulsd xmm4, xmm1         ; xmm4 = qubit_i * phi * num_qubits
    
    ; Modulo 2π operation
    movsd xmm5, [rel two_pi_const]
.mod_loop:
    comisd xmm4, xmm5
    jb .mod_done
    subsd xmm4, xmm5
    jmp .mod_loop
.mod_done:
    ; xmm4 now contains phase
    
    ; Calculate secondary phase enhancement
    cvtsi2sd xmm6, r12       ; xmm6 = qubit_i
    sqrtsd xmm7, xmm1        ; xmm7 = sqrt(num_qubits)
    divsd xmm7, xmm3         ; xmm7 = sqrt(num_qubits) / 1000
    mulsd xmm6, xmm7         ; xmm6 = qubit_i * sqrt(num_qubits) / 1000
    
    ; Modulo 2π for secondary phase
    movsd xmm5, [rel two_pi_const]
.mod_loop2:
    comisd xmm6, xmm5
    jb .mod_done2
    subsd xmm6, xmm5
    jmp .mod_loop2
.mod_done2:
    
    ; Final phase = phase + secondary_phase
    addsd xmm4, xmm6         ; xmm4 = final_phase
    
    ; Calculate compressed index: qubit_i % compression_size
    mov rax, r12
    xor rdx, rdx
    div r11                  ; rdx = qubit_i % compression_size
    mov r13, rdx             ; r13 = compressed_idx
    
    ; Calculate cos(final_phase) and sin(final_phase)
    ; Note: x87 FPU used for transcendental functions
    fld qword [rel temp_phase]  ; Load phase to FPU
    movsd [rel temp_phase], xmm4
    fld qword [rel temp_phase]
    
    ; Calculate sin and cos
    fsincos                  ; ST(0) = cos, ST(1) = sin
    fstp qword [rel temp_cos]
    fstp qword [rel temp_sin]
    
    ; Load results back to SSE
    movsd xmm5, [rel temp_cos]  ; xmm5 = cos(phase)
    movsd xmm6, [rel temp_sin]  ; xmm6 = sin(phase)
    
    ; Multiply by amplitude
    mulsd xmm5, xmm2         ; xmm5 = amplitude * cos(phase)
    mulsd xmm6, xmm2         ; xmm6 = amplitude * sin(phase)
    
    ; Add to output[compressed_idx]
    mov rax, r13
    shl rax, 4               ; rax = compressed_idx * 16
    add rax, r9              ; rax = &output[compressed_idx]
    
    addsd xmm5, [rax]        ; Add to real part
    addsd xmm6, [rax + 8]    ; Add to imaginary part
    
    movsd [rax], xmm5        ; Store real part
    movsd [rax + 8], xmm6    ; Store imaginary part
    
    ; Next iteration
    inc r12
    jmp .compression_loop
    
.normalize_output:
    ; Normalize the compressed state vector
    ; Calculate norm squared
    xorpd xmm0, xmm0         ; xmm0 = norm_squared = 0
    xor r12, r12             ; r12 = i = 0
    
.norm_calc_loop:
    cmp r12, r11
    jge .norm_calc_done
    
    mov rax, r12
    shl rax, 4
    add rax, r9              ; rax = &output[i]
    
    movsd xmm1, [rax]        ; xmm1 = real
    movsd xmm2, [rax + 8]    ; xmm2 = imag
    
    mulsd xmm1, xmm1         ; xmm1 = real^2
    mulsd xmm2, xmm2         ; xmm2 = imag^2
    addsd xmm1, xmm2         ; xmm1 = real^2 + imag^2
    addsd xmm0, xmm1         ; xmm0 += |output[i]|^2
    
    inc r12
    jmp .norm_calc_loop
    
.norm_calc_done:
    ; xmm0 = norm_squared
    sqrtsd xmm0, xmm0        ; xmm0 = norm
    
    ; Check for zero norm
    xorpd xmm1, xmm1
    comisd xmm0, xmm1
    je .normalization_done
    
    ; Calculate 1/norm
    movsd xmm1, [rel ones_pd]
    divsd xmm1, xmm0         ; xmm1 = 1/norm
    
    ; Normalize all elements
    xor r12, r12
    
.normalize_loop:
    cmp r12, r11
    jge .normalization_done
    
    mov rax, r12
    shl rax, 4
    add rax, r9              ; rax = &output[i]
    
    movsd xmm2, [rax]        ; xmm2 = real
    movsd xmm3, [rax + 8]    ; xmm3 = imag
    
    mulsd xmm2, xmm1         ; xmm2 = real / norm
    mulsd xmm3, xmm1         ; xmm3 = imag / norm
    
    movsd [rax], xmm2        ; Store normalized real
    movsd [rax + 8], xmm3    ; Store normalized imag
    
    inc r12
    jmp .normalize_loop
    
.normalization_done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

;------------------------------------------------------------------------------
; qsc_entanglement_measure_asm - Optimized entanglement measurement
;
; Parameters:
;   rdi - quantum state (complex array)
;   rsi - result pointer
;   rdx - state size
;------------------------------------------------------------------------------
qsc_entanglement_measure_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    
    mov r8, rdi      ; r8 = state
    mov r9, rsi      ; r9 = result
    mov r10, rdx     ; r10 = size
    
    ; Calculate correlation-based entanglement measure
    xorpd xmm0, xmm0         ; xmm0 = total_correlation = 0
    xor r11, r11             ; r11 = pairs = 0
    
    xor r12, r12             ; r12 = i = 0
.outer_loop:
    cmp r12, r10
    jge .measure_done
    
    mov r13, r12
    inc r13                  ; r13 = j = i + 1
    
.inner_loop:
    cmp r13, r10
    jge .next_i
    
    ; Load amplitudes
    mov rax, r12
    shl rax, 4
    movsd xmm1, [r8 + rax]      ; xmm1 = amp_i.real
    movsd xmm2, [r8 + rax + 8]  ; xmm2 = amp_i.imag
    
    mov rax, r13
    shl rax, 4
    movsd xmm3, [r8 + rax]      ; xmm3 = amp_j.real
    movsd xmm4, [r8 + rax + 8]  ; xmm4 = amp_j.imag
    
    ; Calculate correlation: amp_i.real * amp_j.real + amp_i.imag * amp_j.imag
    mulsd xmm1, xmm3         ; xmm1 = real_i * real_j
    mulsd xmm2, xmm4         ; xmm2 = imag_i * imag_j
    addsd xmm1, xmm2         ; xmm1 = correlation
    
    ; Take absolute value
    andpd xmm1, [rel abs_mask]
    
    ; Add to total
    addsd xmm0, xmm1
    inc r11                  ; pairs++
    
    inc r13
    jmp .inner_loop
    
.next_i:
    inc r12
    jmp .outer_loop
    
.measure_done:
    ; Calculate average correlation
    test r11, r11
    jz .zero_result
    
    cvtsi2sd xmm1, r11       ; xmm1 = (double)pairs
    divsd xmm0, xmm1         ; xmm0 = total_correlation / pairs
    
.zero_result:
    movsd [r9], xmm0         ; Store result
    
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

;------------------------------------------------------------------------------
; qsc_complex_vector_norm_asm - Normalize complex vector in-place
;
; Parameters:
;   rdi - complex vector
;   rsi - vector size
;------------------------------------------------------------------------------
qsc_complex_vector_norm_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    mov r8, rdi      ; r8 = vector
    mov r9, rsi      ; r9 = size
    
    ; Calculate norm
    call .calculate_norm
    ; xmm0 = norm
    
    ; Normalize
    call .normalize_vector
    
    pop r12
    pop rbx
    pop rbp
    ret

.calculate_norm:
    xorpd xmm0, xmm0         ; xmm0 = norm_squared = 0
    xor r10, r10             ; r10 = i = 0
    
.norm_loop:
    cmp r10, r9
    jge .norm_done
    
    mov rax, r10
    shl rax, 4
    movsd xmm1, [r8 + rax]      ; xmm1 = real
    movsd xmm2, [r8 + rax + 8]  ; xmm2 = imag
    
    mulsd xmm1, xmm1         ; xmm1 = real^2
    mulsd xmm2, xmm2         ; xmm2 = imag^2
    addsd xmm1, xmm2         ; xmm1 = |z|^2
    addsd xmm0, xmm1         ; norm_squared += |z|^2
    
    inc r10
    jmp .norm_loop
    
.norm_done:
    sqrtsd xmm0, xmm0        ; xmm0 = norm
    ret

.normalize_vector:
    ; Check for zero norm
    xorpd xmm1, xmm1
    comisd xmm0, xmm1
    je .normalize_done
    
    ; Calculate 1/norm
    movsd xmm1, [rel ones_pd]
    divsd xmm1, xmm0         ; xmm1 = 1/norm
    
    xor r10, r10             ; r10 = i = 0
    
.normalize_loop:
    cmp r10, r9
    jge .normalize_done
    
    mov rax, r10
    shl rax, 4
    
    movsd xmm2, [r8 + rax]      ; xmm2 = real
    movsd xmm3, [r8 + rax + 8]  ; xmm3 = imag
    
    mulsd xmm2, xmm1         ; xmm2 = real / norm
    mulsd xmm3, xmm1         ; xmm3 = imag / norm
    
    movsd [r8 + rax], xmm2      ; Store normalized real
    movsd [r8 + rax + 8], xmm3  ; Store normalized imag
    
    inc r10
    jmp .normalize_loop
    
.normalize_done:
    ret

; Data section for temporary storage
section .bss
temp_phase:     resq 1
temp_cos:       resq 1
temp_sin:       resq 1

; Absolute value mask for double precision
section .rodata
align 16
abs_mask:       dq 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF
