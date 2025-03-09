; N-body simulation force calculation in x86_64 NASM assembly
; Implements the calc_force function from the original C code
; Function signature: void calc_force_asm(Particle* particles, int num, int start, int stop, double* accelerations)

section .text
global calc_force_asm

; Register usage:
; rdi = Particle* particles (first argument)
; rsi = int num (second argument)
; rdx = int start (third argument)
; rcx = int stop (fourth argument)
; r8 = double* accelerations (fifth argument)

calc_force_asm:
    ; Function prologue - save non-volatile registers
    push rbp
    push r15
    push r14
    push r13
    push r12
    push rbx
    sub rsp, 128                 ; Allocate stack space for temporary variables (aligned to 16 bytes)

    ; Load arguments
    mov r12, rdi                 ; r12 = particles
    mov r13, rsi                 ; r13 = num
    mov r14, rdx                 ; r14 = start
    mov r15, rcx                 ; r15 = stop
    mov rbx, r8                  ; rbx = accelerations
    
    ; Check if start >= stop
    cmp r14, r15
    jge .end_outer_loop
    
    ; Load constants
    vbroadcastsd ymm15, [rel min_dist]   ; ymm15 = 0.01 (minimum distance)
    vbroadcastsd ymm14, [rel gravconst]  ; ymm14 = 0.001 (gravitational constant)
    
    ; Load pointers to particle arrays
    mov rdi, [r12]               ; rdi = particles->mass
    mov rsi, [r12 + 8]           ; rsi = particles->old_x
    mov rdx, [r12 + 16]          ; rdx = particles->old_y
    mov rcx, [r12 + 24]          ; rcx = particles->old_z
    mov r8, [r12 + 32]           ; r8 = particles->x
    mov r9, [r12 + 40]           ; r9 = particles->y
    mov r10, [r12 + 48]          ; r10 = particles->z

.outer_loop:
    ; Check if we've reached the end
    cmp r14, r15
    jge .end_outer_loop
    
    ; Load particle i position and broadcast to all elements
    vmovsd xmm0, [r8 + r14*8]     ; xmm0 = x_i
    vbroadcastsd ymm0, xmm0       ; ymm0 = x_i (broadcast to all elements)
    
    vmovsd xmm1, [r9 + r14*8]     ; xmm1 = y_i
    vbroadcastsd ymm1, xmm1       ; ymm1 = y_i
    
    vmovsd xmm2, [r10 + r14*8]    ; xmm2 = z_i
    vbroadcastsd ymm2, xmm2       ; ymm2 = z_i
    
    ; Load particle i+1 position (if i+1 < stop)
    mov rax, r14
    add rax, 1
    cmp rax, r15
    jge .skip_i_plus_1
    
    vmovsd xmm3, [r8 + rax*8]     ; xmm3 = x_i+1
    vbroadcastsd ymm3, xmm3       ; ymm3 = x_i+1
    
    vmovsd xmm4, [r9 + rax*8]     ; xmm4 = y_i+1
    vbroadcastsd ymm4, xmm4       ; ymm4 = y_i+1
    
    vmovsd xmm5, [r10 + rax*8]    ; xmm5 = z_i+1
    vbroadcastsd ymm5, xmm5       ; ymm5 = z_i+1
    
.skip_i_plus_1:
    ; Initialize loop counter
    xor rbp, rbp                  ; rbp = j = 0
    
    ; Calculate indices for acceleration array
    mov rax, r14
    imul rax, 3                   ; rax = i*3
    
    ; Initialize acceleration accumulators for particle i
    vxorpd ymm8, ymm8, ymm8       ; ymm8 = 0 (acceleration x)
    vxorpd ymm9, ymm9, ymm9       ; ymm9 = 0 (acceleration y)
    vxorpd ymm10, ymm10, ymm10    ; ymm10 = 0 (acceleration z)
    
    ; Initialize acceleration accumulators for particle i+1
    vxorpd ymm11, ymm11, ymm11    ; ymm11 = 0 (acceleration x)
    vxorpd ymm12, ymm12, ymm12    ; ymm12 = 0 (acceleration y)
    vxorpd ymm13, ymm13, ymm13    ; ymm13 = 0 (acceleration z)

    ; ymm6 - 7 are free

.inner_loop:
    ; Check if we've processed all particles
    cmp rbp, r13
    jge .end_inner_loop
    
    ; Load 4 particles at once (j, j+1, j+2, j+3)
    vmovupd ymm6, [rsi + rbp*8]   ; ymm6 = old_x[j:j+3]
    vmovupd ymm7, [rdx + rbp*8]   ; ymm7 = old_y[j:j+3]
    vmovupd ymm8, [rcx + rbp*8]   ; ymm8 = old_z[j:j+3]
    vmovupd ymm9, [rdi + rbp*8]   ; ymm9 = mass[j:j+3]
    
    ; Calculate dx, dy, dz for particle i
    vsubpd ymm10, ymm6, ymm0      ; ymm10 = dx = old_x[j:j+3] - x_i
    vsubpd ymm11, ymm7, ymm1      ; ymm11 = dy = old_y[j:j+3] - y_i
    vsubpd ymm12, ymm8, ymm2      ; ymm12 = dz = old_z[j:j+3] - z_i
    
    ; Calculate distance squared for particle i
    vmulpd ymm13, ymm10, ymm10    ; ymm13 = dx*dx
    vfmadd231pd ymm13, ymm11, ymm11 ; ymm13 += dy*dy
    vfmadd231pd ymm13, ymm12, ymm12 ; ymm13 += dz*dz
    
    ; Calculate distance for particle i
    vsqrtpd ymm13, ymm13          ; ymm13 = sqrt(dx*dx + dy*dy + dz*dz)
    vmaxpd ymm13, ymm13, ymm15    ; ymm13 = max(sqrt_d2, 0.01)
    
    ; Calculate d^3 for particle i
    vmulpd ymm0, ymm13, ymm13     ; ymm0 = d^2
    vmulpd ymm0, ymm0, ymm13      ; ymm0 = d^3
    
    ; Calculate 1/d^3 * G for particle i
    vdivpd ymm0, ymm14, ymm0      ; ymm0 = G/d^3
    
    ; Calculate force factors
    vmulpd ymm0, ymm0, ymm9       ; ymm0 = mass[j:j+3] * G/d^3
    
    ; Calculate acceleration components for particle i
    vmulpd ymm6, ymm0, ymm10      ; ymm6 = dx * force_factor
    vmulpd ymm7, ymm0, ymm11      ; ymm7 = dy * force_factor
    vmulpd ymm8, ymm0, ymm12      ; ymm8 = dz * force_factor
    
    ; Horizontal sum for acceleration x
    vextractf128 xmm0, ymm6, 1    ; Extract high 128 bits
    vaddpd xmm0, xmm0, xmm6       ; Add high and low 128 bits
    vhaddpd xmm0, xmm0, xmm0      ; Horizontal add within 128 bits
    
    ; Add to acceleration array for particle i, x component
    vmovsd xmm1, [rbx + rax*8]    ; xmm1 = accelerations[i*3+0]
    vaddsd xmm0, xmm0, xmm1       ; xmm0 += accelerations[i*3+0]
    vmovsd [rbx + rax*8], xmm0    ; accelerations[i*3+0] = xmm0
    
    ; Horizontal sum for acceleration y
    vextractf128 xmm0, ymm7, 1    ; Extract high 128 bits
    vaddpd xmm0, xmm0, xmm7       ; Add high and low 128 bits
    vhaddpd xmm0, xmm0, xmm0      ; Horizontal add within 128 bits
    
    ; Add to acceleration array for particle i, y component
    vmovsd xmm1, [rbx + rax*8 + 8] ; xmm1 = accelerations[i*3+1]
    vaddsd xmm0, xmm0, xmm1       ; xmm0 += accelerations[i*3+1]
    vmovsd [rbx + rax*8 + 8], xmm0 ; accelerations[i*3+1] = xmm0
    
    ; Horizontal sum for acceleration z
    vextractf128 xmm0, ymm8, 1    ; Extract high 128 bits
    vaddpd xmm0, xmm0, xmm8       ; Add high and low 128 bits
    vhaddpd xmm0, xmm0, xmm0      ; Horizontal add within 128 bits
    
    ; Add to acceleration array for particle i, z component
    vmovsd xmm1, [rbx + rax*8 + 16] ; xmm1 = accelerations[i*3+2]
    vaddsd xmm0, xmm0, xmm1       ; xmm0 += accelerations[i*3+2]
    vmovsd [rbx + rax*8 + 16], xmm0 ; accelerations[i*3+2] = xmm0
    
    ; Check if i+1 < stop
    mov r11, r14
    add r11, 1
    cmp r11, r15
    jge .skip_i_plus_1_calc
    
    ; Calculate dx, dy, dz for particle i+1
    vmovupd ymm6, [rsi + rbp*8]   ; ymm6 = old_x[j:j+3]
    vmovupd ymm7, [rdx + rbp*8]   ; ymm7 = old_y[j:j+3]
    vmovupd ymm8, [rcx + rbp*8]   ; ymm8 = old_z[j:j+3]
    
    vsubpd ymm10, ymm6, ymm3      ; ymm10 = dx = old_x[j:j+3] - x_i+1
    vsubpd ymm11, ymm7, ymm4      ; ymm11 = dy = old_y[j:j+3] - y_i+1
    vsubpd ymm12, ymm8, ymm5      ; ymm12 = dz = old_z[j:j+3] - z_i+1
    
    ; Calculate distance squared for particle i+1
    vmulpd ymm13, ymm10, ymm10    ; ymm13 = dx*dx
    vfmadd231pd ymm13, ymm11, ymm11 ; ymm13 += dy*dy
    vfmadd231pd ymm13, ymm12, ymm12 ; ymm13 += dz*dz
    
    ; Calculate distance for particle i+1
    vsqrtpd ymm13, ymm13          ; ymm13 = sqrt(dx*dx + dy*dy + dz*dz)
    vmaxpd ymm13, ymm13, ymm15    ; ymm13 = max(sqrt_d2, 0.01)
    
    ; Calculate d^3 for particle i+1
    vmulpd ymm0, ymm13, ymm13     ; ymm0 = d^2
    vmulpd ymm0, ymm0, ymm13      ; ymm0 = d^3
    
    ; Calculate 1/d^3 * G for particle i+1
    vdivpd ymm0, ymm14, ymm0      ; ymm0 = G/d^3
    
    ; Calculate force factors
    vmulpd ymm0, ymm0, ymm9       ; ymm0 = mass[j:j+3] * G/d^3
    
    ; Calculate acceleration components for particle i+1
    vmulpd ymm6, ymm0, ymm10      ; ymm6 = dx * force_factor
    vmulpd ymm7, ymm0, ymm11      ; ymm7 = dy * force_factor
    vmulpd ymm8, ymm0, ymm12      ; ymm8 = dz * force_factor
    
    ; Calculate index for particle i+1
    mov r11, r14
    add r11, 1
    imul r11, 3                   ; r11 = (i+1)*3
    
    ; Horizontal sum for acceleration x
    vextractf128 xmm0, ymm6, 1    ; Extract high 128 bits
    vaddpd xmm0, xmm0, xmm6       ; Add high and low 128 bits
    vhaddpd xmm0, xmm0, xmm0      ; Horizontal add within 128 bits
    
    ; Add to acceleration array for particle i+1, x component
    vmovsd xmm1, [rbx + r11*8]    ; xmm1 = accelerations[(i+1)*3+0]
    vaddsd xmm0, xmm0, xmm1       ; xmm0 += accelerations[(i+1)*3+0]
    vmovsd [rbx + r11*8], xmm0    ; accelerations[(i+1)*3+0] = xmm0
    
    ; Horizontal sum for acceleration y
    vextractf128 xmm0, ymm7, 1    ; Extract high 128 bits
    vaddpd xmm0, xmm0, xmm7       ; Add high and low 128 bits
    vhaddpd xmm0, xmm0, xmm0      ; Horizontal add within 128 bits
    
    ; Add to acceleration array for particle i+1, y component
    vmovsd xmm1, [rbx + r11*8 + 8] ; xmm1 = accelerations[(i+1)*3+1]
    vaddsd xmm0, xmm0, xmm1       ; xmm0 += accelerations[(i+1)*3+1]
    vmovsd [rbx + r11*8 + 8], xmm0 ; accelerations[(i+1)*3+1] = xmm0
    
    ; Horizontal sum for acceleration z
    vextractf128 xmm0, ymm8, 1    ; Extract high 128 bits
    vaddpd xmm0, xmm0, xmm8       ; Add high and low 128 bits
    vhaddpd xmm0, xmm0, xmm0      ; Horizontal add within 128 bits
    
    ; Add to acceleration array for particle i+1, z component
    vmovsd xmm1, [rbx + r11*8 + 16] ; xmm1 = accelerations[(i+1)*3+2]
    vaddsd xmm0, xmm0, xmm1       ; xmm0 += accelerations[(i+1)*3+2]
    vmovsd [rbx + r11*8 + 16], xmm0 ; accelerations[(i+1)*3+2] = xmm0
    
.skip_i_plus_1_calc:
    ; Restore particle i position
    vmovsd xmm0, [r8 + r14*8]     ; xmm0 = x_i
    vbroadcastsd ymm0, xmm0       ; ymm0 = x_i (broadcast to all elements)
    
    vmovsd xmm1, [r9 + r14*8]     ; xmm1 = y_i
    vbroadcastsd ymm1, xmm1       ; ymm1 = y_i
    
    vmovsd xmm2, [r10 + r14*8]    ; xmm2 = z_i
    vbroadcastsd ymm2, xmm2       ; ymm2 = z_i
    
    ; Increment j by 4 and continue inner loop
    add rbp, 4
    jmp .inner_loop
    
.end_inner_loop:
    ; Increment i by 2 and continue outer loop
    add r14, 2
    jmp .outer_loop

.end_outer_loop:
    ; Function epilogue - restore non-volatile registers
    add rsp, 128
    pop rbx
    pop r12
    pop r13
    pop r14
    pop r15
    pop rbp
    vzeroupper                   ; Zero upper bits of YMM registers
    ret

section .data
    align 8
    min_dist:  dq 0.01   ; Minimum distance
    gravconst: dq 0.001  ; Gravitational constant
