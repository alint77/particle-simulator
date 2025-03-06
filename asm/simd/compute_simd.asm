; compute_simd.asm - SIMD-optimized core computation for particle simulation
; Compile with: nasm -f elf64 compute_simd.asm -o compute.o

section .text
global compute_timestep

; void compute_timestep(Particles* particles, int num)
; RDI = particles struct pointer
; RSI = num particles (not used, as it's stored in the struct)
compute_timestep:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 64          ; Allocate stack space for temporary variables

    ; Extract pointers from Particles struct
    mov     r8, [rdi]        ; r8 = particles->x
    mov     r9, [rdi+8]      ; r9 = particles->y
    mov     r10, [rdi+16]    ; r10 = particles->z
    mov     r11, [rdi+24]    ; r11 = particles->old_x
    mov     r12, [rdi+32]    ; r12 = particles->old_y
    mov     r13, [rdi+40]    ; r13 = particles->old_z
    mov     r14, [rdi+48]    ; r14 = particles->vx
    mov     r15, [rdi+56]    ; r15 = particles->vy
    mov     rbx, [rdi+64]    ; rbx = particles->vz
    mov     rax, [rdi+72]    ; rax = particles->mass
    mov     rsi, [rdi+80]    ; rsi = particles->num (overwrite the second parameter)

    ; Save pointers to stack for later use
    mov     [rsp], r8        ; Save x pointer
    mov     [rsp+8], r9      ; Save y pointer
    mov     [rsp+16], r10    ; Save z pointer
    mov     [rsp+24], r11    ; Save old_x pointer
    mov     [rsp+32], r12    ; Save old_y pointer
    mov     [rsp+40], r13    ; Save old_z pointer
    mov     [rsp+48], r14    ; Save vx pointer
    mov     [rsp+56], r15    ; Save vy pointer
    mov     [rsp+64], rbx    ; Save vz pointer
    mov     [rsp+72], rax    ; Save mass pointer

    ; LOOP1: Save current positions to old positions
    ; This is now done in the init function, so we can skip it here

    ; LOOP2: Calculate forces and update positions
    xor     rcx, rcx        ; i = 0
.loop2:
    cmp     rcx, rsi        ; compare i with num
    jge     .loop2_end      ; if i >= num, end loop

    ; Load pointers from stack
    mov     r8, [rsp]        ; r8 = x pointer
    mov     r9, [rsp+8]      ; r9 = y pointer
    mov     r10, [rsp+16]    ; r10 = z pointer
    mov     r11, [rsp+24]    ; r11 = old_x pointer
    mov     r12, [rsp+32]    ; r12 = old_y pointer
    mov     r13, [rsp+40]    ; r13 = old_z pointer
    mov     r14, [rsp+48]    ; r14 = vx pointer
    mov     r15, [rsp+56]    ; r15 = vy pointer
    mov     rbx, [rsp+64]    ; rbx = vz pointer
    mov     rax, [rsp+72]    ; rax = mass pointer

    ; Broadcast particle i data to YMM registers
    vbroadcastsd ymm0, [r11 + rcx*8]   ; old_x[i]
    vbroadcastsd ymm1, [r12 + rcx*8]   ; old_y[i]
    vbroadcastsd ymm2, [r13 + rcx*8]   ; old_z[i]
    vbroadcastsd ymm3, [rax + rcx*8]   ; mass[i]
    
    ; Initialize acceleration accumulators to zero
    vxorpd  ymm4, ymm4, ymm4    ; axi = 0
    vxorpd  ymm5, ymm5, ymm5    ; ayi = 0
    vxorpd  ymm6, ymm6, ymm6    ; azi = 0

    ; Inner loop j = i + 1
    lea     rdx, [rcx + 1]      ; j = i + 1

.inner_loop:
    ; Check if we have at least 4 more particles to process
    lea     rdi, [rdx + 4]
    cmp     rdi, rsi
    jg      .remainder_loop     ; Less than 4 particles left, handle separately

    ; Load data for 4 particles at once (j, j+1, j+2, j+3)
    ; With SoA layout, we can load 4 consecutive elements from each array
    vmovupd ymm7, [r11 + rdx*8]       ; old_x for particles j, j+1, j+2, j+3
    vmovupd ymm8, [r12 + rdx*8]       ; old_y for particles j, j+1, j+2, j+3
    vmovupd ymm9, [r13 + rdx*8]       ; old_z for particles j, j+1, j+2, j+3
    vmovupd ymm10, [rax + rdx*8]      ; mass for particles j, j+1, j+2, j+3

    ; Calculate dx, dy, dz
    vsubpd  ymm7, ymm7, ymm0    ; dx = old_x[j] - old_x[i]
    vsubpd  ymm8, ymm8, ymm1    ; dy = old_y[j] - old_y[i]
    vsubpd  ymm9, ymm9, ymm2    ; dz = old_z[j] - old_z[i]

    ; Calculate d^2 = dx*dx + dy*dy + dz*dz
    vmulpd  ymm11, ymm7, ymm7   ; dx*dx
    vfmadd231pd ymm11, ymm8, ymm8   ; dx*dx + dy*dy
    vfmadd231pd ymm11, ymm9, ymm9   ; dx*dx + dy*dy + dz*dz

    ; Calculate d = sqrt(d^2)
    vsqrtpd ymm11, ymm11        ; sqrt(dx*dx + dy*dy + dz*dz)

    ; Max with 0.01 to avoid division by zero
    vbroadcastsd ymm12, [rel min_distance]
    vmaxpd  ymm11, ymm11, ymm12 ; d = max(d, 0.01)

    ; Calculate d^3
    vmulpd  ymm12, ymm11, ymm11 ; d^2
    vmulpd  ymm12, ymm12, ymm11 ; d^3

    ; Calculate GRAVCONST / d^3
    vbroadcastsd ymm13, [rel grav_const]
    vdivpd  ymm12, ymm13, ymm12 ; GRAVCONST / d^3

    ; Calculate force multipliers
    vmulpd  ymm13, ymm12, ymm10 ; (GRAVCONST / d^3) * mass[j]
    vmulpd  ymm14, ymm12, ymm3  ; (GRAVCONST / d^3) * mass[i]

    ; Update acceleration for particle i
    vfmadd231pd ymm4, ymm13, ymm7   ; axi += dx * (GRAVCONST / d^3) * mass[j]
    vfmadd231pd ymm5, ymm13, ymm8   ; ayi += dy * (GRAVCONST / d^3) * mass[j]
    vfmadd231pd ymm6, ymm13, ymm9   ; azi += dz * (GRAVCONST / d^3) * mass[j]

    ; Load velocities for particles j, j+1, j+2, j+3
    vmovupd ymm10, [r14 + rdx*8]     ; vx for particles j, j+1, j+2, j+3
    vmovupd ymm11, [r15 + rdx*8]     ; vy for particles j, j+1, j+2, j+3
    vmovupd ymm12, [rbx + rdx*8]     ; vz for particles j, j+1, j+2, j+3

    ; Update velocities for particles j, j+1, j+2, j+3
    vfnmadd231pd ymm10, ymm14, ymm7   ; vx[j] -= dx * (GRAVCONST / d^3) * mass[i]
    vfnmadd231pd ymm11, ymm14, ymm8   ; vy[j] -= dy * (GRAVCONST / d^3) * mass[i]
    vfnmadd231pd ymm12, ymm14, ymm9   ; vz[j] -= dz * (GRAVCONST / d^3) * mass[i]

    ; Store updated velocities
    vmovupd [r14 + rdx*8], ymm10    ; store vx
    vmovupd [r15 + rdx*8], ymm11    ; store vy
    vmovupd [rbx + rdx*8], ymm12    ; store vz

    ; Move to next 4 particles
    add     rdx, 4
    jmp     .inner_loop

.remainder_loop:
    ; Handle remaining particles one by one
    cmp     rdx, rsi
    jge     .inner_loop_end

    ; Load data for particle j
    vmovsd  xmm7, [r11 + rdx*8]      ; old_x[j]
    vmovsd  xmm8, [r12 + rdx*8]      ; old_y[j]
    vmovsd  xmm9, [r13 + rdx*8]      ; old_z[j]
    vmovsd  xmm10, [rax + rdx*8]     ; mass[j]

    ; Calculate dx, dy, dz
    vsubsd  xmm7, xmm7, xmm0    ; dx = old_x[j] - old_x[i]
    vsubsd  xmm8, xmm8, xmm1    ; dy = old_y[j] - old_y[i]
    vsubsd  xmm9, xmm9, xmm2    ; dz = old_z[j] - old_z[i]

    ; Calculate d^2 = dx*dx + dy*dy + dz*dz
    vmulsd  xmm11, xmm7, xmm7   ; dx*dx
    vfmadd231sd xmm11, xmm8, xmm8   ; dx*dx + dy*dy
    vfmadd231sd xmm11, xmm9, xmm9   ; dx*dx + dy*dy + dz*dz

    ; Calculate d = sqrt(d^2)
    vsqrtsd xmm11, xmm11, xmm11        ; sqrt(dx*dx + dy*dy + dz*dz)

    ; Max with 0.01 to avoid division by zero
    vmovsd  xmm12, [rel min_distance]
    vmaxsd  xmm11, xmm11, xmm12 ; d = max(d, 0.01)

    ; Calculate d^3
    vmulsd  xmm12, xmm11, xmm11 ; d^2
    vmulsd  xmm12, xmm12, xmm11 ; d^3

    ; Calculate GRAVCONST / d^3
    vmovsd  xmm13, [rel grav_const]
    vdivsd  xmm12, xmm13, xmm12 ; GRAVCONST / d^3

    ; Calculate force multipliers
    vmulsd  xmm13, xmm12, xmm10 ; (GRAVCONST / d^3) * mass[j]
    vmulsd  xmm14, xmm12, xmm3  ; (GRAVCONST / d^3) * mass[i]

    ; Update acceleration for particle i (accumulate in the first element of ymm4,5,6)
    vfmadd231sd xmm4, xmm13, xmm7   ; axi += dx * (GRAVCONST / d^3) * mass[j]
    vfmadd231sd xmm5, xmm13, xmm8   ; ayi += dy * (GRAVCONST / d^3) * mass[j]
    vfmadd231sd xmm6, xmm13, xmm9   ; azi += dz * (GRAVCONST / d^3) * mass[j]

    ; Load velocities for particle j
