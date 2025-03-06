; compute.asm - Core computation for particle simulation
; Compile with: nasm -f elf64 compute.asm -o compute.o

section .text
global compute_timestep

; typedef struct {
; double old_x;
; double old_y;
; double old_z;
; double mass;
; double vx;
; double vy;
; double vz;
; double x;
; double y;
; double z;
; } Particle;

; offsets:
; old_x = 0
; old_y = 8
; old_z = 16
; mass = 24
; vx = 32
; vy = 40
; vz = 48
; x = 56
; y = 64
; z = 72

; void compute_timestep(Particle* particles, int num)
; RDI = particles array pointer
; RSI = num particles
compute_timestep:
    push    rbp
    mov     rbp, rsp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    ; Save parameters
    mov     r12, rdi        ; r12 = particles array
    mov     r13, rsi        ; r13 = num particles

    ; LOOP1: Save current positions to old positions
    xor     rcx, rcx        ; i = 0
.loop1:
    cmp     rcx, r13        ; compare i with num
    jge     .loop1_end      ; if i >= num, end loop

    ; Calculate offset into particles array (i * sizeof(Particle))
    mov     rax, rcx
    imul    rax, 80         ; sizeof(Particle) = 80 bytes (10 doubles * 8)
    
    ; Copy current positions to old positions
    vmovsd  xmm0, [r12 + rax + 56]    ; load x
    vmovsd  [r12 + rax], xmm0         ; store to old_x
    vmovsd  xmm0, [r12 + rax + 64]    ; load y
    vmovsd  [r12 + rax + 8], xmm0     ; store to old_y
    vmovsd  xmm0, [r12 + rax + 72]    ; load z
    vmovsd  [r12 + rax + 16], xmm0    ; store to old_z

    inc     rcx
    jmp     .loop1

.loop1_end:
    ; Cache constants
    vmovsd  xmm15, [rel min_distance]  ; min_distance
    vmovsd  xmm14, [rel grav_const]    ; gravitational constant

    ; LOOP2: Calculate forces and update positions
    xor     rcx, rcx        ; i = 0
.loop2:
    cmp     rcx, r13        ; compare i with num
    jge     .loop2_end      ; if i >= num, end loop

    ; Calculate base offset for particle i
    mov     rax, rcx
    imul    rax, 80         ; sizeof(Particle) = 80 bytes

    ; Load particle i data into registers
    vmovsd  xmm13, [r12 + rax + 24]   ; mass_i
    vmovpd  ymm12, [r12 + rax + 56]   ; x_i
    vmovsd  xmm11, [r12 + rax + 64]   ; y_i
    vmovsd  xmm10, [r12 + rax + 72]   ; z_i
    vmovsd  xmm9, [r12 + rax + 32]    ; vx_i
    vmovsd  xmm8, [r12 + rax + 40]    ; vy_i
    vmovsd  xmm7, [r12 + rax + 48]    ; vz_i

    ; Inner loop j = i + 1
    mov     rdx, rcx
    inc     rdx             ; j = i + 1

    ; Align inner loop for better performance
    align 64
.inner_loop:
    cmp     rdx, r13        ; compare j with num
    jge     .inner_loop_end ; if j >= num, end inner loop

    ; Calculate offset for particle j
    mov     rbx, rdx
    imul    rbx, 80         ; sizeof(Particle) = 80 bytes

    ; Prefetch next particle data
    prefetcht0 [r12 + rbx + 80]
    
    ; Load particle j data
    vmovsd  xmm0, [r12 + rbx]         ; old_x_j
    vmovsd  xmm1, [r12 + rbx + 8]     ; old_y_j
    vmovsd  xmm2, [r12 + rbx + 16]    ; old_z_j
    vmovsd  xmm3, [r12 + rbx + 24]    ; mass_j
    
    ; Calculate dx, dy, dz
    vsubsd  xmm0, xmm0, xmm12         ; dx = old_x_j - x_i
    vsubsd  xmm1, xmm1, xmm11         ; dy = old_y_j - y_i
    vsubsd  xmm2, xmm2, xmm10         ; dz = old_z_j - z_i
    
    ; Calculate d^2 = dx*dx + dy*dy + dz*dz using AVX2 FMA
    vmulsd  xmm4, xmm0, xmm0          ; dx*dx
    vfmadd231sd xmm4, xmm1, xmm1      ; += dy*dy
    vfmadd231sd xmm4, xmm2, xmm2      ; += dz*dz
    
    ; Calculate d = sqrt(d^2) and max with min_distance
    vsqrtsd xmm4, xmm4, xmm4          ; d = sqrt(d^2)
    vmaxsd  xmm4, xmm4, xmm15         ; d = max(d, 0.01)
    
    ; Calculate 1/d^3
    vmulsd  xmm5, xmm4, xmm4          ; d^2
    vmulsd  xmm5, xmm5, xmm4          ; d^3
    vdivsd  xmm5, xmm14, xmm5          ; G/d^3
    
    ; Calculate forces
    vmulsd  xmm4, xmm5, xmm3          ; G*m_j/d^3 for particle i
    
    vmulsd  xmm5, xmm5, xmm13         ; G*m_i/d^3 for particle j
    ; Update velocities for particle i using AVX2 FMA
    vfmadd231sd xmm9, xmm0, xmm4      ; vx_i += dx * (G*m_j/d^3)
    vfmadd231sd xmm8, xmm1, xmm4      ; vy_i += dy * (G*m_j/d^3)
    vfmadd231sd xmm7, xmm2, xmm4      ; vz_i += dz * (G*m_j/d^3)
    
    
    ; Update velocities for particle j using AVX2 FMA
    vfnmadd123sd xmm0, xmm5,[r12 + rbx + 32]     ; vx_j -= dx * (G*m_i/d^3)
    vfnmadd123sd xmm1, xmm5,[r12 + rbx + 40]     ; vy_j -= dy * (G*m_i/d^3)
    vfnmadd123sd xmm2, xmm5,[r12 + rbx + 48]     ; vz_j -= dz * (G*m_i/d^3)
    
    ; Store updated velocities for particle j
    vmovsd  [r12 + rbx + 32], xmm0asm/compute.asm    ; store vx_j
    vmovsd  [r12 + rbx + 40], xmm1asm/compute.asm    ; store vy_j
    vmovsd  [r12 + rbx + 48], xmm2asm/compute.asm    ; store vz_j

    inc     rdx
    jmp     .inner_loop


.inner_loop_end:
    ; Update positions for particle i using old positions
    movsd  xmm0, [r12 + rax]         ; old_x
    addsd  xmm0, xmm9                ; old_x + vx_i
    movsd  [r12 + rax + 56], xmm0    ; store new x
    
    movsd  xmm0, [r12 + rax + 8]     ; old_y
    addsd  xmm0, xmm8                ; old_y + vy_i
    movsd  [r12 + rax + 64], xmm0    ; store new y
    
    movsd  xmm0, [r12 + rax + 16]    ; old_z
    addsd  xmm0, xmm7                ; old_z + vz_i
    movsd  [r12 + rax + 72], xmm0    ; store new z
    
    ; Store updated velocities for particle i
    movsd  [r12 + rax + 32], xmm9    ; store vx_i
    movsd  [r12 + rax + 40], xmm8    ; store vy_i
    movsd  [r12 + rax + 48], xmm7    ; store vz_i

    inc     rcx
    jmp     .loop2

.loop2_end:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

section .data
    align 16
    grav_const:      dq 0.001    ; GRAVCONST
    min_distance:    dq 0.01     ; Minimum distance threshold
