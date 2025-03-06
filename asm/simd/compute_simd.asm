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
    vmovupd  ymm12, [r12 + rax + 56]   ; x_i, y_i, z_i, old_x[i+1] (garbage/padding)
    vmovupd  ymm11, [r12 + rax + 32]    ; vx_i, vy_i, vz_i, x_i (garbage/padding) accumulator
    
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
    vmovupd  ymm0, [r12 + rbx]         ; old_x_j, old_y_j, old_z_j, mass_j
    vmovsd  xmm1, [r12 + rbx + 24]    ; mass_j
    
    ; Calculate dx, dy, dz
    vsubpd  ymm0, ymm0, ymm12         ; dx = old_x_j - x_i , dy = old_y_j - y_i, dz = old_z_j - z_i
    vmovupd ymm3, ymm0                ; dx, dy, dz, mass_j
    
    ; Calculate d^2 = dx*dx + dy*dy + dz*dz using AVX2 FMA
    vmulpd  ymm0, ymm0, ymm0          ; dx^2, dy^2, dz^2

    ; Sum dx^2, dy^2, dz^2
    vextractf128 xmm2, ymm0, 1
    vaddpd  xmm0, xmm0, xmm2          ; dx^2 + dy^2, dz^2, 0, 0
    vhaddpd xmm0, xmm0, xmm0          ; dx^2 + dy^2 + dz^2, 0, 0, 0

    
    ; Calculate d = sqrt(d^2) and max with min_distance
    vsqrtsd xmm0, xmm0, xmm0          ; d = sqrt(d^2)
    vmaxsd  xmm0, xmm0, xmm15         ; d = max(d, 0.01)
    
    ; Calculate 1/d^3
    vmulsd  xmm5, xmm0, xmm0          ; d^2
    vmulsd  xmm5, xmm5, xmm0          ; d^3
    vdivsd  xmm5, xmm14, xmm5          ; G/d^3
    
    ; Calculate forces
    vmulsd  xmm4, xmm5, xmm3          ; G*m_j/d^3 for particle i
    vmulsd  xmm5, xmm5, xmm13         ; G*m_i/d^3 for particle j

    ; Update velocities for particle i using AVX2 FMA

    vfmadd231pd ymm11, ymm3, ymm4    ; vx_i += dx * (G*m_j/d^3), vy_i += dy * (G*m_j/d^3), vz_i += dz * (G*m_j/d^3)
    
    ; Update velocities for particle j using AVX2 FMA

    vfnmadd123pd ymm5, ymm3, [r12 + rax + 32]    ; vx_j -= dx * (G*m_i/d^3), vy_j -= dy * (G*m_i/d^3), vz_j -= dz * (G*m_i/d^3)

    ; Store updated velocities for particle j
    vmovupd  [r12 + rbx + 32], xmm5    ; store vx_j
    ; store the third element of ymm5 to [r12 + rbx + 48] (vz_j)
    vextractf128 xmm6, ymm5, 1      ; Extract high 128 bits (elements 2,3) to xmm6
    vmovsd [r12 + rbx + 48], xmm6   ; Store the lowest double from xmm6 (element 2 from ymm5)

    inc     rdx
    jmp     .inner_loop


.inner_loop_end:
    ; Update positions for particle i using old positions

    ; Load updated velocities for particle i
    vmovupd  ymm9, [r12 + rax ]      ; old_x, old_y, old_z, mass_i
    vaddpd  ymm9, ymm9, ymm11        ; old_x + vx_i, old_y + vy_i, old_z + vz_i
    vmovupd  [r12 + rax + 56], xmm9  ; store new x, y, z
    vextractf128 xmm6, ymm9, 1        ;  
    vmovsd [r12 + rbx + 48], xmm6   ; 

    
    ; Store updated velocities for particle i
    vmovupd  [r12 + rax + 32], xmm11    ; store vx_i, vy_i, vz_i, x_i (garbage/padding)
    ; store the third element of ymm11 to [r12 + rax + 48]
    vextractf128 xmm7, ymm11, 1        ; Extract vz_i (third element)
    vmovsd [r12 + rax + 48], xmm7   ; Store vz_i


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
