#ifndef PARTICLES_H
#define PARTICLES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NUM 20000
#define TS 10
#define GRAVCONST 0.001

// Structure of Arrays (SoA) layout for better SIMD processing
typedef struct {
    // Current positions
    double* x;
    double* y;
    double* z;
    
    // Old positions (from previous timestep)
    double* old_x;
    double* old_y;
    double* old_z;
    
    // Velocities
    double* vx;
    double* vy;
    double* vz;
    
    // Mass
    double* mass;
    
    // Number of particles
    int num;
} Particles;

// Function declarations
int init(Particles* particles, int num);
void calc_centre_mass(double* com, Particles* particles, double totalMass, int N);
void compute_timestep(Particles* particles, int num);
void free_particles(Particles* particles);

#endif // PARTICLES_H
