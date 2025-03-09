#ifndef PARTICLES_H
#define PARTICLES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <immintrin.h>

/*
   Simulation parameters
*/
#define NUM 20000
#define TS 10
#define THREAD_COUNT 16
#define GRAVCONST 0.001

typedef struct {
    double* mass;  // Masses
    double* old_x; // Old positions
    double* old_y;
    double* old_z;
    double* x;     // Current positions
    double* y;
    double* z;
    double* vx;    // Velocities
    double* vy;
    double* vz;
    int num;       // Number of particles
} Particle;

typedef struct {
    Particle* particles;
    int start;
    int stop;
    int thread_id;
    double* accelerations;
} ThreadArgs;

// Function prototypes
int init(Particle* particles, int num);
void calc_centre_mass(double* com, Particle* particles, double totalMass, int N);

// Assembly function declaration
extern void calc_force_asm(Particle* particles, int num, int start, int stop, double* accelerations);

// Thread function prototype
void* calc_force_thread(void* args);

#endif /* PARTICLES_H */
