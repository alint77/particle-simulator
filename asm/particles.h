#ifndef PARTICLES_H
#define PARTICLES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NUM 20000
#define TS 10
#define GRAVCONST 0.001

typedef struct {
    double old_x;
    double old_y;
    double old_z;
    double mass;
    double vx;
    double vy;
    double vz;
    double x;
    double y;
    double z;
} Particle;

// Function declarations
int init(Particle* particles, int num);
void calc_centre_mass(double* com, Particle* particles, double totalMass, int N);
void compute_timestep(Particle* particles, int num);

#endif // PARTICLES_H
