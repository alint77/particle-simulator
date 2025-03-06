#include "particles.h"

// Allocate memory for particle arrays
int allocate_particles(Particles* particles, int num) {
    // Set the number of particles
    particles->num = num;
    
    // Allocate memory for each array with 32-byte alignment for optimal SIMD performance
    particles->x = (double*)aligned_alloc(32, num * sizeof(double));
    particles->y = (double*)aligned_alloc(32, num * sizeof(double));
    particles->z = (double*)aligned_alloc(32, num * sizeof(double));
    particles->old_x = (double*)aligned_alloc(32, num * sizeof(double));
    particles->old_y = (double*)aligned_alloc(32, num * sizeof(double));
    particles->old_z = (double*)aligned_alloc(32, num * sizeof(double));
    particles->vx = (double*)aligned_alloc(32, num * sizeof(double));
    particles->vy = (double*)aligned_alloc(32, num * sizeof(double));
    particles->vz = (double*)aligned_alloc(32, num * sizeof(double));
    particles->mass = (double*)aligned_alloc(32, num * sizeof(double));
    
    // Check if any allocation failed
    if (!particles->x || !particles->y || !particles->z || 
        !particles->old_x || !particles->old_y || !particles->old_z || 
        !particles->vx || !particles->vy || !particles->vz || 
        !particles->mass) {
        free_particles(particles);
        return -1;
    }
    
    return 0;
}

// Free memory for particle arrays
void free_particles(Particles* particles) {
    if (particles->x) free(particles->x);
    if (particles->y) free(particles->y);
    if (particles->z) free(particles->z);
    if (particles->old_x) free(particles->old_x);
    if (particles->old_y) free(particles->old_y);
    if (particles->old_z) free(particles->old_z);
    if (particles->vx) free(particles->vx);
    if (particles->vy) free(particles->vy);
    if (particles->vz) free(particles->vz);
    if (particles->mass) free(particles->mass);
    
    // Reset pointers to NULL
    particles->x = particles->y = particles->z = NULL;
    particles->old_x = particles->old_y = particles->old_z = NULL;
    particles->vx = particles->vy = particles->vz = NULL;
    particles->mass = NULL;
}

int init(Particles* particles, int num) {
    double min_pos = -50.0, mult = +100.0, maxVel = +5.0;
    double recip = 1.0 / (double)RAND_MAX;

    for (int i = 0; i < num; i++) {
        particles->x[i] = min_pos + mult * (double)rand() * recip;
        particles->y[i] = min_pos + mult * (double)rand() * recip;
        particles->z[i] = 0.0 + mult * (double)rand() * recip;
        particles->vx[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles->vy[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles->vz[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles->mass[i] = 0.1 + 10.0 * (double)rand() * recip;
        
        // Initialize old positions to current positions
        particles->old_x[i] = particles->x[i];
        particles->old_y[i] = particles->y[i];
        particles->old_z[i] = particles->z[i];
    }
    return 0;
}

void calc_centre_mass(double* com, Particles* particles, double totalMass, int N) {
    com[0] = 0.0;
    com[1] = 0.0;
    com[2] = 0.0;

    for (int i = 0; i < N; i++) {
        com[0] += particles->mass[i] * particles->x[i];
        com[1] += particles->mass[i] * particles->y[i];
        com[2] += particles->mass[i] * particles->z[i];
    }

    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
}

int main(int argc, char* argv[]) {
    int num = NUM;
    int timesteps = TS;
    double totalMass = 0.0;
    double com[3];
    struct timeval wallStart, wallEnd;

    printf("Initializing for %d particles in x,y,z space...", num);

    // Allocate memory for particles
    Particles particles;
    if (allocate_particles(&particles, num) != 0) {
        printf("\n ERROR in memory allocation - aborting\n");
        return -99;
    }
    printf("  (malloc-ed)  ");

    // Initialize particles
    if (init(&particles, num) != 0) {
        printf("\n ERROR during init() - aborting\n");
        free_particles(&particles);
        return -99;
    }
    printf("  INIT COMPLETE\n");

    // Calculate total mass
    for (int i = 0; i < num; i++) {
        totalMass += particles.mass[i];
    }

    // Calculate initial center of mass
    calc_centre_mass(com, &particles, totalMass, num);
    printf("At t=0, centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);

    printf("Now to integrate for %d timesteps\n", timesteps);
    gettimeofday(&wallStart, NULL);

    // Main time stepping loop
    for (int time = 1; time <= timesteps; time++) {
        compute_timestep(&particles, num);
        calc_centre_mass(com, &particles, totalMass, num);
        printf("End of timestep %d, centre of mass = (%.3f,%.3f,%.3f)\n", 
               time, com[0], com[1], com[2]);
    }

    // Calculate and print timing information
    gettimeofday(&wallEnd, NULL);
    double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);
    double WALLtimeTaken = 1.0E-06 * ((wallSecs*1000000) + 
                          (wallEnd.tv_usec - wallStart.tv_usec));

    printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", 
           num, timesteps, WALLtimeTaken);
    calc_centre_mass(com, &particles, totalMass, num);
    printf("Centre of mass = (%.5f,%.5f,%.5f)\n", com[0], com[1], com[2]);

    // Free memory
    free_particles(&particles);
    return 0;
}
