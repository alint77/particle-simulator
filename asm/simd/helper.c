#include "particles.h"

int init(Particle* particles, int num) {
    double min_pos = -50.0, mult = +100.0, maxVel = +5.0;
    double recip = 1.0 / (double)RAND_MAX;

    for (int i = 0; i < num; i++) {
        particles[i].x = min_pos + mult * (double)rand() * recip;
        particles[i].y = min_pos + mult * (double)rand() * recip;
        particles[i].z = 0.0 + mult * (double)rand() * recip;
        particles[i].vx = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles[i].vy = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles[i].vz = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles[i].mass = 0.1 + 10.0 * (double)rand() * recip;
        
        
    }
    return 0;
}

void calc_centre_mass(double* com, Particle* particles, double totalMass, int N) {
    com[0] = 0.0;
    com[1] = 0.0;
    com[2] = 0.0;

    for (int i = 0; i < N; i++) {
        com[0] += particles[i].mass * particles[i].x;
        com[1] += particles[i].mass * particles[i].y;
        com[2] += particles[i].mass * particles[i].z;
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

    Particle* particles = (Particle*)malloc(num * sizeof(Particle) + sizeof(double) );
    if (particles == NULL) {
        printf("\n ERROR in malloc - aborting\n");
        return -99;
    }
    printf("  (malloc-ed)  ");

    if (init(particles, num) != 0) {
        printf("\n ERROR during init() - aborting\n");
        free(particles);
        return -99;
    }
    printf("  INIT COMPLETE\n");

    for (int i = 0; i < num; i++) {
        totalMass += particles[i].mass;
    }

    calc_centre_mass(com, particles, totalMass, num);
    printf("At t=0, centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);

    printf("Now to integrate for %d timesteps\n", timesteps);
    gettimeofday(&wallStart, NULL);

    for (int time = 1; time <= timesteps; time++) {
        compute_timestep(particles, num);
        calc_centre_mass(com, particles, totalMass, num);
        printf("End of timestep %d, centre of mass = (%.3f,%.3f,%.3f)\n", 
               time, com[0], com[1], com[2]);
    }

    gettimeofday(&wallEnd, NULL);
    double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);
    double WALLtimeTaken = 1.0E-06 * ((wallSecs*1000000) + 
                          (wallEnd.tv_usec - wallStart.tv_usec));

    printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", 
           num, timesteps, WALLtimeTaken);
    calc_centre_mass(com, particles, totalMass, num);
    printf("Centre of mass = (%.5f,%.5f,%.5f)\n", com[0], com[1], com[2]);

    free(particles);
    return 0;
}
