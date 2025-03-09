#include "particles.h"

// Thread function that calls the assembly implementation
void* calc_force_thread(void* args) {
    ThreadArgs* targs = (ThreadArgs*)args;
    Particle* particles = targs->particles;
    int num = particles->num;
    int start = targs->start;
    int stop = targs->stop;
    double* accelerations = targs->accelerations;

    // Call the assembly implementation
    calc_force_asm(particles, num, start, stop, accelerations);
    
    return NULL;
}

// Initialize particles with random positions, velocities, and masses
int init(Particle* particles, int num) {
    int i;
    double min_pos = -50.0, mult = +100.0, maxVel = +5.0;
    double recip = 1.0 / (double)RAND_MAX;

    for (i = 0; i < num; i++) {
        particles->x[i] = min_pos + mult * (double)rand() * recip;
        particles->y[i] = min_pos + mult * (double)rand() * recip;
        particles->z[i] = 0.0 + mult * (double)rand() * recip;
        particles->vx[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles->vy[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles->vz[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
        particles->mass[i] = 0.1 + 10 * (double)rand() * recip; // mass is 0.1 up to 10.1
    }
    return 0;
}

// Calculate the center of mass of the system
void calc_centre_mass(double* com, Particle* particles, double totalMass, int N) {
    int i;
    com[0] = 0.0;
    com[1] = 0.0;
    com[2] = 0.0;
    for (i = 0; i < N; i++) {
        com[0] += particles->mass[i] * particles->x[i];
        com[1] += particles->mass[i] * particles->y[i];
        com[2] += particles->mass[i] * particles->z[i];
    }
    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
}

// Main function
int main(int argc, char *argv[]) {
    int i;
    int num;             // total number of particles in simulation
    int time, timesteps; // for time stepping
    int rc;              // return code
    double totalMass;
    double com[3];

    /* vars for timing */
    struct timeval wallStart, wallEnd;
    gettimeofday(&wallStart, NULL); // save start time in 'wallStart'

    /* determine size of particles */
    num = NUM;
    timesteps = TS;

    printf("Initializing for %d particles in x,y,z space...", num);

    /* malloc arrays for particle particles */
    Particle particles;
    particles.num = num;
    particles.x = (double*)malloc(num * sizeof(double));
    particles.y = (double*)malloc(num * sizeof(double));
    particles.z = (double*)malloc(num * sizeof(double));
    particles.old_x = (double*)malloc(num * sizeof(double));
    particles.old_y = (double*)malloc(num * sizeof(double));
    particles.old_z = (double*)malloc(num * sizeof(double));
    particles.vx = (double*)malloc(num * sizeof(double));
    particles.vy = (double*)malloc(num * sizeof(double));
    particles.vz = (double*)malloc(num * sizeof(double));
    particles.mass = (double*)malloc(num * sizeof(double));

    // Check if any malloc failed
    if (!particles.x || !particles.y || !particles.z || !particles.old_x || !particles.old_y || 
        !particles.old_z || !particles.vx || !particles.vy || !particles.vz || !particles.mass) {
        printf("\n ERROR in malloc - aborting\n");
        return -99;
    } else {
        printf("  (malloc-ed)  ");
    }

    // initialise
    rc = init(&particles, num);
    if (rc != 0) {
        printf("\n ERROR during init() - aborting\n");
        return -99;
    } else {
        printf("  INIT COMPLETE\n");
    }

    totalMass = 0.0;
    for (i = 0; i < num; i++) {
        totalMass += particles.mass[i];
    }

    // output a metric (centre of mass) for checking
    calc_centre_mass(com, &particles, totalMass, num);
    printf("At t=0, centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);

    printf("Now to integrate for %d timesteps\n", timesteps);
    double* accelerations = (double*)calloc(num * 3, sizeof(double));
    
    for (time = 1; time <= timesteps; time++) {
        // LOOP1: take snapshot to use on RHS when looping for updates
        for (i = 0; i < num; i++) {
            particles.old_x[i] = particles.x[i];
            particles.old_y[i] = particles.y[i];
            particles.old_z[i] = particles.z[i];
            accelerations[i * 3 + 0] = 0.0;
            accelerations[i * 3 + 1] = 0.0;
            accelerations[i * 3 + 2] = 0.0;
        }

        pthread_t threads[THREAD_COUNT];
        ThreadArgs thread_args[THREAD_COUNT];

        for (int t = 0; t < THREAD_COUNT; t++) {
            int start = (num/THREAD_COUNT)*t;
            int stop = (num/THREAD_COUNT)*(t+1);
            thread_args[t] = (ThreadArgs){&particles, start, stop, t, accelerations};
            pthread_create(&threads[t], NULL, calc_force_thread, (void*)&thread_args[t]);
        }

        for (int t = 0; t < THREAD_COUNT; t++) {
            pthread_join(threads[t], NULL);
        }

        //LOOP3: update position etc per particle
        for (int i = 0; i < num; i++) {
            particles.vx[i] += accelerations[i * 3 + 0];
            particles.vy[i] += accelerations[i * 3 + 1];
            particles.vz[i] += accelerations[i * 3 + 2];
            // calc new position
            particles.x[i] = particles.old_x[i] + particles.vx[i];
            particles.y[i] = particles.old_y[i] + particles.vy[i];
            particles.z[i] = particles.old_z[i] + particles.vz[i];
        }

        calc_centre_mass(com, &particles, totalMass, num);
        printf("End of timestep %d, centre of mass = (%.3f,%.3f,%.3f)\n", time, com[0], com[1], com[2]);
    }
    free(accelerations);

    gettimeofday(&wallEnd, NULL);
    double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);
    double WALLtimeTaken = 1.0E-06 * ((wallSecs * 1000000) + (wallEnd.tv_usec - wallStart.tv_usec));

    printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", num, timesteps, WALLtimeTaken);
    calc_centre_mass(com, &particles, totalMass, num);
    printf("Centre of mass = (%.5f,%.5f,%.5f)\n", com[0], com[1], com[2]);

    // Free memory
    free(particles.x);
    free(particles.y);
    free(particles.z);
    free(particles.old_x);
    free(particles.old_y);
    free(particles.old_z);
    free(particles.vx);
    free(particles.vy);
    free(particles.vz);
    free(particles.mass);

    return 0;
}
