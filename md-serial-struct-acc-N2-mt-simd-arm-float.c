#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h> // for wallclock timing functions
#include <pthread.h>
#include <arm_neon.h>  // ARM NEON intrinsics

/*
   hard-wire simulation parameters
*/
#define NUM 20000
#define TS 10
#define THREAD_COUNT 8
#define GRAVCONST 0.001f

typedef struct {
    float* mass;   // Masses
    float* old_x;  // Old positions
    float* old_y;
    float* old_z;
    float* x;      // Current positions
    float* y;
    float* z;
    float* vx;     // Velocities
    float* vy;
    float* vz;
    int num;       // Number of particles
} Particle;

typedef struct {
    Particle* particles;
    int start;
    int stop;
    int thread_id;
    float* accelerations;
} ThreadArgs;

void* calc_force(void* args);
int init(Particle* particles, int num);
void calc_centre_mass(float* com, Particle* particles, float totalMass, int N);
static inline float vaddvq_f32_sum(float32x4_t v);

int main(int argc, char *argv[]) {
    int i, j;
    int num;             // user defined (argv[1]) total number of gas molecules in simulation
    int time, timesteps; // for time stepping, including user defined (argv[2]) number of timesteps to integrate
    int rc;              // return code
    float dx, dy, dz, d, F;
    float totalMass;
    float com[3];

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
    particles.x = (float*)aligned_alloc(32,num * sizeof(float));
    particles.y = (float*)aligned_alloc(32,num * sizeof(float));
    particles.z = (float*)aligned_alloc(32,num * sizeof(float));
    particles.old_x = (float*)aligned_alloc(32,num * sizeof(float));
    particles.old_y = (float*)aligned_alloc(32,num * sizeof(float));
    particles.old_z = (float*)aligned_alloc(32,num * sizeof(float));
    particles.vx = (float*)aligned_alloc(32,num * sizeof(float));
    particles.vy = (float*)aligned_alloc(32,num * sizeof(float));
    particles.vz = (float*)aligned_alloc(32,num * sizeof(float));
    particles.mass = (float*)aligned_alloc(32,num * sizeof(float));

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

    totalMass = 0.0f;
    for (i = 0; i < num; i++) {
        totalMass += particles.mass[i];
    }

    // output a metric (centre of mass) for checking
    calc_centre_mass(com, &particles, totalMass, num);
    printf("At t=0, centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);

    printf("Now to integrate for %d timesteps\n", timesteps);
    float* accelerations = (float*)calloc(num * 3, sizeof(float));
    
    for (time = 1; time <= timesteps; time++) {
        // LOOP1: take snapshot to use on RHS when looping for updates
        for (i = 0; i < num; i++) {
            particles.old_x[i] = particles.x[i];
            particles.old_y[i] = particles.y[i];
            particles.old_z[i] = particles.z[i];
            accelerations[i * 3 + 0] = 0.0f;
            accelerations[i * 3 + 1] = 0.0f;
            accelerations[i * 3 + 2] = 0.0f;
        }

        pthread_t threads[THREAD_COUNT];
        ThreadArgs thread_args[THREAD_COUNT];

        for (int t = 0; t < THREAD_COUNT; t++) {
            int start = (num/THREAD_COUNT)*t;
            int stop = (num/THREAD_COUNT)*(t+1);
            thread_args[t] = (ThreadArgs){&particles, start, stop, t, accelerations};
            pthread_create(&threads[t], NULL, calc_force, (void*)&thread_args[t]);
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

int init(Particle* particles, int num) {
    int i;
    float min_pos = -50.0f, mult = +100.0f, maxVel = +5.0f;
    float recip = 1.0f / (float)RAND_MAX;

    for (i = 0; i < num; i++) {
        particles->x[i] = min_pos + mult * (float)rand() * recip;
        particles->y[i] = min_pos + mult * (float)rand() * recip;
        particles->z[i] = 0.0f + mult * (float)rand() * recip;
        particles->vx[i] = -maxVel + 2.0f * maxVel * (float)rand() * recip;
        particles->vy[i] = -maxVel + 2.0f * maxVel * (float)rand() * recip;
        particles->vz[i] = -maxVel + 2.0f * maxVel * (float)rand() * recip;
        particles->mass[i] = 0.1f + 10.0f * (float)rand() * recip; // mass is 0.1 up to 10.1
    }
    return 0;
}

void calc_centre_mass(float* com, Particle* particles, float totalMass, int N) {
    int i;
    com[0] = 0.0f;
    com[1] = 0.0f;
    com[2] = 0.0f;
    for (i = 0; i < N; i++) {
        com[0] += particles->mass[i] * particles->x[i];
        com[1] += particles->mass[i] * particles->y[i];
        com[2] += particles->mass[i] * particles->z[i];
    }
    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
}

void* calc_force(void* args) {
    ThreadArgs* targs = (ThreadArgs*)args;
    Particle* particles = targs->particles;
    int num = particles->num;
    int start = targs->start;
    int stop = targs->stop;
    float* accelerations = targs->accelerations;

    // ARM NEON implementation with float32
    for (int i = start; i < stop; i+=2) {
        // Create vectors with repeated values for particle i
        float32x4_t x_i = vdupq_n_f32(particles->x[i]);
        float32x4_t y_i = vdupq_n_f32(particles->y[i]);
        float32x4_t z_i = vdupq_n_f32(particles->z[i]);

        // Create vectors with repeated values for particle i+1
        float32x4_t x_i2 = vdupq_n_f32(particles->x[i+1]);
        float32x4_t y_i2 = vdupq_n_f32(particles->y[i+1]);
        float32x4_t z_i2 = vdupq_n_f32(particles->z[i+1]);

        // Process particles in groups of 4 (NEON processes 4 floats at a time)
        for (int j = 0; j < num; j += 4) {

            float32x4_t old_x = vld1q_f32(&particles->old_x[j]);
            float32x4_t old_y = vld1q_f32(&particles->old_y[j]);
            float32x4_t old_z = vld1q_f32(&particles->old_z[j]);
            float32x4_t mass = vld1q_f32(&particles->mass[j]);

            // Calculate distance vectors for particle i
            float32x4_t dx = vsubq_f32(old_x, x_i);
            float32x4_t dy = vsubq_f32(old_y, y_i);
            float32x4_t dz = vsubq_f32(old_z, z_i);

            // Calculate distance vectors for particle i+1
            float32x4_t dx2 = vsubq_f32(old_x, x_i2);
            float32x4_t dy2 = vsubq_f32(old_y, y_i2);
            float32x4_t dz2 = vsubq_f32(old_z, z_i2);
            
            // Sum of squares for distance calculation
            float32x4_t d_sq = vmlaq_f32(vmlaq_f32(vmulq_f32(dx, dx), dy, dy), dz, dz);          
            float32x4_t d2_sq = vmlaq_f32(vmlaq_f32(vmulq_f32(dx2, dx2), dy2, dy2), dz2, dz2); 

            // Calculate distance (sqrt of sum of squares)
            float32x4_t sqrt_d = vsqrtq_f32(d_sq);
            float32x4_t sqrt_d2 = vsqrtq_f32(d2_sq);
            
            // Apply minimum distance threshold (0.01)
            float32x4_t min_dist = vdupq_n_f32(0.01f);
            float32x4_t d = vmaxq_f32(sqrt_d, min_dist);
            float32x4_t d_2 = vmaxq_f32(sqrt_d2, min_dist);
            
            // Calculate d^2 and d^3
            float32x4_t d2 = vmulq_f32(d, d);
            float32x4_t d2_2 = vmulq_f32(d_2, d_2);
            float32x4_t d3 = vmulq_f32(d2, d);
            float32x4_t d3_2 = vmulq_f32(d2_2, d_2);
            
            // Calculate GRAVCONST/d^3
            float32x4_t grav_const = vdupq_n_f32(GRAVCONST);
            float32x4_t d3_inv = vdivq_f32(grav_const, d3);
            float32x4_t d3_inv_2 = vdivq_f32(grav_const, d3_2);
            
            // Calculate acceleration factors
            float32x4_t temp_ai = vmulq_f32(d3_inv, mass);
            float32x4_t temp_ai_2 = vmulq_f32(d3_inv_2, mass);
            
            // Calculate acceleration components and accumulate
            float32x4_t ax = vmulq_f32(temp_ai, dx);
            float32x4_t ay = vmulq_f32(temp_ai, dy);
            float32x4_t az = vmulq_f32(temp_ai, dz);
            
            float32x4_t ax2 = vmulq_f32(temp_ai_2, dx2);
            float32x4_t ay2 = vmulq_f32(temp_ai_2, dy2);
            float32x4_t az2 = vmulq_f32(temp_ai_2, dz2);
            
            // Sum the vector components and add to acceleration arrays
            accelerations[i * 3 + 0] += vaddvq_f32(ax);
            accelerations[i * 3 + 1] += vaddvq_f32(ay);
            accelerations[i * 3 + 2] += vaddvq_f32(az);
            
            accelerations[(i+1) * 3 + 0] += vaddvq_f32(ax2);
            accelerations[(i+1) * 3 + 1] += vaddvq_f32(ay2);
            accelerations[(i+1) * 3 + 2] += vaddvq_f32(az2);
        }
    }
    return NULL;
}

// Helper function to sum elements in a NEON vector
static inline float vaddvq_f32_sum(float32x4_t v) {
    float result[4];
    vst1q_f32(result, v);
    return result[0] + result[1] + result[2] + result[3];
}
