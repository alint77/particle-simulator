#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h> // for wallclock timing functions
#include <pthread.h>
#include <immintrin.h>

/*
   hard-wire simulation parameters
*/
#define NUM 20000
#define TS 10
#define THREAD_COUNT 8
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

void* calc_force(void* args);
int init(Particle* particles, int num);
void calc_centre_mass(double* com, Particle* particles, double totalMass, int N);
static inline double hsums(__m256d a);

int main(int argc, char *argv[]) {
    int num;             // user defined (argv[1]) total number of gas molecules in simulation
    int time, timesteps; // for time stepping, including user defined (argv[2]) number of timesteps to integrate
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
    for (int i = 0; i < num; i++) {
        totalMass += particles.mass[i];
    }

    // output a metric (centre of mass) for checking
    calc_centre_mass(com, &particles, totalMass, num);
    printf("At t=0, centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);

    printf("Now to integrate for %d timesteps\n", timesteps);
    double* accelerations = (double*)calloc(num * 3, sizeof(double));
    
    for (time = 1; time <= timesteps; time++) {
        // LOOP1: take snapshot to use on RHS when looping for updates
        for (int i = 0; i < num; i++) {
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

void* calc_force(void* args) {
    ThreadArgs* targs = (ThreadArgs*)args;
    Particle* particles = targs->particles;
    int num = particles->num;
    int start = targs->start;
    int stop = targs->stop;
    double* accelerations = targs->accelerations;

    for (int i = start; i < stop; i+=2) {

        __m256d x_i = _mm256_broadcast_sd(&particles->x[i]);
        __m256d y_i = _mm256_broadcast_sd(&particles->y[i]);
        __m256d z_i = _mm256_broadcast_sd(&particles->z[i]);

        __m256d x_i2 = _mm256_broadcast_sd(&particles->x[i+1]);
        __m256d y_i2 = _mm256_broadcast_sd(&particles->y[i+1]);
        __m256d z_i2 = _mm256_broadcast_sd(&particles->z[i+1]);

        for (int j = 0; j < num; j += 4) {

            __m256d old_x = _mm256_loadu_pd(&particles->old_x[j]);
            __m256d old_y = _mm256_loadu_pd(&particles->old_y[j]);
            __m256d old_z = _mm256_loadu_pd(&particles->old_z[j]);
            __m256d mass = _mm256_loadu_pd(&particles->mass[j]);

            __m256d dx = _mm256_sub_pd(old_x, x_i);
            __m256d dy = _mm256_sub_pd(old_y, y_i);
            __m256d dz = _mm256_sub_pd(old_z, z_i);

            __m256d dx2 = _mm256_sub_pd(old_x, x_i2);
            __m256d dy2 = _mm256_sub_pd(old_y, y_i2);
            __m256d dz2 = _mm256_sub_pd(old_z, z_i2);
            
            __m256d sqrt_d2 = _mm256_sqrt_pd(_mm256_fmadd_pd(dz, dz, 
                                            _mm256_fmadd_pd(dy, dy, 
                                            _mm256_mul_pd(dx, dx))));
            __m256d sqrt_d2_2 = _mm256_sqrt_pd(_mm256_fmadd_pd(dz2, dz2,
                                              _mm256_fmadd_pd(dy2, dy2,
                                              _mm256_mul_pd(dx2, dx2))));

            __m256d d = _mm256_max_pd(sqrt_d2, _mm256_set1_pd(0.01));
            __m256d d_2 = _mm256_max_pd(sqrt_d2_2, _mm256_set1_pd(0.01));

            __m256d d2 = _mm256_mul_pd(d, d);
            __m256d d2_2 = _mm256_mul_pd(d_2, d_2);
            __m256d d3 = _mm256_mul_pd(d2, d);
            __m256d d3_2 = _mm256_mul_pd(d2_2, d_2);
            __m256d d3_inv = _mm256_div_pd(_mm256_set1_pd(GRAVCONST), d3);
            __m256d d3_inv_2 = _mm256_div_pd(_mm256_set1_pd(GRAVCONST), d3_2);
            
            __m256d temp_ai = _mm256_mul_pd(d3_inv, mass);
            __m256d temp_ai_2 = _mm256_mul_pd(d3_inv_2, mass);

            // // Skip self-interactions
            // __m256d mask = _mm256_castsi256_pd(_mm256_setr_epi64x(
            //     (j+0 != i) ? -1LL : 0LL,
            //     (j+1 != i) ? -1LL : 0LL,
            //     (j+2 != i) ? -1LL : 0LL,
            //     (j+3 != i) ? -1LL : 0LL
            // ));
            
            accelerations[i * 3 + 0] += hsums(_mm256_mul_pd(temp_ai, dx));
            accelerations[i * 3 + 1] += hsums(_mm256_mul_pd(temp_ai, dy));
            accelerations[i * 3 + 2] += hsums(_mm256_mul_pd(temp_ai, dz));

            accelerations[(i+1) * 3 + 0] += hsums(_mm256_mul_pd(temp_ai_2, dx2));
            accelerations[(i+1) * 3 + 1] += hsums(_mm256_mul_pd(temp_ai_2, dy2));
            accelerations[(i+1) * 3 + 2] += hsums(_mm256_mul_pd(temp_ai_2, dz2));
        }
    }
    return NULL;
}

static inline double hsums(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); 
    vlow  = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}