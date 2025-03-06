#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <immintrin.h>
#include <omp.h>

#define NUM 20000
#define TS 10
#define GRAVCONST 0.001

int init(double *, double *, double *, double *, double *, double *, double *, int);
void calc_centre_mass(double *com, double *x, double *y, double *z, double *mass, double totalMass, int N);

int main(int argc, char *argv[]) {
    int i, j, num = NUM, timesteps = TS, rc;
    double *mass, *x, *y, *z, *vx, *vy, *vz, *old_x, *old_y, *old_z;
    double totalMass, com[3];
    struct timeval wallStart, wallEnd;

    // Allocate memory
    mass = (double *)malloc(num * sizeof(double));
    x = (double *)malloc(num * sizeof(double));
    y = (double *)malloc(num * sizeof(double));
    z = (double *)malloc(num * sizeof(double));
    vx = (double *)malloc(num * sizeof(double));
    vy = (double *)malloc(num * sizeof(double));
    vz = (double *)malloc(num * sizeof(double));
    old_x = (double *)malloc(num * sizeof(double));
    old_y = (double *)malloc(num * sizeof(double));
    old_z = (double *)malloc(num * sizeof(double));

    // Initialization
    rc = init(mass, x, y, z, vx, vy, vz, num);
    if (rc != 0) return -99;

    totalMass = 0.0;
    for (i = 0; i < num; i++) totalMass += mass[i];
    calc_centre_mass(com, x, y, z, mass, totalMass, num);
    printf("Initial center of mass: (%g, %g, %g)\n", com[0], com[1], com[2]);

    gettimeofday(&wallStart, NULL);

    // Main simulation loop
    for (int time = 1; time <= timesteps; time++) {
        // Save old positions
        #pragma omp parallel for
        for (i = 0; i < num; i++) {
            old_x[i] = x[i];
            old_y[i] = y[i];
            old_z[i] = z[i];
        }

        // Process particles with combined SIMD+OpenMP
        #pragma omp parallel for schedule(dynamic)
        for (i = 0; i < num; i++) {
            __m256d axi = _mm256_setzero_pd();
            __m256d ayi = _mm256_setzero_pd();
            __m256d azi = _mm256_setzero_pd();
            
            __m256d xi = _mm256_broadcast_sd(x+i);
            __m256d yi = _mm256_broadcast_sd(y+i);
            __m256d zi = _mm256_broadcast_sd(z+i);
            __m256d mi = _mm256_broadcast_sd(mass+i);

            // Process 4 particles at a time with 2 threads
            #pragma omp parallel for num_threads(8)
            for (j = i+1; j < num; j += 4) {
                if (j + 4 <= num) {
                    __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(old_x+j), xi);
                    __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(old_y+j), yi);
                    __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(old_z+j), zi);
                    
                    __m256d d2 = _mm256_fmadd_pd(dz, dz, 
                        _mm256_fmadd_pd(dy, dy, 
                        _mm256_mul_pd(dx, dx)));
                    __m256d d = _mm256_max_pd(_mm256_sqrt_pd(d2), _mm256_set1_pd(0.01));
                    
                    __m256d inv_d3 = _mm256_div_pd(_mm256_set1_pd(GRAVCONST), 
                        _mm256_mul_pd(d, _mm256_mul_pd(d, d)));
                    
                    __m256d mj = _mm256_loadu_pd(mass+j);
                    __m256d acc_i = _mm256_mul_pd(inv_d3, mj);
                    __m256d acc_j = _mm256_mul_pd(inv_d3, mi);

                    axi = _mm256_fmadd_pd(acc_i, dx, axi);
                    ayi = _mm256_fmadd_pd(acc_i, dy, ayi);
                    azi = _mm256_fmadd_pd(acc_i, dz, azi);

                    _mm256_storeu_pd(vx+j, _mm256_fnmadd_pd(acc_j, dx, _mm256_loadu_pd(vx+j)));
                    _mm256_storeu_pd(vy+j, _mm256_fnmadd_pd(acc_j, dy, _mm256_loadu_pd(vy+j)));
                    _mm256_storeu_pd(vz+j, _mm256_fnmadd_pd(acc_j, dz, _mm256_loadu_pd(vz+j)));
                }
            }

            // Horizontal sum of accelerations
            double ax_sum[4], ay_sum[4], az_sum[4];
            _mm256_storeu_pd(ax_sum, axi);
            _mm256_storeu_pd(ay_sum, ayi);
            _mm256_storeu_pd(az_sum, azi);
            
            double ax_total = ax_sum[0] + ax_sum[1] + ax_sum[2] + ax_sum[3];
            double ay_total = ay_sum[0] + ay_sum[1] + ay_sum[2] + ay_sum[3];
            double az_total = az_sum[0] + az_sum[1] + az_sum[2] + az_sum[3];

            // Update positions and velocities
            vx[i] += ax_total;
            vy[i] += ay_total;
            vz[i] += az_total;
            x[i] = old_x[i] + vx[i];
            y[i] = old_y[i] + vy[i];
            z[i] = old_z[i] + vz[i];
        }

        calc_centre_mass(com, x, y, z, mass, totalMass, num);
        printf("Timestep %d center: (%.3f, %.3f, %.3f)\n", time, com[0], com[1], com[2]);
    }

    gettimeofday(&wallEnd, NULL);                                                                    // end time
  double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);                                           // just integral number of seconds
  double WALLtimeTaken = 1.0E-06 * ((wallSecs * 1000000) + (wallEnd.tv_usec - wallStart.tv_usec)); // and now with any microseconds

  printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", num, timesteps, WALLtimeTaken);
  // output a metric (centre of mass) for checking
  calc_centre_mass(com, x, y, z, mass, totalMass, num);
  printf("Centre of mass = (%.24f,%.24f,%.24f)\n", com[0], com[1], com[2]);
}
int init(double *mass, double *x, double *y, double *z, double *vx, double *vy, double *vz, int num)
{
  // random numbers to set initial conditions - do not parallelise or amend order of random number usage
  int i;
  double comp;
  double min_pos = -50.0, mult = +100.0, maxVel = +5.0;
  double recip = 1.0 / (double)RAND_MAX;

  for (i = 0; i < num; i++)
  {
    x[i] = min_pos + mult * (double)rand() * recip;
    y[i] = min_pos + mult * (double)rand() * recip;
    z[i] = 0.0 + mult * (double)rand() * recip;
    vx[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
    vy[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
    vz[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;

    mass[i] = 0.1 + 10 * (double)rand() * recip; // mass is 0.1 up to 10.1
  }
  return 0;
} // init

void output_particles(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *mass, int n)
{
  int i;
  printf("num \t position (x,y,z) \t velocity (vx, vy, vz)\t mass \n");
  for (i = 0; i < n; i++)
  {
    printf("%d \t %f %f %f \t %f %f %f \t %f \n", i, x[i], y[i], z[i], vx[i], vy[i], vz[i], mass[i]);
  }
}

void calc_centre_mass(double *com, double *x, double *y, double *z, double *mass, double totalMass, int N)
{
  int i;
  // calculate the centre of mass, com(x,y,z)
  
    com[0] = 0.0;
    com[1] = 0.0;
    com[2] = 0.0;
    for (i = 0; i < N; i++)
    {
      com[0] += mass[i] * x[i];
      com[1] += mass[i] * y[i];
      com[2] += mass[i] * z[i];
    }
    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
  
}

// Keep original init and calc_centre_mass functions from md-serial-org-simd-exp.c