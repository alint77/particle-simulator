#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h> // for wallclock timing functions
#include <immintrin.h>
/*
   hard-wire simulation parameters
*/
#define NUM 20000
#define TS 10

int init(double *, double *, double *, double *, double *, double *, double *, double *);
void output_particles(double *, double *, double *, double *, double *, double *, double *, int);
void calc_centre_mass(double *, double *, double *, double *, double *, double, int);

int main(int argc, char *argv[])
{
  int i, j;
  int time;                                                            // for time stepping, including user defined (argv[2]) NUMber of timesteps to integrate
  int rc;                                                              // return code  
  double GRAVCONST = 0.001;
  double totalMass;
  double mult_i, mult_j;
  double com[3];

  double* x = (double*)malloc(NUM * sizeof(double));
  double* y = (double*)malloc(NUM * sizeof(double));
  double* z = (double*)malloc(NUM * sizeof(double));
  double* vx = (double*)malloc(NUM * sizeof(double));
  double* vy = (double*)malloc(NUM * sizeof(double));
  double* vz = (double*)malloc(NUM * sizeof(double));
  double* old_x = (double*)malloc(NUM * sizeof(double));
  double* old_y = (double*)malloc(NUM * sizeof(double));
  double* old_z = (double*)malloc(NUM * sizeof(double));
  double* mass = (double*)malloc(NUM * sizeof(double));

  int n = sizeof(__m256d)/sizeof(double);

  /* vars for timing */
  struct timeval wallStart, wallEnd;
  gettimeofday(&wallStart, NULL); // save start time in 'wallStart'

  // initialise
  rc = init(mass, x, y, z, vx, vy, vz, &totalMass);
  if (rc != 0)
  {
    printf("\n ERROR during init() - aborting\n");
    return -99;
  }
  else
  {
    printf("  INIT COMPLETE\n");
  }

  // DEBUG: output_particles(x,y,z, vx,vy,vz, mass, NUM);
  // output a metric (centre of mass) for checking
  calc_centre_mass(com, x, y, z, mass, totalMass, NUM);
  printf("At t=0, centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);

  /*
     MAIN TIME STEPPING LOOP

     We 'save' old (entry to timestep loop) values to use on RHS of:

     For each molecule we will: calc forces due to other
     particles & update change in velocity and thus position.

     Having looped over the particles (using 'old' values on the right
     hand side as a crude approximation to real life) we then have
     updated all particles independently.

     After completing all time-steps, we output the the time taken and
     the centre of mass of the final system configuration.

  */

  printf("Now to integrate for %d timesteps\n", TS);



  __m256d* xi, *yi, *zi, *vxi, *vyi, *vzi, *old_xi, *old_yi, *old_zi, *mi;

  // time=0 was initial conditions
  for (time = 1; time <= TS; time++)
  {
    // LOOP1: save old values
    for (i = 0; i < NUM; i+=n)
    {
      _mm256_storeu_si256((__m256i*)&old_x[i], _mm256_loadu_si256((__m256i*)&x[i]));
      _mm256_storeu_si256((__m256i*)&old_y[i], _mm256_loadu_si256((__m256i*)&y[i]));
      _mm256_storeu_si256((__m256i*)&old_z[i], _mm256_loadu_si256((__m256i*)&z[i]));

      // old_x[i] = x[i];
      // old_y[i] = y[i];
      // old_z[i] = z[i];
    }

    double temp_d, temp_z;

    // LOOP2: update position etc per particle (based on old data)
    for (i = 0; i < NUM; i+=n)
    {
      // calc forces on body i due to particles (j != i)
      for (j = i+1; j < NUM; j+=n)
      {
        

        __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(&old_x[i]),_mm256_loadu_pd(&x[i]));
        __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(&old_y[i]),_mm256_loadu_pd(&y[i]));
        __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(&old_z[i]),_mm256_loadu_pd(&z[i]));

        __m256d temp_d = _mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx,dx),_mm256_mul_pd(dy,dy)),_mm256_mul_pd(dz,dz)));
        __m256d d = _mm256_max_pd(temp_d,_mm256_set1_pd(0.01));

        // F = GRAVCONST * mass[i] * mass[j] / (d * d);
        // temp_z = GRAVCONST / (d * d * d);
        __m256d mult_i = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(GRAVCONST),_mm256_loadu_pd(&mass[j])),_mm256_mul_pd(d,d));
        __m256d mult_j = _mm256_mul_pd(_mm256_set1_pd(-GRAVCONST),_mm256_div_pd(_mm256_loadu_pd(&mass[i]),_mm256_mul_pd(d,d)));

        // calculate acceleration due to the force, F

        // ax = (F / mass[i]) * dx / d;
        // ay = (F / mass[i]) * dy / d;
        // az = (F / mass[i]) * dz / d;

        // approximate velocities in "unit time"
        _mm256_storeu_pd(&vx[i],_mm256_add_pd(_mm256_loadu_pd(&vx[i]),_mm256_mul_pd(mult_i,dx)));
        _mm256_storeu_pd(&vy[i],_mm256_add_pd(_mm256_loadu_pd(&vy[i]),_mm256_mul_pd(mult_i,dy)));
        _mm256_storeu_pd(&vz[i],_mm256_add_pd(_mm256_loadu_pd(&vz[i]),_mm256_mul_pd(mult_i,dz)));
        // vz[i] += mult_i * dz;
        _mm256_storeu_pd(&vx[j],_mm256_add_pd(_mm256_loadu_pd(&vx[j]),_mm256_mul_pd(mult_j,dx)));
        _mm256_storeu_pd(&vy[j],_mm256_add_pd(_mm256_loadu_pd(&vy[j]),_mm256_mul_pd(mult_j,dy)));
        _mm256_storeu_pd(&vz[j],_mm256_add_pd(_mm256_loadu_pd(&vz[j]),_mm256_mul_pd(mult_j,dz)));
        // vx[j] += mult_j * dx;
        // vy[j] += mult_j * dy;
        // vz[j] += mult_j * dz;
      }

      // calc new position
      _mm256_storeu_pd(&x[i],_mm256_add_pd(_mm256_loadu_pd(&old_x[i]),_mm256_loadu_pd(&vx[i])));
      _mm256_storeu_pd(&y[i],_mm256_add_pd(_mm256_loadu_pd(&old_y[i]),_mm256_loadu_pd(&vy[i])));
      _mm256_storeu_pd(&z[i],_mm256_add_pd(_mm256_loadu_pd(&old_z[i]),_mm256_loadu_pd(&vz[i])));
      
      // x[i] = old_x[i] + vx[i];
      // y[i] = old_y[i] + vy[i];
      // z[i] = old_z[i] + vz[i];

    } // end of LOOP 2
    // DEBUG: output_particles(x,y,z, vx,vy,vz, mass, NUM);
    //  output a metric (centre of mass) for checking
    calc_centre_mass(com, x, y, z, mass, totalMass, NUM);
    printf("End of timestep %d, centre of mass = (%.3f,%.3f,%.3f)\n", time, com[0], com[1], com[2]);
  } // time steps

  gettimeofday(&wallEnd, NULL);                                                                    // end time
  double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);                                           // just integral NUMber of seconds
  double WALLtimeTaken = 1.0E-06 * ((wallSecs * 1000000) + (wallEnd.tv_usec - wallStart.tv_usec)); // and now with any microseconds

  printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", NUM, TS, WALLtimeTaken);
  // output a metric (centre of mass) for checking
  calc_centre_mass(com, x, y, z, mass, totalMass, NUM);
  printf("Centre of mass = (%.5f,%.5f,%.5f)\n", com[0], com[1], com[2]);
  free(mass);
  free(x);
  free(y);
  free(z);
  free(vx);
  free(vy);
  free(vz);
  free(old_x);
  free(old_y);
  free(old_z);

} // main

int init(double *mass, double *x, double *y, double *z, double *vx, double *vy, double *vz, double *totalMass)
{
  // random NUMbers to set initial conditions - do not parallelise or amend order of random NUMber usage
  int i;
  double comp;
  double min_pos = -50.0, mult = +100.0, maxVel = +5.0;
  double recip = 1.0 / (double)RAND_MAX;

  for (i = 0; i < NUM; i++)
  {
    x[i] = min_pos + mult * (double)rand() * recip;
    y[i] = min_pos + mult * (double)rand() * recip;
    z[i] = 0.0 + mult * (double)rand() * recip;
    vx[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
    vy[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
    vz[i] = -maxVel + 2.0 * maxVel * (double)rand() * recip;
    mass[i] = 0.1 + 10 * (double)rand() * recip; // mass is 0.1 up to 10.1
    *totalMass += mass[i];
  }
  return 0;
} // init

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
