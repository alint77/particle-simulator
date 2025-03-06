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

int init(double *, double *, double *, double *, double *, double *, double *, int);
void output_particles(double *, double *, double *, double *, double *, double *, double *, int);
void calc_centre_mass(double *, double *, double *, double *, double *, double, int);

int main(int argc, char *argv[])
{
  int i, j;
  int num;                                 // user defined (argv[1]) total number of gas molecules in simulation
  int time, timesteps;                     // for time stepping, including user defined (argv[2]) number of timesteps to integrate
  int rc;                                  // return code
  double *mass, *x, *y, *z, *vx, *vy, *vz; // 1D array for mass, (x,y,z) position, (vx, vy, vz) velocity
  double dx, dy, dz, d, F, GRAVCONST = 0.001;
  double ax, ay, az;
  double *old_x, *old_y, *old_z; // save previous values whilst doing global updates
  double totalMass;
  double com[3];

  /* vars for timing */
  struct timeval wallStart, wallEnd;
  gettimeofday(&wallStart, NULL); // save start time in 'wallStart'

  /* determine size of system */
  num = NUM;
  timesteps = TS;

  printf("Initializing for %d particles in x,y,z space...", num);

  /* malloc arrays and pass ref to init(). NOTE: init() uses random numbers */
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

  // should check all rc but let's just see if last malloc worked
  if (old_z == NULL)
  {
    printf("\n ERROR in malloc for (at least) old_mass - aborting\n");
    return -99;
  }
  else
  {
    printf("  (malloc-ed)  ");
  }

  // initialise
  rc = init(mass, x, y, z, vx, vy, vz, num);
  if (rc != 0)
  {
    printf("\n ERROR during init() - aborting\n");
    return -99;
  }
  else
  {
    printf("  INIT COMPLETE\n");
  }

  totalMass = 0.0;
  for (i = 0; i < num; i++)
  {
    totalMass += mass[i];
  }
  // DEBUG: output_particles(x,y,z, vx,vy,vz, mass, num);
  // output a metric (centre of mass) for checking
  calc_centre_mass(com, x, y, z, mass, totalMass, num);
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

  printf("Now to integrate for %d timesteps\n", timesteps);
  int c = 0;
  // time=0 was initial conditions
  for (time = 1; time <= timesteps; time++)
  {

    // LOOP1: take snapshot to use on RHS when looping for updates
    for (i = 0; i < num; i++)
    {
      old_x[i] = x[i];
      old_y[i] = y[i];
      old_z[i] = z[i];
    }

    double temp_d, temp_z,temp_,temp_const,temp_ai,temp_aj,axj,ayj,azj;

    // LOOP2: update position etc per particle (based on old data)
    for (i = 0; i < num; i++)
    {
      

      __m256d xi = _mm256_broadcast_sd(&x[i]);
      __m256d yi = _mm256_broadcast_sd(&y[i]);
      __m256d zi = _mm256_broadcast_sd(&z[i]);
      __m256d vxi = _mm256_broadcast_sd(&vx[i]); // also acts as accumulator for the acceleration caused by other particles in the according axis
      __m256d vyi = _mm256_broadcast_sd(&vy[i]); // also acts as accumulator for the acceleration caused by other particles in the according axis
      __m256d vzi = _mm256_broadcast_sd(&vz[i]); // also acts as accumulator for the acceleration caused by other particles in the according axis
      __m256d mi = _mm256_broadcast_sd(&mass[i]);

      
      // calc forces on body i due to particles (j != i)
      for (j = i+1; j+4 < num; j+=4)
      {
        
        __m256d old_xj = _mm256_loadu_pd(&old_x[j]);
        __m256d old_yj = _mm256_loadu_pd(&old_y[j]);
        __m256d old_zj = _mm256_loadu_pd(&old_z[j]);

        // calculate the distance between the two particles
        __m256d dx = _mm256_sub_pd(old_xj,xi);
        __m256d dy = _mm256_sub_pd(old_yj,yi);
        __m256d dz = _mm256_sub_pd(old_zj,zi);

        __m256d sqrt_d2 = _mm256_sqrt_pd(_mm256_fmadd_pd(dz, dz, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx))));

        __m256d d = _mm256_max_pd(sqrt_d2, _mm256_set1_pd(0.01));

        __m256d d3 = _mm256_mul_pd(_mm256_mul_pd(d, d), d);
        __m256d d3_inv = _mm256_div_pd(_mm256_set1_pd(GRAVCONST), d3);

        __m256d mult_j = _mm256_mul_pd(d3_inv, mi);
        __m256d mult_i = _mm256_mul_pd(d3_inv, _mm256_loadu_pd(&mass[j]));
        
        // calculate acceleration due to the force, F
        __m256d ax = _mm256_mul_pd(mult_i, dx);
        __m256d ay = _mm256_mul_pd(mult_i, dy);
        __m256d az = _mm256_mul_pd(mult_i, dz);

        vxi = _mm256_add_pd(vxi, ax);
        vyi = _mm256_add_pd(vyi, ay);
        vzi = _mm256_add_pd(vzi, az);

        __m256d axj = _mm256_mul_pd(mult_j, dx);
        __m256d ayj = _mm256_mul_pd(mult_j, dy);
        __m256d azj = _mm256_mul_pd(mult_j, dz);

        _mm256_storeu_pd(&vx[j],_mm256_fnmadd_pd(mult_j,dx,_mm256_loadu_pd(&vx[j])));
        _mm256_storeu_pd(&vy[j],_mm256_fnmadd_pd(mult_j,dy,_mm256_loadu_pd(&vy[j])));
        _mm256_storeu_pd(&vz[j],_mm256_fnmadd_pd(mult_j,dz,_mm256_loadu_pd(&vz[j])));

      }
      while (j<num && j+4>num)
        {
        // printf("i: %d, j: %d\n", i, j);
        // Handle special case
        dx = old_x[j] - x[i];
        dy = old_y[j] - y[i];
        dz = old_z[j] - z[i];

        temp_d =sqrt(dx*dx + dy*dy + dz*dz);
        d = temp_d>0.01 ? temp_d : 0.01;
        
        temp_const = GRAVCONST / (d*d*d);
        temp_ai = temp_const * mass[j];
        temp_aj = temp_const * mass[i]; ;

        // calculate acceleration due to the force, F
        ax = temp_ai * dx;
        ay = temp_ai * dy;
        az = temp_ai * dz;
        
        axj = temp_aj * dx;
        ayj = temp_aj * dy;
        azj = temp_aj * dz;
        
        // approximate velocities in "unit time"
        vxi[0] += ax;
        vyi[0] += ay;
        vzi[0] += az;

        vx[j] -= axj;
        vy[j] -= ayj;
        vz[j] -= azj;
        j++;
        }

      x[i] = old_x[i] + vxi[0]+vxi[1]+vxi[2]+vxi[3];
      y[i] = old_y[i] + vyi[0]+vyi[1]+vyi[2]+vyi[3];
      z[i] = old_z[i] + vzi[0]+vzi[1]+vzi[2]+vzi[3];
    }

    // DEBUG: output_particles(x,y,z, vx,vy,vz, mass, num);
    //  output a metric (centre of mass) for checking
    calc_centre_mass(com, x, y, z, mass, totalMass, num);
    printf("End of timestep %d, centre of mass = (%.3f,%.3f,%.3f)\n", time, com[0], com[1], com[2]);
  } // time steps

  gettimeofday(&wallEnd, NULL);                                                                    // end time
  double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);                                           // just integral number of seconds
  double WALLtimeTaken = 1.0E-06 * ((wallSecs * 1000000) + (wallEnd.tv_usec - wallStart.tv_usec)); // and now with any microseconds

  printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", num, timesteps, WALLtimeTaken);
  // output a metric (centre of mass) for checking
  calc_centre_mass(com, x, y, z, mass, totalMass, num);
  printf("Centre of mass = (%.24f,%.24f,%.24f)\n", com[0], com[1], com[2]);
  printf("c: %d\n", c);
} // main

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
