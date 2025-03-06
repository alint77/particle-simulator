#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>  // for wallclock timing functions                                                                               
/* 
   hard-wire simulation parameters
*/
#define NUM 20000
#define TS  10

typedef struct {
  double mass;
  double old_x;
  double old_y;
  double old_z;
  double x;
  double y;
  double z;
  double vx;
  double vy;
  double vz;
} Particle;

int init(Particle*, int);
void calc_centre_mass(double*,Particle*, double, int);
int main(int argc, char* argv[]) {
  int i, j;
  int num;     // user defined (argv[1]) total number of gas molecules in simulation
  int time, timesteps; // for time stepping, including user defined (argv[2]) number of timesteps to integrate
  int rc;      // return code
  double dx, dy, dz, d, F, GRAVCONST=0.001;
  double totalMass;  
  double com[3];

  /* vars for timing */
  struct timeval wallStart, wallEnd;
  gettimeofday(&wallStart, NULL);    // save start time in 'wallStart' 
  
  /* determine size of system */
  num = NUM;
  timesteps = TS;

  printf("Initializing for %d particles in x,y,z space...", num);

  /* malloc arrays and pass ref to init(). NOTE: init() uses random numbers */
  Particle* particles = (Particle *) malloc(num * sizeof(Particle));


  // should check all rc but let's just see if last malloc worked
  if (particles == NULL) {
    printf("\n ERROR in malloc for (at least) old_mass - aborting\n");
    return -99;
  } 
  else {
    printf("  (malloc-ed)  ");
  }

  // initialise
  rc = init(particles, num);
  if (rc != 0) {
    printf("\n ERROR during init() - aborting\n");
    return -99;
  }
  else {
    printf("  INIT COMPLETE\n");
  }

  totalMass = 0.0;
  for (i=0; i<num; i++) {
    totalMass += particles[i].mass;
  }
  // DEBUG: output_particles(x,y,z, vx,vy,vz, mass, num);
  // output a metric (centre of mass) for checking
  calc_centre_mass(com,particles,totalMass,num);
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

  // time=0 was initial conditions
  for (time=1; time<=timesteps; time++) {

    // LOOP1: take snapshot to use on RHS when looping for updates
    for (i=0; i<num; i++) {
      particles[i].old_x = particles[i].x;
      particles[i].old_y = particles[i].y;
      particles[i].old_z = particles[i].z;
    }

    double temp_const, temp_aj, temp_ai, temp_d,axj,ayj,azj,ax,ay,az;

    // LOOP2: update position etc per particle (based on old data)
    for(i=0; i<num; i++) {
      // calc forces on body i due to particles (j != i)
      for (j=i+1; j<num; j++) {
        
          dx = particles[j].old_x - particles[i].x;
          dy = particles[j].old_y - particles[i].y;
	        dz = particles[j].old_z - particles[i].z;

          temp_d =sqrt(dx*dx + dy*dy + dz*dz);
          d = temp_d>0.01 ? temp_d : 0.01;
          
          temp_const = GRAVCONST / (d*d*d);
          temp_ai = temp_const * particles[j].mass;
          temp_aj = temp_const * particles[i].mass;


	        // calculate acceleration due to the force, F
          ax = temp_ai * dx;
          ay = temp_ai * dy;
	        az = temp_ai * dz;
	        
          axj = temp_aj * dx;
          ayj = temp_aj * dy;
          azj = temp_aj * dz;
          
          // approximate velocities in "unit time"
          particles[i].vx += ax;
          particles[i].vy += ay;
	        particles[i].vz += az;

          particles[j].vx -= axj;
          particles[j].vy -= ayj;
          particles[j].vz -= azj;
      } 

      // calc new position 
      particles[i].x = particles[i].old_x + particles[i].vx;
      particles[i].y = particles[i].old_y + particles[i].vy;
      particles[i].z = particles[i].old_z + particles[i].vz;

    } // end of LOOP 2
    //DEBUG: output_particles(x,y,z, vx,vy,vz, mass, num);    
    // output a metric (centre of mass) for checking
    calc_centre_mass(com, particles,totalMass,num);
    printf("End of timestep %d, centre of mass = (%.3f,%.3f,%.3f)\n", time, com[0], com[1], com[2]);
  } // time steps

  gettimeofday(&wallEnd, NULL); // end time                                                                                 
  double wallSecs = (wallEnd.tv_sec - wallStart.tv_sec);           // just integral number of seconds
  double WALLtimeTaken = 1.0E-06 * ((wallSecs*1000000) + (wallEnd.tv_usec - wallStart.tv_usec)); // and now with any microseconds

  printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", num, timesteps, WALLtimeTaken);
  // output a metric (centre of mass) for checking
  calc_centre_mass(com, particles,totalMass,num);
  printf("Centre of mass = (%.5f,%.5f,%.5f)\n", com[0], com[1], com[2]);
} // main


int init(Particle* particles, int num) {
  // random numbers to set initial conditions - do not parallelise or amend order of random number usage
  int i;
  double comp;
  double min_pos = -50.0, mult = +100.0, maxVel = +5.0;
  double recip = 1.0 / (double)RAND_MAX;

  for (i=0; i<num; i++) {
    particles[i].x = min_pos + mult*(double)rand() * recip;  
    particles[i].y = min_pos + mult*(double)rand() * recip;  
    particles[i].z = 0.0 + mult*(double)rand() * recip;   
    particles[i].vx = -maxVel + 2.0*maxVel*(double)rand() * recip;   
    particles[i].vy = -maxVel + 2.0*maxVel*(double)rand() * recip;   
    particles[i].vz = -maxVel + 2.0*maxVel*(double)rand() * recip;
    particles[i].mass = 0.1 + 10*(double)rand() * recip;  // mass is 0.1 up to 10.1
  }
  return 0;
} // init


void calc_centre_mass(double *com,Particle* particles, double totalMass, int N) {
  int i;
   // calculate the centre of mass, com(x,y,z)
    com[0] = 0.0;     com[1] = 0.0;     com[2] = 0.0; 
    for (i=0; i<N; i++) {
      com[0] += particles[i].mass*particles[i].x;
      com[1] += particles[i].mass*particles[i].y;
      com[2] += particles[i].mass*particles[i].z;
    }
    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
  return;
}
