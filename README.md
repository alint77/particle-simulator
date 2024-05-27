The aim of this project is to optimise the simple N-body simulation code provided to run as fast as possible.\
\
the hardware the code is benchmarked on is: \
\
Intel Core i5-8365U, 16GB 2133 Dual-Channel. Running Ubuntu 20.04.\
\
Since this is a laptop, and for accurate benchmarking, predictable and repeatable CPU clocks are crucial, the CPU's TDP for PL1 and PL2 were equally set to 25W, to eliminate performance degredation over time. Also the CPU and Cache were undervolted by 102mV to achieve ~300Mhz higher effective core clock.\
\
first a baseline performance needs to be established.\
\
for compiler optimisations, only -O0 and -Ofast will be tested. GCC was the compiler of choice. Also every test result provided represents the best of 3 runs.\
\
we'll ignore the MPI implementation for now.\
\
compile command:\
gcc 'filename.c' -O{0,fast} -lm ({-fopenmp} for OpenMP) ({-mavx -mfma} for SIMD)\

BASELINE PERFORMANCE FOR 20k PARTICLES: \

md-serial-org:\
-O0:    102.1s\
-Ofast: 44.5s\
\
md-OpenMP-org:\
-O0:    22.93s\
-Ofast: 8.01s \

REFERENCE CENTRE OF MASS:\
(-0.09509,-0.16562,49.64602)\
(This result is treated as the ground truth, thus optimised runs must always result in these values)\
\
Optimisation process for the serial implementation was started by removing extra/irrelevent variables:\
    old_mass is completely obsolete and was removed.\
    totalMass now gets calculated inside the init function.\
    the calc_centre_mass function ran twice for no reason so the outer loop was removed.\
    the nuumber of particles are set at compile time so there's no reason to allocate heap memory. every malloc was removed.\
\
These are the results after these minor adjustments:\
-O0:    81.28s (+25.6%)\
-Ofast: 23.35s (+90.5%)\
\
By diving deeper into the main logic of the algorithm, some optimisations can be done to reduce the number of arithmatic calculations on each iteration:\
    F,ax,ay,az were removed since they contained many duplicate calculations, instead mult_i was introduced. This reduces the total number of divide operations (which are the most expensive) from 7 to only 1.\
\
The results after:\
-O0:    53.46s (+52.0%) (+90.9%)\
-Ofast: 19.12s (+22.1%) (+132.7%)\
\
The original nested loop iterates over each particle twice, which is not ideal. This can be fixed by setting j=i+1 and updating velocity values of j in addition to i during each iteration (should be noted that the force to particle j is applied in the opposite direction). This halves the total number of iterations in the main loop.\
\
Results:\
-O0:    36.12s (+48.0%) (+182.6%)\
-Ofast: 15.32s (+24.8%) (+190.4%)\





