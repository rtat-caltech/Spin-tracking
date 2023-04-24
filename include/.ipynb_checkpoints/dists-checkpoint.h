#include <math.h>
#include <cmath>

#if __HIPCC__
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#elif __NVCC__

#else
#include <random>
#endif

#if __HIPCC__
double normal01(hiprandState* t);
double maxboltz(hiprandState * t, const double sqrtkT_m);
double unif02pi(hiprandState* t, const double u);
double exponential(hiprandState* t, const double tc);
#elif __NVCC__

#else
double normal01(std::normal_distribution<double> dist, std::mt19937_64 gen);
double maxboltz(std::normal_distribution<double> dist, std::mt19937_64 gen, const double sqrtkT_m);
double unif02pi(std::uniform_distribution<double> dist, std::mt19937_64 gen);
double exponential(std::uniform_distribution<double> dist, std::mt19937_64 gen, const double tc);

#endif