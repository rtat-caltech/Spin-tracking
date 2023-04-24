#pragma once

#include "../include/dists.h"

#if __HIPCC__
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#elif __NVCC__

#else
#include <random>
#endif

#if defined(__HIPCC__)
double normal01(hiprandState* t){
	return hiprand_normal_double(t);
}

double maxboltz(hiprandState * t, const double sqrtkT_m){
    return normal01(t) * sqrtkT_m;
}

double unif02pi(hiprandState* t, const double u){
    return hiprand_uniform_double(t) * 2.0 * M_PI;
}

double exponential(hiprandState* t, const double tc){
    return - tc * log(1.0 - hiprand_uniform_double(t));
}

#elif defined(__NVCC__)

#else
double normal01(std::normal_distribution<double> dist, std::mt19937_64 gen){
	return dist(gen);
}

double maxboltz(std::normal_distribution<double> dist, std::mt19937_64 gen, const double sqrtkT_m){
    return normal01(dist, gen) * sqrtkT_m;
}

double unif02pi(std::uniform_distribution<double> dist, std::mt19937_64 gen){
    return dist(gen) * 2.0 * M_PI;
}

double exponential(std::uniform_distribution<double> dist, std::mt19937_64 gen, const double tc){
    return - tc * log(1.0 - dist(gen));
}

#endif



