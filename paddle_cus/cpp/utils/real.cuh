#pragma once

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float - real;
    const real EPSILON = 1.0e-6f;
#endif