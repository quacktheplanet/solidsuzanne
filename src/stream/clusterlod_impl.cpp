// Compiles the vendored, multi-threaded copy of meshoptimizer's clusterlod reference implementation into this library.
#include "meshoptimizer.h"

#include <assert.h>

#define CLUSTERLOD_IMPLEMENTATION
#include "clusterlod_mt.h"
