#include "VxTimeProfiler.h"

#include <time.h>
#include <string.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

VxTimeProfiler &VxTimeProfiler::operator=(const VxTimeProfiler &t) {
    if (&t != this)
        memcpy(Times, t.Times, sizeof(Times));
    return *this;
}

void VxTimeProfiler::Reset() {
#if defined(__APPLE__)
    *(uint64_t *) &Times[0] = mach_absolute_time();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *(uint64_t *) &Times[0] = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

float VxTimeProfiler::Current() {
#if defined(__APPLE__)
    uint64_t elapsed = mach_absolute_time() - *(uint64_t *) &Times[0];
    static mach_timebase_info_data_t timebase = {0, 0};
    if (timebase.denom == 0)
        mach_timebase_info(&timebase);
    // Convert to milliseconds
    *(uint64_t *) &Times[2] = elapsed;
    return (float)(elapsed * timebase.numer / timebase.denom) / 1000000.0f;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    uint64_t elapsed = now - *(uint64_t *) &Times[0];
    *(uint64_t *) &Times[2] = elapsed;
    // Convert nanoseconds to milliseconds
    return (float)elapsed / 1000000.0f;
#endif
}
