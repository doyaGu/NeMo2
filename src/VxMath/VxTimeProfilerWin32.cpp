#include "VxTimeProfiler.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

extern float g_MSecondsPerCycle;

VxTimeProfiler &VxTimeProfiler::operator=(const VxTimeProfiler &t) {
    if (&t != this)
        memcpy(Times, t.Times, sizeof(Times));
    return *this;
}

void VxTimeProfiler::Reset() {
    *(DWORD64 *) &Times[0] = __rdtsc();
}

float VxTimeProfiler::Current() {
    *(DWORD64 *) &Times[2] = __rdtsc() - *(DWORD64 *) &Times[0];
    return (float) *(DWORD64 *) &Times[2] * g_MSecondsPerCycle;
}
