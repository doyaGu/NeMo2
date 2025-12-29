#include "VxMathDefines.h"
#include "XString.h"
#include "VxWindowFunctions.h"

#include <dlfcn.h>
#include <stdio.h>

INSTANCE_HANDLE g_CKModule;

// POSIX shared library constructor and destructor
__attribute__((constructor))
static void CK2Init() {
    // Get handle to this shared library
    Dl_info info;
    if (dladdr((void*)CK2Init, &info)) {
        g_CKModule = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    }
}

__attribute__((destructor))
static void CK2Cleanup() {
    if (g_CKModule) {
        dlclose(g_CKModule);
        g_CKModule = NULL;
    }
}
