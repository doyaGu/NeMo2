#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "VxMathDefines.h"
#include "XString.h"
#include "VxWindowFunctions.h"

INSTANCE_HANDLE g_CKModule;

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID lpReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            OutputDebugString(TEXT("Loading CK2.dll"));
            g_CKModule = hModule;
            break;
        case DLL_PROCESS_DETACH:
            OutputDebugString(TEXT("Unloading CK2.dll"));
            break;
    }
    return TRUE;
}
