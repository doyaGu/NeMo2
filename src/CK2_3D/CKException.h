#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

// Installs the CK2_3D unhandled exception filter and prepares symbol handling.
void CKInstallExceptionHandler();

// Removes the previously installed exception filter and releases symbol resources.
void CKRemoveExceptionHandler();

// Unhandled exception callback; exposed for testing or explicit invocation.
LONG WINAPI CKExceptionHandler(EXCEPTION_POINTERS *pExceptionInfo);
