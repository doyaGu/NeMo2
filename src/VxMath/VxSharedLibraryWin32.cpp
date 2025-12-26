#include "VxSharedLibrary.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "VxWindowFunctions.h"

// Creates an unattached VxLibrary
VxSharedLibrary::VxSharedLibrary() {
    m_LibraryHandle = NULL;
}

// Attaches an existing Library to a VxLibrary
void VxSharedLibrary::Attach(INSTANCE_HANDLE LibraryHandle) {
    m_LibraryHandle = LibraryHandle;
}

// Loads the shared Library from disk
INSTANCE_HANDLE VxSharedLibrary::Load(char *LibraryName) {
    if (m_LibraryHandle)
        ReleaseLibrary();
    XWORD fpuCtrlWord = VxGetFPUControlWord();
    m_LibraryHandle = (INSTANCE_HANDLE) LoadLibraryA(LibraryName);
    VxSetFPUControlWord(fpuCtrlWord);
    return m_LibraryHandle;
}

// Unloads the shared Library
void VxSharedLibrary::ReleaseLibrary() {
    if (m_LibraryHandle)
        FreeLibrary((HMODULE) m_LibraryHandle);
}

// Retrieves a function pointer from the library
void *VxSharedLibrary::GetFunctionPtr(char *FunctionName) {
    if (m_LibraryHandle && FunctionName)
        return (void *) (GetProcAddress((HMODULE) m_LibraryHandle, FunctionName));
    else
        return NULL;
}
