#include "VxSharedLibrary.h"

#include <dlfcn.h>

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
    m_LibraryHandle = dlopen(LibraryName, RTLD_NOW | RTLD_LOCAL);
    return m_LibraryHandle;
}

// Unloads the shared Library
void VxSharedLibrary::ReleaseLibrary() {
    if (m_LibraryHandle)
        dlclose(m_LibraryHandle);
}

// Retrieves a function pointer from the library
void *VxSharedLibrary::GetFunctionPtr(char *FunctionName) {
    if (m_LibraryHandle && FunctionName)
        return dlsym(m_LibraryHandle, FunctionName);
    else
        return NULL;
}
