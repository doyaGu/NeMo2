#ifndef VXSHAREDLIBRARY_H
#define VXSHAREDLIBRARY_H

#include "VxMathDefines.h"

/**
 * @brief Utility class for loading and interacting with shared libraries (DLLs).
 *
 * @example
 * @code
 * // To load a DLL, retrieve a specific function, and call it:
 * VxSharedLibrary shl;
 * INSTANCE_HANDLE dllInst = shl.Load("MyLibrary.dll"); // INSTANCE_HANDLE is HMODULE on Win32
 *
 * if (dllInst) {
 *     // Define the function pointer type
 *     typedef void (*MyFuncType)(int);
 *
 *     MyFuncType MyFunc = (MyFuncType)shl.GetFunctionPtr("MyFunctionInTheDll");
 *     if (MyFunc) {
 *         MyFunc(123);
 *     }
 *
 *     shl.ReleaseLibrary();
 * }
 * @endcode
 */
class VxSharedLibrary {
public:
    /// @brief Creates an unattached VxSharedLibrary object.
    VX_EXPORT VxSharedLibrary();

    /**
     * @brief Attaches an existing library handle to this object.
     * @param LibraryHandle A handle to an already loaded library.
     */
    VX_EXPORT void Attach(INSTANCE_HANDLE LibraryHandle);

    /**
     * @brief Loads a shared library from disk.
     * @param LibraryName The name or path of the library to load.
     * @return A handle to the loaded library, or NULL on failure.
     */
    VX_EXPORT INSTANCE_HANDLE Load(char *LibraryName);

    /// @brief Unloads the shared library if it was loaded by this object.
    VX_EXPORT void ReleaseLibrary();

    /**
     * @brief Retrieves a pointer to a function within the loaded library.
     * @param FunctionName The name of the function to retrieve.
     * @return A void pointer to the function, or NULL if the function is not found.
     */
    VX_EXPORT void *GetFunctionPtr(char *FunctionName);

protected:
    /// @brief Handle to the loaded shared library.
    INSTANCE_HANDLE m_LibraryHandle;
};

#endif // VXSHAREDLIBRARY_H
