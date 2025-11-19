#ifndef VXMEMORY_H
#define VXMEMORY_H

#include <new>

#include "VxMathCompiler.h"

//------ Memory Management

/**
 * @brief Custom memory allocation function.
 * @param n Number of bytes to allocate.
 * @return A pointer to the allocated memory, or null on failure.
 */
VX_EXPORT void *mynew(unsigned int n);

/**
 * @brief Custom memory deallocation function.
 * @param a A pointer to the memory to deallocate.
 */
VX_EXPORT void mydelete(void *a);

/**
 * @brief Custom memory allocation function for arrays.
 * @param n Number of bytes to allocate.
 * @return A pointer to the allocated memory, or null on failure.
 */
VX_EXPORT void *mynewrarray(unsigned int n);

/**
 * @brief Custom memory deallocation function for arrays.
 * @param a A pointer to the memory to deallocate.
 */
VX_EXPORT void mydeletearray(void *a);

//---- Aligned memory allocation

/**
 * @brief Allocates a block of memory with a specified alignment.
 * @param size The size in bytes of the memory to allocate.
 * @param align The alignment boundary. Must be a power of two.
 * @return A pointer to the aligned allocated memory, or null on failure.
 */
VX_EXPORT void *VxNewAligned(int size, int align);

/**
 * @brief Deallocates a block of memory that was allocated with VxNewAligned.
 * @param ptr A pointer to the aligned memory to deallocate.
 */
VX_EXPORT void VxDeleteAligned(void *ptr);

#ifndef VxMalloc
/// @brief Macro to call the custom memory allocation function.
#define VxMalloc(n) mynew(n)
#endif

#ifndef VxFree
/// @brief Macro to call the custom memory deallocation function.
#define VxFree(a) mydelete(a)
#endif

#ifndef VxNew
/// @brief Macro for creating a single object using the custom allocator and placement new.
#define VxNew(x) new (VxMalloc(sizeof(x))) x
#endif

/**
 * @brief Deletes a single object created with VxNew or a similar custom allocator.
 *
 * @details This function correctly calls the object's destructor before freeing the memory.
 * @tparam T The type of the object to delete.
 * @param ptr A pointer to the object to be deleted. It is safe to pass a null pointer.
 */
template <class T>
void VxDelete(T *ptr) {
    if (ptr) {
        void *tmp = ptr;
        (ptr)->~T();
        VxFree(tmp);
    }
}

/**
 * @brief Allocates and constructs an array of objects.
 *
 * @details This function allocates raw memory and then uses placement new to construct each object in the array.
 * @tparam T The type of the objects to allocate.
 * @param cnt The number of objects to allocate in the array.
 * @return A pointer to the beginning of the allocated array, or null on failure.
 */
template <class T>
T *VxAllocate(unsigned int cnt) {
    T *ptr = (T *) VxMalloc(sizeof(T) * cnt);
    if (ptr) {
        for (unsigned int i = 0; i < cnt; ++i)
            new(&ptr[i]) T;
    }
    return ptr;
}

/**
 * @brief Destructs and deallocates an array of objects.
 *
 * @details This function calls the destructor for each object in the array and then frees the raw memory.
 * @tparam T The type of the objects to deallocate.
 * @param ptr A pointer to the array to be deallocated. It is safe to pass a null pointer.
 * @param cnt The number of objects in the array.
 */
template <class T>
void VxDeallocate(T *ptr, unsigned int cnt) {
    if (ptr) {
        void *tmp = ptr;
        for (unsigned int i = 0; i < cnt; ++i)
            (&ptr[i])->~T();
        VxFree(tmp);
    }
}

#endif // VXMEMORY_H
