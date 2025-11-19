#ifndef VXMEMORYPOOL_H
#define VXMEMORYPOOL_H

#include "VxMathDefines.h"

/**
 * @brief Allocates a block of memory aligned to a specified boundary.
 * @param size The size of the memory block to allocate, in bytes.
 * @param align The alignment boundary, which must be a power of two.
 * @return A pointer to the allocated aligned memory, or NULL if allocation fails.
 * @see VxDeleteAligned
 */
VX_EXPORT void *VxNewAligned(int size, int align);

/**
 * @brief Frees a block of memory that was allocated with VxNewAligned.
 * @param ptr A pointer to the memory block to be freed.
 * @see VxNewAligned
 */
VX_EXPORT void VxDeleteAligned(void *ptr);

/**
 * @brief A simple memory buffer manager that ensures aligned memory allocation.
 *
 * @remarks
 * This class manages a memory buffer that is automatically allocated with a
 * specified alignment (16 bytes by default) and deallocated when the object
 * goes out of scope. It is useful for temporary storage that requires SIMD alignment.
 *
 * Do not call `delete` or `VxDeleteAligned` on the pointer returned by the `Buffer()` method;
 * memory is managed by the class's destructor.
 * @see VxNewAligned, VxDeleteAligned
 */
class VxMemoryPool {
public:
    /**
     * @brief Destructor that frees the allocated memory.
     */
    ~VxMemoryPool() {
        VxDeleteAligned(m_Memory);
        m_Memory = NULL;
    }

    /**
     * @brief Constructs the memory pool and optionally allocates an initial buffer.
     * @param size The initial size of the buffer to allocate, in units of XDWORDs (4 bytes).
     */
    VxMemoryPool(int size = 0) {
        m_Memory = NULL;
        m_Allocated = 0;
        Allocate(size);
    }

    /**
     * @brief Returns a pointer to the managed memory buffer.
     * @return A void pointer to the start of the allocated memory.
     */
    void *Buffer() const {
        return m_Memory;
    }

    /**
     * @brief Returns the currently allocated size of the memory buffer.
     * @return The allocated size in units of XDWORDs (4 bytes).
     */
    int AllocatedSize() const {
        return m_Allocated;
    }

    /**
     * @brief Allocates or reallocates the memory buffer.
     * @param size The desired size of the buffer, in units of XDWORDs (4 bytes).
     * @remarks If the requested size is larger than the currently allocated size,
     * the existing buffer is deleted and a new, larger one is allocated.
     * The contents of the old buffer are not preserved.
     */
    void Allocate(int size) {
        if (m_Allocated < size) {
            VxDeleteAligned(m_Memory);
            m_Memory = (XBYTE *) VxNewAligned(size * sizeof(XDWORD), 16);
            m_Allocated = size;
        }
    }

protected:
    XBYTE *m_Memory; ///< Pointer to the aligned memory buffer.
    int m_Allocated; ///< The size of the allocated buffer in XDWORDs.
};

#endif // VXMEMORYPOOL_H
