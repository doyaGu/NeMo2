#include "VxMemory.h"

// Basic memory allocation functions
void *mynew(unsigned int n) {
    void *ptr = operator new(n);
    return ptr;
}

void mydelete(void *a) {
    if (a) {
        operator delete(a);
    }
}

// Array allocation functions
void *mynewarray(unsigned int n) {
    void *ptr = operator new[](n);
    return ptr;
}

void mydeletearray(void *a) {
    if (a) {
        operator delete[](a);
    }
}

void *VxNewAligned(int size, int align) {
#if VX_COMPILER_MSVC
    return _aligned_malloc(size, align);
#else
    if (align < sizeof(void *)) {
        align = sizeof(void *);
    }
    if ((align & (align - 1)) != 0) {
        size_t pow2 = sizeof(void *);
        while (pow2 < align) {
            pow2 <<= 1;
        }
        align = pow2;
    }
    const size_t totalSize = size + align - 1 + sizeof(void *);
    void *raw = malloc(totalSize);
    if (!raw) {
        return nullptr;
    }
    const uintptr_t rawAddr = reinterpret_cast<uintptr_t>(raw) + sizeof(void *);
    const uintptr_t alignedAddr = (rawAddr + (align - 1)) & ~(static_cast<uintptr_t>(align) - 1);
    reinterpret_cast<void **>(alignedAddr)[-1] = raw;
    return reinterpret_cast<void *>(alignedAddr);
#endif
}

void VxDeleteAligned(void *ptr) {
#if VX_COMPILER_MSVC
    _aligned_free(ptr);
#else
    if (!ptr) {
        return;
    }
    void *raw = reinterpret_cast<void **>(ptr)[-1];
    free(raw);
#endif
}
