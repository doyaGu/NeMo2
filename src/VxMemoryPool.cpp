#include "VxMemoryPool.h"
#include <cstdint>

void *VxNewAligned(int size, int align) {
    XBYTE *ptr = new XBYTE[size + align + 4];
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (~static_cast<uintptr_t>(align - 1) & (addr + align + 3));
    *(reinterpret_cast<XBYTE**>(aligned - 4)) = ptr;
    return reinterpret_cast<void*>(aligned);
}

void VxDeleteAligned(void *ptr) {
    if (ptr) delete[] *((XBYTE **) ptr - 1);
}
