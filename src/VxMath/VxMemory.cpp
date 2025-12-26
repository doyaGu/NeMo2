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
