#include "VxMutex.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

VxMutex::VxMutex() {
    m_Mutex = (void *) new CRITICAL_SECTION;
    InitializeCriticalSection((LPCRITICAL_SECTION) m_Mutex);
}

VxMutex::~VxMutex() {
    DeleteCriticalSection((LPCRITICAL_SECTION) m_Mutex);
    delete (CRITICAL_SECTION *) m_Mutex;
}

XBOOL VxMutex::EnterMutex() {
    EnterCriticalSection((LPCRITICAL_SECTION) m_Mutex);
    return TRUE;
}

XBOOL VxMutex::LeaveMutex() {
    LeaveCriticalSection((LPCRITICAL_SECTION) m_Mutex);
    return TRUE;
}

XBOOL VxMutex::operator++(int) {
    return EnterMutex();
}

XBOOL VxMutex::operator--(int) {
    return LeaveMutex();
}
