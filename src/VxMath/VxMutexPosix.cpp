#include "VxMutex.h"

#include <pthread.h>

VxMutex::VxMutex() {
    m_Mutex = (void *) new pthread_mutex_t;
    pthread_mutex_init((pthread_mutex_t *) m_Mutex, nullptr);
}

VxMutex::~VxMutex() {
    pthread_mutex_destroy((pthread_mutex_t *) m_Mutex);
    delete (pthread_mutex_t *) m_Mutex;
}

XBOOL VxMutex::EnterMutex() {
    pthread_mutex_lock((pthread_mutex_t *) m_Mutex);
    return TRUE;
}

XBOOL VxMutex::LeaveMutex() {
    pthread_mutex_unlock((pthread_mutex_t *) m_Mutex);
    return TRUE;
}

XBOOL VxMutex::operator++(int) {
    return EnterMutex();
}

XBOOL VxMutex::operator--(int) {
    return LeaveMutex();
}
