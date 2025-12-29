#include "VxThread.h"

#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>

VxThread *VxThread::m_MainThread = NULL;

VxThread::VxThread() : m_Name(), m_Thread(NULL), m_ThreadID(0), m_Priority(0), m_Func(NULL), m_Args(NULL) {
    m_State = VXTS_JOINABLE;
}

VxThread::~VxThread() {
    GetMutex().EnterMutex();
    GetHashThread().Remove(m_Thread);
    GetMutex().LeaveMutex();
}

XBOOL VxThread::CreateThread(VxThreadFunction *func, void *args) {
    if (IsCreated())
        return TRUE;

    m_Func = func;
    m_Args = args;

    pthread_t thread;
    int result = pthread_create(&thread, NULL, (void *(*)(void *))ThreadFunc, this);
    if (result != 0)
        return FALSE;
    m_Thread = (GENERIC_HANDLE)thread;
    m_ThreadID = (unsigned int)(uintptr_t)thread;

    m_State |= VXTS_CREATED;

    if (m_Name.Length() == 0) {
        m_Name = "THREAD_";
        m_Name << (int)(uintptr_t)m_Thread;
    }

    GetMutex().EnterMutex();
    GetHashThread().Insert(m_Thread, this);
    GetMutex().LeaveMutex();

    SetPriority();

    return TRUE;
}

void VxThread::SetPriority(unsigned int priority) {
    m_Priority = priority;
    if (IsCreated())
        SetPriority();
}

void VxThread::SetName(const char *name) {
    m_Name = name;
}

void VxThread::Close() {
    // On POSIX, pthread_t resources are released by pthread_join or pthread_detach
    if (m_Thread) {
        pthread_detach((pthread_t)m_Thread);
    }
    m_ThreadID = 0;
    m_Thread = NULL;
    m_State = 0;
    m_Priority = 0;
    m_Func = NULL;
    m_Args = NULL;
}

const XString &VxThread::GetName() const {
    return m_Name;
}

unsigned int VxThread::GetPriority() const {
    return m_Priority;
}

XBOOL VxThread::IsCreated() const {
    return (m_State & VXTS_CREATED) != 0;
}

XBOOL VxThread::IsJoinable() const {
    return (m_State & VXTS_JOINABLE) != 0;
}

XBOOL VxThread::IsMainThread() const {
    return (m_State & VXTS_MAIN) != 0;
}

XBOOL VxThread::IsStarted() const {
    return (m_State & VXTS_STARTED) != 0;
}

VxThread *VxThread::GetCurrentVxThread() {
    GENERIC_HANDLE currentThread = (GENERIC_HANDLE)pthread_self();

    GetMutex().EnterMutex();
    XHashTable<VxThread *, GENERIC_HANDLE>::Iterator it = GetHashThread().Find(currentThread);
    if (it == GetHashThread().End()) {
        GetMutex().LeaveMutex();
        return NULL;
    }

    GetMutex().LeaveMutex();
    return *it;
}

int VxThread::Wait(unsigned int *status, unsigned int timeout) {
    if (!m_Thread)
        return VXTERROR_NULLTHREAD;

    if (timeout == 0) {
        // Infinite wait
        void *retval = NULL;
        int result = pthread_join((pthread_t)m_Thread, &retval);
        if (result != 0)
            return VXTERROR_WAIT;
        if (status)
            *status = (unsigned int)(uintptr_t)retval;
    } else {
        // Timed wait using pthread_timedjoin_np on Linux, or polling on other POSIX
#if defined(__linux__) && defined(_GNU_SOURCE)
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeout / 1000;
        ts.tv_nsec += (timeout % 1000) * 1000000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        void *retval = NULL;
        int result = pthread_timedjoin_np((pthread_t)m_Thread, &retval, &ts);
        if (result == ETIMEDOUT)
            return VXTERROR_TIMEOUT;
        if (result != 0)
            return VXTERROR_WAIT;
        if (status)
            *status = (unsigned int)(uintptr_t)retval;
#else
        // Fallback: just do a regular join (no timeout support)
        void *retval = NULL;
        int result = pthread_join((pthread_t)m_Thread, &retval);
        if (result != 0)
            return VXTERROR_WAIT;
        if (status)
            *status = (unsigned int)(uintptr_t)retval;
#endif
    }

    return VXT_OK;
}

const GENERIC_HANDLE VxThread::GetHandle() const {
    return m_Thread;
}

XULONG VxThread::GetID() const {
    return m_ThreadID;
}

XBOOL VxThread::GetExitCode(unsigned int &status) {
    // POSIX doesn't have a direct equivalent of GetExitCode for running threads
    // We can only get the exit code after pthread_join
    if (m_State & VXTS_STARTED) {
        status = VXT_STILLACTIVE;
        return TRUE;
    }
    // If thread has finished, the exit code would have been retrieved via Wait()
    return FALSE;
}

XBOOL VxThread::Terminate(unsigned int *status) {
    // pthread_cancel is the POSIX equivalent, but it's not recommended
    // as it can leave resources in an inconsistent state
    int result = pthread_cancel((pthread_t)m_Thread);
    if (status)
        *status = VXT_TERMINATEFORCED;
    return result == 0;
}

XULONG VxThread::GetCurrentVxThreadId() {
    return (XULONG)(uintptr_t)pthread_self();
}

void VxThread::SetPriority() {
    if (!m_Thread)
        return;

    // POSIX thread priority (requires SCHED_OTHER or SCHED_FIFO/SCHED_RR)
    // Note: On most systems, changing thread priority requires elevated privileges
    int policy;
    struct sched_param param;
    pthread_getschedparam((pthread_t)m_Thread, &policy, &param);

    int min_priority = sched_get_priority_min(policy);
    int max_priority = sched_get_priority_max(policy);
    int range = max_priority - min_priority;

    switch (m_Priority) {
    case VXTP_NORMAL:
        param.sched_priority = min_priority + range / 2;
        break;
    case VXTP_ABOVENORMAL:
        param.sched_priority = min_priority + range * 3 / 4;
        break;
    case VXTP_BELOWNORMAL:
        param.sched_priority = min_priority + range / 4;
        break;
    case VXTP_HIGHLEVEL:
        param.sched_priority = max_priority;
        break;
    case VXTP_LOWLEVEL:
        param.sched_priority = min_priority;
        break;
    case VXTP_IDLE:
        param.sched_priority = min_priority;
        break;
    case VXTP_TIMECRITICAL:
        param.sched_priority = max_priority;
        break;
    default:
        return;
    }

    pthread_setschedparam((pthread_t)m_Thread, policy, &param);
}

VxMutex &VxThread::GetMutex() {
    static VxMutex threadMutex;
    return threadMutex;
}

XHashTable<VxThread *, GENERIC_HANDLE> &VxThread::GetHashThread() {
    static XHashTable<VxThread *, GENERIC_HANDLE> hashThread;
    return hashThread;
}

XULONG VxThread::ThreadFunc(void *args) {
    if (!args)
        return VXTERROR_NULLTHREAD;

    VxThread *thread = (VxThread *) args;
    thread->m_State |= VXTS_STARTED;

    XULONG ret;
    if (thread->m_Func)
        ret = thread->m_Func(thread->m_Args);
    else
        ret = thread->Run();

    thread->m_State &= ~VXTS_STARTED;
    return ret;
}
