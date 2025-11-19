#include "VxThread.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

VxThread::VxThread() : m_Name(), m_Thread(NULL), m_Priority(0), m_Func(NULL), m_Args(NULL) {
    m_State = VXTS_JOINABLE;
}

VxThread::~VxThread() {
    GetHashThread().Remove(m_Thread);
}

XBOOL VxThread::CreateThread(VxThreadFunction *func, void *args) {
    if (IsCreated())
        return TRUE;

    m_Func = func;
    m_Args = args;
    m_Thread = ::CreateThread(NULL, 0, ThreadFunc, this, 0, (LPDWORD) &m_ThreadID);
    if (!m_Thread)
        return FALSE;

    if (m_Name.Length() == 0) {
        m_Name = "THREAD_";
        m_Name << *(int *) &m_Thread;
    }

    GetMutex().EnterMutex();
    GetHashThread().Insert(m_Thread, this);
    GetMutex().LeaveMutex();

    SetPriority();

    return TRUE;
}

void VxThread::SetPriority(unsigned int priority) {
    if (IsCreated())
        m_Priority = priority;
}

void VxThread::SetName(const char *name) {
    if (IsCreated())
        m_Name = name;
}

void VxThread::Close() {
    if (m_Thread)
        ::CloseHandle((HANDLE) m_Thread);
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
    HANDLE currentThread = ::GetCurrentThread();
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
    DWORD dwMilliseconds = (timeout != 0) ? timeout : -1;
    switch (::WaitForSingleObject(m_Thread, dwMilliseconds)) {
    case WAIT_ABANDONED:
        return VXTERROR_WAIT;
    case WAIT_TIMEOUT:
        return VXTERROR_TIMEOUT;
    case WAIT_FAILED:
        return VXTERROR_WAIT;
    default:
        break;
    }

    if (*status == VXTS_INITIALE || GetExitCode(*status))
        return VXTERROR_WAIT;
    return VXTERROR_EXITCODE;
}

const GENERIC_HANDLE VxThread::GetHandle() const {
    return m_Thread;
}

XULONG VxThread::GetID() const {
    return m_ThreadID;
}

XBOOL VxThread::GetExitCode(unsigned int &status) {
    if (!::GetExitCodeThread(m_Thread, (LPDWORD) &status))
        return FALSE;

    if (status == STILL_ACTIVE)
        status = VXT_STILLACTIVE;
    return TRUE;
}

XBOOL VxThread::Terminate(unsigned int *status) {
    if (status)
        return ::TerminateThread((HANDLE) m_Thread, *status);
    else
        return ::TerminateThread((HANDLE) m_Thread, 0);
}

XULONG VxThread::GetCurrentVxThreadId() {
    return ::GetCurrentThreadId();
}

void VxThread::SetPriority() {
    switch (m_Priority) {
    case VXTP_NORMAL:
        ::SetThreadPriority(m_Thread, THREAD_PRIORITY_NORMAL);
        break;
    case VXTP_ABOVENORMAL:
        ::SetThreadPriority(m_Thread, THREAD_PRIORITY_ABOVE_NORMAL);
        break;
    case VXTP_BELOWNORMAL:
        ::SetThreadPriority(m_Thread, THREAD_PRIORITY_BELOW_NORMAL);
        break;
    case VXTP_HIGHLEVEL:
        ::SetThreadPriority(m_Thread, THREAD_PRIORITY_HIGHEST);
        break;
    case VXTP_LOWLEVEL:
        ::SetThreadPriority(m_Thread, THREAD_PRIORITY_LOWEST);
        break;
    case VXTP_IDLE:
        ::SetThreadPriority(m_Thread, THREAD_PRIORITY_IDLE);
        break;
    case VXTP_TIMECRITICAL:
        ::SetThreadPriority(m_Thread, THREAD_PRIORITY_TIME_CRITICAL);
    default:
        break;
    }
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

    thread->m_State = VXTS_INITIALE;
    return ret;
}
