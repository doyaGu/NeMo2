#ifndef VXTHREAD_H
#define VXTHREAD_H

#include "XString.h"
#include "XHashTable.h"
#include "VxMutex.h"

#define VXTERROR_TIMEOUT		9	///< The timeout for an operation was reached.
#define VXTERROR_NULLTHREAD		50	///< The thread object is null.
#define VXTERROR_WAIT			51	///< An error occurred while waiting for a thread to join.
#define VXTERROR_EXITCODE		52	///< An error occurred while trying to get the exit code of a thread.
#define VXT_OK					53	///< No error occurred.

/**
 * @enum VXTHREAD_STATE
 * @brief Describes the current state of a VxThread object.
 * @see VxThread
 */
typedef enum VXTHREAD_STATE {
    VXTS_INITIALE = 0x00000000L, ///< Initial state before the system thread is created.
    VXTS_MAIN     = 0x00000001L, ///< The thread object represents the main application thread.
    VXTS_CREATED  = 0x00000002L, ///< The system thread has been created successfully.
    VXTS_STARTED  = 0x00000004L, ///< The thread has been started and is potentially running.
    VXTS_JOINABLE = 0x00000008L  ///< The thread can be waited upon using `Join()`.
} VXTHREAD_STATE;

/**
 * @enum VXTHREAD_PRIORITY
 * @brief Defines possible priority levels for a thread.
 * @see VxThread, VxThread::SetPriority
 */
typedef enum VXTHREAD_PRIORITY {
    VXTP_NORMAL       = 0, ///< Normal priority.
    VXTP_ABOVENORMAL  = 1, ///< Priority above normal.
    VXTP_BELOWNORMAL  = 2, ///< Priority below normal.
    VXTP_HIGHLEVEL    = 3, ///< High priority.
    VXTP_LOWLEVEL     = 4, ///< Low priority.
    VXTP_IDLE         = 5, ///< Idle priority, runs only when the system is idle.
    VXTP_TIMECRITICAL = 6  ///< Time-critical priority, for tasks that need immediate attention.
} VXTHREAD_PRIORITY;

/// @brief Special exit code indicating that a thread is still active.
const unsigned int VXT_STILLACTIVE = 1000000;

/// @brief Special exit code indicating that a thread was forcibly terminated.
const unsigned int VXT_TERMINATEFORCED = 1000001;

/**
 * @brief Defines the function prototype for a thread's execution routine.
 * @param args A void pointer to arguments passed to the thread function.
 * @return An exit code for the thread. A value greater than 0 is user-defined.
 * @see VxThread, VxThread::Run
 */
typedef unsigned int VxThreadFunction(void *args);

/**
 * @brief Represents a system thread of execution.
 *
 * @remarks
 * There are two ways to use this class:
 * 1. Inherit from VxThread and override the `Run()` method with your thread's logic.
 * 2. Create a VxThread instance and provide a `VxThreadFunction` pointer during creation.
 *
 * @see VxThread::Run, VxThread::CreateThread, VxMutex
 */
class VxThread {
public:
    /// @brief Default constructor.
    VX_EXPORT VxThread();
    /// @brief Virtual destructor.
    VX_EXPORT virtual ~VxThread();

    /**
     * @brief Creates and starts the system thread.
     * @param func A pointer to a VxThreadFunction to execute. If NULL, the virtual `Run()` method is used.
     * @param args A pointer to arguments to be passed to the thread function.
     * @return TRUE if the thread was created successfully, FALSE otherwise.
     */
    VX_EXPORT XBOOL CreateThread(VxThreadFunction *func = 0, void *args = 0);

    /**
     * @brief Sets the priority of the thread.
     * @param priority A value from the `VXTHREAD_PRIORITY` enumeration.
     */
    VX_EXPORT void SetPriority(unsigned int priority);

    /// @brief Sets a descriptive name for the thread.
    VX_EXPORT void SetName(const char *name);

    /// @brief Closes the thread handle, releasing system resources.
    VX_EXPORT void Close();

    /// @brief Gets the name of the thread.
    VX_EXPORT const XString &GetName() const;

    /// @brief Gets the priority of the thread.
    VX_EXPORT unsigned int GetPriority() const;

    /// @brief Checks if the system thread has been created.
    VX_EXPORT XBOOL IsCreated() const;

    /// @brief Checks if the thread is joinable.
    VX_EXPORT XBOOL IsJoinable() const;

    /// @brief Checks if this object represents the main thread.
    VX_EXPORT XBOOL IsMainThread() const;

    /// @brief Checks if the thread has been started.
    VX_EXPORT XBOOL IsStarted() const;

    /// @brief Gets the VxThread object for the currently executing thread.
    VX_EXPORT static VxThread *GetCurrentVxThread();

    /**
     * @brief Waits for the thread to terminate.
     * @param status Pointer to an unsigned int to receive the thread's exit code. Can be NULL.
     * @param timeout The maximum time to wait in milliseconds. 0 means wait indefinitely.
     * @return An error code, such as VXT_OK or VXTERROR_TIMEOUT.
     */
    VX_EXPORT int Wait(unsigned int *status = 0, unsigned int timeout = 0);

    /// @brief Gets the native handle of the thread.
    VX_EXPORT const GENERIC_HANDLE GetHandle() const;

    /// @brief Gets the unique ID of the thread.
    VX_EXPORT XULONG GetID() const;

    /**
     * @brief Retrieves the exit code of a terminated thread.
     * @param status Reference to an unsigned int to receive the exit code. If the thread is still active, it will be set to `VXT_STILLACTIVE`.
     * @return TRUE if the exit code was retrieved successfully, FALSE otherwise.
     */
    VX_EXPORT XBOOL GetExitCode(unsigned int &status);

    /**
     * @brief Forcibly terminates the thread.
     * @param status Pointer to an unsigned int to receive the termination status. Can be NULL.
     * @return TRUE if termination was successful, FALSE otherwise.
     * @warning Forcible termination should be used with extreme caution as it can lead to resource leaks and deadlocks.
     */
    VX_EXPORT XBOOL Terminate(unsigned int *status = 0);

    /// @brief Gets the unique ID of the currently executing thread.
    VX_EXPORT static XULONG GetCurrentVxThreadId();

protected:
    /**
     * @brief The main execution method for the thread if no `VxThreadFunction` is provided.
     * @return The exit code of the thread. By default, this method returns `VXT_OK`.
     * @remarks If you create a thread without a `VxThreadFunction`, you must create a derived class
     * that overrides this `Run` method.
     * @see CreateThread, VxThreadFunction
     */
    virtual unsigned int Run() { return VXT_OK; }

    XString m_Name; ///< The name of the thread.

private:
    /// @brief Private copy constructor to prevent copying.
    VxThread(const VxThread &);
    /// @brief Private assignment operator to prevent assignment.
    VxThread &operator=(const VxThread &);

    /// @brief Internal method to set the thread's priority.
    void SetPriority();

    /// @brief Gets the global mutex for managing the thread list.
    static VxMutex &GetMutex();

    /// @brief Gets the global hash table mapping thread handles to VxThread objects.
    static XHashTable<VxThread *, GENERIC_HANDLE> &GetHashThread();

    /// @brief The internal static function that the OS calls to start the thread.
    static XULONG __stdcall ThreadFunc(void *args);

    GENERIC_HANDLE m_Thread;     ///< The native thread handle.
    unsigned int m_ThreadID;     ///< The unique thread identifier.
    unsigned int m_State;        ///< The current state of the thread, from `VXTHREAD_STATE`.
    unsigned int m_Priority;     ///< The priority of the thread, from `VXTHREAD_PRIORITY`.
    VxThreadFunction *m_Func;    ///< Pointer to the external function to execute.
    void *m_Args;                ///< Pointer to the arguments for the thread function.

    static VxThread *m_MainThread; ///< Static pointer to the main thread object.
};

#endif // VXTHREAD_H
