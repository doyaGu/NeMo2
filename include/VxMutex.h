#ifndef VXMUTEX_H
#define VXMUTEX_H

#include "VxMathDefines.h"

/**
 * @brief Represents a mutual-exclusion synchronization object.
 * @remarks A mutex is used to protect shared resources from being simultaneously
 * accessed by multiple threads.
 * @see VxThread, VxMutexLock
 */
class VxMutex {
public:
    /// @brief Constructs the mutex object.
    VX_EXPORT VxMutex();
    /// @brief Destroys the mutex object.
    VX_EXPORT ~VxMutex();

    /**
     * @brief Acquires a lock on the mutex, blocking if necessary.
     * @return TRUE if the lock was acquired successfully, FALSE otherwise.
     */
    VX_EXPORT XBOOL EnterMutex();

    /**
     * @brief Releases the lock on the mutex.
     * @return TRUE if the lock was released successfully, FALSE otherwise.
     */
    VX_EXPORT XBOOL LeaveMutex();

    /**
     * @brief Acquires a lock on the mutex (postfix increment operator).
     * @return The result of EnterMutex().
     */
    VX_EXPORT XBOOL operator++(int);

    /**
     * @brief Releases the lock on the mutex (postfix decrement operator).
     * @return The result of LeaveMutex().
     */
    VX_EXPORT XBOOL operator--(int);

private:
    void *m_Mutex; ///< A handle to the underlying OS-specific mutex object.

    /// @brief Private copy constructor to prevent copying.
    VxMutex(const VxMutex &);
    /// @brief Private assignment operator to prevent assignment.
    VxMutex &operator=(const VxMutex &);
};

/**
 * @brief A RAII-style lock guard for a VxMutex.
 * @remarks This class acquires a mutex lock upon construction and releases it
 * upon destruction, ensuring the mutex is properly unlocked even if exceptions occur.
 * @see VxThread, VxMutex
 */
class VxMutexLock {
    VxMutex &m_Mutex;

public:
    /**
     * @brief Constructs the lock guard and acquires a lock on the provided mutex.
     * @param Mutex The VxMutex object to lock.
     */
    VxMutexLock(VxMutex &Mutex) : m_Mutex(Mutex) { m_Mutex++; }
    /**
     * @brief Destructs the lock guard and releases the mutex lock.
     */
    ~VxMutexLock() { m_Mutex--; }
};

/**
 * @brief A template class that bundles data with a mutex to ensure thread-safe access.
 * @tparam T The type of the data to be protected.
 * @remarks To ensure thread-safe access to your data, wrap it in this class.
 * Then, to modify or read the data, create a scoped Accessor or ConstAccessor object.
 * The lock is held for the lifetime of the accessor object.
 *
 * @example
 * @code
 *   VxDataMutexed<int> shared_value;
 *
 *   // In a thread that writes to the value:
 *   {
 *     VxDataMutexed<int>::Accessor access(&shared_value);
 *     access.Value() = 42;
 *   } // Mutex is automatically released here
 *
 *   // In a thread that reads the value:
 *   int local_copy;
 *   {
 *     VxDataMutexed<int>::ConstAccessor access(&shared_value);
 *     local_copy = access.Value();
 *   } // Mutex is automatically released here
 * @endcode
 *
 * @see VxThread, VxMutexLock, VxMutex
 */
template <class T>
class VxDataMutexed {
public:
    /// @brief Default constructor.
    VxDataMutexed() {}

    /**
     * @brief Initialization constructor.
     * @param value The initial value for the protected data.
     * @remarks The type T must have a copy assignment operator.
     */
    VxDataMutexed(const T &value) {
        m_Value = value;
    }

    /**
     * @brief A nested class that provides thread-safe, mutable access to the protected data.
     */
    class Accessor {
    public:
        /**
         * @brief Constructs an accessor and locks the associated VxDataMutexed object.
         * @param dm Pointer to the VxDataMutexed object to access.
         */
        Accessor(VxDataMutexed<T> *dm) {
            m_DataM = dm;
            m_DataM->m_Mutex.EnterMutex();
        }

        /**
         * @brief Destroys the accessor and unlocks the associated VxDataMutexed object.
         */
        ~Accessor() {
            m_DataM->m_Mutex.LeaveMutex();
        }

        /**
         * @brief Provides a reference to the protected data.
         * @return A mutable reference to the value.
         */
        T &Value() {
            return m_DataM->m_Value;
        }

    private:
        /// @brief Private default constructor.
        Accessor();
        /// @brief Private copy constructor.
        Accessor(const Accessor &);
        /// @brief Private assignment operator.
        Accessor &operator=(const Accessor &);

        VxDataMutexed<T> *m_DataM;
    };

    /**
     * @brief A nested class that provides thread-safe, const access to the protected data.
     */
    class ConstAccessor {
    public:
        /**
         * @brief Constructs an accessor and locks the associated VxDataMutexed object.
         * @param dm Pointer to the VxDataMutexed object to access.
         */
        ConstAccessor(const VxDataMutexed<T> *dm) {
            m_DataM = dm;
            m_DataM->m_Mutex.EnterMutex();
        }

        /**
         * @brief Destroys the accessor and unlocks the associated VxDataMutexed object.
         */
        ~ConstAccessor() {
            m_DataM->m_Mutex.LeaveMutex();
        }

        /**
         * @brief Provides a const reference to the protected data.
         * @return A const reference to the value.
         */
        const T &Value() const {
            return m_DataM->m_Value;
        }

    private:
        /// @brief Private default constructor.
        ConstAccessor();
        /// @brief Private copy constructor.
        ConstAccessor(const ConstAccessor &);
        /// @brief Private assignment operator.
        ConstAccessor &operator=(const ConstAccessor &);

        const VxDataMutexed<T> *m_DataM;
    };

    mutable VxMutex m_Mutex; ///< The mutex protecting the data.

    T m_Value; ///< The data being protected.

private:
    /// @brief Private copy constructor to prevent copying of the container itself.
    VxDataMutexed(const VxDataMutexed &);
    /// @brief Private assignment operator to prevent assignment.
    VxDataMutexed &operator=(const VxDataMutexed &);
};

#endif // VXMUTEX_H
