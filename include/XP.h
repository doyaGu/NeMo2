#ifndef XP_H
#define XP_H

/**
 * @class XP
 * @brief A simple smart pointer class for single objects (auto_ptr-like).
 *
 * @tparam T The type of the object being pointed to.
 *
 * @remarks
 * XP objects automatically manage the lifetime of a dynamically allocated object,
 * deleting it when the XP object goes out of scope. Ownership of the pointer
 * is transferred on copy construction or assignment. This class uses `delete`,
 * so for arrays allocated with `new[]`, use the `XAP` class.
 *
 * @see XAP
 */
template <class T>
class XP {
public:
    /**
     * @brief Constructs an XP object, taking ownership of the provided pointer.
     * @param p A pointer to a dynamically allocated object.
     */
    explicit XP(T *p) : m_Pointee(p) {}

    /**
     * @brief Destructor. Deletes the managed object.
     */
    ~XP() { delete m_Pointee; }

    /**
     * @brief Copy constructor. Transfers ownership of the pointer from another XP object.
     * @param a The source XP object, which will release ownership.
     */
    XP(XP<T> &a) : m_Pointee(a.Release()) {}

    /**
     * @brief Assignment operator. Transfers ownership from another XP object.
     * @param a The source XP object, which will release ownership.
     * @return A reference to this object.
     */
    XP<T> &operator=(XP<T> &a) {
        if (this != &a)
            Set(a.Release());
        return *this;
    }

    /**
     * @brief Provides member access to the managed object.
     * @return A pointer to the managed object.
     */
    T *operator->() const { return m_Pointee; }

    /**
     * @brief Dereferences the pointer to the managed object.
     * @return A reference to the managed object.
     */
    T &operator*() const { return *m_Pointee; }

    /**
     * @brief Allows implicit conversion to the raw pointer type.
     * @return The raw pointer to the managed object.
     */
    operator T *() const { return m_Pointee; }

    /**
     * @brief Releases ownership of the managed pointer.
     * @return The raw pointer, which is no longer managed by this XP object.
     *         The caller is now responsible for deleting the pointer.
     */
    T *Release() {
        T *oldp = m_Pointee;
        m_Pointee = 0;
        return oldp;
    }

    /**
     * @brief Takes ownership of a new pointer, deleting the previously managed one.
     * @param p The new pointer to manage. Defaults to NULL.
     */
    void Set(T *p = NULL) {
        if (m_Pointee != p) {
            delete m_Pointee;
            m_Pointee = p;
        }
    }

protected:
    /// @brief The raw pointer to the managed object.
    T *m_Pointee;
};

/**
 * @class XAP
 * @brief A simple smart pointer class for dynamically allocated arrays (auto_ptr-like).
 *
 * @tparam T The type of the elements in the array.
 *
 * @remarks
 * XAP objects automatically manage the lifetime of a dynamically allocated array,
 * deleting it using `delete[]` when the XAP object goes out of scope. Ownership
 * of the pointer is transferred on copy construction or assignment. For single
 * objects allocated with `new`, use the `XP` class.
 *
 * @see XP
 */
template <class T>
class XAP {
public:
    /**
     * @brief Constructs an XAP object, taking ownership of the provided pointer.
     * @param p A pointer to a dynamically allocated array.
     */
    explicit XAP(T *p = NULL) : m_Pointee(p) {}

    /**
     * @brief Destructor. Deletes the managed array using `delete[]`.
     */
    ~XAP() { delete[] m_Pointee; }

    /**
     * @brief Copy constructor. Transfers ownership of the pointer from another XAP object.
     * @param a The source XAP object, which will release ownership.
     */
    XAP(XAP<T> &a) : m_Pointee(a.Release()) {}

    /**
     * @brief Assignment operator. Transfers ownership from another XAP object.
     * @param a The source XAP object, which will release ownership.
     * @return A reference to this object.
     */
    XAP<T> &operator=(XAP<T> &a) {
        if (this != &a)
            Set(a.Release());
        return *this;
    }

    /**
     * @brief Dereferences the pointer to the managed array.
     * @return A reference to the first element of the array.
     */
    T &operator*() const { return *m_Pointee; }

    /**
     * @brief Allows implicit conversion to the raw pointer type.
     * @return The raw pointer to the managed array.
     */
    operator T *() const { return m_Pointee; }

    /**
     * @brief Releases ownership of the managed pointer.
     * @return The raw pointer, which is no longer managed by this XAP object.
     *         The caller is now responsible for deleting the array.
     */
    T *Release() {
        T *oldp = m_Pointee;
        m_Pointee = 0;
        return oldp;
    }

    /**
     * @brief Takes ownership of a new pointer, deleting the previously managed one.
     * @param p The new pointer to the array to manage. Defaults to NULL.
     */
    void Set(T *p = 0) {
        if (m_Pointee != p) {
            delete[] m_Pointee;
            m_Pointee = p;
        }
    }

protected:
    /// @brief The raw pointer to the managed array.
    T *m_Pointee;
};

#endif // XP_H
