#ifndef XSMARTPTR_H
#define XSMARTPTR_H

#if VX_MSVC > 1000
#pragma warning(disable : 4284)
#endif

/**
 * @class XRefCount
 * @brief A base class providing reference counting functionality.
 *
 * @remarks
 * Inherit from this class to make your objects compatible with `XSmartPtr`.
 * This enables shared ownership of objects.
 *
 * @see XSmartPtr, XP
 */
class XRefCount {
public:
    /// @brief The reference counter, marked as mutable to allow modification in const contexts.
    mutable unsigned int m_RefCount;

    /// @brief Virtual destructor.
    virtual ~XRefCount() {}

    /// @brief Default constructor, initializes the reference count to 0.
    XRefCount() : m_RefCount(0) {}

    /// @brief Assignment operator, does not copy the reference count.
    XRefCount &operator=(const XRefCount &) { return *this; }
    /// @brief Copy constructor, initializes the new object's reference count to 0.
    XRefCount(const XRefCount &) { m_RefCount = 0; }
};

/**
 * @class XSmartPtr
 * @brief A smart pointer for shared ownership of objects derived from `XRefCount`.
 *
 * @tparam T The type of the managed object. Must be derived from `XRefCount`.
 *
 * @remarks
 * `XSmartPtr` implements reference counting to manage the lifetime of an object.
 * The object is deleted when the last `XSmartPtr` pointing to it is destroyed or reassigned.
 *
 * @see XRefCount, XP
 */
template <class T>
class XSmartPtr {
public:
    /// @brief Default constructor. Initializes to a null pointer.
    XSmartPtr() : m_Pointee(NULL) {}

    /**
     * @brief Constructor that takes ownership of a raw pointer.
     * @param p A pointer to an object derived from `XRefCount`.
     */
    explicit XSmartPtr(T *p) : m_Pointee(p) { AddRef(); }

    /**
     * @brief Destructor. Decrements the reference count and deletes the object if the count reaches zero.
     */
    ~XSmartPtr() { Release(); }

    /**
     * @brief Copy constructor. Increments the reference count of the shared object.
     * @param a The `XSmartPtr` to copy from.
     */
    XSmartPtr(const XSmartPtr &a) : m_Pointee(a.m_Pointee) { AddRef(); }

    /**
     * @brief Assignment operator from another `XSmartPtr`.
     * @param a The `XSmartPtr` to assign from.
     * @return A reference to this smart pointer.
     */
    XSmartPtr<T> &operator=(const XSmartPtr &a) { return operator=(a.m_Pointee); }

    /**
     * @brief Assignment operator from a raw pointer.
     * @param p A raw pointer to the object to be managed.
     * @return A reference to this smart pointer.
     */
    XSmartPtr<T> &operator=(T *p) {
        if (p) ++p->m_RefCount;
        Release();
        m_Pointee = p;
        return *this;
    }

    /// @brief Provides member access to the managed object.
    T *operator->() const { return m_Pointee; }
    /// @brief Dereferences the pointer to the managed object.
    T &operator*() const { return *m_Pointee; }
    /// @brief Allows implicit conversion to the raw pointer type.
    operator T *() const { return m_Pointee; }

protected:
    /// @brief Increments the reference count of the managed object.
    void AddRef() { if (m_Pointee) ++m_Pointee->m_RefCount; }

    /// @brief Decrements the reference count and deletes the object if it reaches zero.
    void Release() {
        if (m_Pointee && (--(m_Pointee->m_RefCount) == 0))
            delete m_Pointee;
    }

    /// @brief The raw pointer to the managed object.
    T *m_Pointee;
};

/**
 * @class XPtrStrided
 * @brief An iterator-like class for traversing arrays with a custom stride.
 *
 * @tparam T The type of the elements in the strided array.
 *
 * @remarks
 * This class wraps a raw pointer and a stride value, allowing standard
 * iterator operations like `++`, `[]`, and `*` to work correctly on
 * non-contiguous data, such as a specific component within an array of vertices.
 *
 * @see XP
 */
template <class T>
class XPtrStrided {
public:
    /// @brief Default constructor. Initializes to a null pointer and zero stride.
    XPtrStrided() : m_Ptr(NULL), m_Stride(0) {}

    /**
     * @brief Constructs a strided pointer.
     * @param Ptr A void pointer to the first element.
     * @param Stride The byte offset between consecutive elements.
     */
    XPtrStrided(void *Ptr, int Stride) : m_Ptr((unsigned char *) Ptr), m_Stride(Stride) {}

    /// @brief Implicitly converts to a raw pointer of the specified type.
    operator T *() { return (T *) m_Ptr; }

    /// @brief Dereferences the pointer to access the current element.
    T &operator*() { return *(T *) m_Ptr; }
    const T &operator*() const { return *(T *) m_Ptr; }
    /// @brief Provides member access to the current element.
    T *operator->() { return (T *) m_Ptr; }

    /// @brief Provides array-style access with an offset.
    const T &operator[](unsigned short iCount) const { return *(T *) (m_Ptr + iCount * m_Stride); }
    T &operator[](unsigned short iCount) { return *(T *) (m_Ptr + iCount * m_Stride); }
    const T &operator[](int iCount) const { return *(T *) (m_Ptr + iCount * m_Stride); }
    T &operator[](int iCount) { return *(T *) (m_Ptr + iCount * m_Stride); }
    const T &operator[](unsigned int iCount) const { return *(T *) (m_Ptr + iCount * m_Stride); }
    T &operator[](unsigned int iCount) { return *(T *) (m_Ptr + iCount * m_Stride); }
    const T &operator[](unsigned long iCount) const { return *(T *) (m_Ptr + iCount * m_Stride); }
    T &operator[](unsigned long iCount) { return *(T *) (m_Ptr + iCount * m_Stride); }

    /**
     * @brief Pre-increment operator. Advances the pointer to the next element.
     */
    XPtrStrided &operator++() {
        m_Ptr += m_Stride;
        return *this;
    }

    /**
     * @brief Post-increment operator. Advances the pointer to the next element.
     */
    XPtrStrided operator++(int) {
        XPtrStrided tmp = *this;
        m_Ptr += m_Stride;
        return tmp;
    }

    /**
     * @brief Returns a new strided pointer advanced by `n` elements.
     * @param n The number of elements to advance.
     */
    XPtrStrided operator+(int n) { return XPtrStrided(m_Ptr + n * m_Stride, m_Stride); }
    /**
     * @brief Advances the pointer by `n` elements.
     * @param n The number of elements to advance.
     */
    XPtrStrided &operator+=(int n) {
        m_Ptr += n * m_Stride;
        return *this;
    }

private:
    unsigned char *m_Ptr; ///< The current pointer address.
    int m_Stride;         ///< The byte stride between elements.
};

#endif // XSMARTPTR_H
