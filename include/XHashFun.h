#ifndef XHASHFUN_H
#define XHASHFUN_H

#include <ctype.h>

#include "VxMathDefines.h"
#include "XString.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4996)
#endif

/**
 * @file
 * @brief Defines a series of hash function objects and equality comparators
 *        for use with hash table templates.
 */

/**
 * @struct XEqual
 * @brief A generic equality comparison functor.
 * @remarks This template provides a default equality comparison using `operator==`.
 *          It is specialized for various types like C-style strings.
 */
template <class K>
struct XEqual {
    /// @brief Compares two keys for equality.
    int operator()(const K &iK1, const K &iK2) const {
        return (iK1 == iK2);
    }
};

/// @brief Specialization of `XEqual` for `const char*` keys.
template <>
struct XEqual<const char *> {
    /// @brief Compares two C-style strings for equality.
    int operator()(const char *iS1, const char *iS2) const { return !strcmp(iS1, iS2); }
};

/// @brief Specialization of `XEqual` for `char*` keys.
template <>
struct XEqual<char *> {
    /// @brief Compares two C-style strings for equality.
    int operator()(const char *iS1, const char *iS2) const { return !strcmp(iS1, iS2); }
};

/**
 * @struct XEqualXStringI
 * @brief Case-insensitive equality comparator for `XString` keys.
 */
struct XEqualXStringI {
    /// @brief Performs a case-insensitive comparison of two `XString` objects.
    int operator()(const XString &iS1, const XString &iS2) const { return !iS1.ICompare(iS2); }
};

/**
 * @struct XEqualStringI
 * @brief Case-insensitive equality comparator for C-style string keys.
 */
struct XEqualStringI {
    /// @brief Performs a case-insensitive comparison of two C-style strings.
    int operator()(const char *iS1, const char *iS2) const { return !stricmp(iS1, iS2); }
};

/**
 * @struct XHashFun
 * @brief A generic hash function object (functor).
 * @remarks
 * This template provides a default hash function that performs a simple cast to `int`.
 * It is specialized for common types to provide better hash distributions.
 * These functors are designed to be used when declaring a hash table.
 */
template <class K>
struct XHashFun {
    /// @brief Default hash function.
    int operator()(const K &iK) {
        return (int) iK;
    }
};

/**
 * @brief A common string hashing algorithm.
 * @param _s The null-terminated C-style string to hash.
 * @return An integer hash value.
 */
inline int XHashString(const char *_s) {
    unsigned int _h = 0;
    for (; *_s; ++_s)
        _h = 5 * _h + *_s;
    return (int)_h;
}

/**
 * @brief A common case-insensitive string hashing algorithm.
 * @param _s The null-terminated C-style string to hash.
 * @return An integer hash value.
 */
inline int XHashStringI(const char *_s) {
    unsigned int _h = 0;
    for (; *_s; ++_s)
        _h = 5 * _h + tolower(*_s);
    return (int)_h;
}

/// @brief Specialization of `XHashFun` for `char*` keys.
template <>
struct XHashFun<char *> {
    int operator()(const char *_s) const { return XHashString(_s); }
};

/// @brief Specialization of `XHashFun` for `const char*` keys.
template <>
struct XHashFun<const char *> {
    int operator()(const char *_s) const { return XHashString(_s); }
};

/// @brief Specialization of `XHashFun` for `XString` keys.
template <>
struct XHashFun<XString> {
    int operator()(const XString &_s) const { return XHashString(_s.CStr()); }
};

/// @brief Specialization of `XHashFun` for `float` keys.
template <>
struct XHashFun<float> {
    int operator()(const float _x) const { return *(int *) &_x; }
};

/// @brief Specialization of `XHashFun` for `void*` keys.
template <>
struct XHashFun<void *> {
    int operator()(const void *_x) const { return *(int *) _x >> 8; }
};

/// @brief Specialization of `XHashFun` for `XGUID` keys.
template <>
struct XHashFun<XGUID> {
    int operator()(const XGUID &_s) const { return _s.d1; }
};

/**
 * @struct XHashFunStringI
 * @brief Case-insensitive hash function for C-style strings.
 */
struct XHashFunStringI {
    int operator()(const char *_s) const { return XHashStringI(_s); }
};

/**
 * @struct XHashFunXStringI
 * @brief Case-insensitive hash function for `XString` objects.
 */
struct XHashFunXStringI {
    int operator()(const XString &_s) const { return XHashStringI(_s.CStr()); }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // XHASHFUN_H
