#ifndef XUTIL_H
#define XUTIL_H

#include "VxMathDefines.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/// @brief A macro for assertions, enabled in debug builds.
#define XASSERT(a) assert(a)

/// @brief A function pointer type for comparison functions used in sorting.
typedef int (__cdecl *VxSortFunc)(const void *elem1, const void *elem2);

/**
 * @brief Selects one of two references based on a condition.
 * @tparam T The type of the references.
 * @param iCond The boolean condition.
 * @param iT1 The reference to return if `iCond` is TRUE.
 * @param iT2 The reference to return if `iCond` is FALSE.
 * @return `iT1` if `iCond` is true, otherwise `iT2`.
 */
template <class T>
const T &ChooseRef(XBOOL iCond, const T &iT1, const T &iT2) {
    return iCond ? iT1 : iT2;
}

/**
 * @brief Clamps a value to a specified interval.
 * @tparam T The type of the value.
 * @param t The value to clamp (passed by reference).
 * @param min The minimum allowed value.
 * @param max The maximum allowed value.
 * @see XMin, XMax
 */
template <class T>
void XThreshold(T &t, const T &min, const T &max) {
    if (t < min) t = min;
    else if (t > max) t = max;
}

/**
 * @brief Returns the minimum of two values.
 * @tparam T The type of the values.
 * @param a The first value.
 * @param b The second value.
 * @return The smaller of the two values.
 * @see XMax
 */
template <class T>
const T &XMin(const T &a, const T &b) {
    return (a < b) ? a : b;
}

/**
 * @brief Returns the maximum of two values.
 * @tparam T The type of the values.
 * @param a The first value.
 * @param b The second value.
 * @return The larger of the two values.
 * @see XMin
 */
template <class T>
const T &XMax(const T &a, const T &b) {
    return (a > b) ? a : b;
}

/**
 * @brief Finds the minimum and maximum of two values.
 * @tparam T The type of the values.
 * @param a The first value.
 * @param b The second value.
 * @param min Output reference for the minimum value.
 * @param max Output reference for the maximum value.
 * @see XMin, XMax
 */
template <class T>
void XMinMax(const T &a, const T &b, T &min, T &max) {
    if (a < b) { min = a; max = b; }
    else { min = b; max = a; }
}

/**
 * @brief Returns the minimum of three values.
 * @tparam T The type of the values.
 * @return The smallest of the three values.
 * @see XMax
 */
template <class T>
const T &XMin(const T &a, const T &b, const T &c) {
    return (a < b) ? ((c < a) ? c : a) : ((c < b) ? c : b);
}

/**
 * @brief Returns the maximum of three values.
 * @tparam T The type of the values.
 * @return The largest of the three values.
 * @see XMin
 */
template <class T>
const T &XMax(const T &a, const T &b, const T &c) {
    return (a > b) ? ((c > a) ? c : a) : ((c > b) ? c : b);
}

/**
 * @brief Finds the minimum and maximum of three values.
 * @tparam T The type of the values.
 * @param a The first value.
 * @param b The second value.
 * @param c The third value.
 * @param min Output reference for the minimum value.
 * @param max Output reference for the maximum value.
 * @see XMin, XMax
 */
template <class T>
void XMinMax(const T &a, const T &b, const T &c, T &min, T &max) {
    if (a < b) {
        if (c < a) { min = c; max = b; }
        else { min = a; max = (b < c) ? c : b; }
    } else {
        if (c < b) { min = c; max = a; }
        else { min = b; max = (a < c) ? c : a; }
    }
}

/**
 * @brief Finds the minimum, median, and maximum of three values.
 * @tparam T The type of the values.
 * @param a The first value.
 * @param b The second value.
 * @param c The third value.
 * @param min Output reference for the minimum value.
 * @param med Output reference for the median value.
 * @param max Output reference for the maximum value.
 */
template <class T>
void XMinMax(const T &a, const T &b, const T &c, T &min, T &med, T &max) {
    if (a < b) {
        if (c < a) { min = c; med = a; max = b; }
        else { min = a; if (b < c) { med = b; max = c; } else { med = c; max = b; } }
    } else {
        if (c < b) { min = c; med = b; max = a; }
        else { min = b; if (a < c) { max = c; med = a; } else { max = a; med = c; } }
    }
}

/**
 * @brief Swaps the values of two variables.
 * @tparam T The type of the variables.
 * @param a The first variable.
 * @param b The second variable.
 */
template <class T>
void XSwap(T &a, T &b) {
    T c = a;
    a = b;
    b = c;
}

/**
 * @brief Returns the absolute value of a number.
 * @tparam T The type of the number.
 */
template <class T>
T XAbs(T a) {
    return (a > 0) ? a : -a;
}

/// @brief Template specialization of `XAbs` for `float`.
template <>
inline float XAbs<float>(float a) { return (float) ::fabs(a); }

/**
 * @brief Returns the absolute value of a float.
 */
inline float XFabs(float a) {
    return (float) ::fabs(a);
}

/**
 * @brief Returns a mask for the lowest set bit in an integer.
 * @param v The integer value.
 * @return An integer with only the lowest bit of `v` set.
 */
inline int LowestBitMask(int v) {
    return (v & -v);
}

/**
 * @brief Checks if a number is a power of 2.
 * @param x The integer to check.
 * @return TRUE if `x` is a power of 2, FALSE otherwise.
 */
inline XBOOL Is2Power(int x) {
    return x != 0 && x == LowestBitMask(x);
}

/**
 * @brief Returns the smallest power of 2 that is greater than or equal to `v`.
 * @param v The integer value.
 * @return The nearest superior power of 2.
 */
inline int Near2Power(int v) {
    int i = LowestBitMask(v);
    while (i < v) i <<= 1;
    return i;
}

/**
 * @class XGUID
 * @brief Represents a Globally Unique Identifier.
 * @remarks Comparison operators are defined, allowing `XGUID` objects to be
 * used as keys in sorted containers or hash tables.
 */
class XGUID {
public:
    /**
     * @brief Constructs an XGUID.
     * @param gd1 The first 32-bit part of the GUID.
     * @param gd2 The second 32-bit part of the GUID.
     */
    explicit XGUID(XDWORD gd1 = 0, XDWORD gd2 = 0) : d1(gd1), d2(gd2) {}

    friend XBOOL operator==(const XGUID &v1, const XGUID &v2);
    friend XBOOL operator!=(const XGUID &v1, const XGUID &v2);
    friend XBOOL operator<(const XGUID &v1, const XGUID &v2);
    friend XBOOL operator<=(const XGUID &v1, const XGUID &v2);
    friend XBOOL operator>(const XGUID &v1, const XGUID &v2);
    friend XBOOL operator>=(const XGUID &v1, const XGUID &v2);

    /**
     * @brief Checks if the GUID is valid (not zero).
     * @return TRUE if either part of the GUID is non-zero.
     */
    XBOOL inline IsValid() { return d1 || d2; }

    XDWORD d1; ///< The first 32-bit part of the GUID.
    XDWORD d2; ///< The second 32-bit part of the GUID.
};

inline XBOOL operator==(const XGUID &v1, const XGUID &v2) { return ((v1.d1 == v2.d1) && (v1.d2 == v2.d2)); }
inline XBOOL operator!=(const XGUID &v1, const XGUID &v2) { return ((v1.d1 != v2.d1) || (v1.d2 != v2.d2)); }

inline XBOOL operator<(const XGUID &v1, const XGUID &v2) {
    if (v1.d1 < v2.d1) return true;
    if (v1.d1 == v2.d1) return (v1.d2 < v2.d2);
    return false;
}

inline XBOOL operator<=(const XGUID &v1, const XGUID &v2) {
    return (v1.d1 < v2.d1) || ((v1.d1 == v2.d1) && (v1.d2 <= v2.d2));
}

inline XBOOL operator>(const XGUID &v1, const XGUID &v2) {
    if (v1.d1 > v2.d1) return true;
    if (v1.d1 == v2.d1) return (v1.d2 > v2.d2);
    return false;
}

inline XBOOL operator>=(const XGUID &v1, const XGUID &v2) {
    return (v1.d1 > v2.d1) || ((v1.d1 == v2.d1) && (v1.d2 >= v2.d2));
}

#endif // XUTIL_H
