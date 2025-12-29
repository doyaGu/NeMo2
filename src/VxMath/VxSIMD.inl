//-------------------------------------------------------------------------------------
// VxSIMD.inl -- SIMD Inline Implementations
//
// All SIMD implementations as inline functions following DirectXMath patterns.
// This file is included at the end of VxSIMD.h
//-------------------------------------------------------------------------------------

#pragma once

// Include necessary type headers
#include "VxVector.h"
#include "VxMatrix.h"
#include "VxQuaternion.h"
#include "VxRay.h"
#include "VxPlane.h"
#include "VxRect.h"
#include "Vx2dVector.h"
#include "VxFrustum.h"

#if defined(VX_SIMD_SSE)

//-------------------------------------------------------------------------------------
// Vector Load/Store Operations
//-------------------------------------------------------------------------------------

/**
 * @brief Loads 3 floats into a SIMD register (w component is 0)
 * @remarks Follows DirectXMath XMLoadFloat3 pattern for optimal performance
 */
VX_SIMD_INLINE __m128 VxSIMDLoadFloat3(const float *ptr) noexcept {
    __m128 xy = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(ptr)));
    __m128 z = _mm_load_ss(&ptr[2]);
    return _mm_movelh_ps(xy, z); // [x, y, z, 0]
}

/**
 * @brief Stores the first 3 components of a SIMD register
 * @remarks Follows DirectXMath XMStoreFloat3 pattern for optimal performance
 */
VX_SIMD_INLINE void VxSIMDStoreFloat3(float *ptr, __m128 v) noexcept {
    _mm_store_sd(reinterpret_cast<double *>(ptr), _mm_castps_pd(v));
    __m128 z = _mm_movehl_ps(v, v);
    _mm_store_ss(&ptr[2], z);
}

/**
 * @brief Loads 4 floats into a SIMD register
 */
VX_SIMD_INLINE __m128 VxSIMDLoadFloat4(const float *ptr) noexcept {
    return _mm_loadu_ps(ptr);
}

/**
 * @brief Stores 4 floats from a SIMD register
 */
VX_SIMD_INLINE void VxSIMDStoreFloat4(float *ptr, __m128 v) noexcept {
    _mm_storeu_ps(ptr, v);
}

/**
 * @brief Loads 4 aligned floats into a SIMD register
 */
VX_SIMD_INLINE __m128 VxSIMDLoadFloat4Aligned(const float *ptr) noexcept {
    return _mm_load_ps(ptr);
}

/**
 * @brief Stores 4 aligned floats from a SIMD register
 */
VX_SIMD_INLINE void VxSIMDStoreFloat4Aligned(float *ptr, __m128 v) noexcept {
    _mm_store_ps(ptr, v);
}

//-------------------------------------------------------------------------------------
// Vector Dot Product Operations
//-------------------------------------------------------------------------------------

/**
 * @brief Computes dot product of two 3D vectors (result in all components)
 */
VX_SIMD_INLINE __m128 VxSIMDDotProduct3(__m128 a, __m128 b) noexcept {
#if defined(VX_SIMD_SSE4_1)
    return _mm_dp_ps(a, b, 0x7F);
#else
    __m128 mul = _mm_mul_ps(a, b);
    __m128 sum_xy = _mm_add_ss(mul, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)));
    __m128 sum_xyz = _mm_add_ss(sum_xy, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 2, 2, 2)));
    return _mm_shuffle_ps(sum_xyz, sum_xyz, _MM_SHUFFLE(0, 0, 0, 0));
#endif
}

/**
 * @brief Computes dot product of two 4D vectors (result in all components)
 */
VX_SIMD_INLINE __m128 VxSIMDDotProduct4(__m128 a, __m128 b) noexcept {
#if defined(VX_SIMD_SSE4_1)
    return _mm_dp_ps(a, b, 0xFF);
#else
    __m128 mul = _mm_mul_ps(a, b);
    __m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(mul, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(sums, shuf);
    return _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(0, 0, 0, 0));
#endif
}

//-------------------------------------------------------------------------------------
// Vector Cross Product
//-------------------------------------------------------------------------------------

/**
 * @brief Computes cross product of two 3D vectors
 */
VX_SIMD_INLINE __m128 VxSIMDCrossProduct3(__m128 a, __m128 b) noexcept {
    __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 b_zxy = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 a_zxy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
    return VX_FNMADD_PS(a_zxy, b_yzx, _mm_mul_ps(a_yzx, b_zxy));
}

//-------------------------------------------------------------------------------------
// Reciprocal Square Root Operations
//-------------------------------------------------------------------------------------

/**
 * @brief Computes reciprocal square root (fast approximation)
 */
VX_SIMD_INLINE __m128 VxSIMDReciprocalSqrt(__m128 v) noexcept {
    return _mm_rsqrt_ps(v);
}

/**
 * @brief Computes reciprocal square root (accurate with Newton-Raphson)
 */
VX_SIMD_INLINE __m128 VxSIMDReciprocalSqrtAccurate(__m128 v) noexcept {
    __m128 rsqrt = _mm_rsqrt_ps(v);
    __m128 halfV = _mm_mul_ps(VX_SIMD_NR_HALF, v);
    __m128 rsqrt_sq = _mm_mul_ps(rsqrt, rsqrt);
    __m128 correction = VX_FNMADD_PS(halfV, rsqrt_sq, VX_SIMD_NR_THREE_HALF);
    return _mm_mul_ps(rsqrt, correction);
}

/**
 * @brief Computes reciprocal (1/x) with Newton-Raphson refinement
 */
VX_SIMD_INLINE __m128 VxSIMDReciprocalAccurate(__m128 v) noexcept {
    __m128 rcp = _mm_rcp_ps(v);
    __m128 two = _mm_set1_ps(2.0f);
    return _mm_mul_ps(rcp, VX_FNMADD_PS(v, rcp, two));
}

//-------------------------------------------------------------------------------------
// Vector Normalization Operations
//-------------------------------------------------------------------------------------

/**
 * @brief Normalizes a 3D vector
 */
VX_SIMD_INLINE __m128 VxSIMDNormalize3(__m128 v) noexcept {
    __m128 dot = VxSIMDDotProduct3(v, v);
    __m128 mask = _mm_cmpgt_ps(dot, VX_SIMD_EPSILON);
    __m128 safeDot = _mm_max_ps(dot, VX_SIMD_EPSILON);
    __m128 invLen = VxSIMDReciprocalSqrtAccurate(safeDot);
    __m128 normalized = _mm_mul_ps(v, invLen);
    __m128 keepOriginal = _mm_andnot_ps(mask, v);
    __m128 useNormalized = _mm_and_ps(mask, normalized);
    return _mm_or_ps(keepOriginal, useNormalized);
}

/**
 * @brief Normalizes a 4D vector
 */
VX_SIMD_INLINE __m128 VxSIMDNormalize4(__m128 v) noexcept {
    __m128 dot = VxSIMDDotProduct4(v, v);
    __m128 invLen = VxSIMDReciprocalSqrtAccurate(dot);
    return _mm_mul_ps(v, invLen);
}

//-------------------------------------------------------------------------------------
// Matrix-Vector Multiplication Operations
//-------------------------------------------------------------------------------------

/**
 * @brief Performs matrix-vector multiplication for 3D vector (with translation)
 */
VX_SIMD_INLINE __m128 VxSIMDMatrixMultiplyVector3(const float *mat, __m128 v) noexcept {
    __m128 r0 = _mm_loadu_ps(&mat[0]);
    __m128 r1 = _mm_loadu_ps(&mat[4]);
    __m128 r2 = _mm_loadu_ps(&mat[8]);
    __m128 r3 = _mm_loadu_ps(&mat[12]);

    __m128 v_x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

    __m128 result = VX_FMADD_PS(r2, v_z, r3);
    result = VX_FMADD_PS(r1, v_y, result);
    result = VX_FMADD_PS(r0, v_x, result);
    return result;
}

/**
 * @brief Performs matrix rotation for 3D vector (no translation)
 */
VX_SIMD_INLINE __m128 VxSIMDMatrixRotateVector3(const float *mat, __m128 v) noexcept {
    __m128 r0 = _mm_loadu_ps(&mat[0]);
    __m128 r1 = _mm_loadu_ps(&mat[4]);
    __m128 r2 = _mm_loadu_ps(&mat[8]);

    __m128 v_x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

    __m128 result = _mm_mul_ps(r2, v_z);
    result = VX_FMADD_PS(r1, v_y, result);
    result = VX_FMADD_PS(r0, v_x, result);
    return result;
}

/**
 * @brief Performs matrix-vector multiplication for 4D vector
 */
VX_SIMD_INLINE __m128 VxSIMDMatrixMultiplyVector4(const float *mat, __m128 v) noexcept {
    __m128 r0 = _mm_loadu_ps(&mat[0]);
    __m128 r1 = _mm_loadu_ps(&mat[4]);
    __m128 r2 = _mm_loadu_ps(&mat[8]);
    __m128 r3 = _mm_loadu_ps(&mat[12]);

    __m128 v_x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v_w = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

    __m128 result = VX_FMADD_PS(r2, v_z, _mm_mul_ps(r3, v_w));
    result = VX_FMADD_PS(r1, v_y, result);
    result = VX_FMADD_PS(r0, v_x, result);
    return result;
}

//-------------------------------------------------------------------------------------
// Vector Arithmetic Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE __m128 VxSIMDAdd3(__m128 a, __m128 b) noexcept {
    return _mm_add_ps(a, b);
}

VX_SIMD_INLINE __m128 VxSIMDSubtract3(__m128 a, __m128 b) noexcept {
    return _mm_sub_ps(a, b);
}

VX_SIMD_INLINE __m128 VxSIMDScale(__m128 v, float scalar) noexcept {
    return _mm_mul_ps(v, _mm_set1_ps(scalar));
}

VX_SIMD_INLINE float VxSIMDLength3(__m128 v) noexcept {
    __m128 dot = VxSIMDDotProduct3(v, v);
    __m128 len = _mm_sqrt_ss(dot);
    float result;
    _mm_store_ss(&result, len);
    return result;
}

VX_SIMD_INLINE float VxSIMDLengthSquared3(__m128 v) noexcept {
    __m128 dot = VxSIMDDotProduct3(v, v);
    float result;
    _mm_store_ss(&result, dot);
    return result;
}

VX_SIMD_INLINE __m128 VxSIMDLerp(__m128 a, __m128 b, float t) noexcept {
    __m128 tVec = _mm_set1_ps(t);
    __m128 diff = _mm_sub_ps(b, a);
    return VX_FMADD_PS(diff, tVec, a);
}

VX_SIMD_INLINE __m128 VxSIMDMin(__m128 a, __m128 b) noexcept {
    return _mm_min_ps(a, b);
}

VX_SIMD_INLINE __m128 VxSIMDMax(__m128 a, __m128 b) noexcept {
    return _mm_max_ps(a, b);
}

VX_SIMD_INLINE __m128 VxSIMDReflect3(__m128 incident, __m128 normal) noexcept {
    __m128 dot = VxSIMDDotProduct3(incident, normal);
    __m128 two_dot = _mm_add_ps(dot, dot);
    return VX_FNMADD_PS(two_dot, normal, incident);
}

//-------------------------------------------------------------------------------------
// Quaternion Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE __m128 VxSIMDQuaternionNormalize(__m128 q) noexcept {
    __m128 magSq = VxSIMDDotProduct4(q, q);
    __m128 isZero = _mm_cmpeq_ps(magSq, _mm_setzero_ps());
    int zeroMask = _mm_movemask_ps(isZero);
    if (zeroMask & 1) {
        return VX_SIMD_QUAT_IDENTITY;
    }
    __m128 invNorm = VxSIMDReciprocalSqrtAccurate(magSq);
    return _mm_mul_ps(q, invNorm);
}

VX_SIMD_INLINE __m128 VxSIMDQuaternionMultiply(__m128 qa, __m128 qb) noexcept {
    __m128 aw = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 t0 = _mm_mul_ps(aw, qb);

    __m128 ax = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 b_perm1 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(0, 1, 2, 3));
    __m128 t1 = _mm_mul_ps(ax, b_perm1);
    t1 = _mm_mul_ps(t1, VX_SIMD_QUAT_SIGN1);

    __m128 ay = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 b_perm2 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 t2 = _mm_mul_ps(ay, b_perm2);
    t2 = _mm_mul_ps(t2, VX_SIMD_QUAT_SIGN2);

    __m128 az = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 b_perm3 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 t3 = _mm_mul_ps(az, b_perm3);
    t3 = _mm_mul_ps(t3, VX_SIMD_QUAT_SIGN3);

    __m128 result = _mm_add_ps(t0, t1);
    result = _mm_add_ps(result, t2);
    result = _mm_add_ps(result, t3);

    return result;
}

VX_SIMD_INLINE __m128 VxSIMDQuaternionConjugate(__m128 q) noexcept {
    static const __m128 conjugateMask = _mm_setr_ps(-1.0f, -1.0f, -1.0f, 1.0f);
    return _mm_mul_ps(q, conjugateMask);
}

//-------------------------------------------------------------------------------------
// Matrix Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDMatrixMultiply(float *result, const float *a, const float *b) noexcept {
    const __m128 a0 = _mm_loadu_ps(a + 0);
    const __m128 a1 = _mm_loadu_ps(a + 4);
    const __m128 a2 = _mm_loadu_ps(a + 8);
    const __m128 a3 = _mm_loadu_ps(a + 12);

    for (int i = 0; i < 4; ++i) {
        const __m128 bRow = _mm_loadu_ps(b + i * 4);
        const __m128 b_x = _mm_shuffle_ps(bRow, bRow, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 b_y = _mm_shuffle_ps(bRow, bRow, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 b_z = _mm_shuffle_ps(bRow, bRow, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 b_w = _mm_shuffle_ps(bRow, bRow, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 res = VX_FMADD_PS(a2, b_z, _mm_mul_ps(a3, b_w));
        res = VX_FMADD_PS(a1, b_y, res);
        res = VX_FMADD_PS(a0, b_x, res);
        _mm_storeu_ps(result + i * 4, res);
    }
}

VX_SIMD_INLINE void VxSIMDMatrixTranspose(float *result, const float *src) noexcept {
    __m128 r0 = _mm_loadu_ps(src + 0);
    __m128 r1 = _mm_loadu_ps(src + 4);
    __m128 r2 = _mm_loadu_ps(src + 8);
    __m128 r3 = _mm_loadu_ps(src + 12);

    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

    _mm_storeu_ps(result + 0, r0);
    _mm_storeu_ps(result + 4, r1);
    _mm_storeu_ps(result + 8, r2);
    _mm_storeu_ps(result + 12, r3);
}

//-------------------------------------------------------------------------------------
// High-Level Vector Operations (working with VxVector types)
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDNormalizeVector(VxVector *v) noexcept {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 dot = VxSIMDDotProduct3(vec, vec);
    __m128 mask = _mm_cmpgt_ps(dot, VX_SIMD_EPSILON);
    __m128 safeDot = _mm_max_ps(dot, VX_SIMD_EPSILON);
    __m128 invLen = VxSIMDReciprocalSqrtAccurate(safeDot);
    __m128 normalized = _mm_mul_ps(vec, invLen);
    __m128 keepOriginal = _mm_andnot_ps(mask, vec);
    __m128 useNormalized = _mm_and_ps(mask, normalized);
    vec = _mm_or_ps(keepOriginal, useNormalized);
    VxSIMDStoreFloat3(&v->x, vec);
}

VX_SIMD_INLINE void VxSIMDAddVector(VxVector *result, const VxVector *a, const VxVector *b) noexcept {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = _mm_add_ps(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

VX_SIMD_INLINE void VxSIMDSubtractVector(VxVector *result, const VxVector *a, const VxVector *b) noexcept {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = _mm_sub_ps(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

VX_SIMD_INLINE void VxSIMDScaleVector(VxVector *result, const VxVector *v, float scalar) noexcept {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 scaleVec = _mm_set1_ps(scalar);
    __m128 resultVec = _mm_mul_ps(vec, scaleVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

VX_SIMD_INLINE float VxSIMDDotVector(const VxVector *a, const VxVector *b) noexcept {
    const __m128 aVec = VxSIMDLoadFloat3(&a->x);
    const __m128 bVec = VxSIMDLoadFloat3(&b->x);
    const __m128 mul = _mm_mul_ps(aVec, bVec);

    __m128 y = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 sum = _mm_add_ss(mul, y);
    __m128 z = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 2, 2, 2));
    sum = _mm_add_ss(sum, z);

    float result;
    _mm_store_ss(&result, sum);
    return result;
}

VX_SIMD_INLINE void VxSIMDCrossVector(VxVector *result, const VxVector *a, const VxVector *b) noexcept {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = VxSIMDCrossProduct3(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

VX_SIMD_INLINE float VxSIMDLengthVector(const VxVector *v) noexcept {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 sq = _mm_mul_ps(vec, vec);

    __m128 y = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 sum = _mm_add_ss(sq, y);
    __m128 z = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 2, 2, 2));
    sum = _mm_add_ss(sum, z);

    float lengthSq;
    _mm_store_ss(&lengthSq, sum);
    return sqrtf(lengthSq);
}

VX_SIMD_INLINE float VxSIMDLengthSquaredVector(const VxVector *v) noexcept {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 sq = _mm_mul_ps(vec, vec);

    __m128 y = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 sum = _mm_add_ss(sq, y);
    __m128 z = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 2, 2, 2));
    sum = _mm_add_ss(sum, z);

    float lengthSq;
    _mm_store_ss(&lengthSq, sum);
    return lengthSq;
}

VX_SIMD_INLINE float VxSIMDDistanceVector(const VxVector *a, const VxVector *b) noexcept {
    VxVector diff;
    VxSIMDSubtractVector(&diff, a, b);
    return VxSIMDLengthVector(&diff);
}

VX_SIMD_INLINE void VxSIMDLerpVector(VxVector *result, const VxVector *a, const VxVector *b, float t) noexcept {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 tVec = _mm_set1_ps(t);
    __m128 diff = _mm_sub_ps(bVec, aVec);
    __m128 resultVec = VX_FMADD_PS(diff, tVec, aVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

VX_SIMD_INLINE void VxSIMDReflectVector(VxVector *result, const VxVector *incident, const VxVector *normal) noexcept {
    __m128 iVec = VxSIMDLoadFloat3(&incident->x);
    __m128 nVec = VxSIMDLoadFloat3(&normal->x);
    __m128 dotVec = VxSIMDDotProduct3(iVec, nVec);
    __m128 twoVec = VX_SIMD_TWO;
    __m128 factor = _mm_mul_ps(twoVec, dotVec);
    __m128 reflection = VX_FNMADD_PS(factor, nVec, iVec);
    VxSIMDStoreFloat3(&result->x, reflection);
}

VX_SIMD_INLINE void VxSIMDMinimizeVector(VxVector *result, const VxVector *a, const VxVector *b) noexcept {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = _mm_min_ps(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

VX_SIMD_INLINE void VxSIMDMaximizeVector(VxVector *result, const VxVector *a, const VxVector *b) noexcept {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = _mm_max_ps(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

VX_SIMD_INLINE void VxSIMDRotateVector(VxVector *result, const VxMatrix *mat, const VxVector *v) noexcept {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 res = VxSIMDMatrixRotateVector3((const float *) &(*mat)[0][0], vec);
    VxSIMDStoreFloat3(&result->x, res);
}

//-------------------------------------------------------------------------------------
// High-Level Matrix Operations (working with VxMatrix types)
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDMultiplyMatrixM(VxMatrix *result, const VxMatrix *a, const VxMatrix *b) noexcept {
    const float *ap = static_cast<const float *>(static_cast<const void *>(*a));
    const float *bp = static_cast<const float *>(static_cast<const void *>(*b));

    const __m128 a0 = _mm_loadu_ps(ap + 0);
    const __m128 a1 = _mm_loadu_ps(ap + 4);
    const __m128 a2 = _mm_loadu_ps(ap + 8);
    const __m128 a3 = _mm_loadu_ps(ap + 12);

    alignas(16) float out[16];

    for (int i = 0; i < 4; ++i) {
        const __m128 bRow = _mm_loadu_ps(bp + i * 4);
        const __m128 b_x = _mm_shuffle_ps(bRow, bRow, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 b_y = _mm_shuffle_ps(bRow, bRow, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 b_z = _mm_shuffle_ps(bRow, bRow, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 b_w = _mm_shuffle_ps(bRow, bRow, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 res = VX_FMADD_PS(a2, b_z, _mm_mul_ps(a3, b_w));
        res = VX_FMADD_PS(a1, b_y, res);
        res = VX_FMADD_PS(a0, b_x, res);
        _mm_storeu_ps(out + i * 4, res);
    }

    // Enforce 3D transformation constraints
    out[0 * 4 + 3] = 0.0f;
    out[1 * 4 + 3] = 0.0f;
    out[2 * 4 + 3] = 0.0f;
    out[3 * 4 + 3] = 1.0f;

    memcpy(result, out, sizeof(out));
}

VX_SIMD_INLINE void VxSIMDTransposeMatrixM(VxMatrix *result, const VxMatrix *a) noexcept {
    __m128 r0 = _mm_loadu_ps((const float *) &(*a)[0][0]);
    __m128 r1 = _mm_loadu_ps((const float *) &(*a)[1][0]);
    __m128 r2 = _mm_loadu_ps((const float *) &(*a)[2][0]);
    __m128 r3 = _mm_loadu_ps((const float *) &(*a)[3][0]);

    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

    VxMatrix temp;
    _mm_storeu_ps((float *) &temp[0][0], r0);
    _mm_storeu_ps((float *) &temp[1][0], r1);
    _mm_storeu_ps((float *) &temp[2][0], r2);
    _mm_storeu_ps((float *) &temp[3][0], r3);

    *result = temp;
}

VX_SIMD_INLINE void VxSIMDMultiplyMatrixVector(VxVector *result, const VxMatrix *mat, const VxVector *v) noexcept {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 res = VxSIMDMatrixMultiplyVector3((const float *) &(*mat)[0][0], vec);
    VxSIMDStoreFloat3(&result->x, res);
}

VX_SIMD_INLINE void VxSIMDMultiplyMatrixVector4(VxVector4 *result, const VxMatrix *mat, const VxVector4 *v) noexcept {
    __m128 vec = VxSIMDLoadFloat4((const float *) v);
    __m128 m0 = _mm_loadu_ps((const float *) &(*mat)[0][0]);
    __m128 m1 = _mm_loadu_ps((const float *) &(*mat)[1][0]);
    __m128 m2 = _mm_loadu_ps((const float *) &(*mat)[2][0]);
    __m128 m3 = _mm_loadu_ps((const float *) &(*mat)[3][0]);

    __m128 v_x = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v_w = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(3, 3, 3, 3));

    __m128 res = VX_FMADD_PS(m2, v_z, _mm_mul_ps(m3, v_w));
    res = VX_FMADD_PS(m1, v_y, res);
    res = VX_FMADD_PS(m0, v_x, res);
    VxSIMDStoreFloat4((float *) result, res);
}

VX_SIMD_INLINE void VxSIMDMatrixIdentity(VxMatrix *mat) noexcept {
    _mm_storeu_ps(&(*mat)[0][0], VX_SIMD_IDENTITY_R0);
    _mm_storeu_ps(&(*mat)[1][0], VX_SIMD_IDENTITY_R1);
    _mm_storeu_ps(&(*mat)[2][0], VX_SIMD_IDENTITY_R2);
    _mm_storeu_ps(&(*mat)[3][0], VX_SIMD_IDENTITY_R3);
}

//-------------------------------------------------------------------------------------
// High-Level Quaternion Operations (working with VxQuaternion types)
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDNormalizeQuaternion(VxQuaternion *q) noexcept {
    __m128 quat = _mm_loadu_ps(&q->x);
    __m128 magSq = VxSIMDDotProduct4(quat, quat);

    __m128 isZero = _mm_cmpeq_ps(magSq, VX_SIMD_ZERO);
    int zeroMask = _mm_movemask_ps(isZero);

    if (zeroMask & 1) {
        _mm_storeu_ps(&q->x, VX_SIMD_QUAT_IDENTITY);
        return;
    }

    __m128 invNorm = VxSIMDReciprocalSqrtAccurate(magSq);
    __m128 result = _mm_mul_ps(quat, invNorm);
    _mm_storeu_ps(&q->x, result);
}

VX_SIMD_INLINE void VxSIMDMultiplyQuaternion(VxQuaternion *result, const VxQuaternion *a, const VxQuaternion *b) noexcept {
    __m128 qa = _mm_loadu_ps(&a->x);
    __m128 qb = _mm_loadu_ps(&b->x);
    __m128 r = VxSIMDQuaternionMultiply(qa, qb);
    _mm_storeu_ps(&result->x, r);
}

VX_SIMD_INLINE void VxSIMDSlerpQuaternion(VxQuaternion *result, float t, const VxQuaternion *a, const VxQuaternion *b) noexcept {
    __m128 qa = _mm_loadu_ps(&a->x);
    __m128 qb = _mm_loadu_ps(&b->x);

    __m128 cosOmegaVec = VxSIMDDotProduct4(qa, qb);
    float cosOmega;
    _mm_store_ss(&cosOmega, cosOmegaVec);

    float k0, k1;
    float sign = 1.0f;

    if (cosOmega < 0.0f) {
        cosOmega = -cosOmega;
        sign = -1.0f;
    }

    float oneMinusCos = 1.0f - cosOmega;
    if (oneMinusCos < 0.01f) {
        k0 = 1.0f - t;
        k1 = t * sign;
    } else {
        float omega = acosf(cosOmega);
        float invSinOmega = 1.0f / sinf(omega);
        k0 = sinf((1.0f - t) * omega) * invSinOmega;
        k1 = sinf(t * omega) * invSinOmega * sign;
    }

    __m128 k0Vec = _mm_set1_ps(k0);
    __m128 k1Vec = _mm_set1_ps(k1);
    __m128 r = VX_FMADD_PS(qb, k1Vec, _mm_mul_ps(qa, k0Vec));
    _mm_storeu_ps(&result->x, r);
}

//-------------------------------------------------------------------------------------
// Vector4 Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDAddVector4(VxVector4 *result, const VxVector4 *a, const VxVector4 *b) noexcept {
    __m128 aVec = VxSIMDLoadFloat4((const float *) a);
    __m128 bVec = VxSIMDLoadFloat4((const float *) b);
    __m128 resultVec = _mm_add_ps(aVec, bVec);
    VxSIMDStoreFloat4((float *) result, resultVec);
}

VX_SIMD_INLINE void VxSIMDSubtractVector4(VxVector4 *result, const VxVector4 *a, const VxVector4 *b) noexcept {
    __m128 aVec = VxSIMDLoadFloat4((const float *) a);
    __m128 bVec = VxSIMDLoadFloat4((const float *) b);
    __m128 resultVec = _mm_sub_ps(aVec, bVec);
    VxSIMDStoreFloat4((float *) result, resultVec);
}

VX_SIMD_INLINE void VxSIMDScaleVector4(VxVector4 *result, const VxVector4 *v, float scalar) noexcept {
    __m128 vec = VxSIMDLoadFloat4((const float *) v);
    __m128 scaleVec = _mm_set1_ps(scalar);
    __m128 resultVec = _mm_mul_ps(vec, scaleVec);
    VxSIMDStoreFloat4((float *) result, resultVec);
}

VX_SIMD_INLINE float VxSIMDDotVector4(const VxVector4 *a, const VxVector4 *b) noexcept {
    __m128 aVec = VxSIMDLoadFloat4((const float *) a);
    __m128 bVec = VxSIMDLoadFloat4((const float *) b);
    __m128 dotVec = VxSIMDDotProduct4(aVec, bVec);
    float result;
    _mm_store_ss(&result, dotVec);
    return result;
}

VX_SIMD_INLINE void VxSIMDLerpVector4(VxVector4 *result, const VxVector4 *a, const VxVector4 *b, float t) noexcept {
    __m128 aVec = VxSIMDLoadFloat4((const float *) a);
    __m128 bVec = VxSIMDLoadFloat4((const float *) b);
    __m128 tVec = _mm_set1_ps(t);
    __m128 diff = _mm_sub_ps(bVec, aVec);
    __m128 resultVec = VX_FMADD_PS(diff, tVec, aVec);
    VxSIMDStoreFloat4((float *) result, resultVec);
}

//-------------------------------------------------------------------------------------
// Array Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDInterpolateFloatArray(float *result, const float *a, const float *b, float factor, int count) noexcept {
    const __m128 factorVec = _mm_set1_ps(factor);

    const int simdCount = count & ~3;
    for (int i = 0; i < simdCount; i += 4) {
        const __m128 aVec = _mm_loadu_ps(a + i);
        const __m128 bVec = _mm_loadu_ps(b + i);
        const __m128 diff = _mm_sub_ps(bVec, aVec);
        const __m128 resultVec = VX_FMADD_PS(diff, factorVec, aVec);
        _mm_storeu_ps(result + i, resultVec);
    }

    for (int i = simdCount; i < count; ++i) {
        result[i] = a[i] + (b[i] - a[i]) * factor;
    }
}

VX_SIMD_INLINE void VxSIMDInterpolateVectorArray(void *result, const void *a, const void *b, float factor, int count, XULONG strideResult, XULONG strideInput) noexcept {
    const char *srcA = static_cast<const char *>(a);
    const char *srcB = static_cast<const char *>(b);
    char *dst = static_cast<char *>(result);

    const __m128 factorVec = _mm_set1_ps(factor);

    for (int i = 0; i < count; ++i) {
        const VxVector *vecA = reinterpret_cast<const VxVector *>(srcA + i * strideInput);
        const VxVector *vecB = reinterpret_cast<const VxVector *>(srcB + i * strideInput);
        VxVector *vecResult = reinterpret_cast<VxVector *>(dst + i * strideResult);

        __m128 aVec = VxSIMDLoadFloat3(&vecA->x);
        __m128 bVec = VxSIMDLoadFloat3(&vecB->x);
        __m128 diff = _mm_sub_ps(bVec, aVec);
        __m128 resultVec = VX_FMADD_PS(diff, factorVec, aVec);
        VxSIMDStoreFloat3(&vecResult->x, resultVec);
    }
}

//-------------------------------------------------------------------------------------
// Ray Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDRayTransform(VxRay *dest, const VxRay *ray, const VxMatrix *mat) noexcept {
    VxSIMDMultiplyMatrixVector(&dest->m_Origin, mat, &ray->m_Origin);
    VxSIMDRotateVector(&dest->m_Direction, mat, &ray->m_Direction);
}

//-------------------------------------------------------------------------------------
// Plane Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDPlaneCreateFromPoint(VxPlane *plane, const VxVector *normal, const VxVector *point) noexcept {
    plane->m_Normal = *normal;

    __m128 n = VxSIMDLoadFloat3(&plane->m_Normal.x);
    const __m128 mul = _mm_mul_ps(n, n);
    __m128 sum = _mm_add_ss(mul, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 2, 2, 2)));

    float magSqScalar;
    _mm_store_ss(&magSqScalar, sum);
    if (magSqScalar > EPSILON) {
        const __m128 mag = _mm_sqrt_ss(sum);
        const __m128 invMag = _mm_div_ss(_mm_set_ss(1.0f), mag);
        const __m128 invMag4 = _mm_shuffle_ps(invMag, invMag, _MM_SHUFFLE(0, 0, 0, 0));
        n = _mm_mul_ps(n, invMag4);
        VxSIMDStoreFloat3(&plane->m_Normal.x, n);
    }

    const __m128 p = VxSIMDLoadFloat3(&point->x);
    const __m128 dp = _mm_mul_ps(VxSIMDLoadFloat3(&plane->m_Normal.x), p);
    __m128 dsum = _mm_add_ss(dp, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(1, 1, 1, 1)));
    dsum = _mm_add_ss(dsum, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(2, 2, 2, 2)));
    float d;
    _mm_store_ss(&d, dsum);
    plane->m_D = -d;
}

VX_SIMD_INLINE void VxSIMDPlaneCreateFromTriangle(VxPlane *plane, const VxVector *a, const VxVector *b, const VxVector *c) noexcept {
    VxVector edge1, edge2, n;
    VxSIMDSubtractVector(&edge1, b, a);
    VxSIMDSubtractVector(&edge2, c, a);
    VxSIMDCrossVector(&n, &edge1, &edge2);

    __m128 nn = VxSIMDLoadFloat3(&n.x);
    const __m128 mul = _mm_mul_ps(nn, nn);
    __m128 sum = _mm_add_ss(mul, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 2, 2, 2)));
    float magSqScalar;
    _mm_store_ss(&magSqScalar, sum);

    if (magSqScalar > EPSILON) {
        const __m128 mag = _mm_sqrt_ss(sum);
        const __m128 invMag = _mm_div_ss(_mm_set_ss(1.0f), mag);
        const __m128 invMag4 = _mm_shuffle_ps(invMag, invMag, _MM_SHUFFLE(0, 0, 0, 0));
        nn = _mm_mul_ps(nn, invMag4);
        VxSIMDStoreFloat3(&n.x, nn);
    } else {
        n = VxVector(0.0f, 0.0f, 1.0f);
    }

    VxSIMDPlaneCreateFromPoint(plane, &n, a);
}

//-------------------------------------------------------------------------------------
// Rect Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDRectTransform(VxRect *rect, const VxRect *destScreen, const VxRect *srcScreen) noexcept {
    const __m128 r = _mm_loadu_ps(&rect->left);
    const __m128 s = _mm_loadu_ps(&srcScreen->left);
    const __m128 d = _mm_loadu_ps(&destScreen->left);

    const __m128 s_right = _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 s_left = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 s_bottom = _mm_shuffle_ps(s, s, _MM_SHUFFLE(3, 3, 3, 3));
    const __m128 s_top = _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 w = _mm_sub_ps(s_right, s_left);
    const __m128 h = _mm_sub_ps(s_bottom, s_top);
    const __m128 srcSize = _mm_unpacklo_ps(w, h);
    const __m128 srcInvSize = _mm_div_ps(_mm_set1_ps(1.0f), srcSize);

    const __m128 normalized = _mm_mul_ps(_mm_sub_ps(r, s), srcInvSize);

    const __m128 d_right = _mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 d_left = _mm_shuffle_ps(d, d, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 d_bottom = _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 3, 3, 3));
    const __m128 d_top = _mm_shuffle_ps(d, d, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 dw = _mm_sub_ps(d_right, d_left);
    const __m128 dh = _mm_sub_ps(d_bottom, d_top);
    const __m128 destSize = _mm_unpacklo_ps(dw, dh);

    const __m128 result = _mm_add_ps(_mm_mul_ps(normalized, destSize), d);
    _mm_storeu_ps(&rect->left, result);
}

VX_SIMD_INLINE void VxSIMDRectTransformBySize(VxRect *rect, const Vx2DVector *destScreenSize, const Vx2DVector *srcScreenSize) noexcept {
    __m128 rectVec = _mm_loadu_ps(&rect->left);

    __m128 srcInvSize = _mm_setr_ps(
        1.0f / srcScreenSize->x,
        1.0f / srcScreenSize->y,
        1.0f / srcScreenSize->x,
        1.0f / srcScreenSize->y
    );

    __m128 destSize = _mm_setr_ps(
        destScreenSize->x,
        destScreenSize->y,
        destScreenSize->x,
        destScreenSize->y
    );

    __m128 result = _mm_mul_ps(_mm_mul_ps(rectVec, srcInvSize), destSize);
    _mm_storeu_ps(&rect->left, result);
}

VX_SIMD_INLINE void VxSIMDRectTransformToHomogeneous(VxRect *rect, const VxRect *screen) noexcept {
    float width = rect->right - rect->left;
    float height = rect->bottom - rect->top;

    __m128 rectVec = _mm_loadu_ps(&rect->left);
    __m128 screenVec = _mm_loadu_ps(&screen->left);

    __m128 screenRightBottom = _mm_shuffle_ps(screenVec, screenVec, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 screenSize = _mm_sub_ps(screenRightBottom, screenVec);
    __m128 screenInvSize = _mm_div_ps(_mm_set1_ps(1.0f), screenSize);

    __m128 minTransformed = _mm_mul_ps(_mm_sub_ps(rectVec, screenVec), screenInvSize);

    float leftNorm, topNorm;
    _mm_store_ss(&leftNorm, minTransformed);
    _mm_store_ss(&topNorm, _mm_shuffle_ps(minTransformed, minTransformed, _MM_SHUFFLE(1, 1, 1, 1)));

    float screenInvWidth, screenInvHeight;
    _mm_store_ss(&screenInvWidth, screenInvSize);
    _mm_store_ss(&screenInvHeight, _mm_shuffle_ps(screenInvSize, screenInvSize, _MM_SHUFFLE(1, 1, 1, 1)));

    rect->left = leftNorm;
    rect->top = topNorm;
    rect->right = leftNorm + width * screenInvWidth;
    rect->bottom = topNorm + height * screenInvHeight;
}

VX_SIMD_INLINE void VxSIMDRectTransformFromHomogeneous(VxRect *rect, const VxRect *screen) noexcept {
    float width = rect->right - rect->left;
    float height = rect->bottom - rect->top;

    __m128 rectVec = _mm_loadu_ps(&rect->left);
    __m128 screenVec = _mm_loadu_ps(&screen->left);

    __m128 screenRightBottom = _mm_shuffle_ps(screenVec, screenVec, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 screenSize = _mm_sub_ps(screenRightBottom, screenVec);

    __m128 minTransformed = _mm_add_ps(screenVec, _mm_mul_ps(rectVec, screenSize));

    float leftTrans, topTrans;
    _mm_store_ss(&leftTrans, minTransformed);
    _mm_store_ss(&topTrans, _mm_shuffle_ps(minTransformed, minTransformed, _MM_SHUFFLE(1, 1, 1, 1)));

    float screenWidth, screenHeight;
    _mm_store_ss(&screenWidth, screenSize);
    _mm_store_ss(&screenHeight, _mm_shuffle_ps(screenSize, screenSize, _MM_SHUFFLE(1, 1, 1, 1)));

    rect->left = leftTrans;
    rect->top = topTrans;
    rect->right = leftTrans + width * screenWidth;
    rect->bottom = topTrans + height * screenHeight;
}

//-------------------------------------------------------------------------------------
// BBox Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE int VxSIMDBboxClassify(const VxBbox *self, const VxBbox *other, const VxVector *point) noexcept {
    __m128 selfMin = VxSIMDLoadFloat3(&self->Min.x);
    __m128 selfMax = VxSIMDLoadFloat3(&self->Max.x);
    __m128 ptVec = VxSIMDLoadFloat3(&point->x);
    __m128 otherMin = VxSIMDLoadFloat3(&other->Min.x);
    __m128 otherMax = VxSIMDLoadFloat3(&other->Max.x);

    XULONG ptFlags = 0;
    __m128 ptLessThanMin = _mm_cmplt_ps(ptVec, selfMin);
    __m128 ptGreaterThanMax = _mm_cmpgt_ps(ptVec, selfMax);
    int ptLessFlags = _mm_movemask_ps(ptLessThanMin);
    int ptGreaterFlags = _mm_movemask_ps(ptGreaterThanMax);

    if (ptLessFlags & 1) ptFlags |= VXCLIP_LEFT;
    if (ptGreaterFlags & 1) ptFlags |= VXCLIP_RIGHT;
    if (ptLessFlags & 2) ptFlags |= VXCLIP_BOTTOM;
    if (ptGreaterFlags & 2) ptFlags |= VXCLIP_TOP;
    if (ptLessFlags & 4) ptFlags |= VXCLIP_BACK;
    if (ptGreaterFlags & 4) ptFlags |= VXCLIP_FRONT;

    XULONG box2Flags = 0;
    __m128 otherMaxLessThanMin = _mm_cmplt_ps(otherMax, selfMin);
    __m128 otherMinGreaterThanMax = _mm_cmpgt_ps(otherMin, selfMax);
    int box2LessFlags = _mm_movemask_ps(otherMaxLessThanMin);
    int box2GreaterFlags = _mm_movemask_ps(otherMinGreaterThanMax);

    if (box2LessFlags & 4) box2Flags |= VXCLIP_BACK;
    if (box2GreaterFlags & 4) box2Flags |= VXCLIP_FRONT;
    if (box2LessFlags & 1) box2Flags |= VXCLIP_LEFT;
    if (box2GreaterFlags & 1) box2Flags |= VXCLIP_RIGHT;
    if (box2LessFlags & 2) box2Flags |= VXCLIP_BOTTOM;
    if (box2GreaterFlags & 2) box2Flags |= VXCLIP_TOP;

    if (ptFlags) {
        if (!box2Flags) {
            __m128 box2MinGE = _mm_cmpge_ps(otherMin, selfMin);
            __m128 box2MaxLE = _mm_cmple_ps(otherMax, selfMax);
            __m128 box2Inside = _mm_and_ps(box2MinGE, box2MaxLE);

            if ((_mm_movemask_ps(box2Inside) & 0x7) == 0x7) {
                return -1;
            }

            __m128 selfMinGE = _mm_cmpge_ps(selfMin, otherMin);
            __m128 selfMaxLE = _mm_cmple_ps(selfMax, otherMax);
            __m128 selfInside = _mm_and_ps(selfMinGE, selfMaxLE);

            if ((_mm_movemask_ps(selfInside) & 0x7) == 0x7) {
                __m128 ptLessThanOtherMin = _mm_cmplt_ps(ptVec, otherMin);
                __m128 ptGreaterThanOtherMax = _mm_cmpgt_ps(ptVec, otherMax);
                __m128 ptOutside = _mm_or_ps(ptLessThanOtherMin, ptGreaterThanOtherMax);
                if (_mm_movemask_ps(ptOutside) & 0x7) {
                    return 1;
                }
            }
        }
    } else {
        if (box2Flags) return -1;
        __m128 selfMinGE = _mm_cmpge_ps(selfMin, otherMin);
        __m128 selfMaxLE = _mm_cmple_ps(selfMax, otherMax);
        __m128 selfInside = _mm_and_ps(selfMinGE, selfMaxLE);
        if ((_mm_movemask_ps(selfInside) & 0x7) == 0x7) {
            return 1;
        }
    }

    return 0;
}

VX_SIMD_INLINE void VxSIMDBboxClassifyVertices(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, XULONG *flags) noexcept {
    __m128 bboxMin = VxSIMDLoadFloat3(&self->Min.x);
    __m128 bboxMax = VxSIMDLoadFloat3(&self->Max.x);

    for (int i = 0; i < count; ++i) {
        const float *v = reinterpret_cast<const float *>(vertices + i * stride);
        __m128 vertex = VxSIMDLoadFloat3(v);

        __m128 lessThanMin = _mm_cmplt_ps(vertex, bboxMin);
        __m128 greaterThanMax = _mm_cmpgt_ps(vertex, bboxMax);
        int lessFlags = _mm_movemask_ps(lessThanMin);
        int greaterFlags = _mm_movemask_ps(greaterThanMax);

        XULONG flag = 0;
        if (lessFlags & 4) flag |= VXCLIP_BACK;
        else if (greaterFlags & 4) flag |= VXCLIP_FRONT;
        if (lessFlags & 2) flag |= VXCLIP_BOTTOM;
        else if (greaterFlags & 2) flag |= VXCLIP_TOP;
        if (lessFlags & 1) flag |= VXCLIP_LEFT;
        else if (greaterFlags & 1) flag |= VXCLIP_RIGHT;

        flags[i] = flag;
    }
}

VX_SIMD_INLINE void VxSIMDBboxTransformTo(const VxBbox *self, VxVector *points, const VxMatrix *mat) noexcept {
    VxSIMDMultiplyMatrixVector(&points[0], mat, &self->Min);

    const float sizeX = self->Max.x - self->Min.x;
    const float sizeY = self->Max.y - self->Min.y;
    const float sizeZ = self->Max.z - self->Min.z;

    const VxVector xVec(sizeX * (*mat)[0][0], sizeX * (*mat)[0][1], sizeX * (*mat)[0][2]);
    const VxVector yVec(sizeY * (*mat)[1][0], sizeY * (*mat)[1][1], sizeY * (*mat)[1][2]);
    const VxVector zVec(sizeZ * (*mat)[2][0], sizeZ * (*mat)[2][1], sizeZ * (*mat)[2][2]);

    points[1] = points[0] + zVec;
    points[2] = points[0] + yVec;
    points[3] = points[2] + zVec;
    points[4] = points[0] + xVec;
    points[5] = points[4] + zVec;
    points[6] = points[4] + yVec;
    points[7] = points[6] + zVec;
}

VX_SIMD_INLINE void VxSIMDBboxTransformFrom(VxBbox *dest, const VxBbox *src, const VxMatrix *mat) noexcept {
    VxVector center((src->Min.x + src->Max.x) * 0.5f,
                    (src->Min.y + src->Max.y) * 0.5f,
                    (src->Min.z + src->Max.z) * 0.5f);

    VxSIMDMultiplyMatrixVector(&dest->Min, mat, &center);

    const float sizeX = src->Max.x - src->Min.x;
    const float sizeY = src->Max.y - src->Min.y;
    const float sizeZ = src->Max.z - src->Min.z;

    const VxVector xVec(sizeX * (*mat)[0][0], sizeX * (*mat)[0][1], sizeX * (*mat)[0][2]);
    const VxVector yVec(sizeY * (*mat)[1][0], sizeY * (*mat)[1][1], sizeY * (*mat)[1][2]);
    const VxVector zVec(sizeZ * (*mat)[2][0], sizeZ * (*mat)[2][1], sizeZ * (*mat)[2][2]);

    const float halfX = (fabsf(xVec.x) + fabsf(yVec.x) + fabsf(zVec.x)) * 0.5f;
    const float halfY = (fabsf(xVec.y) + fabsf(yVec.y) + fabsf(zVec.y)) * 0.5f;
    const float halfZ = (fabsf(xVec.z) + fabsf(yVec.z) + fabsf(zVec.z)) * 0.5f;

    dest->Max.x = dest->Min.x + halfX;
    dest->Max.y = dest->Min.y + halfY;
    dest->Max.z = dest->Min.z + halfZ;

    dest->Min.x = dest->Min.x - halfX;
    dest->Min.y = dest->Min.y - halfY;
    dest->Min.z = dest->Min.z - halfZ;
}

//-------------------------------------------------------------------------------------
// Frustum Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE void VxSIMDFrustumUpdate(VxFrustum *frustum) noexcept {
    frustum->Update();
}

VX_SIMD_INLINE void VxSIMDFrustumComputeVertices(const VxFrustum *frustum, VxVector *vertices) noexcept {
    const VxVector &dir = frustum->GetDir();
    const VxVector &right = frustum->GetRight();
    const VxVector &up = frustum->GetUp();
    const VxVector &origin = frustum->GetOrigin();
    const float dMin = frustum->GetDMin();
    const float rBound = frustum->GetRBound();
    const float uBound = frustum->GetUBound();
    const float dRatio = frustum->GetDRatio();

    VxVector nearDirVec, rightVec, upVec;
    VxSIMDScaleVector(&nearDirVec, &dir, dMin);
    VxSIMDScaleVector(&rightVec, &right, rBound);
    VxSIMDScaleVector(&upVec, &up, uBound);

    VxVector leftVec, rightVec2, temp1;
    VxSIMDSubtractVector(&leftVec, &nearDirVec, &rightVec);
    VxSIMDAddVector(&rightVec2, &nearDirVec, &rightVec);

    VxSIMDSubtractVector(&vertices[0], &leftVec, &upVec);
    VxSIMDAddVector(&vertices[1], &leftVec, &upVec);
    VxSIMDAddVector(&vertices[2], &rightVec2, &upVec);
    VxSIMDSubtractVector(&vertices[3], &rightVec2, &upVec);

    for (int i = 0; i < 4; i++) {
        VxVector farVec;
        VxSIMDScaleVector(&farVec, &vertices[i], dRatio);
        VxSIMDAddVector(&vertices[i + 4], &origin, &farVec);
        VxSIMDAddVector(&temp1, &vertices[i], &origin);
        vertices[i] = temp1;
    }
}

VX_SIMD_INLINE void VxSIMDFrustumTransform(VxFrustum *frustum, const VxMatrix *invWorldMat) noexcept {
    VxVector &right = frustum->GetRight();
    VxVector &up = frustum->GetUp();
    VxVector &dir = frustum->GetDir();
    VxVector &origin = frustum->GetOrigin();
    float &rBound = frustum->GetRBound();
    float &uBound = frustum->GetUBound();
    float &dMin = frustum->GetDMin();
    float &dMax = frustum->GetDMax();
    const float dRatio = frustum->GetDRatio();

    VxSIMDScaleVector(&right, &right, rBound);
    VxSIMDScaleVector(&up, &up, uBound);
    VxSIMDScaleVector(&dir, &dir, dMin);

    VxVector newOrigin;
    VxSIMDMultiplyMatrixVector(&newOrigin, invWorldMat, &origin);
    origin = newOrigin;

    VxVector resultVectors[3];
    VxSIMDRotateVector(&resultVectors[0], invWorldMat, &right);
    VxSIMDRotateVector(&resultVectors[1], invWorldMat, &up);
    VxSIMDRotateVector(&resultVectors[2], invWorldMat, &dir);

    float newRBound = VxSIMDLengthVector(&resultVectors[0]);
    float newUBound = VxSIMDLengthVector(&resultVectors[1]);
    float newDMin = VxSIMDLengthVector(&resultVectors[2]);

    rBound = newRBound;
    uBound = newUBound;
    dMin = newDMin;
    dMax = newDMin * dRatio;

    VxSIMDScaleVector(&right, &resultVectors[0], 1.0f / newRBound);
    VxSIMDScaleVector(&up, &resultVectors[1], 1.0f / newUBound);
    VxSIMDScaleVector(&dir, &resultVectors[2], 1.0f / newDMin);

    frustum->Update();
}

//-------------------------------------------------------------------------------------
// Box Projection Operations
//-------------------------------------------------------------------------------------

VX_SIMD_INLINE XBOOL VxSIMDTransformBox2D(const VxMatrix *worldProjection, const VxBbox *box, VxRect *extents, const VxRect *screenSize, VXCLIP_FLAGS *orClipFlags, VXCLIP_FLAGS *andClipFlags) noexcept {
    if (!box || !box->IsValid()) {
        if (orClipFlags) *orClipFlags = (VXCLIP_FLAGS) VXCLIP_ALL;
        if (andClipFlags) *andClipFlags = (VXCLIP_FLAGS) VXCLIP_ALL;
        return FALSE;
    }

    __m128 verts[8];
    verts[0] = VxSIMDMatrixMultiplyVector3((const float *) &(*worldProjection)[0][0], VxSIMDLoadFloat3(&box->Min.x));

    const float dx = box->Max.x - box->Min.x;
    const float dy = box->Max.y - box->Min.y;
    const float dz = box->Max.z - box->Min.z;

    const __m128 col0 = _mm_loadu_ps((const float *) &(*worldProjection)[0][0]);
    const __m128 col1 = _mm_loadu_ps((const float *) &(*worldProjection)[1][0]);
    const __m128 col2 = _mm_loadu_ps((const float *) &(*worldProjection)[2][0]);

    const __m128 deltaX = _mm_mul_ps(col0, _mm_set1_ps(dx));
    const __m128 deltaY = _mm_mul_ps(col1, _mm_set1_ps(dy));
    const int vertexCount = (dz == 0.0f) ? 4 : 8;

    if (vertexCount == 4) {
        verts[1] = _mm_add_ps(verts[0], deltaX);
        verts[2] = _mm_add_ps(verts[0], deltaY);
        verts[3] = _mm_add_ps(verts[1], deltaY);
    } else {
        const __m128 deltaZ = _mm_mul_ps(col2, _mm_set1_ps(dz));
        verts[1] = _mm_add_ps(verts[0], deltaZ);
        verts[2] = _mm_add_ps(verts[0], deltaY);
        verts[3] = _mm_add_ps(verts[1], deltaY);
        verts[4] = _mm_add_ps(verts[0], deltaX);
        verts[5] = _mm_add_ps(verts[4], deltaZ);
        verts[6] = _mm_add_ps(verts[4], deltaY);
        verts[7] = _mm_add_ps(verts[5], deltaY);
    }

    XULONG allOr = 0;
    XULONG allAnd = 0xFFFFFFFFu;
    const __m128 zero = _mm_setzero_ps();

    for (int i = 0; i < vertexCount; ++i) {
        __m128 v = verts[i];
        __m128 w = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 negW = _mm_sub_ps(zero, w);

        __m128 x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

        XULONG flags = 0;
        if (_mm_comilt_ss(x, negW)) flags |= VXCLIP_LEFT;
        if (_mm_comigt_ss(x, w)) flags |= VXCLIP_RIGHT;
        if (_mm_comilt_ss(y, negW)) flags |= VXCLIP_BOTTOM;
        if (_mm_comigt_ss(y, w)) flags |= VXCLIP_TOP;
        if (_mm_comilt_ss(z, zero)) flags |= VXCLIP_FRONT;
        if (_mm_comigt_ss(z, w)) flags |= VXCLIP_BACK;

        allOr |= flags;
        allAnd &= flags;
    }

    if (extents && screenSize && (allAnd & VXCLIP_ALL) == 0) {
        __m128 minXY = _mm_set1_ps(1000000.0f);
        __m128 maxXY = _mm_set1_ps(-1000000.0f);

        const float halfWidth = (screenSize->right - screenSize->left) * 0.5f;
        const float halfHeight = (screenSize->bottom - screenSize->top) * 0.5f;
        const float centerX = halfWidth + screenSize->left;
        const float centerY = halfHeight + screenSize->top;

        __m128 viewScale = _mm_setr_ps(halfWidth, -halfHeight, 0.0f, 0.0f);
        __m128 viewOffset = _mm_setr_ps(centerX, centerY, 0.0f, 0.0f);

        for (int i = 0; i < vertexCount; ++i) {
            __m128 v = verts[i];
            __m128 w = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

            float wf;
            _mm_store_ss(&wf, w);
            if (wf <= 0.0f) continue;

            __m128 rcp = _mm_rcp_ss(w);
            __m128 two = _mm_set_ss(2.0f);
            rcp = _mm_mul_ss(rcp, _mm_sub_ss(two, _mm_mul_ss(w, rcp)));
            __m128 invW = _mm_shuffle_ps(rcp, rcp, _MM_SHUFFLE(0, 0, 0, 0));

            __m128 projected = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(v, invW), viewScale), viewOffset);
            minXY = _mm_min_ps(minXY, projected);
            maxXY = _mm_max_ps(maxXY, projected);
        }

        alignas(16) float minf[4], maxf[4];
        _mm_store_ps(minf, minXY);
        _mm_store_ps(maxf, maxXY);

        extents->left = minf[0];
        extents->bottom = maxf[1];
        extents->right = maxf[0];
        extents->top = minf[1];
    }

    if ((allOr & VXCLIP_FRONT) != 0 && extents && screenSize) {
        if ((allOr & VXCLIP_LEFT) != 0) extents->left = screenSize->left;
        if ((allOr & VXCLIP_RIGHT) != 0) extents->right = screenSize->right;
        if ((allOr & VXCLIP_TOP) != 0) extents->top = screenSize->top;
        if ((allOr & VXCLIP_BOTTOM) != 0) extents->bottom = screenSize->bottom;
    }

    if (orClipFlags) *orClipFlags = (VXCLIP_FLAGS) allOr;
    if (andClipFlags) *andClipFlags = (VXCLIP_FLAGS) allAnd;
    return (allAnd & VXCLIP_ALL) == 0;
}

VX_SIMD_INLINE void VxSIMDProjectBoxZExtents(const VxMatrix *worldProjection, const VxBbox *box, float *zhMin, float *zhMax) noexcept {
    if (!zhMin || !zhMax) return;
    *zhMin = 1.0e10f;
    *zhMax = -1.0e10f;

    if (!box || !box->IsValid()) {
        return;
    }

    __m128 corners[8];
    corners[0] = VxSIMDMatrixMultiplyVector3((const float *) &(*worldProjection)[0][0], VxSIMDLoadFloat3(&box->Min.x));

    const float dx = box->Max.x - box->Min.x;
    const float dy = box->Max.y - box->Min.y;
    const float dz = box->Max.z - box->Min.z;

    const __m128 col0 = _mm_loadu_ps((const float *) &(*worldProjection)[0][0]);
    const __m128 col1 = _mm_loadu_ps((const float *) &(*worldProjection)[1][0]);
    const __m128 col2 = _mm_loadu_ps((const float *) &(*worldProjection)[2][0]);
    const __m128 deltaX = _mm_mul_ps(col0, _mm_set1_ps(dx));
    const __m128 deltaY = _mm_mul_ps(col1, _mm_set1_ps(dy));
    const __m128 deltaZ = _mm_mul_ps(col2, _mm_set1_ps(dz));

    int vertexCount;
    if (fabsf(dz) < EPSILON) {
        vertexCount = 4;
        corners[1] = _mm_add_ps(corners[0], deltaX);
        corners[2] = _mm_add_ps(corners[0], deltaY);
        corners[3] = _mm_add_ps(corners[1], deltaY);
    } else {
        vertexCount = 8;
        corners[1] = _mm_add_ps(corners[0], deltaZ);
        corners[2] = _mm_add_ps(corners[0], deltaY);
        corners[3] = _mm_add_ps(corners[1], deltaY);
        corners[4] = _mm_add_ps(corners[0], deltaX);
        corners[5] = _mm_add_ps(corners[4], deltaZ);
        corners[6] = _mm_add_ps(corners[4], deltaY);
        corners[7] = _mm_add_ps(corners[5], deltaY);
    }

    for (int i = 0; i < vertexCount; ++i) {
        alignas(16) float v[4];
        _mm_storeu_ps(v, corners[i]);
        const float z = v[2];
        const float w = v[3];

        if (w <= EPSILON)
            continue;

        const float projZ = z / w;
        if (projZ < *zhMin) *zhMin = projZ;
        if (projZ > *zhMax) *zhMax = projZ;
    }
}

#endif // VX_SIMD_SSE
