#include "VxSIMD.h"

#include "VxVector.h"
#include "VxMatrix.h"
#include "VxQuaternion.h"
#include "VxRay.h"
#include "VxPlane.h"
#include "VxRect.h"
#include "VxFrustum.h"
#include "Vx2dVector.h"
#include "VxEigenMatrix.h"

#if defined(VX_SIMD_X86)

/**
 * @brief SSE Baseline Variant
 *
 * @remarks
 * This file contains SSE-optimized implementations that work on all
 * x86/x64 CPUs with SSE support (essentially all modern x86 CPUs).
 * These serve as the baseline fallback when AVX+FMA is not available.
 */

void VxSIMDNormalizeVector_SSE(VxVector *v) {
    // Optimized SSE normalization using rsqrt with Newton-Raphson refinement
    // Match VxVector::Normalize() semantics (sqrt-based, epsilon guard)
    __m128 vec = VxSIMDLoadFloat3(&v->x);

    // dot = x*x + y*y + z*z using horizontal adds
    const __m128 mul = _mm_mul_ps(vec, vec);
    __m128 sum = _mm_add_ss(mul, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 2, 2, 2)));

    float magSqScalar;
    _mm_store_ss(&magSqScalar, sum);
    if (magSqScalar > EPSILON) {
        // Use rsqrt with Newton-Raphson for fast, accurate 1/sqrt
        // rsqrt gives ~12 bits precision; one NR iteration gives ~24 bits
        __m128 rsqrt = _mm_rsqrt_ss(sum);
        // Newton-Raphson: rsqrt' = rsqrt * (1.5 - 0.5 * magSq * rsqrt * rsqrt)
        __m128 half = _mm_set_ss(0.5f);
        __m128 three_half = _mm_set_ss(1.5f);
        __m128 rsqrt_sq = _mm_mul_ss(rsqrt, rsqrt);
        __m128 half_sum = _mm_mul_ss(half, sum);
        __m128 correction = _mm_sub_ss(three_half, _mm_mul_ss(half_sum, rsqrt_sq));
        rsqrt = _mm_mul_ss(rsqrt, correction);

        // Broadcast and multiply
        const __m128 invMag4 = _mm_shuffle_ps(rsqrt, rsqrt, _MM_SHUFFLE(0, 0, 0, 0));
        vec = _mm_mul_ps(vec, invMag4);
    }

    VxSIMDStoreFloat3(&v->x, vec);
}

void VxSIMDRotateVector_SSE(VxVector *result, const VxMatrix *mat, const VxVector *v) {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 res = VxSIMDMatrixRotateVector3((const float*)&(*mat)[0][0], vec);
    VxSIMDStoreFloat3(&result->x, res);
}

void VxSIMDMultiplyMatrix_SSE(VxMatrix *result, const VxMatrix *a, const VxMatrix *b) {
    // NOTE:
    // VxMatrix::operator[] reinterprets float storage as VxVector4. That is UB under
    // strict-aliasing and can produce scalar/SIMD mismatches under optimization.
    // Use raw float pointer access here to guarantee identical memory interpretation.
    const float* ap = static_cast<const float*>(static_cast<const void*>(*a));
    const float* bp = static_cast<const float*>(static_cast<const void*>(*b));

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

        const __m128 res = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(a0, b_x), _mm_mul_ps(a1, b_y)),
            _mm_add_ps(_mm_mul_ps(a2, b_z), _mm_mul_ps(a3, b_w)));
        _mm_storeu_ps(out + i * 4, res);
    }

    // Enforce 3D transformation constraints
    out[0 * 4 + 3] = 0.0f;
    out[1 * 4 + 3] = 0.0f;
    out[2 * 4 + 3] = 0.0f;
    out[3 * 4 + 3] = 1.0f;

    memcpy(result, out, sizeof(out));
}

void VxSIMDMultiplyMatrix4_SSE(VxMatrix *result, const VxMatrix *a, const VxMatrix *b) {
    const float* ap = static_cast<const float*>(static_cast<const void*>(*a));
    const float* bp = static_cast<const float*>(static_cast<const void*>(*b));

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

        const __m128 res = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(a0, b_x), _mm_mul_ps(a1, b_y)),
            _mm_add_ps(_mm_mul_ps(a2, b_z), _mm_mul_ps(a3, b_w)));

        _mm_storeu_ps(out + i * 4, res);
    }

    memcpy(result, out, sizeof(out));
}

void VxSIMDTransposeMatrix_SSE(VxMatrix *result, const VxMatrix *a) {
    __m128 r0 = _mm_loadu_ps((const float*)&(*a)[0][0]);
    __m128 r1 = _mm_loadu_ps((const float*)&(*a)[1][0]);
    __m128 r2 = _mm_loadu_ps((const float*)&(*a)[2][0]);
    __m128 r3 = _mm_loadu_ps((const float*)&(*a)[3][0]);

    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

    VxMatrix temp;
    _mm_storeu_ps((float*)&temp[0][0], r0);
    _mm_storeu_ps((float*)&temp[1][0], r1);
    _mm_storeu_ps((float*)&temp[2][0], r2);
    _mm_storeu_ps((float*)&temp[3][0], r3);

    *result = temp;
}

void VxSIMDMultiplyMatrixVector_SSE(VxVector *result, const VxMatrix *mat, const VxVector *v) {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 res = VxSIMDMatrixMultiplyVector3((const float*)&(*mat)[0][0], vec);
    VxSIMDStoreFloat3(&result->x, res);
}

void VxSIMDMultiplyMatrixVector4_SSE(VxVector4 *result, const VxMatrix *mat, const VxVector4 *v) {
    __m128 vec = VxSIMDLoadFloat4((const float*)v);
    __m128 m0 = _mm_loadu_ps((const float*)&(*mat)[0][0]);
    __m128 m1 = _mm_loadu_ps((const float*)&(*mat)[1][0]);
    __m128 m2 = _mm_loadu_ps((const float*)&(*mat)[2][0]);
    __m128 m3 = _mm_loadu_ps((const float*)&(*mat)[3][0]);

    __m128 v_x = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v_w = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(3, 3, 3, 3));

    __m128 res = _mm_add_ps(_mm_add_ps(_mm_mul_ps(m0, v_x), _mm_mul_ps(m1, v_y)),
                            _mm_add_ps(_mm_mul_ps(m2, v_z), _mm_mul_ps(m3, v_w)));
    VxSIMDStoreFloat4((float*)result, res);
}

void VxSIMDRotateVectorOp_SSE(VxVector *result, const VxMatrix *mat, const VxVector *v) {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 res = VxSIMDMatrixRotateVector3((const float*)&(*mat)[0][0], vec);
    VxSIMDStoreFloat3(&result->x, res);
}

void VxSIMDMultiplyMatrixVectorMany_SSE(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride) {
    if (count <= 0) return;

    const char *srcPtr = reinterpret_cast<const char *>(vectors);
    char *dstPtr = reinterpret_cast<char *>(results);

    for (int i = 0; i < count; ++i) {
        const VxVector *vec = reinterpret_cast<const VxVector *>(srcPtr + i * stride);
        VxVector *result = reinterpret_cast<VxVector *>(dstPtr + i * stride);

        __m128 v = VxSIMDLoadFloat3(&vec->x);
        __m128 r = VxSIMDMatrixMultiplyVector3((const float*)&(*mat)[0][0], v);
        VxSIMDStoreFloat3(&result->x, r);
    }
}

void VxSIMDRotateVectorMany_SSE(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride) {
    if (count <= 0) return;

    const char *srcPtr = reinterpret_cast<const char *>(vectors);
    char *dstPtr = reinterpret_cast<char *>(results);

    for (int i = 0; i < count; ++i) {
        const VxVector *vec = reinterpret_cast<const VxVector *>(srcPtr + i * stride);
        VxVector *result = reinterpret_cast<VxVector *>(dstPtr + i * stride);

        __m128 v = VxSIMDLoadFloat3(&vec->x);
        __m128 r = VxSIMDMatrixRotateVector3((const float*)&(*mat)[0][0], v);
        VxSIMDStoreFloat3(&result->x, r);
    }
}

void VxSIMDNormalizeQuaternion_SSE(VxQuaternion *q) {
    // Optimized SSE quaternion normalization using rsqrt with Newton-Raphson
    // Load all 4 components (x, y, z, w)
    __m128 quat = _mm_loadu_ps(&q->x);

    // Compute dot product: x*x + y*y + z*z + w*w
    __m128 mul = _mm_mul_ps(quat, quat);
    // Sum all 4 components
    __m128 sum = _mm_add_ps(mul, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
    // Now sum has magSq broadcast to all lanes

    float magSqScalar;
    _mm_store_ss(&magSqScalar, sum);

    const float epsilonSq = EPSILON * EPSILON;
    if (magSqScalar <= epsilonSq) {
        return; // Below threshold, keep original quaternion to match scalar behavior
    }

    // Use rsqrt with Newton-Raphson for fast, accurate 1/sqrt
    __m128 rsqrt = _mm_rsqrt_ps(sum);
    // Newton-Raphson iteration: rsqrt' = rsqrt * (1.5 - 0.5 * magSq * rsqrt^2)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_half = _mm_set1_ps(1.5f);
    __m128 rsqrt_sq = _mm_mul_ps(rsqrt, rsqrt);
    __m128 half_sum = _mm_mul_ps(half, sum);
    __m128 correction = _mm_sub_ps(three_half, _mm_mul_ps(half_sum, rsqrt_sq));
    rsqrt = _mm_mul_ps(rsqrt, correction);

    // Normalize
    __m128 result = _mm_mul_ps(quat, rsqrt);
    _mm_storeu_ps(&q->x, result);
}

void VxSIMDMultiplyQuaternion_SSE(VxQuaternion *result, const VxQuaternion *a, const VxQuaternion *b) {
    // Optimized SSE quaternion multiplication using minimal shuffles
    // Load quaternions as {x, y, z, w}
    __m128 qa = _mm_loadu_ps(&a->x);
    __m128 qb = _mm_loadu_ps(&b->x);

    // Quaternion multiplication formula:
    // result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
    // result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
    // result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    // result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z

    // Broadcast a.w to all lanes: {w, w, w, w}
    __m128 aw = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(3, 3, 3, 3));

    // Term 0: aw * {bx, by, bz, bw}
    __m128 t0 = _mm_mul_ps(aw, qb);

    // Shuffle b for subsequent terms
    // For ax: need {bw, bz, by, bx} permuted with signs +,-,+,-
    __m128 ax = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 b_perm1 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(0, 1, 2, 3)); // {w, z, y, x}
    __m128 t1 = _mm_mul_ps(ax, b_perm1);

    // For ay: need {bz, bw, bx, by} permuted with signs +,+,-,-
    __m128 ay = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 b_perm2 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(1, 0, 3, 2)); // {y, x, w, z}
    __m128 t2 = _mm_mul_ps(ay, b_perm2);

    // For az: need {by, bx, bw, bz} permuted with signs -,+,+,-
    __m128 az = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 b_perm3 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(2, 3, 0, 1)); // {z, w, x, y}
    __m128 t3 = _mm_mul_ps(az, b_perm3);

    // Apply signs using efficient add/sub pattern
    // t0 signs: +,+,+,+ (all positive)
    // t1 signs: +,-,+,- -> addsub pattern
    // t2 signs: +,+,-,- -> add for xy, sub for zw
    // t3 signs: -,+,+,- -> sub for x, add for yz, sub for w

    // Use SSE3 addsub if available, otherwise use explicit sign masks
#if defined(VX_SIMD_SSE3)
    // t0 + t1 with alternating sign: (t0 +- t1) but pattern is wrong
    // Manually handle for correctness
    __m128 sign1 = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    t1 = _mm_mul_ps(t1, sign1);
    __m128 sign2 = _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f);
    t2 = _mm_mul_ps(t2, sign2);
    __m128 sign3 = _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f);
    t3 = _mm_mul_ps(t3, sign3);
#else
    // Pre-computed sign masks
    __m128 sign1 = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    t1 = _mm_mul_ps(t1, sign1);
    __m128 sign2 = _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f);
    t2 = _mm_mul_ps(t2, sign2);
    __m128 sign3 = _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f);
    t3 = _mm_mul_ps(t3, sign3);
#endif

    // Sum all terms
    __m128 r = _mm_add_ps(_mm_add_ps(t0, t1), _mm_add_ps(t2, t3));

    _mm_storeu_ps(&result->x, r);
}

void VxSIMDSlerpQuaternion_SSE(VxQuaternion *result, float t, const VxQuaternion *a, const VxQuaternion *b) {
    __m128 qa = _mm_loadu_ps(&a->x);
    __m128 qb = _mm_loadu_ps(&b->x);

    __m128 cosOmegaVec = VxSIMDDotProduct4(qa, qb);
    float cosOmega;
    _mm_store_ss(&cosOmega, cosOmegaVec);

    float k0;
    float k1;

    if (cosOmega >= 0.0f) {
        float oneMinusCos = 1.0f - cosOmega;
        if (oneMinusCos < 0.01f) {
            k0 = 1.0f - t;
            k1 = t;
        } else {
            float omega = acosf(cosOmega);
            float invSinOmega = 1.0f / sinf(omega);
            k0 = sinf((1.0f - t) * omega) * invSinOmega;
            k1 = sinf(t * omega) * invSinOmega;
        }
    } else {
        float oneMinusCosNeg = 1.0f - (-cosOmega);
        if (oneMinusCosNeg < 0.01f) {
            k0 = 1.0f - t;
            k1 = -t;
        } else {
            float omega = acosf(-cosOmega);
            float invSinOmega = 1.0f / sinf(omega);
            k0 = sinf((1.0f - t) * omega) * invSinOmega;
            k1 = -sinf(t * omega) * invSinOmega;
        }
    }

    __m128 k0Vec = _mm_set1_ps(k0);
    __m128 k1Vec = _mm_set1_ps(k1);
    __m128 r = _mm_add_ps(_mm_mul_ps(qa, k0Vec), _mm_mul_ps(qb, k1Vec));
    _mm_storeu_ps(&result->x, r);
}

int VxSIMDConvertPixelBatch32_SSE(const XULONG* srcPixels, XULONG* dstPixels, int count, const VxPixelSimdConfig& config) {
    if (!config.enabled) {
        return 0;
    }

    const int simdCount = count & ~3;
    if (simdCount <= 0) {
        return 0;
    }

    const __m128i alphaVec = config.alphaFill ? _mm_set1_epi32(static_cast<int>(config.alphaFillComponent)) : _mm_setzero_si128();
    __m128i srcMaskVec[4];
    __m128i dstMaskVec[4];
    for (int c = 0; c < 4; ++c) {
        srcMaskVec[c] = _mm_set1_epi32(static_cast<int>(config.srcMasks[c]));
        dstMaskVec[c] = _mm_set1_epi32(static_cast<int>(config.dstMasks[c]));
    }

    for (int i = 0; i < simdCount; i += 4) {
        __m128i srcVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcPixels + i));
        __m128i dstVec = alphaVec;

        for (int c = 0; c < 4; ++c) {
            if (!config.channelCopy[c]) {
                continue;
            }

            __m128i channel = _mm_and_si128(srcVec, srcMaskVec[c]);
            if (config.srcShiftRight[c]) {
                __m128i shift = _mm_cvtsi32_si128(config.srcShiftRight[c]);
                channel = _mm_srl_epi32(channel, shift);
            }
            if (config.dstShiftLeft[c]) {
                __m128i shift = _mm_cvtsi32_si128(config.dstShiftLeft[c]);
                channel = _mm_sll_epi32(channel, shift);
            }
            channel = _mm_and_si128(channel, dstMaskVec[c]);
            dstVec = _mm_or_si128(dstVec, channel);
        }

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dstPixels + i), dstVec);
    }

    return simdCount;
}

int VxSIMDApplyAlphaBatch32_SSE(XULONG* pixels, int count, XBYTE alphaValue, XULONG alphaMask, XULONG alphaShift) {
    const int simdCount = count & ~3;
    if (simdCount <= 0) {
        return 0;
    }

    const XULONG alphaComponent = (static_cast<XULONG>(alphaValue) << alphaShift) & alphaMask;
    const XULONG colorMask = ~alphaMask;

    const __m128i alphaMaskVec = _mm_set1_epi32(static_cast<int>(alphaMask));
    const __m128i colorMaskVec = _mm_set1_epi32(static_cast<int>(colorMask));
    __m128i alphaVec = _mm_set1_epi32(static_cast<int>(alphaComponent));
    alphaVec = _mm_and_si128(alphaVec, alphaMaskVec);

    for (int i = 0; i < simdCount; i += 4) {
        __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pixels + i));
        __m128i masked = _mm_and_si128(src, colorMaskVec);
        __m128i result = _mm_or_si128(masked, alphaVec);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pixels + i), result);
    }

    return simdCount;
}

int VxSIMDApplyVariableAlphaBatch32_SSE(XULONG* pixels, const XBYTE* alphaValues, int count, XULONG alphaMask, XULONG alphaShift) {
    const int simdCount = count & ~3;
    if (simdCount <= 0) {
        return 0;
    }

    const XULONG colorMask = ~alphaMask;
    const __m128i alphaMaskVec = _mm_set1_epi32(static_cast<int>(alphaMask));
    const __m128i colorMaskVec = _mm_set1_epi32(static_cast<int>(colorMask));
    const __m128i zero = _mm_setzero_si128();
    const __m128i alphaShiftVec = _mm_cvtsi32_si128(alphaShift);

    for (int i = 0; i < simdCount; i += 4) {
        unsigned int packedAlpha = 0;
        memcpy(&packedAlpha, alphaValues + i, sizeof(packedAlpha));

        __m128i alphaBytes = _mm_cvtsi32_si128(static_cast<int>(packedAlpha));
        alphaBytes = _mm_unpacklo_epi8(alphaBytes, zero);
        alphaBytes = _mm_unpacklo_epi16(alphaBytes, zero);

        __m128i alphaVec = _mm_sll_epi32(alphaBytes, alphaShiftVec);
        alphaVec = _mm_and_si128(alphaVec, alphaMaskVec);

        __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pixels + i));
        __m128i masked = _mm_and_si128(src, colorMaskVec);
        __m128i result = _mm_or_si128(masked, alphaVec);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pixels + i), result);
    }

    return simdCount;
}

void VxSIMDInterpolateFloatArray_SSE(float *result, const float *a, const float *b, float factor, int count) {
    // Match scalar VxMath::InterpolateFloatArray semantics and evaluation order:
    // result = a + (b - a) * factor
    const __m128 factorVec = _mm_set1_ps(factor);

    const int simdCount = count & ~3; // Process 4 at a time
    for (int i = 0; i < simdCount; i += 4) {
        const __m128 aVec = _mm_loadu_ps(a + i);
        const __m128 bVec = _mm_loadu_ps(b + i);
        const __m128 diff = _mm_sub_ps(bVec, aVec);
        const __m128 resultVec = _mm_add_ps(aVec, _mm_mul_ps(diff, factorVec));
        _mm_storeu_ps(result + i, resultVec);
    }

    for (int i = simdCount; i < count; ++i) {
        result[i] = a[i] + (b[i] - a[i]) * factor;
    }
}

void VxSIMDInterpolateVectorArray_SSE(void *result, const void *a, const void *b, float factor, int count, XULONG strideResult, XULONG strideInput) {
    // Simple implementation - process vectors one by one
    const char* srcA = static_cast<const char*>(a);
    const char* srcB = static_cast<const char*>(b);
    char* dst = static_cast<char*>(result);

    for (int i = 0; i < count; ++i) {
        const VxVector* vecA = reinterpret_cast<const VxVector*>(srcA + i * strideInput);
        const VxVector* vecB = reinterpret_cast<const VxVector*>(srcB + i * strideInput);
        VxVector* vecResult = reinterpret_cast<VxVector*>(dst + i * strideResult);

        __m128 aVec = VxSIMDLoadFloat3(&vecA->x);
        __m128 bVec = VxSIMDLoadFloat3(&vecB->x);
        __m128 factorVec = _mm_set1_ps(factor);
        __m128 oneMinusFactor = _mm_set1_ps(1.0f - factor);
        __m128 resultVec = _mm_add_ps(_mm_mul_ps(aVec, oneMinusFactor), _mm_mul_ps(bVec, factorVec));
        VxSIMDStoreFloat3(&vecResult->x, resultVec);
    }
}

XBOOL VxSIMDTransformBox2D_SSE(const VxMatrix *worldProjection, const VxBbox *box, VxRect *screenSize, VxRect *extents, VXCLIP_FLAGS *orClipFlags, VXCLIP_FLAGS *andClipFlags) {
    // Optimized SSE implementation based on original binary patterns
    if (!box || !box->IsValid()) {
        if (orClipFlags) *orClipFlags = VXCLIP_ALL;
        if (andClipFlags) *andClipFlags = VXCLIP_ALL;
        return FALSE;
    }

    // Ensure deterministic output even when fully clipped.
    if (extents) {
        extents->left = 0.0f;
        extents->top = 0.0f;
        extents->right = 0.0f;
        extents->bottom = 0.0f;
    }

    __m128 verts[8];

    // Transform Min corner (xyz with implicit w=1)
    verts[0] = VxSIMDMatrixMultiplyVector3((const float*)&(*worldProjection)[0][0], VxSIMDLoadFloat3(&box->Min.x));

    const float dx = box->Max.x - box->Min.x;
    const float dy = box->Max.y - box->Min.y;
    const float dz = box->Max.z - box->Min.z;

    const __m128 col0 = _mm_loadu_ps((const float*)&(*worldProjection)[0][0]);
    const __m128 col1 = _mm_loadu_ps((const float*)&(*worldProjection)[1][0]);
    const __m128 col2 = _mm_loadu_ps((const float*)&(*worldProjection)[2][0]);

    const __m128 deltaX = _mm_mul_ps(col0, _mm_set1_ps(dx));
    const __m128 deltaY = _mm_mul_ps(col1, _mm_set1_ps(dy));
    const int vertexCount = (dz == 0.0f) ? 4 : 8;

    if (vertexCount == 4) {
        // 4-vertex case: dz == 0, use deltaX first
        verts[1] = _mm_add_ps(verts[0], deltaX);
        verts[2] = _mm_add_ps(verts[0], deltaY);
        verts[3] = _mm_add_ps(verts[1], deltaY);
    } else {
        // 8-vertex case: use deltaZ first to match original binary vertex order
        const __m128 deltaZ = _mm_mul_ps(col2, _mm_set1_ps(dz));
        verts[1] = _mm_add_ps(verts[0], deltaZ);              // v[0] + deltaZ
        verts[2] = _mm_add_ps(verts[0], deltaY);              // v[0] + deltaY
        verts[3] = _mm_add_ps(verts[1], deltaY);              // v[1] + deltaY
        verts[4] = _mm_add_ps(verts[0], deltaX);              // v[0] + deltaX
        verts[5] = _mm_add_ps(verts[4], deltaZ);              // v[4] + deltaZ
        verts[6] = _mm_add_ps(verts[4], deltaY);              // v[4] + deltaY
        verts[7] = _mm_add_ps(verts[5], deltaY);              // v[5] + deltaY
    }

    XULONG allOr = 0;
    XULONG allAnd = 0xFFFFFFFFu;

    // SSE clip test in homogeneous coordinates - process 2 vertices at a time
    // Using SSE comparison and mask generation similar to original binary
    for (int i = 0; i < vertexCount; i += 2) {
        __m128 v0 = verts[i];
        __m128 v1 = (i + 1 < vertexCount) ? verts[i + 1] : v0;

        // Extract components - x,y,z,w for each vertex
        alignas(16) float v0f[4], v1f[4];
        _mm_store_ps(v0f, v0);
        _mm_store_ps(v1f, v1);

        XULONG flags0 = 0, flags1 = 0;

        // Test vertex 0
        {
            const float x = v0f[0], y = v0f[1], z = v0f[2], w = v0f[3];
            if (-w > x) flags0 |= VXCLIP_LEFT;
            if (x > w) flags0 |= VXCLIP_RIGHT;
            if (-w > y) flags0 |= VXCLIP_BOTTOM;
            if (y > w) flags0 |= VXCLIP_TOP;
            if (z < 0.0f) flags0 |= VXCLIP_FRONT;
            if (z > w) flags0 |= VXCLIP_BACK;
        }

        // Test vertex 1 (only if valid)
        if (i + 1 < vertexCount) {
            const float x = v1f[0], y = v1f[1], z = v1f[2], w = v1f[3];
            if (-w > x) flags1 |= VXCLIP_LEFT;
            if (x > w) flags1 |= VXCLIP_RIGHT;
            if (-w > y) flags1 |= VXCLIP_BOTTOM;
            if (y > w) flags1 |= VXCLIP_TOP;
            if (z < 0.0f) flags1 |= VXCLIP_FRONT;
            if (z > w) flags1 |= VXCLIP_BACK;
        } else {
            flags1 = flags0; // Duplicate for correct AND
        }

        allOr |= flags0 | flags1;
        allAnd &= flags0 & flags1;
    }

    if (extents && screenSize && (allAnd & VXCLIP_ALL) == 0) {
        // Use SSE min/max operations for screen bounds
        __m128 minXY = _mm_set1_ps(1000000.0f);
        __m128 maxXY = _mm_set1_ps(-1000000.0f);

        const float halfWidth = (screenSize->right - screenSize->left) * 0.5f;
        const float halfHeight = (screenSize->bottom - screenSize->top) * 0.5f;
        const float centerX = halfWidth + screenSize->left;
        const float centerY = halfHeight + screenSize->top;

        // Precompute scale/offset for viewport transformation
        __m128 viewScale = _mm_setr_ps(halfWidth, -halfHeight, 0.0f, 0.0f);
        __m128 viewOffset = _mm_setr_ps(centerX, centerY, 0.0f, 0.0f);

        for (int i = 0; i < vertexCount; ++i) {
            __m128 v = verts[i];
            __m128 w = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

            // Skip vertices behind camera (w <= 0)
            float wf;
            _mm_store_ss(&wf, w);
            if (wf <= 0.0f) continue;

            // Compute 1/w using fast reciprocal with Newton-Raphson
            __m128 rcp = _mm_rcp_ss(w);
            __m128 two = _mm_set_ss(2.0f);
            rcp = _mm_mul_ss(rcp, _mm_sub_ss(two, _mm_mul_ss(w, rcp)));
            __m128 invW = _mm_shuffle_ps(rcp, rcp, _MM_SHUFFLE(0, 0, 0, 0));

            // Project: screenXY = v.xy * invW * scale + offset
            __m128 projected = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(v, invW), viewScale), viewOffset);

            // Update min/max using SSE
            minXY = _mm_min_ps(minXY, projected);
            maxXY = _mm_max_ps(maxXY, projected);
        }

        // Extract results
        alignas(16) float minf[4], maxf[4];
        _mm_store_ps(minf, minXY);
        _mm_store_ps(maxf, maxXY);

        extents->left = minf[0];
        extents->bottom = maxf[1];
        extents->right = maxf[0];
        extents->top = minf[1];
    }

    // Only clamp to screen edges when some vertices are clipped by the near plane.
    if ((allOr & VXCLIP_FRONT) != 0 && extents && screenSize) {
        if ((allOr & VXCLIP_LEFT) != 0) extents->left = screenSize->left;
        if ((allOr & VXCLIP_RIGHT) != 0) extents->right = screenSize->right;
        if ((allOr & VXCLIP_TOP) != 0) extents->top = screenSize->top;
        if ((allOr & VXCLIP_BOTTOM) != 0) extents->bottom = screenSize->bottom;
    }

    if (orClipFlags) *orClipFlags = (VXCLIP_FLAGS)allOr;
    if (andClipFlags) *andClipFlags = (VXCLIP_FLAGS)allAnd;
    return (allAnd & VXCLIP_ALL) == 0;
}

void VxSIMDProjectBoxZExtents_SSE(const VxMatrix *worldProjection, const VxBbox *box, float *zhMin, float *zhMax) {
    if (!zhMin || !zhMax) return;
    *zhMin = 1.0e10f;
    *zhMax = -1.0e10f;

    if (!box || !box->IsValid()) {
        return;
    }

    __m128 corners[8];
    corners[0] = VxSIMDMatrixMultiplyVector3((const float*)&(*worldProjection)[0][0], VxSIMDLoadFloat3(&box->Min.x));

    const float dx = box->Max.x - box->Min.x;
    const float dy = box->Max.y - box->Min.y;
    const float dz = box->Max.z - box->Min.z;

    const __m128 col0 = _mm_loadu_ps((const float*)&(*worldProjection)[0][0]);
    const __m128 col1 = _mm_loadu_ps((const float*)&(*worldProjection)[1][0]);
    const __m128 col2 = _mm_loadu_ps((const float*)&(*worldProjection)[2][0]);
    const __m128 deltaX = _mm_mul_ps(col0, _mm_set1_ps(dx));
    const __m128 deltaY = _mm_mul_ps(col1, _mm_set1_ps(dy));

    const __m128 deltaZ = _mm_mul_ps(col2, _mm_set1_ps(dz));

    // Match the scalar/binary degeneracy handling (FLT_EPSILON thresholds).
    int vertexCount = 8;
    if (dz <= FLT_EPSILON) {
        vertexCount = 4;
        corners[1] = _mm_add_ps(corners[0], deltaX);
        corners[2] = _mm_add_ps(corners[0], deltaY);
        corners[3] = _mm_add_ps(corners[1], deltaY);
    } else if (dx <= FLT_EPSILON) {
        vertexCount = 4;
        corners[1] = _mm_add_ps(corners[0], deltaZ);
        corners[2] = _mm_add_ps(corners[0], deltaY);
        corners[3] = _mm_add_ps(corners[1], deltaY);
    } else if (dy <= FLT_EPSILON) {
        vertexCount = 4;
        corners[1] = _mm_add_ps(corners[0], deltaZ);
        corners[2] = _mm_add_ps(corners[0], deltaX);
        corners[3] = _mm_add_ps(corners[1], deltaX);
    } else {
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
        float z = v[2];
        const float w = v[3];

        if (w < 0.0f) {
            z = 0.0f;
        } else {
            z = z / w; // matches original behavior (w==0 produces inf)
        }

        if (z < *zhMin) *zhMin = z;
        if (z >= *zhMax) *zhMax = z;
    }
}

XBOOL VxSIMDComputeBestFitBBox_SSE(const XBYTE *points, XULONG stride, int count, VxMatrix *bboxMatrix, float additionalBorder) {
    if (!bboxMatrix) {
        return FALSE;
    }

    if (count <= 0 || !points) {
        bboxMatrix->SetIdentity();
        return FALSE;
    }

    // Eigen decomposition is handled by existing code; SIMD is used for the projection/min-max pass.
    VxEigenMatrix eigenMat;
    eigenMat.Covariance((float*)points, stride, count);
    eigenMat.EigenStuff3();

    // Copy eigenvectors into the bbox matrix (transposed) to match the scalar/binary.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            (*bboxMatrix)[i][j] = eigenMat[j][i];
        }
        // Ensure W is zero before any SIMD cross-product/shuffles (the helper uses all lanes).
        (*bboxMatrix)[i][3] = 0.0f;
    }
    (*bboxMatrix)[3][0] = 0.0f;
    (*bboxMatrix)[3][1] = 0.0f;
    (*bboxMatrix)[3][2] = 0.0f;
    (*bboxMatrix)[3][3] = 1.0f;

    // Normalize the first two axes
    (*bboxMatrix)[0].Normalize();
    (*bboxMatrix)[1].Normalize();

    // Third axis = cross(axis0, axis1), using the same scalar formula as the binary.
    {
        const float ax0 = (*bboxMatrix)[0][0];
        const float ay0 = (*bboxMatrix)[0][1];
        const float az0 = (*bboxMatrix)[0][2];
        const float ax1 = (*bboxMatrix)[1][0];
        const float ay1 = (*bboxMatrix)[1][1];
        const float az1 = (*bboxMatrix)[1][2];

        (*bboxMatrix)[2][0] = ay0 * az1 - ay1 * az0;
        (*bboxMatrix)[2][1] = az0 * ax1 - az1 * ax0;
        (*bboxMatrix)[2][2] = ax0 * ay1 - ay0 * ax1;
        (*bboxMatrix)[2][3] = 0.0f;
    }

    // Min/max projections along the three axes
    float minX, maxX, minY, maxY, minZ, maxZ;
    {
        const float* p0 = (const float*)points;
        const __m128 pt = _mm_set_ps(0.0f, p0[2], p0[1], p0[0]);

        const __m128 axis0 = _mm_loadu_ps((const float*)&(*bboxMatrix)[0][0]);
        const __m128 axis1 = _mm_loadu_ps((const float*)&(*bboxMatrix)[1][0]);
        const __m128 axis2 = _mm_loadu_ps((const float*)&(*bboxMatrix)[2][0]);

        const __m128 m0 = _mm_mul_ps(pt, axis0);
        const __m128 m1 = _mm_mul_ps(pt, axis1);
        const __m128 m2 = _mm_mul_ps(pt, axis2);

        __m128 s0 = _mm_add_ss(m0, _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(1, 1, 1, 1)));
        s0 = _mm_add_ss(s0, _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(2, 2, 2, 2)));
        __m128 s1 = _mm_add_ss(m1, _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(1, 1, 1, 1)));
        s1 = _mm_add_ss(s1, _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(2, 2, 2, 2)));
        __m128 s2 = _mm_add_ss(m2, _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(1, 1, 1, 1)));
        s2 = _mm_add_ss(s2, _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(2, 2, 2, 2)));

        _mm_store_ss(&minX, s0);
        _mm_store_ss(&minY, s1);
        _mm_store_ss(&minZ, s2);
        maxX = minX;
        maxY = minY;
        maxZ = minZ;
    }

    const __m128 axis0 = _mm_loadu_ps((const float*)&(*bboxMatrix)[0][0]);
    const __m128 axis1 = _mm_loadu_ps((const float*)&(*bboxMatrix)[1][0]);
    const __m128 axis2 = _mm_loadu_ps((const float*)&(*bboxMatrix)[2][0]);

    for (int i = 0; i < count; ++i) {
        const float* p = (const float*)(points + i * stride);
        const __m128 pt = _mm_set_ps(0.0f, p[2], p[1], p[0]);

        const __m128 m0 = _mm_mul_ps(pt, axis0);
        const __m128 m1 = _mm_mul_ps(pt, axis1);
        const __m128 m2 = _mm_mul_ps(pt, axis2);

        __m128 s0 = _mm_add_ss(m0, _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(1, 1, 1, 1)));
        s0 = _mm_add_ss(s0, _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(2, 2, 2, 2)));
        __m128 s1 = _mm_add_ss(m1, _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(1, 1, 1, 1)));
        s1 = _mm_add_ss(s1, _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(2, 2, 2, 2)));
        __m128 s2 = _mm_add_ss(m2, _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(1, 1, 1, 1)));
        s2 = _mm_add_ss(s2, _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(2, 2, 2, 2)));

        float x, y, z;
        _mm_store_ss(&x, s0);
        _mm_store_ss(&y, s1);
        _mm_store_ss(&z, s2);

        // Match scalar/binary update ordering.
        if (x >= maxX) {
            maxX = x;
        } else if (x < minX) {
            minX = x;
        }
        if (y >= maxY) {
            maxY = y;
        } else if (y < minY) {
            minY = y;
        }
        if (z >= maxZ) {
            maxZ = z;
        } else if (z < minZ) {
            minZ = z;
        }
    }

    const float centerX = (minX + maxX) * 0.5f;
    const float centerY = (minY + maxY) * 0.5f;
    const float centerZ = (minZ + maxZ) * 0.5f;

    const float halfX = (maxX - minX) * 0.5f + additionalBorder;
    const float halfY = (maxY - minY) * 0.5f + additionalBorder;
    const float halfZ = (maxZ - minZ) * 0.5f + additionalBorder;

    // Transform center back to world space using the unscaled axes (matches scalar/binary).
    (*bboxMatrix)[3][0] = centerX * (*bboxMatrix)[0][0] + centerY * (*bboxMatrix)[1][0] + centerZ * (*bboxMatrix)[2][0];
    (*bboxMatrix)[3][1] = centerX * (*bboxMatrix)[0][1] + centerY * (*bboxMatrix)[1][1] + centerZ * (*bboxMatrix)[2][1];
    (*bboxMatrix)[3][2] = centerX * (*bboxMatrix)[0][2] + centerY * (*bboxMatrix)[1][2] + centerZ * (*bboxMatrix)[2][2];
    (*bboxMatrix)[3][3] = 1.0f;

    // Scale axes by extents.
    {
        __m128 a = _mm_loadu_ps((const float*)&(*bboxMatrix)[0][0]);
        a = _mm_mul_ps(a, _mm_set1_ps(halfX));
        _mm_storeu_ps((float*)&(*bboxMatrix)[0][0], a);
        (*bboxMatrix)[0][3] = 0.0f;
    }
    {
        __m128 a = _mm_loadu_ps((const float*)&(*bboxMatrix)[1][0]);
        a = _mm_mul_ps(a, _mm_set1_ps(halfY));
        _mm_storeu_ps((float*)&(*bboxMatrix)[1][0], a);
        (*bboxMatrix)[1][3] = 0.0f;
    }
    {
        __m128 a = _mm_loadu_ps((const float*)&(*bboxMatrix)[2][0]);
        a = _mm_mul_ps(a, _mm_set1_ps(halfZ));
        _mm_storeu_ps((float*)&(*bboxMatrix)[2][0], a);
        (*bboxMatrix)[2][3] = 0.0f;
    }

    // Validate for reasonable values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (fabs((*bboxMatrix)[i][j]) > 1000000.0f) {
                return FALSE;
            }
        }
    }

    return TRUE;
}

int VxSIMDBboxClassify_SSE(const VxBbox *self, const VxBbox *other, const VxVector *point) {
    // Load bbox min/max and point using SSE
    __m128 selfMin = VxSIMDLoadFloat3(&self->Min.x);
    __m128 selfMax = VxSIMDLoadFloat3(&self->Max.x);
    __m128 ptVec = VxSIMDLoadFloat3(&point->x);
    __m128 otherMin = VxSIMDLoadFloat3(&other->Min.x);
    __m128 otherMax = VxSIMDLoadFloat3(&other->Max.x);

    // Classify point against this box
    XULONG ptFlags = 0;

    // Compare point with self bounds
    __m128 ptLessThanMin = _mm_cmplt_ps(ptVec, selfMin);
    __m128 ptGreaterThanMax = _mm_cmpgt_ps(ptVec, selfMax);

    // Extract comparison results
    int ptLessFlags = _mm_movemask_ps(ptLessThanMin);
    int ptGreaterFlags = _mm_movemask_ps(ptGreaterThanMax);

    // Build flags based on axis (x=0, y=1, z=2)
    if (ptLessFlags & 1) ptFlags |= VXCLIP_LEFT;      // x < Min.x
    if (ptGreaterFlags & 1) ptFlags |= VXCLIP_RIGHT;  // x > Max.x
    if (ptLessFlags & 2) ptFlags |= VXCLIP_BOTTOM;    // y < Min.y
    if (ptGreaterFlags & 2) ptFlags |= VXCLIP_TOP;    // y > Max.y
    if (ptLessFlags & 4) ptFlags |= VXCLIP_BACK;      // z < Min.z
    if (ptGreaterFlags & 4) ptFlags |= VXCLIP_FRONT;  // z > Max.z

    // Classify box2 against this box
    XULONG box2Flags = 0;

    __m128 otherMaxLessThanMin = _mm_cmplt_ps(otherMax, selfMin);
    __m128 otherMinGreaterThanMax = _mm_cmpgt_ps(otherMin, selfMax);

    int box2LessFlags = _mm_movemask_ps(otherMaxLessThanMin);
    int box2GreaterFlags = _mm_movemask_ps(otherMinGreaterThanMax);

    // Build flags (z checked first in original code)
    if (box2LessFlags & 4) box2Flags |= VXCLIP_BACK;
    if (box2GreaterFlags & 4) box2Flags |= VXCLIP_FRONT;
    if (box2LessFlags & 1) box2Flags |= VXCLIP_LEFT;
    if (box2GreaterFlags & 1) box2Flags |= VXCLIP_RIGHT;
    if (box2LessFlags & 2) box2Flags |= VXCLIP_BOTTOM;
    if (box2GreaterFlags & 2) box2Flags |= VXCLIP_TOP;

    if (ptFlags) {
        if (!box2Flags) {
            // Check if box2 is inside this box
            __m128 box2MinGE = _mm_cmpge_ps(otherMin, selfMin);
            __m128 box2MaxLE = _mm_cmple_ps(otherMax, selfMax);
            __m128 box2Inside = _mm_and_ps(box2MinGE, box2MaxLE);

            if ((_mm_movemask_ps(box2Inside) & 0x7) == 0x7) {
                return -1;
            }

            // Check if this box is inside box2
            __m128 selfMinGE = _mm_cmpge_ps(selfMin, otherMin);
            __m128 selfMaxLE = _mm_cmple_ps(selfMax, otherMax);
            __m128 selfInside = _mm_and_ps(selfMinGE, selfMaxLE);

            if ((_mm_movemask_ps(selfInside) & 0x7) == 0x7) {
                // Check if point is NOT in box2
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

        // Check if this box is inside box2
        __m128 selfMinGE = _mm_cmpge_ps(selfMin, otherMin);
        __m128 selfMaxLE = _mm_cmple_ps(selfMax, otherMax);
        __m128 selfInside = _mm_and_ps(selfMinGE, selfMaxLE);

        if ((_mm_movemask_ps(selfInside) & 0x7) == 0x7) {
            return 1;
        }
    }

    return 0;
}

void VxSIMDBboxClassifyVertices_SSE(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, XULONG *flags) {
    // Load bbox min/max using SSE
    __m128 bboxMin = VxSIMDLoadFloat3(&self->Min.x);
    __m128 bboxMax = VxSIMDLoadFloat3(&self->Max.x);

    for (int i = 0; i < count; ++i) {
        const float *v = reinterpret_cast<const float *>(vertices + i * stride);

        // Load vertex using SSE
        __m128 vertex = VxSIMDLoadFloat3(v);

        // Compare with bbox bounds
        __m128 lessThanMin = _mm_cmplt_ps(vertex, bboxMin);
        __m128 greaterThanMax = _mm_cmpgt_ps(vertex, bboxMax);

        // Extract comparison results
        int lessFlags = _mm_movemask_ps(lessThanMin);
        int greaterFlags = _mm_movemask_ps(greaterThanMax);

        // Build classification flags
        XULONG flag = 0;

        // Check Z axis first (as in original)
        if (lessFlags & 4) flag |= VXCLIP_BACK;
        else if (greaterFlags & 4) flag |= VXCLIP_FRONT;

        // Check Y axis
        if (lessFlags & 2) flag |= VXCLIP_BOTTOM;
        else if (greaterFlags & 2) flag |= VXCLIP_TOP;

        // Check X axis
        if (lessFlags & 1) flag |= VXCLIP_LEFT;
        else if (greaterFlags & 1) flag |= VXCLIP_RIGHT;

        flags[i] = flag;
    }
}

void VxSIMDBboxClassifyVerticesOneAxis_SSE(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, int axis, XULONG *flags) {
    if (axis < 0 || axis > 2) {
        // Invalid axis, set all flags to 0
        for (int i = 0; i < count; ++i)
            flags[i] = 0;
        return;
    }

    const float maxVal = *(&self->Max.x + axis);
    const float minVal = *(&self->Min.x + axis);

    // Create SSE vectors for comparison
    __m128 minVec = _mm_set1_ps(minVal);
    __m128 maxVec = _mm_set1_ps(maxVal);

    for (int i = 0; i < count; ++i) {
        const float *v = reinterpret_cast<const float *>(vertices + i * stride + axis * sizeof(float));

        // Load the single axis value and broadcast it
        __m128 val = _mm_load_ss(v);

        // Compare
        __m128 lessThan = _mm_cmplt_ss(val, minVec);
        __m128 greaterThan = _mm_cmpgt_ss(val, maxVec);

        // Extract results
        XULONG flag = 0;
        if (_mm_movemask_ps(lessThan) & 1) flag = 1;
        else if (_mm_movemask_ps(greaterThan) & 1) flag = 2;

        flags[i] = flag;
    }
}

void VxSIMDBboxTransformTo_SSE(const VxBbox *self, VxVector *points, const VxMatrix *mat) {
    // Match scalar VxBbox::TransformTo ordering exactly:
    // [0]=Min, [1]=Min+Z, [2]=Min+Y, [3]=Min+Y+Z, [4]=Min+X, [5]=Min+X+Z, [6]=Min+X+Y, [7]=Min+X+Y+Z

    // Transform the Min corner first
    VxSIMDMultiplyMatrixVector_SSE(&points[0], mat, &self->Min);

    // Size differences
    const float sizeX = self->Max.x - self->Min.x;
    const float sizeY = self->Max.y - self->Min.y;
    const float sizeZ = self->Max.z - self->Min.z;

    // Axis vectors in world space: size * matrix column
    const VxVector xVec(sizeX * (*mat)[0][0], sizeX * (*mat)[0][1], sizeX * (*mat)[0][2]);
    const VxVector yVec(sizeY * (*mat)[1][0], sizeY * (*mat)[1][1], sizeY * (*mat)[1][2]);
    const VxVector zVec(sizeZ * (*mat)[2][0], sizeZ * (*mat)[2][1], sizeZ * (*mat)[2][2]);

    // Build corners by adding combinations
    points[1] = points[0] + zVec;
    points[2] = points[0] + yVec;
    points[3] = points[2] + zVec;
    points[4] = points[0] + xVec;
    points[5] = points[4] + zVec;
    points[6] = points[4] + yVec;
    points[7] = points[6] + zVec;
}

void VxSIMDBboxTransformFrom_SSE(VxBbox *dest, const VxBbox *src, const VxMatrix *mat) {
    // Match scalar VxBbox::TransformFrom exactly (center + half-extents method).

    // Center of source box
    VxVector center((src->Min.x + src->Max.x) * 0.5f,
                   (src->Min.y + src->Max.y) * 0.5f,
                   (src->Min.z + src->Max.z) * 0.5f);

    // Transform center -> new center stored in dest->Min (scalar uses Min as temporary center)
    VxSIMDMultiplyMatrixVector_SSE(&dest->Min, mat, &center);

    // Size differences
    const float sizeX = src->Max.x - src->Min.x;
    const float sizeY = src->Max.y - src->Min.y;
    const float sizeZ = src->Max.z - src->Min.z;

    const VxVector xVec(sizeX * (*mat)[0][0], sizeX * (*mat)[0][1], sizeX * (*mat)[0][2]);
    const VxVector yVec(sizeY * (*mat)[1][0], sizeY * (*mat)[1][1], sizeY * (*mat)[1][2]);
    const VxVector zVec(sizeZ * (*mat)[2][0], sizeZ * (*mat)[2][1], sizeZ * (*mat)[2][2]);

    const float halfX = (XAbs(xVec.x) + XAbs(yVec.x) + XAbs(zVec.x)) * 0.5f;
    const float halfY = (XAbs(xVec.y) + XAbs(yVec.y) + XAbs(zVec.y)) * 0.5f;
    const float halfZ = (XAbs(xVec.z) + XAbs(yVec.z) + XAbs(zVec.z)) * 0.5f;

    dest->Max.x = dest->Min.x + halfX;
    dest->Max.y = dest->Min.y + halfY;
    dest->Max.z = dest->Min.z + halfZ;

    dest->Min.x = dest->Min.x - halfX;
    dest->Min.y = dest->Min.y - halfY;
    dest->Min.z = dest->Min.z - halfZ;
}

void VxSIMDFrustumUpdate_SSE(VxFrustum *frustum) {
    // Use the public Update() method which computes all derived quantities and planes
    frustum->Update();
}

void VxSIMDFrustumComputeVertices_SSE(const VxFrustum *frustum, VxVector *vertices) {
    // Get values through public accessors
    const VxVector &dir = frustum->GetDir();
    const VxVector &right = frustum->GetRight();
    const VxVector &up = frustum->GetUp();
    const VxVector &origin = frustum->GetOrigin();
    const float dMin = frustum->GetDMin();
    const float rBound = frustum->GetRBound();
    const float uBound = frustum->GetUBound();
    const float dRatio = frustum->GetDRatio();

    // Scale direction vectors by distance and bounds using SSE
    VxVector nearDirVec, rightVec, upVec;
    VxSIMDScaleVector_SSE(&nearDirVec, &dir, dMin);
    VxSIMDScaleVector_SSE(&rightVec, &right, rBound);
    VxSIMDScaleVector_SSE(&upVec, &up, uBound);

    // Compute near plane vertices relative to origin using SSE.
    // Must match scalar VxFrustum::ComputeVertices ordering exactly:
    // [0]=Near-Bottom-Left, [1]=Near-Top-Left, [2]=Near-Top-Right, [3]=Near-Bottom-Right
    VxVector leftVec;
    VxVector rightVec2;
    VxVector temp1;

    VxSIMDSubtractVector_SSE(&leftVec, &nearDirVec, &rightVec);  // nearDirVec - rightVec
    VxSIMDAddVector_SSE(&rightVec2, &nearDirVec, &rightVec);     // nearDirVec + rightVec

    // vertices[0] = leftVec - upVec
    VxSIMDSubtractVector_SSE(&vertices[0], &leftVec, &upVec);

    // vertices[1] = leftVec + upVec
    VxSIMDAddVector_SSE(&vertices[1], &leftVec, &upVec);

    // vertices[2] = rightVec2 + upVec
    VxSIMDAddVector_SSE(&vertices[2], &rightVec2, &upVec);

    // vertices[3] = rightVec2 - upVec
    VxSIMDSubtractVector_SSE(&vertices[3], &rightVec2, &upVec);

    // Compute far vertices and adjust near vertices to world space using SSE
    for (int i = 0; i < 4; i++) {
        // Scale near vertex by depth ratio to get far vector
        VxVector farVec;
        VxSIMDScaleVector_SSE(&farVec, &vertices[i], dRatio);

        // Far vertex is origin plus scaled vector
        VxSIMDAddVector_SSE(&vertices[i + 4], &origin, &farVec);

        // Adjust near vertex to be in world space
        VxSIMDAddVector_SSE(&temp1, &vertices[i], &origin);
        vertices[i] = temp1;
    }
}

void VxSIMDFrustumTransform_SSE(VxFrustum *frustum, const VxMatrix *invWorldMat) {
    // Get mutable references through public accessors
    VxVector &right = frustum->GetRight();
    VxVector &up = frustum->GetUp();
    VxVector &dir = frustum->GetDir();
    VxVector &origin = frustum->GetOrigin();
    float &rBound = frustum->GetRBound();
    float &uBound = frustum->GetUBound();
    float &dMin = frustum->GetDMin();
    float &dMax = frustum->GetDMax();
    const float dRatio = frustum->GetDRatio();

    // Scale direction vectors by their bounds before transformation using SSE
    VxSIMDScaleVector_SSE(&right, &right, rBound);
    VxSIMDScaleVector_SSE(&up, &up, uBound);
    VxSIMDScaleVector_SSE(&dir, &dir, dMin);

    // Transform the origin (full matrix with translation)
    VxVector newOrigin;
    VxSIMDMultiplyMatrixVector_SSE(&newOrigin, invWorldMat, &origin);
    origin = newOrigin;

    // Transform the scaled direction vectors (rotation only) using SSE
    VxVector resultVectors[3];
    VxSIMDRotateVectorOp_SSE(&resultVectors[0], invWorldMat, &right);
    VxSIMDRotateVectorOp_SSE(&resultVectors[1], invWorldMat, &up);
    VxSIMDRotateVectorOp_SSE(&resultVectors[2], invWorldMat, &dir);

    // Extract new magnitudes (bounds) using SSE
    float newRBound = VxSIMDLengthVector_SSE(&resultVectors[0]);
    float newUBound = VxSIMDLengthVector_SSE(&resultVectors[1]);
    float newDMin = VxSIMDLengthVector_SSE(&resultVectors[2]);

    // Update bounds and distances
    rBound = newRBound;
    uBound = newUBound;
    dMin = newDMin;
    dMax = newDMin * dRatio; // Preserve ratio

    // Normalize direction vectors using SSE.
    // Match scalar VxFrustum::Transform behavior: it divides unconditionally.
    VxSIMDScaleVector_SSE(&right, &resultVectors[0], 1.0f / newRBound);
    VxSIMDScaleVector_SSE(&up, &resultVectors[1], 1.0f / newUBound);
    VxSIMDScaleVector_SSE(&dir, &resultVectors[2], 1.0f / newDMin);

    // Update the planes with the new transformed vectors
    frustum->Update();
}

// Additional Vector operations
void VxSIMDAddVector_SSE(VxVector *result, const VxVector *a, const VxVector *b) {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = _mm_add_ps(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

void VxSIMDSubtractVector_SSE(VxVector *result, const VxVector *a, const VxVector *b) {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = _mm_sub_ps(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

void VxSIMDScaleVector_SSE(VxVector *result, const VxVector *v, float scalar) {
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 scaleVec = _mm_set1_ps(scalar);
    __m128 resultVec = _mm_mul_ps(vec, scaleVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

float VxSIMDDotVector_SSE(const VxVector *a, const VxVector *b) {
    // Match scalar evaluation order: (x*x + y*y) + z*z
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

void VxSIMDCrossVector_SSE(VxVector *result, const VxVector *a, const VxVector *b) {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = VxSIMDCrossProduct3(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

float VxSIMDLengthVector_SSE(const VxVector *v) {
    // Match scalar evaluation order as closely as possible:
    // sqrtf((x*x + y*y) + z*z)
    __m128 vec = VxSIMDLoadFloat3(&v->x);
    __m128 sq = _mm_mul_ps(vec, vec);

    __m128 y = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 sum = _mm_add_ss(sq, y); // x^2 + y^2

    __m128 z = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 2, 2, 2));
    sum = _mm_add_ss(sum, z); // (x^2 + y^2) + z^2

    float lengthSq;
    _mm_store_ss(&lengthSq, sum);
    return sqrtf(lengthSq);
}

float VxSIMDLengthSquaredVector_SSE(const VxVector *v) {
    // Match scalar evaluation order as closely as possible:
    // (x*x + y*y) + z*z
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

float VxSIMDDistanceVector_SSE(const VxVector *a, const VxVector *b) {
    VxVector diff;
    VxSIMDSubtractVector_SSE(&diff, a, b);
    return VxSIMDLengthVector_SSE(&diff);
}

void VxSIMDLerpVector_SSE(VxVector *result, const VxVector *a, const VxVector *b, float t) {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 tVec = _mm_set1_ps(t);
    __m128 oneMinusTVec = _mm_set1_ps(1.0f - t);
    __m128 resultVec = _mm_add_ps(_mm_mul_ps(aVec, oneMinusTVec), _mm_mul_ps(bVec, tVec));
    VxSIMDStoreFloat3(&result->x, resultVec);
}

void VxSIMDReflectVector_SSE(VxVector *result, const VxVector *incident, const VxVector *normal) {
    // r = i - 2.0 * (i·n) * n
    __m128 iVec = VxSIMDLoadFloat3(&incident->x);
    __m128 nVec = VxSIMDLoadFloat3(&normal->x);
    __m128 dotVec = VxSIMDDotProduct3(iVec, nVec);
    __m128 twoVec = _mm_set1_ps(2.0f);
    __m128 factor = _mm_mul_ps(twoVec, dotVec);
    __m128 reflection = _mm_sub_ps(iVec, _mm_mul_ps(factor, nVec));
    VxSIMDStoreFloat3(&result->x, reflection);
}

void VxSIMDMinimizeVector_SSE(VxVector *result, const VxVector *a, const VxVector *b) {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = _mm_min_ps(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

void VxSIMDMaximizeVector_SSE(VxVector *result, const VxVector *a, const VxVector *b) {
    __m128 aVec = VxSIMDLoadFloat3(&a->x);
    __m128 bVec = VxSIMDLoadFloat3(&b->x);
    __m128 resultVec = _mm_max_ps(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, resultVec);
}

// Vector4 operations
void VxSIMDAddVector4_SSE(VxVector4 *result, const VxVector4 *a, const VxVector4 *b) {
    __m128 aVec = VxSIMDLoadFloat4((const float*)a);
    __m128 bVec = VxSIMDLoadFloat4((const float*)b);
    __m128 resultVec = _mm_add_ps(aVec, bVec);
    VxSIMDStoreFloat4((float*)result, resultVec);
}

void VxSIMDSubtractVector4_SSE(VxVector4 *result, const VxVector4 *a, const VxVector4 *b) {
    __m128 aVec = VxSIMDLoadFloat4((const float*)a);
    __m128 bVec = VxSIMDLoadFloat4((const float*)b);
    __m128 resultVec = _mm_sub_ps(aVec, bVec);
    VxSIMDStoreFloat4((float*)result, resultVec);
}

void VxSIMDScaleVector4_SSE(VxVector4 *result, const VxVector4 *v, float scalar) {
    __m128 vec = VxSIMDLoadFloat4((const float*)v);
    __m128 scaleVec = _mm_set1_ps(scalar);
    __m128 resultVec = _mm_mul_ps(vec, scaleVec);
    VxSIMDStoreFloat4((float*)result, resultVec);
}

float VxSIMDDotVector4_SSE(const VxVector4 *a, const VxVector4 *b) {
    __m128 aVec = VxSIMDLoadFloat4((const float*)a);
    __m128 bVec = VxSIMDLoadFloat4((const float*)b);
    __m128 dotVec = VxSIMDDotProduct4(aVec, bVec);
    float result;
    _mm_store_ss(&result, dotVec);
    return result;
}

void VxSIMDLerpVector4_SSE(VxVector4 *result, const VxVector4 *a, const VxVector4 *b, float t) {
    __m128 aVec = VxSIMDLoadFloat4((const float*)a);
    __m128 bVec = VxSIMDLoadFloat4((const float*)b);
    __m128 tVec = _mm_set1_ps(t);
    __m128 oneMinusTVec = _mm_set1_ps(1.0f - t);
    __m128 resultVec = _mm_add_ps(_mm_mul_ps(aVec, oneMinusTVec), _mm_mul_ps(bVec, tVec));
    VxSIMDStoreFloat4((float*)result, resultVec);
}

// Matrix strided operations (basic implementations)
void VxSIMDMultiplyMatrixVectorStrided_SSE(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count) {
    // Simple implementation - process one by one
    for (int i = 0; i < count; ++i) {
        VxVector* vec = reinterpret_cast<VxVector*>(static_cast<char*>(src->Ptr) + i * src->Stride);
        VxVector* result = reinterpret_cast<VxVector*>(static_cast<char*>(dest->Ptr) + i * dest->Stride);
        VxSIMDMultiplyMatrixVector_SSE(result, mat, vec);
    }
}

void VxSIMDMultiplyMatrixVector4Strided_SSE(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count) {
    // Simple implementation - process one by one
    for (int i = 0; i < count; ++i) {
        VxVector4* vec = reinterpret_cast<VxVector4*>(static_cast<char*>(src->Ptr) + i * src->Stride);
        VxVector4* result = reinterpret_cast<VxVector4*>(static_cast<char*>(dest->Ptr) + i * dest->Stride);
        VxSIMDMultiplyMatrixVector4_SSE(result, mat, vec);
    }
}

void VxSIMDRotateVectorStrided_SSE(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count) {
    // Simple implementation - process one by one
    for (int i = 0; i < count; ++i) {
        VxVector* vec = reinterpret_cast<VxVector*>(static_cast<char*>(src->Ptr) + i * src->Stride);
        VxVector* result = reinterpret_cast<VxVector*>(static_cast<char*>(dest->Ptr) + i * dest->Stride);
        VxSIMDRotateVectorOp_SSE(result, mat, vec);
    }
}

// Matrix utility operations (basic implementations)
void VxSIMDMatrixIdentity_SSE(VxMatrix *mat) {
    __m128 zero = _mm_setzero_ps();
    __m128 row0 = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
    __m128 row1 = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
    __m128 row2 = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
    __m128 row3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);

    _mm_storeu_ps(&(*mat)[0][0], row0);
    _mm_storeu_ps(&(*mat)[1][0], row1);
    _mm_storeu_ps(&(*mat)[2][0], row2);
    _mm_storeu_ps(&(*mat)[3][0], row3);
}

void VxSIMDMatrixInverse_SSE(VxMatrix *result, const VxMatrix *mat) {
    // Load matrix rows
    __m128 row0 = _mm_loadu_ps(&(*mat)[0][0]);
    __m128 row1 = _mm_loadu_ps(&(*mat)[1][0]);
    __m128 row2 = _mm_loadu_ps(&(*mat)[2][0]);
    __m128 row3 = _mm_loadu_ps(&(*mat)[3][0]);

    // Extract 3x3 submatrix elements for determinant calculation
    const float a00 = (*mat)[0][0], a01 = (*mat)[0][1], a02 = (*mat)[0][2];
    const float a10 = (*mat)[1][0], a11 = (*mat)[1][1], a12 = (*mat)[1][2];
    const float a20 = (*mat)[2][0], a21 = (*mat)[2][1], a22 = (*mat)[2][2];

    // Calculate determinant
    const float minor1 = a11 * a22 - a12 * a21;
    const float minor2 = a12 * a20 - a10 * a22;
    const float minor3 = a10 * a21 - a11 * a20;
    const float det = a00 * minor1 + a01 * minor2 + a02 * minor3;

    if (fabsf(det) < EPSILON) {
        VxSIMDMatrixIdentity_SSE(result);
        return;
    }

    const float invDet = 1.0f / det;
    __m128 invDetVec = _mm_set1_ps(invDet);

    // Calculate cofactor matrix using SSE
    // Row 0: [minor1, (a02*a21 - a01*a22), (a01*a12 - a02*a11), 0]
    __m128 cofactor0 = _mm_setr_ps(
        minor1,
        a02 * a21 - a01 * a22,
        a01 * a12 - a02 * a11,
        0.0f
    );
    cofactor0 = _mm_mul_ps(cofactor0, invDetVec);

    // Row 1: [minor2, (a00*a22 - a02*a20), (a02*a10 - a00*a12), 0]
    __m128 cofactor1 = _mm_setr_ps(
        minor2,
        a00 * a22 - a02 * a20,
        a02 * a10 - a00 * a12,
        0.0f
    );
    cofactor1 = _mm_mul_ps(cofactor1, invDetVec);

    // Row 2: [minor3, (a01*a20 - a00*a21), (a00*a11 - a01*a10), 0]
    __m128 cofactor2 = _mm_setr_ps(
        minor3,
        a01 * a20 - a00 * a21,
        a00 * a11 - a01 * a10,
        0.0f
    );
    cofactor2 = _mm_mul_ps(cofactor2, invDetVec);

    // Store inverse rotation/scale part
    _mm_storeu_ps(&(*result)[0][0], cofactor0);
    _mm_storeu_ps(&(*result)[1][0], cofactor1);
    _mm_storeu_ps(&(*result)[2][0], cofactor2);

    // Calculate translation part matching scalar layout:
    // result[3][c] = -(result[0][c]*tx + result[1][c]*ty + result[2][c]*tz)
    const __m128 trans = _mm_setr_ps((*mat)[3][0], (*mat)[3][1], (*mat)[3][2], 0.0f);

    auto dot3 = [&](const __m128 a) -> float {
        const __m128 prod = _mm_mul_ps(a, trans);
        const __m128 shuf = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(2, 3, 0, 1));
        const __m128 sums = _mm_add_ps(prod, shuf);
        const __m128 hi = _mm_movehl_ps(sums, sums);
        const __m128 dot = _mm_add_ss(sums, hi);
        return _mm_cvtss_f32(dot);
    };

    const __m128 invC0 = _mm_setr_ps((*result)[0][0], (*result)[1][0], (*result)[2][0], 0.0f);
    const __m128 invC1 = _mm_setr_ps((*result)[0][1], (*result)[1][1], (*result)[2][1], 0.0f);
    const __m128 invC2 = _mm_setr_ps((*result)[0][2], (*result)[1][2], (*result)[2][2], 0.0f);

    const float t0 = -dot3(invC0);
    const float t1 = -dot3(invC1);
    const float t2 = -dot3(invC2);

    // Store translation row
    const __m128 row3_result = _mm_setr_ps(t0, t1, t2, 1.0f);
    _mm_storeu_ps(&(*result)[3][0], row3_result);
}

float VxSIMDMatrixDeterminant_SSE(const VxMatrix *mat) {
    // Use only 3x3 upper-left submatrix for 3D transformations
    const float a00 = (*mat)[0][0], a01 = (*mat)[0][1], a02 = (*mat)[0][2];
    const float a10 = (*mat)[1][0], a11 = (*mat)[1][1], a12 = (*mat)[1][2];
    const float a20 = (*mat)[2][0], a21 = (*mat)[2][1], a22 = (*mat)[2][2];

    const float minor1 = a11 * a22 - a12 * a21;
    const float minor2 = a12 * a20 - a10 * a22;
    const float minor3 = a10 * a21 - a11 * a20;

    return a00 * minor1 + a01 * minor2 + a02 * minor3;
}

void VxSIMDMatrixFromRotation_SSE(VxMatrix *result, const VxVector *axis, float angle) {
    // Fast path for zero rotation
    if (fabsf(angle) < EPSILON) {
        VxSIMDMatrixIdentity_SSE(result);
        return;
    }

    // Load and normalize axis vector using SSE
    __m128 axisVec = VxSIMDLoadFloat3(&axis->x);
    __m128 lenSqVec = VxSIMDDotProduct3(axisVec, axisVec);
    float lenSq;
    _mm_store_ss(&lenSq, lenSqVec);

    __m128 normalizedAxis;
    if (lenSq > EPSILON) {
        // Match scalar semantics: invLen = 1 / sqrt(lenSq)
        __m128 len = _mm_sqrt_ss(lenSqVec);
        __m128 invLen = _mm_div_ss(_mm_set_ss(1.0f), len);
        invLen = _mm_shuffle_ps(invLen, invLen, _MM_SHUFFLE(0, 0, 0, 0));
        normalizedAxis = _mm_mul_ps(axisVec, invLen);
    } else {
        // Default to Z-axis
        normalizedAxis = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
    }

    // Extract normalized components
    float x, y, z;
    VxSIMDStoreFloat3(&x, normalizedAxis);
    y = *(&x + 1);
    z = *(&x + 2);

    // Pre-calculate trigonometric values
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float t = 1.0f - c;

    // Use SSE to calculate common terms
    __m128 axisVec_broadcast = _mm_setr_ps(x, y, z, 0.0f);
    __m128 squared = _mm_mul_ps(axisVec_broadcast, axisVec_broadcast); // {x*x, y*y, z*z, 0}

    __m128 tVec = _mm_set1_ps(t);
    __m128 cVec = _mm_set1_ps(c);
    __m128 sVec = _mm_set1_ps(s);

    // Calculate diagonal elements: axis_squared * t + c
    __m128 diagonal = _mm_add_ps(_mm_mul_ps(squared, tVec), cVec);

    // Calculate off-diagonal terms
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float xs = x * s;
    const float ys = y * s;
    const float zs = z * s;
    const float xyt = xy * t;
    const float xzt = xz * t;
    const float yzt = yz * t;

    // Build matrix rows using SSE
    __m128 row0 = _mm_setr_ps(
        _mm_cvtss_f32(diagonal),
        xyt + zs,
        xzt - ys,
        0.0f
    );

    __m128 row1 = _mm_setr_ps(
        xyt - zs,
        _mm_cvtss_f32(_mm_shuffle_ps(diagonal, diagonal, _MM_SHUFFLE(1, 1, 1, 1))),
        yzt + xs,
        0.0f
    );

    __m128 row2 = _mm_setr_ps(
        xzt + ys,
        yzt - xs,
        _mm_cvtss_f32(_mm_shuffle_ps(diagonal, diagonal, _MM_SHUFFLE(2, 2, 2, 2))),
        0.0f
    );

    __m128 row3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);

    // Store result
    _mm_storeu_ps(&(*result)[0][0], row0);
    _mm_storeu_ps(&(*result)[1][0], row1);
    _mm_storeu_ps(&(*result)[2][0], row2);
    _mm_storeu_ps(&(*result)[3][0], row3);
}

void VxSIMDMatrixFromRotationOrigin_SSE(VxMatrix *result, const VxVector *axis, const VxVector *origin, float angle) {
    VxSIMDMatrixFromRotation_SSE(result, axis, angle);

    // Translation calculation: T = origin + R * (-origin) using SSE
    __m128 originVec = VxSIMDLoadFloat3(&origin->x);
    __m128 negOrigin = _mm_sub_ps(_mm_setzero_ps(), originVec);

    // Apply rotation to negative origin vector using SSE
    __m128 rotatedNegOrigin = VxSIMDMatrixRotateVector3((const float*)&(*result)[0][0], negOrigin);

    // Add back origin: T = origin + R * (-origin)
    __m128 translation = _mm_add_ps(originVec, rotatedNegOrigin);

    // Store final translation
    VxSIMDStoreFloat3(&(*result)[3][0], translation);
    (*result)[3][3] = 1.0f;
}

void VxSIMDMatrixFromEuler_SSE(VxMatrix *result, float eax, float eay, float eaz) {
    // Calculate trigonometric values
    float cx, sx, cy, sy, cz, sz;
    if (fabsf(eax) <= EPSILON) {
        cx = 1.0f;
        sx = eax;
    } else {
        cx = cosf(eax);
        sx = sinf(eax);
    }
    if (fabsf(eay) <= EPSILON) {
        cy = 1.0f;
        sy = eay;
    } else {
        cy = cosf(eay);
        sy = sinf(eay);
    }
    if (fabsf(eaz) <= EPSILON) {
        cz = 1.0f;
        sz = eaz;
    } else {
        cz = cosf(eaz);
        sz = sinf(eaz);
    }

    // Load trig values into SSE registers for parallel computation
    __m128 cosVec = _mm_setr_ps(cx, cy, cz, 0.0f);
    __m128 sinVec = _mm_setr_ps(sx, sy, sz, 0.0f);

    // Pre-calculate common terms using SSE
    __m128 sxVec = _mm_set1_ps(sx);
    __m128 cxVec = _mm_set1_ps(cx);
    __m128 syVec = _mm_set1_ps(sy);
    __m128 cyVec = _mm_set1_ps(cy);
    __m128 szVec = _mm_set1_ps(sz);
    __m128 czVec = _mm_set1_ps(cz);

    // Calculate sxsy and cxsy
    __m128 sxsy_vec = _mm_mul_ps(sxVec, syVec);
    __m128 cxsy_vec = _mm_mul_ps(cxVec, syVec);

    float sxsy = _mm_cvtss_f32(sxsy_vec);
    float cxsy = _mm_cvtss_f32(cxsy_vec);
    float cycz = cy * cz;
    float cysz = cy * sz;

    // Build matrix rows using SSE
    __m128 row0 = _mm_setr_ps(
        cycz,
        cysz,
        -sy,
        0.0f
    );

    __m128 row1 = _mm_setr_ps(
        sxsy * cz - cx * sz,
        sxsy * sz + cx * cz,
        sx * cy,
        0.0f
    );

    __m128 row2 = _mm_setr_ps(
        cxsy * cz + sx * sz,
        cxsy * sz - sx * cz,
        cx * cy,
        0.0f
    );

    __m128 row3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);

    // Store result
    _mm_storeu_ps(&(*result)[0][0], row0);
    _mm_storeu_ps(&(*result)[1][0], row1);
    _mm_storeu_ps(&(*result)[2][0], row2);
    _mm_storeu_ps(&(*result)[3][0], row3);
}

void VxSIMDMatrixToEuler_SSE(const VxMatrix *mat, float *eax, float *eay, float *eaz) {
    // Calculate magnitude for gimbal lock detection
    const float m00 = (*mat)[0][0];
    const float m01 = (*mat)[0][1];
    const float m02 = (*mat)[0][2];
    const float m12 = (*mat)[1][2];
    const float m22 = (*mat)[2][2];
    const float m21 = (*mat)[2][1];
    const float m11 = (*mat)[1][1];

    const float magnitude = sqrtf(m00 * m00 + m01 * m01);

    if (magnitude < EPSILON) {
        // Gimbal lock case
        *eay = atan2f(-m02, magnitude);
        *eax = atan2f(-m21, m11);
        *eaz = 0.0f;
    } else {
        // Normal case
        *eay = atan2f(-m02, magnitude);
        *eax = atan2f(m12, m22);
        *eaz = atan2f(m01, m00);
    }
}

void VxSIMDMatrixInterpolate_SSE(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b) {
    // Delegate to scalar for binary-accurate results
    Vx3DInterpolateMatrix(step, *result, *a, *b);
}

void VxSIMDMatrixInterpolateNoScale_SSE(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b) {
    // Delegate to scalar for binary-accurate results
    Vx3DInterpolateMatrixNoScale(step, *result, *a, *b);
}

void VxSIMDMatrixDecompose_SSE(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale) {
    if (!mat || !quat || !pos || !scale) return;
    // Delegate to scalar for binary-accurate results
    Vx3DDecomposeMatrix(*mat, *quat, *pos, *scale);
}

// Forward declarations for decomposition helpers implemented later in this file.
static float VxSIMDMatrixNorm3_SSE(const VxMatrix &M, bool isOneNorm);
static void VxSIMDMatrixAdjoint3_SSE(const VxMatrix &in, VxMatrix &out);
float VxSIMDMatrixPolarDecomposition_SSE(const VxMatrix &M_in, VxMatrix &Q, VxMatrix &S);
VxVector VxSIMDMatrixSpectralDecomposition_SSE(const VxMatrix &S_in, VxMatrix &U_out);

float VxSIMDMatrixDecomposeTotal_SSE(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot) {
    if (!mat) return 1.0f;
    // Delegate to scalar for binary-accurate results
    VxQuaternion q, u;
    VxVector p, s;
    float det = Vx3DDecomposeMatrixTotal(*mat, q, p, s, u);
    if (quat) *quat = q;
    if (pos) *pos = p;
    if (scale) *scale = s;
    if (uRot) *uRot = u;
    return det;
}

float VxSIMDMatrixDecomposeTotalPtr_SSE(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot) {
    if (!mat) return 1.0f;
    // Delegate to scalar for binary-accurate results
    return Vx3DDecomposeMatrixTotalPtr(*mat, quat, pos, scale, uRot);
}

// Quaternion additional operations
void VxSIMDQuaternionFromMatrix_SSE(VxQuaternion *result, const VxMatrix *mat, XBOOL matIsUnit, XBOOL restoreMat) {
    if (!result || !mat) return;

    float original[9];
    if (!matIsUnit && restoreMat) {
        original[0] = (*mat)[0][0];
        original[1] = (*mat)[0][1];
        original[2] = (*mat)[0][2];
        original[3] = (*mat)[1][0];
        original[4] = (*mat)[1][1];
        original[5] = (*mat)[1][2];
        original[6] = (*mat)[2][0];
        original[7] = (*mat)[2][1];
        original[8] = (*mat)[2][2];
    }

    if (!matIsUnit) {
        VxMatrix &nonConstMat = const_cast<VxMatrix &>(*mat);

        __m128 row0 = VxSIMDLoadFloat3(&nonConstMat[0][0]);
        __m128 row1 = VxSIMDLoadFloat3(&nonConstMat[1][0]);

        // Normalize row0
        {
            __m128 dot = VxSIMDDotProduct3(row0, row0);
            float magSq;
            _mm_store_ss(&magSq, dot);
            if (magSq > EPSILON) {
                __m128 mag = _mm_sqrt_ss(dot);
                __m128 invMag = _mm_div_ss(_mm_set_ss(1.0f), mag);
                __m128 invMag4 = _mm_shuffle_ps(invMag, invMag, _MM_SHUFFLE(0, 0, 0, 0));
                row0 = _mm_mul_ps(row0, invMag4);
            }
        }

        // Normalize row1
        {
            __m128 dot = VxSIMDDotProduct3(row1, row1);
            float magSq;
            _mm_store_ss(&magSq, dot);
            if (magSq > EPSILON) {
                __m128 mag = _mm_sqrt_ss(dot);
                __m128 invMag = _mm_div_ss(_mm_set_ss(1.0f), mag);
                __m128 invMag4 = _mm_shuffle_ps(invMag, invMag, _MM_SHUFFLE(0, 0, 0, 0));
                row1 = _mm_mul_ps(row1, invMag4);
            }
        }

        VxSIMDStoreFloat3(&nonConstMat[0][0], row0);
        VxSIMDStoreFloat3(&nonConstMat[1][0], row1);

        // row2 = cross(row0, row1)
        __m128 a_yzx = _mm_shuffle_ps(row0, row0, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 b_zxy = _mm_shuffle_ps(row1, row1, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 a_zxy = _mm_shuffle_ps(row0, row0, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 b_yzx = _mm_shuffle_ps(row1, row1, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 row2 = _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx));
        VxSIMDStoreFloat3(&nonConstMat[2][0], row2);
    }

    const float m00 = (*mat)[0][0];
    const float m11 = (*mat)[1][1];
    const float m22 = (*mat)[2][2];
    const float trace = m00 + m11 + m22;

    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f);
        result->w = s * 0.5f;
        s = 0.5f / s;
        result->x = ((*mat)[2][1] - (*mat)[1][2]) * s;
        result->y = ((*mat)[0][2] - (*mat)[2][0]) * s;
        result->z = ((*mat)[1][0] - (*mat)[0][1]) * s;
    } else {
        int i = 0;
        if ((*mat)[1][1] > (*mat)[0][0]) i = 1;
        if ((*mat)[2][2] > (*mat)[i][i]) i = 2;

        static const int next[3] = {1, 2, 0};
        int j = next[i];
        int k = next[j];

        float s = sqrtf((*mat)[i][i] - (*mat)[j][j] - (*mat)[k][k] + 1.0f);
        float *q[4] = {&result->x, &result->y, &result->z, &result->w};
        *q[i] = s * 0.5f;

        if (s > EPSILON) {
            s = 0.5f / s;
            *q[3] = ((*mat)[k][j] - (*mat)[j][k]) * s;
            *q[j] = ((*mat)[j][i] + (*mat)[i][j]) * s;
            *q[k] = ((*mat)[k][i] + (*mat)[i][k]) * s;
        } else {
            *q[3] = 1.0f;
            *q[j] = 0.0f;
            *q[k] = 0.0f;
        }
    }

    if (!matIsUnit && restoreMat) {
        VxMatrix &nonConstMat = const_cast<VxMatrix &>(*mat);
        nonConstMat[0][0] = original[0];
        nonConstMat[0][1] = original[1];
        nonConstMat[0][2] = original[2];
        nonConstMat[1][0] = original[3];
        nonConstMat[1][1] = original[4];
        nonConstMat[1][2] = original[5];
        nonConstMat[2][0] = original[6];
        nonConstMat[2][1] = original[7];
        nonConstMat[2][2] = original[8];
    }
}

void VxSIMDQuaternionToMatrix_SSE(VxMatrix *result, const VxQuaternion *q) {
    // Optimized SSE implementation based on original binary patterns
    // Uses efficient SIMD operations throughout to minimize scalar extractions

    // Load quaternion {x, y, z, w}
    __m128 quat = _mm_loadu_ps(&q->x);

    // Compute squared magnitude: x*x + y*y + z*z + w*w using SSE4.1 dpps if available
    __m128 qSq = _mm_mul_ps(quat, quat);
#if defined(VX_SIMD_SSE4_1)
    __m128 norm = _mm_dp_ps(quat, quat, 0xFF);
#else
    // Horizontal add for norm
    __m128 shuf = _mm_shuffle_ps(qSq, qSq, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(qSq, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    __m128 norm = _mm_add_ss(sums, shuf);
    norm = _mm_shuffle_ps(norm, norm, _MM_SHUFFLE(0, 0, 0, 0));
#endif

    // Check for degenerate quaternion
    float normScalar;
    _mm_store_ss(&normScalar, norm);
    if (normScalar < EPSILON) {
        VxSIMDMatrixIdentity_SSE(result);
        return;
    }

    // Compute s = 2.0 / norm using fast reciprocal with Newton-Raphson refinement
    __m128 two = _mm_set1_ps(2.0f);
    __m128 rcp = _mm_rcp_ps(norm);
    // Newton-Raphson iteration: rcp' = rcp * (2 - norm * rcp)
    rcp = _mm_mul_ps(rcp, _mm_sub_ps(two, _mm_mul_ps(norm, rcp)));
    __m128 s = _mm_mul_ps(two, rcp); // s = 2.0 / norm

    // Compute scaled quaternion components
    __m128 qs = _mm_mul_ps(quat, s);

    // Extract scalar values for matrix construction
    alignas(16) float qf[4], qsf[4];
    _mm_store_ps(qf, quat);
    _mm_store_ps(qsf, qs);

    const float x = qf[0], y = qf[1], z = qf[2], w = qf[3];
    const float xs = qsf[0], ys = qsf[1], zs = qsf[2];

    // Compute products
    const float xx = x * xs;
    const float xy = x * ys;
    const float xz = x * zs;
    const float yy = y * ys;
    const float yz = y * zs;
    const float zz = z * zs;
    const float wx = w * xs;
    const float wy = w * ys;
    const float wz = w * zs;

    // Build matrix rows using SSE stores
    __m128 row0 = _mm_setr_ps(1.0f - (yy + zz), xy - wz, xz + wy, 0.0f);
    __m128 row1 = _mm_setr_ps(xy + wz, 1.0f - (xx + zz), yz - wx, 0.0f);
    __m128 row2 = _mm_setr_ps(xz - wy, yz + wx, 1.0f - (xx + yy), 0.0f);
    __m128 row3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);

    // Store result
    _mm_storeu_ps(&(*result)[0][0], row0);
    _mm_storeu_ps(&(*result)[1][0], row1);
    _mm_storeu_ps(&(*result)[2][0], row2);
    _mm_storeu_ps(&(*result)[3][0], row3);
}

void VxSIMDQuaternionFromRotation_SSE(VxQuaternion *result, const VxVector *axis, float angle) {
    // Create matrix from rotation and convert to quaternion
    // This approach uses our optimized SSE matrix functions
    VxMatrix mat;
    VxSIMDMatrixFromRotation_SSE(&mat, axis, angle);
    VxSIMDQuaternionFromMatrix_SSE(result, &mat, TRUE, TRUE);
}

void VxSIMDQuaternionFromEuler_SSE(VxQuaternion *result, float eax, float eay, float eaz) {
    // Create matrix from Euler angles and convert to quaternion
    // This approach uses our optimized SSE matrix functions
    VxMatrix mat;
    VxSIMDMatrixFromEuler_SSE(&mat, eax, eay, eaz);
    VxSIMDQuaternionFromMatrix_SSE(result, &mat, TRUE, TRUE);
}

void VxSIMDQuaternionToEuler_SSE(const VxQuaternion *q, float *eax, float *eay, float *eaz) {
    // Convert quaternion to matrix and extract Euler angles
    // This approach uses our optimized SSE matrix functions
    VxMatrix mat;
    VxSIMDQuaternionToMatrix_SSE(&mat, q);
    VxSIMDMatrixToEuler_SSE(&mat, eax, eay, eaz);
}

void VxSIMDQuaternionMultiplyInPlace_SSE(VxQuaternion *self, const VxQuaternion *rhs) {
    VxQuaternion temp;
    VxSIMDMultiplyQuaternion_SSE(&temp, self, rhs);
    *self = temp;
}

void VxSIMDQuaternionConjugate_SSE(VxQuaternion *result, const VxQuaternion *q) {
    __m128 qVec = _mm_loadu_ps(&q->x);
    __m128 signFlip = _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f);
    __m128 resultVec = _mm_mul_ps(qVec, signFlip);
    _mm_storeu_ps(&result->x, resultVec);
}

void VxSIMDQuaternionDivide_SSE(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q) {
    if (!result || !p || !q) return;

    // Division: P / Q = P * conjugate(Q) (for unit quaternions)
    // conjugate(Q) = (-Q.x, -Q.y, -Q.z, Q.w)
    // Using SSE for parallel computation
    const __m128 pV = _mm_loadu_ps(&p->x); // x y z w
    const __m128 qV = _mm_loadu_ps(&q->x); // x y z w

    // Negate q's xyz components to get conjugate
    const __m128 negMask = _mm_set_ps(0.0f, -0.0f, -0.0f, -0.0f);
    const __m128 qConj = _mm_xor_ps(qV, negMask); // -x, -y, -z, w

    // Now multiply P * conjugate(Q) using standard quaternion multiplication
    // result.x = p.w * qc.x + p.x * qc.w + p.y * qc.z - p.z * qc.y
    // result.y = p.w * qc.y - p.x * qc.z + p.y * qc.w + p.z * qc.x
    // result.z = p.w * qc.z + p.x * qc.y - p.y * qc.x + p.z * qc.w
    // result.w = p.w * qc.w - p.x * qc.x - p.y * qc.y - p.z * qc.z

    __m128 pW = _mm_shuffle_ps(pV, pV, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 t0 = _mm_mul_ps(pW, qConj);

    __m128 pX = _mm_shuffle_ps(pV, pV, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 qc_perm1 = _mm_shuffle_ps(qConj, qConj, _MM_SHUFFLE(0, 1, 2, 3)); // w, z, y, x
    __m128 t1 = _mm_mul_ps(pX, qc_perm1);

    __m128 pY = _mm_shuffle_ps(pV, pV, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 qc_perm2 = _mm_shuffle_ps(qConj, qConj, _MM_SHUFFLE(1, 0, 3, 2)); // y, x, w, z
    __m128 t2 = _mm_mul_ps(pY, qc_perm2);

    __m128 pZ = _mm_shuffle_ps(pV, pV, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 qc_perm3 = _mm_shuffle_ps(qConj, qConj, _MM_SHUFFLE(2, 3, 0, 1)); // z, w, x, y
    __m128 t3 = _mm_mul_ps(pZ, qc_perm3);

    // Apply signs: t0(+,+,+,+), t1(+,-,+,-), t2(+,+,-,-), t3(-,+,+,-)
    __m128 sign1 = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    t1 = _mm_mul_ps(t1, sign1);
    __m128 sign2 = _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f);
    t2 = _mm_mul_ps(t2, sign2);
    __m128 sign3 = _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f);
    t3 = _mm_mul_ps(t3, sign3);

    __m128 r = _mm_add_ps(_mm_add_ps(t0, t1), _mm_add_ps(t2, t3));

    _mm_storeu_ps(&result->x, r);
}

void VxSIMDQuaternionSnuggle_SSE(VxQuaternion *result, VxQuaternion *quat, VxVector *scale) {
    if (!result || !quat || !scale) {
        if (result) {
            result->x = 0.0f;
            result->y = 0.0f;
            result->z = 0.0f;
            result->w = 1.0f;
        }
        return;
    }
    // Delegate to scalar for binary-accurate results
    *result = Vx3DQuaternionSnuggle(quat, scale);
}

// -----------------------------------------------------------------------------
// SSE helpers matching scalar decomposition algorithms (3x3 focus)
// -----------------------------------------------------------------------------

static float VxSIMDMatrixNorm3_SSE(const VxMatrix &M, bool isOneNorm) {
    float maxNorm = 0.0f;
    for (int j = 0; j < 3; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < 3; ++i) {
            sum += fabsf(isOneNorm ? M[i][j] : M[j][i]);
        }
        if (sum > maxNorm) maxNorm = sum;
    }
    return maxNorm;
}

static void VxSIMDMatrixAdjoint3_SSE(const VxMatrix &in, VxMatrix &out) {
    const float m00 = in[0][0], m01 = in[0][1], m02 = in[0][2];
    const float m10 = in[1][0], m11 = in[1][1], m12 = in[1][2];
    const float m20 = in[2][0], m21 = in[2][1], m22 = in[2][2];

    out[0][0] = m11 * m22 - m12 * m21;
    out[1][0] = m12 * m20 - m10 * m22;
    out[2][0] = m10 * m21 - m11 * m20;

    out[0][1] = m02 * m21 - m01 * m22;
    out[1][1] = m00 * m22 - m02 * m20;
    out[2][1] = m01 * m20 - m00 * m21;

    out[0][2] = m01 * m12 - m02 * m11;
    out[1][2] = m02 * m10 - m00 * m12;
    out[2][2] = m00 * m11 - m01 * m10;
}

float VxSIMDMatrixPolarDecomposition_SSE(const VxMatrix &M_in, VxMatrix &Q, VxMatrix &S) {
    VxMatrix E;
    VxSIMDTransposeMatrix_SSE(&E, &M_in);
    E[0][3] = E[1][3] = E[2][3] = E[3][0] = E[3][1] = E[3][2] = 0.0f;
    E[3][3] = 1.0f;

    float E_one_norm = VxSIMDMatrixNorm3_SSE(E, true);
    float E_inf_norm = VxSIMDMatrixNorm3_SSE(E, false);

    for (int iter = 0; iter < 20; ++iter) {
        VxMatrix E_adj;
        VxSIMDMatrixAdjoint3_SSE(E, E_adj);

        // det_E = dot(E.row0, E_adj.row0)
        __m128 e0 = VxSIMDLoadFloat3(&E[0][0]);
        __m128 a0 = VxSIMDLoadFloat3(&E_adj[0][0]);
        __m128 detV = VxSIMDDotProduct3(e0, a0);
        float det_E;
        _mm_store_ss(&det_E, detV);
        if (fabsf(det_E) < 1e-12f) break;

        float E_adj_one_norm = VxSIMDMatrixNorm3_SSE(E_adj, true);
        float E_adj_inf_norm = VxSIMDMatrixNorm3_SSE(E_adj, false);

        float gamma = sqrtf(sqrtf((E_adj_one_norm * E_adj_inf_norm) / (E_one_norm * E_inf_norm)) / fabsf(det_E));
        float c1 = 0.5f * gamma;
        float c2 = 0.5f / (gamma * det_E);

        VxMatrix E_next = E;
        for (int i = 0; i < 3; ++i) {
            __m128 eRow = VxSIMDLoadFloat3(&E[i][0]);

            // Build column i of E_adj (since scalar uses E_adj[j][i])
            __m128 col = _mm_setr_ps(E_adj[0][i], E_adj[1][i], E_adj[2][i], 0.0f);

            __m128 v = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c1), eRow), _mm_mul_ps(_mm_set1_ps(c2), col));
            VxSIMDStoreFloat3(&E_next[i][0], v);
        }

        VxMatrix E_diff = E;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                E_diff[i][j] = E_next[i][j] - E[i][j];

        E = E_next;
        if (VxSIMDMatrixNorm3_SSE(E_diff, true) < E_one_norm * 1e-6f) break;

        E_one_norm = VxSIMDMatrixNorm3_SSE(E, true);
        E_inf_norm = VxSIMDMatrixNorm3_SSE(E, false);
    }

    VxSIMDTransposeMatrix_SSE(&Q, &E);
    VxMatrix Q_T;
    VxSIMDTransposeMatrix_SSE(&Q_T, &Q);
    VxSIMDMultiplyMatrix_SSE(&S, &Q_T, &M_in);

    // Symmetrize S
    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j) {
            float val = 0.5f * (S[i][j] + S[j][i]);
            S[i][j] = val;
            S[j][i] = val;
        }

    for (int i = 0; i < 3; ++i) {
        Q[i][3] = Q[3][i] = 0.0f;
        S[i][3] = S[3][i] = 0.0f;
    }
    Q[3][3] = 1.0f;
    S[3][3] = 1.0f;

    return VxSIMDMatrixDeterminant_SSE(&Q);
}

VxVector VxSIMDMatrixSpectralDecomposition_SSE(const VxMatrix &S_in, VxMatrix &U_out) {
    U_out.SetIdentity();

    float d[3] = {S_in[0][0], S_in[1][1], S_in[2][2]};
    float o[3] = {S_in[0][1], S_in[0][2], S_in[1][2]};

    for (int sweep = 0; sweep < 20; ++sweep) {
        float sum_off_diag = fabsf(o[0]) + fabsf(o[1]) + fabsf(o[2]);
        if (sum_off_diag < 1e-9f) break;

        const int p_map[3] = {0, 0, 1};
        const int q_map[3] = {1, 2, 2};
        for (int idx = 0; idx < 3; ++idx) {
            int p = p_map[idx];
            int q = q_map[idx];

            float S_pq = (p == 0 && q == 1) ? o[0] : ((p == 0 && q == 2) ? o[1] : o[2]);
            if (fabsf(S_pq) < 1e-9f) continue;

            float diff = d[q] - d[p];
            float t;
            if (fabsf(diff) + fabsf(S_pq) * 100.0f == fabsf(diff)) {
                t = S_pq / diff;
            } else {
                float theta = diff / (2.0f * S_pq);
                t = 1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
                if (theta < 0.0f) t = -t;
            }
            float c = 1.0f / sqrtf(1.0f + t * t);
            float s = t * c;
            float tau = s / (1.0f + c);
            float h = t * S_pq;

            d[p] -= h;
            d[q] += h;

            int r = 3 - p - q;
            float S_pr = (p == 0 && r == 1) ? o[0] : ((p == 0 && r == 2) ? o[1] : o[2]);
            if (p > r) S_pr = (r == 0 && p == 1) ? o[0] : ((r == 0 && p == 2) ? o[1] : o[2]);
            float S_qr = (q == 0 && r == 1) ? o[0] : ((q == 0 && r == 2) ? o[1] : o[2]);
            if (q > r) S_qr = (r == 0 && q == 1) ? o[0] : ((r == 0 && q == 2) ? o[1] : o[2]);

            float next_S_pr = S_pr - s * (S_qr + S_pr * tau);
            float next_S_qr = S_qr + s * (S_pr - S_qr * tau);
            if (p == 0 && r == 1) o[0] = next_S_pr;
            else if (p == 0 && r == 2) o[1] = next_S_pr;
            else o[2] = next_S_pr;
            if (q == 0 && r == 1) o[0] = next_S_qr;
            else if (q == 0 && r == 2) o[1] = next_S_qr;
            else o[2] = next_S_qr;
            if (idx == 0) o[0] = 0;
            if (idx == 1) o[1] = 0;
            if (idx == 2) o[2] = 0;

            for (int k = 0; k < 3; ++k) {
                float g = U_out[k][p];
                float h_u = U_out[k][q];
                U_out[k][p] = g - s * (h_u + g * tau);
                U_out[k][q] = h_u + s * (g - h_u * tau);
            }
        }
    }

    return VxVector(d[0], d[1], d[2]);
}

void VxSIMDQuaternionLn_SSE(VxQuaternion *result, const VxQuaternion *q) {
    // Quaternion natural logarithm using SSE
    __m128 qVec = _mm_loadu_ps(&q->x);

    // Compute magnitude of vector part (x, y, z)
    __m128 vecPart = _mm_and_ps(qVec, _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0)));
    __m128 magSqVec = VxSIMDDotProduct3(vecPart, vecPart);
    float magSq;
    _mm_store_ss(&magSq, magSqVec);
    float magnitude = sqrtf(magSq);

    float scale;
    if (magnitude == 0.0f) {
        scale = 0.0f;
    } else {
        scale = atan2f(magnitude, q->w) / magnitude;
    }

    // Result = {scale * x, scale * y, scale * z, 0}
    __m128 scaleVec = _mm_set1_ps(scale);
    __m128 resultVec = _mm_mul_ps(vecPart, scaleVec);
    resultVec = _mm_and_ps(resultVec, _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0)));

    _mm_storeu_ps(&result->x, resultVec);
}

void VxSIMDQuaternionExp_SSE(VxQuaternion *result, const VxQuaternion *q) {
    // Quaternion exponential using SSE
    __m128 qVec = _mm_loadu_ps(&q->x);

    // Compute magnitude of vector part (x, y, z)
    __m128 vecPart = _mm_and_ps(qVec, _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0)));
    __m128 magSqVec = VxSIMDDotProduct3(vecPart, vecPart);
    float magSq;
    _mm_store_ss(&magSq, magSqVec);
    float magnitude = sqrtf(magSq);

    float scale;
    if (magnitude < EPSILON) {
        scale = 1.0f;
    } else {
        scale = sinf(magnitude) / magnitude;
    }

    // Result = {scale * x, scale * y, scale * z, cos(magnitude)}
    __m128 scaleVec = _mm_set1_ps(scale);
    __m128 resultVec = _mm_mul_ps(vecPart, scaleVec);

    // Set w component (using SSE2 compatible method)
    float w = cosf(magnitude);
    __m128 wVec = _mm_set_ss(w);
    wVec = _mm_shuffle_ps(wVec, wVec, _MM_SHUFFLE(0, 0, 0, 0));
    // Mask: keep xyz from resultVec, w from wVec
    __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0));
    resultVec = _mm_or_ps(_mm_and_ps(resultVec, mask), _mm_andnot_ps(mask, wVec));

    _mm_storeu_ps(&result->x, resultVec);
}

void VxSIMDQuaternionLnDif_SSE(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q) {
    // LnDif = Ln(Q / P) using SSE
    VxQuaternion div;
    VxSIMDQuaternionDivide_SSE(&div, q, p);
    VxSIMDQuaternionLn_SSE(result, &div);
}

void VxSIMDQuaternionSquad_SSE(VxQuaternion *result, float t, const VxQuaternion *quat1, const VxQuaternion *quat1Out, const VxQuaternion *quat2In, const VxQuaternion *quat2) {
    // Call the global quaternion squad function
    *result = Squad(t, *quat1, *quat1Out, *quat2In, *quat2);
}

// Ray operations (basic implementations)
void VxSIMDRayTransform_SSE(VxRay *dest, const VxRay *ray, const VxMatrix *mat) {
    VxSIMDMultiplyMatrixVector_SSE(&dest->m_Origin, mat, &ray->m_Origin);
    VxSIMDRotateVectorOp_SSE(&dest->m_Direction, mat, &ray->m_Direction);
}

// Plane operations (basic implementations)
void VxSIMDPlaneCreateFromPoint_SSE(VxPlane *plane, const VxVector *normal, const VxVector *point) {
    // Match VxPlane::Create(n, p): normalize normal if magSq > EPSILON
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

    // D = -(n · p)
    const __m128 p = VxSIMDLoadFloat3(&point->x);
    const __m128 dp = _mm_mul_ps(VxSIMDLoadFloat3(&plane->m_Normal.x), p);
    __m128 dsum = _mm_add_ss(dp, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(1, 1, 1, 1)));
    dsum = _mm_add_ss(dsum, _mm_shuffle_ps(dp, dp, _MM_SHUFFLE(2, 2, 2, 2)));
    float d;
    _mm_store_ss(&d, dsum);
    plane->m_D = -d;
}

void VxSIMDPlaneCreateFromTriangle_SSE(VxPlane *plane, const VxVector *a, const VxVector *b, const VxVector *c) {
    // Match VxPlane::Create(a,b,c): normalize cross, with degenerate fallback to Z-up
    VxVector edge1, edge2, n;
    VxSIMDSubtractVector_SSE(&edge1, b, a);
    VxSIMDSubtractVector_SSE(&edge2, c, a);
    VxSIMDCrossVector_SSE(&n, &edge1, &edge2);

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

    VxSIMDPlaneCreateFromPoint_SSE(plane, &n, a);
}

// Rect operations with SSE
void VxSIMDRectTransform_SSE(VxRect *rect, const VxRect *destScreen, const VxRect *srcScreen) {
    // Match VxRect::Transform(destScreen, srcScreen)
    // Note: normalization uses per-edge offsets (left-top-right-bottom) and width/height scaling.
    const __m128 r = _mm_loadu_ps(&rect->left);      // {l,t,r,b}
    const __m128 s = _mm_loadu_ps(&srcScreen->left); // {l,t,r,b}
    const __m128 d = _mm_loadu_ps(&destScreen->left);

    // src sizes replicated: {w, h, w, h}
    const __m128 s_right = _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 s_left  = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 s_bottom = _mm_shuffle_ps(s, s, _MM_SHUFFLE(3, 3, 3, 3));
    const __m128 s_top = _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 w = _mm_sub_ps(s_right, s_left);
    const __m128 h = _mm_sub_ps(s_bottom, s_top);
    const __m128 srcSize = _mm_unpacklo_ps(w, h);
    const __m128 srcInvSize = _mm_div_ps(_mm_set1_ps(1.0f), srcSize);

    // normalized = (rect - srcEdges) * invSize, where srcEdges is {src.left,src.top,src.right,src.bottom}
    const __m128 normalized = _mm_mul_ps(_mm_sub_ps(r, s), srcInvSize);

    // dest sizes replicated: {w, h, w, h}
    const __m128 d_right = _mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 d_left  = _mm_shuffle_ps(d, d, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 d_bottom = _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 3, 3, 3));
    const __m128 d_top = _mm_shuffle_ps(d, d, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 dw = _mm_sub_ps(d_right, d_left);
    const __m128 dh = _mm_sub_ps(d_bottom, d_top);
    const __m128 destSize = _mm_unpacklo_ps(dw, dh);

    // result = normalized * destSize + destEdges ({dest.left, dest.top, dest.right, dest.bottom})
    const __m128 result = _mm_add_ps(_mm_mul_ps(normalized, destSize), d);
    _mm_storeu_ps(&rect->left, result);
}

void VxSIMDRectTransformBySize_SSE(VxRect *rect, const Vx2DVector *destScreenSize, const Vx2DVector *srcScreenSize) {
    // Load rect
    __m128 rectVec = _mm_loadu_ps(&rect->left);

    // Calculate scale factors: dest/src for both x and y, replicated for left/right and top/bottom
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

    // Apply transformation: rect * (destSize / srcSize)
    __m128 result = _mm_mul_ps(_mm_mul_ps(rectVec, srcInvSize), destSize);

    // Store result
    _mm_storeu_ps(&rect->left, result);
}

void VxSIMDRectTransformToHomogeneous_SSE(VxRect *rect, const VxRect *screen) {
    // Save original width and height
    float width = rect->right - rect->left;
    float height = rect->bottom - rect->top;

    // Load rectangles
    __m128 rectVec = _mm_loadu_ps(&rect->left);
    __m128 screenVec = _mm_loadu_ps(&screen->left);

    // Calculate screen dimensions
    __m128 screenRightBottom = _mm_shuffle_ps(screenVec, screenVec, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 screenSize = _mm_sub_ps(screenRightBottom, screenVec);
    __m128 screenInvSize = _mm_div_ps(_mm_set1_ps(1.0f), screenSize);

    // Transform min corner: (rect.min - screen.min) * screenInvSize
    __m128 minTransformed = _mm_mul_ps(_mm_sub_ps(rectVec, screenVec), screenInvSize);

    // Extract normalized min
    float leftNorm, topNorm;
    _mm_store_ss(&leftNorm, minTransformed);
    _mm_store_ss(&topNorm, _mm_shuffle_ps(minTransformed, minTransformed, _MM_SHUFFLE(1, 1, 1, 1)));

    // Get screen width and height for transformation
    float screenInvWidth, screenInvHeight;
    _mm_store_ss(&screenInvWidth, screenInvSize);
    _mm_store_ss(&screenInvHeight, _mm_shuffle_ps(screenInvSize, screenInvSize, _MM_SHUFFLE(1, 1, 1, 1)));

    // Update rect
    rect->left = leftNorm;
    rect->top = topNorm;
    rect->right = leftNorm + width * screenInvWidth;
    rect->bottom = topNorm + height * screenInvHeight;
}

void VxSIMDRectTransformFromHomogeneous_SSE(VxRect *rect, const VxRect *screen) {
    // Save original width and height
    float width = rect->right - rect->left;
    float height = rect->bottom - rect->top;

    // Load rectangles
    __m128 rectVec = _mm_loadu_ps(&rect->left);
    __m128 screenVec = _mm_loadu_ps(&screen->left);

    // Calculate screen dimensions
    __m128 screenRightBottom = _mm_shuffle_ps(screenVec, screenVec, _MM_SHUFFLE(3, 2, 3, 2));
    __m128 screenSize = _mm_sub_ps(screenRightBottom, screenVec);

    // Transform min corner: screen.min + rect.min * screenSize
    __m128 minTransformed = _mm_add_ps(screenVec, _mm_mul_ps(rectVec, screenSize));

    // Extract transformed min
    float leftTrans, topTrans;
    _mm_store_ss(&leftTrans, minTransformed);
    _mm_store_ss(&topTrans, _mm_shuffle_ps(minTransformed, minTransformed, _MM_SHUFFLE(1, 1, 1, 1)));

    // Get screen width and height for transformation
    float screenWidth, screenHeight;
    _mm_store_ss(&screenWidth, screenSize);
    _mm_store_ss(&screenHeight, _mm_shuffle_ps(screenSize, screenSize, _MM_SHUFFLE(1, 1, 1, 1)));

    // Update rect
    rect->left = leftTrans;
    rect->top = topTrans;
    rect->right = leftTrans + width * screenWidth;
    rect->bottom = topTrans + height * screenHeight;
}

#endif
