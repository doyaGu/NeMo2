#include "VxSIMD.h"

#include "VxVector.h"
#include "VxMatrix.h"
#include "VxQuaternion.h"

#if defined(VX_SIMD_X86)

/**
 * @brief AVX+FMA Fast Path Variant
 *
 * @remarks
 * This file contains AVX and FMA-optimized implementations for CPUs
 * that support these instruction sets. FMA (Fused Multiply-Add) provides
 * better performance and accuracy for matrix operations.
 *
 * Important: We issue vzeroupper before returning to avoid AVX-SSE
 * transition penalties when mixing 256-bit and 128-bit code.
 */

void VxSIMDNormalizeVector_AVX(VxVector *v) {
    // Match VxVector::Normalize() semantics (sqrt-based, epsilon guard)
    __m128 vec = VxSIMDLoadFloat3(&v->x);

    const __m128 mul = _mm_mul_ps(vec, vec);
    __m128 sum = _mm_add_ss(mul, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 2, 2, 2)));

    float magSqScalar;
    _mm_store_ss(&magSqScalar, sum);
    if (magSqScalar > EPSILON) {
        const __m128 mag = _mm_sqrt_ss(sum);
        const __m128 invMag = _mm_div_ss(_mm_set_ss(1.0f), mag);
        const __m128 invMag4 = _mm_shuffle_ps(invMag, invMag, _MM_SHUFFLE(0, 0, 0, 0));
        vec = _mm_mul_ps(vec, invMag4);
    }

    VxSIMDStoreFloat3(&v->x, vec);
    _mm256_zeroupper();
}

void VxSIMDRotateVector_AVX(VxVector *result, const VxMatrix *mat, const VxVector *v) {
    // Load vector (use 4 elements for alignment)
    __m128 vec = VxSIMDLoadFloat3(&v->x);

    // Load matrix rows (3x3 rotation part)
    __m128 m0 = _mm_loadu_ps(&(*mat)[0][0]);
    __m128 m1 = _mm_loadu_ps(&(*mat)[1][0]);
    __m128 m2 = _mm_loadu_ps(&(*mat)[2][0]);

    // Broadcast vector components
    __m128 v_x = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));

    // Use FMA for better performance: result = m0 * v_x + m1 * v_y + m2 * v_z
    __m128 res = _mm_mul_ps(m0, v_x);
    res = _mm_fmadd_ps(m1, v_y, res);  // res += m1 * v_y
    res = _mm_fmadd_ps(m2, v_z, res);  // res += m2 * v_z

    VxSIMDStoreFloat3(&result->x, res);
    _mm256_zeroupper();
}

void VxSIMDMultiplyMatrix_AVX(VxMatrix *result, const VxMatrix *a, const VxMatrix *b) {
    // Avoid strict-aliasing UB from VxMatrix::operator[] reinterpret casts.
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

        __m128 res = _mm_mul_ps(a0, b_x);
        res = _mm_fmadd_ps(a1, b_y, res);
        res = _mm_fmadd_ps(a2, b_z, res);
        res = _mm_fmadd_ps(a3, b_w, res);

        _mm_storeu_ps(out + i * 4, res);
    }

    out[0 * 4 + 3] = 0.0f;
    out[1 * 4 + 3] = 0.0f;
    out[2 * 4 + 3] = 0.0f;
    out[3 * 4 + 3] = 1.0f;

    memcpy(result, out, sizeof(out));

    // Issue vzeroupper to avoid AVX-SSE transition penalty
    _mm256_zeroupper();
}

void VxSIMDMultiplyMatrix4_AVX(VxMatrix *result, const VxMatrix *a, const VxMatrix *b) {
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

        __m128 res = _mm_mul_ps(a0, b_x);
        res = _mm_fmadd_ps(a1, b_y, res);
        res = _mm_fmadd_ps(a2, b_z, res);
        res = _mm_fmadd_ps(a3, b_w, res);

        _mm_storeu_ps(out + i * 4, res);
    }

    memcpy(result, out, sizeof(out));
    _mm256_zeroupper();
}

void VxSIMDTransposeMatrix_AVX(VxMatrix *result, const VxMatrix *a) {
    // Load rows
    __m128 r0 = _mm_loadu_ps(&(*a)[0][0]);
    __m128 r1 = _mm_loadu_ps(&(*a)[1][0]);
    __m128 r2 = _mm_loadu_ps(&(*a)[2][0]);
    __m128 r3 = _mm_loadu_ps(&(*a)[3][0]);

    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

    VxMatrix temp;
    _mm_storeu_ps(&temp[0][0], r0);
    _mm_storeu_ps(&temp[1][0], r1);
    _mm_storeu_ps(&temp[2][0], r2);
    _mm_storeu_ps(&temp[3][0], r3);

    *result = temp;
    _mm256_zeroupper();
}

void VxSIMDMultiplyMatrixVector_AVX(VxVector *result, const VxMatrix *mat, const VxVector *v) {
    __m128 vec = VxSIMDLoadFloat3(&v->x);

    __m128 m0 = _mm_loadu_ps(&(*mat)[0][0]);
    __m128 m1 = _mm_loadu_ps(&(*mat)[1][0]);
    __m128 m2 = _mm_loadu_ps(&(*mat)[2][0]);
    __m128 m3 = _mm_loadu_ps(&(*mat)[3][0]);

    __m128 v_x = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));

    // Use FMA
    __m128 res = _mm_mul_ps(m0, v_x);
    res = _mm_fmadd_ps(m1, v_y, res);
    res = _mm_fmadd_ps(m2, v_z, res);
    res = _mm_add_ps(res, m3);  // Add translation

    VxSIMDStoreFloat3(&result->x, res);
    _mm256_zeroupper();
}

void VxSIMDMultiplyMatrixVector4_AVX(VxVector4 *result, const VxMatrix *mat, const VxVector4 *v) {
    __m128 vec = _mm_loadu_ps((const float*)v);

    __m128 m0 = _mm_loadu_ps(&(*mat)[0][0]);
    __m128 m1 = _mm_loadu_ps(&(*mat)[1][0]);
    __m128 m2 = _mm_loadu_ps(&(*mat)[2][0]);
    __m128 m3 = _mm_loadu_ps(&(*mat)[3][0]);

    __m128 v_x = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v_w = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(3, 3, 3, 3));

    // Use FMA
    __m128 res = _mm_mul_ps(m0, v_x);
    res = _mm_fmadd_ps(m1, v_y, res);
    res = _mm_fmadd_ps(m2, v_z, res);
    res = _mm_fmadd_ps(m3, v_w, res);

    _mm_storeu_ps((float*)result, res);
    _mm256_zeroupper();
}

void VxSIMDRotateVectorOp_AVX(VxVector *result, const VxMatrix *mat, const VxVector *v) {
    __m128 vec = VxSIMDLoadFloat3(&v->x);

    __m128 m0 = _mm_loadu_ps(&(*mat)[0][0]);
    __m128 m1 = _mm_loadu_ps(&(*mat)[1][0]);
    __m128 m2 = _mm_loadu_ps(&(*mat)[2][0]);

    __m128 v_x = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));

    // Use FMA
    __m128 res = _mm_mul_ps(m0, v_x);
    res = _mm_fmadd_ps(m1, v_y, res);
    res = _mm_fmadd_ps(m2, v_z, res);

    VxSIMDStoreFloat3(&result->x, res);
    _mm256_zeroupper();
}

void VxSIMDMultiplyMatrixVectorMany_AVX(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride) {
    if (count <= 0) return;

    // Pre-load matrix rows
    __m128 m0 = _mm_loadu_ps(&(*mat)[0][0]);
    __m128 m1 = _mm_loadu_ps(&(*mat)[1][0]);
    __m128 m2 = _mm_loadu_ps(&(*mat)[2][0]);
    __m128 m3 = _mm_loadu_ps(&(*mat)[3][0]);

    const char *srcPtr = reinterpret_cast<const char *>(vectors);
    char *dstPtr = reinterpret_cast<char *>(results);

    for (int i = 0; i < count; ++i) {
        const VxVector *vec = reinterpret_cast<const VxVector *>(srcPtr + i * stride);
        VxVector *result = reinterpret_cast<VxVector *>(dstPtr + i * stride);

        __m128 v = VxSIMDLoadFloat3(&vec->x);

        __m128 v_x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 v_y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 v_z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

        // Use FMA
        __m128 res = _mm_mul_ps(m0, v_x);
        res = _mm_fmadd_ps(m1, v_y, res);
        res = _mm_fmadd_ps(m2, v_z, res);
        res = _mm_add_ps(res, m3);

        VxSIMDStoreFloat3(&result->x, res);
    }

    _mm256_zeroupper();
}

void VxSIMDRotateVectorMany_AVX(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride) {
    if (count <= 0) return;

    // Pre-load matrix rows
    __m128 m0 = _mm_loadu_ps(&(*mat)[0][0]);
    __m128 m1 = _mm_loadu_ps(&(*mat)[1][0]);
    __m128 m2 = _mm_loadu_ps(&(*mat)[2][0]);

    const char *srcPtr = reinterpret_cast<const char *>(vectors);
    char *dstPtr = reinterpret_cast<char *>(results);

    for (int i = 0; i < count; ++i) {
        const VxVector *vec = reinterpret_cast<const VxVector *>(srcPtr + i * stride);
        VxVector *result = reinterpret_cast<VxVector *>(dstPtr + i * stride);

    __m128 v = VxSIMDLoadFloat3(&vec->x);

        __m128 v_x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 v_y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 v_z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

        // Use FMA
        __m128 res = _mm_mul_ps(m0, v_x);
        res = _mm_fmadd_ps(m1, v_y, res);
        res = _mm_fmadd_ps(m2, v_z, res);

        VxSIMDStoreFloat3(&result->x, res);
    }

    _mm256_zeroupper();
}

void VxSIMDNormalizeQuaternion_AVX(VxQuaternion *q) {
    // Load all 4 components
    __m128 quat = _mm_loadu_ps(&q->x);

    __m128 magSq = VxSIMDDotProduct4(quat, quat);

    float magSqScalar;
    _mm_store_ss(&magSqScalar, magSq);

    const float epsilonSq = EPSILON * EPSILON;
    if (magSqScalar <= epsilonSq) {
        _mm256_zeroupper();
        return; // Preserve original quaternion for tiny magnitudes
    }

    __m128 invMag = VxSIMDReciprocalSqrtAccurate(magSq);
    __m128 result = _mm_mul_ps(quat, invMag);
    _mm_storeu_ps(&q->x, result);

    _mm256_zeroupper();
}

void VxSIMDMultiplyQuaternion_AVX(VxQuaternion *result, const VxQuaternion *a, const VxQuaternion *b) {
    // Load quaternions as {x, y, z, w}
    __m128 qa = _mm_loadu_ps(&a->x);
    __m128 qb = _mm_loadu_ps(&b->x);

    // Quaternion multiplication formula:
    // result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
    // result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
    // result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    // result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z

    // Extract components
    __m128 aw = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 ax = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 ay = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 az = _mm_shuffle_ps(qa, qa, _MM_SHUFFLE(2, 2, 2, 2));

    // Compute terms using FMA
    // t0 = {aw*bx, aw*by, aw*bz, aw*bw}
    __m128 t0 = _mm_mul_ps(aw, qb);

    // t1 = {ax*bw, ax*(-bz), ax*by, ax*(-bx)}
    __m128 b_perm1 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(0, 1, 2, 3)); // {bw, bz, by, bx}
    __m128 sign1 = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    __m128 t1 = _mm_fmadd_ps(ax, _mm_mul_ps(b_perm1, sign1), t0);

    // t2 = {ay*bz, ay*bw, ay*(-bx), ay*(-by)}
    __m128 b_perm2 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(1, 0, 3, 2)); // {bz, bw, bx, by}
    __m128 sign2 = _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f);
    __m128 t2 = _mm_fmadd_ps(ay, _mm_mul_ps(b_perm2, sign2), t1);

    // t3 = {(-az)*by, az*bx, az*bw, az*(-bz)}
    __m128 b_perm3 = _mm_shuffle_ps(qb, qb, _MM_SHUFFLE(2, 3, 0, 1)); // {by, bx, bw, bz}
    __m128 sign3 = _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f);
    __m128 r = _mm_fmadd_ps(az, _mm_mul_ps(b_perm3, sign3), t2);

    _mm_storeu_ps(&result->x, r);
    _mm256_zeroupper();
}

void VxSIMDSlerpQuaternion_AVX(VxQuaternion *result, float t, const VxQuaternion *a, const VxQuaternion *b) {
    // Load quaternions
    __m128 qa = _mm_loadu_ps(&a->x);
    __m128 qb = _mm_loadu_ps(&b->x);

    __m128 cosOmega_vec = VxSIMDDotProduct4(qa, qb);

    float cosOmega;
    _mm_store_ss(&cosOmega, cosOmega_vec);

    float k0, k1;

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

    // Interpolate using FMA
    __m128 k0_vec = _mm_set1_ps(k0);
    __m128 k1_vec = _mm_set1_ps(k1);
    __m128 r = _mm_fmadd_ps(qb, k1_vec, _mm_mul_ps(qa, k0_vec));
    _mm_storeu_ps(&result->x, r);

    _mm256_zeroupper();
}

int VxSIMDConvertPixelBatch32_AVX(const XULONG* srcPixels, XULONG* dstPixels, int count, const VxPixelSimdConfig& config) {
    if (!config.enabled) {
        return 0;
    }

    const int simdCount = count & ~7;
    if (simdCount <= 0) {
        return 0;
    }

    __m256i alphaVec = config.alphaFill ? _mm256_set1_epi32(static_cast<int>(config.alphaFillComponent)) : _mm256_setzero_si256();
    __m256i srcMaskVec[4];
    __m256i dstMaskVec[4];
    for (int c = 0; c < 4; ++c) {
        srcMaskVec[c] = _mm256_set1_epi32(static_cast<int>(config.srcMasks[c]));
        dstMaskVec[c] = _mm256_set1_epi32(static_cast<int>(config.dstMasks[c]));
    }

    for (int i = 0; i < simdCount; i += 8) {
        __m256i srcVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcPixels + i));
        __m256i dstVec = alphaVec;

        for (int c = 0; c < 4; ++c) {
            if (!config.channelCopy[c]) {
                continue;
            }

            __m256i channel = _mm256_and_si256(srcVec, srcMaskVec[c]);
            if (config.srcShiftRight[c]) {
                __m128i shift = _mm_cvtsi32_si128(config.srcShiftRight[c]);
                channel = _mm256_srl_epi32(channel, shift);
            }
            if (config.dstShiftLeft[c]) {
                __m128i shift = _mm_cvtsi32_si128(config.dstShiftLeft[c]);
                channel = _mm256_sll_epi32(channel, shift);
            }
            channel = _mm256_and_si256(channel, dstMaskVec[c]);
            dstVec = _mm256_or_si256(dstVec, channel);
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstPixels + i), dstVec);
    }

    _mm256_zeroupper();
    return simdCount;
}

int VxSIMDApplyAlphaBatch32_AVX(XULONG* pixels, int count, XBYTE alphaValue, XULONG alphaMask, XULONG alphaShift) {
    const int simdCount = count & ~7;
    if (simdCount <= 0) {
        return 0;
    }

    const XULONG alphaComponent = (static_cast<XULONG>(alphaValue) << alphaShift) & alphaMask;
    const XULONG colorMask = ~alphaMask;

    const __m256i alphaMaskVec = _mm256_set1_epi32(static_cast<int>(alphaMask));
    const __m256i colorMaskVec = _mm256_set1_epi32(static_cast<int>(colorMask));
    __m256i alphaVec = _mm256_set1_epi32(static_cast<int>(alphaComponent));
    alphaVec = _mm256_and_si256(alphaVec, alphaMaskVec);

    for (int i = 0; i < simdCount; i += 8) {
        __m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels + i));
        __m256i masked = _mm256_and_si256(src, colorMaskVec);
        __m256i result = _mm256_or_si256(masked, alphaVec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pixels + i), result);
    }

    _mm256_zeroupper();
    return simdCount;
}

int VxSIMDApplyVariableAlphaBatch32_AVX(XULONG* pixels, const XBYTE* alphaValues, int count, XULONG alphaMask, XULONG alphaShift) {
    const int simdCount = count & ~7;
    if (simdCount <= 0) {
        return 0;
    }

    const XULONG colorMask = ~alphaMask;
    const __m256i alphaMaskVec = _mm256_set1_epi32(static_cast<int>(alphaMask));
    const __m256i colorMaskVec = _mm256_set1_epi32(static_cast<int>(colorMask));
    const __m128i shift = _mm_cvtsi32_si128(alphaShift);

    for (int i = 0; i < simdCount; i += 8) {
        __m128i alphaBytes128 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(alphaValues + i));
        __m256i alphaBytes = _mm256_cvtepu8_epi32(alphaBytes128);
        __m256i alphaVec = _mm256_sll_epi32(alphaBytes, shift);
        alphaVec = _mm256_and_si256(alphaVec, alphaMaskVec);

        __m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels + i));
        __m256i masked = _mm256_and_si256(src, colorMaskVec);
        __m256i result = _mm256_or_si256(masked, alphaVec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pixels + i), result);
    }

    _mm256_zeroupper();
    return simdCount;
}

void VxSIMDInterpolateFloatArray_AVX(float *result, const float *a, const float *b, float factor, int count) {
    if (count <= 0) {
        return;
    }

    // Match scalar: result = a + (b - a) * factor
    const __m256 factorVec = _mm256_set1_ps(factor);

#if defined(_MSC_VER)
    float* __restrict out = result;
    const float* __restrict aa = a;
    const float* __restrict bb = b;
#else
    float* out = result;
    const float* aa = a;
    const float* bb = b;
#endif

    const bool aligned32 = (((std::uintptr_t)out | (std::uintptr_t)aa | (std::uintptr_t)bb) & 31u) == 0u;

    const int simdCount16 = count & ~15; // 16 at a time (2x ymm)
    int i = 0;
    if (aligned32) {
        for (; i < simdCount16; i += 16) {
            const __m256 a0 = _mm256_load_ps(aa + i);
            const __m256 b0 = _mm256_load_ps(bb + i);
            const __m256 a1 = _mm256_load_ps(aa + i + 8);
            const __m256 b1 = _mm256_load_ps(bb + i + 8);

            const __m256 d0 = _mm256_sub_ps(b0, a0);
            const __m256 d1 = _mm256_sub_ps(b1, a1);
            const __m256 r0 = _mm256_add_ps(a0, _mm256_mul_ps(d0, factorVec));
            const __m256 r1 = _mm256_add_ps(a1, _mm256_mul_ps(d1, factorVec));

            _mm256_store_ps(out + i, r0);
            _mm256_store_ps(out + i + 8, r1);
        }
    } else {
        for (; i < simdCount16; i += 16) {
            const __m256 a0 = _mm256_loadu_ps(aa + i);
            const __m256 b0 = _mm256_loadu_ps(bb + i);
            const __m256 a1 = _mm256_loadu_ps(aa + i + 8);
            const __m256 b1 = _mm256_loadu_ps(bb + i + 8);

            const __m256 d0 = _mm256_sub_ps(b0, a0);
            const __m256 d1 = _mm256_sub_ps(b1, a1);
            const __m256 r0 = _mm256_add_ps(a0, _mm256_mul_ps(d0, factorVec));
            const __m256 r1 = _mm256_add_ps(a1, _mm256_mul_ps(d1, factorVec));

            _mm256_storeu_ps(out + i, r0);
            _mm256_storeu_ps(out + i + 8, r1);
        }
    }

    const int simdCount8 = count & ~7;
    for (; i < simdCount8; i += 8) {
        const __m256 aVec = _mm256_loadu_ps(aa + i);
        const __m256 bVec = _mm256_loadu_ps(bb + i);
        const __m256 d = _mm256_sub_ps(bVec, aVec);
        const __m256 r = _mm256_add_ps(aVec, _mm256_mul_ps(d, factorVec));
        _mm256_storeu_ps(out + i, r);
    }

    for (; i < count; ++i) {
        out[i] = aa[i] + (bb[i] - aa[i]) * factor;
    }

    _mm256_zeroupper();
}

void VxSIMDInterpolateVectorArray_AVX(void *result, const void *a, const void *b, float factor, int count, XULONG strideResult, XULONG strideInput) {
    VxSIMDInterpolateVectorArray_SSE(result, a, b, factor, count, strideResult, strideInput);
}

XBOOL VxSIMDTransformBox2D_AVX(const VxMatrix *worldProjection, const VxBbox *box, VxRect *screenSize, VxRect *extents, VXCLIP_FLAGS *orClipFlags, VXCLIP_FLAGS *andClipFlags) {
    return VxSIMDTransformBox2D_SSE(worldProjection, box, screenSize, extents, orClipFlags, andClipFlags);
}

void VxSIMDProjectBoxZExtents_AVX(const VxMatrix *worldProjection, const VxBbox *box, float *zhMin, float *zhMax) {
    VxSIMDProjectBoxZExtents_SSE(worldProjection, box, zhMin, zhMax);
}

XBOOL VxSIMDComputeBestFitBBox_AVX(const XBYTE *points, XULONG stride, int count, VxMatrix *bboxMatrix, float additionalBorder) {
    return VxSIMDComputeBestFitBBox_SSE(points, stride, count, bboxMatrix, additionalBorder);
}

int VxSIMDBboxClassify_AVX(const VxBbox *self, const VxBbox *other, const VxVector *point) {
    return VxSIMDBboxClassify_SSE(self, other, point);
}

void VxSIMDBboxClassifyVertices_AVX(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, XULONG *flags) {
    VxSIMDBboxClassifyVertices_SSE(self, count, vertices, stride, flags);
}

void VxSIMDBboxClassifyVerticesOneAxis_AVX(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, int axis, XULONG *flags) {
    VxSIMDBboxClassifyVerticesOneAxis_SSE(self, count, vertices, stride, axis, flags);
}

void VxSIMDBboxTransformTo_AVX(const VxBbox *self, VxVector *points, const VxMatrix *mat) {
    VxSIMDBboxTransformTo_SSE(self, points, mat);
}

void VxSIMDBboxTransformFrom_AVX(VxBbox *dest, const VxBbox *src, const VxMatrix *mat) {
    VxSIMDBboxTransformFrom_SSE(dest, src, mat);
}

void VxSIMDFrustumUpdate_AVX(VxFrustum *frustum) {
    VxSIMDFrustumUpdate_SSE(frustum);
}

void VxSIMDFrustumComputeVertices_AVX(const VxFrustum *frustum, VxVector *vertices) {
    VxSIMDFrustumComputeVertices_SSE(frustum, vertices);
}

void VxSIMDFrustumTransform_AVX(VxFrustum *frustum, const VxMatrix *invWorldMat) {
    VxSIMDFrustumTransform_SSE(frustum, invWorldMat);
}

// Additional Vector operations stubs
void VxSIMDAddVector_AVX(VxVector *result, const VxVector *a, const VxVector *b) {
    VxSIMDAddVector_SSE(result, a, b);
    _mm256_zeroupper();
}

void VxSIMDSubtractVector_AVX(VxVector *result, const VxVector *a, const VxVector *b) {
    VxSIMDSubtractVector_SSE(result, a, b);
    _mm256_zeroupper();
}

void VxSIMDScaleVector_AVX(VxVector *result, const VxVector *v, float scalar) {
    VxSIMDScaleVector_SSE(result, v, scalar);
    _mm256_zeroupper();
}

float VxSIMDDotVector_AVX(const VxVector *a, const VxVector *b) {
    // Implement directly (avoid calling SSE variant + vzeroupper cost).
    // Compiled with AVX enabled, so these are VEX-encoded XMM ops (no AVX<->SSE transition penalty).
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

void VxSIMDCrossVector_AVX(VxVector *result, const VxVector *a, const VxVector *b) {
    const __m128 aVec = VxSIMDLoadFloat3(&a->x);
    const __m128 bVec = VxSIMDLoadFloat3(&b->x);
    const __m128 r = VxSIMDCrossProduct3(aVec, bVec);
    VxSIMDStoreFloat3(&result->x, r);
}

float VxSIMDLengthVector_AVX(const VxVector *v) {
    float result = VxSIMDLengthVector_SSE(v);
    _mm256_zeroupper();
    return result;
}

float VxSIMDLengthSquaredVector_AVX(const VxVector *v) {
    float result = VxSIMDLengthSquaredVector_SSE(v);
    _mm256_zeroupper();
    return result;
}

float VxSIMDDistanceVector_AVX(const VxVector *a, const VxVector *b) {
    float result = VxSIMDDistanceVector_SSE(a, b);
    _mm256_zeroupper();
    return result;
}

void VxSIMDLerpVector_AVX(VxVector *result, const VxVector *a, const VxVector *b, float t) {
    VxSIMDLerpVector_SSE(result, a, b, t);
    _mm256_zeroupper();
}

void VxSIMDReflectVector_AVX(VxVector *result, const VxVector *incident, const VxVector *normal) {
    VxSIMDReflectVector_SSE(result, incident, normal);
    _mm256_zeroupper();
}

void VxSIMDMinimizeVector_AVX(VxVector *result, const VxVector *a, const VxVector *b) {
    VxSIMDMinimizeVector_SSE(result, a, b);
    _mm256_zeroupper();
}

void VxSIMDMaximizeVector_AVX(VxVector *result, const VxVector *a, const VxVector *b) {
    VxSIMDMaximizeVector_SSE(result, a, b);
    _mm256_zeroupper();
}

// Vector4 operations stubs
void VxSIMDAddVector4_AVX(VxVector4 *result, const VxVector4 *a, const VxVector4 *b) {
    VxSIMDAddVector4_SSE(result, a, b);
    _mm256_zeroupper();
}

void VxSIMDSubtractVector4_AVX(VxVector4 *result, const VxVector4 *a, const VxVector4 *b) {
    VxSIMDSubtractVector4_SSE(result, a, b);
    _mm256_zeroupper();
}

void VxSIMDScaleVector4_AVX(VxVector4 *result, const VxVector4 *v, float scalar) {
    VxSIMDScaleVector4_SSE(result, v, scalar);
    _mm256_zeroupper();
}

float VxSIMDDotVector4_AVX(const VxVector4 *a, const VxVector4 *b) {
    float result = VxSIMDDotVector4_SSE(a, b);
    _mm256_zeroupper();
    return result;
}

void VxSIMDLerpVector4_AVX(VxVector4 *result, const VxVector4 *a, const VxVector4 *b, float t) {
    VxSIMDLerpVector4_SSE(result, a, b, t);
    _mm256_zeroupper();
}

// Matrix strided operations stubs
void VxSIMDMultiplyMatrixVectorStrided_AVX(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count) {
    VxSIMDMultiplyMatrixVectorStrided_SSE(dest, src, mat, count);
    _mm256_zeroupper();
}

void VxSIMDMultiplyMatrixVector4Strided_AVX(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count) {
    VxSIMDMultiplyMatrixVector4Strided_SSE(dest, src, mat, count);
    _mm256_zeroupper();
}

void VxSIMDRotateVectorStrided_AVX(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count) {
    VxSIMDRotateVectorStrided_SSE(dest, src, mat, count);
    _mm256_zeroupper();
}

// Matrix utility operations stubs
void VxSIMDMatrixIdentity_AVX(VxMatrix *mat) {
    VxSIMDMatrixIdentity_SSE(mat);
    _mm256_zeroupper();
}

void VxSIMDMatrixInverse_AVX(VxMatrix *result, const VxMatrix *mat) {
    VxSIMDMatrixInverse_SSE(result, mat);
    _mm256_zeroupper();
}

float VxSIMDMatrixDeterminant_AVX(const VxMatrix *mat) {
    float result = VxSIMDMatrixDeterminant_SSE(mat);
    _mm256_zeroupper();
    return result;
}

void VxSIMDMatrixFromRotation_AVX(VxMatrix *result, const VxVector *axis, float angle) {
    VxSIMDMatrixFromRotation_SSE(result, axis, angle);
    _mm256_zeroupper();
}

void VxSIMDMatrixFromRotationOrigin_AVX(VxMatrix *result, const VxVector *axis, const VxVector *origin, float angle) {
    VxSIMDMatrixFromRotationOrigin_SSE(result, axis, origin, angle);
    _mm256_zeroupper();
}

void VxSIMDMatrixFromEuler_AVX(VxMatrix *result, float eax, float eay, float eaz) {
    VxSIMDMatrixFromEuler_SSE(result, eax, eay, eaz);
    _mm256_zeroupper();
}

void VxSIMDMatrixToEuler_AVX(const VxMatrix *mat, float *eax, float *eay, float *eaz) {
    VxSIMDMatrixToEuler_SSE(mat, eax, eay, eaz);
    _mm256_zeroupper();
}

void VxSIMDMatrixInterpolate_AVX(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b) {
    VxSIMDMatrixInterpolate_SSE(step, result, a, b);
    _mm256_zeroupper();
}

void VxSIMDMatrixInterpolateNoScale_AVX(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b) {
    VxSIMDMatrixInterpolateNoScale_SSE(step, result, a, b);
    _mm256_zeroupper();
}

void VxSIMDMatrixDecompose_AVX(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale) {
    VxSIMDMatrixDecompose_SSE(mat, quat, pos, scale);
    _mm256_zeroupper();
}

float VxSIMDMatrixDecomposeTotal_AVX(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot) {
    float result = VxSIMDMatrixDecomposeTotal_SSE(mat, quat, pos, scale, uRot);
    _mm256_zeroupper();
    return result;
}

float VxSIMDMatrixDecomposeTotalPtr_AVX(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot) {
    float result = VxSIMDMatrixDecomposeTotalPtr_SSE(mat, quat, pos, scale, uRot);
    _mm256_zeroupper();
    return result;
}

// Quaternion additional operations stubs
void VxSIMDQuaternionFromMatrix_AVX(VxQuaternion *result, const VxMatrix *mat, XBOOL matIsUnit, XBOOL restoreMat) {
    VxSIMDQuaternionFromMatrix_SSE(result, mat, matIsUnit, restoreMat);
    _mm256_zeroupper();
}

void VxSIMDQuaternionToMatrix_AVX(VxMatrix *result, const VxQuaternion *q) {
    VxSIMDQuaternionToMatrix_SSE(result, q);
    _mm256_zeroupper();
}

void VxSIMDQuaternionFromRotation_AVX(VxQuaternion *result, const VxVector *axis, float angle) {
    VxSIMDQuaternionFromRotation_SSE(result, axis, angle);
    _mm256_zeroupper();
}

void VxSIMDQuaternionFromEuler_AVX(VxQuaternion *result, float eax, float eay, float eaz) {
    VxSIMDQuaternionFromEuler_SSE(result, eax, eay, eaz);
    _mm256_zeroupper();
}

void VxSIMDQuaternionToEuler_AVX(const VxQuaternion *q, float *eax, float *eay, float *eaz) {
    VxSIMDQuaternionToEuler_SSE(q, eax, eay, eaz);
    _mm256_zeroupper();
}

void VxSIMDQuaternionMultiplyInPlace_AVX(VxQuaternion *self, const VxQuaternion *rhs) {
    VxSIMDQuaternionMultiplyInPlace_SSE(self, rhs);
    _mm256_zeroupper();
}

void VxSIMDQuaternionConjugate_AVX(VxQuaternion *result, const VxQuaternion *q) {
    VxSIMDQuaternionConjugate_SSE(result, q);
    _mm256_zeroupper();
}

void VxSIMDQuaternionDivide_AVX(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q) {
    VxSIMDQuaternionDivide_SSE(result, p, q);
    _mm256_zeroupper();
}

void VxSIMDQuaternionSnuggle_AVX(VxQuaternion *result, VxQuaternion *quat, VxVector *scale) {
    VxSIMDQuaternionSnuggle_SSE(result, quat, scale);
    _mm256_zeroupper();
}

void VxSIMDQuaternionLn_AVX(VxQuaternion *result, const VxQuaternion *q) {
    VxSIMDQuaternionLn_SSE(result, q);
    _mm256_zeroupper();
}

void VxSIMDQuaternionExp_AVX(VxQuaternion *result, const VxQuaternion *q) {
    VxSIMDQuaternionExp_SSE(result, q);
    _mm256_zeroupper();
}

void VxSIMDQuaternionLnDif_AVX(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q) {
    VxSIMDQuaternionLnDif_SSE(result, p, q);
    _mm256_zeroupper();
}

void VxSIMDQuaternionSquad_AVX(VxQuaternion *result, float t, const VxQuaternion *quat1, const VxQuaternion *quat1Out, const VxQuaternion *quat2In, const VxQuaternion *quat2) {
    VxSIMDQuaternionSquad_SSE(result, t, quat1, quat1Out, quat2In, quat2);
    _mm256_zeroupper();
}

// Ray operations stubs
void VxSIMDRayTransform_AVX(VxRay *dest, const VxRay *ray, const VxMatrix *mat) {
    VxSIMDRayTransform_SSE(dest, ray, mat);
    _mm256_zeroupper();
}

// Plane operations stubs
void VxSIMDPlaneCreateFromPoint_AVX(VxPlane *plane, const VxVector *normal, const VxVector *point) {
    VxSIMDPlaneCreateFromPoint_SSE(plane, normal, point);
    _mm256_zeroupper();
}

void VxSIMDPlaneCreateFromTriangle_AVX(VxPlane *plane, const VxVector *a, const VxVector *b, const VxVector *c) {
    VxSIMDPlaneCreateFromTriangle_SSE(plane, a, b, c);
    _mm256_zeroupper();
}

// Rect operations stubs
void VxSIMDRectTransform_AVX(VxRect *rect, const VxRect *destScreen, const VxRect *srcScreen) {
    VxSIMDRectTransform_SSE(rect, destScreen, srcScreen);
    _mm256_zeroupper();
}

void VxSIMDRectTransformBySize_AVX(VxRect *rect, const Vx2DVector *destScreenSize, const Vx2DVector *srcScreenSize) {
    VxSIMDRectTransformBySize_SSE(rect, destScreenSize, srcScreenSize);
    _mm256_zeroupper();
}

void VxSIMDRectTransformToHomogeneous_AVX(VxRect *rect, const VxRect *screen) {
    VxSIMDRectTransformToHomogeneous_SSE(rect, screen);
    _mm256_zeroupper();
}

void VxSIMDRectTransformFromHomogeneous_AVX(VxRect *rect, const VxRect *screen) {
    VxSIMDRectTransformFromHomogeneous_SSE(rect, screen);
    _mm256_zeroupper();
}

#endif // VX_SIMD_X86
