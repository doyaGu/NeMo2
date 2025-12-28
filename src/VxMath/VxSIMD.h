#ifndef VXSIMD_H
#define VXSIMD_H

#include "VxMathDefines.h"

// ============================================================================
// SIMD Configuration and CPU Feature Detection
// ============================================================================

#include <cstdlib>
#include <cmath>

// Platform detection
#if defined(_MSC_VER)
#define VX_SIMD_MSVC 1
#include <malloc.h>
#elif defined(__GNUC__) || defined(__clang__)
#define VX_SIMD_GCC 1
#endif

// Architecture detection
#if defined(_M_X64) || defined(_M_AMD64) || defined(__x86_64__) || defined(__amd64__)
#define VX_SIMD_X64 1
#define VX_SIMD_X86 1   /* x64 is an x86 family for our needs */
#elif defined(_M_IX86) || defined(__i386__)
#define VX_SIMD_X86 1
#elif defined(_M_ARM64) || defined(__aarch64__)
#define VX_SIMD_ARM64 1
#define VX_SIMD_ARM 1
#elif defined(_M_ARM) || defined(__arm__)
#define VX_SIMD_ARM 1
#endif

/* x86/x64 SIMD feature detection */
#if defined(VX_SIMD_X86)

/* Use SIMDe for portable SIMD intrinsics */
#define SIMDE_ENABLE_NATIVE_ALIASES

/*
 * By default, prefer IEEE-ish behavior so SIMD matches scalar code
 * more closely. Projects that want maximum throughput can opt in.
 */
#if !defined(VX_SIMD_USE_SIMDE_FAST_MATH)
#define VX_SIMD_USE_SIMDE_FAST_MATH 0
#endif

/* Optional SIMDe performance knobs (opt-in). */
#if VX_SIMD_USE_SIMDE_FAST_MATH
#if !defined(SIMDE_NO_FAST_NANS)
#define SIMDE_FAST_NANS
#endif
#if !defined(SIMDE_NO_FAST_ROUND_MODE)
#define SIMDE_FAST_ROUND_MODE
#endif
#if !defined(SIMDE_NO_FAST_EXCEPTIONS)
#define SIMDE_FAST_EXCEPTIONS
#endif
#endif

/* SSE (x64 implies SSE2 baseline; for 32-bit MSVC use _M_IX86_FP) */
#if defined(VX_SIMD_X64) \
      || (defined(_MSC_VER) && defined(_M_IX86_FP) && _M_IX86_FP >= 1) \
      || defined(__SSE__)
#define VX_SIMD_SSE 1
#include <simde/x86/sse.h>
#endif

/* SSE2 */
#if defined(VX_SIMD_X64) \
      || (defined(_MSC_VER) && defined(_M_IX86_FP) && _M_IX86_FP >= 2) \
      || defined(__SSE2__)
#define VX_SIMD_SSE2 1
#include <simde/x86/sse2.h>
#endif

/* SSE3 */
#if defined(__SSE3__)
#define VX_SIMD_SSE3 1
#include <simde/x86/sse3.h>
#endif

/* SSSE3 */
#if defined(__SSSE3__)
#define VX_SIMD_SSSE3 1
#include <simde/x86/ssse3.h>
#endif

/* SSE4.1 */
#if defined(__SSE4_1__)
#define VX_SIMD_SSE4_1 1
#include <simde/x86/sse4.1.h>
#endif

/* SSE4.2 */
#if defined(__SSE4_2__)
#define VX_SIMD_SSE4_2 1
#include <simde/x86/sse4.2.h>
#endif

/* AVX / AVX2 / FMA / AVX-512: compilers usually define these when enabled */
#if defined(__AVX__)
#define VX_SIMD_AVX 1
#include <simde/x86/avx.h>
#endif

#if defined(__AVX2__)
#define VX_SIMD_AVX2 1
#include <simde/x86/avx2.h>
#endif

#if defined(__FMA__)
#define VX_SIMD_FMA 1
#include <simde/x86/fma.h>
#endif

#if defined(__AVX512F__)
#define VX_SIMD_AVX512 1
#include <simde/x86/avx512/f.h>
#endif
#endif

// CPU feature detection at runtime
#if defined(VX_SIMD_X86)
#if defined(VX_SIMD_MSVC)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

/**
 * @brief CPU feature flags for runtime detection
 */
struct VxSIMDFeatures {
    bool SSE = false;
    bool SSE2 = false;
    bool SSE3 = false;
    bool SSSE3 = false;
    bool SSE4_1 = false;
    bool SSE4_2 = false;
    bool AVX = false;
    bool AVX2 = false;
    bool FMA = false;
    bool AVX512F = false;
    bool XSAVE = false;
    bool OSXSAVE = false;
};

/**
 * @brief Detects CPU features at runtime
 * @return A VxSIMDFeatures structure with detected capabilities
 */
VX_EXPORT VxSIMDFeatures VxDetectSIMDFeatures();

/**
 * @brief Gets the cached CPU features (only detects once)
 * @return Reference to the cached CPU features
 */
VX_EXPORT const VxSIMDFeatures &VxGetSIMDFeatures();

/**
 * @brief Returns a string describing available SIMD instruction sets
 */
VX_EXPORT const char *VxGetSIMDInfo();

// ============================================================================
// SIMD Helper Macros
// ============================================================================

// Alignment macros for SIMD data
#if defined(VX_SIMD_MSVC)
#define VX_ALIGN(x) __declspec(align(x))
#elif defined(VX_SIMD_GCC)
#define VX_ALIGN(x) __attribute__((aligned(x)))
#else
#define VX_ALIGN(x)
#endif

static inline void *VxAlignedMalloc(size_t size, size_t alignment) {
#if defined(VX_SIMD_MSVC)
    return _aligned_malloc(size, alignment);
#else
    if (alignment < sizeof(void *)) {
        alignment = sizeof(void *);
    }
    if ((alignment & (alignment - 1)) != 0) {
        size_t pow2 = sizeof(void *);
        while (pow2 < alignment) {
            pow2 <<= 1;
        }
        alignment = pow2;
    }
    const size_t totalSize = size + alignment - 1 + sizeof(void *);
    void *raw = malloc(totalSize);
    if (!raw) {
        return nullptr;
    }
    const uintptr_t rawAddr = reinterpret_cast<uintptr_t>(raw) + sizeof(void *);
    const uintptr_t alignedAddr = (rawAddr + (alignment - 1)) & ~(static_cast<uintptr_t>(alignment) - 1);
    reinterpret_cast<void **>(alignedAddr)[-1] = raw;
    return reinterpret_cast<void *>(alignedAddr);
#endif
}

static inline void VxAlignedFree(void *ptr) {
#if defined(VX_SIMD_MSVC)
    _aligned_free(ptr);
#else
    if (!ptr) {
        return;
    }
    void *raw = reinterpret_cast<void **>(ptr)[-1];
    free(raw);
#endif
}

#define VX_ALIGNED_MALLOC(size, alignment) VxAlignedMalloc(size, alignment)
#define VX_ALIGNED_FREE(ptr) VxAlignedFree(ptr)

// Common SIMD alignments
#define VX_ALIGN_SSE VX_ALIGN(16)
#define VX_ALIGN_AVX VX_ALIGN(32)
#define VX_ALIGN_AVX512 VX_ALIGN(64)

// Force inline for SIMD functions
#if defined(VX_SIMD_MSVC)
#define VX_SIMD_INLINE __forceinline
#elif defined(VX_SIMD_GCC)
#define VX_SIMD_INLINE inline __attribute__((always_inline))
#else
#define VX_SIMD_INLINE inline
#endif

// ============================================================================
// SIMD Type Definitions
// ============================================================================

#if defined(VX_SIMD_SSE)
typedef __m128 vx_simd_float4;
typedef __m128i vx_simd_int4;
#endif

#if defined(VX_SIMD_AVX)
typedef __m256 vx_simd_float8;
typedef __m256i vx_simd_int8;
#endif

// ============================================================================
// SIMD Dispatch Declarations
// ============================================================================

// Forward declarations for types used by dispatch signatures.
struct VxVector;
class VxVector4;
struct VxQuaternion;
class VxMatrix;
class VxRay;
class VxPlane;
class VxRect;
struct Vx2DVector;
struct VxStridedData;
struct VxBbox;
class VxFrustum;

// Vector operations
typedef void (*VxSIMDNormalizeVectorFn)(VxVector *v);
typedef void (*VxSIMDRotateVectorFn)(VxVector *result, const VxMatrix *mat, const VxVector *v);
typedef void (*VxSIMDAddVectorFn)(VxVector *result, const VxVector *a, const VxVector *b);
typedef void (*VxSIMDSubtractVectorFn)(VxVector *result, const VxVector *a, const VxVector *b);
typedef void (*VxSIMDScaleVectorFn)(VxVector *result, const VxVector *v, float scalar);
typedef float (*VxSIMDDotVectorFn)(const VxVector *a, const VxVector *b);
typedef void (*VxSIMDCrossVectorFn)(VxVector *result, const VxVector *a, const VxVector *b);
typedef float (*VxSIMDLengthVectorFn)(const VxVector *v);
typedef float (*VxSIMDLengthSquaredVectorFn)(const VxVector *v);
typedef float (*VxSIMDDistanceVectorFn)(const VxVector *a, const VxVector *b);
typedef void (*VxSIMDLerpVectorFn)(VxVector *result, const VxVector *a, const VxVector *b, float t);
typedef void (*VxSIMDReflectVectorFn)(VxVector *result, const VxVector *incident, const VxVector *normal);
typedef void (*VxSIMDMinimizeVectorFn)(VxVector *result, const VxVector *a, const VxVector *b);
typedef void (*VxSIMDMaximizeVectorFn)(VxVector *result, const VxVector *a, const VxVector *b);

// Matrix operations
typedef void (*VxSIMDMultiplyMatrixFn)(VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
typedef void (*VxSIMDMultiplyMatrix4Fn)(VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
typedef void (*VxSIMDTransposeMatrixFn)(VxMatrix *result, const VxMatrix *a);

// Matrix-vector operations
typedef void (*VxSIMDMultiplyMatrixVectorFn)(VxVector *result, const VxMatrix *mat, const VxVector *v);
typedef void (*VxSIMDMultiplyMatrixVector4Fn)(VxVector4 *result, const VxMatrix *mat, const VxVector4 *v);
typedef void (*VxSIMDRotateVectorOpFn)(VxVector *result, const VxMatrix *mat, const VxVector *v);

// Batch operations
typedef void (*VxSIMDMultiplyMatrixVectorManyFn)(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride);
typedef void (*VxSIMDRotateVectorManyFn)(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride);
typedef void (*VxSIMDMultiplyMatrixVectorStridedFn)(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);
typedef void (*VxSIMDMultiplyMatrixVector4StridedFn)(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);
typedef void (*VxSIMDRotateVectorStridedFn)(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);

// Quaternion operations
typedef void (*VxSIMDNormalizeQuaternionFn)(VxQuaternion *q);
typedef void (*VxSIMDMultiplyQuaternionFn)(VxQuaternion *result, const VxQuaternion *a, const VxQuaternion *b);
typedef void (*VxSIMDSlerpQuaternionFn)(VxQuaternion *result, float t, const VxQuaternion *a, const VxQuaternion *b);
typedef void (*VxSIMDQuaternionFromMatrixFn)(VxQuaternion *result, const VxMatrix *mat, XBOOL matIsUnit, XBOOL restoreMat);
typedef void (*VxSIMDQuaternionToMatrixFn)(VxMatrix *result, const VxQuaternion *q);
typedef void (*VxSIMDQuaternionFromRotationFn)(VxQuaternion *result, const VxVector *axis, float angle);
typedef void (*VxSIMDQuaternionFromEulerFn)(VxQuaternion *result, float eax, float eay, float eaz);
typedef void (*VxSIMDQuaternionToEulerFn)(const VxQuaternion *q, float *eax, float *eay, float *eaz);
typedef void (*VxSIMDQuaternionMultiplyInPlaceFn)(VxQuaternion *self, const VxQuaternion *rhs);
typedef void (*VxSIMDQuaternionConjugateFn)(VxQuaternion *result, const VxQuaternion *q);
typedef void (*VxSIMDQuaternionDivideFn)(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q);
typedef void (*VxSIMDQuaternionSnuggleFn)(VxQuaternion *result, VxQuaternion *quat, VxVector *scale);
typedef void (*VxSIMDQuaternionLnFn)(VxQuaternion *result, const VxQuaternion *q);
typedef void (*VxSIMDQuaternionExpFn)(VxQuaternion *result, const VxQuaternion *q);
typedef void (*VxSIMDQuaternionLnDifFn)(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q);
typedef void (*VxSIMDQuaternionSquadFn)(VxQuaternion *result, float t, const VxQuaternion *quat1, const VxQuaternion *quat1Out, const VxQuaternion *quat2In, const VxQuaternion *quat2);

struct VxPixelSimdConfig {
    bool enabled;
    bool alphaFill;
    bool channelCopy[4]{};
    int srcShiftRight[4]{};
    int dstShiftLeft[4]{};
    XULONG srcMasks[4]{};
    XULONG dstMasks[4]{};
    XULONG alphaFillComponent;

    VxPixelSimdConfig() : enabled(false), alphaFill(false), alphaFillComponent(0) {
        for (int i = 0; i < 4; ++i) {
            channelCopy[i] = false;
            srcShiftRight[i] = 0;
            dstShiftLeft[i] = 0;
            srcMasks[i] = 0;
            dstMasks[i] = 0;
        }
    }
};

typedef int (*VxSIMDConvertPixelBatch32Fn)(const XULONG *srcPixels, XULONG *dstPixels, int count, const VxPixelSimdConfig &config);
typedef int (*VxSIMDApplyAlphaBatch32Fn)(XULONG *pixels, int count, XBYTE alphaValue, XULONG alphaMask, XULONG alphaShift);
typedef int (*VxSIMDApplyVariableAlphaBatch32Fn)(XULONG *pixels, const XBYTE *alphaValues, int count, XULONG alphaMask, XULONG alphaShift);

// Ray operations
typedef void (*VxSIMDRayTransformFn)(VxRay *dest, const VxRay *ray, const VxMatrix *mat);

// Plane operations
typedef void (*VxSIMDPlaneCreateFromPointFn)(VxPlane *plane, const VxVector *normal, const VxVector *point);
typedef void (*VxSIMDPlaneCreateFromTriangleFn)(VxPlane *plane, const VxVector *a, const VxVector *b, const VxVector *c);

// Rect operations
typedef void (*VxSIMDRectTransformFn)(VxRect *rect, const VxRect *destScreen, const VxRect *srcScreen);
typedef void (*VxSIMDRectTransformBySizeFn)(VxRect *rect, const Vx2DVector *destScreenSize, const Vx2DVector *srcScreenSize);
typedef void (*VxSIMDRectTransformToHomogeneousFn)(VxRect *rect, const VxRect *screen);
typedef void (*VxSIMDRectTransformFromHomogeneousFn)(VxRect *rect, const VxRect *screen);

// Miscellaneous geometry helpers
typedef void (*VxSIMDInterpolateFloatArrayFn)(float *result, const float *a, const float *b, float factor, int count);
typedef void (*VxSIMDInterpolateVectorArrayFn)(void *result, const void *a, const void *b, float factor, int count, XULONG strideResult, XULONG strideInput);
typedef XBOOL (*VxSIMDTransformBox2DFn)(const VxMatrix *worldProjection, const VxBbox *box, VxRect *screenSize, VxRect *extents, VXCLIP_FLAGS *orClipFlags, VXCLIP_FLAGS *andClipFlags);
typedef void (*VxSIMDProjectBoxZExtentsFn)(const VxMatrix *worldProjection, const VxBbox *box, float *zhMin, float *zhMax);
typedef XBOOL (*VxSIMDComputeBestFitBBoxFn)(const XBYTE *points, XULONG stride, int count, VxMatrix *bboxMatrix, float additionalBorder);

// Bounding box helpers
typedef int (*VxSIMDBboxClassifyFn)(const VxBbox *self, const VxBbox *other, const VxVector *point);
typedef void (*VxSIMDBboxClassifyVerticesFn)(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, XULONG *flags);
typedef void (*VxSIMDBboxClassifyVerticesOneAxisFn)(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, int axis, XULONG *flags);
typedef void (*VxSIMDBboxTransformToFn)(const VxBbox *self, VxVector *points, const VxMatrix *mat);
typedef void (*VxSIMDBboxTransformFromFn)(VxBbox *dest, const VxBbox *src, const VxMatrix *mat);

// Frustum helpers
typedef void (*VxSIMDFrustumUpdateFn)(VxFrustum *frustum);
typedef void (*VxSIMDFrustumComputeVerticesFn)(const VxFrustum *frustum, VxVector *vertices);
typedef void (*VxSIMDFrustumTransformFn)(VxFrustum *frustum, const VxMatrix *invWorldMat);

// Matrix decomposition and utility operations
typedef void (*VxSIMDMatrixIdentityFn)(VxMatrix *mat);
typedef void (*VxSIMDMatrixInverseFn)(VxMatrix *result, const VxMatrix *mat);
typedef float (*VxSIMDMatrixDeterminantFn)(const VxMatrix *mat);
typedef void (*VxSIMDMatrixFromRotationFn)(VxMatrix *result, const VxVector *axis, float angle);
typedef void (*VxSIMDMatrixFromRotationOriginFn)(VxMatrix *result, const VxVector *axis, const VxVector *origin, float angle);
typedef void (*VxSIMDMatrixFromEulerFn)(VxMatrix *result, float eax, float eay, float eaz);
typedef void (*VxSIMDMatrixToEulerFn)(const VxMatrix *mat, float *eax, float *eay, float *eaz);
typedef void (*VxSIMDInterpolateMatrixFn)(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
typedef void (*VxSIMDInterpolateMatrixNoScaleFn)(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
typedef void (*VxSIMDDecomposeMatrixFn)(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale);
typedef float (*VxSIMDDecomposeMatrixTotalFn)(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot);
typedef float (*VxSIMDDecomposeMatrixTotalPtrFn)(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot);

// Vector4 operations
typedef void (*VxSIMDAddVector4Fn)(VxVector4 *result, const VxVector4 *a, const VxVector4 *b);
typedef void (*VxSIMDSubtractVector4Fn)(VxVector4 *result, const VxVector4 *a, const VxVector4 *b);
typedef void (*VxSIMDScaleVector4Fn)(VxVector4 *result, const VxVector4 *v, float scalar);
typedef float (*VxSIMDDotVector4Fn)(const VxVector4 *a, const VxVector4 *b);
typedef void (*VxSIMDLerpVector4Fn)(VxVector4 *result, const VxVector4 *a, const VxVector4 *b, float t);

/**
 * @brief Dispatch table for vector operations
 */
struct VxSIMDVectorOps {
    VxSIMDNormalizeVectorFn NormalizeVector;
    VxSIMDRotateVectorFn RotateVector;
    VxSIMDAddVectorFn Add;
    VxSIMDSubtractVectorFn Subtract;
    VxSIMDScaleVectorFn Scale;
    VxSIMDDotVectorFn Dot;
    VxSIMDCrossVectorFn Cross;
    VxSIMDLengthVectorFn Length;
    VxSIMDLengthSquaredVectorFn LengthSquared;
    VxSIMDDistanceVectorFn Distance;
    VxSIMDLerpVectorFn Lerp;
    VxSIMDReflectVectorFn Reflect;
    VxSIMDMinimizeVectorFn Minimize;
    VxSIMDMaximizeVectorFn Maximize;
};

/**
 * @brief Dispatch table for vector4 operations
 */
struct VxSIMDVector4Ops {
    VxSIMDAddVector4Fn Add;
    VxSIMDSubtractVector4Fn Subtract;
    VxSIMDScaleVector4Fn Scale;
    VxSIMDDotVector4Fn Dot;
    VxSIMDLerpVector4Fn Lerp;
};

/**
 * @brief Dispatch table for matrix operations
 */
struct VxSIMDMatrixOps {
    VxSIMDMultiplyMatrixFn MultiplyMatrix;
    VxSIMDMultiplyMatrix4Fn MultiplyMatrix4;
    VxSIMDTransposeMatrixFn TransposeMatrix;
    VxSIMDMultiplyMatrixVectorFn MultiplyMatrixVector;
    VxSIMDMultiplyMatrixVector4Fn MultiplyMatrixVector4;
    VxSIMDRotateVectorOpFn RotateVectorOp;
    VxSIMDMultiplyMatrixVectorManyFn MultiplyMatrixVectorMany;
    VxSIMDRotateVectorManyFn RotateVectorMany;
    VxSIMDMultiplyMatrixVectorStridedFn MultiplyMatrixVectorStrided;
    VxSIMDMultiplyMatrixVector4StridedFn MultiplyMatrixVector4Strided;
    VxSIMDRotateVectorStridedFn RotateVectorStrided;
    VxSIMDMatrixIdentityFn Identity;
    VxSIMDMatrixInverseFn Inverse;
    VxSIMDMatrixDeterminantFn Determinant;
    VxSIMDMatrixFromRotationFn FromAxisAngle;
    VxSIMDMatrixFromRotationOriginFn FromAxisAngleOrigin;
    VxSIMDMatrixFromEulerFn FromEulerAngles;
    VxSIMDMatrixToEulerFn ToEulerAngles;
    VxSIMDInterpolateMatrixFn Interpolate;
    VxSIMDInterpolateMatrixNoScaleFn InterpolateNoScale;
    VxSIMDDecomposeMatrixFn Decompose;
    VxSIMDDecomposeMatrixTotalFn DecomposeTotal;
    VxSIMDDecomposeMatrixTotalPtrFn DecomposeTotalPtr;
};

/**
 * @brief Dispatch table for quaternion operations
 */
struct VxSIMDQuaternionOps {
    VxSIMDNormalizeQuaternionFn NormalizeQuaternion;
    VxSIMDMultiplyQuaternionFn MultiplyQuaternion;
    VxSIMDSlerpQuaternionFn SlerpQuaternion;
    VxSIMDQuaternionFromMatrixFn FromMatrix;
    VxSIMDQuaternionToMatrixFn ToMatrix;
    VxSIMDQuaternionFromRotationFn FromAxisAngle;
    VxSIMDQuaternionFromEulerFn FromEulerAngles;
    VxSIMDQuaternionToEulerFn ToEulerAngles;
    VxSIMDQuaternionMultiplyInPlaceFn MultiplyInPlace;
    VxSIMDQuaternionConjugateFn Conjugate;
    VxSIMDQuaternionDivideFn Divide;
    VxSIMDQuaternionSnuggleFn Snuggle;
    VxSIMDQuaternionLnFn Ln;
    VxSIMDQuaternionExpFn Exp;
    VxSIMDQuaternionLnDifFn LnDif;
    VxSIMDQuaternionSquadFn Squad;
};

/**
 * @brief Dispatch table for pixel operations
 */
struct VxSIMDPixelOps {
    VxSIMDConvertPixelBatch32Fn ConvertPixelBatch32;
    VxSIMDApplyAlphaBatch32Fn ApplyAlphaBatch32;
    VxSIMDApplyVariableAlphaBatch32Fn ApplyVariableAlphaBatch32;
};

/**
 * @brief Dispatch table for ray operations
 */
struct VxSIMDRayOps {
    VxSIMDRayTransformFn Transform;
};

/**
 * @brief Dispatch table for plane operations
 */
struct VxSIMDPlaneOps {
    VxSIMDPlaneCreateFromPointFn CreateFromPoint;
    VxSIMDPlaneCreateFromTriangleFn CreateFromTriangle;
};

/**
 * @brief Dispatch table for rectangle operations
 */
struct VxSIMDRectOps {
    VxSIMDRectTransformFn Transform;
    VxSIMDRectTransformBySizeFn TransformBySize;
    VxSIMDRectTransformToHomogeneousFn TransformToHomogeneous;
    VxSIMDRectTransformFromHomogeneousFn TransformFromHomogeneous;
};

struct VxSIMDArrayOps {
    VxSIMDInterpolateFloatArrayFn InterpolateFloatArray;
    VxSIMDInterpolateVectorArrayFn InterpolateVectorArray;
};

struct VxSIMDGeometryOps {
    VxSIMDTransformBox2DFn TransformBox2D;
    VxSIMDProjectBoxZExtentsFn ProjectBoxZExtents;
    VxSIMDComputeBestFitBBoxFn ComputeBestFitBBox;
};

struct VxSIMDBboxOps {
    VxSIMDBboxClassifyFn Classify;
    VxSIMDBboxClassifyVerticesFn ClassifyVertices;
    VxSIMDBboxClassifyVerticesOneAxisFn ClassifyVerticesOneAxis;
    VxSIMDBboxTransformToFn TransformTo;
    VxSIMDBboxTransformFromFn TransformFrom;
};

struct VxSIMDFrustumOps {
    VxSIMDFrustumUpdateFn Update;
    VxSIMDFrustumComputeVerticesFn ComputeVertices;
    VxSIMDFrustumTransformFn Transform;
};

/**
 * @brief Master SIMD dispatch table containing all operation categories
 */
struct VxSIMDDispatch {
    VxSIMDVectorOps Vector;
    VxSIMDVector4Ops Vector4;
    VxSIMDMatrixOps Matrix;
    VxSIMDQuaternionOps Quaternion;
    VxSIMDRayOps Ray;
    VxSIMDPlaneOps Plane;
    VxSIMDRectOps Rect;
    VxSIMDPixelOps Pixel;
    VxSIMDArrayOps Array;
    VxSIMDGeometryOps Geometry;
    VxSIMDBboxOps Bbox;
    VxSIMDFrustumOps Frustum;
    const char *VariantName;
};

VX_EXPORT const VxSIMDDispatch *VxGetSIMDDispatch();
VX_EXPORT void VxResetSIMDDispatch();

#if defined(VX_SIMD_X86)
// SSE implementations
VX_EXPORT void VxSIMDNormalizeVector_SSE(VxVector *v);
VX_EXPORT void VxSIMDRotateVector_SSE(VxVector *result, const VxMatrix *mat, const VxVector *v);
VX_EXPORT void VxSIMDMultiplyMatrix_SSE(VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
VX_EXPORT void VxSIMDMultiplyMatrix4_SSE(VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
VX_EXPORT void VxSIMDTransposeMatrix_SSE(VxMatrix *result, const VxMatrix *a);
VX_EXPORT void VxSIMDMultiplyMatrixVector_SSE(VxVector *result, const VxMatrix *mat, const VxVector *v);
VX_EXPORT void VxSIMDMultiplyMatrixVector4_SSE(VxVector4 *result, const VxMatrix *mat, const VxVector4 *v);
VX_EXPORT void VxSIMDRotateVectorOp_SSE(VxVector *result, const VxMatrix *mat, const VxVector *v);
VX_EXPORT void VxSIMDMultiplyMatrixVectorMany_SSE(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride);
VX_EXPORT void VxSIMDRotateVectorMany_SSE(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride);
VX_EXPORT void VxSIMDNormalizeQuaternion_SSE(VxQuaternion *q);
VX_EXPORT void VxSIMDMultiplyQuaternion_SSE(VxQuaternion *result, const VxQuaternion *a, const VxQuaternion *b);
VX_EXPORT void VxSIMDSlerpQuaternion_SSE(VxQuaternion *result, float t, const VxQuaternion *a, const VxQuaternion *b);
VX_EXPORT int VxSIMDConvertPixelBatch32_SSE(const XULONG *srcPixels, XULONG *dstPixels, int count, const VxPixelSimdConfig &config);
VX_EXPORT int VxSIMDApplyAlphaBatch32_SSE(XULONG *pixels, int count, XBYTE alphaValue, XULONG alphaMask, XULONG alphaShift);
VX_EXPORT int VxSIMDApplyVariableAlphaBatch32_SSE(XULONG *pixels, const XBYTE *alphaValues, int count, XULONG alphaMask, XULONG alphaShift);
VX_EXPORT void VxSIMDInterpolateFloatArray_SSE(float *result, const float *a, const float *b, float factor, int count);
VX_EXPORT void VxSIMDInterpolateVectorArray_SSE(void *result, const void *a, const void *b, float factor, int count, XULONG strideResult, XULONG strideInput);
VX_EXPORT XBOOL VxSIMDTransformBox2D_SSE(const VxMatrix *worldProjection, const VxBbox *box, VxRect *screenSize, VxRect *extents, VXCLIP_FLAGS *orClipFlags, VXCLIP_FLAGS *andClipFlags);
VX_EXPORT void VxSIMDProjectBoxZExtents_SSE(const VxMatrix *worldProjection, const VxBbox *box, float *zhMin, float *zhMax);
VX_EXPORT XBOOL VxSIMDComputeBestFitBBox_SSE(const XBYTE *points, XULONG stride, int count, VxMatrix *bboxMatrix, float additionalBorder);
VX_EXPORT int VxSIMDBboxClassify_SSE(const VxBbox *self, const VxBbox *other, const VxVector *point);
VX_EXPORT void VxSIMDBboxClassifyVertices_SSE(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, XULONG *flags);
VX_EXPORT void VxSIMDBboxClassifyVerticesOneAxis_SSE(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, int axis, XULONG *flags);
VX_EXPORT void VxSIMDBboxTransformTo_SSE(const VxBbox *self, VxVector *points, const VxMatrix *mat);
VX_EXPORT void VxSIMDBboxTransformFrom_SSE(VxBbox *dest, const VxBbox *src, const VxMatrix *mat);
VX_EXPORT void VxSIMDFrustumUpdate_SSE(VxFrustum *frustum);
VX_EXPORT void VxSIMDFrustumComputeVertices_SSE(const VxFrustum *frustum, VxVector *vertices);
VX_EXPORT void VxSIMDFrustumTransform_SSE(VxFrustum *frustum, const VxMatrix *invWorldMat);

// Additional Vector operations
VX_EXPORT void VxSIMDAddVector_SSE(VxVector *result, const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDSubtractVector_SSE(VxVector *result, const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDScaleVector_SSE(VxVector *result, const VxVector *v, float scalar);
VX_EXPORT float VxSIMDDotVector_SSE(const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDCrossVector_SSE(VxVector *result, const VxVector *a, const VxVector *b);
VX_EXPORT float VxSIMDLengthVector_SSE(const VxVector *v);
VX_EXPORT float VxSIMDLengthSquaredVector_SSE(const VxVector *v);
VX_EXPORT float VxSIMDDistanceVector_SSE(const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDLerpVector_SSE(VxVector *result, const VxVector *a, const VxVector *b, float t);
VX_EXPORT void VxSIMDReflectVector_SSE(VxVector *result, const VxVector *incident, const VxVector *normal);
VX_EXPORT void VxSIMDMinimizeVector_SSE(VxVector *result, const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDMaximizeVector_SSE(VxVector *result, const VxVector *a, const VxVector *b);

// Vector4 operations
VX_EXPORT void VxSIMDAddVector4_SSE(VxVector4 *result, const VxVector4 *a, const VxVector4 *b);
VX_EXPORT void VxSIMDSubtractVector4_SSE(VxVector4 *result, const VxVector4 *a, const VxVector4 *b);
VX_EXPORT void VxSIMDScaleVector4_SSE(VxVector4 *result, const VxVector4 *v, float scalar);
VX_EXPORT float VxSIMDDotVector4_SSE(const VxVector4 *a, const VxVector4 *b);
VX_EXPORT void VxSIMDLerpVector4_SSE(VxVector4 *result, const VxVector4 *a, const VxVector4 *b, float t);

// Matrix strided operations
VX_EXPORT void VxSIMDMultiplyMatrixVectorStrided_SSE(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);
VX_EXPORT void VxSIMDMultiplyMatrixVector4Strided_SSE(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);
VX_EXPORT void VxSIMDRotateVectorStrided_SSE(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);

// Matrix utility operations
VX_EXPORT void VxSIMDMatrixIdentity_SSE(VxMatrix *mat);
VX_EXPORT void VxSIMDMatrixInverse_SSE(VxMatrix *result, const VxMatrix *mat);
VX_EXPORT float VxSIMDMatrixDeterminant_SSE(const VxMatrix *mat);
VX_EXPORT void VxSIMDMatrixFromRotation_SSE(VxMatrix *result, const VxVector *axis, float angle);
VX_EXPORT void VxSIMDMatrixFromRotationOrigin_SSE(VxMatrix *result, const VxVector *axis, const VxVector *origin, float angle);
VX_EXPORT void VxSIMDMatrixFromEuler_SSE(VxMatrix *result, float eax, float eay, float eaz);
VX_EXPORT void VxSIMDMatrixToEuler_SSE(const VxMatrix *mat, float *eax, float *eay, float *eaz);
VX_EXPORT void VxSIMDMatrixInterpolate_SSE(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
VX_EXPORT void VxSIMDMatrixInterpolateNoScale_SSE(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
VX_EXPORT void VxSIMDMatrixDecompose_SSE(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale);
VX_EXPORT float VxSIMDMatrixDecomposeTotal_SSE(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot);
VX_EXPORT float VxSIMDMatrixDecomposeTotalPtr_SSE(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot);

// Quaternion additional operations
VX_EXPORT void VxSIMDQuaternionFromMatrix_SSE(VxQuaternion *result, const VxMatrix *mat, XBOOL matIsUnit, XBOOL restoreMat);
VX_EXPORT void VxSIMDQuaternionToMatrix_SSE(VxMatrix *result, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionFromRotation_SSE(VxQuaternion *result, const VxVector *axis, float angle);
VX_EXPORT void VxSIMDQuaternionFromEuler_SSE(VxQuaternion *result, float eax, float eay, float eaz);
VX_EXPORT void VxSIMDQuaternionToEuler_SSE(const VxQuaternion *q, float *eax, float *eay, float *eaz);
VX_EXPORT void VxSIMDQuaternionMultiplyInPlace_SSE(VxQuaternion *self, const VxQuaternion *rhs);
VX_EXPORT void VxSIMDQuaternionConjugate_SSE(VxQuaternion *result, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionDivide_SSE(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionSnuggle_SSE(VxQuaternion *result, VxQuaternion *quat, VxVector *scale);
VX_EXPORT void VxSIMDQuaternionLn_SSE(VxQuaternion *result, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionExp_SSE(VxQuaternion *result, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionLnDif_SSE(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionSquad_SSE(VxQuaternion *result, float t, const VxQuaternion *quat1, const VxQuaternion *quat1Out, const VxQuaternion *quat2In, const VxQuaternion *quat2);

// Ray operations
VX_EXPORT void VxSIMDRayTransform_SSE(VxRay *dest, const VxRay *ray, const VxMatrix *mat);

// Plane operations
VX_EXPORT void VxSIMDPlaneCreateFromPoint_SSE(VxPlane *plane, const VxVector *normal, const VxVector *point);
VX_EXPORT void VxSIMDPlaneCreateFromTriangle_SSE(VxPlane *plane, const VxVector *a, const VxVector *b, const VxVector *c);

// Rect operations
VX_EXPORT void VxSIMDRectTransform_SSE(VxRect *rect, const VxRect *destScreen, const VxRect *srcScreen);
VX_EXPORT void VxSIMDRectTransformBySize_SSE(VxRect *rect, const Vx2DVector *destScreenSize, const Vx2DVector *srcScreenSize);
VX_EXPORT void VxSIMDRectTransformToHomogeneous_SSE(VxRect *rect, const VxRect *screen);
VX_EXPORT void VxSIMDRectTransformFromHomogeneous_SSE(VxRect *rect, const VxRect *screen);

// AVX+FMA implementations
VX_EXPORT void VxSIMDNormalizeVector_AVX(VxVector *v);
VX_EXPORT void VxSIMDRotateVector_AVX(VxVector *result, const VxMatrix *mat, const VxVector *v);
VX_EXPORT void VxSIMDMultiplyMatrix_AVX(VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
VX_EXPORT void VxSIMDMultiplyMatrix4_AVX(VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
VX_EXPORT void VxSIMDTransposeMatrix_AVX(VxMatrix *result, const VxMatrix *a);
VX_EXPORT void VxSIMDMultiplyMatrixVector_AVX(VxVector *result, const VxMatrix *mat, const VxVector *v);
VX_EXPORT void VxSIMDMultiplyMatrixVector4_AVX(VxVector4 *result, const VxMatrix *mat, const VxVector4 *v);
VX_EXPORT void VxSIMDRotateVectorOp_AVX(VxVector *result, const VxMatrix *mat, const VxVector *v);
VX_EXPORT void VxSIMDMultiplyMatrixVectorMany_AVX(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride);
VX_EXPORT void VxSIMDRotateVectorMany_AVX(VxVector *results, const VxMatrix *mat, const VxVector *vectors, int count, int stride);
VX_EXPORT void VxSIMDNormalizeQuaternion_AVX(VxQuaternion *q);
VX_EXPORT void VxSIMDMultiplyQuaternion_AVX(VxQuaternion *result, const VxQuaternion *a, const VxQuaternion *b);
VX_EXPORT void VxSIMDSlerpQuaternion_AVX(VxQuaternion *result, float t, const VxQuaternion *a, const VxQuaternion *b);
VX_EXPORT int VxSIMDConvertPixelBatch32_AVX(const XULONG *srcPixels, XULONG *dstPixels, int count, const VxPixelSimdConfig &config);
VX_EXPORT int VxSIMDApplyAlphaBatch32_AVX(XULONG *pixels, int count, XBYTE alphaValue, XULONG alphaMask, XULONG alphaShift);
VX_EXPORT int VxSIMDApplyVariableAlphaBatch32_AVX(XULONG *pixels, const XBYTE *alphaValues, int count, XULONG alphaMask, XULONG alphaShift);
VX_EXPORT void VxSIMDInterpolateFloatArray_AVX(float *result, const float *a, const float *b, float factor, int count);
VX_EXPORT void VxSIMDInterpolateVectorArray_AVX(void *result, const void *a, const void *b, float factor, int count, XULONG strideResult, XULONG strideInput);
VX_EXPORT XBOOL VxSIMDTransformBox2D_AVX(const VxMatrix *worldProjection, const VxBbox *box, VxRect *screenSize, VxRect *extents, VXCLIP_FLAGS *orClipFlags, VXCLIP_FLAGS *andClipFlags);
VX_EXPORT void VxSIMDProjectBoxZExtents_AVX(const VxMatrix *worldProjection, const VxBbox *box, float *zhMin, float *zhMax);
VX_EXPORT XBOOL VxSIMDComputeBestFitBBox_AVX(const XBYTE *points, XULONG stride, int count, VxMatrix *bboxMatrix, float additionalBorder);
VX_EXPORT int VxSIMDBboxClassify_AVX(const VxBbox *self, const VxBbox *other, const VxVector *point);
VX_EXPORT void VxSIMDBboxClassifyVertices_AVX(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, XULONG *flags);
VX_EXPORT void VxSIMDBboxClassifyVerticesOneAxis_AVX(const VxBbox *self, int count, const XBYTE *vertices, XULONG stride, int axis, XULONG *flags);
VX_EXPORT void VxSIMDBboxTransformTo_AVX(const VxBbox *self, VxVector *points, const VxMatrix *mat);
VX_EXPORT void VxSIMDBboxTransformFrom_AVX(VxBbox *dest, const VxBbox *src, const VxMatrix *mat);
VX_EXPORT void VxSIMDFrustumUpdate_AVX(VxFrustum *frustum);
VX_EXPORT void VxSIMDFrustumComputeVertices_AVX(const VxFrustum *frustum, VxVector *vertices);
VX_EXPORT void VxSIMDFrustumTransform_AVX(VxFrustum *frustum, const VxMatrix *invWorldMat);

// Additional Vector operations
VX_EXPORT void VxSIMDAddVector_AVX(VxVector *result, const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDSubtractVector_AVX(VxVector *result, const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDScaleVector_AVX(VxVector *result, const VxVector *v, float scalar);
VX_EXPORT float VxSIMDDotVector_AVX(const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDCrossVector_AVX(VxVector *result, const VxVector *a, const VxVector *b);
VX_EXPORT float VxSIMDLengthVector_AVX(const VxVector *v);
VX_EXPORT float VxSIMDLengthSquaredVector_AVX(const VxVector *v);
VX_EXPORT float VxSIMDDistanceVector_AVX(const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDLerpVector_AVX(VxVector *result, const VxVector *a, const VxVector *b, float t);
VX_EXPORT void VxSIMDReflectVector_AVX(VxVector *result, const VxVector *incident, const VxVector *normal);
VX_EXPORT void VxSIMDMinimizeVector_AVX(VxVector *result, const VxVector *a, const VxVector *b);
VX_EXPORT void VxSIMDMaximizeVector_AVX(VxVector *result, const VxVector *a, const VxVector *b);

// Vector4 operations
VX_EXPORT void VxSIMDAddVector4_AVX(VxVector4 *result, const VxVector4 *a, const VxVector4 *b);
VX_EXPORT void VxSIMDSubtractVector4_AVX(VxVector4 *result, const VxVector4 *a, const VxVector4 *b);
VX_EXPORT void VxSIMDScaleVector4_AVX(VxVector4 *result, const VxVector4 *v, float scalar);
VX_EXPORT float VxSIMDDotVector4_AVX(const VxVector4 *a, const VxVector4 *b);
VX_EXPORT void VxSIMDLerpVector4_AVX(VxVector4 *result, const VxVector4 *a, const VxVector4 *b, float t);

// Matrix strided operations
VX_EXPORT void VxSIMDMultiplyMatrixVectorStrided_AVX(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);
VX_EXPORT void VxSIMDMultiplyMatrixVector4Strided_AVX(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);
VX_EXPORT void VxSIMDRotateVectorStrided_AVX(VxStridedData *dest, VxStridedData *src, const VxMatrix *mat, int count);

// Matrix utility operations
VX_EXPORT void VxSIMDMatrixIdentity_AVX(VxMatrix *mat);
VX_EXPORT void VxSIMDMatrixInverse_AVX(VxMatrix *result, const VxMatrix *mat);
VX_EXPORT float VxSIMDMatrixDeterminant_AVX(const VxMatrix *mat);
VX_EXPORT void VxSIMDMatrixFromRotation_AVX(VxMatrix *result, const VxVector *axis, float angle);
VX_EXPORT void VxSIMDMatrixFromRotationOrigin_AVX(VxMatrix *result, const VxVector *axis, const VxVector *origin, float angle);
VX_EXPORT void VxSIMDMatrixFromEuler_AVX(VxMatrix *result, float eax, float eay, float eaz);
VX_EXPORT void VxSIMDMatrixToEuler_AVX(const VxMatrix *mat, float *eax, float *eay, float *eaz);
VX_EXPORT void VxSIMDMatrixInterpolate_AVX(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
VX_EXPORT void VxSIMDMatrixInterpolateNoScale_AVX(float step, VxMatrix *result, const VxMatrix *a, const VxMatrix *b);
VX_EXPORT void VxSIMDMatrixDecompose_AVX(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale);
VX_EXPORT float VxSIMDMatrixDecomposeTotal_AVX(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot);
VX_EXPORT float VxSIMDMatrixDecomposeTotalPtr_AVX(const VxMatrix *mat, VxQuaternion *quat, VxVector *pos, VxVector *scale, VxQuaternion *uRot);

// Quaternion additional operations
VX_EXPORT void VxSIMDQuaternionFromMatrix_AVX(VxQuaternion *result, const VxMatrix *mat, XBOOL matIsUnit, XBOOL restoreMat);
VX_EXPORT void VxSIMDQuaternionToMatrix_AVX(VxMatrix *result, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionFromRotation_AVX(VxQuaternion *result, const VxVector *axis, float angle);
VX_EXPORT void VxSIMDQuaternionFromEuler_AVX(VxQuaternion *result, float eax, float eay, float eaz);
VX_EXPORT void VxSIMDQuaternionToEuler_AVX(const VxQuaternion *q, float *eax, float *eay, float *eaz);
VX_EXPORT void VxSIMDQuaternionMultiplyInPlace_AVX(VxQuaternion *self, const VxQuaternion *rhs);
VX_EXPORT void VxSIMDQuaternionConjugate_AVX(VxQuaternion *result, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionDivide_AVX(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionSnuggle_AVX(VxQuaternion *result, VxQuaternion *quat, VxVector *scale);
VX_EXPORT void VxSIMDQuaternionLn_AVX(VxQuaternion *result, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionExp_AVX(VxQuaternion *result, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionLnDif_AVX(VxQuaternion *result, const VxQuaternion *p, const VxQuaternion *q);
VX_EXPORT void VxSIMDQuaternionSquad_AVX(VxQuaternion *result, float t, const VxQuaternion *quat1, const VxQuaternion *quat1Out, const VxQuaternion *quat2In, const VxQuaternion *quat2);

// Ray operations
VX_EXPORT void VxSIMDRayTransform_AVX(VxRay *dest, const VxRay *ray, const VxMatrix *mat);

// Plane operations
VX_EXPORT void VxSIMDPlaneCreateFromPoint_AVX(VxPlane *plane, const VxVector *normal, const VxVector *point);
VX_EXPORT void VxSIMDPlaneCreateFromTriangle_AVX(VxPlane *plane, const VxVector *a, const VxVector *b, const VxVector *c);

// Rect operations
VX_EXPORT void VxSIMDRectTransform_AVX(VxRect *rect, const VxRect *destScreen, const VxRect *srcScreen);
VX_EXPORT void VxSIMDRectTransformBySize_AVX(VxRect *rect, const Vx2DVector *destScreenSize, const Vx2DVector *srcScreenSize);
VX_EXPORT void VxSIMDRectTransformToHomogeneous_AVX(VxRect *rect, const VxRect *screen);
VX_EXPORT void VxSIMDRectTransformFromHomogeneous_AVX(VxRect *rect, const VxRect *screen);

#endif

// ============================================================================
// SIMD Utility Functions
// ============================================================================

#if defined(VX_SIMD_SSE)

/**
 * @brief Loads 3 floats into a SIMD register (w component is undefined)
 */
VX_SIMD_INLINE __m128 VxSIMDLoadFloat3(const float *ptr) {
    // Load x,y as one 64-bit chunk + z as scalar.
    // Use memcpy to avoid strict-aliasing UB from type-punning `float*`.
    uint64_t xy_u64;
    std::memcpy(&xy_u64, ptr, sizeof(xy_u64));
    const __m128i xy_i = _mm_cvtsi64_si128(static_cast<long long>(xy_u64));
    const __m128 xy = _mm_castsi128_ps(xy_i);
    const __m128 z = _mm_set_ss(ptr[2]);
    return _mm_movelh_ps(xy, z); // [x y z 0]
}

/**
 * @brief Stores the first 3 components of a SIMD register
 */
VX_SIMD_INLINE void VxSIMDStoreFloat3(float *ptr, __m128 v) {
    _mm_store_ss(&ptr[0], v);
    _mm_store_ss(&ptr[1], _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)));
    _mm_store_ss(&ptr[2], _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)));
}

/**
 * @brief Loads 4 floats into a SIMD register
 */
VX_SIMD_INLINE __m128 VxSIMDLoadFloat4(const float *ptr) {
    return _mm_loadu_ps(ptr);
}

/**
 * @brief Stores 4 floats from a SIMD register
 */
VX_SIMD_INLINE void VxSIMDStoreFloat4(float *ptr, __m128 v) {
    _mm_storeu_ps(ptr, v);
}

/**
 * @brief Loads 4 aligned floats into a SIMD register
 */
VX_SIMD_INLINE __m128 VxSIMDLoadFloat4Aligned(const float *ptr) {
    return _mm_load_ps(ptr);
}

/**
 * @brief Stores 4 aligned floats from a SIMD register
 */
VX_SIMD_INLINE void VxSIMDStoreFloat4Aligned(float *ptr, __m128 v) {
    _mm_store_ps(ptr, v);
}

/**
 * @brief Computes dot product of two 3D vectors (result in all components)
 */
VX_SIMD_INLINE __m128 VxSIMDDotProduct3(__m128 a, __m128 b) {
#if defined(VX_SIMD_SSE4_1)
    return _mm_dp_ps(a, b, 0x7F);
#else
    __m128 mul = _mm_mul_ps(a, b);
    __m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 1, 0, 3));
    __m128 sums = _mm_add_ps(mul, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(0, 0, 0, 0));
#endif
}

/**
 * @brief Computes dot product of two 4D vectors (result in all components)
 */
VX_SIMD_INLINE __m128 VxSIMDDotProduct4(__m128 a, __m128 b) {
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

/**
 * @brief Computes cross product of two 3D vectors
 */
VX_SIMD_INLINE __m128 VxSIMDCrossProduct3(__m128 a, __m128 b) {
    __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));
    return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
}

/**
 * @brief Computes reciprocal square root (fast approximation)
 */
VX_SIMD_INLINE __m128 VxSIMDReciprocalSqrt(__m128 v) {
    return _mm_rsqrt_ps(v);
}

/**
 * @brief Computes reciprocal square root (accurate)
 */
VX_SIMD_INLINE __m128 VxSIMDReciprocalSqrtAccurate(__m128 v) {
    __m128 rsqrt = _mm_rsqrt_ps(v);
    // Newton-Raphson iteration: rsqrt * (1.5 - 0.5 * v * rsqrt * rsqrt)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_half = _mm_set1_ps(1.5f);
    return _mm_mul_ps(rsqrt, _mm_sub_ps(three_half, _mm_mul_ps(_mm_mul_ps(half, v), _mm_mul_ps(rsqrt, rsqrt))));
}

/**
 * @brief Normalizes a 3D vector
 */
VX_SIMD_INLINE __m128 VxSIMDNormalize3(__m128 v) {
    __m128 dot = VxSIMDDotProduct3(v, v);
    __m128 epsilon = _mm_set1_ps(EPSILON);
    __m128 mask = _mm_cmpgt_ps(dot, epsilon);
    __m128 safeDot = _mm_max_ps(dot, epsilon);
    __m128 invLen = VxSIMDReciprocalSqrtAccurate(safeDot);
    __m128 normalized = _mm_mul_ps(v, invLen);
    __m128 keepOriginal = _mm_andnot_ps(mask, v);
    __m128 useNormalized = _mm_and_ps(mask, normalized);
    return _mm_or_ps(keepOriginal, useNormalized);
}

/**
 * @brief Normalizes a 4D vector
 */
VX_SIMD_INLINE __m128 VxSIMDNormalize4(__m128 v) {
    __m128 dot = VxSIMDDotProduct4(v, v);
    __m128 invLen = VxSIMDReciprocalSqrtAccurate(dot);
    return _mm_mul_ps(v, invLen);
}

/**
 * @brief Performs matrix-vector multiplication for 3D vector (with translation)
 */
VX_SIMD_INLINE __m128 VxSIMDMatrixMultiplyVector3(const float *mat, __m128 v) {
    __m128 r0 = _mm_loadu_ps(&mat[0]);
    __m128 r1 = _mm_loadu_ps(&mat[4]);
    __m128 r2 = _mm_loadu_ps(&mat[8]);
    __m128 r3 = _mm_loadu_ps(&mat[12]);

    __m128 v_x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(r0, v_x), _mm_mul_ps(r1, v_y)), _mm_add_ps(_mm_mul_ps(r2, v_z), r3));
}

/**
 * @brief Performs matrix rotation for 3D vector (no translation)
 */
VX_SIMD_INLINE __m128 VxSIMDMatrixRotateVector3(const float *mat, __m128 v) {
    __m128 r0 = _mm_loadu_ps(&mat[0]);
    __m128 r1 = _mm_loadu_ps(&mat[4]);
    __m128 r2 = _mm_loadu_ps(&mat[8]);

    __m128 v_x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v_y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v_z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

    return _mm_add_ps(_mm_add_ps(_mm_mul_ps(r0, v_x), _mm_mul_ps(r1, v_y)), _mm_mul_ps(r2, v_z));
}

#endif // VX_SIMD_SSE

#endif // VXSIMD_H
