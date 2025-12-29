#include "VxSIMD.h"

#include <string>
#include <sstream>

#if defined(VX_SIMD_X86)
    #if defined(VX_SIMD_MSVC)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#endif

#if defined(VX_SIMD_X86)

/**
 * @brief Executes CPUID instruction to query CPU features
 */
static void RunCPUID(int info[4], int function_id, int subfunction_id = 0) {
#if defined(VX_SIMD_MSVC)
    __cpuidex(info, function_id, subfunction_id);
#elif defined(VX_SIMD_GCC)
    __cpuid_count(function_id, subfunction_id, info[0], info[1], info[2], info[3]);
#endif
}

static unsigned long long QueryXCR0() {
#if defined(VX_SIMD_MSVC)
    return _xgetbv(0);
#elif defined(VX_SIMD_GCC)
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#else
    return 0ull;
#endif
}

static bool SupportsExtendedYmmState(const VxSIMDFeatures &features) {
    if (!features.OSXSAVE || !features.XSAVE) {
        return false;
    }

    const unsigned long long xcr0 = QueryXCR0();
    // Require SSE (bit 1) and AVX (bit 2) state saving.
    return (xcr0 & 0x6ull) == 0x6ull;
}

VxSIMDFeatures VxDetectSIMDFeatures() {
    VxSIMDFeatures features;

    int info[4];

    // Get maximum supported function ID
    RunCPUID(info, 0, 0);
    int max_id = info[0];

    if (max_id >= 1) {
        // Query feature flags from function 1
        RunCPUID(info, 1, 0);

        // ECX register (info[2])
        features.SSE3   = (info[2] & (1 << 0)) != 0;
        features.SSSE3  = (info[2] & (1 << 9)) != 0;
        features.FMA    = (info[2] & (1 << 12)) != 0;
        features.SSE4_1 = (info[2] & (1 << 19)) != 0;
        features.SSE4_2 = (info[2] & (1 << 20)) != 0;
        features.XSAVE  = (info[2] & (1 << 26)) != 0;
        features.OSXSAVE= (info[2] & (1 << 27)) != 0;
        features.AVX    = (info[2] & (1 << 28)) != 0;

        // EDX register (info[3])
        features.SSE    = (info[3] & (1 << 25)) != 0;
        features.SSE2   = (info[3] & (1 << 26)) != 0;
    }

    if (max_id >= 7) {
        // Query extended features from function 7
        RunCPUID(info, 7, 0);

        // EBX register (info[1])
        features.AVX2    = (info[1] & (1 << 5)) != 0;
        features.AVX512F = (info[1] & (1 << 16)) != 0;
    }

    return features;
}

#elif defined(VX_SIMD_ARM)

VxSIMDFeatures VxDetectSIMDFeatures() {
    VxSIMDFeatures features;
    // NEON support has been removed
    return features;
}

#else

VxSIMDFeatures VxDetectSIMDFeatures() {
    return VxSIMDFeatures(); // No SIMD support
}

#endif

const VxSIMDFeatures& VxGetSIMDFeatures() {
    static VxSIMDFeatures features = VxDetectSIMDFeatures();
    return features;
}

// ============================================================================
// SIMD Dispatch Implementation
// ============================================================================

// Global dispatch table instance cached after first initialization.
static VxSIMDDispatch g_DispatchTable;
static bool g_DispatchInitialized = false;

using DispatchInitFn = void (*)(VxSIMDDispatch &, const char *);

// ============================================================================
// Template-based Dispatch Table Configuration
// ============================================================================

/**
 * @brief Trait struct to select SIMD implementation variant at compile-time
 * 
 * Each specialization provides static function pointers for a specific SIMD variant.
 * The default (Scalar) implementation is defined here, with specialized versions
 * for SSE, AVX, NEON, etc.
 */
template<typename Variant>
struct SIMDFunctions {};


/**
 * @brief Template function to configure the dispatch table with a specific SIMD variant
 * 
 * This template eliminates repetitive assignment code by using compile-time
 * function pointer resolution from the SIMDFunctions trait.
 */
template <typename Variant>
static void ConfigureDispatch(VxSIMDDispatch &table, const char *variantName) {
    using F = SIMDFunctions<Variant>;

    // Vector operations
    table.Vector.NormalizeVector = F::NormalizeVector;
    table.Vector.RotateVector = F::RotateVector;
    table.Vector.Add = F::AddVector;
    table.Vector.Subtract = F::SubtractVector;
    table.Vector.Scale = F::ScaleVector;
    table.Vector.Dot = F::DotVector;
    table.Vector.Cross = F::CrossVector;
    table.Vector.Length = F::LengthVector;
    table.Vector.LengthSquared = F::LengthSquaredVector;
    table.Vector.Distance = F::DistanceVector;
    table.Vector.Lerp = F::LerpVector;
    table.Vector.Reflect = F::ReflectVector;
    table.Vector.Minimize = F::MinimizeVector;
    table.Vector.Maximize = F::MaximizeVector;

    // Vector4 operations
    table.Vector4.Add = F::AddVector4;
    table.Vector4.Subtract = F::SubtractVector4;
    table.Vector4.Scale = F::ScaleVector4;
    table.Vector4.Dot = F::DotVector4;
    table.Vector4.Lerp = F::LerpVector4;

    // Matrix operations
    table.Matrix.MultiplyMatrix = F::MultiplyMatrix;
    table.Matrix.MultiplyMatrix4 = F::MultiplyMatrix4;
    table.Matrix.TransposeMatrix = F::TransposeMatrix;
    table.Matrix.MultiplyMatrixVector = F::MultiplyMatrixVector;
    table.Matrix.MultiplyMatrixVector4 = F::MultiplyMatrixVector4;
    table.Matrix.RotateVectorOp = F::RotateVectorOp;
    table.Matrix.MultiplyMatrixVectorMany = F::MultiplyMatrixVectorMany;
    table.Matrix.RotateVectorMany = F::RotateVectorMany;
    table.Matrix.MultiplyMatrixVectorStrided = F::MultiplyMatrixVectorStrided;
    table.Matrix.MultiplyMatrixVector4Strided = F::MultiplyMatrixVector4Strided;
    table.Matrix.RotateVectorStrided = F::RotateVectorStrided;
    table.Matrix.Identity = F::MatrixIdentity;
    table.Matrix.Inverse = F::MatrixInverse;
    table.Matrix.Determinant = F::MatrixDeterminant;
    table.Matrix.FromAxisAngle = F::MatrixFromRotation;
    table.Matrix.FromAxisAngleOrigin = F::MatrixFromRotationOrigin;
    table.Matrix.FromEulerAngles = F::MatrixFromEuler;
    table.Matrix.ToEulerAngles = F::MatrixToEuler;
    table.Matrix.Interpolate = F::MatrixInterpolate;
    table.Matrix.InterpolateNoScale = F::MatrixInterpolateNoScale;
    table.Matrix.Decompose = F::MatrixDecompose;
    table.Matrix.DecomposeTotal = F::MatrixDecomposeTotal;
    table.Matrix.DecomposeTotalPtr = F::MatrixDecomposeTotalPtr;

    // Quaternion operations
    table.Quaternion.NormalizeQuaternion = F::NormalizeQuaternion;
    table.Quaternion.MultiplyQuaternion = F::MultiplyQuaternion;
    table.Quaternion.SlerpQuaternion = F::SlerpQuaternion;
    table.Quaternion.FromMatrix = F::QuaternionFromMatrix;
    table.Quaternion.ToMatrix = F::QuaternionToMatrix;
    table.Quaternion.FromAxisAngle = F::QuaternionFromRotation;
    table.Quaternion.FromEulerAngles = F::QuaternionFromEuler;
    table.Quaternion.ToEulerAngles = F::QuaternionToEuler;
    table.Quaternion.MultiplyInPlace = F::QuaternionMultiplyInPlace;
    table.Quaternion.Conjugate = F::QuaternionConjugate;
    table.Quaternion.Divide = F::QuaternionDivide;
    table.Quaternion.Snuggle = F::QuaternionSnuggle;
    table.Quaternion.Ln = F::QuaternionLn;
    table.Quaternion.Exp = F::QuaternionExp;
    table.Quaternion.LnDif = F::QuaternionLnDif;
    table.Quaternion.Squad = F::QuaternionSquad;

    // Ray operations
    table.Ray.Transform = F::RayTransform;

    // Plane operations
    table.Plane.CreateFromPoint = F::PlaneCreateFromPoint;
    table.Plane.CreateFromTriangle = F::PlaneCreateFromTriangle;

    // Rect operations
    table.Rect.Transform = F::RectTransform;
    table.Rect.TransformBySize = F::RectTransformBySize;
    table.Rect.TransformToHomogeneous = F::RectTransformToHomogeneous;
    table.Rect.TransformFromHomogeneous = F::RectTransformFromHomogeneous;

    // Array operations
    table.Array.InterpolateFloatArray = F::InterpolateFloatArray;
    table.Array.InterpolateVectorArray = F::InterpolateVectorArray;

    // Geometry operations
    table.Geometry.TransformBox2D = F::TransformBox2D;
    table.Geometry.ProjectBoxZExtents = F::ProjectBoxZExtents;
    table.Geometry.ComputeBestFitBBox = F::ComputeBestFitBBox;

    // Bbox operations
    table.Bbox.Classify = F::BboxClassify;
    table.Bbox.ClassifyVertices = F::BboxClassifyVertices;
    table.Bbox.ClassifyVerticesOneAxis = F::BboxClassifyVerticesOneAxis;
    table.Bbox.TransformTo = F::BboxTransformTo;
    table.Bbox.TransformFrom = F::BboxTransformFrom;

    // Frustum operations
    table.Frustum.Update = F::FrustumUpdate;
    table.Frustum.ComputeVertices = F::FrustumComputeVertices;
    table.Frustum.Transform = F::FrustumTransform;

    table.VariantName = variantName;
}

// ============================================================================
// SIMD Variant Specializations
// ============================================================================

#if defined(VX_SIMD_X86) && defined(VX_SIMD_SSE)
// SSE variant tag
struct SSEVariant {};

template<>
struct SIMDFunctions<SSEVariant> {
    // All operations use SSE implementations - no scalar fallbacks
    static constexpr auto NormalizeVector = &VxSIMDNormalizeVector_SSE;
    static constexpr auto RotateVector = &VxSIMDRotateVector_SSE;
    static constexpr auto AddVector = &VxSIMDAddVector_SSE;
    static constexpr auto SubtractVector = &VxSIMDSubtractVector_SSE;
    static constexpr auto ScaleVector = &VxSIMDScaleVector_SSE;
    static constexpr auto DotVector = &VxSIMDDotVector_SSE;
    static constexpr auto CrossVector = &VxSIMDCrossVector_SSE;
    static constexpr auto LengthVector = &VxSIMDLengthVector_SSE;
    static constexpr auto LengthSquaredVector = &VxSIMDLengthSquaredVector_SSE;
    static constexpr auto DistanceVector = &VxSIMDDistanceVector_SSE;
    static constexpr auto LerpVector = &VxSIMDLerpVector_SSE;
    static constexpr auto ReflectVector = &VxSIMDReflectVector_SSE;
    static constexpr auto MinimizeVector = &VxSIMDMinimizeVector_SSE;
    static constexpr auto MaximizeVector = &VxSIMDMaximizeVector_SSE;

    static constexpr auto AddVector4 = &VxSIMDAddVector4_SSE;
    static constexpr auto SubtractVector4 = &VxSIMDSubtractVector4_SSE;
    static constexpr auto ScaleVector4 = &VxSIMDScaleVector4_SSE;
    static constexpr auto DotVector4 = &VxSIMDDotVector4_SSE;
    static constexpr auto LerpVector4 = &VxSIMDLerpVector4_SSE;

    static constexpr auto MultiplyMatrix = &VxSIMDMultiplyMatrix_SSE;
    static constexpr auto MultiplyMatrix4 = &VxSIMDMultiplyMatrix4_SSE;
    static constexpr auto TransposeMatrix = &VxSIMDTransposeMatrix_SSE;
    static constexpr auto MultiplyMatrixVector = &VxSIMDMultiplyMatrixVector_SSE;
    static constexpr auto MultiplyMatrixVector4 = &VxSIMDMultiplyMatrixVector4_SSE;
    static constexpr auto RotateVectorOp = &VxSIMDRotateVectorOp_SSE;
    static constexpr auto MultiplyMatrixVectorMany = &VxSIMDMultiplyMatrixVectorMany_SSE;
    static constexpr auto RotateVectorMany = &VxSIMDRotateVectorMany_SSE;
    static constexpr auto MultiplyMatrixVectorStrided = &VxSIMDMultiplyMatrixVectorStrided_SSE;
    static constexpr auto MultiplyMatrixVector4Strided = &VxSIMDMultiplyMatrixVector4Strided_SSE;
    static constexpr auto RotateVectorStrided = &VxSIMDRotateVectorStrided_SSE;
    static constexpr auto MatrixIdentity = &VxSIMDMatrixIdentity_SSE;
    static constexpr auto MatrixInverse = &VxSIMDMatrixInverse_SSE;
    static constexpr auto MatrixDeterminant = &VxSIMDMatrixDeterminant_SSE;
    static constexpr auto MatrixFromRotation = &VxSIMDMatrixFromRotation_SSE;
    static constexpr auto MatrixFromRotationOrigin = &VxSIMDMatrixFromRotationOrigin_SSE;
    static constexpr auto MatrixFromEuler = &VxSIMDMatrixFromEuler_SSE;
    static constexpr auto MatrixToEuler = &VxSIMDMatrixToEuler_SSE;
    static constexpr auto MatrixInterpolate = &VxSIMDMatrixInterpolate_SSE;
    static constexpr auto MatrixInterpolateNoScale = &VxSIMDMatrixInterpolateNoScale_SSE;
    static constexpr auto MatrixDecompose = &VxSIMDMatrixDecompose_SSE;
    static constexpr auto MatrixDecomposeTotal = &VxSIMDMatrixDecomposeTotal_SSE;
    static constexpr auto MatrixDecomposeTotalPtr = &VxSIMDMatrixDecomposeTotalPtr_SSE;

    static constexpr auto NormalizeQuaternion = &VxSIMDNormalizeQuaternion_SSE;
    static constexpr auto MultiplyQuaternion = &VxSIMDMultiplyQuaternion_SSE;
    static constexpr auto SlerpQuaternion = &VxSIMDSlerpQuaternion_SSE;
    static constexpr auto QuaternionFromMatrix = &VxSIMDQuaternionFromMatrix_SSE;
    static constexpr auto QuaternionToMatrix = &VxSIMDQuaternionToMatrix_SSE;
    static constexpr auto QuaternionFromRotation = &VxSIMDQuaternionFromRotation_SSE;
    static constexpr auto QuaternionFromEuler = &VxSIMDQuaternionFromEuler_SSE;
    static constexpr auto QuaternionToEuler = &VxSIMDQuaternionToEuler_SSE;
    static constexpr auto QuaternionMultiplyInPlace = &VxSIMDQuaternionMultiplyInPlace_SSE;
    static constexpr auto QuaternionConjugate = &VxSIMDQuaternionConjugate_SSE;
    static constexpr auto QuaternionDivide = &VxSIMDQuaternionDivide_SSE;
    static constexpr auto QuaternionSnuggle = &VxSIMDQuaternionSnuggle_SSE;
    static constexpr auto QuaternionLn = &VxSIMDQuaternionLn_SSE;
    static constexpr auto QuaternionExp = &VxSIMDQuaternionExp_SSE;
    static constexpr auto QuaternionLnDif = &VxSIMDQuaternionLnDif_SSE;
    static constexpr auto QuaternionSquad = &VxSIMDQuaternionSquad_SSE;

    static constexpr auto RayTransform = &VxSIMDRayTransform_SSE;

    static constexpr auto PlaneCreateFromPoint = &VxSIMDPlaneCreateFromPoint_SSE;
    static constexpr auto PlaneCreateFromTriangle = &VxSIMDPlaneCreateFromTriangle_SSE;

    static constexpr auto RectTransform = &VxSIMDRectTransform_SSE;
    static constexpr auto RectTransformBySize = &VxSIMDRectTransformBySize_SSE;
    static constexpr auto RectTransformToHomogeneous = &VxSIMDRectTransformToHomogeneous_SSE;
    static constexpr auto RectTransformFromHomogeneous = &VxSIMDRectTransformFromHomogeneous_SSE;

    static constexpr auto InterpolateFloatArray = &VxSIMDInterpolateFloatArray_SSE;
    static constexpr auto InterpolateVectorArray = &VxSIMDInterpolateVectorArray_SSE;

    static constexpr auto TransformBox2D = &VxSIMDTransformBox2D_SSE;
    static constexpr auto ProjectBoxZExtents = &VxSIMDProjectBoxZExtents_SSE;
    static constexpr auto ComputeBestFitBBox = &VxSIMDComputeBestFitBBox_SSE;

    static constexpr auto BboxClassify = &VxSIMDBboxClassify_SSE;
    static constexpr auto BboxClassifyVertices = &VxSIMDBboxClassifyVertices_SSE;
    static constexpr auto BboxClassifyVerticesOneAxis = &VxSIMDBboxClassifyVerticesOneAxis_SSE;
    static constexpr auto BboxTransformTo = &VxSIMDBboxTransformTo_SSE;
    static constexpr auto BboxTransformFrom = &VxSIMDBboxTransformFrom_SSE;

    static constexpr auto FrustumUpdate = &VxSIMDFrustumUpdate_SSE;
    static constexpr auto FrustumComputeVertices = &VxSIMDFrustumComputeVertices_SSE;
    static constexpr auto FrustumTransform = &VxSIMDFrustumTransform_SSE;
};

static void ConfigureSSEDispatch(VxSIMDDispatch &table, const char *variantName) {
    ConfigureDispatch<SSEVariant>(table, variantName);
}
#endif

#if defined(VX_BUILD_AVX)
// AVX variant tag
struct AVXVariant {};

template<>
struct SIMDFunctions<AVXVariant> {
    // SIMD-optimized operations
    static constexpr auto NormalizeVector = &VxSIMDNormalizeVector_AVX;
    static constexpr auto RotateVector = &VxSIMDRotateVector_AVX;
    static constexpr auto MultiplyMatrix = &VxSIMDMultiplyMatrix_AVX;
    static constexpr auto MultiplyMatrix4 = &VxSIMDMultiplyMatrix4_AVX;
    static constexpr auto TransposeMatrix = &VxSIMDTransposeMatrix_AVX;
    static constexpr auto MultiplyMatrixVector = &VxSIMDMultiplyMatrixVector_AVX;
    static constexpr auto MultiplyMatrixVector4 = &VxSIMDMultiplyMatrixVector4_AVX;
    static constexpr auto RotateVectorOp = &VxSIMDRotateVectorOp_AVX;
    static constexpr auto MultiplyMatrixVectorMany = &VxSIMDMultiplyMatrixVectorMany_AVX;
    static constexpr auto RotateVectorMany = &VxSIMDRotateVectorMany_AVX;
    static constexpr auto NormalizeQuaternion = &VxSIMDNormalizeQuaternion_AVX;
    static constexpr auto MultiplyQuaternion = &VxSIMDMultiplyQuaternion_AVX;
    static constexpr auto SlerpQuaternion = &VxSIMDSlerpQuaternion_AVX;
    static constexpr auto InterpolateFloatArray = &VxSIMDInterpolateFloatArray_AVX;
    static constexpr auto InterpolateVectorArray = &VxSIMDInterpolateVectorArray_AVX;
    static constexpr auto TransformBox2D = &VxSIMDTransformBox2D_AVX;
    static constexpr auto ProjectBoxZExtents = &VxSIMDProjectBoxZExtents_AVX;
    static constexpr auto ComputeBestFitBBox = &VxSIMDComputeBestFitBBox_AVX;
    static constexpr auto BboxClassify = &VxSIMDBboxClassify_AVX;
    static constexpr auto BboxClassifyVertices = &VxSIMDBboxClassifyVertices_AVX;
    static constexpr auto BboxClassifyVerticesOneAxis = &VxSIMDBboxClassifyVerticesOneAxis_AVX;
    static constexpr auto BboxTransformTo = &VxSIMDBboxTransformTo_AVX;
    static constexpr auto BboxTransformFrom = &VxSIMDBboxTransformFrom_AVX;
    static constexpr auto FrustumUpdate = &VxSIMDFrustumUpdate_AVX;
    static constexpr auto FrustumComputeVertices = &VxSIMDFrustumComputeVertices_AVX;
    static constexpr auto FrustumTransform = &VxSIMDFrustumTransform_AVX;

    // Additional operations not yet optimized in AVX (use SSE implementations)
    static constexpr auto AddVector = &VxSIMDAddVector_SSE;
    static constexpr auto SubtractVector = &VxSIMDSubtractVector_SSE;
    static constexpr auto ScaleVector = &VxSIMDScaleVector_SSE;
    static constexpr auto DotVector = &VxSIMDDotVector_SSE;
    static constexpr auto CrossVector = &VxSIMDCrossVector_SSE;
    static constexpr auto LengthVector = &VxSIMDLengthVector_SSE;
    static constexpr auto LengthSquaredVector = &VxSIMDLengthSquaredVector_SSE;
    static constexpr auto DistanceVector = &VxSIMDDistanceVector_SSE;
    static constexpr auto LerpVector = &VxSIMDLerpVector_SSE;
    static constexpr auto ReflectVector = &VxSIMDReflectVector_SSE;
    static constexpr auto MinimizeVector = &VxSIMDMinimizeVector_SSE;
    static constexpr auto MaximizeVector = &VxSIMDMaximizeVector_SSE;
    static constexpr auto AddVector4 = &VxSIMDAddVector4_SSE;
    static constexpr auto SubtractVector4 = &VxSIMDSubtractVector4_SSE;
    static constexpr auto ScaleVector4 = &VxSIMDScaleVector4_SSE;
    static constexpr auto DotVector4 = &VxSIMDDotVector4_SSE;
    static constexpr auto LerpVector4 = &VxSIMDLerpVector4_SSE;
    static constexpr auto MultiplyMatrixVectorStrided = &VxSIMDMultiplyMatrixVectorStrided_SSE;
    static constexpr auto MultiplyMatrixVector4Strided = &VxSIMDMultiplyMatrixVector4Strided_SSE;
    static constexpr auto RotateVectorStrided = &VxSIMDRotateVectorStrided_SSE;
    static constexpr auto MatrixIdentity = &VxSIMDMatrixIdentity_SSE;
    static constexpr auto MatrixInverse = &VxSIMDMatrixInverse_SSE;
    static constexpr auto MatrixDeterminant = &VxSIMDMatrixDeterminant_SSE;
    static constexpr auto MatrixFromRotation = &VxSIMDMatrixFromRotation_SSE;
    static constexpr auto MatrixFromRotationOrigin = &VxSIMDMatrixFromRotationOrigin_SSE;
    static constexpr auto MatrixFromEuler = &VxSIMDMatrixFromEuler_SSE;
    static constexpr auto MatrixToEuler = &VxSIMDMatrixToEuler_SSE;
    static constexpr auto MatrixInterpolate = &VxSIMDMatrixInterpolate_SSE;
    static constexpr auto MatrixInterpolateNoScale = &VxSIMDMatrixInterpolateNoScale_SSE;
    static constexpr auto MatrixDecompose = &VxSIMDMatrixDecompose_SSE;
    static constexpr auto MatrixDecomposeTotal = &VxSIMDMatrixDecomposeTotal_SSE;
    static constexpr auto MatrixDecomposeTotalPtr = &VxSIMDMatrixDecomposeTotalPtr_SSE;
    static constexpr auto QuaternionFromMatrix = &VxSIMDQuaternionFromMatrix_SSE;
    static constexpr auto QuaternionToMatrix = &VxSIMDQuaternionToMatrix_SSE;
    static constexpr auto QuaternionFromRotation = &VxSIMDQuaternionFromRotation_SSE;
    static constexpr auto QuaternionFromEuler = &VxSIMDQuaternionFromEuler_SSE;
    static constexpr auto QuaternionToEuler = &VxSIMDQuaternionToEuler_SSE;
    static constexpr auto QuaternionMultiplyInPlace = &VxSIMDQuaternionMultiplyInPlace_SSE;
    static constexpr auto QuaternionConjugate = &VxSIMDQuaternionConjugate_SSE;
    static constexpr auto QuaternionDivide = &VxSIMDQuaternionDivide_SSE;
    static constexpr auto QuaternionSnuggle = &VxSIMDQuaternionSnuggle_SSE;
    static constexpr auto QuaternionLn = &VxSIMDQuaternionLn_SSE;
    static constexpr auto QuaternionExp = &VxSIMDQuaternionExp_SSE;
    static constexpr auto QuaternionLnDif = &VxSIMDQuaternionLnDif_SSE;
    static constexpr auto QuaternionSquad = &VxSIMDQuaternionSquad_SSE;
    static constexpr auto RayTransform = &VxSIMDRayTransform_SSE;
    static constexpr auto PlaneCreateFromPoint = &VxSIMDPlaneCreateFromPoint_SSE;
    static constexpr auto PlaneCreateFromTriangle = &VxSIMDPlaneCreateFromTriangle_SSE;
    static constexpr auto RectTransform = &VxSIMDRectTransform_SSE;
    static constexpr auto RectTransformBySize = &VxSIMDRectTransformBySize_SSE;
    static constexpr auto RectTransformToHomogeneous = &VxSIMDRectTransformToHomogeneous_SSE;
    static constexpr auto RectTransformFromHomogeneous = &VxSIMDRectTransformFromHomogeneous_SSE;
};

static void ConfigureAVXDispatch(VxSIMDDispatch &table, const char *variantName) {
    ConfigureDispatch<AVXVariant>(table, variantName);
}
#endif


// ============================================================================
// Dispatch Table Initialization
// ============================================================================

static void InitializeDispatchTable() {
    if (g_DispatchInitialized) {
        return;
    }

    g_DispatchTable = {};
    const VxSIMDFeatures &features = VxGetSIMDFeatures();

    struct DispatchCandidate {
        bool available;
        DispatchInitFn init;
        const char *name;
    };

    const DispatchCandidate candidates[] = {
#if defined(VX_SIMD_X86)
#if defined(VX_BUILD_AVX)
    {(features.AVX && features.AVX2 && features.FMA && SupportsExtendedYmmState(features)), ConfigureAVXDispatch, "AVX+FMA (fast path)"},
#endif
#if defined(VX_SIMD_SSE)
    {true, ConfigureSSEDispatch, "SSE (baseline)"},
#endif
#endif
    };

    for (const DispatchCandidate &candidate : candidates) {
        if (!candidate.available) {
            continue;
        }

        candidate.init(g_DispatchTable, candidate.name);
        g_DispatchInitialized = true;
        break;
    }
}

const VxSIMDDispatch *VxGetSIMDDispatch() {
    if (!g_DispatchInitialized) {
        InitializeDispatchTable();
    }
    return &g_DispatchTable;
}

void VxResetSIMDDispatch() {
    g_DispatchInitialized = false;
}

const char *VxGetSIMDInfo() {
    static std::string info_string;
    if (info_string.empty()) {
        const VxSIMDFeatures &features = VxGetSIMDFeatures();
        std::ostringstream oss;

        oss << "VxMath SIMD Support: ";

        bool has_any = false;

#if defined(VX_SIMD_X86)
        if (features.SSE) { oss << "SSE "; has_any = true; }
        if (features.SSE2) { oss << "SSE2 "; has_any = true; }
        if (features.SSE3) { oss << "SSE3 "; has_any = true; }
        if (features.SSSE3) { oss << "SSSE3 "; has_any = true; }
        if (features.SSE4_1) { oss << "SSE4.1 "; has_any = true; }
        if (features.SSE4_2) { oss << "SSE4.2 "; has_any = true; }
        if (features.AVX) { oss << "AVX "; has_any = true; }
        if (features.AVX2) { oss << "AVX2 "; has_any = true; }
        if (features.FMA) { oss << "FMA "; has_any = true; }
        if (features.AVX512F) { oss << "AVX512F "; has_any = true; }
#endif

        if (!has_any) {
            oss << "None (SIMD not available)";
        }

        // Add active dispatch variant
        const VxSIMDDispatch *dispatch = VxGetSIMDDispatch();
        oss << "\nActive variant: " << dispatch->VariantName;

        info_string = oss.str();
    }

    return info_string.c_str();
}