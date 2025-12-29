/**
 * @file SIMDFeatureTest.cpp
 * @brief Tests for SIMD feature detection and infrastructure.
 *
 * Tests cover:
 * - CPU feature detection (VxDetectSIMDFeatures)
 * - Feature caching (VxGetSIMDFeatures)
 * - SIMD info string (VxGetSIMDInfo)
 * - Aligned memory allocation (VxAlignedMalloc/VxAlignedFree)
 * - Dispatch table validity
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>

#include "VxSIMD.h"
#include "SIMDTestCommon.h"

namespace {

//=============================================================================
// Feature Detection Tests
//=============================================================================

TEST(SIMDFeatureDetection, VxDetectSIMDFeatures_ReturnsValidStruct) {
    VxSIMDFeatures features = VxDetectSIMDFeatures();

    // On x86/x64, at minimum SSE/SSE2 should be available (baseline for 64-bit)
#if defined(VX_SIMD_X64)
    EXPECT_TRUE(features.SSE) << "SSE should be available on x64";
    EXPECT_TRUE(features.SSE2) << "SSE2 should be available on x64";
#endif

    // Advanced features imply their prerequisites
    if (features.SSE4_2) {
        EXPECT_TRUE(features.SSE4_1) << "SSE4.2 implies SSE4.1";
    }
    if (features.SSE4_1) {
        EXPECT_TRUE(features.SSSE3) << "SSE4.1 implies SSSE3";
    }
    if (features.SSSE3) {
        EXPECT_TRUE(features.SSE3) << "SSSE3 implies SSE3";
    }
    if (features.SSE3) {
        EXPECT_TRUE(features.SSE2) << "SSE3 implies SSE2";
    }
    if (features.AVX2) {
        EXPECT_TRUE(features.AVX) << "AVX2 implies AVX";
    }
    if (features.AVX512F) {
        EXPECT_TRUE(features.AVX2) << "AVX512F implies AVX2";
        EXPECT_TRUE(features.AVX) << "AVX512F implies AVX";
    }
}

TEST(SIMDFeatureDetection, VxGetSIMDFeatures_ReturnsCachedValue) {
    // Call multiple times - should return same reference
    const VxSIMDFeatures& ref1 = VxGetSIMDFeatures();
    const VxSIMDFeatures& ref2 = VxGetSIMDFeatures();

    EXPECT_EQ(&ref1, &ref2) << "VxGetSIMDFeatures should return cached reference";

    // Values should be identical
    EXPECT_EQ(ref1.SSE, ref2.SSE);
    EXPECT_EQ(ref1.SSE2, ref2.SSE2);
    EXPECT_EQ(ref1.SSE3, ref2.SSE3);
    EXPECT_EQ(ref1.AVX, ref2.AVX);
    EXPECT_EQ(ref1.AVX2, ref2.AVX2);
    EXPECT_EQ(ref1.FMA, ref2.FMA);
}

TEST(SIMDFeatureDetection, VxGetSIMDInfo_ReturnsNonEmptyString) {
    const char* info = VxGetSIMDInfo();

    ASSERT_NE(info, nullptr);
    EXPECT_GT(strlen(info), 0) << "SIMD info string should not be empty";
    EXPECT_NE(strstr(info, "VxMath"), nullptr) << "Info should mention VxMath";

    // Should mention active variant
    EXPECT_NE(strstr(info, "Active variant"), nullptr);
}

TEST(SIMDFeatureDetection, VxGetSIMDInfo_ContainsDetectedFeatures) {
    const char* info = VxGetSIMDInfo();
    const VxSIMDFeatures& features = VxGetSIMDFeatures();

    // Verify detected features appear in info string
#if defined(VX_SIMD_X86)
    if (features.SSE) {
        EXPECT_NE(strstr(info, "SSE"), nullptr) << "Info should list SSE if available";
    }
    if (features.SSE2) {
        EXPECT_NE(strstr(info, "SSE2"), nullptr) << "Info should list SSE2 if available";
    }
    if (features.AVX) {
        EXPECT_NE(strstr(info, "AVX"), nullptr) << "Info should list AVX if available";
    }
    if (features.AVX2) {
        EXPECT_NE(strstr(info, "AVX2"), nullptr) << "Info should list AVX2 if available";
    }
    if (features.FMA) {
        EXPECT_NE(strstr(info, "FMA"), nullptr) << "Info should list FMA if available";
    }
#endif
}

//=============================================================================
// Aligned Memory Tests
//=============================================================================

class SIMDAlignedMemoryTest : public ::testing::Test {
protected:
    void TearDown() override {
        for (void* ptr : m_allocated) {
            VxAlignedFree(ptr);
        }
        m_allocated.clear();
    }

    void* AllocAndTrack(size_t size, size_t alignment) {
        void* ptr = VxAlignedMalloc(size, alignment);
        if (ptr) m_allocated.push_back(ptr);
        return ptr;
    }

    std::vector<void*> m_allocated;
};

TEST_F(SIMDAlignedMemoryTest, VxAlignedMalloc_ReturnsAligned16) {
    void* ptr = AllocAndTrack(64, 16);

    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 16, 0)
        << "Pointer should be 16-byte aligned";
}

TEST_F(SIMDAlignedMemoryTest, VxAlignedMalloc_ReturnsAligned32) {
    void* ptr = AllocAndTrack(128, 32);

    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 32, 0)
        << "Pointer should be 32-byte aligned";
}

TEST_F(SIMDAlignedMemoryTest, VxAlignedMalloc_ReturnsAligned64) {
    void* ptr = AllocAndTrack(256, 64);

    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0)
        << "Pointer should be 64-byte aligned";
}

TEST_F(SIMDAlignedMemoryTest, VxAlignedMalloc_VariousSizes) {
    const size_t sizes[] = {1, 7, 15, 16, 31, 32, 63, 64, 100, 1000, 4096};
    const size_t alignments[] = {16, 32, 64};

    for (size_t align : alignments) {
        for (size_t size : sizes) {
            void* ptr = AllocAndTrack(size, align);
            ASSERT_NE(ptr, nullptr) << "Allocation failed for size=" << size << ", align=" << align;
            EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % align, 0)
                << "Misaligned for size=" << size << ", align=" << align;

            // Write to all bytes to verify memory is usable
            std::memset(ptr, 0xAB, size);
        }
    }
}

TEST_F(SIMDAlignedMemoryTest, VxAlignedMalloc_ZeroSize) {
    // Zero size allocation behavior is implementation-defined
    // But we should not crash
    void* ptr = VxAlignedMalloc(0, 16);
    // If returns non-null, it should still be aligned
    if (ptr) {
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 16, 0);
        VxAlignedFree(ptr);
    }
}

TEST_F(SIMDAlignedMemoryTest, VxAlignedFree_HandleNull) {
    // Should not crash on null
    VxAlignedFree(nullptr);
    SUCCEED() << "VxAlignedFree(nullptr) did not crash";
}

TEST_F(SIMDAlignedMemoryTest, VxAlignedMalloc_MemoryIsWritable) {
    const size_t size = 1024;
    float* ptr = static_cast<float*>(AllocAndTrack(size * sizeof(float), 32));
    ASSERT_NE(ptr, nullptr);

    // Write pattern
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<float>(i * 1.5);
    }

    // Verify
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i * 1.5));
    }
}

//=============================================================================
// Dispatch Table Tests
//=============================================================================

TEST(SIMDDispatchTable, VxGetSIMDDispatch_ReturnsValidTable) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();

    ASSERT_NE(dispatch, nullptr);
    EXPECT_NE(dispatch->VariantName, nullptr);
    EXPECT_GT(strlen(dispatch->VariantName), 0) << "Variant name should not be empty";
}

TEST(SIMDDispatchTable, VectorOpsAreValid) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    EXPECT_NE(dispatch->Vector.NormalizeVector, nullptr);
    EXPECT_NE(dispatch->Vector.RotateVector, nullptr);
    EXPECT_NE(dispatch->Vector.Add, nullptr);
    EXPECT_NE(dispatch->Vector.Subtract, nullptr);
    EXPECT_NE(dispatch->Vector.Scale, nullptr);
    EXPECT_NE(dispatch->Vector.Dot, nullptr);
    EXPECT_NE(dispatch->Vector.Cross, nullptr);
    EXPECT_NE(dispatch->Vector.Length, nullptr);
    EXPECT_NE(dispatch->Vector.LengthSquared, nullptr);
    EXPECT_NE(dispatch->Vector.Distance, nullptr);
    EXPECT_NE(dispatch->Vector.Lerp, nullptr);
    EXPECT_NE(dispatch->Vector.Reflect, nullptr);
    EXPECT_NE(dispatch->Vector.Minimize, nullptr);
    EXPECT_NE(dispatch->Vector.Maximize, nullptr);
}

TEST(SIMDDispatchTable, Vector4OpsAreValid) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    EXPECT_NE(dispatch->Vector4.Add, nullptr);
    EXPECT_NE(dispatch->Vector4.Subtract, nullptr);
    EXPECT_NE(dispatch->Vector4.Scale, nullptr);
    EXPECT_NE(dispatch->Vector4.Dot, nullptr);
    EXPECT_NE(dispatch->Vector4.Lerp, nullptr);
}

TEST(SIMDDispatchTable, MatrixOpsAreValid) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    EXPECT_NE(dispatch->Matrix.MultiplyMatrix, nullptr);
    EXPECT_NE(dispatch->Matrix.MultiplyMatrix4, nullptr);
    EXPECT_NE(dispatch->Matrix.TransposeMatrix, nullptr);
    EXPECT_NE(dispatch->Matrix.MultiplyMatrixVector, nullptr);
    EXPECT_NE(dispatch->Matrix.MultiplyMatrixVector4, nullptr);
    EXPECT_NE(dispatch->Matrix.RotateVectorOp, nullptr);
    EXPECT_NE(dispatch->Matrix.MultiplyMatrixVectorMany, nullptr);
    EXPECT_NE(dispatch->Matrix.RotateVectorMany, nullptr);
    EXPECT_NE(dispatch->Matrix.MultiplyMatrixVectorStrided, nullptr);
    EXPECT_NE(dispatch->Matrix.MultiplyMatrixVector4Strided, nullptr);
    EXPECT_NE(dispatch->Matrix.RotateVectorStrided, nullptr);
    EXPECT_NE(dispatch->Matrix.Identity, nullptr);
    EXPECT_NE(dispatch->Matrix.Inverse, nullptr);
    EXPECT_NE(dispatch->Matrix.Determinant, nullptr);
    EXPECT_NE(dispatch->Matrix.FromAxisAngle, nullptr);
    EXPECT_NE(dispatch->Matrix.FromAxisAngleOrigin, nullptr);
    EXPECT_NE(dispatch->Matrix.FromEulerAngles, nullptr);
    EXPECT_NE(dispatch->Matrix.ToEulerAngles, nullptr);
    EXPECT_NE(dispatch->Matrix.Interpolate, nullptr);
    EXPECT_NE(dispatch->Matrix.InterpolateNoScale, nullptr);
    EXPECT_NE(dispatch->Matrix.Decompose, nullptr);
    EXPECT_NE(dispatch->Matrix.DecomposeTotal, nullptr);
    EXPECT_NE(dispatch->Matrix.DecomposeTotalPtr, nullptr);
}

TEST(SIMDDispatchTable, QuaternionOpsAreValid) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    EXPECT_NE(dispatch->Quaternion.NormalizeQuaternion, nullptr);
    EXPECT_NE(dispatch->Quaternion.MultiplyQuaternion, nullptr);
    EXPECT_NE(dispatch->Quaternion.SlerpQuaternion, nullptr);
    EXPECT_NE(dispatch->Quaternion.FromMatrix, nullptr);
    EXPECT_NE(dispatch->Quaternion.ToMatrix, nullptr);
    EXPECT_NE(dispatch->Quaternion.FromAxisAngle, nullptr);
    EXPECT_NE(dispatch->Quaternion.FromEulerAngles, nullptr);
    EXPECT_NE(dispatch->Quaternion.ToEulerAngles, nullptr);
    EXPECT_NE(dispatch->Quaternion.MultiplyInPlace, nullptr);
    EXPECT_NE(dispatch->Quaternion.Conjugate, nullptr);
    EXPECT_NE(dispatch->Quaternion.Divide, nullptr);
    EXPECT_NE(dispatch->Quaternion.Snuggle, nullptr);
    EXPECT_NE(dispatch->Quaternion.Ln, nullptr);
    EXPECT_NE(dispatch->Quaternion.Exp, nullptr);
    EXPECT_NE(dispatch->Quaternion.LnDif, nullptr);
    EXPECT_NE(dispatch->Quaternion.Squad, nullptr);
}

TEST(SIMDDispatchTable, GeometryOpsAreValid) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    EXPECT_NE(dispatch->Ray.Transform, nullptr);
    EXPECT_NE(dispatch->Plane.CreateFromPoint, nullptr);
    EXPECT_NE(dispatch->Plane.CreateFromTriangle, nullptr);
    EXPECT_NE(dispatch->Rect.Transform, nullptr);
    EXPECT_NE(dispatch->Rect.TransformBySize, nullptr);
    EXPECT_NE(dispatch->Rect.TransformToHomogeneous, nullptr);
    EXPECT_NE(dispatch->Rect.TransformFromHomogeneous, nullptr);
    EXPECT_NE(dispatch->Geometry.TransformBox2D, nullptr);
    EXPECT_NE(dispatch->Geometry.ProjectBoxZExtents, nullptr);
    EXPECT_NE(dispatch->Geometry.ComputeBestFitBBox, nullptr);
}

TEST(SIMDDispatchTable, BboxOpsAreValid) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    EXPECT_NE(dispatch->Bbox.Classify, nullptr);
    EXPECT_NE(dispatch->Bbox.ClassifyVertices, nullptr);
    EXPECT_NE(dispatch->Bbox.ClassifyVerticesOneAxis, nullptr);
    EXPECT_NE(dispatch->Bbox.TransformTo, nullptr);
    EXPECT_NE(dispatch->Bbox.TransformFrom, nullptr);
}

TEST(SIMDDispatchTable, FrustumOpsAreValid) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    EXPECT_NE(dispatch->Frustum.Update, nullptr);
    EXPECT_NE(dispatch->Frustum.ComputeVertices, nullptr);
    EXPECT_NE(dispatch->Frustum.Transform, nullptr);
}

TEST(SIMDDispatchTable, ArrayOpsAreValid) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    EXPECT_NE(dispatch->Array.InterpolateFloatArray, nullptr);
    EXPECT_NE(dispatch->Array.InterpolateVectorArray, nullptr);
}

//=============================================================================
// Dispatch Reset Tests
//=============================================================================

TEST(SIMDDispatchReset, VxResetSIMDDispatch_AllowsReinitialization) {
    // Get initial dispatch
    const VxSIMDDispatch* dispatch1 = VxGetSIMDDispatch();
    ASSERT_NE(dispatch1, nullptr);
    const char* name1 = dispatch1->VariantName;

    // Reset and get again
    VxResetSIMDDispatch();
    const VxSIMDDispatch* dispatch2 = VxGetSIMDDispatch();

    ASSERT_NE(dispatch2, nullptr);
    // Should be same variant (same CPU)
    EXPECT_STREQ(dispatch2->VariantName, name1);
}

} // namespace
