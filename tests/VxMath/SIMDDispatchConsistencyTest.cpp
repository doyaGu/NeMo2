#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "VxSIMD.h"
#include "VxMath.h"
#include "VxMathDefines.h"
#include "VxPlane.h"
#include "VxQuaternion.h"
#include "VxRay.h"
#include "VxRect.h"
#include "VxVector.h"
#include "VxMatrix.h"
#include "VxFrustum.h"

// Use centralized test helpers
#include "VxMathTestHelpers.h"

using namespace VxMathTest;

namespace {

// Default tolerance for SIMD consistency tests
constexpr float kTol = STANDARD_TOL;

// Alias for backward compatibility in this file
inline bool NearTol(float a, float b, float tol = kTol) {
    return ScaleRelativeNear(a, b, tol);
}

// Use MakeTRS from VxMathTestGenerators.h via RandomGenerator
static VxMatrix MakeTRS(float tx, float ty, float tz, float eax, float eay, float eaz, float sx, float sy, float sz) {
    return RandomGenerator::MakeTRS(tx, ty, tz, eax, eay, eaz, sx, sy, sz);
}

//=============================================================================
// Scalar Reference Implementations (unique to this file for SIMD validation)
//=============================================================================

static void ScalarMatrixFromAxisAngleOrigin(VxMatrix& out, const VxVector& axis, const VxVector& origin, float angle) {
    Vx3DMatrixFromRotation(out, axis, angle);
    VxVector rotatedOrigin;
    Vx3DRotateVector(&rotatedOrigin, out, &origin);
    out[3][0] = origin.x - rotatedOrigin.x;
    out[3][1] = origin.y - rotatedOrigin.y;
    out[3][2] = origin.z - rotatedOrigin.z;
    out[3][3] = 1.0f;
}

} // namespace

TEST(SIMDDispatchConsistency, TransformBox2D_MatchesScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Geometry.TransformBox2D, nullptr);

    VxMatrix m;
    m.SetIdentity();

    // Mild translation so screen-space extents are non-trivial.
    m[3][0] = 0.25f;
    m[3][1] = -0.5f;
    m[3][2] = 0.75f;

    const VxBbox box(VxVector(-1.0f, -2.0f, 0.5f), VxVector(4.0f, 5.0f, 2.0f));

    VxRect screen(0.0f, 0.0f, 1920.0f, 1080.0f);

    VxRect extScalar;
    VXCLIP_FLAGS orScalar{}, andScalar{};
    const XBOOL okScalar = VxTransformBox2D(m, box, &screen, &extScalar, orScalar, andScalar);

    VxRect extSimd;
    VXCLIP_FLAGS orSimd{}, andSimd{};
    const XBOOL okSimd = dispatch->Geometry.TransformBox2D(&m, &box, &screen, &extSimd, &orSimd, &andSimd);

    EXPECT_EQ(okScalar, okSimd);
    EXPECT_EQ(orScalar, orSimd);
    EXPECT_EQ(andScalar, andSimd);
    ExpectNearRect(extScalar, extSimd);
}

TEST(SIMDDispatchConsistency, ProjectBoxZExtents_MatchesScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Geometry.ProjectBoxZExtents, nullptr);

    VxMatrix m;
    m.SetIdentity();

    const VxBbox box(VxVector(-1.0f, -2.0f, 0.5f), VxVector(4.0f, 5.0f, 2.0f));

    float zMinScalar = 0.0f, zMaxScalar = 0.0f;
    VxProjectBoxZExtents(m, box, zMinScalar, zMaxScalar);

    float zMinSimd = 0.0f, zMaxSimd = 0.0f;
    dispatch->Geometry.ProjectBoxZExtents(&m, &box, &zMinSimd, &zMaxSimd);

    EXPECT_TRUE(NearTol(zMinScalar, zMinSimd));
    EXPECT_TRUE(NearTol(zMaxScalar, zMaxSimd));
}

TEST(SIMDDispatchConsistency, ComputeBestFitBBox_MatchesScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Geometry.ComputeBestFitBBox, nullptr);

    // Deterministic point cloud (16 points)
    std::vector<float> pts;
    pts.reserve(16 * 3);
    for (int i = 0; i < 16; ++i) {
        const float t = static_cast<float>(i) * 0.25f;
        pts.push_back(std::cos(t) * 3.0f + 1.0f);
        pts.push_back(std::sin(t) * 2.0f - 0.5f);
        pts.push_back((t - 2.0f) * 0.75f);
    }

    VxMatrix mScalar;
    VxMatrix mSimd;

    const XBOOL okScalar = VxComputeBestFitBBox(reinterpret_cast<const XBYTE*>(pts.data()), sizeof(float) * 3, 16, mScalar, 0.1f);
    const XBOOL okSimd = dispatch->Geometry.ComputeBestFitBBox(reinterpret_cast<const XBYTE*>(pts.data()), sizeof(float) * 3, 16, &mSimd, 0.1f);

    EXPECT_EQ(okScalar, okSimd);
    if (okScalar && okSimd) {
        ExpectNearMatrix(mScalar, mSimd, 2e-4f);
    }
}

TEST(SIMDDispatchConsistency, PlaneCreateFromPoint_MatchesScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Plane.CreateFromPoint, nullptr);

    const VxVector n(0.0f, 0.0f, 10.0f);
    const VxVector p(1.25f, -2.0f, 3.5f);

    VxPlane scalar;
    scalar.Create(n, p);

    VxPlane simd;
    dispatch->Plane.CreateFromPoint(&simd, &n, &p);

    ExpectNearPlane(scalar, simd);
}

TEST(SIMDDispatchConsistency, PlaneCreateFromTriangle_DegenerateMatchesScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Plane.CreateFromTriangle, nullptr);

    // Degenerate (colinear) triangle
    const VxVector a(0.0f, 0.0f, 0.0f);
    const VxVector b(1.0f, 1.0f, 1.0f);
    const VxVector c(2.0f, 2.0f, 2.0f);

    VxPlane scalar;
    scalar.Create(a, b, c);

    VxPlane simd;
    dispatch->Plane.CreateFromTriangle(&simd, &a, &b, &c);

    ExpectNearPlane(scalar, simd);
}

TEST(SIMDDispatchConsistency, RectTransforms_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Rect.Transform, nullptr);
    ASSERT_NE(dispatch->Rect.TransformToHomogeneous, nullptr);
    ASSERT_NE(dispatch->Rect.TransformFromHomogeneous, nullptr);

    const VxRect src(10.0f, 20.0f, 110.0f, 220.0f);
    const VxRect dst(0.0f, 0.0f, 1920.0f, 1080.0f);

    VxRect rScalar(12.0f, 30.0f, 60.0f, 200.0f);
    VxRect rSimd = rScalar;

    rScalar.Transform(dst, src);
    dispatch->Rect.Transform(&rSimd, &dst, &src);
    ExpectNearRect(rScalar, rSimd);

    VxRect screen(0.0f, 0.0f, 800.0f, 600.0f);

    VxRect hScalar = rScalar;
    VxRect hSimd = rScalar;
    hScalar.TransformToHomogeneous(screen);
    dispatch->Rect.TransformToHomogeneous(&hSimd, &screen);
    ExpectNearRect(hScalar, hSimd);

    VxRect backScalar = hScalar;
    VxRect backSimd = hSimd;
    backScalar.TransformFromHomogeneous(screen);
    dispatch->Rect.TransformFromHomogeneous(&backSimd, &screen);
    ExpectNearRect(backScalar, backSimd);
}

TEST(SIMDDispatchConsistency, VectorOps_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    ASSERT_NE(dispatch->Vector.NormalizeVector, nullptr);
    ASSERT_NE(dispatch->Vector.RotateVector, nullptr);
    ASSERT_NE(dispatch->Vector.Add, nullptr);
    ASSERT_NE(dispatch->Vector.Subtract, nullptr);
    ASSERT_NE(dispatch->Vector.Scale, nullptr);
    ASSERT_NE(dispatch->Vector.Dot, nullptr);
    ASSERT_NE(dispatch->Vector.Cross, nullptr);
    ASSERT_NE(dispatch->Vector.Length, nullptr);
    ASSERT_NE(dispatch->Vector.LengthSquared, nullptr);
    ASSERT_NE(dispatch->Vector.Distance, nullptr);
    ASSERT_NE(dispatch->Vector.Lerp, nullptr);
    ASSERT_NE(dispatch->Vector.Reflect, nullptr);
    ASSERT_NE(dispatch->Vector.Minimize, nullptr);
    ASSERT_NE(dispatch->Vector.Maximize, nullptr);

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> distSmall(-1e-6f, 1e-6f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int iter = 0; iter < 200; ++iter) {
        VxVector a(dist(rng), dist(rng), dist(rng));
        VxVector b(dist(rng), dist(rng), dist(rng));
        VxVector n(dist(rng), dist(rng), dist(rng));
        if ((iter % 25) == 0) {
            a = VxVector(distSmall(rng), distSmall(rng), distSmall(rng));
        }
        const float s = dist(rng);
        const float t = dist01(rng);

        VxVector addScalar = a + b;
        VxVector addSimd;
        dispatch->Vector.Add(&addSimd, &a, &b);
        ExpectNearVec3(addScalar, addSimd);

        VxVector subScalar = a - b;
        VxVector subSimd;
        dispatch->Vector.Subtract(&subSimd, &a, &b);
        ExpectNearVec3(subScalar, subSimd);

        VxVector scaleScalar = a * s;
        VxVector scaleSimd;
        dispatch->Vector.Scale(&scaleSimd, &a, s);
        ExpectNearVec3(scaleScalar, scaleSimd);

        const float dotScalar = DotProduct(a, b);
        const float dotSimd = dispatch->Vector.Dot(&a, &b);
        EXPECT_TRUE(NearTol(dotScalar, dotSimd));

        VxVector crossScalar = CrossProduct(a, b);
        VxVector crossSimd;
        dispatch->Vector.Cross(&crossSimd, &a, &b);
        ExpectNearVec3(crossScalar, crossSimd);

        const float lenScalar = a.Magnitude();
        const float lenSimd = dispatch->Vector.Length(&a);
        EXPECT_TRUE(NearTol(lenScalar, lenSimd));

        const float lenSqScalar = a.SquareMagnitude();
        const float lenSqSimd = dispatch->Vector.LengthSquared(&a);
        EXPECT_TRUE(NearTol(lenSqScalar, lenSqSimd));

        const float distScalarV = (a - b).Magnitude();
        const float distSimdV = dispatch->Vector.Distance(&a, &b);
        EXPECT_TRUE(NearTol(distScalarV, distSimdV));

        VxVector lerpScalar = a + (b - a) * t;
        VxVector lerpSimd;
        dispatch->Vector.Lerp(&lerpSimd, &a, &b, t);
        ExpectNearVec3(lerpScalar, lerpSimd);

        VxVector nNorm = n;
        nNorm.Normalize();
        const VxVector reflectScalar = Reflect(a, nNorm);
        VxVector reflectSimd;
        dispatch->Vector.Reflect(&reflectSimd, &a, &nNorm);
        ExpectNearVec3(reflectScalar, reflectSimd);

        const VxVector minScalar(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
        VxVector minSimd;
        dispatch->Vector.Minimize(&minSimd, &a, &b);
        ExpectNearVec3(minScalar, minSimd);

        const VxVector maxScalar(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
        VxVector maxSimd;
        dispatch->Vector.Maximize(&maxSimd, &a, &b);
        ExpectNearVec3(maxScalar, maxSimd);

        VxVector normScalar = a;
        normScalar.Normalize();
        VxVector normSimd = a;
        dispatch->Vector.NormalizeVector(&normSimd);
        ExpectNearVec3(normScalar, normSimd);

        VxMatrix rot;
        Vx3DMatrixFromEulerAngles(rot, dist(rng) * 0.1f, dist(rng) * 0.1f, dist(rng) * 0.1f);
        VxVector rotScalar;
        Vx3DRotateVector(&rotScalar, rot, &a);
        VxVector rotSimd;
        dispatch->Vector.RotateVector(&rotSimd, &rot, &a);
        ExpectNearVec3(rotScalar, rotSimd);
    }
}

TEST(SIMDDispatchConsistency, Vector4Ops_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Vector4.Add, nullptr);
    ASSERT_NE(dispatch->Vector4.Subtract, nullptr);
    ASSERT_NE(dispatch->Vector4.Scale, nullptr);
    ASSERT_NE(dispatch->Vector4.Dot, nullptr);
    ASSERT_NE(dispatch->Vector4.Lerp, nullptr);

    std::mt19937 rng(0x1234567u);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int iter = 0; iter < 200; ++iter) {
        const VxVector4 a(dist(rng), dist(rng), dist(rng), dist(rng));
        const VxVector4 b(dist(rng), dist(rng), dist(rng), dist(rng));
        const float s = dist(rng);
        const float t = dist01(rng);

        const VxVector4 addScalar(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        VxVector4 addSimd;
        dispatch->Vector4.Add(&addSimd, &a, &b);
        ExpectNearVec4(addScalar, addSimd);

        const VxVector4 subScalar(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
        VxVector4 subSimd;
        dispatch->Vector4.Subtract(&subSimd, &a, &b);
        ExpectNearVec4(subScalar, subSimd);

        const VxVector4 scaleScalar(a.x * s, a.y * s, a.z * s, a.w * s);
        VxVector4 scaleSimd;
        dispatch->Vector4.Scale(&scaleSimd, &a, s);
        ExpectNearVec4(scaleScalar, scaleSimd);

        const float dotScalar = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        const float dotSimd = dispatch->Vector4.Dot(&a, &b);
        EXPECT_TRUE(NearTol(dotScalar, dotSimd));

        const VxVector4 lerpScalar(a.x + (b.x - a.x) * t,
                                   a.y + (b.y - a.y) * t,
                                   a.z + (b.z - a.z) * t,
                                   a.w + (b.w - a.w) * t);
        VxVector4 lerpSimd;
        dispatch->Vector4.Lerp(&lerpSimd, &a, &b, t);
        ExpectNearVec4(lerpScalar, lerpSimd);
    }
}

TEST(SIMDDispatchConsistency, MatrixOps_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    ASSERT_NE(dispatch->Matrix.MultiplyMatrix, nullptr);
    ASSERT_NE(dispatch->Matrix.MultiplyMatrix4, nullptr);
    ASSERT_NE(dispatch->Matrix.TransposeMatrix, nullptr);
    ASSERT_NE(dispatch->Matrix.MultiplyMatrixVector, nullptr);
    ASSERT_NE(dispatch->Matrix.MultiplyMatrixVector4, nullptr);
    ASSERT_NE(dispatch->Matrix.RotateVectorOp, nullptr);
    ASSERT_NE(dispatch->Matrix.MultiplyMatrixVectorMany, nullptr);
    ASSERT_NE(dispatch->Matrix.RotateVectorMany, nullptr);
    ASSERT_NE(dispatch->Matrix.MultiplyMatrixVectorStrided, nullptr);
    ASSERT_NE(dispatch->Matrix.MultiplyMatrixVector4Strided, nullptr);
    ASSERT_NE(dispatch->Matrix.RotateVectorStrided, nullptr);
    ASSERT_NE(dispatch->Matrix.Identity, nullptr);
    ASSERT_NE(dispatch->Matrix.Inverse, nullptr);
    ASSERT_NE(dispatch->Matrix.Determinant, nullptr);
    ASSERT_NE(dispatch->Matrix.FromAxisAngle, nullptr);
    ASSERT_NE(dispatch->Matrix.FromAxisAngleOrigin, nullptr);
    ASSERT_NE(dispatch->Matrix.FromEulerAngles, nullptr);
    ASSERT_NE(dispatch->Matrix.ToEulerAngles, nullptr);
    ASSERT_NE(dispatch->Matrix.Interpolate, nullptr);
    ASSERT_NE(dispatch->Matrix.InterpolateNoScale, nullptr);
    ASSERT_NE(dispatch->Matrix.Decompose, nullptr);
    ASSERT_NE(dispatch->Matrix.DecomposeTotal, nullptr);
    ASSERT_NE(dispatch->Matrix.DecomposeTotalPtr, nullptr);

    std::mt19937 rng(0xBADC0DEu);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> distPos(-5.0f, 5.0f);
    std::uniform_real_distribution<float> distScale(0.5f, 2.0f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int iter = 0; iter < 120; ++iter) {
        SCOPED_TRACE(::testing::Message() << "iter=" << iter);
        const VxMatrix a = MakeTRS(distPos(rng), distPos(rng), distPos(rng), dist(rng), dist(rng), dist(rng), distScale(rng), distScale(rng), distScale(rng));
        const VxMatrix b = MakeTRS(distPos(rng), distPos(rng), distPos(rng), dist(rng), dist(rng), dist(rng), distScale(rng), distScale(rng), distScale(rng));

        SCOPED_TRACE("MultiplyMatrix");
        VxMatrix mulScalar;
        Vx3DMultiplyMatrix(mulScalar, a, b);
        VxMatrix mulSimd;
        dispatch->Matrix.MultiplyMatrix(&mulSimd, &a, &b);
        ExpectNearMatrix(mulScalar, mulSimd, 2e-5f);

        SCOPED_TRACE("MultiplyMatrix4");
        VxMatrix mul4Scalar;
        Vx3DMultiplyMatrix4(mul4Scalar, a, b);
        VxMatrix mul4Simd;
        dispatch->Matrix.MultiplyMatrix4(&mul4Simd, &a, &b);
        ExpectNearMatrix(mul4Scalar, mul4Simd, 2e-5f);

        SCOPED_TRACE("TransposeMatrix");
        VxMatrix trScalar;
        Vx3DTransposeMatrix(trScalar, a);
        VxMatrix trSimd;
        dispatch->Matrix.TransposeMatrix(&trSimd, &a);
        ExpectNearMatrix(trScalar, trSimd, 2e-5f);

        const VxVector v(distPos(rng), distPos(rng), distPos(rng));
        SCOPED_TRACE("MultiplyMatrixVector");
        VxVector mvScalar;
        Vx3DMultiplyMatrixVector(&mvScalar, a, &v);
        VxVector mvSimd;
        dispatch->Matrix.MultiplyMatrixVector(&mvSimd, &a, &v);
        ExpectNearVec3(mvScalar, mvSimd, 2e-5f);

        const VxVector4 v4(distPos(rng), distPos(rng), distPos(rng), distPos(rng));
        SCOPED_TRACE("MultiplyMatrixVector4");
        VxVector4 mv4Scalar;
        Vx3DMultiplyMatrixVector4(&mv4Scalar, a, &v4);
        VxVector4 mv4Simd;
        dispatch->Matrix.MultiplyMatrixVector4(&mv4Simd, &a, &v4);
        ExpectNearVec4(mv4Scalar, mv4Simd, 2e-5f);

        SCOPED_TRACE("RotateVectorOp");
        VxVector rvScalar;
        Vx3DRotateVector(&rvScalar, a, &v);
        VxVector rvSimd;
        dispatch->Matrix.RotateVectorOp(&rvSimd, &a, &v);
        ExpectNearVec3(rvScalar, rvSimd, 2e-5f);

        // Many/strided variants
        struct PaddedVec { VxVector v; std::uint32_t pad; };
        std::vector<PaddedVec> src(16);
        std::vector<PaddedVec> dstScalar(16);
        std::vector<PaddedVec> dstSimd(16);
        for (int i = 0; i < 16; ++i) {
            src[i].v = VxVector(distPos(rng), distPos(rng), distPos(rng));
            src[i].pad = 0xDEADBEEF;
            dstScalar[i].pad = 0xAAAAAAAA;
            dstSimd[i].pad = 0xBBBBBBBB;
        }

        SCOPED_TRACE("MultiplyMatrixVectorMany");
        Vx3DMultiplyMatrixVectorMany(&dstScalar[0].v, a, &src[0].v, 16, static_cast<int>(sizeof(PaddedVec)));
        dispatch->Matrix.MultiplyMatrixVectorMany(&dstSimd[0].v, &a, &src[0].v, 16, static_cast<int>(sizeof(PaddedVec)));
        for (int i = 0; i < 16; ++i) {
            ExpectNearVec3(dstScalar[i].v, dstSimd[i].v, 2e-5f);
            EXPECT_EQ(dstScalar[i].pad, 0xAAAAAAAAu);
            EXPECT_EQ(dstSimd[i].pad, 0xBBBBBBBBu);
        }

        SCOPED_TRACE("RotateVectorMany");
        Vx3DRotateVectorMany(&dstScalar[0].v, a, &src[0].v, 16, static_cast<int>(sizeof(PaddedVec)));
        dispatch->Matrix.RotateVectorMany(&dstSimd[0].v, &a, &src[0].v, 16, static_cast<int>(sizeof(PaddedVec)));
        for (int i = 0; i < 16; ++i) {
            ExpectNearVec3(dstScalar[i].v, dstSimd[i].v, 2e-5f);
        }

        VxStridedData srcStrided(&src[0].v, static_cast<unsigned int>(sizeof(PaddedVec)));
        VxStridedData dstStridedScalar(&dstScalar[0].v, static_cast<unsigned int>(sizeof(PaddedVec)));
        VxStridedData dstStridedSimd(&dstSimd[0].v, static_cast<unsigned int>(sizeof(PaddedVec)));

        SCOPED_TRACE("MultiplyMatrixVectorStrided");
        Vx3DMultiplyMatrixVectorStrided(&dstStridedScalar, &srcStrided, a, 16);
        dispatch->Matrix.MultiplyMatrixVectorStrided(&dstStridedSimd, &srcStrided, &a, 16);
        for (int i = 0; i < 16; ++i) {
            ExpectNearVec3(dstScalar[i].v, dstSimd[i].v, 2e-5f);
        }

        // Vector4 strided
        struct PaddedVec4 { VxVector4 v; std::uint32_t pad; };
        std::vector<PaddedVec4> src4(16);
        std::vector<PaddedVec4> dst4Scalar(16);
        std::vector<PaddedVec4> dst4Simd(16);
        for (int i = 0; i < 16; ++i) {
            src4[i].v = VxVector4(distPos(rng), distPos(rng), distPos(rng), distPos(rng));
            src4[i].pad = 0x11111111;
            dst4Scalar[i].pad = 0x22222222;
            dst4Simd[i].pad = 0x33333333;
        }
        VxStridedData src4Strided(&src4[0].v, static_cast<unsigned int>(sizeof(PaddedVec4)));
        VxStridedData dst4StridedScalar(&dst4Scalar[0].v, static_cast<unsigned int>(sizeof(PaddedVec4)));
        VxStridedData dst4StridedSimd(&dst4Simd[0].v, static_cast<unsigned int>(sizeof(PaddedVec4)));
        SCOPED_TRACE("MultiplyMatrixVector4Strided");
        Vx3DMultiplyMatrixVector4Strided(&dst4StridedScalar, &src4Strided, a, 16);
        dispatch->Matrix.MultiplyMatrixVector4Strided(&dst4StridedSimd, &src4Strided, &a, 16);
        for (int i = 0; i < 16; ++i) {
            ExpectNearVec4(dst4Scalar[i].v, dst4Simd[i].v, 2e-5f);
            EXPECT_EQ(dst4Scalar[i].pad, 0x22222222u);
            EXPECT_EQ(dst4Simd[i].pad, 0x33333333u);
        }

        // Rotate strided
        SCOPED_TRACE("RotateVectorStrided");
        Vx3DRotateVectorStrided(&dstStridedScalar, &srcStrided, a, 16);
        dispatch->Matrix.RotateVectorStrided(&dstStridedSimd, &srcStrided, &a, 16);
        for (int i = 0; i < 16; ++i) {
            ExpectNearVec3(dstScalar[i].v, dstSimd[i].v, 2e-5f);
        }

        // Identity/inverse/determinant
        SCOPED_TRACE("Identity");
        VxMatrix idScalar;
        idScalar.SetIdentity();
        VxMatrix idSimd;
        dispatch->Matrix.Identity(&idSimd);
        ExpectNearMatrix(idScalar, idSimd, 0.0f);

        SCOPED_TRACE("Inverse");
        VxMatrix invScalar;
        Vx3DInverseMatrix(invScalar, a);
        VxMatrix invSimd;
        dispatch->Matrix.Inverse(&invSimd, &a);
        ExpectNearMatrix(invScalar, invSimd, 5e-5f);

        SCOPED_TRACE("Determinant");
        const float detScalar = Vx3DMatrixDeterminant(a);
        const float detSimd = dispatch->Matrix.Determinant(&a);
        EXPECT_TRUE(NearTol(detScalar, detSimd, 5e-5f));

        // From axis-angle (no origin)
        SCOPED_TRACE("FromAxisAngle");
        const VxVector axis(dist(rng), dist(rng), dist(rng));
        const float ang = dist(rng);
        VxMatrix axScalar;
        Vx3DMatrixFromRotation(axScalar, axis, ang);
        VxMatrix axSimd;
        dispatch->Matrix.FromAxisAngle(&axSimd, &axis, ang);
        ExpectNearMatrix(axScalar, axSimd, 2e-5f);

        // From axis-angle with origin
        SCOPED_TRACE("FromAxisAngleOrigin");
        const VxVector origin(distPos(rng), distPos(rng), distPos(rng));
        VxMatrix axoScalar;
        ScalarMatrixFromAxisAngleOrigin(axoScalar, axis, origin, ang);
        VxMatrix axoSimd;
        dispatch->Matrix.FromAxisAngleOrigin(&axoSimd, &axis, &origin, ang);
        ExpectNearMatrix(axoScalar, axoSimd, 3e-5f);

        // Euler conversions
        SCOPED_TRACE("FromEulerAngles");
        const float eax = dist(rng);
        const float eay = dist(rng);
        const float eaz = dist(rng);
        VxMatrix eScalar;
        Vx3DMatrixFromEulerAngles(eScalar, eax, eay, eaz);
        VxMatrix eSimd;
        dispatch->Matrix.FromEulerAngles(&eSimd, eax, eay, eaz);
        ExpectNearMatrix(eScalar, eSimd, 2e-5f);

        SCOPED_TRACE("ToEulerAngles");
        float oaxS = 0, oayS = 0, oazS = 0;
        float oaxV = 0, oayV = 0, oazV = 0;
        Vx3DMatrixToEulerAngles(eScalar, &oaxS, &oayS, &oazS);
        dispatch->Matrix.ToEulerAngles(&eSimd, &oaxV, &oayV, &oazV);
        EXPECT_TRUE(NearTol(oaxS, oaxV, 2e-5f));
        EXPECT_TRUE(NearTol(oayS, oayV, 2e-5f));
        EXPECT_TRUE(NearTol(oazS, oazV, 2e-5f));

        SCOPED_TRACE("Interpolate");
        const float step = dist01(rng);
        VxMatrix iScalar;
        Vx3DInterpolateMatrix(step, iScalar, a, b);
        VxMatrix iSimd;
        dispatch->Matrix.Interpolate(step, &iSimd, &a, &b);
        ExpectNearMatrix(iScalar, iSimd, 6e-5f);

        SCOPED_TRACE("InterpolateNoScale");
        VxMatrix insScalar;
        Vx3DInterpolateMatrixNoScale(step, insScalar, a, b);
        VxMatrix insSimd;
        dispatch->Matrix.InterpolateNoScale(step, &insSimd, &a, &b);
        ExpectNearMatrix(insScalar, insSimd, 6e-5f);

        SCOPED_TRACE("Decompose");
        VxQuaternion qS;
        VxVector pS, scS;
        Vx3DDecomposeMatrix(a, qS, pS, scS);
        VxQuaternion qV;
        VxVector pV, scV;
        dispatch->Matrix.Decompose(&a, &qV, &pV, &scV);
        ExpectNearQuat(qS, qV, 7e-5f);
        ExpectNearVec3(pS, pV, 7e-5f);
        ExpectNearVec3(scS, scV, 7e-5f);

        SCOPED_TRACE("DecomposeTotal");
        VxQuaternion uS;
        VxQuaternion uV;
        const float retS = Vx3DDecomposeMatrixTotal(a, qS, pS, scS, uS);
        const float retV = dispatch->Matrix.DecomposeTotal(&a, &qV, &pV, &scV, &uV);
        EXPECT_TRUE(NearTol(retS, retV, 7e-5f));
        ExpectNearQuat(qS, qV, 7e-5f);
        ExpectNearVec3(pS, pV, 7e-5f);
        ExpectNearVec3(scS, scV, 7e-5f);
        ExpectNearQuat(uS, uV, 7e-5f);

        SCOPED_TRACE("DecomposeTotalPtr");
        VxQuaternion qP;
        VxVector pP, scP;
        VxQuaternion uP;
        const float retPtrS = Vx3DDecomposeMatrixTotalPtr(a, &qP, &pP, &scP, &uP);

        VxQuaternion qP2;
        VxVector pP2, scP2;
        VxQuaternion uP2;
        const float retPtrV = dispatch->Matrix.DecomposeTotalPtr(&a, &qP2, &pP2, &scP2, &uP2);

        EXPECT_TRUE(NearTol(retPtrS, retPtrV, 7e-5f));
        ExpectNearQuat(qP, qP2, 7e-5f);
        ExpectNearVec3(pP, pP2, 7e-5f);
        ExpectNearVec3(scP, scP2, 7e-5f);
        ExpectNearQuat(uP, uP2, 7e-5f);
    }
}

TEST(SIMDDispatchConsistency, QuaternionOps_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    ASSERT_NE(dispatch->Quaternion.NormalizeQuaternion, nullptr);
    ASSERT_NE(dispatch->Quaternion.MultiplyQuaternion, nullptr);
    ASSERT_NE(dispatch->Quaternion.SlerpQuaternion, nullptr);
    ASSERT_NE(dispatch->Quaternion.FromMatrix, nullptr);
    ASSERT_NE(dispatch->Quaternion.ToMatrix, nullptr);
    ASSERT_NE(dispatch->Quaternion.FromAxisAngle, nullptr);
    ASSERT_NE(dispatch->Quaternion.FromEulerAngles, nullptr);
    ASSERT_NE(dispatch->Quaternion.ToEulerAngles, nullptr);
    ASSERT_NE(dispatch->Quaternion.MultiplyInPlace, nullptr);
    ASSERT_NE(dispatch->Quaternion.Conjugate, nullptr);
    ASSERT_NE(dispatch->Quaternion.Divide, nullptr);
    ASSERT_NE(dispatch->Quaternion.Snuggle, nullptr);
    ASSERT_NE(dispatch->Quaternion.Ln, nullptr);
    ASSERT_NE(dispatch->Quaternion.Exp, nullptr);
    ASSERT_NE(dispatch->Quaternion.LnDif, nullptr);
    ASSERT_NE(dispatch->Quaternion.Squad, nullptr);

    std::mt19937 rng(0xFACEFEEDu);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> distPos(-5.0f, 5.0f);

    for (int iter = 0; iter < 120; ++iter) {
        SCOPED_TRACE(::testing::Message() << "iter=" << iter);
        VxQuaternion a(dist(rng), dist(rng), dist(rng), dist(rng));
        VxQuaternion b(dist(rng), dist(rng), dist(rng), dist(rng));

        VxQuaternion an = a;
        an.Normalize();
        VxQuaternion bn = b;
        bn.Normalize();

        SCOPED_TRACE("NormalizeQuaternion");
        VxQuaternion normScalar = a;
        normScalar.Normalize();
        VxQuaternion normSimd = a;
        dispatch->Quaternion.NormalizeQuaternion(&normSimd);
        ExpectNearQuat(normScalar, normSimd, 7e-5f);

        SCOPED_TRACE("MultiplyQuaternion");
        const VxQuaternion mulScalar = Vx3DQuaternionMultiply(an, bn);
        VxQuaternion mulSimd;
        dispatch->Quaternion.MultiplyQuaternion(&mulSimd, &an, &bn);
        ExpectNearQuat(mulScalar, mulSimd, 7e-5f);

        const float t = dist01(rng);
        SCOPED_TRACE("SlerpQuaternion");
        const VxQuaternion slerpScalar = Slerp(t, an, bn);
        VxQuaternion slerpSimd;
        dispatch->Quaternion.SlerpQuaternion(&slerpSimd, t, &an, &bn);
        ExpectNearQuat(slerpScalar, slerpSimd, 8e-5f);

        const VxVector axis(dist(rng), dist(rng), dist(rng));
        const float ang = dist(rng);
        SCOPED_TRACE("FromAxisAngle");
        VxQuaternion axScalar;
        axScalar.FromRotation(axis, ang);
        VxQuaternion axSimd;
        dispatch->Quaternion.FromAxisAngle(&axSimd, &axis, ang);
        ExpectNearQuat(axScalar, axSimd, 7e-5f);

        const float eax = dist(rng), eay = dist(rng), eaz = dist(rng);
        SCOPED_TRACE("FromEulerAngles");
        VxQuaternion eScalar;
        eScalar.FromEulerAngles(eax, eay, eaz);
        VxQuaternion eSimd;
        dispatch->Quaternion.FromEulerAngles(&eSimd, eax, eay, eaz);
        ExpectNearQuat(eScalar, eSimd, 8e-5f);

        SCOPED_TRACE("ToEulerAngles");
        float oaxS = 0, oayS = 0, oazS = 0;
        float oaxV = 0, oayV = 0, oazV = 0;
        eScalar.ToEulerAngles(&oaxS, &oayS, &oazS);
        dispatch->Quaternion.ToEulerAngles(&eSimd, &oaxV, &oayV, &oazV);
        EXPECT_TRUE(NearTol(oaxS, oaxV, 2e-4f));
        EXPECT_TRUE(NearTol(oayS, oayV, 2e-4f));
        EXPECT_TRUE(NearTol(oazS, oazV, 2e-4f));

        // Matrix conversions
        SCOPED_TRACE("FromMatrix");
        const VxMatrix m = MakeTRS(distPos(rng), distPos(rng), distPos(rng), dist(rng), dist(rng), dist(rng), 1.0f, 1.0f, 1.0f);
        VxQuaternion fmScalar;
        {
            VxMatrix mCopy = m;
            fmScalar.FromMatrix(mCopy, TRUE, TRUE);
        }
        VxQuaternion fmSimd;
        {
            VxMatrix mCopy = m;
            dispatch->Quaternion.FromMatrix(&fmSimd, &mCopy, TRUE, TRUE);
        }
        ExpectNearQuat(fmScalar, fmSimd, 1e-4f);

        SCOPED_TRACE("ToMatrix");
        VxMatrix tmScalar;
        fmScalar.ToMatrix(tmScalar);
        VxMatrix tmSimd;
        dispatch->Quaternion.ToMatrix(&tmSimd, &fmSimd);
        ExpectNearMatrix(tmScalar, tmSimd, 2e-4f);

        // In-place multiply
        SCOPED_TRACE("MultiplyInPlace");
        VxQuaternion inScalar = an;
        inScalar = Vx3DQuaternionMultiply(inScalar, bn);
        VxQuaternion inSimd = an;
        dispatch->Quaternion.MultiplyInPlace(&inSimd, &bn);
        ExpectNearQuat(inScalar, inSimd, 8e-5f);

        // Conjugate / divide
        SCOPED_TRACE("Conjugate");
        const VxQuaternion conjScalar = Vx3DQuaternionConjugate(an);
        VxQuaternion conjSimd;
        dispatch->Quaternion.Conjugate(&conjSimd, &an);
        ExpectNearQuat(conjScalar, conjSimd, 0.0f);

        SCOPED_TRACE("Divide");
        const VxQuaternion divScalar = Vx3DQuaternionDivide(an, bn);
        VxQuaternion divSimd;
        dispatch->Quaternion.Divide(&divSimd, &an, &bn);
        ExpectNearQuat(divScalar, divSimd, 1e-4f);

        // Snuggle
        SCOPED_TRACE("Snuggle");
        VxQuaternion snQ1 = fmScalar;
        VxVector snS1(1.2f, 0.7f, 2.0f);
        const VxQuaternion snResScalar = Vx3DQuaternionSnuggle(&snQ1, &snS1);
        VxQuaternion snQ2 = fmScalar;
        VxVector snS2(1.2f, 0.7f, 2.0f);
        VxQuaternion snResSimd;
        dispatch->Quaternion.Snuggle(&snResSimd, &snQ2, &snS2);
        ExpectNearQuat(snResScalar, snResSimd, 2e-4f);
        ExpectNearQuat(snQ1, snQ2, 2e-4f);
        ExpectNearVec3(snS1, snS2, 2e-4f);

        SCOPED_TRACE("Ln");
        const VxQuaternion lnScalar = Ln(an);
        VxQuaternion lnSimd;
        dispatch->Quaternion.Ln(&lnSimd, &an);
        ExpectNearQuat(lnScalar, lnSimd, 2e-4f);

        SCOPED_TRACE("Exp");
        const VxQuaternion expScalar = Exp(lnScalar);
        VxQuaternion expSimd;
        dispatch->Quaternion.Exp(&expSimd, &lnSimd);
        ExpectNearQuat(expScalar, expSimd, 2e-4f);

        SCOPED_TRACE("LnDif");
        const VxQuaternion lndifScalar = LnDif(an, bn);
        VxQuaternion lndifSimd;
        dispatch->Quaternion.LnDif(&lndifSimd, &an, &bn);
        ExpectNearQuat(lndifScalar, lndifSimd, 3e-4f);

        const VxQuaternion q1Out = Slerp(0.33f, an, bn);
        const VxQuaternion q2In = Slerp(0.66f, an, bn);
        SCOPED_TRACE("Squad");
        const VxQuaternion squadScalar = Squad(t, an, q1Out, q2In, bn);
        VxQuaternion squadSimd;
        dispatch->Quaternion.Squad(&squadSimd, t, &an, &q1Out, &q2In, &bn);
        ExpectNearQuat(squadScalar, squadSimd, 3e-4f);
    }
}

TEST(SIMDDispatchConsistency, RayOps_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Ray.Transform, nullptr);

    std::mt19937 rng(0x424242u);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> ang(-1.0f, 1.0f);

    for (int iter = 0; iter < 80; ++iter) {
        const VxVector o(dist(rng), dist(rng), dist(rng));
        const VxVector d(dist(rng), dist(rng), dist(rng));
        VxRay ray;
        ray.m_Origin = o;
        ray.m_Direction = d;

        const VxMatrix m = MakeTRS(dist(rng), dist(rng), dist(rng), ang(rng), ang(rng), ang(rng), 1.0f, 1.0f, 1.0f);

        VxRay scalar;
        ray.Transform(scalar, m);
        VxRay simd;
        dispatch->Ray.Transform(&simd, &ray, &m);

        ExpectNearVec3(scalar.m_Origin, simd.m_Origin, 2e-5f);
        ExpectNearVec3(scalar.m_Direction, simd.m_Direction, 2e-5f);
    }
}

TEST(SIMDDispatchConsistency, ArrayOps_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Array.InterpolateFloatArray, nullptr);
    ASSERT_NE(dispatch->Array.InterpolateVectorArray, nullptr);

    std::mt19937 rng(0xABCDEFu);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int iter = 0; iter < 50; ++iter) {
        const int count = 257;
        std::vector<float> a(count);
        std::vector<float> b(count);
        std::vector<float> outScalar(count);
        std::vector<float> outSimd(count);
        for (int i = 0; i < count; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        const float t = dist01(rng);
        InterpolateFloatArray(outScalar.data(), a.data(), b.data(), t, count);
        dispatch->Array.InterpolateFloatArray(outSimd.data(), a.data(), b.data(), t, count);
        for (int i = 0; i < count; ++i) {
            EXPECT_TRUE(NearTol(outScalar[i], outSimd[i], 2e-5f)) << "i=" << i;
        }

        struct PaddedV { VxVector v; std::uint32_t pad; };
        std::vector<PaddedV> va(65);
        std::vector<PaddedV> vb(65);
        std::vector<PaddedV> vrS(65);
        std::vector<PaddedV> vrV(65);
        for (int i = 0; i < 65; ++i) {
            va[i].v = VxVector(dist(rng), dist(rng), dist(rng));
            vb[i].v = VxVector(dist(rng), dist(rng), dist(rng));
            va[i].pad = 0x11111111;
            vb[i].pad = 0x22222222;
            vrS[i].pad = 0x33333333;
            vrV[i].pad = 0x44444444;
        }
        const XULONG stride = static_cast<XULONG>(sizeof(PaddedV));
        InterpolateVectorArray(&vrS[0].v, &va[0].v, &vb[0].v, t, 65, stride, stride);
        dispatch->Array.InterpolateVectorArray(&vrV[0].v, &va[0].v, &vb[0].v, t, 65, stride, stride);
        for (int i = 0; i < 65; ++i) {
            ExpectNearVec3(vrS[i].v, vrV[i].v, 2e-5f);
            EXPECT_EQ(vrS[i].pad, 0x33333333u);
            EXPECT_EQ(vrV[i].pad, 0x44444444u);
        }
    }
}

TEST(SIMDDispatchConsistency, BboxOps_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);

    ASSERT_NE(dispatch->Bbox.Classify, nullptr);
    ASSERT_NE(dispatch->Bbox.ClassifyVertices, nullptr);
    ASSERT_NE(dispatch->Bbox.ClassifyVerticesOneAxis, nullptr);
    ASSERT_NE(dispatch->Bbox.TransformTo, nullptr);
    ASSERT_NE(dispatch->Bbox.TransformFrom, nullptr);

    std::mt19937 rng(0x515151u);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> distPos(-5.0f, 5.0f);

    for (int iter = 0; iter < 80; ++iter) {
        VxVector minA(dist(rng), dist(rng), dist(rng));
        VxVector maxA(dist(rng), dist(rng), dist(rng));
        if (minA.x > maxA.x) std::swap(minA.x, maxA.x);
        if (minA.y > maxA.y) std::swap(minA.y, maxA.y);
        if (minA.z > maxA.z) std::swap(minA.z, maxA.z);

        VxVector minB(dist(rng), dist(rng), dist(rng));
        VxVector maxB(dist(rng), dist(rng), dist(rng));
        if (minB.x > maxB.x) std::swap(minB.x, maxB.x);
        if (minB.y > maxB.y) std::swap(minB.y, maxB.y);
        if (minB.z > maxB.z) std::swap(minB.z, maxB.z);

        const VxBbox a(minA, maxA);
        const VxBbox b(minB, maxB);
        const VxVector p(dist(rng), dist(rng), dist(rng));

        const int cScalar = a.Classify(b, p);
        const int cSimd = dispatch->Bbox.Classify(&a, &b, &p);
        EXPECT_EQ(cScalar, cSimd);

        // Vertices classification
        std::vector<VxVector> verts(64);
        for (int i = 0; i < 64; ++i) {
            verts[i] = VxVector(dist(rng), dist(rng), dist(rng));
        }
        std::vector<XULONG> flagsS(64, 0);
        std::vector<XULONG> flagsV(64, 0);
        a.ClassifyVertices(static_cast<int>(verts.size()), reinterpret_cast<XBYTE*>(verts.data()), static_cast<XULONG>(sizeof(VxVector)), flagsS.data());
        dispatch->Bbox.ClassifyVertices(&a, static_cast<int>(verts.size()), reinterpret_cast<const XBYTE*>(verts.data()), static_cast<XULONG>(sizeof(VxVector)), flagsV.data());
        EXPECT_EQ(flagsS, flagsV);

        const int axis = iter % 3;
        std::fill(flagsS.begin(), flagsS.end(), 0);
        std::fill(flagsV.begin(), flagsV.end(), 0);
        a.ClassifyVerticesOneAxis(static_cast<int>(verts.size()), reinterpret_cast<XBYTE*>(verts.data()), static_cast<XULONG>(sizeof(VxVector)), axis, flagsS.data());
        dispatch->Bbox.ClassifyVerticesOneAxis(&a, static_cast<int>(verts.size()), reinterpret_cast<const XBYTE*>(verts.data()), static_cast<XULONG>(sizeof(VxVector)), axis, flagsV.data());
        EXPECT_EQ(flagsS, flagsV);

        // TransformTo/From
        const VxMatrix m = MakeTRS(distPos(rng), distPos(rng), distPos(rng), distPos(rng) * 0.2f, distPos(rng) * 0.2f, distPos(rng) * 0.2f, 1.0f, 1.0f, 1.0f);

        VxVector ptsS[8];
        VxVector ptsV[8];
        a.TransformTo(ptsS, m);
        dispatch->Bbox.TransformTo(&a, ptsV, &m);
        for (int i = 0; i < 8; ++i) {
            ExpectNearVec3(ptsS[i], ptsV[i], 3e-5f);
        }

        VxBbox tfS;
        tfS.TransformFrom(a, m);
        VxBbox tfV;
        dispatch->Bbox.TransformFrom(&tfV, &a, &m);
        ExpectNearBbox(tfS, tfV, 4e-5f);
    }
}

TEST(SIMDDispatchConsistency, FrustumOps_MatchScalar) {
    const VxSIMDDispatch* dispatch = VxGetSIMDDispatch();
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(dispatch->Frustum.Update, nullptr);
    ASSERT_NE(dispatch->Frustum.ComputeVertices, nullptr);
    ASSERT_NE(dispatch->Frustum.Transform, nullptr);

    std::mt19937 rng(0xF00DF00Du);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> distPos(-5.0f, 5.0f);
    std::uniform_real_distribution<float> distNear(0.1f, 2.0f);
    std::uniform_real_distribution<float> distFar(3.0f, 25.0f);
    std::uniform_real_distribution<float> distFov(0.5f, 1.5f);
    std::uniform_real_distribution<float> distAspect(0.5f, 2.0f);

    for (int iter = 0; iter < 80; ++iter) {
        // Build a roughly orthonormal basis
        VxVector right(1.0f, 0.0f, 0.0f);
        VxVector up(0.0f, 1.0f, 0.0f);
        VxVector dir(0.0f, 0.0f, 1.0f);
        VxMatrix r;
        Vx3DMatrixFromEulerAngles(r, dist(rng), dist(rng), dist(rng));
        Vx3DRotateVector(&right, r, &right);
        Vx3DRotateVector(&up, r, &up);
        Vx3DRotateVector(&dir, r, &dir);
        right.Normalize();
        up.Normalize();
        dir.Normalize();

        const VxVector origin(distPos(rng), distPos(rng), distPos(rng));
        const float nearP = distNear(rng);
        const float farP = std::max(distFar(rng), nearP + 0.5f);
        const float fov = distFov(rng);
        const float aspect = distAspect(rng);

        VxFrustum fScalar(origin, right, up, dir, nearP, farP, fov, aspect);
        VxFrustum fSimd = fScalar;

        fScalar.Update();
        dispatch->Frustum.Update(&fSimd);

        ExpectNearPlane(fScalar.GetNearPlane(), fSimd.GetNearPlane(), 6e-5f);
        ExpectNearPlane(fScalar.GetFarPlane(), fSimd.GetFarPlane(), 6e-5f);
        ExpectNearPlane(fScalar.GetLeftPlane(), fSimd.GetLeftPlane(), 6e-5f);
        ExpectNearPlane(fScalar.GetRightPlane(), fSimd.GetRightPlane(), 6e-5f);
        ExpectNearPlane(fScalar.GetUpPlane(), fSimd.GetUpPlane(), 6e-5f);
        ExpectNearPlane(fScalar.GetBottomPlane(), fSimd.GetBottomPlane(), 6e-5f);

        VxVector vS[8];
        VxVector vV[8];
        fScalar.ComputeVertices(vS);
        dispatch->Frustum.ComputeVertices(&fSimd, vV);
        for (int i = 0; i < 8; ++i) {
            ExpectNearVec3(vS[i], vV[i], 7e-5f);
        }

        const VxMatrix invWorld = MakeTRS(distPos(rng), distPos(rng), distPos(rng), dist(rng) * 0.25f, dist(rng) * 0.25f, dist(rng) * 0.25f, 1.0f, 1.0f, 1.0f);
        VxFrustum tfS = fScalar;
        VxFrustum tfV = fSimd;
        tfS.Transform(invWorld);
        dispatch->Frustum.Transform(&tfV, &invWorld);

        VxVector tVS[8];
        VxVector tVV[8];
        tfS.ComputeVertices(tVS);
        tfV.ComputeVertices(tVV);
        for (int i = 0; i < 8; ++i) {
            ExpectNearVec3(tVS[i], tVV[i], 1e-4f);
        }
    }
}
