/**
 * @file VxMathTestEdgeCases.h
 * @brief Edge case and degenerate value generators for VxMath tests.
 *
 * This header provides collections of edge case values for testing
 * boundary conditions, degenerate inputs, and numerical edge cases.
 */

#ifndef VXMATH_TEST_EDGE_CASES_H
#define VXMATH_TEST_EDGE_CASES_H

#include <vector>
#include <tuple>
#include <cmath>
#include <limits>

#include "VxMath.h"

namespace VxMathTest {

//=============================================================================
// Vector Edge Cases
//=============================================================================

/**
 * @struct VectorEdgeCases
 * @brief Collection of edge case vectors for testing.
 */
struct VectorEdgeCases {
    /// Zero vector (0, 0, 0)
    static VxVector Zero() { return VxVector(0.0f, 0.0f, 0.0f); }

    /// Unit vector along X axis
    static VxVector UnitX() { return VxVector(1.0f, 0.0f, 0.0f); }

    /// Unit vector along Y axis
    static VxVector UnitY() { return VxVector(0.0f, 1.0f, 0.0f); }

    /// Unit vector along Z axis
    static VxVector UnitZ() { return VxVector(0.0f, 0.0f, 1.0f); }

    /// Negative unit vector along X axis
    static VxVector NegUnitX() { return VxVector(-1.0f, 0.0f, 0.0f); }

    /// Negative unit vector along Y axis
    static VxVector NegUnitY() { return VxVector(0.0f, -1.0f, 0.0f); }

    /// Negative unit vector along Z axis
    static VxVector NegUnitZ() { return VxVector(0.0f, 0.0f, -1.0f); }

    /// Vector with all ones
    static VxVector Ones() { return VxVector(1.0f, 1.0f, 1.0f); }

    /// Vector with all negative ones
    static VxVector NegOnes() { return VxVector(-1.0f, -1.0f, -1.0f); }

    /// Vector with infinity components
    static VxVector Infinity() {
        return VxVector(std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity());
    }

    /// Vector with negative infinity components
    static VxVector NegInfinity() {
        return VxVector(-std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity());
    }

    /// Vector with NaN components
    static VxVector NaN() {
        return VxVector(std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::quiet_NaN());
    }

    /// Vector with very small (near-zero) components
    static VxVector VerySmall() {
        return VxVector(1e-38f, 1e-38f, 1e-38f);
    }

    /// Vector with very large components
    static VxVector VeryLarge() {
        return VxVector(1e38f, 1e38f, 1e38f);
    }

    /// Vector with denormalized float components
    static VxVector Denormalized() {
        float denorm = std::numeric_limits<float>::min() / 2.0f;
        return VxVector(denorm, denorm, denorm);
    }

    /// Vector with mixed positive/negative small values
    static VxVector MixedSmall() {
        return VxVector(1e-7f, -1e-7f, 1e-7f);
    }

    /// Vector with epsilon magnitude
    static VxVector Epsilon() {
        float eps = std::numeric_limits<float>::epsilon();
        return VxVector(eps, eps, eps);
    }

    /// Get all edge case vectors for comprehensive testing
    static std::vector<VxVector> All() {
        return {
            Zero(), UnitX(), UnitY(), UnitZ(),
            NegUnitX(), NegUnitY(), NegUnitZ(),
            Ones(), NegOnes(),
            VerySmall(), VeryLarge(),
            Denormalized(), MixedSmall(), Epsilon()
            // Note: Infinity and NaN excluded as they break most operations
        };
    }

    /// Get all edge case vectors including problematic ones
    static std::vector<VxVector> AllIncludingInvalid() {
        auto cases = All();
        cases.push_back(Infinity());
        cases.push_back(NegInfinity());
        cases.push_back(NaN());
        return cases;
    }
};

//=============================================================================
// Matrix Edge Cases
//=============================================================================

/**
 * @struct MatrixEdgeCases
 * @brief Collection of edge case matrices for testing.
 */
struct MatrixEdgeCases {
    /// Identity matrix
    static VxMatrix Identity() {
        VxMatrix m;
        m.SetIdentity();
        return m;
    }

    /// Zero matrix (all elements 0)
    static VxMatrix Zero() {
        VxMatrix m;
        m.Clear();
        return m;
    }

    /// Singular matrix (determinant = 0)
    static VxMatrix Singular() {
        VxMatrix m;
        m.SetIdentity();
        // Make first row zero to create singular matrix
        m[0][0] = 0.0f;
        m[0][1] = 0.0f;
        m[0][2] = 0.0f;
        m[0][3] = 0.0f;
        return m;
    }

    /// Near-singular matrix (very small determinant)
    static VxMatrix NearSingular() {
        VxMatrix m;
        m.SetIdentity();
        m[0][0] = 1e-7f;
        return m;
    }

    /// Reflection matrix (negative determinant)
    static VxMatrix Reflection() {
        VxMatrix m;
        m.SetIdentity();
        m[0][0] = -1.0f; // Reflect across YZ plane
        return m;
    }

    /// Very small scale matrix
    static VxMatrix VerySmallScale() {
        VxMatrix m;
        m.SetIdentity();
        m[0][0] = 1e-6f;
        m[1][1] = 1e-6f;
        m[2][2] = 1e-6f;
        return m;
    }

    /// Very large scale matrix
    static VxMatrix VeryLargeScale() {
        VxMatrix m;
        m.SetIdentity();
        m[0][0] = 1e6f;
        m[1][1] = 1e6f;
        m[2][2] = 1e6f;
        return m;
    }

    /// Matrix in gimbal lock configuration (90 degree Y rotation)
    static VxMatrix GimbalLock() {
        VxMatrix m;
        Vx3DMatrixFromEulerAngles(m, 0.0f, PI / 2.0f, 0.0f);
        return m;
    }

    /// Rotation matrix (90 degrees around X)
    static VxMatrix Rotation90X() {
        VxMatrix m;
        Vx3DMatrixFromRotation(m, VxVector(1.0f, 0.0f, 0.0f), PI / 2.0f);
        return m;
    }

    /// Rotation matrix (180 degrees around Z)
    static VxMatrix Rotation180Z() {
        VxMatrix m;
        Vx3DMatrixFromRotation(m, VxVector(0.0f, 0.0f, 1.0f), PI);
        return m;
    }

    /// Pure translation matrix
    static VxMatrix PureTranslation() {
        VxMatrix m;
        m.SetIdentity();
        m[3][0] = 100.0f;
        m[3][1] = 200.0f;
        m[3][2] = 300.0f;
        return m;
    }

    /// Non-uniform scale matrix
    static VxMatrix NonUniformScale() {
        VxMatrix m;
        m.SetIdentity();
        m[0][0] = 2.0f;
        m[1][1] = 0.5f;
        m[2][2] = 3.0f;
        return m;
    }

    /// Get all edge case matrices for comprehensive testing
    static std::vector<VxMatrix> All() {
        return {
            Identity(), Singular(), NearSingular(),
            Reflection(), VerySmallScale(), VeryLargeScale(),
            GimbalLock(), Rotation90X(), Rotation180Z(),
            PureTranslation(), NonUniformScale()
            // Note: Zero matrix excluded as it breaks most operations
        };
    }

    /// Get invertible matrices only
    static std::vector<VxMatrix> Invertible() {
        return {
            Identity(), Reflection(),
            Rotation90X(), Rotation180Z(),
            PureTranslation(), NonUniformScale()
        };
    }
};

//=============================================================================
// Quaternion Edge Cases
//=============================================================================

/**
 * @struct QuaternionEdgeCases
 * @brief Collection of edge case quaternions for testing.
 */
struct QuaternionEdgeCases {
    /// Identity quaternion (no rotation)
    static VxQuaternion Identity() {
        return VxQuaternion(0.0f, 0.0f, 0.0f, 1.0f);
    }

    /// Zero quaternion (degenerate, not a valid rotation)
    static VxQuaternion Zero() {
        return VxQuaternion(0.0f, 0.0f, 0.0f, 0.0f);
    }

    /// Pure imaginary quaternion (w = 0, represents 180-degree rotation)
    static VxQuaternion PureImaginary() {
        VxQuaternion q(1.0f, 0.0f, 0.0f, 0.0f);
        q.Normalize();
        return q;
    }

    /// Near-identity quaternion (very small rotation)
    static VxQuaternion NearIdentity() {
        VxQuaternion q;
        q.FromRotation(VxVector(0.0f, 1.0f, 0.0f), 1e-6f);
        return q;
    }

    /// 180-degree rotation around X axis
    static VxQuaternion Rotation180X() {
        VxQuaternion q;
        q.FromRotation(VxVector(1.0f, 0.0f, 0.0f), PI);
        return q;
    }

    /// 90-degree rotation around Y axis
    static VxQuaternion Rotation90Y() {
        VxQuaternion q;
        q.FromRotation(VxVector(0.0f, 1.0f, 0.0f), PI / 2.0f);
        return q;
    }

    /// 45-degree rotation around Z axis
    static VxQuaternion Rotation45Z() {
        VxQuaternion q;
        q.FromRotation(VxVector(0.0f, 0.0f, 1.0f), PI / 4.0f);
        return q;
    }

    /// Negative of identity (represents same rotation as identity)
    static VxQuaternion NegativeIdentity() {
        return VxQuaternion(0.0f, 0.0f, 0.0f, -1.0f);
    }

    /// Get the antipodal quaternion (same rotation, opposite representation)
    static VxQuaternion Antipodal(const VxQuaternion& q) {
        return VxQuaternion(-q.x, -q.y, -q.z, -q.w);
    }

    /// Get all edge case quaternions for comprehensive testing
    static std::vector<VxQuaternion> All() {
        return {
            Identity(), NearIdentity(),
            PureImaginary(), Rotation180X(),
            Rotation90Y(), Rotation45Z(),
            NegativeIdentity()
            // Note: Zero quaternion excluded as it's not a valid rotation
        };
    }

    /// Get all including degenerate cases
    static std::vector<VxQuaternion> AllIncludingInvalid() {
        auto cases = All();
        cases.push_back(Zero());
        return cases;
    }
};

//=============================================================================
// Geometric Edge Cases
//=============================================================================

/**
 * @struct GeometricEdgeCases
 * @brief Collection of edge case geometric primitives for testing.
 */
struct GeometricEdgeCases {
    /// Degenerate plane (zero normal)
    static VxPlane DegeneratePlane() {
        VxPlane p;
        p.m_Normal = VxVector(0.0f, 0.0f, 0.0f);
        p.m_D = 0.0f;
        return p;
    }

    /// XY plane (z = 0)
    static VxPlane XYPlane() {
        VxPlane p;
        p.Create(VxVector(0.0f, 0.0f, 1.0f), VxVector(0.0f, 0.0f, 0.0f));
        return p;
    }

    /// XZ plane (y = 0)
    static VxPlane XZPlane() {
        VxPlane p;
        p.Create(VxVector(0.0f, 1.0f, 0.0f), VxVector(0.0f, 0.0f, 0.0f));
        return p;
    }

    /// YZ plane (x = 0)
    static VxPlane YZPlane() {
        VxPlane p;
        p.Create(VxVector(1.0f, 0.0f, 0.0f), VxVector(0.0f, 0.0f, 0.0f));
        return p;
    }

    /// Collinear triangle points (cannot form a valid plane)
    static std::tuple<VxVector, VxVector, VxVector> CollinearTriangle() {
        return std::make_tuple(
            VxVector(0.0f, 0.0f, 0.0f),
            VxVector(1.0f, 1.0f, 1.0f),
            VxVector(2.0f, 2.0f, 2.0f)
        );
    }

    /// Invalid/uninitialized bbox
    static VxBbox InvalidBbox() {
        VxBbox b;
        // Default constructor creates invalid bbox
        return b;
    }

    /// Empty bbox (Min > Max)
    static VxBbox EmptyBbox() {
        return VxBbox(VxVector(1.0f, 1.0f, 1.0f), VxVector(-1.0f, -1.0f, -1.0f));
    }

    /// Point bbox (Min == Max)
    static VxBbox PointBbox() {
        VxVector p(5.0f, 5.0f, 5.0f);
        return VxBbox(p, p);
    }

    /// Unit bbox centered at origin
    static VxBbox UnitBbox() {
        return VxBbox(VxVector(-0.5f, -0.5f, -0.5f), VxVector(0.5f, 0.5f, 0.5f));
    }

    /// Very large bbox
    static VxBbox LargeBbox() {
        return VxBbox(VxVector(-1e6f, -1e6f, -1e6f), VxVector(1e6f, 1e6f, 1e6f));
    }

    /// Zero-radius sphere (point)
    static VxSphere PointSphere() {
        return VxSphere(VxVector(0.0f, 0.0f, 0.0f), 0.0f);
    }

    /// Unit sphere at origin
    static VxSphere UnitSphere() {
        return VxSphere(VxVector(0.0f, 0.0f, 0.0f), 1.0f);
    }

    /// Very small sphere
    static VxSphere TinySphere() {
        return VxSphere(VxVector(0.0f, 0.0f, 0.0f), 1e-6f);
    }

    /// Degenerate ray (zero direction)
    static VxRay DegenerateRay() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
        r.m_Direction = VxVector(0.0f, 0.0f, 0.0f);
        return r;
    }

    /// Ray along X axis from origin
    static VxRay XAxisRay() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
        r.m_Direction = VxVector(1.0f, 0.0f, 0.0f);
        return r;
    }

    /// Ray along Y axis from origin
    static VxRay YAxisRay() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
        r.m_Direction = VxVector(0.0f, 1.0f, 0.0f);
        return r;
    }

    /// Ray along Z axis from origin
    static VxRay ZAxisRay() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
        r.m_Direction = VxVector(0.0f, 0.0f, 1.0f);
        return r;
    }

    /// Ray parallel to XY plane
    static VxRay RayParallelToXY() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 5.0f);
        r.m_Direction = VxVector(1.0f, 1.0f, 0.0f);
        r.m_Direction.Normalize();
        return r;
    }
};

//=============================================================================
// Intersection Test Cases
//=============================================================================

/**
 * @struct IntersectionEdgeCases
 * @brief Edge cases specifically for intersection testing.
 */
struct IntersectionEdgeCases {
    /// Ray starting inside a unit sphere at origin
    static std::pair<VxRay, VxSphere> RayInsideSphere() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
        r.m_Direction = VxVector(1.0f, 0.0f, 0.0f);
        return std::make_pair(r, VxSphere(VxVector(0.0f, 0.0f, 0.0f), 2.0f));
    }

    /// Ray tangent to sphere (grazes surface)
    static std::pair<VxRay, VxSphere> RayTangentToSphere() {
        VxRay r;
        r.m_Origin = VxVector(-5.0f, 1.0f, 0.0f);
        r.m_Direction = VxVector(1.0f, 0.0f, 0.0f);
        return std::make_pair(r, VxSphere(VxVector(0.0f, 0.0f, 0.0f), 1.0f));
    }

    /// Ray parallel to plane (no intersection)
    static std::pair<VxRay, VxPlane> RayParallelToPlane() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 5.0f);
        r.m_Direction = VxVector(1.0f, 0.0f, 0.0f);

        VxPlane p;
        p.Create(VxVector(0.0f, 0.0f, 1.0f), VxVector(0.0f, 0.0f, 0.0f));

        return std::make_pair(r, p);
    }

    /// Ray in the plane (infinite intersections)
    static std::pair<VxRay, VxPlane> RayInPlane() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
        r.m_Direction = VxVector(1.0f, 1.0f, 0.0f);
        r.m_Direction.Normalize();

        VxPlane p;
        p.Create(VxVector(0.0f, 0.0f, 1.0f), VxVector(0.0f, 0.0f, 0.0f));

        return std::make_pair(r, p);
    }

    /// Ray starting inside bbox
    static std::pair<VxRay, VxBbox> RayInsideBbox() {
        VxRay r;
        r.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
        r.m_Direction = VxVector(1.0f, 0.0f, 0.0f);

        VxBbox b(VxVector(-1.0f, -1.0f, -1.0f), VxVector(1.0f, 1.0f, 1.0f));

        return std::make_pair(r, b);
    }

    /// Ray grazing bbox edge
    static std::pair<VxRay, VxBbox> RayGrazingBboxEdge() {
        VxRay r;
        r.m_Origin = VxVector(-5.0f, 1.0f, 0.0f);
        r.m_Direction = VxVector(1.0f, 0.0f, 0.0f);

        VxBbox b(VxVector(-1.0f, -1.0f, -1.0f), VxVector(1.0f, 1.0f, 1.0f));

        return std::make_pair(r, b);
    }
};

} // namespace VxMathTest

#endif // VXMATH_TEST_EDGE_CASES_H
