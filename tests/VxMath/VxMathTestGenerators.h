/**
 * @file VxMathTestGenerators.h
 * @brief Random value generators for VxMath test data.
 *
 * This header provides a comprehensive RandomGenerator class that can generate
 * random values for all VxMath types with reproducible results via seeding.
 *
 * Consolidates generators from:
 * - VxMathTestHelpers.h (RandomFloat, RandomVector, etc.)
 * - SIMDDispatchConsistencyTest.cpp (MakeTRS helper)
 */

#ifndef VXMATH_TEST_GENERATORS_H
#define VXMATH_TEST_GENERATORS_H

#include <random>
#include <cmath>

#include "VxMath.h"
#include "VxMathDefines.h"

namespace VxMathTest {

/**
 * @class RandomGenerator
 * @brief Seeded random generator for reproducible test data.
 *
 * All generators use a controllable seed for reproducibility. The default
 * seed (42) ensures consistent test results across runs.
 */
class RandomGenerator {
public:
    /// Default seed for reproducible tests
    static constexpr uint32_t DEFAULT_SEED = 42;

    /// Construct with optional seed
    explicit RandomGenerator(uint32_t seed = DEFAULT_SEED)
        : m_seed(seed), m_rng(seed) {}

    /// Reset the generator with a new seed
    void SetSeed(uint32_t seed) {
        m_seed = seed;
        m_rng.seed(seed);
    }

    /// Get the current seed
    uint32_t GetSeed() const { return m_seed; }

    //=========================================================================
    // Scalar Generators
    //=========================================================================

    /// Generate a random float in range [min, max]
    float Float(float min = -100.0f, float max = 100.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(m_rng);
    }

    /// Generate a random positive float in range [min, max]
    float PositiveFloat(float min = 0.0f, float max = 100.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(m_rng);
    }

    /// Generate a random angle in range [min, max] (default 0 to 2*PI)
    float Angle(float min = 0.0f, float max = 2.0f * PI) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(m_rng);
    }

    /// Generate a very small float for near-zero testing
    float SmallFloat(float magnitude = 1e-6f) {
        std::uniform_real_distribution<float> dist(-magnitude, magnitude);
        return dist(m_rng);
    }

    /// Generate a float suitable for interpolation [0, 1]
    float InterpolationFactor() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(m_rng);
    }

    /// Generate a scale factor (always positive)
    float Scale(float min = 0.1f, float max = 10.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(m_rng);
    }

    //=========================================================================
    // Vector Generators
    //=========================================================================

    /// Generate a random VxVector with components in range [min, max]
    VxVector Vector(float min = -100.0f, float max = 100.0f) {
        return VxVector(Float(min, max), Float(min, max), Float(min, max));
    }

    /// Generate a random unit vector (normalized)
    VxVector UnitVector() {
        VxVector v = Vector(-1.0f, 1.0f);
        // Avoid zero vector
        while (v.SquareMagnitude() < 1e-10f) {
            v = Vector(-1.0f, 1.0f);
        }
        v.Normalize();
        return v;
    }

    /// Generate a very small vector for near-zero testing
    VxVector SmallVector(float magnitude = 1e-6f) {
        return VxVector(SmallFloat(magnitude), SmallFloat(magnitude), SmallFloat(magnitude));
    }

    /// Generate a random VxVector4 with components in range [min, max]
    VxVector4 Vector4(float min = -100.0f, float max = 100.0f) {
        return VxVector4(Float(min, max), Float(min, max), Float(min, max), Float(min, max));
    }

    /// Generate a random Vx2DVector with components in range [min, max]
    Vx2DVector Vector2D(float min = -100.0f, float max = 100.0f) {
        return Vx2DVector(Float(min, max), Float(min, max));
    }

    //=========================================================================
    // Matrix Generators
    //=========================================================================

    /// Generate an identity matrix
    VxMatrix IdentityMatrix() {
        VxMatrix m;
        m.SetIdentity();
        return m;
    }

    /// Generate a random rotation matrix
    VxMatrix RotationMatrix() {
        VxMatrix m;
        VxVector axis = UnitVector();
        float angle = Angle();
        Vx3DMatrixFromRotation(m, axis, angle);
        return m;
    }

    /// Generate a random translation matrix
    VxMatrix TranslationMatrix(float range = 100.0f) {
        VxMatrix m;
        m.SetIdentity();
        m[3][0] = Float(-range, range);
        m[3][1] = Float(-range, range);
        m[3][2] = Float(-range, range);
        return m;
    }

    /// Generate a random uniform scale matrix
    VxMatrix ScaleMatrix(float minScale = 0.1f, float maxScale = 10.0f) {
        VxMatrix m;
        m.SetIdentity();
        float s = Scale(minScale, maxScale);
        m[0][0] = s;
        m[1][1] = s;
        m[2][2] = s;
        return m;
    }

    /// Generate a random non-uniform scale matrix
    VxMatrix NonUniformScaleMatrix(float minScale = 0.1f, float maxScale = 10.0f) {
        VxMatrix m;
        m.SetIdentity();
        m[0][0] = Scale(minScale, maxScale);
        m[1][1] = Scale(minScale, maxScale);
        m[2][2] = Scale(minScale, maxScale);
        return m;
    }

    /**
     * @brief Generate a random TRS (Translation-Rotation-Scale) matrix.
     *
     * This is the MakeTRS helper from SIMDDispatchConsistencyTest.cpp.
     *
     * @param transRange Range for translation components
     * @param scaleMin Minimum scale factor
     * @param scaleMax Maximum scale factor
     * @return Combined TRS matrix
     */
    VxMatrix TRSMatrix(float transRange = 100.0f, float scaleMin = 0.5f, float scaleMax = 2.0f) {
        float tx = Float(-transRange, transRange);
        float ty = Float(-transRange, transRange);
        float tz = Float(-transRange, transRange);
        float eax = Float(-2.0f, 2.0f);
        float eay = Float(-2.0f, 2.0f);
        float eaz = Float(-2.0f, 2.0f);
        float sx = Scale(scaleMin, scaleMax);
        float sy = Scale(scaleMin, scaleMax);
        float sz = Scale(scaleMin, scaleMax);

        return MakeTRS(tx, ty, tz, eax, eay, eaz, sx, sy, sz);
    }

    /// Generate a random orthogonal matrix (rotation only, no scale)
    VxMatrix OrthogonalMatrix() {
        return RotationMatrix();
    }

    /// Generate a random perspective projection matrix
    VxMatrix PerspectiveMatrix() {
        VxMatrix m;
        float fov = Float(0.5f, 1.5f);
        float aspect = Float(0.5f, 2.0f);
        float nearPlane = Float(0.1f, 2.0f);
        float farPlane = nearPlane + Float(10.0f, 1000.0f);
        m.Perspective(fov, aspect, nearPlane, farPlane);
        return m;
    }

    /// Generate a random orthographic projection matrix
    VxMatrix OrthographicMatrix() {
        VxMatrix m;
        float zoom = Float(1.0f, 100.0f);
        float aspect = Float(0.5f, 2.0f);
        float nearPlane = Float(0.1f, 2.0f);
        float farPlane = nearPlane + Float(10.0f, 1000.0f);
        m.Orthographic(zoom, aspect, nearPlane, farPlane);
        return m;
    }

    //=========================================================================
    // Quaternion Generators
    //=========================================================================

    /// Generate a random unit quaternion (normalized, valid rotation)
    VxQuaternion UnitQuaternion() {
        VxQuaternion q(Float(-1.0f, 1.0f), Float(-1.0f, 1.0f),
                       Float(-1.0f, 1.0f), Float(-1.0f, 1.0f));
        // Avoid zero quaternion
        while ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) < 1e-10f) {
            q = VxQuaternion(Float(-1.0f, 1.0f), Float(-1.0f, 1.0f),
                            Float(-1.0f, 1.0f), Float(-1.0f, 1.0f));
        }
        q.Normalize();
        return q;
    }

    /// Generate a random quaternion (not normalized)
    VxQuaternion ArbitraryQuaternion() {
        return VxQuaternion(Float(-10.0f, 10.0f), Float(-10.0f, 10.0f),
                           Float(-10.0f, 10.0f), Float(-10.0f, 10.0f));
    }

    /// Generate a quaternion representing a small rotation
    VxQuaternion SmallRotationQuaternion(float maxAngle = 0.1f) {
        VxVector axis = UnitVector();
        float angle = Float(0.0f, maxAngle);
        VxQuaternion q;
        q.FromRotation(axis, angle);
        return q;
    }

    /// Generate a quaternion from axis-angle
    VxQuaternion QuaternionFromAxisAngle() {
        VxVector axis = UnitVector();
        float angle = Angle();
        VxQuaternion q;
        q.FromRotation(axis, angle);
        return q;
    }

    //=========================================================================
    // Geometric Type Generators
    //=========================================================================

    /// Generate a random ray
    VxRay Ray() {
        VxRay r;
        r.m_Origin = Vector();
        r.m_Direction = UnitVector();
        return r;
    }

    /// Generate a random plane
    VxPlane Plane() {
        VxPlane p;
        VxVector normal = UnitVector();
        VxVector point = Vector();
        p.Create(normal, point);
        return p;
    }

    /// Generate a random axis-aligned bounding box
    VxBbox Bbox(float maxSize = 10.0f) {
        VxVector center = Vector(-50.0f, 50.0f);
        VxVector halfSize(
            PositiveFloat(0.1f, maxSize),
            PositiveFloat(0.1f, maxSize),
            PositiveFloat(0.1f, maxSize)
        );
        return VxBbox(center - halfSize, center + halfSize);
    }

    /// Generate a random sphere
    VxSphere Sphere(float maxRadius = 10.0f) {
        VxVector center = Vector(-50.0f, 50.0f);
        float radius = PositiveFloat(0.1f, maxRadius);
        return VxSphere(center, radius);
    }

    /// Generate a random rect
    VxRect Rect(float maxSize = 100.0f) {
        float left = Float(0.0f, maxSize);
        float top = Float(0.0f, maxSize);
        float width = PositiveFloat(1.0f, maxSize);
        float height = PositiveFloat(1.0f, maxSize);
        return VxRect(left, top, left + width, top + height);
    }

    /// Generate a random frustum
    VxFrustum Frustum() {
        VxVector origin = Vector(-50.0f, 50.0f);
        VxVector right = UnitVector();
        VxVector up = CrossProduct(right, UnitVector());
        up.Normalize();
        VxVector dir = CrossProduct(right, up);
        dir.Normalize();

        float nearPlane = PositiveFloat(0.1f, 2.0f);
        float farPlane = nearPlane + PositiveFloat(10.0f, 100.0f);
        float fov = Float(0.5f, 1.5f);
        float aspect = Float(0.5f, 2.0f);

        return VxFrustum(origin, right, up, dir, nearPlane, farPlane, fov, aspect);
    }

    //=========================================================================
    // Static Helper Functions (from SIMDDispatchConsistencyTest.cpp)
    //=========================================================================

    /**
     * @brief Create a TRS matrix from components.
     *
     * This is the MakeTRS helper from SIMDDispatchConsistencyTest.cpp.
     */
    static VxMatrix MakeTRS(float tx, float ty, float tz,
                            float eax, float eay, float eaz,
                            float sx, float sy, float sz) {
        VxMatrix r;
        Vx3DMatrixFromEulerAngles(r, eax, eay, eaz);

        VxMatrix s;
        s.SetIdentity();
        s[0][0] = sx;
        s[1][1] = sy;
        s[2][2] = sz;

        VxMatrix rs;
        Vx3DMultiplyMatrix(rs, r, s);
        rs[3][0] = tx;
        rs[3][1] = ty;
        rs[3][2] = tz;
        rs[3][3] = 1.0f;
        return rs;
    }

private:
    uint32_t m_seed;
    std::mt19937 m_rng;
};

//=============================================================================
// Backward-Compatible Global Functions
//=============================================================================

/// Global generator instance (seed 42 for reproducibility)
inline RandomGenerator& GetDefaultGenerator() {
    static RandomGenerator g_defaultGenerator(42);
    return g_defaultGenerator;
}

/// Generate a random float (backward compatible)
inline float RandomFloat(float min = -100.0f, float max = 100.0f) {
    return GetDefaultGenerator().Float(min, max);
}

/// Generate a random VxVector (backward compatible)
inline VxVector RandomVector(float min = -100.0f, float max = 100.0f) {
    return GetDefaultGenerator().Vector(min, max);
}

/// Generate a random unit vector (backward compatible)
inline VxVector RandomUnitVector() {
    return GetDefaultGenerator().UnitVector();
}

/// Generate a random rotation matrix (backward compatible)
inline VxMatrix RandomRotationMatrix() {
    return GetDefaultGenerator().RotationMatrix();
}

/// Generate a random transform matrix (backward compatible)
inline VxMatrix RandomTransformMatrix() {
    return GetDefaultGenerator().TRSMatrix();
}

} // namespace VxMathTest

#endif // VXMATH_TEST_GENERATORS_H
