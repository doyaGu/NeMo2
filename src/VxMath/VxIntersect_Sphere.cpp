#include "VxIntersect.h"

#include "VxVector.h"
#include "VxRay.h"
#include "VxSphere.h"
#include "VxPlane.h"

/**
 * Solves quadratic equation ax^2 + bx + c = 0
 * @param a - coefficient of x^2
 * @param b - coefficient of x  
 * @param c - constant term
 * @param t1 - pointer to store first solution
 * @param t2 - pointer to store second solution
 * @return 1 if solutions found, 0 if no real solutions
 */
XBOOL QuadraticFormula(float a, float b, float c, float *t1, float *t2) {
    // Treat a == 0 OR unordered(a, 0) (i.e., NaN) as a linear equation.
    if (!(a > 0.0f || a < 0.0f)) {
        if (b == 0.0f) {
            *t1 = 0.0f;
            *t2 = 0.0f;
            return 0;
        }

        const float root = -(c / b);
        *t1 = root;
        *t2 = root;
        return 1;
    }

    const double disc = (double)b * (double)b - 4.0 * (double)a * (double)c;

    // Two distinct real solutions only when disc > 0.
    if (disc > 0.0) {
        const double sqrt_disc = sqrt(disc);
        const double inv2a = 1.0 / ((double)a + (double)a);
        *t1 = (float)((sqrt_disc - (double)b) * inv2a);
        *t2 = (float)((-(double)b - sqrt_disc) * inv2a);
        return 1;
    }

    // One repeated root only when disc == 0.
    if (disc == 0.0) {
        const float root = (float)(-(double)b / ((double)a + (double)a));
        *t1 = root;
        *t2 = root;
        return 1;
    }

    *t1 = 0.0f;
    *t2 = 0.0f;
    return 0;
}

//--------- Spheres

// Sphere-Sphere intersection with movement
XBOOL VxIntersect::SphereSphere(const VxSphere &iS1, const VxVector &iP1, const VxSphere &iS2, const VxVector &iP2,
                                float *oCollisionTime1, float *oCollisionTime2) {
    // Calculate movement vectors from sphere centers to target positions
    VxVector movement1 = iP1 - iS1.Center();
    VxVector movement2 = iP2 - iS2.Center();

    // Vector from sphere1 center to sphere2 center
    VxVector centerDiff = iS2.Center() - iS1.Center();

    // Relative movement vector (difference of movements)
    VxVector relativeMovement = movement2 - movement1;

    // Sum of radii and its square
    float radiusSum = iS1.Radius() + iS2.Radius();
    float radiusSumSquared = radiusSum * radiusSum;

    // Square distance between sphere centers
    float centerDistSquared = SquareMagnitude(centerDiff);

    // Check if spheres are already intersecting
    if (centerDistSquared <= radiusSumSquared) {
        *oCollisionTime1 = 0.0f;
        *oCollisionTime2 = 0.0f;
        return TRUE;
    }

    // Set up quadratic equation coefficients for collision detection
    float c = centerDistSquared - radiusSumSquared;
    float b = 2.0f * DotProduct(relativeMovement, centerDiff);
    float a = SquareMagnitude(relativeMovement);

    // Solve the quadratic equation
    if (!QuadraticFormula(a, b, c, oCollisionTime1, oCollisionTime2))
        return FALSE;

    // Ensure t1 <= t2 (swap if necessary)
    if (*oCollisionTime1 > *oCollisionTime2) {
        float temp = *oCollisionTime1;
        *oCollisionTime1 = *oCollisionTime2;
        *oCollisionTime2 = temp;
    }

    // Check if collision occurs within valid time range [0, 1]
    return (*oCollisionTime1 <= 1.0f && *oCollisionTime1 >= 0.0f);
}

// Intersection Ray - Sphere
int VxIntersect::RaySphere(const VxRay &iRay, const VxSphere &iSphere, VxVector *oInter1, VxVector *oInter2) {
    // Normalize the ray direction vector.
    const float invMagnitude = 1.0f / sqrtf(SquareMagnitude(iRay.m_Direction));
    const VxVector normalizedDir = iRay.m_Direction * invMagnitude;

    // Vector from ray origin to sphere center
    const VxVector &center = iSphere.Center();
    float toCenterX = center.x - iRay.m_Origin.x;
    float toCenterY = center.y - iRay.m_Origin.y;
    float toCenterZ = center.z - iRay.m_Origin.z;

    // Project the toCenter vector onto the normalized ray direction
    float projection = normalizedDir.z * toCenterZ + normalizedDir.y * toCenterY + toCenterX * normalizedDir.x;

    // Calculate the discriminant using geometric approach
    float radius = iSphere.Radius();
    float radiusSquared = radius * radius;
    float discriminant = radiusSquared - (toCenterZ * toCenterZ + toCenterY * toCenterY + toCenterX * toCenterX - projection * projection);

    if (discriminant < 0.0f)
        return 0; // No intersection

    if (discriminant == 0.0f) {
        // Single intersection point (ray is tangent to sphere)
        *oInter1 = iRay.m_Origin + normalizedDir * projection;
        return 1;
    }

    // Two intersection points
    float sqrtDiscriminant = sqrtf(discriminant);
    float t1 = projection - sqrtDiscriminant;
    float t2 = projection + sqrtDiscriminant;

    // Compute first intersection point
    *oInter1 = iRay.m_Origin + normalizedDir * t1;

    // Compute second intersection point
    *oInter2 = iRay.m_Origin + normalizedDir * t2;

    return 2;
}

// Intersection Sphere - AABB
XBOOL VxIntersect::SphereAABB(const VxSphere &iSphere, const VxBbox &iBox) {
    const VxVector boxCenter = (iBox.Min + iBox.Max) * 0.5f;

    VxPlane plane;
    plane.Create(boxCenter - iSphere.Center(), iSphere.Center());

    const float signedDist = plane.Classify(iBox);
    return (fabsf(signedDist) <= iSphere.Radius());
}
