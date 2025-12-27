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
    // Handle degenerate case where a = 0 (not a quadratic equation)
    if (a == 0.0f) {
        if (b != 0.0f) {
            // Linear equation: bx + c = 0, solution: x = -c/b
            float linear_root = -c / b;
            *t1 = linear_root;
            *t2 = linear_root;
            return 1;
        } else {
            // Degenerate case: 0x + c = 0
            // No valid solution for our purposes
            *t1 = 0.0f;
            *t2 = 0.0f;
            return 0;
        }
    }

    // Calculate discriminant: b^2 - 4ac
    // Use double precision for intermediate calculations to avoid precision loss
    double discriminant = (double) b * b - 4.0 * (double) a * c;

    // Check if discriminant is negative (no real solutions)
    if (discriminant < 0.0) {
        return 0;
    }

    // Check if discriminant is zero (one repeated root)
    if (discriminant == 0.0) {
        float repeated_root = -b / (2.0f * a);
        *t1 = repeated_root;
        *t2 = repeated_root;
        return 1;
    }

    // Two distinct real solutions
    double sqrt_discriminant = sqrt(discriminant);
    double denominator = 2.0 * (double) a;

    *t1 = (float) ((-b + sqrt_discriminant) / denominator);
    *t2 = (float) ((-b - sqrt_discriminant) / denominator);

    return 1;
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
    // Normalize the ray direction vector
    float invMagnitude = 1.0f / sqrtf(SquareMagnitude(iRay.m_Direction));
    VxVector normalizedDir = iRay.m_Direction * invMagnitude;

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
    // Get sphere center via accessor
    const VxVector &sphereCenter = iSphere.Center();
    
    // Calculate the center and half-extents of the bounding box
    VxVector boxSum(iBox.Min.x + iBox.Max.x, iBox.Min.y + iBox.Max.y, iBox.Min.z + iBox.Max.z);
    VxVector boxCenter(boxSum.x * 0.5f, boxSum.y * 0.5f, boxSum.z * 0.5f);

    // Vector from box center to sphere center
    VxVector centerDiff;
    centerDiff.x = boxCenter.x - sphereCenter.x;
    centerDiff.y = boxCenter.y - sphereCenter.y;
    centerDiff.z = boxCenter.z - sphereCenter.z;

    // Create a plane from the center difference and sphere center
    VxPlane plane;
    plane.m_Normal.z = 0.0f;
    plane.Create(centerDiff, sphereCenter);

    // Initialize min and max points for box-plane intersection test
    VxVector minPt = iBox.Min;
    VxVector maxPt = iBox.Max;

    // For each axis, select the appropriate extreme point based on plane normal
    for (int i = 0; i < 3; i++) {
        if (*(&plane.m_Normal.x + i) < 0.0f) {
            *(&minPt.x + i) = *(&iBox.Max.x + i);
            *(&maxPt.x + i) = *(&iBox.Min.x + i);
        }
    }

    // Calculate signed distances from extreme points to the plane
    float minDist = plane.m_Normal.z * minPt.z + plane.m_Normal.y * minPt.y + minPt.x * plane.m_Normal.x + plane.m_D;

    float finalDist;
    if (minDist <= 0.0f) {
        float maxDist = plane.m_Normal.z * maxPt.z + plane.m_Normal.y * maxPt.y + maxPt.x * plane.m_Normal.x + plane.m_D;
        if (maxDist >= 0.0f) {
            finalDist = 0.0f;
        } else {
            finalDist = maxDist;
        }
    } else {
        finalDist = minDist;
    }

    // Return true if the absolute distance is within the sphere's radius
    return (fabsf(finalDist) <= iSphere.Radius());
}
