#include "VxIntersect.h"

#include "VxVector.h"
#include "VxRay.h"
#include "VxPlane.h"

//---------- Planes

// Intersection Ray - Plane
XBOOL VxIntersect::RayPlane(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist) {
    // Calculate the denominator: dot(ray.Dir, plane.Normal)
    float denom = DotProduct(plane.m_Normal, ray.m_Direction);

    // Check if ray is parallel to plane (or nearly parallel)
    if (XAbs(denom) < EPSILON)
        return FALSE;

    // Calculate distance along ray
    dist = -(DotProduct(plane.m_Normal, ray.m_Origin) + plane.m_D) / denom;

    // For a ray, intersection must be in positive direction (with epsilon tolerance)
    if (dist < -EPSILON)
        return FALSE;

    // Calculate intersection point
    point = ray.m_Origin + ray.m_Direction * dist;
    return TRUE;
}

// Intersection Ray - Plane with culling (only intersect from front)
XBOOL VxIntersect::RayPlaneCulled(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist) {
    // Calculate the denominator: dot(ray.Dir, plane.Normal)
    float denom = DotProduct(plane.m_Normal, ray.m_Direction);

    // Ensure ray is coming from the front of the plane (denom < -epsilon)
    if (denom >= -EPSILON)
        return FALSE;

    // Calculate distance along ray
    dist = -(DotProduct(plane.m_Normal, ray.m_Origin) + plane.m_D) / denom;

    // For a ray, intersection must be in positive direction (with epsilon tolerance)
    if (dist < -EPSILON)
        return FALSE;

    // Calculate intersection point
    point = ray.m_Origin + ray.m_Direction * dist;
    return TRUE;
}

// Intersection Segment - Plane
XBOOL VxIntersect::SegmentPlane(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist) {
    // Calculate the denominator: dot(ray.Dir, plane.Normal)
    float denom = DotProduct(plane.m_Normal, ray.m_Direction);

    // Check if segment is parallel to plane (or nearly parallel)
    if (XAbs(denom) < EPSILON)
        return FALSE;

    // Calculate distance along ray
    dist = -(DotProduct(plane.m_Normal, ray.m_Origin) + plane.m_D) / denom;

    // For a segment, intersection must be between -epsilon and 1+epsilon
    if (dist < -EPSILON)
        return FALSE;
    if (dist > 1.0000001f)
        return FALSE;

    // Calculate intersection point
    point = ray.m_Origin + ray.m_Direction * dist;
    return TRUE;
}

// Intersection Segment - Plane with culling
XBOOL VxIntersect::SegmentPlaneCulled(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist) {
    // Calculate the denominator: dot(ray.Dir, plane.Normal)
    float denom = DotProduct(plane.m_Normal, ray.m_Direction);

    // Ensure segment is coming from the front of the plane (denom < -epsilon)
    if (denom >= -EPSILON)
        return FALSE;

    // Calculate distance along ray
    dist = -(DotProduct(plane.m_Normal, ray.m_Origin) + plane.m_D) / denom;

    // For a segment, intersection must be between -epsilon and 1+epsilon
    if (dist < -EPSILON)
        return FALSE;
    if (dist > 1.0000001f)
        return FALSE;

    // Calculate intersection point
    point = ray.m_Origin + ray.m_Direction * dist;
    return TRUE;
}

// Intersection Line - Plane
XBOOL VxIntersect::LinePlane(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist) {
    // Calculate the denominator: dot(ray.Dir, plane.Normal)
    float denom = DotProduct(plane.m_Normal, ray.m_Direction);

    // Check if line is parallel to plane (or nearly parallel)
    if (XAbs(denom) < EPSILON)
        return FALSE;

    // Calculate distance along ray
    dist = -(DotProduct(plane.m_Normal, ray.m_Origin) + plane.m_D) / denom;

    // Calculate intersection point
    point = ray.m_Origin + ray.m_Direction * dist;
    return TRUE;
}

// Intersection Box - Plane
XBOOL VxIntersect::BoxPlane(const VxBbox &box, const VxPlane &plane) {
    VxVector minPt = box.Min;
    VxVector maxPt = box.Max;

    // For each axis, select min or max depending on normal direction
    for (int i = 0; i < 3; i++) {
        if (plane.m_Normal[i] < 0.0f) {
            minPt[i] = box.Max[i];
            maxPt[i] = box.Min[i];
        }
    }

    // Calculate signed distances from extreme points to plane
    float minDist = DotProduct(plane.m_Normal, minPt) + plane.m_D;
    float maxDist = DotProduct(plane.m_Normal, maxPt) + plane.m_D;

    // Box intersects plane if:
    // - minDist <= 0 AND maxDist >= 0 (spans the plane)
    // - OR maxDist == 0 (touches the plane)
    return (minDist <= 0.0f && maxDist >= 0.0f) || (maxDist == 0.0f);
}

// Intersection Box - Plane with transformation
XBOOL VxIntersect::BoxPlane(const VxBbox &box, const VxMatrix &mat, const VxPlane &plane) {
    // Calculate half-size vector
    VxVector halfSize = (box.Max - box.Min) * 0.5f;
    
    // Calculate the three transformed half-extent vectors
    VxVector axis0 = VxVector(mat[0][0], mat[0][1], mat[0][2]) * halfSize.x;
    VxVector axis1 = VxVector(mat[1][0], mat[1][1], mat[1][2]) * halfSize.y;
    VxVector axis2 = VxVector(mat[2][0], mat[2][1], mat[2][2]) * halfSize.z;

    // Calculate the projection radius of box onto plane normal
    float radius = XAbs(DotProduct(plane.m_Normal, axis0)) +
                   XAbs(DotProduct(plane.m_Normal, axis1)) +
                   XAbs(DotProduct(plane.m_Normal, axis2));

    // Calculate box center and transform it
    VxVector center = (box.Min + box.Max) * 0.5f;
    VxVector transformedCenter;
    Vx3DMultiplyMatrixVector(&transformedCenter, mat, &center);

    // Calculate distance from box center to plane
    float dist = DotProduct(plane.m_Normal, transformedCenter) + plane.m_D;

    // Box intersects plane if |distance| <= radius
    if (XAbs(dist) <= radius) {
        return TRUE;
    }
    
    return dist == 0.0f;
}

// Intersection Face - Plane
XBOOL VxIntersect::FacePlane(const VxVector &A0, const VxVector &A1, const VxVector &A2, const VxPlane &plane) {
    // Get signed distances from vertices to plane
    float d0 = plane.Classify(A0);
    float d1 = plane.Classify(A1);
    float d2 = plane.Classify(A2);

    // Face intersects plane if vertices are on different sides
    // or at least one vertex is on the plane
    if ((d0 * d1 <= 0.0f) || (d1 * d2 <= 0.0f) || (d2 * d0 <= 0.0f))
        return TRUE;

    return FALSE;
}

// Intersection of 3 planes using Cramer's rule
XBOOL VxIntersect::Planes(const VxPlane &plane1, const VxPlane &plane2, const VxPlane &plane3, VxVector &p) {
    // Set up the coefficient matrix
    VxMatrix mat;
    mat[0][0] = plane1.m_Normal.x; mat[0][1] = plane1.m_Normal.y; mat[0][2] = plane1.m_Normal.z;
    mat[1][0] = plane2.m_Normal.x; mat[1][1] = plane2.m_Normal.y; mat[1][2] = plane2.m_Normal.z;
    mat[2][0] = plane3.m_Normal.x; mat[2][1] = plane3.m_Normal.y; mat[2][2] = plane3.m_Normal.z;

    // Calculate determinant
    float det = Vx3DMatrixDeterminant(mat);
    if (det == 0.0f)
        return FALSE;

    // Calculate x-coordinate using Cramer's rule
    mat[0][0] = -plane1.m_D; mat[0][1] = plane1.m_Normal.y; mat[0][2] = plane1.m_Normal.z;
    mat[1][0] = -plane2.m_D; mat[1][1] = plane2.m_Normal.y; mat[1][2] = plane2.m_Normal.z;
    mat[2][0] = -plane3.m_D; mat[2][1] = plane3.m_Normal.y; mat[2][2] = plane3.m_Normal.z;
    float detX = Vx3DMatrixDeterminant(mat);

    // Calculate y-coordinate using Cramer's rule
    mat[0][0] = plane1.m_Normal.x; mat[0][1] = -plane1.m_D; mat[0][2] = plane1.m_Normal.z;
    mat[1][0] = plane2.m_Normal.x; mat[1][1] = -plane2.m_D; mat[1][2] = plane2.m_Normal.z;
    mat[2][0] = plane3.m_Normal.x; mat[2][1] = -plane3.m_D; mat[2][2] = plane3.m_Normal.z;
    float detY = Vx3DMatrixDeterminant(mat);

    // Calculate z-coordinate using Cramer's rule
    mat[0][0] = plane1.m_Normal.x; mat[0][1] = plane1.m_Normal.y; mat[0][2] = -plane1.m_D;
    mat[1][0] = plane2.m_Normal.x; mat[1][1] = plane2.m_Normal.y; mat[1][2] = -plane2.m_D;
    mat[2][0] = plane3.m_Normal.x; mat[2][1] = plane3.m_Normal.y; mat[2][2] = -plane3.m_D;
    float detZ = Vx3DMatrixDeterminant(mat);

    // Calculate final result
    float invDet = 1.0f / det;
    p.x = detX * invDet;
    p.y = detY * invDet;
    p.z = detZ * invDet;

    return TRUE;
}