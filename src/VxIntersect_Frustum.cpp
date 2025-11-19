#include "VxIntersect.h"

#include "VxVector.h"
#include "VxRay.h"
#include "VxPlane.h"
#include "VxFrustum.h"

//--------- Frustum

// Intersection Frustum - Face
XBOOL VxIntersect::FrustumFace(const VxFrustum &frustum, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2) {
    const VxPlane &nearPlane = frustum.GetNearPlane();
    const VxPlane &farPlane = frustum.GetFarPlane();
    const VxPlane &leftPlane = frustum.GetLeftPlane();
    const VxPlane &rightPlane = frustum.GetRightPlane();
    const VxPlane &upPlane = frustum.GetUpPlane();
    const VxPlane &bottomPlane = frustum.GetBottomPlane();

    // Test against near plane
    float d0 = DotProduct(nearPlane.m_Normal, pt0) + nearPlane.m_D;
    float d1 = DotProduct(nearPlane.m_Normal, pt1) + nearPlane.m_D;
    float d2 = DotProduct(nearPlane.m_Normal, pt2) + nearPlane.m_D;

    float minDist = d0;
    if (d0 > 0.0f) {
        if (d1 < 0.0f) goto far_plane_test;
        if (d1 < minDist) minDist = d1;
    } else {
        if (d1 > 0.0f) goto far_plane_test;
        if (d1 > minDist) minDist = d1;
    }

    if (minDist > 0.0f) {
        if (d2 < 0.0f) goto far_plane_test;
        if (d2 < minDist) minDist = d2;
    } else {
        if (d2 > 0.0f) goto far_plane_test;
        if (d2 > minDist) minDist = d2;
    }

    if (minDist > 0.0f)
        return FALSE;

far_plane_test:
    // Test against far plane
    d0 = DotProduct(farPlane.m_Normal, pt0) + farPlane.m_D;
    d1 = DotProduct(farPlane.m_Normal, pt1) + farPlane.m_D;
    d2 = DotProduct(farPlane.m_Normal, pt2) + farPlane.m_D;

    minDist = d0;
    if (d0 > 0.0f) {
        if (d1 < 0.0f) goto left_plane_test;
        if (d1 < minDist) minDist = d1;
    } else {
        if (d1 > 0.0f) goto left_plane_test;
        if (d1 > minDist) minDist = d1;
    }

    if (minDist > 0.0f) {
        if (d2 < 0.0f) goto left_plane_test;
        if (d2 < minDist) minDist = d2;
    } else {
        if (d2 > 0.0f) goto left_plane_test;
        if (d2 > minDist) minDist = d2;
    }

    if (minDist > 0.0f)
        return FALSE;

left_plane_test:
    // Test against left plane
    d0 = DotProduct(leftPlane.m_Normal, pt0) + leftPlane.m_D;
    d1 = DotProduct(leftPlane.m_Normal, pt1) + leftPlane.m_D;
    d2 = DotProduct(leftPlane.m_Normal, pt2) + leftPlane.m_D;

    minDist = d0;
    if (d0 > 0.0f) {
        if (d1 < 0.0f) goto right_plane_test;
        if (d1 < minDist) minDist = d1;
    } else {
        if (d1 > 0.0f) goto right_plane_test;
        if (d1 > minDist) minDist = d1;
    }

    if (minDist > 0.0f) {
        if (d2 < 0.0f) goto right_plane_test;
        if (d2 < minDist) minDist = d2;
    } else {
        if (d2 > 0.0f) goto right_plane_test;
        if (d2 > minDist) minDist = d2;
    }

    if (minDist > 0.0f)
        return FALSE;

right_plane_test:
    // Test against right plane
    d0 = DotProduct(rightPlane.m_Normal, pt0) + rightPlane.m_D;
    d1 = DotProduct(rightPlane.m_Normal, pt1) + rightPlane.m_D;
    d2 = DotProduct(rightPlane.m_Normal, pt2) + rightPlane.m_D;

    minDist = d0;
    if (d0 > 0.0f) {
        if (d1 < 0.0f) goto bottom_plane_test;
        if (d1 < minDist) minDist = d1;
    } else {
        if (d1 > 0.0f) goto bottom_plane_test;
        if (d1 > minDist) minDist = d1;
    }

    if (minDist > 0.0f) {
        if (d2 < 0.0f) goto bottom_plane_test;
        if (d2 < minDist) minDist = d2;
    } else {
        if (d2 > 0.0f) goto bottom_plane_test;
        if (d2 > minDist) minDist = d2;
    }

    if (minDist > 0.0f)
        return FALSE;

bottom_plane_test:
    // Test against bottom plane using ClassifyFace
    float result = bottomPlane.ClassifyFace(pt0, pt1, pt2);
    if (result > 0.0f)
        return FALSE;

    // Test against up plane using ClassifyFace
    result = upPlane.ClassifyFace(pt0, pt1, pt2);
    return (result <= 0.0f);
}

// Intersection Frustum - AABB
XBOOL VxIntersect::FrustumAABB(const VxFrustum &frustum, const VxBbox &box) {
    // Compute box center and half-size
    VxVector halfSize = (box.Max - box.Min) * 0.5f;
    VxVector center = (box.Max + box.Min) * 0.5f;

    // Transform center relative to frustum origin
    VxVector relativeCenter = center - frustum.GetOrigin();

    // Create axis matrix (identity for AABB)
    VxVector axis[3];
    axis[0] = VxVector(1.0f, 0.0f, 0.0f);
    axis[1] = VxVector(0.0f, 1.0f, 0.0f);
    axis[2] = VxVector(0.0f, 0.0f, 1.0f);

    // Test against near plane
    float centerDist = DotProduct(relativeCenter, frustum.GetDir()) * frustum.GetDMin();
    float radius = XAbs(DotProduct(frustum.GetUp(), axis[0])) * frustum.GetUBound() * halfSize.x +
                   XAbs(DotProduct(frustum.GetUp(), axis[1])) * frustum.GetUBound() * halfSize.y +
                   XAbs(DotProduct(frustum.GetUp(), axis[2])) * frustum.GetUBound() * halfSize.z +
                   XAbs(DotProduct(frustum.GetRight(), axis[0])) * frustum.GetRBound() * halfSize.x +
                   XAbs(DotProduct(frustum.GetRight(), axis[1])) * frustum.GetRBound() * halfSize.y +
                   XAbs(DotProduct(frustum.GetRight(), axis[2])) * frustum.GetRBound() * halfSize.z;

    float nearBound = centerDist - radius;
    float farBound = centerDist + radius;

    if (nearBound < 0.0f)
        nearBound *= frustum.GetDRatio();
    if (farBound < 0.0f)
        farBound *= frustum.GetDRatio();

    // Check if box intersects with frustum range
    if (centerDist + radius < nearBound)
        return FALSE;
    if (centerDist - radius > farBound)
        return FALSE;

    // Test against all frustum planes
    const VxPlane &nearPlane = frustum.GetNearPlane();
    const VxPlane &farPlane = frustum.GetFarPlane();
    const VxPlane &leftPlane = frustum.GetLeftPlane();
    const VxPlane &rightPlane = frustum.GetRightPlane();
    const VxPlane &upPlane = frustum.GetUpPlane();
    const VxPlane &bottomPlane = frustum.GetBottomPlane();

    // For each plane, compute distance from center and radius
    float d, r;

    // Near plane
    d = DotProduct(relativeCenter, nearPlane.m_Normal);
    r = XAbs(DotProduct(nearPlane.m_Normal, axis[0])) * halfSize.x +
        XAbs(DotProduct(nearPlane.m_Normal, axis[1])) * halfSize.y +
        XAbs(DotProduct(nearPlane.m_Normal, axis[2])) * halfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Far plane
    d = DotProduct(relativeCenter, farPlane.m_Normal);
    r = XAbs(DotProduct(farPlane.m_Normal, axis[0])) * halfSize.x +
        XAbs(DotProduct(farPlane.m_Normal, axis[1])) * halfSize.y +
        XAbs(DotProduct(farPlane.m_Normal, axis[2])) * halfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Left plane
    d = DotProduct(relativeCenter, leftPlane.m_Normal);
    r = XAbs(DotProduct(leftPlane.m_Normal, axis[0])) * halfSize.x +
        XAbs(DotProduct(leftPlane.m_Normal, axis[1])) * halfSize.y +
        XAbs(DotProduct(leftPlane.m_Normal, axis[2])) * halfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Right plane
    d = DotProduct(relativeCenter, rightPlane.m_Normal);
    r = XAbs(DotProduct(rightPlane.m_Normal, axis[0])) * halfSize.x +
        XAbs(DotProduct(rightPlane.m_Normal, axis[1])) * halfSize.y +
        XAbs(DotProduct(rightPlane.m_Normal, axis[2])) * halfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Up plane
    d = DotProduct(relativeCenter, upPlane.m_Normal);
    r = XAbs(DotProduct(upPlane.m_Normal, axis[0])) * halfSize.x +
        XAbs(DotProduct(upPlane.m_Normal, axis[1])) * halfSize.y +
        XAbs(DotProduct(upPlane.m_Normal, axis[2])) * halfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Bottom plane
    d = DotProduct(relativeCenter, bottomPlane.m_Normal);
    r = XAbs(DotProduct(bottomPlane.m_Normal, axis[0])) * halfSize.x +
        XAbs(DotProduct(bottomPlane.m_Normal, axis[1])) * halfSize.y +
        XAbs(DotProduct(bottomPlane.m_Normal, axis[2])) * halfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    return TRUE;
}

// Intersection Frustum - OBB
XBOOL VxIntersect::FrustumOBB(const VxFrustum &frustum, const VxBbox &box, const VxMatrix &mat) {
    // Compute box center and half-size
    VxVector halfSize = (box.Max - box.Min) * 0.5f;
    VxVector center = (box.Max + box.Min) * 0.5f;

    // Transform center to world space
    VxVector worldCenter;
    Vx3DMultiplyMatrixVector(&worldCenter, mat, &center);

    // Transform center relative to frustum origin
    VxVector relativeCenter = worldCenter - frustum.GetOrigin();

    // Extract and normalize transformation axes, compute scaled half-sizes
    VxVector axis[3];
    axis[0] = VxVector(mat[0][0], mat[0][1], mat[0][2]);
    axis[1] = VxVector(mat[1][0], mat[1][1], mat[1][2]);
    axis[2] = VxVector(mat[2][0], mat[2][1], mat[2][2]);

    float axisLength[3];
    axisLength[0] = Magnitude(axis[0]);
    axisLength[1] = Magnitude(axis[1]);
    axisLength[2] = Magnitude(axis[2]);

    axis[0] /= axisLength[0];
    axis[1] /= axisLength[1];
    axis[2] /= axisLength[2];

    VxVector scaledHalfSize;
    scaledHalfSize.x = halfSize.x * axisLength[0];
    scaledHalfSize.y = halfSize.y * axisLength[1];
    scaledHalfSize.z = halfSize.z * axisLength[2];

    // Test against all frustum planes
    const VxPlane &nearPlane = frustum.GetNearPlane();
    const VxPlane &farPlane = frustum.GetFarPlane();
    const VxPlane &leftPlane = frustum.GetLeftPlane();
    const VxPlane &rightPlane = frustum.GetRightPlane();
    const VxPlane &upPlane = frustum.GetUpPlane();
    const VxPlane &bottomPlane = frustum.GetBottomPlane();

    // For each plane, compute distance from center and radius
    float d, r;

    // Near plane
    d = DotProduct(relativeCenter, nearPlane.m_Normal);
    r = XAbs(DotProduct(nearPlane.m_Normal, axis[0])) * scaledHalfSize.x +
        XAbs(DotProduct(nearPlane.m_Normal, axis[1])) * scaledHalfSize.y +
        XAbs(DotProduct(nearPlane.m_Normal, axis[2])) * scaledHalfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Far plane
    d = DotProduct(relativeCenter, farPlane.m_Normal);
    r = XAbs(DotProduct(farPlane.m_Normal, axis[0])) * scaledHalfSize.x +
        XAbs(DotProduct(farPlane.m_Normal, axis[1])) * scaledHalfSize.y +
        XAbs(DotProduct(farPlane.m_Normal, axis[2])) * scaledHalfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Left plane
    d = DotProduct(relativeCenter, leftPlane.m_Normal);
    r = XAbs(DotProduct(leftPlane.m_Normal, axis[0])) * scaledHalfSize.x +
        XAbs(DotProduct(leftPlane.m_Normal, axis[1])) * scaledHalfSize.y +
        XAbs(DotProduct(leftPlane.m_Normal, axis[2])) * scaledHalfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Right plane
    d = DotProduct(relativeCenter, rightPlane.m_Normal);
    r = XAbs(DotProduct(rightPlane.m_Normal, axis[0])) * scaledHalfSize.x +
        XAbs(DotProduct(rightPlane.m_Normal, axis[1])) * scaledHalfSize.y +
        XAbs(DotProduct(rightPlane.m_Normal, axis[2])) * scaledHalfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Up plane
    d = DotProduct(relativeCenter, upPlane.m_Normal);
    r = XAbs(DotProduct(upPlane.m_Normal, axis[0])) * scaledHalfSize.x +
        XAbs(DotProduct(upPlane.m_Normal, axis[1])) * scaledHalfSize.y +
        XAbs(DotProduct(upPlane.m_Normal, axis[2])) * scaledHalfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    // Bottom plane
    d = DotProduct(relativeCenter, bottomPlane.m_Normal);
    r = XAbs(DotProduct(bottomPlane.m_Normal, axis[0])) * scaledHalfSize.x +
        XAbs(DotProduct(bottomPlane.m_Normal, axis[1])) * scaledHalfSize.y +
        XAbs(DotProduct(bottomPlane.m_Normal, axis[2])) * scaledHalfSize.z;
    if (d + r < 0.0f || d - r > 0.0f)
        return FALSE;

    return TRUE;
}

// Intersection Frustum - Box (general)
XBOOL VxIntersect::FrustumBox(const VxFrustum &frustum, const VxBbox &box, const VxMatrix &mat) {
    return FrustumOBB(frustum, box, mat);
}