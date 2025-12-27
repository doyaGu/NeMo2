#include "VxIntersect.h"

#include "VxVector.h"
#include "VxRay.h"
#include "VxPlane.h"
#include "VxFrustum.h"
#include "VxMatrix.h"

//--------- Frustum

// Intersection Frustum - Face
// Matches binary at 0x24289c90
XBOOL VxIntersect::FrustumFace(const VxFrustum &frustum, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2) {
    float d0, d1, d2, minDist;
    const VxPlane &nearPlane = frustum.GetNearPlane();
    const VxPlane &farPlane = frustum.GetFarPlane();
    const VxPlane &leftPlane = frustum.GetLeftPlane();
    const VxPlane &rightPlane = frustum.GetRightPlane();
    const VxPlane &bottomPlane = frustum.GetBottomPlane();
    const VxPlane &upPlane = frustum.GetUpPlane();

    // Test against near plane
    d0 = nearPlane.Classify(pt0);
    d1 = nearPlane.Classify(pt1);
    
    minDist = d0;
    if (d0 >= 0.0f) {
        if (d1 < 0.0f) goto far_plane_test;
        if (d1 < minDist) minDist = d1;
    } else {
        if (d1 >= 0.0f) goto far_plane_test;
        if (d1 > minDist) minDist = d1;
    }

    d2 = nearPlane.Classify(pt2);
    if (minDist >= 0.0f) {
        if (d2 < 0.0f) goto far_plane_test;
        if (d2 < minDist) minDist = d2;
    } else {
        if (d2 >= 0.0f) goto far_plane_test;
        if (d2 > minDist) minDist = d2;
    }

    if (minDist >= 0.0f)
        return FALSE;

far_plane_test:
    d0 = farPlane.Classify(pt0);
    d1 = farPlane.Classify(pt1);

    minDist = d0;
    if (d0 >= 0.0f) {
        if (d1 < 0.0f) goto left_plane_test;
        if (d1 < minDist) minDist = d1;
    } else {
        if (d1 >= 0.0f) goto left_plane_test;
        if (d1 > minDist) minDist = d1;
    }

    d2 = farPlane.Classify(pt2);
    if (minDist >= 0.0f) {
        if (d2 < 0.0f) goto left_plane_test;
        if (d2 < minDist) minDist = d2;
    } else {
        if (d2 >= 0.0f) goto left_plane_test;
        if (d2 > minDist) minDist = d2;
    }

    if (minDist >= 0.0f)
        return FALSE;

left_plane_test:
    d0 = leftPlane.Classify(pt0);
    d1 = leftPlane.Classify(pt1);

    minDist = d0;
    if (d0 >= 0.0f) {
        if (d1 < 0.0f) goto right_plane_test;
        if (d1 < minDist) minDist = d1;
    } else {
        if (d1 >= 0.0f) goto right_plane_test;
        if (d1 > minDist) minDist = d1;
    }

    d2 = leftPlane.Classify(pt2);
    if (minDist >= 0.0f) {
        if (d2 < 0.0f) goto right_plane_test;
        if (d2 < minDist) minDist = d2;
    } else {
        if (d2 >= 0.0f) goto right_plane_test;
        if (d2 > minDist) minDist = d2;
    }

    if (minDist >= 0.0f)
        return FALSE;

right_plane_test:
    d0 = rightPlane.Classify(pt0);
    d1 = rightPlane.Classify(pt1);

    minDist = d0;
    if (d0 >= 0.0f) {
        if (d1 < 0.0f) goto bottom_plane_test;
        if (d1 < minDist) minDist = d1;
    } else {
        if (d1 >= 0.0f) goto bottom_plane_test;
        if (d1 > minDist) minDist = d1;
    }

    d2 = rightPlane.Classify(pt2);
    if (minDist >= 0.0f) {
        if (d2 < 0.0f) goto bottom_plane_test;
        if (d2 < minDist) minDist = d2;
    } else {
        if (d2 >= 0.0f) goto bottom_plane_test;
        if (d2 > minDist) minDist = d2;
    }

    if (minDist >= 0.0f)
        return FALSE;

bottom_plane_test:
    // Test against bottom plane using ClassifyFace
    if (bottomPlane.ClassifyFace(pt0, pt1, pt2) > 0.0f)
        return FALSE;

    // Test against up plane using ClassifyFace
    return (upPlane.ClassifyFace(pt0, pt1, pt2) <= 0.0f);
}

// Helper inline function for axis separation test used by FrustumAABB and FrustumOBB
// Returns FALSE if separated, TRUE if potentially overlapping
static inline XBOOL TestFrustumAxis(
    float centerProj, 
    float boxRadius,
    float frustumRadius,
    float frustumCenterMin,
    float dRatio)
{
    float minBound = frustumCenterMin - frustumRadius;
    if (minBound < 0.0f)
        minBound = minBound * dRatio;
    
    float maxBound = frustumCenterMin + frustumRadius;
    if (maxBound < 0.0f)
        maxBound = maxBound * dRatio;
    
    if (centerProj + boxRadius < minBound)
        return FALSE;
    if (centerProj - boxRadius > maxBound)
        return FALSE;
    
    return TRUE;
}

// Intersection Frustum - AABB
// Matches binary at 0x2428a160
XBOOL VxIntersect::FrustumAABB(const VxFrustum &frustum, const VxBbox &box) {
    // Get frustum properties via accessors
    const VxVector &origin = frustum.GetOrigin();
    const VxVector &dir = frustum.GetDir();
    const VxVector &up = frustum.GetUp();
    const VxVector &right = frustum.GetRight();
    float rBound = frustum.GetRBound();
    float uBound = frustum.GetUBound();
    float dMin = frustum.GetDMin();
    float dRatio = frustum.GetDRatio();
    
    const VxPlane &farPlane = frustum.GetFarPlane();
    const VxPlane &bottomPlane = frustum.GetBottomPlane();
    const VxPlane &upPlane = frustum.GetUpPlane();
    const VxPlane &leftPlane = frustum.GetLeftPlane();
    const VxPlane &rightPlane = frustum.GetRightPlane();

    // Compute box half-size and center
    VxVector halfSize = (box.Max - box.Min) * 0.5f;
    VxVector center = (box.Max + box.Min) * 0.5f;

    // Transform center relative to frustum origin
    VxVector relativeCenter(
        center.x - origin.x, 
        center.y - origin.y, 
        center.z - origin.z
    );

    // For AABB, box axes are identity - create 3x3 identity matrix
    float boxAxis[3][3];
    for (int i = 0; i < 3; i++) {
        boxAxis[i][0] = 0.0f;
        boxAxis[i][1] = 0.0f;
        boxAxis[i][2] = 0.0f;
    }
    boxAxis[0][0] = 1.0f;
    boxAxis[1][1] = 1.0f;
    boxAxis[2][2] = 1.0f;

    float centerProj, boxRadius, frustumRadius, frustumCenterMin;

    // Test X axis (1, 0, 0)
    centerProj = relativeCenter.x * boxAxis[0][0] + relativeCenter.y * boxAxis[0][1] + relativeCenter.z * boxAxis[0][2];
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[0][0] * boxAxis[i][0] + boxAxis[0][1] * boxAxis[i][1] + boxAxis[0][2] * boxAxis[i][2]) * (&halfSize.x)[i];
    }
    frustumRadius = fabsf((up.z + up.y) * 0.0f + up.x) * uBound +
                    fabsf((right.z + right.y) * 0.0f + right.x) * rBound;
    frustumCenterMin = ((dir.z + dir.y) * 0.0f + dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Y axis (0, 1, 0)
    centerProj = relativeCenter.x * boxAxis[1][0] + relativeCenter.y * boxAxis[1][1] + relativeCenter.z * boxAxis[1][2];
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[1][0] * boxAxis[i][0] + boxAxis[1][1] * boxAxis[i][1] + boxAxis[1][2] * boxAxis[i][2]) * (&halfSize.x)[i];
    }
    frustumRadius = fabsf((up.z + up.x) * 0.0f + up.y) * uBound +
                    fabsf((right.z + right.x) * 0.0f + right.y) * rBound;
    frustumCenterMin = ((dir.z + dir.x) * 0.0f + dir.y) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Z axis (0, 0, 1)
    centerProj = relativeCenter.x * boxAxis[2][0] + relativeCenter.y * boxAxis[2][1] + relativeCenter.z * boxAxis[2][2];
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[2][0] * boxAxis[i][0] + boxAxis[2][1] * boxAxis[i][1] + boxAxis[2][2] * boxAxis[i][2]) * (&halfSize.x)[i];
    }
    frustumRadius = fabsf((up.y + up.x) * 0.0f + up.z) * uBound +
                    fabsf((right.y + right.x) * 0.0f + right.z) * rBound;
    frustumCenterMin = ((dir.y + dir.x) * 0.0f + dir.z) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Far plane axis
    const VxVector &farNormal = farPlane.GetNormal();
    centerProj = relativeCenter.x * farNormal.x + 
                 relativeCenter.y * farNormal.y + 
                 relativeCenter.z * farNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][0] * farNormal.x + 
                          boxAxis[i][1] * farNormal.y + 
                          boxAxis[i][2] * farNormal.z) * (&halfSize.x)[i];
    }
    frustumRadius = fabsf(farNormal.z * up.z + 
                          farNormal.y * up.y + 
                          farNormal.x * up.x) * uBound +
                    fabsf(farNormal.z * right.z + 
                          farNormal.y * right.y + 
                          farNormal.x * right.x) * rBound;
    frustumCenterMin = (farNormal.z * dir.z + 
                        farNormal.y * dir.y + 
                        dir.x * farNormal.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Bottom plane axis
    const VxVector &bottomNormal = bottomPlane.GetNormal();
    centerProj = relativeCenter.x * bottomNormal.x + 
                 relativeCenter.y * bottomNormal.y + 
                 relativeCenter.z * bottomNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][0] * bottomNormal.x + 
                          boxAxis[i][1] * bottomNormal.y + 
                          boxAxis[i][2] * bottomNormal.z) * (&halfSize.x)[i];
    }
    frustumRadius = fabsf(bottomNormal.z * up.z + 
                          bottomNormal.y * up.y + 
                          bottomNormal.x * up.x) * uBound +
                    fabsf(bottomNormal.z * right.z + 
                          bottomNormal.y * right.y + 
                          right.x * bottomNormal.x) * rBound;
    frustumCenterMin = (bottomNormal.z * dir.z + 
                        bottomNormal.y * dir.y + 
                        bottomNormal.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Up plane axis
    const VxVector &upNormal = upPlane.GetNormal();
    centerProj = relativeCenter.x * upNormal.x + 
                 relativeCenter.y * upNormal.y + 
                 relativeCenter.z * upNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][0] * upNormal.x + 
                          boxAxis[i][1] * upNormal.y + 
                          boxAxis[i][2] * upNormal.z) * (&halfSize.x)[i];
    }
    frustumRadius = fabsf(upNormal.z * up.z + 
                          upNormal.y * up.y + 
                          upNormal.x * up.x) * uBound +
                    fabsf(upNormal.z * right.z + 
                          upNormal.y * right.y + 
                          upNormal.x * right.x) * rBound;
    frustumCenterMin = (upNormal.z * dir.z + 
                        upNormal.y * dir.y + 
                        upNormal.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Left plane axis
    const VxVector &leftNormal = leftPlane.GetNormal();
    centerProj = relativeCenter.x * leftNormal.x + 
                 relativeCenter.y * leftNormal.y + 
                 relativeCenter.z * leftNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][0] * leftNormal.x + 
                          boxAxis[i][1] * leftNormal.y + 
                          boxAxis[i][2] * leftNormal.z) * (&halfSize.x)[i];
    }
    frustumRadius = fabsf(leftNormal.z * up.z + 
                          leftNormal.y * up.y + 
                          leftNormal.x * up.x) * uBound +
                    fabsf(leftNormal.z * right.z + 
                          leftNormal.y * right.y + 
                          leftNormal.x * right.x) * rBound;
    frustumCenterMin = (leftNormal.z * dir.z + 
                        leftNormal.y * dir.y + 
                        leftNormal.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Right plane axis
    const VxVector &rightNormal = rightPlane.GetNormal();
    centerProj = relativeCenter.x * rightNormal.x + 
                 relativeCenter.y * rightNormal.y + 
                 relativeCenter.z * rightNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][0] * rightNormal.x + 
                          boxAxis[i][1] * rightNormal.y + 
                          boxAxis[i][2] * rightNormal.z) * (&halfSize.x)[i];
    }
    frustumRadius = fabsf(rightNormal.z * up.z + 
                          rightNormal.y * up.y + 
                          up.x * rightNormal.x) * uBound +
                    fabsf(rightNormal.z * right.z + 
                          rightNormal.y * right.y + 
                          rightNormal.x * right.x) * rBound;
    frustumCenterMin = (rightNormal.z * dir.z + 
                        rightNormal.y * dir.y + 
                        rightNormal.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    return TRUE;
}

// Intersection Frustum - OBB
// Matches binary at 0x2428aaa0
XBOOL VxIntersect::FrustumOBB(const VxFrustum &frustum, const VxBbox &box, const VxMatrix &mat) {
    // Get frustum properties via accessors
    const VxVector &origin = frustum.GetOrigin();
    const VxVector &dir = frustum.GetDir();
    const VxVector &up = frustum.GetUp();
    const VxVector &right = frustum.GetRight();
    float rBound = frustum.GetRBound();
    float uBound = frustum.GetUBound();
    float dMin = frustum.GetDMin();
    float dRatio = frustum.GetDRatio();
    
    const VxPlane &farPlane = frustum.GetFarPlane();
    const VxPlane &bottomPlane = frustum.GetBottomPlane();
    const VxPlane &upPlane = frustum.GetUpPlane();
    const VxPlane &leftPlane = frustum.GetLeftPlane();
    const VxPlane &rightPlane = frustum.GetRightPlane();

    // Compute box half-size
    VxVector halfSize = (box.Max - box.Min) * 0.5f;

    // Compute box center in world space
    VxVector localCenter = (box.Max + box.Min) * 0.5f;
    VxVector worldCenter;
    Vx3DMultiplyMatrixVector(&worldCenter, mat, &localCenter);

    // Transform center relative to frustum origin
    VxVector relativeCenter(
        worldCenter.x - origin.x, 
        worldCenter.y - origin.y, 
        worldCenter.z - origin.z
    );

    // Extract axes from matrix
    VxVector axis0(mat[0][0], mat[0][1], mat[0][2]);
    VxVector axis1(mat[1][0], mat[1][1], mat[1][2]);
    VxVector axis2(mat[2][0], mat[2][1], mat[2][2]);
    
    // Compute axis lengths
    float len0 = sqrtf(axis0.x * axis0.x + axis0.y * axis0.y + axis0.z * axis0.z);
    float len1 = sqrtf(axis1.x * axis1.x + axis1.y * axis1.y + axis1.z * axis1.z);
    float len2 = sqrtf(axis2.x * axis2.x + axis2.y * axis2.y + axis2.z * axis2.z);
    
    // Normalize axes
    float invLen0 = 1.0f / len0;
    float invLen1 = 1.0f / len1;
    float invLen2 = 1.0f / len2;
    
    axis0.x *= invLen0; axis0.y *= invLen0; axis0.z *= invLen0;
    axis1.x *= invLen1; axis1.y *= invLen1; axis1.z *= invLen1;
    axis2.x *= invLen2; axis2.y *= invLen2; axis2.z *= invLen2;
    
    // Scale half sizes by axis lengths
    float scaledHalfSize0 = halfSize.x * len0;
    float scaledHalfSize1 = halfSize.y * len1;
    float scaledHalfSize2 = halfSize.z * len2;

    // Store axes in array for iteration
    float boxAxis[3][3];
    boxAxis[0][0] = axis0.x; boxAxis[0][1] = axis0.y; boxAxis[0][2] = axis0.z;
    boxAxis[1][0] = axis1.x; boxAxis[1][1] = axis1.y; boxAxis[1][2] = axis1.z;
    boxAxis[2][0] = axis2.x; boxAxis[2][1] = axis2.y; boxAxis[2][2] = axis2.z;
    
    float scaledHalfSizes[3] = { scaledHalfSize0, scaledHalfSize1, scaledHalfSize2 };

    float centerProj, boxRadius, frustumRadius, frustumCenterMin;

    // Test against box axis 0
    centerProj = relativeCenter.x * axis0.x + relativeCenter.y * axis0.y + relativeCenter.z * axis0.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(axis0.x * boxAxis[i][0] + axis0.y * boxAxis[i][1] + axis0.z * boxAxis[i][2]) * scaledHalfSizes[i];
    }
    frustumRadius = fabsf(axis0.z * up.z + axis0.y * up.y + axis0.x * up.x) * uBound +
                    fabsf(axis0.z * right.z + axis0.y * right.y + axis0.x * right.x) * rBound;
    frustumCenterMin = (axis0.z * dir.z + axis0.y * dir.y + axis0.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test against box axis 1
    centerProj = relativeCenter.x * axis1.x + relativeCenter.y * axis1.y + relativeCenter.z * axis1.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(axis1.x * boxAxis[i][0] + axis1.y * boxAxis[i][1] + axis1.z * boxAxis[i][2]) * scaledHalfSizes[i];
    }
    frustumRadius = fabsf(axis1.z * up.z + axis1.y * up.y + axis1.x * up.x) * uBound +
                    fabsf(axis1.z * right.z + axis1.y * right.y + axis1.x * right.x) * rBound;
    frustumCenterMin = (axis1.z * dir.z + axis1.y * dir.y + axis1.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test against box axis 2
    centerProj = relativeCenter.x * axis2.x + relativeCenter.y * axis2.y + relativeCenter.z * axis2.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(axis2.x * boxAxis[i][0] + axis2.y * boxAxis[i][1] + axis2.z * boxAxis[i][2]) * scaledHalfSizes[i];
    }
    frustumRadius = fabsf(axis2.z * up.z + axis2.y * up.y + axis2.x * up.x) * uBound +
                    fabsf(axis2.z * right.z + axis2.y * right.y + axis2.x * right.x) * rBound;
    frustumCenterMin = (axis2.z * dir.z + axis2.y * dir.y + axis2.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Far plane axis
    const VxVector &farNormal = farPlane.GetNormal();
    centerProj = relativeCenter.x * farNormal.x + 
                 relativeCenter.y * farNormal.y + 
                 relativeCenter.z * farNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][2] * farNormal.z + 
                          boxAxis[i][0] * farNormal.x + 
                          boxAxis[i][1] * farNormal.y) * scaledHalfSizes[i];
    }
    frustumRadius = fabsf(farNormal.z * up.z + 
                          farNormal.y * up.y + 
                          farNormal.x * up.x) * uBound +
                    fabsf(farNormal.z * right.z + 
                          farNormal.y * right.y + 
                          farNormal.x * right.x) * rBound;
    frustumCenterMin = (farNormal.z * dir.z + 
                        farNormal.y * dir.y + 
                        dir.x * farNormal.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Bottom plane axis
    const VxVector &bottomNormal = bottomPlane.GetNormal();
    centerProj = relativeCenter.x * bottomNormal.x + 
                 relativeCenter.y * bottomNormal.y + 
                 relativeCenter.z * bottomNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][2] * bottomNormal.z + 
                          boxAxis[i][0] * bottomNormal.x + 
                          boxAxis[i][1] * bottomNormal.y) * scaledHalfSizes[i];
    }
    frustumRadius = fabsf(bottomNormal.z * up.z + 
                          bottomNormal.y * up.y + 
                          bottomNormal.x * up.x) * uBound +
                    fabsf(bottomNormal.z * right.z + 
                          bottomNormal.y * right.y + 
                          right.x * bottomNormal.x) * rBound;
    frustumCenterMin = (bottomNormal.z * dir.z + 
                        bottomNormal.y * dir.y + 
                        bottomNormal.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Up plane axis
    const VxVector &upNormal = upPlane.GetNormal();
    centerProj = relativeCenter.x * upNormal.x + 
                 relativeCenter.y * upNormal.y + 
                 relativeCenter.z * upNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][0] * upNormal.x + 
                          boxAxis[i][2] * upNormal.z + 
                          boxAxis[i][1] * upNormal.y) * scaledHalfSizes[i];
    }
    frustumRadius = fabsf(upNormal.z * up.z + 
                          upNormal.y * up.y + 
                          upNormal.x * up.x) * uBound +
                    fabsf(upNormal.z * right.z + 
                          upNormal.y * right.y + 
                          upNormal.x * right.x) * rBound;
    frustumCenterMin = (upNormal.z * dir.z + 
                        upNormal.y * dir.y + 
                        upNormal.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Left plane axis
    const VxVector &leftNormal = leftPlane.GetNormal();
    centerProj = relativeCenter.x * leftNormal.x + 
                 relativeCenter.y * leftNormal.y + 
                 relativeCenter.z * leftNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][2] * leftNormal.z + 
                          boxAxis[i][0] * leftNormal.x + 
                          boxAxis[i][1] * leftNormal.y) * scaledHalfSizes[i];
    }
    frustumRadius = fabsf(leftNormal.z * up.z + 
                          leftNormal.y * up.y + 
                          leftNormal.x * up.x) * uBound +
                    fabsf(leftNormal.z * right.z + 
                          leftNormal.y * right.y + 
                          leftNormal.x * right.x) * rBound;
    frustumCenterMin = (leftNormal.z * dir.z + 
                        leftNormal.y * dir.y + 
                        leftNormal.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    // Test Right plane axis
    const VxVector &rightNormal = rightPlane.GetNormal();
    centerProj = relativeCenter.x * rightNormal.x + 
                 relativeCenter.y * rightNormal.y + 
                 relativeCenter.z * rightNormal.z;
    boxRadius = 0.0f;
    for (int i = 0; i < 3; i++) {
        boxRadius += fabsf(boxAxis[i][0] * rightNormal.x + 
                          boxAxis[i][2] * rightNormal.z + 
                          boxAxis[i][1] * rightNormal.y) * scaledHalfSizes[i];
    }
    frustumRadius = fabsf(rightNormal.z * up.z + 
                          rightNormal.y * up.y + 
                          up.x * rightNormal.x) * uBound +
                    fabsf(rightNormal.z * right.z + 
                          rightNormal.y * right.y + 
                          rightNormal.x * right.x) * rBound;
    frustumCenterMin = (rightNormal.z * dir.z + 
                        rightNormal.y * dir.y + 
                        rightNormal.x * dir.x) * dMin;
    if (!TestFrustumAxis(centerProj, boxRadius, frustumRadius, frustumCenterMin, dRatio))
        return FALSE;

    return TRUE;
}

// Intersection Frustum - Box (general, for visibility testing)
// Matches binary at 0x2428b5b0
XBOOL VxIntersect::FrustumBox(const VxFrustum &frustum, const VxBbox &box, const VxMatrix &mat) {
    // Get frustum planes via accessors
    const VxPlane &nearPlane = frustum.GetNearPlane();
    const VxPlane &farPlane = frustum.GetFarPlane();
    const VxPlane &leftPlane = frustum.GetLeftPlane();
    const VxPlane &rightPlane = frustum.GetRightPlane();
    const VxPlane &upPlane = frustum.GetUpPlane();
    const VxPlane &bottomPlane = frustum.GetBottomPlane();

    // Extract axes from matrix
    VxVector axis0(mat[0][0], mat[0][1], mat[0][2]);
    VxVector axis1(mat[1][0], mat[1][1], mat[1][2]);
    VxVector axis2(mat[2][0], mat[2][1], mat[2][2]);

    // Compute box half-size and scale axes
    VxVector halfSize = (box.Max - box.Min) * 0.5f;
    axis0.x *= halfSize.x; axis0.y *= halfSize.x; axis0.z *= halfSize.x;
    axis1.x *= halfSize.y; axis1.y *= halfSize.y; axis1.z *= halfSize.y;
    axis2.x *= halfSize.z; axis2.y *= halfSize.z; axis2.z *= halfSize.z;

    // Compute box center in world space
    VxVector localCenter = (box.Max + box.Min);
    localCenter.x *= 0.5f;
    localCenter.y *= 0.5f;
    localCenter.z *= 0.5f;
    VxVector worldCenter;
    Vx3DMultiplyMatrixVector(&worldCenter, mat, &localCenter);

    float r, d;
    const VxVector &nearNormal = nearPlane.GetNormal();
    const VxVector &farNormal = farPlane.GetNormal();
    const VxVector &leftNormal = leftPlane.GetNormal();
    const VxVector &rightNormal = rightPlane.GetNormal();

    // Near plane test
    // In the binary (0x2428b5b0), the logic is:
    // - If d < r && d >= -r: return TRUE (intersection)
    // - If d < 0: continue to next plane (box center on inside side)
    // - If d >= 0: goto end_test (box center on outside side)
    r = fabsf(DotProduct(nearNormal, axis2));
    r += fabsf(DotProduct(nearNormal, axis1));
    r += fabsf(DotProduct(nearNormal, axis0));
    d = DotProduct(nearNormal, worldCenter) + nearPlane.m_D;
    
    if (d < r && d >= -r) return TRUE;
    if (d >= 0.0f) goto end_test;

    // Far plane test
    r = fabsf(DotProduct(farNormal, axis2));
    r += fabsf(DotProduct(farNormal, axis1));
    r += fabsf(DotProduct(farNormal, axis0));
    d = DotProduct(farNormal, worldCenter) + farPlane.m_D;
    
    if (d < r && d >= -r) return TRUE;
    if (d >= 0.0f) goto end_test;

    // Left plane test
    r = fabsf(DotProduct(leftNormal, axis2));
    r += fabsf(DotProduct(leftNormal, axis1));
    r += fabsf(DotProduct(leftNormal, axis0));
    d = DotProduct(leftNormal, worldCenter) + leftPlane.m_D;
    
    if (d < r && d >= -r) return TRUE;
    if (d >= 0.0f) goto end_test;

    // Right plane test
    r = fabsf(DotProduct(rightNormal, axis2));
    r += fabsf(DotProduct(rightNormal, axis1));
    r += fabsf(DotProduct(rightNormal, axis0));
    d = DotProduct(rightNormal, worldCenter) + rightPlane.m_D;
    
    if (d < r && d >= -r) return TRUE;
    if (d >= 0.0f) goto end_test;

    // Create boxaxis array for XClassify: [axis0, axis1, axis2, worldCenter]
    {
        VxVector boxaxis[4];
        boxaxis[0] = axis0;
        boxaxis[1] = axis1;
        boxaxis[2] = axis2;
        boxaxis[3] = worldCenter;

        // Up plane test (using XClassify)
        // XClassify returns: 0 if intersecting, d otherwise
        // If d >= 0 (outside), goto end_test
        d = upPlane.XClassify(boxaxis);
        if (d >= 0.0f) goto end_test;

        // Bottom plane test (using XClassify)
        d = bottomPlane.XClassify(boxaxis);
        if (d >= 0.0f) goto end_test;
    }

    return TRUE;

end_test:
    return (d == 0.0f);
}
