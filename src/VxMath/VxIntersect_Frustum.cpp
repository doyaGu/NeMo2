#include "VxIntersect.h"

#include "VxVector.h"
#include "VxRay.h"
#include "VxPlane.h"
#include "VxFrustum.h"
#include "VxMatrix.h"

//--------- Frustum

// Intersection Frustum - Face
XBOOL VxIntersect::FrustumFace(const VxFrustum &frustum, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2) {
    // Plane convention: Classify(point) > 0 => outside.
    // Uses a face-classification test for the bottom/top planes.
    // Using ClassifyFace consistently matches the reject rule:
    // reject only if the whole triangle is strictly outside one plane.
    if (frustum.GetNearPlane().ClassifyFace(pt0, pt1, pt2) > 0.0f)
        return FALSE;
    if (frustum.GetFarPlane().ClassifyFace(pt0, pt1, pt2) > 0.0f)
        return FALSE;
    if (frustum.GetLeftPlane().ClassifyFace(pt0, pt1, pt2) > 0.0f)
        return FALSE;
    if (frustum.GetRightPlane().ClassifyFace(pt0, pt1, pt2) > 0.0f)
        return FALSE;
    if (frustum.GetBottomPlane().ClassifyFace(pt0, pt1, pt2) > 0.0f)
        return FALSE;
    if (frustum.GetUpPlane().ClassifyFace(pt0, pt1, pt2) > 0.0f)
        return FALSE;

    return TRUE;
}

// Intersection Frustum - AABB
XBOOL VxIntersect::FrustumAABB(const VxFrustum &frustum, const VxBbox &box) {
    // It is equivalent to the OBB test with an identity transform.
    VxMatrix id;
    id.SetIdentity();
    return FrustumOBB(frustum, box, id);
}

// Intersection Frustum - OBB
XBOOL VxIntersect::FrustumOBB(const VxFrustum &frustum, const VxBbox &box, const VxMatrix &mat) {
    // It performs SAT-style interval overlap tests along:
    //  - the 3 box axes (matrix rows, normalized)
    //  - the frustum's 5 plane normals (far, bottom, up, left, right)

    // Box half-size in local space.
    VxVector halfSize = (box.Max - box.Min) * 0.5f;

    // Box center transformed to world, then expressed relative to frustum origin.
    VxVector localCenter = (box.Max + box.Min) * 0.5f;
    VxVector worldCenter;
    Vx3DMultiplyMatrixVector(&worldCenter, mat, &localCenter);
    VxVector relCenter = worldCenter - frustum.GetOrigin();

    // Box axes are the matrix rows (rotation+scale). Normalize them and fold scale into halfSize.
    VxVector axis0(mat[0].x, mat[0].y, mat[0].z);
    VxVector axis1(mat[1].x, mat[1].y, mat[1].z);
    VxVector axis2(mat[2].x, mat[2].y, mat[2].z);

    float len0 = sqrtf(SquareMagnitude(axis0));
    float len1 = sqrtf(SquareMagnitude(axis1));
    float len2 = sqrtf(SquareMagnitude(axis2));

    if (len0 != 0.0f) axis0 *= (1.0f / len0);
    if (len1 != 0.0f) axis1 *= (1.0f / len1);
    if (len2 != 0.0f) axis2 *= (1.0f / len2);

    halfSize.x *= len0;
    halfSize.y *= len1;
    halfSize.z *= len2;

    const VxVector &frDir = frustum.GetDir();
    const VxVector &frUp = frustum.GetUp();
    const VxVector &frRight = frustum.GetRight();
    const float rBound = frustum.GetRBound();
    const float uBound = frustum.GetUBound();
    const float dMin = frustum.GetDMin();
    const float dRatio = frustum.GetDRatio();

    auto axisTest = [&](const VxVector &axis) -> XBOOL {
        const float c = DotProduct(relCenter, axis);
        const float r = XAbs(DotProduct(axis, axis0)) * halfSize.x +
            XAbs(DotProduct(axis, axis1)) * halfSize.y +
            XAbs(DotProduct(axis, axis2)) * halfSize.z;

        const float frBound = XAbs(DotProduct(axis, frUp)) * uBound + XAbs(DotProduct(axis, frRight)) * rBound;

        const float d = DotProduct(axis, frDir) * dMin;

        float frMin = d - frBound;
        if (frMin < 0.0f)
            frMin *= dRatio;

        float frMax = d + frBound;
        if (frMax > 0.0f)
            frMax *= dRatio;

        if (c + r < frMin)
            return FALSE;
        if (c - r > frMax)
            return FALSE;
        return TRUE;
    };

    // Test along box axes.
    if (!axisTest(axis0)) return FALSE;
    if (!axisTest(axis1)) return FALSE;
    if (!axisTest(axis2)) return FALSE;

    // Test along frustum plane normals.
    if (!axisTest(frustum.GetFarPlane().GetNormal())) return FALSE;
    if (!axisTest(frustum.GetBottomPlane().GetNormal())) return FALSE;
    if (!axisTest(frustum.GetUpPlane().GetNormal())) return FALSE;
    if (!axisTest(frustum.GetLeftPlane().GetNormal())) return FALSE;
    if (!axisTest(frustum.GetRightPlane().GetNormal())) return FALSE;

    return TRUE;
}

// Intersection Frustum - Box (deprecated, alias for FrustumOBB)
XBOOL VxIntersect::FrustumBox(const VxFrustum &frustum, const VxBbox &box, const VxMatrix &mat) {
    // Build a transformed oriented box representation (3 scaled axes + transformed center)
    // and reject if it lies strictly outside any frustum plane.

    VxVector axis[4];
    axis[0] = mat[0] * ((box.Max.x - box.Min.x) * 0.5f);
    axis[1] = mat[1] * ((box.Max.y - box.Min.y) * 0.5f);
    axis[2] = mat[2] * ((box.Max.z - box.Min.z) * 0.5f);

    VxVector center = box.GetCenter();
    Vx3DMultiplyMatrixVector(axis + 3, mat, &center);

    if (frustum.GetNearPlane().XClassify(axis) > 0.0f) return FALSE;
    if (frustum.GetFarPlane().XClassify(axis) > 0.0f) return FALSE;
    if (frustum.GetLeftPlane().XClassify(axis) > 0.0f) return FALSE;
    if (frustum.GetRightPlane().XClassify(axis) > 0.0f) return FALSE;
    if (frustum.GetBottomPlane().XClassify(axis) > 0.0f) return FALSE;
    if (frustum.GetUpPlane().XClassify(axis) > 0.0f) return FALSE;

    return TRUE;
}
