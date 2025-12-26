#include "VxIntersect.h"

#include <float.h>

#include "VxVector.h"
#include "VxMatrix.h"
#include "VxRay.h"
#include "VxOBB.h"
#include "VxPlane.h"

//----------- Boxes

// Intersection Ray - Box (simple boolean version)
XBOOL VxIntersect::RayBox(const VxRay &ray, const VxBbox &box) {
    // Get box center and half-extents
    VxVector boxCenter = (box.Max + box.Min) * 0.5f;
    VxVector boxHalfSize = (box.Max - box.Min) * 0.5f;

    // Vector from box center to ray origin
    VxVector p = ray.m_Origin - boxCenter;

    // Test if ray origin is outside box on any axis
    if (XAbs(p.x) > boxHalfSize.x && p.x * ray.m_Direction.x >= 0.0f) return FALSE;
    if (XAbs(p.y) > boxHalfSize.y && p.y * ray.m_Direction.y >= 0.0f) return FALSE;
    if (XAbs(p.z) > boxHalfSize.z && p.z * ray.m_Direction.z >= 0.0f) return FALSE;

    // Test separating axes from cross products
    VxVector cross = CrossProduct(p, ray.m_Direction);

    if (XAbs(cross.x) > boxHalfSize.y * XAbs(ray.m_Direction.z) + boxHalfSize.z * XAbs(ray.m_Direction.y)) return FALSE;
    if (XAbs(cross.y) > boxHalfSize.x * XAbs(ray.m_Direction.z) + boxHalfSize.z * XAbs(ray.m_Direction.x)) return FALSE;
    if (XAbs(cross.z) > boxHalfSize.x * XAbs(ray.m_Direction.y) + boxHalfSize.y * XAbs(ray.m_Direction.x)) return FALSE;

    return TRUE;
}

// Intersection Ray - Box (detailed version with intersection points and normals)
int VxIntersect::RayBox(const VxRay &ray, const VxBbox &box, VxVector &inpoint, VxVector *outpoint, VxVector *innormal, VxVector *outnormal) {
    float tNear = -FLT_MAX;
    float tFar = FLT_MAX;
    int nearAxis = -1;
    float nearSign = 0.0f;
    int farAxis = -1;
    float farSign = 0.0f;

    // Test each axis
    for (int i = 0; i < 3; i++) {
        float rayDir = ray.m_Direction[i];
        float rayOrigin = ray.m_Origin[i];
        float boxMin = box.Min[i];
        float boxMax = box.Max[i];

        if (XAbs(rayDir) < EPSILON) {
            // Ray is parallel to the slab
            if (rayOrigin < boxMin || rayOrigin > boxMax)
                return 0;
        } else {
            float invDir = 1.0f / rayDir;
            float t1 = (boxMin - rayOrigin) * invDir;
            float t2 = (boxMax - rayOrigin) * invDir;

            float sign1 = -1.0f;
            float sign2 = 1.0f;

            if (t1 > t2) {
                // Swap t1 and t2
                float temp = t1;
                t1 = t2;
                t2 = temp;
                sign1 = 1.0f;
                sign2 = -1.0f;
            }

            if (t1 > tNear) {
                tNear = t1;
                nearAxis = i;
                nearSign = sign1;
            }

            if (t2 < tFar) {
                tFar = t2;
                farAxis = i;
                farSign = sign2;
            }

            if (tNear > tFar || tFar < EPSILON)
                return 0;
        }
    }

    // Calculate intersection points
    inpoint = ray.m_Origin + ray.m_Direction * tNear;
    if (outpoint)
        *outpoint = ray.m_Origin + ray.m_Direction * tFar;

    // Calculate normals
    if (innormal) {
        *innormal = VxVector::axis0();
        if (nearAxis >= 0)
            (*innormal)[nearAxis] = nearSign;
    }

    if (outnormal) {
        *outnormal = VxVector::axis0();
        if (farAxis >= 0)
            (*outnormal)[farAxis] = farSign;
    }

    if (tNear < EPSILON)
        return -1; // Ray starts inside box
    else
        return 1; // Normal intersection
}

// Intersection Segment - Box (simple boolean version)
XBOOL VxIntersect::SegmentBox(const VxRay &segment, const VxBbox &box) {
    // Get box center and half-extents
    VxVector boxCenter = (box.Max + box.Min) * 0.5f;
    VxVector boxHalfSize = (box.Max - box.Min) * 0.5f;

    // Segment center and half-vector
    VxVector segmentHalf = segment.m_Direction * 0.5f;
    VxVector segmentCenter = segment.m_Origin + segmentHalf;

    // Vector from box center to segment center
    VxVector d = segmentCenter - boxCenter;

    // Test separating axes
    if (XAbs(d.x) > boxHalfSize.x + XAbs(segmentHalf.x)) return FALSE;
    if (XAbs(d.y) > boxHalfSize.y + XAbs(segmentHalf.y)) return FALSE;
    if (XAbs(d.z) > boxHalfSize.z + XAbs(segmentHalf.z)) return FALSE;

    // Test cross product axes
    VxVector cross = CrossProduct(d, segment.m_Direction);

    if (XAbs(cross.x) > boxHalfSize.y * XAbs(segment.m_Direction.z) + boxHalfSize.z * XAbs(segment.m_Direction.y))
        return FALSE;
    if (XAbs(cross.y) > boxHalfSize.x * XAbs(segment.m_Direction.z) + boxHalfSize.z * XAbs(segment.m_Direction.x))
        return FALSE;
    if (XAbs(cross.z) > boxHalfSize.x * XAbs(segment.m_Direction.y) + boxHalfSize.y * XAbs(segment.m_Direction.x))
        return FALSE;

    return TRUE;
}

// Intersection Segment - Box (detailed version)
int VxIntersect::SegmentBox(const VxRay &segment, const VxBbox &box, VxVector &inpoint, VxVector *outpoint, VxVector *innormal, VxVector *outnormal) {
    float tNear = 0.0f; // Segment starts at t=0
    float tFar = 1.0f;  // Segment ends at t=1
    int nearAxis = -1;
    float nearSign = 0.0f;
    int farAxis = -1;
    float farSign = 0.0f;

    // Test each axis
    for (int i = 0; i < 3; i++) {
        float rayDir = segment.m_Direction[i];
        float rayOrigin = segment.m_Origin[i];
        float boxMin = box.Min[i];
        float boxMax = box.Max[i];

        if (XAbs(rayDir) < EPSILON) {
            // Segment is parallel to the slab
            if (rayOrigin < boxMin || rayOrigin > boxMax)
                return 0;
        } else {
            float invDir = 1.0f / rayDir;
            float t1 = (boxMin - rayOrigin) * invDir;
            float t2 = (boxMax - rayOrigin) * invDir;

            float sign1 = -1.0f;
            float sign2 = 1.0f;

            if (t1 > t2) {
                // Swap t1 and t2
                float temp = t1;
                t1 = t2;
                t2 = temp;
                sign1 = 1.0f;
                sign2 = -1.0f;
            }

            if (t1 > tNear) {
                tNear = t1;
                nearAxis = i;
                nearSign = sign1;
            }

            if (t2 < tFar) {
                tFar = t2;
                farAxis = i;
                farSign = sign2;
            }

            if (tNear > tFar)
                return 0;
        }
    }

    // Check if segment intersects box (must have overlap with [0,1])
    if (tNear > 1.0f + EPSILON || tFar < -EPSILON)
        return 0;

    // Clamp to segment range
    if (tNear < EPSILON) {
        tNear = 0.0f;
        nearAxis = -1; // Mark as starting inside
    }
    if (tFar > 1.0f - EPSILON) {
        tFar = 1.0f;
        farAxis = -1; // Mark as ending inside
    }

    // Calculate intersection points
    inpoint = segment.m_Origin + segment.m_Direction * tNear;
    if (outpoint)
        *outpoint = segment.m_Origin + segment.m_Direction * tFar;

    // Calculate normals
    if (innormal) {
        *innormal = VxVector::axis0();
        if (nearAxis >= 0)
            (*innormal)[nearAxis] = nearSign;
    }

    if (outnormal) {
        *outnormal = VxVector::axis0();
        if (farAxis >= 0)
            (*outnormal)[farAxis] = farSign;
    }

    if (tNear < EPSILON)
        return -1; // Segment starts inside box
    else
        return 1; // Normal intersection
}

// Intersection Line - Box (simple boolean version)
XBOOL VxIntersect::LineBox(const VxRay &line, const VxBbox &box) {
    // Get box center and half-extents
    VxVector boxCenter = (box.Max + box.Min) * 0.5f;
    VxVector boxHalfSize = (box.Max - box.Min) * 0.5f;

    // Vector from box center to line origin
    VxVector d = line.m_Origin - boxCenter;

    // Test separating axes from cross products
    VxVector cross = CrossProduct(d, line.m_Direction);

    if (XAbs(cross.x) > boxHalfSize.y * XAbs(line.m_Direction.z) + boxHalfSize.z * XAbs(line.m_Direction.y))
        return FALSE;
    if (XAbs(cross.y) > boxHalfSize.x * XAbs(line.m_Direction.z) + boxHalfSize.z * XAbs(line.m_Direction.x))
        return FALSE;
    if (XAbs(cross.z) > boxHalfSize.x * XAbs(line.m_Direction.y) + boxHalfSize.y * XAbs(line.m_Direction.x))
        return FALSE;

    return TRUE;
}

// Intersection Line - Box (detailed version)
int VxIntersect::LineBox(const VxRay &line, const VxBbox &box, VxVector &inpoint, VxVector *outpoint, VxVector *innormal, VxVector *outnormal) {
    float tNear = -FLT_MAX;
    float tFar = FLT_MAX;
    int nearAxis = -1;
    float nearSign = 0.0f;
    int farAxis = -1;
    float farSign = 0.0f;

    // Test each axis
    for (int i = 0; i < 3; i++) {
        float rayDir = line.m_Direction[i];
        float rayOrigin = line.m_Origin[i];
        float boxMin = box.Min[i];
        float boxMax = box.Max[i];

        if (XAbs(rayDir) < EPSILON) {
            // Line is parallel to the slab
            if (rayOrigin < boxMin || rayOrigin > boxMax)
                return 0;
        } else {
            float invDir = 1.0f / rayDir;
            float t1 = (boxMin - rayOrigin) * invDir;
            float t2 = (boxMax - rayOrigin) * invDir;

            float sign1 = -1.0f;
            float sign2 = 1.0f;

            if (XAbs(t2) < XAbs(t1)) {
                // Swap t1 and t2
                float temp = t1;
                t1 = t2;
                t2 = temp;
                sign1 = 1.0f;
                sign2 = -1.0f;
            }

            if (XAbs(tNear) < XAbs(t1)) {
                tNear = t1;
                nearAxis = i;
                nearSign = sign1;
            }

            if (XAbs(t2) < XAbs(tFar)) {
                tFar = t2;
                farAxis = i;
                farSign = sign2;
            }

            if (XAbs(tFar) < XAbs(tNear))
                return 0;
        }
    }

    // Calculate intersection points
    inpoint = line.m_Origin + line.m_Direction * tNear;
    if (outpoint)
        *outpoint = line.m_Origin + line.m_Direction * tFar;

    // Calculate normals
    if (innormal) {
        *innormal = VxVector::axis0();
        if (nearAxis >= 0)
            (*innormal)[nearAxis] = nearSign;
    }

    if (outnormal) {
        *outnormal = VxVector::axis0();
        if (farAxis >= 0)
            (*outnormal)[farAxis] = farSign;
    }

    return 1;
}

// Intersection Box - Box
XBOOL VxIntersect::AABBAABB(const VxBbox &box1, const VxBbox &box2) {
    // Check for separation along all axes
    if (box1.Min.x > box2.Max.x || box2.Min.x > box1.Max.x) return FALSE;
    if (box1.Min.y > box2.Max.y || box2.Min.y > box1.Max.y) return FALSE;
    if (box1.Min.z > box2.Max.z || box2.Min.z > box1.Max.z) return FALSE;

    // No separation found, boxes intersect
    return TRUE;
}

// AABB - OBB intersection
XBOOL VxIntersect::AABBOBB(const VxBbox &box1, const VxOBB &box2) {
    // Get AABB center and half-extents
    VxVector aabbCenter = (box1.Max + box1.Min) * 0.5f;
    VxVector aabbHalfSize = (box1.Max - box1.Min) * 0.5f;

    // Vector from AABB center to OBB center
    VxVector T = box2.m_Center - aabbCenter;

    // Test AABB axes
    for (int i = 0; i < 3; i++) {
        float ra = aabbHalfSize[i];
        float rb = XAbs(DotProduct(box2.m_Axis[0], VxVector::axis0())) * box2.m_Extents[0] +
                   XAbs(DotProduct(box2.m_Axis[1], VxVector::axis0())) * box2.m_Extents[1] +
                   XAbs(DotProduct(box2.m_Axis[2], VxVector::axis0())) * box2.m_Extents[2];

        if (XAbs(T[i]) > ra + rb) return FALSE;
    }

    // Test OBB axes
    for (int i = 0; i < 3; i++) {
        float ra = aabbHalfSize.x * XAbs(box2.m_Axis[i].x) +
            aabbHalfSize.y * XAbs(box2.m_Axis[i].y) +
            aabbHalfSize.z * XAbs(box2.m_Axis[i].z);
        float rb = box2.m_Extents[i];

        float t = DotProduct(T, box2.m_Axis[i]);

        if (XAbs(t) > ra + rb) return FALSE;
    }

    // Test cross product axes (9 axes)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            VxVector axis;
            axis[i] = (i == 0) ? 1.0f : 0.0f;
            if (i == 1) axis[i] = 1.0f;
            if (i == 2) axis[i] = 1.0f;

            VxVector crossAxis = CrossProduct(axis, box2.m_Axis[j]);

            float ra = 0.0f;
            for (int k = 0; k < 3; k++) {
                if (k != i) ra += aabbHalfSize[k] * XAbs(crossAxis[k]);
            }

            float rb = 0.0f;
            for (int k = 0; k < 3; k++) {
                if (k != j) rb += box2.m_Extents[k] * XAbs(DotProduct(crossAxis, box2.m_Axis[k]));
            }

            if (XAbs(DotProduct(T, crossAxis)) > ra + rb) return FALSE;
        }
    }

    return TRUE;
}

// OBB - OBB intersection using SAT (Separating Axis Theorem)
XBOOL VxIntersect::OBBOBB(const VxOBB &box1, const VxOBB &box2) {
    // Translation vector between box centers
    VxVector T = box2.m_Center - box1.m_Center;

    // Matrix for rotation from box1 to box2
    float R[3][3];
    float absR[3][3];

    // Compute R and absR (absolute values of R)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j] = DotProduct(box1.m_Axis[i], box2.m_Axis[j]);
            absR[i][j] = XAbs(R[i][j]);
        }
    }

    // Bring translation vector into box1's coordinate frame
    VxVector Tbox1;
    Tbox1.x = DotProduct(T, box1.m_Axis[0]);
    Tbox1.y = DotProduct(T, box1.m_Axis[1]);
    Tbox1.z = DotProduct(T, box1.m_Axis[2]);

    // Test box1's axes (A0, A1, A2)
    for (int i = 0; i < 3; i++) {
        float ra = box1.m_Extents[i];
        float rb = box2.m_Extents[0] * absR[i][0] +
            box2.m_Extents[1] * absR[i][1] +
            box2.m_Extents[2] * absR[i][2];

        if (XAbs(Tbox1[i]) > ra + rb) return FALSE;
    }

    // Test box2's axes (B0, B1, B2)
    for (int i = 0; i < 3; i++) {
        float ra = box1.m_Extents[0] * absR[0][i] +
            box1.m_Extents[1] * absR[1][i] +
            box1.m_Extents[2] * absR[2][i];
        float rb = box2.m_Extents[i];

        float t = DotProduct(T, box2.m_Axis[i]);

        if (XAbs(t) > ra + rb) return FALSE;
    }

    // Test axis A0 x B0
    float ra = box1.m_Extents[1] * absR[2][0] + box1.m_Extents[2] * absR[1][0];
    float rb = box2.m_Extents[1] * absR[0][2] + box2.m_Extents[2] * absR[0][1];
    float t = Tbox1.z * R[1][0] - Tbox1.y * R[2][0];
    if (XAbs(t) > ra + rb) return FALSE;

    // Test axis A0 x B1
    ra = box1.m_Extents[1] * absR[2][1] + box1.m_Extents[2] * absR[1][1];
    rb = box2.m_Extents[0] * absR[0][2] + box2.m_Extents[2] * absR[0][0];
    t = Tbox1.z * R[1][1] - Tbox1.y * R[2][1];
    if (XAbs(t) > ra + rb) return FALSE;

    // Test axis A0 x B2
    ra = box1.m_Extents[1] * absR[2][2] + box1.m_Extents[2] * absR[1][2];
    rb = box2.m_Extents[0] * absR[0][1] + box2.m_Extents[1] * absR[0][0];
    t = Tbox1.z * R[1][2] - Tbox1.y * R[2][2];
    if (XAbs(t) > ra + rb) return FALSE;

    // Test axis A1 x B0
    ra = box1.m_Extents[0] * absR[2][0] + box1.m_Extents[2] * absR[0][0];
    rb = box2.m_Extents[1] * absR[1][2] + box2.m_Extents[2] * absR[1][1];
    t = Tbox1.x * R[2][0] - Tbox1.z * R[0][0];
    if (XAbs(t) > ra + rb) return FALSE;

    // Test axis A1 x B1
    ra = box1.m_Extents[0] * absR[2][1] + box1.m_Extents[2] * absR[0][1];
    rb = box2.m_Extents[0] * absR[1][2] + box2.m_Extents[2] * absR[1][0];
    t = Tbox1.x * R[2][1] - Tbox1.z * R[0][1];
    if (XAbs(t) > ra + rb) return FALSE;

    // Test axis A1 x B2
    ra = box1.m_Extents[0] * absR[2][2] + box1.m_Extents[2] * absR[0][2];
    rb = box2.m_Extents[0] * absR[1][1] + box2.m_Extents[1] * absR[1][0];
    t = Tbox1.x * R[2][2] - Tbox1.z * R[0][2];
    if (XAbs(t) > ra + rb) return FALSE;

    // Test axis A2 x B0
    ra = box1.m_Extents[0] * absR[1][0] + box1.m_Extents[1] * absR[0][0];
    rb = box2.m_Extents[1] * absR[2][2] + box2.m_Extents[2] * absR[2][1];
    t = Tbox1.y * R[0][0] - Tbox1.x * R[1][0];
    if (XAbs(t) > ra + rb) return FALSE;

    // Test axis A2 x B1
    ra = box1.m_Extents[0] * absR[1][1] + box1.m_Extents[1] * absR[0][1];
    rb = box2.m_Extents[0] * absR[2][2] + box2.m_Extents[2] * absR[2][0];
    t = Tbox1.y * R[0][1] - Tbox1.x * R[1][1];
    if (XAbs(t) > ra + rb) return FALSE;

    // Test axis A2 x B2
    ra = box1.m_Extents[0] * absR[1][2] + box1.m_Extents[1] * absR[0][2];
    rb = box2.m_Extents[0] * absR[2][1] + box2.m_Extents[1] * absR[2][0];
    t = Tbox1.y * R[0][2] - Tbox1.x * R[1][2];
    if (XAbs(t) > ra + rb) return FALSE;

    // No separating axis found, boxes must intersect
    return TRUE;
}

// AABB - Face (triangle) intersection
XBOOL VxIntersect::AABBFace(const VxBbox &box, const VxVector &A0, const VxVector &A1, const VxVector &A2, const VxVector &N) {
    // 1. Check if any vertex of the triangle is inside the box
    if (box.VectorIn(A0) || box.VectorIn(A1) || box.VectorIn(A2))
        return TRUE;

    // 2. Check if the box intersects the plane of the triangle
    VxPlane plane(N, A0);

    // Test all 8 corners of the box against the plane
    float minDist = FLT_MAX;
    float maxDist = -FLT_MAX;

    VxVector corners[8] = {
        VxVector(box.Min.x, box.Min.y, box.Min.z),
        VxVector(box.Max.x, box.Min.y, box.Min.z),
        VxVector(box.Min.x, box.Max.y, box.Min.z),
        VxVector(box.Max.x, box.Max.y, box.Min.z),
        VxVector(box.Min.x, box.Min.y, box.Max.z),
        VxVector(box.Max.x, box.Min.y, box.Max.z),
        VxVector(box.Min.x, box.Max.y, box.Max.z),
        VxVector(box.Max.x, box.Max.y, box.Max.z)
    };

    for (int i = 0; i < 8; i++) {
        float dist = plane.Classify(corners[i]);
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }

    // If all corners are on the same side of the plane, no intersection
    if (minDist > 0.0f || maxDist < 0.0f)
        return FALSE;

    // 3. Check if any edge of the triangle intersects the box
    VxRay edge;

    // Edge A0A1
    edge.m_Origin = A0;
    edge.m_Direction = A1 - A0;
    if (SegmentBox(edge, box))
        return TRUE;

    // Edge A1A2
    edge.m_Origin = A1;
    edge.m_Direction = A2 - A1;
    if (SegmentBox(edge, box))
        return TRUE;

    // Edge A2A0
    edge.m_Origin = A2;
    edge.m_Direction = A0 - A2;
    if (SegmentBox(edge, box))
        return TRUE;

    // 4. Check if any edge of the box intersects the triangle
    // Test 12 edges of the box against the triangle
    static const int edgeIndices[12][2] = {
        {0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5}, {2, 3},
        {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7}
    };

    VxVector point;
    float dist;

    for (int i = 0; i < 12; i++) {
        edge.m_Origin = corners[edgeIndices[i][0]];
        edge.m_Direction = corners[edgeIndices[i][1]] - corners[edgeIndices[i][0]];
        if (SegmentFace(edge, A0, A1, A2, N, point, dist))
            return TRUE;
    }

    return FALSE;
}
