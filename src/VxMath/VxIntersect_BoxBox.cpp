#include "VxIntersect.h"

#include <float.h>

#include "VxVector.h"
#include "VxMatrix.h"
#include "VxRay.h"
#include "VxOBB.h"
#include "VxPlane.h"

//----------- Boxes

// Intersection Ray - Box (simple boolean version)
// Aligned with binary at 0x2428c650
XBOOL VxIntersect::RayBox(const VxRay &ray, const VxBbox &box) {
    // Get box center and half-extents
    float boxHalfX = (box.Max.x - box.Min.x) * 0.5f;
    float boxHalfY = (box.Max.y - box.Min.y) * 0.5f;
    float boxHalfZ = (box.Max.z - box.Min.z) * 0.5f;
    
    VxVector boxCenter = (box.Max + box.Min) * 0.5f;

    // Vector from box center to ray origin
    float px = ray.m_Origin.x - boxCenter.x;
    float py = ray.m_Origin.y - boxCenter.y;
    float pz = ray.m_Origin.z - boxCenter.z;

    // Test if ray origin is outside box on any axis and heading away
    if (!(XAbs(px) <= boxHalfX)) {
        if (px * ray.m_Direction.x >= 0.0f) return FALSE;
    }
    if (!(XAbs(py) <= boxHalfY)) {
        if (py * ray.m_Direction.y >= 0.0f) return FALSE;
    }
    if (!(XAbs(pz) <= boxHalfZ)) {
        if (pz * ray.m_Direction.z >= 0.0f) return FALSE;
    }

    // Test separating axes from cross products (p x Direction)
    float crossX = pz * ray.m_Direction.y - py * ray.m_Direction.z;
    float crossY = px * ray.m_Direction.z - pz * ray.m_Direction.x;
    float crossZ = py * ray.m_Direction.x - px * ray.m_Direction.y;

    float absDirY = XAbs(ray.m_Direction.y);
    float absDirZ = XAbs(ray.m_Direction.z);
    
    if (XAbs(crossX) > absDirZ * boxHalfY + absDirY * boxHalfZ) return FALSE;
    
    float absDirX = XAbs(ray.m_Direction.x);
    
    if (XAbs(crossY) > absDirX * boxHalfZ + absDirZ * boxHalfX) return FALSE;
    if (XAbs(crossZ) > absDirX * boxHalfY + absDirY * boxHalfX) return FALSE;

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
// Aligned with binary at 0x2428cae0
XBOOL VxIntersect::SegmentBox(const VxRay &segment, const VxBbox &box) {
    // Get box center and half-extents
    float boxHalfX = (box.Max.x - box.Min.x) * 0.5f;
    float boxHalfY = (box.Max.y - box.Min.y) * 0.5f;
    float boxHalfZ = (box.Max.z - box.Min.z) * 0.5f;

    // Segment half-vector
    float segHalfX = segment.m_Direction.x * 0.5f;
    float segHalfY = segment.m_Direction.y * 0.5f;
    float segHalfZ = segment.m_Direction.z * 0.5f;
    
    // Segment center
    float segCenterX = segHalfX + segment.m_Origin.x;
    float segCenterY = segHalfY + segment.m_Origin.y;
    float segCenterZ = segHalfZ + segment.m_Origin.z;
    
    VxVector boxCenter = (box.Max + box.Min) * 0.5f;

    // Vector from box center to segment center
    float dx = segCenterX - boxCenter.x;
    float dy = segCenterY - boxCenter.y;
    float dz = segCenterZ - boxCenter.z;

    // Test separating axes
    float absSegHalfX = XAbs(segHalfX);
    if (XAbs(dx) > absSegHalfX + boxHalfX) return FALSE;
    
    float absSegHalfY = XAbs(segHalfY);
    if (XAbs(dy) > absSegHalfY + boxHalfY) return FALSE;
    
    float absSegHalfZ = XAbs(segHalfZ);
    if (XAbs(dz) > absSegHalfZ + boxHalfZ) return FALSE;

    // Test cross product axes (d x Direction)
    float crossY = segHalfZ * dx - dz * segHalfX;
    float crossZ = dy * segHalfX - segHalfY * dx;
    
    // Cross X = dz * segHalfY - segHalfZ * dy
    if (XAbs(dz * segHalfY - segHalfZ * dy) > absSegHalfZ * boxHalfY + absSegHalfY * boxHalfZ)
        return FALSE;
    if (XAbs(crossY) > absSegHalfZ * boxHalfX + absSegHalfX * boxHalfZ)
        return FALSE;
    if (XAbs(crossZ) > absSegHalfY * boxHalfX + absSegHalfX * boxHalfY)
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
// Aligned with binary at 0x2428cf70
XBOOL VxIntersect::LineBox(const VxRay &line, const VxBbox &box) {
    // Get box center and half-extents
    float boxHalfX = (box.Max.x - box.Min.x) * 0.5f;
    float boxHalfY = (box.Max.y - box.Min.y) * 0.5f;
    float boxHalfZ = (box.Max.z - box.Min.z) * 0.5f;
    
    VxVector boxCenter = (box.Max + box.Min) * 0.5f;

    // Vector from box center to line origin
    float dx = line.m_Origin.x - boxCenter.x;
    float dy = line.m_Origin.y - boxCenter.y;
    float dz = line.m_Origin.z - boxCenter.z;

    // Test separating axes from cross products (d x Direction)
    float crossX = dz * line.m_Direction.y - dy * line.m_Direction.z;
    float crossY = dx * line.m_Direction.z - dz * line.m_Direction.x;
    float crossZ = dy * line.m_Direction.x - dx * line.m_Direction.y;

    float absDirY = XAbs(line.m_Direction.y);
    float absDirZ = XAbs(line.m_Direction.z);
    
    if (XAbs(crossX) > absDirZ * boxHalfY + absDirY * boxHalfZ)
        return FALSE;
    
    float absDirX = XAbs(line.m_Direction.x);
    
    if (XAbs(crossY) > absDirX * boxHalfZ + absDirZ * boxHalfX)
        return FALSE;
    if (XAbs(crossZ) > absDirX * boxHalfY + absDirY * boxHalfX)
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
// Aligned with binary at 0x242877b0
XBOOL VxIntersect::AABBAABB(const VxBbox &box1, const VxBbox &box2) {
    // Check for separation along all axes
    // Binary uses: box1.Min.x > box2.Max.x (not >=) for non-intersection
    if (!(box1.Min.x <= box2.Max.x)) return FALSE;
    if (!(box1.Min.y <= box2.Max.y)) return FALSE;
    if (!(box1.Min.z <= box2.Max.z)) return FALSE;
    if (box1.Max.x < box2.Min.x) return FALSE;
    if (box1.Max.y < box2.Min.y) return FALSE;
    if (box1.Max.z < box2.Min.z) return FALSE;

    // No separation found, boxes intersect
    return TRUE;
}

// AABB - OBB intersection
// Aligned with binary at 0x24287dd0
XBOOL VxIntersect::AABBOBB(const VxBbox &box1, const VxOBB &box2) {
    // Get AABB center and half-extents
    float aabbHalfX = (box1.Max.x - box1.Min.x) * 0.5f;
    float aabbHalfY = (box1.Max.y - box1.Min.y) * 0.5f;
    float aabbHalfZ = (box1.Max.z - box1.Min.z) * 0.5f;
    
    VxVector aabbCenter = (box1.Max + box1.Min) * 0.5f;

    // Vector from AABB center to OBB center
    float Tx = box2.m_Center.x - aabbCenter.x;
    float Ty = box2.m_Center.y - aabbCenter.y;
    float Tz = box2.m_Center.z - aabbCenter.z;

    // OBB axis components
    float R00 = box2.m_Axis[0].x, R01 = box2.m_Axis[1].x, R02 = box2.m_Axis[2].x;
    float R10 = box2.m_Axis[0].y, R11 = box2.m_Axis[1].y, R12 = box2.m_Axis[2].y;
    float R20 = box2.m_Axis[0].z, R21 = box2.m_Axis[1].z, R22 = box2.m_Axis[2].z;

    float absR00 = XAbs(R00), absR01 = XAbs(R01), absR02 = XAbs(R02);
    float absR10 = XAbs(R10), absR11 = XAbs(R11), absR12 = XAbs(R12);
    float absR20 = XAbs(R20), absR21 = XAbs(R21), absR22 = XAbs(R22);

    // Test AABB X axis
    if (XAbs(Tx) > aabbHalfX + absR00 * box2.m_Extents.x + absR01 * box2.m_Extents.y + absR02 * box2.m_Extents.z)
        return FALSE;

    // Test AABB Y axis
    if (XAbs(Ty) > aabbHalfY + absR10 * box2.m_Extents.x + absR11 * box2.m_Extents.y + absR12 * box2.m_Extents.z)
        return FALSE;

    // Test AABB Z axis
    if (XAbs(Tz) > aabbHalfZ + absR20 * box2.m_Extents.x + absR21 * box2.m_Extents.y + absR22 * box2.m_Extents.z)
        return FALSE;

    // Test OBB axis 0
    if (XAbs(Tx * R00 + Ty * R10 + Tz * R20) > box2.m_Extents.x + absR00 * aabbHalfX + absR10 * aabbHalfY + absR20 * aabbHalfZ)
        return FALSE;

    // Test OBB axis 1
    if (XAbs(Tx * R01 + Ty * R11 + Tz * R21) > box2.m_Extents.y + absR01 * aabbHalfX + absR11 * aabbHalfY + absR21 * aabbHalfZ)
        return FALSE;

    // Test OBB axis 2
    if (XAbs(Tx * R02 + Ty * R12 + Tz * R22) > box2.m_Extents.z + absR02 * aabbHalfX + absR12 * aabbHalfY + absR22 * aabbHalfZ)
        return FALSE;

    // Test cross product axes
    // AABB X axis x OBB axis 0
    if (XAbs(Tz * R10 - R20 * Ty) > absR20 * aabbHalfY + absR10 * aabbHalfZ + absR02 * box2.m_Extents.y + absR01 * box2.m_Extents.z)
        return FALSE;

    // AABB X axis x OBB axis 1
    if (XAbs(Tz * R11 - R21 * Ty) > absR21 * aabbHalfY + absR11 * aabbHalfZ + absR00 * box2.m_Extents.z + absR02 * box2.m_Extents.x)
        return FALSE;

    // AABB X axis x OBB axis 2
    if (XAbs(Tz * R12 - R22 * Ty) > absR22 * aabbHalfY + absR12 * aabbHalfZ + absR00 * box2.m_Extents.y + absR01 * box2.m_Extents.x)
        return FALSE;

    // AABB Y axis x OBB axis 0
    if (XAbs(R20 * Tx - Tz * R00) > absR20 * aabbHalfX + absR00 * aabbHalfZ + absR12 * box2.m_Extents.y + absR11 * box2.m_Extents.z)
        return FALSE;

    // AABB Y axis x OBB axis 1
    if (XAbs(R21 * Tx - Tz * R01) > absR10 * box2.m_Extents.z + absR12 * box2.m_Extents.x + absR21 * aabbHalfX + absR01 * aabbHalfZ)
        return FALSE;

    // AABB Y axis x OBB axis 2
    if (XAbs(R22 * Tx - Tz * R02) > absR10 * box2.m_Extents.y + absR11 * box2.m_Extents.x + absR22 * aabbHalfX + absR02 * aabbHalfZ)
        return FALSE;

    // AABB Z axis x OBB axis 0
    if (XAbs(Ty * R00 - R10 * Tx) > absR10 * aabbHalfX + absR00 * aabbHalfY + absR22 * box2.m_Extents.y + absR21 * box2.m_Extents.z)
        return FALSE;

    // AABB Z axis x OBB axis 1
    if (XAbs(Ty * R01 - R11 * Tx) > absR11 * aabbHalfX + absR01 * aabbHalfY + absR20 * box2.m_Extents.z + absR22 * box2.m_Extents.x)
        return FALSE;

    // AABB Z axis x OBB axis 2
    if (XAbs(Ty * R02 - R12 * Tx) > absR12 * aabbHalfX + absR02 * aabbHalfY + absR20 * box2.m_Extents.y + absR21 * box2.m_Extents.x)
        return FALSE;

    return TRUE;
}

// OBB - OBB intersection using SAT (Separating Axis Theorem)
// Aligned with binary at 0x24287830
XBOOL VxIntersect::OBBOBB(const VxOBB &box1, const VxOBB &box2) {
    // Translation vector between box centers
    float Tx = box2.m_Center.x - box1.m_Center.x;
    float Ty = box2.m_Center.y - box1.m_Center.y;
    float Tz = box2.m_Center.z - box1.m_Center.z;

    // Rotation matrix from box1 to box2 coordinate system
    // R[i][j] = dot(box1.Axis[i], box2.Axis[j])
    float R00 = box1.m_Axis[0].x * box2.m_Axis[0].x + box1.m_Axis[0].y * box2.m_Axis[0].y + box1.m_Axis[0].z * box2.m_Axis[0].z;
    float R01 = box1.m_Axis[0].x * box2.m_Axis[1].x + box1.m_Axis[0].y * box2.m_Axis[1].y + box1.m_Axis[0].z * box2.m_Axis[1].z;
    float R02 = box1.m_Axis[0].x * box2.m_Axis[2].x + box1.m_Axis[0].y * box2.m_Axis[2].y + box1.m_Axis[0].z * box2.m_Axis[2].z;

    // T in box1's coordinate frame for axis A0
    float T0 = Tx * box1.m_Axis[0].x + Ty * box1.m_Axis[0].y + Tz * box1.m_Axis[0].z;
    float absR00 = XAbs(R00);
    float absR01 = XAbs(R01);
    float absR02 = XAbs(R02);

    // Test axis A0
    if (XAbs(T0) > box1.m_Extents.x + absR00 * box2.m_Extents.x + absR01 * box2.m_Extents.y + absR02 * box2.m_Extents.z)
        return FALSE;

    float R10 = box1.m_Axis[1].x * box2.m_Axis[0].x + box1.m_Axis[1].y * box2.m_Axis[0].y + box1.m_Axis[1].z * box2.m_Axis[0].z;
    float R11 = box1.m_Axis[1].x * box2.m_Axis[1].x + box1.m_Axis[1].y * box2.m_Axis[1].y + box1.m_Axis[1].z * box2.m_Axis[1].z;
    float R12 = box1.m_Axis[1].x * box2.m_Axis[2].x + box1.m_Axis[1].y * box2.m_Axis[2].y + box1.m_Axis[1].z * box2.m_Axis[2].z;

    // T in box1's coordinate frame for axis A1
    float T1 = Tx * box1.m_Axis[1].x + Ty * box1.m_Axis[1].y + Tz * box1.m_Axis[1].z;
    float absR10 = XAbs(R10);
    float absR11 = XAbs(R11);
    float absR12 = XAbs(R12);

    // Test axis A1
    if (XAbs(T1) > box1.m_Extents.y + absR10 * box2.m_Extents.x + absR11 * box2.m_Extents.y + absR12 * box2.m_Extents.z)
        return FALSE;

    float R20 = box1.m_Axis[2].x * box2.m_Axis[0].x + box1.m_Axis[2].y * box2.m_Axis[0].y + box1.m_Axis[2].z * box2.m_Axis[0].z;
    float R21 = box1.m_Axis[2].x * box2.m_Axis[1].x + box1.m_Axis[2].y * box2.m_Axis[1].y + box1.m_Axis[2].z * box2.m_Axis[1].z;
    float R22 = box1.m_Axis[2].x * box2.m_Axis[2].x + box1.m_Axis[2].y * box2.m_Axis[2].y + box1.m_Axis[2].z * box2.m_Axis[2].z;

    // T in box1's coordinate frame for axis A2
    float T2 = Tx * box1.m_Axis[2].x + Ty * box1.m_Axis[2].y + Tz * box1.m_Axis[2].z;
    float absR20 = XAbs(R20);
    float absR21 = XAbs(R21);
    float absR22 = XAbs(R22);

    // Test axis A2
    if (XAbs(T2) > box1.m_Extents.z + absR20 * box2.m_Extents.x + absR21 * box2.m_Extents.y + absR22 * box2.m_Extents.z)
        return FALSE;

    // Test axis B0
    if (XAbs(Tx * box2.m_Axis[0].x + Ty * box2.m_Axis[0].y + Tz * box2.m_Axis[0].z) >
        box2.m_Extents.x + absR00 * box1.m_Extents.x + absR10 * box1.m_Extents.y + absR20 * box1.m_Extents.z)
        return FALSE;

    // Test axis B1
    if (XAbs(Tx * box2.m_Axis[1].x + Ty * box2.m_Axis[1].y + Tz * box2.m_Axis[1].z) >
        box2.m_Extents.y + absR01 * box1.m_Extents.x + absR11 * box1.m_Extents.y + absR21 * box1.m_Extents.z)
        return FALSE;

    // Test axis B2
    if (XAbs(Tx * box2.m_Axis[2].x + Ty * box2.m_Axis[2].y + Tz * box2.m_Axis[2].z) >
        box2.m_Extents.z + absR02 * box1.m_Extents.x + absR12 * box1.m_Extents.y + absR22 * box1.m_Extents.z)
        return FALSE;

    // Test axis A0 x B0
    if (XAbs(T2 * R10 - R20 * T1) > absR20 * box1.m_Extents.y + absR10 * box1.m_Extents.z + absR02 * box2.m_Extents.y + absR01 * box2.m_Extents.z)
        return FALSE;

    // Test axis A0 x B1
    if (XAbs(T2 * R11 - R21 * T1) > absR21 * box1.m_Extents.y + absR11 * box1.m_Extents.z + absR02 * box2.m_Extents.x + absR00 * box2.m_Extents.z)
        return FALSE;

    // Test axis A0 x B2
    if (XAbs(T2 * R12 - R22 * T1) > absR22 * box1.m_Extents.y + absR12 * box1.m_Extents.z + absR01 * box2.m_Extents.x + absR00 * box2.m_Extents.y)
        return FALSE;

    // Test axis A1 x B0
    if (XAbs(R20 * T0 - T2 * R00) > absR20 * box1.m_Extents.x + absR00 * box1.m_Extents.z + absR12 * box2.m_Extents.y + absR11 * box2.m_Extents.z)
        return FALSE;

    // Test axis A1 x B1
    if (XAbs(R21 * T0 - T2 * R01) > absR21 * box1.m_Extents.x + absR01 * box1.m_Extents.z + absR12 * box2.m_Extents.x + absR10 * box2.m_Extents.z)
        return FALSE;

    // Test axis A1 x B2
    if (XAbs(R22 * T0 - T2 * R02) > absR22 * box1.m_Extents.x + absR02 * box1.m_Extents.z + absR11 * box2.m_Extents.x + absR10 * box2.m_Extents.y)
        return FALSE;

    // Test axis A2 x B0
    if (XAbs(T1 * R00 - R10 * T0) > absR10 * box1.m_Extents.x + absR00 * box1.m_Extents.y + absR22 * box2.m_Extents.y + absR21 * box2.m_Extents.z)
        return FALSE;

    // Test axis A2 x B1
    if (XAbs(T1 * R01 - R11 * T0) > absR11 * box1.m_Extents.x + absR01 * box1.m_Extents.y + absR22 * box2.m_Extents.x + absR20 * box2.m_Extents.z)
        return FALSE;

    // Test axis A2 x B2
    if (XAbs(T1 * R02 - R12 * T0) > absR12 * box1.m_Extents.x + absR02 * box1.m_Extents.y + absR21 * box2.m_Extents.x + absR20 * box2.m_Extents.y)
        return FALSE;

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
