#include "VxVector.h"

#include "VxMatrix.h"

int VxBbox::Classify(const VxBbox &box2, const VxVector &pt) const {
    // Classify point against this box
    XULONG ptFlags = 0;
    if (pt.x < Min.x) ptFlags |= VXCLIP_LEFT;
    else if (pt.x > Max.x) ptFlags |= VXCLIP_RIGHT;

    if (pt.y < Min.y) ptFlags |= VXCLIP_BOTTOM;
    else if (pt.y > Max.y) ptFlags |= VXCLIP_TOP;

    if (pt.z < Min.z) ptFlags |= VXCLIP_BACK;
    else if (pt.z > Max.z) ptFlags |= VXCLIP_FRONT;

    // Classify box2 against this box
    XULONG box2Flags = 0;
    if (box2.Max.z < Min.z) box2Flags |= VXCLIP_BACK;
    else if (box2.Min.z > Max.z) box2Flags |= VXCLIP_FRONT;

    if (box2.Max.x < Min.x) box2Flags |= VXCLIP_LEFT;
    else if (box2.Min.x > Max.x) box2Flags |= VXCLIP_RIGHT;

    if (box2.Max.y < Min.y) box2Flags |= VXCLIP_BOTTOM;
    else if (box2.Min.y > Max.y) box2Flags |= VXCLIP_TOP;

    if (ptFlags) {
        if (!box2Flags) {
            // Check if box2 is inside this box
            if (box2.Min.x >= Min.x && box2.Min.y >= Min.y && box2.Min.z >= Min.z &&
                box2.Max.x <= Max.x && box2.Max.y <= Max.y && box2.Max.z <= Max.z) {
                return -1;
            }

            // Check if this box is inside box2
            if (Min.x >= box2.Min.x && Min.y >= box2.Min.y && Min.z >= box2.Min.z &&
                Max.x <= box2.Max.x && Max.y <= box2.Max.y && Max.z <= box2.Max.z) {
                // Check if point is NOT in box2
                if (pt.x < box2.Min.x || pt.x > box2.Max.x ||
                    pt.y < box2.Min.y || pt.y > box2.Max.y ||
                    pt.z < box2.Min.z || pt.z > box2.Max.z) {
                    return 1;
                }
            }
        }
    } else {
        if (box2Flags) return -1;

        // Check if this box is inside box2
        if (Min.x >= box2.Min.x && Min.y >= box2.Min.y && Min.z >= box2.Min.z &&
            Max.x <= box2.Max.x && Max.y <= box2.Max.y && Max.z <= box2.Max.z) {
            return 1;
        }
    }

    return 0;
}

void VxBbox::ClassifyVertices(const int iVcount, XBYTE *iVertices, XULONG iStride, XULONG *oFlags) const {
    const float maxX = Max.x;
    const float maxY = Max.y;
    const float maxZ = Max.z;
    const float minX = Min.x;
    const float minY = Min.y;
    const float minZ = Min.z;

    for (int i = 0; i < iVcount; i++) {
        const float *v = reinterpret_cast<const float *>(iVertices + i * iStride);

        XULONG flag = 0;

        // Check Z axis
        if (v[2] < minZ) flag |= VXCLIP_BACK;
        else if (v[2] > maxZ) flag |= VXCLIP_FRONT;

        // Check Y axis
        if (v[1] < minY) flag |= VXCLIP_BOTTOM;
        else if (v[1] > maxY) flag |= VXCLIP_TOP;

        // Check X axis
        if (v[0] < minX) flag |= VXCLIP_LEFT;
        else if (v[0] > maxX) flag |= VXCLIP_RIGHT;

        oFlags[i] = flag;
    }
}

void VxBbox::ClassifyVerticesOneAxis(const int iVcount, XBYTE *iVertices, XULONG iStride, const int iAxis, XULONG *oFlags) const {
    if (iAxis < 0 || iAxis > 2) {
        // Invalid axis, set all flags to 0
        for (int i = 0; i < iVcount; i++)
            oFlags[i] = 0;
        return;
    }

    const float maxVal = *(&Max.x + iAxis);
    const float minVal = *(&Min.x + iAxis);

    for (int i = 0; i < iVcount; i++) {
        const float *v = reinterpret_cast<const float *>(iVertices + i * iStride + iAxis * sizeof(float));

        XULONG flag = 0;
        if (*v < minVal) flag = 1;
        else if (*v > maxVal) flag = 2;

        oFlags[i] = flag;
    }
}

void VxBbox::TransformTo(VxVector *pts, const VxMatrix &Mat) const {
    // Transform the Min corner first
    Vx3DMultiplyMatrixVector(&pts[0], Mat, &Min);

    // Calculate size differences
    const float sizeX = Max.x - Min.x;
    const float sizeY = Max.y - Min.y;
    const float sizeZ = Max.z - Min.z;

    // Calculate transformation vectors for each axis
    const VxVector xVec(sizeX * Mat[0][0], sizeX * Mat[0][1], sizeX * Mat[0][2]);
    const VxVector yVec(sizeY * Mat[1][0], sizeY * Mat[1][1], sizeY * Mat[1][2]);
    const VxVector zVec(sizeZ * Mat[2][0], sizeZ * Mat[2][1], sizeZ * Mat[2][2]);

    // Build the 8 corners by adding combinations of the transformation vectors
    pts[1] = pts[0] + zVec;           // Min + Z
    pts[2] = pts[0] + yVec;           // Min + Y
    pts[3] = pts[2] + zVec;           // Min + Y + Z
    pts[4] = pts[0] + xVec;           // Min + X
    pts[5] = pts[4] + zVec;           // Min + X + Z
    pts[6] = pts[4] + yVec;           // Min + X + Y
    pts[7] = pts[6] + zVec;           // Min + X + Y + Z
}

void VxBbox::TransformFrom(const VxBbox &sbox, const VxMatrix &Mat) {
    // Calculate center of source box
    VxVector center((sbox.Min.x + sbox.Max.x) * 0.5f,
                    (sbox.Min.y + sbox.Max.y) * 0.5f,
                    (sbox.Min.z + sbox.Max.z) * 0.5f);

    // Transform center to get new center
    Vx3DMultiplyMatrixVector(&Min, Mat, &center);

    // Calculate size differences
    const float sizeX = sbox.Max.x - sbox.Min.x;
    const float sizeY = sbox.Max.y - sbox.Min.y;
    const float sizeZ = sbox.Max.z - sbox.Min.z;

    // Transform size vectors and take absolute values
    const VxVector xVec(sizeX * Mat[0][0], sizeX * Mat[0][1], sizeX * Mat[0][2]);
    const VxVector yVec(sizeY * Mat[1][0], sizeY * Mat[1][1], sizeY * Mat[1][2]);
    const VxVector zVec(sizeZ * Mat[2][0], sizeZ * Mat[2][1], sizeZ * Mat[2][2]);

    // Calculate half extents by summing absolute values
    const float halfX = (XAbs(xVec.x) + XAbs(yVec.x) + XAbs(zVec.x)) * 0.5f;
    const float halfY = (XAbs(xVec.y) + XAbs(yVec.y) + XAbs(zVec.y)) * 0.5f;
    const float halfZ = (XAbs(xVec.z) + XAbs(yVec.z) + XAbs(zVec.z)) * 0.5f;

    // Set new bounds
    Max.x = Min.x + halfX;
    Max.y = Min.y + halfY;
    Max.z = Min.z + halfZ;

    Min.x = Min.x - halfX;
    Min.y = Min.y - halfY;
    Min.z = Min.z - halfZ;
}