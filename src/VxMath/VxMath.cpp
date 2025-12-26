#include "VxMath.h"

#include <stdio.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "VxEigenMatrix.h"

extern void InitializeTables();

HINSTANCE g_hinstDLL;
CRITICAL_SECTION g_CriticalSection;

void InitVxMath() {
    VxDetectProcessor();
    InitializeTables();
    ::InitializeCriticalSection(&g_CriticalSection);
}

void InterpolateFloatArray(void *Res, void *array1, void *array2, float factor, int count) {
    if (!Res || !array1 || !array2 || count <= 0) {
        return;
    }

    float *pRes = (float *) Res;
    float *pArray1 = (float *) array1;
    float *pArray2 = (float *) array2;

    // Ensure we have at least 4 elements as per documentation requirement
    if (count < 4) {
        // Fallback for smaller arrays
        for (int i = 0; i < count; i++) {
            pRes[i] = pArray1[i] + (pArray2[i] - pArray1[i]) * factor;
        }
        return;
    }

    // Process data in blocks of 4 where possible for better performance
    int blockCount = count >> 2;
    int remainder = count & 3;

    // Process 4 elements at once
    for (int i = 0; i < blockCount; i++) {
        pRes[0] = pArray1[0] + (pArray2[0] - pArray1[0]) * factor;
        pRes[1] = pArray1[1] + (pArray2[1] - pArray1[1]) * factor;
        pRes[2] = pArray1[2] + (pArray2[2] - pArray1[2]) * factor;
        pRes[3] = pArray1[3] + (pArray2[3] - pArray1[3]) * factor;

        pRes += 4;
        pArray1 += 4;
        pArray2 += 4;
    }

    // Process remaining elements
    for (int i = 0; i < remainder; i++) {
        pRes[i] = pArray1[i] + (pArray2[i] - pArray1[i]) * factor;
    }
}

void InterpolateVectorArray(void *Res, void *Inarray1, void *Inarray2, float factor, int count, XULONG StrideRes, XULONG StrideIn) {
    if (!Res || !Inarray1 || !Inarray2 || count <= 0) {
        return;
    }

    // Ensure we have at least 2 elements as per documentation requirement
    if (count < 2) {
        if (count == 1) {
            VxVector *vRes = (VxVector *) Res;
            const VxVector *v1 = (const VxVector *) Inarray1;
            const VxVector *v2 = (const VxVector *) Inarray2;

            vRes->x = v1->x + (v2->x - v1->x) * factor;
            vRes->y = v1->y + (v2->y - v1->y) * factor;
            vRes->z = v1->z + (v2->z - v1->z) * factor;
        }
        return;
    }

    XBYTE *pRes = (XBYTE *) Res;
    XBYTE *pArray1 = (XBYTE *) Inarray1;
    XBYTE *pArray2 = (XBYTE *) Inarray2;

    // Use default stride if zero
    if (StrideRes == 0) StrideRes = sizeof(VxVector);
    if (StrideIn == 0) StrideIn = sizeof(VxVector);

    for (int i = 0; i < count; i++) {
        VxVector *vRes = (VxVector *) pRes;
        const VxVector *v1 = (const VxVector *) pArray1;
        const VxVector *v2 = (const VxVector *) pArray2;

        // Interpolate each component
        vRes->x = v1->x + (v2->x - v1->x) * factor;
        vRes->y = v1->y + (v2->y - v1->y) * factor;
        vRes->z = v1->z + (v2->z - v1->z) * factor;

        // Advance pointers
        pRes += StrideRes;
        pArray1 += StrideIn;
        pArray2 += StrideIn;
    }
}

XBOOL VxTransformBox2D(const VxMatrix &World_ProjectionMat, const VxBbox &box, VxRect *ScreenSize, VxRect *Extents, VXCLIP_FLAGS &OrClipFlags, VXCLIP_FLAGS &AndClipFlags) {
    // Validate input
    if (!box.IsValid()) {
        OrClipFlags = VXCLIP_ALL;
        AndClipFlags = VXCLIP_ALL;
        return FALSE;
    }

    // Use local variables for thread safety
    VxVector4 transformedVertices[8];
    XULONG allVerticesFlags = 0;
    XULONG allVerticesFlagsAnded = 0xFFFFFFFF;

    // Transform first corner (Min)
    Vx3DMultiplyMatrixVector4(&transformedVertices[0], World_ProjectionMat, &box.Min);

    // Calculate box dimensions
    VxVector boxSize = box.Max - box.Min;
    float dx = boxSize.x;
    float dy = boxSize.y;
    float dz = boxSize.z;

    int vertexCount;

    // Pre-calculate matrix-vector products for better performance
    // Extract matrix elements for clarity and performance
    const VxVector4 &matRow0 = World_ProjectionMat[0];
    const VxVector4 &matRow1 = World_ProjectionMat[1];
    const VxVector4 &matRow2 = World_ProjectionMat[2];

    VxVector4 deltaX(matRow0[0] * dx, matRow0[1] * dx, matRow0[2] * dx, matRow0[3] * dx);
    VxVector4 deltaY(matRow1[0] * dy, matRow1[1] * dy, matRow1[2] * dy, matRow1[3] * dy);

    if (XAbs(dz) < EPSILON) {
        // Box is flat in Z dimension - only need 4 vertices
        vertexCount = 4;

        // Calculate corners
        transformedVertices[1] = transformedVertices[0] + deltaX;              // Min + (dx, 0, 0)
        transformedVertices[2] = transformedVertices[0] + deltaY;              // Min + (0, dy, 0)
        transformedVertices[3] = transformedVertices[1] + deltaY;              // Min + (dx, dy, 0)
    } else {
        // Full 3D box - need all 8 vertices
        vertexCount = 8;

        VxVector4 deltaZ(matRow2[0] * dz, matRow2[1] * dz, matRow2[2] * dz, matRow2[3] * dz);

        // Calculate all 8 corners
        transformedVertices[1] = transformedVertices[0] + deltaX;              // Min + (dx, 0, 0)
        transformedVertices[2] = transformedVertices[0] + deltaY;              // Min + (0, dy, 0)
        transformedVertices[3] = transformedVertices[1] + deltaY;              // Min + (dx, dy, 0)
        transformedVertices[4] = transformedVertices[0] + deltaZ;              // Min + (0, 0, dz)
        transformedVertices[5] = transformedVertices[4] + deltaX;              // Min + (dx, 0, dz)
        transformedVertices[6] = transformedVertices[4] + deltaY;              // Min + (0, dy, dz)
        transformedVertices[7] = transformedVertices[5] + deltaY;              // Min + (dx, dy, dz)
    }

    // Check each vertex against clip planes
    for (int i = 0; i < vertexCount; i++) {
        XULONG vertexFlags = 0;
        const VxVector4 &vertex = transformedVertices[i];

        // Check against frustum planes using homogeneous coordinates
        if (-vertex.w > vertex.x) vertexFlags |= VXCLIP_LEFT;
        if (vertex.x > vertex.w) vertexFlags |= VXCLIP_RIGHT;
        if (-vertex.w > vertex.y) vertexFlags |= VXCLIP_BOTTOM;
        if (vertex.y > vertex.w) vertexFlags |= VXCLIP_TOP;
        if (vertex.z < 0.0f) vertexFlags |= VXCLIP_BACK;
        if (vertex.z > vertex.w) vertexFlags |= VXCLIP_FRONT;

        // Update combined flags
        allVerticesFlags |= vertexFlags;
        allVerticesFlagsAnded &= vertexFlags;
    }

    // Calculate screen coordinates if box is at least partially visible and screen extents are requested
    if (Extents && ScreenSize && (allVerticesFlagsAnded & VXCLIP_ALL) == 0) {
        float minX = 1.0e10f;
        float minY = 1.0e10f;
        float maxX = -1.0e10f;
        float maxY = -1.0e10f;

        // Calculate viewport parameters
        float halfWidth = (ScreenSize->right - ScreenSize->left) * 0.5f;
        float halfHeight = (ScreenSize->bottom - ScreenSize->top) * 0.5f;
        float centerX = halfWidth + ScreenSize->left;
        float centerY = halfHeight + ScreenSize->top;

        // Project vertices and update extents
        for (int i = 0; i < vertexCount; i++) {
            const VxVector4 &vertex = transformedVertices[i];

            // Skip points behind viewer or at viewer position
            if (vertex.w <= EPSILON)
                continue;

            // Perform perspective division
            float invW = 1.0f / vertex.w;

            // Convert to screen coordinates
            float screenX = vertex.x * invW * halfWidth + centerX;
            float screenY = centerY - vertex.y * invW * halfHeight;

            // Update extents
            if (screenX < minX) minX = screenX;
            if (screenY < minY) minY = screenY;
            if (screenX > maxX) maxX = screenX;
            if (screenY > maxY) maxY = screenY;
        }

        // Set rectangle extents only if we found valid points
        if (minX <= maxX && minY <= maxY) {
            Extents->left = minX;
            Extents->top = minY;
            Extents->right = maxX;
            Extents->bottom = maxY;

            // Clamp to screen bounds if partially off-screen
            if (allVerticesFlags & VXCLIP_LEFT && Extents->left < ScreenSize->left)
                Extents->left = ScreenSize->left;
            if (allVerticesFlags & VXCLIP_RIGHT && Extents->right > ScreenSize->right)
                Extents->right = ScreenSize->right;
            if (allVerticesFlags & VXCLIP_TOP && Extents->top < ScreenSize->top)
                Extents->top = ScreenSize->top;
            if (allVerticesFlags & VXCLIP_BOTTOM && Extents->bottom > ScreenSize->bottom)
                Extents->bottom = ScreenSize->bottom;
        } else {
            // No valid projection found - set to empty rectangle
            Extents->left = Extents->right = 0;
            Extents->top = Extents->bottom = 0;
        }
    }

    // Return flags
    OrClipFlags = (VXCLIP_FLAGS) allVerticesFlags;
    AndClipFlags = (VXCLIP_FLAGS) allVerticesFlagsAnded;

    // Return TRUE if box is at least partially visible
    return (allVerticesFlagsAnded & VXCLIP_ALL) == 0;
}

void VxProjectBoxZExtents(const VxMatrix &World_ProjectionMat, const VxBbox &box, float &ZhMin, float &ZhMax) {
    // Initialize return values
    ZhMin = 1.0e10f;
    ZhMax = -1.0e10f;

    // Validate input
    if (!box.IsValid()) {
        return;
    }

    VxVector4 transformedVertex;

    // Transform box minimum corner
    Vx3DMultiplyMatrixVector4(&transformedVertex, World_ProjectionMat, &box.Min);

    // Calculate box dimensions
    VxVector boxSize = box.Max - box.Min;
    float dx = boxSize.x;
    float dy = boxSize.y;
    float dz = boxSize.z;

    // Pre-calculate matrix transformations for better performance
    const VxVector4 &matRow0 = World_ProjectionMat[0];
    const VxVector4 &matRow1 = World_ProjectionMat[1];
    const VxVector4 &matRow2 = World_ProjectionMat[2];

    // Store z,w pairs for each corner
    VxVector4 corners[8];
    int vertexCount;

    corners[0] = transformedVertex;  // Min corner

    if (XAbs(dz) < EPSILON) {
        // Box is flat in Z dimension - only need 4 corners
        vertexCount = 4;

        VxVector4 deltaX(matRow0[0] * dx, matRow0[1] * dx, matRow0[2] * dx, matRow0[3] * dx);
        VxVector4 deltaY(matRow1[0] * dy, matRow1[1] * dy, matRow1[2] * dy, matRow1[3] * dy);

        corners[1] = corners[0] + deltaX;      // Min + (dx, 0, 0)
        corners[2] = corners[0] + deltaY;      // Min + (0, dy, 0)
        corners[3] = corners[1] + deltaY;      // Min + (dx, dy, 0)
    } else {
        // Full 3D box - need all 8 corners
        vertexCount = 8;

        VxVector4 deltaX(matRow0[0] * dx, matRow0[1] * dx, matRow0[2] * dx, matRow0[3] * dx);
        VxVector4 deltaY(matRow1[0] * dy, matRow1[1] * dy, matRow1[2] * dy, matRow1[3] * dy);
        VxVector4 deltaZ(matRow2[0] * dz, matRow2[1] * dz, matRow2[2] * dz, matRow2[3] * dz);

        corners[1] = corners[0] + deltaZ;      // Min + (0, 0, dz)
        corners[2] = corners[0] + deltaY;      // Min + (0, dy, 0)
        corners[3] = corners[1] + deltaY;      // Min + (0, dy, dz)
        corners[4] = corners[0] + deltaX;      // Min + (dx, 0, 0)
        corners[5] = corners[4] + deltaZ;      // Min + (dx, 0, dz)
        corners[6] = corners[4] + deltaY;      // Min + (dx, dy, 0)
        corners[7] = corners[5] + deltaY;      // Min + (dx, dy, dz)
    }

    // Find min and max projected Z values
    for (int i = 0; i < vertexCount; i++) {
        float z = corners[i].z;
        float w = corners[i].w;

        // Skip if w is zero or negative (behind viewer)
        if (w <= EPSILON)
            continue;

        // Calculate perspective-divided Z
        float projZ = z / w;

        // Update min/max
        if (projZ < ZhMin) ZhMin = projZ;
        if (projZ > ZhMax) ZhMax = projZ;
    }

    // Handle case where all vertices are behind the viewer
    if (ZhMin > ZhMax) {
        ZhMin = 0.0f;
        ZhMax = 1.0f;
    }
}

XBOOL VxPtInRect(CKRECT *rect, CKPOINT *pt) {
    if (pt->x < rect->left)
        return FALSE;
    if (pt->x > rect->right)
        return FALSE;
    return pt->y <= rect->bottom && pt->y >= rect->top;
}

XBOOL VxComputeBestFitBBox(const XBYTE *Points, XULONG Stride, int Count, VxMatrix &BBoxMatrix, float AdditionalBorder) {
    if (Count <= 0 || !Points) {
        BBoxMatrix.SetIdentity();
        return FALSE;
    }

    // Create and compute covariance matrix
    VxEigenMatrix eigenMat;
    eigenMat.Covariance((float *) Points, Stride, Count);
    eigenMat.EigenStuff3();

    // Copy eigenvectors to the bounding box matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            BBoxMatrix[i][j] = eigenMat[i][j];
        }
    }

    // Normalize the first two axes
    BBoxMatrix[0].Normalize();
    BBoxMatrix[1].Normalize();

    // Compute the third axis using cross product for orthogonality
    float mx = BBoxMatrix[0][1] * BBoxMatrix[1][2] - BBoxMatrix[0][2] * BBoxMatrix[1][1];  // A.y*B.z - A.z*B.y
    float my = BBoxMatrix[0][2] * BBoxMatrix[1][0] - BBoxMatrix[0][0] * BBoxMatrix[1][2];  // A.z*B.x - A.x*B.z
    float mz = BBoxMatrix[0][0] * BBoxMatrix[1][1] - BBoxMatrix[0][1] * BBoxMatrix[1][0];  // A.x*B.y - A.y*B.x

    BBoxMatrix[2][0] = mx;
    BBoxMatrix[2][1] = my;
    BBoxMatrix[2][2] = mz;

    // Project all points onto the principal axes to find min/max extents
    const float *firstPoint = (const float *) Points;

    // Initialize min/max with the first point
    float projX = firstPoint[0] * BBoxMatrix[0][0] +
                  firstPoint[1] * BBoxMatrix[0][1] +
                  firstPoint[2] * BBoxMatrix[0][2];
    float minX = projX, maxX = projX;

    float projY = firstPoint[0] * BBoxMatrix[1][0] +
                  firstPoint[1] * BBoxMatrix[1][1] +
                  firstPoint[2] * BBoxMatrix[1][2];
    float minY = projY, maxY = projY;

    float projZ = firstPoint[0] * BBoxMatrix[2][0] +
                  firstPoint[1] * BBoxMatrix[2][1] +
                  firstPoint[2] * BBoxMatrix[2][2];
    float minZ = projZ, maxZ = projZ;

    // Find min/max extents for all points
    // Fixed: Process all points including the first one for consistency with decompiled code
    for (int i = 0; i < Count; i++) {
        const float *point = (const float *) (Points + i * Stride);

        float x = point[0] * BBoxMatrix[0][0] +
                  point[1] * BBoxMatrix[0][1] +
                  point[2] * BBoxMatrix[0][2];

        float y = point[0] * BBoxMatrix[1][0] +
                  point[1] * BBoxMatrix[1][1] +
                  point[2] * BBoxMatrix[1][2];

        float z = point[0] * BBoxMatrix[2][0] +
                  point[1] * BBoxMatrix[2][1] +
                  point[2] * BBoxMatrix[2][2];

        // Update min/max values
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
    }

    // Calculate center point in principal axis space
    float centerX = (minX + maxX) * 0.5f;
    float centerY = (minY + maxY) * 0.5f;
    float centerZ = (minZ + maxZ) * 0.5f;

    // Calculate half-extents with additional border
    float halfExtentX = (maxX - minX) * 0.5f + AdditionalBorder;
    float halfExtentY = (maxY - minY) * 0.5f + AdditionalBorder;
    float halfExtentZ = (maxZ - minZ) * 0.5f + AdditionalBorder;

    // Scale the axes by their respective half-extents
    for (int i = 0; i < 3; i++) {
        const float extent = (i == 0) ? halfExtentX : (i == 1) ? halfExtentY : halfExtentZ;

        BBoxMatrix[i][0] *= extent;
        BBoxMatrix[i][1] *= extent;
        BBoxMatrix[i][2] *= extent;
        BBoxMatrix[i][3] = 0.0f; // Set homogeneous component to 0
    }

    // Transform center from principal axis space to world space
    float worldX = centerX * BBoxMatrix[0][0] + centerY * BBoxMatrix[1][0] + centerZ * BBoxMatrix[2][0];
    float worldY = centerX * BBoxMatrix[0][1] + centerY * BBoxMatrix[1][1] + centerZ * BBoxMatrix[2][1];
    float worldZ = centerX * BBoxMatrix[0][2] + centerY * BBoxMatrix[1][2] + centerZ * BBoxMatrix[2][2];

    // Set transformed center position
    BBoxMatrix[3][0] = worldX;
    BBoxMatrix[3][1] = worldY;
    BBoxMatrix[3][2] = worldZ;
    BBoxMatrix[3][3] = 1.0f;

    // Validate matrix for reasonable values
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (fabs(BBoxMatrix[i][j]) > 1000000.0f)
                return FALSE;
        }
    }

    return TRUE;
}

#ifndef VX_LIB
BOOL APIENTRY DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            g_hinstDLL = hinstDLL;
            InitVxMath();
            break;
        case DLL_PROCESS_DETACH:
            DeleteCriticalSection(&g_CriticalSection);
            break;
        default:
            break;
    }
    return TRUE;
}
#endif
