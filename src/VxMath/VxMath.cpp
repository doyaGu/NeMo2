#include "VxMath.h"

#include <stdio.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

#include "VxEigenMatrix.h"

extern void InitializeTables();

#if defined(_WIN32)
HINSTANCE g_hinstDLL;
CRITICAL_SECTION g_CriticalSection;
#endif

void InitVxMath() {
    VxDetectProcessor();
    InitializeTables();
#if defined(_WIN32)
    ::InitializeCriticalSection(&g_CriticalSection);
#endif
}

void InterpolateFloatArray(void *Res, void *array1, void *array2, float factor, XDWORD count) {
    float *pRes = static_cast<float *>(Res);
    float *pArray1 = static_cast<float *>(array1);
    float *pArray2 = static_cast<float *>(array2);

    do {
        *pRes = *pArray1 + (*pArray2 - *pArray1) * factor;
        ++pArray1;
        ++pRes;
        ++pArray2;
        --count;
    } while (count != 0);
}

void InterpolateVectorArray(void *Res, void *Inarray1, void *Inarray2, float factor, XDWORD count, XDWORD StrideRes, XDWORD StrideIn) {
    VxVector *pRes = static_cast<VxVector *>(Res);
    VxVector *pArray1 = static_cast<VxVector *>(Inarray1);
    VxVector *pArray2 = static_cast<VxVector *>(Inarray2);

    do {
        pRes->z = pArray1->z + (pArray2->z - pArray1->z) * factor;
        pRes->y = pArray1->y + (pArray2->y - pArray1->y) * factor;
        pRes->x = pArray1->x + (pArray2->x - pArray1->x) * factor;
        pArray1 = reinterpret_cast<VxVector *>(reinterpret_cast<char *>(pArray1) + StrideIn);
        pArray2 = reinterpret_cast<VxVector *>(reinterpret_cast<char *>(pArray2) + StrideIn);
        pRes = reinterpret_cast<VxVector *>(reinterpret_cast<char *>(pRes) + StrideRes);
        --count;
    } while (count != 0);
}

XBOOL VxTransformBox2D(const VxMatrix &World_ProjectionMat, const VxBbox &box, VxRect *ScreenSize, VxRect *Extents, VXCLIP_FLAGS &OrClipFlags, VXCLIP_FLAGS &AndClipFlags) {
    // Validate input
    if (!box.IsValid()) {
        OrClipFlags = VXCLIP_ALL;
        AndClipFlags = VXCLIP_ALL;
        return FALSE;
    }

    // Initialize extents to zero for deterministic output
    if (Extents) {
        Extents->left = 0.0f;
        Extents->top = 0.0f;
        Extents->right = 0.0f;
        Extents->bottom = 0.0f;
    }

    // Use local variables for thread safety
    VxVector4 transformedVertices[8];
    XDWORD allVerticesFlags = 0;
    XDWORD allVerticesFlagsAnded = 0xFFFFFFFF;

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

    if (dz == 0.0f) {
        // Box is flat in Z dimension - only need 4 vertices
        vertexCount = 4;

        // Calculate corners - order matches original binary
        transformedVertices[1] = transformedVertices[0] + deltaX;              // Min + (dx, 0, 0)
        transformedVertices[2] = transformedVertices[0] + deltaY;              // Min + (0, dy, 0)
        transformedVertices[3] = transformedVertices[1] + deltaY;              // Min + (dx, dy, 0)
    } else {
        // Full 3D box - need all 8 vertices
        vertexCount = 8;

        VxVector4 deltaZ(matRow2[0] * dz, matRow2[1] * dz, matRow2[2] * dz, matRow2[3] * dz);

        // Calculate all 8 corners - order matches original binary exactly
        transformedVertices[1] = transformedVertices[0] + deltaZ;              // Min + (0, 0, dz)
        transformedVertices[2] = transformedVertices[0] + deltaY;              // Min + (0, dy, 0)
        transformedVertices[3] = transformedVertices[1] + deltaY;              // Min + (0, dy, dz)
        transformedVertices[4] = transformedVertices[0] + deltaX;              // Min + (dx, 0, 0)
        transformedVertices[5] = transformedVertices[4] + deltaZ;              // Min + (dx, 0, dz)
        transformedVertices[6] = transformedVertices[4] + deltaY;              // Min + (dx, dy, 0)
        transformedVertices[7] = transformedVertices[5] + deltaY;              // Min + (dx, dy, dz)
    }

    // Check each vertex against clip planes
    for (int i = 0; i < vertexCount; i++) {
        XDWORD vertexFlags = 0;
        const VxVector4 &vertex = transformedVertices[i];

        // Check against frustum planes using homogeneous coordinates
        // Order and semantics match original binary exactly
        if (-vertex.w > vertex.x) vertexFlags |= VXCLIP_LEFT;    // bit 4 (0x10)
        if (vertex.x > vertex.w) vertexFlags |= VXCLIP_RIGHT;    // bit 5 (0x20)
        if (-vertex.w > vertex.y) vertexFlags |= VXCLIP_BOTTOM;  // bit 7 (0x80)
        if (vertex.y > vertex.w) vertexFlags |= VXCLIP_TOP;      // bit 6 (0x40)
        if (vertex.z < 0.0f) vertexFlags |= VXCLIP_FRONT;        // bit 8 (0x100) - z behind near plane
        if (vertex.z > vertex.w) vertexFlags |= VXCLIP_BACK;     // bit 9 (0x200) - z beyond far plane

        // Update combined flags
        allVerticesFlags |= vertexFlags;
        allVerticesFlagsAnded &= vertexFlags;
    }

    // Calculate screen coordinates if box is at least partially visible and screen extents are requested
    if (Extents && ScreenSize && !(allVerticesFlagsAnded & VXCLIP_ALL)) {
        float minX = 1.0e6f;
        float minY = 1.0e6f;
        float maxX = -1.0e6f;
        float maxY = -1.0e6f;

        // Calculate viewport parameters
        float halfWidth = (ScreenSize->right - ScreenSize->left) * 0.5f;
        float halfHeight = (ScreenSize->bottom - ScreenSize->top) * 0.5f;
        float centerX = halfWidth + ScreenSize->left;
        float centerY = halfHeight + ScreenSize->top;

        // Project vertices and update extents
        for (int i = 0; i < vertexCount; i++) {
            VxVector4 &vertex = transformedVertices[i];

            // Skip points behind viewer or at viewer position (w <= 0)
            if (vertex.w <= 0.0f)
                continue;

            // Perform perspective division
            float invW = 1.0f / vertex.w;
            vertex.w = invW;  // Store invW back (matches original binary)

            // Convert to screen coordinates
            float screenX = vertex.x * invW * halfWidth + centerX;
            float screenY = centerY - vertex.y * invW * halfHeight;

            // Store back into vertex (matches original binary behavior)
            vertex.x = screenX;
            vertex.y = screenY;

            // Update extents
            if (screenX < minX) minX = screenX;
            if (screenY < minY) minY = screenY;
            if (screenX > maxX) maxX = screenX;
            if (screenY > maxY) maxY = screenY;
        }

        // Set rectangle extents
        Extents->left = minX;
        Extents->top = minY;
        Extents->right = maxX;
        Extents->bottom = maxY;
    }

    // Clamp to screen bounds only when some vertices are behind near plane (VXCLIP_FRONT set)
    if ((allVerticesFlags & VXCLIP_FRONT) && Extents && ScreenSize) {
        if (allVerticesFlags & VXCLIP_LEFT)   Extents->left = ScreenSize->left;
        if (allVerticesFlags & VXCLIP_RIGHT)  Extents->right = ScreenSize->right;
        if (allVerticesFlags & VXCLIP_TOP)    Extents->top = ScreenSize->top;
        if (allVerticesFlags & VXCLIP_BOTTOM) Extents->bottom = ScreenSize->bottom;
    }

    // Return flags
    OrClipFlags = (VXCLIP_FLAGS) allVerticesFlags;
    AndClipFlags = (VXCLIP_FLAGS) allVerticesFlagsAnded;

    // Return TRUE if box is at least partially visible
    return !(allVerticesFlagsAnded & VXCLIP_ALL);
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


/**
 * Fills an array with copies of a structure
 */
XBOOL VxFillStructure(XDWORD Count, void *Dst, XDWORD Stride, XDWORD SizeSrc, const void *Src) {
    if (!Src || !Dst || !Count || !SizeSrc || !Stride || (SizeSrc & 3) != 0)
        return FALSE;

    XDWORD *pDst = static_cast<XDWORD *>(Dst);
    const XDWORD *pSrc = static_cast<const XDWORD *>(Src);

    switch (SizeSrc) {
    case 4: {
        XDWORD cnt = Count;
        XDWORD v = *pSrc;
        do {
            *pDst = v;
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + Stride);
            --cnt;
        } while (cnt);
        break;
    }
    case 8: {
        XDWORD cnt = Count;
        XDWORD v0 = pSrc[0];
        XDWORD v1 = pSrc[1];
        do {
            pDst[0] = v0;
            pDst[1] = v1;
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + Stride);
            --cnt;
        } while (cnt);
        break;
    }
    case 12: {
        XDWORD cnt = Count;
        XDWORD v0 = pSrc[0];
        XDWORD v1 = pSrc[1];
        XDWORD v2 = pSrc[2];
        do {
            pDst[0] = v0;
            pDst[1] = v1;
            pDst[2] = v2;
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + Stride);
            --cnt;
        } while (cnt);
        break;
    }
    case 16: {
        XDWORD cnt = Count;
        XDWORD v0 = pSrc[0];
        XDWORD v1 = pSrc[1];
        XDWORD v2 = pSrc[2];
        XDWORD v3 = pSrc[3];
        do {
            pDst[0] = v0;
            pDst[1] = v1;
            pDst[2] = v2;
            pDst[3] = v3;
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + Stride);
            --cnt;
        } while (cnt);
        break;
    }
    default: {
        XDWORD cnt = Count;
        do {
            XDWORD dwordCount = SizeSrc >> 2;
            do {
                *pDst++ = *pSrc++;
                --dwordCount;
            } while (dwordCount);
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + Stride - SizeSrc);
            pSrc = static_cast<const XDWORD *>(Src);
            --cnt;
        } while (cnt);
        break;
    }
    }

    return TRUE;
}

/**
 * Copies structures from one array to another
 */
XBOOL VxCopyStructure(XDWORD Count, void *Dst, XDWORD OutStride, XDWORD SizeSrc, const void *Src, XDWORD InStride) {
    if (!Src || !Dst || !Count || !SizeSrc || !OutStride || !InStride || (SizeSrc & 3) != 0)
        return FALSE;

    const XDWORD *pSrc = static_cast<const XDWORD *>(Src);
    XDWORD *pDst = static_cast<XDWORD *>(Dst);

    switch (SizeSrc) {
    case 4: {
        XDWORD cnt = Count;
        do {
            *pDst = *pSrc;
            pSrc = reinterpret_cast<const XDWORD *>(reinterpret_cast<const char *>(pSrc) + InStride);
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + OutStride);
            --cnt;
        } while (cnt);
        break;
    }
    case 8: {
        XDWORD cnt = Count;
        do {
            XDWORD v1 = pSrc[1];
            pDst[0] = pSrc[0];
            pDst[1] = v1;
            pSrc = reinterpret_cast<const XDWORD *>(reinterpret_cast<const char *>(pSrc) + InStride);
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + OutStride);
            --cnt;
        } while (cnt);
        break;
    }
    case 12: {
        XDWORD cnt = Count;
        do {
            XDWORD v1 = pSrc[1];
            pDst[0] = pSrc[0];
            XDWORD v2 = pSrc[2];
            pDst[1] = v1;
            pDst[2] = v2;
            pSrc = reinterpret_cast<const XDWORD *>(reinterpret_cast<const char *>(pSrc) + InStride);
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + OutStride);
            --cnt;
        } while (cnt);
        break;
    }
    case 16: {
        XDWORD cnt = Count;
        do {
            XDWORD v1 = pSrc[1];
            pDst[0] = pSrc[0];
            pDst[1] = v1;
            XDWORD v3 = pSrc[3];
            pDst[2] = pSrc[2];
            pDst[3] = v3;
            pSrc = reinterpret_cast<const XDWORD *>(reinterpret_cast<const char *>(pSrc) + InStride);
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + OutStride);
            --cnt;
        } while (cnt);
        break;
    }
    default: {
        XDWORD cnt = Count;
        XDWORD InStrideAdj = InStride - SizeSrc;
        do {
            XDWORD dwordCount = SizeSrc >> 2;
            do {
                *pDst++ = *pSrc++;
                --dwordCount;
            } while (dwordCount);
            pSrc = reinterpret_cast<const XDWORD *>(reinterpret_cast<const char *>(pSrc) + InStrideAdj);
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + OutStride - SizeSrc);
            --cnt;
        } while (cnt);
        break;
    }
    }

    return TRUE;
}

/**
 * Copies structures from an array to another using an index array
 */
XBOOL VxIndexedCopy(const VxStridedData &Dst, const VxStridedData &Src, XDWORD SizeSrc, XDWORD *Indices, XDWORD IndexCount) {
    // Validate parameters
    if (IndexCount == 0)
        return FALSE;
    if (SizeSrc == 0 || (SizeSrc & 3) != 0)
        return FALSE;
    if (Dst.Ptr == nullptr || Src.Ptr == nullptr || Indices == nullptr)
        return FALSE;

    XDWORD DstStride = Dst.Stride;
    VxStridedData srcCopy = Src;

    switch (SizeSrc) {
    case 4: {
        XDWORD cnt = IndexCount;
        XDWORD *pDst = static_cast<XDWORD *>(Dst.Ptr);
        XDWORD *pIdx = Indices;
        do {
            *pDst = *reinterpret_cast<XDWORD *>(static_cast<char *>(srcCopy.Ptr) + *pIdx * srcCopy.Stride);
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + DstStride);
            ++pIdx;
            --cnt;
        } while (cnt);
        break;
    }
    case 8: {
        XDWORD cnt = IndexCount;
        XDWORD *pDst = static_cast<XDWORD *>(Dst.Ptr);
        XDWORD *pIdx = Indices;
        do {
            XDWORD *pSrc = reinterpret_cast<XDWORD *>(static_cast<char *>(srcCopy.Ptr) + *pIdx * srcCopy.Stride);
            XDWORD v1 = pSrc[1];
            pDst[0] = pSrc[0];
            pDst[1] = v1;
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + DstStride);
            ++pIdx;
            --cnt;
        } while (cnt);
        break;
    }
    case 12: {
        XDWORD cnt = IndexCount;
        XDWORD *pDst = static_cast<XDWORD *>(Dst.Ptr);
        XDWORD *pIdx = Indices;
        do {
            XDWORD *pSrc = reinterpret_cast<XDWORD *>(static_cast<char *>(srcCopy.Ptr) + *pIdx * srcCopy.Stride);
            XDWORD v1 = pSrc[1];
            pDst[0] = pSrc[0];
            pDst[1] = v1;
            pDst[2] = pSrc[2];
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + DstStride);
            ++pIdx;
            --cnt;
        } while (cnt);
        break;
    }
    case 16: {
        XDWORD cnt = IndexCount;
        XDWORD *pDst = static_cast<XDWORD *>(Dst.Ptr);
        XDWORD *pIdx = Indices;
        do {
            XDWORD *pSrc = reinterpret_cast<XDWORD *>(static_cast<char *>(srcCopy.Ptr) + *pIdx * srcCopy.Stride);
            XDWORD v1 = pSrc[1];
            pDst[0] = pSrc[0];
            pDst[1] = v1;
            XDWORD v3 = pSrc[3];
            pDst[2] = pSrc[2];
            pDst[3] = v3;
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + DstStride);
            ++pIdx;
            --cnt;
        } while (cnt);
        break;
    }
    case 20: {
        XDWORD cnt = IndexCount;
        XDWORD *pDst = static_cast<XDWORD *>(Dst.Ptr);
        XDWORD *pIdx = Indices;
        do {
            XDWORD *pSrc = reinterpret_cast<XDWORD *>(static_cast<char *>(srcCopy.Ptr) + *pIdx * srcCopy.Stride);
            XDWORD v1 = pSrc[1];
            pDst[0] = pSrc[0];
            pDst[1] = v1;
            XDWORD v3 = pSrc[3];
            pDst[2] = pSrc[2];
            pDst[3] = v3;
            pDst[4] = pSrc[4];
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + DstStride);
            ++pIdx;
            --cnt;
        } while (cnt);
        break;
    }
    case 24: {
        XDWORD cnt = IndexCount;
        XDWORD *pDst = static_cast<XDWORD *>(Dst.Ptr);
        XDWORD *pIdx = Indices;
        do {
            XDWORD *pSrc = reinterpret_cast<XDWORD *>(static_cast<char *>(srcCopy.Ptr) + *pIdx * srcCopy.Stride);
            XDWORD v1 = pSrc[1];
            pDst[0] = pSrc[0];
            pDst[1] = v1;
            XDWORD v3 = pSrc[3];
            pDst[2] = pSrc[2];
            pDst[3] = v3;
            XDWORD v5 = pSrc[5];
            pDst[4] = pSrc[4];
            pDst[5] = v5;
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDst) + DstStride);
            ++pIdx;
            --cnt;
        } while (cnt);
        break;
    }
    default: {
        XDWORD cnt = IndexCount;
        XDWORD *pDst = static_cast<XDWORD *>(Dst.Ptr);
        XDWORD *pIdx = Indices;
        do {
            XDWORD *pSrc = reinterpret_cast<XDWORD *>(static_cast<char *>(srcCopy.Ptr) + *pIdx * srcCopy.Stride);
            XDWORD *pDstSave = pDst;
            XDWORD dwordCount = SizeSrc >> 2;
            do {
                *pDst++ = *pSrc++;
                --dwordCount;
            } while (dwordCount);
            pDst = reinterpret_cast<XDWORD *>(reinterpret_cast<char *>(pDstSave) + DstStride);
            ++pIdx;
            --cnt;
        } while (cnt);
        break;
    }
    }

    return TRUE;
}

XBOOL VxPtInRect(CKRECT *rect, CKPOINT *pt) {
    if (pt->x < rect->left)
        return FALSE;
    if (pt->x > rect->right)
        return FALSE;
    return pt->y <= rect->bottom && pt->y >= rect->top;
}

XBOOL VxComputeBestFitBBox(const XBYTE *Points, XDWORD Stride, XDWORD Count, VxMatrix &BBoxMatrix, float AdditionalBorder) {
    if (Count > 0 && Points) {
        // Create and compute covariance matrix
        VxEigenMatrix eigenMat;
        eigenMat.Covariance((float *) Points, Stride, Count);
        eigenMat.EigenStuff3();

        // Copy eigenvectors to the bounding box matrix (transposed)
        // eigenMat[j][i] -> BBoxMatrix[i][j]
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                BBoxMatrix[i][j] = eigenMat[j][i];
            }
            BBoxMatrix[i][3] = 0.0f;
        }
        BBoxMatrix[3][0] = 0.0f;
        BBoxMatrix[3][1] = 0.0f;
        BBoxMatrix[3][2] = 0.0f;
        BBoxMatrix[3][3] = 1.0f;

        // Normalize the first two axes
        BBoxMatrix[0].Normalize();
        BBoxMatrix[1].Normalize();

        // Compute the third axis using cross product for orthogonality
        // Cross = (A.y*B.z - A.z*B.y, A.z*B.x - A.x*B.z, A.x*B.y - A.y*B.x)
        float mx = BBoxMatrix[0][1] * BBoxMatrix[1][2] - BBoxMatrix[0][2] * BBoxMatrix[1][1];
        float my = BBoxMatrix[0][2] * BBoxMatrix[1][0] - BBoxMatrix[0][0] * BBoxMatrix[1][2];
        float mz = BBoxMatrix[0][0] * BBoxMatrix[1][1] - BBoxMatrix[0][1] * BBoxMatrix[1][0];

        BBoxMatrix[2][0] = mx;
        BBoxMatrix[2][1] = my;
        BBoxMatrix[2][2] = mz;
        BBoxMatrix[2][3] = 0.0f;

        // Project all points onto the principal axes to find min/max extents
        const float *pPoint = (const float *) Points;

        // Initialize min/max with the first point projection
        float projX = pPoint[0] * BBoxMatrix[0][0] + pPoint[1] * BBoxMatrix[0][1] + pPoint[2] * BBoxMatrix[0][2];
        float minX = projX, maxX = projX;

        float projY = pPoint[0] * BBoxMatrix[1][0] + pPoint[1] * BBoxMatrix[1][1] + pPoint[2] * BBoxMatrix[1][2];
        float minY = projY, maxY = projY;

        float projZ = pPoint[0] * BBoxMatrix[2][0] + pPoint[1] * BBoxMatrix[2][1] + pPoint[2] * BBoxMatrix[2][2];
        float minZ = projZ, maxZ = projZ;

        // Find min/max extents for all points
        for (XDWORD i = 0; i < Count; ++i) {
            const float *p = (const float *)(Points + i * Stride);

            float x = p[0] * BBoxMatrix[0][0] + p[1] * BBoxMatrix[0][1] + p[2] * BBoxMatrix[0][2];
            if (x >= maxX)
                maxX = x;
            else if (x < minX)
                minX = x;

            float y = p[0] * BBoxMatrix[1][0] + p[1] * BBoxMatrix[1][1] + p[2] * BBoxMatrix[1][2];
            if (y >= maxY)
                maxY = y;
            else if (y < minY)
                minY = y;

            float z = p[0] * BBoxMatrix[2][0] + p[1] * BBoxMatrix[2][1] + p[2] * BBoxMatrix[2][2];
            if (z >= maxZ)
                maxZ = z;
            else if (z < minZ)
                minZ = z;
        }

        // Calculate center point in principal axis space
        float centerX = (minX + maxX) * 0.5f;
        float centerY = (minY + maxY) * 0.5f;
        float centerZ = (minZ + maxZ) * 0.5f;

        // Calculate half-extents with additional border
        float halfX = (maxX - minX) * 0.5f + AdditionalBorder;
        float halfY = (maxY - minY) * 0.5f + AdditionalBorder;
        float halfZ = (maxZ - minZ) * 0.5f + AdditionalBorder;

        // Transform center from principal axis space to world space (using unscaled axes)
        BBoxMatrix[3][0] = centerX * BBoxMatrix[0][0] + centerY * BBoxMatrix[1][0] + centerZ * BBoxMatrix[2][0];
        BBoxMatrix[3][1] = centerX * BBoxMatrix[0][1] + centerY * BBoxMatrix[1][1] + centerZ * BBoxMatrix[2][1];
        BBoxMatrix[3][2] = centerX * BBoxMatrix[0][2] + centerY * BBoxMatrix[1][2] + centerZ * BBoxMatrix[2][2];
        BBoxMatrix[3][3] = 1.0f;

        // Scale the axes by their respective half-extents
        for (int j = 0; j < 3; ++j) {
            BBoxMatrix[0][j] *= halfX;
        }
        BBoxMatrix[0][3] = 0.0f;

        for (int j = 0; j < 3; ++j) {
            BBoxMatrix[1][j] *= halfY;
        }
        BBoxMatrix[1][3] = 0.0f;

        for (int j = 0; j < 3; ++j) {
            BBoxMatrix[2][j] *= halfZ;
        }
        BBoxMatrix[2][3] = 0.0f;

        // Validate matrix for reasonable values
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (fabs(BBoxMatrix[i][j]) >= 1.0e6f)
                    return FALSE;
            }
        }

        return TRUE;
    } else {
        BBoxMatrix.SetIdentity();
    }
    return FALSE;
}

#if defined(_WIN32) && !defined(VX_LIB)
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
