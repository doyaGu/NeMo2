#include "VxMath.h"

/**
 * Fills an array with copies of a structure
 */
XBOOL VxFillStructure(int Count, void *Dst, XULONG Stride, XULONG SizeSrc, void *Src) {
    // Validate parameters
    if (!Src || !Dst || Count <= 0 || SizeSrc == 0 || Stride == 0 || (SizeSrc & 3) != 0)
        return FALSE;

    // Get pointers for 32-bit aligned access
    XDWORD *pSrc = (XDWORD *) Src;
    XBYTE *pDst = (XBYTE *) Dst;

    // Handle common structure sizes with optimized code
    switch (SizeSrc) {
    case 4: // 4-byte structure (single DWORD)
    {
        XDWORD value = *pSrc;
        for (int i = 0; i < Count; i++) {
            *(XDWORD *) pDst = value;
            pDst += Stride;
        }
        break;
    }

    case 8: // 8-byte structure (2 DWORDs)
    {
        XDWORD value1 = pSrc[0];
        XDWORD value2 = pSrc[1];
        for (int i = 0; i < Count; i++) {
            XDWORD *dest = (XDWORD *) pDst;
            dest[0] = value1;
            dest[1] = value2;
            pDst += Stride;
        }
        break;
    }

    case 12: // 12-byte structure (3 DWORDs, e.g., VxVector)
    {
        XDWORD value1 = pSrc[0];
        XDWORD value2 = pSrc[1];
        XDWORD value3 = pSrc[2];
        for (int i = 0; i < Count; i++) {
            XDWORD *dest = (XDWORD *) pDst;
            dest[0] = value1;
            dest[1] = value2;
            dest[2] = value3;
            pDst += Stride;
        }
        break;
    }

    case 16: // 16-byte structure (4 DWORDs, e.g., VxVector4)
    {
        XDWORD value1 = pSrc[0];
        XDWORD value2 = pSrc[1];
        XDWORD value3 = pSrc[2];
        XDWORD value4 = pSrc[3];
        for (int i = 0; i < Count; i++) {
            XDWORD *dest = (XDWORD *) pDst;
            dest[0] = value1;
            dest[1] = value2;
            dest[2] = value3;
            dest[3] = value4;
            pDst += Stride;
        }
        break;
    }

    default: // General case for any size
    {
        XULONG dwordCount = SizeSrc >> 2; // Divide by 4 to get DWORD count
        for (int i = 0; i < Count; i++) {
            XDWORD *dest = (XDWORD *) pDst;
            for (XULONG j = 0; j < dwordCount; j++) {
                dest[j] = pSrc[j];
            }
            pDst += Stride;
        }
        break;
    }
    }

    return TRUE;
}

/**
 * Copies structures from one array to another
 */
XBOOL VxCopyStructure(int Count, void *Dst, XULONG OutStride, XULONG SizeSrc, void *Src, XULONG InStride) {
    // Validate parameters
    if (!Src || !Dst || Count <= 0 || SizeSrc == 0 || OutStride == 0 || InStride == 0 || (SizeSrc & 3) != 0)
        return FALSE;

    // Get pointers for byte access
    XBYTE *pSrc = (XBYTE *) Src;
    XBYTE *pDst = (XBYTE *) Dst;

    // Handle common structure sizes with optimized code
    switch (SizeSrc) {
    case 4: // 4-byte structure (single DWORD)
    {
        for (int i = 0; i < Count; i++) {
            *(XDWORD *) pDst = *(XDWORD *) pSrc;
            pSrc += InStride;
            pDst += OutStride;
        }
        break;
    }

    case 8: // 8-byte structure (2 DWORDs)
    {
        for (int i = 0; i < Count; i++) {
            XDWORD *src = (XDWORD *) pSrc;
            XDWORD *dest = (XDWORD *) pDst;
            dest[0] = src[0];
            dest[1] = src[1];
            pSrc += InStride;
            pDst += OutStride;
        }
        break;
    }

    case 12: // 12-byte structure (3 DWORDs, e.g., VxVector)
    {
        for (int i = 0; i < Count; i++) {
            XDWORD *src = (XDWORD *) pSrc;
            XDWORD *dest = (XDWORD *) pDst;
            dest[0] = src[0];
            dest[1] = src[1];
            dest[2] = src[2];
            pSrc += InStride;
            pDst += OutStride;
        }
        break;
    }

    case 16: // 16-byte structure (4 DWORDs, e.g., VxVector4)
    {
        for (int i = 0; i < Count; i++) {
            XDWORD *src = (XDWORD *) pSrc;
            XDWORD *dest = (XDWORD *) pDst;
            dest[0] = src[0];
            dest[1] = src[1];
            dest[2] = src[2];
            dest[3] = src[3];
            pSrc += InStride;
            pDst += OutStride;
        }
        break;
    }

    default: // General case for any size
    {
        XULONG dwordCount = SizeSrc >> 2; // Divide by 4 to get DWORD count
        for (int i = 0; i < Count; i++) {
            XDWORD *src = (XDWORD *) pSrc;
            XDWORD *dest = (XDWORD *) pDst;
            for (XULONG j = 0; j < dwordCount; j++) {
                dest[j] = src[j];
            }
            pSrc += InStride;
            pDst += OutStride;
        }
        break;
    }
    }

    return TRUE;
}

/**
 * Copies structures from an array to another using an index array
 */
XBOOL VxIndexedCopy(const VxStridedData &Dst, const VxStridedData &Src, XULONG SizeSrc, int *Indices, int IndexCount) {
    // Validate parameters
    if (!Dst.Ptr || !Src.Ptr || !Indices || IndexCount <= 0 || SizeSrc == 0 || (SizeSrc & 3) != 0)
        return FALSE;

    // Handle common structure sizes with optimized code
    switch (SizeSrc) {
    case 4: // 4-byte structure (single DWORD)
    {
        for (int i = 0; i < IndexCount; i++) {
            XDWORD *src = (XDWORD *) ((XBYTE *) Src.Ptr + Indices[i] * Src.Stride);
            XDWORD *dst = (XDWORD *) ((XBYTE *) Dst.Ptr + i * Dst.Stride);
            *dst = *src;
        }
        break;
    }

    case 8: // 8-byte structure (2 DWORDs)
    {
        for (int i = 0; i < IndexCount; i++) {
            XDWORD *src = (XDWORD *) ((XBYTE *) Src.Ptr + Indices[i] * Src.Stride);
            XDWORD *dst = (XDWORD *) ((XBYTE *) Dst.Ptr + i * Dst.Stride);
            dst[0] = src[0];
            dst[1] = src[1];
        }
        break;
    }

    case 12: // 12-byte structure (3 DWORDs, e.g., VxVector)
    {
        for (int i = 0; i < IndexCount; i++) {
            XDWORD *src = (XDWORD *) ((XBYTE *) Src.Ptr + Indices[i] * Src.Stride);
            XDWORD *dst = (XDWORD *) ((XBYTE *) Dst.Ptr + i * Dst.Stride);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
        break;
    }

    case 16: // 16-byte structure (4 DWORDs, e.g., VxVector4)
    {
        for (int i = 0; i < IndexCount; i++) {
            XDWORD *src = (XDWORD *) ((XBYTE *) Src.Ptr + Indices[i] * Src.Stride);
            XDWORD *dst = (XDWORD *) ((XBYTE *) Dst.Ptr + i * Dst.Stride);
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
        }
        break;
    }

    default: // General case for any size
    {
        XULONG dwordCount = SizeSrc >> 2; // Divide by 4 to get DWORD count
        for (int i = 0; i < IndexCount; i++) {
            XDWORD *src = (XDWORD *) ((XBYTE *) Src.Ptr + Indices[i] * Src.Stride);
            XDWORD *dst = (XDWORD *) ((XBYTE *) Dst.Ptr + i * Dst.Stride);
            for (XULONG j = 0; j < dwordCount; j++) {
                dst[j] = src[j];
            }
        }
        break;
    }
    }

    return TRUE;
}
