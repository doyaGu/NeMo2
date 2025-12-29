#include "VxMath.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
extern CRITICAL_SECTION g_CriticalSection;
#endif

#include "VxSIMD.h"
#include "VxBlitEngine.h"

//------------------------------------------------------------------------------
// Bit manipulation functions
//------------------------------------------------------------------------------

XULONG GetBitCount(XULONG dwMask) {
    if (dwMask == 0) return 0;

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    return __popcnt(dwMask);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(dwMask);
#else
    XULONG count = 0;

    // Skip trailing zeros until first set bit
    while ((dwMask & 1) == 0) {
        dwMask >>= 1;
    }

    // Count consecutive set bits
    while ((dwMask & 1) != 0) {
        dwMask >>= 1;
        ++count;
    }

    return count;
#endif
}

XULONG GetBitShift(XULONG dwMask) {
    if (dwMask == 0) return 0;

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    unsigned long index;
    _BitScanForward(&index, dwMask);
    return index;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(dwMask);
#else
    XULONG shift = 0;

    // Count trailing zeros until first set bit
    while ((dwMask & 1) == 0) {
        dwMask >>= 1;
        ++shift;
    }

    return shift;
#endif
}

void VxGetBitCounts(const VxImageDescEx &desc, XULONG &Rbits, XULONG &Gbits, XULONG &Bbits, XULONG &Abits) {
    Abits = GetBitCount(desc.AlphaMask);
    Rbits = GetBitCount(desc.RedMask);
    Gbits = GetBitCount(desc.GreenMask);
    Bbits = GetBitCount(desc.BlueMask);
}

void VxGetBitShifts(const VxImageDescEx &desc, XULONG &Rshift, XULONG &Gshift, XULONG &Bshift, XULONG &Ashift) {
    Ashift = GetBitShift(desc.AlphaMask);
    Rshift = GetBitShift(desc.RedMask);
    Gshift = GetBitShift(desc.GreenMask);
    Bshift = GetBitShift(desc.BlueMask);
}

//------------------------------------------------------------------------------
// Pixel format utilities
//------------------------------------------------------------------------------

VX_PIXELFORMAT VxImageDesc2PixelFormat(const VxImageDescEx &desc) {
    return VxBlitEngine::GetPixelFormat(desc);
}

void VxPixelFormat2ImageDesc(VX_PIXELFORMAT Pf, VxImageDescEx &desc) {
    VxBlitEngine::ConvertPixelFormat(Pf, desc);
}

const char *VxPixelFormat2String(VX_PIXELFORMAT Pf) {
    return VxBlitEngine::PixelFormat2String(Pf);
}

void VxBppToMask(VxImageDescEx &desc) {
    switch (desc.BitsPerPixel) {
    case 8:
        // RGB 332 format
        desc.AlphaMask = 0x00;
        desc.RedMask = 0xE0;
        desc.GreenMask = 0x1C;
        desc.BlueMask = 0x03;
        break;
    case 15:
    case 16:
        desc.AlphaMask = 0x0000;
        desc.BlueMask = 0x001F;
        desc.GreenMask = 0x03E0;
        desc.RedMask = 0x7C00;
        break;
    case 24:
        desc.AlphaMask = 0x00000000;
        desc.RedMask = 0x00FF0000;
        desc.GreenMask = 0x0000FF00;
        desc.BlueMask = 0x000000FF;
        break;
    case 32:
        desc.AlphaMask = 0xFF000000;
        desc.RedMask = 0x00FF0000;
        desc.GreenMask = 0x0000FF00;
        desc.BlueMask = 0x000000FF;
        break;
    default:
        break;
    }
}

//------------------------------------------------------------------------------
// Image blitting and conversion functions
//------------------------------------------------------------------------------

void VxDoBlit(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
#if defined(_WIN32)
    EnterCriticalSection(&g_CriticalSection);
    TheBlitter.DoBlit(src_desc, dst_desc);
    LeaveCriticalSection(&g_CriticalSection);
#else
    TheBlitter.DoBlit(src_desc, dst_desc);
#endif
}

void VxDoBlitUpsideDown(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
#if defined(_WIN32)
    EnterCriticalSection(&g_CriticalSection);
    TheBlitter.DoBlitUpsideDown(src_desc, dst_desc);
    LeaveCriticalSection(&g_CriticalSection);
#else
    TheBlitter.DoBlitUpsideDown(src_desc, dst_desc);
#endif
}

void VxDoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE AlphaValue) {
#if defined(_WIN32)
    EnterCriticalSection(&g_CriticalSection);
    TheBlitter.DoAlphaBlit(dst_desc, AlphaValue);
    LeaveCriticalSection(&g_CriticalSection);
#else
    TheBlitter.DoAlphaBlit(dst_desc, AlphaValue);
#endif
}

void VxDoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE *AlphaValues) {
#if defined(_WIN32)
    EnterCriticalSection(&g_CriticalSection);
    TheBlitter.DoAlphaBlit(dst_desc, AlphaValues);
    LeaveCriticalSection(&g_CriticalSection);
#else
    TheBlitter.DoAlphaBlit(dst_desc, AlphaValues);
#endif
}

void VxResizeImage32(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
#if defined(_WIN32)
    EnterCriticalSection(&g_CriticalSection);
    TheBlitter.ResizeImage(src_desc, dst_desc);
    LeaveCriticalSection(&g_CriticalSection);
#else
    TheBlitter.ResizeImage(src_desc, dst_desc);
#endif
}

void VxGenerateMipMap(const VxImageDescEx &src_desc, XBYTE *Buffer) {
    int Height = src_desc.Height;
    int BytesPerLine = src_desc.BytesPerLine;
    XBYTE *Image = src_desc.Image;

    int dstWidth = src_desc.Width >> 1;
    int dstHeight = Height >> 1;

#if defined(VX_SIMD_SSE2)
    const bool useSSE2 = VxGetSIMDFeatures().SSE2;
#else
    const bool useSSE2 = false;
#endif

    if (dstWidth == 0) {
        // Width is 0, only vertical averaging
        if (dstHeight) {
            XBYTE *dstPtr = Buffer;
            XBYTE *srcPtr = Image;
            int h = dstHeight;
#if defined(VX_SIMD_SSE2)
            if (useSSE2) {
                __m128i zero = _mm_setzero_si128();
                // Process 4 output pixels at a time if possible
                while (h >= 4) {
                    for (int i = 0; i < 4; i++) {
                        __m128i p0 = _mm_cvtsi32_si128(*(int *) srcPtr);
                        __m128i p1 = _mm_cvtsi32_si128(*(int *) (srcPtr + BytesPerLine));
                        __m128i lo0 = _mm_unpacklo_epi8(p0, zero);
                        __m128i lo1 = _mm_unpacklo_epi8(p1, zero);
                        __m128i sum = _mm_add_epi16(lo0, lo1);
                        sum = _mm_srli_epi16(sum, 1);
                        __m128i result = _mm_packus_epi16(sum, zero);
                        *(int *) dstPtr = _mm_cvtsi128_si32(result);
                        dstPtr += BytesPerLine;
                        srcPtr += BytesPerLine * 2;
                    }
                    h -= 4;
                }
            }
#endif
            while (h > 0) {
                XULONG p0 = *(XULONG *) srcPtr;
                XULONG p1 = *(XULONG *) (srcPtr + BytesPerLine);

                XULONG rb0 = p0 & 0x00FF00FF;
                XULONG rb1 = p1 & 0x00FF00FF;
                XULONG ag0 = (p0 & 0xFF00FF00) >> 8;
                XULONG ag1 = (p1 & 0xFF00FF00) >> 8;

                XULONG rbAvg = ((rb0 + rb1) >> 1) & 0x00FF00FF;
                XULONG agAvg = ((ag0 + ag1) << 7) & 0xFF00FF00;

                *(XULONG *) dstPtr = rbAvg | agAvg;

                dstPtr += BytesPerLine;
                srcPtr += BytesPerLine * 2;
                --h;
            }
        }
        return;
    }

    if (dstHeight == 0) {
        // Height is 0, only horizontal averaging
        XBYTE *dstPtr = Buffer;
        XULONG *srcPtr = (XULONG *) Image;

#if defined(VX_SIMD_SSE2)
        if (useSSE2) {
            // SSE2 path - process 2 output pixels at a time (4 source pixels)
            __m128i zero = _mm_setzero_si128();
            int w = dstWidth;
            while (w >= 2) {
                // Load 4 source pixels (16 bytes)
                __m128i src = _mm_loadu_si128((const __m128i *) srcPtr);
                // Unpack bytes to 16-bit
                __m128i lo = _mm_unpacklo_epi8(src, zero);
                __m128i hi = _mm_unpackhi_epi8(src, zero);

                // Extract odd/even pixels and average
                __m128i p0 = _mm_unpacklo_epi64(lo, hi);
                __m128i p1 = _mm_unpackhi_epi64(lo, hi);
                __m128i sum = _mm_add_epi16(p0, p1);
                sum = _mm_srli_epi16(sum, 1);

                // Pack back to bytes
                __m128i result = _mm_packus_epi16(sum, zero);
                _mm_storel_epi64((__m128i *) dstPtr, result);

                srcPtr += 4;
                dstPtr += 8;
                w -= 2;
            }
            // Handle remaining pixel
            while (w > 0) {
                XULONG p0 = srcPtr[0];
                XULONG p1 = srcPtr[1];
                XULONG rb0 = p0 & 0x00FF00FF;
                XULONG rb1 = p1 & 0x00FF00FF;
                XULONG ag0 = (p0 & 0xFF00FF00) >> 8;
                XULONG ag1 = (p1 & 0xFF00FF00) >> 8;
                XULONG rbAvg = ((rb0 + rb1) >> 1) & 0x00FF00FF;
                XULONG agAvg = ((ag0 + ag1) << 7) & 0xFF00FF00;
                *(XULONG *) dstPtr = rbAvg | agAvg;
                srcPtr += 2;
                dstPtr += 4;
                --w;
            }
        } else
#endif
        {
            // Plain C path
            int w = dstWidth;
            do {
                XULONG p0 = srcPtr[0];
                XULONG p1 = srcPtr[1];

                XULONG rb0 = p0 & 0x00FF00FF;
                XULONG rb1 = p1 & 0x00FF00FF;
                XULONG ag0 = (p0 & 0xFF00FF00) >> 8;
                XULONG ag1 = (p1 & 0xFF00FF00) >> 8;

                XULONG rbAvg = ((rb0 + rb1) >> 1) & 0x00FF00FF;
                XULONG agAvg = ((ag0 + ag1) << 7) & 0xFF00FF00;

                *(XULONG *) dstPtr = rbAvg | agAvg;
                srcPtr += 2;
                dstPtr += 4;
                --w;
            } while (w);
        }
        return;
    }

    // Both width and height > 0, 2D averaging
    XBYTE *dstPtr = Buffer;
    XULONG *srcPtr = (XULONG *) Image;

#if defined(VX_SIMD_SSE2)
    if (useSSE2) {
        // SSE2 path - process 2 output pixels at a time
        __m128i zero = _mm_setzero_si128();
        int h = dstHeight;
        do {
            XULONG *rowStart = srcPtr;
            int w = dstWidth;
            while (w >= 2) {
                // Load 4 source pixels from current row
                __m128i src0 = _mm_loadu_si128((const __m128i *) srcPtr);
                // Load 4 source pixels from next row
                __m128i src1 = _mm_loadu_si128((const __m128i *) ((XBYTE *) srcPtr + BytesPerLine));

                // Unpack to 16-bit
                __m128i lo0 = _mm_unpacklo_epi8(src0, zero);
                __m128i hi0 = _mm_unpackhi_epi8(src0, zero);
                __m128i lo1 = _mm_unpacklo_epi8(src1, zero);
                __m128i hi1 = _mm_unpackhi_epi8(src1, zero);

                // Sum all 4 pixels for each output pixel
                __m128i p0 = _mm_unpacklo_epi64(lo0, hi0);
                __m128i p1 = _mm_unpackhi_epi64(lo0, hi0);
                __m128i p2 = _mm_unpacklo_epi64(lo1, hi1);
                __m128i p3 = _mm_unpackhi_epi64(lo1, hi1);

                __m128i sum = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(p0, p1), p2), p3);
                sum = _mm_srli_epi16(sum, 2);

                // Pack back to bytes
                __m128i result = _mm_packus_epi16(sum, zero);
                _mm_storel_epi64((__m128i *) dstPtr, result);

                srcPtr += 4;
                dstPtr += 8;
                w -= 2;
            }
            // Handle remaining pixel
            while (w > 0) {
                XULONG p0 = srcPtr[0];
                XULONG p1 = srcPtr[1];
                XULONG p2 = *(XULONG *) ((XBYTE *) srcPtr + BytesPerLine);
                XULONG p3 = *(XULONG *) ((XBYTE *) srcPtr + BytesPerLine + 4);

                XULONG rb = (p0 & 0x00FF00FF) + (p1 & 0x00FF00FF) +
                    (p2 & 0x00FF00FF) + (p3 & 0x00FF00FF);
                XULONG ag = ((p0 & 0xFF00FF00) >> 8) + ((p1 & 0xFF00FF00) >> 8) +
                    ((p2 & 0xFF00FF00) >> 8) + ((p3 & 0xFF00FF00) >> 8);

                XULONG rbAvg = (rb >> 2) & 0x00FF00FF;
                XULONG agAvg = ((ag >> 2) << 8) & 0xFF00FF00;

                *(XULONG *) dstPtr = rbAvg | agAvg;
                srcPtr += 2;
                dstPtr += 4;
                --w;
            }
            srcPtr = (XULONG *) ((XBYTE *) rowStart + BytesPerLine * 2);
            --h;
        } while (h);
    } else
#endif
    {
        // Plain C path
        int h = dstHeight;
        do {
            XULONG *rowStart = srcPtr;
            int w = dstWidth;
            do {
                XULONG p0 = srcPtr[0];
                XULONG p1 = srcPtr[1];
                XULONG p2 = *(XULONG *) ((XBYTE *) srcPtr + BytesPerLine);
                XULONG p3 = *(XULONG *) ((XBYTE *) srcPtr + BytesPerLine + 4);

                XULONG rb = (p0 & 0x00FF00FF) + (p1 & 0x00FF00FF) +
                    (p2 & 0x00FF00FF) + (p3 & 0x00FF00FF);
                XULONG ag = ((p0 & 0xFF00FF00) >> 8) + ((p1 & 0xFF00FF00) >> 8) +
                    ((p2 & 0xFF00FF00) >> 8) + ((p3 & 0xFF00FF00) >> 8);

                XULONG rbAvg = (rb >> 2) & 0x00FF00FF;
                XULONG agAvg = ((ag >> 2) << 8) & 0xFF00FF00;

                *(XULONG *) dstPtr = rbAvg | agAvg;
                srcPtr += 2;
                dstPtr += 4;
                --w;
            } while (w);
            srcPtr = (XULONG *) ((XBYTE *) rowStart + BytesPerLine * 2);
            --h;
        } while (h);
    }
}

//------------------------------------------------------------------------------
// Normal and bump map generation
//------------------------------------------------------------------------------

// Helper function to pack normal vector into ARGB pixel
// Original formula: ((normal * 127.0 + 128.0) for each component, with alpha derived from height
static XULONG PackNormalToPixel(const float *normal, float height) {
    // normal[0] = nx, normal[1] = ny, normal[2] = nz
    int nx_color = (int) (normal[0] * 127.0f + 128.0f);
    int ny_color = (int) (normal[1] * 127.0f + 128.0f);
    int nz_color = (int) (normal[2] * 127.0f + 128.0f);
    int alpha = (int) (height * -255.0f);

    // Pack as ARGB: (alpha << 24) | (red << 16) | (green << 8) | blue
    // Original packs: ((nx - alpha<<8) << 8 + ny) << 8 + nz
    return (XULONG) (((((nx_color - (alpha << 8)) << 8) + ny_color) << 8) + nz_color);
}

#if defined(VX_SIMD_SSE2)
// SSE version: Pack 4 normals to pixels at once
static void PackNormalToPixelSSE(const __m128 &dx4, const __m128 &dy4, const __m128 &invLen4,
                                 const __m128 &lum4, XULONG *outPixels) {
    __m128 scale127 = _mm_set1_ps(127.0f);
    __m128 offset128 = _mm_set1_ps(128.0f);
    __m128 scaleNeg255 = _mm_set1_ps(-255.0f);

    // Calculate normal components
    __m128 nx = _mm_mul_ps(dx4, invLen4);
    __m128 ny = _mm_mul_ps(dy4, invLen4);
    __m128 nz = invLen4;

    // Convert to color space: normal * 127 + 128
    __m128 nx_color = _mm_add_ps(_mm_mul_ps(nx, scale127), offset128);
    __m128 ny_color = _mm_add_ps(_mm_mul_ps(ny, scale127), offset128);
    __m128 nz_color = _mm_add_ps(_mm_mul_ps(nz, scale127), offset128);
    __m128 alpha = _mm_mul_ps(lum4, scaleNeg255);

    // Convert to integers
    __m128i nx_i = _mm_cvttps_epi32(nx_color);
    __m128i ny_i = _mm_cvttps_epi32(ny_color);
    __m128i nz_i = _mm_cvttps_epi32(nz_color);
    __m128i alpha_i = _mm_cvttps_epi32(alpha);

    // Pack: ((nx - alpha<<8) << 8 + ny) << 8 + nz
    // Extract and pack each pixel
    alignas(16) int nx_arr[4], ny_arr[4], nz_arr[4], alpha_arr[4];
    _mm_store_si128((__m128i *) nx_arr, nx_i);
    _mm_store_si128((__m128i *) ny_arr, ny_i);
    _mm_store_si128((__m128i *) nz_arr, nz_i);
    _mm_store_si128((__m128i *) alpha_arr, alpha_i);

    for (int i = 0; i < 4; i++) {
        outPixels[i] = (XULONG) (((((nx_arr[i] - (alpha_arr[i] << 8)) << 8) + ny_arr[i]) << 8) + nz_arr[i]);
    }
}

// SSE version: Calculate luminance for 4 pixels (returns float values)
static __m128 CalculateLuminanceSSE4(const XULONG *pixels, const __m128 &scale) {
    // Extract R, G, B from each pixel and sum them
    alignas(16) float lum[4];
    for (int i = 0; i < 4; i++) {
        XULONG p = pixels[i];
        lum[i] = (float) ((p & 0xFF) + ((p >> 8) & 0xFF) + ((p >> 16) & 0xFF));
    }
    return _mm_mul_ps(_mm_load_ps(lum), scale);
}
#endif

XBOOL VxConvertToNormalMap(const VxImageDescEx &image, XULONG ColorMask) {
    if (image.BitsPerPixel != 32) return FALSE;
    if (image.Width == 0 || image.Height == 0) return FALSE;
    if (image.Image == nullptr) return FALSE;

    XBYTE *Image = image.Image;
    int Width = image.Width;
    int Height = image.Height;

#if defined(VX_SIMD_SSE2)
    const bool useSSE2 = VxGetSIMDFeatures().SSE2;
#else
    const bool useSSE2 = false;
#endif

    if (ColorMask == 0xFFFFFFFF) {
        // Use luminance from RGB
        const float scale = 0.0013071896f; // 1.0 / 765.0 (255*3)

#if defined(VX_SIMD_SSE2)
        if (useSSE2 && Width >= 8) {
            __m128 scaleVec = _mm_set1_ps(scale);
            __m128 one = _mm_set1_ps(1.0f);

            for (int y = 0; y < Height - 1; y++) {
                XULONG *rowCurrent = (XULONG *) (Image + y * image.BytesPerLine);
                XULONG *rowBelow = (XULONG *) (Image + (y + 1) * image.BytesPerLine);

                int x = 0;
                // Process 4 pixels at a time
                for (; x + 4 < Width; x += 4) {
                    XULONG *pCurrent = rowCurrent + x;

                    // Load 4 current pixels
                    alignas(16) XULONG current4[4] = {pCurrent[0], pCurrent[1], pCurrent[2], pCurrent[3]};
                    // Load 4 right pixels
                    alignas(16) XULONG right4[4] = {pCurrent[1], pCurrent[2], pCurrent[3], pCurrent[4]};
                    // Load 4 below pixels
                    alignas(16) XULONG below4[4] = {rowBelow[x], rowBelow[x + 1], rowBelow[x + 2], rowBelow[x + 3]};

                    // Calculate luminances
                    __m128 lum0 = CalculateLuminanceSSE4(current4, scaleVec);
                    __m128 lum1 = CalculateLuminanceSSE4(right4, scaleVec);
                    __m128 lum2 = CalculateLuminanceSSE4(below4, scaleVec);

                    // Calculate derivatives
                    __m128 dx = _mm_sub_ps(lum1, lum0);
                    __m128 dy = _mm_sub_ps(lum2, lum0);

                    // Calculate length^2 = dx^2 + dy^2 + 1
                    __m128 lenSq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)), one);

                    // Calculate inverse length using fast reciprocal sqrt
                    __m128 invLen = VxSIMDReciprocalSqrtAccurate(lenSq);

                    // Pack and store 4 output pixels
                    alignas(16) XULONG outPixels[4];
                    PackNormalToPixelSSE(dx, dy, invLen, lum0, outPixels);
                    pCurrent[0] = outPixels[0];
                    pCurrent[1] = outPixels[1];
                    pCurrent[2] = outPixels[2];
                    pCurrent[3] = outPixels[3];
                }

                // Handle remaining pixels with scalar code
                for (; x < Width; x++) {
                    XULONG *pCurrent = rowCurrent + x;
                    XULONG *pRight = (x + 1 < Width) ? (pCurrent + 1) : pCurrent;
                    XULONG *pBelow = rowBelow + x;

                    XULONG p0 = *pCurrent;
                    float lum0f = ((p0 & 0xFF) + ((p0 >> 8) & 0xFF) + ((p0 >> 16) & 0xFF)) * scale;

                    XULONG p1 = *pRight;
                    float lum1f = ((p1 & 0xFF) + ((p1 >> 8) & 0xFF) + ((p1 >> 16) & 0xFF)) * scale;

                    XULONG p2 = *pBelow;
                    float lum2f = ((p2 & 0xFF) + ((p2 >> 8) & 0xFF) + ((p2 >> 16) & 0xFF)) * scale;

                    float dxf = lum1f - lum0f;
                    float dyf = lum2f - lum0f;
                    float invLenf = 1.0f / sqrtf(dxf * dxf + dyf * dyf + 1.0f);

                    float normal[3] = {dxf * invLenf, dyf * invLenf, invLenf};
                    *pCurrent = PackNormalToPixel(normal, lum0f);
                }
            }
        } else
#endif
        {
            // Scalar fallback
            for (int y = 0; y < Height - 1; y++) {
                for (int x = 0; x < Width; x++) {
                    XULONG *pCurrent = (XULONG *) (Image + y * image.BytesPerLine + x * 4);
                    XULONG *pRight = (XULONG *) (Image + y * image.BytesPerLine + (x + 1) * 4);
                    XULONG *pBelow = (XULONG *) (Image + (y + 1) * image.BytesPerLine + x * 4);

                    // Calculate luminance for current pixel (R + G + B)
                    XULONG p0 = *pCurrent;
                    float lum0 = ((p0 & 0xFF) + ((p0 >> 8) & 0xFF) + ((p0 >> 16) & 0xFF)) * scale;

                    // Calculate luminance for right pixel
                    XULONG p1 = (x + 1 < Width) ? *pRight : p0;
                    float lum1 = ((p1 & 0xFF) + ((p1 >> 8) & 0xFF) + ((p1 >> 16) & 0xFF)) * scale;

                    // Calculate luminance for below pixel
                    XULONG p2 = *pBelow;
                    float lum2 = ((p2 & 0xFF) + ((p2 >> 8) & 0xFF) + ((p2 >> 16) & 0xFF)) * scale;

                    // Calculate derivatives
                    float dx = lum1 - lum0;
                    float dy = lum2 - lum0;

                    // Calculate normal vector and normalize
                    float invLen = 1.0f / sqrtf(dx * dx + dy * dy + 1.0f);
                    float normal[3];
                    normal[0] = dx * invLen;
                    normal[1] = dy * invLen;
                    normal[2] = invLen;

                    *pCurrent = PackNormalToPixel(normal, lum0);
                }
            }
        }
    } else {
        // Use specified color mask
        XBYTE BitShift = (XBYTE) GetBitShift(ColorMask);
        const float scale = 0.003921568859f; // 1.0 / 255.0

        for (int y = 0; y < Height - 1; y++) {
            for (int x = 0; x < Width; x++) {
                XULONG *pCurrent = (XULONG *) (Image + y * image.BytesPerLine + x * 4);
                XULONG *pRight = (XULONG *) (Image + y * image.BytesPerLine + (x + 1) * 4);
                XULONG *pBelow = (XULONG *) (Image + (y + 1) * image.BytesPerLine + x * 4);

                // Extract masked value for current pixel
                XULONG v0 = ((*pCurrent) & ColorMask) >> BitShift;
                float h0 = (float) v0 * scale;

                // Extract masked value for right pixel
                XULONG v1 = (x + 1 < Width) ? (((*pRight) & ColorMask) >> BitShift) : v0;
                float h1 = (float) v1 * scale;

                // Extract masked value for below pixel
                XULONG v2 = ((*pBelow) & ColorMask) >> BitShift;
                float h2 = (float) v2 * scale;

                // Calculate derivatives
                float dx = h1 - h0;
                float dy = h2 - h0;

                // Calculate normal vector and normalize
                float invLen = 1.0f / sqrtf(dx * dx + dy * dy + 1.0f);
                float normal[3];
                normal[0] = dx * invLen;
                normal[1] = dy * invLen;
                normal[2] = invLen;

                *pCurrent = PackNormalToPixel(normal, h0);
            }
        }
    }

    // Copy second-to-last row to last row
    memcpy(
        Image + (Height - 1) * image.BytesPerLine,
        Image + (Height - 2) * image.BytesPerLine,
        image.BytesPerLine
    );

    return TRUE;
}

#if defined(VX_SIMD_SSE2)
static inline int SseLuminance(XULONG pixel) {
    return (int) ((pixel & 0xFF) + ((pixel >> 8) & 0xFF) + ((pixel >> 16) & 0xFF));
}
#endif

XBOOL VxConvertToBumpMap(const VxImageDescEx &image) {
    if (image.BitsPerPixel != 32) return FALSE;

    // Allocate temporary copy of the image
    int imageSize = image.BytesPerLine * image.Height;
    XBYTE *tempImage = (XBYTE *) operator new(imageSize);
    memcpy(tempImage, image.Image, imageSize);

    XBYTE *srcPtr = tempImage;
    XBYTE *dstPtr = image.Image;
    int Width = image.Width;
    int Height = image.Height;
    int BytesPerLine = image.BytesPerLine;

    // Calculate pointer to last row of source (for wrap-around)
    XBYTE *lastRowPtr = tempImage + BytesPerLine * (Height - 1);

    for (int y = 0; (unsigned int) y < (unsigned int) Height; y++) {
        // Get pointer to row above (wrap to last row if at top)
        XBYTE *abovePtr;
        if (y == 0) {
            abovePtr = lastRowPtr;
        } else {
            abovePtr = srcPtr - BytesPerLine;
        }

        // Get pointer to row below (use current row if at bottom)
        XBYTE *belowPtr;
        if (y != Height - 1) {
            belowPtr = srcPtr + BytesPerLine;
        } else {
            belowPtr = srcPtr; // At last row, reuse current
        }

        XBYTE *currentPtr = srcPtr + 4; // Point to second pixel in row

        // Calculate left pixel luminance (last pixel of row for wrap)
        int leftPixel = *(int *) (srcPtr + (Width - 1) * 4);
        int leftLum = (leftPixel & 0xFF) + ((leftPixel >> 8) & 0xFF) + ((leftPixel >> 16) & 0xFF);

        // Calculate below-left luminance for first pixel
        int belowVal = *(int *) belowPtr;
        int belowLum = (belowVal & 0xFF) + ((belowVal >> 8) & 0xFF) + ((belowVal >> 16) & 0xFF);

        // Calculate above luminance for first pixel
        int aboveVal = *(int *) abovePtr;
        int aboveLum = (aboveVal & 0xFF) + ((aboveVal >> 8) & 0xFF) + ((aboveVal >> 16) & 0xFF);

        // Calculate current pixel luminance
        int currVal = *(int *) srcPtr;
        int currLum = (currVal & 0xFF) + ((currVal >> 8) & 0xFF) + ((currVal >> 16) & 0xFF);

        // Calculate right pixel luminance
        int rightVal = *(int *) (belowPtr + 4);
        int rightLum = (rightVal & 0xFF) + ((rightVal >> 8) & 0xFF) + ((rightVal >> 16) & 0xFF);

        // First pixel of row
        *dstPtr = (currLum <= 1) ? 127 : 63;
        dstPtr[1] = (XBYTE) (aboveLum - belowLum + leftLum - rightLum);
        dstPtr[2] = (XBYTE) (leftLum - currLum);
        dstPtr[3] = 0;
        dstPtr += 4;

        belowPtr += 4;
        abovePtr += 4;

#if defined(VX_SIMD_SSE2)
        // SSE2 optimized middle pixels - process 4 pixels at a time
        if (VxGetSIMDFeatures().SSE2 && Width >= 6) {
            __m128i zero = _mm_setzero_si128();

            unsigned int x = 1;
            // Process 4 pixels at a time when possible
            while (x + 4 < (unsigned int) (Width - 1)) {
                // Load 4 current pixels
                __m128i curr4 = _mm_loadu_si128((const __m128i *) currentPtr);
                __m128i below4 = _mm_loadu_si128((const __m128i *) belowPtr);
                __m128i above4 = _mm_loadu_si128((const __m128i *) abovePtr);
                __m128i left4 = _mm_loadu_si128((const __m128i *) (currentPtr - 4));
                __m128i right4 = _mm_loadu_si128((const __m128i *) (currentPtr + 4));

                // Process each of 4 pixels
                for (int i = 0; i < 4; i++) {
                    XULONG currPx = _mm_cvtsi128_si32(curr4);
                    XULONG belowPx = _mm_cvtsi128_si32(below4);
                    XULONG abovePx = _mm_cvtsi128_si32(above4);
                    XULONG leftPx = _mm_cvtsi128_si32(left4);
                    XULONG rightPx = _mm_cvtsi128_si32(right4);

                    int cLum = SseLuminance(currPx);
                    int bLum = SseLuminance(belowPx);
                    int aLum = SseLuminance(abovePx);
                    int lLum = SseLuminance(leftPx);
                    int rLum = SseLuminance(rightPx);

                    *dstPtr = (cLum <= 1) ? 127 : 63;
                    dstPtr[1] = (XBYTE) (aLum - bLum);
                    dstPtr[2] = (XBYTE) (lLum - rLum);
                    dstPtr[3] = 0;
                    dstPtr += 4;

                    // Shift to next pixel
                    curr4 = _mm_srli_si128(curr4, 4);
                    below4 = _mm_srli_si128(below4, 4);
                    above4 = _mm_srli_si128(above4, 4);
                    left4 = _mm_srli_si128(left4, 4);
                    right4 = _mm_srli_si128(right4, 4);
                }

                currentPtr += 16;
                belowPtr += 16;
                abovePtr += 16;
                x += 4;
            }

            // Handle remaining middle pixels with scalar code
            for (; x < (unsigned int) (Width - 1); x++) {
                leftPixel = *(int *) (currentPtr - 4);
                leftLum = (leftPixel & 0xFF) + ((leftPixel >> 8) & 0xFF) + ((leftPixel >> 16) & 0xFF);

                int nextRightVal = *(int *) (currentPtr + 4);
                int nextRightLum = (nextRightVal & 0xFF) + ((nextRightVal >> 8) & 0xFF) + ((nextRightVal >> 16) & 0xFF);

                currVal = *(int *) currentPtr;
                currLum = (currVal & 0xFF) + ((currVal >> 8) & 0xFF) + ((currVal >> 16) & 0xFF);

                belowVal = *(int *) belowPtr;
                belowLum = (belowVal & 0xFF) + ((belowVal >> 8) & 0xFF) + ((belowVal >> 16) & 0xFF);

                aboveVal = *(int *) abovePtr;
                aboveLum = (aboveVal & 0xFF) + ((aboveVal >> 8) & 0xFF) + ((aboveVal >> 16) & 0xFF);

                *dstPtr = (currLum <= 1) ? 127 : 63;
                dstPtr[1] = (XBYTE) (aboveLum - belowLum);
                dstPtr[2] = (XBYTE) (leftLum - nextRightLum);
                dstPtr[3] = 0;

                currentPtr += 4;
                belowPtr += 4;
                abovePtr += 4;
                dstPtr += 4;
            }
        } else
#endif
        {
            // Scalar middle pixels
            for (unsigned int x = 1; x < (unsigned int) (Width - 1); x++) {
                // Get values
                leftPixel = *(int *) (currentPtr - 4);
                leftLum = (leftPixel & 0xFF) + ((leftPixel >> 8) & 0xFF) + ((leftPixel >> 16) & 0xFF);

                int nextRightVal = *(int *) (currentPtr + 4);
                int nextRightLum = (nextRightVal & 0xFF) + ((nextRightVal >> 8) & 0xFF) + ((nextRightVal >> 16) & 0xFF);

                currVal = *(int *) currentPtr;
                currLum = (currVal & 0xFF) + ((currVal >> 8) & 0xFF) + ((currVal >> 16) & 0xFF);

                belowVal = *(int *) belowPtr;
                belowLum = (belowVal & 0xFF) + ((belowVal >> 8) & 0xFF) + ((belowVal >> 16) & 0xFF);

                aboveVal = *(int *) abovePtr;
                aboveLum = (aboveVal & 0xFF) + ((aboveVal >> 8) & 0xFF) + ((aboveVal >> 16) & 0xFF);

                *dstPtr = (currLum <= 1) ? 127 : 63;
                dstPtr[1] = (XBYTE) (aboveLum - belowLum);
                dstPtr[2] = (XBYTE) (leftLum - nextRightLum);
                dstPtr[3] = 0;

                currentPtr += 4;
                belowPtr += 4;
                abovePtr += 4;
                dstPtr += 4;
            }
        }

        // Last pixel of row
        leftPixel = *(int *) (srcPtr + (Width - 2) * 4);
        leftLum = (leftPixel & 0xFF) + ((leftPixel >> 8) & 0xFF) + ((leftPixel >> 16) & 0xFF);

        // Right pixel wraps to first pixel
        int firstVal = *(int *) srcPtr;
        int firstLum = (firstVal & 0xFF) + ((firstVal >> 8) & 0xFF) + ((firstVal >> 16) & 0xFF);

        currVal = *(int *) currentPtr;
        currLum = (currVal & 0xFF) + ((currVal >> 8) & 0xFF) + ((currVal >> 16) & 0xFF);

        belowVal = *(int *) belowPtr;
        belowLum = (belowVal & 0xFF) + ((belowVal >> 8) & 0xFF) + ((belowVal >> 16) & 0xFF);

        aboveVal = *(int *) abovePtr;
        aboveLum = (aboveVal & 0xFF) + ((aboveVal >> 8) & 0xFF) + ((aboveVal >> 16) & 0xFF);

        *dstPtr = (currLum <= 1) ? 127 : 63;
        dstPtr[1] = (XBYTE) (aboveLum - belowLum);
        dstPtr[2] = (XBYTE) (leftLum - firstLum);
        dstPtr[3] = 0;
        dstPtr += 4;

        srcPtr += BytesPerLine;
    }

    operator delete(tempImage);
    return TRUE;
}

//------------------------------------------------------------------------------
// Color Quantization
//------------------------------------------------------------------------------

// Global quantization sampling factor (1 = best quality)
static int QuantizationSamplingFactor = 15;

int GetQuantizationSamplingFactor() {
    return QuantizationSamplingFactor;
}

void SetQuantizationSamplingFactor(int factor) {
    QuantizationSamplingFactor = factor;
}
