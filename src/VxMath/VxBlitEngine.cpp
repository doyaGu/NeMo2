#include "VxBlitEngine.h"

#include <stdlib.h>
#include <algorithm>

#include "VxMath.h"
#include "VxSIMD.h"
#include "NeuQuant.h"

//------------------------------------------------------------------------------
// Global Blitter Instance
//------------------------------------------------------------------------------

VxBlitEngine TheBlitter;

//------------------------------------------------------------------------------
// Pixel Format Tables
//------------------------------------------------------------------------------

// Standard pixel format definitions
struct PixelFormatDef {
    XWORD BitsPerPixel;
    XDWORD RedMask;
    XDWORD GreenMask;
    XDWORD BlueMask;
    XDWORD AlphaMask;
    const char *Description;
};

static const PixelFormatDef s_PixelFormats[] = {
    {0, 0, 0, 0, 0, "Unknown"},
    {32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000, "32 bits ARGB 8888"},
    {32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000, "32 bits RGB  888"},
    {24, 0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000, "24 bits RGB  888"},
    {16, 0xF800, 0x07E0, 0x001F, 0x0000, "16 bits RGB  565"},
    {16, 0x7C00, 0x03E0, 0x001F, 0x0000, "16 bits RGB  555"},
    {16, 0x7C00, 0x03E0, 0x001F, 0x8000, "16 bits ARGB 1555"},
    {16, 0x0F00, 0x00F0, 0x000F, 0xF000, "16 bits ARGB 4444"},
    {8, 0xE0, 0x1C, 0x03, 0x00, "8 bits RGB  332"},
    {8, 0x30, 0x0C, 0x03, 0xC0, "8 bits ARGB 2222"},
    {32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000, "32 bits ABGR 8888"},
    {32, 0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF, "32 bits RGBA 8888"},
    {32, 0x0000FF00, 0x00FF0000, 0xFF000000, 0x000000FF, "32 bits BGRA 8888"},
    {32, 0x0000FF00, 0x00FF0000, 0xFF000000, 0x00000000, "32 bits BGR  888"},
    {24, 0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000, "24 bits BGR  888"},
    {16, 0x001F, 0x07E0, 0xF800, 0x0000, "16 bits BGR  565"},
    {16, 0x001F, 0x03E0, 0x7C00, 0x0000, "16 bits BGR  555"},
    {16, 0x001F, 0x03E0, 0x7C00, 0x8000, "16 bits ABGR 1555"},
    {16, 0x000F, 0x00F0, 0x0F00, 0xF000, "16 bits ABGR 4444"},
    // DXT formats
    {4, 0, 0, 0, 0, "Compressed DXT1"},
    {8, 0, 0, 0, 0, "Compressed DXT2"},
    {8, 0, 0, 0, 0, "Compressed DXT3"},
    {8, 0, 0, 0, 0, "Compressed DXT4"},
    {8, 0, 0, 0, 0, "Compressed DXT5"},
    // Bump map formats
    {16, 0x00FF, 0xFF00, 0x0000, 0x0000, "V8U8 BumpMap"},
    {32, 0xFFFF, 0xFFFF0000, 0x0000, 0x0000, "V16U16 BumpMap"},
    {16, 0x001F, 0x03E0, 0x7C00, 0x0000, "L6V5U5 BumpMap"},
    {32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000, "L8V8U8 BumpMap"}
};

static const int NUM_PIXEL_FORMATS = sizeof(s_PixelFormats) / sizeof(s_PixelFormats[0]);

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

// Get bit shift for a mask (position of least significant bit)
static inline XDWORD GetBitShiftLocal(XDWORD dwMask) {
    if (!dwMask) return 0;
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    unsigned long index;
    _BitScanForward(&index, dwMask);
    return index;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(dwMask);
#else
    XDWORD shift = 0;
    while ((dwMask & 1) == 0) {
        dwMask >>= 1;
        ++shift;
    }
    return shift;
#endif
}

// Get bit count for a mask (number of set bits)
static inline XDWORD GetBitCountLocal(XDWORD dwMask) {
    if (!dwMask) return 0;
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    return __popcnt(dwMask);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(dwMask);
#else
    XDWORD count = 0;
    while (dwMask) {
        count += dwMask & 1;
        dwMask >>= 1;
    }
    return count;
#endif
}

// Read a pixel of variable size
static inline XDWORD ReadPixel(const XBYTE *ptr, int bpp) {
    switch (bpp) {
    case 1: return *ptr;
    case 2: return *(const XWORD *)ptr;
    case 3: return ptr[0] | (ptr[1] << 8) | (ptr[2] << 16);
    case 4: return *(const XDWORD *)ptr;
    default: return 0;
    }
}

// Write a pixel of variable size
static inline void WritePixel(XBYTE *ptr, int bpp, XDWORD pixel) {
    switch (bpp) {
    case 1: *ptr = (XBYTE)pixel; break;
    case 2: *(XWORD *)ptr = (XWORD)pixel; break;
    case 3:
        ptr[0] = (XBYTE)pixel;
        ptr[1] = (XBYTE)(pixel >> 8);
        ptr[2] = (XBYTE)(pixel >> 16);
        break;
    case 4: *(XDWORD *)ptr = pixel; break;
    }
}

//------------------------------------------------------------------------------
// Generic Line Blitting Functions
//------------------------------------------------------------------------------

// Same format copy - just memcpy
static void CopyLineSame(const VxBlitInfo *info) {
    memcpy(info->dstLine, info->srcLine, info->copyBytes);
}

// Helper to convert a component from one bit width to another
static inline XDWORD ConvertComponent(XDWORD value, int srcBits, int dstBits, int dstShift) {
    if (srcBits == 0) return 0;
    if (dstBits == 0) return 0;
    
    // Scale the value from srcBits to dstBits
    if (srcBits > dstBits) {
        // Truncate: take the MSBs (shift right)
        value >>= (srcBits - dstBits);
    } else if (srcBits < dstBits) {
        // Expand: replicate bits to fill
        // E.g., 5-bit 0x1F to 8-bit should give 0xFF
        int shift = dstBits - srcBits;
        XDWORD expanded = value << shift;
        // Replicate lower bits for proper expansion
        if (srcBits > 0 && shift > 0) {
            XDWORD rep = value >> (srcBits - std::min(srcBits, shift));
            expanded |= rep;
        }
        value = expanded;
    }
    // Position the value at the destination shift
    return value << dstShift;
}

// Generic pixel conversion with proper bit width handling
template<int SrcBpp, int DstBpp>
static void CopyLineGeneric(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XBYTE *dst = info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        XDWORD pixel = ReadPixel(src, SrcBpp);

        // Extract components (shift to bit 0)
        XDWORD r = info->srcRedMask ? ((pixel & info->srcRedMask) >> info->redShiftSrc) : 0;
        XDWORD g = info->srcGreenMask ? ((pixel & info->srcGreenMask) >> info->greenShiftSrc) : 0;
        XDWORD b = info->srcBlueMask ? ((pixel & info->srcBlueMask) >> info->blueShiftSrc) : 0;
        XDWORD a = info->srcAlphaMask ? ((pixel & info->srcAlphaMask) >> info->alphaShiftSrc) : 
                   (info->alphaBitsDst > 0 ? ((1u << info->alphaBitsDst) - 1) : 0); // Default to full alpha

        // Convert bit widths and position
        XDWORD dstPixel = ConvertComponent(r, info->redBitsSrc, info->redBitsDst, info->redShiftDst) |
                          ConvertComponent(g, info->greenBitsSrc, info->greenBitsDst, info->greenShiftDst) |
                          ConvertComponent(b, info->blueBitsSrc, info->blueBitsDst, info->blueShiftDst) |
                          ConvertComponent(a, info->alphaBitsSrc, info->alphaBitsDst, info->alphaShiftDst);

        WritePixel(dst, DstBpp, dstPixel);

        src += SrcBpp;
        dst += DstBpp;
    }
}

// Explicit instantiations
static void CopyLineGeneric_8_8(const VxBlitInfo *info) { CopyLineGeneric<1, 1>(info); }
static void CopyLineGeneric_8_16(const VxBlitInfo *info) { CopyLineGeneric<1, 2>(info); }
static void CopyLineGeneric_8_24(const VxBlitInfo *info) { CopyLineGeneric<1, 3>(info); }
static void CopyLineGeneric_8_32(const VxBlitInfo *info) { CopyLineGeneric<1, 4>(info); }
static void CopyLineGeneric_16_8(const VxBlitInfo *info) { CopyLineGeneric<2, 1>(info); }
static void CopyLineGeneric_16_16(const VxBlitInfo *info) { CopyLineGeneric<2, 2>(info); }
static void CopyLineGeneric_16_24(const VxBlitInfo *info) { CopyLineGeneric<2, 3>(info); }
static void CopyLineGeneric_16_32(const VxBlitInfo *info) { CopyLineGeneric<2, 4>(info); }
static void CopyLineGeneric_24_8(const VxBlitInfo *info) { CopyLineGeneric<3, 1>(info); }
static void CopyLineGeneric_24_16(const VxBlitInfo *info) { CopyLineGeneric<3, 2>(info); }
static void CopyLineGeneric_24_24(const VxBlitInfo *info) { CopyLineGeneric<3, 3>(info); }
static void CopyLineGeneric_24_32(const VxBlitInfo *info) { CopyLineGeneric<3, 4>(info); }
static void CopyLineGeneric_32_8(const VxBlitInfo *info) { CopyLineGeneric<4, 1>(info); }
static void CopyLineGeneric_32_16(const VxBlitInfo *info) { CopyLineGeneric<4, 2>(info); }
static void CopyLineGeneric_32_24(const VxBlitInfo *info) { CopyLineGeneric<4, 3>(info); }
static void CopyLineGeneric_32_32(const VxBlitInfo *info) { CopyLineGeneric<4, 4>(info); }

//------------------------------------------------------------------------------
// Optimized Specific Format Functions
//------------------------------------------------------------------------------

// 32-bit ARGB to 32-bit RGB (just clear alpha)
static void CopyLine_32ARGB_32RGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Process 4 pixels at a time with loop unrolling
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        dst[x + 0] = src[x + 0] & 0x00FFFFFF;
        dst[x + 1] = src[x + 1] & 0x00FFFFFF;
        dst[x + 2] = src[x + 2] & 0x00FFFFFF;
        dst[x + 3] = src[x + 3] & 0x00FFFFFF;
    }
    for (; x < width; ++x) {
        dst[x] = src[x] & 0x00FFFFFF;
    }
}

// 32-bit RGB to 32-bit ARGB (add full alpha)
static void CopyLine_32RGB_32ARGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Process 4 pixels at a time with loop unrolling
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        dst[x + 0] = src[x + 0] | 0xFF000000;
        dst[x + 1] = src[x + 1] | 0xFF000000;
        dst[x + 2] = src[x + 2] | 0xFF000000;
        dst[x + 3] = src[x + 3] | 0xFF000000;
    }
    for (; x < width; ++x) {
        dst[x] = src[x] | 0xFF000000;
    }
}

// 32-bit ARGB to 24-bit RGB
static void CopyLine_32ARGB_24RGB(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XBYTE *dst = info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        dst[0] = src[0]; // B
        dst[1] = src[1]; // G
        dst[2] = src[2]; // R
        src += 4;
        dst += 3;
    }
}

// 24-bit RGB to 32-bit ARGB
static void CopyLine_24RGB_32ARGB(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XBYTE *dst = info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        dst[0] = src[0]; // B
        dst[1] = src[1]; // G
        dst[2] = src[2]; // R
        dst[3] = 0xFF;   // A
        src += 3;
        dst += 4;
    }
}

// 32-bit ARGB to 16-bit 565
static void CopyLine_32ARGB_565RGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;

    // Process 4 pixels at a time with loop unrolling
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        XDWORD p0 = src[x + 0], p1 = src[x + 1], p2 = src[x + 2], p3 = src[x + 3];
        dst[x + 0] = (XWORD)(((p0 >> 8) & 0xF800) | ((p0 >> 5) & 0x07E0) | ((p0 >> 3) & 0x001F));
        dst[x + 1] = (XWORD)(((p1 >> 8) & 0xF800) | ((p1 >> 5) & 0x07E0) | ((p1 >> 3) & 0x001F));
        dst[x + 2] = (XWORD)(((p2 >> 8) & 0xF800) | ((p2 >> 5) & 0x07E0) | ((p2 >> 3) & 0x001F));
        dst[x + 3] = (XWORD)(((p3 >> 8) & 0xF800) | ((p3 >> 5) & 0x07E0) | ((p3 >> 3) & 0x001F));
    }
    for (; x < width; ++x) {
        XDWORD pixel = src[x];
        dst[x] = (XWORD)(((pixel >> 8) & 0xF800) | ((pixel >> 5) & 0x07E0) | ((pixel >> 3) & 0x001F));
    }
}

// 32-bit ARGB to 16-bit 555
static void CopyLine_32ARGB_555RGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;

    // Process 4 pixels at a time with loop unrolling
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        XDWORD p0 = src[x + 0], p1 = src[x + 1], p2 = src[x + 2], p3 = src[x + 3];
        dst[x + 0] = (XWORD)(((p0 >> 9) & 0x7C00) | ((p0 >> 6) & 0x03E0) | ((p0 >> 3) & 0x001F));
        dst[x + 1] = (XWORD)(((p1 >> 9) & 0x7C00) | ((p1 >> 6) & 0x03E0) | ((p1 >> 3) & 0x001F));
        dst[x + 2] = (XWORD)(((p2 >> 9) & 0x7C00) | ((p2 >> 6) & 0x03E0) | ((p2 >> 3) & 0x001F));
        dst[x + 3] = (XWORD)(((p3 >> 9) & 0x7C00) | ((p3 >> 6) & 0x03E0) | ((p3 >> 3) & 0x001F));
    }
    for (; x < width; ++x) {
        XDWORD pixel = src[x];
        dst[x] = (XWORD)(((pixel >> 9) & 0x7C00) | ((pixel >> 6) & 0x03E0) | ((pixel >> 3) & 0x001F));
    }
}

// 32-bit ARGB to 16-bit 1555 ARGB
static void CopyLine_32ARGB_1555ARGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        XDWORD pixel = src[x];
        XWORD a = ((pixel >> 16) & 0x8000);
        XWORD r = ((pixel >> 9) & 0x7C00);
        XWORD g = ((pixel >> 6) & 0x03E0);
        XWORD b = ((pixel >> 3) & 0x001F);
        dst[x] = a | r | g | b;
    }
}

// 32-bit ARGB to 16-bit 4444 ARGB
static void CopyLine_32ARGB_4444ARGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        XDWORD pixel = src[x];
        XWORD a = ((pixel >> 16) & 0xF000);
        XWORD r = ((pixel >> 12) & 0x0F00);
        XWORD g = ((pixel >> 8) & 0x00F0);
        XWORD b = ((pixel >> 4) & 0x000F);
        dst[x] = a | r | g | b;
    }
}

// 16-bit 565 to 32-bit ARGB
static void CopyLine_565RGB_32ARGB(const VxBlitInfo *info) {
    const XWORD *src = (const XWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Process 4 pixels at a time with loop unrolling
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        XWORD p0 = src[x + 0], p1 = src[x + 1], p2 = src[x + 2], p3 = src[x + 3];
        dst[x + 0] = 0xFF000000 | ((p0 & 0xF800) << 8) | ((p0 & 0xE000) << 3) | ((p0 & 0x07E0) << 5) | ((p0 & 0x0600) >> 1) | ((p0 & 0x001F) << 3) | ((p0 & 0x001C) >> 2);
        dst[x + 1] = 0xFF000000 | ((p1 & 0xF800) << 8) | ((p1 & 0xE000) << 3) | ((p1 & 0x07E0) << 5) | ((p1 & 0x0600) >> 1) | ((p1 & 0x001F) << 3) | ((p1 & 0x001C) >> 2);
        dst[x + 2] = 0xFF000000 | ((p2 & 0xF800) << 8) | ((p2 & 0xE000) << 3) | ((p2 & 0x07E0) << 5) | ((p2 & 0x0600) >> 1) | ((p2 & 0x001F) << 3) | ((p2 & 0x001C) >> 2);
        dst[x + 3] = 0xFF000000 | ((p3 & 0xF800) << 8) | ((p3 & 0xE000) << 3) | ((p3 & 0x07E0) << 5) | ((p3 & 0x0600) >> 1) | ((p3 & 0x001F) << 3) | ((p3 & 0x001C) >> 2);
    }
    for (; x < width; ++x) {
        XWORD pixel = src[x];
        XDWORD r = ((pixel & 0xF800) << 8) | ((pixel & 0xE000) << 3);
        XDWORD g = ((pixel & 0x07E0) << 5) | ((pixel & 0x0600) >> 1);
        XDWORD b = ((pixel & 0x001F) << 3) | ((pixel & 0x001C) >> 2);
        dst[x] = 0xFF000000 | r | g | b;
    }
}

// 16-bit 555 to 32-bit ARGB
static void CopyLine_555RGB_32ARGB(const VxBlitInfo *info) {
    const XWORD *src = (const XWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Process 4 pixels at a time with loop unrolling
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        XWORD p0 = src[x + 0], p1 = src[x + 1], p2 = src[x + 2], p3 = src[x + 3];
        dst[x + 0] = 0xFF000000 | ((p0 & 0x7C00) << 9) | ((p0 & 0x7000) << 4) | ((p0 & 0x03E0) << 6) | ((p0 & 0x0380) << 1) | ((p0 & 0x001F) << 3) | ((p0 & 0x001C) >> 2);
        dst[x + 1] = 0xFF000000 | ((p1 & 0x7C00) << 9) | ((p1 & 0x7000) << 4) | ((p1 & 0x03E0) << 6) | ((p1 & 0x0380) << 1) | ((p1 & 0x001F) << 3) | ((p1 & 0x001C) >> 2);
        dst[x + 2] = 0xFF000000 | ((p2 & 0x7C00) << 9) | ((p2 & 0x7000) << 4) | ((p2 & 0x03E0) << 6) | ((p2 & 0x0380) << 1) | ((p2 & 0x001F) << 3) | ((p2 & 0x001C) >> 2);
        dst[x + 3] = 0xFF000000 | ((p3 & 0x7C00) << 9) | ((p3 & 0x7000) << 4) | ((p3 & 0x03E0) << 6) | ((p3 & 0x0380) << 1) | ((p3 & 0x001F) << 3) | ((p3 & 0x001C) >> 2);
    }
    for (; x < width; ++x) {
        XWORD pixel = src[x];
        XDWORD r = ((pixel & 0x7C00) << 9) | ((pixel & 0x7000) << 4);
        XDWORD g = ((pixel & 0x03E0) << 6) | ((pixel & 0x0380) << 1);
        XDWORD b = ((pixel & 0x001F) << 3) | ((pixel & 0x001C) >> 2);
        dst[x] = 0xFF000000 | r | g | b;
    }
}

// 16-bit 1555 to 32-bit ARGB
static void CopyLine_1555ARGB_32ARGB(const VxBlitInfo *info) {
    const XWORD *src = (const XWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        XWORD pixel = src[x];
        XDWORD a = (pixel & 0x8000) ? 0xFF000000 : 0x00000000;
        XDWORD r = ((pixel & 0x7C00) << 9) | ((pixel & 0x7000) << 4);
        XDWORD g = ((pixel & 0x03E0) << 6) | ((pixel & 0x0380) << 1);
        XDWORD b = ((pixel & 0x001F) << 3) | ((pixel & 0x001C) >> 2);
        dst[x] = a | r | g | b;
    }
}

// 16-bit 4444 to 32-bit ARGB
static void CopyLine_4444ARGB_32ARGB(const VxBlitInfo *info) {
    const XWORD *src = (const XWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        XWORD pixel = src[x];
        XDWORD a = ((pixel & 0xF000) << 16) | ((pixel & 0xF000) << 12);
        XDWORD r = ((pixel & 0x0F00) << 12) | ((pixel & 0x0F00) << 8);
        XDWORD g = ((pixel & 0x00F0) << 8) | ((pixel & 0x00F0) << 4);
        XDWORD b = ((pixel & 0x000F) << 4) | (pixel & 0x000F);
        dst[x] = a | r | g | b;
    }
}

//------------------------------------------------------------------------------
// Paletted Image Blitting Functions
//------------------------------------------------------------------------------

// 8-bit paletted to 32-bit ARGB
static void CopyLine_Paletted8_32ARGB(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const XBYTE *colorMap = info->colorMap;
    const int bpc = info->bytesPerColorEntry;

    if (!colorMap) return;

    // Pre-build lookup table for the palette (avoid per-pixel bpc multiply)
    XDWORD palette32[256];
    if (bpc >= 4) {
        for (int i = 0; i < 256; ++i) {
            const XBYTE *entry = colorMap + i * bpc;
            palette32[i] = (entry[3] << 24) | (entry[2] << 16) | (entry[1] << 8) | entry[0];
        }
    } else {
        for (int i = 0; i < 256; ++i) {
            const XBYTE *entry = colorMap + i * bpc;
            palette32[i] = 0xFF000000 | (entry[2] << 16) | (entry[1] << 8) | entry[0];
        }
    }

    // Process with loop unrolling
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        dst[x + 0] = palette32[src[x + 0]];
        dst[x + 1] = palette32[src[x + 1]];
        dst[x + 2] = palette32[src[x + 2]];
        dst[x + 3] = palette32[src[x + 3]];
    }
    for (; x < width; ++x) {
        dst[x] = palette32[src[x]];
    }
}

// 8-bit paletted to 24-bit RGB
static void CopyLine_Paletted8_24RGB(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XBYTE *dst = info->dstLine;
    const int width = info->width;
    const XBYTE *colorMap = info->colorMap;
    const int bpc = info->bytesPerColorEntry;

    if (!colorMap) return;

    // Pre-build lookup table (store BGR in each entry)
    XDWORD palette32[256];
    for (int i = 0; i < 256; ++i) {
        const XBYTE *entry = colorMap + i * bpc;
        palette32[i] = (entry[2] << 16) | (entry[1] << 8) | entry[0];
    }

    for (int x = 0; x < width; ++x) {
        XDWORD color = palette32[src[x]];
        dst[0] = (XBYTE)color;          // B
        dst[1] = (XBYTE)(color >> 8);   // G
        dst[2] = (XBYTE)(color >> 16);  // R
        dst += 3;
    }
}

// 8-bit paletted to 16-bit 565
static void CopyLine_Paletted8_565RGB(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;
    const XBYTE *colorMap = info->colorMap;
    const int bpc = info->bytesPerColorEntry;

    if (!colorMap) return;

    // Pre-build 565 lookup table
    XWORD palette16[256];
    for (int i = 0; i < 256; ++i) {
        const XBYTE *entry = colorMap + i * bpc;
        palette16[i] = ((entry[2] >> 3) << 11) | ((entry[1] >> 2) << 5) | (entry[0] >> 3);
    }

    // Process with loop unrolling
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        dst[x + 0] = palette16[src[x + 0]];
        dst[x + 1] = palette16[src[x + 1]];
        dst[x + 2] = palette16[src[x + 2]];
        dst[x + 3] = palette16[src[x + 3]];
    }
    for (; x < width; ++x) {
        dst[x] = palette16[src[x]];
    }
}

// 8-bit paletted to 16-bit with alpha (1555, 4444)
static void CopyLine_Paletted8_16Alpha(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;
    const XBYTE *colorMap = info->colorMap;
    const int bpc = info->bytesPerColorEntry;

    if (!colorMap) return;

    // Get shift amounts for destination format
    const int rShift = info->redShiftDst;
    const int gShift = info->greenShiftDst;
    const int bShift = info->blueShiftDst;
    const int aShift = info->alphaShiftDst;
    
    // Get bit widths from masks
    XDWORD rWidth = GetBitCountLocal(info->dstRedMask);
    XDWORD gWidth = GetBitCountLocal(info->dstGreenMask);
    XDWORD bWidth = GetBitCountLocal(info->dstBlueMask);
    XDWORD aWidth = GetBitCountLocal(info->dstAlphaMask);
    
    for (int x = 0; x < width; ++x) {
        int idx = src[x];
        const XBYTE *entry = colorMap + idx * bpc;
        XBYTE b_val = entry[0];
        XBYTE g_val = entry[1];
        XBYTE r_val = entry[2];
        XBYTE a_val = (bpc >= 4) ? entry[3] : 0xFF;
        
        XWORD r = (r_val >> (8 - rWidth)) << rShift;
        XWORD g = (g_val >> (8 - gWidth)) << gShift;
        XWORD b = (b_val >> (8 - bWidth)) << bShift;
        XWORD a = (a_val >> (8 - aWidth)) << aShift;
        
        dst[x] = r | g | b | a;
    }
}

// 8-bit paletted to 8-bit (index copy or remap)
static void CopyLine_Paletted8_8(const VxBlitInfo *info) {
    memcpy(info->dstLine, info->srcLine, info->width);
}

//------------------------------------------------------------------------------
// Channel Swap Functions (Scalar)
//------------------------------------------------------------------------------

// 32-bit ARGB to 32-bit ABGR (swap R and B channels)
static void CopyLine_32ARGB_32ABGR(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    int x = 0;
    // Process 4 pixels at a time with loop unrolling
    for (; x + 4 <= width; x += 4) {
        XDWORD p0 = src[x + 0];
        XDWORD p1 = src[x + 1];
        XDWORD p2 = src[x + 2];
        XDWORD p3 = src[x + 3];
        dst[x + 0] = (p0 & 0xFF00FF00) | ((p0 & 0x00FF0000) >> 16) | ((p0 & 0x000000FF) << 16);
        dst[x + 1] = (p1 & 0xFF00FF00) | ((p1 & 0x00FF0000) >> 16) | ((p1 & 0x000000FF) << 16);
        dst[x + 2] = (p2 & 0xFF00FF00) | ((p2 & 0x00FF0000) >> 16) | ((p2 & 0x000000FF) << 16);
        dst[x + 3] = (p3 & 0xFF00FF00) | ((p3 & 0x00FF0000) >> 16) | ((p3 & 0x000000FF) << 16);
    }
    for (; x < width; ++x) {
        XDWORD p = src[x];
        dst[x] = (p & 0xFF00FF00) | ((p & 0x00FF0000) >> 16) | ((p & 0x000000FF) << 16);
    }
}

// 32-bit ABGR to 32-bit ARGB (same operation - swap is symmetric)
static void CopyLine_32ABGR_32ARGB(const VxBlitInfo *info) {
    CopyLine_32ARGB_32ABGR(info);
}

// 32-bit ARGB to 32-bit RGBA (rotate bytes left)
static void CopyLine_32ARGB_32RGBA(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    int x = 0;
    for (; x + 4 <= width; x += 4) {
        XDWORD p0 = src[x + 0];
        XDWORD p1 = src[x + 1];
        XDWORD p2 = src[x + 2];
        XDWORD p3 = src[x + 3];
        dst[x + 0] = (p0 << 8) | (p0 >> 24);
        dst[x + 1] = (p1 << 8) | (p1 >> 24);
        dst[x + 2] = (p2 << 8) | (p2 >> 24);
        dst[x + 3] = (p3 << 8) | (p3 >> 24);
    }
    for (; x < width; ++x) {
        XDWORD p = src[x];
        dst[x] = (p << 8) | (p >> 24);
    }
}

// 32-bit RGBA to 32-bit ARGB (rotate bytes right)
static void CopyLine_32RGBA_32ARGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    int x = 0;
    for (; x + 4 <= width; x += 4) {
        XDWORD p0 = src[x + 0];
        XDWORD p1 = src[x + 1];
        XDWORD p2 = src[x + 2];
        XDWORD p3 = src[x + 3];
        dst[x + 0] = (p0 >> 8) | (p0 << 24);
        dst[x + 1] = (p1 >> 8) | (p1 << 24);
        dst[x + 2] = (p2 >> 8) | (p2 << 24);
        dst[x + 3] = (p3 >> 8) | (p3 << 24);
    }
    for (; x < width; ++x) {
        XDWORD p = src[x];
        dst[x] = (p >> 8) | (p << 24);
    }
}

// 32-bit ARGB to 32-bit BGRA (full byte reversal)
static void CopyLine_32ARGB_32BGRA(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    int x = 0;
    for (; x + 4 <= width; x += 4) {
        XDWORD p0 = src[x + 0];
        XDWORD p1 = src[x + 1];
        XDWORD p2 = src[x + 2];
        XDWORD p3 = src[x + 3];
        dst[x + 0] = ((p0 >> 24) & 0xFF) | ((p0 >> 8) & 0xFF00) | ((p0 << 8) & 0xFF0000) | ((p0 << 24) & 0xFF000000);
        dst[x + 1] = ((p1 >> 24) & 0xFF) | ((p1 >> 8) & 0xFF00) | ((p1 << 8) & 0xFF0000) | ((p1 << 24) & 0xFF000000);
        dst[x + 2] = ((p2 >> 24) & 0xFF) | ((p2 >> 8) & 0xFF00) | ((p2 << 8) & 0xFF0000) | ((p2 << 24) & 0xFF000000);
        dst[x + 3] = ((p3 >> 24) & 0xFF) | ((p3 >> 8) & 0xFF00) | ((p3 << 8) & 0xFF0000) | ((p3 << 24) & 0xFF000000);
    }
    for (; x < width; ++x) {
        XDWORD p = src[x];
        dst[x] = ((p >> 24) & 0xFF) | ((p >> 8) & 0xFF00) | ((p << 8) & 0xFF0000) | ((p << 24) & 0xFF000000);
    }
}

// 32-bit BGRA to 32-bit ARGB (same operation - reversal is symmetric)
static void CopyLine_32BGRA_32ARGB(const VxBlitInfo *info) {
    CopyLine_32ARGB_32BGRA(info);
}

//------------------------------------------------------------------------------
// Premultiplied Alpha Functions (Scalar)
//------------------------------------------------------------------------------

// Premultiply alpha (ARGB)
static void PremultiplyAlpha_32ARGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        XDWORD p = src[x];
        XDWORD a = (p >> 24) & 0xFF;
        XDWORD r = ((p >> 16) & 0xFF) * a / 255;
        XDWORD g = ((p >> 8) & 0xFF) * a / 255;
        XDWORD b = (p & 0xFF) * a / 255;
        dst[x] = (a << 24) | (r << 16) | (g << 8) | b;
    }
}

// Unpremultiply alpha (ARGB)
static void UnpremultiplyAlpha_32ARGB(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    for (int x = 0; x < width; ++x) {
        XDWORD p = src[x];
        XDWORD a = (p >> 24) & 0xFF;
        if (a == 0) {
            dst[x] = 0;
        } else if (a == 255) {
            dst[x] = p;
        } else {
            XDWORD r = XMin((XDWORD)255, ((p >> 16) & 0xFF) * 255 / a);
            XDWORD g = XMin((XDWORD)255, ((p >> 8) & 0xFF) * 255 / a);
            XDWORD b = XMin((XDWORD)255, (p & 0xFF) * 255 / a);
            dst[x] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

//------------------------------------------------------------------------------
// SSE Optimized Functions
//------------------------------------------------------------------------------

#if defined(VX_SIMD_SSE2)

// SSE2: 32-bit ARGB to 32-bit RGB (clear alpha) - process 4 pixels at a time
static void CopyLine_32ARGB_32RGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i mask = _mm_set1_epi32(0x00FFFFFF);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));
        pixels = _mm_and_si128(pixels, mask);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        dst[x] = src[x] & 0x00FFFFFF;
    }
}

// SSE2: 32-bit RGB to 32-bit ARGB (add full alpha) - process 4 pixels at a time
static void CopyLine_32RGB_32ARGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i alphaMask = _mm_set1_epi32(0xFF000000);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));
        pixels = _mm_or_si128(pixels, alphaMask);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        dst[x] = src[x] | 0xFF000000;
    }
}

// SSE2: Set alpha channel to constant value - process 4 pixels at a time
static void SetAlpha_32_SSE(const VxBlitInfo *info) {
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const XDWORD alphaValue = info->alphaValue & info->dstAlphaMask;
    const XDWORD alphaMaskInv = info->alphaMaskInv;

    const __m128i vAlphaValue = _mm_set1_epi32(alphaValue);
    const __m128i vAlphaMaskInv = _mm_set1_epi32(alphaMaskInv);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(dst + x));
        pixels = _mm_and_si128(pixels, vAlphaMaskInv);
        pixels = _mm_or_si128(pixels, vAlphaValue);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        dst[x] = (dst[x] & alphaMaskInv) | alphaValue;
    }
}

// SSE2: Copy alpha from byte array - process 4 pixels at a time
static void CopyAlpha_32_SSE(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const int alphaShift = info->alphaShiftDst;
    const XDWORD alphaMask = info->dstAlphaMask;
    const XDWORD alphaMaskInv = info->alphaMaskInv;

    const __m128i vAlphaMaskInv = _mm_set1_epi32(alphaMaskInv);
    const __m128i zero = _mm_setzero_si128();
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        // Load 4 alpha bytes and expand to 32-bit
        __m128i alphas = _mm_cvtsi32_si128(*(const int *)(src + x));
        alphas = _mm_unpacklo_epi8(alphas, zero);
        alphas = _mm_unpacklo_epi16(alphas, zero);

        // Shift to alpha position
        alphas = _mm_slli_epi32(alphas, alphaShift);

        // Load destination, clear alpha, set new alpha
        __m128i pixels = _mm_loadu_si128((const __m128i *)(dst + x));
        pixels = _mm_and_si128(pixels, vAlphaMaskInv);
        pixels = _mm_or_si128(pixels, alphas);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD alpha = ((XDWORD)src[x] << alphaShift) & alphaMask;
        dst[x] = (dst[x] & alphaMaskInv) | alpha;
    }
}

// SSE2: 32-bit ARGB to 16-bit 565 - process 4 pixels at a time
static void CopyLine_32ARGB_565RGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;

    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));

        // Extract R (bits 16-23 -> bits 11-15)
        __m128i r = _mm_and_si128(_mm_srli_epi32(pixels, 8), _mm_set1_epi32(0xF800));
        // Extract G (bits 8-15 -> bits 5-10)
        __m128i g = _mm_and_si128(_mm_srli_epi32(pixels, 5), _mm_set1_epi32(0x07E0));
        // Extract B (bits 0-7 -> bits 0-4)
        __m128i b = _mm_and_si128(_mm_srli_epi32(pixels, 3), _mm_set1_epi32(0x001F));

        // Combine
        __m128i result = _mm_or_si128(_mm_or_si128(r, g), b);

        // Pack 32-bit to 16-bit using shuffle (SSE2 compatible, avoids signed saturation)
        result = _mm_shufflelo_epi16(result, _MM_SHUFFLE(2, 0, 2, 0));
        result = _mm_shufflehi_epi16(result, _MM_SHUFFLE(2, 0, 2, 0));
        result = _mm_shuffle_epi32(result, _MM_SHUFFLE(2, 0, 2, 0));
        _mm_storel_epi64((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD pixel = src[x];
        XWORD r = ((pixel >> 8) & 0xF800);
        XWORD g = ((pixel >> 5) & 0x07E0);
        XWORD b = ((pixel >> 3) & 0x001F);
        dst[x] = r | g | b;
    }
}

// SSE2: 32-bit ARGB to 16-bit 4444 - process 4 pixels at a time
static void CopyLine_32ARGB_4444ARGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;

    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));

        // Extract A (bits 24-31 -> bits 12-15)
        __m128i a = _mm_and_si128(_mm_srli_epi32(pixels, 16), _mm_set1_epi32(0xF000));
        // Extract R (bits 16-23 -> bits 8-11)
        __m128i r = _mm_and_si128(_mm_srli_epi32(pixels, 12), _mm_set1_epi32(0x0F00));
        // Extract G (bits 8-15 -> bits 4-7)
        __m128i g = _mm_and_si128(_mm_srli_epi32(pixels, 8), _mm_set1_epi32(0x00F0));
        // Extract B (bits 0-7 -> bits 0-3)
        __m128i b = _mm_and_si128(_mm_srli_epi32(pixels, 4), _mm_set1_epi32(0x000F));

        // Combine
        __m128i result = _mm_or_si128(_mm_or_si128(a, r), _mm_or_si128(g, b));

        // Pack 32-bit to 16-bit using shuffle (SSE2 compatible, avoids signed saturation)
        result = _mm_shufflelo_epi16(result, _MM_SHUFFLE(2, 0, 2, 0));
        result = _mm_shufflehi_epi16(result, _MM_SHUFFLE(2, 0, 2, 0));
        result = _mm_shuffle_epi32(result, _MM_SHUFFLE(2, 0, 2, 0));
        _mm_storel_epi64((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD pixel = src[x];
        XWORD a = ((pixel >> 16) & 0xF000);
        XWORD r = ((pixel >> 12) & 0x0F00);
        XWORD g = ((pixel >> 8) & 0x00F0);
        XWORD b = ((pixel >> 4) & 0x000F);
        dst[x] = a | r | g | b;
    }
}

// SSE2: 16-bit 565 to 32-bit ARGB - process 8 pixels at a time
static void CopyLine_565RGB_32ARGB_SSE(const VxBlitInfo *info) {
    const XWORD *src = (const XWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i alpha = _mm_set1_epi32(0xFF000000);
    const __m128i zero = _mm_setzero_si128();
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        // Load 4 pixels (64 bits)
        __m128i pixels = _mm_loadl_epi64((const __m128i *)(src + x));

        // Unpack to 32-bit
        pixels = _mm_unpacklo_epi16(pixels, zero);

        // Extract and expand R (5 bits -> 8 bits)
        __m128i r = _mm_and_si128(pixels, _mm_set1_epi32(0xF800));
        r = _mm_slli_epi32(r, 8);
        r = _mm_or_si128(r, _mm_srli_epi32(r, 5));
        r = _mm_and_si128(r, _mm_set1_epi32(0x00FF0000));

        // Extract and expand G (6 bits -> 8 bits)
        __m128i g = _mm_and_si128(pixels, _mm_set1_epi32(0x07E0));
        g = _mm_slli_epi32(g, 5);
        g = _mm_or_si128(g, _mm_srli_epi32(g, 6));
        g = _mm_and_si128(g, _mm_set1_epi32(0x0000FF00));

        // Extract and expand B (5 bits -> 8 bits)
        __m128i b = _mm_and_si128(pixels, _mm_set1_epi32(0x001F));
        b = _mm_slli_epi32(b, 3);
        b = _mm_or_si128(b, _mm_srli_epi32(b, 5));
        b = _mm_and_si128(b, _mm_set1_epi32(0x000000FF));

        // Combine ARGB
        __m128i result = _mm_or_si128(_mm_or_si128(alpha, r), _mm_or_si128(g, b));
        _mm_storeu_si128((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XWORD pixel = src[x];
        XDWORD r = ((pixel & 0xF800) << 8) | ((pixel & 0xE000) << 3);
        XDWORD g = ((pixel & 0x07E0) << 5) | ((pixel & 0x0600) >> 1);
        XDWORD b = ((pixel & 0x001F) << 3) | ((pixel & 0x001C) >> 2);
        dst[x] = 0xFF000000 | r | g | b;
    }
}

// SSE2: 32-bit ARGB to 16-bit 555 - process 4 pixels at a time
static void CopyLine_32ARGB_555RGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;

    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));

        // Extract R (bits 16-23 -> bits 10-14)
        __m128i r = _mm_and_si128(_mm_srli_epi32(pixels, 9), _mm_set1_epi32(0x7C00));
        // Extract G (bits 8-15 -> bits 5-9)
        __m128i g = _mm_and_si128(_mm_srli_epi32(pixels, 6), _mm_set1_epi32(0x03E0));
        // Extract B (bits 0-7 -> bits 0-4)
        __m128i b = _mm_and_si128(_mm_srli_epi32(pixels, 3), _mm_set1_epi32(0x001F));

        // Combine
        __m128i result = _mm_or_si128(_mm_or_si128(r, g), b);

        // Pack 32-bit to 16-bit using shuffle (SSE2 compatible, avoids signed saturation)
        result = _mm_shufflelo_epi16(result, _MM_SHUFFLE(2, 0, 2, 0));
        result = _mm_shufflehi_epi16(result, _MM_SHUFFLE(2, 0, 2, 0));
        result = _mm_shuffle_epi32(result, _MM_SHUFFLE(2, 0, 2, 0));
        _mm_storel_epi64((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD pixel = src[x];
        XWORD r = ((pixel >> 9) & 0x7C00);
        XWORD g = ((pixel >> 6) & 0x03E0);
        XWORD b = ((pixel >> 3) & 0x001F);
        dst[x] = r | g | b;
    }
}

// SSE2: 32-bit ARGB to 16-bit 1555 - process 4 pixels at a time
static void CopyLine_32ARGB_1555ARGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;

    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));

        // Extract A (bit 31 -> bit 15)
        __m128i a = _mm_and_si128(_mm_srli_epi32(pixels, 16), _mm_set1_epi32(0x8000));
        // Extract R (bits 16-23 -> bits 10-14)
        __m128i r = _mm_and_si128(_mm_srli_epi32(pixels, 9), _mm_set1_epi32(0x7C00));
        // Extract G (bits 8-15 -> bits 5-9)
        __m128i g = _mm_and_si128(_mm_srli_epi32(pixels, 6), _mm_set1_epi32(0x03E0));
        // Extract B (bits 0-7 -> bits 0-4)
        __m128i b = _mm_and_si128(_mm_srli_epi32(pixels, 3), _mm_set1_epi32(0x001F));

        // Combine
        __m128i result = _mm_or_si128(_mm_or_si128(a, r), _mm_or_si128(g, b));

        // Pack 32-bit to 16-bit using shuffle (SSE2 compatible)
        // Each 32-bit lane has result in low 16 bits
        // Shuffle to pack: [0,1,2,3] -> [0,2] (low 64 bits)
        result = _mm_shufflelo_epi16(result, _MM_SHUFFLE(2, 0, 2, 0));
        result = _mm_shufflehi_epi16(result, _MM_SHUFFLE(2, 0, 2, 0));
        result = _mm_shuffle_epi32(result, _MM_SHUFFLE(2, 0, 2, 0));
        _mm_storel_epi64((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD pixel = src[x];
        XWORD a = ((pixel >> 16) & 0x8000);
        XWORD r = ((pixel >> 9) & 0x7C00);
        XWORD g = ((pixel >> 6) & 0x03E0);
        XWORD b = ((pixel >> 3) & 0x001F);
        dst[x] = a | r | g | b;
    }
}

// SSE2: 16-bit 555 to 32-bit ARGB - process 4 pixels at a time
static void CopyLine_555RGB_32ARGB_SSE(const VxBlitInfo *info) {
    const XWORD *src = (const XWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i alpha = _mm_set1_epi32(0xFF000000);
    const __m128i zero = _mm_setzero_si128();
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        // Load 4 pixels (64 bits)
        __m128i pixels = _mm_loadl_epi64((const __m128i *)(src + x));

        // Unpack to 32-bit
        pixels = _mm_unpacklo_epi16(pixels, zero);

        // Extract and expand R (5 bits -> 8 bits)
        __m128i r = _mm_and_si128(pixels, _mm_set1_epi32(0x7C00));
        r = _mm_slli_epi32(r, 9);
        r = _mm_or_si128(r, _mm_srli_epi32(r, 5));
        r = _mm_and_si128(r, _mm_set1_epi32(0x00FF0000));

        // Extract and expand G (5 bits -> 8 bits)
        __m128i g = _mm_and_si128(pixels, _mm_set1_epi32(0x03E0));
        g = _mm_slli_epi32(g, 6);
        g = _mm_or_si128(g, _mm_srli_epi32(g, 5));
        g = _mm_and_si128(g, _mm_set1_epi32(0x0000FF00));

        // Extract and expand B (5 bits -> 8 bits)
        __m128i b = _mm_and_si128(pixels, _mm_set1_epi32(0x001F));
        b = _mm_slli_epi32(b, 3);
        b = _mm_or_si128(b, _mm_srli_epi32(b, 5));
        b = _mm_and_si128(b, _mm_set1_epi32(0x000000FF));

        // Combine ARGB
        __m128i result = _mm_or_si128(_mm_or_si128(alpha, r), _mm_or_si128(g, b));
        _mm_storeu_si128((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XWORD pixel = src[x];
        XDWORD r = ((pixel & 0x7C00) << 9) | ((pixel & 0x7000) << 4);
        XDWORD g = ((pixel & 0x03E0) << 6) | ((pixel & 0x0380) << 1);
        XDWORD b = ((pixel & 0x001F) << 3) | ((pixel & 0x001C) >> 2);
        dst[x] = 0xFF000000 | r | g | b;
    }
}

// SSE2: 16-bit 1555 to 32-bit ARGB - process 4 pixels at a time
static void CopyLine_1555ARGB_32ARGB_SSE(const VxBlitInfo *info) {
    const XWORD *src = (const XWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i zero = _mm_setzero_si128();
    const __m128i alphaMask16 = _mm_set1_epi32(0x8000);
    const __m128i alphaFull = _mm_set1_epi32(0xFF000000);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        // Load 4 pixels (64 bits)
        __m128i pixels = _mm_loadl_epi64((const __m128i *)(src + x));

        // Unpack to 32-bit
        pixels = _mm_unpacklo_epi16(pixels, zero);

        // Extract A (1 bit -> 8 bits)
        // Check if alpha bit is set, then produce 0xFF000000 or 0
        __m128i a = _mm_and_si128(pixels, alphaMask16);
        // Compare equal to alphaMask16 (not zero) to avoid signed comparison issue
        __m128i aMask = _mm_cmpeq_epi32(a, alphaMask16);
        a = _mm_and_si128(aMask, alphaFull);

        // Extract and expand R (5 bits -> 8 bits)
        __m128i r = _mm_and_si128(pixels, _mm_set1_epi32(0x7C00));
        r = _mm_slli_epi32(r, 9);
        r = _mm_or_si128(r, _mm_srli_epi32(r, 5));
        r = _mm_and_si128(r, _mm_set1_epi32(0x00FF0000));

        // Extract and expand G (5 bits -> 8 bits)
        __m128i g = _mm_and_si128(pixels, _mm_set1_epi32(0x03E0));
        g = _mm_slli_epi32(g, 6);
        g = _mm_or_si128(g, _mm_srli_epi32(g, 5));
        g = _mm_and_si128(g, _mm_set1_epi32(0x0000FF00));

        // Extract and expand B (5 bits -> 8 bits)
        __m128i b = _mm_and_si128(pixels, _mm_set1_epi32(0x001F));
        b = _mm_slli_epi32(b, 3);
        b = _mm_or_si128(b, _mm_srli_epi32(b, 5));
        b = _mm_and_si128(b, _mm_set1_epi32(0x000000FF));

        // Combine ARGB
        __m128i result = _mm_or_si128(_mm_or_si128(a, r), _mm_or_si128(g, b));
        _mm_storeu_si128((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XWORD pixel = src[x];
        XDWORD a = (pixel & 0x8000) ? 0xFF000000 : 0x00000000;
        XDWORD r = ((pixel & 0x7C00) << 9) | ((pixel & 0x7000) << 4);
        XDWORD g = ((pixel & 0x03E0) << 6) | ((pixel & 0x0380) << 1);
        XDWORD b = ((pixel & 0x001F) << 3) | ((pixel & 0x001C) >> 2);
        dst[x] = a | r | g | b;
    }
}

// SSE2: 16-bit 4444 to 32-bit ARGB - process 4 pixels at a time
static void CopyLine_4444ARGB_32ARGB_SSE(const VxBlitInfo *info) {
    const XWORD *src = (const XWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i zero = _mm_setzero_si128();
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        // Load 4 pixels (64 bits)
        __m128i pixels = _mm_loadl_epi64((const __m128i *)(src + x));

        // Unpack to 32-bit
        pixels = _mm_unpacklo_epi16(pixels, zero);

        // Extract and expand A (4 bits -> 8 bits)
        __m128i a = _mm_and_si128(pixels, _mm_set1_epi32(0xF000));
        a = _mm_slli_epi32(a, 16);
        a = _mm_or_si128(a, _mm_srli_epi32(a, 4));
        a = _mm_and_si128(a, _mm_set1_epi32(0xFF000000));

        // Extract and expand R (4 bits -> 8 bits)
        __m128i r = _mm_and_si128(pixels, _mm_set1_epi32(0x0F00));
        r = _mm_slli_epi32(r, 12);
        r = _mm_or_si128(r, _mm_srli_epi32(r, 4));
        r = _mm_and_si128(r, _mm_set1_epi32(0x00FF0000));

        // Extract and expand G (4 bits -> 8 bits)
        __m128i g = _mm_and_si128(pixels, _mm_set1_epi32(0x00F0));
        g = _mm_slli_epi32(g, 8);
        g = _mm_or_si128(g, _mm_srli_epi32(g, 4));
        g = _mm_and_si128(g, _mm_set1_epi32(0x0000FF00));

        // Extract and expand B (4 bits -> 8 bits)
        __m128i b = _mm_and_si128(pixels, _mm_set1_epi32(0x000F));
        b = _mm_slli_epi32(b, 4);
        b = _mm_or_si128(b, _mm_and_si128(pixels, _mm_set1_epi32(0x000F)));
        b = _mm_and_si128(b, _mm_set1_epi32(0x000000FF));

        // Combine ARGB
        __m128i result = _mm_or_si128(_mm_or_si128(a, r), _mm_or_si128(g, b));
        _mm_storeu_si128((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XWORD pixel = src[x];
        XDWORD a = ((pixel & 0xF000) << 16) | ((pixel & 0xF000) << 12);
        XDWORD r = ((pixel & 0x0F00) << 12) | ((pixel & 0x0F00) << 8);
        XDWORD g = ((pixel & 0x00F0) << 8) | ((pixel & 0x00F0) << 4);
        XDWORD b = ((pixel & 0x000F) << 4) | (pixel & 0x000F);
        dst[x] = a | r | g | b;
    }
}

// SSE2: 24-bit RGB to 32-bit ARGB
static void CopyLine_24RGB_32ARGB_SSE(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i alpha = _mm_set1_epi32(0xFF000000);
    int x = 0;

    // Process 16 pixels at a time (48 bytes) using SIMD shuffle
    // This loads 16 bytes at a time and shuffles to extract RGB values
    for (; x + 16 <= width; x += 16) {
        // Load 48 bytes (16 RGB pixels) in 3 chunks of 16 bytes
        __m128i chunk0 = _mm_loadu_si128((const __m128i *)(src));       // bytes 0-15
        __m128i chunk1 = _mm_loadu_si128((const __m128i *)(src + 16));  // bytes 16-31
        __m128i chunk2 = _mm_loadu_si128((const __m128i *)(src + 32));  // bytes 32-47

        // Process first 4 pixels from chunk0 (bytes 0-11)
        // Shuffle mask to convert BGR BGR BGR BGR to BGRA BGRA BGRA BGRA
        // We need to insert 0xFF at alpha position
        const __m128i shuf0 = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
        __m128i out0 = _mm_shuffle_epi8(chunk0, shuf0);
        out0 = _mm_or_si128(out0, alpha);
        _mm_storeu_si128((__m128i *)(dst + x), out0);

        // Process next 4 pixels from chunk0 (byte 12-15) + chunk1 (bytes 0-8)
        // We need bytes 12,13,14,15 from chunk0 and 0,1,2,3,4,5,6,7 from chunk1
        __m128i blend1 = _mm_alignr_epi8(chunk1, chunk0, 12);  // bytes 12-27
        __m128i out1 = _mm_shuffle_epi8(blend1, shuf0);
        out1 = _mm_or_si128(out1, alpha);
        _mm_storeu_si128((__m128i *)(dst + x + 4), out1);

        // Process next 4 pixels from chunk1 (bytes 8-15) + chunk2 (bytes 0-3)
        __m128i blend2 = _mm_alignr_epi8(chunk2, chunk1, 8);   // bytes 24-39
        __m128i out2 = _mm_shuffle_epi8(blend2, shuf0);
        out2 = _mm_or_si128(out2, alpha);
        _mm_storeu_si128((__m128i *)(dst + x + 8), out2);

        // Process last 4 pixels from chunk2 (bytes 4-15)
        __m128i out3 = _mm_shuffle_epi8(_mm_srli_si128(chunk2, 4), shuf0);
        out3 = _mm_or_si128(out3, alpha);
        _mm_storeu_si128((__m128i *)(dst + x + 12), out3);

        src += 48;
    }

    // Process 4 pixels at a time for remainder
    for (; x + 4 <= width; x += 4) {
        XDWORD p0 = src[0] | (src[1] << 8) | (src[2] << 16);
        XDWORD p1 = src[3] | (src[4] << 8) | (src[5] << 16);
        XDWORD p2 = src[6] | (src[7] << 8) | (src[8] << 16);
        XDWORD p3 = src[9] | (src[10] << 8) | (src[11] << 16);

        __m128i pixels = _mm_set_epi32(p3, p2, p1, p0);
        pixels = _mm_or_si128(pixels, alpha);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);

        src += 12;
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        dst[x] = src[0] | (src[1] << 8) | (src[2] << 16) | 0xFF000000;
        src += 3;
    }
}

// SSE2: 32-bit ARGB to 24-bit RGB
static void CopyLine_32ARGB_24RGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XBYTE *dst = info->dstLine;
    const int width = info->width;

    int x = 0;

    // Process 16 pixels at a time (64 bytes in, 48 bytes out)
    for (; x + 16 <= width; x += 16) {
        // Load 16 ARGB pixels (64 bytes)
        __m128i p0 = _mm_loadu_si128((const __m128i *)(src + x));
        __m128i p1 = _mm_loadu_si128((const __m128i *)(src + x + 4));
        __m128i p2 = _mm_loadu_si128((const __m128i *)(src + x + 8));
        __m128i p3 = _mm_loadu_si128((const __m128i *)(src + x + 12));

        // Shuffle to remove alpha channel
        // Input:  B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3
        // Output: B0 G0 R0 B1 G1 R1 B2 G2 R2 B3 G3 R3 XX XX XX XX
        const __m128i shuf = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
        
        __m128i rgb0 = _mm_shuffle_epi8(p0, shuf);  // 12 bytes valid
        __m128i rgb1 = _mm_shuffle_epi8(p1, shuf);  // 12 bytes valid
        __m128i rgb2 = _mm_shuffle_epi8(p2, shuf);  // 12 bytes valid
        __m128i rgb3 = _mm_shuffle_epi8(p3, shuf);  // 12 bytes valid

        // Combine into 3 output registers (48 bytes total)
        // out0: rgb0[0-11] + rgb1[0-3]
        // out1: rgb1[4-11] + rgb2[0-7]
        // out2: rgb2[8-11] + rgb3[0-11]
        
        __m128i out0 = _mm_or_si128(rgb0, _mm_slli_si128(rgb1, 12));
        __m128i out1 = _mm_or_si128(_mm_srli_si128(rgb1, 4), _mm_slli_si128(rgb2, 8));
        __m128i out2 = _mm_or_si128(_mm_srli_si128(rgb2, 8), _mm_slli_si128(rgb3, 4));

        _mm_storeu_si128((__m128i *)(dst), out0);
        _mm_storeu_si128((__m128i *)(dst + 16), out1);
        _mm_storeu_si128((__m128i *)(dst + 32), out2);

        dst += 48;
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD pixel = src[x];
        dst[0] = (XBYTE)pixel;
        dst[1] = (XBYTE)(pixel >> 8);
        dst[2] = (XBYTE)(pixel >> 16);
        dst += 3;
    }
}

// SSE2: 8-bit paletted to 32-bit ARGB (batch lookup)
static void CopyLine_Paletted8_32ARGB_SSE(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const XBYTE *colorMap = info->colorMap;
    const int bpc = info->bytesPerColorEntry;

    if (!colorMap) return;

    // Pre-build palette as 32-bit ARGB
    XDWORD palette32[256];
    for (int i = 0; i < 256; ++i) {
        const XBYTE *entry = colorMap + i * bpc;
        XDWORD b = entry[0];
        XDWORD g = entry[1];
        XDWORD r = entry[2];
        XDWORD a = (bpc >= 4) ? entry[3] : 0xFF;
        palette32[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }

    // Process pixels with table lookup
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        dst[x + 0] = palette32[src[x + 0]];
        dst[x + 1] = palette32[src[x + 1]];
        dst[x + 2] = palette32[src[x + 2]];
        dst[x + 3] = palette32[src[x + 3]];
    }

    for (; x < width; ++x) {
        dst[x] = palette32[src[x]];
    }
}

// SSE2: 32-bit ARGB to 32-bit ABGR (swap R and B channels)
static void CopyLine_32ARGB_32ABGR_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Shuffle mask to swap R and B: ARGB -> ABGR
    // Input bytes:  B G R A (little endian: byte 0=B, 1=G, 2=R, 3=A)
    // Output bytes: R G B A (little endian: byte 0=R, 1=G, 2=B, 3=A)
    const __m128i shufMask = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));
        pixels = _mm_shuffle_epi8(pixels, shufMask);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD p = src[x];
        // ARGB -> ABGR: swap R and B
        dst[x] = (p & 0xFF00FF00) | ((p & 0x00FF0000) >> 16) | ((p & 0x000000FF) << 16);
    }
}

// SSE2: 32-bit ABGR to 32-bit ARGB (swap R and B channels)
static void CopyLine_32ABGR_32ARGB_SSE(const VxBlitInfo *info) {
    // Same operation - swap is symmetric
    CopyLine_32ARGB_32ABGR_SSE(info);
}

// SSE2: 32-bit ARGB to 32-bit RGBA (rotate bytes left)
static void CopyLine_32ARGB_32RGBA_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Shuffle mask: ARGB (BGRA in memory) -> RGBA (ABGR in memory)
    // Input bytes:  B G R A -> Output: A B G R
    const __m128i shufMask = _mm_setr_epi8(3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));
        pixels = _mm_shuffle_epi8(pixels, shufMask);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD p = src[x];
        // ARGB -> RGBA: A moves from MSB to LSB, RGB shift left
        dst[x] = (p << 8) | (p >> 24);
    }
}

// SSE2: 32-bit RGBA to 32-bit ARGB (rotate bytes right)
static void CopyLine_32RGBA_32ARGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Shuffle mask: RGBA (ABGR in memory) -> ARGB (BGRA in memory)
    // Input bytes:  A B G R -> Output: B G R A
    const __m128i shufMask = _mm_setr_epi8(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));
        pixels = _mm_shuffle_epi8(pixels, shufMask);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD p = src[x];
        // RGBA -> ARGB: A moves from LSB to MSB, RGB shift right
        dst[x] = (p >> 8) | (p << 24);
    }
}

// SSE2: 32-bit ARGB to 32-bit BGRA
static void CopyLine_32ARGB_32BGRA_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Shuffle mask: ARGB (BGRA in memory) -> BGRA (ARGB in memory)
    // Input bytes:  B G R A -> Output: A R G B
    const __m128i shufMask = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));
        pixels = _mm_shuffle_epi8(pixels, shufMask);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    // Handle remaining pixels - full byte reversal
    for (; x < width; ++x) {
        XDWORD p = src[x];
        dst[x] = ((p & 0xFF000000) >> 24) | ((p & 0x00FF0000) >> 8) | 
                 ((p & 0x0000FF00) << 8)  | ((p & 0x000000FF) << 24);
    }
}

// SSE2: 32-bit BGRA to 32-bit ARGB
static void CopyLine_32BGRA_32ARGB_SSE(const VxBlitInfo *info) {
    // Same operation - reversal is symmetric
    CopyLine_32ARGB_32BGRA_SSE(info);
}

// SSE2: Premultiply alpha (ARGB) using accurate division by 255
// Uses the formula: (c * a + 127) / 255 ≈ (c * a + 128 + ((c * a + 128) >> 8)) >> 8
static void PremultiplyAlpha_32ARGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i zero = _mm_setzero_si128();
    const __m128i round = _mm_set1_epi16(128);
    int x = 0;

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));

        // Extract alpha for each pixel and broadcast to all channels
        // Alpha is at byte positions 3, 7, 11, 15 (in BGRA memory order)
        __m128i alpha32 = _mm_srli_epi32(pixels, 24);

        // Unpack pixels to 16-bit for multiplication
        __m128i lo16 = _mm_unpacklo_epi8(pixels, zero);  // Pixels 0, 1 -> 16-bit
        __m128i hi16 = _mm_unpackhi_epi8(pixels, zero);  // Pixels 2, 3 -> 16-bit

        // Create alpha values for each 16-bit word
        // For pixels 0,1: alpha32 has A0 in low 32 bits, A1 in next 32 bits
        __m128i a01 = _mm_unpacklo_epi32(alpha32, alpha32);
        __m128i a23 = _mm_unpackhi_epi32(alpha32, alpha32);
        __m128i alo = _mm_packs_epi32(a01, a01);  // A0 A0 A0 A0 A1 A1 A1 A1
        __m128i ahi = _mm_packs_epi32(a23, a23);  // A2 A2 A2 A2 A3 A3 A3 A3

        // Multiply: product = c * a
        __m128i prodlo = _mm_mullo_epi16(lo16, alo);
        __m128i prodhi = _mm_mullo_epi16(hi16, ahi);

        // Accurate division by 255: (prod + 128 + ((prod + 128) >> 8)) >> 8
        __m128i tmplo = _mm_add_epi16(prodlo, round);
        __m128i tmphi = _mm_add_epi16(prodhi, round);
        tmplo = _mm_add_epi16(tmplo, _mm_srli_epi16(tmplo, 8));
        tmphi = _mm_add_epi16(tmphi, _mm_srli_epi16(tmphi, 8));
        __m128i reslo = _mm_srli_epi16(tmplo, 8);
        __m128i reshi = _mm_srli_epi16(tmphi, 8);

        // Pack back to 8-bit
        __m128i result = _mm_packus_epi16(reslo, reshi);

        // Restore original alpha (alpha channel should not be modified)
        __m128i alphaMask = _mm_set1_epi32(0xFF000000);
        result = _mm_or_si128(_mm_andnot_si128(alphaMask, result),
                              _mm_and_si128(alphaMask, pixels));

        _mm_storeu_si128((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD p = src[x];
        XDWORD a = (p >> 24) & 0xFF;
        XDWORD r = (((p >> 16) & 0xFF) * a + 127) / 255;
        XDWORD g = (((p >> 8) & 0xFF) * a + 127) / 255;
        XDWORD b = ((p & 0xFF) * a + 127) / 255;
        dst[x] = (a << 24) | (r << 16) | (g << 8) | b;
    }
}

// SSE2: Unpremultiply alpha (ARGB)
// Division is tricky to vectorize, but we can use lookup tables or SIMD division
static void UnpremultiplyAlpha_32ARGB_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    // Use scalar for unpremultiply - division by variable is hard to vectorize
    // and the lookup table approach would require too much memory
    for (int x = 0; x < width; ++x) {
        XDWORD p = src[x];
        XDWORD a = (p >> 24) & 0xFF;
        if (a == 0) {
            dst[x] = 0;
        } else if (a == 255) {
            dst[x] = p;
        } else {
            XDWORD r = XMin((XDWORD)255, ((p >> 16) & 0xFF) * 255 / a);
            XDWORD g = XMin((XDWORD)255, ((p >> 8) & 0xFF) * 255 / a);
            XDWORD b = XMin((XDWORD)255, (p & 0xFF) * 255 / a);
            dst[x] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

// SSE2: Fill 32-bit image with solid color
static void FillLine_32_SSE(XDWORD *dst, int width, XDWORD color) {
    __m128i fillColor = _mm_set1_epi32(color);
    int x = 0;

    // Process 16 pixels at a time (64 bytes)
    for (; x + 16 <= width; x += 16) {
        _mm_storeu_si128((__m128i *)(dst + x), fillColor);
        _mm_storeu_si128((__m128i *)(dst + x + 4), fillColor);
        _mm_storeu_si128((__m128i *)(dst + x + 8), fillColor);
        _mm_storeu_si128((__m128i *)(dst + x + 12), fillColor);
    }

    // Process 4 pixels at a time
    for (; x + 4 <= width; x += 4) {
        _mm_storeu_si128((__m128i *)(dst + x), fillColor);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        dst[x] = color;
    }
}

// SSE2: Fill 16-bit image with solid color
static void FillLine_16_SSE(XWORD *dst, int width, XWORD color) {
    __m128i fillColor = _mm_set1_epi16(color);
    int x = 0;

    // Process 32 pixels at a time (64 bytes)
    for (; x + 32 <= width; x += 32) {
        _mm_storeu_si128((__m128i *)(dst + x), fillColor);
        _mm_storeu_si128((__m128i *)(dst + x + 8), fillColor);
        _mm_storeu_si128((__m128i *)(dst + x + 16), fillColor);
        _mm_storeu_si128((__m128i *)(dst + x + 24), fillColor);
    }

    // Process 8 pixels at a time
    for (; x + 8 <= width; x += 8) {
        _mm_storeu_si128((__m128i *)(dst + x), fillColor);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        dst[x] = color;
    }
}

// SSE2: Clear alpha channel (set to 0) - useful for RGB32 format
static void ClearAlpha_32_SSE(const VxBlitInfo *info) {
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const __m128i mask = _mm_set1_epi32(0x00FFFFFF);
    int x = 0;

    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((__m128i *)(dst + x));
        pixels = _mm_and_si128(pixels, mask);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    for (; x < width; ++x) {
        dst[x] &= 0x00FFFFFF;
    }
}

// SSE2: Set alpha channel to full (255)
static void SetFullAlpha_32_SSE(const VxBlitInfo *info) {
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const __m128i alphaMask = _mm_set1_epi32(0xFF000000);
    int x = 0;

    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((__m128i *)(dst + x));
        pixels = _mm_or_si128(pixels, alphaMask);
        _mm_storeu_si128((__m128i *)(dst + x), pixels);
    }

    for (; x < width; ++x) {
        dst[x] |= 0xFF000000;
    }
}

// SSE2: Invert colors (keep alpha)
static void InvertColors_32_SSE(const VxBlitInfo *info) {
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const __m128i invertMask = _mm_set1_epi32(0x00FFFFFF);
    int x = 0;

    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((__m128i *)(dst + x));
        // XOR RGB channels to invert, keep alpha
        __m128i inverted = _mm_xor_si128(pixels, invertMask);
        _mm_storeu_si128((__m128i *)(dst + x), inverted);
    }

    for (; x < width; ++x) {
        dst[x] ^= 0x00FFFFFF;
    }
}

// SSE2: Convert to grayscale using luminance formula
// Y = 0.299*R + 0.587*G + 0.114*B (using fixed point: 77*R + 150*G + 29*B) >> 8
static void Grayscale_32_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i zero = _mm_setzero_si128();
    const __m128i rCoef = _mm_set1_epi16(77);   // 0.299 * 256
    const __m128i gCoef = _mm_set1_epi16(150);  // 0.587 * 256
    const __m128i bCoef = _mm_set1_epi16(29);   // 0.114 * 256
    int x = 0;

    for (; x + 4 <= width; x += 4) {
        __m128i pixels = _mm_loadu_si128((const __m128i *)(src + x));

        // Extract and save alpha
        __m128i alpha = _mm_and_si128(pixels, _mm_set1_epi32(0xFF000000));

        // Unpack to 16-bit
        __m128i lo = _mm_unpacklo_epi8(pixels, zero);
        __m128i hi = _mm_unpackhi_epi8(pixels, zero);

        // Extract B, G, R channels (in BGRA memory order)
        // lo: B0 G0 R0 A0 B1 G1 R1 A1
        __m128i b_lo = _mm_and_si128(lo, _mm_set1_epi32(0x0000FFFF));
        __m128i g_lo = _mm_and_si128(_mm_srli_epi32(lo, 16), _mm_set1_epi32(0x0000FFFF));
        __m128i b_hi = _mm_and_si128(hi, _mm_set1_epi32(0x0000FFFF));
        __m128i g_hi = _mm_and_si128(_mm_srli_epi32(hi, 16), _mm_set1_epi32(0x0000FFFF));

        // For simplicity, use scalar for accurate grayscale
        // SSE grayscale requires more complex channel extraction
        break;
    }

    // Fall back to scalar for now (grayscale has complex channel access)
    for (; x < width; ++x) {
        XDWORD p = src[x];
        XDWORD a = p & 0xFF000000;
        XDWORD r = (p >> 16) & 0xFF;
        XDWORD g = (p >> 8) & 0xFF;
        XDWORD b = p & 0xFF;
        XDWORD y = (77 * r + 150 * g + 29 * b) >> 8;
        dst[x] = a | (y << 16) | (y << 8) | y;
    }
}

// SSE2: Multiply blend (dst = src * dst / 255)
static void MultiplyBlend_32_SSE(const VxBlitInfo *info) {
    const XDWORD *src = (const XDWORD *)info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;

    const __m128i zero = _mm_setzero_si128();
    const __m128i round = _mm_set1_epi16(128);
    int x = 0;

    for (; x + 4 <= width; x += 4) {
        __m128i srcPix = _mm_loadu_si128((const __m128i *)(src + x));
        __m128i dstPix = _mm_loadu_si128((const __m128i *)(dst + x));

        // Unpack to 16-bit
        __m128i srcLo = _mm_unpacklo_epi8(srcPix, zero);
        __m128i srcHi = _mm_unpackhi_epi8(srcPix, zero);
        __m128i dstLo = _mm_unpacklo_epi8(dstPix, zero);
        __m128i dstHi = _mm_unpackhi_epi8(dstPix, zero);

        // Multiply
        __m128i prodLo = _mm_mullo_epi16(srcLo, dstLo);
        __m128i prodHi = _mm_mullo_epi16(srcHi, dstHi);

        // Divide by 255: (prod + 128 + ((prod + 128) >> 8)) >> 8
        __m128i tmpLo = _mm_add_epi16(prodLo, round);
        __m128i tmpHi = _mm_add_epi16(prodHi, round);
        tmpLo = _mm_add_epi16(tmpLo, _mm_srli_epi16(tmpLo, 8));
        tmpHi = _mm_add_epi16(tmpHi, _mm_srli_epi16(tmpHi, 8));
        __m128i resLo = _mm_srli_epi16(tmpLo, 8);
        __m128i resHi = _mm_srli_epi16(tmpHi, 8);

        // Pack back to 8-bit
        __m128i result = _mm_packus_epi16(resLo, resHi);
        _mm_storeu_si128((__m128i *)(dst + x), result);
    }

    // Handle remaining pixels
    for (; x < width; ++x) {
        XDWORD s = src[x];
        XDWORD d = dst[x];
        XDWORD r = (((s >> 16) & 0xFF) * ((d >> 16) & 0xFF) + 127) / 255;
        XDWORD g = (((s >> 8) & 0xFF) * ((d >> 8) & 0xFF) + 127) / 255;
        XDWORD b = ((s & 0xFF) * (d & 0xFF) + 127) / 255;
        XDWORD a = (((s >> 24) & 0xFF) * ((d >> 24) & 0xFF) + 127) / 255;
        dst[x] = (a << 24) | (r << 16) | (g << 8) | b;
    }
}

#endif // VX_SIMD_SSE2

//------------------------------------------------------------------------------
// Alpha Blitting Functions
//------------------------------------------------------------------------------

static void SetAlpha_8(const VxBlitInfo *info) {
    XBYTE *dst = info->dstLine;
    const int width = info->width;
    const XBYTE alphaValue = (XBYTE)(info->alphaValue & info->dstAlphaMask);
    const XBYTE alphaMaskInv = (XBYTE)info->alphaMaskInv;

    for (int x = 0; x < width; ++x) {
        dst[x] = (dst[x] & alphaMaskInv) | alphaValue;
    }
}

static void SetAlpha_16(const VxBlitInfo *info) {
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;
    const XWORD alphaValue = (XWORD)(info->alphaValue & info->dstAlphaMask);
    const XWORD alphaMaskInv = (XWORD)info->alphaMaskInv;

    for (int x = 0; x < width; ++x) {
        dst[x] = (dst[x] & alphaMaskInv) | alphaValue;
    }
}

static void SetAlpha_32(const VxBlitInfo *info) {
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const XDWORD alphaValue = info->alphaValue & info->dstAlphaMask;
    const XDWORD alphaMaskInv = info->alphaMaskInv;

    for (int x = 0; x < width; ++x) {
        dst[x] = (dst[x] & alphaMaskInv) | alphaValue;
    }
}

static void CopyAlpha_8(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XBYTE *dst = info->dstLine;
    const int width = info->width;
    const int alphaShift = info->alphaShiftDst;
    const XBYTE alphaMask = (XBYTE)info->dstAlphaMask;
    const XBYTE alphaMaskInv = (XBYTE)info->alphaMaskInv;

    for (int x = 0; x < width; ++x) {
        XBYTE alpha = (src[x] << alphaShift) & alphaMask;
        dst[x] = (dst[x] & alphaMaskInv) | alpha;
    }
}

static void CopyAlpha_16(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XWORD *dst = (XWORD *)info->dstLine;
    const int width = info->width;
    const int alphaShift = info->alphaShiftDst;
    const XWORD alphaMask = (XWORD)info->dstAlphaMask;
    const XWORD alphaMaskInv = (XWORD)info->alphaMaskInv;

    // Calculate how many bits the alpha channel has
    XWORD mask = alphaMask;
    int alphaBits = 0;
    while (mask) {
        alphaBits += mask & 1;
        mask >>= 1;
    }

    // For 1-bit alpha (1555): use MSB threshold
    // For 4-bit alpha (4444): use high-nibble (>>4) mapping
    for (int x = 0; x < width; ++x) {
        XWORD alpha = 0;
        if (alphaBits == 1) {
            alpha = ((src[x] >> 7) & 0x1) << alphaShift;
        } else if (alphaBits == 4) {
            alpha = ((XWORD)(src[x] >> 4) << alphaShift) & alphaMask;
        } else {
            // Fallback: scale by shifting
            int scaleShift = 8 - alphaBits;
            alpha = ((XWORD)(src[x] >> scaleShift) << alphaShift) & alphaMask;
        }
        dst[x] = (dst[x] & alphaMaskInv) | alpha;
    }
}

static void CopyAlpha_32(const VxBlitInfo *info) {
    const XBYTE *src = info->srcLine;
    XDWORD *dst = (XDWORD *)info->dstLine;
    const int width = info->width;
    const int alphaShift = info->alphaShiftDst;
    const XDWORD alphaMask = info->dstAlphaMask;
    const XDWORD alphaMaskInv = info->alphaMaskInv;

    for (int x = 0; x < width; ++x) {
        XDWORD alpha = ((XDWORD)src[x] << alphaShift) & alphaMask;
        dst[x] = (dst[x] & alphaMaskInv) | alpha;
    }
}

//------------------------------------------------------------------------------
// Resize Functions (32-bit bilinear)
//------------------------------------------------------------------------------

// Equal Y, Equal X - just copy
static void ResizeHLine_EqualY_EqualX_32Bpp(const VxResizeInfo *info) {
    memcpy(info->dstRow, info->srcRow, 4 * info->w2);
}

// Equal Y, Shrink X - horizontal downscale
static void ResizeHLine_EqualY_ShrinkX_32Bpp(const VxResizeInfo *info) {
    const XDWORD *src = info->srcRow;
    XDWORD *dst = info->dstRow;

    int srcX = 0; // Fixed point 16.16
    for (int x = 0; x < info->w2; ++x) {
        int srcIdx = srcX >> 16;
        int frac = srcX & 0xFFFF;

        if (frac == 0 || srcIdx + 1 >= info->w1) {
            dst[x] = src[srcIdx];
        } else {
            // Bilinear interpolation
            XDWORD p0 = src[srcIdx];
            XDWORD p1 = src[srcIdx + 1];

            XDWORD r = (((p0 >> 16) & 0xFF) * (0x10000 - frac) + ((p1 >> 16) & 0xFF) * frac) >> 16;
            XDWORD g = (((p0 >> 8) & 0xFF) * (0x10000 - frac) + ((p1 >> 8) & 0xFF) * frac) >> 16;
            XDWORD b = ((p0 & 0xFF) * (0x10000 - frac) + (p1 & 0xFF) * frac) >> 16;
            XDWORD a = (((p0 >> 24) & 0xFF) * (0x10000 - frac) + ((p1 >> 24) & 0xFF) * frac) >> 16;

            dst[x] = (a << 24) | (r << 16) | (g << 8) | b;
        }

        srcX += info->wr1;
    }
}

// Equal Y, Grow X - horizontal upscale
static void ResizeHLine_EqualY_GrowX_32Bpp(const VxResizeInfo *info) {
    const XDWORD *src = info->srcRow;
    XDWORD *dst = info->dstRow;

    int srcX = 0; // Fixed point 16.16
    for (int x = 0; x < info->w2; ++x) {
        int srcIdx = srcX >> 16;
        int frac = srcX & 0xFFFF;

        if (frac == 0 || srcIdx + 1 >= info->w1) {
            dst[x] = src[XMin(srcIdx, info->w1 - 1)];
        } else {
            // Bilinear interpolation
            XDWORD p0 = src[srcIdx];
            XDWORD p1 = src[XMin(srcIdx + 1, info->w1 - 1)];

            XDWORD r = (((p0 >> 16) & 0xFF) * (0x10000 - frac) + ((p1 >> 16) & 0xFF) * frac) >> 16;
            XDWORD g = (((p0 >> 8) & 0xFF) * (0x10000 - frac) + ((p1 >> 8) & 0xFF) * frac) >> 16;
            XDWORD b = ((p0 & 0xFF) * (0x10000 - frac) + (p1 & 0xFF) * frac) >> 16;
            XDWORD a = (((p0 >> 24) & 0xFF) * (0x10000 - frac) + ((p1 >> 24) & 0xFF) * frac) >> 16;

            dst[x] = (a << 24) | (r << 16) | (g << 8) | b;
        }

        srcX += info->wr1;
    }
}

// Shrink Y, Equal X - vertical downscale (accumulate rows)
static void ResizeHLine_ShrinkY_EqualX_32Bpp(const VxResizeInfo *info) {
    // Simple row copy for now - full implementation would accumulate multiple rows
    memcpy(info->dstRow, info->srcRow, 4 * info->w2);
}

// Shrink Y, Shrink X - both axes downscale
static void ResizeHLine_ShrinkY_ShrinkX_32Bpp(const VxResizeInfo *info) {
    // Use horizontal shrink
    ResizeHLine_EqualY_ShrinkX_32Bpp(info);
}

// Shrink Y, Grow X - vertical down, horizontal up
static void ResizeHLine_ShrinkY_GrowX_32Bpp(const VxResizeInfo *info) {
    ResizeHLine_EqualY_GrowX_32Bpp(info);
}

// Grow Y, Equal X - vertical upscale (interpolate between rows)
static void ResizeHLine_GrowY_EqualX_32Bpp(const VxResizeInfo *info) {
    memcpy(info->dstRow, info->srcRow, 4 * info->w2);
}

// Grow Y, Shrink X - vertical up, horizontal down
static void ResizeHLine_GrowY_ShrinkX_32Bpp(const VxResizeInfo *info) {
    ResizeHLine_EqualY_ShrinkX_32Bpp(info);
}

// Grow Y, Grow X - both axes upscale
static void ResizeHLine_GrowY_GrowX_32Bpp(const VxResizeInfo *info) {
    ResizeHLine_EqualY_GrowX_32Bpp(info);
}

//------------------------------------------------------------------------------
// VxBlitEngine Implementation
//------------------------------------------------------------------------------

VxBlitEngine::VxBlitEngine() {
    // Clear all tables
    memset(m_GenericBlitTable, 0, sizeof(m_GenericBlitTable));
    memset(m_SpecificBlitTable, 0, sizeof(m_SpecificBlitTable));
    memset(m_SetAlphaTable, 0, sizeof(m_SetAlphaTable));
    memset(m_CopyAlphaTable, 0, sizeof(m_CopyAlphaTable));
    memset(m_PalettedBlitTable, 0, sizeof(m_PalettedBlitTable));
    memset(&m_BlitInfo, 0, sizeof(m_BlitInfo));

    // Build function tables
    BuildGenericTables();
    BuildX86Table();
    BuildSSETable();
}

VxBlitEngine::~VxBlitEngine() {
    // Nothing to free - XArray handles its own memory
}

void VxBlitEngine::BuildGenericTables() {
    // Generic blitting functions indexed by [srcBpp-1][dstBpp-1]
    m_GenericBlitTable[0][0] = CopyLineGeneric_8_8;
    m_GenericBlitTable[0][1] = CopyLineGeneric_8_16;
    m_GenericBlitTable[0][2] = CopyLineGeneric_8_24;
    m_GenericBlitTable[0][3] = CopyLineGeneric_8_32;

    m_GenericBlitTable[1][0] = CopyLineGeneric_16_8;
    m_GenericBlitTable[1][1] = CopyLineGeneric_16_16;
    m_GenericBlitTable[1][2] = CopyLineGeneric_16_24;
    m_GenericBlitTable[1][3] = CopyLineGeneric_16_32;

    m_GenericBlitTable[2][0] = CopyLineGeneric_24_8;
    m_GenericBlitTable[2][1] = CopyLineGeneric_24_16;
    m_GenericBlitTable[2][2] = CopyLineGeneric_24_24;
    m_GenericBlitTable[2][3] = CopyLineGeneric_24_32;

    m_GenericBlitTable[3][0] = CopyLineGeneric_32_8;
    m_GenericBlitTable[3][1] = CopyLineGeneric_32_16;
    m_GenericBlitTable[3][2] = CopyLineGeneric_32_24;
    m_GenericBlitTable[3][3] = CopyLineGeneric_32_32;

    // Alpha functions
    m_SetAlphaTable[0] = SetAlpha_8;
    m_SetAlphaTable[1] = SetAlpha_16;
    m_SetAlphaTable[2] = nullptr; // 24-bit rarely has alpha
    m_SetAlphaTable[3] = SetAlpha_32;

    m_CopyAlphaTable[0] = CopyAlpha_8;
    m_CopyAlphaTable[1] = CopyAlpha_16;
    m_CopyAlphaTable[2] = nullptr;
    m_CopyAlphaTable[3] = CopyAlpha_32;
}

void VxBlitEngine::BuildX86Table() {
    // Specific format conversion functions for common cases

    // 32-bit ARGB (format 1) to other formats
    m_SpecificBlitTable[1][2] = CopyLine_32ARGB_32RGB;    // ARGB -> RGB (32)
    m_SpecificBlitTable[1][3] = CopyLine_32ARGB_24RGB;    // ARGB -> RGB (24)
    m_SpecificBlitTable[1][4] = CopyLine_32ARGB_565RGB;   // ARGB -> 565
    m_SpecificBlitTable[1][5] = CopyLine_32ARGB_555RGB;   // ARGB -> 555
    m_SpecificBlitTable[1][6] = CopyLine_32ARGB_1555ARGB; // ARGB -> 1555
    m_SpecificBlitTable[1][7] = CopyLine_32ARGB_4444ARGB; // ARGB -> 4444
    m_SpecificBlitTable[1][10] = CopyLine_32ARGB_32ABGR;  // ARGB -> ABGR
    m_SpecificBlitTable[1][11] = CopyLine_32ARGB_32RGBA;  // ARGB -> RGBA
    m_SpecificBlitTable[1][12] = CopyLine_32ARGB_32BGRA;  // ARGB -> BGRA

    // 32-bit RGB (format 2) to ARGB
    m_SpecificBlitTable[2][1] = CopyLine_32RGB_32ARGB;

    // 24-bit RGB (format 3) to 32-bit
    m_SpecificBlitTable[3][1] = CopyLine_24RGB_32ARGB;

    // 16-bit to 32-bit
    m_SpecificBlitTable[4][1] = CopyLine_565RGB_32ARGB;
    m_SpecificBlitTable[5][1] = CopyLine_555RGB_32ARGB;
    m_SpecificBlitTable[6][1] = CopyLine_1555ARGB_32ARGB;
    m_SpecificBlitTable[7][1] = CopyLine_4444ARGB_32ARGB;

    // 32-bit ABGR (format 10) to other formats
    m_SpecificBlitTable[10][1] = CopyLine_32ABGR_32ARGB;  // ABGR -> ARGB

    // 32-bit RGBA (format 11) to other formats
    m_SpecificBlitTable[11][1] = CopyLine_32RGBA_32ARGB;  // RGBA -> ARGB

    // 32-bit BGRA (format 12) to other formats
    m_SpecificBlitTable[12][1] = CopyLine_32BGRA_32ARGB;  // BGRA -> ARGB

    // Paletted image functions
    m_PalettedBlitTable[0][0] = CopyLine_Paletted8_8;     // 8-bit pal -> 8-bit
    m_PalettedBlitTable[0][1] = CopyLine_Paletted8_565RGB; // 8-bit pal -> 16-bit 565
    m_PalettedBlitTable[0][2] = CopyLine_Paletted8_24RGB;  // 8-bit pal -> 24-bit
    m_PalettedBlitTable[0][3] = CopyLine_Paletted8_32ARGB; // 8-bit pal -> 32-bit
}

void VxBlitEngine::BuildSSETable() {
#if defined(VX_SIMD_SSE2)
    // SSE2-optimized functions override scalar versions for better performance

    // 32-bit ARGB <-> RGB conversions
    m_SpecificBlitTable[1][2] = CopyLine_32ARGB_32RGB_SSE;    // ARGB -> RGB (32)
    m_SpecificBlitTable[2][1] = CopyLine_32RGB_32ARGB_SSE;    // RGB -> ARGB (32)

    // 32-bit to 24-bit and vice versa
    m_SpecificBlitTable[1][3] = CopyLine_32ARGB_24RGB_SSE;    // ARGB -> RGB (24)
    m_SpecificBlitTable[3][1] = CopyLine_24RGB_32ARGB_SSE;    // RGB (24) -> ARGB

    // 32-bit to 16-bit conversions
    m_SpecificBlitTable[1][4] = CopyLine_32ARGB_565RGB_SSE;   // ARGB -> 565
    m_SpecificBlitTable[1][5] = CopyLine_32ARGB_555RGB_SSE;   // ARGB -> 555
    m_SpecificBlitTable[1][6] = CopyLine_32ARGB_1555ARGB_SSE; // ARGB -> 1555
    m_SpecificBlitTable[1][7] = CopyLine_32ARGB_4444ARGB_SSE; // ARGB -> 4444

    // 32-bit channel swap conversions (SSE optimized)
    m_SpecificBlitTable[1][10] = CopyLine_32ARGB_32ABGR_SSE;  // ARGB -> ABGR
    m_SpecificBlitTable[1][11] = CopyLine_32ARGB_32RGBA_SSE;  // ARGB -> RGBA
    m_SpecificBlitTable[1][12] = CopyLine_32ARGB_32BGRA_SSE;  // ARGB -> BGRA
    m_SpecificBlitTable[10][1] = CopyLine_32ABGR_32ARGB_SSE;  // ABGR -> ARGB
    m_SpecificBlitTable[11][1] = CopyLine_32RGBA_32ARGB_SSE;  // RGBA -> ARGB
    m_SpecificBlitTable[12][1] = CopyLine_32BGRA_32ARGB_SSE;  // BGRA -> ARGB

    // 16-bit to 32-bit conversions
    m_SpecificBlitTable[4][1] = CopyLine_565RGB_32ARGB_SSE;   // 565 -> ARGB
    m_SpecificBlitTable[5][1] = CopyLine_555RGB_32ARGB_SSE;   // 555 -> ARGB
    m_SpecificBlitTable[6][1] = CopyLine_1555ARGB_32ARGB_SSE; // 1555 -> ARGB
    m_SpecificBlitTable[7][1] = CopyLine_4444ARGB_32ARGB_SSE; // 4444 -> ARGB

    // Alpha functions
    m_SetAlphaTable[3] = SetAlpha_32_SSE;
    m_CopyAlphaTable[3] = CopyAlpha_32_SSE;

    // Paletted with SSE optimization
    m_PalettedBlitTable[0][3] = CopyLine_Paletted8_32ARGB_SSE; // 8-bit pal -> 32-bit (SSE)
#endif
}

VX_PIXELFORMAT VxBlitEngine::GetPixelFormat(const VxImageDescEx &desc) {
    // Check for DXT/bump formats first
    if (desc.Flags >= _DXT1 && desc.Flags < MAX_PIXEL_FORMATS) {
        return static_cast<VX_PIXELFORMAT>(desc.Flags);
    }

    // Search through standard formats
    const XDWORD bpp = desc.BitsPerPixel;
    const XDWORD rmask = desc.RedMask;
    const XDWORD gmask = desc.GreenMask;
    const XDWORD bmask = desc.BlueMask;
    const XDWORD amask = desc.AlphaMask;

    for (int i = 1; i < NUM_PIXEL_FORMATS && i < _DXT1; ++i) {
        const PixelFormatDef &fmt = s_PixelFormats[i];
        if (fmt.BitsPerPixel == bpp &&
            fmt.RedMask == rmask &&
            fmt.GreenMask == gmask &&
            fmt.BlueMask == bmask &&
            fmt.AlphaMask == amask) {
            return static_cast<VX_PIXELFORMAT>(i);
        }
    }

    return UNKNOWN_PF;
}

void VxBlitEngine::ConvertPixelFormat(VX_PIXELFORMAT Pf, VxImageDescEx &desc) {
    if (Pf <= UNKNOWN_PF || Pf >= MAX_PIXEL_FORMATS) return;

    const PixelFormatDef &fmt = s_PixelFormats[Pf];
    desc.BitsPerPixel = fmt.BitsPerPixel;
    desc.RedMask = fmt.RedMask;
    desc.GreenMask = fmt.GreenMask;
    desc.BlueMask = fmt.BlueMask;
    desc.AlphaMask = fmt.AlphaMask;

    if (Pf >= _DXT1) {
        desc.Flags = Pf;
    } else {
        desc.Flags = 0;
    }
}

const char *VxBlitEngine::PixelFormat2String(VX_PIXELFORMAT Pf) {
    if (Pf < 0 || Pf >= NUM_PIXEL_FORMATS) {
        return "";
    }
    return s_PixelFormats[Pf].Description;
}

void VxBlitEngine::SetupBlitInfo(VxBlitInfo &info, const VxImageDescEx &src_desc,
                                  const VxImageDescEx &dst_desc) {
    info.srcBytesPerPixel = src_desc.BitsPerPixel / 8;
    info.dstBytesPerPixel = dst_desc.BitsPerPixel / 8;
    info.width = src_desc.Width;
    info.copyBytes = src_desc.Width * info.srcBytesPerPixel;

    info.srcRedMask = src_desc.RedMask;
    info.srcGreenMask = src_desc.GreenMask;
    info.srcBlueMask = src_desc.BlueMask;
    info.srcAlphaMask = src_desc.AlphaMask;

    info.dstRedMask = dst_desc.RedMask;
    info.dstGreenMask = dst_desc.GreenMask;
    info.dstBlueMask = dst_desc.BlueMask;
    info.dstAlphaMask = dst_desc.AlphaMask;

    // Pre-compute shifts (position of LSB)
    info.redShiftSrc = static_cast<int>(GetBitShiftLocal(src_desc.RedMask));
    info.greenShiftSrc = static_cast<int>(GetBitShiftLocal(src_desc.GreenMask));
    info.blueShiftSrc = static_cast<int>(GetBitShiftLocal(src_desc.BlueMask));
    info.alphaShiftSrc = static_cast<int>(GetBitShiftLocal(src_desc.AlphaMask));

    info.redShiftDst = static_cast<int>(GetBitShiftLocal(dst_desc.RedMask));
    info.greenShiftDst = static_cast<int>(GetBitShiftLocal(dst_desc.GreenMask));
    info.blueShiftDst = static_cast<int>(GetBitShiftLocal(dst_desc.BlueMask));
    info.alphaShiftDst = static_cast<int>(GetBitShiftLocal(dst_desc.AlphaMask));

    // Compute bit widths for each channel
    info.redBitsSrc = static_cast<int>(GetBitCountLocal(src_desc.RedMask));
    info.greenBitsSrc = static_cast<int>(GetBitCountLocal(src_desc.GreenMask));
    info.blueBitsSrc = static_cast<int>(GetBitCountLocal(src_desc.BlueMask));
    info.alphaBitsSrc = static_cast<int>(GetBitCountLocal(src_desc.AlphaMask));

    info.redBitsDst = static_cast<int>(GetBitCountLocal(dst_desc.RedMask));
    info.greenBitsDst = static_cast<int>(GetBitCountLocal(dst_desc.GreenMask));
    info.blueBitsDst = static_cast<int>(GetBitCountLocal(dst_desc.BlueMask));
    info.alphaBitsDst = static_cast<int>(GetBitCountLocal(dst_desc.AlphaMask));

    info.alphaMaskInv = ~dst_desc.AlphaMask;

    info.colorMap = src_desc.ColorMap;
    info.colorMapEntries = src_desc.ColorMapEntries;
    info.bytesPerColorEntry = src_desc.BytesPerColorEntry;
}

VxBlitLineFunc VxBlitEngine::GetBlitFunction(const VxImageDescEx &src_desc,
                                              const VxImageDescEx &dst_desc) {
    VX_PIXELFORMAT srcFmt = GetPixelFormat(src_desc);
    VX_PIXELFORMAT dstFmt = GetPixelFormat(dst_desc);

    // Check for paletted source image
    if (src_desc.ColorMapEntries > 0 && src_desc.ColorMap != nullptr) {
        int srcBpp = src_desc.BitsPerPixel / 8;
        int dstBpp = dst_desc.BitsPerPixel / 8;
        
        // Only 8-bit paletted sources are currently supported
        if (srcBpp == 1 && dstBpp >= 1 && dstBpp <= 4) {
            // For 16-bit destinations with alpha (1555, 4444), use alpha-aware function
            if (dstBpp == 2 && dst_desc.AlphaMask != 0) {
                return CopyLine_Paletted8_16Alpha;
            }
            VxBlitLineFunc func = m_PalettedBlitTable[0][dstBpp - 1];
            if (func) return func;
        }
    }

    // Same format - use memcpy
    if (srcFmt == dstFmt && srcFmt != UNKNOWN_PF) {
        return CopyLineSame;
    }

    // Check for specific optimized function
    if (srcFmt > UNKNOWN_PF && srcFmt < FORMAT_TABLE_SIZE &&
        dstFmt > UNKNOWN_PF && dstFmt < FORMAT_TABLE_SIZE) {
        VxBlitLineFunc func = m_SpecificBlitTable[srcFmt][dstFmt];
        if (func) return func;
    }

    // DXT formats not supported for direct blitting
    if ((srcFmt >= _DXT1 && srcFmt <= _DXT5) ||
        (dstFmt >= _DXT1 && dstFmt <= _DXT5)) {
        return nullptr;
    }

    // Fall back to generic function
    int srcBpp = src_desc.BitsPerPixel / 8;
    int dstBpp = dst_desc.BitsPerPixel / 8;

    if (srcBpp >= 1 && srcBpp <= 4 && dstBpp >= 1 && dstBpp <= 4) {
        return m_GenericBlitTable[srcBpp - 1][dstBpp - 1];
    }

    return nullptr;
}

VxBlitLineFunc VxBlitEngine::GetSetAlphaFunction(const VxImageDescEx &dst_desc) {
    if (!dst_desc.AlphaMask) return nullptr;

    int bpp = dst_desc.BitsPerPixel / 8;
    if (bpp >= 1 && bpp <= 4) {
        return m_SetAlphaTable[bpp - 1];
    }
    return nullptr;
}

VxBlitLineFunc VxBlitEngine::GetCopyAlphaFunction(const VxImageDescEx &dst_desc) {
    if (!dst_desc.AlphaMask) return nullptr;

    int bpp = dst_desc.BitsPerPixel / 8;
    if (bpp >= 1 && bpp <= 4) {
        return m_CopyAlphaTable[bpp - 1];
    }
    return nullptr;
}

void VxBlitEngine::DoBlit(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    // Original binary precondition: only allow if 32-bit OR same dimensions
    // Non-32-bit images with different dimensions are rejected
    if (src_desc.BitsPerPixel != 32) {
        if (dst_desc.Width != src_desc.Width || dst_desc.Height != src_desc.Height) {
            return;
        }
    }

    if (!src_desc.Image) return;
    if (!dst_desc.Image) return;

    // Get blit function
    VxBlitLineFunc blitFunc = GetBlitFunction(src_desc, dst_desc);
    if (!blitFunc) return;

    // Setup blit info
    SetupBlitInfo(m_BlitInfo, src_desc, dst_desc);
    m_BlitInfo.srcBytesPerLine = src_desc.BytesPerLine;
    m_BlitInfo.dstBytesPerLine = dst_desc.BytesPerLine;

    // Handle quantization if destination is paletted and source is not
    // (paletted-to-paletted uses direct index copy, not quantization)
    if (dst_desc.ColorMapEntries > 0 && src_desc.ColorMapEntries == 0) {
        if (!QuantizeImage(src_desc, dst_desc)) {
            return;
        }
    }

    // Check if resize is needed (only for 32-bit)
    if (dst_desc.Width != src_desc.Width || dst_desc.Height != src_desc.Height) {
        // At this point src must be 32-bit (per precondition above)
        DoBlitWithResize(src_desc, dst_desc, blitFunc);
        return;
    }

    // Same-size blit: process each scanline
    if (src_desc.Height <= 0) return;

    const XBYTE *srcRow = src_desc.Image;
    XBYTE *dstRow = dst_desc.Image;

    for (int y = 0; y < src_desc.Height; ++y) {
        m_BlitInfo.srcLine = srcRow;
        m_BlitInfo.dstLine = dstRow;

        blitFunc(&m_BlitInfo);

        srcRow += src_desc.BytesPerLine;
        dstRow += dst_desc.BytesPerLine;
    }
}

void VxBlitEngine::DoBlitUpsideDown(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    if (src_desc.Width != dst_desc.Width) return;
    if (src_desc.Height != dst_desc.Height) return;
    if (!src_desc.Image || !dst_desc.Image) return;

    VxBlitLineFunc blitFunc = GetBlitFunction(src_desc, dst_desc);
    if (!blitFunc) return;

    SetupBlitInfo(m_BlitInfo, src_desc, dst_desc);
    m_BlitInfo.srcBytesPerLine = src_desc.BytesPerLine;
    m_BlitInfo.dstBytesPerLine = dst_desc.BytesPerLine;

    // Handle quantization if destination is paletted and source is not
    // (paletted-to-paletted uses direct index copy, not quantization)
    if (dst_desc.ColorMapEntries > 0 && src_desc.ColorMapEntries == 0) {
        if (!QuantizeImage(src_desc, dst_desc)) {
            return;
        }
    }

    // Source starts at last row, destination at first row (matching original binary)
    const XBYTE *srcRow = src_desc.Image + (src_desc.Height - 1) * src_desc.BytesPerLine;
    XBYTE *dstRow = dst_desc.Image;

    for (int y = 0; y < src_desc.Height; ++y) {
        m_BlitInfo.srcLine = srcRow;
        m_BlitInfo.dstLine = dstRow;

        blitFunc(&m_BlitInfo);

        srcRow -= src_desc.BytesPerLine;  // Source goes backwards
        dstRow += dst_desc.BytesPerLine;  // Destination goes forward
    }
}

void VxBlitEngine::DoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE AlphaValue) {
    if (!dst_desc.AlphaMask) return;
    if (!dst_desc.Image) return;
    if (dst_desc.Width < 0) return;

    VxBlitLineFunc alphaFunc = GetSetAlphaFunction(dst_desc);
    if (!alphaFunc) return;

    // Setup blit info for alpha operation
    m_BlitInfo.dstBytesPerPixel = dst_desc.BitsPerPixel / 8;
    m_BlitInfo.dstAlphaMask = dst_desc.AlphaMask;
    m_BlitInfo.alphaMaskInv = ~dst_desc.AlphaMask;
    m_BlitInfo.width = dst_desc.Width;
    m_BlitInfo.dstBytesPerLine = dst_desc.BytesPerLine;

    // Scale 8-bit alpha to target bit depth
    // Get the shift and bit count of the alpha mask
    XDWORD alphaMask = dst_desc.AlphaMask;
    int alphaShift = static_cast<int>(GetBitShiftLocal(alphaMask));
    
    // Count bits in the alpha mask
    XDWORD tempMask = alphaMask >> alphaShift;
    int alphaBits = 0;
    while (tempMask & 1) {
        ++alphaBits;
        tempMask >>= 1;
    }
    
    // Scale the 8-bit alpha value to the target bit depth
    XDWORD scaledAlpha;
    if (alphaBits >= 8) {
        // Direct placement for 8+ bit alpha
        scaledAlpha = (XDWORD)AlphaValue << alphaShift;
    } else {
        // Scale down from 8-bit to target bits by taking top bits
        int shiftAmount = 8 - alphaBits;
        scaledAlpha = ((XDWORD)AlphaValue >> shiftAmount) << alphaShift;
    }
    m_BlitInfo.alphaValue = scaledAlpha & alphaMask;

    XBYTE *row = dst_desc.Image;
    for (int y = 0; y < dst_desc.Height; ++y) {
        m_BlitInfo.dstLine = row;
        alphaFunc(&m_BlitInfo);
        row += dst_desc.BytesPerLine;
    }
}

void VxBlitEngine::DoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE *AlphaValues) {
    if (!dst_desc.AlphaMask || !dst_desc.Image || !AlphaValues) return;
    if (dst_desc.Width < 0) return;

    VxBlitLineFunc alphaFunc = GetCopyAlphaFunction(dst_desc);
    if (!alphaFunc) return;

    m_BlitInfo.dstBytesPerPixel = dst_desc.BitsPerPixel / 8;
    m_BlitInfo.dstAlphaMask = dst_desc.AlphaMask;
    m_BlitInfo.alphaMaskInv = ~dst_desc.AlphaMask;
    m_BlitInfo.width = dst_desc.Width;
    m_BlitInfo.dstBytesPerLine = dst_desc.BytesPerLine;
    m_BlitInfo.alphaShiftDst = static_cast<int>(GetBitShiftLocal(dst_desc.AlphaMask));

    XBYTE *row = dst_desc.Image;
    const XBYTE *alphaRow = AlphaValues;

    for (int y = 0; y < dst_desc.Height; ++y) {
        m_BlitInfo.srcLine = alphaRow;
        m_BlitInfo.dstLine = row;

        alphaFunc(&m_BlitInfo);

        row += dst_desc.BytesPerLine;
        alphaRow += dst_desc.Width;
    }
}

//------------------------------------------------------------------------------
// FillImage - Fill image with solid color (uses SIMD when available)
//------------------------------------------------------------------------------

void VxBlitEngine::FillImage(const VxImageDescEx &dst_desc, XDWORD color) {
    if (!dst_desc.Image) return;
    if (dst_desc.Width <= 0 || dst_desc.Height <= 0) return;

    const int bpp = dst_desc.BitsPerPixel / 8;
    XBYTE *row = dst_desc.Image;

    if (bpp == 4) {
        // 32-bit fill with SIMD
        for (int y = 0; y < dst_desc.Height; ++y) {
#if defined(VX_SIMD_SSE2)
            FillLine_32_SSE((XDWORD *)row, dst_desc.Width, color);
#else
            XDWORD *dst = (XDWORD *)row;
            for (int x = 0; x < dst_desc.Width; ++x) {
                dst[x] = color;
            }
#endif
            row += dst_desc.BytesPerLine;
        }
    } else if (bpp == 3) {
        // 24-bit fill
        XBYTE b = (XBYTE)(color & 0xFF);
        XBYTE g = (XBYTE)((color >> 8) & 0xFF);
        XBYTE r = (XBYTE)((color >> 16) & 0xFF);
        for (int y = 0; y < dst_desc.Height; ++y) {
            XBYTE *dst = row;
            for (int x = 0; x < dst_desc.Width; ++x) {
                dst[0] = b;
                dst[1] = g;
                dst[2] = r;
                dst += 3;
            }
            row += dst_desc.BytesPerLine;
        }
    } else if (bpp == 2) {
        // 16-bit fill with SIMD
        XWORD color16 = (XWORD)(color & 0xFFFF);
        for (int y = 0; y < dst_desc.Height; ++y) {
#if defined(VX_SIMD_SSE2)
            FillLine_16_SSE((XWORD *)row, dst_desc.Width, color16);
#else
            XWORD *dst = (XWORD *)row;
            for (int x = 0; x < dst_desc.Width; ++x) {
                dst[x] = color16;
            }
#endif
            row += dst_desc.BytesPerLine;
        }
    } else if (bpp == 1) {
        // 8-bit fill
        XBYTE color8 = (XBYTE)(color & 0xFF);
        for (int y = 0; y < dst_desc.Height; ++y) {
            memset(row, color8, dst_desc.Width);
            row += dst_desc.BytesPerLine;
        }
    }
}

//------------------------------------------------------------------------------
// PremultiplyAlpha - Convert to premultiplied alpha
//------------------------------------------------------------------------------

void VxBlitEngine::PremultiplyAlpha(const VxImageDescEx &desc) {
    if (!desc.Image) return;
    if (desc.BitsPerPixel != 32 || desc.AlphaMask == 0) return;
    if (desc.Width <= 0 || desc.Height <= 0) return;

    // Setup blitinfo for the premultiply function
    VxBlitInfo info = {};
    info.width = desc.Width;

#if defined(VX_SIMD_SSE2)
    VxBlitLineFunc premulFunc = PremultiplyAlpha_32ARGB_SSE;
#else
    VxBlitLineFunc premulFunc = PremultiplyAlpha_32ARGB;
#endif

    XBYTE *row = desc.Image;
    for (int y = 0; y < desc.Height; ++y) {
        info.srcLine = row;
        info.dstLine = row;  // In-place
        premulFunc(&info);
        row += desc.BytesPerLine;
    }
}

//------------------------------------------------------------------------------
// UnpremultiplyAlpha - Convert from premultiplied alpha back to straight
//------------------------------------------------------------------------------

void VxBlitEngine::UnpremultiplyAlpha(const VxImageDescEx &desc) {
    if (!desc.Image) return;
    if (desc.BitsPerPixel != 32 || desc.AlphaMask == 0) return;
    if (desc.Width <= 0 || desc.Height <= 0) return;

    VxBlitInfo info = {};
    info.width = desc.Width;

#if defined(VX_SIMD_SSE2)
    VxBlitLineFunc unpremulFunc = UnpremultiplyAlpha_32ARGB_SSE;
#else
    VxBlitLineFunc unpremulFunc = UnpremultiplyAlpha_32ARGB;
#endif

    XBYTE *row = desc.Image;
    for (int y = 0; y < desc.Height; ++y) {
        info.srcLine = row;
        info.dstLine = row;  // In-place
        unpremulFunc(&info);
        row += desc.BytesPerLine;
    }
}

//------------------------------------------------------------------------------
// SwapRedBlue - Swap R and B channels (ARGB <-> ABGR)
//------------------------------------------------------------------------------

void VxBlitEngine::SwapRedBlue(const VxImageDescEx &desc) {
    if (!desc.Image) return;
    if (desc.BitsPerPixel != 32) return;
    if (desc.Width <= 0 || desc.Height <= 0) return;

    VxBlitInfo info = {};
    info.width = desc.Width;

#if defined(VX_SIMD_SSE2)
    VxBlitLineFunc swapFunc = CopyLine_32ARGB_32ABGR_SSE;
#else
    VxBlitLineFunc swapFunc = CopyLine_32ARGB_32ABGR;
#endif

    XBYTE *row = desc.Image;
    for (int y = 0; y < desc.Height; ++y) {
        info.srcLine = row;
        info.dstLine = row;  // In-place
        swapFunc(&info);
        row += desc.BytesPerLine;
    }
}

//------------------------------------------------------------------------------
// ClearAlpha - Set alpha channel to 0 (fully transparent)
//------------------------------------------------------------------------------

void VxBlitEngine::ClearAlpha(const VxImageDescEx &desc) {
    if (!desc.Image) return;
    if (desc.BitsPerPixel != 32) return;
    if (desc.Width <= 0 || desc.Height <= 0) return;

    VxBlitInfo info = {};
    info.width = desc.Width;

    XBYTE *row = desc.Image;
    for (int y = 0; y < desc.Height; ++y) {
        info.dstLine = row;
#if defined(VX_SIMD_SSE2)
        ClearAlpha_32_SSE(&info);
#else
        XDWORD *dst = (XDWORD *)row;
        for (int x = 0; x < desc.Width; ++x) {
            dst[x] &= 0x00FFFFFF;
        }
#endif
        row += desc.BytesPerLine;
    }
}

//------------------------------------------------------------------------------
// SetFullAlpha - Set alpha channel to 255 (fully opaque)
//------------------------------------------------------------------------------

void VxBlitEngine::SetFullAlpha(const VxImageDescEx &desc) {
    if (!desc.Image) return;
    if (desc.BitsPerPixel != 32) return;
    if (desc.Width <= 0 || desc.Height <= 0) return;

    VxBlitInfo info = {};
    info.width = desc.Width;

    XBYTE *row = desc.Image;
    for (int y = 0; y < desc.Height; ++y) {
        info.dstLine = row;
#if defined(VX_SIMD_SSE2)
        SetFullAlpha_32_SSE(&info);
#else
        XDWORD *dst = (XDWORD *)row;
        for (int x = 0; x < desc.Width; ++x) {
            dst[x] |= 0xFF000000;
        }
#endif
        row += desc.BytesPerLine;
    }
}

//------------------------------------------------------------------------------
// InvertColors - Invert RGB channels (keep alpha)
//------------------------------------------------------------------------------

void VxBlitEngine::InvertColors(const VxImageDescEx &desc) {
    if (!desc.Image) return;
    if (desc.BitsPerPixel != 32) return;
    if (desc.Width <= 0 || desc.Height <= 0) return;

    VxBlitInfo info = {};
    info.width = desc.Width;

    XBYTE *row = desc.Image;
    for (int y = 0; y < desc.Height; ++y) {
        info.dstLine = row;
#if defined(VX_SIMD_SSE2)
        InvertColors_32_SSE(&info);
#else
        XDWORD *dst = (XDWORD *)row;
        for (int x = 0; x < desc.Width; ++x) {
            dst[x] ^= 0x00FFFFFF;
        }
#endif
        row += desc.BytesPerLine;
    }
}

//------------------------------------------------------------------------------
// ConvertToGrayscale - Convert to grayscale using luminance formula
//------------------------------------------------------------------------------

void VxBlitEngine::ConvertToGrayscale(const VxImageDescEx &desc) {
    if (!desc.Image) return;
    if (desc.BitsPerPixel != 32) return;
    if (desc.Width <= 0 || desc.Height <= 0) return;

    VxBlitInfo info = {};
    info.width = desc.Width;

    XBYTE *row = desc.Image;
    for (int y = 0; y < desc.Height; ++y) {
        info.srcLine = row;
        info.dstLine = row;
#if defined(VX_SIMD_SSE2)
        Grayscale_32_SSE(&info);
#else
        XDWORD *dst = (XDWORD *)row;
        for (int x = 0; x < desc.Width; ++x) {
            XDWORD p = dst[x];
            XDWORD a = p & 0xFF000000;
            XDWORD r = (p >> 16) & 0xFF;
            XDWORD g = (p >> 8) & 0xFF;
            XDWORD b = p & 0xFF;
            XDWORD y_val = (77 * r + 150 * g + 29 * b) >> 8;
            dst[x] = a | (y_val << 16) | (y_val << 8) | y_val;
        }
#endif
        row += desc.BytesPerLine;
    }
}

//------------------------------------------------------------------------------
// MultiplyBlend - Perform multiply blend: dst = src * dst / 255
//------------------------------------------------------------------------------

void VxBlitEngine::MultiplyBlend(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    if (!src_desc.Image || !dst_desc.Image) return;
    if (src_desc.BitsPerPixel != 32 || dst_desc.BitsPerPixel != 32) return;
    if (src_desc.Width != dst_desc.Width || src_desc.Height != dst_desc.Height) return;
    if (src_desc.Width <= 0 || src_desc.Height <= 0) return;

    VxBlitInfo info = {};
    info.width = src_desc.Width;

    const XBYTE *srcRow = src_desc.Image;
    XBYTE *dstRow = dst_desc.Image;
    for (int y = 0; y < src_desc.Height; ++y) {
        info.srcLine = srcRow;
        info.dstLine = dstRow;
#if defined(VX_SIMD_SSE2)
        MultiplyBlend_32_SSE(&info);
#else
        const XDWORD *src = (const XDWORD *)srcRow;
        XDWORD *dst = (XDWORD *)dstRow;
        for (int x = 0; x < src_desc.Width; ++x) {
            XDWORD s = src[x];
            XDWORD d = dst[x];
            XDWORD r = (((s >> 16) & 0xFF) * ((d >> 16) & 0xFF) + 127) / 255;
            XDWORD g = (((s >> 8) & 0xFF) * ((d >> 8) & 0xFF) + 127) / 255;
            XDWORD b = ((s & 0xFF) * (d & 0xFF) + 127) / 255;
            XDWORD a = (((s >> 24) & 0xFF) * ((d >> 24) & 0xFF) + 127) / 255;
            dst[x] = (a << 24) | (r << 16) | (g << 8) | b;
        }
#endif
        srcRow += src_desc.BytesPerLine;
        dstRow += dst_desc.BytesPerLine;
    }
}

//------------------------------------------------------------------------------
// DoBlitWithResize - Internal blit with resize (matches original binary)
//------------------------------------------------------------------------------

void VxBlitEngine::DoBlitWithResize(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc, VxBlitLineFunc blitFunc) {
    // Store destination width for line blit
    m_BlitInfo.width = dst_desc.Width;
    m_BlitInfo.copyBytes = dst_desc.Width * 4;

    // Ensure resize buffer is large enough
    size_t bufferSize = src_desc.Width + dst_desc.Width + 1;
    if (m_ResizeBuffer.Size() < bufferSize) {
        m_ResizeBuffer.Reserve(bufferSize);
    }

    // Setup resize info (fixed-point 16.16)
    VxResizeInfo resizeInfo;
    resizeInfo.w1 = src_desc.Width;
    resizeInfo.w2 = dst_desc.Width;
    resizeInfo.h1 = src_desc.Height;
    resizeInfo.h2 = dst_desc.Height;
    resizeInfo.wr1 = (src_desc.Width << 16) / dst_desc.Width;
    resizeInfo.hr1 = (src_desc.Height << 16) / dst_desc.Height;
    resizeInfo.srcPitch = src_desc.BytesPerLine >> 2;  // Source pitch in DWORDs

    // Intermediate buffer: resized horizontal line stored here before format conversion
    resizeInfo.dstRow = m_ResizeBuffer.Begin() + src_desc.Width;  // Output buffer for resize
    
    // Setup pointers for the blit function
    m_BlitInfo.srcLine = (const XBYTE *)resizeInfo.dstRow;
    XBYTE *dstRow = dst_desc.Image;

    const XDWORD *srcImage = (const XDWORD *)src_desc.Image;
    int srcHeight = src_desc.Height;
    int dstHeight = dst_desc.Height;

    // Select resize function based on X dimension relationship
    typedef void (*ResizeLineFunc)(const VxResizeInfo *);
    ResizeLineFunc growYFunc, shrinkYFunc, equalYFunc;

    if (src_desc.Width == dst_desc.Width) {
        growYFunc = ResizeHLine_GrowY_EqualX_32Bpp;
        shrinkYFunc = ResizeHLine_ShrinkY_EqualX_32Bpp;
        equalYFunc = ResizeHLine_EqualY_EqualX_32Bpp;
    } else if (src_desc.Width < dst_desc.Width) {
        growYFunc = ResizeHLine_GrowY_GrowX_32Bpp;
        shrinkYFunc = ResizeHLine_ShrinkY_GrowX_32Bpp;
        equalYFunc = ResizeHLine_EqualY_GrowX_32Bpp;
    } else {
        growYFunc = ResizeHLine_GrowY_ShrinkX_32Bpp;
        shrinkYFunc = ResizeHLine_ShrinkY_ShrinkX_32Bpp;
        equalYFunc = ResizeHLine_EqualY_ShrinkX_32Bpp;
    }

    if (srcHeight < dstHeight) {
        // Growing Y: interpolate vertically
        int yAccum = 0;
        int yMax = (srcHeight << 16) - 0x10000;

        // Phase 1: while we haven't reached the last source row
        while (yAccum < yMax && yAccum >= 0) {
            int srcY = yAccum >> 16;
            resizeInfo.srcRow = srcImage + srcY * resizeInfo.srcPitch;
            growYFunc(&resizeInfo);

            // Blit the resized line to destination
            m_BlitInfo.dstLine = dstRow;
            blitFunc(&m_BlitInfo);

            dstRow += dst_desc.BytesPerLine;
            yAccum += resizeInfo.hr1;
        }

        // Phase 2: fill remaining rows with last source row (equal Y copy)
        if ((yAccum >> 16) < dstHeight) {
            resizeInfo.srcRow = srcImage + (srcHeight - 1) * resizeInfo.srcPitch;
            while ((yAccum >> 16) < srcHeight) {
                equalYFunc(&resizeInfo);

                m_BlitInfo.dstLine = dstRow;
                blitFunc(&m_BlitInfo);

                dstRow += dst_desc.BytesPerLine;
                yAccum += resizeInfo.hr1;
            }
        }
    } else if (srcHeight > dstHeight) {
        // Shrinking Y: skip source rows
        int yAccum = 0;
        for (int y = 0; y < dstHeight; ++y) {
            int srcY = yAccum >> 16;
            resizeInfo.srcRow = srcImage + srcY * resizeInfo.srcPitch;
            shrinkYFunc(&resizeInfo);

            m_BlitInfo.dstLine = dstRow;
            blitFunc(&m_BlitInfo);

            dstRow += dst_desc.BytesPerLine;
            yAccum += resizeInfo.hr1;
        }
    } else {
        // Equal Y: 1:1 vertical mapping
        resizeInfo.srcRow = srcImage;
        for (int y = 0; y < srcHeight; ++y) {
            equalYFunc(&resizeInfo);

            m_BlitInfo.dstLine = dstRow;
            blitFunc(&m_BlitInfo);

            dstRow += dst_desc.BytesPerLine;
            resizeInfo.srcRow += resizeInfo.srcPitch;
        }
    }
}

void VxBlitEngine::ResizeImage(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    if (!src_desc.Image || !dst_desc.Image) return;
    if (src_desc.Width <= 0 || src_desc.Height <= 0) return;
    if (dst_desc.Width <= 0 || dst_desc.Height <= 0) return;

    // Native high-quality bilinear resize implementation
    int srcChannels = src_desc.BitsPerPixel / 8;
    int dstChannels = dst_desc.BitsPerPixel / 8;

    // Handle 32-bit images directly with optimized bilinear interpolation
    if (srcChannels == 4 && dstChannels == 4) {
        ResizeBilinear32(src_desc, dst_desc);
        return;
    }

    // Handle 24-bit images directly
    if (srcChannels == 3 && dstChannels == 3) {
        ResizeBilinear24(src_desc, dst_desc);
        return;
    }

    // For mixed bit depths, convert to 32-bit, resize, then convert back
    if (srcChannels >= 2) {
        // Allocate temporary 32-bit buffers
        XArray<XBYTE> srcTemp, dstTemp;
        srcTemp.Reserve(src_desc.Width * src_desc.Height * 4);
        dstTemp.Reserve(dst_desc.Width * dst_desc.Height * 4);

        // Convert source to 32-bit ARGB
        VxImageDescEx src32;
        src32.Width = src_desc.Width;
        src32.Height = src_desc.Height;
        src32.BitsPerPixel = 32;
        src32.BytesPerLine = src_desc.Width * 4;
        src32.RedMask = 0x00FF0000;
        src32.GreenMask = 0x0000FF00;
        src32.BlueMask = 0x000000FF;
        src32.AlphaMask = 0xFF000000;
        src32.Image = srcTemp.Begin();

        // Blit to 32-bit format
        DoBlit(src_desc, src32);

        // Create destination 32-bit descriptor
        VxImageDescEx dst32;
        dst32.Width = dst_desc.Width;
        dst32.Height = dst_desc.Height;
        dst32.BitsPerPixel = 32;
        dst32.BytesPerLine = dst_desc.Width * 4;
        dst32.RedMask = 0x00FF0000;
        dst32.GreenMask = 0x0000FF00;
        dst32.BlueMask = 0x000000FF;
        dst32.AlphaMask = 0xFF000000;
        dst32.Image = dstTemp.Begin();

        // Resize in 32-bit
        ResizeBilinear32(src32, dst32);

        // Convert back to destination format
        DoBlit(dst32, dst_desc);
        return;
    }

    // Fallback: simple nearest-neighbor for 8-bit and non-standard formats
    ResizeNearestNeighbor(src_desc, dst_desc);
}

//------------------------------------------------------------------------------
// Native High-Performance Bilinear Resize Implementation
//------------------------------------------------------------------------------

void VxBlitEngine::ResizeBilinear32(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    const int srcW = src_desc.Width;
    const int srcH = src_desc.Height;
    const int dstW = dst_desc.Width;
    const int dstH = dst_desc.Height;

    // Fixed-point scale factors (16.16 format)
    const int scaleX = (srcW << 16) / dstW;
    const int scaleY = (srcH << 16) / dstH;

#if defined(VX_SIMD_SSE2)
    // SSE2 optimized bilinear resize
    for (int y = 0; y < dstH; ++y) {
        // Calculate source Y coordinate and fraction
        const int srcYFixed = y * scaleY;
        const int srcY0 = srcYFixed >> 16;
        const int srcY1 = XMin(srcY0 + 1, srcH - 1);
        const int fracY = srcYFixed & 0xFFFF;
        const int invFracY = 0x10000 - fracY;

        const XDWORD *srcRow0 = (const XDWORD *)(src_desc.Image + srcY0 * src_desc.BytesPerLine);
        const XDWORD *srcRow1 = (const XDWORD *)(src_desc.Image + srcY1 * src_desc.BytesPerLine);
        XDWORD *dstRow = (XDWORD *)(dst_desc.Image + y * dst_desc.BytesPerLine);

        // SSE constants for this row
        const __m128i vFracY = _mm_set1_epi16((short)(fracY >> 8));
        const __m128i vInvFracY = _mm_set1_epi16((short)(invFracY >> 8));
        const __m128i zero = _mm_setzero_si128();

        int srcXFixed = 0;
        for (int x = 0; x < dstW; ++x) {
            const int srcX0 = srcXFixed >> 16;
            const int srcX1 = XMin(srcX0 + 1, srcW - 1);
            const int fracX = srcXFixed & 0xFFFF;
            const int invFracX = 0x10000 - fracX;

            // Load 4 corner pixels
            XDWORD p00 = srcRow0[srcX0];
            XDWORD p10 = srcRow0[srcX1];
            XDWORD p01 = srcRow1[srcX0];
            XDWORD p11 = srcRow1[srcX1];

            // Convert fractional weights to 8-bit for SSE (0-256 range)
            // Use rounding for better precision
            int wX0 = (invFracX + 128) >> 8;
            int wX1 = 256 - wX0;  // Ensure wX0 + wX1 = 256
            int wY0 = (invFracY + 128) >> 8;
            int wY1 = 256 - wY0;  // Ensure wY0 + wY1 = 256

            // Calculate bilinear weights for 4 corners
            // Use higher precision intermediate calculation
            int w00 = (wX0 * wY0 + 128) >> 8;
            int w10 = (wX1 * wY0 + 128) >> 8;
            int w01 = (wX0 * wY1 + 128) >> 8;
            int w11 = 256 - w00 - w10 - w01;  // Ensure total = 256

            // Unpack to 16-bit and multiply
            __m128i px00 = _mm_cvtsi32_si128(p00);
            __m128i px10 = _mm_cvtsi32_si128(p10);
            __m128i px01 = _mm_cvtsi32_si128(p01);
            __m128i px11 = _mm_cvtsi32_si128(p11);

            px00 = _mm_unpacklo_epi8(px00, zero);
            px10 = _mm_unpacklo_epi8(px10, zero);
            px01 = _mm_unpacklo_epi8(px01, zero);
            px11 = _mm_unpacklo_epi8(px11, zero);

            __m128i vw00 = _mm_set1_epi16((short)w00);
            __m128i vw10 = _mm_set1_epi16((short)w10);
            __m128i vw01 = _mm_set1_epi16((short)w01);
            __m128i vw11 = _mm_set1_epi16((short)w11);

            __m128i result = _mm_mullo_epi16(px00, vw00);
            result = _mm_add_epi16(result, _mm_mullo_epi16(px10, vw10));
            result = _mm_add_epi16(result, _mm_mullo_epi16(px01, vw01));
            result = _mm_add_epi16(result, _mm_mullo_epi16(px11, vw11));

            // Shift right by 8 and pack back to bytes
            result = _mm_srli_epi16(result, 8);
            result = _mm_packus_epi16(result, zero);

            dstRow[x] = _mm_cvtsi128_si32(result);
            srcXFixed += scaleX;
        }
    }
#else
    // Scalar bilinear resize (fallback)
    for (int y = 0; y < dstH; ++y) {
        const int srcYFixed = y * scaleY;
        const int srcY0 = srcYFixed >> 16;
        const int srcY1 = XMin(srcY0 + 1, srcH - 1);
        const int fracY = srcYFixed & 0xFFFF;
        const int invFracY = 0x10000 - fracY;

        const XDWORD *srcRow0 = (const XDWORD *)(src_desc.Image + srcY0 * src_desc.BytesPerLine);
        const XDWORD *srcRow1 = (const XDWORD *)(src_desc.Image + srcY1 * src_desc.BytesPerLine);
        XDWORD *dstRow = (XDWORD *)(dst_desc.Image + y * dst_desc.BytesPerLine);

        int srcXFixed = 0;
        for (int x = 0; x < dstW; ++x) {
            const int srcX0 = srcXFixed >> 16;
            const int srcX1 = XMin(srcX0 + 1, srcW - 1);
            const int fracX = srcXFixed & 0xFFFF;
            const int invFracX = 0x10000 - fracX;

            XDWORD p00 = srcRow0[srcX0];
            XDWORD p10 = srcRow0[srcX1];
            XDWORD p01 = srcRow1[srcX0];
            XDWORD p11 = srcRow1[srcX1];

            // Bilinear interpolation for each channel
            XDWORD a = ((((p00 >> 24) & 0xFF) * invFracX + ((p10 >> 24) & 0xFF) * fracX) >> 16) * invFracY +
                       ((((p01 >> 24) & 0xFF) * invFracX + ((p11 >> 24) & 0xFF) * fracX) >> 16) * fracY;
            XDWORD r = ((((p00 >> 16) & 0xFF) * invFracX + ((p10 >> 16) & 0xFF) * fracX) >> 16) * invFracY +
                       ((((p01 >> 16) & 0xFF) * invFracX + ((p11 >> 16) & 0xFF) * fracX) >> 16) * fracY;
            XDWORD g = ((((p00 >> 8) & 0xFF) * invFracX + ((p10 >> 8) & 0xFF) * fracX) >> 16) * invFracY +
                       ((((p01 >> 8) & 0xFF) * invFracX + ((p11 >> 8) & 0xFF) * fracX) >> 16) * fracY;
            XDWORD b = (((p00 & 0xFF) * invFracX + (p10 & 0xFF) * fracX) >> 16) * invFracY +
                       (((p01 & 0xFF) * invFracX + (p11 & 0xFF) * fracX) >> 16) * fracY;

            dstRow[x] = ((a >> 16) << 24) | ((r >> 16) << 16) | ((g >> 16) << 8) | (b >> 16);
            srcXFixed += scaleX;
        }
    }
#endif
}

void VxBlitEngine::ResizeBilinear24(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    const int srcW = src_desc.Width;
    const int srcH = src_desc.Height;
    const int dstW = dst_desc.Width;
    const int dstH = dst_desc.Height;

    const int scaleX = (srcW << 16) / dstW;
    const int scaleY = (srcH << 16) / dstH;

    for (int y = 0; y < dstH; ++y) {
        const int srcYFixed = y * scaleY;
        const int srcY0 = srcYFixed >> 16;
        const int srcY1 = XMin(srcY0 + 1, srcH - 1);
        const int fracY = srcYFixed & 0xFFFF;
        const int invFracY = 0x10000 - fracY;

        const XBYTE *srcRow0 = src_desc.Image + srcY0 * src_desc.BytesPerLine;
        const XBYTE *srcRow1 = src_desc.Image + srcY1 * src_desc.BytesPerLine;
        XBYTE *dstRow = dst_desc.Image + y * dst_desc.BytesPerLine;

        int srcXFixed = 0;
        for (int x = 0; x < dstW; ++x) {
            const int srcX0 = srcXFixed >> 16;
            const int srcX1 = XMin(srcX0 + 1, srcW - 1);
            const int fracX = srcXFixed & 0xFFFF;
            const int invFracX = 0x10000 - fracX;

            const XBYTE *p00 = srcRow0 + srcX0 * 3;
            const XBYTE *p10 = srcRow0 + srcX1 * 3;
            const XBYTE *p01 = srcRow1 + srcX0 * 3;
            const XBYTE *p11 = srcRow1 + srcX1 * 3;

            // Bilinear interpolation for each channel (BGR order)
            for (int c = 0; c < 3; ++c) {
                XDWORD v = (((p00[c] * invFracX + p10[c] * fracX) >> 16) * invFracY +
                            ((p01[c] * invFracX + p11[c] * fracX) >> 16) * fracY) >> 16;
                dstRow[x * 3 + c] = (XBYTE)v;
            }

            srcXFixed += scaleX;
        }
    }
}

void VxBlitEngine::ResizeNearestNeighbor(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    const int srcW = src_desc.Width;
    const int srcH = src_desc.Height;
    const int dstW = dst_desc.Width;
    const int dstH = dst_desc.Height;
    const int srcBpp = src_desc.BitsPerPixel / 8;
    const int dstBpp = dst_desc.BitsPerPixel / 8;

    const int scaleX = (srcW << 16) / dstW;
    const int scaleY = (srcH << 16) / dstH;

    for (int y = 0; y < dstH; ++y) {
        const int srcY = (y * scaleY) >> 16;
        const XBYTE *srcRow = src_desc.Image + XMin(srcY, srcH - 1) * src_desc.BytesPerLine;
        XBYTE *dstRow = dst_desc.Image + y * dst_desc.BytesPerLine;

        int srcXFixed = 0;
        for (int x = 0; x < dstW; ++x) {
            const int srcX = srcXFixed >> 16;
            const XBYTE *srcPx = srcRow + XMin(srcX, srcW - 1) * srcBpp;
            XBYTE *dstPx = dstRow + x * dstBpp;

            // Copy pixel bytes
            for (int b = 0; b < XMin(srcBpp, dstBpp); ++b) {
                dstPx[b] = srcPx[b];
            }
            // Fill extra bytes with 0xFF (alpha) if destination is larger
            for (int b = srcBpp; b < dstBpp; ++b) {
                dstPx[b] = 0xFF;
            }

            srcXFixed += scaleX;
        }
    }
}

//------------------------------------------------------------------------------
// Median-Cut Algorithm
//------------------------------------------------------------------------------

// Color box for median-cut algorithm
struct ColorBox {
    int rMin, rMax;
    int gMin, gMax;
    int bMin, bMax;
    int count;
    int pixelStartIdx;
    int pixelCount;
};

// Pixel with color components
struct ColorPixel {
    XBYTE r, g, b;
    int count;
};

// Comparators for sorting
struct CompareR { bool operator()(const ColorPixel &a, const ColorPixel &b) { return a.r < b.r; } };
struct CompareG { bool operator()(const ColorPixel &a, const ColorPixel &b) { return a.g < b.g; } };
struct CompareB { bool operator()(const ColorPixel &a, const ColorPixel &b) { return a.b < b.b; } };

// Find color distance squared
inline int ColorDistSq(int r1, int g1, int b1, int r2, int g2, int b2) {
    int dr = r1 - r2;
    int dg = g1 - g2;
    int db = b1 - b2;
    // Weighted by human perception: green > red > blue
    return dr * dr * 2 + dg * dg * 4 + db * db;
}

//------------------------------------------------------------------------------
// NeuQuant-based Quantization (Default)
//------------------------------------------------------------------------------

XBOOL VxBlitEngine::QuantizeImage(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    // Quantization requires 256 color palette
    if (dst_desc.ColorMapEntries != 256) return FALSE;
    if (!dst_desc.ColorMap) return FALSE;

    int srcBpp = src_desc.BitsPerPixel / 8;
    if (srcBpp != 3 && srcBpp != 4) return FALSE;

    if (!src_desc.Image || !dst_desc.Image) return FALSE;
    if (src_desc.Width <= 0 || src_desc.Height <= 0) return FALSE;
    if (dst_desc.Width != src_desc.Width || dst_desc.Height != src_desc.Height) return FALSE;

    const int totalPixels = src_desc.Width * src_desc.Height;

    // Step 1: Convert image to RGB buffer for NeuQuant
    // NeuQuant expects RGB triplets (R, G, B order)
    const int rgbLen = totalPixels * 3;
    XArray<XBYTE> rgbBuffer;
    rgbBuffer.Resize(rgbLen);

    const XBYTE *srcRow = src_desc.Image;
    XBYTE *rgbPtr = rgbBuffer.Begin();

    for (int y = 0; y < src_desc.Height; ++y) {
        const XBYTE *src = srcRow;
        for (int x = 0; x < src_desc.Width; ++x) {
            // Source is BGR(A), NeuQuant expects RGB
            rgbPtr[0] = src[2]; // R
            rgbPtr[1] = src[1]; // G
            rgbPtr[2] = src[0]; // B
            rgbPtr += 3;
            src += srcBpp;
        }
        srcRow += src_desc.BytesPerLine;
    }

    // Step 2: Run NeuQuant
    NeuQuant nq;
    int sampleFac = GetQuantizationSamplingFactor();
    if (sampleFac < 1) sampleFac = 1;
    if (sampleFac > 30) sampleFac = 30;

    nq.initnet(rgbBuffer.Begin(), rgbLen, sampleFac);
    nq.learn();
    nq.unbiasnet();
    nq.inxbuild();

    // Step 3: Write palette
    XBYTE *palette = dst_desc.ColorMap;
    int paletteBytesPer = dst_desc.BytesPerColorEntry;
    if (paletteBytesPer < 3) paletteBytesPer = 3;

    for (int i = 0; i < 256; ++i) {
        XBYTE r, g, b;
        nq.getpalette(i, r, g, b);
        XBYTE *entry = palette + i * paletteBytesPer;
        entry[0] = b;
        entry[1] = g;
        entry[2] = r;
        if (paletteBytesPer >= 4) {
            entry[3] = 0xFF;
        }
    }

    // Step 4: Map pixels to palette indices
    srcRow = src_desc.Image;
    XBYTE *dstRow = dst_desc.Image;

    for (int y = 0; y < src_desc.Height; ++y) {
        const XBYTE *src = srcRow;
        XBYTE *dst = dstRow;

        for (int x = 0; x < src_desc.Width; ++x) {
            // Source is BGR(A)
            int b = src[0];
            int g = src[1];
            int r = src[2];

            // Find best matching palette index
            int idx = nq.inxsearch(r, g, b);
            *dst = (XBYTE)idx;

            src += srcBpp;
            dst++;
        }

        srcRow += src_desc.BytesPerLine;
        dstRow += dst_desc.BytesPerLine;
    }

    return TRUE;
}

//------------------------------------------------------------------------------
// Median-Cut Quantization
//------------------------------------------------------------------------------

XBOOL VxBlitEngine::QuantizeImageMedianCut(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    // Quantization requires 256 color palette
    if (dst_desc.ColorMapEntries != 256) return FALSE;
    if (!dst_desc.ColorMap) return FALSE;

    int srcBpp = src_desc.BitsPerPixel / 8;
    if (srcBpp != 3 && srcBpp != 4) return FALSE;

    if (!src_desc.Image || !dst_desc.Image) return FALSE;
    if (src_desc.Width <= 0 || src_desc.Height <= 0) return FALSE;
    if (dst_desc.Width != src_desc.Width || dst_desc.Height != src_desc.Height) return FALSE;

    const int totalPixels = src_desc.Width * src_desc.Height;
    const int maxColors = 256;

    // Step 1: Build color histogram
    // Use a hash table for color counting
    struct ColorEntry {
        XDWORD color;
        int count;
        int next;
    };

    const int hashSize = 65536;
    XArray<int> hashTable;
    hashTable.Resize(hashSize);
    for (int i = 0; i < hashSize; ++i) {
        hashTable[i] = -1;
    }

    XArray<ColorEntry> colorEntries;
    colorEntries.Resize(totalPixels / 4 + 256);
    int numUniqueColors = 0;

    // Scan all pixels and build histogram
    const XBYTE *srcRow = src_desc.Image;
    for (int y = 0; y < src_desc.Height; ++y) {
        const XBYTE *src = srcRow;
        for (int x = 0; x < src_desc.Width; ++x) {
            XBYTE b = src[0];
            XBYTE g = src[1];
            XBYTE r = src[2];

            // Create color key (5-6-5 quantized for hash)
            XDWORD colorKey = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
            XDWORD fullColor = (r << 16) | (g << 8) | b;

            // Search hash chain
            int idx = hashTable[colorKey];
            bool found = false;
            while (idx >= 0) {
                if (colorEntries[idx].color == fullColor) {
                    colorEntries[idx].count++;
                    found = true;
                    break;
                }
                idx = colorEntries[idx].next;
            }

            if (!found) {
                // Add new color
                ColorEntry entry;
                entry.color = fullColor;
                entry.count = 1;
                entry.next = hashTable[colorKey];
                
                int newIdx = numUniqueColors++;
                if (newIdx >= (int)colorEntries.Size()) {
                    colorEntries.Resize(colorEntries.Size() + 1024);
                }
                colorEntries[newIdx] = entry;
                hashTable[colorKey] = newIdx;
            }

            src += srcBpp;
        }
        srcRow += src_desc.BytesPerLine;
    }

    // Step 2: Build pixel array for median-cut
    XArray<ColorPixel> pixels;
    pixels.Resize(numUniqueColors);

    for (int i = 0; i < numUniqueColors; ++i) {
        ColorPixel cp;
        cp.r = (colorEntries[i].color >> 16) & 0xFF;
        cp.g = (colorEntries[i].color >> 8) & 0xFF;
        cp.b = colorEntries[i].color & 0xFF;
        cp.count = colorEntries[i].count;
        pixels[i] = cp;
    }

    // Step 3: Median-cut algorithm
    XArray<ColorBox> boxes;
    boxes.Resize(maxColors);

    // Initialize first box containing all colors
    ColorBox initialBox;
    initialBox.rMin = initialBox.gMin = initialBox.bMin = 255;
    initialBox.rMax = initialBox.gMax = initialBox.bMax = 0;
    initialBox.count = 0;
    initialBox.pixelStartIdx = 0;
    initialBox.pixelCount = numUniqueColors;

    for (int i = 0; i < numUniqueColors; ++i) {
        initialBox.rMin = XMin(initialBox.rMin, (int)pixels[i].r);
        initialBox.rMax = XMax(initialBox.rMax, (int)pixels[i].r);
        initialBox.gMin = XMin(initialBox.gMin, (int)pixels[i].g);
        initialBox.gMax = XMax(initialBox.gMax, (int)pixels[i].g);
        initialBox.bMin = XMin(initialBox.bMin, (int)pixels[i].b);
        initialBox.bMax = XMax(initialBox.bMax, (int)pixels[i].b);
        initialBox.count += pixels[i].count;
    }

    boxes[0] = initialBox;
    int numBoxes = 1;

    // Split boxes until we have maxColors
    while (numBoxes < maxColors) {
        // Find box with largest range to split (use max of ranges, not product)
        // Using max range instead of volume handles edge cases where colors 
        // are perfectly aligned on one or more axes
        int splitIdx = -1;
        int maxRange = 0;

        for (int i = 0; i < numBoxes; ++i) {
            int rRange = boxes[i].rMax - boxes[i].rMin;
            int gRange = boxes[i].gMax - boxes[i].gMin;
            int bRange = boxes[i].bMax - boxes[i].bMin;
            int largestRange = XMax(rRange, XMax(gRange, bRange));
            
            if (boxes[i].pixelCount <= 1) continue;
            if (largestRange == 0) continue; // Can't split a box with no color variation

            if (largestRange > maxRange) {
                maxRange = largestRange;
                splitIdx = i;
            }
        }

        if (splitIdx < 0) break;

        ColorBox &box = boxes[splitIdx];

        // Find longest axis
        int rRange = box.rMax - box.rMin;
        int gRange = box.gMax - box.gMin;
        int bRange = box.bMax - box.bMin;

        // Sort pixels by longest axis
        ColorPixel *boxPixels = pixels.Begin() + box.pixelStartIdx;
        if (rRange >= gRange && rRange >= bRange) {
            std::sort(boxPixels, boxPixels + box.pixelCount, CompareR());
        } else if (gRange >= bRange) {
            std::sort(boxPixels, boxPixels + box.pixelCount, CompareG());
        } else {
            std::sort(boxPixels, boxPixels + box.pixelCount, CompareB());
        }

        // Split at median
        int splitPoint = box.pixelCount / 2;
        if (splitPoint == 0) splitPoint = 1;

        // Create new box for second half
        ColorBox newBox;
        newBox.pixelStartIdx = box.pixelStartIdx + splitPoint;
        newBox.pixelCount = box.pixelCount - splitPoint;

        // Update original box
        box.pixelCount = splitPoint;

        // Recalculate bounds for both boxes
        auto recalcBounds = [&pixels](ColorBox &b) {
            b.rMin = b.gMin = b.bMin = 255;
            b.rMax = b.gMax = b.bMax = 0;
            b.count = 0;
            for (int i = 0; i < b.pixelCount; ++i) {
                ColorPixel &cp = pixels[b.pixelStartIdx + i];
                b.rMin = XMin(b.rMin, (int)cp.r);
                b.rMax = XMax(b.rMax, (int)cp.r);
                b.gMin = XMin(b.gMin, (int)cp.g);
                b.gMax = XMax(b.gMax, (int)cp.g);
                b.bMin = XMin(b.bMin, (int)cp.b);
                b.bMax = XMax(b.bMax, (int)cp.b);
                b.count += cp.count;
            }
        };

        recalcBounds(box);
        recalcBounds(newBox);

        boxes[numBoxes++] = newBox;
    }

    // Step 4: Calculate palette colors (weighted average of each box)
    XBYTE *palette = dst_desc.ColorMap;
    int paletteBytesPer = dst_desc.BytesPerColorEntry;
    if (paletteBytesPer < 3) paletteBytesPer = 3;

    for (int i = 0; i < numBoxes; ++i) {
        ColorBox &box = boxes[i];
        long long rSum = 0, gSum = 0, bSum = 0;
        int totalWeight = 0;

        for (int j = 0; j < box.pixelCount; ++j) {
            ColorPixel &cp = pixels[box.pixelStartIdx + j];
            rSum += cp.r * cp.count;
            gSum += cp.g * cp.count;
            bSum += cp.b * cp.count;
            totalWeight += cp.count;
        }

        XBYTE *entry = palette + i * paletteBytesPer;
        if (totalWeight > 0) {
            entry[0] = (XBYTE)(bSum / totalWeight);
            entry[1] = (XBYTE)(gSum / totalWeight);
            entry[2] = (XBYTE)(rSum / totalWeight);
        } else {
            entry[0] = entry[1] = entry[2] = 0;
        }
        if (paletteBytesPer >= 4) {
            entry[3] = 0xFF;
        }
    }

    // Fill remaining palette entries with black
    for (int i = numBoxes; i < maxColors; ++i) {
        XBYTE *entry = palette + i * paletteBytesPer;
        entry[0] = entry[1] = entry[2] = 0;
        if (paletteBytesPer >= 4) {
            entry[3] = 0xFF;
        }
    }

    // Step 5: Map pixels to palette
    srcRow = src_desc.Image;
    XBYTE *dstRow = dst_desc.Image;

    for (int y = 0; y < src_desc.Height; ++y) {
        const XBYTE *src = srcRow;
        XBYTE *dst = dstRow;

        for (int x = 0; x < src_desc.Width; ++x) {
            XBYTE b = src[0];
            XBYTE g = src[1];
            XBYTE r = src[2];

            // Find closest palette entry
            int bestIdx = 0;
            int bestDist = 0x7FFFFFFF;

            for (int i = 0; i < numBoxes; ++i) {
                XBYTE *entry = palette + i * paletteBytesPer;
                int dist = ColorDistSq(r, g, b, entry[2], entry[1], entry[0]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                    if (dist == 0) break;
                }
            }

            *dst = (XBYTE)bestIdx;

            src += srcBpp;
            dst++;
        }

        srcRow += src_desc.BytesPerLine;
        dstRow += dst_desc.BytesPerLine;
    }

    return TRUE;
}
