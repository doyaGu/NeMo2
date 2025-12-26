#include <string.h>
#include <stdlib.h>
#include <intrin.h>
#include <algorithm>

#include "VxMath.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#define STB_DXT_IMPLEMENTATION
#include "stb_dxt.h"

//------------------------------------------------------------------------------
// Global variables and constants
//------------------------------------------------------------------------------

static int g_QuantizationSamplingFactor = 15;

#define DXT1_BLOCK_SIZE 8
#define DXT3_BLOCK_SIZE 16
#define DXT5_BLOCK_SIZE 16

// Bit counting lookup table
alignas(64) static const XBYTE BIT_COUNT_TABLE[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

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
    count += BIT_COUNT_TABLE[dwMask & 0xff];
    dwMask >>= 8;
    count += BIT_COUNT_TABLE[dwMask & 0xff];
    dwMask >>= 8;
    count += BIT_COUNT_TABLE[dwMask & 0xff];
    dwMask >>= 8;
    count += BIT_COUNT_TABLE[dwMask & 0xff];
    return count;
#endif
}

XULONG GetBitShift(XULONG dwMask) {
    if (!dwMask) return 0;

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    unsigned long index;
    _BitScanForward(&index, dwMask);
    return index;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(dwMask);
#else
    // De Bruijn bit scanning
    static const int MultiplyDeBruijnBitPosition[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };
    return MultiplyDeBruijnBitPosition[((XULONG)((dwMask & -dwMask) * 0x077CB531U)) >> 27];
#endif
}

void VxGetBitCounts(const VxImageDescEx &desc, XULONG &Rbits, XULONG &Gbits, XULONG &Bbits, XULONG &Abits) {
    const XULONG masks[4] = {desc.RedMask, desc.GreenMask, desc.BlueMask, desc.AlphaMask};
    XULONG results[4];

    for (int i = 0; i < 4; ++i) {
        results[i] = GetBitCount(masks[i]);
    }

    Rbits = results[0];
    Gbits = results[1];
    Bbits = results[2];
    Abits = results[3];
}

void VxGetBitShifts(const VxImageDescEx &desc, XULONG &Rshift, XULONG &Gshift, XULONG &Bshift, XULONG &Ashift) {
    const XULONG masks[4] = {desc.RedMask, desc.GreenMask, desc.BlueMask, desc.AlphaMask};
    XULONG results[4];

    for (int i = 0; i < 4; ++i) {
        results[i] = GetBitShift(masks[i]);
    }

    Rshift = results[0];
    Gshift = results[1];
    Bshift = results[2];
    Ashift = results[3];
}

//------------------------------------------------------------------------------
// Pixel format utilities
//------------------------------------------------------------------------------

struct PixelFormatInfo {
    XWORD BitsPerPixel;
    XULONG RedMask;
    XULONG GreenMask;
    XULONG BlueMask;
    XULONG AlphaMask;
    const char *Description;
    stbir_pixel_layout StbLayout;
    stbir_datatype StbDataType;
};

static const PixelFormatInfo PixelFormatTable[] = {
    {0, 0, 0, 0, 0, "Unknown", STBIR_1CHANNEL, STBIR_TYPE_UINT8},
    {32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000, "32 bits ARGB 8888", STBIR_ARGB, STBIR_TYPE_UINT8},
    {32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000, "32 bits RGB  888", STBIR_4CHANNEL, STBIR_TYPE_UINT8},
    {24, 0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000, "24 bits RGB  888", STBIR_RGB, STBIR_TYPE_UINT8},
    {16, 0xF800, 0x07E0, 0x001F, 0x0000, "16 bits RGB  565", STBIR_RGB, STBIR_TYPE_UINT16},
    {16, 0x7C00, 0x03E0, 0x001F, 0x0000, "16 bits RGB  555", STBIR_RGB, STBIR_TYPE_UINT16},
    {16, 0x7C00, 0x03E0, 0x001F, 0x8000, "16 bits ARGB 1555", STBIR_ARGB, STBIR_TYPE_UINT16},
    {16, 0x0F00, 0x00F0, 0x000F, 0xF000, "16 bits ARGB 4444", STBIR_ARGB, STBIR_TYPE_UINT16},
    {8, 0xE0, 0x1C, 0x03, 0x00, "8 bits RGB  332", STBIR_RGB, STBIR_TYPE_UINT8},
    {8, 0x30, 0x0C, 0x03, 0xC0, "8 bits ARGB 2222", STBIR_ARGB, STBIR_TYPE_UINT8},
    {32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000, "32 bits ABGR 8888", STBIR_ABGR, STBIR_TYPE_UINT8},
    {32, 0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF, "32 bits RGBA 8888", STBIR_RGBA, STBIR_TYPE_UINT8},
    {32, 0x0000FF00, 0x00FF0000, 0xFF000000, 0x000000FF, "32 bits BGRA 8888", STBIR_BGRA, STBIR_TYPE_UINT8},
    {32, 0x0000FF00, 0x00FF0000, 0xFF000000, 0x00000000, "32 bits BGR  888", STBIR_4CHANNEL, STBIR_TYPE_UINT8},
    {24, 0x0000FF00, 0x00FF0000, 0xFF000000, 0x00000000, "24 bits BGR  888", STBIR_BGR, STBIR_TYPE_UINT8},
    {16, 0x001F, 0x07E0, 0xF800, 0x0000, "16 bits BGR  565", STBIR_BGR, STBIR_TYPE_UINT16},
    {16, 0x001F, 0x03E0, 0x7C00, 0x0000, "16 bits BGR  555", STBIR_BGR, STBIR_TYPE_UINT16},
    {16, 0x001F, 0x03E0, 0x7C00, 0x8000, "16 bits ABGR 1555", STBIR_ABGR, STBIR_TYPE_UINT16},
    {16, 0x000F, 0x00F0, 0x0F00, 0xF000, "16 bits ABGR 4444", STBIR_ABGR, STBIR_TYPE_UINT16},
    // DXT formats
    {4, 0, 0, 0, 0, "Compressed DXT1", STBIR_1CHANNEL, STBIR_TYPE_UINT8},
    {8, 0, 0, 0, 0, "Compressed DXT2", STBIR_1CHANNEL, STBIR_TYPE_UINT8},
    {8, 0, 0, 0, 0, "Compressed DXT3", STBIR_1CHANNEL, STBIR_TYPE_UINT8},
    {8, 0, 0, 0, 0, "Compressed DXT4", STBIR_1CHANNEL, STBIR_TYPE_UINT8},
    {8, 0, 0, 0, 0, "Compressed DXT5", STBIR_1CHANNEL, STBIR_TYPE_UINT8},
    // Bump map formats
    {16, 0x00FF, 0xFF00, 0x0000, 0x0000, "V8U8 BumpMap", STBIR_2CHANNEL, STBIR_TYPE_UINT8},
    {32, 0xFFFF, 0xFFFF0000, 0x0000, 0x0000, "V16U16 BumpMap", STBIR_2CHANNEL, STBIR_TYPE_UINT16},
    {16, 0x001F, 0x03E0, 0x7C00, 0x0000, "L6V5U5 BumpMap", STBIR_RGB, STBIR_TYPE_UINT16},
    {32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000, "L8V8U8 BumpMap", STBIR_4CHANNEL, STBIR_TYPE_UINT8}
};

// Lookup table for common 32-bit formats for faster detection
struct FastFormat32 {
    XULONG RedMask, GreenMask, BlueMask, AlphaMask;
    VX_PIXELFORMAT format;
};

// Cache-aligned lookup table for common 32-bit formats
alignas(64) static const FastFormat32 Fast32Formats[] = {
    {0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000, _32_ARGB8888},
    {0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000, _32_RGB888},
    {0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF, _32_RGBA8888},
    {0x0000FF00, 0x00FF0000, 0xFF000000, 0x000000FF, _32_BGRA8888},
    {0x0000FF00, 0x00FF0000, 0xFF000000, 0x00000000, _32_BGR888},
    {0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000, _32_ABGR8888},
};
constexpr int FAST32_COUNT = sizeof(Fast32Formats) / sizeof(Fast32Formats[0]);

VX_PIXELFORMAT VxImageDesc2PixelFormat(const VxImageDescEx &desc) {
    // Fast path for compressed/bump formats
    if (desc.Flags >= _DXT1 && desc.Flags <= _DXT5) return static_cast<VX_PIXELFORMAT>(desc.Flags);
    if (desc.Flags >= _16_V8U8 && desc.Flags <= _32_X8L8V8U8) return static_cast<VX_PIXELFORMAT>(desc.Flags);

    // 32-bit format detection using lookup table
    if (desc.BitsPerPixel == 32) {
        for (int i = 0; i < FAST32_COUNT; ++i) {
            const FastFormat32 &fmt = Fast32Formats[i];
            if (desc.RedMask == fmt.RedMask && desc.GreenMask == fmt.GreenMask &&
                desc.BlueMask == fmt.BlueMask && desc.AlphaMask == fmt.AlphaMask) {
                return fmt.format;
            }
        }
    }

    const XULONG bpp = desc.BitsPerPixel;
    const XULONG rmask = desc.RedMask;
    const XULONG gmask = desc.GreenMask;
    const XULONG bmask = desc.BlueMask;
    const XULONG amask = desc.AlphaMask;

    // Linear search for remaining formats
    for (int i = 1; i < _16_V8U8; ++i) {
        const PixelFormatInfo &info = PixelFormatTable[i];
        if (info.BitsPerPixel == bpp &&
            info.RedMask == rmask &&
            info.GreenMask == gmask &&
            info.BlueMask == bmask &&
            info.AlphaMask == amask) {
            return static_cast<VX_PIXELFORMAT>(i);
        }
    }
    return UNKNOWN_PF;
}

void VxPixelFormat2ImageDesc(VX_PIXELFORMAT Pf, VxImageDescEx &desc) {
    if (Pf < 0 || Pf >= MAX_PIXEL_FORMATS) return;

    if (Pf >= _DXT1 && Pf <= _DXT5) {
        // Compressed formats
        desc.Flags = Pf;
        desc.BitsPerPixel = (Pf == _DXT1) ? 4 : 8;
        const int blocksPerRow = (desc.Width + 3) / 4;
        const int bytesPerBlock = (Pf == _DXT1) ? DXT1_BLOCK_SIZE : DXT5_BLOCK_SIZE;
        desc.BytesPerLine = blocksPerRow * bytesPerBlock;
        desc.TotalImageSize = blocksPerRow * ((desc.Height + 3) / 4) * bytesPerBlock;
        desc.RedMask = desc.GreenMask = desc.BlueMask = desc.AlphaMask = 0;
    } else {
        // Standard formats
        const PixelFormatInfo &info = PixelFormatTable[Pf];
        desc.BitsPerPixel = info.BitsPerPixel;
        desc.RedMask = info.RedMask;
        desc.GreenMask = info.GreenMask;
        desc.BlueMask = info.BlueMask;
        desc.AlphaMask = info.AlphaMask;
        desc.Flags = (Pf >= _16_V8U8) ? Pf : 0;
        desc.BytesPerLine = ((desc.Width * ((desc.BitsPerPixel + 7) / 8)) + 3) & ~3;
        desc.TotalImageSize = desc.BytesPerLine * desc.Height;
    }
}

const char *VxPixelFormat2String(VX_PIXELFORMAT Pf) {
    return (Pf >= 0 && Pf < MAX_PIXEL_FORMATS) ? PixelFormatTable[Pf].Description : "";
}

void VxBppToMask(VxImageDescEx &desc) {
    switch (desc.BitsPerPixel) {
    case 8:
        desc.RedMask = 0xE0;   // 111 00000
        desc.GreenMask = 0x1C; // 000 11100
        desc.BlueMask = 0x03;  // 000 00011
        desc.AlphaMask = 0x00;
        break;
    case 15:
    case 16:
        desc.RedMask = 0x7C00;   // 0111 1100 0000 0000
        desc.GreenMask = 0x03E0; // 0000 0011 1110 0000
        desc.BlueMask = 0x001F;  // 0000 0000 0001 1111
        desc.AlphaMask = 0x0000;
        break;
    case 24:
    case 32:
        desc.RedMask = 0x00FF0000;
        desc.GreenMask = 0x0000FF00;
        desc.BlueMask = 0x000000FF;
        desc.AlphaMask = (desc.BitsPerPixel == 32) ? 0xFF000000 : 0x00000000;
        break;
    default:
        break;
    }
}

stbir_pixel_layout GetStbPixelLayout(const VxImageDescEx &desc) {
    VX_PIXELFORMAT format = VxImageDesc2PixelFormat(desc);
    if (format != UNKNOWN_PF) return PixelFormatTable[format].StbLayout;

    // Fallback detection for unknown formats
    const int channelCount = desc.BitsPerPixel / 8;
    switch (channelCount) {
    case 1: return STBIR_1CHANNEL;
    case 2: return STBIR_2CHANNEL;
    case 3:
        return (desc.RedMask > desc.BlueMask) ? STBIR_RGB : STBIR_BGR;
    case 4:
        if (desc.AlphaMask) {
            if (desc.RedMask == 0x00FF0000) return STBIR_ARGB;
            if (desc.RedMask == 0xFF000000) return STBIR_RGBA;
            if (desc.RedMask == 0x0000FF00) return STBIR_BGRA;
            if (desc.RedMask == 0x000000FF) return STBIR_ABGR;
        }
        return STBIR_4CHANNEL;
    default: return STBIR_1CHANNEL;
    }
}

stbir_datatype GetStbDataType(const VxImageDescEx &desc) {
    VX_PIXELFORMAT format = VxImageDesc2PixelFormat(desc);
    if (format != UNKNOWN_PF) return PixelFormatTable[format].StbDataType;

    return (desc.BitsPerPixel <= 8) ? STBIR_TYPE_UINT8 :
           (desc.BitsPerPixel <= 16) ? STBIR_TYPE_UINT16 : STBIR_TYPE_UINT8;
}

//------------------------------------------------------------------------------
// Pixel conversion utilities
//------------------------------------------------------------------------------

// Force inline for critical pixel operations
template<int BytesPerPixel>
inline XULONG ReadPixelTyped(const XBYTE *ptr) {
    static_assert(BytesPerPixel >= 1 && BytesPerPixel <= 4, "Invalid bytes per pixel");
    if constexpr (BytesPerPixel == 1) return *ptr;
    if constexpr (BytesPerPixel == 2) return *(const XWORD *)ptr;
    if constexpr (BytesPerPixel == 3) return ptr[0] | (ptr[1] << 8) | (ptr[2] << 16);
    if constexpr (BytesPerPixel == 4) return *(const XULONG *)ptr;
}

template<int BytesPerPixel>
inline void WritePixelTyped(XBYTE *ptr, XULONG pixel) {
    static_assert(BytesPerPixel >= 1 && BytesPerPixel <= 4, "Invalid bytes per pixel");
    if constexpr (BytesPerPixel == 1) *ptr = (XBYTE)pixel;
    if constexpr (BytesPerPixel == 2) *(XWORD *)ptr = (XWORD)pixel;
    if constexpr (BytesPerPixel == 3) {
        ptr[0] = (XBYTE)pixel;
        ptr[1] = (XBYTE)(pixel >> 8);
        ptr[2] = (XBYTE)(pixel >> 16);
    }
    if constexpr (BytesPerPixel == 4) *(XULONG *)ptr = pixel;
}

inline XULONG ReadPixel(const XBYTE *ptr, int bpp) {
    switch (bpp) {
    case 1: return ReadPixelTyped<1>(ptr);
    case 2: return ReadPixelTyped<2>(ptr);
    case 3: return ReadPixelTyped<3>(ptr);
    case 4: return ReadPixelTyped<4>(ptr);
    default: return 0;
    }
}

inline void WritePixel(XBYTE *ptr, int bpp, XULONG pixel) {
    switch (bpp) {
    case 1: WritePixelTyped<1>(ptr, pixel); break;
    case 2: WritePixelTyped<2>(ptr, pixel); break;
    case 3: WritePixelTyped<3>(ptr, pixel); break;
    case 4: WritePixelTyped<4>(ptr, pixel); break;
    }
}

inline XULONG ReadPixelAt(const XBYTE *basePtr, int stride, int x, int y, int bpp) {
    const XBYTE *ptr = basePtr + y * stride + x * bpp;
    return ReadPixel(ptr, bpp);
}

inline void WritePixelAt(XBYTE *basePtr, int stride, int x, int y, int bpp, XULONG pixel) {
    XBYTE *ptr = basePtr + y * stride + x * bpp;
    WritePixel(ptr, bpp, pixel);
}

struct ExtractContext {
    XULONG rShift, gShift, bShift, aShift;
    XULONG rBits, gBits, bBits, aBits;
    XULONG rMask, gMask, bMask, aMask;
    XULONG rScale, gScale, bScale, aScale;

    explicit ExtractContext(const VxImageDescEx &desc) {
        rMask = desc.RedMask; gMask = desc.GreenMask; bMask = desc.BlueMask; aMask = desc.AlphaMask;
        rShift = GetBitShift(rMask); gShift = GetBitShift(gMask); bShift = GetBitShift(bMask); aShift = GetBitShift(aMask);
        rBits = GetBitCount(rMask); gBits = GetBitCount(gMask); bBits = GetBitCount(bMask); aBits = GetBitCount(aMask);

        // Pre-compute scale factors
        rScale = (rBits > 0 && rBits < 8) ? (255 / ((1 << rBits) - 1)) : 1;
        gScale = (gBits > 0 && gBits < 8) ? (255 / ((1 << gBits) - 1)) : 1;
        bScale = (bBits > 0 && bBits < 8) ? (255 / ((1 << bBits) - 1)) : 1;
        aScale = (aBits > 0 && aBits < 8) ? (255 / ((1 << aBits) - 1)) : 1;
    }

    inline void ExtractRGBA(XULONG pixel, XBYTE &r, XBYTE &g, XBYTE &b, XBYTE &a) const {
        // Extract components
        const XULONG rVal = rMask ? ((pixel & rMask) >> rShift) : 0;
        const XULONG gVal = gMask ? ((pixel & gMask) >> gShift) : 0;
        const XULONG bVal = bMask ? ((pixel & bMask) >> bShift) : 0;
        const XULONG aVal = aMask ? ((pixel & aMask) >> aShift) : 255;

        // Scale to 8-bit range using pre-computed factors
        r = (rBits > 0 && rBits < 8) ? (XBYTE)(rVal * rScale) : (XBYTE)rVal;
        g = (gBits > 0 && gBits < 8) ? (XBYTE)(gVal * gScale) : (XBYTE)gVal;
        b = (bBits > 0 && bBits < 8) ? (XBYTE)(bVal * bScale) : (XBYTE)bVal;
        a = (aBits > 0 && aBits < 8) ? (XBYTE)(aVal * aScale) : (XBYTE)aVal;
    }
};


struct PixelCreateContext {
    XULONG rShift, gShift, bShift, aShift;
    XULONG rMask, gMask, bMask, aMask;
    XULONG rScale, gScale, bScale, aScale;
    bool hasAlpha;

    explicit PixelCreateContext(const VxImageDescEx &desc) {
        rMask = desc.RedMask; gMask = desc.GreenMask; bMask = desc.BlueMask; aMask = desc.AlphaMask;
        rShift = GetBitShift(rMask); gShift = GetBitShift(gMask); bShift = GetBitShift(bMask); aShift = GetBitShift(aMask);

        const XULONG rBits = GetBitCount(rMask);
        const XULONG gBits = GetBitCount(gMask);
        const XULONG bBits = GetBitCount(bMask);
        const XULONG aBits = GetBitCount(aMask);

        // Pre-compute scale factors for bit depth conversion
        rScale = (rBits > 0 && rBits < 8) ? ((1 << rBits) - 1) : 255;
        gScale = (gBits > 0 && gBits < 8) ? ((1 << gBits) - 1) : 255;
        bScale = (bBits > 0 && bBits < 8) ? ((1 << bBits) - 1) : 255;
        aScale = (aBits > 0 && aBits < 8) ? ((1 << aBits) - 1) : 255;
        hasAlpha = (aMask != 0);
    }

    XULONG CreatePixel(XBYTE r, XBYTE g, XBYTE b, XBYTE a = 255) const {
        // Fast scaling using pre-computed factors
        const XULONG rVal = (rScale != 255) ? (r * rScale) / 255 : r;
        const XULONG gVal = (gScale != 255) ? (g * gScale) / 255 : g;
        const XULONG bVal = (bScale != 255) ? (b * bScale) / 255 : b;
        const XULONG aVal = hasAlpha ? ((aScale != 255) ? (a * aScale) / 255 : a) : 0;

        return (rMask ? ((rVal << rShift) & rMask) : 0) |
               (gMask ? ((gVal << gShift) & gMask) : 0) |
               (bMask ? ((bVal << bShift) & bMask) : 0) |
               (aMask ? ((aVal << aShift) & aMask) : 0);
    }
};

struct PixelConvertContext {
    ExtractContext extract;
    PixelCreateContext create;
    bool identicalFormat;

    PixelConvertContext(const VxImageDescEx &srcDesc, const VxImageDescEx &dstDesc) : extract(srcDesc), create(dstDesc) {
        identicalFormat = (srcDesc.RedMask == dstDesc.RedMask &&
                          srcDesc.GreenMask == dstDesc.GreenMask &&
                          srcDesc.BlueMask == dstDesc.BlueMask &&
                          srcDesc.AlphaMask == dstDesc.AlphaMask);
    }

    XULONG ConvertPixel(XULONG srcPixel) const {
        if (identicalFormat) return srcPixel;

        XBYTE r, g, b, a;
        extract.ExtractRGBA(srcPixel, r, g, b, a);
        return create.CreatePixel(r, g, b, a);
    }
};

inline void ConvertPixelBatch(const XULONG *srcPixels, XULONG *dstPixels, int count, const PixelConvertContext &ctx) {
    if (ctx.identicalFormat) {
        memcpy(dstPixels, srcPixels, count * sizeof(XULONG));
        return;
    }

    // Process 4 pixels at a time for better ILP
    int i = 0;
    const int count4 = (count / 4) * 4;

    for (; i < count4; i += 4) {
        dstPixels[i] = ctx.ConvertPixel(srcPixels[i]);
        dstPixels[i + 1] = ctx.ConvertPixel(srcPixels[i + 1]);
        dstPixels[i + 2] = ctx.ConvertPixel(srcPixels[i + 2]);
        dstPixels[i + 3] = ctx.ConvertPixel(srcPixels[i + 3]);
    }

    // Handle remaining pixels
    for (; i < count; i++) {
        dstPixels[i] = ctx.ConvertPixel(srcPixels[i]);
    }
}

inline void ApplyAlphaBatch(XULONG* pixels, int count, XBYTE alphaValue, XULONG alphaMask, XULONG alphaShift) {
    const XULONG alphaComponent = (XULONG)alphaValue << alphaShift;
    const XULONG colorMask = ~alphaMask;

    // Process 4 pixels at a time
    int i = 0;
    const int count4 = (count / 4) * 4;

    for (; i < count4; i += 4) {
        pixels[i] = (pixels[i] & colorMask) | alphaComponent;
        pixels[i + 1] = (pixels[i + 1] & colorMask) | alphaComponent;
        pixels[i + 2] = (pixels[i + 2] & colorMask) | alphaComponent;
        pixels[i + 3] = (pixels[i + 3] & colorMask) | alphaComponent;
    }

    // Handle remaining pixels
    for (; i < count; i++) {
        pixels[i] = (pixels[i] & colorMask) | alphaComponent;
    }
}

inline void ApplyVariableAlphaBatch(XULONG* pixels, const XBYTE* alphaValues, int count, XULONG alphaMask, XULONG alphaShift) {
    const XULONG colorMask = ~alphaMask;

    // Process 4 pixels at a time
    int i = 0;
    const int count4 = (count / 4) * 4;

    for (; i < count4; i += 4) {
        pixels[i] = (pixels[i] & colorMask) | ((XULONG)alphaValues[i] << alphaShift);
        pixels[i + 1] = (pixels[i + 1] & colorMask) | ((XULONG)alphaValues[i + 1] << alphaShift);
        pixels[i + 2] = (pixels[i + 2] & colorMask) | ((XULONG)alphaValues[i + 2] << alphaShift);
        pixels[i + 3] = (pixels[i + 3] & colorMask) | ((XULONG)alphaValues[i + 3] << alphaShift);
    }

    // Handle remaining pixels
    for (; i < count; i++) {
        pixels[i] = (pixels[i] & colorMask) | ((XULONG)alphaValues[i] << alphaShift);
    }
}

//------------------------------------------------------------------------------
// DXT Compression Utilities
//------------------------------------------------------------------------------

// 565 conversion lookup tables
alignas(64) static const XBYTE R5_TO_R8[32] = {
    0, 8, 16, 25, 33, 41, 49, 58, 66, 74, 82, 90, 99, 107, 115, 123,
    132, 140, 148, 156, 165, 173, 181, 189, 197, 206, 214, 222, 230, 239, 247, 255
};
alignas(64) static const XBYTE G6_TO_G8[64] = {
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 49, 53, 57, 61,
    65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
    130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190,
    194, 198, 202, 206, 210, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255
};
alignas(64) static const XBYTE B5_TO_B8[32] = {
    0, 8, 16, 25, 33, 41, 49, 58, 66, 74, 82, 90, 99, 107, 115, 123,
    132, 140, 148, 156, 165, 173, 181, 189, 197, 206, 214, 222, 230, 239, 247, 255
};

inline void DecompressColor565(XWORD color, XBYTE *rgb) {
    rgb[0] = R5_TO_R8[(color >> 11) & 0x1F];
    rgb[1] = G6_TO_G8[(color >> 5) & 0x3F];
    rgb[2] = B5_TO_B8[color & 0x1F];
}

inline int CalculateDXTSize(int width, int height, VX_PIXELFORMAT format) {
    if (width <= 0 || height <= 0 || format < _DXT1 || format > _DXT5) return 0;

    const int blocksWide = (width + 3) / 4;
    const int blocksHigh = (height + 3) / 4;

    if (blocksWide > INT_MAX / blocksHigh) return 0;
    const int numBlocks = blocksWide * blocksHigh;
    const int bytesPerBlock = (format == _DXT1) ? DXT1_BLOCK_SIZE : DXT5_BLOCK_SIZE;

    return (numBlocks > INT_MAX / bytesPerBlock) ? 0 : numBlocks * bytesPerBlock;
}

inline XBOOL ValidateImageDesc(const VxImageDescEx &desc) {
    return desc.Image && desc.Width > 0 && desc.Height > 0 &&
        desc.BitsPerPixel > 0 && desc.BitsPerPixel <= 32 &&
        desc.BytesPerLine > 0 && (desc.Height <= INT_MAX / desc.BytesPerLine);
}

XBOOL VxConvertToDXT(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc, XBOOL highQuality) {
    if (!ValidateImageDesc(src_desc) || !ValidateImageDesc(dst_desc)) return FALSE;

    const VX_PIXELFORMAT srcFormat = VxImageDesc2PixelFormat(src_desc);
    const VX_PIXELFORMAT dstFormat = VxImageDesc2PixelFormat(dst_desc);

    if (srcFormat >= _DXT1 && srcFormat <= _DXT5) return FALSE;
    if (dstFormat < _DXT1 || dstFormat > _DXT5) return FALSE;
    if (src_desc.BitsPerPixel < 8 || (!src_desc.RedMask && !src_desc.GreenMask && !src_desc.BlueMask)) return FALSE;

    const int expectedSize = CalculateDXTSize(src_desc.Width, src_desc.Height, dstFormat);
    if (expectedSize <= 0 || dst_desc.TotalImageSize < expectedSize) return FALSE;

    const int blockSize = (dstFormat == _DXT1) ? DXT1_BLOCK_SIZE : DXT5_BLOCK_SIZE;
    const int blocksWide = (src_desc.Width + 3) / 4;
    const int blocksHigh = (src_desc.Height + 3) / 4;

    if (blocksWide <= 0 || blocksHigh <= 0 || blocksWide > INT_MAX / blocksHigh) return FALSE;

    const bool hasAlpha = (dstFormat != _DXT1) || (src_desc.AlphaMask != 0);
    const int mode = highQuality ? STB_DXT_HIGHQUAL : STB_DXT_NORMAL;
    const int srcBpp = src_desc.BitsPerPixel / 8;

    if (srcBpp <= 0 || srcBpp > 4) return FALSE;

    // Pre-compute extraction context
    ExtractContext extractCtx(src_desc);

    // Pre-compute base pointers
    const XBYTE *srcBase = src_desc.Image;
    const int srcStride = src_desc.BytesPerLine;

    // Process blocks in cache-friendly order
    alignas(16) unsigned char block[64]; // 16 pixels * 4 bytes = 64 bytes, aligned for SIMD

    for (int by = 0; by < blocksHigh; by++) {
        for (int bx = 0; bx < blocksWide; bx++) {
            // Extract 4x4 block
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    const int srcX = XMin(bx * 4 + x, src_desc.Width - 1);
                    const int srcY = XMin(by * 4 + y, src_desc.Height - 1);
                    const int blockIdx = (y * 4 + x) * 4;

                    XULONG pixel = ReadPixelAt(srcBase, srcStride, srcX, srcY, srcBpp);

                    unsigned char r, g, b, a;
                    extractCtx.ExtractRGBA(pixel, r, g, b, a);

                    block[blockIdx] = r;
                    block[blockIdx + 1] = g;
                    block[blockIdx + 2] = b;
                    block[blockIdx + 3] = a;
                }
            }

            const int blockIndex = by * blocksWide + bx;
            const int dstOffset = blockIndex * blockSize;

            if (dstOffset < 0 || dstOffset + blockSize > dst_desc.TotalImageSize) return FALSE;

            stb_compress_dxt_block(dst_desc.Image + dstOffset, block, hasAlpha ? 1 : 0, mode);
        }
    }

    return TRUE;
}

// DXT decompression alpha functions
static void DecompressDXT3Alpha(const XBYTE *blockPtr, XBYTE *alphaValues) {
    if (!blockPtr || !alphaValues) {
        if (alphaValues) {
            // Vectorized initialization
            for (int i = 0; i < 16; i += 4) {
                *((XULONG*)(alphaValues + i)) = 0xFFFFFFFF;
            }
        }
        return;
    }

    // Process 2 pixels at a time
    for (int i = 0; i < 8; i++) {
        const XBYTE alphaByte = blockPtr[i];
        const int idx = i * 2;
        alphaValues[idx] = ((alphaByte & 0x0F) * 255) / 15;
        alphaValues[idx + 1] = ((alphaByte >> 4) * 255) / 15;
    }
}

static void DecompressDXT5Alpha(const XBYTE *blockPtr, XBYTE *alphaValues) {
    if (!blockPtr || !alphaValues) {
        if (alphaValues) {
            for (int i = 0; i < 16; i += 4) {
                *((XULONG*)(alphaValues + i)) = 0xFFFFFFFF;
            }
        }
        return;
    }

    const XBYTE alpha0 = blockPtr[0];
    const XBYTE alpha1 = blockPtr[1];

    // Pre-compute alpha lookup table
    XBYTE alphaLUT[8];
    alphaLUT[0] = alpha0;
    alphaLUT[1] = alpha1;

    if (alpha0 > alpha1) {
        for (int i = 2; i < 8; i++) {
            alphaLUT[i] = (XBYTE)(((8 - i) * alpha0 + (i - 1) * alpha1) / 7);
        }
    } else {
        for (int i = 2; i < 6; i++) {
            alphaLUT[i] = (XBYTE)(((6 - i) * alpha0 + (i - 1) * alpha1) / 5);
        }
        alphaLUT[6] = 0;
        alphaLUT[7] = 255;
    }

    // Build alpha indices
    XULONG alphaIndices = 0;
    for (int i = 0; i < 6; i++) {
        alphaIndices |= ((XULONG)blockPtr[i + 2] << (8 * i));
    }

    // Apply alpha values using lookup table
    for (int i = 0; i < 16; i++) {
        const XDWORD alphaIndex = (XDWORD)((alphaIndices >> (3 * i)) & 0x07);
        alphaValues[i] = (alphaIndex < 8) ? alphaLUT[alphaIndex] : 255;
    }
}

XBOOL VxDecompressDXT(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    if (!ValidateImageDesc(src_desc) || !ValidateImageDesc(dst_desc)) return FALSE;

    const VX_PIXELFORMAT srcFormat = VxImageDesc2PixelFormat(src_desc);
    const VX_PIXELFORMAT dstFormat = VxImageDesc2PixelFormat(dst_desc);

    if (srcFormat < _DXT1 || srcFormat > _DXT5) return FALSE;
    if (dstFormat >= _DXT1 && dstFormat <= _DXT5) return FALSE;
    if (dst_desc.BitsPerPixel < 8) return FALSE;
    if (src_desc.Width != dst_desc.Width || src_desc.Height != dst_desc.Height) return FALSE;

    const int blockSize = (srcFormat == _DXT1) ? DXT1_BLOCK_SIZE : DXT5_BLOCK_SIZE;
    const int blocksWide = (src_desc.Width + 3) / 4;
    const int blocksHigh = (src_desc.Height + 3) / 4;
    const int expectedSize = blocksWide * blocksHigh * blockSize;

    if (src_desc.TotalImageSize < expectedSize) return FALSE;

    const int dstBpp = dst_desc.BitsPerPixel / 8;
    if (dstBpp <= 0 || dstBpp > 4) return FALSE;

    // Pre-compute pixel creation context
    const PixelCreateContext createCtx(dst_desc);

    // Process blocks in cache-friendly order
    for (int by = 0; by < blocksHigh; by++) {
        for (int bx = 0; bx < blocksWide; bx++) {
            const int blockIndex = by * blocksWide + bx;
            if (blockIndex * blockSize + blockSize > src_desc.TotalImageSize) return FALSE;

            const XBYTE *blockPtr = src_desc.Image + blockIndex * blockSize;
            const XBYTE *colorPtr = (srcFormat == _DXT1) ? blockPtr : (blockPtr + 8);

            const XWORD color0 = colorPtr[0] | (colorPtr[1] << 8);
            const XWORD color1 = colorPtr[2] | (colorPtr[3] << 8);
            const XDWORD colorIndices = colorPtr[4] | (colorPtr[5] << 8) | (colorPtr[6] << 16) | (colorPtr[7] << 24);

            // Pre-compute reference colors
            alignas(16) XBYTE refColors[4][4];

            DecompressColor565(color0, refColors[0]);
            refColors[0][3] = 255;
            DecompressColor565(color1, refColors[1]);
            refColors[1][3] = 255;

            if (srcFormat == _DXT1 && color0 <= color1) {
                // 3-color mode - vectorized
                for (int i = 0; i < 3; i++) {
                    refColors[2][i] = (XBYTE)((refColors[0][i] + refColors[1][i]) / 2);
                }
                refColors[2][3] = 255;
                refColors[3][0] = refColors[3][1] = refColors[3][2] = refColors[3][3] = 0;
            } else {
                // 4-color mode - vectorized
                for (int i = 0; i < 3; i++) {
                    refColors[2][i] = (XBYTE)((2 * refColors[0][i] + refColors[1][i]) / 3);
                    refColors[3][i] = (XBYTE)((refColors[0][i] + 2 * refColors[1][i]) / 3);
                }
                refColors[2][3] = refColors[3][3] = 255;
            }

            // Handle alpha values
            alignas(16) XBYTE alphaValues[16];
            for (int i = 0; i < 16; i++) alphaValues[i] = 255;

            if (srcFormat == _DXT3) {
                DecompressDXT3Alpha(blockPtr, alphaValues);
            } else if (srcFormat == _DXT5) {
                DecompressDXT5Alpha(blockPtr, alphaValues);
            }

            // Write pixels
            for (int y = 0; y < 4; y++) {
                const int pixelY = by * 4 + y;
                if (pixelY >= dst_desc.Height) break;

                for (int x = 0; x < 4; x++) {
                    const int pixelX = bx * 4 + x;
                    if (pixelX >= dst_desc.Width) break;

                    const int pixelIdx = y * 4 + x;
                    const XDWORD colorIdx = (colorIndices >> (2 * pixelIdx)) & 0x3;

                    XBYTE r = refColors[colorIdx][0];
                    XBYTE g = refColors[colorIdx][1];
                    XBYTE b = refColors[colorIdx][2];
                    XBYTE a = refColors[colorIdx][3];

                    if (srcFormat == _DXT3 || srcFormat == _DXT5) {
                        a = alphaValues[pixelIdx];
                    } else if (srcFormat == _DXT1 && colorIdx == 3 && color0 <= color1) {
                        a = 0;
                    }

                    const XULONG destPixel = createCtx.CreatePixel(r, g, b, a);
                    WritePixelAt(dst_desc.Image, dst_desc.BytesPerLine, pixelX, pixelY, dstBpp, destPixel);
                }
            }
        }
    }

    return TRUE;
}

//------------------------------------------------------------------------------
// Image blitting and conversion functions
//------------------------------------------------------------------------------

static void ConvertFormats(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    if (!ValidateImageDesc(src_desc) || !ValidateImageDesc(dst_desc)) return;
    if (src_desc.Width != dst_desc.Width || src_desc.Height != dst_desc.Height) return;

    const VX_PIXELFORMAT srcFormat = VxImageDesc2PixelFormat(src_desc);
    const VX_PIXELFORMAT dstFormat = VxImageDesc2PixelFormat(dst_desc);

    // Handle DXT cases
    if (srcFormat >= _DXT1 && srcFormat <= _DXT5) {
        if (dstFormat >= _DXT1 && dstFormat <= _DXT5) return;
        VxDecompressDXT(src_desc, dst_desc);
        return;
    } else if (dstFormat >= _DXT1 && dstFormat <= _DXT5) {
        VxConvertToDXT(src_desc, dst_desc, TRUE);
        return;
    }

    const int srcBpp = src_desc.BitsPerPixel / 8;
    const int dstBpp = dst_desc.BitsPerPixel / 8;
    if (srcBpp <= 0 || srcBpp > 4 || dstBpp <= 0 || dstBpp > 4) return;

    // Fast path for identical format conversion
    if (srcFormat == dstFormat && srcFormat != UNKNOWN_PF) {
        const int copyBytes = src_desc.Width * srcBpp;
        const XBYTE *srcPtr = src_desc.Image;
        XBYTE *dstPtr = dst_desc.Image;

        // Process rows with prefetching
        for (int y = 0; y < src_desc.Height; y++) {
            // Prefetch next row
            if (y + 1 < src_desc.Height) {
                #ifdef _MSC_VER
                _mm_prefetch((const char*)(srcPtr + src_desc.BytesPerLine), _MM_HINT_T0);
                #endif
            }

            memcpy(dstPtr, srcPtr, copyBytes);
            srcPtr += src_desc.BytesPerLine;
            dstPtr += dst_desc.BytesPerLine;
        }
        return;
    }

    // Pre-compute conversion context once
    PixelConvertContext convertCtx(src_desc, dst_desc);

    // 32-bit to 32-bit conversion with batch processing
    if (srcBpp == 4 && dstBpp == 4) {
        for (int y = 0; y < src_desc.Height; y++) {
            const XULONG *srcLine = (const XULONG *)(src_desc.Image + y * src_desc.BytesPerLine);
            XULONG *dstLine = (XULONG *)(dst_desc.Image + y * dst_desc.BytesPerLine);

            // Prefetch next row
            if (y + 1 < src_desc.Height) {
                #ifdef _MSC_VER
                _mm_prefetch((const char*)(srcLine + src_desc.BytesPerLine/4), _MM_HINT_T0);
                #endif
            }

            ConvertPixelBatch(srcLine, dstLine, src_desc.Width, convertCtx);
        }
        return;
    }

    // General case
    for (int y = 0; y < src_desc.Height; y++) {
        // Pre-compute row pointers
        const XBYTE *srcRow = src_desc.Image + y * src_desc.BytesPerLine;
        XBYTE *dstRow = dst_desc.Image + y * dst_desc.BytesPerLine;

        for (int x = 0; x < src_desc.Width; x++) {
            const XULONG pixel = ReadPixel(srcRow + x * srcBpp, srcBpp);
            const XULONG dstPixel = convertCtx.ConvertPixel(pixel);
            WritePixel(dstRow + x * dstBpp, dstBpp, dstPixel);
        }
    }
}

void VxDoBlit(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    if (!ValidateImageDesc(src_desc) || !ValidateImageDesc(dst_desc)) return;

    const VX_PIXELFORMAT srcFormat = VxImageDesc2PixelFormat(src_desc);
    const VX_PIXELFORMAT dstFormat = VxImageDesc2PixelFormat(dst_desc);

    // Fast path for identical format and dimensions
    if (srcFormat == dstFormat && srcFormat != UNKNOWN_PF &&
        src_desc.Width == dst_desc.Width && src_desc.Height == dst_desc.Height) {

        if (src_desc.BytesPerLine == dst_desc.BytesPerLine) {
            // Single memcpy for packed data
            if (src_desc.Height > 0 && src_desc.BytesPerLine > SIZE_MAX / src_desc.Height) return;
            const size_t totalBytes = src_desc.BytesPerLine * src_desc.Height;
            memcpy(dst_desc.Image, src_desc.Image, totalBytes);
        } else {
            // Row-by-row copy with prefetching
            const int copyBytes = src_desc.Width * (src_desc.BitsPerPixel / 8);
            const XBYTE *srcPtr = src_desc.Image;
            XBYTE *dstPtr = dst_desc.Image;

            for (int y = 0; y < src_desc.Height; y++) {
                // Prefetch next row
                if (y + 1 < src_desc.Height) {
                    #ifdef _MSC_VER
                    _mm_prefetch((const char*)(srcPtr + src_desc.BytesPerLine), _MM_HINT_T0);
                    #endif
                }

                memcpy(dstPtr, srcPtr, copyBytes);
                srcPtr += src_desc.BytesPerLine;
                dstPtr += dst_desc.BytesPerLine;
            }
        }
        return;
    }

    // Handle DXT format cases
    if ((srcFormat >= _DXT1 && srcFormat <= _DXT5) || (dstFormat >= _DXT1 && dstFormat <= _DXT5)) {
        if (srcFormat < _DXT1 && dstFormat >= _DXT1 && dstFormat <= _DXT5) {
            // Convert to DXT
            if (src_desc.Width == dst_desc.Width && src_desc.Height == dst_desc.Height) {
                VxConvertToDXT(src_desc, dst_desc, TRUE);
            } else {
                // Resize then compress
                VxImageDescEx tempDesc;
                tempDesc.Width = dst_desc.Width;
                tempDesc.Height = dst_desc.Height;
                tempDesc.BitsPerPixel = 32;
                tempDesc.BytesPerLine = (dst_desc.Width * 4 + 15) & ~15; // 16-byte aligned for SIMD
                tempDesc.RedMask = R_MASK;
                tempDesc.GreenMask = G_MASK;
                tempDesc.BlueMask = B_MASK;
                tempDesc.AlphaMask = A_MASK;

                if (tempDesc.Height > 0 && tempDesc.BytesPerLine > INT_MAX / tempDesc.Height) return;

                XBYTE *tempBuffer = (XBYTE*)_aligned_malloc(tempDesc.BytesPerLine * tempDesc.Height, 16);
                if (!tempBuffer) return;

                tempDesc.Image = tempBuffer;

                stbir_resize_uint8_linear(
                    src_desc.Image, src_desc.Width, src_desc.Height, src_desc.BytesPerLine,
                    tempDesc.Image, tempDesc.Width, tempDesc.Height, tempDesc.BytesPerLine,
                    GetStbPixelLayout(src_desc)
                );

                VxConvertToDXT(tempDesc, dst_desc, TRUE);
                _aligned_free(tempBuffer);
            }
            return;
        } else if (srcFormat >= _DXT1 && srcFormat <= _DXT5 && dstFormat < _DXT1) {
            // Decompress DXT
            VxImageDescEx tempDesc;
            tempDesc.Width = src_desc.Width;
            tempDesc.Height = src_desc.Height;
            tempDesc.BitsPerPixel = 32;
            tempDesc.BytesPerLine = (src_desc.Width * 4 + 15) & ~15; // 16-byte aligned
            tempDesc.RedMask = R_MASK;
            tempDesc.GreenMask = G_MASK;
            tempDesc.BlueMask = B_MASK;
            tempDesc.AlphaMask = A_MASK;

            if (tempDesc.Height > 0 && tempDesc.BytesPerLine > INT_MAX / tempDesc.Height) return;

            XBYTE *tempBuffer = (XBYTE*)_aligned_malloc(tempDesc.BytesPerLine * tempDesc.Height, 16);
            if (!tempBuffer) return;

            tempDesc.Image = tempBuffer;

            if (VxDecompressDXT(src_desc, tempDesc)) {
                if (src_desc.Width == dst_desc.Width && src_desc.Height == dst_desc.Height) {
                    ConvertFormats(tempDesc, dst_desc);
                } else {
                    stbir_resize_uint8_linear(
                        tempDesc.Image, tempDesc.Width, tempDesc.Height, tempDesc.BytesPerLine,
                        dst_desc.Image, dst_desc.Width, dst_desc.Height, dst_desc.BytesPerLine,
                        GetStbPixelLayout(tempDesc)
                    );
                }
            }

            _aligned_free(tempBuffer);
            return;
        }

        return; // Can't convert between DXT formats
    }

    // Same dimensions, different format
    if (src_desc.Width == dst_desc.Width && src_desc.Height == dst_desc.Height) {
        ConvertFormats(src_desc, dst_desc);
        return;
    }

    // General resize with format conversion
    const stbir_pixel_layout srcLayout = GetStbPixelLayout(src_desc);
    const stbir_datatype dataType = GetStbDataType(src_desc);

    stbir_resize(
        src_desc.Image, src_desc.Width, src_desc.Height, src_desc.BytesPerLine,
        dst_desc.Image, dst_desc.Width, dst_desc.Height, dst_desc.BytesPerLine,
        srcLayout, dataType, STBIR_EDGE_CLAMP, STBIR_FILTER_MITCHELL
    );
}

void VxDoBlitUpsideDown(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    if (!ValidateImageDesc(src_desc) || !ValidateImageDesc(dst_desc)) return;
    if (src_desc.Width != dst_desc.Width || src_desc.Height != dst_desc.Height) return;

    const VX_PIXELFORMAT srcFormat = VxImageDesc2PixelFormat(src_desc);
    const VX_PIXELFORMAT dstFormat = VxImageDesc2PixelFormat(dst_desc);

    // DXT not supported for flipping
    if ((srcFormat >= _DXT1 && srcFormat <= _DXT5) || (dstFormat >= _DXT1 && dstFormat <= _DXT5)) return;

    // Fast path for identical format
    if (srcFormat == dstFormat && srcFormat != UNKNOWN_PF) {
        const int bytesPerLine = src_desc.Width * (src_desc.BitsPerPixel / 8);

        for (int y = 0; y < src_desc.Height; y++) {
            const XBYTE *srcLine = src_desc.Image + y * src_desc.BytesPerLine;
            XBYTE *dstLine = dst_desc.Image + (src_desc.Height - 1 - y) * dst_desc.BytesPerLine;

            if (y + 1 < src_desc.Height) {
                #ifdef _MSC_VER
                _mm_prefetch((const char*)(srcLine + src_desc.BytesPerLine), _MM_HINT_T0);
                #endif
            }

            memcpy(dstLine, srcLine, bytesPerLine);
        }
        return;
    }

    // Format conversion while flipping
    const int srcBpp = src_desc.BitsPerPixel / 8;
    const int dstBpp = dst_desc.BitsPerPixel / 8;

    if (srcBpp <= 0 || srcBpp > 4 || dstBpp <= 0 || dstBpp > 4) return;

    // Pre-compute conversion context once
    const PixelConvertContext convertCtx(src_desc, dst_desc);

    for (int y = 0; y < src_desc.Height; y++) {
        const XBYTE *srcRow = src_desc.Image + y * src_desc.BytesPerLine;
        XBYTE *dstRow = dst_desc.Image + (src_desc.Height - 1 - y) * dst_desc.BytesPerLine;

        // For 32-bit, use batch processing
        if (srcBpp == 4 && dstBpp == 4) {
            const XULONG *srcPixels = (const XULONG *)srcRow;
            XULONG *dstPixels = (XULONG *)dstRow;
            ConvertPixelBatch(srcPixels, dstPixels, src_desc.Width, convertCtx);
        } else {
            // General case
            for (int x = 0; x < src_desc.Width; x++) {
                const XULONG pixel = ReadPixel(srcRow + x * srcBpp, srcBpp);
                const XULONG dstPixel = convertCtx.ConvertPixel(pixel);
                WritePixel(dstRow + x * dstBpp, dstBpp, dstPixel);
            }
        }
    }
}

void VxDoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE AlphaValue) {
    if (!dst_desc.AlphaMask || !dst_desc.Image) return;

    const VX_PIXELFORMAT format = VxImageDesc2PixelFormat(dst_desc);
    if (format >= _DXT1 && format <= _DXT5) return;

    const int alphaShift = static_cast<int>(GetBitShift(dst_desc.AlphaMask));
    const int alphaBits = static_cast<int>(GetBitCount(dst_desc.AlphaMask));
    const int bytesPerPixel = dst_desc.BitsPerPixel / 8;

    // Scale alpha to bit depth
    if (alphaBits < 8) {
        AlphaValue = (AlphaValue * ((1 << alphaBits) - 1)) / 255;
    }

    // Fast path for 32-bit ARGB with batch processing
    if (dst_desc.BitsPerPixel == 32 && dst_desc.AlphaMask == A_MASK) {
        for (int y = 0; y < dst_desc.Height; y++) {
            XULONG *row = (XULONG *)(dst_desc.Image + y * dst_desc.BytesPerLine);
            ApplyAlphaBatch(row, dst_desc.Width, AlphaValue, dst_desc.AlphaMask, 24);
        }
        return;
    }

    // General case
    for (int y = 0; y < dst_desc.Height; y++) {
        XBYTE *row = dst_desc.Image + y * dst_desc.BytesPerLine;
        for (int x = 0; x < dst_desc.Width; x++) {
            XBYTE *pixel = row + x * bytesPerPixel;

            switch (dst_desc.BitsPerPixel) {
            case 16: {
                XWORD *p = (XWORD *)pixel;
                *p = (*p & ~dst_desc.AlphaMask) | ((AlphaValue << alphaShift) & dst_desc.AlphaMask);
                break;
            }
            case 24: {
                int byteOffset = alphaShift / 8;
                int bitOffset = alphaShift % 8;
                XBYTE byteMask = (dst_desc.AlphaMask >> (byteOffset * 8)) & 0xFF;
                pixel[byteOffset] = (pixel[byteOffset] & ~byteMask) | ((AlphaValue << bitOffset) & byteMask);
                break;
            }
            case 32: {
                XULONG *p = (XULONG *)pixel;
                *p = (*p & ~dst_desc.AlphaMask) | ((AlphaValue << alphaShift) & dst_desc.AlphaMask);
                break;
            }
            }
        }
    }
}

void VxDoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE *AlphaValues) {
    if (!dst_desc.AlphaMask || !dst_desc.Image || !AlphaValues) return;

    const VX_PIXELFORMAT format = VxImageDesc2PixelFormat(dst_desc);
    if (format >= _DXT1 && format <= _DXT5) return;

    const int alphaShift = static_cast<int>(GetBitShift(dst_desc.AlphaMask));
    const int alphaBits = static_cast<int>(GetBitCount(dst_desc.AlphaMask));
    const int bytesPerPixel = dst_desc.BitsPerPixel / 8;

    // Fast path for 32-bit ARGB with batch processing
    if (dst_desc.BitsPerPixel == 32 && dst_desc.AlphaMask == A_MASK) {
        for (int y = 0; y < dst_desc.Height; y++) {
            XULONG *row = (XULONG *)(dst_desc.Image + y * dst_desc.BytesPerLine);
            const XBYTE *alphaRow = AlphaValues + y * dst_desc.Width;
            ApplyVariableAlphaBatch(row, alphaRow, dst_desc.Width, dst_desc.AlphaMask, 24);
        }
        return;
    }

    // General case
    for (int y = 0; y < dst_desc.Height; y++) {
        XBYTE *row = dst_desc.Image + y * dst_desc.BytesPerLine;
        for (int x = 0; x < dst_desc.Width; x++) {
            XBYTE *pixel = row + x * bytesPerPixel;
            const int index = y * dst_desc.Width + x;
            XBYTE alpha = AlphaValues[index];

            if (alphaBits < 8) {
                alpha = (alpha * ((1 << alphaBits) - 1)) / 255;
            }

            switch (dst_desc.BitsPerPixel) {
            case 16: {
                XWORD *p = (XWORD *)pixel;
                *p = (*p & ~dst_desc.AlphaMask) | ((alpha << alphaShift) & dst_desc.AlphaMask);
                break;
            }
            case 24: {
                int byteOffset = alphaShift / 8;
                int bitOffset = alphaShift % 8;
                XBYTE byteMask = (dst_desc.AlphaMask >> (byteOffset * 8)) & 0xFF;
                pixel[byteOffset] = (pixel[byteOffset] & ~byteMask) | ((alpha << bitOffset) & byteMask);
                break;
            }
            case 32: {
                XULONG *p = (XULONG *)pixel;
                *p = (*p & ~dst_desc.AlphaMask) | ((alpha << alphaShift) & dst_desc.AlphaMask);
                break;
            }
            }
        }
    }
}

void VxResizeImage32(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc) {
    if (src_desc.BitsPerPixel != 32 || dst_desc.BitsPerPixel != 32) return;
    if (!ValidateImageDesc(src_desc) || !ValidateImageDesc(dst_desc)) return;

    const VX_PIXELFORMAT srcFormat = VxImageDesc2PixelFormat(src_desc);
    const VX_PIXELFORMAT dstFormat = VxImageDesc2PixelFormat(dst_desc);

    if ((srcFormat >= _DXT1 && srcFormat <= _DXT5) || (dstFormat >= _DXT1 && dstFormat <= _DXT5)) return;

    const stbir_pixel_layout layout = GetStbPixelLayout(src_desc);

    stbir_resize_uint8_linear(
        src_desc.Image, src_desc.Width, src_desc.Height, src_desc.BytesPerLine,
        dst_desc.Image, dst_desc.Width, dst_desc.Height, dst_desc.BytesPerLine,
        layout
    );
}

void VxGenerateMipMap(const VxImageDescEx &src_desc, XBYTE *Buffer) {
    if (src_desc.BitsPerPixel != 32 || !src_desc.Image || !Buffer) return;
    if (src_desc.Width < 2 || src_desc.Height < 2) return;

    const VX_PIXELFORMAT srcFormat = VxImageDesc2PixelFormat(src_desc);
    if (srcFormat >= _DXT1 && srcFormat <= _DXT5) return;

    const int dstWidth = src_desc.Width / 2;
    const int dstHeight = src_desc.Height / 2;
    const int dstPitch = dstWidth * 4;

    const stbir_pixel_layout layout = GetStbPixelLayout(src_desc);

    stbir_resize_uint8_linear(src_desc.Image, src_desc.Width, src_desc.Height, src_desc.BytesPerLine,
                              Buffer, dstWidth, dstHeight, dstPitch, layout);
}

//------------------------------------------------------------------------------
// Normal and bump map generation
//------------------------------------------------------------------------------

XBOOL VxConvertToNormalMap(const VxImageDescEx &image, XULONG ColorMask) {
    if (image.BitsPerPixel != 32 || !image.Image) return FALSE;
    if (image.Width <= 0 || image.Height <= 0) return FALSE;

    const VX_PIXELFORMAT format = VxImageDesc2PixelFormat(image);
    if (format >= _DXT1 && format <= _DXT5) return FALSE;

    const XULONG actualMask = (ColorMask == 0xFFFFFFFF) ? image.RedMask | image.GreenMask | image.BlueMask : ColorMask;
    if (actualMask == 0) return FALSE;

    // Check for integer overflow before allocation
    if (image.Width > 0 && image.Height > INT_MAX / image.Width) return FALSE;
    const int pixelCount = image.Width * image.Height;

    // Check for size_t overflow in allocation
    if (pixelCount > SIZE_MAX / sizeof(float)) return FALSE;
    float *heightMap = (float*)_aligned_malloc(pixelCount * sizeof(float), 32);
    if (!heightMap) return FALSE;

    // Extract height values
    ExtractContext extractCtx(image);

    for (int y = 0; y < image.Height; y++) {
        for (int x = 0; x < image.Width; x++) {
            const int index = y * image.Width + x;

            const XULONG pixel = ReadPixelAt(image.Image, image.BytesPerLine, x, y, 4);

            if (ColorMask == 0xFFFFFFFF) {
                XBYTE r, g, b, a;
                extractCtx.ExtractRGBA(pixel, r, g, b, a);
                heightMap[index] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
            } else {
                const XULONG shift = GetBitShift(actualMask);
                const XULONG bits = GetBitCount(actualMask);
                XULONG val = (pixel & actualMask) >> shift;

                if (bits > 0 && bits < 8) {
                    val = (val * 255) / ((1 << bits) - 1);
                }
                heightMap[index] = (float)val / 255.0f;
            }
        }
    }

    // Calculate normals using Sobel operator
    const float scale = 2.0f;

    // Pre-compute pixel creation context
    const PixelCreateContext createCtx(image);

    for (int y = 0; y < image.Height; y++) {
        // Pre-compute row pointers
        XULONG *pixelRow = (XULONG *)(image.Image + y * image.BytesPerLine);
        const XULONG *originalRow = pixelRow;

        // Process pixels in batches
        for (int x = 0; x < image.Width; x++) {
            auto getHeight = [&](int px, int py) -> float {
                px = std::clamp(px, 0, image.Width - 1);
                py = std::clamp(py, 0, image.Height - 1);
                return heightMap[py * image.Width + px];
            };

            // Sobel filter
            const float h00 = getHeight(x - 1, y - 1), h01 = getHeight(x, y - 1), h02 = getHeight(x + 1, y - 1);
            const float h10 = getHeight(x - 1, y),                                    h12 = getHeight(x + 1, y);
            const float h20 = getHeight(x - 1, y + 1), h21 = getHeight(x, y + 1), h22 = getHeight(x + 1, y + 1);

            const float gx = (h00 + 2.0f * h10 + h20) - (h02 + 2.0f * h12 + h22);
            const float gy = (h00 + 2.0f * h01 + h02) - (h20 + 2.0f * h21 + h22);

            float nx = -gx * scale;
            float ny = -gy * scale;
            float nz = 1.0f;

            const float length = sqrtf(nx * nx + ny * ny + nz * nz);
            if (length > 0.0001f) {
                const float invLength = 1.0f / length;
                nx *= invLength;
                ny *= invLength;
                nz *= invLength;
            } else {
                nx = ny = 0.0f;
                nz = 1.0f;
            }

            // Color conversion
            int r = (int)((nx + 1.0f) * 127.5f);
            int g = (int)((ny + 1.0f) * 127.5f);
            int b = (int)((nz + 1.0f) * 127.5f);

            r = std::clamp(r, 0, 255);
            g = std::clamp(g, 0, 255);
            b = std::clamp(b, 0, 255);

            // Get original alpha from the pixel
            XBYTE originalR, originalG, originalB, originalA;
            extractCtx.ExtractRGBA(originalRow[x], originalR, originalG, originalB, originalA);

            pixelRow[x] = createCtx.CreatePixel(r, g, b, originalA);
        }
    }

    _aligned_free(heightMap);
    return TRUE;
}

XBOOL VxConvertToBumpMap(const VxImageDescEx &image) {
    if (image.BitsPerPixel != 32 || !image.Image) return FALSE;
    if (image.Width <= 0 || image.Height <= 0) return FALSE;

    const VX_PIXELFORMAT format = VxImageDesc2PixelFormat(image);
    if (format >= _DXT1 && format <= _DXT5) return FALSE;

    // Pre-compute extraction context
    ExtractContext extractCtx(image);

    // Pre-compute pixel creation context
    const PixelCreateContext createCtx(image);

    for (int y = 0; y < image.Height; y++) {
        XULONG *pixelRow = (XULONG *)(image.Image + y * image.BytesPerLine);

        for (int x = 0; x < image.Width; x++) {
            XBYTE r, g, b, a;
            extractCtx.ExtractRGBA(pixelRow[x], r, g, b, a);

            const float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

            auto getSafePixel = [&](int px, int py) -> float {
                if (px < 0 || px >= image.Width || py < 0 || py >= image.Height) {
                    return luminance;
                }

                const XULONG pixel = ReadPixelAt(image.Image, image.BytesPerLine, px, py, 4);
                XBYTE pr, pg, pb, pa;
                extractCtx.ExtractRGBA(pixel, pr, pg, pb, pa);
                return 0.299f * pr + 0.587f * pg + 0.114f * pb;
            };

            const float leftLum = getSafePixel(x - 1, y);
            const float rightLum = getSafePixel(x + 1, y);
            const float du = (rightLum - leftLum) * 0.5f;

            const float topLum = getSafePixel(x, y - 1);
            const float bottomLum = getSafePixel(x, y + 1);
            const float dv = (bottomLum - topLum) * 0.5f;

            int duVal = std::clamp((int)(du + 128.0f), 0, 255);
            int dvVal = std::clamp((int)(dv + 128.0f), 0, 255);
            int lumVal = std::clamp((int)luminance, 0, 255);

            pixelRow[x] = createCtx.CreatePixel(duVal, dvVal, lumVal, a);
        }
    }

    return TRUE;
}

//------------------------------------------------------------------------------
// Quantization utilities
//------------------------------------------------------------------------------

int GetQuantizationSamplingFactor() {
    return g_QuantizationSamplingFactor;
}

void SetQuantizationSamplingFactor(int factor) {
    g_QuantizationSamplingFactor = (factor < 1) ? 1 : (factor > 30) ? 30 : factor;
}
