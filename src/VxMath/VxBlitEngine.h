#ifndef VXBLITENGINE_H
#define VXBLITENGINE_H

#include "VxMathDefines.h"
#include "VxImageDescEx.h"
#include "XArray.h"

// Suppress C4251 warning for DLL interface
#if VX_MSVC > 1000
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

/**
 * @brief Structure used for line blitting operations.
 *
 * This structure contains all the information needed to perform a single
 * scanline blit operation between source and destination images.
 */
struct VxBlitInfo {
    const XBYTE *srcLine;   ///< Pointer to the source scanline
    XBYTE *dstLine;         ///< Pointer to the destination scanline
    XDWORD srcBytesPerLine; ///< Source bytes per line (stride)
    XDWORD dstBytesPerLine; ///< Destination bytes per line (stride)
    int width;              ///< Number of pixels to process
    int copyBytes;          ///< Number of bytes to copy per line

    // Source image descriptor (partial copy)
    XDWORD srcBytesPerPixel;
    XDWORD srcRedMask;
    XDWORD srcGreenMask;
    XDWORD srcBlueMask;
    XDWORD srcAlphaMask;

    // Destination image descriptor (partial copy)
    XDWORD dstBytesPerPixel;
    XDWORD dstRedMask;
    XDWORD dstGreenMask;
    XDWORD dstBlueMask;
    XDWORD dstAlphaMask;

    // Pre-computed shift values (position of LSB of each mask)
    int redShiftSrc;
    int greenShiftSrc;
    int blueShiftSrc;
    int alphaShiftSrc;
    int redShiftDst;
    int greenShiftDst;
    int blueShiftDst;
    int alphaShiftDst;

    // Bit widths of each channel
    int redBitsSrc;
    int greenBitsSrc;
    int blueBitsSrc;
    int alphaBitsSrc;
    int redBitsDst;
    int greenBitsDst;
    int blueBitsDst;
    int alphaBitsDst;

    // Inverted alpha mask for clearing
    XDWORD alphaMaskInv;

    // Alpha value for constant alpha blit
    XDWORD alphaValue;

    // Color map (for paletted images)
    const XBYTE *colorMap;
    int colorMapEntries;
    int bytesPerColorEntry;
};

/**
 * @brief Structure used for image resizing operations.
 *
 * Contains all the parameters needed for bilinear filtered image resizing.
 */
struct VxResizeInfo {
    int wr1;           ///< Width ratio 1 (srcWidth << 16 / dstWidth) - fixed point
    int hr1;           ///< Height ratio 1 (srcHeight << 16 / dstHeight) - fixed point
    int w1;            ///< Source width
    int w2;            ///< Destination width
    int h1;            ///< Source height
    int h2;            ///< Destination height
    int srcPitch;      ///< Source bytes per line / 4 (for 32-bit)
    XDWORD *dstRow;    ///< Pointer to destination row
    const XDWORD *srcRow; ///< Pointer to source row
};

/// Function pointer type for line blitting operations
typedef void (*VxBlitLineFunc)(const VxBlitInfo *info);

/**
 * @class VxBlitEngine
 * @brief High-performance image blitting and format conversion engine.
 *
 * VxBlitEngine provides optimized routines for:
 * - Format conversion between different pixel formats
 * - Image resizing with bilinear filtering
 * - Alpha channel manipulation
 * - Upside-down blitting for coordinate system conversions
 *
 * The engine uses a dispatch table to select the optimal routine for each
 * source/destination format combination, with special optimizations for
 * common formats like 32-bit ARGB.
 *
 * @remarks
 * This class is designed to match the original Virtools VxBlitEngine behavior.
 * A global instance (TheBlitter) is created and used by the VxDoBlit functions.
 */
class VX_EXPORT VxBlitEngine {
public:
    /**
     * @brief Default constructor. Initializes internal structures.
     */
    VxBlitEngine();

    /**
     * @brief Destructor. Releases allocated resources.
     */
    ~VxBlitEngine();

    /**
     * @brief Performs a blit operation with optional format conversion and resizing.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     */
    void DoBlit(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Performs a blit operation, flipping the image vertically.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     */
    void DoBlitUpsideDown(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Sets the alpha channel of an image to a constant value.
     * @param dst_desc Destination image descriptor.
     * @param AlphaValue The alpha value to set (0-255).
     */
    void DoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE AlphaValue);

    /**
     * @brief Sets the alpha channel of an image using an array of alpha values.
     * @param dst_desc Destination image descriptor.
     * @param AlphaValues Array of alpha values, one per pixel.
     */
    void DoAlphaBlit(const VxImageDescEx &dst_desc, XBYTE *AlphaValues);

    /**
     * @brief Resizes a 32-bit image using bilinear filtering.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     */
    void ResizeImage(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Quantizes a 24/32-bit image to 8-bit paletted format using NeuQuant.
     * @param src_desc Source image descriptor (24 or 32-bit).
     * @param dst_desc Destination image descriptor (8-bit paletted).
     * @return TRUE if quantization succeeded, FALSE otherwise.
     *
     * Uses the NeuQuant neural-net image quantization algorithm by Anthony Dekker.
     * The quality can be controlled using SetQuantizationSamplingFactor().
     */
    XBOOL QuantizeImage(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Quantizes a 24/32-bit image using the median-cut algorithm.
     * @param src_desc Source image descriptor (24 or 32-bit).
     * @param dst_desc Destination image descriptor (8-bit paletted).
     * @return TRUE if quantization succeeded, FALSE otherwise.
     */
    XBOOL QuantizeImageMedianCut(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Converts a VxImageDescEx to a VX_PIXELFORMAT enum.
     * @param desc The image descriptor to analyze.
     * @return The corresponding VX_PIXELFORMAT value, or UNKNOWN_PF if not recognized.
     */
    static VX_PIXELFORMAT GetPixelFormat(const VxImageDescEx &desc);

    /**
     * @brief Fills an image descriptor with the parameters for a given pixel format.
     * @param Pf The pixel format.
     * @param desc The descriptor to fill.
     */
    static void ConvertPixelFormat(VX_PIXELFORMAT Pf, VxImageDescEx &desc);

    /**
     * @brief Returns a human-readable string for a pixel format.
     * @param Pf The pixel format.
     * @return A string describing the format.
     */
    static const char *PixelFormat2String(VX_PIXELFORMAT Pf);

    /**
     * @brief Fills an image with a solid color.
     * @param dst_desc Destination image descriptor.
     * @param color The color to fill with (format-dependent, typically ARGB).
     */
    void FillImage(const VxImageDescEx &dst_desc, XDWORD color);

    /**
     * @brief Converts an image to premultiplied alpha format in-place.
     * @param desc Image descriptor (must be 32-bit ARGB).
     *
     * Premultiplied alpha means RGB values are multiplied by the alpha value,
     * which is useful for efficient alpha blending operations.
     */
    void PremultiplyAlpha(const VxImageDescEx &desc);

    /**
     * @brief Converts an image from premultiplied alpha back to straight alpha.
     * @param desc Image descriptor (must be 32-bit ARGB).
     *
     * This reverses the PremultiplyAlpha operation, dividing RGB values by alpha.
     */
    void UnpremultiplyAlpha(const VxImageDescEx &desc);

    /**
     * @brief Swaps R and B channels in an image (ARGB <-> ABGR conversion).
     * @param desc Image descriptor (must be 32-bit).
     */
    void SwapRedBlue(const VxImageDescEx &desc);

    /**
     * @brief Clears the alpha channel to 0 (fully transparent).
     * @param desc Image descriptor (must be 32-bit).
     */
    void ClearAlpha(const VxImageDescEx &desc);

    /**
     * @brief Sets the alpha channel to 255 (fully opaque).
     * @param desc Image descriptor (must be 32-bit).
     */
    void SetFullAlpha(const VxImageDescEx &desc);

    /**
     * @brief Inverts the RGB color channels while preserving alpha.
     * @param desc Image descriptor (must be 32-bit).
     */
    void InvertColors(const VxImageDescEx &desc);

    /**
     * @brief Converts an image to grayscale using luminance formula.
     * @param desc Image descriptor (must be 32-bit).
     *
     * Uses the standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
     */
    void ConvertToGrayscale(const VxImageDescEx &desc);

    /**
     * @brief Performs multiply blend between source and destination.
     * @param src_desc Source image descriptor (must be 32-bit).
     * @param dst_desc Destination image descriptor (must be 32-bit, modified in-place).
     *
     * Each channel: dst = src * dst / 255
     */
    void MultiplyBlend(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Initializes the generic (C) blitting function table.
     */
    void BuildGenericTables();

    /**
     * @brief Initializes the x86-optimized blitting function table.
     */
    void BuildX86Table();

    /**
     * @brief Initializes the SSE-optimized blitting function table (if available).
     *
     * Uses simde for portable SSE intrinsics across platforms.
     */
    void BuildSSETable();

private:
    /**
     * @brief Gets the appropriate blitting function for the given formats.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     * @return Pointer to the blitting function, or NULL if no suitable function.
     */
    VxBlitLineFunc GetBlitFunction(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Gets the appropriate set-alpha function for the given format.
     * @param dst_desc Destination image descriptor.
     * @return Pointer to the set-alpha function, or NULL if not applicable.
     */
    VxBlitLineFunc GetSetAlphaFunction(const VxImageDescEx &dst_desc);

    /**
     * @brief Gets the appropriate copy-alpha function for the given format.
     * @param dst_desc Destination image descriptor.
     * @return Pointer to the copy-alpha function, or NULL if not applicable.
     */
    VxBlitLineFunc GetCopyAlphaFunction(const VxImageDescEx &dst_desc);

    /**
     * @brief Sets up the VxBlitInfo structure for a blit operation.
     * @param info The structure to fill.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     */
    void SetupBlitInfo(VxBlitInfo &info, const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief High-performance bilinear resize for 32-bit images.
     * Uses SSE2 when available for optimal performance.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     */
    void ResizeBilinear32(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Bilinear resize for 24-bit images.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     */
    void ResizeBilinear24(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Nearest-neighbor resize fallback for non-standard formats.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     */
    void ResizeNearestNeighbor(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

    /**
     * @brief Internal blit with resize using fixed-point scanline functions.
     * Called by DoBlit when source is 32-bit and dimensions differ.
     * @param src_desc Source image descriptor.
     * @param dst_desc Destination image descriptor.
     * @param blitFunc The blit function to use for format conversion.
     */
    void DoBlitWithResize(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc, VxBlitLineFunc blitFunc);

    // Function dispatch tables
    // Layout: [srcBpp][dstBpp] for generic, [srcFormat][dstFormat] for specific
    static const int TABLE_SIZE = 16;        // 16 entries per dimension
    static const int FORMAT_TABLE_SIZE = 19; // Up to format 18 (before DXT)
    static const int ALPHA_TABLE_SIZE = 4;   // 8, 16, 24, 32 bits

    // Generic blitting functions indexed by [srcBytesPerPixel-1][dstBytesPerPixel-1]
    VxBlitLineFunc m_GenericBlitTable[4][4];

    // Specific blitting functions for known format pairs
    // Indexed by [srcFormat][dstFormat]
    VxBlitLineFunc m_SpecificBlitTable[FORMAT_TABLE_SIZE][FORMAT_TABLE_SIZE];

    // Set alpha functions indexed by bytes per pixel
    VxBlitLineFunc m_SetAlphaTable[ALPHA_TABLE_SIZE];

    // Copy alpha functions indexed by bytes per pixel
    VxBlitLineFunc m_CopyAlphaTable[ALPHA_TABLE_SIZE];

    // Paletted image conversion functions
    VxBlitLineFunc m_PalettedBlitTable[4][4]; // [paletteType][dstBytesPerPixel-1]

    // Current blit info (used during blit operations)
    VxBlitInfo m_BlitInfo;

    // Temporary buffer for resizing operations
    XArray<XDWORD> m_ResizeBuffer;
};

/**
 * @brief Global blitter instance used by VxDoBlit functions.
 */
extern VX_EXPORT VxBlitEngine TheBlitter;

#if VX_MSVC > 1000
#pragma warning(pop)
#endif

#endif // VXBLITENGINE_H
