#ifndef VXIMAGEDESCEX_H
#define VXIMAGEDESCEX_H

#include "VxMathDefines.h"

#include <string.h>

/**
 * @struct VxImageDescEx
 * @brief Describes an image's properties and provides access to its data.
 *
 * @remarks
 * This structure extends a basic image description with support for a colormap (palette),
 * a direct pointer to the image data, and other properties like dimensions and pixel format.
 */
typedef struct VxImageDescEx {
    int Size;     ///< Size of this structure, for versioning.
    XULONG Flags; ///< Flags for special formats (e.g., compression types like DXT1). 0 for standard formats.

    int Width;  ///< Width of the image in pixels.
    int Height; ///< Height of the image in pixels.
    union {
        int BytesPerLine;   ///< The pitch (number of bytes per row) of the image.
        int TotalImageSize; ///< For compressed formats, the total size of the image data in bytes.
    };

    int BitsPerPixel; ///< Number of bits per pixel.
    union {
        XULONG RedMask;    ///< Bitmask for the red component.
        XULONG BumpDuMask; ///< Bitmask for the Du component in a bump map.
    };

    union {
        XULONG GreenMask;  ///< Bitmask for the green component.
        XULONG BumpDvMask; ///< Bitmask for the Dv component in a bump map.
    };

    union {
        XULONG BlueMask;    ///< Bitmask for the blue component.
        XULONG BumpLumMask; ///< Bitmask for the luminance component in a bump map.
    };

    XULONG AlphaMask; ///< Bitmask for the alpha component.

    short BytesPerColorEntry; ///< The size of each entry in the colormap, in bytes.
    short ColorMapEntries;    ///< The number of entries in the colormap. If non-zero, the image is palettized.

    XBYTE *ColorMap; ///< Pointer to the colormap data (palette). Can be NULL.
    XBYTE *Image;    ///< Pointer to the raw image pixel data.

    /**
     * @brief Default constructor. Initializes the structure to zero and sets the Size field.
     */
    VxImageDescEx() {
        Size = sizeof(VxImageDescEx);
        memset((XBYTE *) this + 4, 0, Size - 4);
    }

    /**
     * @brief Copies data from another VxImageDescEx structure.
     * @param desc The source structure to copy from.
     */
    void Set(const VxImageDescEx &desc) {
        Size = sizeof(VxImageDescEx);
        if (desc.Size < Size)
            memset((XBYTE *) this + 4, 0, Size - 4);
        if (desc.Size > Size)
            return;
        memcpy((XBYTE *) this + 4, (XBYTE *) &desc + 4, desc.Size - 4);
    }

    /**
     * @brief Checks if the image format includes an alpha channel.
     * @return TRUE if an alpha channel is present (either via AlphaMask or a DXT compression flag), FALSE otherwise.
     */
    XBOOL HasAlpha() const {
        return ((AlphaMask != 0) || (Flags >= _DXT1));
    }

    /**
     * @brief Compares this description with another for equality.
     * @param desc The description to compare against.
     * @return A non-zero value if all fields (except Image and ColorMap pointers) are identical, otherwise zero.
     */
    int operator==(const VxImageDescEx &desc) const {
        return (Size == desc.Size &&
                Height == desc.Height && Width == desc.Width &&
                BitsPerPixel == desc.BitsPerPixel && BytesPerLine == desc.BytesPerLine &&
                RedMask == desc.RedMask && GreenMask == desc.GreenMask &&
                BlueMask == desc.BlueMask && AlphaMask == desc.AlphaMask &&
                BytesPerColorEntry == desc.BytesPerColorEntry && ColorMapEntries == desc.ColorMapEntries);
    }

    /**
     * @brief Compares this description with another for inequality.
     * @param desc The description to compare against.
     * @return A non-zero value if any field (except Image and ColorMap pointers) is different, otherwise zero.
     */
    int operator!=(const VxImageDescEx &desc) const {
        return (Size != desc.Size ||
                Height != desc.Height || Width != desc.Width ||
                BitsPerPixel != desc.BitsPerPixel || BytesPerLine != desc.BytesPerLine ||
                RedMask != desc.RedMask || GreenMask != desc.GreenMask ||
                BlueMask != desc.BlueMask || AlphaMask != desc.AlphaMask ||
                BytesPerColorEntry != desc.BytesPerColorEntry || ColorMapEntries != desc.ColorMapEntries);
    }
} VxImageDescEx;

#endif // VXIMAGEDESCEX_H
