/**
 * @file BlitEngineTestHelpers.h
 * @brief Common test utilities and helpers for VxBlitEngine tests.
 *
 * Provides image creation, comparison, pattern generation utilities,
 * and image file output for visual verification.
 */

#ifndef BLITENGINE_TEST_HELPERS_H
#define BLITENGINE_TEST_HELPERS_H

#include "VxMath.h"
#include "VxImageDescEx.h"
#include "VxMathDefines.h"
#include "VxBlitEngine.h"
#include "XString.h"
#include "CKTypes.h"
#include "CKError.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>
#include <string>
#include <cstdio>

// Platform-specific includes for directory creation
#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0755)
#endif

namespace BlitEngineTest {

//==============================================================================
// Image Output Directory Management
//==============================================================================

/**
 * @brief Creates the test output directory if it doesn't exist.
 * @return The path to the output directory.
 */
inline std::string GetTestOutputDir() {
    std::string dir = "test_output";
    MKDIR(dir.c_str());
    return dir;
}

/**
 * @brief Utility class for writing test images to files for visual verification.
 */
class ImageWriter {
public:
    /**
     * @brief Saves a 32-bit ARGB image to a PNG file.
     * @param filename Base filename (without extension).
     * @param data Pointer to image data (ARGB format).
     * @param width Image width.
     * @param height Image height.
     * @param testName Optional test name to prefix the filename.
     * @return true if successful.
     */
    static bool SavePNG32(const std::string& filename, const XBYTE* data, 
                          int width, int height, const std::string& testName = "") {
        std::string dir = GetTestOutputDir();
        std::string fullPath = dir + "/" + (testName.empty() ? "" : testName + "_") + filename + ".png";
        
        // Convert ARGB to RGBA for stb_image_write
        std::vector<XBYTE> rgba(width * height * 4);
        const XDWORD* src = reinterpret_cast<const XDWORD*>(data);
        for (int i = 0; i < width * height; ++i) {
            XDWORD pixel = src[i];
            rgba[i * 4 + 0] = (pixel >> 16) & 0xFF; // R
            rgba[i * 4 + 1] = (pixel >> 8) & 0xFF;  // G
            rgba[i * 4 + 2] = pixel & 0xFF;         // B
            rgba[i * 4 + 3] = (pixel >> 24) & 0xFF; // A
        }
        
        return stbi_write_png(fullPath.c_str(), width, height, 4, rgba.data(), width * 4) != 0;
    }

    /**
     * @brief Saves a 24-bit RGB image to a PNG file.
     */
    static bool SavePNG24(const std::string& filename, const XBYTE* data,
                          int width, int height, const std::string& testName = "") {
        std::string dir = GetTestOutputDir();
        std::string fullPath = dir + "/" + (testName.empty() ? "" : testName + "_") + filename + ".png";
        
        // Convert BGR to RGB for stb_image_write
        std::vector<XBYTE> rgb(width * height * 3);
        for (int i = 0; i < width * height; ++i) {
            rgb[i * 3 + 0] = data[i * 3 + 2]; // R (from B)
            rgb[i * 3 + 1] = data[i * 3 + 1]; // G
            rgb[i * 3 + 2] = data[i * 3 + 0]; // B (from R)
        }
        
        return stbi_write_png(fullPath.c_str(), width, height, 3, rgb.data(), width * 3) != 0;
    }

    /**
     * @brief Saves a 16-bit image to PNG by converting to 24-bit.
     * @param is565 True for 565 format, false for 555 format.
     */
    static bool SavePNG16(const std::string& filename, const XBYTE* data,
                          int width, int height, bool is565,
                          const std::string& testName = "") {
        std::string dir = GetTestOutputDir();
        std::string fullPath = dir + "/" + (testName.empty() ? "" : testName + "_") + filename + ".png";
        
        std::vector<XBYTE> rgb(width * height * 3);
        const XWORD* src = reinterpret_cast<const XWORD*>(data);
        
        for (int i = 0; i < width * height; ++i) {
            XWORD pixel = src[i];
            if (is565) {
                rgb[i * 3 + 0] = ((pixel >> 11) & 0x1F) * 255 / 31; // R
                rgb[i * 3 + 1] = ((pixel >> 5) & 0x3F) * 255 / 63;  // G
                rgb[i * 3 + 2] = (pixel & 0x1F) * 255 / 31;         // B
            } else {
                rgb[i * 3 + 0] = ((pixel >> 10) & 0x1F) * 255 / 31; // R
                rgb[i * 3 + 1] = ((pixel >> 5) & 0x1F) * 255 / 31;  // G
                rgb[i * 3 + 2] = (pixel & 0x1F) * 255 / 31;         // B
            }
        }
        
        return stbi_write_png(fullPath.c_str(), width, height, 3, rgb.data(), width * 3) != 0;
    }

    /**
     * @brief Saves an 8-bit paletted image to PNG by converting to 32-bit.
     */
    static bool SavePNG8Paletted(const std::string& filename, const XBYTE* data,
                                  const XBYTE* palette, int bytesPerEntry,
                                  int width, int height, const std::string& testName = "") {
        std::string dir = GetTestOutputDir();
        std::string fullPath = dir + "/" + (testName.empty() ? "" : testName + "_") + filename + ".png";
        
        std::vector<XBYTE> rgba(width * height * 4);
        for (int i = 0; i < width * height; ++i) {
            int idx = data[i];
            const XBYTE* entry = palette + idx * bytesPerEntry;
            rgba[i * 4 + 0] = entry[2]; // R
            rgba[i * 4 + 1] = entry[1]; // G
            rgba[i * 4 + 2] = entry[0]; // B
            rgba[i * 4 + 3] = (bytesPerEntry >= 4) ? entry[3] : 255; // A
        }
        
        return stbi_write_png(fullPath.c_str(), width, height, 4, rgba.data(), width * 4) != 0;
    }

    /**
     * @brief Generic save that auto-detects format from VxImageDescEx.
     */
    static bool SaveFromDesc(const std::string& filename, const VxImageDescEx& desc,
                             const std::string& testName = "") {
        if (!desc.Image) return false;
        
        if (desc.BitsPerPixel == 32) {
            return SavePNG32(filename, desc.Image, desc.Width, desc.Height, testName);
        } else if (desc.BitsPerPixel == 24) {
            return SavePNG24(filename, desc.Image, desc.Width, desc.Height, testName);
        } else if (desc.BitsPerPixel == 16) {
            bool is565 = (desc.GreenMask == 0x07E0);
            return SavePNG16(filename, desc.Image, desc.Width, desc.Height, is565, testName);
        } else if (desc.BitsPerPixel == 8 && desc.ColorMap) {
            return SavePNG8Paletted(filename, desc.Image, desc.ColorMap, 
                                     desc.BytesPerColorEntry, desc.Width, desc.Height, testName);
        }
        return false;
    }
};

//==============================================================================
// Image Buffer Management
//==============================================================================

/**
 * @brief RAII wrapper for image buffer allocation.
 */
class ImageBuffer {
public:
    ImageBuffer() = default;

    explicit ImageBuffer(size_t size) : data_(size, 0) {}

    ImageBuffer(size_t size, XBYTE fill) : data_(size, fill) {}

    XBYTE* Data() { return data_.data(); }
    const XBYTE* Data() const { return data_.data(); }
    size_t Size() const { return data_.size(); }

    void Resize(size_t size) { data_.resize(size, 0); }
    void Fill(XBYTE value) { std::fill(data_.begin(), data_.end(), value); }
    void Clear() { Fill(0); }

    XBYTE& operator[](size_t idx) { return data_[idx]; }
    const XBYTE& operator[](size_t idx) const { return data_[idx]; }

private:
    std::vector<XBYTE> data_;
};

/**
 * @brief RAII wrapper for palette buffer allocation (256 entries).
 */
class PaletteBuffer {
public:
    PaletteBuffer(int bytesPerEntry = 4) : bytesPerEntry_(bytesPerEntry), data_(256 * bytesPerEntry, 0) {}

    XBYTE* Data() { return data_.data(); }
    const XBYTE* Data() const { return data_.data(); }
    int BytesPerEntry() const { return bytesPerEntry_; }
    size_t Size() const { return data_.size(); }
    
    // Access a palette entry as XDWORD (assumes BGRA format)
    XDWORD GetColor(int index) const {
        const XBYTE* entry = data_.data() + index * bytesPerEntry_;
        if (bytesPerEntry_ >= 4) {
            return (entry[3] << 24) | (entry[2] << 16) | (entry[1] << 8) | entry[0]; // ARGB from BGRA
        } else {
            return (0xFF << 24) | (entry[2] << 16) | (entry[1] << 8) | entry[0]; // RGB with full alpha
        }
    }
    
    // Set a palette entry from ARGB color (stores as BGRA)
    void SetColor(int index, XDWORD argb) {
        XBYTE* entry = data_.data() + index * bytesPerEntry_;
        entry[0] = static_cast<XBYTE>(argb & 0xFF);         // B
        entry[1] = static_cast<XBYTE>((argb >> 8) & 0xFF);  // G
        entry[2] = static_cast<XBYTE>((argb >> 16) & 0xFF); // R
        if (bytesPerEntry_ >= 4) {
            entry[3] = static_cast<XBYTE>((argb >> 24) & 0xFF); // A
        }
    }
    
    // Fill entire palette with black
    void Clear() {
        std::fill(data_.begin(), data_.end(), 0);
    }

private:
    int bytesPerEntry_;
    std::vector<XBYTE> data_;
};

//==============================================================================
// Image Descriptor Factory
//==============================================================================

/**
 * @brief Factory class for creating VxImageDescEx with various pixel formats.
 */
class ImageDescFactory {
public:
    /**
     * @brief Creates a 32-bit ARGB image descriptor.
     */
    static VxImageDescEx Create32BitARGB(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 32;
        desc.BytesPerLine = width * 4;
        desc.RedMask = 0x00FF0000;
        desc.GreenMask = 0x0000FF00;
        desc.BlueMask = 0x000000FF;
        desc.AlphaMask = 0xFF000000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 32-bit RGB (no alpha) image descriptor.
     */
    static VxImageDescEx Create32BitRGB(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 32;
        desc.BytesPerLine = width * 4;
        desc.RedMask = 0x00FF0000;
        desc.GreenMask = 0x0000FF00;
        desc.BlueMask = 0x000000FF;
        desc.AlphaMask = 0x00000000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 24-bit RGB image descriptor.
     */
    static VxImageDescEx Create24BitRGB(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 24;
        desc.BytesPerLine = width * 3;
        desc.RedMask = 0x00FF0000;
        desc.GreenMask = 0x0000FF00;
        desc.BlueMask = 0x000000FF;
        desc.AlphaMask = 0x00000000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 16-bit RGB 565 image descriptor.
     */
    static VxImageDescEx Create16Bit565(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 16;
        desc.BytesPerLine = width * 2;
        desc.RedMask = 0xF800;
        desc.GreenMask = 0x07E0;
        desc.BlueMask = 0x001F;
        desc.AlphaMask = 0x0000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 16-bit RGB 555 image descriptor.
     */
    static VxImageDescEx Create16Bit555(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 16;
        desc.BytesPerLine = width * 2;
        desc.RedMask = 0x7C00;
        desc.GreenMask = 0x03E0;
        desc.BlueMask = 0x001F;
        desc.AlphaMask = 0x0000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 16-bit ARGB 1555 image descriptor.
     */
    static VxImageDescEx Create16Bit1555(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 16;
        desc.BytesPerLine = width * 2;
        desc.RedMask = 0x7C00;
        desc.GreenMask = 0x03E0;
        desc.BlueMask = 0x001F;
        desc.AlphaMask = 0x8000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 16-bit ARGB 4444 image descriptor.
     */
    static VxImageDescEx Create16Bit4444(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 16;
        desc.BytesPerLine = width * 2;
        desc.RedMask = 0x0F00;
        desc.GreenMask = 0x00F0;
        desc.BlueMask = 0x000F;
        desc.AlphaMask = 0xF000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 32-bit ABGR (BGR with alpha in MSB) image descriptor.
     */
    static VxImageDescEx Create32BitABGR(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 32;
        desc.BytesPerLine = width * 4;
        desc.RedMask = 0x000000FF;
        desc.GreenMask = 0x0000FF00;
        desc.BlueMask = 0x00FF0000;
        desc.AlphaMask = 0xFF000000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 32-bit RGBA image descriptor.
     */
    static VxImageDescEx Create32BitRGBA(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 32;
        desc.BytesPerLine = width * 4;
        desc.RedMask = 0xFF000000;
        desc.GreenMask = 0x00FF0000;
        desc.BlueMask = 0x0000FF00;
        desc.AlphaMask = 0x000000FF;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 16-bit BGR 565 image descriptor.
     */
    static VxImageDescEx Create16BitBGR565(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 16;
        desc.BytesPerLine = width * 2;
        desc.RedMask = 0x001F;
        desc.GreenMask = 0x07E0;
        desc.BlueMask = 0xF800;
        desc.AlphaMask = 0x0000;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates an 8-bit paletted image descriptor.
     */
    static VxImageDescEx Create8BitPaletted(int width, int height,
                                            XBYTE* image = nullptr,
                                            XBYTE* colorMap = nullptr,
                                            int bytesPerColorEntry = 4) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 8;
        desc.BytesPerLine = width;
        desc.RedMask = 0;
        desc.GreenMask = 0;
        desc.BlueMask = 0;
        desc.AlphaMask = 0;
        desc.ColorMapEntries = 256;
        desc.BytesPerColorEntry = static_cast<short>(bytesPerColorEntry);
        desc.ColorMap = colorMap;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Creates a 4-bit paletted image descriptor.
     */
    static VxImageDescEx Create4BitPaletted(int width, int height,
                                            XBYTE* image = nullptr,
                                            XBYTE* colorMap = nullptr,
                                            int bytesPerColorEntry = 4) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 4;
        desc.BytesPerLine = (width + 1) / 2;  // 2 pixels per byte
        desc.RedMask = 0;
        desc.GreenMask = 0;
        desc.BlueMask = 0;
        desc.AlphaMask = 0;
        desc.ColorMapEntries = 16;
        desc.BytesPerColorEntry = static_cast<short>(bytesPerColorEntry);
        desc.ColorMap = colorMap;
        desc.Image = image;
        return desc;
    }

    /**
     * @brief Calculates buffer size needed for an image.
     */
    static size_t CalcBufferSize(const VxImageDescEx& desc) {
        return static_cast<size_t>(desc.BytesPerLine) * desc.Height;
    }
};

//==============================================================================
// Test Pattern Generators
//==============================================================================

/**
 * @brief Generates various test patterns for image testing.
 */
class PatternGenerator {
public:
    /**
     * @brief Fills a 32-bit ARGB buffer with a solid color.
     */
    static void FillSolid32(XBYTE* buffer, int width, int height,
                            XBYTE r, XBYTE g, XBYTE b, XBYTE a = 255) {
        XDWORD* pixels = reinterpret_cast<XDWORD*>(buffer);
        XDWORD color = (a << 24) | (r << 16) | (g << 8) | b;
        for (int i = 0; i < width * height; ++i) {
            pixels[i] = color;
        }
    }

    /**
     * @brief Fills a 24-bit RGB buffer with a solid color.
     */
    static void FillSolid24(XBYTE* buffer, int width, int height,
                            XBYTE r, XBYTE g, XBYTE b) {
        for (int i = 0; i < width * height; ++i) {
            buffer[i * 3 + 0] = b;
            buffer[i * 3 + 1] = g;
            buffer[i * 3 + 2] = r;
        }
    }

    /**
     * @brief Creates a gradient pattern in 32-bit ARGB format.
     */
    static void FillGradient32(XBYTE* buffer, int width, int height, bool horizontal = true) {
        XDWORD* pixels = reinterpret_cast<XDWORD*>(buffer);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int val = horizontal ? (x * 255 / (width > 1 ? width - 1 : 1))
                                     : (y * 255 / (height > 1 ? height - 1 : 1));
                pixels[y * width + x] = 0xFF000000 | (val << 16) | (val << 8) | val;
            }
        }
    }

    /**
     * @brief Creates a checkerboard pattern in 32-bit ARGB format.
     */
    static void FillCheckerboard32(XBYTE* buffer, int width, int height,
                                   int squareSize = 8,
                                   XDWORD color1 = 0xFFFFFFFF,
                                   XDWORD color2 = 0xFF000000) {
        XDWORD* pixels = reinterpret_cast<XDWORD*>(buffer);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool isWhite = ((x / squareSize) + (y / squareSize)) % 2 == 0;
                pixels[y * width + x] = isWhite ? color1 : color2;
            }
        }
    }

    /**
     * @brief Creates RGB color bars pattern in 32-bit ARGB format.
     */
    static void FillColorBars32(XBYTE* buffer, int width, int height) {
        XDWORD* pixels = reinterpret_cast<XDWORD*>(buffer);
        const XDWORD colors[] = {
            0xFFFF0000, // Red
            0xFF00FF00, // Green
            0xFF0000FF, // Blue
            0xFFFFFF00, // Yellow
            0xFFFF00FF, // Magenta
            0xFF00FFFF, // Cyan
            0xFFFFFFFF, // White
            0xFF000000  // Black
        };
        int barWidth = width / 8;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int barIdx = std::min(x / (barWidth > 0 ? barWidth : 1), 7);
                pixels[y * width + x] = colors[barIdx];
            }
        }
    }

    /**
     * @brief Creates a unique pattern where each pixel has unique RGB values.
     */
    static void FillUniquePixels32(XBYTE* buffer, int width, int height) {
        XDWORD* pixels = reinterpret_cast<XDWORD*>(buffer);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                // Add offset to ensure first pixel (idx=0) is not pure black
                // This prevents false "empty output" detection when alpha is stripped
                XBYTE r = static_cast<XBYTE>(((idx + 1) * 7) & 0xFF);
                XBYTE g = static_cast<XBYTE>(((idx + 1) * 13) & 0xFF);
                XBYTE b = static_cast<XBYTE>(((idx + 1) * 23) & 0xFF);
                // Ensure at least one RGB channel is non-zero
                if (r == 0 && g == 0 && b == 0) r = 1;
                pixels[idx] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
    }

    /**
     * @brief Creates a standard 256-color palette (web-safe style).
     */
    static void CreateStandardPalette(XBYTE* palette, int bytesPerEntry = 4) {
        int idx = 0;
        // Create a 6x6x6 color cube (216 colors) + grayscale
        for (int r = 0; r < 6; ++r) {
            for (int g = 0; g < 6; ++g) {
                for (int b = 0; b < 6; ++b) {
                    if (idx >= 256) break;
                    XBYTE* entry = palette + idx * bytesPerEntry;
                    entry[0] = static_cast<XBYTE>(b * 51);
                    entry[1] = static_cast<XBYTE>(g * 51);
                    entry[2] = static_cast<XBYTE>(r * 51);
                    if (bytesPerEntry >= 4) entry[3] = 255;
                    ++idx;
                }
            }
        }
        // Fill remaining with grayscale
        for (int i = idx; i < 256; ++i) {
            XBYTE gray = static_cast<XBYTE>((i - idx) * 255 / (256 - idx));
            XBYTE* entry = palette + i * bytesPerEntry;
            entry[0] = gray;
            entry[1] = gray;
            entry[2] = gray;
            if (bytesPerEntry >= 4) entry[3] = 255;
        }
    }
};

//==============================================================================
// Image Comparison Utilities
//==============================================================================

/**
 * @brief Utilities for comparing images with tolerance.
 */
class ImageComparator {
public:
    /**
     * @brief Compares two 32-bit images pixel by pixel.
     * @return Number of differing pixels.
     */
    static int Compare32(const XBYTE* img1, const XBYTE* img2,
                         int width, int height, int tolerance = 0) {
        const XDWORD* p1 = reinterpret_cast<const XDWORD*>(img1);
        const XDWORD* p2 = reinterpret_cast<const XDWORD*>(img2);
        int diffCount = 0;

        for (int i = 0; i < width * height; ++i) {
            if (!PixelsMatch32(p1[i], p2[i], tolerance)) {
                ++diffCount;
            }
        }
        return diffCount;
    }

    /**
     * @brief Checks if two 32-bit pixels match within tolerance.
     */
    static bool PixelsMatch32(XDWORD p1, XDWORD p2, int tolerance) {
        int a1 = (p1 >> 24) & 0xFF, a2 = (p2 >> 24) & 0xFF;
        int r1 = (p1 >> 16) & 0xFF, r2 = (p2 >> 16) & 0xFF;
        int g1 = (p1 >> 8) & 0xFF, g2 = (p2 >> 8) & 0xFF;
        int b1 = p1 & 0xFF, b2 = p2 & 0xFF;

        return std::abs(a1 - a2) <= tolerance &&
               std::abs(r1 - r2) <= tolerance &&
               std::abs(g1 - g2) <= tolerance &&
               std::abs(b1 - b2) <= tolerance;
    }

    /**
     * @brief Calculates Mean Squared Error between two 32-bit images.
     */
    static double CalcMSE32(const XBYTE* img1, const XBYTE* img2,
                            int width, int height, bool includeAlpha = true) {
        const XDWORD* p1 = reinterpret_cast<const XDWORD*>(img1);
        const XDWORD* p2 = reinterpret_cast<const XDWORD*>(img2);
        double sum = 0.0;
        int numChannels = includeAlpha ? 4 : 3;

        for (int i = 0; i < width * height; ++i) {
            int r1 = (p1[i] >> 16) & 0xFF, r2 = (p2[i] >> 16) & 0xFF;
            int g1 = (p1[i] >> 8) & 0xFF, g2 = (p2[i] >> 8) & 0xFF;
            int b1 = p1[i] & 0xFF, b2 = p2[i] & 0xFF;

            sum += (r1 - r2) * (r1 - r2);
            sum += (g1 - g2) * (g1 - g2);
            sum += (b1 - b2) * (b1 - b2);

            if (includeAlpha) {
                int a1 = (p1[i] >> 24) & 0xFF, a2 = (p2[i] >> 24) & 0xFF;
                sum += (a1 - a2) * (a1 - a2);
            }
        }

        return sum / (width * height * numChannels);
    }

    /**
     * @brief Calculates Peak Signal-to-Noise Ratio between two images.
     */
    static double CalcPSNR32(const XBYTE* img1, const XBYTE* img2,
                             int width, int height) {
        double mse = CalcMSE32(img1, img2, width, height, false);
        if (mse < 0.0001) return 100.0; // Essentially identical
        return 10.0 * std::log10(255.0 * 255.0 / mse);
    }

    /**
     * @brief Extracts a specific channel from a 32-bit ARGB pixel.
     */
    static XBYTE ExtractChannel(XDWORD pixel, char channel) {
        switch (channel) {
            case 'a': case 'A': return (pixel >> 24) & 0xFF;
            case 'r': case 'R': return (pixel >> 16) & 0xFF;
            case 'g': case 'G': return (pixel >> 8) & 0xFF;
            case 'b': case 'B': return pixel & 0xFF;
            default: return 0;
        }
    }
};

//==============================================================================
// Test Fixture Base Class
//==============================================================================

/**
 * @brief Wrapper class for blitter operations that uses global functions.
 * 
 * This allows tests to work with both ground-truth DLLs (which may not export
 * VxBlitEngine class) and the built library.
 */
class BlitterWrapper {
public:
    void DoBlit(const VxImageDescEx& src, const VxImageDescEx& dst) {
        VxDoBlit(src, dst);
    }
    
    void DoBlitUpsideDown(const VxImageDescEx& src, const VxImageDescEx& dst) {
        VxDoBlitUpsideDown(src, dst);
    }
    
    void DoAlphaBlit(const VxImageDescEx& dst, XBYTE alphaValue) {
        VxDoAlphaBlit(dst, alphaValue);
    }
    
    void DoAlphaBlit(const VxImageDescEx& dst, XBYTE* alphaValues) {
        VxDoAlphaBlit(dst, alphaValues);
    }
    
    CKERROR ResizeImage(const VxImageDescEx& src, const VxImageDescEx& dst) {
        // VxResizeImage32 doesn't return error code - assume success
        VxResizeImage32(src, dst);
        return CK_OK;
    }
    
    CKERROR QuantizeImage(const VxImageDescEx& src, VxImageDescEx& dst) {
        // Call TheBlitter.QuantizeImage directly
        return TheBlitter.QuantizeImage(src, dst) ? CK_OK : CKERR_INVALIDPARAMETER;
    }

    void FillImage(VxImageDescEx& desc, XDWORD color) {
        TheBlitter.FillImage(desc, color);
    }

    void PremultiplyAlpha(VxImageDescEx& desc) {
        TheBlitter.PremultiplyAlpha(desc);
    }

    void UnpremultiplyAlpha(VxImageDescEx& desc) {
        TheBlitter.UnpremultiplyAlpha(desc);
    }

    void SwapRedBlue(VxImageDescEx& desc) {
        TheBlitter.SwapRedBlue(desc);
    }

    void ClearAlpha(VxImageDescEx& desc) {
        TheBlitter.ClearAlpha(desc);
    }

    void SetFullAlpha(VxImageDescEx& desc) {
        TheBlitter.SetFullAlpha(desc);
    }

    void InvertColors(VxImageDescEx& desc) {
        TheBlitter.InvertColors(desc);
    }

    void ConvertToGrayscale(VxImageDescEx& desc) {
        TheBlitter.ConvertToGrayscale(desc);
    }

    void MultiplyBlend(const VxImageDescEx& src, VxImageDescEx& dst) {
        TheBlitter.MultiplyBlend(src, dst);
    }
    
    static VX_PIXELFORMAT GetPixelFormat(const VxImageDescEx& desc) {
        return VxImageDesc2PixelFormat(desc);
    }
    
    static void ConvertPixelFormat(VX_PIXELFORMAT fmt, XDWORD& aMask, 
                                    XDWORD& rMask, XDWORD& gMask, XDWORD& bMask) {
        VxImageDescEx desc;
        VxPixelFormat2ImageDesc(fmt, desc);
        aMask = desc.AlphaMask;
        rMask = desc.RedMask;
        gMask = desc.GreenMask;
        bMask = desc.BlueMask;
    }
    
    static XString PixelFormat2String(VX_PIXELFORMAT fmt) {
        // Simple format name lookup
        static const char* names[] = {
            "UNKNOWN_PF", "_32_ARGB8888", "_32_RGB888", "_24_RGB888",
            "_16_RGB565", "_16_RGB555", "_16_ARGB1555", "_16_ARGB4444",
            "_8_RGB332", "_8_ARGB2222", "_32_ABGR8888", "_32_RGBA8888",
            "_32_BGRA8888", "_24_BGR888"
        };
        if (fmt >= 0 && fmt < sizeof(names)/sizeof(names[0])) {
            return XString(names[fmt]);
        }
        return XString("UNKNOWN");
    }
};

/**
 * @brief Base test fixture for VxBlitEngine tests.
 */
class BlitEngineTestBase : public ::testing::Test {
protected:
    BlitterWrapper blitter;

    void SetUp() override {
        // Nothing special needed
    }

    void TearDown() override {
        // Nothing to clean up
    }

    /**
     * @brief Helper to allocate and setup a source/destination pair.
     */
    struct ImagePair {
        ImageBuffer srcBuffer;
        ImageBuffer dstBuffer;
        VxImageDescEx srcDesc;
        VxImageDescEx dstDesc;
    };

    /**
     * @brief Creates a pair of images for testing format conversion.
     */
    ImagePair CreateImagePair(
        std::function<VxImageDescEx(int, int, XBYTE*)> srcFactory,
        std::function<VxImageDescEx(int, int, XBYTE*)> dstFactory,
        int width, int height) {

        ImagePair pair;
        
        // Create descriptors to determine buffer sizes
        VxImageDescEx srcTemp = srcFactory(width, height, nullptr);
        VxImageDescEx dstTemp = dstFactory(width, height, nullptr);
        
        // Allocate buffers
        pair.srcBuffer.Resize(ImageDescFactory::CalcBufferSize(srcTemp));
        pair.dstBuffer.Resize(ImageDescFactory::CalcBufferSize(dstTemp));
        
        // Create final descriptors with buffer pointers
        pair.srcDesc = srcFactory(width, height, pair.srcBuffer.Data());
        pair.dstDesc = dstFactory(width, height, pair.dstBuffer.Data());
        
        return pair;
    }
};

} // namespace BlitEngineTest

#endif // BLITENGINE_TEST_HELPERS_H
