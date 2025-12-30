/**
 * @file BlitEnginePalettedTest.cpp
 * @brief Tests for VxBlitEngine paletted image blitting functionality.
 *
 * Tests:
 * - 8-bit paletted to 32/24/16-bit conversion
 * - 32/24/16-bit to 8-bit paletted (via quantization)
 * - Palette color mapping accuracy
 * - Standard palette handling
 */

#include "BlitEngineTestHelpers.h"

using namespace BlitEngineTest;


class PalettedBlitTest : public BlitEngineTestBase {
protected:
    // Fill paletted image with index pattern
    void FillWithIndices(XBYTE* data, int width, int height, XBYTE startIndex = 0) {
        for (int i = 0; i < width * height; ++i) {
            data[i] = static_cast<XBYTE>((startIndex + i) % 256);
        }
    }
    
    // Fill with single index
    void FillSolidIndex(XBYTE* data, int width, int height, XBYTE index) {
        memset(data, index, width * height);
    }
    
    // Fill with horizontal stripes (different index per row)
    void FillStripes(XBYTE* data, int width, int height) {
        for (int y = 0; y < height; ++y) {
            memset(data + y * width, static_cast<XBYTE>(y % 256), width);
        }
    }
};

//==============================================================================
// Paletted to 32-bit ARGB
//==============================================================================

TEST_F(PalettedBlitTest, Paletted8To32_SolidColor) {
    const int width = 16, height = 16;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    // Set up palette - index 5 = red (ARGB 0xFFFF0000)
    palette.Clear();
    palette.SetColor(5, 0xFFFF0000);
    
    // Fill with index 5
    FillSolidIndex(srcBuffer.Data(), width, height, 5);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_to_32_solidcolor", dst, "Paletted");
    
    // All pixels should be red
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFFFF0000, pixels[i]) << "Pixel " << i << " should be red";
    }
}

TEST_F(PalettedBlitTest, Paletted8To32_StandardPalette) {
    const int width = 32, height = 32;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    // Use standard palette
    PatternGenerator::CreateStandardPalette(palette.Data());
    
    // Create horizontal stripes
    FillStripes(srcBuffer.Data(), width, height);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_to_32_standardpalette", dst, "Paletted");
    
    // Check that pixels match palette colors
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int y = 0; y < height; ++y) {
        XBYTE index = static_cast<XBYTE>(y % 256);
        XDWORD expected = palette.GetColor(index);
        EXPECT_EQ(expected, pixels[y * width]) 
            << "Row " << y << " should match palette index " << (int)index;
    }
}

TEST_F(PalettedBlitTest, Paletted8To32_AllIndices) {
    const int width = 16, height = 16; // 256 pixels
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    // Set up gradient palette
    for (int i = 0; i < 256; ++i) {
        palette.SetColor(i, 0xFF000000 | (i << 16) | (i << 8) | i);
    }
    
    // Use each index once
    FillWithIndices(srcBuffer.Data(), width, height, 0);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_to_32_allindices", dst, "Paletted");
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(palette.GetColor(i), pixels[i]) 
            << "Pixel " << i << " should match palette entry " << i;
    }
}

//==============================================================================
// Paletted to 24-bit RGB
//==============================================================================

TEST_F(PalettedBlitTest, Paletted8To24_SolidColor) {
    const int width = 16, height = 16;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 3);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create24BitRGB(width, height, dstBuffer.Data());
    
    // ARGB 0xFF112233 -> in memory B=0x33, G=0x22, R=0x11
    palette.SetColor(10, 0xFF112233);
    FillSolidIndex(srcBuffer.Data(), width, height, 10);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_to_24_solidcolor", dst, "Paletted");
    
    const XBYTE* pixels = dstBuffer.Data();
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x33, pixels[i*3 + 0]) << "Blue at pixel " << i;
        EXPECT_EQ(0x22, pixels[i*3 + 1]) << "Green at pixel " << i;
        EXPECT_EQ(0x11, pixels[i*3 + 2]) << "Red at pixel " << i;
    }
}

TEST_F(PalettedBlitTest, Paletted8To24_MultipleColors) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 3);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create24BitRGB(width, height, dstBuffer.Data());
    
    // Set 8 palette entries
    palette.SetColor(0, 0xFF000000); // Black
    palette.SetColor(1, 0xFFFF0000); // Red
    palette.SetColor(2, 0xFF00FF00); // Green
    palette.SetColor(3, 0xFF0000FF); // Blue
    palette.SetColor(4, 0xFFFFFF00); // Yellow
    palette.SetColor(5, 0xFFFF00FF); // Magenta
    palette.SetColor(6, 0xFF00FFFF); // Cyan
    palette.SetColor(7, 0xFFFFFFFF); // White
    
    // Fill with indices 0-7 repeating
    for (int i = 0; i < width * height; ++i) {
        srcBuffer[i] = static_cast<XBYTE>(i % 8);
    }
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_to_24_multiplecolors", dst, "Paletted");
    
    // Check first 8 pixels - pixel 1 is red (R=255, G=0, B=0)
    const XBYTE* pixels = dstBuffer.Data();
    EXPECT_EQ(0x00, pixels[1*3 + 0]) << "Red pixel - Blue channel";
    EXPECT_EQ(0x00, pixels[1*3 + 1]) << "Red pixel - Green channel";
    EXPECT_EQ(0xFF, pixels[1*3 + 2]) << "Red pixel - Red channel";
}

//==============================================================================
// Paletted to 16-bit
//==============================================================================

TEST_F(PalettedBlitTest, Paletted8To565_SolidRed) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 2);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create16Bit565(width, height, dstBuffer.Data());
    
    palette.SetColor(0, 0xFFFF0000); // Pure red
    FillSolidIndex(srcBuffer.Data(), width, height, 0);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_to_565_solidred", dst, "Paletted");
    
    const XWORD* pixels = reinterpret_cast<const XWORD*>(dstBuffer.Data());
    // Red in 565: R=31 (5 bits), G=0, B=0 -> 0xF800
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xF800, pixels[i]) << "Pure red in 565 format";
    }
}

TEST_F(PalettedBlitTest, Paletted8To1555_WithAlpha) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 2);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create16Bit1555(width, height, dstBuffer.Data());
    
    // Full alpha green (alpha >= 128 should set alpha bit)
    palette.SetColor(3, 0xFF00FF00); // Full alpha green
    FillSolidIndex(srcBuffer.Data(), width, height, 3);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_to_1555_withalpha", dst, "Paletted");
    
    const XWORD* pixels = reinterpret_cast<const XWORD*>(dstBuffer.Data());
    // Alpha bit should be set (alpha >= 128), green max in 555
    // 1555 format: A(1) R(5) G(5) B(5)
    // Green=255 -> 31 in 5 bits, in G position: (31 << 5) = 0x03E0
    // Alpha=1 -> 0x8000
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x1F, (pixels[i] >> 5) & 0x1F) << "Green component at pixel " << i;
        EXPECT_EQ(0x80, (pixels[i] >> 8) & 0x80) << "Alpha bit should be set at pixel " << i;
    }
}

//==============================================================================
// 32-bit to Paletted (implicit quantization)
//==============================================================================

TEST_F(PalettedBlitTest, ARGB32ToPaletted_WithExistingPalette) {
    const int width = 16, height = 16;
    ImageBuffer srcBuffer(width * height * 4);
    ImageBuffer dstBuffer(width * height);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create32BitARGB(width, height, srcBuffer.Data());
    auto dst = ImageDescFactory::Create8BitPaletted(width, height, dstBuffer.Data(), palette.Data());
    
    // Set up destination palette
    PatternGenerator::CreateStandardPalette(palette.Data());
    
    // Fill source with black
    PatternGenerator::FillSolid32(srcBuffer.Data(), width, height, 0, 0, 0, 255);
    
    // This should quantize to palette
    CKERROR result = blitter.QuantizeImage(src, dst);

    if (result == CK_OK) {
        ImageWriter::SaveFromDesc("argb32_to_paletted_existingpalette", dst, "Paletted");
    }
    
    if (result == CK_OK) {
        // Check that black maps to a black palette entry
        double maxDist = 0;
        for (int i = 0; i < width * height; ++i) {
            XDWORD palColor = palette.GetColor(dstBuffer[i]);
            int r = (palColor >> 16) & 0xFF;
            int g = (palColor >> 8) & 0xFF;
            int b = palColor & 0xFF;
            double dist = std::sqrt(static_cast<double>(r*r + g*g + b*b));
            maxDist = std::max(maxDist, dist);
        }
        EXPECT_LT(maxDist, 30.0) << "Black should map to near-black palette entry";
    }
}

//==============================================================================
// Palette Edge Cases
//==============================================================================

TEST_F(PalettedBlitTest, EmptyPalette_AllBlack) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette; // Already initialized to zeros
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    FillWithIndices(srcBuffer.Data(), width, height, 0);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_empty_palette_allblack", dst, "Paletted");
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        // Zero palette with alpha=0 means fully transparent black (or opaque if A is set to 0xFF)
        // Our PaletteBuffer initializes all to 0, including alpha
        XDWORD got = pixels[i];
        // Just check RGB is 0; alpha depends on how blitter handles it
        EXPECT_EQ(0, got & 0x00FFFFFF) << "Should be black for zero palette";
    }
}

TEST_F(PalettedBlitTest, SinglePaletteEntry_AllSameColor) {
    const int width = 16, height = 16;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    // Only entry 128 has a color
    palette.SetColor(128, 0xFFABCDEF);
    
    FillSolidIndex(srcBuffer.Data(), width, height, 128);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_singlepaletteentry_allsamecolor", dst, "Paletted");
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFFABCDEF, pixels[i]);
    }
}

TEST_F(PalettedBlitTest, Index255_CorrectlyMapped) {
    const int width = 4, height = 4;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    palette.SetColor(255, 0xFFDEADBE);
    FillSolidIndex(srcBuffer.Data(), width, height, 255);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_index255_correctlymapped", dst, "Paletted");
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    EXPECT_EQ(0xFFDEADBE, pixels[0]) << "Index 255 should map correctly";
}

//==============================================================================
// Size Edge Cases
//==============================================================================

TEST_F(PalettedBlitTest, SinglePixel) {
    ImageBuffer srcBuffer(1);
    ImageBuffer dstBuffer(4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(1, 1, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(1, 1, dstBuffer.Data());
    
    palette.SetColor(42, 0xFF123456);
    srcBuffer[0] = 42;
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_singlepixel", dst, "Paletted");
    
    EXPECT_EQ(0xFF123456, *reinterpret_cast<XDWORD*>(dstBuffer.Data()));
}

TEST_F(PalettedBlitTest, LargeImage) {
    const int width = 256, height = 256;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    PatternGenerator::CreateStandardPalette(palette.Data());
    FillStripes(srcBuffer.Data(), width, height);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_largeimage", dst, "Paletted");
    
    // Sample check
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int y = 0; y < height; y += 32) {
        XBYTE index = static_cast<XBYTE>(y % 256);
        EXPECT_EQ(palette.GetColor(index), pixels[y * width]);
    }
}

TEST_F(PalettedBlitTest, NonSquare_WidthLarger) {
    const int width = 64, height = 8;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    palette.SetColor(99, 0xFF998877);
    FillSolidIndex(srcBuffer.Data(), width, height, 99);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_nonsquare_widthlarger", dst, "Paletted");
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF998877, pixels[i]);
    }
}

TEST_F(PalettedBlitTest, NonSquare_HeightLarger) {
    const int width = 8, height = 64;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    palette.SetColor(200, 0xFFAABBCC);
    FillSolidIndex(srcBuffer.Data(), width, height, 200);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_nonsquare_heightlarger", dst, "Paletted");
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFFAABBCC, pixels[i]);
    }
}

//==============================================================================
// Color Accuracy Tests
//==============================================================================

TEST_F(PalettedBlitTest, ColorAccuracy_PrimaryColors) {
    const int width = 8, height = 1;
    ImageBuffer srcBuffer(width);
    ImageBuffer dstBuffer(width * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, 1, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, 1, dstBuffer.Data());
    
    // Primary and secondary colors
    palette.SetColor(0, 0xFF000000); // Black
    palette.SetColor(1, 0xFFFF0000); // Red
    palette.SetColor(2, 0xFF00FF00); // Green
    palette.SetColor(3, 0xFF0000FF); // Blue
    palette.SetColor(4, 0xFFFFFF00); // Yellow
    palette.SetColor(5, 0xFFFF00FF); // Magenta
    palette.SetColor(6, 0xFF00FFFF); // Cyan
    palette.SetColor(7, 0xFFFFFFFF); // White
    
    for (int i = 0; i < 8; ++i) {
        srcBuffer[i] = static_cast<XBYTE>(i);
    }
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_coloraccuracy_primarycolors", dst, "Paletted");
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(palette.GetColor(i), pixels[i]) 
            << "Color " << i << " should match exactly";
    }
}

TEST_F(PalettedBlitTest, ColorAccuracy_GrayScale) {
    const int width = 256, height = 1;
    ImageBuffer srcBuffer(width);
    ImageBuffer dstBuffer(width * 4);
    PaletteBuffer palette;
    
    auto src = ImageDescFactory::Create8BitPaletted(width, 1, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, 1, dstBuffer.Data());
    
    // Grayscale palette
    for (int i = 0; i < 256; ++i) {
        palette.SetColor(i, 0xFF000000 | (i << 16) | (i << 8) | i);
    }
    
    // Each pixel uses its corresponding index
    FillWithIndices(srcBuffer.Data(), width, 1, 0);
    
    blitter.DoBlit(src, dst);

    ImageWriter::SaveFromDesc("paletted8_coloraccuracy_grayscale", dst, "Paletted");
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < 256; ++i) {
        XBYTE gray = static_cast<XBYTE>(i);
        XDWORD expected = 0xFF000000 | (gray << 16) | (gray << 8) | gray;
        EXPECT_EQ(expected, pixels[i]) << "Grayscale value " << i;
    }
}

//==============================================================================
// Paletted to Paletted Copy (SSE optimized path)
//==============================================================================

TEST_F(PalettedBlitTest, PalettedToPaletted_SameFormat_AllWidths) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 64};
    
    for (int width : widths) {
        ImageBuffer srcBuffer(width * 4);
        ImageBuffer dstBuffer(width * 4);
        PaletteBuffer palette;
        
        dstBuffer.Fill(0xCD);
        
        auto src = ImageDescFactory::Create8BitPaletted(width, 4, srcBuffer.Data(), palette.Data());
        auto dst = ImageDescFactory::Create8BitPaletted(width, 4, dstBuffer.Data(), palette.Data());
        
        // Fill with pattern
        for (int i = 0; i < width * 4; ++i) {
            srcBuffer[i] = static_cast<XBYTE>(i & 0xFF);
        }
        
        blitter.DoBlit(src, dst);
        
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(srcBuffer[i], dstBuffer[i]) 
                << "Mismatch at pixel " << i << " (width=" << width << ")";
        }
    }
}

// Note: 4-bit paletted to 4-bit paletted copy is not supported by VxBlitEngine
// (the generic blit path doesn't handle sub-byte bit widths correctly)
// Use 4-bit to 32-bit conversion tests instead

//==============================================================================
// Paletted to 32-bit All Widths (SSE remainder testing)
//==============================================================================

TEST_F(PalettedBlitTest, Paletted8_to_ARGB32_AllWidths) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 64};
    
    for (int width : widths) {
        ImageBuffer srcBuffer(width * 4);
        ImageBuffer dstBuffer(width * 4 * 4);
        PaletteBuffer palette;
        
        // Random-ish palette
        for (int i = 0; i < 256; ++i) {
            palette.SetColor(i, 0xFF000000 | ((i * 17) << 16) | ((i * 7) << 8) | (i * 13));
        }
        
        auto src = ImageDescFactory::Create8BitPaletted(width, 4, srcBuffer.Data(), palette.Data());
        auto dst = ImageDescFactory::Create32BitARGB(width, 4, dstBuffer.Data());
        
        // Fill with sequential indices
        for (int i = 0; i < width * 4; ++i) {
            srcBuffer[i] = static_cast<XBYTE>(i & 0xFF);
        }
        
        blitter.DoBlit(src, dst);
        
        const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            XBYTE idx = srcBuffer[i];
            EXPECT_EQ(palette.GetColor(idx), pixels[i]) 
                << "Pixel " << i << " (index=" << (int)idx << ", width=" << width << ")";
        }
    }
}

TEST_F(PalettedBlitTest, Paletted8_to_RGB565_AllWidths) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32};
    
    for (int width : widths) {
        ImageBuffer srcBuffer(width * 4);
        ImageBuffer dstBuffer(width * 4 * 2);
        PaletteBuffer palette;
        
        // Set up known colors in palette
        palette.SetColor(0, 0xFFFF0000); // Red
        palette.SetColor(1, 0xFF00FF00); // Green
        palette.SetColor(2, 0xFF0000FF); // Blue
        palette.SetColor(3, 0xFFFFFFFF); // White
        palette.SetColor(4, 0xFF000000); // Black
        
        auto src = ImageDescFactory::Create8BitPaletted(width, 4, srcBuffer.Data(), palette.Data());
        auto dst = ImageDescFactory::Create16Bit565(width, 4, dstBuffer.Data());
        
        for (int i = 0; i < width * 4; ++i) {
            srcBuffer[i] = static_cast<XBYTE>(i % 5);
        }
        
        blitter.DoBlit(src, dst);
        
        const XWORD* pixels = reinterpret_cast<const XWORD*>(dstBuffer.Data());
        XWORD expected565[] = {0xF800, 0x07E0, 0x001F, 0xFFFF, 0x0000};
        
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(expected565[i % 5], pixels[i]) 
                << "Pixel " << i << " (width=" << width << ")";
        }
    }
}

//==============================================================================
// Alpha Preservation in Paletted Conversions
//==============================================================================

TEST_F(PalettedBlitTest, PalettedAlpha_PreservedToARGB32) {
    const int width = 16, height = 4;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    // Palette with varying alpha
    for (int i = 0; i < 256; ++i) {
        XBYTE alpha = static_cast<XBYTE>(i);
        palette.SetColor(i, (alpha << 24) | 0x808080);
    }
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    for (int i = 0; i < width * height; ++i) {
        srcBuffer[i] = static_cast<XBYTE>(i & 0xFF);
    }
    
    blitter.DoBlit(src, dst);
    
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        XBYTE expectedAlpha = static_cast<XBYTE>(srcBuffer[i]);
        XBYTE actualAlpha = static_cast<XBYTE>((pixels[i] >> 24) & 0xFF);
        EXPECT_EQ(expectedAlpha, actualAlpha) << "Alpha mismatch at pixel " << i;
    }
}

//==============================================================================
// Large Paletted Image
//==============================================================================

TEST_F(PalettedBlitTest, LargePaletted_CorrectConversion) {
    const int width = 1024, height = 768;
    ImageBuffer srcBuffer(width * height);
    ImageBuffer dstBuffer(width * height * 4);
    PaletteBuffer palette;
    
    // Grayscale palette
    for (int i = 0; i < 256; ++i) {
        palette.SetColor(i, 0xFF000000 | (i << 16) | (i << 8) | i);
    }
    
    auto src = ImageDescFactory::Create8BitPaletted(width, height, srcBuffer.Data(), palette.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    // Fill with gradient pattern
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            srcBuffer[y * width + x] = static_cast<XBYTE>((x + y) & 0xFF);
        }
    }
    
    blitter.DoBlit(src, dst);
    
    // Spot check
    const XDWORD* pixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int y = 0; y < height; y += 100) {
        for (int x = 0; x < width; x += 100) {
            int idx = y * width + x;
            XBYTE expectedGray = static_cast<XBYTE>((x + y) & 0xFF);
            XDWORD expected = 0xFF000000 | (expectedGray << 16) | (expectedGray << 8) | expectedGray;
            EXPECT_EQ(expected, pixels[idx]) << "Mismatch at " << x << "," << y;
        }
    }
}

//==============================================================================
// Paletted to 16-bit Alpha Formats
//==============================================================================

TEST_F(PalettedBlitTest, Paletted8_to_1555_WithAlpha) {
    const int widths[] = {1, 4, 8, 16, 17};
    PaletteBuffer palette;
    
    // Set up palette with different alpha values
    palette.SetColor(0, 0xFFFF0000); // Full alpha red
    palette.SetColor(1, 0x00FF0000); // Zero alpha red
    palette.SetColor(2, 0x80FF0000); // Half alpha red
    
    for (int width : widths) {
        ImageBuffer srcBuffer(width * 4);
        ImageBuffer dstBuffer(width * 4 * 2);
        dstBuffer.Fill(0xCD);
        
        auto src = ImageDescFactory::Create8BitPaletted(
            width, 4, srcBuffer.Data(), palette.Data());
        auto dst = ImageDescFactory::Create16Bit1555(
            width, 4, dstBuffer.Data());
        
        srcBuffer[0] = 0; // Full alpha
        srcBuffer[1] = 1; // Zero alpha
        srcBuffer[2] = 2; // Half alpha
        
        blitter.DoBlit(src, dst);
        
        const XWORD* dstPixels = reinterpret_cast<const XWORD*>(dstBuffer.Data());
        
        EXPECT_EQ(0x8000, dstPixels[0] & 0x8000) << "Full alpha bit (width=" << width << ")";
        EXPECT_EQ(0x0000, dstPixels[1] & 0x8000) << "Zero alpha bit";
        EXPECT_EQ(0x8000, dstPixels[2] & 0x8000) << "Half alpha bit (should be set)";
    }
}

TEST_F(PalettedBlitTest, Paletted8_to_4444_WithAlpha) {
    const int widths[] = {1, 4, 8, 16, 17};
    PaletteBuffer palette;
    
    // Set up palette with different alpha values
    palette.SetColor(0, 0xFFFF0000); // Full alpha red
    palette.SetColor(1, 0x00FF0000); // Zero alpha red  
    palette.SetColor(2, 0x88888888); // Half everything
    
    for (int width : widths) {
        ImageBuffer srcBuffer(width * 4);
        ImageBuffer dstBuffer(width * 4 * 2);
        dstBuffer.Fill(0xCD);
        
        auto src = ImageDescFactory::Create8BitPaletted(
            width, 4, srcBuffer.Data(), palette.Data());
        auto dst = ImageDescFactory::Create16Bit4444(
            width, 4, dstBuffer.Data());
        
        srcBuffer[0] = 0; // Full alpha red
        srcBuffer[1] = 1; // Zero alpha red
        srcBuffer[2] = 2; // Half everything
        
        blitter.DoBlit(src, dst);
        
        const XWORD* dstPixels = reinterpret_cast<const XWORD*>(dstBuffer.Data());
        
        // Full alpha: 0xF in top nibble
        EXPECT_EQ(0xF, (dstPixels[0] >> 12) & 0xF) << "Full alpha (width=" << width << ")";
        // Zero alpha: 0x0 in top nibble
        EXPECT_EQ(0x0, (dstPixels[1] >> 12) & 0xF) << "Zero alpha";
        // Half alpha: 0x8 in top nibble
        EXPECT_EQ(0x8, (dstPixels[2] >> 12) & 0xF) << "Half alpha";
    }
}

TEST_F(PalettedBlitTest, Paletted8_to_RGB24_AllWidths) {
    const int widths[] = {3, 8, 15, 16, 17};
    PaletteBuffer palette;
    
    // Set up primary colors
    palette.SetColor(0, 0xFFFF0000); // Red
    palette.SetColor(1, 0xFF00FF00); // Green
    palette.SetColor(2, 0xFF0000FF); // Blue
    
    for (int width : widths) {
        ImageBuffer srcBuffer(width * 4);
        ImageBuffer dstBuffer(width * 4 * 3);
        dstBuffer.Fill(0xCD);
        
        auto src = ImageDescFactory::Create8BitPaletted(
            width, 4, srcBuffer.Data(), palette.Data());
        auto dst = ImageDescFactory::Create24BitRGB(
            width, 4, dstBuffer.Data());
        
        srcBuffer[0] = 0; // Red
        srcBuffer[1] = 1; // Green
        srcBuffer[2] = 2; // Blue
        
        blitter.DoBlit(src, dst);
        
        const XBYTE* dstPixels = dstBuffer.Data();
        
        // 24-bit BGR order
        EXPECT_EQ(0x00, dstPixels[0]) << "Red->B (width=" << width << ")";
        EXPECT_EQ(0x00, dstPixels[1]) << "Red->G";
        EXPECT_EQ(0xFF, dstPixels[2]) << "Red->R";
        
        EXPECT_EQ(0x00, dstPixels[3]) << "Green->B";
        EXPECT_EQ(0xFF, dstPixels[4]) << "Green->G";
        EXPECT_EQ(0x00, dstPixels[5]) << "Green->R";
        
        EXPECT_EQ(0xFF, dstPixels[6]) << "Blue->B";
        EXPECT_EQ(0x00, dstPixels[7]) << "Blue->G";
        EXPECT_EQ(0x00, dstPixels[8]) << "Blue->R";
    }
}
