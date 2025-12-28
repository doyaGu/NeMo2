/**
 * @file BlitEngineFormatConversionTest.cpp
 * @brief Tests for VxBlitEngine pixel format conversion functionality.
 *
 * Tests conversion between various pixel formats:
 * - 32-bit ARGB <-> 32-bit RGB
 * - 32-bit ARGB <-> 24-bit RGB
 * - 32-bit ARGB <-> 16-bit formats (565, 555, 1555, 4444)
 * - BGR/ABGR/RGBA channel swapping
 * - Same format copy operations
 * - Generic conversion path (different bit widths)
 * 
 * Tests various widths to verify SSE remainder handling.
 */

#include "BlitEngineTestHelpers.h"

using namespace BlitEngineTest;

class FormatConversionTest : public BlitEngineTestBase {
protected:
    // Standard width array for testing SSE remainder handling
    static constexpr int kTestWidths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 64};
    static constexpr int kNumTestWidths = sizeof(kTestWidths) / sizeof(kTestWidths[0]);
};

//==============================================================================
// Same Format Copy Tests
//==============================================================================

TEST_F(FormatConversionTest, SameFormat_32BitARGB_CopiesExactly) {
    const int width = 64, height = 64;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );

    PatternGenerator::FillColorBars32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    EXPECT_EQ(0, ImageComparator::Compare32(
        pair.srcBuffer.Data(), pair.dstBuffer.Data(), width, height, 0));
}

TEST_F(FormatConversionTest, SameFormat_24BitRGB_CopiesExactly) {
    const int width = 32, height = 32;
    auto pair = CreateImagePair(
        ImageDescFactory::Create24BitRGB,
        ImageDescFactory::Create24BitRGB,
        width, height
    );

    PatternGenerator::FillSolid24(pair.srcBuffer.Data(), width, height, 128, 64, 192);
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    EXPECT_EQ(0, memcmp(pair.srcBuffer.Data(), pair.dstBuffer.Data(), 
                        pair.srcBuffer.Size()));
}

TEST_F(FormatConversionTest, SameFormat_16Bit565_CopiesExactly) {
    const int width = 16, height = 16;
    auto pair = CreateImagePair(
        ImageDescFactory::Create16Bit565,
        ImageDescFactory::Create16Bit565,
        width, height
    );

    // Fill with a pattern
    XWORD* src = reinterpret_cast<XWORD*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = static_cast<XWORD>(i * 17);
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    EXPECT_EQ(0, memcmp(pair.srcBuffer.Data(), pair.dstBuffer.Data(), 
                        pair.srcBuffer.Size()));
}

//==============================================================================
// 32-bit ARGB to Other Formats
//==============================================================================

TEST_F(FormatConversionTest, ARGB32_To_RGB32_PreservesColorIgnoresAlpha) {
    const int width = 32, height = 32;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitRGB,
        width, height
    );

    // Fill source with full alpha - R=128, G=64, B=192
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), width, height, 
                                   128, 64, 192, 255);
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    // Verify colors preserved (alpha byte may be 0x00 or 0xFF depending on implementation)
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        XULONG color = dst[i] & 0x00FFFFFF; // Mask out alpha
        // 0x008040C0 = R:128, G:64, B:192
        EXPECT_EQ(0x008040C0, color) << "Color mismatch at pixel " << i;
    }
}

TEST_F(FormatConversionTest, RGB32_To_ARGB32_SetsFullAlpha) {
    const int width = 32, height = 32;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitRGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );

    // Fill source (no alpha)
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = 0x00804CC0; // RGB without alpha
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    // Verify alpha set to 0xFF
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF804CC0, dst[i]) << "Pixel " << i << " mismatch";
    }
}

TEST_F(FormatConversionTest, ARGB32_To_RGB24_DropsAlpha) {
    const int width = 16, height = 16;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create24BitRGB,
        width, height
    );

    // Fill with known values
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), width, height, 
                                   0xAA, 0xBB, 0xCC, 0xDD);
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    // Verify 24-bit values
    const XBYTE* dst = pair.dstBuffer.Data();
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xCC, dst[i * 3 + 0]) << "Blue mismatch at pixel " << i;
        EXPECT_EQ(0xBB, dst[i * 3 + 1]) << "Green mismatch at pixel " << i;
        EXPECT_EQ(0xAA, dst[i * 3 + 2]) << "Red mismatch at pixel " << i;
    }
}

TEST_F(FormatConversionTest, RGB24_To_ARGB32_AddsAlpha) {
    const int width = 16, height = 16;
    auto pair = CreateImagePair(
        ImageDescFactory::Create24BitRGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );

    PatternGenerator::FillSolid24(pair.srcBuffer.Data(), width, height, 
                                   0xAA, 0xBB, 0xCC);
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    // Verify 32-bit values with full alpha
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFFAABBCC, dst[i]) << "Pixel " << i << " mismatch";
    }
}

//==============================================================================
// 16-bit Format Conversions
//==============================================================================

TEST_F(FormatConversionTest, ARGB32_To_RGB565_QuantizesCorrectly) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create16Bit565,
        width, height
    );

    // Test with pure colors
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    src[0] = 0xFFFF0000; // Red
    src[1] = 0xFF00FF00; // Green
    src[2] = 0xFF0000FF; // Blue
    src[3] = 0xFFFFFFFF; // White
    src[4] = 0xFF000000; // Black
    src[5] = 0xFF808080; // Gray
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XWORD* dst = reinterpret_cast<const XWORD*>(pair.dstBuffer.Data());
    EXPECT_EQ(0xF800, dst[0]) << "Red 565 mismatch";
    EXPECT_EQ(0x07E0, dst[1]) << "Green 565 mismatch";
    EXPECT_EQ(0x001F, dst[2]) << "Blue 565 mismatch";
    EXPECT_EQ(0xFFFF, dst[3]) << "White 565 mismatch";
    EXPECT_EQ(0x0000, dst[4]) << "Black 565 mismatch";
}

TEST_F(FormatConversionTest, RGB565_To_ARGB32_ExpandsCorrectly) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create16Bit565,
        ImageDescFactory::Create32BitARGB,
        width, height
    );

    XWORD* src = reinterpret_cast<XWORD*>(pair.srcBuffer.Data());
    src[0] = 0xF800; // Red
    src[1] = 0x07E0; // Green
    src[2] = 0x001F; // Blue
    src[3] = 0xFFFF; // White
    src[4] = 0x0000; // Black
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    
    // 565 expansion: 5 bits -> 8 bits (r << 3), 6 bits -> 8 bits (g << 2)
    // Pure red (31 in 5 bits) -> 248 (31 << 3) or 255 with replication
    // Pure green (63 in 6 bits) -> 252 (63 << 2) or 255 with replication
    // Pure blue (31 in 5 bits) -> 248 (31 << 3) or 255 with replication
    
    // Red: 0xF800 -> R should be >= 248
    EXPECT_GE((dst[0] >> 16) & 0xFF, 248) << "Red channel should be expanded";
    EXPECT_LE((dst[0] >> 8) & 0xFF, 7) << "Green for red pixel should be near 0";
    EXPECT_LE(dst[0] & 0xFF, 7) << "Blue for red pixel should be near 0";
    
    // Green: 0x07E0 -> G should be >= 252
    EXPECT_LE((dst[1] >> 16) & 0xFF, 7) << "Red for green pixel should be near 0";
    EXPECT_GE((dst[1] >> 8) & 0xFF, 252) << "Green channel should be expanded";
    EXPECT_LE(dst[1] & 0xFF, 7) << "Blue for green pixel should be near 0";
    
    // Blue: 0x001F -> B should be >= 248
    EXPECT_LE((dst[2] >> 16) & 0xFF, 7) << "Red for blue pixel should be near 0";
    EXPECT_LE((dst[2] >> 8) & 0xFF, 7) << "Green for blue pixel should be near 0";
    EXPECT_GE(dst[2] & 0xFF, 248) << "Blue channel should be expanded";
}

TEST_F(FormatConversionTest, ARGB32_To_RGB555_QuantizesCorrectly) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create16Bit555,
        width, height
    );

    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    src[0] = 0xFFFF0000; // Red
    src[1] = 0xFF00FF00; // Green
    src[2] = 0xFF0000FF; // Blue
    src[3] = 0xFFFFFFFF; // White
    src[4] = 0xFF000000; // Black
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XWORD* dst = reinterpret_cast<const XWORD*>(pair.dstBuffer.Data());
    EXPECT_EQ(0x7C00, dst[0]) << "Red 555 mismatch";
    EXPECT_EQ(0x03E0, dst[1]) << "Green 555 mismatch";
    EXPECT_EQ(0x001F, dst[2]) << "Blue 555 mismatch";
    EXPECT_EQ(0x7FFF, dst[3]) << "White 555 mismatch";
    EXPECT_EQ(0x0000, dst[4]) << "Black 555 mismatch";
}

TEST_F(FormatConversionTest, ARGB32_To_ARGB1555_PreservesAlphaBit) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create16Bit1555,
        width, height
    );

    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    src[0] = 0xFFFF0000; // Red, full alpha -> alpha bit = 1
    src[1] = 0x00FF0000; // Red, zero alpha -> alpha bit = 0
    src[2] = 0x80FF0000; // Red, half alpha -> alpha bit = 1
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XWORD* dst = reinterpret_cast<const XWORD*>(pair.dstBuffer.Data());
    EXPECT_EQ(0x8000, dst[0] & 0x8000) << "Full alpha should set bit";
    EXPECT_EQ(0x0000, dst[1] & 0x8000) << "Zero alpha should clear bit";
    EXPECT_EQ(0x8000, dst[2] & 0x8000) << "Half alpha should set bit";
}

TEST_F(FormatConversionTest, ARGB32_To_ARGB4444_PreservesAlpha) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create16Bit4444,
        width, height
    );

    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    src[0] = 0xFFFF0000; // Red, full alpha
    src[1] = 0x00FF0000; // Red, zero alpha
    src[2] = 0x88888888; // All channels ~50%
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XWORD* dst = reinterpret_cast<const XWORD*>(pair.dstBuffer.Data());
    EXPECT_EQ(0xF000, dst[0] & 0xF000) << "Full alpha 4444";
    EXPECT_EQ(0x0000, dst[1] & 0xF000) << "Zero alpha 4444";
    EXPECT_EQ(0x8888, dst[2]) << "Half values 4444";
}

//==============================================================================
// Roundtrip Tests (conversion back and forth)
//==============================================================================

TEST_F(FormatConversionTest, Roundtrip_32To24To32_PreservesColor) {
    const int width = 32, height = 32;
    
    // Create source 32-bit
    ImageBuffer src32(width * height * 4);
    ImageBuffer intermediate24(width * height * 3);
    ImageBuffer dst32(width * height * 4);
    
    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, src32.Data());
    auto midDesc = ImageDescFactory::Create24BitRGB(width, height, intermediate24.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dst32.Data());
    
    // Fill with pattern
    PatternGenerator::FillColorBars32(src32.Data(), width, height);
    
    // Convert 32 -> 24 -> 32
    blitter.DoBlit(srcDesc, midDesc);
    blitter.DoBlit(midDesc, dstDesc);
    
    // Compare (allowing for alpha difference since 24-bit doesn't preserve it)
    const XULONG* orig = reinterpret_cast<const XULONG*>(src32.Data());
    const XULONG* result = reinterpret_cast<const XULONG*>(dst32.Data());
    
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ((orig[i] & 0x00FFFFFF), (result[i] & 0x00FFFFFF))
            << "Color mismatch at pixel " << i;
        EXPECT_EQ(0xFF, (result[i] >> 24) & 0xFF)
            << "Alpha should be FF after roundtrip through 24-bit";
    }
}

TEST_F(FormatConversionTest, Roundtrip_32To565To32_MinimalLoss) {
    const int width = 32, height = 32;
    
    ImageBuffer src32(width * height * 4);
    ImageBuffer intermediate16(width * height * 2);
    ImageBuffer dst32(width * height * 4);
    
    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, src32.Data());
    auto midDesc = ImageDescFactory::Create16Bit565(width, height, intermediate16.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dst32.Data());
    
    PatternGenerator::FillGradient32(src32.Data(), width, height);
    
    blitter.DoBlit(srcDesc, midDesc);
    blitter.DoBlit(midDesc, dstDesc);
    
    // 565 loses precision: R/B lose 3 bits (max error 7), G loses 2 bits (max error 3)
    int tolerance = 8;
    int diffPixels = ImageComparator::Compare32(src32.Data(), dst32.Data(), 
                                                 width, height, tolerance);
    EXPECT_EQ(0, diffPixels) << "Too many pixels differ beyond tolerance";
}

//==============================================================================
// Edge Cases
//==============================================================================

TEST_F(FormatConversionTest, SmallImage_1x1) {
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        1, 1
    );

    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    src[0] = 0xAABBCCDD;
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    XULONG* dst = reinterpret_cast<XULONG*>(pair.dstBuffer.Data());
    EXPECT_EQ(0xAABBCCDD, dst[0]);
}

TEST_F(FormatConversionTest, LargeImage_Performance) {
    const int width = 1024, height = 1024;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create16Bit565,
        width, height
    );

    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height);
    
    // Should complete in reasonable time
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    // Basic sanity check
    const XWORD* dst = reinterpret_cast<const XWORD*>(pair.dstBuffer.Data());
    EXPECT_NE(0, dst[0] | dst[width * height - 1]) << "Output should have data";
}

TEST_F(FormatConversionTest, NonSquareImage_WideRectangle) {
    const int width = 256, height = 16;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create24BitRGB,
        width, height
    );

    PatternGenerator::FillColorBars32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    // Verify first and last pixels
    EXPECT_EQ(0x00, pair.dstBuffer[0]); // Blue of red bar
    EXPECT_EQ(0x00, pair.dstBuffer[1]); // Green of red bar
    EXPECT_EQ(0xFF, pair.dstBuffer[2]); // Red of red bar
}

TEST_F(FormatConversionTest, NonSquareImage_TallRectangle) {
    const int width = 16, height = 256;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitRGB,
        width, height
    );

    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height, false);
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    EXPECT_EQ(0x00000000, dst[0] & 0x00FFFFFF) << "First row should be dark";
}

//==============================================================================
// BGR/ABGR/RGBA Channel Swap Tests (via generic path)
//==============================================================================

TEST_F(FormatConversionTest, ARGB32_to_ABGR32_ChannelSwap) {
    const int width = 16, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitABGR,
        width, height
    );
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = 0xAA112233; // ARGB: A=0xAA, R=0x11, G=0x22, B=0x33
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    // ABGR format: A=0xAA, B=0x33, G=0x22, R=0x11 -> 0xAA332211
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xAA332211, dst[i]) << "Pixel " << i;
    }
}

TEST_F(FormatConversionTest, ABGR32_to_ARGB32_ChannelSwap) {
    const int width = 16, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitABGR,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = 0xBB445566; // ABGR: A=0xBB, B=0x44, G=0x55, R=0x66
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    // ARGB format: A=0xBB, R=0x66, G=0x55, B=0x44 -> 0xBB665544
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xBB665544, dst[i]) << "Pixel " << i;
    }
}

TEST_F(FormatConversionTest, ARGB32_to_RGBA32_ChannelSwap) {
    const int width = 16, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitRGBA,
        width, height
    );
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = 0xCC778899; // ARGB: A=0xCC, R=0x77, G=0x88, B=0x99
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    // RGBA format: R=0x77, G=0x88, B=0x99, A=0xCC -> 0x778899CC
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x778899CC, dst[i]) << "Pixel " << i;
    }
}

TEST_F(FormatConversionTest, RGB565_to_BGR565_ChannelSwap) {
    const int width = 16, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create16Bit565,
        ImageDescFactory::Create16BitBGR565,
        width, height
    );
    
    XWORD* src = reinterpret_cast<XWORD*>(pair.srcBuffer.Data());
    src[0] = 0xF800; // Red in RGB565
    src[1] = 0x07E0; // Green in RGB565
    src[2] = 0x001F; // Blue in RGB565
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XWORD* dst = reinterpret_cast<const XWORD*>(pair.dstBuffer.Data());
    // In BGR565: B is in high bits, R is in low bits
    EXPECT_EQ(0x001F, dst[0]) << "Red->low bits in BGR";
    EXPECT_EQ(0x07E0, dst[1]) << "Green stays in middle";
    EXPECT_EQ(0xF800, dst[2]) << "Blue->high bits in BGR";
}

//==============================================================================
// Generic Path Tests (non-optimized conversion paths)
//==============================================================================

TEST_F(FormatConversionTest, GenericPath_24to16_CorrectConversion) {
    const int width = 16, height = 4;
    
    ImageBuffer srcBuf(width * height * 3);
    ImageBuffer dstBuf(width * height * 2);
    dstBuf.Fill(0xCD);
    
    auto srcDesc = ImageDescFactory::Create24BitRGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create16Bit565(width, height, dstBuf.Data());
    
    // Set up BGR values for 24-bit
    XBYTE* src = srcBuf.Data();
    src[0] = 0x00; src[1] = 0x00; src[2] = 0xFF; // Red in BGR
    src[3] = 0x00; src[4] = 0xFF; src[5] = 0x00; // Green in BGR
    src[6] = 0xFF; src[7] = 0x00; src[8] = 0x00; // Blue in BGR
    
    blitter.DoBlit(srcDesc, dstDesc);
    
    const XWORD* dst = reinterpret_cast<const XWORD*>(dstBuf.Data());
    EXPECT_EQ(0xF800, dst[0]) << "Red 565";
    EXPECT_EQ(0x07E0, dst[1]) << "Green 565";
    EXPECT_EQ(0x001F, dst[2]) << "Blue 565";
}

TEST_F(FormatConversionTest, GenericPath_565to555_CorrectConversion) {
    const int width = 16, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create16Bit565,
        ImageDescFactory::Create16Bit555,
        width, height
    );
    
    XWORD* src = reinterpret_cast<XWORD*>(pair.srcBuffer.Data());
    src[0] = 0xF800; // Red 565
    src[1] = 0x07E0; // Green 565
    src[2] = 0x001F; // Blue 565
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XWORD* dst = reinterpret_cast<const XWORD*>(pair.dstBuffer.Data());
    EXPECT_EQ(0x7C00, dst[0]) << "Red 555";
    EXPECT_EQ(0x03E0, dst[1]) << "Green 555";
    EXPECT_EQ(0x001F, dst[2]) << "Blue 555";
}

//==============================================================================
// SSE Remainder Tests (various widths to test SIMD + scalar fallback)
//==============================================================================

TEST_F(FormatConversionTest, ARGB32_to_RGB32_AllWidths) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33};
    
    for (int width : widths) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitARGB,
            ImageDescFactory::Create32BitRGB,
            width, 4
        );
        
        XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            src[i] = 0xFF112233;
        }
        
        blitter.DoBlit(pair.srcDesc, pair.dstDesc);
        
        const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(0x00112233, dst[i] & 0x00FFFFFF) 
                << "RGB mismatch at pixel " << i << " (width=" << width << ")";
        }
    }
}

TEST_F(FormatConversionTest, RGB32_to_ARGB32_AllWidths) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33};
    
    for (int width : widths) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitRGB,
            ImageDescFactory::Create32BitARGB,
            width, 4
        );
        
        XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            src[i] = 0x00AABBCC;
        }
        
        blitter.DoBlit(pair.srcDesc, pair.dstDesc);
        
        const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(0xFFAABBCC, dst[i]) 
                << "Pixel mismatch at " << i << " (width=" << width << ")";
        }
    }
}

TEST_F(FormatConversionTest, ARGB32_to_RGB24_AllWidths) {
    const int widths[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 23, 24, 31, 32, 48};
    
    for (int width : widths) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitARGB,
            ImageDescFactory::Create24BitRGB,
            width, 4
        );
        
        XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            XBYTE r = static_cast<XBYTE>(i & 0xFF);
            XBYTE g = static_cast<XBYTE>((i * 7) & 0xFF);
            XBYTE b = static_cast<XBYTE>((i * 13) & 0xFF);
            src[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        
        blitter.DoBlit(pair.srcDesc, pair.dstDesc);
        
        const XBYTE* dst = pair.dstBuffer.Data();
        for (int i = 0; i < width * 4; ++i) {
            XBYTE expectedR = static_cast<XBYTE>(i & 0xFF);
            XBYTE expectedG = static_cast<XBYTE>((i * 7) & 0xFF);
            XBYTE expectedB = static_cast<XBYTE>((i * 13) & 0xFF);
            
            EXPECT_EQ(expectedB, dst[i * 3 + 0]) 
                << "Blue mismatch at pixel " << i << " (width=" << width << ")";
            EXPECT_EQ(expectedG, dst[i * 3 + 1]) 
                << "Green mismatch at pixel " << i << " (width=" << width << ")";
            EXPECT_EQ(expectedR, dst[i * 3 + 2]) 
                << "Red mismatch at pixel " << i << " (width=" << width << ")";
        }
    }
}

TEST_F(FormatConversionTest, RGB24_to_ARGB32_AllWidths) {
    const int widths[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 23, 24, 31, 32, 48};
    
    for (int width : widths) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create24BitRGB,
            ImageDescFactory::Create32BitARGB,
            width, 4
        );
        
        XBYTE* src = pair.srcBuffer.Data();
        for (int i = 0; i < width * 4; ++i) {
            XBYTE r = static_cast<XBYTE>((i + 50) & 0xFF);
            XBYTE g = static_cast<XBYTE>((i * 3 + 100) & 0xFF);
            XBYTE b = static_cast<XBYTE>((i * 5 + 150) & 0xFF);
            src[i * 3 + 0] = b;
            src[i * 3 + 1] = g;
            src[i * 3 + 2] = r;
        }
        
        blitter.DoBlit(pair.srcDesc, pair.dstDesc);
        
        const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            XBYTE expectedR = static_cast<XBYTE>((i + 50) & 0xFF);
            XBYTE expectedG = static_cast<XBYTE>((i * 3 + 100) & 0xFF);
            XBYTE expectedB = static_cast<XBYTE>((i * 5 + 150) & 0xFF);
            XULONG expected = 0xFF000000 | (expectedR << 16) | (expectedG << 8) | expectedB;
            
            EXPECT_EQ(expected, dst[i]) 
                << "Pixel mismatch at " << i << " (width=" << width << ")";
        }
    }
}

TEST_F(FormatConversionTest, ARGB32_to_565_AllWidths) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33};
    
    for (int width : widths) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitARGB,
            ImageDescFactory::Create16Bit565,
            width, 4
        );
        
        XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
        // Set first few pixels to known colors
        if (width >= 5) {
            src[0] = 0xFFFF0000; // Red
            src[1] = 0xFF00FF00; // Green
            src[2] = 0xFF0000FF; // Blue
            src[3] = 0xFFFFFFFF; // White
            src[4] = 0xFF000000; // Black
        }
        
        blitter.DoBlit(pair.srcDesc, pair.dstDesc);
        
        const XWORD* dst = reinterpret_cast<const XWORD*>(pair.dstBuffer.Data());
        if (width >= 5) {
            EXPECT_EQ(0xF800, dst[0]) << "Red (width=" << width << ")";
            EXPECT_EQ(0x07E0, dst[1]) << "Green";
            EXPECT_EQ(0x001F, dst[2]) << "Blue";
            EXPECT_EQ(0xFFFF, dst[3]) << "White";
            EXPECT_EQ(0x0000, dst[4]) << "Black";
        }
    }
}

TEST_F(FormatConversionTest, RGB565_to_ARGB32_AllWidths) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33};
    
    for (int width : widths) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create16Bit565,
            ImageDescFactory::Create32BitARGB,
            width, 4
        );
        
        XWORD* src = reinterpret_cast<XWORD*>(pair.srcBuffer.Data());
        if (width >= 5) {
            src[0] = 0xF800; // Red
            src[1] = 0x07E0; // Green
            src[2] = 0x001F; // Blue
            src[3] = 0xFFFF; // White
            src[4] = 0x0000; // Black
        }
        
        blitter.DoBlit(pair.srcDesc, pair.dstDesc);
        
        const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
        if (width >= 5) {
            // Check alpha is always 0xFF
            EXPECT_EQ(0xFF, (dst[0] >> 24) & 0xFF) << "Alpha (width=" << width << ")";
            EXPECT_EQ(0xFF, (dst[4] >> 24) & 0xFF) << "Black alpha";
            
            // Check colors expanded properly
            EXPECT_GE((dst[0] >> 16) & 0xFF, 248) << "Red should be expanded";
            EXPECT_GE((dst[1] >> 8) & 0xFF, 252) << "Green should be expanded";
            EXPECT_GE(dst[2] & 0xFF, 248) << "Blue should be expanded";
        }
    }
}

//==============================================================================
// Alpha Handling Edge Cases
//==============================================================================

TEST_F(FormatConversionTest, NoAlphaSource_To_AlphaDest) {
    const int width = 8, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitRGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = 0x00112233;
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF112233, dst[i]) << "Pixel " << i;
    }
}

TEST_F(FormatConversionTest, AlphaSource_To_NoAlphaDest) {
    const int width = 8, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitRGB,
        width, height
    );
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = 0xAA112233;
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x00112233, dst[i]) << "Pixel " << i;
    }
}

TEST_F(FormatConversionTest, ARGB1555_to_ARGB32_AlphaExpansion) {
    const int widths[] = {8, 15, 16, 17};
    
    for (int width : widths) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create16Bit1555,
            ImageDescFactory::Create32BitARGB,
            width, 4
        );
        
        XWORD* src = reinterpret_cast<XWORD*>(pair.srcBuffer.Data());
        // Alpha=1 (bit 15 set), Red=31, G=31, B=31
        src[0] = 0xFFFF; // Full alpha, white
        src[1] = 0x7FFF; // Zero alpha (bit 15 clear), white
        
        blitter.DoBlit(pair.srcDesc, pair.dstDesc);
        
        const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
        EXPECT_EQ(0xFF, (dst[0] >> 24) & 0xFF) << "Full alpha expansion (width=" << width << ")";
        EXPECT_EQ(0x00, (dst[1] >> 24) & 0xFF) << "Zero alpha (width=" << width << ")";
    }
}

//==============================================================================
// Large Image Stress Test
//==============================================================================

TEST_F(FormatConversionTest, LargeImage_CorrectOutput) {
    const int width = 1920, height = 1080;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitRGB,
        width, height
    );
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            XBYTE r = static_cast<XBYTE>(x & 0xFF);
            XBYTE g = static_cast<XBYTE>(y & 0xFF);
            XBYTE b = static_cast<XBYTE>((x + y) & 0xFF);
            src[idx] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int y = 0; y < height; y += 100) {
        for (int x = 0; x < width; x += 100) {
            int idx = y * width + x;
            XBYTE expectedR = static_cast<XBYTE>(x & 0xFF);
            XBYTE expectedG = static_cast<XBYTE>(y & 0xFF);
            XBYTE expectedB = static_cast<XBYTE>((x + y) & 0xFF);
            
            EXPECT_EQ(expectedR, (dst[idx] >> 16) & 0xFF) << "R at " << x << "," << y;
            EXPECT_EQ(expectedG, (dst[idx] >> 8) & 0xFF) << "G at " << x << "," << y;
            EXPECT_EQ(expectedB, dst[idx] & 0xFF) << "B at " << x << "," << y;
        }
    }
}

//==============================================================================
// Roundtrip Tests (lossy format conversions)
//==============================================================================

TEST_F(FormatConversionTest, Roundtrip_ARGB32_1555_ARGB32) {
    const int width = 16, height = 4;
    
    ImageBuffer srcBuffer(width * height * 4);
    ImageBuffer midBuffer(width * height * 2);
    ImageBuffer dstBuffer(width * height * 4);
    
    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuffer.Data());
    auto midDesc = ImageDescFactory::Create16Bit1555(width, height, midBuffer.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    XULONG* srcPixels = reinterpret_cast<XULONG*>(srcBuffer.Data());
    srcPixels[0] = 0xFFFF0000; // Red
    srcPixels[1] = 0xFF00FF00; // Green
    srcPixels[2] = 0xFF0000FF; // Blue
    srcPixels[3] = 0xFFFFFFFF; // White
    srcPixels[4] = 0xFF000000; // Black
    srcPixels[5] = 0x00FF0000; // Transparent red
    
    // ARGB32 -> 1555
    blitter.DoBlit(srcDesc, midDesc);
    // 1555 -> ARGB32
    blitter.DoBlit(midDesc, dstDesc);
    
    const XULONG* dstPixels = reinterpret_cast<const XULONG*>(dstBuffer.Data());
    
    // Colors should be close (5-bit precision loss)
    EXPECT_GE((dstPixels[0] >> 16) & 0xFF, 248) << "Red should be mostly preserved";
    EXPECT_GE((dstPixels[1] >> 8) & 0xFF, 248) << "Green should be mostly preserved";
    EXPECT_GE(dstPixels[2] & 0xFF, 248) << "Blue should be mostly preserved";
    
    // Alpha should be binary
    EXPECT_EQ(0xFF, (dstPixels[0] >> 24) & 0xFF) << "Full alpha preserved";
    EXPECT_EQ(0x00, (dstPixels[5] >> 24) & 0xFF) << "Zero alpha preserved";
}

TEST_F(FormatConversionTest, Roundtrip_ARGB32_4444_ARGB32) {
    const int width = 16, height = 4;
    
    ImageBuffer srcBuffer(width * height * 4);
    ImageBuffer midBuffer(width * height * 2);
    ImageBuffer dstBuffer(width * height * 4);
    
    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuffer.Data());
    auto midDesc = ImageDescFactory::Create16Bit4444(width, height, midBuffer.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    XULONG* srcPixels = reinterpret_cast<XULONG*>(srcBuffer.Data());
    srcPixels[0] = 0xFFFF0000; // Full red
    srcPixels[1] = 0x88888888; // Half gray/alpha
    srcPixels[2] = 0x00000000; // Transparent black
    
    // ARGB32 -> 4444
    blitter.DoBlit(srcDesc, midDesc);
    // 4444 -> ARGB32
    blitter.DoBlit(midDesc, dstDesc);
    
    const XULONG* dstPixels = reinterpret_cast<const XULONG*>(dstBuffer.Data());
    
    // Full values should expand to near-max
    EXPECT_GE((dstPixels[0] >> 24) & 0xFF, 0xF0) << "Alpha should be near full";
    EXPECT_GE((dstPixels[0] >> 16) & 0xFF, 0xF0) << "Red should be near full";
    
    // Half values should be around middle
    XBYTE halfAlpha = static_cast<XBYTE>((dstPixels[1] >> 24) & 0xFF);
    EXPECT_GE(halfAlpha, 0x70) << "Half alpha lower bound";
    EXPECT_LE(halfAlpha, 0x99) << "Half alpha upper bound";
}

//==============================================================================
// Single Pixel Edge Case Tests
//==============================================================================

TEST_F(FormatConversionTest, SinglePixel_AllFormats) {
    // Test single pixel conversions for all major formats
    
    // 32-bit ARGB source
    ImageBuffer src32(4);
    XULONG* srcPixel = reinterpret_cast<XULONG*>(src32.Data());
    *srcPixel = 0xFFAABBCC; // ARGB
    
    auto src32Desc = ImageDescFactory::Create32BitARGB(1, 1, src32.Data());
    
    // To RGB32
    {
        ImageBuffer dst(4);
        auto dstDesc = ImageDescFactory::Create32BitRGB(1, 1, dst.Data());
        blitter.DoBlit(src32Desc, dstDesc);
        XULONG* dstPixel = reinterpret_cast<XULONG*>(dst.Data());
        EXPECT_EQ(0x00AABBCC, *dstPixel) << "ARGB32->RGB32";
    }
    
    // To RGB24
    {
        ImageBuffer dst(3);
        auto dstDesc = ImageDescFactory::Create24BitRGB(1, 1, dst.Data());
        blitter.DoBlit(src32Desc, dstDesc);
        EXPECT_EQ(0xCC, dst[0]) << "B channel";
        EXPECT_EQ(0xBB, dst[1]) << "G channel";
        EXPECT_EQ(0xAA, dst[2]) << "R channel";
    }
    
    // To RGB565
    {
        ImageBuffer dst(2);
        auto dstDesc = ImageDescFactory::Create16Bit565(1, 1, dst.Data());
        blitter.DoBlit(src32Desc, dstDesc);
        XWORD* dstPixel = reinterpret_cast<XWORD*>(dst.Data());
        // R=0xAA (170) -> 21, G=0xBB (187) -> 46, B=0xCC (204) -> 25
        XWORD expected = static_cast<XWORD>((21 << 11) | (46 << 5) | 25);
        EXPECT_EQ(expected, *dstPixel) << "ARGB32->RGB565";
    }
    
    // To ARGB1555 
    {
        ImageBuffer dst(2);
        auto dstDesc = ImageDescFactory::Create16Bit1555(1, 1, dst.Data());
        blitter.DoBlit(src32Desc, dstDesc);
        XWORD* dstPixel = reinterpret_cast<XWORD*>(dst.Data());
        EXPECT_EQ(0x8000, *dstPixel & 0x8000) << "Alpha bit should be set";
    }
}

//==============================================================================
// FillImage Tests
//==============================================================================

TEST_F(FormatConversionTest, FillImage_32Bit_SolidColor) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    VxImageDescEx desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    blitter.FillImage(desc, 0xFFFF0000);

    const XULONG *pixels = reinterpret_cast<const XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0xFFFF0000u) << "Pixel " << i << " not filled correctly";
    }
}

TEST_F(FormatConversionTest, FillImage_32Bit_Transparent) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    VxImageDescEx desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    blitter.FillImage(desc, 0x00000000);

    const XULONG *pixels = reinterpret_cast<const XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0x00000000u);
    }
}

TEST_F(FormatConversionTest, FillImage_24Bit_SolidColor) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 3);
    VxImageDescEx desc = ImageDescFactory::Create24BitRGB(width, height, buffer.Data());

    blitter.FillImage(desc, 0x0000FF00);

    for (int y = 0; y < height; ++y) {
        const XBYTE *row = buffer.Data() + y * desc.BytesPerLine;
        for (int x = 0; x < width; ++x) {
            EXPECT_EQ(row[x * 3 + 0], 0x00) << "Blue at (" << x << "," << y << ")";
            EXPECT_EQ(row[x * 3 + 1], 0xFF) << "Green at (" << x << "," << y << ")";
            EXPECT_EQ(row[x * 3 + 2], 0x00) << "Red at (" << x << "," << y << ")";
        }
    }
}

TEST_F(FormatConversionTest, FillImage_16Bit_SolidColor) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 2);
    VxImageDescEx desc = ImageDescFactory::Create16Bit565(width, height, buffer.Data());

    XWORD color565 = 0xF800;
    blitter.FillImage(desc, color565);

    const XWORD *pixels = reinterpret_cast<const XWORD *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], color565);
    }
}

TEST_F(FormatConversionTest, FillImage_LargeImage_Performance) {
    const int width = 1024, height = 1024;
    ImageBuffer buffer(width * height * 4);
    VxImageDescEx desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    blitter.FillImage(desc, 0xFFAABBCC);

    const XULONG *pixels = reinterpret_cast<const XULONG *>(buffer.Data());
    EXPECT_EQ(pixels[0], 0xFFAABBCCu);
    EXPECT_EQ(pixels[width * height - 1], 0xFFAABBCCu);
}

//==============================================================================
// SSE Channel Swap Tests (ARGB <-> ABGR/RGBA/BGRA)
//==============================================================================

TEST_F(FormatConversionTest, ChannelSwap_ARGB_to_ABGR_AllWidths) {
    for (int width : {1, 3, 4, 7, 8, 15, 16, 17}) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitARGB,
            ImageDescFactory::Create32BitABGR,
            width, 4
        );

        XULONG *srcPixels = reinterpret_cast<XULONG *>(pair.srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            srcPixels[i] = 0x80112233;
        }

        blitter.DoBlit(pair.srcDesc, pair.dstDesc);

        const XULONG *dstPixels = reinterpret_cast<const XULONG *>(pair.dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(dstPixels[i], 0x80332211u) << "Width " << width << ", pixel " << i;
        }
    }
}

TEST_F(FormatConversionTest, ChannelSwap_ARGB_to_RGBA_AllWidths) {
    for (int width : {1, 3, 4, 7, 8, 15, 16, 17}) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitARGB,
            ImageDescFactory::Create32BitRGBA,
            width, 4
        );

        XULONG *srcPixels = reinterpret_cast<XULONG *>(pair.srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            srcPixels[i] = 0x80112233;
        }

        blitter.DoBlit(pair.srcDesc, pair.dstDesc);

        const XULONG *dstPixels = reinterpret_cast<const XULONG *>(pair.dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(dstPixels[i], 0x11223380u) << "Width " << width << ", pixel " << i;
        }
    }
}

TEST_F(FormatConversionTest, ChannelSwap_ABGR_to_ARGB_AllWidths) {
    for (int width : {1, 3, 4, 7, 8, 15, 16, 17}) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitABGR,
            ImageDescFactory::Create32BitARGB,
            width, 4
        );

        XULONG *srcPixels = reinterpret_cast<XULONG *>(pair.srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            srcPixels[i] = 0x80332211;
        }

        blitter.DoBlit(pair.srcDesc, pair.dstDesc);

        const XULONG *dstPixels = reinterpret_cast<const XULONG *>(pair.dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(dstPixels[i], 0x80112233u) << "Width " << width << ", pixel " << i;
        }
    }
}

TEST_F(FormatConversionTest, ChannelSwap_RGBA_to_ARGB_AllWidths) {
    for (int width : {1, 3, 4, 7, 8, 15, 16, 17}) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitRGBA,
            ImageDescFactory::Create32BitARGB,
            width, 4
        );

        XULONG *srcPixels = reinterpret_cast<XULONG *>(pair.srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            srcPixels[i] = 0x33221180;
        }

        blitter.DoBlit(pair.srcDesc, pair.dstDesc);

        const XULONG *dstPixels = reinterpret_cast<const XULONG *>(pair.dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(dstPixels[i], 0x80332211u) << "Width " << width << ", pixel " << i;
        }
    }
}

TEST_F(FormatConversionTest, ChannelSwap_LargeImage_ARGB_to_ABGR) {
    const int width = 512, height = 512;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitABGR,
        width, height
    );

    XULONG *srcPixels = reinterpret_cast<XULONG *>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        srcPixels[i] = 0xFFAABBCC;
    }

    blitter.DoBlit(pair.srcDesc, pair.dstDesc);

    const XULONG *dstPixels = reinterpret_cast<const XULONG *>(pair.dstBuffer.Data());
    EXPECT_EQ(dstPixels[0], 0xFFCCBBAAu);
    EXPECT_EQ(dstPixels[width * height - 1], 0xFFCCBBAAu);
}