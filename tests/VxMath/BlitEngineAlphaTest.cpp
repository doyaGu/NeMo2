/**
 * @file BlitEngineAlphaTest.cpp
 * @brief Tests for VxBlitEngine alpha channel manipulation functionality.
 *
 * Tests:
 * - Setting constant alpha values
 * - Copying alpha from byte arrays
 * - Alpha preservation during format conversions
 * - Alpha behavior in various pixel formats
 */

#include "BlitEngineTestHelpers.h"

using namespace BlitEngineTest;

class AlphaBlitTest : public BlitEngineTestBase {};

//==============================================================================
// Constant Alpha Tests (DoAlphaBlit with single value)
//==============================================================================

TEST_F(AlphaBlitTest, SetConstantAlpha_32BitARGB_FullAlpha) {
    const int width = 32, height = 32;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    // Fill with known color and zero alpha
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 128, 64, 192, 0);
    
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(255));
    
    // Save output image
    ImageWriter::SaveFromDesc("alpha_const_full", desc, "Alpha");
    
    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF, (pixels[i] >> 24) & 0xFF) 
            << "Alpha should be 255 at pixel " << i;
        EXPECT_EQ(128, (pixels[i] >> 16) & 0xFF) 
            << "Red should be preserved at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, SetConstantAlpha_32BitARGB_ZeroAlpha) {
    const int width = 32, height = 32;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    // Fill with full alpha
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 128, 64, 192, 255);
    
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(0));
    
    // Save output image
    ImageWriter::SaveFromDesc("alpha_const_zero", desc, "Alpha");
    
    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x00, (pixels[i] >> 24) & 0xFF) 
            << "Alpha should be 0 at pixel " << i;
        EXPECT_EQ(128, (pixels[i] >> 16) & 0xFF) 
            << "Red should be preserved at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, SetConstantAlpha_32BitARGB_HalfAlpha) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    PatternGenerator::FillColorBars32(buffer.Data(), width, height);
    
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(128));
    
    // Save output image
    ImageWriter::SaveFromDesc("alpha_const_half", desc, "Alpha");
    
    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(128, (pixels[i] >> 24) & 0xFF) 
            << "Alpha should be 128 at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, SetConstantAlpha_16Bit1555_SetsAlphaBit) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit1555(width, height, buffer.Data());
    
    // Fill with zeros
    buffer.Fill(0);
    
    // Set alpha >= 128 should set the bit
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(200));
    
    // Save output image
    ImageWriter::SaveFromDesc("alpha_1555_set", desc, "Alpha");
    
    const XWORD* pixels = reinterpret_cast<const XWORD*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x8000, pixels[i] & 0x8000) 
            << "Alpha bit should be set at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, SetConstantAlpha_16Bit1555_ClearsAlphaBit) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit1555(width, height, buffer.Data());
    
    // Fill with all bits set
    buffer.Fill(0xFF);
    
    // Set alpha < 128 should clear the bit
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(50));
    
    // Save output image
    ImageWriter::SaveFromDesc("alpha_1555_clear", desc, "Alpha");
    
    const XWORD* pixels = reinterpret_cast<const XWORD*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x0000, pixels[i] & 0x8000) 
            << "Alpha bit should be clear at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, SetConstantAlpha_16Bit4444_SetsAlpha) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit4444(width, height, buffer.Data());
    
    buffer.Fill(0);
    
    // Alpha 0xF0 should become 0xF in 4-bit
    blitter.DoAlphaBlit(desc, 0xF0);
    
    // Save output image
    ImageWriter::SaveFromDesc("alpha_4444_set", desc, "Alpha");
    
    const XWORD* pixels = reinterpret_cast<const XWORD*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xF000, pixels[i] & 0xF000) 
            << "Alpha nibble should be 0xF at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, SetConstantAlpha_NoAlphaFormat_NoOp) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitRGB(width, height, buffer.Data());
    
    // Fill with pattern
    PatternGenerator::FillGradient32(buffer.Data(), width, height);
    
    // Copy original for comparison
    ImageBuffer original(buffer.Size());
    memcpy(original.Data(), buffer.Data(), buffer.Size());
    
    // This should do nothing since format has no alpha
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(128));
    
    // Should be unchanged
    EXPECT_EQ(0, memcmp(buffer.Data(), original.Data(), buffer.Size()));
}

//==============================================================================
// Copy Alpha Array Tests (DoAlphaBlit with array)
//==============================================================================

TEST_F(AlphaBlitTest, CopyAlphaArray_32BitARGB_CopiesCorrectly) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    ImageBuffer alphaValues(width * height);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    // Fill with known color
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 255, 128, 64, 0);
    
    // Create gradient alpha
    for (int i = 0; i < width * height; ++i) {
        alphaValues[i] = static_cast<XBYTE>(i * 4);
    }
    
    blitter.DoAlphaBlit(desc, alphaValues.Data());
    
    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(alphaValues[i], (pixels[i] >> 24) & 0xFF) 
            << "Alpha mismatch at pixel " << i;
        EXPECT_EQ(255, (pixels[i] >> 16) & 0xFF) 
            << "Red should be preserved at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, CopyAlphaArray_16Bit4444_QuantizesCorrectly) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 2);
    ImageBuffer alphaValues(width * height);
    auto desc = ImageDescFactory::Create16Bit4444(width, height, buffer.Data());
    
    buffer.Fill(0);
    
    // Test specific alpha values
    alphaValues[0] = 0x00; // -> 0x0
    alphaValues[1] = 0x10; // -> 0x1
    alphaValues[2] = 0x80; // -> 0x8
    alphaValues[3] = 0xFF; // -> 0xF
    
    blitter.DoAlphaBlit(desc, alphaValues.Data());
    
    // Save output image
    ImageWriter::SaveFromDesc("alpha_4444_array", desc, "Alpha");
    
    const XWORD* pixels = reinterpret_cast<const XWORD*>(buffer.Data());
    EXPECT_EQ(0x0000, pixels[0] & 0xF000) << "Alpha 0x00 -> 0x0";
    EXPECT_EQ(0x1000, pixels[1] & 0xF000) << "Alpha 0x10 -> 0x1";
    EXPECT_EQ(0x8000, pixels[2] & 0xF000) << "Alpha 0x80 -> 0x8";
    EXPECT_EQ(0xF000, pixels[3] & 0xF000) << "Alpha 0xFF -> 0xF";
}

TEST_F(AlphaBlitTest, CopyAlphaArray_16Bit1555_ThresholdsCorrectly) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 2);
    ImageBuffer alphaValues(width * height);
    auto desc = ImageDescFactory::Create16Bit1555(width, height, buffer.Data());
    
    buffer.Fill(0x7F); // Color data, alpha bit clear
    
    // Test threshold behavior
    alphaValues[0] = 0;   // -> bit 0
    alphaValues[1] = 127; // -> bit 0 (below 128)
    alphaValues[2] = 128; // -> bit 1 (at threshold)
    alphaValues[3] = 255; // -> bit 1
    
    blitter.DoAlphaBlit(desc, alphaValues.Data());
    
    // Save output image
    ImageWriter::SaveFromDesc("alpha_1555_array", desc, "Alpha");
    
    const XWORD* pixels = reinterpret_cast<const XWORD*>(buffer.Data());
    EXPECT_EQ(0x0000, pixels[0] & 0x8000) << "Alpha 0 -> bit 0";
    EXPECT_EQ(0x0000, pixels[1] & 0x8000) << "Alpha 127 -> bit 0";
    EXPECT_EQ(0x8000, pixels[2] & 0x8000) << "Alpha 128 -> bit 1";
    EXPECT_EQ(0x8000, pixels[3] & 0x8000) << "Alpha 255 -> bit 1";
}

TEST_F(AlphaBlitTest, CopyAlphaArray_PreservesColor) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    ImageBuffer alphaValues(width * height, 128);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    // Create unique pattern
    PatternGenerator::FillUniquePixels32(buffer.Data(), width, height);
    
    // Save colors for comparison
    ImageBuffer colorCopy(buffer.Size());
    memcpy(colorCopy.Data(), buffer.Data(), buffer.Size());
    
    blitter.DoAlphaBlit(desc, alphaValues.Data());
    
    const XULONG* original = reinterpret_cast<const XULONG*>(colorCopy.Data());
    const XULONG* result = reinterpret_cast<const XULONG*>(buffer.Data());
    
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(original[i] & 0x00FFFFFF, result[i] & 0x00FFFFFF)
            << "Color should be preserved at pixel " << i;
        EXPECT_EQ(128, (result[i] >> 24) & 0xFF)
            << "Alpha should be 128 at pixel " << i;
    }
}

//==============================================================================
// Alpha Channel Preservation During Blit
//==============================================================================

TEST_F(AlphaBlitTest, BlitPreservesAlpha_32BitARGBToARGB) {
    const int width = 16, height = 16;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    // Create image with varying alpha
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        XBYTE alpha = static_cast<XBYTE>(i);
        src[i] = (alpha << 24) | 0x00FF8040;
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    EXPECT_EQ(0, ImageComparator::Compare32(
        pair.srcBuffer.Data(), pair.dstBuffer.Data(), width, height, 0));
}

TEST_F(AlphaBlitTest, BlitLosesAlpha_ARGBTo24Bit) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create24BitRGB,
        width, height
    );
    
    // Fill with varying alpha
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = ((i * 4) << 24) | 0x00AABBCC;
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    // Check colors preserved
    const XBYTE* dst = pair.dstBuffer.Data();
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xCC, dst[i * 3 + 0]) << "Blue";
        EXPECT_EQ(0xBB, dst[i * 3 + 1]) << "Green";
        EXPECT_EQ(0xAA, dst[i * 3 + 2]) << "Red";
    }
}

TEST_F(AlphaBlitTest, BlitRestoresAlpha_24BitToARGB) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create24BitRGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    PatternGenerator::FillSolid24(pair.srcBuffer.Data(), width, height, 
                                   0xAA, 0xBB, 0xCC);
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF, (dst[i] >> 24) & 0xFF) 
            << "Alpha should be 0xFF after 24->32 conversion";
    }
}

//==============================================================================
// Edge Cases
//==============================================================================

TEST_F(AlphaBlitTest, AlphaOnSinglePixel) {
    ImageBuffer buffer(4); // Single 32-bit pixel
    auto desc = ImageDescFactory::Create32BitARGB(1, 1, buffer.Data());
    
    XULONG* pixel = reinterpret_cast<XULONG*>(buffer.Data());
    *pixel = 0x00FFAA55;
    
    blitter.DoAlphaBlit(desc, 0xDD);
    
    EXPECT_EQ(0xDDFFAA55, *pixel);
}

TEST_F(AlphaBlitTest, AlphaOnLargeImage) {
    const int width = 512, height = 512;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    PatternGenerator::FillCheckerboard32(buffer.Data(), width, height);
    
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(192));
    
    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    // Sample check
    EXPECT_EQ(192, (pixels[0] >> 24) & 0xFF);
    EXPECT_EQ(192, (pixels[width * height - 1] >> 24) & 0xFF);
    EXPECT_EQ(192, (pixels[width * height / 2] >> 24) & 0xFF);
}

TEST_F(AlphaBlitTest, AlphaWithGradientAlphaArray) {
    const int width = 256, height = 1;
    ImageBuffer buffer(width * 4);
    ImageBuffer alphaValues(width);
    auto desc = ImageDescFactory::Create32BitARGB(width, 1, buffer.Data());
    
    buffer.Fill(0);
    
    // Create gradient 0-255
    for (int i = 0; i < 256; ++i) {
        alphaValues[i] = static_cast<XBYTE>(i);
    }
    
    blitter.DoAlphaBlit(desc, alphaValues.Data());
    
    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(i, (pixels[i] >> 24) & 0xFF) 
            << "Alpha gradient mismatch at index " << i;
    }
}

//==============================================================================
// Alpha Blit All Widths (SSE remainder testing)
//==============================================================================

TEST_F(AlphaBlitTest, AlphaBlit_AllWidths_UniformAlpha) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 48, 64};
    
    for (int width : widths) {
        ImageBuffer buffer(width * 4 * 4);
        auto desc = ImageDescFactory::Create32BitARGB(width, 4, buffer.Data());
        
        XULONG* pixels = reinterpret_cast<XULONG*>(buffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            pixels[i] = 0x00112233; // Zero alpha initially
        }
        
        blitter.DoAlphaBlit(desc, 0xAA);
        
        for (int i = 0; i < width * 4; ++i) {
            EXPECT_EQ(0xAA112233, pixels[i]) 
                << "Pixel " << i << " (width=" << width << ")";
        }
    }
}

TEST_F(AlphaBlitTest, AlphaBlit_AllWidths_PerPixelAlpha) {
    const int widths[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33};
    
    for (int width : widths) {
        ImageBuffer buffer(width * 4 * 4);
        ImageBuffer alphaValues(width * 4);
        auto desc = ImageDescFactory::Create32BitARGB(width, 4, buffer.Data());
        
        XULONG* pixels = reinterpret_cast<XULONG*>(buffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            pixels[i] = 0x00112233;
            alphaValues[i] = static_cast<XBYTE>(i & 0xFF);
        }
        
        blitter.DoAlphaBlit(desc, alphaValues.Data());
        
        for (int i = 0; i < width * 4; ++i) {
            XBYTE expectedAlpha = static_cast<XBYTE>(i & 0xFF);
            EXPECT_EQ(expectedAlpha, (pixels[i] >> 24) & 0xFF) 
                << "Pixel " << i << " (width=" << width << ")";
        }
    }
}

//==============================================================================
// 16-bit Alpha Channel Tests
//==============================================================================

TEST_F(AlphaBlitTest, Alpha1555_AllWidths) {
    const int widths[] = {1, 4, 7, 8, 15, 16, 17, 32};
    
    for (int width : widths) {
        ImageBuffer srcBuffer(width * 4 * 2);
        ImageBuffer dstBuffer(width * 4 * 4);
        
        auto src = ImageDescFactory::Create16Bit1555(width, 4, srcBuffer.Data());
        auto dst = ImageDescFactory::Create32BitARGB(width, 4, dstBuffer.Data());
        
        XWORD* srcPixels = reinterpret_cast<XWORD*>(srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            if (i % 2 == 0) {
                srcPixels[i] = 0xFFFF; // Alpha=1, white
            } else {
                srcPixels[i] = 0x7FFF; // Alpha=0, white
            }
        }
        
        blitter.DoBlit(src, dst);
        
        const XULONG* dstPixels = reinterpret_cast<const XULONG*>(dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            if (i % 2 == 0) {
                EXPECT_EQ(0xFF, (dstPixels[i] >> 24) & 0xFF) 
                    << "Alpha should be full (width=" << width << ", i=" << i << ")";
            } else {
                EXPECT_EQ(0x00, (dstPixels[i] >> 24) & 0xFF) 
                    << "Alpha should be zero (width=" << width << ", i=" << i << ")";
            }
        }
    }
}

TEST_F(AlphaBlitTest, Alpha4444_AllWidths) {
    const int widths[] = {1, 4, 7, 8, 15, 16, 17, 32};
    
    for (int width : widths) {
        ImageBuffer srcBuffer(width * 4 * 2);
        ImageBuffer dstBuffer(width * 4 * 4);
        
        auto src = ImageDescFactory::Create16Bit4444(width, 4, srcBuffer.Data());
        auto dst = ImageDescFactory::Create32BitARGB(width, 4, dstBuffer.Data());
        
        XWORD* srcPixels = reinterpret_cast<XWORD*>(srcBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            // ARGB4444: A=F, R=5, G=5, B=5 -> 0xF555
            srcPixels[i] = 0xF555;
        }
        
        blitter.DoBlit(src, dst);
        
        const XULONG* dstPixels = reinterpret_cast<const XULONG*>(dstBuffer.Data());
        for (int i = 0; i < width * 4; ++i) {
            // Alpha should be expanded from 4-bit F (15) to 8-bit FF (255)
            EXPECT_EQ(0xFF, (dstPixels[i] >> 24) & 0xFF) 
                << "Alpha should be expanded (width=" << width << ", i=" << i << ")";
        }
    }
}

//==============================================================================
// Alpha Preservation Through Format Conversions
//==============================================================================

TEST_F(AlphaBlitTest, AlphaPreserved_ARGB32_to_ABGR32) {
    const int width = 16, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitABGR,
        width, height
    );
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        XBYTE alpha = static_cast<XBYTE>(i * 4);
        src[i] = (alpha << 24) | 0x112233;
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        XBYTE expectedAlpha = static_cast<XBYTE>(i * 4);
        EXPECT_EQ(expectedAlpha, (dst[i] >> 24) & 0xFF) 
            << "Alpha should be preserved at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, AlphaPreserved_ARGB32_to_RGBA32) {
    const int width = 16, height = 4;
    
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitRGBA,
        width, height
    );
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        XBYTE alpha = static_cast<XBYTE>(255 - i);
        src[i] = (alpha << 24) | 0xAABBCC;
    }
    
    blitter.DoBlit(pair.srcDesc, pair.dstDesc);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        XBYTE expectedAlpha = static_cast<XBYTE>(255 - i);
        // RGBA format: alpha is in low byte
        EXPECT_EQ(expectedAlpha, dst[i] & 0xFF) 
            << "Alpha should be in low byte at pixel " << i;
    }
}

//==============================================================================
// Zero and Full Alpha Edge Cases
//==============================================================================

TEST_F(AlphaBlitTest, ZeroAlpha_AllPixels) {
    const int width = 32, height = 32;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    XULONG* pixels = reinterpret_cast<XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFF112233; // Start with full alpha
    }
    
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(0x00));
    
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x00112233, pixels[i]) << "Pixel " << i;
    }
}

TEST_F(AlphaBlitTest, FullAlpha_AllPixels) {
    const int width = 32, height = 32;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    XULONG* pixels = reinterpret_cast<XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0x00445566; // Start with zero alpha
    }
    
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(0xFF));
    
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF445566, pixels[i]) << "Pixel " << i;
    }
}

//==============================================================================
// Premultiply Alpha Tests
//==============================================================================

TEST_F(AlphaBlitTest, PremultiplyAlpha_FullOpaque) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFFFF8040;
    }

    blitter.PremultiplyAlpha(desc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0xFFFF8040u) << "Pixel " << i << " should be unchanged for fully opaque";
    }
}

TEST_F(AlphaBlitTest, PremultiplyAlpha_HalfTransparent) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0x80FF8040;
    }

    blitter.PremultiplyAlpha(desc);

    for (int i = 0; i < width * height; ++i) {
        XBYTE a = (pixels[i] >> 24) & 0xFF;
        XBYTE r = (pixels[i] >> 16) & 0xFF;
        XBYTE g = (pixels[i] >> 8) & 0xFF;
        XBYTE b = pixels[i] & 0xFF;
        EXPECT_EQ(a, 0x80) << "Alpha should be preserved";
        EXPECT_NEAR(r, 0x80, 1) << "Red should be ~0x80";
        EXPECT_NEAR(g, 0x40, 1) << "Green should be ~0x40";
        EXPECT_NEAR(b, 0x20, 1) << "Blue should be ~0x20";
    }
}

TEST_F(AlphaBlitTest, PremultiplyAlpha_FullTransparent) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0x00FF8040;
    }

    blitter.PremultiplyAlpha(desc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0x00000000u) << "Pixel " << i << " should be fully transparent";
    }
}

//==============================================================================
// Unpremultiply Alpha Tests
//==============================================================================

TEST_F(AlphaBlitTest, UnpremultiplyAlpha_FullOpaque) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFFFF8040;
    }

    blitter.UnpremultiplyAlpha(desc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0xFFFF8040u) << "Pixel " << i << " should be unchanged for fully opaque";
    }
}

TEST_F(AlphaBlitTest, UnpremultiplyAlpha_HalfTransparent) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0x80804020;
    }

    blitter.UnpremultiplyAlpha(desc);

    for (int i = 0; i < width * height; ++i) {
        XBYTE a = (pixels[i] >> 24) & 0xFF;
        XBYTE r = (pixels[i] >> 16) & 0xFF;
        XBYTE g = (pixels[i] >> 8) & 0xFF;
        XBYTE b = pixels[i] & 0xFF;
        EXPECT_EQ(a, 0x80) << "Alpha should be preserved";
        EXPECT_NEAR(r, 0xFF, 2) << "Red should be ~0xFF";
        EXPECT_NEAR(g, 0x80, 2) << "Green should be ~0x80";
        EXPECT_NEAR(b, 0x40, 2) << "Blue should be ~0x40";
    }
}

//==============================================================================
// Premultiply/Unpremultiply Roundtrip Test
//==============================================================================

TEST_F(AlphaBlitTest, PremultiplyUnpremultiply_Roundtrip) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFF112233 + i;
    }

    ImageBuffer original(width * height * 4);
    memcpy(original.Data(), buffer.Data(), width * height * 4);

    blitter.PremultiplyAlpha(desc);
    blitter.UnpremultiplyAlpha(desc);

    const XULONG *origPixels = reinterpret_cast<const XULONG *>(original.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_NEAR((pixels[i] >> 16) & 0xFF, (origPixels[i] >> 16) & 0xFF, 1) << "Red at pixel " << i;
        EXPECT_NEAR((pixels[i] >> 8) & 0xFF, (origPixels[i] >> 8) & 0xFF, 1) << "Green at pixel " << i;
        EXPECT_NEAR(pixels[i] & 0xFF, origPixels[i] & 0xFF, 1) << "Blue at pixel " << i;
    }
}

//==============================================================================
// SwapRedBlue Tests
//==============================================================================

TEST_F(AlphaBlitTest, SwapRedBlue_RedToBlue) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFFFF0000;
    }

    blitter.SwapRedBlue(desc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0xFF0000FFu) << "Red should become blue at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, SwapRedBlue_GreenUnchanged) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFF00FF00;
    }

    blitter.SwapRedBlue(desc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0xFF00FF00u) << "Green should be unchanged at pixel " << i;
    }
}

TEST_F(AlphaBlitTest, SwapRedBlue_DoubleSwap_Roundtrip) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0x80112233;
    }

    blitter.SwapRedBlue(desc);
    blitter.SwapRedBlue(desc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0x80112233u) << "Double swap should restore original";
    }
}

//==============================================================================
// Large Image Performance Tests
//==============================================================================

TEST_F(AlphaBlitTest, PremultiplyAlpha_LargeImage) {
    const int width = 512, height = 512;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0x80FF8040;
    }

    blitter.PremultiplyAlpha(desc);

    EXPECT_EQ((pixels[0] >> 24) & 0xFF, 0x80) << "Alpha preserved at start";
    EXPECT_EQ((pixels[width * height - 1] >> 24) & 0xFF, 0x80) << "Alpha preserved at end";
}

TEST_F(AlphaBlitTest, SwapRedBlue_LargeImage) {
    const int width = 512, height = 512;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFFAABBCC;
    }

    blitter.SwapRedBlue(desc);

    EXPECT_EQ(pixels[0], 0xFFCCBBAAu);
    EXPECT_EQ(pixels[width * height - 1], 0xFFCCBBAAu);
}

//==============================================================================
// ClearAlpha / SetFullAlpha Tests
//==============================================================================

TEST_F(AlphaBlitTest, ClearAlpha_AllWidths) {
    for (int width : {1, 3, 4, 7, 8, 15, 16, 17, 33}) {
        const int height = 4;
        ImageBuffer buffer(width * height * 4);
        auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

        XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
        for (int i = 0; i < width * height; ++i) {
            pixels[i] = 0xFFAABBCC;
        }

        blitter.ClearAlpha(desc);

        for (int i = 0; i < width * height; ++i) {
            EXPECT_EQ(pixels[i], 0x00AABBCCu) 
                << "Width " << width << ", pixel " << i << ": Alpha should be 0, colors preserved";
        }
    }
}

TEST_F(AlphaBlitTest, SetFullAlpha_AllWidths) {
    for (int width : {1, 3, 4, 7, 8, 15, 16, 17, 33}) {
        const int height = 4;
        ImageBuffer buffer(width * height * 4);
        auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

        XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
        for (int i = 0; i < width * height; ++i) {
            pixels[i] = 0x00AABBCC;
        }

        blitter.SetFullAlpha(desc);

        for (int i = 0; i < width * height; ++i) {
            EXPECT_EQ(pixels[i], 0xFFAABBCCu) 
                << "Width " << width << ", pixel " << i << ": Alpha should be FF, colors preserved";
        }
    }
}

//==============================================================================
// InvertColors Tests
//==============================================================================

TEST_F(AlphaBlitTest, InvertColors_AllWidths) {
    for (int width : {1, 3, 4, 7, 8, 15, 16, 17, 33}) {
        const int height = 4;
        ImageBuffer buffer(width * height * 4);
        auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

        XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
        for (int i = 0; i < width * height; ++i) {
            pixels[i] = 0x80AABBCC;
        }

        blitter.InvertColors(desc);

        for (int i = 0; i < width * height; ++i) {
            EXPECT_EQ(pixels[i], 0x80554433u) 
                << "Width " << width << ", pixel " << i << ": RGB should be inverted, alpha preserved";
        }
    }
}

TEST_F(AlphaBlitTest, InvertColors_DoubleInvert_Roundtrip) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0x80112233 + i;
    }

    ImageBuffer original(width * height * 4);
    memcpy(original.Data(), buffer.Data(), width * height * 4);

    blitter.InvertColors(desc);
    blitter.InvertColors(desc);

    const XULONG *origPixels = reinterpret_cast<const XULONG *>(original.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], origPixels[i]) << "Double invert should restore original at pixel " << i;
    }
}

//==============================================================================
// ConvertToGrayscale Tests
//==============================================================================

TEST_F(AlphaBlitTest, ConvertToGrayscale_AllWidths) {
    for (int width : {1, 3, 4, 7, 8, 15, 16, 17, 33}) {
        const int height = 4;
        ImageBuffer buffer(width * height * 4);
        auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

        XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
        for (int i = 0; i < width * height; ++i) {
            pixels[i] = 0x80FF0000;
        }

        blitter.ConvertToGrayscale(desc);

        for (int i = 0; i < width * height; ++i) {
            XBYTE a = (pixels[i] >> 24) & 0xFF;
            XBYTE r = (pixels[i] >> 16) & 0xFF;
            XBYTE g = (pixels[i] >> 8) & 0xFF;
            XBYTE b = pixels[i] & 0xFF;
            EXPECT_EQ(a, 0x80) << "Width " << width << ", pixel " << i << ": Alpha should be preserved";
            EXPECT_EQ(r, g) << "Width " << width << ", pixel " << i << ": R should equal G";
            EXPECT_EQ(g, b) << "Width " << width << ", pixel " << i << ": G should equal B";
            EXPECT_NEAR(r, 76, 2) << "Width " << width << ", pixel " << i << ": Grayscale of red ~76";
        }
    }
}

TEST_F(AlphaBlitTest, ConvertToGrayscale_White) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFFFFFFFF;
    }

    blitter.ConvertToGrayscale(desc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0xFFFFFFFFu) << "White should remain white";
    }
}

TEST_F(AlphaBlitTest, ConvertToGrayscale_Black) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    XULONG *pixels = reinterpret_cast<XULONG *>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        pixels[i] = 0xFF000000;
    }

    blitter.ConvertToGrayscale(desc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(pixels[i], 0xFF000000u) << "Black should remain black";
    }
}

//==============================================================================
// MultiplyBlend Tests
//==============================================================================

TEST_F(AlphaBlitTest, MultiplyBlend_White) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height * 4);
    ImageBuffer dstBuffer(width * height * 4);
    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuffer.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());

    XULONG *srcPixels = reinterpret_cast<XULONG *>(srcBuffer.Data());
    XULONG *dstPixels = reinterpret_cast<XULONG *>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        srcPixels[i] = 0xFFFFFFFF;
        dstPixels[i] = 0xFF804020;
    }

    blitter.MultiplyBlend(srcDesc, dstDesc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(dstPixels[i], 0xFF804020u) << "Multiplying by white should preserve color";
    }
}

TEST_F(AlphaBlitTest, MultiplyBlend_Black) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height * 4);
    ImageBuffer dstBuffer(width * height * 4);
    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuffer.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());

    XULONG *srcPixels = reinterpret_cast<XULONG *>(srcBuffer.Data());
    XULONG *dstPixels = reinterpret_cast<XULONG *>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        srcPixels[i] = 0xFF000000;
        dstPixels[i] = 0xFFFFFFFF;
    }

    blitter.MultiplyBlend(srcDesc, dstDesc);

    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(dstPixels[i] & 0x00FFFFFF, 0x00000000u) << "Multiplying by black should give black";
        EXPECT_EQ((dstPixels[i] >> 24) & 0xFF, 0xFF) << "Alpha should be preserved";
    }
}

TEST_F(AlphaBlitTest, MultiplyBlend_HalfValues) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height * 4);
    ImageBuffer dstBuffer(width * height * 4);
    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuffer.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());

    XULONG *srcPixels = reinterpret_cast<XULONG *>(srcBuffer.Data());
    XULONG *dstPixels = reinterpret_cast<XULONG *>(dstBuffer.Data());
    for (int i = 0; i < width * height; ++i) {
        srcPixels[i] = 0xFF808080;
        dstPixels[i] = 0xFF808080;
    }

    blitter.MultiplyBlend(srcDesc, dstDesc);

    for (int i = 0; i < width * height; ++i) {
        XBYTE r = (dstPixels[i] >> 16) & 0xFF;
        XBYTE g = (dstPixels[i] >> 8) & 0xFF;
        XBYTE b = dstPixels[i] & 0xFF;
        EXPECT_NEAR(r, 0x40, 2) << "Red should be ~0x40";
        EXPECT_NEAR(g, 0x40, 2) << "Green should be ~0x40";
        EXPECT_NEAR(b, 0x40, 2) << "Blue should be ~0x40";
    }
}

