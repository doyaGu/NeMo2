/**
 * @file BlitEngineUpsideDownTest.cpp
 * @brief Tests for VxBlitEngine vertical flip (upside down) blitting.
 *
 * Tests:
 * - DoBlitUpsideDown basic functionality
 * - Vertical flip correctness
 * - Various pixel formats
 * - Edge cases
 */

#include "BlitEngineTestHelpers.h"

using namespace BlitEngineTest;

class UpsideDownBlitTest : public BlitEngineTestBase {
protected:
    // Verify that rows are flipped
    bool VerifyFlipped32(const XBYTE* src, const XBYTE* dst, int width, int height) {
        const XDWORD* srcPixels = reinterpret_cast<const XDWORD*>(src);
        const XDWORD* dstPixels = reinterpret_cast<const XDWORD*>(dst);
        
        for (int y = 0; y < height; ++y) {
            int flippedY = height - 1 - y;
            for (int x = 0; x < width; ++x) {
                if (srcPixels[y * width + x] != dstPixels[flippedY * width + x]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    bool VerifyFlipped24(const XBYTE* src, const XBYTE* dst, int width, int height) {
        int rowBytes = width * 3;
        for (int y = 0; y < height; ++y) {
            int flippedY = height - 1 - y;
            if (memcmp(src + y * rowBytes, dst + flippedY * rowBytes, rowBytes) != 0) {
                return false;
            }
        }
        return true;
    }
    
    bool VerifyFlipped16(const XBYTE* src, const XBYTE* dst, int width, int height) {
        const XWORD* srcPixels = reinterpret_cast<const XWORD*>(src);
        const XWORD* dstPixels = reinterpret_cast<const XWORD*>(dst);
        
        for (int y = 0; y < height; ++y) {
            int flippedY = height - 1 - y;
            for (int x = 0; x < width; ++x) {
                if (srcPixels[y * width + x] != dstPixels[flippedY * width + x]) {
                    return false;
                }
            }
        }
        return true;
    }
};

//==============================================================================
// Basic 32-bit Upside Down Tests
//==============================================================================

TEST_F(UpsideDownBlitTest, Basic32_FlipsCorrectly) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    // Create distinct pattern per row
    XDWORD* srcPixels = reinterpret_cast<XDWORD*>(pair.srcBuffer.Data());
    for (int y = 0; y < height; ++y) {
        XDWORD rowColor = 0xFF000000 | ((y * 30) << 16);
        for (int x = 0; x < width; ++x) {
            srcPixels[y * width + x] = rowColor;
        }
    }
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped32(pair.srcBuffer.Data(), pair.dstBuffer.Data(), 
                                 width, height));
}

TEST_F(UpsideDownBlitTest, Basic32_ColorBars) {
    const int width = 32, height = 32;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    PatternGenerator::FillColorBars32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped32(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

TEST_F(UpsideDownBlitTest, Basic32_Gradient) {
    const int width = 16, height = 16;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped32(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

TEST_F(UpsideDownBlitTest, Basic32_UniquePixels) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    PatternGenerator::FillUniquePixels32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped32(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

//==============================================================================
// 24-bit Upside Down Tests
//==============================================================================

TEST_F(UpsideDownBlitTest, Basic24_FlipsCorrectly) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create24BitRGB,
        ImageDescFactory::Create24BitRGB,
        width, height
    );
    
    // Create distinct pattern per row
    for (int y = 0; y < height; ++y) {
        XBYTE rowValue = static_cast<XBYTE>(y * 30);
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            pair.srcBuffer[idx + 0] = rowValue;
            pair.srcBuffer[idx + 1] = static_cast<XBYTE>(255 - rowValue);
            pair.srcBuffer[idx + 2] = 128;
        }
    }
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped24(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

TEST_F(UpsideDownBlitTest, Basic24_SolidColor) {
    const int width = 16, height = 16;
    auto pair = CreateImagePair(
        ImageDescFactory::Create24BitRGB,
        ImageDescFactory::Create24BitRGB,
        width, height
    );
    
    PatternGenerator::FillSolid24(pair.srcBuffer.Data(), width, height,
                                   0xAA, 0xBB, 0xCC);
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    // Solid color should be same after flip
    EXPECT_EQ(0, memcmp(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                        pair.srcBuffer.Size()));
}

//==============================================================================
// 16-bit Upside Down Tests
//==============================================================================

TEST_F(UpsideDownBlitTest, Basic565_FlipsCorrectly) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create16Bit565,
        ImageDescFactory::Create16Bit565,
        width, height
    );
    
    // Create row patterns
    XWORD* srcPixels = reinterpret_cast<XWORD*>(pair.srcBuffer.Data());
    for (int y = 0; y < height; ++y) {
        XWORD rowValue = static_cast<XWORD>(y << 11); // Red varies by row
        for (int x = 0; x < width; ++x) {
            srcPixels[y * width + x] = rowValue;
        }
    }
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped16(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

TEST_F(UpsideDownBlitTest, Basic1555_FlipsCorrectly) {
    const int width = 8, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create16Bit1555,
        ImageDescFactory::Create16Bit1555,
        width, height
    );
    
    XWORD* srcPixels = reinterpret_cast<XWORD*>(pair.srcBuffer.Data());
    for (int y = 0; y < height; ++y) {
        XWORD rowValue = static_cast<XWORD>((y % 2) << 15) | (y << 10);
        for (int x = 0; x < width; ++x) {
            srcPixels[y * width + x] = rowValue;
        }
    }
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped16(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

//==============================================================================
// Cross-Format Upside Down Tests
//==============================================================================

TEST_F(UpsideDownBlitTest, CrossFormat_32To24_FlipsAndConverts) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height * 4);
    ImageBuffer dstBuffer(width * height * 3);
    
    auto src = ImageDescFactory::Create32BitARGB(width, height, srcBuffer.Data());
    auto dst = ImageDescFactory::Create24BitRGB(width, height, dstBuffer.Data());
    
    // Row pattern in 32-bit
    XDWORD* srcPixels = reinterpret_cast<XDWORD*>(srcBuffer.Data());
    for (int y = 0; y < height; ++y) {
        XDWORD rowColor = 0xFF000000 | (y * 30 << 16) | (y * 20 << 8) | (y * 10);
        for (int x = 0; x < width; ++x) {
            srcPixels[y * width + x] = rowColor;
        }
    }
    
    blitter.DoBlitUpsideDown(src, dst);
    
    // Verify flip by checking row 0 of dst matches row height-1 of src
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        XDWORD srcColor = srcPixels[srcY * width];
        XBYTE expectedR = (srcColor >> 16) & 0xFF;
        XBYTE expectedG = (srcColor >> 8) & 0xFF;
        XBYTE expectedB = srcColor & 0xFF;
        
        int dstIdx = y * width * 3;
        EXPECT_EQ(expectedB, dstBuffer[dstIdx + 0]) << "Blue row " << y;
        EXPECT_EQ(expectedG, dstBuffer[dstIdx + 1]) << "Green row " << y;
        EXPECT_EQ(expectedR, dstBuffer[dstIdx + 2]) << "Red row " << y;
    }
}

TEST_F(UpsideDownBlitTest, CrossFormat_24To32_FlipsAndConverts) {
    const int width = 8, height = 8;
    ImageBuffer srcBuffer(width * height * 3);
    ImageBuffer dstBuffer(width * height * 4);
    
    auto src = ImageDescFactory::Create24BitRGB(width, height, srcBuffer.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    // Row pattern in 24-bit
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            srcBuffer[idx + 0] = static_cast<XBYTE>(y * 10);       // B
            srcBuffer[idx + 1] = static_cast<XBYTE>(y * 20);       // G
            srcBuffer[idx + 2] = static_cast<XBYTE>(y * 30);       // R
        }
    }
    
    blitter.DoBlitUpsideDown(src, dst);
    
    // Verify flip
    const XDWORD* dstPixels = reinterpret_cast<const XDWORD*>(dstBuffer.Data());
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        XBYTE expectedR = static_cast<XBYTE>(srcY * 30);
        XBYTE expectedG = static_cast<XBYTE>(srcY * 20);
        XBYTE expectedB = static_cast<XBYTE>(srcY * 10);
        
        XDWORD pixel = dstPixels[y * width];
        EXPECT_EQ(expectedR, (pixel >> 16) & 0xFF) << "Red row " << y;
        EXPECT_EQ(expectedG, (pixel >> 8) & 0xFF) << "Green row " << y;
        EXPECT_EQ(expectedB, pixel & 0xFF) << "Blue row " << y;
    }
}

//==============================================================================
// Edge Cases
//==============================================================================

TEST_F(UpsideDownBlitTest, SingleRow_NoFlip) {
    const int width = 16, height = 1;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    PatternGenerator::FillColorBars32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    // Single row should be identical
    EXPECT_EQ(0, memcmp(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                        pair.srcBuffer.Size()));
}

TEST_F(UpsideDownBlitTest, TwoRows_Swapped) {
    const int width = 8, height = 2;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    XDWORD* srcPixels = reinterpret_cast<XDWORD*>(pair.srcBuffer.Data());
    // Row 0: all red
    for (int x = 0; x < width; ++x) {
        srcPixels[x] = 0xFFFF0000;
    }
    // Row 1: all blue
    for (int x = 0; x < width; ++x) {
        srcPixels[width + x] = 0xFF0000FF;
    }
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    const XDWORD* dstPixels = reinterpret_cast<const XDWORD*>(pair.dstBuffer.Data());
    // Row 0 should now be blue
    EXPECT_EQ(0xFF0000FF, dstPixels[0]);
    // Row 1 should now be red
    EXPECT_EQ(0xFFFF0000, dstPixels[width]);
}

TEST_F(UpsideDownBlitTest, SinglePixel) {
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        1, 1
    );
    
    *reinterpret_cast<XDWORD*>(pair.srcBuffer.Data()) = 0xFFAABBCC;
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_EQ(0xFFAABBCC, *reinterpret_cast<XDWORD*>(pair.dstBuffer.Data()));
}

TEST_F(UpsideDownBlitTest, LargeImage) {
    const int width = 256, height = 256;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped32(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

TEST_F(UpsideDownBlitTest, NonSquare_WideImage) {
    const int width = 64, height = 8;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped32(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

TEST_F(UpsideDownBlitTest, NonSquare_TallImage) {
    const int width = 8, height = 64;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );
    
    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height);
    
    blitter.DoBlitUpsideDown(pair.srcDesc, pair.dstDesc);
    
    EXPECT_TRUE(VerifyFlipped32(pair.srcBuffer.Data(), pair.dstBuffer.Data(),
                                 width, height));
}

//==============================================================================
// Double Flip (Should Return to Original)
//==============================================================================

TEST_F(UpsideDownBlitTest, DoubleFlip_ReturnsToOriginal) {
    const int width = 16, height = 16;
    ImageBuffer srcBuffer(width * height * 4);
    ImageBuffer tempBuffer(width * height * 4);
    ImageBuffer dstBuffer(width * height * 4);
    
    auto src = ImageDescFactory::Create32BitARGB(width, height, srcBuffer.Data());
    auto temp = ImageDescFactory::Create32BitARGB(width, height, tempBuffer.Data());
    auto dst = ImageDescFactory::Create32BitARGB(width, height, dstBuffer.Data());
    
    PatternGenerator::FillUniquePixels32(srcBuffer.Data(), width, height);
    
    // Flip twice
    blitter.DoBlitUpsideDown(src, temp);
    blitter.DoBlitUpsideDown(temp, dst);
    
    // Should be back to original
    EXPECT_EQ(0, memcmp(srcBuffer.Data(), dstBuffer.Data(), srcBuffer.Size()));
}
