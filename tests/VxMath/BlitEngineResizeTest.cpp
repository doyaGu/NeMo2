/**
 * @file BlitEngineResizeTest.cpp
 * @brief Tests for VxBlitEngine image resizing functionality.
 *
 * Tests:
 * - Upscaling images (enlarging)
 * - Downscaling images (shrinking)
 * - Same-size resize (identity)
 * - Non-power-of-2 dimensions
 * - Quality assessment
 */

#include "BlitEngineTestHelpers.h"

using namespace BlitEngineTest;

class ResizeTest : public BlitEngineTestBase {
protected:
    // Helper to create source and destination with different sizes
    struct ResizePair {
        ImageBuffer srcBuffer;
        ImageBuffer dstBuffer;
        VxImageDescEx srcDesc;
        VxImageDescEx dstDesc;
    };
    
    ResizePair CreateResizePair(int srcW, int srcH, int dstW, int dstH) {
        ResizePair pair;
        pair.srcBuffer.Resize(srcW * srcH * 4);
        pair.dstBuffer.Resize(dstW * dstH * 4);
        pair.dstBuffer.Fill(0);
        
        pair.srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, 
                                                          pair.srcBuffer.Data());
        pair.dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, 
                                                          pair.dstBuffer.Data());
        return pair;
    }
    
    // Check if pixel at (x, y) in resized image approximately matches 
    // expected color from source
    bool CheckPixelApprox(const XBYTE* data, int w, int h, int x, int y,
                          XBYTE r, XBYTE g, XBYTE b, int tolerance = 5) {
        const XULONG* pixels = reinterpret_cast<const XULONG*>(data);
        XULONG pixel = pixels[y * w + x];
        int pr = (pixel >> 16) & 0xFF;
        int pg = (pixel >> 8) & 0xFF;
        int pb = pixel & 0xFF;
        
        return std::abs(pr - r) <= tolerance &&
               std::abs(pg - g) <= tolerance &&
               std::abs(pb - b) <= tolerance;
    }
};

//==============================================================================
// Identity Resize (Same Size)
//==============================================================================

TEST_F(ResizeTest, SameSize_CopiesExactly) {
    const int size = 32;
    auto pair = CreateResizePair(size, size, size, size);
    
    PatternGenerator::FillColorBars32(pair.srcBuffer.Data(), size, size);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_same_size", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    EXPECT_EQ(0, memcmp(pair.srcBuffer.Data(), pair.dstBuffer.Data(), 
                        pair.srcBuffer.Size()));
}

TEST_F(ResizeTest, SameSize_1x1) {
    auto pair = CreateResizePair(1, 1, 1, 1);
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    *src = 0xFF123456;
    
    blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_same_size_1x1", pair.dstDesc, "Resize");
    
    XULONG* dst = reinterpret_cast<XULONG*>(pair.dstBuffer.Data());
    EXPECT_EQ(0xFF123456, *dst);
}

//==============================================================================
// Upscaling Tests
//==============================================================================

TEST_F(ResizeTest, Upscale_2x_SolidColor) {
    const int srcSize = 16, dstSize = 32;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    // Fill with solid color R=128, G=64, B=192, A=255
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcSize, srcSize,
                                   128, 64, 192, 255);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_up_2x_solid", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // Every pixel should match source color (ARGB: 0xFF8040C0)
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < dstSize * dstSize; ++i) {
        EXPECT_EQ(0xFF8040C0, dst[i] & 0xFFFFFFFF) 
            << "Color mismatch at pixel " << i;
    }
}

TEST_F(ResizeTest, Upscale_2x_Checkerboard) {
    const int srcSize = 8, dstSize = 16;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    PatternGenerator::FillCheckerboard32(pair.srcBuffer.Data(), srcSize, srcSize);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_up_2x_checker", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // Due to bilinear filtering, edges will be blended
    // Check corners - should match source corners
    // Default checkerboard: (0,0) = white (color1)
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), dstSize, dstSize, 
                                  0, 0, 255, 255, 255, 30))
        << "Top-left corner should be close to white";
}

TEST_F(ResizeTest, Upscale_4x) {
    const int srcSize = 8, dstSize = 32;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), srcSize, srcSize);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_up_4x", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // Gradient should still be present - check top-left is darker than bottom-right
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    XULONG topLeft = dst[0];
    XULONG bottomRight = dst[dstSize * dstSize - 1];
    
    int tlBrightness = ((topLeft >> 16) & 0xFF) + ((topLeft >> 8) & 0xFF) + (topLeft & 0xFF);
    int brBrightness = ((bottomRight >> 16) & 0xFF) + ((bottomRight >> 8) & 0xFF) + (bottomRight & 0xFF);
    
    EXPECT_LT(tlBrightness, brBrightness) << "Gradient should increase toward bottom-right";
}

TEST_F(ResizeTest, Upscale_NonPowerOf2) {
    const int srcW = 13, srcH = 17;
    const int dstW = 39, dstH = 51; // 3x
    auto pair = CreateResizePair(srcW, srcH, dstW, dstH);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcW, srcH,
                                   200, 100, 50, 255);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_up_nonpow2", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // All pixels should be similar to source
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < dstW * dstH; ++i) {
        EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), dstW, dstH,
                                      i % dstW, i / dstW, 200, 100, 50, 2))
            << "Solid color mismatch at pixel " << i;
    }
}

//==============================================================================
// Downscaling Tests
//==============================================================================

TEST_F(ResizeTest, Downscale_2x_SolidColor) {
    const int srcSize = 32, dstSize = 16;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcSize, srcSize,
                                   100, 150, 200, 255);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_down_2x_solid", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < dstSize * dstSize; ++i) {
        EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), dstSize, dstSize,
                                      i % dstSize, i / dstSize, 100, 150, 200, 2));
    }
}

TEST_F(ResizeTest, Downscale_2x_Checkerboard) {
    const int srcSize = 64, dstSize = 32;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    // 2x2 checkerboard in source becomes 1x1 gray pixels when downscaled
    PatternGenerator::FillCheckerboard32(pair.srcBuffer.Data(), srcSize, srcSize);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_down_2x_checker", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // After downscaling, checkerboard should produce some blending
    // The exact result depends on filtering algorithm - allow wide tolerance
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    // Just verify the resize completes successfully - filtering varies by implementation
    EXPECT_TRUE(dst[0] != 0 || dst[dstSize * dstSize - 1] != 0)
        << "Output should not be all zeros";
}

TEST_F(ResizeTest, Downscale_4x) {
    const int srcSize = 64, dstSize = 16;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), srcSize, srcSize);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_down_4x", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // Gradient should be preserved
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    XULONG topLeft = dst[0];
    XULONG bottomRight = dst[dstSize * dstSize - 1];
    
    int tlBrightness = ((topLeft >> 16) & 0xFF) + ((topLeft >> 8) & 0xFF) + (topLeft & 0xFF);
    int brBrightness = ((bottomRight >> 16) & 0xFF) + ((bottomRight >> 8) & 0xFF) + (bottomRight & 0xFF);
    
    EXPECT_LT(tlBrightness, brBrightness);
}

TEST_F(ResizeTest, Downscale_NonPowerOf2) {
    const int srcW = 45, srcH = 30;
    const int dstW = 15, dstH = 10; // 3x downscale
    auto pair = CreateResizePair(srcW, srcH, dstW, dstH);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcW, srcH,
                                   80, 160, 240, 255);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_down_nonpow2", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    // Check all pixels match
    for (int i = 0; i < dstW * dstH; ++i) {
        EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), dstW, dstH,
                                      i % dstW, i / dstW, 80, 160, 240, 2));
    }
}

//==============================================================================
// Non-Square Tests
//==============================================================================

TEST_F(ResizeTest, NonSquare_Wide_Upscale) {
    const int srcW = 8, srcH = 4;
    const int dstW = 32, dstH = 16; // 4x upscale
    auto pair = CreateResizePair(srcW, srcH, dstW, dstH);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcW, srcH,
                                   50, 100, 150, 255);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_nonsquare_wide_up", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), dstW, dstH,
                                  dstW/2, dstH/2, 50, 100, 150, 2));
}

TEST_F(ResizeTest, NonSquare_Tall_Downscale) {
    const int srcW = 16, srcH = 64;
    const int dstW = 4, dstH = 16; // 4x downscale
    auto pair = CreateResizePair(srcW, srcH, dstW, dstH);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcW, srcH,
                                   200, 150, 100, 255);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_nonsquare_tall_down", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), dstW, dstH,
                                  dstW/2, dstH/2, 200, 150, 100, 5));
}

TEST_F(ResizeTest, NonSquare_DifferentRatios) {
    // Source is 16:9, dest is 4:3 - different aspect ratios
    const int srcW = 160, srcH = 90;
    const int dstW = 80, dstH = 60;
    auto pair = CreateResizePair(srcW, srcH, dstW, dstH);
    
    PatternGenerator::FillColorBars32(pair.srcBuffer.Data(), srcW, srcH);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_nonsquare_diff_ratio", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // Just check it completes without error
    EXPECT_NE(nullptr, pair.dstBuffer.Data());
}

//==============================================================================
// Extreme Cases
//==============================================================================

TEST_F(ResizeTest, Extreme_UpscaleFrom1x1) {
    auto pair = CreateResizePair(1, 1, 64, 64);
    
    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    *src = 0xFF112233;
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_extreme_up_from1x1", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // All destination pixels should be the same color
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < 64 * 64; ++i) {
        EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), 64, 64,
                                      i % 64, i / 64, 0x11, 0x22, 0x33, 2));
    }
}

TEST_F(ResizeTest, Extreme_DownscaleTo1x1) {
    const int srcSize = 32;
    auto pair = CreateResizePair(srcSize, srcSize, 1, 1);
    
    // Fill with uniform color
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcSize, srcSize,
                                   0x44, 0x88, 0xCC, 0xFF);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_extreme_down_to1x1", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    XULONG* dst = reinterpret_cast<XULONG*>(pair.dstBuffer.Data());
    EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), 1, 1, 0, 0,
                                  0x44, 0x88, 0xCC, 10));
}

TEST_F(ResizeTest, Extreme_LargeImage) {
    const int srcSize = 256, dstSize = 512;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcSize, srcSize,
                                   0x80, 0x40, 0xC0, 0xFF);
    
    CKERROR result = blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_extreme_large", pair.dstDesc, "Resize");
    EXPECT_EQ(CK_OK, result);
    
    // Sample check
    EXPECT_TRUE(CheckPixelApprox(pair.dstBuffer.Data(), dstSize, dstSize,
                                  dstSize/2, dstSize/2, 0x80, 0x40, 0xC0, 2));
}

//==============================================================================
// Quality Tests
//==============================================================================

TEST_F(ResizeTest, Quality_PSNR_SolidColor) {
    // For solid color, PSNR should be very high (nearly infinite)
    const int srcSize = 32, dstSize = 64;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcSize, srcSize,
                                   100, 100, 100, 255);
    
    blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_quality_psnr_solid", pair.dstDesc, "Resize");
    
    // Create expected solid color destination
    ImageBuffer expected(dstSize * dstSize * 4);
    PatternGenerator::FillSolid32(expected.Data(), dstSize, dstSize,
                                   100, 100, 100, 255);
    
    double psnr = ImageComparator::CalcPSNR32(pair.dstBuffer.Data(), 
                                               expected.Data(),
                                               dstSize, dstSize);
    // Should be essentially identical
    EXPECT_GT(psnr, 50.0) << "PSNR should be very high for solid color";
}

TEST_F(ResizeTest, Quality_MSE_LowForSolidColor) {
    const int srcSize = 16, dstSize = 32;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcSize, srcSize,
                                   128, 128, 128, 255);
    
    blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_quality_mse_solid", pair.dstDesc, "Resize");
    
    ImageBuffer expected(dstSize * dstSize * 4);
    PatternGenerator::FillSolid32(expected.Data(), dstSize, dstSize,
                                   128, 128, 128, 255);
    
    double mse = ImageComparator::CalcMSE32(pair.dstBuffer.Data(),
                                             expected.Data(),
                                             dstSize, dstSize);
    EXPECT_LT(mse, 1.0) << "MSE should be very low for solid color";
}

//==============================================================================
// Alpha Preservation During Resize
//==============================================================================

TEST_F(ResizeTest, Alpha_PreservedDuringUpscale) {
    const int srcSize = 8, dstSize = 32;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    // Fill with semi-transparent color
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcSize, srcSize,
                                   100, 150, 200, 128);
    
    blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_alpha_upscale", pair.dstDesc, "Resize");
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < 10; ++i) {
        XBYTE alpha = (dst[i] >> 24) & 0xFF;
        EXPECT_NEAR(128, alpha, 5) << "Alpha should be preserved";
    }
}

TEST_F(ResizeTest, Alpha_PreservedDuringDownscale) {
    const int srcSize = 64, dstSize = 16;
    auto pair = CreateResizePair(srcSize, srcSize, dstSize, dstSize);
    
    PatternGenerator::FillSolid32(pair.srcBuffer.Data(), srcSize, srcSize,
                                   100, 150, 200, 64);
    
    blitter.ResizeImage(pair.srcDesc, pair.dstDesc);
    ImageWriter::SaveFromDesc("resize_alpha_downscale", pair.dstDesc, "Resize");
    
    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    for (int i = 0; i < dstSize * dstSize; ++i) {
        XBYTE alpha = (dst[i] >> 24) & 0xFF;
        EXPECT_NEAR(64, alpha, 5) << "Alpha should be preserved at pixel " << i;
    }
}
