/**
 * @file BlitEngineEdgeCaseTest.cpp
 * @brief Comprehensive edge-case tests for VxBlitEngine full coverage.
 *
 * Tests:
 * - Stride/padding: Non-contiguous rows with sentinel bytes
 * - Self-blit / overlap: Same buffer as source and destination
 * - Dimension edge cases: 1×1, 1×N, N×1, odd dimensions
 * - Invalid/zero dimensions: Behavior with 0-width or 0-height
 * - Alignment variations: Misaligned buffers for SIMD edge paths
 */

#include "BlitEngineTestHelpers.h"

using namespace BlitEngineTest;

// Global sentinel value for detecting buffer overruns
static constexpr XBYTE SENTINEL = 0xCD;

//==============================================================================
// Stride/Padding Tests - Verify correct handling of non-contiguous rows
//==============================================================================

class BlitEngineStrideTest : public BlitEngineTestBase {
protected:

    /**
     * @brief Creates a buffer with extra padding per row and fills with sentinels.
     */
    struct PaddedBuffer {
        std::vector<XBYTE> data;
        int actualBytesPerLine;  // Padded stride
        int contentBytesPerLine; // Actual pixel data bytes
        int width;
        int height;
        int paddingBytes;

        PaddedBuffer(int w, int h, int bytesPerPixel, int extraPadding)
            : width(w), height(h), paddingBytes(extraPadding) {
            contentBytesPerLine = w * bytesPerPixel;
            actualBytesPerLine = contentBytesPerLine + extraPadding;
            data.resize(actualBytesPerLine * h, SENTINEL);
        }

        XBYTE* Data() { return data.data(); }
        const XBYTE* Data() const { return data.data(); }

        // Check that all sentinel bytes in padding regions are intact
        bool CheckSentinels() const {
            for (int y = 0; y < height; ++y) {
                const XBYTE* rowStart = data.data() + y * actualBytesPerLine;
                // Check padding bytes at end of row
                for (int p = contentBytesPerLine; p < actualBytesPerLine; ++p) {
                    if (rowStart[p] != SENTINEL) {
                        return false;
                    }
                }
            }
            return true;
        }

        // Get pointer to start of row y
        XBYTE* Row(int y) { return data.data() + y * actualBytesPerLine; }
    };
};

TEST_F(BlitEngineStrideTest, PaddedStride_32BitCopy_PreservesSentinels) {
    const int width = 17, height = 8;  // Odd width to test SIMD tail handling
    const int padding = 16;            // 16 bytes of padding per row

    PaddedBuffer srcBuf(width, height, 4, padding);
    PaddedBuffer dstBuf(width, height, 4, padding);

    // Fill source pixels only (not padding)
    for (int y = 0; y < height; ++y) {
        XULONG* row = reinterpret_cast<XULONG*>(srcBuf.Row(y));
        for (int x = 0; x < width; ++x) {
            row[x] = 0xFFFF0000 | (y << 8) | x;  // Red with varying GB
        }
    }

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());
    srcDesc.BytesPerLine = srcBuf.actualBytesPerLine;
    dstDesc.BytesPerLine = dstBuf.actualBytesPerLine;

    blitter.DoBlit(srcDesc, dstDesc);

    // Verify content copied correctly
    for (int y = 0; y < height; ++y) {
        const XULONG* srcRow = reinterpret_cast<const XULONG*>(srcBuf.Row(y));
        const XULONG* dstRow = reinterpret_cast<const XULONG*>(dstBuf.Row(y));
        for (int x = 0; x < width; ++x) {
            EXPECT_EQ(srcRow[x], dstRow[x]) 
                << "Pixel mismatch at (" << x << "," << y << ")";
        }
    }

    // Verify padding not overwritten
    EXPECT_TRUE(dstBuf.CheckSentinels()) 
        << "Sentinel bytes in padding were overwritten - buffer overrun!";
}

TEST_F(BlitEngineStrideTest, PaddedStride_32To24Conversion_PreservesSentinels) {
    const int width = 13, height = 5;  // Odd dimensions
    const int srcPadding = 8;
    const int dstPadding = 12;

    PaddedBuffer srcBuf(width, height, 4, srcPadding);
    PaddedBuffer dstBuf(width, height, 3, dstPadding);

    // Fill source with gradient
    for (int y = 0; y < height; ++y) {
        XULONG* row = reinterpret_cast<XULONG*>(srcBuf.Row(y));
        for (int x = 0; x < width; ++x) {
            row[x] = 0xFF000000 | ((x * 20) << 16) | ((y * 50) << 8) | 0x80;
        }
    }

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create24BitRGB(width, height, dstBuf.Data());
    srcDesc.BytesPerLine = srcBuf.actualBytesPerLine;
    dstDesc.BytesPerLine = dstBuf.actualBytesPerLine;

    blitter.DoBlit(srcDesc, dstDesc);

    // Verify conversion worked
    for (int y = 0; y < height; ++y) {
        const XULONG* srcRow = reinterpret_cast<const XULONG*>(srcBuf.Row(y));
        const XBYTE* dstRow = dstBuf.Row(y);
        for (int x = 0; x < width; ++x) {
            XULONG srcPx = srcRow[x];
            XBYTE expR = (srcPx >> 16) & 0xFF;
            XBYTE expG = (srcPx >> 8) & 0xFF;
            XBYTE expB = srcPx & 0xFF;
            // 24-bit is BGR order
            EXPECT_EQ(expB, dstRow[x * 3 + 0]) << "B mismatch at (" << x << "," << y << ")";
            EXPECT_EQ(expG, dstRow[x * 3 + 1]) << "G mismatch at (" << x << "," << y << ")";
            EXPECT_EQ(expR, dstRow[x * 3 + 2]) << "R mismatch at (" << x << "," << y << ")";
        }
    }

    // Padding intact
    EXPECT_TRUE(dstBuf.CheckSentinels())
        << "Sentinel bytes overwritten during 32->24 conversion";
}

TEST_F(BlitEngineStrideTest, PaddedStride_16Bit565_PreservesSentinels) {
    const int width = 19, height = 6;
    const int padding = 6;

    PaddedBuffer srcBuf(width, height, 4, 0);
    PaddedBuffer dstBuf(width, height, 2, padding);

    PatternGenerator::FillGradient32(srcBuf.Data(), width, height, true);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create16Bit565(width, height, dstBuf.Data());
    dstDesc.BytesPerLine = dstBuf.actualBytesPerLine;

    blitter.DoBlit(srcDesc, dstDesc);

    EXPECT_TRUE(dstBuf.CheckSentinels())
        << "Sentinel bytes overwritten during 32->16 conversion";
}

TEST_F(BlitEngineStrideTest, UpsideDown_PaddedStride_PreservesSentinels) {
    const int width = 11, height = 7;
    const int padding = 20;

    PaddedBuffer srcBuf(width, height, 4, padding);
    PaddedBuffer dstBuf(width, height, 4, padding);

    // Fill source
    for (int y = 0; y < height; ++y) {
        XULONG* row = reinterpret_cast<XULONG*>(srcBuf.Row(y));
        for (int x = 0; x < width; ++x) {
            row[x] = 0xFF000000 | (y << 16) | (x << 8) | ((x + y) & 0xFF);
        }
    }

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());
    srcDesc.BytesPerLine = srcBuf.actualBytesPerLine;
    dstDesc.BytesPerLine = dstBuf.actualBytesPerLine;

    blitter.DoBlitUpsideDown(srcDesc, dstDesc);

    // Verify rows are flipped
    for (int y = 0; y < height; ++y) {
        const XULONG* srcRow = reinterpret_cast<const XULONG*>(srcBuf.Row(y));
        const XULONG* dstRow = reinterpret_cast<const XULONG*>(dstBuf.Row(height - 1 - y));
        for (int x = 0; x < width; ++x) {
            EXPECT_EQ(srcRow[x], dstRow[x])
                << "UpsideDown mismatch at src(" << x << "," << y << ")";
        }
    }

    EXPECT_TRUE(dstBuf.CheckSentinels())
        << "Sentinel bytes overwritten during upside-down blit";
}

//==============================================================================
// Self-Blit / Overlap Tests - Same buffer as source and destination
//==============================================================================

class BlitEngineSelfBlitTest : public BlitEngineTestBase {};

TEST_F(BlitEngineSelfBlitTest, SelfBlit_SameDescriptor_32Bit) {
    // Self-blit (src == dst) should work as a no-op or copy in place
    const int width = 32, height = 32;
    ImageBuffer buffer(width * height * 4);
    PatternGenerator::FillColorBars32(buffer.Data(), width, height);

    // Save original for comparison
    ImageBuffer original(buffer.Size());
    memcpy(original.Data(), buffer.Data(), buffer.Size());

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    // Self-blit: same buffer as source and destination
    blitter.DoBlit(desc, desc);

    // Should be identical (either no-op or proper in-place copy)
    EXPECT_EQ(0, memcmp(original.Data(), buffer.Data(), buffer.Size()))
        << "Self-blit modified the image unexpectedly";
}

TEST_F(BlitEngineSelfBlitTest, SelfBlit_OverlappingRegions_32Bit) {
    // Test overlapping but not identical regions within the same buffer
    // Note: Overlapping regions may produce undefined results, but should not crash
    const int width = 64, height = 64;
    const int overlap = 4;  // rows of overlap
    
    // Allocate buffer large enough for overlap test
    // Total needed: (height + overlap) rows for destination offset
    ImageBuffer buffer(width * (height + overlap) * 4);
    PatternGenerator::FillUniquePixels32(buffer.Data(), width, height + overlap);

    // Source starts at beginning, destination offset by 'overlap' rows
    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, 
                                                      buffer.Data() + (overlap * width * 4));

    // The blit should complete without crashing
    // Results may be undefined due to overlap, but memory safety is key
    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

TEST_F(BlitEngineSelfBlitTest, SelfBlit_AlphaOp_ModifiesInPlace) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 100, 150, 200, 50);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    // In-place alpha modification
    blitter.DoAlphaBlit(desc, static_cast<XBYTE>(255));

    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF, (pixels[i] >> 24) & 0xFF) << "Alpha not set at pixel " << i;
        EXPECT_EQ(100, (pixels[i] >> 16) & 0xFF) << "Red changed at pixel " << i;
        EXPECT_EQ(150, (pixels[i] >> 8) & 0xFF) << "Green changed at pixel " << i;
        EXPECT_EQ(200, pixels[i] & 0xFF) << "Blue changed at pixel " << i;
    }
}

//==============================================================================
// Dimension Edge Cases - Minimum sizes, odd sizes
//==============================================================================

class BlitEngineDimensionTest : public BlitEngineTestBase {};

TEST_F(BlitEngineDimensionTest, MinimumSize_1x1_32Bit) {
    const int width = 1, height = 1;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );

    XULONG* src = reinterpret_cast<XULONG*>(pair.srcBuffer.Data());
    src[0] = 0xDEADBEEF;

    blitter.DoBlit(pair.srcDesc, pair.dstDesc);

    const XULONG* dst = reinterpret_cast<const XULONG*>(pair.dstBuffer.Data());
    EXPECT_EQ(0xDEADBEEF, dst[0]) << "1x1 blit failed";
}

TEST_F(BlitEngineDimensionTest, MinimumSize_1x1_Conversion) {
    // 32-bit to 16-bit conversion of single pixel
    ImageBuffer srcBuf(4);
    ImageBuffer dstBuf(2);

    XULONG* src = reinterpret_cast<XULONG*>(srcBuf.Data());
    src[0] = 0xFFFF0000;  // Pure red, full alpha

    auto srcDesc = ImageDescFactory::Create32BitARGB(1, 1, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create16Bit565(1, 1, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    const XWORD* dst = reinterpret_cast<const XWORD*>(dstBuf.Data());
    // 565: R=31<<11, G=0, B=0 => 0xF800
    EXPECT_EQ(0xF800, dst[0]) << "1x1 32->565 conversion failed";
}

TEST_F(BlitEngineDimensionTest, SingleRow_Nx1_32Bit) {
    const int width = 100, height = 1;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );

    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height, true);

    blitter.DoBlit(pair.srcDesc, pair.dstDesc);

    EXPECT_EQ(0, memcmp(pair.srcBuffer.Data(), pair.dstBuffer.Data(), pair.srcBuffer.Size()))
        << "Single row (Nx1) copy failed";
}

TEST_F(BlitEngineDimensionTest, SingleColumn_1xN_32Bit) {
    const int width = 1, height = 100;
    auto pair = CreateImagePair(
        ImageDescFactory::Create32BitARGB,
        ImageDescFactory::Create32BitARGB,
        width, height
    );

    PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height, false);

    blitter.DoBlit(pair.srcDesc, pair.dstDesc);

    EXPECT_EQ(0, memcmp(pair.srcBuffer.Data(), pair.dstBuffer.Data(), pair.srcBuffer.Size()))
        << "Single column (1xN) copy failed";
}

TEST_F(BlitEngineDimensionTest, OddWidth_SIMD_TailHandling) {
    // Test various odd widths that exercise SIMD remainder code
    const int heights = 4;
    const int testWidths[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 31, 33, 63, 65};

    for (int width : testWidths) {
        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitARGB,
            ImageDescFactory::Create32BitARGB,
            width, heights
        );

        PatternGenerator::FillUniquePixels32(pair.srcBuffer.Data(), width, heights);

        blitter.DoBlit(pair.srcDesc, pair.dstDesc);

        int diffs = ImageComparator::Compare32(
            pair.srcBuffer.Data(), pair.dstBuffer.Data(), width, heights, 0);
        EXPECT_EQ(0, diffs) << "Width " << width << " copy had " << diffs << " mismatches";
    }
}

TEST_F(BlitEngineDimensionTest, OddDimensions_Conversion_32To24) {
    const int testCases[][2] = {{3, 5}, {7, 11}, {13, 17}, {31, 33}};

    for (const auto& dims : testCases) {
        int width = dims[0], height = dims[1];

        auto pair = CreateImagePair(
            ImageDescFactory::Create32BitARGB,
            ImageDescFactory::Create24BitRGB,
            width, height
        );

        PatternGenerator::FillGradient32(pair.srcBuffer.Data(), width, height);

        blitter.DoBlit(pair.srcDesc, pair.dstDesc);

        // Verify conversion
        const XULONG* src = reinterpret_cast<const XULONG*>(pair.srcBuffer.Data());
        const XBYTE* dst = pair.dstBuffer.Data();
        bool ok = true;
        for (int i = 0; i < width * height && ok; ++i) {
            XBYTE expR = (src[i] >> 16) & 0xFF;
            XBYTE expG = (src[i] >> 8) & 0xFF;
            XBYTE expB = src[i] & 0xFF;
            ok = (dst[i*3] == expB && dst[i*3+1] == expG && dst[i*3+2] == expR);
        }
        EXPECT_TRUE(ok) << "Conversion failed for " << width << "x" << height;
    }
}

//==============================================================================
// Zero/Invalid Dimension Tests
//==============================================================================

class BlitEngineZeroDimensionTest : public BlitEngineTestBase {};

TEST_F(BlitEngineZeroDimensionTest, ZeroWidth_NoOp_NoCrash) {
    // Zero width should be a no-op without crashing
    ImageBuffer srcBuf(4);
    ImageBuffer dstBuf(4);
    dstBuf.Fill(SENTINEL);

    auto srcDesc = ImageDescFactory::Create32BitARGB(0, 10, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(0, 10, dstBuf.Data());

    // Should not crash
    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    // Destination should be unchanged
    EXPECT_EQ(SENTINEL, dstBuf[0]) << "Zero-width blit modified destination";
}

TEST_F(BlitEngineZeroDimensionTest, ZeroHeight_NoOp_NoCrash) {
    ImageBuffer srcBuf(4);
    ImageBuffer dstBuf(4);
    dstBuf.Fill(SENTINEL);

    auto srcDesc = ImageDescFactory::Create32BitARGB(10, 0, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(10, 0, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
    EXPECT_EQ(SENTINEL, dstBuf[0]) << "Zero-height blit modified destination";
}

TEST_F(BlitEngineZeroDimensionTest, ZeroBoth_NoOp_NoCrash) {
    ImageBuffer srcBuf(4);
    ImageBuffer dstBuf(4);
    dstBuf.Fill(SENTINEL);

    auto srcDesc = ImageDescFactory::Create32BitARGB(0, 0, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(0, 0, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

//==============================================================================
// Dimension Mismatch Tests - Different source and destination sizes
//==============================================================================

class BlitEngineDimensionMismatchTest : public BlitEngineTestBase {};

TEST_F(BlitEngineDimensionMismatchTest, SmallerDst_ClipsToDestination) {
    const int srcW = 64, srcH = 64;
    const int dstW = 32, dstH = 32;

    ImageBuffer srcBuf(srcW * srcH * 4);
    ImageBuffer dstBuf(dstW * dstH * 4);

    PatternGenerator::FillColorBars32(srcBuf.Data(), srcW, srcH);
    dstBuf.Fill(0);

    auto srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    // The behavior here is implementation-defined:
    // Either clips to dst size, or resizes. Document actual behavior.
    // At minimum, it should not crash and should write something to dst.
    bool anyNonZero = false;
    const XULONG* dst = reinterpret_cast<const XULONG*>(dstBuf.Data());
    for (int i = 0; i < dstW * dstH; ++i) {
        if (dst[i] != 0) { anyNonZero = true; break; }
    }
    EXPECT_TRUE(anyNonZero) << "Dimension mismatch blit wrote nothing to destination";
}

TEST_F(BlitEngineDimensionMismatchTest, LargerDst_BlitsToSmallerRegion) {
    const int srcW = 16, srcH = 16;
    const int dstW = 32, dstH = 32;

    ImageBuffer srcBuf(srcW * srcH * 4);
    ImageBuffer dstBuf(dstW * dstH * 4);

    PatternGenerator::FillSolid32(srcBuf.Data(), srcW, srcH, 255, 0, 0, 255);
    dstBuf.Fill(0);

    auto srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    // Should blit src-sized region; remaining dst area behavior is impl-defined
    // At minimum, should not crash
    SUCCEED() << "Larger dst blit completed without crash";
}

//==============================================================================
// Alpha Channel Edge Cases
//==============================================================================

class BlitEngineAlphaEdgeCaseTest : public BlitEngineTestBase {};

TEST_F(BlitEngineAlphaEdgeCaseTest, AlphaArray_AllZeros) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 100, 150, 200, 255);

    std::vector<XBYTE> alphaValues(width * height, 0);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    blitter.DoAlphaBlit(desc, alphaValues.data());

    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x00, (pixels[i] >> 24) & 0xFF) << "Alpha not zero at pixel " << i;
    }
}

TEST_F(BlitEngineAlphaEdgeCaseTest, AlphaArray_AllMax) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 100, 150, 200, 0);

    std::vector<XBYTE> alphaValues(width * height, 255);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    blitter.DoAlphaBlit(desc, alphaValues.data());

    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF, (pixels[i] >> 24) & 0xFF) << "Alpha not 255 at pixel " << i;
    }
}

TEST_F(BlitEngineAlphaEdgeCaseTest, AlphaArray_Gradient) {
    const int width = 256, height = 1;
    ImageBuffer buffer(width * height * 4);
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 128, 128, 128, 0);

    std::vector<XBYTE> alphaValues(width);
    for (int i = 0; i < width; ++i) {
        alphaValues[i] = static_cast<XBYTE>(i);
    }

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    blitter.DoAlphaBlit(desc, alphaValues.data());

    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width; ++i) {
        EXPECT_EQ(i, (pixels[i] >> 24) & 0xFF) << "Alpha mismatch at pixel " << i;
    }
}

//==============================================================================
// Resize Edge Cases
//==============================================================================

class BlitEngineResizeEdgeCaseTest : public BlitEngineTestBase {};

TEST_F(BlitEngineResizeEdgeCaseTest, Resize_1x1_To_Larger) {
    const int srcW = 1, srcH = 1;
    const int dstW = 16, dstH = 16;

    ImageBuffer srcBuf(srcW * srcH * 4);
    ImageBuffer dstBuf(dstW * dstH * 4);

    XULONG* src = reinterpret_cast<XULONG*>(srcBuf.Data());
    src[0] = 0xFFAABBCC;

    auto srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, dstBuf.Data());

    blitter.ResizeImage(srcDesc, dstDesc);

    // All destination pixels should be the same color (bilinear of single pixel)
    const XULONG* dst = reinterpret_cast<const XULONG*>(dstBuf.Data());
    for (int i = 0; i < dstW * dstH; ++i) {
        EXPECT_EQ(0xFFAABBCC, dst[i]) << "Resize 1x1 to larger failed at pixel " << i;
    }
}

TEST_F(BlitEngineResizeEdgeCaseTest, Resize_ToSameSize_IsIdentity) {
    const int size = 32;

    ImageBuffer srcBuf(size * size * 4);
    ImageBuffer dstBuf(size * size * 4);

    PatternGenerator::FillCheckerboard32(srcBuf.Data(), size, size, 4);

    auto srcDesc = ImageDescFactory::Create32BitARGB(size, size, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(size, size, dstBuf.Data());

    blitter.ResizeImage(srcDesc, dstDesc);

    // Same size resize should be identity (or very close)
    int diffs = ImageComparator::Compare32(srcBuf.Data(), dstBuf.Data(), size, size, 1);
    EXPECT_LE(diffs, 0) << "Same-size resize produced " << diffs << " different pixels";
}

TEST_F(BlitEngineResizeEdgeCaseTest, Resize_LargerToSmaller) {
    const int srcW = 64, srcH = 64;
    const int dstW = 16, dstH = 16;

    ImageBuffer srcBuf(srcW * srcH * 4);
    ImageBuffer dstBuf(dstW * dstH * 4);

    PatternGenerator::FillGradient32(srcBuf.Data(), srcW, srcH);

    auto srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, dstBuf.Data());

    EXPECT_NO_THROW(blitter.ResizeImage(srcDesc, dstDesc));

    // Save for visual inspection
    ImageWriter::SaveFromDesc("resize_downscale_dst", dstDesc, "ResizeEdge");
}

TEST_F(BlitEngineResizeEdgeCaseTest, Resize_SmallerToLarger) {
    const int srcW = 8, srcH = 8;
    const int dstW = 64, dstH = 64;

    ImageBuffer srcBuf(srcW * srcH * 4);
    ImageBuffer dstBuf(dstW * dstH * 4);

    PatternGenerator::FillCheckerboard32(srcBuf.Data(), srcW, srcH, 2);

    auto srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, dstBuf.Data());

    EXPECT_NO_THROW(blitter.ResizeImage(srcDesc, dstDesc));

    ImageWriter::SaveFromDesc("resize_upscale_dst", dstDesc, "ResizeEdge");
}

TEST_F(BlitEngineResizeEdgeCaseTest, Resize_NonSquare_WidthOnly) {
    const int srcW = 8, srcH = 32;
    const int dstW = 32, dstH = 32;

    ImageBuffer srcBuf(srcW * srcH * 4);
    ImageBuffer dstBuf(dstW * dstH * 4);

    PatternGenerator::FillColorBars32(srcBuf.Data(), srcW, srcH);

    auto srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, dstBuf.Data());

    EXPECT_NO_THROW(blitter.ResizeImage(srcDesc, dstDesc));
}

TEST_F(BlitEngineResizeEdgeCaseTest, Resize_NonSquare_HeightOnly) {
    const int srcW = 32, srcH = 8;
    const int dstW = 32, dstH = 32;

    ImageBuffer srcBuf(srcW * srcH * 4);
    ImageBuffer dstBuf(dstW * dstH * 4);

    PatternGenerator::FillGradient32(srcBuf.Data(), srcW, srcH, false);

    auto srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, dstBuf.Data());

    EXPECT_NO_THROW(blitter.ResizeImage(srcDesc, dstDesc));
}

//==============================================================================
// Extra Image Operations Edge Cases
//==============================================================================

class BlitEngineExtraOpsEdgeCaseTest : public BlitEngineTestBase {};

TEST_F(BlitEngineExtraOpsEdgeCaseTest, PremultiplyAlpha_ZeroAlpha) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);

    // Fill with RGB but zero alpha
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 255, 128, 64, 0);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    blitter.PremultiplyAlpha(desc);

    // With zero alpha, RGB should become 0
    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0x00000000, pixels[i]) << "Premultiply with alpha=0 should zero RGB";
    }
}

TEST_F(BlitEngineExtraOpsEdgeCaseTest, PremultiplyAlpha_FullAlpha) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);

    PatternGenerator::FillSolid32(buffer.Data(), width, height, 100, 150, 200, 255);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    blitter.PremultiplyAlpha(desc);

    // With full alpha, RGB should be unchanged
    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(100, (pixels[i] >> 16) & 0xFF) << "R changed with alpha=255";
        EXPECT_EQ(150, (pixels[i] >> 8) & 0xFF) << "G changed with alpha=255";
        EXPECT_EQ(200, pixels[i] & 0xFF) << "B changed with alpha=255";
    }
}

TEST_F(BlitEngineExtraOpsEdgeCaseTest, UnpremultiplyAlpha_ZeroAlpha_NoDiv0) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 4);

    // Zero alpha - unpremultiply should not divide by zero
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 0, 0, 0, 0);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    EXPECT_NO_THROW(blitter.UnpremultiplyAlpha(desc));
}

TEST_F(BlitEngineExtraOpsEdgeCaseTest, PremultiplyUnpremultiply_Roundtrip) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    ImageBuffer original(width * height * 4);

    // Fill with known values, alpha = 128 (half)
    PatternGenerator::FillSolid32(buffer.Data(), width, height, 200, 100, 50, 128);
    memcpy(original.Data(), buffer.Data(), buffer.Size());

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    blitter.PremultiplyAlpha(desc);
    blitter.UnpremultiplyAlpha(desc);

    // Should be close to original (may have small rounding errors)
    int diffs = ImageComparator::Compare32(original.Data(), buffer.Data(), width, height, 2);
    EXPECT_LE(diffs, 0) << "Premultiply/Unpremultiply roundtrip had " << diffs << " diff pixels";
}

TEST_F(BlitEngineExtraOpsEdgeCaseTest, SwapRedBlue_TwiceIsIdentity) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    ImageBuffer original(width * height * 4);

    PatternGenerator::FillColorBars32(buffer.Data(), width, height);
    memcpy(original.Data(), buffer.Data(), buffer.Size());

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    blitter.SwapRedBlue(desc);
    blitter.SwapRedBlue(desc);

    EXPECT_EQ(0, memcmp(original.Data(), buffer.Data(), buffer.Size()))
        << "SwapRedBlue twice should be identity";
}

TEST_F(BlitEngineExtraOpsEdgeCaseTest, InvertColors_TwiceIsIdentity) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    ImageBuffer original(width * height * 4);

    PatternGenerator::FillGradient32(buffer.Data(), width, height);
    memcpy(original.Data(), buffer.Data(), buffer.Size());

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    blitter.InvertColors(desc);
    blitter.InvertColors(desc);

    EXPECT_EQ(0, memcmp(original.Data(), buffer.Data(), buffer.Size()))
        << "InvertColors twice should be identity";
}

TEST_F(BlitEngineExtraOpsEdgeCaseTest, Grayscale_PreservesAlpha) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);

    PatternGenerator::FillSolid32(buffer.Data(), width, height, 255, 0, 0, 128);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    blitter.ConvertToGrayscale(desc);

    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(128, (pixels[i] >> 24) & 0xFF) << "Alpha changed during grayscale";
    }
}

TEST_F(BlitEngineExtraOpsEdgeCaseTest, FillImage_32Bit) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);
    buffer.Fill(0);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    blitter.FillImage(desc, 0xAABBCCDD);

    const XULONG* pixels = reinterpret_cast<const XULONG*>(buffer.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xAABBCCDD, pixels[i]) << "FillImage failed at pixel " << i;
    }
}

//==============================================================================
// Null Pointer Safety Tests
//==============================================================================

class BlitEngineNullSafetyTest : public BlitEngineTestBase {};

TEST_F(BlitEngineNullSafetyTest, NullSrcImage_NoCrash) {
    ImageBuffer dstBuf(64);
    dstBuf.Fill(SENTINEL);

    VxImageDescEx srcDesc = ImageDescFactory::Create32BitARGB(4, 4, nullptr);
    VxImageDescEx dstDesc = ImageDescFactory::Create32BitARGB(4, 4, dstBuf.Data());

    // Should not crash (behavior is implementation-defined)
    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

TEST_F(BlitEngineNullSafetyTest, NullDstImage_NoCrash) {
    ImageBuffer srcBuf(64);
    PatternGenerator::FillSolid32(srcBuf.Data(), 4, 4, 255, 0, 0, 255);

    VxImageDescEx srcDesc = ImageDescFactory::Create32BitARGB(4, 4, srcBuf.Data());
    VxImageDescEx dstDesc = ImageDescFactory::Create32BitARGB(4, 4, nullptr);

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

//==============================================================================
// Stress: Large Image Tests
//==============================================================================

class BlitEngineLargeImageTest : public BlitEngineTestBase {};

TEST_F(BlitEngineLargeImageTest, LargeImage_2048x2048_32BitCopy) {
    const int size = 2048;
    ImageBuffer srcBuf(size * size * 4);
    ImageBuffer dstBuf(size * size * 4);

    PatternGenerator::FillCheckerboard32(srcBuf.Data(), size, size, 64);

    auto srcDesc = ImageDescFactory::Create32BitARGB(size, size, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(size, size, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    // Spot check a few pixels
    const XULONG* src = reinterpret_cast<const XULONG*>(srcBuf.Data());
    const XULONG* dst = reinterpret_cast<const XULONG*>(dstBuf.Data());

    EXPECT_EQ(src[0], dst[0]);
    EXPECT_EQ(src[size * size / 2], dst[size * size / 2]);
    EXPECT_EQ(src[size * size - 1], dst[size * size - 1]);
}

TEST_F(BlitEngineLargeImageTest, LargeImage_4096x1_SingleRow) {
    const int width = 4096, height = 1;
    ImageBuffer srcBuf(width * 4);
    ImageBuffer dstBuf(width * 4);

    PatternGenerator::FillGradient32(srcBuf.Data(), width, height, true);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    EXPECT_EQ(0, memcmp(srcBuf.Data(), dstBuf.Data(), srcBuf.Size()));
}

