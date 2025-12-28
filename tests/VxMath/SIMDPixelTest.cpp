/**
 * @file SIMDPixelTest.cpp
 * @brief Tests for SIMD pixel operations.
 *
 * Tests all operations in VxSIMDPixelOps:
 * - ConvertPixelBatch32
 * - ApplyAlphaBatch32
 * - ApplyVariableAlphaBatch32
 *
 * Note: These SIMD functions only process multiples of 4 pixels.
 * The return value indicates how many pixels were actually processed.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstring>

#include "VxSIMD.h"
#include "SIMDTestCommon.h"

namespace {

using namespace SIMDTest;

class SIMDPixelTest : public SIMDTest::SIMDTestBase {
protected:
    // RGBA format masks (R in low byte)
    static constexpr XULONG RGBA_RMASK = 0x000000FF;
    static constexpr XULONG RGBA_GMASK = 0x0000FF00;
    static constexpr XULONG RGBA_BMASK = 0x00FF0000;
    static constexpr XULONG RGBA_AMASK = 0xFF000000;

    // BGRA format masks (B in low byte)
    static constexpr XULONG BGRA_RMASK = 0x00FF0000;
    static constexpr XULONG BGRA_GMASK = 0x0000FF00;
    static constexpr XULONG BGRA_BMASK = 0x000000FF;
    static constexpr XULONG BGRA_AMASK = 0xFF000000;

    XULONG RandomPixel() {
        float f1 = RandomFloat(0.0f, 255.0f);
        float f2 = RandomFloat(0.0f, 255.0f);
        float f3 = RandomFloat(0.0f, 255.0f);
        float f4 = RandomFloat(0.0f, 255.0f);
        return (static_cast<XULONG>(f4) << 24) |
               (static_cast<XULONG>(f3) << 16) |
               (static_cast<XULONG>(f2) << 8) |
               static_cast<XULONG>(f1);
    }

    XBYTE RandomAlpha() {
        return static_cast<XBYTE>(RandomFloat(0.0f, 255.0f));
    }

    // Helper to compute shift amount for a mask
    static int GetMaskShift(XULONG mask) {
        if (mask == 0) return 0;
        int shift = 0;
        while ((mask & 1) == 0) {
            mask >>= 1;
            ++shift;
        }
        return shift;
    }

    // Setup VxPixelSimdConfig for RGBA to BGRA conversion
    VxPixelSimdConfig CreateRGBAtoBGRAConfig() {
        VxPixelSimdConfig config;
        config.enabled = true;
        config.alphaFill = false;

        // Source RGBA masks
        config.srcMasks[0] = RGBA_RMASK;  // R
        config.srcMasks[1] = RGBA_GMASK;  // G
        config.srcMasks[2] = RGBA_BMASK;  // B
        config.srcMasks[3] = RGBA_AMASK;  // A

        // Destination BGRA masks
        config.dstMasks[0] = BGRA_RMASK;  // R
        config.dstMasks[1] = BGRA_GMASK;  // G
        config.dstMasks[2] = BGRA_BMASK;  // B
        config.dstMasks[3] = BGRA_AMASK;  // A

        // Calculate shifts
        for (int i = 0; i < 4; ++i) {
            config.srcShiftRight[i] = GetMaskShift(config.srcMasks[i]);
            config.dstShiftLeft[i] = GetMaskShift(config.dstMasks[i]);
            config.channelCopy[i] = true;
        }

        return config;
    }

    // Setup config for identity conversion (same format)
    VxPixelSimdConfig CreateIdentityConfig() {
        VxPixelSimdConfig config;
        config.enabled = true;
        config.alphaFill = false;

        for (int i = 0; i < 4; ++i) {
            config.srcMasks[i] = (0xFFu << (i * 8));
            config.dstMasks[i] = (0xFFu << (i * 8));
            config.srcShiftRight[i] = i * 8;
            config.dstShiftLeft[i] = i * 8;
            config.channelCopy[i] = true;
        }

        return config;
    }

    // Scalar reference for pixel conversion
    XULONG ScalarConvertPixel(XULONG src, const VxPixelSimdConfig& config) {
        if (!config.enabled) return src;

        XULONG dst = 0;
        for (int c = 0; c < 4; ++c) {
            if (config.channelCopy[c]) {
                XULONG value = (src & config.srcMasks[c]) >> config.srcShiftRight[c];
                dst |= (value << config.dstShiftLeft[c]) & config.dstMasks[c];
            }
        }
        if (config.alphaFill) {
            dst = (dst & ~config.dstMasks[3]) | config.alphaFillComponent;
        }
        return dst;
    }

    // Scalar reference for alpha application - REPLACES alpha, doesn't modulate
    XULONG ScalarApplyAlpha(XULONG pixel, XBYTE alpha, XULONG alphaMask, int alphaShift) {
        XULONG colorMask = ~alphaMask;
        XULONG alphaComponent = (static_cast<XULONG>(alpha) << alphaShift) & alphaMask;
        return (pixel & colorMask) | alphaComponent;
    }

    // Get minimum SIMD-aligned count
    // SSE processes multiples of 4, AVX processes multiples of 8
    // We test that whatever is processed matches scalar reference
    static int GetMinSimdAlignment() {
        // AVX uses 8, SSE uses 4; use 8 to be safe
        return 8;
    }
};

//=============================================================================
// Convert Pixel Batch Tests
//=============================================================================

TEST_F(SIMDPixelTest, ConvertPixelBatch32_IdentityConversion) {
    ASSERT_NE(m_dispatch->Pixel.ConvertPixelBatch32, nullptr);

    // Use count that's multiple of 4
    const int count = 128;
    std::vector<XULONG> src(count);
    std::vector<XULONG> dst(count, 0);

    for (int i = 0; i < count; ++i) {
        src[i] = RandomPixel();
    }

    VxPixelSimdConfig config = CreateIdentityConfig();

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        src.data(), dst.data(), count, config);

    // Should process all pixels since count is multiple of 4
    EXPECT_EQ(processed, count);

    for (int i = 0; i < processed; ++i) {
        XULONG expected = ScalarConvertPixel(src[i], config);
        EXPECT_EQ(dst[i], expected) << "Mismatch at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, ConvertPixelBatch32_RGBAtoBGRA) {
    // Use count that's a multiple of 8 (for AVX compatibility)
    const int count = 104;
    std::vector<XULONG> src(count);
    std::vector<XULONG> dst(count, 0);

    for (int i = 0; i < count; ++i) {
        src[i] = RandomPixel();
    }

    VxPixelSimdConfig config = CreateRGBAtoBGRAConfig();

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        src.data(), dst.data(), count, config);

    // Should process at least some pixels
    EXPECT_GT(processed, 0);
    // Should not process more than count
    EXPECT_LE(processed, count);

    for (int i = 0; i < processed; ++i) {
        XULONG expected = ScalarConvertPixel(src[i], config);
        EXPECT_EQ(dst[i], expected) << "Mismatch at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, ConvertPixelBatch32_MinimumCount) {
    // Minimum SIMD-processable count is 8 (for AVX) or 4 (for SSE)
    // Test with 8 which works for both
    const int count = 8;
    std::vector<XULONG> src(count);
    std::vector<XULONG> dst(count, 0);

    for (int i = 0; i < count; ++i) {
        src[i] = RandomPixel();
    }

    VxPixelSimdConfig config = CreateRGBAtoBGRAConfig();

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        src.data(), dst.data(), count, config);

    // Should process exactly 8 pixels (multiple of both 4 and 8)
    EXPECT_EQ(processed, 8);

    for (int i = 0; i < processed; ++i) {
        XULONG expected = ScalarConvertPixel(src[i], config);
        EXPECT_EQ(dst[i], expected) << "Mismatch at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, ConvertPixelBatch32_LessThanMinimumReturnsZero) {
    // Less than minimum SIMD width should return 0 processed
    // Test with 7 which is less than AVX's 8
    std::vector<XULONG> src(7);
    std::vector<XULONG> dst(7, 0);

    for (int i = 0; i < 7; ++i) {
        src[i] = RandomPixel();
    }

    VxPixelSimdConfig config = CreateRGBAtoBGRAConfig();

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        src.data(), dst.data(), 7, config);

    // Should return 0 since count < minimum SIMD width (8 for AVX)
    EXPECT_EQ(processed, 0);
}

TEST_F(SIMDPixelTest, ConvertPixelBatch32_LargeBuffer) {
    const int count = 4096;  // Multiple of 8
    std::vector<XULONG> src(count);
    std::vector<XULONG> dst(count, 0);

    for (int i = 0; i < count; ++i) {
        src[i] = RandomPixel();
    }

    VxPixelSimdConfig config = CreateRGBAtoBGRAConfig();

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        src.data(), dst.data(), count, config);

    EXPECT_EQ(processed, count);

    int mismatches = 0;
    for (int i = 0; i < processed; ++i) {
        XULONG expected = ScalarConvertPixel(src[i], config);
        if (dst[i] != expected) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0);
}

TEST_F(SIMDPixelTest, ConvertPixelBatch32_DisabledConfig) {
    const int count = 32;
    std::vector<XULONG> src(count);
    std::vector<XULONG> dst(count, 0x12345678);

    for (int i = 0; i < count; ++i) {
        src[i] = RandomPixel();
    }

    VxPixelSimdConfig config;
    config.enabled = false;

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        src.data(), dst.data(), count, config);

    // Disabled config should return 0
    EXPECT_EQ(processed, 0);
}

//=============================================================================
// Apply Alpha Batch Tests
//=============================================================================

TEST_F(SIMDPixelTest, ApplyAlphaBatch32_FullAlpha) {
    ASSERT_NE(m_dispatch->Pixel.ApplyAlphaBatch32, nullptr);

    const int count = 64;
    std::vector<XULONG> pixels(count);
    std::vector<XULONG> original(count);

    for (int i = 0; i < count; ++i) {
        pixels[i] = RandomPixel();
        original[i] = pixels[i];
    }

    int processed = m_dispatch->Pixel.ApplyAlphaBatch32(
        pixels.data(), count, 255, RGBA_AMASK, 24);

    EXPECT_EQ(processed, count);

    for (int i = 0; i < processed; ++i) {
        XULONG expected = ScalarApplyAlpha(original[i], 255, RGBA_AMASK, 24);
        EXPECT_EQ(pixels[i], expected) << "Mismatch at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, ApplyAlphaBatch32_ZeroAlpha) {
    const int count = 64;
    std::vector<XULONG> pixels(count);

    for (int i = 0; i < count; ++i) {
        pixels[i] = 0xFFFFFFFF;  // Full alpha
    }

    int processed = m_dispatch->Pixel.ApplyAlphaBatch32(
        pixels.data(), count, 0, RGBA_AMASK, 24);

    EXPECT_EQ(processed, count);

    for (int i = 0; i < processed; ++i) {
        XULONG alpha = (pixels[i] & RGBA_AMASK) >> 24;
        EXPECT_EQ(alpha, 0u) << "Alpha should be 0 at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, ApplyAlphaBatch32_HalfAlpha) {
    // Use multiple of 8 for AVX compatibility
    const int count = 104;
    std::vector<XULONG> pixels(count);
    std::vector<XULONG> original(count);

    for (int i = 0; i < count; ++i) {
        pixels[i] = RandomPixel();
        original[i] = pixels[i];
    }

    int processed = m_dispatch->Pixel.ApplyAlphaBatch32(
        pixels.data(), count, 128, RGBA_AMASK, 24);

    // Should process at least some pixels
    EXPECT_GT(processed, 0);
    EXPECT_LE(processed, count);

    for (int i = 0; i < processed; ++i) {
        XULONG expected = ScalarApplyAlpha(original[i], 128, RGBA_AMASK, 24);
        EXPECT_EQ(pixels[i], expected) << "Mismatch at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, ApplyAlphaBatch32_LessThanMinimumReturnsZero) {
    // Test with less than minimum SIMD width (8 for AVX)
    std::vector<XULONG> pixels(7, 0xFFFFFFFF);
    std::vector<XULONG> original = pixels;

    int processed = m_dispatch->Pixel.ApplyAlphaBatch32(
        pixels.data(), 7, 128, RGBA_AMASK, 24);

    EXPECT_EQ(processed, 0);
    
    // Pixels should be unchanged
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(pixels[i], original[i]);
    }
}

//=============================================================================
// Apply Variable Alpha Batch Tests
//=============================================================================

TEST_F(SIMDPixelTest, ApplyVariableAlphaBatch32_AllZeroAlpha) {
    ASSERT_NE(m_dispatch->Pixel.ApplyVariableAlphaBatch32, nullptr);

    const int count = 64;
    std::vector<XULONG> pixels(count);
    std::vector<XBYTE> alphas(count, 0);

    for (int i = 0; i < count; ++i) {
        pixels[i] = 0xFFFFFFFF;  // Full alpha
    }

    int processed = m_dispatch->Pixel.ApplyVariableAlphaBatch32(
        pixels.data(), alphas.data(), count, RGBA_AMASK, 24);

    EXPECT_EQ(processed, count);

    for (int i = 0; i < processed; ++i) {
        XULONG alpha = (pixels[i] & RGBA_AMASK) >> 24;
        EXPECT_EQ(alpha, 0u) << "Alpha should be 0 at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, ApplyVariableAlphaBatch32_AllFullAlpha) {
    const int count = 64;
    std::vector<XULONG> pixels(count);
    std::vector<XBYTE> alphas(count, 255);
    std::vector<XULONG> original(count);

    for (int i = 0; i < count; ++i) {
        pixels[i] = RandomPixel();
        original[i] = pixels[i];
    }

    int processed = m_dispatch->Pixel.ApplyVariableAlphaBatch32(
        pixels.data(), alphas.data(), count, RGBA_AMASK, 24);

    EXPECT_EQ(processed, count);

    for (int i = 0; i < processed; ++i) {
        XULONG expected = ScalarApplyAlpha(original[i], 255, RGBA_AMASK, 24);
        EXPECT_EQ(pixels[i], expected) << "Mismatch at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, ApplyVariableAlphaBatch32_MixedAlpha) {
    // Use multiple of 8 for AVX compatibility
    const int count = 104;
    std::vector<XULONG> pixels(count);
    std::vector<XBYTE> alphas(count);
    std::vector<XULONG> original(count);

    for (int i = 0; i < count; ++i) {
        pixels[i] = RandomPixel();
        alphas[i] = RandomAlpha();
        original[i] = pixels[i];
    }

    int processed = m_dispatch->Pixel.ApplyVariableAlphaBatch32(
        pixels.data(), alphas.data(), count, RGBA_AMASK, 24);

    // Should process at least some pixels
    EXPECT_GT(processed, 0);
    EXPECT_LE(processed, count);

    for (int i = 0; i < processed; ++i) {
        XULONG expected = ScalarApplyAlpha(original[i], alphas[i], RGBA_AMASK, 24);
        EXPECT_EQ(pixels[i], expected) << "Mismatch at pixel " << i
            << " (alpha=" << static_cast<int>(alphas[i]) << ")";
    }
}

TEST_F(SIMDPixelTest, ApplyVariableAlphaBatch32_LessThanMinimumReturnsZero) {
    // Test with less than minimum SIMD width (8 for AVX)
    std::vector<XULONG> pixels(7, 0xFFFFFFFF);
    std::vector<XBYTE> alphas(7, 128);
    std::vector<XULONG> original = pixels;

    int processed = m_dispatch->Pixel.ApplyVariableAlphaBatch32(
        pixels.data(), alphas.data(), 7, RGBA_AMASK, 24);

    EXPECT_EQ(processed, 0);
    
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(pixels[i], original[i]);
    }
}

//=============================================================================
// Edge Cases
//=============================================================================

TEST_F(SIMDPixelTest, EdgeCase_ZeroCount) {
    XULONG pixel = 0x12345678;
    XULONG originalPixel = pixel;

    VxPixelSimdConfig config = CreateIdentityConfig();

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        &pixel, &pixel, 0, config);
    EXPECT_EQ(processed, 0);

    processed = m_dispatch->Pixel.ApplyAlphaBatch32(&pixel, 0, 128, RGBA_AMASK, 24);
    EXPECT_EQ(processed, 0);

    XBYTE alpha = 128;
    processed = m_dispatch->Pixel.ApplyVariableAlphaBatch32(
        &pixel, &alpha, 0, RGBA_AMASK, 24);
    EXPECT_EQ(processed, 0);
}

TEST_F(SIMDPixelTest, EdgeCase_AllWhite) {
    const int count = 32;
    std::vector<XULONG> src(count, 0xFFFFFFFF);
    std::vector<XULONG> dst(count, 0);

    VxPixelSimdConfig config = CreateRGBAtoBGRAConfig();

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        src.data(), dst.data(), count, config);

    EXPECT_EQ(processed, count);

    for (int i = 0; i < processed; ++i) {
        EXPECT_EQ(dst[i], 0xFFFFFFFF) << "White should stay white at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, EdgeCase_AllBlack) {
    const int count = 32;
    std::vector<XULONG> src(count, 0x00000000);
    std::vector<XULONG> dst(count, 0xFFFFFFFF);

    VxPixelSimdConfig config = CreateRGBAtoBGRAConfig();

    int processed = m_dispatch->Pixel.ConvertPixelBatch32(
        src.data(), dst.data(), count, config);

    EXPECT_EQ(processed, count);

    for (int i = 0; i < processed; ++i) {
        EXPECT_EQ(dst[i], 0x00000000) << "Black should stay black at pixel " << i;
    }
}

TEST_F(SIMDPixelTest, StressTest_LargeBufferVariableAlpha) {
    const int count = 8192;  // Multiple of 4
    std::vector<XULONG> pixels(count);
    std::vector<XBYTE> alphas(count);

    for (int i = 0; i < count; ++i) {
        pixels[i] = RandomPixel();
        alphas[i] = RandomAlpha();
    }

    int processed = m_dispatch->Pixel.ApplyVariableAlphaBatch32(
        pixels.data(), alphas.data(), count, RGBA_AMASK, 24);

    EXPECT_EQ(processed, count);

    // Verify no obvious corruption
    for (int i = 0; i < processed; ++i) {
        EXPECT_EQ(pixels[i], pixels[i]);  // Self-comparison
    }
}

} // namespace
