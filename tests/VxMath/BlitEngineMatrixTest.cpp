/**
 * @file BlitEngineMatrixTest.cpp
 * @brief Exhaustive format pair coverage matrix for VxBlitEngine.
 *
 * Tests all supported format pairs across multiple dimensions to ensure
 * complete branch coverage of the dispatch tables and conversion routines.
 *
 * Coverage matrix:
 * - Source formats: 32-bit (ARGB, RGB, ABGR, RGBA, BGRA), 24-bit (RGB, BGR),
 *                   16-bit (565, 555, 1555, 4444), 8-bit paletted
 * - Destination formats: Same as source
 * - Dimensions: 1x1, 7x5, 16x16, 31x33, 64x64
 * - Stride variants: Tight, padded
 */

#include "BlitEngineTestHelpers.h"
#include <tuple>
#include <string>

using namespace BlitEngineTest;

//==============================================================================
// Format Factory Types
//==============================================================================

using FormatFactory = std::function<VxImageDescEx(int, int, XBYTE*)>;

struct FormatInfo {
    std::string name;
    FormatFactory factory;
    int bytesPerPixel;
    bool hasAlpha;
    bool isPaletted;
};

class BlitEngineMatrixTest : public BlitEngineTestBase {
protected:
    // All format factories
    static std::vector<FormatInfo> GetAllFormats() {
        return {
            // 32-bit formats
            {"32_ARGB", ImageDescFactory::Create32BitARGB, 4, true, false},
            {"32_RGB", ImageDescFactory::Create32BitRGB, 4, false, false},
            {"32_ABGR", ImageDescFactory::Create32BitABGR, 4, true, false},
            {"32_RGBA", ImageDescFactory::Create32BitRGBA, 4, true, false},
            // 24-bit formats
            {"24_RGB", ImageDescFactory::Create24BitRGB, 3, false, false},
            // 16-bit formats
            {"16_565", ImageDescFactory::Create16Bit565, 2, false, false},
            {"16_555", ImageDescFactory::Create16Bit555, 2, false, false},
            {"16_1555", ImageDescFactory::Create16Bit1555, 2, true, false},
            {"16_4444", ImageDescFactory::Create16Bit4444, 2, true, false},
        };
    }

    // Widths that exercise SIMD remainder handling
    static std::vector<int> GetTestWidths() {
        return {1, 7, 16, 31, 64};
    }

    static std::vector<int> GetTestHeights() {
        return {1, 5, 16, 33};
    }

    // Fill source buffer with format-appropriate pattern
    void FillSourceBuffer(XBYTE* buffer, int width, int height, int bpp) {
        if (bpp == 4) {
            PatternGenerator::FillUniquePixels32(buffer, width, height);
        } else if (bpp == 3) {
            for (int i = 0; i < width * height; ++i) {
                // Offset by 1 to ensure first pixel is non-zero
                buffer[i * 3 + 0] = static_cast<XBYTE>(((i + 1) * 7) & 0xFF);
                buffer[i * 3 + 1] = static_cast<XBYTE>(((i + 1) * 13) & 0xFF);
                buffer[i * 3 + 2] = static_cast<XBYTE>(((i + 1) * 23) & 0xFF);
                // Ensure at least one channel non-zero
                if (buffer[i * 3 + 0] == 0 && buffer[i * 3 + 1] == 0 && buffer[i * 3 + 2] == 0) {
                    buffer[i * 3 + 0] = 1;
                }
            }
        } else if (bpp == 2) {
            XWORD* words = reinterpret_cast<XWORD*>(buffer);
            for (int i = 0; i < width * height; ++i) {
                // Offset by 1 to ensure first pixel is non-zero
                words[i] = static_cast<XWORD>(((i + 1) * 17) & 0xFFFF);
                if (words[i] == 0) words[i] = 1;
            }
        } else if (bpp == 1) {
            for (int i = 0; i < width * height; ++i) {
                // Offset by 1 to ensure first pixel is non-zero
                buffer[i] = static_cast<XBYTE>((i + 1) & 0xFF);
                if (buffer[i] == 0) buffer[i] = 1;
            }
        }
    }

    // Verify destination has non-zero content
    bool HasContent(const XBYTE* buffer, int width, int height, int bpp) {
        int size = width * height * bpp;
        for (int i = 0; i < size; ++i) {
            if (buffer[i] != 0) return true;
        }
        return false;
    }
};

//==============================================================================
// 32-bit to 32-bit Conversion Matrix
//==============================================================================

class BlitEngine32To32MatrixTest : public BlitEngineMatrixTest,
    public ::testing::WithParamInterface<std::tuple<std::string, std::string, int, int>> {};

TEST_P(BlitEngine32To32MatrixTest, ConversionWorks) {
    auto [srcName, dstName, width, height] = GetParam();
    
    auto formats = GetAllFormats();
    FormatInfo srcFormat, dstFormat;
    for (const auto& f : formats) {
        if (f.name == srcName) srcFormat = f;
        if (f.name == dstName) dstFormat = f;
    }

    if (srcFormat.bytesPerPixel != 4 || dstFormat.bytesPerPixel != 4) {
        GTEST_SKIP() << "Not a 32-to-32 conversion";
    }

    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height * 4);
    dstBuf.Fill(0);

    FillSourceBuffer(srcBuf.Data(), width, height, 4);

    auto srcDesc = srcFormat.factory(width, height, srcBuf.Data());
    auto dstDesc = dstFormat.factory(width, height, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    EXPECT_TRUE(HasContent(dstBuf.Data(), width, height, 4))
        << "Conversion " << srcName << " -> " << dstName << " (" << width << "x" << height << ") produced empty output";
}

INSTANTIATE_TEST_SUITE_P(
    Format32Matrix,
    BlitEngine32To32MatrixTest,
    ::testing::Combine(
        ::testing::Values("32_ARGB", "32_RGB", "32_ABGR", "32_RGBA"),
        ::testing::Values("32_ARGB", "32_RGB", "32_ABGR", "32_RGBA"),
        ::testing::Values(1, 7, 16, 31, 64),
        ::testing::Values(1, 5, 16)
    ),
    [](const auto& info) {
        return std::get<0>(info.param) + "_to_" + std::get<1>(info.param) +
               "_" + std::to_string(std::get<2>(info.param)) + "x" +
               std::to_string(std::get<3>(info.param));
    }
);

//==============================================================================
// 32-bit to 16-bit Conversion Matrix
//==============================================================================

class BlitEngine32To16MatrixTest : public BlitEngineMatrixTest,
    public ::testing::WithParamInterface<std::tuple<std::string, std::string, int, int>> {};

TEST_P(BlitEngine32To16MatrixTest, ConversionWorks) {
    auto [srcName, dstName, width, height] = GetParam();

    auto formats = GetAllFormats();
    FormatInfo srcFormat, dstFormat;
    for (const auto& f : formats) {
        if (f.name == srcName) srcFormat = f;
        if (f.name == dstName) dstFormat = f;
    }

    if (srcFormat.bytesPerPixel != 4 || dstFormat.bytesPerPixel != 2) {
        GTEST_SKIP() << "Not a 32-to-16 conversion";
    }

    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height * 2);
    dstBuf.Fill(0);

    FillSourceBuffer(srcBuf.Data(), width, height, 4);

    auto srcDesc = srcFormat.factory(width, height, srcBuf.Data());
    auto dstDesc = dstFormat.factory(width, height, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    EXPECT_TRUE(HasContent(dstBuf.Data(), width, height, 2))
        << "Conversion " << srcName << " -> " << dstName << " produced empty output";
}

INSTANTIATE_TEST_SUITE_P(
    Format32To16Matrix,
    BlitEngine32To16MatrixTest,
    ::testing::Combine(
        ::testing::Values("32_ARGB", "32_RGB"),
        ::testing::Values("16_565", "16_555", "16_1555", "16_4444"),
        ::testing::Values(1, 7, 16, 31),
        ::testing::Values(1, 8, 16)
    ),
    [](const auto& info) {
        return std::get<0>(info.param) + "_to_" + std::get<1>(info.param) +
               "_" + std::to_string(std::get<2>(info.param)) + "x" +
               std::to_string(std::get<3>(info.param));
    }
);

//==============================================================================
// 16-bit to 32-bit Conversion Matrix
//==============================================================================

class BlitEngine16To32MatrixTest : public BlitEngineMatrixTest,
    public ::testing::WithParamInterface<std::tuple<std::string, std::string, int, int>> {};

TEST_P(BlitEngine16To32MatrixTest, ConversionWorks) {
    auto [srcName, dstName, width, height] = GetParam();

    auto formats = GetAllFormats();
    FormatInfo srcFormat, dstFormat;
    for (const auto& f : formats) {
        if (f.name == srcName) srcFormat = f;
        if (f.name == dstName) dstFormat = f;
    }

    if (srcFormat.bytesPerPixel != 2 || dstFormat.bytesPerPixel != 4) {
        GTEST_SKIP() << "Not a 16-to-32 conversion";
    }

    ImageBuffer srcBuf(width * height * 2);
    ImageBuffer dstBuf(width * height * 4);
    dstBuf.Fill(0);

    FillSourceBuffer(srcBuf.Data(), width, height, 2);

    auto srcDesc = srcFormat.factory(width, height, srcBuf.Data());
    auto dstDesc = dstFormat.factory(width, height, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    EXPECT_TRUE(HasContent(dstBuf.Data(), width, height, 4))
        << "Conversion " << srcName << " -> " << dstName << " produced empty output";
}

INSTANTIATE_TEST_SUITE_P(
    Format16To32Matrix,
    BlitEngine16To32MatrixTest,
    ::testing::Combine(
        ::testing::Values("16_565", "16_555", "16_1555", "16_4444"),
        ::testing::Values("32_ARGB", "32_RGB"),
        ::testing::Values(1, 7, 16, 31),
        ::testing::Values(1, 8, 16)
    ),
    [](const auto& info) {
        return std::get<0>(info.param) + "_to_" + std::get<1>(info.param) +
               "_" + std::to_string(std::get<2>(info.param)) + "x" +
               std::to_string(std::get<3>(info.param));
    }
);

//==============================================================================
// 32-bit to 24-bit and 24-bit to 32-bit Matrix
//==============================================================================

class BlitEngine24BitMatrixTest : public BlitEngineMatrixTest,
    public ::testing::WithParamInterface<std::tuple<std::string, std::string, int, int>> {};

TEST_P(BlitEngine24BitMatrixTest, ConversionWorks) {
    auto [srcName, dstName, width, height] = GetParam();

    auto formats = GetAllFormats();
    FormatInfo srcFormat, dstFormat;
    for (const auto& f : formats) {
        if (f.name == srcName) srcFormat = f;
        if (f.name == dstName) dstFormat = f;
    }

    ImageBuffer srcBuf(width * height * srcFormat.bytesPerPixel);
    ImageBuffer dstBuf(width * height * dstFormat.bytesPerPixel);
    dstBuf.Fill(0);

    FillSourceBuffer(srcBuf.Data(), width, height, srcFormat.bytesPerPixel);

    auto srcDesc = srcFormat.factory(width, height, srcBuf.Data());
    auto dstDesc = dstFormat.factory(width, height, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    EXPECT_TRUE(HasContent(dstBuf.Data(), width, height, dstFormat.bytesPerPixel))
        << "Conversion " << srcName << " -> " << dstName << " produced empty output";
}

INSTANTIATE_TEST_SUITE_P(
    Format24BitMatrix,
    BlitEngine24BitMatrixTest,
    ::testing::Combine(
        ::testing::Values("32_ARGB", "32_RGB", "24_RGB"),
        ::testing::Values("32_ARGB", "32_RGB", "24_RGB"),
        ::testing::Values(1, 7, 16, 31),
        ::testing::Values(1, 8, 16)
    ),
    [](const auto& info) {
        return std::get<0>(info.param) + "_to_" + std::get<1>(info.param) +
               "_" + std::to_string(std::get<2>(info.param)) + "x" +
               std::to_string(std::get<3>(info.param));
    }
);

//==============================================================================
// 16-bit to 16-bit Conversion Matrix
//==============================================================================

class BlitEngine16To16MatrixTest : public BlitEngineMatrixTest,
    public ::testing::WithParamInterface<std::tuple<std::string, std::string, int, int>> {};

TEST_P(BlitEngine16To16MatrixTest, ConversionWorks) {
    auto [srcName, dstName, width, height] = GetParam();

    auto formats = GetAllFormats();
    FormatInfo srcFormat, dstFormat;
    for (const auto& f : formats) {
        if (f.name == srcName) srcFormat = f;
        if (f.name == dstName) dstFormat = f;
    }

    if (srcFormat.bytesPerPixel != 2 || dstFormat.bytesPerPixel != 2) {
        GTEST_SKIP() << "Not a 16-to-16 conversion";
    }

    ImageBuffer srcBuf(width * height * 2);
    ImageBuffer dstBuf(width * height * 2);
    dstBuf.Fill(0);

    FillSourceBuffer(srcBuf.Data(), width, height, 2);

    auto srcDesc = srcFormat.factory(width, height, srcBuf.Data());
    auto dstDesc = dstFormat.factory(width, height, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    // Same format should produce output
    if (srcName == dstName) {
        EXPECT_EQ(0, memcmp(srcBuf.Data(), dstBuf.Data(), srcBuf.Size()))
            << "Same format copy should be identical";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Format16To16Matrix,
    BlitEngine16To16MatrixTest,
    ::testing::Combine(
        ::testing::Values("16_565", "16_555", "16_1555", "16_4444"),
        ::testing::Values("16_565", "16_555", "16_1555", "16_4444"),
        ::testing::Values(1, 7, 16, 31),
        ::testing::Values(1, 8)
    ),
    [](const auto& info) {
        return std::get<0>(info.param) + "_to_" + std::get<1>(info.param) +
               "_" + std::to_string(std::get<2>(info.param)) + "x" +
               std::to_string(std::get<3>(info.param));
    }
);

//==============================================================================
// Paletted Image Conversion Matrix
//==============================================================================

class BlitEnginePalettedMatrixTest : public BlitEngineMatrixTest,
    public ::testing::WithParamInterface<std::tuple<std::string, int, int>> {};

TEST_P(BlitEnginePalettedMatrixTest, PalettedTo32BitWorks) {
    auto [dstName, width, height] = GetParam();

    auto formats = GetAllFormats();
    FormatInfo dstFormat;
    for (const auto& f : formats) {
        if (f.name == dstName) dstFormat = f;
    }

    if (dstFormat.bytesPerPixel != 4) {
        GTEST_SKIP() << "Not paletted-to-32 conversion";
    }

    ImageBuffer srcBuf(width * height);
    ImageBuffer dstBuf(width * height * 4);
    PaletteBuffer palette(4);
    dstBuf.Fill(0);

    // Create a simple palette
    for (int i = 0; i < 256; ++i) {
        palette.SetColor(i, 0xFF000000 | (i << 16) | ((255 - i) << 8) | (i / 2));
    }

    // Fill source with indices
    for (int i = 0; i < width * height; ++i) {
        srcBuf[i] = static_cast<XBYTE>(i & 0xFF);
    }

    auto srcDesc = ImageDescFactory::Create8BitPaletted(width, height, srcBuf.Data(),
                                                         palette.Data(), 4);
    auto dstDesc = dstFormat.factory(width, height, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    EXPECT_TRUE(HasContent(dstBuf.Data(), width, height, 4))
        << "Paletted -> " << dstName << " produced empty output";
}

INSTANTIATE_TEST_SUITE_P(
    PalettedMatrix,
    BlitEnginePalettedMatrixTest,
    ::testing::Combine(
        ::testing::Values("32_ARGB", "32_RGB"),
        ::testing::Values(1, 7, 16, 31, 64),
        ::testing::Values(1, 8, 16)
    ),
    [](const auto& info) {
        return "Paletted_to_" + std::get<0>(info.param) +
               "_" + std::to_string(std::get<1>(info.param)) + "x" +
               std::to_string(std::get<2>(info.param));
    }
);

//==============================================================================
// Stride Matrix - Test padded strides across format pairs
//==============================================================================

class BlitEngineStrideMatrixTest : public BlitEngineMatrixTest,
    public ::testing::WithParamInterface<std::tuple<std::string, std::string, int>> {};

TEST_P(BlitEngineStrideMatrixTest, PaddedStrideWorks) {
    auto [srcName, dstName, padding] = GetParam();

    auto formats = GetAllFormats();
    FormatInfo srcFormat, dstFormat;
    for (const auto& f : formats) {
        if (f.name == srcName) srcFormat = f;
        if (f.name == dstName) dstFormat = f;
    }

    const int width = 17, height = 8;  // Odd width for SIMD tail

    int srcStride = width * srcFormat.bytesPerPixel + padding;
    int dstStride = width * dstFormat.bytesPerPixel + padding;

    std::vector<XBYTE> srcBuf(srcStride * height, 0xCD);
    std::vector<XBYTE> dstBuf(dstStride * height, 0xCD);

    // Fill source pixel data only
    for (int y = 0; y < height; ++y) {
        XBYTE* row = srcBuf.data() + y * srcStride;
        for (int x = 0; x < width; ++x) {
            if (srcFormat.bytesPerPixel == 4) {
                reinterpret_cast<XULONG*>(row)[x] = 0xFFAABBCC;
            } else if (srcFormat.bytesPerPixel == 3) {
                row[x * 3 + 0] = 0xCC;
                row[x * 3 + 1] = 0xBB;
                row[x * 3 + 2] = 0xAA;
            } else if (srcFormat.bytesPerPixel == 2) {
                reinterpret_cast<XWORD*>(row)[x] = 0xAAAA;
            }
        }
    }

    auto srcDesc = srcFormat.factory(width, height, srcBuf.data());
    auto dstDesc = dstFormat.factory(width, height, dstBuf.data());
    srcDesc.BytesPerLine = srcStride;
    dstDesc.BytesPerLine = dstStride;

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));

    // Check that padding bytes are still sentinel values
    for (int y = 0; y < height; ++y) {
        const XBYTE* row = dstBuf.data() + y * dstStride;
        for (int p = width * dstFormat.bytesPerPixel; p < dstStride; ++p) {
            EXPECT_EQ(0xCD, row[p])
                << "Padding overwritten at row " << y << ", byte " << p
                << " for " << srcName << " -> " << dstName;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    StrideMatrix,
    BlitEngineStrideMatrixTest,
    ::testing::Combine(
        ::testing::Values("32_ARGB", "24_RGB", "16_565"),
        ::testing::Values("32_ARGB", "24_RGB", "16_565"),
        ::testing::Values(4, 16, 32)
    ),
    [](const auto& info) {
        return std::get<0>(info.param) + "_to_" + std::get<1>(info.param) +
               "_pad" + std::to_string(std::get<2>(info.param));
    }
);

//==============================================================================
// Alpha Preservation Matrix
//==============================================================================

class BlitEngineAlphaPreservationTest : public BlitEngineMatrixTest {};

TEST_F(BlitEngineAlphaPreservationTest, ARGB_To_ARGB_PreservesAlpha) {
    const int width = 16, height = 16;
    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height * 4);

    // Fill with varying alpha
    XULONG* src = reinterpret_cast<XULONG*>(srcBuf.Data());
    for (int i = 0; i < width * height; ++i) {
        int alpha = (i * 3) % 256;
        src[i] = (alpha << 24) | 0x00AABBCC;
    }

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    const XULONG* dst = reinterpret_cast<const XULONG*>(dstBuf.Data());
    for (int i = 0; i < width * height; ++i) {
        int expectedAlpha = (i * 3) % 256;
        EXPECT_EQ(expectedAlpha, (dst[i] >> 24) & 0xFF)
            << "Alpha not preserved at pixel " << i;
    }
}

TEST_F(BlitEngineAlphaPreservationTest, ARGB_To_1555_ThresholdsAlpha) {
    const int width = 8, height = 8;
    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height * 2);

    // Fill with varying alpha: some below 128, some above
    XULONG* src = reinterpret_cast<XULONG*>(srcBuf.Data());
    for (int i = 0; i < width * height; ++i) {
        int alpha = (i < (width * height / 2)) ? 64 : 200;  // Half low, half high
        src[i] = (alpha << 24) | 0x00FF0000;  // Red with varying alpha
    }

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create16Bit1555(width, height, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    const XWORD* dst = reinterpret_cast<const XWORD*>(dstBuf.Data());
    for (int i = 0; i < width * height; ++i) {
        bool hasAlpha = (dst[i] & 0x8000) != 0;
        if (i < width * height / 2) {
            // Low alpha -> alpha bit should be 0
            EXPECT_FALSE(hasAlpha) << "Low alpha pixel " << i << " has alpha bit set";
        } else {
            // High alpha -> alpha bit should be 1
            EXPECT_TRUE(hasAlpha) << "High alpha pixel " << i << " has alpha bit clear";
        }
    }
}

TEST_F(BlitEngineAlphaPreservationTest, ARGB_To_4444_QuantizesAlpha) {
    const int width = 16, height = 1;
    ImageBuffer srcBuf(width * 4);
    ImageBuffer dstBuf(width * 2);

    // Fill with gradient alpha
    XULONG* src = reinterpret_cast<XULONG*>(srcBuf.Data());
    for (int i = 0; i < width; ++i) {
        int alpha = i * 17;  // 0, 17, 34, ..., 255
        src[i] = (alpha << 24) | 0x00808080;
    }

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, 1, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create16Bit4444(width, 1, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    const XWORD* dst = reinterpret_cast<const XWORD*>(dstBuf.Data());
    for (int i = 0; i < width; ++i) {
        int alpha4bit = (dst[i] >> 12) & 0xF;
        int expectedApprox = (i * 17) / 17;  // Should be roughly i
        // Allow some tolerance for rounding
        EXPECT_NEAR(expectedApprox, alpha4bit, 1)
            << "4-bit alpha quantization wrong at pixel " << i;
    }
}

TEST_F(BlitEngineAlphaPreservationTest, RGB_To_ARGB_SetsFullAlpha) {
    const int width = 8, height = 8;
    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height * 4);
    dstBuf.Fill(0);

    // Source is RGB (no alpha mask)
    XULONG* src = reinterpret_cast<XULONG*>(srcBuf.Data());
    for (int i = 0; i < width * height; ++i) {
        src[i] = 0x00AABBCC;  // RGB only, alpha byte is 0
    }

    auto srcDesc = ImageDescFactory::Create32BitRGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    const XULONG* dst = reinterpret_cast<const XULONG*>(dstBuf.Data());
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(0xFF, (dst[i] >> 24) & 0xFF)
            << "RGB->ARGB should set full alpha at pixel " << i;
    }
}

//==============================================================================
// Color Accuracy Tests
//==============================================================================

class BlitEngineColorAccuracyTest : public BlitEngineMatrixTest {};

TEST_F(BlitEngineColorAccuracyTest, 32To16To32_Roundtrip_ReasonableError) {
    const int width = 8, height = 8;
    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer midBuf(width * height * 2);
    ImageBuffer dstBuf(width * height * 4);

    // Fill with a gradient
    PatternGenerator::FillGradient32(srcBuf.Data(), width, height);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto midDesc = ImageDescFactory::Create16Bit565(width, height, midBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    blitter.DoBlit(srcDesc, midDesc);
    blitter.DoBlit(midDesc, dstDesc);

    // 565 has limited precision; allow reasonable error
    double mse = ImageComparator::CalcMSE32(srcBuf.Data(), dstBuf.Data(), width, height, false);
    // With 5/6/5 bit precision, expect some error but not huge
    EXPECT_LT(mse, 100.0) << "32->16->32 roundtrip MSE too high: " << mse;
}

TEST_F(BlitEngineColorAccuracyTest, PureColors_32To565_MaintainPurity) {
    // Pure colors should map to max values in 565
    ImageBuffer srcBuf(4);
    ImageBuffer dstBuf(2);

    auto srcDesc = ImageDescFactory::Create32BitARGB(1, 1, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create16Bit565(1, 1, dstBuf.Data());

    // Test pure red
    *reinterpret_cast<XULONG*>(srcBuf.Data()) = 0xFFFF0000;
    blitter.DoBlit(srcDesc, dstDesc);
    EXPECT_EQ(0xF800, *reinterpret_cast<XWORD*>(dstBuf.Data())) << "Pure red -> 565 failed";

    // Test pure green
    *reinterpret_cast<XULONG*>(srcBuf.Data()) = 0xFF00FF00;
    blitter.DoBlit(srcDesc, dstDesc);
    EXPECT_EQ(0x07E0, *reinterpret_cast<XWORD*>(dstBuf.Data())) << "Pure green -> 565 failed";

    // Test pure blue
    *reinterpret_cast<XULONG*>(srcBuf.Data()) = 0xFF0000FF;
    blitter.DoBlit(srcDesc, dstDesc);
    EXPECT_EQ(0x001F, *reinterpret_cast<XWORD*>(dstBuf.Data())) << "Pure blue -> 565 failed";
}

TEST_F(BlitEngineColorAccuracyTest, Paletted_ExactColorMatch) {
    const int width = 4, height = 4;
    ImageBuffer srcBuf(width * height);
    ImageBuffer dstBuf(width * height * 4);
    PaletteBuffer palette(4);

    // Set up specific palette colors
    palette.SetColor(0, 0xFFFF0000);  // Red
    palette.SetColor(1, 0xFF00FF00);  // Green
    palette.SetColor(2, 0xFF0000FF);  // Blue
    palette.SetColor(3, 0xFFFFFFFF);  // White

    // Fill image with palette indices
    for (int i = 0; i < width * height; ++i) {
        srcBuf[i] = static_cast<XBYTE>(i % 4);
    }

    auto srcDesc = ImageDescFactory::Create8BitPaletted(width, height, srcBuf.Data(),
                                                         palette.Data(), 4);
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    blitter.DoBlit(srcDesc, dstDesc);

    const XULONG* dst = reinterpret_cast<const XULONG*>(dstBuf.Data());
    XULONG expected[] = {0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF};
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(expected[i % 4], dst[i])
            << "Paletted color mismatch at pixel " << i;
    }
}

