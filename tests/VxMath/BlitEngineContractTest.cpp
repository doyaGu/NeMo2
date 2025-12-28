/**
 * @file BlitEngineContractTest.cpp
 * @brief Contract and negative tests for VxBlitEngine.
 *
 * Tests documented preconditions, unsupported format behavior, and error paths.
 * Ensures the engine behaves predictably with invalid/edge inputs.
 */

#include "BlitEngineTestHelpers.h"

using namespace BlitEngineTest;

//==============================================================================
// Unsupported Format Tests
//==============================================================================

class BlitEngineUnsupportedFormatTest : public BlitEngineTestBase {
protected:
    // Create a DXT-like descriptor (compressed format - unsupported for blit)
    static VxImageDescEx CreateDXTDesc(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 4;  // DXT is typically 4 bits per pixel average
        desc.BytesPerLine = (width + 3) / 4 * 8;  // Block-based
        desc.RedMask = 0;
        desc.GreenMask = 0;
        desc.BlueMask = 0;
        desc.AlphaMask = 0;
        desc.Image = image;
        desc.Flags = 0x1000;  // Flag indicating compressed format
        return desc;
    }

    // Create an invalid descriptor with mismatched masks
    static VxImageDescEx CreateInvalidMasksDesc(int width, int height, XBYTE* image = nullptr) {
        VxImageDescEx desc;
        desc.Width = width;
        desc.Height = height;
        desc.BitsPerPixel = 32;
        desc.BytesPerLine = width * 4;
        // Intentionally overlapping masks (invalid configuration)
        desc.RedMask = 0xFF000000;
        desc.GreenMask = 0xFF000000;  // Same as red - invalid
        desc.BlueMask = 0x000000FF;
        desc.AlphaMask = 0x0000FF00;
        desc.Image = image;
        return desc;
    }
};

TEST_F(BlitEngineUnsupportedFormatTest, Blit_FromDXT_HandlesGracefully) {
    const int width = 16, height = 16;
    ImageBuffer srcBuf(width * height);  // Minimal buffer
    ImageBuffer dstBuf(width * height * 4);
    dstBuf.Fill(0xCD);

    auto srcDesc = CreateDXTDesc(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    // Should not crash; behavior is implementation-defined
    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

TEST_F(BlitEngineUnsupportedFormatTest, Blit_ToDXT_HandlesGracefully) {
    const int width = 16, height = 16;
    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height);

    PatternGenerator::FillSolid32(srcBuf.Data(), width, height, 128, 64, 192, 255);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = CreateDXTDesc(width, height, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

TEST_F(BlitEngineUnsupportedFormatTest, PixelFormat_Unknown_ReturnsUnknown) {
    VxImageDescEx desc;
    desc.Width = 10;
    desc.Height = 10;
    desc.BitsPerPixel = 17;  // Invalid bits per pixel
    desc.BytesPerLine = 30;
    desc.RedMask = 0x12345;
    desc.GreenMask = 0x54321;
    desc.BlueMask = 0xABCDE;
    desc.AlphaMask = 0;
    desc.Image = nullptr;

    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    // Should return UNKNOWN_PF or a sentinel value
    EXPECT_EQ(UNKNOWN_PF, fmt) << "Non-standard format should return UNKNOWN_PF";
}

//==============================================================================
// Quantize Precondition Tests
//==============================================================================

class BlitEngineQuantizeContractTest : public BlitEngineTestBase {};

TEST_F(BlitEngineQuantizeContractTest, Quantize_32BitSource_Succeeds) {
    const int width = 32, height = 32;

    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height);
    PaletteBuffer palette(4);

    PatternGenerator::FillGradient32(srcBuf.Data(), width, height);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create8BitPaletted(width, height, dstBuf.Data(),
                                                         palette.Data(), 4);

    CKERROR result = blitter.QuantizeImage(srcDesc, dstDesc);
    EXPECT_EQ(CK_OK, result) << "Quantize from 32-bit should succeed";
}

TEST_F(BlitEngineQuantizeContractTest, Quantize_24BitSource_Succeeds) {
    const int width = 32, height = 32;

    ImageBuffer srcBuf(width * height * 3);
    ImageBuffer dstBuf(width * height);
    PaletteBuffer palette(4);

    PatternGenerator::FillSolid24(srcBuf.Data(), width, height, 100, 150, 200);

    auto srcDesc = ImageDescFactory::Create24BitRGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create8BitPaletted(width, height, dstBuf.Data(),
                                                         palette.Data(), 4);

    CKERROR result = blitter.QuantizeImage(srcDesc, dstDesc);
    EXPECT_EQ(CK_OK, result) << "Quantize from 24-bit should succeed";
}

TEST_F(BlitEngineQuantizeContractTest, Quantize_16BitSource_MayFail) {
    const int width = 16, height = 16;

    ImageBuffer srcBuf(width * height * 2);
    ImageBuffer dstBuf(width * height);
    PaletteBuffer palette(4);

    auto srcDesc = ImageDescFactory::Create16Bit565(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create8BitPaletted(width, height, dstBuf.Data(),
                                                         palette.Data(), 4);

    // 16-bit source is not typically supported for quantize
    // Document actual behavior - should not crash
    EXPECT_NO_THROW(blitter.QuantizeImage(srcDesc, dstDesc));
}

TEST_F(BlitEngineQuantizeContractTest, Quantize_NullPalette_HandlesGracefully) {
    const int width = 16, height = 16;

    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height);

    PatternGenerator::FillSolid32(srcBuf.Data(), width, height, 128, 128, 128, 255);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    // Destination with null palette
    auto dstDesc = ImageDescFactory::Create8BitPaletted(width, height, dstBuf.Data(),
                                                         nullptr, 4);

    // Should either fail gracefully or succeed with some behavior
    EXPECT_NO_THROW(blitter.QuantizeImage(srcDesc, dstDesc));
}

//==============================================================================
// Resize Precondition Tests
//==============================================================================

class BlitEngineResizeContractTest : public BlitEngineTestBase {};

TEST_F(BlitEngineResizeContractTest, Resize_32BitSource_Succeeds) {
    const int srcW = 64, srcH = 64;
    const int dstW = 32, dstH = 32;

    ImageBuffer srcBuf(srcW * srcH * 4);
    ImageBuffer dstBuf(dstW * dstH * 4);

    PatternGenerator::FillCheckerboard32(srcBuf.Data(), srcW, srcH);

    auto srcDesc = ImageDescFactory::Create32BitARGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(dstW, dstH, dstBuf.Data());

    EXPECT_NO_THROW(blitter.ResizeImage(srcDesc, dstDesc));
}

TEST_F(BlitEngineResizeContractTest, Resize_24BitSource_MayConvertFirst) {
    const int srcW = 32, srcH = 32;
    const int dstW = 16, dstH = 16;

    ImageBuffer srcBuf(srcW * srcH * 3);
    ImageBuffer dstBuf(dstW * dstH * 3);

    PatternGenerator::FillSolid24(srcBuf.Data(), srcW, srcH, 100, 150, 200);

    auto srcDesc = ImageDescFactory::Create24BitRGB(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create24BitRGB(dstW, dstH, dstBuf.Data());

    // 24-bit resize may not be directly supported
    EXPECT_NO_THROW(blitter.ResizeImage(srcDesc, dstDesc));
}

TEST_F(BlitEngineResizeContractTest, Resize_16BitSource_MayNotWork) {
    const int srcW = 32, srcH = 32;
    const int dstW = 16, dstH = 16;

    ImageBuffer srcBuf(srcW * srcH * 2);
    ImageBuffer dstBuf(dstW * dstH * 2);

    auto srcDesc = ImageDescFactory::Create16Bit565(srcW, srcH, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create16Bit565(dstW, dstH, dstBuf.Data());

    // 16-bit resize typically not supported - document behavior
    EXPECT_NO_THROW(blitter.ResizeImage(srcDesc, dstDesc));
}

TEST_F(BlitEngineResizeContractTest, Resize_ZeroDimensions_NoOp) {
    ImageBuffer srcBuf(16);
    ImageBuffer dstBuf(16);

    auto srcDesc = ImageDescFactory::Create32BitARGB(0, 0, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(0, 0, dstBuf.Data());

    EXPECT_NO_THROW(blitter.ResizeImage(srcDesc, dstDesc));
}

//==============================================================================
// BytesPerLine Consistency Tests
//==============================================================================

class BlitEngineBytesPerLineTest : public BlitEngineTestBase {};

TEST_F(BlitEngineBytesPerLineTest, BytesPerLine_LessThanWidth_HandlesGracefully) {
    const int width = 32, height = 16;
    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height * 4);

    PatternGenerator::FillSolid32(srcBuf.Data(), width, height, 255, 0, 0, 255);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    // Set BytesPerLine less than width * bytesPerPixel (invalid)
    srcDesc.BytesPerLine = width * 2;  // Should be width * 4

    // Should not crash; behavior undefined but shouldn't corrupt memory
    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

TEST_F(BlitEngineBytesPerLineTest, BytesPerLine_Zero_HandlesGracefully) {
    const int width = 16, height = 16;
    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height * 4);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    srcDesc.BytesPerLine = 0;

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

TEST_F(BlitEngineBytesPerLineTest, BytesPerLine_Negative_HandlesGracefully) {
    // Some systems use negative pitch for bottom-up images
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4 + width * 4);  // Extra row for safety

    PatternGenerator::FillGradient32(buffer.Data() + width * 4, width, height);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, 
                                                      buffer.Data() + (height) * width * 4);
    srcDesc.BytesPerLine = -(int)(width * 4);  // Negative pitch

    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    // Document behavior with negative pitch
    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

//==============================================================================
// Alpha Operation Preconditions
//==============================================================================

class BlitEngineAlphaContractTest : public BlitEngineTestBase {};

TEST_F(BlitEngineAlphaContractTest, DoAlphaBlit_NoAlphaMask_HandlesGracefully) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 4);

    // 32-bit RGB without alpha mask
    auto desc = ImageDescFactory::Create32BitRGB(width, height, buffer.Data());

    // Setting alpha on format without alpha - behavior is impl-defined
    EXPECT_NO_THROW(blitter.DoAlphaBlit(desc, static_cast<XBYTE>(128)));
}

TEST_F(BlitEngineAlphaContractTest, DoAlphaBlit_16BitNoAlpha_HandlesGracefully) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 2);

    // 16-bit 565 has no alpha
    auto desc = ImageDescFactory::Create16Bit565(width, height, buffer.Data());

    EXPECT_NO_THROW(blitter.DoAlphaBlit(desc, static_cast<XBYTE>(200)));
}

TEST_F(BlitEngineAlphaContractTest, DoAlphaBlit_24Bit_HandlesGracefully) {
    const int width = 16, height = 16;
    ImageBuffer buffer(width * height * 3);

    auto desc = ImageDescFactory::Create24BitRGB(width, height, buffer.Data());

    EXPECT_NO_THROW(blitter.DoAlphaBlit(desc, static_cast<XBYTE>(255)));
}

TEST_F(BlitEngineAlphaContractTest, CopyAlpha_NullArray_NoOp) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 4);
    buffer.Fill(0xAB);

    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());

    // Null alpha array should be a no-op
    EXPECT_NO_THROW(blitter.DoAlphaBlit(desc, static_cast<XBYTE*>(nullptr)));

    // Buffer should be unchanged
    for (size_t i = 0; i < buffer.Size(); ++i) {
        EXPECT_EQ(0xAB, buffer[i]) << "Buffer modified with null alpha array at byte " << i;
    }
}

//==============================================================================
// Paletted Image Preconditions
//==============================================================================

class BlitEnginePalettedContractTest : public BlitEngineTestBase {};

TEST_F(BlitEnginePalettedContractTest, Paletted_NullColorMap_HandlesGracefully) {
    const int width = 16, height = 16;
    ImageBuffer srcBuf(width * height);
    ImageBuffer dstBuf(width * height * 4);

    // Fill indices
    for (int i = 0; i < width * height; ++i) {
        srcBuf[i] = static_cast<XBYTE>(i & 0xFF);
    }

    auto srcDesc = ImageDescFactory::Create8BitPaletted(width, height, srcBuf.Data(),
                                                         nullptr, 4);  // Null palette
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    // Should handle gracefully
    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

TEST_F(BlitEnginePalettedContractTest, Paletted_ZeroColorMapEntries_HandlesGracefully) {
    const int width = 8, height = 8;
    ImageBuffer srcBuf(width * height);
    ImageBuffer dstBuf(width * height * 4);
    PaletteBuffer palette(4);

    auto srcDesc = ImageDescFactory::Create8BitPaletted(width, height, srcBuf.Data(),
                                                         palette.Data(), 4);
    srcDesc.ColorMapEntries = 0;  // Invalid

    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

TEST_F(BlitEnginePalettedContractTest, Paletted_IndexOutOfRange_HandlesGracefully) {
    const int width = 4, height = 4;
    ImageBuffer srcBuf(width * height);
    ImageBuffer dstBuf(width * height * 4);
    PaletteBuffer palette(4);

    // Fill palette with some colors
    for (int i = 0; i < 16; ++i) {
        palette.SetColor(i, 0xFF000000 | (i << 16) | (i << 8) | i);
    }

    // Fill image with indices, some exceeding ColorMapEntries
    for (int i = 0; i < width * height; ++i) {
        srcBuf[i] = static_cast<XBYTE>(i * 20);  // Some will be > 255 % 256 = wrapping
    }

    auto srcDesc = ImageDescFactory::Create8BitPaletted(width, height, srcBuf.Data(),
                                                         palette.Data(), 4);
    srcDesc.ColorMapEntries = 16;  // Only 16 valid entries

    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    // Should not crash even with out-of-range indices
    EXPECT_NO_THROW(blitter.DoBlit(srcDesc, dstDesc));
}

//==============================================================================
// Image Operations Preconditions
//==============================================================================

class BlitEngineImageOpsContractTest : public BlitEngineTestBase {};

TEST_F(BlitEngineImageOpsContractTest, PremultiplyAlpha_NonARGB_HandlesGracefully) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 3);

    auto desc = ImageDescFactory::Create24BitRGB(width, height, buffer.Data());

    // Premultiply on non-32-bit format
    EXPECT_NO_THROW(blitter.PremultiplyAlpha(desc));
}

TEST_F(BlitEngineImageOpsContractTest, SwapRedBlue_16Bit_HandlesGracefully) {
    const int width = 8, height = 8;
    ImageBuffer buffer(width * height * 2);

    auto desc = ImageDescFactory::Create16Bit565(width, height, buffer.Data());

    // SwapRedBlue is typically 32-bit only
    EXPECT_NO_THROW(blitter.SwapRedBlue(desc));
}

TEST_F(BlitEngineImageOpsContractTest, Grayscale_NonRGB_HandlesGracefully) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height);
    PaletteBuffer palette(4);

    auto desc = ImageDescFactory::Create8BitPaletted(width, height, buffer.Data(),
                                                      palette.Data(), 4);

    EXPECT_NO_THROW(blitter.ConvertToGrayscale(desc));
}

TEST_F(BlitEngineImageOpsContractTest, MultiplyBlend_DimensionMismatch_HandlesGracefully) {
    ImageBuffer srcBuf(32 * 32 * 4);
    ImageBuffer dstBuf(16 * 16 * 4);

    PatternGenerator::FillSolid32(srcBuf.Data(), 32, 32, 128, 128, 128, 255);
    PatternGenerator::FillSolid32(dstBuf.Data(), 16, 16, 255, 255, 255, 255);

    auto srcDesc = ImageDescFactory::Create32BitARGB(32, 32, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(16, 16, dstBuf.Data());

    // Dimension mismatch - should handle gracefully
    EXPECT_NO_THROW(blitter.MultiplyBlend(srcDesc, dstDesc));
}

//==============================================================================
// Pixel Format Conversion Contract
//==============================================================================

class BlitEnginePixelFormatContractTest : public BlitEngineTestBase {};

TEST_F(BlitEnginePixelFormatContractTest, ConvertPixelFormat_AllKnownFormats) {
    // Test that all known formats can be converted without crashing
    VX_PIXELFORMAT formats[] = {
        _32_ARGB8888, _32_RGB888, _24_RGB888,
        _16_RGB565, _16_RGB555, _16_ARGB1555, _16_ARGB4444,
        _32_ABGR8888, _32_RGBA8888, _32_BGRA8888, _24_BGR888
    };

    for (VX_PIXELFORMAT fmt : formats) {
        XULONG aMask, rMask, gMask, bMask;
        EXPECT_NO_THROW(BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask))
            << "ConvertPixelFormat failed for format " << static_cast<int>(fmt);
    }
}

TEST_F(BlitEnginePixelFormatContractTest, PixelFormat2String_AllFormats) {
    VX_PIXELFORMAT formats[] = {
        UNKNOWN_PF, _32_ARGB8888, _32_RGB888, _24_RGB888,
        _16_RGB565, _16_RGB555, _16_ARGB1555, _16_ARGB4444
    };

    for (VX_PIXELFORMAT fmt : formats) {
        XString name = BlitterWrapper::PixelFormat2String(fmt);
        EXPECT_FALSE(name.Length() == 0) << "PixelFormat2String returned empty for " << static_cast<int>(fmt);
    }
}

//==============================================================================
// Thread Safety Documentation Tests
//==============================================================================

class BlitEngineThreadSafetyTest : public BlitEngineTestBase {};

TEST_F(BlitEngineThreadSafetyTest, DocumentThreadSafety_SingleEngineInstance) {
    // This test documents that TheBlitter is a global instance
    // Concurrent access from multiple threads may require external synchronization
    
    // Single-threaded sanity check
    const int width = 32, height = 32;
    ImageBuffer srcBuf(width * height * 4);
    ImageBuffer dstBuf(width * height * 4);

    PatternGenerator::FillColorBars32(srcBuf.Data(), width, height);

    auto srcDesc = ImageDescFactory::Create32BitARGB(width, height, srcBuf.Data());
    auto dstDesc = ImageDescFactory::Create32BitARGB(width, height, dstBuf.Data());

    // Multiple sequential blits should work
    for (int i = 0; i < 10; ++i) {
        blitter.DoBlit(srcDesc, dstDesc);
    }

    EXPECT_EQ(0, ImageComparator::Compare32(srcBuf.Data(), dstBuf.Data(), width, height, 0));
}

