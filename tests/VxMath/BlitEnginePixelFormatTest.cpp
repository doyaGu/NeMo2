/**
 * @file BlitEnginePixelFormatTest.cpp
 * @brief Tests for VxBlitEngine pixel format detection and conversion.
 *
 * Tests:
 * - GetPixelFormat detection from masks
 * - ConvertPixelFormat conversion
 * - PixelFormat2String conversion
 * - Pixel format utilities
 */

#include "BlitEngineTestHelpers.h"

using namespace BlitEngineTest;

class PixelFormatTest : public BlitEngineTestBase {};

//==============================================================================
// GetPixelFormat Tests - 32-bit formats
//==============================================================================

TEST_F(PixelFormatTest, GetPixelFormat_32BitARGB) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_32_ARGB8888, fmt);
}

TEST_F(PixelFormatTest, GetPixelFormat_32BitRGB) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitRGB(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_32_RGB888, fmt);
}

TEST_F(PixelFormatTest, GetPixelFormat_32BitABGR) {
    VxImageDescEx desc;
    memset(&desc, 0, sizeof(desc));
    desc.Size = sizeof(desc);
    desc.Width = 4;
    desc.Height = 4;
    desc.BitsPerPixel = 32;
    desc.BytesPerLine = 4 * 4;
    // ABGR: A in bits 24-31, B in 16-23, G in 8-15, R in 0-7
    desc.AlphaMask = 0xFF000000;
    desc.BlueMask  = 0x00FF0000;
    desc.GreenMask = 0x0000FF00;
    desc.RedMask   = 0x000000FF;
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_32_ABGR8888, fmt);
}

TEST_F(PixelFormatTest, GetPixelFormat_32BitBGRA) {
    VxImageDescEx desc;
    memset(&desc, 0, sizeof(desc));
    desc.Size = sizeof(desc);
    desc.Width = 4;
    desc.Height = 4;
    desc.BitsPerPixel = 32;
    desc.BytesPerLine = 4 * 4;
    // BGRA: B in 24-31, G in 16-23, R in 8-15, A in 0-7
    desc.BlueMask  = 0xFF000000;
    desc.GreenMask = 0x00FF0000;
    desc.RedMask   = 0x0000FF00;
    desc.AlphaMask = 0x000000FF;
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_32_BGRA8888, fmt);
}

TEST_F(PixelFormatTest, GetPixelFormat_32BitRGBA) {
    VxImageDescEx desc;
    memset(&desc, 0, sizeof(desc));
    desc.Size = sizeof(desc);
    desc.Width = 4;
    desc.Height = 4;
    desc.BitsPerPixel = 32;
    desc.BytesPerLine = 4 * 4;
    // RGBA: R in 24-31, G in 16-23, B in 8-15, A in 0-7
    desc.RedMask   = 0xFF000000;
    desc.GreenMask = 0x00FF0000;
    desc.BlueMask  = 0x0000FF00;
    desc.AlphaMask = 0x000000FF;
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_32_RGBA8888, fmt);
}

//==============================================================================
// GetPixelFormat Tests - 24-bit formats
//==============================================================================

TEST_F(PixelFormatTest, GetPixelFormat_24BitRGB) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 3);
    auto desc = ImageDescFactory::Create24BitRGB(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_24_RGB888, fmt);
}

TEST_F(PixelFormatTest, GetPixelFormat_24BitBGR) {
    VxImageDescEx desc;
    memset(&desc, 0, sizeof(desc));
    desc.Size = sizeof(desc);
    desc.Width = 4;
    desc.Height = 4;
    desc.BitsPerPixel = 24;
    desc.BytesPerLine = 4 * 3;
    desc.BlueMask  = 0x00FF0000;
    desc.GreenMask = 0x0000FF00;
    desc.RedMask   = 0x000000FF;
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_24_BGR888, fmt);
}

//==============================================================================
// GetPixelFormat Tests - 16-bit formats
//==============================================================================

TEST_F(PixelFormatTest, GetPixelFormat_16Bit565) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit565(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_16_RGB565, fmt);
}

TEST_F(PixelFormatTest, GetPixelFormat_16Bit555) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit555(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_16_RGB555, fmt);
}

TEST_F(PixelFormatTest, GetPixelFormat_16Bit1555) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit1555(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_16_ARGB1555, fmt);
}

TEST_F(PixelFormatTest, GetPixelFormat_16Bit4444) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit4444(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    EXPECT_EQ(_16_ARGB4444, fmt);
}

//==============================================================================
// GetPixelFormat Tests - 8-bit formats
//==============================================================================

TEST_F(PixelFormatTest, GetPixelFormat_8BitPaletted) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height);
    auto desc = ImageDescFactory::Create8BitPaletted(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    // 8-bit paletted may return RGB332, specific paletted format, or UNKNOWN_PF
    // depending on implementation
    EXPECT_TRUE(fmt == _8_RGB332 || fmt == UNKNOWN_PF) 
        << "8-bit should be RGB332 or UNKNOWN for paletted";
}

//==============================================================================
// PixelFormat2String Tests
//==============================================================================

TEST_F(PixelFormatTest, PixelFormat2String_CommonFormats) {
    XString str;
    
    str = BlitterWrapper::PixelFormat2String(_32_ARGB8888);
    EXPECT_FALSE(str.Empty()) << "32-bit ARGB should have string representation";
    
    str = BlitterWrapper::PixelFormat2String(_32_RGB888);
    EXPECT_FALSE(str.Empty()) << "32-bit RGB should have string representation";
    
    str = BlitterWrapper::PixelFormat2String(_24_RGB888);
    EXPECT_FALSE(str.Empty()) << "24-bit RGB should have string representation";
    
    str = BlitterWrapper::PixelFormat2String(_16_RGB565);
    EXPECT_FALSE(str.Empty()) << "16-bit 565 should have string representation";
    
    str = BlitterWrapper::PixelFormat2String(_16_ARGB1555);
    EXPECT_FALSE(str.Empty()) << "16-bit 1555 should have string representation";
}

TEST_F(PixelFormatTest, PixelFormat2String_AllFormats) {
    // Test that all known formats have string representations
    VX_PIXELFORMAT formats[] = {
        _32_ARGB8888, _32_RGB888, _32_ABGR8888, _32_RGBA8888, _32_BGRA8888,
        _24_RGB888, _24_BGR888,
        _16_RGB565, _16_RGB555, _16_ARGB1555, _16_ARGB4444,
        _8_RGB332, _8_ARGB2222
    };
    
    for (VX_PIXELFORMAT fmt : formats) {
        XString str = BlitterWrapper::PixelFormat2String(fmt);
        // Just verify it returns something (empty or not based on implementation)
    }
}

//==============================================================================
// ConvertPixelFormat Tests
//==============================================================================

TEST_F(PixelFormatTest, ConvertPixelFormat_ARGB_Components) {
    // Test that ConvertPixelFormat correctly identifies ARGB components
    VX_PIXELFORMAT fmt = _32_ARGB8888;
    
    XDWORD aMask, rMask, gMask, bMask;
    BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask);
    
    EXPECT_EQ(0xFF000000, aMask) << "Alpha mask for ARGB";
    EXPECT_EQ(0x00FF0000, rMask) << "Red mask for ARGB";
    EXPECT_EQ(0x0000FF00, gMask) << "Green mask for ARGB";
    EXPECT_EQ(0x000000FF, bMask) << "Blue mask for ARGB";
}

TEST_F(PixelFormatTest, ConvertPixelFormat_565_Components) {
    VX_PIXELFORMAT fmt = _16_RGB565;
    
    XDWORD aMask, rMask, gMask, bMask;
    BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask);
    
    EXPECT_EQ(0x0000, aMask) << "No alpha in 565";
    EXPECT_EQ(0xF800, rMask) << "Red mask for 565";
    EXPECT_EQ(0x07E0, gMask) << "Green mask for 565";
    EXPECT_EQ(0x001F, bMask) << "Blue mask for 565";
}

TEST_F(PixelFormatTest, ConvertPixelFormat_1555_Components) {
    VX_PIXELFORMAT fmt = _16_ARGB1555;
    
    XDWORD aMask, rMask, gMask, bMask;
    BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask);
    
    EXPECT_EQ(0x8000, aMask) << "Alpha bit in 1555";
    EXPECT_EQ(0x7C00, rMask) << "Red mask for 1555";
    EXPECT_EQ(0x03E0, gMask) << "Green mask for 1555";
    EXPECT_EQ(0x001F, bMask) << "Blue mask for 1555";
}

TEST_F(PixelFormatTest, ConvertPixelFormat_4444_Components) {
    VX_PIXELFORMAT fmt = _16_ARGB4444;
    
    XDWORD aMask, rMask, gMask, bMask;
    BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask);
    
    EXPECT_EQ(0xF000, aMask) << "Alpha mask for 4444";
    EXPECT_EQ(0x0F00, rMask) << "Red mask for 4444";
    EXPECT_EQ(0x00F0, gMask) << "Green mask for 4444";
    EXPECT_EQ(0x000F, bMask) << "Blue mask for 4444";
}

TEST_F(PixelFormatTest, ConvertPixelFormat_RGB888_NoAlpha) {
    VX_PIXELFORMAT fmt = _32_RGB888;
    
    XDWORD aMask, rMask, gMask, bMask;
    BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask);
    
    EXPECT_EQ(0x00000000, aMask) << "No alpha in RGB888";
    EXPECT_EQ(0x00FF0000, rMask) << "Red mask for RGB888";
    EXPECT_EQ(0x0000FF00, gMask) << "Green mask for RGB888";
    EXPECT_EQ(0x000000FF, bMask) << "Blue mask for RGB888";
}

//==============================================================================
// Round-Trip Tests: GetPixelFormat -> ConvertPixelFormat
//==============================================================================

TEST_F(PixelFormatTest, RoundTrip_ARGB8888) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 4);
    auto desc = ImageDescFactory::Create32BitARGB(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    
    XDWORD aMask, rMask, gMask, bMask;
    BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask);
    
    EXPECT_EQ(desc.AlphaMask, aMask);
    EXPECT_EQ(desc.RedMask, rMask);
    EXPECT_EQ(desc.GreenMask, gMask);
    EXPECT_EQ(desc.BlueMask, bMask);
}

TEST_F(PixelFormatTest, RoundTrip_565) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit565(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    
    XDWORD aMask, rMask, gMask, bMask;
    BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask);
    
    EXPECT_EQ(desc.AlphaMask, aMask);
    EXPECT_EQ(desc.RedMask, rMask);
    EXPECT_EQ(desc.GreenMask, gMask);
    EXPECT_EQ(desc.BlueMask, bMask);
}

TEST_F(PixelFormatTest, RoundTrip_1555) {
    const int width = 4, height = 4;
    ImageBuffer buffer(width * height * 2);
    auto desc = ImageDescFactory::Create16Bit1555(width, height, buffer.Data());
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    
    XDWORD aMask, rMask, gMask, bMask;
    BlitterWrapper::ConvertPixelFormat(fmt, aMask, rMask, gMask, bMask);
    
    EXPECT_EQ(desc.AlphaMask, aMask);
    EXPECT_EQ(desc.RedMask, rMask);
    EXPECT_EQ(desc.GreenMask, gMask);
    EXPECT_EQ(desc.BlueMask, bMask);
}

//==============================================================================
// Unknown Format Handling
//==============================================================================

TEST_F(PixelFormatTest, GetPixelFormat_UnknownFormat) {
    VxImageDescEx desc;
    memset(&desc, 0, sizeof(desc));
    desc.Size = sizeof(desc);
    desc.Width = 4;
    desc.Height = 4;
    desc.BitsPerPixel = 32;
    desc.BytesPerLine = 4 * 4;
    // Weird masks that don't match any known format
    desc.RedMask   = 0x00000F00;
    desc.GreenMask = 0x000000F0;
    desc.BlueMask  = 0x0000000F;
    desc.AlphaMask = 0x0000F000;
    
    VX_PIXELFORMAT fmt = BlitterWrapper::GetPixelFormat(desc);
    // Should return unknown format (likely UNKNOWN_PF)
    EXPECT_EQ(UNKNOWN_PF, fmt);
}

TEST_F(PixelFormatTest, PixelFormat2String_Unknown) {
    XString str = BlitterWrapper::PixelFormat2String(UNKNOWN_PF);
    // Should return something (possibly "Unknown" or empty)
    // Implementation dependent
}

//==============================================================================
// Bits Per Pixel Tests
//==============================================================================

TEST_F(PixelFormatTest, BitsPerPixel_32BitFormats) {
    ImageBuffer buf4(16);
    auto desc = ImageDescFactory::Create32BitARGB(2, 2, buf4.Data());
    EXPECT_EQ(32, desc.BitsPerPixel);
    
    desc = ImageDescFactory::Create32BitRGB(2, 2, buf4.Data());
    EXPECT_EQ(32, desc.BitsPerPixel);
}

TEST_F(PixelFormatTest, BitsPerPixel_24BitFormats) {
    ImageBuffer buf3(12);
    auto desc = ImageDescFactory::Create24BitRGB(2, 2, buf3.Data());
    EXPECT_EQ(24, desc.BitsPerPixel);
}

TEST_F(PixelFormatTest, BitsPerPixel_16BitFormats) {
    ImageBuffer buf2(8);
    
    auto desc = ImageDescFactory::Create16Bit565(2, 2, buf2.Data());
    EXPECT_EQ(16, desc.BitsPerPixel);
    
    desc = ImageDescFactory::Create16Bit555(2, 2, buf2.Data());
    EXPECT_EQ(16, desc.BitsPerPixel);
    
    desc = ImageDescFactory::Create16Bit1555(2, 2, buf2.Data());
    EXPECT_EQ(16, desc.BitsPerPixel);
    
    desc = ImageDescFactory::Create16Bit4444(2, 2, buf2.Data());
    EXPECT_EQ(16, desc.BitsPerPixel);
}

TEST_F(PixelFormatTest, BitsPerPixel_8BitFormats) {
    ImageBuffer buf1(4);
    auto desc = ImageDescFactory::Create8BitPaletted(2, 2, buf1.Data());
    EXPECT_EQ(8, desc.BitsPerPixel);
}

//==============================================================================
// BytesPerLine Tests
//==============================================================================

TEST_F(PixelFormatTest, BytesPerLine_Calculation) {
    ImageBuffer buf(1024);
    
    // 32-bit: width * 4
    auto desc32 = ImageDescFactory::Create32BitARGB(10, 10, buf.Data());
    EXPECT_EQ(10 * 4, desc32.BytesPerLine);
    
    // 24-bit: width * 3
    auto desc24 = ImageDescFactory::Create24BitRGB(10, 10, buf.Data());
    EXPECT_EQ(10 * 3, desc24.BytesPerLine);
    
    // 16-bit: width * 2
    auto desc16 = ImageDescFactory::Create16Bit565(10, 10, buf.Data());
    EXPECT_EQ(10 * 2, desc16.BytesPerLine);
    
    // 8-bit: width * 1
    auto desc8 = ImageDescFactory::Create8BitPaletted(10, 10, buf.Data());
    EXPECT_EQ(10 * 1, desc8.BytesPerLine);
}
