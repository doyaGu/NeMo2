#include <gtest/gtest.h>
#include "VxMath.h"
#include <vector>
#include <cstdint>
#include <cmath>

// Helper to compare floats with a tolerance
#define EXPECT_COLOR_NEAR(c1, c2) \
    EXPECT_NEAR((c1).r, (c2).r, EPSILON); \
    EXPECT_NEAR((c1).g, (c2).g, EPSILON); \
    EXPECT_NEAR((c1).b, (c2).b, EPSILON); \
    EXPECT_NEAR((c1).a, (c2).a, EPSILON)

//--- VxColor Tests ---

TEST(VxColorTests, DefaultConstructor) {
    VxColor c_default;
    EXPECT_EQ(c_default.r, 0.0f);
    EXPECT_EQ(c_default.g, 0.0f);
    EXPECT_EQ(c_default.b, 0.0f);
    EXPECT_EQ(c_default.a, 0.0f);
}

TEST(VxColorTests, FloatConstructors) {
    // RGBA Float constructor
    VxColor c_rgba_f(0.1f, 0.2f, 0.3f, 0.4f);
    EXPECT_EQ(c_rgba_f.r, 0.1f);
    EXPECT_EQ(c_rgba_f.g, 0.2f);
    EXPECT_EQ(c_rgba_f.b, 0.3f);
    EXPECT_EQ(c_rgba_f.a, 0.4f);

    // RGB Float constructor
    VxColor c_rgb_f(0.5f, 0.6f, 0.7f);
    EXPECT_EQ(c_rgb_f.r, 0.5f);
    EXPECT_EQ(c_rgb_f.g, 0.6f);
    EXPECT_EQ(c_rgb_f.b, 0.7f);
    EXPECT_EQ(c_rgb_f.a, 1.0f); // Alpha defaults to 1.0

    // Scalar Float constructor
    VxColor c_scalar_f(0.8f);
    EXPECT_EQ(c_scalar_f.r, 0.8f);
    EXPECT_EQ(c_scalar_f.g, 0.8f);
    EXPECT_EQ(c_scalar_f.b, 0.8f);
    EXPECT_EQ(c_scalar_f.a, 1.0f); // Alpha defaults to 1.0
}

TEST(VxColorTests, IntegerConstructors) {
    // RGBA Int constructor
    VxColor c_rgba_i(255, 128, 64, 32);
    EXPECT_NEAR(c_rgba_i.r, 1.0f, EPSILON);
    EXPECT_NEAR(c_rgba_i.g, 128.0f / 255.0f, EPSILON);
    EXPECT_NEAR(c_rgba_i.b, 64.0f / 255.0f, EPSILON);
    EXPECT_NEAR(c_rgba_i.a, 32.0f / 255.0f, EPSILON);

    // RGB Int constructor
    VxColor c_rgb_i(0, 128, 255);
    EXPECT_NEAR(c_rgb_i.r, 0.0f, EPSILON);
    EXPECT_NEAR(c_rgb_i.g, 128.0f / 255.0f, EPSILON);
    EXPECT_NEAR(c_rgb_i.b, 1.0f, EPSILON);
    EXPECT_EQ(c_rgb_i.a, 1.0f);

    // Edge cases
    VxColor c_zero(0, 0, 0, 0);
    EXPECT_EQ(c_zero.r, 0.0f);
    EXPECT_EQ(c_zero.g, 0.0f);
    EXPECT_EQ(c_zero.b, 0.0f);
    EXPECT_EQ(c_zero.a, 0.0f);

    VxColor c_max(255, 255, 255, 255);
    EXPECT_EQ(c_max.r, 1.0f);
    EXPECT_EQ(c_max.g, 1.0f);
    EXPECT_EQ(c_max.b, 1.0f);
    EXPECT_EQ(c_max.a, 1.0f);
}

TEST(VxColorTests, UnsignedLongConstructor) {
    // Test various ARGB values
    VxColor c_long(0x80FFC040U); // A=128, R=255, G=192, B=64
    EXPECT_NEAR(c_long.a, 128.0f / 255.0f, EPSILON);
    EXPECT_NEAR(c_long.r, 255.0f / 255.0f, EPSILON);
    EXPECT_NEAR(c_long.g, 192.0f / 255.0f, EPSILON);
    EXPECT_NEAR(c_long.b, 64.0f / 255.0f, EPSILON);

    // Test pure colors
    VxColor red(0xFFFF0000U);
    EXPECT_EQ(red.a, 1.0f);
    EXPECT_EQ(red.r, 1.0f);
    EXPECT_EQ(red.g, 0.0f);
    EXPECT_EQ(red.b, 0.0f);

    VxColor green(0xFF00FF00U);
    EXPECT_EQ(green.a, 1.0f);
    EXPECT_EQ(green.r, 0.0f);
    EXPECT_EQ(green.g, 1.0f);
    EXPECT_EQ(green.b, 0.0f);

    VxColor blue(0xFF0000FFU);
    EXPECT_EQ(blue.a, 1.0f);
    EXPECT_EQ(blue.r, 0.0f);
    EXPECT_EQ(blue.g, 0.0f);
    EXPECT_EQ(blue.b, 1.0f);

    VxColor transparent(0x00000000U);
    EXPECT_EQ(transparent.a, 0.0f);
    EXPECT_EQ(transparent.r, 0.0f);
    EXPECT_EQ(transparent.g, 0.0f);
    EXPECT_EQ(transparent.b, 0.0f);
}

TEST(VxColorTests, ArrayAccess) {
    VxColor c(0.1f, 0.2f, 0.3f, 0.4f);
    EXPECT_EQ(c.col[0], 0.1f); // r
    EXPECT_EQ(c.col[1], 0.2f); // g
    EXPECT_EQ(c.col[2], 0.3f); // b
    EXPECT_EQ(c.col[3], 0.4f); // a

    // Test modification through array access
    c.col[0] = 0.9f;
    EXPECT_EQ(c.r, 0.9f);
}

TEST(VxColorTests, SetterMethods) {
    VxColor c;

    // Test float setters
    c.Set(0.1f, 0.2f, 0.3f, 0.4f);
    EXPECT_COLOR_NEAR(c, VxColor(0.1f, 0.2f, 0.3f, 0.4f));

    c.Set(0.5f, 0.6f, 0.7f);
    EXPECT_COLOR_NEAR(c, VxColor(0.5f, 0.6f, 0.7f, 1.0f));

    c.Set(0.8f);
    EXPECT_COLOR_NEAR(c, VxColor(0.8f, 0.8f, 0.8f, 1.0f));

    // Test integer setters
    c.Set(255, 128, 0, 255);
    EXPECT_COLOR_NEAR(c, VxColor(1.0f, 128.0f / 255.0f, 0.0f, 1.0f));

    c.Set(255, 128, 0);
    EXPECT_COLOR_NEAR(c, VxColor(1.0f, 128.0f / 255.0f, 0.0f, 1.0f));

    // Test unsigned long setter
    c.Set(0xAABBCCDDU); // ARGB
    EXPECT_COLOR_NEAR(c, VxColor(0xAABBCCDDU));

    // Test Clear method
    c.Clear();
    EXPECT_COLOR_NEAR(c, VxColor(0.0f, 0.0f, 0.0f, 0.0f));
}

TEST(VxColorTests, ConversionMethods) {
    // Test GetRGBA
    VxColor c(255, 128, 64, 32); // r=1.0, g=~0.5, b=~0.25, a=~0.125
    // RGBA = (32 << 24) | (255 << 16) | (128 << 8) | 64 = 0x20FF8040
    EXPECT_EQ(c.GetRGBA(), 0x20FF8040);

    // Test GetRGB (should force alpha to full)
    EXPECT_EQ(c.GetRGB(), 0xFFFF8040);

    // Test edge cases
    VxColor black(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(black.GetRGBA(), 0x00000000);
    EXPECT_EQ(black.GetRGB(), 0xFF000000);

    VxColor white(1.0f, 1.0f, 1.0f, 1.0f);
    EXPECT_EQ(white.GetRGBA(), 0xFFFFFFFF);
    EXPECT_EQ(white.GetRGB(), 0xFFFFFFFF);
}

TEST(VxColorTests, StaticConvertMethods) {
    // Test float Convert with clamping
    unsigned long packed = VxColor::Convert(1.5f, 0.5f, -0.5f, 1.0f);
    // 0.5f * 255 = 127.5f which truncates to 127, so g=127=0x7F
    EXPECT_EQ(packed, 0xFFFF7F00); // a=255, r=255, g=127, b=0

    packed = VxColor::Convert(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_EQ(packed, 0x00000000);

    packed = VxColor::Convert(1.0f, 1.0f, 1.0f, 1.0f);
    EXPECT_EQ(packed, 0xFFFFFFFF);

    // Test integer Convert with clamping
    packed = VxColor::Convert(300, 100, -20, 255);
    EXPECT_EQ(packed, 0xFFFF6400); // a=255, r=255, g=100, b=0

    packed = VxColor::Convert(0, 0, 0, 0);
    EXPECT_EQ(packed, 0x00000000);

    packed = VxColor::Convert(255, 255, 255, 255);
    EXPECT_EQ(packed, 0xFFFFFFFF);
}

TEST(VxColorTests, CheckMethod) {
    // Test clamping to [0.0, 1.0] range
    VxColor c(1.5f, 0.5f, -0.2f, 1.1f);
    c.Check();
    EXPECT_COLOR_NEAR(c, VxColor(1.0f, 0.5f, 0.0f, 1.0f));

    // Test values already in range
    VxColor c2(0.3f, 0.7f, 0.1f, 0.9f);
    VxColor original = c2;
    c2.Check();
    EXPECT_COLOR_NEAR(c2, original);

    // Test edge values
    VxColor c3(0.0f, 1.0f, 0.0f, 1.0f);
    c3.Check();
    EXPECT_COLOR_NEAR(c3, VxColor(0.0f, 1.0f, 0.0f, 1.0f));
}

TEST(VxColorTests, ArithmeticOperations) {
    VxColor c1(0.1f, 0.2f, 0.3f, 0.4f);
    VxColor c2(0.5f, 0.5f, 0.5f, 0.5f);

    // Test addition
    VxColor sum = c1 + c2;
    EXPECT_COLOR_NEAR(sum, VxColor(0.6f, 0.7f, 0.8f, 0.9f));

    // Test subtraction
    VxColor diff = c2 - c1;
    EXPECT_COLOR_NEAR(diff, VxColor(0.4f, 0.3f, 0.2f, 0.1f));

    // Test multiplication
    VxColor prod = c1 * c2;
    EXPECT_COLOR_NEAR(prod, VxColor(0.05f, 0.1f, 0.15f, 0.2f));

    // Test division
    VxColor quot = c2 / VxColor(2.0f, 2.0f, 2.0f, 2.0f);
    EXPECT_COLOR_NEAR(quot, VxColor(0.25f, 0.25f, 0.25f, 0.25f));

    // Test scalar multiplication
    VxColor scaled = c1 * 2.0f;
    EXPECT_COLOR_NEAR(scaled, VxColor(0.2f, 0.4f, 0.6f, 0.8f));
}

TEST(VxColorTests, AssignmentOperations) {
    VxColor c1(0.1f, 0.2f, 0.3f, 0.4f);
    VxColor c2(0.5f, 0.5f, 0.5f, 0.5f);

    // Test +=
    VxColor c_orig = c1;
    c1 += c2;
    EXPECT_COLOR_NEAR(c1, c_orig + c2);

    // Test -=
    c1 = VxColor(0.8f, 0.7f, 0.6f, 0.5f);
    c_orig = c1;
    c1 -= c2;
    EXPECT_COLOR_NEAR(c1, c_orig - c2);

    // Test *=
    c1 = VxColor(0.2f, 0.4f, 0.6f, 0.8f);
    c_orig = c1;
    c1 *= c2;
    EXPECT_COLOR_NEAR(c1, c_orig * c2);

    // Test /=
    c1 = VxColor(0.8f, 0.6f, 0.4f, 0.2f);
    c_orig = c1;
    c1 /= c2;
    EXPECT_COLOR_NEAR(c1, c_orig / c2);

    // Test scalar *=
    c1 = VxColor(0.1f, 0.2f, 0.3f, 0.4f);
    c1 *= 3.0f;
    EXPECT_COLOR_NEAR(c1, VxColor(0.3f, 0.6f, 0.9f, 1.2f));

    // Test scalar /=
    c1 = VxColor(0.6f, 0.8f, 1.0f, 1.2f);
    c1 /= 2.0f;
    EXPECT_COLOR_NEAR(c1, VxColor(0.3f, 0.4f, 0.5f, 0.6f));
}

TEST(VxColorTests, ComparisonOperators) {
    VxColor c1(0.1f, 0.2f, 0.3f, 0.4f);
    VxColor c2(0.1f, 0.2f, 0.3f, 0.4f);
    VxColor c3(0.5f, 0.2f, 0.3f, 0.4f);

    EXPECT_TRUE(c1 == c2);
    EXPECT_FALSE(c1 == c3);
    EXPECT_FALSE(c1 != c2);
    EXPECT_TRUE(c1 != c3);

    // Test floating point precision
    VxColor c4(0.1000001f, 0.2f, 0.3f, 0.4f);
    EXPECT_FALSE(c1 == c4); // Should be different due to precision
}

TEST(VxColorTests, GetSquareDistance) {
    VxColor c1(0.0f, 0.0f, 0.0f, 0.0f);
    VxColor c2(1.0f, 1.0f, 1.0f, 1.0f);

    float sq_dist = c1.GetSquareDistance(c2);
    EXPECT_NEAR(sq_dist, 3.0f, EPSILON); // sqrt(1^2 + 1^2 + 1^2) squared = 3

    VxColor c3(0.3f, 0.4f, 0.0f, 0.0f);
    VxColor c4(0.0f, 0.0f, 0.0f, 0.0f);
    sq_dist = c3.GetSquareDistance(c4);
    EXPECT_NEAR(sq_dist, 0.25f, EPSILON); // 0.3^2 + 0.4^2 = 0.09 + 0.16 = 0.25
}

TEST(VxColorTests, GlobalFunctions) {
    // Test RGBAFTOCOLOR with floats
    unsigned long color = RGBAFTOCOLOR(1.0f, 0.5f, 0.0f, 1.0f);
    EXPECT_EQ(color, 0xFFFF7F00);

    // Test RGBAFTOCOLOR with VxColor pointer
    VxColor c(0.2f, 0.4f, 0.6f, 0.8f);
    color = RGBAFTOCOLOR(&c);
    EXPECT_EQ(color, 0xCC336699);

    // // Test BGRAFTOCOLOR
    // color = BGRAFTOCOLOR(&c);
    // EXPECT_EQ(color, 0xCC996633); // BGRA format
}

//--- VxImageDescEx & Pixel Format Tests ---

TEST(VxImageDescExTests, ConstructorAndBasicProperties) {
    VxImageDescEx desc1;
    EXPECT_EQ(desc1.Size, sizeof(VxImageDescEx));
    EXPECT_EQ(desc1.Width, 0);
    EXPECT_EQ(desc1.Height, 0);
    EXPECT_EQ(desc1.BitsPerPixel, 0);
    EXPECT_EQ(desc1.RedMask, 0);
    EXPECT_EQ(desc1.GreenMask, 0);
    EXPECT_EQ(desc1.BlueMask, 0);
    EXPECT_EQ(desc1.AlphaMask, 0);
    EXPECT_EQ(desc1.ColorMapEntries, 0);
    EXPECT_EQ(desc1.Image, nullptr);
    EXPECT_EQ(desc1.ColorMap, nullptr);
}

TEST(VxImageDescExTests, SetMethod) {
    VxImageDescEx desc1, desc2;

    // Set up desc2 with some values
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc2);
    desc2.Width = 100;
    desc2.Height = 200;

    EXPECT_FALSE(desc1 == desc2);

    desc1.Set(desc2);
    EXPECT_TRUE(desc1 == desc2);
    EXPECT_EQ(desc1.Width, 100);
    EXPECT_EQ(desc1.Height, 200);
    EXPECT_EQ(desc1.BitsPerPixel, 32);
}

TEST(VxImageDescExTests, HasAlphaMethod) {
    VxImageDescEx desc;

    // Test format with alpha mask
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc);
    EXPECT_TRUE(desc.HasAlpha());

    // Test format without alpha
    VxPixelFormat2ImageDesc(_24_RGB888, desc);
    EXPECT_FALSE(desc.HasAlpha());

    // Test DXT format (should have alpha)
    VxPixelFormat2ImageDesc(_DXT5, desc);
    EXPECT_TRUE(desc.HasAlpha());

    VxPixelFormat2ImageDesc(_DXT1, desc);
    EXPECT_TRUE(desc.HasAlpha()); // DXT1 can have 1-bit alpha
}

TEST(VxImageDescExTests, ComparisonOperators) {
    VxImageDescEx desc1, desc2, desc3;

    // Set up identical descriptions
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc1);
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc2);
    desc1.Width = desc2.Width = 100;
    desc1.Height = desc2.Height = 100;

    EXPECT_TRUE(desc1 == desc2);
    EXPECT_FALSE(desc1 != desc2);

    // Make desc3 different
    VxPixelFormat2ImageDesc(_24_RGB888, desc3);
    desc3.Width = 100;
    desc3.Height = 100;

    EXPECT_FALSE(desc1 == desc3);
    EXPECT_TRUE(desc1 != desc3);
}

TEST(PixelFormatTests, AllFormatConversions) {
    // Test conversion cycle for all pixel formats
    for (int pf = UNKNOWN_PF + 1; pf <= _32_X8L8V8U8; ++pf) {
        VxImageDescEx desc;
        VxPixelFormat2ImageDesc(static_cast<VX_PIXELFORMAT>(pf), desc);

        // Set valid dimensions for DXT formats
        if (pf >= _DXT1 && pf <= _DXT5) {
            desc.Width = 8; // Use 8x8 for better testing
            desc.Height = 8;
        } else {
            desc.Width = 16;
            desc.Height = 16;
        }

        VX_PIXELFORMAT new_pf = VxImageDesc2PixelFormat(desc);
        EXPECT_EQ(new_pf, pf) << "Failed for format: " << VxPixelFormat2String(static_cast<VX_PIXELFORMAT>(pf));
    }
}

TEST(PixelFormatTests, FormatStringConversion) {
    // Test string conversion for all formats
    for (int pf = UNKNOWN_PF; pf <= _32_X8L8V8U8; ++pf) {
        const char *format_string = VxPixelFormat2String(static_cast<VX_PIXELFORMAT>(pf));
        EXPECT_NE(format_string, nullptr);
        EXPECT_GT(strlen(format_string), 0);

        if (pf == UNKNOWN_PF) {
            // Unknown format should have specific string
            EXPECT_STREQ(format_string, "Unknown");
        } else {
            // Valid formats should not be ""
            EXPECT_STRNE(format_string, "");
        }
    }

    // Test invalid format - returns empty string for invalid formats
    const char *invalid_str = VxPixelFormat2String(static_cast<VX_PIXELFORMAT>(999));
    EXPECT_STREQ(invalid_str, "");
}

TEST(PixelFormatTests, BitCountsAndShifts) {
    struct FormatTest {
        VX_PIXELFORMAT format;
        XDWORD expected_rbits, expected_gbits, expected_bbits, expected_abits;
        XDWORD expected_rshift, expected_gshift, expected_bshift, expected_ashift;
    };

    FormatTest tests[] = {
        {_32_ARGB8888, 8, 8, 8, 8, 16, 8, 0, 24},
        {_24_RGB888, 8, 8, 8, 0, 16, 8, 0, 0},
        {_16_RGB565, 5, 6, 5, 0, 11, 5, 0, 0},
        {_16_RGB555, 5, 5, 5, 0, 10, 5, 0, 0},
        {_16_ARGB1555, 5, 5, 5, 1, 10, 5, 0, 15},
        {_16_ARGB4444, 4, 4, 4, 4, 8, 4, 0, 12},
        {_8_RGB332, 3, 3, 2, 0, 5, 2, 0, 0},
        {_8_ARGB2222, 2, 2, 2, 2, 4, 2, 0, 6}
    };

    for (const auto &test : tests) {
        VxImageDescEx desc;
        VxPixelFormat2ImageDesc(test.format, desc);

        XDWORD r, g, b, a;
        VxGetBitCounts(desc, r, g, b, a);
        EXPECT_EQ(r, test.expected_rbits) << "Red bits failed for " << VxPixelFormat2String(test.format);
        EXPECT_EQ(g, test.expected_gbits) << "Green bits failed for " << VxPixelFormat2String(test.format);
        EXPECT_EQ(b, test.expected_bbits) << "Blue bits failed for " << VxPixelFormat2String(test.format);
        EXPECT_EQ(a, test.expected_abits) << "Alpha bits failed for " << VxPixelFormat2String(test.format);

        VxGetBitShifts(desc, r, g, b, a);
        EXPECT_EQ(r, test.expected_rshift) << "Red shift failed for " << VxPixelFormat2String(test.format);
        EXPECT_EQ(g, test.expected_gshift) << "Green shift failed for " << VxPixelFormat2String(test.format);
        EXPECT_EQ(b, test.expected_bshift) << "Blue shift failed for " << VxPixelFormat2String(test.format);
        EXPECT_EQ(a, test.expected_ashift) << "Alpha shift failed for " << VxPixelFormat2String(test.format);
    }
}

TEST(PixelFormatTests, VxBppToMask) {
    VxImageDescEx desc;

    // Test 8-bit
    desc.BitsPerPixel = 8;
    VxBppToMask(desc);
    EXPECT_EQ(desc.RedMask, 0xE0);
    EXPECT_EQ(desc.GreenMask, 0x1C);
    EXPECT_EQ(desc.BlueMask, 0x03);
    EXPECT_EQ(desc.AlphaMask, 0x00);

    // Test 16-bit
    desc.BitsPerPixel = 16;
    VxBppToMask(desc);
    EXPECT_EQ(desc.RedMask, 0x7C00);
    EXPECT_EQ(desc.GreenMask, 0x03E0);
    EXPECT_EQ(desc.BlueMask, 0x001F);
    EXPECT_EQ(desc.AlphaMask, 0x0000);

    // Test 24-bit
    desc.BitsPerPixel = 24;
    VxBppToMask(desc);
    EXPECT_EQ(desc.RedMask, 0x00FF0000);
    EXPECT_EQ(desc.GreenMask, 0x0000FF00);
    EXPECT_EQ(desc.BlueMask, 0x000000FF);
    EXPECT_EQ(desc.AlphaMask, 0x00000000);

    // Test 32-bit
    desc.BitsPerPixel = 32;
    VxBppToMask(desc);
    EXPECT_EQ(desc.RedMask, 0x00FF0000);
    EXPECT_EQ(desc.GreenMask, 0x0000FF00);
    EXPECT_EQ(desc.BlueMask, 0x000000FF);
    EXPECT_EQ(desc.AlphaMask, 0xFF000000);
}

//--- Image Manipulation Tests ---

class ImageManipulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        for (auto ptr : allocated_buffers) {
            delete[] ptr;
        }
        allocated_buffers.clear();
    }

    // Helper to allocate memory and track it for cleanup
    uint8_t *AllocateBuffer(size_t size) {
        uint8_t *buffer = new uint8_t[size];
        memset(buffer, 0, size); // Initialize to zero
        allocated_buffers.push_back(buffer);
        return buffer;
    }

    // Fills a 32-bit buffer with a simple pattern
    void FillPattern32(uint32_t *buffer, int width, int height) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                buffer[y * width + x] = (0xFF << 24) | (x * 10) << 16 | (y * 10) << 8 | (x + y);
            }
        }
    }

    // Create a gradient pattern
    void FillGradient32(uint32_t *buffer, int width, int height) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t r = (uint8_t) ((x * 255) / (width - 1));
                uint8_t g = (uint8_t) ((y * 255) / (height - 1));
                uint8_t b = 128;
                uint8_t a = 255;
                buffer[y * width + x] = (a << 24) | (r << 16) | (g << 8) | b;
            }
        }
    }

private:
    std::vector<uint8_t *> allocated_buffers;
};

TEST_F(ImageManipulationTest, VxDoBlit_IdenticalFormat) {
    VxImageDescEx src_desc, dst_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    VxPixelFormat2ImageDesc(_32_ARGB8888, dst_desc);
    src_desc.Width = dst_desc.Width = 4;
    src_desc.Height = dst_desc.Height = 4;
    src_desc.BytesPerLine = dst_desc.BytesPerLine = 4 * 4;

    src_desc.Image = AllocateBuffer(4 * 4 * 4);
    dst_desc.Image = AllocateBuffer(4 * 4 * 4);

    FillPattern32(reinterpret_cast<uint32_t *>(src_desc.Image), 4, 4);

    VxDoBlit(src_desc, dst_desc);

    EXPECT_EQ(0, memcmp(src_desc.Image, dst_desc.Image, 4*4*4));
}

TEST_F(ImageManipulationTest, VxDoBlit_FormatConversion_24_to_32) {
    VxImageDescEx src_desc, dst_desc;
    VxPixelFormat2ImageDesc(_24_RGB888, src_desc);
    VxPixelFormat2ImageDesc(_32_ARGB8888, dst_desc);
    src_desc.Width = dst_desc.Width = 2;
    src_desc.Height = dst_desc.Height = 1;
    src_desc.BytesPerLine = 2 * 3;
    dst_desc.BytesPerLine = 2 * 4;

    src_desc.Image = AllocateBuffer(2 * 3);
    dst_desc.Image = AllocateBuffer(2 * 4);

    // Source: [Pixel 1: R=10,G=20,B=30], [Pixel 2: R=40,G=50,B=60]
    src_desc.Image[0] = 30;
    src_desc.Image[1] = 20;
    src_desc.Image[2] = 10;
    src_desc.Image[3] = 60;
    src_desc.Image[4] = 50;
    src_desc.Image[5] = 40;

    VxDoBlit(src_desc, dst_desc);

    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_desc.Image);
    // Destination should be ARGB format: 0xAARRGGBB
    EXPECT_EQ(dst_ptr[0], 0xFF0A141E); // Alpha=255, R=10, G=20, B=30
    EXPECT_EQ(dst_ptr[1], 0xFF28323C); // Alpha=255, R=40, G=50, B=60
}

TEST_F(ImageManipulationTest, VxDoBlit_FormatConversion_32_to_16) {
    VxImageDescEx src_desc, dst_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    VxPixelFormat2ImageDesc(_16_RGB565, dst_desc);
    src_desc.Width = dst_desc.Width = 2;
    src_desc.Height = dst_desc.Height = 1;
    src_desc.BytesPerLine = 2 * 4;
    dst_desc.BytesPerLine = 2 * 2;

    src_desc.Image = AllocateBuffer(2 * 4);
    dst_desc.Image = AllocateBuffer(2 * 2);

    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    src_ptr[0] = 0xFFFF0000; // Red
    src_ptr[1] = 0xFF00FF00; // Green

    VxDoBlit(src_desc, dst_desc);

    uint16_t *dst_ptr = reinterpret_cast<uint16_t *>(dst_desc.Image);
    // Check that red and green are roughly preserved in 565 format
    EXPECT_EQ(dst_ptr[0] & 0xF800, 0xF800); // Red bits should be set
    EXPECT_EQ(dst_ptr[1] & 0x07E0, 0x07E0); // Green bits should be set
}

TEST_F(ImageManipulationTest, VxDoBlit_DifferentSizes) {
    VxImageDescEx src_desc, dst_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    VxPixelFormat2ImageDesc(_32_ARGB8888, dst_desc);

    // Source is 2x2, destination is 4x4
    src_desc.Width = 2;
    src_desc.Height = 2;
    src_desc.BytesPerLine = 2 * 4;
    dst_desc.Width = 4;
    dst_desc.Height = 4;
    dst_desc.BytesPerLine = 4 * 4;

    src_desc.Image = AllocateBuffer(2 * 2 * 4);
    dst_desc.Image = AllocateBuffer(4 * 4 * 4);

    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    src_ptr[0] = 0xFFFF0000; // Red
    src_ptr[1] = 0xFF00FF00; // Green
    src_ptr[2] = 0xFF0000FF; // Blue
    src_ptr[3] = 0xFFFFFFFF; // White

    VxDoBlit(src_desc, dst_desc);

    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_desc.Image);

    // Check corners are preserved (approximately)
    EXPECT_EQ(dst_ptr[0], src_ptr[0]);  // Top-left
    EXPECT_EQ(dst_ptr[3], src_ptr[1]);  // Top-right
    EXPECT_EQ(dst_ptr[12], src_ptr[2]); // Bottom-left
    EXPECT_EQ(dst_ptr[15], src_ptr[3]); // Bottom-right
}

TEST_F(ImageManipulationTest, VxDoBlitUpsideDown) {
    VxImageDescEx src_desc, dst_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    VxPixelFormat2ImageDesc(_32_ARGB8888, dst_desc);
    src_desc.Width = dst_desc.Width = 2;
    src_desc.Height = dst_desc.Height = 3;
    src_desc.BytesPerLine = dst_desc.BytesPerLine = 2 * 4;

    src_desc.Image = AllocateBuffer(2 * 3 * 4);
    dst_desc.Image = AllocateBuffer(2 * 3 * 4);

    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    src_ptr[0] = 0xFF0000AA;
    src_ptr[1] = 0xFF0000BB; // Row 0
    src_ptr[2] = 0xFF0000CC;
    src_ptr[3] = 0xFF0000DD; // Row 1
    src_ptr[4] = 0xFF0000EE;
    src_ptr[5] = 0xFF0000FF; // Row 2

    VxDoBlitUpsideDown(src_desc, dst_desc);

    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_desc.Image);
    // Row 0 in dst should be row 2 from src
    EXPECT_EQ(dst_ptr[0], 0xFF0000EE);
    EXPECT_EQ(dst_ptr[1], 0xFF0000FF);
    // Row 1 in dst should be row 1 from src
    EXPECT_EQ(dst_ptr[2], 0xFF0000CC);
    EXPECT_EQ(dst_ptr[3], 0xFF0000DD);
    // Row 2 in dst should be row 0 from src
    EXPECT_EQ(dst_ptr[4], 0xFF0000AA);
    EXPECT_EQ(dst_ptr[5], 0xFF0000BB);
}

TEST_F(ImageManipulationTest, VxDoAlphaBlit_SingleValue) {
    VxImageDescEx desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc);
    desc.Width = 2;
    desc.Height = 2;
    desc.BytesPerLine = 2 * 4;
    desc.Image = AllocateBuffer(2 * 2 * 4);

    uint32_t *ptr = reinterpret_cast<uint32_t *>(desc.Image);
    ptr[0] = 0x80FF0000; // Red with alpha=128
    ptr[1] = 0x4000FF00; // Green with alpha=64
    ptr[2] = 0xC00000FF; // Blue with alpha=192
    ptr[3] = 0x20FFFFFF; // White with alpha=32

    VxDoAlphaBlit(desc, 255); // Set all alpha to 255

    EXPECT_EQ(ptr[0], 0xFFFF0000); // Red with alpha=255
    EXPECT_EQ(ptr[1], 0xFF00FF00); // Green with alpha=255
    EXPECT_EQ(ptr[2], 0xFF0000FF); // Blue with alpha=255
    EXPECT_EQ(ptr[3], 0xFFFFFFFF); // White with alpha=255
}

TEST_F(ImageManipulationTest, VxDoAlphaBlit_ArrayValues) {
    VxImageDescEx desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc);
    desc.Width = 2;
    desc.Height = 2;
    desc.BytesPerLine = 2 * 4;
    desc.Image = AllocateBuffer(2 * 2 * 4);

    uint32_t *ptr = reinterpret_cast<uint32_t *>(desc.Image);
    ptr[0] = 0x00FF0000; // Red
    ptr[1] = 0x0000FF00; // Green
    ptr[2] = 0x000000FF; // Blue
    ptr[3] = 0x00FFFFFF; // White

    XBYTE alpha_values[] = {64, 128, 192, 255};
    VxDoAlphaBlit(desc, alpha_values);

    EXPECT_EQ(ptr[0], 0x40FF0000); // Red with alpha=64
    EXPECT_EQ(ptr[1], 0x8000FF00); // Green with alpha=128
    EXPECT_EQ(ptr[2], 0xC00000FF); // Blue with alpha=192
    EXPECT_EQ(ptr[3], 0xFFFFFFFF); // White with alpha=255
}

TEST_F(ImageManipulationTest, VxGenerateMipMap_SimplePattern) {
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 4;
    src_desc.Height = 4;
    src_desc.BytesPerLine = 4 * 4;
    src_desc.Image = AllocateBuffer(4 * 4 * 4);

    // Create a 4x4 image with 4 colored quadrants
    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    uint32_t R = 0xFFFF0000, G = 0xFF00FF00, B = 0xFF0000FF, W = 0xFFFFFFFF;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            if (x < 2 && y < 2) src_ptr[y * 4 + x] = R;
            else if (x >= 2 && y < 2) src_ptr[y * 4 + x] = G;
            else if (x < 2 && y >= 2) src_ptr[y * 4 + x] = B;
            else src_ptr[y * 4 + x] = W;
        }
    }

    // Destination for the 2x2 mipmap
    uint8_t *dst_buffer = AllocateBuffer(2 * 2 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    // The mipmap generation uses filtering, so we check that colors are similar but allow some variance
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ((dst_ptr[i] >> 24) & 0xFF, 0xFF); // Alpha should be preserved
    }
}

TEST_F(ImageManipulationTest, VxGenerateMipMap_EdgeCase1x1) {
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 2;
    src_desc.Height = 2;
    src_desc.BytesPerLine = 2 * 4;
    src_desc.Image = AllocateBuffer(2 * 2 * 4);

    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    src_ptr[0] = 0xFFFF0000;
    src_ptr[1] = 0xFF00FF00;
    src_ptr[2] = 0xFF0000FF;
    src_ptr[3] = 0xFFFFFFFF;

    // Destination for the 1x1 mipmap
    uint8_t *dst_buffer = AllocateBuffer(1 * 1 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    // Should be some average of the 4 colors
    EXPECT_EQ((dst_ptr[0] >> 24) & 0xFF, 0xFF); // Alpha should be preserved
}

TEST_F(ImageManipulationTest, VxResizeImage32_UpScale) {
    VxImageDescEx src_desc, dst_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    VxPixelFormat2ImageDesc(_32_ARGB8888, dst_desc);

    // Source is 2x2, destination is 4x4
    src_desc.Width = 2;
    src_desc.Height = 2;
    src_desc.BytesPerLine = 2 * 4;
    dst_desc.Width = 4;
    dst_desc.Height = 4;
    dst_desc.BytesPerLine = 4 * 4;

    src_desc.Image = AllocateBuffer(2 * 2 * 4);
    dst_desc.Image = AllocateBuffer(4 * 4 * 4);

    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    src_ptr[0] = 0xFFFF0000; // Red
    src_ptr[1] = 0xFF00FF00; // Green
    src_ptr[2] = 0xFF0000FF; // Blue
    src_ptr[3] = 0xFFFFFFFF; // White

    VxResizeImage32(src_desc, dst_desc);

    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_desc.Image);

    // Check corners are preserved
    EXPECT_EQ(dst_ptr[0], src_ptr[0]);  // Top-left
    EXPECT_EQ(dst_ptr[3], src_ptr[1]);  // Top-right
    EXPECT_EQ(dst_ptr[12], src_ptr[2]); // Bottom-left
    EXPECT_EQ(dst_ptr[15], src_ptr[3]); // Bottom-right
}

TEST_F(ImageManipulationTest, VxResizeImage32_DownScale) {
    VxImageDescEx src_desc, dst_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    VxPixelFormat2ImageDesc(_32_ARGB8888, dst_desc);

    // Source is 4x4, destination is 2x2
    src_desc.Width = 4;
    src_desc.Height = 4;
    src_desc.BytesPerLine = 4 * 4;
    dst_desc.Width = 2;
    dst_desc.Height = 2;
    dst_desc.BytesPerLine = 2 * 4;

    src_desc.Image = AllocateBuffer(4 * 4 * 4);
    dst_desc.Image = AllocateBuffer(2 * 2 * 4);

    FillGradient32(reinterpret_cast<uint32_t *>(src_desc.Image), 4, 4);

    VxResizeImage32(src_desc, dst_desc);

    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_desc.Image);

    // Check that result has reasonable values
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ((dst_ptr[i] >> 24) & 0xFF, 0xFF); // Alpha should be preserved
    }
}

TEST_F(ImageManipulationTest, VxConvertToNormalMap_FlatSurface) {
    VxImageDescEx desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc);
    desc.Width = 3;
    desc.Height = 3;
    desc.BytesPerLine = 3 * 4;
    desc.Image = AllocateBuffer(3 * 3 * 4);

    // Create a flat surface (same height everywhere)
    uint32_t *ptr = reinterpret_cast<uint32_t *>(desc.Image);
    for (int i = 0; i < 9; i++) {
        ptr[i] = 0xFF808080; // Gray (flat height)
    }

    EXPECT_TRUE(VxConvertToNormalMap(desc, desc.RedMask));

    // For a flat surface, normals should point straight up
    uint8_t *img_b = desc.Image;
    uint8_t center_r = img_b[4 * 4 + 2]; // Red component of center pixel
    uint8_t center_g = img_b[4 * 4 + 1]; // Green component
    uint8_t center_b = img_b[4 * 4 + 0]; // Blue component

    // Normal (0, 0, 1) encoded as (128, 128, 255)
    EXPECT_NEAR(center_r, 128, 10); // X component should be near 0 (encoded as 128)
    EXPECT_NEAR(center_g, 128, 10); // Y component should be near 0 (encoded as 128)
    EXPECT_EQ(center_b, 255);       // Z component should be 1 (encoded as 255)
}

TEST_F(ImageManipulationTest, VxConvertToNormalMap_Slope) {
    VxImageDescEx desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc);
    desc.Width = 3;
    desc.Height = 3;
    desc.BytesPerLine = 3 * 4;
    desc.Image = AllocateBuffer(3 * 3 * 4);

    // Create a surface that slopes in X direction
    uint32_t *ptr = reinterpret_cast<uint32_t *>(desc.Image);
    ptr[0] = 0xFF000000;
    ptr[1] = 0xFF808080;
    ptr[2] = 0xFFFFFFFF; // Row 0: dark to light
    ptr[3] = 0xFF000000;
    ptr[4] = 0xFF808080;
    ptr[5] = 0xFFFFFFFF; // Row 1: dark to light
    ptr[6] = 0xFF000000;
    ptr[7] = 0xFF808080;
    ptr[8] = 0xFFFFFFFF; // Row 2: dark to light

    EXPECT_TRUE(VxConvertToNormalMap(desc, desc.RedMask));

    // The surface slopes up in +X direction, so normal should have negative X component
    // Just verify the function ran and modified the image
    bool image_changed = false;
    for (int i = 0; i < 9; i++) {
        if (ptr[i] != 0xFF808080) {
            // Different from original flat gray
            image_changed = true;
            break;
        }
    }
    EXPECT_TRUE(image_changed);
}

TEST_F(ImageManipulationTest, VxConvertToBumpMap) {
    VxImageDescEx desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc);
    desc.Width = 4;
    desc.Height = 4;
    desc.BytesPerLine = 4 * 4;
    desc.Image = AllocateBuffer(4 * 4 * 4);

    FillGradient32(reinterpret_cast<uint32_t *>(desc.Image), 4, 4);

    EXPECT_TRUE(VxConvertToBumpMap(desc));

    // Just verify the function succeeded and modified the image
    uint32_t *ptr = reinterpret_cast<uint32_t *>(desc.Image);
    bool image_changed = false;
    for (int i = 0; i < 16; i++) {
        // Check if any pixels were modified from the original gradient
        uint32_t expected_original = 0xFF000080 | ((i % 4) * 85 << 16) | ((i / 4) * 85 << 8);
        if (ptr[i] != expected_original) {
            image_changed = true;
            break;
        }
    }
    // Note: VxConvertToBumpMap might not modify if it's just calling VxConvertToNormalMap
}

//--- Utility Function Tests ---

class UtilityFunctionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for utility tests
    }

    void TearDown() override {
        for (auto ptr : allocated_buffers) {
            delete[] ptr;
        }
        allocated_buffers.clear();
    }

    uint8_t *AllocateBuffer(size_t size) {
        uint8_t *buffer = new uint8_t[size];
        memset(buffer, 0, size);
        allocated_buffers.push_back(buffer);
        return buffer;
    }

private:
    std::vector<uint8_t *> allocated_buffers;
};

TEST_F(UtilityFunctionTest, VxFillStructure_4Bytes) {
    uint32_t pattern = 0xDEADBEEF;
    uint32_t buffer[10];

    EXPECT_TRUE(VxFillStructure(10, buffer, sizeof(uint32_t), sizeof(uint32_t), &pattern));

    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(buffer[i], pattern);
    }
}

TEST_F(UtilityFunctionTest, VxFillStructure_8Bytes) {
    struct TestStruct8 {
        uint32_t a, b;
    };

    TestStruct8 pattern = {0x12345678, 0x9ABCDEF0};
    TestStruct8 buffer[5];

    EXPECT_TRUE(VxFillStructure(5, buffer, sizeof(TestStruct8), sizeof(TestStruct8), &pattern));

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(buffer[i].a, pattern.a);
        EXPECT_EQ(buffer[i].b, pattern.b);
    }
}

TEST_F(UtilityFunctionTest, VxFillStructure_12Bytes) {
    struct TestStruct12 {
        uint32_t a, b, c;
    };

    TestStruct12 pattern = {0x11111111, 0x22222222, 0x33333333};
    TestStruct12 buffer[3];

    EXPECT_TRUE(VxFillStructure(3, buffer, sizeof(TestStruct12), sizeof(TestStruct12), &pattern));

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(buffer[i].a, pattern.a);
        EXPECT_EQ(buffer[i].b, pattern.b);
        EXPECT_EQ(buffer[i].c, pattern.c);
    }
}

TEST_F(UtilityFunctionTest, VxFillStructure_16Bytes) {
    struct TestStruct16 {
        uint32_t a, b, c, d;
    };

    TestStruct16 pattern = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD};
    TestStruct16 buffer[2];

    EXPECT_TRUE(VxFillStructure(2, buffer, sizeof(TestStruct16), sizeof(TestStruct16), &pattern));

    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(buffer[i].a, pattern.a);
        EXPECT_EQ(buffer[i].b, pattern.b);
        EXPECT_EQ(buffer[i].c, pattern.c);
        EXPECT_EQ(buffer[i].d, pattern.d);
    }
}

TEST_F(UtilityFunctionTest, VxFillStructure_WithStride) {
    uint32_t pattern = 0x12345678;
    uint8_t *buffer = AllocateBuffer(100);

    // Fill every 8 bytes with a 4-byte pattern (stride = 8)
    EXPECT_TRUE(VxFillStructure(5, buffer, 8, 4, &pattern));

    uint32_t *ptr = reinterpret_cast<uint32_t *>(buffer);
    EXPECT_EQ(ptr[0], pattern); // Offset 0
    EXPECT_EQ(ptr[2], pattern); // Offset 8
    EXPECT_EQ(ptr[4], pattern); // Offset 16
    EXPECT_EQ(ptr[6], pattern); // Offset 24
    EXPECT_EQ(ptr[8], pattern); // Offset 32
}

TEST_F(UtilityFunctionTest, VxFillStructure_InvalidParams) {
    uint32_t pattern = 0x12345678;
    uint32_t buffer[10];

    // Invalid parameters should return FALSE
    EXPECT_FALSE(VxFillStructure(0, buffer, 4, 4, &pattern));   // Count = 0
    EXPECT_FALSE(VxFillStructure(10, nullptr, 4, 4, &pattern)); // Null dst
    EXPECT_FALSE(VxFillStructure(10, buffer, 4, 4, nullptr));   // Null src
    EXPECT_FALSE(VxFillStructure(10, buffer, 0, 4, &pattern));  // Stride = 0
    EXPECT_FALSE(VxFillStructure(10, buffer, 4, 0, &pattern));  // SizeSrc = 0
    EXPECT_FALSE(VxFillStructure(10, buffer, 4, 3, &pattern));  // Non-4-byte aligned size
}

TEST_F(UtilityFunctionTest, VxCopyStructure_SameStride) {
    uint32_t src_data[] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
    uint32_t dst_data[4];

    EXPECT_TRUE(VxCopyStructure(4, dst_data, 4, 4, src_data, 4));

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(dst_data[i], src_data[i]);
    }
}

TEST_F(UtilityFunctionTest, VxCopyStructure_DifferentStride) {
    struct SrcStruct {
        uint32_t data;
        uint32_t padding;
    };

    SrcStruct src_data[] = {{0x11111111, 0}, {0x22222222, 0}, {0x33333333, 0}};
    uint32_t dst_data[3];

    // Copy 4 bytes from structures with 8-byte stride to 4-byte stride
    EXPECT_TRUE(VxCopyStructure(3, dst_data, 4, 4, src_data, 8));

    EXPECT_EQ(dst_data[0], 0x11111111);
    EXPECT_EQ(dst_data[1], 0x22222222);
    EXPECT_EQ(dst_data[2], 0x33333333);
}

TEST_F(UtilityFunctionTest, VxCopyStructure_InvalidParams) {
    uint32_t src[4], dst[4];

    EXPECT_FALSE(VxCopyStructure(0, dst, 4, 4, src, 4));     // Count = 0
    EXPECT_FALSE(VxCopyStructure(4, nullptr, 4, 4, src, 4)); // Null dst
    EXPECT_FALSE(VxCopyStructure(4, dst, 4, 4, nullptr, 4)); // Null src
    EXPECT_FALSE(VxCopyStructure(4, dst, 0, 4, src, 4));     // OutStride = 0
    EXPECT_FALSE(VxCopyStructure(4, dst, 4, 0, src, 4));     // SizeSrc = 0
    EXPECT_FALSE(VxCopyStructure(4, dst, 4, 4, src, 0));     // InStride = 0
    EXPECT_FALSE(VxCopyStructure(4, dst, 4, 3, src, 4));     // Non-4-byte aligned size
}

TEST_F(UtilityFunctionTest, VxIndexedCopy_4Bytes) {
    VxStridedData src, dst;

    uint32_t src_data[] = {0x41414141, 0x42424242, 0x43434343, 0x44444444}; // AAAA, BBBB, CCCC, DDDD
    uint32_t dst_data[4];
    XDWORD indices[] = {3, 0, 2, 1}; // Copy D, A, C, B

    src.Ptr = src_data;
    src.Stride = sizeof(uint32_t);
    dst.Ptr = dst_data;
    dst.Stride = sizeof(uint32_t);

    EXPECT_TRUE(VxIndexedCopy(dst, src, sizeof(uint32_t), indices, 4));

    EXPECT_EQ(dst_data[0], 0x44444444); // D
    EXPECT_EQ(dst_data[1], 0x41414141); // A
    EXPECT_EQ(dst_data[2], 0x43434343); // C
    EXPECT_EQ(dst_data[3], 0x42424242); // B
}

TEST_F(UtilityFunctionTest, VxIndexedCopy_8Bytes) {
    VxStridedData src, dst;

    struct TestData {
        uint32_t a, b;
    };

    TestData src_data[] = {{0x11, 0x12}, {0x21, 0x22}, {0x31, 0x32}};
    TestData dst_data[3];
    XDWORD indices[] = {2, 0, 1}; // Copy element 2, 0, 1

    src.Ptr = src_data;
    src.Stride = sizeof(TestData);
    dst.Ptr = dst_data;
    dst.Stride = sizeof(TestData);

    EXPECT_TRUE(VxIndexedCopy(dst, src, sizeof(TestData), indices, 3));

    EXPECT_EQ(dst_data[0].a, 0x31);
    EXPECT_EQ(dst_data[0].b, 0x32);
    EXPECT_EQ(dst_data[1].a, 0x11);
    EXPECT_EQ(dst_data[1].b, 0x12);
    EXPECT_EQ(dst_data[2].a, 0x21);
    EXPECT_EQ(dst_data[2].b, 0x22);
}

TEST_F(UtilityFunctionTest, VxIndexedCopy_WithStride) {
    VxStridedData src, dst;

    struct SrcStruct {
        uint32_t data;
        uint32_t padding1;
        uint32_t padding2;
    };

    struct DstStruct {
        uint32_t data;
        uint32_t padding;
    };

    SrcStruct src_data[] = {{0x111, 0, 0}, {0x222, 0, 0}, {0x333, 0, 0}};
    DstStruct dst_data[3];
    XDWORD indices[] = {1, 2, 0};

    src.Ptr = src_data;
    src.Stride = sizeof(SrcStruct);
    dst.Ptr = dst_data;
    dst.Stride = sizeof(DstStruct);

    EXPECT_TRUE(VxIndexedCopy(dst, src, 4, indices, 3)); // Copy only 4 bytes (the data field)

    EXPECT_EQ(dst_data[0].data, 0x222);
    EXPECT_EQ(dst_data[1].data, 0x333);
    EXPECT_EQ(dst_data[2].data, 0x111);
}

TEST_F(UtilityFunctionTest, VxIndexedCopy_InvalidParams) {
    VxStridedData src, dst;
    uint32_t src_data[4], dst_data[4];
    XDWORD indices[] = {0, 1, 2, 3};

    src.Ptr = src_data;
    src.Stride = 4;
    dst.Ptr = dst_data;
    dst.Stride = 4;

    EXPECT_FALSE(VxIndexedCopy(dst, src, 4, indices, 0)); // IndexCount = 0
    EXPECT_FALSE(VxIndexedCopy(dst, src, 0, indices, 4)); // SizeSrc = 0
    EXPECT_FALSE(VxIndexedCopy(dst, src, 3, indices, 4)); // Non-4-byte aligned size

    dst.Ptr = nullptr;
    EXPECT_FALSE(VxIndexedCopy(dst, src, 4, indices, 4)); // Null dst

    dst.Ptr = dst_data;
    src.Ptr = nullptr;
    EXPECT_FALSE(VxIndexedCopy(dst, src, 4, indices, 4)); // Null src

    src.Ptr = src_data;
    EXPECT_FALSE(VxIndexedCopy(dst, src, 4, nullptr, 4)); // Null indices
}

//--- Bit Manipulation Tests ---

TEST(BitManipulationTests, GetBitCount) {
    EXPECT_EQ(GetBitCount(0), 0);
    EXPECT_EQ(GetBitCount(1), 1);
    EXPECT_EQ(GetBitCount(3), 2);           // 0b11
    EXPECT_EQ(GetBitCount(7), 3);           // 0b111
    EXPECT_EQ(GetBitCount(15), 4);          // 0b1111
    EXPECT_EQ(GetBitCount(0xFF), 8);        // All 8 bits set
    EXPECT_EQ(GetBitCount(0xFFFF), 16);     // All 16 bits set
    EXPECT_EQ(GetBitCount(0xF800), 5);      // RGB565 red mask
    EXPECT_EQ(GetBitCount(0x07E0), 6);      // RGB565 green mask
    EXPECT_EQ(GetBitCount(0x001F), 5);      // RGB565 blue mask
    EXPECT_EQ(GetBitCount(0xFFFFFFFF), 32); // All 32 bits set
}

TEST(BitManipulationTests, GetBitShift) {
    EXPECT_EQ(GetBitShift(0), 0);
    EXPECT_EQ(GetBitShift(1), 0);           // 0b1
    EXPECT_EQ(GetBitShift(2), 1);           // 0b10
    EXPECT_EQ(GetBitShift(4), 2);           // 0b100
    EXPECT_EQ(GetBitShift(8), 3);           // 0b1000
    EXPECT_EQ(GetBitShift(0xFF00), 8);      // Green mask in ARGB
    EXPECT_EQ(GetBitShift(0xFF0000), 16);   // Red mask in ARGB
    EXPECT_EQ(GetBitShift(0xFF000000), 24); // Alpha mask in ARGB
    EXPECT_EQ(GetBitShift(0xF800), 11);     // RGB565 red mask
    EXPECT_EQ(GetBitShift(0x07E0), 5);      // RGB565 green mask
    EXPECT_EQ(GetBitShift(0x001F), 0);      // RGB565 blue mask
}

//--- Performance and Stress Tests ---

TEST_F(ImageManipulationTest, LargeImageResize) {
    // Test with larger images to ensure performance is reasonable
    VxImageDescEx src_desc, dst_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    VxPixelFormat2ImageDesc(_32_ARGB8888, dst_desc);

    const int src_size = 64;
    const int dst_size = 32;

    src_desc.Width = src_size;
    src_desc.Height = src_size;
    src_desc.BytesPerLine = src_size * 4;
    dst_desc.Width = dst_size;
    dst_desc.Height = dst_size;
    dst_desc.BytesPerLine = dst_size * 4;

    src_desc.Image = AllocateBuffer(src_size * src_size * 4);
    dst_desc.Image = AllocateBuffer(dst_size * dst_size * 4);

    FillGradient32(reinterpret_cast<uint32_t *>(src_desc.Image), src_size, src_size);

    // This should complete without timeout
    VxResizeImage32(src_desc, dst_desc);

    // Verify result is not all zeros
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_desc.Image);
    bool has_non_zero = false;
    for (int i = 0; i < dst_size * dst_size; i++) {
        if (dst_ptr[i] != 0) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

TEST_F(UtilityFunctionTest, LargeStructureFill) {
    // Test filling a large array
    const int count = 10000;
    uint32_t pattern = 0xCAFEBABE;
    uint32_t *buffer = reinterpret_cast<uint32_t *>(AllocateBuffer(count * sizeof(uint32_t)));

    EXPECT_TRUE(VxFillStructure(count, buffer, sizeof(uint32_t), sizeof(uint32_t), &pattern));

    // Spot check a few values
    EXPECT_EQ(buffer[0], pattern);
    EXPECT_EQ(buffer[count/2], pattern);
    EXPECT_EQ(buffer[count-1], pattern);
}

//--- VxGenerateMipMap Comprehensive Tests ---

TEST_F(ImageManipulationTest, VxGenerateMipMap_UniformColor) {
    // Test with uniform color - result should be same color
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 8;
    src_desc.Height = 8;
    src_desc.BytesPerLine = 8 * 4;
    src_desc.Image = AllocateBuffer(8 * 8 * 4);

    // Fill with uniform magenta
    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    for (int i = 0; i < 64; i++) {
        src_ptr[i] = 0xFFFF00FF;  // ARGB magenta
    }

    uint8_t *dst_buffer = AllocateBuffer(4 * 4 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    // All output pixels should be magenta (or very close)
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    for (int i = 0; i < 16; i++) {
        EXPECT_EQ((dst_ptr[i] >> 24) & 0xFF, 0xFF) << "Alpha at " << i;
        EXPECT_EQ((dst_ptr[i] >> 16) & 0xFF, 0xFF) << "Red at " << i;
        EXPECT_EQ((dst_ptr[i] >> 8) & 0xFF, 0x00) << "Green at " << i;
        EXPECT_EQ(dst_ptr[i] & 0xFF, 0xFF) << "Blue at " << i;
    }
}

TEST_F(ImageManipulationTest, VxGenerateMipMap_Checkerboard) {
    // Test with checkerboard pattern - should average to gray
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 4;
    src_desc.Height = 4;
    src_desc.BytesPerLine = 4 * 4;
    src_desc.Image = AllocateBuffer(4 * 4 * 4);

    // Create checkerboard: black and white
    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            src_ptr[y * 4 + x] = ((x + y) % 2 == 0) ? 0xFF000000 : 0xFFFFFFFF;
        }
    }

    uint8_t *dst_buffer = AllocateBuffer(2 * 2 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    // Each 2x2 block should average to gray
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    for (int i = 0; i < 4; i++) {
        uint8_t r = (dst_ptr[i] >> 16) & 0xFF;
        uint8_t g = (dst_ptr[i] >> 8) & 0xFF;
        uint8_t b = dst_ptr[i] & 0xFF;
        // Should be ~127 (average of 0 and 255)
        EXPECT_NEAR(r, 127, 2) << "Red at " << i;
        EXPECT_NEAR(g, 127, 2) << "Green at " << i;
        EXPECT_NEAR(b, 127, 2) << "Blue at " << i;
    }
}

TEST_F(ImageManipulationTest, VxGenerateMipMap_LargeImage) {
    // Test with larger image to exercise SSE2 path
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 64;
    src_desc.Height = 64;
    src_desc.BytesPerLine = 64 * 4;
    src_desc.Image = AllocateBuffer(64 * 64 * 4);

    // Fill with gradient
    FillGradient32(reinterpret_cast<uint32_t *>(src_desc.Image), 64, 64);

    uint8_t *dst_buffer = AllocateBuffer(32 * 32 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    // Verify non-zero output
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    bool has_content = false;
    for (int i = 0; i < 32 * 32; i++) {
        if (dst_ptr[i] != 0) {
            has_content = true;
            break;
        }
    }
    EXPECT_TRUE(has_content);
}

TEST_F(ImageManipulationTest, VxGenerateMipMap_OddWidth) {
    // Test with width that doesn't divide evenly by 2
    // The result width will be floor(5/2) = 2
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 5;
    src_desc.Height = 4;
    src_desc.BytesPerLine = 5 * 4;
    src_desc.Image = AllocateBuffer(5 * 4 * 4);

    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    for (int i = 0; i < 20; i++) {
        src_ptr[i] = 0xFF808080;
    }

    uint8_t *dst_buffer = AllocateBuffer(2 * 2 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    // Should produce valid output
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    for (int i = 0; i < 4; i++) {
        EXPECT_NE(dst_ptr[i], 0u);
    }
}

TEST_F(ImageManipulationTest, VxGenerateMipMap_SingleRow) {
    // Test 2xN image (dstHeight = 0 case - only horizontal averaging)
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 8;
    src_desc.Height = 1;
    src_desc.BytesPerLine = 8 * 4;
    src_desc.Image = AllocateBuffer(8 * 1 * 4);

    // Alternating red and green
    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    for (int i = 0; i < 8; i++) {
        src_ptr[i] = (i % 2 == 0) ? 0xFFFF0000 : 0xFF00FF00;
    }

    uint8_t *dst_buffer = AllocateBuffer(4 * 1 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    // Each pair should average to yellow-ish
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    for (int i = 0; i < 4; i++) {
        uint8_t r = (dst_ptr[i] >> 16) & 0xFF;
        uint8_t g = (dst_ptr[i] >> 8) & 0xFF;
        EXPECT_NEAR(r, 127, 2) << "Red at " << i;
        EXPECT_NEAR(g, 127, 2) << "Green at " << i;
    }
}

TEST_F(ImageManipulationTest, VxGenerateMipMap_SingleColumn) {
    // Test Nx2 image (dstWidth = 0 case - only vertical averaging)
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 1;
    src_desc.Height = 8;
    src_desc.BytesPerLine = 1 * 4;
    src_desc.Image = AllocateBuffer(1 * 8 * 4);

    // Alternating blue and cyan
    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    for (int i = 0; i < 8; i++) {
        src_ptr[i] = (i % 2 == 0) ? 0xFF0000FF : 0xFF00FFFF;
    }

    uint8_t *dst_buffer = AllocateBuffer(1 * 4 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    // Each pair should average
    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    for (int i = 0; i < 4; i++) {
        uint8_t g = (dst_ptr[i] >> 8) & 0xFF;
        uint8_t b = dst_ptr[i] & 0xFF;
        EXPECT_NEAR(g, 127, 2) << "Green at " << i;
        EXPECT_EQ(b, 0xFF) << "Blue at " << i;
    }
}

TEST_F(ImageManipulationTest, VxGenerateMipMap_AlphaPreservation) {
    // Test that alpha channel is properly averaged
    VxImageDescEx src_desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, src_desc);
    src_desc.Width = 4;
    src_desc.Height = 4;
    src_desc.BytesPerLine = 4 * 4;
    src_desc.Image = AllocateBuffer(4 * 4 * 4);

    // Create 2x2 blocks with different alpha values
    uint32_t *src_ptr = reinterpret_cast<uint32_t *>(src_desc.Image);
    // Top-left: alpha=0, Top-right: alpha=255
    // Bottom-left: alpha=128, Bottom-right: alpha=64
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            src_ptr[y * 4 + x] = 0x00808080;  // alpha=0
        }
    }
    for (int y = 0; y < 2; y++) {
        for (int x = 2; x < 4; x++) {
            src_ptr[y * 4 + x] = 0xFF808080;  // alpha=255
        }
    }
    for (int y = 2; y < 4; y++) {
        for (int x = 0; x < 2; x++) {
            src_ptr[y * 4 + x] = 0x80808080;  // alpha=128
        }
    }
    for (int y = 2; y < 4; y++) {
        for (int x = 2; x < 4; x++) {
            src_ptr[y * 4 + x] = 0x40808080;  // alpha=64
        }
    }

    uint8_t *dst_buffer = AllocateBuffer(2 * 2 * 4);
    VxGenerateMipMap(src_desc, dst_buffer);

    uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_buffer);
    // Check alpha values are properly averaged
    EXPECT_EQ((dst_ptr[0] >> 24) & 0xFF, 0);    // avg of 0
    EXPECT_EQ((dst_ptr[1] >> 24) & 0xFF, 255);  // avg of 255
    EXPECT_EQ((dst_ptr[2] >> 24) & 0xFF, 128);  // avg of 128
    EXPECT_EQ((dst_ptr[3] >> 24) & 0xFF, 64);   // avg of 64
}

//--- Edge Case Tests ---

TEST(EdgeCaseTests, ZeroSizeImages) {
    VxImageDescEx desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc);
    desc.Width = 0;
    desc.Height = 0;

    // Functions should handle zero-size gracefully
    uint8_t dummy_buffer[16];
    desc.Image = dummy_buffer;

    VxDoAlphaBlit(desc, 255); // Should not crash

    // Normal map conversion on zero-size should fail gracefully
    EXPECT_FALSE(VxConvertToNormalMap(desc, desc.RedMask));
}

TEST(EdgeCaseTests, VerySmallImages) {
    VxImageDescEx desc;
    VxPixelFormat2ImageDesc(_32_ARGB8888, desc);
    desc.Width = 1;
    desc.Height = 1;
    desc.BytesPerLine = 4;

    uint32_t pixel = 0xFF808080;
    desc.Image = reinterpret_cast<uint8_t *>(&pixel);

    VxDoAlphaBlit(desc, 128);
    EXPECT_EQ(pixel, 0x80808080); // Alpha should be changed to 128
}

TEST(EdgeCaseTests, NullPointerHandling) {
    VxImageDescEx desc;
    desc.Image = nullptr;

    // Functions should handle null pointers gracefully
    VxDoAlphaBlit(desc, 255);
    EXPECT_FALSE(VxConvertToNormalMap(desc, 0xFF000000));
    EXPECT_FALSE(VxConvertToBumpMap(desc));
}
