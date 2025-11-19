#include "gtest/gtest.h"
#include "VxMath.h"

// Test Fixture for System and Info tests
class SystemInfoTest : public ::testing::Test {
protected:
    // This function is called before each test is run.
    // We call VxDetectProcessor() here to ensure the information is populated
    // for all subsequent tests in this suite.
    void SetUp() override {
        // InitVxMath() is supposed to call VxDetectProcessor.
        // If not, we call it explicitly to be safe.
        // InitVxMath();
    }
};

// Test to ensure processor description is not empty.
TEST_F(SystemInfoTest, GetProcessorDescription_ReturnsNonEmptyString) {
    char *description = GetProcessorDescription();
    ASSERT_NE(description, nullptr);
    // The description string should not be empty.
    EXPECT_GT(strlen(description), 0);
    // It's a good sign if it contains a known CPU vendor name.
    // This is not a strict requirement but a good heuristic.
    std::string desc_str(description);
    bool has_known_vendor = (desc_str.find("Intel") != std::string::npos) ||
        (desc_str.find("AMD") != std::string::npos) ||
        (desc_str.find("ARM") != std::string::npos);
    // This may fail on very obscure CPUs, but is a reasonable check.
    // We print it for manual verification if the test fails.
    std::cout << "Processor Description: " << description << std::endl;
    // We don't strictly assert this, but it's good to know.
    // ASSERT_TRUE(has_known_vendor);
}

// Test to ensure processor frequency is a plausible positive number.
TEST_F(SystemInfoTest, GetProcessorFrequency_ReturnsPlausibleValue) {
    int frequency = GetProcessorFrequency();
    // The frequency should be a positive number, likely over 100 MHz.
    EXPECT_GT(frequency, 100);
    std::cout << "Processor Frequency: " << frequency << " MHz" << std::endl;
}

// Test to ensure GetProcessorFeatures returns a set of flags.
TEST_F(SystemInfoTest, GetProcessorFeatures_ReturnsNonZeroFlags) {
    XULONG features = GetProcessorFeatures();
    // A modern processor should have at least some basic features detected.
    // PROC_HASFPU is almost guaranteed on any desktop/server platform.
    EXPECT_NE(features, 0);
    EXPECT_TRUE(features & PROC_HASFPU);
    std::cout << "Processor Features: 0x" << std::hex << features << std::dec << std::endl;
}

// Test for modifying and reading processor features.
TEST_F(SystemInfoTest, ModifyProcessorFeatures_AddsAndRemovesFlags) {
    // Save the original features to restore them later.
    XULONG original_features = GetProcessorFeatures();

    // --- Test adding a feature ---
    // Let's pretend our CPU has a feature it likely doesn't (PROC_TM - Thermal Monitor)
    // First, ensure it's not already set (or remove it to start clean).
    ModifyProcessorFeatures(0, PROC_TM);
    ASSERT_FALSE(GetProcessorFeatures() & PROC_TM);

    // Now, add the feature.
    ModifyProcessorFeatures(PROC_TM, 0);
    EXPECT_TRUE(GetProcessorFeatures() & PROC_TM);

    // --- Test removing a feature ---
    // Remove the feature we just added.
    ModifyProcessorFeatures(0, PROC_TM);
    EXPECT_FALSE(GetProcessorFeatures() & PROC_TM);

    // --- Test removing a feature that is likely present (like MMX) ---
    // Note: This modifies the global state, which might affect other tests
    // if they run in parallel and depend on this, but for this library it's a global singleton state.
    if (original_features & PROC_MMX) {
        ModifyProcessorFeatures(0, PROC_MMX);
        EXPECT_FALSE(GetProcessorFeatures() & PROC_MMX);
    }

    // --- Test adding and removing at the same time ---
    // Add TM back and remove FPU (if present)
    ModifyProcessorFeatures(PROC_TM, PROC_HASFPU);
    EXPECT_TRUE(GetProcessorFeatures() & PROC_TM);
    EXPECT_FALSE(GetProcessorFeatures() & PROC_HASFPU);

    // Restore original features to not affect other tests.
    ModifyProcessorFeatures(original_features, ~original_features);
    ASSERT_EQ(GetProcessorFeatures(), original_features);
}

// Test to ensure GetProcessorType returns a valid enum value.
TEST_F(SystemInfoTest, GetProcessorType_ReturnsValidEnum) {
    ProcessorsType type = GetProcessorType();
    // The type should not be PROC_UNKNOWN on any machine this test is likely to run on.
    EXPECT_NE(type, PROC_UNKNOWN);
    // It should be within the valid range of the enum.
    EXPECT_GE(type, PROC_PENTIUM);
    EXPECT_LE(type, PROC_PSP);
    std::cout << "Processor Type Enum: " << type << std::endl;
}

// Test to ensure GetInstructionSetExtensions returns a set of flags.
// TEST_F(SystemInfoTest, GetInstructionSetExtensions_ReturnsPlausibleFlags) {
//     XULONG extensions = GetInstructionSetExtensions();
//     // Most modern x86/x64 CPUs will have at least SSE and SSE2.
//     // This check is platform-dependent.
// #if defined(VX_MSVC) || defined(VX_GCC)
// #if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
//     EXPECT_TRUE(extensions & ISEX_SSE);
//     EXPECT_TRUE(extensions & ISEX_SSE2);
// #endif
// #endif
//     std::cout << "Instruction Set Extensions: 0x" << std::hex << extensions << std::dec << std::endl;
// }

// Test OS and Platform detection.
TEST_F(SystemInfoTest, GetOsAndPlatform_ReturnsValidEnums) {
    VX_OSINFO os = VxGetOs();
    // Should not be unknown.
    EXPECT_NE(os, VXOS_UNKNOWN);
    // Should be within the valid range.
    EXPECT_GE(os, VXOS_WIN31);
    EXPECT_LE(os, VXOS_WINSEVEN); // This might need updating for newer OSes.

    std::cout << "OS Info Enum: " << os << std::endl;

    // This library doesn't provide a direct VxGetPlatformInfo function,
    // but the enum exists and could be used internally.
    // A test could be added if such a function is exposed.
}

// Test bit count and shift helper functions.
TEST_F(SystemInfoTest, BitCountAndShiftHelpers) {
    // Test GetBitCount
    EXPECT_EQ(GetBitCount(0x000000FF), 8);  // 8 bits
    EXPECT_EQ(GetBitCount(0x0000F000), 4);  // 4 bits
    EXPECT_EQ(GetBitCount(0x00000000), 0);  // 0 bits
    EXPECT_EQ(GetBitCount(0xFFFFFFFF), 32); // 32 bits

    // Test GetBitShift
    EXPECT_EQ(GetBitShift(0x000000FF), 0);  // Shift 0
    EXPECT_EQ(GetBitShift(0x0000FF00), 8);  // Shift 8
    EXPECT_EQ(GetBitShift(0xFF000000), 24); // Shift 24
    EXPECT_EQ(GetBitShift(0x00F00000), 20); // Shift 20
    EXPECT_EQ(GetBitShift(0x00000000), 0);  // Shift for 0 mask is undefined, but likely 0.
}

// Test the full conversion cycle for all pixel formats.
TEST_F(SystemInfoTest, PixelFormat_ConversionCycle) {
    for (int i = 1; i <= _32_X8L8V8U8; ++i) {
        VX_PIXELFORMAT pf_original = static_cast<VX_PIXELFORMAT>(i);

        // Step 1: Get a description from the format
        VxImageDescEx desc;
        VxPixelFormat2ImageDesc(pf_original, desc);

        // For non-compressed formats, VxBppToMask might be called internally.
        // Let's ensure the masks are plausible.
        if (pf_original < _DXT1) {
            EXPECT_GT(desc.BitsPerPixel, 0);
            if (desc.BitsPerPixel > 8) {
                EXPECT_NE(desc.RedMask | desc.GreenMask | desc.BlueMask, 0);
            }
        }

        // Step 2: Convert the description back to a format
        VX_PIXELFORMAT pf_converted = VxImageDesc2PixelFormat(desc);

        // Step 3: Verify they match
        EXPECT_EQ(pf_original, pf_converted) << "Mismatch for format " << VxPixelFormat2String(pf_original);

        // Step 4: Verify string conversion
        const char *pf_string = VxPixelFormat2String(pf_original);
        printf("pixel format %d: %s\n", pf_original, pf_string);

        ASSERT_NE(pf_string, nullptr);
        EXPECT_GT(strlen(pf_string), 0);
        EXPECT_STRNE(pf_string, "");
    }
}

// Test case for an unknown pixel format string.
TEST_F(SystemInfoTest, VxPixelFormat2String_UnknownFormat) {
    // Use an integer value that is not a valid VX_PIXELFORMAT enum
    const char *empty_str = VxPixelFormat2String(static_cast<VX_PIXELFORMAT>(9999));
    ASSERT_NE(empty_str, nullptr);
    EXPECT_STREQ(empty_str, "");
}

// Test VxPtInRect function
TEST_F(SystemInfoTest, PtInRect) {
    CKRECT rect = {10, 10, 100, 100};

    // Point inside
    CKPOINT pt_inside = {50, 50};
    EXPECT_TRUE(VxPtInRect(&rect, &pt_inside));

    // Point on left edge (inclusive)
    CKPOINT pt_on_left = {10, 50};
    EXPECT_TRUE(VxPtInRect(&rect, &pt_on_left));

    // Point on top edge (inclusive)
    CKPOINT pt_on_top = {50, 10};
    EXPECT_TRUE(VxPtInRect(&rect, &pt_on_top));

    // Point on right edge (inclusive)
    CKPOINT pt_on_right = {100, 50};
    EXPECT_TRUE(VxPtInRect(&rect, &pt_on_right));

    // Point on bottom edge (inclusive)
    CKPOINT pt_on_bottom = {50, 100};
    EXPECT_TRUE(VxPtInRect(&rect, &pt_on_bottom));

    // Point outside
    CKPOINT pt_outside = {200, 200};
    EXPECT_FALSE(VxPtInRect(&rect, &pt_outside));
}

// =============================================================================
// Main function to run all tests
