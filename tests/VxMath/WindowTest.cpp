#include "gtest/gtest.h"
#include "VxMath.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Platform-specific includes for creating a temporary window for tests
#ifdef _WIN32
#include <Windows.h>
#else
// Define WIN_HANDLE as a void pointer on non-Windows platforms for compilation
using WIN_HANDLE = void*;
#endif

// A test fixture to manage temporary resources like files and windows.
class VxWindowFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for filesystem tests
        m_tempDir = std::filesystem::temp_directory_path() / "VxMath_Test_Temp";
        std::filesystem::create_directories(m_tempDir);

        // Create a temporary native window to get a valid handle for windowing function tests
#ifdef _WIN32
        WNDCLASS wc = {};
        wc.lpfnWndProc = DefWindowProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = "VxTestWindow";
        RegisterClass(&wc);

        m_testWindow = CreateWindowEx(
            0, "VxTestWindow", "VxMath Test Window", WS_OVERLAPPED,
            CW_USEDEFAULT, CW_USEDEFAULT, 200, 200,
            nullptr, nullptr, GetModuleHandle(nullptr), nullptr
        );
#endif
    }

    void TearDown() override {
        // Clean up the temporary directory
        std::filesystem::remove_all(m_tempDir);

        // Clean up the temporary window
#ifdef _WIN32
        if (m_testWindow) {
            DestroyWindow(static_cast<HWND>(m_testWindow));
        }
        UnregisterClass("VxTestWindow", GetModuleHandle(nullptr));
#endif
    }

    // Helper to create a file with specific content in our temp directory
    std::string CreateTempFile(const std::string &filename, const std::string &content) {
        std::filesystem::path filePath = m_tempDir / filename;
        std::ofstream file(filePath);
        file << content;
        file.close();
        return filePath.string();
    }

    std::filesystem::path m_tempDir;
    WIN_HANDLE m_testWindow = nullptr;
};

// --- Keyboard and Input Tests ---

TEST_F(VxWindowFunctionsTest, ScanCodeToAscii) {
    // Test without shift key
    unsigned char keyState[256] = {0};
    // Scan code for 'A' key (VK_A) on a US keyboard is often 0x1E
    XDWORD scanCodeA = 0x1E;
    EXPECT_EQ(VxScanCodeToAscii(scanCodeA, keyState), 'a');

    // Test with shift key pressed
    keyState[VK_SHIFT] = 0x80; // Set high bit to indicate key is down
    EXPECT_EQ(VxScanCodeToAscii(scanCodeA, keyState), 'a');

    // Test a non-printable key
    keyState[VK_SHIFT] = 0;
    XDWORD scanCodeF1 = 0x3B; // Scan code for F1
    EXPECT_EQ(VxScanCodeToAscii(scanCodeF1, keyState), 0);
}

TEST_F(VxWindowFunctionsTest, ScanCodeToName) {
    char keyName[256];
    // This is highly layout-dependent, but we can test special keys
    XDWORD scanCodeLeft = 0xCB;
    EXPECT_GT(VxScanCodeToName(scanCodeLeft, keyName), 0);
    // The exact name can vary, but it should be something like "Left" or "Left Arrow"
    std::string name(keyName);
    EXPECT_NE(name.find("Left"), std::string::npos);
}

// --- Cursor Management Tests ---

TEST_F(VxWindowFunctionsTest, CursorFunctions) {
    // These functions' effects are visual and hard to automate a check for.
    // The test ensures they can be called without crashing.
    EXPECT_NO_THROW(VxShowCursor(FALSE));
    EXPECT_NO_THROW(VxShowCursor(TRUE));

    EXPECT_TRUE(VxSetCursor(VXCURSOR_NORMALSELECT));
    EXPECT_TRUE(VxSetCursor(VXCURSOR_BUSY));
    EXPECT_TRUE(VxSetCursor(VXCURSOR_MOVE));
    EXPECT_TRUE(VxSetCursor(VXCURSOR_LINKSELECT));
}

// --- FPU Control Word Tests ---
TEST_F(VxWindowFunctionsTest, FPUControlWord) {
    // Get the initial FPU control word
    XWORD initialFpu = VxGetFPUControlWord();

    // Set it to a known state (e.g., the base state) and then back
    EXPECT_NO_THROW(VxSetBaseFPUControlWord());
    XWORD baseFpu = VxGetFPUControlWord();
    EXPECT_NE(baseFpu, 0); // Should not be zero

    // Restore the initial state
    EXPECT_NO_THROW(VxSetFPUControlWord(initialFpu));
    EXPECT_EQ(VxGetFPUControlWord(), initialFpu);
}

// --- Filesystem and Path Tests ---

TEST_F(VxWindowFunctionsTest, FilesystemOperations) {
    // Test GetTempPath
    XString tempPath = VxGetTempPath();
    EXPECT_FALSE(tempPath.IsEmpty());
    EXPECT_TRUE(std::filesystem::exists(tempPath.CStr()));

    // Test MakeDirectory
    std::filesystem::path newDir = m_tempDir / "NewDirectory";
    EXPECT_TRUE(VxMakeDirectory((char*)newDir.string().c_str()));
    EXPECT_TRUE(std::filesystem::exists(newDir));

    // Test RemoveDirectory on an empty directory
    EXPECT_TRUE(VxRemoveDirectory((char*)newDir.string().c_str()));
    EXPECT_FALSE(std::filesystem::exists(newDir));

    // Test DeleteDirectory on a non-empty directory
    std::filesystem::path parentDir = m_tempDir / "Parent";
    std::filesystem::path childFile = parentDir / "child.txt";
    VxMakeDirectory((char *) parentDir.string().c_str());
    CreateTempFile("Parent/child.txt", "content");
    EXPECT_TRUE(std::filesystem::exists(childFile));
    EXPECT_TRUE(VxDeleteDirectory((char*)parentDir.string().c_str()));
    EXPECT_FALSE(std::filesystem::exists(parentDir));

    // Test CreateFileTree
    std::filesystem::path treePath = m_tempDir / "a" / "b" / "c.txt";
    EXPECT_TRUE(VxCreateFileTree((char*)treePath.string().c_str()));
    EXPECT_TRUE(std::filesystem::exists(treePath.parent_path()));
}

TEST_F(VxWindowFunctionsTest, CurrentDirectory) {
    char originalDir[MAX_PATH];
    ASSERT_TRUE(VxGetCurrentDirectory(originalDir));

    // Set current directory to our temp path
    ASSERT_TRUE(VxSetCurrentDirectory((char*)m_tempDir.string().c_str()));

    // Verify it was set
    char newDir[MAX_PATH];
    ASSERT_TRUE(VxGetCurrentDirectory(newDir));
    EXPECT_EQ(std::filesystem::path(newDir), m_tempDir);

    // Restore original directory
    ASSERT_TRUE(VxSetCurrentDirectory(originalDir));
}

TEST_F(VxWindowFunctionsTest, MakePath) {
    char fullPath[MAX_PATH];
#ifdef _WIN32
    const char *path = "C:\\Temp";
#else
        const char* path = "/tmp";
#endif
    const char *file = "test.txt";

    EXPECT_TRUE(VxMakePath(fullPath, (char*)path, (char*)file));
    std::filesystem::path expectedPath = std::filesystem::path(path) / file;
    EXPECT_EQ(std::filesystem::path(fullPath), expectedPath);
}

TEST_F(VxWindowFunctionsTest, TestDiskSpace) {
    // Check for a small amount of space (should always be available)
    EXPECT_TRUE(VxTestDiskSpace(m_tempDir.string().c_str(), 1024)); // 1 KB

    // Check for an impossibly large amount of space (should fail)
    // Using ULONG_MAX which is guaranteed to be larger than available disk space.
    // EXPECT_FALSE(VxTestDiskSpace(m_tempDir.string().c_str(), ULONG_MAX));
}

// --- Environment and Module Tests ---

TEST_F(VxWindowFunctionsTest, EnvironmentVariable) {
    const char *varName = "VX_MATH_TEST_VAR";
    const char *varValue = "HelloWorld123";
    XString readValue;

    // Test setting a variable
    ASSERT_TRUE(VxSetEnvironmentVariable((char*)varName, (char*)varValue));

    // Test getting the variable
    ASSERT_TRUE(VxGetEnvironmentVariable((char*)varName, readValue));
    EXPECT_STREQ(readValue.CStr(), varValue);

    // Clean up by unsetting the variable
    ASSERT_TRUE(VxSetEnvironmentVariable((char*)varName, nullptr));
    EXPECT_FALSE(VxGetEnvironmentVariable((char*)varName, readValue));
}

TEST_F(VxWindowFunctionsTest, UrlEscaping) {
    XString escapedUrl;
    const char *originalUrl = "http://example.com/a path?q=a&b=c";
    const char *expectedEscaped = "http%3A%2F%2Fexample.com%2Fa%20path%3Fq%3Da%26b%3Dc";

    VxEscapeURL((char *) originalUrl, escapedUrl);
    EXPECT_STREQ(escapedUrl.CStr(), expectedEscaped);

    VxUnEscapeUrl(escapedUrl);
    EXPECT_STREQ(escapedUrl.CStr(), originalUrl);
}

TEST_F(VxWindowFunctionsTest, ModuleFunctions) {
    // Get handle to the current executable
    INSTANCE_HANDLE hMod = VxGetModuleHandle(nullptr);
    ASSERT_NE(hMod, nullptr);

    // Get the file name of the current executable
    char modulePath[MAX_PATH];
    XDWORD pathLen = VxGetModuleFileName(hMod, modulePath, MAX_PATH);
    ASSERT_GT(pathLen, 0);
    EXPECT_TRUE(std::filesystem::exists(modulePath));
}

// --- Windowing Function Tests (Windows-specific) ---

#ifdef _WIN32
TEST_F(VxWindowFunctionsTest, WindowRectFunctions) {
    ASSERT_NE(m_testWindow, nullptr);

    CKRECT rect;
    ASSERT_TRUE(VxGetClientRect(m_testWindow, &rect));
    // Client rect top-left should be (0,0) and size should be positive
    EXPECT_EQ(rect.left, 0);
    EXPECT_EQ(rect.top, 0);
    EXPECT_GT(rect.right, rect.left);
    EXPECT_GT(rect.bottom, rect.top);

    ASSERT_TRUE(VxGetWindowRect(m_testWindow, &rect));
    // Window rect should have positive size, but not necessarily at (0,0)
    EXPECT_GT(rect.right, rect.left);
    EXPECT_GT(rect.bottom, rect.top);
}

TEST_F(VxWindowFunctionsTest, WindowCoordinateConversion) {
    ASSERT_NE(m_testWindow, nullptr);

    CKPOINT pt = {10, 15};
    CKPOINT originalPt = pt;
    CKPOINT screenPt;

    // Test ClientToScreen
    ASSERT_TRUE(VxClientToScreen(m_testWindow, &pt));
    screenPt = pt; // Save screen coordinates
    // Screen coordinates should not be the same as original client coordinates
    EXPECT_TRUE(screenPt.x != originalPt.x || screenPt.y != originalPt.y);

    // Test ScreenToClient (round-trip)
    ASSERT_TRUE(VxScreenToClient(m_testWindow, &pt));
    EXPECT_EQ(pt.x, originalPt.x);
    EXPECT_EQ(pt.y, originalPt.y);
}

TEST_F(VxWindowFunctionsTest, WindowHierarchyAndMovement) {
    ASSERT_NE(m_testWindow, nullptr);

    // Test GetParent (should be NULL for a top-level window)
    EXPECT_EQ(VxGetParent(m_testWindow), nullptr);

    // Test MoveWindow
    int newX = 100, newY = 150, newW = 300, newH = 250;
    ASSERT_TRUE(VxMoveWindow(m_testWindow, newX, newY, newW, newH, TRUE));

    CKRECT rect;
    ASSERT_TRUE(VxGetWindowRect(m_testWindow, &rect));
    // Allow for small discrepancies due to window borders/title bar
    EXPECT_NEAR(rect.left, newX, 10);
    EXPECT_NEAR(rect.top, newY, 40);
    EXPECT_NEAR(rect.right - rect.left, newW, 20);
    EXPECT_NEAR(rect.bottom - rect.top, newH, 50);
}
#endif

// --- GDI and Font Tests (Windows-specific) ---

#ifdef _WIN32
TEST_F(VxWindowFunctionsTest, FontLifecycleAndInfo) {
    // Create a font
    FONT_HANDLE hFont = VxCreateFont((char *) "Arial", 16, 400, FALSE, FALSE);
    ASSERT_NE(hFont, nullptr);

    // Get font info and verify
    VXFONTINFO fontInfo;
    ASSERT_TRUE(VxGetFontInfo(hFont, fontInfo));
    EXPECT_STREQ(fontInfo.FaceName.CStr(), "Arial");
    EXPECT_EQ(fontInfo.Height, 21);
    EXPECT_EQ(fontInfo.Weight, 400);
    EXPECT_FALSE(fontInfo.Italic);
    EXPECT_FALSE(fontInfo.Underline);

    // Delete the font
    EXPECT_NO_THROW(VxDeleteFont(hFont));
}

TEST_F(VxWindowFunctionsTest, BitmapOperations) {
    // 1. Create a bitmap description for a 2x2 32-bit image
    VxImageDescEx desc;
    desc.Width = 2;
    desc.Height = 2;
    desc.BitsPerPixel = 32;
    VxBppToMask(desc); // Sets standard ARGB masks

    // 2. Create the GDI bitmap object
    BITMAP_HANDLE hBitmap = VxCreateBitmap(desc);
    ASSERT_NE(hBitmap, nullptr);

    // 3. Create source pixel data to copy into the bitmap
    std::vector<XDWORD> pixels = {0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF};
    desc.Image = (XBYTE *) pixels.data();
    desc.BytesPerLine = 2 * 4;

    // 4. Copy the data into the GDI bitmap
    ASSERT_TRUE(VxCopyBitmap(hBitmap, desc));

    // 5. Convert the GDI bitmap back to a raw buffer
    VxImageDescEx outDesc;
    XBYTE *outPixels = VxConvertBitmap(hBitmap, outDesc);
    ASSERT_NE(outPixels, nullptr);

    // 6. Verify the round-trip data
    EXPECT_EQ(outDesc.Width, desc.Width);
    EXPECT_EQ(outDesc.Height, desc.Height);
    EXPECT_EQ(outDesc.BitsPerPixel, desc.BitsPerPixel);
    EXPECT_EQ(memcmp(outPixels, pixels.data(), pixels.size() * sizeof(XDWORD)), 0);

    // Clean up
    delete[] outPixels;
    VxDeleteBitmap(hBitmap);
}
#endif

// --- Tests for functions that are not easily automated ---

// A message box test will block execution, so it's disabled by default.
// To run it, rename the test to remove the "DISABLED_" prefix.
TEST_F(VxWindowFunctionsTest, DISABLED_MessageBox) {
#ifdef _WIN32
    // This test just ensures the function can be called.
    // It will pop up a message box during the test run.
    VxMessageBox(m_testWindow, (char *) "Test Message", (char *) "VxMath Test", 0);
#endif
}
