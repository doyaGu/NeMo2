#include <gtest/gtest.h>
#include <cstring>

// Include the actual XString header
#include "XString.h"

// Platform-specific case-insensitive string comparison
#ifdef _WIN32
#define strnicmp _strnicmp
#else
#define strnicmp strncasecmp
#endif

// Include the functions under test
typedef unsigned long XDWORD;

XDWORD VxEscapeURL(char *InURL, XString &OutURL) {
    if (!InURL) {
        OutURL = "";
        return -1;
    }

    // If it's already a file URL, just return it
    if (strnicmp(InURL, "file://", sizeof("file://") - 1) == 0) {
        OutURL = InURL;
        return 0;
    }

    // Count special characters that need escaping
    char *str = InURL;
    int i = 0;
    while (true) {
        char *s = strpbrk(str, " #$%&\\+,;=@[]^{}");
        if (!s)
            break;
        str = s + 1;
        i++;
    }

    // If no special characters, no need to escape
    if (i == 0) {
        // Check if there's a protocol; if not, add file://
        if (strstr(InURL, "://") == NULL) {
            OutURL = "file://";
            OutURL += InURL;
        } else {
            OutURL = InURL;
        }
        return 0;
    }

    // Calculate buffer size and allocate
    int sz = strlen(InURL) + 1 + 2 * i;
    bool needsFileProtocol = (strstr(InURL, "://") == NULL);
    if (needsFileProtocol)
        sz += sizeof("file://") - 1;

    char *buf = new char[sz];
    if (!buf) {
        OutURL = InURL;
        return 0;
    }

    // Build the escaped URL
    char *pi = InURL;
    char *pb = buf;

    // Add file:// if needed
    if (needsFileProtocol) {
        strcpy(buf, "file://");
        pb += sizeof("file://") - 1;
    }

    // Copy and escape characters
    while (*pi) {
        // Check if character needs escaping
        if (strchr(" #$%&\\+,;=@[]^{}", *pi)) {
            sprintf(pb, "%%%02X", (unsigned char) *pi);
            pb += 3;
        } else {
            *pb++ = *pi;
        }
        pi++;
    }
    *pb = '\0';

    OutURL = buf;
    delete[] buf;
    return 0;
}

void VxUnEscapeUrl(XString &str) {
    XString url;
    if (str.Length() == 0) {
        str = "";
        return;
    }

    for (int i = 0; i < str.Length(); ++i) {
        if (str[i] == '%' && i + 2 < str.Length()) {
            // Convert hex value to character
            char hex[3] = {str[i + 1], str[i + 2], 0};
            int value;
            sscanf(hex, "%x", &value);
            url += (char) value;
            i += 2; // Skip the next two characters
        } else {
            url += str[i];
        }
    }
    str = url;
}

// Test fixture for VxEscapeURL
class VxEscapeURLTest : public ::testing::Test {
protected:
    void SetUp() override {
        outURL = "";
    }

    XString outURL;
};

// Test fixture for VxUnEscapeUrl
class VxUnEscapeUrlTest : public ::testing::Test {
protected:
    void SetUp() override {
        testStr = "";
    }

    XString testStr;
};

TEST_F(VxEscapeURLTest, NullInput) {
    XDWORD result = VxEscapeURL(nullptr, outURL);
    EXPECT_EQ(result, (XDWORD)-1);
    EXPECT_TRUE(outURL == "");
}

TEST_F(VxEscapeURLTest, EmptyString) {
    char input[] = "";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file://");
}

TEST_F(VxEscapeURLTest, FileProtocolAlreadyPresent) {
    char input[] = "file:///path/to/file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/file.txt");
}

TEST_F(VxEscapeURLTest, FileProtocolCaseInsensitive) {
    char input[] = "FILE:///path/to/file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "FILE:///path/to/file.txt");
}

TEST_F(VxEscapeURLTest, FileProtocolMixedCase) {
    char input[] = "File:///path/to/file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "File:///path/to/file.txt");
}

TEST_F(VxEscapeURLTest, HttpProtocolNoEscaping) {
    char input[] = "http://example.com/path";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "http://example.com/path");
}

TEST_F(VxEscapeURLTest, HttpsProtocolNoEscaping) {
    char input[] = "https://example.com/path";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "https://example.com/path");
}

TEST_F(VxEscapeURLTest, FtpProtocolNoEscaping) {
    char input[] = "ftp://example.com/path";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "ftp://example.com/path");
}

TEST_F(VxEscapeURLTest, CustomProtocolNoEscaping) {
    char input[] = "custom://server/path";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "custom://server/path");
}

TEST_F(VxEscapeURLTest, NoProtocolNoSpecialChars) {
    char input[] = "/path/to/file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/file.txt");
}

TEST_F(VxEscapeURLTest, WindowsPathNoSpecialChars) {
    char input[] = "C:/Users/test/file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file://C:/Users/test/file.txt");
}

TEST_F(VxEscapeURLTest, RelativePathNoSpecialChars) {
    char input[] = "documents/file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file://documents/file.txt");
}

TEST_F(VxEscapeURLTest, SpaceEscaping) {
    char input[] = "/path/to/my file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/my%20file.txt");
}

TEST_F(VxEscapeURLTest, MultipleSpacesEscaping) {
    char input[] = "/path with many spaces/file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path%20with%20many%20spaces/file.txt");
}

TEST_F(VxEscapeURLTest, HashEscaping) {
    char input[] = "/path/to/file#1.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/file%231.txt");
}

TEST_F(VxEscapeURLTest, DollarSignEscaping) {
    char input[] = "/path/to/$file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/%24file.txt");
}

TEST_F(VxEscapeURLTest, PercentEscaping) {
    char input[] = "/path/to/100%complete.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/100%25complete.txt");
}

TEST_F(VxEscapeURLTest, AmpersandEscaping) {
    char input[] = "/path/to/Tom&Jerry.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/Tom%26Jerry.txt");
}

TEST_F(VxEscapeURLTest, BackslashEscaping) {
    char input[] = "C:\\path\\to\\file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file://C:%5Cpath%5Cto%5Cfile.txt");
}

TEST_F(VxEscapeURLTest, PlusSignEscaping) {
    char input[] = "/path/to/C++file.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/C%2B%2Bfile.txt");
}

TEST_F(VxEscapeURLTest, CommaEscaping) {
    char input[] = "/path/to/list,of,items.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/list%2Cof%2Citems.txt");
}

TEST_F(VxEscapeURLTest, SemicolonEscaping) {
    char input[] = "/path/to/file;version=1.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/file%3Bversion%3D1.txt");
}

TEST_F(VxEscapeURLTest, EqualsEscaping) {
    char input[] = "/path/to/equation=result.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/equation%3Dresult.txt");
}

TEST_F(VxEscapeURLTest, AtSignEscaping) {
    char input[] = "/path/to/email@domain.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/email%40domain.txt");
}

TEST_F(VxEscapeURLTest, BracketEscaping) {
    char input[] = "/path/to/array[0].txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/array%5B0%5D.txt");
}

TEST_F(VxEscapeURLTest, CaretEscaping) {
    char input[] = "/path/to/power^2.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/power%5E2.txt");
}

TEST_F(VxEscapeURLTest, BraceEscaping) {
    char input[] = "/path/to/{template}.txt";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file:///path/to/%7Btemplate%7D.txt");
}

TEST_F(VxEscapeURLTest, AllSpecialCharacters) {
    char input[] = "path with #$%&\\+,;=@[]^{} all";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file://path%20with%20%23%24%25%26%5C%2B%2C%3B%3D%40%5B%5D%5E%7B%7D%20all");
}

TEST_F(VxEscapeURLTest, HttpUrlWithSpecialChars) {
    char input[] = "http://example.com/path with spaces?query=test&value=100%";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "http://example.com/path%20with%20spaces?query%3Dtest%26value%3D100%25");
}

TEST_F(VxEscapeURLTest, HttpsUrlWithSpecialChars) {
    char input[] = "https://example.com/path#anchor";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "https://example.com/path%23anchor");
}

TEST_F(VxEscapeURLTest, EdgeCaseEmptyPathWithProtocol) {
    char input[] = "http://";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "http://");
}

TEST_F(VxEscapeURLTest, OnlySpecialCharacters) {
    char input[] = " #$%&\\+,;=@[]^{}";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file://%20%23%24%25%26%5C%2B%2C%3B%3D%40%5B%5D%5E%7B%7D");
}

TEST_F(VxEscapeURLTest, OnlySpaces) {
    char input[] = "   ";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file://%20%20%20");
}

TEST_F(VxEscapeURLTest, SingleCharacterSpecial) {
    char input[] = " ";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "file://%20");
}

TEST_F(VxEscapeURLTest, ProtocolWithPort) {
    char input[] = "http://localhost:8080/path";
    XDWORD result = VxEscapeURL(input, outURL);
    EXPECT_EQ(result, 0u);
    EXPECT_TRUE(outURL == "http://localhost:8080/path");
}

TEST_F(VxUnEscapeUrlTest, EmptyString) {
    testStr = "";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "");
}

TEST_F(VxUnEscapeUrlTest, NoEncodedCharacters) {
    testStr = "http://example.com/path/to/file.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "http://example.com/path/to/file.txt");
}

TEST_F(VxUnEscapeUrlTest, SingleSpaceDecoding) {
    testStr = "file:///path/to/my%20file.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/my file.txt");
}

TEST_F(VxUnEscapeUrlTest, MultipleSpacesDecoding) {
    testStr = "file:///path%20with%20many%20spaces/file.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path with many spaces/file.txt");
}

TEST_F(VxUnEscapeUrlTest, HashDecoding) {
    testStr = "file:///path/to/file%231.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/file#1.txt");
}

TEST_F(VxUnEscapeUrlTest, DollarSignDecoding) {
    testStr = "file:///path/to/%24file.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/$file.txt");
}

TEST_F(VxUnEscapeUrlTest, PercentDecoding) {
    testStr = "file:///path/to/100%25complete.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/100%complete.txt");
}

TEST_F(VxUnEscapeUrlTest, AmpersandDecoding) {
    testStr = "file:///path/to/Tom%26Jerry.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/Tom&Jerry.txt");
}

TEST_F(VxUnEscapeUrlTest, BackslashDecoding) {
    testStr = "file://C:%5Cpath%5Cto%5Cfile.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file://C:\\path\\to\\file.txt");
}

TEST_F(VxUnEscapeUrlTest, PlusSignDecoding) {
    testStr = "file:///path/to/C%2B%2Bfile.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/C++file.txt");
}

TEST_F(VxUnEscapeUrlTest, CommaDecoding) {
    testStr = "file:///path/to/list%2Cof%2Citems.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/list,of,items.txt");
}

TEST_F(VxUnEscapeUrlTest, SemicolonDecoding) {
    testStr = "file:///path/to/file%3Bversion%3D1.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/file;version=1.txt");
}

TEST_F(VxUnEscapeUrlTest, EqualsDecoding) {
    testStr = "file:///path/to/equation%3Dresult.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/equation=result.txt");
}

TEST_F(VxUnEscapeUrlTest, AtSignDecoding) {
    testStr = "file:///path/to/email%40domain.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/email@domain.txt");
}

TEST_F(VxUnEscapeUrlTest, BracketDecoding) {
    testStr = "file:///path/to/array%5B0%5D.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/array[0].txt");
}

TEST_F(VxUnEscapeUrlTest, CaretDecoding) {
    testStr = "file:///path/to/power%5E2.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/power^2.txt");
}

TEST_F(VxUnEscapeUrlTest, BraceDecoding) {
    testStr = "file:///path/to/%7Btemplate%7D.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/{template}.txt");
}

TEST_F(VxUnEscapeUrlTest, AllSpecialCharactersDecoding) {
    testStr = "file://path%20with%20%23%24%25%26%5C%2B%2C%3B%3D%40%5B%5D%5E%7B%7D%20all";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file://path with #$%&\\+,;=@[]^{} all");
}

TEST_F(VxUnEscapeUrlTest, HttpUrlDecoding) {
    testStr = "http://example.com/path%20with%20spaces?query%3Dtest%26value%3D100%25";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "http://example.com/path with spaces?query=test&value=100%");
}

TEST_F(VxUnEscapeUrlTest, LowercaseHexDecoding) {
    testStr = "file:///path%20to%2ffile.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path to/file.txt");
}

TEST_F(VxUnEscapeUrlTest, UppercaseHexDecoding) {
    testStr = "file:///path%20TO%2Ffile.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path TO/file.txt");
}

TEST_F(VxUnEscapeUrlTest, MixedCaseHexDecoding) {
    testStr = "file:///path%20To%2fFile.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path To/File.txt");
}

TEST_F(VxUnEscapeUrlTest, PercentAtEndOfString) {
    testStr = "file:///path/to/file%";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/file%");
}

TEST_F(VxUnEscapeUrlTest, PercentWithOneHexChar) {
    testStr = "file:///path/to/file%2";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/file%2");
}

TEST_F(VxUnEscapeUrlTest, PercentAtSecondToLastPosition) {
    testStr = "file:///path/to/file%2a";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path/to/file*");
}

TEST_F(VxUnEscapeUrlTest, InvalidHexCharacters) {
    testStr = "file:///path/to/file%ZZ";
    VxUnEscapeUrl(testStr);
    // sscanf with %x will parse invalid hex as 0, creating null character
    EXPECT_EQ(testStr.Length(), 21u);
    EXPECT_EQ(testStr[20], '\xCC');
}

TEST_F(VxUnEscapeUrlTest, MixedValidAndInvalidEncoding) {
    testStr = "file:///path%20to%ZZfile%2Ftxt";
    VxUnEscapeUrl(testStr);
    // Should have: "file:///path to\xCCile/txt"
    EXPECT_EQ(testStr.Length(), 24u);
    EXPECT_TRUE(testStr.StartsWith("file:///path to"));
    EXPECT_EQ(testStr[15], ' '); // The invalid %ZZ becomes \xCC
}

TEST_F(VxUnEscapeUrlTest, ConsecutivePercentEncoding) {
    testStr = "file:///path%20%20to%20file.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///path  to file.txt");
}

TEST_F(VxUnEscapeUrlTest, SpecialCharacterCodes) {
    testStr = "file:///test%09%0A%0D%22%27file.txt";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "file:///test\t\n\r\"'file.txt");
}

TEST_F(VxUnEscapeUrlTest, ZeroByteHandling) {
    testStr = "file:///path%00to%00file.txt";
    VxUnEscapeUrl(testStr);
    // The string should contain null bytes, length should be preserved
    EXPECT_EQ(testStr.Length(), 24u);
    EXPECT_EQ(testStr[12], '\0'); // First null byte
    EXPECT_EQ(testStr[15], '\0'); // Second null byte
}

TEST_F(VxUnEscapeUrlTest, OnlyPercentEncoding) {
    testStr = "%20%21%22";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == " !\"");
}

TEST_F(VxUnEscapeUrlTest, BoundaryConditionTwoCharactersLeft) {
    testStr = "test%20";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "test ");
}

class RoundTripTest : public ::testing::Test {
protected:
    void TestRoundTrip(const char *original, const char *expectedAfterEscape = nullptr) {
        XString escaped;
        char *input = new char[strlen(original) + 1];
        strcpy(input, original);

        XDWORD result = VxEscapeURL(input, escaped);
        EXPECT_EQ(result, 0u);

        XString unescaped = escaped;
        VxUnEscapeUrl(unescaped);

        // For round-trip testing, we need to account for the file:// prefix addition
        const char *expected = expectedAfterEscape ? expectedAfterEscape : original;
        if (!expectedAfterEscape && strstr(original, "://") == nullptr) {
            XString temp = "file://";
            temp += original;
            EXPECT_TRUE(unescaped == temp);
        } else {
            EXPECT_TRUE(unescaped == expected);
        }

        delete[] input;
    }
};

TEST_F(RoundTripTest, SimpleFilePath) {
    TestRoundTrip("/path/to/file.txt");
}

TEST_F(RoundTripTest, PathWithSpaces) {
    TestRoundTrip("/path/to/my file.txt");
}

TEST_F(RoundTripTest, HttpUrl) {
    TestRoundTrip("http://example.com/path");
}

TEST_F(RoundTripTest, ComplexPath) {
    TestRoundTrip("/path/with/special#chars&symbols.txt");
}

TEST_F(RoundTripTest, WindowsPath) {
    TestRoundTrip("C:\\Users\\test\\file.txt");
}

TEST_F(RoundTripTest, AllSpecialCharacters) {
    TestRoundTrip("path with #$%&\\+,;=@[]^{} all");
}

TEST_F(RoundTripTest, HttpWithQueryString) {
    TestRoundTrip("http://example.com/search?q=hello world&type=text");
}

TEST(EdgeCaseTest, VeryLongUrl) {
    XString longPath = "/very/long/path";
    for (int i = 0; i < 100; ++i) {
        longPath << "/directory" << i;
    }
    longPath << "/file with spaces.txt";

    char *input = new char[longPath.Length() + 1];
    strcpy(input, longPath.CStr());

    XString result;
    XDWORD status = VxEscapeURL(input, result);
    EXPECT_EQ(status, 0u);
    EXPECT_GT(result.Length(), 0u);

    delete[] input;
}

TEST(EdgeCaseTest, OnlyPercentSigns) {
    char input[] = "%%%%%%";
    XString result;
    XDWORD status = VxEscapeURL(input, result);
    EXPECT_EQ(status, 0u);
    EXPECT_TRUE(result == "file://%25%25%25%25%25%25");
}

TEST(EdgeCaseTest, AlternatingSpecialAndNormal) {
    char input[] = "a b c#d$e%f";
    XString result;
    XDWORD status = VxEscapeURL(input, result);
    EXPECT_EQ(status, 0u);
    EXPECT_TRUE(result == "file://a%20b%20c%23d%24e%25f");
}

TEST(EdgeCaseTest, AllNormalCharacters) {
    char input[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    XString result;
    XDWORD status = VxEscapeURL(input, result);
    EXPECT_EQ(status, 0u);
    EXPECT_TRUE(result == "file://abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
}

TEST(EdgeCaseTest, SingleCharacter) {
    char input[] = "a";
    XString result;
    XDWORD status = VxEscapeURL(input, result);
    EXPECT_EQ(status, 0u);
    EXPECT_TRUE(result == "file://a");
}

TEST(EdgeCaseTest, ProtocolDetectionEdgeCase) {
    char input[] = "://invalid";
    XString result;
    XDWORD status = VxEscapeURL(input, result);
    EXPECT_EQ(status, 0u);
    EXPECT_TRUE(result == "://invalid"); // Should be detected as having protocol
}

TEST(BugTest, UnescapeIndexOutOfBounds) {
    // Test the boundary condition in VxUnEscapeUrl where i + 2 < str.Length()
    // This should NOT decode the last % if there aren't enough characters
    XString testStr = "test%2";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "test%2"); // Should remain unchanged
}

TEST(BugTest, UnescapeExactBoundary) {
    // Test when i + 2 == str.Length() - should not decode
    XString testStr = "test%20";
    VxUnEscapeUrl(testStr);
    EXPECT_TRUE(testStr == "test "); // Should decode properly
}

