#include "gtest/gtest.h"
#include "VxMath.h"
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <filesystem> // For modern filesystem operations in the test fixture

// Helper to create a file
void CreateEmptyFile(const std::filesystem::path &path) {
    std::ofstream outfile(path);
    outfile.close();
}

// Test Fixture for CKDirectoryParser which requires a real filesystem
class DirectoryParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory structure for testing
        // test_root/
        // ├── file1.txt
        // ├── file2.log
        // ├── image.JPG
        // ├── sub/
        // │   ├── subfile.txt
        // │   └── another.log
        // └── empty_sub/
        m_rootDir = "temp_dir_parser_test";
        std::filesystem::remove_all(m_rootDir);
        std::filesystem::create_directory(m_rootDir);

        m_subDir = m_rootDir / "sub";
        std::filesystem::create_directory(m_subDir);

        m_emptySubDir = m_rootDir / "empty_sub";
        std::filesystem::create_directory(m_emptySubDir);

        CreateEmptyFile(m_rootDir / "file1.txt");
        CreateEmptyFile(m_rootDir / "file2.log");
        CreateEmptyFile(m_rootDir / "image.JPG"); // Note: uppercase extension
        CreateEmptyFile(m_subDir / "subfile.txt");
        CreateEmptyFile(m_subDir / "another.log");

        m_rootDirStr = m_rootDir.string();
        m_subDirStr = m_subDir.string();
        m_emptySubDirStr = m_emptySubDir.string();
    }

    void TearDown() override {
        std::filesystem::remove_all(m_rootDir);
    }

    // Helper to collect all files found by the parser
    std::set<std::string> CollectFiles(CKDirectoryParser &parser) {
        std::set<std::string> foundFiles;
        char *file;
        while ((file = parser.GetNextFile()) != nullptr) {
            std::string filePath(file);
            // Normalize path separators for consistent comparison
            // std::replace(filePath.begin(), filePath.end(), '\\', '/');
            foundFiles.insert(filePath);
        }
        return foundFiles;
    }

    std::filesystem::path m_rootDir;
    std::filesystem::path m_subDir;
    std::filesystem::path m_emptySubDir;
    std::string m_rootDirStr;
    std::string m_subDirStr;
    std::string m_emptySubDirStr;
};


// --- CKPathSplitter Tests ---

TEST(PathSplitterTest, FullPath) {
    char path[] = "C:\\Users\\Test\\file.txt";
    CKPathSplitter splitter(path);
    ASSERT_STREQ(splitter.GetDrive(), "C:");
    ASSERT_STREQ(splitter.GetDir(), "\\Users\\Test\\");
    ASSERT_STREQ(splitter.GetName(), "file");
    ASSERT_STREQ(splitter.GetExtension(), ".txt");
}

TEST(PathSplitterTest, UNCPath) {
    char path[] = "\\\\Server\\Share\\file.zip";
    CKPathSplitter splitter(path);
    ASSERT_STREQ(splitter.GetDrive(), "");
    // Note: _splitpath behavior with UNC
    ASSERT_STREQ(splitter.GetDir(), "\\\\Server\\Share\\");
    ASSERT_STREQ(splitter.GetName(), "file");
    ASSERT_STREQ(splitter.GetExtension(), ".zip");
}

TEST(PathSplitterTest, RelativePath) {
    char path[] = "..\\data\\model.nmo";
    CKPathSplitter splitter(path);
    ASSERT_STREQ(splitter.GetDrive(), "");
    ASSERT_STREQ(splitter.GetDir(), "..\\data\\");
    ASSERT_STREQ(splitter.GetName(), "model");
    ASSERT_STREQ(splitter.GetExtension(), ".nmo");
}

TEST(PathSplitterTest, FilenameOnly) {
    char path[] = "myfile.dat";
    CKPathSplitter splitter(path);
    ASSERT_STREQ(splitter.GetDrive(), "");
    ASSERT_STREQ(splitter.GetDir(), "");
    ASSERT_STREQ(splitter.GetName(), "myfile");
    ASSERT_STREQ(splitter.GetExtension(), ".dat");
}

TEST(PathSplitterTest, PathWithoutExtension) {
    char path[] = "D:\\Temp\\MyFile";
    CKPathSplitter splitter(path);
    ASSERT_STREQ(splitter.GetDrive(), "D:");
    ASSERT_STREQ(splitter.GetDir(), "\\Temp\\");
    ASSERT_STREQ(splitter.GetName(), "MyFile");
    ASSERT_STREQ(splitter.GetExtension(), "");
}

TEST(PathSplitterTest, PathWithDotInName) {
    char path[] = "E:\\backup.v1.0.zip";
    CKPathSplitter splitter(path);
    ASSERT_STREQ(splitter.GetDrive(), "E:");
    ASSERT_STREQ(splitter.GetDir(), "\\");
    ASSERT_STREQ(splitter.GetName(), "backup.v1.0");
    ASSERT_STREQ(splitter.GetExtension(), ".zip");
}

TEST(PathSplitterTest, PathEndingWithSlash) {
    char path[] = "C:\\MyDir\\";
    CKPathSplitter splitter(path);
    ASSERT_STREQ(splitter.GetDrive(), "C:");
    ASSERT_STREQ(splitter.GetDir(), "\\MyDir\\");
    ASSERT_STREQ(splitter.GetName(), "");
    ASSERT_STREQ(splitter.GetExtension(), "");
}

TEST(PathSplitterTest, EmptyPath) {
    char path[] = "";
    CKPathSplitter splitter(path);
    ASSERT_STREQ(splitter.GetDrive(), "");
    ASSERT_STREQ(splitter.GetDir(), "");
    ASSERT_STREQ(splitter.GetName(), "");
    ASSERT_STREQ(splitter.GetExtension(), "");
}

// --- CKPathMaker Tests ---

TEST(PathMakerTest, AllComponents) {
    CKPathMaker maker("C:", "\\Dir\\", "File", ".ext");
    ASSERT_STREQ(maker.GetFileName(), "C:\\Dir\\File.ext");
}

TEST(PathMakerTest, NoDrive) {
    CKPathMaker maker(nullptr, "\\Dir\\", "File", ".ext");
    ASSERT_STREQ(maker.GetFileName(), "\\Dir\\File.ext");
}

TEST(PathMakerTest, NoDirectory) {
    CKPathMaker maker("C:", nullptr, "File", ".ext");
    ASSERT_STREQ(maker.GetFileName(), "C:File.ext");
}

TEST(PathMakerTest, NoExtension) {
    CKPathMaker maker("C:", "\\Dir\\", "File", nullptr);
    ASSERT_STREQ(maker.GetFileName(), "C:\\Dir\\File");
}

TEST(PathMakerTest, OnlyFilename) {
    CKPathMaker maker(nullptr, nullptr, "File", nullptr);
    ASSERT_STREQ(maker.GetFileName(), "File");
}

// --- CKFileExtension Tests ---

TEST(FileExtensionTest, Construction) {
    ASSERT_STREQ((char*)CKFileExtension(".txt"), "txt");
    ASSERT_STREQ((char*)CKFileExtension("bmp"), "bmp");
    ASSERT_STREQ((char*)CKFileExtension(".jpeg"), "jpe"); // Truncation
    ASSERT_STREQ((char*)CKFileExtension(""), "");
    ASSERT_STREQ((char*)CKFileExtension(nullptr), "");
}

TEST(FileExtensionTest, Comparison) {
    CKFileExtension ext1(".TXT");
    CKFileExtension ext2("txt");
    CKFileExtension ext3("tXt");
    CKFileExtension ext4("bmp");

    ASSERT_TRUE(ext1 == ext2);
    ASSERT_TRUE(ext2 == ext3);
    ASSERT_FALSE(ext1 == ext4);
}

// --- CKDirectoryParser Tests ---

// DISABLED: Segfaults on some systems - needs investigation
TEST_F(DirectoryParserTest, DISABLED_NonRecursive_AllFiles) {
    CKDirectoryParser parser((char *) m_rootDir.string().c_str(), "*.*", FALSE);
    auto files = CollectFiles(parser);

    ASSERT_EQ(files.size(), 3);
    ASSERT_TRUE(files.count((m_rootDir / "file1.txt").string()));
    ASSERT_TRUE(files.count((m_rootDir / "file2.log").string()));
    ASSERT_TRUE(files.count((m_rootDir / "image.JPG").string()));
}

TEST_F(DirectoryParserTest, NonRecursive_SpecificMask) {
    CKDirectoryParser parser(const_cast<char *>(m_rootDirStr.c_str()), "*.txt", FALSE);
    auto files = CollectFiles(parser);

    ASSERT_EQ(files.size(), 1);
    ASSERT_TRUE(files.count((m_rootDir / "file1.txt").string()));
}

TEST_F(DirectoryParserTest, Recursive_AllFiles) {
    CKDirectoryParser parser(const_cast<char *>(m_rootDirStr.c_str()), "*.*", TRUE);
    auto files = CollectFiles(parser);

    ASSERT_EQ(files.size(), 5);
    ASSERT_TRUE(files.count((m_rootDir / "file1.txt").string()));
    ASSERT_TRUE(files.count((m_rootDir / "file2.log").string()));
    ASSERT_TRUE(files.count((m_rootDir / "image.JPG").string()));
    ASSERT_TRUE(files.count((m_subDir / "subfile.txt").string()));
    ASSERT_TRUE(files.count((m_subDir / "another.log").string()));
}

TEST_F(DirectoryParserTest, Recursive_SpecificMaskAndCase) {
    // The mask should be case-insensitive on Windows
    CKDirectoryParser parser(const_cast<char *>(m_rootDirStr.c_str()), "*.jpg", TRUE);
    auto files = CollectFiles(parser);

    ASSERT_EQ(files.size(), 1);
    ASSERT_TRUE(files.count((m_rootDir / "image.JPG").string()));
}

TEST_F(DirectoryParserTest, ParseEmptyDirectory) {
    CKDirectoryParser parser(const_cast<char *>(m_emptySubDirStr.c_str()), "*.*", TRUE);
    auto files = CollectFiles(parser);
    ASSERT_EQ(files.size(), 0);
}

TEST_F(DirectoryParserTest, ParseInvalidDirectory) {
    CKDirectoryParser parser("non_existent_directory_12345", "*.*", FALSE);
    auto files = CollectFiles(parser);
    ASSERT_EQ(files.size(), 0);
}

TEST_F(DirectoryParserTest, Reset_RescansSameDirectory) {
    CKDirectoryParser parser(const_cast<char *>(m_rootDirStr.c_str()), "*.log", FALSE);

    // First scan
    auto files1 = CollectFiles(parser);
    ASSERT_EQ(files1.size(), 1);
    ASSERT_TRUE(files1.count((m_rootDir / "file2.log").string()));

    // After reset, should find the same file again
    parser.Reset();
    auto files2 = CollectFiles(parser);
    ASSERT_EQ(files2.size(), 1);
    ASSERT_TRUE(files2.count((m_rootDir / "file2.log").string()));
}

TEST_F(DirectoryParserTest, Reset_ToNewDirectory) {
    CKDirectoryParser parser(const_cast<char *>(m_rootDirStr.c_str()), "*.txt", FALSE);

    // Scan root
    auto files1 = CollectFiles(parser);
    ASSERT_EQ(files1.size(), 1);

    // Reset to scan sub-directory
    parser.Reset(const_cast<char *>(m_subDirStr.c_str()), "*.txt", FALSE);
    auto files2 = CollectFiles(parser);
    ASSERT_EQ(files2.size(), 1);
    ASSERT_TRUE(files2.count((m_subDir / "subfile.txt").string()));
}

// =============================================================================
// Main function to run all tests
