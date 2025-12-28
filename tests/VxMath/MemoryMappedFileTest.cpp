//
// Test suite for VxMemoryMappedFile
//
// This test file requires the Google Test framework and a temporary
// directory for filesystem operations. It covers:
// 1. Successful mapping of valid files (including empty ones).
// 2. Data integrity verification after mapping.
// 3. Correct handling of file size reporting.
// 4. Graceful failure when mapping non-existent files or directories.
// 5. Proper resource (file handle) release upon destruction.
//

#include "VxMath.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Use std::filesystem for test I/O; it is more reliable across platforms.

// Helper function to create a file with specific content
void CreateFileWithContent(const std::string& path, const std::string& content) {
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error(std::string("Failed to open file for writing: ") + path);
    }
    file.write(content.c_str(), content.length());
    file.flush();
    file.close();
}

// Helper function to remove a file
void RemoveFile(const std::string& path) {
    std::remove(path.c_str());
}

// A test fixture for managing a temporary directory for file I/O tests
class VxMemoryMappedFileTest : public ::testing::Test {
protected:
    // This function is called before each test is run
    void SetUp() override {
        const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string uniqueName =
            std::string("VxMMF_TestDir_") + info->test_suite_name() + "_" + info->name();
        temp_dir = (std::filesystem::temp_directory_path() / uniqueName).string();

        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
        std::filesystem::create_directories(temp_dir, ec);
        ASSERT_FALSE(ec) << "Failed to create temp directory: " << temp_dir;
    }

    // This function is called after each test is run
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(temp_dir, ec);
    }

    // Helper to get the full path for a file in our temp directory
    std::string GetFullPath(const std::string& filename) const {
        return (std::filesystem::path(temp_dir) / filename).string();
    }

protected:
    std::string temp_dir;
};

// Test Case 1: Successful mapping of a valid, non-empty file
TEST_F(VxMemoryMappedFileTest, MapsValidFileSuccessfully) {
    const std::string filename = GetFullPath("test.txt");
    const std::string content = "Hello Mapped World!";
    CreateFileWithContent(filename, content);

    {
        std::error_code ec;
        const auto size = std::filesystem::file_size(filename, ec);
        ASSERT_FALSE(ec) << "Failed to stat test file for sanity check: " << filename << " (" << ec.message() << ")";
        ASSERT_EQ(static_cast<size_t>(size), content.length()) << "Sanity check: file size mismatch before mapping.";
    }

    // Construct the object with the path to the valid file
    VxMemoryMappedFile mmf(const_cast<char*>(filename.c_str()));

    // 1. Verify the mapping was successful
    ASSERT_TRUE(mmf.IsValid()) << "MMF should be valid for an existing file. ErrorType=" << (int)mmf.GetErrorType() << " FileSize=" << mmf.GetFileSize();
    ASSERT_EQ(mmf.GetErrorType(), VxMMF_NoError) << "Error type should be NoError for a successful mapping.";

    // 2. Verify the file size is reported correctly
    ASSERT_EQ(mmf.GetFileSize(), content.length()) << "Reported file size should match actual content length.";

    // 3. Verify the memory content is correct
    void* base_ptr = mmf.GetBase();
    ASSERT_NE(base_ptr, nullptr) << "Base pointer should not be null for a valid mapping.";

    // Compare the memory content with the original string
    EXPECT_EQ(0, memcmp(base_ptr, content.c_str(), content.length())) << "Memory content does not match file content.";
}

// Test Case 2: Handling of zero-byte (empty) files
TEST_F(VxMemoryMappedFileTest, HandlesEmptyFileGracefully) {
    const std::string filename = GetFullPath("empty.txt");
    CreateFileWithContent(filename, ""); // Create a 0-byte file

    VxMemoryMappedFile mmf(const_cast<char*>(filename.c_str()));

    // 1. Verify the mapping is considered valid
    ASSERT_FALSE(mmf.IsValid()) << "MMF should not be valid for an empty file.";
    ASSERT_EQ(mmf.GetErrorType(), VxMMF_FileMapping) << "Error type should be FileMapping for an empty file.";

    // 2. Verify the file size is 0
    ASSERT_EQ(mmf.GetFileSize(), 0) << "File size for an empty file should be 0.";

    // 3. Verify GetBase() returns nullptr
    ASSERT_EQ(mmf.GetBase(), nullptr) << "Base pointer for an empty file should be null.";
}

// Test Case 3: Error handling for non-existent files
TEST_F(VxMemoryMappedFileTest, FailsGracefullyForNonExistentFile) {
    const std::string filename = GetFullPath("non_existent_file.txt");

    // Attempt to map a file that does not exist
    VxMemoryMappedFile mmf(const_cast<char*>(filename.c_str()));

    // 1. Verify the mapping is marked as invalid
    ASSERT_FALSE(mmf.IsValid()) << "MMF should be invalid for a non-existent file.";

    // 2. Verify the correct error type is reported
    ASSERT_EQ(mmf.GetErrorType(), VxMMF_FileOpen) << "Error type should be FileOpen.";

    // 3. Verify size and base pointer reflect the failure
    ASSERT_EQ(mmf.GetFileSize(), 0) << "File size should be 0 on failure.";
    ASSERT_EQ(mmf.GetBase(), nullptr) << "Base pointer should be null on failure.";
}

// Test Case 4: Error handling when trying to map a directory
TEST_F(VxMemoryMappedFileTest, FailsGracefullyForDirectoryPath) {
    // Use the path to the temporary directory itself
    const std::string& directory_path = temp_dir;

    VxMemoryMappedFile mmf(const_cast<char*>(directory_path.c_str()));

    // 1. Verify the mapping is invalid
    ASSERT_FALSE(mmf.IsValid()) << "MMF should be invalid when given a directory path.";

    // 2. Verify the error type indicates a file opening problem
    ASSERT_EQ(mmf.GetErrorType(), VxMMF_FileOpen) << "Error type should be FileOpen for a directory.";

    // 3. Verify size and base pointer are in a failed state
    ASSERT_EQ(mmf.GetFileSize(), 0);
    ASSERT_EQ(mmf.GetBase(), nullptr);
}

// Test Case 5: Ensure resources are released upon destruction
TEST_F(VxMemoryMappedFileTest, DestructorReleasesFileHandle) {
    const std::string filename = GetFullPath("test.txt");
    CreateFileWithContent(filename, "File to be deleted.");

    {
        std::error_code ec;
        const auto size = std::filesystem::file_size(filename, ec);
        ASSERT_FALSE(ec) << "Failed to stat test file for sanity check: " << filename << " (" << ec.message() << ")";
        ASSERT_GT(size, 0u) << "Sanity check: expected non-empty file.";
    }

    {
        // Create the MMF object in a limited scope
        VxMemoryMappedFile mmf(const_cast<char*>(filename.c_str()));
        ASSERT_TRUE(mmf.IsValid()) << "File must be mapped correctly before testing destruction. ErrorType=" << (int)mmf.GetErrorType() << " FileSize=" << mmf.GetFileSize();
    } // mmf is destroyed here, its destructor runs

    // After the MMF object is destroyed, we should be able to delete the file.
    // If the file handle was not released, this would fail on some operating systems.
    int remove_result = std::remove(filename.c_str());

    ASSERT_EQ(remove_result, 0) << "File could not be deleted, suggesting the file handle was not released by the destructor.";
}

// Test Case 6: Destructor handles an invalid state without crashing
TEST_F(VxMemoryMappedFileTest, DestructorHandlesInvalidStateSafely) {
    // This test ensures that if the MMF object fails to initialize,
    // its destructor can still run without causing a crash (e.g., by trying
    // to close null handles).

    {
        VxMemoryMappedFile mmf(const_cast<char*>("non_existent_file.txt"));
        ASSERT_FALSE(mmf.IsValid());
    } // Destructor runs here

    // The test passes if it completes without crashing.
    SUCCEED();
}