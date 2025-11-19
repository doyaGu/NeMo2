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
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#define RMDIR(path) _rmdir(path)
#else
#include <sys/stat.h>
#include <unistd.h>
#define MKDIR(path) mkdir(path, 0755)
#define RMDIR(path) rmdir(path)
#endif

// Helper function to create a file with specific content
void CreateFileWithContent(const std::string& path, const std::string& content) {
    std::ofstream file(path, std::ios::binary);
    if (file) {
        file.write(content.c_str(), content.length());
    }
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
        // Create a unique temporary directory for this test run
        temp_dir = "VxMMF_TestDir";

        // Clean up any old directory first
        RemoveDirectoryRecursive(temp_dir);

        // Create the new directory
        MKDIR(temp_dir.c_str());
    }

    // This function is called after each test is run
    void TearDown() override {
        // Clean up the temporary directory and all its contents
        RemoveDirectoryRecursive(temp_dir);
    }

    // Helper to get the full path for a file in our temp directory
    std::string GetFullPath(const std::string& filename) const {
        return temp_dir + "/" + filename;
    }

private:
    void RemoveDirectoryRecursive(const std::string& path) {
        // This is a simplified recursive delete for test cleanup.
        // A more robust implementation would use OS-specific directory iteration.
        // For this test's purpose, we just need to delete known files and then the dir.
        RemoveFile(GetFullPath("test.txt"));
        RemoveFile(GetFullPath("empty.txt"));
        RemoveFile(GetFullPath("large_file.bin"));
        RMDIR(path.c_str());
    }

protected:
    std::string temp_dir;
};

// Test Case 1: Successful mapping of a valid, non-empty file
TEST_F(VxMemoryMappedFileTest, MapsValidFileSuccessfully) {
    const std::string filename = GetFullPath("test.txt");
    const std::string content = "Hello Mapped World!";
    CreateFileWithContent(filename, content);

    // Construct the object with the path to the valid file
    VxMemoryMappedFile mmf(const_cast<char*>(filename.c_str()));

    // 1. Verify the mapping was successful
    ASSERT_TRUE(mmf.IsValid()) << "MMF should be valid for an existing file.";
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
        // Create the MMF object in a limited scope
        VxMemoryMappedFile mmf(const_cast<char*>(filename.c_str()));
        ASSERT_TRUE(mmf.IsValid()) << "File must be mapped correctly before testing destruction.";
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