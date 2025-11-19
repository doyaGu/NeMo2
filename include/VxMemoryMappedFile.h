#ifndef VXMEMORYMAPPEDFILE_H
#define VXMEMORYMAPPEDFILE_H

#include "VxMathDefines.h"

/**
 * @enum VxMMF_Error
 * @brief Defines possible error codes when opening a memory-mapped file.
 * @see VxMemoryMappedFile::GetErrorType
 */
enum VxMMF_Error {
    VxMMF_NoError,     ///< No error occurred.
    VxMMF_FileOpen,    ///< The specified file could not be opened.
    VxMMF_FileMapping, ///< A file mapping object could not be created.
    VxMMF_MapView      ///< A view of the file could not be mapped into the address space.
};

/**
 * @brief Utility class for read-only access to a file via memory mapping.
 *
 * @remarks
 * This class maps a file's contents into a memory buffer, allowing for efficient
 * read-only access as if it were an in-memory array of bytes. The resources
 * are automatically released when the object is destroyed.
 *
 * @example
 * @code
 * VxMemoryMappedFile mappedFile("C:\\data\\myfile.dat");
 * if (mappedFile.IsValid()) {
 *     DWORD fileSize = mappedFile.GetFileSize();
 *     const BYTE* buffer = (const BYTE*)mappedFile.GetBase();
 *     // 'buffer' now points to the file's content and can be read.
 * }
 * @endcode
 */
class VxMemoryMappedFile {
public:
    /**
     * @brief Constructs the object and attempts to map the specified file into memory.
     * @param pszFileName The path to the file to be mapped.
     */
    VX_EXPORT VxMemoryMappedFile(char *pszFileName);

    /**
     * @brief Destructor. Unmaps the file view and closes file handles.
     */
    VX_EXPORT ~VxMemoryMappedFile();

    /**
     * @brief Returns a pointer to the start of the mapped memory buffer.
     * @return A void pointer to the base of the mapped file. Returns NULL on failure.
     * @remarks The returned pointer provides read-only access to the file's data.
     * Do not attempt to write to this memory or deallocate it.
     */
    VX_EXPORT void *GetBase();

    /**
     * @brief Returns the size of the mapped file in bytes.
     * @return The size of the file.
     */
    VX_EXPORT XULONG GetFileSize();

    /**
     * @brief Checks if the file was successfully opened and mapped.
     * @return TRUE if the mapping is valid, FALSE otherwise.
     */
    VX_EXPORT XBOOL IsValid();

    /**
     * @brief Returns an error code indicating the reason for failure if IsValid() is FALSE.
     * @return A value from the VxMMF_Error enumeration.
     */
    VX_EXPORT VxMMF_Error GetErrorType();

private:
    GENERIC_HANDLE m_hFile;        ///< Handle to the open file.
    GENERIC_HANDLE m_hFileMapping; ///< Handle to the file-mapping object.
    void *m_pMemoryMappedFileBase; ///< Base address of the mapped view of the file.
    XULONG m_cbFile;               ///< The size of the file.
    VxMMF_Error m_errCode;         ///< Stores the error code if an operation fails.
};

#endif // VXMEMORYMAPPEDFILE_H
