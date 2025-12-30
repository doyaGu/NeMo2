#include "VxMemoryMappedFile.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define INVALID_HANDLE_VALUE ((void*)(long)-1)

VxMemoryMappedFile::VxMemoryMappedFile(char *pszFileName)
    : m_hFile(INVALID_HANDLE_VALUE),
      m_hFileMapping(NULL),
      m_pMemoryMappedFileBase(NULL),
      m_cbFile(0),
      m_errCode(VxMMF_FileOpen) {

    int fd = open(pszFileName, O_RDONLY);
    if (fd == -1) {
        m_errCode = VxMMF_FileOpen;
        return;
    }
    m_hFile = (GENERIC_HANDLE)(intptr_t)fd;

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        m_hFile = INVALID_HANDLE_VALUE;
        m_errCode = VxMMF_FileOpen;
        return;
    }
    m_cbFile = static_cast<size_t>(sb.st_size);

    m_pMemoryMappedFileBase = mmap(NULL, m_cbFile, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m_pMemoryMappedFileBase == MAP_FAILED) {
        m_pMemoryMappedFileBase = NULL;
        close(fd);
        m_hFile = INVALID_HANDLE_VALUE;
        m_errCode = VxMMF_MapView;
        return;
    }

    // On POSIX, we can close the file descriptor after mmap
    // The mapping remains valid
    close(fd);
    m_hFile = INVALID_HANDLE_VALUE;

    m_errCode = VxMMF_NoError;
}

VxMemoryMappedFile::~VxMemoryMappedFile() {
    if (m_pMemoryMappedFileBase && m_cbFile > 0)
        munmap(m_pMemoryMappedFileBase, m_cbFile);

    m_errCode = VxMMF_FileOpen;
}

/***********************************************************************
Summary: Returns a pointer to the mapped memory buffer.
Remarks: The returned pointer should not be deleted nor should it be
used for writing purpose.
***********************************************************************/
void *VxMemoryMappedFile::GetBase() {
    return m_pMemoryMappedFileBase;
}

/***********************************************************************
Summary: Returns the file size in bytes.
***********************************************************************/
size_t VxMemoryMappedFile::GetFileSize() {
    return m_cbFile;
}

/***********************************************************************
Summary: Returns the file was successfully opened and mapped to a memory buffer.
***********************************************************************/
XBOOL VxMemoryMappedFile::IsValid() {
    return VxMMF_NoError == m_errCode;
}

/***********************************************************************
Summary: Returns whether there was an error opening the file.
***********************************************************************/
VxMMF_Error VxMemoryMappedFile::GetErrorType() {
    return m_errCode;
}
