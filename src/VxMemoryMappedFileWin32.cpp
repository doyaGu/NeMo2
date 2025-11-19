#include "VxMemoryMappedFile.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

VxMemoryMappedFile::VxMemoryMappedFile(char *pszFileName)
    : m_hFile(INVALID_HANDLE_VALUE),
      m_hFileMapping(NULL),
      m_pMemoryMappedFileBase(NULL),
      m_cbFile(0),
      m_errCode(VxMMF_FileOpen) {
    m_hFile = CreateFileA(pszFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m_hFile == INVALID_HANDLE_VALUE) {
        m_errCode = VxMMF_FileOpen;
        return;
    }

    m_cbFile = ::GetFileSize(m_hFile, NULL);
    m_hFileMapping = CreateFileMappingA(m_hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m_hFileMapping) {
        CloseHandle(m_hFile);
        m_hFile = INVALID_HANDLE_VALUE;
        m_errCode = VxMMF_FileMapping;
        return;
    }

    m_pMemoryMappedFileBase = MapViewOfFile(m_hFileMapping, FILE_MAP_READ, 0, 0, 0);
    if (!m_pMemoryMappedFileBase) {
        CloseHandle(m_hFileMapping);
        m_hFileMapping = NULL;
        CloseHandle(m_hFile);
        m_hFile = INVALID_HANDLE_VALUE;
        m_errCode = VxMMF_MapView;
    }

    m_errCode = VxMMF_NoError;
}

VxMemoryMappedFile::~VxMemoryMappedFile() {
    if (m_pMemoryMappedFileBase)
        UnmapViewOfFile(m_pMemoryMappedFileBase);

    if (m_hFileMapping)
        CloseHandle(m_hFileMapping);

    if (m_hFile != INVALID_HANDLE_VALUE)
        CloseHandle(m_hFile);

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
XULONG VxMemoryMappedFile::GetFileSize() {
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
