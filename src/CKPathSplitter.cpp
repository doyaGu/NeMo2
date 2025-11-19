#include "CKPathSplitter.h"

#include <stdlib.h>

CKPathSplitter::CKPathSplitter(char *file) : m_Drive(), m_Dir(), m_Filename(), m_Ext() {
    if (file) {
        _splitpath_s(file, m_Drive, _MAX_DRIVE, m_Dir, _MAX_DIR,
                    m_Filename, _MAX_FNAME, m_Ext, _MAX_EXT);
    }
}

CKPathSplitter::~CKPathSplitter() {}

char *CKPathSplitter::GetDrive() {
    return m_Drive;
}

char *CKPathSplitter::GetDir() {
    return m_Dir;
}

char *CKPathSplitter::GetName() {
    return m_Filename;
}

char *CKPathSplitter::GetExtension() {
    return m_Ext;
}

CKPathMaker::CKPathMaker(char *Drive, char *Directory, char *Fname, char *Extension) : m_FileName() {
    _makepath_s(m_FileName, _MAX_PATH, Drive, Directory, Fname, Extension);
}

char *CKPathMaker::GetFileName() {
    return m_FileName;
}
