#include "CKPathSplitter.h"

#include <stdlib.h>

#if !defined(_WIN32)
#include <ctype.h>
#include <string>

static void CopyComponent(char *dest, size_t dest_size, const std::string &src) {
    if (!dest || dest_size == 0) {
        return;
    }
    size_t count = src.size();
    if (count >= dest_size) {
        count = dest_size - 1;
    }
    if (count > 0) {
        memcpy(dest, src.data(), count);
    }
    dest[count] = '\0';
}

static void VxSplitPath(const char *path, char *drive, size_t drive_size, char *dir, size_t dir_size,
                 char *fname, size_t fname_size, char *ext, size_t ext_size) {
    std::string input = path ? path : "";
    std::string drive_part;
    std::string path_part = input;

    if (input.size() >= 2 && isalpha(static_cast<unsigned char>(input[0])) && input[1] == ':') {
        drive_part = input.substr(0, 2);
        path_part = input.substr(2);
    }

    size_t slash = path_part.find_last_of("/\\");
    std::string dir_part;
    std::string file_part = path_part;
    if (slash != std::string::npos) {
        dir_part = path_part.substr(0, slash + 1);
        file_part = path_part.substr(slash + 1);
    }

    std::string fname_part;
    std::string ext_part;
    size_t dot = file_part.find_last_of('.');
    if (dot != std::string::npos && dot != 0) {
        fname_part = file_part.substr(0, dot);
        ext_part = file_part.substr(dot);
    } else {
        fname_part = file_part;
    }

    CopyComponent(drive, drive_size, drive_part);
    CopyComponent(dir, dir_size, dir_part);
    CopyComponent(fname, fname_size, fname_part);
    CopyComponent(ext, ext_size, ext_part);
}

static void VxMakePath(char *path, size_t path_size, const char *drive, const char *dir,
                const char *fname, const char *ext) {
    std::string result;
    if (drive && *drive) {
        result += drive;
    }
    if (dir && *dir) {
        result += dir;
    }
    if (fname && *fname) {
        result += fname;
    }
    if (ext && *ext) {
        result += ext;
    }

    CopyComponent(path, path_size, result);
}

#define _splitpath_s VxSplitPath
#define _makepath_s VxMakePath
#endif

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
