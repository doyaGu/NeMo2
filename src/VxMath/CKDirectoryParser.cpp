#include "CKDirectoryParser.h"

#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#else
#include <dirent.h>
#include <fnmatch.h>
#include <sys/stat.h>
#endif

#if !defined(_WIN32)
static bool IsDirectory(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

static bool MatchMask(const char *name, const char *mask) {
    if (!mask || !*mask) {
        return true;
    }
    if (strcmp(mask, "*.*") == 0) {
        mask = "*";
    }
    int flags = 0;
#ifdef FNM_CASEFOLD
    flags |= FNM_CASEFOLD;
#endif
    return fnmatch(mask, name ? name : "", flags) == 0;
}
#endif

CKDirectoryParser::CKDirectoryParser(char *dir, char *fileMask, XBOOL recurse) {
    m_FindData = NULL;
    m_StartDir = NULL;
    m_FullFileName = NULL;
    m_FileMask = NULL;
    m_SubParser = NULL;
#if defined(_WIN32)
    m_hFile = -1;
#else
    m_hFile = 0;
#endif
    m_State = 0;
    Reset(dir, fileMask, recurse);
}

CKDirectoryParser::~CKDirectoryParser() {
    Clean();

    delete[] m_StartDir;
    delete[] m_FullFileName;
    delete[] m_FileMask;
}

char *CKDirectoryParser::GetNextFile() {
#if defined(_WIN32)
    char buf[MAX_PATH];

    // Handle non-recursive mode or first phase of recursive mode (files in current directory)
    if ((m_State & 2) == 0) {
        _snprintf_s(buf, MAX_PATH, _TRUNCATE, "%s\\%s", m_StartDir, m_FileMask);

        // Use a loop to skip directories instead of recursive calls
        while (true) {
            if (m_hFile == -1) {
                m_hFile = _findfirst(buf, (_finddata_t*)m_FindData);
                if (m_hFile == -1) {
                    // No files found, move to subdirectory search if recursive
                    if ((m_State & 1) != 0) {
                        m_State |= 2;
                    } else {
                        return NULL;
                    }
                    break; // Exit file search loop
                }
            } else {
                if (_findnext(m_hFile, (_finddata_t*)m_FindData) != 0) {
                    // No more files, close handle and move to subdirectory search if recursive
                    _findclose(m_hFile);
                    m_hFile = -1;
                    if ((m_State & 1) != 0) {
                        m_State |= 2;
                    } else {
                        return NULL;
                    }
                    break; // Exit file search loop
                }
            }

            // Check if current item is a file (not directory)
            if ((((_finddata_t*)m_FindData)->attrib & _A_SUBDIR) == 0) {
                _snprintf_s(m_FullFileName, MAX_PATH, _TRUNCATE, "%s\\%s", m_StartDir,
                           ((_finddata_t*)m_FindData)->name);
                return m_FullFileName;
            }
            // If it's a directory, continue the loop to find next item
        }
    }

    // Handle recursive subdirectory search
    if ((m_State & 2) != 0) {
        // Check if we have an active subparser
        if (m_SubParser) {
            char* ret = m_SubParser->GetNextFile();
            if (ret) {
                return ret;
            } else {
                // Subparser is exhausted, clean it up
                delete m_SubParser;
                m_SubParser = NULL;
            }
        }

        // Look for next subdirectory
        while (true) {
            if (m_hFile == -1) {
                _snprintf_s(buf, MAX_PATH, _TRUNCATE, "%s\\*.*", m_StartDir);
                m_hFile = _findfirst(buf, (_finddata_t*)m_FindData);
                if (m_hFile == -1) {
                    return NULL; // No subdirectories found
                }
            } else {
                if (_findnext(m_hFile, (_finddata_t*)m_FindData) != 0) {
                    // No more subdirectories
                    _findclose(m_hFile);
                    m_hFile = -1;
                    return NULL;
                }
            }

            // Check if current item is a valid subdirectory (not . or ..)
            if (strcmp(((_finddata_t*)m_FindData)->name, ".") != 0 &&
                strcmp(((_finddata_t*)m_FindData)->name, "..") != 0 &&
                (((_finddata_t*)m_FindData)->attrib & _A_SUBDIR) != 0) {

                char dir[MAX_PATH];
                _snprintf_s(dir, MAX_PATH, _TRUNCATE, "%s\\%s", m_StartDir,
                           ((_finddata_t*)m_FindData)->name);
                m_SubParser = new CKDirectoryParser(dir, m_FileMask, TRUE);

                char* ret = m_SubParser->GetNextFile();
                if (ret) {
                    return ret;
                } else {
                    // This subdirectory has no files, clean up and continue looking
                    delete m_SubParser;
                    m_SubParser = NULL;
                }
            }
            // Continue looking for next subdirectory
        }
    }

    return NULL;
#else
    char buf[MAX_PATH];

    if ((m_State & 2) == 0) {
        while (true) {
            DIR *dir = reinterpret_cast<DIR *>(m_hFile);
            if (!dir) {
                dir = opendir(m_StartDir ? m_StartDir : ".");
                if (!dir) {
                    if ((m_State & 1) != 0) {
                        m_State |= 2;
                    } else {
                        return NULL;
                    }
                    break;
                }
                m_hFile = reinterpret_cast<intptr_t>(dir);
            }

            struct dirent *entry = readdir(dir);
            if (!entry) {
                closedir(dir);
                m_hFile = 0;
                if ((m_State & 1) != 0) {
                    m_State |= 2;
                } else {
                    return NULL;
                }
                break;
            }

            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            snprintf(buf, MAX_PATH, "%s/%s", m_StartDir, entry->d_name);
            if (IsDirectory(buf)) {
                continue;
            }
            if (!MatchMask(entry->d_name, m_FileMask)) {
                continue;
            }

            snprintf(m_FullFileName, MAX_PATH, "%s/%s", m_StartDir, entry->d_name);
            return m_FullFileName;
        }
    }

    if ((m_State & 2) != 0) {
        if (m_SubParser) {
            char *ret = m_SubParser->GetNextFile();
            if (ret) {
                return ret;
            }
            delete m_SubParser;
            m_SubParser = NULL;
        }

        while (true) {
            DIR *dir = reinterpret_cast<DIR *>(m_hFile);
            if (!dir) {
                dir = opendir(m_StartDir ? m_StartDir : ".");
                if (!dir) {
                    return NULL;
                }
                m_hFile = reinterpret_cast<intptr_t>(dir);
            }

            struct dirent *entry = readdir(dir);
            if (!entry) {
                closedir(dir);
                m_hFile = 0;
                return NULL;
            }

            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            snprintf(buf, MAX_PATH, "%s/%s", m_StartDir, entry->d_name);
            if (IsDirectory(buf)) {
                m_SubParser = new CKDirectoryParser(buf, m_FileMask, TRUE);
                char *ret = m_SubParser->GetNextFile();
                if (ret) {
                    return ret;
                }
                delete m_SubParser;
                m_SubParser = NULL;
            }
        }
    }

    return NULL;
#endif
}

void CKDirectoryParser::Reset(char *dir, char *fileMask, XBOOL recurse) {
#if defined(_WIN32)
    // Clean up existing resources
    Clean();

    // Preserve old values if new ones are NULL
    char* oldStartDir = m_StartDir;
    char* oldFileMask = m_FileMask;

    // Always clean up the full filename buffer
    delete[] m_FullFileName;

    // Allocate new resources
    m_FindData = new _finddata_t;
    m_FullFileName = new char[MAX_PATH];

    // Set directory
    if (dir) {
        delete[] oldStartDir; // Clean up old directory
        size_t dirLen = strlen(dir);
        m_StartDir = new char[dirLen + 1];
        strcpy_s(m_StartDir, dirLen + 1, dir);

        // Remove trailing slash/backslash
        if (dirLen > 0 && (m_StartDir[dirLen - 1] == '\\' || m_StartDir[dirLen - 1] == '/')) {
            m_StartDir[dirLen - 1] = '\0';
        }
    } else {
        // Keep the old directory if NULL is passed
        m_StartDir = oldStartDir;
    }

    // Set file mask
    if (fileMask) {
        delete[] oldFileMask; // Clean up old mask
        size_t maskLen = strlen(fileMask);
        m_FileMask = new char[maskLen + 1];
        strcpy_s(m_FileMask, maskLen + 1, fileMask);
    } else {
        // Keep the old file mask if NULL is passed
        m_FileMask = oldFileMask;
    }

    // Initialize state
    m_State = recurse ? 1 : 0;
    m_hFile = -1;
    m_SubParser = NULL;
#else
    Clean();

    char *oldStartDir = m_StartDir;
    char *oldFileMask = m_FileMask;

    delete[] m_FullFileName;

    m_FullFileName = new char[MAX_PATH];
    m_FindData = NULL;

    if (dir) {
        delete[] oldStartDir;
        size_t dirLen = strlen(dir);
        m_StartDir = new char[dirLen + 1];
        strcpy_s(m_StartDir, dirLen + 1, dir);

        if (dirLen > 0 && (m_StartDir[dirLen - 1] == '\\' || m_StartDir[dirLen - 1] == '/')) {
            m_StartDir[dirLen - 1] = '\0';
        }
    } else {
        m_StartDir = oldStartDir;
    }

    if (fileMask) {
        delete[] oldFileMask;
        size_t maskLen = strlen(fileMask);
        m_FileMask = new char[maskLen + 1];
        strcpy_s(m_FileMask, maskLen + 1, fileMask);
    } else {
        m_FileMask = oldFileMask;
    }

    m_State = recurse ? 1 : 0;
    m_hFile = 0;
    m_SubParser = NULL;
#endif
}

void CKDirectoryParser::Clean() {
#if defined(_WIN32)
    if (m_hFile != -1) {
        _findclose(m_hFile);
        m_hFile = -1;
    }

    if (m_SubParser) {
        delete m_SubParser;
        m_SubParser = NULL;
    }

    if (m_FindData) {
        delete (_finddata_t*)m_FindData;
        m_FindData = NULL;
    }
#else
    if (m_hFile) {
        closedir(reinterpret_cast<DIR *>(m_hFile));
        m_hFile = 0;
    }

    if (m_SubParser) {
        delete m_SubParser;
        m_SubParser = NULL;
    }

    m_FindData = NULL;
#endif
}
