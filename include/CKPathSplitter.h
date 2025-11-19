#ifndef CKPATHSPLITTER_H
#define CKPATHSPLITTER_H

#include "VxMathDefines.h"

#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4996)
#endif

/**
 * @brief Utility class for extracting components from a file path.
 *
 * @remarks
 * CKPathSplitter and CKPathMaker are useful for manipulating filenames and paths.
 * The CKPathSplitter class is used to break a path into its four components:
 * Drive, Directory, Filename, and Extension.
 * The CKPathMaker class creates a path string from these four components.
 *
 * @example
 * @code
 * // Example of changing a file's name:
 * CKPathSplitter splitter(OldFilename);
 *
 * char* OldName = splitter.GetName();
 * char* newname = ChangeName(OldName); // A hypothetical function to change the name
 * CKPathMaker maker(splitter.GetDrive(), splitter.GetDir(), newname, splitter.GetExtension());
 * char* NewFilename = maker.GetFileName();
 * @endcode
 *
 * @see CKPathMaker
 */
class VX_EXPORT CKPathSplitter {
public:
    /**
     * @brief Constructs the object from a full path string.
     * @param file A pointer to the full path of the file.
     */
    explicit CKPathSplitter(char *file);

    /**
     * @brief Destructor.
     */
    ~CKPathSplitter();

    /**
     * @brief Returns the drive letter, followed by a colon (e.g., "C:").
     * @return A pointer to the string containing the drive.
     */
    char *GetDrive();

    /**
     * @brief Returns the directory path, including the trailing slash (e.g., "\\path\\to\\file\\").
     * @return A pointer to the string containing the directory path.
     */
    char *GetDir();

    /**
     * @brief Returns the file name without its extension.
     * @return A pointer to the string containing the file name.
     */
    char *GetName();

    /**
     * @brief Returns the file extension, including the leading period (e.g., ".txt").
     * @return A pointer to the string containing the file extension.
     */
    char *GetExtension();

protected:
    char m_Drive[_MAX_DRIVE]; ///< Buffer to store the drive component.
    char m_Dir[_MAX_DIR];     ///< Buffer to store the directory component.
    char m_Filename[_MAX_FNAME]; ///< Buffer to store the filename component.
    char m_Ext[_MAX_EXT];     ///< Buffer to store the extension component.
};

/**
 * @brief Utility class for creating a full path from its components.
 *
 * @remarks
 * CKPathSplitter and CKPathMaker are useful for manipulating filenames and paths.
 * The CKPathMaker class creates a path string from its four components:
 * Drive, Directory, Filename, and Extension.
 *
 * @example
 * @code
 * // Example of changing a file's name:
 * CKPathSplitter splitter(OldFilename);
 *
 * char* OldName = splitter.GetName();
 * char* newname = ChangeName(OldName); // A hypothetical function to change the name
 * CKPathMaker maker(splitter.GetDrive(), splitter.GetDir(), newname, splitter.GetExtension());
 * char* NewFilename = maker.GetFileName();
 * @endcode
 *
 * @see CKPathSplitter
 */
class VX_EXPORT CKPathMaker {
public:
    /**
     * @brief Constructs a path from its individual components.
     * @param Drive The drive letter (e.g., "C:"). Can be NULL.
     * @param Directory The directory path (e.g., "\\path\\"). Can be NULL.
     * @param Fname The file name without extension. Can be NULL.
     * @param Extension The file extension with a leading dot (e.g., ".txt"). Can be NULL.
     */
    CKPathMaker(char *Drive, char *Directory, char *Fname, char *Extension);

    /**
     * @brief Returns the constructed full path.
     * @return A pointer to the string containing the full path.
     */
    char *GetFileName();

protected:
    char m_FileName[_MAX_PATH]; ///< Buffer to store the constructed full path.
};

/**
 * @brief A simple storage class for filename extensions.
 * @see CKPathSplitter, CKPathMaker
 */
struct CKFileExtension {
    /**
     * @brief Default constructor. Initializes the extension to an empty string.
     */
    CKFileExtension() : m_Data() {}

    /**
     * @brief Constructs a CKFileExtension from a string.
     * @param s A character pointer to the extension string (e.g., ".txt" or "txt"). Stores up to 3 characters.
     */
    CKFileExtension(const char *s) {
        if (!s)
            m_Data[0] = 0;
        else {
            if (s[0] == '.')
                s = &s[1];
            int len = strlen(s);
            if (len > 3)
                len = 3;
            memcpy(m_Data, s, len);
            m_Data[len] = '\0';
        }
    }

    /**
     * @brief Compares two extensions for equality, case-insensitively.
     * @param s The CKFileExtension to compare against.
     * @return Non-zero if the extensions are equal, zero otherwise.
     */
    int operator==(const CKFileExtension &s) {
        return !strcmpi(m_Data, s.m_Data);
    }

    /**
     * @brief Implicit conversion to a non-constant character pointer.
     * @return A `char*` pointing to the internal extension data.
     */
    operator char *() { return m_Data; }

    /**
     * @brief Implicit conversion to a constant character pointer.
     * @return A `const char*` pointing to the internal extension data.
     */
    operator const char *() { return m_Data; }

    /// @brief Internal storage for the extension string (up to 3 characters plus null terminator).
    char m_Data[4];
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // CKPATHSPLITTER_H
