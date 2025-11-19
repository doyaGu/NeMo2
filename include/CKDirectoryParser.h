#ifndef CKDIRECTORYPARSER_H
#define CKDIRECTORYPARSER_H

#include "VxMathDefines.h"

/**
 * @brief Utility class for parsing directories.
 *
 * @remarks
 * The CKDirectoryParser class is useful for iterating within a directory hierarchy.
 * It can be used to parse a single directory for files matching an extension filter or
 * to parse a complete directory tree.
 * On Win32 platforms, it is subject to extension mismatch limitations; for example,
 * if you search for *.ext, files like *.exta will also be listed.
 *
 * @example
 * @code
 * // Counts and creates a list of files with the .cpp extension
 * // in a directory tree.
 * CKDirectoryParser MyParser(Directory, "*.cpp", TRUE);
 *
 * XClassArray<XString> FileList;
 * char* str = NULL;
 * while ((str = MyParser.GetNextFile())) {
 *     // str contains the full path of the file.
 *     FileList.PushBack(XString(str));
 * }
 * @endcode
 *
 * @see CKPathMaker, CKPathSplitter
 */
class CKDirectoryParser {
public:
    /**
     * @brief Constructs a CKDirectoryParser object.
     * @param dir The directory to start parsing from.
     * @param fileMask The file mask to filter files (e.g., "*.txt").
     * @param recurse If TRUE, parsing will be recursive into subdirectories. Defaults to FALSE.
     */
    VX_EXPORT CKDirectoryParser(char *dir, char *fileMask, XBOOL recurse = FALSE);

    /**
     * @brief Destroys the CKDirectoryParser object.
     */
    VX_EXPORT ~CKDirectoryParser();

    /**
     * @brief Retrieves the next file matching the specified criteria.
     * @return A character pointer to the full path of the next file, or NULL if no more files are found.
     */
    VX_EXPORT char *GetNextFile();

    /**
     * @brief Resets the parser to a new or initial state.
     * @param dir The new directory to parse. If NULL, the original directory is used.
     * @param fileMask The new file mask. If NULL, the original mask is used.
     * @param recurse The new recursive setting.
     */
    VX_EXPORT void Reset(char *dir = NULL, char *fileMask = NULL, XBOOL recurse = FALSE);

protected:
    /// @brief Cleans up internal resources.
    void Clean();

    void *m_FindData;               ///< Stores find data from the operating system.
    int m_hFile;                    ///< Handle for the directory search.
    char *m_FullFileName;           ///< Buffer for the full path of the currently found file.
    char *m_StartDir;               ///< The starting directory for the parse operation.
    char *m_FileMask;               ///< The file mask for filtering files.
    XULONG m_State;                 ///< Internal state of the parser.
    CKDirectoryParser *m_SubParser; ///< A parser for handling subdirectories in recursive mode.
};

#endif // CKDIRECTORYPARSER_H
