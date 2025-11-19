#ifndef XSTRING_H
#define XSTRING_H

#include <ctype.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "VxMathDefines.h"
#include "XUtil.h"
#include "XClassArray.h"

#if VX_HAS_CXX11
#include <algorithm>  // For std::move
#include <iterator>   // For std::reverse_iterator
#include <stdexcept>  // For std::out_of_range
#endif

class XString;

/**
 * @class XBaseString
 * @brief A lightweight wrapper around a C-style string (`char*` or `const char*`).
 *
 * @remarks
 * This class provides basic string functionalities without managing the memory of the
 * string it points to. It does not duplicate the string data upon creation and
 * does not deallocate it upon destruction. It calculates and caches the string's
 * length for fast access.
 *
 * @see XString
 */
class XBaseString {
    friend class XString;

public:
    /**
     * @brief Default constructor. Creates an empty string wrapper.
     */
    XBaseString() : m_Buffer(0), m_Length(0), m_Allocated(0) {}

    /**
     * @brief Constructs a base string from a C-style string literal.
     * @param iString The null-terminated string to wrap. Its length is calculated on construction.
     * @warning The provided string is not copied; only its pointer is stored.
     */
    XBaseString(const char *iString) : m_Allocated(0) {
        if (iString) {
            m_Buffer = (char *) iString;
            m_Length = 0xffff;
            while (m_Buffer[++m_Length]);
        } else {
            m_Buffer = NULL;
            m_Length = 0;
        }
    }

    /**
     * @brief Destructor. Does nothing, as it does not own the string buffer.
     */
    ~XBaseString() {}

    /**
     * @brief Returns the length of the string.
     * @return The length of the string in characters, excluding the null terminator.
     */
    XWORD Length() const {
        return m_Length;
    }

    /**
     * @brief Checks if the string is empty.
     * @return TRUE if the length is 0, FALSE otherwise.
     */
    XBOOL Empty() const {
        return m_Length == 0;
    }

    /**
     * @brief Gets the string as a constant C-style string.
     * @return A `const char*` pointer to the string buffer. Returns "" for a null string.
     */
    const char *CStr() const {
        return (m_Buffer) ? m_Buffer : "";
    }

    /**
     * @brief Converts the string to an integer.
     * @return The integer representation of the string, or 0 if conversion is not possible.
     */
    int ToInt() const {
        return (m_Buffer) ? atoi(m_Buffer) : 0;
    }

    /**
     * @brief Converts the string to a float.
     * @return The float representation of the string, or 0.0f if conversion is not possible.
     */
    float ToFloat() const {
        return (m_Buffer) ? (float) atof(m_Buffer) : 0.0f;
    }

    /**
     * @brief Converts the string to a double.
     * @return The double representation of the string, or 0.0 if conversion is not possible.
     */
    double ToDouble() const {
        return (m_Buffer) ? atof(m_Buffer) : 0.0;
    }

    /**
     * @brief Accesses a character by index (read-only).
     * @param i The zero-based index of the character to access.
     * @return The character at the specified index.
     */
    const char operator[](XWORD i) const {
        XASSERT(i < m_Length);
        return m_Buffer[i];
    }

    /**
     * @brief Accesses a character by index (read-only).
     * @param i The zero-based index of the character to access.
     * @return The character at the specified index.
     */
    const char operator[](int i) const {
        XASSERT(i < m_Length);
        return m_Buffer[i];
    }

protected:
    /// The pointer to the character buffer.
    char *m_Buffer;
    /// The length of the string.
    XWORD m_Length;
    /// The allocated size of the buffer (used by XString).
    XWORD m_Allocated;
};

/**
 * @class XString
 * @brief A class for representing and manipulating strings.
 *
 * @remarks
 * This class manages its own memory and always duplicates the string data it is given.
 * It provides a wide range of functionalities including comparison, searching,
 * modification, and formatting.
 *
 * @see XBaseString
 */
class XString : public XBaseString {
public:
    /// @brief A constant representing a position not found in search operations.
    enum { NOTFOUND = 0xffff };

    /// @typedef Iterator A mutable iterator.
    typedef char *Iterator;
    /// @typedef ConstIterator A constant iterator.
    typedef const char *ConstIterator;
#if VX_HAS_CXX11
    /// @typedef ReverseIterator A mutable reverse iterator.
    typedef std::reverse_iterator<Iterator> ReverseIterator;
    /// @typedef ConstReverseIterator A constant reverse iterator.
    typedef std::reverse_iterator<ConstIterator> ConstReverseIterator;
#endif

    /**
     * @brief Default constructor. Creates an empty string.
     */
    XString() : XBaseString() {}

    /**
     * @brief Substring constructor. Creates a string from a portion of a C-style string.
     * @param iString The source C-style string.
     * @param iLength The number of characters to copy. If 0, copies the entire string.
     */
    VX_EXPORT XString(const char *iString, int iLength = 0);

    /**
     * @brief Reserving constructor. Creates an empty string with a pre-allocated capacity.
     * @param iLength The number of characters to reserve space for.
     */
    VX_EXPORT explicit XString(int iLength);

    /**
     * @brief Copy constructor.
     * @param iSrc The XString object to copy.
     */
    VX_EXPORT XString(const XString &iSrc);

    /**
     * @brief Copy constructor from an XBaseString.
     * @param iSrc The XBaseString object to copy.
     */
    VX_EXPORT XString(const XBaseString &iSrc);

#if VX_HAS_CXX11
    /**
     * @brief Move constructor (C++11).
     * @param iSrc The XString object to move from.
     */
    XString(XString &&iSrc) VX_NOEXCEPT : XBaseString() {
        m_Buffer = iSrc.m_Buffer;
        m_Length = iSrc.m_Length;
        m_Allocated = iSrc.m_Allocated;
        iSrc.m_Buffer = NULL;
        iSrc.m_Length = 0;
        iSrc.m_Allocated = 0;
    }
#endif

    /**
     * @brief Destructor. Deallocates the string's buffer.
     */
    VX_EXPORT ~XString();

    /**
     * @brief Assignment operator from an XString.
     * @param iSrc The XString to assign from.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator=(const XString &iSrc);

    /**
     * @brief Assignment operator from a C-style string.
     * @param iSrc The C-style string to assign from.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator=(const char *iSrc);

    /**
     * @brief Assignment operator for a single character.
     * @param c The character to assign.
     * @return A reference to this string.
     */
    // XString &operator=(char c) {
    //     m_Length = 1;
    //     CheckSize(1);
    //     m_Buffer[0] = c;
    //     m_Buffer[1] = '\0';
    //     return *this;
    // }

    /**
     * @brief Assignment operator from an XBaseString.
     * @param iSrc The XBaseString to assign from.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator=(const XBaseString &iSrc);

#if VX_HAS_CXX11
    /**
     * @brief Move assignment operator (C++11).
     * @param iSrc The XString to move assign from.
     * @return A reference to this string.
     */
    XString &operator=(XString &&iSrc) VX_NOEXCEPT {
        if (this != &iSrc) {
            delete[] m_Buffer;
            m_Buffer = iSrc.m_Buffer;
            m_Length = iSrc.m_Length;
            m_Allocated = iSrc.m_Allocated;
            iSrc.m_Buffer = NULL;
            iSrc.m_Length = 0;
            iSrc.m_Allocated = 0;
        }
        return *this;
    }
#endif

    /**
     * @brief Creates the string from a C-style string and an optional length.
     * @param iString The source C-style string.
     * @param iLength The number of characters to copy. If 0, copies the entire string.
     * @return A reference to this string.
     */
    VX_EXPORT XString &Create(const char *iString, int iLength = 0);

    /**
     * @brief Gets the string as a mutable C-style string.
     * @return A `char*` pointer to the string buffer. Returns "" for a null string.
     */
    char *Str() { return (m_Buffer) ? m_Buffer : (char *) ""; }

    /**
     * @brief Accesses a character by index (read/write).
     * @param i The zero-based index of the character to access.
     * @return A reference to the character at the specified index.
     */
    char &operator[](XWORD i) {
        XASSERT(i < m_Length);
        return m_Buffer[i];
    }

    /**
     * @brief Accesses a character by index (read/write).
     * @param i The zero-based index of the character to access.
     * @return A reference to the character at the specified index.
     */
    char &operator[](int i) {
        XASSERT(i < m_Length);
        return m_Buffer[i];
    }

    /**
     * @brief Accesses a character by index (read-only).
     * @param i The zero-based index of the character to access.
     * @return The character at the specified index.
     */
    char operator[](XWORD i) const {
        XASSERT(i < m_Length);
        return m_Buffer[i];
    }

    /**
     * @brief Accesses a character by index (read-only).
     * @param i The zero-based index of the character to access.
     * @return The character at the specified index.
     */
    char operator[](int i) const {
        XASSERT(i < m_Length);
        return m_Buffer[i];
    }

    /**
     * @brief Accesses a character by index with bounds checking.
     * @param i The zero-based index of the character to access.
     * @return A reference to the character if the index is valid.
     * @throw std::out_of_range If the index `i` is out of bounds (C++11 only).
     */
#if VX_HAS_CXX11
    char &At(int i) {
        if (i < 0 || i >= m_Length)
            throw std::out_of_range("Index out of range in XString::At");
        return m_Buffer[i];
    }

    /**
     * @brief Accesses a character by index with bounds checking (const version).
     * @param i The zero-based index of the character to access.
     * @return A constant reference to the character if the index is valid.
     * @throw std::out_of_range If the index `i` is out of bounds (C++11 only).
     */
    const char &At(int i) const {
        if (i < 0 || i >= m_Length)
            throw std::out_of_range("Index out of range in XString::At");
        return m_Buffer[i];
    }
#endif

    /**
     * @brief Formats the string using a `sprintf`-style format string and arguments.
     * @param iFormat The format string.
     * @param ... Variable arguments for the format string.
     * @return A reference to this string.
     */
    VX_EXPORT XString &Format(const char *iFormat, ...);

    /**
     * @brief Converts all characters of the string to uppercase.
     * @return A reference to this string.
     */
    VX_EXPORT XString &ToUpper();

    /**
     * @brief Converts all characters of the string to lowercase.
     * @return A reference to this string.
     */
    VX_EXPORT XString &ToLower();

    /**
     * @brief Compares this string with another string (case-sensitive).
     * @param iStr The string to compare with.
     * @return
     * - `< 0`: if this string is less than `iStr`.
     * - `0`: if this string is identical to `iStr`.
     * - `> 0`: if this string is greater than `iStr`.
     * @see ICompare, NCompare
     */
    int Compare(const XBaseString &iStr) const {
        if (m_Length == 0)
            return -iStr.m_Length; // Null strings
        if (iStr.m_Length == 0)
            return m_Length;

        char *s1 = m_Buffer;
        char *s2 = iStr.m_Buffer;

        int Lenght4 = (m_Length > iStr.m_Length) ? (iStr.m_Length >> 2) : (m_Length >> 2);
        //--- Compare dwords by dwords....
        while ((Lenght4-- > 0) && (*(XDWORD *) s1 == *(XDWORD *) s2))
            s1 += 4, s2 += 4;

        //----- remaining bytes...
        while ((*s1 == *s2) && *s1)
            ++s1, ++s2;
        return (*s1 - *s2);
    }

    /**
     * @brief Compares the first `iN` characters of this string with another string (case-sensitive).
     * @param iStr The string to compare with.
     * @param iN The number of characters to compare.
     * @return
     * - `< 0`: if this string's substring is less than `iStr`'s.
     * - `0`: if the substrings are identical.
     * - `> 0`: if this string's substring is greater than `iStr`'s.
     * @see ICompare, NICompare
     */
    int NCompare(const XBaseString &iStr, int iN) const {
        if (m_Length == 0)
            return -iStr.m_Length; // Null strings
        if (iStr.m_Length == 0)
            return m_Length;

        return strncmp(m_Buffer, iStr.m_Buffer, iN);
    }

    /**
     * @brief Compares this string with another string (case-insensitive).
     * @param iStr The string to compare with.
     * @return
     * - `< 0`: if this string is less than `iStr`.
     * - `0`: if this string is identical to `iStr`.
     * - `> 0`: if this string is greater than `iStr`.
     * @see Compare, NICompare
     */
    VX_EXPORT int ICompare(const XBaseString &iStr) const;

    /**
     * @brief Compares the first `iN` characters of this string with another string (case-insensitive).
     * @param iStr The string to compare with.
     * @param iN The number of characters to compare.
     * @return
     * - `< 0`: if this string's substring is less than `iStr`'s.
     * - `0`: if the substrings are identical.
     * - `> 0`: if this string's substring is greater than `iStr`'s.
     * @see ICompare, NCompare
     */
    int NICompare(const XBaseString &iStr, int iN) const {
        if (m_Length == 0)
            return -iStr.m_Length; // Null strings
        if (iStr.m_Length == 0)
            return m_Length;

        char *s1 = m_Buffer;
        char *s2 = iStr.m_Buffer;

        int index = 0;
        char c1, c2;

        while (index < iN) {
            // Check if we've reached the end of either string
            if (*s1 == '\0' || *s2 == '\0') {
                if (*s1 == '\0' && *s2 == '\0') return 0;  // Both ended at same position
                return (*s1 == '\0') ? -1 : 1;  // One ended before the other
            }

            c1 = tolower(*s1);
            c2 = tolower(*s2);

            if (c1 != c2) {
                return c1 - c2;
            }

            s1++;
            s2++;
            index++;
        }

        return 0;  // All compared characters were equal
    }

    /// @name Complex string operations

    /**
     * @brief Removes whitespace (' ', '\t', '\n') from the beginning and end of the string.
     * @return A reference to this string.
     */
    VX_EXPORT XString &Trim();

    /**
     * @brief Replaces sequences of whitespace (' ', '\t', '\n') with a single space.
     * @return A reference to this string.
     */
    VX_EXPORT XString &Strip();

    /**
     * @brief Checks if the string contains a given substring.
     * @param iStr The substring to search for.
     * @return TRUE if `iStr` is found, FALSE otherwise.
     */
    XBOOL Contains(const XBaseString &iStr) const {
        return Find(iStr) != NOTFOUND;
    }

    /**
     * @brief Checks if the string starts with a given prefix (case-sensitive).
     * @param iStr The prefix to check for.
     * @return TRUE if the string starts with `iStr`, FALSE otherwise.
     */
    XBOOL StartsWith(const XBaseString &iStr) const {
        if (m_Length >= iStr.m_Length) {
            return NCompare(iStr, iStr.m_Length) == 0;
        }
        return FALSE;
    }

    /**
     * @brief Checks if the string starts with a given prefix (case-insensitive).
     * @param iStr The prefix to check for.
     * @return TRUE if the string starts with `iStr`, FALSE otherwise.
     */
    XBOOL IStartsWith(const XBaseString &iStr) const {
        if (m_Length >= iStr.m_Length) {
            return NICompare(iStr, iStr.m_Length) == 0;
        }
        return FALSE;
    }

    /**
     * @brief Checks if the string ends with a given suffix (case-sensitive).
     * @param iStr The suffix to check for.
     * @return TRUE if the string ends with `iStr`, FALSE otherwise.
     */
    XBOOL EndsWith(const XBaseString &iStr) const {
        if (m_Length >= iStr.m_Length) {
            XString buffer(m_Buffer + (m_Length - iStr.m_Length));
            return buffer.NCompare(iStr, iStr.m_Length) == 0;
        }
        return FALSE;
    }

    /**
     * @brief Checks if the string ends with a given suffix (case-insensitive).
     * @param iStr The suffix to check for.
     * @return TRUE if the string ends with `iStr`, FALSE otherwise.
     */
    XBOOL IEndsWith(const XBaseString &iStr) const {
        if (m_Length >= iStr.m_Length) {
            XString buffer(m_Buffer + (m_Length - iStr.m_Length));
            return buffer.NICompare(iStr, iStr.m_Length) == 0;
        }
        return FALSE;
    }

    /**
     * @brief Finds the first occurrence of a character in the string.
     * @param iCar The character to find.
     * @param iStart The position to start the search from.
     * @return The zero-based index of the first occurrence, or `NOTFOUND`.
     */
    VX_EXPORT XWORD Find(char iCar, XWORD iStart = 0) const;

    /**
     * @brief Finds the first occurrence of a substring (case-sensitive).
     * @param iStr The substring to find.
     * @param iStart The position to start the search from.
     * @return The zero-based index of the first occurrence, or `NOTFOUND`.
     */
    VX_EXPORT XWORD Find(const XBaseString &iStr, XWORD iStart = 0) const;

    /**
     * @brief Finds the first occurrence of a substring (case-insensitive).
     * @param iStr The substring to find.
     * @param iStart The position to start the search from.
     * @return The zero-based index of the first occurrence, or `NOTFOUND`.
     */
    // VX_EXPORT XWORD IFind(const XBaseString &iStr, XWORD iStart = 0) const;

    /**
     * @brief Finds the last occurrence of a character in the string.
     * @param iCar The character to find.
     * @param iStart The position to start the search from (searches backwards). If NOTFOUND, starts from the end.
     * @return The zero-based index of the last occurrence, or `NOTFOUND`.
     */
    VX_EXPORT XWORD RFind(char iCar, XWORD iStart = NOTFOUND) const;

    /**
     * @brief Extracts a substring.
     * @param iStart The starting position of the substring.
     * @param iLength The length of the substring. If 0, extracts to the end of the string.
     * @return A new XString object containing the substring.
     */
    VX_EXPORT XString Substring(XWORD iStart, XWORD iLength = 0) const;

    /**
     * @brief Modifies the string to be a substring of itself.
     * @param iStart The starting position of the new string.
     * @param iLength The length of the new string.
     * @return A reference to this string.
     */
    VX_EXPORT XString &Crop(XWORD iStart, XWORD iLength);

    /**
     * @brief Removes a part of the string.
     * @param iStart The starting position of the section to remove.
     * @param iLength The number of characters to remove.
     * @return A reference to this string.
     */
    VX_EXPORT XString &Cut(XWORD iStart, XWORD iLength);

    /**
     * @brief Replaces all occurrences of a character with another character.
     * @param iSrc The character to be replaced.
     * @param iDest The character to replace with.
     * @return The number of replacements made.
     */
    VX_EXPORT int Replace(char iSrc, char iDest);

    /**
     * @brief Replaces all occurrences of a substring with another string.
     * @param iSrc The substring to be replaced.
     * @param iDest The string to replace with.
     * @return The number of replacements made.
     */
    VX_EXPORT int Replace(const XBaseString &iSrc, const XBaseString &iDest);

    /**
     * @brief Splits the string into a collection of substrings based on a delimiter.
     * @param delimiter The character to split the string by.
     * @param[out] parts A `XClassArray<XString>` that will be filled with the substrings.
     * @return The number of substrings found.
     */
    int Split(char delimiter, XClassArray<XString> &parts) const {
        if (m_Length == 0)
            return 0;

        parts.Clear();

        XWORD start = 0;
        XWORD pos;

        while ((pos = Find(delimiter, start)) != NOTFOUND) {
            // Handle the case where we need an empty string (consecutive delimiters)
            if (pos == start) {
                parts.PushBack(XString());
            } else {
                parts.PushBack(Substring(start, pos - start));
            }
            start = pos + 1;
        }

        // Add the last part if exists
        if (start < m_Length) {
            parts.PushBack(Substring(start));
        } else if (start == m_Length) {
            // If string ends with delimiter, add empty string
            parts.PushBack(XString());
        }

        return parts.Size();
    }

    /**
     * @brief Joins a collection of strings into a single string, separated by a delimiter.
     * @param parts The collection of `XString` objects to join.
     * @param delimiter The C-style string to insert between each part.
     * @return A new `XString` containing the joined parts.
     */
    static XString Join(const XClassArray<XString> &parts, const char *delimiter) {
        if (parts.Size() == 0)
            return XString();

        XString result;
        int delimLen = delimiter ? strlen(delimiter) : 0;

        // Calculate total size
        XWORD totalSize = 0;
        for (int i = 0; i < parts.Size(); i++) {
            totalSize += parts[i].Length();
            if (i < parts.Size() - 1 && delimLen > 0)
                totalSize += delimLen;
        }

        // Allocate space
        result.Reserve(totalSize);

        // Join strings
        for (int i = 0; i < parts.Size(); i++) {
            result << parts[i];
            if (i < parts.Size() - 1 && delimLen > 0)
                result << delimiter;
        }

        return result;
    }

    /// @name Comparison operators
    //@{

    /** @brief Equality comparison with a C-style string. */
    int operator==(const char *iStr) const {
        return !Compare(iStr);
    }

    /** @brief Inequality comparison with a C-style string. */
    int operator!=(const char *iStr) const {
        return Compare(iStr);
    }

    /** @brief Less-than comparison with a C-style string. */
    int operator<(const char *iStr) const {
        return Compare(iStr) < 0;
    }

    /** @brief Less-than-or-equal comparison with a C-style string. */
    int operator<=(const char *iStr) const {
        return Compare(iStr) <= 0;
    }

    /** @brief Greater-than comparison with a C-style string. */
    int operator>(const char *iStr) const {
        return Compare(iStr) > 0;
    }

    /** @brief Greater-than-or-equal comparison with a C-style string. */
    int operator>=(const char *iStr) const {
        return Compare(iStr) >= 0;
    }

    /** @brief Three-way comparison with a C-style string. */
    int operator-(const char *iStr) const {
        return Compare(iStr);
    }

    /** @brief Equality comparison with an XBaseString. */
    int operator==(const XBaseString &iStr) const {
        return !Compare(iStr);
    }

    /** @brief Inequality comparison with an XBaseString. */
    int operator!=(const XBaseString &iStr) const {
        return Compare(iStr);
    }

    /** @brief Less-than comparison with an XBaseString. */
    int operator<(const XBaseString &iStr) const {
        return Compare(iStr) < 0;
    }

    /** @brief Less-than-or-equal comparison with an XBaseString. */
    int operator<=(const XBaseString &iStr) const {
        return Compare(iStr) <= 0;
    }

    /** @brief Greater-than comparison with an XBaseString. */
    int operator>(const XBaseString &iStr) const {
        return Compare(iStr) > 0;
    }

    /** @brief Greater-than-or-equal comparison with an XBaseString. */
    int operator>=(const XBaseString &iStr) const {
        return Compare(iStr) >= 0;
    }

    /** @brief Three-way comparison with an XBaseString. */
    int operator-(const XBaseString &iStr) const {
        return Compare(iStr);
    }

    //@}

    /// @name Concatenation operators
    //@{

    /**
     * @brief Appends a C-style string.
     * @param iString The string to append.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator<<(const char *iString);

    /**
     * @brief Appends an XBaseString.
     * @param iString The string to append.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator<<(const XBaseString &iString);

    /**
     * @brief Appends a single character.
     * @param iValue The character to append.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator<<(char iValue);

    /**
     * @brief Appends an integer.
     * @param iValue The integer to append, converted to a string.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator<<(int iValue);

    /**
     * @brief Appends an unsigned integer.
     * @param iValue The unsigned integer to append, converted to a string.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator<<(unsigned int iValue);

    /**
     * @brief Appends a float.
     * @param iValue The float to append, converted to a string.
     * @return A reference to this string.
     */
    VX_EXPORT XString &operator<<(float iValue);

    /**
     * @brief Appends a pointer address.
     * @param iValue The pointer to append, its address converted to a string.
     * @return A reference to this string.
     */
    // VX_EXPORT XString &operator<<(const void *iValue);

    /**
     * @brief Creates a new string by concatenating this string with a C-style string.
     * @param iString The string to concatenate.
     * @return A new XString object.
     */
    XString operator+(const char *iString) const {
        XString tmp = *this;
        return tmp << iString;
    }

    /**
     * @brief Creates a new string by concatenating this string with an XBaseString.
     * @param iString The string to concatenate.
     * @return A new XString object.
     */
    XString operator+(const XBaseString &iString) const {
        XString tmp = *this;
        return tmp << iString;
    }

    /**
     * @brief Creates a new string by concatenating this string with a character.
     * @param iValue The character to concatenate.
     * @return A new XString object.
     */
    XString operator+(char iValue) const {
        XString tmp = *this;
        return tmp << iValue;
    }

    /**
     * @brief Creates a new string by concatenating this string with an integer.
     * @param iValue The integer to concatenate.
     * @return A new XString object.
     */
    XString operator+(int iValue) const {
        XString tmp = *this;
        return tmp << iValue;
    }

    /**
     * @brief Creates a new string by concatenating this string with an unsigned integer.
     * @param iValue The unsigned integer to concatenate.
     * @return A new XString object.
     */
    XString operator+(unsigned int iValue) const {
        XString tmp = *this;
        return tmp << iValue;
    }

    /**
     * @brief Creates a new string by concatenating this string with a float.
     * @param iValue The float to concatenate.
     * @return A new XString object.
     */
    XString operator+(float iValue) const {
        XString tmp = *this;
        return tmp << iValue;
    }

    //@}

    /// @name Capacity functions
    //@{

    /**
     * @brief Returns the current capacity of the string's internal buffer.
     * @return The number of characters that can be stored without reallocation.
     * @see Reserve
     */
    XWORD Capacity() {
        return m_Allocated;
    }

    /**
     * @brief Requests that the string capacity be at least enough to contain `iLength` characters.
     * @param iLength The new capacity of the string.
     */
    VX_EXPORT void Reserve(XWORD iLength);

    /**
     * @brief Resizes the string to a length of `iLength` characters.
     * @param iLength The new length for the string.
     */
    void Resize(XWORD iLength) {
        Reserve(iLength);
        if (iLength != 0)
            m_Buffer[iLength] = '\0';
        else if (m_Buffer)
            m_Buffer[0] = '\0';
        if (m_Length > iLength)
            m_Length = iLength;
    }

    /**
     * @brief Returns whether the string is empty.
     * @return `true` if the string's length is 0, `false` otherwise.
     */
    bool IsEmpty() const { return m_Length == 0; }
    //@}

    /// @name Backwards Compatibility functions
    //@{
    /** @brief Appends an XString. Alias for `operator<<`. */
    XString &operator+=(const XString &v) { return (*this) << v; }
    /** @brief Appends a C-style string. Alias for `operator<<`. */
    XString &operator+=(const char *v) { return (*this) << v; }
    /** @brief Appends a character. Alias for `operator<<`. */
    XString &operator+=(char v) { return (*this) << v; }
    /** @brief Appends an integer. Alias for `operator<<`. */
    XString &operator+=(int v) { return (*this) << v; }
    /** @brief Appends a float. Alias for `operator<<`. */
    XString &operator+=(float v) { return (*this) << v; }
    //@}

    /**
     * @name Iterator methods
     * @brief Methods to get iterators to the beginning and end of the string.
     */
    //@{
    /** @brief Returns a mutable iterator to the beginning. */
    Iterator Begin() { return m_Buffer; }
    /** @brief Returns a constant iterator to the beginning. */
    ConstIterator Begin() const { return m_Buffer; }
    /** @brief Returns a mutable iterator to the end. */
    Iterator End() { return m_Buffer + m_Length; }
    /** @brief Returns a constant iterator to the end. */
    ConstIterator End() const { return m_Buffer + m_Length; }

#if VX_HAS_CXX11
    /** @brief Returns a mutable iterator to the beginning (C++11 STL compatibility). */
    Iterator begin() { return m_Buffer; }
    /** @brief Returns a constant iterator to the beginning (C++11 STL compatibility). */
    ConstIterator begin() const { return m_Buffer; }
    /** @brief Returns a mutable iterator to the end (C++11 STL compatibility). */
    Iterator end() { return m_Buffer + m_Length; }
    /** @brief Returns a constant iterator to the end (C++11 STL compatibility). */
    ConstIterator end() const { return m_Buffer + m_Length; }
    /** @brief Returns a constant iterator to the beginning (C++11 STL compatibility). */
    ConstIterator cbegin() const { return m_Buffer; }
    /** @brief Returns a constant iterator to the end (C++11 STL compatibility). */
    ConstIterator cend() const { return m_Buffer + m_Length; }

    /** @brief Returns a mutable reverse iterator to the beginning. */
    ReverseIterator rbegin() { return ReverseIterator(end()); }
    /** @brief Returns a constant reverse iterator to the beginning. */
    ConstReverseIterator rbegin() const { return ConstReverseIterator(end()); }
    /** @brief Returns a mutable reverse iterator to the end. */
    ReverseIterator rend() { return ReverseIterator(begin()); }
    /** @brief Returns a constant reverse iterator to the end. */
    ConstReverseIterator rend() const { return ConstReverseIterator(begin()); }
    /** @brief Returns a constant reverse iterator to the beginning (C++11 STL compatibility). */
    ConstReverseIterator crbegin() const { return ConstReverseIterator(end()); }
    /** @brief Returns a constant reverse iterator to the end (C++11 STL compatibility). */
    ConstReverseIterator crend() const { return ConstReverseIterator(begin()); }
#endif
    //@}

protected:
    /**
     * @brief Ensures the internal buffer has enough space for a given length.
     * @param iLength The required length, excluding the null terminator.
     */
    VX_EXPORT void CheckSize(int iLength);

    /**
     * @brief Assigns a new value to the string from a buffer and length.
     * @param iBuffer The source buffer to copy from.
     * @param iLength The number of characters to copy.
     */
    VX_EXPORT void Assign(const char *iBuffer, int iLength);
};

#endif // XSTRING_H
