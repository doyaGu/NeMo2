#include "XString.h"

#include <ctype.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

// Substring Ctor
XString::XString(const char *iString, int iLength) : XBaseString() {
    if (!iString || *iString == '\0') {
        m_Buffer = NULL;
        m_Length = 0;
        m_Allocated = 0;
        return;
    }

    m_Length = (iLength > 0) ? iLength : strlen(iString);
    m_Allocated = m_Length + 1;
    m_Buffer = new char[m_Allocated];
    strncpy(m_Buffer, iString, m_Length);
    m_Buffer[m_Length] = '\0';
}

// Reserving Ctor
XString::XString(int iLength) : XBaseString() {
    if (iLength <= 0) {
        m_Buffer = NULL;
        m_Length = 0;
        m_Allocated = 0;
        return;
    }

    m_Length = 0;
    m_Allocated = iLength;
    m_Buffer = new char[iLength];
    memset(m_Buffer, 0, m_Allocated);
}

// Copy Ctor
XString::XString(const XString &iSrc) : XBaseString() {
    if (iSrc.m_Length == 0) {
        m_Buffer = NULL;
        m_Length = 0;
        m_Allocated = 0;
        return;
    }

    m_Length = iSrc.m_Length;
    m_Allocated = iSrc.m_Length + 1;
    m_Buffer = new char[m_Allocated];
    memcpy(m_Buffer, iSrc.m_Buffer, m_Length + 1);
}

// Copy Ctor from XBaseString
XString::XString(const XBaseString &iSrc) : XBaseString() {
    if (iSrc.m_Length == 0) {
        m_Buffer = NULL;
        m_Length = 0;
        m_Allocated = 0;
        return;
    }

    m_Length = iSrc.m_Length;
    m_Allocated = iSrc.m_Length + 1;
    m_Buffer = new char[m_Allocated];
    memcpy(m_Buffer, iSrc.m_Buffer, m_Length + 1);
}

// Dtor
XString::~XString() {
    delete[] m_Buffer;
}

// operator =
XString &XString::operator=(const XString &iSrc) {
    if (this != &iSrc)
        Assign(iSrc.m_Buffer, iSrc.m_Length);
    return *this;
}

// operator =
XString &XString::operator=(const char *iSrc) {
    if (iSrc) {
        Assign(iSrc, strlen(iSrc));
    } else {
        m_Length = 0;
        if (m_Buffer)
            m_Buffer[0] = '\0';
    }
    return *this;
}

// operator =
XString &XString::operator=(const XBaseString &iSrc) {
    if (this != &iSrc) {
        Assign(iSrc.m_Buffer, iSrc.m_Length);
    }
    return *this;
}

// Create from a string and a length
XString &XString::Create(const char *iString, int iLength) {
    if (iString && iLength > 0) {
        CheckSize(iLength);
        m_Length = iLength;
        strncpy(m_Buffer, iString, iLength);
        m_Buffer[m_Length] = '\0';
    } else {
        m_Length = 0;
        if (m_Buffer)
            m_Buffer[0] = '\0';
    }
    return *this;
}

// Format the string sprintf style
XString &XString::Format(const char *iFormat, ...) {
    va_list args;
    va_start(args, iFormat);

    va_list args_copy;
    va_copy(args_copy, args);
    int len = vsnprintf(NULL, 0, iFormat, args_copy);
    va_end(args_copy);

    if (len > 0) {
        CheckSize(len);
        vsnprintf(m_Buffer, len + 1, iFormat, args);
        m_Length = len;
    } else {
        m_Length = 0;
        if (m_Buffer)
            m_Buffer[0] = '\0';
    }

    va_end(args);
    return *this;
}

// Capitalize all the characters of the string
XString &XString::ToUpper() {
    for (XWORD i = 0; i < m_Length; ++i)
        m_Buffer[i] = toupper(m_Buffer[i]);
    return *this;
}

// Uncapitalize all the characters of the string
XString &XString::ToLower() {
    for (XWORD i = 0; i < m_Length; ++i)
        m_Buffer[i] = tolower(m_Buffer[i]);
    return *this;
}

// Compare the strings (ignore the case).
int XString::ICompare(const XBaseString &iStr) const {
    if (!m_Length)
        return -iStr.m_Length;
    if (!iStr.m_Length)
        return m_Length;

    const char *s1 = m_Buffer;
    const char *s2 = iStr.m_Buffer;
    char c1, c2;

    do {
        c1 = tolower(*s1);
        c2 = tolower(*s2);
        if (c1 != c2)
            return c1 - c2;
        s1++;
        s2++;
    } while (c1 != '\0' && c2 != '\0');

    return c1 - c2;
}

// Remove whitespace from start and end
XString &XString::Trim() {
    if (m_Length == 0)
        return *this;

    // Trim end
    while (m_Length > 0 && isspace(m_Buffer[m_Length - 1])) {
        --m_Length;
    }
    m_Buffer[m_Length] = '\0';

    // Trim start
    if (m_Length > 0) {
        XWORD start = 0;
        while (start < m_Length && isspace(m_Buffer[start])) {
            ++start;
        }
        if (start > 0) {
            m_Length -= start;
            memmove(m_Buffer, &m_Buffer[start], m_Length + 1);
        }
    }
    return *this;
}

XString &XString::Strip() {
    if (m_Length == 0)
        return *this;

    XWORD writePos = 0;
    XBOOL lastWasSpace = FALSE;

    for (XWORD readPos = 0; readPos < m_Length; ++readPos) {
        if (isspace(m_Buffer[readPos])) {
            if (!lastWasSpace) {
                m_Buffer[writePos++] = ' ';
                lastWasSpace = TRUE;
            }
        } else {
            m_Buffer[writePos++] = m_Buffer[readPos];
            lastWasSpace = FALSE;
        }
    }

    m_Length = writePos;
    m_Buffer[m_Length] = '\0';
    return *this;
}

// Find a character in the string
XWORD XString::Find(char iCar, XWORD iStart) const {
    if (m_Length == 0 || iStart >= m_Length)
        return NOTFOUND;

    char *str = strchr(&m_Buffer[iStart], iCar);
    if (str)
        return str - m_Buffer;
    else
        return NOTFOUND;
}

// Find a string in the string
XWORD XString::Find(const XBaseString &iStr, XWORD iStart) const {
    if (m_Length == 0 || iStr.m_Length == 0 || iStart >= m_Length)
        return NOTFOUND;

    char *str = strstr(&m_Buffer[iStart], iStr.m_Buffer);
    if (str)
        return str - m_Buffer;
    else
        return NOTFOUND;
}

// XWORD XString::IFind(const XBaseString &iStr, XWORD iStart) const {
//     if (m_Length == 0 || iStr.m_Length == 0 || iStart >= m_Length)
//         return NOTFOUND;
//
//     // Manual case-insensitive search
//     for (XWORD i = iStart; i <= m_Length - iStr.m_Length; ++i) {
//         XBOOL match = TRUE;
//         for (XWORD j = 0; j < iStr.m_Length; ++j) {
//             if (tolower(m_Buffer[i + j]) != tolower(iStr.m_Buffer[j])) {
//                 match = FALSE;
//                 break;
//             }
//         }
//         if (match)
//             return i;
//     }
//     return NOTFOUND;
// }

XWORD XString::RFind(char iCar, XWORD iStart) const {
    if (m_Length == 0)
        return NOTFOUND;

    XWORD start = (iStart == NOTFOUND) ? m_Length - 1 : iStart;
    if (start >= m_Length)
        start = m_Length - 1;

    for (XWORD i = start + 1; i > 0; --i) {
        if (m_Buffer[i - 1] == iCar)
            return i - 1;
    }
    return NOTFOUND;
}

// Create a substring
XString XString::Substring(XWORD iStart, XWORD iLength) const {
    if (iStart >= m_Length)
        return XString();

    XWORD actualLength = (iLength == 0) ? m_Length - iStart : iLength;
    if (iStart + actualLength > m_Length)
        actualLength = m_Length - iStart;

    return XString(&m_Buffer[iStart], actualLength);
}

// Crop the string
XString &XString::Crop(XWORD iStart, XWORD iLength) {
    if (iStart >= m_Length) {
        m_Length = 0;
        if (m_Buffer)
            m_Buffer[0] = '\0';
        return *this;
    }

    if (iStart + iLength > m_Length)
        iLength = m_Length - iStart;

    if (iStart > 0)
        memmove(m_Buffer, &m_Buffer[iStart], iLength);

    m_Length = iLength;
    m_Buffer[m_Length] = '\0';
    return *this;
}

// Cut part of the string
XString &XString::Cut(XWORD iStart, XWORD iLength) {
    if (iStart >= m_Length)
        return *this;

    if (iStart + iLength > m_Length)
        iLength = m_Length - iStart;

    memmove(&m_Buffer[iStart], &m_Buffer[iStart + iLength], m_Length - (iStart + iLength) + 1);
    m_Length -= iLength;
    return *this;
}

// Replace all occurrences of a character
int XString::Replace(char iSrc, char iDest) {
    int count = 0;
    for (XWORD i = 0; i < m_Length; ++i) {
        if (m_Buffer[i] == iSrc) {
            m_Buffer[i] = iDest;
            ++count;
        }
    }
    return count;
}

int XString::Replace(const XBaseString &iSrc, const XBaseString &iDest) {
    if (iSrc.m_Length == 0)
        return 0;

    int count = 0;
    XWORD pos = 0;

    // Count occurrences first
    while ((pos = Find(iSrc, pos)) != NOTFOUND) {
        count++;
        pos += iSrc.m_Length;
    }

    if (count == 0)
        return 0;

    // Calculate new length
    XWORD newLength = m_Length + count * (iDest.m_Length - iSrc.m_Length);

    if (iSrc.m_Length == iDest.m_Length) {
        // Simple case: same length replacement
        pos = 0;
        while ((pos = Find(iSrc, pos)) != NOTFOUND) {
            memcpy(&m_Buffer[pos], iDest.m_Buffer, iDest.m_Length);
            pos += iSrc.m_Length;
        }
    } else {
        // Complex case: different length replacement
        char *newBuffer = new char[newLength + 1];
        XWORD srcPos = 0, destPos = 0;

        pos = 0;
        while ((pos = Find(iSrc, srcPos)) != NOTFOUND) {
            // Copy text before match
            XWORD copyLen = pos - srcPos;
            if (copyLen > 0) {
                memcpy(&newBuffer[destPos], &m_Buffer[srcPos], copyLen);
                destPos += copyLen;
            }

            // Copy replacement text
            if (iDest.m_Length > 0) {
                memcpy(&newBuffer[destPos], iDest.m_Buffer, iDest.m_Length);
                destPos += iDest.m_Length;
            }

            srcPos = pos + iSrc.m_Length;
        }

        // Copy remaining text
        if (srcPos < m_Length) {
            memcpy(&newBuffer[destPos], &m_Buffer[srcPos], m_Length - srcPos);
            destPos += m_Length - srcPos;
        }

        newBuffer[newLength] = '\0';

        delete[] m_Buffer;
        m_Buffer = newBuffer;
        m_Length = newLength;
        m_Allocated = newLength + 1;
    }

    return count;
}

XString &XString::operator<<(const char *iString) {
    if (iString) {
        XWORD len = strlen(iString);
        if (len > 0) {
            CheckSize(m_Length + len);
            strcpy(&m_Buffer[m_Length], iString);
            m_Length += len;
        }
    }
    return *this;
}

// Concatenation operator
XString &XString::operator<<(const XBaseString &iString) {
    if (iString.m_Length > 0) {
        CheckSize(m_Length + iString.m_Length);
        memcpy(&m_Buffer[m_Length], iString.m_Buffer, iString.m_Length);
        m_Length += iString.m_Length;
        m_Buffer[m_Length] = '\0';
    }
    return *this;
}

// Concatenation operator
XString &XString::operator<<(char iValue) {
    CheckSize(m_Length + 1);
    m_Buffer[m_Length++] = iValue;
    m_Buffer[m_Length] = '\0';
    return *this;
}

// Concatenation operator
XString &XString::operator<<(int iValue) {
    char buf[32];  // Increased buffer size for safety
    int len = snprintf(buf, sizeof(buf), "%d", iValue);
    if (len > 0) {
        CheckSize(m_Length + len);
        strcpy(&m_Buffer[m_Length], buf);
        m_Length += len;
    }
    return *this;
}

// Concatenation operator
XString &XString::operator<<(unsigned int iValue) {
    char buf[32];  // Increased buffer size for safety
    int len = snprintf(buf, sizeof(buf), "%u", iValue);
    if (len > 0) {
        CheckSize(m_Length + len);
        strcpy(&m_Buffer[m_Length], buf);
        m_Length += len;
    }
    return *this;
}

// Concatenation operator
XString &XString::operator<<(float iValue) {
    char buf[32];  // Increased buffer size for safety
    int len = snprintf(buf, sizeof(buf), "%f", iValue);  // Use %g for cleaner float output
    if (len > 0) {
        CheckSize(m_Length + len);
        strcpy(&m_Buffer[m_Length], buf);
        m_Length += len;
    }
    return *this;
}

// XString &XString::operator<<(const void *iValue) {
//     if (iValue) {
//         char buf[32];
//         int len = snprintf(buf, sizeof(buf), "%p", iValue);
//         if (len > 0) {
//             CheckSize(m_Length + len);
//             strcpy(&m_Buffer[m_Length], buf);
//             m_Length += len;
//         }
//     }
//     return *this;
// }

void XString::Reserve(XDWORD iLength) {
    if (iLength + 1 > m_Allocated) {
        m_Allocated = iLength + 1;
        char *buf = new char[m_Allocated];
        if (m_Length > 0) {
            memcpy(buf, m_Buffer, m_Length + 1);
        } else {
            buf[0] = '\0';
        }
        delete[] m_Buffer;
        m_Buffer = buf;
    }
}

void XString::CheckSize(int iLength) {
    if (iLength + 1 > m_Allocated) {
        // Grow buffer with some extra space to reduce reallocations
        m_Allocated = (iLength + 1) * 2;
        char *buf = new char[m_Allocated];
        if (m_Length > 0) {
            memcpy(buf, m_Buffer, m_Length + 1);
        } else {
            buf[0] = '\0';
        }
        delete[] m_Buffer;
        m_Buffer = buf;
    }
}

// Assign a string to the current string
void XString::Assign(const char *iBuffer, int iLength) {
    if (iLength > 0 && iBuffer) {
        CheckSize(iLength);
        memcpy(m_Buffer, iBuffer, iLength);
        m_Length = iLength;
        m_Buffer[m_Length] = '\0';
    } else {
        m_Length = 0;
        if (m_Buffer) {
            m_Buffer[0] = '\0';
        }
    }
}