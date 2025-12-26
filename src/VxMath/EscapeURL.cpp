#include "VxWindowFunctions.h"

static bool IsHexDigit(char c) {
    return (c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f');
}

static int HexCharToValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return 0;
}

static bool HasProtocol(const char *url) {
    // Look for "://" but ensure it's at the beginning of a protocol
    // A protocol should start with a letter and contain only alphanumeric chars and +.-
    const char *colonSlashSlash = strstr(url, "://");
    if (!colonSlashSlash) return false;

    // Check that everything before "://" looks like a valid protocol
    for (const char *p = url; p < colonSlashSlash; ++p) {
        if (p == url) {
            // First character must be a letter
            if (!(((*p >= 'a') && (*p <= 'z')) || ((*p >= 'A') && (*p <= 'Z')))) {
                return false;
            }
        } else {
            // Subsequent characters can be alphanumeric or +.-
            if (!(((*p >= 'a') && (*p <= 'z')) ||
                ((*p >= 'A') && (*p <= 'Z')) ||
                ((*p >= '0') && (*p <= '9')) ||
                (*p == '+') || (*p == '.') || (*p == '-'))) {
                return false;
            }
        }
    }

    return true;
}

XULONG VxEscapeURL(char *InURL, XString &OutURL) {
    // Handle null input
    if (!InURL) {
        OutURL = "";
        return (XULONG) -1;
    }

    // Handle empty string
    if (*InURL == '\0') {
        OutURL = "file://";
        return 0;
    }

    bool hasProtocol = HasProtocol(InURL);
    bool isFileProtocol = false;

    // Check if it's already a file URL (case insensitive)
    if (hasProtocol && strnicmp(InURL, "file://", 7) == 0) {
        isFileProtocol = true;
    }

    // Count special characters that need escaping
    char *str = InURL;
    int specialCharCount = 0;

    // If it's a file protocol, start counting after "file://"
    if (isFileProtocol) {
        str += 7; // Skip "file://"
    }

    while (*str) {
        if (strchr(" #$%&\\+,/:;=?@[]^{}", *str)) {
            specialCharCount++;
        }
        str++;
    }

    // If no special characters and it's already a complete URL, return as-is
    if (specialCharCount == 0) {
        if (hasProtocol) {
            OutURL = InURL;
        } else {
            OutURL = "file://";
            OutURL += InURL;
        }
        return 0;
    }

    // Calculate required buffer size
    int originalLen = strlen(InURL);
    int bufferSize = originalLen + 1 + (2 * specialCharCount); // +1 for null terminator, +2 for each escaped char

    if (!hasProtocol) {
        bufferSize += 7; // Add space for "file://"
    }

    // Allocate buffer
    char *buf = new(std::nothrow) char[bufferSize];
    if (!buf) {
        OutURL = "";
        return (XULONG) -1; // Memory allocation failed
    }

    // Build the escaped URL
    char *pi = InURL;
    char *pb = buf;

    // Add file:// if no protocol exists
    if (!hasProtocol) {
        strcpy(pb, "file://");
        pb += 7;
    } else if (isFileProtocol) {
        // Copy the "file://" part as-is
        strncpy(pb, pi, 7);
        pb += 7;
        pi += 7;
    }

    // Process the rest of the URL
    while (*pi) {
        if (strchr(" #$%&\\+,/:;=?@[]^{}", *pi)) {
            // Escape special character
            sprintf(pb, "%%%02X", (unsigned char) *pi);
            pb += 3;
        } else {
            // Copy normal character
            *pb++ = *pi;
        }
        pi++;
    }
    *pb = '\0';

    OutURL = buf;
    delete[] buf;
    return 0;
}

void VxUnEscapeUrl(XString &str) {
    if (str.Length() == 0) return;

    XString result;
    result.Reserve(str.Length());

    for (int i = 0; i < str.Length(); ++i) {
        if (str[i] == '%' && i + 2 < str.Length()) {
            char hex1 = str[i + 1];
            char hex2 = str[i + 2];

            // Validate that both characters are valid hex digits
            if (IsHexDigit(hex1) && IsHexDigit(hex2)) {
                // Convert hex to character
                int value = (HexCharToValue(hex1) << 4) | HexCharToValue(hex2);
                result += (char) value;
                i += 2; // Skip the next two characters
            } else {
                // Invalid hex sequence, keep the % and continue
                result += str[i];
            }
        } else {
            // Regular character or incomplete % sequence at end
            result += str[i];
        }
    }

    str = result;
}

// Alternative version of VxUnEscapeUrl with more strict validation
void VxUnEscapeUrlStrict(XString &str) {
    if (str.Length() == 0) {
        return;
    }

    XString result;
    result.Reserve(str.Length());

    for (int i = 0; i < str.Length(); ++i) {
        if (str[i] == '%') {
            // Check if we have enough characters for a complete hex sequence
            if (i + 2 < str.Length()) {
                char hex1 = str[i + 1];
                char hex2 = str[i + 2];

                if (IsHexDigit(hex1) && IsHexDigit(hex2)) {
                    // Valid hex sequence
                    int value = (HexCharToValue(hex1) << 4) | HexCharToValue(hex2);
                    result += (char) value;
                    i += 2;
                } else {
                    // Invalid hex sequence - could either skip or treat as literal
                    // This version treats as literal %
                    result += str[i];
                }
            } else {
                // Incomplete sequence at end of string
                result += str[i];
            }
        } else {
            result += str[i];
        }
    }

    str = result;
}
