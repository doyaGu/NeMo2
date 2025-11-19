#ifndef VXWINDOWFUNCTION_H
#define VXWINDOWFUNCTION_H

#include "VxMathDefines.h"
#include "XString.h"

struct VxImageDescEx;

// KeyBoard Functions

/**
 * @brief Converts a scan code to its ASCII equivalent based on the current keyboard state.
 * @param scancode The hardware scan code of the key.
 * @param keystate An array of 256 bytes representing the state of all keys (pressed/unpressed).
 * @return The corresponding ASCII character, or 0 if no direct mapping exists.
 */
VX_EXPORT char VxScanCodeToAscii(XULONG scancode, unsigned char keystate[256]);

/**
 * @brief Retrieves the descriptive name of a key from its scan code.
 * @param scancode The hardware scan code of the key.
 * @param keyName A character buffer to receive the name of the key (e.g., "SHIFT", "A").
 * @return The length of the key name string.
 */
VX_EXPORT int VxScanCodeToName(XULONG scancode, char *keyName);

// Cursor function

/**
 * @enum VXCURSOR_POINTER
 * @brief Defines the standard appearances for the mouse cursor.
 * @see VxSetCursor
 */
typedef enum VXCURSOR_POINTER {
    VXCURSOR_NORMALSELECT = 1, ///< The standard arrow cursor.
    VXCURSOR_BUSY         = 2, ///< The busy (hourglass or equivalent) cursor.
    VXCURSOR_MOVE         = 3, ///< The move (four-headed arrow) cursor.
    VXCURSOR_LINKSELECT   = 4  ///< The link select (hand) cursor.
} VXCURSOR_POINTER;

/**
 * @brief Shows or hides the mouse cursor.
 * @param show If TRUE, the cursor is shown. If FALSE, it is hidden.
 * @return The new display counter for the cursor.
 */
VX_EXPORT int VxShowCursor(XBOOL show);

/**
 * @brief Sets the appearance of the mouse cursor.
 * @param cursorID A value from the `VXCURSOR_POINTER` enumeration.
 * @return TRUE on success, FALSE on failure.
 */
VX_EXPORT XBOOL VxSetCursor(VXCURSOR_POINTER cursorID);

/// @brief Gets the current Floating-Point Unit (FPU) control word.
VX_EXPORT XWORD VxGetFPUControlWord();
/// @brief Sets the Floating-Point Unit (FPU) control word.
VX_EXPORT void VxSetFPUControlWord(XWORD Fpu);
/// @brief Sets the FPU control word to a default state (disables exceptions, rounds to nearest, single precision).
VX_EXPORT void VxSetBaseFPUControlWord();

//-------Library Function

/// @brief Adds a directory to the search path for loading shared libraries.
VX_EXPORT void VxAddLibrarySearchPath(char *path);

/// @brief Retrieves the value of an environment variable.
VX_EXPORT XBOOL VxGetEnvironmentVariable(char *envName, XString &envValue);
/// @brief Sets the value of an environment variable.
VX_EXPORT XBOOL VxSetEnvironmentVariable(char *envName, char *envValue);

/// @brief Converts a URL string into its escaped format (e.g., replacing spaces with %20).
VX_EXPORT XULONG VxEscapeURL(char *InURL, XString &OutURL);
/// @brief Converts an escaped URL string back to its original format.
VX_EXPORT void VxUnEscapeUrl(XString &str);

//------ Window Functions
/// @brief Retrieves the handle of the window at a specified point on the screen.
VX_EXPORT WIN_HANDLE VxWindowFromPoint(CKPOINT pt);
/// @brief Retrieves the dimensions of a window's client area.
VX_EXPORT XBOOL VxGetClientRect(WIN_HANDLE Win, CKRECT *rect);
/// @brief Retrieves the dimensions of a window's bounding rectangle (including title bar, borders, etc.).
VX_EXPORT XBOOL VxGetWindowRect(WIN_HANDLE Win, CKRECT *rect);
/// @brief Converts screen coordinates of a point to client-area coordinates.
VX_EXPORT XBOOL VxScreenToClient(WIN_HANDLE Win, CKPOINT *pt);
/// @brief Converts client-area coordinates of a point to screen coordinates.
VX_EXPORT XBOOL VxClientToScreen(WIN_HANDLE Win, CKPOINT *pt);

/// @brief Changes the parent window of a child window.
VX_EXPORT WIN_HANDLE VxSetParent(WIN_HANDLE Child, WIN_HANDLE Parent);
/// @brief Retrieves the handle of a window's parent.
VX_EXPORT WIN_HANDLE VxGetParent(WIN_HANDLE Win);
/// @brief Changes the position and dimensions of a window.
VX_EXPORT XBOOL VxMoveWindow(WIN_HANDLE Win, int x, int y, int Width, int Height, XBOOL Repaint);

/// @brief Gets the path of the system's temporary directory.
VX_EXPORT XString VxGetTempPath();
/// @brief Creates a new directory.
VX_EXPORT XBOOL VxMakeDirectory(char *path);
/**
 * @brief Removes an existing, empty directory.
 * @deprecated For removing a directory and its contents, see VxDeleteDirectory.
 */
VX_EXPORT XBOOL VxRemoveDirectory(char *path);
/// @brief Deletes a directory and all its contents recursively.
VX_EXPORT XBOOL VxDeleteDirectory(char *path);
/// @brief Gets the current working directory for the application.
VX_EXPORT XBOOL VxGetCurrentDirectory(char *path);
/// @brief Sets the current working directory for the application.
VX_EXPORT XBOOL VxSetCurrentDirectory(char *path);
/// @brief Combines a path and a filename into a full path string.
VX_EXPORT XBOOL VxMakePath(char *fullpath, char *path, char *file);
/// @brief Checks if a specified amount of disk space is available in a directory.
VX_EXPORT XBOOL VxTestDiskSpace(const char *dir, XULONG size);

/// @brief Displays a standard message box.
VX_EXPORT int VxMessageBox(WIN_HANDLE hWnd, char *lpText, char *lpCaption, XULONG uType);

//------ Process access {secret}
/// @brief Retrieves the full path of the file for the specified module.
VX_EXPORT XULONG VxGetModuleFileName(INSTANCE_HANDLE Handle, char *string, XULONG StringSize);
/// @brief Retrieves a module handle for the specified module name.
VX_EXPORT INSTANCE_HANDLE VxGetModuleHandle(const char *filename);

//------ Recreates the whole file path (not the file itself) {secret}
/// @brief Creates all directories in a given file path if they do not already exist.
VX_EXPORT XBOOL VxCreateFileTree(char *file);

//------ URL Download {secret}
/// @brief Downloads a file from a URL and stores it in the browser's cache.
VX_EXPORT XULONG VxURLDownloadToCacheFile(char *File, char *CachedFile, int szCachedFile);

//------ Bitmap Functions
/// @brief Creates a device-dependent bitmap from an image description.
VX_EXPORT BITMAP_HANDLE VxCreateBitmap(const VxImageDescEx &desc);
/// @brief Deletes a bitmap handle and releases its resources.
VX_EXPORT void VxDeleteBitmap(BITMAP_HANDLE Bitmap);
/// @brief Converts a device-dependent bitmap to a device-independent format described by VxImageDescEx.
VX_EXPORT XBYTE *VxConvertBitmap(BITMAP_HANDLE Bitmap, VxImageDescEx &desc);
/// @brief Converts a bitmap to a 24-bit color format.
VX_EXPORT BITMAP_HANDLE VxConvertBitmapTo24(BITMAP_HANDLE Bitmap);
/// @brief Copies image data into an existing device-dependent bitmap.
VX_EXPORT XBOOL VxCopyBitmap(BITMAP_HANDLE Bitmap, const VxImageDescEx &desc);

/// @brief Gets information about the current operating system.
VX_EXPORT VX_OSINFO VxGetOs();

/**
 * @struct VXFONTINFO
 * @brief Holds descriptive information about a font.
 */
typedef struct VXFONTINFO {
    XString FaceName; ///< The typeface name of the font.
    int Height;       ///< The height of the font.
    int Weight;       ///< The weight of the font (e.g., 400 for normal, 700 for bold).
    XBOOL Italic;     ///< True if the font is italic.
    XBOOL Underline;  ///< True if the font is underlined.
} VXFONTINFO;

/**
 * @enum VXTEXT_ALIGNMENT
 * @brief Defines text alignment flags for drawing text within a rectangle.
 */
typedef enum VXTEXT_ALIGNMENT {
    VXTEXT_CENTER  = 0x00000001, ///< Synonym for VTEXT_HCENTER.
    VXTEXT_LEFT    = 0x00000002, ///< Aligns text to the left edge.
    VXTEXT_RIGHT   = 0x00000004, ///< Aligns text to the right edge.
    VXTEXT_TOP     = 0x00000008, ///< Aligns text to the top edge.
    VXTEXT_BOTTOM  = 0x00000010, ///< Aligns text to the bottom edge.
    VXTEXT_VCENTER = 0x00000020, ///< Centers text vertically.
    VXTEXT_HCENTER = 0x00000040, ///< Centers text horizontally.
} VXTEXT_ALIGNMENT;

//------ Font  Functions
/**
 * @brief Creates a font handle with specified characteristics.
 * @param FontName The name of the font typeface.
 * @param FontSize The height of the font.
 * @param Weight The weight of the font (e.g., 400 for normal, 700 for bold).
 * @param italic Specifies if the font should be italic.
 * @param underline Specifies if the font should be underlined.
 * @return A handle to the created font, or NULL on failure.
 */
VX_EXPORT FONT_HANDLE VxCreateFont(char *FontName, int FontSize, int Weight, XBOOL italic, XBOOL underline);

/// @brief Retrieves information about a given font handle.
VX_EXPORT XBOOL VxGetFontInfo(FONT_HANDLE Font, VXFONTINFO &desc);

/**
 * @brief Draws text onto a bitmap.
 * @param Bitmap The handle of the bitmap to draw on.
 * @param Font The handle of the font to use.
 * @param string The text string to draw.
 * @param rect Pointer to a CKRECT structure that specifies the rectangle in which to format the text.
 * @param Align A bitmask of `VXTEXT_ALIGNMENT` flags.
 * @param BkColor The background color for the text.
 * @param FontColor The color of the font.
 * @return TRUE on success, FALSE on failure.
 */
VX_EXPORT XBOOL VxDrawBitmapText(BITMAP_HANDLE Bitmap, FONT_HANDLE Font, char *string, CKRECT *rect, XULONG Align, XULONG BkColor, XULONG FontColor);

/// @brief Deletes a font handle and releases its resources.
VX_EXPORT void VxDeleteFont(FONT_HANDLE Font);

#endif // VXWINDOWFUNCTION_H
