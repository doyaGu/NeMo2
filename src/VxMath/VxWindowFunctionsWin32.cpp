#include "VxWindowFunctions.h"

#include <Windows.h>

#include <direct.h>
#include <shellapi.h>
#include <string>

#include "XString.h"
#include "VxColor.h"
#include "VxImageDescEx.h"
#include "VxSharedLibrary.h"

extern void VxDoBlitUpsideDown(const VxImageDescEx &src_desc, const VxImageDescEx &dst_desc);

char VxScanCodeToAscii(XULONG scancode, unsigned char keystate[256]) {
    unsigned char state[256];

    GetKeyboardState(state);
    if (keystate[VK_PRINT]) {
        state[VK_SHIFT] |= 0x81; // 0x81: the key is down and toggled
        state[VK_LSHIFT] |= 0x81;
    }
    if (keystate[0x36]) // VK_6
    {
        state[VK_SHIFT] |= 0x81;
        state[VK_RSHIFT] |= 0x81;
    }
    if (keystate[0x38]) // VK_8
    {
        state[VK_MENU] |= 0x81;
        state[VK_LMENU] |= 0x81;
    }
    if (keystate[0xB8]) // reserved
    {
        state[VK_CONTROL] |= 0x81;
        state[VK_MENU] |= 0x81;
        state[VK_LCONTROL] |= 0x81;
        state[VK_RMENU] |= 0x81;
    }
    if (keystate[VK_NONCONVERT]) {
        state[VK_CONTROL] |= 0x81;
        state[VK_LCONTROL] |= 0x81;
    }
    if (keystate[0x9D]) // unassigned
    {
        state[VK_CONTROL] |= 0x81;
        state[VK_RCONTROL] |= 0x81;
    }
    UINT vkey = MapVirtualKeyA(scancode, MAPVK_VSC_TO_VK);
    WORD ch;
    int ret = ToAscii(vkey, scancode, state, &ch, 0);
    return (ret != 0) ? ch : '\0';
}

int VxScanCodeToName(XULONG scancode, char *keyName) {
    DWORD code = (scancode & 0x7F) << 16;
    if (scancode > 0x7F)
        code |= 0x1000000;

    char name[32];
    int len = GetKeyNameTextA(code, name, 32);
    if (len > 0)
        strcpy(keyName, name);
    return len + 1;
}

int VxShowCursor(XBOOL show) {
    return ShowCursor(show);
}

XBOOL VxSetCursor(VXCURSOR_POINTER cursorID) {
    HCURSOR cursorNew = NULL;
    HCURSOR cursorNow = GetCursor();
    switch (cursorID) {
    case 1:
        cursorNew = LoadCursorA(0, (LPCSTR) IDC_ARROW);
        break;
    case 2:
        cursorNew = LoadCursorA(0, (LPCSTR) IDC_WAIT);
        break;
    case 3:
        cursorNew = LoadCursorA(0, (LPCSTR) IDC_SIZEALL);
        break;
    default:
        return TRUE;
    }
    if (cursorNew && cursorNew != cursorNow)
        SetCursor(cursorNew);
    return TRUE;
}

XWORD VxGetFPUControlWord() {
    XWORD cw = 0;
#if defined(_MSC_VER)  // MSVC
    __asm {
        fstcw cw
    }
#elif defined(__GNUC__) || defined(__clang__)  // GCC / Clang
    __asm__ __volatile__ ("fstcw %0" : "=m" (cw));
#else
#error "Unsupported compiler"
#endif
    return cw;
}

void VxSetFPUControlWord(XWORD Fpu) {
#if defined(_MSC_VER)  // MSVC
    __asm {
        fldcw Fpu
    }
#elif defined(__GNUC__) || defined(__clang__)  // GCC / Clang
    __asm__ __volatile__ ("fldcw %0" : : "m" (Fpu));
#else
#error "Unsupported compiler"
#endif
}

void VxSetBaseFPUControlWord() {}

void VxAddLibrarySearchPath(char *path) {
    char buffer[4096];
    GetEnvironmentVariableA("PATH", buffer, sizeof(buffer));
    XString newPath;
    newPath << path << ';' << buffer;
    if (newPath[newPath.Length() - 1] != ';')
        newPath << ';';
    SetEnvironmentVariableA("PATH", newPath.CStr());
}

XBOOL VxGetEnvironmentVariable(char *envName, XString &envValue) {
    if (!envName) {
        envValue = "";
        return FALSE;
    }

    DWORD required = GetEnvironmentVariableA(envName, nullptr, 0);
    if (required == 0) {
        envValue = "";
        return FALSE;
    }

    std::string buffer(required, '\0');
    DWORD written = GetEnvironmentVariableA(envName, buffer.data(), required);
    if (written == 0 || written >= required) {
        envValue = "";
        return FALSE;
    }

    buffer.resize(written);
    envValue = buffer.c_str();
    return TRUE;
}

XBOOL VxSetEnvironmentVariable(char *envName, char *envValue) {
    return SetEnvironmentVariableA(envName, envValue);
}

//------ Window Functions
WIN_HANDLE VxWindowFromPoint(CKPOINT pt) {
    return WindowFromPoint(*(LPPOINT) &pt);
}

XBOOL VxGetClientRect(WIN_HANDLE Win, CKRECT *rect) {
    return GetClientRect((HWND) Win, (LPRECT) rect);
}

XBOOL VxGetWindowRect(WIN_HANDLE Win, CKRECT *rect) {
    return GetWindowRect((HWND) Win, (LPRECT) rect);
}

XBOOL VxScreenToClient(WIN_HANDLE Win, CKPOINT *pt) {
    return ScreenToClient((HWND) Win, (LPPOINT) pt);
}

XBOOL VxClientToScreen(WIN_HANDLE Win, CKPOINT *pt) {
    return ClientToScreen((HWND) Win, (LPPOINT) pt);
}

WIN_HANDLE VxSetParent(WIN_HANDLE Child, WIN_HANDLE Parent) {
    return SetParent((HWND) Child, (HWND) Parent);
}

WIN_HANDLE VxGetParent(WIN_HANDLE Win) {
    return GetParent((HWND) Win);
}

XBOOL VxMoveWindow(WIN_HANDLE Win, int x, int y, int Width, int Height, XBOOL Repaint) {
    return MoveWindow((HWND) Win, x, y, Width, Height, Repaint);
}

XString VxGetTempPath() {
    char buffer[512];
    GetTempPathA(512, buffer);
    return XString(buffer);
}

XBOOL VxMakeDirectory(char *path) {
    return mkdir(path) == 0;
}

XBOOL VxRemoveDirectory(char *path) {
    return RemoveDirectoryA(path);
}

XBOOL VxDeleteDirectory(char *path) {
    VxSharedLibrary lib;
    if (!lib.Load("Shell32.dll"))
        return FALSE;

    int ret = FALSE;
    typedef int (WINAPI *LPFSHPROC)(LPSHFILEOPSTRUCTA);
    LPFSHPROC mySHFileOperation = (LPFSHPROC) lib.GetFunctionPtr("SHFileOperation");
    if (mySHFileOperation || (mySHFileOperation = (LPFSHPROC) lib.GetFunctionPtr("SHFileOperationA"))) {
        SHFILEOPSTRUCTA fileOp;
        memset(&fileOp, 0, sizeof(SHFILEOPSTRUCTA));
        fileOp.pFrom = path;
        fileOp.wFunc = FO_DELETE;
        fileOp.fFlags = FOF_NO_UI;
        ret = mySHFileOperation(&fileOp) == 0;
    }
    lib.ReleaseLibrary();
    return ret;
}

XBOOL VxGetCurrentDirectory(char *path) {
    return getcwd(path, MAX_PATH) != NULL;
}

XBOOL VxSetCurrentDirectory(char *path) {
    return chdir(path) == 0;
}

XBOOL VxMakePath(char *fullpath, char *path, char *file) {
    if (!path || !file || !fullpath)
        return FALSE;

    strcpy(fullpath, path);
    int pathLen = strlen(path);

    if (pathLen >= MAX_PATH - strlen(file) - 1) return FALSE;

    // Check if we need to add a path separator
    if (pathLen > 0 && fullpath[pathLen - 1] != '\\' && fullpath[pathLen - 1] != '/') {
        fullpath[pathLen] = '\\';
        ++pathLen; // Increment position since we added a separator
    }

    strcpy(&fullpath[pathLen], file); // Use current pathLen, not pathLen + 1
    return TRUE;
}

XBOOL VxTestDiskSpace(const char *dir, XULONG size) {
    HMODULE hKernel32 = GetModuleHandleA("kernel32.dll");
    typedef BOOL (__stdcall *LPFNGETDISKFREESPACEEXA)(LPCSTR, PULARGE_INTEGER, PULARGE_INTEGER, PULARGE_INTEGER);
    LPFNGETDISKFREESPACEEXA lpfnGetDiskFreeSpaceExA = (LPFNGETDISKFREESPACEEXA) GetProcAddress(hKernel32, "GetDiskFreeSpaceExA");

    XBOOL ret;
    if (lpfnGetDiskFreeSpaceExA) {
        ULARGE_INTEGER freeBytesAvailableToCaller;
        ULARGE_INTEGER totalNumberOfBytes;
        ULARGE_INTEGER totalNumberOfFreeBytes;
        ret = lpfnGetDiskFreeSpaceExA(dir, &freeBytesAvailableToCaller, &totalNumberOfBytes, &totalNumberOfFreeBytes);
        return ret && size <= freeBytesAvailableToCaller.QuadPart;
    } else {
        DWORD sectorsPerCluster = 0;
        DWORD bytesPerSector = 0;
        DWORD numberOfFreeClusters = 0;
        DWORD totalNumberOfClusters = 0;
        ret = GetDiskFreeSpaceA(dir, &sectorsPerCluster, &bytesPerSector, &numberOfFreeClusters, &totalNumberOfClusters);
        return ret && size <= sectorsPerCluster * bytesPerSector * numberOfFreeClusters;
    }
}

int VxMessageBox(WIN_HANDLE hWnd, char *lpText, char *lpCaption, XULONG uType) {
    return MessageBoxA((HWND) hWnd, lpText, lpCaption, uType);
}

XULONG VxGetModuleFileName(INSTANCE_HANDLE Handle, char *string, XULONG StringSize) {
    return GetModuleFileNameA((HMODULE) Handle, string, StringSize);
}

INSTANCE_HANDLE VxGetModuleHandle(const char *filename) {
    return GetModuleHandleA(filename);
}

XBOOL VxCreateFileTree(char *file) {
    XString filepath = file;
    if (filepath.Length() <= 2)
        return FALSE;

    for (char *pch = &filepath[3]; *pch != '\0'; ++pch) {
        if (*pch != '/' && *pch != '\\')
            continue;
        *pch = '\0';
        if (GetFileAttributesA(filepath.CStr()) == -1)
            if (!CreateDirectoryA(filepath.CStr(), NULL))
                break;
        *pch = '\\';
    }
    return TRUE;
}

XULONG VxURLDownloadToCacheFile(char *File, char *CachedFile, int szCachedFile) {
    char *cachedFile = CachedFile;
    CachedFile[0] = '\0';

    VxSharedLibrary sl;
    if (!sl.Load("Urlmon.dll"))
        return E_FAIL;

    typedef HRESULT (__stdcall *LPFNURLDOWNLOADTOCACHEFILE)(LPUNKNOWN, LPCSTR, LPTSTR, DWORD, DWORD, IBindStatusCallback *);
    LPFNURLDOWNLOADTOCACHEFILE lpfnURLDownloadToCacheFileA = (LPFNURLDOWNLOADTOCACHEFILE) sl.GetFunctionPtr("URLDownloadToCacheFileA");
    XULONG ret = E_FAIL;
    if (lpfnURLDownloadToCacheFileA)
        ret = lpfnURLDownloadToCacheFileA(NULL, File, cachedFile, szCachedFile, 0, NULL);
    sl.ReleaseLibrary();
    return ret;
}

BITMAP_HANDLE VxCreateBitmap(const VxImageDescEx &desc) {
    // Create a bitmap information header
    BITMAPINFO bmi;
    memset(&bmi, 0, sizeof(BITMAPINFO));

    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = desc.Width;
    bmi.bmiHeader.biHeight = desc.Height;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 24;
    bmi.bmiHeader.biCompression = BI_RGB;

    // Create a DIB section
    HBITMAP bitmap = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, NULL, NULL, 0);
    if (!bitmap) return NULL;

    // Get bitmap information
    BITMAP bm;
    DIBSECTION dibSection;
    GetObjectA(bitmap, sizeof(BITMAP), &bm);
    GetObjectA(bitmap, sizeof(DIBSECTION), &dibSection);

    // Calculate proper bytes per line
    int bytePerLine = dibSection.dsBm.bmWidthBytes;
    if ((bytePerLine & 3) != 0 && bytePerLine != dibSection.dsBmih.biSizeImage / dibSection.dsBm.bmHeight) {
        bytePerLine = (int) dibSection.dsBmih.biSizeImage / dibSection.dsBm.bmHeight;
    }

    if (!bm.bmBits) return NULL;

    VxImageDescEx dstDesc;
    dstDesc.Width = desc.Width;
    dstDesc.Height = desc.Height;
    dstDesc.BytesPerLine = bytePerLine;
    dstDesc.BitsPerPixel = bm.bmBitsPixel;
    dstDesc.RedMask = R_MASK;
    dstDesc.GreenMask = G_MASK;
    dstDesc.BlueMask = B_MASK;
    dstDesc.AlphaMask = 0;
    dstDesc.Image = static_cast<XBYTE *>(bm.bmBits);

    if (desc.Image)
        VxDoBlitUpsideDown(desc, dstDesc);

    return bitmap;
}

void VxDeleteBitmap(BITMAP_HANDLE Bitmap) {
    if (Bitmap) DeleteObject(Bitmap);
}

XBYTE *VxConvertBitmap(BITMAP_HANDLE Bitmap, VxImageDescEx &desc) {
    BITMAP_HANDLE bitmap24 = VxConvertBitmapTo24(Bitmap);
    if (!bitmap24) return NULL;

    // Get bitmap info
    DIBSECTION dibSection;
    if (!GetObjectA(bitmap24, sizeof(DIBSECTION), &dibSection)) {
        if (bitmap24 != Bitmap)
            DeleteObject(bitmap24);
        return NULL;
    }

    if (!dibSection.dsBm.bmBits) {
        if (bitmap24 != Bitmap)
            DeleteObject(bitmap24);
        return NULL;
    }

    // Create source desc
    VxImageDescEx srcDesc;
    srcDesc.Width = dibSection.dsBm.bmWidth;
    srcDesc.Height = dibSection.dsBm.bmHeight;
    srcDesc.BytesPerLine = dibSection.dsBm.bmWidthBytes;
    srcDesc.BitsPerPixel = dibSection.dsBm.bmBitsPixel;
    srcDesc.RedMask = R_MASK;
    srcDesc.GreenMask = G_MASK;
    srcDesc.BlueMask = B_MASK;
    srcDesc.AlphaMask = 0;
    srcDesc.Image = static_cast<XBYTE *>(dibSection.dsBm.bmBits);

    // Create destination desc (32-bit)
    VxImageDescEx dstDesc = srcDesc;
    dstDesc.BytesPerLine = 4 * srcDesc.Width;
    dstDesc.BitsPerPixel = 32;

    XBYTE *newImage = new XBYTE[4 * srcDesc.Width * dstDesc.Height];
    if (!newImage) {
        if (bitmap24 != Bitmap)
            DeleteObject(bitmap24);
        return NULL;
    }
    dstDesc.Image = newImage;

    VxDoBlitUpsideDown(srcDesc, dstDesc);

    if (bitmap24 != Bitmap)
        DeleteObject(bitmap24);

    desc = dstDesc;
    desc.Image = newImage;

    return newImage;
}

BITMAP_HANDLE VxConvertBitmapTo24(BITMAP_HANDLE Bitmap) {
    if (!Bitmap) return NULL;

    BITMAP bm;
    GetObjectA(Bitmap, sizeof(BITMAP), &bm);

    if (bm.bmBits && bm.bmBitsPixel == 24)
        return Bitmap;

    HDC srcDC = CreateCompatibleDC(NULL);
    if (!srcDC) return NULL;

    HGDIOBJ oldSrcObj = SelectObject(srcDC, Bitmap);

    // Create 24-bit DIB section
    BITMAPINFO bmi;
    memset(&bmi, 0, sizeof(bmi));
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = bm.bmWidth;
    bmi.bmiHeader.biHeight = bm.bmHeight;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 24;
    bmi.bmiHeader.biCompression = BI_RGB;

    HBITMAP bitmap24 = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, NULL, NULL, 0);
    if (!bitmap24) {
        SelectObject(srcDC, oldSrcObj);
        DeleteDC(srcDC);
        return NULL;
    }

    HDC dstDC = CreateCompatibleDC(srcDC);
    if (!dstDC) {
        SelectObject(srcDC, oldSrcObj);
        DeleteDC(srcDC);
        DeleteObject(bitmap24);
        return NULL;
    }

    HGDIOBJ oldDstObj = SelectObject(dstDC, bitmap24);

    // Copy bitmap
    BitBlt(dstDC, 0, 0, bm.bmWidth, bm.bmHeight, srcDC, 0, 0, SRCCOPY);

    // Cleanup
    SelectObject(srcDC, oldSrcObj);
    SelectObject(dstDC, oldDstObj);
    DeleteDC(srcDC);
    DeleteDC(dstDC);

    return bitmap24;
}

XBOOL VxCopyBitmap(BITMAP_HANDLE Bitmap, const VxImageDescEx &desc) {
    BITMAP_HANDLE bitmap24 = VxConvertBitmapTo24(Bitmap);
    if (!bitmap24) return FALSE;

    // Get extended bitmap info
    DIBSECTION dibSection;
    if (!GetObjectA(bitmap24, sizeof(DIBSECTION), &dibSection)) {
        if (bitmap24 != Bitmap) DeleteObject(bitmap24);
        return FALSE;
    }

    if (!dibSection.dsBm.bmBits) {
        if (bitmap24 != Bitmap) DeleteObject(bitmap24);
        return FALSE;
    }

    // Calculate proper bytes per line
    int bytePerLine = dibSection.dsBm.bmWidthBytes;
    if ((bytePerLine & 3) != 0 && bytePerLine != dibSection.dsBmih.biSizeImage / dibSection.dsBm.bmHeight) {
        bytePerLine = (int) dibSection.dsBmih.biSizeImage / dibSection.dsBm.bmHeight;
    }

    // Create source desc
    VxImageDescEx srcDesc;
    srcDesc.Width = dibSection.dsBm.bmWidth;
    srcDesc.Height = dibSection.dsBm.bmHeight;
    srcDesc.BytesPerLine = bytePerLine;
    srcDesc.BitsPerPixel = dibSection.dsBm.bmBitsPixel;
    srcDesc.RedMask = R_MASK;
    srcDesc.GreenMask = G_MASK;
    srcDesc.BlueMask = B_MASK;
    srcDesc.AlphaMask = 0;
    srcDesc.Image = static_cast<XBYTE *>(dibSection.dsBm.bmBits);

    VxDoBlitUpsideDown(srcDesc, desc);

    if (bitmap24 != Bitmap) DeleteObject(bitmap24);

    return TRUE;
}

VX_OSINFO VxGetOs() {
    OSVERSIONINFOA osvi;
    memset(&osvi, 0, sizeof(OSVERSIONINFOA));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOA);
    GetVersionExA(&osvi);
    switch (osvi.dwPlatformId) {
    case VER_PLATFORM_WIN32s:
        return VXOS_WIN31;
    case VER_PLATFORM_WIN32_WINDOWS:
        if (osvi.dwMinorVersion == 0)
            return VXOS_WIN95;
        else if (osvi.dwMinorVersion < 0x5A)
            return VXOS_WIN98;
        else
            return VXOS_WINME;
    case VER_PLATFORM_WIN32_NT:
        if (osvi.dwMajorVersion == 4)
            return VXOS_WINNT4;
        if (osvi.dwMajorVersion > 4)
            return VXOS_WIN2K;
    default:
        return VXOS_UNKNOWN;
    }
}

FONT_HANDLE VxCreateFont(char *FontName, int FontSize, int Weight, XBOOL italic, XBOOL underline) {
    if (!FontName || FontSize <= 0) return NULL;

    HDC hDC = CreateCompatibleDC(NULL);
    if (!hDC) return NULL;

    // Convert point size to logical units
    int logicalHeight = -MulDiv(FontSize, GetDeviceCaps(hDC, LOGPIXELSY), 72);

    HFONT hFont = CreateFontA(
        logicalHeight,               // Height
        0,                           // Width (0 means "match height")
        0,                           // Escapement
        0,                           // Orientation
        Weight,                      // Weight
        italic,                      // Italic
        underline,                   // Underline
        0,                           // StrikeOut
        ANSI_CHARSET,                // CharSet
        OUT_DEFAULT_PRECIS,          // OutputPrecision
        CLIP_DEFAULT_PRECIS,         // ClipPrecision
        ANTIALIASED_QUALITY,         // Quality
        DEFAULT_PITCH | FF_DONTCARE, // Pitch and Family
        FontName                     // FaceName
    );

    DeleteDC(hDC);

    return hFont;
}

XBOOL VxGetFontInfo(FONT_HANDLE Font, VXFONTINFO &desc) {
    if (!Font) return FALSE;

    LOGFONTA logFont;
    if (!GetObjectA(Font, sizeof(LOGFONTA), &logFont)) return FALSE;

    // Copy data to the VXFONTINFO structure
    desc.FaceName = logFont.lfFaceName;
    desc.Height = (logFont.lfHeight < 0) ? -logFont.lfHeight : logFont.lfHeight;
    desc.Weight = logFont.lfWeight;
    desc.Italic = logFont.lfItalic ? TRUE : FALSE;
    desc.Underline = logFont.lfUnderline ? TRUE : FALSE;

    return TRUE;
}

XBOOL VxDrawBitmapText(BITMAP_HANDLE Bitmap, FONT_HANDLE Font, char *string, CKRECT *rect, XULONG Align, XULONG BkColor, XULONG FontColor) {
    if (!Bitmap || !Font || !string || !rect) return FALSE;

    HDC hDC = CreateCompatibleDC(NULL);
    if (!hDC) return FALSE;

    // Select the bitmap and font into the DC
    HGDIOBJ oldBitmap = SelectObject(hDC, Bitmap);
    HGDIOBJ oldFont = SelectObject(hDC, Font);
    if (!oldBitmap || !oldFont) {
        if (oldBitmap) SelectObject(hDC, oldBitmap);
        if (oldFont) SelectObject(hDC, oldFont);
        DeleteDC(hDC);
        return FALSE;
    }

    // Convert RGB values (the original code swaps R and B components)
    COLORREF textColor = RGB(
        (FontColor & 0xFF),        // B
        ((FontColor >> 8) & 0xFF), // G
        ((FontColor >> 16) & 0xFF) // R
    );

    COLORREF bkColor = RGB(
        (BkColor & 0xFF),        // B
        ((BkColor >> 8) & 0xFF), // G
        ((BkColor >> 16) & 0xFF) // R
    );

    // Set text and background colors
    SetTextColor(hDC, textColor);
    SetBkColor(hDC, bkColor);

    // Convert alignment flags
    UINT drawFlags = 0;

    // Horizontal alignment
    if (Align & VXTEXT_CENTER || Align & VXTEXT_HCENTER) {
        drawFlags |= DT_CENTER;
    } else if (Align & VXTEXT_RIGHT) {
        drawFlags |= DT_RIGHT;
    } else {
        drawFlags |= DT_LEFT;
    }

    // Vertical alignment
    if (Align & VXTEXT_BOTTOM) {
        drawFlags |= DT_BOTTOM | DT_SINGLELINE;
    } else if (Align & VXTEXT_VCENTER) {
        drawFlags |= DT_VCENTER | DT_SINGLELINE;
    } else {
        drawFlags |= DT_TOP | DT_SINGLELINE;
    }

    // Draw the text
    int result = DrawTextA(hDC, string, -1, (LPRECT) rect, drawFlags);

    // Restore original objects and clean up
    SelectObject(hDC, oldFont);
    SelectObject(hDC, oldBitmap);
    DeleteDC(hDC);

    return result != 0;
}

void VxDeleteFont(FONT_HANDLE Font) {
    if (Font) DeleteObject(Font);
}
