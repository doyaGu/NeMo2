#include "VxWindowFunctions.h"

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <dirent.h>
#include <limits.h>
#include <fenv.h>
#include <dlfcn.h>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include "XString.h"
#include "VxColor.h"
#include "VxImageDescEx.h"

// ============================================================================
// Keyboard Functions (stubs on POSIX)
// ============================================================================

char VxScanCodeToAscii(XULONG scancode, unsigned char keystate[256]) {
    // POSIX: No direct equivalent, return null character
    (void)scancode;
    (void)keystate;
    return '\0';
}

int VxScanCodeToName(XULONG scancode, char *keyName) {
    // POSIX: No direct equivalent
    (void)scancode;
    if (keyName)
        keyName[0] = '\0';
    return 1;
}

// ============================================================================
// Cursor Functions (stubs on POSIX)
// ============================================================================

int VxShowCursor(XBOOL show) {
    // POSIX: No direct equivalent, return 0
    (void)show;
    return 0;
}

XBOOL VxSetCursor(VXCURSOR_POINTER cursorID) {
    // POSIX: No direct equivalent
    (void)cursorID;
    return TRUE;
}

// ============================================================================
// FPU Control Functions
// ============================================================================

XWORD VxGetFPUControlWord() {
    XWORD cw = 0;
#if defined(__GNUC__) || defined(__clang__)
#   if defined(__i386__) || defined(__x86_64__)
    __asm__ __volatile__ ("fstcw %0" : "=m" (cw));
#   else
    // Non-x86 architecture: use fenv
    fenv_t fenv;
    fegetenv(&fenv);
    // Return a placeholder value
    cw = 0x027F; // Default FPU control word
#   endif
#endif
    return cw;
}

void VxSetFPUControlWord(XWORD Fpu) {
#if defined(__GNUC__) || defined(__clang__)
#   if defined(__i386__) || defined(__x86_64__)
    __asm__ __volatile__ ("fldcw %0" : : "m" (Fpu));
#   else
    // Non-x86 architecture: no-op
    (void)Fpu;
#   endif
#endif
}

void VxSetBaseFPUControlWord() {}

// ============================================================================
// Environment Variable Functions
// ============================================================================

void VxAddLibrarySearchPath(char *path) {
    const char *currentPath = getenv("LD_LIBRARY_PATH");
    XString newPath;
    newPath << path;
    if (currentPath && currentPath[0] != '\0')
        newPath << ':' << currentPath;
    setenv("LD_LIBRARY_PATH", newPath.CStr(), 1);
}

XBOOL VxGetEnvironmentVariable(char *envName, XString &envValue) {
    if (!envName) {
        envValue = "";
        return FALSE;
    }

    const char *value = getenv(envName);
    if (!value) {
        envValue = "";
        return FALSE;
    }
    envValue = value;
    return TRUE;
}

XBOOL VxSetEnvironmentVariable(char *envName, char *envValue) {
    if (envValue)
        return setenv(envName, envValue, 1) == 0;
    else
        return unsetenv(envName) == 0;
}

// ============================================================================
// Window Functions (stubs on POSIX)
// ============================================================================

WIN_HANDLE VxWindowFromPoint(CKPOINT pt) {
    (void)pt;
    return NULL;
}

XBOOL VxGetClientRect(WIN_HANDLE Win, CKRECT *rect) {
    (void)Win;
    if (rect)
        memset(rect, 0, sizeof(CKRECT));
    return FALSE;
}

XBOOL VxGetWindowRect(WIN_HANDLE Win, CKRECT *rect) {
    (void)Win;
    if (rect)
        memset(rect, 0, sizeof(CKRECT));
    return FALSE;
}

XBOOL VxScreenToClient(WIN_HANDLE Win, CKPOINT *pt) {
    (void)Win;
    (void)pt;
    return FALSE;
}

XBOOL VxClientToScreen(WIN_HANDLE Win, CKPOINT *pt) {
    (void)Win;
    (void)pt;
    return FALSE;
}

WIN_HANDLE VxSetParent(WIN_HANDLE Child, WIN_HANDLE Parent) {
    (void)Child;
    (void)Parent;
    return NULL;
}

WIN_HANDLE VxGetParent(WIN_HANDLE Win) {
    (void)Win;
    return NULL;
}

XBOOL VxMoveWindow(WIN_HANDLE Win, int x, int y, int Width, int Height, XBOOL Repaint) {
    (void)Win;
    (void)x;
    (void)y;
    (void)Width;
    (void)Height;
    (void)Repaint;
    return FALSE;
}

// ============================================================================
// Directory and Path Functions
// ============================================================================

XString VxGetTempPath() {
    const char *tmpdir = getenv("TMPDIR");
    if (tmpdir)
        return XString(tmpdir);
    tmpdir = getenv("TMP");
    if (tmpdir)
        return XString(tmpdir);
    tmpdir = getenv("TEMP");
    if (tmpdir)
        return XString(tmpdir);
    return XString("/tmp/");
}

XBOOL VxMakeDirectory(char *path) {
    return mkdir(path, 0755) == 0;
}

XBOOL VxRemoveDirectory(char *path) {
    return rmdir(path) == 0;
}

// Helper function to recursively delete directory on POSIX
static XBOOL VxDeleteDirectoryRecursive(const char *path) {
    DIR *dir = opendir(path);
    if (!dir)
        return FALSE;

    struct dirent *entry;
    char fullpath[PATH_MAX];

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;

        snprintf(fullpath, sizeof(fullpath), "%s/%s", path, entry->d_name);

        struct stat st;
        if (stat(fullpath, &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                VxDeleteDirectoryRecursive(fullpath);
            } else {
                unlink(fullpath);
            }
        }
    }

    closedir(dir);
    return rmdir(path) == 0;
}

XBOOL VxDeleteDirectory(char *path) {
    return VxDeleteDirectoryRecursive(path);
}

XBOOL VxGetCurrentDirectory(char *path) {
    return getcwd(path, PATH_MAX) != NULL;
}

XBOOL VxSetCurrentDirectory(char *path) {
    return chdir(path) == 0;
}

XBOOL VxMakePath(char *fullpath, char *path, char *file) {
    if (!path || !file || !fullpath)
        return FALSE;

    strcpy(fullpath, path);
    int pathLen = strlen(path);

    if (pathLen >= PATH_MAX - (int)strlen(file) - 1) return FALSE;

    // Check if we need to add a path separator
    if (pathLen > 0 && fullpath[pathLen - 1] != '/') {
        fullpath[pathLen] = '/';
        ++pathLen;
    }

    strcpy(&fullpath[pathLen], file);
    return TRUE;
}

XBOOL VxTestDiskSpace(const char *dir, XULONG size) {
    struct statvfs st;
    if (statvfs(dir, &st) != 0)
        return FALSE;
    unsigned long long freeBytes = (unsigned long long)st.f_bavail * st.f_frsize;
    return size <= freeBytes;
}

int VxMessageBox(WIN_HANDLE hWnd, char *lpText, char *lpCaption, XULONG uType) {
    // POSIX: Print to stderr and return 0
    (void)hWnd;
    (void)uType;
    fprintf(stderr, "[%s] %s\n", lpCaption ? lpCaption : "Message", lpText ? lpText : "");
    return 0;
}

XULONG VxGetModuleFileName(INSTANCE_HANDLE Handle, char *string, XULONG StringSize) {
#if defined(__APPLE__)
    if (Handle == NULL) {
        uint32_t size = StringSize;
        if (_NSGetExecutablePath(string, &size) == 0)
            return strlen(string);
        return 0;
    }
    Dl_info info;
    if (dladdr(Handle, &info) && info.dli_fname) {
        strncpy(string, info.dli_fname, StringSize - 1);
        string[StringSize - 1] = '\0';
        return strlen(string);
    }
    return 0;
#else
    if (Handle == NULL) {
        ssize_t len = readlink("/proc/self/exe", string, StringSize - 1);
        if (len != -1) {
            string[len] = '\0';
            return len;
        }
        return 0;
    }
    Dl_info info;
    if (dladdr(Handle, &info) && info.dli_fname) {
        strncpy(string, info.dli_fname, StringSize - 1);
        string[StringSize - 1] = '\0';
        return strlen(string);
    }
    return 0;
#endif
}

INSTANCE_HANDLE VxGetModuleHandle(const char *filename) {
    if (filename == NULL)
        return dlopen(NULL, RTLD_NOW);
    return dlopen(filename, RTLD_NOW | RTLD_NOLOAD);
}

XBOOL VxCreateFileTree(char *file) {
    XString filepath = file;
    if (filepath.Length() <= 1)
        return FALSE;

    for (char *pch = &filepath[1]; *pch != '\0'; ++pch) {
        if (*pch != '/')
            continue;
        *pch = '\0';
        struct stat st;
        if (stat(filepath.CStr(), &st) != 0)
            mkdir(filepath.CStr(), 0755);
        *pch = '/';
    }
    return TRUE;
}

XULONG VxURLDownloadToCacheFile(char *File, char *CachedFile, int szCachedFile) {
    // POSIX: Not implemented - would require libcurl or similar
    (void)File;
    if (CachedFile && szCachedFile > 0)
        CachedFile[0] = '\0';
    return 1; // Return error
}

// ============================================================================
// Bitmap Functions (stubs on POSIX)
// ============================================================================

BITMAP_HANDLE VxCreateBitmap(const VxImageDescEx &desc) {
    (void)desc;
    return NULL;
}

void VxDeleteBitmap(BITMAP_HANDLE Bitmap) {
    (void)Bitmap;
}

XBYTE *VxConvertBitmap(BITMAP_HANDLE Bitmap, VxImageDescEx &desc) {
    (void)Bitmap;
    (void)desc;
    return NULL;
}

BITMAP_HANDLE VxConvertBitmapTo24(BITMAP_HANDLE Bitmap) {
    (void)Bitmap;
    return NULL;
}

XBOOL VxCopyBitmap(BITMAP_HANDLE Bitmap, const VxImageDescEx &desc) {
    (void)Bitmap;
    (void)desc;
    return FALSE;
}

// ============================================================================
// OS Info Function
// ============================================================================

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

VX_OSINFO VxGetOs() {
#if defined(__ANDROID__)
    return VXOS_ANDROID;
#elif defined(__APPLE__)
    #if TARGET_OS_IPHONE || TARGET_OS_IOS
        return VXOS_IOS;
    #else
        return VXOS_MACOS;
    #endif
#elif defined(__FreeBSD__)
    return VXOS_FREEBSD;
#elif defined(__linux__)
    return VXOS_LINUX;
#else
    return VXOS_UNKNOWN;
#endif
}

// ============================================================================
// Font Functions (stubs on POSIX)
// ============================================================================

FONT_HANDLE VxCreateFont(char *FontName, int FontSize, int Weight, XBOOL italic, XBOOL underline) {
    (void)FontName;
    (void)FontSize;
    (void)Weight;
    (void)italic;
    (void)underline;
    return NULL;
}

XBOOL VxGetFontInfo(FONT_HANDLE Font, VXFONTINFO &desc) {
    (void)Font;
    (void)desc;
    return FALSE;
}

XBOOL VxDrawBitmapText(BITMAP_HANDLE Bitmap, FONT_HANDLE Font, char *string, CKRECT *rect, XULONG Align, XULONG BkColor, XULONG FontColor) {
    (void)Bitmap;
    (void)Font;
    (void)string;
    (void)rect;
    (void)Align;
    (void)BkColor;
    (void)FontColor;
    return FALSE;
}

void VxDeleteFont(FONT_HANDLE Font) {
    (void)Font;
}
