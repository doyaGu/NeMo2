#include "CKDebugLogger.h"

#include <cstdarg>
#include <cstdlib>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

CKDebugLogger &CKDebugLogger::Instance() {
    static CKDebugLogger instance;
    return instance;
}

CKDebugLogger::CKDebugLogger()
    : m_LogFilePath(), m_DebuggerEnabled(true), m_FileEnabled(true), m_File(nullptr) {
    char modulePath[MAX_PATH] = {0};
    HMODULE module = nullptr;

    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&CKDebugLogger::Instance), &module)) {
        if (GetModuleFileNameA(module, modulePath, MAX_PATH) > 0) {
            char drive[_MAX_DRIVE] = {0};
            char dir[_MAX_DIR] = {0};
            _splitpath_s(modulePath, drive, _MAX_DRIVE, dir, _MAX_DIR, nullptr, 0, nullptr, 0);

            char logPath[MAX_PATH] = {0};
            _makepath_s(logPath, drive, dir, "CK2_3D_Debug", "log");
            m_LogFilePath = logPath;
        }
    }

    if (m_LogFilePath.empty()) {
        m_LogFilePath = "CK2_3D_Debug.log";
    }

    const char *envPath = getenv("CK2_3D_LOG");
    if (envPath && *envPath) {
        m_LogFilePath = envPath;
    }
}

CKDebugLogger::~CKDebugLogger() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_File) {
        fflush(m_File);
        fclose(m_File);
        m_File = nullptr;
    }
}

void CKDebugLogger::EnableDebuggerOutput(bool enable) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_DebuggerEnabled = enable;
}

void CKDebugLogger::EnableFileOutput(bool enable) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_FileEnabled = enable;
    if (!m_FileEnabled && m_File) {
        fclose(m_File);
        m_File = nullptr;
    }
}

void CKDebugLogger::SetLogFilePath(const char *path) {
    if (!path || !*path) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_Mutex);
    m_LogFilePath = path;
    if (m_File) {
        fclose(m_File);
        m_File = nullptr;
    }
}

void CKDebugLogger::Log(const char *msg) {
    if (!msg) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_Mutex);

    if (m_DebuggerEnabled) {
        OutputDebugStringA(msg);
        OutputDebugStringA("\n");
    }

    if (m_FileEnabled) {
        OpenFileIfNeeded();
        if (m_File) {
            fprintf(m_File, "%s\n", msg);
            fflush(m_File);
        }
    }
}

void CKDebugLogger::Logf(const char *fmt, ...) {
    if (!fmt) {
        return;
    }

    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(buffer, sizeof(buffer), _TRUNCATE, fmt, args);
    va_end(args);

    Log(buffer);
}

void CKDebugLogger::LogTagged(const char *tag, const char *msg) {
    const char *safeTag = (tag && *tag) ? tag : "General";
    const char *safeMsg = msg ? msg : "";
    Logf("[CK2_3D] [%s] %s", safeTag, safeMsg);
}

void CKDebugLogger::LogTaggedf(const char *tag, const char *fmt, ...) {
    if (!fmt) {
        return;
    }

    const char *safeTag = (tag && *tag) ? tag : "General";

    char messageBuffer[1024];
    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(messageBuffer, sizeof(messageBuffer), _TRUNCATE, fmt, args);
    va_end(args);

    Logf("[CK2_3D] [%s] %s", safeTag, messageBuffer);
}

void CKDebugLogger::Flush() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_File) {
        fflush(m_File);
    }
}

void CKDebugLogger::OpenFileIfNeeded() {
    if (m_File || !m_FileEnabled) {
        return;
    }

    fopen_s(&m_File, m_LogFilePath.c_str(), "w");
}
