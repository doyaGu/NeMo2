#pragma once

#include <cstdio>
#include <mutex>
#include <string>

class CKDebugLogger {
public:
    static CKDebugLogger &Instance();

    void EnableDebuggerOutput(bool enable);
    void EnableFileOutput(bool enable);
    void SetLogFilePath(const char *path);

    void Log(const char *msg);
    void Logf(const char *fmt, ...);
    void LogTagged(const char *tag, const char *msg);
    void LogTaggedf(const char *tag, const char *fmt, ...);
    void Flush();

private:
    CKDebugLogger();
    ~CKDebugLogger();

    CKDebugLogger(const CKDebugLogger &) = delete;
    CKDebugLogger &operator=(const CKDebugLogger &) = delete;

    void OpenFileIfNeeded();

    std::string m_LogFilePath;
    bool m_DebuggerEnabled;
    bool m_FileEnabled;
    FILE *m_File;
    std::mutex m_Mutex;
};

#define CK_LOG_RAW(msg) CKDebugLogger::Instance().Log(msg)
#define CK_LOG_RAW_FMT(fmt, ...) CKDebugLogger::Instance().Logf(fmt, __VA_ARGS__)
#define CK_LOG(category, msg) CKDebugLogger::Instance().LogTagged(category, msg)
#define CK_LOG_FMT(category, fmt, ...) CKDebugLogger::Instance().LogTaggedf(category, fmt, __VA_ARGS__)
