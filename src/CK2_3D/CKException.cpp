#include "CKException.h"

#include <DbgHelp.h>
#include <cstdarg>
#include <cstdio>

#pragma comment(lib, "DbgHelp.lib")

#include "CKDebugLogger.h"

namespace {
    HANDLE g_ProcessHandle = GetCurrentProcess();
    bool g_SymbolsInitialized = false;
    LONG (WINAPI *g_PreviousFilter)(EXCEPTION_POINTERS *) = nullptr;

    void LogMessage(const char *fmt, ...) {
        char buffer[1024];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);
        CKDebugLogger::Instance().Log(buffer);
    }

    bool EnsureSymbolsInitialized() {
        if (g_SymbolsInitialized) {
            return true;
        }

        DWORD options = SymGetOptions();
        options |= SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES;
        SymSetOptions(options);

        if (SymInitialize(g_ProcessHandle, nullptr, TRUE)) {
            g_SymbolsInitialized = true;
            return true;
        }

        LogMessage("[CK2_3D] SymInitialize failed (err=%lu)", GetLastError());
        return false;
    }

    void CleanupSymbols() {
        if (!g_SymbolsInitialized) {
            return;
        }
        SymCleanup(g_ProcessHandle);
        g_SymbolsInitialized = false;
    }

    void LogRegisterState(const CONTEXT &ctx) {
#if defined(_M_IX86)
        LogMessage(
            "[CK2_3D] Registers: EAX=%08lX EBX=%08lX ECX=%08lX EDX=%08lX ESI=%08lX EDI=%08lX EBP=%08lX ESP=%08lX EIP=%08lX",
            ctx.Eax, ctx.Ebx, ctx.Ecx, ctx.Edx, ctx.Esi, ctx.Edi, ctx.Ebp, ctx.Esp, ctx.Eip);
#elif defined(_M_X64)
        LogMessage(
            "[CK2_3D] Registers: RAX=%016llX RBX=%016llX RCX=%016llX RDX=%016llX RSI=%016llX RDI=%016llX RBP=%016llX RSP=%016llX RIP=%016llX",
            ctx.Rax, ctx.Rbx, ctx.Rcx, ctx.Rdx, ctx.Rsi, ctx.Rdi, ctx.Rbp, ctx.Rsp, ctx.Rip);
#else
        LogMessage("[CK2_3D] Register logging not supported on this architecture");
#endif
    }

    void LogExceptionHeader(EXCEPTION_RECORD *record) {
        char moduleName[MAX_PATH] = "Unknown";
        HMODULE hModule = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(record->ExceptionAddress), &hModule) && hModule) {
            GetModuleFileNameA(hModule, moduleName, MAX_PATH);
        }

        LogMessage("[CK2_3D] CRASH: code=0x%08lX flags=0x%08lX at=0x%p module=%s thread=0x%04X",
                   record->ExceptionCode, record->ExceptionFlags, record->ExceptionAddress, moduleName,
                   GetCurrentThreadId());

        if (record->NumberParameters > 0) {
            const DWORD count = record->NumberParameters;
            char paramBuffer[512];
            int offset = std::snprintf(paramBuffer, sizeof(paramBuffer), "[CK2_3D] Exception parameters (%lu):", count);
            for (DWORD i = 0; i < count && offset < static_cast<int>(sizeof(paramBuffer)); ++i) {
                offset += std::snprintf(paramBuffer + offset, sizeof(paramBuffer) - offset, " [%lu]=0x%p", i,
                                        reinterpret_cast<void *>(record->ExceptionInformation[i]));
            }
            CKDebugLogger::Instance().Log(paramBuffer);
        }
    }

    void LogStackTrace(CONTEXT context) {
#if defined(_M_IX86)
        const DWORD machineType = IMAGE_FILE_MACHINE_I386;
        STACKFRAME64 frame{};
        frame.AddrPC.Offset = context.Eip;
        frame.AddrFrame.Offset = context.Ebp;
        frame.AddrStack.Offset = context.Esp;
#elif defined(_M_X64)
        const DWORD machineType = IMAGE_FILE_MACHINE_AMD64;
        STACKFRAME64 frame{};
        frame.AddrPC.Offset = context.Rip;
        frame.AddrFrame.Offset = context.Rbp;
        frame.AddrStack.Offset = context.Rsp;
#else
        CKDebugLogger::Instance().Log("[CK2_3D] Stack trace not supported on this architecture");
        return;
#endif

        frame.AddrPC.Mode = AddrModeFlat;
        frame.AddrFrame.Mode = AddrModeFlat;
        frame.AddrStack.Mode = AddrModeFlat;

        LogMessage("[CK2_3D] Stack trace:");

        const bool symbolsReady = EnsureSymbolsInitialized();
        for (int i = 0; i < 64; ++i) {
            if (!StackWalk64(machineType, g_ProcessHandle, GetCurrentThread(), &frame, &context, nullptr,
                             SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
                break;
            }

            if (frame.AddrPC.Offset == 0) {
                break;
            }

            DWORD64 addr = frame.AddrPC.Offset;

            char moduleName[MAX_PATH] = "";
            HMODULE hMod = nullptr;
            if (GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCSTR>(addr), &hMod) && hMod) {
                GetModuleFileNameA(hMod, moduleName, MAX_PATH);
            }

            char symbolBuffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = {};
            SYMBOL_INFO *symbol = reinterpret_cast<SYMBOL_INFO *>(symbolBuffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = MAX_SYM_NAME;
            DWORD64 displacement = 0;
            const char *symbolName = "<unknown>";

            if (symbolsReady && SymFromAddr(g_ProcessHandle, addr, &displacement, symbol)) {
                symbolName = symbol->Name;
            }

            IMAGEHLP_LINE64 lineInfo{};
            lineInfo.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
            DWORD lineDisplacement = 0;
            const char *fileName = nullptr;
            if (symbolsReady && SymGetLineFromAddr64(g_ProcessHandle, addr, &lineDisplacement, &lineInfo)) {
                fileName = lineInfo.FileName;
            }

            char lineBuffer[512];
            if (fileName && fileName[0]) {
                std::snprintf(lineBuffer, sizeof(lineBuffer), "[CK2_3D]   #%02d 0x%p %s + 0x%llX (%s:%lu)", i,
                              reinterpret_cast<void *>(addr), symbolName,
                              static_cast<unsigned long long>(displacement), fileName, lineInfo.LineNumber);
            } else if (moduleName[0] != '\0') {
                std::snprintf(lineBuffer, sizeof(lineBuffer), "[CK2_3D]   #%02d 0x%p %s + 0x%llX (%s)", i,
                              reinterpret_cast<void *>(addr), symbolName,
                              static_cast<unsigned long long>(displacement), moduleName);
            } else {
                std::snprintf(lineBuffer, sizeof(lineBuffer), "[CK2_3D]   #%02d 0x%p %s + 0x%llX", i,
                              reinterpret_cast<void *>(addr), symbolName,
                              static_cast<unsigned long long>(displacement));
            }

            CKDebugLogger::Instance().Log(lineBuffer);
        }
    }
} // namespace

void CKInstallExceptionHandler() {
    g_PreviousFilter = SetUnhandledExceptionFilter(CKExceptionHandler);
    CKDebugLogger::Instance().Log("[CK2_3D] Unhandled exception filter installed");
}

void CKRemoveExceptionHandler() {
    SetUnhandledExceptionFilter(g_PreviousFilter);
    g_PreviousFilter = nullptr;
    CleanupSymbols();
    CKDebugLogger::Instance().Log("[CK2_3D] Unhandled exception filter removed");
}

LONG WINAPI CKExceptionHandler(EXCEPTION_POINTERS *pExceptionInfo) {
    if (!pExceptionInfo || !pExceptionInfo->ExceptionRecord || !pExceptionInfo->ContextRecord) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    SYSTEMTIME st;
    GetLocalTime(&st);
    LogMessage("[CK2_3D] ===== Unhandled exception =====");
    LogMessage("[CK2_3D] Time: %04u-%02u-%02u %02u:%02u:%02u.%03u", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute,
               st.wSecond, st.wMilliseconds);

    LogExceptionHeader(pExceptionInfo->ExceptionRecord);
    LogRegisterState(*pExceptionInfo->ContextRecord);
    LogStackTrace(*pExceptionInfo->ContextRecord);
    LogMessage("[CK2_3D] ===== End of report =====");

    return EXCEPTION_CONTINUE_SEARCH;
}
