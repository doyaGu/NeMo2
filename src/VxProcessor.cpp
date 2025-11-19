#include "VxMath.h"

#include <stdio.h>
#include <string.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#if defined(__GNUC__)
#include <cpuid.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

// Global variables
float g_MSecondsPerCycle;
ProcessorsType g_ProcessorType;
int g_ProcessorFrequency;
XULONG g_ProcessorFeatures;
XULONG g_InstructionSetExtensions;
char g_ProcDescription[64];

typedef struct CpuIdRegs {
    XULONG eax;
    XULONG ebx;
    XULONG ecx;
    XULONG edx;
} CpuIdRegs;

static CpuIdRegs cpuid(XULONG eax) {
    CpuIdRegs regs;
#if defined(__GNUC__)
    __cpuid(eax, regs.eax, regs.ebx, regs.ecx, regs.edx);
#else
    int regsArray[4];
    __cpuid(regsArray, (int) eax);
    regs.eax = regsArray[0];
    regs.ebx = regsArray[1];
    regs.ecx = regsArray[2];
    regs.edx = regsArray[3];
#endif
    return regs;
}

static inline CpuIdRegs cpuidex(XULONG eax, XULONG ecx) {
    CpuIdRegs regs;
#if defined(__GNUC__)
    __cpuid_count(eax, ecx, regs.eax, regs.ebx, regs.ecx, regs.edx);
#else
    int regsArray[4];
    __cpuidex(regsArray, (int) eax, (int) ecx);
    regs.eax = regsArray[0];
    regs.ebx = regsArray[1];
    regs.ecx = regsArray[2];
    regs.edx = regsArray[3];
#endif
    return regs;
}

// CPU Vendor identification
typedef enum {
    CPU_VENDOR_UNKNOWN = 0,
    CPU_VENDOR_INTEL,
    CPU_VENDOR_AMD
} CpuVendor;

static CpuVendor GetCpuVendor() {
    CpuIdRegs regs = cpuid(0);

    // Intel: "GenuineIntel"
    if (regs.ebx == 0x756E6547 && regs.edx == 0x49656E69 && regs.ecx == 0x6C65746E) {
        return CPU_VENDOR_INTEL;
    }
    // AMD: "AuthenticAMD"
    else if (regs.ebx == 0x68747541 && regs.edx == 0x69746E65 && regs.ecx == 0x444D4163) {
        return CPU_VENDOR_AMD;
    }

    return CPU_VENDOR_UNKNOWN;
}

static void GetProcessorNameFromCpuid(char *name, size_t maxLen) {
    CpuIdRegs regs;
    char brandString[49] = {0}; // 48 chars + null terminator

    // Check if extended functions are available
    regs = cpuid(0x80000000);
    if (regs.eax < 0x80000004) {
        strncpy(name, "Unknown Processor", maxLen - 1);
        name[maxLen - 1] = '\0';
        return;
    }

    // Get brand string from CPUID 0x80000002-0x80000004
    regs = cpuid(0x80000002);
    memcpy(&brandString[0], &regs.eax, 4);
    memcpy(&brandString[4], &regs.ebx, 4);
    memcpy(&brandString[8], &regs.ecx, 4);
    memcpy(&brandString[12], &regs.edx, 4);

    regs = cpuid(0x80000003);
    memcpy(&brandString[16], &regs.eax, 4);
    memcpy(&brandString[20], &regs.ebx, 4);
    memcpy(&brandString[24], &regs.ecx, 4);
    memcpy(&brandString[28], &regs.edx, 4);

    regs = cpuid(0x80000004);
    memcpy(&brandString[32], &regs.eax, 4);
    memcpy(&brandString[36], &regs.ebx, 4);
    memcpy(&brandString[40], &regs.ecx, 4);
    memcpy(&brandString[44], &regs.edx, 4);

    // Trim leading spaces
    char *trimmed = brandString;
    while (*trimmed == ' ') trimmed++;

    strncpy(name, trimmed, maxLen - 1);
    name[maxLen - 1] = '\0';
}

static ProcessorsType DetermineProcessorType() {
    CpuVendor vendor = GetCpuVendor();
    CpuIdRegs regs = cpuid(1);

    int family = ((regs.eax >> 8) & 0xF) + ((regs.eax >> 20) & 0xFF);
    int model = ((regs.eax >> 4) & 0xF) + (((regs.eax >> 16) & 0xF) << 4);

    // Check for MMX support
    XBOOL hasMMX = (regs.edx & (1 << 23)) != 0;

    switch (vendor) {
    case CPU_VENDOR_INTEL:
        if (family == 5) {
            // Pentium family
            return hasMMX ? PROC_PENTIUMMMX : PROC_PENTIUM;
        } else if (family == 6) {
            if (model < 3) {
                return PROC_PENTIUMPRO;
            } else if (model < 7) {
                return PROC_PENTIUM2;
            } else if (model < 11) {
                return PROC_PENTIUM3;
            } else {
                return PROC_PENTIUM3; // Modern Intel cores
            }
        } else if (family == 15) {
            return PROC_PENTIUM4;
        }
        break;

    case CPU_VENDOR_AMD:
        if (family >= 6) {
            return PROC_ATHLON;
        }
        break;

    default:
        break;
    }

    return PROC_UNKNOWN;
}

static void DetectInstructionSets() {
    g_InstructionSetExtensions = ISEX_NONE;

    // Check basic feature support from CPUID function 1
    CpuIdRegs regs1 = cpuid(1);

    // SSE support (bit 25 in EDX)
    if (regs1.edx & (1 << 25)) {
        g_InstructionSetExtensions |= ISEX_SSE;
    }

    // SSE2 support (bit 26 in EDX)
    if (regs1.edx & (1 << 26)) {
        g_InstructionSetExtensions |= ISEX_SSE2;
    }

    // SSE3 support (bit 0 in ECX)
    if (regs1.ecx & (1 << 0)) {
        g_InstructionSetExtensions |= ISEX_SSE3;
    }

    // SSSE3 support (bit 9 in ECX)
    if (regs1.ecx & (1 << 9)) {
        g_InstructionSetExtensions |= ISEX_SSSE3;
    }

    // SSE4.1 support (bit 19 in ECX)
    if (regs1.ecx & (1 << 19)) {
        g_InstructionSetExtensions |= ISEX_SSE41;
    }

    // SSE4.2 support (bit 20 in ECX)
    if (regs1.ecx & (1 << 20)) {
        g_InstructionSetExtensions |= ISEX_SSE42;
    }

    // AVX support (bit 28 in ECX)
    if (regs1.ecx & (1 << 28)) {
        g_InstructionSetExtensions |= ISEX_AVX;
    }

    // Check extended features from CPUID function 7
    CpuIdRegs regs0 = cpuid(0);
    if (regs0.eax >= 7) {
        CpuIdRegs regs7 = cpuidex(7, 0);

        // AVX2 support (bit 5 in EBX)
        if (regs7.ebx & (1 << 5)) {
            g_InstructionSetExtensions |= ISEX_AVX2;
        }

        // Additional newer instruction sets can be added here
        // FMA3 support (bit 12 in ECX from CPUID 1)
        if (regs1.ecx & (1 << 12)) {
            g_InstructionSetExtensions |= ISEX_FMA3;
        }

        // BMI1 support (bit 3 in EBX)
        if (regs7.ebx & (1 << 3)) {
            g_InstructionSetExtensions |= ISEX_BMI1;
        }

        // BMI2 support (bit 8 in EBX)
        if (regs7.ebx & (1 << 8)) {
            g_InstructionSetExtensions |= ISEX_BMI2;
        }

        // AVX-512F support (bit 16 in EBX)
        if (regs7.ebx & (1 << 16)) {
            g_InstructionSetExtensions |= ISEX_AVX512F;
        }

        // AVX-512DQ support (bit 17 in EBX)
        if (regs7.ebx & (1 << 17)) {
            g_InstructionSetExtensions |= ISEX_AVX512DQ;
        }

        // AVX-512BW support (bit 30 in EBX)
        if (regs7.ebx & (1 << 30)) {
            g_InstructionSetExtensions |= ISEX_AVX512BW;
        }

        // AVX-512VL support (bit 31 in EBX)
        if (regs7.ebx & (1 << 31)) {
            g_InstructionSetExtensions |= ISEX_AVX512VL;
        }

        // AVX-512VNNI support (bit 11 in ECX)
        if (regs7.ecx & (1 << 11)) {
            g_InstructionSetExtensions |= ISEX_AVX512VNNI;
        }

        // AVXVNNI support (bit 4 in EAX from CPUID 7, subleaf 1)
        CpuIdRegs regs7_1 = cpuidex(7, 1);
        if (regs7_1.eax & (1 << 4)) {
            g_InstructionSetExtensions |= ISEX_AVXVNNI;
        }

        // AMX support (AMX-TILE bit 24 in EDX, AMX-INT8 bit 25 in EDX)
        if ((regs7.edx & (1 << 24)) && (regs7.edx & (1 << 25))) {
            g_InstructionSetExtensions |= ISEX_AMX;
        }
    }
}

void VxDetectProcessor() {
    static XBOOL ProcessorDetected = FALSE;
    if (ProcessorDetected)
        return;

    ::OutputDebugString("VxMath: Detecting processor------------------------\n");

    // Timing measurement (unchanged from original)
    HANDLE hThread = ::GetCurrentThread();
    int priority = ::GetThreadPriority(hThread);
    ::SetThreadPriority(hThread, THREAD_PRIORITY_IDLE);
    ::Sleep(10);
    static LARGE_INTEGER freq;
    ::QueryPerformanceFrequency(&freq);
    DWORD64 timeStamp1 = __rdtsc();
    LARGE_INTEGER freq1;
    ::QueryPerformanceCounter(&freq1);
    DWORD64 timeStamp2 = __rdtsc();
    {
        int x = 0, y = 0, z = 0;
        for (int i = 0; i < 500000; ++i) {
            x += ++y;
            z += x;
            x -= z;
        }
    }
    LARGE_INTEGER freq2;
    ::QueryPerformanceCounter(&freq2);
    DWORD64 timeStamp3 = __rdtsc();
    double t1 = (double) (freq2.QuadPart - freq1.QuadPart) / (double) freq.QuadPart;
    double t2 = (double) (timeStamp3 - timeStamp2);
    g_MSecondsPerCycle = (float) (1000.0 * t1 / t2);
    g_ProcessorFrequency = (int) (t2 / t1 / 1000000.0);
    ::SetThreadPriority(hThread, priority);

    // Determine processor type using CPUID
    g_ProcessorType = DetermineProcessorType();

    // Get processor name from CPUID
    GetProcessorNameFromCpuid(g_ProcDescription, sizeof(g_ProcDescription));

    // Determine processor features from CPUID function 1
    CpuIdRegs regs = cpuid(1);
    g_ProcessorFeatures = regs.edx; // Standard features

    // Detect instruction set extensions
    DetectInstructionSets();

    // Update processor description with frequency
    char tempDesc[64];
    strncpy(tempDesc, g_ProcDescription, sizeof(tempDesc) - 1);
    tempDesc[sizeof(tempDesc) - 1] = '\0';

    if (g_ProcessorFrequency < 1000) {
        _snprintf(g_ProcDescription, sizeof(g_ProcDescription) - 1, "%s %d MHz", tempDesc, g_ProcessorFrequency);
    } else {
        _snprintf(g_ProcDescription, sizeof(g_ProcDescription) - 1, "%s %.2f GHz", tempDesc,
                  g_ProcessorFrequency / 1000.0);
    }
    g_ProcDescription[sizeof(g_ProcDescription) - 1] = '\0';

    ProcessorDetected = TRUE;
}

char *GetProcessorDescription() {
    return g_ProcDescription;
}

int GetProcessorFrequency() {
    return g_ProcessorFrequency;
}

XULONG GetProcessorFeatures() {
    return g_ProcessorFeatures;
}

void ModifyProcessorFeatures(XULONG Add, XULONG Remove) {
    g_ProcessorFeatures = (g_ProcessorFeatures | Add) & ~Remove;
}

ProcessorsType GetProcessorType() {
    return g_ProcessorType;
}

XULONG GetInstructionSetExtensions() {
    return g_InstructionSetExtensions;
}
