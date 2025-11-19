#ifndef VXMATHDEFINES_H
#define VXMATHDEFINES_H

#include "VxMathCompiler.h"

#include "VxMemory.h"

/// @brief Internal helper macro to stringify a token.
#define _QUOTE(x) #x
/// @brief Macro to stringify a token.
#define QUOTE(x) _QUOTE(x)

/// @brief Creates a string literal of the form "file(line) : ". Useful for compiler messages.
#define __FILE__LINE__ __FILE__ "(" QUOTE(__LINE__) ") : "

/// @brief Emits a message during compilation.
#define NOTE(x) message(x)
/// @brief Emits the current file and line number as a message during compilation.
#define FILE_LINE message(__FILE__LINE__)

/// @brief Emits a formatted "TODO" message during compilation, including file and line number.
#define TODO(x) message(__FILE__LINE__ "\n"               \
    " ------------------------------------------------\n" \
    "|  TODO :   " #x "\n"                                \
    " -------------------------------------------------\n")

/// @brief Emits a formatted "TOFIX" message during compilation, including file and line number.
#define TOFIX(x) message(__FILE__LINE__ "\n"              \
    " ------------------------------------------------\n" \
    "|  TOFIX :  " #x "\n"                                \
    " -------------------------------------------------\n")

/// @brief Emits a compact "TODO" message during compilation.
#define todo(x) message(__FILE__LINE__ " TODO :   " #x "\n")

/// @brief Emits a compact "TOFIX" message during compilation.
#define tofix(x) message(__FILE__LINE__ " TOFIX:   " #x "\n")

#ifndef EPSILON
/// @brief A small floating-point value used for comparisons.
#define EPSILON 1.192092896e-07F
#endif

#ifndef PI
/// @brief The mathematical constant Pi (π).
#define PI 3.1415926535f
#endif

#ifndef HALFPI
/// @brief Half of the mathematical constant Pi (π/2).
#define HALFPI 1.5707963267f
#endif

#ifndef HALF_RANDMAX
/// @brief Half of the maximum value returned by the standard rand() function (0x7FFF / 2).
#define HALF_RANDMAX 0x3fff
#endif

#ifndef INVHALF_RANDMAX
/// @brief The inverse of HALF_RANDMAX.
#define INVHALF_RANDMAX 6.10389e-005f
#endif

#ifndef INV_RANDMAX
/// @brief The inverse of RAND_MAX.
#define INV_RANDMAX 3.05185e-005f
#endif

/// @brief A pre-calculated constant (8192 / PI).
#define _8192ONPI 2607.594587617613f

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef NULL
#define NULL 0
#endif

typedef char *XSTRING;
typedef char XCHAR;
typedef int XBOOL;
typedef unsigned char XBYTE;
typedef unsigned short XWORD;
typedef unsigned int XDWORD;
typedef unsigned long XULONG;

/// @brief Generic function pointer type.
typedef int (VX_STDCALL *FUNC_PTR)();
/// @brief Generic handle for a window.
typedef void *WIN_HANDLE;
/// @brief Generic handle for an instance.
typedef void *INSTANCE_HANDLE;
/// @brief Generic handle.
typedef void *GENERIC_HANDLE;
/// @brief Handle for a bitmap resource.
typedef void *BITMAP_HANDLE;
/// @brief Handle for a font resource.
typedef void *FONT_HANDLE;

/**
 * @brief Structure representing a rectangle with integer coordinates.
 */
typedef struct CKRECT {
    int left;   ///< The x-coordinate of the upper-left corner.
    int top;    ///< The y-coordinate of the upper-left corner.
    int right;  ///< The x-coordinate of the lower-right corner.
    int bottom; ///< The y-coordinate of the lower-right corner.
} CKRECT;

/**
 * @brief Structure representing a point with 2D integer coordinates.
 */
typedef struct {
    int x, y; ///< The x and y coordinates of the point.
} CKPOINT;

// Forward declarations
class VxMatrix;
struct VxStridedData;

/**
 * @brief Base structure for storage of strided data.
 * @details This provides a pointer to data and the stride between consecutive elements.
 */
typedef struct VxStridedDataBase {
    union {
        void *Ptr;           ///< Generic pointer to the data.
        unsigned char *CPtr; ///< Character pointer to the data.
    };

    unsigned int Stride; ///< The byte offset between consecutive elements.

    /// @brief Conversion operator to a const VxStridedData reference.
    operator const VxStridedData &() const { return *(const VxStridedData *) this; }
} VxStridedDataBase;

/**
 * @brief Structure for storage of strided data.
 * @details This class extends VxStridedDataBase with constructors for convenience.
 */
typedef struct VxStridedData : public VxStridedDataBase {
    /// @brief Default constructor, initializes Ptr and Stride to 0.
    VxStridedData() {
        Ptr = NULL;
        Stride = 0;
    }

    /// @brief Constructor to initialize with a pointer and stride.
    /// @param iPtr Pointer to the data.
    /// @param iStride The byte offset between consecutive elements.
    VxStridedData(void *iPtr, unsigned int iStride) {
        Ptr = iPtr;
        Stride = iStride;
    }
} VxStridedData;

/**
 * @brief Enumerations of different processor types.
 *
 * @remarks The processor type and frequency are detected at startup of any program using the VxMath library.
 * @see GetProcessorType
 */
typedef enum ProcessorsType {
    PROC_UNKNOWN         = -1, ///< Processor type can not be detected
    PROC_PENTIUM         = 0,  ///< Standard Pentium Processor
    PROC_PENTIUMMMX      = 1,  ///< Standard Pentium Processor with MMX support
    PROC_PENTIUMPRO      = 2,  ///< Intel Pentium Pro Processor
    PROC_K63DNOW         = 3,  ///< AMD K6 + 3DNow ! support
    PROC_PENTIUM2        = 4,  ///< Intel Pentium II Processor Model 3
    PROC_PENTIUM2XEON    = 5,  ///< Intel Pentium II Processor Xeon or Celeron Model 5
    PROC_PENTIUM2CELERON = 6,  ///< Intel Pentium II Processor Celeron Model 6
    PROC_PENTIUM3        = 7,  ///< Intel Pentium III Processor
    PROC_ATHLON          = 9,  ///< AMD K7 Athlon Processor
    PROC_PENTIUM4        = 10, ///< Intel Pentium 4  Processor
    PROC_PPC_ARM         = 11, ///< Pocket PC ARM Device
    PROC_PPC_MIPS        = 12, ///< Pocket PC MIPS Device
    PROC_PPC_G3          = 13, ///< Power  PC G3
    PROC_PPC_G4          = 14, ///< Power  PC G4
    PROC_PSX2            = 15, ///< MIPS PSX2
    PROC_XBOX2           = 16, ///< XBOX2 CPU
    PROC_PSP             = 17  ///< PSP CPU
} ProcessorsType;

/// @brief The number of standard pixel formats.
#define NB_STDPIXEL_FORMATS 19
/// @brief The maximum number of supported pixel formats.
#define MAX_PIXEL_FORMATS 28

/**
 * @brief Operating systems enumeration.
 * @see VxGetOs
 */
typedef enum VX_OSINFO {
    VXOS_UNKNOWN,  ///< Unknown operating system
    VXOS_WIN31,    ///< Windows 3.1
    VXOS_WIN95,    ///< Windows 95
    VXOS_WIN98,    ///< Windows 98
    VXOS_WINME,    ///< Windows ME
    VXOS_WINNT4,   ///< Windows NT 4.0
    VXOS_WIN2K,    ///< Windows 2000
    VXOS_WINXP,    ///< Windows XP
    VXOS_MACOS9,   ///< Mac OS 9
    VXOS_MACOSX,   ///< Mac OS X
    VXOS_XBOX,     ///< Xbox
    VXOS_LINUXX86, ///< Linux on x86
    VXOS_WINCE1,   ///< Windows CE 1.0
    VXOS_WINCE2,   ///< Windows CE 2.0
    VXOS_WINCE3,   ///< Windows CE 3.0
    VXOS_PSX2,     ///< PlayStation 2
    VXOS_XBOX2,    ///< Xbox 2 (alias for Xbox 360)
    VXOS_WINVISTA, ///< Windows Vista
    VXOS_PSP,      ///< PlayStation Portable
    VXOS_XBOX360,  ///< Xbox 360
    VXOS_WII,      ///< Nintendo Wii
    VXOS_WINSEVEN  ///< Windows 7
} VX_OSINFO;

/**
 * @brief Platform enumeration.
 * @see VxGetPlatform
 */
typedef enum VX_PLATFORMINFO {
    VXPLATFORM_UNKNOWN = -1, ///< Unknown platform
    VXPLATFORM_WINDOWS = 0,  ///< Windows platform
    VXPLATFORM_MAC     = 1,  ///< Macintosh platform
    VXPLATFORM_XBOX    = 2,  ///< Xbox platform
    VXPLATFORM_WINCE   = 3,  ///< Windows CE platform
    VXPLATFORM_LINUX   = 4,  ///< Linux platform
    VXPLATFORM_PSX2    = 5,  ///< PlayStation 2 platform
    VXPLATFORM_XBOX2   = 6,  ///< Xbox 360 platform
    VXPLATFORM_PSP     = 7,  ///< PlayStation Portable platform
    VXPLATFORM_WII     = 8,  ///< Nintendo Wii platform
} VX_PLATFORMINFO;

/**
 * @brief Pixel format types.
 * @see VxImageDesc2PixelFormat, VxPixelFormat2ImageDesc
 */
typedef enum VX_PIXELFORMAT {
    UNKNOWN_PF   = 0,  ///< Unknown pixel format
    _32_ARGB8888 = 1,  ///< 32-bit ARGB pixel format with alpha
    _32_RGB888   = 2,  ///< 32-bit RGB pixel format without alpha
    _24_RGB888   = 3,  ///< 24-bit RGB pixel format
    _16_RGB565   = 4,  ///< 16-bit RGB pixel format
    _16_RGB555   = 5,  ///< 16-bit RGB pixel format (5 bits per color)
    _16_ARGB1555 = 6,  ///< 16-bit ARGB pixel format (5 bits per color + 1 bit for alpha)
    _16_ARGB4444 = 7,  ///< 16-bit ARGB pixel format (4 bits per color)
    _8_RGB332    = 8,  ///< 8-bit  RGB pixel format
    _8_ARGB2222  = 9,  ///< 8-bit  ARGB pixel format
    _32_ABGR8888 = 10, ///< 32-bit ABGR pixel format
    _32_RGBA8888 = 11, ///< 32-bit RGBA pixel format
    _32_BGRA8888 = 12, ///< 32-bit BGRA pixel format
    _32_BGR888   = 13, ///< 32-bit BGR pixel format
    _24_BGR888   = 14, ///< 24-bit BGR pixel format
    _16_BGR565   = 15, ///< 16-bit BGR pixel format
    _16_BGR555   = 16, ///< 16-bit BGR pixel format (5 bits per color)
    _16_ABGR1555 = 17, ///< 16-bit ABGR pixel format (5 bits per color + 1 bit for alpha)
    _16_ABGR4444 = 18, ///< 16-bit ABGR pixel format (4 bits per color)
    _DXT1        = 19, ///< S3/DirectX Texture Compression 1
    _DXT2        = 20, ///< S3/DirectX Texture Compression 2
    _DXT3        = 21, ///< S3/DirectX Texture Compression 3
    _DXT4        = 22, ///< S3/DirectX Texture Compression 4
    _DXT5        = 23, ///< S3/DirectX Texture Compression 5
    _16_V8U8     = 24, ///< 16-bit Bump Map format (8 bits per color)
    _32_V16U16   = 25, ///< 32-bit Bump Map format (16 bits per color)
    _16_L6V5U5   = 26, ///< 16-bit Bump Map format with luminance
    _32_X8L8V8U8 = 27, ///< 32-bit Bump Map format with luminance
} VX_PIXELFORMAT;

/**
 * @brief Vertex clipping flags.
 *
 * @remarks When using functions such as VxTransformBox2D which performs frustum tests, vertices are assigned
 * clipping flags to indicate in which part of the viewing frustum they are.
 * @see VxTransformBox2D
 */
typedef enum VXCLIP_FLAGS {
    VXCLIP_LEFT      = 0x00000010, ///< Vertex is clipped by the left viewing plane
    VXCLIP_RIGHT     = 0x00000020, ///< Vertex is clipped by the right viewing plane
    VXCLIP_TOP       = 0x00000040, ///< Vertex is clipped by the top viewing plane
    VXCLIP_BOTTOM    = 0x00000080, ///< Vertex is clipped by the bottom viewing plane
    VXCLIP_FRONT     = 0x00000100, ///< Vertex is clipped by the front viewing plane
    VXCLIP_BACK      = 0x00000200, ///< Vertex is clipped by the rear viewing plane
    VXCLIP_BACKFRONT = 0x00000300, ///< Combination of BACK and FRONT flags
    VXCLIP_ALL       = 0x000003F0, ///< All flags Combined
} VXCLIP_FLAGS;

/**
 * @brief Bounding box clipping flags.
 */
enum VXCLIP_BOXFLAGS {
    VXCLIP_BOXLEFT   = 0x01, ///< Box is at least partially to the left of the frustum.
    VXCLIP_BOXBOTTOM = 0x02, ///< Box is at least partially below the frustum.
    VXCLIP_BOXBACK   = 0x04, ///< Box is at least partially behind the frustum.
    VXCLIP_BOXRIGHT  = 0x08, ///< Box is at least partially to the right of the frustum.
    VXCLIP_BOXTOP    = 0x10, ///< Box is at least partially above the frustum.
    VXCLIP_BOXFRONT  = 0x20  ///< Box is at least partially in front of the frustum.
};

/**
 * @brief Enumerations of different processor features.
 *
 * @remarks The processor type and frequency are detected at startup of any program using the VxMath library.
 * @see GetProcessorFeatures, ModifyProcessorFeatures
 */
typedef enum ProcessorsFeatures {
    PROC_HASFPU    = 0x00000001, ///< Fpu on Chip
    PROC_V86       = 0x00000002, ///< Virtual 8086 Mode Extensions
    PROC_DE        = 0x00000004, ///< Debugging Extensions
    PROC_PSE       = 0x00000008, ///< Page size Extensions
    PROC_TIMESTAMP = 0x00000010, ///< Time stamp counter available
    PROC_MSR       = 0x00000020, ///< Model specific register
    PROC_PAE       = 0x00000040, ///< Physical Address Extension
    PROC_MCE       = 0x00000080, ///< Machine Check Exception
    PROC_CMPXCHG8B = 0x00000100, ///< CMPXCHG8B instruction available
    PROC_APIC      = 0x00000200, ///< APIC on Chip
    PROC_RESERVED  = 0x00000400, ///< Reserved
    PROC_SEP       = 0x00000800, ///< Fast System Call
    PROC_MTRR      = 0x00001000, ///< Memory Type Range Registers
    PROC_PGE       = 0x00002000, ///< Page Global Enable
    PROC_MCA       = 0x00004000, ///< Machine Check Architecture
    PROC_CMOV      = 0x00008000, ///< CMOVcc instructions
    PROC_PAT       = 0x00010000, ///< Page Attribute Table
    PROC_PST32     = 0x00020000, ///< 32-Bit Page Size Extension
    PROC_PN        = 0x00040000, ///< Processor Number
    PROC_MMX       = 0x00800000, ///< MMX instructions available
    PROC_FXSR      = 0x01000000, ///< Fast state save/restore
    PROC_SIMD      = 0x02000000, ///< SIMD instructions available (SSE)
    PROC_WNI       = 0x04000000, ///< Willamette new instructions available (SSE2)
    PROC_SS        = 0x08000000, ///< Self snoop
    PROC_HTT       = 0x10000000, ///< Hyper Threading Technology
    PROC_TM        = 0x20000000, ///< Thermal Monitoring
} ProcessorsFeatures;

typedef enum InstructionSetExtensions {
    ISEX_NONE        = 0x00000000, ///< No Extensions
    ISEX_SSE         = 0x00000001, ///< SSE (Streaming SIMD Extensions)
    ISEX_SSE2        = 0x00000002, ///< SSE2 (Streaming SIMD Extensions 2)
    ISEX_SSE3        = 0x00000004, ///< SSE3 (Streaming SIMD Extensions 3)
    ISEX_SSSE3       = 0x00000008, ///< SSSE3 (Supplemental Streaming SIMD Extensions 3)
    ISEX_SSE41       = 0x00000010, ///< SSE4.1 (Streaming SIMD Extensions 4.1)
    ISEX_SSE42       = 0x00000020, ///< SSE4.2 (Streaming SIMD Extensions 4.2)
    ISEX_AVX         = 0x00000040, ///< AVX (Advanced Vector Extensions)
    ISEX_AVX2        = 0x00000080, ///< AVX2 (Advanced Vector Extensions 2)
    ISEX_FMA3        = 0x00000100, ///< FMA3 (Fused Multiply-Add 3-operand)
    ISEX_BMI1        = 0x00000200, ///< BMI1 (Bit Manipulation Instruction Set 1)
    ISEX_BMI2        = 0x00000400, ///< BMI2 (Bit Manipulation Instruction Set 2)
    ISEX_AVX512F     = 0x00000800, ///< AVX-512 Foundation
    ISEX_AVX512DQ    = 0x00001000, ///< AVX-512 Doubleword and Quadword Instructions
    ISEX_AVX512BW    = 0x00002000, ///< AVX-512 Byte and Word Instructions
    ISEX_AVX512VL    = 0x00004000, ///< AVX-512 Vector Length Extensions
    ISEX_AVX512VNNI  = 0x00008000, ///< AVX-512 Vector Neural Network Instructions
    ISEX_AVXVNNI     = 0x00010000, ///< AVX Vector Neural Network Instructions
    ISEX_AMX         = 0x00020000, ///< AMX (Advanced Matrix Extensions)
} InstructionSetExtensions;

#endif // VXMATHDEFINES_H
