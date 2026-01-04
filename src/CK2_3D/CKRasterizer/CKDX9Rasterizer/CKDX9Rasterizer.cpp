#include "CKDX9Rasterizer.h"

PFN_D3DXDeclaratorFromFVF D3DXDeclaratorFromFVF = NULL;
PFN_D3DXFVFFromDeclarator D3DXFVFFromDeclarator = NULL;
PFN_D3DXAssembleShader D3DXAssembleShader = NULL;
PFN_D3DXDisassembleShader D3DXDisassembleShader = NULL;
PFN_D3DXLoadSurfaceFromSurface D3DXLoadSurfaceFromSurface = NULL;
PFN_D3DXLoadSurfaceFromMemory D3DXLoadSurfaceFromMemory = NULL;
PFN_D3DXCreateTextureFromFileExA D3DXCreateTextureFromFileExA = NULL;

D3DFORMAT VxPixelFormatToD3DFormat(VX_PIXELFORMAT pf)
{
    switch (pf)
    {
        case _32_ARGB8888: return D3DFMT_A8R8G8B8; // 32-bit ARGB pixel format with alpha
        case _32_RGB888: return D3DFMT_X8R8G8B8; // 32-bit RGB pixel format without alpha
        case _24_RGB888: return D3DFMT_R8G8B8; // 24-bit RGB pixel format
        case _16_RGB565: return D3DFMT_R5G6B5; // 16-bit RGB pixel format
        case _16_RGB555: return D3DFMT_X1R5G5B5; // 16-bit RGB pixel format (5 bits per color)
        case _16_ARGB1555: return D3DFMT_A1R5G5B5; // 16-bit ARGB pixel format (5 bits per color + 1 bit for alpha)
        case _16_ARGB4444: return D3DFMT_A4R4G4B4; // 16-bit ARGB pixel format (4 bits per color)
        case _8_RGB332: return D3DFMT_R3G3B2; // 8-bit  RGB pixel format
        case _8_ARGB2222: return D3DFMT_UNKNOWN; // 8-bit  ARGB pixel format
        case _32_ABGR8888: return D3DFMT_A8B8G8R8; // 32-bit ABGR pixel format
        case _32_RGBA8888: return D3DFMT_UNKNOWN; // 32-bit RGBA pixel format
        case _32_BGRA8888: return D3DFMT_UNKNOWN; // 32-bit BGRA pixel format
        case _32_BGR888: return D3DFMT_X8B8G8R8; // 32-bit BGR pixel format
        case _24_BGR888: return D3DFMT_UNKNOWN; // 24-bit BGR pixel format
        case _16_BGR565: return D3DFMT_UNKNOWN; // 16-bit BGR pixel format
        case _16_BGR555: return D3DFMT_UNKNOWN; // 16-bit BGR pixel format (5 bits per color)
        case _16_ABGR1555: return D3DFMT_UNKNOWN; // 16-bit ABGR pixel format (5 bits per color + 1 bit for alpha)
        case _16_ABGR4444: return D3DFMT_UNKNOWN; // 16-bit ABGR pixel format (4 bits per color)
        case _DXT1: return D3DFMT_DXT1; // S3/DirectX Texture Compression 1
        case _DXT2: return D3DFMT_DXT2; // S3/DirectX Texture Compression 2
        case _DXT3: return D3DFMT_DXT3; // S3/DirectX Texture Compression 3
        case _DXT4: return D3DFMT_DXT4; // S3/DirectX Texture Compression 4
        case _DXT5: return D3DFMT_DXT5; // S3/DirectX Texture Compression 5
        case _16_V8U8: return D3DFMT_V8U8; // 16-bit Bump Map format (8 bits per color)
        case _32_V16U16: return D3DFMT_V16U16; // 32-bit Bump Map format (16 bits per color)
        case _16_L6V5U5: return D3DFMT_L6V5U5; // 16-bit Bump Map format with luminance
        case _32_X8L8V8U8: return D3DFMT_X8L8V8U8; // 32-bit Bump Map format with luminance
        default: return D3DFMT_UNKNOWN;
    }
}

VX_PIXELFORMAT D3DFormatToVxPixelFormat(D3DFORMAT ddpf)
{
    switch (ddpf)
    {
        case D3DFMT_A8R8G8B8: return _32_ARGB8888; // 32-bit ARGB pixel format with alpha
        case D3DFMT_X8R8G8B8: return _32_RGB888; // 32-bit RGB pixel format without alpha
        case D3DFMT_R8G8B8: return _24_RGB888; // 24-bit RGB pixel format
        case D3DFMT_R5G6B5: return _16_RGB565; // 16-bit RGB pixel format
        case D3DFMT_X1R5G5B5: return _16_RGB555; // 16-bit RGB pixel format (5 bits per color)
        case D3DFMT_A1R5G5B5: return _16_ARGB1555; // 16-bit ARGB pixel format (5 bits per color + 1 bit for alpha)
        case D3DFMT_A4R4G4B4: return _16_ARGB4444; // 16-bit ARGB pixel format (4 bits per color)
        case D3DFMT_R3G3B2: return _8_RGB332; // 8-bit  RGB pixel format
        case D3DFMT_UNKNOWN: return _8_ARGB2222; // 8-bit  ARGB pixel format
        case D3DFMT_A8B8G8R8: return _32_ABGR8888; // 32-bit ABGR pixel format
        case D3DFMT_X8B8G8R8: return _32_BGR888; // 32-bit BGR pixel format
        case D3DFMT_DXT1: return _DXT1; // S3/DirectX Texture Compression 1
        case D3DFMT_DXT2: return _DXT2; // S3/DirectX Texture Compression 2
        case D3DFMT_DXT3: return _DXT3; // S3/DirectX Texture Compression 3
        case D3DFMT_DXT4: return _DXT4; // S3/DirectX Texture Compression 4
        case D3DFMT_DXT5: return _DXT5; // S3/DirectX Texture Compression 5
        case D3DFMT_V8U8: return _16_V8U8; // 16-bit Bump Map format (8 bits per color)
        case D3DFMT_V16U16: return _32_V16U16; // 32-bit Bump Map format (16 bits per color)
        case D3DFMT_L6V5U5: return _16_L6V5U5; // 16-bit Bump Map format with luminance
        case D3DFMT_X8L8V8U8: return _32_X8L8V8U8; // 32-bit Bump Map format with luminance
        default: return UNKNOWN_PF;
    }
}

D3DFORMAT TextureDescToD3DFormat(CKTextureDesc *desc)
{
    if (!desc)
        return D3DFMT_UNKNOWN;
    return VxPixelFormatToD3DFormat(VxImageDesc2PixelFormat(desc->Format));
}

void D3DFormatToTextureDesc(D3DFORMAT ddpf, CKTextureDesc *desc)
{
    if (!desc)
        return;

    // Set appropriate flags based on pixel format
    desc->Flags = CKRST_TEXTURE_VALID;

    // Add RGB flag for all formats except pure alpha
    desc->Flags |= CKRST_TEXTURE_RGB;

    // Add alpha flag for formats that support alpha
    switch (ddpf)
    {
        case D3DFMT_A8R8G8B8:
        case D3DFMT_A1R5G5B5:
        case D3DFMT_A4R4G4B4:
        case D3DFMT_A8B8G8R8:
        case D3DFMT_DXT1: // DXT1 has 1-bit alpha in some variations
        case D3DFMT_DXT2:
        case D3DFMT_DXT3:
        case D3DFMT_DXT4:
        case D3DFMT_DXT5:
            desc->Flags |= CKRST_TEXTURE_ALPHA;
            break;
    }

    // Handle compressed texture formats
    if (ddpf == D3DFMT_DXT1 ||
        ddpf == D3DFMT_DXT2 ||
        ddpf == D3DFMT_DXT3 ||
        ddpf == D3DFMT_DXT4 ||
        ddpf == D3DFMT_DXT5)
    {
        desc->Flags |= CKRST_TEXTURE_COMPRESSION;
    }

    VX_PIXELFORMAT vxpf = D3DFormatToVxPixelFormat(ddpf);
    VxPixelFormat2ImageDesc(vxpf, desc->Format);
}

CKRasterizer *CKDX9RasterizerStart(WIN_HANDLE AppWnd)
{
    HMODULE handle = LoadLibraryA("d3d9.dll");
    if (handle)
    {
        CKRasterizer *rasterizer = new CKDX9Rasterizer;
        if (!rasterizer)
            return NULL;
        if (!rasterizer->Start(AppWnd))
        {
            delete rasterizer;
            FreeLibrary(handle);
            return NULL;
        }
        return rasterizer;
    }
    return NULL;
}

void CKDX9RasterizerClose(CKRasterizer *rst)
{
    if (rst)
    {
        rst->Close();
        delete rst;
    }
}

PLUGIN_EXPORT void CKRasterizerGetInfo(CKRasterizerInfo *info)
{
    info->StartFct = CKDX9RasterizerStart;
    info->CloseFct = CKDX9RasterizerClose;
    info->Desc = "DirectX 9 Rasterizer";
}

CKDX9Rasterizer::CKDX9Rasterizer() : m_D3D9(NULL), m_Init(FALSE), m_BlendStages() {}

CKDX9Rasterizer::~CKDX9Rasterizer()
{
    Close();
}

XBOOL CKDX9Rasterizer::Start(WIN_HANDLE AppWnd)
{
    InitBlendStages();
    m_MainWindow = AppWnd;
    m_Init = TRUE;

    // Create the D3D object, which is needed to create the D3DDevice.
#ifdef USE_D3D9EX
    HRESULT hr = Direct3DCreate9Ex(D3D_SDK_VERSION, &m_D3D9);
    if (FAILED(hr))
    {
        m_D3D9 = NULL;
        return FALSE;
    }
#else
    m_D3D9 = Direct3DCreate9(D3D_SDK_VERSION);
    if (!m_D3D9)
        return FALSE;
#endif

	// Load D3DX
	if (!D3DXDeclaratorFromFVF ||
        !D3DXFVFFromDeclarator ||
        !D3DXAssembleShader ||
        !D3DXDisassembleShader ||
        !D3DXLoadSurfaceFromSurface ||
        !D3DXLoadSurfaceFromMemory ||
        !D3DXCreateTextureFromFileExA)
	{
        HMODULE module = NULL;
        const TCHAR *d3dxVersions[] = {
            TEXT("d3dx9_43.dll"), TEXT("d3dx9_42.dll"), TEXT("d3dx9_41.dll"),
            TEXT("d3dx9_40.dll"), TEXT("d3dx9_39.dll"), TEXT("d3dx9_38.dll"),
        };

        for (int i = 0; i < sizeof(d3dxVersions) / sizeof(d3dxVersions[0]); ++i)
        {
            module = LoadLibrary(d3dxVersions[i]);
            if (module)
                break;
        }

        if (module)
        {
            D3DXDeclaratorFromFVF = reinterpret_cast<PFN_D3DXDeclaratorFromFVF>(GetProcAddress(module, "D3DXDeclaratorFromFVF"));
            D3DXFVFFromDeclarator = reinterpret_cast<PFN_D3DXFVFFromDeclarator>(GetProcAddress(module, "D3DXFVFFromDeclarator"));
            D3DXAssembleShader = reinterpret_cast<PFN_D3DXAssembleShader>(GetProcAddress(module, "D3DXAssembleShader"));
            D3DXDisassembleShader = reinterpret_cast<PFN_D3DXDisassembleShader>(GetProcAddress(module, "D3DXDisassembleShader"));
            D3DXLoadSurfaceFromSurface = reinterpret_cast<PFN_D3DXLoadSurfaceFromSurface>(GetProcAddress(module, "D3DXLoadSurfaceFromSurface"));
            D3DXLoadSurfaceFromMemory = reinterpret_cast<PFN_D3DXLoadSurfaceFromMemory>(GetProcAddress(module, "D3DXLoadSurfaceFromMemory"));
            D3DXCreateTextureFromFileExA = reinterpret_cast<PFN_D3DXCreateTextureFromFileExA>(GetProcAddress(module, "D3DXCreateTextureFromFileExA"));
        }
    }

    UINT count = m_D3D9->GetAdapterCount();
    for (UINT i = 0; i < count; ++i)
    {
        CKDX9RasterizerDriver *driver = new CKDX9RasterizerDriver(this);
        if (!driver->InitializeCaps(i, D3DDEVTYPE_HAL))
            delete driver;
        else
            m_Drivers.PushBack(driver);
    }

    return TRUE;
}

void CKDX9Rasterizer::Close()
{
    if (!m_Init)
        return;

    // Clean up blend stages
    for (int i = 0; i < 256; i++)
    {
        if (m_BlendStages[i])
        {
            delete m_BlendStages[i];
            m_BlendStages[i] = NULL;
        }
    }

    if (m_D3D9)
    {
        m_D3D9->Release();
        m_D3D9 = NULL;
    }

    while (m_Drivers.Size() != 0)
    {
        CKRasterizerDriver *driver = m_Drivers.PopBack();
        delete driver;
    }

    m_Init = FALSE;
}

void CKDX9Rasterizer::InitBlendStages()
{
    memset(m_BlendStages, NULL, sizeof(m_BlendStages));

    // Modulate (ZERO, SRCCOLOR and DESTCOLOR, ZERO)
    CreateBlendStage(VXBLEND_ZERO, VXBLEND_SRCCOLOR, 
                     D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_SELECTARG1, D3DTA_CURRENT, D3DTA_CURRENT);
            
    CreateBlendStage(VXBLEND_DESTCOLOR, VXBLEND_ZERO, 
                     D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_SELECTARG1, D3DTA_CURRENT, D3DTA_CURRENT);

    // Additive (ONE, ONE)
    CreateBlendStage(VXBLEND_ONE, VXBLEND_ONE, 
                     D3DTOP_ADD, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_SELECTARG1, D3DTA_CURRENT, D3DTA_CURRENT);

    // Replace (ONE, ZERO)
    CreateBlendStage(VXBLEND_ONE, VXBLEND_ZERO, 
                     D3DTOP_SELECTARG1, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_SELECTARG1, D3DTA_TEXTURE, D3DTA_CURRENT);

    // Alpha blend (SRCALPHA, INVSRCALPHA)
    CreateBlendStage(VXBLEND_SRCALPHA, VXBLEND_INVSRCALPHA, 
                     D3DTOP_BLENDTEXTUREALPHA, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_SELECTARG1, D3DTA_TEXTURE, D3DTA_CURRENT);

    // Premultiplied alpha (ONE, INVSRCALPHA)
    CreateBlendStage(VXBLEND_ONE, VXBLEND_INVSRCALPHA, 
                     D3DTOP_BLENDTEXTUREALPHA, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_SELECTARG1, D3DTA_TEXTURE, D3DTA_CURRENT);

    // Multiply (ZERO, INVSRCCOLOR)
    CreateBlendStage(VXBLEND_ZERO, VXBLEND_INVSRCCOLOR, 
                     D3DTOP_MODULATEINVALPHA_ADDCOLOR, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_SELECTARG1, D3DTA_CURRENT, D3DTA_CURRENT);

    // Additive alpha (SRCALPHA, ONE)
    CreateBlendStage(VXBLEND_SRCALPHA, VXBLEND_ONE, 
                     D3DTOP_BLENDTEXTUREALPHA, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_SELECTARG1, D3DTA_TEXTURE, D3DTA_CURRENT);

    // Modulate 2X (double brightness)
    CreateBlendStage(VXBLEND_DESTCOLOR, VXBLEND_SRCCOLOR,
                     D3DTOP_MODULATE2X, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_MODULATE2X, D3DTA_TEXTURE, D3DTA_CURRENT);

    // Modulate 4X (quadruple brightness)
    CreateBlendStage(VXBLEND_DESTCOLOR, VXBLEND_SRCALPHA,
                     D3DTOP_MODULATE4X, D3DTA_TEXTURE, D3DTA_CURRENT,
                     D3DTOP_MODULATE4X, D3DTA_TEXTURE, D3DTA_CURRENT);

    // Subtract
    CreateBlendStage(VXBLEND_INVSRCCOLOR, VXBLEND_SRCCOLOR,
                     D3DTOP_SUBTRACT, D3DTA_CURRENT, D3DTA_TEXTURE,
                     D3DTOP_SUBTRACT, D3DTA_CURRENT, D3DTA_TEXTURE);
}

void CKDX9Rasterizer::CreateBlendStage(VXBLEND_MODE srcBlend, VXBLEND_MODE destBlend,
                                       D3DTEXTUREOP colorOp, DWORD colorArg1, DWORD colorArg2,
                                       D3DTEXTUREOP alphaOp, DWORD alphaArg1, DWORD alphaArg2)
{
    CKStageBlend *b = new CKStageBlend;
    b->Cop = colorOp;
    b->Carg1 = colorArg1;
    b->Carg2 = colorArg2;
    b->Aop = alphaOp;
    b->Aarg1 = alphaArg1;
    b->Aarg2 = alphaArg2;
    m_BlendStages[STAGEBLEND(srcBlend, destBlend)] = b;
}