#include "CKRasterizer.h"

CKRasterizerDriver::CKRasterizerDriver()
    : m_DisplayModes(),
      m_TextureFormats(),
      m_Desc(),
      m_Contexts() {
    m_Stereo = FALSE;
    m_Hardware = FALSE;
    m_CapsUpToDate = FALSE;
    m_Owner = NULL;
    m_DriverIndex = 0;
    memset(&m_3DCaps, 0, sizeof(m_3DCaps));
    memset(&m_2DCaps, 0, sizeof(m_2DCaps));
    m_Desc = "NULL Rasterizer";
}

CKRasterizerDriver::~CKRasterizerDriver() {}

CKRasterizerContext *CKRasterizerDriver::CreateContext() {
    CKRasterizerContext *context = new CKRasterizerContext();
    context->m_Driver = this;
    m_Contexts.PushBack(context);
    return context;
}

CKBOOL CKRasterizerDriver::DestroyContext(CKRasterizerContext *Context) {
    m_Contexts.Remove(Context);
    if (Context) delete Context;
    return TRUE;
}

void CKRasterizerDriver::InitNULLRasterizerCaps(CKRasterizer *Owner) {
    m_Owner = Owner;
    m_Desc = "NULL Rasterizer";
    m_CapsUpToDate = TRUE;
    m_Hardware = FALSE;
    m_DriverIndex = 0;

    m_DisplayModes.Resize(1);
    VxDisplayMode &dm = m_DisplayModes[0];
    dm.Bpp = 32;
    dm.Width = 640;
    dm.Height = 480;
    dm.RefreshRate = 0;

    m_TextureFormats.Resize(1);
    VxPixelFormat2ImageDesc(_32_ARGB8888, m_TextureFormats[0].Format);

    memset(&m_3DCaps, 0, sizeof(m_3DCaps));
    memset(&m_2DCaps, 0, sizeof(m_2DCaps));
    m_2DCaps.Caps = CKRST_2DCAPS_WINDOWED | CKRST_2DCAPS_3D | CKRST_2DCAPS_GDI;
}
