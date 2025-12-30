#include "CKRasterizer.h"

CKDWORD GetMsb(CKDWORD data, CKDWORD index) {
#define OPERAND_SIZE (sizeof(CKDWORD) * 8)
    CKDWORD i = OPERAND_SIZE - 1;
    if (data != 0) {
        while (!(data & (1 << (OPERAND_SIZE - 1)))) {
            data <<= 1;
            --i;
        }
    } else {
        i = 0;
    }
    return (i > index) ? index : i;
#undef OPERAND_SIZE
}

CKDWORD GetLsb(CKDWORD data, CKDWORD index) {
    CKDWORD i = 0;
    if (data != 0) {
        while (!(data & 1)) {
            data >>= 1;
            ++i;
        }
    }
    return (i > index) ? index : i;
}

CKRasterizerContext::CKRasterizerContext()
    : m_Textures(),
      m_Sprites(),
      m_VertexBuffers(),
      m_IndexBuffers(),
      m_VertexShaders(),
      m_PixelFormat(),
      m_CurrentMaterialData(),
      m_CurrentLightData(),
      m_DirtyRects() {
    m_Driver = NULL;
    m_PosX = 0;
    m_PosY = 0;
    m_Width = 0;
    m_Height = 0;
    m_Window = NULL;
    m_Fullscreen = 0;
    m_RefreshRate = 0;
    m_SceneBegined = FALSE;
    m_MatrixUptodate = 0;
    m_TransparentMode = 0;
    m_Bpp = 0;
    m_ZBpp = 0;
    m_PixelFormat = UNKNOWN_PF;
    m_StencilBpp = 0;

    m_TotalMatrix = VxMatrix::Identity();
    m_WorldMatrix = VxMatrix::Identity();
    m_ViewMatrix = VxMatrix::Identity();
    m_ProjectionMatrix = VxMatrix::Identity();

    m_Textures.Resize(INIT_OBJECTSLOTS);
    m_Sprites.Resize(INIT_OBJECTSLOTS);
    m_VertexBuffers.Resize(INIT_OBJECTSLOTS);
    m_IndexBuffers.Resize(INIT_OBJECTSLOTS);
    m_VertexShaders.Resize(INIT_OBJECTSLOTS);
    m_PixelShaders.Resize(INIT_OBJECTSLOTS);

    m_Textures.Memset(NULL);
    m_Sprites.Memset(NULL);
    m_VertexBuffers.Memset(NULL);
    m_IndexBuffers.Memset(NULL);
    m_VertexShaders.Memset(NULL);
    m_PixelShaders.Memset(NULL);

    m_PresentInterval = 0;
    m_CurrentPresentInterval = 0;
    m_Antialias = 0;
    m_EnableScreenDump = 0;

    memset(m_StateCache, 0, sizeof(m_StateCache));
    InitDefaultRenderStatesValue();
    FlushRenderStateCache();
    m_RenderStateCacheMiss = 0;
    m_RenderStateCacheHit = 0;

    m_InverseWinding = 0;
    m_EnsureVertexShader = 0;
    m_UnityMatrixMask = 0;
}

CKRasterizerContext::~CKRasterizerContext() {}

CKBOOL CKRasterizerContext::SetMaterial(CKMaterialData *mat) {
    if (mat)
        memcpy(&m_CurrentMaterialData, mat, sizeof(m_CurrentMaterialData));
    return FALSE;
}

CKBOOL CKRasterizerContext::SetViewport(CKViewportData *data) {
    memcpy(&m_ViewportData, data, sizeof(m_ViewportData));
    return TRUE;
}

CKBOOL CKRasterizerContext::SetTransformMatrix(VXMATRIX_TYPE Type, const VxMatrix &Mat) {
    switch (Type) {
    case VXMATRIX_WORLD:
        memcpy(&m_WorldMatrix, Mat, sizeof(m_WorldMatrix));
        Vx3DMultiplyMatrix(m_ModelViewMatrix, m_ViewMatrix, m_WorldMatrix);
        m_MatrixUptodate &= 0xFE;
        break;
    case VXMATRIX_VIEW:
        memcpy(&m_ViewMatrix, Mat, sizeof(m_ViewMatrix));
        Vx3DMultiplyMatrix(m_ModelViewMatrix, m_ViewMatrix, m_WorldMatrix);
        m_MatrixUptodate = 0;
        break;
    case VXMATRIX_PROJECTION:
        memcpy(&m_ProjectionMatrix, Mat, sizeof(m_ProjectionMatrix));
        m_MatrixUptodate = 0;
        break;
    default:
        break;
    }
    return TRUE;
}

CKBOOL CKRasterizerContext::DeleteObject(CKDWORD ObjIndex, CKRST_OBJECTTYPE Type) {
    if (ObjIndex >= (CKDWORD) m_Textures.Size())
        return FALSE;

    switch (Type) {
    case CKRST_OBJ_TEXTURE:
        if (m_Textures[ObjIndex])
            delete m_Textures[ObjIndex];
        m_Textures[ObjIndex] = NULL;
        break;
    case CKRST_OBJ_SPRITE:
        if (m_Sprites[ObjIndex])
            delete m_Sprites[ObjIndex];
        m_Sprites[ObjIndex] = NULL;
        break;
    case CKRST_OBJ_VERTEXBUFFER:
        if (m_VertexBuffers[ObjIndex])
            delete m_VertexBuffers[ObjIndex];
        m_VertexBuffers[ObjIndex] = NULL;
        break;
    case CKRST_OBJ_INDEXBUFFER:
        if (m_IndexBuffers[ObjIndex])
            delete m_IndexBuffers[ObjIndex];
        m_IndexBuffers[ObjIndex] = NULL;
        break;
    case CKRST_OBJ_VERTEXSHADER:
        if (m_VertexShaders[ObjIndex])
            delete m_VertexShaders[ObjIndex];
        m_VertexShaders[ObjIndex] = NULL;
        break;
    case CKRST_OBJ_PIXELSHADER:
        if (m_PixelShaders[ObjIndex])
            delete m_PixelShaders[ObjIndex];
        m_PixelShaders[ObjIndex] = NULL;
        break;
    default:
        return FALSE;
    }
    return TRUE;
}

CKBOOL CKRasterizerContext::FlushObjects(CKDWORD TypeMask) {
    if ((TypeMask & CKRST_OBJ_TEXTURE) != 0)
        for (XArray<CKTextureDesc *>::Iterator it = m_Textures.Begin(); it != m_Textures.End(); ++it) {
            if (*it)
                delete *it;
            *it = NULL;
        }

    if ((TypeMask & CKRST_OBJ_SPRITE) != 0)
        for (XArray<CKSpriteDesc *>::Iterator it = m_Sprites.Begin(); it != m_Sprites.End(); ++it) {
            if (*it)
                delete *it;
            *it = NULL;
        }

    if ((TypeMask & CKRST_OBJ_VERTEXBUFFER) != 0)
        for (XArray<CKVertexBufferDesc *>::Iterator it = m_VertexBuffers.Begin(); it != m_VertexBuffers.End(); ++it) {
            if (*it)
                delete *it;
            *it = NULL;
        }

    if ((TypeMask & CKRST_OBJ_INDEXBUFFER) != 0)
        for (XArray<CKIndexBufferDesc *>::Iterator it = m_IndexBuffers.Begin(); it != m_IndexBuffers.End(); ++it) {
            if (*it)
                delete *it;
            *it = NULL;
        }

    if ((TypeMask & CKRST_OBJ_VERTEXSHADER) != 0)
        for (XArray<CKVertexShaderDesc *>::Iterator it = m_VertexShaders.Begin(); it != m_VertexShaders.End(); ++it) {
            if (*it)
                delete *it;
            *it = NULL;
        }

    if ((TypeMask & CKRST_OBJ_PIXELSHADER) != 0)
        for (XArray<CKPixelShaderDesc *>::Iterator it = m_PixelShaders.Begin(); it != m_PixelShaders.End(); ++it) {
            if (*it)
                delete *it;
            *it = NULL;
        }

    return TRUE;
}

void CKRasterizerContext::UpdateObjectArrays(CKRasterizer *rst) {
    int newSize = rst->m_ObjectsIndex.Size();
    int oldSize = m_Textures.Size();
    if (newSize != oldSize) {
        m_Textures.Resize(newSize);
        m_Sprites.Resize(newSize);
        m_VertexBuffers.Resize(newSize);
        m_IndexBuffers.Resize(newSize);
        m_VertexShaders.Resize(newSize);
        m_PixelShaders.Resize(newSize);

        int gapSize = (newSize - oldSize) * sizeof(void *);
        memset(&m_Textures[oldSize], 0, gapSize);
        memset(&m_Sprites[oldSize], 0, gapSize);
        memset(&m_VertexBuffers[oldSize], 0, gapSize);
        memset(&m_IndexBuffers[oldSize], 0, gapSize);
        memset(&m_VertexShaders[oldSize], 0, gapSize);
        memset(&m_PixelShaders[oldSize], 0, gapSize);
    }
}

CKTextureDesc *CKRasterizerContext::GetTextureData(CKDWORD Texture) {
    if (Texture >= (CKDWORD) m_Textures.Size())
        return NULL;
    CKTextureDesc *data = m_Textures[Texture];
    if (!data)
        return NULL;
    if ((data->Flags & CKRST_TEXTURE_VALID) == 0)
        return NULL;
    return data;
}

CKBOOL CKRasterizerContext::LoadSprite(CKDWORD Sprite, const VxImageDescEx &SurfDesc) {
    if (Sprite >= (CKDWORD) m_Sprites.Size())
        return FALSE;

    CKSpriteDesc *sprite = m_Sprites[Sprite];
    if (!sprite)
        return FALSE;

    if (sprite->Textures.Size() == 0)
        return FALSE;

    VxImageDescEx surface = SurfDesc;
    int bytesPerPixel = SurfDesc.BitsPerPixel / 8;

    CKBYTE *image = m_Driver->m_Owner->AllocateObjects(sprite->Textures[0].sw * sprite->Textures[0].sh);
    if (!image)
        return FALSE;
    surface.Image = image;

    for (XArray<CKSPRTextInfo>::Iterator it = sprite->Textures.Begin(); it != sprite->Textures.End(); ++it) {
        int spriteBytesPerLine = it->w * bytesPerPixel;
        int textureBytesPerLine = it->sw * bytesPerPixel;

        if (it->w != it->sw || it->h != it->sh)
            memset(image, 0, it->sh * textureBytesPerLine);

        XBYTE *src = &SurfDesc.Image[it->y * SurfDesc.BytesPerLine + it->x * bytesPerPixel];
        for (int h = 0; h < it->h; ++h) {
            memcpy(image, src, spriteBytesPerLine);
            image += textureBytesPerLine;
            src += SurfDesc.BytesPerLine;
        }

        surface.BytesPerLine = textureBytesPerLine;
        surface.Width = it->sw;
        surface.Height = it->sh;
        LoadTexture(it->IndexTexture, surface);
    }

    return TRUE;
}

CKSpriteDesc *CKRasterizerContext::GetSpriteData(CKDWORD Sprite) {
    if (Sprite >= (CKDWORD) m_Sprites.Size())
        return NULL;
    CKSpriteDesc *data = m_Sprites[Sprite];
    if (!data)
        return NULL;
    if ((data->Flags & CKRST_TEXTURE_VALID) == 0)
        return NULL;
    return data;
}

CKVertexBufferDesc *CKRasterizerContext::GetVertexBufferData(CKDWORD VB) {
    if (VB >= (CKDWORD) m_VertexBuffers.Size())
        return NULL;
    CKVertexBufferDesc *data = m_VertexBuffers[VB];
    if (!data)
        return NULL;
    if ((data->m_Flags & CKRST_VB_VALID) == 0)
        return NULL;
    return data;
}

CKBOOL CKRasterizerContext::TransformVertices(int VertexCount, VxTransformData *Data) {
    if (!Data->InVertices)
        return FALSE;

    unsigned int offscreen = 0;
    UpdateMatrices(WORLD_TRANSFORM);

    VxVector4 *outVertices = (VxVector4 *) Data->OutVertices;
    unsigned int outStride = Data->OutStride;
    if (!outVertices) {
        outVertices = (VxVector4 *) m_Driver->m_Owner->AllocateObjects(
            VertexCount * (sizeof(VxVector4) / sizeof(CKDWORD)));
        outStride = sizeof(VxVector4);
    }

    VxStridedData out(outVertices, outStride);
    VxStridedData in(Data->InVertices, Data->InStride);
    Vx3DMultiplyMatrixVector4Strided(&out, &in, m_TotalMatrix, VertexCount);

    if (Data->ClipFlags) {
        offscreen = 0xFFFFFFFF;
        for (int v = 0; v < VertexCount; ++v) {
            unsigned int clipFlag = 0;

            float w = outVertices->w;
            if (-w > outVertices->x)
                clipFlag |= VXCLIP_LEFT;
            if (outVertices->x > w)
                clipFlag |= VXCLIP_RIGHT;
            if (-w > outVertices->y)
                clipFlag |= VXCLIP_BOTTOM;
            if (outVertices->y > w)
                clipFlag |= VXCLIP_TOP;
            if (outVertices->z < 0.0f)
                clipFlag |= VXCLIP_FRONT;
            if (outVertices->z > w)
                clipFlag |= VXCLIP_BACK;

            offscreen &= clipFlag;
            Data->ClipFlags[v] = clipFlag;
            outVertices += outStride;
        }
    }

    VxVector4 *screenVertices = (VxVector4 *) Data->ScreenVertices;
    if (screenVertices) {
        float halfWidth = m_ViewportData.ViewWidth * 0.5f;
        float halfHeight = m_ViewportData.ViewHeight * 0.5f;
        float centerX = m_ViewportData.ViewX + halfWidth;
        float centerY = m_ViewportData.ViewY + halfHeight;
        for (int v = 0; v < VertexCount; ++v) {
            float w = 1.0f / outVertices->w;
            screenVertices->w = w;
            screenVertices->z = w * outVertices->z;
            screenVertices->y = centerY - outVertices->y * w * halfHeight;
            screenVertices->x = centerX + outVertices->x * w * halfWidth;
            screenVertices += Data->ScreenStride;
        }
    }

    Data->m_Offscreen = offscreen & VXCLIP_ALL;
    return TRUE;
}

CKDWORD CKRasterizerContext::ComputeBoxVisibility(const VxBbox &box, CKBOOL World, VxRect *extents) {
    UpdateMatrices(World ? VIEW_TRANSFORM : WORLD_TRANSFORM);

    VXCLIP_FLAGS orClipFlags, andClipFlags;
    if (extents) {
        VxRect screen(
            (float) m_ViewportData.ViewX,
            (float) m_ViewportData.ViewY,
            (float) (m_ViewportData.ViewX + m_ViewportData.ViewWidth),
            (float) (m_ViewportData.ViewY + m_ViewportData.ViewHeight));
        if (World)
            VxTransformBox2D(m_ViewProjMatrix, box, &screen, extents, orClipFlags, andClipFlags);
        else
            VxTransformBox2D(m_TotalMatrix, box, &screen, extents, orClipFlags, andClipFlags);
    } else {
        if (World)
            VxTransformBox2D(m_ViewProjMatrix, box, NULL, NULL, orClipFlags, andClipFlags);
        else
            VxTransformBox2D(m_TotalMatrix, box, NULL, NULL, orClipFlags, andClipFlags);
    }

    if (andClipFlags & VXCLIP_ALL)
        return CBV_OFFSCREEN;
    else if (orClipFlags & VXCLIP_ALL)
        return CBV_VISIBLE;
    else
        return CBV_ALLINSIDE;
}

void CKRasterizerContext::InitDefaultRenderStatesValue() {
    m_StateCache[VXRENDERSTATE_SHADEMODE].DefaultValue = 2;
    m_StateCache[VXRENDERSTATE_SRCBLEND].DefaultValue = 2;
    m_StateCache[VXRENDERSTATE_ALPHAFUNC].DefaultValue = 8;
    m_StateCache[VXRENDERSTATE_STENCILFUNC].DefaultValue = 8;
    m_StateCache[VXRENDERSTATE_STENCILMASK].DefaultValue = 0xFFFFFFFF;
    m_StateCache[VXRENDERSTATE_STENCILWRITEMASK].DefaultValue = 0xFFFFFFFF;
    m_StateCache[VXRENDERSTATE_ANTIALIAS].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_TEXTUREPERSPECTIVE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_ZENABLE].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_FILLMODE].DefaultValue = 3;
    m_StateCache[VXRENDERSTATE_LINEPATTERN].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_ZWRITEENABLE].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_ALPHATESTENABLE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_DESTBLEND].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_CULLMODE].DefaultValue = 3;
    m_StateCache[VXRENDERSTATE_ZFUNC].DefaultValue = 4;
    m_StateCache[VXRENDERSTATE_ALPHAREF].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_DITHERENABLE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_ALPHABLENDENABLE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_FOGENABLE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_SPECULARENABLE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_FOGCOLOR].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_FOGSTART].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_FOGEND].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_FOGDENSITY].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_EDGEANTIALIAS].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_ZBIAS].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_RANGEFOGENABLE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_STENCILENABLE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_STENCILFAIL].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_STENCILZFAIL].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_STENCILPASS].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_STENCILREF].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_TEXTUREFACTOR].DefaultValue = A_MASK;
    m_StateCache[VXRENDERSTATE_WRAP0].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_WRAP1].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_WRAP2].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_WRAP3].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_WRAP4].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_WRAP5].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_WRAP6].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_WRAP7].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_CLIPPING].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_LIGHTING].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_AMBIENT].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_FOGVERTEXMODE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_FOGPIXELMODE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_COLORVERTEX].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_LOCALVIEWER].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_NORMALIZENORMALS].DefaultValue = 1;
    m_StateCache[VXRENDERSTATE_CLIPPLANEENABLE].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_INVERSEWINDING].DefaultValue = 0;
    m_StateCache[VXRENDERSTATE_TEXTURETARGET].DefaultValue = 0;
}

CKIndexBufferDesc *CKRasterizerContext::GetIndexBufferData(CKDWORD IB) {
    if (IB >= (CKDWORD) m_IndexBuffers.Size())
        return NULL;
    CKIndexBufferDesc *data = m_IndexBuffers[IB];
    if (!data)
        return NULL;
    if ((data->m_Flags & CKRST_VB_VALID) == 0)
        return NULL;
    return data;
}

CKBOOL CKRasterizerContext::CreateSprite(CKDWORD Sprite, CKSpriteDesc *DesiredFormat) {
    if (Sprite >= (CKDWORD) m_Sprites.Size() || !DesiredFormat)
        return FALSE;

    CKDWORD width = DesiredFormat->Format.Width;
    CKDWORD height = DesiredFormat->Format.Height;
    CKDWORD maxWidthMsb = GetMsb(m_Driver->m_3DCaps.MaxTextureWidth, 32);
    CKDWORD maxHeightMsb = GetMsb(m_Driver->m_3DCaps.MaxTextureHeight, 32);

    CKDWORD minTextureWidth = m_Driver->m_3DCaps.MinTextureWidth;
    if (minTextureWidth < 8)
        minTextureWidth = 8;

    CKSPRTextInfo wti[16] = {};
    int wc = 0;

    CKDWORD widthMsb = GetMsb(width, maxWidthMsb);
    CKDWORD widthLsb = GetLsb(width, maxWidthMsb);
    short sw = (short) (1 << widthMsb);

    if (width < minTextureWidth) {
        wti[0].x = 0;
        wti[0].w = (short) width;
        wti[0].sw = (short) minTextureWidth;
        wc = 1;
    } else if (widthMsb == widthLsb && width == sw) {
        wti[0].x = 0;
        wti[0].w = sw;
        wti[0].sw = sw;
        wc = 1;
    } else if (widthMsb + 1 <= maxWidthMsb && (1 << (widthMsb + 1)) - width <= 32) {
        wti[0].x = 0;
        wti[0].w = (short) width;
        wti[0].sw = (short) (1 << (widthMsb + 1));
        wc = 1;
    } else {
        short x = 0;
        short w = (short) width;
        for (CKSPRTextInfo *pti = &wti[0]; w >= minTextureWidth && wc < 15; ++pti) {
            sw = (short) (1 << widthMsb);

            pti->x = x;
            pti->w = sw;
            pti->sw = sw;

            x += sw;
            w -= sw;
            widthMsb = GetMsb(w, maxWidthMsb);
            ++wc;
        }
        if (w != 0) {
            wti[wc].x = x;
            wti[wc].w = w;
            wti[wc].sw = (short) minTextureWidth;
            ++wc;
        }
    }

    CKSPRTextInfo hti[16] = {};
    int hc = 0;

    CKDWORD heightMsb = GetMsb(height, maxHeightMsb);
    CKDWORD heightLsb = GetLsb(height, maxHeightMsb);
    short sh = (short) (1 << heightMsb);

    CKDWORD maxRatio = m_Driver->m_3DCaps.MaxTextureRatio;
    CKDWORD minHeight = m_Driver->m_3DCaps.MinTextureHeight;

    if (maxRatio != 0) {
        CKDWORD h = (wti[0].sw / maxRatio);
        if (minHeight < h)
            minHeight = h;
    }

    if (height < minHeight) {
        hti[0].y = 0;
        hti[0].h = (short) height;
        hti[0].sh = (short) minHeight;
        hc = 1;
    } else if (heightMsb == heightLsb && height == sh) {
        hti[0].y = 0;
        hti[0].h = sh;
        hti[0].sh = sh;
        hc = 1;
    } else if (heightMsb + 1 <= maxHeightMsb && (1 << (heightMsb + 1)) - height <= 32) {
        hti[0].y = 0;
        hti[0].h = (short) height;
        hti[0].sh = (short) (1 << (heightMsb + 1));
        hc = 1;
    } else {
        short y = 0;
        short h = (short) height;
        for (CKSPRTextInfo *pti = &hti[0]; h >= minHeight && hc < 15; ++pti) {
            sh = (short) (1 << heightMsb);

            pti->y = y;
            pti->h = sh;
            pti->sh = sh;

            y += sh;
            h -= sh;
            heightMsb = GetMsb(h, maxHeightMsb);
            ++hc;
        }
        if (h != 0) {
            hti[hc].y = y;
            hti[hc].h = h;
            hti[hc].sh = (short) minHeight;
            ++hc;
        }
    }

    CKSpriteDesc *sprite = m_Sprites[Sprite];
    if (sprite)
        delete sprite;

    sprite = new CKSpriteDesc;
    m_Sprites[Sprite] = sprite;

    sprite->Textures.Resize(wc * hc);
    sprite->Textures.Memset(0);
    sprite->Owner = m_Driver->m_Owner;

    for (int j = 0; j < hc; ++j) {
        for (int i = 0; i < wc; ++i) {
            CKSPRTextInfo *info = &sprite->Textures[j * wc + i];
            info->x = wti[i].x;
            info->w = wti[i].w;
            info->sw = wti[i].sw;
            info->y = hti[j].y;
            info->h = hti[j].h;
            info->sh = hti[j].sh;
            info->IndexTexture = m_Driver->m_Owner->CreateObjectIndex(CKRST_OBJ_TEXTURE);
            DesiredFormat->Format.Width = info->sw;
            DesiredFormat->Format.Height = info->sh;
            CreateObject(info->IndexTexture, CKRST_OBJ_TEXTURE, DesiredFormat);
        }
    }

    sprite->Flags |= CKRST_TEXTURE_SPRITE;
    sprite->Format.Width = width;
    sprite->Format.Height = height;
    sprite->MipMapCount = 0;

    CKTextureDesc *tex = m_Textures[sprite->Textures[0].IndexTexture];
    if (!tex)
        return FALSE;

    sprite->Format = tex->Format;
    sprite->Format.Width = width;
    sprite->Format.Height = height;
    sprite->Flags = tex->Flags;
    return TRUE;
}

void CKRasterizerContext::UpdateMatrices(CKDWORD Flags) {
    if ((Flags & m_MatrixUptodate) == 0) {
        if ((Flags & WORLD_TRANSFORM) != 0)
            Vx3DMultiplyMatrix4(m_TotalMatrix, m_ProjectionMatrix, m_ModelViewMatrix);
        if ((Flags & VIEW_TRANSFORM) != 0)
            Vx3DMultiplyMatrix4(m_ViewProjMatrix, m_ProjectionMatrix, m_ViewMatrix);
        m_MatrixUptodate |= Flags;
    }
}

CKDWORD CKRasterizerContext::GetDynamicVertexBuffer(CKDWORD VertexFormat, CKDWORD VertexCount, CKDWORD VertexSize, CKDWORD AddKey) {
    if ((m_Driver->m_3DCaps.CKRasterizerSpecificCaps & CKRST_SPECIFICCAPS_CANDOVERTEXBUFFER) == 0)
        return 0;

    CKDWORD index = VertexFormat & (CKRST_VF_RASTERPOS | CKRST_VF_NORMAL);
    index |= (VertexFormat & (CKRST_VF_DIFFUSE | CKRST_VF_SPECULAR | CKRST_VF_TEXMASK)) >> 3;
    index >>= 2;
    index |= AddKey << 7;
    index += 1;

    CKVertexBufferDesc *vb = m_VertexBuffers[index];
    if (!vb || vb->m_MaxVertexCount < VertexCount) {
        if (vb) {
            delete vb;
            m_VertexBuffers[index] = NULL;
        }

        CKVertexBufferDesc nvb;
        nvb.m_Flags = CKRST_VB_WRITEONLY | CKRST_VB_DYNAMIC;
        nvb.m_VertexFormat = VertexFormat;
        nvb.m_VertexSize = VertexSize;
        nvb.m_MaxVertexCount = (VertexCount + 100 > DEFAULT_VB_SIZE) ? VertexCount + 100 : DEFAULT_VB_SIZE;
        if (AddKey != 0)
            nvb.m_Flags |= CKRST_VB_SHARED;
        CreateObject(index, CKRST_OBJ_VERTEXBUFFER, &nvb);
    }

    return index;
}
