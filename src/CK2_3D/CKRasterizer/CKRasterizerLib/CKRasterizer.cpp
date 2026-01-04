#include "CKRasterizer.h"

#include <stdio.h>

int ObjTypeIndex(CKRST_OBJECTTYPE Type) {
    switch (Type) {
    case CKRST_OBJ_TEXTURE: return 0;
    case CKRST_OBJ_SPRITE: return 1;
    case CKRST_OBJ_VERTEXBUFFER: return 2;
    case CKRST_OBJ_INDEXBUFFER: return 3;
    case CKRST_OBJ_VERTEXSHADER: return 4;
    case CKRST_OBJ_PIXELSHADER: return 5;
    default: return 0;
    }
}

CKRasterizer *CKNULLRasterizerStart(WIN_HANDLE AppWnd) {
    CKRasterizer *rst = new CKRasterizer;
    if (!rst)
        return NULL;

    if (!rst->Start(AppWnd)) {
        delete rst;
        rst = NULL;
    }

    return rst;
}

void CKNULLRasterizerClose(CKRasterizer *rst) {
    if (rst) {
        rst->Close();
        delete rst;
    }
}

CKRasterizer::CKRasterizer()
    : m_Objects(),
      m_ObjectsIndex(),
      m_OtherRasterizers(),
      m_ProblematicDrivers(),
      m_Drivers(),
      m_FullscreenContext(NULL) {
    m_ObjectsIndex.Resize(INIT_OBJECTSLOTS);
    m_ObjectsIndex.Fill(0);
    memset(m_ObjectsIndex.Begin(), CKRST_OBJ_VERTEXBUFFER, INIT_OBJECTSLOTS / 2);

    m_FirstFreeIndex[ObjTypeIndex(CKRST_OBJ_TEXTURE)] = 1;
    m_FirstFreeIndex[ObjTypeIndex(CKRST_OBJ_SPRITE)] = 1;
    m_FirstFreeIndex[ObjTypeIndex(CKRST_OBJ_VERTEXBUFFER)] = INIT_OBJECTSLOTS / 2;
    m_FirstFreeIndex[ObjTypeIndex(CKRST_OBJ_INDEXBUFFER)] = 1;
    m_FirstFreeIndex[ObjTypeIndex(CKRST_OBJ_VERTEXSHADER)] = 1;
    m_FirstFreeIndex[ObjTypeIndex(CKRST_OBJ_PIXELSHADER)] = 1;
}

CKRasterizer::~CKRasterizer() {
    m_FullscreenContext = NULL;
}

CKBOOL CKRasterizer::Start(WIN_HANDLE AppWnd) {
    m_MainWindow = AppWnd;
    CKRasterizerDriver *driver = new CKRasterizerDriver;
    driver->InitNULLRasterizerCaps(this);
    m_Drivers.PushBack(driver);
    return TRUE;
}

CKDWORD CKRasterizer::CreateObjectIndex(CKRST_OBJECTTYPE Type, CKBOOL WarnOthers) {
    int i = 0;
    int objectsIndexCount = m_ObjectsIndex.Size();
    for (i = m_FirstFreeIndex[ObjTypeIndex(Type)]; i < objectsIndexCount; ++i)
        if ((Type & m_ObjectsIndex[i]) == 0)
            break;

    if (i > objectsIndexCount) {
        m_ObjectsIndex.Resize(2 * i);
        memset(&m_ObjectsIndex[i], 0, i);

        int driverCount = GetDriverCount();
        for (int d = 0; d < driverCount; d++) {
            CKRasterizerDriver *driver = GetDriver(d);
            if (driver) {
                for (XArray<CKRasterizerContext *>::Iterator it = driver->m_Contexts.Begin();
                     it != driver->m_Contexts.End(); ++it)
                    (*it)->UpdateObjectArrays(this);
            }
        }
    }

    m_ObjectsIndex[i] |= Type;
    m_FirstFreeIndex[ObjTypeIndex(Type)] = i + 1;

    if (WarnOthers) {
        for (CKRasterizer **it = m_OtherRasterizers.Begin(); it != m_OtherRasterizers.End(); ++it)
            (*it)->CreateObjectIndex(Type, FALSE);
    }

    return i;
}

CKBOOL CKRasterizer::ReleaseObjectIndex(CKDWORD ObjectIndex, CKRST_OBJECTTYPE Type, CKBOOL WarnOthers) {
    if (ObjectIndex > (CKDWORD) m_ObjectsIndex.Size())
        return FALSE;
    if ((m_ObjectsIndex[ObjectIndex] & Type) == 0)
        return FALSE;

    m_ObjectsIndex[ObjectIndex] &= ~(CKBYTE) Type;

    int driverCount = GetDriverCount();
    for (int d = 0; d < driverCount; d++) {
        CKRasterizerDriver *driver = GetDriver(d);
        if (driver) {
            for (XArray<CKRasterizerContext *>::Iterator it = driver->m_Contexts.Begin();
                 it != driver->m_Contexts.End(); ++it)
                (*it)->DeleteObject(ObjectIndex, Type);
        }
    }

    if (ObjectIndex < m_FirstFreeIndex[ObjTypeIndex(Type)])
        m_FirstFreeIndex[ObjTypeIndex(Type)] = ObjectIndex;

    if (WarnOthers)
        for (CKRasterizer **it = m_OtherRasterizers.Begin(); it != m_OtherRasterizers.End(); ++it)
            (*it)->ReleaseObjectIndex(ObjectIndex, Type, FALSE);

    return TRUE;
}

XBYTE *CKRasterizer::AllocateObjects(int size) {
    m_Objects.Allocate(size);
    return (XBYTE *) m_Objects.Buffer();
}

void CKRasterizer::LinkRasterizer(CKRasterizer *rst) {
    if (rst != this)
        m_OtherRasterizers.PushBack(rst);
}

void CKRasterizer::RemoveLinkedRasterizer(CKRasterizer *rst) {
    if (rst != this)
        m_OtherRasterizers.Remove(rst);
}

CKBOOL CKRasterizer::LoadVideoCardFile(const char *FileName) {
    VxConfiguration config;
    int cline;
    XString error;
    int sectionCount;
    if (!config.BuildFromFile(FileName, cline, error) ||
        (sectionCount = config.GetNumberOfSubSections()) == 0)
        return FALSE;

    ConstSectionIt sectionIt = config.BeginSections();
    for (VxConfigurationSection *section = *sectionIt;
         section != NULL && --sectionCount > 0; section = config.GetNextSection(sectionIt)) {
        CKDriverProblems driverProblems;

        VxConfigurationEntry *companyEntry = section->GetEntry("Company");
        if (companyEntry)
            driverProblems.m_Vendor = companyEntry->GetValue();

        VxConfigurationEntry *rendererEntry = section->GetEntry("Renderer");
        if (rendererEntry)
            driverProblems.m_Renderer = rendererEntry->GetValue();

        VxConfigurationEntry *exactVersionEntry = section->GetEntry("ExactVersion");
        if (exactVersionEntry) {
            driverProblems.m_Version = exactVersionEntry->GetValue();
            driverProblems.m_VersionMustBeExact = TRUE;
        }

        VxConfigurationEntry *upToVersionEntry = section->GetEntry("UpToVersion");
        if (upToVersionEntry) {
            driverProblems.m_Version = upToVersionEntry->GetValue();
            driverProblems.m_VersionMustBeExact = FALSE;
        }

        VxConfigurationEntry *deviceDescEntry = section->GetEntry("DeviceDesc");
        if (deviceDescEntry)
            driverProblems.m_DeviceDesc = deviceDescEntry->GetValue();

        VxConfigurationEntry *bugClampToEdgeEntry = section->GetEntry("Bug_ClampEdge");
        if (bugClampToEdgeEntry)
            bugClampToEdgeEntry->GetValueAsInteger(driverProblems.m_ClampToEdgeBug);

        VxConfigurationEntry *onlyIn16Entry = section->GetEntry("OnlyIn16Bpp");
        if (onlyIn16Entry)
            onlyIn16Entry->GetValueAsInteger(driverProblems.m_OnlyIn16);

        VxConfigurationEntry *onlyIn32Entry = section->GetEntry("OnlyIn32Bpp");
        if (onlyIn32Entry)
            onlyIn32Entry->GetValueAsInteger(driverProblems.m_OnlyIn32);

        VxConfigurationEntry *maxTextureWidthEntry = section->GetEntry("MaxTextureWidth");
        if (maxTextureWidthEntry)
            maxTextureWidthEntry->GetValueAsInteger(driverProblems.m_RealMaxTextureWidth);

        VxConfigurationEntry *maxTextureHeightEntry = section->GetEntry("MaxTextureHeight");
        if (maxTextureHeightEntry)
            maxTextureHeightEntry->GetValueAsInteger(driverProblems.m_RealMaxTextureHeight);

        VxConfigurationSection *bugRGBASection = section->GetSubSection("Bug_RGBA");
        if (bugRGBASection) {
            VxConfigurationEntry *bug32ARGB8888 = bugRGBASection->GetEntry("_32_ARGB8888");
            if (bug32ARGB8888)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_32_ARGB8888);

            VxConfigurationEntry *bug32RGB888 = bugRGBASection->GetEntry("_32_RGB888");
            if (bug32RGB888)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_32_RGB888);

            VxConfigurationEntry *bug24RGB888 = bugRGBASection->GetEntry("_24_RGB888");
            if (bug24RGB888)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_24_RGB888);

            VxConfigurationEntry *bug16RGB565 = bugRGBASection->GetEntry("_16_RGB565");
            if (bug16RGB565)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_16_RGB565);

            VxConfigurationEntry *bug16RGB555 = bugRGBASection->GetEntry("_16_RGB555");
            if (bug16RGB555)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_16_RGB555);

            VxConfigurationEntry *bug16ARGB1555 = bugRGBASection->GetEntry("_16_ARGB1555");
            if (bug16ARGB1555)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_16_ARGB1555);

            VxConfigurationEntry *bug16ARGB4444 = bugRGBASection->GetEntry("_16_ARGB4444");
            if (bug16ARGB4444)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_16_ARGB4444);

            VxConfigurationEntry *bug8RGB332 = bugRGBASection->GetEntry("_8_RGB332");
            if (bug8RGB332)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_8_RGB332);

            VxConfigurationEntry *bug8ARGB2222 = bugRGBASection->GetEntry("_8_ARGB2222");
            if (bug8ARGB2222)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_8_ARGB2222);

            VxConfigurationEntry *bugDXT1 = bugRGBASection->GetEntry("_DXT1");
            if (bugDXT1)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_DXT1);

            VxConfigurationEntry *bugDXT3 = bugRGBASection->GetEntry("_DXT3");
            if (bugDXT3)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_DXT3);

            VxConfigurationEntry *bugDXT5 = bugRGBASection->GetEntry("_DXT5");
            if (bugDXT5)
                driverProblems.m_TextureFormatsRGBABug.PushBack(_DXT5);
        }

        VxConfigurationSection *OsSection = section->GetSubSection("Os");
        if (OsSection) {
            VxConfigurationEntry *win95 = OsSection->GetEntry("VXOS_WIN95");
            if (win95)
                driverProblems.m_ConcernedOS.PushBack(VXOS_WIN95);

            VxConfigurationEntry *win98 = OsSection->GetEntry("VXOS_WIN98");
            if (win98)
                driverProblems.m_ConcernedOS.PushBack(VXOS_WIN98);

            VxConfigurationEntry *winNT4 = OsSection->GetEntry("VXOS_WINNT4");
            if (winNT4)
                driverProblems.m_ConcernedOS.PushBack(VXOS_WINNT4);

            VxConfigurationEntry *win2k = OsSection->GetEntry("VXOS_WIN2K");
            if (win2k)
                driverProblems.m_ConcernedOS.PushBack(VXOS_WIN2K);

            VxConfigurationEntry *winXP = OsSection->GetEntry("VXOS_WINXP");
            if (winXP)
                driverProblems.m_ConcernedOS.PushBack(VXOS_WINXP);

            VxConfigurationEntry *macOS = OsSection->GetEntry("VXOS_MACOS");
            if (macOS)
                driverProblems.m_ConcernedOS.PushBack(VXOS_MACOS);

            VxConfigurationEntry *linux = OsSection->GetEntry("VXOS_LINUX");
            if (linux)
                driverProblems.m_ConcernedOS.PushBack(VXOS_LINUX);
        }

        m_ProblematicDrivers.PushBack(driverProblems);
    }

    return TRUE;
}

CKDriverProblems *CKRasterizer::FindDriverProblems(const XString &Vendor, const XString &Renderer, const XString &Version, const XString &DeviceDesc, int Bpp) {
    if (m_ProblematicDrivers.Size() == 0)
        return NULL;

    for (XClassArray<CKDriverProblems>::Iterator it = m_ProblematicDrivers.Begin(); it != m_ProblematicDrivers.End(); ++it) {
        if (Vendor != "" && it->m_Vendor == Vendor) {
            if (it->m_Renderer != "" && it->m_Renderer != Renderer)
                continue;
        } else {
            if (it->m_DeviceDesc != DeviceDesc)
                continue;
        }

        if (it->m_Version != "" && Version != "") {
            if (it->m_VersionMustBeExact) {
                if (it->m_Version != Version)
                    continue;
            } else {
                int major1, minor1, patch1;
                sscanf(it->m_Version.CStr(), "%d.%d.%d", &major1, &minor1, &patch1);
                int major2, minor2, patch2;
                sscanf(Version.CStr(), "%d.%d.%d", &major2, &minor2, &patch2);
                if (major1 > major2 || major1 == major2 && (minor1 > minor2 || minor1 == minor2 && patch1 >= patch2))
                    continue;
            }
        }

        if (it->m_OnlyIn16 && Bpp != 16)
            continue;
        if (it->m_OnlyIn32 && Bpp != 32)
            continue;

        if (!it->m_ConcernedOS.IsHere(VxGetOs()))
            continue;

        return it;
    }

    return NULL;
}

void ConvertAttenuationModelFromDX5(float &_a0, float &_a1, float &_a2, float range) {
    _a0 = 1.0f / (_a0 + _a1 + _a2);
    _a1 = (_a2 + _a2 + _a1) * (_a0 / range) * _a0;
    _a2 = _a0 * _a2 * _a0 / (range * range) + _a1 * _a1 / _a0;
}

CKDWORD CKRSTGetVertexFormat(CKRST_DPFLAGS DpFlags, CKDWORD &VertexSize) {
    CKDWORD count = 0;
    for (CKDWORD flag = CKRST_DP_STAGEFLAGS(DpFlags); flag != 0; flag >>= 1)
        ++count;

    CKDWORD format;
    if (DpFlags & CKRST_DP_TRANSFORM) {
        format = CKRST_VF_POSITION;
        VertexSize = sizeof(VxVector);

        if (DpFlags & CKRST_DP_LIGHT) {
            format |= CKRST_VF_NORMAL;
            VertexSize += sizeof(VxVector);
        } else {
            if (DpFlags & CKRST_DP_DIFFUSE) {
                format |= CKRST_VF_DIFFUSE;
                VertexSize += sizeof(CKDWORD);
            }

            if (DpFlags & CKRST_DP_SPECULAR) {
                format |= CKRST_VF_SPECULAR;
                VertexSize += sizeof(CKDWORD);
            }
        }
    } else {
        format = CKRST_VF_RASTERPOS;
        VertexSize = sizeof(VxVector4);

        if (DpFlags & CKRST_DP_DIFFUSE) {
            format |= CKRST_VF_DIFFUSE;
            VertexSize += sizeof(CKDWORD);
        }

        if (DpFlags & CKRST_DP_SPECULAR) {
            format |= CKRST_VF_SPECULAR;
            VertexSize += sizeof(CKDWORD);
        }
    }

    format |= CKRST_VF_TEXCOUNT(count);
    VertexSize += count * 2 * sizeof(float);

    return format;
}

CKDWORD CKRSTGetVertexSize(CKDWORD VertexFormat) {
    CKDWORD vertexSize;
    switch (VertexFormat & 0xF) {
    case CKRST_VF_POSITION:
        vertexSize = 12;
        break;
    case CKRST_VF_RASTERPOS:
    case CKRST_VF_POSITION1W:
        vertexSize = 16;
        break;
    case CKRST_VF_POSITION2W:
        vertexSize = 20;
        break;
    case CKRST_VF_POSITION3W:
        vertexSize = 24;
        break;
    case CKRST_VF_POSITION4W:
        vertexSize = 28;
        break;
    case CKRST_VF_POSITION5W:
        vertexSize = 32;
        break;
    }

    if ((VertexFormat & CKRST_VF_NORMAL) != 0)
        vertexSize += 12;
    if ((VertexFormat & CKRST_VF_DIFFUSE) != 0)
        vertexSize += 4;
    if ((VertexFormat & CKRST_VF_PSIZE) != 0)
        vertexSize += 4;
    if ((VertexFormat & CKRST_VF_SPECULAR) != 0)
        vertexSize += 4;

    CKWORD tex = VertexFormat >> 16;
    if (tex == 0) {
        vertexSize += 8 * ((VertexFormat & CKRST_VF_TEXMASK) >> 8);
        return vertexSize;
    }

    if ((VertexFormat & CKRST_VF_TEXMASK) != 0) {
        CKDWORD texSize[4] = {8, 12, 16, 4};
        for (int i = 0; i < CKRST_MAX_STAGES - 1; ++i) {
            vertexSize += texSize[tex & 3];
            tex >>= 2;
        }
    }

    return vertexSize;
}

CKBYTE *CKRSTLoadVertexBuffer(CKBYTE *VBMem, CKDWORD VFormat, CKDWORD VSize, VxDrawPrimitiveData *data) {
    CKBYTE *positionPtr = (CKBYTE *) data->PositionPtr;
    if (VFormat == CKRST_VF_VERTEX &&
        VSize == sizeof(CKVertex) &&
        data->PositionStride == sizeof(CKVertex) &&
        data->NormalStride == sizeof(CKVertex) &&
        data->TexCoordStride == sizeof(CKVertex) &&
        data->NormalPtr == positionPtr + sizeof(VxVector) &&
        data->TexCoordPtr == positionPtr + sizeof(VxVector4) + 2 * sizeof(CKDWORD)) {
        memcpy(VBMem, positionPtr, data->VertexCount * sizeof(CKVertex));
        return &VBMem[data->VertexCount * sizeof(CKVertex)];
    }

    int offset;
    if (VFormat & CKRST_VF_RASTERPOS) {
        VxCopyStructure(data->VertexCount, VBMem, VSize, sizeof(VxVector4), data->PositionPtr, data->PositionStride);
        offset = sizeof(VxVector4);
    } else {
        VxCopyStructure(data->VertexCount, VBMem, VSize, sizeof(VxVector), data->PositionPtr, data->PositionStride);
        offset = sizeof(VxVector);
    }

    if (VFormat & CKRST_VF_NORMAL) {
        if (data->NormalPtr)
            VxCopyStructure(data->VertexCount, &VBMem[offset], VSize, sizeof(VxVector), data->NormalPtr, data->NormalStride);
        offset += sizeof(VxVector);
    }

    if (VFormat & CKRST_VF_DIFFUSE) {
        if (data->ColorPtr) {
            VxCopyStructure(data->VertexCount, &VBMem[offset], VSize, sizeof(CKDWORD), data->ColorPtr, data->ColorStride);
        } else {
            CKDWORD src = 0xFFFFFFFF;
            VxFillStructure(data->VertexCount, &VBMem[offset], VSize, sizeof(CKDWORD), &src);
        }
        offset += sizeof(CKDWORD);
    }

    if (VFormat & CKRST_VF_SPECULAR) {
        if (data->SpecularColorPtr) {
            VxCopyStructure(data->VertexCount, &VBMem[offset], VSize, sizeof(CKDWORD), data->SpecularColorPtr, data->SpecularColorStride);
        } else {
            CKDWORD src = 0;
            VxFillStructure(data->VertexCount, &VBMem[offset], VSize, sizeof(CKDWORD), &src);
        }
        offset += sizeof(CKDWORD);
    }

    CKDWORD texCount = CKRST_VF_GETTEXCOUNT(VFormat);
    if (texCount != 0) {
        if (data->TexCoordPtr)
            VxCopyStructure(data->VertexCount, &VBMem[offset], VSize, 2 * sizeof(float), data->TexCoordPtr, data->TexCoordStride);
        offset += 2 * sizeof(float);
    }
    if (texCount > 1) {
        for (CKDWORD i = 0; i < texCount - 1; ++i) {
            if (data->TexCoordPtrs[i])
                VxCopyStructure(data->VertexCount, &VBMem[offset], VSize, 2 * sizeof(float), data->TexCoordPtrs[i], data->TexCoordStrides[i]);
            offset += 2 * sizeof(float);
        }
    }

    return &VBMem[data->VertexCount * VSize];
}

void CKRSTSetupDPFromVertexBuffer(CKBYTE *VBMem, CKVertexBufferDesc *VB, VxDrawPrimitiveData &DpData) {
    DpData.PositionPtr = VBMem;
    DpData.PositionStride = VB->m_VertexSize;

    CKBYTE *ptr;
    if ((VB->m_VertexFormat & CKRST_VF_POSITION) != 0)
        ptr = VBMem + 12;
    else
        ptr = VBMem + 16;

    if ((VB->m_VertexFormat & CKRST_VF_NORMAL) != 0) {
        DpData.NormalPtr = ptr;
        DpData.NormalStride = VB->m_VertexSize;
        ptr += 12;
    } else {
        DpData.NormalPtr = NULL;
        DpData.NormalStride = 0;
    }

    if ((VB->m_VertexFormat & CKRST_VF_DIFFUSE) != 0) {
        DpData.ColorPtr = ptr;
        DpData.ColorStride = VB->m_VertexSize;
        ptr += 4;
    } else {
        DpData.ColorPtr = NULL;
        DpData.ColorStride = 0;
    }

    if ((VB->m_VertexFormat & CKRST_VF_SPECULAR) != 0) {
        DpData.SpecularColorPtr = ptr;
        DpData.SpecularColorStride = VB->m_VertexSize;
        ptr += 4;
    } else {
        DpData.SpecularColorPtr = NULL;
        DpData.SpecularColorStride = 0;
    }

    DpData.TexCoordPtr = ptr;
    DpData.TexCoordStride = VB->m_VertexSize;
    ptr += 8;

    memset(DpData.TexCoordPtrs, NULL, sizeof(DpData.TexCoordPtrs));
    memset(DpData.TexCoordStrides, 0, sizeof(DpData.TexCoordStrides));
    if ((VB->m_VertexFormat & CKRST_VF_TEXMASK) > CKRST_VF_TEX1)
        for (int i = 0; i < CKRST_MAX_STAGES - 1; ++i) {
            DpData.TexCoordPtrs[i] = ptr;
            DpData.TexCoordStrides[i] = VB->m_VertexSize;
            ptr += 8;
        }
}
