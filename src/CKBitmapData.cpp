#include "CKBitmapData.h"

#include "VxMath.h"
#include "CKGlobals.h"
#include "CKFile.h"
#include "CKStateChunk.h"
#include "CKPluginManager.h"
#include "CKPathManager.h"

extern CKSTRING CKJustFile(CKSTRING path);

CKMovieInfo::CKMovieInfo(const XString &FileName)
    : m_MovieFileName(FileName),
      m_MovieReader(nullptr),
      m_MovieCurrentSlot(-1) {
}

CKMovieInfo::~CKMovieInfo() {
    if (m_MovieReader)
        m_MovieReader->Release();
}

CKBitmapData::CKBitmapData(const CKBitmapData &rhs)
    : m_MovieInfo(rhs.m_MovieInfo),
      m_Slots(rhs.m_Slots),
      m_Width(rhs.m_Width),
      m_Height(rhs.m_Height),
      m_CurrentSlot(rhs.m_CurrentSlot),
      m_PickThreshold(rhs.m_PickThreshold),
      m_BitmapFlags(rhs.m_BitmapFlags),
      m_TransColor(rhs.m_TransColor),
      m_SaveProperties(rhs.m_SaveProperties),
      m_SaveOptions(rhs.m_SaveOptions) {
}

CKBitmapData &CKBitmapData::operator=(const CKBitmapData &rhs) {
    if (this != &rhs) {
        m_MovieInfo = rhs.m_MovieInfo;
        m_Slots = rhs.m_Slots;
        m_Width = rhs.m_Width;
        m_Height = rhs.m_Height;
        m_CurrentSlot = rhs.m_CurrentSlot;
        m_PickThreshold = rhs.m_PickThreshold;
        m_BitmapFlags = rhs.m_BitmapFlags;
        m_TransColor = rhs.m_TransColor;
        m_SaveProperties = rhs.m_SaveProperties;
        m_SaveOptions = rhs.m_SaveOptions;
    }
    return *this;
}

CKBOOL CKBitmapData::CreateImage(int Width, int Height, int BPP, int Slot) {
    // Original condition: (flags & 1) || (slots.size <= 1 && Slot == 0)
    if ((m_BitmapFlags & CKBITMAPDATA_INVALID) != 0 || (m_Slots.Size() <= 1 && Slot == 0)) {
        m_Width = Width;
        m_Height = Height;
        // Original: LOBYTE(flags) = flags & ~5 | 4 (clears bits 0,2, sets bit 2)
        m_BitmapFlags = (m_BitmapFlags & ~(CKBITMAPDATA_INVALID | CKBITMAPDATA_FORCERESTORE)) | CKBITMAPDATA_FORCERESTORE;
    } else if (m_Width != Width || m_Height != Height) {
        return FALSE;
    }

    if (m_Width == 0 || m_Height == 0) {
        m_BitmapFlags |= CKBITMAPDATA_INVALID;
        return FALSE;
    }

    const int count = m_Slots.Size();
    if (Slot >= count) {
        m_Slots.Resize(Slot + 1);
        for (int i = count; i <= Slot; ++i) {
            m_Slots[i] = new CKBitmapSlot;
        }
    }

    CKBitmapSlot *slot = m_Slots[Slot];
    slot->m_FileName = "";
    VxDeleteAligned(slot->m_DataBuffer);
    slot->m_DataBuffer = (CKDWORD *)VxNewAligned(4 * m_Height * m_Width, 16);

    CKDWORD *buffer = slot->m_DataBuffer;
    unsigned int dwordCount = m_Width * m_Height;
    if (dwordCount) {
        for (unsigned int i = 0; i < dwordCount; ++i) {
            buffer[i] = A_MASK;
        }
    }

    return TRUE;
}

CKBOOL CKBitmapData::SaveImage(CKSTRING Name, int Slot, CKBOOL CKUseFormat) {
    if (!Name) return FALSE;

    CKPathSplitter splitter(Name);
    char extension[_MAX_PATH] = ".bmp";
    CKBitmapReader *reader = nullptr;
    CKBitmapProperties *saveProps = m_SaveProperties;

    if (CKUseFormat && saveProps) {
        reader = CKGetPluginManager()->GetBitmapReader(saveProps->m_Ext, &saveProps->m_ReaderGuid);
        if (reader) {
            snprintf(extension, sizeof(extension), ".%s", saveProps->m_Ext.m_Data);
        } else {
            saveProps = nullptr; // Fallback to default handling
        }
    } else {
        char *ext = splitter.GetExtension();
        if (ext && *ext != '\0') {
            snprintf(extension, sizeof(extension), "%s", ext);
        }
    }

    if (!reader) {
        CKFileExtension desiredExt(extension + 1); // Skip leading dot
        reader = CKGetPluginManager()->GetBitmapReader(desiredExt, nullptr);
        if (reader) {
            reader->GetBitmapDefaultProperties(&saveProps);
        }
    }

    if (!reader) {
        return FALSE;
    }

    CKBYTE *surfaceData = LockSurfacePtr(Slot);
    if (!surfaceData) {
        reader->Release();
        return FALSE;
    }

    CKPathMaker pathMaker(splitter.GetDrive(), splitter.GetDir(), splitter.GetName(), extension);
    XString fullPath = pathMaker.GetFileName();

    VxImageDescEx imgDesc;
    GetImageDesc(imgDesc);

    saveProps->m_Format = imgDesc;
    saveProps->m_Data = surfaceData;

    CKBOOL result = reader->SaveFile(fullPath.Str(), saveProps);

    saveProps->m_Data = nullptr;
    reader->Release();

    return result;
}

CKBOOL CKBitmapData::SaveImageAlpha(CKSTRING Name, int Slot) {
    if (!Name)
        return FALSE;

    CKPathSplitter pathSplitter(Name);
    char extension[_MAX_PATH] = ".bmp";
    CKBOOL result = FALSE;

    char *fileExt = pathSplitter.GetExtension();
    if (fileExt && strlen(fileExt)) {
        sprintf(extension, "%s", fileExt);
    }

    CKFileExtension desiredExt(extension + 1); // Skip leading dot
    CKBitmapReader *reader = CKGetPluginManager()->GetBitmapReader(desiredExt, nullptr);
    if (!reader)
        return FALSE;

    CKBitmapProperties *saveProps = nullptr;
    reader->GetBitmapDefaultProperties(&saveProps);

    CKBYTE *surfaceData = (CKBYTE *)LockSurfacePtr(Slot);
    if (!surfaceData) {
        reader->Release();
        return FALSE;
    }

    CKPathMaker pathMaker(pathSplitter.GetDrive(), pathSplitter.GetDir(), pathSplitter.GetName(), extension);
    GetImageDesc(saveProps->m_Format);

    const int pixelCount = m_Width * m_Height;
    CKBYTE *alphaBuffer = new CKBYTE[4 * pixelCount];

    // Original loop: copy alpha byte to R,G,B channels, set A to 0xFF
    if (pixelCount > 0) {
        CKBYTE *srcAlpha = surfaceData + 3;  // Point to alpha byte (offset 3 in ARGB)
        CKBYTE *dst = alphaBuffer;
        for (int i = 0; i < pixelCount; ++i) {
            CKBYTE alpha = *srcAlpha;
            dst[0] = alpha;      // R
            dst[1] = alpha;      // G
            dst[2] = alpha;      // B
            dst[3] = 0xFF;       // A (opaque)
            srcAlpha += 4;
            dst += 4;
        }
    }

    saveProps->m_Data = alphaBuffer;
    result = reader->SaveFile(pathMaker.GetFileName(), saveProps);

    delete[] alphaBuffer;
    saveProps->m_Data = nullptr;
    reader->Release();

    return result;
}

CKSTRING CKBitmapData::GetMovieFileName() {
    if (!m_MovieInfo)
        return nullptr;
    return m_MovieInfo->m_MovieFileName.Str();
}

CKBYTE *CKBitmapData::LockSurfacePtr(int Slot) {
    // Original uses unsigned comparison: if slot >= size, use m_CurrentSlot
    if ((unsigned int)Slot >= (unsigned int)m_Slots.Size())
        Slot = m_CurrentSlot;

    if (m_Slots.IsEmpty())
        return nullptr;

    return (CKBYTE *)m_Slots[Slot]->m_DataBuffer;
}

CKBOOL CKBitmapData::ReleaseSurfacePtr(int Slot) {
    m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
    return TRUE;
}

CKSTRING CKBitmapData::GetSlotFileName(int Slot) {
    // Original uses unsigned comparison
    if ((unsigned int)Slot >= (unsigned int)m_Slots.Size())
        return nullptr;

    CKSTRING result = m_Slots[Slot]->m_FileName.Str();
    if (!result)
        return "";
    return result;
}

CKBOOL CKBitmapData::SetSlotFileName(int Slot, CKSTRING Filename) {
    // Original uses unsigned comparison and doesn't check Filename for NULL
    if ((unsigned int)Slot >= (unsigned int)m_Slots.Size())
        return FALSE;

    m_Slots[Slot]->m_FileName = Filename;
    return TRUE;
}

CKBOOL CKBitmapData::GetImageDesc(VxImageDescEx &oDesc) {
    oDesc.AlphaMask = A_MASK;
    oDesc.RedMask = R_MASK;
    oDesc.GreenMask = G_MASK;
    oDesc.BlueMask = B_MASK;
    oDesc.Width = m_Width;
    oDesc.Height = m_Height;
    oDesc.BitsPerPixel = 32;
    oDesc.BytesPerLine = 4 * m_Width;
    oDesc.Image = nullptr;
    return TRUE;
}

int CKBitmapData::GetSlotCount() {
    if (!m_MovieInfo)
        return m_Slots.Size();

    CKMovieReader *reader = m_MovieInfo->m_MovieReader;
    return reader ? reader->GetMovieFrameCount() : 0;
}

CKBOOL CKBitmapData::SetSlotCount(int Count) {
    if (m_MovieInfo) return FALSE;

    const int oldSlotCount = m_Slots.Size();
    if (Count < oldSlotCount) {
        for (int i = Count; i < oldSlotCount; ++i) {
            CKBitmapSlot *slot = m_Slots[i];
            if (slot) {
                delete slot;
            }
        }
    }
    m_Slots.Resize(Count);
    if (Count > oldSlotCount) {
        for (int i = oldSlotCount; i < Count; ++i) {
            m_Slots[i] = new CKBitmapSlot;
        }
    }
    if (Count == 0) m_BitmapFlags |= CKBITMAPDATA_INVALID;
    return TRUE;
}

CKBOOL CKBitmapData::SetCurrentSlot(int Slot) {
    if (!m_MovieInfo) {
        // Original uses unsigned comparison
        if ((unsigned int)Slot >= (unsigned int)m_Slots.Size())
            return FALSE;
        if (Slot != m_CurrentSlot) {
            m_CurrentSlot = Slot;
            if (!(m_BitmapFlags & CKBITMAPDATA_CUBEMAP)) {
                m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
            }
        }
        return TRUE;
    }

    // Movie mode: check if same slot requested
    if (Slot == m_MovieInfo->m_MovieCurrentSlot)
        return TRUE;

    CKMovieReader *reader = m_MovieInfo->m_MovieReader;
    if (!reader)
        return FALSE;

    // Update movie slot first, then read frame
    m_MovieInfo->m_MovieCurrentSlot = Slot;
    CKMovieProperties *movieProps = nullptr;
    if (reader->ReadFrame(Slot, &movieProps) != CK_OK)
        return FALSE;

    // Get buffer from first slot
    CKBYTE *buffer = (CKBYTE *)(*(CKDWORD **)m_Slots.Begin());
    if (!buffer)
        return FALSE;

    VxImageDescEx dstDesc;
    GetImageDesc(dstDesc);
    dstDesc.Image = buffer;
    movieProps->m_Format.Image = (CKBYTE *)movieProps->m_Data;
    VxDoBlitUpsideDown(movieProps->m_Format, dstDesc);
    m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
    return TRUE;
}

int CKBitmapData::GetCurrentSlot() {
    return m_MovieInfo ? m_MovieInfo->m_MovieCurrentSlot : m_CurrentSlot;
}

CKBOOL CKBitmapData::ReleaseSlot(int Slot) {
    if (m_MovieInfo)
        return FALSE;
    if ((unsigned int)Slot >= (unsigned int)m_Slots.Size())
        return FALSE;

    CKBitmapSlot *slot = m_Slots[Slot];
    if (slot) {
        delete slot;
    }
    m_Slots.RemoveAt(Slot);

    if (m_CurrentSlot >= m_Slots.Size()) {
        m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
        m_CurrentSlot = m_Slots.Size() - 1;
    }
    if (m_Slots.IsEmpty())
        m_BitmapFlags |= CKBITMAPDATA_INVALID;
    return TRUE;
}

CKBOOL CKBitmapData::ReleaseAllSlots() {
    if (m_MovieInfo)
        return FALSE;

    int size = m_Slots.Size();
    for (int i = 0; i < size; ++i) {
        CKBitmapSlot *slot = m_Slots[i];
        if (slot) {
            delete slot;
        }
    }

    // Clear() uses Free() internally which handles memory deallocation
    m_Slots.Clear();

    m_BitmapFlags |= CKBITMAPDATA_INVALID;
    return TRUE;
}

CKBOOL CKBitmapData::SetPixel(int x, int y, CKDWORD col, int slot) {
    unsigned int targetSlot = (slot < 0) ? m_CurrentSlot : slot;
    if (targetSlot >= (unsigned int)m_Slots.Size())
        return FALSE;

    CKDWORD *buffer = m_Slots[targetSlot]->m_DataBuffer;
    if (!buffer)
        return FALSE;

    buffer[x + y * m_Width] = col;
    return TRUE;
}

CKDWORD CKBitmapData::GetPixel(int x, int y, int slot) {
    unsigned int targetSlot = (slot < 0) ? m_CurrentSlot : slot;
    if (targetSlot >= (unsigned int)m_Slots.Size())
        return 0;

    CKDWORD *buffer = m_Slots[targetSlot]->m_DataBuffer;
    if (!buffer)
        return 0;

    return buffer[x + y * m_Width];
}

void CKBitmapData::SetTransparentColor(CKDWORD Color) {
    if ((m_BitmapFlags & CKBITMAPDATA_TRANSPARENT) && (m_TransColor != Color)) {
        m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
    }
    m_TransColor = Color;
}

void CKBitmapData::SetTransparent(CKBOOL Transparency) {
    if (Transparency) {
        if (!(m_BitmapFlags & CKBITMAPDATA_TRANSPARENT)) {
            m_BitmapFlags |= CKBITMAPDATA_TRANSPARENT | CKBITMAPDATA_FORCERESTORE;
        }
    } else {
        m_BitmapFlags &= ~CKBITMAPDATA_TRANSPARENT;
    }
}

void CKBitmapData::SetSaveFormat(CKBitmapProperties *format) {
    delete[] reinterpret_cast<CKBYTE *>(m_SaveProperties);
    m_SaveProperties = CKCopyBitmapProperties(format);
}

CKBOOL CKBitmapData::ResizeImages(int Width, int Height) {
    if (m_MovieInfo)
        return FALSE;
    if (m_Slots.IsEmpty())
        return FALSE;
    if (m_Width <= 0 || m_Height <= 0)
        return FALSE;
    if (Width <= 0 || Height <= 0)
        return FALSE;
    if (m_Width == Width && m_Height == Height)
        return TRUE;

    VxImageDescEx srcDesc;
    srcDesc.Width = m_Width;
    srcDesc.Height = m_Height;
    srcDesc.BitsPerPixel = 32;
    srcDesc.BytesPerLine = m_Width * 4;
    srcDesc.RedMask = R_MASK;
    srcDesc.GreenMask = G_MASK;
    srcDesc.BlueMask = B_MASK;
    srcDesc.AlphaMask = A_MASK;

    VxImageDescEx dstDesc;
    dstDesc.Width = Width;
    dstDesc.Height = Height;
    dstDesc.BitsPerPixel = 32;
    dstDesc.BytesPerLine = Width * 4;
    dstDesc.RedMask = R_MASK;
    dstDesc.GreenMask = G_MASK;
    dstDesc.BlueMask = B_MASK;
    dstDesc.AlphaMask = A_MASK;

    for (int i = 0; i < m_Slots.Size(); ++i) {
        CKBitmapSlot *slot = m_Slots[i];
        if (slot) {
            slot->Resize(srcDesc, dstDesc);
        }
    }

    m_Width = Width;
    m_Height = Height;
    return TRUE;
}

CKBOOL CKBitmapData::LoadSlotImage(XString Name, int Slot) {
    CKSTRING nameStr = Name.Str();

    CKPathSplitter splitter(nameStr);
    CKSTRING extStr = splitter.GetExtension();
    
    CKFileExtension extension(extStr + 1);  // Skip the dot
    CKPluginManager *pm = CKGetPluginManager();
    CKBitmapReader *reader = pm->GetBitmapReader(extension);
    if (!reader)
        return FALSE;

    nameStr = Name.Str();

    CKBitmapProperties *props = nullptr;
    if (reader->ReadFile(nameStr, &props) != 0 || !props || !props->m_Data) {
        reader->Release();
        return FALSE;
    }

    const int slotCount = GetSlotCount();
    if (Slot >= slotCount)  // Original uses >= not >
        SetSlotCount(Slot + 1);

    if (!CreateImage(props->m_Format.Width, props->m_Format.Height, 32, Slot)) {
        SetSlotCount(slotCount);
        reader->ReleaseMemory(props->m_Data);
        props->m_Data = nullptr;
        if (props->m_Format.ColorMap)
            reader->ReleaseMemory(props->m_Format.ColorMap);
        props->m_Format.ColorMap = nullptr;
        reader->Release();
        return FALSE;
    }

    CKBYTE *data = LockSurfacePtr(Slot);
    if (!data) {
        reader->ReleaseMemory(props->m_Data);
        props->m_Data = nullptr;

        if (props->m_Format.ColorMap)
            reader->ReleaseMemory(props->m_Format.ColorMap);
        props->m_Format.ColorMap = nullptr;

        reader->Release();
        return FALSE;
    }

    VxImageDescEx desc;
    GetImageDesc(desc);

    props->m_Format.Image = (XBYTE *)props->m_Data;
    desc.Image = data;

    VxDoBlit(props->m_Format, desc);

    if (props->m_Format.AlphaMask == 0)
        VxDoAlphaBlit(desc, 0xFF);

    nameStr = Name.Str();
    SetSlotFileName(Slot, nameStr);
    SetSaveFormat(props);

    reader->ReleaseMemory(props->m_Data);
    props->m_Data = nullptr;

    if (props->m_Format.ColorMap)
        reader->ReleaseMemory(props->m_Format.ColorMap);
    props->m_Format.ColorMap = nullptr;

    reader->Release();
    return TRUE;
}

CKBOOL CKBitmapData::LoadMovieFile(XString Name) {
    // Original deletes existing movie info first
    if (m_MovieInfo) {
        delete m_MovieInfo;
        m_MovieInfo = nullptr;
    }

    // Create new movie information
    CKMovieProperties *movieProps = nullptr;
    m_MovieInfo = CreateMovieInfo(Name, &movieProps);
    if (!m_MovieInfo) {
        return FALSE;
    }

    // Create image slot with movie dimensions
    CreateImage(movieProps->m_Format.Width, movieProps->m_Format.Height, 32, 0);

    // Set active slot for movie playback
    SetCurrentSlot(0);

    return TRUE;
}

CKMovieInfo *CKBitmapData::CreateMovieInfo(XString s, CKMovieProperties **mp) {
    CKMovieInfo *movieInfo = new CKMovieInfo(s);
    if (!movieInfo)
        return nullptr;

    CKPathSplitter splitter(s.Str());
    CKFileExtension ext(splitter.GetExtension());

    CKPluginManager *pm = CKGetPluginManager();
    CKMovieReader *reader = pm->GetMovieReader(ext, nullptr);
    
    // Original sets m_MovieReader immediately
    movieInfo->m_MovieReader = reader;
    
    if (!reader) {
        delete movieInfo;
        return nullptr;
    }

    if (reader->OpenFile(s.Str()) != CK_OK ||
        movieInfo->m_MovieReader->ReadFrame(0, mp) != CK_OK) {
        movieInfo->m_MovieReader->Release();
        movieInfo->m_MovieReader = nullptr;
        delete movieInfo;
        return nullptr;
    }

    return movieInfo;
}

void CKBitmapData::SetMovieInfo(CKMovieInfo *mi) {
    if (m_MovieInfo)
        delete m_MovieInfo;
    m_MovieInfo = mi;
}

CKBitmapData::CKBitmapData() {
    m_Width = -1;
    m_Height = -1;
    m_MovieInfo = nullptr;
    m_PickThreshold = 0;
    m_BitmapFlags = CKBITMAPDATA_INVALID;
    m_TransColor = 0;
    m_CurrentSlot = 0;
    m_SaveProperties = nullptr;
    m_SaveOptions = CKTEXTURE_USEGLOBAL;
}

CKBitmapData::~CKBitmapData() {
    if (m_MovieInfo) {
        delete m_MovieInfo;
        m_MovieInfo = nullptr;
    }

    ReleaseAllSlots();
    delete[] reinterpret_cast<CKBYTE *>(m_SaveProperties);
}

void CKBitmapData::SetAlphaForTransparentColor(const VxImageDescEx &desc) {
    // Extract RGB components from transparency color (ignore alpha)
    const CKDWORD transRGB = m_TransColor & 0x00FEFEFE;

    // Calculate total pixels
    const int pixelCount = desc.Width * desc.Height;

    // Get raw pixel data pointer (assuming 32bpp ARGB format)
    CKDWORD *pixels = (CKDWORD *)desc.Image;

    if (!pixels) return;

    // Process each pixel
    for (int i = 0; i < pixelCount; ++i) {
        // Preserve RGB components while ignoring LSBs
        const CKDWORD pixelRGB = pixels[i] & 0x00FEFEFE;

        // Compare with transparency color (RGB only)
        if (pixelRGB == transRGB) {
            // Match: Set alpha to 0 (transparent) | keep RGB
            pixels[i] = pixelRGB;
        } else {
            // No match: Set alpha to 255 (opaque) | keep RGB
            pixels[i] = pixelRGB | 0xFF000000;
        }
    }
}

void CKBitmapData::SetBorderColorForClamp(const VxImageDescEx &desc) {
    m_BitmapFlags |= CKBITMAPDATA_CLAMPUPTODATE;

    CKBYTE *Image = desc.Image;
    int Height = desc.Height;
    int BytesPerLine = desc.BytesPerLine;
    int Width = desc.Width;

    // First loop: clear alpha on left (first pixel) and right (last pixel) of each row
    CKBYTE *leftPtr = Image;
    CKDWORD *rightPtr = (CKDWORD *)(Image + BytesPerLine - 4);
    do {
        CKDWORD rightVal = *rightPtr & 0x00FFFFFF;
        *(CKDWORD *)leftPtr &= 0x00FFFFFF;
        *rightPtr = rightVal;
        leftPtr += BytesPerLine;
        rightPtr = (CKDWORD *)((CKBYTE *)rightPtr + BytesPerLine);
        --Height;
    } while (Height);

    // Second loop: clear alpha on top row and bottom row
    // After first loop, rightPtr points one row past the image
    // v9 = rightPtr - BytesPerLine = last pixel of last row
    CKBYTE *topPtr = Image;
    CKDWORD *bottomPtr = (CKDWORD *)((CKBYTE *)rightPtr - BytesPerLine);
    do {
        CKDWORD bottomVal = *bottomPtr & 0x00FFFFFF;
        *(CKDWORD *)topPtr &= 0x00FFFFFF;
        *bottomPtr = bottomVal;
        topPtr += 4;
        --bottomPtr;
        --Width;
    } while (Width);
}

CKBOOL CKBitmapData::SetSlotImage(int Slot, void *buffer, VxImageDescEx &bdesc) {
    if (!buffer)
        return FALSE;

    if (!CreateImage(bdesc.Width, bdesc.Height, 0, Slot))
        return FALSE;

    VxImageDescEx desc;
    GetImageDesc(desc);

    CKBYTE *data = LockSurfacePtr(Slot);
    if (!data)
        return FALSE;

    desc.Image = data;
    bdesc.Image = (XBYTE *)buffer;
    VxDoBlit(bdesc, desc);

    m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
    delete[] static_cast<CKBYTE *>(buffer);
    delete[] bdesc.ColorMap;
    return TRUE;
}

CKBOOL CKBitmapData::DumpToChunk(CKStateChunk *chnk, CKContext *ctx, CKFile *f, CKDWORD Identifiers[4]) {
    // 1. Determine initial save options
    CK_BITMAP_SAVEOPTIONS effectiveSaveOptions = m_SaveOptions;
    if (effectiveSaveOptions == CKTEXTURE_USEGLOBAL) {
        effectiveSaveOptions = ctx->m_GlobalImagesSaveOptions;
    }

    const int slotCount = GetSlotCount();
    VxImageDescEx imageDescForSaving;
    GetImageDesc(imageDescForSaving);

    // 2. Handle Movie files (Highest Priority)
    CKSTRING movieFile = GetMovieFileName();
    if (movieFile && *movieFile) {
        // Ensure movieFile is not NULL and not empty
        chnk->WriteIdentifier(Identifiers[0]);
        chnk->WriteString(CKJustFile(movieFile));
        if (f && effectiveSaveOptions == CKTEXTURE_INCLUDEORIGINALFILE) {
            f->IncludeFile(movieFile, BITMAP_PATH_IDX);
        }
        return TRUE;
    }

    // 3. Adjust save options based on context
    // If it was global, and not a movie, it might become RAWDATA if global options are not well-defined for images.
    // More importantly, if not saving to a file, force RAWDATA.
    if (effectiveSaveOptions == CKTEXTURE_USEGLOBAL) {
        effectiveSaveOptions = CKTEXTURE_RAWDATA;
    }
    if (!f) {
        // If f is NULL, we are not saving to a persistent CKFile, must be raw.
        effectiveSaveOptions = CKTEXTURE_RAWDATA;
    }

    XBitArray externalOrIncludedSlots(1);
    externalOrIncludedSlots.CheckSize(slotCount);

    // 4. Process CKTEXTURE_EXTERNAL: Mark slots that have explicit filenames
    if (effectiveSaveOptions == CKTEXTURE_EXTERNAL) {
        for (int i = 0; i < slotCount; ++i) {
            if (GetSlotFileName(i) && *(GetSlotFileName(i)) != '\0') {
                // Slot has a filename
                externalOrIncludedSlots.Set(i);
            } else {
                // If any slot is external but has no filename, the whole operation must fallback
                effectiveSaveOptions = CKTEXTURE_RAWDATA;
                break; // No need to check further slots for this option
            }
        }
    }

    // 5. Determine effective CKBitmapProperties to use for saving
    CKBitmapProperties *propertiesForImageFormat = nullptr;
    if (m_SaveOptions == CKTEXTURE_USEGLOBAL) {
        propertiesForImageFormat = ctx->m_GlobalImagesSaveFormat;
    } else {
        propertiesForImageFormat = m_SaveProperties;
    }

    // 6. Process CKTEXTURE_INCLUDEORIGINALFILE (only if saving to a CKFile)
    if (f && effectiveSaveOptions == CKTEXTURE_INCLUDEORIGINALFILE) {
        for (int i = 0; i < slotCount; ++i) {
            CKSTRING slotFileName = GetSlotFileName(i);
            if (slotFileName && *slotFileName) {
                if (f->IncludeFile(slotFileName, BITMAP_PATH_IDX)) {
                    externalOrIncludedSlots.Set(i); // Mark as "included"
                } else {
                    // If couldn't include, fallback for all
                    effectiveSaveOptions = CKTEXTURE_RAWDATA;
                    break;
                }
            } else {
                // If file should be included but has no name, fallback for all
                effectiveSaveOptions = CKTEXTURE_RAWDATA;
                break;
            }
        }
    }

    // 7. Save data based on the final effectiveSaveOptions
    bool savedViaImageFormat = false;
    if (effectiveSaveOptions == CKTEXTURE_IMAGEFORMAT) {
        if (propertiesForImageFormat) {
            CKPluginManager *pluginManager = CKGetPluginManager();
            CKBitmapReader *reader = pluginManager->GetBitmapReader(
                propertiesForImageFormat->m_Ext,
                &propertiesForImageFormat->m_ReaderGuid
            );

            if (reader) {
                chnk->WriteIdentifier(Identifiers[1]); // IMAGEFORMAT_CHUNK_ID
                chnk->WriteInt(slotCount);
                chnk->WriteInt(imageDescForSaving.Width);
                chnk->WriteInt(imageDescForSaving.Height);
                chnk->WriteInt(imageDescForSaving.BitsPerPixel); // BPP of the source data

                for (int i = 0; i < slotCount; ++i) {
                    // Prepare a temporary VxImageDescEx for this slot's data
                    VxImageDescEx currentSlotDesc = imageDescForSaving; // Copy general desc
                    currentSlotDesc.Image = LockSurfacePtr(i);
                    if (currentSlotDesc.Image) {
                        // WriteReaderBitmap needs to know the format of propertiesForImageFormat.m_Data
                        // but it receives the actual image data via currentSlotDesc.Image.
                        // The propertiesForImageFormat provides the TARGET format info.
                        chnk->WriteReaderBitmap(currentSlotDesc, reader, propertiesForImageFormat);
                        // Assuming LockSurfacePtr/WriteReaderBitmap handles release or next Lock will
                    } else {
                        // Handle LockSurfacePtr failure for a slot if necessary,
                        // e.g., write an empty placeholder or error out.
                        // For now, assume WriteReaderBitmap handles NULL image if that's a valid case for it.
                        chnk->WriteReaderBitmap(currentSlotDesc, reader, propertiesForImageFormat); // Pass NULL image
                    }
                }
                reader->Release();
                savedViaImageFormat = true;
            } else {
                // Reader for specified IMAGEFORMAT not found, fallback to RAWDATA
                effectiveSaveOptions = CKTEXTURE_RAWDATA;
            }
        } else {
            // No propertiesForImageFormat specified, fallback to RAWDATA
            effectiveSaveOptions = CKTEXTURE_RAWDATA;
        }
    }

    // Fallback or direct save as RAWDATA (also used for EXTERNAL/INCLUDE)
    if (!savedViaImageFormat) {
        chnk->WriteIdentifier(Identifiers[2]); // RAWDATA_CHUNK_ID
        chnk->WriteInt(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            VxImageDescEx currentSlotDesc = imageDescForSaving; // Copy general desc
            if (externalOrIncludedSlots.IsSet(i)) {
                currentSlotDesc.Image = nullptr; // Marked as external/included, don't save raw bytes
            } else {
                currentSlotDesc.Image = LockSurfacePtr(i);
            }
            chnk->WriteRawBitmap(currentSlotDesc);
            // Assuming LockSurfacePtr/WriteRawBitmap handles release or next Lock will
        }
    }

    // 8. Write slot filenames
    chnk->WriteIdentifier(Identifiers[3]); // FILENAMES_CHUNK_ID
    chnk->WriteInt(slotCount);
    for (int i = 0; i < slotCount; ++i) {
        chnk->WriteString(CKJustFile(GetSlotFileName(i)));
    }

    return TRUE;
}

CKBOOL CKBitmapData::ReadFromChunk(CKStateChunk *chnk, CKContext *ctx, CKFile *f, CKDWORD Identifiers[5]) {
    XBitArray slotsMissing(1);
    CKBOOL anyDataBlockProcessed = FALSE;

    // for (int s = 0; s < m_Slots.Size(); ++s) {
    //     if (m_Slots[s]) {
    //         m_Slots[s]->m_FileName = "";
    //         VxDeleteAligned(m_Slots[s]->m_DataBuffer);
    //         m_Slots[s]->m_DataBuffer = nullptr;
    //     }
    // }

    // --- 1. Identifiers[1] (Reader-formatted bitmap data) ---
    if (chnk->SeekIdentifier(Identifiers[1])) {
        anyDataBlockProcessed = TRUE;
        VxImageDescEx desc;

        const int slotCount = chnk->ReadInt();
        const int width = chnk->ReadInt();
        const int height = chnk->ReadInt();
        const int bpp = chnk->ReadInt();

        SetSlotCount(slotCount);
        slotsMissing.CheckSize(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            slotsMissing.Set(i);
            if (width > 0 && height > 0) {
                if (CreateImage(width, height, bpp, i)) {
                    GetImageDesc(desc);
                    desc.Image = LockSurfacePtr(i);
                    if (chnk->ReadReaderBitmap(desc)) {
                        slotsMissing.Unset(i);
                        ReleaseSurfacePtr(i);
                    }
                }
            }
        }
    }
    // --- 2. Identifiers[2] (Raw bitmap data) ---
    else if (chnk->SeekIdentifier(Identifiers[2])) {
        anyDataBlockProcessed = TRUE;
        const int slotCount = chnk->ReadInt();
        SetSlotCount(slotCount);
        slotsMissing.CheckSize(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            VxImageDescEx srcDesc;
            VxImageDescEx destDesc;
            
            CKBYTE *srcImageData = chnk->ReadRawBitmap(srcDesc);
            if (srcImageData) {
                slotsMissing.Unset(i);

                CreateImage(srcDesc.Width, srcDesc.Height, srcDesc.BitsPerPixel, i);
                GetImageDesc(destDesc);
                CKBYTE *destImageData = LockSurfacePtr(i);
                if (destImageData) {
                    srcDesc.Image = srcImageData;
                    destDesc.Image = destImageData;
                    VxDoBlitUpsideDown(srcDesc, destDesc);
                    ReleaseSurfacePtr(i);
                }
                delete[] srcImageData;
            } else {
                slotsMissing.Set(i);
            }
        }
    }
    // --- 3. Identifiers[4] (Bitmap2 format - old format) ---
    else if (chnk->SeekIdentifier(Identifiers[4])) {
        anyDataBlockProcessed = TRUE;
        VxImageDescEx desc;

        const int slotCount = chnk->ReadInt();
        SetSlotCount(slotCount);
        slotsMissing.CheckSize(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            CKBYTE *imageData = chnk->ReadBitmap2(desc);
            if (imageData) {
                slotsMissing.Unset(i);
                SetSlotImage(i, imageData, desc);
            } else {
                slotsMissing.Set(i);
            }
        }
    }

    // --- 4. Identifiers[3] (Slot Filenames / External references) ---
    if (chnk->SeekIdentifier(Identifiers[3])) {
        const int slotCount = chnk->ReadInt();
        SetSlotCount(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            XString fileName;
            chnk->ReadString(fileName);
            if (fileName.Length() > 1) {
                const CKBOOL hasDataLoaded = !slotsMissing.IsSet(i);
                if ((hasDataLoaded && anyDataBlockProcessed)) {
                    SetSlotFileName(i, fileName.Str() ? fileName.Str() : "");
                } else {
                    ctx->GetPathManager()->ResolveFileName(fileName, BITMAP_PATH_IDX);
                    if (!LoadSlotImage(fileName, i)) {
                        SetSlotFileName(i, fileName.Str() ? fileName.Str() : "");
                    }
                }
            }
        }
    }

    // --- 5. Identifiers[0] (Movie file reference) ---
    if (chnk->SeekIdentifier(Identifiers[0])) {
        XString movieFile;
        chnk->ReadString(movieFile);
        if (movieFile.Length() > 1) {
            ctx->GetPathManager()->ResolveFileName(movieFile, BITMAP_PATH_IDX);
            LoadMovieFile(movieFile);
        }
    }

    // 6. Update CKBITMAPDATA_INVALID flag based on final dimensions
    if (m_Width == 0 || m_Height == 0) {
        m_BitmapFlags |= CKBITMAPDATA_INVALID;
    }

    return TRUE;
}
