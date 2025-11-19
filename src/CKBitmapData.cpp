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

CKBOOL CKBitmapData::CreateImage(int Width, int Height, int BPP, int Slot) {
    if ((m_BitmapFlags & CKBITMAPDATA_INVALID) != 0 || (m_Slots.Size() == 1 && Slot == 0)) {
        m_Width = Width;
        m_Height = Height;
        m_BitmapFlags &= ~CKBITMAPDATA_INVALID;
        m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
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
    if (!slot) {
        return FALSE;
    }
    slot->m_FileName = "";
    VxDeleteAligned(slot->m_DataBuffer);
    slot->m_DataBuffer = (CKDWORD *)VxNewAligned(4 * m_Height * m_Width, 16);
    if (!slot->m_DataBuffer) return FALSE;

    CKDWORD *buffer = slot->m_DataBuffer;
    int dwordCount = m_Width * m_Height;
    for (int i = 0; i < dwordCount; ++i) {
        buffer[i] = A_MASK;
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
    CKBitmapReader *reader = nullptr;
    CKBOOL result = FALSE;

    char *fileExt = pathSplitter.GetExtension();
    if (fileExt && *fileExt) {
        snprintf(extension, sizeof(extension), "%s", fileExt);
    }

    CKFileExtension desiredExt(extension + 1); // Skip leading dot
    reader = CKGetPluginManager()->GetBitmapReader(desiredExt, nullptr);
    if (!reader)
        return FALSE;

    CKBitmapProperties *saveProps = nullptr;
    reader->GetBitmapDefaultProperties(&saveProps);

    CKBYTE *surfaceData = LockSurfacePtr(Slot);
    if (!surfaceData) {
        reader->Release();
        return FALSE;
    }

    const int pixelCount = m_Width * m_Height;
    CKDWORD *alphaBuffer = new CKDWORD[pixelCount];

    CKDWORD *src = (CKDWORD *)surfaceData;
    for (int i = 0; i < pixelCount; ++i) {
        CKDWORD alpha = (src[i] >> 24) & 0xFF; // Extract alpha
        alphaBuffer[i] = 0xFF000000 |          // Full opacity
            (alpha << 16) |                    // R channel
            (alpha << 8) |                     // G channel
            alpha;                             // B channel
    }

    VxImageDescEx imgDesc;
    GetImageDesc(imgDesc);

    saveProps->m_Format = imgDesc;
    saveProps->m_Data = alphaBuffer;

    CKPathMaker pathMaker(pathSplitter.GetDrive(), pathSplitter.GetDir(), pathSplitter.GetName(), extension);
    XString outputPath = pathMaker.GetFileName();

    result = reader->SaveFile(outputPath.Str(), saveProps) != 0;

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
    if (Slot < 0 || Slot >= m_Slots.Size())
        Slot = m_CurrentSlot;

    if (m_Slots.IsEmpty())
        return nullptr;

    if (Slot < 0 || Slot >= m_Slots.Size())
        return nullptr;

    CKBitmapSlot *slot = m_Slots[Slot];
    if (!slot)
        return nullptr;

    return (CKBYTE *)slot->m_DataBuffer;
}

CKBOOL CKBitmapData::ReleaseSurfacePtr(int Slot) {
    m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
    return TRUE;
}

CKSTRING CKBitmapData::GetSlotFileName(int Slot) {
    if (Slot < 0 || Slot >= m_Slots.Size())
        return (CKSTRING)"";

    return m_Slots[Slot]->m_FileName.Str();
}

CKBOOL CKBitmapData::SetSlotFileName(int Slot, CKSTRING Filename) {
    if (Slot < 0 || Slot >= m_Slots.Size() || !Filename)
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
    if (m_MovieInfo || Count < 0) return FALSE;

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
        if (Slot < 0 || Slot >= m_Slots.Size()) return FALSE;
        if (Slot != m_CurrentSlot) {
            m_CurrentSlot = Slot;
            if (!(m_BitmapFlags & CKBITMAPDATA_CUBEMAP)) {
                m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
            }
        }
        return TRUE;
    }

    if (Slot == m_MovieInfo->m_MovieCurrentSlot) return TRUE;

    CKMovieReader *reader = m_MovieInfo->m_MovieReader;
    if (!reader) return FALSE;

    if (m_Slots.IsEmpty()) return FALSE;

    CKBitmapSlot *targetSlot = m_Slots[0];
    if (!targetSlot || !targetSlot->m_DataBuffer) return FALSE;

    m_MovieInfo->m_MovieCurrentSlot = Slot;
    CKMovieProperties *movieProps = nullptr;
    if (reader->ReadFrame(Slot, &movieProps) != CK_OK || !movieProps) {
        return FALSE;
    }

    VxImageDescEx srcDesc = movieProps->m_Format;
    VxImageDescEx dstDesc;
    GetImageDesc(dstDesc);
    srcDesc.Image = (CKBYTE *)movieProps->m_Data;
    dstDesc.Image = (CKBYTE *)targetSlot->m_DataBuffer;
    VxDoBlitUpsideDown(srcDesc, dstDesc);
    m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
    return TRUE;
}

int CKBitmapData::GetCurrentSlot() {
    return m_MovieInfo ? m_MovieInfo->m_MovieCurrentSlot : m_CurrentSlot;
}

CKBOOL CKBitmapData::ReleaseSlot(int Slot) {
    if (m_MovieInfo || Slot < 0 || Slot >= m_Slots.Size()) return FALSE;

    CKBitmapSlot *slot = m_Slots[Slot];
    if (slot) {
        delete slot;
    }
    m_Slots.RemoveAt(Slot);

    if (m_CurrentSlot >= m_Slots.Size()) {
        m_CurrentSlot = m_Slots.Size() - 1;
        m_BitmapFlags |= CKBITMAPDATA_FORCERESTORE;
    }
    if (m_Slots.IsEmpty()) m_BitmapFlags |= CKBITMAPDATA_INVALID;
    return TRUE;
}

CKBOOL CKBitmapData::ReleaseAllSlots() {
    if (m_MovieInfo) return FALSE;

    for (int i = 0; i < m_Slots.Size(); ++i) {
        CKBitmapSlot *slot = m_Slots[i];
        if (slot) {
            delete slot;
        }
    }

    m_Slots.Clear();
    m_BitmapFlags |= CKBITMAPDATA_INVALID;
    return TRUE;
}

CKBOOL CKBitmapData::SetPixel(int x, int y, CKDWORD col, int slot) {
    int targetSlot = (slot >= 0) ? slot : m_CurrentSlot;
    if (targetSlot < 0 || targetSlot >= m_Slots.Size() || x < 0 || x >= m_Width || y < 0 || y >= m_Height) return FALSE;

    CKDWORD *buffer = m_Slots[targetSlot]->m_DataBuffer;
    if (!buffer) return FALSE;

    buffer[x + y * m_Width] = col;
    return TRUE;
}

CKDWORD CKBitmapData::GetPixel(int x, int y, int slot) {
    int targetSlot = (slot >= 0) ? slot : m_CurrentSlot;
    if (targetSlot < 0 || targetSlot >= m_Slots.Size() || x < 0 || x >= m_Width || y < 0 || y >= m_Height) return 0;

    CKDWORD *buffer = m_Slots[targetSlot]->m_DataBuffer;
    return buffer ? buffer[x + y * m_Width] : 0;
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
    CKDeletePointer(m_SaveProperties);
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

    VxImageDescEx srcDesc, dstDesc;

    srcDesc.Size = sizeof(VxImageDescEx);
    srcDesc.Width = m_Width;
    srcDesc.Height = m_Height;
    srcDesc.BitsPerPixel = 32;
    srcDesc.BytesPerLine = m_Width * 4;
    srcDesc.RedMask = R_MASK;
    srcDesc.GreenMask = G_MASK;
    srcDesc.BlueMask = B_MASK;
    srcDesc.AlphaMask = A_MASK;

    dstDesc.Size = sizeof(VxImageDescEx);
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
    CKPathSplitter splitter((CKSTRING)Name.CStr());
    CKSTRING extStr = splitter.GetExtension();
    if (!extStr || *extStr == '\0')
        return FALSE;

    CKFileExtension extension(extStr + 1);
    CKPluginManager *pm = CKGetPluginManager();
    CKBitmapReader *reader = pm->GetBitmapReader(extension);
    if (!reader)
        return FALSE;

    CKBitmapProperties *props = nullptr;
    if (reader->ReadFile((CKSTRING)Name.CStr(), &props) != 0 || !props || !props->m_Data) {
        reader->Release();
        return FALSE;
    }

    const int slotCount = GetSlotCount();
    if (Slot > slotCount)
        SetSlotCount(Slot + 1);

    if (!CreateImage(props->m_Format.Width, props->m_Format.Height, 32, Slot)) {
        SetSlotCount(slotCount);
        reader->Release();
        return FALSE;
    }

    CKBYTE *data = LockSurfacePtr(Slot);
    if (!data) {
        reader->ReleaseMemory(props->m_Data);
        props->m_Data = nullptr;

        delete props->m_Format.ColorMap;
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

    SetSlotFileName(Slot, (CKSTRING)Name.CStr());
    SetSaveFormat(props);

    reader->ReleaseMemory(props->m_Data);
    props->m_Data = nullptr;

    delete props->m_Format.ColorMap;
    props->m_Format.ColorMap = nullptr;

    reader->Release();
    return TRUE;
}

CKBOOL CKBitmapData::LoadMovieFile(XString Name) {
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
    if (!CreateImage(movieProps->m_Format.Width, movieProps->m_Format.Height, 32, 0)) {
        delete m_MovieInfo;
        m_MovieInfo = nullptr;
        return FALSE;
    }

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
    if (!reader) {
        delete movieInfo;
        return nullptr;
    }

    if (reader->OpenFile(s.Str()) != CK_OK ||
        reader->ReadFrame(0, mp) != CK_OK) {
        reader->Release();
        movieInfo->m_MovieReader = nullptr;
        delete movieInfo;
        return nullptr;
    }

    movieInfo->m_MovieReader = reader;
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
    CKDeletePointer(m_SaveProperties);
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

    if (!desc.Image || desc.Width < 2 || desc.Height < 2)
        return;

    const int bytesPerLine = desc.BytesPerLine;
    const int width = desc.Width;
    const int height = desc.Height;

    // Process vertical borders (left/right columns)
    for (int y = 0; y < height; ++y) {
        CKDWORD *row = reinterpret_cast<CKDWORD *>(desc.Image + y * bytesPerLine);

        // Left border: copy second column to first
        row[0] = row[1] & 0x00FFFFFF; // Preserve RGB, clear alpha

        // Right border: copy second-last column to last
        row[width - 1] = row[width - 2] & 0x00FFFFFF;
    }

    // Process horizontal borders (top/bottom rows)
    for (int x = 0; x < width; ++x) {
        // Top border: copy second row to first
        CKDWORD *topRow = reinterpret_cast<CKDWORD *>(desc.Image);
        CKDWORD *secondRow = reinterpret_cast<CKDWORD *>(desc.Image + bytesPerLine);
        topRow[x] = secondRow[x] & 0x00FFFFFF;

        // Bottom border: copy second-last row to last
        CKDWORD *bottomRow = reinterpret_cast<CKDWORD *>(desc.Image + (height - 1) * bytesPerLine);
        CKDWORD *penultimateRow = reinterpret_cast<CKDWORD *>(desc.Image + (height - 2) * bytesPerLine);
        bottomRow[x] = penultimateRow[x] & 0x00FFFFFF;
    }
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
    delete[] (XBYTE *)buffer;
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
    VxImageDescEx imageDescForSaving; // Used for Width, Height, BPP, and as a template
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
    if (m_SaveOptions == CKTEXTURE_USEGLOBAL && effectiveSaveOptions == CKTEXTURE_USEGLOBAL) {
        // Check if it's still USEGLOBAL after context check
        effectiveSaveOptions = CKTEXTURE_RAWDATA;
        // Fallback if global options didn't resolve to a concrete type for non-movies
    }
    if (!f) {
        // If f is NULL, we are not saving to a persistent CKFile, must be raw.
        effectiveSaveOptions = CKTEXTURE_RAWDATA;
    }


    XBitArray externalOrIncludedSlots(slotCount > 0 ? slotCount : 1);

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
                if (f->IncludeFile(slotFileName, -1)) {
                    // Assuming -1 for default/all paths search
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

    // Fallback or direct save as RAWDATA
    if (!savedViaImageFormat && effectiveSaveOptions == CKTEXTURE_RAWDATA) {
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
    XBitArray slotsProcessed;
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
        slotsProcessed.CheckSize(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            if (width > 0 && height > 0) {
                if (CreateImage(width, height, bpp, i)) {
                    GetImageDesc(desc);
                    desc.Image = LockSurfacePtr(i);
                    if (chnk->ReadReaderBitmap(desc)) {
                        slotsProcessed.Unset(i);
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
        slotsProcessed.CheckSize(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            VxImageDescEx srcDesc;
            CKBYTE *srcImageData = chnk->ReadRawBitmap(srcDesc);
            if (srcImageData) {
                slotsProcessed.Unset(i);
                if (CreateImage(srcDesc.Width, srcDesc.Height, srcDesc.BitsPerPixel, i)) {
                    VxImageDescEx destDesc;
                    GetImageDesc(destDesc);
                    CKBYTE *destImageData = LockSurfacePtr(i);
                    if (destImageData) {
                        srcDesc.Image = srcImageData;
                        destDesc.Image = destImageData;
                        VxDoBlitUpsideDown(srcDesc, destDesc);
                        ReleaseSurfacePtr(i);
                    }
                }
                delete[] srcImageData;
            } else {
                slotsProcessed.Set(i);
            }
        }
    }
    // --- 3. Identifiers[4] (Bitmap2 format - old format) ---
    else if (chnk->SeekIdentifier(Identifiers[4])) {
        anyDataBlockProcessed = TRUE;
        VxImageDescEx desc;

        const int slotCount = chnk->ReadInt();
        SetSlotCount(slotCount);
        slotsProcessed.CheckSize(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            CKBYTE *imageData = chnk->ReadBitmap2(desc);
            if (imageData) {
                SetSlotImage(i, imageData, desc);
            } else {
                slotsProcessed.Set(i);
            }
        }
    }

    // --- 4. Identifiers[3] (Slot Filenames / External references) ---
    if (chnk->SeekIdentifier(Identifiers[3])) {
        const int slotCount = chnk->ReadInt();
        SetSlotCount(slotCount);
        slotsProcessed.CheckSize(slotCount);

        for (int i = 0; i < slotCount; ++i) {
            XString fileName;
            chnk->ReadString(fileName);
            if (fileName.Length() > 0) {
                bool needsLoadingFromFile = !slotsProcessed.IsSet(i);
                if (!anyDataBlockProcessed) {
                    needsLoadingFromFile = true;
                }

                if (needsLoadingFromFile) {
                    XString resolvedFileName = fileName;
                    ctx->GetPathManager()->ResolveFileName(resolvedFileName, BITMAP_PATH_IDX);
                    if (!LoadSlotImage(resolvedFileName.Str(), i)) {
                        SetSlotFileName(i, resolvedFileName.Str());
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
            XString resolvedMovieFile = movieFile;
            ctx->GetPathManager()->ResolveFileName(resolvedMovieFile, BITMAP_PATH_IDX);
            LoadMovieFile(resolvedMovieFile.Str());
        }
    }

    // 6. Update CKBITMAPDATA_INVALID flag based on final dimensions
    if (m_Width == 0 || m_Height == 0) {
        m_BitmapFlags |= CKBITMAPDATA_INVALID;
    } else {
        m_BitmapFlags &= ~CKBITMAPDATA_INVALID;
    }

    return TRUE;
}
