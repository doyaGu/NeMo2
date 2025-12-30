#include "RCKSpriteText.h"

#include "CKStateChunk.h"
#include "CKBitmapData.h"

CK_CLASSID RCKSpriteText::m_ClassID = CKCID_SPRITETEXT;

// Constructor: 0x10061f00
RCKSpriteText::RCKSpriteText(CKContext *Context, CKSTRING name) : RCKSprite(Context, name) {
    m_Text = nullptr;
    m_Font = nullptr;
    m_FontColor = 0xFFFFFFFF; // White
    m_BkColor = 0;
    m_Flags = 1; // Default alignment

    // Clear VX_2DSPRITE flag (bit 8)
    CKDWORD flags = RCK2dEntity::m_Flags;
    flags &= ~0x0100; // Clear bit 8
    RCK2dEntity::m_Flags = flags;

    RCKSprite::SetTransparent(TRUE);
}

// Destructor: 0x10061fb6
RCKSpriteText::~RCKSpriteText() {
    ClearFont();
    delete[] m_Text;
    m_Text = nullptr;
}

// GetClassID: 0x10062519
CK_CLASSID RCKSpriteText::GetClassID() {
    return m_ClassID;
}

// GetMemoryOccupation: 0x10062529
int RCKSpriteText::GetMemoryOccupation() {
    int size = RCKSprite::GetMemoryOccupation() + (sizeof(RCKSpriteText) - sizeof(RCKSprite));
    // Add owned text string memory
    if (m_Text) {
        size += (int)strlen(m_Text) + 1;
    }
    return size;
}

// SetText: 0x10062021
void RCKSpriteText::SetText(CKSTRING text) {
    if (m_Text) {
        delete[] m_Text;
    }
    m_Text = nullptr;

    if (text) {
        size_t len = strlen(text);
        m_Text = new char[len + 1];
        strcpy(m_Text, text);
    }

    Redraw();
}

// GetText: 0x100620ac
CKSTRING RCKSpriteText::GetText() {
    return m_Text;
}

// SetTextColor: 0x100620c0
void RCKSpriteText::SetTextColor(CKDWORD col) {
    m_FontColor = col;
    Redraw();
}

// GetTextColor: 0x100620e1
CKDWORD RCKSpriteText::GetTextColor() {
    return m_FontColor;
}

// SetBackgroundColor: 0x100620f5
void RCKSpriteText::SetBackgroundColor(CKDWORD col) {
    m_BkColor = col;
    Redraw();
}

// GetBackgroundTextColor: 0x10062116
CKDWORD RCKSpriteText::GetBackgroundTextColor() {
    return m_BkColor;
}

// SetFont: 0x10062349
void RCKSpriteText::SetFont(CKSTRING FontName, int FontSize, int Weight, CKBOOL italic, CKBOOL underline) {
    if (m_Font) {
        VxDeleteFont(m_Font);
    }
    m_Font = VxCreateFont(FontName, FontSize, Weight, italic, underline);
    Redraw();
}

// SetAlign: 0x1006212a
void RCKSpriteText::SetAlign(CKSPRITETEXT_ALIGNMENT align) {
    m_Flags &= 0xFFFF0000;
    m_Flags |= (CKDWORD) align & 0xFFFF;
    Redraw();
}

// GetAlign: 0x10062173
CKSPRITETEXT_ALIGNMENT RCKSpriteText::GetAlign() {
    return (CKSPRITETEXT_ALIGNMENT) (m_Flags & 0xFFFF0000);
}

// ClearFont: 0x1006218c
void RCKSpriteText::ClearFont() {
    if (m_Font) {
        VxDeleteFont(m_Font);
    }
    m_Font = 0;
}

// IsUpToDate: 0x10062500
CKBOOL RCKSpriteText::IsUpToDate() {
    return (m_Flags & 0x10000) != 0;
}

// Render: 0x100621c3
CKERROR RCKSpriteText::Render(CKRenderContext *dev) {
    if (!IsUpToDate() && IsVisible()) {
        Redraw();
    }
    return RCK2dEntity::Render(dev);
}

// Redraw: 0x100623a3
void RCKSpriteText::Redraw() {
    int slot = GetCurrentSlot();
    CKRECT rect;
    rect.right = GetWidth();
    rect.bottom = GetHeight();
    rect.left = 0;
    rect.top = 0;

    VxImageDescEx desc;
    m_BitmapData.GetImageDesc(desc);

    CKBYTE *surfacePtr = LockSurfacePtr(slot);
    if (surfacePtr) {
        // Fill with transparent color
        VxFillStructure(desc.Height * desc.Width, surfacePtr, 4, 4, &m_BitmapData.m_TransColor);
        desc.Image = surfacePtr;

        if (m_Text) {
            BITMAP_HANDLE bitmap = VxCreateBitmap(desc);
            if (bitmap) {
                VxDrawBitmapText(bitmap, m_Font, m_Text, &rect, m_Flags, m_BkColor, m_FontColor);
                VxCopyBitmap(bitmap, desc);
                VxDeleteBitmap(bitmap);
            }
        }

        // Mark as up to date
        m_Flags |= 0x10000;
        ReleaseSurfacePtr(slot);
    }
}

// Save: 0x100621ff
CKStateChunk *RCKSpriteText::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_SPRITETEXT, file);
    CKStateChunk *parentChunk = RCK2dEntity::Save(file, flags);

    chunk->StartWrite();
    chunk->AddChunkAndDelete(parentChunk);

    // Write text (identifier 0x1000000)
    chunk->WriteIdentifier(0x1000000);
    chunk->WriteString(m_Text);

    // Write font info (identifier 0x2000000)
    VXFONTINFO fontInfo;
    VxGetFontInfo(m_Font, fontInfo);

    chunk->WriteIdentifier(0x2000000);
    chunk->WriteString((CKSTRING) fontInfo.FaceName.CStr());
    chunk->WriteInt(fontInfo.Height);
    chunk->WriteInt(fontInfo.Weight);
    chunk->WriteInt(fontInfo.Italic);
    chunk->WriteInt(fontInfo.Underline);

    // Write colors (identifier 0x4000000)
    chunk->WriteIdentifier(0x4000000);
    chunk->WriteDword(m_FontColor);
    chunk->WriteDword(m_BkColor);

    chunk->CloseChunk();
    return chunk;
}

// Load: 0x10062547
CKERROR RCKSpriteText::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    RCKSprite::Load(chunk, file);

    // Read text (identifier 0x1000000)
    if (chunk->SeekIdentifier(0x1000000)) {
        CKSTRING text = nullptr;
        chunk->ReadString(&text);
        SetText(text);
        CKDeletePointer(text);
    }

    // Read font info (identifier 0x2000000)
    if (chunk->SeekIdentifier(0x2000000)) {
        CKSTRING fontName = nullptr;
        chunk->ReadString(&fontName);
        int fontSize = chunk->ReadInt();
        int weight = chunk->ReadInt();
        int italic = chunk->ReadInt();
        int underline = chunk->ReadInt();
        SetFont(fontName, fontSize, weight, italic, underline);
        CKDeletePointer(fontName);
    }

    // Read colors (identifier 0x4000000)
    if (chunk->SeekIdentifier(0x4000000)) {
        m_FontColor = chunk->ReadDword();
        m_BkColor = chunk->ReadDword();
    }

    Redraw();
    return CK_OK;
}

// Copy: 0x10062712
CKERROR RCKSpriteText::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = RCKSprite::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKSpriteText &src = static_cast<RCKSpriteText &>(o);
    m_Text = src.m_Text; // Note: shallow copy like original
    m_FontColor = src.m_FontColor;
    m_BkColor = src.m_BkColor;
    m_Font = src.m_Font;
    m_Flags = src.m_Flags;

    return CK_OK;
}

// GetClassName: 0x10062659
CKSTRING RCKSpriteText::GetClassName() {
    return (CKSTRING) "SpriteText";
}

// GetDependenciesCount: returns 0 (no dependencies)
int RCKSpriteText::GetDependenciesCount(int mode) {
    return 0;
}

// GetDependencies: 0x1006266a
CKSTRING RCKSpriteText::GetDependencies(int i, int mode) {
    return nullptr;
}

// Register: 0x10062671
void RCKSpriteText::Register() {
    CKPARAMETERFROMCLASS(RCKSpriteText, CKPGUID_SPRITETEXT);
}

// CreateInstance: 0x100626a2
CKSpriteText *RCKSpriteText::CreateInstance(CKContext *Context) {
    return (CKSpriteText *) new RCKSpriteText(Context, nullptr);
}
