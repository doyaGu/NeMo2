#ifndef RCKSPRITETEXT_H
#define RCKSPRITETEXT_H

#include "RCKSprite.h"

class RCKSpriteText : public RCKSprite {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKSpriteText.h"
#undef CK_3DIMPLEMENTATION

    explicit RCKSpriteText(CKContext *Context, CKSTRING name = nullptr);
    ~RCKSpriteText() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    CKERROR Render(CKRenderContext *Dev) override;

    int GetMemoryOccupation() override;

    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKSpriteText *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    void ClearFont();
    void Redraw();
    CKBOOL IsUpToDate();

    CKSTRING m_Text;
    CKDWORD m_FontColor;
    CKDWORD m_BkColor;
    FONT_HANDLE m_Font;
    CKDWORD m_Flags;
};

#endif // RCKSPRITETEXT_H
