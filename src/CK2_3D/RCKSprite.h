#ifndef RCKSPRITE_H
#define RCKSPRITE_H

#include "CKBitmapData.h"
#include "RCK2dEntity.h"
#include "CKSprite.h"

class RCKSprite : public RCK2dEntity {
    friend class RCKRenderManager;

public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKSprite.h"
#undef CK_3DIMPLEMENTATION

    explicit RCKSprite(CKContext *Context, CKSTRING name = nullptr);
    ~RCKSprite() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    CKERROR Draw(CKRenderContext *dev) override;

    void RestoreInitialSize() override;

    void CopySpriteData(RCKSprite *src);

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKSprite *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    // CKSprite members
    CKBitmapData m_BitmapData;
    VX_PIXELFORMAT m_VideoFormat;
    CKRasterizerContext *m_RasterizerContext;
    CKDWORD m_ObjectIndex;
};

#endif // RCKSPRITE_H
