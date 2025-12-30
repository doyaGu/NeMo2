#ifndef RCKLIGHT_H
#define RCKLIGHT_H

#include "CKRenderEngineTypes.h"

#include "RCK3dEntity.h"
#include "CKLight.h"

class RCKLight : public RCK3dEntity {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKLight.h"
#undef CK_3DIMPLEMENTATION

    explicit RCKLight(CKContext *Context, CKSTRING name = nullptr);
    ~RCKLight() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKLight *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    CKBOOL Setup(CKRasterizerContext *rst, CKDWORD lightIndex);

protected:
    CKLightData m_LightData;
    CKDWORD m_Flags;
    float m_LightPower;
};

#endif // RCKLIGHT_H
