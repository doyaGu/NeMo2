#ifndef RCKTARGETLIGHT_H
#define RCKTARGETLIGHT_H

#include "RCKLight.h"
#include "CKTargetLight.h"

class RCKTargetLight : public RCKLight {
public:

    explicit RCKTargetLight(CKContext *Context, CKSTRING name = nullptr);
    ~RCKTargetLight() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;
    
    CKBOOL IsObjectUsed(CKObject *obj, CK_CLASSID cid) override;
    
    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    
    // Override GetTarget/SetTarget from CKLight
    CK3dEntity *GetTarget() override;
    void SetTarget(CK3dEntity *target) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKTargetLight *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    CK_ID m_Target; // Target entity ID
};

#endif // RCKTARGETLIGHT_H
