#ifndef RCKTARGETCAMERA_H
#define RCKTARGETCAMERA_H

#include "RCKCamera.h"
#include "CKTargetCamera.h"

class RCKTargetCamera : public RCKCamera {
public:

    explicit RCKTargetCamera(CKContext *Context, CKSTRING name = nullptr);
    ~RCKTargetCamera() override;
    CK_CLASSID GetClassID() override;

    void CheckPostDeletion() override;
    CKBOOL IsObjectUsed(CKObject *o, CK_CLASSID cid) override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    void AddToScene(CKScene *scene, CKBOOL dependencies = TRUE) override;
    void RemoveFromScene(CKScene *scene, CKBOOL dependencies = TRUE) override;
    
    // Override GetTarget/SetTarget from CKCamera
    CK3dEntity *GetTarget() override;
    void SetTarget(CK3dEntity *target) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKTargetCamera *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    CK_ID m_Target; // Target entity ID
};

#endif // RCKTARGETCAMERA_H
