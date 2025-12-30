#ifndef RCKBODYPART_H
#define RCKBODYPART_H

#include "RCK3dObject.h"
#include "CKBodyPart.h"

class RCKBodyPart : public RCK3dObject {
    friend class RCKCharacter;  // Allow RCKCharacter to access protected members
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKBodyPart.h"
#undef CK_3DIMPLEMENTATION

    explicit RCKBodyPart(CKContext *Context, CKSTRING name = nullptr);
    ~RCKBodyPart() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKBodyPart *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
     RCKCharacter *m_Character;
     RCKAnimation *m_ExclusiveAnimation;
     CKIkJoint m_RotationJoint;
};

#endif // RCKBODYPART_H
