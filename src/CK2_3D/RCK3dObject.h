#ifndef RCK3DOBJECT_H
#define RCK3DOBJECT_H

#include "RCK3dEntity.h"

class RCK3dObject : public RCK3dEntity {
public:
    explicit RCK3dObject(CKContext *Context, CKSTRING name = nullptr);
    ~RCK3dObject() override;
    CK_CLASSID GetClassID() override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CK3dObject *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;
};

#endif // RCK3DOBJECT_H
