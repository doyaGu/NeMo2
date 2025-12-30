#include "RCK3dObject.h"
#include "CKContext.h"

CK_CLASSID RCK3dObject::m_ClassID = CKCID_3DOBJECT;

RCK3dObject::RCK3dObject(CKContext *Context, CKSTRING name) : RCK3dEntity(Context, name) {}

RCK3dObject::~RCK3dObject() {}

CK_CLASSID RCK3dObject::GetClassID() {
    return m_ClassID;
}

//=============================================================================
// Static Class Methods (for class registration)
// Based on IDA Pro analysis of original CK2_3D.dll
//=============================================================================

CKSTRING RCK3dObject::GetClassName() {
    return (CKSTRING) "3D Object";
}

int RCK3dObject::GetDependenciesCount(int mode) {
    return 0;
}

CKSTRING RCK3dObject::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCK3dObject::Register() {
    // Register associated parameter GUID
    CKPARAMETERFROMCLASS(RCK3dObject, CKPGUID_OBJECT3D);
}

CK3dObject *RCK3dObject::CreateInstance(CKContext *Context) {
    RCK3dObject *obj = new RCK3dObject(Context, nullptr);
    return reinterpret_cast<CK3dObject *>(obj);
}
