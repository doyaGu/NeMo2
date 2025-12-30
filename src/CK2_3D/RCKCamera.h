#ifndef RCKCAMERA_H
#define RCKCAMERA_H

#include "RCK3dEntity.h"
#include "CKCamera.h"

class RCKCamera : public RCK3dEntity {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKCamera.h"
#undef CK_3DIMPLEMENTATION

    //--------------------------------------------------------
    ////               Private Part

    //----------------------------------------------------------
    // Internal functions

    explicit RCKCamera(CKContext *Context, CKSTRING name = nullptr);
    ~RCKCamera() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    //--------------------------------------------
    // Dependencies Functions	{Secret}
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    //--------------------------------------------
    // Class Registering	{Secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKCamera *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    float m_Fov;
    float m_FrontPlane;
    float m_BackPlane;
    CKDWORD m_ProjectionType;
    float m_OrthographicZoom;
    int m_Width;
    int m_Height;
};

#endif // RCKCAMERA_H
