#ifndef RCKTEXTURE_H
#define RCKTEXTURE_H

#include "CKTexture.h"

class RCKTexture : public CKTexture {
    friend class RCKRenderManager;

public:
    CKBOOL Create(int Width, int Height, int BPP, int Slot) override;
    CKBOOL LoadImage(CKSTRING Name, int Slot) override;
    CKBOOL LoadMovie(CKSTRING Name) override;
    CKBOOL SetAsCurrent(CKRenderContext *Dev, CKBOOL Clamping, int TextureStage) override;
    CKBOOL Restore(CKBOOL Clamp) override;
    CKBOOL SystemToVideoMemory(CKRenderContext *Dev, CKBOOL Clamping) override;
    CKBOOL FreeVideoMemory() override;
    CKBOOL IsInVideoMemory() override;
    CKBOOL CopyContext(CKRenderContext *ctx, VxRect *Src, VxRect *Dest, int CubeMapFace) override;
    CKBOOL UseMipmap(int UseMipMap) override;
    int GetMipmapCount() override;
    CKBOOL GetVideoTextureDesc(VxImageDescEx &desc) override;
    VX_PIXELFORMAT GetVideoPixelFormat() override;
    CKBOOL GetSystemTextureDesc(VxImageDescEx &desc) override;
    void SetDesiredVideoFormat(VX_PIXELFORMAT Format) override;
    VX_PIXELFORMAT GetDesiredVideoFormat() override;
    CKBOOL SetUserMipMapMode(CKBOOL UserMipmap) override;
    CKBOOL GetUserMipMapLevel(int Level, VxImageDescEx &ResultImage) override;
    int GetRstTextureIndex() override;

    explicit RCKTexture(CKContext *Context, CKSTRING name = nullptr);
    ~RCKTexture() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKTexture *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    VX_PIXELFORMAT m_DesiredVideoFormat;
    CKRasterizerContext *m_RasterizerContext;
    CKDWORD m_MipMapLevel;
    CKDWORD m_ObjectIndex;
    XClassArray<VxImageDescEx> *m_MipMaps;
};

#endif // RCKTEXTURE_H
