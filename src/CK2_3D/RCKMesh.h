#ifndef RCKMESH_H
#define RCKMESH_H

#include "CKRenderEngineTypes.h"

#include "CKMesh.h"

// Forward declarations for generic functions
void BuildFaceNormalsGenericFunc(CKFace *faces, CKWORD *indices, int faceCount, VxVertex *vertices, int vertexCount);
void BuildNormalsGenericFunc(CKFace *faces, CKWORD *indices, int faceCount, VxVertex *vertices, int vertexCount);
int RayIntersectionGenericFunc(RCKMesh *mesh, VxVector &origin, VxVector &direction, VxIntersectionDesc *desc, CK_RAYINTERSECTION mode, const VxMatrix &worldMatrix);
void NormalizeGenericFunc(VxVertex *vertices, int count);

class RCKMesh : public CKMesh {
    // Friend function for ray intersection (needs access to protected members)
    friend int RayIntersectionGenericFunc(RCKMesh *mesh, VxVector &origin, VxVector &direction, 
                                               VxIntersectionDesc *desc, CK_RAYINTERSECTION mode, 
                                               const VxMatrix &worldMatrix);
public:

    //--------------------------------------------------------
    ////               Private Part

    //----------------------------------------------------------
    // Internal functions
    void Show(CK_OBJECT_SHOWOPTION show = CKSHOW) override;
    
    // Add all virtual function declarations from base CKMesh class
    virtual CKBOOL IsTransparent() override;
    virtual void SetTransparent(CKBOOL Transparency) override;
    virtual void SetWrapMode(VXTEXTURE_WRAPMODE Mode) override;
    virtual VXTEXTURE_WRAPMODE GetWrapMode() override;
    virtual void SetLitMode(VXMESH_LITMODE Mode) override;
    virtual VXMESH_LITMODE GetLitMode() override;
    virtual CKDWORD GetFlags() override;
    virtual void SetFlags(CKDWORD Flags) override;
    virtual void *GetPositionsPtr(CKDWORD *Stride) override;
    virtual void *GetNormalsPtr(CKDWORD *Stride) override;
    virtual void *GetColorsPtr(CKDWORD *Stride) override;
    virtual void *GetSpecularColorsPtr(CKDWORD *Stride) override;
    virtual void *GetTextureCoordinatesPtr(CKDWORD *Stride, int channel) override;
    virtual void VertexMove() override;
    virtual void UVChanged() override;
    virtual void NormalChanged() override;
    virtual void ColorChanged() override;
    virtual void BuildNormals() override;
    virtual void BuildFaceNormals() override;
    virtual CKBOOL SetVertexCount(int Count) override;
    virtual int GetVertexCount() override;
    virtual void SetVertexPosition(int Index, VxVector *Vector) override;
    virtual void SetVertexNormal(int Index, VxVector *Vector) override;
    virtual void SetVertexColor(int Index, CKDWORD Color) override;
    virtual void SetVertexTextureCoordinates(int Index, float u, float v, int channel) override;
    virtual void GetVertexPosition(int Index, VxVector *Vector) override;
    virtual void GetVertexNormal(int Index, VxVector *Vector) override;
    virtual CKDWORD GetVertexColor(int Index) override;
    virtual void GetVertexTextureCoordinates(int Index, float *u, float *v, int channel) override;
    virtual CKBOOL SetFaceCount(int Count) override;
    virtual int GetFaceCount() override;
    virtual void SetFaceVertexIndex(int FaceIndex, int Vertex1, int Vertex2, int Vertex3) override;
    virtual void GetFaceVertexIndex(int FaceIndex, int &Vertex1, int &Vertex2, int &Vertex3) override;
    virtual void SetFaceMaterial(int FaceIndex, CKMaterial *Mat) override;
    virtual CKMaterial *GetFaceMaterial(int Index) override;
    virtual CKWORD *GetFacesIndices() override;
    virtual float GetRadius() override;
    virtual const VxBbox &GetLocalBox() override;
    virtual void GetBaryCenter(VxVector *Vector) override;
    virtual CKBOOL SetLineCount(int Count) override;
    virtual int GetLineCount() override;
    virtual void SetLine(int LineIndex, int VIndex1, int VIndex2) override;
    virtual void GetLine(int LineIndex, int *VIndex1, int *VIndex2) override;
    
    // Add missing virtual functions from base CKMesh class
    virtual CKWORD *GetLineIndices() override;
    virtual void SetVertexWeightsCount(int count) override;
    virtual int GetVertexWeightsCount() override;
    virtual void SetVertexWeight(int index, float w) override;
    virtual float GetVertexWeight(int index) override;
    virtual void Clean(CKBOOL KeepVertices) override;
    virtual void InverseWinding() override;
    virtual void Consolidate() override;
    virtual void UnOptimize() override;
    virtual CKBOOL AddPreRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument, CKBOOL Temporary) override;
    virtual CKBOOL RemovePreRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument) override;
    virtual CKBOOL AddPostRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument, CKBOOL Temporary) override;
    virtual CKBOOL RemovePostRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument) override;
    virtual void SetRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument) override;
    virtual void SetDefaultRenderCallBack() override;
    virtual void RemoveAllCallbacks() override;
    virtual CKBOOL AddSubMeshPreRenderCallBack(CK_SUBMESHRENDERCALLBACK Function, void *Argument, CKBOOL Temporary) override;
    virtual CKBOOL RemoveSubMeshPreRenderCallBack(CK_SUBMESHRENDERCALLBACK Function, void *Argument) override;
    virtual CKBOOL AddSubMeshPostRenderCallBack(CK_SUBMESHRENDERCALLBACK Function, void *Argument, CKBOOL Temporary) override;
    virtual CKBOOL RemoveSubMeshPostRenderCallBack(CK_SUBMESHRENDERCALLBACK Function, void *Argument) override;
    virtual int GetMaterialCount() override;
    virtual CKMaterial *GetMaterial(int index) override;
    virtual void SetVerticesRendered(int count) override;
    virtual int GetVerticesRendered() override;
    virtual void EnablePMGeoMorph(CKBOOL enable) override;
    virtual CKBOOL IsPMGeoMorphEnabled() override;
    virtual void SetPMGeoMorphStep(int gs) override;
    virtual int GetPMGeoMorphStep() override;

    virtual CKBYTE *GetModifierVertices(CKDWORD *Stride) override;
    virtual int GetModifierVertexCount() override;
    virtual void ModifierVertexMove(CKBOOL RebuildNormals, CKBOOL RebuildFaceNormals) override;
    virtual CKBYTE *GetModifierUVs(CKDWORD *Stride, int channel = -1) override;
    virtual int GetModifierUVCount(int channel = -1) override;
    virtual void ModifierUVMove() override;
    virtual void SetVertexSpecularColor(int Index, CKDWORD Color) override;
    virtual CKDWORD GetVertexSpecularColor(int Index) override;
    virtual void TranslateVertices(VxVector *Vector) override;
    virtual void ScaleVertices(VxVector *Vector, VxVector *Pivot = NULL) override;
    virtual void ScaleVertices3f(float X, float Y, float Z, VxVector *Pivot = NULL) override;
    virtual void RotateVertices(VxVector *Vector, float Angle) override;
    virtual const VxVector &GetFaceNormal(int Index) override;
    virtual CKWORD GetFaceChannelMask(int FaceIndex) override;
    virtual VxVector &GetFaceVertex(int FaceIndex, int VIndex) override;
    virtual CKBYTE *GetFaceNormalsPtr(CKDWORD *Stride) override;
    virtual void SetFaceMaterialEx(int *FaceIndices, int FaceCount, CKMaterial *Mat) override;
    virtual void SetFaceChannelMask(int FaceIndex, CKWORD ChannelMask) override;
    virtual void ReplaceMaterial(CKMaterial *oldMat, CKMaterial *newMat) override;
    virtual void ChangeFaceChannelMask(int FaceIndex, CKWORD AddChannelMask, CKWORD RemoveChannelMask) override;
    virtual void ApplyGlobalMaterial(CKMaterial *Mat) override;
    virtual void DissociateAllFaces() override;
    virtual void CreateLineStrip(int StartingLine, int Count, int StartingVertexIndex) override;
    virtual int GetChannelCount() override;
    virtual int AddChannel(CKMaterial *Mat, CKBOOL CopySrcUv = TRUE) override;
    virtual void RemoveChannelByMaterial(CKMaterial *Mat) override;
    virtual void RemoveChannel(CKMaterial *material);
    virtual void RemoveChannel(int Index) override;
    virtual int GetChannelByMaterial(CKMaterial *mat) override;
    virtual void ActivateChannel(int Index, CKBOOL Active = TRUE) override;
    virtual CKBOOL IsChannelActive(int Index) override;
    virtual void ActivateAllChannels(CKBOOL Active = TRUE) override;
    virtual void LitChannel(int Index, CKBOOL Lit = TRUE) override;
    virtual CKBOOL IsChannelLit(int Index) override;
    virtual CKDWORD GetChannelFlags(int Index) override;
    virtual void SetChannelFlags(int Index, CKDWORD Flags) override;
    virtual CKMaterial *GetChannelMaterial(int Index) override;
    virtual VXBLEND_MODE GetChannelSourceBlend(int Index) override;
    virtual VXBLEND_MODE GetChannelDestBlend(int Index) override;
    virtual void SetChannelMaterial(int Index, CKMaterial *Mat) override;
    virtual void SetChannelSourceBlend(int Index, VXBLEND_MODE BlendMode) override;
    virtual void SetChannelDestBlend(int Index, VXBLEND_MODE BlendMode) override;
    virtual CKERROR Render(CKRenderContext *Dev, CK3dEntity *Mov) override;
    virtual float *GetVertexWeightsPtr() override;
    virtual void LoadVertices(CKStateChunk *chunk) override;
    virtual CKERROR CreatePM() override;
    virtual void DestroyPM() override;
    virtual CKBOOL IsPM() override;

    CKDWORD GetSaveFlags();
    void UpdateBoundingVolumes(CKBOOL force);
    int CreateNewMaterialGroup(CKMaterial *mat);
    int GetMaterialGroupIndex(CKMaterial *mat, CKBOOL create);
    void DeleteRenderGroup();

    // Picking methods
    CKBOOL Pick2D(const Vx2DVector &pt, VxIntersectionDesc *desc, RCKRenderContext *rc, RCK3dEntity *ent);

    // Internal rendering methods
    int DefaultRender(RCKRenderContext *rc, RCK3dEntity *ent);
    int RenderGroup(RCKRenderContext *dev, CKMaterialGroup *group, RCK3dEntity *ent, VxDrawPrimitiveData *data);
    int RenderChannels(RCKRenderContext *dev, RCK3dEntity *ent, VxDrawPrimitiveData *data, int fogEnable);
    int CreateRenderGroups();
    void UpdateChannelIndices();
    CKBOOL CheckHWVertexBuffer(CKRasterizerContext *rst, VxDrawPrimitiveData *data);
    CKBOOL CheckHWIndexBuffer(CKRasterizerContext *rst);

    // Progressive mesh rendering (IDA: 0x100257b1)
    void BuildRenderMesh();

    // Render-group remap helpers (IDA: VBuffer stored in CKMaterialGroup::m_RemapData)
    CKVBuffer *GetVBuffer(CKMaterialGroup *group) const;
    void DeleteVBuffer(CKMaterialGroup *group);
    void ResetMaterialGroup(CKMaterialGroup *group, int a2);
    void UpdateHasValidPrimitives(CKMaterialGroup *group);

    explicit RCKMesh(CKContext *Context, CKSTRING name = nullptr);
    ~RCKMesh() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;
    
    // Internal vertex loading helper for optimized buffer operations
    int ILoadVertices(CKStateChunk *chunk, CKDWORD *loadFlags);

    void CheckPreDeletion() override;

    int GetMemoryOccupation() override;
    int IsObjectUsed(CKObject *o, CK_CLASSID cid) override;

    //--------------------------------------------
    // Dependencies Functions	{Secret}
    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    void AddToScene(CKScene *scene, CKBOOL dependencies = TRUE) override;
    void RemoveFromScene(CKScene *scene, CKBOOL dependencies = TRUE) override;

    //--------------------------------------------
    // Class Registering	{Secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKMesh *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    CKDWORD m_Flags;
    VxVector m_BaryCenter;
    float m_Radius;
    VxBbox m_LocalBox;
    XArray<CKFace> m_Faces;
    XArray<CKWORD> m_FaceVertexIndices;
    XArray<CKWORD> m_LineIndices;
    XArray<VxVertex> m_Vertices;
    XArray<float> *m_VertexWeights;
    XArray<VxColors> m_VertexColors;
    CKDWORD m_DrawFlags;
    CKDWORD m_FaceChannelMask;
    XClassArray<VxMaterialChannel> m_MaterialChannels;
    XArray<int> m_ActiveTextureChannels; // Extra texture coordinate channel indices used for multitexture
    XArray<CKMaterialGroup *> m_MaterialGroups;
    CKDWORD m_Valid;
    CKDWORD m_VertexBufferReady; // Non-zero when HW vertex buffer is up to date
    CKDWORD m_VertexBuffer;
    CKDWORD m_IndexBuffer;
    CKProgressiveMesh *m_ProgressiveMesh;
    CKCallbacksContainer *m_RenderCallbacks;
    CKCallbacksContainer *m_SubMeshCallbacks;
};

#endif // RCKMESH_H
