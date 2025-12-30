#ifndef RCK3DENTITY_H
#define RCK3DENTITY_H

#include "CKRenderEngineTypes.h"

#include "RCKRenderObject.h"
#include "CK3dEntity.h"
#include "CKSceneGraph.h"
#include "VxQuaternion.h"
#include "VxRect.h"
#include "VxMatrix.h"

class RCKSkin;

class RCK3dEntity : public RCKRenderObject {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CK3dEntity.h"
#undef CK_3DIMPLEMENTATION

    //--------------------------------------------------------
    ////               Private Part

    //----------------------------------------------------------
    // Internal functions
    CKBOOL IsHiddenByParent() override;
    CKBOOL IsVisible() override;
    void Show(CK_OBJECT_SHOWOPTION show) override;

    explicit RCK3dEntity(CKContext *Context, CKSTRING name = nullptr);
    ~RCK3dEntity() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;
    void PostLoad() override;

    void PreDelete() override;
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

    CKBOOL IsToBeRendered() override;
    void SetZOrder(int Z) override;
    int GetZOrder() override;
    CKBOOL IsToBeRenderedLast() override;
    
    // Matrix change propagation methods
    void WorldMatrixChanged(int updateChildren, int keepScale);
    void LocalMatrixChanged(int updateChildren, int keepScale);
    void WorldPositionChanged(int updateChildren, int keepScale);
    
    // Save the current world matrix as the last frame matrix
    void SaveLastFrameMatrix();
    
    // Place hierarchy management
    void UpdatePlace(CK_ID placeId);

    //--------------------------------------------
    // Class Registering	{Secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CK3dEntity *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    // --- Layout verified against IDA: sizeof(RCK3dEntity) = 424 bytes (0x1A8) ---
    // Offset 0x58: Parent entity pointer
    RCK3dEntity *m_Parent;
    // Offset 0x5C: Array of mesh objects (12 bytes: begin, end, alloc)
    XObjectPointerArray m_Meshes;
    // Offset 0x68: Current active mesh
    RCKMesh *m_CurrentMesh;
    // Offset 0x6C: Reference place (CKPlace*)
    CK_ID m_Place;
    // Offset 0x70: Object animations array pointer (4 bytes)
    // IDA incorrectly shows XSObjectPointerArray but it's actually a pointer to heap-allocated array
    XObjectPointerArray *m_ObjectAnimations;
    // Offset 0x74: 3D Entity flags (CK_3DENTITY_FLAGS) - stores presence indicators and state flags
    // This includes CK_3DENTITY_PLACEVALID (0x10000), CK_3DENTITY_PARENTVALID (0x20000), etc.
    CKDWORD m_3dEntityFlags;
    // Offset 0x78: Pointer to last frame matrix (dynamically allocated)
    VxMatrix *m_LastFrameMatrix;
    // Offset 0x7C: Skin for skeletal animation
    RCKSkin *m_Skin;
    // Offset 0x80: Local transformation matrix (64 bytes)
    VxMatrix m_LocalMatrix;
    // Offset 0xC0: World transformation matrix (64 bytes)
    VxMatrix m_WorldMatrix;
    // Offset 0x100: Moveable flags (visibility, rendering hints, etc.)
    CKDWORD m_MoveableFlags;
    // Offset 0x104: Cached inverse world matrix (64 bytes)
    VxMatrix m_InverseWorldMatrix;
    // Offset 0x144: Child entities array (8 bytes: begin, end)
    XSObjectPointerArray m_Children;
    // Offset 0x14C: Local space bounding box (24 bytes)
    VxBbox m_LocalBoundingBox;
    // Offset 0x164: World space bounding box (24 bytes)
    VxBbox m_WorldBoundingBox;
    // Offset 0x17C: Hierarchical bounding box including children (24 bytes)
    VxBbox m_HierarchicalBox;
    // Offset 0x194: 2D screen render extents (16 bytes)
    VxRect m_RenderExtents;
    // Offset 0x1A4: Scene graph node for render ordering
    CKSceneGraphNode *m_SceneGraphNode;
};

#endif // RCK3DENTITY_H
