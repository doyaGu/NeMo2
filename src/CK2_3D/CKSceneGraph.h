#ifndef CKSCENEGRAPH_H
#define CKSCENEGRAPH_H

#include "CKRenderEngineTypes.h"

class RCKRenderContext;
class RCKPlace;

//--- CKSceneGraphNode flags
// Bitmask stored in CKSceneGraphNode::m_Flags.
typedef enum CKSCENEGRAPHNODE_FLAGS
{
    // Visibility / frustum test state (bits 0-1)
    CKSGN_INSIDEFRUSTUM        = 0x00000001, // Set when inside frustum
    CKSGN_OUTSIDEFRUSTUM       = 0x00000002, // Set when outside frustum
    CKSGN_FRUSTUM_MASK         = 0x00000003,

    // Hierarchical bounding box state (bits 2-3)
    CKSGN_BOXVALID             = 0x00000004, // Hierarchy bbox valid
    CKSGN_BOXCOMPUTED          = 0x00000008, // Hierarchy bbox computed
    CKSGN_BOX_MASK             = 0x0000000C,

    // Sorting / bookkeeping
    CKSGN_NEEDSORT             = 0x00000010, // Children need sorting
    CKSGN_INTRANSPARENTLIST    = 0x00000020, // Added to transparent object list
} CKSCENEGRAPHNODE_FLAGS;

// Forward declaration
struct CKSceneGraphNode;

// Transparent object entry for sorting
struct CKTransparentObject {
    CKSceneGraphNode *m_Node; // The scene graph node
    // NOTE: Field order must match the original CK2_3D.dll layout (12 bytes total).
    // IDA shows VxProjectBoxZExtents writes ZhMin at +4 and ZhMax at +8.
    float m_ZhMin;            // Minimum Z in homogeneous coordinates (for depth sorting)
    float m_ZhMax;            // Maximum Z in homogeneous coordinates
};

struct CKSceneGraphNode {
    explicit CKSceneGraphNode(RCK3dEntity *entity = nullptr);
    ~CKSceneGraphNode();

    void AddNode(CKSceneGraphNode *node);
    void RemoveNode(CKSceneGraphNode *node);
    void PrioritiesChanged();
    void SetRenderContextMask(CKDWORD mask, CKBOOL force);
    void EntityFlagsChanged(CKBOOL updateParent);
    void InvalidateBox(CKBOOL propagate);
    void SetPriority(int priority, CKBOOL propagate);
    CKBOOL IsToBeParsed();
    CKDWORD Rebuild();
    CKDWORD ComputeHierarchicalBox();

    // Rendering traversal
    void NoTestsTraversal(RCKRenderContext *dev, CKDWORD flags);
    void SetAsPotentiallyVisible();
    void SetAsInsideFrustum();
    void SetAsOutsideFrustum();
    void SortNodes();
    void ClearTransparentFlags();
    CKBOOL CheckHierarchyFrustum();
    CKBOOL IsAllOutsideFrustum() const;

    // Flags helpers
    CKDWORD GetFlags() const { return m_Flags; }
    void SetFlags(CKDWORD flags) { m_Flags |= flags; }
    void ClearFlags(CKDWORD flags) { m_Flags &= ~flags; }
    void SetFlagState(CKDWORD flags, CKBOOL enabled) { enabled ? SetFlags(flags) : ClearFlags(flags); }
    CKBOOL HasAnyFlags(CKDWORD flags) const { return (m_Flags & flags) != 0; }
    CKBOOL HasAllFlags(CKDWORD flags) const { return (m_Flags & flags) == flags; }

    CKBOOL NeedsSort() const { return HasAnyFlags((CKDWORD)CKSGN_NEEDSORT); }
    void MarkNeedSort() { SetFlags((CKDWORD)CKSGN_NEEDSORT); }
    void ClearNeedSort() { ClearFlags((CKDWORD)CKSGN_NEEDSORT); }

    CKBOOL IsInTransparentList() const { return HasAnyFlags((CKDWORD)CKSGN_INTRANSPARENTLIST); }
    void MarkInTransparentList() { SetFlags((CKDWORD)CKSGN_INTRANSPARENTLIST); }
    void ClearInTransparentList() { ClearFlags((CKDWORD)CKSGN_INTRANSPARENTLIST); }

    CKBOOL IsHierarchyBoxComputed() const { return HasAnyFlags((CKDWORD)CKSGN_BOXCOMPUTED); }
    CKBOOL IsHierarchyBoxValid() const { return HasAnyFlags((CKDWORD)CKSGN_BOXVALID); }
    void InvalidateHierarchyBox() { ClearFlags((CKDWORD)CKSGN_BOX_MASK); }

    RCK3dEntity *m_Entity;
    CKDWORD m_TimeFpsCalc;
    CKDWORD m_Flags;
    int m_Index;
    VxBbox m_Bbox;
    short m_Priority;
    short m_MaxPriority;
    CKDWORD m_RenderContextMask;
    CKDWORD m_EntityMask;
    CKSceneGraphNode *m_Parent;
    XArray<CKSceneGraphNode *> m_Children;
    int m_ChildToBeParsedCount;
};

struct CKSceneGraphRootNode : CKSceneGraphNode {
    void RenderTransparentObjects(RCKRenderContext *rc, CKDWORD flags);
    void SortTransparentObjects(RCKRenderContext *rc, CKDWORD flags);
    void AddTransparentObject(CKSceneGraphNode *node);

    void Clear();
    void Check();

    XClassArray<CKTransparentObject> m_TransparentObjects; // Transparent objects for sorting
};

#endif // CKSCENEGRAPH_H
