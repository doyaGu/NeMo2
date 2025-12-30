#include "RCKSprite3D.h"

#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKSceneGraph.h"
#include "CKRasterizer.h"
#include "CKRasterizerTypes.h"
#include "RCK3dEntity.h"
#include "RCKMaterial.h"
#include "RCKRenderContext.h"

extern CKBOOL PreciseTexturePick(CKMaterial *mat, float u, float v);

/*************************************************
Summary: PreSave method for RCKSprite3D.
Purpose: Prepares sprite for saving by handling material dependencies.
Remarks:
- Calls base class RCK3dEntity::PreSave() first to handle entity dependencies
- Saves the material object for later serialization
- Ensures material is properly saved with the sprite data

Implementation based on decompilation at 0x1004326A:
- Simple implementation that only saves the material object
- Uses CKFile::SaveObject for dependency handling

Arguments:
- file: The file context for saving
- flags: Save flags controlling behavior and dependency handling
*************************************************/
void RCKSprite3D::PreSave(CKFile *file, CKDWORD flags) {
    // Call base class PreSave first to handle entity dependencies
    RCK3dEntity::PreSave(file, flags);

    // Save material object
    file->SaveObject((CKObject *) m_Material, flags);
}

/*************************************************
Summary: Save method for RCKSprite3D.
Purpose: Saves sprite data to a state chunk including mode, size, offset, UV mapping, and material.
Remarks:
- Calls base class RCK3dEntity::Save() first to handle entity data
- Creates sprite-specific state chunk with identifier 0x400000
- Saves sprite mode (billboard, fixed, etc.)
- Calculates and saves sprite size from bounding box
- Saves offset position for sprite placement
- Saves UV mapping rectangle coordinates
- Saves material object reference

Implementation based on decompilation at 0x1004329D:
- Uses chunk identifier 0x400000 for sprite-specific data
- Calculates size from local bounding box dimensions
- Saves offset as 2D vector
- Saves UV mapping rectangle as four floats
- Saves material object reference

Arguments:
- file: The file context for saving (may be NULL for memory operations)
- flags: Save flags controlling behavior and data inclusion
Return Value:
- CKStateChunk*: The created state chunk containing sprite data
*************************************************/
CKStateChunk *RCKSprite3D::Save(CKFile *file, CKDWORD flags) {
    // Call base class save first to handle entity data
    CKStateChunk *baseChunk = RCK3dEntity::Save(file, flags);

    // IDA: early return when saving without file and no class-specific save flags are set
    if (!file && !(flags & CK_STATESAVE_SPRITE3DONLY)) {
        return baseChunk;
    }

    // Create sprite-specific state chunk
    CKStateChunk *spriteChunk = CreateCKStateChunk(CKCID_SPRITE3D, file);
    if (!spriteChunk) {
        return baseChunk;
    }

    spriteChunk->StartWrite();
    spriteChunk->AddChunkAndDelete(baseChunk);

    // Write sprite-specific data with identifier 0x400000
    spriteChunk->WriteIdentifier(CK_STATESAVE_SPRITE3DDATA);
    spriteChunk->WriteDword(m_Mode);

    // Calculate size from bounding box (half width and height)
    float halfWidth = (m_LocalBoundingBox.Max.x - m_LocalBoundingBox.Min.x) * 0.5f;
    float halfHeight = (m_LocalBoundingBox.Max.y - m_LocalBoundingBox.Min.y) * 0.5f;
    spriteChunk->WriteFloat(halfWidth);
    spriteChunk->WriteFloat(halfHeight);

    // Save offset position
    spriteChunk->WriteFloat(m_Offset.x);
    spriteChunk->WriteFloat(m_Offset.y);

    // Save UV mapping rectangle
    spriteChunk->WriteFloat(m_Rect.left);
    spriteChunk->WriteFloat(m_Rect.top);
    spriteChunk->WriteFloat(m_Rect.right);
    spriteChunk->WriteFloat(m_Rect.bottom);

    // Save material object
    spriteChunk->WriteObject((CKObject *) m_Material);

    // Close chunk appropriately based on class type
    if (GetClassID() == CKCID_SPRITE3D) {
        spriteChunk->CloseChunk();
    } else {
        spriteChunk->UpdateDataSize();
    }

    return spriteChunk;
}

/*************************************************
Summary: Load method for RCKSprite3D.
Purpose: Loads sprite data from a state chunk including mode, size, offset, UV mapping, and material.
Remarks:
- Calls base class RCK3dEntity::Load() first to handle entity data
- Loads sprite mode and calculates full size from half dimensions
- Sets sprite size and offset using appropriate methods
- Loads UV mapping rectangle and applies it to the sprite
- Loads material object reference
- Sets object flags to mark sprite as modified after loading

Implementation based on decompilation at 0x1004340F:
- Uses chunk identifier 0x400000 for sprite-specific data
- Converts half dimensions to full size for SetSize method
- Uses SetOffset and SetUVMapping methods for proper initialization
- Sets object flag 0x400 to indicate sprite modification

Arguments:
- chunk: The state chunk containing sprite data
- file: The file context for loading (may be NULL)
Return Value:
- CKERROR: 0 for success, -1 for invalid chunk
*************************************************/
CKERROR RCKSprite3D::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) {
        return CKERR_INVALIDPARAMETER;
    }

    // Call base class load first to handle entity data
    RCK3dEntity::Load(chunk, file);

    if (chunk->SeekIdentifier(CK_STATESAVE_SPRITE3DDATA)) {
        // Load sprite mode
        m_Mode = chunk->ReadDword();

        // Load half dimensions and calculate full size
        float halfWidth = chunk->ReadFloat();
        float halfHeight = chunk->ReadFloat();
        Vx2DVector size(halfWidth * 2.0f, halfHeight * 2.0f);
        SetSize(size);

        // Load offset position
        m_Offset.x = chunk->ReadFloat();
        m_Offset.y = chunk->ReadFloat();
        Vx2DVector offset(m_Offset.x, m_Offset.y);
        SetOffset(offset);

        // Load UV mapping rectangle
        m_Rect.left = chunk->ReadFloat();
        m_Rect.top = chunk->ReadFloat();
        m_Rect.right = chunk->ReadFloat();
        m_Rect.bottom = chunk->ReadFloat();
        SetUVMapping(m_Rect);

        // Load material object
        m_Material = (RCKMaterial *) chunk->ReadObject(m_Context);

        // Mark sprite as modified after loading
        ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
    }

    return CK_OK;
}

RCKSprite3D::RCKSprite3D(CKContext *Context, CKSTRING name)
    : RCK3dEntity(Context, name) {
    m_Mode = VXSPRITE3D_BILLBOARD;
    m_Material = nullptr;

    // Initialize offset to center
    m_Offset.x = 0.0f;
    m_Offset.y = 0.0f;

    // Initialize local bounding box for 2x2 sprite centered at origin
    m_LocalBoundingBox.Min.x = -1.0f;
    m_LocalBoundingBox.Min.y = -1.0f;
    m_LocalBoundingBox.Min.z = 0.0f;
    m_LocalBoundingBox.Max.x = 1.0f;
    m_LocalBoundingBox.Max.y = 1.0f;
    m_LocalBoundingBox.Max.z = 0.0f;

    // Initialize UV mapping to full texture
    m_Rect.left = 0.0f;
    m_Rect.top = 0.0f;
    m_Rect.right = 1.0f;
    m_Rect.bottom = 1.0f;
}

RCKSprite3D::~RCKSprite3D() {}

CK_CLASSID RCKSprite3D::m_ClassID = CKCID_SPRITE3D;

CK_CLASSID RCKSprite3D::GetClassID() {
    return m_ClassID;
}

int RCKSprite3D::GetMemoryOccupation() {
    // Base class memory + Sprite3D specific members
    return RCK3dEntity::GetMemoryOccupation() + (sizeof(RCKSprite3D) - sizeof(RCK3dEntity));
}

/*************************************************
Summary: CheckPreDeletion method for RCKSprite3D.
Purpose: Clears material reference if the material is being deleted.
Remarks:
- Calls base class CheckPreDeletion first
- Checks if material is flagged for deletion and clears reference

Implementation based on decompilation at 0x100431DC:
- Uses sub_1000CEB0 which checks if object is being deleted
- Clears m_Material to nullptr if material is being deleted
*************************************************/
void RCKSprite3D::CheckPreDeletion() {
    // Call base class first
    RCK3dEntity::CheckPreDeletion();

    // Check if material is being deleted
    if (m_Material) {
        if (m_Material->IsToBeDeleted()) {
            m_Material = nullptr;
        }
    }
}

/*************************************************
Summary: IsObjectUsed method for RCKSprite3D.
Purpose: Checks if the given object is used by this sprite.
Remarks:
- Returns TRUE if the object is this sprite's material
- Otherwise delegates to base class

Implementation based on decompilation at 0x10043238:
*************************************************/
int RCKSprite3D::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    if (obj == (CKObject *) m_Material) {
        return TRUE;
    }
    return RCK3dEntity::IsObjectUsed(obj, cid);
}

/*************************************************
Summary: PrepareDependencies method for RCKSprite3D.
Purpose: Prepares dependencies for copy/save operations.
Remarks:
- Calls base class PrepareDependencies first
- If material dependency bit is set, prepares material

Implementation based on decompilation at 0x100436BB:
- Checks CKDependenciesContext::GetClassDependencies for CKCID_SPRITE3D
- Bit 0 (& 1) controls material dependency
*************************************************/
CKERROR RCKSprite3D::PrepareDependencies(CKDependenciesContext &context) {
    // Call base class first
    CKERROR err = RCK3dEntity::PrepareDependencies(context);
    if (err != CK_OK) {
        return err;
    }

    // Check if material dependency is enabled
    if ((context.GetClassDependencies(CKCID_SPRITE3D) & 1) != 0) {
        if (m_Material) {
            ((CKObject *) m_Material)->PrepareDependencies(context);
        }
    }

    return context.FinishPrepareDependencies((CKObject *) this, m_ClassID);
}

/*************************************************
Summary: RemapDependencies method for RCKSprite3D.
Purpose: Remaps object references after copy/load operations.
Remarks:
- Calls base class RemapDependencies first
- Remaps material reference to copied material

Implementation based on decompilation at 0x10043734:
*************************************************/
CKERROR RCKSprite3D::RemapDependencies(CKDependenciesContext &context) {
    // Call base class first
    CKERROR err = RCK3dEntity::RemapDependencies(context);
    if (err != CK_OK) {
        return err;
    }

    // Remap material reference
    CKMaterial *material = (CKMaterial *) context.Remap((CKObject *) m_Material);
    SetMaterial(material);

    return CK_OK;
}

CKERROR RCKSprite3D::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = RCK3dEntity::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKSprite3D *src = (RCKSprite3D *) &o;

    m_Material = src->m_Material;
    m_LocalBoundingBox = src->m_LocalBoundingBox;
    m_Offset = src->m_Offset;
    m_Rect = src->m_Rect;
    m_Mode = src->m_Mode;

    return CK_OK;
}

void RCKSprite3D::SetMaterial(CKMaterial *mat) {
    m_Material = (RCKMaterial *) mat;
}

CKMaterial *RCKSprite3D::GetMaterial() {
    return (CKMaterial *) m_Material;
}

void RCKSprite3D::SetSize(Vx2DVector &size) {
    // Reset Z bounds
    m_LocalBoundingBox.Max.z = 0.0f;
    m_LocalBoundingBox.Min.z = 0.0f;

    // Calculate bounds based on offset and size
    // Offset is in range [-1, 1], where 0 means centered
    float halfWidth = size.x * 0.5f;
    float halfHeight = size.y * 0.5f;

    m_LocalBoundingBox.Min.x = (m_Offset.x - 1.0f) * halfWidth;
    m_LocalBoundingBox.Min.y = (m_Offset.y - 1.0f) * halfHeight;
    m_LocalBoundingBox.Max.x = (m_Offset.x + 1.0f) * halfWidth;
    m_LocalBoundingBox.Max.y = (m_Offset.y + 1.0f) * halfHeight;

    // Invalidate scene graph bounding box
    if (m_SceneGraphNode)
        m_SceneGraphNode->InvalidateBox(TRUE);
}

void RCKSprite3D::GetSize(Vx2DVector &size) {
    size.x = m_LocalBoundingBox.Max.x - m_LocalBoundingBox.Min.x;
    size.y = m_LocalBoundingBox.Max.y - m_LocalBoundingBox.Min.y;
}

void RCKSprite3D::SetOffset(Vx2DVector &offset) {
    m_Offset = offset;

    // Calculate current half size
    float halfWidth = (m_LocalBoundingBox.Max.x - m_LocalBoundingBox.Min.x) * 0.5f;
    float halfHeight = (m_LocalBoundingBox.Max.y - m_LocalBoundingBox.Min.y) * 0.5f;

    // Reset Z bounds
    m_LocalBoundingBox.Max.z = 0.0f;
    m_LocalBoundingBox.Min.z = 0.0f;

    // Recalculate bounds based on new offset
    m_LocalBoundingBox.Min.x = (m_Offset.x - 1.0f) * halfWidth;
    m_LocalBoundingBox.Min.y = (m_Offset.y - 1.0f) * halfHeight;
    m_LocalBoundingBox.Max.x = (m_Offset.x + 1.0f) * halfWidth;
    m_LocalBoundingBox.Max.y = (m_Offset.y + 1.0f) * halfHeight;

    // Invalidate scene graph bounding box
    if (m_SceneGraphNode)
        m_SceneGraphNode->InvalidateBox(TRUE);
}

void RCKSprite3D::GetOffset(Vx2DVector &offset) {
    offset = m_Offset;
}

void RCKSprite3D::SetUVMapping(VxRect &rect) {
    m_Rect = rect;
}

void RCKSprite3D::GetUVMapping(VxRect &rect) {
    rect = m_Rect;
}

void RCKSprite3D::SetMode(VXSPRITE3D_TYPE mode) {
    m_Mode = mode;
}

VXSPRITE3D_TYPE RCKSprite3D::GetMode() {
    return (VXSPRITE3D_TYPE) m_Mode;
}

CKBOOL RCKSprite3D::IsToBeRendered() {
    const CKBOOL visible = (m_MoveableFlags & VX_MOVEABLE_VISIBLE) != 0;
    return visible && m_Material;
}

CKBOOL RCKSprite3D::IsToBeRenderedLast() {
    if ((m_MoveableFlags & VX_MOVEABLE_RENDERFIRST) != 0)
        return FALSE;
    if (m_Material)
        return m_Material->IsAlphaTransparent();
    return FALSE;
}

CKBOOL RCKSprite3D::IsInViewFrustrum(CKRenderContext *rc, CKDWORD flags) {
    RCKRenderContext *dev = (RCKRenderContext *) rc;

    UpdateOrientation(rc);
    ModifyMoveableFlags(VX_MOVEABLE_EXTENTSUPTODATE, 0);

    if (!(flags & 0x100)) {
        dev->SetWorldTransformationMatrix(m_WorldMatrix);
    }

    CKDWORD vis;
    if (flags & 0xFF) {
        m_RenderExtents = VxRect(100000000.0f, 100000000.0f, -100000000.0f, -100000000.0f);
        vis = dev->m_RasterizerContext->ComputeBoxVisibility(m_LocalBoundingBox, FALSE, &m_RenderExtents);
    } else {
        vis = dev->m_RasterizerContext->ComputeBoxVisibility(m_LocalBoundingBox, FALSE, nullptr);
    }

    if (!vis) {
        if (m_SceneGraphNode)
            m_SceneGraphNode->SetAsOutsideFrustum();
        return FALSE;
    }

    if (vis == 2) {
        if (m_SceneGraphNode)
            m_SceneGraphNode->SetAsInsideFrustum();
    }

    return TRUE;
}

void RCKSprite3D::UpdateBox(CKBOOL World) {
    if (!(m_MoveableFlags & VX_MOVEABLE_UPTODATE) && World) {
        m_WorldBoundingBox.TransformFrom(m_LocalBoundingBox, m_WorldMatrix);
        m_MoveableFlags |= VX_MOVEABLE_BOXVALID | VX_MOVEABLE_UPTODATE;
    }
}

CKBOOL RCKSprite3D::SetBoundingBox(const VxBbox *BBox, CKBOOL Local) {
    return TRUE;
}

void RCKSprite3D::UpdateOrientation(CKRenderContext *rc) {
    if (!rc)
        return;

    RCKRenderContext *dev = (RCKRenderContext *) rc;
    if (!dev->m_RenderedScene)
        return;

    RCK3dEntity *rootEntity = (RCK3dEntity *) dev->m_RenderedScene->GetRootEntity();
    if (!rootEntity)
        return;

    const VxMatrix &rootWorld = rootEntity->m_WorldMatrix;

    const CKDWORD mode = m_Mode;
    if (mode == VXSPRITE3D_BILLBOARD) {
        // IDA: memcpy 0x30 bytes from root entity world matrix to this world matrix
        // (copies only the 3 orientation rows; keeps translation intact)
        memcpy(&m_WorldMatrix, &rootWorld, sizeof(VxMatrix) - sizeof(VxVector4));
        WorldMatrixChanged(TRUE, TRUE);
    } else if (mode == VXSPRITE3D_XROTATE) {
        // IDA: row0 = axisX
        const VxVector &axisX = VxVector::axisX();
        m_WorldMatrix[0][0] = axisX.x;
        m_WorldMatrix[0][1] = axisX.y;
        m_WorldMatrix[0][2] = axisX.z;

        // IDA: v0 = normalize( (0, -rootWorld[1][2], rootWorld[1][1]) ) => row2
        VxVector v0(0.0f, -rootWorld[1][2], rootWorld[1][1]);
        v0.Normalize();
        m_WorldMatrix[2][0] = v0.x;
        m_WorldMatrix[2][1] = v0.y;
        m_WorldMatrix[2][2] = v0.z;

        // IDA: v1 = normalize( (0, row2.z, -row2.y) ) => row1
        VxVector v1(0.0f, m_WorldMatrix[2][2], -m_WorldMatrix[2][1]);
        v1.Normalize();
        m_WorldMatrix[1][0] = v1.x;
        m_WorldMatrix[1][1] = v1.y;
        m_WorldMatrix[1][2] = v1.z;

        WorldMatrixChanged(TRUE, TRUE);
    } else if (mode == VXSPRITE3D_YROTATE) {
        // IDA: row1 = axisY
        const VxVector &axisY = VxVector::axisY();
        m_WorldMatrix[1][0] = axisY.x;
        m_WorldMatrix[1][1] = axisY.y;
        m_WorldMatrix[1][2] = axisY.z;

        // IDA: v0 = normalize( (rootWorld[2][2], 0, -rootWorld[2][0]) ) => row0
        VxVector v0(rootWorld[2][2], 0.0f, -rootWorld[2][0]);
        v0.Normalize();
        m_WorldMatrix[0][0] = v0.x;
        m_WorldMatrix[0][1] = v0.y;
        m_WorldMatrix[0][2] = v0.z;

        // IDA: v1 = normalize( (-row0.z, 0, row0.x) ) => row2
        VxVector v1(-m_WorldMatrix[0][2], 0.0f, m_WorldMatrix[0][0]);
        v1.Normalize();
        m_WorldMatrix[2][0] = v1.x;
        m_WorldMatrix[2][1] = v1.y;
        m_WorldMatrix[2][2] = v1.z;

        WorldMatrixChanged(TRUE, TRUE);
    }
}

//=============================================================================
// Static Class Methods (for class registration)
// Based on IDA Pro analysis of original CK2_3D.dll
//=============================================================================

CKSTRING RCKSprite3D::GetClassName() {
    return (CKSTRING) "3D Sprite";
}

/*************************************************
Summary: GetDependenciesCount static method for RCKSprite3D.
Purpose: Returns the number of dependency types for the given mode.
Remarks:
- Mode 1 (CK_DEPENDENCIES_COPY): Returns 1 (Material)
- Mode 2 (CK_DEPENDENCIES_SAVE): Returns 1 (Material)
- Mode 3 (CK_DEPENDENCIES_DELETE): Returns 0
- Mode 4 (CK_DEPENDENCIES_REPLACE): Returns 1 (Material)

Implementation based on decompilation at 0x10043580
*************************************************/
int RCKSprite3D::GetDependenciesCount(int mode) {
    switch (mode) {
    case CK_DEPENDENCIES_COPY:
        return 1;
    case CK_DEPENDENCIES_DELETE:
        return 1;
    case CK_DEPENDENCIES_REPLACE:
        return 0;
    case CK_DEPENDENCIES_SAVE:
        return 1;
    default:
        return 0;
    }
}

/*************************************************
Summary: GetDependencies static method for RCKSprite3D.
Purpose: Returns the name of the i-th dependency type.
Remarks:
- Index 0: Returns "Material"
- Other indices: Returns nullptr

Implementation based on decompilation at 0x100435D2
*************************************************/
CKSTRING RCKSprite3D::GetDependencies(int i, int mode) {
    if (i == 0) {
        return (CKSTRING) "Material";
    }
    return nullptr;
}

void RCKSprite3D::Register() {
    // Based on IDA decompilation
    CKCLASSNOTIFYFROMCID(RCKSprite3D, CKCID_MATERIAL);

    // Register associated parameter GUID: CKPGUID_SPRITE3D
    CKPARAMETERFROMCLASS(RCKSprite3D, CKPGUID_SPRITE3D);

    // Register default dependencies
    CKCLASSDEFAULTCOPYDEPENDENCIES(RCKSprite3D, CK_DEPENDENCIES_COPY);
}

CKSprite3D *RCKSprite3D::CreateInstance(CKContext *Context) {
    RCKSprite3D *sprite = new RCKSprite3D(Context, nullptr);
    return reinterpret_cast<CKSprite3D *>(sprite);
}

//=============================================================================
// FillBatch - Fills a sprite batch with vertex data
// Based on IDA decompilation at 0x100428A8
//=============================================================================

void RCKSprite3D::FillBatch(CKSprite3DBatch *batch) {
    if (!batch)
        return;

    // Check if clipping is needed from scene graph node
    // If the node's flag bit 0 is not set, we need clipping (from IDA: m_Flags & 1)
    CKSceneGraphNode *node = (CKSceneGraphNode *) m_SceneGraphNode;
    if (node) {
        // Set batch flag if clipping needed (flag bit 0 == 0 means not inside frustum)
        batch->m_Flags |= node->CheckHierarchyFrustum() ? 0 : 1;
    }

    // Expand vertex array by 4 vertices (one quad for the sprite)
    // This adds 4 vertices (128 bytes since CKVertex is 32 bytes)
    int currentSize = batch->m_Vertices.Size();
    batch->m_Vertices.Resize(currentSize + 4);

    // Get pointer to the 4 new vertices at the end
    CKVertex *vertices = batch->m_Vertices.Begin() + currentSize;

    // Dimensions from local bbox
    const float width = m_LocalBoundingBox.Max.x - m_LocalBoundingBox.Min.x;
    const float height = m_LocalBoundingBox.Max.y - m_LocalBoundingBox.Min.y;

    // World matrix columns 0 and 1 scaled by size, plus translation (row 3)
    const VxVector scaledX(
        m_WorldMatrix[0][0] * width,
        m_WorldMatrix[0][1] * width,
        m_WorldMatrix[0][2] * width);
    const VxVector scaledY(
        m_WorldMatrix[1][0] * height,
        m_WorldMatrix[1][1] * height,
        m_WorldMatrix[1][2] * height);

    const VxVector position(m_WorldMatrix[3][0], m_WorldMatrix[3][1], m_WorldMatrix[3][2]);

    // Offsets are (offset - 1) * 0.5 per IDA
    const float offsetX = (m_Offset.x - 1.0f) * 0.5f;
    const float offsetY = (m_Offset.y - 1.0f) * 0.5f;

    const VxVector basePos = position + (scaledX * offsetX) + (scaledY * offsetY);

    // Vertex 0: base
    vertices[0].V.x = basePos.x;
    vertices[0].V.y = basePos.y;
    vertices[0].V.z = basePos.z;
    vertices[0].V.w = 1.0f;

    // Vertex 1: base + Y
    const VxVector pos1 = basePos + scaledY;
    vertices[1].V.x = pos1.x;
    vertices[1].V.y = pos1.y;
    vertices[1].V.z = pos1.z;
    vertices[1].V.w = 1.0f;

    // Vertex 2: base + Y + X
    const VxVector pos2 = pos1 + scaledX;
    vertices[2].V.x = pos2.x;
    vertices[2].V.y = pos2.y;
    vertices[2].V.z = pos2.z;
    vertices[2].V.w = 1.0f;

    // Vertex 3: base + X
    const VxVector pos3 = basePos + scaledX;
    vertices[3].V.x = pos3.x;
    vertices[3].V.y = pos3.y;
    vertices[3].V.z = pos3.z;
    vertices[3].V.w = 1.0f;

    // Set texture coordinates from m_Rect
    // Vertex 0: left, bottom
    vertices[0].tu = m_Rect.left;
    vertices[0].tv = m_Rect.bottom;

    // Vertex 1: left, top
    vertices[1].tu = m_Rect.left;
    vertices[1].tv = m_Rect.top;

    // Vertex 2: right, top
    vertices[2].tu = m_Rect.right;
    vertices[2].tv = m_Rect.top;

    // Vertex 3: right, bottom
    vertices[3].tu = m_Rect.right;
    vertices[3].tv = m_Rect.bottom;
}

//=============================================================================
// Render - Renders the sprite to the render context
// Based on IDA decompilation at 0x100424CE
//
// The sprite rendering uses a batching system where sprites with the same
// material are grouped together for efficient rendering.
//=============================================================================

CKBOOL RCKSprite3D::Render(CKRenderContext *Dev, CKDWORD Flags) {
    RCKRenderContext *dev = (RCKRenderContext *) Dev;

    // Local VxTimeProfiler constructed at entry (used by Dev debug mode).
    VxTimeProfiler profiler;

    // If VX_MOVEABLE_EXTENTSUPTODATE is set (0x20), the object is treated as transparent / already-validated.
    if ((m_MoveableFlags & VX_MOVEABLE_EXTENTSUPTODATE) != 0) {
        if (m_Callbacks && (Flags & CK_RENDER_CLEARVIEWPORT) == 0) {
            dev->SetWorldTransformationMatrix(m_WorldMatrix);
        }
    } else {
        if (!IsInViewFrustrum(Dev, Flags)) {
            if ((dev->m_Flags & 1) != 0) {
                dev->m_CurrentObjectDesc << m_Name;
                if (IsToBeRenderedLast())
                    dev->m_CurrentObjectDesc << " (as transparent Object)";
                dev->m_CurrentObjectDesc << " : " << "Not drawn";
                dev->m_CurrentObjectDesc << profiler.Current() << " ms \n";
                if (--dev->m_FpsInterval <= 0)
                    dev->BackToFront(CK_RENDER_USECURRENTSETTINGS);
            }
            return TRUE;
        }
    }

    if (m_Callbacks) {
        if (m_Callbacks->m_PreCallBacks.Size() > 0) {
            dev->m_ObjectsCallbacksTimeProfiler.Reset();
            dev->m_RasterizerContext->SetVertexShader(0);

            for (int i = 0; i < m_Callbacks->m_PreCallBacks.Size(); ++i) {
                VxCallBack &cb = m_Callbacks->m_PreCallBacks[i];
                if (cb.callback) {
                    ((CK_RENDEROBJECT_CALLBACK) cb.callback)(Dev, (CKRenderObject *) this, cb.argument);
                }
            }

            dev->m_Stats.ObjectsCallbacksTime += dev->m_ObjectsCallbacksTimeProfiler.Current();
        }

        if (m_Material) {
            dev->AddSprite3DBatch(this);
        }

        if (m_Callbacks->m_PostCallBacks.Size() > 0) {
            dev->m_ObjectsCallbacksTimeProfiler.Reset();
            dev->m_RasterizerContext->SetVertexShader(0);

            for (int i = 0; i < m_Callbacks->m_PostCallBacks.Size(); ++i) {
                VxCallBack &cb = m_Callbacks->m_PostCallBacks[i];
                if (cb.callback) {
                    ((CK_RENDEROBJECT_CALLBACK) cb.callback)(Dev, (CKRenderObject *) this, cb.argument);
                }
            }

            dev->m_Stats.ObjectsCallbacksTime += dev->m_ObjectsCallbacksTimeProfiler.Current();
        }
    } else {
        if (m_Material) {
            dev->AddSprite3DBatch(this);
        }
    }

    if ((Flags & 0xFF) != 0) {
        dev->AddExtents2D(m_RenderExtents, (CKObject *) this);
    }

    if ((dev->m_Flags & 1) != 0) {
        dev->m_CurrentObjectDesc << m_Name;
        if (IsToBeRenderedLast())
            dev->m_CurrentObjectDesc << " (as transparent Object)";
        dev->m_CurrentObjectDesc << " : " << "Drawn";
        dev->m_CurrentObjectDesc << profiler.Current() << " ms \n";
        if (--dev->m_FpsInterval <= 0)
            dev->BackToFront(CK_RENDER_USECURRENTSETTINGS);
    }

    return TRUE;
}

//=============================================================================
// RayIntersection - Performs ray intersection test with the sprite quad
// Based on IDA decompilation at 0x10042F2A
//
// The sprite is treated as a quad in 3D space. The ray is transformed
// into local space, then intersected with the XY plane at Z=0.
// If the intersection point is within the sprite bounds, it's a hit.
//=============================================================================

int RCKSprite3D::RayIntersection(const VxVector *Pos1, const VxVector *Pos2,
                                 VxIntersectionDesc *Desc, CK3dEntity *Ref,
                                 CK_RAYINTERSECTION iOptions) {
    // Transform ray into local space
    VxVector localPos1 = *Pos1;
    VxVector localPos2 = *Pos2;

    if (Ref != (CK3dEntity *) this) {
        InverseTransform(&localPos1, Pos1, Ref);
        InverseTransform(&localPos2, Pos2, Ref);
    }

    // Calculate ray direction
    VxVector rayDir;
    rayDir.x = localPos2.x - localPos1.x;
    rayDir.y = localPos2.y - localPos1.y;
    rayDir.z = localPos2.z - localPos1.z;

    // The sprite is in the XY plane at Z=0
    // Find intersection with Z=0 plane
    if (rayDir.z == 0.0f) {
        return 0; // Ray parallel to sprite plane
    }

    // t = -localPos1.z / rayDir.z gives us the parameter where ray hits Z=0
    float t = -localPos1.z / rayDir.z;

    // Calculate intersection point
    VxVector intersectPoint;
    intersectPoint.x = localPos1.x + t * rayDir.x;
    intersectPoint.y = localPos1.y + t * rayDir.y;
    intersectPoint.z = 0.0f; // On the sprite plane

    // Check if intersection point is within sprite bounds
    if (intersectPoint.x < m_LocalBoundingBox.Min.x ||
        intersectPoint.x > m_LocalBoundingBox.Max.x ||
        intersectPoint.y < m_LocalBoundingBox.Min.y ||
        intersectPoint.y > m_LocalBoundingBox.Max.y) {
        return 0; // Outside sprite bounds
    }

    // Fill in intersection description if provided
    if (Desc) {
        Desc->Distance = t;

        // Scale distance by the input segment length (in the referential given by Ref)
        VxVector dir;
        dir.x = Pos2->x - Pos1->x;
        dir.y = Pos2->y - Pos1->y;
        dir.z = Pos2->z - Pos1->z;
        Desc->Distance = Magnitude(dir) * Desc->Distance;

        Desc->FaceIndex = 0;
        Desc->IntersectionPoint = intersectPoint;

        const float uvWidth = m_Rect.GetWidth();
        const float uvHeight = m_Rect.GetHeight();
        const float spriteWidth = m_LocalBoundingBox.Max.x - m_LocalBoundingBox.Min.x;
        const float spriteHeight = m_LocalBoundingBox.Max.y - m_LocalBoundingBox.Min.y;

        Desc->TexU = (intersectPoint.x - m_LocalBoundingBox.Min.x) * uvWidth / spriteWidth + m_Rect.left;
        Desc->TexV = m_Rect.bottom - (intersectPoint.y - m_LocalBoundingBox.Min.y) * uvHeight / spriteHeight;

        // Precise alpha-tested pick (material alpha ref + texture)
        if (m_Material && !PreciseTexturePick(m_Material, Desc->TexU, Desc->TexV)) {
            return 0;
        }

        Desc->IntersectionNormal.x = 0.0f;
        Desc->IntersectionNormal.y = 0.0f;
        Desc->IntersectionNormal.z = -1.0f;
    }

    return 1; // Intersection found
}
