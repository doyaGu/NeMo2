#include "RCK3dEntity.h"

#include "VxMath.h"
#include "CKSkin.h"
#include "CKMesh.h"
#include "CKSceneGraph.h"
#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKPlace.h"
#include "CKScene.h"
#include "CKDependencies.h"
#include "CKRasterizer.h"
#include "RCKRenderContext.h"
#include "RCKRenderManager.h"
#include "RCKMesh.h"
#include "RCKObjectAnimation.h"
#include "RCKSkin.h"

extern int (*g_RayIntersection)(RCKMesh *, VxVector &, VxVector &, VxIntersectionDesc *, CK_RAYINTERSECTION, VxMatrix const &);

CK_CLASSID RCK3dEntity::m_ClassID = CKCID_3DENTITY;

/*************************************************
Summary: PostLoad method for RCK3dEntity.
Purpose: Called after loading to finalize the entity state.
Remarks:
- Updates skin if present to ensure proper vertex binding
- Processes object animations with current scale transformation
- Adds entity to current level's render contexts if loaded from file
- Calls base class PostLoad to complete initialization

Implementation based on decompilation at 0x1000A138:
- Checks if m_Skin exists and calls UpdateSkin()
- Iterates through m_ObjectAnimations and applies scale transformation
- Adds to render contexts if in a level
- Calls CKObject::PostLoad() at the end
*************************************************/
void RCK3dEntity::PostLoad() {
    if (m_Skin) {
        UpdateSkin();
    }

    if (m_ObjectAnimations && m_ObjectAnimations->Size() > 0) {
        VxVector scale;
        GetScale(&scale, TRUE);

        for (CKObject **it = m_ObjectAnimations->Begin(); it != m_ObjectAnimations->End(); ++it) {
            RCKObjectAnimation *anim = (RCKObjectAnimation *) *it;
            if (anim)
                anim->CheckScaleKeys(scale);
        }
    }

    CKObject::PostLoad();
}

/*************************************************
Summary: Load method for RCK3dEntity.
Purpose: Loads entity data from a state chunk.
Remarks:
- Supports both old and new format versions for backward compatibility
- Loads meshes, animations, and transformation matrices
- Handles complex skin data with bones and vertex weights
- Restores parent-child relationships in scene hierarchy
- Processes entity flags and moveable properties

Implementation based on decompilation at 0x1000A7B9:
- Calls CKBeObject::Load() first
- Preserves moveable flags with 0x40000 mask
- Loads object animations (chunk 0x2000)
- Loads meshes (chunk 0x4000)
- Loads main entity data (chunk 0x100000) with flags, matrix, parent, priority
- Supports legacy format with separate chunks (0x8000, 0x10000, 0x20000)
- Loads skin data (chunk 0x200000) with bones, vertices, weights, normals
- Sets world matrix if not in file context
- Sets default bounding box if needed

Arguments:
- chunk: The state chunk containing entity data
- file: The file context for loading (may be NULL)
Return Value:
- CKERROR: 0 for success, error code otherwise
*************************************************/
CKERROR RCK3dEntity::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Call base class load first to handle basic object data
    CKERROR err = CKBeObject::Load(chunk, file);
    if (err != CK_OK) {
        return err;
    }

    // Preserve moveable flags that should not be modified during load
    // Based on IDA: v16 = GetMoveableFlags() & 0x40000
    CKDWORD preservedFlags = GetMoveableFlags() & VX_MOVEABLE_WORLDALIGNED;

    // Initialize identity matrix for transformation
    VxMatrix worldMatrix = VxMatrix::Identity();

    // Load object animations (chunk 0x2000)
    // Deduplicate using last-pointer tracking as in IDA
    if (chunk->SeekIdentifier(CK_STATESAVE_ANIMATION)) {
        if (!m_ObjectAnimations) {
            m_ObjectAnimations = new XObjectPointerArray();
        }

        XObjectPointerArray tempAnims;
        tempAnims.Load(m_Context, chunk);

        CKObject *lastAnim = nullptr;
        for (CKObject **it = tempAnims.Begin(); it != tempAnims.End(); ++it) {
            CKObject *anim = *it;
            if (anim && anim != lastAnim) {
                m_ObjectAnimations->AddIfNotHere(anim);
                lastAnim = anim;
            }
        }
    }

    // Load meshes (chunk 0x4000)
    // Based on IDA and docs: First reads current mesh, then mesh array
    if (chunk->SeekIdentifier(CK_STATESAVE_MESHS)) {
        // Read current mesh object first
        CKMesh *currentMesh = (CKMesh *) chunk->ReadObject(m_Context);

        // Load mesh array into a temporary array to preserve original ordering/deduplication
        XObjectPointerArray tempMeshes;
        tempMeshes.Load(m_Context, chunk);

        CKObject *lastMesh = nullptr;
        for (CKObject **it = tempMeshes.Begin(); it != tempMeshes.End(); ++it) {
            CKMesh *mesh = (CKMesh *) *it;
            if (mesh && mesh != lastMesh) {
                m_Meshes.AddIfNotHere(mesh);
                lastMesh = mesh;
            }
        }

        // Set current mesh
        SetCurrentMesh(currentMesh, TRUE);
    }

    // Load main entity data (chunk 0x100000) - new format
    // Matches IDA implementation @0x1000a7b9 EXACTLY
    if (chunk->SeekIdentifier(CK_STATESAVE_3DENTITYNDATA)) {
        // IDA line 117: Dword = CKStateChunk::ReadDword(a2);
        CKDWORD entityFlags = chunk->ReadDword();

        // IDA line 118: this->SetFlags(this, Dword & 0xFFFFEFDF);
        // SetFlags stores (entityFlags & mask) to m_3dEntityFlags
        // The mask 0xFFFFEFDF clears bits 5 and 12, but preserves presence indicators
        SetFlags(entityFlags & ~(CK_3DENTITY_RESERVED0 | CK_3DENTITY_UPDATELASTFRAME));

        const CKDWORD moveableFlagsRaw = chunk->ReadDword();
        const CKDWORD clearMask = VX_MOVEABLE_UPTODATE | VX_MOVEABLE_USERBOX | VX_MOVEABLE_BOXVALID |
            VX_MOVEABLE_HASMOVED | VX_MOVEABLE_INVERSEWORLDMATVALID | VX_MOVEABLE_DONTUPDATEFROMPARENT |
            VX_MOVEABLE_STENCILONLY | VX_MOVEABLE_RESERVED2;
        CKDWORD moveableFlags = moveableFlagsRaw & ~clearMask;

        // IDA line 120-123: if ( v16 ) v53 |= 0x40000u;
        if (preservedFlags) {
            moveableFlags |= VX_MOVEABLE_WORLDALIGNED;
        }

        // IDA line 124-127: if ( (v53 & 0x100000) != 0 ) SetPriority(10000)
        if ((moveableFlags & VX_MOVEABLE_RENDERFIRST) != 0) {
            if (m_SceneGraphNode) {
                m_SceneGraphNode->SetPriority(10000, FALSE);
            }
        }

        // IDA line 128-143: Read matrix row vectors
        VxVector row0, row1, row2, row3;
        chunk->ReadVector(&row0);
        chunk->ReadVector(&row1);
        chunk->ReadVector(&row2);
        chunk->ReadVector(&row3);

        // Construct world matrix from row vectors
        worldMatrix[0][0] = row0.x;
        worldMatrix[0][1] = row0.y;
        worldMatrix[0][2] = row0.z;
        worldMatrix[0][3] = 0.0f;
        worldMatrix[1][0] = row1.x;
        worldMatrix[1][1] = row1.y;
        worldMatrix[1][2] = row1.z;
        worldMatrix[1][3] = 0.0f;
        worldMatrix[2][0] = row2.x;
        worldMatrix[2][1] = row2.y;
        worldMatrix[2][2] = row2.z;
        worldMatrix[2][3] = 0.0f;
        worldMatrix[3][0] = row3.x;
        worldMatrix[3][1] = row3.y;
        worldMatrix[3][2] = row3.z;
        worldMatrix[3][3] = 1.0f;

        // IDA line 144-146: Calculate determinant for handedness
        VxVector cross = CrossProduct(row0, row1);
        float dot = DotProduct(cross, row2);
        if (dot < 0.0f) {
            moveableFlags |= VX_MOVEABLE_INDIRECTMATRIX;
        } else {
            moveableFlags &= ~VX_MOVEABLE_INDIRECTMATRIX;
        }

        // IDA line 147-150: Sync VX_MOVEABLE_VISIBLE from CKObject::IsVisible()
        // if ( CKObject::IsVisible(this) ) v53 |= 2u; else v53 &= ~2u;
        if (IsVisible()) {
            moveableFlags |= VX_MOVEABLE_VISIBLE;
        } else {
            moveableFlags &= ~VX_MOVEABLE_VISIBLE;
        }

        // IDA line 151-154: Sync VX_MOVEABLE_HIERARCHICALHIDE from CK_OBJECT_HIERACHICALHIDE
        // if ( sub_1000CE70(this) ) v53 |= 0x10000000u; else v53 &= ~0x10000000u;
        if (m_ObjectFlags & CK_OBJECT_HIERACHICALHIDE) {
            moveableFlags |= VX_MOVEABLE_HIERARCHICALHIDE;
        } else {
            moveableFlags &= ~VX_MOVEABLE_HIERARCHICALHIDE;
        }

        // IDA line 155: SetMoveableFlags(this, v53);
        SetMoveableFlags(moveableFlags);

        // IDA line 156-157: Read optional Place (if entityFlags & 0x10000)
        if (m_3dEntityFlags & CK_3DENTITY_PLACEVALID) {
            chunk->ReadObjectID(); // Place reference (deprecated)
        }

        // IDA line 158-166: Read optional parent (if entityFlags & 0x20000)
        if (m_3dEntityFlags & CK_3DENTITY_PARENTVALID) {
            CK3dEntity *parent = (CK3dEntity *) chunk->ReadObject(m_Context);
            SetParent(parent, TRUE);
        } else {
            SetParent(nullptr, TRUE);
        }

        // IDA line 167-171: Read optional priority (if entityFlags & 0x100000)
        if (m_3dEntityFlags & CK_3DENTITY_ZORDERVALID) {
            int priority = chunk->ReadInt();
            if (m_SceneGraphNode) {
                m_SceneGraphNode->SetPriority(priority, FALSE);
            }
        }
    }

    // Based on IDA: separate chunks for parent (0x8000), flags (0x10000), matrix (0x20000)
    if (chunk->SeekIdentifier(CK_STATESAVE_PARENT)) {
        CK3dEntity *parent = (CK3dEntity *) chunk->ReadObject(m_Context);
        if (parent) {
            SetParent(parent, TRUE);
        }
    }

    if (chunk->SeekIdentifier(CK_STATESAVE_3DENTITYFLAGS)) {
        CKDWORD flags = chunk->ReadDword();
        SetFlags(flags);

        const CKDWORD moveableFlagsRaw = chunk->ReadDword();
        CKDWORD moveableFlags = moveableFlagsRaw & ~(VX_MOVEABLE_UPTODATE | VX_MOVEABLE_USERBOX);
        moveableFlags &= ~(VX_MOVEABLE_INVERSEWORLDMATVALID | VX_MOVEABLE_DONTUPDATEFROMPARENT);
        moveableFlags &= ~0xFF00;

        if (preservedFlags) {
            moveableFlags |= VX_MOVEABLE_WORLDALIGNED;
        }
        SetMoveableFlags(moveableFlags);
    }

    if (chunk->SeekIdentifier(CK_STATESAVE_3DENTITYMATRIX)) {
        chunk->Skip(1); // IDA shows a padding byte consumed before the matrix
        chunk->ReadMatrix(worldMatrix);

        // Recompute handedness flag based on loaded matrix
        VxVector row0(worldMatrix[0][0], worldMatrix[0][1], worldMatrix[0][2]);
        VxVector row1(worldMatrix[1][0], worldMatrix[1][1], worldMatrix[1][2]);
        VxVector row2(worldMatrix[2][0], worldMatrix[2][1], worldMatrix[2][2]);
        VxVector cross = CrossProduct(row0, row1);
        float dot = DotProduct(cross, row2);

        CKDWORD moveableFlags = GetMoveableFlags();
        if (dot < 0.0f) {
            moveableFlags |= VX_MOVEABLE_INDIRECTMATRIX;
        } else {
            moveableFlags &= ~VX_MOVEABLE_INDIRECTMATRIX;
        }
        SetMoveableFlags(moveableFlags);
    }

    // Set world matrix respecting file/context rules from original implementation (IDA: 0x1000AE7C..0x1000AF11)
    // - If file != NULL: always SetWorldMatrix
    // - Else: apply the loaded world matrix unless the *preserved* VX_MOVEABLE_WORLDALIGNED gate is set.
    //         If the current mesh has an initial value in the current scene, the gate is cleared and we apply.
    CKDWORD preservedWorldAligned = preservedFlags;
    if (!file) {
        CKMesh *meshForInit = GetCurrentMesh();
        CKScene *currentScene = m_Context ? m_Context->GetCurrentScene() : nullptr;
        if (meshForInit && currentScene && currentScene->GetObjectInitialValue(meshForInit)) {
            preservedWorldAligned = 0;
        }
    }

    if (file || !preservedWorldAligned) {
        SetWorldMatrix(worldMatrix, TRUE);
    }

    // Load skin data (chunk 0x200000) - complex vertex weighting system
    // Based on IDA: checks for chunk, calls CreateSkin(), reads bone/vertex data
    if (chunk->SeekIdentifier(CK_STATESAVE_3DENTITYSKINDATA)) {
        // IDA: `if (SeekIdentifier && CreateSkin())`
        if (!m_Skin) {
            CreateSkin();
        }

        if (m_Skin) {
            const int dataVersion = chunk->GetDataVersion();

            // IDA: if (DataVersion < 6) Skip(1)
            if (dataVersion < 6) {
                chunk->Skip(1);
            }

            // IDA: ReadMatrix into m_ObjectInitMatrix then compute inverse
            chunk->ReadMatrix(m_Skin->m_ObjectInitMatrix);
            Vx3DInverseMatrix(m_Skin->m_InverseWorldMatrix, m_Skin->m_ObjectInitMatrix);

            // IDA: Sequence = StartReadSequence(); SetBoneCount(Sequence)
            const int boneCount = chunk->StartReadSequence();
            m_Skin->SetBoneCount(boneCount);

            // IDA: first loop reads bone objects
            for (int i = 0; i < boneCount; ++i) {
                CK3dEntity *boneEntity = (CK3dEntity *) chunk->ReadObject(m_Context);
                CKSkinBoneData *boneData = m_Skin->GetBoneData(i);
                if (boneData) {
                    boneData->SetBone(boneEntity);
                }
            }

            // IDA: second loop reads bone flags + (optional skip) + matrix
            for (int i = 0; i < boneCount; ++i) {
                const CKDWORD boneFlags = chunk->ReadDword();
                if (dataVersion < 6) {
                    chunk->Skip(1);
                }
                CKSkinBoneData *boneData = m_Skin->GetBoneData(i);
                if (boneData) {
                    if (RCKSkinBoneData *rckBoneData = static_cast<RCKSkinBoneData *>(boneData)) {
                        rckBoneData->SetFlags(boneFlags);
                    }
                    VxMatrix boneInitInv;
                    chunk->ReadMatrix(boneInitInv);
                    boneData->SetBoneInitialInverseMatrix(boneInitInv);
                } else {
                    VxMatrix dummy;
                    chunk->ReadMatrix(dummy);
                }
            }

            // IDA: v44 = ReadInt(); SetVertexCount(v44)
            const int vertexCount = chunk->ReadInt();
            m_Skin->SetVertexCount(vertexCount);

            for (int i = 0; i < vertexCount; ++i) {
                const int vertexBoneCount = chunk->ReadInt();
                RCKSkinVertexData *vertexData = static_cast<RCKSkinVertexData *>(m_Skin->GetVertexData(i));
                if (vertexData) {
                    vertexData->SetBoneCount(vertexBoneCount);
                }

                if (dataVersion < 6) {
                    chunk->Skip(1);
                }

                VxVector initPos;
                chunk->ReadVector(&initPos);
                if (vertexData) {
                    vertexData->SetInitialPos(initPos);
                }

                if (dataVersion < 6) {
                    chunk->Skip(1);
                }

                if (vertexBoneCount > 0 && vertexData) {
                    chunk->ReadAndFillBuffer_LEndian(4 * vertexBoneCount, (void *) vertexData->GetBonesArray());
                } else if (vertexBoneCount > 0) {
                    // Consume data even if vertex storage is missing.
                    // We can't easily skip because format is little-endian arrays; read into a temp buffer.
                    XArray<CKDWORD> tmp;
                    tmp.Resize(vertexBoneCount);
                    chunk->ReadAndFillBuffer_LEndian(4 * vertexBoneCount, tmp.Begin());
                }

                if (dataVersion < 6) {
                    chunk->Skip(1);
                }

                if (vertexBoneCount > 0 && vertexData) {
                    chunk->ReadAndFillBuffer_LEndian(4 * vertexBoneCount, (void *) vertexData->GetWeightsArray());
                } else if (vertexBoneCount > 0) {
                    XArray<float> tmp;
                    tmp.Resize(vertexBoneCount);
                    chunk->ReadAndFillBuffer_LEndian(4 * vertexBoneCount, tmp.Begin());
                }
            }

            // IDA: normals chunk does NOT store a count; it stores exactly 12*vertexCount bytes.
            if (chunk->SeekIdentifier(CK_STATESAVE_3DENTITYSKINDATANORMALS)) {
                m_Skin->SetNormalCount(vertexCount);
                if (vertexCount > 0) {
                    chunk->ReadAndFillBuffer_LEndian(12 * vertexCount, m_Skin->m_Normals.Begin());
                }
            }
        }
    }

    // Set default bounding box if needed
    // Based on IDA: triggered when CK_3DENTITY_BBOXVALID (0x80000) and CK_OBJECT_INTERFACEOBJ are set
    if ((m_3dEntityFlags & CK_3DENTITY_PORTAL) && (m_ObjectFlags & CK_OBJECT_INTERFACEOBJ)) {
        VxBbox defaultBbox;
        defaultBbox.Min = VxVector(-1.0f, -1.0f, -1.0f);
        defaultBbox.Max = VxVector(1.0f, 1.0f, 1.0f);
        SetBoundingBox(&defaultBbox, TRUE);
    }

    return CK_OK;
}

/*************************************************
Summary: PreSave method for RCK3dEntity.
Purpose: Prepares entity for saving by handling dependencies.
Remarks:
- Saves mesh objects (except for curves which don't have meshes)
- Saves object animations for later serialization
- Handles skin-related mesh flags for proper rendering
- Saves children entities if requested by save flags

Implementation based on decompilation at 0x1000A1C2:
- Calls CKBeObject::PreSave() first
- Saves mesh objects if not CKCID_CURVE using XClassArray::Begin/size
- Saves object animations if present using XClassArray::Begin/size
- Sets skin-related mesh flags (0x200000) if m_Skin exists
- Saves children if flags & 0x40000 is set

Arguments:
- file: The file context for saving
- flags: Save flags controlling behavior and dependency handling
*************************************************/
void RCK3dEntity::PreSave(CKFile *file, CKDWORD flags) {
    // Call base class PreSave first to handle basic object dependencies
    CKBeObject::PreSave(file, flags);

    // Save mesh objects (except for curves which don't have meshes)
    // Based on decompilation: checks GetClassID() != CKCID_CURVE
    if (file && GetClassID() != CKCID_CURVE) {
        file->SaveObjects(m_Meshes.Begin(), m_Meshes.Size(), flags);
    }

    // Save object animations
    // Based on decompilation: checks m_ObjectAnimations pointer
    if (file && m_ObjectAnimations) {
        file->SaveObjects(m_ObjectAnimations->Begin(), m_ObjectAnimations->Size(), flags);
    }

    // Handle skin-related mesh flags for proper rendering
    // Based on decompilation: checks m_Skin, calls GetCurrentMesh()
    if (m_Skin) {
        CKMesh *mesh = GetCurrentMesh();
        if (mesh) {
            mesh->SetFlags(mesh->GetFlags() | VXMESH_PROCEDURALPOS);
        }
    }

    // Save children entities if requested by save flags
    // Based on decompilation: checks flags & 0x40000
    if (file && (flags & CK_STATESAVE_3DENTITYHIERARCHY)) {
        file->SaveObjects(m_Children.Begin(), m_Children.Size());
    }
}

/*************************************************
Summary: Save method for RCK3dEntity.
Purpose: Saves entity data to a state chunk.
Remarks:
- Creates and populates state chunk with entity data
- Saves meshes, animations, and transformation data
- Handles complex skin data with bones and vertex weights
- Manages parent-child relationships in scene hierarchy
- Supports both base entity and derived class saving

Implementation based on decompilation at 0x1000A2B2:
- Gets base class chunk with CKBeObject::Save()
- Returns early if !file && (flags & 0x3FF000) == 0
- Creates CKStateChunk with CKCID_3DENTITY
- Saves mesh data (chunk 0x4000) if not curve and has meshes
- Saves object animations (chunk 0x2000) if present
- Saves main entity data (chunk 0x100000):
  - Builds entity flags based on parent, place, priority
  - Writes flags, moveableFlags
  - Writes world matrix as row vectors
  - Writes optional parent, place, priority data
- Saves skin data (chunk 0x200000) if present:
  - Writes m_ObjectInitMatrix
  - Saves bone data: count, objects, matrices
  - Saves vertex data: count, positions, weights, bone indices
  - Saves normal data (chunk 0x1000) if present
- Closes chunk appropriately based on class type

Arguments:
- file: The file context for saving
- flags: Save flags controlling behavior
Return Value:
- CKStateChunk*: The created state chunk containing entity data
*************************************************/
CKStateChunk *RCK3dEntity::Save(CKFile *file, CKDWORD flags) {
    // Get base class chunk first
    CKStateChunk *baseChunk = CKBeObject::Save(file, flags);
    if (!file && !(flags & CK_STATESAVE_3DENTITYONLY)) {
        return baseChunk;
    }

    // Create entity state chunk
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_3DENTITY, file);
    if (!chunk) {
        return baseChunk;
    }

    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    // Save mesh data (chunk 0x4000)
    // Based on IDA: checks GetClassID() != CKCID_CURVE and writes current mesh first
    if (GetClassID() != CKCID_CURVE && (m_CurrentMesh || m_Meshes.Size() > 0)) {
        chunk->WriteIdentifier(CK_STATESAVE_MESHS);
        chunk->WriteObject(m_CurrentMesh);
        m_Meshes.Save(chunk);
    }

    // Save object animations (chunk 0x2000)
    // Based on IDA: checks m_ObjectAnimations pointer and size
    if (m_ObjectAnimations && m_ObjectAnimations->Size() > 0) {
        chunk->WriteIdentifier(CK_STATESAVE_ANIMATION);
        m_ObjectAnimations->Save(chunk);
    }

    // Save main entity data (chunk 0x100000)
    // Based on IDA Save function @0x1000a2b2
    {
        chunk->WriteIdentifier(CK_STATESAVE_3DENTITYNDATA);

        // Get Place object for presence check
        CKObject *placeObject = m_Context->GetObject(m_Place);

        // Build entity flags (stored in m_3dEntityFlags)
        // Based on IDA: conditionally sets/clears presence indicator bits
        // CK_3DENTITY_PARENTVALID (0x20000) - has parent
        if (m_Parent) {
            m_3dEntityFlags |= CK_3DENTITY_PARENTVALID;
        } else {
            m_3dEntityFlags &= ~CK_3DENTITY_PARENTVALID;
        }

        // CK_3DENTITY_PLACEVALID (0x10000) - has place
        if (placeObject) {
            m_3dEntityFlags |= CK_3DENTITY_PLACEVALID;
        } else {
            m_3dEntityFlags &= ~CK_3DENTITY_PLACEVALID;
        }

        // CK_3DENTITY_ZORDERVALID (0x100000) - has priority
        int priority = GetZOrder();
        if (priority != 0) {
            m_3dEntityFlags |= CK_3DENTITY_ZORDERVALID;
        } else {
            m_3dEntityFlags &= ~CK_3DENTITY_ZORDERVALID;
        }

        // Write entity flags and moveable flags
        chunk->WriteDword(m_3dEntityFlags);
        chunk->WriteDword(GetMoveableFlags());

        // Write world matrix as 4 row vectors
        // Based on IDA: uses sub_1000C160 to extract row vectors
        const VxMatrix &mat = GetWorldMatrix();
        VxVector row0(mat[0][0], mat[0][1], mat[0][2]);
        VxVector row1(mat[1][0], mat[1][1], mat[1][2]);
        VxVector row2(mat[2][0], mat[2][1], mat[2][2]);
        VxVector row3(mat[3][0], mat[3][1], mat[3][2]);

        chunk->WriteVector(&row0);
        chunk->WriteVector(&row1);
        chunk->WriteVector(&row2);
        chunk->WriteVector(&row3);

        // Write optional Place reference (if CK_3DENTITY_PLACEVALID)
        if (placeObject) {
            chunk->WriteObject(placeObject);
        }

        // Write optional parent (if CK_3DENTITY_PARENTVALID)
        if (m_Parent) {
            chunk->WriteObject(m_Parent);
        }

        // Write optional priority (if CK_3DENTITY_ZORDERVALID)
        if (priority != 0) {
            chunk->WriteInt(priority);
        }
    }

    // Save skin data (chunk 0x200000)
    // Based on IDA: checks m_Skin, writes bone/vertex data
    if (m_Skin) {
        chunk->WriteIdentifier(CK_STATESAVE_3DENTITYSKINDATA);

        // Write object initialization matrix
        chunk->WriteMatrix(m_Skin->GetObjectInitMatrix());

        // Write bone data: first an object-ID sequence, then flags/matrices
        int boneCount = m_Skin->GetBoneCount();
        chunk->StartObjectIDSequence(boneCount);

        for (int i = 0; i < boneCount; i++) {
            CKSkinBoneData *boneData = m_Skin->GetBoneData(i);
            CK3dEntity *bone = boneData ? boneData->GetBone() : nullptr;
            chunk->WriteObjectSequence(bone);
        }

        for (int i = 0; i < boneCount; i++) {
            CKSkinBoneData *boneData = m_Skin->GetBoneData(i);
            // Flags are unknown in current implementation; write zero to preserve layout
            chunk->WriteDword(0);

            VxMatrix boneMatrix;
            if (boneData) {
                boneMatrix = static_cast<RCKSkinBoneData *>(boneData)->GetInitialInverseMatrix();
            } else {
                boneMatrix.SetIdentity();
            }
            chunk->WriteMatrix(boneMatrix);
        }

        // Write vertex data
        int vertexCount = m_Skin->GetVertexCount();
        chunk->WriteInt(vertexCount);

        for (int i = 0; i < vertexCount; i++) {
            RCKSkinVertexData *vertexData = static_cast<RCKSkinVertexData *>(m_Skin->GetVertexData(i));
            if (!vertexData) {
                chunk->WriteInt(0);
                VxVector zero(0.0f, 0.0f, 0.0f);
                chunk->WriteVector(&zero);
                continue;
            }

            int vertexBoneCount = vertexData->GetBoneCount();
            chunk->WriteInt(vertexBoneCount);

            VxVector &initPos = vertexData->GetInitialPos();
            chunk->WriteVector(&initPos);

            if (vertexBoneCount > 0) {
                chunk->WriteBufferNoSize_LEndian(4 * vertexBoneCount, vertexData->GetBonesArray());
                chunk->WriteBufferNoSize_LEndian(4 * vertexBoneCount, vertexData->GetWeightsArray());
            }
        }

        // Write normal data if present (chunk 0x1000)
        int normalCount = m_Skin->GetNormalCount();
        if (normalCount > 0 && normalCount == m_Skin->GetVertexCount()) {
            chunk->WriteIdentifier(CK_STATESAVE_3DENTITYSKINDATANORMALS);
            chunk->WriteInt(normalCount);

            for (int i = 0; i < normalCount; i++) {
                VxVector &normal = m_Skin->GetNormal(i);
                chunk->WriteVector(&normal);
            }
        }
    }

    // Close chunk appropriately based on class type
    // Based on IDA: checks GetClassID() == CKCID_3DENTITY
    if (GetClassID() == CKCID_3DENTITY)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();

    return chunk;
}

// =====================================================
// RCK3dEntity Constructor/Destructor
// =====================================================

RCK3dEntity::RCK3dEntity(CKContext *Context, CKSTRING name) : RCKRenderObject(Context, name) {
    m_Place = 0;
    m_Parent = nullptr;
    m_3dEntityFlags = 0,
    m_CurrentMesh = nullptr;
    m_ObjectAnimations = nullptr,
    m_Skin = nullptr,
    m_LastFrameMatrix = nullptr,

    m_LocalMatrix = VxMatrix::Identity();
    m_WorldMatrix = VxMatrix::Identity();
    m_InverseWorldMatrix = VxMatrix::Identity();

    m_MoveableFlags = VX_MOVEABLE_PICKABLE | VX_MOVEABLE_VISIBLE | VX_MOVEABLE_WORLDALIGNED | VX_MOVEABLE_RENDERCHANNELS;

    // Create scene graph node - critical for rendering!
    // Based on IDA analysis at 0x10005086: RCK3dEntity constructor calls RCKRenderManager::CreateNode
    RCKRenderManager *renderManager = (RCKRenderManager *) m_Context->GetRenderManager();
    if (renderManager) {
        m_SceneGraphNode = renderManager->CreateNode(this);
    }
}

/*************************************************
Summary: Destructor for RCK3dEntity.
Purpose: Cleans up all allocated resources.

Implementation based on decompilation at 0x100050b5:
- Sets vftable pointer
- Destroys skin if present
- Frees object animations array
- Deletes m_LastFrameMatrix (operator delete)
- Destroys scene graph node
- Clears children array
- Clears meshes array
- Calls base class destructor
*************************************************/
RCK3dEntity::~RCK3dEntity() {
    // Destroy skin if present
    if (m_Skin) {
        RCK3dEntity::DestroySkin();
    }
}

CK_CLASSID RCK3dEntity::GetClassID() {
    return m_ClassID;
}

int RCK3dEntity::GetMemoryOccupation() {
    // IDA: 0x1000A0AC
    int size = RCKRenderObject::GetMemoryOccupation() + (sizeof(RCK3dEntity) - sizeof(RCKRenderObject));
    size += m_Meshes.GetMemoryOccupation(FALSE);
    size += m_Children.GetMemoryOccupation(FALSE);

    if (m_Callbacks) {
        size += m_Callbacks->m_PreCallBacks.GetMemoryOccupation(FALSE);
        size += m_Callbacks->m_PostCallBacks.GetMemoryOccupation(FALSE);
        size += sizeof(CKCallbacksContainer);
    }

    return size;
}

// =====================================================
// Parent/Child Hierarchy Methods
// =====================================================

int RCK3dEntity::GetChildrenCount() const {
    return m_Children.Size();
}

CK3dEntity *RCK3dEntity::GetChild(int pos) const {
    return (CK3dEntity *) m_Children[pos];
}

CKBOOL RCK3dEntity::SetParent(CK3dEntity *Parent, CKBOOL KeepWorldPos) {
    RCK3dEntity *parent = (RCK3dEntity *) Parent;

    // Cannot parent to self
    if (parent == this)
        return FALSE;

    // Already has this parent
    if (m_Parent == parent)
        return TRUE;

    // Cannot parent a Place to another Place
    if (CKIsChildClassOf(this, CKCID_PLACE) && parent && CKIsChildClassOf(parent, CKCID_PLACE))
        return FALSE;

    // Check for cycles - cannot parent to a descendant
    if (parent) {
        for (RCK3dEntity *ancestor = parent->m_Parent; ancestor; ancestor = ancestor->m_Parent) {
            if (ancestor == this)
                return FALSE;
        }
    }

    // Remove from old parent's children array
    if (m_Parent) {
        m_Parent->m_Children.Remove(this);
    }

    // Remove this node from its current scene graph parent (if it has one)
    if (m_SceneGraphNode && m_SceneGraphNode->m_Parent) {
        m_SceneGraphNode->m_Parent->RemoveNode(m_SceneGraphNode);
    }

    // Set new parent
    m_Parent = parent;

    // Add to new parent
    if (parent) {
        // Add to parent's children array
        parent->m_Children.PushBack(this);
        // Add to parent's scene graph node
        if (parent->m_SceneGraphNode && m_SceneGraphNode) {
            parent->m_SceneGraphNode->AddNode(m_SceneGraphNode);
        }
    } else {
        // No parent - check if object should be added to root scene graph
        // Only add to root if not marked for deletion (CK_OBJECT_TOBEDELETED = 0x10)
        if (!IsToBeDeleted() && m_SceneGraphNode) {
            RCKRenderManager *rm = (RCKRenderManager *) m_Context->GetRenderManager();
            if (rm && rm->GetRootNode()) {
                rm->GetRootNode()->AddNode(m_SceneGraphNode);
            }
        }
    }

    // Handle matrix update
    if (KeepWorldPos) {
        if (m_Parent) {
            // Recompute local matrix: LocalMatrix = InverseParentWorld * WorldMatrix
            const VxMatrix &invParent = m_Parent->GetInverseWorldMatrix();
            Vx3DMultiplyMatrix(m_LocalMatrix, invParent, m_WorldMatrix);
        } else {
            // No parent - local = world
            m_LocalMatrix = m_WorldMatrix;
        }
    } else {
        // Local matrix stays, world matrix needs recalculation
        LocalMatrixChanged(FALSE, TRUE);
    }

    // Update Place reference
    CK_ID newPlace = 0;
    if (CKIsChildClassOf(parent, CKCID_PLACE)) {
        // Parent is a Place - use it as reference
        newPlace = parent ? parent->GetID() : 0;
    } else if (parent) {
        // Inherit parent's Place
        newPlace = parent->m_Place;
    }

    if (newPlace != m_Place) {
        UpdatePlace(newPlace);
    }

    return TRUE;
}

CK3dEntity *RCK3dEntity::GetParent() const {
    return (CK3dEntity *) m_Parent;
}

CKBOOL RCK3dEntity::AddChild(CK3dEntity *Child, CKBOOL KeepWorldPos) {
    if (!Child)
        return FALSE;
    return ((RCK3dEntity *) Child)->SetParent((CK3dEntity *) this, KeepWorldPos);
}

CKBOOL RCK3dEntity::AddChildren(const XObjectPointerArray &Children, CKBOOL KeepWorldPos) {
    for (CKObject **it = Children.Begin(); it != Children.End(); ++it) {
        if (*it) {
            (*it)->ModifyObjectFlags(CK_OBJECT_TEMPMARKER, 0);
        }
    }

    // Second pass: Add children that are 3D entities
    int result = 0;
    for (CKObject **it = Children.Begin(); it != Children.End(); ++it) {
        if (CKIsChildClassOf(*it, CKCID_3DENTITY)) {
            RCK3dEntity *entity = (RCK3dEntity *) *it;
            // Only add if no parent OR parent doesn't have marker flag
            if (!entity->GetParent() || !(entity->GetParent()->GetObjectFlags() & CK_OBJECT_TEMPMARKER)) {
                result |= entity->SetParent((CK3dEntity *) this, KeepWorldPos);
            }
        }
    }

    // Third pass: Clear the marker flag
    for (CKObject **it = Children.Begin(); it != Children.End(); ++it) {
        if (*it) {
            (*it)->ModifyObjectFlags(0, CK_OBJECT_TEMPMARKER);
        }
    }

    return result;
}

CKBOOL RCK3dEntity::RemoveChild(CK3dEntity *Mov) {
    if (!Mov)
        return FALSE;
    return Mov->SetParent(nullptr, TRUE);
}

CKBOOL RCK3dEntity::CheckIfSameKindOfHierarchy(CK3dEntity *Mov, CKBOOL SameRecur) const {
    // IDA: CheckIfSameKindOfHierarchy at 0x1000980D
    if (!Mov)
        return FALSE;

    int myChildCount = GetChildrenCount();
    int otherChildCount = Mov->GetChildrenCount();

    // Child counts must match
    if (myChildCount != otherChildCount)
        return FALSE;

    if (SameRecur) {
        // Same order recursion - check children in order
        for (int i = 0; i < myChildCount; i++) {
            RCK3dEntity *myChild = (RCK3dEntity *) this->GetChild(i);
            CK3dEntity *otherChild = Mov->GetChild(i);
            if (!myChild->CheckIfSameKindOfHierarchy(otherChild, TRUE))
                return FALSE;
        }
        return TRUE;
    } else {
        // Non-ordered matching - find any matching child
        for (int i = 0; i < myChildCount; i++) {
            RCK3dEntity *myChild = (RCK3dEntity *) this->GetChild(i);
            int k;
            for (k = 0; k < otherChildCount; k++) {
                CK3dEntity *otherChild = Mov->GetChild(k);
                if (myChild->CheckIfSameKindOfHierarchy(otherChild, FALSE))
                    break;
            }
            if (k >= otherChildCount)
                return FALSE;
        }
        return TRUE;
    }
}

CK3dEntity *RCK3dEntity::HierarchyParser(CK3dEntity *current) const {
    // IDA: HierarchyParser at 0x10009715
    RCK3dEntity *entity = (RCK3dEntity *) current;

    if (current) {
        // If current has children, return first child
        if (entity->m_Children.Size() > 0) {
            return entity->GetChild(0);
        }

        // Walk up the hierarchy looking for next sibling
        while (true) {
            RCK3dEntity *parent = entity->m_Parent;
            if (!parent)
                return nullptr;

            // Find current's index in parent's children
            int childCount = parent->m_Children.Size();
            int idx = 0;
            do {
                if (parent->m_Children[idx] == entity)
                    break;
                idx++;
            } while (idx < childCount);
            idx++; // Move to next sibling

            // If there's a next sibling, return it
            if (idx != childCount) {
                return parent->GetChild(idx);
            }

            // Stop if we've reached the root (this object)
            if (parent == this)
                return nullptr;

            // Move up to parent and continue
            entity = parent;
        }
    } else {
        // current is NULL - start from this object's first child
        if (m_Children.Size() > 0) {
            return this->GetChild(0);
        }
        return nullptr;
    }
}

// =====================================================
// Flags Methods
// =====================================================

CKDWORD RCK3dEntity::GetFlags() const {
    return m_3dEntityFlags;
}

/*************************************************
Summary: SetFlags - Sets the 3D entity flags.
Purpose: Stores flags to m_3dEntityFlags and handles CK_3DENTITY_UPDATELASTFRAME changes.

Implementation based on IDA decompilation at 0x100055dd:
- Saves old CK_3DENTITY_UPDATELASTFRAME (0x1000) state
- Stores new flags to m_3dEntityFlags (offset 0x74)
- If UPDATELASTFRAME changed:
  - If cleared: Unregister from render manager, delete m_LastFrameMatrix
  - If set: Register with render manager
*************************************************/
void RCK3dEntity::SetFlags(CKDWORD Flags) {
    const CKDWORD oldUpdateLastFrame = m_3dEntityFlags & CK_3DENTITY_UPDATELASTFRAME;
    m_3dEntityFlags = Flags;

    const CKBOOL updateLastFrame = (m_3dEntityFlags & CK_3DENTITY_UPDATELASTFRAME) != 0;
    RCKRenderManager *rm = (RCKRenderManager *) m_Context->GetRenderManager();

    // Register/unregister entity for last-frame tracking based on flag transition
    if (oldUpdateLastFrame && !updateLastFrame) {
        if (rm) {
            rm->UnregisterLastFrameEntity(this);
        }
        if (m_LastFrameMatrix) {
            delete m_LastFrameMatrix;
            m_LastFrameMatrix = nullptr;
        }
    } else if (!oldUpdateLastFrame && updateLastFrame) {
        if (rm) {
            rm->RegisterLastFrameEntity(this);
        }
    }
}

void RCK3dEntity::SetPickable(CKBOOL Pick) {
    if (Pick) {
        m_MoveableFlags |= VX_MOVEABLE_PICKABLE;
    } else {
        m_MoveableFlags &= ~VX_MOVEABLE_PICKABLE;
    }
}

CKBOOL RCK3dEntity::IsPickable() const {
    return (m_MoveableFlags & VX_MOVEABLE_PICKABLE) != 0;
}

void RCK3dEntity::SetRenderChannels(CKBOOL RenderChannels) {
    if (RenderChannels) {
        m_MoveableFlags |= VX_MOVEABLE_RENDERCHANNELS;
    } else {
        m_MoveableFlags &= ~VX_MOVEABLE_RENDERCHANNELS;
    }
}

CKBOOL RCK3dEntity::AreRenderChannelsVisible() const {
    return (m_MoveableFlags & VX_MOVEABLE_RENDERCHANNELS) != 0;
}

void RCK3dEntity::IgnoreAnimations(CKBOOL ignore) {
    // IDA: 0x10005AC6
    // Toggles CK_3DENTITY_IGNOREANIMATION (0x400) in 3D entity flags.
    const CKDWORD flags = GetFlags();
    if (ignore) {
        SetFlags(flags | CK_3DENTITY_IGNOREANIMATION);
    } else {
        SetFlags(flags & ~CK_3DENTITY_IGNOREANIMATION);
    }
}

CKBOOL RCK3dEntity::AreAnimationIgnored() const {
    // IDA: 0x10005AF9
    return (GetFlags() & CK_3DENTITY_IGNOREANIMATION) != 0;
}

CKBOOL RCK3dEntity::IsAllInsideFrustrum() const {
    return m_SceneGraphNode && m_SceneGraphNode->CheckHierarchyFrustum();
}

CKBOOL RCK3dEntity::IsAllOutsideFrustrum() const {
    return m_SceneGraphNode && m_SceneGraphNode->IsAllOutsideFrustum();
}

void RCK3dEntity::SetRenderAsTransparent(CKBOOL Trans) {
    if (Trans) {
        m_MoveableFlags |= VX_MOVEABLE_RENDERLAST;
    } else {
        m_MoveableFlags &= ~VX_MOVEABLE_RENDERLAST;
    }
}

CKDWORD RCK3dEntity::GetMoveableFlags() const {
    return m_MoveableFlags;
}

/*************************************************
Summary: SetMoveableFlags - Sets the moveable flags to a specific value.
Purpose: Computes which flags to add/remove and delegates to ModifyMoveableFlags.

Implementation based on decompilation at 0x10005751:
- Computes Add = (~(~Flags & m_MoveableFlags)) & (Flags ^ m_MoveableFlags)
- Computes Remove = ~Flags & m_MoveableFlags
- Calls ModifyMoveableFlags(Add, Remove)
This ensures proper side effects (visibility sync, scene graph notification)
*************************************************/
void RCK3dEntity::SetMoveableFlags(CKDWORD Flags) {
    // IDA: return this->ModifyMoveableFlags(
    //         this,
    //         ~(~a2 & this->m_MoveableFlags) & (a2 ^ this->m_MoveableFlags),
    //         ~a2 & this->m_MoveableFlags);
    CKDWORD toRemove = ~Flags & m_MoveableFlags;
    CKDWORD toAdd = ~(~Flags & m_MoveableFlags) & (Flags ^ m_MoveableFlags);
    ModifyMoveableFlags(toAdd, toRemove);
}

/*************************************************
Summary: ModifyMoveableFlags - Modifies moveable flags with proper side effects.
Purpose: Adds and removes flags, syncs with object flags, notifies scene graph.

Implementation based on decompilation at 0x1000579f:
- Clears 'Remove' bits from m_MoveableFlags
- Sets 'Add' bits to m_MoveableFlags
- If VX_MOVEABLE_VISIBLE or VX_MOVEABLE_HIERARCHICALHIDE changed:
  - Sync with m_ObjectFlags (CK_OBJECT_VISIBLE = 0x1000, CK_OBJECT_HIERACHICALHIDE = 0x200)
  - Notify scene graph node via EntityFlagsChanged
- If VX_MOVEABLE_RENDERLAST or VX_MOVEABLE_RENDERFIRST changed:
  - Update scene graph node priority (10000 for RENDERFIRST, 0 otherwise)
*************************************************/
CKDWORD RCK3dEntity::ModifyMoveableFlags(CKDWORD Add, CKDWORD Remove) {
    // Apply flag changes
    m_MoveableFlags &= ~Remove;
    m_MoveableFlags |= Add;

    // Check if visibility-related flags changed
    if (((Remove | Add) & (VX_MOVEABLE_VISIBLE | VX_MOVEABLE_HIERARCHICALHIDE)) != 0) {
        // Sync VX_MOVEABLE_VISIBLE with CK_OBJECT_VISIBLE
        if ((Remove & VX_MOVEABLE_VISIBLE) != 0) {
            // Clear CK_OBJECT_VISIBLE (0x1000) - IDA mask: 0xFFFFEFFF
            m_ObjectFlags &= ~CK_OBJECT_VISIBLE;
        }
        if ((Add & VX_MOVEABLE_VISIBLE) != 0) {
            // Set CK_OBJECT_VISIBLE
            m_ObjectFlags |= CK_OBJECT_VISIBLE;
        }

        // Sync VX_MOVEABLE_HIERARCHICALHIDE with CK_OBJECT_HIERACHICALHIDE
        if ((Remove & VX_MOVEABLE_HIERARCHICALHIDE) != 0) {
            // Clear CK_OBJECT_HIERACHICALHIDE (0x200)
            m_ObjectFlags &= ~CK_OBJECT_HIERACHICALHIDE;
        }
        if ((Add & VX_MOVEABLE_HIERARCHICALHIDE) != 0) {
            // Set CK_OBJECT_HIERACHICALHIDE
            m_ObjectFlags |= CK_OBJECT_HIERACHICALHIDE;
        }

        // Notify scene graph node
        if (m_SceneGraphNode) {
            m_SceneGraphNode->EntityFlagsChanged(TRUE);
        }
    }

    // Check if render order flags changed
    if (((Remove | Add) & (VX_MOVEABLE_RENDERLAST | VX_MOVEABLE_RENDERFIRST)) != 0) {
        if (m_SceneGraphNode) {
            if ((Remove & VX_MOVEABLE_RENDERFIRST) != 0) {
                m_SceneGraphNode->SetPriority(0, 0);
            }
            if ((Add & VX_MOVEABLE_RENDERFIRST) != 0) {
                m_SceneGraphNode->SetPriority(10000, 0);
            }
        }
    }

    return m_MoveableFlags;
}

// =====================================================
// Mesh Methods
// =====================================================

CKMesh *RCK3dEntity::GetCurrentMesh() const {
    return m_CurrentMesh;
}

CKMesh *RCK3dEntity::SetCurrentMesh(CKMesh *m, CKBOOL add_if_not_here) {
    CKMesh *old = m_CurrentMesh;

    if (old == m) {
        return old;
    }

    if (m) {
        SetBoundingBox(nullptr);
    }

    m_CurrentMesh = (RCKMesh *) m;

    // Changing mesh invalidates cached boxes/skin update state.
    m_MoveableFlags &= ~VX_MOVEABLE_UPTODATE;
    if (m_SceneGraphNode)
        m_SceneGraphNode->InvalidateBox(TRUE);

    if (m_CurrentMesh) {
        if (add_if_not_here)
            AddMesh(m);
    }

    return old;
}

int RCK3dEntity::GetMeshCount() const {
    return m_Meshes.Size();
}

CKMesh *RCK3dEntity::GetMesh(int pos) const {
    if (pos < 0 || pos >= m_Meshes.Size())
        return nullptr;
    return (CKMesh *) m_Meshes[pos];
}

CKERROR RCK3dEntity::AddMesh(CKMesh *mesh) {
    if (!mesh)
        return CKERR_INVALIDPARAMETER;

    m_Meshes.AddIfNotHere(mesh);

    return CK_OK;
}

CKERROR RCK3dEntity::RemoveMesh(CKMesh *mesh) {
    if (!mesh)
        return CKERR_INVALIDPARAMETER;

    if (!m_Meshes.Erase(mesh))
        return CKERR_NOTFOUND;

    if (m_CurrentMesh == mesh) {
        CKMesh *newCurrent = m_Meshes.Size() > 0 ? (CKMesh *) m_Meshes[0] : nullptr;
        SetCurrentMesh(newCurrent, FALSE);
    }

    return CK_OK;
}

// =====================================================
// Transform Methods
// =====================================================

void RCK3dEntity::LookAt(const VxVector *Pos, CK3dEntity *Ref, CKBOOL KeepChildren) {
    // IDA @ 0x10007D68
    // - Reads current WORLD position from world matrix row 3
    // - If Ref: transforms Pos by Ref->Transform(..., Ref=NULL)
    // - dir = targetWorld - currentWorldPos
    // - If |dir| < 1.1920929e-7, do nothing
    // - If Cross(dir, currentDirAxis) is zero, do nothing
    // - dir = Normalize(dir)
    // - right = Cross(axisY, dir)
    // - up    = Cross(dir, right)
    // - If up.y == 0, do nothing
    // - If Dot(up, oldUpAxis) < 0, negate up and right (avoid flip)
    // - Calls SetOrientation(dir, up, right, Ref=NULL)

    const VxVector currentWorldPos(m_WorldMatrix[3][0], m_WorldMatrix[3][1], m_WorldMatrix[3][2]);

    VxVector targetWorld;
    if (Ref) {
        Ref->Transform(&targetWorld, Pos, nullptr);
    } else {
        targetWorld = *Pos;
    }

    VxVector dir = targetWorld - currentWorldPos;
    const float dirLen = Magnitude(dir);
    if (dirLen < EPSILON) {
        return;
    }

    const VxVector currentDirAxis(m_WorldMatrix[2][0], m_WorldMatrix[2][1], m_WorldMatrix[2][2]);
    const VxVector dirCrossCurrent = CrossProduct(dir, currentDirAxis);
    if (SquareMagnitude(dirCrossCurrent) == 0.0f) {
        return;
    }

    dir = dir * (1.0f / dirLen);

    const VxVector *axisY = &VxVector::axisY();
    VxVector right = CrossProduct(*axisY, dir);
    VxVector up = CrossProduct(dir, right);

    if (fabsf(up.y) == 0.0f) {
        return;
    }

    const VxVector oldUpAxis(m_WorldMatrix[1][0], m_WorldMatrix[1][1], m_WorldMatrix[1][2]);
    if (DotProduct(up, oldUpAxis) < 0.0f) {
        right = right * -1.0f;
        up = up * -1.0f;
    }

    SetOrientation(&dir, &up, &right, nullptr, KeepChildren);
}

void RCK3dEntity::Rotate3f(float X, float Y, float Z, float Angle, CK3dEntity *Ref, CKBOOL KeepChildren) {
    // IDA: 0x10008100
    // If Angle == 0, do nothing (otherwise calls Rotate(vector,...)).
    if (Angle == 0.0f)
        return;

    VxVector axis(X, Y, Z);
    Rotate(&axis, Angle, Ref, KeepChildren);
}

/*************************************************
Summary: Rotate - Rotates the entity around an axis by an angle.
Purpose: Applies a rotation to the entity's orientation.

Implementation based on decompilation at 0x100077f5:
- Check if angle is significant (> ~0.00000012)
- If Ref, transform axis from Ref space to world space using TransformVector
- Build rotation matrix from (transformed) axis and angle
- Multiply rotation by current world matrix
- Preserve original position (row 3)
- Call SetWorldMatrix to apply
*************************************************/
void RCK3dEntity::Rotate(const VxVector *Axis, float Angle, CK3dEntity *Ref, CKBOOL KeepChildren) {
    // IDA: if ( sub_100518A0(Angle) >= 0.00000011920929 ) - check angle threshold
    if (fabsf(Angle) < EPSILON)
        return;

    // IDA takes the position from GetWorldMatrix()
    const VxMatrix &world = GetWorldMatrix();
    VxVector originalPos(world[3][0], world[3][1], world[3][2]);

    // Transform axis if Ref is provided
    VxVector worldAxis;
    if (Ref) {
        // IDA: Vx3DRotateVector(&v10, &Ref->m_WorldMatrix, Axis);
        Vx3DRotateVector(&worldAxis, Ref->GetWorldMatrix(), Axis);
    } else {
        worldAxis = *Axis;
    }

    // Build rotation matrix from axis-angle
    VxMatrix rot;
    Vx3DMatrixFromRotation(rot, worldAxis, Angle);

    // IDA: Vx3DMultiplyMatrix(&v11, &v8, &this->m_WorldMatrix);
    VxMatrix newWorld;
    Vx3DMultiplyMatrix(newWorld, rot, m_WorldMatrix);

    // IDA: Preserve original position in row 3
    newWorld[3][0] = originalPos.x;
    newWorld[3][1] = originalPos.y;
    newWorld[3][2] = originalPos.z;

    // IDA: this->SetWorldMatrix(this, &v11, KeepChildren);
    SetWorldMatrix(newWorld, KeepChildren);
}

void RCK3dEntity::Translate3f(float X, float Y, float Z, CK3dEntity *Ref, CKBOOL KeepChildren) {
    VxVector trans(X, Y, Z);
    Translate(&trans, Ref, KeepChildren);
}

/*************************************************
Summary: Translate - Translates the entity by a vector.
Purpose: Adds a displacement to the current world position.

Implementation based on decompilation at 0x10006bf9:
- If Ref, transform the vector from Ref space to world space (direction only)
- Add the transformed vector to current world position
- Call WorldPositionChanged to propagate
*************************************************/
void RCK3dEntity::Translate(const VxVector *Vect, CK3dEntity *Ref, CKBOOL KeepChildren) {
    // IDA: v12 = *Vect; if (Ref) Ref->TransformVector(Ref, &v12, Vect, 0);
    VxVector trans = *Vect;
    if (Ref) {
        // Transform direction vector from Ref's local space to world space
        Ref->TransformVector(&trans, Vect, nullptr);
    }

    // IDA: Add to world matrix row 3 (translation)
    m_WorldMatrix[3][0] += trans.x;
    m_WorldMatrix[3][1] += trans.y;
    m_WorldMatrix[3][2] += trans.z;

    // Propagate changes
    WorldPositionChanged(KeepChildren, 1);
}

void RCK3dEntity::AddScale3f(float X, float Y, float Z, CKBOOL KeepChildren, CKBOOL Local) {
    VxVector scale(X, Y, Z);
    AddScale(&scale, KeepChildren, Local);
}

void RCK3dEntity::AddScale(const VxVector *Scale, CKBOOL KeepChildren, CKBOOL Local) {
    // IDA @ 0x10007C09
    // - Builds a diagonal scale matrix (0 components are clamped to 1e-6)
    // - Multiplies current matrix by that scale matrix
    // - Calls SetLocalMatrix/SetWorldMatrix (not SetScale)
    VxMatrix scaleMat;
    scaleMat.SetIdentity();

    constexpr float kEps = 0.000001f;
    const float sx = (Scale->x == 0.0f) ? kEps : Scale->x;
    const float sy = (Scale->y == 0.0f) ? kEps : Scale->y;
    const float sz = (Scale->z == 0.0f) ? kEps : Scale->z;

    scaleMat[0][0] = sx;
    scaleMat[1][1] = sy;
    scaleMat[2][2] = sz;

    VxMatrix newMat;
    if (Local) {
        Vx3DMultiplyMatrix(newMat, m_LocalMatrix, scaleMat);
        SetLocalMatrix(newMat, KeepChildren);
    } else {
        Vx3DMultiplyMatrix(newMat, m_WorldMatrix, scaleMat);
        SetWorldMatrix(newMat, KeepChildren);
    }
}

void RCK3dEntity::SetPosition3f(float X, float Y, float Z, CK3dEntity *Ref, CKBOOL KeepChildren) {
    VxVector pos(X, Y, Z);
    SetPosition(&pos, Ref, KeepChildren);
}

/*************************************************
Summary: SetPosition - Sets the position of the entity.
Purpose: Modifies the translation component of the world matrix.

Implementation based on decompilation at 0x10006ad8:
- Gets pointer to row 3 of world matrix (translation component)
- If Ref, transforms position from Ref space to world space
- Otherwise, directly copies position
- Calls WorldPositionChanged to update matrices and propagate to children
*************************************************/
void RCK3dEntity::SetPosition(const VxVector *Pos, CK3dEntity *Ref, CKBOOL KeepChildren) {
    // IDA: v4 = sub_1000C160(&this->m_WorldMatrix, 3); // Get row 3 (translation)
    //      v5 = sub_1000CC40(v4, 0); // Get column 0
    //      if ( Ref ) Ref->Transform(Ref, (VxVector *)v5, Pos, 0);
    //      else memcpy(v5, Pos, 0xCu);
    //      RCK3dEntity::WorldPositionChanged(this, KeepChildren, 1);

    // Get pointer to translation component (row 3) of world matrix
    VxVector *worldPos = &m_WorldMatrix[3];

    if (Ref) {
        // IDA writes directly into the translation vector via Ref->Transform(..., 0)
        Ref->Transform(worldPos, Pos, nullptr);
    } else {
        // Direct copy (12 bytes)
        memcpy(worldPos, Pos, sizeof(VxVector));
    }

    // Propagate changes - keepScale=1 preserves scale in bounding box calculations
    WorldPositionChanged(KeepChildren, TRUE);
}

/*************************************************
Summary: GetPosition - Gets the position of the entity.
Purpose: Returns the translation component of the world or local matrix.

Implementation based on decompilation at 0x10006b51:
- If Ref is our parent: return local matrix translation (position relative to parent)
- If Ref is another entity: return world position transformed to Ref's space
- If Ref is null: return world position directly
*************************************************/
void RCK3dEntity::GetPosition(VxVector *Pos, CK3dEntity *Ref) const {
    if (Ref) {
        if (Ref == (CK3dEntity *) m_Parent) {
            // IDA: Return local matrix row 3 (position relative to parent)
            Pos->x = m_LocalMatrix[3][0];
            Pos->y = m_LocalMatrix[3][1];
            Pos->z = m_LocalMatrix[3][2];
        } else {
            // IDA passes a pointer to the world translation vector into Ref->InverseTransform(..., 0)
            const VxVector *worldPos = &m_WorldMatrix[3];
            Ref->InverseTransform(Pos, worldPos, nullptr);
        }
    } else {
        // Return world matrix row 3 (absolute world position)
        Pos->x = m_WorldMatrix[3][0];
        Pos->y = m_WorldMatrix[3][1];
        Pos->z = m_WorldMatrix[3][2];
    }
}

void RCK3dEntity::SetOrientation(const VxVector *Dir, const VxVector *Up, const VxVector *Right, CK3dEntity *Ref, CKBOOL KeepChildren) {
    // IDA @ 0x10006CC3
    constexpr float kEps = 0.000001f;

    VxVector prevScale;
    GetScale(&prevScale, FALSE);
    if (prevScale.x == 0.0f) prevScale.x = kEps;
    if (prevScale.y == 0.0f) prevScale.y = kEps;
    if (prevScale.z == 0.0f) prevScale.z = kEps;

    // Write orientation axes (0.0f components are forced to 1e-6 before normalize+scale).
    if (Ref) {
        VxVector rotDir;
        VxVector rotUp;
        VxVector rotRight;
        Vx3DRotateVector(&rotDir, Ref->GetWorldMatrix(), Dir);
        Vx3DRotateVector(&rotUp, Ref->GetWorldMatrix(), Up);
        if (Right) {
            Vx3DRotateVector(&rotRight, Ref->GetWorldMatrix(), Right);
        } else {
            rotRight = CrossProduct(rotUp, rotDir);
        }

        m_WorldMatrix[0][0] = (rotRight.x == 0.0f) ? kEps : rotRight.x;
        m_WorldMatrix[0][1] = (rotRight.y == 0.0f) ? kEps : rotRight.y;
        m_WorldMatrix[0][2] = (rotRight.z == 0.0f) ? kEps : rotRight.z;

        m_WorldMatrix[1][0] = (rotUp.x == 0.0f) ? kEps : rotUp.x;
        m_WorldMatrix[1][1] = (rotUp.y == 0.0f) ? kEps : rotUp.y;
        m_WorldMatrix[1][2] = (rotUp.z == 0.0f) ? kEps : rotUp.z;

        m_WorldMatrix[2][0] = (rotDir.x == 0.0f) ? kEps : rotDir.x;
        m_WorldMatrix[2][1] = (rotDir.y == 0.0f) ? kEps : rotDir.y;
        m_WorldMatrix[2][2] = (rotDir.z == 0.0f) ? kEps : rotDir.z;
    } else {
        if (Right) {
            m_WorldMatrix[0][0] = (Right->x == 0.0f) ? kEps : Right->x;
            m_WorldMatrix[0][1] = (Right->y == 0.0f) ? kEps : Right->y;
            m_WorldMatrix[0][2] = (Right->z == 0.0f) ? kEps : Right->z;
        } else {
            const VxVector cross = CrossProduct(*Up, *Dir);
            m_WorldMatrix[0][0] = (cross.x == 0.0f) ? kEps : cross.x;
            m_WorldMatrix[0][1] = (cross.y == 0.0f) ? kEps : cross.y;
            m_WorldMatrix[0][2] = (cross.z == 0.0f) ? kEps : cross.z;
        }

        m_WorldMatrix[1][0] = (Up->x == 0.0f) ? kEps : Up->x;
        m_WorldMatrix[1][1] = (Up->y == 0.0f) ? kEps : Up->y;
        m_WorldMatrix[1][2] = (Up->z == 0.0f) ? kEps : Up->z;

        m_WorldMatrix[2][0] = (Dir->x == 0.0f) ? kEps : Dir->x;
        m_WorldMatrix[2][1] = (Dir->y == 0.0f) ? kEps : Dir->y;
        m_WorldMatrix[2][2] = (Dir->z == 0.0f) ? kEps : Dir->z;
    }

    // Re-apply existing per-axis scale.
    // IDA does this in two steps: normalizedAxis = axis * (1.0 / Magnitude(axis)) ; scaledAxis = normalizedAxis * savedScale.
    {
        const VxVector axis(m_WorldMatrix[0][0], m_WorldMatrix[0][1], m_WorldMatrix[0][2]);
        const float invMag = static_cast<float>(1.0 / static_cast<double>(axis.Magnitude()));
        const VxVector normalized = axis * invMag;
        const VxVector scaled = normalized * prevScale.x;
        m_WorldMatrix[0][0] = scaled.x;
        m_WorldMatrix[0][1] = scaled.y;
        m_WorldMatrix[0][2] = scaled.z;
    }
    {
        const VxVector axis(m_WorldMatrix[1][0], m_WorldMatrix[1][1], m_WorldMatrix[1][2]);
        const float invMag = static_cast<float>(1.0 / static_cast<double>(axis.Magnitude()));
        const VxVector normalized = axis * invMag;
        const VxVector scaled = normalized * prevScale.y;
        m_WorldMatrix[1][0] = scaled.x;
        m_WorldMatrix[1][1] = scaled.y;
        m_WorldMatrix[1][2] = scaled.z;
    }
    {
        const VxVector axis(m_WorldMatrix[2][0], m_WorldMatrix[2][1], m_WorldMatrix[2][2]);
        const float invMag = static_cast<float>(1.0 / static_cast<double>(axis.Magnitude()));
        const VxVector normalized = axis * invMag;
        const VxVector scaled = normalized * prevScale.z;
        m_WorldMatrix[2][0] = scaled.x;
        m_WorldMatrix[2][1] = scaled.y;
        m_WorldMatrix[2][2] = scaled.z;
    }

    WorldMatrixChanged(KeepChildren, TRUE);
}

void RCK3dEntity::GetOrientation(VxVector *Dir, VxVector *Up, VxVector *Right, CK3dEntity *Ref) {
    // IDA @ 0x1000758C
    // - If Ref: computes (Ref^-1 * thisWorld), then returns normalized axes
    // - Else: returns normalized axes from GetWorldMatrix()
    if (Ref) {
        VxMatrix tmp;
        Vx3DMultiplyMatrix(tmp, Ref->GetInverseWorldMatrix(), m_WorldMatrix);

        if (Right) {
            Right->x = tmp[0][0];
            Right->y = tmp[0][1];
            Right->z = tmp[0][2];
            Right->Normalize();
        }
        if (Up) {
            Up->x = tmp[1][0];
            Up->y = tmp[1][1];
            Up->z = tmp[1][2];
            Up->Normalize();
        }
        if (Dir) {
            Dir->x = tmp[2][0];
            Dir->y = tmp[2][1];
            Dir->z = tmp[2][2];
            Dir->Normalize();
        }
        return;
    }

    const VxMatrix &world = GetWorldMatrix();
    if (Right) {
        *Right = VxVector(world[0][0], world[0][1], world[0][2]);
        Right->Normalize();
    }
    if (Up) {
        *Up = VxVector(world[1][0], world[1][1], world[1][2]);
        Up->Normalize();
    }
    if (Dir) {
        *Dir = VxVector(world[2][0], world[2][1], world[2][2]);
        Dir->Normalize();
    }
}

// =====================================================
// Virtual Method Implementations
// =====================================================

void RCK3dEntity::PreDelete() {
    // IDA @ 0x1000b8bf
    // Call base class first
    CKBeObject::PreDelete();

    // Iterate through children - detach those not being deleted
    int index = 0;
    int childCount = GetChildrenCount();
    while (childCount-- > 0) {
        RCK3dEntity *child = (RCK3dEntity *) GetChild(index);
        if (!child || (child->GetObjectFlags() & CK_OBJECT_TOBEDELETED)) {
            // Child is null or being deleted, skip
            ++index;
        } else {
            // Child is not being deleted, detach from this parent
            child->SetParent(nullptr, TRUE);
        }
    }

    // Handle parent relationship
    RCK3dEntity *parent = (RCK3dEntity *) GetParent();
    if (parent) {
        // If parent is not being deleted, detach from parent
        if (!(parent->GetObjectFlags() & CK_OBJECT_TOBEDELETED)) {
            SetParent(nullptr, TRUE);
        }
    } else {
        // No parent - remove scene graph node from its parent
        if (m_SceneGraphNode) {
            CKSceneGraphNode *graphParent = m_SceneGraphNode->m_Parent;
            if (graphParent) {
                graphParent->RemoveNode(m_SceneGraphNode);
            }
        }
    }
}

void RCK3dEntity::CheckPreDeletion() {
    // IDA @ 0x1000B994
    CKObject::CheckPreDeletion();

    m_Meshes.Check();
    if (m_CurrentMesh && m_CurrentMesh->IsToBeDeleted()) {
        m_CurrentMesh = nullptr;
    }

    if (m_ObjectAnimations) {
        m_ObjectAnimations->Check();
    }

    if (m_Skin) {
        const int boneCount = m_Skin->GetBoneCount();
        for (int i = 0; i < boneCount; ++i) {
            CKSkinBoneData *boneData = m_Skin->GetBoneData(i);
            if (!boneData)
                continue;
            CK3dEntity *bone = boneData->GetBone();
            if (bone && bone->IsToBeDeleted()) {
                boneData->SetBone(nullptr);
            }
        }
    }
}

int RCK3dEntity::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    // IDA @ 0x1000A016
    if (cid == CKCID_ANIMATION) {
        if (m_ObjectAnimations && m_ObjectAnimations->FindObject(o))
            return TRUE;
        return FALSE;
    }

    if (cid == CKCID_MESH || cid == CKCID_PATCHMESH) {
        return m_Meshes.FindObject(o);
    }

    return CKBeObject::IsObjectUsed(o, cid);
}

CKERROR RCK3dEntity::PrepareDependencies(CKDependenciesContext &context) {
    // IDA: PrepareDependencies at 0x1000b4d5
    CKERROR err = CKBeObject::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    // Get class-specific dependency flags for CKCID_3DENTITY (33)
    CKDWORD classDeps = context.GetClassDependencies(CKCID_3DENTITY);

    // Bit 0 (1): Process meshes
    if (classDeps & 1) {
        for (int i = 0; i < GetMeshCount(); ++i) {
            CKMesh *mesh = GetMesh(i);
            if (mesh)
                mesh->PrepareDependencies(context);
        }
    }

    // Bit 1 (2): Process children
    if (classDeps & 2) {
        for (int i = 0; i < GetChildrenCount(); ++i) {
            CK3dEntity *child = GetChild(i);
            if (child)
                child->PrepareDependencies(context);
        }
    }

    // Bit 2 (4): Process object animations
    if (classDeps & 4) {
        for (int i = 0; i < GetObjectAnimationCount(); ++i) {
            CKObjectAnimation *anim = GetObjectAnimation(i);
            if (anim)
                anim->PrepareDependencies(context);
        }
    }

    // Process skin bones if skin exists and (not mode REPLACE or meshes flag is set)
    // IDA: if ( this->m_Skin && (!sub_1000CE10(a2, 2) || (ClassDependencies & 1) != 0) )
    // sub_1000CE10 checks if mode == 2 (CK_DEPENDENCIES_REPLACE)
    if (m_Skin && (!context.IsInMode(CK_DEPENDENCIES_REPLACE) || (classDeps & 1))) {
        CKMesh *currentMesh = GetCurrentMesh();
        if (currentMesh)
            currentMesh->PrepareDependencies(context);

        for (int i = 0; i < m_Skin->GetBoneCount(); ++i) {
            CKSkinBoneData *boneData = m_Skin->GetBoneData(i);
            if (boneData) {
                CK3dEntity *boneEntity = boneData->GetBone();
                if (boneEntity)
                    boneEntity->PrepareDependencies(context);
            }
        }
    }

    return context.FinishPrepareDependencies(this, m_ClassID);
}

/*************************************************
Summary: RemapDependencies - Remaps object pointers after a copy/paste operation.
Purpose: Updates internal object references to point to newly created copies.

Implementation based on decompilation at 0x1000b6dd:
- Calls CKBeObject::RemapDependencies
- Gets ClassDependencies for mode 33
- If ClassID == 42 (CKCID_CHARACTER), adds flag 2 to ClassDependencies
- Remaps m_Place
- If flag 1: Remaps m_Meshs array and m_CurrentMesh
- If flag 2: Remaps parent via GetParent/SetParent
- If flag 4: Remaps m_ObjectAnimations
- Remaps skin bones if m_Skin exists
*************************************************/
CKERROR RCK3dEntity::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKBeObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    // Get class dependencies and check for CKCID_CHARACTER special case
    CKDWORD classDeps = context.GetClassDependencies(CKCID_3DENTITY);
    if (GetClassID() == CKCID_CHARACTER)
        classDeps |= 2;

    // Remap m_Place (IDA: sub_1000CDA0 => CKDependenciesContext::RemapID)
    CK_ID placeId = m_Place;
    placeId = context.RemapID(placeId);
    m_Place = placeId;

    // Remap meshes if flag 1 set
    if (classDeps & 1) {
        m_Meshes.Remap(context);
        RCKMesh *remappedMesh = (RCKMesh *) context.Remap(m_CurrentMesh);
        SetCurrentMesh(remappedMesh, FALSE);
    }

    // Remap parent if flag 2 set
    if (classDeps & 2) {
        CK3dEntity *parent = GetParent();
        CK3dEntity *remappedParent = (CK3dEntity *) context.Remap(parent);
        if (remappedParent)
            SetParent(remappedParent, TRUE);
    }

    // Remap object animations if flag 4 set
    if ((classDeps & 4) && m_ObjectAnimations) {
        m_ObjectAnimations->Remap(context);
    }

    // Remap skin bones
    if (m_Skin) {
        int boneCount = m_Skin->GetBoneCount();
        for (int i = 0; i < boneCount; ++i) {
            CKSkinBoneData *boneData = m_Skin->GetBoneData(i);
            if (boneData) {
                // Remap bone entity
                CK3dEntity *bone = boneData->GetBone();
                CK3dEntity *remappedBone = (CK3dEntity *) context.Remap(bone);
                boneData->SetBone(remappedBone);

                // Remap bone's parent if exists
                if (remappedBone) {
                    CK3dEntity *boneParent = remappedBone->GetParent();
                    CK3dEntity *remappedBoneParent = (CK3dEntity *) context.Remap(boneParent);
                    if (remappedBoneParent)
                        remappedBone->SetParent(remappedBoneParent, TRUE);
                }
            }
        }
    }

    return CK_OK;
}

/*************************************************
Summary: Copy - Copies entity data from source object.
Purpose: Deep copies all relevant entity data respecting dependency context.

Implementation based on decompilation at 0x1000ba8b:
- SetFlags from source
- Call CKBeObject::Copy
- SetFlags from source again
- Copy MoveableFlags masked with 0xFF3FBFEB
- Copy Parent via GetParent/SetParent
- Copy ZOrder via GetZOrder/SetZOrder
- Copy WorldMatrix
- Copy m_Meshs array
- Copy CurrentMesh via GetCurrentMesh/SetCurrentMesh
- If ClassDependencies flag 4: Copy ObjectAnimations
- If source has Skin: Create new Skin and copy
*************************************************/
CKERROR RCK3dEntity::Copy(CKObject &o, CKDependenciesContext &context) {
    RCK3dEntity &src = (RCK3dEntity &) o;

    // Set flags from source before CKBeObject::Copy
    SetFlags(src.GetFlags());

    CKERROR err = CKBeObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    // Get class dependencies
    CKDWORD classDeps = context.GetClassDependencies(CKCID_3DENTITY);

    // Set flags from source again
    SetFlags(src.GetFlags());

    // Copy moveable flags with mask 0xFF3FBFEB
    const CKDWORD mask = VX_MOVEABLE_UPTODATE | VX_MOVEABLE_USERBOX | VX_MOVEABLE_BOXVALID | VX_MOVEABLE_INVERSEWORLDMATVALID | VX_MOVEABLE_DONTUPDATEFROMPARENT;
    CKDWORD moveableFlags = src.GetMoveableFlags() & ~mask;
    SetMoveableFlags(moveableFlags);

    // Copy parent
    CK3dEntity *srcParent = src.GetParent();
    SetParent(srcParent, TRUE);

    // Copy ZOrder
    int zOrder = src.GetZOrder();
    SetZOrder(zOrder);

    // Copy world matrix
    SetWorldMatrix(src.m_WorldMatrix, TRUE);

    // Copy meshes array
    m_Meshes = src.m_Meshes;

    // Copy current mesh
    CKMesh *srcMesh = src.GetCurrentMesh();
    SetCurrentMesh(srcMesh);

    // Copy object animations if flag 4 set
    if (classDeps & 4) {
        int animCount = src.GetObjectAnimationCount();
        for (int i = 0; i < animCount; ++i) {
            CKObjectAnimation *anim = src.GetObjectAnimation(i);
            AddObjectAnimation(anim);
        }
    }

    // Copy skin if source has one
    if (src.m_Skin) {
        m_Skin = new RCKSkin(*src.m_Skin);
    }

    return CK_OK;
}

/*************************************************
Summary: AddToScene - Adds entity and optionally dependencies to a scene.
Purpose: Registers entity with scene, optionally adding meshes, animations, and children.

Implementation based on decompilation at 0x10009e01:
- Returns if scene is NULL
- Calls CKBeObject::AddToScene
- If dependencies flag set:
  - Adds all meshes to scene
  - Adds all object animations to scene
  - Adds children that are not already in scene
*************************************************/
void RCK3dEntity::AddToScene(CKScene *scene, CKBOOL dependencies) {
    if (!scene)
        return;

    CKBeObject::AddToScene(scene, dependencies);

    if (dependencies) {
        // Add meshes to scene
        for (CKObject **it = m_Meshes.Begin(); it != m_Meshes.End(); ++it) {
            CKBeObject *mesh = (CKBeObject *) *it;
            if (mesh)
                mesh->AddToScene(scene, dependencies);
        }

        // Add object animations to scene
        if (m_ObjectAnimations) {
            for (CKObject **it = m_ObjectAnimations->Begin(); it != m_ObjectAnimations->End(); ++it) {
                CKBeObject *anim = (CKBeObject *) *it;
                if (anim)
                    anim->AddToScene(scene, dependencies);
            }
        }

        // Add children that are not already in scene
        int childCount = GetChildrenCount();
        for (int i = 0; i < childCount; ++i) {
            RCK3dEntity *child = (RCK3dEntity *) GetChild(i);
            if (child && !child->IsInScene(scene))
                child->AddToScene(scene, dependencies);
        }
    }
}

/*************************************************
Summary: RemoveFromScene - Removes entity and optionally dependencies from a scene.
Purpose: Unregisters entity from scene, optionally removing meshes and animations.

Implementation based on decompilation at 0x10009f41:
- Returns if scene is NULL
- Calls CKBeObject::RemoveFromScene
- If dependencies flag set:
  - Removes all meshes from scene
  - Removes all object animations from scene
Note: Does NOT recursively remove children (different from AddToScene)
*************************************************/
void RCK3dEntity::RemoveFromScene(CKScene *scene, CKBOOL dependencies) {
    if (!scene)
        return;

    CKBeObject::RemoveFromScene(scene, dependencies);

    if (dependencies) {
        // Remove meshes from scene
        for (CKObject **it = m_Meshes.Begin(); it != m_Meshes.End(); ++it) {
            CKBeObject *mesh = (CKBeObject *) *it;
            if (mesh)
                mesh->RemoveFromScene(scene, dependencies);
        }

        // Remove object animations from scene
        if (m_ObjectAnimations) {
            for (CKObject **it = m_ObjectAnimations->Begin(); it != m_ObjectAnimations->End(); ++it) {
                CKBeObject *anim = (CKBeObject *) *it;
                if (anim)
                    anim->RemoveFromScene(scene, dependencies);
            }
        }
    }
}

/*************************************************
Summary: SetZOrder - Sets the rendering Z-order (priority).
Purpose: Uses scene graph node priority for render ordering.
*************************************************/
void RCK3dEntity::SetZOrder(int Z) {
    if (m_SceneGraphNode) {
        m_SceneGraphNode->SetPriority(Z, FALSE);
    }
}

/*************************************************
Summary: GetZOrder - Gets the rendering Z-order (priority).
Purpose: Returns scene graph node priority.
*************************************************/
int RCK3dEntity::GetZOrder() {
    if (m_SceneGraphNode) {
        return (int) m_SceneGraphNode->m_MaxPriority - 10000;
    }
    return 0;
}

/*************************************************
Summary: IsToBeRenderedLast - Check if entity should render after opaque objects.
Purpose: Used for transparency sorting in render pipeline.

Implementation based on decompilation at 0x10005910:
- If VX_MOVEABLE_ZBUFONLY (0x100000) is set, return FALSE (render early)
- If VX_MOVEABLE_RENDERLAST (0x10000) is set, return TRUE
- Else if mesh exists, return mesh->IsTransparent()
- Else return FALSE
*************************************************/
CKBOOL RCK3dEntity::IsToBeRenderedLast() {
    // IDA 0x10005910: if ( (this->m_MoveableFlags & 0x00100000) != 0 ) return 0;
    // => VX_MOVEABLE_RENDERFIRST
    if ((m_MoveableFlags & VX_MOVEABLE_RENDERFIRST) != 0)
        return FALSE;

    // IDA: if ( (this->m_MoveableFlags & 0x10000) != 0 ) return 1;
    if ((m_MoveableFlags & VX_MOVEABLE_RENDERLAST) != 0)
        return TRUE;

    // IDA: if ( this->m_CurrentMesh ) return this->m_CurrentMesh->IsTransparent(this->m_CurrentMesh);
    if (m_CurrentMesh) {
        return m_CurrentMesh->IsTransparent();
    }

    return FALSE;
}

/*************************************************
Summary: IsToBeRendered - Check if entity should be rendered.
Purpose: Combines visibility check with mesh/callback availability.

Implementation based on decompilation at 0x100058be:
- Must have VX_MOVEABLE_VISIBLE (0x2) set
- AND must have either:
  - A current mesh with VXMESH_VISIBLE (0x2) flag, OR
  - Callbacks registered
*************************************************/
CKBOOL RCK3dEntity::IsToBeRendered() {
    // IDA: return (this->m_MoveableFlags & 2) != 0
    //     && (this->m_CurrentMesh && (this->m_CurrentMesh->m_Flags & 2) != 0 || this->m_Callbacks);
    if ((m_MoveableFlags & VX_MOVEABLE_VISIBLE) == 0)
        return FALSE;

    if (m_CurrentMesh && (m_CurrentMesh->GetFlags() & VXMESH_VISIBLE) != 0)
        return TRUE;

    if (m_Callbacks)
        return TRUE;

    return FALSE;
}

/*************************************************
Summary: WorldMatrixChanged - Called when the world matrix has been directly modified.
Purpose: Updates internal state, local matrix, and propagates changes to children.

Implementation based on decompilation at 0x100062d6:
- Invalidates scene graph node's bounding box
- Clears VX_MOVEABLE_UPTODATE (0x4), VX_MOVEABLE_WORLDALIGNED (0x40000), and VX_MOVEABLE_INVERSEWORLDMATVALID (0x400000)
- If HASMOVED (0x20000) is set, clears 0x40000000; else sets HASMOVED and notifies RenderManager
- Updates local matrix: if parent exists, LocalMatrix = ParentInverse * WorldMatrix; else LocalMatrix = WorldMatrix
- If updateChildren: recalculate inverse and update children's local matrices to keep world position
- Else: propagate LocalMatrixChanged to children (so they move with parent)
*************************************************/
void RCK3dEntity::WorldMatrixChanged(int updateChildren, int keepScale) {
    // Invalidate scene graph node's bounding box
    if (m_SceneGraphNode) {
        m_SceneGraphNode->InvalidateBox(keepScale);
    }

    // IDA: this->m_MoveableFlags &= 0xFFBBFFFB (clears bits 0x00440004)
    // => clears VX_MOVEABLE_UPTODATE (0x4), VX_MOVEABLE_WORLDALIGNED (0x40000), VX_MOVEABLE_INVERSEWORLDMATVALID (0x400000)
    m_MoveableFlags &= ~(VX_MOVEABLE_UPTODATE | VX_MOVEABLE_WORLDALIGNED | VX_MOVEABLE_INVERSEWORLDMATVALID);

    // Check VX_MOVEABLE_HASMOVED (0x20000) flag
    if ((m_MoveableFlags & VX_MOVEABLE_HASMOVED) != 0) {
        // Already moved this frame - clear the "needs update" bit
        m_MoveableFlags &= ~VX_MOVEABLE_RESERVED2;
    } else {
        // First move this frame - set HASMOVED and notify render manager
        m_MoveableFlags |= VX_MOVEABLE_HASMOVED;
        RCKRenderManager *renderManager = (RCKRenderManager *) m_Context->GetRenderManager();
        if (renderManager) {
            renderManager->AddMovedEntity(this);
        }
    }

    // Update local matrix based on parent relationship
    if (m_Parent) {
        // LocalMatrix = ParentInverseWorld * WorldMatrix
        const VxMatrix &parentInverse = m_Parent->GetInverseWorldMatrix();
        Vx3DMultiplyMatrix(m_LocalMatrix, parentInverse, m_WorldMatrix);
    } else {
        // No parent - local matrix equals world matrix
        m_LocalMatrix = m_WorldMatrix;
    }

    // Handle children
    if (updateChildren) {
        // Keep children's world positions - recalculate their local matrices
        GetInverseWorldMatrix(); // Force recalculation of inverse matrix

        for (CKObject **it = m_Children.Begin(); it != m_Children.End(); ++it) {
            RCK3dEntity *child = (RCK3dEntity *) *it;
            if (child && !(child->m_MoveableFlags & VX_MOVEABLE_DONTUPDATEFROMPARENT)) {
                // Child's local = OurInverse * Child's World
                Vx3DMultiplyMatrix(child->m_LocalMatrix, m_InverseWorldMatrix, child->m_WorldMatrix);
            }
        }
    } else {
        // Children move with parent - propagate local matrix changes
        for (CKObject **it = m_Children.Begin(); it != m_Children.End(); ++it) {
            RCK3dEntity *child = (RCK3dEntity *) *it;
            if (child && !(child->m_MoveableFlags & VX_MOVEABLE_DONTUPDATEFROMPARENT)) {
                child->LocalMatrixChanged(FALSE, FALSE);
            }
        }
    }
}

/*************************************************
Summary: LocalMatrixChanged - Called when the local matrix has been modified.
Purpose: Recomputes world matrix from local * parent's world, propagates to children.

Implementation based on decompilation at 0x100064a4:
- Invalidates scene graph node's bounding box
- Clears VX_MOVEABLE_UPTODATE (0x4), VX_MOVEABLE_WORLDALIGNED (0x40000), VX_MOVEABLE_INVERSEWORLDMATVALID (0x400000)
- Handles HASMOVED flag and notifies render manager
- Updates world matrix: if parent exists, WorldMatrix = ParentWorld * LocalMatrix; else WorldMatrix = LocalMatrix
- Propagates changes to children similar to WorldMatrixChanged
*************************************************/
void RCK3dEntity::LocalMatrixChanged(int updateChildren, int keepScale) {
    // Invalidate scene graph node's bounding box
    if (m_SceneGraphNode) {
        m_SceneGraphNode->InvalidateBox(keepScale);
    }

    // IDA uses the same mask as WorldMatrixChanged: clears 0x00440004
    m_MoveableFlags &= ~(VX_MOVEABLE_UPTODATE | VX_MOVEABLE_WORLDALIGNED | VX_MOVEABLE_INVERSEWORLDMATVALID);

    // Check VX_MOVEABLE_HASMOVED (0x20000) flag
    if ((m_MoveableFlags & VX_MOVEABLE_HASMOVED) != 0) {
        m_MoveableFlags &= ~VX_MOVEABLE_RESERVED2;
    } else {
        m_MoveableFlags |= VX_MOVEABLE_HASMOVED;
        RCKRenderManager *renderManager = (RCKRenderManager *) m_Context->GetRenderManager();
        if (renderManager) {
            renderManager->AddMovedEntity(this);
        }
    }

    // Update world matrix based on parent relationship
    if (m_Parent) {
        // WorldMatrix = ParentWorld * LocalMatrix
        Vx3DMultiplyMatrix(m_WorldMatrix, m_Parent->m_WorldMatrix, m_LocalMatrix);
    } else {
        // No parent - world matrix equals local matrix
        m_WorldMatrix = m_LocalMatrix;
    }

    // Handle children
    if (updateChildren) {
        // Keep children's world positions
        GetInverseWorldMatrix();

        for (CKObject **it = m_Children.Begin(); it != m_Children.End(); ++it) {
            RCK3dEntity *child = (RCK3dEntity *) *it;
            if (child && !(child->m_MoveableFlags & VX_MOVEABLE_DONTUPDATEFROMPARENT)) {
                Vx3DMultiplyMatrix(child->m_LocalMatrix, m_InverseWorldMatrix, child->m_WorldMatrix);
            }
        }
    } else {
        // Children move with parent
        for (CKObject **it = m_Children.Begin(); it != m_Children.End(); ++it) {
            RCK3dEntity *child = (RCK3dEntity *) *it;
            if (child && !(child->m_MoveableFlags & VX_MOVEABLE_DONTUPDATEFROMPARENT)) {
                child->LocalMatrixChanged(FALSE, FALSE);
            }
        }
    }
}

/*************************************************
Summary: WorldPositionChanged - Called when only the position (translation) of world matrix changed.
Purpose: Optimized update when only position changes - updates inverse matrix incrementally.

Implementation based on decompilation at 0x1000666b:
- Invalidates scene graph bounding box
- Clears VX_MOVEABLE_UPTODATE (0x4) and VX_MOVEABLE_WORLDALIGNED (0x40000) (mask 0xFFFBFFFB)
- Handles HASMOVED flag and notifies render manager
- Updates local matrix from parent's inverse * world matrix
- If inverse matrix was valid (0x400000), updates only the translation part incrementally
- Propagates changes to children similar to WorldMatrixChanged
*************************************************/
void RCK3dEntity::WorldPositionChanged(int updateChildren, int keepScale) {
    // Invalidate scene graph node's bounding box
    if (m_SceneGraphNode) {
        m_SceneGraphNode->InvalidateBox(keepScale);
    }

    // IDA: this->m_MoveableFlags &= 0xFFFBFFFB; (clears bits 0x00040004)
    // => clears VX_MOVEABLE_UPTODATE (0x4) and VX_MOVEABLE_WORLDALIGNED (0x40000)
    // This is different from WorldMatrixChanged which clears 0xFFBBFFFB
    m_MoveableFlags &= ~(VX_MOVEABLE_UPTODATE | VX_MOVEABLE_WORLDALIGNED);

    // Check VX_MOVEABLE_HASMOVED (0x20000) flag
    if ((m_MoveableFlags & VX_MOVEABLE_HASMOVED) != 0) {
        // Already moved this frame - clear the "needs update" bit (0x40000000)
        m_MoveableFlags &= ~VX_MOVEABLE_RESERVED2;
    } else {
        // First move this frame - set HASMOVED and notify render manager
        m_MoveableFlags |= VX_MOVEABLE_HASMOVED;
        RCKRenderManager *renderManager = (RCKRenderManager *) m_Context->GetRenderManager();
        if (renderManager) {
            renderManager->AddMovedEntity(this);
        }
    }

    // Update local matrix based on parent relationship
    if (m_Parent) {
        // LocalMatrix = ParentInverseWorld * WorldMatrix
        const VxMatrix &parentInverse = m_Parent->GetInverseWorldMatrix();
        Vx3DMultiplyMatrix(m_LocalMatrix, parentInverse, m_WorldMatrix);
    } else {
        // No parent - local matrix equals world matrix
        m_LocalMatrix = m_WorldMatrix;
    }

    // IDA: If inverse matrix was valid, update only the translation part incrementally
    // This is an optimization - we can update just the translation of the inverse
    // instead of recomputing the entire inverse matrix
    if ((m_MoveableFlags & VX_MOVEABLE_INVERSEWORLDMATVALID) != 0) {
        // IDA: Compute new inverse translation: inv_T = -(R^T * T)
        // Where R is the rotation part and T is the translation
        // inv[3][0] = -(inv[0][0]*world[3][0] + inv[1][0]*world[3][1] + inv[2][0]*world[3][2])
        // inv[3][1] = -(inv[0][1]*world[3][0] + inv[1][1]*world[3][1] + inv[2][1]*world[3][2])
        // inv[3][2] = -(inv[0][2]*world[3][0] + inv[1][2]*world[3][1] + inv[2][2]*world[3][2])
        float tx = m_WorldMatrix[3][0];
        float ty = m_WorldMatrix[3][1];
        float tz = m_WorldMatrix[3][2];

        m_InverseWorldMatrix[3][0] = -(m_InverseWorldMatrix[0][0] * tx +
                                       m_InverseWorldMatrix[1][0] * ty +
                                       m_InverseWorldMatrix[2][0] * tz);
        m_InverseWorldMatrix[3][1] = -(m_InverseWorldMatrix[0][1] * tx +
                                       m_InverseWorldMatrix[1][1] * ty +
                                       m_InverseWorldMatrix[2][1] * tz);
        m_InverseWorldMatrix[3][2] = -(m_InverseWorldMatrix[0][2] * tx +
                                       m_InverseWorldMatrix[1][2] * ty +
                                       m_InverseWorldMatrix[2][2] * tz);
    }

    // Handle children
    if (updateChildren) {
        // Keep children's world positions - recalculate their local matrices
        GetInverseWorldMatrix(); // Force recalculation of inverse matrix if needed

        for (CKObject **it = m_Children.Begin(); it != m_Children.End(); ++it) {
            RCK3dEntity *child = (RCK3dEntity *) *it;
            if (child && (child->m_MoveableFlags & VX_MOVEABLE_DONTUPDATEFROMPARENT) == 0) {
                // Child's local = OurInverse * Child's World
                Vx3DMultiplyMatrix(child->m_LocalMatrix, m_InverseWorldMatrix, child->m_WorldMatrix);
            }
        }
    } else {
        // Children move with parent - propagate local matrix changes
        for (CKObject **it = m_Children.Begin(); it != m_Children.End(); ++it) {
            RCK3dEntity *child = (RCK3dEntity *) *it;
            if (child && (child->m_MoveableFlags & VX_MOVEABLE_DONTUPDATEFROMPARENT) == 0) {
                child->LocalMatrixChanged(FALSE, FALSE);
            }
        }
    }
}

// =====================================================
// Place Hierarchy Management
// =====================================================

/*************************************************
Summary: SaveLastFrameMatrix - Saves the current world matrix as the last frame matrix.
Purpose: Used for motion blur / temporal interpolation between frames.

Implementation based on decompilation at 0x100087BD:
- Allocates m_LastFrameMatrix if not already allocated
- Copies current world matrix to last frame matrix
*************************************************/
void RCK3dEntity::SaveLastFrameMatrix() {
    if (!m_LastFrameMatrix) {
        m_LastFrameMatrix = new VxMatrix();
    }
    memcpy(m_LastFrameMatrix, &m_WorldMatrix, sizeof(VxMatrix));
}

void RCK3dEntity::UpdatePlace(CK_ID placeId) {
    // IDA: UpdatePlace at 0x100092C3
    // Set this entity's place and recursively update all children
    m_Place = placeId;

    for (CKObject **it = m_Children.Begin(); it != m_Children.End(); ++it) {
        RCK3dEntity *child = (RCK3dEntity *) *it;
        if (child) {
            child->UpdatePlace(placeId);
        }
    }
}

// =====================================================
// Stubs for remaining methods
// =====================================================

void RCK3dEntity::SetQuaternion(const VxQuaternion *Quat, CK3dEntity *Ref, CKBOOL KeepChildren, CKBOOL KeepScale) {
    // IDA @ 0x10008962
    // - Optionally preserves LOCAL scale (KeepScale): GetScale(Local=TRUE) then SetScale(KeepChildren=0, Local=TRUE)
    // - Preserves current WORLD translation (row 3)
    // - If Ref: world = RefWorld * QuatMatrix, then restore translation
    // - Else: world = QuatMatrix, then restore translation
    // - Calls WorldMatrixChanged(KeepChildren, 1)

    VxVector savedLocalScale;
    if (KeepScale) {
        GetScale(&savedLocalScale, TRUE);
    }

    const VxVector savedWorldPos(m_WorldMatrix[3][0], m_WorldMatrix[3][1], m_WorldMatrix[3][2]);

    if (Ref) {
        VxMatrix rot;
        Quat->ToMatrix(rot);
        Vx3DMultiplyMatrix(m_WorldMatrix, Ref->GetWorldMatrix(), rot);
    } else {
        Quat->ToMatrix(m_WorldMatrix);
    }

    // Restore translation
    m_WorldMatrix[3][0] = savedWorldPos.x;
    m_WorldMatrix[3][1] = savedWorldPos.y;
    m_WorldMatrix[3][2] = savedWorldPos.z;

    WorldMatrixChanged(KeepChildren, 1);

    if (KeepScale) {
        SetScale(&savedLocalScale, 0, TRUE);
    }
}

void RCK3dEntity::GetQuaternion(VxQuaternion *Quat, CK3dEntity *Ref) {
    // IDA @ 0x10008A7A
    // - If Ref: quat from (Ref^-1 * thisWorld)
    // - Else: quat from thisWorld
    // - Uses FromMatrix(MatIsUnit=FALSE, RestoreMat=TRUE)
    if (Ref) {
        VxMatrix tmp;
        Vx3DMultiplyMatrix(tmp, Ref->GetInverseWorldMatrix(), m_WorldMatrix);
        Quat->FromMatrix(tmp, FALSE, TRUE);
        return;
    }

    Quat->FromMatrix(m_WorldMatrix, FALSE, TRUE);
}

void RCK3dEntity::SetScale3f(float X, float Y, float Z, CKBOOL KeepChildren, CKBOOL Local) {
    VxVector scale(X, Y, Z);
    SetScale(&scale, KeepChildren, Local);
}

void RCK3dEntity::SetScale(const VxVector *Scale, CKBOOL KeepChildren, CKBOOL Local) {
    // IDA @ 0x1000791A
    // - Selects axis vectors from local/world matrix based on Local
    // - For each axis: axis = Normalize(axis) * scaleComponent
    // - If any Scale component is 0.0, clamps it to 1e-6
    // - Calls LocalMatrixChanged/WorldMatrixChanged
    constexpr float kEps = 0.000001f;
    const float sx = (Scale->x == 0.0f) ? kEps : Scale->x;
    const float sy = (Scale->y == 0.0f) ? kEps : Scale->y;
    const float sz = (Scale->z == 0.0f) ? kEps : Scale->z;

    VxMatrix &mat = Local ? m_LocalMatrix : m_WorldMatrix;

    {
        VxVector axis0 = mat[0];
        axis0.Normalize();
        axis0 = axis0 * sx;
        mat[0][0] = axis0.x;
        mat[0][1] = axis0.y;
        mat[0][2] = axis0.z;
    }
    {
        VxVector axis1 = mat[1];
        axis1.Normalize();
        axis1 = axis1 * sy;
        mat[1][0] = axis1.x;
        mat[1][1] = axis1.y;
        mat[1][2] = axis1.z;
    }
    {
        VxVector axis2 = mat[2];
        axis2.Normalize();
        axis2 = axis2 * sz;
        mat[2][0] = axis2.x;
        mat[2][1] = axis2.y;
        mat[2][2] = axis2.z;
    }

    if (Local) {
        LocalMatrixChanged(KeepChildren, TRUE);
    } else {
        WorldMatrixChanged(KeepChildren, TRUE);
    }
}

void RCK3dEntity::GetScale(VxVector *Scale, CKBOOL Local) {
    if (!Scale)
        return;

    const VxMatrix &mat = Local ? m_LocalMatrix : m_WorldMatrix;
    VxVector axis0 = mat[0];
    VxVector axis1 = mat[1];
    VxVector axis2 = mat[2];
    Scale->x = axis0.Magnitude();
    Scale->y = axis1.Magnitude();
    Scale->z = axis2.Magnitude();
}

static void RCK3dEntity_ConstructMatrix(VxMatrix &dst, const VxVector *pos, const VxVector *scale, const VxQuaternion *quat) {
    quat->ToMatrix(dst);

    for (int i = 0; i < 3; ++i) {
        dst[0][i] *= scale->x;
        dst[1][i] *= scale->y;
        dst[2][i] *= scale->z;
    }

    dst[3][0] = pos->x;
    dst[3][1] = pos->y;
    dst[3][2] = pos->z;
}

static void RCK3dEntity_ConstructMatrixEx(VxMatrix &dst, const VxVector *pos, const VxVector *scale, const VxQuaternion *quat, const VxQuaternion *shear, float sign) {
    (void) sign;

    VxMatrix shearMat;
    shear->ToMatrix(shearMat);

    const float u00 = shearMat[0][0];
    const float u01 = shearMat[0][1];
    const float u02 = shearMat[0][2];
    const float u10 = shearMat[1][0];
    const float u11 = shearMat[1][1];
    const float u12 = shearMat[1][2];
    const float u20 = shearMat[2][0];
    const float u21 = shearMat[2][1];
    const float u22 = shearMat[2][2];

    const float sx = scale->x;
    const float sy = scale->y;
    const float sz = scale->z;

    VxMatrix s;
    s.SetIdentity();

    s[0][0] = u00 * u00 * sx + u10 * u10 * sy + u20 * u20 * sz;
    s[1][0] = u01 * u00 * sx + u11 * u10 * sy + u21 * u20 * sz;
    s[2][0] = u02 * u00 * sx + u12 * u10 * sy + u22 * u20 * sz;

    s[1][1] = u01 * u01 * sx + u11 * u11 * sy + u21 * u21 * sz;
    s[2][1] = u02 * u01 * sx + u12 * u11 * sy + u22 * u21 * sz;

    s[2][2] = u02 * u02 * sx + u12 * u12 * sy + u22 * u22 * sz;

    s[0][1] = s[1][0];
    s[0][2] = s[2][0];
    s[1][2] = s[2][1];

    s[0][3] = s[1][3] = s[2][3] = 0.0f;
    s[3][0] = s[3][1] = s[3][2] = 0.0f;
    s[3][3] = 1.0f;

    VxMatrix quatMat;
    quat->ToMatrix(quatMat);

    Vx3DMultiplyMatrix(dst, quatMat, s);

    dst[3][0] = pos->x;
    dst[3][1] = pos->y;
    dst[3][2] = pos->z;
}

CKBOOL RCK3dEntity::ConstructWorldMatrix(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat) {
    RCK3dEntity_ConstructMatrix(m_WorldMatrix, Pos, Scale, Quat);
    WorldMatrixChanged(FALSE, TRUE);
    return TRUE;
}

CKBOOL RCK3dEntity::ConstructWorldMatrixEx(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat, const VxQuaternion *Shear, float Sign) {
    RCK3dEntity_ConstructMatrixEx(m_WorldMatrix, Pos, Scale, Quat, Shear, Sign);
    WorldMatrixChanged(FALSE, TRUE);
    return TRUE;
}

CKBOOL RCK3dEntity::ConstructLocalMatrix(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat) {
    RCK3dEntity_ConstructMatrix(m_LocalMatrix, Pos, Scale, Quat);
    LocalMatrixChanged(FALSE, TRUE);
    return TRUE;
}

CKBOOL RCK3dEntity::ConstructLocalMatrixEx(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat, const VxQuaternion *Shear, float Sign) {
    RCK3dEntity_ConstructMatrixEx(m_LocalMatrix, Pos, Scale, Quat, Shear, Sign);
    LocalMatrixChanged(FALSE, TRUE);
    return TRUE;
}

CKBOOL RCK3dEntity::Render(CKRenderContext *Dev, CKDWORD Flags) {
    RCKRenderContext *dev = (RCKRenderContext *) Dev;

    // IDA: local VxTimeProfiler constructed at entry (used by Dev debug mode).
    VxTimeProfiler profiler;

    // Must have mesh or callbacks
    if (!m_CurrentMesh && !m_Callbacks)
        return FALSE;

    CKBOOL isPM = FALSE;

    // IDA: sub_1000D2F0 (0x1000D2F0)
    // Flush pending Sprite3D batches (only when needed).
    dev->FlushSprite3DBatchesIfNeeded();

    // Check if extents are up to date (meaning we've already been verified visible)
    if ((m_MoveableFlags & VX_MOVEABLE_EXTENTSUPTODATE) != 0) {
        // Extents are valid - we know we're visible
        if ((Flags & CK_RENDER_CLEARVIEWPORT) == 0) {
            dev->SetWorldTransformationMatrix(m_WorldMatrix);
        }
    } else {
        // Need to check frustum visibility
        if (!IsInViewFrustrum(Dev, Flags)) {
            // IDA: In debug render mode, append a "Not drawn" line and periodically flip.
            if ((dev->m_Flags & 1) != 0) {
                dev->m_CurrentObjectDesc << m_Name;
                if (IsToBeRenderedLast())
                    dev->m_CurrentObjectDesc << " (as transparent Object)";
                dev->m_CurrentObjectDesc << " : " << "Not drawn";
                dev->m_CurrentObjectDesc << profiler.Current() << " ms \n";
                if (--dev->m_FpsInterval <= 0)
                    dev->BackToFront(CK_RENDER_USECURRENTSETTINGS);
            }

            // Object is outside frustum - skip rendering but still return success
            return TRUE;
        }
    }

    // Handle indirect matrix (mirrored objects)
    CKDWORD savedInverseWinding = 0;
    if ((m_MoveableFlags & VX_MOVEABLE_INDIRECTMATRIX) != 0) {
        dev->m_RasterizerContext->GetRenderState(VXRENDERSTATE_INVERSEWINDING, &savedInverseWinding);
        dev->m_RasterizerContext->SetRenderState(VXRENDERSTATE_INVERSEWINDING, savedInverseWinding == 0 ? 1 : 0);
    }

    // Handle skin update for non-PM meshes
    if (m_Skin && m_CurrentMesh) {
        if (m_CurrentMesh->IsPM()) {
            isPM = TRUE;
        } else {
            // Update skin before callbacks
            dev->m_SkinTimeProfiler.Reset();
            UpdateSkin();
            dev->m_Stats.SkinTime += dev->m_SkinTimeProfiler.Current();
        }
    }

    // Execute callbacks if present
    if (m_Callbacks) {
        // Execute pre-render callbacks
        if (m_Callbacks->m_PreCallBacks.Size() > 0) {
            dev->m_ObjectsCallbacksTimeProfiler.Reset();
            dev->m_RasterizerContext->SetVertexShader(0);

            for (int i = 0; i < m_Callbacks->m_PreCallBacks.Size(); i++) {
                VxCallBack &cb = m_Callbacks->m_PreCallBacks[i];
                ((CK_RENDEROBJECT_CALLBACK) cb.callback)(Dev, (CK3dEntity *) this, cb.argument);
            }

            dev->m_Stats.ObjectsCallbacksTime += dev->m_ObjectsCallbacksTimeProfiler.Current();
        }

        // Update skin for PM meshes after pre-callbacks
        if (isPM) {
            dev->m_SkinTimeProfiler.Reset();
            UpdateSkin();
            dev->m_Stats.SkinTime += dev->m_SkinTimeProfiler.Current();
        }

        // Execute render callback (replaces default rendering) or default render
        if (m_Callbacks->m_Callback) {
            ((CK_RENDEROBJECT_CALLBACK) m_Callbacks->m_Callback->callback)(Dev, (CK3dEntity *) this, m_Callbacks->m_Callback->argument);
        } else if (m_CurrentMesh && (m_CurrentMesh->GetFlags() & VXMESH_VISIBLE) != 0) {
            dev->m_Current3dEntity = this;
            m_CurrentMesh->Render(Dev, (CK3dEntity *) this);
            dev->m_Current3dEntity = nullptr;
        }

        // Execute post-render callbacks
        if (m_Callbacks->m_PostCallBacks.Size() > 0) {
            dev->m_ObjectsCallbacksTimeProfiler.Reset();
            dev->m_RasterizerContext->SetVertexShader(0);

            for (int i = 0; i < m_Callbacks->m_PostCallBacks.Size(); i++) {
                VxCallBack &cb = m_Callbacks->m_PostCallBacks[i];
                ((CK_RENDEROBJECT_CALLBACK) cb.callback)(Dev, (CK3dEntity *) this, cb.argument);
            }

            dev->m_Stats.ObjectsCallbacksTime += dev->m_ObjectsCallbacksTimeProfiler.Current();
        }
    } else {
        // No callbacks - just render the mesh
        if (m_CurrentMesh && (m_CurrentMesh->GetFlags() & VXMESH_VISIBLE) != 0) {
            dev->m_Current3dEntity = this;
            m_CurrentMesh->Render(Dev, (CK3dEntity *) this);
            dev->m_Current3dEntity = nullptr;
        }
    }

    // Restore inverse winding if changed
    if ((m_MoveableFlags & VX_MOVEABLE_INDIRECTMATRIX) != 0) {
        // IDA toggles again based on the current state (not restoring the saved value).
        CKDWORD currentWinding = 0;
        dev->m_RasterizerContext->GetRenderState(VXRENDERSTATE_INVERSEWINDING, &currentWinding);
        dev->m_RasterizerContext->SetRenderState(VXRENDERSTATE_INVERSEWINDING, currentWinding == 0 ? 1 : 0);
    }

    // Update render extents if requested
    if ((Flags & CKRENDER_UPDATEEXTENTS) != 0) {
        dev->AddExtents2D(m_RenderExtents, this);
    }

    // IDA: In debug render mode, append a per-object line and periodically flip.
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

int RCK3dEntity::RayIntersection(const VxVector *Pos1, const VxVector *Pos2, VxIntersectionDesc *Desc, CK3dEntity *Ref, CK_RAYINTERSECTION iOptions) {
    // IDA: 0x10008DC5
    // Notes:
    // - Desc is allowed to be nullptr
    // - If Ref != this, the segment endpoints are transformed to this local using InverseTransform(..., Ref)
    // - If Desc != nullptr, Desc->Object is set to Ref before the ray test, then restored to this on success
    // - If hit, Desc->Distance is scaled by the input segment length (in the referential given by Ref)
    if (!m_CurrentMesh || !Pos1 || !Pos2)
        return 0;

    RCKMesh *mesh = m_CurrentMesh;
    if (!mesh)
        return 0;

    VxVector localP1 = *Pos1;
    VxVector localP2 = *Pos2;
    if (Ref != (CK3dEntity *) this) {
        InverseTransform(&localP1, Pos1, Ref);
        InverseTransform(&localP2, Pos2, Ref);
    }

    VxVector dir = localP2 - localP1;

    if (Desc)
        Desc->Object = (CKRenderObject *) Ref;

    const int hit = g_RayIntersection ? g_RayIntersection(mesh, localP1, dir, Desc, iOptions, m_WorldMatrix) : 0;
    if (hit && Desc) {
        Desc->Object = (CKRenderObject *) this;
        Desc->Distance *= (*Pos2 - *Pos1).Magnitude();
    }

    return hit;
}

void RCK3dEntity::GetRenderExtents(VxRect &rect) const {
    rect = m_RenderExtents;
}

const VxMatrix &RCK3dEntity::GetLastFrameMatrix() const {
    // IDA at 0x10008850:
    // - if m_LastFrameMatrix != nullptr, return it
    // - else return &m_WorldMatrix
    if (m_LastFrameMatrix)
        return *m_LastFrameMatrix;
    return m_WorldMatrix;
}

/*************************************************
Summary: SetLocalMatrix - Sets the local transformation matrix.
Purpose: Updates local matrix and propagates change via LocalMatrixChanged.

Implementation based on decompilation at 0x10008874:
- Copies the matrix to m_LocalMatrix
- Calls LocalMatrixChanged(KeepChildren, 1)
*************************************************/
void RCK3dEntity::SetLocalMatrix(const VxMatrix &Mat, CKBOOL KeepChildren) {
    m_LocalMatrix = Mat;
    LocalMatrixChanged(KeepChildren, TRUE);
}

const VxMatrix &RCK3dEntity::GetLocalMatrix() const {
    return m_LocalMatrix;
}

/*************************************************
Summary: SetWorldMatrix - Sets the world transformation matrix.
Purpose: Updates world matrix and propagates change via WorldMatrixChanged.

Implementation based on decompilation at 0x100088c4:
- Copies the matrix to m_WorldMatrix via qmemcpy
- Calls WorldMatrixChanged(KeepChildren, 1)
*************************************************/
void RCK3dEntity::SetWorldMatrix(const VxMatrix &Mat, CKBOOL KeepChildren) {
    m_WorldMatrix = Mat;
    WorldMatrixChanged(KeepChildren, TRUE);
}

const VxMatrix &RCK3dEntity::GetWorldMatrix() const {
    return m_WorldMatrix;
}

/*************************************************
Summary: GetInverseWorldMatrix - Returns the inverse of the world matrix.
Purpose: Uses lazy evaluation with caching via INVERSEWORLDMATVALID flag.

Implementation based on decompilation at 0x10008909:
- If VX_MOVEABLE_INVERSEWORLDMATVALID (0x400000) is NOT set:
  - Set the flag
  - Compute inverse: m_InverseWorldMatrix = inverse(m_WorldMatrix)
- Return reference to cached inverse matrix
*************************************************/
const VxMatrix &RCK3dEntity::GetInverseWorldMatrix() const {
    // Lazy evaluation - compute inverse only when needed
    // IDA: if ( (this->m_MoveableFlags & 0x400000) == 0 )
    if (!(m_MoveableFlags & VX_MOVEABLE_INVERSEWORLDMATVALID)) {
        // Mark as valid and compute
        // Note: const_cast required as this is logically const (caching)
        RCK3dEntity *mutableThis = const_cast<RCK3dEntity *>(this);
        mutableThis->m_MoveableFlags |= VX_MOVEABLE_INVERSEWORLDMATVALID;
        Vx3DInverseMatrix(mutableThis->m_InverseWorldMatrix, m_WorldMatrix);
    }
    return m_InverseWorldMatrix;
}

void RCK3dEntity::Transform(VxVector *Dest, const VxVector *Src, CK3dEntity *Ref) const {
    if (Ref) {
        VxMatrix tmp;
        Vx3DMultiplyMatrix(tmp, Ref->GetInverseWorldMatrix(), m_WorldMatrix);
        Vx3DMultiplyMatrixVector(Dest, tmp, Src);
    } else {
        Vx3DMultiplyMatrixVector(Dest, m_WorldMatrix, Src);
    }
}

void RCK3dEntity::InverseTransform(VxVector *Dest, const VxVector *Src, CK3dEntity *Ref) const {
    const VxMatrix &invWorld = GetInverseWorldMatrix();
    if (Ref) {
        VxMatrix tmp;
        Vx3DMultiplyMatrix(tmp, invWorld, Ref->GetWorldMatrix());
        Vx3DMultiplyMatrixVector(Dest, tmp, Src);
    } else {
        Vx3DMultiplyMatrixVector(Dest, invWorld, Src);
    }
}

void RCK3dEntity::TransformVector(VxVector *Dest, const VxVector *Src, CK3dEntity *Ref) const {
    if (Ref) {
        VxMatrix tmp;
        Vx3DMultiplyMatrix(tmp, Ref->GetInverseWorldMatrix(), m_WorldMatrix);
        Vx3DRotateVector(Dest, tmp, Src);
    } else {
        Vx3DRotateVector(Dest, m_WorldMatrix, Src);
    }
}

void RCK3dEntity::InverseTransformVector(VxVector *Dest, const VxVector *Src, CK3dEntity *Ref) const {
    const VxMatrix &invWorld = GetInverseWorldMatrix();
    if (Ref) {
        VxMatrix tmp;
        Vx3DMultiplyMatrix(tmp, invWorld, Ref->GetWorldMatrix());
        Vx3DRotateVector(Dest, tmp, Src);
    } else {
        Vx3DRotateVector(Dest, invWorld, Src);
    }
}

void RCK3dEntity::TransformMany(VxVector *Dest, const VxVector *Src, int count, CK3dEntity *Ref) const {
    if (Ref) {
        VxMatrix tmp;
        Vx3DMultiplyMatrix(tmp, Ref->GetInverseWorldMatrix(), m_WorldMatrix);
        Vx3DMultiplyMatrixVectorMany(Dest, tmp, Src, count, sizeof(VxVector));
    } else {
        Vx3DMultiplyMatrixVectorMany(Dest, m_WorldMatrix, Src, count, sizeof(VxVector));
    }
}

void RCK3dEntity::InverseTransformMany(VxVector *Dest, const VxVector *Src, int count, CK3dEntity *Ref) const {
    const VxMatrix &invWorld = GetInverseWorldMatrix();
    if (Ref) {
        VxMatrix tmp;
        Vx3DMultiplyMatrix(tmp, invWorld, Ref->GetWorldMatrix());
        Vx3DMultiplyMatrixVectorMany(Dest, tmp, Src, count, sizeof(VxVector));
    } else {
        Vx3DMultiplyMatrixVectorMany(Dest, invWorld, Src, count, sizeof(VxVector));
    }
}

void RCK3dEntity::ChangeReferential(CK3dEntity *Ref) {
    // Preserve current world transform, then reparent and recompute local matrix
    VxMatrix world = GetWorldMatrix();

    if (Ref) {
        VxMatrix invRef;
        Vx3DInverseMatrix(invRef, Ref->GetWorldMatrix());

        VxMatrix newLocal;
        Vx3DMultiplyMatrix(newLocal, invRef, world);
        SetParent(Ref, TRUE);
        SetLocalMatrix(newLocal, TRUE);
    } else {
        // Removing referential: keep world as local and clear parent
        SetParent(nullptr, TRUE);
        SetLocalMatrix(world, TRUE);
    }
}

CKPlace *RCK3dEntity::GetReferencePlace() const {
    return (CKPlace *) m_Context->GetObject(m_Place);
}

void RCK3dEntity::AddObjectAnimation(CKObjectAnimation *anim) {
    if (anim) {
        // Allocate animation array if needed (IDA: operator new(0xCu))
        if (!m_ObjectAnimations) {
            m_ObjectAnimations = new XObjectPointerArray();
        }
        m_ObjectAnimations->PushBack(anim);
    }
}

void RCK3dEntity::RemoveObjectAnimation(CKObjectAnimation *anim) {
    if (!m_ObjectAnimations) return;
    for (CKObject **it = m_ObjectAnimations->Begin(); it != m_ObjectAnimations->End(); ++it) {
        if (*it == anim) {
            m_ObjectAnimations->Remove(it);
            return;
        }
    }
}

CKObjectAnimation *RCK3dEntity::GetObjectAnimation(int index) const {
    if (!m_ObjectAnimations || index < 0 || index >= m_ObjectAnimations->Size())
        return nullptr;
    return (CKObjectAnimation *) (*m_ObjectAnimations)[index];
}

int RCK3dEntity::GetObjectAnimationCount() const {
    return m_ObjectAnimations ? m_ObjectAnimations->Size() : 0;
}

CKSkin *RCK3dEntity::CreateSkin() {
    DestroySkin();

    m_Skin = new RCKSkin();

    if (m_SceneGraphNode)
        m_SceneGraphNode->InvalidateBox(TRUE);
    return m_Skin;
}

CKBOOL RCK3dEntity::DestroySkin() {
    if (m_Skin) {
        delete m_Skin;
        m_Skin = nullptr;
    }

    return TRUE;
}

CKBOOL RCK3dEntity::UpdateSkin() {
    if (!m_Skin)
        return FALSE;

    // IDA: ?UpdateSkin@RCK3dEntity@@UAEHXZ @ 0x1000529E
    // - Updates m_Skin->m_InverseWorldMatrix depending on CK_3DENTITY_ENABLESKINOFFSET
    // - Forces mesh dynamic hint
    // - Uses modifier vertices and optionally normals
    // - Calls ModifierVertexMove with (RebuildNormals, RebuildFaceNormals) swapped depending on CalcPoints/CalcPointsEx

    if ((m_3dEntityFlags & CK_3DENTITY_ENABLESKINOFFSET) != 0) {
        Vx3DInverseMatrix(m_Skin->m_InverseWorldMatrix, m_Skin->m_ObjectInitMatrix);
    } else {
        m_Skin->m_InverseWorldMatrix = GetInverseWorldMatrix();
    }

    RCKMesh *mesh = static_cast<RCKMesh *>(GetCurrentMesh());
    if (!mesh)
        return FALSE;

    // Ensure dynamic hint is set
    mesh->SetFlags(mesh->GetFlags() | VXMESH_HINTDYNAMIC);

    int modifierVertexCount = mesh->GetModifierVertexCount();

    if (mesh->IsPM() && mesh->IsPMGeoMorphEnabled()) {
        modifierVertexCount = m_Skin->GetVertexCount();
        // Clear CK_OBJECT_UPTODATE on the mesh (PM geomorph path)
        mesh->ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
    }

    if (m_Skin->GetVertexCount() < modifierVertexCount)
        return FALSE;

    if (m_SceneGraphNode)
        m_SceneGraphNode->InvalidateBox(TRUE);

    CKDWORD vStride = 0;
    CKBYTE *vertexPtr = mesh->GetModifierVertices(&vStride);

    if (m_Skin->GetNormalCount() != 0) {
        CKDWORD nStride = 0;
        CKBYTE *normalPtr = static_cast<CKBYTE *>(mesh->GetNormalsPtr(&nStride));
        if (m_Skin->CalcPointsEx(modifierVertexCount, vertexPtr, vStride, normalPtr, nStride)) {
            mesh->ModifierVertexMove(FALSE, TRUE);
            return TRUE;
        }
    } else {
        if (m_Skin->CalcPoints(modifierVertexCount, vertexPtr, vStride)) {
            mesh->ModifierVertexMove(TRUE, FALSE);
            return TRUE;
        }
    }

    return FALSE;
}

CKSkin *RCK3dEntity::GetSkin() const {
    return m_Skin;
}

/*************************************************
Summary: UpdateBox - Updates the entity's bounding boxes.
Purpose: Computes local and world bounding boxes from mesh or user-defined box.

Implementation based on decompilation at 0x10006113:
- If VX_MOVEABLE_USERBOX (0x10) is set: use user-defined box, transform to world
- If mesh exists and not up-to-date: get box from mesh, merge with skin if present, transform to world
- If no mesh: use world position as point box
- Sets VX_MOVEABLE_BOXVALID (0x4000) when box is up-to-date
*************************************************/
void RCK3dEntity::UpdateBox(CKBOOL World) {
    (void) World; // Original implementation ignores this parameter (IDA: 0x10006113)

    if ((m_MoveableFlags & VX_MOVEABLE_USERBOX) != 0) {
        m_MoveableFlags |= VX_MOVEABLE_BOXVALID;
        m_WorldBoundingBox.TransformFrom(m_LocalBoundingBox, m_WorldMatrix);
    } else if (m_CurrentMesh) {
        if (!(m_MoveableFlags & VX_MOVEABLE_UPTODATE) || !(m_CurrentMesh->GetFlags() & VXMESH_BOUNDINGUPTODATE)) {
            m_MoveableFlags |= VX_MOVEABLE_BOXVALID;

            // Base local bbox from mesh
            m_LocalBoundingBox = m_CurrentMesh->GetLocalBox();

            // DLL merges a skin bbox computed from bone positions (sub_100407B0)
            if (m_Skin) {
                VxBbox skinBox;
                m_Skin->CalcBonesBBox(m_Context, (CK3dEntity *) this, &skinBox);
                m_LocalBoundingBox.Merge(skinBox);
            }

            m_WorldBoundingBox.TransformFrom(m_LocalBoundingBox, m_WorldMatrix);
            m_MoveableFlags |= VX_MOVEABLE_UPTODATE;
        }
    } else {
        // No mesh: world bbox is a point at world translation; local bbox is zeroed; BOXVALID is cleared.
        const VxVector pos = m_WorldMatrix[3];
        m_WorldBoundingBox.Max = pos;
        m_WorldBoundingBox.Min = pos;
        memset(&m_LocalBoundingBox, 0, sizeof(VxBbox));
        m_MoveableFlags &= ~VX_MOVEABLE_BOXVALID;
    }
}

const VxBbox &RCK3dEntity::GetBoundingBox(CKBOOL Local) {
    // Ensure bounding boxes are up-to-date
    UpdateBox(!Local);
    return Local ? m_LocalBoundingBox : m_WorldBoundingBox;
}

CKBOOL RCK3dEntity::SetBoundingBox(const VxBbox *BBox, CKBOOL Local) {
    // IDA: ?SetBoundingBox@RCK3dEntity@@UAEHPBUVxBbox@@H@Z @ 0x10009181
    if (BBox) {
        if (Local) {
            m_LocalBoundingBox = *BBox;
            m_WorldBoundingBox.TransformFrom(m_LocalBoundingBox, m_WorldMatrix);
        } else {
            m_WorldBoundingBox = *BBox;
            const VxMatrix &invWorld = GetInverseWorldMatrix();
            m_LocalBoundingBox.TransformFrom(m_WorldBoundingBox, invWorld);
        }

        if (m_SceneGraphNode)
            m_SceneGraphNode->InvalidateBox(TRUE);

        // Mark as user-defined and valid.
        m_MoveableFlags |= (VX_MOVEABLE_USERBOX | VX_MOVEABLE_BOXVALID);
    } else {
        // Clear USERBOX and UPTODATE (matches 0xFFFFFFEB mask in IDA)
        m_MoveableFlags &= ~(VX_MOVEABLE_USERBOX | VX_MOVEABLE_UPTODATE);
        if (m_SceneGraphNode)
            m_SceneGraphNode->InvalidateBox(TRUE);
    }

    return TRUE;
}

const VxBbox &RCK3dEntity::GetHierarchicalBox(CKBOOL Local) {
    // IDA: ?GetHierarchicalBox@RCK3dEntity@@UAEABUVxBbox@@H@Z @ 0x1000926B
    // World hierarchical box is stored on the scene-graph node.
    if (!m_SceneGraphNode) {
        return GetBoundingBox(Local);
    }

    m_SceneGraphNode->ComputeHierarchicalBox();
    const VxBbox &worldBox = m_SceneGraphNode->m_Bbox;

    if (!Local)
        return worldBox;

    const VxMatrix &invWorld = GetInverseWorldMatrix();
    m_HierarchicalBox.TransformFrom(worldBox, invWorld);
    return m_HierarchicalBox;
}

CKBOOL RCK3dEntity::GetBaryCenter(VxVector *Pos) {
    // IDA: ?GetBaryCenter@RCK3dEntity@@UAEHPAUVxVector@@@Z @ 0x100090C1
    if (!Pos)
        return FALSE;

    if (m_CurrentMesh) {
        VxVector localBary;
        m_CurrentMesh->GetBaryCenter(&localBary);
        Transform(Pos, &localBary, nullptr);
        return TRUE;
    }

    // No mesh: return world translation and return FALSE.
    Pos->x = m_WorldMatrix[3][0];
    Pos->y = m_WorldMatrix[3][1];
    Pos->z = m_WorldMatrix[3][2];
    return FALSE;
}

float RCK3dEntity::GetRadius() {
    // IDA: ?GetRadius@RCK3dEntity@@UAEMXZ @ 0x10008EE2
    if (m_CurrentMesh) {
        VxVector row0 = m_WorldMatrix[0];
        VxVector row1 = m_WorldMatrix[1];
        VxVector row2 = m_WorldMatrix[2];
        const float sx = row0.Magnitude();
        const float sy = row1.Magnitude();
        const float sz = row2.Magnitude();
        float maxScale = sx;
        if (sy > maxScale)
            maxScale = sy;
        if (sz > maxScale)
            maxScale = sz;

        return m_CurrentMesh->GetRadius() * maxScale;
    }

    UpdateBox(TRUE);
    if (!(m_MoveableFlags & VX_MOVEABLE_BOXVALID))
        return 0.0f;

    const float dx = m_WorldBoundingBox.Max.x - m_WorldBoundingBox.Min.x;
    const float dy = m_WorldBoundingBox.Max.y - m_WorldBoundingBox.Min.y;
    const float dz = m_WorldBoundingBox.Max.z - m_WorldBoundingBox.Min.z;

    float maxExtent = dx;
    if (dy > maxExtent)
        maxExtent = dy;
    if (dz > maxExtent)
        maxExtent = dz;

    return maxExtent * 0.5f;
}

/*************************************************
Summary: Show - Shows or hides the entity.
Purpose: Controls entity visibility and hierarchical hiding.

Implementation based on decompilation at 0x10005494:
- Calls CKObject::Show() first
- Clears hierarchy-affecting flags
- Sets VX_MOVEABLE_VISIBLE based on show parameter
- Sets VX_MOVEABLE_HIERARCHICALHIDE if hiding children
- Notifies scene graph of flag changes
*************************************************/
void RCK3dEntity::Show(CK_OBJECT_SHOWOPTION show) {
    // IDA: 0x10005494

    // Call base class Show
    CKObject::Show(show);

    m_MoveableFlags &= ~VX_MOVEABLE_HIERARCHICALHIDE;

    // Set or clear VX_MOVEABLE_VISIBLE (0x2) based on show parameter
    // show & 1 means "show" (CKSHOW = 1)
    // show & 2 means "hide children too" (CKHIDE_HIERARCHY = 2)
    if ((show & CKSHOW) != 0) {
        m_MoveableFlags |= VX_MOVEABLE_VISIBLE;
    } else {
        m_MoveableFlags &= ~VX_MOVEABLE_VISIBLE;

        // If hiding children too, set hierarchical hide flag
        if ((show & CKHIERARCHICALHIDE) != 0) {
            m_MoveableFlags |= VX_MOVEABLE_HIERARCHICALHIDE;
        }
    }

    // Notify scene graph node about flag changes
    if (m_SceneGraphNode) {
        m_SceneGraphNode->EntityFlagsChanged(TRUE);
    }
}

/*************************************************
Summary: IsHiddenByParent - Checks if any parent has hierarchical hide flag.
Purpose: Used for visibility determination in scene graph.

Implementation based on decompilation at 0x10005556:
- Walks up the parent chain
- Returns TRUE if any parent has VX_MOVEABLE_HIERARCHICALHIDE (0x10000000) set
- Returns FALSE if no parent has the flag
Note: Original uses m_MoveableFlags, NOT CK_OBJECT_VISIBLE
*************************************************/
CKBOOL RCK3dEntity::IsHiddenByParent() {
    for (RCK3dEntity *parent = m_Parent; parent != nullptr; parent = parent->m_Parent) {
        if ((parent->m_MoveableFlags & VX_MOVEABLE_HIERARCHICALHIDE) != 0) {
            return TRUE;
        }
    }
    return FALSE;
}

/*************************************************
Summary: IsVisible - Returns TRUE if the entity should be rendered.
Purpose: Combines object visibility with parent hierarchy check.

Implementation based on decompilation at 0x1000552b:
- Returns CKObject::IsVisible() && !IsHiddenByParent()
*************************************************/
CKBOOL RCK3dEntity::IsVisible() {
    return CKObject::IsVisible() && !IsHiddenByParent();
}

CKBOOL RCK3dEntity::IsInViewFrustrum(CKRenderContext *rc, CKDWORD flags) {
    // IDA: 0x10005D69
    if (!rc)
        return FALSE;

    RCKRenderContext *dev = (RCKRenderContext *) rc;
    if (!dev->m_RasterizerContext)
        return TRUE;

    const CKBOOL updateExtents = (CKBYTE) flags != 0;
    if (updateExtents) {
        // IDA resets extents to an inverted huge rect
        m_RenderExtents = VxRect(100000000.0f, 100000000.0f, -100000000.0f, -100000000.0f);
    }

    // Mark extents as up-to-date for this test pass
    ModifyMoveableFlags(VX_MOVEABLE_EXTENTSUPTODATE, 0);

    CKDWORD vis = 1;

    // USERBOX + BOXVALID path uses the local bbox and current world matrix
    if ((m_MoveableFlags & (VX_MOVEABLE_USERBOX | VX_MOVEABLE_BOXVALID)) == (VX_MOVEABLE_USERBOX | VX_MOVEABLE_BOXVALID)) {
        if (!(flags & CK_RENDER_CLEARVIEWPORT))
            dev->SetWorldTransformationMatrix(m_WorldMatrix);

        VxRect *ext = updateExtents ? &m_RenderExtents : nullptr;
        vis = dev->m_RasterizerContext->ComputeBoxVisibility(m_LocalBoundingBox, FALSE, ext);
    } else if (m_CurrentMesh) {
        // If the mesh has no vertices, consider it not visible
        if (m_CurrentMesh->GetVertexCount() <= 0)
            return FALSE;

        // If mesh geometry changed, clear UPTODATE and invalidate scene-graph boxes
        if (!(m_CurrentMesh->GetFlags() & 0x1)) {
            m_MoveableFlags &= ~VX_MOVEABLE_UPTODATE;
            if (m_SceneGraphNode)
                m_SceneGraphNode->InvalidateBox(TRUE);
        }

        const VxBbox &meshLocalBox = m_CurrentMesh->GetLocalBox();

        if (!(flags & CK_RENDER_CLEARVIEWPORT))
            dev->SetWorldTransformationMatrix(m_WorldMatrix);

        VxRect *ext = updateExtents ? &m_RenderExtents : nullptr;

        if (m_Skin) {
            // With skin, IDA uses the entity cached local bbox
            vis = dev->m_RasterizerContext->ComputeBoxVisibility(m_LocalBoundingBox, FALSE, ext);
        } else {
            vis = dev->m_RasterizerContext->ComputeBoxVisibility(meshLocalBox, FALSE, ext);
        }
    } else {
        // No mesh: transform the origin and treat as a 1x1 extent
        VxVector4 in(0.0f, 0.0f, 0.0f, 1.0f);
        VxVector4 outH;
        VxVector4 outS;
        unsigned int clip = 0;
        VxTransformData td;
        memset(&td, 0, sizeof(td));
        td.InVertices = &in;
        td.InStride = 16;
        td.OutVertices = &outH;
        td.OutStride = 16;
        td.ScreenVertices = &outS;
        td.ScreenStride = 16;
        td.ClipFlags = &clip;

        dev->m_RasterizerContext->SetTransformMatrix(VXMATRIX_WORLD, m_WorldMatrix);
        dev->m_RasterizerContext->TransformVertices(1, &td);

        if (updateExtents) {
            const float x = outS.x;
            const float y = outS.y;
            const float w = 1.0f;
            const float h = 1.0f;
            if (x < m_RenderExtents.left)
                m_RenderExtents.left = x;
            if (y < m_RenderExtents.top)
                m_RenderExtents.top = y;
            if (x + w > m_RenderExtents.right)
                m_RenderExtents.right = x + w;
            if (y + h > m_RenderExtents.bottom)
                m_RenderExtents.bottom = y + h;
        }

        vis = td.m_Offscreen ? 0 : 2;
    }

    if (vis) {
        if (vis == 2 && m_SceneGraphNode)
            m_SceneGraphNode->SetAsInsideFrustum();
        return TRUE;
    }

    // IDA: sub_1000D2B0(&node->m_Entity) => node->m_Flags |= 2
    if (m_SceneGraphNode)
        m_SceneGraphNode->SetAsOutsideFrustum();
    return FALSE;
}

CKBOOL RCK3dEntity::IsInViewFrustrumHierarchic(CKRenderContext *rc) {
    // IDA: 0x10006095
    if (!rc)
        return FALSE;

    RCKRenderContext *dev = (RCKRenderContext *) rc;
    if (!dev->m_RasterizerContext || !m_SceneGraphNode)
        return TRUE;

    m_SceneGraphNode->SetAsPotentiallyVisible();
    m_SceneGraphNode->ComputeHierarchicalBox();

    const CKDWORD vis = dev->m_RasterizerContext->ComputeBoxVisibility(m_SceneGraphNode->m_Bbox, TRUE, nullptr);
    if (vis) {
        if (vis == 2)
            m_SceneGraphNode->SetAsInsideFrustum();
        return TRUE;
    }

    m_SceneGraphNode->SetAsOutsideFrustum();
    return FALSE;
}

//=============================================================================
// Static Class Methods (for class registration)
//=============================================================================

CKSTRING RCK3dEntity::GetClassName() {
    return (CKSTRING) "3D Entity";
}

int RCK3dEntity::GetDependenciesCount(int mode) {
    // Based on IDA decompilation at 0x1000b339
    switch (mode) {
    case 1: return 3; // Copy mode
    case 2: return 3; // Delete mode
    case 3: return 0; // Replace mode
    case 4: return 3; // Save mode
    default: return 0;
    }
}

CKSTRING RCK3dEntity::GetDependencies(int i, int mode) {
    // Based on IDA decompilation at 0x1000b38b
    switch (i) {
    case 0: return (CKSTRING) "Meshes";
    case 1: return (CKSTRING) "Children";
    case 2: return (CKSTRING) "Animation";
    default: return nullptr;
    }
}

void RCK3dEntity::Register() {
    // Based on IDA decompilation at 0x1000b3c4
    CKCLASSNOTIFYFROMCID(RCK3dEntity, CKCID_OBJECTANIMATION);
    CKCLASSNOTIFYFROMCID(RCK3dEntity, CKCID_MESH);
    CKCLASSNOTIFYFROMCID(RCK3dEntity, CKCID_3DENTITY);

    // Register associated parameter GUID: {5B8A05D5, 31EA28D4}
    CKPARAMETERFROMCLASS(RCK3dEntity, CKPGUID_3DENTITY);

    // Register default dependencies for different modes
    CKCLASSDEFAULTCOPYDEPENDENCIES(RCK3dEntity, 6);
    CKCLASSDEFAULTDELETEDEPENDENCIES(RCK3dEntity, CK_DEPENDENCIES_SAVE);
    CKCLASSDEFAULTSAVEDEPENDENCIES(RCK3dEntity, 7);
}

CK3dEntity *RCK3dEntity::CreateInstance(CKContext *Context) {
    // Based on IDA decompilation at 0x1000b45f
    // Object size is 0x1A8 (424 bytes)
    RCK3dEntity *ent = new RCK3dEntity(Context, nullptr);
    return reinterpret_cast<CK3dEntity *>(ent);
}
