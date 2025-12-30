#include "RCKCharacter.h"

#include "VxMath.h"
#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "CKScene.h"
#include "CKDependencies.h"
#include "RCK3dEntity.h"
#include "RCKBodyPart.h"
#include "RCKAnimation.h"
#include "RCKKeyedAnimation.h"
#include "RCKObjectAnimation.h"

//=============================================================================
// Helper function corresponding to sub_10048148 in IDA
// Iterates animation entities and sets exclusive animation on body parts
// When exclusiveAnim is non-null, sets the body part's exclusive animation to it
// When exclusiveAnim is null, clears the body part's exclusive animation
//=============================================================================

static void NotifyBodyPartsInAnimation(CKAnimation *anim, CKAnimation *exclusiveAnim) {
    // This only works for CKKeyedAnimation which has GetAnimation/GetAnimationCount methods
    if (!anim || !CKIsChildClassOf(anim, CKCID_KEYEDANIMATION)) {
        return;
    }

    CKKeyedAnimation *keyedAnim = (CKKeyedAnimation *)anim;
    int count = keyedAnim->GetAnimationCount();

    for (int i = 0; i < count; ++i) {
        CKObjectAnimation *objAnim = keyedAnim->GetAnimation(i);
        if (!objAnim)
            continue;

        // Get the entity from the object animation (vtable+0xF4)
        CK3dEntity *entity = objAnim->Get3dEntity();
        if (!entity)
            continue;

        // Check if it's a body part (CKCID_BODYPART = 0x2A = 42)
        if (CKIsChildClassOf(entity, CKCID_BODYPART)) {
            // Call SetExclusiveAnimation (vtable+0x1D8) on the body part
            CKBodyPart *bp = (CKBodyPart *)entity;
            bp->SetExclusiveAnimation(exclusiveAnim);
        }
    }
}

RCKCharacter::RCKCharacter(CKContext *Context, CKSTRING name)
    : RCK3dEntity(Context, name),
      m_BodyParts(),
      m_Animations(),
      m_SecondaryAnimations(nullptr),
      m_SecondaryAnimationsCount(0),
      m_SecondaryAnimationsAllocated(0),
      m_RootBodyPart(nullptr),
      m_ActiveAnimation(nullptr),
      m_AnimDest(nullptr),
      m_Warper(nullptr),
      m_FrameDest(0.0f),
      field_1D4(0),
      m_FloorRef(nullptr),
      m_AnimationLevelOfDetail(1.0f),
      m_FrameSrc(0.0f),
      m_AnimSrc(nullptr),
      m_TransitionMode(0) {
    // Based on IDA decompilation at 0x1000F9B0
    // Create the warper animation (internal transition animation)
    CKBOOL isDynamic = m_Context->IsInDynamicCreationMode();
    m_Warper = (RCKKeyedAnimation *)m_Context->CreateObject(
        CKCID_KEYEDANIMATION,
        nullptr,
        CK_OBJECTCREATION_SameDynamic,
        nullptr);

    if (m_Warper) {
        // Set the character reference on the warper
        ((RCKAnimation *)m_Warper)->m_Character = this;

        // Set flag 0x08 (CKANIMATION_INTERNAL) on the warper
        CKDWORD flags = ((RCKAnimation *)m_Warper)->m_Flags;
        flags |= CKANIMATION_ALLOWTURN;
        ((RCKAnimation *)m_Warper)->m_Flags = flags;

        // Modify object flags: set 0x23 (CK_OBJECT_NOTTOBELISTEDANDSAVED)
        ((CKObject *)m_Warper)->ModifyObjectFlags(CK_OBJECT_NOTTOBELISTEDANDSAVED, 0);
    }
}

RCKCharacter::~RCKCharacter() {
    // Based on IDA decompilation at 0x1000FB5B
    // Delete secondary animations array
    if (m_SecondaryAnimations) {
        delete[] m_SecondaryAnimations;
        m_SecondaryAnimations = nullptr;
    }

    // Clear animations array
    m_Animations.Clear();

    // Clear body parts array
    m_BodyParts.Clear();
}

//=============================================================================
// Serialization Methods
//=============================================================================

void RCKCharacter::PreSave(CKFile *file, CKDWORD flags) {
    // Call base class PreSave first
    RCK3dEntity::PreSave(file, flags);

    // Save root body part
    if (m_RootBodyPart) {
        file->SaveObject((CKObject *) m_RootBodyPart);
    }

    // Save all body parts
    int bpCount = m_BodyParts.Size();
    for (int i = 0; i < bpCount; ++i) {
        CKObject *bp = m_BodyParts.GetObject(i);
        if (bp) {
            file->SaveObject(bp);
        }
    }

    // Save all animations
    int animCount = m_Animations.Size();
    for (int i = 0; i < animCount; ++i) {
        CKObject *anim = m_Animations.GetObject(i);
        if (anim) {
            file->SaveObject(anim);
        }
    }
}

CKStateChunk *RCKCharacter::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = RCK3dEntity::Save(file, flags);

    if (!file && !(flags & CK_STATESAVE_CHARACTERONLY)) {
        return baseChunk;
    }

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_CHARACTER, file);
    if (!chunk) {
        return baseChunk;
    }

    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    // Save body parts array with identifier 0x400000
    chunk->WriteIdentifier(CK_STATESAVE_CHARACTERBODYPARTS);
    m_BodyParts.Save(chunk);

    // Optionally save body parts as sub-chunks
    if (!file && (flags & CK_STATESAVE_CHARACTERSAVEPARTS) != 0) {
        chunk->WriteIdentifier(CK_STATESAVE_CHARACTERSAVEPARTS);
        int count = m_BodyParts.Size();
        chunk->StartSubChunkSequence(count);

        for (int i = 0; i < count; ++i) {
            CKObject *bp = m_BodyParts.GetObject(i);
            CKStateChunk *subChunk = nullptr;
            if (bp) {
                subChunk = bp->Save(nullptr, CK_STATESAVE_BODYPARTALL);
            }
            chunk->WriteSubChunkSequence(subChunk);
            if (subChunk) {
                DeleteCKStateChunk(subChunk);
            }
        }
    }

    // Save animations array with identifier 0xFFC00000
    chunk->WriteIdentifier(CK_STATESAVE_CHARACTERONLY);
    if (file) {
        m_Animations.Save(chunk);
    }

    // Save character object references in sequence
    chunk->StartObjectIDSequence(4);
    chunk->WriteObjectSequence((CKObject *) m_ActiveAnimation);
    chunk->WriteObjectSequence((CKObject *) m_AnimDest);
    chunk->WriteObjectSequence((CKObject *) m_RootBodyPart);
    chunk->WriteObjectSequence((CKObject *) m_FloorRef);

    if (GetClassID() == CKCID_CHARACTER) {
        chunk->CloseChunk();
    } else {
        chunk->UpdateDataSize();
    }

    return chunk;
}

CKERROR RCKCharacter::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) {
        return CKERR_INVALIDPARAMETER;
    }

    RCK3dEntity::Load(chunk, file);

    // Handle legacy format (data version < 5)
    if (chunk->GetDataVersion() < 5) {
        if (file) {
            if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERBODYPARTS)) {
                m_BodyParts.Clear();
                m_BodyParts.Load(m_Context, chunk);
            }

            if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERANIMATIONS)) {
                m_Animations.Clear();
                m_Animations.Load(m_Context, chunk);

                // Original reads these immediately after loading animations
                m_ActiveAnimation = (RCKKeyedAnimation *) chunk->ReadObject(m_Context);
                m_AnimDest = (RCKAnimation *) chunk->ReadObject(m_Context);
            }
        } else {
            // Save current and next active animations
            if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERSAVEANIMS)) {
                (void) chunk->ReadDword();
                m_ActiveAnimation = (RCKKeyedAnimation *) chunk->ReadObject(m_Context);
                m_AnimDest = (RCKAnimation *) chunk->ReadObject(m_Context);
            }

            // Optional body parts sub-chunk sequence
            if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERSAVEPARTS)) {
                const int count = (int) chunk->ReadDword();
                for (int i = 0; i < count; ++i) {
                    const CK_ID objID = chunk->ReadObjectID();
                    CKObject *bp = m_Context->GetObject(objID);
                    CKStateChunk *subChunk = chunk->ReadSubChunk();
                    if (bp && subChunk) {
                        bp->Load(subChunk, nullptr);
                    }
                    if (subChunk) {
                        DeleteCKStateChunk(subChunk);
                    }
                }
            }
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERROOT)) {
            m_RootBodyPart = (RCKBodyPart *) chunk->ReadObject(m_Context);
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERFLOORREF)) {
            m_FloorRef = (RCK3dEntity *) chunk->ReadObject(m_Context);
        }
    }
    // Current format (data version >= 5)
    else {
        if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERBODYPARTS)) {
            m_BodyParts.Clear();
            m_BodyParts.Load(m_Context, chunk);
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERSAVEPARTS)) {
            const int sequenceCount = chunk->StartReadSequence();
            if (sequenceCount == m_BodyParts.Size()) {
                for (int i = 0; i < sequenceCount; ++i) {
                    CKStateChunk *subChunk = chunk->ReadSubChunk();
                    CKObject *bp = m_BodyParts.GetObject(i);
                    if (bp && subChunk) {
                        bp->Load(subChunk, nullptr);
                    }
                    if (subChunk) {
                        DeleteCKStateChunk(subChunk);
                    }
                }
            }
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_CHARACTERONLY)) {
            if (file) {
                m_Animations.Clear();
                m_Animations.Load(m_Context, chunk);
            }

            chunk->StartReadSequence();
            m_ActiveAnimation = (RCKKeyedAnimation *) chunk->ReadObject(m_Context);
            m_AnimDest = (RCKAnimation *) chunk->ReadObject(m_Context);
            m_RootBodyPart = (RCKBodyPart *) chunk->ReadObject(m_Context);
            m_FloorRef = (RCK3dEntity *) chunk->ReadObject(m_Context);

            if (!m_RootBodyPart && GetChildrenCount()) {
                m_RootBodyPart = (RCKBodyPart *) GetChild(0);
            }
        }
    }

    return CK_OK;
}

//=============================================================================
// Static Class Methods
//=============================================================================

CK_CLASSID RCKCharacter::m_ClassID = CKCID_CHARACTER;

CKSTRING RCKCharacter::GetClassName() {
    return (CKSTRING) "Character";
}

int RCKCharacter::GetDependenciesCount(int mode) {
    if (mode == CK_DEPENDENCIES_COPY) {
        return 1;
    }
    return 0;
}

CKSTRING RCKCharacter::GetDependencies(int i, int mode) {
    if (i == 0 && mode == 1) {
        return (CKSTRING) "Share Animations";
    }
    return nullptr;
}

void RCKCharacter::Register() {
    CKCLASSNOTIFYFROMCID(RCKCharacter, CKCID_ANIMATION);
    CKCLASSNOTIFYFROMCID(RCKCharacter, CKCID_BODYPART);
    CKPARAMETERFROMCLASS(RCKCharacter, CKPGUID_CHARACTER);
    CKCLASSDEFAULTCOPYDEPENDENCIES(RCKCharacter, CK_DEPENDENCIES_COPY);
}

CKCharacter *RCKCharacter::CreateInstance(CKContext *Context) {
    return (CKCharacter *) new RCKCharacter(Context, nullptr);
}

//=============================================================================
// Virtual Method Overrides
//=============================================================================

CK_CLASSID RCKCharacter::GetClassID() {
    return m_ClassID;
}

void RCKCharacter::CheckPreDeletion() {
    // Based on decompilation at 0x1001242F
    RCK3dEntity::CheckPreDeletion();

    // Check animations array (removes deleted objects)
    m_Animations.Check();

    // Clear active animation if marked for deletion
    if (m_ActiveAnimation && ((CKObject *) m_ActiveAnimation)->IsToBeDeleted()) {
        m_ActiveAnimation = nullptr;
    }

    // Clear destination animation if marked for deletion
    if (m_AnimDest && ((CKObject *) m_AnimDest)->IsToBeDeleted()) {
        m_AnimDest = nullptr;
    }

    // Check body parts array (removes deleted objects)
    m_BodyParts.Check();

    // Clear root body part if marked for deletion
    if (m_RootBodyPart && ((CKObject *) m_RootBodyPart)->IsToBeDeleted()) {
        m_RootBodyPart = nullptr;
    }

    // Clear floor reference if marked for deletion
    if (m_FloorRef && ((CKObject *) m_FloorRef)->IsToBeDeleted()) {
        m_FloorRef = nullptr;
    }
}

int RCKCharacter::GetMemoryOccupation() {
    return RCK3dEntity::GetMemoryOccupation() + (sizeof(RCKCharacter) - sizeof(RCK3dEntity)) + m_BodyParts.GetMemoryOccupation(FALSE) + m_Animations.GetMemoryOccupation(FALSE);
}

int RCKCharacter::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    // Based on decompilation at 0x10012558
    if (cid == CKCID_ANIMATION) {
        // Check if animation is in our list
        if (m_Animations.FindObject(o)) {
            return 1;
        }
    } else if (cid == CKCID_BODYPART) {
        // Check if body part is in our list
        if (m_BodyParts.FindObject(o)) {
            return 1;
        }
    }

    return RCK3dEntity::IsObjectUsed(o, cid);
}

void RCKCharacter::Show(CK_OBJECT_SHOWOPTION show) {
    // Based on decompilation at 0x1000FCBE
    RCK3dEntity::Show(show);

    // Show all body parts with same option
    int count = m_BodyParts.Size();
    for (int i = 0; i < count; ++i) {
        CKObject *bp = m_BodyParts.GetObject(i);
        if (bp) {
            ((CKRenderObject *) bp)->Show(show);
        }
    }
}

float RCKCharacter::GetRadius() {
    // Based on decompilation at 0x1000FC23
    const VxBbox &box = GetHierarchicalBox(FALSE);

    float dx = box.Max.x - box.Min.x;
    float dy = box.Max.y - box.Min.y;
    float dz = box.Max.z - box.Min.z;

    float maxDim = (dy >= dx) ? dy : dx;
    if (dz >= maxDim) {
        maxDim = dz;
    }

    return maxDim * 0.5f;
}

const VxBbox &RCKCharacter::GetBoundingBox(CKBOOL Local) {
    // Based on decompilation at 0x10011CF9
    return GetHierarchicalBox(Local);
}

CKBOOL RCKCharacter::GetBaryCenter(VxVector *Pos) {
    // Based on decompilation at 0x10011D18
    if (Pos) {
        const VxBbox &box = GetHierarchicalBox(FALSE);
        VxVector sum = box.Min + box.Max;
        *Pos = sum * 0.5f;
    }
    return TRUE;
}

void RCKCharacter::AddToScene(CKScene *scene, CKBOOL dependencies) {
    // Based on IDA decompilation at 0x1001227F
    if (!scene) return;

    RCK3dEntity::AddToScene(scene, dependencies);

    if (dependencies) {
        // Add all body parts to scene
        for (CKObject **it = m_BodyParts.Begin(); it != m_BodyParts.End(); ++it) {
            CKObject *bp = *it;
            if (bp) {
                ((CKBeObject *)bp)->AddToScene(scene, dependencies);
            }
        }

        // Add all animations to scene
        for (CKObject **it = m_Animations.Begin(); it != m_Animations.End(); ++it) {
            CKObject *anim = *it;
            if (anim) {
                ((CKBeObject *)anim)->AddToScene(scene, dependencies);
            }
        }
    }
}

void RCKCharacter::RemoveFromScene(CKScene *scene, CKBOOL dependencies) {
    // Based on IDA decompilation at 0x10012357
    // NOTE: Original has a bug - it calls AddToScene on animations instead of RemoveFromScene
    // We preserve this behavior for binary compatibility
    if (!scene) return;

    RCK3dEntity::RemoveFromScene(scene, dependencies);

    if (dependencies) {
        // Remove all body parts from scene
        for (CKObject **it = m_BodyParts.Begin(); it != m_BodyParts.End(); ++it) {
            CKObject *bp = *it;
            if (bp) {
                ((CKBeObject *)bp)->RemoveFromScene(scene, dependencies);
            }
        }

        // BUG in original: This should call RemoveFromScene, but original calls AddToScene
        // Keeping original behavior for binary compatibility
        for (CKObject **it = m_Animations.Begin(); it != m_Animations.End(); ++it) {
            CKObject *anim = *it;
            if (anim) {
                ((CKBeObject *)anim)->AddToScene(scene, dependencies);
            }
        }
    }
}

//=============================================================================
// Dependency Methods
//=============================================================================

CKERROR RCKCharacter::PrepareDependencies(CKDependenciesContext &context) {
    // Based on decompilation at 0x10012E79
    CKERROR err = RCK3dEntity::PrepareDependencies(context);
    if (err != CK_OK) {
        return err;
    }

    CKDWORD charDeps = context.GetClassDependencies(CKCID_CHARACTER);
    CKDWORD animDeps = context.GetClassDependencies(CKCID_ANIMATION);

    // Prepare body parts
    m_BodyParts.Prepare(context);

    // Prepare floor reference
    if (m_FloorRef) {
        ((CKObject *) m_FloorRef)->PrepareDependencies(context);
    }

    // Prepare animations if needed (if not copying, or if anim deps bit 2 set, or char deps bit 0 set)
    if (!context.IsInMode(CK_DEPENDENCIES_COPY) || (animDeps & 4) != 0 || (charDeps & 1) != 0) {
        m_Animations.Prepare(context);
    }

    // Prepare secondary animations if deleting
    if (context.IsInMode(CK_DEPENDENCIES_DELETE)) {
        for (CKWORD i = 0; i < m_SecondaryAnimationsAllocated; ++i) {
            if (m_SecondaryAnimations && m_SecondaryAnimations[i].Animation) {
                ((CKObject *) m_SecondaryAnimations[i].Animation)->PrepareDependencies(context);
            }
        }

        if (m_Warper) {
            ((CKObject *) m_Warper)->PrepareDependencies(context);
        }
    }

    return context.FinishPrepareDependencies((CKObject *) this, m_ClassID);
}

CKERROR RCKCharacter::RemapDependencies(CKDependenciesContext &context) {
    // Based on decompilation at 0x10012FE6
    CKERROR err = RCK3dEntity::RemapDependencies(context);
    if (err != CK_OK) {
        return err;
    }

    // Get class dependencies (result ignored but call matches IDA)
    context.GetClassDependencies(CKCID_CHARACTER);

    // Remap body parts
    m_BodyParts.Remap(context);

    // Remap animations
    m_Animations.Remap(context);

    // Remap active animation (always, no conditional)
    m_ActiveAnimation = (RCKKeyedAnimation *) context.Remap((CKObject *) m_ActiveAnimation);

    // Remap floor reference
    m_FloorRef = (RCK3dEntity *) context.Remap((CKObject *) m_FloorRef);

    // Remap destination animation
    m_AnimDest = (RCKAnimation *) context.Remap((CKObject *) m_AnimDest);

    // Remap root body part
    m_RootBodyPart = (RCKBodyPart *) context.Remap((CKObject *) m_RootBodyPart);

    return CK_OK;
}

CKERROR RCKCharacter::Copy(CKObject &o, CKDependenciesContext &context) {
    // Based on decompilation at 0x100130AE
    CKERROR err = RCK3dEntity::Copy(o, context);
    if (err != CK_OK) {
        return err;
    }

    RCKCharacter *src = (RCKCharacter *) &o;

    // Check class dependencies for animation copy
    CKDWORD charDeps = context.GetClassDependencies(CKCID_CHARACTER);
    CKDWORD animDeps = context.GetClassDependencies(CKCID_ANIMATION);

    // Copy animations if deps allow (CKCID_ANIMATION=33, check bit 2; or CKCID_CHARACTER bit 0)
    if ((animDeps & 4) != 0 || (charDeps & 1) != 0) {
        // Copy animations array
        m_Animations = src->m_Animations;

        // Copy active animation reference
        m_ActiveAnimation = src->m_ActiveAnimation;

        // Copy destination animation reference
        m_AnimDest = src->m_AnimDest;

        // If dest anim was the source's warper, use our warper
        if (m_AnimDest == (RCKAnimation *)src->m_Warper) {
            m_AnimDest = (RCKAnimation *)m_Warper;
        }
    }

    // Copy body parts array
    m_BodyParts = src->m_BodyParts;

    // Copy root body part reference
    m_RootBodyPart = src->m_RootBodyPart;

    // Copy active animation reference (yes, again - matches IDA)
    m_ActiveAnimation = src->m_ActiveAnimation;

    // Copy frame destination
    m_FrameDest = src->m_FrameDest;

    // Copy field_1D4
    field_1D4 = src->field_1D4;

    // Copy floor reference
    m_FloorRef = src->m_FloorRef;

    return CK_OK;
}

//=============================================================================
// Body Part Methods
//=============================================================================

CKERROR RCKCharacter::AddBodyPart(CKBodyPart *part) {
    // Based on decompilation at 0x1000FE1E
    if (!part) {
        return CKERR_INVALIDPARAMETER;
    }

    // Check if already in list
    if (m_BodyParts.FindObject((CKObject *) part)) {
        return CKERR_ALREADYPRESENT;
    }

    // Add to our list
    m_BodyParts.PushBack((CKObject *) part);

    // Get previous character and remove from it
    RCKBodyPart *rckPart = (RCKBodyPart *) part;
    RCKCharacter *prevChar = rckPart->m_Character;
    if (prevChar && prevChar != this) {
        // Call RemoveBodyPart on the previous character
        prevChar->RemoveBodyPart(part);
    }

    // Set character reference on body part
    rckPart->m_Character = this;

    // If no parent, add as child and set as root
    if (!((CK3dEntity *) part)->GetParent()) {
        AddChild((CK3dEntity *) part, TRUE);
        m_RootBodyPart = (RCKBodyPart *) part;
    }

    // Find floor reference if not set
    if (!m_FloorRef) {
        FindFloorReference();
    }

    return CK_OK;
}

CKERROR RCKCharacter::RemoveBodyPart(CKBodyPart *part) {
    // Based on decompilation at 0x1000FF30
    if (!part) {
        return CKERR_INVALIDPARAMETER;
    }

    // Find and remove from list
    int removed = m_BodyParts.RemoveObject((CKObject *) part);
    if (removed < 0) {
        return CKERR_INVALIDPARAMETER;
    }

    // Clear character reference
    ((RCKBodyPart *) part)->m_Character = nullptr;

    // Update floor reference if it was this body part
    if ((CKBodyPart *) m_FloorRef == part) {
        m_FloorRef = nullptr;
        FindFloorReference();
    }

    return CK_OK;
}

CKBodyPart *RCKCharacter::GetRootBodyPart() {
    return (CKBodyPart *) m_RootBodyPart;
}

CKERROR RCKCharacter::SetRootBodyPart(CKBodyPart *part) {
    // Based on decompilation at 0x1000FF9A
    if ((RCKBodyPart *) part != m_RootBodyPart) {
        // Clear previous root's character reference
        if (m_RootBodyPart) {
            m_RootBodyPart->m_Character = nullptr;
        }

        m_RootBodyPart = (RCKBodyPart *) part;

        // Set new root's character reference
        if (m_RootBodyPart) {
            m_RootBodyPart->m_Character = this;
        }
    }
    return CK_OK;
}

CKBodyPart *RCKCharacter::GetBodyPart(int index) {
    if (index < 0 || index >= m_BodyParts.Size()) {
        return nullptr;
    }
    return (CKBodyPart *) m_BodyParts.GetObject(index);
}

int RCKCharacter::GetBodyPartCount() {
    return m_BodyParts.Size();
}

//=============================================================================
// Animation Methods
//=============================================================================

CKERROR RCKCharacter::AddAnimation(CKAnimation *anim) {
    // Based on decompilation at 0x10010776
    if (!anim) {
        return CKERR_INVALIDPARAMETER;
    }

    // Check if already in list
    if (m_Animations.FindObject((CKObject *) anim)) {
        return CK_OK; // Already present
    }

    // Add to our list
    m_Animations.PushBack((CKObject *) anim);

    // Get previous character and remove from it via RCKAnimation
    RCKAnimation *rckAnim = (RCKAnimation *) anim;
    RCKCharacter *prevChar = rckAnim->m_Character;
    if (prevChar && prevChar != this) {
        // Call RemoveAnimation on the previous character
        prevChar->RemoveAnimation(anim);
    }

    // Set character reference on animation
    rckAnim->m_Character = this;

    // Handle keyed animations
    if (CKIsChildClassOf((CKObject *) anim, CKCID_KEYEDANIMATION)) {
        if (m_RootBodyPart) {
            // If animation has no root entity, use our root body part
            if (!rckAnim->m_RootEntity) {
                rckAnim->m_RootEntity = (RCK3dEntity *) m_RootBodyPart;
            }
        } else {
            // If we have no root body part, use animation's root entity
            m_RootBodyPart = (RCKBodyPart *) rckAnim->m_RootEntity;
        }

        // Center animation at current frame
        float frame = ((CKAnimation *) anim)->GetFrame();
        ((CKKeyedAnimation *) anim)->CenterAnimation(frame);
    }

    return CK_OK;
}

CKERROR RCKCharacter::RemoveAnimation(CKAnimation *anim) {
    if (!anim) {
        return CKERR_INVALIDPARAMETER;
    }

    // Find and remove from list
    int removed = m_Animations.RemoveObject((CKObject *) anim);
    if (removed < 0) {
        return CKERR_INVALIDPARAMETER;
    }

    // Clear character reference
    ((RCKAnimation *) anim)->m_Character = nullptr;

    return CK_OK;
}

CKAnimation *RCKCharacter::GetAnimation(int index) {
    if (index < 0 || index >= m_Animations.Size()) {
        return nullptr;
    }
    return (CKAnimation *) m_Animations.GetObject(index);
}

int RCKCharacter::GetAnimationCount() {
    return m_Animations.Size();
}

CKAnimation *RCKCharacter::GetWarper() {
    return (CKAnimation *) m_Warper;
}

//=============================================================================
// Active Animation Methods
//=============================================================================

CKAnimation *RCKCharacter::GetActiveAnimation() {
    return (CKAnimation *) m_ActiveAnimation;
}

CKAnimation *RCKCharacter::GetNextActiveAnimation() {
    return (CKAnimation *) m_AnimDest;
}

CKERROR RCKCharacter::SetActiveAnimation(CKAnimation *anim) {
    m_ActiveAnimation = (RCKKeyedAnimation *) anim;
    return CK_OK;
}

CKERROR RCKCharacter::SetNextActiveAnimation(CKAnimation *anim, CKDWORD transitionmode, float warplength) {
    // Based on IDA decompilation at 0x10010982
    VxTimeProfiler profiler;

    RCKKeyedAnimation *destAnim = (RCKKeyedAnimation *)anim;

    if (!destAnim) {
        // Null animation passed - clear destination
        m_AnimDest = nullptr;
        if (m_ActiveAnimation) {
            // If active animation can be interrupted, clear it
            if ((((RCKAnimation *)m_ActiveAnimation)->m_Flags & CKANIMATION_CANBEBREAK) != 0) {
                m_ActiveAnimation = nullptr;
            }
        }
        return CKERR_INVALIDPARAMETER;
    }

    // Handle CK_TRANSITION_FROMANIMATION flag (0x200)
    if ((transitionmode & CK_TRANSITION_FROMANIMATION) != 0) {
        CKDWORD animTransMode = anim->GetTransitionMode();
        if (animTransMode) {
            transitionmode = animTransMode;
        }
    }

    // Check animation belongs to this character
    if (((RCKAnimation *)destAnim)->m_Character != this) {
        return CKERR_INVALIDPARAMETER;
    }

    if (!m_ActiveAnimation) {
        // No active animation - set destination and initialize
        m_AnimDest = (RCKAnimation *)destAnim;
        anim->SetStep(0.0f);
        if (anim->GetClassID() == CKCID_KEYEDANIMATION) {
            ((CKKeyedAnimation *)destAnim)->CenterAnimation(0.0f);
        }
        anim->SetStep(0.0f);
        m_FrameDest = 0.0f;
        return CK_OK;
    }

    // CK_TRANSITION_LOOPIFEQUAL (0x80) - If same animation, loop
    if ((transitionmode & CK_TRANSITION_LOOPIFEQUAL) != 0 && destAnim == m_ActiveAnimation) {
        // Check if step > 0.3 (animation is playing)
        if (((RCKAnimation *)destAnim)->m_Step > 0.3f) {
            m_AnimDest = (RCKAnimation *)destAnim;
            m_FrameDest = 0.0f;
        }
        return CK_OK;
    }

    RCKKeyedAnimation *activeAnim = m_ActiveAnimation;
    CKDWORD activeFlags = 0;
    if (activeAnim) {
        activeFlags = ((RCKAnimation *)activeAnim)->m_Flags;
    }

    // Check if we can break or animation step is 0
    if (((transitionmode & CK_TRANSITION_FROMNOW) == CK_TRANSITION_FROMNOW && (activeFlags & CKANIMATION_CANBEBREAK) != 0) ||
        ((RCKAnimation *)m_ActiveAnimation)->m_Step == 0.0f) {
        
        if (activeAnim == destAnim) {
            // Same animation - just set destination
            m_AnimDest = (RCKAnimation *)destAnim;
            m_FrameDest = 0.0f;
        } else {
            // Different animation - align and switch
            AlignCharacterWithRootPosition();
            m_ActiveAnimation = destAnim;
            if (anim->GetClassID() == CKCID_KEYEDANIMATION) {
                ((CKKeyedAnimation *)destAnim)->CenterAnimation(0.0f);
            }
            anim->SetStep(0.0f);
            m_AnimDest = nullptr;
            m_FrameDest = 0.0f;
        }
    }
    // CK_TRANSITION_TOSTART (0x08) - Wait until current ends
    else if ((transitionmode & CK_TRANSITION_TOSTART) == CK_TRANSITION_TOSTART) {
        m_AnimDest = (RCKAnimation *)destAnim;
        m_FrameDest = 0.0f;
    }
    // CK_TRANSITION_WARPMASK (0x132) - Warp transitions
    else if ((transitionmode & CK_TRANSITION_WARPMASK) != 0) {
        if (m_AnimDest != (RCKAnimation *)destAnim && destAnim) {
            if (anim->GetClassID() == CKCID_KEYEDANIMATION) {
                float frameDest = 0.0f;

                // Check if we're currently warping and dest can't be interrupted
                if (m_ActiveAnimation == m_Warper && m_AnimDest && !((CKAnimation *)m_AnimDest)->CanBeInterrupt()) {
                    return CK_OK;
                }

                if (m_ActiveAnimation) {
                    if (((CKObject *)m_ActiveAnimation)->GetClassID() == CKCID_KEYEDANIMATION) {
                        if (((CKAnimation *)m_ActiveAnimation)->CanBeInterrupt()) {
                            // Animation can be interrupted - create warp now
                            AlignCharacterWithRootPosition();
                            frameDest = m_Warper->CreateTransition(
                                (CKAnimation *)m_ActiveAnimation,
                                anim,
                                transitionmode,
                                warplength,
                                0.0f);
                            m_FrameSrc = ((CKAnimation *)m_ActiveAnimation)->GetFrame();
                            ((CKKeyedAnimation *)m_Warper)->CenterAnimation(0.0f);
                            ((CKAnimation *)m_Warper)->SetStep(0.0f);
                            m_AnimSrc = (RCKAnimation *)m_ActiveAnimation;
                            m_ActiveAnimation = m_Warper;
                            m_AnimDest = (RCKAnimation *)destAnim;
                            m_FrameDest = frameDest;
                            m_TransitionMode = transitionmode;
                        } else {
                            // Can't interrupt - check if at end
                            float curFrame = ((CKAnimation *)m_ActiveAnimation)->GetFrame();
                            float curLen = ((CKAnimation *)m_ActiveAnimation)->GetLength();
                            CKBOOL atEnd = (curLen == curFrame);

                            if (((CKAnimation *)m_ActiveAnimation)->IsLinkedToFrameRate()) {
                                float checkFrame = ((CKAnimation *)m_ActiveAnimation)->GetFrame();
                                float checkLen = ((CKAnimation *)m_ActiveAnimation)->GetLength();
                                atEnd = (checkLen - 0.3f <= checkFrame);
                            }

                            if (atEnd) {
                                AlignCharacterWithRootPosition();
                                if (((CKObject *)m_ActiveAnimation)->GetClassID() == CKCID_KEYEDANIMATION) {
                                    frameDest = m_Warper->CreateTransition(
                                        (CKAnimation *)m_ActiveAnimation,
                                        anim,
                                        transitionmode,
                                        warplength,
                                        0.0f);
                                }
                                ((CKKeyedAnimation *)m_Warper)->CenterAnimation(0.0f);
                                ((CKAnimation *)m_Warper)->SetStep(0.0f);
                                m_ActiveAnimation = m_Warper;
                                m_AnimDest = (RCKAnimation *)destAnim;
                                m_FrameDest = frameDest;
                            }
                        }
                    }
                } else {
                    // No active animation
                    m_ActiveAnimation = destAnim;
                    m_AnimDest = nullptr;
                    m_FrameDest = 0.0f;
                }
            } else {
                // Not keyed animation
                m_ActiveAnimation = destAnim;
                m_AnimDest = nullptr;
                m_FrameDest = 0.0f;
            }
        }
    } else {
        // Default - just set destination
        m_AnimDest = (RCKAnimation *)destAnim;
        m_FrameDest = 0.0f;
    }

    m_Context->AddProfileTime(CK_PROFILE_ANIMATIONTIME, profiler.Current());
    return CK_OK;
}

//=============================================================================
// Animation Processing Methods
//=============================================================================

void RCKCharacter::ProcessAnimation(float deltat) {
    // Based on decompilation at 0x10011038

    VxTimeProfiler profiler;

    RCKAnimation *destAnim = m_AnimDest;
    RCKKeyedAnimation *destKeyed = nullptr;
    if (destAnim && CKIsChildClassOf((CKObject *) destAnim, CKCID_KEYEDANIMATION)) {
        destKeyed = (RCKKeyedAnimation *) destAnim;
    }

    CKAnimation *activeAnim = (CKAnimation *) m_ActiveAnimation;
    RCKKeyedAnimation *activeKeyed = nullptr;
    if (m_ActiveAnimation && CKIsChildClassOf((CKObject *) m_ActiveAnimation, CKCID_KEYEDANIMATION)) {
        activeKeyed = m_ActiveAnimation;
    }

    bool doActivePostUpdate = true;
    float nextFrameForActive = 0.0f;

    if (activeAnim) {
        float curFrame = activeAnim->GetFrame();
        if (activeAnim->GetLength() != curFrame || activeAnim->CanBeInterrupt()) {
            const float activeLen = activeAnim->GetLength();
            float nextFrame = activeAnim->GetNextFrame(deltat);

            if (nextFrame < activeLen) {
                if (nextFrame < 0.0f) {
                    AlignCharacterWithRootPosition();
                    m_ActiveAnimation = (RCKKeyedAnimation *) m_AnimDest;
                    m_AnimDest = nullptr;

                    if (destAnim) {
                        const float destLen = destAnim->GetLength();
                        if (destLen != 0.0f) {
                            while (nextFrame < 0.0f) {
                                nextFrame += destLen;
                            }
                        }

                        const float destFrame = nextFrame + m_FrameDest;
                        if (destKeyed) {
                            destKeyed->CenterAnimation(destFrame);
                        }
                        destAnim->SetFrame(destFrame);
                    }

                    doActivePostUpdate = false;
                }
            } else {
                if (activeAnim->CanBeInterrupt() || m_AnimDest) {
                    AlignCharacterWithRootPosition();

                    if (activeKeyed && m_RootBodyPart) {
                        CKObjectAnimation *rootObjAnim = activeKeyed->GetAnimation((CK3dEntity *) m_RootBodyPart);
                        if (rootObjAnim) {
                            VxVector posEnd;
                            VxVector posCentered;
                            const float centeredFrame = activeAnim->GetFrame();
                            rootObjAnim->EvaluatePosition(activeLen, posEnd);
                            rootObjAnim->EvaluatePosition(centeredFrame, posCentered);
                            VxVector delta = posEnd - posCentered;
                            m_RootBodyPart->Translate(&delta, (CK3dEntity *) this, FALSE);

                            VxVector pos;
                            ((CK3dEntity *) m_RootBodyPart)->GetPosition(&pos, nullptr);
                            SetPosition(&pos, nullptr, TRUE);
                        }
                    }

                    m_ActiveAnimation = (RCKKeyedAnimation *) m_AnimDest;
                    m_AnimDest = nullptr;

                    if (destAnim) {
                        if (destKeyed) {
                            destKeyed->CenterAnimation(m_FrameDest);
                        }

                        if (activeLen != 0.0f) {
                            while (nextFrame >= activeLen) {
                                nextFrame -= activeLen;
                            }
                        }

                        const float destFrame = nextFrame + m_FrameDest;
                        destAnim->SetFrame(destFrame);

                        if (m_RootBodyPart) {
                            VxVector pos;
                            ((CK3dEntity *) m_RootBodyPart)->GetPosition(&pos, nullptr);
                            SetPosition(&pos, nullptr, TRUE);
                        }

                        if (destKeyed) {
                            destKeyed->CenterAnimation(destFrame);
                        }
                    }

                    doActivePostUpdate = false;
                } else {
                    nextFrame = activeLen;
                }
            }

            nextFrameForActive = nextFrame;
        } else {
            m_ActiveAnimation = (RCKKeyedAnimation *) m_AnimDest;
            m_AnimDest = nullptr;

            if (destAnim) {
                if (destKeyed) {
                    destKeyed->CenterAnimation(m_FrameDest);
                }

                float destLen = ((CKAnimation *) m_ActiveAnimation)->GetLength();
                if (destLen != 0.0f) {
                    while (curFrame >= destLen) {
                        curFrame -= destLen;
                    }
                }
                destAnim->SetFrame(curFrame + m_FrameDest);
            }

            doActivePostUpdate = false;
        }
    } else {
        m_ActiveAnimation = (RCKKeyedAnimation *) m_AnimDest;
        AlignCharacterWithRootPosition();
        m_AnimDest = nullptr;
        doActivePostUpdate = false;
    }

    if (doActivePostUpdate) {
        activeAnim = (CKAnimation *) m_ActiveAnimation;
        activeKeyed = nullptr;
        if (m_ActiveAnimation && CKIsChildClassOf((CKObject *) m_ActiveAnimation, CKCID_KEYEDANIMATION)) {
            activeKeyed = m_ActiveAnimation;
        }

        if (activeAnim && activeKeyed) {
            activeAnim->SetFrame(nextFrameForActive);
            if (m_RootBodyPart) {
                VxVector pos;
                ((CK3dEntity *) m_RootBodyPart)->GetPosition(&pos, nullptr);
                SetPosition(&pos, nullptr, TRUE);
            }
            activeKeyed->CenterAnimation(nextFrameForActive);
        }
    }

    for (int i = 0; i < m_SecondaryAnimationsCount; ++i) {
        CKSecondaryAnimation &secAnim = m_SecondaryAnimations[i];
        const CK_SECONDARYANIMATION_RUNTIME_MODE mode = secAnim.Mode;

        if (mode == CKSECONDARYANIMATIONRUNTIME_STARTINGWARP) {
            RCKKeyedAnimation *anim = secAnim.Animation;
            if (!anim) {
                continue;
            }

            const float next = ((CKAnimation *)anim)->GetNextFrame(deltat);
            const float remaining = next - ((CKAnimation *)anim)->GetLength();
            if (remaining < 0.0f) {
                ((CKAnimation *)anim)->SetFrame(next);
            } else {
                secAnim.Mode = CKSECONDARYANIMATIONRUNTIME_PLAYING;
                auto *source = (CKAnimation *)m_Context->GetObject(secAnim.AnimID);
                PreDeleteBodyPartsForAnimation(source);

                const float startFrame = secAnim.GetStartingFrame();
                if (source) {
                    source->SetFrame(remaining + startFrame);
                }
            }
            continue;
        }

        if (mode == CKSECONDARYANIMATIONRUNTIME_STOPPINGWARP) {
            RCKKeyedAnimation *anim = secAnim.Animation;
            if (!anim) {
                continue;
            }

            const float next = ((CKAnimation *)anim)->GetNextFrame(deltat);
            const float remaining = next - ((CKAnimation *)anim)->GetLength();
            if (remaining < 0.0f) {
                ((CKAnimation *)anim)->SetFrame(next);
            } else {
                const int removeIndex = i;
                --i;
                RemoveSecondaryAnimationAt(removeIndex);
            }
            continue;
        }

        if (mode != CKSECONDARYANIMATIONRUNTIME_PLAYING) {
            continue;
        }

        auto *anim = (CKAnimation *)m_Context->GetObject(secAnim.AnimID);
        if (!anim) {
            continue;
        }

        const float next = anim->GetNextFrame(deltat);
        const float remaining = next - anim->GetLength();
        if (remaining < 0.0f) {
            anim->SetFrame(next);
            continue;
        }

        int removeIt = 0;
        if ((secAnim.Flags & CKSECONDARYANIMATION_LOOP) != 0) {
            anim->SetFrame(remaining);
        } else if ((secAnim.Flags & CKSECONDARYANIMATION_LOOPNTIMES) != 0) {
            if ((int)--secAnim.LoopCountRemaining > 0) {
                anim->SetFrame(remaining);
            } else if ((secAnim.Flags & CKSECONDARYANIMATION_LASTFRAME) != 0) {
                anim->SetFrame(anim->GetLength());
                secAnim.Flags &= ~CKSECONDARYANIMATION_LOOPNTIMES;
            } else {
                removeIt = 1;
            }
        } else if ((secAnim.Flags & CKSECONDARYANIMATION_LASTFRAME) != 0) {
            anim->SetFrame(anim->GetLength());
        } else {
            removeIt = 1;
        }

        if (removeIt) {
            if ((secAnim.Flags & CKSECONDARYANIMATION_DOWARP) != 0 && CKIsChildClassOf((CKObject *)anim, CKCID_KEYEDANIMATION)) {
                RCKKeyedAnimation *active = m_ActiveAnimation;
                if (active && CKIsChildClassOf((CKObject *)active, CKCID_KEYEDANIMATION)) {
                    RCKKeyedAnimation *transition = secAnim.Animation;
                    if (!transition) {
                        CKBOOL isDynamic = m_Context->IsInDynamicCreationMode();
                        transition = (RCKKeyedAnimation *)m_Context->CreateObject(
                            CKCID_KEYEDANIMATION,
                            nullptr,
                            isDynamic ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK,
                            nullptr);
                        secAnim.Animation = transition;
                        if (transition) {
                            ((RCKAnimation *)transition)->m_Flags |= CKANIMATION_SECONDARYWARPER;
                        }
                    }

                    if (transition) {
                        float targetFrame = ((CKAnimation *)active)->GetFrame() + secAnim.WarpLength;
                        const float activeLen = ((CKAnimation *)active)->GetLength();
                        while (targetFrame >= activeLen) {
                            targetFrame -= activeLen;
                        }

                        transition->CreateTransition(anim, (CKAnimation *)active, 0, secAnim.WarpLength, targetFrame);
                        secAnim.Flags |= CKSECONDARYANIMATION_DOWARP;
                        ((CKAnimation *)transition)->SetFrame(0.0f);
                        PreDeleteBodyPartsForAnimation((CKAnimation *)transition);
                        secAnim.Mode = CKSECONDARYANIMATIONRUNTIME_STOPPINGWARP;
                        continue;
                    }
                }
            }

            const int removeIndex = i;
            --i;
            RemoveSecondaryAnimationAt(removeIndex);
        }
    }

    m_MoveableFlags &= ~VX_MOVEABLE_CHARACTERRENDERED;
    m_Context->AddProfileTime(CK_PROFILE_ANIMATIONTIME, profiler.Current());
}

void RCKCharacter::SetAutomaticProcess(CKBOOL process) {
    // Based on decompilation at 0x10010FE9
    if (process) {
        m_3dEntityFlags |= CK_3DENTITY_CHARACTERDOPROCESS;
    } else {
        m_3dEntityFlags &= ~CK_3DENTITY_CHARACTERDOPROCESS;
    }
}

CKBOOL RCKCharacter::IsAutomaticProcess() {
    // Based on IDA decompilation at 0x10011022
    // Returns non-zero if flag is set (cast to CKBOOL)
    return (m_3dEntityFlags & CK_3DENTITY_CHARACTERDOPROCESS) != 0;
}

void RCKCharacter::GetEstimatedVelocity(float deltat, VxVector *velocity) {
    // Based on decompilation at 0x1000FD26
    if (!velocity) return;

    // Default to zero
    *velocity = VxVector::axis0();

    CKAnimation *active = GetActiveAnimation();
    if (!active || !CKIsChildClassOf((CKObject *) active, CKCID_KEYEDANIMATION)) {
        return;
    }

    CKKeyedAnimation *keyedAnim = (CKKeyedAnimation *) active;
    CKObjectAnimation *rootAnim = keyedAnim->GetAnimation((CK3dEntity *) m_RootBodyPart);

    if (rootAnim) {
        float currentFrame = keyedAnim->GetFrame();
        float nextFrame = keyedAnim->GetNextFrame(deltat);

        VxVector pos1, pos2;
        rootAnim->EvaluatePosition(currentFrame, pos1);
        rootAnim->EvaluatePosition(nextFrame, pos2);

        *velocity = pos2 - pos1;
    }
}

//=============================================================================
// Secondary Animation Methods
//=============================================================================

CKERROR RCKCharacter::PlaySecondaryAnimation(CKAnimation *anim, float StartingFrame, CK_SECONDARYANIMATION_FLAGS PlayFlags, float warplength, int LoopCount) {
    // Based on IDA decompilation at 0x10010011
    if (!anim) {
        return CKERR_INVALIDPARAMETER;
    }

    // Check animation belongs to this character
    if (((RCKAnimation *)anim)->m_Character != this) {
        return CKERR_INVALIDPARAMETER;
    }

    // Handle FROMANIMATION flag - get mode from animation itself
    if ((PlayFlags & CKSECONDARYANIMATION_FROMANIMATION) != 0) {
        CK_SECONDARYANIMATION_FLAGS animMode = anim->GetSecondaryAnimationMode();
        if (animMode) {
            PlayFlags = animMode;
        }
    }

    CK_ID animID = anim->GetID();

    // Check if already playing
    for (int i = 0; i < m_SecondaryAnimationsCount; ++i) {
        if (m_SecondaryAnimations[i].AnimID == animID) {
            return CK_OK; // Already playing
        }
    }

    CKSecondaryAnimation *oldArray = m_SecondaryAnimations;

    // Ensure we have space
    if (m_SecondaryAnimationsAllocated <= m_SecondaryAnimationsCount) {
        m_SecondaryAnimations = new CKSecondaryAnimation[m_SecondaryAnimationsAllocated + 2];
        if (oldArray) {
            memcpy(m_SecondaryAnimations, oldArray, sizeof(CKSecondaryAnimation) * m_SecondaryAnimationsAllocated);
            delete[] oldArray;
        }
        // Initialize new slots (2 new entries)
        memset(&m_SecondaryAnimations[m_SecondaryAnimationsCount], 0, sizeof(CKSecondaryAnimation) * 2);
        m_SecondaryAnimationsAllocated += 2;
    }

    // Handle DOWARP flag - create transition from active animation
    if ((PlayFlags & CKSECONDARYANIMATION_DOWARP) != 0) {
        if (CKIsChildClassOf((CKObject *)anim, CKCID_KEYEDANIMATION)) {
            RCKKeyedAnimation *activeKeyed = m_ActiveAnimation;
            if (activeKeyed && CKIsChildClassOf((CKObject *)activeKeyed, CKCID_KEYEDANIMATION)) {
                RCKKeyedAnimation *transition = m_SecondaryAnimations[m_SecondaryAnimationsCount].Animation;
                if (!transition) {
                    CKBOOL isDynamic = m_Context->IsInDynamicCreationMode();
                    transition = (RCKKeyedAnimation *)m_Context->CreateObject(
                        CKCID_KEYEDANIMATION,
                        nullptr,
                        isDynamic ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK,
                        nullptr);

                    CKDWORD flags = ((RCKAnimation *)transition)->m_Flags;
                    flags |= CKANIMATION_SECONDARYWARPER;
                    ((RCKAnimation *)transition)->m_Flags = flags;
                    ((CKKeyedAnimation *)transition)->Clear();
                }

                m_SecondaryAnimations[m_SecondaryAnimationsCount].Animation = transition;
                transition->CreateTransition((CKAnimation *)activeKeyed, anim, 0, warplength, StartingFrame);
                m_SecondaryAnimations[m_SecondaryAnimationsCount].Mode = CKSECONDARYANIMATIONRUNTIME_STOPPINGWARP;
                m_SecondaryAnimations[m_SecondaryAnimationsCount].WarpLength = warplength;
                ((CKAnimation *)transition)->SetFrame(0.0f);
                PreDeleteBodyPartsForAnimation((CKAnimation *)transition);
                m_SecondaryAnimations[m_SecondaryAnimationsCount].Mode = CKSECONDARYANIMATIONRUNTIME_STARTINGWARP;
            }
        }
    } else {
        // No warp - just set the frame and start playing
        anim->SetFrame(StartingFrame);
        if (CKIsChildClassOf((CKObject *)anim, CKCID_KEYEDANIMATION)) {
            PreDeleteBodyPartsForAnimation(anim);
        }
        m_SecondaryAnimations[m_SecondaryAnimationsCount].Mode = CKSECONDARYANIMATIONRUNTIME_PLAYING;
    }

    // Set common fields
    m_SecondaryAnimations[m_SecondaryAnimationsCount].SetStartingFrame(StartingFrame);
    m_SecondaryAnimations[m_SecondaryAnimationsCount].AnimID = animID;
    m_SecondaryAnimations[m_SecondaryAnimationsCount].Flags = PlayFlags;
    m_SecondaryAnimations[m_SecondaryAnimationsCount].LoopCountRemaining = LoopCount;

    m_SecondaryAnimationsCount++;
    return CK_OK;
}

CKERROR RCKCharacter::StopSecondaryAnimation(CKAnimation *anim, CKBOOL warp, float warplength) {
    // Based on IDA decompilation at 0x1001044B
    if (!anim) {
        return CKERR_INVALIDPARAMETER;
    }

    if (((RCKAnimation *)anim)->m_Character != this) {
        return CKERR_INVALIDPARAMETER;
    }

    CK_ID animID = anim->GetID();

    // Find the secondary animation
    int idx = -1;
    for (int i = 0; i < m_SecondaryAnimationsCount && m_SecondaryAnimations[i].AnimID != animID; ++i) {
        idx = i;
    }

    // Check if we found it (need to verify the condition matched)
    if (idx >= m_SecondaryAnimationsCount || m_SecondaryAnimations[idx].AnimID != animID) {
        // Re-scan properly
        idx = -1;
        for (int i = 0; i < m_SecondaryAnimationsCount; ++i) {
            if (m_SecondaryAnimations[i].AnimID == animID) {
                idx = i;
                break;
            }
        }
    }

    if (idx < 0 || idx >= m_SecondaryAnimationsCount) {
        return CK_OK; // Not playing
    }

    int warped = 0;
    if (warp) {
        if (m_SecondaryAnimations[idx].Mode == CKSECONDARYANIMATIONRUNTIME_STOPPINGWARP) {
            return CK_OK;
        }

        RCKKeyedAnimation *activeKeyed = m_ActiveAnimation;
        if (activeKeyed && CKIsChildClassOf((CKObject *)activeKeyed, CKCID_KEYEDANIMATION)) {
            RCKKeyedAnimation *transition = m_SecondaryAnimations[idx].Animation;
            if (!transition) {
                CKBOOL isDynamic = m_Context->IsInDynamicCreationMode();
                transition = (RCKKeyedAnimation *)m_Context->CreateObject(
                    CKCID_KEYEDANIMATION,
                    nullptr,
                    isDynamic ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK,
                    nullptr);
                m_SecondaryAnimations[idx].Animation = transition;
                if (transition) {
                    ((RCKAnimation *)transition)->m_Flags |= CKANIMATION_SECONDARYWARPER;
                }
            }

            CKAnimation *fromAnim = nullptr;
            if (m_SecondaryAnimations[idx].Mode == CKSECONDARYANIMATIONRUNTIME_PLAYING) {
                fromAnim = (CKAnimation *)m_Context->GetObject(m_SecondaryAnimations[idx].AnimID);
            } else {
                fromAnim = (CKAnimation *)m_SecondaryAnimations[idx].Animation;
            }

            float targetFrame = ((CKAnimation *)activeKeyed)->GetFrame() + warplength;
            const float activeLen = ((CKAnimation *)activeKeyed)->GetLength();
            while (targetFrame >= activeLen) {
                targetFrame -= activeLen;
            }

            if (transition && fromAnim && CKIsChildClassOf((CKObject *)fromAnim, CKCID_KEYEDANIMATION)) {
                transition->CreateTransition(fromAnim, (CKAnimation *)activeKeyed, 0, warplength, targetFrame);
                m_SecondaryAnimations[idx].Flags |= CKSECONDARYANIMATION_DOWARP;
                m_SecondaryAnimations[idx].WarpLength = warplength;
                ((CKAnimation *)transition)->SetFrame(0.0f);
                PreDeleteBodyPartsForAnimation((CKAnimation *)transition);
                m_SecondaryAnimations[idx].Mode = CKSECONDARYANIMATIONRUNTIME_STOPPINGWARP;
                warped = 1;
            }
        }
    }

    if (!warped) {
        RemoveSecondaryAnimationAt(idx);
    }
    return CK_OK;
}

CKERROR RCKCharacter::StopSecondaryAnimation(CKAnimation *anim, float warplength) {
    // Based on IDA decompilation at 0x100106FC
    if (!anim) {
        return CKERR_INVALIDPARAMETER;
    }

    // Get secondary animation mode from animation and check DOWARP flag
    // The mode is returned as-is from GetSecondaryAnimationMode, then AND with 0x800000
    // 0x800000 = CKSECONDARYANIMATION_DOWARP (0x80) << 16 after it's stored in m_Flags
    // But GetSecondaryAnimationMode returns the shifted-back value, so we check 0x80
    int mode = anim->GetSecondaryAnimationMode();
    CKBOOL warp = (mode & CKSECONDARYANIMATION_DOWARP) != 0;

    return StopSecondaryAnimation(anim, warp, warplength);
}

int RCKCharacter::GetSecondaryAnimationsCount() {
    return m_SecondaryAnimationsCount;
}

CKAnimation *RCKCharacter::GetSecondaryAnimation(int index) {
    if (index < 0 || index >= m_SecondaryAnimationsCount || !m_SecondaryAnimations) {
        return nullptr;
    }
    return (CKAnimation *)m_Context->GetObject(m_SecondaryAnimations[index].AnimID);
}

void RCKCharacter::FlushSecondaryAnimations() {
    // Based on IDA decompilation at 0x100103CD
    // First find floor reference if not set
    if (!m_FloorRef) {
        FindFloorReference();
    }

    // Iterate through secondary animations and update body parts
    for (CKWORD i = 0; i < m_SecondaryAnimationsCount; ++i) {
        RCKAnimation *anim = (RCKAnimation *)m_Context->GetObjectA(m_SecondaryAnimations[i].AnimID);
        if (anim) {
            // For each animation, iterate through entities and notify body parts
            // This corresponds to sub_10048148 which iterates entities and calls method at vtable[118]
            // on body parts with parameter 0
            NotifyBodyPartsInAnimation(anim, 0);
        }
    }

    m_SecondaryAnimationsCount = 0;
}

//=============================================================================
// Utility Methods
//=============================================================================

void RCKCharacter::AlignCharacterWithRootPosition() {
    // Based on decompilation at 0x10011DE8
    CKAnimation *active = GetActiveAnimation();

    if (!active) {
        // No active animation - just set position from root body part
        if (m_RootBodyPart) {
            VxVector pos;
            ((CK3dEntity *) m_RootBodyPart)->GetPosition(&pos, nullptr);
            SetPosition(&pos, nullptr, TRUE);
        }
        return;
    }

    if (!CKIsChildClassOf((CKObject *) active, CKCID_KEYEDANIMATION)) {
        return;
    }

    CKKeyedAnimation *keyedAnim = (CKKeyedAnimation *) active;

    if (m_RootBodyPart) {
        // Get position from root body part and set character position
        VxVector pos;
        ((CK3dEntity *) m_RootBodyPart)->GetPosition(&pos, nullptr);
        SetPosition(&pos, nullptr, TRUE);
    }

    // Center animation
    float frame = keyedAnim->GetFrame();
    keyedAnim->CenterAnimation(frame);
}

CK3dEntity *RCKCharacter::GetFloorReferenceObject() {
    return (CK3dEntity *) m_FloorRef;
}

void RCKCharacter::SetFloorReferenceObject(CK3dEntity *FloorRef) {
    m_FloorRef = (RCK3dEntity *) FloorRef;
}

void RCKCharacter::SetAnimationLevelOfDetail(float LOD) {
    m_AnimationLevelOfDetail = LOD;
}

float RCKCharacter::GetAnimationLevelOfDetail() {
    return m_AnimationLevelOfDetail;
}

void RCKCharacter::GetWarperParameters(CKDWORD *TransitionMode, CKAnimation **AnimSrc, float *FrameSrc,
                                       CKAnimation **AnimDest, float *FrameDest) {
    // Based on decompilation at 0x10010911
    if (TransitionMode) *TransitionMode = m_TransitionMode;
    if (AnimSrc) *AnimSrc = (CKAnimation *) m_AnimSrc;
    if (AnimDest) *AnimDest = (CKAnimation *) m_AnimDest;
    if (FrameSrc) *FrameSrc = m_FrameSrc;
    if (FrameDest) *FrameDest = m_FrameDest;
}

//=============================================================================
// Private Helper Methods
//=============================================================================

void RCKCharacter::FindFloorReference() {
    // Based on decompilation at 0x10012018
    if (!m_RootBodyPart) {
        return;
    }

    static constexpr const char *kFloorRef = "FloorRef";
    static constexpr const char *kFootsteps = "Footsteps";
    static constexpr const char *kFoot = "Foot";
    static constexpr const char *kPas = "Pas";

    for (CKObject **it = m_RootBodyPart->m_Children.Begin(); it != m_RootBodyPart->m_Children.End(); ++it) {
        CKObject *child = *it;
        if (!child) continue;
        CKSTRING name = child->GetName();
        if (name && strstr(name, kFloorRef)) {
            m_FloorRef = (RCK3dEntity *) child;
            return;
        }
    }

    for (CKObject **it = m_RootBodyPart->m_Children.Begin(); it != m_RootBodyPart->m_Children.End(); ++it) {
        CKObject *child = *it;
        if (!child) continue;
        CKSTRING name = child->GetName();
        if (name && strstr(name, kFootsteps)) {
            m_FloorRef = (RCK3dEntity *) child;
            return;
        }
    }

    for (CKObject **it = m_RootBodyPart->m_Children.Begin(); it != m_RootBodyPart->m_Children.End(); ++it) {
        CKObject *child = *it;
        if (!child) continue;
        CKSTRING name = child->GetName();
        if (name && strstr(name, kFoot)) {
            m_FloorRef = (RCK3dEntity *) child;
            return;
        }
    }

    for (CKObject **it = m_RootBodyPart->m_Children.Begin(); it != m_RootBodyPart->m_Children.End(); ++it) {
        CKObject *child = *it;
        if (!child) continue;
        CKSTRING name = child->GetName();
        if (name && strstr(name, kPas)) {
            m_FloorRef = (RCK3dEntity *) child;
            return;
        }
    }
}

void RCKCharacter::RemoveSecondaryAnimationAt(int index) {
    // Based on IDA decompilation at 0x10011AB1
    if (index < 0 || index >= m_SecondaryAnimationsCount) {
        return;
    }

    CKAnimation *anim = (CKAnimation *)m_Context->GetObject(m_SecondaryAnimations[index].AnimID);
    if (anim) {
        PreDeleteBodyPartsForAnimation(nullptr); // Pass null to indicate no animation context
    }

    const int newCount = m_SecondaryAnimationsCount - 1;
    m_SecondaryAnimationsCount = (CKWORD)newCount;

    if (index == newCount) {
        // Last element - just clear it
        m_SecondaryAnimations[index].StartingFrameBits = 0;
        m_SecondaryAnimations[index].AnimID = 0;
        m_SecondaryAnimations[index].Flags = 0;
        m_SecondaryAnimations[index].WarpLength = 0;
        m_SecondaryAnimations[index].Padding = 0;
        m_SecondaryAnimations[index].LoopCountRemaining = 0;
        return;
    }

    // Preserve the Animation pointer from the element being removed
    RCKKeyedAnimation *preserved = m_SecondaryAnimations[index].Animation;

    // Move remaining elements down
    for (int i = index; i < newCount; ++i) {
        memcpy(&m_SecondaryAnimations[i], &m_SecondaryAnimations[i + 1], sizeof(CKSecondaryAnimation));
    }

    // Clear the last slot and restore preserved animation pointer
    m_SecondaryAnimations[newCount].StartingFrameBits = 0;
    m_SecondaryAnimations[newCount].AnimID = 0;
    m_SecondaryAnimations[newCount].Flags = 0;
    m_SecondaryAnimations[newCount].WarpLength = 0;
    m_SecondaryAnimations[newCount].Padding = 0;
    m_SecondaryAnimations[newCount].LoopCountRemaining = 0;
    m_SecondaryAnimations[newCount].Animation = preserved;
}

void RCKCharacter::PreDeleteBodyPartsForAnimation(CKAnimation *anim) {
    // Based on decompilation helper at 0x10048148
    // Iterates ObjectAnimations in a keyed animation and calls PreDelete on body parts.
    if (!anim || !CKIsChildClassOf((CKObject *) anim, CKCID_KEYEDANIMATION)) {
        return;
    }

    auto *keyed = static_cast<CKKeyedAnimation *>(anim);
    const int count = keyed->GetAnimationCount();
    for (int idx = 0; idx < count; ++idx) {
        CKObjectAnimation *objAnim = keyed->GetAnimation(idx);
        if (!objAnim) {
            continue;
        }

        CK3dEntity *ent = objAnim->Get3dEntity();
        if (ent && CKIsChildClassOf((CKObject *) ent, CKCID_BODYPART)) {
            ent->PreDelete();
        }
    }
}
