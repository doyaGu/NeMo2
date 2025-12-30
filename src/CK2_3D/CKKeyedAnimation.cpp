#include "RCKKeyedAnimation.h"

#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "CKDependencies.h"
#include "CKGlobals.h"
#include "CKMemoryPool.h"
#include "RCKObjectAnimation.h"
#include "RCK3dEntity.h"
#include "RCKCharacter.h"

CK_CLASSID RCKKeyedAnimation::m_ClassID = CKCID_KEYEDANIMATION;

//=============================================================================
// Constructor/Destructor
//=============================================================================

RCKKeyedAnimation::RCKKeyedAnimation(CKContext *Context, CKSTRING name)
    : RCKAnimation(Context, name),
      m_Merged(FALSE),
      m_MergeFactor(0.5f),
      m_RootAnimation(nullptr),
      m_Vector(0.0f, 0.0f, 0.0f) {}

RCKKeyedAnimation::~RCKKeyedAnimation() {
    RCKKeyedAnimation::Clear();
}

CK_CLASSID RCKKeyedAnimation::GetClassID() {
    return m_ClassID;
}

//=============================================================================
// Virtual Methods from CKObject
//=============================================================================

void RCKKeyedAnimation::CheckPreDeletion() {
    RCKAnimation::CheckPreDeletion();

    // Check if any animations were deleted
    if (m_Animations.Check()) {
        ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
        UpdateRootEntity();
    }

    // Clear root animation if it's been deleted
    if (m_RootAnimation) {
        if (m_RootAnimation->IsToBeDeleted()) {
            m_RootAnimation = nullptr;
        }
    }
}

CKBOOL RCKKeyedAnimation::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    if (cid == CKCID_OBJECTANIMATION && m_Animations.IsHere(obj))
        return TRUE;

    return CKObject::IsObjectUsed(obj, cid);
}

void RCKKeyedAnimation::PreSave(CKFile *file, CKDWORD flags) {
    CKObject::PreSave(file, flags);
    file->SaveObjects((CKObject **)m_Animations.Begin(), m_Animations.Size(), flags);
}

//=============================================================================
// Serialization
//=============================================================================

CKStateChunk *RCKKeyedAnimation::Save(CKFile *file, CKDWORD flags) {
    // Create state chunk with class ID 18 (CKCID_KEYEDANIMATION)
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_KEYEDANIMATION, file);
    CKStateChunk *baseChunk = RCKAnimation::Save(file, flags);

    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    // Save animations list (identifier 0x1000)
    if (file || (flags & CK_STATESAVE_KEYEDANIMANIMLIST) != 0) {
        chunk->WriteIdentifier(CK_STATESAVE_KEYEDANIMANIMLIST);
        m_Animations.Save(chunk);
    }

    // Save merged state and factor (identifier 0x100000)
    if (file || (flags & CK_STATESAVE_KEYEDANIMMERGE) != 0) {
        chunk->WriteIdentifier(CK_STATESAVE_KEYEDANIMMERGE);
        chunk->WriteInt(m_Merged);
        chunk->WriteFloat(m_MergeFactor);
    }

    // Save animation sub-chunks (identifier 0x200000) - only when not saving to file
    if (!file && (flags & CK_STATESAVE_KEYEDANIMSUBANIMS) != 0) {
        chunk->WriteIdentifier(CK_STATESAVE_KEYEDANIMSUBANIMS);
        chunk->WriteDword(m_Animations.Size());
        for (CKObject **it = m_Animations.Begin(); it != m_Animations.End(); ++it) {
            CKObject *obj = *it;
            CKStateChunk *subChunk = obj ? obj->Save(nullptr, flags) : nullptr;
            chunk->WriteObject(obj);
            chunk->WriteSubChunk(subChunk);
            DeleteCKStateChunk(subChunk);
        }
    }

    // Close the chunk
    if (GetClassID() == CKCID_KEYEDANIMATION) {
        chunk->CloseChunk();
    } else {
        chunk->UpdateDataSize();
    }

    return chunk;
}

CKERROR RCKKeyedAnimation::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Load base class
    RCKAnimation::Load(chunk, file);

    // Load animations list
    if (chunk->SeekIdentifier(CK_STATESAVE_KEYEDANIMANIMLIST)) {
        m_Animations.Clear();
        m_Animations.Load(m_Context, chunk);
        m_Animations.Check();
    }

    // Load merged state and factor
    if (chunk->SeekIdentifier(CK_STATESAVE_KEYEDANIMMERGE)) {
        m_Merged = chunk->ReadInt();
        m_MergeFactor = chunk->ReadFloat();
    }

    // Load animation sub-chunks - only when not loading from file
    if (!file) {
        if (chunk->SeekIdentifier(CK_STATESAVE_KEYEDANIMSUBANIMS)) {
            int count = chunk->ReadDword();
            for (int i = 0; i < count; ++i) {
                CKObject *obj = chunk->ReadObject(m_Context);
                CKStateChunk *subChunk = chunk->ReadSubChunk();
                if (obj) {
                    obj->Load(subChunk, nullptr);
                }
                DeleteCKStateChunk(subChunk);
            }
        }
        ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
        UpdateRootEntity();
    }

    m_Flags &= ~CKANIMATION_SUBANIMSSORTED;

    // Process each animation and set up parent references
    for (CKObject **it = m_Animations.Begin(); it != m_Animations.End(); ++it) {
        RCKObjectAnimation *anim = (RCKObjectAnimation *) *it;
        SetParentKeyedAnimation(anim, this);

        // Check for app data containing vector
        VxVector *appData = (VxVector *) anim->GetAppData();
        if (appData) {
            m_Vector = *appData;
            delete appData;
            anim->SetAppData(nullptr);
        }
    }

    return CK_OK;
}

int RCKKeyedAnimation::GetMemoryOccupation() {
    int size = RCKAnimation::GetMemoryOccupation() + (sizeof(RCKKeyedAnimation) - sizeof(RCKAnimation));
    size += m_Animations.GetMemoryOccupation(FALSE);
    return size;
}

CKERROR RCKKeyedAnimation::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    m_Animations.Prepare(context);
    return context.FinishPrepareDependencies(this, RCKKeyedAnimation::m_ClassID);
}

CKERROR RCKKeyedAnimation::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = RCKAnimation::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    m_Animations.Remap(context);
    return CK_OK;
}

CKERROR RCKKeyedAnimation::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = RCKAnimation::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKKeyedAnimation *src = (RCKKeyedAnimation *) &o;

    m_RootAnimation = nullptr;
    m_Animations = src->m_Animations;  // Array copy
    m_Merged = src->m_Merged;
    m_MergeFactor = src->m_MergeFactor;
    m_Vector = src->m_Vector;

    return CK_OK;
}

//=============================================================================
// CKKeyedAnimation Virtual Methods
//=============================================================================

CKERROR RCKKeyedAnimation::AddAnimation(CKObjectAnimation *anim) {
    if (!anim)
        return CKERR_INVALIDPARAMETER;

    m_Animations.PushBack(anim);
    SetParentKeyedAnimation((RCKObjectAnimation *) anim, this);

    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
    m_Flags &= ~CKANIMATION_SUBANIMSSORTED;

    return CK_OK;
}

CKERROR RCKKeyedAnimation::RemoveAnimation(CKObjectAnimation *anim) {
    if (!anim)
        return CKERR_INVALIDPARAMETER;

    m_Animations.RemoveObject(anim);

    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
    m_Flags &= ~CKANIMATION_SUBANIMSSORTED;

    return CK_OK;
}

int RCKKeyedAnimation::GetAnimationCount() {
    return m_Animations.Size();
}

CKObjectAnimation *RCKKeyedAnimation::GetAnimation(CK3dEntity *ent) {
    if (ent == (CK3dEntity *) m_RootEntity) {
        if (m_RootAnimation) {
            return (CKObjectAnimation *) m_RootAnimation;
        }

        // Search for and cache root animation
        for (CKObject **it = m_Animations.Begin(); it != m_Animations.End(); ++it) {
            RCKObjectAnimation *anim = (RCKObjectAnimation *) *it;
            if (anim && anim->Get3dEntity() == ent) {
                m_RootAnimation = anim;
                return (CKObjectAnimation *) anim;
            }
        }
        return nullptr;
    }

    // Search for animation by entity
    for (CKObject **it = m_Animations.Begin(); it != m_Animations.End(); ++it) {
        RCKObjectAnimation *anim = (RCKObjectAnimation *) *it;
        if (anim && anim->Get3dEntity() == ent) {
            return (CKObjectAnimation *) anim;
        }
    }
    return nullptr;
}

CKObjectAnimation *RCKKeyedAnimation::GetAnimation(int index) {
    return (CKObjectAnimation *) m_Animations[index];
}

void RCKKeyedAnimation::Clear() {
    m_Animations.Clear();
    m_RootAnimation = nullptr;
}

//=============================================================================
// Animation Manipulation Methods
//=============================================================================

void RCKKeyedAnimation::CenterAnimation(float frame) {
    if (!m_RootAnimation) {
        m_RootAnimation = (RCKObjectAnimation *) GetAnimation((CK3dEntity *) m_RootEntity);
    }

    if (m_RootAnimation) {
        VxVector position(0.0f, 0.0f, 0.0f);
        m_RootAnimation->EvaluatePosition(frame, position);

        // Negate the position
        m_Vector.x = -position.x;
        m_Vector.y = -position.y;
        m_Vector.z = -position.z;

        SetParentKeyedAnimation(m_RootAnimation, this);
    }
}

CKAnimation *RCKKeyedAnimation::CreateMergedAnimation(CKAnimation *anim2, CKBOOL dynamic) {
    if (!anim2)
        return nullptr;

    // Get the character associated with this animation
    CKCharacter *character = GetCharacter();
    if (!character)
        return nullptr;

    RCKKeyedAnimation *otherAnim = (RCKKeyedAnimation *) anim2;

    // Determine creation options based on GetFlags() & 0x80 (CK_OBJECT_DYNAMIC)
    CKDWORD dynamicFlag = GetFlags() & CK_OBJECT_DYNAMIC;
    CK_OBJECTCREATION_OPTIONS options = dynamicFlag ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK;

    // Create name for the merged animation
    char buffer[260];
    CKSTRING otherName = otherAnim->GetName();
    CKSTRING thisName = GetName();

    if (!otherName)
        otherName = "";
    if (thisName)
        sprintf(buffer, "%s+%s", thisName, otherName);
    else
        sprintf(buffer, "%s+%s", "", otherName);

    // Create a new keyed animation object
    RCKKeyedAnimation *merged = (RCKKeyedAnimation *) m_Context->CreateObject(CKCID_KEYEDANIMATION, buffer, options, nullptr);
    if (!merged)
        return nullptr;

    // Set the character for the merged animation
    merged->m_Character = (RCKCharacter *) character;

    // Add merged animation to character's scenes
    ((CKBeObject *) character)->AddToSelfScenes((CKSceneObject *) merged);

    // Set length to the max of both animations
    float maxLength = (otherAnim->GetLength() >= m_Length) ? otherAnim->GetLength() : m_Length;

    // Set merged animation properties
    merged->m_Merged = TRUE;
    merged->m_RootEntity = m_RootEntity;

    // Process all animations from this keyed animation
    int thisAnimCount = m_Animations.Size();

    for (int i = 0; i < thisAnimCount; ++i) {
        RCKObjectAnimation *thisObjAnim = (RCKObjectAnimation *) m_Animations[i];
        if (!thisObjAnim)
            continue;

        CK3dEntity *entity = thisObjAnim->Get3dEntity();
        if (!entity)
            continue;

        // Find matching entity in other animation
        CKObjectAnimation *otherObjAnim = otherAnim->GetAnimation(entity);

        if (otherObjAnim) {
            // Both animations have this entity, create merged object animation
            // Pass dynamic flag to sub-animation's CreateMergedAnimation
            // Note: IDA shows flag manipulation before call, but simplified here
            CKObjectAnimation *mergedObjAnim = thisObjAnim->CreateMergedAnimation(otherObjAnim, dynamicFlag ? TRUE : FALSE);
            if (mergedObjAnim) {
                merged->AddAnimation(mergedObjAnim);
            }
        } else if (!dynamic) {
            // Only this animation has this entity - copy it and scale length
            CKDependencies dep;
            CKCopyDefaultClassDependencies(dep, CK_DEPENDENCIES_COPY);
            dep.m_Flags = CK_DEPENDENCIES_CUSTOM;

            CKObjectAnimation *copiedAnim = (CKObjectAnimation *) m_Context->CopyObject(
                thisObjAnim, &dep, nullptr, options);

            if (copiedAnim) {
                // Scale the animation length: maxLength / original length
                float origLength = copiedAnim->GetLength();
                if (origLength > 0.0f) {
                    float scale = maxLength / origLength;
                    copiedAnim->SetLength(scale);
                }
                merged->AddAnimation(copiedAnim);
            }
        }
    }

    // Process animations from other animation that are not in this animation
    if (!dynamic) {
        int otherAnimCount = otherAnim->GetAnimationCount();
        for (int j = 0; j < otherAnimCount; ++j) {
            RCKObjectAnimation *otherObjAnim = (RCKObjectAnimation *) otherAnim->GetAnimation(j);
            if (!otherObjAnim)
                continue;

            CK3dEntity *otherEntity = otherObjAnim->Get3dEntity();
            if (!otherEntity)
                continue;

            // Check if this entity already exists in this animation
            CKObjectAnimation *existingAnim = GetAnimation(otherEntity);

            if (!existingAnim) {
                // Copy this animation and scale length
                CKDependencies dep;
                CKCopyDefaultClassDependencies(dep, CK_DEPENDENCIES_COPY);
                dep.m_Flags = CK_DEPENDENCIES_CUSTOM;

                CKObjectAnimation *copiedAnim = (CKObjectAnimation *) m_Context->CopyObject(
                    otherObjAnim, &dep, nullptr, options);

                if (copiedAnim) {
                    // Scale the animation length: maxLength / original length
                    float origLength = copiedAnim->GetLength();
                    if (origLength > 0.0f) {
                        float scale = maxLength / origLength;
                        copiedAnim->SetLength(scale);
                    }
                    merged->AddAnimation(copiedAnim);
                }
            }
        }
    }

    // Set the length of merged animation
    merged->SetLength(maxLength);

    // Update animation state
    merged->ModifyObjectFlags(CK_OBJECT_UPTODATE, 0);
    merged->UpdateRootEntity();

    return (CKAnimation *) merged;
}

float RCKKeyedAnimation::CreateTransition(CKAnimation *out, CKAnimation *in, CKDWORD OutTransitionMode, float length, float FrameTo) {
    // IDA: 0x10048b57 - RCKKeyedAnimation::CreateTransition
    // Note: Parameter order is (out, in) based on IDA analysis
    if (!out)
        return 0.0f;

    RCKKeyedAnimation *animOut = (RCKKeyedAnimation *) out;
    RCKKeyedAnimation *animIn = (RCKKeyedAnimation *) in;

    // Ensure length is at least 1 frame
    if (length < 1.0f)
        length = 1.0f;

    if (!animIn)
        return 0.0f;

    // Set SUBANIMSSORTED flag on this and both animations
    m_Flags |= CKANIMATION_SUBANIMSSORTED;

    if (!(animIn->m_Flags & CKANIMATION_SUBANIMSSORTED)) {
        // SortAnimations would be called here
        animIn->m_Flags |= CKANIMATION_SUBANIMSSORTED;
    }

    if (!(animOut->m_Flags & CKANIMATION_SUBANIMSSORTED)) {
        // SortAnimations would be called here
        animOut->m_Flags |= CKANIMATION_SUBANIMSSORTED;
    }

    // Handle OutTransitionMode flag 0x100 - compute FrameTo from animation length
    if ((OutTransitionMode & CK_TRANSITION_WARPTOSAMEPOS) == CK_TRANSITION_WARPTOSAMEPOS) {
        FrameTo = animIn->GetLength() * animOut->m_MergeFactor;
    }

    // If both animations are the same, return 0
    if (animIn == animOut)
        return 0.0f;

    // Get the current step from animIn
    float inStep = animIn->GetLength();

    // Compute step values
    float stepTo = FrameTo / animOut->m_MergeFactor;

    // Set up the root entity and character from animOut
    m_RootEntity = animOut->m_RootEntity;
    m_RootAnimation = nullptr;
    m_Character = animOut->m_Character;

    // Check that both animations have the same character
    if (animIn->m_Character != m_Character)
        return -1.0f;

    // Get the number of steps in animOut
    float outStepCount = fabsf(animOut->GetLength());

    // Copy the ALLOWTURN flag from animOut
    LinkToFrameRate((CKBOOL)(outStepCount), (float)(int)outStepCount);

    if (animOut->m_Flags & CKANIMATION_ALLOWTURN) {
        m_Flags |= CKANIMATION_ALLOWTURN;
    } else {
        m_Flags &= ~CKANIMATION_ALLOWTURN;
    }

    // Get counts of object animations
    int inAnimCount = animIn->GetAnimationCount();
    int outAnimCount = animOut->GetAnimationCount();

    // Build mapping arrays: outList[i] = animOut subanim, outToIn[i] = matching animIn subanim (by entity)
    // Use CKMemoryPool (dword-sized) to avoid std::vector and match SDK conventions.
    const size_t ptrBytes = sizeof(RCKObjectAnimation *);
    const size_t mappingBytes = (size_t)outAnimCount * ptrBytes * 2;
    const int mappingDwords = (int)((mappingBytes + sizeof(XDWORD) - 1) / sizeof(XDWORD));

    CKMemoryPool mappingPool(m_Context, mappingDwords);
    char *mappingMem = (char *)mappingPool.Mem();
    memset(mappingMem, 0, mappingBytes);

    RCKObjectAnimation **outToIn = (RCKObjectAnimation **)mappingMem;
    RCKObjectAnimation **outList = (RCKObjectAnimation **)(mappingMem + (size_t)outAnimCount * ptrBytes);

    // Build the mapping: for each animation in animOut, find matching entity in animIn
    for (int i = 0; i < outAnimCount; ++i) {
        RCKObjectAnimation *outObjAnim = (RCKObjectAnimation *)animOut->GetAnimation(i);
        outList[i] = outObjAnim;

        if (!outObjAnim)
            continue;

        CK3dEntity *entity = outObjAnim->Get3dEntity();
        if (!entity)
            continue;

        for (CKObject **it = animIn->m_Animations.Begin(); it != animIn->m_Animations.End(); ++it) {
            RCKObjectAnimation *inObjAnim = (RCKObjectAnimation *)*it;
            if (inObjAnim && inObjAnim->Get3dEntity() == entity) {
                outToIn[i] = inObjAnim;
                break;
            }
        }
    }

    // Check if we're using velocity-based warp (0x20 flag)
    CKAnimKey *velocityData = nullptr;
    // Simplified: skip velocity calculations (would need CKMemoryPool)
    // The original uses complex position/rotation velocity matching

    // Check the animations array
    m_Animations.Check();

    // Ensure we have enough object animations (match outAnimCount)
    int currentCount = m_Animations.Size();
    if (currentCount < outAnimCount) {
        int neededCount = outAnimCount - currentCount;
        for (int i = 0; i < neededCount; ++i) {
            CKBOOL isDynamic = (GetFlags() & CK_OBJECT_DYNAMIC) != 0;
            CK_OBJECTCREATION_OPTIONS opts = isDynamic ? CK_OBJECTCREATION_DYNAMIC : CK_OBJECTCREATION_NONAMECHECK;

            RCKObjectAnimation *newAnim = (RCKObjectAnimation *) m_Context->CreateObject(
                CKCID_OBJECTANIMATION, nullptr, opts, nullptr);

            if (newAnim) {
                m_Animations.PushBack(newAnim);
                SetParentKeyedAnimation(newAnim, this);
            }
        }
    }

    // Mark all object animations with flag 0xC0000000 initially
    for (CKObject **it = m_Animations.Begin(); it != m_Animations.End(); ++it) {
        RCKObjectAnimation *objAnim = (RCKObjectAnimation *) *it;
        if (objAnim) {
            objAnim->m_Flags |= (CK_OBJECTANIMATION_WARPER | CK_OBJECTANIMATION_RESERVED);
        }
    }

    // Create transition for each object animation
    // Check if animIn == this (special case for in-place transition)
    if (animIn == (RCKKeyedAnimation *) this) {
        for (int k = 0; k < outAnimCount; ++k) {
            RCKObjectAnimation *outObjAnim = outList[k];
            RCKObjectAnimation *inObjAnim = outToIn[k];

            if (outObjAnim && inObjAnim) {
                inObjAnim->m_Flags &= ~CK_OBJECTANIMATION_WARPER;

                CKBOOL veloc = FALSE;
                CKBOOL dontTurn = FALSE;

                if (m_RootEntity == (RCK3dEntity *) outObjAnim->Get3dEntity()) {
                    veloc = (OutTransitionMode & CK_TRANSITION_USEVELOCITY) != 0;
                    dontTurn = (animIn->m_Flags & CKANIMATION_ALIGNORIENTATION) != 0;
                }

                inObjAnim->CreateTransition(length, inObjAnim, inStep, outObjAnim, stepTo, veloc, dontTurn, velocityData);
            }
        }
    } else {
        // Normal case: animIn != this
        int thisAnimCount = m_Animations.Size();
        for (int k = 0; k < thisAnimCount; ++k) {
            RCKObjectAnimation *outObjAnim = nullptr;
            RCKObjectAnimation *inObjAnim = nullptr;

            if (k < outAnimCount) {
                outObjAnim = outList[k];
                inObjAnim = outToIn[k];
            }

            RCKObjectAnimation *objAnim = (RCKObjectAnimation *) m_Animations[k];
            if (!objAnim)
                continue;

            objAnim->m_Flags &= ~CK_OBJECTANIMATION_WARPER;

            if (outObjAnim && inObjAnim && objAnim) {
                CKBOOL veloc = FALSE;
                CKBOOL dontTurn = FALSE;

                if (m_RootEntity == (RCK3dEntity *) outObjAnim->Get3dEntity()) {
                    veloc = (OutTransitionMode & CK_TRANSITION_USEVELOCITY) != 0;
                    dontTurn = (animIn->m_Flags & CKANIMATION_ALIGNORIENTATION) != 0;
                }

                objAnim->CreateTransition(length, inObjAnim, inStep, outObjAnim, stepTo, veloc, dontTurn, velocityData);
            } else {
                objAnim->ClearAll();
            }
        }
    }

    // Clear any animations that weren't used (still have flag 0x80000000)
    for (CKObject **it = m_Animations.Begin(); it != m_Animations.End(); ++it) {
        RCKObjectAnimation *objAnim = (RCKObjectAnimation *) *it;
        if (objAnim) {
            if (objAnim->m_Flags & CK_OBJECTANIMATION_WARPER) {
                objAnim->ClearAll();
            }
            objAnim->m_Flags &= ~CK_OBJECTANIMATION_WARPER;
        }
    }

    // Set the length
    SetLength(length);

    // Return the computed step value
    return stepTo * animOut->m_MergeFactor;
}


// sub_100499C1 - Update root entity by walking up hierarchy from first animation's entity
void RCKKeyedAnimation::UpdateRootEntity() {
    m_RootAnimation = nullptr;
    m_RootEntity = nullptr;
    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);

    if (m_Animations.Size() == 0)
        return;

    // Get first animation's entity
    RCKObjectAnimation *firstAnim = (RCKObjectAnimation *) m_Animations[0];
    if (!firstAnim)
        return;

    CK3dEntity *entity = firstAnim->Get3dEntity();
    while (entity) {
        CK3dEntity *parent = entity->GetParent();
        if (parent) {
            // Check if parent's class ID is 40 (CKCID_CHARACTER)
            if (parent->GetClassID() == CKCID_CHARACTER) {
                m_RootEntity = (RCK3dEntity *) entity;
                return;
            }
        }
        // Check if parent is our character's root (cast m_Character to CK3dEntity*)
        if (parent == (CK3dEntity *) m_Character) {
            m_RootEntity = (RCK3dEntity *) entity;
            return;
        }
        entity = parent;
    }
}

// sub_1004A2D0 - Set parent keyed animation on object animation
// Note: This must be called from within RCKKeyedAnimation methods due to friend access
void RCKKeyedAnimation::SetParentKeyedAnimation(RCKObjectAnimation *objAnim, RCKKeyedAnimation *parent) {
    if (objAnim) {
        objAnim->m_ParentKeyedAnimation = parent;
    }
}

//=============================================================================
// Static Class Methods (for class registration)
//=============================================================================

CKSTRING RCKKeyedAnimation::GetClassName() {
    return (CKSTRING) "Keyed Animation";
}

int RCKKeyedAnimation::GetDependenciesCount(int mode) {
    return 0;
}

CKSTRING RCKKeyedAnimation::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCKKeyedAnimation::Register() {
    // Based on IDA analysis
    CKCLASSNOTIFYFROM(RCKKeyedAnimation, RCKObjectAnimation);

    // Register associated parameter GUID: {0x5BF0D34D, 0x19E69236} (same as Animation)
    CKPARAMETERFROMCLASS(RCKKeyedAnimation, CKPGUID_ANIMATION);
}

CKKeyedAnimation *RCKKeyedAnimation::CreateInstance(CKContext *Context) {
    // Object size is 0x54 (84 bytes)
    RCKKeyedAnimation *anim = new RCKKeyedAnimation(Context, nullptr);
    return reinterpret_cast<CKKeyedAnimation *>(anim);
}
