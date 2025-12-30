#include "RCKTargetLight.h"

#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "RCKLight.h"
#include "RCK3dEntity.h"

// Static class ID definition
CK_CLASSID RCKTargetLight::m_ClassID = CKCID_TARGETLIGHT;

/*************************************************
Summary: Constructor for RCKTargetLight.
Purpose: Initializes target light with no target.
Remarks:
- Calls base class RCKLight constructor
- Sets m_Target to 0 (no target)

Implementation based on decompilation at 0x10044180.
*************************************************/
RCKTargetLight::RCKTargetLight(CKContext *Context, CKSTRING name)
    : RCKLight(Context, name), m_Target(0) {}

/*************************************************
Summary: Destructor for RCKTargetLight.
Purpose: Cleans up target light resources.
Remarks:
- Must call SetTarget(NULL) to clear flags on old target
- Base class destructor handles light cleanup

Implementation based on decompilation at 0x100441B6.
*************************************************/
RCKTargetLight::~RCKTargetLight() {
    // Clear target and its flags before base destructor runs
    SetTarget(nullptr);
}

//=============================================================================
// Target Methods
//=============================================================================

/*************************************************
Summary: Gets the target entity of this light.
Return Value: Pointer to the target CK3dEntity, or NULL if no target.
Remarks:
- Uses m_Context to resolve the target ID to an object

Implementation based on decompilation at 0x10044209.
*************************************************/
CK3dEntity *RCKTargetLight::GetTarget() {
    return static_cast<CK3dEntity *>(m_Context->GetObject(m_Target));
}

/*************************************************
Summary: Sets the target entity of this light.
Purpose: Sets an entity that the light will point toward.
Remarks:
- Clears flags on old target, sets flags on new target
- Uses object flags to manage target state:
  - Clears flag 0x100 and sets flag 0x2 on old target
  - Sets flag 0x100 and clears flag 0x2 on new target
- Stores target object ID

Implementation based on decompilation at 0x10044229.
*************************************************/
void RCKTargetLight::SetTarget(CK3dEntity *target) {
    // Don't allow setting self as target (compares pointers)
    if (target == reinterpret_cast<CK3dEntity *>(this))
        return;

    // Clear flags on old target
    CK3dEntity *oldTarget = static_cast<CK3dEntity *>(m_Context->GetObject(m_Target));
    if (oldTarget) {
        CKDWORD flags = oldTarget->GetFlags();
        flags &= ~0x100; // Clear "is target" flag (BYTE1 &= ~1)
        flags |= 0x2;    // Set flag 0x2
        oldTarget->SetFlags(flags);
    }

    // Set flags on new target
    if (target) {
        CKDWORD flags = target->GetFlags();
        flags |= 0x100; // Set "is target" flag (BYTE1 |= 1)
        flags &= ~0x2;  // Clear flag 0x2 (& 0xFFFFFFFD)
        target->SetFlags(flags);
        m_Target = target->GetID();
    } else {
        m_Target = 0;
    }
}

//=============================================================================
// CKObject Overrides
//=============================================================================

/*************************************************
Summary: Returns the class ID of this object.
Return Value: CKCID_TARGETLIGHT class identifier.

Implementation based on decompilation at 0x100443cc.
*************************************************/
CK_CLASSID RCKTargetLight::GetClassID() {
    return m_ClassID;
}

/*************************************************
Summary: Returns the memory footprint of this object.
Return Value: Memory size in bytes.
Remarks:
- Adds 4 bytes for m_Target to base class size

Implementation based on decompilation at 0x10044412.
*************************************************/
int RCKTargetLight::GetMemoryOccupation() {
    return RCKLight::GetMemoryOccupation() + (sizeof(RCKTargetLight) - sizeof(RCKLight));
}

/*************************************************
Summary: Copies target light data from another object.
Purpose: Deep copy including target reference.
Remarks:
- Calls base class Copy first
- Copies m_Target ID

Implementation based on decompilation at 0x1004479a.
*************************************************/
CKERROR RCKTargetLight::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = RCKLight::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKTargetLight *srcLight = static_cast<RCKTargetLight *>(&o);
    m_Target = srcLight->m_Target;

    return CK_OK;
}

/*************************************************
Summary: Checks if the specified object is used by this target light.
Purpose: Used by the engine to determine object dependencies.
Remarks:
- First checks if the object is the target entity
- Then delegates to base class for other checks

Implementation based on decompilation at 0x10044430.
*************************************************/
CKBOOL RCKTargetLight::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    // Check if the object is our target
    if (obj && obj->GetID() == m_Target)
        return TRUE;

    // Delegate to base class for other checks
    return RCK3dEntity::IsObjectUsed(obj, cid);
}

/*************************************************
Summary: Prepares dependencies for this target light.
Purpose: Called during copy/save operations to gather dependent objects.
Remarks:
- Calls base class PrepareDependencies first
- If class dependencies include CKCID_TARGETLIGHT, prepares target object
- Finishes by calling FinishPrepareDependencies

Implementation based on decompilation at 0x100446E0.
*************************************************/
CKERROR RCKTargetLight::PrepareDependencies(CKDependenciesContext &context) {
    // Call base class first
    CKERROR result = RCK3dEntity::PrepareDependencies(context);
    if (result != CK_OK)
        return result;

    // Check if we need to prepare target dependencies
    if (context.GetClassDependencies(CKCID_TARGETLIGHT) & 1) {
        CKObject *targetObject = m_Context->GetObject(m_Target);
        if (targetObject) {
            targetObject->PrepareDependencies(context);
        }
    }

    // Finish preparing dependencies for this class
    return context.FinishPrepareDependencies(this, m_ClassID);
}

/*************************************************
Summary: Remaps dependencies after a copy operation.
Purpose: Updates internal object references to point to copied objects.
Remarks:
- Calls base class RemapDependencies first
- Remaps m_Target to the new copied target object

Implementation based on decompilation at 0x1004475D.
*************************************************/
CKERROR RCKTargetLight::RemapDependencies(CKDependenciesContext &context) {
    // Call base class first
    CKERROR result = RCK3dEntity::RemapDependencies(context);
    if (result != CK_OK)
        return result;

    // Remap target ID
    if (m_Target) {
        CKObject *remapped = context.Remap(m_Context->GetObject(m_Target));
        m_Target = remapped ? remapped->GetID() : 0;
    }

    return CK_OK;
}

/*************************************************
Summary: PreSave method for RCKTargetLight.
Purpose: Prepares target light for saving by handling target object dependencies.
Remarks:
- Calls base class RCK3dEntity::PreSave() first to handle entity dependencies
- Saves the target object for later serialization
- Ensures target object is properly saved with the light data

Implementation based on decompilation at 0x10044467:
- Simple implementation that only saves the target object
- Uses CKContext::GetObjectA to retrieve target object by ID
- Uses CKFile::SaveObject for dependency handling

Arguments:
- file: The file context for saving
- flags: Save flags controlling behavior and dependency handling
*************************************************/
void RCKTargetLight::PreSave(CKFile *file, CKDWORD flags) {
    // Call base class PreSave first to handle entity dependencies
    RCK3dEntity::PreSave(file, flags);

    // Save target object
    CKObject *targetObject = m_Context->GetObject(m_Target);
    file->SaveObject(targetObject, flags);
}

/*************************************************
Summary: Save method for RCKTargetLight.
Purpose: Saves target light data to a state chunk including target object reference.
Remarks:
- Calls base class RCKLight::Save() first to handle light data
- Creates target light-specific state chunk with identifier 0x80000000
- Saves target object reference for light orientation
- Maintains backward compatibility with base light serialization

Implementation based on decompilation at 0x100444A3:
- Uses chunk identifier 0x80000000 for target-specific data
- Retrieves target object using CKContext::GetObjectA
- Saves target object reference in the state chunk
- Handles both file and memory-based saving modes

Arguments:
- file: The file context for saving (may be NULL for memory operations)
- flags: Save flags controlling behavior and data inclusion
Return Value:
- CKStateChunk*: The created state chunk containing target light data
*************************************************/
CKStateChunk *RCKTargetLight::Save(CKFile *file, CKDWORD flags) {
    // Call base class save first to handle light data
    CKStateChunk *baseChunk = RCKLight::Save(file, flags);

    // Return early if no file context and not in specific save modes
    if (!file && (flags & 0xF0000000) == 0) {
        return baseChunk;
    }

    // Create target light-specific state chunk
    CKStateChunk *targetLightChunk = CreateCKStateChunk(CKCID_TARGETLIGHT, file);
    if (!targetLightChunk) {
        return baseChunk;
    }

    targetLightChunk->StartWrite();
    targetLightChunk->AddChunkAndDelete(baseChunk);

    // Write target-specific data with identifier 0x80000000
    targetLightChunk->WriteIdentifier(CK_STATESAVE_TLIGHTTARGET);

    // Save target object reference
    CKObject *targetObject = m_Context->GetObject(m_Target);
    targetLightChunk->WriteObject(targetObject);

    // Close chunk appropriately based on class type
    if (GetClassID() == CKCID_TARGETLIGHT) {
        targetLightChunk->CloseChunk();
    } else {
        targetLightChunk->UpdateDataSize();
    }

    return targetLightChunk;
}

/*************************************************
Summary: Load method for RCKTargetLight.
Purpose: Loads target light data from a state chunk including target object reference.
Remarks:
- Calls base class RCKLight::Load() first to handle light data
- Loads target object ID from state chunk
- Stores target object ID for later object resolution
- Maintains backward compatibility with base light deserialization

Implementation based on decompilation at 0x1004454F:
- Uses chunk identifier 0x80000000 for target-specific data
- Reads target object ID using CKStateChunk::ReadObjectID
- Stores target ID for later object resolution
- Simple implementation focused on target reference restoration

Arguments:
- chunk: The state chunk containing target light data
- file: The file context for loading (may be NULL)
Return Value:
- CKERROR: 0 for success, -1 for invalid chunk
*************************************************/
CKERROR RCKTargetLight::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) {
        return CKERR_INVALIDPARAMETER;
    }

    // Call base class load first to handle light data
    RCKLight::Load(chunk, file);

    // Load target object ID
    if (chunk->SeekIdentifier(CK_STATESAVE_TLIGHTTARGET)) {
        m_Target = chunk->ReadObjectID();
    }

    return CK_OK;
}

//=============================================================================
// Static Class Methods (for class registration)
// Based on IDA Pro analysis of original CK2_3D.dll
//=============================================================================

CKSTRING RCKTargetLight::GetClassName() {
    return (CKSTRING) "Target Light";
}

/*************************************************
Summary: Returns the number of dependencies for this class.
Purpose: Used by the engine to query class-level dependency information.
Remarks:
- mode 1 (CK_DEPENDENCIES_COPY): 1
- mode 2 (CK_DEPENDENCIES_DELETE): 1
- mode 3 (CK_DEPENDENCIES_REPLACE): 0
- mode 4 (CK_DEPENDENCIES_SAVE): 1
- default: 0

Implementation based on decompilation at 0x100445B0.
*************************************************/
int RCKTargetLight::GetDependenciesCount(int mode) {
    switch (mode) {
    case 1: // CK_DEPENDENCIES_COPY
        return 1;
    case 2: // CK_DEPENDENCIES_DELETE
        return 1;
    case 3: // CK_DEPENDENCIES_REPLACE
        return 0;
    case 4: // CK_DEPENDENCIES_SAVE
        return 1;
    default:
        return 0;
    }
}

/*************************************************
Summary: Returns the name of a dependency.
Purpose: Used by the engine to get dependency names for UI or debugging.
Remarks:
- Returns "Target" for index 0
- Returns nullptr for other indices

Implementation based on decompilation at 0x10044610.
*************************************************/
CKSTRING RCKTargetLight::GetDependencies(int i, int mode) {
    if (i == 0)
        return (CKSTRING) "Target";
    return nullptr;
}

void RCKTargetLight::Register() {
    // Based on IDA decompilation
    CKCLASSNOTIFYFROMCID(RCKTargetLight, CKCID_3DENTITY);

    // Register associated parameter GUID: CKPGUID_TARGETLIGHT
    CKPARAMETERFROMCLASS(RCKTargetLight, CKPGUID_TARGETLIGHT);

    // Register default dependencies
    CKCLASSDEFAULTCOPYDEPENDENCIES(RCKTargetLight, CK_DEPENDENCIES_COPY);
}

CKTargetLight *RCKTargetLight::CreateInstance(CKContext *Context) {
    RCKTargetLight *light = new RCKTargetLight(Context, nullptr);
    return reinterpret_cast<CKTargetLight *>(light);
}
