#include "RCKTargetCamera.h"

#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "CKDependencies.h"
#include "RCKCamera.h"
#include "RCK3dEntity.h"

// Static class ID definition
CK_CLASSID RCKTargetCamera::m_ClassID = CKCID_TARGETCAMERA;

/*************************************************
Summary: Constructor for RCKTargetCamera.
Purpose: Initializes target camera with no target.
Remarks:
- Calls base class RCKCamera constructor
- Sets m_Target to 0 (no target)

Implementation based on decompilation at 0x10043ae0.
*************************************************/
RCKTargetCamera::RCKTargetCamera(CKContext *Context, CKSTRING name)
    : RCKCamera(Context, name), m_Target(0) {}

/*************************************************
Summary: Destructor for RCKTargetCamera.
Purpose: Cleans up target camera resources.
Remarks:
- Clears target to properly update target's flags
- Base class destructor handles camera cleanup

Implementation based on decompilation at 0x10043b16.
*************************************************/
RCKTargetCamera::~RCKTargetCamera() {
    // Clear target to update flags properly
    SetTarget(nullptr);
}

//=============================================================================
// Target Methods
//=============================================================================

/*************************************************
Summary: Gets the target entity of this camera.
Return Value: Pointer to the target CK3dEntity, or NULL if no target.
Remarks:
- Uses m_Context to resolve the target ID to an object

Implementation based on decompilation at 0x10043b69.
*************************************************/
CK3dEntity *RCKTargetCamera::GetTarget() {
    return static_cast<CK3dEntity *>(m_Context->GetObject(m_Target));
}

/*************************************************
Summary: Sets the target entity of this camera.
Purpose: Sets an entity that the camera will look at.
Remarks:
- Clears flags on old target, sets flags on new target
- Uses object flags to manage target state:
  - Clears flag 0x200 and sets flag 0x2 on old target
  - Sets flag 0x200 and clears flag 0x2 on new target
- Stores target object ID

Implementation based on decompilation at 0x10043b89.
*************************************************/
void RCKTargetCamera::SetTarget(CK3dEntity *target) {
    // Don't allow setting self as target
    if (target && target->GetID() == GetID())
        return;

    // Get old target
    CK3dEntity *oldTarget = static_cast<CK3dEntity *>(m_Context->GetObject(m_Target));

    // Don't do anything if same target
    if (oldTarget == target)
        return;

    // Clear flags on old target
    if (oldTarget) {
        CKDWORD flags = oldTarget->GetFlags();
        flags &= ~CK_3DENTITY_TARGETCAMERA; // Clear "is target" flag
        flags |= CK_3DENTITY_FRAME;         // Set some other flag
        oldTarget->SetFlags(flags);
    }

    // Set flags on new target
    if (target) {
        CKDWORD flags = target->GetFlags();
        flags |= CK_3DENTITY_TARGETCAMERA; // Set "is target" flag
        flags &= ~CK_3DENTITY_FRAME;       // Clear some other flag
        target->SetFlags(flags);
        m_Target = target->GetID();
    } else {
        m_Target = 0;
    }
}

void RCKTargetCamera::AddToScene(CKScene *scene, CKBOOL dependencies) {
    // IDA: 0x10043C63
    if (!scene) {
        return;
    }

    RCK3dEntity::AddToScene(scene, dependencies);

    if (dependencies) {
        CK3dEntity *target = static_cast<CK3dEntity *>(m_Context->GetObject(m_Target));
        if (target) {
            target->AddToScene(scene, dependencies);
        }
    }
}

void RCKTargetCamera::RemoveFromScene(CKScene *scene, CKBOOL dependencies) {
    // IDA: 0x10043CCE
    if (!scene) {
        return;
    }

    RCK3dEntity::RemoveFromScene(scene, dependencies);

    if (dependencies) {
        CK3dEntity *target = static_cast<CK3dEntity *>(m_Context->GetObject(m_Target));
        if (target) {
            target->RemoveFromScene(scene, dependencies);
        }
    }
}

//=============================================================================
// CKObject Overrides
//=============================================================================

/*************************************************
Summary: Returns the class ID of this object.
Return Value: CKCID_TARGETCAMERA class identifier.

Implementation based on decompilation at 0x10043d39.
*************************************************/
CK_CLASSID RCKTargetCamera::GetClassID() {
    return m_ClassID;
}

/*************************************************
Summary: Checks and clears references to deleted objects.
Purpose: Called during object deletion to clean up references.
Remarks:
- Calls base class CheckPostDeletion first
- Clears m_Target if the target object no longer exists

Implementation based on decompilation at 0x10043d49.
*************************************************/
void RCKTargetCamera::CheckPostDeletion() {
    CKObject::CheckPostDeletion();

    // Clear target if the object no longer exists
    if (!m_Context->GetObject(m_Target)) {
        m_Target = 0;
    }
}

/*************************************************
Summary: Checks if an object is used by this target camera.
Return Value: TRUE if the object is the target, FALSE otherwise.
Remarks:
- Returns TRUE if the given object is the target
- Otherwise delegates to base class

Implementation based on decompilation at 0x10043db9.
*************************************************/
CKBOOL RCKTargetCamera::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    if (o && o->GetID() == m_Target)
        return TRUE;
    return RCK3dEntity::IsObjectUsed(o, cid);
}

/*************************************************
Summary: Returns the memory footprint of this object.
Return Value: Memory size in bytes.
Remarks:
- Adds 4 bytes for m_Target to base class size

Implementation based on decompilation at 0x10043d7f.
*************************************************/
int RCKTargetCamera::GetMemoryOccupation() {
    return RCKCamera::GetMemoryOccupation() + (sizeof(RCKTargetCamera) - sizeof(RCKCamera));
}

/*************************************************
Summary: Prepares dependencies for copy/save operations.
Purpose: Ensures target entity is included in dependencies.
Remarks:
- Calls base class PrepareDependencies first
- If class dependencies include target (bit 0), prepares target entity
- Finishes with FinishPrepareDependencies

Implementation based on decompilation at 0x1004404d.
*************************************************/
CKERROR RCKTargetCamera::PrepareDependencies(CKDependenciesContext &context) {
    CKERROR err = RCK3dEntity::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    // Check if target is included in dependencies (bit 0)
    if (context.GetClassDependencies(CKCID_TARGETCAMERA) & 1) {
        CKObject *targetObj = m_Context->GetObject(m_Target);
        if (targetObj) {
            targetObj->PrepareDependencies(context);
        }
    }

    return context.FinishPrepareDependencies(this, m_ClassID);
}

/*************************************************
Summary: Remaps object IDs after copy/paste operations.
Purpose: Updates target ID to point to copied object.
Remarks:
- Calls base class RemapDependencies first
- Remaps m_Target using context's Remap function

Implementation based on decompilation at 0x100440ca.
*************************************************/
CKERROR RCKTargetCamera::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = RCK3dEntity::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    // Remap target ID
    CKObject *targetObj = m_Context->GetObject(m_Target);
    CKObject *remapped = context.Remap(targetObj);
    if (remapped) {
        m_Target = remapped->GetID();
    }

    return CK_OK;
}

/*************************************************
Summary: Copies target camera data from another object.
Purpose: Deep copy including target reference.
Remarks:
- Calls base class Copy first
- Copies m_Target ID

Implementation based on decompilation at 0x10044107.
*************************************************/
CKERROR RCKTargetCamera::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = RCKCamera::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKTargetCamera *srcCamera = static_cast<RCKTargetCamera *>(&o);
    m_Target = srcCamera->m_Target;

    return CK_OK;
}

/*************************************************
Summary: PreSave method for RCKTargetCamera.
Purpose: Prepares target camera for saving by handling target object dependencies.
Remarks:
- Calls base class RCK3dEntity::PreSave() first to handle entity dependencies
- Saves the target object for later serialization
- Ensures target object is properly saved with the camera data

Implementation based on decompilation at 0x10043DD4:
- Simple implementation that only saves the target object
- Uses CKContext::GetObjectA to retrieve target object by ID
- Uses CKFile::SaveObject for dependency handling

Arguments:
- file: The file context for saving
- flags: Save flags controlling behavior and dependency handling
*************************************************/
void RCKTargetCamera::PreSave(CKFile *file, CKDWORD flags) {
    // Call base class PreSave first to handle entity dependencies
    RCK3dEntity::PreSave(file, flags);

    // Save target object
    CKObject *targetObject = m_Context->GetObject(m_Target);
    file->SaveObject(targetObject, flags);
}

/*************************************************
Summary: Save method for RCKTargetCamera.
Purpose: Saves target camera data to a state chunk including target object reference.
Remarks:
- Calls base class RCKCamera::Save() first to handle camera data
- Creates target camera-specific state chunk with identifier 0x10000000
- Saves target object reference for camera orientation
- Maintains backward compatibility with base camera serialization

Implementation based on decompilation at 0x10043E10:
- Uses chunk identifier 0x10000000 for target-specific data
- Retrieves target object using CKContext::GetObjectA
- Saves target object reference in the state chunk
- Handles both file and memory-based saving modes

Arguments:
- file: The file context for saving (may be NULL for memory operations)
- flags: Save flags controlling behavior and data inclusion
Return Value:
- CKStateChunk*: The created state chunk containing target camera data
*************************************************/
CKStateChunk *RCKTargetCamera::Save(CKFile *file, CKDWORD flags) {
    // Call base class save first to handle camera data
    CKStateChunk *baseChunk = RCKCamera::Save(file, flags);

    // Return early if no file context and not in specific save modes
    if (!file && !(flags & CK_STATESAVE_TCAMERAONLY)) {
        return baseChunk;
    }

    // Create target camera-specific state chunk
    CKStateChunk *targetCameraChunk = CreateCKStateChunk(CKCID_TARGETCAMERA, file);
    if (!targetCameraChunk) {
        return baseChunk;
    }

    targetCameraChunk->StartWrite();
    targetCameraChunk->AddChunkAndDelete(baseChunk);

    targetCameraChunk->WriteIdentifier(CK_STATESAVE_TCAMERATARGET);

    // Save target object reference
    CKObject *targetObject = m_Context->GetObject(m_Target);
    targetCameraChunk->WriteObject(targetObject);

    // Close chunk appropriately based on class type
    if (GetClassID() == CKCID_TARGETCAMERA) {
        targetCameraChunk->CloseChunk();
    } else {
        targetCameraChunk->UpdateDataSize();
    }

    return targetCameraChunk;
}

/*************************************************
Summary: Load method for RCKTargetCamera.
Purpose: Loads target camera data from a state chunk including target object reference.
Remarks:
- Calls base class RCKCamera::Load() first to handle camera data
- Loads target object ID from state chunk
- Stores target object ID for later object resolution
- Maintains backward compatibility with base camera deserialization

Implementation based on decompilation at 0x10043EBC:
- Uses chunk identifier 0x10000000 for target-specific data
- Reads target object ID using CKStateChunk::ReadObjectID
- Stores target ID for later object resolution
- Simple implementation focused on target reference restoration

Arguments:
- chunk: The state chunk containing target camera data
- file: The file context for loading (may be NULL)
Return Value:
- CKERROR: 0 for success, -1 for invalid chunk
*************************************************/
CKERROR RCKTargetCamera::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) {
        return CKERR_INVALIDPARAMETER;
    }

    // Call base class load first to handle camera data
    RCKCamera::Load(chunk, file);

    // Load target object ID
    if (chunk->SeekIdentifier(CK_STATESAVE_TCAMERATARGET)) {
        m_Target = chunk->ReadObjectID();
    }

    return CK_OK;
}

//=============================================================================
// Static Class Methods (for class registration)
// Based on IDA Pro analysis of original CK2_3D.dll
//=============================================================================

CKSTRING RCKTargetCamera::GetClassName() {
    return (CKSTRING) "Target Camera";
}

/*************************************************
Summary: Returns the number of dependencies for this class.
Purpose: Tells the system how many dependency strings exist.
Remarks:
- Mode 1 (CK_DEPENDENCIES_COPY): 1 dependency (Target)
- Mode 2 (CK_DEPENDENCIES_SAVE): 1 dependency (Target)
- Mode 3: 0 dependencies
- Mode 4 (CK_DEPENDENCIES_DELETE): 1 dependency (Target)
- Other modes: 0 dependencies

Implementation based on decompilation at 0x10043f12.
*************************************************/
int RCKTargetCamera::GetDependenciesCount(int mode) {
    switch (mode) {
    case 1: // CK_DEPENDENCIES_COPY
        return 1;
    case 2: // CK_DEPENDENCIES_SAVE
        return 1;
    case 3:
        return 0;
    case 4: // CK_DEPENDENCIES_DELETE
        return 1;
    default:
        return 0;
    }
}

/*************************************************
Summary: Returns the name of a dependency at index i.
Purpose: Provides human-readable dependency names for UI.
Remarks:
- Index 0 returns "Target" for valid modes
- Other indices return NULL

Implementation based on decompilation at 0x10043f70.
*************************************************/
CKSTRING RCKTargetCamera::GetDependencies(int i, int mode) {
    if (i == 0)
        return (CKSTRING) "Target";
    return nullptr;
}

void RCKTargetCamera::Register() {
    // Based on IDA decompilation
    CKCLASSNOTIFYFROMCID(RCKTargetCamera, CKCID_3DENTITY);

    // Register associated parameter GUID: CKPGUID_TARGETCAMERA
    CKPARAMETERFROMCLASS(RCKTargetCamera, CKPGUID_TARGETCAMERA);

    // Register default dependencies
    CKCLASSDEFAULTCOPYDEPENDENCIES(RCKTargetCamera, CK_DEPENDENCIES_COPY);
}

CKTargetCamera *RCKTargetCamera::CreateInstance(CKContext *Context) {
    RCKTargetCamera *cam = new RCKTargetCamera(Context, nullptr);
    return reinterpret_cast<CKTargetCamera *>(cam);
}
