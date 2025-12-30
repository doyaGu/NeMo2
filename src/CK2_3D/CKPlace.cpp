#include "RCKPlace.h"

#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "CKObject.h"
#include "CKBeObject.h"
#include "CK3dEntity.h"
#include "CKCamera.h"
#include "PlaceFitter.h"

// Static class ID
CK_CLASSID RCKPlace::m_ClassID = CKCID_PLACE;

/**
 * @brief RCKPlace constructor
 * @param Context The CKContext instance
 * @param name Optional name for the place
 *
 * IDA: 0x1003e240
 */
RCKPlace::RCKPlace(CKContext *Context, CKSTRING name) : RCK3dEntity(Context, name) {
    m_Level = 0;
    m_Camera = 0;
    m_Priority = 20000;
    m_ClippingRect.Clear();
}

/**
 * @brief RCKPlace destructor
 *
 * IDA: 0x1003e2e8
 */
RCKPlace::~RCKPlace() {
    // XSArray destructor clears m_Portals automatically
}

/**
 * @brief Get the class ID for RCKPlace
 * @return The class ID (CKCID_PLACE = 22)
 *
 * IDA: 0x1003e418
 */
CK_CLASSID RCKPlace::GetClassID() {
    return m_ClassID;
}

/**
 * @brief Returns the bounding box for this place
 * @param local Whether to get the local or world bounding box
 * @return Reference to the bounding box
 *
 * IDA: 0x1003e352 - Simply returns GetHierarchicalBox
 */
const VxBbox &RCKPlace::GetBoundingBox(CKBOOL local) {
    return GetHierarchicalBox(local);
}

/**
 * @brief Add this place to a scene
 * @param scene The scene to add to
 * @param dependencies Whether to add dependencies
 *
 * IDA: 0x1003e371 - Simple null check then base call
 */
void RCKPlace::AddToScene(CKScene *scene, CKBOOL dependencies) {
    if (scene) {
        RCK3dEntity::AddToScene(scene, dependencies);
    }
}

/**
 * @brief Remove this place from a scene
 * @param scene The scene to remove from
 * @param dependencies Whether to remove dependencies
 *
 * IDA: 0x1003e396 - Calls base, then recursively removes children
 */
void RCKPlace::RemoveFromScene(CKScene *scene, CKBOOL dependencies) {
    if (scene) {
        RCK3dEntity::RemoveFromScene(scene, dependencies);

        // Recursively remove all children from the scene
        int childCount = GetChildrenCount();
        for (int i = 0; i < childCount; ++i) {
            CK3dEntity *child = GetChild(i);
            if (child) {
                child->RemoveFromScene(scene, dependencies);
            }
        }
    }
}

/**
 * @brief Check and clean up portals before deletion
 *
 * IDA: 0x1003e428 - Removes portals referencing objects marked for deletion
 */
void RCKPlace::CheckPreDeletion() {
    RCK3dEntity::CheckPreDeletion();

    // Iterate through portals and remove entries with objects marked for deletion
    for (CKPortalEntry *it = m_Portals.Begin(); it < m_Portals.End();) {
        CKPortalEntry &entry = *it;

        // Check if place or portal is marked for deletion
        CKBOOL placeToBeDeleted = entry.place && reinterpret_cast<CKObject *>(entry.place)->IsToBeDeleted();
        CKBOOL portalToBeDeleted = entry.portal && reinterpret_cast<CKObject *>(entry.portal)->IsToBeDeleted();

        if (placeToBeDeleted || portalToBeDeleted) {
            it = m_Portals.Remove(it);
            // Don't increment i, check the new element at this index
        } else {
            ++it;
        }
    }
}

/**
 * @brief Check and clear invalid object references after deletion
 *
 * IDA: 0x1003e4a9 - Clears invalid m_Level and m_Camera
 */
void RCKPlace::CheckPostDeletion() {
    CKObject::CheckPostDeletion();

    // Clear auxiliary object reference if it no longer exists
    if (!m_Context->GetObject(m_Level)) {
        m_Level = 0;
    }

    // Clear camera reference if it no longer exists
    if (!m_Context->GetObject(m_Camera)) {
        m_Camera = 0;
    }
}

/**
 * @brief Get memory occupation for the place
 * @return Memory size in bytes
 *
 * IDA: 0x1003e508 - Base + 32 + portal array size
 */
int RCKPlace::GetMemoryOccupation() {
    // Base class memory + place-specific fields + portal array
    int size = RCK3dEntity::GetMemoryOccupation() + (sizeof(RCKPlace) - sizeof(RCK3dEntity));
    size += m_Portals.GetMemoryOccupation(FALSE);
    return size;
}

/**
 * @brief Check if an object is used by this place
 * @param o The object to check
 * @param cid The class ID of the object
 * @return TRUE if the object is used
 *
 * IDA: 0x1003e544 - Checks if CK3dEntity is in portals
 */
CKBOOL RCKPlace::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    // Check if it's a 3D entity
    if (CKIsChildClassOf(cid, CKCID_3DENTITY)) {
        // Check all portals
        for (int i = 0; i < m_Portals.Size(); ++i) {
            // Check if object is the place or the portal geometry
            if (reinterpret_cast<CKObject *>(m_Portals[i].place) == o ||
                static_cast<CKObject *>(m_Portals[i].portal) == o) {
                return TRUE;
            }
        }
    }
    return RCK3dEntity::IsObjectUsed(o, cid);
}

/**
 * @brief Pre-save processing
 * @param file The file being saved to
 * @param flags Save flags
 *
 * IDA: 0x1003e662 - Calls base and saves camera object
 */
void RCKPlace::PreSave(CKFile *file, CKDWORD flags) {
    RCK3dEntity::PreSave(file, flags);

    // Save the camera object reference
    CKObject *camera = m_Context->GetObject(m_Camera);
    file->SaveObject(camera, flags);
}

/**
 * @brief Save the place data to a state chunk
 * @param file The CKFile to save to (can be nullptr for standalone saving)
 * @param flags Save flags
 * @return A new CKStateChunk containing the saved data
 *
 * IDA: 0x1003e69e
 */
CKStateChunk *RCKPlace::Save(CKFile *file, CKDWORD flags) {
    // Call CKBeObject::Save (not RCK3dEntity) to get base chunk
    CKStateChunk *baseChunk = CKBeObject::Save(file, flags);

    // Create a new state chunk for place data
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_PLACE, file);
    chunk->StartWrite();

    // Add the base class chunk
    chunk->AddChunkAndDelete(baseChunk);

    // Save camera if flag is set
    if ((flags & CK_STATESAVE_PLACECAMERA) != 0) {
        chunk->WriteIdentifier(CK_STATESAVE_PLACECAMERA);
        CKObject *camera = m_Context->GetObject(m_Camera);
        chunk->WriteObject(camera);
    }

    // Save level if saving to file and flag is set
    if (file && (flags & CK_STATESAVE_PLACELEVEL) != 0) {
        chunk->WriteIdentifier(CK_STATESAVE_PLACELEVEL);
        CKObject *level = m_Context->GetObject(m_Level);
        chunk->WriteObject(level);
    }

    // Save portals if they exist and flag is set
    if (m_Portals.Size() > 0 && (flags & CK_STATESAVE_PLACEPORTALS) != 0) {
        chunk->WriteIdentifier(CK_STATESAVE_PLACEPORTALS);
        chunk->WriteInt(m_Portals.Size());

        // Save each portal entry (place, portal geometry)
        for (CKPortalEntry *it = m_Portals.Begin(); it < m_Portals.End(); ++it) {
            chunk->WriteObject(reinterpret_cast<CKObject *>(it->place));
            chunk->WriteObject(reinterpret_cast<CKObject *>(it->portal));
        }
    }

    // Close the chunk
    chunk->CloseChunk();

    return chunk;
}

/**
 * @brief Load place data from a state chunk
 * @param chunk The CKStateChunk containing the saved data
 * @param file The CKFile being loaded from
 * @return CKERROR indicating success or failure
 *
 * IDA: 0x1003e7f1 (0x1003e943)
 */
CKERROR RCKPlace::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Call CKBeObject::Load
    CKBeObject::Load(chunk, file);

    // Update moveable flags based on visibility
    CKDWORD moveableFlags = GetMoveableFlags();
    if (IsVisible())
        moveableFlags |= VX_MOVEABLE_VISIBLE;
    else
        moveableFlags &= ~VX_MOVEABLE_VISIBLE;

    // Update 2D entity flag based on hierarchical hide state
    if (IsHierarchicallyHide())
        moveableFlags |= VX_MOVEABLE_HIERARCHICALHIDE;
    else
        moveableFlags &= ~VX_MOVEABLE_HIERARCHICALHIDE;

    SetMoveableFlags(moveableFlags);

    // Only process file-specific data if loading from file
    if (!file)
        return CK_OK;

    // Load children objects (0x4000)
    if (chunk->SeekIdentifier(CK_STATESAVE_PLACEREFERENCES)) {
        XObjectArray children;
        children.Load(chunk);

        for (int i = 0; i < children.Size(); ++i) {
            CK3dEntity *child = reinterpret_cast<CK3dEntity *>(children.GetObject(m_Context, i));
            if (child) {
                // Only add if child's reference place is not this place
                // and parent is not already in the children array
                if (child->GetReferencePlace() != reinterpret_cast<CKPlace *>(this)) {
                    CKObject *parent = child->GetParent();
                    if (!children.FindObject(parent)) {
                        AddChild(child, TRUE);
                    }
                }
            }
        }
    }

    // Load camera (0x2000)
    if (chunk->SeekIdentifier(CK_STATESAVE_PLACECAMERA)) {
        m_Camera = chunk->ReadObjectID();
    }

    // Load level reference (0x8000)
    if (chunk->SeekIdentifier(CK_STATESAVE_PLACELEVEL)) {
        m_Level = chunk->ReadObjectID();
    }

    // Load portals (0x1000)
    // NOTE: Does NOT add bidirectionally - the file stores both directions
    if (chunk->SeekIdentifier(CK_STATESAVE_PLACEPORTALS)) {
        int portalCount = chunk->ReadInt();

        for (int j = 0; j < portalCount; ++j) {
            CKPortalEntry entry;
            entry.place = reinterpret_cast<CKPlace *>(chunk->ReadObject(m_Context));
            entry.portal = reinterpret_cast<CK3dEntity *>(chunk->ReadObject(m_Context));

            if (entry.place) {
                m_Portals.PushBack(entry);
            }
        }
    }

    return CK_OK;
}

/**
 * @brief Copy place data to another object
 * @param o The target object
 * @param context Dependencies context
 * @return CKERROR indicating success or failure
 */
CKERROR RCKPlace::Copy(CKObject &o, CKDependenciesContext &context) {
    // Base class copy
    RCK3dEntity::Copy(o, context);

    CKDWORD deps = context.GetClassDependencies(CKCID_PLACE);

    RCKPlace *src = reinterpret_cast<RCKPlace *>(&o);
    m_Level = src->m_Level;
    m_Camera = src->m_Camera;
    if (deps & CK_DEPENDENCIES_COPY) {
        m_Portals = src->m_Portals;
    }

    return CK_OK;
}

// Static class registration methods
CKSTRING RCKPlace::GetClassName() {
    return (CKSTRING) "Place";
}

int RCKPlace::GetDependenciesCount(int mode) {
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

CKSTRING RCKPlace::GetDependencies(int i, int mode) {
    if (i != 0)
        return nullptr;
    return (CKSTRING) "Portals";
}

void RCKPlace::Register() {
    // Register notifications for classes that can affect this place
    CKCLASSNOTIFYFROMCID(RCKPlace, CKCID_3DENTITY);
    CKCLASSNOTIFYFROMCID(RCKPlace, CKCID_LEVEL);
    CKCLASSNOTIFYFROMCID(RCKPlace, CKCID_CAMERA);
    CKPARAMETERFROMCLASS(RCKPlace, CKPGUID_PLACE);
}

CKPlace *RCKPlace::CreateInstance(CKContext *Context) {
    return reinterpret_cast<CKPlace *>(new RCKPlace(Context));
}

//-----------------------------------------------------------------------------
// CKPlace virtual method implementations
//-----------------------------------------------------------------------------

/**
 * @brief Get the default camera for this place
 * @return Pointer to the default camera, or nullptr if not set
 *
 * IDA: 0x1003e610
 */
CKCamera *RCKPlace::GetDefaultCamera() {
    return reinterpret_cast<CKCamera *>(m_Context->GetObject(m_Camera));
}

/**
 * @brief Set the default camera for this place
 * @param cam Pointer to the default camera to set, or nullptr to clear
 *
 * IDA: 0x1003e62d
 */
void RCKPlace::SetDefaultCamera(CKCamera *cam) {
    m_Camera = cam ? cam->GetID() : 0;
}

/**
 * @brief Add a portal connection between this place and another
 * @param place The destination place
 * @param portal The portal geometry (can be nullptr for "visible everywhere")
 *
 * IDA: 0x1003ea2a
 * 
 * Logic:
 * 1. If portal is provided, it must have VX_PORTAL flag (0x80000)
 * 2. First search for (place, nullptr) - if found, don't add (already visible everywhere)
 * 3. Then search for (place, portal) - if found, don't add duplicate
 * 4. Add to both this place and the target place (bidirectional)
 */
void RCKPlace::AddPortal(CKPlace *place, CK3dEntity *portal) {
    if (!place) {
        return;
    }

    // Portal must have VX_PORTAL flag if specified
    if (portal && (portal->GetFlags() & CK_3DENTITY_PORTAL) == 0) {
        return;
    }

    CKPortalEntry entry;
    entry.place = place;
    entry.portal = nullptr;

    CKPortalEntry *it = m_Portals.Find(entry);
    if (it == m_Portals.End()) {
        entry.portal = portal;
        it = m_Portals.Find(entry);
        if (it == m_Portals.End()) {
            // Not found, add the portal entry
            m_Portals.PushBack(entry);
            entry.place = reinterpret_cast<CKPlace *>(this);
            m_Portals.PushBack(entry);
        }
    }
}

/**
 * @brief Remove a portal connection between this place and another
 * @param place The destination place
 * @param portal The portal geometry
 *
 * IDA: 0x1003eaec
 * Removes from both this place and the target place
 */
void RCKPlace::RemovePortal(CKPlace *place, CK3dEntity *portal) {
    CKPortalEntry entry;
    entry.place = place;
    entry.portal = portal;
    m_Portals.Remove(entry);
    entry.place = reinterpret_cast<CKPlace *>(this);
    m_Portals.Remove(entry);
}

/**
 * @brief Get the number of portals in this place
 * @return Number of portal entries
 *
 * IDA: 0x1003eb31
 */
int RCKPlace::GetPortalCount() {
    return m_Portals.Size();
}

/**
 * @brief Get information about a specific portal
 * @param i Index of the portal
 * @param portal Output pointer to receive the portal geometry
 * @return The destination place for the portal
 *
 * IDA: 0x1003eb4a
 */
CKPlace *RCKPlace::GetPortal(int i, CK3dEntity **portal) {
    CKPortalEntry *entry = m_Portals.At(i);
    if (!entry) {
        return nullptr;
    }

    if (portal) {
        *portal = entry->portal;
    }

    return entry->place;
}

/**
 * @brief Get the viewport clipping rectangle for this place
 * @return Reference to the clipping rectangle
 *
 * This is used by the portal manager to restrict rendering
 */
VxRect &RCKPlace::ViewportClip() {
    return m_ClippingRect;
}

/**
 * @brief Compute the best fitting bounding box matrix between two places
 * @param p2 The second place
 * @param BBoxMatrix Output matrix for the bounding box
 * @return TRUE if successful
 *
 * IDA: 0x1003eb8e - Uses NearestPointGrid/PlaceFitter classes
 */
CKBOOL RCKPlace::ComputeBestFitBBox(CKPlace *p2, VxMatrix &BBoxMatrix) {
    if (!p2) return FALSE;

    PlaceFitter fitter;
    return fitter.ComputeBestFitBBox(reinterpret_cast<CK3dEntity *>(this),
                                     reinterpret_cast<CK3dEntity *>(p2),
                                     BBoxMatrix);
}
