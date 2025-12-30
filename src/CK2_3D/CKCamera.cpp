#include "RCKCamera.h"

#include "CKStateChunk.h"
#include "CKFile.h"

// Static class ID definition
CK_CLASSID RCKCamera::m_ClassID = CKCID_CAMERA;

/*************************************************
Summary: Constructor for RCKCamera.
Purpose: Initializes camera with default values.
Remarks:
- Sets default FOV to 0.5 (approximately 30 degrees)
- Sets perspective projection (type = 1)
- Sets orthographic zoom to 1.0
- Sets aspect ratio to 4:3
- Sets front plane to 1.0, back plane to 4000.0
- Marks camera as modified (flag CK_OBJECT_UPTODATE)

Implementation based on decompilation at 0x1000eee0.
*************************************************/
RCKCamera::RCKCamera(CKContext *Context, CKSTRING name)
    : RCK3dEntity(Context, name) {
    m_Fov = 0.5f;
    m_ProjectionType = CK_PERSPECTIVEPROJECTION;
    m_OrthographicZoom = 1.0f;
    m_Width = 4;
    m_Height = 3;
    m_FrontPlane = 1.0f;
    m_BackPlane = 4000.0f;

    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
}

/*************************************************
Summary: Destructor for RCKCamera.
Purpose: Cleans up camera resources.
Remarks:
- Base class destructor handles entity cleanup
*************************************************/
RCKCamera::~RCKCamera() {
    // Base class destructor handles cleanup
}

//=============================================================================
// Clipping Plane Methods
//=============================================================================

/*************************************************
Summary: Gets the front clipping plane distance.
Return Value: Front clipping plane distance.

Implementation based on decompilation at 0x1000f053.
*************************************************/
float RCKCamera::GetFrontPlane() {
    return m_FrontPlane;
}

/*************************************************
Summary: Sets the front clipping plane distance.
Purpose: Sets the distance below which nothing is seen.
Remarks:
- Marks camera as modified (flag CK_OBJECT_UPTODATE)

Implementation based on decompilation at 0x1000f067.
*************************************************/
void RCKCamera::SetFrontPlane(float front) {
    m_FrontPlane = front;
    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
}

/*************************************************
Summary: Gets the back clipping plane distance.
Return Value: Back clipping plane distance.

Implementation based on decompilation at 0x1000f08f.
*************************************************/
float RCKCamera::GetBackPlane() {
    return m_BackPlane;
}

/*************************************************
Summary: Sets the back clipping plane distance.
Purpose: Sets the distance beyond which nothing is seen.
Remarks:
- Marks camera as modified (flag CK_OBJECT_UPTODATE)

Implementation based on decompilation at 0x1000f0a3.
*************************************************/
void RCKCamera::SetBackPlane(float back) {
    m_BackPlane = back;
    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
}

//=============================================================================
// Field of View Methods
//=============================================================================

/*************************************************
Summary: Gets the field of view.
Return Value: Field of view in radians.

Implementation based on decompilation at 0x1000f0cb.
*************************************************/
float RCKCamera::GetFov() {
    return m_Fov;
}

/*************************************************
Summary: Sets the field of view.
Purpose: Sets the angle of view in radians.
Remarks:
- Marks camera as modified (flag CK_OBJECT_UPTODATE)

Implementation based on decompilation at 0x1000f0df.
*************************************************/
void RCKCamera::SetFov(float fov) {
    m_Fov = fov;
    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
}

//=============================================================================
// Projection Type Methods
//=============================================================================

/*************************************************
Summary: Gets the projection type.
Return Value: CK_PERSPECTIVEPROJECTION or CK_ORTHOGRAPHICPROJECTION.

Implementation based on decompilation at 0x1000f107.
*************************************************/
int RCKCamera::GetProjectionType() {
    return m_ProjectionType;
}

/*************************************************
Summary: Sets the projection type.
Purpose: Sets perspective or orthographic projection.
Remarks:
- Marks camera as modified (flag CK_OBJECT_UPTODATE)

Implementation based on decompilation at 0x1000f11b.
*************************************************/
void RCKCamera::SetProjectionType(int proj) {
    m_ProjectionType = proj;
    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
}

//=============================================================================
// Orthographic Zoom Methods
//=============================================================================

/*************************************************
Summary: Gets the orthographic zoom value.
Return Value: Zoom factor for orthographic projection.

Implementation based on decompilation at 0x1000f16b.
*************************************************/
float RCKCamera::GetOrthographicZoom() {
    return m_OrthographicZoom;
}

/*************************************************
Summary: Sets the orthographic zoom value.
Purpose: Sets zoom factor for orthographic projection.
Remarks:
- Marks camera as modified (flag CK_OBJECT_UPTODATE)

Implementation based on decompilation at 0x1000f143.
*************************************************/
void RCKCamera::SetOrthographicZoom(float zoom) {
    m_OrthographicZoom = zoom;
    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
}

//=============================================================================
// Aspect Ratio Methods
//=============================================================================

/*************************************************
Summary: Sets the aspect ratio.
Purpose: Sets width and height for viewport sizing.
Remarks:
- Marks camera as modified (flag CK_OBJECT_UPTODATE)

Implementation based on decompilation at 0x1000f17f.
*************************************************/
void RCKCamera::SetAspectRatio(int width, int height) {
    m_Width = width;
    m_Height = height;
    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
}

/*************************************************
Summary: Gets the aspect ratio.
Purpose: Returns width and height for viewport sizing.

Implementation based on decompilation at 0x1000f1b3.
*************************************************/
void RCKCamera::GetAspectRatio(int &width, int &height) {
    width = m_Width;
    height = m_Height;
}

//=============================================================================
// Projection Matrix Methods
//=============================================================================

/*************************************************
Summary: Computes the projection matrix.
Purpose: Calculates perspective or orthographic projection matrix.
Remarks:
- Uses FOV, aspect ratio, and clipping planes
- Checks projection type bit 0 to determine perspective vs orthographic

Implementation based on decompilation at 0x1000efbc.
*************************************************/
void RCKCamera::ComputeProjectionMatrix(VxMatrix &mat) {
    float aspect = (float) m_Width / (float) m_Height;

    if (m_ProjectionType & CK_PERSPECTIVEPROJECTION) {
        // Perspective projection
        mat.Perspective(m_Fov, aspect, m_FrontPlane, m_BackPlane);
    } else {
        // Orthographic projection
        mat.Orthographic(m_OrthographicZoom, aspect, m_FrontPlane, m_BackPlane);
    }
}

//=============================================================================
// Roll Methods
//=============================================================================

/*************************************************
Summary: Resets the camera roll to vertical.
Purpose: Aligns the camera's up axis with world up.
Remarks:
- Gets current orientation vectors
- Cross product with world up to get new right vector
- Reconstructs up vector from right and direction
- Handles near-vertical looking case

Implementation based on decompilation at 0x1000f26c.
*************************************************/
void RCKCamera::ResetRoll() {
    VxVector dir, up, right;
    GetOrientation(&dir, &up, &right, nullptr);

    const VxVector zero(0.0f, 0.0f, 0.0f);
    right = CrossProduct(zero, dir);

    // Check if looking nearly straight up/down
    if (right.Magnitude() < 0.1f) {
        const VxVector fallbackAxis(0.0f, 0.0f, 1.0f);
        right = CrossProduct(fallbackAxis, dir);
    }

    // New up vector = dir cross right
    up = CrossProduct(dir, right);

    up.Normalize();
    right.Normalize();

    SetOrientation(&dir, &up, &right, nullptr, FALSE);
}

/*************************************************
Summary: Rolls the camera by an angle.
Purpose: Rotates camera around its Z (look) axis.
Remarks:
- Creates rotation matrix around Z axis
- Multiplies with current local matrix
- Sets result as new local matrix

Implementation based on decompilation at 0x1000f1f6.
*************************************************/
void RCKCamera::Roll(float angle) {
    VxVector axis(0.0f, 0.0f, 1.0f);
    VxMatrix rotation;
    Vx3DMatrixFromRotation(rotation, axis, angle);

    VxMatrix result;
    Vx3DMultiplyMatrix(result, m_LocalMatrix, rotation);

    SetLocalMatrix(result, FALSE);
}

//=============================================================================
// Target Methods (for CKCamera base class - no target support)
//=============================================================================

/*************************************************
Summary: Gets the target of the camera.
Return Value: Always NULL for CKCamera (targets only supported in CKTargetCamera).

Implementation based on decompilation at 0x1000f1dc.
*************************************************/
CK3dEntity *RCKCamera::GetTarget() {
    return nullptr;
}

/*************************************************
Summary: Sets the target of the camera.
Remarks: Does nothing for CKCamera (targets only supported in CKTargetCamera).

Implementation based on decompilation at 0x1000f1e9.
*************************************************/
void RCKCamera::SetTarget(CK3dEntity *target) {
    // CKCamera does not support targets - this is for CKTargetCamera
}

//=============================================================================
// CKObject Overrides
//=============================================================================

/*************************************************
Summary: Returns the class ID of this object.
Return Value: CKCID_CAMERA class identifier.

Implementation based on decompilation at 0x1000f65f.
*************************************************/
CK_CLASSID RCKCamera::GetClassID() {
    return m_ClassID;
}

/*************************************************
Summary: Returns the memory footprint of this object.
Return Value: Memory size in bytes.
Remarks:
- Adds 28 bytes for camera-specific data to base class size

Implementation based on decompilation at 0x1000f380.
*************************************************/
int RCKCamera::GetMemoryOccupation() {
    return RCK3dEntity::GetMemoryOccupation() + (sizeof(RCKCamera) - sizeof(RCK3dEntity));
}

/*************************************************
Summary: Copies camera data from another camera object.
Purpose: Deep copy of all camera properties.
Remarks:
- Calls base class Copy first
- Copies all camera-specific fields
- Marks camera as modified

Implementation based on decompilation at 0x1000f72e.
*************************************************/
CKERROR RCKCamera::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = RCK3dEntity::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKCamera *srcCamera = static_cast<RCKCamera *>(&o);

    m_Fov = srcCamera->m_Fov;
    m_FrontPlane = srcCamera->m_FrontPlane;
    m_BackPlane = srcCamera->m_BackPlane;
    m_ProjectionType = srcCamera->m_ProjectionType;
    m_OrthographicZoom = srcCamera->m_OrthographicZoom;
    m_Width = srcCamera->m_Width;
    m_Height = srcCamera->m_Height;

    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);

    return CK_OK;
}

/*************************************************
Summary: Save method for RCKCamera.
Purpose: Saves camera data to a state chunk including projection type, FOV, planes, and viewport dimensions.
Remarks:
- Calls base class RCK3dEntity::Save() first to handle entity data
- Creates camera-specific state chunk with identifier 0xFC00000
- Saves projection type (perspective/orthographic), field of view, orthographic zoom
- Packs width and height into single DWORD for efficient storage
- Saves front and back clipping planes
- Handles chunk closing based on class type for proper serialization

Implementation based on decompilation at 0x1000F396:
- Uses chunk identifier 0xFC00000 for camera-specific data
- Packs width/height as (height << 16) | width for compact storage
- Supports both file and memory-based saving modes
- Maintains backward compatibility with data versioning

Arguments:
- file: The file context for saving (may be NULL for memory operations)
- flags: Save flags controlling behavior and data inclusion
Return Value:
- CKStateChunk*: The created state chunk containing camera data
*************************************************/
CKStateChunk *RCKCamera::Save(CKFile *file, CKDWORD flags) {
    // Call base class save first to handle entity data
    CKStateChunk *baseChunk = RCK3dEntity::Save(file, flags);

    // Return early if no file context and not in specific save modes
    if (!file && (flags & CK_STATESAVE_CAMERAONLY) == 0) {
        return baseChunk;
    }

    // Create camera-specific state chunk
    CKStateChunk *cameraChunk = CreateCKStateChunk(CKCID_CAMERA, file);
    if (!cameraChunk) {
        return baseChunk;
    }

    cameraChunk->StartWrite();
    cameraChunk->AddChunkAndDelete(baseChunk);

    // Write camera-specific data with identifier CK_STATESAVE_CAMERAONLY
    cameraChunk->WriteIdentifier(CK_STATESAVE_CAMERAONLY);
    cameraChunk->WriteDword(m_ProjectionType);
    cameraChunk->WriteFloat(m_Fov);
    cameraChunk->WriteFloat(m_OrthographicZoom);

    // Pack width and height into single DWORD for efficient storage.
    // IDA: (SLOWORD(m_Height) << 16) | SLOWORD(m_Width)
    CKDWORD packedDimensions = (static_cast<CKDWORD>(static_cast<CKWORD>(m_Height)) << 16) | static_cast<CKDWORD>(static_cast<CKWORD>(m_Width));
    cameraChunk->WriteDword(packedDimensions);

    // Write clipping planes
    cameraChunk->WriteFloat(m_FrontPlane);
    cameraChunk->WriteFloat(m_BackPlane);

    // Close chunk appropriately based on class type
    if (GetClassID() == CKCID_CAMERA) {
        cameraChunk->CloseChunk();
    } else {
        cameraChunk->UpdateDataSize();
    }

    return cameraChunk;
}

/*************************************************
Summary: Load method for RCKCamera.
Purpose: Loads camera data from a state chunk including projection settings and viewport configuration.
Remarks:
- Calls base class RCK3dEntity::Load() first to handle entity data
- Supports both legacy format (data version < 5) and current format
- Legacy format uses separate identifiers for each camera property
- Current format uses single identifier CK_STATESAVE_CAMERAONLY with packed data
- Unpacks width/height from single DWORD storage
- Sets object flags to mark camera as modified after loading

Implementation based on decompilation at 0x1000F4A6:
- Legacy format: separate chunks for FOV (0x400000), projection type (0x800000),
  orthographic zoom (0x1000000), dimensions (0x2000000), planes (0x4000000)
- Current format: single chunk CK_STATESAVE_CAMERAONLY with all camera data
- Handles width/height unpacking from packed DWORD
- Sets object flag CK_OBJECT_UPTODATE to indicate camera modification

Arguments:
- chunk: The state chunk containing camera data
- file: The file context for loading (may be NULL)
Return Value:
- CKERROR: 0 for success, -1 for invalid chunk
*************************************************/
CKERROR RCKCamera::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) {
        return CKERR_INVALIDPARAMETER;
    }

    // Call base class load first to handle entity data
    RCK3dEntity::Load(chunk, file);

    // Handle legacy format (data version < 5)
    if (chunk->GetDataVersion() < 5) {
        // Legacy format uses separate identifiers for each property

        if (chunk->SeekIdentifier(CK_STATESAVE_CAMERAFOV)) {
            m_Fov = chunk->ReadFloat();
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_CAMERAPROJTYPE)) {
            m_ProjectionType = chunk->ReadDword();
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_CAMERAOTHOZOOM)) {
            m_OrthographicZoom = chunk->ReadFloat();
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_CAMERAASPECT)) {
            m_Width = chunk->ReadInt();
            m_Height = chunk->ReadInt();
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_CAMERAPLANES)) {
            m_FrontPlane = chunk->ReadFloat();
            m_BackPlane = chunk->ReadFloat();
        }
    } else {
        // Current format (data version >= 5)
        if (chunk->SeekIdentifier(CK_STATESAVE_CAMERAONLY)) {
            // Current format uses single identifier with all camera data

            m_ProjectionType = chunk->ReadInt();
            m_Fov = chunk->ReadFloat();
            m_OrthographicZoom = chunk->ReadFloat();

            // Unpack width and height from single DWORD
            CKDWORD packedDimensions = chunk->ReadDword();
            m_Width = static_cast<int>(packedDimensions & 0xFFFF);
            m_Height = static_cast<int>((packedDimensions >> 16) & 0xFFFF);

            m_FrontPlane = chunk->ReadFloat();
            m_BackPlane = chunk->ReadFloat();
        }
    }

    // Mark camera as modified after loading
    ModifyObjectFlags(0, CK_OBJECT_UPTODATE);

    return CK_OK;
}

//=============================================================================
// Static Class Methods (for class registration)
// Based on IDA Pro analysis of original CK2_3D.dll
//=============================================================================

CKSTRING RCKCamera::GetClassName() {
    return (CKSTRING) "Camera";
}

int RCKCamera::GetDependenciesCount(int mode) {
    return 0;
}

CKSTRING RCKCamera::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCKCamera::Register() {
    CKPARAMETERFROMCLASS(RCKCamera, CKPGUID_CAMERA);
}

CKCamera *RCKCamera::CreateInstance(CKContext *Context) {
    RCKCamera *cam = new RCKCamera(Context, nullptr);
    return reinterpret_cast<CKCamera *>(cam);
}
