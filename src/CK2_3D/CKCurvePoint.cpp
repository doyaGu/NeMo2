#include "RCKCurvePoint.h"

#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKObject.h"
#include "CKBeObject.h"
#include "CKCurve.h"
#include "RCK3dEntity.h"

// Static class ID
CK_CLASSID RCKCurvePoint::m_ClassID = CKCID_CURVEPOINT;

/**
 * @brief RCKCurvePoint constructor
 * @param Context The CKContext instance
 * @param name Optional name for the curve point
 */
RCKCurvePoint::RCKCurvePoint(CKContext *Context, CKSTRING name)
    : RCK3dEntity(Context, name) {
    m_Tension = 0.0f;
    m_Continuity = 0.0f;
    m_Bias = 0.0f;
    m_Curve = nullptr;
    m_UseTCB = FALSE;
    m_Linear = FALSE;
    m_Length = 0.0f;
}

/**
 * @brief RCKCurvePoint destructor
 */
RCKCurvePoint::~RCKCurvePoint() {}

/**
 * @brief Get the class ID for RCKCurvePoint
 * @return The class ID (36)
 */
CK_CLASSID RCKCurvePoint::GetClassID() {
    return m_ClassID;
}

/**
 * @brief Save the curve point data to a state chunk
 * @param file The CKFile to save to (can be nullptr for standalone saving)
 * @param flags Save flags
 * @return A new CKStateChunk containing the saved data
 */
CKStateChunk *RCKCurvePoint::Save(CKFile *file, CKDWORD flags) {
    // Call base class Save first
    CKStateChunk *baseChunk = RCK3dEntity::Save(file, flags);

    // If no file and no special flags, return base chunk only
    if (!file && !(flags & CK_STATESAVE_CURVEONLY))
        return baseChunk;

    // Create a new state chunk for curve point data
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_CURVEPOINT, file);
    chunk->StartWrite();

    // Add the base class chunk
    chunk->AddChunkAndDelete(baseChunk);

    // Write curve point specific data
    chunk->WriteIdentifier(CK_STATESAVE_CURVEPOINTDEFAULTDATA);

    // Write curve point properties
    chunk->WriteObject(reinterpret_cast<CKObject *>(m_Curve));
    chunk->WriteInt(m_UseTCB);
    chunk->WriteInt(m_Linear);
    chunk->WriteFloat(m_Tension);
    chunk->WriteFloat(m_Continuity);
    chunk->WriteFloat(m_Bias);
    chunk->WriteVector(&m_TangentIn);
    chunk->WriteVector(&m_m_TangentOut);

    // Close or update the chunk based on class ID
    if (GetClassID() == CKCID_CURVEPOINT)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();

    return chunk;
}

/**
 * @brief Load curve point data from a state chunk
 * @param chunk The CKStateChunk containing the saved data
 * @param file The CKFile being loaded from
 * @return CKERROR indicating success or failure
 */
CKERROR RCKCurvePoint::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Call base class Load first
    RCK3dEntity::Load(chunk, file);

    // Handle different data versions
    if (chunk->GetDataVersion() < 5) {
        // Legacy format handling for data version < 5
        if (chunk->SeekIdentifier(CK_STATESAVE_CURVEPOINTDEFAULTDATA)) {
            // Load basic curve point properties
            m_Curve = reinterpret_cast<CKCurve *>(chunk->ReadObject(m_Context));
            m_UseTCB = chunk->ReadInt();
            m_Linear = chunk->ReadInt();

            // Load position data (legacy format)
            VxVector position;
            position.x = chunk->ReadFloat();
            position.y = chunk->ReadFloat();
            position.z = chunk->ReadFloat();
            SetPosition(&position, nullptr, FALSE);
        }

        // Load TCB parameters (legacy format)
        if (chunk->SeekIdentifier(CK_STATESAVE_CURVEPOINTTCB)) {
            m_Tension = chunk->ReadFloat();
            m_Continuity = chunk->ReadFloat();
            m_Bias = chunk->ReadFloat();
        }

        // Load reserved vector (legacy format)
        if (chunk->SeekIdentifier(CK_STATESAVE_CURVEPOINTCURVEPOS)) {
            m_ReservedVector.x = chunk->ReadFloat();
            m_ReservedVector.y = chunk->ReadFloat();
            m_ReservedVector.z = chunk->ReadFloat();
        }

        // Load tangent vectors (legacy format)
        if (chunk->SeekIdentifier(CK_STATESAVE_CURVEPOINTTANGENTS)) {
            m_TangentIn.x = chunk->ReadFloat();
            m_TangentIn.y = chunk->ReadFloat();
            m_TangentIn.z = chunk->ReadFloat();
            m_m_TangentOut.x = chunk->ReadFloat();
            m_m_TangentOut.y = chunk->ReadFloat();
            m_m_TangentOut.z = chunk->ReadFloat();
        }
    } else {
        // Modern format for data version >= 5
        if (chunk->SeekIdentifier(CK_STATESAVE_CURVEPOINTDEFAULTDATA)) {
            // Load all curve point properties in modern format
            m_Curve = reinterpret_cast<CKCurve *>(chunk->ReadObject(m_Context));
            m_UseTCB = chunk->ReadInt();
            m_Linear = chunk->ReadInt();
            m_Tension = chunk->ReadFloat();
            m_Continuity = chunk->ReadFloat();
            m_Bias = chunk->ReadFloat();
            chunk->ReadVector(&m_TangentIn);
            chunk->ReadVector(&m_m_TangentOut);
        }
    }

    return CK_OK;
}

/**
 * @brief Get memory occupation for the curve point
 * @return Memory size in bytes
 */
int RCKCurvePoint::GetMemoryOccupation() {
    return RCK3dEntity::GetMemoryOccupation() + (sizeof(RCKCurvePoint) - sizeof(RCK3dEntity));
}

CKERROR RCKCurvePoint::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = RCK3dEntity::RemapDependencies(context);
    if (err != CK_OK) {
        return err;
    }

    m_Curve = (CKCurve *) context.Remap(m_Curve);
    SetParent(m_Curve, TRUE);
    return CK_OK;
}

/**
 * @brief Copy curve point data to another object
 * @param o The target object
 * @param context Dependencies context
 * @return CKERROR indicating success or failure
 */
CKERROR RCKCurvePoint::Copy(CKObject &o, CKDependenciesContext &context) {
    // Base class copy
    CKERROR err = RCK3dEntity::Copy(o, context);
    if (err != CK_OK) {
        return err;
    }

    // Copy curve point specific data
    RCKCurvePoint &src = (RCKCurvePoint &) o;
    m_Curve = src.m_Curve;
    m_Tension = src.m_Tension;
    m_Continuity = src.m_Continuity;
    m_Bias = src.m_Bias;
    m_Length = src.m_Length;
    m_ReservedVector = src.m_ReservedVector;
    m_TangentIn = src.m_TangentIn;
    m_m_TangentOut = src.m_m_TangentOut;
    m_NotUsedVector = src.m_NotUsedVector;
    m_UseTCB = src.m_UseTCB;
    m_Linear = src.m_Linear;

    return CK_OK;
}

// Transformation methods implementation
void RCKCurvePoint::Rotate(const VxVector *Axis, float Angle, CK3dEntity *Ref, CKBOOL KeepChildren) {
    RCK3dEntity::Rotate(Axis, Angle, Ref, KeepChildren);
    NotifyUpdate();
}

void RCKCurvePoint::Translate(const VxVector *Vect, CK3dEntity *Ref, CKBOOL KeepChildren) {
    RCK3dEntity::Translate(Vect, Ref, KeepChildren);
    NotifyUpdate();
}

void RCKCurvePoint::AddScale(const VxVector *Scale, CKBOOL KeepChildren, CKBOOL Local) {
    RCK3dEntity::AddScale(Scale, KeepChildren, Local);
    NotifyUpdate();
}

void RCKCurvePoint::SetPosition(const VxVector *Pos, CK3dEntity *Ref, CKBOOL KeepChildren) {
    RCK3dEntity::SetPosition(Pos, Ref, KeepChildren);
    NotifyUpdate();
}

void RCKCurvePoint::SetOrientation(const VxVector *Dir, const VxVector *Up, const VxVector *Right, CK3dEntity *Ref,
                                   CKBOOL KeepChildren) {
    RCK3dEntity::SetOrientation(Dir, Up, Right, Ref, KeepChildren);
    NotifyUpdate();
}

void RCKCurvePoint::SetQuaternion(const VxQuaternion *Quat, CK3dEntity *Ref, CKBOOL KeepChildren, CKBOOL KeepScale) {
    RCK3dEntity::SetQuaternion(Quat, Ref, KeepChildren, KeepScale);
    NotifyUpdate();
}

void RCKCurvePoint::SetScale(const VxVector *Scale, CKBOOL KeepChildren, CKBOOL Local) {
    RCK3dEntity::SetScale(Scale, KeepChildren, Local);
    NotifyUpdate();
}

CKBOOL RCKCurvePoint::ConstructWorldMatrix(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat) {
    CKBOOL result = RCK3dEntity::ConstructWorldMatrix(Pos, Scale, Quat);
    NotifyUpdate();
    return result;
}

CKBOOL RCKCurvePoint::ConstructWorldMatrixEx(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat,
                                             const VxQuaternion *Shear, float Sign) {
    CKBOOL result = RCK3dEntity::ConstructWorldMatrixEx(Pos, Scale, Quat, Shear, Sign);
    NotifyUpdate();
    return result;
}

CKBOOL RCKCurvePoint::ConstructLocalMatrix(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat) {
    CKBOOL result = RCK3dEntity::ConstructLocalMatrix(Pos, Scale, Quat);
    NotifyUpdate();
    return result;
}

CKBOOL RCKCurvePoint::ConstructLocalMatrixEx(const VxVector *Pos, const VxVector *Scale, const VxQuaternion *Quat,
                                             const VxQuaternion *Shear, float Sign) {
    CKBOOL result = RCK3dEntity::ConstructLocalMatrixEx(Pos, Scale, Quat, Shear, Sign);
    NotifyUpdate();
    return result;
}

void RCKCurvePoint::SetLocalMatrix(const VxMatrix &Mat, CKBOOL KeepChildren) {
    RCK3dEntity::SetLocalMatrix(Mat, KeepChildren);
    NotifyUpdate();
}

void RCKCurvePoint::SetWorldMatrix(const VxMatrix &Mat, CKBOOL KeepChildren) {
    RCK3dEntity::SetWorldMatrix(Mat, KeepChildren);
    NotifyUpdate();
}

// =====================================================
// CKCurvePoint-specific virtual methods
// =====================================================

CKCurve *RCKCurvePoint::GetCurve() {
    return m_Curve;
}

void RCKCurvePoint::SetCurve(CKCurve *curve) {
    if (curve)
        SetParent(curve, TRUE);
    m_Curve = curve;
}

float RCKCurvePoint::GetBias() {
    return m_Bias;
}

void RCKCurvePoint::SetBias(float bias) {
    m_Bias = bias;
    NotifyUpdate();
}

float RCKCurvePoint::GetTension() {
    return m_Tension;
}

void RCKCurvePoint::SetTension(float tension) {
    m_Tension = tension;
    NotifyUpdate();
}

float RCKCurvePoint::GetContinuity() {
    return m_Continuity;
}

void RCKCurvePoint::SetContinuity(float continuity) {
    m_Continuity = continuity;
    NotifyUpdate();
}

CKBOOL RCKCurvePoint::IsLinear() {
    return m_Linear;
}

void RCKCurvePoint::SetLinear(CKBOOL linear) {
    m_Linear = linear;
    NotifyUpdate();
}

void RCKCurvePoint::UseTCB(CKBOOL use) {
    m_UseTCB = use;
}

CKBOOL RCKCurvePoint::IsTCB() {
    return m_UseTCB;
}

float RCKCurvePoint::GetLength() {
    return m_Length;
}

void RCKCurvePoint::GetTangents(VxVector *InTangent, VxVector *OutTangent) {
    if (InTangent) *InTangent = m_TangentIn;
    if (OutTangent) *OutTangent = m_m_TangentOut;
}

void RCKCurvePoint::SetTangents(VxVector *InTangent, VxVector *OutTangent) {
    if (InTangent) m_TangentIn = *InTangent;
    if (OutTangent) m_m_TangentOut = *OutTangent;
    NotifyUpdate();
}

void RCKCurvePoint::SetCurveLength(float length) {
    m_Length = length;
}

void RCKCurvePoint::GetReservedVector(VxVector *vector) const {
    if (vector) {
        *vector = m_ReservedVector;
    }
}

void RCKCurvePoint::SetReservedVector(const VxVector *vector) {
    if (vector) {
        m_ReservedVector = *vector;
    }
}

void RCKCurvePoint::GetFittedVector(VxVector *vector) const {
    if (vector) {
        *vector = m_NotUsedVector;
    }
}

void RCKCurvePoint::SetFittedVector(const VxVector *vector) {
    if (vector) {
        m_NotUsedVector = *vector;
    }
}

void RCKCurvePoint::NotifyUpdate() {
    if (m_Curve) {
        m_Curve->ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
    }
}

// Static class registration methods
CKSTRING RCKCurvePoint::GetClassName() {
    return (CKSTRING) "Curve Point";
}

int RCKCurvePoint::GetDependenciesCount(int mode) {
    return 0; // No dependencies
}

CKSTRING RCKCurvePoint::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCKCurvePoint::Register() {
    // Based on IDA decompilation
    CKPARAMETERFROMCLASS(RCKCurvePoint, CKPGUID_CURVEPOINT);
}

CKCurvePoint *RCKCurvePoint::CreateInstance(CKContext *Context) {
    return reinterpret_cast<CKCurvePoint *>(new RCKCurvePoint(Context));
}
