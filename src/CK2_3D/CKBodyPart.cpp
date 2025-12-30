#include "RCKBodyPart.h"

#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKObject.h"
#include "CKDependencies.h"
#include "CKEnums.h"

#include "RCK3dEntity.h"
#include "RCKCharacter.h"
#include "RCKAnimation.h"

// Static class ID
CK_CLASSID RCKBodyPart::m_ClassID = CKCID_BODYPART;

/**
 * @brief RCKBodyPart constructor
 * @param Context The CKContext instance
 * @param name Optional name for the body part
 */
RCKBodyPart::RCKBodyPart(CKContext *Context, CKSTRING name)
    : RCK3dObject(Context, name),
      m_Character(nullptr),
      m_ExclusiveAnimation(nullptr) {
    // IDA: m_RotationJoint.m_Flags = 7, vectors initialized to 0
    m_RotationJoint.m_Flags = 7;
    m_RotationJoint.m_Min.Set(0.0f, 0.0f, 0.0f);
    m_RotationJoint.m_Max.Set(0.0f, 0.0f, 0.0f);
    m_RotationJoint.m_Damping.Set(0.0f, 0.0f, 0.0f);
}

/**
 * @brief RCKBodyPart destructor
 */
RCKBodyPart::~RCKBodyPart() {}

/**
 * @brief Get the class ID for RCKBodyPart
 * @return The class ID (42)
 */
CK_CLASSID RCKBodyPart::GetClassID() {
    return m_ClassID;
}

/**
 * @brief Save the body part data to a state chunk
 * @param file The CKFile to save to (can be nullptr for standalone saving)
 * @param flags Save flags
 * @return A new CKStateChunk containing the saved data
 */
CKStateChunk *RCKBodyPart::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = RCK3dEntity::Save(file, flags);

    // Keep the same early-return optimization used across the repo for "memory-only" saves.
    if (!file && (flags & CK_STATESAVE_BODYPARTONLY) == 0)
        return baseChunk;

    CKStateChunk *chunk = CreateCKStateChunk(m_ClassID, file);
    if (!chunk)
        return baseChunk;

    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    // Identifier 0x04000000: character reference + optional rotation joint block.
    chunk->WriteIdentifier(0x04000000);
    chunk->WriteObject(reinterpret_cast<CKObject *>(m_Character));

    // Rotation joint block is present only if CK_3DENTITY_IKJOINTVALID is set.
    if ((GetFlags() & CK_3DENTITY_IKJOINTVALID) != 0) {
        chunk->WriteBufferNoSize_LEndian(sizeof(CKIkJoint), &m_RotationJoint);
    }

    if (GetClassID() == m_ClassID)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();

    return chunk;
}

/**
 * @brief Load body part data from a state chunk
 * @param chunk The CKStateChunk containing the saved data
 * @param file The CKFile being loaded from
 * @return CKERROR indicating success or failure
 */
CKERROR RCKBodyPart::Load(CKStateChunk *chunk, CKFile *file) {
    // IDA: 0x1000ea7a
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    RCK3dEntity::Load(chunk, file);

    const int version = chunk->GetDataVersion();

    if (version >= 5) {
        if (chunk->SeekIdentifier(0x04000000)) {
            m_Character = static_cast<RCKCharacter *>(chunk->ReadObject(m_Context));
            if ((GetFlags() & CK_3DENTITY_IKJOINTVALID) != 0) {
                chunk->ReadAndFillBuffer_LEndian(sizeof(CKIkJoint), &m_RotationJoint);
            }
        }
    } else {
        // Legacy format (version < 5): 0x01000000 holds 6 VxVectors
        // v5[0..2]: Used for building rotation flags (non-zero = flag set)
        // v5[3..5]: Min, Max, Damping vectors
        // NOTE: The flag calculation uses (base << (i - 1)) which is UB for i=0,
        // but MSVC produces a right shift, resulting in unusual flag mappings.
        // We match this behavior exactly for binary compatibility.
        if (chunk->SeekIdentifier(0x01000000)) {
            VxVector v5[6];
            std::memset(v5, 0, sizeof(v5));
            chunk->ReadAndFillBuffer_LEndian(v5);

            // IDA: Min = v5[3], Max = v5[4], Damping = v5[5]
            m_RotationJoint.m_Damping = v5[5];
            m_RotationJoint.m_Max = v5[4];
            m_RotationJoint.m_Min = v5[3];
            m_RotationJoint.m_Flags = 0;

            // Build flags from v5[0], v5[1], v5[2]
            for (int i = 0; i < 3; ++i) {
                if (*(&v5[0].x + i) != 0.0f)
                    m_RotationJoint.m_Flags |= (1 << (i - 1));
                if (*(&v5[1].x + i) != 0.0f)
                    m_RotationJoint.m_Flags |= (16 << (i - 1));
                if (*(&v5[2].x + i) != 0.0f)
                    m_RotationJoint.m_Flags |= (256 << (i - 1));
            }
        }

        if (chunk->SeekIdentifier(0x04000000)) {
            m_Character = static_cast<RCKCharacter *>(chunk->ReadObject(m_Context));
        }
    }

    return CK_OK;
}

/**
 * @brief Get memory occupation for the body part
 * @return Memory size in bytes
 */
int RCKBodyPart::GetMemoryOccupation() {
    return RCK3dEntity::GetMemoryOccupation() + (sizeof(RCKBodyPart) - sizeof(RCK3dEntity));
}

/**
 * @brief Copy body part data to another object
 * @param o The target object
 * @param context Dependencies context
 * @return CKERROR indicating success or failure
 */
CKERROR RCKBodyPart::PrepareDependencies(CKDependenciesContext &context) {
    // IDA: 0x1000ed38 - just calls base and FinishPrepareDependencies
    CKERROR err = RCK3dEntity::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR RCKBodyPart::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = RCK3dEntity::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    m_Character = static_cast<RCKCharacter *>(context.Remap(reinterpret_cast<CKObject *>(m_Character)));
    m_ExclusiveAnimation = static_cast<RCKAnimation *>(context.Remap(reinterpret_cast<CKObject *>(m_ExclusiveAnimation)));
    return CK_OK;
}

CKERROR RCKBodyPart::Copy(CKObject &o, CKDependenciesContext &context) {
    // IDA: 0x1000edd5 - Copy FROM source (o) TO this
    CKERROR err = RCK3dEntity::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKBodyPart &src = (RCKBodyPart &) o;
    m_Character = src.m_Character;
    m_ExclusiveAnimation = src.m_ExclusiveAnimation;
    std::memcpy(&m_RotationJoint, &src.m_RotationJoint, sizeof(m_RotationJoint));
    return CK_OK;
}

// Static class registration methods
CKSTRING RCKBodyPart::GetClassName() {
    return (CKSTRING) "BodyPart";
}

int RCKBodyPart::GetDependenciesCount(int mode) {
    return 0; // No dependencies
}

CKSTRING RCKBodyPart::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCKBodyPart::Register() {
    // IDA: 0x1000ec81 - CKGUID(0x4BA2618E, 0x7AAD0D02)
    CKCLASSNOTIFYFROMCID(RCKBodyPart, CKCID_CHARACTER);
    CKPARAMETERFROMCLASS(RCKBodyPart, CKPGUID_BODYPART);
}

CKBodyPart *RCKBodyPart::CreateInstance(CKContext *Context) {
    return reinterpret_cast<CKBodyPart *>(new RCKBodyPart(Context));
}

// Additional body part methods
CKCharacter *RCKBodyPart::GetCharacter() const {
    return reinterpret_cast<CKCharacter *>(m_Character);
}

void RCKBodyPart::SetExclusiveAnimation(const CKAnimation *anim) {
    m_ExclusiveAnimation = reinterpret_cast<RCKAnimation *>(const_cast<CKAnimation *>(anim));
}

CKAnimation *RCKBodyPart::GetExclusiveAnimation() const {
    return reinterpret_cast<CKAnimation *>(m_ExclusiveAnimation);
}

void RCKBodyPart::GetRotationJoint(CKIkJoint *joint) const {
    // IDA: 0x1000e857 - direct memcpy, no null check
    std::memcpy(joint, &m_RotationJoint, sizeof(CKIkJoint));
}

void RCKBodyPart::SetRotationJoint(const CKIkJoint *joint) {
    // IDA: 0x1000e87b - direct memcpy, no null check or flag modification
    std::memcpy(&m_RotationJoint, joint, sizeof(CKIkJoint));
}

CKERROR RCKBodyPart::FitToJoint() {
    // IDA: 0x1000e89f - uses Vx3DMatrixToEulerAngles/FromEulerAngles directly
    VxVector euler(0.0f, 0.0f, 0.0f);

    // Extract Euler angles from local matrix
    Vx3DMatrixToEulerAngles(GetLocalMatrix(), &euler.x, &euler.y, &euler.z);

    // Apply joint limits for each axis
    // NOTE: The original DLL uses (16 << (i - 1)) which creates an off-by-one mapping:
    //   i=0: 16 >> 1 = 8 (undefined but MSVC gives 8) - checks non-standard bit
    //   i=1: 16 << 0 = 16 = CK_IKJOINT_LIMIT_X - checks X limit for Y axis!
    //   i=2: 16 << 1 = 32 = CK_IKJOINT_LIMIT_Y - checks Y limit for Z axis!
    // We match this behavior exactly for binary compatibility.
    for (int i = 0; i < 3; ++i) {
        if (((16 << (i - 1)) & m_RotationJoint.m_Flags) != 0) {
            float *angle = &euler.x + i;
            float *minVal = &m_RotationJoint.m_Min.x + i;
            float *maxVal = &m_RotationJoint.m_Max.x + i;

            if (*angle < *minVal)
                *angle = *minVal;
            if (*angle > *maxVal)
                *angle = *maxVal;
        }
    }

    // Apply modified euler angles back to local matrix
    Vx3DMatrixFromEulerAngles(m_LocalMatrix, euler.x, euler.y, euler.z);
    LocalMatrixChanged(FALSE, TRUE);

    return CK_OK;
}
