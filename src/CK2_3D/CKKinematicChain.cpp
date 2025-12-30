/**
 * @file CKKinematicChain.cpp
 * @brief Implementation of CKKinematicChain class for inverse kinematics.
 *
 * Reverse engineered from CK2_3D.dll
 */

#include "RCKKinematicChain.h"

#include "CKFile.h"
#include "CKStateChunk.h"
#include "RCKBodyPart.h"

// Static class ID
CK_CLASSID RCKKinematicChain::m_ClassID = CKCID_KINEMATICCHAIN;

//=============================================================================
// Class Registration
//=============================================================================

/**
 * Constructor - Based on IDA decompilation at 0x100525B0
 * Object size: 0x24 (36 bytes)
 */
RCKKinematicChain::RCKKinematicChain(CKContext *Context, CKSTRING name)
    : CKKinematicChain(Context, name),
      m_StartEffector(nullptr),
      m_EndEffector(nullptr),
      m_ChainBodyCount(0),
      m_ChainData(nullptr) {}

/**
 * Destructor - Based on IDA decompilation at 0x10052601
 */
RCKKinematicChain::~RCKKinematicChain() {
    m_StartEffector = nullptr;
    m_EndEffector = nullptr;
    if (m_ChainData) {
        delete[] m_ChainData;
    }
}

CKSTRING RCKKinematicChain::GetClassName() {
    return (CKSTRING) "Kinematic Chain";
}

int RCKKinematicChain::GetDependenciesCount(int mode) {
    return 0;
}

CKSTRING RCKKinematicChain::GetDependencies(int i, int mode) {
    return nullptr;
}

/**
 * Register - Based on IDA decompilation at 0x10055F91
 */
void RCKKinematicChain::Register() {
    // Based on IDA decompilation
    CKCLASSNOTIFYFROMCID(RCKKinematicChain, CKCID_BODYPART);
    CKCLASSNOTIFYFROMCID(RCKKinematicChain, CKCID_CHARACTER);
    CKPARAMETERFROMCLASS(RCKKinematicChain, CKPGUID_KINEMATICCHAIN);
    CKCLASSDEFAULTOPTIONS(RCKKinematicChain, CK_GENERALOPTIONS_NODUPLICATENAMECHECK);
}

/**
 * CreateInstance - Based on IDA decompilation at 0x10056000
 */
CKKinematicChain *RCKKinematicChain::CreateInstance(CKContext *Context) {
    return new RCKKinematicChain(Context);
}

//=============================================================================
// Save/Load Implementation
//=============================================================================

/**
 * Save - Based on IDA decompilation at 0x10055E51
 */
CKStateChunk *RCKKinematicChain::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_KINEMATICCHAIN, file);
    if (!chunk)
        return nullptr;

    // Get base class chunk
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    if (file || (flags & CK_STATESAVE_KINEMATICCHAINALL) != 0) {
        chunk->WriteIdentifier(CK_STATESAVE_KINEMATICCHAINALL);
        chunk->WriteObject(nullptr);
        chunk->WriteObject(m_StartEffector);
        chunk->WriteObject(m_EndEffector);
    }

    if (GetClassID() == CKCID_KINEMATICCHAIN)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();

    return chunk;
}

/**
 * Load - Based on IDA decompilation at 0x10055F05
 */
CKERROR RCKKinematicChain::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Load base object data
    CKERROR err = CKObject::Load(chunk, file);
    if (err != CK_OK)
        return err;

    if (chunk->SeekIdentifier(CK_STATESAVE_KINEMATICCHAINALL)) {
        chunk->ReadObject(m_Context);  // Discard placeholder
        m_StartEffector = (CKBodyPart *)chunk->ReadObject(m_Context);
        m_EndEffector = (CKBodyPart *)chunk->ReadObject(m_Context);
    }

    return CK_OK;
}

//=============================================================================
// Base class overrides
//=============================================================================

CK_CLASSID RCKKinematicChain::GetClassID() {
    return m_ClassID;
}

/**
 * CheckPreDeletion - Based on IDA decompilation at 0x10054C08
 */
void RCKKinematicChain::CheckPreDeletion() {
    if (m_EndEffector && (m_EndEffector->GetObjectFlags() & CK_OBJECT_TOBEDELETED))
        m_EndEffector = nullptr;

    if (m_StartEffector && (m_StartEffector->GetObjectFlags() & CK_OBJECT_TOBEDELETED))
        m_StartEffector = nullptr;
}

/**
 * GetMemoryOccupation - Based on IDA decompilation at 0x10054C57
 * Returns: base + 16 + 116 * chain count
 */
int RCKKinematicChain::GetMemoryOccupation() {
    return CKObject::GetMemoryOccupation() + (sizeof(RCKKinematicChain) - sizeof(CKObject)) + (sizeof(CKIKChainBodyData) * m_ChainBodyCount);
}

/**
 * IsObjectUsed - Based on IDA decompilation at 0x10054C86
 */
CKBOOL RCKKinematicChain::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    if (cid != CKCID_BODYPART)
        return CKObject::IsObjectUsed(obj, cid);

    if (obj == (CKObject *)m_EndEffector)
        return TRUE;
    if (obj == (CKObject *)m_StartEffector)
        return TRUE;

    return CKObject::IsObjectUsed(obj, cid);
}

/**
 * RemapDependencies - Based on IDA decompilation at 0x10056062
 */
CKERROR RCKKinematicChain::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    m_StartEffector = (CKBodyPart *)context.Remap(m_StartEffector);
    m_EndEffector = (CKBodyPart *)context.Remap(m_EndEffector);
    return CK_OK;
}

/**
 * Copy - Based on IDA decompilation at 0x100560B7
 */
CKERROR RCKKinematicChain::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    auto &src = static_cast<RCKKinematicChain &>(o);
    m_StartEffector = src.m_StartEffector;
    m_EndEffector = src.m_EndEffector;
    return CK_OK;
}

//=============================================================================
// CKKinematicChain virtual methods
//=============================================================================

/**
 * GetChainLength - Based on IDA decompilation at 0x1005267E
 */
float RCKKinematicChain::GetChainLength(CKBodyPart *End) {
    CKBodyPart *startEffector = m_StartEffector;
    CKBodyPart *endEffector = m_EndEffector;

    if (End)
        endEffector = End;

    if (startEffector == endEffector)
        return 0.0f;
    if (!endEffector)
        return 0.0f;

    CK3dEntity *parent = endEffector->GetParent();
    if (!parent)
        return 0.0f;

    VxVector endPos, parentPos;
    endEffector->GetPosition(&endPos, nullptr);
    parent->GetPosition(&parentPos, nullptr);

    VxVector diff = endPos - parentPos;
    float distance = Magnitude(diff);

    return distance + GetChainLength((CKBodyPart *)parent);
}

/**
 * GetChainBodyCount - Based on IDA decompilation at 0x100527E4
 */
int RCKKinematicChain::GetChainBodyCount(CKBodyPart *End) {
    CKBodyPart *startEffector = m_StartEffector;
    CKBodyPart *endEffector = m_EndEffector;

    if (End)
        endEffector = End;

    if (startEffector == endEffector)
        return 1;
    if (!startEffector)
        return 0;
    if (!endEffector)
        return 0;

    CKBodyPart *parent = (CKBodyPart *)endEffector->GetParent();
    if (parent)
        return GetChainBodyCount(parent) + 1;
    else
        return 0;
}

/**
 * GetStartEffector - Based on IDA decompilation at 0x10054BCF
 */
CKBodyPart *RCKKinematicChain::GetStartEffector() {
    return m_StartEffector;
}

/**
 * SetStartEffector - Based on IDA decompilation at 0x10054BE0
 */
CKERROR RCKKinematicChain::SetStartEffector(CKBodyPart *start) {
    m_StartEffector = start;
    return CK_OK;
}

/**
 * GetEffector - Based on IDA decompilation at 0x10052753
 */
CKBodyPart *RCKKinematicChain::GetEffector(int pos) {
    CKBodyPart *endEffector = m_EndEffector;

    if (!endEffector)
        return nullptr;
    if (!m_StartEffector)
        return nullptr;

    int count = GetChainBodyCount(endEffector);

    if (pos < 0)
        return nullptr;
    if (pos >= count)
        return nullptr;

    CKBodyPart *current = endEffector;
    for (int i = count - 1; current && i != pos; --i) {
        current = (CKBodyPart *)current->GetParent();
    }

    return current;
}

/**
 * GetEndEffector - Based on IDA decompilation at 0x10054BA6
 */
CKBodyPart *RCKKinematicChain::GetEndEffector() {
    return m_EndEffector;
}

/**
 * SetEndEffector - Based on IDA decompilation at 0x10054BB7
 */
CKERROR RCKKinematicChain::SetEndEffector(CKBodyPart *end) {
    m_EndEffector = end;
    return CK_OK;
}

//=============================================================================
// IK Solver
//=============================================================================

/**
 * Helper: compare vector components - Based on IDA sub_10056170
 */
static CKBOOL CompareVectorComponents(const VxVector &a, const VxVector &b) {
    return (a.x < b.x) && (a.y < b.y) && (a.z < b.z);
}

/**
 * IKSetEffectorPos - Based on IDA decompilation at 0x10053CCE
 */
CKERROR RCKKinematicChain::IKSetEffectorPos(VxVector *pos, CK3dEntity *ref, CKBodyPart *body) {
    CKBodyPart *startEffector = m_StartEffector;
    CKBodyPart *endEffector = m_EndEffector;

    if (body)
        endEffector = body;

    if (!pos)
        return CKERR_INVALIDPARAMETER;
    if (!startEffector)
        return CKERR_INVALIDOPERATION;
    if (!endEffector)
        return CKERR_INVALIDOPERATION;

    CKBodyPart *ikEndEffector = endEffector;

    // Calculate chain body count
    m_ChainBodyCount = GetChainBodyCount(endEffector);
    if ((int)m_ChainBodyCount < 2)
        return CKERR_INVALIDOPERATION;

    // Free existing chain data
    if (m_ChainData) {
        delete[] m_ChainData;
    }

    // Allocate new chain data
    m_ChainData = new CKIKChainBodyData[m_ChainBodyCount];

    // Initialize chain data for each body part (traverse from end to start)
    CKBodyPart *currentBody = endEffector;
    for (int i = (int)m_ChainBodyCount - 1; i >= 0; --i) {
        m_ChainData[i].m_BodyPart = currentBody;

        // Get rotation joint constraints
        currentBody->GetRotationJoint(&m_ChainData[i].m_RotationJoint);

        m_ChainData[i].m_IsLocked = 0;
        m_ChainData[i].m_WasAtLimit = 0;

        // Check if all axes are locked
        if ((m_ChainData[i].m_RotationJoint.m_Flags & CK_IKJOINT_ACTIVE) == 0) {
            m_ChainData[i].m_IsLocked = 1;
        }

        // Extract euler angles from local matrix
        VxMatrix localMatrix = currentBody->GetLocalMatrix();
        VxVector eulerAngles;
        Vx3DMatrixToEulerAngles(localMatrix, &eulerAngles.x, &eulerAngles.y, &eulerAngles.z);

        // Check joint limits for each axis
        if (!m_ChainData[i].m_IsLocked) {
            for (int j = 0; j < 3; ++j) {
                CKDWORD axisBit = (1 << j);   // CK_IKJOINT_ACTIVE_X/Y/Z
                CKDWORD limitBit = (16 << j); // CK_IKJOINT_LIMIT_X/Y/Z

                if ((m_ChainData[i].m_RotationJoint.m_Flags & axisBit) &&
                    (m_ChainData[i].m_RotationJoint.m_Flags & limitBit)) {
                    float angle = (&eulerAngles.x)[j];
                    while (angle > PI)
                        angle -= 2 * PI;
                    while (angle < -PI)
                        angle += 2 * PI;

                    float minAngle = (&m_ChainData[i].m_RotationJoint.m_Min.x)[j];
                    float maxAngle = (&m_ChainData[i].m_RotationJoint.m_Max.x)[j];

                    if ((angle - 0.05f) <= minAngle) {
                        m_ChainData[i].m_IsLocked = 1;
                        m_ChainData[i].m_WasAtLimit = 1;
                    }
                    if ((angle + 0.05f) >= maxAngle) {
                        m_ChainData[i].m_IsLocked = 1;
                        m_ChainData[i].m_WasAtLimit = 1;
                    }
                }
            }
        }

        // Move to parent
        currentBody = (CKBodyPart *)currentBody->GetParent();
    }

    // Get positions
    VxVector currentEndPos;
    VxVector targetPos = *pos;
    VxVector startPos;

    endEffector->GetPosition(&currentEndPos, nullptr);
    if (ref)
        ref->Transform(&targetPos, pos, nullptr);

    startEffector->GetPosition(&startPos, nullptr);

    // Calculate distance to target and chain length
    VxVector toTarget = targetPos - startPos;
    float chainLength = GetChainLength(endEffector);
    float targetDistance = Magnitude(toTarget);

    // Clamp target if beyond reach
    if (targetDistance > chainLength) {
        if (targetDistance == 0.0f)
            return CKERR_INVALIDOPERATION;

        toTarget *= chainLength / targetDistance;
        targetPos = startPos + toTarget;
    }

    // Special handling for 2-body chains
    if (m_ChainBodyCount == 2 && targetDistance < chainLength) {
        if (targetDistance == 0.0f)
            return CKERR_INVALIDOPERATION;

        toTarget *= chainLength / targetDistance;
        targetPos = startPos + toTarget;
    }

    // Find first and last unlocked joints
    int firstUnlocked = -1;
    int lastUnlocked = -1;

    for (int i = 0; i < (int)m_ChainBodyCount - 1; ++i) {
        if (!m_ChainData[i].m_IsLocked) {
            firstUnlocked = i;
            break;
        }
    }

    if (firstUnlocked == -1) {
        for (int i = 0; i < (int)m_ChainBodyCount - 1; ++i) {
            if (m_ChainData[i].m_WasAtLimit) {
                firstUnlocked = i;
                break;
            }
        }
    }

    if (firstUnlocked == -1)
        return CKERR_INVALIDOPERATION;

    for (int i = (int)m_ChainBodyCount - 2; i >= 0; --i) {
        if (!m_ChainData[i].m_IsLocked) {
            lastUnlocked = i;
            break;
        }
    }

    if (lastUnlocked == -1) {
        for (int i = (int)m_ChainBodyCount - 2; i >= 0; --i) {
            if (m_ChainData[i].m_WasAtLimit) {
                lastUnlocked = i;
                break;
            }
        }
    }

    if (lastUnlocked == -1)
        return CKERR_INVALIDOPERATION;

    // Get position of first unlocked body part
    VxVector firstUnlockedPos;
    m_ChainData[firstUnlocked].m_BodyPart->GetPosition(&firstUnlockedPos, nullptr);

    VxVector ikDelta;
    VxVector bodyPos;

    // Calculate remaining chain length from first unlocked
    ikDelta = targetPos - firstUnlockedPos;
    VxVector prevPos = firstUnlockedPos;
    chainLength = 0.0f;

    for (int i = firstUnlocked + 1; i < (int)m_ChainBodyCount; ++i) {
        if (!m_ChainData[i].m_IsLocked || i == (int)m_ChainBodyCount - 1) {
            m_ChainData[i].m_BodyPart->GetPosition(&bodyPos, nullptr);
            VxVector segment = bodyPos - prevPos;
            chainLength += Magnitude(segment);
            prevPos = bodyPos;
        }
    }

    // Clamp delta if beyond remaining reach
    targetDistance = Magnitude(ikDelta);
    if (targetDistance > chainLength) {
        if (targetDistance == 0.0f)
            return CKERR_INVALIDOPERATION;

        ikDelta *= chainLength / targetDistance;
        targetPos = firstUnlockedPos + ikDelta;
    }

    // Special handling when first == last unlocked
    if (firstUnlocked == lastUnlocked && targetDistance < chainLength) {
        if (targetDistance == 0.0f)
            return CKERR_INVALIDOPERATION;

        ikDelta *= chainLength / targetDistance;
        targetPos = firstUnlockedPos + ikDelta;
    }

    // Save current transforms for rollback
    VxMatrix *savedMatrices = new VxMatrix[m_ChainBodyCount];
    for (CKDWORD i = 0; i < m_ChainBodyCount; ++i) {
        savedMatrices[i] = m_ChainData[i].m_BodyPart->GetWorldMatrix();
    }

    // IK solver iteration
    int converged = 0;
    float stepSize = 10.0f;
    int maxIterations = 30;

    endEffector->GetPosition(&currentEndPos, nullptr);
    VxVector errorVec = targetPos - currentEndPos;
    float initialError = Magnitude(errorVec);

    int iteration = 0;
    while (iteration < maxIterations && !converged) {
        endEffector->GetPosition(&currentEndPos, nullptr);

        VxVector toGoal = targetPos - currentEndPos;
        toGoal *= stepSize;

        CKERROR err = IKRotateToward(&toGoal, body);
        if (err != CK_OK) {
            delete[] savedMatrices;
            return err;
        }

        endEffector->GetPosition(&currentEndPos, nullptr);
        errorVec = targetPos - currentEndPos;
        float newError = Magnitude(errorVec);

        if (newError <= 0.05f)
            converged = 1;

        VxVector toGoalCopy = toGoal;
        float stepMag = Magnitude(toGoalCopy);
        float expectedImprovement = stepMag * stepSize;

        if (newError >= expectedImprovement) {
            // Step too large, reduce and restore
            stepSize *= 2.0f;
            for (CKDWORD i = 0; i < m_ChainBodyCount; ++i) {
                m_ChainData[i].m_BodyPart->SetWorldMatrix(savedMatrices[i]);
            }
        } else if (2.0f * newError < expectedImprovement) {
            stepSize *= 0.75f;
        }

        ++iteration;
    }

    // Check final error against initial
    if (!converged) {
        VxVector initErrorVec(initialError, initialError, initialError);
        if (CompareVectorComponents(initErrorVec, errorVec)) {
            // Final error is worse, restore
            for (CKDWORD i = 0; i < m_ChainBodyCount; ++i) {
                m_ChainData[i].m_BodyPart->SetWorldMatrix(savedMatrices[i]);
            }
        }
    }

    delete[] savedMatrices;
    return CK_OK;
}

/**
 * IKRotateToward - Based on IDA decompilation at 0x10052859
 * Jacobian-based IK solver using SVD.
 */
CKERROR RCKKinematicChain::IKRotateToward(VxVector *targetDelta, CKBodyPart *endBody) {
    if (!m_ChainData)
        return CKERR_INVALIDOPERATION;

    int chainBodyCount = m_ChainBodyCount;
    CKBodyPart *endEffector = m_EndEffector;
    if (endBody)
        endEffector = endBody;

    // DOF parameters (rotation only = 3)
    int dofType = 3;
    int dofCount = 3 * (chainBodyCount - 1);
    int iterationCount = 0;
    int limitViolated = 0;

    // Allocate workspace arrays
    float *lockedAxes = new float[dofCount];
    float *dofActiveFlags = new float[dofCount];
    float *computedAngleDeltas = new float[dofCount];
    float *singularValues = new float[dofCount];
    float *targetVecArray = new float[dofType];

    float **jacobianAxes = new float *[dofType];
    for (int i = 0; i < dofType; ++i) {
        jacobianAxes[i] = new float[dofCount];
    }

    float **vMatrix = new float *[dofCount];
    for (int i = 0; i < dofCount; ++i) {
        vMatrix[i] = new float[dofCount];
    }

    // Initialize arrays
    memset(targetVecArray, 0, sizeof(float) * dofType);
    memset(dofActiveFlags, 0, sizeof(float) * dofCount);
    memset(lockedAxes, 0, sizeof(float) * dofCount);

    targetVecArray[0] = targetDelta->x;
    targetVecArray[1] = targetDelta->y;
    targetVecArray[2] = targetDelta->z;

    CKERROR result = CK_OK;

    // Main constraint iteration loop
    while (TRUE) {
        VxMatrix jointWorldMatrix;
        VxVector eulerAngles;

        // Get end effector world position
        jointWorldMatrix = m_ChainData[chainBodyCount - 1].m_BodyPart->GetWorldMatrix();
        VxVector endEffectorPosition(jointWorldMatrix[3][0], jointWorldMatrix[3][1], jointWorldMatrix[3][2]);

        int dofIdx = 0;
        int jointIdx = 0;

        // Build Jacobian for each joint
        while (jointIdx < chainBodyCount - 1) {
            jointWorldMatrix = m_ChainData[jointIdx].m_BodyPart->GetWorldMatrix();

            VxVector colX(jointWorldMatrix[0][0], jointWorldMatrix[1][0], jointWorldMatrix[2][0]);
            VxVector colY(jointWorldMatrix[0][1], jointWorldMatrix[1][1], jointWorldMatrix[2][1]);
            VxVector colZ(jointWorldMatrix[0][2], jointWorldMatrix[1][2], jointWorldMatrix[2][2]);
            VxVector bodyPosition(jointWorldMatrix[3][0], jointWorldMatrix[3][1], jointWorldMatrix[3][2]);

            VxVector leverArm = endEffectorPosition - bodyPosition;

            VxVector crossX = CrossProduct(colX, leverArm);
            VxVector crossY = CrossProduct(colY, leverArm);
            VxVector crossZ = CrossProduct(colZ, leverArm);

            // X axis
            CKBOOL xActive = (m_ChainData[jointIdx].m_RotationJoint.m_Flags & CK_IKJOINT_ACTIVE_X) &&
                             !((int)lockedAxes[dofIdx]);
            dofActiveFlags[dofIdx] = xActive ? 1.0f : 0.0f;
            jacobianAxes[0][dofIdx] = xActive ? crossX.x : 0.0f;
            jacobianAxes[1][dofIdx] = xActive ? crossX.y : 0.0f;
            jacobianAxes[2][dofIdx] = xActive ? crossX.z : 0.0f;

            // Y axis
            CKBOOL yActive = (m_ChainData[jointIdx].m_RotationJoint.m_Flags & CK_IKJOINT_ACTIVE_Y) &&
                             !((int)lockedAxes[dofIdx + 1]);
            dofActiveFlags[dofIdx + 1] = yActive ? 1.0f : 0.0f;
            jacobianAxes[0][dofIdx + 1] = yActive ? crossY.x : 0.0f;
            jacobianAxes[1][dofIdx + 1] = yActive ? crossY.y : 0.0f;
            jacobianAxes[2][dofIdx + 1] = yActive ? crossY.z : 0.0f;

            // Z axis
            CKBOOL zActive = (m_ChainData[jointIdx].m_RotationJoint.m_Flags & CK_IKJOINT_ACTIVE_Z) &&
                             !((int)lockedAxes[dofIdx + 2]);
            dofActiveFlags[dofIdx + 2] = zActive ? 1.0f : 0.0f;
            jacobianAxes[0][dofIdx + 2] = zActive ? crossZ.x : 0.0f;
            jacobianAxes[1][dofIdx + 2] = zActive ? crossZ.y : 0.0f;
            jacobianAxes[2][dofIdx + 2] = zActive ? crossZ.z : 0.0f;

            ++jointIdx;
            dofIdx += 3;
        }

        // Check if any DOF is active
        int activeDofCount = 0;
        for (int i = 0; i < dofCount; ++i) {
            if (dofActiveFlags[i] != 0.0f)
                activeDofCount = 1;
        }

        if (!activeDofCount) {
            result = CKERR_INVALIDOPERATION;
            goto CLEANUP_AND_RETURN;
        }

        // SVD decomposition
        SVDDecompose(jacobianAxes, dofType, dofCount, singularValues, vMatrix);

        // Check and handle near-zero singular values
        limitViolated = 0;
        memset(lockedAxes, 0, sizeof(float) * dofCount);

        for (int i = 0; i < dofCount; ++i) {
            if (fabsf(singularValues[i]) < 0.00001f) {
                singularValues[i] = 0.0f;
                lockedAxes[i] = 1.0f;
                ++limitViolated;
            }
        }

        // Solve using SVD
        SVDSolve(jacobianAxes, singularValues, vMatrix, dofType, dofCount, targetVecArray, computedAngleDeltas);

        // Save current transforms
        jointIdx = 0;
        dofIdx = 0;
        while (jointIdx < chainBodyCount - 1) {
            m_ChainData[jointIdx].m_LocalTransform = m_ChainData[jointIdx].m_BodyPart->GetLocalMatrix();
            ++jointIdx;
            dofIdx += 3;
        }

        // Apply angle deltas
        VxMatrix rotMatrix;
        Vx3DMatrixIdentity(rotMatrix);

        jointIdx = 0;
        dofIdx = 0;
        limitViolated = 0;

        while (jointIdx < chainBodyCount - 1) {
            VxVector angleDeltas(0.0f, 0.0f, 0.0f);
            VxMatrix localMatrix = m_ChainData[jointIdx].m_BodyPart->GetLocalMatrix();
            Vx3DMatrixToEulerAngles(localMatrix, &eulerAngles.x, &eulerAngles.y, &eulerAngles.z);

            for (int axisIdx = 0; axisIdx < 3; ++axisIdx) {
                if (dofActiveFlags[dofIdx + axisIdx] != 0.0f) {
                    (&angleDeltas.x)[axisIdx] = computedAngleDeltas[dofIdx + axisIdx];
                    (&eulerAngles.x)[axisIdx] += (&angleDeltas.x)[axisIdx];

                    // Normalize angle to [-PI, PI]
                    while ((&eulerAngles.x)[axisIdx] > PI)
                        (&eulerAngles.x)[axisIdx] -= 2 * PI;
                    while ((&eulerAngles.x)[axisIdx] < -PI)
                        (&eulerAngles.x)[axisIdx] += 2 * PI;

                    // Check limits
                    CKDWORD limitBit = (16 << axisIdx);
                    if (m_ChainData[jointIdx].m_RotationJoint.m_Flags & limitBit) {
                        float minAngle = (&m_ChainData[jointIdx].m_RotationJoint.m_Min.x)[axisIdx];
                        float maxAngle = (&m_ChainData[jointIdx].m_RotationJoint.m_Max.x)[axisIdx];

                        if ((&eulerAngles.x)[axisIdx] < minAngle)
                            limitViolated = 1;
                        if ((&eulerAngles.x)[axisIdx] > maxAngle)
                            limitViolated = 1;
                    }
                }
            }

            // Apply rotation
            Vx3DMatrixFromEulerAngles(rotMatrix, angleDeltas.x, angleDeltas.y, angleDeltas.z);
            VxMatrix resultMatrix;
            Vx3DMultiplyMatrix(resultMatrix, localMatrix, rotMatrix);
            m_ChainData[jointIdx].m_BodyPart->SetLocalMatrix(resultMatrix);

            ++jointIdx;
            dofIdx += 3;
        }

        // Exit if no limits violated
        if (!limitViolated)
            break;

        // Lock most constrained axis
        ++iterationCount;
        if (iterationCount >= chainBodyCount + 1)
            break;

        int mostConstrainedDof = -1;
        float largestViolation = -10000.0f;

        jointIdx = 0;
        dofIdx = 0;
        while (jointIdx < chainBodyCount - 1) {
            // Restore original transform
            m_ChainData[jointIdx].m_BodyPart->SetLocalMatrix(m_ChainData[jointIdx].m_LocalTransform);

            Vx3DMatrixToEulerAngles(m_ChainData[jointIdx].m_LocalTransform, &eulerAngles.x, &eulerAngles.y,
                                    &eulerAngles.z);

            for (int axisIdx = 0; axisIdx < 3; ++axisIdx) {
                if (dofActiveFlags[dofIdx + axisIdx] != 0.0f) {
                    CKDWORD limitBit = (16 << axisIdx);
                    if (m_ChainData[jointIdx].m_RotationJoint.m_Flags & limitBit) {
                        float minAngle = (&m_ChainData[jointIdx].m_RotationJoint.m_Min.x)[axisIdx];
                        float maxAngle = (&m_ChainData[jointIdx].m_RotationJoint.m_Max.x)[axisIdx];
                        float midAngle = (minAngle + maxAngle) / 2.0f;
                        float distFromMid = fabsf((&eulerAngles.x)[axisIdx] - midAngle);

                        if (distFromMid > largestViolation) {
                            mostConstrainedDof = dofIdx + axisIdx;
                            largestViolation = distFromMid;
                        }
                    }
                }
            }

            ++jointIdx;
            dofIdx += 3;
        }

        if (mostConstrainedDof >= 0) {
            lockedAxes[mostConstrainedDof] = 1.0f;
        }
    }

CLEANUP_AND_RETURN:
    // Free allocated arrays
    for (int i = 0; i < dofType; ++i)
        delete[] jacobianAxes[i];
    delete[] jacobianAxes;

    for (int i = 0; i < dofCount; ++i)
        delete[] vMatrix[i];
    delete[] vMatrix;

    delete[] singularValues;
    delete[] lockedAxes;
    delete[] targetVecArray;
    delete[] computedAngleDeltas;
    delete[] dofActiveFlags;

    return result;
}

/**
 * SVDDecompose - Based on IDA decompilation at 0x10054D88
 * Singular Value Decomposition using Householder reduction.
 */
void RCKKinematicChain::SVDDecompose(float **a, int m, int n, float *w, float **v) {
    float *rv1 = new float[n];
    float anorm = 0.0f;
    float scale = 0.0f;
    float g = 0.0f;

    // Householder reduction to bidiagonal form
    for (int i = 0; i < n; ++i) {
        int l = i + 1;
        rv1[i] = scale * g;
        g = 0.0f;
        float s = 0.0f;
        scale = 0.0f;

        if (i < m) {
            for (int k = i; k < m; ++k)
                scale += fabsf(a[k][i]);

            if (scale != 0.0f) {
                for (int k = i; k < m; ++k) {
                    a[k][i] /= scale;
                    s += a[k][i] * a[k][i];
                }

                float f = a[i][i];
                g = (f >= 0.0f) ? -fabsf(sqrtf(s)) : fabsf(sqrtf(s));
                float h = f * g - s;
                a[i][i] = f - g;

                for (int j = l; j < n; ++j) {
                    s = 0.0f;
                    for (int k = i; k < m; ++k)
                        s += a[k][i] * a[k][j];
                    f = s / h;
                    for (int k = i; k < m; ++k)
                        a[k][j] += f * a[k][i];
                }

                for (int k = i; k < m; ++k)
                    a[k][i] *= scale;
            }
        }

        w[i] = scale * g;
        g = 0.0f;
        s = 0.0f;
        scale = 0.0f;

        if (i < m && i != n - 1) {
            for (int k = l; k < n; ++k)
                scale += fabsf(a[i][k]);

            if (scale != 0.0f) {
                for (int k = l; k < n; ++k) {
                    a[i][k] /= scale;
                    s += a[i][k] * a[i][k];
                }

                float f = a[i][l];
                g = (f >= 0.0f) ? -fabsf(sqrtf(s)) : fabsf(sqrtf(s));
                float h = f * g - s;
                a[i][l] = f - g;

                for (int k = l; k < n; ++k)
                    rv1[k] = a[i][k] / h;

                for (int j = l; j < m; ++j) {
                    s = 0.0f;
                    for (int k = l; k < n; ++k)
                        s += a[j][k] * a[i][k];
                    for (int k = l; k < n; ++k)
                        a[j][k] += s * rv1[k];
                }

                for (int k = l; k < n; ++k)
                    a[i][k] *= scale;
            }
        }

        float tmp = fabsf(w[i]) + fabsf(rv1[i]);
        if (tmp > anorm)
            anorm = tmp;
    }

    // Accumulation of right-hand transformations
    for (int i = n - 1; i >= 0; --i) {
        int l = i + 1;
        if (i < n - 1) {
            if (g != 0.0f) {
                for (int j = l; j < n; ++j)
                    v[j][i] = (a[i][j] / a[i][l]) / g;

                for (int j = l; j < n; ++j) {
                    float s = 0.0f;
                    for (int k = l; k < n; ++k)
                        s += a[i][k] * v[k][j];
                    for (int k = l; k < n; ++k)
                        v[k][j] += s * v[k][i];
                }
            }

            for (int j = l; j < n; ++j) {
                v[j][i] = 0.0f;
                v[i][j] = 0.0f;
            }
        }

        v[i][i] = 1.0f;
        g = rv1[i];
    }

    // Accumulation of left-hand transformations
    int minmn = (m < n) ? m : n;
    for (int i = minmn - 1; i >= 0; --i) {
        int l = i + 1;
        g = w[i];

        for (int j = l; j < n; ++j)
            a[i][j] = 0.0f;

        if (g == 0.0f) {
            for (int j = i; j < m; ++j)
                a[j][i] = 0.0f;
        } else {
            g = 1.0f / g;
            for (int j = l; j < n; ++j) {
                float s = 0.0f;
                for (int k = l; k < m; ++k)
                    s += a[k][i] * a[k][j];
                float f = (s / a[i][i]) * g;
                for (int k = i; k < m; ++k)
                    a[k][j] += f * a[k][i];
            }

            for (int j = i; j < m; ++j)
                a[j][i] *= g;
        }

        a[i][i] += 1.0f;
    }

    // Diagonalization of the bidiagonal form
    for (int k = n - 1; k >= 0; --k) {
        for (int its = 1; its <= 30; ++its) {
            int flag = 1;
            int l, nm;

            for (l = k; l >= 0; --l) {
                nm = l - 1;
                if (fabsf(rv1[l]) + anorm == anorm) {
                    flag = 0;
                    break;
                }
                if (nm >= 0 && fabsf(w[nm]) + anorm == anorm)
                    break;
            }

            if (flag) {
                float c = 0.0f;
                float s = 1.0f;
                for (int i = l; i <= k; ++i) {
                    float f = s * rv1[i];
                    rv1[i] = c * rv1[i];
                    if (fabsf(f) + anorm == anorm)
                        break;

                    g = w[i];
                    float h = sqrtf(f * f + g * g);
                    w[i] = h;
                    h = 1.0f / h;
                    c = g * h;
                    s = -f * h;

                    for (int j = 0; j < m; ++j) {
                        float y = a[j][nm];
                        float z = a[j][i];
                        a[j][nm] = y * c + z * s;
                        a[j][i] = z * c - y * s;
                    }
                }
            }

            float z = w[k];
            if (l == k) {
                if (z < 0.0f) {
                    w[k] = -z;
                    for (int j = 0; j < n; ++j)
                        v[j][k] = -v[j][k];
                }
                break;
            }

            if (its == 30) {
                delete[] rv1;
                return;
            }

            float x = w[l];
            nm = k - 1;
            float y = (nm >= 0) ? w[nm] : 0.0f;
            g = (nm >= 0) ? rv1[nm] : 0.0f;
            float h = rv1[k];
            float f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0f * h * y);
            g = sqrtf(f * f + 1.0f);
            float sign_g = (f >= 0.0f) ? fabsf(g) : -fabsf(g);
            f = ((x - z) * (x + z) + h * ((y / (f + sign_g)) - h)) / x;

            float c = 1.0f;
            float s = 1.0f;

            for (int j = l; j <= nm; ++j) {
                int i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = sqrtf(f * f + h * h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;

                for (int jj = 0; jj < n; ++jj) {
                    x = v[jj][j];
                    z = v[jj][i];
                    v[jj][j] = x * c + z * s;
                    v[jj][i] = z * c - x * s;
                }

                z = sqrtf(f * f + h * h);
                w[j] = z;
                if (z != 0.0f) {
                    z = 1.0f / z;
                    c = f * z;
                    s = h * z;
                }

                f = c * g + s * y;
                x = c * y - s * g;

                for (int jj = 0; jj < m; ++jj) {
                    y = a[jj][j];
                    z = a[jj][i];
                    a[jj][j] = y * c + z * s;
                    a[jj][i] = z * c - y * s;
                }
            }

            rv1[l] = 0.0f;
            rv1[k] = f;
            w[k] = x;
        }
    }

    delete[] rv1;
}

/**
 * SVDSolve - Based on IDA decompilation at 0x10055D21
 * Solve linear system using SVD results.
 */
void RCKKinematicChain::SVDSolve(float **u, float *w, float **v, int m, int n, float *b, float *x) {
    float *tmp = new float[n];

    for (int j = 0; j < n; ++j) {
        float s = 0.0f;
        if (w[j] != 0.0f) {
            for (int i = 0; i < m; ++i)
                s += u[i][j] * b[i];
            s /= w[j];
        }
        tmp[j] = s;
    }

    for (int j = 0; j < n; ++j) {
        float s = 0.0f;
        for (int jj = 0; jj < n; ++jj)
            s += v[j][jj] * tmp[jj];
        x[j] = s;
    }

    delete[] tmp;
}
