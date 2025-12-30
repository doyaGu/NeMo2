#include "RCKSkin.h"

#include "VxMath.h"
#include "VxMatrix.h"
#include "CK3dEntity.h"

//-----------------------------------------------------------------------------
// RCKSkinBoneData Implementation
//-----------------------------------------------------------------------------

RCKSkinBoneData::RCKSkinBoneData() : m_Bone(nullptr), m_BoneFlags(0) {
    Vx3DMatrixIdentity(m_InitialInverseMatrix);
    Vx3DMatrixIdentity(m_TransformMatrix);
}

void RCKSkinBoneData::SetBone(CK3dEntity *ent) {
    m_Bone = ent;
}

CK3dEntity *RCKSkinBoneData::GetBone() {
    return m_Bone;
}

void RCKSkinBoneData::SetBoneInitialInverseMatrix(const VxMatrix &M) {
    m_InitialInverseMatrix = M;
}

//-----------------------------------------------------------------------------
// RCKSkinVertexData Implementation
//-----------------------------------------------------------------------------

RCKSkinVertexData::RCKSkinVertexData()
    : m_BoneCount(0), m_Bones(nullptr), m_Weights(nullptr), m_InitialPos(0.0f, 0.0f, 0.0f) {}

RCKSkinVertexData::~RCKSkinVertexData() {
    RCKSkinVertexData::SetBoneCount(0);
}

void RCKSkinVertexData::SetBoneCount(int BoneCount) {
    int *oldBones = m_Bones;
    float *oldWeights = m_Weights;

    if (BoneCount <= 0) {
        m_Bones = nullptr;
        m_Weights = nullptr;
    } else {
        // Allocate new arrays
        m_Bones = new int[BoneCount];
        m_Weights = new float[BoneCount];

        // Copy existing data if any
        if (m_BoneCount > BoneCount)
            m_BoneCount = BoneCount;

        if (m_BoneCount > 0) {
            memcpy(m_Bones, oldBones, m_BoneCount * sizeof(int));
            memcpy(m_Weights, oldWeights, m_BoneCount * sizeof(float));
        }
    }

    // Free old arrays
    delete[] oldBones;
    delete[] oldWeights;

    m_BoneCount = BoneCount;
}

int RCKSkinVertexData::GetBoneCount() {
    return m_BoneCount;
}

int RCKSkinVertexData::GetBone(int n) {
    return m_Bones[n];
}

void RCKSkinVertexData::SetBone(int n, int BoneIdx) {
    m_Bones[n] = BoneIdx;
}

float RCKSkinVertexData::GetWeight(int n) {
    return m_Weights[n];
}

void RCKSkinVertexData::SetWeight(int n, float Weight) {
    m_Weights[n] = Weight;
}

VxVector &RCKSkinVertexData::GetInitialPos() {
    return m_InitialPos;
}

void RCKSkinVertexData::SetInitialPos(VxVector &pos) {
    m_InitialPos = pos;
}

//-----------------------------------------------------------------------------
// RCKSkin Implementation
//-----------------------------------------------------------------------------

RCKSkin::RCKSkin()
    : m_Flags(0) {
    Vx3DMatrixIdentity(m_ObjectInitMatrix);
    Vx3DMatrixIdentity(m_InverseWorldMatrix);
}

RCKSkin::~RCKSkin() {
    // XClassArrays will clean up their contents automatically
}

void RCKSkin::SetObjectInitMatrix(const VxMatrix &Mat) {
    m_ObjectInitMatrix = Mat;
    Vx3DInverseMatrix(m_InverseWorldMatrix, m_ObjectInitMatrix);
}

void RCKSkin::SetBoneCount(int BoneCount) {
    // Clear cached point lists if they exist
    if (m_Points.Size() > 0)
        ClearBonePointLists();

    int currentCount = m_BoneData.Size();
    if (BoneCount != currentCount) {
        if (BoneCount <= 0) {
            m_BoneData.Clear();
        } else {
            m_BoneData.Resize(BoneCount);
        }
    }
}

void RCKSkin::SetVertexCount(int Count) {
    // Clear cached point lists if they exist
    if (m_Points.Size() > 0)
        ClearBonePointLists();

    int currentCount = m_VertexData.Size();
    if (Count != currentCount) {
        if (Count <= 0) {
            m_VertexData.Clear();
        } else {
            m_VertexData.Resize(Count);
        }
    }
}

int RCKSkin::GetBoneCount() {
    return m_BoneData.Size();
}

int RCKSkin::GetVertexCount() {
    return m_VertexData.Size();
}

void RCKSkin::ConstructBoneTransfoMatrices(CKContext *context) {
    int boneCount = m_BoneData.Size();
    for (int i = 0; i < boneCount; ++i) {
        RCKSkinBoneData &boneData = m_BoneData[i];
        CK3dEntity *bone = boneData.GetBone();

        if (bone) {
            // Get bone's current world matrix
            const VxMatrix &boneWorldMatrix = bone->GetWorldMatrix();

            // Compute: InverseWorld * BoneWorld -> temp
            VxMatrix temp;
            Vx3DMultiplyMatrix(temp, m_InverseWorldMatrix, boneWorldMatrix);

            // Store in transform matrix
            boneData.GetTransformMatrix() = temp;

            // Multiply by bone's initial inverse matrix: TransformMatrix * InitialInverse
            VxMatrix &transformMatrix = boneData.GetTransformMatrix();
            Vx3DMultiplyMatrix(transformMatrix, transformMatrix, boneData.GetInitialInverseMatrix());

            // Multiply by object init matrix: TransformMatrix * ObjectInit
            Vx3DMultiplyMatrix(transformMatrix, transformMatrix, m_ObjectInitMatrix);
        } else {
            // Bone entity is invalid, clear the reference
            boneData.SetBone(nullptr);
        }
    }
}

CKBOOL RCKSkin::CalcPointsEx(int VertexCount, CKBYTE *VertexPtr, CKDWORD VStride, CKBYTE *NormalPtr, CKDWORD NStride) {
    if (!VertexPtr)
        return FALSE;

    int skinVertexCount = m_VertexData.Size();
    if (skinVertexCount == 0)
        return FALSE;

    if (VertexCount == 0)
        return FALSE;

    // Build bone point lists if not already done
    if (m_Points.Size() == 0)
        BuildBonePointLists();

    // Initialize output arrays
    if (m_Flags & 1) {
        // Weighted mode: handle vertices individually based on bone count
        for (int i = 0; i < skinVertexCount; ++i) {
            VxVector *outPos = (VxVector *) (VertexPtr + i * VStride);
            RCKSkinVertexData &vd = m_VertexData[i];
            int boneCount = vd.GetBoneCount();

            if (boneCount == 0) {
                *outPos = vd.GetInitialPos();
            } else if (boneCount == 1 && vd.GetWeight(0) == 1.0f) {
                // Single bone with full weight - clear for accumulation
                outPos->x = 0.0f;
                outPos->y = 0.0f;
                outPos->z = 0.0f;
            } else {
                // Multiple bones or partial weight - start with remainder
                float totalWeight = 0.0f;
                for (int b = 0; b < boneCount; ++b)
                    totalWeight += vd.GetWeight(b);

                float remainder = 1.0f - totalWeight;
                if (remainder > 0.0f) {
                    VxVector &initPos = vd.GetInitialPos();
                    outPos->x = initPos.x * remainder;
                    outPos->y = initPos.y * remainder;
                    outPos->z = initPos.z * remainder;
                } else {
                    outPos->x = 0.0f;
                    outPos->y = 0.0f;
                    outPos->z = 0.0f;
                }
            }
        }
    } else {
        // Standard mode: zero all vertices
        VxVector zero(0.0f, 0.0f, 0.0f);
        VxFillStructure(VertexCount, VertexPtr, VStride, sizeof(VxVector), &zero);
    }

    // Zero normals if provided
    if (m_Normals.Size() > 0 && NormalPtr) {
        VxVector zero(0.0f, 0.0f, 0.0f);
        VxFillStructure(VertexCount, NormalPtr, NStride, sizeof(VxVector), &zero);
    }

    bool anyBoneProcessed = false;
    int boneCount = m_BoneData.Size();

    // Process each bone
    for (int boneIdx = 0; boneIdx < boneCount; ++boneIdx) {
        RCKSkinBonePoints &bonePoints = m_Points[boneIdx];
        int pointCount = bonePoints.m_WeightedVertices.Size();

        if (pointCount == 0)
            continue;

        RCKSkinBoneData &boneData = m_BoneData[boneIdx];
        CK3dEntity *bone = boneData.GetBone();

        if (!bone) {
            boneData.SetBone(nullptr);
            continue;
        }

        // Update bone transform matrix in CalcPoints (like original does)
        const VxMatrix &boneWorldMatrix = bone->GetWorldMatrix();
        VxMatrix temp;
        Vx3DMultiplyMatrix(temp, m_InverseWorldMatrix, boneWorldMatrix);
        boneData.GetTransformMatrix() = temp;
        Vx3DMultiplyMatrix(boneData.GetTransformMatrix(), boneData.GetTransformMatrix(), boneData.GetInitialInverseMatrix());
        Vx3DMultiplyMatrix(boneData.GetTransformMatrix(), boneData.GetTransformMatrix(), m_ObjectInitMatrix);

        anyBoneProcessed = true;
        const VxMatrix &transformMatrix = boneData.GetTransformMatrix();

        // Transform vertices affected by this bone
        for (int p = 0; p < pointCount; ++p) {
            CKWORD vertexIdx = bonePoints.m_VertexIndices[p];
            if (vertexIdx >= VertexCount)
                continue;

            VxVector4 &weightedVertex = bonePoints.m_WeightedVertices[p];
            VxVector *outPos = (VxVector *) (VertexPtr + vertexIdx * VStride);

            // Transform position
            VxVector transformed;
            VxVector srcPos(weightedVertex.x, weightedVertex.y, weightedVertex.z);
            Vx3DMultiplyMatrixVector(&transformed, transformMatrix, &srcPos);

            // Accumulate weighted result
            float weight = weightedVertex.w;
            outPos->x += transformed.x * weight;
            outPos->y += transformed.y * weight;
            outPos->z += transformed.z * weight;
        }

        // Transform normals if provided
        if (NormalPtr && bonePoints.m_WeightedNormals.Size() > 0) {
            int normalCount = bonePoints.m_WeightedNormals.Size();
            for (int n = 0; n < normalCount; ++n) {
                CKWORD vertexIdx = bonePoints.m_VertexIndices[n];
                if (vertexIdx >= VertexCount)
                    continue;

                VxVector4 &weightedNormal = bonePoints.m_WeightedNormals[n];
                VxVector *outNorm = (VxVector *) (NormalPtr + vertexIdx * NStride);

                // Rotate normal (no translation)
                VxVector transformed;
                VxVector srcNorm(weightedNormal.x, weightedNormal.y, weightedNormal.z);
                Vx3DRotateVector(&transformed, transformMatrix, &srcNorm);

                // Accumulate weighted result
                float weight = weightedNormal.w;
                outNorm->x += transformed.x * weight;
                outNorm->y += transformed.y * weight;
                outNorm->z += transformed.z * weight;
            }
        }
    }

    // If no bones were processed, copy initial positions directly
    if (!anyBoneProcessed) {
        for (int i = 0; i < skinVertexCount && i < VertexCount; ++i) {
            VxVector *outPos = (VxVector *) (VertexPtr + i * VStride);
            *outPos = m_VertexData[i].GetInitialPos();
        }
    }

    return TRUE;
}

CKBOOL RCKSkin::CalcPoints(int VertexCount, CKBYTE *VertexPtr, CKDWORD VStride) {
    return CalcPointsEx(VertexCount, VertexPtr, VStride, nullptr, 0);
}

CKSkinBoneData *RCKSkin::GetBoneData(int BoneIdx) {
    return &m_BoneData[BoneIdx];
}

CKSkinVertexData *RCKSkin::GetVertexData(int VertexIdx) {
    return &m_VertexData[VertexIdx];
}

void RCKSkin::RemapVertices(const XArray<int> &permutation) {
    int vertexCount = m_VertexData.Size();
    if ((int) permutation.Size() != vertexCount)
        return;

    // Clear cached point lists
    ClearBonePointLists();

    // Create new vertex data array
    XClassArray<RCKSkinVertexData> newVertexData;
    newVertexData.Resize(vertexCount);

    // Count valid vertices and remap
    int validCount = 0;
    for (int i = 0; i < vertexCount; ++i) {
        if (permutation[i] >= 0) {
            // Copy vertex data to new position
            int newIdx = permutation[i];
            newVertexData[newIdx] = m_VertexData[i];
            ++validCount;
        }
    }

    // Resize to actual count
    newVertexData.Resize(validCount);
    m_VertexData = newVertexData;

    // Clear bone/weight pointers in old array positions
    for (int i = 0; i < vertexCount; ++i) {
        // The old array elements will be destroyed, so we don't need to clear manually
    }

    // Remap normals if present
    if (m_Normals.Size() > 0) {
        XClassArray<VxVector> newNormals;
        newNormals.Resize(vertexCount);

        int normalCount = 0;
        for (int j = 0; j < vertexCount; ++j) {
            if (permutation[j] >= 0) {
                int newIdx = permutation[j];
                newNormals[newIdx] = m_Normals[j];
                ++normalCount;
            }
        }

        newNormals.Resize(normalCount);
        m_Normals = newNormals;
    }

    // Rebuild bone point lists
    BuildBonePointLists();
}

void RCKSkin::SetNormalCount(int Count) {
    m_Normals.Resize(Count);
}

int RCKSkin::GetNormalCount() {
    return m_Normals.Size();
}

void RCKSkin::SetNormal(int Index, const VxVector &Norm) {
    m_Normals[Index] = Norm;
}

VxVector &RCKSkin::GetNormal(int Index) {
    return m_Normals[Index];
}

void RCKSkin::ClearBonePointLists() {
    m_Points.Clear();
}

void RCKSkin::BuildBonePointLists() {
    int boneCount = m_BoneData.Size();
    int vertexCount = m_VertexData.Size();
    bool hasNormals = m_Normals.Size() > 0;

    // Allocate bone point arrays
    m_Points.Resize(boneCount);

    // First pass: count vertices per bone
    XArray<int> vertexCounts(boneCount);
    for (int i = 0; i < boneCount; ++i)
        vertexCounts[i] = 0;

    for (int v = 0; v < vertexCount; ++v) {
        RCKSkinVertexData &vd = m_VertexData[v];
        int numBones = vd.GetBoneCount();
        for (int b = 0; b < numBones; ++b) {
            int boneIdx = vd.GetBone(b);
            if (boneIdx >= 0 && boneIdx < boneCount)
                vertexCounts[boneIdx]++;
        }
    }

    // Allocate arrays for each bone
    for (int i = 0; i < boneCount; ++i) {
        m_Points[i].m_WeightedVertices.Resize(vertexCounts[i]);
        if (hasNormals)
            m_Points[i].m_WeightedNormals.Resize(vertexCounts[i]);
        m_Points[i].m_VertexIndices.Resize(vertexCounts[i]);
        vertexCounts[i] = 0; // Reset for use as insertion index
    }

    // Second pass: fill arrays
    for (int v = 0; v < vertexCount; ++v) {
        RCKSkinVertexData &vd = m_VertexData[v];
        int numBones = vd.GetBoneCount();
        VxVector &initPos = vd.GetInitialPos();

        for (int b = 0; b < numBones; ++b) {
            int boneIdx = vd.GetBone(b);
            if (boneIdx >= 0 && boneIdx < boneCount) {
                float weight = vd.GetWeight(b);
                int idx = vertexCounts[boneIdx]++;

                // Store weighted vertex (position with weight in w)
                VxVector4 &wv = m_Points[boneIdx].m_WeightedVertices[idx];
                wv.x = initPos.x;
                wv.y = initPos.y;
                wv.z = initPos.z;
                wv.w = weight;

                // Store weighted normal if available
                if (hasNormals) {
                    VxVector &norm = m_Normals[v];
                    VxVector4 &wn = m_Points[boneIdx].m_WeightedNormals[idx];
                    wn.x = norm.x;
                    wn.y = norm.y;
                    wn.z = norm.z;
                    wn.w = weight;
                }

                // Store vertex index
                m_Points[boneIdx].m_VertexIndices[idx] = (CKWORD) v;
            }
        }
    }
}

CKBOOL RCKSkin::CalcLocalBBox(CKContext *context, VxBbox *bbox) {
    if (!bbox)
        return FALSE;

    const int skinVertexCount = m_VertexData.Size();
    if (skinVertexCount <= 0)
        return FALSE;

    // Build bone point lists if not already done
    if (m_Points.Size() == 0)
        BuildBonePointLists();

    XClassArray<VxVector> outPositions;
    outPositions.Resize(skinVertexCount);

    // Initialize output array similarly to CalcPoints
    if (m_Flags & 1) {
        for (int i = 0; i < skinVertexCount; ++i) {
            RCKSkinVertexData &vd = m_VertexData[i];
            const int boneCount = vd.GetBoneCount();
            if (boneCount == 0) {
                outPositions[i] = vd.GetInitialPos();
            } else if (boneCount == 1 && vd.GetWeight(0) == 1.0f) {
                outPositions[i].x = 0.0f;
                outPositions[i].y = 0.0f;
                outPositions[i].z = 0.0f;
            } else {
                float totalWeight = 0.0f;
                for (int b = 0; b < boneCount; ++b)
                    totalWeight += vd.GetWeight(b);

                const float remainder = 1.0f - totalWeight;
                if (remainder > 0.0f) {
                    const VxVector &initPos = vd.GetInitialPos();
                    outPositions[i].x = initPos.x * remainder;
                    outPositions[i].y = initPos.y * remainder;
                    outPositions[i].z = initPos.z * remainder;
                } else {
                    outPositions[i].x = 0.0f;
                    outPositions[i].y = 0.0f;
                    outPositions[i].z = 0.0f;
                }
            }
        }
    } else {
        VxVector zero(0.0f, 0.0f, 0.0f);
        VxFillStructure(skinVertexCount, (CKBYTE *) outPositions.Begin(), sizeof(VxVector), sizeof(VxVector), &zero);
    }

    bool anyBoneProcessed = false;
    const int boneCount = m_BoneData.Size();

    for (int boneIdx = 0; boneIdx < boneCount; ++boneIdx) {
        RCKSkinBonePoints &bonePoints = m_Points[boneIdx];
        const int pointCount = bonePoints.m_WeightedVertices.Size();
        if (pointCount == 0)
            continue;

        RCKSkinBoneData &boneData = m_BoneData[boneIdx];
        CK3dEntity *bone = boneData.GetBone();
        if (!bone) {
            boneData.SetBone(nullptr);
            continue;
        }

        // Keep transform-matrix construction aligned with CalcPoints
        const VxMatrix &boneWorldMatrix = bone->GetWorldMatrix();
        VxMatrix temp;
        Vx3DMultiplyMatrix(temp, m_InverseWorldMatrix, boneWorldMatrix);
        boneData.GetTransformMatrix() = temp;
        Vx3DMultiplyMatrix(boneData.GetTransformMatrix(), boneData.GetTransformMatrix(), boneData.GetInitialInverseMatrix());
        Vx3DMultiplyMatrix(boneData.GetTransformMatrix(), boneData.GetTransformMatrix(), m_ObjectInitMatrix);

        anyBoneProcessed = true;
        const VxMatrix &transformMatrix = boneData.GetTransformMatrix();

        for (int p = 0; p < pointCount; ++p) {
            const CKWORD vertexIdx = bonePoints.m_VertexIndices[p];
            if ((int) vertexIdx >= skinVertexCount)
                continue;

            VxVector4 &weightedVertex = bonePoints.m_WeightedVertices[p];
            VxVector srcPos(weightedVertex.x, weightedVertex.y, weightedVertex.z);

            VxVector transformed;
            Vx3DMultiplyMatrixVector(&transformed, transformMatrix, &srcPos);

            const float weight = weightedVertex.w;
            outPositions[vertexIdx].x += transformed.x * weight;
            outPositions[vertexIdx].y += transformed.y * weight;
            outPositions[vertexIdx].z += transformed.z * weight;
        }
    }

    if (!anyBoneProcessed) {
        for (int i = 0; i < skinVertexCount; ++i)
            outPositions[i] = m_VertexData[i].GetInitialPos();
    }

    bbox->Reset();
    for (int i = 0; i < skinVertexCount; ++i)
        bbox->Merge(outPositions[i]);

    return TRUE;
}

CKBOOL RCKSkin::CalcBonesBBox(CKContext *context, CK3dEntity *owner, VxBbox *bbox) {
    (void) context;
    if (!bbox)
        return FALSE;

    bbox->Reset();

    const VxMatrix *invOwner = nullptr;
    if (owner) {
        invOwner = &owner->GetInverseWorldMatrix();
    }

    const int boneCount = m_BoneData.Size();
    VxVector tmp;
    for (int i = 0; i < boneCount; ++i) {
        CK3dEntity *bone = m_BoneData[i].GetBone();
        if (!bone)
            continue;

        const VxMatrix &boneWorld = bone->GetWorldMatrix();
        const VxVector bonePos(boneWorld[3][0], boneWorld[3][1], boneWorld[3][2]);

        if (invOwner) {
            Vx3DMultiplyMatrixVector(&tmp, *invOwner, &bonePos);
            bbox->Merge(tmp);
        } else {
            bbox->Merge(bonePos);
        }
    }

    return TRUE;
}
