#include "RCKObjectAnimation.h"

#include "VxMath.h"
#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "CKMemoryPool.h"
#include "CKBodyPart.h"
#include "CKSceneGraph.h"
#include "RCKKeyedAnimation.h"
#include "RCK3dEntity.h"
#include "RCKMesh.h"

CK_CLASSID RCKObjectAnimation::m_ClassID = CKCID_OBJECTANIMATION;

//=============================================================================
// Constructor/Destructor
//=============================================================================

RCKObjectAnimation::RCKObjectAnimation(CKContext *Context, CKSTRING name)
    : CKObjectAnimation(Context, name),
      m_KeyframeData(nullptr),
      m_Flags(0),
      m_Entity(nullptr),
      m_CurrentStep(0.0f),
      m_MergeFactor(0.5f),
      m_Anim1(nullptr),
      m_Anim2(nullptr),
      m_field_38(0),
      m_ParentKeyedAnimation(nullptr) {
    // Allocate and initialize keyframe data
    m_KeyframeData = new CKKeyframeData();
    if (m_KeyframeData) {
        m_KeyframeData->m_ObjectAnimation = this;
    }
}

RCKObjectAnimation::~RCKObjectAnimation() {
    // Handle keyframe data reference counting and ownership transfer
    // Based on IDA decompilation at 0x10056A50
    if (m_KeyframeData) {
        // Decrement reference count
        --m_KeyframeData->m_RefCount;

        if (m_KeyframeData->m_RefCount > 0) {
            // Other animations are sharing this data
            // If we are the current owner, transfer ownership to another animation
            if (m_KeyframeData->m_ObjectAnimation == this) {
                // Find another animation that shares this keyframe data
                int count = m_Context->GetObjectsCountByClassID(CKCID_OBJECTANIMATION);
                CK_ID *ids = m_Context->GetObjectsListByClassID(CKCID_OBJECTANIMATION);

                for (int i = 0; i < count; ++i) {
                    RCKObjectAnimation *other = static_cast<RCKObjectAnimation *>(m_Context->GetObjectA(ids[i]));
                    if (other && other->m_KeyframeData == m_KeyframeData && other != this) {
                        // Transfer ownership
                        m_KeyframeData->m_ObjectAnimation = other;
                        break;
                    }
                }
            }
        } else {
            // We're the last user - delete the keyframe data
            if (m_KeyframeData) {
                delete m_KeyframeData;
            }
        }
        m_KeyframeData = nullptr;
    }
}

CK_CLASSID RCKObjectAnimation::GetClassID() {
    return m_ClassID;
}

//=============================================================================
// Serialization
//=============================================================================

CKStateChunk *RCKObjectAnimation::Save(CKFile *file, CKDWORD flags) {
    // Call base class save first
    CKStateChunk *baseChunk = CKObject::Save(file, flags);

    // If no file and no flags set, just return base chunk
    if (!file && !(flags & CK_STATESAVE_OBJANIMALL))
        return baseChunk;

    // Create new chunk for ObjectAnimation data
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_OBJECTANIMATION, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    // Root position vector (stored in parent keyed animation when this is the root animation)
    VxVector rootPos(0.0f);
    if (m_ParentKeyedAnimation && m_ParentKeyedAnimation->GetRootAnimationInternal() == this)
        rootPos = m_ParentKeyedAnimation->GetRootVectorInternal();

    // Check if we should share animation data with another ObjectAnimation
    RCKObjectAnimation *sharedAnim = nullptr;

    if (m_KeyframeData && m_KeyframeData->m_ObjectAnimation && m_KeyframeData->m_ObjectAnimation != this) {
        CKObjectAnimation *owner = m_KeyframeData->m_ObjectAnimation;
        if (file) {
            // Only reference shared data if the owner object is actually part of the file save set.
            if (file->IsObjectToBeSaved(owner->GetID()))
                sharedAnim = static_cast<RCKObjectAnimation *>(owner);
        } else {
            sharedAnim = static_cast<RCKObjectAnimation *>(owner);
        }
    }

    if (sharedAnim) {
        // Write reference to shared animation data (identifier 0x2000000)
        chunk->WriteIdentifier(CK_STATESAVE_OBJANIMSHARED);
        chunk->WriteObject(sharedAnim);
        chunk->WriteFloat(rootPos.x);
        chunk->WriteFloat(rootPos.y);
        chunk->WriteFloat(rootPos.z);
        // Write 4 reserved floats
        chunk->WriteFloat(0.0f);
        chunk->WriteFloat(0.0f);
        chunk->WriteFloat(0.0f);
        chunk->WriteFloat(0.0f);
        chunk->WriteDword(m_Flags);
        chunk->WriteObject(m_Entity);

        // Write merged animation data if flag CK_OBJECTANIMATION_MERGED is set
        if (IsMerged()) {
            chunk->WriteFloat(m_MergeFactor);
            chunk->WriteObject(m_Anim1);
            chunk->WriteObject(m_Anim2);
        }
    } else {
        // Mark this as the owner of the keyframe data when saving
        if (file && m_KeyframeData) {
            m_KeyframeData->m_ObjectAnimation = this;
        }

        // Write full keyframe data (identifier 0x4000000)
        chunk->WriteIdentifier(CK_STATESAVE_OBJANIMCONTROLLERS);
        chunk->WriteFloat(rootPos.x);
        chunk->WriteFloat(rootPos.y);
        chunk->WriteFloat(rootPos.z);
        // Write 4 reserved floats
        chunk->WriteFloat(0.0f);
        chunk->WriteFloat(0.0f);
        chunk->WriteFloat(0.0f);
        chunk->WriteFloat(0.0f);
        chunk->WriteDword(m_Flags);
        chunk->WriteObject(m_Entity);

        // Write animation length from keyframe data
        if (m_KeyframeData)
            chunk->WriteFloat(m_KeyframeData->m_Length);
        else
            chunk->WriteFloat(0.0f);

        // Write merged animation data if flag CK_OBJECTANIMATION_MERGED is set
        if (IsMerged()) {
            chunk->WriteFloat(m_MergeFactor);
            chunk->WriteObject(m_Anim1);
            chunk->WriteObject(m_Anim2);
        }

        // Write controller data (position, rotation, scale, scaleAxis, morph)
        if (m_KeyframeData) {
            // Position controller
            if (m_KeyframeData->m_PositionController) {
                CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
                int size = ctrl->DumpKeysTo(nullptr) >> 2;
                if (size > 0) {
                    chunk->WriteDword(ctrl->m_Type);
                    chunk->WriteDword(size);
                    void *buffer = chunk->LockWriteBuffer(size);
                    ctrl->DumpKeysTo(buffer);
                    chunk->Skip(size);
                }
            }

            // Rotation controller
            if (m_KeyframeData->m_RotationController) {
                CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
                int size = ctrl->DumpKeysTo(nullptr) >> 2;
                if (size > 0) {
                    chunk->WriteDword(ctrl->m_Type);
                    chunk->WriteDword(size);
                    void *buffer = chunk->LockWriteBuffer(size);
                    ctrl->DumpKeysTo(buffer);
                    chunk->Skip(size);
                }
            }

            // Scale controller
            if (m_KeyframeData->m_ScaleController) {
                CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
                int size = ctrl->DumpKeysTo(nullptr) >> 2;
                if (size > 0) {
                    chunk->WriteDword(ctrl->m_Type);
                    chunk->WriteDword(size);
                    void *buffer = chunk->LockWriteBuffer(size);
                    ctrl->DumpKeysTo(buffer);
                    chunk->Skip(size);
                }
            }

            // Scale axis controller
            if (m_KeyframeData->m_ScaleAxisController) {
                CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
                int size = ctrl->DumpKeysTo(nullptr) >> 2;
                if (size > 0) {
                    chunk->WriteDword(ctrl->m_Type);
                    chunk->WriteDword(size);
                    void *buffer = chunk->LockWriteBuffer(size);
                    ctrl->DumpKeysTo(buffer);
                    chunk->Skip(size);
                }
            }

            // Morph controller
            if (m_KeyframeData->m_MorphController) {
                CKMorphController *ctrl = reinterpret_cast<CKMorphController *>(m_KeyframeData->m_MorphController);
                int size = ctrl->DumpKeysTo(nullptr) >> 2;
                if (size > 0) {
                    chunk->WriteDword(ctrl->m_Type);
                    chunk->WriteDword(size);
                    void *buffer = chunk->LockWriteBuffer(size);
                    ctrl->DumpKeysTo(buffer);
                    chunk->Skip(size);
                }
            }
        }

        // Write terminator
        chunk->WriteDword(0);
    }

    chunk->CloseChunk();
    return chunk;
}

CKERROR RCKObjectAnimation::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Load base class data
    CKObject::Load(chunk, file);

    // Clear existing data
    Clear();

    VxVector rootPos(0.0f);

    if (chunk->GetDataVersion() >= 1) {
        // New format (data version >= 1)
        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMSHARED)) {
            // Shared animation reference format
            CKObjectAnimation *sharedAnim = (CKObjectAnimation *) chunk->ReadObject(m_Context);
            rootPos.x = chunk->ReadFloat();
            rootPos.y = chunk->ReadFloat();
            rootPos.z = chunk->ReadFloat();
            // Skip 4 reserved floats
            chunk->ReadFloat();
            chunk->ReadFloat();
            chunk->ReadFloat();
            chunk->ReadFloat();

            m_Flags = chunk->ReadDword();
            m_Entity = (RCK3dEntity *) chunk->ReadObject(m_Context);

            if (m_Flags & 0x80) {
                m_MergeFactor = chunk->ReadFloat();
                m_Anim1 = (RCKObjectAnimation *) chunk->ReadObject(m_Context);
                m_Anim2 = (RCKObjectAnimation *) chunk->ReadObject(m_Context);
            }

            // Strictly matches original: always call ShareDataFrom (nullptr creates new keyframe data)
            ShareDataFrom(sharedAnim);
        } else if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMCONTROLLERS)) {
            // Full keyframe data format
            ResetKeyframeData();

            rootPos.x = chunk->ReadFloat();
            rootPos.y = chunk->ReadFloat();
            rootPos.z = chunk->ReadFloat();
            // Skip 4 reserved floats
            chunk->ReadFloat();
            chunk->ReadFloat();
            chunk->ReadFloat();
            chunk->ReadFloat();

            m_Flags = chunk->ReadDword();
            m_Entity = (RCK3dEntity *) chunk->ReadObject(m_Context);
            float length = chunk->ReadFloat();
            SetKeyframeLength(length);

            if (m_Flags & 0x80) {
                m_MergeFactor = chunk->ReadFloat();
                m_Anim1 = (RCKObjectAnimation *) chunk->ReadObject(m_Context);
                m_Anim2 = (RCKObjectAnimation *) chunk->ReadObject(m_Context);
            }

            // Read controllers
            CKANIMATION_CONTROLLER ctrlType;
            while ((ctrlType = (CKANIMATION_CONTROLLER) chunk->ReadDword()) != 0) {
                CKAnimController *ctrl = CreateController(ctrlType);
                CKDWORD dataSize = chunk->ReadDword();
                if (ctrl) {
                    void *buffer = chunk->LockReadBuffer();
                    ctrl->ReadKeysFrom(buffer);
                }
                chunk->Skip(dataSize);
            }
        } else if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMNEWDATA)) {
            // Legacy/new snapshot format (identifier 0x1000) as implemented by CK2_3D.dll
            ResetKeyframeData();

            rootPos.x = chunk->ReadFloat();
            rootPos.y = chunk->ReadFloat();
            rootPos.z = chunk->ReadFloat();
            // Skip 4 reserved floats
            chunk->ReadFloat();
            chunk->ReadFloat();
            chunk->ReadFloat();
            chunk->ReadFloat();

            const int morphVertexCount = chunk->ReadInt();
            const int morphKeyCount = chunk->ReadInt();

            m_Flags = chunk->ReadDword();
            m_Entity = (RCK3dEntity *) chunk->ReadObject(m_Context);
            const float length = chunk->ReadFloat();
            SetKeyframeLength(length);

            if (IsMerged()) {
                m_MergeFactor = chunk->ReadFloat();
                m_Anim1 = (RCKObjectAnimation *) chunk->ReadObject(m_Context);
                m_Anim2 = (RCKObjectAnimation *) chunk->ReadObject(m_Context);
            }

            // Morph keys (positions)
            struct MorphAccess : RCKMorphController {
                using RCKMorphController::m_Keys;
                using RCKMorphController::m_VertexCount;
            };
            struct LinPosAccess : RCKLinearPositionController {
                using RCKLinearPositionController::m_Keys;
            };
            struct LinSclAccess : RCKLinearScaleController {
                using RCKLinearScaleController::m_Keys;
            };
            struct LinRotAccess : RCKLinearRotationController {
                using RCKLinearRotationController::m_Keys;
            };
            struct LinAxisAccess : RCKLinearScaleAxisController {
                using RCKLinearScaleAxisController::m_Keys;
            };

            if (morphKeyCount > 0) {
                // Strictly matches original: allocate keys array directly and read raw buffers (no AddKey sorting).
                auto *morphCtrl = static_cast<MorphAccess *>(static_cast<RCKMorphController *>(CreateController(CKANIMATION_MORPH_CONTROL)));
                if (morphCtrl) {
                    morphCtrl->m_NbKeys = morphKeyCount;
                    morphCtrl->m_VertexCount = morphVertexCount;
                    morphCtrl->SetLength(length);

                    morphCtrl->m_Keys = new CKMorphKey[morphKeyCount];
                    for (int i = 0; i < morphKeyCount; ++i) {
                        morphCtrl->m_Keys[i].TimeStep = chunk->ReadFloat();
                        morphCtrl->m_Keys[i].PosArray = nullptr;
                        morphCtrl->m_Keys[i].NormArray = nullptr;

                        const CKDWORD sizeBytes = chunk->ReadDword();
                        if (sizeBytes) {
                            const int vecCount = (int)(sizeBytes / sizeof(VxVector));
                            if (vecCount > 0) {
                                morphCtrl->m_Keys[i].PosArray = new VxVector[vecCount];
                                chunk->ReadAndFillBuffer_LEndian((int)sizeBytes, morphCtrl->m_Keys[i].PosArray);
                            }
                        }
                    }
                }
            }

            // Linear position controller
            {
                auto *posCtrl = static_cast<LinPosAccess *>(static_cast<RCKLinearPositionController *>(CreateController(CKANIMATION_LINPOS_CONTROL)));
                const CKDWORD bufSize = chunk->ReadDword();
                const CKDWORD keyCount = chunk->ReadDword();

                if (posCtrl)
                    posCtrl->m_NbKeys = (int)keyCount;

                if (posCtrl && keyCount) {
                    posCtrl->m_Keys = new CKPositionKey[keyCount];
                    chunk->ReadAndFillBuffer_LEndian((int)bufSize, posCtrl->m_Keys);
                }

                if (posCtrl && !keyCount)
                    DeleteController(CKANIMATION_LINPOS_CONTROL);
            }

            // Linear scale controller
            {
                auto *sclCtrl = static_cast<LinSclAccess *>(static_cast<RCKLinearScaleController *>(CreateController(CKANIMATION_LINSCL_CONTROL)));
                const CKDWORD bufSize = chunk->ReadDword();
                const CKDWORD keyCount = chunk->ReadDword();

                if (sclCtrl)
                    sclCtrl->m_NbKeys = (int)keyCount;

                if (sclCtrl && keyCount) {
                    sclCtrl->m_Keys = new CKScaleKey[keyCount];
                    chunk->ReadAndFillBuffer_LEndian((int)bufSize, sclCtrl->m_Keys);
                }

                if (sclCtrl && !keyCount)
                    DeleteController(CKANIMATION_LINSCL_CONTROL);
            }

            // Linear rotation controller
            {
                auto *rotCtrl = static_cast<LinRotAccess *>(static_cast<RCKLinearRotationController *>(CreateController(CKANIMATION_LINROT_CONTROL)));
                const CKDWORD rotBufSize = chunk->ReadDword();
                const CKDWORD rotKeyCount = chunk->ReadDword();

                if (rotCtrl)
                    rotCtrl->m_NbKeys = (int)rotKeyCount;

                if (rotCtrl && rotKeyCount) {
                    rotCtrl->m_Keys = new CKRotationKey[rotKeyCount];
                    chunk->ReadAndFillBuffer_LEndian((int)rotBufSize, rotCtrl->m_Keys);
                }

                if (rotCtrl && !rotKeyCount)
                    DeleteController(CKANIMATION_LINROT_CONTROL);

                // Off-axis scale axis controller is stored as float blocks (6 floats per key)
                const CKDWORD axisBufSize = chunk->ReadDword();
                const CKDWORD axisKeyCount = chunk->ReadDword();
                if (axisKeyCount) {
                    float *tmp = new float[(size_t)axisKeyCount * 6];
                    chunk->ReadAndFillBuffer_LEndian((int)axisBufSize, tmp);

                    auto *axisCtrl = static_cast<LinAxisAccess *>(static_cast<RCKLinearScaleAxisController *>(CreateController(CKANIMATION_LINSCLAXIS_CONTROL)));
                    if (axisCtrl) {
                        axisCtrl->m_NbKeys = (int)axisKeyCount;
                        axisCtrl->m_Keys = new CKScaleAxisKey[axisKeyCount];
                        for (CKDWORD i = 0; i < axisKeyCount; ++i) {
                            axisCtrl->m_Keys[i].TimeStep = tmp[i * 6 + 0];
                            axisCtrl->m_Keys[i].Rot = *(VxQuaternion *)&tmp[i * 6 + 2];
                        }
                    }
                    delete[] tmp;
                }
            }

            // Optional morph normals formats
            if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMMORPHCOMP) && m_KeyframeData && m_KeyframeData->m_MorphController) {
                CKMorphController *morphCtrl = m_KeyframeData->m_MorphController;
                for (int i = 0; i < morphCtrl->GetKeyCount(); ++i) {
                    CKMorphKey *key = static_cast<CKMorphKey *>(morphCtrl->GetKey(i));
                    if (!key)
                        continue;
                    delete[] key->NormArray;
                    key->NormArray = nullptr;

                    const CKDWORD sizeBytes = chunk->ReadDword();
                    if (!sizeBytes)
                        continue;
                    const int count = (int)(sizeBytes / sizeof(VxCompressedVector));
                    if (count <= 0)
                        continue;
                    key->NormArray = new VxCompressedVector[count];
                    chunk->ReadAndFillBuffer_LEndian16((int)sizeBytes, key->NormArray);
                }
            }

            if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMMORPHNORMALS) && m_KeyframeData && m_KeyframeData->m_MorphController) {
                CKMorphController *morphCtrl = m_KeyframeData->m_MorphController;
                for (int i = 0; i < morphCtrl->GetKeyCount(); ++i) {
                    CKMorphKey *key = static_cast<CKMorphKey *>(morphCtrl->GetKey(i));
                    if (!key)
                        continue;
                    delete[] key->NormArray;
                    key->NormArray = nullptr;

                    // Strictly matches original CK2_3D.dll (sub_1005C4B0 usage):
                    // sizeBytes -> count = sizeBytes >> 2; read sizeBytes into a temp buffer, then
                    // for each entry copy WORD[0] and WORD[2] into VxCompressedVector (xa, ya).
                    const CKDWORD sizeBytes = chunk->ReadDword();
                    if (!sizeBytes)
                        continue;
                    const int count = (int)(sizeBytes >> 2);
                    if (count <= 0)
                        continue;

                    CKBYTE *tmp = new CKBYTE[(size_t)count * 8];
                    chunk->ReadAndFillBuffer_LEndian((int)sizeBytes, tmp);

                    key->NormArray = new VxCompressedVector[count];
                    CKWORD *words = reinterpret_cast<CKWORD *>(tmp);
                    for (int v = 0; v < count; ++v) {
                        key->NormArray[v].xa = (short)words[v * 4 + 0];
                        key->NormArray[v].ya = (short)words[v * 4 + 2];
                    }
                    delete[] tmp;
                }
            }
        }
    } else {
        // Very old format (data version < 1)
        ResetKeyframeData();

        chunk->SeekIdentifier(CK_STATESAVE_OBJANIMMORPHKEYS); // Skip old identifier if present

        // Morph keys (legacy)
        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMMORPHKEYS2)) {
            const int morphKeyCount = chunk->ReadInt();
            if (morphKeyCount > 0) {
                struct MorphAccess : RCKMorphController {
                    using RCKMorphController::m_Keys;
                    using RCKMorphController::m_VertexCount;
                };

                auto *morphCtrl = static_cast<MorphAccess *>(static_cast<RCKMorphController *>(CreateController(CKANIMATION_MORPH_CONTROL)));
                const int morphVertexCount = chunk->ReadInt();
                if (morphCtrl) {
                    morphCtrl->m_NbKeys = morphKeyCount;
                    morphCtrl->m_VertexCount = morphVertexCount;
                    morphCtrl->m_Keys = new CKMorphKey[morphKeyCount];
                    for (int i = 0; i < morphKeyCount; ++i) {
                        morphCtrl->m_Keys[i].TimeStep = chunk->ReadFloat();
                        morphCtrl->m_Keys[i].PosArray = nullptr;
                        morphCtrl->m_Keys[i].NormArray = nullptr;

                        const CKDWORD sizeBytes = chunk->ReadDword();
                        const int vecCount = sizeBytes ? (int)(sizeBytes / sizeof(VxVector)) : 0;
                        if (vecCount > 0) {
                            morphCtrl->m_Keys[i].PosArray = new VxVector[vecCount];
                            chunk->ReadAndFillBuffer_LEndian((int)sizeBytes, morphCtrl->m_Keys[i].PosArray);
                        }
                    }
                }
            }
        }

        // Position keys (legacy)
        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMPOSKEYS)) {
            struct LinPosAccess : RCKLinearPositionController {
                using RCKLinearPositionController::m_Keys;
            };

            auto *posCtrl = static_cast<LinPosAccess *>(static_cast<RCKLinearPositionController *>(CreateController(CKANIMATION_LINPOS_CONTROL)));
            const CKDWORD bufSize = chunk->ReadDword();
            const CKDWORD keyCount = chunk->ReadDword();

            if (posCtrl)
                posCtrl->m_NbKeys = (int)keyCount;

            if (posCtrl && keyCount) {
                posCtrl->m_Keys = new CKPositionKey[keyCount];
                chunk->ReadAndFillBuffer_LEndian((int)bufSize, posCtrl->m_Keys);
            }

            if (posCtrl && !keyCount)
                DeleteController(CKANIMATION_LINPOS_CONTROL);
        }

        // Rotation keys (legacy) + scale axis data
        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMROTKEYS)) {
            struct LinRotAccess : RCKLinearRotationController {
                using RCKLinearRotationController::m_Keys;
            };
            struct LinAxisAccess : RCKLinearScaleAxisController {
                using RCKLinearScaleAxisController::m_Keys;
            };

            auto *rotCtrl = static_cast<LinRotAccess *>(static_cast<RCKLinearRotationController *>(CreateController(CKANIMATION_LINROT_CONTROL)));
            const CKDWORD rotBufSize = chunk->ReadDword();
            const CKDWORD rotKeyCount = chunk->ReadDword();

            if (rotCtrl)
                rotCtrl->m_NbKeys = (int)rotKeyCount;

            if (rotCtrl && rotKeyCount) {
                rotCtrl->m_Keys = new CKRotationKey[rotKeyCount];
                chunk->ReadAndFillBuffer_LEndian((int)rotBufSize, rotCtrl->m_Keys);
            }

            if (rotCtrl && !rotKeyCount)
                DeleteController(CKANIMATION_LINROT_CONTROL);

            const CKDWORD axisBufSize = chunk->ReadDword();
            const CKDWORD axisKeyCount = chunk->ReadDword();
            if (axisKeyCount) {
                float *tmp = new float[(size_t)axisKeyCount * 6];
                chunk->ReadAndFillBuffer_LEndian((int)axisBufSize, tmp);

                auto *axisCtrl = static_cast<LinAxisAccess *>(static_cast<RCKLinearScaleAxisController *>(CreateController(CKANIMATION_LINSCLAXIS_CONTROL)));
                if (axisCtrl) {
                    axisCtrl->m_NbKeys = (int)axisKeyCount;
                    axisCtrl->m_Keys = new CKScaleAxisKey[axisKeyCount];
                    for (CKDWORD i = 0; i < axisKeyCount; ++i) {
                        axisCtrl->m_Keys[i].TimeStep = tmp[i * 6 + 0];
                        axisCtrl->m_Keys[i].Rot = *(VxQuaternion *)&tmp[i * 6 + 2];
                    }
                }
                delete[] tmp;
            }
        }

        // Scale keys (legacy)
        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMSCLKEYS)) {
            struct LinSclAccess : RCKLinearScaleController {
                using RCKLinearScaleController::m_Keys;
            };

            auto *sclCtrl = static_cast<LinSclAccess *>(static_cast<RCKLinearScaleController *>(CreateController(CKANIMATION_LINSCL_CONTROL)));
            const CKDWORD bufSize = chunk->ReadDword();
            const CKDWORD keyCount = chunk->ReadDword();

            if (sclCtrl)
                sclCtrl->m_NbKeys = (int)keyCount;

            if (sclCtrl && keyCount) {
                sclCtrl->m_Keys = new CKScaleKey[keyCount];
                chunk->ReadAndFillBuffer_LEndian((int)bufSize, sclCtrl->m_Keys);
            }

            if (sclCtrl && !keyCount)
                DeleteController(CKANIMATION_LINSCL_CONTROL);
        }

        // Read various old format identifiers
        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMFLAGS))
            m_Flags = chunk->ReadDword();

        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMENTITY))
            m_Entity = (RCK3dEntity *) chunk->ReadObject(m_Context);

        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMLENGTH)) {
            // Strictly matches original old-format behavior: only sets keyframe data length.
            if (m_KeyframeData)
                m_KeyframeData->m_Length = chunk->ReadFloat();
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMMERGE)) {
            m_MergeFactor = chunk->ReadFloat();
            if (chunk->ReadInt())
                m_Flags |= CK_OBJECTANIMATION_MERGED;
            else
                m_Flags &= ~CK_OBJECTANIMATION_MERGED;

            m_Anim1 = (RCKObjectAnimation *) chunk->ReadObject(m_Context);
            m_Anim2 = (RCKObjectAnimation *) chunk->ReadObject(m_Context);
        }

        if (chunk->SeekIdentifier(CK_STATESAVE_OBJANIMNEWDATA)) {
            rootPos.x = chunk->ReadFloat();
            rootPos.y = chunk->ReadFloat();
            rootPos.z = chunk->ReadFloat();
        }
    }

    // Store root position as app data if non-zero
    if (rootPos.x != 0.0f || rootPos.y != 0.0f || rootPos.z != 0.0f) {
        VxVector *appData = new VxVector(rootPos);
        SetAppData(appData);
    }

    return CK_OK;
}

int RCKObjectAnimation::GetMemoryOccupation() {
    // Call base class first to get base size
    int size = CKObject::GetMemoryOccupation();

    // Add RCKObjectAnimation-specific fields
    size += sizeof(RCKObjectAnimation) - sizeof(CKObject);

    // If we own the keyframe data, add its size
    if (m_KeyframeData && m_KeyframeData->m_ObjectAnimation == this) {
        // Add sizes of all controllers using a helper function
        // This matches the DLL's call to sub_1004B397(&this->m_KeyframeData->m_PositionController)
        if (m_KeyframeData->m_PositionController) {
            CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
            size += ctrl->DumpKeysTo(nullptr);
        }
        if (m_KeyframeData->m_RotationController) {
            CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
            size += ctrl->DumpKeysTo(nullptr);
        }
        if (m_KeyframeData->m_ScaleController) {
            CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
            size += ctrl->DumpKeysTo(nullptr);
        }
        if (m_KeyframeData->m_ScaleAxisController) {
            CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
            size += ctrl->DumpKeysTo(nullptr);
        }
        if (m_KeyframeData->m_MorphController) {
            CKMorphController *ctrl = reinterpret_cast<CKMorphController *>(m_KeyframeData->m_MorphController);
            size += ctrl->DumpKeysTo(nullptr);
        }
    }

    return size;
}

CKERROR RCKObjectAnimation::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = CKObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKObjectAnimation *src = (RCKObjectAnimation *) &o;

    // Check if we should copy controllers or share data
    CKDWORD deps = context.GetClassDependencies(CKCID_OBJECTANIMATION);
    if (deps & 1) {
        // Deep copy - clone the keyframe data
        Clone(src);
    } else {
        // Share data
        ShareDataFrom(src);
    }

    // Copy remaining fields
    m_Flags = src->m_Flags;
    m_MergeFactor = src->m_MergeFactor;
    m_CurrentStep = src->m_CurrentStep;
    m_field_38 = src->m_field_38;
    m_ParentKeyedAnimation = src->m_ParentKeyedAnimation;

    return CK_OK;
}

// Based on IDA decompilation at 0x100585aa
void RCKObjectAnimation::CheckPreDeletion() {
    CKObject::CheckPreDeletion();
    
    // Clear m_Anim1 if it's being deleted
    if (m_Anim1 && m_Anim1->IsToBeDeleted())
        m_Anim1 = nullptr;
    
    // Clear m_Anim2 if it's being deleted
    if (m_Anim2 && m_Anim2->IsToBeDeleted())
        m_Anim2 = nullptr;
    
    // Clear m_Entity if it's being deleted
    if (m_Entity && m_Entity->IsToBeDeleted())
        m_Entity = nullptr;
}

// Based on IDA decompilation at 0x10058662
CKBOOL RCKObjectAnimation::IsObjectUsed(CKObject *obj, CK_CLASSID cid) {
    if (obj == (CKObject *)m_Anim1)
        return TRUE;
    if (obj == (CKObject *)m_Anim2)
        return TRUE;
    if (obj == (CKObject *)m_Entity)
        return TRUE;
    return CKObject::IsObjectUsed(obj, cid);
}

// Based on IDA decompilation at 0x1005c31d
CKERROR RCKObjectAnimation::RemapDependencies(CKDependenciesContext &context) {
    CKERROR err = CKObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;
    
    // Remap m_Entity (offset 0x24 = 36 bytes)
    m_Entity = (RCK3dEntity *)context.Remap((CKObject *)m_Entity);
    
    // Remap m_Anim1 (offset 0x30 = 48 bytes)
    m_Anim1 = (RCKObjectAnimation *)context.Remap((CKObject *)m_Anim1);
    
    // Remap m_Anim2 (offset 0x34 = 52 bytes)
    m_Anim2 = (RCKObjectAnimation *)context.Remap((CKObject *)m_Anim2);
    
    return CK_OK;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Controllers
//=============================================================================

CKAnimController *RCKObjectAnimation::CreateController(CKANIMATION_CONTROLLER ControlType) {
    // CreateController delegates to CKKeyframeData::CreateController
    // Based on IDA analysis of CKKeyframeData::CreateController at 0x1004A922
    if (!m_KeyframeData)
        return nullptr;

    // Get the base controller type (mask out specific type to get category)
    CKDWORD baseType = ControlType & CKANIMATION_CONTROLLER_MASK;

    // Delete existing controller of the same category
    switch (baseType) {
    case CKANIMATION_CONTROLLER_POS:
        if (m_KeyframeData->m_PositionController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
            m_KeyframeData->m_PositionController = nullptr;
        }
        break;
    case CKANIMATION_CONTROLLER_ROT:
        if (m_KeyframeData->m_RotationController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
            m_KeyframeData->m_RotationController = nullptr;
        }
        break;
    case CKANIMATION_CONTROLLER_SCL:
        if (m_KeyframeData->m_ScaleController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
            m_KeyframeData->m_ScaleController = nullptr;
        }
        break;
    case CKANIMATION_CONTROLLER_SCLAXIS:
        if (m_KeyframeData->m_ScaleAxisController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
            m_KeyframeData->m_ScaleAxisController = nullptr;
        }
        break;
    case CKANIMATION_CONTROLLER_MORPH:
        if (m_KeyframeData->m_MorphController) {
            delete reinterpret_cast<CKMorphController *>(m_KeyframeData->m_MorphController);
            m_KeyframeData->m_MorphController = nullptr;
        }
        break;
    default:
        return nullptr;
    }

    // Create new controller and assign to appropriate slot
    CKAnimController *ctrl = m_KeyframeData->CreateController(ControlType);
    if (!ctrl)
        return nullptr;

    // Assign to the appropriate controller slot
    switch (baseType) {
    case CKANIMATION_CONTROLLER_POS:
        m_KeyframeData->m_PositionController = ctrl;
        break;
    case CKANIMATION_CONTROLLER_ROT:
        m_KeyframeData->m_RotationController = ctrl;
        break;
    case CKANIMATION_CONTROLLER_SCL:
        m_KeyframeData->m_ScaleController = ctrl;
        break;
    case CKANIMATION_CONTROLLER_SCLAXIS:
        m_KeyframeData->m_ScaleAxisController = ctrl;
        break;
    case CKANIMATION_CONTROLLER_MORPH:
        m_KeyframeData->m_MorphController = reinterpret_cast<CKMorphController *>(ctrl);
        break;
    }

    return ctrl;
}

CKBOOL RCKObjectAnimation::DeleteController(CKANIMATION_CONTROLLER ControlType) {
    if (!m_KeyframeData)
        return FALSE;

    // Get the base controller type
    CKDWORD baseType = ControlType & CKANIMATION_CONTROLLER_MASK;
    CKAnimController *controller = nullptr;

    switch (baseType) {
    case CKANIMATION_CONTROLLER_POS:
        controller = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
        if (controller) {
            delete controller;
            m_KeyframeData->m_PositionController = 0;
        }
        break;

    case CKANIMATION_CONTROLLER_ROT:
        controller = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
        if (controller) {
            delete controller;
            m_KeyframeData->m_RotationController = 0;
        }
        break;

    case CKANIMATION_CONTROLLER_SCL:
        controller = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
        if (controller) {
            delete controller;
            m_KeyframeData->m_ScaleController = 0;
        }
        break;

    case CKANIMATION_CONTROLLER_SCLAXIS:
        controller = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
        if (controller) {
            delete controller;
            m_KeyframeData->m_ScaleAxisController = 0;
        }
        break;

    case CKANIMATION_CONTROLLER_MORPH:
        if (m_KeyframeData->m_MorphController) {
            delete reinterpret_cast<CKMorphController *>(m_KeyframeData->m_MorphController);
            m_KeyframeData->m_MorphController = nullptr;
        }
        break;

    default:
        return FALSE;
    }

    return TRUE;
}

CKAnimController *RCKObjectAnimation::GetPositionController() {
    if (m_KeyframeData)
        return reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
    return nullptr;
}

CKAnimController *RCKObjectAnimation::GetScaleController() {
    if (m_KeyframeData)
        return reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
    return nullptr;
}

CKAnimController *RCKObjectAnimation::GetRotationController() {
    if (m_KeyframeData)
        return reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
    return nullptr;
}

CKAnimController *RCKObjectAnimation::GetScaleAxisController() {
    if (m_KeyframeData)
        return reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
    return nullptr;
}

CKMorphController *RCKObjectAnimation::GetMorphController() {
    if (m_KeyframeData)
        return reinterpret_cast<CKMorphController *>(m_KeyframeData->m_MorphController);
    return nullptr;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Evaluation
//=============================================================================

CKBOOL RCKObjectAnimation::EvaluatePosition(float Time, VxVector &Pos) {
    // Flag 0x04 disables position evaluation
    if (m_Flags & CK_OBJECTANIMATION_IGNOREPOS) {
        Pos.Set(0.0f, 0.0f, 0.0f);
        return FALSE;
    }

    // Check for merged animation (flag 0x80)
    if (IsMerged()) {
        if (m_MergeFactor == 0.0f) {
            // Use only first animation
            float normalizedTime = Time / m_KeyframeData->m_Length;
            float anim1Time = normalizedTime * m_Anim1->m_KeyframeData->m_Length;
            return m_Anim1->EvaluatePosition(anim1Time, Pos);
        } else if (m_MergeFactor == 1.0f) {
            // Use only second animation
            float normalizedTime = Time / m_KeyframeData->m_Length;
            float anim2Time = normalizedTime * m_Anim2->m_KeyframeData->m_Length;
            return m_Anim2->EvaluatePosition(anim2Time, Pos);
        } else {
            // Blend between both animations
            VxVector pos2;
            float normalizedTime = Time / m_KeyframeData->m_Length;
            float anim1Time = normalizedTime * m_Anim1->m_KeyframeData->m_Length;
            float anim2Time = normalizedTime * m_Anim2->m_KeyframeData->m_Length;

            CKBOOL res1 = m_Anim1->EvaluatePosition(anim1Time, Pos);
            CKBOOL res2 = m_Anim2->EvaluatePosition(anim2Time, pos2);

            if (res1 && res2) {
                // Interpolate: Pos = Pos * (1 - factor) + pos2 * factor
                Pos = Pos * (1.0f - m_MergeFactor) + pos2 * m_MergeFactor;
                return TRUE;
            } else if (res1) {
                return TRUE;
            } else if (res2) {
                Pos = pos2;
                return TRUE;
            }
            return FALSE;
        }
    }

    // Direct controller evaluation
    if (m_KeyframeData && m_KeyframeData->m_PositionController) {
        CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
        return ctrl->Evaluate(Time, &Pos);
    }

    Pos.Set(0.0f, 0.0f, 0.0f);
    return FALSE;
}

CKBOOL RCKObjectAnimation::EvaluateScale(float Time, VxVector &Scl) {
    // Flag 0x10 disables scale evaluation
    if (m_Flags & CK_OBJECTANIMATION_IGNORESCALE) {
        Scl.Set(1.0f, 1.0f, 1.0f);
        return FALSE;
    }

    // Check for merged animation (flag 0x80)
    if (IsMerged()) {
        VxVector scale2;
        float normalizedTime = Time / m_KeyframeData->m_Length;
        float anim1Time = normalizedTime * m_Anim1->m_KeyframeData->m_Length;
        float anim2Time = normalizedTime * m_Anim2->m_KeyframeData->m_Length;

        CKBOOL res1 = m_Anim1->EvaluateScale(anim1Time, Scl);
        CKBOOL res2 = m_Anim2->EvaluateScale(anim2Time, scale2);

        if (res1 && res2) {
            // Interpolate: Scl = Scl * (1 - factor) + scale2 * factor
            Scl = Scl * (1.0f - m_MergeFactor) + scale2 * m_MergeFactor;
            return TRUE;
        } else if (res1) {
            return TRUE;
        } else if (res2) {
            Scl = scale2;
            return TRUE;
        }
        return FALSE;
    }

    // Direct controller evaluation
    if (m_KeyframeData && m_KeyframeData->m_ScaleController) {
        CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
        return ctrl->Evaluate(Time, &Scl);
    }

    Scl.Set(1.0f, 1.0f, 1.0f);
    return FALSE;
}

CKBOOL RCKObjectAnimation::EvaluateRotation(float Time, VxQuaternion &Rot) {
    // Flag 0x08 disables rotation evaluation
    if (m_Flags & CK_OBJECTANIMATION_IGNOREROT) {
        Rot = VxQuaternion(); // Identity quaternion
        return FALSE;
    }

    // Check for merged animation (flag 0x80)
    if (IsMerged()) {
        if (m_MergeFactor == 0.0f) {
            // Use only first animation
            float normalizedTime = Time / m_KeyframeData->m_Length;
            float anim1Time = normalizedTime * m_Anim1->m_KeyframeData->m_Length;
            return m_Anim1->EvaluateRotation(anim1Time, Rot);
        } else if (m_MergeFactor == 1.0f) {
            // Use only second animation
            float normalizedTime = Time / m_KeyframeData->m_Length;
            float anim2Time = normalizedTime * m_Anim2->m_KeyframeData->m_Length;
            return m_Anim2->EvaluateRotation(anim2Time, Rot);
        } else {
            // Blend between both animations using Slerp
            VxQuaternion rot1, rot2;
            float normalizedTime = Time / m_KeyframeData->m_Length;
            float anim1Time = normalizedTime * m_Anim1->m_KeyframeData->m_Length;
            float anim2Time = normalizedTime * m_Anim2->m_KeyframeData->m_Length;

            CKBOOL res1 = m_Anim1->EvaluateRotation(anim1Time, rot1);
            CKBOOL res2 = m_Anim2->EvaluateRotation(anim2Time, rot2);

            if (res1 && res2) {
                // Slerp interpolation
                Rot = Slerp(m_MergeFactor, rot1, rot2);
                return TRUE;
            } else if (res1) {
                Rot = rot1;
                return TRUE;
            } else if (res2) {
                Rot = rot2;
                return TRUE;
            }
            return FALSE;
        }
    }

    // Direct controller evaluation
    if (m_KeyframeData && m_KeyframeData->m_RotationController) {
        CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
        return ctrl->Evaluate(Time, &Rot);
    }

    Rot = VxQuaternion(); // Identity quaternion
    return FALSE;
}

CKBOOL RCKObjectAnimation::EvaluateScaleAxis(float Time, VxQuaternion &ScaleAxis) {
    // Flag 0x40 disables scale axis evaluation
    if (m_Flags & CK_OBJECTANIMATION_IGNORESCALEROT)
        return FALSE;

    // Check for merged animation (flag 0x80)
    if (IsMerged()) {
        VxQuaternion axis1, axis2;
        float normalizedTime = Time / m_KeyframeData->m_Length;
        float anim1Time = normalizedTime * m_Anim1->m_KeyframeData->m_Length;
        float anim2Time = normalizedTime * m_Anim2->m_KeyframeData->m_Length;

        CKBOOL res1 = m_Anim1->EvaluateScaleAxis(anim1Time, axis1);
        CKBOOL res2 = m_Anim2->EvaluateScaleAxis(anim2Time, axis2);

        if (res1 && res2) {
            // Slerp interpolation
            ScaleAxis = Slerp(m_MergeFactor, axis1, axis2);
            return TRUE;
        } else if (res1) {
            ScaleAxis = axis1;
            return TRUE;
        } else if (res2) {
            ScaleAxis = axis2;
            return TRUE;
        }
        return FALSE;
    }

    // Direct controller evaluation
    if (m_KeyframeData && m_KeyframeData->m_ScaleAxisController) {
        CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
        return ctrl->Evaluate(Time, &ScaleAxis);
    }

    return FALSE;
}

CKBOOL RCKObjectAnimation::EvaluateMorphTarget(float Time, int VertexCount, VxVector *Vertices, CKDWORD VStride, VxCompressedVector *Normals) {
    // Based on IDA decompilation at 0x100574B4

    // Check flag 0x20 - morph disabled
    if (m_Flags & CK_OBJECTANIMATION_IGNOREMORPH)
        return FALSE;

    // Check for merged animation (flag 0x80)
    if (!IsMerged()) {
        // Direct morph controller evaluation
        if (m_KeyframeData && m_KeyframeData->m_MorphController) {
            RCKMorphController *ctrl = reinterpret_cast<RCKMorphController *>(m_KeyframeData->m_MorphController);
            return ctrl->Evaluate(Time, VertexCount, Vertices, VStride, Normals);
        }
        return FALSE;
    }

    // Merged animation - blend morph data from both sub-animations
    int vertCount1 = m_Anim1->GetMorphVertexCount();
    int vertCount2 = m_Anim2->GetMorphVertexCount();

    // If vertex counts don't match, use whichever animation has morph data
    if (!vertCount1 || vertCount1 != vertCount2) {
        if (vertCount1)
            return m_Anim1->EvaluateMorphTarget(Time, VertexCount, Vertices, VStride, Normals);
        if (vertCount2)
            return m_Anim2->EvaluateMorphTarget(Time, VertexCount, Vertices, VStride, Normals);
        return FALSE;
    }

    // Allocate temporary buffers for blending
    // Need space for 2 vertex arrays + 2 normal arrays (if needed)
    int bufferSize = vertCount1 * sizeof(VxVector) * 2;
    if (Normals)
        bufferSize += vertCount1 * sizeof(VxCompressedVector) * 2;
    CKMemoryPool pool(m_Context, bufferSize);
    VxVector *tempVerts1 = (VxVector *) pool.Mem();
    VxVector *tempVerts2 = tempVerts1 + vertCount1;
    VxCompressedVector *tempNorms = (VxCompressedVector *) (tempVerts2 + vertCount1);

    // Check if both animations have normal data
    CKBOOL hasNorm1 = FALSE, hasNorm2 = FALSE;
    if (Normals) {
        hasNorm1 = m_Anim1->HasMorphNormalInfo();
        hasNorm2 = m_Anim2->HasMorphNormalInfo();
    }

    VxCompressedVector *normPtr1 = hasNorm1 ? tempNorms : nullptr;
    VxCompressedVector *normPtr2 = hasNorm2 ? tempNorms + vertCount1 : nullptr;

    // Scale time to each animation's length
    float normalizedTime = Time / m_KeyframeData->m_Length;
    float time1 = normalizedTime * m_Anim1->m_KeyframeData->m_Length;
    float time2 = normalizedTime * m_Anim2->m_KeyframeData->m_Length;

    // Evaluate both animations
    if (hasNorm1) {
        m_Anim1->EvaluateMorphTarget(time1, VertexCount, tempVerts1, 12, normPtr1);
        m_Anim2->EvaluateMorphTarget(time2, VertexCount, tempVerts2, 12, normPtr2);
    } else if (hasNorm2) {
        m_Anim1->EvaluateMorphTarget(time1, VertexCount, tempVerts1, 12, normPtr1);
        m_Anim2->EvaluateMorphTarget(time2, VertexCount, tempVerts2, 12, Normals);
    } else {
        m_Anim1->EvaluateMorphTarget(time1, VertexCount, tempVerts1, 12, nullptr);
        m_Anim2->EvaluateMorphTarget(time2, VertexCount, tempVerts2, 12, nullptr);
    }

    // Blend vertices: result = v1 * (1 - factor) + v2 * factor
    float factor = m_MergeFactor;
    float invFactor = 1.0f - factor;

    VxVector *outVert = Vertices;
    for (int i = 0; i < VertexCount; ++i) {
        outVert->x = tempVerts1[i].x * invFactor + tempVerts2[i].x * factor;
        outVert->y = tempVerts1[i].y * invFactor + tempVerts2[i].y * factor;
        outVert->z = tempVerts1[i].z * invFactor + tempVerts2[i].z * factor;
        outVert = (VxVector *) ((char *) outVert + VStride);
    }

    // Blend normals if both have normal info
    if (hasNorm1 && hasNorm2 && Normals) {
        for (int i = 0; i < VertexCount; ++i) {
            // Interpolate compressed normals using their xa/ya fields
            Normals[i].xa = (short) ((float) normPtr1[i].xa * invFactor + (float) normPtr2[i].xa * factor);
            Normals[i].ya = (short) ((float) normPtr1[i].ya * invFactor + (float) normPtr2[i].ya * factor);
        }
    }

    return TRUE;
}

CKBOOL RCKObjectAnimation::EvaluateKeys(float step, VxQuaternion *rot, VxVector *pos, VxVector *scale, VxQuaternion *ScaleRot) {
    // Clamp step to [0, 1]
    if (step > 1.0f)
        step = 1.0f;
    if (step < 0.0f)
        step = 0.0f;

    // Convert step to time
    float Time = step * m_KeyframeData->m_Length;

    // Evaluate each component if requested
    if (pos)
        EvaluatePosition(Time, *pos);
    if (rot)
        EvaluateRotation(Time, *rot);
    if (scale)
        EvaluateScale(Time, *scale);
    if (ScaleRot)
        EvaluateScaleAxis(Time, *ScaleRot);

    return TRUE;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Has Info
//=============================================================================

CKBOOL RCKObjectAnimation::HasMorphNormalInfo() {
    // Check for merged animation
    if (IsMerged()) {
        // Merged animation - check both sub-animations
        if (m_Anim1 && m_Anim1->HasMorphNormalInfo())
            return TRUE;
        if (m_Anim2 && m_Anim2->HasMorphNormalInfo())
            return TRUE;
        return FALSE;
    }

    // Direct check: controller exists, has keys, and first key has normal data
    if (m_KeyframeData && m_KeyframeData->m_MorphController) {
        CKMorphController *ctrl = m_KeyframeData->m_MorphController;
        if (ctrl->GetKeyCount() > 0) {
            CKMorphKey *key = static_cast<CKMorphKey *>(ctrl->GetKey(0));
            if (key && key->NormArray)
                return TRUE;
        }
    }
    return FALSE;
}

CKBOOL RCKObjectAnimation::HasMorphInfo() {
    if (IsMerged()) {
        // Merged animation - check both sub-animations
        if (m_Anim1 && m_Anim1->HasMorphInfo())
            return TRUE;
        if (m_Anim2 && m_Anim2->HasMorphInfo())
            return TRUE;
        return FALSE;
    }
    return m_KeyframeData && m_KeyframeData->m_MorphController != nullptr;
}

CKBOOL RCKObjectAnimation::HasScaleInfo() {
    if (IsMerged()) {
        // Merged animation - check both sub-animations
        if (m_Anim1 && m_Anim1->HasScaleInfo())
            return TRUE;
        if (m_Anim2 && m_Anim2->HasScaleInfo())
            return TRUE;
        return FALSE;
    }
    return m_KeyframeData && m_KeyframeData->m_ScaleController != 0;
}

CKBOOL RCKObjectAnimation::HasPositionInfo() {
    if (IsMerged()) {
        // Merged animation - check both sub-animations
        if (m_Anim1 && m_Anim1->HasPositionInfo())
            return TRUE;
        if (m_Anim2 && m_Anim2->HasPositionInfo())
            return TRUE;
        return FALSE;
    }
    return m_KeyframeData && m_KeyframeData->m_PositionController != 0;
}

CKBOOL RCKObjectAnimation::HasRotationInfo() {
    if (IsMerged()) {
        // Merged animation - check both sub-animations
        if (m_Anim1 && m_Anim1->HasRotationInfo())
            return TRUE;
        if (m_Anim2 && m_Anim2->HasRotationInfo())
            return TRUE;
        return FALSE;
    }
    return m_KeyframeData && m_KeyframeData->m_RotationController != 0;
}

CKBOOL RCKObjectAnimation::HasScaleAxisInfo() {
    if (IsMerged()) {
        // Merged animation - check both sub-animations
        if (m_Anim1 && m_Anim1->HasScaleAxisInfo())
            return TRUE;
        if (m_Anim2 && m_Anim2->HasScaleAxisInfo())
            return TRUE;
        return FALSE;
    }
    return m_KeyframeData && m_KeyframeData->m_ScaleAxisController != 0;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Adding Keys
//=============================================================================

void RCKObjectAnimation::AddPositionKey(float TimeStep, VxVector *pos) {
    // Based on IDA decompilation at 0x100567EC
    if (!pos)
        return;

    // Create controller if it doesn't exist
    if (!m_KeyframeData->m_PositionController)
        CreateController(CKANIMATION_LINPOS_CONTROL);

    // Create position key and add it
    CKPositionKey key(TimeStep, *pos);
    CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
    if (ctrl)
        ctrl->AddKey(&key);
}

void RCKObjectAnimation::AddRotationKey(float TimeStep, VxQuaternion *rot) {
    // Based on IDA decompilation at 0x10056842
    if (!rot)
        return;

    // Create controller if it doesn't exist
    if (!m_KeyframeData->m_RotationController)
        CreateController(CKANIMATION_LINROT_CONTROL);

    // Create rotation key and add it
    CKRotationKey key(TimeStep, *rot);
    CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
    if (ctrl)
        ctrl->AddKey(&key);
}

void RCKObjectAnimation::AddScaleKey(float TimeStep, VxVector *scl) {
    // Based on IDA decompilation at 0x100568A0
    if (!scl)
        return;

    // Create controller if it doesn't exist
    if (!m_KeyframeData->m_ScaleController)
        CreateController(CKANIMATION_LINSCL_CONTROL);

    // Create scale key and add it (CKScaleKey is typedef of CKPositionKey)
    CKScaleKey key(TimeStep, *scl);
    CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
    if (ctrl)
        ctrl->AddKey(&key);
}

void RCKObjectAnimation::AddScaleAxisKey(float TimeStep, VxQuaternion *sclaxis) {
    // Based on IDA decompilation at 0x10056918
    if (!sclaxis)
        return;

    // Create controller if it doesn't exist
    if (!m_KeyframeData->m_ScaleAxisController)
        CreateController(CKANIMATION_LINSCLAXIS_CONTROL);

    // Create scale axis key and add it (CKScaleAxisKey is typedef of CKRotationKey)
    CKScaleAxisKey key(TimeStep, *sclaxis);
    CKAnimController *ctrl = reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
    if (ctrl)
        ctrl->AddKey(&key);
}

void RCKObjectAnimation::CheckScaleKeys(VxVector &scale) {
    if (m_KeyframeData) {
        if (m_KeyframeData->m_RotationController && !m_KeyframeData->m_ScaleController) {
            CreateController(CKANIMATION_LINSCL_CONTROL);
        }
        if (m_KeyframeData->m_ScaleController) {
            CKAnimController *ctrl = m_KeyframeData->m_ScaleController;
            if (ctrl->GetKeyCount() == 0) {
                // Add a scale key at time 0 with the provided scale
                CKScaleKey key(0.0f, scale);
                ctrl->AddKey(&key);
            }
        }
    }
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Comparison and Sharing
//=============================================================================

CKBOOL RCKObjectAnimation::Compare(CKObjectAnimation *anim, float threshold) {
    // Based on IDA decompilation at 0x1005667D
    if (!anim)
        return FALSE;

    // Same animation = equal
    if (anim == (CKObjectAnimation *) this)
        return TRUE;

    // Compare keyframe data using CKKeyframeData::Compare (sub_1004A804)
    RCKObjectAnimation *other = (RCKObjectAnimation *) anim;
    CKKeyframeData *kf1 = m_KeyframeData;
    CKKeyframeData *kf2 = other->m_KeyframeData;

    if (!kf2)
        return FALSE;

    // Check lengths match
    if (kf1->m_Length != kf2->m_Length)
        return FALSE;

    // Compare position controllers
    if (kf1->m_PositionController) {
        if (!kf1->m_PositionController->Compare(kf2->m_PositionController, threshold))
            return FALSE;
    }

    // Compare scale controllers
    if (kf1->m_ScaleController) {
        if (!kf1->m_ScaleController->Compare(kf2->m_ScaleController, threshold))
            return FALSE;
    }

    // Compare rotation controllers
    if (kf1->m_RotationController) {
        if (!kf1->m_RotationController->Compare(kf2->m_RotationController, threshold))
            return FALSE;
    }

    // Compare scale axis controllers
    if (kf1->m_ScaleAxisController) {
        if (!kf1->m_ScaleAxisController->Compare(kf2->m_ScaleAxisController, threshold))
            return FALSE;
    }

    // Compare morph controllers
    if (kf1->m_MorphController) {
        if (!kf1->m_MorphController->Compare(kf2->m_MorphController, threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKObjectAnimation::ShareDataFrom(CKObjectAnimation *anim) {
    // Based on IDA decompilation at 0x100566B9
    RCKObjectAnimation *srcAnim = (RCKObjectAnimation *) anim;

    // Can't share from self
    if (srcAnim == this)
        return FALSE;

    // Release current keyframe data
    if (m_KeyframeData) {
        // Decrement reference count
        if (--m_KeyframeData->m_RefCount <= 0) {
            // We're the last user - delete the keyframe data
            if (m_KeyframeData) {
                delete m_KeyframeData;
            }
        }
        m_KeyframeData = nullptr;
    }

    // Share from source animation or create new
    if (srcAnim) {
        // Share the keyframe data with source
        m_KeyframeData = srcAnim->m_KeyframeData;
        if (m_KeyframeData)
            ++m_KeyframeData->m_RefCount; // Increment reference count
    } else {
        // Create new keyframe data using constructor (not memset)
        m_KeyframeData = new CKKeyframeData();
        m_KeyframeData->m_ObjectAnimation = this;
    }

    return TRUE;
}

CKObjectAnimation *RCKObjectAnimation::Shared() {
    if (m_KeyframeData)
        return m_KeyframeData->m_ObjectAnimation;
    return nullptr;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Flags
//=============================================================================

void RCKObjectAnimation::SetFlags(CKDWORD flags) {
    m_Flags = flags;
}

CKDWORD RCKObjectAnimation::GetFlags() {
    return m_Flags;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Clearing
//=============================================================================

void RCKObjectAnimation::Clear() {
    m_Flags = 0;
    m_Entity = nullptr;
    m_Anim1 = nullptr;
    m_Anim2 = nullptr;
    m_MergeFactor = 0.5f;
    m_CurrentStep = 0.0f;
}

void RCKObjectAnimation::ClearAll() {
    Clear();
    // Also clear keyframe data if we own it
    if (m_KeyframeData && m_KeyframeData->m_ObjectAnimation == this) {
        // Clear all controllers
        m_KeyframeData->m_PositionController = 0;
        m_KeyframeData->m_RotationController = 0;
        m_KeyframeData->m_ScaleController = 0;
        m_KeyframeData->m_ScaleAxisController = 0;
        m_KeyframeData->m_MorphController = nullptr;
        m_KeyframeData->m_Length = 0.0f;
    }
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Merged Animations
//=============================================================================

float RCKObjectAnimation::GetMergeFactor() {
    return m_MergeFactor;
}

void RCKObjectAnimation::SetMergeFactor(float factor) {
    m_MergeFactor = factor;
}

CKBOOL RCKObjectAnimation::IsMerged() {
    // Check flag bit 7 (0x80) to determine if this is a merged animation
    return (m_Flags & CK_OBJECTANIMATION_MERGED) != 0;
}

CKObjectAnimation *RCKObjectAnimation::CreateMergedAnimation(CKObjectAnimation *subanim2, CKBOOL Dynamic) {
    if (!subanim2)
        return nullptr;

    // Determine creation options
    CK_OBJECTCREATION_OPTIONS Options = CK_OBJECTCREATION_NONAMECHECK;
    if (GetFlags() & 0x20000000)
        Options = CK_OBJECTCREATION_DYNAMIC;

    // Create a new ObjectAnimation object
    RCKObjectAnimation *merged = (RCKObjectAnimation *) m_Context->CreateObject(
        CKCID_OBJECTANIMATION, GetName(), Options, nullptr);
    if (!merged)
        return nullptr;

    // Set the length to the max of the two animations
    float len1 = GetLength();
    float len2 = subanim2->GetLength();
    merged->SetLength(len1 >= len2 ? len1 : len2);

    // Set the merged flag (0x80)
    merged->m_Flags |= CK_OBJECTANIMATION_MERGED;

    // Set the source animations
    merged->m_Anim1 = this;
    merged->m_Anim2 = (RCKObjectAnimation *) subanim2;

    // Set the entity
    merged->Set3dEntity((CK3dEntity *) m_Entity);

    return merged;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Animation Length
//=============================================================================

void RCKObjectAnimation::SetLength(float nbframe) {
    if (m_KeyframeData)
        m_KeyframeData->m_Length = nbframe;
}

float RCKObjectAnimation::GetLength() {
    if (m_KeyframeData)
        return m_KeyframeData->m_Length;
    return 0.0f;
}

void RCKObjectAnimation::GetVelocity(float step, VxVector *vel) {
    // Based on IDA decompilation at 0x10057efd
    // This function calculates the positional velocity at a given step
    // by evaluating position at two nearby time steps and computing the difference
    
    if (!vel)
        return;
    
    VxVector pos1, pos2;
    VxVector velocity(0.0f, 0.0f, 0.0f);
    
    float evalStep = 0.0f;
    
    // Calculate frame from step
    float frame = step * m_KeyframeData->m_Length;
    
    // Determine which direction to sample (forward or backward)
    if (frame + 1.0f < m_KeyframeData->m_Length) {
        // Can sample forward
        evalStep = (frame + 1.0f) / m_KeyframeData->m_Length;
    } else {
        // Sample backward
        evalStep = (frame - 1.0f) / m_KeyframeData->m_Length;
    }
    
    // Evaluate position at current step (stored in pos1)
    EvaluateKeys(step, nullptr, &pos1, nullptr, nullptr);
    
    // Evaluate position at nearby step (stored in pos2)
    EvaluateKeys(evalStep, nullptr, &pos2, nullptr, nullptr);
    
    // Calculate velocity as position difference if entity exists
    if (m_Entity) {
        if (step < evalStep) {
            // Forward difference: velocity = pos2 - pos1
            velocity.x = pos2.x - pos1.x;
            velocity.y = pos2.y - pos1.y;
            velocity.z = pos2.z - pos1.z;
        } else {
            // Backward difference: velocity = pos1 - pos2
            velocity.x = pos1.x - pos2.x;
            velocity.y = pos1.y - pos2.y;
            velocity.z = pos1.z - pos2.z;
        }
    }
    
    *vel = velocity;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Current Position
//=============================================================================

CKERROR RCKObjectAnimation::SetStep(float step, CKKeyedAnimation *anim) {
    m_CurrentStep = step;

    // Calculate the frame from step
    float frame = step * (m_KeyframeData ? m_KeyframeData->m_Length : 0.0f);

    // Check if entity is valid
    if (!m_Entity)
        return CKERR_NOTINITIALIZED;

    // Handle special case for CKANIMATION_FORCESETSTEP (-1)
    if (anim == (CKKeyedAnimation *) -1) {
        anim = nullptr;
    } else {
        // Check if entity ignores animations (flag 0x400)
        if (m_Entity->GetMoveableFlags() & 0x400)
            return CK_OK;
    }

    // Check if entity is a body part with exclusive animation
    if (CKIsChildClassOf(m_Entity, CKCID_BODYPART)) {
        CKAnimation *exclusive = ((CKBodyPart *) m_Entity)->GetExclusiveAnimation();
        if (exclusive && (CKKeyedAnimation *) exclusive != anim)
            return CK_OK;
    }

    // Evaluate transform components
    CKDWORD transformFlags = 0;
    VxVector pos(0, 0, 0), scale(1, 1, 1);
    VxQuaternion rot, scaleAxis;

    if (EvaluatePosition(frame, pos))
        transformFlags |= 4; // Has position
    if (EvaluateRotation(frame, rot))
        transformFlags |= 8; // Has rotation
    if (EvaluateScale(frame, scale))
        transformFlags |= 1; // Has scale
    if (EvaluateScaleAxis(frame, scaleAxis))
        transformFlags |= 2; // Has scale axis

    // Apply transforms to entity
    if (transformFlags) {
        VxMatrix localMatrix = m_Entity->GetLocalMatrix();

        if (transformFlags == 4) {
            // Only position - just set position component of local matrix
            localMatrix[3][0] = pos.x;
            localMatrix[3][1] = pos.y;
            localMatrix[3][2] = pos.z;
        } else {
            // Need to decompose current matrix to fill missing components
            if ((transformFlags & 0xF) != 0xF) {
                VxQuaternion tempRot;
                VxVector tempPos, tempScale;
                Vx3DDecomposeMatrix(localMatrix, tempRot, tempPos, tempScale);
                if (!(transformFlags & 8)) rot = tempRot;
                if (!(transformFlags & 4)) pos = tempPos;
                if (!(transformFlags & 1)) scale = tempScale;
            }

            // Build matrix from quaternion rotation and components
            VxMatrix rotMatrix;
            rot.ToMatrix(rotMatrix);

            // Apply scale
            VxMatrix scaleMatrix;
            Vx3DMatrixIdentity(scaleMatrix);
            scaleMatrix[0][0] = scale.x;
            scaleMatrix[1][1] = scale.y;
            scaleMatrix[2][2] = scale.z;

            // Combine: result = scale * rotation
            Vx3DMultiplyMatrix(localMatrix, scaleMatrix, rotMatrix);

            // Set position
            localMatrix[3][0] = pos.x;
            localMatrix[3][1] = pos.y;
            localMatrix[3][2] = pos.z;
        }

        m_Entity->SetLocalMatrix(localMatrix);

        // Notify matrix change if not driven by character animation
        if (!anim || !anim->GetCharacter())
            m_Entity->LocalMatrixChanged(FALSE, FALSE);
    }

    // Morph animation processing
    // Based on IDA decompilation at 0x100578CA (lines 130-182)
    if (HasMorphInfo()) {
        // Only process if entity doesn't have a skin (skinned entities have different morph handling)
        if (!m_Entity->m_Skin) {
            RCKMesh *currentMesh = m_Entity->m_CurrentMesh;
            if (currentMesh) {
                // Set mesh flag indicating morph data has changed (0x40000)
                CKDWORD meshFlags = currentMesh->GetFlags();
                currentMesh->SetFlags(meshFlags | 0x40000);

                // Invalidate bounding box since vertices will move
                if (m_Entity->m_SceneGraphNode)
                    m_Entity->m_SceneGraphNode->InvalidateBox(TRUE);

                // Get vertex counts
                int morphVertexCount = GetMorphVertexCount();
                int meshVertexCount = currentMesh->GetModifierVertexCount();
                CKBOOL hasNormalInfo = HasMorphNormalInfo();

                // Only proceed if mesh has vertices and they fit within morph data
                if (meshVertexCount > 0 && meshVertexCount <= morphVertexCount) {
                    // Allocate temporary buffer for normals if needed
                    // VxCompressedVector = 4 bytes per normal
                    CKMemoryPool memPool(m_Context, morphVertexCount * sizeof(VxCompressedVector));
                    VxCompressedVector *normalBuffer = hasNormalInfo ? (VxCompressedVector *) memPool.Mem() : nullptr;

                    // Get modifier vertices from mesh
                    CKDWORD vertexStride = 0;
                    CKBYTE *vertices = currentMesh->GetModifierVertices(&vertexStride);

                    // Evaluate morph target into mesh vertices
                    EvaluateMorphTarget(frame, meshVertexCount, (VxVector *) vertices, vertexStride, normalBuffer);

                    // Copy normals if we have normal info
                    if (hasNormalInfo) {
                        CKDWORD normalStride = 0;
                        VxVector *normals = (VxVector *) currentMesh->GetNormalsPtr(&normalStride);
                        if (normals && normalBuffer) {
                            for (int i = 0; i < meshVertexCount; i++) {
                                // Decompress the compressed normal into full normal
                                // Use operator= for conversion from compressed to full vector
                                *normals = normalBuffer[i];
                                normals = (VxVector *) ((CKBYTE *) normals + normalStride);
                            }
                        }
                        // Notify mesh of vertex movement, don't rebuild normals since we provided them
                        currentMesh->ModifierVertexMove(FALSE, TRUE);
                    } else {
                        // Notify mesh of vertex movement, rebuild normals automatically
                        currentMesh->ModifierVertexMove(TRUE, TRUE);
                    }
                }
            }
        }
    }

    return CK_OK;
}

// Based on IDA decompilation at 0x10058042
CKERROR RCKObjectAnimation::SetFrame(float frame, CKKeyedAnimation *anim) {
    if (!m_KeyframeData || m_KeyframeData->m_Length == 0.0f)
        return CKERR_INVALIDOPERATION;  // -123 in IDA
    
    float step = frame / m_KeyframeData->m_Length;
    return SetStep(step, anim);
}

float RCKObjectAnimation::GetCurrentStep() {
    return m_CurrentStep;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - 3D Entity
//=============================================================================

// Based on IDA decompilation at 0x1005808e
void RCKObjectAnimation::Set3dEntity(CK3dEntity *ent) {
    RCK3dEntity *newEntity = reinterpret_cast<RCK3dEntity *>(ent);
    
    if (m_Entity != newEntity) {
        // Remove from old entity's animation list
        if (m_Entity) {
            reinterpret_cast<CK3dEntity *>(m_Entity)->RemoveObjectAnimation(this);
        }
        
        // Set new entity
        m_Entity = newEntity;
        
        // Add to new entity's animation list
        if (newEntity) {
            reinterpret_cast<CK3dEntity *>(newEntity)->AddObjectAnimation(this);
        }
    }
}

CK3dEntity *RCKObjectAnimation::Get3dEntity() {
    return reinterpret_cast<CK3dEntity *>(m_Entity);
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Morph
//=============================================================================

int RCKObjectAnimation::GetMorphVertexCount() {
    // Based on IDA decompilation at 0x100580F9

    // Check for merged animation (flag 0x80)
    if (IsMerged()) {
        // Return max vertex count from both sub-animations
        int count1 = m_Anim1->GetMorphVertexCount();
        int count2 = m_Anim2->GetMorphVertexCount();
        return (count1 > count2) ? count1 : count2;
    }

    // Direct morph controller vertex count
    if (m_KeyframeData && m_KeyframeData->m_MorphController) {
        // Cast to our implementation class to access m_VertexCount
        RCKMorphController *ctrl = reinterpret_cast<RCKMorphController *>(m_KeyframeData->m_MorphController);
        return ctrl->GetMorphVertexCount();
    }

    return 0;
}

//=============================================================================
// CKObjectAnimation Virtual Methods - Transition and Clone
//=============================================================================

void RCKObjectAnimation::CreateTransition(float length, CKObjectAnimation *AnimIn, float StepFrom, CKObjectAnimation *AnimOut, float StepTo, CKBOOL Veloc, CKBOOL DontTurn, CKAnimKey *startingset) {
    // Based on IDA decompilation at 0x1005A5EE
    // This creates a smooth transition between two animations by creating interpolated keyframes

    RCKObjectAnimation *animIn = (RCKObjectAnimation *) AnimIn;
    RCKObjectAnimation *animOut = (RCKObjectAnimation *) AnimOut;

    if (!animIn || !animOut)
        return;

    // Calculate actual frame times from normalized steps
    float frameFrom = StepFrom * animIn->m_KeyframeData->m_Length;
    float frameTo = StepTo * animOut->m_KeyframeData->m_Length;

    // Check what info both animations have
    CKBOOL hasScale = animIn->HasScaleInfo() && animOut->HasScaleInfo();
    CKBOOL hasPosition = animIn->HasPositionInfo() && animOut->HasPositionInfo();
    CKBOOL hasRotation = animIn->HasRotationInfo() && animOut->HasRotationInfo();
    CKBOOL hasScaleAxis = animIn->HasScaleAxisInfo() && animOut->HasScaleAxisInfo();

    // Temporary storage for evaluated keys
    VxVector startPos(0, 0, 0), endPos(0, 0, 0);
    VxVector startScale(1, 1, 1), endScale(1, 1, 1);
    VxQuaternion startRot, endRot;
    VxQuaternion startScaleAxis, endScaleAxis;

    // If no starting set provided, evaluate from AnimIn
    if (!startingset) {
        if (hasPosition) animIn->EvaluatePosition(frameFrom, startPos);
        if (hasRotation) animIn->EvaluateRotation(frameFrom, startRot);
        if (hasScale) animIn->EvaluateScale(frameFrom, startScale);
        if (hasScaleAxis) animIn->EvaluateScaleAxis(frameFrom, startScaleAxis);
    } else {
        // Use provided starting set
        // startingset is CKAnimKey* which should point to a structure containing pos, rot, scale, scaleAxis
        // For now just evaluate from AnimIn if we don't know the exact structure
        if (hasPosition) animIn->EvaluatePosition(frameFrom, startPos);
        if (hasRotation) animIn->EvaluateRotation(frameFrom, startRot);
        if (hasScale) animIn->EvaluateScale(frameFrom, startScale);
        if (hasScaleAxis) animIn->EvaluateScaleAxis(frameFrom, startScaleAxis);
    }

    // Evaluate end values from AnimOut
    if (hasPosition) animOut->EvaluatePosition(frameTo, endPos);
    if (hasRotation) animOut->EvaluateRotation(frameTo, endRot);
    if (hasScale) animOut->EvaluateScale(frameTo, endScale);
    if (hasScaleAxis) animOut->EvaluateScaleAxis(frameTo, endScaleAxis);

    // Reset transition animation state
    m_Anim1 = nullptr;
    m_Anim2 = nullptr;
    m_MergeFactor = 0.5f;
    m_CurrentStep = 0.0f;
    m_Entity = animOut->m_Entity;

    // Create position controller with 2 keys (start and end)
    if (hasPosition) {
        // Delete existing controller
        if (m_KeyframeData->m_PositionController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
            m_KeyframeData->m_PositionController = nullptr;
        }

        CKAnimController *posCtrl = CreateController(CKANIMATION_LINPOS_CONTROL);
        if (posCtrl) {
            CKPositionKey startKey(0.0f, startPos);
            CKPositionKey endKey(length, endPos);
            posCtrl->AddKey(&startKey);
            posCtrl->AddKey(&endKey);
        }
    } else {
        // Delete position controller if exists
        if (m_KeyframeData->m_PositionController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_PositionController);
            m_KeyframeData->m_PositionController = nullptr;
        }
    }

    // Create scale controller with 2 keys
    if (hasScale) {
        if (m_KeyframeData->m_ScaleController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
            m_KeyframeData->m_ScaleController = nullptr;
        }

        CKAnimController *sclCtrl = CreateController(CKANIMATION_LINSCL_CONTROL);
        if (sclCtrl) {
            CKScaleKey startKey(0.0f, startScale);
            CKScaleKey endKey(length, endScale);
            sclCtrl->AddKey(&startKey);
            sclCtrl->AddKey(&endKey);
        }
    } else {
        if (m_KeyframeData->m_ScaleController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleController);
            m_KeyframeData->m_ScaleController = nullptr;
        }
    }

    // Create rotation controller with 2 keys
    if (hasRotation) {
        if (m_KeyframeData->m_RotationController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
            m_KeyframeData->m_RotationController = nullptr;
        }

        CKAnimController *rotCtrl = CreateController(CKANIMATION_LINROT_CONTROL);
        if (rotCtrl) {
            CKRotationKey startKey(0.0f, startRot);
            CKRotationKey endKey(length, endRot);
            rotCtrl->AddKey(&startKey);
            rotCtrl->AddKey(&endKey);
        }
    } else {
        if (m_KeyframeData->m_RotationController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_RotationController);
            m_KeyframeData->m_RotationController = nullptr;
        }
    }

    // Create scale axis controller with 2 keys
    if (hasScaleAxis) {
        if (m_KeyframeData->m_ScaleAxisController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
            m_KeyframeData->m_ScaleAxisController = nullptr;
        }

        CKAnimController *sclAxisCtrl = CreateController(CKANIMATION_LINSCLAXIS_CONTROL);
        if (sclAxisCtrl) {
            CKScaleAxisKey startKey(0.0f, startScaleAxis);
            CKScaleAxisKey endKey(length, endScaleAxis);
            sclAxisCtrl->AddKey(&startKey);
            sclAxisCtrl->AddKey(&endKey);
        }
    } else {
        if (m_KeyframeData->m_ScaleAxisController) {
            delete reinterpret_cast<CKAnimController *>(m_KeyframeData->m_ScaleAxisController);
            m_KeyframeData->m_ScaleAxisController = nullptr;
        }
    }

    // Morph controller transition handling
    // Based on IDA decompilation at 0x1005A5EE (lines 391-665)
    // This creates morph transition between two animations

    // Check if both animations have morph normal info
    CKBOOL hasMorphNormals = animIn->HasMorphNormalInfo() && animOut->HasMorphNormalInfo();

    if (animOut->HasMorphInfo()) {
        int outMorphVertexCount = animOut->GetMorphVertexCount();

        // Allocate temporary buffer for morph data if needed
        int bufferSize = outMorphVertexCount * sizeof(VxVector);
        if (hasMorphNormals)
            bufferSize += outMorphVertexCount * sizeof(VxCompressedVector);
        CKMemoryPool memPool(m_Context, bufferSize);
        VxVector *tempMorphBuffer = (VxVector *) memPool.Mem();
        VxCompressedVector *tempNormalBuffer = hasMorphNormals
                                                   ? (VxCompressedVector *) ((CKBYTE *) tempMorphBuffer +
                                                       outMorphVertexCount * sizeof(VxVector))
                                                   : nullptr;

        // Check if AnimIn is this animation (special case for in-place transition)
        CKBOOL inPlaceTransition = (animIn == this);
        CKBOOL hasStartData = FALSE;

        // Case 1: AnimIn == this and both have morph with matching vertex counts
        if (inPlaceTransition && animIn->HasMorphInfo()) {
            int inMorphVertexCount = animIn->GetMorphVertexCount();
            if (outMorphVertexCount == inMorphVertexCount) {
                // Evaluate morph from AnimIn into temp buffer
                animIn->EvaluateMorphTarget(frameFrom, outMorphVertexCount,
                                            tempMorphBuffer, sizeof(VxVector), tempNormalBuffer);
                hasStartData = TRUE;
            }
        }

        // Create or reuse morph controller
        RCKMorphController *morphCtrl = reinterpret_cast<RCKMorphController *>(m_KeyframeData->m_MorphController);

        // Check if we can reuse existing controller
        if (!(morphCtrl && morphCtrl->m_NbKeys == 2 &&
            morphCtrl->GetMorphVertexCount() == outMorphVertexCount)) {
            // Create new morph controller
            morphCtrl = reinterpret_cast<RCKMorphController *>(CreateController(CKANIMATION_MORPH_CONTROL));
            if (morphCtrl) {
                morphCtrl->SetMorphVertexCount(outMorphVertexCount);
                morphCtrl->AddKey(0.0f, hasMorphNormals);
                morphCtrl->AddKey(length, hasMorphNormals);
            }
        }

        if (morphCtrl) {
            CKMorphKey *startKey = (CKMorphKey *) morphCtrl->GetKey(0);
            CKMorphKey *endKey = (CKMorphKey *) morphCtrl->GetKey(1);

            if (startKey && endKey) {
                // Ensure normal buffers match expectations
                if (hasMorphNormals) {
                    if (!startKey->NormArray)
                        startKey->NormArray = new VxCompressedVector[outMorphVertexCount];
                    if (!endKey->NormArray)
                        endKey->NormArray = new VxCompressedVector[outMorphVertexCount];
                } else {
                    delete[] startKey->NormArray;
                    startKey->NormArray = nullptr;
                    delete[] endKey->NormArray;
                    endKey->NormArray = nullptr;
                }

                if (hasStartData) {
                    // Copy start morph data from temp buffer
                    if (startKey->PosArray)
                        memcpy(startKey->PosArray, tempMorphBuffer, outMorphVertexCount * sizeof(VxVector));
                    if (hasMorphNormals && startKey->NormArray && tempNormalBuffer)
                        memcpy(startKey->NormArray, tempNormalBuffer, outMorphVertexCount * sizeof(VxCompressedVector));
                } else if (animIn && animIn->HasMorphInfo()) {
                    // Evaluate start from AnimIn
                    int inMorphVertexCount = animIn->GetMorphVertexCount();
                    if (outMorphVertexCount == inMorphVertexCount) {
                        animIn->EvaluateMorphTarget(frameFrom, outMorphVertexCount,
                                                    startKey->PosArray, sizeof(VxVector), startKey->NormArray);
                    }
                } else if (m_Entity && m_Entity->m_CurrentMesh) {
                    // Fallback: copy from current mesh vertices
                    RCKMesh *currentMesh = m_Entity->m_CurrentMesh;
                    int meshVertexCount = currentMesh->GetModifierVertexCount();
                    if (meshVertexCount <= outMorphVertexCount && startKey->PosArray) {
                        CKDWORD vertexStride = 0;
                        CKBYTE *meshVertices = currentMesh->GetModifierVertices(&vertexStride);
                        if (meshVertices) {
                            for (int i = 0; i < meshVertexCount; i++) {
                                memcpy(&startKey->PosArray[i], meshVertices, sizeof(VxVector));
                                meshVertices += vertexStride;
                            }
                        }

                        // Copy normals from mesh if needed
                        if (startKey->NormArray && animOut->HasMorphNormalInfo()) {
                            CKDWORD normalStride = 0;
                            VxVector *meshNormals = (VxVector *) currentMesh->GetNormalsPtr(&normalStride);
                            if (meshNormals) {
                                for (int i = 0; i < meshVertexCount; i++) {
                                    // Use operator= for conversion
                                    startKey->NormArray[i] = *meshNormals;
                                    meshNormals = (VxVector *) ((CKBYTE *) meshNormals + normalStride);
                                }
                            }
                        }
                    }
                }

                // Evaluate end morph from AnimOut
                animOut->EvaluateMorphTarget(frameTo, outMorphVertexCount,
                                             endKey->PosArray, sizeof(VxVector), endKey->NormArray);
            }
        }
    } else {
        // AnimOut has no morph info, clear morph controller
        if (m_KeyframeData->m_MorphController) {
            delete reinterpret_cast<CKMorphController *>(m_KeyframeData->m_MorphController);
            m_KeyframeData->m_MorphController = nullptr;
        }
    }
}

void RCKObjectAnimation::Clone(CKObjectAnimation *anim) {
    // Based on IDA decompilation at 0x10056C1F
    // Clear current data first
    ClearAll();

    RCKObjectAnimation *srcAnim = (RCKObjectAnimation *) anim;
    if (srcAnim) {
        // Share data from source - this copies the keyframe data reference
        ShareDataFrom(srcAnim);

        // Copy other members
        m_Entity = srcAnim->m_Entity;
        m_Anim1 = srcAnim->m_Anim1;
        m_Anim2 = srcAnim->m_Anim2;
        m_MergeFactor = srcAnim->m_MergeFactor;
        m_CurrentStep = srcAnim->m_CurrentStep;
    }
}

// Ensure keyframe data is exclusive to this animation and clear controllers
void RCKObjectAnimation::ResetKeyframeData() {
    // Strictly matches original CK2_3D.dll behavior (sub_1004A48A):
    // delete all controllers, set pointers to null, set length to 0.
    if (!m_KeyframeData)
        m_KeyframeData = new CKKeyframeData();

    delete m_KeyframeData->m_PositionController;
    m_KeyframeData->m_PositionController = nullptr;

    delete m_KeyframeData->m_ScaleController;
    m_KeyframeData->m_ScaleController = nullptr;

    delete m_KeyframeData->m_RotationController;
    m_KeyframeData->m_RotationController = nullptr;

    delete m_KeyframeData->m_ScaleAxisController;
    m_KeyframeData->m_ScaleAxisController = nullptr;

    delete m_KeyframeData->m_MorphController;
    m_KeyframeData->m_MorphController = nullptr;

    m_KeyframeData->m_Length = 0.0f;
    m_KeyframeData->m_ObjectAnimation = this;
}

void RCKObjectAnimation::SetKeyframeLength(float length) {
    if (!m_KeyframeData)
        return;
    m_KeyframeData->m_Length = length;
    if (m_KeyframeData->m_PositionController)
        m_KeyframeData->m_PositionController->SetLength(length);
    if (m_KeyframeData->m_ScaleController)
        m_KeyframeData->m_ScaleController->SetLength(length);
    if (m_KeyframeData->m_RotationController)
        m_KeyframeData->m_RotationController->SetLength(length);
    if (m_KeyframeData->m_ScaleAxisController)
        m_KeyframeData->m_ScaleAxisController->SetLength(length);
    if (m_KeyframeData->m_MorphController)
        m_KeyframeData->m_MorphController->SetLength(length);
}

CKSTRING RCKObjectAnimation::GetClassName() {
    return (CKSTRING) "ObjectAnimation";
}

int RCKObjectAnimation::GetDependenciesCount(int mode) {
    return 0;
}

CKSTRING RCKObjectAnimation::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCKObjectAnimation::Register() {
    // Based on IDA decompilation
    CKCLASSNOTIFYFROMCID(RCKObjectAnimation, CKCID_3DENTITY);
    CKCLASSNOTIFYFROMCID(RCKObjectAnimation, CKCID_OBJECTANIMATION);
    CKPARAMETERFROMCLASS(RCKObjectAnimation, CKPGUID_OBJECTANIMATION);
    CKCLASSDEFAULTOPTIONS(RCKObjectAnimation, CK_GENERALOPTIONS_NODUPLICATENAMECHECK);
}

CKObjectAnimation *RCKObjectAnimation::CreateInstance(CKContext *Context) {
    return new RCKObjectAnimation(Context, nullptr);
}
