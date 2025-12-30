#include "RCKPatchMesh.h"

#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "CKMaterial.h"
#include "CKRenderEngineTypes.h"

CK_CLASSID RCKPatchMesh::m_ClassID = CKCID_PATCHMESH;

static void PatchMeshPreRenderCallback(CKRenderContext *rc, CK3dEntity *ent, CKMesh *mesh, void *data);

//=============================================================================
// Constructor/Destructor
//=============================================================================

/**
 * @brief Constructor for RCKPatchMesh
 * Based on IDA analysis at 0x1003056e
 */
RCKPatchMesh::RCKPatchMesh(CKContext *Context, CKSTRING name)
    : RCKMesh(Context, name),
      m_Verts(nullptr),
      m_Vecs(nullptr),
      m_VertCount(0),
      m_VecCount(0),
      m_PatchFlags(CK_PATCHMESH_BUILDNORMALS),
      m_TextureChannelCount(1),
      m_IterationCount(0),
      m_TessVertexBase(0),
      m_TessFaceBase(0),
      m_TessEdgeVertexCount(0),
      m_TessInteriorVertexCount(0),
      m_TessTotalVertices(0),
      m_TessWorkData0(0),
      m_TessWorkData1(0),
      m_TessWorkData2(0),
      m_PatchChanged(TRUE),
      m_CornerVertexMap(nullptr) {
    // Initialize tessellation work vectors
    for (int i = 0; i < 3; ++i) {
        m_TessWorkVectors[i] = VxVector(0, 0, 0);
    }

    // Initialize texture patches with one default channel
    m_TexturePatches.Resize(1);

    // Register pre-render callback for BuildRenderMesh
    RCKMesh::AddPreRenderCallBack(PatchMeshPreRenderCallback, this, FALSE);
}

/**
 * @brief Destructor for RCKPatchMesh
 */
RCKPatchMesh::~RCKPatchMesh() {
    // Clear releases all allocated memory
    RCKPatchMesh::Clear();

    // Remove the pre-render callback
    RCKMesh::RemovePreRenderCallBack(PatchMeshPreRenderCallback, this);
}

//=============================================================================
// Pre-Render Callback
//=============================================================================

/**
 * @brief Pre-render callback that triggers mesh building
 * Based on IDA analysis at 0x10030540
 */
static void PatchMeshPreRenderCallback(CKRenderContext *rc, CK3dEntity *ent, CKMesh *mesh, void *data) {
    if (CKIsChildClassOf(mesh, CKCID_PATCHMESH)) {
        static_cast<RCKPatchMesh *>(mesh)->BuildRenderMesh();
    }
}

//=============================================================================
// Clear Method
//=============================================================================

/**
 * @brief Clears all patch data from the mesh
 * Based on IDA analysis at 0x100307a9
 */
void RCKPatchMesh::Clear() {
    // Free vertex/vector array
    delete[] m_Verts;
    m_Verts = nullptr;
    m_Vecs = nullptr;

    // Clear patches array
    m_Patches.Clear();

    // Reset texture patches to single default channel
    m_TexturePatches.Resize(1);
    m_TextureChannelCount = 1;

    // Clear the first texture channel's data
    if (m_TexturePatches.Size() > 0) {
        m_TexturePatches[0].Patches.Clear();
        m_TexturePatches[0].UVs.Clear();
    }

    // Free corner vertex map allocated data
    if (m_CornerVertexMap) {
        delete[] m_CornerVertexMap;
        m_CornerVertexMap = nullptr;
    }

    // Clear shared vertex sources array
    m_SharedVertexSources.Resize(0);

    // Reset state
    m_VertCount = 0;
    m_VecCount = 0;
    m_PatchChanged = TRUE;
    m_PatchFlags = CK_PATCHMESH_BUILDNORMALS; // = 2
}

//=============================================================================
// Load Methods
//=============================================================================

/**
 * @brief Loads patch mesh vertices from a state chunk
 * Based on IDA analysis at 0x1003afd8
 */
void RCKPatchMesh::LoadVertices(CKStateChunk *chunk) {
    if (chunk->SeekIdentifier(CK_STATESAVE_PATCHMESHDATA3)) {
        // New format (0x8000000)
        CKDWORD patchFlags = chunk->ReadDword();
        patchFlags |= CK_PATCHMESH_BUILDNORMALS;
        patchFlags &= ~CK_PATCHMESH_UPTODATE;
        m_PatchFlags = patchFlags;

        chunk->ReadInt(); // unused

        const int vecCount = chunk->ReadInt();
        const CKDWORD bufferSize = chunk->ReadDword();
        const CKDWORD totalCount = chunk->ReadDword();

        if (totalCount) {
            const int expectedTotal = m_VertCount + m_VecCount;
            if (!m_Verts || expectedTotal != static_cast<int>(totalCount) || vecCount != m_VecCount) {
                SetVertVecCount(static_cast<int>(totalCount) - vecCount, vecCount);
            }
            chunk->ReadAndFillBuffer_LEndian(bufferSize, m_Verts);
        }

        m_VecCount = vecCount;
        m_VertCount = static_cast<int>(totalCount) - m_VecCount;
        m_Vecs = (m_Verts && m_VecCount > 0) ? &m_Verts[m_VertCount] : nullptr;
    } else if (chunk->SeekIdentifier(CK_STATESAVE_PATCHMESHDATA2)) {
        // Old format (0x1000000)
        CKDWORD patchFlags = chunk->ReadDword();
        patchFlags |= CK_PATCHMESH_BUILDNORMALS;
        patchFlags &= ~CK_PATCHMESH_UPTODATE;
        m_PatchFlags = patchFlags;

        chunk->ReadObjectID(); // default material (legacy)
        chunk->ReadInt();      // unused

        const int vecCount = chunk->ReadInt();
        const CKDWORD bufferSize = chunk->ReadDword();
        const CKDWORD totalCount = chunk->ReadDword();

        if (totalCount) {
            const int expectedTotal = m_VertCount + m_VecCount;
            if (!m_Verts || expectedTotal != static_cast<int>(totalCount) || vecCount != m_VecCount) {
                SetVertVecCount(static_cast<int>(totalCount) - vecCount, vecCount);
            }
            chunk->ReadAndFillBuffer_LEndian(bufferSize, m_Verts);
        }

        m_VecCount = vecCount;
        m_VertCount = static_cast<int>(totalCount) - m_VecCount;
        m_Vecs = (m_Verts && m_VecCount > 0) ? &m_Verts[m_VertCount] : nullptr;
    } else {
        // Obsolete format - cannot load
        const char *v7 = GetName();
        m_Context->OutputToConsoleExBeep("%s : Obsolete version of PatchMesh format: Cannot Load", v7);
    }
}

/**
 * @brief Loads patch mesh data from a state chunk
 * Based on IDA analysis at 0x1003b208
 */
CKERROR RCKPatchMesh::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    RCKMesh::Load(chunk, file);

    if (chunk->SeekIdentifier(CK_STATESAVE_PATCHMESHDATA3)) {
        // New format
        Clear();

        CKDWORD patchFlags = chunk->ReadDword();
        patchFlags |= CK_PATCHMESH_BUILDNORMALS;
        patchFlags &= ~static_cast<CKDWORD>(CK_PATCHMESH_UPTODATE | CK_PATCHMESH_MATERIALSUPTODATE);
        m_PatchFlags = patchFlags;

        SetIterationCount(chunk->ReadInt());

        m_VecCount = chunk->ReadInt();
        const CKDWORD bufferSize = chunk->ReadDword();
        const CKDWORD totalCount = chunk->ReadDword();

        if (totalCount) {
            m_Verts = new VxVector[totalCount];
            chunk->ReadAndFillBuffer_LEndian(bufferSize, m_Verts);
        }

        m_VertCount = static_cast<int>(totalCount) - m_VecCount;
        m_Vecs = (m_Verts && m_VecCount > 0) ? &m_Verts[m_VertCount] : nullptr;

        // Patches
        const int patchCount = chunk->StartReadSequence();
        SetPatchCount(patchCount);

        // First pass: materials
        for (int i = 0; i < patchCount; ++i) {
            m_Patches[i].Material = chunk->ReadObjectID();
        }

        // Second pass: patch data
        for (int i = 0; i < patchCount; ++i) {
            m_Patches[i].type = chunk->ReadDword();
            m_Patches[i].SmoothingGroup = chunk->ReadDword();
            chunk->ReadAndFillBuffer_LEndian16(40, (void *) m_Patches[i].v);
        }

        // Patch edges
        const CKDWORD edgeBufferSize = chunk->ReadDword();
        const int edgeCount = static_cast<int>(chunk->ReadDword());
        if (edgeCount) {
            m_PatchEdges.Resize(edgeCount);
            chunk->ReadAndFillBuffer_LEndian16(edgeBufferSize, m_PatchEdges.Begin());
        } else {
            m_PatchEdges.Clear();
        }

        // Texture patch channels
        const int channelCount = chunk->StartReadSequence();
        m_TexturePatches.Resize(channelCount);

        // First pass: materials
        for (int i = 0; i < channelCount; ++i) {
            m_TexturePatches[i].Material = chunk->ReadObjectID();
        }

        // Second pass: channel data
        for (int i = 0; i < m_TexturePatches.Size(); ++i) {
            m_TexturePatches[i].Flags = chunk->ReadDword();
            m_TexturePatches[i].Type = chunk->ReadDword();
            m_TexturePatches[i].SubType = chunk->ReadDword();

            const CKDWORD patchesBufferSize = chunk->ReadDword();
            const int patchesCount = static_cast<int>(chunk->ReadDword());
            if (patchesCount) {
                m_TexturePatches[i].Patches.Resize(patchesCount);
                chunk->ReadAndFillBuffer_LEndian16(patchesBufferSize, m_TexturePatches[i].Patches.Begin());
            } else {
                m_TexturePatches[i].Patches.Clear();
            }

            const CKDWORD uvsBufferSize = chunk->ReadDword();
            const int uvsCount = static_cast<int>(chunk->ReadDword());
            if (uvsCount) {
                m_TexturePatches[i].UVs.Resize(uvsCount);
                chunk->ReadAndFillBuffer_LEndian(uvsBufferSize, m_TexturePatches[i].UVs.Begin());
            } else {
                m_TexturePatches[i].UVs.Clear();
            }
        }
    } else if (chunk->SeekIdentifier(CK_STATESAVE_PATCHMESHDATA2)) {
        // Legacy format
        Clear();

        CKDWORD patchFlags = chunk->ReadDword();
        patchFlags |= CK_PATCHMESH_BUILDNORMALS;
        patchFlags &= ~CK_PATCHMESH_UPTODATE;
        m_PatchFlags = patchFlags;

        const CK_ID defaultMaterial = chunk->ReadObjectID();
        SetIterationCount(chunk->ReadInt());

        m_VecCount = chunk->ReadInt();
        const CKDWORD bufferSize = chunk->ReadDword();
        const CKDWORD totalCount = chunk->ReadDword();

        if (totalCount) {
            m_Verts = new VxVector[totalCount];
            chunk->ReadAndFillBuffer_LEndian(bufferSize, m_Verts);
        }

        m_VertCount = static_cast<int>(totalCount) - m_VecCount;
        m_Vecs = (m_Verts && m_VecCount > 0) ? &m_Verts[m_VertCount] : nullptr;

        // Legacy patch records (88 bytes per patch)
        const CKDWORD patchBufSize = chunk->ReadDword();
        const int patchCount = static_cast<int>(chunk->ReadDword());

        void *legacyPatches = nullptr;
        if (patchCount) {
            legacyPatches = operator new(88 * patchCount);
            chunk->ReadAndFillBuffer_LEndian(patchBufSize, legacyPatches);
        }

        SetPatchCount(patchCount);
        for (int i = 0; i < patchCount; ++i) {
            CKPatch &dst = m_Patches[i];
            dst.auxs = nullptr;
            dst.SmoothingGroup = 0xFFFFFFFF;
            dst.Material = defaultMaterial;
            for (int k = 0; k < 4; ++k)
                dst.v[k] = 0;
            for (int k = 0; k < 8; ++k)
                dst.vec[k] = 0;
            for (int k = 0; k < 4; ++k)
                dst.interior[k] = 0;
            for (int k = 0; k < 4; ++k)
                dst.edge[k] = 0;

            const CKBYTE *rec = static_cast<const CKBYTE *>(legacyPatches) + 88 * i;
            dst.type = *reinterpret_cast<const CKDWORD *>(rec + 0);

            dst.v[0] = *reinterpret_cast<const short *>(rec + 4);
            dst.v[1] = *reinterpret_cast<const short *>(rec + 8);
            dst.v[2] = *reinterpret_cast<const short *>(rec + 12);
            dst.v[3] = *reinterpret_cast<const short *>(rec + 16);

            // vecs stored as 32-bit values, low 16 bits are used
            dst.vec[0] = *reinterpret_cast<const short *>(rec + 20);
            dst.vec[1] = *reinterpret_cast<const short *>(rec + 24);
            dst.vec[2] = *reinterpret_cast<const short *>(rec + 28);
            dst.vec[3] = *reinterpret_cast<const short *>(rec + 32);
            dst.vec[4] = *reinterpret_cast<const short *>(rec + 36);
            dst.vec[5] = *reinterpret_cast<const short *>(rec + 40);
            dst.vec[6] = *reinterpret_cast<const short *>(rec + 44);
            dst.vec[7] = *reinterpret_cast<const short *>(rec + 48);

            dst.interior[0] = *reinterpret_cast<const short *>(rec + 52);
            dst.interior[1] = *reinterpret_cast<const short *>(rec + 56);
            dst.interior[2] = *reinterpret_cast<const short *>(rec + 60);
            dst.interior[3] = *reinterpret_cast<const short *>(rec + 64);
        }

        // Legacy edge records (24 bytes per edge)
        const CKDWORD edgeBufSize = chunk->ReadDword();
        const int edgeCount = static_cast<int>(chunk->ReadDword());
        void *legacyEdges = nullptr;
        if (edgeCount) {
            legacyEdges = operator new(24 * edgeCount);
            chunk->ReadAndFillBuffer_LEndian(edgeBufSize, legacyEdges);
        }

        SetEdgeCount(edgeCount);
        for (int i = 0; i < edgeCount; ++i) {
            const CKBYTE *rec = static_cast<const CKBYTE *>(legacyEdges) + 24 * i;
            const CKWORD v1 = *reinterpret_cast<const CKWORD *>(rec + 0);
            const CKWORD vec12 = *reinterpret_cast<const CKWORD *>(rec + 4);
            const CKWORD vec21 = *reinterpret_cast<const CKWORD *>(rec + 8);
            const CKWORD v2 = *reinterpret_cast<const CKWORD *>(rec + 12);
            const CKWORD patch1 = *reinterpret_cast<const CKWORD *>(rec + 16);
            const CKWORD patch2 = *reinterpret_cast<const CKWORD *>(rec + 20);

            m_PatchEdges[i].v1 = static_cast<short>(v1);
            m_PatchEdges[i].vec12 = static_cast<short>(vec12);
            m_PatchEdges[i].vec21 = static_cast<short>(vec21);
            m_PatchEdges[i].v2 = static_cast<short>(v2);
            m_PatchEdges[i].patch1 = static_cast<short>(patch1);
            m_PatchEdges[i].patch2 = static_cast<short>(patch2);
        }

        // Legacy TVPatch + UVs only exist for default channel (-1 => index 0)
        const CKDWORD tvPatchBufSize = chunk->ReadDword();
        const int tvPatchCount = static_cast<int>(chunk->ReadDword());
        void *legacyTVPatches = nullptr;
        if (tvPatchCount) {
            legacyTVPatches = operator new(16 * tvPatchCount);
            chunk->ReadAndFillBuffer_LEndian(tvPatchBufSize, legacyTVPatches);
        }

        const CKDWORD uvBufSize = chunk->ReadDword();
        const int uvCount = static_cast<int>(chunk->ReadDword());
        void *legacyUVs = nullptr;
        if (uvCount) {
            legacyUVs = operator new(8 * uvCount);
            chunk->ReadAndFillBuffer_LEndian(uvBufSize, legacyUVs);
        }

        SetTVPatchCount(tvPatchCount, -1);
        for (int i = 0; i < tvPatchCount; ++i) {
            const CKBYTE *rec = static_cast<const CKBYTE *>(legacyTVPatches) + 16 * i;
            CKTVPatch &dst = m_TexturePatches[0].Patches[i];
            dst.tv[0] = *reinterpret_cast<const short *>(rec + 0);
            dst.tv[1] = *reinterpret_cast<const short *>(rec + 4);
            dst.tv[2] = *reinterpret_cast<const short *>(rec + 8);
            dst.tv[3] = *reinterpret_cast<const short *>(rec + 12);
        }

        SetTVCount(uvCount, -1);
        if (uvCount) {
            memcpy(m_TexturePatches[0].UVs.Begin(), legacyUVs, 8 * uvCount);
        }

        // Optional smoothing group array
        if (chunk->SeekIdentifier(CK_STATESAVE_PATCHMESHSMOOTH)) {
            const CKDWORD smoothBufSize = chunk->ReadDword();
            const int smoothCount = static_cast<int>(chunk->ReadDword());
            if (smoothCount > 0) {
                CKDWORD *smooth = static_cast<CKDWORD *>(operator new(sizeof(CKDWORD) * smoothCount));
                chunk->ReadAndFillBuffer_LEndian(smoothBufSize, smooth);
                const int applyCount = (patchCount < smoothCount) ? patchCount : smoothCount;
                for (int i = 0; i < applyCount; ++i) {
                    m_Patches[i].SmoothingGroup = smooth[i];
                }
                operator delete(smooth);
            }
        }

        // Optional per-patch materials
        if (chunk->SeekIdentifier(CK_STATESAVE_PATCHMESHMATERIALS)) {
            const int seqCount = chunk->StartReadSequence();
            const int applyCount = (patchCount < seqCount) ? patchCount : seqCount;
            for (int i = 0; i < applyCount; ++i) {
                m_Patches[i].Material = chunk->ReadObjectID();
            }
        }

        operator delete(legacyPatches);
        operator delete(legacyEdges);
        operator delete(legacyTVPatches);
        operator delete(legacyUVs);
    } else {
        CKSTRING name = GetName();
        m_Context->OutputToConsoleExBeep("%s : Obsolete version of PatchMesh format: Cannot Load", name);
    }

    BuildRenderMesh();
    return CK_OK;
}

/**
 * @brief Saves patch mesh data to a state chunk
 * Based on IDA analysis at 0x1003ac69
 */
CKStateChunk *RCKPatchMesh::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *baseChunk = RCKMesh::Save(file, flags);
    if (!file && (flags & CK_STATESAVE_PATCHMESHONLY) == 0)
        return baseChunk;

    CKStateChunk *chunk = CreateCKStateChunk(CKCID_PATCHMESH, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    chunk->WriteIdentifier(CK_STATESAVE_PATCHMESHDATA3);
    chunk->WriteDword(m_PatchFlags);
    chunk->WriteInt(m_IterationCount);
    chunk->WriteInt(m_VecCount);

    // WriteArray_LEndian(count, stride, data): [dword bytes][dword count][raw bytes]
    const int totalCount = m_VertCount + m_VecCount;
    const CKDWORD vertBytes = static_cast<CKDWORD>(totalCount * 12);
    chunk->WriteDword(vertBytes);
    chunk->WriteDword(static_cast<CKDWORD>(totalCount));
    if (vertBytes)
        chunk->WriteBufferNoSize_LEndian(static_cast<int>(vertBytes), m_Verts);

    // Write patches
    int patchCount = m_Patches.Size();
    chunk->StartObjectIDSequence(patchCount);

    // First pass: write material IDs
    for (int i = 0; i < patchCount; ++i) {
        CKObject *mat = m_Context->GetObject(m_Patches[i].Material);
        chunk->WriteObjectSequence(mat);
    }

    // Second pass: write patch data
    for (int i = 0; i < patchCount; ++i) {
        chunk->WriteDword(m_Patches[i].type);
        chunk->WriteDword(m_Patches[i].SmoothingGroup);
        chunk->WriteBufferNoSize_LEndian16(40, (void *) m_Patches[i].v);
    }

    // WriteArray_LEndian16(count, stride, data): [dword bytes][dword count][raw bytes]
    const int edgeCount = m_PatchEdges.Size();
    const CKDWORD edgeBytes = static_cast<CKDWORD>(edgeCount * 12);
    chunk->WriteDword(edgeBytes);
    chunk->WriteDword(static_cast<CKDWORD>(edgeCount));
    if (edgeBytes)
        chunk->WriteBufferNoSize_LEndian16(static_cast<int>(edgeBytes), m_PatchEdges.Begin());

    // Write texture patch channels
    int channelCount = m_TexturePatches.Size();
    chunk->StartObjectIDSequence(channelCount);

    // First pass: write material IDs
    for (int i = 0; i < channelCount; ++i) {
        CKObject *mat = m_Context->GetObject(m_TexturePatches[i].Material);
        chunk->WriteObjectSequence(mat);
    }

    // Second pass: write channel data
    for (int i = 0; i < channelCount; ++i) {
        chunk->WriteDword(m_TexturePatches[i].Flags);
        chunk->WriteDword(m_TexturePatches[i].Type);
        chunk->WriteDword(m_TexturePatches[i].SubType);

        // WriteArray_LEndian16(TVPatch)
        const int patchesCount = m_TexturePatches[i].Patches.Size();
        const CKDWORD patchesBytes = static_cast<CKDWORD>(patchesCount * 8);
        chunk->WriteDword(patchesBytes);
        chunk->WriteDword(static_cast<CKDWORD>(patchesCount));
        if (patchesBytes)
            chunk->WriteBufferNoSize_LEndian16(static_cast<int>(patchesBytes), m_TexturePatches[i].Patches.Begin());

        // WriteArray_LEndian(UV)
        const int uvsCount = m_TexturePatches[i].UVs.Size();
        const CKDWORD uvsBytes = static_cast<CKDWORD>(uvsCount * 8);
        chunk->WriteDword(uvsBytes);
        chunk->WriteDword(static_cast<CKDWORD>(uvsCount));
        if (uvsBytes)
            chunk->WriteBufferNoSize_LEndian(static_cast<int>(uvsBytes), m_TexturePatches[i].UVs.Begin());
    }

    if (GetClassID() == CKCID_PATCHMESH)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();

    return chunk;
}

/**
 * @brief Prepares patch mesh for saving by saving referenced objects
 * Based on IDA analysis at 0x1003c333
 */
void RCKPatchMesh::PreSave(CKFile *file, CKDWORD flags) {
    RCKMesh::PreSave(file, flags);

    // Save patch materials
    for (int i = 0; i < m_Patches.Size(); ++i) {
        CKObject *mat = m_Context->GetObject(m_Patches[i].Material);
        if (mat)
            file->SaveObject(mat, flags);
    }

    // Save texture channel materials (skip default channel 0)
    for (int i = 1; i < m_TexturePatches.Size(); ++i) {
        CKObject *mat = m_Context->GetObject(m_TexturePatches[i].Material);
        if (mat)
            file->SaveObject(mat, flags);
    }
}

//=============================================================================
// Static Class Methods (for class registration)
//=============================================================================

CKSTRING RCKPatchMesh::GetClassName() {
    return (CKSTRING) "Patch Mesh";
}

int RCKPatchMesh::GetDependenciesCount(int mode) {
    return 0;
}

CKSTRING RCKPatchMesh::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCKPatchMesh::Register() {
    CKCLASSNOTIFYFROMCID(RCKPatchMesh, CKCID_MATERIAL);
    CKPARAMETERFROMCLASS(RCKPatchMesh, CKPGUID_PATCHMESH);
    CKCLASSDEFAULTOPTIONS(RCKPatchMesh, CK_GENERALOPTIONS_CANUSECURRENTOBJECT);
}

CKPatchMesh *RCKPatchMesh::CreateInstance(CKContext *Context) {
    RCKPatchMesh *patchMesh = new RCKPatchMesh(Context, nullptr);
    return reinterpret_cast<CKPatchMesh *>(patchMesh);
}

//=============================================================================
// CKPatchMesh Virtual Methods Implementation
//=============================================================================

CK_CLASSID RCKPatchMesh::GetClassID() {
    return m_ClassID;
}

/**
 * @brief Returns memory occupation of the patch mesh
 * Based on IDA analysis at 0x1003abb7
 */
int RCKPatchMesh::GetMemoryOccupation() {
    int base = RCKMesh::GetMemoryOccupation() + (sizeof(RCKPatchMesh) - sizeof(RCKMesh));

    // Vertex and vector memory (VxVector = 12 bytes)
    int vertVecMem = sizeof(VxVector) * (m_VertCount + m_VecCount);

    // Calculate texture channel memory
    int channelMem = m_TexturePatches.GetMemoryOccupation(FALSE); // sizeof(CKPatchChannel) each
    for (int i = 0; i < m_TexturePatches.Size(); ++i) {
        int patchesSize = m_TexturePatches[i].Patches.GetMemoryOccupation(FALSE);
        int uvsSize = m_TexturePatches[i].UVs.GetMemoryOccupation(FALSE);
        channelMem += sizeof(Vx2DVector) * patchesSize + sizeof(Vx2DVector) * uvsSize + sizeof(CKPatchChannel);
    }

    // Patches memory (sizeof(CKPatch) each)
    int patchesMem = m_Patches.GetMemoryOccupation(FALSE);

    // Edges memory (sizeof(CKPatchEdge) each)
    int edgesMem = m_PatchEdges.GetMemoryOccupation(FALSE);

    return base + vertVecMem + channelMem + patchesMem + edgesMem;
}

/**
 * @brief Copy patch mesh from another object
 * Based on IDA analysis at 0x1003cd40
 */
CKERROR RCKPatchMesh::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = RCKMesh::Copy(o, context);
    if (err != CK_OK)
        return err;

    context.GetClassDependencies(CKCID_MESH);

    // Save the source object and load into this
    CKStateChunk *chunk = o.Save(nullptr, CK_STATESAVE_PATCHMESHONLY);
    if (!chunk)
        return CKERR_OUTOFMEMORY;

    Load(chunk, nullptr);
    delete chunk;

    return CK_OK;
}

/**
 * @brief Convert from a regular CKMesh (not implemented)
 * Based on IDA analysis at 0x1003412a - returns CKERR_NOTIMPLEMENTED
 */
CKERROR RCKPatchMesh::FromMesh(CKMesh *m) {
    return CKERR_NOTIMPLEMENTED;
}

/**
 * @brief Convert to a regular CKMesh (not implemented)
 * Based on IDA analysis at 0x1003413c - returns CKERR_NOTIMPLEMENTED
 */
CKERROR RCKPatchMesh::ToMesh(CKMesh *m, int stepcount) {
    return CKERR_NOTIMPLEMENTED;
}

/**
 * @brief Set the iteration count for tessellation
 * Based on IDA analysis at 0x100308ad
 */
void RCKPatchMesh::SetIterationCount(int count) {
    if (m_IterationCount != count) {
        if (count < 0)
            count = 0;
        m_IterationCount = count;
        m_PatchFlags &= ~CK_PATCHMESH_UPTODATE;    // Clear UPTODATE flag
        m_PatchFlags |= CK_PATCHMESH_BUILDNORMALS; // Set BUILDNORMALS flag
    }
}

int RCKPatchMesh::GetIterationCount() {
    return m_IterationCount;
}

/**
 * @brief Build the render mesh from patch data
 * Based on IDA analysis at 0x1003424f
 *
 * This is the core tessellation function that converts patch data into
 * a renderable triangle mesh. It handles:
 * - Triangle patches (3 corners) and quad patches (4 corners)
 * - Configurable tessellation level via m_IterationCount
 * - Smooth edge handling for normal interpolation
 * - Texture coordinate generation
 *
 * The original function is ~25KB (0x609f bytes), making it one of the
 * largest functions in CK2_3D.dll.
 */
void RCKPatchMesh::BuildRenderMesh() {
    // Early exit conditions
    if ((m_PatchFlags & CK_PATCHMESH_UPTODATE) != 0)
        return;
    if (m_PatchEdges.Size() == 0)
        return;
    if (m_Patches.Size() == 0)
        return;
    if (m_VertCount == 0)
        return;
    if (m_VecCount == 0)
        return;

    m_Valid = FALSE;

    int patchCount = m_Patches.Size();
    int edgeCount = m_PatchEdges.Size();
    m_TextureChannelCount = m_TexturePatches.Size();

    // Rebuild derived connectivity state when patch topology changed.
    if (m_PatchChanged) {
        if (m_CornerVertexMap) {
            delete[] m_CornerVertexMap;
            m_CornerVertexMap = nullptr;
        }

        m_SharedVertexSources.Resize(0);

        if (m_HardEdgeFlags.Size() != edgeCount)
            m_HardEdgeFlags.Resize(edgeCount);
        if (m_SmoothEdgeFlags.Size() != edgeCount)
            m_SmoothEdgeFlags.Resize(edgeCount);
        for (int i = 0; i < edgeCount; ++i) {
            m_HardEdgeFlags[i] = 0;
            m_SmoothEdgeFlags[i] = 0;
        }

        // Hard edge decision: smoothing group + UV seam (channel 0).
        for (int i = 0; i < edgeCount; ++i) {
            const CKPatchEdge &edge = m_PatchEdges[i];
            CKBOOL hard = FALSE;
            if (edge.patch2 < 0) {
                hard = TRUE;
            } else {
                const CKPatch &p1 = m_Patches[edge.patch1];
                const CKPatch &p2 = m_Patches[edge.patch2];
                if ((p1.SmoothingGroup & p2.SmoothingGroup) == 0)
                    hard = TRUE;
                if (!hard && !DoPatchesShareUVOnEdge(i))
                    hard = TRUE;
            }
            if (hard)
                m_HardEdgeFlags[i] = 1;
            else
                m_SmoothEdgeFlags[i] = 1;
        }

        // Corner duplication (shared vertices) based on smooth-connected patch components per control vertex.
        EnsureCornerVertexMapAllocated(patchCount);
        for (int i = 0; i < 4 * patchCount; ++i)
            m_CornerVertexMap[i] = (CKDWORD)-1;

        // Build per-vertex incident patch corner list.
        XArray<int> cornerPatch;
        XArray<int> cornerCorner;
        cornerPatch.Resize(0);
        cornerCorner.Resize(0);

        XArray<int> offsets;
        offsets.Resize(m_VertCount + 1);
        for (int v = 0; v <= m_VertCount; ++v)
            offsets[v] = 0;

        for (int p = 0; p < patchCount; ++p) {
            CKPatch &patch = m_Patches[p];
            int ccount = patch.type;
            if (ccount > 4)
                ccount = 4;
            for (int c = 0; c < ccount; ++c) {
                int v = patch.v[c];
                if (v >= 0 && v < m_VertCount)
                    offsets[v + 1] += 1;
            }
        }

        for (int v = 1; v <= m_VertCount; ++v)
            offsets[v] += offsets[v - 1];

        int totalCorners = offsets[m_VertCount];
        cornerPatch.Resize(totalCorners);
        cornerCorner.Resize(totalCorners);

        XArray<int> cursor;
        cursor.Resize(m_VertCount);
        for (int v = 0; v < m_VertCount; ++v)
            cursor[v] = offsets[v];

        for (int p = 0; p < patchCount; ++p) {
            CKPatch &patch = m_Patches[p];
            int ccount = patch.type;
            if (ccount > 4)
                ccount = 4;
            for (int c = 0; c < ccount; ++c) {
                int v = patch.v[c];
                if (v < 0 || v >= m_VertCount)
                    continue;
                int idx = cursor[v]++;
                cornerPatch[idx] = p;
                cornerCorner[idx] = c;
            }
        }

        // For each control vertex: flood-fill smooth connectivity across incident patches.
        XArray<CKBYTE> visited;
        for (int v = 0; v < m_VertCount; ++v) {
            int begin = offsets[v];
            int end = offsets[v + 1];
            int count = end - begin;
            if (count <= 0)
                continue;

            visited.Resize(count);
            for (int i = 0; i < count; ++i)
                visited[i] = 0;

            int componentIndex = 0;
            for (int i = 0; i < count; ++i) {
                if (visited[i])
                    continue;

                int assignedVertexIndex = v;
                if (componentIndex > 0) {
                    // Duplicate control vertex for this disconnected component.
                    int src = v;
                    m_SharedVertexSources.PushBack((void *)(size_t)src);
                    assignedVertexIndex = m_VertCount + (int)m_SharedVertexSources.Size() - 1;
                }
                componentIndex++;

                // Simple BFS queue on incident corners.
                XArray<int> queue;
                queue.Resize(0);
                queue.PushBack(i);
                visited[i] = 1;

                for (int qi = 0; qi < (int)queue.Size(); ++qi) {
                    int local = queue[qi];
                    int p = cornerPatch[begin + local];
                    int c = cornerCorner[begin + local];
                    CKPatch &patch = m_Patches[p];

                    m_CornerVertexMap[p * 4 + c] = (CKDWORD)assignedVertexIndex;

                    // Expand to other incident corners through smooth edges of this patch that touch this vertex.
                    int edgeCountLocal = patch.type;
                    if (edgeCountLocal > 4)
                        edgeCountLocal = 4;
                    for (int e = 0; e < edgeCountLocal; ++e) {
                        int eidx = patch.edge[e];
                        if (eidx < 0 || eidx >= m_PatchEdges.Size())
                            continue;
                        const CKPatchEdge &pe = m_PatchEdges[eidx];
                        if (pe.v1 != v && pe.v2 != v)
                            continue;
                        if (IsEdgeHard(eidx))
                            continue;

                        int otherPatch = (pe.patch1 == p) ? pe.patch2 : pe.patch1;
                        if (otherPatch < 0 || otherPatch >= patchCount)
                            continue;

                        // Find corresponding incident corner entry for otherPatch.
                        for (int j = 0; j < count; ++j) {
                            if (visited[j])
                                continue;
                            if (cornerPatch[begin + j] == otherPatch) {
                                visited[j] = 1;
                                queue.PushBack(j);
                            }
                        }
                    }
                }
            }
        }

        // Backfill any still-unassigned corners by reusing same control vertex index.
        for (int p = 0; p < patchCount; ++p) {
            CKPatch &patch = m_Patches[p];
            int ccount = patch.type;
            if (ccount > 4)
                ccount = 4;
            for (int c = 0; c < ccount; ++c) {
                if ((int)m_CornerVertexMap[p * 4 + c] == -1)
                    m_CornerVertexMap[p * 4 + c] = (CKDWORD)patch.v[c];
            }
        }

        m_PatchChanged = FALSE;
    } else {
        EnsureCornerVertexMapAllocated(patchCount);
    }

    // Clamp iteration count if vertex count would exceed 16-bit indices.
    int steps = m_IterationCount + 1;
    if (steps < 1)
        steps = 1;

    int totalVertices = 0;
    int totalFaces = 0;
    XArray<EdgeTessInfo> edgeTess;
    edgeTess.Resize(edgeCount);

    while (true) {
        int intermediate = steps - 1;
        int cornerVertices = m_VertCount + (int)m_SharedVertexSources.Size();
        int edgeVertices = 0;
        int interiorVertices = 0;
        totalFaces = 0;

        for (int e = 0; e < edgeCount; ++e) {
            const CKPatchEdge &edge = m_PatchEdges[e];
            CKBOOL hard = IsEdgeHard(e);
            edgeTess[e].Hard = hard;
            edgeTess[e].BasePatch1 = -1;
            edgeTess[e].BasePatch2 = -1;

            if (intermediate <= 0)
                continue;

            if (edge.patch2 >= 0 && hard)
                edgeVertices += 2 * intermediate;
            else
                edgeVertices += intermediate;
        }

        for (int p = 0; p < patchCount; ++p) {
            CKPatch &patch = m_Patches[p];
            if (patch.type == CK_PATCH_TRI) {
                totalFaces += steps * steps;
                if (steps >= 3)
                    interiorVertices += (steps - 1) * (steps - 2) / 2;
            } else if (patch.type == CK_PATCH_QUAD) {
                totalFaces += 2 * steps * steps;
                if (steps >= 2)
                    interiorVertices += (steps - 1) * (steps - 1);
            }
        }

        totalVertices = cornerVertices + edgeVertices + interiorVertices;

        if (totalVertices <= 0xFDE8 || m_IterationCount == 0)
            break;

        // Reduce iteration and retry.
        m_IterationCount -= 1;
        steps = m_IterationCount + 1;
        if (steps < 1)
            steps = 1;
    }

    // AUTOSMOOTH: recompute vecs/interiors from control vertex positions.
    if ((m_PatchFlags & CK_PATCHMESH_AUTOSMOOTH) != 0) {
        // Temporary per-control-vertex normals.
        XArray<VxVertex> temp;
        temp.Resize(m_VertCount);
        for (int i = 0; i < m_VertCount; ++i) {
            temp[i].m_Position = m_Verts[i];
            temp[i].m_Normal = VxVector(0.0f, 0.0f, 0.0f);
        }

        for (int p = 0; p < patchCount; ++p) {
            CKPatch &patch = m_Patches[p];
            if (patch.type == CK_PATCH_TRI) {
                VxVector &a = m_Verts[patch.v[0]];
                VxVector &b = m_Verts[patch.v[1]];
                VxVector &c = m_Verts[patch.v[2]];
                VxVector ab = b - a;
                VxVector ac = c - a;
                VxVector n = CrossProduct(ab, ac);
                temp[patch.v[0]].m_Normal += n;
                temp[patch.v[1]].m_Normal += n;
                temp[patch.v[2]].m_Normal += n;
            } else if (patch.type == CK_PATCH_QUAD) {
                VxVector &a = m_Verts[patch.v[0]];
                VxVector &b = m_Verts[patch.v[1]];
                VxVector &c = m_Verts[patch.v[2]];
                VxVector &d = m_Verts[patch.v[3]];
                VxVector ab = b - a;
                VxVector ac = c - a;
                VxVector ad = d - a;
                VxVector n1 = CrossProduct(ab, ac);
                VxVector n2 = CrossProduct(ac, ad);
                VxVector n = n1 + n2;
                temp[patch.v[0]].m_Normal += n;
                temp[patch.v[1]].m_Normal += n;
                temp[patch.v[2]].m_Normal += n;
                temp[patch.v[3]].m_Normal += n;
            }
        }

        NormalizeGenericFunc(temp.Begin(), temp.Size());

        // Update edge vectors.
        for (int e = 0; e < edgeCount; ++e) {
            const CKPatchEdge &edge = m_PatchEdges[e];
            VxVector p1 = m_Verts[edge.v1];
            VxVector p2 = m_Verts[edge.v2];
            VxVector d = p2 - p1;

            VxVector n1 = temp[edge.v1].m_Normal;
            VxVector n2 = temp[edge.v2].m_Normal;

            float d1 = d.x * n1.x + d.y * n1.y + d.z * n1.z;
            float d2 = d.x * n2.x + d.y * n2.y + d.z * n2.z;
            VxVector t1 = d - n1 * d1;
            VxVector t2 = d - n2 * d2;

            m_Vecs[edge.vec12] = p1 + t1 * (1.0f / 3.0f);
            m_Vecs[edge.vec21] = p2 - t2 * (1.0f / 3.0f);
        }

        for (int p = 0; p < patchCount; ++p)
            ComputePatchInteriors(p);
    }

    // Ensure mesh has the correct number of extra material channels: (textureChannels - 1).
    int desiredExtraChannels = m_TexturePatches.Size() - 1;
    if (desiredExtraChannels < 0)
        desiredExtraChannels = 0;
    while (GetChannelCount() > desiredExtraChannels)
        RemoveChannel(GetChannelCount() - 1);
    while (GetChannelCount() < desiredExtraChannels) {
        int texIndex = GetChannelCount() + 1;
        CKMaterial *mat = nullptr;
        if (texIndex >= 0 && texIndex < m_TexturePatches.Size()) {
            CK_ID matId = m_TexturePatches[texIndex].Material;
            if (matId)
                mat = (CKMaterial *)m_Context->GetObjectA(matId);
        }
        if (!mat)
            break;
        AddChannel(mat, FALSE);
    }

    // Allocate vertex + face data.
    SetVertexCount(totalVertices);
    SetFaceCount(totalFaces);

    CKDWORD posStride = 0;
    VxVector *positions = (VxVector *)GetPositionsPtr(&posStride);
    if (!positions)
        return;

    // Base control vertices (including duplicated shared sources).
    for (int i = 0; i < m_VertCount; ++i) {
        VxVector *dst = (VxVector *)((CKBYTE *)positions + (size_t)i * (size_t)posStride);
        *dst = m_Verts[i];
    }
    for (int i = 0; i < (int)m_SharedVertexSources.Size(); ++i) {
        int src = (int)(size_t)m_SharedVertexSources[i];
        if (src < 0 || src >= m_VertCount)
            src = 0;
        int dstIndex = m_VertCount + i;
        VxVector *dst = (VxVector *)((CKBYTE *)positions + (size_t)dstIndex * (size_t)posStride);
        *dst = m_Verts[src];
    }

    // Prepare storage indices.
    int writeVertex = m_VertCount + (int)m_SharedVertexSources.Size();
    int intermediate = steps - 1;
    float invSteps = 1.0f / (float)steps;

    // Edge vertex allocation bases.
    for (int e = 0; e < edgeCount; ++e) {
        const CKPatchEdge &edge = m_PatchEdges[e];
        if (intermediate <= 0) {
            edgeTess[e].BasePatch1 = -1;
            edgeTess[e].BasePatch2 = -1;
            continue;
        }

        edgeTess[e].BasePatch1 = writeVertex;
        writeVertex += intermediate;

        if (edge.patch2 >= 0 && edgeTess[e].Hard) {
            edgeTess[e].BasePatch2 = writeVertex;
            writeVertex += intermediate;
        } else {
            edgeTess[e].BasePatch2 = -1;
        }
    }

    // Interior vertex bases per patch.
    XArray<int> interiorBase;
    interiorBase.Resize(patchCount);
    for (int p = 0; p < patchCount; ++p) {
        CKPatch &patch = m_Patches[p];
        if (patch.type == CK_PATCH_TRI) {
            int count = (steps >= 3) ? ((steps - 1) * (steps - 2) / 2) : 0;
            interiorBase[p] = (count > 0) ? writeVertex : -1;
            writeVertex += count;
        } else if (patch.type == CK_PATCH_QUAD) {
            int count = (steps >= 2) ? ((steps - 1) * (steps - 1)) : 0;
            interiorBase[p] = (count > 0) ? writeVertex : -1;
            writeVertex += count;
        } else {
            interiorBase[p] = -1;
        }
    }

    // Sanity.
    if (writeVertex != totalVertices) {
        // Keep going; indices will still be within allocation for consistent formulas.
    }

    // Ensure patch auxiliary data is ready.
    for (int p = 0; p < patchCount; ++p)
        ComputePatchInteriors(p);

    // Fill edge vertices positions and UVs.
    for (int e = 0; e < edgeCount; ++e) {
        const CKPatchEdge &edge = m_PatchEdges[e];
        if (intermediate <= 0)
            continue;

        // For each participating patch side, evaluate along the boundary of that patch.
        for (int side = 0; side < 2; ++side) {
            int patchIndex = (side == 0) ? edge.patch1 : edge.patch2;
            int base = (side == 0) ? edgeTess[e].BasePatch1 : edgeTess[e].BasePatch2;
            if (patchIndex < 0 || base < 0)
                continue;

            CKPatch &patch = m_Patches[patchIndex];
            int cornerA = GetPatchCornerForVertex(patch, edge.v1);
            int cornerB = GetPatchCornerForVertex(patch, edge.v2);
            if (cornerA < 0 || cornerB < 0)
                continue;

            int vA = (int)m_CornerVertexMap[patchIndex * 4 + cornerA];
            int vB = (int)m_CornerVertexMap[patchIndex * 4 + cornerB];

            for (int i = 1; i < steps; ++i) {
                float t = (float)i * invSteps;
                VxVector pos;

                if (patch.type == CK_PATCH_TRI) {
                    // Barycentric along edge between corners.
                    float u = 0.0f, v = 0.0f, w = 0.0f;
                    if ((cornerA == 0 && cornerB == 1) || (cornerA == 1 && cornerB == 0)) {
                        u = (cornerA == 0) ? (1.0f - t) : t;
                        v = (cornerA == 0) ? t : (1.0f - t);
                        w = 0.0f;
                    } else if ((cornerA == 1 && cornerB == 2) || (cornerA == 2 && cornerB == 1)) {
                        u = 0.0f;
                        v = (cornerA == 1) ? (1.0f - t) : t;
                        w = (cornerA == 1) ? t : (1.0f - t);
                    } else {
                        v = 0.0f;
                        u = (cornerA == 0) ? (1.0f - t) : t;
                        w = (cornerA == 0) ? t : (1.0f - t);
                    }
                    EvaluateTriPatch(&patch, u, v, w, &pos);
                } else {
                    // Quad edge: map to (u,v) along boundary.
                    // Corner order: 0:(0,0) 1:(1,0) 2:(1,1) 3:(0,1)
                    float u = 0.0f, v = 0.0f;
                    if ((cornerA == 0 && cornerB == 1) || (cornerA == 1 && cornerB == 0)) {
                        u = (cornerA == 0) ? t : (1.0f - t);
                        v = 0.0f;
                    } else if ((cornerA == 1 && cornerB == 2) || (cornerA == 2 && cornerB == 1)) {
                        u = 1.0f;
                        v = (cornerA == 1) ? t : (1.0f - t);
                    } else if ((cornerA == 2 && cornerB == 3) || (cornerA == 3 && cornerB == 2)) {
                        u = (cornerA == 3) ? t : (1.0f - t);
                        v = 1.0f;
                    } else {
                        u = 0.0f;
                        v = (cornerA == 0) ? t : (1.0f - t);
                    }
                    EvaluateQuadPatch(&patch, u, v, &pos);
                }

                int outIndex = base + (i - 1);
                VxVector *dst = (VxVector *)((CKBYTE *)positions + (size_t)outIndex * (size_t)posStride);
                *dst = pos;

                // UV: linear interpolation between corner UVs.
                for (int tc = 0; tc < m_TexturePatches.Size(); ++tc) {
                    CKDWORD uvStride = 0;
                    void *uvPtr = GetTextureChannelPtr(tc, &uvStride);
                    if (!uvPtr)
                        continue;
                    float u0, v0, u1, v1;
                    if (!GetCornerTextureCoordinate(tc, patchIndex, cornerA, u0, v0))
                        continue;
                    if (!GetCornerTextureCoordinate(tc, patchIndex, cornerB, u1, v1))
                        continue;
                    float uu = u0 + (u1 - u0) * t;
                    float vv = v0 + (v1 - v0) * t;
                    WriteTextureCoordinate(uvPtr, uvStride, outIndex, uu, vv);
                }
            }
        }
    }

    // Fill interior vertices and per-corner UVs.
    for (int p = 0; p < patchCount; ++p) {
        CKPatch &patch = m_Patches[p];
        CKMaterial *patchMat = GetPatchMaterial(p);

        // Write UVs for the 3/4 corners mapped by m_CornerVertexMap.
        int ccount = patch.type;
        if (ccount > 4)
            ccount = 4;
        for (int c = 0; c < ccount; ++c) {
            int vIdx = (int)m_CornerVertexMap[p * 4 + c];
            for (int tc = 0; tc < m_TexturePatches.Size(); ++tc) {
                CKDWORD uvStride = 0;
                void *uvPtr = GetTextureChannelPtr(tc, &uvStride);
                if (!uvPtr)
                    continue;
                float uu, vv;
                if (!GetCornerTextureCoordinate(tc, p, c, uu, vv))
                    continue;
                WriteTextureCoordinate(uvPtr, uvStride, vIdx, uu, vv);
            }
        }

        int base = interiorBase[p];
        if (patch.type == CK_PATCH_QUAD && base >= 0 && steps >= 2) {
            for (int j = 1; j < steps; ++j) {
                float v = (float)j * invSteps;
                for (int i = 1; i < steps; ++i) {
                    float u = (float)i * invSteps;
                    VxVector pos;
                    EvaluateQuadPatch(&patch, u, v, &pos);
                    int idx = base + (j - 1) * (steps - 1) + (i - 1);
                    VxVector *dst = (VxVector *)((CKBYTE *)positions + (size_t)idx * (size_t)posStride);
                    *dst = pos;

                    // UV bilinear from corners.
                    for (int tc = 0; tc < m_TexturePatches.Size(); ++tc) {
                        CKDWORD uvStride = 0;
                        void *uvPtr = GetTextureChannelPtr(tc, &uvStride);
                        if (!uvPtr)
                            continue;
                        float u00, v00, u10, v10, u11, v11, u01, v01;
                        if (!GetCornerTextureCoordinate(tc, p, 0, u00, v00))
                            continue;
                        if (!GetCornerTextureCoordinate(tc, p, 1, u10, v10))
                            continue;
                        if (!GetCornerTextureCoordinate(tc, p, 2, u11, v11))
                            continue;
                        if (!GetCornerTextureCoordinate(tc, p, 3, u01, v01))
                            continue;
                        float uu = (1.0f - u) * (1.0f - v) * u00 + u * (1.0f - v) * u10 + u * v * u11 + (1.0f - u) * v * u01;
                        float vv = (1.0f - u) * (1.0f - v) * v00 + u * (1.0f - v) * v10 + u * v * v11 + (1.0f - u) * v * v01;
                        WriteTextureCoordinate(uvPtr, uvStride, idx, uu, vv);
                    }
                }
            }
        } else if (patch.type == CK_PATCH_TRI && base >= 0 && steps >= 3) {
            int cursor = 0;
            for (int row = 1; row <= steps - 2; ++row) {
                float w = (float)row * invSteps;
                int cols = steps - row - 1;
                for (int col = 1; col <= cols; ++col) {
                    float u = (float)col * invSteps;
                    float v = 1.0f - u - w;
                    if (v <= 0.0f)
                        continue;
                    VxVector pos;
                    EvaluateTriPatch(&patch, u, v, w, &pos);
                    int idx = base + cursor;
                    cursor++;
                    VxVector *dst = (VxVector *)((CKBYTE *)positions + (size_t)idx * (size_t)posStride);
                    *dst = pos;

                    // UV barycentric from corners.
                    for (int tc = 0; tc < m_TexturePatches.Size(); ++tc) {
                        CKDWORD uvStride = 0;
                        void *uvPtr = GetTextureChannelPtr(tc, &uvStride);
                        if (!uvPtr)
                            continue;
                        float u0, v0, u1, v1, u2, v2;
                        if (!GetCornerTextureCoordinate(tc, p, 0, u0, v0))
                            continue;
                        if (!GetCornerTextureCoordinate(tc, p, 1, u1, v1))
                            continue;
                        if (!GetCornerTextureCoordinate(tc, p, 2, u2, v2))
                            continue;
                        float uu = u0 * u + u1 * v + u2 * w;
                        float vv = v0 * u + v1 * v + v2 * w;
                        WriteTextureCoordinate(uvPtr, uvStride, idx, uu, vv);
                    }
                }
            }
        }

        (void)patchMat;
    }

    // Build faces.
    int face = 0;
    for (int p = 0; p < patchCount; ++p) {
        CKPatch &patch = m_Patches[p];
        CKMaterial *patchMat = GetPatchMaterial(p);

        if (patch.type == CK_PATCH_QUAD) {
            for (int j = 0; j < steps; ++j) {
                for (int i = 0; i < steps; ++i) {
                    int v00 = ComputeQuadVertexIndex(p, steps, i, j, interiorBase, edgeTess);
                    int v10 = ComputeQuadVertexIndex(p, steps, i + 1, j, interiorBase, edgeTess);
                    int v01 = ComputeQuadVertexIndex(p, steps, i, j + 1, interiorBase, edgeTess);
                    int v11 = ComputeQuadVertexIndex(p, steps, i + 1, j + 1, interiorBase, edgeTess);
                    SetFaceVertexIndex(face, v00, v10, v01);
                    SetFaceMaterial(face, patchMat);
                    face++;
                    SetFaceVertexIndex(face, v10, v11, v01);
                    SetFaceMaterial(face, patchMat);
                    face++;
                }
            }
        } else if (patch.type == CK_PATCH_TRI) {
            for (int vrow = 0; vrow < steps; ++vrow) {
                for (int ucol = 0; ucol < steps - vrow; ++ucol) {
                    int u0 = ucol;
                    int v0 = vrow;
                    int u1 = ucol + 1;
                    int v1 = vrow;
                    int u2 = ucol;
                    int v2 = vrow + 1;

                    int a = ComputeTriVertexIndex(p, steps, u0, v0, interiorBase, edgeTess);
                    int b = ComputeTriVertexIndex(p, steps, u1, v1, interiorBase, edgeTess);
                    int c = ComputeTriVertexIndex(p, steps, u2, v2, interiorBase, edgeTess);
                    SetFaceVertexIndex(face, a, b, c);
                    SetFaceMaterial(face, patchMat);
                    face++;

                    if (ucol + vrow < steps - 2) {
                        int u3 = ucol + 1;
                        int v3 = vrow + 1;
                        int d = ComputeTriVertexIndex(p, steps, u3, v3, interiorBase, edgeTess);
                        SetFaceVertexIndex(face, b, d, c);
                        SetFaceMaterial(face, patchMat);
                        face++;
                    }
                }
            }
        }
    }

    // Build normals if requested (geometry already duplicates Hard-edge vertices).
    if ((m_PatchFlags & CK_PATCHMESH_BUILDNORMALS) != 0)
        BuildNormals();

    m_PatchFlags |= (CK_PATCHMESH_UPTODATE | CK_PATCHMESH_MATERIALSUPTODATE);
    VertexMove();
}

CKBOOL RCKPatchMesh::IsEdgeHard(int edgeIndex) const {
    if (edgeIndex < 0 || edgeIndex >= m_HardEdgeFlags.Size())
        return TRUE;
    return (m_HardEdgeFlags[edgeIndex] != 0);
}

int RCKPatchMesh::GetPatchCornerForVertex(const CKPatch &patch, int vertexIndex) const {
    int cornerCount = patch.type;
    if (cornerCount > 4)
        cornerCount = 4;
    for (int c = 0; c < cornerCount; ++c) {
        if (patch.v[c] == vertexIndex)
            return c;
    }
    return -1;
}

void *RCKPatchMesh::GetTextureChannelPtr(int textureChannel, CKDWORD *strideOut) const {
    if (!strideOut)
        return nullptr;
    if (textureChannel < 0 || textureChannel >= m_TexturePatches.Size())
        return nullptr;
    int meshChannel = (textureChannel == 0) ? -1 : (textureChannel - 1);
    return const_cast<RCKPatchMesh *>(this)->GetTextureCoordinatesPtr(strideOut, meshChannel);
}

void RCKPatchMesh::WriteTextureCoordinate(void *base, CKDWORD stride, int vertexIndex, float u, float v) const {
    if (!base || stride == 0 || vertexIndex < 0)
        return;
    CKBYTE *dst = (CKBYTE *)base + (size_t)vertexIndex * (size_t)stride;
    float *uv = (float *)dst;
    uv[0] = u;
    uv[1] = v;
}

CKBOOL RCKPatchMesh::GetCornerTextureCoordinate(int textureChannel, int patchIndex, int cornerIndex, float &outU, float &outV) const {
    if (textureChannel < 0 || textureChannel >= m_TexturePatches.Size())
        return FALSE;
    const CKPatchChannel &channel = m_TexturePatches[textureChannel];
    if (patchIndex < 0 || patchIndex >= channel.Patches.Size())
        return FALSE;
    if (cornerIndex < 0 || cornerIndex >= 4)
        return FALSE;
    const CKTVPatch &tvPatch = channel.Patches[patchIndex];
    int uvIndex = tvPatch.tv[cornerIndex];
    if (uvIndex < 0 || uvIndex >= channel.UVs.Size())
        return FALSE;
    const VxUV &uv = channel.UVs[uvIndex];
    outU = uv.u;
    outV = uv.v;
    return TRUE;
}

CKBOOL RCKPatchMesh::DoPatchesShareUVOnEdge(int edgeIndex) const {
    if (edgeIndex < 0 || edgeIndex >= m_PatchEdges.Size())
        return FALSE;
    if (m_TexturePatches.Size() == 0)
        return FALSE;
    const CKPatchEdge &edge = m_PatchEdges[edgeIndex];
    if (edge.patch1 < 0 || edge.patch2 < 0)
        return FALSE;

    const CKPatch &patchA = m_Patches[edge.patch1];
    const CKPatch &patchB = m_Patches[edge.patch2];
    int cornerA1 = GetPatchCornerForVertex(patchA, edge.v1);
    int cornerA2 = GetPatchCornerForVertex(patchA, edge.v2);
    int cornerB1 = GetPatchCornerForVertex(patchB, edge.v1);
    int cornerB2 = GetPatchCornerForVertex(patchB, edge.v2);
    if (cornerA1 < 0 || cornerA2 < 0 || cornerB1 < 0 || cornerB2 < 0)
        return FALSE;

    float uA1, vA1, uA2, vA2, uB1, vB1, uB2, vB2;
    if (!GetCornerTextureCoordinate(0, edge.patch1, cornerA1, uA1, vA1) ||
        !GetCornerTextureCoordinate(0, edge.patch2, cornerB1, uB1, vB1) ||
        !GetCornerTextureCoordinate(0, edge.patch1, cornerA2, uA2, vA2) ||
        !GetCornerTextureCoordinate(0, edge.patch2, cornerB2, uB2, vB2)) {
        return FALSE;
    }

    if (uA1 != uB1 || vA1 != vB1 || uA2 != uB2 || vA2 != vB2)
        return FALSE;

    return TRUE;
}

void RCKPatchMesh::EnsureCornerVertexMapAllocated(int patchCount) {
    if (patchCount <= 0)
        return;
    int required = patchCount * 4;
    if (!m_CornerVertexMap) {
        m_CornerVertexMap = new CKDWORD[required];
    }
}

int RCKPatchMesh::TriInteriorOffset(int steps, int row, int col) const {
    int offset = 0;
    for (int r = 1; r < row; ++r) {
        offset += (steps - r - 1);
    }
    offset += (col - 1);
    return offset;
}

int RCKPatchMesh::ComputeQuadVertexIndex(int patchIndex, int steps, int i, int j, const XArray<int> &interiorBase, const XArray<EdgeTessInfo> &edgeTess) const {
    if (patchIndex < 0 || patchIndex >= m_Patches.Size() || !m_CornerVertexMap)
        return -1;
    const CKPatch &patch = m_Patches[patchIndex];
    int cornerBase = patchIndex * 4;

    if (i == 0 && j == 0)
        return (int)m_CornerVertexMap[cornerBase + 0];
    if (i == steps && j == 0)
        return (int)m_CornerVertexMap[cornerBase + 1];
    if (i == steps && j == steps)
        return (int)m_CornerVertexMap[cornerBase + 2];
    if (i == 0 && j == steps)
        return (int)m_CornerVertexMap[cornerBase + 3];

    if (j == 0) {
        int eidx = patch.edge[0];
        if (eidx < 0 || eidx >= m_PatchEdges.Size())
            return -1;
        const CKPatchEdge &e = m_PatchEdges[eidx];
        if (eidx >= edgeTess.Size())
            return -1;
        const EdgeTessInfo &info = edgeTess[eidx];
        int base = (e.patch2 >= 0 && info.Hard && e.patch2 == patchIndex) ? info.BasePatch2 : info.BasePatch1;
        if (base < 0)
            return -1;
        CKBOOL forward = (e.v1 == patch.v[0] && e.v2 == patch.v[1]);
        int k = i;
        if (!forward)
            k = steps - i;
        if (k <= 0)
            return (int)m_CornerVertexMap[cornerBase + 0];
        if (k >= steps)
            return (int)m_CornerVertexMap[cornerBase + 1];
        return base + (k - 1);
    }
    if (i == steps) {
        int eidx = patch.edge[1];
        if (eidx < 0 || eidx >= m_PatchEdges.Size())
            return -1;
        const CKPatchEdge &e = m_PatchEdges[eidx];
        if (eidx >= edgeTess.Size())
            return -1;
        const EdgeTessInfo &info = edgeTess[eidx];
        int base = (e.patch2 >= 0 && info.Hard && e.patch2 == patchIndex) ? info.BasePatch2 : info.BasePatch1;
        if (base < 0)
            return -1;
        CKBOOL forward = (e.v1 == patch.v[1] && e.v2 == patch.v[2]);
        int k = j;
        if (!forward)
            k = steps - j;
        if (k <= 0)
            return (int)m_CornerVertexMap[cornerBase + 1];
        if (k >= steps)
            return (int)m_CornerVertexMap[cornerBase + 2];
        return base + (k - 1);
    }
    if (j == steps) {
        int eidx = patch.edge[2];
        if (eidx < 0 || eidx >= m_PatchEdges.Size())
            return -1;
        const CKPatchEdge &e = m_PatchEdges[eidx];
        if (eidx >= edgeTess.Size())
            return -1;
        const EdgeTessInfo &info = edgeTess[eidx];
        int base = (e.patch2 >= 0 && info.Hard && e.patch2 == patchIndex) ? info.BasePatch2 : info.BasePatch1;
        if (base < 0)
            return -1;
        CKBOOL forward = (e.v1 == patch.v[3] && e.v2 == patch.v[2]);
        int k = i;
        if (!forward)
            k = steps - i;
        if (k <= 0)
            return (int)m_CornerVertexMap[cornerBase + 3];
        if (k >= steps)
            return (int)m_CornerVertexMap[cornerBase + 2];
        return base + (k - 1);
    }
    if (i == 0) {
        int eidx = patch.edge[3];
        if (eidx < 0 || eidx >= m_PatchEdges.Size())
            return -1;
        const CKPatchEdge &e = m_PatchEdges[eidx];
        if (eidx >= edgeTess.Size())
            return -1;
        const EdgeTessInfo &info = edgeTess[eidx];
        int base = (e.patch2 >= 0 && info.Hard && e.patch2 == patchIndex) ? info.BasePatch2 : info.BasePatch1;
        if (base < 0)
            return -1;
        CKBOOL forward = (e.v1 == patch.v[0] && e.v2 == patch.v[3]);
        int k = j;
        if (!forward)
            k = steps - j;
        if (k <= 0)
            return (int)m_CornerVertexMap[cornerBase + 0];
        if (k >= steps)
            return (int)m_CornerVertexMap[cornerBase + 3];
        return base + (k - 1);
    }

    int base = (patchIndex < interiorBase.Size()) ? interiorBase[patchIndex] : -1;
    if (base < 0)
        return -1;
    return base + (j - 1) * (steps - 1) + (i - 1);
}

int RCKPatchMesh::ComputeTriVertexIndex(int patchIndex, int steps, int uSteps, int vSteps, const XArray<int> &interiorBase, const XArray<EdgeTessInfo> &edgeTess) const {
    if (patchIndex < 0 || patchIndex >= m_Patches.Size() || !m_CornerVertexMap)
        return -1;
    const CKPatch &patch = m_Patches[patchIndex];
    int cornerBase = patchIndex * 4;
    int wSteps = steps - uSteps - vSteps;

    if (uSteps == 0 && vSteps == 0)
        return (int)m_CornerVertexMap[cornerBase + 0];
    if (uSteps == steps && vSteps == 0)
        return (int)m_CornerVertexMap[cornerBase + 1];
    if (uSteps == 0 && vSteps == steps)
        return (int)m_CornerVertexMap[cornerBase + 2];

    if (wSteps == 0) {
        int eidx = patch.edge[0];
        if (eidx < 0 || eidx >= m_PatchEdges.Size() || eidx >= edgeTess.Size())
            return -1;
        const CKPatchEdge &e = m_PatchEdges[eidx];
        const EdgeTessInfo &info = edgeTess[eidx];
        int base = (e.patch2 >= 0 && info.Hard && e.patch2 == patchIndex) ? info.BasePatch2 : info.BasePatch1;
        if (base < 0)
            return -1;
        CKBOOL forward = (e.v1 == patch.v[0] && e.v2 == patch.v[1]);
        int k = uSteps;
        if (!forward)
            k = steps - uSteps;
        if (k <= 0)
            return (int)m_CornerVertexMap[cornerBase + 0];
        if (k >= steps)
            return (int)m_CornerVertexMap[cornerBase + 1];
        return base + (k - 1);
    }
    if (uSteps == 0) {
        int eidx = patch.edge[1];
        if (eidx < 0 || eidx >= m_PatchEdges.Size() || eidx >= edgeTess.Size())
            return -1;
        const CKPatchEdge &e = m_PatchEdges[eidx];
        const EdgeTessInfo &info = edgeTess[eidx];
        int base = (e.patch2 >= 0 && info.Hard && e.patch2 == patchIndex) ? info.BasePatch2 : info.BasePatch1;
        if (base < 0)
            return -1;
        CKBOOL forward = (e.v1 == patch.v[1] && e.v2 == patch.v[2]);
        int k = vSteps;
        if (!forward)
            k = steps - vSteps;
        if (k <= 0)
            return (int)m_CornerVertexMap[cornerBase + 1];
        if (k >= steps)
            return (int)m_CornerVertexMap[cornerBase + 2];
        return base + (k - 1);
    }
    if (vSteps == 0) {
        int eidx = patch.edge[2];
        if (eidx < 0 || eidx >= m_PatchEdges.Size() || eidx >= edgeTess.Size())
            return -1;
        const CKPatchEdge &e = m_PatchEdges[eidx];
        const EdgeTessInfo &info = edgeTess[eidx];
        int base = (e.patch2 >= 0 && info.Hard && e.patch2 == patchIndex) ? info.BasePatch2 : info.BasePatch1;
        if (base < 0)
            return -1;
        CKBOOL forward = (e.v1 == patch.v[2] && e.v2 == patch.v[0]);
        int k = wSteps;
        if (!forward)
            k = steps - wSteps;
        if (k <= 0)
            return (int)m_CornerVertexMap[cornerBase + 2];
        if (k >= steps)
            return (int)m_CornerVertexMap[cornerBase + 0];
        return base + (k - 1);
    }

    int base = (patchIndex < interiorBase.Size()) ? interiorBase[patchIndex] : -1;
    if (base < 0 || wSteps <= 0 || uSteps <= 0)
        return -1;
    int offset = TriInteriorOffset(steps, wSteps, uSteps);
    return base + offset;
}

/**
 * @brief Evaluate a triangular Bezier patch at parametric coordinates
 */
void RCKPatchMesh::EvaluateTriPatch(CKPatch *patch, float u, float v, float w, VxVector *result) {
    // Tri patch evaluation matches the original: degree-4 triangular Bezier.
    // Control points:
    // - 3 corners (verts)
    // - 9 elevated edge points (patch->auxs)
    // - 3 interior points (patch->interior[0..2] in vecs)
    if (!patch || !result)
        return;
    if (!patch->auxs)
        return;

    const VxVector &A = m_Verts[patch->v[0]];
    const VxVector &B = m_Verts[patch->v[1]];
    const VxVector &C = m_Verts[patch->v[2]];

    const VxVector &AB1 = patch->auxs[0];
    const VxVector &AB2 = patch->auxs[1];
    const VxVector &AB3 = patch->auxs[2];
    const VxVector &BC1 = patch->auxs[3];
    const VxVector &BC2 = patch->auxs[4];
    const VxVector &BC3 = patch->auxs[5];
    const VxVector &CA1 = patch->auxs[6];
    const VxVector &CA2 = patch->auxs[7];
    const VxVector &CA3 = patch->auxs[8];

    const VxVector &I0 = m_Vecs[patch->interior[0]];
    const VxVector &I1 = m_Vecs[patch->interior[1]];
    const VxVector &I2 = m_Vecs[patch->interior[2]];

    float u2 = u * u;
    float v2 = v * v;
    float w2 = w * w;
    float u3 = u2 * u;
    float v3 = v2 * v;
    float w3 = w2 * w;
    float u4 = u3 * u;
    float v4 = v3 * v;
    float w4 = w3 * w;

    VxVector P(0.0f, 0.0f, 0.0f);
    P += A * u4;
    P += B * v4;
    P += C * w4;

    // Edge terms (multinomial coefficients for degree 4)
    P += AB1 * (4.0f * u3 * v);
    P += AB2 * (6.0f * u2 * v2);
    P += AB3 * (4.0f * u * v3);

    P += BC1 * (4.0f * v3 * w);
    P += BC2 * (6.0f * v2 * w2);
    P += BC3 * (4.0f * v * w3);

    P += CA3 * (4.0f * u3 * w);
    P += CA2 * (6.0f * u2 * w2);
    P += CA1 * (4.0f * u * w3);

    // Interior terms
    P += I0 * (12.0f * u2 * v * w);
    P += I1 * (12.0f * u * v2 * w);
    P += I2 * (12.0f * u * v * w2);

    *result = P;
}

/**
 * @brief Evaluate a quadrilateral Bezier patch at parametric coordinates
 */
void RCKPatchMesh::EvaluateQuadPatch(CKPatch *patch, float u, float v, VxVector *result) {
    if (!patch || !result)
        return;

    auto cubic = [](const VxVector &p0, const VxVector &p1, const VxVector &p2, const VxVector &p3, float t) -> VxVector {
        float it = 1.0f - t;
        float it2 = it * it;
        float t2 = t * t;
        float b0 = it2 * it;
        float b1 = 3.0f * t * it2;
        float b2 = 3.0f * t2 * it;
        float b3 = t2 * t;
        return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
    };

    // Map Virtools quad patch control points to a 4x4 bicubic grid.
    const VxVector &P00 = m_Verts[patch->v[0]];
    const VxVector &P30 = m_Verts[patch->v[1]];
    const VxVector &P33 = m_Verts[patch->v[2]];
    const VxVector &P03 = m_Verts[patch->v[3]];

    const VxVector &P10 = m_Vecs[patch->vec[0]];
    const VxVector &P20 = m_Vecs[patch->vec[1]];
    const VxVector &P31 = m_Vecs[patch->vec[2]];
    const VxVector &P32 = m_Vecs[patch->vec[3]];
    const VxVector &P23 = m_Vecs[patch->vec[4]];
    const VxVector &P13 = m_Vecs[patch->vec[5]];
    const VxVector &P02 = m_Vecs[patch->vec[6]];
    const VxVector &P01 = m_Vecs[patch->vec[7]];

    const VxVector &P11 = m_Vecs[patch->interior[0]];
    const VxVector &P21 = m_Vecs[patch->interior[1]];
    const VxVector &P22 = m_Vecs[patch->interior[2]];
    const VxVector &P12 = m_Vecs[patch->interior[3]];

    VxVector Q0 = cubic(P00, P10, P20, P30, u);
    VxVector Q1 = cubic(P01, P11, P21, P31, u);
    VxVector Q2 = cubic(P02, P12, P22, P32, u);
    VxVector Q3 = cubic(P03, P13, P23, P33, u);
    *result = cubic(Q0, Q1, Q2, Q3, v);
}

/**
 * @brief Clean the render mesh data
 * Based on IDA analysis at 0x1003a2ee
 */
void RCKPatchMesh::CleanRenderMesh() {
    SetVertexCount(0);
    SetFaceCount(0);
    UnOptimize();
    m_PatchChanged = TRUE;
    m_PatchFlags &= 0xFFFFFFFA; // Clear bits 0, 2
}

/**
 * @brief Compute auxiliary points for a patch (Bezier control points)
 * Based on IDA analysis at 0x10031362
 *
 * For triangle patches, computes 9 auxiliary points (3 per edge) that
 * define the Bezier surface. These are used during tessellation to
 * evaluate the curved surface.
 */
void RCKPatchMesh::ComputePatchAux(int index) {
    if (index < 0 || index >= m_Patches.Size())
        return;

    CKPatch *patch = &m_Patches[index];

    // Auxiliary lookup order for triangle patches
    static const int auxOrder[3] = {1, 2, 0};

    // Allocate auxiliary points if not already present (9 points for triangle)
    if (!patch->auxs) {
        patch->auxs = new VxVector[9];
        for (int i = 0; i < 9; ++i) {
            patch->auxs[i] = VxVector(0, 0, 0);
        }
    }

    int auxIndex = 0;
    int vecIdx = 0;

    for (int i = 0; i < 3; ++i) {
        // Get corner vertices for this edge
        VxVector *v0 = &m_Verts[patch->v[i]];
        VxVector *v1 = &m_Verts[patch->v[auxOrder[i]]];

        // Get edge control vectors
        VxVector *vec0 = &m_Vecs[patch->vec[vecIdx]];
        VxVector *vec1 = &m_Vecs[patch->vec[vecIdx + 1]];

        // Compute auxiliary points using cubic Bezier subdivision
        // aux[0] = v0 + 0.75 * (vec0 - v0)
        VxVector diff0 = *vec0 - *v0;
        patch->auxs[auxIndex] = *v0 + diff0 * 0.75f;

        // aux[1] = vec0 + 0.5 * (vec1 - vec0)
        VxVector diff1 = *vec1 - *vec0;
        patch->auxs[auxIndex + 1] = *vec0 + diff1 * 0.5f;

        // aux[2] = vec1 + 0.25 * (v1 - vec1)
        VxVector diff2 = *v1 - *vec1;
        patch->auxs[auxIndex + 2] = *vec1 + diff2 * 0.25f;

        auxIndex += 3;
        vecIdx += 2;
    }
}

/**
 * @brief Compute interior vector points for a patch
 * Based on IDA analysis at 0x100315f0
 *
 * For both triangle and quad patches, computes the interior control
 * vectors that define the surface curvature in the patch interior.
 */
void RCKPatchMesh::ComputePatchInteriors(int index) {
    if (index < 0 || index >= m_Patches.Size())
        return;

    CKPatch *patch = &m_Patches[index];

    if (patch->type == 3) {
        // Triangle patch - compute 3 interior points
        // First compute auxiliary points
        ComputePatchAux(index);

        // interior[0] = vec[5] + (vec[0] - v[0])
        VxVector diff0 = m_Vecs[patch->vec[0]] - m_Verts[patch->v[0]];
        m_Vecs[patch->interior[0]] = m_Vecs[patch->vec[5]] + diff0;

        // interior[1] = vec[1] + (vec[2] - v[1])
        VxVector diff1 = m_Vecs[patch->vec[2]] - m_Verts[patch->v[1]];
        m_Vecs[patch->interior[1]] = m_Vecs[patch->vec[1]] + diff1;

        // interior[2] = vec[3] + (vec[4] - v[2])
        VxVector diff2 = m_Vecs[patch->vec[4]] - m_Verts[patch->v[2]];
        m_Vecs[patch->interior[2]] = m_Vecs[patch->vec[3]] + diff2;
    } else if (patch->type == 4) {
        // Quad patch - compute 4 interior points

        // interior[0] = vec[7] + (vec[0] - v[0])
        VxVector diff0 = m_Vecs[patch->vec[0]] - m_Verts[patch->v[0]];
        m_Vecs[patch->interior[0]] = m_Vecs[patch->vec[7]] + diff0;

        // interior[1] = vec[1] + (vec[2] - v[1])
        VxVector diff1 = m_Vecs[patch->vec[2]] - m_Verts[patch->v[1]];
        m_Vecs[patch->interior[1]] = m_Vecs[patch->vec[1]] + diff1;

        // interior[2] = vec[3] + (vec[4] - v[2])
        VxVector diff2 = m_Vecs[patch->vec[4]] - m_Verts[patch->v[2]];
        m_Vecs[patch->interior[2]] = m_Vecs[patch->vec[3]] + diff2;

        // interior[3] = vec[5] + (vec[6] - v[3])
        VxVector diff3 = m_Vecs[patch->vec[6]] - m_Verts[patch->v[3]];
        m_Vecs[patch->interior[3]] = m_Vecs[patch->vec[5]] + diff3;
    }
}

CKDWORD RCKPatchMesh::GetPatchFlags() {
    return m_PatchFlags;
}

void RCKPatchMesh::SetPatchFlags(CKDWORD Flags) {
    m_PatchFlags = Flags;
}

/**
 * @brief Set vertex and vector counts, allocating combined array
 * Based on IDA analysis at 0x10030921
 */
void RCKPatchMesh::SetVertVecCount(int VertCount, int VecCount) {
    m_VertCount = VertCount;
    m_VecCount = VecCount;

    if (m_Verts) {
        delete[] m_Verts;
        m_Verts = nullptr;
    }

    int totalCount = VertCount + VecCount;
    if (totalCount > 0) {
        VxVector *verts = new VxVector[totalCount];
        if (verts) {
            for (int i = 0; i < totalCount; ++i) {
                verts[i] = VxVector(0, 0, 0);
            }
            m_Verts = verts;
        }
    }

    if (m_Verts && VecCount > 0) {
        m_Vecs = &m_Verts[VertCount];
    } else {
        m_Vecs = nullptr;
    }

    m_PatchChanged = TRUE;
}

int RCKPatchMesh::GetVertCount() {
    return m_VertCount;
}

/**
 * @brief Set a vertex position
 * Based on IDA analysis at 0x10030a74
 */
void RCKPatchMesh::SetVert(int index, VxVector *cp) {
    if (index >= 0 && index < (int) m_VertCount && cp) {
        m_Verts[index] = *cp;
        m_PatchFlags &= ~CK_PATCHMESH_UPTODATE; // Clear UPTODATE flag
    }
}

/**
 * @brief Get a vertex position
 * Based on IDA analysis at 0x10030ad0
 */
void RCKPatchMesh::GetVert(int index, VxVector *cp) {
    if (index >= 0 && index < (int) m_VertCount && cp) {
        *cp = m_Verts[index];
    }
}

VxVector *RCKPatchMesh::GetVerts() {
    return m_Verts;
}

int RCKPatchMesh::GetVecCount() {
    return m_VecCount;
}

/**
 * @brief Set a vector position
 * Based on IDA analysis at 0x10030b17
 */
void RCKPatchMesh::SetVec(int index, VxVector *cp) {
    if (index >= 0 && index < (int) m_VecCount && cp) {
        m_Vecs[index] = *cp;
        m_PatchFlags &= ~CK_PATCHMESH_UPTODATE; // Clear UPTODATE flag
    }
}

/**
 * @brief Get a vector position
 * Based on IDA analysis at 0x10030b73
 */
void RCKPatchMesh::GetVec(int index, VxVector *cp) {
    if (index >= 0 && index < (int) m_VecCount && cp) {
        *cp = m_Vecs[index];
    }
}

VxVector *RCKPatchMesh::GetVecs() {
    return m_Vecs;
}

/**
 * @brief Set number of edges
 * Based on IDA analysis at 0x10030be2
 */
void RCKPatchMesh::SetEdgeCount(int count) {
    m_PatchEdges.Resize(count);
    m_PatchChanged = TRUE;
}

int RCKPatchMesh::GetEdgeCount() {
    return m_PatchEdges.Size();
}

/**
 * @brief Set edge data
 * Based on IDA analysis at 0x10030c27
 */
void RCKPatchMesh::SetEdge(int index, CKPatchEdge *edge) {
    if (index >= 0 && index < m_PatchEdges.Size() && edge) {
        m_PatchEdges[index] = *edge;
    }
}

/**
 * @brief Get edge data
 * Based on IDA analysis at 0x10030c76
 */
void RCKPatchMesh::GetEdge(int index, CKPatchEdge *edge) {
    if (index >= 0 && index < m_PatchEdges.Size() && edge) {
        *edge = m_PatchEdges[index];
    }
}

CKPatchEdge *RCKPatchMesh::GetEdges() {
    return m_PatchEdges.Begin();
}

/**
 * @brief Set number of patches
 * Based on IDA analysis at 0x10030cdc
 */
void RCKPatchMesh::SetPatchCount(int count) {
    m_Patches.Resize(count);
    m_PatchChanged = TRUE;
}

int RCKPatchMesh::GetPatchCount() {
    return m_Patches.Size();
}

/**
 * @brief Set patch data
 * Based on IDA analysis at 0x10030d21
 * Copies 56 bytes (size of CKPatch without auxs) and clears auxs
 */
void RCKPatchMesh::SetPatch(int index, CKPatch *p) {
    if (index >= 0 && index < m_Patches.Size() && p) {
        // Copy 56 bytes (all fields except auxs)
        memcpy(&m_Patches[index], p, 56);
        m_Patches[index].auxs = nullptr; // Clear runtime auxs pointer
        m_PatchChanged = TRUE;
    }
}

/**
 * @brief Get patch data
 * Based on IDA analysis at 0x10030d91
 */
void RCKPatchMesh::GetPatch(int index, CKPatch *p) {
    if (index >= 0 && index < m_Patches.Size() && p) {
        memcpy(p, &m_Patches[index], 56);
        p->auxs = nullptr; // Clear runtime auxs pointer
    }
}

/**
 * @brief Get patch smoothing group
 * Based on IDA analysis at 0x10030dfe
 */
CKDWORD RCKPatchMesh::GetPatchSM(int index) {
    if (index >= 0 && index < m_Patches.Size())
        return m_Patches[index].SmoothingGroup;
    return 0;
}

/**
 * @brief Set patch smoothing group
 * Based on IDA analysis at 0x10030e3d
 */
void RCKPatchMesh::SetPatchSM(int index, CKDWORD smoothing) {
    if (index >= 0 && index < m_Patches.Size()) {
        m_Patches[index].SmoothingGroup = smoothing;
        m_PatchChanged = TRUE;
    }
}

/**
 * @brief Get patch material
 * Based on IDA analysis at 0x10030e8a
 */
CKMaterial *RCKPatchMesh::GetPatchMaterial(int index) {
    if (index >= 0 && index < m_Patches.Size())
        return (CKMaterial *) m_Context->GetObject(m_Patches[index].Material);
    return nullptr;
}

/**
 * @brief Set patch material
 * Based on IDA analysis at 0x10030ed5
 */
void RCKPatchMesh::SetPatchMaterial(int index, CKMaterial *mat) {
    if (index >= 0 && index < m_Patches.Size()) {
        CK_ID newId = mat ? mat->GetID() : 0;
        if (newId != m_Patches[index].Material) {
            m_Patches[index].Material = newId;
            m_PatchFlags &= ~0x5;
        }
    }
}

CKPatch *RCKPatchMesh::GetPatches() {
    return m_Patches.Begin();
}

//=============================================================================
// Texture Patch Methods
// These use Channel+1 to index into m_TexturePatches
// Channel -1 = index 0 (default), Channel 0 = index 1, etc.
//=============================================================================

/**
 * @brief Set texture patch count for a channel
 * Based on IDA analysis at 0x1003102e
 */
void RCKPatchMesh::SetTVPatchCount(int count, int Channel) {
    int channelIndex = Channel + 1;
    if (channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        if (count == 0) {
            m_TexturePatches[channelIndex].Patches.Clear();
        } else {
            m_TexturePatches[channelIndex].Patches.Resize(count);
        }
    }
}

/**
 * @brief Get texture patch count for a channel
 * Based on IDA analysis at 0x1003107f
 */
int RCKPatchMesh::GetTVPatchCount(int Channel) {
    int channelIndex = Channel + 1;
    if (channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        return m_TexturePatches[channelIndex].Patches.Size();
    }
    return 0;
}

/**
 * @brief Set texture patch data
 * Based on IDA analysis at 0x100310a8
 */
void RCKPatchMesh::SetTVPatch(int index, CKTVPatch *tvpatch, int Channel) {
    int channelIndex = Channel + 1;
    if (index >= 0 && channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        if (index < m_TexturePatches[channelIndex].Patches.Size() && tvpatch) {
            m_TexturePatches[channelIndex].Patches[index] = *tvpatch;
        }
    }
}

/**
 * @brief Get texture patch data
 * Based on IDA analysis at 0x1003110f
 */
void RCKPatchMesh::GetTVPatch(int index, CKTVPatch *tvpatch, int Channel) {
    int channelIndex = Channel + 1;
    if (index >= 0 && channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        if (index < m_TexturePatches[channelIndex].Patches.Size() && tvpatch) {
            *tvpatch = m_TexturePatches[channelIndex].Patches[index];
        }
    }
}

/**
 * @brief Get pointer to texture patches array
 * Based on IDA analysis at 0x10031172
 */
CKTVPatch *RCKPatchMesh::GetTVPatches(int Channel) {
    int channelIndex = Channel + 1;
    if (channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        return m_TexturePatches[channelIndex].Patches.Begin();
    }
    return nullptr;
}

/**
 * @brief Set texture coordinate count for a channel
 * Based on IDA analysis at 0x1003119b
 */
void RCKPatchMesh::SetTVCount(int count, int Channel) {
    int channelIndex = Channel + 1;
    if (channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        if (count == 0) {
            m_TexturePatches[channelIndex].UVs.Clear();
        } else {
            m_TexturePatches[channelIndex].UVs.Resize(count);
        }
    }
}

/**
 * @brief Get texture coordinate count for a channel
 * Based on IDA analysis at 0x100311f2
 */
int RCKPatchMesh::GetTVCount(int Channel) {
    int channelIndex = Channel + 1;
    if (channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        return m_TexturePatches[channelIndex].UVs.Size();
    }
    return 0;
}

/**
 * @brief Set texture coordinate values
 * Based on IDA analysis at 0x1003121e
 */
void RCKPatchMesh::SetTV(int index, float u, float v, int Channel) {
    int channelIndex = Channel + 1;
    if (index >= 0 && channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        if (index < m_TexturePatches[channelIndex].UVs.Size()) {
            m_TexturePatches[channelIndex].UVs[index].u = u;
            m_TexturePatches[channelIndex].UVs[index].v = v;
        }
    }
}

/**
 * @brief Get texture coordinate values
 * Based on IDA analysis at 0x100312a8
 */
void RCKPatchMesh::GetTV(int index, float *u, float *v, int Channel) {
    int channelIndex = Channel + 1;
    if (index >= 0 && channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        if (index < m_TexturePatches[channelIndex].UVs.Size()) {
            if (u) *u = m_TexturePatches[channelIndex].UVs[index].u;
            if (v) *v = m_TexturePatches[channelIndex].UVs[index].v;
            return;
        }
    }
    if (u) *u = 0.0f;
    if (v) *v = 0.0f;
}

/**
 * @brief Get pointer to texture coordinates array
 * Based on IDA analysis at 0x10031336
 */
VxUV *RCKPatchMesh::GetTVs(int Channel) {
    int channelIndex = Channel + 1;
    if (channelIndex >= 0 && channelIndex < m_TexturePatches.Size()) {
        return m_TexturePatches[channelIndex].UVs.Begin();
    }
    return nullptr;
}
