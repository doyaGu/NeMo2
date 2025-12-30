#include "RCKMesh.h"

#include "CKMemoryPool.h"
#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKContext.h"
#include "CKScene.h"
#include "CKMaterial.h"
#include "CK3dEntity.h"
#include "CKSkin.h"
#include "CKObjectAnimation.h"
#include "CKKeyframeData.h"
#include "CKRasterizer.h"
#include "RCKRenderManager.h"
#include "RCKRenderContext.h"
#include "RCK3dEntity.h"
#include "RCKMaterial.h"
#include "MeshStriper.h"
#include "NvStripifier.h"

// External global for transparency update flag
extern CKBOOL g_UpdateTransparency;

// External VxMath normal building functions
extern void (*g_BuildNormalsFunc)(CKFace *, CKWORD *, int, VxVertex *, int);
extern void (*g_BuildFaceNormalsFunc)(CKFace *, CKWORD *, int, VxVertex *, int);

CKVBuffer *RCKMesh::GetVBuffer(CKMaterialGroup *group) const {
    if (!group || !group->m_RemapData)
        return nullptr;
    return reinterpret_cast<CKVBuffer *>(static_cast<uintptr_t>(group->m_RemapData));
}

void RCKMesh::DeleteVBuffer(CKMaterialGroup *group) {
    if (!group || !group->m_RemapData)
        return;
    delete reinterpret_cast<CKVBuffer *>(static_cast<uintptr_t>(group->m_RemapData));
    group->m_RemapData = 0;
}

void RCKMesh::ResetMaterialGroup(CKMaterialGroup *group, int a2) {
    if (!group)
        return;

    group->m_Primitives.Resize(1);
    if (group->m_Primitives.Size() > 0) {
        // IDA: Resize to 3*a2 then back to 0 (reserve semantics)
        group->m_Primitives[0].m_Indices.Resize(3 * a2);
        group->m_Primitives[0].m_Indices.Resize(0);
        group->m_Primitives[0].m_Type = VX_TRIANGLELIST;
        group->m_Primitives[0].m_IndexBufferOffset = -1;
    }

    group->m_FaceIndices.Resize(0);
    group->m_HasValidPrimitives = 0;
    group->m_MinVertexIndex = 0x10000;
    group->m_MaxVertexIndex = 0;
    group->m_BaseVertex = 0;

    DeleteVBuffer(group);
    group->m_VertexCount = 0;
}

void RCKMesh::UpdateHasValidPrimitives(CKMaterialGroup *group) {
    if (!group)
        return;
    group->m_HasValidPrimitives = 0;
    for (int i = 0; i < group->m_Primitives.Size(); ++i) {
        if (group->m_Primitives[i].m_Indices.Size() > 0) {
            group->m_HasValidPrimitives = 1;
            return;
        }
    }
}

CKVBuffer::CKVBuffer(int vertexCount) : m_Vertices(0), m_Colors(0), m_UVs(0), m_VertexRemap(0) {
    if (vertexCount)
        Resize(vertexCount);
}

CKVBuffer::~CKVBuffer() = default;

void CKVBuffer::Resize(int vertexCount) {
    m_Vertices.Resize(vertexCount);
    m_Colors.Resize(vertexCount);
    m_VertexRemap.Resize(vertexCount);
}

void CKVBuffer::Update(RCKMesh *mesh, int force) {
    if (!mesh)
        return;

    const int vertexCount = m_VertexRemap.Size();
    if (vertexCount <= 0)
        return;

    // Ensure storage matches remap size
    if (m_Vertices.Size() != vertexCount || m_Colors.Size() != vertexCount)
        Resize(vertexCount);

    const CKDWORD flags = mesh->GetFlags();

    // IDA masks: 0x28000 (pos+normal), 0x10000 (color), 0x4000 (uv + channels)
    if (force || (flags & 0x3C000) != 0) {
        if (force || (flags & 0x28000) != 0) {
            CKDWORD srcStride = 0;
            void *srcPtr = mesh->GetPositionsPtr(&srcStride);
            if (srcPtr && srcStride) {
                VxStridedData dst(m_Vertices.Begin(), 32);
                VxStridedData src(srcPtr, srcStride);
                VxIndexedCopy(dst, src, 0x18u, m_VertexRemap.Begin(), vertexCount);
            }
        }

        if (force || (flags & 0x10000) != 0) {
            CKDWORD srcStride = 0;
            void *srcPtr = mesh->GetColorsPtr(&srcStride);
            if (srcPtr && srcStride) {
                VxStridedData dst(m_Colors.Begin(), 8);
                VxStridedData src(srcPtr, srcStride);
                VxIndexedCopy(dst, src, 8u, m_VertexRemap.Begin(), vertexCount);
            }
        }

        if (force || (flags & 0x4000) != 0) {
            // Copy main UVs into m_Vertices UV field
            CKDWORD srcStride = 0;
            void *srcPtr = mesh->GetTextureCoordinatesPtr(&srcStride, -1);
            if (srcPtr && srcStride) {
                VxStridedData dst(&m_Vertices[0].m_UV, 32);
                VxStridedData src(srcPtr, srcStride);
                VxIndexedCopy(dst, src, 8u, m_VertexRemap.Begin(), vertexCount);
            }

            const int channelCount = mesh->GetChannelCount();
            m_UVs.Resize(channelCount);

            for (int c = 0; c < channelCount; ++c) {
                if (!mesh->GetChannelMaterial(c)) {
                    m_UVs[c].Resize(0);
                    continue;
                }

                CKDWORD channelStride = 0;
                void *channelPtr = mesh->GetTextureCoordinatesPtr(&channelStride, c);
                if (!channelPtr || !channelStride) {
                    m_UVs[c].Resize(0);
                    continue;
                }

                m_UVs[c].Resize(vertexCount);
                VxStridedData dstUv(m_UVs[c].Begin(), 8);
                VxStridedData srcUv(channelPtr, channelStride);
                VxIndexedCopy(dstUv, srcUv, 8u, m_VertexRemap.Begin(), vertexCount);
            }
        }
    }
}

/**
 * @brief Progressive mesh pre-render callback for LOD processing.
 *
 * Called before rendering to update progressive mesh level based on distance/criteria.
 */
static int SnapPMVertexCount(int target, int maxVertices, int morphStep) {
    int clamped = target;
    if (clamped < 3)
        clamped = 3;
    if (clamped > maxVertices)
        clamped = maxVertices;

    if (morphStep <= 0) {
        int level = maxVertices;
        while ((level / 2) >= 3 && clamped <= (level / 2)) {
            level /= 2;
        }
        return level;
    }

    int step = morphStep;
    int offset = (maxVertices - clamped + step - 1) / step;
    int snapped = maxVertices - offset * step;
    if (snapped < 3)
        snapped = 3;
    return snapped;
}

static void FilterTriangleList(const CKWORD *indices, int indexCount, int vertexLimit, XArray<CKWORD> &out) {
    out.Clear();
    if (!indices || indexCount <= 0 || vertexLimit <= 0)
        return;

    out.Reserve(indexCount);
    for (int i = 0; i + 2 < indexCount; i += 3) {
        CKWORD a = indices[i];
        CKWORD b = indices[i + 1];
        CKWORD c = indices[i + 2];
        if (a < vertexLimit && b < vertexLimit && c < vertexLimit) {
            out.PushBack(a);
            out.PushBack(b);
            out.PushBack(c);
        }
    }
}

static void FilterLineList(const CKWORD *indices, int indexCount, int vertexLimit, XArray<CKWORD> &out) {
    out.Clear();
    if (!indices || indexCount <= 0 || vertexLimit <= 0)
        return;

    out.Reserve(indexCount);
    for (int i = 0; i + 1 < indexCount; i += 2) {
        CKWORD a = indices[i];
        CKWORD b = indices[i + 1];
        if (a < vertexLimit && b < vertexLimit) {
            out.PushBack(a);
            out.PushBack(b);
        }
    }
}

static int ClampPMVertexCount(RCKMesh *mesh, int target) {
    if (!mesh)
        return 0;
    int maxVertices = mesh->GetVertexCount();
    int morphStep = mesh->GetPMGeoMorphStep();
    return SnapPMVertexCount(target, maxVertices, morphStep);
}

void ProgressiveMeshPreRenderCallback(CKRenderContext *ctx, CK3dEntity *entity, CKMesh *meshObj, void *data) {
    // Match IDA at 0x100238dd
    // The original callback simply calls BuildRenderMesh
    RCKMesh *mesh = (RCKMesh *) (meshObj ? meshObj : data);
    if (mesh)
        mesh->BuildRenderMesh();
}

// Constructor
RCKMesh::RCKMesh(CKContext *Context, CKSTRING name) : CKMesh(Context, name) {
    // Match IDA at 0x1001bab0

    m_ObjectFlags = CK_OBJECT_VISIBLE;

    m_VertexWeights = nullptr;
    m_Flags = VXMESH_VISIBLE | VXMESH_RENDERCHANNELS;
    m_Radius = 0.0f;
    m_ProgressiveMesh = nullptr;

    // Reserve space for material groups and create initial group
    m_MaterialGroups.Reserve(2);
    CreateNewMaterialGroup(nullptr);

    // Create vertex and index buffers via RenderManager
    RCKRenderManager *renderManager = (RCKRenderManager *) Context->GetRenderManager();
    if (renderManager) {
        m_VertexBuffer = renderManager->CreateObjectIndex(CKRST_OBJ_VERTEXBUFFER);
        m_IndexBuffer = renderManager->CreateObjectIndex(CKRST_OBJ_INDEXBUFFER);
    } else {
        m_VertexBuffer = 0;
        m_IndexBuffer = 0;
    }

    // Clear local box
    memset(&m_LocalBox, 0, sizeof(m_LocalBox));

    // Initialize callback containers and other fields
    m_RenderCallbacks = nullptr;
    m_SubMeshCallbacks = nullptr;
    m_FaceChannelMask = 0;
    m_Valid = 0;
    m_VertexBufferReady = 0;
}

// Destructor
RCKMesh::~RCKMesh() {
    // Match IDA at 0x1001bc99

    // Clean up material channels (sub_1001BE0B in IDA)
    for (int i = 0; i < m_MaterialChannels.Size(); i++) {
        VxMaterialChannel &channel = m_MaterialChannels[i];
        if (channel.m_UVs != nullptr) {
            delete[] channel.m_UVs;
            channel.m_UVs = nullptr;
        }
    }

    // Delete render groups
    DeleteRenderGroup();

    // Remove all callbacks
    RemoveAllCallbacks();

    // Clean up vertex weights array
    if (m_VertexWeights) {
        delete m_VertexWeights;
        m_VertexWeights = nullptr;
    }

    // Release vertex buffer
    if (m_VertexBuffer) {
        RCKRenderManager *renderManager = (RCKRenderManager *) m_Context->GetRenderManager();
        if (renderManager) {
            renderManager->ReleaseObjectIndex(m_VertexBuffer, CKRST_OBJ_VERTEXBUFFER);
        }
        m_VertexBuffer = 0;
    }

    // Release index buffer
    if (m_IndexBuffer) {
        RCKRenderManager *renderManager = (RCKRenderManager *) m_Context->GetRenderManager();
        if (renderManager) {
            renderManager->ReleaseObjectIndex(m_IndexBuffer, CKRST_OBJ_INDEXBUFFER);
        }
        m_IndexBuffer = 0;
    }
}

// Get class ID
// Match IDA at 0x10028ac8
CK_CLASSID RCKMesh::GetClassID() {
    return m_ClassID;
}

// Show/Hide mesh
void RCKMesh::Show(CK_OBJECT_SHOWOPTION show) {
    // Match IDA at 0x1001cdc3
    CKObject::Show(show);

    // Update mesh visibility flag
    if ((show & CKSHOW) != 0) {
        m_Flags |= VXMESH_VISIBLE; // Set visible flag
    } else {
        m_Flags &= ~VXMESH_VISIBLE; // Clear visible flag
    }
}

// Transparency methods
CKBOOL RCKMesh::IsTransparent() {
    // Match IDA at 0x1001cf38
    // Check FORCETRANSPARENCY flag
    if ((m_Flags & VXMESH_FORCETRANSPARENCY) != 0) {
        return TRUE;
    }

    // Check if transparency is up to date and not forcing update
    if (!(m_Flags & VXMESH_TRANSPARENCYUPTODATE) || g_UpdateTransparency) {
        // Set TRANSPARENCYUPTODATE flag
        m_Flags |= VXMESH_TRANSPARENCYUPTODATE;

        // Create render groups if not optimized
        if ((m_Flags & VXMESH_OPTIMIZED) == 0) {
            CreateRenderGroups();
        }

        // Clear transparency flag
        m_Flags &= ~VXMESH_HASTRANSPARENCY;

        // Check each material group for transparency (starting from index 1)
        for (int i = 1; i < m_MaterialGroups.Size(); ++i) {
            CKMaterialGroup *group = m_MaterialGroups[i];
            if (group && group->m_Material && group->m_Material->IsAlphaTransparent()) {
                m_Flags |= VXMESH_HASTRANSPARENCY;
                return TRUE;
            }
        }
    }

    return (m_Flags & VXMESH_HASTRANSPARENCY) != 0;
}

void RCKMesh::SetTransparent(CKBOOL Transparency) {
    // Match IDA at 0x1001cf05
    if (Transparency) {
        m_Flags |= VXMESH_FORCETRANSPARENCY;
    } else {
        m_Flags &= ~VXMESH_FORCETRANSPARENCY;
    }
}

// Wrap mode methods
void RCKMesh::SetWrapMode(VXTEXTURE_WRAPMODE Mode) {
    // Match IDA at 0x1001ce5b
    // Bit 0 of Mode = WRAPU (0x100), Bit 1 = WRAPV (0x200)
    if ((Mode & VXTEXTUREWRAP_U) != 0) {
        m_Flags |= VXMESH_WRAPU;
    } else {
        m_Flags &= ~VXMESH_WRAPU;
    }

    if ((Mode & VXTEXTUREWRAP_V) != 0) {
        m_Flags |= VXMESH_WRAPV;
    } else {
        m_Flags &= ~VXMESH_WRAPV;
    }
}

VXTEXTURE_WRAPMODE RCKMesh::GetWrapMode() {
    // Match IDA at 0x1001cebc
    int result = 0;
    if ((m_Flags & VXMESH_WRAPU) != 0) {
        result |= VXTEXTUREWRAP_U;
    }
    if ((m_Flags & VXMESH_WRAPV) != 0) {
        result |= VXTEXTUREWRAP_V;
    }
    return (VXTEXTURE_WRAPMODE) result;
}

// Lighting mode methods
void RCKMesh::SetLitMode(VXMESH_LITMODE Mode) {
    // Match IDA at 0x1001ce04
    if (Mode) {
        m_Flags &= ~VXMESH_PRELITMODE;
    } else {
        m_Flags |= VXMESH_PRELITMODE;
    }
}

VXMESH_LITMODE RCKMesh::GetLitMode() {
    // Match IDA at 0x1001ce37
    // Returns 1 (LIT) if PRELITMODE bit is clear, 0 (PRELIT) if set
    return (m_Flags & VXMESH_PRELITMODE) ? VX_PRELITMESH : VX_LITMESH;
}

// Flags methods
CKDWORD RCKMesh::GetFlags() {
    return m_Flags;
}

void RCKMesh::SetFlags(CKDWORD Flags) {
    // Match IDA at 0x1001cd80
    m_Flags = Flags;
    // Update object transparency flag based on mesh transparency flag
    if ((m_Flags & VXMESH_VISIBLE) != 0) {
        m_ObjectFlags |= CK_OBJECT_VISIBLE;
    } else {
        m_ObjectFlags &= ~CK_OBJECT_VISIBLE;
    }
}

// Vertex data access methods
void *RCKMesh::GetPositionsPtr(CKDWORD *Stride) {
    *Stride = sizeof(VxVertex);
    return m_Vertices.Begin();
}

void *RCKMesh::GetNormalsPtr(CKDWORD *Stride) {
    // Match IDA at 0x1001c3b4
    *Stride = sizeof(VxVertex);
    if (m_Vertices.Size() > 0) {
        // Return pointer to m_Normal field (offset 12 in VxVertex)
        return &m_Vertices[0].m_Normal;
    }
    return nullptr;
}

void *RCKMesh::GetColorsPtr(CKDWORD *Stride) {
    // Match IDA at 0x1001c130
    if (m_VertexColors.Size() == 0) {
        return nullptr;
    }
    *Stride = sizeof(VxColors);
    return m_VertexColors.Begin();
}

void *RCKMesh::GetSpecularColorsPtr(CKDWORD *Stride) {
    // Match IDA at 0x1001c1f7
    *Stride = sizeof(VxColors);
    if (m_VertexColors.Size() > 0) {
        // Return pointer to specular color (offset +4 in VxColors)
        return (CKBYTE *) m_VertexColors.Begin() + 4;
    }
    return nullptr;
}

void *RCKMesh::GetTextureCoordinatesPtr(CKDWORD *Stride, int channel) {
    // Match IDA at 0x1001c59a
    if (!Stride) {
        return nullptr;
    }

    // channel == -1 means use vertex UVs
    if (channel == -1) {
        *Stride = sizeof(VxVertex);
        return &m_Vertices[0].m_UV;
    }

    if (channel >= m_MaterialChannels.Size()) {
        return nullptr;
    }

    VxMaterialChannel &matChannel = m_MaterialChannels[channel];

    // If flag 0x800000 is set, use vertex UVs
    if ((matChannel.m_Flags & VXCHANNEL_SAMEUV) != 0) {
        *Stride = sizeof(VxVertex);
        return &m_Vertices[0].m_UV;
    }

    // Use channel-specific UV array
    *Stride = sizeof(VxUV);
    return matChannel.m_UVs;
}

// Vertex manipulation notifications
void RCKMesh::VertexMove() {
    // Match IDA at 0x1001e115
    m_Flags &= ~VXMESH_BOUNDINGUPTODATE;
    m_Flags |= VXMESH_POS_CHANGED;
    m_Valid = FALSE;
}

void RCKMesh::UVChanged() {
    // Match IDA at 0x1001e14e
    m_Flags |= VXMESH_UV_CHANGED;
    m_Valid = FALSE;
}

void RCKMesh::NormalChanged() {
    // Match IDA at 0x1001e175
    m_Flags |= VXMESH_NORMAL_CHANGED;
    m_Flags &= ~VXMESH_NORMAL_CHANGED;
    m_Valid = FALSE;
}

void RCKMesh::ColorChanged() {
    // Match IDA at 0x1001e1ae
    m_Flags |= VXMESH_COLOR_CHANGED;
    m_Valid = FALSE;
}

// Normal building methods
void RCKMesh::BuildNormals() {
    // Match IDA at 0x1001e39e
    if (m_Faces.Size() == 0) return;
    if (m_Vertices.Size() == 0) return;

    m_Flags |= VXMESH_NORMAL_CHANGED | VXMESH_GENNORMALS;

    // Call the VxMath normal building function
    g_BuildNormalsFunc(m_Faces.Begin(),
                       m_FaceVertexIndices.Begin(),
                       m_Faces.Size(),
                       m_Vertices.Begin(),
                       m_Vertices.Size());
}

void RCKMesh::BuildFaceNormals() {
    // Match IDA at 0x1001e42e
    if (m_Faces.Size() == 0) return;
    if (m_Vertices.Size() == 0) return;

    // Call the VxMath face normal building function
    g_BuildFaceNormalsFunc(m_Faces.Begin(),
                           m_FaceVertexIndices.Begin(),
                           m_Faces.Size(),
                           m_Vertices.Begin(),
                           m_Vertices.Size());
}

// Vertex count management
CKBOOL RCKMesh::SetVertexCount(int Count) {
    // Match IDA at 0x1001be5f
    if (Count < 0) {
        Count = 0;
    }

    // Align to 4-byte boundary with 3 extra
    int alignedCount = (Count + 3) & ~0x03;

    int currentCount = m_Vertices.Size();
    if (currentCount == Count) {
        return TRUE;
    }

    // Resize vertex arrays - first to aligned, then to actual count
    m_Vertices.Resize(alignedCount);
    m_VertexColors.Resize(alignedCount);
    m_Vertices.Resize(Count);
    m_VertexColors.Resize(Count);

    // Initialize new vertices
    if (currentCount < Count) {
        // Initialize new vertices to zero
        VxVertex *newVertices = &m_Vertices[currentCount];
        memset(newVertices, 0, sizeof(VxVertex) * (Count - currentCount));

        // Initialize new colors with VxFillStructure pattern: {0xFFFFFFFF, 0}
        CKDWORD defaultColor[2] = {0xFFFFFFFF, 0};
        VxFillStructure(Count - currentCount, &m_VertexColors[currentCount], 8, 8, defaultColor);
    }

    // Update material channels UV data
    for (VxMaterialChannel *channel = m_MaterialChannels.Begin();
         channel != m_MaterialChannels.End();
         channel++) {
        // Delete old UV array
        if (channel->m_UVs) {
            delete[] channel->m_UVs;
        }
        channel->m_UVs = nullptr;

        // Allocate new UV array if not using vertex UV
        if (!(channel->m_Flags & VXCHANNEL_SAMEUV)) {
            channel->m_UVs = new Vx2DVector[Count];
            // Initialize UV array
            for (int i = 0; i < Count; ++i) {
                channel->m_UVs[i].x = 0.0f;
                channel->m_UVs[i].y = 0.0f;
            }
        }
    }

    // Update vertex weights if present
    if (m_VertexWeights) {
        m_VertexWeights->Resize(Count);
        if (currentCount < Count) {
            // Initialize new weights to zero
            memset(&(*m_VertexWeights)[currentCount], 0, sizeof(float) * (Count - currentCount));
        }
    }

    m_Flags &= ~VXMESH_BOUNDINGUPTODATE;
    return TRUE;
}

int RCKMesh::GetVertexCount() {
    return m_Vertices.Size();
}

// Vertex position methods
void RCKMesh::SetVertexPosition(int Index, VxVector *Vector) {
    if (Index >= 0 && Index < m_Vertices.Size()) {
        m_Vertices[Index].m_Position = *Vector;
        VertexMove();
    }
}

void RCKMesh::GetVertexPosition(int Index, VxVector *Vector) {
    if (Index >= 0 && Index < m_Vertices.Size()) {
        *Vector = m_Vertices[Index].m_Position;
    }
}

// Vertex normal methods
void RCKMesh::SetVertexNormal(int Index, VxVector *Vector) {
    if (Index >= 0 && Index < m_Vertices.Size()) {
        m_Vertices[Index].m_Normal = *Vector;
        NormalChanged();
    }
}

void RCKMesh::GetVertexNormal(int Index, VxVector *Vector) {
    if (Index >= 0 && Index < m_Vertices.Size()) {
        *Vector = m_Vertices[Index].m_Normal;
    }
}

// Vertex color methods
void RCKMesh::SetVertexColor(int Index, CKDWORD Color) {
    if (Index >= 0 && Index < m_VertexColors.Size()) {
        m_VertexColors[Index].Color = Color;
        ColorChanged();
    }
}

// Match IDA at 0x1001c0f2
CKDWORD RCKMesh::GetVertexColor(int Index) {
    if (Index >= 0 && Index < m_VertexColors.Size()) {
        return m_VertexColors[Index].Color;
    }
    return 0;
}

// Vertex texture coordinate methods
// Match IDA at 0x1001c3f3
void RCKMesh::SetVertexTextureCoordinates(int Index, float u, float v, int channel) {
    if (Index >= 0 && Index < m_Vertices.Size()) {
        if (channel >= 0) {
            if (channel < m_MaterialChannels.Size()) {
                if (m_MaterialChannels[channel].m_UVs) {
                    m_MaterialChannels[channel].m_UVs[Index].x = u;
                    m_MaterialChannels[channel].m_UVs[Index].y = v;
                }
            }
        } else {
            // channel < 0 means set the vertex's own UV
            m_Vertices[Index].m_UV.x = u;
            m_Vertices[Index].m_UV.y = v;
        }
        UVChanged();
    }
}

void RCKMesh::GetVertexTextureCoordinates(int Index, float *u, float *v, int channel) {
    // Match IDA at 0x1001c4c7
    if (Index < 0 || Index >= m_Vertices.Size())
        return;

    if (channel < 0) {
        // channel < 0 means use default UV from m_Vertices
        *u = m_Vertices[Index].m_UV.x;
        *v = m_Vertices[Index].m_UV.y;
    } else {
        // IDA shows unsigned comparison for channel bounds check
        if (channel >= m_MaterialChannels.Size())
            return;
        if (m_MaterialChannels[channel].m_UVs != nullptr) {
            *u = m_MaterialChannels[channel].m_UVs[Index].x;
            *v = m_MaterialChannels[channel].m_UVs[Index].y;
        }
    }
}

// Face count management
CKBOOL RCKMesh::SetFaceCount(int Count) {
    // Match IDA at 0x1001c646
    int oldCount = m_Faces.Size();

    m_Faces.Resize(Count);
    m_FaceVertexIndices.Resize(Count * 3);

    // Initialize new faces
    if (oldCount < Count) {
        // Clear new face vertex indices
        memset(&m_FaceVertexIndices[oldCount * 3], 0, sizeof(CKWORD) * 3 * (Count - oldCount));

        // Set channel mask to all channels
        m_FaceChannelMask = 0xFFFF;
    }

    // Initialize new faces with channel mask -1 (0xFFFF)
    // Note: IDA shows this loop runs even if oldCount >= Count (just won't execute)
    for (int i = oldCount; i < Count; i++) {
        m_Faces[i].m_ChannelMask = 0xFFFF;
    }

    UnOptimize();
    return TRUE;
}

int RCKMesh::GetFaceCount() {
    return m_Faces.Size();
}

// Face vertex index methods
void RCKMesh::SetFaceVertexIndex(int FaceIndex, int Vertex1, int Vertex2, int Vertex3) {
    // Match IDA at 0x1001c70d
    if (FaceIndex >= 0 && FaceIndex < m_Faces.Size()) {
        m_FaceVertexIndices[FaceIndex * 3] = Vertex1;
        m_FaceVertexIndices[FaceIndex * 3 + 1] = Vertex2;
        m_FaceVertexIndices[FaceIndex * 3 + 2] = Vertex3;
        UnOptimize();
    }
}

void RCKMesh::GetFaceVertexIndex(int FaceIndex, int &Vertex1, int &Vertex2, int &Vertex3) {
    // Match IDA at 0x1001c7f0
    if (FaceIndex >= 0 && FaceIndex < m_Faces.Size()) {
        Vertex1 = m_FaceVertexIndices[FaceIndex * 3];
        Vertex2 = m_FaceVertexIndices[FaceIndex * 3 + 1];
        Vertex3 = m_FaceVertexIndices[FaceIndex * 3 + 2];
    }
}

// Face material methods
// Match IDA at 0x1001c892
void RCKMesh::SetFaceMaterial(int FaceIndex, CKMaterial *Mat) {
    if (FaceIndex >= 0 && FaceIndex < m_Faces.Size()) {
        const int matGroupIndex = GetMaterialGroupIndex(Mat, TRUE);
        if (matGroupIndex != m_Faces[FaceIndex].m_MatIndex)
            UnOptimize();
        m_Faces[FaceIndex].m_MatIndex = matGroupIndex;
    }
}

// Match IDA at 0x1001c909
CKMaterial *RCKMesh::GetFaceMaterial(int Index) {
    if (Index < 0 || Index >= m_Faces.Size())
        return nullptr;
    return m_MaterialGroups[m_Faces[Index].m_MatIndex]->m_Material;
}

// Face indices access
CKWORD *RCKMesh::GetFacesIndices() {
    return m_FaceVertexIndices.Begin();
}

// Geometry calculations
float RCKMesh::GetRadius() {
    // Match IDA at 0x1001f13d
    if (!(m_Flags & VXMESH_BOUNDINGUPTODATE)) {
        UpdateBoundingVolumes(FALSE);
    }
    return m_Radius;
}

const VxBbox &RCKMesh::GetLocalBox() {
    // Match IDA at 0x1001f163
    if (!(m_Flags & VXMESH_BOUNDINGUPTODATE)) {
        UpdateBoundingVolumes(FALSE);
    }
    return m_LocalBox;
}

void RCKMesh::GetBaryCenter(VxVector *Vector) {
    // Match IDA at 0x1001f189
    if (!(m_Flags & VXMESH_BOUNDINGUPTODATE)) {
        UpdateBoundingVolumes(FALSE);
    }
    *Vector = m_BaryCenter;
}

// Line operations
// Match IDA at 0x1001e2cd
CKBOOL RCKMesh::SetLineCount(int Count) {
    m_LineIndices.Resize(2 * Count);
    return TRUE;
}

// Match IDA at 0x1001e2f3
int RCKMesh::GetLineCount() {
    return m_LineIndices.Size() >> 1;
}

// Match IDA at 0x1001e30e
void RCKMesh::SetLine(int LineIndex, int VIndex1, int VIndex2) {
    m_LineIndices[2 * LineIndex] = VIndex1;
    m_LineIndices[2 * LineIndex + 1] = VIndex2;
}

// Match IDA at 0x1001e353
void RCKMesh::GetLine(int LineIndex, int *VIndex1, int *VIndex2) {
    *VIndex1 = m_LineIndices[2 * LineIndex];
    *VIndex2 = m_LineIndices[2 * LineIndex + 1];
}

CKWORD *RCKMesh::GetLineIndices() {
    return m_LineIndices.Begin();
}

// Vertex weight system
void RCKMesh::SetVertexWeightsCount(int count) {
    if (count <= 0) {
        if (m_VertexWeights) {
            delete m_VertexWeights;
            m_VertexWeights = nullptr;
        }
        return;
    }

    if (!m_VertexWeights) {
        m_VertexWeights = new XArray<float>();
        m_VertexWeights->Resize(count);
        for (int i = 0; i < count; i++) {
            (*m_VertexWeights)[i] = 1.0f;
        }
    } else {
        // Resize existing weights
        int oldCount = (int) m_VertexWeights->Size();
        m_VertexWeights->Resize(count);

        // Initialize new weights to 1.0f
        for (int i = oldCount; i < count; i++) {
            (*m_VertexWeights)[i] = 1.0f;
        }
    }
}

int RCKMesh::GetVertexWeightsCount() {
    return m_VertexWeights ? (int) m_VertexWeights->Size() : 0;
}

void RCKMesh::SetVertexWeight(int index, float w) {
    // Match IDA at 0x1001f5dd
    if (m_VertexWeights) {
        (*m_VertexWeights)[index] = w;
    }
}

float RCKMesh::GetVertexWeight(int index) {
    // Match IDA at 0x1001f60d
    if (m_VertexWeights) {
        return (*m_VertexWeights)[index];
    }
    return 0.0f;
}

// Mesh operations
void RCKMesh::Clean(CKBOOL KeepVertices) {
    // Match IDA at 0x1001f09a
    m_Faces.Clear();
    m_FaceVertexIndices.Clear();
    m_LineIndices.Clear();

    DeleteRenderGroup();

    // Clear material channels (sub_1001BE0B in IDA)
    for (int i = 0; i < m_MaterialChannels.Size(); i++) {
        VxMaterialChannel &channel = m_MaterialChannels[i];
        if (channel.m_UVs != nullptr) {
            delete[] channel.m_UVs;
            channel.m_UVs = nullptr;
        }
    }
    m_MaterialChannels.Clear();

    if (!KeepVertices) {
        m_Vertices.Clear();
        m_VertexColors.Clear();

        if (m_VertexWeights) {
            delete m_VertexWeights;
            m_VertexWeights = nullptr;
        }
    }
}

void RCKMesh::InverseWinding() {
    // Match IDA at 0x1001eaf1
    int faceCount = m_Faces.Size();
    int secondVertexOffset = 1; // Start at index 1 (second vertex of first face)
    for (int i = 0; i < faceCount; ++i) {
        // Swap vertices at indices secondVertexOffset and secondVertexOffset+1 (i.e., indices 1&2, 4&5, 7&8, ...)
        CKWORD temp = m_FaceVertexIndices[secondVertexOffset];
        m_FaceVertexIndices[secondVertexOffset] = m_FaceVertexIndices[secondVertexOffset + 1];
        m_FaceVertexIndices[secondVertexOffset + 1] = temp;
        secondVertexOffset += 3;
    }
    UnOptimize();
    ModifierVertexMove(TRUE, TRUE);
}

void RCKMesh::Consolidate() {
    // Match IDA at 0x1001eb80
    // This method removes degenerate faces (with zero normal) and unused vertices

    int removedFaces = 0;
    int removedVertices = 0;
    int faceCount = m_Faces.Size();
    int newFaceCount = faceCount;
    int i = 0;

    // Remove degenerate faces (faces with zero normal)
    while (i < newFaceCount) {
        CKFace &face = m_Faces[i];
        VxVector &normal = face.m_Normal;

        if (normal.x == 0.0f && normal.y == 0.0f && normal.z == 0.0f) {
            --newFaceCount;
            ++removedFaces;

            if (i != newFaceCount) {
                // Swap with last valid face
                m_Faces[i] = m_Faces[newFaceCount];
                m_FaceVertexIndices[i * 3] = m_FaceVertexIndices[newFaceCount * 3];
                m_FaceVertexIndices[i * 3 + 1] = m_FaceVertexIndices[newFaceCount * 3 + 1];
                m_FaceVertexIndices[i * 3 + 2] = m_FaceVertexIndices[newFaceCount * 3 + 2];
            }
        } else {
            ++i;
        }
    }

    if (newFaceCount != faceCount) {
        SetFaceCount(newFaceCount);
    }

    // Mark used vertices
    int vertexCount = m_Vertices.Size();
    XArray<int> vertexMap;
    vertexMap.Resize(vertexCount);
    for (i = 0; i < vertexCount; ++i) {
        vertexMap[i] = -2; // Mark as unused
    }

    // Mark vertices used by faces
    for (i = 0; i < m_FaceVertexIndices.Size(); ++i) {
        vertexMap[m_FaceVertexIndices[i]] = 666666; // Mark as used
    }

    // Compact vertices
    int newVertexCount = vertexCount;
    for (i = 0; i < newVertexCount; ++i) {
        if (vertexMap[i] == -2) {
            --newVertexCount;
            ++removedVertices;

            // Find last used vertex
            int k = newVertexCount;
            while (k > i && vertexMap[k] == -2) {
                --k;
            }

            if (i == k) {
                vertexMap[i] = -1;
            } else {
                // Move vertex k to position i
                m_Vertices[i] = m_Vertices[k];
                m_VertexColors[i] = m_VertexColors[k];
                vertexMap[k] = i;
                vertexMap[i] = -2;
            }
        } else if (vertexMap[i] == 666666) {
            vertexMap[i] = i; // Keep at same position
        }
    }

    // Resize and remap face indices
    if (newVertexCount != vertexCount) {
        m_Vertices.Resize(newVertexCount);
        m_VertexColors.Resize(newVertexCount);

        for (i = 0; i < m_FaceVertexIndices.Size(); ++i) {
            m_FaceVertexIndices[i] = vertexMap[m_FaceVertexIndices[i]];
        }
    }

    // Update skins if vertices were removed (IDA lines 126-140)
    if (removedVertices) {
        const XObjectPointerArray &entities = m_Context->GetObjectListByType(CKCID_3DENTITY, TRUE);
        for (CKObject **it = entities.Begin(); it != entities.End(); ++it) {
            CK3dEntity *entity = (CK3dEntity *) *it;
            if (entity && entity->GetCurrentMesh() == this) {
                CKSkin *skin = entity->GetSkin();
                if (skin) {
                    skin->RemapVertices(vertexMap);
                }
            }
        }
    }

    if (removedFaces || removedVertices) {
        UnOptimize();
    }
}

void RCKMesh::UnOptimize() {
    // Match IDA at 0x1002a980
    m_Flags &= ~(VXMESH_OPTIMIZED | VXMESH_TRANSPARENCYUPTODATE);
}

// Callback system methods
CKBOOL RCKMesh::AddPreRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument, CKBOOL Temporary) {
    if (!m_RenderCallbacks) {
        m_RenderCallbacks = new CKCallbacksContainer();
    }
    return m_RenderCallbacks->AddPreCallback((void *) Function, Argument, Temporary, m_Context->GetRenderManager());
}

CKBOOL RCKMesh::RemovePreRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument) {
    if (!m_RenderCallbacks) {
        return FALSE;
    }
    return m_RenderCallbacks->RemovePreCallback((void *) Function, Argument);
}

CKBOOL RCKMesh::AddPostRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument, CKBOOL Temporary) {
    if (!m_RenderCallbacks) {
        m_RenderCallbacks = new CKCallbacksContainer();
    }
    return m_RenderCallbacks->AddPostCallback((void *) Function, Argument, Temporary, m_Context->GetRenderManager());
}

CKBOOL RCKMesh::RemovePostRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument) {
    if (!m_RenderCallbacks) {
        return FALSE;
    }
    return m_RenderCallbacks->RemovePostCallback((void *) Function, Argument);
}

void RCKMesh::SetRenderCallBack(CK_MESHRENDERCALLBACK Function, void *Argument) {
    // Match IDA at 0x1001de07
    if (!m_RenderCallbacks) {
        m_RenderCallbacks = new CKCallbacksContainer();
    }

    // Allocate m_Callback if not present
    if (!m_RenderCallbacks->m_Callback) {
        m_RenderCallbacks->m_Callback = new VxCallBack();
    }

    // Set the main callback
    m_RenderCallbacks->m_Callback->callback = (void *) Function;
    m_RenderCallbacks->m_Callback->argument = Argument;
    m_RenderCallbacks->m_Callback->temp = FALSE;
}

void RCKMesh::SetDefaultRenderCallBack() {
    // Match IDA at 0x1001df02
    if (m_RenderCallbacks) {
        delete m_RenderCallbacks->m_Callback;
        m_RenderCallbacks->m_Callback = nullptr;
    }
}

void RCKMesh::RemoveAllCallbacks() {
    // Match IDA at 0x1001df48
    RCKRenderManager *rm = (RCKRenderManager *) m_Context->GetRenderManager();

    if (m_RenderCallbacks) {
        rm->RemoveTemporaryCallback(m_RenderCallbacks);
    }

    if (m_RenderCallbacks) {
        delete m_RenderCallbacks;
    }

    if (m_SubMeshCallbacks) {
        rm->RemoveTemporaryCallback(m_SubMeshCallbacks);
    }

    if (m_SubMeshCallbacks) {
        delete m_SubMeshCallbacks;
    }

    m_SubMeshCallbacks = nullptr;
    m_RenderCallbacks = nullptr;
}

// Sub-mesh callback methods
CKBOOL RCKMesh::AddSubMeshPreRenderCallBack(CK_SUBMESHRENDERCALLBACK Function, void *Argument, CKBOOL Temporary) {
    if (!m_SubMeshCallbacks) {
        m_SubMeshCallbacks = new CKCallbacksContainer();
    }
    return m_SubMeshCallbacks->AddPreCallback((void *) Function, Argument, Temporary, m_Context->GetRenderManager());
}

CKBOOL RCKMesh::RemoveSubMeshPreRenderCallBack(CK_SUBMESHRENDERCALLBACK Function, void *Argument) {
    if (!m_SubMeshCallbacks) {
        return FALSE;
    }
    return m_SubMeshCallbacks->RemovePreCallback((void *) Function, Argument);
}

CKBOOL RCKMesh::AddSubMeshPostRenderCallBack(CK_SUBMESHRENDERCALLBACK Function, void *Argument, CKBOOL Temporary) {
    if (!m_SubMeshCallbacks) {
        m_SubMeshCallbacks = new CKCallbacksContainer();
    }
    return m_SubMeshCallbacks->AddPostCallback((void *) Function, Argument, Temporary, m_Context->GetRenderManager());
}

CKBOOL RCKMesh::RemoveSubMeshPostRenderCallBack(CK_SUBMESHRENDERCALLBACK Function, void *Argument) {
    if (!m_SubMeshCallbacks) {
        return FALSE;
    }
    return m_SubMeshCallbacks->RemovePostCallback((void *) Function, Argument);
}

// Material management
// Match IDA at 0x1001e01b
int RCKMesh::GetMaterialCount() {
    if (!(m_Flags & VXMESH_OPTIMIZED))
        CreateRenderGroups();
    return m_MaterialGroups.Size() - 1;
}

// Match IDA at 0x1001e04c
CKMaterial *RCKMesh::GetMaterial(int index) {
    if (!(m_Flags & VXMESH_OPTIMIZED))
        CreateRenderGroups();
    if ((index + 1) < m_MaterialGroups.Size())
        return m_MaterialGroups[index + 1]->m_Material;
    else
        return nullptr;
}

// Progressive mesh methods (skipped as requested)
void RCKMesh::SetVerticesRendered(int count) {
    // Match IDA at 0x100246d9
    if (!m_ProgressiveMesh)
        return;

    // Clamp count between 0 and vertex count
    int maxCount = m_Vertices.Size();
    int minCount = 0;
    if (count < minCount) count = minCount;
    if (count > maxCount) count = maxCount;

    if (m_ProgressiveMesh->m_VertexCount != count) {
        m_ProgressiveMesh->m_VertexCount = count;
        CKObject::ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
    }
}

int RCKMesh::GetVerticesRendered() {
    // Match IDA at 0x10024751
    if (m_ProgressiveMesh)
        return m_ProgressiveMesh->m_VertexCount;
    return 0;
}

void RCKMesh::EnablePMGeoMorph(CKBOOL enable) {
    // Match IDA at 0x100256aa
    if (!m_ProgressiveMesh)
        return;

    if (m_ProgressiveMesh->m_MorphEnabled != enable) {
        m_ProgressiveMesh->m_MorphEnabled = enable;
        CKObject::ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
    }
}

CKBOOL RCKMesh::IsPMGeoMorphEnabled() {
    return (m_ProgressiveMesh && m_ProgressiveMesh->m_MorphEnabled) ? TRUE : FALSE;
}

void RCKMesh::SetPMGeoMorphStep(int gs) {
    // Match IDA at 0x10025731
    if (!m_ProgressiveMesh)
        return;

    if (m_ProgressiveMesh->m_MorphStep != gs) {
        m_ProgressiveMesh->m_MorphStep = gs;
        CKObject::ModifyObjectFlags(0, CK_OBJECT_UPTODATE);
    }
}

int RCKMesh::GetPMGeoMorphStep() {
    return m_ProgressiveMesh ? m_ProgressiveMesh->m_MorphStep : 0;
}

/**
 * @brief Optimized vertex loading using buffer-based approach.
 *
 * Based on IDA decompilation at 0x10027E1E (RCKMesh::ILoadVertices).
 * Uses LockReadBuffer and VxCopyStructure for efficient bulk memory operations.
 *
 * @param chunk The state chunk to read from
 * @param loadFlags Output parameter for save flags
 * @return 0 on success, -1 on error
 */
int RCKMesh::ILoadVertices(CKStateChunk *chunk, CKDWORD *loadFlags) {
    if (chunk->GetDataVersion() < 9)
        return -1;

    if (!chunk->SeekIdentifier(CK_STATESAVE_MESHVERTICES))
        return 0;

    int vertexCount = chunk->ReadInt();
    SetVertexCount(vertexCount);

    if (vertexCount == 0)
        return 0;

    // Read save flags
    *loadFlags = chunk->ReadDword();

    // Lock read buffer for direct access
    CKDWORD *buffer = (CKDWORD *) chunk->LockReadBuffer();
    CKDWORD *ptr = buffer + 1;

    // Get buffer size and convert endianness
    CKDWORD bufferSize = CKConvertEndian32(*buffer);
    CKConvertEndianArray32(buffer, bufferSize);

    // Read positions (if not skipped via flag 0x10)
    if (!(*loadFlags & 0x10)) {
        VxCopyStructure(vertexCount, m_Vertices.Begin(), 0x20, 0x0C, ptr, 0x0C);
        ptr += 3 * vertexCount;
    }

    // Read diffuse colors
    if (*loadFlags & 0x01) {
        // Only first value - fill all with same color
        m_VertexColors[0].Color = *ptr++;
        VxFillStructure(vertexCount - 1, &m_VertexColors[1].Color, 8, 4, &m_VertexColors[0].Color);
    } else {
        // All colors differ
        VxCopyStructure(vertexCount, &m_VertexColors[0].Color, 8, 4, ptr, 4);
        ptr += vertexCount;
    }

    // Read specular colors
    if (*loadFlags & 0x02) {
        // Only first value - fill all with same color
        m_VertexColors[0].Specular = *ptr++;
        VxFillStructure(vertexCount - 1, &m_VertexColors[1].Specular, 8, 4, &m_VertexColors[0].Specular);
    } else {
        // All specular colors differ
        VxCopyStructure(vertexCount, &m_VertexColors[0].Specular, 8, 4, ptr, 4);
        ptr += vertexCount;
    }

    // Read normals (if not skipped via flag 0x04)
    if (!(*loadFlags & 0x04)) {
        // Copy normals to m_Vertices[i].m_Normal (offset 12 from start of VxVertex)
        char *normalDst = (char *) m_Vertices.Begin() + 12;
        VxCopyStructure(vertexCount, normalDst, 0x20, 0x0C, ptr, 0x0C);
        ptr += 3 * vertexCount;
    }

    // Read UV coordinates
    float *fptr = (float *) ptr;
    if (*loadFlags & 0x08) {
        // Only first value - fill all with same UV
        m_Vertices[0].m_UV.x = *fptr++;
        m_Vertices[0].m_UV.y = *fptr++;
        VxFillStructure(vertexCount - 1, &m_Vertices[1].m_UV, 0x20, 8, &m_Vertices[0].m_UV);
    } else {
        // All UVs differ - copy all
        VxCopyStructure(vertexCount, &m_Vertices[0].m_UV, 0x20, 8, fptr, 8);
    }

    // Restore endianness for buffer and return
    CKConvertEndianArray32(buffer, bufferSize);

    return 0;
}

// CKObject overrides
void RCKMesh::PreSave(CKFile *file, CKDWORD flags) {
    CKBeObject::PreSave(file, flags);

    const int groupCount = (int) m_MaterialGroups.Size();
    for (int i = 0; i < groupCount; ++i) {
        CKMaterialGroup *group = m_MaterialGroups[i];
        file->SaveObject(group ? (CKObject *) group->m_Material : nullptr, flags);
    }

    const int channelCount = GetChannelCount();
    for (int j = 0; j < channelCount; ++j) {
        CKMaterial *mat = GetChannelMaterial(j);
        file->SaveObject((CKObject *) mat, flags);
    }
}

CKStateChunk *RCKMesh::Save(CKFile *file, CKDWORD flags) {
    // Call base class save
    CKStateChunk *baseChunk = CKBeObject::Save(file, flags);

    // Early return check - based on IDA at 0x100273bb
    if (!file && !(flags & CK_STATESAVE_MESHONLY))
        return baseChunk;

    // Create mesh-specific chunk
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_MESH, file);
    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    // Write mesh flags - always written
    chunk->WriteIdentifier(CK_STATESAVE_MESHFLAGS);
    chunk->WriteDword(m_Flags);

    // Only write mesh geometry if not a patch mesh
    if (GetClassID() != CKCID_PATCHMESH) {
        // Write material groups
        int groupCount = (int) m_MaterialGroups.Size();
        if (groupCount > 0) {
            chunk->WriteIdentifier(CK_STATESAVE_MESHMATERIALS);
            chunk->WriteInt(groupCount);
            for (int i = 0; i < groupCount; i++) {
                CKMaterialGroup *group = m_MaterialGroups[i];
                chunk->WriteObject(group ? group->m_Material : nullptr);
                chunk->WriteInt(0); // Reserved
            }
        }

        // Write face data
        int faceCount = GetFaceCount();
        if (faceCount > 0) {
            chunk->WriteIdentifier(CK_STATESAVE_MESHFACES);
            chunk->WriteInt(faceCount);
            for (int j = 0; j < faceCount; j++) {
                // Pack vertex indices: first two as DWORD
                CKWORD idx0 = m_FaceVertexIndices[3 * j];
                CKWORD idx1 = m_FaceVertexIndices[3 * j + 1];
                CKDWORD indices01 = idx0 | (idx1 << 16);
                chunk->WriteDwordAsWords(indices01);

                // Third index and material index
                CKWORD idx2 = m_FaceVertexIndices[3 * j + 2];
                CKWORD matIdx = (CKWORD) m_Faces[j].m_MatIndex;
                CKDWORD idx2AndMat = idx2 | (matIdx << 16);
                chunk->WriteDwordAsWords(idx2AndMat);
            }
        }

        // Write line data using bulk write
        int lineCount = GetLineCount();
        if (lineCount > 0) {
            chunk->WriteIdentifier(CK_STATESAVE_MESHLINES);
            chunk->WriteInt(lineCount);
            // Bulk write line indices - based on IDA at 0x100275b5
            chunk->WriteBuffer_LEndian16(lineCount * 2 * sizeof(CKWORD), m_LineIndices.Begin());
        }

        // Write vertex data using optimized buffer write
        int vertexCount = GetVertexCount();
        if (vertexCount > 0) {
            chunk->WriteIdentifier(CK_STATESAVE_MESHVERTICES);
            CKDWORD saveFlags = GetSaveFlags();
            chunk->WriteInt(vertexCount);
            chunk->WriteDword(saveFlags);

            // Lock write buffer for optimized vertex data - 11 dwords max per vertex
            // Based on IDA at 0x10027632
            CKDWORD *buffer = (CKDWORD *) chunk->LockWriteBuffer(11 * vertexCount);
            float *ptr = (float *) (buffer + 1);

            // Write positions (skip if flag 0x10 set)
            if (!(saveFlags & 0x10)) {
                VxVertex *vtx = m_Vertices.Begin();
                for (int k = 0; k < vertexCount; k++) {
                    *ptr++ = vtx->m_Position.x;
                    *ptr++ = vtx->m_Position.y;
                    *ptr++ = vtx->m_Position.z;
                    vtx++;
                }
            }

            // Write diffuse colors - first value always, rest if flag 0x1 not set
            CKDWORD *dptr = (CKDWORD *) ptr;
            *dptr++ = m_VertexColors[0].Color;
            if (!(saveFlags & 0x01)) {
                for (int k = 1; k < vertexCount; k++)
                    *dptr++ = m_VertexColors[k].Color;
            }

            // Write specular colors - first value always, rest if flag 0x2 not set
            *dptr++ = m_VertexColors[0].Specular;
            if (!(saveFlags & 0x02)) {
                for (int k = 1; k < vertexCount; k++)
                    *dptr++ = m_VertexColors[k].Specular;
            }

            // Write normals (skip if flag 0x4 set)
            if (!(saveFlags & 0x04)) {
                ptr = (float *) dptr;
                VxVertex *vtx = m_Vertices.Begin();
                for (int k = 0; k < vertexCount; k++) {
                    *ptr++ = vtx->m_Normal.x;
                    *ptr++ = vtx->m_Normal.y;
                    *ptr++ = vtx->m_Normal.z;
                    vtx++;
                }
                dptr = (CKDWORD *) ptr;
            }

            // Write UV coordinates - first value always, rest if flag 0x8 not set
            ptr = (float *) dptr;
            *ptr++ = m_Vertices[0].m_UV.x;
            *ptr++ = m_Vertices[0].m_UV.y;
            if (!(saveFlags & 0x08)) {
                for (int k = 1; k < vertexCount; k++) {
                    *ptr++ = m_Vertices[k].m_UV.x;
                    *ptr++ = m_Vertices[k].m_UV.y;
                }
            }

            // Calculate written size and finalize buffer
            int written = (int) (((char *) ptr - (char *) buffer) >> 2);
            *buffer = (CKDWORD) written;
            CKConvertEndianArray32(buffer, written);
            chunk->Skip(written);
        }

        // Write material channels
        int channelCount = GetChannelCount();
        if (channelCount > 0) {
            chunk->WriteIdentifier(CK_STATESAVE_MESHCHANNELS);
            chunk->WriteInt(channelCount);
            for (int ch = 0; ch < channelCount; ch++) {
                VxMaterialChannel &channel = m_MaterialChannels[ch];
                chunk->WriteObject(channel.m_Material);
                chunk->WriteDword(channel.m_Flags);
                chunk->WriteDword((CKDWORD) channel.m_SourceBlend);
                chunk->WriteDword((CKDWORD) channel.m_DestBlend);

                int uvCount = channel.m_UVs ? GetVertexCount() : 0;
                chunk->WriteInt(uvCount);
                for (int uv = 0; uv < uvCount; uv++) {
                    chunk->WriteFloat(channel.m_UVs[uv].x);
                    chunk->WriteFloat(channel.m_UVs[uv].y);
                }
            }
        }
    }

    // Write vertex weights
    if (m_VertexWeights && m_VertexWeights->Size() > 0) {
        chunk->WriteIdentifier(CK_STATESAVE_MESHWEIGHTS);

        // Check if all weights are equal (optimization)
        float *begin = (float *) m_VertexWeights->Begin();
        float *end = (float *) m_VertexWeights->End();
        float *p = begin;
        for (; p < end && *p == *begin; ++p);

        int weightCount = (int) m_VertexWeights->Size();
        chunk->WriteInt(weightCount);

        if (p != end) {
            // Weights differ - write buffer
            chunk->WriteBuffer_LEndian(weightCount * sizeof(float), begin);
        }
        chunk->WriteFloat(*begin); // Always write first weight
    }

    // Write face channel masks if not all 0xFFFF
    CKWORD maskAnd = 0xFFFF;
    for (int i = 0; i < GetFaceCount(); i++)
        maskAnd &= m_Faces[i].m_ChannelMask;

    if (maskAnd != 0xFFFF && m_MaterialChannels.Size() > 0) {
        chunk->WriteIdentifier(CK_STATESAVE_MESHFACECHANMASK);
        int faceCount = GetFaceCount();
        chunk->WriteInt(faceCount);

        // Pack two channel masks per DWORD
        for (int i = 0; i < faceCount / 2; i++) {
            CKDWORD packed = m_Faces[2 * i].m_ChannelMask |
                ((CKDWORD) m_Faces[2 * i + 1].m_ChannelMask << 16);
            chunk->WriteDwordAsWords(packed);
        }
        // Handle odd face count
        if (faceCount & 1)
            chunk->WriteWord(m_Faces[faceCount - 1].m_ChannelMask);
    }

    // Write progressive mesh data
    if (m_ProgressiveMesh && m_ProgressiveMesh->m_Data.Size() > 0) {
        chunk->WriteIdentifier(CK_STATESAVE_PROGRESSIVEMESH);
        chunk->WriteInt(m_ProgressiveMesh->m_VertexCount);
        chunk->WriteInt(m_ProgressiveMesh->m_MorphEnabled);
        chunk->WriteInt(m_ProgressiveMesh->m_MorphStep);
        int dataSize = (int) m_ProgressiveMesh->m_Data.Size();
        chunk->WriteBufferNoSize_LEndian(dataSize * sizeof(CKDWORD), m_ProgressiveMesh->m_Data.Begin());
    }

    // Close the chunk
    if (GetClassID() == CKCID_MESH)
        chunk->CloseChunk();
    else
        chunk->UpdateDataSize();

    return chunk;
}

CKERROR RCKMesh::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk)
        return CKERR_INVALIDPARAMETER;

    // Call base class load
    CKBeObject::Load(chunk, file);

    // Clear existing channels if not loading from file
    if (!file) {
        while (GetChannelCount() > 0)
            RemoveChannel(0);
    }

    // Get data version for format handling
    int dataVersion = chunk->GetDataVersion();
    CKDWORD loadFlags = 0;

    // Read mesh flags
    if (chunk->SeekIdentifier(CK_STATESAVE_MESHFLAGS)) {
        CKDWORD flags = chunk->ReadDword();
        flags &= VXMESH_ALLFLAGS;
        const CKDWORD allowedMeshFlags =
            VXMESH_VISIBLE |
            VXMESH_RENDERCHANNELS |
            VXMESH_HASTRANSPARENCY |
            VXMESH_PRELITMODE |
            VXMESH_WRAPU |
            VXMESH_WRAPV |
            VXMESH_FORCETRANSPARENCY |
            VXMESH_TRANSPARENCYUPTODATE |
            VXMESH_UV_CHANGED |
            VXMESH_NORMAL_CHANGED |
            VXMESH_COLOR_CHANGED |
            VXMESH_POS_CHANGED |
            VXMESH_HINTDYNAMIC |
            VXMESH_GENNORMALS |
            VXMESH_PROCEDURALUV |
            VXMESH_PROCEDURALPOS |
            VXMESH_STRIPIFY |
            VXMESH_MONOMATERIAL |
            VXMESH_PM_BUILDNORM |
            VXMESH_BWEIGHTS_CHANGED;
        flags &= allowedMeshFlags;
        SetFlags(flags);
    }

    // Temporary material group index mapping
    XArray<int> groupIndices;
    groupIndices.Resize(0);

    if (dataVersion >= 9) {
        // New format (version 9+)
        if (GetClassID() != CKCID_PATCHMESH) {
            // Load material groups
            if (chunk->SeekIdentifier(CK_STATESAVE_MESHMATERIALS)) {
                int groupCount = chunk->ReadInt();

                // Clear existing material groups - IDA calls DeleteRenderGroup here
                DeleteRenderGroup();

                groupIndices.Resize(groupCount);

                for (int i = 0; i < groupCount; i++) {
                    CKMaterial *mat = (CKMaterial *) chunk->ReadObject(m_Context);
                    if (i > 0 && !mat) {
                        // Null material after first - use group 0
                        groupIndices[i] = 0;
                    } else {
                        CKMaterialGroup *group = new CKMaterialGroup();
                        group->m_Material = (RCKMaterial *) mat;
                        groupIndices[i] = (int) m_MaterialGroups.Size();
                        m_MaterialGroups.PushBack(group);
                    }
                    chunk->ReadInt(); // Reserved
                }
            }

            // Load vertex data using optimized buffer read
            ILoadVertices(chunk, &loadFlags);

            // Load face data
            if (chunk->SeekIdentifier(CK_STATESAVE_MESHFACES)) {
                int faceCount = chunk->ReadInt();
                if (faceCount > 0) {
                    SetFaceCount(faceCount);
                    for (int j = 0; j < faceCount; j++) {
                        // Read packed indices
                        CKDWORD indices01 = chunk->ReadDwordAsWords();
                        m_FaceVertexIndices[3 * j] = (CKWORD) (indices01 & 0xFFFF);
                        m_FaceVertexIndices[3 * j + 1] = (CKWORD) (indices01 >> 16);

                        CKDWORD idx2AndMat = chunk->ReadDwordAsWords();
                        m_FaceVertexIndices[3 * j + 2] = (CKWORD) (idx2AndMat & 0xFFFF);

                        // Map material index through group indices if available
                        int matIdx = (idx2AndMat >> 16);
                        if (groupIndices.Size() > 0 && matIdx < groupIndices.Size())
                            m_Faces[j].m_MatIndex = groupIndices[matIdx];
                        else
                            m_Faces[j].m_MatIndex = matIdx;
                    }
                }
            }

            // Load line data using bulk read
            if (chunk->SeekIdentifier(CK_STATESAVE_MESHLINES)) {
                int lineCount = chunk->ReadInt();
                SetLineCount(lineCount);
                chunk->ReadAndFillBuffer_LEndian16(m_LineIndices.Begin());
            }

            // Rebuild geometry
            UnOptimize();
            if (loadFlags & 0x04)
                BuildNormals();
            else
                BuildFaceNormals();
        }
    } else {
        // Legacy format (version < 9) - use old loader
        if (GetClassID() != CKCID_PATCHMESH) {
            // Call legacy load function (sub_1007C340 in IDA)
            // For now, fall back to simple element-wise reading
            if (chunk->SeekIdentifier(CK_STATESAVE_MESHVERTICES)) {
                int vertexCount = chunk->ReadInt();
                CKDWORD saveFlags = chunk->ReadDword();

                if (vertexCount > 0) {
                    SetVertexCount(vertexCount);

                    // Read vertex positions
                    for (int i = 0; i < vertexCount; i++) {
                        VxVector pos;
                        pos.x = chunk->ReadFloat();
                        pos.y = chunk->ReadFloat();
                        pos.z = chunk->ReadFloat();
                        SetVertexPosition(i, &pos);
                    }

                    // Read vertex normals if present
                    if (saveFlags & 0x02) {
                        for (int i = 0; i < vertexCount; i++) {
                            VxVector normal;
                            normal.x = chunk->ReadFloat();
                            normal.y = chunk->ReadFloat();
                            normal.z = chunk->ReadFloat();
                            SetVertexNormal(i, &normal);
                        }
                    }

                    // Read vertex colors if present
                    if (saveFlags & 0x04) {
                        for (int i = 0; i < vertexCount; i++) {
                            CKDWORD color = chunk->ReadDword();
                            SetVertexColor(i, color);
                        }
                    }
                }
            }

            // Load face data (old format)
            if (chunk->SeekIdentifier(CK_STATESAVE_MESHFACES)) {
                int faceCount = chunk->ReadInt();
                if (faceCount > 0) {
                    SetFaceCount(faceCount);
                    for (int i = 0; i < faceCount; i++) {
                        int idx0 = chunk->ReadWord();
                        int idx1 = chunk->ReadWord();
                        int idx2 = chunk->ReadWord();
                        int matIdx = (int) chunk->ReadDword();
                        SetFaceVertexIndex(i, idx0, idx1, idx2);
                        m_Faces[i].m_MatIndex = matIdx;
                    }
                }
            }

            // Load line data (old format)
            if (chunk->SeekIdentifier(CK_STATESAVE_MESHLINES)) {
                int lineCount = chunk->ReadInt();
                SetLineCount(lineCount);
                for (int i = 0; i < lineCount; i++) {
                    int idx0 = chunk->ReadWord();
                    int idx1 = chunk->ReadWord();
                    SetLine(i, idx0, idx1);
                }
            }
        }
    }

    // Load material channels - applies to all versions
    if (GetClassID() != CKCID_PATCHMESH && chunk->SeekIdentifier(CK_STATESAVE_MESHCHANNELS)) {
        int channelCount = chunk->ReadInt();
        for (int k = 0; k < channelCount; k++) {
            CKMaterial *material = (CKMaterial *) chunk->ReadObject(m_Context);
            int channelIndex = AddChannel(material, TRUE);

            if (channelIndex < 0) {
                // Skip channel data if channel wasn't added
                chunk->ReadDword(); // flags
                chunk->ReadDword(); // source blend
                chunk->ReadDword(); // dest blend
                int uvCount = chunk->ReadInt();
                for (int m = 0; m < uvCount; m++) {
                    chunk->ReadFloat();
                    chunk->ReadFloat();
                }
            } else {
                m_MaterialChannels[channelIndex].m_Flags = chunk->ReadDword();
                m_MaterialChannels[channelIndex].m_SourceBlend = (VXBLEND_MODE) chunk->ReadDword();
                m_MaterialChannels[channelIndex].m_DestBlend = (VXBLEND_MODE) chunk->ReadDword();

                int uvCount = chunk->ReadInt();
                if (uvCount > 0) {
                    CKDWORD stride;
                    float *uvPtr = (float *) GetTextureCoordinatesPtr(&stride, channelIndex);
                    for (int m = 0; m < uvCount; m++) {
                        uvPtr[0] = chunk->ReadFloat();
                        uvPtr[1] = chunk->ReadFloat();
                        uvPtr = (float *) ((char *) uvPtr + stride);
                    }
                }
            }
        }
    }

    // Load vertex weights
    int weightSize = chunk->SeekIdentifierAndReturnSize(CK_STATESAVE_MESHWEIGHTS);
    if (weightSize > 0) {
        int weightCount = chunk->ReadInt();

        // Format is either:
        // - constant: [int count][float w0]                         => 8 bytes
        // - full:     [int count][int bytes][float[count] ...]      => 8 + 4*count bytes
        // - full+tail:[int count][int bytes][float[count] ...][w0]  => 12 + 4*count bytes
        const int expectedConstantSize = (int) (sizeof(CKDWORD) + sizeof(float));
        if (weightSize == expectedConstantSize) {
            SetVertexWeightsCount(weightCount);
            float weight = chunk->ReadFloat();
            if (m_VertexWeights) {
                float *begin = (float *) m_VertexWeights->Begin();
                float *end = (float *) m_VertexWeights->End();
                for (float *p = begin; p < end; ++p)
                    *p = weight;
            }
        } else {
            SetVertexWeightsCount(weightCount);
            if (m_VertexWeights)
                chunk->ReadAndFillBuffer_LEndian(m_VertexWeights->Begin());

            const int expectedNoTail = (int) (sizeof(CKDWORD) + sizeof(CKDWORD) + weightCount * sizeof(float));
            if (weightSize >= expectedNoTail + (int) sizeof(float)) {
                // Some formats (and our Save path) append w0; consume it to keep parsing aligned.
                (void) chunk->ReadFloat();
            }
        }
    }

    // Load face channel masks
    if (chunk->SeekIdentifier(CK_STATESAVE_MESHFACECHANMASK)) {
        int maskCount = chunk->ReadInt();
        int faceCount = GetFaceCount();

        // IDA logic: calculate pairs and remainder
        int pairCount = maskCount >> 1;
        int hasRemainder = maskCount - 2 * pairCount;

        // Boundary check: if our face count is less than file's mask count
        if (faceCount < maskCount) {
            pairCount = faceCount >> 1;
            hasRemainder = 0;
        }

        // Read pairs
        for (int i = 0; i < pairCount; i++) {
            CKDWORD packed = chunk->ReadDwordAsWords();
            m_Faces[2 * i].m_ChannelMask = (CKWORD) (packed & 0xFFFF);
            m_Faces[2 * i + 1].m_ChannelMask = (CKWORD) (packed >> 16);
        }

        // Read remainder (odd face)
        if (hasRemainder) {
            CKWORD mask = chunk->ReadWord();
            m_Faces[maskCount - 1].m_ChannelMask = mask;
        }
    }

    // Load progressive mesh data
    int pmSize = chunk->SeekIdentifierAndReturnSize(CK_STATESAVE_PROGRESSIVEMESH);
    if (pmSize == -1) {
        DestroyPM();
    } else {
        if (!m_ProgressiveMesh)
            m_ProgressiveMesh = new CKProgressiveMesh();

        m_ProgressiveMesh->m_VertexCount = chunk->ReadInt();
        m_ProgressiveMesh->m_MorphEnabled = chunk->ReadInt();
        m_ProgressiveMesh->m_MorphStep = chunk->ReadInt();
        pmSize -= 12;

        m_ProgressiveMesh->m_Data.Resize(pmSize / sizeof(CKDWORD));
        chunk->ReadAndFillBuffer_LEndian(pmSize, m_ProgressiveMesh->m_Data.Begin());

        AddPreRenderCallBack(ProgressiveMeshPreRenderCallback, this, FALSE);
    }

    return CK_OK;
}

void RCKMesh::CheckPreDeletion() {
    // Match IDA at 0x10026521
    // Call base class first
    CKObject::CheckPreDeletion();

    // Collect materials that need to be removed (checking IsToBeDeleted flag)
    XArray<CKMaterial *> materialsToRemove;
    int matCount = GetMaterialCount();
    for (int i = 0; i < matCount; ++i) {
        CKMaterial *mat = GetMaterial(i);
        if (mat && mat->IsToBeDeleted()) {
            materialsToRemove.PushBack(mat);
        }
    }

    // Replace marked materials with NULL
    for (CKMaterial **it = materialsToRemove.Begin(); it != materialsToRemove.End(); ++it) {
        ReplaceMaterial(*it, NULL);
    }

    // Clear channel materials that are being deleted
    int channelCount = GetChannelCount();
    for (int i = 0; i < channelCount; ++i) {
        CKMaterial *mat = GetChannelMaterial(i);
        if (mat && mat->IsToBeDeleted()) {
            SetChannelMaterial(i, NULL);
        }
    }
}

int RCKMesh::GetMemoryOccupation() {
    // Match IDA at 0x10026678
    // Call base class and add this class size
    int size = CKBeObject::GetMemoryOccupation() + (sizeof(RCKMesh) - sizeof(CKBeObject));

    // Vertex array: sizeof(VxVertex) per vertex
    size += m_Vertices.GetMemoryOccupation(FALSE);

    // Vertex colors: sizeof(VxColors) per color
    size += m_VertexColors.GetMemoryOccupation(FALSE);

    // Faces: sizeof(CKFace) per face
    size += m_Faces.GetMemoryOccupation(FALSE);

    // Render callbacks
    if (m_RenderCallbacks) {
        // Add callback array sizes + callback container
        size += m_RenderCallbacks->m_PreCallBacks.GetMemoryOccupation(FALSE);
        size += m_RenderCallbacks->m_PostCallBacks.GetMemoryOccupation(FALSE);
        size += sizeof(CKCallbacksContainer);
    }

    // Material groups: sizeof(CKMaterialGroup) each plus face indices
    size += m_MaterialGroups.GetMemoryOccupation(FALSE);
    for (int i = 0; i < m_MaterialGroups.Size(); ++i) {
        CKMaterialGroup *group = m_MaterialGroups[i];
        if (group) {
            // Add size from face indices array
            size += group->m_FaceIndices.GetMemoryOccupation(FALSE);
            // Each primitive entry has indices
            for (int j = 0; j < group->m_Primitives.Size(); ++j) {
                size += group->m_Primitives[j].m_Indices.GetMemoryOccupation(FALSE);
            }
        }
    }

    // Material channels: sizeof(VxMaterialChannel) each
    for (int i = 0; i < m_MaterialChannels.Size(); ++i) {
        size += sizeof(VxMaterialChannel);
        // If channel has UV data
        if (m_MaterialChannels[i].m_UVs) {
            size += m_Vertices.GetMemoryOccupation(FALSE);
        }
        // If channel has custom face indices
        if (m_MaterialChannels[i].m_FaceIndices) {
            size += m_MaterialChannels[i].m_FaceIndices->GetMemoryOccupation(FALSE);
        }
    }

    // Progressive mesh
    if (m_ProgressiveMesh) {
        size += m_ProgressiveMesh->m_Data.GetMemoryOccupation(FALSE);
    }

    return size;
}

int RCKMesh::IsObjectUsed(CKObject *o, CK_CLASSID cid) {
    // Match IDA at 0x10026869
    // Check if object is a material
    if (cid == CKCID_MATERIAL) {
        // Check material groups first (not channels!)
        int matCount = GetMaterialCount();
        for (int i = 0; i < matCount; ++i) {
            if (o == GetMaterial(i)) {
                return 1;
            }
        }
    }
    // Call base class for other checks
    return CKBeObject::IsObjectUsed(o, cid);
}

// Dependencies Functions
CKERROR RCKMesh::PrepareDependencies(CKDependenciesContext &context) {
    // Call base class first - matches IDA at 0x10028c2a
    CKERROR result = CKBeObject::PrepareDependencies(context);
    if (result != CK_OK)
        return result;

    // Check if material dependencies should be processed (flag bit 0)
    // Corresponds to IDA's GetClassDependencies(a2, 32) & 1
    if ((context.GetClassDependencies(CKCID_MESH) & 1) != 0) {
        // Iterate through all materials in material groups
        int materialCount = GetMaterialCount();
        for (int i = 0; i < materialCount; ++i) {
            CKMaterial *mat = GetMaterial(i);
            if (mat) {
                mat->PrepareDependencies(context);
            }
        }

        // Iterate through material channels
        for (int i = 0; i < m_MaterialChannels.Size(); ++i) {
            if (m_MaterialChannels[i].m_Material) {
                m_MaterialChannels[i].m_Material->PrepareDependencies(context);
            }
        }
    }

    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR RCKMesh::RemapDependencies(CKDependenciesContext &context) {
    // Call base class first - matches IDA at 0x10028d3c
    CKERROR result = CKBeObject::RemapDependencies(context);
    if (result != CK_OK)
        return result;

    // Check if material dependencies should be remapped (flag bit 0)
    // Corresponds to IDA's GetClassDependencies(a2, 32) & 1
    if ((context.GetClassDependencies(CKCID_MESH) & 1) != 0) {
        // Remap material groups
        for (int i = 0; i < m_MaterialGroups.Size(); ++i) {
            CKMaterialGroup *group = m_MaterialGroups[i];
            if (group && group->m_Material) {
                group->m_Material = (RCKMaterial *) context.Remap(group->m_Material);
            }
        }

        // Remap material channel materials
        for (int i = 0; i < m_MaterialChannels.Size(); ++i) {
            if (m_MaterialChannels[i].m_Material) {
                m_MaterialChannels[i].m_Material = (RCKMaterial *) context.Remap(m_MaterialChannels[i].m_Material);
            }
        }
    }

    return CK_OK;
}

CKERROR RCKMesh::Copy(CKObject &o, CKDependenciesContext &context) {
    // Match IDA at 0x10028e2b
    // Call base class first
    CKERROR result = CKBeObject::Copy(o, context);
    if (result != CK_OK)
        return result;

    RCKMesh *source = (RCKMesh *) &o;

    // Get class dependencies flags
    CKDWORD classDeps = context.GetClassDependencies(CKCID_MESH);

    // Copy transparency flag (bit 0x02) from source
    m_Flags &= ~VXMESH_VISIBLE;
    m_Flags |= source->m_Flags & VXMESH_VISIBLE;

    // Copy vertex data using memcpy for efficiency
    int vertexCount = source->GetVertexCount();
    SetVertexCount(vertexCount);
    if (vertexCount > 0) {
        // Copy vertices (VxVertex = 32 bytes)
        if (m_Vertices.Size() > 0 && source->m_Vertices.Size() > 0) {
            memcpy(&m_Vertices[0], &source->m_Vertices[0], 32 * vertexCount);
        }
        // Copy vertex colors (VxColors = 8 bytes)
        if (m_VertexColors.Size() > 0 && source->m_VertexColors.Size() > 0) {
            memcpy(&m_VertexColors[0], &source->m_VertexColors[0], 8 * vertexCount);
        }
    }

    // Delete existing render groups and copy material groups from source
    DeleteRenderGroup();
    for (int i = 0; i < source->m_MaterialGroups.Size(); ++i) {
        CKMaterialGroup *srcGroup = source->m_MaterialGroups[i];
        // Create new group and copy from source (matches IDA VxMaterial copy pattern)
        CKMaterialGroup *newGroup = new CKMaterialGroup();
        if (newGroup) {
            // Copy material pointer
            newGroup->m_Material = srcGroup->m_Material;
            // Copy all other fields from source group
            newGroup->m_HasValidPrimitives = srcGroup->m_HasValidPrimitives;
            newGroup->m_MinVertexIndex = srcGroup->m_MinVertexIndex;
            newGroup->m_MaxVertexIndex = srcGroup->m_MaxVertexIndex;
            newGroup->m_BaseVertex = srcGroup->m_BaseVertex;
            newGroup->m_VertexCount = srcGroup->m_VertexCount;
            newGroup->m_RemapData = 0;
            if (srcGroup->m_RemapData) {
                CKVBuffer *srcVb = GetVBuffer(srcGroup);
                if (srcVb) {
                    CKVBuffer *dstVb = new CKVBuffer();
                    if (dstVb) {
                        dstVb->m_Vertices = srcVb->m_Vertices;
                        dstVb->m_Colors = srcVb->m_Colors;
                        dstVb->m_UVs = srcVb->m_UVs;
                        dstVb->m_VertexRemap = srcVb->m_VertexRemap;
                        newGroup->m_RemapData = static_cast<CKDWORD>(reinterpret_cast<uintptr_t>(dstVb));
                    }
                }
            }
            m_MaterialGroups.PushBack(newGroup);
        }
    }

    // Copy face data
    int faceCount = source->GetFaceCount();
    m_Faces.Resize(faceCount);
    m_FaceVertexIndices.Resize(3 * faceCount);
    if (faceCount > 0) {
        // Copy face structures (CKFace = sizeof(CKFace) bytes per face)
        if (m_Faces.Size() > 0 && source->m_Faces.Size() > 0) {
            memcpy(&m_Faces[0], &source->m_Faces[0], sizeof(CKFace) * faceCount);
        }
        // Copy face vertex indices (3 WORDs per face)
        if (m_FaceVertexIndices.Size() > 0 && source->m_FaceVertexIndices.Size() > 0) {
            memcpy(&m_FaceVertexIndices[0], &source->m_FaceVertexIndices[0], sizeof(CKWORD) * 3 * faceCount);
        }
    }
    SetFaceCount(faceCount);

    // Copy channels
    int channelCount = source->GetChannelCount();
    for (int i = 0; i < channelCount; ++i) {
        CKMaterial *channelMat = source->GetChannelMaterial(i);
        AddChannel(channelMat, FALSE);
        SetChannelSourceBlend(i, source->GetChannelSourceBlend(i));
        SetChannelDestBlend(i, source->GetChannelDestBlend(i));
        SetChannelFlags(i, source->GetChannelFlags(i));

        // Copy channel UV coords if not using default UVs (flag 0x800000)
        if (!(source->GetChannelFlags(i) & VXCHANNEL_SAMEUV)) {
            CKDWORD destStride, srcStride;
            void *destUV = GetTextureCoordinatesPtr(&destStride, i);
            void *srcUV = source->GetTextureCoordinatesPtr(&srcStride, i);
            if (destUV && srcUV) {
                CKBYTE *dest = (CKBYTE *) destUV;
                CKBYTE *src = (CKBYTE *) srcUV;
                for (int v = 0; v < vertexCount; ++v) {
                    // Copy UV (2 floats = 8 bytes)
                    memcpy(dest, src, 8);
                    dest += destStride;
                    src += srcStride;
                }
            }
        }
    }

    // Copy vertex weights
    int weightsCount = source->GetVertexWeightsCount();
    SetVertexWeightsCount(weightsCount);
    if (weightsCount > 0) {
        float *destWeights = GetVertexWeightsPtr();
        float *srcWeights = source->GetVertexWeightsPtr();
        if (destWeights && srcWeights) {
            memcpy(destWeights, srcWeights, sizeof(float) * weightsCount);
        }
    }

    // Copy progressive mesh if present
    // Match IDA at 0x10029251-0x100292cc
    if (source->m_ProgressiveMesh) {
        // Allocate and copy using the copy constructor (matches IDA sub_100298F0)
        m_ProgressiveMesh = new CKProgressiveMesh(*source->m_ProgressiveMesh);
        // Add the pre-render callback
        AddPreRenderCallBack(ProgressiveMeshPreRenderCallback, this, FALSE);
    }

    return CK_OK;
}

// Scene management
void RCKMesh::AddToScene(CKScene *scene, CKBOOL dependencies) {
    // Match IDA at 0x10026411
    if (!scene)
        return;

    CKBeObject::AddToScene(scene, dependencies);

    if (dependencies) {
        // Add all materials to the scene
        int materialCount = GetMaterialCount();
        for (int i = 0; i < materialCount; ++i) {
            CKMaterial *mat = GetMaterial(i);
            if (mat) {
                mat->AddToScene(scene, dependencies);
            }
        }
    }
}

void RCKMesh::RemoveFromScene(CKScene *scene, CKBOOL dependencies) {
    // Match IDA at 0x10026499
    if (!scene)
        return;

    CKBeObject::RemoveFromScene(scene, dependencies);

    if (dependencies) {
        // Remove all materials from the scene
        int materialCount = GetMaterialCount();
        for (int i = 0; i < materialCount; ++i) {
            CKMaterial *mat = GetMaterial(i);
            if (mat) {
                mat->RemoveFromScene(scene, dependencies);
            }
        }
    }
}

// Class Registering
CKSTRING RCKMesh::GetClassName() {
    return (CKSTRING) "Mesh";
}

int RCKMesh::GetDependenciesCount(int mode) {
    int count;
    switch (mode) {
    case 1:
        count = 1;
        break;
    case 2:
        count = 1;
        break;
    case 3:
        count = 0;
        break;
    case 4:
        count = 1;
        break;
    default:
        count = 0;
        break;
    }
    return count;
}

CKSTRING RCKMesh::GetDependencies(int i, int mode) {
    if (i == 0) {
        return (CKSTRING) "Material";
    }
    return nullptr;
}

void RCKMesh::Register() {
    // Based on IDA decompilation at 0x10028b53
    CKCLASSNOTIFYFROMCID(RCKMesh, CKCID_MATERIAL);
    CKPARAMETERFROMCLASS(RCKMesh, CKPGUID_MESH);
    CKCLASSDEFAULTOPTIONS(RCKMesh, CK_GENERALOPTIONS_CANUSECURRENTOBJECT);
}

CKMesh *RCKMesh::CreateInstance(CKContext *Context) {
    return new RCKMesh(Context);
}

CK_CLASSID RCKMesh::m_ClassID = CKCID_MESH;

// Helper methods
CKDWORD RCKMesh::GetSaveFlags() {
    // Match IDA at 0x10026f1f
    // Start with all save flags enabled (binary 1111)
    // Bit 0 (0x01): All diffuse colors identical - only save first
    // Bit 1 (0x02): All specular colors identical - only save first
    // Bit 2 (0x04): Normals need to be saved (cleared if can be rebuilt from faces)
    // Bit 3 (0x08): All UVs identical - only save first
    // Bit 4 (0x10): Skip positions (flag 0x200000 set)
    CKDWORD flags = 0x0F;

    int vertexCount = m_Vertices.Size();
    int faceCount = m_Faces.Size();

    // Get first vertex color and specular for comparison
    CKDWORD firstColor = m_VertexColors.Size() > 0 ? m_VertexColors[0].Color : 0;
    CKDWORD firstSpecular = m_VertexColors.Size() > 0 ? m_VertexColors[0].Specular : 0;

    // Check flag 0x200000 - if set, add flag 0x10 (skip positions)
    if ((m_Flags & VXMESH_PROCEDURALPOS) != 0)
        flags |= 0x10;

    // Check if all UVs are identical (if not 0x100000 flag set)
    // IDA: if UV differs, clear bit 3 (0x08)
    if (!(m_Flags & VXMESH_PROCEDURALUV) && vertexCount > 0) {
        float firstU = m_Vertices[0].m_UV.x;
        float firstV = m_Vertices[0].m_UV.y;
        for (int i = 0; i < vertexCount; ++i) {
            if (m_Vertices[i].m_UV.x != firstU || m_Vertices[i].m_UV.y != firstV) {
                flags &= ~0x08; // UVs differ, clear bit 3
                break;
            }
        }
    }

    // Check if all diffuse colors are identical
    // IDA: if color differs, clear bit 0 (0x01)
    for (int j = 0; j < vertexCount; ++j) {
        if (m_VertexColors[j].Color != firstColor) {
            flags &= ~0x01;
            break;
        }
    }

    // Check if all specular colors are identical
    // IDA: if specular differs, clear bit 1 (0x02)
    for (int j = 0; j < vertexCount; ++j) {
        if (m_VertexColors[j].Specular != firstSpecular) {
            flags &= ~0x02;
            break;
        }
    }

    // Check if normals need to be saved (0x280000 flag check)
    // IDA: when (m_Flags & 0x280000) == 0, it calls BuildFaceNormals and then
    // tries to determine whether vertex normals can be regenerated from faces.
    if (!(m_Flags & (VXMESH_GENNORMALS | VXMESH_PROCEDURALPOS)) && vertexCount > 0) {
        BuildFaceNormals();

        // Build vertex normals from face normals
        XArray<VxVector> vertexNormals;
        vertexNormals.Resize(vertexCount);
        for (int i = 0; i < vertexCount; ++i) {
            vertexNormals[i] = VxVector(0, 0, 0);
        }

        CKWORD *faceIndices = m_FaceVertexIndices.Begin();
        for (int f = 0; f < faceCount; ++f) {
            VxVector &faceNormal = m_Faces[f].m_Normal;
            vertexNormals[faceIndices[0]] += faceNormal;
            vertexNormals[faceIndices[1]] += faceNormal;
            vertexNormals[faceIndices[2]] += faceNormal;
            faceIndices += 3;
        }

        // Calculate average difference between computed and stored normals
        VxVector totalDiff(0, 0, 0);
        for (int i = 0; i < vertexCount; ++i) {
            VxVector computed = vertexNormals[i];
            computed.Normalize();
            VxVector stored = m_Vertices[i].m_Normal;
            stored.Normalize();
            VxVector diff = computed - stored;
            totalDiff += diff;
        }
        totalDiff *= (1.0f / vertexCount);

        // If normals can be rebuilt from faces, clear bit 2 (don't save them)
        if (totalDiff.Magnitude() < 0.001f) {
            flags &= ~0x04;
        }
    }

    return flags;
}

void RCKMesh::UpdateBoundingVolumes(CKBOOL force) {
    // Match IDA at 0x1001f1c4
    // IDA version always computes (no early return based on flag check)
    // The 'force' parameter is passed but not used in the original

    int vertexCount = m_Vertices.Size();
    if (vertexCount <= 0) {
        // IDA: Zero everything and set valid flag
        m_BaryCenter = VxVector(0, 0, 0);
        m_LocalBox.Min = VxVector(0, 0, 0);
        m_LocalBox.Max = VxVector(0, 0, 0);
        m_Radius = 0.0f;
        m_Flags |= 0x01; // Set valid flag
        return;
    }

    // Initialize with first vertex
    VxVector minPos = m_Vertices[0].m_Position;
    VxVector maxPos = m_Vertices[0].m_Position;
    VxVector center = m_Vertices[0].m_Position;

    // Calculate bounding box and accumulate center
    for (int i = 1; i < vertexCount; i++) {
        VxVector &pos = m_Vertices[i].m_Position;
        center += pos;

        if (pos.x < minPos.x) minPos.x = pos.x;
        if (pos.y < minPos.y) minPos.y = pos.y;
        if (pos.z < minPos.z) minPos.z = pos.z;
        if (pos.x > maxPos.x) maxPos.x = pos.x;
        if (pos.y > maxPos.y) maxPos.y = pos.y;
        if (pos.z > maxPos.z) maxPos.z = pos.z;
    }

    // Calculate barycenter
    center *= (1.0f / (float) vertexCount);
    m_BaryCenter = center;

    // Calculate radius squared (distance from barycenter to farthest vertex)
    float radiusSq = 0.0f;
    for (int i = 0; i < vertexCount; i++) {
        VxVector diff = m_Vertices[i].m_Position - m_BaryCenter;
        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        if (distSq > radiusSq) {
            radiusSq = distSq;
        }
    }
    m_Radius = sqrtf(radiusSq);

    // Set bounding box
    m_LocalBox.Min = minPos;
    m_LocalBox.Max = maxPos;

    m_Flags |= VXMESH_BOUNDINGUPTODATE;
}

// Material channel management
int RCKMesh::AddChannel(CKMaterial *material, CKBOOL CopySrcUv) {
    // Match IDA at 0x1001d034
    if (!material)
        return -1;

    // Check if channel already exists
    int existingIndex = GetChannelByMaterial(material);
    if (existingIndex >= 0)
        return existingIndex;

    VxMaterialChannel channel;
    m_MaterialChannels.PushBack(channel);

    // Get pointer to newly added channel
    int channelIndex = m_MaterialChannels.Size() - 1;
    VxMaterialChannel *newChannel = &m_MaterialChannels[channelIndex];

    // IDA: v17.m_Material = a2; (the new channel must reference the material)
    newChannel->m_Material = material;

    // Allocate and initialize UV array (original allocates and zeros even when CopySrcUv == FALSE)
    int vertexCount = m_Vertices.Size();
    Vx2DVector *uv = new Vx2DVector[vertexCount];
    for (int i = 0; i < vertexCount; ++i) {
        uv[i].x = 0.0f;
        uv[i].y = 0.0f;
    }
    newChannel->m_UVs = uv;

    // Copy UV from vertices if requested
    if (CopySrcUv) {
        VxVertex *vertices = m_Vertices.Begin();
        for (int i = 0; i < vertexCount; ++i) {
            uv[i].x = vertices[i].m_UV.x;
            uv[i].y = vertices[i].m_UV.y;
        }
    }

    UVChanged();
    m_FaceChannelMask = 0xFFFF;

    // Update face channel masks
    CKWORD channelBit = static_cast<CKWORD>(1u << channelIndex);
    for (CKFace *face = m_Faces.Begin(); face != m_Faces.End(); ++face) {
        face->m_ChannelMask |= channelBit;
    }

    return channelIndex;
}

void RCKMesh::RemoveChannel(CKMaterial *material) {
    int index = GetChannelByMaterial(material);
    if (index >= 0) {
        RemoveChannel(index);
    }
}

void RCKMesh::RemoveChannel(int index) {
    // Match IDA at 0x1001d2a9
    if (index < m_MaterialChannels.Size()) {
        VxMaterialChannel &channel = m_MaterialChannels[index];
        // Clear channel (VxMaterialChannel::Clear semantics)
        channel.Clear();
        m_MaterialChannels.RemoveAt(index);
        m_FaceChannelMask = 0xFFFF;
        UVChanged();
    }
}

// Match IDA at 0x1001d696
int RCKMesh::GetChannelByMaterial(CKMaterial *material) {
    if (!material)
        return -1;
    for (int i = 0; i < m_MaterialChannels.Size(); i++) {
        if (m_MaterialChannels[i].m_Material == material) {
            return i;
        }
    }
    return -1;
}

void RCKMesh::DeleteRenderGroup() {
    // Match IDA at 0x1001e1d8
    for (int i = 0; i < m_MaterialGroups.Size(); ++i) {
        CKMaterialGroup *group = m_MaterialGroups[i];
        if (group) {
            DeleteVBuffer(group);
            delete group;
        }
    }
    m_MaterialGroups.Clear();
    m_Valid = FALSE;
}

// Match IDA at 0x1002AB90
int RCKMesh::GetMaterialGroupIndex(CKMaterial *mat, CKBOOL create) {
    for (int i = 0; i < m_MaterialGroups.Size(); ++i) {
        if (m_MaterialGroups[i]->m_Material == mat)
            return i;
    }
    if (create)
        return CreateNewMaterialGroup(mat);
    else
        return -1;
}

// Match IDA at 0x1002ac70
int RCKMesh::CreateNewMaterialGroup(CKMaterial *mat) {
    CKMaterialGroup *group = new CKMaterialGroup(mat);
    m_MaterialGroups.PushBack(group);
    return m_MaterialGroups.Size() - 1;
}

void RCKMesh::DestroyPM() {
    // Match IDA at 0x10025637
    if (!m_ProgressiveMesh)
        return;

    // Delete the progressive mesh data
    delete m_ProgressiveMesh;
    m_ProgressiveMesh = nullptr;

    // Remove the pre-render callback
    RemovePreRenderCallBack((CK_MESHRENDERCALLBACK) ProgressiveMeshPreRenderCallback, this);

    // Rebuild render groups
    CreateRenderGroups();
}

/**
 * @brief Look up parent vertex for progressive mesh at given LOD level.
 *
 * This is a helper function matching IDA sub_10023DD7.
 * Given a vertex index and current LOD vertex count, finds the
 * parent vertex index by traversing the collapse chain.
 *
 * @param pm Progressive mesh data
 * @param vertexIndex Original vertex index
 * @return Parent vertex index at current LOD level
 */
static int PMGetParentVertex(CKProgressiveMesh *pm, int vertexIndex) {
    if (pm->m_VertexCount <= 0)
        return 0;

    while (vertexIndex >= pm->m_VertexCount) {
        vertexIndex = (int) pm->m_Data[vertexIndex];
    }
    return vertexIndex;
}

/**
 * @brief Look up parent vertex with LOD limit for geo-morphing.
 *
 * This is a helper function matching IDA sub_10023E13.
 * Similar to PMGetParentVertex but with an explicit vertex count limit.
 *
 * @param pm Progressive mesh data
 * @param vertexIndex Original vertex index
 * @param vertexCount Maximum vertex count (LOD level)
 * @return Parent vertex index at given LOD level
 */
static int PMGetParentVertexEx(CKProgressiveMesh *pm, int vertexIndex, int vertexCount) {
    if (vertexCount <= 0)
        return 0;

    while (vertexIndex >= vertexCount) {
        vertexIndex = (int) pm->m_Data[vertexIndex];
    }
    return vertexIndex;
}

/**
 * @brief Linear interpolation between two VxVectors.
 *
 * Matches IDA sub_10029720.
 *
 * @param result Output vector
 * @param t Interpolation factor (0-1)
 * @param from Source vector
 * @param to Target vector
 */
static void VxVectorLerp(VxVector *result, float t, const VxVector *from, const VxVector *to) {
    result->x = from->x + (to->x - from->x) * t;
    result->y = from->y + (to->y - from->y) * t;
    result->z = from->z + (to->z - from->z) * t;
}

void RCKMesh::BuildRenderMesh() {
    // Match IDA at 0x100257b1
    // Check if object is up-to-date; if so, nothing to do
    if (m_ObjectFlags & CK_OBJECT_UPTODATE)
        return;

    // Must have progressive mesh data
    if (!m_ProgressiveMesh)
        return;

    // Prepare material groups with proper primitive sizes
    const int faceCount = m_FaceVertexIndices.Size() / 3;
    for (int i = 0; i < m_MaterialGroups.Size(); i++) {
        CKMaterialGroup *group = m_MaterialGroups[i];
        // Resize primitives for face count (IDA: sub_1002A0F0)
        group->m_Primitives.Resize(1);
        CKPrimitiveEntry &prim = group->m_Primitives[0];
        prim.m_Indices.Reserve(3 * faceCount);
        prim.m_Indices.Resize(0);
        prim.m_Type = VX_TRIANGLELIST;
        group->m_FaceIndices.Resize(0);
        group->m_HasValidPrimitives = 0x10000;
        group->m_MaxVertexIndex = 0;
        group->m_BaseVertex = 0;
        group->m_MinVertexIndex = 0;
        // Set vertex count for the group
        group->m_VertexCount = m_Vertices.Size();
    }

    // Allocate temporary face index buffer
    int pmVertexCount = 0;
    XArray<CKWORD> renderIndices;
    renderIndices.Reserve(m_FaceVertexIndices.Size());

    if (m_ProgressiveMesh->m_MorphEnabled) {
        // Geo-morphing enabled - need to interpolate vertices
        m_Valid = FALSE;

        // Determine current and target LOD levels
        int targetVertexCount = m_Vertices.Size();
        int currentCount = targetVertexCount;

        if (m_ProgressiveMesh->m_MorphStep) {
            // With morph step, use additive stepping
            while (currentCount > m_ProgressiveMesh->m_VertexCount) {
                currentCount -= m_ProgressiveMesh->m_MorphStep;
            }
            int nextLod = currentCount + m_ProgressiveMesh->m_MorphStep;
            if (nextLod >= targetVertexCount)
                nextLod = targetVertexCount;
            targetVertexCount = nextLod;
        } else if (m_ProgressiveMesh->m_VertexCount) {
            // Without morph step, use power-of-2 stepping
            while (currentCount >= m_ProgressiveMesh->m_VertexCount) {
                currentCount >>= 1;
            }
            targetVertexCount = currentCount * 2;
        } else {
            currentCount = 0;
            targetVertexCount = 1;
        }

        // Calculate interpolation factor
        float lerpFactor = 0.0f;
        if (targetVertexCount != currentCount) {
            lerpFactor = (float) (m_ProgressiveMesh->m_VertexCount - currentCount) /
                         (float) (targetVertexCount - currentCount);
        }

        // Store target vertex count in reserved field for later use
        m_ProgressiveMesh->m_Reserved = targetVertexCount;

        // Resize interpolated vertex storage
        m_ProgressiveMesh->m_EdgeCollapseData.Resize(m_Vertices.Size() * sizeof(VxVertex) / sizeof(CKDWORD));

        // Track which vertices have been processed
        XArray<int> processed;
        processed.Resize(m_Vertices.Size());
        memset(processed.Begin(), 0, processed.Size() * sizeof(int));

        // Get pointers to face data
        CKWORD *faceIndices = m_FaceVertexIndices.Begin();
        CKFace *faces = m_Faces.Begin();
        VxVertex *interpVertices = (VxVertex *) m_ProgressiveMesh->m_EdgeCollapseData.Begin();

        // Process each face
        for (int faceIdx = 0; faceIdx < m_Faces.Size(); faceIdx++) {
            // Get remapped vertex indices at target LOD
            CKWORD v0 = (CKWORD) PMGetParentVertexEx(m_ProgressiveMesh, faceIndices[0], targetVertexCount);
            CKWORD v1 = (CKWORD) PMGetParentVertexEx(m_ProgressiveMesh, faceIndices[1], targetVertexCount);
            CKWORD v2 = (CKWORD) PMGetParentVertexEx(m_ProgressiveMesh, faceIndices[2], targetVertexCount);

            // Add face index to appropriate material group
            CKWORD faceIdxWord = (CKWORD) faceIdx;
            m_MaterialGroups[faces[faceIdx].m_MatIndex]->m_FaceIndices.PushBack(faceIdxWord);

            // Skip degenerate faces (where two or more vertices collapsed together)
            if (v0 == v1 || v1 == v2 || v2 == v0)
                break;

            // Get vertex indices at current LOD for interpolation
            CKWORD c0 = (CKWORD) PMGetParentVertexEx(m_ProgressiveMesh, faceIndices[0], currentCount);
            CKWORD c1 = (CKWORD) PMGetParentVertexEx(m_ProgressiveMesh, faceIndices[1], currentCount);
            CKWORD c2 = (CKWORD) PMGetParentVertexEx(m_ProgressiveMesh, faceIndices[2], currentCount);

            // Process each vertex of the face
            CKWORD targetVerts[3] = {v0, v1, v2};
            CKWORD currentVerts[3] = {c0, c1, c2};

            for (int vi = 0; vi < 3; vi++) {
                int tv = targetVerts[vi];
                int cv = currentVerts[vi];

                if (!processed[tv]) {
                    processed[tv] = 1;

                    // Interpolate position
                    VxVertex *targetVert = &m_Vertices[tv];
                    VxVertex *currentVert = &m_Vertices[cv];
                    VxVectorLerp(&interpVertices[tv].m_Position, lerpFactor, &currentVert->m_Position, &targetVert->m_Position);

                    // Interpolate normal
                    VxVectorLerp(&interpVertices[tv].m_Normal, lerpFactor, &currentVert->m_Normal, &targetVert->m_Normal);

                    // Interpolate UV coordinates
                    interpVertices[tv].m_UV.x = currentVert->m_UV.x * lerpFactor +
                                                targetVert->m_UV.x * (1.0f - lerpFactor);
                    interpVertices[tv].m_UV.y = currentVert->m_UV.y * lerpFactor +
                                                targetVert->m_UV.y * (1.0f - lerpFactor);
                }
            }

            // Add indices to material group primitives
            CKMaterialGroup *group = m_MaterialGroups[faces[faceIdx].m_MatIndex];
            group->m_Primitives[0].m_Indices.PushBack(v0);
            group->m_Primitives[0].m_Indices.PushBack(v1);
            group->m_Primitives[0].m_Indices.PushBack(v2);

            // Track indices for normal rebuilding
            renderIndices.PushBack(v0);
            renderIndices.PushBack(v1);
            renderIndices.PushBack(v2);

            faceIndices += 3;
        }

        pmVertexCount = renderIndices.Size();
    } else {
        // Geo-morphing disabled - simple vertex remapping
        m_ProgressiveMesh->m_Reserved = m_ProgressiveMesh->m_VertexCount;

        CKWORD *faceIndices = m_FaceVertexIndices.Begin();
        CKFace *faces = m_Faces.Begin();

        for (int faceIdx = 0; faceIdx < m_Faces.Size(); faceIdx++) {
            // Get remapped vertex indices at current LOD
            CKWORD v0 = (CKWORD) PMGetParentVertex(m_ProgressiveMesh, faceIndices[0]);
            CKWORD v1 = (CKWORD) PMGetParentVertex(m_ProgressiveMesh, faceIndices[1]);
            CKWORD v2 = (CKWORD) PMGetParentVertex(m_ProgressiveMesh, faceIndices[2]);

            // Skip degenerate faces
            if (v0 == v1 || v1 == v2 || v2 == v0)
                break;

            // Add face index to material group
            CKWORD faceIdxWord = (CKWORD) faceIdx;
            m_MaterialGroups[faces[faceIdx].m_MatIndex]->m_FaceIndices.PushBack(faceIdxWord);

            // Add indices to material group primitives
            CKMaterialGroup *group = m_MaterialGroups[faces[faceIdx].m_MatIndex];
            group->m_Primitives[0].m_Indices.PushBack(v0);
            group->m_Primitives[0].m_Indices.PushBack(v1);
            group->m_Primitives[0].m_Indices.PushBack(v2);

            // Track for normal rebuilding
            renderIndices.PushBack(v0);
            renderIndices.PushBack(v1);
            renderIndices.PushBack(v2);
            pmVertexCount += 3;

            faceIndices += 3;
        }
    }

    // Rebuild normals if needed (IDA: flag 0x1000000)
    if (pmVertexCount && (m_Flags & VXMESH_PM_BUILDNORM)) {
        g_BuildNormalsFunc(m_Faces.Begin(),
                          renderIndices.Begin(),
                          pmVertexCount / 3,
                          m_Vertices.Begin(),
                          m_ProgressiveMesh->m_Reserved);
        m_Flags &= ~VXMESH_PM_BUILDNORM;
    }

    // Update material channels
    for (int ch = 0; ch < m_MaterialChannels.Size(); ch++) {
        VxMaterialChannel &channel = m_MaterialChannels[ch];
        if (!channel.m_FaceIndices) {
            channel.m_FaceIndices = new XArray<CKWORD>();
        }
        channel.m_FaceIndices->Resize(0);

        for (int i = 0; i < m_MaterialGroups.Size(); i++) {
            CKMaterialGroup *group = m_MaterialGroups[i];
            XArray<CKWORD> &primIndices = group->m_Primitives[0].m_Indices;
            for (CKWORD *it = primIndices.Begin(); it != primIndices.End(); ++it) {
                channel.m_FaceIndices->PushBack(*it);
            }
        }
    }

    // Clean up empty material groups (IDA: sub_1002A380)
    // Keep group 0, remove others that have no faces
    int groupIdx = 1;
    while (groupIdx < m_MaterialGroups.Size()) {
        CKMaterialGroup *group = m_MaterialGroups[groupIdx];
        if (!group->m_Material && group->m_FaceIndices.Size() == 0) {
            delete group;
            m_MaterialGroups.RemoveAt(groupIdx);

            // Update face material indices
            for (int fi = 0; fi < m_Faces.Size(); fi++) {
                if (m_Faces[fi].m_MatIndex > groupIdx)
                    m_Faces[fi].m_MatIndex--;
            }
        } else {
            groupIdx++;
        }
    }

    // Mark mesh as modified
    m_FaceChannelMask = 0;
    m_Flags |= VXMESH_POS_CHANGED;
    CKObject::ModifyObjectFlags(CK_OBJECT_UPTODATE, 0);
    VertexMove();
}

// =============================================
// Missing virtual method implementations
// =============================================

CKBYTE *RCKMesh::GetModifierVertices(CKDWORD *Stride) {
    // Match IDA at 0x1002a7a0
    return (CKBYTE *) GetPositionsPtr(Stride);
}

int RCKMesh::GetModifierVertexCount() {
    // Match IDA at 0x1002a7c0
    if (m_ProgressiveMesh) {
        return m_ProgressiveMesh->m_VertexCount;
    }
    return m_Vertices.Size();
}

void RCKMesh::ModifierVertexMove(CKBOOL RebuildNormals, CKBOOL RebuildFaceNormals) {
    // Match IDA at 0x1001e0a5
    if (GetLitMode()) {
        if (RebuildNormals) {
            BuildNormals();
        } else if (RebuildFaceNormals) {
            BuildFaceNormals();
        }
    } else if (RebuildFaceNormals || RebuildNormals) {
        BuildFaceNormals();
    }
    VertexMove();
}

CKBYTE *RCKMesh::GetModifierUVs(CKDWORD *Stride, int channel) {
    // Match IDA at 0x1002a800
    return (CKBYTE *) GetTextureCoordinatesPtr(Stride, channel);
}

int RCKMesh::GetModifierUVCount(int channel) {
    // Match IDA at 0x1002a830
    return GetModifierVertexCount();
}

void RCKMesh::ModifierUVMove() {
    UVChanged();
}

// Match IDA at 0x1001c16a
void RCKMesh::SetVertexSpecularColor(int Index, CKDWORD Color) {
    if (Index >= 0 && Index < m_VertexColors.Size()) {
        m_VertexColors[Index].Specular = Color;
        ColorChanged();
    }
}

// Match IDA at 0x1001c1b8
CKDWORD RCKMesh::GetVertexSpecularColor(int Index) {
    if (Index >= 0 && Index < m_VertexColors.Size()) {
        return m_VertexColors[Index].Specular;
    }
    return 0;
}

// Match IDA at 0x1001e865
void RCKMesh::TranslateVertices(VxVector *Vector) {
    CKDWORD stride;
    int vertexCount = GetModifierVertexCount();
    if (vertexCount <= 0)
        return;

    VxVector *positions = (VxVector *) GetModifierVertices(&stride);
    if (!positions)
        return;

    for (int i = 0; i < vertexCount; ++i) {
        *positions += *Vector;
        positions = (VxVector *) ((CKBYTE *) positions + stride);
    }
    ModifierVertexMove(TRUE, FALSE);
}

// Match IDA at 0x1001e4ac
void RCKMesh::ScaleVertices(VxVector *Vector, VxVector *Pivot) {
    CKDWORD stride;
    int vertexCount = GetModifierVertexCount();
    if (vertexCount <= 0)
        return;

    float *positions = (float *) GetModifierVertices(&stride);
    if (!positions)
        return;

    CKDWORD normalStride;
    VxVector *normals = (VxVector *) GetNormalsPtr(&normalStride);

    // CKPatchMesh (ClassID 53) doesn't scale normals
    if (GetClassID() != CKCID_PATCHMESH && normals) {
        if (Pivot) {
            for (int i = 0; i < vertexCount; ++i) {
                positions[0] = (positions[0] - Pivot->x) * Vector->x + Pivot->x;
                positions[1] = (positions[1] - Pivot->y) * Vector->y + Pivot->y;
                positions[2] = (positions[2] - Pivot->z) * Vector->z + Pivot->z;
                normals->x *= Vector->x;
                normals->y *= Vector->y;
                normals->z *= Vector->z;
                normals->Normalize();
                positions = (float *) ((CKBYTE *) positions + stride);
                normals = (VxVector *) ((CKBYTE *) normals + normalStride);
            }
        } else {
            for (int i = 0; i < vertexCount; ++i) {
                positions[0] *= Vector->x;
                positions[1] *= Vector->y;
                positions[2] *= Vector->z;
                normals->x *= Vector->x;
                normals->y *= Vector->y;
                normals->z *= Vector->z;
                normals->Normalize();
                positions = (float *) ((CKBYTE *) positions + stride);
                normals = (VxVector *) ((CKBYTE *) normals + normalStride);
            }
        }
        ModifierVertexMove(FALSE, TRUE);
    } else {
        if (Pivot) {
            for (int i = 0; i < vertexCount; ++i) {
                positions[0] = (positions[0] - Pivot->x) * Vector->x + Pivot->x;
                positions[1] = (positions[1] - Pivot->y) * Vector->y + Pivot->y;
                positions[2] = (positions[2] - Pivot->z) * Vector->z + Pivot->z;
                positions = (float *) ((CKBYTE *) positions + stride);
            }
        } else {
            for (int i = 0; i < vertexCount; ++i) {
                positions[0] *= Vector->x;
                positions[1] *= Vector->y;
                positions[2] *= Vector->z;
                positions = (float *) ((CKBYTE *) positions + stride);
            }
        }
        ModifierVertexMove(TRUE, TRUE);
    }
}

// Match IDA at 0x1001e792
void RCKMesh::ScaleVertices3f(float X, float Y, float Z, VxVector *Pivot) {
    VxVector scale(X, Y, Z);
    ScaleVertices(&scale, Pivot);
}

// Match IDA at 0x1001e7c8
void RCKMesh::RotateVertices(VxVector *Vector, float Angle) {
    CKDWORD stride;
    int vertexCount = GetModifierVertexCount();
    if (vertexCount <= 0)
        return;

    VxVector *positions = (VxVector *) GetModifierVertices(&stride);
    VxMatrix rot;
    Vx3DMatrixFromRotation(rot, *Vector, Angle);

    for (int i = 0; i < vertexCount; ++i) {
        Vx3DRotateVector(positions, rot, positions);
        positions = (VxVector *) ((CKBYTE *) positions + stride);
    }
    ModifierVertexMove(TRUE, FALSE);
}

// Match IDA at 0x1001cd16 - unsigned comparison, returns axis0() on error
const VxVector &RCKMesh::GetFaceNormal(int Index) {
    if (Index < m_Faces.Size()) {
        return m_Faces[Index].m_Normal;
    }
    return VxVector::axis0();
}

// Match IDA at 0x1001ccf6 - no bounds check in original
CKWORD RCKMesh::GetFaceChannelMask(int FaceIndex) {
    return m_Faces[FaceIndex].m_ChannelMask;
}

// Match IDA at 0x1001c7b7 - no bounds checking in original
VxVector &RCKMesh::GetFaceVertex(int FaceIndex, int VIndex) {
    int vertexIndex = m_FaceVertexIndices[FaceIndex * 3 + VIndex];
    return m_Vertices[vertexIndex].m_Position;
}

CKBYTE *RCKMesh::GetFaceNormalsPtr(CKDWORD *Stride) {
    if (Stride) *Stride = sizeof(CKFace);
    if (m_Faces.Size() > 0) {
        return (CKBYTE *) &m_Faces[0].m_Normal;
    }
    return nullptr;
}

// Match IDA at 0x1001c958
void RCKMesh::SetFaceMaterialEx(int *FaceIndices, int FaceCount, CKMaterial *Mat) {
    CKWORD matGroupIndex = (CKWORD) GetMaterialGroupIndex(Mat, TRUE);
    for (int i = 0; i < FaceCount; ++i) {
        m_Faces[FaceIndices[i]].m_MatIndex = matGroupIndex;
    }
    UnOptimize();
}

// Match IDA at 0x1001cbee
void RCKMesh::SetFaceChannelMask(int FaceIndex, CKWORD ChannelMask) {
    CKWORD maskDelta = m_Faces[FaceIndex].m_ChannelMask ^ ChannelMask;
    m_Faces[FaceIndex].m_ChannelMask = ChannelMask;
    m_FaceChannelMask |= maskDelta;
}

// Match IDA at 0x1001c9bf
void RCKMesh::ReplaceMaterial(CKMaterial *oldMat, CKMaterial *newMat) {
    if (oldMat == newMat)
        return;

    const int oldGroupIndex = GetMaterialGroupIndex(oldMat, FALSE);
    if (oldGroupIndex < 0)
        return;

    if (oldGroupIndex == 0) {
        // oldMat is at index 0, just update faces with index 0 to newMat's index
        CKWORD newGroupIndex = (CKWORD) GetMaterialGroupIndex(newMat, TRUE);
        for (int j = 0; j < m_Faces.Size(); ++j) {
            if (m_Faces[j].m_MatIndex == 0)
                m_Faces[j].m_MatIndex = newGroupIndex;
        }
        UnOptimize();
    } else if (newMat) {
        // Just update the material pointer in the group
        m_MaterialGroups[oldGroupIndex]->m_Material = (RCKMaterial *) newMat;
    } else {
        // newMat is null, reassign faces and remove the group
        for (int i = 0; i < m_Faces.Size(); ++i) {
            if (m_Faces[i].m_MatIndex == oldGroupIndex) {
                m_Faces[i].m_MatIndex = 0;
            } else if (m_Faces[i].m_MatIndex > oldGroupIndex) {
                --m_Faces[i].m_MatIndex;
            }
        }
        CKMaterialGroup *oldGroup = m_MaterialGroups[oldGroupIndex];
        if (oldGroup)
            delete oldGroup;
        m_MaterialGroups.RemoveAt(oldGroupIndex);
        UnOptimize();
    }
}

// Match IDA at 0x1001cc4e
void RCKMesh::ChangeFaceChannelMask(int FaceIndex, CKWORD AddChannelMask, CKWORD RemoveChannelMask) {
    CKWORD updatedMask = ~RemoveChannelMask & (AddChannelMask | m_Faces[FaceIndex].m_ChannelMask);
    CKWORD maskDelta = m_Faces[FaceIndex].m_ChannelMask ^ updatedMask;
    m_Faces[FaceIndex].m_ChannelMask = updatedMask;
    m_FaceChannelMask |= maskDelta;
}

// Match IDA at 0x1001cb8a
void RCKMesh::ApplyGlobalMaterial(CKMaterial *Mat) {
    CKWORD matGroupIndex = (CKWORD) GetMaterialGroupIndex(Mat, TRUE);
    for (CKFace *i = m_Faces.Begin(); i < m_Faces.End(); ++i)
        i->m_MatIndex = matGroupIndex;
    UnOptimize();
}

// Match IDA at 0x1001e8e4
void RCKMesh::DissociateAllFaces() {
    // Create temporary copies of current vertex and color data
    XArray<VxVertex> tempVertices;
    XArray<VxColors> tempColors;
    tempVertices = m_Vertices;
    tempColors = m_VertexColors;

    // Get total number of face vertex indices (faces * 3)
    int indexCount = m_FaceVertexIndices.Size();

    // Resize vertex array to hold one vertex per face index
    SetVertexCount(indexCount);

    // If weights exist, duplicate them for each vertex
    if (m_VertexWeights) {
        XArray<float> tempWeights;
        tempWeights.Resize(indexCount);
        for (int i = 0; i < indexCount; ++i) {
            CKWORD oldIndex = m_FaceVertexIndices[i];
            tempWeights[i] = (*m_VertexWeights)[oldIndex];
        }
        *m_VertexWeights = tempWeights;
    }

    // Copy vertex and color data from original locations to new sequential locations
    for (int j = 0; j < indexCount; ++j) {
        CKWORD oldIndex = m_FaceVertexIndices[j];
        m_Vertices[j] = tempVertices[oldIndex];
        m_VertexColors[j] = tempColors[oldIndex];
        // Set each index to its sequential value
        m_FaceVertexIndices[j] = (CKWORD) j;
    }

    UnOptimize();
}

void RCKMesh::CreateLineStrip(int StartingLine, int Count, int StartingVertexIndex) {
    if (Count <= 0) return;
    for (int i = 0; i < Count; i++) {
        SetLine(StartingLine + i, StartingVertexIndex + i, StartingVertexIndex + i + 1);
    }
}

int RCKMesh::GetChannelCount() {
    return m_MaterialChannels.Size();
}

void RCKMesh::RemoveChannelByMaterial(CKMaterial *Mat) {
    int index = GetChannelByMaterial(Mat);
    if (index >= 0) {
        RemoveChannel(index);
    }
}

// Match IDA at 0x1001d31e
void RCKMesh::ActivateChannel(int Index, CKBOOL Active) {
    if (Index < m_MaterialChannels.Size()) {
        if (Active) {
            m_MaterialChannels[Index].m_Flags |= VXCHANNEL_ACTIVE;
        } else {
            m_MaterialChannels[Index].m_Flags &= ~VXCHANNEL_ACTIVE;
        }
    }
}

// Match IDA at 0x1001d418
CKBOOL RCKMesh::IsChannelActive(int Index) {
    if (Index >= m_MaterialChannels.Size())
        return FALSE;
    return (m_MaterialChannels[Index].m_Flags & VXCHANNEL_ACTIVE) != 0;
}

// Match IDA at 0x1001d391
void RCKMesh::ActivateAllChannels(CKBOOL Active) {
    for (int i = 0; i < m_MaterialChannels.Size(); ++i) {
        if (Active) {
            m_MaterialChannels[i].m_Flags |= VXCHANNEL_ACTIVE;
        } else {
            m_MaterialChannels[i].m_Flags &= ~VXCHANNEL_ACTIVE;
        }
    }
}

// Match IDA at 0x1001d454
void RCKMesh::LitChannel(int Index, CKBOOL Lit) {
    if (Index < m_MaterialChannels.Size()) {
        if (Lit) {
            // Lit = TRUE means unlit mode off (clear bit)
            m_MaterialChannels[Index].m_Flags &= ~VXCHANNEL_NOTLIT;
        } else {
            // Lit = FALSE means unlit mode on (set bit)
            m_MaterialChannels[Index].m_Flags |= VXCHANNEL_NOTLIT;
        }
    }
}

// Match IDA at 0x1001d4cd
CKBOOL RCKMesh::IsChannelLit(int Index) {
    return Index < m_MaterialChannels.Size() && !(m_MaterialChannels[Index].m_Flags & VXCHANNEL_NOTLIT);
}

// Match IDA at 0x1001d510
CKDWORD RCKMesh::GetChannelFlags(int Index) {
    if (Index >= m_MaterialChannels.Size())
        return 0;
    return m_MaterialChannels[Index].m_Flags;
}

// Match IDA at 0x1001d549
void RCKMesh::SetChannelFlags(int Index, CKDWORD Flags) {
    if (Index < m_MaterialChannels.Size()) {
        if (Flags & VXCHANNEL_SAMEUV) {
            // Delete UV array if using same UV
            delete[] m_MaterialChannels[Index].m_UVs;
            m_MaterialChannels[Index].m_UVs = nullptr;
        } else if (!m_MaterialChannels[Index].m_UVs && m_Vertices.Size()) {
            // Allocate UV array if not already allocated and has vertices
            int vertexCount = m_Vertices.Size();
            Vx2DVector *uvArray = new Vx2DVector[vertexCount];
            if (uvArray) {
                for (int i = 0; i < vertexCount; ++i) {
                    uvArray[i].x = 0.0f;
                    uvArray[i].y = 0.0f;
                }
            }
            m_MaterialChannels[Index].m_UVs = uvArray;
        }
        m_MaterialChannels[Index].m_Flags = Flags;
    }
}

CKMaterial *RCKMesh::GetChannelMaterial(int Index) {
    // Match IDA at 0x1001d731 - uses unsigned comparison
    if (Index >= m_MaterialChannels.Size())
        return nullptr;
    return m_MaterialChannels[Index].m_Material;
}

// Match IDA at 0x1001d7a2
VXBLEND_MODE RCKMesh::GetChannelSourceBlend(int Index) {
    if (Index >= m_MaterialChannels.Size())
        return VXBLEND_ZERO;
    return m_MaterialChannels[Index].m_SourceBlend;
}

// Match IDA at 0x1001d816
VXBLEND_MODE RCKMesh::GetChannelDestBlend(int Index) {
    if (Index >= m_MaterialChannels.Size())
        return VXBLEND_ZERO;
    return m_MaterialChannels[Index].m_DestBlend;
}

// Match IDA at 0x1001d6f9
void RCKMesh::SetChannelMaterial(int Index, CKMaterial *Mat) {
    if (Index < m_MaterialChannels.Size()) {
        m_MaterialChannels[Index].m_Material = Mat;
    }
}

// Match IDA at 0x1001d76a
void RCKMesh::SetChannelSourceBlend(int Index, VXBLEND_MODE BlendMode) {
    if (Index < m_MaterialChannels.Size()) {
        m_MaterialChannels[Index].m_SourceBlend = BlendMode;
    }
}

// Match IDA at 0x1001d7de
void RCKMesh::SetChannelDestBlend(int Index, VXBLEND_MODE BlendMode) {
    if (Index < m_MaterialChannels.Size()) {
        m_MaterialChannels[Index].m_DestBlend = BlendMode;
    }
}

CKERROR RCKMesh::Render(CKRenderContext *Dev, CK3dEntity *Mov) {
    // Match IDA at 0x1001d852
    RCKRenderContext *rc = (RCKRenderContext *) Dev;
    RCK3dEntity *ent = (RCK3dEntity *) Mov;

    // Check rasterizer context
    if (!rc->m_RasterizerContext)
        return CKERR_INVALIDRENDERCONTEXT;

    // Compute box visibility if entity is different from current
    if (ent != rc->m_Current3dEntity) {
        VxRect extents;
        const VxBbox &box = GetLocalBox();
        rc->m_RasterizerContext->ComputeBoxVisibility(box, FALSE, &extents);
        rc->AddExtents2D(extents, nullptr);
    }

    // Handle render callbacks
    if (m_RenderCallbacks) {
        // Pre-render callbacks - m_PreCallBacks is at offset 0 of CKCallbacksContainer
        // sub_1002C220 returns (End - Begin) / 12, i.e. element count
        int preCount = m_RenderCallbacks->m_PreCallBacks.Size();
        if (preCount > 0) {
            rc->m_RasterizerContext->SetVertexShader(0);
            rc->m_ObjectsCallbacksTimeProfiler.Reset();

            // Iterate with 12-byte stride (sizeof VxCallBack)
            for (VxCallBack *it = m_RenderCallbacks->m_PreCallBacks.Begin();
                 it != m_RenderCallbacks->m_PreCallBacks.End(); it++) {
                CK_MESHRENDERCALLBACK callback = (CK_MESHRENDERCALLBACK) it->callback;
                void *argument = it->argument;
                callback(Dev, Mov, this, argument);
            }

            rc->m_Stats.ObjectsCallbacksTime = rc->m_ObjectsCallbacksTimeProfiler.Current()
                + rc->m_Stats.ObjectsCallbacksTime;
        }

        // Main callback or default render
        if (m_RenderCallbacks->m_Callback) {
            ((CK_MESHRENDERCALLBACK) m_RenderCallbacks->m_Callback->callback)(
                Dev, Mov, this, m_RenderCallbacks->m_Callback->argument);
        } else {
            DefaultRender(rc, ent);
        }

        // Post-render callbacks
        int postCount = m_RenderCallbacks->m_PostCallBacks.Size();
        if (postCount > 0) {
            rc->m_RasterizerContext->SetVertexShader(0);
            rc->m_ObjectsCallbacksTimeProfiler.Reset();

            for (VxCallBack *it = m_RenderCallbacks->m_PostCallBacks.Begin();
                 it != m_RenderCallbacks->m_PostCallBacks.End(); it++) {
                CK_MESHRENDERCALLBACK callback = (CK_MESHRENDERCALLBACK) it->callback;
                void *argument = it->argument;
                callback(Dev, Mov, this, argument);
            }

            rc->m_Stats.ObjectsCallbacksTime = rc->m_ObjectsCallbacksTimeProfiler.Current() + rc->m_Stats.ObjectsCallbacksTime;
        }
    } else {
        DefaultRender(rc, ent);
    }

    return 0;
}

float *RCKMesh::GetVertexWeightsPtr() {
    return m_VertexWeights ? m_VertexWeights->Begin() : nullptr;
}

void RCKMesh::LoadVertices(CKStateChunk *chunk) {
    CKDWORD loadFlags = 0;
    const int result = ILoadVertices(chunk, &loadFlags);
    if (result)
        Load(chunk, nullptr);
}

/**
 * CreatePM - Create Progressive Mesh data
 *
 * Based on IDA decompilation at 0x10024777 (3748 bytes)
 *
 * This function implements a progressive mesh algorithm that reorders vertices
 * by importance for LOD (Level of Detail) rendering. The algorithm:
 * 1. Builds vertex/face connectivity using PMVertexEx and PMFace structures
 * 2. Computes edge collapse costs based on geometric error (distance * normal preservation)
 * 3. Uses a min-heap to extract vertices in order of increasing collapse cost
 * 4. Records the collapse order for progressive reconstruction
 * 5. Reorders mesh vertices so less important vertices come last
 * 6. Updates face indices, material channel UVs, skins, and morph controllers
 */

// Helper: Calculate edge collapse cost between two vertices
// Based on IDA at sub_100238F3
static float CalculateEdgeCollapseCost(PMVertexEx *v1, PMVertexEx *v2) {
    // Distance between vertices
    const float dx = v2->position.x - v1->position.x;
    const float dy = v2->position.y - v1->position.y;
    const float dz = v2->position.z - v1->position.z;
    const float distance = sqrtf(dx * dx + dy * dy + dz * dz);

    // Build list of faces that contain both v1 and v2
    XArray<PMFace *> sharedFaces;
    sharedFaces.Reserve(v1->faces.Size());
    for (PMFace **fit = v1->faces.Begin(); fit != v1->faces.End(); ++fit) {
        PMFace *f = *fit;
        if (f && f->ContainsVertex(v2)) {
            sharedFaces.PushBack(f);
        }
    }

    // Curvature factor
    float curvature = 0.001f;
    for (PMFace **fit = v1->faces.Begin(); fit != v1->faces.End(); ++fit) {
        PMFace *f = *fit;
        if (!f)
            continue;

        float minCurv = 1.0f;
        for (PMFace **sfit = sharedFaces.Begin(); sfit != sharedFaces.End(); ++sfit) {
            PMFace *sf = *sfit;
            if (!sf)
                continue;

            const float dot = f->normal.x * sf->normal.x + f->normal.y * sf->normal.y + f->normal.z * sf->normal.z;
            const float value = (1.002f - dot) * 0.5f;
            if (value < minCurv)
                minCurv = value;
        }

        if (minCurv > curvature)
            curvature = minCurv;
    }

    // Boundary detection: if any neighbor shares only one face with v1, force curvature=1.0
    for (PMVertexEx **nit = v1->neighbors.Begin(); nit != v1->neighbors.End(); ++nit) {
        PMVertexEx *n = *nit;
        if (!n)
            continue;
        int shared = 0;
        for (PMFace **fit = v1->faces.Begin(); fit != v1->faces.End(); ++fit) {
            PMFace *f = *fit;
            if (f && f->ContainsVertex(n))
                ++shared;
        }
        if (shared == 1) {
            curvature = 1.0f;
            break;
        }
    }

    return distance * curvature;
}

// Helper: Calculate minimum collapse cost for a vertex
// Based on IDA at sub_10023AB6
static void CalculateMinCollapseCost(PMVertexEx *v) {
    if (v->neighbors.Size() == 0) {
        v->collapseTarget = nullptr;
        v->collapseCost = -0.01f;
        return;
    }

    v->collapseCost = 1000000.0f;
    v->collapseTarget = nullptr;

    for (PMVertexEx **nit = v->neighbors.Begin(); nit != v->neighbors.End(); ++nit) {
        PMVertexEx *neighbor = *nit;
        if (!neighbor)
            continue;

        const float cost = CalculateEdgeCollapseCost(v, neighbor);
        if (!v->collapseTarget || cost < v->collapseCost) {
            v->collapseTarget = neighbor;
            v->collapseCost = cost;
        }
    }
}

static void RemoveFaceFromVertex(PMVertexEx *v, PMFace *f) {
    if (!v || !f)
        return;
    for (int i = 0; i < v->faces.Size(); ++i) {
        if (v->faces[i] == f) {
            v->faces.RemoveAt(i);
            return;
        }
    }
}

// IDA: sub_100237B5 (replace old vertex by new vertex in a face, update adjacency, recompute normal)
static void ProgressiveMeshReplaceVertexInFace(PMFace *face, PMVertexEx *oldV, PMVertexEx *newV) {
    if (!face || !oldV || !newV)
        return;

    face->ReplaceVertex(oldV, newV);

    // Remove face from old vertex, add to new vertex
    RemoveFaceFromVertex(oldV, face);
    newV->faces.PushBack(face);

    // Clean neighbor links involving oldV
    for (int i = 0; i < 3; ++i) {
        PMVertexEx *v = face->vertices[i];
        PMVertexEx::RemoveIfNonNeighbor(oldV, v);
        PMVertexEx::RemoveIfNonNeighbor(v, oldV);
    }

    // Ensure triangle vertices are all mutual neighbors
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j)
                continue;
            PMVertexEx *a = face->vertices[i];
            PMVertexEx *b = face->vertices[j];
            if (a && b && !a->HasNeighbor(b))
                a->neighbors.PushBack(b);
        }
    }

    face->ComputeNormal();
}

// IDA: sub_10023BE7 (collapse vertex oldV to targetV; if targetV==nullptr just delete oldV)
static void ProgressiveMeshCollapseVertex(PMEdgeCollapseData *pmData, PMVertexEx *oldV, PMVertexEx *targetV) {
    if (!pmData || !oldV)
        return;

    if (!targetV) {
        pmData->RemoveVertex(oldV);
        return;
    }

    XArray<PMVertexEx *> affected;
    affected.Reserve(oldV->neighbors.Size());
    for (PMVertexEx **it = oldV->neighbors.Begin(); it != oldV->neighbors.End(); ++it) {
        affected.PushBack(*it);
    }

    // Remove degenerate faces (those containing targetV)
    for (int i = oldV->faces.Size() - 1; i >= 0; --i) {
        PMFace *f = oldV->faces[i];
        if (f && f->ContainsVertex(targetV)) {
            pmData->DeleteFace(f);
        }
    }

    // Replace oldV by targetV in remaining faces
    for (int i = oldV->faces.Size() - 1; i >= 0; --i) {
        PMFace *f = oldV->faces[i];
        if (f)
            ProgressiveMeshReplaceVertexInFace(f, oldV, targetV);
    }

    pmData->RemoveVertex(oldV);

    // Recompute costs & update heap for affected vertices
    for (PMVertexEx **it = affected.Begin(); it != affected.End(); ++it) {
        PMVertexEx *v = *it;
        if (!v)
            continue;
        CalculateMinCollapseCost(v);
        if (v->heapIndex >= 0) {
            pmData->HeapifyUp(v->heapIndex);
            pmData->HeapifyDown(v->heapIndex);
        }
    }
}

// IDA: sub_10023E13 + CompareFunction for face ordering
static int ProgressiveMeshRemapIndexWhileGE(const CKProgressiveMesh *pm, int idx, int threshold) {
    if (!pm)
        return 0;
    if (threshold <= 0)
        return 0;
    while (idx >= threshold) {
        idx = (int) pm->m_Data[idx];
    }
    return idx;
}

static int ProgressiveMeshFaceDegeneracySteps(const CKProgressiveMesh *pm, CKWORD i0, CKWORD i1, CKWORD i2) {
    if (!pm)
        return 0;
    int count = 0;
    int v = pm->m_VertexCount;
    int a = (int) i0;
    int b = (int) i1;
    int c = (int) i2;
    while (a != b && a != c && b != c) {
        if (a >= --v)
            a = ProgressiveMeshRemapIndexWhileGE(pm, a, v);
        if (b >= v)
            b = ProgressiveMeshRemapIndexWhileGE(pm, b, v);
        if (c >= v)
            c = ProgressiveMeshRemapIndexWhileGE(pm, c, v);
        ++count;
    }
    return count;
}

static int ProgressiveMeshCompareFaces(const CKProgressiveMesh *pm, const CKWORD *aIdx, const CKWORD *bIdx) {
    const int aSteps = ProgressiveMeshFaceDegeneracySteps(pm, aIdx[0], aIdx[1], aIdx[2]);
    const int bSteps = ProgressiveMeshFaceDegeneracySteps(pm, bIdx[0], bIdx[1], bIdx[2]);
    return bSteps - aSteps;
}

// IDA: global dword_10090CBC + CompareFunction (0x10023E4B) used by sub_1002404E.
static const CKProgressiveMesh *g_ProgressiveMeshCompareContext = nullptr;

struct ProgressiveMeshFaceSortEntry {
    CKWORD idx[3];
    CKFace face;
};

static int ProgressiveMeshCompareFunctionWrapper(const void *lhs, const void *rhs) {
    const auto *a = static_cast<const ProgressiveMeshFaceSortEntry *>(lhs);
    const auto *b = static_cast<const ProgressiveMeshFaceSortEntry *>(rhs);
    return ProgressiveMeshCompareFaces(g_ProgressiveMeshCompareContext, a->idx, b->idx);
}

CKERROR RCKMesh::CreatePM() {
    // Match IDA at 0x100247a6: Check for PATCHMESH
    if (CKIsChildClassOf(this, CKCID_PATCHMESH))
        return CKERR_INVALIDPARAMETER;

    // Match IDA at 0x100247c0: Check if already has PM
    if (m_ProgressiveMesh)
        return CKERR_INVALIDPARAMETER;

    // Match IDA at 0x100247df: Consolidate geometry first
    Consolidate();

    int vertexCount = GetVertexCount();
    int faceCount = GetFaceCount();

    // Match IDA at 0x100247ef: Create CKProgressiveMesh (44 bytes)
    // Constructor sets m_VertexCount = -1, m_MorphEnabled = 0, m_MorphStep = 0
    m_ProgressiveMesh = new CKProgressiveMesh();

    // Match IDA at 0x10024863: Add pre-render callback
    AddPreRenderCallBack((CK_MESHRENDERCALLBACK) ProgressiveMeshPreRenderCallback, this, FALSE);

    // Match IDA at 0x10024873: Create PMEdgeCollapseData (36 bytes)
    PMEdgeCollapseData *pmData = new PMEdgeCollapseData();
    m_ProgressiveMesh->m_EdgeCollapseStruct = pmData;

    // Match IDA at 0x100248fb-0x10024911: Reserve vertex array
    pmData->vertices.Reserve(vertexCount);

    // Match IDA at 0x10024916-0x100249d1: Create PMVertexEx for each mesh vertex
    for (int i = 0; i < vertexCount; i++) {
        VxVector pos;
        GetVertexPosition(i, &pos);

        PMVertexEx *v = new PMVertexEx(pos, i);
        v->arrayIndex = pmData->vertices.Size();
        pmData->vertices.PushBack(v);
    }

    // Match IDA at 0x100249e9-0x10024a02: Reserve face array
    pmData->faces.Reserve(faceCount);

    // Match IDA at 0x10024a07-0x10024b90: Create PMFace for each mesh face
    for (int i = 0; i < faceCount; i++) {
        int v0, v1, v2;
        GetFaceVertexIndex(i, v0, v1, v2);

        // Get face normal (returns const VxVector&)
        const VxVector &normal = GetFaceNormal(i);

        // Get face material (optional)
        CKMaterial *mat = GetFaceMaterial(i);

        // Create face with vertices from vertex array
        PMFace *f = new PMFace(
            pmData->vertices[v0],
            pmData->vertices[v1],
            pmData->vertices[v2],
            normal,
            mat
        );
        f->arrayIndex = pmData->faces.Size();
        pmData->faces.PushBack(f);

        // Link vertices to face and establish neighbor relationships
        // Based on IDA at sub_10023577
        PMVertexEx *pv0 = pmData->vertices[v0];
        PMVertexEx *pv1 = pmData->vertices[v1];
        PMVertexEx *pv2 = pmData->vertices[v2];

        pv0->faces.PushBack(f);
        pv1->faces.PushBack(f);
        pv2->faces.PushBack(f);

        pv0->AddNeighbor(pv1);
        pv0->AddNeighbor(pv2);
        pv1->AddNeighbor(pv0);
        pv1->AddNeighbor(pv2);
        pv2->AddNeighbor(pv0);
        pv2->AddNeighbor(pv1);
    }

    // Match IDA at 0x10024ba6: Calculate initial collapse costs for all vertices
    // and add them to the min-heap (sub_10023B7F)
    for (int i = 0; i < vertexCount; i++) {
        PMVertexEx *v = pmData->vertices[i];
        CalculateMinCollapseCost(v);
        pmData->AddToHeap(v);
    }

    // Match IDA at 0x10024bb0-0x10024bd1: Create collapse order tracking array
    XArray<int> collapseOrder; // Maps original vertex index -> new reordered position
    collapseOrder.Resize(vertexCount);

    // Match IDA at 0x10024bfa: Resize parent vertex array
    m_ProgressiveMesh->m_Data.Resize(vertexCount);

    // Match IDA at 0x10024c15-0x10024cc8: Main edge collapse loop
    while (pmData->vertices.Size() > 0) {
        PMVertexEx *v = pmData->PopMinFromHeap();
        if (!v)
            break;

        const int newPosition = pmData->vertices.Size() - 1;
        collapseOrder[v->originalIndex] = newPosition;

        int parentOrigIndex = -1;
        if (v->collapseTarget)
            parentOrigIndex = v->collapseTarget->originalIndex;

        m_ProgressiveMesh->m_Data[pmData->vertices.Size() - 1] = (CKDWORD) parentOrigIndex;
        ProgressiveMeshCollapseVertex(pmData, v, v->collapseTarget);
    }

    // Match IDA at 0x10024cd2-0x10024d6f: Remap parent indices through collapse order
    for (int i = 0; i < vertexCount; i++) {
        int parentOrigIndex = (int) m_ProgressiveMesh->m_Data[i];
        if (parentOrigIndex == -1) {
            m_ProgressiveMesh->m_Data[i] = 0;
        } else {
            m_ProgressiveMesh->m_Data[i] = (CKDWORD) collapseOrder[parentOrigIndex];
        }
    }

    // Match IDA at 0x10024d8a-0x10024ebc: Save original vertex data
    XArray<VxVertex> originalVertices;
    originalVertices.Resize(vertexCount);
    XArray<CKDWORD> originalColors;
    originalColors.Resize(vertexCount * 2); // Color + Specular

    for (int i = 0; i < vertexCount; i++) {
        GetVertexPosition(i, (VxVector *) &originalVertices[i]);
        GetVertexNormal(i, &originalVertices[i].m_Normal);
        GetVertexTextureCoordinates(i, &originalVertices[i].m_UV.x, &originalVertices[i].m_UV.y, -1);
        originalColors[i * 2] = GetVertexColor(i);
        originalColors[i * 2 + 1] = GetVertexSpecularColor(i);
    }

    // Match IDA at 0x10024ebc-0x10024fbb: Reorder vertices according to collapse order
    for (int i = 0; i < vertexCount; i++) {
        int newPos = collapseOrder[i];
        SetVertexPosition(newPos, (VxVector *) &originalVertices[i]);
        SetVertexNormal(newPos, &originalVertices[i].m_Normal);
        SetVertexTextureCoordinates(newPos, originalVertices[i].m_UV.x, originalVertices[i].m_UV.y, -1);
        SetVertexColor(newPos, originalColors[i * 2]);
        SetVertexSpecularColor(newPos, originalColors[i * 2 + 1]);
    }

    // Match IDA at 0x10024fda-0x1002505a: Update face vertex indices
    int finalFaceCount = GetFaceCount();
    for (int f = 0; f < finalFaceCount; f++) {
        int v0, v1, v2;
        GetFaceVertexIndex(f, v0, v1, v2);
        SetFaceVertexIndex(f, collapseOrder[v0], collapseOrder[v1], collapseOrder[v2]);
    }

    // Match IDA at 0x10025074: Set initial vertices rendered
    SetVerticesRendered(vertexCount);

    // Match IDA at 0x1002508d: Store vertex count in PM structure
    m_ProgressiveMesh->m_VertexCount = vertexCount;

    // Match IDA at 0x1002508d: sub_1002404E(this->m_ProgressiveMesh, this)
    // Sort faces using CompareFunction (0x10023E4B) with a global PM context (dword_10090CBC).
    {
        XArray<ProgressiveMeshFaceSortEntry> tmp;
        tmp.Resize(finalFaceCount);

        for (int i = 0; i < finalFaceCount; ++i) {
            int a = 0, b = 0, c = 0;
            GetFaceVertexIndex(i, a, b, c);
            tmp[i].idx[0] = (CKWORD) a;
            tmp[i].idx[1] = (CKWORD) b;
            tmp[i].idx[2] = (CKWORD) c;
            tmp[i].face = m_Faces[i];
        }

        g_ProgressiveMeshCompareContext = m_ProgressiveMesh;
        ::qsort(tmp.Begin(), tmp.Size(), sizeof(ProgressiveMeshFaceSortEntry), ProgressiveMeshCompareFunctionWrapper);
        g_ProgressiveMeshCompareContext = nullptr;

        for (int i = 0; i < finalFaceCount; ++i) {
            SetFaceVertexIndex(i, tmp[i].idx[0], tmp[i].idx[1], tmp[i].idx[2]);
            m_Faces[i] = tmp[i].face;
        }

        CreateRenderGroups();
    }

    // Match IDA at 0x10025092-0x10025200: Remap material channel UVs (allocate new array, copy, delete old)
    for (int c = 0; c < m_MaterialChannels.Size(); ++c) {
        VxMaterialChannel &ch = m_MaterialChannels[c];
        if (!ch.m_UVs)
            continue;

        Vx2DVector *newUV = new Vx2DVector[vertexCount];
        for (int j = 0; j < vertexCount; ++j) {
            const int mapped = collapseOrder[j];
            newUV[mapped] = ch.m_UVs[j];
        }

        delete[] ch.m_UVs;
        ch.m_UVs = newUV;
    }

    // Match IDA at 0x10025219-0x100252aa: Update skins for CK3dEntity that currently uses this mesh
    {
        const XObjectPointerArray &ents = m_Context->GetObjectListByType(CKCID_3DENTITY, TRUE);
        for (CKObject *const *it = ents.Begin(); it != ents.End(); ++it) {
            CK3dEntity *ent = (CK3dEntity *) *it;
            if (!ent)
                continue;
            if (ent->GetCurrentMesh() != this)
                continue;
            CKSkin *skin = ent->GetSkin();
            if (skin)
                skin->RemapVertices(collapseOrder);
        }
    }

    // Match IDA at 0x100252c2-0x100255bb: Update morph controllers
    {
        CK_ID *ids = m_Context->GetObjectsListByClassID(CKCID_OBJECTANIMATION);
        const int count = m_Context->GetObjectsCountByClassID(CKCID_OBJECTANIMATION);
        for (int i = 0; i < count; ++i) {
            CKObjectAnimation *oa = (CKObjectAnimation *) m_Context->GetObjectA(ids[i]);
            if (!oa)
                continue;

            CKMorphController *mc = oa->GetMorphController();
            if (!mc)
                continue;

            CK3dEntity *ent = oa->Get3dEntity();
            if (!ent)
                continue;
            if (ent->GetCurrentMesh() != this)
                continue;

            const int keyCount = mc->GetKeyCount();
            for (int k = 0; k < keyCount; ++k) {
                CKMorphKey *key = (CKMorphKey *) mc->GetKey(k);
                if (!key)
                    continue;

                VxVector *newPos = nullptr;
                VxCompressedVector *newNorm = nullptr;
                if (key->PosArray)
                    newPos = new VxVector[vertexCount];
                if (key->NormArray)
                    newNorm = new VxCompressedVector[vertexCount];

                for (int n = 0; n < vertexCount; ++n) {
                    const int mapped = collapseOrder[n];
                    if (newPos)
                        newPos[mapped] = key->PosArray[n];
                    if (newNorm)
                        newNorm[mapped] = key->NormArray[n];
                }

                delete[] key->PosArray;
                key->PosArray = newPos;
                delete[] key->NormArray;
                key->NormArray = newNorm;
            }
        }
    }

    return CK_OK;
}

CKBOOL RCKMesh::IsPM() {
    return m_ProgressiveMesh != nullptr;
}

//--------------------------------------------
// DefaultRender - Main mesh rendering function
//--------------------------------------------
int RCKMesh::DefaultRender(RCKRenderContext *rc, RCK3dEntity *ent) {
    CKRasterizerContext *rstContext = rc->m_RasterizerContext;

    const int vertexCount = m_Vertices.Size();
    if (vertexCount <= 0)
        return 0;

    const int faceCount = m_Faces.Size();
    const int lineCount = GetLineCount();
    if (!faceCount && !lineCount)
        return 0;

    CKDWORD zbufOnly = 0;
    CKDWORD stencilOnly = 0;
    CKBOOL renderChannels = (m_MaterialChannels.Size() > 0) && ((m_Flags & VXMESH_RENDERCHANNELS) != 0);

    m_DrawFlags = CKRST_DP_DOCLIP;

    if (ent) {
        zbufOnly = ent->m_MoveableFlags & VX_MOVEABLE_ZBUFONLY;
        stencilOnly = ent->m_MoveableFlags & VX_MOVEABLE_STENCILONLY;
        renderChannels = ((ent->m_MoveableFlags & VX_MOVEABLE_RENDERCHANNELS) != 0) && renderChannels;
    }

    const int renderVertexCount = m_ProgressiveMesh ? ClampPMVertexCount(this, GetVerticesRendered()) : vertexCount;

    // Setup VxDrawPrimitiveData (matches IDA setup; strides are based on SDK structs)
    VxDrawPrimitiveData dpData;
    memset(&dpData, 0, sizeof(dpData));

    dpData.VertexCount = renderVertexCount;
    dpData.PositionStride = sizeof(VxVertex);
    dpData.NormalStride = sizeof(VxVertex);
    dpData.TexCoordStride = sizeof(VxVertex);
    dpData.ColorStride = sizeof(VxColors);
    dpData.SpecularColorStride = sizeof(VxColors);

    if (m_Vertices.Size() > 0) {
        dpData.PositionPtr = &m_Vertices[0].m_Position;
        dpData.NormalPtr = &m_Vertices[0].m_Normal;
        dpData.TexCoordPtr = &m_Vertices[0].m_UV;
    }

    dpData.ColorPtr = (m_VertexColors.Size() > 0) ? (void *) &m_VertexColors[0].Color : nullptr;
    dpData.SpecularColorPtr = (m_VertexColors.Size() > 0) ? (void *) &m_VertexColors[0].Specular : nullptr;

    m_ActiveTextureChannels.Clear();

    rc->m_Stats.NbObjectDrawn++;
    rc->m_Stats.NbVerticesProcessed += dpData.VertexCount;

    CKBOOL hasAlphaMaterial = FALSE;
    RCKMaterial *firstMat = nullptr;

    if (faceCount) {
        // Wrap mode (matches IDA)
        const CKDWORD wrapMode = ((m_Flags & VXMESH_WRAPV) ? VXWRAP_V : 0) | ((m_Flags & VXMESH_WRAPU) ? VXWRAP_U : 0);
        rstContext->SetRenderState(VXRENDERSTATE_WRAP0, wrapMode);

        // Ensure optimized render groups exist (VXMESH_OPTIMIZED)
        if (!(m_Flags & VXMESH_OPTIMIZED))
            CreateRenderGroups();

        // Per-face channel mask remap
        m_FaceChannelMask = (CKWORD) m_FaceChannelMask;
        if (m_FaceChannelMask)
            UpdateChannelIndices();

        // Check if any group material is alpha transparent (IDA reads internal flags; SDK method is equivalent)
        for (int i = 0; i < m_MaterialGroups.Size(); ++i) {
            CKMaterialGroup *group = m_MaterialGroups[i];
            if (group && group->m_Material) {
                firstMat = group->m_Material;
                if (group->m_Material->IsAlphaTransparent()) {
                    hasAlphaMaterial = TRUE;
                    break;
                }
            }
        }

        // Z-buffer only rendering mode
        if (zbufOnly) {
            dpData.Flags = m_DrawFlags | CKRST_DP_TRANSFORM;
            rstContext->SetVertexShader(0);
            rstContext->SetTexture(0, 0);
            rstContext->SetRenderState(VXRENDERSTATE_LIGHTING, FALSE);
            rstContext->SetRenderState(VXRENDERSTATE_SPECULARENABLE, FALSE);
            rstContext->SetRenderState(VXRENDERSTATE_ALPHATESTENABLE, FALSE);
            rstContext->SetRenderState(VXRENDERSTATE_CULLMODE, VXCULL_NONE);
            rstContext->SetRenderState(VXRENDERSTATE_FILLMODE, VXFILL_SOLID);
            rstContext->SetRenderState(VXRENDERSTATE_SHADEMODE, VXSHADE_FLAT);
            rstContext->SetRenderState(VXRENDERSTATE_ZWRITEENABLE, TRUE);
            rstContext->SetRenderState(VXRENDERSTATE_ALPHABLENDENABLE, TRUE);
            rstContext->SetRenderState(VXRENDERSTATE_SRCBLEND, VXBLEND_ZERO);
            rstContext->SetRenderState(VXRENDERSTATE_DESTBLEND, VXBLEND_ONE);

            if (m_FaceVertexIndices.Size() > 0)
                rstContext->DrawPrimitive(VX_TRIANGLELIST, m_FaceVertexIndices.Begin(), m_FaceVertexIndices.Size(), &dpData);
        } else if (stencilOnly) {
            // Stencil only rendering mode
            dpData.Flags = m_DrawFlags | CKRST_DP_TRANSFORM;
            rstContext->SetVertexShader(0);
            rstContext->SetTexture(0, 0);
            rstContext->SetRenderState(VXRENDERSTATE_LIGHTING, FALSE);
            rstContext->SetRenderState(VXRENDERSTATE_SPECULARENABLE, FALSE);
            rstContext->SetRenderState(VXRENDERSTATE_CULLMODE, VXCULL_NONE);
            rstContext->SetRenderState(VXRENDERSTATE_FILLMODE, VXFILL_SOLID);
            rstContext->SetRenderState(VXRENDERSTATE_SHADEMODE, VXSHADE_FLAT);
            rstContext->SetRenderState(VXRENDERSTATE_ZWRITEENABLE, FALSE);
            rstContext->SetRenderState(VXRENDERSTATE_ALPHABLENDENABLE, TRUE);
            rstContext->SetRenderState(VXRENDERSTATE_SRCBLEND, VXBLEND_ZERO);
            rstContext->SetRenderState(VXRENDERSTATE_DESTBLEND, VXBLEND_ONE);

            if (m_FaceVertexIndices.Size() > 0)
                rstContext->DrawPrimitive(VX_TRIANGLELIST, m_FaceVertexIndices.Begin(), m_FaceVertexIndices.Size(), &dpData);

            // Match original: stencil-only disables channel passes.
            renderChannels = FALSE;
        } else {
            // Attempt mono-pass multitexturing selection (IDA behavior): mark channels rendered in mono-pass with VXCHANNEL_MONO,
            // and mark the last channel to be rendered in multi-pass with VXCHANNEL_LAST.
            for (int c = 0; c < m_MaterialChannels.Size(); ++c) {
                m_MaterialChannels[c].m_Flags &= ~(VXCHANNEL_MONO | VXCHANNEL_LAST);
            }

            VxMaterialChannel *lastMultiPass = nullptr;
            CKBOOL needsMultiPass = FALSE;

            if (renderChannels && rstContext && rstContext->m_Driver) {
                int maxAdditionalStages = (int) rstContext->m_Driver->m_3DCaps.MaxNumberTextureStage - 1;
                if (maxAdditionalStages < 0)
                    maxAdditionalStages = 0;

                int usedStages = 0;

                if (firstMat)
                    firstMat->SetAsCurrent((CKRenderContext *) rc, (m_Flags & VXMESH_PRELITMODE) == 0, 0);

                for (int c = 0; c < m_MaterialChannels.Size(); ++c) {
                    VxMaterialChannel &channel = m_MaterialChannels[c];
                    if (!channel.m_Material)
                        continue;
                    if ((channel.m_Flags & VXCHANNEL_ACTIVE) == 0)
                        continue;

                    if (usedStages >= maxAdditionalStages) {
                        lastMultiPass = &channel;
                        needsMultiPass = TRUE;
                        continue;
                    }

                    if (!hasAlphaMaterial) {
                        if (channel.m_FaceIndices || zbufOnly || IsTransparent() || !channel.m_Material->GetTexture(0)) {
                            needsMultiPass = TRUE;
                            lastMultiPass = &channel;
                            usedStages = maxAdditionalStages;
                            continue;
                        }

                        const bool allowedBlend =
                            (channel.m_SourceBlend == VXBLEND_ZERO && channel.m_DestBlend == VXBLEND_SRCCOLOR) ||
                            (channel.m_SourceBlend == VXBLEND_DESTCOLOR && channel.m_DestBlend == VXBLEND_ZERO);

                        if ((m_Flags & VXMESH_PRELITMODE) != 0 || (channel.m_Flags & VXCHANNEL_NOTLIT) != 0 || !allowedBlend) {
                            needsMultiPass = TRUE;
                            lastMultiPass = &channel;
                            usedStages = maxAdditionalStages;
                            continue;
                        }
                    }

                    // Try to set stage blend; if it fails, fall back to multi-pass rendering.
                    const int stage = usedStages + 1;
                    if (!rstContext->SetTextureStageState(stage, CKRST_TSS_STAGEBLEND, STAGEBLEND(channel.m_SourceBlend, channel.m_DestBlend))) {
                        needsMultiPass = TRUE;
                        lastMultiPass = &channel;
                        usedStages = maxAdditionalStages;
                        continue;
                    }

                    channel.m_Flags |= VXCHANNEL_MONO;
                    ++usedStages;

                    const int texCoordIndex = (int) m_ActiveTextureChannels.Size() + 1;
                    rstContext->SetTextureStageState(stage, CKRST_TSS_TEXCOORDINDEX, texCoordIndex);

                    channel.m_Material->SetAsCurrent((CKRenderContext *) rc, FALSE, stage);

                    const int slot = (int) m_ActiveTextureChannels.Size();
                    if (slot < 7) {
                        dpData.TexCoordPtrs[slot] = channel.m_UVs;
                        dpData.TexCoordStrides[slot] = sizeof(Vx2DVector);
                    }

                    m_ActiveTextureChannels.PushBack(c);
                }

                if (lastMultiPass)
                    lastMultiPass->m_Flags |= VXCHANNEL_LAST;

                renderChannels = needsMultiPass;
            }

            if (!hasAlphaMaterial && m_ActiveTextureChannels.Size() == 0) {
                rstContext->SetTexture(0, 1);
                rstContext->SetTextureStageState(1, CKRST_TSS_STAGEBLEND, STAGEBLEND(0, 0));
            }

            // Setup draw flags
            dpData.Flags = (CKRST_DP_STAGE(m_ActiveTextureChannels.Size()) | (m_DrawFlags | CKRST_DP_TRANSFORM));
            rstContext->SetTextureStageState(0, CKRST_TSS_TEXCOORDINDEX, 0);

            // Vertex color (prelit) vs lighting
            if ((m_Flags & VXMESH_PRELITMODE) != 0) {
                dpData.Flags |= CKRST_DP_DIFFUSE | CKRST_DP_SPECULAR;
                rstContext->SetRenderState(VXRENDERSTATE_LIGHTING, FALSE);
                rstContext->SetRenderState(VXRENDERSTATE_SPECULARENABLE, TRUE);
            } else {
                dpData.Flags |= CKRST_DP_LIGHT;
                rstContext->SetRenderState(VXRENDERSTATE_LIGHTING, TRUE);
            }

            // Check HW vertex buffer
            VxDrawPrimitiveData *dp = &dpData;
            m_Valid++;
            const CKDWORD vbCaps = CKRST_SPECIFICCAPS_CANDOVERTEXBUFFER | CKRST_SPECIFICCAPS_HARDWARETL;
            if (m_Valid > 3 && (rstContext->m_Driver->m_3DCaps.CKRasterizerSpecificCaps & vbCaps) == vbCaps) {
                if (CheckHWVertexBuffer(rstContext, dp)) {
                    dp = nullptr; // Use HW vertex buffer instead
                    m_VertexBufferReady = 1;
                }
            } else {
                m_VertexBufferReady = 0;
            }

            // Render non-transparent material groups first
            for (int i = 0; i < m_MaterialGroups.Size(); i++) {
                CKMaterialGroup *group = m_MaterialGroups[i];

                if (group && group->m_RemapData && dp) {
                    CKVBuffer *vb = GetVBuffer(group);
                    if (vb)
                        vb->Update(this, 0);
                }

                if (group && (!group->m_Material || !group->m_Material->IsAlphaTransparent())) {
                    if (dp) {
                        VxDrawPrimitiveData groupDp = *dp;
                        RenderGroup(rc, group, ent, &groupDp);
                    } else {
                        RenderGroup(rc, group, ent, nullptr);
                    }
                }
            }

            // Render transparent material groups (starting from second element)
            for (int i = 1; i < m_MaterialGroups.Size(); i++) {
                CKMaterialGroup *group = m_MaterialGroups[i];
                if (group && group->m_Material && group->m_Material->IsAlphaTransparent()) {
                    if (dp) {
                        VxDrawPrimitiveData groupDp = *dp;
                        RenderGroup(rc, group, ent, &groupDp);
                    } else {
                        RenderGroup(rc, group, ent, nullptr);
                    }
                }
            }

            // Clear mesh geometry changed flags
            const CKDWORD meshGeometryChangedFlags =
                VXMESH_UV_CHANGED |
                VXMESH_NORMAL_CHANGED |
                VXMESH_COLOR_CHANGED |
                VXMESH_POS_CHANGED;
            m_Flags &= ~meshGeometryChangedFlags;

            // Wireframe overlay
            if (rc->m_DisplayWireframe) {
                VxMatrix projMat;
                memcpy(&projMat, rc->GetProjectionTransformationMatrix(), sizeof(VxMatrix));
                float origZ = projMat[3][2];
                projMat[3][2] = origZ * 1.003f;
                rc->SetProjectionTransformationMatrix(projMat);

                rstContext->SetTexture(0, 0);
                rstContext->SetVertexShader(0);
                rstContext->SetRenderState(VXRENDERSTATE_LIGHTING, FALSE);
                rstContext->SetRenderState(VXRENDERSTATE_FILLMODE, VXFILL_WIREFRAME);

                VxDrawPrimitiveData wfDp = dpData;
                wfDp.Flags = m_DrawFlags | CKRST_DP_TRANSFORM;
                if (m_FaceVertexIndices.Size() > 0)
                    rstContext->DrawPrimitive(VX_TRIANGLELIST, m_FaceVertexIndices.Begin(), m_FaceVertexIndices.Size(), &wfDp);

                projMat[3][2] = origZ;
                rc->SetProjectionTransformationMatrix(projMat);
            }
        }

        rc->m_Stats.NbTrianglesDrawn += faceCount;

        // Render material channels if needed
        if (renderChannels) {
            CKDWORD fogEnable = FALSE;
            rstContext->GetRenderState(VXRENDERSTATE_FOGENABLE, &fogEnable);
            RenderChannels(rc, ent, &dpData, fogEnable);
            rstContext->SetRenderState(VXRENDERSTATE_FOGENABLE, fogEnable);
        }
    }

    // Render lines
    if (lineCount) {
        VxDrawPrimitiveData lineDp;
        memset(&lineDp, 0, sizeof(lineDp));

        lineDp.VertexCount = renderVertexCount;
        lineDp.PositionStride = sizeof(VxVertex);
        lineDp.ColorStride = sizeof(VxColors);
        lineDp.SpecularColorStride = sizeof(VxColors);
        lineDp.PositionPtr = (m_Vertices.Size() > 0) ? (void *) &m_Vertices[0].m_Position : nullptr;
        lineDp.ColorPtr = (m_VertexColors.Size() > 0) ? (void *) &m_VertexColors[0].Color : nullptr;
        lineDp.SpecularColorPtr = (m_VertexColors.Size() > 0) ? (void *) &m_VertexColors[0].Specular : nullptr;

        rstContext->SetRenderState(VXRENDERSTATE_LIGHTING, FALSE);
        rstContext->SetRenderState(VXRENDERSTATE_ZWRITEENABLE, TRUE);
        rstContext->SetRenderState(VXRENDERSTATE_ZFUNC, VXCMP_LESSEQUAL);
        rstContext->SetRenderState(VXRENDERSTATE_SHADEMODE, VXSHADE_GOURAUD);
        rstContext->SetRenderState(VXRENDERSTATE_ALPHABLENDENABLE, FALSE);
        rstContext->SetTexture(0, 0);

        lineDp.Flags = (m_DrawFlags | CKRST_DP_TRANSFORM | CKRST_DP_DIFFUSE);
        if (m_LineIndices.Size() > 0)
            rstContext->DrawPrimitive(VX_LINELIST, m_LineIndices.Begin(), m_LineIndices.Size(), &lineDp);

        rc->m_Stats.NbLinesDrawn += lineCount;
    }

    // Reset wrap state
    rstContext->SetRenderState(VXRENDERSTATE_WRAP0, 0);

    // Clear texture stages used for mono-pass (IDA)
    if (hasAlphaMaterial || m_ActiveTextureChannels.Size() > 0) {
        for (int i = 1; i < m_ActiveTextureChannels.Size() + 1; ++i)
            rstContext->SetTextureStageState(i, CKRST_TSS_STAGEBLEND, 0);
    }

    return 1;
}

//--------------------------------------------
// RenderGroup - Render a single material group
// IDA: 0x10022829 (1864 bytes)
//--------------------------------------------
int RCKMesh::RenderGroup(RCKRenderContext *dev, CKMaterialGroup *group, RCK3dEntity *ent, VxDrawPrimitiveData *data) {
    CKRasterizerContext *rstContext = dev->m_RasterizerContext;
    RCKMaterial *mat = group->m_Material;

    // Check for pre-render submesh callbacks - sub_1002C220 checks if Size() > 0
    if (m_SubMeshCallbacks && m_SubMeshCallbacks->m_PreCallBacks.Size() > 0) {
        // Execute callbacks if there are any
        if (m_SubMeshCallbacks->m_PreCallBacks.Size() > 0) {
            dev->m_ObjectsCallbacksTimeProfiler.Reset();
            dev->m_RasterizerContext->SetVertexShader(0);

            // Iterate through callbacks (VxCallBack is 12 bytes: callback ptr, argument ptr, flags)
            for (VxCallBack *cb = m_SubMeshCallbacks->m_PreCallBacks.Begin();
                 cb < m_SubMeshCallbacks->m_PreCallBacks.End(); ++cb) {
                // Call: func(dev, ent, this, mat, argument)
                typedef void (*MeshRenderCallback)(RCKRenderContext *, RCK3dEntity *, RCKMesh *, RCKMaterial *, void *);
                MeshRenderCallback func = (MeshRenderCallback) cb->callback;
                func(dev, ent, this, mat, cb->argument);
            }

            dev->m_Stats.ObjectsCallbacksTime += dev->m_ObjectsCallbacksTimeProfiler.Current();
        }

        // If material is null after callbacks, use default
        if (!mat)
            mat = (RCKMaterial *) dev->m_RenderManager->GetDefaultMaterial();
    } else if (mat) {
        // Normal case - set material as current
        mat->SetAsCurrent((CKRenderContext *) dev, !(m_Flags & VXMESH_PRELITMODE), 0);

        // If material has no texture but we have texture stages, set up modulate
        if (!mat->GetTexture(0) && m_ActiveTextureChannels.Size() > 0) {
            rstContext->SetTextureStageState(0, CKRST_TSS_OP, CKRST_TOP_SELECTARG1);
            rstContext->SetTextureStageState(0, CKRST_TSS_ARG1, CKRST_TA_DIFFUSE);
            rstContext->SetTextureStageState(0, CKRST_TSS_AOP, CKRST_TOP_SELECTARG1);
            rstContext->SetTextureStageState(0, CKRST_TSS_AARG1, CKRST_TA_DIFFUSE);
        }
    } else {
        // No material - use default
        mat = (RCKMaterial *) dev->m_RenderManager->GetDefaultMaterial();
        mat->SetAsCurrent((CKRenderContext *) dev, !(m_Flags & VXMESH_PRELITMODE), 0);

        // Check for vertex color alpha blending (flags 0x80 and 0x1000)
        if ((m_Flags & (VXMESH_PRELITMODE | VXMESH_FORCETRANSPARENCY)) == (VXMESH_PRELITMODE | VXMESH_FORCETRANSPARENCY)) {
            rstContext->SetRenderState(VXRENDERSTATE_SRCBLEND, VXBLEND_SRCALPHA);
            rstContext->SetRenderState(VXRENDERSTATE_DESTBLEND, VXBLEND_INVSRCALPHA);
            rstContext->SetRenderState(VXRENDERSTATE_ALPHABLENDENABLE, TRUE);
        }
    }

    // Handle entity-specific render states (IDA)
    if (ent) {
        if ((ent->m_MoveableFlags & VX_MOVEABLE_NOZBUFFERTEST) != 0)
            rstContext->SetRenderState(VXRENDERSTATE_ZFUNC, VXCMP_ALWAYS);
        if ((ent->m_MoveableFlags & VX_MOVEABLE_NOZBUFFERWRITE) != 0)
            rstContext->SetRenderState(VXRENDERSTATE_ZWRITEENABLE, FALSE);
    }

    // Render primitives
    if (data) {
        // Software vertex path (data != null)

        // Update data pointers if group has remapped vertices
        if (group->m_RemapData) {
            CKVBuffer *vb = GetVBuffer(group);
            if (vb && group->m_VertexCount > 0 && vb->m_Vertices.Size() >= (int) group->m_VertexCount &&
                vb->m_Colors.Size() >= (int) group->m_VertexCount) {
                VxVertex *verts = vb->m_Vertices.Begin();
                VxColors *colors = vb->m_Colors.Begin();

                data->VertexCount = group->m_VertexCount;
                data->PositionPtr = &verts[0].m_Position;
                data->NormalPtr = &verts[0].m_Normal;
                data->TexCoordPtr = &verts[0].m_UV;
                data->ColorPtr = &colors[0].Color;
                data->SpecularColorPtr = &colors[0].Specular;

                for (int j = 0; j < m_ActiveTextureChannels.Size(); ++j) {
                    const int channelIdx = m_ActiveTextureChannels[j];
                    Vx2DVector *channelUVs = nullptr;
                    if (channelIdx >= 0 && channelIdx < vb->m_UVs.Size()) {
                        if (vb->m_UVs[channelIdx].Size() == (int) group->m_VertexCount)
                            channelUVs = vb->m_UVs[channelIdx].Begin();
                    }
                    data->TexCoordPtrs[j] = channelUVs;
                    data->TexCoordStrides[j] = sizeof(Vx2DVector);
                }
            }
        }

        // Two-sided alpha transparent materials need back-face pass first
        if (mat->GetFillMode() == VXFILL_SOLID && mat->IsTwoSided() && mat->IsAlphaTransparent()) {
            rstContext->SetRenderState(VXRENDERSTATE_CULLMODE, VXCULL_CW); // 2

            for (CKPrimitiveEntry *prim = group->m_Primitives.Begin(); prim < group->m_Primitives.End(); ++prim) {
                if (prim->m_Indices.Size() > 0) {
                    int indexCount = prim->m_Indices.Size();
                    CKWORD *indices = prim->m_Indices.Begin();
                    rstContext->DrawPrimitive(prim->m_Type, indices, indexCount, data);
                }
            }

            rstContext->SetRenderState(VXRENDERSTATE_CULLMODE, VXCULL_CCW);
        }

        // Main render pass
        for (CKPrimitiveEntry *prim = group->m_Primitives.Begin(); prim < group->m_Primitives.End(); ++prim) {
            if (prim->m_Indices.Size() > 0) {
                int indexCount = prim->m_Indices.Size();
                CKWORD *indices = prim->m_Indices.Begin();
                rstContext->DrawPrimitive(prim->m_Type, indices, indexCount, data);
            }
        }
    } else {
        // Hardware vertex buffer path (data == null)

        // Two-sided alpha transparent materials need back-face pass first
        if (mat->GetFillMode() == VXFILL_SOLID && mat->IsTwoSided() && mat->IsAlphaTransparent()) {
            rstContext->SetRenderState(VXRENDERSTATE_CULLMODE, VXCULL_CW);

            for (CKPrimitiveEntry *prim = group->m_Primitives.Begin(); prim < group->m_Primitives.End(); ++prim) {
                if (prim->m_Indices.Size() > 0) {
                    int indexCount = prim->m_Indices.Size();
                    CKWORD *indices = prim->m_Indices.Begin();
                    rstContext->DrawPrimitiveVB(prim->m_Type, m_VertexBuffer,
                                                group->m_BaseVertex, group->m_VertexCount,
                                                indices, indexCount);
                }
            }

            // Flush vertex buffer (Lock/Unlock with size 0)
            rstContext->LockVertexBuffer(m_VertexBuffer, 0, 0, CKRST_LOCK_DEFAULT);
            rstContext->UnlockVertexBuffer(m_VertexBuffer);

            rstContext->SetRenderState(VXRENDERSTATE_CULLMODE, VXCULL_CCW);
        }

        // Main render pass - check for index buffer usage
        for (CKPrimitiveEntry *prim = group->m_Primitives.Begin(); prim < group->m_Primitives.End(); ++prim) {
            if (prim->m_Indices.Size() > 0) {
                int indexCount = prim->m_Indices.Size();

                if ((int) prim->m_IndexBufferOffset >= 0) {
                    // Use hardware index buffer
                    rstContext->DrawPrimitiveVBIB(prim->m_Type, m_VertexBuffer, m_IndexBuffer,
                                                  group->m_BaseVertex, group->m_VertexCount,
                                                  prim->m_IndexBufferOffset, indexCount);
                } else {
                    // Use software indices with hardware vertex buffer
                    CKWORD *indices = prim->m_Indices.Begin();
                    rstContext->DrawPrimitiveVB(prim->m_Type, m_VertexBuffer,
                                                group->m_BaseVertex, group->m_VertexCount,
                                                indices, indexCount);
                }
            }
        }
    }

    // Execute post-render submesh callbacks
    if (m_SubMeshCallbacks && m_SubMeshCallbacks->m_PostCallBacks.Size() > 0) {
        dev->m_ObjectsCallbacksTimeProfiler.Reset();
        dev->m_RasterizerContext->SetVertexShader(0);

        for (VxCallBack *cb = m_SubMeshCallbacks->m_PostCallBacks.Begin();
             cb < m_SubMeshCallbacks->m_PostCallBacks.End(); ++cb) {
            typedef void (*MeshRenderCallback)(RCKRenderContext *, RCK3dEntity *, RCKMesh *, RCKMaterial *, void *);
            MeshRenderCallback func = (MeshRenderCallback) cb->callback;
            func(dev, ent, this, mat, cb->argument);
        }

        dev->m_Stats.ObjectsCallbacksTime += dev->m_ObjectsCallbacksTimeProfiler.Current();
    }

    return 1;
}

//--------------------------------------------
// RenderChannels - Render material channels (multi-texture passes)
// IDA: 0x10022f71 (1115 bytes)
//--------------------------------------------
int RCKMesh::RenderChannels(RCKRenderContext *dev, RCK3dEntity *ent, VxDrawPrimitiveData *data, int fogEnable) {
    CKRasterizerContext *rstContext = dev->m_RasterizerContext;

    // Setup flags for channel rendering
    data->Flags = m_DrawFlags | CKRST_DP_TRANSFORM | CKRST_DP_STAGES0;

    // Save original color pointers
    void *origColorPtr = data->ColorPtr;
    void *origSpecularPtr = data->SpecularColorPtr;

    // Get and modify projection matrix for Z offset
    VxMatrix projMat = dev->GetProjectionTransformationMatrix();
    float origZ = projMat[3][2];
    float newZ = origZ * 1.001f;
    projMat[3][2] = newZ;
    dev->SetProjectionTransformationMatrix(projMat);

    // Iterate through material channels
    for (int c = 0; c < m_MaterialChannels.Size(); ++c) {
        VxMaterialChannel &channel = m_MaterialChannels[c];
        RCKMaterial *mat = (RCKMaterial *) channel.m_Material;
        CKDWORD &channelFlags = channel.m_Flags;
        XArray<CKWORD> *faceIndices = channel.m_FaceIndices;

        // Skip inactive or already processed channels
        if (!mat)
            continue;
        if (!(channelFlags & VXCHANNEL_ACTIVE))
            continue;
        if ((channelFlags & VXCHANNEL_MONO) != 0)
            continue;
        // Check if channel has face indices and they have data (sub_100047B0)
        if (faceIndices && faceIndices->Size() == 0)
            continue;

        // Match original binary: patch blend/flags without side effects
        VXBLEND_MODE savedSourceBlend = (VXBLEND_MODE) 0;
        VXBLEND_MODE savedDestBlend = (VXBLEND_MODE) 0;
        CKDWORD savedFlags = 0;
        mat->PatchForChannelRender(channel.m_SourceBlend, channel.m_DestBlend, savedSourceBlend, savedDestBlend, savedFlags);

        // Set fog enable based on blend mode (SRCALPHA/INVSRCALPHA)
        if (channel.m_SourceBlend == VXBLEND_SRCALPHA && channel.m_DestBlend == VXBLEND_INVSRCALPHA) {
            rstContext->SetRenderState(VXRENDERSTATE_FOGENABLE, fogEnable);
        } else {
            rstContext->SetRenderState(VXRENDERSTATE_FOGENABLE, FALSE);
        }

        rstContext->SetTextureStageState(0, CKRST_TSS_TEXCOORDINDEX, 0);

        // Check for additive blending (mark as unlit)
        if ((channel.m_SourceBlend == VXBLEND_ZERO && channel.m_DestBlend == VXBLEND_SRCCOLOR) ||
            (channel.m_SourceBlend == VXBLEND_DESTCOLOR && channel.m_DestBlend == VXBLEND_ZERO)) {
            channelFlags |= VXCHANNEL_NOTLIT;
        }

        // Determine if lighting should be used
        CKBOOL useLighting = (m_Flags & VXMESH_PRELITMODE) == 0 && (channelFlags & VXCHANNEL_NOTLIT) == 0;

        // Call CKMaterial::SetAsCurrent(dev, useLighting, 0)
        mat->SetAsCurrent((CKRenderContext *) dev, useLighting, 0);
        rstContext->SetRenderState(VXRENDERSTATE_LIGHTING, useLighting);

        // Handle unlit channel - clear color pointers
        if ((channelFlags & VXCHANNEL_NOTLIT) != 0) {
            data->ColorPtr = nullptr;
            data->SpecularColorPtr = nullptr;
            CKDWORD dpFlags = data->Flags;
            dpFlags = dpFlags & ~CKRST_DP_LIGHT;
            data->Flags = dpFlags;
        } else if ((m_Flags & VXMESH_PRELITMODE) != 0) {
            // Vertex color mode
            data->Flags &= ~CKRST_DP_LIGHT;
            data->Flags |= CKRST_DP_DIFFUSE | CKRST_DP_SPECULAR;
            data->ColorPtr = origColorPtr;
            data->SpecularColorPtr = origSpecularPtr;
        } else {
            // Normal lighting mode
            CKDWORD dpFlags = data->Flags;
            dpFlags = dpFlags | CKRST_DP_LIGHT;
            data->Flags = dpFlags;
        }

        // Setup texture coordinates
        if ((channelFlags & VXCHANNEL_SAMEUV) != 0) {
            // Use main vertex UVs
            data->TexCoordPtr = &m_Vertices[0].m_UV;
            data->TexCoordStride = sizeof(VxVertex);
        } else {
            // Use channel-specific UVs
            data->TexCoordPtr = channel.m_UVs;
            data->TexCoordStride = sizeof(Vx2DVector);
        }

        // Get indices to draw
        CKWORD *indices;
        int indexCount;
        if (faceIndices) {
            indices = faceIndices->Begin();
            indexCount = faceIndices->Size();
        } else {
            indices = m_FaceVertexIndices.Begin();
            indexCount = m_FaceVertexIndices.Size();
        }

        // Draw the channel
        rstContext->DrawPrimitive(VX_TRIANGLELIST, indices, indexCount, data);

        // Restore material state exactly
        mat->RestoreAfterChannelRender(savedSourceBlend, savedDestBlend, savedFlags);
    }

    // Restore texture coordinate pointer
    data->TexCoordPtr = &m_Vertices[0].m_UV;
    data->TexCoordStride = sizeof(VxVertex);
    data->ColorPtr = origColorPtr;
    data->SpecularColorPtr = origSpecularPtr;

    // Restore projection matrix
    projMat[3][2] = origZ;
    dev->SetProjectionTransformationMatrix(projMat);

    return 1;
}

//--------------------------------------------
// CreateRenderGroups - Build material groups for rendering
// IDA: 0x1001f898 (4171 bytes)
//--------------------------------------------
int RCKMesh::CreateRenderGroups() {
    // Set mono-material flag initially and invalidate
    m_Flags |= VXMESH_MONOMATERIAL;
    m_Valid = 0;

    // Check for valid geometry
    int vertexCount = m_Vertices.Size();
    int faceCount = m_Faces.Size();
    if (vertexCount <= 0 || faceCount <= 0) {
        m_Flags |= VXMESH_OPTIMIZED;
        return 0;
    }

    // Reset all existing material groups (IDA: sub_1002A0F0(group, 0))
    for (int i = 0; i < m_MaterialGroups.Size(); i++) {
        ResetMaterialGroup(m_MaterialGroups[i], 0);
    }

    // Allocate temporary tracker array (IDA uses CKMemoryPool(context, vertexCount) and then treats it as CKDWORD[vertexCount])
    CKMemoryPool pool(m_Context, vertexCount);
    CKDWORD *vertexTracker = (CKDWORD *) pool.Mem();

    // Track min/max material indices
    int maxMatIndex = 0;
    int minMatIndex = 2048;

    // Pointer to face vertex indices
    CKWORD *faceIndices = m_FaceVertexIndices.Begin();

    int matGroupCount = m_MaterialGroups.Size();

    if (matGroupCount >= 31) {
        // For >= 31 material groups, use XBitArray per vertex to track materials
        // Each vertex has a bit array to track which materials it's used by
        XClassArray<XBitArray> vertexMaterialSets;
        vertexMaterialSets.Resize(vertexCount);

        CKFace *facePtr = (CKFace *) m_Faces.Begin();
        for (int faceIdx = 0; faceIdx < faceCount; faceIdx++) {
            int matIdx = facePtr->m_MatIndex;

            // Track min/max
            if (matIdx > maxMatIndex) maxMatIndex = matIdx;
            if (matIdx < minMatIndex) minMatIndex = matIdx;

            // Add face index to material group's face list
            CKWORD faceWord = (CKWORD) faceIdx;
            CKMaterialGroup *group = m_MaterialGroups[matIdx];
            group->m_FaceIndices.PushBack(faceWord);

            // For each of the 3 vertices in this face
            for (int v = 0; v < 3; v++) {
                CKWORD vidx = faceIndices[v];
                XBitArray &bits = vertexMaterialSets[vidx];
                // Check if this material was already added for this vertex
                if (!bits.IsSet(matIdx)) {
                    bits.Set(matIdx);
                    // Increment vertex count for this material group
                    group->m_VertexCount++;
                }
            }

            facePtr++;
            faceIndices += 3;
        }
    } else {
        // For < 31 material groups, use 32-bit mask per vertex
        memset(vertexTracker, 0, vertexCount * sizeof(CKDWORD));

        CKFace *facePtr = (CKFace *) m_Faces.Begin();
        for (int faceIdx = 0; faceIdx < faceCount; faceIdx++) {
            int matIdx = facePtr->m_MatIndex;

            // Track min/max
            if (matIdx > maxMatIndex) maxMatIndex = matIdx;
            if (matIdx < minMatIndex) minMatIndex = matIdx;

            // Add face index to material group's face list
            CKWORD faceWord = (CKWORD) faceIdx;
            CKMaterialGroup *group = m_MaterialGroups[matIdx];
            group->m_FaceIndices.PushBack(faceWord);

            // Bit mask for this material
            CKDWORD matBit = 1 << matIdx;

            // For each of the 3 vertices in this face
            for (int v = 0; v < 3; v++) {
                CKWORD vidx = faceIndices[v];
                // Check if this material was already added for this vertex
                if ((vertexTracker[vidx] & matBit) == 0) {
                    vertexTracker[vidx] |= matBit;
                    // Increment vertex count for this material group
                    group->m_VertexCount++;
                }
            }

            facePtr++;
            faceIndices += 3;
        }
    }

    // Accumulate vertex buffer offsets
    CKDWORD totalVertexOffset = 0;

    // Check if single material (mono-material)
    if (minMatIndex == maxMatIndex) {
        // Single material case - copy all face indices directly
        CKMaterialGroup *group = m_MaterialGroups[minMatIndex];

        // Resize index array to hold all indices
        CKPrimitiveEntry &entry = group->m_Primitives[0];
        entry.m_Indices.Resize(faceCount * 3);

        // Copy all indices directly
        memcpy(entry.m_Indices.Begin(), m_FaceVertexIndices.Begin(), faceCount * 3 * sizeof(CKWORD));
        entry.m_IndexBufferOffset = -1;

        // Set vertex range for full mesh
        group->m_MinVertexIndex = 0;
        group->m_MaxVertexIndex = vertexCount;
        group->m_VertexCount = vertexCount;
        group->m_BaseVertex = 0;
    } else {
        // Multiple materials - clear mono-material flag
        m_Flags &= ~VXMESH_MONOMATERIAL;

        // Process each material group (build local remap + indices + VBuffer)
        for (int matIdx = 0; matIdx < m_MaterialGroups.Size(); matIdx++) {
            CKMaterialGroup *group = m_MaterialGroups[matIdx];
            if (!group || group->m_VertexCount == 0 || group->m_FaceIndices.Size() == 0)
                continue;

            const int expectedLocalVertexCount = (int) group->m_VertexCount;

            // Per-group remap buffer
            CKVBuffer *vb = new CKVBuffer(expectedLocalVertexCount);
            group->m_RemapData = static_cast<CKDWORD>(reinterpret_cast<uintptr_t>(vb));
            group->m_VertexCount = 0;

            memset(vertexTracker, 0, vertexCount * sizeof(CKDWORD));

            // Allocate indices for this group
            group->m_Primitives.Resize(1);
            group->m_Primitives[0].m_Type = VX_TRIANGLELIST;
            group->m_Primitives[0].m_IndexBufferOffset = -1;

            const int faceListCount = group->m_FaceIndices.Size();
            // IDA reserves then push-backs indices
            group->m_Primitives[0].m_Indices.Resize(faceListCount * 3);
            group->m_Primitives[0].m_Indices.Resize(0);

            for (int f = 0; f < faceListCount; f++) {
                const int faceIdx = group->m_FaceIndices[f];
                CKWORD *srcIndices = m_FaceVertexIndices.Begin() + faceIdx * 3;

                for (int v = 0; v < 3; v++) {
                    const CKWORD globalIdx = srcIndices[v];
                    if (vertexTracker[globalIdx] == 0) {
                        vertexTracker[globalIdx] = group->m_VertexCount + 1;
                        const int localIdx = (int) group->m_VertexCount;
                        group->m_VertexCount = localIdx + 1;
                        if (vb && localIdx < vb->m_VertexRemap.Size())
                            vb->m_VertexRemap[localIdx] = (int) globalIdx;
                    }

                    group->m_Primitives[0].m_Indices.PushBack((CKWORD) (vertexTracker[globalIdx] - 1));
                }
            }

            // Tighten VBuffer arrays if the earlier unique-count was off
            if (vb && (int) group->m_VertexCount != vb->m_VertexRemap.Size()) {
                vb->Resize((int) group->m_VertexCount);
            }

            group->m_MinVertexIndex = 0;
            group->m_MaxVertexIndex = group->m_VertexCount;
            group->m_BaseVertex = totalVertexOffset;
            totalVertexOffset += group->m_VertexCount;

            if (vb)
                vb->Update(this, 1);
        }
    }

    // Prune invalid groups (IDA keeps index 0)
    if (m_MaterialGroups.Size() > 0) {
        UpdateHasValidPrimitives(m_MaterialGroups[0]);
    }

    int currentIndex = 1;
    for (int idx = 1; idx < m_MaterialGroups.Size();) {
        CKMaterialGroup *group = m_MaterialGroups[idx];
        UpdateHasValidPrimitives(group);

        if (group && group->m_HasValidPrimitives) {
            ++idx;
            ++currentIndex;
            continue;
        }

        if (group) {
            DeleteVBuffer(group);
            delete group;
        }
        m_MaterialGroups.RemoveAt(idx);

        for (int f = 0; f < m_Faces.Size(); f++) {
            if (m_Faces[f].m_MatIndex > currentIndex)
                --m_Faces[f].m_MatIndex;
        }
    }

    // Clear channel face indices
    for (int c = 0; c < m_MaterialChannels.Size(); c++) {
        if (m_MaterialChannels[c].m_FaceIndices) {
            delete m_MaterialChannels[c].m_FaceIndices;
            m_MaterialChannels[c].m_FaceIndices = nullptr;
        }
    }

    // Reset face channel mask (will trigger rebuild on next render)
    m_FaceChannelMask = 0xFFFF;

    // Triangle strip optimization (if enabled)
    if ((m_Flags & VXMESH_STRIPIFY) != 0) {
        for (CKMaterialGroup **it = m_MaterialGroups.Begin(); it < m_MaterialGroups.End(); ++it) {
            CKMaterialGroup *group = *it;
            if (!group)
                continue;

            for (CKPrimitiveEntry *prim = group->m_Primitives.Begin(); prim < group->m_Primitives.End(); ++prim) {
                prim->m_IndexBufferOffset = -1;
                if (prim->m_Type != VX_TRIANGLELIST)
                    continue;
                if (prim->m_Indices.Size() <= 0)
                    continue;

                // Faithful behavior (binary): use NvStripifier::Stripify() to generate strip objects,
                // then NvStripifier::CreateStrips(restartEnabled=1) to emit a connected strip stream.
                // The binary call site uses: Stripify(..., minStripLength=0, cacheSize=16, vertexCount, ...).
                // Note: IDA passes the mesh's global vertex count here (safe upper bound).
                NvStripifier stripifier;
                XArray<NvStripInfo *> strips;
                XArray<CKWORD> outStripIndices;
                CKDWORD outStripCount = 0;

                stripifier.Stripify(
                    prim->m_Indices,
                    /*minStripLength=*/0,
                    /*cacheSize=*/16,
                    /*vertexCount=*/(int)vertexCount,
                    strips);

                stripifier.CreateStrips(
                    strips,
                    outStripIndices,
                    /*restartEnabled=*/true,
                    outStripCount);

                // Binary compares element counts, not byte size.
                if (outStripIndices.Size() > 0 && outStripIndices.Size() < prim->m_Indices.Size()) {
                    prim->m_Type = VX_TRIANGLESTRIP;
                    prim->m_Indices.Swap(outStripIndices);

                    // Binary: calls into RCKRenderManager::m_VertexCacheOptimizer (sub_1002B150)
                    // after a successful strip conversion, passing the vertex cache option value
                    // and a constant 1. The returned float is not used.
                    RCKRenderManager *rm = (RCKRenderManager *) m_Context->GetRenderManager();
                    if (rm) {
                        (void)rm->m_VertexCacheOptimizer.ComputeCacheMetric(prim->m_Indices, rm->m_VertexCache.Value, 1);
                    }
                }

                stripifier.DestroyStrips(strips);
            }
        }
    } else {
        // Vertex cache optimization (if enabled)
        // Binary uses VertexCacheOptimizer stored in RCKRenderManager at offset 0xD0
        RCKRenderManager *rm = (RCKRenderManager *) m_Context->GetRenderManager();
        if (rm && rm->m_VertexCache.Value > 0 && vertexCount > 0) {
            const int cacheSize = rm->m_VertexCache.Value;
            VertexCacheOptimizer &optimizer = rm->m_VertexCacheOptimizer;

            for (CKMaterialGroup **it = m_MaterialGroups.Begin(); it < m_MaterialGroups.End(); ++it) {
                CKMaterialGroup *group = *it;
                if (!group)
                    continue;
                for (CKPrimitiveEntry *prim = group->m_Primitives.Begin(); prim < group->m_Primitives.End(); ++prim) {
                    prim->m_IndexBufferOffset = -1;
                    if (prim->m_Type != VX_TRIANGLELIST)
                        continue;
                    if (prim->m_Indices.Size() <= 0)
                        continue;

                    const int localVertexCount = (int)group->m_VertexCount;
                    const int faceCount = prim->m_Indices.Size() / 3;

                    // Binary: always attempts optimization when enabled.
                    optimizer.Initialize(localVertexCount, faceCount, cacheSize);
                    optimizer.BuildVertexFaceLists(prim->m_Indices);
                    optimizer.ProcessFaces(prim->m_Indices);
                    optimizer.SwapOutput(prim->m_Indices);
                }
            }
        }
    }

    // Mark render groups as built
    m_Flags |= VXMESH_OPTIMIZED;
    return 1;
}

//--------------------------------------------
// UpdateChannelIndices - Update channel face indices
// Builds per-channel index lists for faces that use each material channel
//--------------------------------------------
void RCKMesh::UpdateChannelIndices() {
    // Compute the intersection of all face channel masks
    // (which channel bits are set in ALL faces)
    CKWORD allFacesMask = 0xFFFF;
    for (int i = 0; i < m_Faces.Size(); i++) {
        allFacesMask &= m_Faces[i].m_ChannelMask;
    }

    // Channel bit for iteration
    CKWORD channelBit = 1;

    // Process each material channel
    for (int c = 0; c < m_MaterialChannels.Size(); c++) {
        VxMaterialChannel &channel = m_MaterialChannels[c];

        if ((channelBit & allFacesMask) != 0) {
            // This channel is used by ALL faces - no need for per-channel indices
            // Free the index array if it exists
            if (channel.m_FaceIndices) {
                delete channel.m_FaceIndices;
                channel.m_FaceIndices = nullptr;
            }
        } else if (!channel.m_FaceIndices || (channelBit & m_FaceChannelMask) != 0) {
            // Need to build index list for faces that use this channel

            // Create index array if needed
            if (channel.m_FaceIndices) {
                channel.m_FaceIndices->Clear();
            } else {
                channel.m_FaceIndices = new XArray<CKWORD>();
            }

            // Iterate through faces and collect vertex indices for faces using this channel
            CKWORD *faceIndices = m_FaceVertexIndices.Begin();
            for (int f = 0; f < m_Faces.Size(); f++) {
                if ((channelBit & m_Faces[f].m_ChannelMask) != 0) {
                    // This face uses this channel - add its 3 vertex indices
                    channel.m_FaceIndices->PushBack(faceIndices[0]);
                    channel.m_FaceIndices->PushBack(faceIndices[1]);
                    channel.m_FaceIndices->PushBack(faceIndices[2]);
                }
                faceIndices += 3;
            }
        }

        // Move to next channel bit
        channelBit <<= 1;
    }

    // Clear the face channel mask to indicate indices are up to date
    m_FaceChannelMask = 0;
}

//--------------------------------------------
// CheckHWVertexBuffer - Check and setup hardware vertex buffer
// Returns TRUE if hardware VB was created/updated successfully
//--------------------------------------------
CKBOOL RCKMesh::CheckHWVertexBuffer(CKRasterizerContext *rst, VxDrawPrimitiveData *data) {
    if (!rst || !data)
        return FALSE;

    CKBOOL needNewBuffer = FALSE;

    // Calculate vertex format and size
    CKDWORD vertexSize = 0;
    CKDWORD vertexFormat = CKRSTGetVertexFormat((CKRST_DPFLAGS) data->Flags, vertexSize);

    // Count total vertices needed
    CKDWORD totalVertexCount = 0;
    CKBOOL hasRemappedVertices = FALSE;
    CKBOOL hasDirectVertices = FALSE;

    for (int i = 0; i < m_MaterialGroups.Size(); i++) {
        CKMaterialGroup *group = m_MaterialGroups[i];
        if (group && group->m_Material) {
            if (group->m_RemapData) {
                hasRemappedVertices = TRUE;
                totalVertexCount += group->m_VertexCount;
            } else {
                hasDirectVertices = TRUE;
            }
        }
    }

    // If direct vertices, add base vertex count
    if (hasDirectVertices) {
        totalVertexCount += data->VertexCount;
    }

    // Check if existing vertex buffer is sufficient
    CKVertexBufferDesc *vbDesc = rst->GetVertexBufferData(m_VertexBuffer);
    if (vbDesc) {
        // Check if format or size changed
        if (vbDesc->m_MaxVertexCount < totalVertexCount || vbDesc->m_VertexFormat != vertexFormat) {
            rst->DeleteObject(m_VertexBuffer, CKRST_OBJ_VERTEXBUFFER);
            rst->DeleteObject(m_IndexBuffer, CKRST_OBJ_INDEXBUFFER);
            needNewBuffer = TRUE;
        }
    } else {
        needNewBuffer = TRUE;
    }

    // Create new vertex buffer if needed
    if (needNewBuffer) {
        CKVertexBufferDesc newDesc;
        newDesc.m_Flags = CKRST_VB_VALID | CKRST_VB_WRITEONLY | CKRST_VB_SHARED;
        newDesc.m_MaxVertexCount = totalVertexCount;
        newDesc.m_VertexFormat = vertexFormat;
        newDesc.m_VertexSize = vertexSize;

        if (!rst->CreateObject(m_VertexBuffer, CKRST_OBJ_VERTEXBUFFER, &newDesc)) {
            return FALSE;
        }
        m_VertexBufferReady = 0; // Mark as needing update
    }

    // If buffer is up to date, just check index buffer and return
    if (m_VertexBufferReady != 0) {
        RCKRenderManager *rm = (RCKRenderManager *) m_Context->GetRenderManager();
        if (rm->m_UseIndexBuffers.Value) {
            CheckHWIndexBuffer(rst);
        }
        return TRUE;
    }

    // Lock and fill vertex buffer
    CKBYTE *vbData = (CKBYTE *) rst->LockVertexBuffer(m_VertexBuffer, 0, totalVertexCount, (CKRST_LOCKFLAGS) 0);
    if (!vbData) {
        return FALSE;
    }

    CKDWORD currentOffset = 0;

    // Copy direct vertices first (if any)
    if (hasDirectVertices) {
        CKDWORD count = m_Vertices.Size();
        vbData = CKRSTLoadVertexBuffer(vbData, vertexFormat, vertexSize, data);
        currentOffset += count;
    }

    // Copy remapped vertices for each material group
    if (hasRemappedVertices) {
        VxDrawPrimitiveData localData;
        memcpy(&localData, data, sizeof(VxDrawPrimitiveData));

        for (int i = 0; i < m_MaterialGroups.Size(); i++) {
            CKMaterialGroup *group = m_MaterialGroups[i];
            if (group && group->m_Material) {
                if (group->m_RemapData) {
                    // This group has remapped vertices
                    CKVBuffer *vb = GetVBuffer(group);
                    if (!vb || group->m_VertexCount <= 0) {
                        group->m_BaseVertex = 0;
                        continue;
                    }

                    // Ensure remapped data is up to date before uploading to HW VB.
                    vb->Update(this, 0);

                    if (vb->m_Vertices.Size() < (int) group->m_VertexCount ||
                        vb->m_Colors.Size() < (int) group->m_VertexCount) {
                        group->m_BaseVertex = 0;
                        continue;
                    }

                    VxVertex *remappedVerts = vb->m_Vertices.Begin();
                    VxColors *remappedColors = vb->m_Colors.Begin();

                    // Update local data for this group
                    localData.VertexCount = group->m_VertexCount;
                    localData.PositionPtr = &remappedVerts[0].m_Position;
                    localData.NormalPtr = &remappedVerts[0].m_Normal;
                    localData.TexCoordPtr = &remappedVerts[0].m_UV;
                    localData.ColorPtr = &remappedColors[0].Color;
                    localData.SpecularColorPtr = &remappedColors[0].Specular;

                    // Set up extra tex coords from active channel list
                    for (int t = 0; t < (int) m_ActiveTextureChannels.Size(); t++) {
                        const int channelIdx = m_ActiveTextureChannels[t];
                        Vx2DVector *channelUVs = nullptr;
                        if (channelIdx >= 0 && channelIdx < vb->m_UVs.Size()) {
                            if (vb->m_UVs[channelIdx].Size() == (int) group->m_VertexCount)
                                channelUVs = vb->m_UVs[channelIdx].Begin();
                        }
                        localData.TexCoordPtrs[t] = channelUVs;
                        localData.TexCoordStrides[t] = 8;
                    }

                    // Set the start vertex offset for this group
                    group->m_BaseVertex = currentOffset;
                    currentOffset += group->m_VertexCount;

                    // Load this group's vertices into the buffer
                    vbData = CKRSTLoadVertexBuffer(vbData, vertexFormat, vertexSize, &localData);
                } else {
                    // No remapped vertices for this group
                    group->m_BaseVertex = 0;
                }
            }
        }
    }

    rst->UnlockVertexBuffer(m_VertexBuffer);

    // Check if we should create index buffer
    RCKRenderManager *rm = (RCKRenderManager *) m_Context->GetRenderManager();
    if (rm->m_UseIndexBuffers.Value) {
        CheckHWIndexBuffer(rst);
    }

    return TRUE;
}

//--------------------------------------------
// CheckHWIndexBuffer - Check and setup hardware index buffer
// Returns TRUE if hardware IB was created/updated successfully
//--------------------------------------------
CKBOOL RCKMesh::CheckHWIndexBuffer(CKRasterizerContext *rst) {
    if (!rst)
        return FALSE;

    // Check if rasterizer supports index buffers
    if (!(rst->m_Driver->m_3DCaps.CKRasterizerSpecificCaps & CKRST_SPECIFICCAPS_CANDOINDEXBUFFER))
        return FALSE;

    CKBOOL needNewBuffer = FALSE;
    CKBOOL needUpdate = FALSE;
    CKDWORD totalIndexCount = 0;

    // Count total indices needed and check if any group needs update
    for (int i = 0; i < m_MaterialGroups.Size(); i++) {
        CKMaterialGroup *group = m_MaterialGroups[i];
        if (group && group->m_Material) {
            // Iterate through primitive entries
            for (int p = 0; p < group->m_Primitives.Size(); p++) {
                CKPrimitiveEntry *prim = &group->m_Primitives[p];
                totalIndexCount += prim->m_Indices.Size();
                if (prim->m_IndexBufferOffset < 0) {
                    needUpdate = TRUE;
                }
            }
        }
    }

    // Check if existing index buffer is sufficient
    CKIndexBufferDesc *ibDesc = rst->GetIndexBufferData(m_IndexBuffer);
    CKBOOL needResize = !ibDesc || ibDesc->m_MaxIndexCount < totalIndexCount;

    if (needResize) {
        rst->DeleteObject(m_IndexBuffer, CKRST_OBJ_INDEXBUFFER);

        CKIndexBufferDesc newDesc;
        newDesc.m_Flags = CKRST_VB_VALID | CKRST_VB_WRITEONLY | CKRST_VB_SHARED; 
        newDesc.m_MaxIndexCount = totalIndexCount;

        if (!rst->CreateObject(m_IndexBuffer, CKRST_OBJ_INDEXBUFFER, &newDesc)) {
            // Failed to create index buffer - mark all primitives as software
            for (int i = 0; i < m_MaterialGroups.Size(); i++) {
                CKMaterialGroup *group = m_MaterialGroups[i];
                if (group && group->m_Material) {
                    for (int p = 0; p < group->m_Primitives.Size(); p++) {
                        group->m_Primitives[p].m_IndexBufferOffset = -1;
                    }
                }
            }
            return FALSE;
        }
        needUpdate = TRUE;
    }

    if (!needUpdate)
        return TRUE;

    // Lock and fill index buffer
    CKWORD *ibData = (CKWORD *) rst->LockIndexBuffer(m_IndexBuffer, 0, totalIndexCount, CKRST_LOCK_DISCARD);
    if (!ibData) {
        // Failed to lock - mark all primitives as software
        for (int i = 0; i < m_MaterialGroups.Size(); i++) {
            CKMaterialGroup *group = m_MaterialGroups[i];
            if (group && group->m_Material) {
                for (int p = 0; p < group->m_Primitives.Size(); p++) {
                    group->m_Primitives[p].m_IndexBufferOffset = -1;
                }
            }
        }
        return FALSE;
    }

    CKDWORD currentOffset = 0;

    // Copy indices for each material group
    for (int i = 0; i < m_MaterialGroups.Size(); i++) {
        CKMaterialGroup *group = m_MaterialGroups[i];
        if (group && group->m_Material) {
            for (int p = 0; p < group->m_Primitives.Size(); p++) {
                CKPrimitiveEntry *prim = &group->m_Primitives[p];

                // Set the index buffer offset for this primitive
                prim->m_IndexBufferOffset = currentOffset;

                // Copy indices
                CKDWORD indexCount = prim->m_Indices.Size();
                if (indexCount > 0) {
                    memcpy(ibData, prim->m_Indices.Begin(), indexCount * sizeof(CKWORD));
                    ibData += indexCount;
                    currentOffset += indexCount;
                }
            }
        }
    }

    rst->UnlockIndexBuffer(m_IndexBuffer);

    return TRUE;
}

// IDA: 0x100268d9
// Pick2D tests if a 2D screen point intersects any lines in the mesh
CKBOOL RCKMesh::Pick2D(const Vx2DVector &pt, VxIntersectionDesc *desc, RCKRenderContext *rc, RCK3dEntity *ent) {
    int vertexCount = m_Vertices.Size();
    if (vertexCount == 0)
        return FALSE;

    int lineCount = GetLineCount();
    if (lineCount == 0)
        return FALSE;

    // Transform all vertices to screen space
    XClassArray<VxVector4> screenVertices;
    screenVertices.Resize(vertexCount);

    VxTransformData transformData;
    transformData.ClipFlags = 0;
    transformData.InStride = sizeof(VxVertex);
    transformData.InVertices = m_Vertices.Begin();
    transformData.OutStride = 0;
    transformData.OutVertices = nullptr;
    transformData.ScreenStride = sizeof(VxVector4);
    transformData.ScreenVertices = screenVertices.Begin();

    rc->TransformVertices(vertexCount, &transformData, (CK3dEntity *) ent);

    // Initialize search threshold
    float threshold = 100.0f;

    // Initialize result if provided
    // Match IDA at 0x100269dd
    if (desc) {
        desc->IntersectionPoint = VxVector(0, 0, 0);
        desc->IntersectionNormal = VxVector(0, 0, 0);
        desc->TexU = 0.0f;
        desc->TexV = 0.0f;
        desc->Distance = -1.0f;
        desc->FaceIndex = 0;
    }

    // Get line indices pointer
    CKWORD *lineIndices = m_LineIndices.Begin();

    // Test each line
    for (int i = 0; i < lineCount; ++i) {
        CKWORD idx0 = lineIndices[0];
        CKWORD idx1 = lineIndices[1];

        // Get screen positions of line endpoints
        float x0 = screenVertices[idx0].x;
        float y0 = screenVertices[idx0].y;
        float x1 = screenVertices[idx1].x;
        float y1 = screenVertices[idx1].y;

        // Line direction vector
        float dx = x1 - x0;
        float dy = y1 - y0;

        // Vector from first endpoint to pick point
        float px = pt.x - x0;
        float py = pt.y - y0;

        // Project pick point onto line
        float dot1 = dx * px + dy * py;
        float dot2 = (pt.x - x1) * dx + (pt.y - y1) * dy;

        // Check if projection falls between endpoints
        if (dot1 >= 0.0f && dot2 < 0.0f) {
            // Calculate squared perpendicular distance
            float cross = dx * py - dy * px;
            float lengthSq = dx * dx + dy * dy;
            float distSq = (cross * cross) / lengthSq;

            // Check if within threshold
            if (distSq <= threshold) {
                // Calculate interpolation parameter with perspective correction
                float t = dot1 / lengthSq;

                float w0 = screenVertices[idx0].w;
                float w1 = screenVertices[idx1].w;
                float invW0 = 1.0f / w0;
                float invW1 = 1.0f / w1;

                // Perspective-correct interpolation
                t = (t * invW0) / ((invW0 - invW1) * t + invW1);

                // Get world positions of line endpoints
                VxVertex *v0 = &m_Vertices[idx0];
                VxVertex *v1 = &m_Vertices[idx1];

                // Interpolate intersection point in object space
                VxVector interpPt;
                interpPt.x = v0->m_Position.x + t * (v1->m_Position.x - v0->m_Position.x);
                interpPt.y = v0->m_Position.y + t * (v1->m_Position.y - v0->m_Position.y);
                interpPt.z = v0->m_Position.z + t * (v1->m_Position.z - v0->m_Position.z);

                // Store result - IDA only writes IntersectionPoint (via Sprite field offset)
                // Object is left unchanged (not initialized by this function)
                if (desc) {
                    desc->IntersectionPoint = interpPt;
                }

                return TRUE;
            }
        }

        lineIndices += 2;
    }

    return FALSE;
}
