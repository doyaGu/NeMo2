#ifndef CKRENDERENGINETYPES_H
#define CKRENDERENGINETYPES_H

#include "VxDefines.h"
#include "VxColor.h"
#include "VxVector.h"
#include "Vx2dVector.h"
#include "VxFrustum.h"
#include "XArray.h"
#include "XClassArray.h"
#include "CKTypes.h"
#include "CKRasterizerTypes.h"

class CKRenderContext;

class RCKRenderManager;
class RCKRenderContext;
class RCKKinematicChain;
class RCKMaterial;
class RCKTexture;
class RCKMesh;
class RCKPatchMesh;
class RCKAnimation;
class RCKKeyedAnimation;
class RCKObjectAnimation;
class RCKLayer;
class RCKRenderObject;
class RCK2dEntity;
class RCK3dEntity;
class RCKCamera;
class RCKLight;
class RCKCurvePoint;
class RCKCurve;
class RCK3dObject;
class RCKSprite3D;
class RCKCharacter;
class RCKPlace;
class RCKGrid;
class RCKBodyPart;
class RCKTargetCamera;
class RCKTargetLight;
class RCKSprite;
class RCKSpriteText;

struct VxCallBack {
    void *callback; // 0x00 (4 bytes)
    void *argument; // 0x04 (4 bytes)
    union {
        CKBOOL temp;    // 0x08 (4 bytes) - normal callbacks (0/1)
        void *arg2;     // 0x08 (4 bytes) - temporary callback arrays store the Argument pointer here
    };
    // Total: 12 bytes - NO beforeTransparent field per IDA
};

class CKCallbacksContainer {
public:
    CKCallbacksContainer() : m_Callback(nullptr) {}
    ~CKCallbacksContainer();

    CKBOOL AddPreCallback(void *Function,
                          void *Argument,
                          CKBOOL Temporary,
                          CKRenderManager *renderManager);
    CKBOOL RemovePreCallback(void *Function, void *Argument);

    CKBOOL SetCallBack(void *Function, void *Argument);
    CKBOOL RemoveCallBack();

    CKBOOL AddPostCallback(void *Function,
                           void *Argument,
                           CKBOOL Temporary,
                           CKRenderManager *renderManager);
    CKBOOL RemovePostCallback(void *Function, void *Argument);

    void Clear();
    void ClearPreCallbacks();
    void ClearPostCallbacks();

    void ExecutePreCallbacks(CKRenderContext *dev, CKBOOL temporaryOnly = FALSE);
    void ExecutePostCallbacks(CKRenderContext *dev, CKBOOL temporaryOnly = FALSE);

    // Match IDA CKCallbacksContainer layout (28 bytes total)
    XClassArray<VxCallBack> m_PreCallBacks;  // 0x00, size 12
    VxCallBack *m_Callback;                  // 0x0C, size 4
    XClassArray<VxCallBack> m_PostCallBacks; // 0x10, size 12
};

/**
 * @brief Internal vertex structure for progressive mesh edge collapse algorithm.
 *
 * Size: 0x38 (56 bytes) - matches IDA at sub_1002AA90
 *
 * IDA Layout:
 * 0x00: VxVector position (12 bytes)
 * 0x0C: int originalIndex (4 bytes)
 * 0x10: XArray<PMVertexEx*> neighbors (12 bytes)
 * 0x1C: XArray<PMFace*> faces (12 bytes)
 * 0x28: PMVertexEx* collapseTarget (4 bytes)
 * 0x2C: float collapseCost (4 bytes)
 * 0x30: int heapIndex (4 bytes)
 * 0x34: int flags (4 bytes) - set to -1 when removed
 */
struct PMVertexEx;
struct PMFace;

struct PMVertexEx {
    VxVector position;              // 0x00: Vertex position
    int originalIndex;              // 0x0C: Original vertex index in mesh
    XArray<PMVertexEx *> neighbors; // 0x10: Adjacent vertices
    XArray<PMFace *> faces;         // 0x1C: Faces using this vertex
    PMVertexEx *collapseTarget;     // 0x28: Target vertex for edge collapse
    float collapseCost;             // 0x2C: Cost of collapsing to target
    int arrayIndex;                 // 0x30: Index in vertex array (IDA: set by sub_10024297)
    int heapIndex;                  // 0x34: Index in priority heap (IDA: set to -1 by sub_10024690)

    PMVertexEx() : originalIndex(-1), collapseTarget(nullptr), collapseCost(0.0f), arrayIndex(-1), heapIndex(-1) {
        position.x = position.y = position.z = 0.0f;
    }

        PMVertexEx(const VxVector &pos, int index)
                : position(pos),
                    originalIndex(index),
                    collapseTarget(nullptr),
                    collapseCost(0.0f),
                    arrayIndex(-1),
                    heapIndex(-1) {}

    bool HasNeighbor(PMVertexEx *v) const {
        for (PMVertexEx **it = neighbors.Begin(); it != neighbors.End(); ++it) {
            if (*it == v) return true;
        }
        return false;
    }

    void AddNeighbor(PMVertexEx *v) {
        if (!HasNeighbor(v) && v != this) {
            neighbors.PushBack(v);
        }
    }

    void RemoveNeighbor(PMVertexEx *v) {
        for (int i = 0; i < neighbors.Size(); ++i) {
            if (neighbors[i] == v) {
                neighbors.RemoveAt(i);
                return;
            }
        }
    }

    // IDA: RCKMesh::ProgressiveMeshData::VertexEx::RemoveIfNonNeighbor
    static void RemoveIfNonNeighbor(PMVertexEx *v, PMVertexEx *other);
};

/**
 * @brief Internal face structure for progressive mesh edge collapse algorithm.
 *
 * Size: 0x20 (32 bytes) - matches IDA at sub_10023577
 *
 * IDA Layout:
 * 0x00: PMVertexEx* vertices[3] (12 bytes)
 * 0x0C: VxVector normal (12 bytes)
 * 0x18: void* material (4 bytes)
 * 0x1C: int arrayIndex (4 bytes)
 */
struct PMFace {
    PMVertexEx *vertices[3]; // 0x00: Three vertices of face
    VxVector normal;           // 0x0C: Face normal
    void *material;            // 0x18: Material pointer (optional)
    int arrayIndex;            // 0x1C: Index in face array

    PMFace() : vertices(), material(nullptr), arrayIndex(-1) {
        vertices[0] = vertices[1] = vertices[2] = nullptr;
        normal.x = normal.y = normal.z = 0.0f;
    }

    PMFace(PMVertexEx *v0, PMVertexEx *v1, PMVertexEx *v2, const VxVector &n, void *mat)
        : vertices(), normal(n), material(mat), arrayIndex(-1) {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
    }

    // IDA: sub_1002364A + scalar deleting destructor
    ~PMFace() {
        // Remove this face from each vertex's face list.
        for (int i = 0; i < 3; ++i) {
            PMVertexEx *v = vertices[i];
            if (!v)
                continue;

            for (int fi = 0; fi < v->faces.Size(); ++fi) {
                if (v->faces[fi] == this) {
                    v->faces.RemoveAt(fi);
                    break;
                }
            }
        }

        // Potentially remove neighbor relationships if this face was the last adjacency.
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3;
            PMVertexEx *a = vertices[i];
            PMVertexEx *b = vertices[j];
            if (a && b) {
                PMVertexEx::RemoveIfNonNeighbor(a, b);
                PMVertexEx::RemoveIfNonNeighbor(b, a);
            }
        }
    }

    bool ContainsVertex(PMVertexEx *v) const {
        return vertices[0] == v || vertices[1] == v || vertices[2] == v;
    }

    void ReplaceVertex(PMVertexEx *oldV, PMVertexEx *newV) {
        for (int i = 0; i < 3; ++i) {
            if (vertices[i] == oldV) {
                vertices[i] = newV;
                return;
            }
        }
    }

    void ComputeNormal() {
        if (!vertices[0] || !vertices[1] || !vertices[2]) return;
        VxVector e1, e2;
        e1.x = vertices[1]->position.x - vertices[0]->position.x;
        e1.y = vertices[1]->position.y - vertices[0]->position.y;
        e1.z = vertices[1]->position.z - vertices[0]->position.z;
        e2.x = vertices[2]->position.x - vertices[0]->position.x;
        e2.y = vertices[2]->position.y - vertices[0]->position.y;
        e2.z = vertices[2]->position.z - vertices[0]->position.z;
        // Cross product
        normal.x = e1.y * e2.z - e1.z * e2.y;
        normal.y = e1.z * e2.x - e1.x * e2.z;
        normal.z = e1.x * e2.y - e1.y * e2.x;
        // Normalize
        float len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (len >= EPSILON) {
            normal.x /= len;
            normal.y /= len;
            normal.z /= len;
        }
    }
};

// Out-of-line definition to avoid incomplete-type use before PMFace is defined.
inline void PMVertexEx::RemoveIfNonNeighbor(PMVertexEx *v, PMVertexEx *other) {
    if (!v || !other)
        return;

    // If any face of v still references other, they remain neighbors.
    for (PMFace **fit = v->faces.Begin(); fit != v->faces.End(); ++fit) {
        PMFace *f = *fit;
        if (f && f->ContainsVertex(other))
            return;
    }

    v->RemoveNeighbor(other);
}

/**
 * @brief Internal edge collapse data structure for progressive mesh algorithm.
 *
 * Size: 0x24 (36 bytes) - matches IDA at sub_10029630
 *
 * IDA Layout:
 * 0x00: XArray<PMVertexEx*> vertices (12 bytes)
 * 0x0C: XArray<PMVertexEx*> heap (12 bytes) - min-heap for collapse order
 * 0x18: XArray<PMFace*> faces (12 bytes)
 */
struct PMEdgeCollapseData {
    XArray<PMVertexEx *> vertices; // 0x00: All vertices
    XArray<PMVertexEx *> heap;     // 0x0C: Min-heap by collapse cost
    XArray<PMFace *> faces;        // 0x18: All faces

    PMEdgeCollapseData() {}

    ~PMEdgeCollapseData() {
        // Clean up allocated vertices and faces
        for (PMVertexEx **it = vertices.Begin(); it != vertices.End(); ++it) {
            delete *it;
        }
        for (PMFace **it = faces.Begin(); it != faces.End(); ++it) {
            delete *it;
        }
    }

    // Min-heap operations
    float GetCost(int index) const {
        if (index >= 0 && index < heap.Size() && heap[index]) {
            return heap[index]->collapseCost;
        }
        return 9.9999998e12f; // Very large value - matches IDA
    }

    void HeapifyUp(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (GetCost(index) >= GetCost(parent)) break;

            // Swap
            PMVertexEx *tmp = heap[index];
            heap[index] = heap[parent];
            heap[parent] = tmp;

            // Update heap indices
            if (heap[index]) heap[index]->heapIndex = index;
            if (heap[parent]) heap[parent]->heapIndex = parent;

            index = parent;
        }
    }

    void HeapifyDown(int index) {
        while (true) {
            int left = 2 * index + 1;
            int right = 2 * index + 2;

            float costSelf = GetCost(index);
            float costLeft = GetCost(left);
            float costRight = GetCost(right);

            if (costLeft >= costSelf && costRight >= costSelf) break;

            int smallest = (costLeft < costRight) ? left : right;

            // Swap
            PMVertexEx *tmp = heap[index];
            heap[index] = heap[smallest];
            heap[smallest] = tmp;

            // Update heap indices
            if (heap[index]) heap[index]->heapIndex = index;
            if (heap[smallest]) heap[smallest]->heapIndex = smallest;

            index = smallest;
        }
    }

    void AddToHeap(PMVertexEx *v) {
        const int index = heap.Size();
        heap.PushBack(v);
        v->heapIndex = index;
        HeapifyUp(index);
    }

    // IDA: sub_10024690
    // Note: does NOT shrink the heap array; it inserts a null hole and heapifies it down.
    PMVertexEx *PopMinFromHeap() {
        if (heap.Size() == 0)
            return nullptr;

        PMVertexEx *result = heap[0];
        if (!result)
            return nullptr;

        result->heapIndex = -1;
        heap[0] = nullptr;
        HeapifyDown(0);
        return result;
    }

    void RemoveFromHeap(PMVertexEx *v) {
        if (v->heapIndex < 0 || v->heapIndex >= heap.Size()) return;

        int idx = v->heapIndex;
        int lastIdx = heap.Size() - 1;

        if (idx != lastIdx) {
            heap[idx] = heap[lastIdx];
            if (heap[idx]) heap[idx]->heapIndex = idx;
        }
        heap.RemoveAt(lastIdx);
        v->heapIndex = -1;

        if (idx < heap.Size()) {
            HeapifyUp(idx);
            HeapifyDown(idx);
        }
    }

    void UpdateHeapPosition(PMVertexEx *v) {
        if (v->heapIndex < 0) return;
        HeapifyUp(v->heapIndex);
        HeapifyDown(v->heapIndex);
    }

    // IDA: sub_100242FB (remove vertex by swapping last into removed slot)
    void RemoveVertex(PMVertexEx *v) {
        if (!v)
            return;
        if (v->arrayIndex < 0 || v->arrayIndex >= vertices.Size())
            return;

        const int idx = v->arrayIndex;
        const int lastIdx = vertices.Size() - 1;
        if (idx != lastIdx) {
            vertices[idx] = vertices[lastIdx];
            if (vertices[idx])
                vertices[idx]->arrayIndex = idx;
        }
        vertices.RemoveAt(lastIdx);
        delete v;
    }

    void RemoveFace(PMFace *f) {
        if (f->arrayIndex < 0 || f->arrayIndex >= faces.Size()) return;

        int idx = f->arrayIndex;
        int lastIdx = faces.Size() - 1;

        if (idx != lastIdx) {
            faces[idx] = faces[lastIdx];
            if (faces[idx]) faces[idx]->arrayIndex = idx;
        }
        faces.RemoveAt(lastIdx);
        f->arrayIndex = -1;
    }

    // IDA: sub_10024375 (remove face by swapping last into removed slot and deleting)
    void DeleteFace(PMFace *f) {
        if (!f)
            return;
        if (f->arrayIndex < 0 || f->arrayIndex >= faces.Size())
            return;

        const int idx = f->arrayIndex;
        const int lastIdx = faces.Size() - 1;
        if (idx != lastIdx) {
            faces[idx] = faces[lastIdx];
            if (faces[idx])
                faces[idx]->arrayIndex = idx;
        }
        faces.RemoveAt(lastIdx);
        delete f;
    }
};

/**
 * @brief Progressive mesh structure for LOD (Level of Detail) support.
 *
 * Used by CKMesh to store progressive mesh data for dynamic level of detail.
 * Size: 0x2C (44 bytes) - matches IDA layout
 *
 * IDA Layout:
 * 0x00: field_0 (m_VertexCount) - vertex count for LOD
 * 0x04: m_MorphEnabled - morph enabled flag
 * 0x08: m_MorphStep - morph step value
 * 0x0C: field_C (XVoidArray, 12 bytes) - internal edge collapse data (runtime only)
 * 0x18: field_18 (int, 4 bytes) - reserved/unused
 * 0x1C: field_1C (int/ptr, 4 bytes) - edge collapse structure pointer (runtime only)
 * 0x20: field_20 (XVoidArray/m_Data, 12 bytes) - parent vertex mapping for file I/O
 */
class CKProgressiveMesh {
public:
    int m_VertexCount;                        // 0x00: Vertex Count for LOD rendering
    int m_MorphEnabled;                       // 0x04: Morph enabled flag
    int m_MorphStep;                          // 0x08: Morph step value
    XArray<CKDWORD> m_EdgeCollapseData;       // 0x0C: Internal edge collapse data (runtime)
    int m_Reserved;                           // 0x18: Reserved field
    PMEdgeCollapseData *m_EdgeCollapseStruct; // 0x1C: Edge collapse structure pointer (runtime)
    XArray<CKDWORD> m_Data;                   // 0x20: Parent vertex mapping (file I/O)

    CKProgressiveMesh() : m_VertexCount(-1),
                          m_MorphEnabled(0),
                          m_MorphStep(0),
                          m_Reserved(0),
                          m_EdgeCollapseStruct(nullptr) {}

    // Copy constructor - matches IDA sub_100298F0
    // Note: Does NOT deep copy m_EdgeCollapseStruct (runtime data only)
    CKProgressiveMesh(const CKProgressiveMesh &other)
        : m_VertexCount(other.m_VertexCount),
          m_MorphEnabled(other.m_MorphEnabled),
          m_MorphStep(other.m_MorphStep),
          m_EdgeCollapseData(other.m_EdgeCollapseData),
          m_Reserved(other.m_Reserved),
          m_EdgeCollapseStruct(nullptr),  // Do not copy runtime pointer
          m_Data(other.m_Data) {}

    // Copy assignment operator
    CKProgressiveMesh &operator=(const CKProgressiveMesh &other) {
        if (this != &other) {
            m_VertexCount = other.m_VertexCount;
            m_MorphEnabled = other.m_MorphEnabled;
            m_MorphStep = other.m_MorphStep;
            m_EdgeCollapseData = other.m_EdgeCollapseData;
            m_Reserved = other.m_Reserved;
            // Do not copy m_EdgeCollapseStruct - it's runtime data
            m_Data = other.m_Data;
        }
        return *this;
    }

    ~CKProgressiveMesh() {
        if (m_EdgeCollapseStruct) {
            delete m_EdgeCollapseStruct;
            m_EdgeCollapseStruct = nullptr;
        }
    }
};

// Forward declaration for progressive mesh callback
class CK3dEntity;
class CKMesh;
void ProgressiveMeshPreRenderCallback(CKRenderContext *ctx, CK3dEntity *entity, CKMesh *mesh, void *data);

struct VxOption {
    CKDWORD Value;
    XString Key;

    void Set(XString &key, CKDWORD &value) {
        Key = key;
        Value = value;
    }

    void Set(const char *key, CKDWORD value) {
        Key = key;
        Value = value;
    }
};

struct VxDriverDescEx {
    CKBOOL CapsUpToDate;
    CKDWORD DriverId;
    char DriverDesc[512];
    char DriverDesc2[512];
    CKBOOL Hardware;
    int DisplayModeCount;
    VxDisplayMode *DisplayModes;
    XSArray<VxImageDescEx> TextureFormats;
    Vx2DCapsDesc Caps2D;
    Vx3DCapsDesc Caps3D;
    CKRasterizer *Rasterizer;
    CKRasterizerDriver *RasterizerDriver;
};

struct VxColors {
    CKDWORD Color;
    CKDWORD Specular;
};

struct VxVertex {
    VxVector m_Position;
    VxVector m_Normal;
    Vx2DVector m_UV;
};

struct CKFace {
    // IDA layout - 16 bytes total
    VxVector m_Normal;    // 0x00: Face normal (12 bytes)
    CKWORD m_MatIndex;    // 0x0C: Material index (2 bytes)
    CKWORD m_ChannelMask; // 0x0E: Channel mask (2 bytes)
    // Note: Vertex indices are stored separately in m_FaceVertexIndices array
};

struct VxMaterialChannel {
    Vx2DVector *m_UVs;              // 0x00 - UV pointer for this channel
    CKMaterial *m_Material;        // 0x04
    VXBLEND_MODE m_SourceBlend;    // 0x08
    VXBLEND_MODE m_DestBlend;      // 0x0C
    CKDWORD m_Flags;               // 0x10
    XArray<CKWORD> *m_FaceIndices; // 0x14 - Per-channel face indices (nullptr if channel applies to all faces)

    VxMaterialChannel()
        : m_UVs(nullptr),
          m_Material(nullptr),
          m_SourceBlend(VXBLEND_ZERO),
          m_DestBlend(VXBLEND_SRCCOLOR),
          m_Flags(1),
          m_FaceIndices(nullptr) {}

    void Clear() {
        if (m_FaceIndices) {
            delete m_FaceIndices;
            m_FaceIndices = nullptr;
        }
        delete[] m_UVs;
        m_UVs = nullptr;
    }
};

struct CKPrimitiveEntry {
    VXPRIMITIVETYPE m_Type;
    XArray<CKWORD> m_Indices;
    CKDWORD m_IndexBufferOffset;
};

// IDA: CKVBuffer (0x30 bytes)
// Stores per-material-group remapped vertex data and index remapping.
struct CKVBuffer {
    XArray<VxVertex> m_Vertices;           // 0x00
    XArray<VxColors> m_Colors;             // 0x0C
    XClassArray<XArray<Vx2DVector>> m_UVs; // 0x18 - per-channel UV arrays
    XArray<int> m_VertexRemap;             // 0x24 - local->global vertex indices

    explicit CKVBuffer(int vertexCount = 0);
    ~CKVBuffer();

    void Resize(int vertexCount);
    void Update(RCKMesh *mesh, int force);
};

struct CKMaterialGroup {
    RCKMaterial *m_Material;                    // 0x00
    XClassArray<CKPrimitiveEntry> m_Primitives; // 0x04 - Primitive entries (size 12)
    XArray<CKWORD> m_FaceIndices;               // 0x10 - Face indices for this group
    CKDWORD m_HasValidPrimitives;               // 0x1C - Flag set when at least one primitive has indices
    CKDWORD m_MinVertexIndex;                   // 0x20 - Smallest vertex index used by the group
    CKDWORD m_MaxVertexIndex;                   // 0x24 - Largest vertex index (exclusive) used by the group
    CKDWORD m_BaseVertex;                       // 0x28 - Base vertex offset in shared buffers
    CKDWORD m_VertexCount;                      // 0x2C - Vertex count for this group
    CKDWORD m_RemapData;                        // 0x30 - Pointer to remap buffer / packed remapped vertices

    CKMaterialGroup()
        : m_Material(nullptr),
          m_HasValidPrimitives(0),
          m_MinVertexIndex(0x10000),
          m_MaxVertexIndex(0),
          m_BaseVertex(0),
          m_VertexCount(0),
          m_RemapData(0) {}

    explicit CKMaterialGroup(CKMaterial *mat)
        : m_Material((RCKMaterial *) mat),
          m_HasValidPrimitives(0),
          m_MinVertexIndex(0x10000),
          m_MaxVertexIndex(0),
          m_BaseVertex(0),
          m_VertexCount(0),
          m_RemapData(0) {}
};

// Note: CKVertex is already defined in CKRasterizerTypes.h (32 bytes)
// It has: VxVector4 V, CKDWORD Diffuse, CKDWORD Specular, float tu, float tv

class RCKMaterial; // Forward declaration

// CKSprite3DBatch - Batch structure for optimized sprite rendering (36 bytes)
// Used to batch multiple sprites using the same material
// NOTE: IDA shows m_Materials at offset 0 but code uses it as indices (XArray<CKWORD>)
struct CKSprite3DBatch {
    XArray<CKWORD> m_Indices;         // 0x00 - Index array (12 bytes)
    XClassArray<CKVertex> m_Vertices; // 0x0C - Vertex array (12 bytes)
    CKDWORD m_VertexCount;            // 0x18 - Current vertex count
    CKDWORD m_IndexCount;             // 0x1C - Max allocated index count
    CKDWORD m_Flags;                  // 0x20 - Flags (e.g., needs clipping)

    CKSprite3DBatch() : m_VertexCount(0), m_IndexCount(0), m_Flags(0) {}
};

#endif // CKRENDERENGINETYPES_H
