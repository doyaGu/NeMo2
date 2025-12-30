#ifndef VERTEXCACHEOPTIMIZER_H
#define VERTEXCACHEOPTIMIZER_H

#include "XArray.h"
#include "XBitArray.h"
#include "XClassArray.h"
#include "CKTypes.h"

// ============================================================================
// VertexCacheOptimizer - Reorders triangle indices for better cache utilization
// Binary layout: 88 bytes total
// Used by RCKRenderManager to optimize triangle lists for vertex cache
// ============================================================================
class VertexCacheOptimizer {
public:
    // Internal struct for tracking faces to process
    // Binary layout: 12 bytes
    struct CacheFaces {
        int FaceIndex;      // +0x00: Face index in the triangle list
        int HitCount;       // +0x04: Number of cache hits for this face
        int MinVertexIndex; // +0x08: Minimum vertex index referenced by face

        CacheFaces() : FaceIndex(0), HitCount(0), MinVertexIndex(0x7FFFFFFF) {}
    };

    // Binary layout members:
    XClassArray<XArray<int>> m_VertexFaceLists; // +0x00: Per-vertex list of face indices
    XArray<int> m_CacheEntries;                 // +0x0C: LRU cache entries (vertex indices)
    XBitArray m_CacheSet;                       // +0x18: Bit array for vertices in cache
    XBitArray m_ProcessedFaces;                 // +0x20: Bit array for processed faces
    XArray<CacheFaces> m_ActiveFaces;           // +0x28: Currently active faces
    XArray<CacheFaces *> m_VertexFaceMap;       // +0x34: Per-face pointer to active CacheFaces
    XArray<CKWORD> m_OutputIndices;             // +0x40: Output index array
    int m_ProcessedCount;                       // +0x4C: Number of faces processed
    int m_FaceCount;                            // +0x50: Total face count
    int m_CacheSize;                            // +0x54: Vertex cache size

    VertexCacheOptimizer();
    ~VertexCacheOptimizer();

    // Initialize the optimizer with mesh data
    void Initialize(int vertexCount, int faceCount, int cacheSize);

    // Build per-vertex face lists from indices
    void BuildVertexFaceLists(const XArray<CKWORD> &indices);

    // Process all faces and generate optimized output
    void ProcessFaces(const XArray<CKWORD> &indices);

    // Get the optimized output indices
    XArray<CKWORD> &GetOutputIndices() { return m_OutputIndices; }

    // Swap output indices with target array
    void SwapOutput(XArray<CKWORD> &target) { m_OutputIndices.Swap(target); }

    // Binary: sub_1002B150 - Post-stripify cache metric
    // Called after a primitive is converted to VX_TRIANGLESTRIP.
    // Returns a float metric (unused by the caller in CreateRenderGroups).
    float ComputeCacheMetric(const XArray<CKWORD> &indices, int cacheSize, int mode);

private:
    // Binary: TouchFace (0x1053c) - Add face to output with cache tracking
    void TouchFace(int faceIndex, const XArray<CKWORD> &indices);

    // Binary: TouchFaceUncached (0x1080c) - Add face to output without cache scoring
    void TouchFaceUncached(int faceIndex, const XArray<CKWORD> &indices);

    // Binary: AddCacheEntry (0x109cc) - Add vertex to cache
    void AddCacheEntry(int vertexIndex);

    // Binary: ComputeFaceCost2 (0x10bfc) - Compute cost of adding face
    int ComputeFaceCost2(const XArray<CKWORD> &indices, int faceIndex);

    // non-copyable
    VertexCacheOptimizer(const VertexCacheOptimizer &);
    VertexCacheOptimizer &operator=(const VertexCacheOptimizer &);
};

#endif // VERTEXCACHEOPTIMIZER_H