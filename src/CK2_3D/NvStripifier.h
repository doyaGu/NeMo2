#ifndef NVSTRIPIFIER_H
#define NVSTRIPIFIER_H

#include "XArray.h"
#include "CKTypes.h"
#include "VertexCache.h"

// Binary-faithful NvStripifier reimplementation.
//
// Binary API:
// - Stripify() produces strip objects (NvStripInfo).
// - CreateStrips() emits a single index stream from those strips.
//   - joinStrips=true: strips are connected with degenerate triangles
//   - joinStrips=false: strips separated by -1 (0xFFFF), count returned

// ============================================================================
// NvFaceInfo - Binary layout: 0x24 bytes
// ============================================================================
struct NvFaceInfo {
    int V[3];                // +0x00: Triangle vertex indices
    NvFaceInfo *Neighbor[3]; // +0x0C: Adjacent faces for each edge
    int MarkA;               // +0x18: Committed strip mark
    int MarkB;               // +0x1C: Experiment strip mark
    int Experiment;          // +0x20: Current experiment ID

    NvFaceInfo() {
        V[0] = V[1] = V[2] = -1;
        Neighbor[0] = Neighbor[1] = Neighbor[2] = nullptr;
        MarkA = -1;
        MarkB = -1;
        Experiment = -1;
    }
};

// ============================================================================
// NvEdgeInfo - Binary layout: 0x1C bytes
// ============================================================================
struct NvEdgeInfo {
    int RefCount;       // +0x00: Reference count (initialized to 2)
    NvFaceInfo *Face0;  // +0x04: First face
    NvFaceInfo *Face1;  // +0x08: Second face
    int Vertex0;        // +0x0C: First vertex
    int Vertex1;        // +0x10: Second vertex
    NvEdgeInfo *NextV0; // +0x14: Next edge in Vertex0's bucket
    NvEdgeInfo *NextV1; // +0x18: Next edge in Vertex1's bucket

    NvEdgeInfo() {
        RefCount = 2;
        Face0 = Face1 = nullptr;
        Vertex0 = Vertex1 = -1;
        NextV0 = NextV1 = nullptr;
    }
};

// ============================================================================
// NvStripInfo - Binary layout: 0x24 bytes
// ============================================================================
struct NvStripInfo {
    NvFaceInfo *StartFace; // +0x00: Starting face
    NvEdgeInfo *StartEdge; // +0x04: Starting edge
    unsigned char StartCW; // +0x08: Starting winding (CW=1, CCW=0)
    // padding 3 bytes at +0x09
    XArray<NvFaceInfo *> Faces; // +0x0C: XArray (3 DWORDs: begin, end, cap)
    int StripId;                // +0x18: Strip ID (unique identifier)
    int ExperimentId;           // +0x1C: Experiment ID (-1 if committed)
    unsigned char field_20;     // +0x20: Used in SplitUpStripsAndOptimize

    NvStripInfo() {
        StartFace = nullptr;
        StartEdge = nullptr;
        StartCW = 0;
        StripId = 0;
        ExperimentId = -1;
        field_20 = 0;
    }

    // Binary: NvStripInfo::MarkTriangle (0xC4)
    void MarkTriangle(NvFaceInfo *face) {
        if (ExperimentId < 0) {
            face->Experiment = -1;
            face->MarkA = StripId;
        } else {
            face->Experiment = ExperimentId;
            face->MarkB = StripId;
        }
    }

    // Binary: NvStripInfo::Build (0x834)
    // CRITICAL: Uses STATIC local variables
    void Build(XArray<NvEdgeInfo *> &edgeBuckets, XArray<NvFaceInfo *> &allFaces);

    // Binary: NvStripInfo::Combine (0x11F4)
    void Combine(XArray<NvFaceInfo *> &forward, XArray<NvFaceInfo *> &backward);
};

// ============================================================================
// NvStripifier class layout (from binary analysis):
// +0x00: XArray<unsigned short> m_Scratch (3 DWORDs: begin, end, capacity)
// +0x0C: int m_MinStripLength
// +0x10: int m_CacheSize
// +0x14: float m_Ratio
// +0x18: unsigned char m_FirstTime
// ============================================================================
class NvStripifier {
public:
    NvStripifier();
    ~NvStripifier();

    // Allow NvStripInfo to access private static helpers
    friend struct NvStripInfo;

    // Binary API: void __thiscall Stripify(XArray<CKWORD>*, int minStripLen, int cacheSize, CKWORD vertexCount, XArray<NvStripInfo*>*)
    void Stripify(
        const XArray<CKWORD> &inIndices,
        int minStripLength,
        int cacheSize,
        CKWORD vertexCount,
        XArray<NvStripInfo *> &outStrips);

    // Binary API: void __stdcall CreateStrips(XArray<NvStripInfo*>*, XArray<CKWORD>*, bool joinStrips, CKDWORD* outStripCount)
    // joinStrips=true: output continuous strip flow (degenerate triangles connect strips)
    // joinStrips=false: strips separated by -1 (0xFFFF), strip count written to outStripCount
    static void CreateStrips(
        const XArray<NvStripInfo *> &strips,
        XArray<CKWORD> &outIndices,
        bool joinStrips,
        CKDWORD &outStripCount);

    // Convenience one-shot API (not in original binary, but useful wrapper)
    void Stripify(
        const XArray<CKWORD> &inIndices,
        int minStripLength,
        int cacheSize,
        CKWORD vertexCount,
        bool joinStrips,
        XArray<CKWORD> &outIndices,
        CKDWORD &outStripCount);

    // Cleanup method for allocated strips
    static void DestroyStrips(XArray<NvStripInfo *> &strips);

private:
    // Binary layout: XArray<unsigned short> at +0x00
    XArray<CKWORD> m_Scratch;  // +0x00 (12 bytes)
    int m_MinStripLength;      // +0x0C
    int m_CacheSize;           // +0x10
    float m_Ratio;             // +0x14
    unsigned char m_FirstTime; // +0x18

    // Internal helpers (static in binary)
    static void BuildStripifyInfo(
        const XArray<CKWORD> &indices,
        CKWORD vertexCount,
        XArray<NvFaceInfo *> &outFaces,
        XArray<NvEdgeInfo *> &outEdgeBuckets);

    static bool FindAllStrips(
        XArray<NvStripInfo *> &outAllStrips,
        XArray<NvFaceInfo *> &allFaces,
        XArray<NvEdgeInfo *> &edgeBuckets,
        int numSamples,
        int minStripLength);

    static void SplitUpStripsAndOptimize(
        XArray<NvStripInfo *> &allStrips,
        bool joinStrips,
        XArray<NvEdgeInfo *> &edgeBuckets);

    static NvFaceInfo *FindGoodResetPoint(
        XArray<NvFaceInfo *> &allFaces,
        XArray<NvEdgeInfo *> &edgeBuckets);

    static void CommitStrips(
        XArray<NvStripInfo *> &outStrips,
        XArray<NvStripInfo *> *candidateSet);

    static void RemoveSmallStrips(
        XArray<NvStripInfo *> &allStrips,
        XArray<NvFaceInfo *> &allFaces,
        XArray<NvStripInfo *> &outStrips,
        XArray<NvEdgeInfo *> &edgeBuckets);

    static NvEdgeInfo *FindEdgeInfo(
        XArray<NvEdgeInfo *> &edgeBuckets,
        int v0,
        int v1);

    static NvFaceInfo *FindOtherFace(
        XArray<NvEdgeInfo *> &edgeBuckets,
        int v0,
        int v1,
        NvFaceInfo *notThisFace);

    static int FindStartPoint(
        XArray<NvFaceInfo *> &allFaces,
        XArray<NvEdgeInfo *> &edgeBuckets);

    static bool FindTraversal(
        XArray<NvFaceInfo *> &allFaces,
        XArray<NvEdgeInfo *> &edgeBuckets,
        NvStripInfo *strip,
        NvFaceInfo *&outFace,
        NvEdgeInfo *&outEdge,
        bool &outCW);

    static bool NextIsCW(int numIndices);

    static int GetUniqueVertexInB(NvFaceInfo *a, NvFaceInfo *b);
    static int GetSharedVertex(NvFaceInfo *a, NvFaceInfo *b);

    static bool AlreadyExists(NvFaceInfo *face, const XArray<NvFaceInfo *> &faces);

    static int NumNeighbors(NvFaceInfo *face, XArray<NvEdgeInfo *> &edgeBuckets);

    static void UpdateCacheStrip(VertexCache *cache, NvStripInfo *strip);
    static void UpdateCacheFace(VertexCache *cache, NvFaceInfo *face);
    static float CalcNumHitsStrip(VertexCache *cache, NvStripInfo *strip);
    static int CalcNumHitsFace(VertexCache *cache, NvFaceInfo *face);
    static float AvgStripSize(const XArray<NvStripInfo *> &strips);

    // non-copyable
    NvStripifier(const NvStripifier &);
    NvStripifier &operator=(const NvStripifier &);
};

#endif // NVSTRIPIFIER_H
