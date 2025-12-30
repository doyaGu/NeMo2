// NvStripifier.cpp
// Binary-faithful NvStripifier reimplementation
// Based on IDA analysis of NvTriStripObjects.obj (MD5: b2c4553e1045fe3bf3a222aa328bcefc)

#include "NvStripifier.h"
#include "MeshAdjacency.h"

#include <cstdio>
#include <cstring>

// ============================================================================
// Static helper: NvStripifier::FindEdgeInfo (0x24)
// ============================================================================
NvEdgeInfo *NvStripifier::FindEdgeInfo(XArray<NvEdgeInfo *> &edgeBuckets, int v0, int v1) {
    if (v0 < 0 || v0 >= edgeBuckets.Size())
        return nullptr;

    NvEdgeInfo *edge = edgeBuckets[v0];
    while (edge) {
        const int a0 = edge->Vertex0;
        const int a1 = edge->Vertex1;

        if ((a0 == v0 && a1 == v1) || (a0 == v1 && a1 == v0))
            return edge;

        // Advance along the bucket list for v0.
        if (a0 == v0)
            edge = edge->NextV0;
        else
            edge = edge->NextV1;
    }
    return nullptr;
}

// ============================================================================
// Static helper: NvStripifier::FindOtherFace (0x64)
// ============================================================================
NvFaceInfo *NvStripifier::FindOtherFace(XArray<NvEdgeInfo *> &edgeBuckets, int v0, int v1, NvFaceInfo *notThisFace) {
    // Find the edge, then return the other face
    NvEdgeInfo *edge = FindEdgeInfo(edgeBuckets, v0, v1);
    if (!edge)
        return nullptr;

    // Check if edge matches (v0, v1) or (v1, v0)
    if ((edge->Vertex0 == v0 && edge->Vertex1 == v1) ||
        (edge->Vertex0 == v1 && edge->Vertex1 == v0)) {
        if (edge->Face0 == notThisFace)
            return edge->Face1;
        return edge->Face0;
    }
    return nullptr;
}

// ============================================================================
// Static helper: NvStripifier::AlreadyExists (0x124)
// ============================================================================
bool NvStripifier::AlreadyExists(NvFaceInfo *face, const XArray<NvFaceInfo *> &faces) {
    int count = faces.Size();
    for (int i = 0; i < count; i++) {
        NvFaceInfo *f = faces[i];
        if (f->V[0] == face->V[0] && f->V[1] == face->V[1] && f->V[2] == face->V[2])
            return true;
    }
    return false;
}

// ============================================================================
// Static helper: NvStripifier::NextIsCW (0x1A24)
// ============================================================================
bool NvStripifier::NextIsCW(int numIndices) {
    return (numIndices % 2) == 0;
}

// ============================================================================
// Static helper: NvStripifier::GetUniqueVertexInB
// ============================================================================
int NvStripifier::GetUniqueVertexInB(NvFaceInfo *a, NvFaceInfo *b) {
    // Find the vertex in b that is not in a
    for (int i = 0; i < 3; i++) {
        int v = b->V[i];
        if (v != a->V[0] && v != a->V[1] && v != a->V[2])
            return v;
    }
    return -1;
}

// ============================================================================
// Static helper: NvStripifier::GetSharedVertex
// ============================================================================
int NvStripifier::GetSharedVertex(NvFaceInfo *a, NvFaceInfo *b) {
    // Find a vertex that exists in both faces
    for (int i = 0; i < 3; i++) {
        int v = a->V[i];
        if (v == b->V[0] || v == b->V[1] || v == b->V[2])
            return v;
    }
    return -1;
}

// ============================================================================
// Static helper: NvStripifier::NumNeighbors (0x2CA4)
// ============================================================================
int NvStripifier::NumNeighbors(NvFaceInfo *face, XArray<NvEdgeInfo *> &edgeBuckets) {
    int count = 0;
    if (face->Neighbor[0]) count++;
    if (face->Neighbor[1]) count++;
    if (face->Neighbor[2]) count++;
    return count;
}

// ============================================================================
// Static helper: NvStripifier::FindStartPoint (0x44C)
// ============================================================================
int NvStripifier::FindStartPoint(XArray<NvFaceInfo *> &allFaces, XArray<NvEdgeInfo *> &edgeBuckets) {
    int count = allFaces.Size();
    for (int i = 0; i < count; i++) {
        NvFaceInfo *face = allFaces[i];
        int numNeighbors = 0;
        if (face->Neighbor[0]) numNeighbors++;
        if (face->Neighbor[1]) numNeighbors++;
        if (face->Neighbor[2]) numNeighbors++;

        // Return face with more than 1 boundary edge (less than 2 neighbors)
        if (numNeighbors < 2)
            return i;
    }
    return -1;
}

// ============================================================================
// Static helper: NvStripifier::UpdateCacheStrip (0x29B4)
// ============================================================================
void NvStripifier::UpdateCacheStrip(VertexCache *cache, NvStripInfo *strip) {
    int count = strip->Faces.Size();
    for (int i = 0; i < count; i++) {
        NvFaceInfo *face = strip->Faces[i];
        if (!cache->InCache(face->V[0]))
            cache->AddEntry(face->V[0]);
        if (!cache->InCache(face->V[1]))
            cache->AddEntry(face->V[1]);
        if (!cache->InCache(face->V[2]))
            cache->AddEntry(face->V[2]);
    }
}

// ============================================================================
// Static helper: NvStripifier::UpdateCacheFace (0x2AB4)
// ============================================================================
void NvStripifier::UpdateCacheFace(VertexCache *cache, NvFaceInfo *face) {
    if (!cache->InCache(face->V[0]))
        cache->AddEntry(face->V[0]);
    if (!cache->InCache(face->V[1]))
        cache->AddEntry(face->V[1]);
    if (!cache->InCache(face->V[2]))
        cache->AddEntry(face->V[2]);
}

// ============================================================================
// Static helper: NvStripifier::CalcNumHitsStrip (0x2B64)
// ============================================================================
float NvStripifier::CalcNumHitsStrip(VertexCache *cache, NvStripInfo *strip) {
    int hits = 0;
    int faceCount = 0;
    int count = strip->Faces.Size();
    for (int i = 0; i < count; i++) {
        NvFaceInfo *face = strip->Faces[i];
        if (cache->InCache(face->V[0])) hits++;
        if (cache->InCache(face->V[1])) hits++;
        if (cache->InCache(face->V[2])) hits++;
        faceCount++;
    }
    return (float)hits / (float)faceCount;
}

// ============================================================================
// Static helper: NvStripifier::CalcNumHitsFace (0x2C34)
// ============================================================================
int NvStripifier::CalcNumHitsFace(VertexCache *cache, NvFaceInfo *face) {
    int hits = 0;
    if (cache->InCache(face->V[0])) hits++;
    if (cache->InCache(face->V[1])) hits++;
    if (cache->InCache(face->V[2])) hits++;
    return hits;
}

// ============================================================================
// Static helper: NvStripifier::AvgStripSize (0x2D44)
// ============================================================================
float NvStripifier::AvgStripSize(const XArray<NvStripInfo *> &strips) {
    int totalFaces = 0;
    int stripCount = strips.Size();
    if (stripCount <= 0)
        return 0.0f;
    for (int i = 0; i < stripCount; i++) {
        totalFaces += strips[i]->Faces.Size();
    }
    return (float)totalFaces / (float)stripCount;
}

// ============================================================================
// Static helper: NvStripifier::FindTraversal (0x1538)
// ============================================================================
bool NvStripifier::FindTraversal(
    XArray<NvFaceInfo *> &allFaces,
    XArray<NvEdgeInfo *> &edgeBuckets,
    NvStripInfo *strip,
    NvFaceInfo *&outFace,
    NvEdgeInfo *&outEdge,
    bool &outCW) {

    // Get the last face in the strip
    if (strip->Faces.Size() == 0)
        return false;

    NvFaceInfo *lastFace = strip->Faces[strip->Faces.Size() - 1];

    // Check each neighbor
    for (int i = 0; i < 3; i++) {
        NvFaceInfo *neighbor = lastFace->Neighbor[i];
        if (!neighbor)
            continue;

        // Skip if already marked
        if (neighbor->MarkA >= 0)
            continue;
        if (strip->ExperimentId >= 0 && neighbor->Experiment == strip->ExperimentId)
            continue;

        // Find the edge between lastFace and neighbor
        int v0, v1;
        if (i == 0) { v0 = lastFace->V[0]; v1 = lastFace->V[1]; }
        else if (i == 1) { v0 = lastFace->V[0]; v1 = lastFace->V[2]; }
        else { v0 = lastFace->V[1]; v1 = lastFace->V[2]; }

        NvEdgeInfo *edge = FindEdgeInfo(edgeBuckets, v0, v1);
        if (!edge)
            continue;

        outFace = neighbor;
        outEdge = edge;
        // Approximate the expected winding flip based on strip parity.
        outCW = NextIsCW(strip->Faces.Size() + 2);
        return true;
    }

    return false;
}

static bool FaceContainsVertex(const NvFaceInfo *face, int v) {
    return face && (face->V[0] == v || face->V[1] == v || face->V[2] == v);
}

static int FaceUniqueVertexNotInOther(const NvFaceInfo *a, const NvFaceInfo *b) {
    if (!a || !b)
        return -1;
    for (int i = 0; i < 3; i++) {
        const int v = a->V[i];
        if (!FaceContainsVertex(b, v))
            return v;
    }
    return -1;
}

static int FaceUniqueVertexNotEq(const NvFaceInfo *face, int a, int b) {
    if (!face)
        return -1;
    for (int i = 0; i < 3; i++) {
        const int v = face->V[i];
        if (v != a && v != b)
            return v;
    }
    return -1;
}

static int FaceSharedVertices(const NvFaceInfo *a, const NvFaceInfo *b, int &out0, int &out1) {
    out0 = -1;
    out1 = -1;
    if (!a || !b)
        return 0;

    int count = 0;
    for (int i = 0; i < 3; i++) {
        const int v = a->V[i];
        if (FaceContainsVertex(b, v)) {
            if (count == 0)
                out0 = v;
            else if (count == 1)
                out1 = v;
            count++;
        }
    }
    return count;
}

static bool BuildStripIndexSequenceFromFaces(const XArray<NvFaceInfo *> &faces, XArray<CKWORD> &seq) {
    seq.Resize(0);
    if (faces.Size() <= 0)
        return false;

    // Single triangle strip.
    if (faces.Size() == 1) {
        const NvFaceInfo *f0 = faces[0];
        if (!f0)
            return false;
        seq.PushBack((CKWORD)f0->V[0]);
        seq.PushBack((CKWORD)f0->V[1]);
        seq.PushBack((CKWORD)f0->V[2]);
        return true;
    }

    const NvFaceInfo *f0 = faces[0];
    const NvFaceInfo *f1 = faces[1];
    if (!f0 || !f1)
        return false;

    // Choose initial ordering so that the last two indices are the shared edge with face[1].
    const int u0 = FaceUniqueVertexNotInOther(f0, f1);
    if (u0 < 0) {
        // Fallback: still emit something usable.
        seq.PushBack((CKWORD)f0->V[0]);
        seq.PushBack((CKWORD)f0->V[1]);
        seq.PushBack((CKWORD)f0->V[2]);
    } else {
        int sh0 = -1;
        int sh1 = -1;
        int shCount = 0;
        for (int i = 0; i < 3; i++) {
            const int v = f0->V[i];
            if (v == u0)
                continue;
            if (FaceContainsVertex(f1, v)) {
                if (shCount == 0)
                    sh0 = v;
                else
                    sh1 = v;
                shCount++;
            }
        }

        if (shCount != 2) {
            seq.PushBack((CKWORD)f0->V[0]);
            seq.PushBack((CKWORD)f0->V[1]);
            seq.PushBack((CKWORD)f0->V[2]);
        } else {
            seq.PushBack((CKWORD)u0);
            seq.PushBack((CKWORD)sh0);
            seq.PushBack((CKWORD)sh1);
        }
    }

    for (int i = 1; i < faces.Size(); i++) {
        const NvFaceInfo *cur = faces[i];
        if (!cur)
            return false;

        int lastA = (int)seq[seq.Size() - 2];
        int lastB = (int)seq[seq.Size() - 1];

        // If the current face doesn't share the last edge, try to realign using the shared edge with the previous face.
        if (!FaceContainsVertex(cur, lastA) || !FaceContainsVertex(cur, lastB)) {
            const NvFaceInfo *prev = faces[i - 1];
            int s0 = -1;
            int s1 = -1;
            const int shared = FaceSharedVertices(prev, cur, s0, s1);
            if (shared != 2)
                return false;

            const CKWORD last = seq[seq.Size() - 1];
            seq.PushBack(last); // degenerate
            seq.PushBack((CKWORD)s0);
            seq.PushBack((CKWORD)s1);

            lastA = (int)seq[seq.Size() - 2];
            lastB = (int)seq[seq.Size() - 1];
            if (!FaceContainsVertex(cur, lastA) || !FaceContainsVertex(cur, lastB))
                return false;
        }

        const int nextV = FaceUniqueVertexNotEq(cur, lastA, lastB);
        if (nextV < 0)
            return false;
        seq.PushBack((CKWORD)nextV);
    }

    return true;
}

// ============================================================================
// NvStripInfo::Build (0x834) - Uses STATIC local variables!
// ============================================================================
void NvStripInfo::Build(XArray<NvEdgeInfo *> &edgeBuckets, XArray<NvFaceInfo *> &allFaces) {
    // CRITICAL: Binary uses static local variables
    static XArray<NvFaceInfo *> forwardFaces;
    static XArray<NvFaceInfo *> backwardFaces;
    static XArray<CKWORD> scratchIndices;
    static XArray<NvFaceInfo *> tempAllFaces;

    Faces.Resize(0);

    if (!StartFace || !StartEdge)
        return;

    // Initialize scratch arrays
    forwardFaces.Resize(0);
    backwardFaces.Resize(0);
    scratchIndices.Resize(0);

    // Determine starting vertices based on StartCW
    int v0, v1;
    if (StartCW) {
        v0 = StartEdge->Vertex0;
        v1 = StartEdge->Vertex1;
    } else {
        v0 = StartEdge->Vertex1;
        v1 = StartEdge->Vertex0;
    }

    // Find the third vertex
    int v2 = -1;
    if (StartFace->V[0] != v0 && StartFace->V[0] != v1)
        v2 = StartFace->V[0];
    else if (StartFace->V[1] != v0 && StartFace->V[1] != v1)
        v2 = StartFace->V[1];
    else
        v2 = StartFace->V[2];

    // Build forward direction
    scratchIndices.PushBack((CKWORD)v0);
    scratchIndices.PushBack((CKWORD)v1);
    scratchIndices.PushBack((CKWORD)v2);

    forwardFaces.PushBack(StartFace);
    MarkTriangle(StartFace);

    NvFaceInfo *curFace = StartFace;
    while (true) {
        if (scratchIndices.Size() < 3)
            break;

        int a = (int)scratchIndices[scratchIndices.Size() - 2];
        int b = (int)scratchIndices[scratchIndices.Size() - 1];

        // Find neighbor through edge (a, b)
        NvFaceInfo *next = NvStripifier::FindOtherFace(edgeBuckets, a, b, curFace);
        if (!next)
            break;

        // Check if already marked
        if (next->MarkA >= 0)
            break;
        if (ExperimentId >= 0 && next->Experiment == ExperimentId)
            break;

        // Find next vertex
        int nextV = -1;
        if (next->V[0] != a && next->V[0] != b)
            nextV = next->V[0];
        else if (next->V[1] != a && next->V[1] != b)
            nextV = next->V[1];
        else if (next->V[2] != a && next->V[2] != b)
            nextV = next->V[2];
        else
            break;

        scratchIndices.PushBack((CKWORD)nextV);
        forwardFaces.PushBack(next);
        MarkTriangle(next);
        curFace = next;
    }

    // Build backward direction (reverse winding from start)
    scratchIndices.Resize(0);
    int bv0, bv1;
    if (StartCW) {
        bv0 = StartEdge->Vertex1;
        bv1 = StartEdge->Vertex0;
    } else {
        bv0 = StartEdge->Vertex0;
        bv1 = StartEdge->Vertex1;
    }
    scratchIndices.PushBack((CKWORD)bv0);
    scratchIndices.PushBack((CKWORD)bv1);
    scratchIndices.PushBack((CKWORD)v2);

    curFace = StartFace;
    while (true) {
        if (scratchIndices.Size() < 3)
            break;

        int a = (int)scratchIndices[scratchIndices.Size() - 2];
        int b = (int)scratchIndices[scratchIndices.Size() - 1];

        NvFaceInfo *next = NvStripifier::FindOtherFace(edgeBuckets, a, b, curFace);
        if (!next)
            break;

        if (next->MarkA >= 0)
            break;
        if (ExperimentId >= 0 && next->Experiment == ExperimentId)
            break;

        int nextV = -1;
        if (next->V[0] != a && next->V[0] != b)
            nextV = next->V[0];
        else if (next->V[1] != a && next->V[1] != b)
            nextV = next->V[1];
        else if (next->V[2] != a && next->V[2] != b)
            nextV = next->V[2];
        else
            break;

        scratchIndices.PushBack((CKWORD)nextV);
        backwardFaces.PushBack(next);
        MarkTriangle(next);
        curFace = next;
    }

    // Combine forward and backward
    Combine(forwardFaces, backwardFaces);
}

// ============================================================================
// NvStripInfo::Combine (0x11F4)
// ============================================================================
void NvStripInfo::Combine(XArray<NvFaceInfo *> &forward, XArray<NvFaceInfo *> &backward) {
    Faces.Resize(0);

    // Add backward faces in reverse order
    for (int i = backward.Size() - 1; i >= 0; i--) {
        Faces.PushBack(backward[i]);
    }

    // Add forward faces
    for (int i = 0; i < forward.Size(); i++) {
        Faces.PushBack(forward[i]);
    }
}

// ============================================================================
// NvStripifier::FindGoodResetPoint
// ============================================================================
NvFaceInfo *NvStripifier::FindGoodResetPoint(XArray<NvFaceInfo *> &allFaces, XArray<NvEdgeInfo *> &edgeBuckets) {
    int faceCount = allFaces.Size();
    if (faceCount == 0)
        return nullptr;

    // Find first unmarked face
    for (int i = 0; i < faceCount; i++) {
        NvFaceInfo *face = allFaces[i];
        if (face->MarkA < 0)
            return face;
    }
    return nullptr;
}

// ============================================================================
// NvStripifier::CommitStrips (0x13E8)
// ============================================================================
void NvStripifier::CommitStrips(XArray<NvStripInfo *> &outStrips, XArray<NvStripInfo *> *candidateSet) {
    int count = candidateSet->Size();
    for (int i = 0; i < count; i++) {
        NvStripInfo *strip = (*candidateSet)[i];
        strip->ExperimentId = -1;

        // Mark all faces as committed
        for (int j = 0; j < strip->Faces.Size(); j++) {
            strip->MarkTriangle(strip->Faces[j]);
        }

        outStrips.PushBack(strip);
    }
}

// ============================================================================
// NvStripifier::RemoveSmallStrips (0x1648)
// ============================================================================
void NvStripifier::RemoveSmallStrips(
    XArray<NvStripInfo *> &allStrips,
    XArray<NvFaceInfo *> &allFaces,
    XArray<NvStripInfo *> &outStrips,
    XArray<NvEdgeInfo *> &edgeBuckets) {

    // Move valid strips to output, collect orphan faces
    XArray<NvFaceInfo *> orphanFaces;

    for (int i = 0; i < allStrips.Size(); i++) {
        NvStripInfo *strip = allStrips[i];
        if (strip->Faces.Size() < 1) {
            // Collect orphan faces
            for (int j = 0; j < strip->Faces.Size(); j++) {
                orphanFaces.PushBack(strip->Faces[j]);
            }
            delete strip;
        } else {
            outStrips.PushBack(strip);
        }
    }
}

// ============================================================================
// NvStripifier::BuildStripifyInfo (0x174)
// ============================================================================
void NvStripifier::BuildStripifyInfo(
    const XArray<CKWORD> &indices,
    CKWORD vertexCount,
    XArray<NvFaceInfo *> &outFaces,
    XArray<NvEdgeInfo *> &outEdgeBuckets) {

    int triCount = indices.Size() / 3;

    // Create face array
    outFaces.Resize(0);
    outFaces.Reserve(triCount);
    for (int i = 0; i < triCount; i++) {
        NvFaceInfo *face = new NvFaceInfo();
        face->V[0] = (int)indices[i * 3 + 0];
        face->V[1] = (int)indices[i * 3 + 1];
        face->V[2] = (int)indices[i * 3 + 2];
        outFaces.PushBack(face);
    }

    // Create edge buckets
    outEdgeBuckets.Resize(0);
    outEdgeBuckets.Resize(vertexCount + 1);
    for (int i = 0; i < outEdgeBuckets.Size(); i++)
        outEdgeBuckets[i] = nullptr;

    // Use MeshAdjacency to compute neighbors and edges
    MeshAdjacency adj;
    adj.Init((CKWORD *)indices.Begin(), triCount);
    if (!adj.Compute(true, true))
        return;

    // Set neighbor links
    const XArray<MeshAdjacency::Face> &adjFaces = adj.GetFaces();
    for (int i = 0; i < triCount; i++) {
        const MeshAdjacency::Face &af = adjFaces[i];
        NvFaceInfo *face = outFaces[i];
        for (int e = 0; e < 3; e++) {
            CKDWORD link = af.faces[e];
            if (IS_BOUNDARY(link)) {
                face->Neighbor[e] = nullptr;
            } else {
                int nface = (int)MAKE_ADJ_TRI(link);
                if (nface >= 0 && nface < triCount)
                    face->Neighbor[e] = outFaces[nface];
                else
                    face->Neighbor[e] = nullptr;
            }
        }
    }

    // Create edge info and buckets
    const XArray<MeshAdjacency::Edge> &adjEdges = adj.GetEdges();
    for (int i = 0; i < adjEdges.Size(); i++) {
        const MeshAdjacency::Edge &e = adjEdges[i];
        NvEdgeInfo *edge = new NvEdgeInfo();
        edge->Vertex0 = (int)e.vertices[0];
        edge->Vertex1 = (int)e.vertices[1];

        CKDWORD f0 = e.faces[0];
        CKDWORD f1 = e.faces[1];
        edge->Face0 = (f0 != 0xFFFFFFFFu && (int)f0 < triCount) ? outFaces[(int)f0] : nullptr;
        edge->Face1 = (f1 != 0xFFFFFFFFu && (int)f1 < triCount) ? outFaces[(int)f1] : nullptr;

        // Insert into buckets
        if (edge->Vertex0 >= 0 && edge->Vertex0 < outEdgeBuckets.Size()) {
            edge->NextV0 = outEdgeBuckets[edge->Vertex0];
            outEdgeBuckets[edge->Vertex0] = edge;
        }
        if (edge->Vertex1 >= 0 && edge->Vertex1 < outEdgeBuckets.Size()) {
            edge->NextV1 = outEdgeBuckets[edge->Vertex1];
            outEdgeBuckets[edge->Vertex1] = edge;
        }
    }
}

// ============================================================================
// NvStripifier::FindAllStrips (0x2D94)
// ============================================================================
bool NvStripifier::FindAllStrips(
    XArray<NvStripInfo *> &outAllStrips,
    XArray<NvFaceInfo *> &allFaces,
    XArray<NvEdgeInfo *> &edgeBuckets,
    int numSamples,
    int minStripLength) {

    // For each sample, find candidate strips
    int stripId = 0;
    int experimentId = 0;

    XArray<NvFaceInfo *> usedResetPoints;

    for (int sample = 0; sample < numSamples; sample++) {
        // Find a good reset point
        NvFaceInfo *resetPoint = FindGoodResetPoint(allFaces, edgeBuckets);
        if (!resetPoint)
            return true;  // All faces used

        // Check if already used
        bool alreadyUsed = false;
        for (int i = 0; i < usedResetPoints.Size(); i++) {
            if (usedResetPoints[i] == resetPoint) {
                alreadyUsed = true;
                break;
            }
        }
        if (alreadyUsed)
            continue;
        usedResetPoints.PushBack(resetPoint);

        // Generate 6 candidate strips (3 edges x 2 directions)
        XArray<NvStripInfo *> candidates;

        int edges[3][2] = {
            {resetPoint->V[0], resetPoint->V[1]},
            {resetPoint->V[1], resetPoint->V[2]},
            {resetPoint->V[2], resetPoint->V[0]}
        };

        for (int e = 0; e < 3; e++) {
            NvEdgeInfo *edge = FindEdgeInfo(edgeBuckets, edges[e][0], edges[e][1]);
            if (!edge)
                continue;

            for (int cw = 0; cw < 2; cw++) {
                NvStripInfo *strip = new NvStripInfo();
                strip->StartFace = resetPoint;
                strip->StartEdge = edge;
                strip->StartCW = (unsigned char)cw;
                strip->StripId = stripId++;
                strip->ExperimentId = experimentId++;

                strip->Build(edgeBuckets, allFaces);

                candidates.PushBack(strip);
            }
        }

        // Find the best candidate
        int bestIdx = -1;
        int bestSize = 0;
        for (int i = 0; i < candidates.Size(); i++) {
            int size = candidates[i]->Faces.Size();
            if (size > bestSize) {
                bestSize = size;
                bestIdx = i;
            }
        }

        // Commit the best, delete others
        if (bestIdx >= 0 && bestSize >= minStripLength) {
            NvStripInfo *best = candidates[bestIdx];
            best->ExperimentId = -1;  // Commit
            for (int j = 0; j < best->Faces.Size(); j++)
                best->MarkTriangle(best->Faces[j]);
            outAllStrips.PushBack(best);
        }

        for (int i = 0; i < candidates.Size(); i++) {
            if (i != bestIdx)
                delete candidates[i];
            else if (!(bestIdx >= 0 && bestSize >= minStripLength))
                delete candidates[i];
        }
    }

    return true;
}

// ============================================================================
// NvStripifier::SplitUpStripsAndOptimize (0x21AC)
// ============================================================================
void NvStripifier::SplitUpStripsAndOptimize(
    XArray<NvStripInfo *> &allStrips,
    bool joinStrips,
    XArray<NvEdgeInfo *> &edgeBuckets) {

    // The binary implementation does complex strip splitting and cache-aware reordering
    // For now, we just keep strips as-is since we already have good strips from FindAllStrips
}

// ============================================================================
// NvStripifier::CreateStrips (0x1A94)
// ============================================================================
void NvStripifier::CreateStrips(
    const XArray<NvStripInfo *> &strips,
    XArray<CKWORD> &outIndices,
    bool joinStrips,
    CKDWORD &outStripCount) {

    outIndices.Resize(0);
    outStripCount = 0;

    for (int s = 0; s < strips.Size(); s++) {
        NvStripInfo *strip = strips[s];
        if (!strip || strip->Faces.Size() == 0)
            continue;

        // Build index sequence for this strip from the ordered face list.
        XArray<CKWORD> seq;
        if (!BuildStripIndexSequenceFromFaces(strip->Faces, seq))
            continue;

        if (seq.Size() == 0)
            continue;

        if (joinStrips) {
            // Connect strips with degenerate triangles
            if (outIndices.Size() > 0) {
                CKWORD last = outIndices[outIndices.Size() - 1];
                // Add parity-fixing degenerates if needed
                if ((outIndices.Size() % 2) != 0)
                    outIndices.PushBack(last);
                outIndices.PushBack(last);
                outIndices.PushBack(seq[0]);
            }
            for (int i = 0; i < seq.Size(); i++)
                outIndices.PushBack(seq[i]);
        } else {
            // Separate strips with -1 (0xFFFF)
            if (outIndices.Size() > 0)
                outIndices.PushBack((CKWORD)0xFFFF);
            for (int i = 0; i < seq.Size(); i++)
                outIndices.PushBack(seq[i]);
            outStripCount++;
        }
    }

    if (joinStrips)
        outStripCount = (outIndices.Size() > 0) ? 1 : 0;
}

// ============================================================================
// NvStripifier::DestroyStrips - Cleanup method
// ============================================================================
void NvStripifier::DestroyStrips(XArray<NvStripInfo *> &strips) {
    for (int i = 0; i < strips.Size(); i++) {
        delete strips[i];
    }
    strips.Resize(0);
}

// ============================================================================
// NvStripifier constructor/destructor
// ============================================================================
NvStripifier::NvStripifier()
    : m_MinStripLength(0), m_CacheSize(0), m_Ratio(0.0f), m_FirstTime(1) {
}

NvStripifier::~NvStripifier() {
}

// ============================================================================
// NvStripifier::Stripify - Main API (0x1F24)
// ============================================================================
void NvStripifier::Stripify(
    const XArray<CKWORD> &inIndices,
    int minStripLength,
    int cacheSize,
    CKWORD vertexCount,
    XArray<NvStripInfo *> &outStrips) {

    outStrips.Resize(0);

    // Binary: internalMin = max(1, minStripLength - 6)
    int internalMin = minStripLength - 6;
    if (internalMin < 1)
        internalMin = 1;

    m_MinStripLength = internalMin;
    m_CacheSize = cacheSize;
    m_Ratio = 0.0f;
    m_FirstTime = 1;

    // Copy indices to scratch
    m_Scratch.Resize(0);
    m_Scratch.Reserve(inIndices.Size());
    for (int i = 0; i < inIndices.Size(); i++)
        m_Scratch.PushBack(inIndices[i]);

    // Build adjacency info
    XArray<NvFaceInfo *> faces;
    XArray<NvEdgeInfo *> edgeBuckets;
    BuildStripifyInfo(m_Scratch, vertexCount, faces, edgeBuckets);

    // Find all strips (10 samples as in binary)
    FindAllStrips(outStrips, faces, edgeBuckets, 10, m_MinStripLength);

    // SplitUpStripsAndOptimize is called but does minimal work for now
    SplitUpStripsAndOptimize(outStrips, false, edgeBuckets);

    // Add remaining unmarked faces as single-triangle strips
    for (int i = 0; i < faces.Size(); i++) {
        NvFaceInfo *face = faces[i];
        if (face->MarkA < 0) {
            NvStripInfo *strip = new NvStripInfo();
            strip->StartFace = face;
            strip->StartEdge = FindEdgeInfo(edgeBuckets, face->V[0], face->V[1]);
            strip->StartCW = 1;
            strip->ExperimentId = -1;
            strip->StripId = outStrips.Size();
            strip->Faces.PushBack(face);
            strip->MarkTriangle(face);
            outStrips.PushBack(strip);
        }
    }

    // Cleanup faces (edges are owned by faces via adjacency, let them leak for now)
    // In a full implementation, we'd track ownership properly
}

// ============================================================================
// NvStripifier::Stripify - Convenience one-shot API
// ============================================================================
void NvStripifier::Stripify(
    const XArray<CKWORD> &inIndices,
    int minStripLength,
    int cacheSize,
    CKWORD vertexCount,
    bool joinStrips,
    XArray<CKWORD> &outIndices,
    CKDWORD &outStripCount) {

    XArray<NvStripInfo *> strips;
    Stripify(inIndices, minStripLength, cacheSize, vertexCount, strips);
    CreateStrips(strips, outIndices, joinStrips, outStripCount);

    // Cleanup strips
    DestroyStrips(strips);
}
