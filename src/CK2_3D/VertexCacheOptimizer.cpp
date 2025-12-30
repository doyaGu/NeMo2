#include "VertexCacheOptimizer.h"

// ============================================================================
// VertexCacheOptimizer Implementation
// Binary analysis: Uses greedy algorithm to reorder triangles for vertex cache
// ============================================================================

VertexCacheOptimizer::VertexCacheOptimizer()
    : m_ProcessedCount(0), m_FaceCount(0), m_CacheSize(0) {}

VertexCacheOptimizer::~VertexCacheOptimizer() {}

void VertexCacheOptimizer::Initialize(int vertexCount, int faceCount, int cacheSize) {
    m_FaceCount = faceCount;
    m_CacheSize = cacheSize;
    m_ProcessedCount = 0;

    // IDA: bit arrays are sized for their domains
    m_CacheSet.CheckSize(vertexCount);
    m_CacheSet.Clear();
    m_ProcessedFaces.CheckSize(faceCount);
    m_ProcessedFaces.Clear();

    // Reserve cache entries
    m_CacheEntries.Resize(0);

    // Reserve output indices
    m_OutputIndices.Reserve(faceCount * 3);
    m_OutputIndices.Resize(0);

    // IDA: per-vertex face lists
    m_VertexFaceLists.Resize(vertexCount);

    // IDA: per-face map -> pointer to active CacheFaces
    m_VertexFaceMap.Reserve(faceCount);
    m_VertexFaceMap.Resize(faceCount);
    if (faceCount > 0)
        memset(m_VertexFaceMap.Begin(), 0, faceCount * sizeof(CacheFaces *));
}

void VertexCacheOptimizer::BuildVertexFaceLists(const XArray<CKWORD> &indices) {
    int vertexCount = m_VertexFaceLists.Size();
    const CKWORD *indexPtr = indices.Begin();

    // Resize per-vertex face lists
    m_VertexFaceLists.Resize(vertexCount);
    for (int i = 0; i < vertexCount; i++)
        m_VertexFaceLists[i].Resize(0);

    // Build face lists for each vertex
    for (int faceIdx = 0; faceIdx < m_FaceCount; faceIdx++) {
        CKWORD v0 = indexPtr[0];
        CKWORD v1 = indexPtr[1];
        CKWORD v2 = indexPtr[2];
        indexPtr += 3;

        if (v0 < (unsigned int) vertexCount)
            m_VertexFaceLists[v0].PushBack(faceIdx);
        if (v1 < (unsigned int) vertexCount)
            m_VertexFaceLists[v1].PushBack(faceIdx);
        if (v2 < (unsigned int) vertexCount)
            m_VertexFaceLists[v2].PushBack(faceIdx);
    }

    // Track max faces per vertex
    int maxFacesPerVertex = 0;
    for (int i = 0; i < vertexCount; i++) {
        int count = m_VertexFaceLists[i].Size();
        if (count > maxFacesPerVertex)
            maxFacesPerVertex = count;
    }

    // Reserve active faces array
    m_ActiveFaces.Reserve(m_CacheSize * maxFacesPerVertex);
}

void VertexCacheOptimizer::ProcessFaces(const XArray<CKWORD> &indices) {
    // Initialize first vertex in cache
    AddCacheEntry(0);

    while (m_ProcessedCount < m_FaceCount) {
        // Reserve active faces if needed
        int activeCapacity = (m_ActiveFaces.End() - m_ActiveFaces.Begin()) / sizeof(CacheFaces);
        if (activeCapacity < 0)
            m_ActiveFaces.Reserve(0);
        m_ActiveFaces.Resize(0);

        // Build list of active faces from vertices in cache
        int *cacheBegin = m_CacheEntries.Begin();
        int *cacheEnd = m_CacheEntries.End();
        int maxHitCount = 1;

        for (int *cacheIt = cacheBegin; cacheIt < cacheEnd; ++cacheIt) {
            int vertexIdx = *cacheIt;
            XArray<int> &faceList = m_VertexFaceLists[vertexIdx];
            int *faceBegin = faceList.Begin();
            int *faceEnd = faceList.End();

            int startActiveCount = m_ActiveFaces.Size();

            for (int *faceIt = faceBegin; faceIt < faceEnd; ++faceIt) {
                int faceIdx = *faceIt;
                if (!m_ProcessedFaces.IsSet(faceIdx)) {
                    CacheFaces *existing = m_VertexFaceMap[faceIdx];
                    if (existing) {
                        int hitCount = existing->HitCount + 1;
                        existing->HitCount = hitCount;
                        if (hitCount > maxHitCount)
                            maxHitCount = hitCount;
                        if (vertexIdx < existing->MinVertexIndex)
                            existing->MinVertexIndex = vertexIdx;
                    } else {
                        // Add new active face
                        m_ActiveFaces.Expand(1);
                        CacheFaces *newFace = m_ActiveFaces.End() - 1;
                        newFace->FaceIndex = faceIdx;
                        newFace->HitCount = 1;
                        newFace->MinVertexIndex = vertexIdx;
                        m_VertexFaceMap[faceIdx] = newFace;
                    }
                }
            }

            // Check if we found any new faces
            if (m_ActiveFaces.Size() == startActiveCount) {
                m_VertexFaceLists[vertexIdx].Resize(0);
            }
        }

        if (m_ActiveFaces.Size() > 0) {
            // Choose best face based on cache scoring
            if (maxHitCount == 3) {
                // All vertices in cache - just add all faces with 3 hits
                CacheFaces *activeBegin = m_ActiveFaces.Begin();
                CacheFaces *activeEnd = m_ActiveFaces.End();
                for (CacheFaces *it = activeBegin; it < activeEnd; ++it) {
                    if (it->HitCount == 3) {
                        TouchFaceUncached(it->FaceIndex, indices);
                    }
                }
            } else {
                // Find face with lowest cost
                int bestFaceIdx = 0;
                int bestCost = 1000000;

                CacheFaces *activeBegin = m_ActiveFaces.Begin();
                CacheFaces *activeEnd = m_ActiveFaces.End();
                for (CacheFaces *it = activeBegin; it < activeEnd; ++it) {
                    int cost = ComputeFaceCost2(indices, it->FaceIndex);
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestFaceIdx = it->FaceIndex;
                    }
                }

                TouchFace(bestFaceIdx, indices);
            }

            // Clear vertex face map for processed faces
            CacheFaces *activeBegin = m_ActiveFaces.Begin();
            CacheFaces *activeEnd = m_ActiveFaces.End();
            for (CacheFaces *it = activeBegin; it < activeEnd; ++it) {
                m_VertexFaceMap[it->FaceIndex] = nullptr;
            }
        } else {
            // No active faces - find next unprocessed face
            int nextFace = m_ProcessedFaces.GetUnsetBitPosition(0);
            TouchFace(nextFace, indices);
        }
    }
}

// Binary: TouchFace (0x1053c)
void VertexCacheOptimizer::TouchFace(int faceIndex, const XArray<CKWORD> &indices) {
    // Mark face as processed
    m_ProcessedFaces.Set(faceIndex);
    ++m_ProcessedCount;

    // Get face vertices
    const CKWORD *facePtr = indices.Begin() + faceIndex * 3;

    // Process each vertex
    for (int i = 0; i < 3; i++) {
        CKWORD vertexIdx = facePtr[i];

        // Add to output
        m_OutputIndices.PushBack(vertexIdx);

        // Check if already in cache
        if (m_CacheSet.IsSet(vertexIdx))
            continue;

        // Evict oldest if cache is full
        if (m_CacheEntries.Size() >= m_CacheSize) {
            int evicted = *m_CacheEntries.Begin();
            m_CacheSet.Unset(evicted);
            // Remove first element
            if (m_CacheEntries.Begin() != m_CacheEntries.End()) {
                int count = m_CacheEntries.Size();
                if (count > 1)
                    memmove(m_CacheEntries.Begin(), m_CacheEntries.Begin() + 1, (count - 1) * sizeof(int));
                m_CacheEntries.Resize(count - 1);
            }
        }

        // Add to cache set
        m_CacheSet.Set(vertexIdx);

        // Add to cache entries
        m_CacheEntries.PushBack(vertexIdx);
    }
}

// Binary: TouchFaceUncached (0x1080c)
void VertexCacheOptimizer::TouchFaceUncached(int faceIndex, const XArray<CKWORD> &indices) {
    // Mark face as processed
    m_ProcessedFaces.Set(faceIndex);
    ++m_ProcessedCount;

    // Get face vertices and add to output
    const CKWORD *facePtr = indices.Begin() + faceIndex * 3;
    m_OutputIndices.PushBack(facePtr[0]);
    m_OutputIndices.PushBack(facePtr[1]);
    m_OutputIndices.PushBack(facePtr[2]);
}

// Binary: AddCacheEntry (0x109cc)
void VertexCacheOptimizer::AddCacheEntry(int vertexIndex) {
    // Check if already in cache
    if (m_CacheSet.IsSet(vertexIndex))
        return;

    // Evict oldest if cache is full
    if (m_CacheEntries.Size() >= m_CacheSize) {
        int evicted = *m_CacheEntries.Begin();
        m_CacheSet.Unset(evicted);
        // Remove first element
        if (m_CacheEntries.Begin() != m_CacheEntries.End()) {
            int count = m_CacheEntries.Size();
            if (count > 1)
                memmove(m_CacheEntries.Begin(), m_CacheEntries.Begin() + 1, (count - 1) * sizeof(int));
            m_CacheEntries.Resize(count - 1);
        }
    }

    // Add to cache set
    m_CacheSet.Set(vertexIndex);

    // Add to cache entries
    m_CacheEntries.PushBack(vertexIndex);
}

// Binary: ComputeFaceCost2 (0x10bfc)
int VertexCacheOptimizer::ComputeFaceCost2(const XArray<CKWORD> &indices, int faceIndex) {
    // IDA: missing = 3 - activeHitCount
    CacheFaces *active = (faceIndex >= 0 && faceIndex < m_VertexFaceMap.Size()) ? m_VertexFaceMap[faceIndex] : nullptr;
    int missing = active ? (3 - active->HitCount) : 3;
    if (missing <= 0)
        return 0;

    const int cacheEntriesSize = m_CacheEntries.Size();
    int evictCount = 0;
    if (missing + cacheEntriesSize > m_CacheSize)
        evictCount = missing + cacheEntriesSize - m_CacheSize;

    // Temporarily remove the oldest entries from the cache-set bitarray.
    for (int i = 0; i < evictCount && i < cacheEntriesSize; ++i) {
        m_CacheSet.Unset(m_CacheEntries[i]);
    }

    int penalty = 0;
    const CKWORD *facePtr = indices.Begin() + faceIndex * 3;

    for (int i = 0; i < 3; ++i) {
        const CKWORD vertexIdx = facePtr[i];

        // IDA: only consider vertices NOT in cache
        if (!m_CacheSet.IsSet(vertexIdx)) {
            if (vertexIdx < 0 || vertexIdx >= m_VertexFaceLists.Size())
                continue;

            XArray<int> &faceList = m_VertexFaceLists[vertexIdx];
            for (int *faceIt = faceList.Begin(); faceIt < faceList.End(); ++faceIt) {
                const int neighborFace = *faceIt;
                if (neighborFace == faceIndex)
                    continue;
                if (m_ProcessedFaces.IsSet(neighborFace))
                    continue;
                if (neighborFace < 0 || neighborFace >= m_VertexFaceMap.Size())
                    continue;
                if (!m_VertexFaceMap[neighborFace])
                    continue;

                // Count how many vertices of neighbor would be cached (given the temporary eviction).
                const CKWORD *neighborPtr = indices.Begin() + neighborFace * 3;
                int neighborHits = 1;
                for (int j = 0; j < 3; ++j) {
                    const CKWORD nv = neighborPtr[j];
                    if (nv != vertexIdx && m_CacheSet.IsSet(nv))
                        ++neighborHits;
                }
                if (neighborHits == 3)
                    --penalty;
            }
        }
    }

    // Restore temporarily removed cache entries.
    for (int i = 0; i < evictCount && i < cacheEntriesSize; ++i) {
        m_CacheSet.Set(m_CacheEntries[i]);
    }

    return missing + penalty;
}

float VertexCacheOptimizer::ComputeCacheMetric(const XArray<CKWORD> &indices, int cacheSize, int mode) {
    // Binary: sub_1002B150(this, &indices, cacheSize, mode)
    // - clears this->m_CacheEntries (XArray<int>)
    // - sets this->m_CacheSize = cacheSize
    // - clears a bitset (IDA uses the bit array at +0x20; in our layout that's m_ProcessedFaces)
    // - for each index: if already in cache => no miss; else evict oldest if full, insert, return miss.
    // - returns misses / (indexCount - 2) when mode != 0 else misses / (indexCount / 3)
    // The caller stores the float but does not use it.

    if (cacheSize <= 0)
        return 0.0f;

    const int indexCount = indices.Size();
    if (indexCount <= 0)
        return 0.0f;

    m_CacheEntries.Resize(0);
    m_CacheSize = cacheSize;
    m_ProcessedFaces.Clear();

    int misses = 0;
    for (const CKWORD *it = indices.Begin(); it < indices.End(); ++it) {
        const int v = (int) (*it);

        // In-cache check
        if (m_ProcessedFaces.IsSet(v))
            continue;

        // Miss: evict FIFO if full
        if (m_CacheEntries.Size() == m_CacheSize) {
            const int evicted = *m_CacheEntries.Begin();
            m_ProcessedFaces.Unset(evicted);
            if (m_CacheEntries.Size() > 1) {
                memmove(m_CacheEntries.Begin(), m_CacheEntries.Begin() + 1, (m_CacheEntries.Size() - 1) * sizeof(int));
            }
            m_CacheEntries.Resize(m_CacheEntries.Size() - 1);
        }

        m_CacheEntries.PushBack(v);
        m_ProcessedFaces.Set(v);
        ++misses;
    }

    const int denom = (mode != 0) ? (indexCount - 2) : (indexCount / 3);
    if (denom <= 0)
        return 0.0f;
    return (float) misses / (float) denom;
}
