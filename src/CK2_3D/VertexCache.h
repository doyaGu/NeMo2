#ifndef VERTEXCACHE_H
#define VERTEXCACHE_H

// ============================================================================
// VertexCache - LRU-style vertex cache simulation
// ============================================================================
class VertexCache {
public:
    explicit VertexCache(int size) : m_Entries(nullptr), m_Size(size) {
        m_Entries = new int[size];
        for (int i = 0; i < size; i++)
            m_Entries[i] = -1;
    }

    ~VertexCache() {
        delete[] m_Entries;
    }

    // Binary: VertexCache::InCache (0x19C4)
    bool InCache(int v) const {
        for (int i = 0; i < m_Size; i++) {
            if (m_Entries[i] == v)
                return true;
        }
        return false;
    }

    // Binary: VertexCache::AddEntry (0x19F4)
    int AddEntry(int v) {
        int evicted = m_Entries[m_Size - 1];
        for (int i = m_Size - 2; i >= 0; i--)
            m_Entries[i + 1] = m_Entries[i];
        m_Entries[0] = v;
        return evicted;
    }

    void Clear() {
        for (int i = 0; i < m_Size; i++)
            m_Entries[i] = -1;
    }

    int Size() const { return m_Size; }

private:
    int *m_Entries;  // +0x00
    int m_Size;      // +0x04

    // non-copyable
    VertexCache(const VertexCache &);
    VertexCache &operator=(const VertexCache &);
};

#endif // VERTEXCACHE_H