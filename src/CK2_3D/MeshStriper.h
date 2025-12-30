#ifndef MESHSTRIPER_H
#define MESHSTRIPER_H

#include "MeshAdjacency.h"

// MeshStriper flags (inferred from decompilation)
typedef enum CKMESHSTRIPER_FLAGS {
    CKMESHSTRIPER_INDEX16 = 0x00000001UL,     // output uses 16-bit indices (CKWORD)
    CKMESHSTRIPER_PARITYFIX = 0x00000002UL,   // parity/orientation adjustments enabled
    CKMESHSTRIPER_SORTSEEDS = 0x00000004UL,   // seed ordering uses RadixSorter based on face degree
    CKMESHSTRIPER_CONNECTALL = 0x00000008UL,  // connect all strips into one (degenerate joins)
} CKMESHSTRIPER_FLAGS;

class MeshStriper {
public:
    struct Result {
        CKDWORD NbStrips;      // number of strips
        CKDWORD *StripLengths; // per-strip index counts (NbStrips entries). When connected, points to a single CKDWORD.
        void *StripIndices;    // points to CKWORD or CKDWORD index stream depending on flags
    };

    MeshStriper();
    ~MeshStriper();

    MeshStriper(const MeshStriper &) = delete;
    MeshStriper &operator=(const MeshStriper &) = delete;

    CKBOOL Init(CKWORD *triList, int triCount, CKDWORD flags);
    CKBOOL Compute(Result *out);

private:
    int TrackStrip(CKDWORD faceIndex,
                   CKDWORD startV0,
                   CKDWORD startV1,
                   CKDWORD *outVertices,
                   CKDWORD *outFaces,
                   CKBYTE *used) const;
    CKDWORD ComputeBestStrip(CKDWORD seedFace, CKBYTE *usedGlobal, int faceCount);
    void ConnectAllStrips(Result *io);

    MeshAdjacency m_Adj;
    CKBOOL m_AdjValid;
    CKDWORD m_Flags;
    CKDWORD m_NbStrips;

    XArray<CKDWORD> m_StripLengths;
    XArray<CKWORD> m_Indices16;
    XArray<CKDWORD> m_Indices32;

    CKDWORD m_ConnectedStripLength;
    XArray<CKWORD> m_Connected16;
    XArray<CKDWORD> m_Connected32;
};

#endif // MESHSTRIPER_H
