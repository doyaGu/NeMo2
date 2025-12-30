#include "MeshStriper.h"
#include "RadixSort.h"

static CKDWORD MakeAdjTri(CKDWORD x) { return x & 0x3fffffff; }

static void ReverseRange(CKDWORD *data, int count) {
    if (!data || count <= 1)
        return;
    int i = 0;
    int j = count - 1;
    while (i < j) {
        const CKDWORD tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
        ++i;
        --j;
    }
}

MeshStriper::MeshStriper() : m_AdjValid(0), m_Flags(0), m_NbStrips(0), m_ConnectedStripLength(0) {}

MeshStriper::~MeshStriper() = default;

CKBOOL MeshStriper::Init(CKWORD *triList, int triCount, CKDWORD flags) {
    m_Flags = flags;
    m_AdjValid = 0;
    if (!triList || triCount <= 0)
        return 0;

    m_Adj.Init(triList, triCount);
    if (!m_Adj.Compute(false, true))
        return 0;

    m_AdjValid = 1;
    return 1;
}

CKBOOL MeshStriper::Compute(Result *out) {
    if (!out || !m_AdjValid)
        return 0;

    m_NbStrips = 0;
    m_StripLengths.Resize(0);
    m_Indices16.Resize(0);
    m_Indices32.Resize(0);
    m_Connected16.Resize(0);
    m_Connected32.Resize(0);
    m_ConnectedStripLength = 0;

    const XArray<MeshAdjacency::Face> &faces = m_Adj.GetFaces();
    const int faceCount = faces.Size();
    if (faceCount <= 0)
        return 0;

    XArray<CKBYTE> used;
    used.Resize(faceCount);
    for (int i = 0; i < faceCount; ++i)
        used[i] = 0;

    XArray<CKDWORD> order;
    order.Resize(faceCount);

    if ((m_Flags & CKMESHSTRIPER_SORTSEEDS) != 0) {
        XArray<CKDWORD> degrees;
        degrees.Resize(faceCount);
        for (int f = 0; f < faceCount; ++f) {
            const MeshAdjacency::Face &face = faces[f];
            CKDWORD deg = 0;
            if ((int) face.faces[0] != -1)
                ++deg;
            if ((int) face.faces[1] != -1)
                ++deg;
            if ((int) face.faces[2] != -1)
                ++deg;
            degrees[f] = deg;
        }

        RadixSorter sorter;
        sorter.Sort(degrees.Begin(), (CKDWORD) faceCount, true);
        CKDWORD *sorted = sorter.GetIndices();
        for (int i = 0; i < faceCount; ++i)
            order[i] = sorted[i];
    } else {
        for (int i = 0; i < faceCount; ++i)
            order[i] = (CKDWORD) i;
    }

    CKDWORD consumed = 0;
    for (int idx = 0; idx < faceCount; ++idx) {
        const CKDWORD seedFace = order[idx];
        if (seedFace >= (CKDWORD) faceCount)
            continue;
        if (used[(int) seedFace])
            continue;

        const int before = m_StripLengths.Size();
        const CKDWORD usedFaces = ComputeBestStrip(seedFace, used.Begin(), faceCount);
        if (m_StripLengths.Size() != before) {
            consumed += usedFaces;
            ++m_NbStrips;
        }
        if (consumed == (CKDWORD) faceCount)
            break;
    }

    out->NbStrips = m_NbStrips;
    out->StripLengths = m_StripLengths.Begin();
    out->StripIndices = (m_Flags & CKMESHSTRIPER_INDEX16) ? (void *) m_Indices16.Begin() : (void *) m_Indices32.Begin();

    if ((m_Flags & CKMESHSTRIPER_CONNECTALL) != 0) {
        ConnectAllStrips(out);
    }

    return (out->NbStrips != 0 && out->StripIndices != nullptr) ? TRUE : FALSE;
}

int MeshStriper::TrackStrip(
    CKDWORD faceIndex,
    CKDWORD startV0,
    CKDWORD startV1,
    CKDWORD *outVertices,
    CKDWORD *outFaces,
    CKBYTE *used) const {
    if (!outVertices || !outFaces || !used)
        return 0;

    outVertices[0] = startV0;
    outVertices[1] = startV1;

    const XArray<MeshAdjacency::Face> &faces = m_Adj.GetFaces();
    const int faceCount = faces.Size();
    if (faceIndex >= (CKDWORD) faceCount)
        return 2;
    int outVertexCount = 2;

    CKDWORD v7 = startV0;
    CKDWORD vLast = startV1;
    bool cont = true;
    while (cont) {
        const MeshAdjacency::Face &face = faces[(int) faceIndex];
        const CKDWORD vOpp = face.OppositeVertex(v7, vLast);
        if ((int) vOpp == -1)
            break;
        outVertices[outVertexCount++] = vOpp;
        *outFaces++ = faceIndex;
        used[(int) faceIndex] = 1;

        const CKBYTE edge = face.FindEdge(vLast, vOpp);
        if (edge == 0xff)
            break;
        const CKDWORD link = face.faces[edge];
        if ((int) link == -1) {
            cont = false;
        } else {
            const CKDWORD nextFace = MakeAdjTri(link);
            if (nextFace >= (CKDWORD) faceCount || used[(int) nextFace]) {
                cont = false;
            } else {
                faceIndex = nextFace;
            }
        }

        v7 = vLast;
        vLast = vOpp;
    }

    return outVertexCount;
}

CKDWORD MeshStriper::ComputeBestStrip(CKDWORD seedFace, CKBYTE *usedGlobal, int faceCount) {
    const XArray<MeshAdjacency::Face> &faces = m_Adj.GetFaces();
    const MeshAdjacency::Face &seed = faces[(int) seedFace];

    CKDWORD v87[3] = {seed.vertices[1], seed.vertices[0], seed.vertices[2]};
    CKDWORD v88[3] = {seed.vertices[0], seed.vertices[2], seed.vertices[1]};

    struct Candidate {
        XArray<CKDWORD> verts;
        XArray<CKDWORD> faces;
        CKDWORD initialLen;
        CKDWORD totalLen;
    };

    Candidate cands[3];
    for (int i = 0; i < 3; ++i) {
        cands[i].initialLen = 0;
        cands[i].totalLen = 0;
        cands[i].verts.Resize(faceCount + 5);
        cands[i].faces.Resize(faceCount + 2);
        for (int k = 0; k < cands[i].verts.Size(); ++k)
            cands[i].verts[k] = 0xFFFFFFFFu;
        for (int k = 0; k < cands[i].faces.Size(); ++k)
            cands[i].faces[k] = 0xFFFFFFFFu;

        XArray<CKBYTE> usedWork;
        usedWork.Resize(faceCount);
        memcpy(usedWork.Begin(), usedGlobal, (size_t) faceCount * sizeof(CKBYTE));

        const int initial = TrackStrip(
            seedFace,
            v88[i],
            v87[i],
            cands[i].verts.Begin(),
            cands[i].faces.Begin(),
            usedWork.Begin());
        cands[i].initialLen = (CKDWORD) initial;

        if (initial < 3) {
            cands[i].totalLen = (CKDWORD) initial;
            continue;
        }

        ReverseRange(cands[i].verts.Begin(), initial);
        if (initial >= 2)
            ReverseRange(cands[i].faces.Begin(), initial - 2);

        const int growStart = initial - 3;
        const int extend = TrackStrip(
            seedFace,
            cands[i].verts[growStart],
            cands[i].verts[growStart + 1],
            &cands[i].verts[growStart],
            &cands[i].faces[growStart],
            usedWork.Begin());
        cands[i].totalLen = (CKDWORD) (extend + initial - 3);
    }

    int best = 0;
    if (cands[1].totalLen > cands[0].totalLen)
        best = 1;
    if (cands[2].totalLen > cands[best].totalLen)
        best = 2;

    CKDWORD bestLen = cands[best].totalLen;
    const CKDWORD initialLen = cands[best].initialLen;
    if (bestLen < 3)
        return 0;
    const CKDWORD triUsed = (bestLen >= 2) ? (bestLen - 2) : 0;

    for (CKDWORD t = 0; t < triUsed; ++t) {
        const CKDWORD f = cands[best].faces[(int) t];
        if (f < (CKDWORD) faceCount)
            usedGlobal[(int) f] = 1;
    }

    if ((m_Flags & CKMESHSTRIPER_PARITYFIX) != 0 && (initialLen & 1) != 0) {
        if (bestLen == 3 || bestLen == 4) {
            const CKDWORD tmp = cands[best].verts[1];
            cands[best].verts[1] = cands[best].verts[2];
            cands[best].verts[2] = tmp;
        } else {
            ReverseRange(cands[best].verts.Begin(), (int) bestLen);
            if (((bestLen - initialLen) & 1) != 0) {
                // Insert one extra copy of first vertex at index 1 (shift right).
                for (int i = (int) bestLen; i > 1; --i)
                    cands[best].verts[i] = cands[best].verts[i - 1];
                cands[best].verts[1] = cands[best].verts[0];
                ++bestLen;
            }
        }
    }

    if ((m_Flags & CKMESHSTRIPER_INDEX16) != 0) {
        for (CKDWORD i = 0; i < bestLen; ++i)
            m_Indices16.PushBack((CKWORD) cands[best].verts[(int) i]);
    } else {
        for (CKDWORD i = 0; i < bestLen; ++i)
            m_Indices32.PushBack((CKDWORD) cands[best].verts[(int) i]);
    }
    m_StripLengths.PushBack(bestLen);
    return triUsed;
}

void MeshStriper::ConnectAllStrips(Result *io) {
    if (!io || io->NbStrips == 0)
        return;

    m_ConnectedStripLength = 0;
    if ((m_Flags & CKMESHSTRIPER_INDEX16) != 0)
        m_Connected16.Resize(0);
    else
        m_Connected32.Resize(0);

    const CKDWORD stripCount = io->NbStrips;
    const CKDWORD *lens = io->StripLengths;
    if (!lens)
        return;

    const CKWORD *src16 = ((m_Flags & CKMESHSTRIPER_INDEX16) != 0) ? (const CKWORD *) io->StripIndices : nullptr;
    const CKDWORD *src32 = ((m_Flags & CKMESHSTRIPER_INDEX16) == 0) ? (const CKDWORD *) io->StripIndices : nullptr;
    if (!src16 && !src32)
        return;

    bool haveOutput = false;
    for (CKDWORD s = 0; s < stripCount; ++s) {
        CKDWORD len = lens[(int) s];
        if (!len)
            continue;

        if (haveOutput) {
            CKDWORD prevLast;
            CKDWORD firstCur;
            CKDWORD secondCur;
            if ((m_Flags & CKMESHSTRIPER_INDEX16) != 0) {
                prevLast = m_Connected16[m_Connected16.Size() - 1];
                firstCur = src16[0];
                secondCur = (len >= 2) ? src16[1] : src16[0];
                m_Connected16.PushBack((CKWORD) prevLast);
                m_Connected16.PushBack((CKWORD) firstCur);
            } else {
                prevLast = m_Connected32[m_Connected32.Size() - 1];
                firstCur = src32[0];
                secondCur = (len >= 2) ? src32[1] : src32[0];
                m_Connected32.PushBack(prevLast);
                m_Connected32.PushBack(firstCur);
            }
            m_ConnectedStripLength += 2;

            if ((m_Flags & CKMESHSTRIPER_PARITYFIX) != 0 && (m_ConnectedStripLength & 1) != 0) {
                if (firstCur == secondCur) {
                    if (len) {
                        --len;
                        if ((m_Flags & CKMESHSTRIPER_INDEX16) != 0)
                            ++src16;
                        else
                            ++src32;
                    }
                } else {
                    if ((m_Flags & CKMESHSTRIPER_INDEX16) != 0)
                        m_Connected16.PushBack((CKWORD) firstCur);
                    else
                        m_Connected32.PushBack(firstCur);
                    ++m_ConnectedStripLength;
                }
            }
        }

        if ((m_Flags & CKMESHSTRIPER_INDEX16) != 0) {
            for (CKDWORD i = 0; i < len; ++i)
                m_Connected16.PushBack(src16[i]);
            src16 += len;
        } else {
            for (CKDWORD i = 0; i < len; ++i)
                m_Connected32.PushBack(src32[i]);
            src32 += len;
        }
        m_ConnectedStripLength += len;
        haveOutput = true;
    }

    if (!haveOutput) {
        io->NbStrips = 0;
        io->StripLengths = nullptr;
        io->StripIndices = nullptr;
        m_ConnectedStripLength = 0;
        return;
    }

    io->NbStrips = 1;
    io->StripLengths = &m_ConnectedStripLength;
    io->StripIndices = (m_Flags & CKMESHSTRIPER_INDEX16) ? (void *) m_Connected16.Begin() : (void *) m_Connected32.Begin();
}
