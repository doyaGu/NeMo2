#ifndef MESHADJACENCY_H
#define MESHADJACENCY_H

#include "CKAll.h"

#define MAKE_ADJ_TRI(x) (x & 0x3fffffff)
#define GET_EDGE_NB(x) (x >> 30)
#define IS_BOUNDARY(x) (x == 0xffffffff)

// Mesh Adjacency Computation class usefully to create edges information of faces connectivity
class MeshAdjacency {
public:
    class Edge {
    public:
        CKDWORD vertices[2];
        CKDWORD faces[2];
    };

    class Face {
    public:
        CKBYTE FindEdge(CKDWORD iV0, CKDWORD iV1) const;
        CKDWORD OppositeVertex(CKDWORD iV0, CKDWORD iV1) const;

        CKDWORD vertices[3];
        CKDWORD faces[3];
    };

    MeshAdjacency();
    ~MeshAdjacency();

    void Init(CKWORD *iIndices, int iCount);
    void Init(CKDWORD *iIndices, int iCount);

    bool Compute(bool iEdges, bool iFaces);

    const XArray<Edge> &GetEdges() {
        return m_Edges;
    }
    const XArray<Edge> &GetEdges() const {
        return m_Edges;
    }

    const XArray<Face> &GetFaces() {
        return m_Faces;
    }
    const XArray<Face> &GetFaces() const {
        return m_Faces;
    }

private:
    void AddTriangle(int iIndex, CKDWORD iV0, CKDWORD iV1, CKDWORD iV2);
    void AddEdge(CKDWORD iV0, CKDWORD iV1, CKDWORD iFace, CKDWORD iEdgeIndex);
    bool UpdateLink(CKDWORD iFace1, CKDWORD iFace2, CKDWORD iV0, CKDWORD iV1);

    XArray<Edge> m_Edges;
    XArray<Face> m_Faces;
};

#endif // MESHADJACENCY_H
