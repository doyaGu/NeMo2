#include "MeshAdjacency.h"

#include "RadixSort.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

MeshAdjacency::MeshAdjacency() {}

MeshAdjacency::~MeshAdjacency() {}

/////////////////////////////////////////////////////:
// Init the class with extern data
// the data are then copied inside and no more needed
void MeshAdjacency::Init(CKWORD *iIndices, int iCount) {
    XASSERT(iCount);

    // allocating the arrays
    m_Faces.Resize(iCount);
    m_Edges.Resize(iCount * 3);

    // adding the triangles
    for (int i = 0; i < iCount; ++i) {
        AddTriangle(i,
                    iIndices[i * 3 + 0],
                    iIndices[i * 3 + 1],
                    iIndices[i * 3 + 2]);
    }
}

// {secret}
void MeshAdjacency::Init(CKDWORD *iIndices, int iCount) {
    XASSERT(iCount);

    // allocating the arrays
    m_Faces.Resize(iCount);
    m_Edges.Resize(iCount * 3);

    // adding the triangles
    for (int i = 0; i < iCount; ++i) {
        AddTriangle(i,
                    iIndices[i * 3 + 0],
                    iIndices[i * 3 + 1],
                    iIndices[i * 3 + 2]);
    }
}

bool MeshAdjacency::Compute(bool iEdges, bool iFaces) {
    RadixSorter Core;

    XArray<CKDWORD> tempFaces;
    tempFaces.Resize(m_Edges.Size());
    XArray<CKDWORD> tempV0;
    tempV0.Resize(m_Edges.Size());
    XArray<CKDWORD> tempV1;
    tempV1.Resize(m_Edges.Size());

    int i;
    for (i = 0; i < m_Edges.Size(); i++) {
        tempFaces[i] = m_Edges[i].faces[0];
        tempV0[i] = m_Edges[i].vertices[0];
        tempV1[i] = m_Edges[i].vertices[1];
    }

    // Multiple sort
    CKDWORD *Sorted = Core.Sort(tempFaces.Begin(), m_Edges.Size()).Sort(tempV0.Begin(), m_Edges.Size()).Sort(
        tempV1.Begin(), m_Edges.Size()).GetIndices();

    // Read the list in sorted order, look for similar edges
    CKDWORD LastRef0 = tempV0[Sorted[0]];
    CKDWORD LastRef1 = tempV1[Sorted[0]];
    CKDWORD Count = 0;
    CKDWORD TmpBuffer[3];

    int edgesCount = m_Edges.Size();
    for (i = 0; i < edgesCount; i++) {
        CKDWORD Face = tempFaces[Sorted[i]]; // Owner face
        CKDWORD Ref0 = tempV0[Sorted[i]];    // Vertex ref #1
        CKDWORD Ref1 = tempV1[Sorted[i]];    // Vertex ref #2
        if (Ref0 == LastRef0 && Ref1 == LastRef1) {
            // Current edge is the same as last one
            TmpBuffer[Count++] = Face; // Store face number
            if (Count == 3) {
                return false; // Only works with manifold meshes (i.e. an edge is not shared by more than 2 triangles)
            }
        } else {
            // Here we have a new edge (LastRef0, LastRef1) shared by Count triangles stored in TmpBuffer
            if (Count == 2) {
                // if Count==1 => edge is a boundary edge: it belongs to a single triangle.
                // Hence there's no need to update a link to an adjacent triangle.
                bool Status = UpdateLink(TmpBuffer[0], TmpBuffer[1], LastRef0, LastRef1);
                if (!Status) {
                    return Status;
                }

                m_Edges[Sorted[i - 2]].faces[1] = tempFaces[Sorted[i - 1]];
                m_Edges[Sorted[i - 1]].faces[0] = 0xffffffff; // mark as invalid
            }
            // Reset for next edge
            Count = 0;
            TmpBuffer[Count++] = Face;
            LastRef0 = Ref0;
            LastRef1 = Ref1;
        }
    }

    bool Status = true;
    if (Count == 2)
        Status = UpdateLink(TmpBuffer[0], TmpBuffer[1], LastRef0, LastRef1);

    // We don't need the edges anymore
    if (!iEdges) {
        m_Edges.Resize(0);
    } else {
        // we have to "compress"
        i = 0;
        int lastEdge = edgesCount - 1;
        while (i != lastEdge) {
            if (m_Edges[i++].faces[0] == 0xffffffff) {
                // invalid edge
                bool found = false;
                while (i != lastEdge) {
                    if (m_Edges[lastEdge].faces[0] != 0xffffffff) {
                        found = true;
                        break;
                    }
                    lastEdge--;
                }
                if (found) {
                    // we found a replacement
                    m_Edges[i - 1] = m_Edges[lastEdge];
                    lastEdge--;
                }
            }
        }

        m_Edges.Resize(i + 1);
    }

    return Status;
}

/////////////////////////////////////////////////////:
// Private part

void MeshAdjacency::AddTriangle(int iIndex, CKDWORD iV0, CKDWORD iV1, CKDWORD iV2) {
    // Store vertex-references
    m_Faces[iIndex].vertices[0] = iV0;
    m_Faces[iIndex].vertices[1] = iV1;
    m_Faces[iIndex].vertices[2] = iV2;

    // Reset links
    m_Faces[iIndex].faces[0] = -1;
    m_Faces[iIndex].faces[1] = -1;
    m_Faces[iIndex].faces[2] = -1;

    // Add edge 01 to database
    if (iV0 < iV1)
        AddEdge(iV0, iV1, iIndex, 0);
    else
        AddEdge(iV1, iV0, iIndex, 0);
    // Add edge 02 to database
    if (iV0 < iV2)
        AddEdge(iV0, iV2, iIndex, 1);
    else
        AddEdge(iV2, iV0, iIndex, 1);
    // Add edge 12 to database
    if (iV1 < iV2)
        AddEdge(iV1, iV2, iIndex, 2);
    else
        AddEdge(iV2, iV1, iIndex, 2);
}

void MeshAdjacency::AddEdge(CKDWORD iV0, CKDWORD iV1, CKDWORD iFace, CKDWORD iEdgeIndex) {
    // Store edge data
    m_Edges[iFace * 3 + iEdgeIndex].vertices[0] = iV0;
    m_Edges[iFace * 3 + iEdgeIndex].vertices[1] = iV1;
    // only the face[0] is used for the first part of the algorithm
    m_Edges[iFace * 3 + iEdgeIndex].faces[0] = iFace;
    m_Edges[iFace * 3 + iEdgeIndex].faces[1] = 0xffffffff;
}

bool MeshAdjacency::UpdateLink(CKDWORD iFace1, CKDWORD iFace2, CKDWORD iV0, CKDWORD iV1) {
    Face &tri0 = m_Faces[iFace1]; // Catch the first triangle
    Face &tri1 = m_Faces[iFace2]; // Catch the second triangle

    // Get the edge IDs. 0xff means input references are wrong.
    CKBYTE EdgeNb0 = tri0.FindEdge(iV0, iV1);
    if (EdgeNb0 == 0xff)
        return false;

    CKBYTE EdgeNb1 = tri1.FindEdge(iV0, iV1);
    if (EdgeNb1 == 0xff)
        return false;

    // Update links. The two most significant bits contain the counterpart edge's ID.
    tri0.faces[EdgeNb0] = iFace2 | (CKDWORD(EdgeNb1) << 30);
    tri1.faces[EdgeNb1] = iFace1 | (CKDWORD(EdgeNb0) << 30);

    return true;
}

CKBYTE MeshAdjacency::Face::FindEdge(CKDWORD iV0, CKDWORD iV1) const {
    CKBYTE edgeNb = 0xff;

    if (vertices[0] == iV0 && vertices[1] == iV1)
        edgeNb = 0;
    else if (vertices[0] == iV1 && vertices[1] == iV0)
        edgeNb = 0;
    else if (vertices[0] == iV0 && vertices[2] == iV1)
        edgeNb = 1;
    else if (vertices[0] == iV1 && vertices[2] == iV0)
        edgeNb = 1;
    else if (vertices[1] == iV0 && vertices[2] == iV1)
        edgeNb = 2;
    else if (vertices[1] == iV1 && vertices[2] == iV0)
        edgeNb = 2;
    return edgeNb;
}

CKDWORD MeshAdjacency::Face::OppositeVertex(CKDWORD iV0, CKDWORD iV1) const {
    CKDWORD Ref = 0xffffffff;
    if (vertices[0] == iV0 && vertices[1] == iV1)
        Ref = vertices[2];
    else if (vertices[0] == iV1 && vertices[1] == iV0)
        Ref = vertices[2];
    else if (vertices[0] == iV0 && vertices[2] == iV1)
        Ref = vertices[1];
    else if (vertices[0] == iV1 && vertices[2] == iV0)
        Ref = vertices[1];
    else if (vertices[1] == iV0 && vertices[2] == iV1)
        Ref = vertices[0];
    else if (vertices[1] == iV1 && vertices[2] == iV0)
        Ref = vertices[0];
    return Ref;
}
