#include "PlaceFitter.h"

#include "XArray.h"
#include "VxMath.h"
#include "CK3dEntity.h"
#include "CKMesh.h"
#include "NearestPointGrid.h"

static void CollectHierarchyWorldVertices(CK3dEntity *root, XArray<VxVector> &outWorldPoints) {
    if (!root) return;

    CK3dEntity *current = nullptr;
    while ((current = root->HierarchyParser(current)) != nullptr) {
        CKMesh *mesh = current->GetCurrentMesh();
        if (!mesh) continue;

        // Prefer modifier vertices for CKPatchMesh compatibility.
        CKDWORD stride = 0;
        CKBYTE *vptr = mesh->GetModifierVertices(&stride);
        int vcount = mesh->GetModifierVertexCount();

        if (!vptr || stride == 0 || vcount <= 0) {
            // Fallback to raw positions.
            stride = 0;
            vptr = reinterpret_cast<CKBYTE *>(mesh->GetPositionsPtr(&stride));
            vcount = mesh->GetVertexCount();
        }

        if (!vptr || stride == 0 || vcount <= 0) continue;

        const VxMatrix &world = current->GetWorldMatrix();
        for (int i = 0; i < vcount; ++i, vptr += stride) {
            const VxVector &localPos = *reinterpret_cast<const VxVector *>(vptr);
            VxVector worldPos;
            Vx3DMultiplyMatrixVector(&worldPos, world, &localPos);
            outWorldPoints.PushBack(worldPos);
        }
    }
}

static VxBbox UnionBox(const VxBbox &a, const VxBbox &b) {
    VxBbox u;
    u.Min.x = (a.Min.x < b.Min.x) ? a.Min.x : b.Min.x;
    u.Min.y = (a.Min.y < b.Min.y) ? a.Min.y : b.Min.y;
    u.Min.z = (a.Min.z < b.Min.z) ? a.Min.z : b.Min.z;
    u.Max.x = (a.Max.x > b.Max.x) ? a.Max.x : b.Max.x;
    u.Max.y = (a.Max.y > b.Max.y) ? a.Max.y : b.Max.y;
    u.Max.z = (a.Max.z > b.Max.z) ? a.Max.z : b.Max.z;
    return u;
}

static float Max3(float a, float b, float c) {
    float m = (a > b) ? a : b;
    return (m > c) ? m : c;
}

PlaceFitter::PlaceFitter()
    : m_TargetCells(64),
      m_MaxCells(128),
      m_GridThreshold(0.5f),
      m_MinCommonPoints(3) {}

CKBOOL PlaceFitter::ComputeBestFitBBox(CK3dEntity *p1, CK3dEntity *p2, VxMatrix &bboxMatrix) {
    if (!p1 || !p2) return FALSE;

    XArray<VxVector> points1;
    XArray<VxVector> points2;
    CollectHierarchyWorldVertices(p1, points1);
    CollectHierarchyWorldVertices(p2, points2);

    if (points1.Size() <= 0 || points2.Size() <= 0) return FALSE;

    const VxBbox box1 = p1->GetHierarchicalBox(FALSE);
    const VxBbox box2 = p2->GetHierarchicalBox(FALSE);
    const VxBbox ubox = UnionBox(box1, box2);

    VxVector extent;
    extent.x = ubox.Max.x - ubox.Min.x;
    extent.y = ubox.Max.y - ubox.Min.y;
    extent.z = ubox.Max.z - ubox.Min.z;

    const float maxDim = Max3(extent.x, extent.y, extent.z);
    if (maxDim <= 0.0f) return FALSE;

    // Keep the same basic structure as the original: normalize points into a grid
    // and use NearestPointGrid for proximity matching.
    float cellSize = maxDim / float(m_TargetCells - 1);
    if (cellSize <= 0.0f) cellSize = 1.0f;

    int sizeX = (int) (extent.x / cellSize) + 2;
    int sizeY = (int) (extent.y / cellSize) + 2;
    int sizeZ = (int) (extent.z / cellSize) + 2;

    if (sizeX < 1) sizeX = 1;
    if (sizeY < 1) sizeY = 1;
    if (sizeZ < 1) sizeZ = 1;

    if (sizeX > m_MaxCells) sizeX = m_MaxCells;
    if (sizeY > m_MaxCells) sizeY = m_MaxCells;
    if (sizeZ > m_MaxCells) sizeZ = m_MaxCells;

    NearestPointGrid grid;
    grid.SetGridDimensions(sizeX, sizeY, sizeZ);
    grid.SetThreshold(m_GridThreshold);

    // Add p1 points in grid space.
    for (int i = 0; i < points1.Size(); ++i) {
        const VxVector &w = points1[i];
        VxVector gp;
        gp.x = (w.x - ubox.Min.x) / cellSize;
        gp.y = (w.y - ubox.Min.y) / cellSize;
        gp.z = (w.z - ubox.Min.z) / cellSize;

        if (gp.x >= 0.0f && gp.y >= 0.0f && gp.z >= 0.0f &&
            gp.x < (float) sizeX && gp.y < (float) sizeY && gp.z < (float) sizeZ) {
            grid.AddPoint(gp, i);
        }
    }

    // Match p2 points to p1 points.
    XArray<CKBOOL> used;
    used.Resize(points1.Size());
    used.Memset(FALSE);

    XArray<VxVector> commonWorld;
    for (int j = 0; j < points2.Size(); ++j) {
        const VxVector &w = points2[j];
        VxVector gp;
        gp.x = (w.x - ubox.Min.x) / cellSize;
        gp.y = (w.y - ubox.Min.y) / cellSize;
        gp.z = (w.z - ubox.Min.z) / cellSize;

        if (gp.x < 0.0f || gp.y < 0.0f || gp.z < 0.0f ||
            gp.x >= (float) sizeX || gp.y >= (float) sizeY || gp.z >= (float) sizeZ) {
            continue;
        }

        const int idx = grid.FindNearPoint(gp);
        if (idx >= 0 && idx < points1.Size() && !used[idx]) {
            used[idx] = TRUE;
            commonWorld.PushBack(points1[idx]);
        }
    }

    if (commonWorld.Size() < m_MinCommonPoints) return FALSE;

    return VxComputeBestFitBBox(reinterpret_cast<const XBYTE *>(commonWorld.Begin()),
                                sizeof(VxVector),
                                commonWorld.Size(),
                                bboxMatrix,
                                0.0f);
}
