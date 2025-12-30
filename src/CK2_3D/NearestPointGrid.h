/// @file NearestPointGrid.h
/// @brief Spatial grid for fast nearest point queries

#ifndef NEARESTPOINTGRID_H
#define NEARESTPOINTGRID_H

#include "VxVector.h"
#include "XArray.h"

/// Spatial grid for efficient 3D nearest point search.
/// Usage: SetGridDimensions() -> AddPoint() -> SetThreshold() -> FindNearPoint()
class NearestPointGrid {
protected:
    struct Point {
        VxVector pt;
        int index;
    };

    typedef XArray<Point> Cell;

    XArray<Cell *> m_Grid;  // 3D grid array
    int m_SizeX;             // Grid X dimension
    int m_SizeY;             // Grid Y dimension
    int m_SizeZ;             // Grid Z dimension
    int m_SizeXY;            // SizeX × SizeY
    int m_SizeXYZ;           // Total cell count
    float m_Threshold;       // Max distance for "near" points
    float m_Threshold2;      // Squared threshold

    Cell *&GetCell(const int x, const int y, const int z) {
        return m_Grid[x + m_SizeX * y + m_SizeXY * z];
    }

public:
    NearestPointGrid();
    ~NearestPointGrid();

    /// Sets grid dimensions. Clears all stored points.
    void SetGridDimensions(int sizeX, int sizeY, int sizeZ);

    /// Sets the distance threshold for queries.
    inline void SetThreshold(const float threshold) {
        m_Threshold = threshold;
        m_Threshold2 = threshold * threshold;
    }

    /// Adds a 3D point to the grid.
    void AddPoint(const VxVector &p, int index);

    /// Finds index of a point within threshold distance.
    /// Returns -1 if no point found.
    int FindNearPoint(const VxVector &p);
};

#endif // NEARESTPOINTGRID_H
