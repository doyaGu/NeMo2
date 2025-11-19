#ifndef VXOBB_H
#define VXOBB_H

#include "VxMatrix.h"
#include "VxVector.h"

/**
 * @brief Represents an Oriented Bounding Box (OBB).
 *
 * @remarks
 * An OBB is a rectangular block that is not necessarily aligned with the
 * coordinate axes. It is defined by a center point, three mutually
 * perpendicular axis vectors, and the half-length extents along those axes.
 *
 * A VxOBB is defined as:
 * @code
 * class VxOBB {
 * public:
 *     VxVector m_Center;
 *     VxVector m_Axis[3];
 *     VxVector m_Extents;
 * };
 * @endcode
 *
 * @see VxMatrix, VxVector, VxBbox
 */
class VxOBB {
public:
    /// @brief Default constructor.
    VxOBB() {}

    /**
     * @brief Constructs an OBB from an axis-aligned box and a transformation matrix.
     * @param box The original AABB.
     * @param mat The transformation matrix to apply to the AABB.
     */
    VxOBB(const VxBbox &box, const VxMatrix &mat) { Create(box, mat); }

    /// @brief Gets a mutable reference to the center of the OBB.
    VxVector &GetCenter() { return m_Center; };
    /// @brief Gets a const reference to the center of the OBB.
    const VxVector &GetCenter() const { return m_Center; }

    /// @brief Gets a mutable reference to a specific axis of the OBB.
    VxVector &GetAxis(int i) { return m_Axis[i]; }
    /// @brief Gets a const reference to a specific axis of the OBB.
    const VxVector &GetAxis(int i) const { return m_Axis[i]; }
    /// @brief Gets a mutable pointer to the axes array of the OBB.
    VxVector *GetAxes() { return m_Axis; }
    /// @brief Gets a const pointer to the axes array of the OBB.
    const VxVector *GetAxes() const { return m_Axis; };

    /// @brief Gets a mutable reference to a specific extent of the OBB.
    float &GetExtent(int i) { return m_Extents[i]; }
    /// @brief Gets a const reference to a specific extent of the OBB.
    const float &GetExtent(int i) const { return m_Extents[i]; }
    /// @brief Gets a mutable pointer to the extents vector of the OBB.
    float *GetExtents() { return &m_Extents.x; }
    /// @brief Gets a const pointer to the extents vector of the OBB.
    const float *GetExtents() const { return &m_Extents.x; }

    /**
     * @brief Creates the OBB from an axis-aligned box and a transformation matrix.
     * @param box The source AABB.
     * @param mat The transformation matrix.
     */
    void Create(const VxBbox &box, const VxMatrix &mat) {
        VxVector v = box.GetCenter();
        Vx3DMultiplyMatrixVector(&m_Center, mat, &v);
        m_Axis[0] = *(VxVector *) &mat[0][0];
        m_Axis[1] = *(VxVector *) &mat[1][0];
        m_Axis[2] = *(VxVector *) &mat[2][0];
        m_Extents[0] = Magnitude(m_Axis[0]);
        m_Extents[1] = Magnitude(m_Axis[1]);
        m_Extents[2] = Magnitude(m_Axis[2]);
        m_Axis[0] /= m_Extents[0];
        m_Axis[1] /= m_Extents[1];
        m_Axis[2] /= m_Extents[2];
        m_Extents[0] *= 0.5f * (box.Max[0] - box.Min[0]);
        m_Extents[1] *= 0.5f * (box.Max[1] - box.Min[1]);
        m_Extents[2] *= 0.5f * (box.Max[2] - box.Min[2]);
    }

    /**
     * @brief Checks if a point is inside this OBB.
     * @param iV The point to check.
     * @return TRUE if the point is inside the box, FALSE otherwise.
     */
    XBOOL VectorIn(const VxVector &iV) const {
        VxVector d = (iV - m_Center);

        float xRes = DotProduct(d, m_Axis[0]);
        if (XAbs(xRes) > m_Extents[0]) return FALSE;
        float yRes = DotProduct(d, m_Axis[1]);
        if (XAbs(yRes) > m_Extents[1]) return FALSE;
        float zRes = DotProduct(d, m_Axis[2]);
        if (XAbs(zRes) > m_Extents[2]) return FALSE;
        return TRUE;
    }

    /**
     * @brief Checks if an AABB is completely contained inside this OBB.
     * @param iB The AABB to check.
     * @return TRUE if the AABB is fully inside this OBB, FALSE otherwise.
     */
    XBOOL IsBoxInside(const VxBbox &iB) const {
        if (!VectorIn(iB.Max))
            return FALSE;
        if (!VectorIn(iB.Min))
            return FALSE;
        // Quick test on extrema passed
        // need to check all other 6 points
        VxVector tmp(iB.Min.x, iB.Min.y, iB.Max.z);
        if (!VectorIn(tmp)) return FALSE;
        tmp.y = iB.Max.y;
        if (!VectorIn(tmp)) return FALSE;
        tmp.z = iB.Min.z;
        if (!VectorIn(tmp)) return FALSE;
        tmp.x = iB.Max.x;
        if (!VectorIn(tmp)) return FALSE;
        tmp.y = iB.Min.y;
        if (!VectorIn(tmp)) return FALSE;
        tmp.z = iB.Min.z;
        if (!VectorIn(tmp)) return FALSE;
        return TRUE;
    }

    /// @brief Equality operator.
    bool operator==(const VxOBB &iBox) const {
        return (m_Extents == iBox.m_Extents) &&
            (m_Center == iBox.m_Center) &&
            (m_Axis[0] == iBox.m_Axis[0]) &&
            (m_Axis[1] == iBox.m_Axis[1]) &&
            (m_Axis[2] == iBox.m_Axis[2]);
    }

    /// The center of the box.
    VxVector m_Center;
    /// The three normalized axes defining the orientation of the box.
    VxVector m_Axis[3];
    /// The half-lengths of the box along each of its axes.
    VxVector m_Extents;
};

#endif // VxOBB_H