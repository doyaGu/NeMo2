#ifndef VXSPHERE_H
#define VXSPHERE_H

#include "VxVector.h"

/**
 * @brief Represents a 3D sphere, defined by a center and a radius.
 */
class VxSphere {
public:
    /// @brief Default constructor.
    VxSphere() {}

    /**
     * @brief Constructs a sphere with a given center and radius.
     * @param iCenter The center point of the sphere.
     * @param iRadius The radius of the sphere.
     */
    VxSphere(const VxVector &iCenter, const float iRadius) : m_Center(iCenter), m_Radius(iRadius) {}

    // Accessors
    /// @brief Gets a mutable reference to the sphere's center.
    VxVector &Center() { return m_Center; }
    /// @brief Gets a const reference to the sphere's center.
    const VxVector &Center() const { return m_Center; }
    /// @brief Gets a mutable reference to the sphere's radius.
    float &Radius() { return m_Radius; }
    /// @brief Gets a const reference to the sphere's radius.
    const float &Radius() const { return m_Radius; }

    /**
     * @brief Checks if a point is inside or on the surface of the sphere.
     * @param iPoint The point to check.
     * @return True if the point is inside or on the sphere, false otherwise.
     */
    bool IsPointInside(const VxVector &iPoint) {
        return (SquareMagnitude(iPoint - m_Center) <= (m_Radius * m_Radius));
    }

    /**
     * @brief Checks if an axis-aligned bounding box is completely contained within the sphere.
     * @param iBox The bounding box to check.
     * @return True if the box is totally inside the sphere, false otherwise.
     */
    bool IsBoxTotallyInside(const VxBbox &iBox) {
        VxVector minD = m_Center - iBox.Min;
        VxVector maxD = m_Center - iBox.Max;
        minD.Absolute();
        maxD.Absolute();
        maxD = Maximize(minD, maxD);
        return (maxD.SquareMagnitude() < (m_Radius * m_Radius));
    }

    /**
     * @brief Checks if a point lies exactly on the surface of the sphere.
     * @param iPoint The point to check.
     * @return True if the point is on the surface, false otherwise.
     */
    bool IsPointOnSurface(const VxVector &iPoint) {
        return (SquareMagnitude(iPoint - m_Center) == (m_Radius * m_Radius));
    }

    /// @brief Equality operator.
    bool operator==(const VxSphere &iSphere) const {
        return (m_Radius == iSphere.m_Radius) && (m_Center == iSphere.m_Center);
    }

protected:
    VxVector m_Center; ///< The center point of the sphere.
    float m_Radius;    ///< The radius of the sphere.
};

#endif // VXSPHERE_H
