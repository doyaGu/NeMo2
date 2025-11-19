#include "VxPlane.h"

void VxPlane::Create(const VxVector &n, const VxVector &p) {
    // Create a plane from a normal vector and a point on the plane
    m_Normal = n;

    // Normalize the normal vector
    if (m_Normal.SquareMagnitude() > EPSILON) {
        m_Normal.Normalize();
    }

    // Calculate the D component of the plane equation Ax + By + Cz + D = 0
    // D = -(n · p)
    m_D = -DotProduct(m_Normal, p);
}

void VxPlane::Create(const VxVector &a, const VxVector &b, const VxVector &c) {
    // Create a plane from three points
    // The normal is the cross product of two edges
    VxVector edge1 = b - a;
    VxVector edge2 = c - a;

    m_Normal = CrossProduct(edge1, edge2);

    // Normalize the normal vector
    if (m_Normal.SquareMagnitude() > EPSILON) {
        m_Normal.Normalize();
    } else {
        // Degenerate case - default to Z-up normal
        m_Normal = VxVector(0.0f, 0.0f, 1.0f);
    }

    // Calculate the D component of the plane equation using any of the points
    m_D = -DotProduct(m_Normal, a);
}
