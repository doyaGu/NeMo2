#ifndef VXPLANE_H
#define VXPLANE_H

#include "VxMatrix.h"
#include "VxVector.h"

/**
 * @brief Represents an infinite plane in 3D space.
 *
 * @remarks
 * A VxPlane is defined by a normal vector and a distance from the origin (D),
 * representing the plane equation Ax + By + Cz + D = 0. It is useful for
 * intersection tests and spatial classification. It can be constructed
 * from its equation components, from a normal and a point on the plane,
 * or from three co-planar points.
 *
 * A VxPlane is defined as:
 * @code
 * class VxPlane {
 * public:
 *     VxVector m_Normal; // The normal to the plane
 *     float    m_D;      // The D component of the Ax + By + Cz + D = 0 equation
 * };
 * @endcode
 *
 * @see VxVector, VxRay, VxIntersect
 */
class VxPlane {
public:
    /** @name Constructors
     * A VxPlane can be created from its equation, a normal and a point lying on the plane,
     * or three points belonging to the plane (which must not be collinear).
     */
    ///@{

    /// @brief Default constructor. Initializes to a zero plane (0,0,0,0).
    VxPlane() : m_Normal(0, 0, 0), m_D(0) {}

    /**
     * @brief Constructs a plane from a normal vector and a D value.
     * @param n The normal vector of the plane.
     * @param d The D component of the plane equation.
     */
    VxPlane(const VxVector &n, float d) : m_Normal(n), m_D(d) {}

    /**
     * @brief Constructs a plane from its equation components.
     * @param a The x-component of the normal.
     * @param b The y-component of the normal.
     * @param c The z-component of the normal.
     * @param d The D component of the plane equation.
     */
    VxPlane(float a, float b, float c, float d) : m_Normal(a, b, c), m_D(d) {}

    /**
     * @brief Constructs a plane from a normal vector and a point on the plane.
     * @param n The normal vector of the plane.
     * @param p A point lying on the plane.
     */
    VxPlane(const VxVector &n, const VxVector &p) { Create(n, p); }

    /**
     * @brief Constructs a plane from three points lying on the plane.
     * @param a The first point on the plane.
     * @param b The second point on the plane.
     * @param c The third point on the plane.
     */
    VxPlane(const VxVector &a, const VxVector &b, const VxVector &c) { Create(a, b, c); }

    ///@}

    /// @brief Returns a plane with its normal and D value negated.
    friend const VxPlane operator-(const VxPlane &p);

    /// @brief Gets the normal vector of the plane.
    const VxVector &GetNormal() const { return m_Normal; }

    /**
     * @brief Classifies a point with respect to the plane.
     * @param p A point in space.
     * @return The signed distance to the plane. A positive value means the point is in front of the plane (in the direction of the normal), a negative value means it is behind, and zero means it is on the plane.
     * @see Distance
     */
    float Classify(const VxVector &p) const { return DotProduct(m_Normal, p) + m_D; }

    /**
     * @brief Classifies an axis-aligned bounding box with respect to the plane.
     * @param box An AABB in the same coordinate system as the plane.
     * @return A positive value if the box is entirely in front of the plane, a negative value if it's entirely behind, and 0 if it intersects the plane.
     * @see Distance
     */
    float Classify(const VxBbox &box) const {
        VxVector vmin = box.Min;
        VxVector vmax = box.Max;
        for (int axis = 0; axis < 3; ++axis) {
            if (m_Normal[axis] < 0) {
                vmin[axis] = box.Max[axis];
                vmax[axis] = box.Min[axis];
            }
        }
        float f = Classify(vmin);
        if (f > 0.0f)
            return f;
        f = Classify(vmax);
        if (f >= 0.0f)
            return 0.0f;
        return f;
    }

    /**
     * @brief Classifies an oriented bounding box (OBB) with respect to the plane.
     * @param box The base axis-aligned bounding box.
     * @param mat The transformation matrix that orients the box.
     * @return A positive value if the box is entirely in front of the plane, a negative value if it's entirely behind, and 0 if it intersects the plane.
     * @see Distance
     */
    float Classify(const VxBbox &box, const VxMatrix &mat) const {
        VxVector hsize = box.GetHalfSize();
        float r = XAbs(DotProduct(m_Normal, mat[0] * hsize[0])) +
                  XAbs(DotProduct(m_Normal, mat[1] * hsize[1])) +
                  XAbs(DotProduct(m_Normal, mat[2] * hsize[2]));
        VxVector v = box.GetCenter();
        Vx3DMultiplyMatrixVector(&hsize, mat, &v);
        float d = DotProduct(hsize, m_Normal) + m_D;
        if (d > r)
            return (d - r);
        else if (-d > r)
            return -(-d - r);
        else
            return 0.0f;
    }

    /**
     * @brief Classifies a triangle face with respect to the plane.
     * @param pt0 The first vertex of the face.
     * @param pt1 The second vertex of the face.
     * @param pt2 The third vertex of the face.
     * @return A positive value if the face is entirely in front, a negative value if entirely behind, and 0 if it crosses the plane.
     * @see Distance
     */
    float ClassifyFace(const VxVector &pt0, const VxVector &pt1, const VxVector &pt2) const {
        float d = Classify(pt0);
        float min = d;

        d = Classify(pt1);
        if (min > 0.0f) {
            if (d < 0.0f) return 0.0f;
            if (d < min) min = d;
        } else {
            if (d > 0.0f) return 0.0f;
            if (d > min) min = d;
        }

        d = Classify(pt2);
        if (min > 0.0f) {
            if (d < 0.0f) return 0.0f;
            if (d < min) min = d;
        } else {
            if (d > 0.0f) return 0.0f;
            if (d > min) min = d;
        }
        return min;
    }

    /**
     * @brief Returns the absolute distance from a point to the plane.
     * @param p A point in space.
     * @return The perpendicular distance from the point to the plane.
     * @see Classify
     */
    float Distance(const VxVector &p) const { return XAbs(Classify(p)); }

    /**
     * @brief Returns the point on the plane that is closest to a given point.
     * @param p A point in space.
     * @return The projection of point p onto the plane.
     * @see Classify
     */
    const VxVector NearestPoint(const VxVector &p) const { return p + m_Normal * -Classify(p); }

    /// @brief Creates the plane from a normal and a point on the plane.
    VX_EXPORT void Create(const VxVector &n, const VxVector &p);

    /// @brief Creates the plane from three coplanar points.
    VX_EXPORT void Create(const VxVector &a, const VxVector &b, const VxVector &c);

    /// @brief Equality operator.
    bool operator==(const VxPlane &iPlane) const {
        return (m_Normal == iPlane.m_Normal) && (m_D == iPlane.m_D);
    }

    /**
     * @brief Internal classification method for an oriented box represented by its transformed axes and center.
     * @param boxaxis An array of 4 vectors: the 3 scaled and rotated axes and the transformed center point.
     * @return A negative value if intersecting, a positive value if completely outside.
     */
    float XClassify(const VxVector boxaxis[4]) const {
        float r = XAbs(DotProduct(m_Normal, boxaxis[0])) +
                  XAbs(DotProduct(m_Normal, boxaxis[1])) +
                  XAbs(DotProduct(m_Normal, boxaxis[2]));

        float d = DotProduct(m_Normal, boxaxis[3]) + m_D;
        if (d > r)
            return d;
        else if (d < -r)
            return d;
        else
            return 0.0f;
    }

    /// The normal to the plane.
    VxVector m_Normal;
    /// The D component of the Ax + By + Cz + D = 0 equation.
    float m_D;
};

/**
 * @brief Negates a plane.
 * @param p The plane to negate.
 * @return A new plane with its normal and D value inverted.
 */
inline const VxPlane operator-(const VxPlane &p) {
    return VxPlane(-p.m_Normal, -p.m_D);
}

#endif // VXPLANE_H
