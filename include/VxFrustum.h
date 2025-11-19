#ifndef VXFRUSTUM_H
#define VXFRUSTUM_H

#include "VxMatrix.h"
#include "VxPlane.h"
#include "VxVector.h"

/**
 * @brief Represents a viewing frustum, typically used for visibility culling.
 *
 * @remarks
 * This class defines a 3D frustum volume, which is essential for determining
 * which objects are within a camera's field of view. It is defined by an origin,
 * orientation vectors, and near/far plane distances, along with field of view and aspect ratio.
 * It also pre-computes the six bounding planes of the frustum for efficient intersection tests.
 *
 * @see VxVector, VxRay, VxIntersect
 */
class VxFrustum {
public:
    /**
     * @brief Default constructor.
     */
    VX_EXPORT VxFrustum();

    /**
     * @brief Constructs a frustum from a set of parameters.
     * @param origin The origin point of the frustum (camera position).
     * @param right The right direction vector.
     * @param up The up direction vector.
     * @param dir The viewing direction vector.
     * @param nearplane The distance to the near clipping plane.
     * @param farplane The distance to the far clipping plane.
     * @param fov The vertical field of view, in radians.
     * @param aspectratio The aspect ratio (width / height).
     */
    VX_EXPORT VxFrustum(const VxVector &origin, const VxVector &right, const VxVector &up, const VxVector &dir, float nearplane, float farplane, float fov, float aspectratio);

    /// @brief Gets a mutable reference to the frustum's origin.
    VxVector &GetOrigin() { return m_Origin; }
    /// @brief Gets a const reference to the frustum's origin.
    const VxVector &GetOrigin() const { return m_Origin; }

    /// @brief Gets a mutable reference to the frustum's right vector.
    VxVector &GetRight() { return m_Right; }
    /// @brief Gets a const reference to the frustum's right vector.
    const VxVector &GetRight() const { return m_Right; }

    /// @brief Gets a mutable reference to the frustum's up vector.
    VxVector &GetUp() { return m_Up; }
    /// @brief Gets a const reference to the frustum's up vector.
    const VxVector &GetUp() const { return m_Up; }

    /// @brief Gets a mutable reference to the frustum's direction vector.
    VxVector &GetDir() { return m_Dir; }
    /// @brief Gets a const reference to the frustum's direction vector.
    const VxVector &GetDir() const { return m_Dir; }

    /// @brief Gets a mutable reference to the horizontal bound size at the near plane.
    float &GetRBound() { return m_RBound; }
    /// @brief Gets a const reference to the horizontal bound size at the near plane.
    const float &GetRBound() const { return m_RBound; }

    /// @brief Gets a mutable reference to the vertical bound size at the near plane.
    float &GetUBound() { return m_UBound; }
    /// @brief Gets a const reference to the vertical bound size at the near plane.
    const float &GetUBound() const { return m_UBound; }

    /// @brief Gets a mutable reference to the near plane distance.
    float &GetDMin() { return m_DMin; }
    /// @brief Gets a const reference to the near plane distance.
    const float &GetDMin() const { return m_DMin; }

    /// @brief Gets a mutable reference to the far plane distance.
    float &GetDMax() { return m_DMax; }
    /// @brief Gets a const reference to the far plane distance.
    const float &GetDMax() const { return m_DMax; }

    /// @brief Gets the ratio of far plane distance to near plane distance.
    float GetDRatio() const { return m_DRatio; }
    /// @brief Gets the horizontal field of view factor.
    float GetRF() const { return m_RF; }
    /// @brief Gets the vertical field of view factor.
    float GetUF() const { return m_UF; }

    /// @brief Gets a const reference to the near clipping plane.
    const VxPlane &GetNearPlane() const { return m_NearPlane; }
    /// @brief Gets a const reference to the far clipping plane.
    const VxPlane &GetFarPlane() const { return m_FarPlane; }
    /// @brief Gets a const reference to the left clipping plane.
    const VxPlane &GetLeftPlane() const { return m_LeftPlane; }
    /// @brief Gets a const reference to the right clipping plane.
    const VxPlane &GetRightPlane() const { return m_RightPlane; }
    /// @brief Gets a const reference to the top clipping plane.
    const VxPlane &GetUpPlane() const { return m_UpPlane; }
    /// @brief Gets a const reference to the bottom clipping plane.
    const VxPlane &GetBottomPlane() const { return m_BottomPlane; }

    /**
     * @brief Classifies a point against the frustum's planes.
     * @param v The point to classify.
     * @return A bitmask of VXCLIP flags indicating which planes the point is outside of. Returns 0 if inside.
     */
    XULONG Classify(const VxVector &v) const {
        XULONG flags = 0;
        // Classification of the vertex to the 6 planes
        if (GetNearPlane().Classify(v) > 0.0f) flags |= VXCLIP_FRONT; // the vertex is fully off near
        else if (GetFarPlane().Classify(v) > 0.0f) flags |= VXCLIP_BACK; // the vertex is fully off back
        if (GetLeftPlane().Classify(v) > 0.0f) flags |= VXCLIP_LEFT; // the vertex is fully off left
        else if (GetRightPlane().Classify(v) > 0.0f) flags |= VXCLIP_RIGHT; // the vertex is fully off right
        if (GetBottomPlane().Classify(v) > 0.0f) flags |= VXCLIP_BOTTOM; // the vertex is fully off left
        else if (GetUpPlane().Classify(v) > 0.0f) flags |= VXCLIP_TOP; // the vertex is fully off right
        return flags;
    }

    /**
     * @brief Classifies an axis-aligned bounding box (AABB) against the frustum.
     * @param v The AABB to classify.
     * @return A negative value if intersecting, a positive value if completely outside, and 0 if on a plane.
     */
    float Classify(const VxBbox &v) const {
        float cumul = 1.0f;
        // Classification of the bounding box to the 6 planes
        float f = m_NearPlane.Classify(v);
        if (f > 0.0f) return f; cumul *= f;
        f = m_FarPlane.Classify(v);
        if (f > 0.0f) return f; cumul *= f;
        f = m_LeftPlane.Classify(v);
        if (f > 0.0f) return f; cumul *= f;
        f = m_RightPlane.Classify(v);
        if (f > 0.0f) return f; cumul *= f;
        f = m_UpPlane.Classify(v);
        if (f > 0.0f) return f; cumul *= f;
        f = m_BottomPlane.Classify(v);
        if (f > 0.0f) return f; cumul *= f;
        return -cumul;
    }

    /**
     * @brief Classifies an oriented bounding box (OBB) against the frustum.
     * @param b The original AABB of the box.
     * @param mat The transformation matrix applied to the box.
     * @return A negative value if intersecting, a positive value if completely outside, and 0 if on a plane.
     */
    float Classify(const VxBbox &b, const VxMatrix &mat) const {
        float cumul = 1.0f;

        VxVector axis[4];
        axis[0] = mat[0] * ((b.Max.x - b.Min.x) * 0.5f);
        axis[1] = mat[1] * ((b.Max.y - b.Min.y) * 0.5f);
        axis[2] = mat[2] * ((b.Max.z - b.Min.z) * 0.5f);
        VxVector v = b.GetCenter();
        Vx3DMultiplyMatrixVector(axis + 3, mat, &v);

        // Classification of the bounding box to the 6 planes
        float f = XClassify(axis, m_NearPlane);
        if (f > 0.0f) return f; cumul *= f;
        f = XClassify(axis, m_FarPlane);
        if (f > 0.0f) return f; cumul *= f;
        f = XClassify(axis, m_LeftPlane);
        if (f > 0.0f) return f; cumul *= f;
        f = XClassify(axis, m_RightPlane);
        if (f > 0.0f) return f; cumul *= f;
        f = XClassify(axis, m_UpPlane);
        if (f > 0.0f) return f; cumul *= f;
        f = XClassify(axis, m_BottomPlane);
        if (f > 0.0f) return f; cumul *= f;
        return -cumul;
    }

    /**
     * @brief Checks if a point is inside the frustum.
     * @param v The point to check.
     * @return TRUE if the point is inside or on the boundary of the frustum, FALSE otherwise.
     */
    XBOOL IsInside(const VxVector &v) const {
        if (GetNearPlane().Classify(v) > 0.0f) return FALSE;
        if (GetFarPlane().Classify(v) > 0.0f) return FALSE;
        if (GetLeftPlane().Classify(v) > 0.0f) return FALSE;
        if (GetRightPlane().Classify(v) > 0.0f) return FALSE;
        if (GetBottomPlane().Classify(v) > 0.0f) return FALSE;
        if (GetUpPlane().Classify(v) > 0.0f) return FALSE;
        return TRUE;
    }

    /**
     * @brief Transforms the frustum by an inverse world matrix.
     * @param invworldmat The inverse of the world transformation matrix to apply.
     */
    VX_EXPORT void Transform(const VxMatrix &invworldmat);

    /**
     * @brief Computes the 8 corner vertices of the frustum.
     * @param vertices An array of 8 VxVector to store the computed vertices.
     */
    VX_EXPORT void ComputeVertices(VxVector vertices[8]) const;

    /**
     * @brief Updates the frustum's derived properties (e.g., planes) after modification of its base parameters.
     */
    VX_EXPORT void Update();

    /**
     * @brief Compares this frustum with another for equality.
     * @param iFrustum The frustum to compare against.
     * @return True if all base parameters are equal, false otherwise.
     */
    bool operator==(const VxFrustum &iFrustum) const {
        return (m_Origin == iFrustum.m_Origin) &&
            (m_Right == iFrustum.m_Right) &&
            (m_Up == iFrustum.m_Up) &&
            (m_Dir == iFrustum.m_Dir) &&
            (m_RBound == iFrustum.m_RBound) &&
            (m_UBound == iFrustum.m_UBound) &&
            (m_DMin == iFrustum.m_DMin) &&
            (m_DMax == iFrustum.m_DMax);
    }

protected:
    /**
     * @brief Classifies an OBB against a single plane.
     * @param axis The transformed axes and center of the OBB.
     * @param plane The plane to classify against.
     * @return A negative value if intersecting, a positive value if completely outside, and 0 if on the plane.
     */
    static float XClassify(const VxVector axis[4], const VxPlane &plane) { return plane.XClassify(axis); }

    VxVector m_Origin;  ///< The origin point of the frustum.
    VxVector m_Right;   ///< The right vector of the frustum's orientation.
    VxVector m_Up;      ///< The up vector of the frustum's orientation.
    VxVector m_Dir;     ///< The direction vector of the frustum.

    float m_RBound; ///< Horizontal bound at the near plane.
    float m_UBound; ///< Vertical bound at the near plane.
    float m_DMin;   ///< Near plane distance.
    float m_DMax;   ///< Far plane distance.

    // derived quantities
    float m_DRatio; ///< Ratio of m_DMax / m_DMin.
    float m_RF;     ///< Right factor for plane calculation.
    float m_UF;     ///< Up factor for plane calculation.
    VxPlane m_LeftPlane;    ///< The left clipping plane.
    VxPlane m_RightPlane;   ///< The right clipping plane.
    VxPlane m_UpPlane;      ///< The top clipping plane.
    VxPlane m_BottomPlane;  ///< The bottom clipping plane.
    VxPlane m_NearPlane;    ///< The near clipping plane.
    VxPlane m_FarPlane;     ///< The far clipping plane.
};

#endif // VXFRUSTUM_H
