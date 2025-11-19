#ifndef VXINTERSECT_H
#define VXINTERSECT_H

#include "VxMathDefines.h"

struct VxVector;
struct VxBbox;
class VxMatrix;
class VxRay;
class VxOBB;
class VxPlane;
class VxSphere;
class VxFrustum;

/**
 * @brief Provides static methods for various intersection tests.
 *
 * @remarks
 * The VxIntersect class is a collection of static methods for performing
 * intersection tests between various geometric primitives, such as rays,
 * segments, lines, boxes (AABB and OBB), planes, triangles (faces), spheres, and frustums.
 *
 * @see VxPlane, VxRay, VxBbox, VxVector
 */
class VxIntersect {
public:
    //----------- Boxes

    /// @brief Checks for intersection between a ray and an axis-aligned bounding box (AABB).
    VX_EXPORT static XBOOL RayBox(const VxRay &ray, const VxBbox &box);
    /**
     * @brief Finds the intersection points of a ray with an AABB.
     * @return Returns TRUE if they intersect. Returns -1 in inpoint.x if the ray's origin is inside the box.
     */
    VX_EXPORT static XBOOL RayBox(const VxRay &ray, const VxBbox &box, VxVector &inpoint, VxVector *outpoint = NULL, VxVector *innormal = NULL, VxVector *outnormal = NULL);

    /// @brief Checks for intersection between a line segment and an AABB.
    VX_EXPORT static XBOOL SegmentBox(const VxRay &ray, const VxBbox &box);
    /**
     * @brief Finds the intersection points of a line segment with an AABB.
     * @return Returns TRUE if they intersect. Returns -1 in inpoint.x if the segment's origin is inside the box.
     */
    VX_EXPORT static XBOOL SegmentBox(const VxRay &ray, const VxBbox &box, VxVector &inpoint, VxVector *outpoint = NULL, VxVector *innormal = NULL, VxVector *outnormal = NULL);

    /// @brief Checks for intersection between an infinite line and an AABB.
    VX_EXPORT static XBOOL LineBox(const VxRay &ray, const VxBbox &box);
    /// @brief Finds the intersection points of an infinite line with an AABB.
    VX_EXPORT static XBOOL LineBox(const VxRay &ray, const VxBbox &box, VxVector &inpoint, VxVector *outpoint = NULL, VxVector *innormal = NULL, VxVector *outnormal = NULL);

    /// @brief Checks for intersection between two AABBs.
    VX_EXPORT static XBOOL AABBAABB(const VxBbox &box1, const VxBbox &box2);
    /// @brief Checks for intersection between an AABB and an oriented bounding box (OBB).
    VX_EXPORT static XBOOL AABBOBB(const VxBbox &box1, const VxOBB &box2);
    /// @brief Checks for intersection between two OBBs.
    VX_EXPORT static XBOOL OBBOBB(const VxOBB &box1, const VxOBB &box2);
    /// @brief Checks for intersection between an AABB and a triangle face.
    VX_EXPORT static XBOOL AABBFace(const VxBbox &box1, const VxVector &A0, const VxVector &A1, const VxVector &A2, const VxVector &N);

    //---------- Planes

    /// @brief Intersects a ray with a plane.
    VX_EXPORT static XBOOL RayPlane(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist);
    /// @brief Intersects a ray with a plane, culled by the plane's normal.
    VX_EXPORT static XBOOL RayPlaneCulled(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist);
    /// @brief Intersects a line segment with a plane.
    VX_EXPORT static XBOOL SegmentPlane(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist);
    /// @brief Intersects a line segment with a plane, culled by the plane's normal.
    VX_EXPORT static XBOOL SegmentPlaneCulled(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist);
    /// @brief Intersects an infinite line with a plane.
    VX_EXPORT static XBOOL LinePlane(const VxRay &ray, const VxPlane &plane, VxVector &point, float &dist);
    /// @brief Checks for intersection between an AABB and a plane.
    VX_EXPORT static XBOOL BoxPlane(const VxBbox &box, const VxPlane &plane);
    /// @brief Checks for intersection between a transformed AABB (OBB) and a plane.
    VX_EXPORT static XBOOL BoxPlane(const VxBbox &box, const VxMatrix &mat, const VxPlane &plane);
    /// @brief Checks for intersection between a triangle face and a plane.
    VX_EXPORT static XBOOL FacePlane(const VxVector &A0, const VxVector &A1, const VxVector &A2, const VxPlane &plane);
    /// @brief Finds the intersection point of three planes.
    VX_EXPORT static XBOOL Planes(const VxPlane &plane1, const VxPlane &plane2, const VxPlane &plane3, VxVector &p);

    //---------- Faces

    /// @brief Checks if a point lies within the 2D boundaries of a triangle face.
    VX_EXPORT static XBOOL PointInFace(const VxVector &point, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, int &i1, int &i2);
    /// @brief Intersects a ray with a triangle face.
    VX_EXPORT static XBOOL RayFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, VxVector &res, float &dist);
    /// @brief Intersects a ray with a triangle face, providing barycentric coordinate information.
    VX_EXPORT static XBOOL RayFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2);
    /// @brief Intersects a ray with a front-facing triangle face.
    VX_EXPORT static XBOOL RayFaceCulled(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2);
    /// @brief Intersects a line segment with a triangle face.
    VX_EXPORT static XBOOL SegmentFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, VxVector &res, float &dist);
    /// @brief Intersects a line segment with a triangle face, providing barycentric coordinate information.
    VX_EXPORT static XBOOL SegmentFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2);
    /// @brief Intersects a line segment with a front-facing triangle face.
    VX_EXPORT static XBOOL SegmentFaceCulled(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2);
    /// @brief Intersects an infinite line with a triangle face.
    VX_EXPORT static XBOOL LineFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, VxVector &res, float &dist);
    /// @brief Intersects an infinite line with a triangle face, providing barycentric coordinate information.
    VX_EXPORT static XBOOL LineFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2);

    /// @brief Calculates barycentric coordinates for a point within a triangle face.
    VX_EXPORT static void GetPointCoefficients(const VxVector &pt, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2, const int &i1, const int &i2, float &V0Coef, float &V1Coef, float &V2Coef);

    /// @brief Checks for intersection between two triangle faces.
    VX_EXPORT static XBOOL FaceFace(const VxVector &A0, const VxVector &A1, const VxVector &A2, const VxVector &N0, const VxVector &B0, const VxVector &B1, const VxVector &B2, const VxVector &N1);

    //--------- Frustum

    /// @brief Checks for intersection between a frustum and a triangle face.
    VX_EXPORT static XBOOL FrustumFace(const VxFrustum &frustum, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2);
    /// @brief Checks for intersection between a frustum and an AABB.
    VX_EXPORT static XBOOL FrustumAABB(const VxFrustum &frustum, const VxBbox &box);
    /// @brief Checks for intersection between a frustum and an OBB.
    VX_EXPORT static XBOOL FrustumOBB(const VxFrustum &frustum, const VxBbox &box, const VxMatrix &mat);
    /**
     * @brief Checks for intersection between a frustum and a box (OBB).
     * @deprecated Use FrustumOBB instead.
     */
    VX_EXPORT static XBOOL FrustumBox(const VxFrustum &frustum, const VxBbox &box, const VxMatrix &mat);

    //--------- Spheres
    /// @brief Checks for collision between two moving spheres.
    VX_EXPORT static XBOOL SphereSphere(const VxSphere &iS1, const VxVector &iP1, const VxSphere &iS2, const VxVector &iP2, float *oCollisionTime1, float *oCollisionTime2);
    /// @brief Finds the intersection points of a ray with a sphere. Returns the number of intersections (0, 1, or 2).
    VX_EXPORT static int RaySphere(const VxRay &iRay, const VxSphere &iSphere, VxVector *oInter1, VxVector *oInter2);
    /// @brief Checks for intersection between a sphere and an AABB.
    VX_EXPORT static int SphereAABB(const VxSphere &iSphere, const VxBbox &iBox);
};

#endif // VXINTERSECT_H
