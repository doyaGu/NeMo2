#ifndef VXDISTANCE_H
#define VXDISTANCE_H

#include <math.h>

#include "VxMathDefines.h"

class VxRay;
struct VxVector;

/**
 * @brief Provides static methods for various distance calculations.
 *
 * @remarks
 * The VxDistance class contains a collection of static methods for calculating
 * distances between different geometric primitives such as points, lines, rays,
 * and segments. It offers methods for both squared distance (for efficient
 * comparisons) and standard distance.
 *
 * @see VxPlane, VxRay, VxBbox, VxVector
 */
class VxDistance {
public:
    // Lines - Lines Distances

    /**
     * @brief Calculates the squared distance between two 3D lines.
     * @param line0 The first line, represented by a VxRay.
     * @param line1 The second line, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the first line. Can be NULL.
     * @param t1 Pointer to a float to store the parameter of the closest point on the second line. Can be NULL.
     * @return The squared distance between the two lines.
     * @remarks Using squared distance is faster for comparisons as it avoids a square root operation.
     */
    VX_EXPORT static float LineLineSquareDistance(const VxRay &line0, const VxRay &line1, float *t0 = NULL, float *t1 = NULL);

    /**
     * @brief Calculates the squared distance between a line and a ray.
     * @param line The line, represented by a VxRay.
     * @param ray The ray, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the line. Can be NULL.
     * @param t1 Pointer to a float to store the parameter of the closest point on the ray. Can be NULL.
     * @return The squared distance between the line and the ray.
     */
    VX_EXPORT static float LineRaySquareDistance(const VxRay &line, const VxRay &ray, float *t0 = NULL, float *t1 = NULL);

    /**
     * @brief Calculates the squared distance between a line and a segment.
     * @param line The line, represented by a VxRay.
     * @param segment The segment, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the line. Can be NULL.
     * @param t1 Pointer to a float to store the parameter of the closest point on the segment. Can be NULL.
     * @return The squared distance between the line and the segment.
     */
    VX_EXPORT static float LineSegmentSquareDistance(const VxRay &line, const VxRay &segment, float *t0 = NULL, float *t1 = NULL);

    /**
     * @brief Calculates the squared distance between two rays.
     * @param ray0 The first ray, represented by a VxRay.
     * @param ray1 The second ray, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the first ray. Can be NULL.
     * @param t1 Pointer to a float to store the parameter of the closest point on the second ray. Can be NULL.
     * @return The squared distance between the two rays.
     */
    VX_EXPORT static float RayRaySquareDistance(const VxRay &ray0, const VxRay &ray1, float *t0 = NULL, float *t1 = NULL);

    /**
     * @brief Calculates the squared distance between a ray and a segment.
     * @param ray The ray, represented by a VxRay.
     * @param segment The segment, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the ray. Can be NULL.
     * @param t1 Pointer to a float to store the parameter of the closest point on the segment. Can be NULL.
     * @return The squared distance between the ray and the segment.
     */
    VX_EXPORT static float RaySegmentSquareDistance(const VxRay &ray, const VxRay &segment, float *t0 = NULL, float *t1 = NULL);

    /**
     * @brief Calculates the squared distance between two segments.
     * @param segment0 The first segment, represented by a VxRay.
     * @param segment1 The second segment, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the first segment. Can be NULL.
     * @param t1 Pointer to a float to store the parameter of the closest point on the second segment. Can be NULL.
     * @return The squared distance between the two segments.
     */
    VX_EXPORT static float SegmentSegmentSquareDistance(const VxRay &segment0, const VxRay &segment1, float *t0 = NULL, float *t1 = NULL);

    /**
     * @brief Calculates the distance between two 3D lines.
     * @return The distance between the two lines.
     */
    static float LineLineDistance(const VxRay &line0, const VxRay &line1, float *t0 = NULL, float *t1 = NULL) {
        return sqrtf(LineLineSquareDistance(line0, line1, t0, t1));
    }

    /**
     * @brief Calculates the distance between a line and a ray.
     * @return The distance between the line and the ray.
     */
    static float LineRayDistance(const VxRay &line, const VxRay &ray, float *t0 = NULL, float *t1 = NULL) {
        return sqrtf(LineRaySquareDistance(line, ray, t0, t1));
    }

    /**
     * @brief Calculates the distance between a line and a segment.
     * @return The distance between the line and the segment.
     */
    static float LineSegmentDistance(const VxRay &line, const VxRay &segment, float *t0 = NULL, float *t1 = NULL) {
        return sqrtf(LineSegmentSquareDistance(line, segment, t0, t1));
    }

    /**
     * @brief Calculates the distance between two rays.
     * @return The distance between the two rays.
     */
    static float RayRayDistance(const VxRay &ray0, const VxRay &ray1, float *t0 = NULL, float *t1 = NULL) {
        return sqrtf(RayRaySquareDistance(ray0, ray1, t0, t1));
    }

    /**
     * @brief Calculates the distance between a ray and a segment.
     * @return The distance between the ray and the segment.
     */
    static float RaySegmentDistance(const VxRay &ray, const VxRay &segment, float *t0 = NULL, float *t1 = NULL) {
        return sqrtf(RaySegmentSquareDistance(ray, segment, t0, t1));
    }

    /**
     * @brief Calculates the distance between two segments.
     * @return The distance between the two segments.
     */
    static float SegmentSegmentDistance(const VxRay &segment0, const VxRay &segment1, float *t0 = NULL, float *t1 = NULL) {
        return sqrtf(SegmentSegmentSquareDistance(segment0, segment1, t0, t1));
    }

    // Line - Point distance

    /**
     * @brief Calculates the squared distance between a point and a line.
     * @param point The point.
     * @param line1 The line, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the line. Can be NULL.
     * @return The squared distance between the point and the line.
     */
    VX_EXPORT static float PointLineSquareDistance(const VxVector &point, const VxRay &line1, float *t0 = NULL);

    /**
     * @brief Calculates the squared distance between a point and a ray.
     * @param point The point.
     * @param ray The ray, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the ray. Can be NULL.
     * @return The squared distance between the point and the ray.
     */
    VX_EXPORT static float PointRaySquareDistance(const VxVector &point, const VxRay &ray, float *t0 = NULL);

    /**
     * @brief Calculates the squared distance between a point and a segment.
     * @param point The point.
     * @param segment The segment, represented by a VxRay.
     * @param t0 Pointer to a float to store the parameter of the closest point on the segment. Can be NULL.
     * @return The squared distance between the point and the segment.
     */
    VX_EXPORT static float PointSegmentSquareDistance(const VxVector &point, const VxRay &segment, float *t0 = NULL);

    /**
     * @brief Calculates the distance between a point and a line.
     * @return The distance between the point and the line.
     */
    static float PointLineDistance(const VxVector &point, const VxRay &line, float *t0 = NULL) {
        return sqrtf(PointLineSquareDistance(point, line, t0));
    }

    /**
     * @brief Calculates the distance between a point and a ray.
     * @return The distance between the point and the ray.
     */
    static float PointRayDistance(const VxVector &point, const VxRay &ray, float *t0 = NULL) {
        return sqrtf(PointRaySquareDistance(point, ray, t0));
    }

    /**
     * @brief Calculates the distance between a point and a segment.
     * @return The distance between the point and the segment.
     */
    static float PointSegmentDistance(const VxVector &point, const VxRay &segment, float *t0 = NULL) {
        return sqrtf(PointSegmentSquareDistance(point, segment, t0));
    }
};

#endif // VXDISTANCE_H