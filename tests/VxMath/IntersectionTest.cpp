#include <gtest/gtest.h>

#include <cmath>

#include "VxMath.h"

#include "VxIntersect.h"
#include "VxMatrix.h"
#include "VxOBB.h"
#include "VxPlane.h"
#include "VxRay.h"
#include "VxSphere.h"
#include "VxVector.h"

#include "VxMathTestHelpers.h"

using namespace VxMathTest;

static VxRay MakeRay(const VxVector &origin, const VxVector &direction) {
    // Use the (origin, direction, nullptr) overload.
    return VxRay(origin, direction, nullptr);
}

static VxRay MakeSegment(const VxVector &start, const VxVector &end) {
    // VxRay(start, end) stores direction = end-start, which is what VxIntersect segment APIs expect.
    return VxRay(start, end);
}

// =============================================================================
// Spheres
// =============================================================================

TEST(VxIntersect_Spheres, RaySphere_HitMissInsideTangentAndNonUnitDir) {
    VxSphere sphere(VxVector(0.0f, 0.0f, 0.0f), 2.0f);

    // Direct hit (two intersections)
    {
        VxRay ray = MakeRay(VxVector(0.0f, 0.0f, -5.0f), VxVector(0.0f, 0.0f, 1.0f));
        VxVector inter1, inter2;
        int hitCount = VxIntersect::RaySphere(ray, sphere, &inter1, &inter2);
        EXPECT_EQ(hitCount, 2);
        EXPECT_TRUE(VectorNearBool(inter1, VxVector(0.0f, 0.0f, -2.0f), BINARY_TOL));
        EXPECT_TRUE(VectorNearBool(inter2, VxVector(0.0f, 0.0f, 2.0f), BINARY_TOL));
    }

    // Miss
    {
        VxRay missRay = MakeRay(VxVector(5.0f, 0.0f, -5.0f), VxVector(0.0f, 0.0f, 1.0f));
        VxVector inter1 = VxVector(123.0f, 456.0f, 789.0f);
        VxVector inter2 = VxVector(321.0f, 654.0f, 987.0f);
        int hitCount = VxIntersect::RaySphere(missRay, sphere, &inter1, &inter2);
        EXPECT_EQ(hitCount, 0);
        // Ground-truth returns without writing; keep this as a weak non-regression check.
        EXPECT_EQ(inter1, VxVector(123.0f, 456.0f, 789.0f));
        EXPECT_EQ(inter2, VxVector(321.0f, 654.0f, 987.0f));
    }

    // Ray originating inside sphere: inter1 behind origin, inter2 in front.
    {
        VxRay insideRay = MakeRay(VxVector(0.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f));
        VxVector inter1, inter2;
        int hitCount = VxIntersect::RaySphere(insideRay, sphere, &inter1, &inter2);
        EXPECT_EQ(hitCount, 2);
        EXPECT_TRUE(VectorNearBool(inter1, VxVector(-2.0f, 0.0f, 0.0f), BINARY_TOL * 4.0f));
        EXPECT_TRUE(VectorNearBool(inter2, VxVector(2.0f, 0.0f, 0.0f), BINARY_TOL * 4.0f));
    }

    // Tangent (one intersection)
    {
        VxRay tangentRay = MakeRay(VxVector(2.0f, 0.0f, -5.0f), VxVector(0.0f, 0.0f, 1.0f));
        VxVector inter1 = VxVector::axis0();
        VxVector inter2 = VxVector::axis0();
        int hitCount = VxIntersect::RaySphere(tangentRay, sphere, &inter1, &inter2);
        EXPECT_EQ(hitCount, 1);
        EXPECT_TRUE(VectorNearBool(inter1, VxVector(2.0f, 0.0f, 0.0f), BINARY_TOL));
    }

    // Non-unit direction should be normalized internally
    {
        VxRay ray = MakeRay(VxVector(0.0f, 0.0f, -5.0f), VxVector(0.0f, 0.0f, 10.0f));
        VxVector inter1, inter2;
        int hitCount = VxIntersect::RaySphere(ray, sphere, &inter1, &inter2);
        EXPECT_EQ(hitCount, 2);
        EXPECT_TRUE(VectorNearBool(inter1, VxVector(0.0f, 0.0f, -2.0f), BINARY_TOL));
        EXPECT_TRUE(VectorNearBool(inter2, VxVector(0.0f, 0.0f, 2.0f), BINARY_TOL));
    }
}

TEST(VxIntersect_Spheres, SphereSphere_MovingCollision) {
    // Already intersecting
    {
        VxSphere s1(VxVector(0.0f, 0.0f, 0.0f), 1.0f);
        VxSphere s2(VxVector(1.0f, 0.0f, 0.0f), 1.0f);
        float t1 = -1.0f, t2 = -1.0f;
        XBOOL hit = VxIntersect::SphereSphere(s1, s1.Center(), s2, s2.Center(), &t1, &t2);
        EXPECT_TRUE(hit);
        EXPECT_NEAR(t1, 0.0f, BINARY_TOL);
        EXPECT_NEAR(t2, 0.0f, BINARY_TOL);
    }

    // Collision within [0,1]
    {
        VxSphere s1(VxVector(0.0f, 0.0f, 0.0f), 1.0f);
        VxSphere s2(VxVector(5.0f, 0.0f, 0.0f), 1.0f);

        // s2 moves toward s1 over the interval.
        VxVector p1 = s1.Center();
        VxVector p2 = VxVector(0.0f, 0.0f, 0.0f);

        float tEnter = -1.0f, tExit = -1.0f;
        XBOOL hit = VxIntersect::SphereSphere(s1, p1, s2, p2, &tEnter, &tExit);
        EXPECT_TRUE(hit);
        EXPECT_NEAR(tEnter, 0.6f, BINARY_TOL);
        EXPECT_NEAR(tExit, 1.4f, BINARY_TOL);
    }

    // No collision
    {
        VxSphere s1(VxVector(0.0f, 0.0f, 0.0f), 1.0f);
        VxSphere s2(VxVector(5.0f, 0.0f, 0.0f), 1.0f);
        float tEnter = -1.0f, tExit = -1.0f;
        XBOOL hit = VxIntersect::SphereSphere(s1, s1.Center(), s2, VxVector(10.0f, 0.0f, 0.0f), &tEnter, &tExit);
        EXPECT_FALSE(hit);
    }
}

TEST(VxIntersect_Spheres, SphereAABB_BasicCases) {
    VxBbox box(VxVector(-1, -1, -1), VxVector(1, 1, 1));

    EXPECT_TRUE(VxIntersect::SphereAABB(VxSphere(VxVector(0, 0, 0), 0.5f), box));
    EXPECT_TRUE(VxIntersect::SphereAABB(VxSphere(VxVector(1.2f, 0, 0), 0.5f), box));
    EXPECT_TRUE(VxIntersect::SphereAABB(VxSphere(VxVector(1.2f, 1.2f, 0), 0.5f), box));
    EXPECT_TRUE(VxIntersect::SphereAABB(VxSphere(VxVector(1.2f, 1.2f, 1.2f), 0.5f), box));
    EXPECT_FALSE(VxIntersect::SphereAABB(VxSphere(VxVector(2.0f, 0, 0), 0.5f), box));
}

// =============================================================================
// Boxes
// =============================================================================

TEST(VxIntersect_Boxes, RayBox_BooleanAndDetailedVariants) {
    VxBbox box(VxVector(-1, -1, -1), VxVector(1, 1, 1));

    // Hit
    {
        VxRay ray = MakeRay(VxVector(-5, 0, 0), VxVector(1, 0, 0));
        EXPECT_TRUE(VxIntersect::RayBox(ray, box));

        VxVector inPoint, outPoint, inNormal, outNormal;
        int rc = VxIntersect::RayBox(ray, box, inPoint, &outPoint, &inNormal, &outNormal);
        EXPECT_EQ(rc, 1);
        EXPECT_VEC3_NEAR(inPoint, VxVector(-1, 0, 0), BINARY_TOL);
        EXPECT_VEC3_NEAR(outPoint, VxVector(1, 0, 0), BINARY_TOL);
        EXPECT_EQ(inNormal, VxVector(-1, 0, 0));
        // Ground-truth stores the normal opposite to the direction sign for both entry/exit.
        EXPECT_EQ(outNormal, VxVector(-1, 0, 0));
    }

    // Miss
    {
        VxRay ray = MakeRay(VxVector(-5, 5, 0), VxVector(1, 0, 0));
        EXPECT_FALSE(VxIntersect::RayBox(ray, box));
        VxVector inPoint;
        EXPECT_EQ(VxIntersect::RayBox(ray, box, inPoint, nullptr, nullptr, nullptr), 0);
    }

    // Origin inside => detailed variant returns -1
    {
        VxRay ray = MakeRay(VxVector(0, 0, 0), VxVector(1, 0, 0));
        EXPECT_TRUE(VxIntersect::RayBox(ray, box));
        VxVector inPoint, outPoint;
        int rc = VxIntersect::RayBox(ray, box, inPoint, &outPoint, nullptr, nullptr);
        EXPECT_EQ(rc, -1);
        // Implementation computes inPoint at tNear (negative), outPoint at tFar (positive).
        EXPECT_VEC3_NEAR(outPoint, VxVector(1, 0, 0), BINARY_TOL);
    }

    // Parallel-to-slab behavior: direction ~0 on an axis uses [min, max) containment.
    {
        // X is parallel (0), but origin.x == Max.x should be treated as outside.
        VxRay ray = MakeRay(VxVector(1.0f, 0.0f, 0.0f), VxVector(0.0f, 1.0f, 0.0f));
        VxVector inPoint;
        // Ground-truth treats boundary points as inside for the detailed variant.
        EXPECT_EQ(VxIntersect::RayBox(ray, box, inPoint, nullptr, nullptr, nullptr), -1);
    }
}

TEST(VxIntersect_Boxes, SegmentBox_BooleanAndDetailedVariants) {
    VxBbox box(VxVector(-1, -1, -1), VxVector(1, 1, 1));

    // Segment crosses box (t within [0,1])
    {
        VxRay seg = MakeSegment(VxVector(-2, 0, 0), VxVector(2, 0, 0));
        EXPECT_TRUE(VxIntersect::SegmentBox(seg, box));

        VxVector inPoint, outPoint, inNormal, outNormal;
        int rc = VxIntersect::SegmentBox(seg, box, inPoint, &outPoint, &inNormal, &outNormal);
        EXPECT_EQ(rc, 1);
        EXPECT_VEC3_NEAR(inPoint, VxVector(-1, 0, 0), BINARY_TOL);
        EXPECT_VEC3_NEAR(outPoint, VxVector(1, 0, 0), BINARY_TOL);
        EXPECT_EQ(inNormal, VxVector(-1, 0, 0));
        EXPECT_EQ(outNormal, VxVector(-1, 0, 0));
    }

    // Segment entirely outside
    {
        VxRay seg = MakeSegment(VxVector(2, 2, 2), VxVector(3, 2, 2));
        EXPECT_FALSE(VxIntersect::SegmentBox(seg, box));
        VxVector inPoint;
        EXPECT_EQ(VxIntersect::SegmentBox(seg, box, inPoint, nullptr, nullptr, nullptr), 0);
    }

    // Segment fully contains the box but starts inside with no valid entry within [0,1] => 0 per binary rules.
    {
        VxRay seg = MakeSegment(VxVector(0, 0, 0), VxVector(100, 0, 0));
        VxVector inPoint;
        // Ground-truth returns -1 when the segment origin is inside the box.
        EXPECT_EQ(VxIntersect::SegmentBox(seg, box, inPoint, nullptr, nullptr, nullptr), -1);
    }
}

TEST(VxIntersect_Boxes, LineBox_BooleanAndDetailedVariants) {
    VxBbox box(VxVector(-1, -1, -1), VxVector(1, 1, 1));

    // Infinite line through box
    {
        VxRay line = MakeRay(VxVector(-5, 0, 0), VxVector(1, 0, 0));
        EXPECT_TRUE(VxIntersect::LineBox(line, box));

        VxVector inPoint, outPoint, inNormal, outNormal;
        int rc = VxIntersect::LineBox(line, box, inPoint, &outPoint, &inNormal, &outNormal);
        EXPECT_EQ(rc, 1);
        EXPECT_VEC3_NEAR(inPoint, VxVector(-1, 0, 0), BINARY_TOL);
        EXPECT_VEC3_NEAR(outPoint, VxVector(1, 0, 0), BINARY_TOL);
        EXPECT_EQ(inNormal, VxVector(-1, 0, 0));
        EXPECT_EQ(outNormal, VxVector(-1, 0, 0));
    }

    // Parallel line outside slabs
    {
        VxRay line = MakeRay(VxVector(2, 0, 0), VxVector(0, 1, 0));
        EXPECT_FALSE(VxIntersect::LineBox(line, box));
        VxVector inPoint;
        EXPECT_EQ(VxIntersect::LineBox(line, box, inPoint, nullptr, nullptr, nullptr), 0);
    }
}

TEST(VxIntersect_Boxes, AABBAABB_AABBOBB_OBBOBB) {
    VxBbox aabb(VxVector(-1, -1, -1), VxVector(1, 1, 1));

    // AABBAABB
    EXPECT_TRUE(VxIntersect::AABBAABB(aabb, VxBbox(VxVector(1, 0, 0), VxVector(3, 2, 2))));  // touching
    EXPECT_FALSE(VxIntersect::AABBAABB(aabb, VxBbox(VxVector(2.1f, 0, 0), VxVector(3, 2, 2))));

    // AABBOBB
    {
        VxMatrix id;
        Vx3DMatrixIdentity(id);
        VxOBB obb(aabb, id);
        EXPECT_TRUE(VxIntersect::AABBOBB(aabb, obb));

        VxMatrix trans;
        Vx3DMatrixIdentity(trans);
        trans[3][0] = 100.0f;
        VxOBB farObb(aabb, trans);
        EXPECT_FALSE(VxIntersect::AABBOBB(aabb, farObb));

        VxMatrix rot;
        Vx3DMatrixFromEulerAngles(rot, 0.0f, 0.0f, PI / 4.0f);
        VxOBB rotObb(aabb, rot);
        EXPECT_TRUE(VxIntersect::AABBOBB(aabb, rotObb));
    }

    // OBBOBB
    {
        VxMatrix id;
        Vx3DMatrixIdentity(id);
        VxOBB obb0(aabb, id);

        VxMatrix trans;
        Vx3DMatrixIdentity(trans);
        trans[3][0] = 1.5f;
        VxOBB obb1(aabb, trans);
        EXPECT_TRUE(VxIntersect::OBBOBB(obb0, obb1));

        trans[3][0] = 3.0f;
        VxOBB obb2(aabb, trans);
        EXPECT_FALSE(VxIntersect::OBBOBB(obb0, obb2));
    }
}

TEST(VxIntersect_Boxes, AABBFace_TriangleIntersection) {
    VxBbox box(VxVector(-1, -1, -1), VxVector(1, 1, 1));
    VxVector N(0, 0, 1);

    // Triangle vertex inside the box
    EXPECT_TRUE(VxIntersect::AABBFace(
        box,
        VxVector(0.0f, 0.0f, 0.0f),
        VxVector(0.5f, 0.0f, 0.0f),
        VxVector(0.0f, 0.5f, 0.0f),
        N));

    // Triangle outside but edge crosses the box
    EXPECT_TRUE(VxIntersect::AABBFace(
        box,
        VxVector(-2.0f, 0.0f, 0.0f),
        VxVector(2.0f, 0.0f, 0.0f),
        VxVector(0.0f, 2.0f, 0.0f),
        N));

    // Large triangle enclosing the box in projection (diagonal test path)
    EXPECT_TRUE(VxIntersect::AABBFace(
        box,
        VxVector(-10.0f, -10.0f, 0.0f),
        VxVector(10.0f, -10.0f, 0.0f),
        VxVector(0.0f, 10.0f, 0.0f),
        N));

    // Completely separated
    EXPECT_FALSE(VxIntersect::AABBFace(
        box,
        VxVector(10.0f, 10.0f, 10.0f),
        VxVector(12.0f, 10.0f, 10.0f),
        VxVector(10.0f, 12.0f, 10.0f),
        N));
}

// =============================================================================
// Planes
// =============================================================================

TEST(VxIntersect_Planes, RayPlaneAndCulled) {
    VxPlane plane(VxVector(0.0f, 0.0f, 1.0f), VxVector(0.0f, 0.0f, 0.0f));

    // Intersects (front-facing with normal)
    {
        VxRay ray = MakeRay(VxVector(0.0f, 0.0f, 5.0f), VxVector(0.0f, 0.0f, -1.0f));
        VxVector pt;
        float dist = 0.0f;
        EXPECT_TRUE(VxIntersect::RayPlane(ray, plane, pt, dist));
        EXPECT_TRUE(VxIntersect::RayPlaneCulled(ray, plane, pt, dist));
        EXPECT_VEC3_NEAR(pt, VxVector(0.0f, 0.0f, 0.0f), BINARY_TOL);
        EXPECT_NEAR(dist, 5.0f, BINARY_TOL);
    }

    // Back-facing (culled)
    {
        VxRay ray = MakeRay(VxVector(0.0f, 0.0f, -5.0f), VxVector(0.0f, 0.0f, 1.0f));
        VxVector pt;
        float dist = 0.0f;
        EXPECT_TRUE(VxIntersect::RayPlane(ray, plane, pt, dist));
        EXPECT_FALSE(VxIntersect::RayPlaneCulled(ray, plane, pt, dist));
    }

    // Parallel
    {
        VxRay ray = MakeRay(VxVector(0.0f, 0.0f, 1.0f), VxVector(1.0f, 0.0f, 0.0f));
        VxVector pt;
        float dist = 0.0f;
        EXPECT_FALSE(VxIntersect::RayPlane(ray, plane, pt, dist));
        EXPECT_FALSE(VxIntersect::RayPlaneCulled(ray, plane, pt, dist));
    }
}

TEST(VxIntersect_Planes, SegmentPlane_LinePlane) {
    VxPlane plane(VxVector(0.0f, 0.0f, 1.0f), VxVector(0.0f, 0.0f, 0.0f));

    // Segment intersects within [0,1]
    {
        VxRay seg = MakeSegment(VxVector(0.0f, 0.0f, -1.0f), VxVector(0.0f, 0.0f, 1.0f));
        VxVector pt;
        float t = 0.0f;
        EXPECT_TRUE(VxIntersect::SegmentPlane(seg, plane, pt, t));
        EXPECT_VEC3_NEAR(pt, VxVector(0.0f, 0.0f, 0.0f), BINARY_TOL);
        EXPECT_NEAR(t, 0.5f, BINARY_TOL);
    }

    // Segment intersection occurs outside [0,1]
    {
        VxRay seg = MakeSegment(VxVector(0.0f, 0.0f, 2.0f), VxVector(0.0f, 0.0f, 3.0f));
        VxVector pt;
        float t = 0.0f;
        EXPECT_FALSE(VxIntersect::SegmentPlane(seg, plane, pt, t));
    }

    // Segment culled
    {
        VxRay seg = MakeSegment(VxVector(0.0f, 0.0f, -1.0f), VxVector(0.0f, 0.0f, 1.0f));
        VxVector pt;
        float t = 0.0f;
        EXPECT_FALSE(VxIntersect::SegmentPlaneCulled(seg, plane, pt, t));
    }

    // Line always intersects when not parallel
    {
        VxRay line = MakeRay(VxVector(0.0f, 0.0f, 5.0f), VxVector(0.0f, 0.0f, 1.0f));
        VxVector pt;
        float t = 0.0f;
        EXPECT_TRUE(VxIntersect::LinePlane(line, plane, pt, t));
        EXPECT_VEC3_NEAR(pt, VxVector(0.0f, 0.0f, 0.0f), BINARY_TOL);
        EXPECT_NEAR(t, -5.0f, BINARY_TOL);
    }
}

TEST(VxIntersect_Planes, BoxPlane_AndTransformedBoxPlane) {
    VxBbox box(VxVector(-1, -1, -1), VxVector(1, 1, 1));
    VxPlane planeX0(VxVector(1, 0, 0), VxVector(0, 0, 0));

    EXPECT_TRUE(VxIntersect::BoxPlane(box, planeX0));
    EXPECT_FALSE(VxIntersect::BoxPlane(VxBbox(VxVector(2, -1, -1), VxVector(3, 1, 1)), planeX0));
    EXPECT_TRUE(VxIntersect::BoxPlane(VxBbox(VxVector(0, -1, -1), VxVector(1, 1, 1)), planeX0)); // touching

    VxMatrix id;
    Vx3DMatrixIdentity(id);
    EXPECT_TRUE(VxIntersect::BoxPlane(box, id, planeX0));

    VxMatrix trans;
    Vx3DMatrixIdentity(trans);
    trans[3][0] = 10.0f;
    EXPECT_FALSE(VxIntersect::BoxPlane(box, trans, planeX0));
}

TEST(VxIntersect_Planes, Planes_ThreePlaneIntersection) {
    VxPlane px(VxVector(1, 0, 0), VxVector(1, 0, 0));
    VxPlane py(VxVector(0, 1, 0), VxVector(0, 2, 0));
    VxPlane pz(VxVector(0, 0, 1), VxVector(0, 0, 3));

    VxVector p;
    EXPECT_TRUE(VxIntersect::Planes(px, py, pz, p));
    EXPECT_VEC3_NEAR(p, VxVector(1, 2, 3), BINARY_TOL);

    // Degenerate (parallel planes) => determinant 0
    VxPlane px2(VxVector(1, 0, 0), VxVector(2, 0, 0));
    EXPECT_FALSE(VxIntersect::Planes(px, px2, pz, p));
}

// =============================================================================
// Faces (triangles)
// =============================================================================

TEST(VxIntersect_Faces, PointInFace_AndDominantAxisSelection) {
    VxVector p0(0, 0, 0), p1(1, 0, 0), p2(0, 1, 0);
    VxVector n(0, 0, 1);
    int i1 = -1, i2 = -1;

    EXPECT_TRUE(VxIntersect::PointInFace(VxVector(0.25f, 0.25f, 0.0f), p0, p1, p2, n, i1, i2));
    EXPECT_FALSE(VxIntersect::PointInFace(VxVector(1.25f, 0.25f, 0.0f), p0, p1, p2, n, i1, i2));

    // For normal (0,0,1), projection uses X/Y.
    EXPECT_EQ(i1, 0);
    EXPECT_EQ(i2, 1);
}

TEST(VxIntersect_Faces, RayFace_SegmentFace_LineFace_AndCulledVariants) {
    VxVector p0(0, 0, 0), p1(5, 0, 0), p2(0, 5, 0);
    VxVector n(0, 0, 1);
    VxVector hit;
    float dist = 0.0f;
    int i1 = -1, i2 = -1;

    // Ray hit
    {
        VxRay ray = MakeRay(VxVector(1, 1, 5), VxVector(0, 0, -1));
        EXPECT_TRUE(VxIntersect::RayFace(ray, p0, p1, p2, n, hit, dist));
        EXPECT_VEC3_NEAR(hit, VxVector(1, 1, 0), BINARY_TOL);
        EXPECT_TRUE(VxIntersect::RayFace(ray, p0, p1, p2, n, hit, dist, i1, i2));
        EXPECT_EQ(i1, 0);
        EXPECT_EQ(i2, 1);
    }

    // Ray miss
    {
        VxRay ray = MakeRay(VxVector(6, 6, 5), VxVector(0, 0, -1));
        EXPECT_FALSE(VxIntersect::RayFace(ray, p0, p1, p2, n, hit, dist));
    }

    // RayFaceCulled should reject back-facing
    {
        VxRay rayBack = MakeRay(VxVector(1, 1, -5), VxVector(0, 0, 1));
        EXPECT_TRUE(VxIntersect::RayFace(rayBack, p0, p1, p2, n, hit, dist));
        EXPECT_FALSE(VxIntersect::RayFaceCulled(rayBack, p0, p1, p2, n, hit, dist, i1, i2));
    }

    // Segment hit
    {
        VxRay seg = MakeSegment(VxVector(1, 1, 1), VxVector(1, 1, -1));
        EXPECT_TRUE(VxIntersect::SegmentFace(seg, p0, p1, p2, n, hit, dist));
        EXPECT_VEC3_NEAR(hit, VxVector(1, 1, 0), BINARY_TOL);
        EXPECT_TRUE(VxIntersect::SegmentFace(seg, p0, p1, p2, n, hit, dist, i1, i2));
    }

    // Segment culled (approaching from back)
    {
        VxRay seg = MakeSegment(VxVector(1, 1, -1), VxVector(1, 1, 1));
        EXPECT_TRUE(VxIntersect::SegmentFace(seg, p0, p1, p2, n, hit, dist));
        EXPECT_FALSE(VxIntersect::SegmentFaceCulled(seg, p0, p1, p2, n, hit, dist, i1, i2));
    }

    // Line hit even with negative t
    {
        VxRay line = MakeRay(VxVector(1, 1, -5), VxVector(0, 0, 1));
        EXPECT_TRUE(VxIntersect::LineFace(line, p0, p1, p2, n, hit, dist));
        EXPECT_VEC3_NEAR(hit, VxVector(1, 1, 0), BINARY_TOL);
        EXPECT_TRUE(VxIntersect::LineFace(line, p0, p1, p2, n, hit, dist, i1, i2));
    }
}

TEST(VxIntersect_Faces, GetPointCoefficients_BarycentricBasics) {
    VxVector p0(0, 0, 0), p1(1, 0, 0), p2(0, 1, 0);
    VxVector n(0, 0, 1);

    const VxVector p(0.25f, 0.25f, 0.0f);
    int i1 = -1, i2 = -1;
    ASSERT_TRUE(VxIntersect::PointInFace(p, p0, p1, p2, n, i1, i2));

    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f;
    VxIntersect::GetPointCoefficients(p, p0, p1, p2, i1, i2, v0, v1, v2);

    EXPECT_NEAR(v0 + v1 + v2, 1.0f, BINARY_TOL);
    EXPECT_NEAR(v0, 0.5f, BINARY_TOL);
    EXPECT_NEAR(v1, 0.25f, BINARY_TOL);
    EXPECT_NEAR(v2, 0.25f, BINARY_TOL);
}

TEST(VxIntersect_Faces, FaceFace_BasicOverlapAndSeparation) {
    VxVector a0(0, 0, 0), a1(1, 0, 0), a2(0, 1, 0);
    VxVector n0(0, 0, 1);

    // Overlapping coplanar triangles
    {
        VxVector b0(0.5f, 0.0f, 0.0f), b1(1.5f, 0.0f, 0.0f), b2(0.5f, 1.0f, 0.0f);
        VxVector n1(0, 0, 1);
        EXPECT_TRUE(VxIntersect::FaceFace(a0, a1, a2, n0, b0, b1, b2, n1));
    }

    // Separated coplanar triangles
    {
        VxVector b0(10.0f, 10.0f, 0.0f), b1(11.0f, 10.0f, 0.0f), b2(10.0f, 11.0f, 0.0f);
        VxVector n1(0, 0, 1);
        EXPECT_FALSE(VxIntersect::FaceFace(a0, a1, a2, n0, b0, b1, b2, n1));
    }

    // Parallel planes, separated along Z
    {
        VxVector b0(0.5f, 0.0f, 1.0f), b1(1.5f, 0.0f, 1.0f), b2(0.5f, 1.0f, 1.0f);
        VxVector n1(0, 0, 1);
        EXPECT_FALSE(VxIntersect::FaceFace(a0, a1, a2, n0, b0, b1, b2, n1));
    }
}