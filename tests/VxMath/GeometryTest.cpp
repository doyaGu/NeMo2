#include <gtest/gtest.h>
#include <cmath>
#include "VxPlane.h"
#include "VxRay.h"
#include "VxSphere.h"
#include "VxMatrix.h"
#include "VxOBB.h"
#include "VxIntersect.h"
#include "VxDistance.h"

class VxPlaneTest : public ::testing::Test {
protected:
    void SetUp() override {
        // XY plane at Z=0
        xy_plane = VxPlane(0.0f, 0.0f, 1.0f, 0.0f);

        // Plane passing through origin with normal (1,1,1)
        normal_plane = VxPlane(VxVector(1.0f, 1.0f, 1.0f), 0.0f);

        // Plane from three points forming a triangle
        VxVector p1(1.0f, 0.0f, 0.0f);
        VxVector p2(0.0f, 1.0f, 0.0f);
        VxVector p3(0.0f, 0.0f, 0.0f);
        triangle_plane = VxPlane(p1, p2, p3);
    }

    VxPlane xy_plane, normal_plane, triangle_plane;
};

TEST_F(VxPlaneTest, DefaultConstructor) {
    VxPlane plane;
    EXPECT_FLOAT_EQ(plane.GetNormal().x, 0.0f);
    EXPECT_FLOAT_EQ(plane.GetNormal().y, 0.0f);
    EXPECT_FLOAT_EQ(plane.GetNormal().z, 0.0f);
    EXPECT_FLOAT_EQ(plane.m_D, 0.0f);
}

TEST_F(VxPlaneTest, NormalAndDConstructor) {
    VxVector normal(1.0f, 0.0f, 0.0f);
    float d = -5.0f;
    VxPlane plane(normal, d);

    EXPECT_TRUE(plane.GetNormal() == normal);
    EXPECT_FLOAT_EQ(plane.m_D, d);
}

TEST_F(VxPlaneTest, ComponentConstructor) {
    VxPlane plane(1.0f, 2.0f, 3.0f, 4.0f);

    EXPECT_FLOAT_EQ(plane.GetNormal().x, 1.0f);
    EXPECT_FLOAT_EQ(plane.GetNormal().y, 2.0f);
    EXPECT_FLOAT_EQ(plane.GetNormal().z, 3.0f);
    EXPECT_FLOAT_EQ(plane.m_D, 4.0f);
}

TEST_F(VxPlaneTest, NormalAndPointConstructor) {
    VxVector normal(0.0f, 1.0f, 0.0f);
    VxVector point(3.0f, 5.0f, 7.0f);
    VxPlane plane(normal, point);

    // Point should be on the plane (distance = 0)
    EXPECT_NEAR(plane.Distance(point), 0.0f, EPSILON);
}

TEST_F(VxPlaneTest, ThreePointConstructor) {
    // The triangle plane should be the XY plane (Z=0)
    EXPECT_NEAR(triangle_plane.GetNormal().x, 0.0f, EPSILON);
    EXPECT_NEAR(triangle_plane.GetNormal().y, 0.0f, EPSILON);
    EXPECT_NEAR(XAbs(triangle_plane.GetNormal().z), 1.0f, EPSILON); // Could be +1 or -1
    EXPECT_NEAR(triangle_plane.m_D, 0.0f, EPSILON);
}

TEST_F(VxPlaneTest, ClassifyPoint) {
    VxVector point_above(0.0f, 0.0f, 5.0f);
    VxVector point_below(0.0f, 0.0f, -3.0f);
    VxVector point_on(1.0f, 2.0f, 0.0f);

    EXPECT_GT(xy_plane.Classify(point_above), 0.0f);
    EXPECT_LT(xy_plane.Classify(point_below), 0.0f);
    EXPECT_NEAR(xy_plane.Classify(point_on), 0.0f, EPSILON);
}

TEST_F(VxPlaneTest, ClassifyAABB) {
    VxBbox box_above(VxVector(0.0f, 0.0f, 1.0f), VxVector(1.0f, 1.0f, 2.0f));
    VxBbox box_below(VxVector(0.0f, 0.0f, -2.0f), VxVector(1.0f, 1.0f, -1.0f));
    VxBbox box_intersecting(VxVector(0.0f, 0.0f, -1.0f), VxVector(1.0f, 1.0f, 1.0f));

    EXPECT_GT(xy_plane.Classify(box_above), 0.0f);
    EXPECT_LT(xy_plane.Classify(box_below), 0.0f);
    EXPECT_FLOAT_EQ(xy_plane.Classify(box_intersecting), 0.0f);
}

TEST_F(VxPlaneTest, ClassifyFace) {
    VxVector v1(0.0f, 0.0f, 1.0f);
    VxVector v2(1.0f, 0.0f, 1.0f);
    VxVector v3(0.5f, 1.0f, 1.0f);

    VxVector v4(0.0f, 0.0f, -1.0f);
    VxVector v5(1.0f, 0.0f, -1.0f);
    VxVector v6(0.5f, 1.0f, -1.0f);

    VxVector v7(0.0f, 0.0f, -1.0f);
    VxVector v8(1.0f, 0.0f, 1.0f);
    VxVector v9(0.5f, 1.0f, 0.0f);

    // Face entirely above plane
    EXPECT_GT(xy_plane.ClassifyFace(v1, v2, v3), 0.0f);

    // Face entirely below plane
    EXPECT_LT(xy_plane.ClassifyFace(v4, v5, v6), 0.0f);

    // Face crossing plane
    EXPECT_FLOAT_EQ(xy_plane.ClassifyFace(v7, v8, v9), 0.0f);
}

TEST_F(VxPlaneTest, Distance) {
    VxVector point(0.0f, 0.0f, 5.0f);
    EXPECT_FLOAT_EQ(xy_plane.Distance(point), 5.0f);

    VxVector point2(0.0f, 0.0f, -3.0f);
    EXPECT_FLOAT_EQ(xy_plane.Distance(point2), 3.0f);
}

TEST_F(VxPlaneTest, NearestPoint) {
    VxVector point(3.0f, 4.0f, 5.0f);
    VxVector nearest = xy_plane.NearestPoint(point);

    EXPECT_FLOAT_EQ(nearest.x, 3.0f);
    EXPECT_FLOAT_EQ(nearest.y, 4.0f);
    EXPECT_FLOAT_EQ(nearest.z, 0.0f);

    // The nearest point should be on the plane
    EXPECT_NEAR(xy_plane.Distance(nearest), 0.0f, EPSILON);
}

TEST_F(VxPlaneTest, UnaryMinus) {
    VxPlane negated = -xy_plane;

    EXPECT_FLOAT_EQ(negated.GetNormal().x, -xy_plane.GetNormal().x);
    EXPECT_FLOAT_EQ(negated.GetNormal().y, -xy_plane.GetNormal().y);
    EXPECT_FLOAT_EQ(negated.GetNormal().z, -xy_plane.GetNormal().z);
    EXPECT_FLOAT_EQ(negated.m_D, -xy_plane.m_D);
}

TEST_F(VxPlaneTest, EqualityOperator) {
    VxPlane plane1(1.0f, 2.0f, 3.0f, 4.0f);
    VxPlane plane2(1.0f, 2.0f, 3.0f, 4.0f);
    VxPlane plane3(1.0f, 2.0f, 3.0f, 5.0f);

    EXPECT_TRUE(plane1 == plane2);
    EXPECT_FALSE(plane1 == plane3);
}

class VxRayTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ray from origin along X-axis
        origin_x_ray = VxRay(VxVector(0.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f), nullptr);

        // Ray from (1,1,1) to (2,2,2)
        diagonal_ray = VxRay(VxVector(1.0f, 1.0f, 1.0f), VxVector(2.0f, 2.0f, 2.0f));
    }

    VxRay origin_x_ray, diagonal_ray;
};

TEST_F(VxRayTest, DefaultConstructor) {
    VxRay ray;
    // Default constructor doesn't initialize, so we don't test specific values
}

TEST_F(VxRayTest, StartEndConstructor) {
    VxVector start(1.0f, 2.0f, 3.0f);
    VxVector end(4.0f, 5.0f, 6.0f);
    VxRay ray(start, end);

    EXPECT_TRUE(ray.GetOrigin() == start);
    EXPECT_FLOAT_EQ(ray.GetDirection().x, 3.0f);
    EXPECT_FLOAT_EQ(ray.GetDirection().y, 3.0f);
    EXPECT_FLOAT_EQ(ray.GetDirection().z, 3.0f);
}

TEST_F(VxRayTest, StartDirectionConstructor) {
    VxVector start(1.0f, 2.0f, 3.0f);
    VxVector direction(0.0f, 1.0f, 0.0f);
    VxRay ray(start, direction, nullptr);

    EXPECT_TRUE(ray.GetOrigin() == start);
    EXPECT_TRUE(ray.GetDirection() == direction);
}

TEST_F(VxRayTest, Interpolate) {
    VxVector point;
    origin_x_ray.Interpolate(point, 5.0f);

    EXPECT_FLOAT_EQ(point.x, 5.0f);
    EXPECT_FLOAT_EQ(point.y, 0.0f);
    EXPECT_FLOAT_EQ(point.z, 0.0f);

    // Test with diagonal ray
    diagonal_ray.Interpolate(point, 0.5f);
    EXPECT_FLOAT_EQ(point.x, 1.5f);
    EXPECT_FLOAT_EQ(point.y, 1.5f);
    EXPECT_FLOAT_EQ(point.z, 1.5f);
}

TEST_F(VxRayTest, DistanceToPoint) {
    // Point on the ray
    VxVector point_on_ray(3.0f, 0.0f, 0.0f);
    EXPECT_NEAR(origin_x_ray.Distance(point_on_ray), 0.0f, EPSILON);

    // Point off the ray
    VxVector point_off_ray(2.0f, 3.0f, 0.0f);
    EXPECT_NEAR(origin_x_ray.Distance(point_off_ray), 3.0f, EPSILON);
}

TEST_F(VxRayTest, SquareDistanceToPoint) {
    VxVector point_off_ray(2.0f, 4.0f, 0.0f);
    float square_dist = origin_x_ray.SquareDistance(point_off_ray);
    float dist = origin_x_ray.Distance(point_off_ray);

    EXPECT_NEAR(square_dist, dist * dist, EPSILON);
    EXPECT_NEAR(square_dist, 16.0f, EPSILON);
}

TEST_F(VxRayTest, EqualityOperator) {
    VxRay ray1(VxVector(0.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f), nullptr);
    VxRay ray2(VxVector(0.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f), nullptr);
    VxRay ray3(VxVector(1.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f), nullptr);

    EXPECT_TRUE(ray1 == ray2);
    EXPECT_FALSE(ray1 == ray3);
}

TEST_F(VxRayTest, Accessors) {
    VxVector new_origin(5.0f, 6.0f, 7.0f);
    VxVector new_direction(8.0f, 9.0f, 10.0f);

    origin_x_ray.GetOrigin() = new_origin;
    origin_x_ray.GetDirection() = new_direction;

    EXPECT_TRUE(origin_x_ray.GetOrigin() == new_origin);
    EXPECT_TRUE(origin_x_ray.GetDirection() == new_direction);
}

class VxSphereTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Unit sphere at origin
        unit_sphere = VxSphere(VxVector(0.0f, 0.0f, 0.0f), 1.0f);

        // Sphere at (5,5,5) with radius 3
        offset_sphere = VxSphere(VxVector(5.0f, 5.0f, 5.0f), 3.0f);
    }

    VxSphere unit_sphere, offset_sphere;
};

TEST_F(VxSphereTest, DefaultConstructor) {
    VxSphere sphere;
    // Default constructor doesn't initialize, so we don't test specific values
}

TEST_F(VxSphereTest, ParameterizedConstructor) {
    VxVector center(1.0f, 2.0f, 3.0f);
    float radius = 5.0f;
    VxSphere sphere(center, radius);

    EXPECT_TRUE(sphere.Center() == center);
    EXPECT_FLOAT_EQ(sphere.Radius(), radius);
}

TEST_F(VxSphereTest, Accessors) {
    VxVector new_center(10.0f, 11.0f, 12.0f);
    float new_radius = 7.0f;

    unit_sphere.Center() = new_center;
    unit_sphere.Radius() = new_radius;

    EXPECT_TRUE(unit_sphere.Center() == new_center);
    EXPECT_FLOAT_EQ(unit_sphere.Radius(), new_radius);
}

TEST_F(VxSphereTest, ConstAccessors) {
    const VxSphere& const_sphere = unit_sphere;

    const VxVector& center = const_sphere.Center();
    const float& radius = const_sphere.Radius();

    EXPECT_FLOAT_EQ(center.x, 0.0f);
    EXPECT_FLOAT_EQ(center.y, 0.0f);
    EXPECT_FLOAT_EQ(center.z, 0.0f);
    EXPECT_FLOAT_EQ(radius, 1.0f);
}

TEST_F(VxSphereTest, IsPointInside) {
    // Point at center
    EXPECT_TRUE(unit_sphere.IsPointInside(VxVector(0.0f, 0.0f, 0.0f)));

    // Point on surface
    EXPECT_TRUE(unit_sphere.IsPointInside(VxVector(1.0f, 0.0f, 0.0f)));

    // Point inside
    EXPECT_TRUE(unit_sphere.IsPointInside(VxVector(0.5f, 0.5f, 0.0f)));

    // Point outside
    EXPECT_FALSE(unit_sphere.IsPointInside(VxVector(2.0f, 0.0f, 0.0f)));

    // Point outside (barely)
    EXPECT_FALSE(unit_sphere.IsPointInside(VxVector(1.1f, 0.0f, 0.0f)));
}

TEST_F(VxSphereTest, IsPointOnSurface) {
    // Point on surface
    EXPECT_TRUE(unit_sphere.IsPointOnSurface(VxVector(1.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(unit_sphere.IsPointOnSurface(VxVector(0.0f, 1.0f, 0.0f)));
    EXPECT_TRUE(unit_sphere.IsPointOnSurface(VxVector(0.0f, 0.0f, -1.0f)));

    // Point at center
    EXPECT_FALSE(unit_sphere.IsPointOnSurface(VxVector(0.0f, 0.0f, 0.0f)));

    // Point outside
    EXPECT_FALSE(unit_sphere.IsPointOnSurface(VxVector(2.0f, 0.0f, 0.0f)));

    // Point inside
    EXPECT_FALSE(unit_sphere.IsPointOnSurface(VxVector(0.5f, 0.0f, 0.0f)));
}

TEST_F(VxSphereTest, IsBoxTotallyInside) {
    // Small box at center
    VxBbox small_box(VxVector(-0.2f, -0.2f, -0.2f), VxVector(0.2f, 0.2f, 0.2f));
    EXPECT_TRUE(unit_sphere.IsBoxTotallyInside(small_box));

    // Box too large
    VxBbox large_box(VxVector(-2.0f, -2.0f, -2.0f), VxVector(2.0f, 2.0f, 2.0f));
    EXPECT_FALSE(unit_sphere.IsBoxTotallyInside(large_box));

    // Box touching surface
    VxBbox surface_box(VxVector(-1.0f, -0.1f, -0.1f), VxVector(1.0f, 0.1f, 0.1f));
    EXPECT_FALSE(unit_sphere.IsBoxTotallyInside(surface_box));

    // Box outside sphere
    VxBbox outside_box(VxVector(2.0f, 2.0f, 2.0f), VxVector(3.0f, 3.0f, 3.0f));
    EXPECT_FALSE(unit_sphere.IsBoxTotallyInside(outside_box));
}

TEST_F(VxSphereTest, EqualityOperator) {
    VxSphere sphere1(VxVector(1.0f, 2.0f, 3.0f), 4.0f);
    VxSphere sphere2(VxVector(1.0f, 2.0f, 3.0f), 4.0f);
    VxSphere sphere3(VxVector(1.0f, 2.0f, 3.0f), 5.0f);
    VxSphere sphere4(VxVector(2.0f, 2.0f, 3.0f), 4.0f);

    EXPECT_TRUE(sphere1 == sphere2);
    EXPECT_FALSE(sphere1 == sphere3); // Different radius
    EXPECT_FALSE(sphere1 == sphere4); // Different center
}

class GeometryIntersectionTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(GeometryIntersectionTest, RayPlaneIntersection) {
    // XY plane at Z=0
    VxPlane plane(0.0f, 0.0f, 1.0f, 0.0f);

    // Ray pointing downward from above
    VxRay ray(VxVector(1.0f, 1.0f, 5.0f), VxVector(0.0f, 0.0f, -1.0f), nullptr);

    // Manual intersection calculation
    // Ray: P = origin + t * direction
    // Plane: dot(normal, P) + D = 0
    // Solve: dot(normal, origin + t * direction) + D = 0
    float t = -(DotProduct(plane.GetNormal(), ray.GetOrigin()) + plane.m_D) /
              DotProduct(plane.GetNormal(), ray.GetDirection());

    VxVector intersection_point;
    ray.Interpolate(intersection_point, t);

    EXPECT_FLOAT_EQ(intersection_point.x, 1.0f);
    EXPECT_FLOAT_EQ(intersection_point.y, 1.0f);
    EXPECT_NEAR(intersection_point.z, 0.0f, EPSILON);

    // Verify point is on plane
    EXPECT_NEAR(plane.Distance(intersection_point), 0.0f, EPSILON);
}

TEST_F(GeometryIntersectionTest, RaySphereIntersection) {
    VxSphere sphere(VxVector(0.0f, 0.0f, 0.0f), 1.0f);

    // Ray from outside sphere toward center
    VxRay ray(VxVector(-3.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f), nullptr);

    // Manual intersection calculation using quadratic formula
    VxVector oc = ray.GetOrigin() - sphere.Center();
    float a = DotProduct(ray.GetDirection(), ray.GetDirection());
    float b = 2.0f * DotProduct(oc, ray.GetDirection());
    float c = DotProduct(oc, oc) - sphere.Radius() * sphere.Radius();

    float discriminant = b * b - 4 * a * c;
    EXPECT_GT(discriminant, 0.0f); // Should have intersection

    float t1 = (-b - sqrtf(discriminant)) / (2.0f * a);
    float t2 = (-b + sqrtf(discriminant)) / (2.0f * a);

    VxVector point1, point2;
    ray.Interpolate(point1, t1);
    ray.Interpolate(point2, t2);

    // Both points should be on the sphere surface
    EXPECT_TRUE(sphere.IsPointOnSurface(point1));
    EXPECT_TRUE(sphere.IsPointOnSurface(point2));

    // First intersection should be at (-1,0,0), second at (1,0,0)
    EXPECT_NEAR(point1.x, -1.0f, EPSILON);
    EXPECT_NEAR(point2.x, 1.0f, EPSILON);
}

TEST_F(GeometryIntersectionTest, PlaneBoxClassification) {
    VxPlane plane(1.0f, 0.0f, 0.0f, 0.0f); // YZ plane at X=0

    // Box completely on positive side
    VxBbox positive_box(VxVector(1.0f, -1.0f, -1.0f), VxVector(2.0f, 1.0f, 1.0f));
    EXPECT_GT(plane.Classify(positive_box), 0.0f);

    // Box completely on negative side
    VxBbox negative_box(VxVector(-2.0f, -1.0f, -1.0f), VxVector(-1.0f, 1.0f, 1.0f));
    EXPECT_LT(plane.Classify(negative_box), 0.0f);

    // Box intersecting plane
    VxBbox intersecting_box(VxVector(-1.0f, -1.0f, -1.0f), VxVector(1.0f, 1.0f, 1.0f));
    EXPECT_FLOAT_EQ(plane.Classify(intersecting_box), 0.0f);
}

TEST_F(GeometryIntersectionTest, SphereBoxIntersection) {
    VxSphere sphere(VxVector(0.0f, 0.0f, 0.0f), 2.0f);

    // Box completely inside sphere
    VxBbox inside_box(VxVector(-0.5f, -0.5f, -0.5f), VxVector(0.5f, 0.5f, 0.5f));
    EXPECT_TRUE(sphere.IsBoxTotallyInside(inside_box));

    // Box partially inside sphere (touching surface)
    VxBbox touching_box(VxVector(-2.0f, -0.1f, -0.1f), VxVector(2.0f, 0.1f, 0.1f));
    EXPECT_FALSE(sphere.IsBoxTotallyInside(touching_box));

    // Box completely outside sphere
    VxBbox outside_box(VxVector(5.0f, 5.0f, 5.0f), VxVector(6.0f, 6.0f, 6.0f));
    EXPECT_FALSE(sphere.IsBoxTotallyInside(outside_box));
}

TEST(VxOBBTests, CreationAndQueries) {
    VxBbox aabb(VxVector(-1, -2, -3), VxVector(1, 2, 3));
    VxMatrix mat;
    Vx3DMatrixFromEulerAngles(mat, 0, 0, PI / 4.0f); // 45 deg rot around Z
    mat[3][0] = 10.0f; // Translate

    VxOBB obb(aabb, mat);

    // Center should be translated
    EXPECT_NEAR(obb.GetCenter().x, 10.0f, EPSILON);
    EXPECT_NEAR(obb.GetCenter().y, 0.0f, EPSILON);

    // Axes should be rotated
    float cos45 = sqrt(2.0f) / 2.0f;
    EXPECT_NEAR(obb.GetAxis(0).x, cos45, EPSILON);
    EXPECT_NEAR(obb.GetAxis(0).y, cos45, EPSILON);
    EXPECT_NEAR(obb.GetAxis(1).x, -cos45, EPSILON);
    EXPECT_NEAR(obb.GetAxis(1).y, cos45, EPSILON);

    // Extents should match AABB half-size
    EXPECT_FLOAT_EQ(obb.GetExtent(0), 1.0f);
    EXPECT_FLOAT_EQ(obb.GetExtent(1), 2.0f);
    EXPECT_FLOAT_EQ(obb.GetExtent(2), 3.0f);

    EXPECT_TRUE(obb.VectorIn(obb.GetCenter()));
}

// ==========================================================
// VxDistance Tests
// ==========================================================

TEST(VxDistanceTests, PointSegment) {
    VxVector p0(0,0,0), p1(10,0,0);
    VxRay seg(p0, p1);

    // Closest point is on segment
    VxVector pt1(5, 5, 0);
    float t;
    float distSq = VxDistance::PointSegmentSquareDistance(pt1, seg, &t);
    EXPECT_FLOAT_EQ(distSq, 25.0f);
    EXPECT_NEAR(t, 0.5f, EPSILON);

    // Closest point is endpoint p0
    VxVector pt2(-5, 5, 0);
    distSq = VxDistance::PointSegmentSquareDistance(pt2, seg, &t);
    EXPECT_FLOAT_EQ(distSq, 50.0f); // dist to (-5,5,0) from (0,0,0)
    EXPECT_NEAR(t, 0.0f, EPSILON);

    // Closest point is endpoint p1
    VxVector pt3(15, 5, 0);
    distSq = VxDistance::PointSegmentSquareDistance(pt3, seg, &t);
    EXPECT_FLOAT_EQ(distSq, 50.0f); // dist to (15,5,0) from (10,0,0)
    EXPECT_NEAR(t, 1.0f, EPSILON);
}

TEST(VxDistanceTests, SegmentSegment) {
    // Intersecting segments
    VxRay seg1(VxVector(-1,0,0), VxVector(1,0,0));
    VxRay seg2(VxVector(0,-1,0), VxVector(0,1,0));
    float t0, t1;
    float distSq = VxDistance::SegmentSegmentSquareDistance(seg1, seg2, &t0, &t1);
    EXPECT_NEAR(distSq, 0.0f, EPSILON);
    EXPECT_NEAR(t0, 0.5f, EPSILON);
    EXPECT_NEAR(t1, 0.5f, EPSILON);

    // Parallel segments
    VxRay seg3(VxVector(0,1,0), VxVector(1,1,0)); // Shifted up by 1
    VxRay seg4(VxVector(0,0,0), VxVector(1,0,0));
    distSq = VxDistance::SegmentSegmentSquareDistance(seg3, seg4, &t0, &t1);
    EXPECT_NEAR(distSq, 1.0f, EPSILON);

    // Skew segments
    VxRay seg5(VxVector(0,0,0), VxVector(1,0,0)); // Along X axis
    VxRay seg6(VxVector(0,0,1), VxVector(0,1,1)); // Along Y axis, at z=1
    distSq = VxDistance::SegmentSegmentSquareDistance(seg5, seg6, &t0, &t1);
    EXPECT_NEAR(distSq, 1.0f, EPSILON); // Closest approach is at (0,0,0) and (0,0,1)
    EXPECT_NEAR(t0, 0.0f, EPSILON);
    EXPECT_NEAR(t1, 0.0f, EPSILON);
}
