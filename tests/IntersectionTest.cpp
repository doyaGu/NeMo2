#include <gtest/gtest.h>
#include <cmath>
#include "VxVector.h"
#include "VxRay.h"
#include "VxSphere.h"
#include "VxPlane.h"
#include "VxFrustum.h"
#include "VxIntersect.h"
#include "VxDistance.h"

class IntersectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        epsilon = EPSILON;
    }

    float epsilon;

    // Helper function to compare vectors with tolerance
    bool VectorsApproxEqual(const VxVector& v1, const VxVector& v2, float tol = EPSILON) {
        return std::abs(v1.x - v2.x) < tol &&
               std::abs(v1.y - v2.y) < tol &&
               std::abs(v1.z - v2.z) < tol;
    }
};

TEST_F(IntersectionTest, RaySphereIntersection) {
    VxRay ray;
    ray.m_Origin = VxVector(0.0f, 0.0f, -5.0f);
    ray.m_Direction = VxVector(0.0f, 0.0f, 1.0f);
    ray.m_Direction.Normalize();

    VxSphere sphere(VxVector(0.0f, 0.0f, 0.0f), 2.0f);

    // Test direct hit
    VxVector inter1, inter2;
    int hit_count = VxIntersect::RaySphere(ray, sphere, &inter1, &inter2);
    EXPECT_GT(hit_count, 0);
    EXPECT_TRUE(VectorsApproxEqual(inter1, VxVector(0.0f, 0.0f, -2.0f), epsilon));

    // Test miss
    VxRay miss_ray;
    miss_ray.m_Origin = VxVector(5.0f, 0.0f, -5.0f);
    miss_ray.m_Direction = VxVector(0.0f, 0.0f, 1.0f);
    miss_ray.m_Direction.Normalize();

    VxVector miss_inter1, miss_inter2;
    int miss_hit_count = VxIntersect::RaySphere(miss_ray, sphere, &miss_inter1, &miss_inter2);
    EXPECT_EQ(miss_hit_count, 0);

    // Test ray originating inside sphere
    // Note: Original binary returns intersection points in order of parametric t values
    // When ray starts inside sphere: t1 = projection - sqrt(discriminant) is negative
    // So inter1 is behind the ray origin, inter2 is in front
    VxRay inside_ray;
    inside_ray.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
    inside_ray.m_Direction = VxVector(1.0f, 0.0f, 0.0f);
    inside_ray.m_Direction.Normalize();

    VxVector inside_inter1, inside_inter2;
    int inside_hit_count = VxIntersect::RaySphere(inside_ray, sphere, &inside_inter1, &inside_inter2);
    EXPECT_EQ(inside_hit_count, 2);
    // inter1 is behind (negative t), inter2 is in front (positive t)
    EXPECT_TRUE(VectorsApproxEqual(inside_inter1, VxVector(-2.0f, 0.0f, 0.0f), epsilon * 4.0f));
    EXPECT_TRUE(VectorsApproxEqual(inside_inter2, VxVector(2.0f, 0.0f, 0.0f), epsilon * 4.0f));
}

TEST_F(IntersectionTest, RayPlaneIntersection) {
    VxRay ray;
    ray.m_Origin = VxVector(0.0f, 0.0f, -5.0f);
    ray.m_Direction = VxVector(0.0f, 0.0f, 1.0f);
    ray.m_Direction.Normalize();

    VxPlane plane(VxVector(0.0f, 0.0f, 1.0f), VxVector(0.0f, 0.0f, 0.0f));

    // Test intersection
    VxVector intersection_point;
    float distance;
    bool intersects = VxIntersect::RayPlane(ray, plane, intersection_point, distance);
    EXPECT_TRUE(intersects);
    EXPECT_TRUE(VectorsApproxEqual(intersection_point, VxVector(0.0f, 0.0f, 0.0f), epsilon));

    // Test parallel ray (no intersection)
    VxRay parallel_ray;
    parallel_ray.m_Origin = VxVector(0.0f, 0.0f, -5.0f);
    parallel_ray.m_Direction = VxVector(1.0f, 0.0f, 0.0f);
    parallel_ray.m_Direction.Normalize();

    VxVector parallel_point;
    float parallel_distance;
    bool parallel_intersects = VxIntersect::RayPlane(parallel_ray, plane, parallel_point, parallel_distance);
    EXPECT_FALSE(parallel_intersects);
}

TEST_F(IntersectionTest, RayBboxIntersection) {
    VxRay ray;
    ray.m_Origin = VxVector(5.0f, 5.0f, -5.0f);
    ray.m_Direction = VxVector(-1.0f, -1.0f, 1.0f);
    ray.m_Direction.Normalize();

    VxBbox bbox(VxVector(-1.0f, -1.0f, -1.0f), VxVector(1.0f, 1.0f, 1.0f));

    // Test intersection
    bool intersects = VxIntersect::RayBox(ray, bbox);
    EXPECT_TRUE(intersects);

    // Test miss
    VxRay miss_ray;
    miss_ray.m_Origin = VxVector(5.0f, 5.0f, -5.0f);
    miss_ray.m_Direction = VxVector(1.0f, 1.0f, 1.0f);
    miss_ray.m_Direction.Normalize();

    bool miss_intersects = VxIntersect::RayBox(miss_ray, bbox);
    EXPECT_FALSE(miss_intersects);

    // Test ray originating inside bbox
    VxRay inside_ray;
    inside_ray.m_Origin = VxVector(0.0f, 0.0f, 0.0f);
    inside_ray.m_Direction = VxVector(1.0f, 0.0f, 0.0f);
    inside_ray.m_Direction.Normalize();

    bool inside_intersects = VxIntersect::RayBox(inside_ray, bbox);
    EXPECT_TRUE(inside_intersects);
}

TEST_F(IntersectionTest, DistanceCalculations) {
    // Test point to line distance
    VxVector point(0.0f, 5.0f, 0.0f);
    VxRay line(VxVector(0.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(VxDistance::PointLineDistance(point, line), 5.0f, epsilon);

    // Test line to line distance (skew lines)
    // Line1: from (0,0,0) to (1,0,0) - horizontal along X-axis
    // Line2: from (0,1,0) to (1,1,0) - horizontal along X-axis at y=1
    VxRay line1(VxVector(0.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f));
    VxRay line2(VxVector(0.0f, 1.0f, 0.0f), VxVector(1.0f, 1.0f, 0.0f));
    EXPECT_NEAR(VxDistance::LineLineDistance(line1, line2), 1.0f, epsilon);

    // Test parallel lines
    // parallel1: from (0,0,0) to (1,0,0) - direction (1,0,0)
    // parallel2: from (0,1,0) to (1,1,0) - direction (1,0,0)
    VxRay parallel1(VxVector(0.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f));
    VxRay parallel2(VxVector(0.0f, 1.0f, 0.0f), VxVector(1.0f, 1.0f, 0.0f));
    EXPECT_NEAR(VxDistance::LineLineDistance(parallel1, parallel2), 1.0f, epsilon);

    // Test intersecting lines
    // intersect1: from (0,0,0) to (1,0,0) - along X-axis
    // intersect2: from (0,0,0) to (0,1,0) - along Y-axis (they intersect at origin)
    VxRay intersect1(VxVector(0.0f, 0.0f, 0.0f), VxVector(1.0f, 0.0f, 0.0f));
    VxRay intersect2(VxVector(0.0f, 0.0f, 0.0f), VxVector(0.0f, 1.0f, 0.0f));
    EXPECT_NEAR(VxDistance::LineLineDistance(intersect1, intersect2), 0.0f, epsilon);
}

TEST_F(IntersectionTest, BboxOperations) {
    // Test bbox creation and manipulation
    VxBbox bbox1(VxVector(-1.0f, -2.0f, -3.0f), VxVector(1.0f, 2.0f, 3.0f));

    // Test center calculation
    VxVector center = bbox1.GetCenter();
    EXPECT_TRUE(VectorsApproxEqual(center, VxVector(0.0f, 0.0f, 0.0f), epsilon));

    // Test size calculation
    VxVector size = bbox1.GetSize();
    EXPECT_TRUE(VectorsApproxEqual(size, VxVector(2.0f, 4.0f, 6.0f), epsilon));

    // Test merging
    VxBbox bbox2(VxVector(2.0f, 3.0f, 4.0f), VxVector(3.0f, 4.0f, 5.0f));
    VxBbox merged = bbox1;
    merged.Merge(bbox2);

    VxBbox expected_merged(VxVector(-1.0f, -2.0f, -3.0f), VxVector(3.0f, 4.0f, 5.0f));
    EXPECT_TRUE(VectorsApproxEqual(merged.Min, expected_merged.Min, epsilon));
    EXPECT_TRUE(VectorsApproxEqual(merged.Max, expected_merged.Max, epsilon));

    // Test point inclusion
    EXPECT_TRUE(merged.VectorIn(VxVector(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(merged.VectorIn(VxVector(2.5f, 3.5f, 4.5f)));
    EXPECT_FALSE(merged.VectorIn(VxVector(5.0f, 5.0f, 5.0f)));
    EXPECT_FALSE(merged.VectorIn(VxVector(-2.0f, -3.0f, -4.0f)));
}