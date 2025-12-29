#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <random>

#include "VxFrustum.h"
#include "VxIntersect.h"
#include "VxMathTestHelpers.h"

using namespace VxMathTest;

// Test fixture for VxFrustum
class VxFrustumTest : public ::testing::Test {
protected:
    // Standard perspective frustum for testing
    VxFrustum perspectiveFrustum;
    // Orthographic frustum for testing
    VxFrustum orthoFrustum;

    // Parameters for the perspective frustum
    VxVector perspectiveOrigin = VxVector(0, 0, 0);
    VxVector perspectiveDir = VxVector(0, 0, 1);
    VxVector perspectiveUp = VxVector(0, 1, 0);
    VxVector perspectiveRight = VxVector(1, 0, 0);
    float fov = PI / 2.0f; // 90 degrees FOV
    float aspect = 1.0f;
    float nearPlaneDist = 1.0f;
    float farPlaneDist = 100.0f;

    // Parameters for the orthographic frustum
    VxVector orthoOrigin = VxVector(0, 0, -50); // Moved back to see the origin
    VxVector orthoDir = VxVector(0, 0, 1);
    VxVector orthoUp = VxVector(0, 1, 0);
    VxVector orthoRight = VxVector(1, 0, 0);
    // Orthographic uses bounds instead of FOV, which we can simulate via the constructor
    // by providing pre-calculated bounds. Let's make it 10 units wide and high.
    float orthoRBound = 5.0f;
    float orthoUBound = 5.0f;
    float orthoNear = 1.0f;
    float orthoFar = 100.0f;


    void SetUp() override {
        // Initialize the perspective frustum before each test
        perspectiveFrustum = VxFrustum(
            perspectiveOrigin,
            perspectiveRight,
            perspectiveUp,
            perspectiveDir,
            nearPlaneDist,
            farPlaneDist,
            fov,
            aspect);

        // For the ortho frustum, we create it manually as the constructor is perspective-focused.
        // We will set its properties directly.
        orthoFrustum.GetOrigin() = orthoOrigin;
        orthoFrustum.GetRight() = orthoRight;
        orthoFrustum.GetUp() = orthoUp;
        orthoFrustum.GetDir() = orthoDir;
        orthoFrustum.GetRBound() = orthoRBound;
        orthoFrustum.GetUBound() = orthoUBound;
        orthoFrustum.GetDMin() = orthoNear;
        orthoFrustum.GetDMax() = orthoFar;
        orthoFrustum.Update(); // This calculates the planes
    }
};

// Test the constructor and initial state of the frustum.
TEST_F(VxFrustumTest, ConstructorAndInitialization) {
    // Check that the initial properties match the input parameters.
    EXPECT_VEC3_NEAR(perspectiveFrustum.GetOrigin(), perspectiveOrigin, SIMD_TOL);
    EXPECT_VEC3_NEAR(perspectiveFrustum.GetDir(), perspectiveDir, SIMD_TOL);
    EXPECT_VEC3_NEAR(perspectiveFrustum.GetUp(), perspectiveUp, SIMD_TOL);
    EXPECT_VEC3_NEAR(perspectiveFrustum.GetRight(), perspectiveRight, SIMD_TOL);

    EXPECT_NEAR(perspectiveFrustum.GetDMin(), nearPlaneDist, SIMD_TOL);
    EXPECT_NEAR(perspectiveFrustum.GetDMax(), farPlaneDist, SIMD_TOL);

    // With FOV = 90 degrees and aspect = 1, the half-width at the near plane should be equal to the near plane distance.
    EXPECT_NEAR(perspectiveFrustum.GetRBound(), nearPlaneDist, SIMD_TOL);
    EXPECT_NEAR(perspectiveFrustum.GetUBound(), nearPlaneDist, SIMD_TOL);

    // Binary convention for frustum planes:
    // Near plane: normal points TOWARD camera (-Dir), so points in front of near are "outside" (positive classify)
    // Far plane: normal points AWAY from camera (Dir), so points behind far are "outside" (positive classify)
    VxPlane expectedNearPlane(-perspectiveDir, perspectiveOrigin + perspectiveDir * nearPlaneDist);
    EXPECT_PLANE_NEAR(perspectiveFrustum.GetNearPlane(), expectedNearPlane, SIMD_TOL);

    VxPlane expectedFarPlane(perspectiveDir, perspectiveOrigin + perspectiveDir * farPlaneDist);
    EXPECT_PLANE_NEAR(perspectiveFrustum.GetFarPlane(), expectedFarPlane, SIMD_TOL);

    // Side planes have normals that point toward the inside of the frustum
    // We just verify the frustum is constructed and planes are valid (not checking exact normals)
    // The actual test is whether Classify works correctly
    EXPECT_NE(perspectiveFrustum.GetLeftPlane().GetNormal().SquareMagnitude(), 0.0f);
    EXPECT_NE(perspectiveFrustum.GetRightPlane().GetNormal().SquareMagnitude(), 0.0f);
}

// Test the Update method to ensure it recalculates planes correctly.
TEST_F(VxFrustumTest, Update) {
    VxFrustum frustum = perspectiveFrustum;

    // Change a parameter, e.g., the direction
    frustum.GetDir() = VxVector(0, 0, -1);
    frustum.GetRight() = VxVector(-1, 0, 0); // Must update right vector to stay orthogonal

    // The planes should still be the old ones until we call Update()
    VxPlane oldNearPlane = frustum.GetNearPlane();

    // Now update
    frustum.Update();
    VxPlane newNearPlane = frustum.GetNearPlane();

    // The new near plane should be different from the old one
    EXPECT_FALSE(oldNearPlane == newNearPlane);

    // Binary convention: near plane normal = -Dir
    VxPlane expectedNewNearPlane(-frustum.GetDir(), frustum.GetOrigin() + frustum.GetDir() * frustum.GetDMin());
    EXPECT_PLANE_NEAR(newNearPlane, expectedNewNearPlane, SIMD_TOL);
}


// Test point classification against the perspective frustum.
TEST_F(VxFrustumTest, ClassifyPoint) {
    // A point deep inside the frustum
    VxVector pointInside(0, 0, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointInside), 0) << "Point deep inside should have no clip flags.";
    EXPECT_TRUE(perspectiveFrustum.IsInside(pointInside));

    // A point just inside the near plane
    VxVector pointJustInside(0, 0, nearPlaneDist + 0.1f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointJustInside), 0) << "Point just inside near plane should have no clip flags.";
    EXPECT_TRUE(perspectiveFrustum.IsInside(pointJustInside));

    // A point just in front of the near plane
    VxVector pointInFront(0, 0, nearPlaneDist - 0.1f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointInFront), VXCLIP_FRONT) << "Point in front of near plane should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointInFront));

    // A point just behind the far plane
    VxVector pointBehind(0, 0, farPlaneDist + 0.1f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointBehind), VXCLIP_BACK) << "Point behind far plane should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointBehind));

    // A point to the left of the frustum
    // At z=50 with 90° FOV, frustum extends to x=±50, so x=-60 is outside
    VxVector pointLeft(-60, 0, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointLeft), VXCLIP_LEFT) << "Point left of frustum should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointLeft));

    // A point to the right of the frustum
    VxVector pointRight(60, 0, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointRight), VXCLIP_RIGHT) << "Point right of frustum should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointRight));

    // A point above the frustum
    VxVector pointTop(0, 60, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointTop), VXCLIP_TOP) << "Point above frustum should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointTop));

    // A point below the frustum
    VxVector pointBottom(0, -60, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointBottom), VXCLIP_BOTTOM) << "Point below frustum should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointBottom));

    // A point outside multiple planes (front and left)
    VxVector pointFrontLeft(-5, 0, nearPlaneDist - 0.1f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointFrontLeft), VXCLIP_FRONT | VXCLIP_LEFT) << "Point front-left should have two clip flags.";
}

// Test AABB (Axis-Aligned Bounding Box) classification.
TEST_F(VxFrustumTest, ClassifyAABB) {
    // Box completely inside the frustum
    VxBbox boxInside(VxVector(-1, -1, 10), VxVector(1, 1, 20));
    EXPECT_LT(perspectiveFrustum.Classify(boxInside), 0.0f) << "Box fully inside should return a negative value (no single plane separates).";

    // Box intersecting the near plane
    VxBbox boxIntersectNear(VxVector(-1, -1, 0.5f), VxVector(1, 1, 1.5f));
    EXPECT_FLOAT_EQ(perspectiveFrustum.Classify(boxIntersectNear), 0.0f) << "Box intersecting should return 0.";

    // Box completely in front of the frustum
    VxBbox boxInFront(VxVector(-1, -1, 0.1f), VxVector(1, 1, 0.5f));
    EXPECT_GT(perspectiveFrustum.Classify(boxInFront), 0.0f) << "Box fully in front should return positive value.";

    // Box completely behind the frustum
    VxBbox boxBehind(VxVector(-1, -1, 101), VxVector(1, 1, 102));
    EXPECT_GT(perspectiveFrustum.Classify(boxBehind), 0.0f) << "Box fully behind should return positive value.";

    // Box completely to the left of the frustum
    // At z=10-20, frustum extends to x=-10 to x=-20, so x=-30 to x=-25 is outside
    VxBbox boxLeft(VxVector(-35, -1, 10), VxVector(-25, 1, 20));
    EXPECT_GT(perspectiveFrustum.Classify(boxLeft), 0.0f) << "Box fully to the left should return positive value.";

    // Huge box that engulfs the entire frustum - touches all planes, so cumul product is 0
    // The Classify function returns -cumul, where cumul is product of plane classifications
    // When any plane returns 0 (box touches plane), cumul becomes 0, so result is -0 == 0
    VxBbox boxEngulfing(VxVector(-200, -200, -10), VxVector(200, 200, 200));
    EXPECT_LE(perspectiveFrustum.Classify(boxEngulfing), 0.0f) << "Engulfing box should be classified as inside/touching.";
}


// Test OBB (Oriented Bounding Box) classification.
// Note: FrustumOBB uses a complex SAT-based algorithm that differs from plane-based tests.
// FrustumBox uses plane-based intersection tests.
TEST_F(VxFrustumTest, ClassifyOBB) {
    // Use a box that's clearly inside (not touching any plane)
    // Near plane is at z=10, so start at z=15 to be clear of it
    VxBbox baseBox(VxVector(-1, -1, 15), VxVector(1, 1, 25));

    // Case 1: No transformation (equivalent to AABB test)
    VxMatrix identity;
    Vx3DMatrixIdentity(identity);
    // FrustumBox with identity should work correctly after our fix
    EXPECT_TRUE(VxIntersect::FrustumBox(perspectiveFrustum, baseBox, identity)) << "FrustumBox should return TRUE for inside box.";

    // Case 2: Translated to be outside
    VxMatrix translation;
    Vx3DMatrixIdentity(translation);
    translation[3][0] = 100.0f; // Move it far to the right (outside frustum)
    EXPECT_FALSE(VxIntersect::FrustumBox(perspectiveFrustum, baseBox, translation)) << "FrustumBox should return FALSE for outside box.";

    // Case 3: Box at the edge (intersecting)
    // At z=15-25, frustum extends to approximately x=±15 to x=±25
    VxBbox edgeBox(VxVector(10, -1, 15), VxVector(20, 1, 25)); // Straddles the right edge
    EXPECT_TRUE(VxIntersect::FrustumBox(perspectiveFrustum, edgeBox, identity)) << "FrustumBox should return TRUE for intersecting box.";
}

// ============================================================================
// VxIntersect Frustum APIs
// ============================================================================

TEST_F(VxFrustumTest, VxIntersect_FrustumFace) {
    // Triangle well inside the frustum
    VxVector t0(-1, -1, 10);
    VxVector t1(1, -1, 10);
    VxVector t2(0, 1, 10);
    EXPECT_TRUE(VxIntersect::FrustumFace(perspectiveFrustum, t0, t1, t2));

    // Triangle completely outside (to the right)
    VxVector o0(200, -1, 10);
    VxVector o1(202, -1, 10);
    VxVector o2(200, 1, 10);
    EXPECT_FALSE(VxIntersect::FrustumFace(perspectiveFrustum, o0, o1, o2));

    // Triangle intersecting the frustum edge
    VxVector e0(90, -1, 90);
    VxVector e1(110, -1, 90);
    VxVector e2(100, 1, 90);
    EXPECT_TRUE(VxIntersect::FrustumFace(perspectiveFrustum, e0, e1, e2));
}

TEST_F(VxFrustumTest, VxIntersect_FrustumAABB) {
    // Box inside
    VxBbox inside(VxVector(-1, -1, 10), VxVector(1, 1, 20));
    EXPECT_TRUE(VxIntersect::FrustumAABB(perspectiveFrustum, inside));

    // Box outside to the right
    VxBbox outsideRight(VxVector(200, -1, 10), VxVector(210, 1, 20));
    EXPECT_FALSE(VxIntersect::FrustumAABB(perspectiveFrustum, outsideRight));

    // Box intersecting near plane
    VxBbox intersectNear(VxVector(-1, -1, 0.5f), VxVector(1, 1, 1.5f));
    EXPECT_TRUE(VxIntersect::FrustumAABB(perspectiveFrustum, intersectNear));
}

TEST_F(VxFrustumTest, VxIntersect_FrustumOBB) {
    // Local-space box around origin; use transform to place it in front of camera.
    VxBbox localBox(VxVector(-1, -1, -1), VxVector(1, 1, 1));

    VxMatrix matInside;
    Vx3DMatrixIdentity(matInside);
    matInside[3][2] = 10.0f;
    EXPECT_TRUE(VxIntersect::FrustumOBB(perspectiveFrustum, localBox, matInside));

    VxMatrix matOutside;
    Vx3DMatrixIdentity(matOutside);
    matOutside[3][0] = 200.0f;
    matOutside[3][2] = 10.0f;
    EXPECT_FALSE(VxIntersect::FrustumOBB(perspectiveFrustum, localBox, matOutside));

    // Rotated box inside
    VxMatrix matRot;
    Vx3DMatrixFromEulerAngles(matRot, 0.0f, 0.0f, PI / 4.0f);
    matRot[3][2] = 10.0f;
    EXPECT_TRUE(VxIntersect::FrustumOBB(perspectiveFrustum, localBox, matRot));
}

TEST_F(VxFrustumTest, VxIntersect_FrustumBox_MatchesPlaneClassify) {
    // FrustumBox is the plane-based frustum-vs-OBB test. It should agree with
    // VxFrustum::Classify(box, mat) semantics (<= 0 means inside/touching).
    VxBbox localBox(VxVector(-1, -2, -3), VxVector(1, 2, 3));

    auto expectMatches = [&](const VxMatrix &mat) {
        const bool expected = (perspectiveFrustum.Classify(localBox, mat) <= 0.0f);
        EXPECT_EQ(VxIntersect::FrustumBox(perspectiveFrustum, localBox, mat), expected);
    };

    // Identity at origin: box straddles the near plane (near at z=1), so it intersects.
    VxMatrix id;
    Vx3DMatrixIdentity(id);
    expectMatches(id);

    // Translate forward so the box is clearly inside.
    VxMatrix inside;
    Vx3DMatrixIdentity(inside);
    inside[3][2] = 10.0f;
    expectMatches(inside);

    // Translate far right so it is clearly outside.
    VxMatrix outside;
    Vx3DMatrixIdentity(outside);
    outside[3][0] = 200.0f;
    outside[3][2] = 10.0f;
    expectMatches(outside);

    // Rotate + translate (still inside).
    VxMatrix rot;
    Vx3DMatrixFromEulerAngles(rot, 0.0f, PI / 6.0f, PI / 8.0f);
    rot[3][2] = 10.0f;
    expectMatches(rot);

    // Non-uniform scale + translate.
    VxMatrix scaled;
    Vx3DMatrixIdentity(scaled);
    scaled[0] *= 2.0f;
    scaled[1] *= 0.5f;
    scaled[2] *= 3.0f;
    scaled[3][2] = 10.0f;
    expectMatches(scaled);
}

// Test orthographic frustum classification specifically.
// Note: The VxFrustum class is designed for perspective frustums.
// When used with orthographic-like parameters, the side planes are still perspective-style
// (passing through the frustum origin), so they may not behave like true orthographic planes.
TEST_F(VxFrustumTest, ClassifyPointOrtho) {
    // Point inside the ortho frustum
    // The frustum origin is at (0,0,-50), looking along +Z
    // Near plane is at z = -50 + 1 = -49
    // Far plane is at z = -50 + 100 = 50
    VxVector pointInside(0, 0, 0); // This is between -49 and 50 in z
    EXPECT_EQ(orthoFrustum.Classify(pointInside), 0);
    EXPECT_TRUE(orthoFrustum.IsInside(pointInside));

    // Point in front of the near plane
    // Near plane is at z = -49, so z = -50 is in front
    VxVector pointInFront(0, 0, -50);
    EXPECT_EQ(orthoFrustum.Classify(pointInFront), VXCLIP_FRONT);
    EXPECT_FALSE(orthoFrustum.IsInside(pointInFront));
    
    // Point behind the far plane
    // Far plane is at z = 50, so z = 51 is behind
    VxVector pointBehind(0, 0, 51);
    EXPECT_EQ(orthoFrustum.Classify(pointBehind), VXCLIP_BACK);
    EXPECT_FALSE(orthoFrustum.IsInside(pointBehind));
}


// Test the ComputeVertices method.
TEST_F(VxFrustumTest, ComputeVertices) {
    VxVector vertices[8];
    perspectiveFrustum.ComputeVertices(vertices);

    // The 8 vertices are (binary convention):
    // 0: Near-Bottom-Left
    // 1: Near-Top-Left
    // 2: Near-Top-Right
    // 3: Near-Bottom-Right
    // 4: Far-Bottom-Left
    // 5: Far-Top-Left
    // 6: Far-Top-Right
    // 7: Far-Bottom-Right

    // For our 90-degree FOV frustum at origin, near plane at 1.0:
    // The half-width and half-height at near plane is 1.0.
    EXPECT_VEC3_NEAR(vertices[0], VxVector(-1, -1, 1), SIMD_TOL);  // NBL
    EXPECT_VEC3_NEAR(vertices[1], VxVector(-1, 1, 1), SIMD_TOL);   // NTL
    EXPECT_VEC3_NEAR(vertices[2], VxVector(1, 1, 1), SIMD_TOL);    // NTR
    EXPECT_VEC3_NEAR(vertices[3], VxVector(1, -1, 1), SIMD_TOL);   // NBR

    // At the far plane (100.0), half-width and half-height is 100.0
    EXPECT_VEC3_NEAR(vertices[4], VxVector(-100, -100, 100), SIMD_TOL);  // FBL
    EXPECT_VEC3_NEAR(vertices[5], VxVector(-100, 100, 100), SIMD_TOL);   // FTL
    EXPECT_VEC3_NEAR(vertices[6], VxVector(100, 100, 100), SIMD_TOL);    // FTR
    EXPECT_VEC3_NEAR(vertices[7], VxVector(100, -100, 100), SIMD_TOL);   // FBR
}

// Test the Transform method.
TEST_F(VxFrustumTest, Transform) {
    VxFrustum frustum = perspectiveFrustum;

    // Create a transformation matrix (e.g., move and rotate the camera)
    VxMatrix worldMatrix;
    Vx3DMatrixFromEulerAngles(worldMatrix, 0, PI/2.0f, 0); // 90-degree yaw
    worldMatrix[3][0] = 10.0f; // Move 10 units right

    // The transform function takes the *inverse* of the world matrix
    // This transforms the frustum from world space into the local space of the world matrix
    VxMatrix invWorldMatrix;
    Vx3DInverseMatrix(invWorldMatrix, worldMatrix);

    frustum.Transform(invWorldMatrix);

    // After transforming by the inverse world matrix:
    // - The origin (0,0,0) is transformed by the inverse matrix
    // - The direction vectors are rotated by the inverse rotation
    
    // For a 90-degree Y-axis rotation with translation (10,0,0):
    // The inverse rotation maps (0,0,1) -> (1,0,0) -> (0,0,1) and (1,0,0) -> (0,0,-1)
    // Wait, let me recalculate properly:
    // 90-degree Y rotation: (0,0,1) -> (1,0,0), (1,0,0) -> (0,0,-1)
    // Inverse 90-degree Y rotation: (0,0,1) -> (-1,0,0), (1,0,0) -> (0,0,1)
    
    // So direction (0,0,1) after inverse rotation -> (-1,0,0)
    // Right (1,0,0) after inverse rotation -> (0,0,1)
    // Up (0,1,0) stays (0,1,0)
    
    // For the origin, the inverse matrix includes both rotation and translation:
    // The inverse of [R|T] is [R^T | -R^T*T]
    // Original origin (0,0,0) transformed: 0 * R^T + (-R^T*T) = -R^T*T
    // R^T * (10,0,0) = (0,0,10) for 90-degree inverse Y rotation
    // So new origin = (0, 0, -10)
    
    EXPECT_VEC3_NEAR(frustum.GetOrigin(), VxVector(0, 0, -10), SIMD_TOL);
    EXPECT_VEC3_NEAR(frustum.GetDir(), VxVector(-1, 0, 0), SIMD_TOL);
    EXPECT_VEC3_NEAR(frustum.GetUp(), VxVector(0, 1, 0), SIMD_TOL);
    EXPECT_VEC3_NEAR(frustum.GetRight(), VxVector(0, 0, 1), SIMD_TOL);

    // Binary convention: near plane normal = -Dir = -(-1,0,0) = (1,0,0)
    // Near plane point = origin + Dir * DMin = (0,0,-10) + (-1,0,0)*1 = (-1,0,-10)
    VxPlane expectedNearPlane(VxVector(1,0,0), VxVector(-1,0,-10));
    EXPECT_PLANE_NEAR(frustum.GetNearPlane(), expectedNearPlane, SIMD_TOL);
}

// Test the equality operator
TEST_F(VxFrustumTest, EqualityOperator) {
    VxFrustum frustum1 = perspectiveFrustum;
    VxFrustum frustum2 = perspectiveFrustum;

    EXPECT_TRUE(frustum1 == frustum2);
    
    // Change a property and check for inequality
    frustum2.GetDMax() = 200.0f;
    EXPECT_FALSE(frustum1 == frustum2);
}