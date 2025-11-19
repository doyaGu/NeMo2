#include "VxIntersect.h"

#include "VxVector.h"
#include "VxMatrix.h"
#include "VxPlane.h"

//---------- Faces

// Is a point inside the boundary of a face
XBOOL VxIntersect::PointInFace(const VxVector &point, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                               const VxVector &norm, int &i1, int &i2) {
    // Find dominant axis of normal to determine projection plane
    float nx = XAbs(norm.x);
    float ny = XAbs(norm.y);
    float nz = XAbs(norm.z);

    // Select the most significant plane to project onto
    i1 = 1; // Y
    i2 = 2; // Z

    if (nx < ny) {
        i1 = 0; // X
        i2 = 2; // Z
        nx = ny;
    }

    if (nx < nz) {
        i1 = 0; // X
        i2 = 1; // Y
    }

    // Get 2D coordinates
    float p0_i1 = pt0[i1], p0_i2 = pt0[i2];
    float p1_i1 = pt1[i1], p1_i2 = pt1[i2];
    float p2_i1 = pt2[i1], p2_i2 = pt2[i2];
    float pt_i1 = point[i1], pt_i2 = point[i2];

    // Perform three edge tests using cross products
    char flags = 1;

    // Test edge pt0->pt1 vs point
    if ((pt_i1 - p1_i1) * (p2_i2 - p1_i2) - (p2_i1 - p1_i1) * (pt_i2 - p1_i2) >= 0.0f) {
        flags = 2;
    }

    // Test edge pt1->pt2 vs point
    if ((pt_i1 - p2_i1) * (p0_i2 - p2_i2) - (pt_i2 - p2_i2) * (p0_i1 - p2_i1) >= 0.0f) {
        flags &= 2;
    } else {
        flags &= 1;
    }

    if (!flags) return FALSE;

    // Test edge pt2->pt0 vs point
    if ((pt_i1 - p0_i1) * (p1_i2 - p0_i2) - (pt_i2 - p0_i2) * (p1_i1 - p0_i1) >= 0.0f) {
        return (flags & 2) != 0;
    }

    return (flags & 1) != 0;
}

// Intersection Ray - Face
XBOOL VxIntersect::RayFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                           const VxVector &norm, VxVector &res, float &dist) {
    // Create plane from normal and point
    VxPlane plane;
    plane.Create(norm, pt0);

    // Test ray-plane intersection
    if (!RayPlane(ray, plane, res, dist)) {
        return FALSE;
    }

    // Check if intersection point is inside the triangle face
    int i1, i2;
    return PointInFace(res, pt0, pt1, pt2, norm, i1, i2);
}

// Overloaded RayFace with dominant axes
XBOOL VxIntersect::RayFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                           const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2) {
    // Create plane from normal and point
    VxPlane plane;
    plane.Create(norm, pt0);

    // Test ray-plane intersection
    if (!RayPlane(ray, plane, res, dist)) {
        return FALSE;
    }

    // Check if intersection point is inside the triangle face
    return PointInFace(res, pt0, pt1, pt2, norm, i1, i2);
}

// Intersection Ray - Face with culling (only from front)
XBOOL VxIntersect::RayFaceCulled(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                                 const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2) {
    // Create plane from normal and point
    VxPlane plane;
    plane.Create(norm, pt0);

    // Test ray-plane intersection with culling
    if (!RayPlaneCulled(ray, plane, res, dist)) {
        return FALSE;
    }

    // Check if intersection point is inside the triangle face
    return PointInFace(res, pt0, pt1, pt2, norm, i1, i2);
}

// Intersection Segment - Face
XBOOL VxIntersect::SegmentFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                               const VxVector &norm, VxVector &res, float &dist) {
    // Create plane from normal and point
    VxPlane plane;
    plane.Create(norm, pt0);

    // Test segment-plane intersection
    if (!SegmentPlane(ray, plane, res, dist)) {
        return FALSE;
    }

    // Check if intersection point is inside the triangle face
    int i1, i2;
    return PointInFace(res, pt0, pt1, pt2, norm, i1, i2);
}

// Overloaded SegmentFace with dominant axes
XBOOL VxIntersect::SegmentFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                               const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2) {
    // Create plane from normal and point
    VxPlane plane;
    plane.Create(norm, pt0);

    // Test segment-plane intersection
    if (!SegmentPlane(ray, plane, res, dist)) {
        return FALSE;
    }

    // Check if intersection point is inside the triangle face
    return PointInFace(res, pt0, pt1, pt2, norm, i1, i2);
}

// Intersection Segment - Face with culling
XBOOL VxIntersect::SegmentFaceCulled(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                                     const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2) {
    // Create plane from normal and point
    VxPlane plane;
    plane.Create(norm, pt0);

    // Test segment-plane intersection with culling
    if (!SegmentPlaneCulled(ray, plane, res, dist)) {
        return FALSE;
    }

    // Check if intersection point is inside the triangle face
    return PointInFace(res, pt0, pt1, pt2, norm, i1, i2);
}

// Intersection Line - Face
XBOOL VxIntersect::LineFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                            const VxVector &norm, VxVector &res, float &dist) {
    // Create plane from normal and point
    VxPlane plane;
    plane.Create(norm, pt0);

    // Test line-plane intersection
    if (!LinePlane(ray, plane, res, dist)) {
        return FALSE;
    }

    // Check if intersection point is inside the triangle face
    int i1, i2;
    return PointInFace(res, pt0, pt1, pt2, norm, i1, i2);
}

// Overloaded LineFace with dominant axes
XBOOL VxIntersect::LineFace(const VxRay &ray, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                            const VxVector &norm, VxVector &res, float &dist, int &i1, int &i2) {
    // Create plane from normal and point
    VxPlane plane;
    plane.Create(norm, pt0);

    // Test line-plane intersection
    if (!LinePlane(ray, plane, res, dist)) {
        return FALSE;
    }

    // Check if intersection point is inside the triangle face
    return PointInFace(res, pt0, pt1, pt2, norm, i1, i2);
}

// Calculate barycentric coordinates for point in face
void VxIntersect::GetPointCoefficients(const VxVector &pt, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                                       const int &i1, const int &i2, float &V0Coef, float &V1Coef, float &V2Coef) {
    // Get 2D coordinates for the calculation
    float p0_i1 = pt0[i1], p0_i2 = pt0[i2];
    float p1_i1 = pt1[i1], p1_i2 = pt1[i2];
    float p2_i1 = pt2[i1], p2_i2 = pt2[i2];
    float pt_i1 = pt[i1], pt_i2 = pt[i2];

    // Compute vectors from pt0
    float v1_i1 = pt_i1 - p0_i1;
    float v1_i2 = pt_i2 - p0_i2;
    float v2_i1 = p1_i1 - p0_i1;
    float v2_i2 = p1_i2 - p0_i2;
    float v3_i1 = p2_i1 - p0_i1;
    float v3_i2 = p2_i2 - p0_i2;

    // Solve 2D linear system: v1 = V1Coef * v2 + V2Coef * v3
    if (v2_i1 == 0.0f) {
        V2Coef = v1_i1 / v3_i1;
        V1Coef = (v1_i2 - V2Coef * v3_i2) / v2_i2;
    } else {
        float denom = v3_i2 * v2_i1 - v3_i1 * v2_i2;
        V2Coef = (v2_i1 * v1_i2 - v2_i2 * v1_i1) / denom;
        V1Coef = (v1_i1 - V2Coef * v3_i1) / v2_i1;
    }

    V0Coef = 1.0f - (V1Coef + V2Coef);
}

int coplanar_tri_tri(const VxVector &N, const VxVector &V0, const VxVector &V1, const VxVector &V2,
                     const VxVector &U0, const VxVector &U1, const VxVector &U2);

// Intersection Face - Face
XBOOL VxIntersect::FaceFace(const VxVector &A0, const VxVector &A1, const VxVector &A2, const VxVector &N0,
                            const VxVector &B0, const VxVector &B1, const VxVector &B2, const VxVector &N1) {
    // Compute signed distances of triangle B vertices to plane of triangle A
    float d_A = -(DotProduct(N0, A0));
    float dB0 = DotProduct(N0, B0) + d_A;
    float dB1 = DotProduct(N0, B1) + d_A;
    float dB2 = DotProduct(N0, B2) + d_A;

    // Apply epsilon tolerance
    if (XAbs(dB0) < EPSILON) dB0 = 0.0f;
    if (XAbs(dB1) < EPSILON) dB1 = 0.0f;
    if (XAbs(dB2) < EPSILON) dB2 = 0.0f;

    // Check if all vertices of triangle B are on same side of plane A
    float dB0_dB1 = dB0 * dB1;
    float dB0_dB2 = dB0 * dB2;
    if (dB0_dB1 > 0.0f && dB0_dB2 > 0.0f) {
        return FALSE; // Triangle B is completely on one side of plane A
    }

    // Compute signed distances of triangle A vertices to plane of triangle B
    float d_B = -(DotProduct(N1, B0));
    float dA0 = DotProduct(N1, A0) + d_B;
    float dA1 = DotProduct(N1, A1) + d_B;
    float dA2 = DotProduct(N1, A2) + d_B;

    // Apply epsilon tolerance
    if (XAbs(dA0) < EPSILON) dA0 = 0.0f;
    if (XAbs(dA1) < EPSILON) dA1 = 0.0f;
    if (XAbs(dA2) < EPSILON) dA2 = 0.0f;

    // Check if all vertices of triangle A are on same side of plane B
    float dA0_dA1 = dA0 * dA1;
    float dA0_dA2 = dA0 * dA2;
    if (dA0_dA1 > 0.0f && dA0_dA2 > 0.0f) {
        return FALSE; // Triangle A is completely on one side of plane B
    }

    // Compute direction of intersection line
    VxVector D = CrossProduct(N0, N1);

    // Find the largest component of D to choose projection axis
    float ax = XAbs(D.x);
    float ay = XAbs(D.y);
    float az = XAbs(D.z);

    int maxc = 0;
    if (ay > ax) {
        maxc = 1;
        ax = ay;
    }
    if (az > ax) {
        maxc = 2;
    }

    // Project triangles onto coordinate plane where intersection line has largest component
    float vp0 = 0.0f, vp1 = 0.0f, vp2 = 0.0f, up0 = 0.0f, up1 = 0.0f, up2 = 0.0f;
    switch (maxc) {
        case 0: // Project onto YZ plane
            vp0 = A0.y; vp1 = A1.y; vp2 = A2.y;
            up0 = B0.y; up1 = B1.y; up2 = B2.y;
            break;
        case 1: // Project onto XZ plane
            vp0 = A0.x; vp1 = A1.x; vp2 = A2.x;
            up0 = B0.x; up1 = B1.x; up2 = B2.x;
            break;
        case 2: // Project onto XY plane
            vp0 = A0.x; vp1 = A1.x; vp2 = A2.x;
            up0 = B0.x; up1 = B1.x; up2 = B2.x;
            break;
    }

    // Compute intervals for triangle A
    float isect1[2], isect2[2];

    // Handle triangle A projection
    if (dA0_dA1 > 0.0f) {
        // dA0 and dA1 have same sign, dA2 is different
        isect1[0] = vp2 + (vp0 - vp2) * dA2 / (dA2 - dA0);
        isect1[1] = vp2 + (vp1 - vp2) * dA2 / (dA2 - dA1);
    } else if (dA0_dA2 > 0.0f) {
        // dA0 and dA2 have same sign, dA1 is different
        isect1[0] = vp1 + (vp0 - vp1) * dA1 / (dA1 - dA0);
        isect1[1] = vp1 + (vp2 - vp1) * dA1 / (dA1 - dA2);
    } else if (dA1 * dA2 > 0.0f || dA0 != 0.0f) {
        // dA1 and dA2 have same sign, dA0 is different
        isect1[0] = vp0 + (vp1 - vp0) * dA0 / (dA0 - dA1);
        isect1[1] = vp0 + (vp2 - vp0) * dA0 / (dA0 - dA2);
    } else if (dA1 != 0.0f) {
        // dA0 == 0, use dA1
        isect1[0] = vp1 + (vp0 - vp1) * dA1 / (dA1 - dA0);
        isect1[1] = vp1 + (vp2 - vp1) * dA1 / (dA1 - dA2);
    } else if (dA2 != 0.0f) {
        // dA0 == dA1 == 0, use dA2
        isect1[0] = vp2 + (vp0 - vp2) * dA2 / (dA2 - dA0);
        isect1[1] = vp2 + (vp1 - vp2) * dA2 / (dA2 - dA1);
    } else {
        // Coplanar case
        return coplanar_tri_tri(N0, A0, A1, A2, B0, B1, B2);
    }

    // Handle triangle B projection
    if (dB0_dB1 > 0.0f) {
        // dB0 and dB1 have same sign, dB2 is different
        isect2[0] = up2 + (up0 - up2) * dB2 / (dB2 - dB0);
        isect2[1] = up2 + (up1 - up2) * dB2 / (dB2 - dB1);
    } else if (dB0_dB2 > 0.0f) {
        // dB0 and dB2 have same sign, dB1 is different
        isect2[0] = up1 + (up0 - up1) * dB1 / (dB1 - dB0);
        isect2[1] = up1 + (up2 - up1) * dB1 / (dB1 - dB2);
    } else if (dB1 * dB2 > 0.0f || dB0 != 0.0f) {
        // dB1 and dB2 have same sign, dB0 is different
        isect2[0] = up0 + (up1 - up0) * dB0 / (dB0 - dB1);
        isect2[1] = up0 + (up2 - up0) * dB0 / (dB0 - dB2);
    } else if (dB1 != 0.0f) {
        // dB0 == 0, use dB1
        isect2[0] = up1 + (up0 - up1) * dB1 / (dB1 - dB0);
        isect2[1] = up1 + (up2 - up1) * dB1 / (dB1 - dB2);
    } else if (dB2 != 0.0f) {
        // dB0 == dB1 == 0, use dB2
        isect2[0] = up2 + (up0 - up2) * dB2 / (dB2 - dB0);
        isect2[1] = up2 + (up1 - up2) * dB2 / (dB2 - dB1);
    } else {
        // Coplanar case
        return coplanar_tri_tri(N0, A0, A1, A2, B0, B1, B2);
    }

    // Sort intervals
    if (isect1[0] > isect1[1]) {
        float tmp = isect1[0];
        isect1[0] = isect1[1];
        isect1[1] = tmp;
    }

    if (isect2[0] > isect2[1]) {
        float tmp = isect2[0];
        isect2[0] = isect2[1];
        isect2[1] = tmp;
    }

    // Check for interval overlap
    return (isect1[1] >= isect2[0] && isect2[1] >= isect1[0]);
}

/* this edge to edge test is based on Franlin Antonio's gem:
   "Faster Line Segment Intersection", in Graphics Gems III,
   pp. 199-202 */
#define EDGE_EDGE_TEST(V0,U0,U1)                      \
  Bx=U0[i0]-U1[i0];                                   \
  By=U0[i1]-U1[i1];                                   \
  Cx=V0[i0]-U0[i0];                                   \
  Cy=V0[i1]-U0[i1];                                   \
  f=Ay*Bx-Ax*By;                                      \
  d=By*Cx-Bx*Cy;                                      \
  if((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f))  \
  {                                                   \
    e=Ax*Cy-Ay*Cx;                                    \
    if(f>0)                                           \
    {                                                 \
      if(e>=0 && e<=f) return 1;                      \
    }                                                 \
    else                                              \
    {                                                 \
      if(e<=0 && e>=f) return 1;                      \
    }                                                 \
  }

#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
  float Ax,Ay,Bx,By,Cx,Cy,e,d,f;               \
  Ax=V1[i0]-V0[i0];                            \
  Ay=V1[i1]-V0[i1];                            \
  /* test edge U0,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U0,U1);                    \
  /* test edge U1,U2 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U1,U2);                    \
  /* test edge U2,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U2,U0);                    \
}

#define POINT_IN_TRI(V0,U0,U1,U2)           \
{                                           \
  float a,b,c,d0,d1,d2;                     \
  /* is T1 completly inside T2? */          \
  /* check if V0 is inside tri(U0,U1,U2) */ \
  a=U1[i1]-U0[i1];                          \
  b=-(U1[i0]-U0[i0]);                       \
  c=-a*U0[i0]-b*U0[i1];                     \
  d0=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U2[i1]-U1[i1];                          \
  b=-(U2[i0]-U1[i0]);                       \
  c=-a*U1[i0]-b*U1[i1];                     \
  d1=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U0[i1]-U2[i1];                          \
  b=-(U0[i0]-U2[i0]);                       \
  c=-a*U2[i0]-b*U2[i1];                     \
  d2=a*V0[i0]+b*V0[i1]+c;                   \
  if(d0*d1>0.0)                             \
  {                                         \
    if(d0*d2>0.0) return 1;                 \
  }                                         \
}

int coplanar_tri_tri(const VxVector &N, const VxVector &V0, const VxVector &V1, const VxVector &V2,
                     const VxVector &U0, const VxVector &U1, const VxVector &U2) {
    float A[3];
    short i0, i1;
    /* first project onto an axis-aligned plane, that maximizes the area */
    /* of the triangles, compute indices: i0,i1. */
    A[0] = fabs(N[0]);
    A[1] = fabs(N[1]);
    A[2] = fabs(N[2]);
    if (A[0] > A[1]) {
        if (A[0] > A[2]) {
            i0 = 1; /* A[0] is greatest */
            i1 = 2;
        } else {
            i0 = 0; /* A[2] is greatest */
            i1 = 1;
        }
    } else /* A[0]<=A[1] */
    {
        if (A[2] > A[1]) {
            i0 = 0; /* A[2] is greatest */
            i1 = 1;
        } else {
            i0 = 0; /* A[1] is greatest */
            i1 = 2;
        }
    }

    /* test all edges of triangle 1 against the edges of triangle 2 */
    EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2);
    EDGE_AGAINST_TRI_EDGES(V1, V2, U0, U1, U2);
    EDGE_AGAINST_TRI_EDGES(V2, V0, U0, U1, U2);

    /* finally, test if tri1 is totally contained in tri2 or vice versa */
    POINT_IN_TRI(V0, U0, U1, U2);
    POINT_IN_TRI(U0, V0, V1, V2);

    return 0;
}
