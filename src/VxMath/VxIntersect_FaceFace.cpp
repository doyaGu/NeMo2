#include "VxIntersect.h"

#include "VxVector.h"
#include "VxMatrix.h"
#include "VxPlane.h"

//---------- Faces

// Is a point inside the boundary of a face
XBOOL VxIntersect::PointInFace(const VxVector &point, const VxVector &pt0, const VxVector &pt1, const VxVector &pt2,
                               const VxVector &norm, int &i1, int &i2) {
    // Find dominant axis of normal to determine projection plane
    float maxAbs = XAbs(norm.x);
    i1 = 1; // Y
    i2 = 2; // Z

    if (maxAbs < XAbs(norm.y)) {
        i1 = 0; // X
        i2 = 2; // Z
        maxAbs = XAbs(norm.y);
    }

    if (maxAbs < XAbs(norm.z)) {
        i1 = 0; // X
        i2 = 1; // Y
    }

    // Perform three edge tests using 2D cross products
    // Test: (point - pt1) x (pt2 - pt1)
    char flags = 1;
    if ((point[i1] - pt1[i1]) * (pt2[i2] - pt1[i2]) - (pt2[i1] - pt1[i1]) * (point[i2] - pt1[i2]) >= 0.0f) {
        flags = 2;
    }

    // Test: (point - pt2) x (pt0 - pt2)
    if ((point[i1] - pt2[i1]) * (pt0[i2] - pt2[i2]) - (point[i2] - pt2[i2]) * (pt0[i1] - pt2[i1]) >= 0.0f) {
        flags &= 2;
    } else {
        flags &= 1;
    }

    if (!flags) return FALSE;

    // Test: (point - pt0) x (pt1 - pt0)
    if ((point[i1] - pt0[i1]) * (pt1[i2] - pt0[i2]) - (point[i2] - pt0[i2]) * (pt1[i1] - pt0[i1]) >= 0.0f) {
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
    // Signed distances of B vertices to plane of A
    const float planeA_d = -DotProduct(N0, A0);
    float distB[3] = {
        DotProduct(N0, B0) + planeA_d,
        DotProduct(N0, B1) + planeA_d,
        DotProduct(N0, B2) + planeA_d,
    };
    if (XAbs(distB[0]) < EPSILON) distB[0] = 0.0f;
    if (XAbs(distB[1]) < EPSILON) distB[1] = 0.0f;
    if (XAbs(distB[2]) < EPSILON) distB[2] = 0.0f;
    if (distB[0] * distB[1] > 0.0f && distB[0] * distB[2] > 0.0f) return FALSE;

    // Signed distances of A vertices to plane of B
    const float planeB_d = -DotProduct(N1, B0);
    float distA[3] = {
        DotProduct(N1, A0) + planeB_d,
        DotProduct(N1, A1) + planeB_d,
        DotProduct(N1, A2) + planeB_d,
    };
    if (XAbs(distA[0]) < EPSILON) distA[0] = 0.0f;
    if (XAbs(distA[1]) < EPSILON) distA[1] = 0.0f;
    if (XAbs(distA[2]) < EPSILON) distA[2] = 0.0f;
    const float distA01 = distA[0] * distA[1];
    const float distA02 = distA[0] * distA[2];
    if (distA01 > 0.0f && distA02 > 0.0f) return FALSE;

    // Direction of intersection line between the two planes
    const VxVector D = CrossProduct(N0, N1);
    const VxVector absD = Absolute(D);

    int maxc = 0;
    float maxAbs = absD[0];
    if (absD[1] >= maxAbs) {
        maxc = 1;
        maxAbs = absD[1];
    }
    if (absD[2] >= maxAbs) {
        maxc = 2;
    }

    // Project vertices on the dominant axis of the intersection direction
    const float vA[3] = {A0[maxc], A1[maxc], A2[maxc]};
    const float vB[3] = {B0[maxc], B1[maxc], B2[maxc]};

    // Compute the interval of intersection on the line for each triangle.
    // This matches the original Möller-style branch structure, but keeps it readable.
    float aBase;
    float aNum0;
    float aNum1;
    float aDen0;
    float aDen1;
    if (distA01 > 0.0f) {
        aBase = vA[2];
        aNum0 = (vA[0] - vA[2]) * distA[2];
        aNum1 = (vA[1] - vA[2]) * distA[2];
        aDen0 = distA[2] - distA[0];
        aDen1 = distA[2] - distA[1];
    } else if (distA02 > 0.0f) {
        aBase = vA[1];
        aNum0 = (vA[0] - vA[1]) * distA[1];
        aNum1 = (vA[2] - vA[1]) * distA[1];
        aDen0 = distA[1] - distA[0];
        aDen1 = distA[1] - distA[2];
    } else if (distA[1] * distA[2] > 0.0f || distA[0] != 0.0f) {
        aBase = vA[0];
        aNum0 = (vA[1] - vA[0]) * distA[0];
        aNum1 = (vA[2] - vA[0]) * distA[0];
        aDen0 = distA[0] - distA[1];
        aDen1 = distA[0] - distA[2];
    } else if (distA[1] != 0.0f) {
        aBase = vA[1];
        aNum0 = (vA[0] - vA[1]) * distA[1];
        aNum1 = (vA[2] - vA[1]) * distA[1];
        aDen0 = distA[1] - distA[0];
        aDen1 = distA[1] - distA[2];
    } else if (distA[2] == 0.0f) {
        return coplanar_tri_tri(N0, A0, A1, A2, B0, B1, B2);
    } else {
        aBase = vA[2];
        aNum0 = (vA[0] - vA[2]) * distA[2];
        aNum1 = (vA[1] - vA[2]) * distA[2];
        aDen0 = distA[2] - distA[0];
        aDen1 = distA[2] - distA[1];
    }

    const float distB01 = distB[0] * distB[1];
    const float distB02 = distB[0] * distB[2];
    float bBase;
    float bNum0;
    float bNum1;
    float bDen0;
    float bDen1;
    if (distB01 > 0.0f) {
        bBase = vB[2];
        bNum0 = (vB[0] - vB[2]) * distB[2];
        bNum1 = (vB[1] - vB[2]) * distB[2];
        bDen0 = distB[2] - distB[0];
        bDen1 = distB[2] - distB[1];
    } else if (distB02 > 0.0f) {
        bBase = vB[1];
        bNum0 = (vB[0] - vB[1]) * distB[1];
        bNum1 = (vB[2] - vB[1]) * distB[1];
        bDen0 = distB[1] - distB[0];
        bDen1 = distB[1] - distB[2];
    } else if (distB[1] * distB[2] > 0.0f || distB[0] != 0.0f) {
        bBase = vB[0];
        bNum0 = (vB[1] - vB[0]) * distB[0];
        bNum1 = (vB[2] - vB[0]) * distB[0];
        bDen0 = distB[0] - distB[1];
        bDen1 = distB[0] - distB[2];
    } else if (distB[1] != 0.0f) {
        bBase = vB[1];
        bNum0 = (vB[0] - vB[1]) * distB[1];
        bNum1 = (vB[2] - vB[1]) * distB[1];
        bDen0 = distB[1] - distB[0];
        bDen1 = distB[1] - distB[2];
    } else if (distB[2] != 0.0f) {
        bBase = vB[2];
        bNum0 = (vB[0] - vB[2]) * distB[2];
        bNum1 = (vB[1] - vB[2]) * distB[2];
        bDen0 = distB[2] - distB[0];
        bDen1 = distB[2] - distB[1];
    } else {
        return coplanar_tri_tri(N0, A0, A1, A2, B0, B1, B2);
    }

    // Avoid divisions: compare intervals after multiplying by common denominator products.
    const float aDenProd = aDen0 * aDen1;
    const float bDenProd = bDen0 * bDen1;
    const float denProd = aDenProd * bDenProd;

    float isectA0 = bDenProd * (aDen1 * aNum0) + aBase * denProd;
    float isectA1 = bDenProd * (aDen0 * aNum1) + aBase * denProd;
    float isectB0 = aDenProd * (bDen1 * bNum0) + bBase * denProd;
    float isectB1 = aDenProd * (bDen0 * bNum1) + bBase * denProd;

    if (isectA0 > isectA1) {
        const float tmp = isectA0;
        isectA0 = isectA1;
        isectA1 = tmp;
    }
    if (isectB0 > isectB1) {
        const float tmp = isectB0;
        isectB0 = isectB1;
        isectB1 = tmp;
    }

    return (isectA1 >= isectB0 && isectB1 >= isectA0);
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
    A[0] = fabsf(N[0]);
    A[1] = fabsf(N[1]);
    A[2] = fabsf(N[2]);
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
