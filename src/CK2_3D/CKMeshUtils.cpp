#include "RCKMesh.h"

#include "CKRasterizer.h"
#include "RCKRenderContext.h"
#include "RCK3dEntity.h"
#include "RCKMaterial.h"
#include "RCKTexture.h"

void (*g_BuildFaceNormalsFunc)(CKFace *, CKWORD *, int, VxVertex *, int);
void (*g_BuildNormalsFunc)(CKFace *, CKWORD *, int, VxVertex *, int);
int (*g_RayIntersection)(RCKMesh *, VxVector &, VxVector &, VxIntersectionDesc *, CK_RAYINTERSECTION, VxMatrix const &);
void (*g_NormalizeFunc)(VxVertex *, int);

void SetProcessorSpecific_FunctionsPtr() {
    // IDA @ 0x1002e020
    // Set default generic implementations
    g_BuildFaceNormalsFunc = BuildFaceNormalsGenericFunc;
    g_BuildNormalsFunc = BuildNormalsGenericFunc;
    g_RayIntersection = RayIntersectionGenericFunc;
    g_NormalizeFunc = NormalizeGenericFunc;

    // IDA lines 6-12: Check for SSE support (bit 25 of processor features)
    // Original code checks GetProcessorFeatures() & 0x2000000 and assigns SIMD versions
    // For now, use generic implementations only as SIMD versions are not implemented
    // TODO: Implement SIMD versions (BuildFaceNormalsSIMDFunc, BuildNormalsSIMDFunc, 
    //       Mesh_RayIntersectionSIMDFunc, NormalizeSIMDFunc) if needed
}

// =====================================================
// Generic Math Functions for Mesh Operations
// IDA @ 0x1002e1ba - 0x1002f52e
// =====================================================

// IDA @ 0x1002e82f: Precise texture pick for alpha testing
CKBOOL PreciseTexturePick(CKMaterial *mat, float u, float v) {
    if (!mat)
        return TRUE;

    CKTexture *tex = mat->GetTexture(0);
    if (!tex)
        return TRUE;

    // Get alpha reference value from material
    int alphaRef = mat->GetAlphaRef();
    if (alphaRef == 0)
        return TRUE;

    // Apply texture addressing mode
    VXTEXTURE_ADDRESSMODE addrMode = mat->GetTextureAddressMode();
    switch (addrMode) {
    case VXTEXTURE_ADDRESSWRAP:
        u = u - (float) (int) u;
        if (u < 0.0f)
            u += 1.0f;
        v = v - (float) (int) v;
        if (v < 0.0f)
            v += 1.0f;
        break;
    case VXTEXTURE_ADDRESSMIRROR: {
        int ui = (int) u;
        u = u - (float) ui;
        if (u < 0.0f)
            u += 1.0f;
        if (ui & 1)
            u = 1.0f - u;
        int vi = (int) v;
        v = v - (float) vi;
        if (v < 0.0f)
            v += 1.0f;
        if (vi & 1)
            v = 1.0f - v;
        break;
    }
    case VXTEXTURE_ADDRESSCLAMP:
    case VXTEXTURE_ADDRESSBORDER:
        if (u < 0.0f)
            u = 0.0f;
        else if (u > 1.0f)
            u = 1.0f;
        if (v < 0.0f)
            v = 0.0f;
        else if (v > 1.0f)
            v = 1.0f;
        break;
    default:
        break;
    }

    // Get pixel coordinates
    int width = tex->GetWidth();
    int height = tex->GetHeight();
    int px = (int) ((float) (width - 1) * u);
    int py = (int) ((float) (height - 1) * v);

    // Get pixel alpha
    CKDWORD pixel = tex->GetPixel(px, py);
    CKDWORD alpha = pixel >> 24;

    return alpha >= (CKDWORD) alphaRef;
}

// IDA @ 0x1002e1ba: Build face normals from vertex positions
void BuildFaceNormalsGenericFunc(CKFace *faces, CKWORD *indices, int faceCount, VxVertex *vertices, int vertexCount) {
    int indexOffset = 0;     // v16
    CKFace *facePtr = faces; // a1
    for (int i = 0; i < faceCount; ++i) {
        // Get vertex positions for this face (IDA lines 22-23)
        const VxVector &v0 = vertices[indices[indexOffset]].m_Position;
        const VxVector &v1 = vertices[indices[indexOffset + 1]].m_Position;
        const VxVector &v2 = vertices[indices[indexOffset + 2]].m_Position;

        // Calculate edge vectors and cross product using SDK functions
        VxVector edge1 = v1 - v0;
        VxVector edge2 = v2 - v0;
        facePtr->m_Normal = CrossProduct(edge1, edge2);

        // Normalize using SDK function (handles zero-length internally)
        float length = Magnitude(facePtr->m_Normal);
        if (length > 0.0f) {
            facePtr->m_Normal *= (1.0f / length);
        }

        // IDA lines 35-36
        ++facePtr;        // IDA line 35: ++a1
        indexOffset += 3; // IDA line 36: v16 += 3
    }
}

// IDA @ 0x1002e2cb: Build vertex normals by averaging face normals
void BuildNormalsGenericFunc(CKFace *faces, CKWORD *indices, int faceCount, VxVertex *vertices, int vertexCount) {
    // First build face normals (IDA line 15)
    BuildFaceNormalsGenericFunc(faces, indices, faceCount, vertices, vertexCount);

    // Clear all vertex normals (IDA lines 16-20)
    VxVertex *v = vertices;
    for (int i = 0; i < vertexCount; ++i) {
        v->m_Normal.Set(0.0f, 0.0f, 0.0f);
        ++v;
    }

    // Accumulate face normals to vertices (IDA lines 21-36)
    int indexOffset = 0;     // v14
    CKFace *facePtr = faces; // IDA uses faces pointer directly (a2)
    for (int j = 0; j < faceCount; ++j) {
        const VxVector &faceNormal = facePtr->m_Normal;

        // Add face normal to each vertex of this face using SDK operator+=
        vertices[indices[indexOffset]].m_Normal += faceNormal;
        int idx1 = indexOffset + 1; // v15

        vertices[indices[idx1]].m_Normal += faceNormal;
        int idx2 = idx1 + 1; // v15 after increment

        vertices[indices[idx2]].m_Normal += faceNormal;

        indexOffset = idx2 + 1; // v14 = v15 + 1 (IDA line 33)
        ++facePtr;              // IDA line 35: ++a2
    }

    // Normalize all vertex normals (IDA lines 37-43)
    VxVertex *vn = vertices;
    for (int k = 0; k < vertexCount; ++k) {
        vn->m_Normal.Normalize();
        ++vn;
    }
}

// IDA @ 0x1002e423: Normalize vertex normals
void NormalizeGenericFunc(VxVertex *vertices, int count) {
    // IDA: a1 treated as VxVector*, stride is 32 bytes (sizeof(VxVertex))
    // Normalizes a1+1 which is the Normal field
    VxVector *ptr = &vertices->m_Position; // Start at Position
    for (int i = 0; i < count; ++i) {
        (ptr + 1)->Normalize();                 // Normalize the Normal (Position + 1 VxVector)
        ptr = (VxVector *) ((char *) ptr + 32); // Move to next VxVertex
    }
}

// IDA @ 0x1002ea85: Mesh ray intersection (generic implementation)
// This is a complex function with spatial partitioning optimization
int RayIntersectionGenericFunc(RCKMesh *mesh, VxVector &origin, VxVector &direction,
                               VxIntersectionDesc *desc, CK_RAYINTERSECTION mode,
                               const VxMatrix &worldMatrix) {
    float minDist = 1.0e35f; // v82
    CKBOOL foundHit = FALSE; // v74

    // Get mesh data directly from protected members (friend access)
    float *vertexPtr = (float *) mesh->m_Vertices.Begin(); // v78
    CKFace *faces = mesh->m_Faces.Begin();                 // v80
    CKWORD *indices = mesh->m_FaceVertexIndices.Begin();   // v77
    int faceCount = mesh->m_Faces.Size();                  // v83
    int vertexCount = mesh->m_Vertices.Size();             // v84
    int hitCount = 0;                                      // v79

    if (faceCount == 0 || vertexCount == 0)
        return 0;

    // Allocate classification buffer for spatial culling (IDA: CKMemoryPool)
    CKDWORD *vertexFlags = new CKDWORD[vertexCount]; // v81
    memset(vertexFlags, 0, sizeof(CKDWORD) * vertexCount);

    int earlyOutMask = 0; // v75

    // Spatial partitioning optimization for large meshes (>= 16 faces)
    if (faceCount >= 16) {
        VxVector axis1, axis2; // v69, v68

        // Create orthogonal basis from ray direction using SDK functions
        // Cross with Y axis first (IDA: VxVector::axisY())
        axis1 = CrossProduct(direction, VxVector::axisY());

        // If direction is parallel to Y, use X axis
        if (SquareMagnitude(axis1) < 0.00000011920929f) {
            axis1 = CrossProduct(direction, VxVector::axisX());
        }

        // Create second orthogonal axis, then recompute axis1
        axis2 = CrossProduct(direction, axis1);
        axis1 = CrossProduct(direction, axis2);

        // Project ray origin onto both axes using SDK DotProduct
        float originProj1 = DotProduct(origin, axis1); // v72
        float originProj2 = DotProduct(origin, axis2); // v71

        earlyOutMask = -1; // All bits set

        // Classify each vertex relative to the ray
        float *vPtr = vertexPtr; // v73
        CKDWORD *flagPtr = vertexFlags;
        for (int v = 0; v < vertexCount; ++v) {
            float proj1 = DotProduct(*(VxVector *) vPtr, axis1);
            if (proj1 >= originProj1)
                *flagPtr |= 2;
            else
                *flagPtr |= 1;

            float proj2 = DotProduct(*(VxVector *) vPtr, axis2);
            if (proj2 >= originProj2)
                *flagPtr |= 8;
            else
                *flagPtr |= 4;

            earlyOutMask &= *flagPtr;

            vPtr += 8; // sizeof(VxVertex) / sizeof(float) = 32/4 = 8
            ++flagPtr;
        }
    }

    // If not all vertices are on one side, perform intersection tests
    if (earlyOutMask == 0) {
        int bestI1 = 0;           // v65
        int bestI2 = 0;           // v67
        float bestFaceIdx = 0.0f; // v66

        const VxVector *faceNormalPtr = (const VxVector *) faces; // v59 - points to face normal
        CKWORD *indexPtr = indices;                               // v64

        // Create ray structure (IDA: sub_100301F0)
        VxRay ray;
        ray.m_Origin = origin;
        ray.m_Direction = direction;

        int i1 = 1; // v57
        int i2 = 2; // v55

        // Select intersection function based on mode (IDA lines 153-158)
        typedef XBOOL (*IntersectFunc)(const VxRay &, const VxVector &, const VxVector &,
                                       const VxVector &, const VxVector &, VxVector &, float &, int &, int &);
        IntersectFunc culledFunc = (IntersectFunc) VxIntersect::RayFaceCulled; // v61
        IntersectFunc twoSidedFunc = (IntersectFunc) VxIntersect::RayFace;     // v62

        if (mode != 0) {
            // CKRAYINTERSECTION_SEGMENT or others
            culledFunc = (IntersectFunc) VxIntersect::SegmentFaceCulled;
            twoSidedFunc = (IntersectFunc) VxIntersect::SegmentFace;
        }

        VxVector intersectPoint; // v60
        float dist;              // v63

        // Test each face (IDA lines 161-187)
        for (int f = 0; f < faceCount; ++f) {
            // Early out using vertex classification (IDA line 163)
            if (!(vertexFlags[indexPtr[2]] & vertexFlags[indexPtr[1]] & vertexFlags[indexPtr[0]])) {
                // Get vertex positions (IDA lines 165-167)
                const VxVector *v0 = (const VxVector *) &vertexPtr[8 * indexPtr[0]];
                const VxVector *v1 = (const VxVector *) &vertexPtr[8 * indexPtr[1]];
                const VxVector *v2 = (const VxVector *) &vertexPtr[8 * indexPtr[2]];

                // Get material and check if two-sided (IDA lines 168-170)
                CKMaterial *mat = mesh->GetFaceMaterial(f);
                CKBOOL twoSided = mat && mat->IsTwoSided();

                // Call appropriate intersection function (IDA line 171)
                CKBOOL hit;
                if (twoSided)
                    hit = twoSidedFunc(ray, *v0, *v1, *v2, *faceNormalPtr, intersectPoint, dist, i1, i2);
                else
                    hit = culledFunc(ray, *v0, *v1, *v2, *faceNormalPtr, intersectPoint, dist, i1, i2);

                if (hit) {
                    ++hitCount;
                    if (dist < minDist) {
                        foundHit = TRUE;
                        minDist = dist;
                        bestI1 = i1;
                        bestI2 = i2;
                        bestFaceIdx = *(float *) &f; // Store face index as float (IDA line 180)
                    }
                }
            }

            // Move to next face (IDA lines 184-186)
            faceNormalPtr = (const VxVector *) ((char *) faceNormalPtr + 16); // sizeof(CKFace) = 16
            indexPtr += 3;
        }

        // Fill in intersection description (IDA lines 188-265)
        if (foundHit && desc) {
            // Calculate intersection point: origin + direction * minDist (IDA lines 190-191)
            desc->IntersectionPoint = origin + direction * minDist;

            // Get face index back from float (IDA line 192)
            int faceIdx = *(int *) &bestFaceIdx;
            int indexBase = faceIdx * 3;

            // Get vertex pointers (IDA lines 193-195)
            // These point to VxVertex structures - v42/v41/v48 in IDA
            const float *vert0 = &vertexPtr[8 * indices[indexBase]];
            const float *vert1 = &vertexPtr[8 * indices[indexBase + 1]];
            const float *vert2 = &vertexPtr[8 * indices[indexBase + 2]];

            // Get barycentric coordinates (IDA lines 196-205)
            float c0, c1, c2; // v46, v45, arg4a
            VxIntersect::GetPointCoefficients(
                desc->IntersectionPoint,
                *(const VxVector *) vert0,
                *(const VxVector *) vert1,
                *(const VxVector *) vert2,
                bestI1, bestI2, c0, c1, c2);

            // Interpolate normal: vert[1] is the Normal in VxVertex (IDA lines 206-210)
            // v42[1] = Normal, accessed as &v42[1].x which is vert0 + 3 floats
            const VxVector &n0 = *(const VxVector *) (vert0 + 3);
            const VxVector &n1 = *(const VxVector *) (vert1 + 3);
            const VxVector &n2 = *(const VxVector *) (vert2 + 3);
            desc->IntersectionNormal = c0 * n0 + c1 * n1 + c2 * n2;

            // Interpolate texture coordinates (IDA lines 211-212)
            // v42[2] = UV in VxVertex, accessed as vert0 + 6 floats
            desc->TexU = c0 * vert0[6] + c1 * vert1[6] + c2 * vert2[6]; // UV.x
            desc->TexV = c0 * vert0[7] + c1 * vert1[7] + c2 * vert2[7]; // UV.y

            // Get material for perspective correction and alpha test (IDA lines 213-262)
            CKMaterial *faceMat = mesh->GetFaceMaterial(faceIdx);
            if (faceMat) {
                // Check if perspective correction is disabled (IDA lines 216-255)
                if (!faceMat->PerspectiveCorrectionEnabled()) {
                    CKContext *ctx = mesh->m_Context;
                    RCKRenderContext *dev = (RCKRenderContext *) ctx->GetPlayerRenderContext();
                    if (dev) {
                        CK3dEntity *rootEntity = dev->m_RenderedScene->GetRootEntity();
                        if (desc->Object == (CKRenderObject *) rootEntity) {
                            // Perform perspective-correct UV interpolation
                            VxMatrix invWorld;
                            Vx3DMatrixIdentity(invWorld);
                            VxMatrix combined;
                            Vx3DMatrixIdentity(combined);

                            const VxMatrix &invWorldMat = rootEntity->GetInverseWorldMatrix();
                            Vx3DMultiplyMatrix(combined, invWorldMat, worldMatrix);

                            VxMatrix projCombined;
                            Vx3DMultiplyMatrix4(projCombined, dev->m_RasterizerContext->m_ProjectionMatrix, combined);

                            // Transform points to clip space
                            VxVector4 clipIntersect, clipV0, clipV1, clipV2;
                            Vx3DMultiplyMatrixVector4(&clipIntersect, projCombined, &desc->IntersectionPoint);
                            Vx3DMultiplyMatrixVector4(&clipV0, projCombined, (const VxVector *) vert0);
                            Vx3DMultiplyMatrixVector4(&clipV1, projCombined, (const VxVector *) vert1);
                            Vx3DMultiplyMatrixVector4(&clipV2, projCombined, (const VxVector *) vert2);

                            // Perspective divide
                            clipV0.w = 1.0f / clipV0.w;
                            clipV1.w = 1.0f / clipV1.w;
                            clipV2.w = 1.0f / clipV2.w;
                            clipIntersect.w = 1.0f / clipIntersect.w;

                            clipV0.x *= clipV0.w;
                            clipV1.x *= clipV1.w;
                            clipV2.x *= clipV2.w;
                            clipIntersect.x *= clipIntersect.w;

                            clipV0.y *= clipV0.w;
                            clipV1.y *= clipV1.w;
                            clipV2.y *= clipV2.w;
                            clipIntersect.y *= clipIntersect.w;

                            // Recalculate barycentric coordinates in screen space
                            int newI1 = 0, newI2 = 1;
                            VxIntersect::GetPointCoefficients(
                                *(const VxVector *) &clipIntersect,
                                *(const VxVector *) &clipV0,
                                *(const VxVector *) &clipV1,
                                *(const VxVector *) &clipV2,
                                newI1, newI2, c0, c1, c2);

                            // Recalculate UVs with perspective-correct interpolation
                            desc->TexU = c0 * vert0[6] + c1 * vert1[6] + c2 * vert2[6];
                            desc->TexV = c0 * vert0[7] + c1 * vert1[7] + c2 * vert2[7];
                        }
                    }
                }

                // Precise texture pick (alpha test) (IDA lines 256-261)
                if (!PreciseTexturePick(faceMat, desc->TexU, desc->TexV)) {
                    delete[] vertexFlags;
                    return 0; // Hit was on transparent pixel
                }
            }

            // Final output (IDA lines 263-264)
            desc->Distance = minDist;
            desc->FaceIndex = faceIdx;
        }
    }

    delete[] vertexFlags;
    return foundHit ? hitCount : 0;
}
