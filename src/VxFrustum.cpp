#include "VxFrustum.h"

VxFrustum::VxFrustum()
    : m_Origin(VxVector::axis0()),
      m_Right(VxVector::axisX()),
      m_Up(VxVector::axisY()),
      m_Dir(VxVector::axisZ()),
      m_RBound(1.0f),
      m_UBound(1.0f),
      m_DMin(1.0f),
      m_DMax(2.0f),
      m_DRatio(0.0f),
      m_RF(0.0f),
      m_UF(0.0f) { Update();}

VxFrustum::VxFrustum(const VxVector &origin, const VxVector &right, const VxVector &up, const VxVector &dir, float nearplane, float farplane, float fov, float aspectratio)
    : m_Origin(origin),
      m_Right(right),
      m_Up(up),
      m_Dir(dir),
      m_RBound(tanf(fov * 0.5f) * aspectratio),
      m_UBound(tanf(fov * 0.5f)),
      m_DMin(nearplane),
      m_DMax(farplane),
      m_DRatio(0.0f),
      m_RF(0.0f),
      m_UF(0.0f) { Update(); }

void VxFrustum::Update() {
    // Calculate derived ratios and factors
    m_DRatio = m_DMax / m_DMin;
    m_RF = m_RBound * m_DMax * -2.0f;
    m_UF = m_UBound * m_DMax * -2.0f;

    // Scale direction vectors for near plane
    VxVector nearDirVec = m_Dir * m_DMin;
    VxVector upVec = m_Up * m_UBound;
    VxVector rightVec = m_Right * m_RBound;

    // Calculate key corner points needed for plane definitions
    VxVector nearBottomLeft = m_Origin + nearDirVec - rightVec - upVec;
    VxVector nearTopRight = m_Origin + nearDirVec + rightVec + upVec;

    // Decompiled code uses farTopRight for the far plane definition.
    VxVector farTopRight = m_Origin + (nearDirVec + rightVec + upVec) * m_DRatio;

    // Create near plane (normal points back towards eye)
    VxVector nearNormal = -m_Dir;
    m_NearPlane.Create(nearNormal, nearBottomLeft);

    // Create far plane (normal points towards eye)
    m_FarPlane.Create(m_Dir, farTopRight);

    // Calculate side plane normals using vectors from the frustum's origin (apex)
    // to the corners of the near plane. This matches the method in the decompiled code.
    VxVector nbl_rel = nearDirVec - rightVec - upVec;
    VxVector nbr_rel = nearDirVec + rightVec - upVec;
    VxVector ntl_rel = nearDirVec - rightVec + upVec;
    VxVector ntr_rel = nearDirVec + rightVec + upVec;

    // Create left plane
    VxVector leftNormal = CrossProduct(ntl_rel, nbl_rel);
    leftNormal.Normalize();
    m_LeftPlane.Create(leftNormal, nearBottomLeft);

    // Create right plane
    VxVector rightNormal = CrossProduct(nbr_rel, ntr_rel);
    rightNormal.Normalize();
    m_RightPlane.Create(rightNormal, nearTopRight);

    // Create bottom plane
    VxVector bottomNormal = CrossProduct(nbr_rel, nbl_rel);
    bottomNormal.Normalize();
    m_BottomPlane.Create(bottomNormal, nearBottomLeft);

    // Create top plane
    VxVector topNormal = CrossProduct(ntl_rel, ntr_rel);
    topNormal.Normalize();
    m_UpPlane.Create(topNormal, nearTopRight);
}

void VxFrustum::ComputeVertices(VxVector vertices[8]) const {
    // Scale direction vectors by distance and bounds
    VxVector nearDirVec = m_Dir * m_DMin;
    VxVector rightVec = m_Right * m_RBound;
    VxVector upVec = m_Up * m_UBound;

    // Compute near plane vertices relative to origin, as seen in the decompilation
    vertices[0] = nearDirVec - rightVec - upVec;
    vertices[1] = nearDirVec + rightVec - upVec;
    vertices[2] = nearDirVec + rightVec + upVec;
    vertices[3] = nearDirVec - rightVec + upVec;

    // Now compute far vertices and adjust near vertices to world space
    for (int i = 0; i < 4; i++) {
        // Get vector from origin to near vertex (which is just vertices[i] at this point)
        const VxVector& nearVec = vertices[i];
        // Scale by depth ratio to get far vector relative to origin
        VxVector farVec = nearVec * m_DRatio;
        // Far vertex is origin plus scaled vector
        vertices[i + 4] = m_Origin + farVec;
        // Adjust near vertex to be in world space
        vertices[i] += m_Origin;
    }
}

void VxFrustum::Transform(const VxMatrix &invworldmat) {
    // Scale direction vectors by their bounds before transformation
    m_Right *= m_RBound;
    m_Up *= m_UBound;
    m_Dir *= m_DMin;

    // Transform the origin (full matrix with translation)
    VxVector newOrigin;
    Vx3DMultiplyMatrixVector(&newOrigin, invworldmat, &m_Origin);
    m_Origin = newOrigin;

    // Transform the scaled direction vectors (rotation only)
    VxVector resultVectors[3];
    Vx3DRotateVectorMany(resultVectors, invworldmat, &m_Right, 3, sizeof(VxVector));

    // Extract new magnitudes (bounds) and normalize direction vectors
    float newRBound = resultVectors[0].Magnitude();
    float newUBound = resultVectors[1].Magnitude();
    float newDMin = resultVectors[2].Magnitude();

    // Update bounds and distances
    m_RBound = newRBound;
    m_UBound = newUBound;
    m_DMin = newDMin;
    m_DMax = newDMin * m_DRatio; // Preserve ratio

    // Normalize direction vectors
    if (newRBound > EPSILON) {
        m_Right = resultVectors[0] * (1.0f / newRBound);
    }
    if (newUBound > EPSILON) {
        m_Up = resultVectors[1] * (1.0f / newUBound);
    }
    if (newDMin > EPSILON) {
        m_Dir = resultVectors[2] * (1.0f / newDMin);
    }

    // Update the planes with the new transformed vectors
    Update();
}
