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
      m_UF(0.0f) { Update(); }

VxFrustum::VxFrustum(const VxVector &origin, const VxVector &right, const VxVector &up, const VxVector &dir, float nearplane, float farplane, float fov, float aspectratio)
    : m_Origin(origin),
      m_Right(right),
      m_Up(up),
      m_Dir(dir),
      m_RBound(tanf(fov * 0.5f) * nearplane),
      m_UBound(tanf(fov * 0.5f) * nearplane * aspectratio),
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

    // Calculate relative vectors from origin to near plane corners
    // nbl_rel = nearDirVec - rightVec - upVec (near-bottom-left relative)
    // ntl_rel = nearDirVec - rightVec + upVec (near-top-left relative)
    // nbr_rel = nearDirVec + rightVec - upVec (near-bottom-right relative)
    // ntr_rel = nearDirVec + rightVec + upVec (near-top-right relative)
    VxVector nbl_rel = nearDirVec - rightVec - upVec;
    VxVector ntl_rel = nearDirVec - rightVec + upVec;
    VxVector nbr_rel = nearDirVec + rightVec - upVec;
    VxVector ntr_rel = nearDirVec + rightVec + upVec;

    // Near plane vertex in world space (used as point on plane)
    VxVector nbl = m_Origin + nbl_rel;
    VxVector ntr = m_Origin + ntr_rel;

    // Far plane vertex (scale near vertex by DRatio and add origin)
    VxVector farVertex = m_Origin + ntr_rel * m_DRatio;

    // Near plane: normal = -m_Dir (pointing toward camera = outward from frustum)
    // Point in front of near plane (closer to camera than near) has Classify > 0
    VxVector nearNormal = -m_Dir;
    m_NearPlane.Create(nearNormal, nbl);

    // Far plane: normal = m_Dir (pointing away from camera = outward from frustum)
    // Point behind far plane (further than far) has Classify > 0
    m_FarPlane.Create(m_Dir, farVertex);

    // For the side planes, we need normals pointing OUTWARD from the frustum
    // The side planes pass through the frustum origin and the near plane edges
    // For perspective, this creates a cone-like shape

    // Left plane: passes through origin and left edge (nbl to ntl)
    // Normal = normalize(CrossProduct(nbl_rel, ntl_rel)) for outward normal pointing left
    VxVector leftNormal = CrossProduct(nbl_rel, ntl_rel);
    leftNormal.Normalize();
    m_LeftPlane.Create(leftNormal, m_Origin);

    // Right plane: passes through origin and right edge (nbr to ntr)
    // Normal = normalize(CrossProduct(ntr_rel, nbr_rel)) for outward normal pointing right
    VxVector rightNormal = CrossProduct(ntr_rel, nbr_rel);
    rightNormal.Normalize();
    m_RightPlane.Create(rightNormal, m_Origin);

    // Bottom plane: passes through origin and bottom edge (nbl to nbr)
    // Normal = normalize(CrossProduct(nbr_rel, nbl_rel)) for outward normal pointing down
    VxVector bottomNormal = CrossProduct(nbr_rel, nbl_rel);
    bottomNormal.Normalize();
    m_BottomPlane.Create(bottomNormal, m_Origin);

    // Top plane: passes through origin and top edge (ntl to ntr)
    // Normal = normalize(CrossProduct(ntl_rel, ntr_rel)) for outward normal pointing up
    VxVector topNormal = CrossProduct(ntl_rel, ntr_rel);
    topNormal.Normalize();
    m_UpPlane.Create(topNormal, m_Origin);
}

void VxFrustum::ComputeVertices(VxVector vertices[8]) const {
    // Scale direction vectors by distance and bounds
    VxVector nearDirVec = m_Dir * m_DMin;
    VxVector rightVec = m_Right * m_RBound;
    VxVector upVec = m_Up * m_UBound;

    // Compute near plane vertices relative to origin
    VxVector leftVec = nearDirVec - rightVec;
    VxVector rightVec2 = nearDirVec + rightVec;

    vertices[0] = leftVec - upVec;   // Near-Bottom-Left
    vertices[1] = leftVec + upVec;   // Near-Top-Left
    vertices[2] = rightVec2 + upVec; // Near-Top-Right
    vertices[3] = rightVec2 - upVec; // Near-Bottom-Right

    // Compute far vertices and adjust near vertices to world space
    // For each near vertex:
    //   far vertex = origin + (near_relative * m_DRatio)
    //   near vertex = origin + near_relative
    for (int i = 0; i < 4; i++) {
        VxVector nearVec = vertices[i];
        vertices[i + 4] = m_Origin + nearVec * m_DRatio;
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
    const float newRBound = Magnitude(resultVectors[0]);
    const float newUBound = Magnitude(resultVectors[1]);
    const float newDMin = Magnitude(resultVectors[2]);

    m_RBound = newRBound;
    m_UBound = newUBound;
    m_DMax = newDMin * m_DRatio;
    m_DMin = newDMin;

    if (newRBound > 0.0f) {
        resultVectors[0] /= newRBound;
    } else {
        resultVectors[0] = VxVector::axis0();
    }

    if (newUBound > 0.0f) {
        resultVectors[1] /= newUBound;
    } else {
        resultVectors[1] = VxVector::axis0();
    }

    if (newDMin > 0.0f) {
        resultVectors[2] /= newDMin;
    } else {
        resultVectors[2] = VxVector::axis0();
    }

    m_Right = resultVectors[0];
    m_Up = resultVectors[1];
    m_Dir = resultVectors[2];

    // Update the planes with the new transformed vectors
    Update();
}
