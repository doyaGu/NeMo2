#include "VxRay.h"

#include "VxMatrix.h"

void VxRay::Transform(VxRay &dest, const VxMatrix &mat) {
    Vx3DMultiplyMatrixVector(&dest.m_Origin, mat, &m_Origin);
    Vx3DRotateVector(&dest.m_Direction, mat, &m_Direction);
}
