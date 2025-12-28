#ifndef VXEIGENMATRIX_H
#define VXEIGENMATRIX_H

#include "VxMatrix.h"

// VxEigenMatrix class for computing eigenvalues/eigenvectors
class VxEigenMatrix : public VxMatrix {
public:
    VxEigenMatrix() = default;

    void Covariance(const float *points, XULONG stride, int count) {
        VxVector sum(0.0f);

        float sumXX = 0.0f;
        float sumYY = 0.0f;
        float sumZZ = 0.0f;

        float sumXY = 0.0f;
        float sumXZ = 0.0f;
        float sumYZ = 0.0f;

        const unsigned char *cursor = (const unsigned char *) points;
        for (int i = 0; i < count; ++i) {
            const VxVector &pt = *(const VxVector *) cursor;
            cursor += stride;

            sum += pt;

            sumXX += pt.x * pt.x;
            sumYY += pt.y * pt.y;
            sumZZ += pt.z * pt.z;

            sumXY += pt.x * pt.y;
            sumXZ += pt.x * pt.z;
            sumYZ += pt.y * pt.z;
        }

        const float invCount = 1.0f / (float) count;
        const VxVector mean = sum * invCount;

        const float avgXX = sumXX * invCount;
        const float avgYY = sumYY * invCount;
        const float avgZZ = sumZZ * invCount;

        const float avgXY = sumXY * invCount;
        const float avgXZ = sumXZ * invCount;
        const float avgYZ = sumYZ * invCount;

        // Symmetric 3x3 (upper-left)
        const float cxy = avgXY - mean.y * mean.x;
        const float cxz = avgXZ - mean.z * mean.x;
        const float cyz = avgYZ - mean.z * mean.y;

        VxVector4 &row0 = (*this)[0];
        VxVector4 &row1 = (*this)[1];
        VxVector4 &row2 = (*this)[2];

        row0[0] = avgXX - mean.x * mean.x;
        row1[1] = avgYY - mean.y * mean.y;
        row2[2] = avgZZ - mean.z * mean.z;

        row0[1] = cxy;
        row1[0] = cxy;
        row0[2] = cxz;
        row2[0] = cxz;
        row1[2] = cyz;
        row2[1] = cyz;

        row0[3] = 0.0f;
        row1[3] = 0.0f;
        row2[3] = 0.0f;
    }

    void Tridiagonal(float *const diagonal, float *const offDiagonal) {
        const VxVector4 &r0 = (*this)[0];
        const VxVector4 &r1 = (*this)[1];
        const VxVector4 &r2 = (*this)[2];

        const float a01 = r0[1];
        const float a02 = r0[2];
        const float a11 = r1[1];
        const float a12 = r1[2];
        const float a22 = r2[2];

        diagonal[0] = m_Data[0][0];
        offDiagonal[2] = 0.0f;

        if (fabs(a02) < EPSILON) {
            diagonal[1] = a11;
            diagonal[2] = a22;
            offDiagonal[0] = a01;
            offDiagonal[1] = a12;

            // Identity
            SetIdentity();
            return;
        }

        const float length = sqrtf(a02 * a02 + a01 * a01);
        const float invLength = 1.0f / length;

        const float u1 = a01 * invLength;
        const float u2 = a02 * invLength;

        // v14 in the binary
        const float tmp = (a22 - a11) * u2 + 2.0f * a12 * u1;

        diagonal[1] = a11 + tmp * u2;
        diagonal[2] = a22 - tmp * u2;
        offDiagonal[0] = length;
        offDiagonal[1] = a12 - tmp * u1;

        // Householder eigenvector matrix as built by the binary
        VxVector4 &row0 = (*this)[0];
        VxVector4 &row1 = (*this)[1];
        VxVector4 &row2 = (*this)[2];

        row0.Set(1.0f, 0.0f, 0.0f, 0.0f);
        row1.Set(0.0f, u1, u2, 0.0f);
        row2.Set(0.0f, u2, -u1, 0.0f);
    }

    XBOOL QLAlgorithm(float *const diagonal, float *const offDiagonal) {
        // Only i=0..1 are processed for 3x3
        for (int i = 0; i <= 1; ++i) {
            int iter = 0;
            for (; iter < 32; ++iter) {
                int m = i;
                while (m <= 1) {
                    const double s = fabs(diagonal[m]) + fabs(diagonal[m + 1]);
                    if (fabs(offDiagonal[m]) + s == s) {
                        break;
                    }
                    ++m;
                }

                if (m == i) {
                    break;
                }

                // Implicit shift
                const float g0 = (diagonal[i + 1] - diagonal[i]) / (offDiagonal[i] + offDiagonal[i]);
                const float r0 = sqrtf(g0 * g0 + 1.0f);
                float g;
                if (g0 >= 0.0f) {
                    g = offDiagonal[i] / (g0 + r0) + diagonal[m] - diagonal[i];
                } else {
                    g = offDiagonal[i] / (g0 - r0) + diagonal[m] - diagonal[i];
                }

                float c = 1.0f;
                float s = 1.0f;
                float p = 0.0f;

                for (int j = m - 1; j >= i; --j) {
                    const float f = c * offDiagonal[j];
                    const float b = s * offDiagonal[j];

                    if (fabs(g) > fabs(f)) {
                        const float t = f / g;
                        const float r = sqrtf(t * t + 1.0f);
                        offDiagonal[j + 1] = r * g;
                        s = 1.0f / r;
                        c = t * s;
                    } else {
                        const float t = g / f;
                        const float r = sqrtf(t * t + 1.0f);
                        offDiagonal[j + 1] = f * r;
                        c = 1.0f / r;
                        s = t * c;
                    }

                    const float g2 = diagonal[j + 1] - p;
                    const float r2 = (diagonal[j] - g2) * c + 2.0f * b * s;
                    p = c * r2;
                    diagonal[j + 1] = g2 + p;
                    g = s * r2 - b;

                    // Update eigenvectors (exact order from the binary)
                    for (int row = 0; row < 3; ++row) {
                        VxVector4 &rowVec = (*this)[row];
                        const float t = rowVec[j + 1];
                        const float u = rowVec[j];
                        rowVec[j + 1] = s * t + c * u;
                        rowVec[j] = s * u - c * t;
                    }
                }

                diagonal[i] -= p;
                offDiagonal[i] = g;
                offDiagonal[m] = 0.0f;
            }

            if (iter == 32) {
                return FALSE;
            }
        }

        return TRUE;
    }

    void EigenStuff3() {
        float diagonal[3];
        float offDiagonal[3];
        Tridiagonal(diagonal, offDiagonal);
        QLAlgorithm(diagonal, offDiagonal);
    }
};

#endif // VXEIGENMATRIX_H
