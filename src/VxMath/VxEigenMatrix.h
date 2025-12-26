#ifndef VXEIGENMATRIX_H
#define VXEIGENMATRIX_H

#include "VxMatrix.h"

// VxEigenMatrix class for computing eigenvalues/eigenvectors
class VxEigenMatrix : public VxMatrix {
public:
    // Compute covariance matrix from 3D points
    XBOOL Covariance(const float *points, XULONG stride, int count) {
        if (!points || count <= 0 || stride < sizeof(float) * 3) {
            return FALSE;
        }

        // Initialize sums
        double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
        double sumXX = 0.0, sumYY = 0.0, sumZZ = 0.0;
        double sumXY = 0.0, sumXZ = 0.0, sumYZ = 0.0;

        // Accumulate sums using double precision for better numerical stability
        for (int i = 0; i < count; i++) {
            const float *pt = (const float *)((const XBYTE *)points + i * stride);
            double x = pt[0];
            double y = pt[1];
            double z = pt[2];

            sumX += x;
            sumY += y;
            sumZ += z;

            sumXX += x * x;
            sumYY += y * y;
            sumZZ += z * z;

            sumXY += x * y;
            sumXZ += x * z;
            sumYZ += y * z;
        }

        // Calculate means
        double invCount = 1.0 / (double)count;
        double meanX = sumX * invCount;
        double meanY = sumY * invCount;
        double meanZ = sumZ * invCount;

        // Calculate covariance matrix entries
        m_Data[0][0] = (float)(sumXX * invCount - meanX * meanX); // Cxx
        m_Data[1][1] = (float)(sumYY * invCount - meanY * meanY); // Cyy
        m_Data[2][2] = (float)(sumZZ * invCount - meanZ * meanZ); // Czz

        // Symmetric off-diagonal elements
        m_Data[0][1] = m_Data[1][0] = (float)(sumXY * invCount - meanX * meanY); // Cxy = Cyx
        m_Data[0][2] = m_Data[2][0] = (float)(sumXZ * invCount - meanX * meanZ); // Cxz = Czx
        m_Data[1][2] = m_Data[2][1] = (float)(sumYZ * invCount - meanY * meanZ); // Cyz = Czy

        return TRUE;
    }

    // Householder tridiagonalization for symmetric 3x3 matrix
    // Stores eigenvectors in the matrix itself
    void Tridiagonal(float *diagonal, float *offDiagonal) {
        // Set diagonal elements
        diagonal[0] = m_Data[0][0];
        diagonal[1] = m_Data[1][1];
        diagonal[2] = m_Data[2][2];

        // For 3x3 matrix, we only need one Householder reflection
        float a01 = m_Data[0][1];
        float a02 = m_Data[0][2];
        float a12 = m_Data[1][2];

        // Check if matrix is already tridiagonal
        if (fabsf(a02) <= EPSILON) {
            // Already tridiagonal
            offDiagonal[0] = a01;
            offDiagonal[1] = a12;
            offDiagonal[2] = 0.0f;

            // Set eigenvectors to identity
            m_Data[0][0] = 1.0f; m_Data[0][1] = 0.0f; m_Data[0][2] = 0.0f;
            m_Data[1][0] = 0.0f; m_Data[1][1] = 1.0f; m_Data[1][2] = 0.0f;
            m_Data[2][0] = 0.0f; m_Data[2][1] = 0.0f; m_Data[2][2] = 1.0f;
        } else {
            // Need Householder reflection
            float length = sqrtf(a01 * a01 + a02 * a02);

            if (length > EPSILON) {
                // Compute Householder vector
                float invLength = 1.0f / length;
                float u1 = a01 * invLength;
                float u2 = a02 * invLength;

                // Apply Householder transformation
                float q11 = diagonal[1];
                float q12 = a12;
                float q22 = diagonal[2];

                float tmp = u1 * q12 + u2 * q22;

                diagonal[1] = q11 - 2.0f * u1 * (u1 * q11 + u2 * q12);
                diagonal[2] = q22 - 2.0f * u2 * (u1 * q12 + u2 * q22);

                offDiagonal[0] = length;
                offDiagonal[1] = q12 - 2.0f * u2 * tmp;
                offDiagonal[2] = 0.0f;

                // Update eigenvector matrix
                m_Data[0][0] = 1.0f; m_Data[0][1] = 0.0f; m_Data[0][2] = 0.0f;
                m_Data[1][0] = 0.0f; m_Data[1][1] = u1;   m_Data[1][2] = u2;
                m_Data[2][0] = 0.0f; m_Data[2][1] = u2;   m_Data[2][2] = -u1;
            }
        }
    }

    // QL Algorithm with implicit shifts for eigenvalue decomposition
    // Returns TRUE on success, FALSE on convergence failure
    XBOOL QLAlgorithm(float *diagonal, float *offDiagonal) {
        const int MAX_ITERATIONS = 32;

        // For each diagonal element
        for (int i = 0; i < 2; i++) { // Only need to process first 2 elements for 3x3
            int iterations = 0;

            while (iterations < MAX_ITERATIONS) {
                // Check for convergence
                int m = i;
                while (m < 2) {
                    float sum = fabsf(diagonal[m]) + fabsf(diagonal[m + 1]);
                    if (fabsf(offDiagonal[m]) <= EPSILON * sum) {
                        break;
                    }
                    m++;
                }

                // If converged, move to next eigenvalue
                if (m == i) {
                    break;
                }

                // Compute implicit shift (Wilkinson shift)
                float g = (diagonal[i + 1] - diagonal[i]) / (2.0f * offDiagonal[i]);
                float r = sqrtf(g * g + 1.0f);

                // Choose sign to avoid cancellation
                if (g >= 0.0f) {
                    g = diagonal[m] - diagonal[i] + offDiagonal[i] / (g + r);
                } else {
                    g = diagonal[m] - diagonal[i] + offDiagonal[i] / (g - r);
                }

                // Apply plane rotations
                float c = 1.0f;
                float s = 1.0f;
                float p = 0.0f;

                for (int j = m - 1; j >= i; j--) {
                    float f = s * offDiagonal[j];
                    float b = c * offDiagonal[j];

                    // Compute rotation parameters
                    if (fabsf(f) >= fabsf(g)) {
                        c = g / f;
                        r = sqrtf(c * c + 1.0f);
                        offDiagonal[j + 1] = f * r;
                        s = 1.0f / r;
                        c *= s;
                    } else {
                        s = f / g;
                        r = sqrtf(s * s + 1.0f);
                        offDiagonal[j + 1] = g * r;
                        c = 1.0f / r;
                        s *= c;
                    }

                    // Update diagonal and off-diagonal elements
                    g = diagonal[j + 1] - p;
                    r = (diagonal[j] - g) * s + 2.0f * c * b;
                    p = s * r;
                    diagonal[j + 1] = g + p;
                    g = c * r - b;

                    // Update eigenvectors
                    for (int k = 0; k < 3; k++) {
                        float t = m_Data[k][j + 1];
                        float u = m_Data[k][j];
                        m_Data[k][j + 1] = s * u + c * t;
                        m_Data[k][j] = c * u - s * t;
                    }
                }

                diagonal[i] -= p;
                offDiagonal[i] = g;
                offDiagonal[m] = 0.0f;

                iterations++;
            }

            // Check for convergence failure
            if (iterations >= MAX_ITERATIONS) {
                return FALSE;
            }
        }

        return TRUE;
    }

    // Sort eigenvalues and eigenvectors in descending order
    void SortEigenvalues(float *eigenvalues) {
        for (int i = 0; i < 2; i++) {
            int maxIndex = i;

            // Find the largest remaining eigenvalue
            for (int j = i + 1; j < 3; j++) {
                if (eigenvalues[j] > eigenvalues[maxIndex]) {
                    maxIndex = j;
                }
            }

            // Swap eigenvalues and eigenvectors
            if (maxIndex != i) {
                // Swap eigenvalues
                float temp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[maxIndex];
                eigenvalues[maxIndex] = temp;

                // Swap eigenvectors (columns)
                for (int k = 0; k < 3; k++) {
                    temp = m_Data[k][i];
                    m_Data[k][i] = m_Data[k][maxIndex];
                    m_Data[k][maxIndex] = temp;
                }
            }
        }
    }

    // Main method: Compute eigenvalues and eigenvectors of the 3x3 matrix
    XBOOL EigenDecomposition(float *eigenvalues, float eigenvectors[3][3]) {
        if (!eigenvalues || !eigenvectors) {
            return FALSE;
        }

        float offDiagonal[3];

        // Step 1: Reduce to tridiagonal form
        Tridiagonal(eigenvalues, offDiagonal);

        // Step 2: Apply QL algorithm
        if (!QLAlgorithm(eigenvalues, offDiagonal)) {
            return FALSE;
        }

        // Step 3: Sort eigenvalues in descending order
        SortEigenvalues(eigenvalues);

        // Copy eigenvectors
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                eigenvectors[i][j] = m_Data[i][j];
            }
        }

        return TRUE;
    }

    // Simplified method based on decompiled code
    void EigenStuff3() {
        float diagonal[3];
        float offDiagonal[3];

        // Reduce to tridiagonal form and compute eigenvectors
        Tridiagonal(diagonal, offDiagonal);

        // Apply QL algorithm to find eigenvalues
        QLAlgorithm(diagonal, offDiagonal);
    }
};

#endif // VXEIGENMATRIX_H