#include "VxQuaternion.h"
#include "VxMatrix.h"

// Global lookup tables used by VxQuaternionSnuggle
static VxQuaternion g_SnuggleLookup[8] = {
    {0.0f, 0.0f, 0.0f, 1.0f}, // Identity
    {0.5f, 0.5f, 0.5f, 0.5f}, // 120-degree rotations
    {0.5f, -0.5f, 0.5f, 0.5f},
    {-0.5f, 0.5f, 0.5f, 0.5f},
    {0.5f, 0.5f, -0.5f, 0.5f},
    {-0.5f, -0.5f, 0.5f, 0.5f},
    {0.5f, -0.5f, -0.5f, 0.5f},
    {1.0f, 0.0f, 0.0f, 0.0f} // 180-degree rotation
};

VxQuaternion Vx3DQuaternionSnuggle(VxQuaternion *Quat, VxVector *Scale) {
    // It handles "snuggling" quaternions to remove scale artifacts
    if (!Quat || !Scale)
        return VxQuaternion(0, 0, 0, 1);

    float x = Scale->x;
    float y = Scale->y;
    float z = Scale->z;

    // Check if scales are nearly equal
    if (x == y && y == z) {
        // Uniform scale - return conjugate
        return Vx3DQuaternionConjugate(*Quat);
    }

    // Determine which scale component is different
    int caseIndex = 0;
    if (x == z) caseIndex = 3;      // y is different
    else if (y == z) caseIndex = 1; // x is different
    else if (x == y) caseIndex = 2; // z is different

    if (caseIndex == 0) {
        // All different - complex case
        VxQuaternion tempQuat = *Quat;

        // Find largest scale component
        float maxScale = XMax(x, XMax(y, z));
        float scales[4] = {tempQuat.x, tempQuat.y, tempQuat.z, tempQuat.w};

        // This involves checking signs and magnitudes of components
        bool signs[4];
        for (int i = 0; i < 4; i++) {
            signs[i] = scales[i] < 0.0f;
            if (signs[i]) scales[i] = -scales[i];
        }

        // Find dominant components and apply appropriate transformation
        float avgScale = (x + y + z + scales[3]) * 0.5f;
        float threshold = sqrtf((scales[0] + scales[1]) * 0.70710677f);

        if (avgScale < threshold) {
            // Use half-angle rotations
            VxQuaternion result(0, 0, 0, 1);
            for (int i = 0; i < 3; i++) {
                result[i] = signs[i] ? -0.5f : 0.5f;
            }

            // Swap scale components based on sign pattern
            bool parity = signs[0] ^ signs[1] ^ signs[2] ^ signs[3];
            if (parity) {
                float temp = x;
                x = y;
                y = z;
                z = temp;
            } else {
                float temp = y;
                y = x;
                x = z;
                z = temp;
            }

            Scale->x = x;
            Scale->y = y;
            Scale->z = z;

            return VxQuaternion(-result.x, -result.y, -result.z, result.w);
        } else {
            // Use simpler transformations
            int maxIndex = 0;
            if (scales[1] > scales[maxIndex]) maxIndex = 1;
            if (scales[2] > scales[maxIndex]) maxIndex = 2;
            if (scales[3] > scales[maxIndex]) maxIndex = 3;

            VxQuaternion result(0, 0, 0, 1);
            result[maxIndex] = signs[maxIndex] ? -1.0f : 1.0f;

            return VxQuaternion(-result.x, -result.y, -result.z, result.w);
        }
    } else {
        // Two components are equal - simpler case
        *Quat = Vx3DQuaternionConjugate(*Quat);

        // Apply appropriate lookup table transformation
        VxQuaternion transform = g_SnuggleLookup[caseIndex];
        VxQuaternion result = Vx3DQuaternionMultiply(*Quat, transform);

        // Swap scale components as needed
        if (caseIndex == 1) {
            float temp = x;
            x = z;
            z = y;
            y = temp;
        } else if (caseIndex == 2) {
            float temp = z;
            z = x;
            x = y;
            y = temp;
        }

        Scale->x = x;
        Scale->y = y;
        Scale->z = z;

        return VxQuaternion(-result.x, -result.y, -result.z, result.w);
    }
}

VxQuaternion Vx3DQuaternionFromMatrix(const VxMatrix &Mat) {
    VxQuaternion quat;
    quat.FromMatrix(Mat, TRUE, TRUE);
    return quat;
}

VxQuaternion Vx3DQuaternionConjugate(const VxQuaternion &Quat) {
    return VxQuaternion(-Quat.x, -Quat.y, -Quat.z, Quat.w);
}

VxQuaternion Vx3DQuaternionMultiply(const VxQuaternion &QuatL, const VxQuaternion &QuatR) {
    return VxQuaternion(
        QuatL.w * QuatR.x + QuatL.x * QuatR.w + QuatL.y * QuatR.z - QuatL.z * QuatR.y, // x
        QuatL.w * QuatR.y - QuatL.x * QuatR.z + QuatL.y * QuatR.w + QuatL.z * QuatR.x, // y
        QuatL.w * QuatR.z + QuatL.x * QuatR.y - QuatL.y * QuatR.x + QuatL.z * QuatR.w, // z
        QuatL.w * QuatR.w - QuatL.x * QuatR.x - QuatL.y * QuatR.y - QuatL.z * QuatR.z  // w
    );
}

VxQuaternion Vx3DQuaternionDivide(const VxQuaternion &P, const VxQuaternion &Q) {
    float newW = P.w * Q.w + P.x * Q.x + P.y * Q.y + P.z * Q.z;
    float newZ = P.z * Q.w - P.w * Q.z - P.y * Q.x + P.x * Q.y;
    float newY = P.y * Q.w - P.w * Q.y - P.x * Q.z + P.z * Q.x;
    float newX = P.w * Q.x - Q.x * P.w - P.z * Q.y + P.y * Q.z;

    return VxQuaternion(newX, newY, newZ, newW);
}

VxQuaternion Slerp(float t, const VxQuaternion &Quat1, const VxQuaternion &Quat2) {
    float cosOmega = Quat1.x * Quat2.x + Quat1.y * Quat2.y + Quat1.z * Quat2.z + Quat1.w * Quat2.w;

    float k0, k1;

    if (cosOmega >= 0.0f) {
        // Positive dot product
        float oneMinusCos = 1.0f - cosOmega;
        if (oneMinusCos < 0.01f) {
            // Linear interpolation for close quaternions
            k0 = 1.0f - t;
            k1 = t;
        } else {
            // Spherical interpolation
            float omega = acosf(cosOmega);
            float invSinOmega = 1.0f / sinf(omega);
            k0 = sinf((1.0f - t) * omega) * invSinOmega;
            k1 = sinf(t * omega) * invSinOmega;
        }
    } else {
        // Negative dot product - negate the interpolation parameter
        float oneMinusCosNeg = 1.0f - (-cosOmega);
        if (oneMinusCosNeg < 0.01f) {
            // Linear interpolation
            k0 = 1.0f - t;
            k1 = -t;
        } else {
            // Spherical interpolation with negated parameter
            float omega = acosf(-cosOmega);
            float invSinOmega = 1.0f / sinf(omega);
            k0 = sinf((1.0f - t) * omega) * invSinOmega;
            k1 = -sinf(t * omega) * invSinOmega;
        }
    }

    // Interpolate
    return VxQuaternion(
        k0 * Quat1.x + k1 * Quat2.x,
        k0 * Quat1.y + k1 * Quat2.y,
        k0 * Quat1.z + k1 * Quat2.z,
        k0 * Quat1.w + k1 * Quat2.w
    );
}

VxQuaternion Squad(float t, const VxQuaternion &Quat1, const VxQuaternion &Quat1Out, const VxQuaternion &Quat2In,
                   const VxQuaternion &Quat2) {
    VxQuaternion slerpA = Slerp(t, Quat1Out, Quat2In);
    VxQuaternion slerpB = Slerp(t, Quat1, Quat2);

    // Quadratic blend factor: 2*t*(1-t)
    float blendFactor = 2.0f * t * (1.0f - t);
    return Slerp(blendFactor, slerpB, slerpA);
}

VxQuaternion LnDif(const VxQuaternion &P, const VxQuaternion &Q) {
    VxQuaternion div = Vx3DQuaternionDivide(Q, P);
    return Ln(div);
}

VxQuaternion Ln(const VxQuaternion &Quat) {
    float magnitude = sqrtf(Quat.x * Quat.x + Quat.y * Quat.y + Quat.z * Quat.z);

    float scale;
    if (magnitude == 0.0f) {
        scale = 0.0f;
    } else {
        scale = atan2f(magnitude, Quat.w) / magnitude;
    }

    return VxQuaternion(
        scale * Quat.x,
        scale * Quat.y,
        scale * Quat.z,
        0.0f
    );
}

VxQuaternion Exp(const VxQuaternion &Quat) {
    float magnitude = sqrtf(Quat.x * Quat.x + Quat.y * Quat.y + Quat.z * Quat.z);

    float scale;
    if (magnitude < EPSILON) {
        scale = 1.0f;
    } else {
        scale = sinf(magnitude) / magnitude;
    }

    return VxQuaternion(scale * Quat.x, scale * Quat.y, scale * Quat.z, cosf(magnitude));
}

void VxQuaternion::FromMatrix(const VxMatrix &Mat, XBOOL MatIsUnit, XBOOL RestoreMat) {
    // Store original matrix values if we need to restore and matrix is not unit
    float original[9];
    if (!MatIsUnit && RestoreMat) {
        original[0] = Mat[0][0];
        original[1] = Mat[0][1];
        original[2] = Mat[0][2];
        original[3] = Mat[1][0];
        original[4] = Mat[1][1];
        original[5] = Mat[1][2];
        original[6] = Mat[2][0];
        original[7] = Mat[2][1];
        original[8] = Mat[2][2];
    }

    if (!MatIsUnit) {
        // Normalize the first two rows and compute the third row as cross product
        VxMatrix &nonConstMat = const_cast<VxMatrix &>(Mat);

        VxVector row0(Mat[0][0], Mat[0][1], Mat[0][2]);
        VxVector row1(Mat[1][0], Mat[1][1], Mat[1][2]);

        row0.Normalize();
        row1.Normalize();

        nonConstMat[0][0] = row0.x;
        nonConstMat[0][1] = row0.y;
        nonConstMat[0][2] = row0.z;
        nonConstMat[1][0] = row1.x;
        nonConstMat[1][1] = row1.y;
        nonConstMat[1][2] = row1.z;

        // Compute third row as cross product of first two normalized rows
        VxVector row2 = CrossProduct(row0, row1);
        nonConstMat[2][0] = row2.x;
        nonConstMat[2][1] = row2.y;
        nonConstMat[2][2] = row2.z;
    }

    // Use Shepperd's method for quaternion extraction
    float trace = Mat[0][0] + Mat[1][1] + Mat[2][2];

    if (trace > 0.0f) {
        // w is the largest component
        float s = sqrtf(trace + 1.0f);
        w = s * 0.5f;
        s = 0.5f / s;
        x = (Mat[2][1] - Mat[1][2]) * s;
        y = (Mat[0][2] - Mat[2][0]) * s;
        z = (Mat[1][0] - Mat[0][1]) * s;
    } else {
        // Find the largest diagonal element
        int i = 0;
        if (Mat[1][1] > Mat[0][0]) i = 1;
        if (Mat[2][2] > Mat[i][i]) i = 2;

        static const int next[3] = {1, 2, 0};
        int j = next[i];
        int k = next[j];

        float s = sqrtf(Mat[i][i] - Mat[j][j] - Mat[k][k] + 1.0f);

        float *q[4] = {&x, &y, &z, &w};
        *q[i] = s * 0.5f;

        if (s > EPSILON) {
            s = 0.5f / s;
            *q[3] = (Mat[k][j] - Mat[j][k]) * s; // w component
            *q[j] = (Mat[j][i] + Mat[i][j]) * s;
            *q[k] = (Mat[k][i] + Mat[i][k]) * s;
        } else {
            // Handle degenerate case
            *q[3] = 1.0f; // w = 1
            *q[j] = 0.0f;
            *q[k] = 0.0f;
        }
    }

    // Restore original matrix if requested
    if (!MatIsUnit && RestoreMat) {
        VxMatrix &nonConstMat = const_cast<VxMatrix &>(Mat);
        nonConstMat[0][0] = original[0];
        nonConstMat[0][1] = original[1];
        nonConstMat[0][2] = original[2];
        nonConstMat[1][0] = original[3];
        nonConstMat[1][1] = original[4];
        nonConstMat[1][2] = original[5];
        nonConstMat[2][0] = original[6];
        nonConstMat[2][1] = original[7];
        nonConstMat[2][2] = original[8];
    }
}

void VxQuaternion::ToMatrix(VxMatrix &Mat) const {
    // Direct computation without norm check - matches decompiled code
    float norm = x * x + y * y + z * z + w * w;
    if (norm < EPSILON) {
        Mat.SetIdentity();
        return;
    }

    float s = 2.0f / norm;

    float xs = x * s;
    float ys = y * s;
    float zs = z * s;

    float wx = w * xs;
    float wy = w * ys;
    float wz = w * zs;

    float xx = x * xs;
    float xy = x * ys;
    float xz = x * zs;

    float yy = y * ys;
    float yz = y * zs;
    float zz = z * zs;

    Mat[0][0] = 1.0f - (yy + zz);
    Mat[0][1] = xy - wz;
    Mat[0][2] = xz + wy;
    Mat[0][3] = 0.0f;

    Mat[1][0] = xy + wz;
    Mat[1][1] = 1.0f - (xx + zz);
    Mat[1][2] = yz - wx;
    Mat[1][3] = 0.0f;

    Mat[2][0] = xz - wy;
    Mat[2][1] = yz + wx;
    Mat[2][2] = 1.0f - (xx + yy);
    Mat[2][3] = 0.0f;

    Mat[3][0] = 0.0f;
    Mat[3][1] = 0.0f;
    Mat[3][2] = 0.0f;
    Mat[3][3] = 1.0f;
}

void VxQuaternion::Multiply(const VxQuaternion &Quat) {
    // Store intermediate values as in decompiled code
    float newX = w * Quat.x + x * Quat.w + y * Quat.z - z * Quat.y;
    float newY = w * Quat.y - x * Quat.z + y * Quat.w + z * Quat.x;
    float newZ = w * Quat.z + x * Quat.y - y * Quat.x + z * Quat.w;
    float newW = w * Quat.w - x * Quat.x - y * Quat.y - z * Quat.z;

    x = newX;
    y = newY;
    z = newZ;
    w = newW;
}

void VxQuaternion::FromRotation(const VxVector &Vector, float Angle) {
    VxMatrix Mat;
    Vx3DMatrixFromRotation(Mat, Vector, Angle);
    FromMatrix(Mat, TRUE, TRUE);
}

void VxQuaternion::FromEulerAngles(float eax, float eay, float eaz) {
    VxMatrix Mat;
    Vx3DMatrixFromEulerAngles(Mat, eax, eay, eaz);
    FromMatrix(Mat, TRUE, TRUE);
}

void VxQuaternion::ToEulerAngles(float *eax, float *eay, float *eaz) const {
    VxMatrix Mat;
    ToMatrix(Mat);
    Vx3DMatrixToEulerAngles(Mat, eax, eay, eaz);
}

void VxQuaternion::Normalize() {
    float norm = sqrtf(x * x + y * y + z * z + w * w);
    if (norm == 0.0f) {
        // Default to identity quaternion if exactly zero
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
        w = 1.0f;
    } else {
        float invNorm = 1.0f / norm;
        x *= invNorm;
        y *= invNorm;
        z *= invNorm;
        w *= invNorm;
    }
}
