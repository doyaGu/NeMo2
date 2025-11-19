#include "VxMatrix.h"

static bool gMatrixInitialized = false;
VxMatrix TheIdentityMatrix;

static void InitializeIdentityMatrix() {
    if (!gMatrixInitialized) {
        TheIdentityMatrix.SetIdentity();
        gMatrixInitialized = true;
    }
}

void Vx3DMatrixIdentity(VxMatrix &Mat) {
    InitializeIdentityMatrix();
    Mat = TheIdentityMatrix;
}

const VxMatrix &VxMatrix::Identity() {
    InitializeIdentityMatrix();
    return TheIdentityMatrix;
}

void Vx3DMultiplyMatrixVector(VxVector *ResultVector, const VxMatrix &Mat, const VxVector *Vector) {
    const float vx = Vector->x;
    const float vy = Vector->y;
    const float vz = Vector->z;

    const float *m0 = Mat[0];
    const float *m1 = Mat[1];
    const float *m2 = Mat[2];
    const float *m3 = Mat[3];

    ResultVector->x = vx * m0[0] + vy * m1[0] + vz * m2[0] + m3[0];
    ResultVector->y = vx * m0[1] + vy * m1[1] + vz * m2[1] + m3[1];
    ResultVector->z = vx * m0[2] + vy * m1[2] + vz * m2[2] + m3[2];
}

void Vx3DMultiplyMatrixVectorMany(VxVector *ResultVectors, const VxMatrix &Mat, const VxVector *Vectors, int count, int stride) {
    if (count <= 0) return;

    const float m00 = Mat[0][0], m01 = Mat[0][1], m02 = Mat[0][2];
    const float m10 = Mat[1][0], m11 = Mat[1][1], m12 = Mat[1][2];
    const float m20 = Mat[2][0], m21 = Mat[2][1], m22 = Mat[2][2];
    const float m30 = Mat[3][0], m31 = Mat[3][1], m32 = Mat[3][2];

    const char *srcPtr = reinterpret_cast<const char *>(Vectors);
    char *dstPtr = reinterpret_cast<char *>(ResultVectors);

    // Process in blocks of 4
    int blockCount = count / 4;
    int remainder = count % 4;

    for (int block = 0; block < blockCount; ++block) {
        for (int i = 0; i < 4; ++i) {
            const VxVector *vec = reinterpret_cast<const VxVector *>(srcPtr + (block * 4 + i) * stride);
            VxVector *result = reinterpret_cast<VxVector *>(dstPtr + (block * 4 + i) * stride);

            const float vx = vec->x;
            const float vy = vec->y;
            const float vz = vec->z;

            result->x = vx * m00 + vy * m10 + vz * m20 + m30;
            result->y = vx * m01 + vy * m11 + vz * m21 + m31;
            result->z = vx * m02 + vy * m12 + vz * m22 + m32;
        }
    }

    // Handle remaining vectors
    for (int i = 0; i < remainder; ++i) {
        const VxVector *vec = reinterpret_cast<const VxVector *>(srcPtr + (blockCount * 4 + i) * stride);
        VxVector *result = reinterpret_cast<VxVector *>(dstPtr + (blockCount * 4 + i) * stride);

        const float vx = vec->x;
        const float vy = vec->y;
        const float vz = vec->z;

        result->x = vx * m00 + vy * m10 + vz * m20 + m30;
        result->y = vx * m01 + vy * m11 + vz * m21 + m31;
        result->z = vx * m02 + vy * m12 + vz * m22 + m32;
    }
}

void Vx3DMultiplyMatrixVector4(VxVector4 *ResultVector, const VxMatrix &Mat, const VxVector4 *Vector) {
    const float vx = Vector->x;
    const float vy = Vector->y;
    const float vz = Vector->z;
    const float vw = Vector->w;

    const float *m0 = Mat[0];
    const float *m1 = Mat[1];
    const float *m2 = Mat[2];
    const float *m3 = Mat[3];

    ResultVector->x = vx * m0[0] + vy * m1[0] + vz * m2[0] + vw * m3[0];
    ResultVector->y = vx * m0[1] + vy * m1[1] + vz * m2[1] + vw * m3[1];
    ResultVector->z = vx * m0[2] + vy * m1[2] + vz * m2[2] + vw * m3[2];
    ResultVector->w = vx * m0[3] + vy * m1[3] + vz * m2[3] + vw * m3[3];
}

void Vx3DMultiplyMatrixVector4(VxVector4 *ResultVector, const VxMatrix &Mat, const VxVector *Vector) {
    const float vx = Vector->x;
    const float vy = Vector->y;
    const float vz = Vector->z;

    const float *m0 = Mat[0];
    const float *m1 = Mat[1];
    const float *m2 = Mat[2];
    const float *m3 = Mat[3];

    ResultVector->x = vx * m0[0] + vy * m1[0] + vz * m2[0] + m3[0];
    ResultVector->y = vx * m0[1] + vy * m1[1] + vz * m2[1] + m3[1];
    ResultVector->z = vx * m0[2] + vy * m1[2] + vz * m2[2] + m3[2];
    ResultVector->w = vx * m0[3] + vy * m1[3] + vz * m2[3] + m3[3];
}

void Vx3DRotateVector(VxVector *ResultVector, const VxMatrix &Mat, const VxVector *Vector) {
    const float vx = Vector->x;
    const float vy = Vector->y;
    const float vz = Vector->z;

    // Only access rotation part of matrix (3x3 upper-left)
    const float *m0 = Mat[0];
    const float *m1 = Mat[1];
    const float *m2 = Mat[2];

    ResultVector->x = vx * m0[0] + vy * m1[0] + vz * m2[0];
    ResultVector->y = vx * m0[1] + vy * m1[1] + vz * m2[1];
    ResultVector->z = vx * m0[2] + vy * m1[2] + vz * m2[2];
}

void Vx3DRotateVectorMany(VxVector *ResultVector, const VxMatrix &Mat, const VxVector *Vector, int count, int stride) {
    if (count <= 0) return;

    const float m00 = Mat[0][0], m01 = Mat[0][1], m02 = Mat[0][2];
    const float m10 = Mat[1][0], m11 = Mat[1][1], m12 = Mat[1][2];
    const float m20 = Mat[2][0], m21 = Mat[2][1], m22 = Mat[2][2];

    const char *srcPtr = reinterpret_cast<const char *>(Vector);
    char *dstPtr = reinterpret_cast<char *>(ResultVector);

    for (int i = 0; i < count; i += 4) {
        int remaining = (count - i < 4) ? count - i : 4;

        for (int j = 0; j < remaining; ++j) {
            const VxVector *vec = reinterpret_cast<const VxVector *>(srcPtr + (i + j) * stride);
            VxVector *result = reinterpret_cast<VxVector *>(dstPtr + (i + j) * stride);

            const float vx = vec->x;
            const float vy = vec->y;
            const float vz = vec->z;

            result->x = vx * m00 + vy * m10 + vz * m20;
            result->y = vx * m01 + vy * m11 + vz * m21;
            result->z = vx * m02 + vy * m12 + vz * m22;
        }
    }
}

void Vx3DMultiplyMatrix(VxMatrix &ResultMat, const VxMatrix &MatA, const VxMatrix &MatB) {
    VxMatrix temp;

    for (int i = 0; i < 4; i++) {
        const float bi0 = MatB[i][0];
        const float bi1 = MatB[i][1];
        const float bi2 = MatB[i][2];
        const float bi3 = MatB[i][3];

        for (int j = 0; j < 4; j++) {
            temp[i][j] = MatA[0][j] * bi0 +
                         MatA[1][j] * bi1 +
                         MatA[2][j] * bi2 +
                         MatA[3][j] * bi3;
        }
    }

    // For standard 3D transformations, ensure the bottom row is [0,0,0,1]
    temp[0][3] = 0.0f;
    temp[1][3] = 0.0f;
    temp[2][3] = 0.0f;
    temp[3][3] = 1.0f;

    ResultMat = temp;
}

void Vx3DMultiplyMatrix4(VxMatrix &ResultMat, const VxMatrix &MatA, const VxMatrix &MatB) {
    VxMatrix temp;

    for (int i = 0; i < 4; i++) {
        const float bi0 = MatB[i][0];
        const float bi1 = MatB[i][1];
        const float bi2 = MatB[i][2];
        const float bi3 = MatB[i][3];

        for (int j = 0; j < 4; j++) {
            temp[i][j] = MatA[0][j] * bi0 +
                         MatA[1][j] * bi1 +
                         MatA[2][j] * bi2 +
                         MatA[3][j] * bi3;
        }
    }

    ResultMat = temp;
}

void Vx3DInverseMatrix(VxMatrix &InverseMat, const VxMatrix &Mat) {
    // Extract the 3x3 rotation/scale submatrix
    const float a00 = Mat[0][0], a01 = Mat[0][1], a02 = Mat[0][2];
    const float a10 = Mat[1][0], a11 = Mat[1][1], a12 = Mat[1][2];
    const float a20 = Mat[2][0], a21 = Mat[2][1], a22 = Mat[2][2];

    // Calculate determinant using rule of Sarrus expansion
    const float minor1 = a11 * a22 - a12 * a21;
    const float minor2 = a12 * a20 - a10 * a22;
    const float minor3 = a10 * a21 - a11 * a20;

    const double det = a00 * minor1 + a01 * minor2 + a02 * minor3;

    if (abs(det) < EPSILON) {
        InverseMat.SetIdentity();
        return;
    }

    const double invDet = 1.0 / det;

    // Calculate inverse of the 3x3 submatrix using cofactor method
    InverseMat[0][0] = static_cast<float>((a11 * a22 - a12 * a21) * invDet);
    InverseMat[0][1] = static_cast<float>((a02 * a21 - a01 * a22) * invDet);
    InverseMat[0][2] = static_cast<float>((a01 * a12 - a02 * a11) * invDet);
    InverseMat[0][3] = 0.0f;

    InverseMat[1][0] = static_cast<float>((a12 * a20 - a10 * a22) * invDet);
    InverseMat[1][1] = static_cast<float>((a00 * a22 - a02 * a20) * invDet);
    InverseMat[1][2] = static_cast<float>((a02 * a10 - a00 * a12) * invDet);
    InverseMat[1][3] = 0.0f;

    InverseMat[2][0] = static_cast<float>((a10 * a21 - a11 * a20) * invDet);
    InverseMat[2][1] = static_cast<float>((a01 * a20 - a00 * a21) * invDet);
    InverseMat[2][2] = static_cast<float>((a00 * a11 - a01 * a10) * invDet);
    InverseMat[2][3] = 0.0f;

    // Calculate the translation part: -R^T * t
    const float tx = Mat[3][0];
    const float ty = Mat[3][1];
    const float tz = Mat[3][2];

    InverseMat[3][0] = -(InverseMat[0][0] * tx + InverseMat[1][0] * ty + InverseMat[2][0] * tz);
    InverseMat[3][1] = -(InverseMat[0][1] * tx + InverseMat[1][1] * ty + InverseMat[2][1] * tz);
    InverseMat[3][2] = -(InverseMat[0][2] * tx + InverseMat[1][2] * ty + InverseMat[2][2] * tz);
    InverseMat[3][3] = 1.0f;
}

float Vx3DMatrixDeterminant(const VxMatrix &Mat) {
    // Use only 3x3 upper-left submatrix for 3D transformations
    const float a00 = Mat[0][0], a01 = Mat[0][1], a02 = Mat[0][2];
    const float a10 = Mat[1][0], a11 = Mat[1][1], a12 = Mat[1][2];
    const float a20 = Mat[2][0], a21 = Mat[2][1], a22 = Mat[2][2];

    const float minor1 = a11 * a22 - a12 * a21;
    const float minor2 = a12 * a20 - a10 * a22;
    const float minor3 = a10 * a21 - a11 * a20;

    return a00 * minor1 + a01 * minor2 + a02 * minor3;
}

void Vx3DMatrixFromRotation(VxMatrix &ResultMat, const VxVector &Vector, float Angle) {
    // Fast path for zero rotation
    if (abs(Angle) < EPSILON) {
        ResultMat.SetIdentity();
        return;
    }

    // Pre-calculate trigonometric values
    const float c = cosf(Angle);
    const float s = sinf(Angle);
    const float t = 1.0f - c;

    // Normalize axis vector with fast inverse square root approximation fallback
    const float lenSq = Vector.x * Vector.x + Vector.y * Vector.y + Vector.z * Vector.z;
    float x, y, z;

    if (lenSq > EPSILON) {
        const float invLen = 1.0f / sqrtf(lenSq);
        x = Vector.x * invLen;
        y = Vector.y * invLen;
        z = Vector.z * invLen;
    } else {
        // Default to Z-axis
        x = 0.0f;
        y = 0.0f;
        z = 1.0f;
    }

    // Pre-calculate common terms for Rodrigues' formula
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float xs = x * s;
    const float ys = y * s;
    const float zs = z * s;
    const float xyt = xy * t;
    const float xzt = xz * t;
    const float yzt = yz * t;

    ResultMat[0][0] = xx * t + c;
    ResultMat[0][1] = xyt + zs;
    ResultMat[0][2] = xzt - ys;
    ResultMat[0][3] = 0.0f;

    ResultMat[1][0] = xyt - zs;
    ResultMat[1][1] = yy * t + c;
    ResultMat[1][2] = yzt + xs;
    ResultMat[1][3] = 0.0f;

    ResultMat[2][0] = xzt + ys;
    ResultMat[2][1] = yzt - xs;
    ResultMat[2][2] = zz * t + c;
    ResultMat[2][3] = 0.0f;

    ResultMat[3][0] = 0.0f;
    ResultMat[3][1] = 0.0f;
    ResultMat[3][2] = 0.0f;
    ResultMat[3][3] = 1.0f;
}

void Vx3DMatrixFromRotationAndOrigin(VxMatrix &ResultMat, const VxVector &Vector, const VxVector &Origin, float Angle) {
    Vx3DMatrixFromRotation(ResultMat, Vector, Angle);

    // Translation calculation: T = origin + R * (-origin)
    const float negOx = -Origin.x;
    const float negOy = -Origin.y;
    const float negOz = -Origin.z;

    // Apply rotation to negative origin vector
    const float *r0 = ResultMat[0];
    const float *r1 = ResultMat[1];
    const float *r2 = ResultMat[2];

    const float rotNegOx = negOx * r0[0] + negOy * r1[0] + negOz * r2[0];
    const float rotNegOy = negOx * r0[1] + negOy * r1[1] + negOz * r2[1];
    const float rotNegOz = negOx * r0[2] + negOy * r1[2] + negOz * r2[2];

    // Set final translation
    ResultMat[3][0] = Origin.x + rotNegOx;
    ResultMat[3][1] = Origin.y + rotNegOy;
    ResultMat[3][2] = Origin.z + rotNegOz;
}

void Vx3DMatrixFromEulerAngles(VxMatrix &Mat, float eax, float eay, float eaz) {
    // Calculate trigonometric values
    float cx, sx, cy, sy, cz, sz;
    if (abs(eax) <= EPSILON) {
        cx = 1.0f;
        sx = eax;
    } else {
        cx = cosf(eax);
        sx = sinf(eax);
    }
    if (abs(eay) <= EPSILON) {
        cy = 1.0f;
        sy = eay;
    } else {
        cy = cosf(eay);
        sy = sinf(eay);
    }
    if (abs(eaz) <= EPSILON) {
        cz = 1.0f;
        sz = eaz;
    } else {
        cz = cosf(eaz);
        sz = sinf(eaz);
    }

    // Pre-calculate common terms for ZYX rotation order
    const float sxsy = sx * sy;
    const float cxsy = cx * sy;
    const float cycz = cy * cz;
    const float cysz = cy * sz;

    // Build matrix
    Mat[0][0] = cycz;
    Mat[0][1] = cysz;
    Mat[0][2] = -sy;
    Mat[0][3] = 0.0f;

    Mat[1][0] = sxsy * cz - cx * sz;
    Mat[1][1] = sxsy * sz + cx * cz;
    Mat[1][2] = sx * cy;
    Mat[1][3] = 0.0f;

    Mat[2][0] = cxsy * cz + sx * sz;
    Mat[2][1] = cxsy * sz - sx * cz;
    Mat[2][2] = cx * cy;
    Mat[2][3] = 0.0f;

    Mat[3][0] = 0.0f;
    Mat[3][1] = 0.0f;
    Mat[3][2] = 0.0f;
    Mat[3][3] = 1.0f;
}

void Vx3DMatrixToEulerAngles(const VxMatrix &Mat, float *eax, float *eay, float *eaz) {
    // Calculate magnitude for gimbal lock detection
    const float m00 = Mat[0][0];
    const float m01 = Mat[0][1];
    const float m02 = Mat[0][2];
    const float m12 = Mat[1][2];
    const float m22 = Mat[2][2];
    const float m21 = Mat[2][1];
    const float m11 = Mat[1][1];

    const float magnitude = sqrtf(m00 * m00 + m01 * m01);

    if (magnitude < EPSILON) {
        // Gimbal lock case
        *eay = atan2f(-m02, magnitude);
        *eax = atan2f(-m21, m11);
        *eaz = 0.0f;
    } else {
        // Normal case
        *eay = atan2f(-m02, magnitude);
        *eax = atan2f(m12, m22);
        *eaz = atan2f(m01, m00);
    }
}

void Vx3DInterpolateMatrix(float step, VxMatrix &Res, const VxMatrix &A, const VxMatrix &B) {
    // Extract components from matrices A and B
    VxQuaternion quatA, quatB;
    VxVector posA, posB, scaleA, scaleB;
    VxQuaternion uRotA, uRotB;

    Vx3DDecomposeMatrixTotal(A, quatA, posA, scaleA, uRotA);
    Vx3DDecomposeMatrixTotal(B, quatB, posB, scaleB, uRotB);

    // Interpolate components
    VxQuaternion quatRes = Slerp(step, quatA, quatB);
    VxVector posRes = Interpolate(step, posA, posB);
    VxVector scaleRes = Interpolate(step, scaleA, scaleB);
    VxQuaternion uRotRes = Slerp(step, uRotA, uRotB);

    // Reconstruct matrix
    Res.SetIdentity();

    // Create and combine transformation matrices
    VxMatrix scaleMat, rotMat, uRotMat;
    scaleMat.SetIdentity();
    scaleMat[0][0] = scaleRes.x;
    scaleMat[1][1] = scaleRes.y;
    scaleMat[2][2] = scaleRes.z;

    quatRes.ToMatrix(rotMat);
    uRotRes.ToMatrix(uRotMat);

    // Combine: R * U * S
    VxMatrix temp;
    Vx3DMultiplyMatrix(temp, uRotMat, scaleMat);
    Vx3DMultiplyMatrix(Res, rotMat, temp);

    // Set translation
    Res[3][0] = posRes.x;
    Res[3][1] = posRes.y;
    Res[3][2] = posRes.z;
}

void Vx3DInterpolateMatrixNoScale(float step, VxMatrix &Res, const VxMatrix &A, const VxMatrix &B) {
    // Extract components from matrices A and B
    VxQuaternion quatA, quatB;
    VxVector posA, posB, scaleA, scaleB;

    Vx3DDecomposeMatrix(A, quatA, posA, scaleA);
    Vx3DDecomposeMatrix(B, quatB, posB, scaleB);

    // Interpolate without scale
    VxQuaternion quatRes = Slerp(step, quatA, quatB);
    VxVector posRes = Interpolate(step, posA, posB);

    // Reconstruct matrix
    quatRes.ToMatrix(Res);

    // Set translation
    Res[3][0] = posRes.x;
    Res[3][1] = posRes.y;
    Res[3][2] = posRes.z;
}

void Vx3DMultiplyMatrixVectorStrided(VxStridedData *Dest, VxStridedData *Src, const VxMatrix &Mat, int count) {
    if (!Dest || !Src || !Dest->Ptr || !Src->Ptr || count <= 0) return;

    const float m00 = Mat[0][0], m01 = Mat[0][1], m02 = Mat[0][2];
    const float m10 = Mat[1][0], m11 = Mat[1][1], m12 = Mat[1][2];
    const float m20 = Mat[2][0], m21 = Mat[2][1], m22 = Mat[2][2];
    const float m30 = Mat[3][0], m31 = Mat[3][1], m32 = Mat[3][2];

    const char *srcPtr = static_cast<const char *>(Src->Ptr);
    char *destPtr = static_cast<char *>(Dest->Ptr);

    for (int i = 0; i < count; ++i) {
        const VxVector *srcVec = reinterpret_cast<const VxVector *>(srcPtr);
        VxVector *destVec = reinterpret_cast<VxVector *>(destPtr);

        const float vx = srcVec->x;
        const float vy = srcVec->y;
        const float vz = srcVec->z;

        destVec->x = vx * m00 + vy * m10 + vz * m20 + m30;
        destVec->y = vx * m01 + vy * m11 + vz * m21 + m31;
        destVec->z = vx * m02 + vy * m12 + vz * m22 + m32;

        srcPtr += Src->Stride;
        destPtr += Dest->Stride;
    }
}

void Vx3DMultiplyMatrixVector4Strided(VxStridedData *Dest, VxStridedData *Src, const VxMatrix &Mat, int count) {
    if (!Dest || !Src || !Dest->Ptr || !Src->Ptr || count <= 0) return;

    const float *m0 = Mat[0];
    const float *m1 = Mat[1];
    const float *m2 = Mat[2];
    const float *m3 = Mat[3];

    const char *srcPtr = static_cast<const char *>(Src->Ptr);
    char *destPtr = static_cast<char *>(Dest->Ptr);

    for (int i = 0; i < count; ++i) {
        const VxVector4 *srcVec = reinterpret_cast<const VxVector4 *>(srcPtr);
        VxVector4 *destVec = reinterpret_cast<VxVector4 *>(destPtr);

        const float vx = srcVec->x;
        const float vy = srcVec->y;
        const float vz = srcVec->z;
        const float vw = srcVec->w;

        destVec->x = vx * m0[0] + vy * m1[0] + vz * m2[0] + vw * m3[0];
        destVec->y = vx * m0[1] + vy * m1[1] + vz * m2[1] + vw * m3[1];
        destVec->z = vx * m0[2] + vy * m1[2] + vz * m2[2] + vw * m3[2];
        destVec->w = vx * m0[3] + vy * m1[3] + vz * m2[3] + vw * m3[3];

        srcPtr += Src->Stride;
        destPtr += Dest->Stride;
    }
}

void Vx3DRotateVectorStrided(VxStridedData *Dest, VxStridedData *Src, const VxMatrix &Mat, int count) {
    if (!Dest || !Src || !Dest->Ptr || !Src->Ptr || count <= 0) return;

    const float m00 = Mat[0][0], m01 = Mat[0][1], m02 = Mat[0][2];
    const float m10 = Mat[1][0], m11 = Mat[1][1], m12 = Mat[1][2];
    const float m20 = Mat[2][0], m21 = Mat[2][1], m22 = Mat[2][2];

    const char *srcPtr = static_cast<const char *>(Src->Ptr);
    char *destPtr = static_cast<char *>(Dest->Ptr);

    for (int i = 0; i < count; ++i) {
        const VxVector *srcVec = reinterpret_cast<const VxVector *>(srcPtr);
        VxVector *destVec = reinterpret_cast<VxVector *>(destPtr);

        const float vx = srcVec->x;
        const float vy = srcVec->y;
        const float vz = srcVec->z;

        destVec->x = vx * m00 + vy * m10 + vz * m20;
        destVec->y = vx * m01 + vy * m11 + vz * m21;
        destVec->z = vx * m02 + vy * m12 + vz * m22;

        srcPtr += Src->Stride;
        destPtr += Dest->Stride;
    }
}

void Vx3DMatrixAdjoint(const VxMatrix &in, VxMatrix &out) {
    const float m00 = in[0][0], m01 = in[0][1], m02 = in[0][2];
    const float m10 = in[1][0], m11 = in[1][1], m12 = in[1][2];
    const float m20 = in[2][0], m21 = in[2][1], m22 = in[2][2];

    out[0][0] = m11 * m22 - m12 * m21;
    out[1][0] = m12 * m20 - m10 * m22;
    out[2][0] = m10 * m21 - m11 * m20;

    out[0][1] = m02 * m21 - m01 * m22;
    out[1][1] = m00 * m22 - m02 * m20;
    out[2][1] = m01 * m20 - m00 * m21;

    out[0][2] = m01 * m12 - m02 * m11;
    out[1][2] = m02 * m10 - m00 * m12;
    out[2][2] = m00 * m11 - m01 * m10;
}

float Vx3DMatrixNorm(const VxMatrix &M, bool isOneNorm) {
    float maxNorm = 0.0f;
    for (int j = 0; j < 3; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < 3; ++i) {
            sum += fabsf(isOneNorm ? M[i][j] : M[j][i]);
        }
        if (sum > maxNorm) {
            maxNorm = sum;
        }
    }
    return maxNorm;
}

float Vx3DMatrixPolarDecomposition(const VxMatrix &M_in, VxMatrix &Q, VxMatrix &S) {
    VxMatrix E;
    Vx3DTransposeMatrix(E, M_in);
    E[0][3] = E[1][3] = E[2][3] = E[3][0] = E[3][1] = E[3][2] = 0.0f;
    E[3][3] = 1.0f;

    float E_one_norm = Vx3DMatrixNorm(E, true);
    float E_inf_norm = Vx3DMatrixNorm(E, false);

    for (int iter = 0; iter < 20; ++iter) {
        VxMatrix E_adj;
        Vx3DMatrixAdjoint(E, E_adj);

        float det_E = DotProduct(VxVector(E[0][0], E[0][1], E[0][2]), VxVector(E_adj[0][0], E_adj[0][1], E_adj[0][2]));

        if (fabsf(det_E) < 1e-12f) break;

        float E_adj_one_norm = Vx3DMatrixNorm(E_adj, true);
        float E_adj_inf_norm = Vx3DMatrixNorm(E_adj, false);

        float gamma = sqrtf(sqrtf((E_adj_one_norm * E_adj_inf_norm) / (E_one_norm * E_inf_norm)) / fabsf(det_E));
        float c1 = 0.5f * gamma;
        float c2 = 0.5f / (gamma * det_E);

        VxMatrix E_next;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                E_next[i][j] = c1 * E[i][j] + c2 * E_adj[j][i];
            }

        VxMatrix E_diff;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                E_diff[i][j] = E_next[i][j] - E[i][j];

        E = E_next;
        if (Vx3DMatrixNorm(E_diff, true) < E_one_norm * 1e-6f) break;

        E_one_norm = Vx3DMatrixNorm(E, true);
        E_inf_norm = Vx3DMatrixNorm(E, false);
    }

    Vx3DTransposeMatrix(Q, E);
    VxMatrix Q_T;
    Vx3DTransposeMatrix(Q_T, Q);
    Vx3DMultiplyMatrix(S, Q_T, M_in);

    // Symmetrize S
    for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j) {
            float val = 0.5f * ((S)[i][j] + (S)[j][i]);
            (S)[i][j] = val;
            (S)[j][i] = val;
        }

    for (int i = 0; i < 3; ++i) {
        Q[i][3] = Q[3][i] = 0.0f;
        S[i][3] = S[3][i] = 0.0f;
    }
    Q[3][3] = 1.0f;
    S[3][3] = 1.0f;

    return Vx3DMatrixDeterminant(Q);
}

VxVector Vx3DMatrixSpectralDecomposition(const VxMatrix &S_in, VxMatrix &U_out) {
    U_out.SetIdentity();

    float d[3] = {S_in[0][0], S_in[1][1], S_in[2][2]};
    float o[3] = {S_in[0][1], S_in[0][2], S_in[1][2]};

    for (int sweep = 0; sweep < 20; ++sweep) {
        float sum_off_diag = fabsf(o[0]) + fabsf(o[1]) + fabsf(o[2]);
        if (sum_off_diag < 1e-9f) break;

        // Iterate through off-diagonal pairs (0,1), (0,2), (1,2)
        const int p_map[3] = {0, 0, 1};
        const int q_map[3] = {1, 2, 2};
        for (int i = 0; i < 3; ++i) {
            int p = p_map[i];
            int q = q_map[i];

            float S_pq = (p == 0 && q == 1) ? o[0] : ((p == 0 && q == 2) ? o[1] : o[2]);
            if (fabsf(S_pq) < 1e-9f) continue;

            float diff = d[q] - d[p];
            float t;
            if (fabsf(diff) + fabsf(S_pq) * 100.0f == fabsf(diff)) {
                t = S_pq / diff;
            } else {
                float theta = diff / (2.0f * S_pq);
                t = 1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f));
                if (theta < 0.0f) t = -t;
            }
            float c = 1.0f / sqrtf(1.0f + t * t);
            float s = t * c;
            float tau = s / (1.0f + c);
            float h = t * S_pq;

            d[p] -= h;
            d[q] += h;

            // Update other off-diagonal elements
            int r = 3 - p - q; // The remaining index
            float S_pr = (p == 0 && r == 1) ? o[0] : ((p == 0 && r == 2) ? o[1] : o[2]);
            if (p > r) S_pr = (r == 0 && p == 1) ? o[0] : ((r == 0 && p == 2) ? o[1] : o[2]);
            float S_qr = (q == 0 && r == 1) ? o[0] : ((q == 0 && r == 2) ? o[1] : o[2]);
            if (q > r) S_qr = (r == 0 && q == 1) ? o[0] : ((r == 0 && q == 2) ? o[1] : o[2]);

            float next_S_pr = S_pr - s * (S_qr + S_pr * tau);
            float next_S_qr = S_qr + s * (S_pr - S_qr * tau);
            if (p == 0 && r == 1) o[0] = next_S_pr;
            else if (p == 0 && r == 2) o[1] = next_S_pr;
            else o[2] = next_S_pr;
            if (q == 0 && r == 1) o[0] = next_S_qr;
            else if (q == 0 && r == 2) o[1] = next_S_qr;
            else o[2] = next_S_qr;
            if (i == 0) o[0] = 0;
            if (i == 1) o[1] = 0;
            if (i == 2) o[2] = 0;


            // Update eigenvector matrix U
            for (int k = 0; k < 3; ++k) {
                float g = U_out[k][p];
                float h_u = U_out[k][q];
                U_out[k][p] = g - s * (h_u + g * tau);
                U_out[k][q] = h_u + s * (g - h_u * tau);
            }
        }
    }

    return VxVector(d[0], d[1], d[2]);
}

void Vx3DTransposeMatrix(VxMatrix &Result, const VxMatrix &A) {
    VxMatrix temp;

    temp[0][0] = A[0][0];
    temp[0][1] = A[1][0];
    temp[0][2] = A[2][0];
    temp[0][3] = A[3][0];
    temp[1][0] = A[0][1];
    temp[1][1] = A[1][1];
    temp[1][2] = A[2][1];
    temp[1][3] = A[3][1];
    temp[2][0] = A[0][2];
    temp[2][1] = A[1][2];
    temp[2][2] = A[2][2];
    temp[2][3] = A[3][2];
    temp[3][0] = A[0][3];
    temp[3][1] = A[1][3];
    temp[3][2] = A[2][3];
    temp[3][3] = A[3][3];

    Result = temp;
}

void Vx3DDecomposeMatrix(const VxMatrix &A, VxQuaternion &Quat, VxVector &Pos, VxVector &Scale) {
    Pos = VxVector(A[3][0], A[3][1], A[3][2]);
    VxMatrix Mat = A;
    Quat.FromMatrix(Mat, FALSE, FALSE);
    Scale.x = Mat[0][0] * A[0][0] + Mat[0][1] * A[0][1] + Mat[0][2] * A[0][2];
    Scale.y = Mat[1][0] * A[1][0] + Mat[1][1] * A[1][1] + Mat[1][2] * A[1][2];
    Scale.z = Mat[2][0] * A[2][0] + Mat[2][1] * A[2][1] + Mat[2][2] * A[2][2];
}

float Vx3DDecomposeMatrixTotal(const VxMatrix &A, VxQuaternion &Quat, VxVector &Pos, VxVector &Scale, VxQuaternion &URot) {
    // Extract position
    Pos = VxVector(A[3][0], A[3][1], A[3][2]);

    VxMatrix Q, S;
    float det = Vx3DMatrixPolarDecomposition(A, Q, S);
    if (det < 0.0f) {
        // Flip Q matrix if determinant is negative
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Q[i][j] = -Q[i][j];
            }
        }
        det = -1.0f;
    } else {
        det = 1.0f;
    }

    Quat = Vx3DQuaternionFromMatrix(Q);

    VxMatrix U;
    Scale = Vx3DMatrixSpectralDecomposition(S, U);
    URot = Vx3DQuaternionFromMatrix(U);

    VxQuaternion snuggleQuat = Vx3DQuaternionSnuggle(&URot, &Scale);
    URot = Vx3DQuaternionMultiply(URot, snuggleQuat);

    return det;
}

float Vx3DDecomposeMatrixTotalPtr(const VxMatrix &A, VxQuaternion *Quat, VxVector *Pos, VxVector *Scale, VxQuaternion *URot) {
    // Extract position
    if (Pos) {
        *Pos = VxVector(A[3][0], A[3][1], A[3][2]);
    }

    if (!Quat && !Scale && !URot) {
        return 1.0f;
    }

    VxMatrix Q, S;
    float det = Vx3DMatrixPolarDecomposition(A, Q, S);

    if (det < 0.0f) {
        // Flip Q matrix if determinant is negative
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Q[i][j] = -Q[i][j];
            }
        }
        det = -1.0f;
    } else {
        det = 1.0f;
    }

    if (Quat) {
        *Quat = Vx3DQuaternionFromMatrix(Q);
    }

    // Handle scale and URot if requested
    if (Scale || URot) {
        VxMatrix U;
        VxVector tempScale = Vx3DMatrixSpectralDecomposition(S, U);
        VxQuaternion tempURot = Vx3DQuaternionFromMatrix(U);

        VxQuaternion snuggleQuat = Vx3DQuaternionSnuggle(&tempURot, &tempScale);

        if (URot) {
            *URot = Vx3DQuaternionMultiply(tempURot, snuggleQuat);
        }

        if (Scale) {
            *Scale = tempScale;
        }
    }

    return det;
}
