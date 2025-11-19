#include "VxVector.h"
#include "VxMatrix.h"

static constexpr int SIN_TABLE_SIZE = 4096;  // 16KB for sin table
static constexpr float SIN_TABLE_SCALE = SIN_TABLE_SIZE / (2.0f * PI);

static float g_SinTable[SIN_TABLE_SIZE + 1];
static float g_CosTable[SIN_TABLE_SIZE + 1];

static bool g_TablesInitialized = false;

void InitializeTables() {
    if (g_TablesInitialized) return;

    for (int i = 0; i <= SIN_TABLE_SIZE; ++i) {
        const float angle = (2.0f * PI * static_cast<float>(i)) / SIN_TABLE_SIZE;
        g_SinTable[i] = sinf(angle);
        g_CosTable[i] = cosf(angle);
    }

    g_TablesInitialized = true;
}

inline float TableSin(float x) {
    InitializeTables();

    // Range reduction and table lookup
    const float scaled = x * SIN_TABLE_SCALE;
    const int index = static_cast<int>(scaled) & (SIN_TABLE_SIZE - 1);
    const float frac = scaled - static_cast<float>(index);

    // Linear interpolation between table entries
    const float v0 = g_SinTable[index];
    const float v1 = g_SinTable[index + 1];
    return v0 + frac * (v1 - v0);
}

inline float TableCos(float x) {
    InitializeTables();

    const float scaled = x * SIN_TABLE_SCALE;
    const int index = static_cast<int>(scaled) & (SIN_TABLE_SIZE - 1);
    const float frac = scaled - static_cast<float>(index);

    const float v0 = g_CosTable[index];
    const float v1 = g_CosTable[index + 1];
    return v0 + frac * (v1 - v0);
}

int radToAngle(float val) {
    return static_cast<int>(val * (180.0f / PI));
}

float Tsin(int angle) {
    const float angleRad = static_cast<float>(angle) * (PI / 180.0f);
    return TableSin(angleRad);
}

float Tcos(int angle) {
    const float angleRad = static_cast<float>(angle) * (PI / 180.0f);
    return TableCos(angleRad);
}

const VxVector VxVector::m_AxisX(1.0f, 0.0f, 0.0f);
const VxVector VxVector::m_AxisY(0.0f, 1.0f, 0.0f);
const VxVector VxVector::m_AxisZ(0.0f, 0.0f, 1.0f);
const VxVector VxVector::m_Axis0(0.0f, 0.0f, 0.0f);
const VxVector VxVector::m_Axis1(1.0f, 1.0f, 1.0f);

VxVector &VxVector::operator=(const VxCompressedVector &v) {
    const float yaAngle = static_cast<float>(v.ya) * (1.0f / 32767.0f) * PI;
    const float xaAngle = static_cast<float>(v.xa) * (1.0f / 32767.0f) * PI;

    const float sinYa = TableSin(yaAngle);
    const float cosYa = TableCos(yaAngle);
    const float sinXa = TableSin(xaAngle);
    const float cosXa = TableCos(xaAngle);

    x = sinYa * sinXa;
    y = -sinXa;
    z = cosYa * cosXa;

    return *this;
}

void VxVector::Normalize() {
    const float magSq = SquareMagnitude();
    if (magSq > EPSILON) {
        const float invMag = 1.0f / sqrtf(magSq);
        x *= invMag;
        y *= invMag;
        z *= invMag;
    }
}

void VxVector::Rotate(const VxMatrix &M) {
    const float ox = x, oy = y, oz = z;

    const float m00 = M[0][0], m10 = M[1][0], m20 = M[2][0];
    const float m01 = M[0][1], m11 = M[1][1], m21 = M[2][1];
    const float m02 = M[0][2], m12 = M[1][2], m22 = M[2][2];

    x = m00 * ox + m10 * oy + m20 * oz;
    y = m01 * ox + m11 * oy + m21 * oz;
    z = m02 * ox + m12 * oy + m22 * oz;
}

const VxVector &VxVector::axisX() { return m_AxisX; }
const VxVector &VxVector::axisY() { return m_AxisY; }
const VxVector &VxVector::axisZ() { return m_AxisZ; }
const VxVector &VxVector::axis0() { return m_Axis0; }
const VxVector &VxVector::axis1() { return m_Axis1; }

const VxVector Rotate(const VxMatrix &mat, const VxVector &pt) {
    const float m00 = mat[0][0], m10 = mat[1][0], m20 = mat[2][0];
    const float m01 = mat[0][1], m11 = mat[1][1], m21 = mat[2][1];
    const float m02 = mat[0][2], m12 = mat[1][2], m22 = mat[2][2];

    return VxVector(
        m00 * pt.x + m10 * pt.y + m20 * pt.z,
        m01 * pt.x + m11 * pt.y + m21 * pt.z,
        m02 * pt.x + m12 * pt.y + m22 * pt.z
    );
}

const VxVector Rotate(const VxVector &v1, const VxVector &v2, float angle) {
    const float axisMagSq = v2.x * v2.x + v2.y * v2.y + v2.z * v2.z;
    const float axisInvMag = 1.0f / sqrtf(axisMagSq);
    const float nx = v2.x * axisInvMag;
    const float ny = v2.y * axisInvMag;
    const float nz = v2.z * axisInvMag;

    const float sinAngle = TableSin(angle);
    const float cosAngle = TableCos(angle);
    const float oneMinusCos = 1.0f - cosAngle;

    const float nx2 = nx * nx, ny2 = ny * ny, nz2 = nz * nz;
    const float nx_ny_omc = nx * ny * oneMinusCos;
    const float nx_nz_omc = nx * nz * oneMinusCos;
    const float ny_nz_omc = ny * nz * oneMinusCos;
    const float sin_nx = sinAngle * nx;
    const float sin_ny = sinAngle * ny;
    const float sin_nz = sinAngle * nz;

    return VxVector(
        v1.x * (cosAngle + oneMinusCos * nx2) + v1.y * (nx_ny_omc + sin_nz) + v1.z * (nx_nz_omc - sin_ny),
        v1.x * (nx_ny_omc - sin_nz) + v1.y * (cosAngle + oneMinusCos * ny2) + v1.z * (ny_nz_omc + sin_nx),
        v1.x * (nx_nz_omc + sin_ny) + v1.y * (ny_nz_omc - sin_nx) + v1.z * (cosAngle + oneMinusCos * nz2)
    );
}

const VxVector Rotate(const VxVector *v1, const VxVector *v2, float angle) {
    return Rotate(*v1, *v2, angle);
}
