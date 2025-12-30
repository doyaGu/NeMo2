#include "RCKKeyframeData.h"

#include "VxMath.h"

#include <cstdlib>
#include <cstring>
#include <cmath>

//===================================================================
// Helper functions for TCB spline interpolation
//===================================================================

// Applies ease-in and ease-out parameters to the interpolation parameter
static float ApplyEaseParameters(float t, float easeTo, float easeFrom) {
    // Clamp input
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;

    // Standard hermite ease function
    float easeTotal = easeTo + easeFrom;
    if (easeTotal > 1.0f) {
        easeTo /= easeTotal;
        easeFrom /= easeTotal;
    }

    // Apply ease parameters using Hermite blending
    float t2 = t * t;
    float t3 = t2 * t;

    // Hermite basis functions modified by ease parameters
    float h1 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    float h2 = -2.0f * t3 + 3.0f * t2;
    float h3 = t3 - 2.0f * t2 + t;
    float h4 = t3 - t2;

    return h1 * 0.0f + h2 * 1.0f + h3 * (1.0f - easeFrom) + h4 * easeTo;
}

//===================================================================
// CKKeyframeData Implementation
//===================================================================

CKKeyframeData::CKKeyframeData()
    : m_PositionController(nullptr),
      m_ScaleController(nullptr),
      m_RotationController(nullptr),
      m_ScaleAxisController(nullptr),
      m_MorphController(nullptr),
      m_Length(100.0f),
      m_RefCount(1),
      m_ObjectAnimation(nullptr) {}

CKKeyframeData::~CKKeyframeData() {
    delete m_PositionController;
    delete m_ScaleController;
    delete m_RotationController;
    delete m_ScaleAxisController;
    delete m_MorphController;
}

CKAnimController *CKKeyframeData::CreateController(CKANIMATION_CONTROLLER type) {
    CKAnimController *controller = nullptr;

    switch (type) {
    case CKANIMATION_LINPOS_CONTROL:
        controller = new RCKLinearPositionController();
        break;
    case CKANIMATION_LINROT_CONTROL:
        controller = new RCKLinearRotationController();
        break;
    case CKANIMATION_LINSCL_CONTROL:
        controller = new RCKLinearScaleController();
        break;
    case CKANIMATION_LINSCLAXIS_CONTROL:
        controller = new RCKLinearScaleAxisController();
        break;
    case CKANIMATION_TCBPOS_CONTROL:
        controller = new RCKTCBPositionController();
        break;
    case CKANIMATION_TCBROT_CONTROL:
        controller = new RCKTCBRotationController();
        break;
    case CKANIMATION_TCBSCL_CONTROL:
        controller = new RCKTCBScaleController();
        break;
    case CKANIMATION_TCBSCLAXIS_CONTROL:
        controller = new RCKTCBScaleAxisController();
        break;
    case CKANIMATION_BEZIERPOS_CONTROL:
        controller = new RCKBezierPositionController();
        break;
    case CKANIMATION_BEZIERSCL_CONTROL:
        controller = new RCKBezierScaleController();
        break;
    case CKANIMATION_MORPH_CONTROL:
        controller = new RCKMorphController();
        break;
    default:
        return nullptr;
    }

    if (controller) {
        controller->SetLength(m_Length);
    }

    return controller;
}

//===================================================================
// RCKLinearPositionController Implementation
//===================================================================

RCKLinearPositionController::RCKLinearPositionController()
    : CKAnimController(CKANIMATION_LINPOS_CONTROL), m_Keys(nullptr) {
    m_Length = 0.0f;
}

RCKLinearPositionController::~RCKLinearPositionController() {
    delete[] m_Keys;
}

CKBOOL RCKLinearPositionController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxVector *result = static_cast<VxVector *>(res);

    // Before first key - return first key value
    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Pos;
        return TRUE;
    }

    // After last key - return last key value
    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Pos;
        return TRUE;
    }

    // Binary search for the interval containing TimeStep
    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    // Linear interpolation between keys[low] and keys[high]
    float t1 = m_Keys[low].TimeStep;
    float t2 = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2 - t1);

    VxVector &p1 = m_Keys[low].Pos;
    VxVector &p2 = m_Keys[high].Pos;

    result->x = p1.x + (p2.x - p1.x) * t;
    result->y = p1.y + (p2.y - p1.y) * t;
    result->z = p1.z + (p2.z - p1.z) * t;

    return TRUE;
}

int RCKLinearPositionController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKPositionKey *posKey = static_cast<CKPositionKey *>(key);
    float time = posKey->TimeStep;

    // Find insertion position (maintain sorted order)
    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        // If key already exists at this time, update it
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i].Pos = posKey->Pos;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    // Allocate new array and copy existing keys
    ++m_NbKeys;
    CKPositionKey *newKeys = new CKPositionKey[m_NbKeys];

    if (m_Keys) {
        // Copy keys before insertion point
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKPositionKey));

        // Copy keys after insertion point
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKPositionKey));

        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *posKey;

    return insertIdx;
}

CKKey *RCKLinearPositionController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKLinearPositionController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        return;
    }

    CKPositionKey *newKeys = new CKPositionKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKPositionKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKPositionKey));

    delete[] m_Keys;
    m_Keys = newKeys;
}

int RCKLinearPositionController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKPositionKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKPositionKey));
    }

    return size;
}

int RCKLinearPositionController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKPositionKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKPositionKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKPositionKey);
}

CKBOOL RCKLinearPositionController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKLinearPositionController *other = static_cast<RCKLinearPositionController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKLinearPositionController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKLinearPositionController *other = static_cast<RCKLinearPositionController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKPositionKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKPositionKey));
    }

    return TRUE;
}

//===================================================================
// RCKLinearRotationController Implementation
//===================================================================

RCKLinearRotationController::RCKLinearRotationController()
    : CKAnimController(CKANIMATION_LINROT_CONTROL), m_Keys(nullptr) {
    m_Length = 0.0f;
}

RCKLinearRotationController::~RCKLinearRotationController() {
    delete[] m_Keys;
}

CKBOOL RCKLinearRotationController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxQuaternion *result = static_cast<VxQuaternion *>(res);

    // Before first key
    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Rot;
        return TRUE;
    }

    // After last key
    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Rot;
        return TRUE;
    }

    // Binary search
    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    // Spherical linear interpolation (Slerp) between quaternions
    float t1 = m_Keys[low].TimeStep;
    float t2 = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2 - t1);

    *result = Slerp(t, m_Keys[low].Rot, m_Keys[high].Rot);

    return TRUE;
}

int RCKLinearRotationController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKRotationKey *rotKey = static_cast<CKRotationKey *>(key);
    float time = rotKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i].Rot = rotKey->Rot;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKRotationKey *newKeys = new CKRotationKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKRotationKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKRotationKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *rotKey;

    return insertIdx;
}

CKKey *RCKLinearRotationController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKLinearRotationController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        return;
    }

    CKRotationKey *newKeys = new CKRotationKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKRotationKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKRotationKey));

    delete[] m_Keys;
    m_Keys = newKeys;
}

int RCKLinearRotationController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKRotationKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKRotationKey));
    }

    return size;
}

int RCKLinearRotationController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKRotationKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKRotationKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKRotationKey);
}

CKBOOL RCKLinearRotationController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKLinearRotationController *other = static_cast<RCKLinearRotationController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKLinearRotationController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKLinearRotationController *other = static_cast<RCKLinearRotationController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKRotationKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKRotationKey));
    }

    return TRUE;
}

//===================================================================
// RCKLinearScaleController Implementation
//===================================================================

RCKLinearScaleController::RCKLinearScaleController()
    : CKAnimController(CKANIMATION_LINSCL_CONTROL), m_Keys(nullptr) {
    m_Length = 0.0f;
}

RCKLinearScaleController::~RCKLinearScaleController() {
    delete[] m_Keys;
}

CKBOOL RCKLinearScaleController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxVector *result = static_cast<VxVector *>(res);

    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Pos;
        return TRUE;
    }

    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Pos;
        return TRUE;
    }

    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    float t1 = m_Keys[low].TimeStep;
    float t2_time = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2_time - t1);

    VxVector &p1 = m_Keys[low].Pos;
    VxVector &p2 = m_Keys[high].Pos;

    result->x = p1.x + (p2.x - p1.x) * t;
    result->y = p1.y + (p2.y - p1.y) * t;
    result->z = p1.z + (p2.z - p1.z) * t;

    return TRUE;
}

int RCKLinearScaleController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKScaleKey *scaleKey = static_cast<CKScaleKey *>(key);
    float time = scaleKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i].Pos = scaleKey->Pos;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKScaleKey *newKeys = new CKScaleKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKScaleKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKScaleKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *scaleKey;

    return insertIdx;
}

CKKey *RCKLinearScaleController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKLinearScaleController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        return;
    }

    CKScaleKey *newKeys = new CKScaleKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKScaleKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKScaleKey));

    delete[] m_Keys;
    m_Keys = newKeys;
}

int RCKLinearScaleController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKScaleKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKScaleKey));
    }

    return size;
}

int RCKLinearScaleController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKScaleKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKScaleKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKScaleKey);
}

CKBOOL RCKLinearScaleController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKLinearScaleController *other = static_cast<RCKLinearScaleController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKLinearScaleController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKLinearScaleController *other = static_cast<RCKLinearScaleController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKScaleKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKScaleKey));
    }

    return TRUE;
}

//===================================================================
// RCKLinearScaleAxisController Implementation
//===================================================================

RCKLinearScaleAxisController::RCKLinearScaleAxisController()
    : CKAnimController(CKANIMATION_LINSCLAXIS_CONTROL), m_Keys(nullptr) {
    m_Length = 0.0f;
}

RCKLinearScaleAxisController::~RCKLinearScaleAxisController() {
    delete[] m_Keys;
}

CKBOOL RCKLinearScaleAxisController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxQuaternion *result = static_cast<VxQuaternion *>(res);

    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Rot;
        return TRUE;
    }

    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Rot;
        return TRUE;
    }

    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    float t1 = m_Keys[low].TimeStep;
    float t2 = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2 - t1);

    *result = Slerp(t, m_Keys[low].Rot, m_Keys[high].Rot);

    return TRUE;
}

int RCKLinearScaleAxisController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKScaleAxisKey *scaleAxisKey = static_cast<CKScaleAxisKey *>(key);
    float time = scaleAxisKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i].Rot = scaleAxisKey->Rot;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKScaleAxisKey *newKeys = new CKScaleAxisKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKScaleAxisKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKScaleAxisKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *scaleAxisKey;

    return insertIdx;
}

CKKey *RCKLinearScaleAxisController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKLinearScaleAxisController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        return;
    }

    CKScaleAxisKey *newKeys = new CKScaleAxisKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKScaleAxisKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKScaleAxisKey));

    delete[] m_Keys;
    m_Keys = newKeys;
}

int RCKLinearScaleAxisController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKScaleAxisKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKScaleAxisKey));
    }

    return size;
}

int RCKLinearScaleAxisController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKScaleAxisKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKScaleAxisKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKScaleAxisKey);
}

CKBOOL RCKLinearScaleAxisController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKLinearScaleAxisController *other = static_cast<RCKLinearScaleAxisController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKLinearScaleAxisController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKLinearScaleAxisController *other = static_cast<RCKLinearScaleAxisController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKScaleAxisKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKScaleAxisKey));
    }

    return TRUE;
}

//===================================================================
// RCKTCBPositionController Implementation
//===================================================================

RCKTCBPositionController::RCKTCBPositionController()
    : CKAnimController(CKANIMATION_TCBPOS_CONTROL), m_Keys(nullptr), m_Tangents(nullptr) {
    m_Length = 0.0f;
}

RCKTCBPositionController::~RCKTCBPositionController() {
    delete[] m_Keys;
    delete[] m_Tangents;
}

void RCKTCBPositionController::ComputeTangents() {
    if (m_NbKeys < 2) {
        delete[] m_Tangents;
        m_Tangents = nullptr;
        return;
    }

    // Allocate tangent arrays: 2 tangents per key (incoming and outgoing)
    delete[] m_Tangents;
    m_Tangents = new VxVector[m_NbKeys * 2];

    for (int i = 0; i < m_NbKeys; ++i) {
        VxVector &tanIn = m_Tangents[i * 2];
        VxVector &tanOut = m_Tangents[i * 2 + 1];

        float tension = m_Keys[i].tension;
        float continuity = m_Keys[i].continuity;
        float bias = m_Keys[i].bias;

        // Compute adjustment factors based on TCB parameters
        float factorIn1 = ((1.0f - tension) * (1.0f + continuity) * (1.0f + bias)) / 2.0f;
        float factorIn2 = ((1.0f - tension) * (1.0f - continuity) * (1.0f - bias)) / 2.0f;
        float factorOut1 = ((1.0f - tension) * (1.0f - continuity) * (1.0f + bias)) / 2.0f;
        float factorOut2 = ((1.0f - tension) * (1.0f + continuity) * (1.0f - bias)) / 2.0f;

        VxVector dp, dn;
        if (i == 0) {
            // First key: use forward difference
            dp = m_Keys[1].Pos - m_Keys[0].Pos;
            dn = dp;
        } else if (i == m_NbKeys - 1) {
            // Last key: use backward difference
            dp = m_Keys[m_NbKeys - 1].Pos - m_Keys[m_NbKeys - 2].Pos;
            dn = dp;
        } else {
            // Interior key: use both differences
            dp = m_Keys[i].Pos - m_Keys[i - 1].Pos;
            dn = m_Keys[i + 1].Pos - m_Keys[i].Pos;
        }

        // Compute incoming tangent
        tanIn.x = factorIn1 * dp.x + factorIn2 * dn.x;
        tanIn.y = factorIn1 * dp.y + factorIn2 * dn.y;
        tanIn.z = factorIn1 * dp.z + factorIn2 * dn.z;

        // Compute outgoing tangent
        tanOut.x = factorOut1 * dp.x + factorOut2 * dn.x;
        tanOut.y = factorOut1 * dp.y + factorOut2 * dn.y;
        tanOut.z = factorOut1 * dp.z + factorOut2 * dn.z;
    }
}

CKBOOL RCKTCBPositionController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxVector *result = static_cast<VxVector *>(res);

    // Compute tangents if not already computed
    if (!m_Tangents)
        ComputeTangents();

    // Before first key
    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Pos;
        return TRUE;
    }

    // After last key
    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Pos;
        return TRUE;
    }

    // Binary search for interval
    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    // Hermite spline interpolation
    float t1 = m_Keys[low].TimeStep;
    float t2 = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2 - t1);

    // Apply ease parameters
    t = ApplyEaseParameters(t, m_Keys[low].easeto, m_Keys[high].easefrom);

    // Hermite basis functions
    float t2_val = t * t;
    float t3 = t2_val * t;

    float h1 = 2.0f * t3 - 3.0f * t2_val + 1.0f; // (2t^3 - 3t^2 + 1)
    float h2 = -2.0f * t3 + 3.0f * t2_val;       // (-2t^3 + 3t^2)
    float h3 = t3 - 2.0f * t2_val + t;           // (t^3 - 2t^2 + t)
    float h4 = t3 - t2_val;                      // (t^3 - t^2)

    VxVector &p1 = m_Keys[low].Pos;
    VxVector &p2 = m_Keys[high].Pos;
    VxVector &m1 = m_Tangents[low * 2 + 1]; // outgoing tangent of low key
    VxVector &m2 = m_Tangents[high * 2];    // incoming tangent of high key

    result->x = h1 * p1.x + h2 * p2.x + h3 * m1.x + h4 * m2.x;
    result->y = h1 * p1.y + h2 * p2.y + h3 * m1.y + h4 * m2.y;
    result->z = h1 * p1.z + h2 * p2.z + h3 * m1.z + h4 * m2.z;

    return TRUE;
}

int RCKTCBPositionController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKTCBPositionKey *tcbKey = static_cast<CKTCBPositionKey *>(key);
    float time = tcbKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i] = *tcbKey;
            // Invalidate tangents
            delete[] m_Tangents;
            m_Tangents = nullptr;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKTCBPositionKey *newKeys = new CKTCBPositionKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKTCBPositionKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKTCBPositionKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *tcbKey;

    // Invalidate tangents
    delete[] m_Tangents;
    m_Tangents = nullptr;

    return insertIdx;
}

CKKey *RCKTCBPositionController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKTCBPositionController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        delete[] m_Tangents;
        m_Tangents = nullptr;
        return;
    }

    CKTCBPositionKey *newKeys = new CKTCBPositionKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKTCBPositionKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKTCBPositionKey));

    delete[] m_Keys;
    m_Keys = newKeys;

    delete[] m_Tangents;
    m_Tangents = nullptr;
}

int RCKTCBPositionController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKTCBPositionKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKTCBPositionKey));
    }

    return size;
}

int RCKTCBPositionController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;
    delete[] m_Tangents;
    m_Tangents = nullptr;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKTCBPositionKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKTCBPositionKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKTCBPositionKey);
}

CKBOOL RCKTCBPositionController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKTCBPositionController *other = static_cast<RCKTCBPositionController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKTCBPositionController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKTCBPositionController *other = static_cast<RCKTCBPositionController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;
    delete[] m_Tangents;
    m_Tangents = nullptr;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKTCBPositionKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKTCBPositionKey));
    }

    return TRUE;
}

//===================================================================
// RCKTCBRotationController Implementation
//===================================================================

RCKTCBRotationController::RCKTCBRotationController()
    : CKAnimController(CKANIMATION_TCBROT_CONTROL), m_Keys(nullptr), m_Tangents(nullptr) {
    m_Length = 0.0f;
}

RCKTCBRotationController::~RCKTCBRotationController() {
    delete[] m_Keys;
    delete[] m_Tangents;
}

void RCKTCBRotationController::ComputeTangents() {
    if (m_NbKeys < 2) {
        delete[] m_Tangents;
        m_Tangents = nullptr;
        return;
    }

    // Allocate tangent arrays
    delete[] m_Tangents;
    m_Tangents = new VxQuaternion[m_NbKeys * 2];

    for (int i = 0; i < m_NbKeys; ++i) {
        VxQuaternion &tanIn = m_Tangents[i * 2];
        VxQuaternion &tanOut = m_Tangents[i * 2 + 1];

        float tension = m_Keys[i].tension;
        float continuity = m_Keys[i].continuity;
        float bias = m_Keys[i].bias;

        // Get quaternion difference between keys
        VxQuaternion qPrev, qCurr, qNext;
        qCurr = m_Keys[i].Rot;

        if (i == 0) {
            qPrev = qCurr;
            qNext = m_Keys[1].Rot;
        } else if (i == m_NbKeys - 1) {
            qPrev = m_Keys[m_NbKeys - 2].Rot;
            qNext = qCurr;
        } else {
            qPrev = m_Keys[i - 1].Rot;
            qNext = m_Keys[i + 1].Rot;
        }

        // For quaternion TCB, use logarithmic representation
        // Simplified implementation: linear blend of quaternions
        float factorIn = (1.0f - tension) * 0.5f;
        float factorOut = (1.0f - tension) * 0.5f;

        // Compute incoming tangent quaternion
        tanIn = Slerp(0.5f, qPrev, qNext);

        // Compute outgoing tangent quaternion
        tanOut = Slerp(0.5f, qPrev, qNext);
    }
}

CKBOOL RCKTCBRotationController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxQuaternion *result = static_cast<VxQuaternion *>(res);

    if (!m_Tangents)
        ComputeTangents();

    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Rot;
        return TRUE;
    }

    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Rot;
        return TRUE;
    }

    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    float t1 = m_Keys[low].TimeStep;
    float t2 = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2 - t1);

    // Apply ease parameters
    t = ApplyEaseParameters(t, m_Keys[low].easeto, m_Keys[high].easefrom);

    // Use Squad (spherical quadrangle) interpolation for smooth rotation
    VxQuaternion &q1 = m_Keys[low].Rot;
    VxQuaternion &q2 = m_Keys[high].Rot;

    // Squad interpolation
    *result = Squad(t, q1, m_Tangents[low * 2 + 1], m_Tangents[high * 2], q2);

    return TRUE;
}

int RCKTCBRotationController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKTCBRotationKey *tcbKey = static_cast<CKTCBRotationKey *>(key);
    float time = tcbKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i] = *tcbKey;
            delete[] m_Tangents;
            m_Tangents = nullptr;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKTCBRotationKey *newKeys = new CKTCBRotationKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKTCBRotationKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKTCBRotationKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *tcbKey;

    delete[] m_Tangents;
    m_Tangents = nullptr;

    return insertIdx;
}

CKKey *RCKTCBRotationController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKTCBRotationController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        delete[] m_Tangents;
        m_Tangents = nullptr;
        return;
    }

    CKTCBRotationKey *newKeys = new CKTCBRotationKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKTCBRotationKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKTCBRotationKey));

    delete[] m_Keys;
    m_Keys = newKeys;

    delete[] m_Tangents;
    m_Tangents = nullptr;
}

int RCKTCBRotationController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKTCBRotationKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKTCBRotationKey));
    }

    return size;
}

int RCKTCBRotationController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;
    delete[] m_Tangents;
    m_Tangents = nullptr;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKTCBRotationKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKTCBRotationKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKTCBRotationKey);
}

CKBOOL RCKTCBRotationController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKTCBRotationController *other = static_cast<RCKTCBRotationController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKTCBRotationController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKTCBRotationController *other = static_cast<RCKTCBRotationController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;
    delete[] m_Tangents;
    m_Tangents = nullptr;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKTCBRotationKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKTCBRotationKey));
    }

    return TRUE;
}

//===================================================================
// RCKTCBScaleController Implementation
//===================================================================

RCKTCBScaleController::RCKTCBScaleController()
    : CKAnimController(CKANIMATION_TCBSCL_CONTROL), m_Keys(nullptr), m_Tangents(nullptr) {
    m_Length = 0.0f;
}

RCKTCBScaleController::~RCKTCBScaleController() {
    delete[] m_Keys;
    delete[] m_Tangents;
}

void RCKTCBScaleController::ComputeTangents() {
    if (m_NbKeys < 2) {
        delete[] m_Tangents;
        m_Tangents = nullptr;
        return;
    }

    delete[] m_Tangents;
    m_Tangents = new VxVector[m_NbKeys * 2];

    for (int i = 0; i < m_NbKeys; ++i) {
        VxVector &tanIn = m_Tangents[i * 2];
        VxVector &tanOut = m_Tangents[i * 2 + 1];

        float tension = m_Keys[i].tension;
        float continuity = m_Keys[i].continuity;
        float bias = m_Keys[i].bias;

        float factorIn1 = ((1.0f - tension) * (1.0f + continuity) * (1.0f + bias)) / 2.0f;
        float factorIn2 = ((1.0f - tension) * (1.0f - continuity) * (1.0f - bias)) / 2.0f;
        float factorOut1 = ((1.0f - tension) * (1.0f - continuity) * (1.0f + bias)) / 2.0f;
        float factorOut2 = ((1.0f - tension) * (1.0f + continuity) * (1.0f - bias)) / 2.0f;

        VxVector dp, dn;
        if (i == 0) {
            dp = m_Keys[1].Pos - m_Keys[0].Pos;
            dn = dp;
        } else if (i == m_NbKeys - 1) {
            dp = m_Keys[m_NbKeys - 1].Pos - m_Keys[m_NbKeys - 2].Pos;
            dn = dp;
        } else {
            dp = m_Keys[i].Pos - m_Keys[i - 1].Pos;
            dn = m_Keys[i + 1].Pos - m_Keys[i].Pos;
        }

        tanIn.x = factorIn1 * dp.x + factorIn2 * dn.x;
        tanIn.y = factorIn1 * dp.y + factorIn2 * dn.y;
        tanIn.z = factorIn1 * dp.z + factorIn2 * dn.z;

        tanOut.x = factorOut1 * dp.x + factorOut2 * dn.x;
        tanOut.y = factorOut1 * dp.y + factorOut2 * dn.y;
        tanOut.z = factorOut1 * dp.z + factorOut2 * dn.z;
    }
}

CKBOOL RCKTCBScaleController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxVector *result = static_cast<VxVector *>(res);

    if (!m_Tangents)
        ComputeTangents();

    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Pos;
        return TRUE;
    }

    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Pos;
        return TRUE;
    }

    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    float t1 = m_Keys[low].TimeStep;
    float t2 = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2 - t1);

    t = ApplyEaseParameters(t, m_Keys[low].easeto, m_Keys[high].easefrom);

    float t2_val = t * t;
    float t3 = t2_val * t;

    float h1 = 2.0f * t3 - 3.0f * t2_val + 1.0f;
    float h2 = -2.0f * t3 + 3.0f * t2_val;
    float h3 = t3 - 2.0f * t2_val + t;
    float h4 = t3 - t2_val;

    VxVector &p1 = m_Keys[low].Pos;
    VxVector &p2 = m_Keys[high].Pos;
    VxVector &m1 = m_Tangents[low * 2 + 1];
    VxVector &m2 = m_Tangents[high * 2];

    result->x = h1 * p1.x + h2 * p2.x + h3 * m1.x + h4 * m2.x;
    result->y = h1 * p1.y + h2 * p2.y + h3 * m1.y + h4 * m2.y;
    result->z = h1 * p1.z + h2 * p2.z + h3 * m1.z + h4 * m2.z;

    return TRUE;
}

int RCKTCBScaleController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKTCBScaleKey *tcbKey = static_cast<CKTCBScaleKey *>(key);
    float time = tcbKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i] = *tcbKey;
            delete[] m_Tangents;
            m_Tangents = nullptr;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKTCBScaleKey *newKeys = new CKTCBScaleKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKTCBScaleKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKTCBScaleKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *tcbKey;

    delete[] m_Tangents;
    m_Tangents = nullptr;

    return insertIdx;
}

CKKey *RCKTCBScaleController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKTCBScaleController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        delete[] m_Tangents;
        m_Tangents = nullptr;
        return;
    }

    CKTCBScaleKey *newKeys = new CKTCBScaleKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKTCBScaleKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1],
               (m_NbKeys - index) * sizeof(CKTCBScaleKey));

    delete[] m_Keys;
    m_Keys = newKeys;

    delete[] m_Tangents;
    m_Tangents = nullptr;
}

int RCKTCBScaleController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKTCBScaleKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKTCBScaleKey));
    }

    return size;
}

int RCKTCBScaleController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;
    delete[] m_Tangents;
    m_Tangents = nullptr;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKTCBScaleKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKTCBScaleKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKTCBScaleKey);
}

CKBOOL RCKTCBScaleController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKTCBScaleController *other = static_cast<RCKTCBScaleController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKTCBScaleController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKTCBScaleController *other = static_cast<RCKTCBScaleController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;
    delete[] m_Tangents;
    m_Tangents = nullptr;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKTCBScaleKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKTCBScaleKey));
    }

    return TRUE;
}

//===================================================================
// RCKTCBScaleAxisController Implementation
//===================================================================

RCKTCBScaleAxisController::RCKTCBScaleAxisController()
    : CKAnimController(CKANIMATION_TCBSCLAXIS_CONTROL), m_Keys(nullptr), m_Tangents(nullptr) {
    m_Length = 0.0f;
}

RCKTCBScaleAxisController::~RCKTCBScaleAxisController() {
    delete[] m_Keys;
    delete[] m_Tangents;
}

void RCKTCBScaleAxisController::ComputeTangents() {
    if (m_NbKeys < 2) {
        delete[] m_Tangents;
        m_Tangents = nullptr;
        return;
    }

    delete[] m_Tangents;
    m_Tangents = new VxQuaternion[m_NbKeys * 2];

    for (int i = 0; i < m_NbKeys; ++i) {
        VxQuaternion &tanIn = m_Tangents[i * 2];
        VxQuaternion &tanOut = m_Tangents[i * 2 + 1];

        VxQuaternion qPrev, qCurr, qNext;
        qCurr = m_Keys[i].Rot;

        if (i == 0) {
            qPrev = qCurr;
            qNext = m_Keys[1].Rot;
        } else if (i == m_NbKeys - 1) {
            qPrev = m_Keys[m_NbKeys - 2].Rot;
            qNext = qCurr;
        } else {
            qPrev = m_Keys[i - 1].Rot;
            qNext = m_Keys[i + 1].Rot;
        }

        tanIn = Slerp(0.5f, qPrev, qNext);
        tanOut = Slerp(0.5f, qPrev, qNext);
    }
}

CKBOOL RCKTCBScaleAxisController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxQuaternion *result = static_cast<VxQuaternion *>(res);

    if (!m_Tangents)
        ComputeTangents();

    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Rot;
        return TRUE;
    }

    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Rot;
        return TRUE;
    }

    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    float t1 = m_Keys[low].TimeStep;
    float t2 = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2 - t1);

    t = ApplyEaseParameters(t, m_Keys[low].easeto, m_Keys[high].easefrom);

    VxQuaternion &q1 = m_Keys[low].Rot;
    VxQuaternion &q2 = m_Keys[high].Rot;

    *result = Squad(t, q1, m_Tangents[low * 2 + 1], m_Tangents[high * 2], q2);

    return TRUE;
}

int RCKTCBScaleAxisController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKTCBScaleAxisKey *tcbKey = static_cast<CKTCBScaleAxisKey *>(key);
    float time = tcbKey->TimeStep;


    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i] = *tcbKey;
            delete[] m_Tangents;
            m_Tangents = nullptr;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKTCBScaleAxisKey *newKeys = new CKTCBScaleAxisKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKTCBScaleAxisKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKTCBScaleAxisKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *tcbKey;

    delete[] m_Tangents;
    m_Tangents = nullptr;

    return insertIdx;
}

CKKey *RCKTCBScaleAxisController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKTCBScaleAxisController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        delete[] m_Tangents;
        m_Tangents = nullptr;
        return;
    }

    CKTCBScaleAxisKey *newKeys = new CKTCBScaleAxisKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKTCBScaleAxisKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKTCBScaleAxisKey));

    delete[] m_Keys;
    m_Keys = newKeys;

    delete[] m_Tangents;
    m_Tangents = nullptr;
}

int RCKTCBScaleAxisController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKTCBScaleAxisKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKTCBScaleAxisKey));
    }

    return size;
}

int RCKTCBScaleAxisController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;
    delete[] m_Tangents;
    m_Tangents = nullptr;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKTCBScaleAxisKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKTCBScaleAxisKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKTCBScaleAxisKey);
}

CKBOOL RCKTCBScaleAxisController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKTCBScaleAxisController *other = static_cast<RCKTCBScaleAxisController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKTCBScaleAxisController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKTCBScaleAxisController *other = static_cast<RCKTCBScaleAxisController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;
    delete[] m_Tangents;
    m_Tangents = nullptr;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKTCBScaleAxisKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKTCBScaleAxisKey));
    }

    return TRUE;
}

//===================================================================
// RCKBezierPositionController Implementation
//===================================================================

RCKBezierPositionController::RCKBezierPositionController()
    : CKAnimController(CKANIMATION_BEZIERPOS_CONTROL), m_Keys(nullptr), m_TangentsComputed(FALSE) {
    m_Length = 0.0f;
}

RCKBezierPositionController::~RCKBezierPositionController() {
    delete[] m_Keys;
}

float RCKBezierPositionController::ComputeKeyDistance(int key1, int key2) {
    if (key1 < 0 || key1 >= m_NbKeys || key2 < 0 || key2 >= m_NbKeys)
        return 0.0f;

    VxVector diff = m_Keys[key2].Pos - m_Keys[key1].Pos;
    return sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
}

void RCKBezierPositionController::ComputeBezierPts() {
    for (int i = 0; i < m_NbKeys; ++i) {
        ComputeBezierPts(i);
    }
    m_TangentsComputed = TRUE;
}

void RCKBezierPositionController::ComputeBezierPts(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    CKBezierPositionKey &key = m_Keys[index];
    CKBEZIERKEY_FLAGS inMode = key.Flags.GetInTangentMode();
    CKBEZIERKEY_FLAGS outMode = key.Flags.GetOutTangentMode();

    int prevIdx = (index + m_NbKeys - 1) % m_NbKeys;
    int nextIdx = (index + 1) % m_NbKeys;

    // Only compute if this is an interior key or we have enough keys
    if (m_NbKeys == 1) {
        key.In = VxVector(0, 0, 0);
        key.Out = VxVector(0, 0, 0);
        return;
    }

    // Compute incoming tangent based on mode
    if (inMode != BEZIER_KEY_TANGENTS) {
        if (inMode == BEZIER_KEY_LINEAR) {
            // Linear: tangent points directly to previous key
            if (index > 0)
                key.In = VxVector(0, 0, 0);
        } else if (inMode == BEZIER_KEY_STEP) {
            key.In = VxVector(0, 0, 0);
        } else if (inMode == BEZIER_KEY_FAST) {
            // Fast: 2 * (next - prev)
            if (index > 0 && index < m_NbKeys - 1) {
                VxVector diff = m_Keys[nextIdx].Pos - m_Keys[prevIdx].Pos;
                key.In = diff * -1.0f; // Negate for incoming
            }
        } else if (inMode == BEZIER_KEY_SLOW) {
            key.In = VxVector(0, 0, 0);
        } else // BEZIER_KEY_AUTOSMOOTH
        {
            // Smooth tangent: average of neighboring segments
            if (index == 0 || index == m_NbKeys - 1) {
                key.In = VxVector(0, 0, 0);
            } else {
                float dist1 = ComputeKeyDistance(prevIdx, index);
                float dist2 = ComputeKeyDistance(index, nextIdx);

                if (dist1 == 0.0f) dist1 = 1.0f;
                if (dist2 == 0.0f) dist2 = 1.0f;

                float ratio = dist1 / (dist1 + dist2);

                VxVector d1 = m_Keys[index].Pos - m_Keys[prevIdx].Pos;
                VxVector d2 = m_Keys[nextIdx].Pos - m_Keys[index].Pos;

                // Catmull-Rom style tangent
                VxVector tangent;
                tangent.x = (ratio / dist1) * d1.x - ((1.0f - ratio) / dist2) * d2.x;
                tangent.y = (ratio / dist1) * d1.y - ((1.0f - ratio) / dist2) * d2.y;
                tangent.z = (ratio / dist1) * d1.z - ((1.0f - ratio) / dist2) * d2.z;

                key.In = tangent;
            }
        }
    }

    // Compute outgoing tangent based on mode
    if (outMode != BEZIER_KEY_TANGENTS) {
        if (outMode == BEZIER_KEY_LINEAR) {
            if (index < m_NbKeys - 1)
                key.Out = VxVector(0, 0, 0);
        } else if (outMode == BEZIER_KEY_STEP) {
            key.Out = VxVector(0, 0, 0);
        } else if (outMode == BEZIER_KEY_FAST) {
            if (index > 0 && index < m_NbKeys - 1) {
                VxVector diff = m_Keys[nextIdx].Pos - m_Keys[prevIdx].Pos;
                key.Out = diff;
            }
        } else if (outMode == BEZIER_KEY_SLOW) {
            key.Out = VxVector(0, 0, 0);
        } else // BEZIER_KEY_AUTOSMOOTH
        {
            if (index == 0 || index == m_NbKeys - 1) {
                key.Out = VxVector(0, 0, 0);
            } else {
                float dist1 = ComputeKeyDistance(prevIdx, index);
                float dist2 = ComputeKeyDistance(index, nextIdx);

                if (dist1 == 0.0f) dist1 = 1.0f;
                if (dist2 == 0.0f) dist2 = 1.0f;

                float ratio = dist1 / (dist1 + dist2);

                VxVector d1 = m_Keys[index].Pos - m_Keys[prevIdx].Pos;
                VxVector d2 = m_Keys[nextIdx].Pos - m_Keys[index].Pos;

                VxVector tangent;
                tangent.x = ((1.0f - ratio) / dist1) * d1.x + (ratio / dist2) * d2.x;
                tangent.y = ((1.0f - ratio) / dist1) * d1.y + (ratio / dist2) * d2.y;
                tangent.z = ((1.0f - ratio) / dist1) * d1.z + (ratio / dist2) * d2.z;

                key.Out = tangent;
            }
        }
    }
}

CKBOOL RCKBezierPositionController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxVector *result = static_cast<VxVector *>(res);

    if (!m_TangentsComputed)
        ComputeBezierPts();

    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Pos;
        return TRUE;
    }

    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Pos;
        return TRUE;
    }

    // Binary search
    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    float t1 = m_Keys[low].TimeStep;
    float t2_time = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2_time - t1);

    // Cubic Bezier interpolation
    // B(t) = (1-t)^3 * P0 + 3(1-t)^2 * t * P1 + 3(1-t) * t^2 * P2 + t^3 * P3
    // where P0 = key[low].Pos, P1 = P0 + Out, P2 = P3 + In, P3 = key[high].Pos

    VxVector &p0 = m_Keys[low].Pos;
    VxVector &p3 = m_Keys[high].Pos;
    VxVector p1 = p0 + m_Keys[low].Out;
    VxVector p2 = p3 + m_Keys[high].In;

    float omt = 1.0f - t;
    float omt2 = omt * omt;
    float omt3 = omt2 * omt;
    float t2 = t * t;
    float t3 = t2 * t;

    result->x = omt3 * p0.x + 3.0f * omt2 * t * p1.x + 3.0f * omt * t2 * p2.x + t3 * p3.x;
    result->y = omt3 * p0.y + 3.0f * omt2 * t * p1.y + 3.0f * omt * t2 * p2.y + t3 * p3.y;
    result->z = omt3 * p0.z + 3.0f * omt2 * t * p1.z + 3.0f * omt * t2 * p2.z + t3 * p3.z;

    return TRUE;
}

int RCKBezierPositionController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKBezierPositionKey *bezKey = static_cast<CKBezierPositionKey *>(key);
    float time = bezKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i] = *bezKey;
            m_TangentsComputed = FALSE;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKBezierPositionKey *newKeys = new CKBezierPositionKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKBezierPositionKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKBezierPositionKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *bezKey;
    m_TangentsComputed = FALSE;

    return insertIdx;
}

CKKey *RCKBezierPositionController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKBezierPositionController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        return;
    }

    CKBezierPositionKey *newKeys = new CKBezierPositionKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKBezierPositionKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKBezierPositionKey));

    delete[] m_Keys;
    m_Keys = newKeys;
    m_TangentsComputed = FALSE;
}

int RCKBezierPositionController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKBezierPositionKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKBezierPositionKey));
    }

    return size;
}

int RCKBezierPositionController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;
    m_TangentsComputed = FALSE;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKBezierPositionKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKBezierPositionKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKBezierPositionKey);
}

CKBOOL RCKBezierPositionController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKBezierPositionController *other = static_cast<RCKBezierPositionController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKBezierPositionController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKBezierPositionController *other = static_cast<RCKBezierPositionController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;
    m_TangentsComputed = FALSE;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKBezierPositionKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKBezierPositionKey));
    }

    return TRUE;
}

//===================================================================
// RCKBezierScaleController Implementation
//===================================================================

RCKBezierScaleController::RCKBezierScaleController()
    : CKAnimController(CKANIMATION_BEZIERSCL_CONTROL), m_Keys(nullptr), m_TangentsComputed(FALSE) {
    m_Length = 0.0f;
}

RCKBezierScaleController::~RCKBezierScaleController() {
    delete[] m_Keys;
}

float RCKBezierScaleController::ComputeKeyDistance(int key1, int key2) {
    if (key1 < 0 || key1 >= m_NbKeys || key2 < 0 || key2 >= m_NbKeys)
        return 0.0f;

    VxVector diff = m_Keys[key2].Pos - m_Keys[key1].Pos;
    return sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
}

void RCKBezierScaleController::ComputeBezierPts() {
    for (int i = 0; i < m_NbKeys; ++i) {
        ComputeBezierPts(i);
    }
    m_TangentsComputed = TRUE;
}

void RCKBezierScaleController::ComputeBezierPts(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    CKBezierScaleKey &key = m_Keys[index];
    CKBEZIERKEY_FLAGS inMode = key.Flags.GetInTangentMode();
    CKBEZIERKEY_FLAGS outMode = key.Flags.GetOutTangentMode();

    int prevIdx = (index + m_NbKeys - 1) % m_NbKeys;
    int nextIdx = (index + 1) % m_NbKeys;

    if (m_NbKeys == 1) {
        key.In = VxVector(0, 0, 0);
        key.Out = VxVector(0, 0, 0);
        return;
    }

    // Compute tangents similar to position controller
    if (inMode != BEZIER_KEY_TANGENTS) {
        if (index == 0 || index == m_NbKeys - 1 || inMode == BEZIER_KEY_LINEAR ||
            inMode == BEZIER_KEY_STEP || inMode == BEZIER_KEY_SLOW) {
            key.In = VxVector(0, 0, 0);
        } else {
            float dist1 = ComputeKeyDistance(prevIdx, index);
            float dist2 = ComputeKeyDistance(index, nextIdx);
            if (dist1 == 0.0f) dist1 = 1.0f;
            if (dist2 == 0.0f) dist2 = 1.0f;

            float ratio = dist1 / (dist1 + dist2);
            VxVector d1 = m_Keys[index].Pos - m_Keys[prevIdx].Pos;
            VxVector d2 = m_Keys[nextIdx].Pos - m_Keys[index].Pos;

            VxVector tangent;
            tangent.x = (ratio / dist1) * d1.x - ((1.0f - ratio) / dist2) * d2.x;
            tangent.y = (ratio / dist1) * d1.y - ((1.0f - ratio) / dist2) * d2.y;
            tangent.z = (ratio / dist1) * d1.z - ((1.0f - ratio) / dist2) * d2.z;
            key.In = tangent;
        }
    }

    if (outMode != BEZIER_KEY_TANGENTS) {
        if (index == 0 || index == m_NbKeys - 1 || outMode == BEZIER_KEY_LINEAR ||
            outMode == BEZIER_KEY_STEP || outMode == BEZIER_KEY_SLOW) {
            key.Out = VxVector(0, 0, 0);
        } else {
            float dist1 = ComputeKeyDistance(prevIdx, index);
            float dist2 = ComputeKeyDistance(index, nextIdx);
            if (dist1 == 0.0f) dist1 = 1.0f;
            if (dist2 == 0.0f) dist2 = 1.0f;

            float ratio = dist1 / (dist1 + dist2);
            VxVector d1 = m_Keys[index].Pos - m_Keys[prevIdx].Pos;
            VxVector d2 = m_Keys[nextIdx].Pos - m_Keys[index].Pos;

            VxVector tangent;
            tangent.x = ((1.0f - ratio) / dist1) * d1.x + (ratio / dist2) * d2.x;
            tangent.y = ((1.0f - ratio) / dist1) * d1.y + (ratio / dist2) * d2.y;
            tangent.z = ((1.0f - ratio) / dist1) * d1.z + (ratio / dist2) * d2.z;
            key.Out = tangent;
        }
    }
}

CKBOOL RCKBezierScaleController::Evaluate(float TimeStep, void *res) {
    if (m_NbKeys <= 0)
        return FALSE;

    VxVector *result = static_cast<VxVector *>(res);

    if (!m_TangentsComputed)
        ComputeBezierPts();

    if (TimeStep <= m_Keys[0].TimeStep) {
        *result = m_Keys[0].Pos;
        return TRUE;
    }

    if (TimeStep >= m_Keys[m_NbKeys - 1].TimeStep) {
        *result = m_Keys[m_NbKeys - 1].Pos;
        return TRUE;
    }

    int low = 0;
    int high = m_NbKeys - 1;
    while (low < high - 1) {
        int mid = (low + high) >> 1;
        if (m_Keys[mid].TimeStep <= TimeStep)
            low = mid;
        else
            high = mid;
    }

    float t1 = m_Keys[low].TimeStep;
    float t2_time = m_Keys[high].TimeStep;
    float t = (TimeStep - t1) / (t2_time - t1);

    VxVector &p0 = m_Keys[low].Pos;
    VxVector &p3 = m_Keys[high].Pos;
    VxVector p1 = p0 + m_Keys[low].Out;
    VxVector p2 = p3 + m_Keys[high].In;

    float omt = 1.0f - t;
    float omt2 = omt * omt;
    float omt3 = omt2 * omt;
    float t2 = t * t;
    float t3 = t2 * t;

    result->x = omt3 * p0.x + 3.0f * omt2 * t * p1.x + 3.0f * omt * t2 * p2.x + t3 * p3.x;
    result->y = omt3 * p0.y + 3.0f * omt2 * t * p1.y + 3.0f * omt * t2 * p2.y + t3 * p3.y;
    result->z = omt3 * p0.z + 3.0f * omt2 * t * p1.z + 3.0f * omt * t2 * p2.z + t3 * p3.z;

    return TRUE;
}

int RCKBezierScaleController::AddKey(CKKey *key) {
    if (!key)
        return -1;

    CKBezierScaleKey *bezKey = static_cast<CKBezierScaleKey *>(key);
    float time = bezKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            m_Keys[i] = *bezKey;
            m_TangentsComputed = FALSE;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKBezierScaleKey *newKeys = new CKBezierScaleKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKBezierScaleKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKBezierScaleKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *bezKey;
    m_TangentsComputed = FALSE;

    return insertIdx;
}

CKKey *RCKBezierScaleController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKBezierScaleController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        return;
    }

    CKBezierScaleKey *newKeys = new CKBezierScaleKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKBezierScaleKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKBezierScaleKey));

    delete[] m_Keys;
    m_Keys = newKeys;
    m_TangentsComputed = FALSE;
}

int RCKBezierScaleController::DumpKeysTo(void *Buffer) {
    int size = sizeof(int) + m_NbKeys * sizeof(CKBezierScaleKey);

    if (Buffer) {
        int *buf = static_cast<int *>(Buffer);
        *buf++ = m_NbKeys;
        memcpy(buf, m_Keys, m_NbKeys * sizeof(CKBezierScaleKey));
    }

    return size;
}

int RCKBezierScaleController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    delete[] m_Keys;
    m_Keys = nullptr;
    m_TangentsComputed = FALSE;

    int *buf = static_cast<int *>(Buffer);
    m_NbKeys = *buf++;

    if (m_NbKeys > 0) {
        m_Keys = new CKBezierScaleKey[m_NbKeys];
        memcpy(m_Keys, buf, m_NbKeys * sizeof(CKBezierScaleKey));
    }

    return sizeof(int) + m_NbKeys * sizeof(CKBezierScaleKey);
}

CKBOOL RCKBezierScaleController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKBezierScaleController *other = static_cast<RCKBezierScaleController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKBezierScaleController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKBezierScaleController *other = static_cast<RCKBezierScaleController *>(control);

    delete[] m_Keys;
    m_Keys = nullptr;
    m_TangentsComputed = FALSE;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKBezierScaleKey[other->m_NbKeys];
        memcpy(m_Keys, other->m_Keys, other->m_NbKeys * sizeof(CKBezierScaleKey));
    }

    return TRUE;
}

//===================================================================
// RCKMorphController Implementation
//===================================================================

RCKMorphController::RCKMorphController()
    : CKMorphController(), m_Keys(nullptr), m_VertexCount(0) {
    m_Length = 0.0f;
}

RCKMorphController::~RCKMorphController() {
    // Clean up all keys and their allocated data
    for (int i = 0; i < m_NbKeys; ++i) {
        delete[] m_Keys[i].PosArray;
        delete[] m_Keys[i].NormArray;
    }
    delete[] m_Keys;
}

CKBOOL RCKMorphController::Evaluate(float TimeStep, void *res) {
    // For morph controller, use the full Evaluate with vertex data
    return FALSE;
}

CKBOOL RCKMorphController::Evaluate(float TimeStep, int VertexCount, void *VertexPtr,
                                    CKDWORD VertexStride, VxCompressedVector *NormalPtr) {
    if (m_NbKeys <= 0 || VertexCount <= 0)
        return FALSE;

    // Find the key index
    int keyIdx = -1;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep >= TimeStep) {
            keyIdx = i;
            break;
        }
    }

    if (keyIdx < 0) {
        // Past last key - use last key
        keyIdx = m_NbKeys - 1;
        CKMorphKey &key = m_Keys[keyIdx];

        if (VertexPtr && key.PosArray) {
            if (VertexStride == sizeof(VxVector)) {
                memcpy(VertexPtr, key.PosArray, VertexCount * sizeof(VxVector));
            } else {
                VxCopyStructure(VertexCount, VertexPtr, VertexStride, sizeof(VxVector), key.PosArray, sizeof(VxVector));
            }
        }

        if (NormalPtr && key.NormArray) {
            memcpy(NormalPtr, key.NormArray, VertexCount * sizeof(VxCompressedVector));
        }

        return TRUE;
    }

    if (keyIdx == 0) {
        // Before or at first key
        CKMorphKey &key = m_Keys[0];

        if (VertexPtr && key.PosArray) {
            if (VertexStride == sizeof(VxVector)) {
                memcpy(VertexPtr, key.PosArray, VertexCount * sizeof(VxVector));
            } else {
                VxCopyStructure(VertexCount, VertexPtr, VertexStride, sizeof(VxVector), key.PosArray, sizeof(VxVector));
            }
        }

        if (NormalPtr && key.NormArray) {
            memcpy(NormalPtr, key.NormArray, VertexCount * sizeof(VxCompressedVector));
        }

        return TRUE;
    }

    // Interpolate between keys
    CKMorphKey &key1 = m_Keys[keyIdx - 1];
    CKMorphKey &key2 = m_Keys[keyIdx];

    float t1 = key1.TimeStep;
    float t2 = key2.TimeStep;
    float t = (TimeStep - t1) / (t2 - t1);

    if (VertexPtr && key1.PosArray && key2.PosArray) {
        char *destPtr = static_cast<char *>(VertexPtr);
        for (int i = 0; i < VertexCount; ++i) {
            VxVector *dest = reinterpret_cast<VxVector *>(destPtr);
            VxVector &p1 = key1.PosArray[i];
            VxVector &p2 = key2.PosArray[i];

            dest->x = p1.x + (p2.x - p1.x) * t;
            dest->y = p1.y + (p2.y - p1.y) * t;
            dest->z = p1.z + (p2.z - p1.z) * t;

            destPtr += VertexStride;
        }
    }

    if (NormalPtr && key1.NormArray && key2.NormArray) {
        // Interpolate compressed normals
        for (int i = 0; i < VertexCount; ++i) {
            // Simple interpolation of compressed normal components
            VxCompressedVector &n1 = key1.NormArray[i];
            VxCompressedVector &n2 = key2.NormArray[i];

            // Decompress, interpolate, and recompress
            // This is a simplified version - the original may use more sophisticated interpolation
            int xa = (int) (n1.xa + (n2.xa - n1.xa) * t);
            int ya = (int) (n1.ya + (n2.ya - n1.ya) * t);

            NormalPtr[i].xa = (short) xa;
            NormalPtr[i].ya = (short) ya;
        }
    }

    return TRUE;
}

int RCKMorphController::AddKey(float TimeStep, CKBOOL AllocateNormals) {
    CKMorphKey newKey;
    newKey.TimeStep = TimeStep;
    newKey.PosArray = nullptr;
    newKey.NormArray = nullptr;

    if (m_VertexCount > 0) {
        newKey.PosArray = new VxVector[m_VertexCount];
        memset(newKey.PosArray, 0, m_VertexCount * sizeof(VxVector));

        if (AllocateNormals) {
            newKey.NormArray = new VxCompressedVector[m_VertexCount];
            memset(newKey.NormArray, 0, m_VertexCount * sizeof(VxCompressedVector));
        }
    }

    return AddKey(&newKey, AllocateNormals);
}

int RCKMorphController::AddKey(CKKey *key, CKBOOL AllocateNormals) {
    if (!key)
        return -1;

    CKMorphKey *morphKey = static_cast<CKMorphKey *>(key);
    float time = morphKey->TimeStep;

    int insertIdx = m_NbKeys;
    for (int i = 0; i < m_NbKeys; ++i) {
        if (m_Keys[i].TimeStep == time) {
            // Replace existing key
            delete[] m_Keys[i].PosArray;
            delete[] m_Keys[i].NormArray;
            m_Keys[i] = *morphKey;
            return i;
        }
        if (m_Keys[i].TimeStep > time) {
            insertIdx = i;
            break;
        }
    }

    ++m_NbKeys;
    CKMorphKey *newKeys = new CKMorphKey[m_NbKeys];

    if (m_Keys) {
        if (insertIdx > 0)
            memcpy(newKeys, m_Keys, insertIdx * sizeof(CKMorphKey));
        if (insertIdx < m_NbKeys - 1)
            memcpy(&newKeys[insertIdx + 1], &m_Keys[insertIdx], (m_NbKeys - 1 - insertIdx) * sizeof(CKMorphKey));
        delete[] m_Keys;
    }

    m_Keys = newKeys;
    m_Keys[insertIdx] = *morphKey;

    return insertIdx;
}

CKKey *RCKMorphController::GetKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return nullptr;
    return &m_Keys[index];
}

void RCKMorphController::RemoveKey(int index) {
    if (index < 0 || index >= m_NbKeys)
        return;

    // Free memory for the key being removed
    delete[] m_Keys[index].PosArray;
    delete[] m_Keys[index].NormArray;

    --m_NbKeys;
    if (m_NbKeys == 0) {
        delete[] m_Keys;
        m_Keys = nullptr;
        return;
    }

    CKMorphKey *newKeys = new CKMorphKey[m_NbKeys];

    if (index > 0)
        memcpy(newKeys, m_Keys, index * sizeof(CKMorphKey));
    if (index < m_NbKeys)
        memcpy(&newKeys[index], &m_Keys[index + 1], (m_NbKeys - index) * sizeof(CKMorphKey));

    delete[] m_Keys;
    m_Keys = newKeys;
}

int RCKMorphController::DumpKeysTo(void *Buffer) {
    // Calculate size: count + vertex count + key data
    int size = sizeof(int) * 2; // m_NbKeys + m_VertexCount

    for (int i = 0; i < m_NbKeys; ++i) {
        size += sizeof(float);  // TimeStep
        size += sizeof(CKBOOL); // has normals flag
        if (m_Keys[i].PosArray)
            size += m_VertexCount * sizeof(VxVector);
        if (m_Keys[i].NormArray)
            size += m_VertexCount * sizeof(VxCompressedVector);
    }

    if (Buffer) {
        char *buf = static_cast<char *>(Buffer);

        *reinterpret_cast<int *>(buf) = m_NbKeys;
        buf += sizeof(int);

        *reinterpret_cast<int *>(buf) = m_VertexCount;
        buf += sizeof(int);

        for (int i = 0; i < m_NbKeys; ++i) {
            *reinterpret_cast<float *>(buf) = m_Keys[i].TimeStep;
            buf += sizeof(float);

            CKBOOL hasNormals = (m_Keys[i].NormArray != nullptr);
            *reinterpret_cast<CKBOOL *>(buf) = hasNormals;
            buf += sizeof(CKBOOL);

            if (m_Keys[i].PosArray) {
                memcpy(buf, m_Keys[i].PosArray, m_VertexCount * sizeof(VxVector));
                buf += m_VertexCount * sizeof(VxVector);
            }

            if (hasNormals && m_Keys[i].NormArray) {
                memcpy(buf, m_Keys[i].NormArray, m_VertexCount * sizeof(VxCompressedVector));
                buf += m_VertexCount * sizeof(VxCompressedVector);
            }
        }
    }

    return size;
}

int RCKMorphController::ReadKeysFrom(void *Buffer) {
    if (!Buffer)
        return 0;

    // Clean up existing data
    for (int i = 0; i < m_NbKeys; ++i) {
        delete[] m_Keys[i].PosArray;
        delete[] m_Keys[i].NormArray;
    }
    delete[] m_Keys;
    m_Keys = nullptr;
    m_NbKeys = 0;

    char *buf = static_cast<char *>(Buffer);

    m_NbKeys = *reinterpret_cast<int *>(buf);
    buf += sizeof(int);

    m_VertexCount = *reinterpret_cast<int *>(buf);
    buf += sizeof(int);

    if (m_NbKeys > 0) {
        m_Keys = new CKMorphKey[m_NbKeys];

        for (int i = 0; i < m_NbKeys; ++i) {
            m_Keys[i].TimeStep = *reinterpret_cast<float *>(buf);
            buf += sizeof(float);

            CKBOOL hasNormals = *reinterpret_cast<CKBOOL *>(buf);
            buf += sizeof(CKBOOL);

            m_Keys[i].PosArray = new VxVector[m_VertexCount];
            memcpy(m_Keys[i].PosArray, buf, m_VertexCount * sizeof(VxVector));
            buf += m_VertexCount * sizeof(VxVector);

            if (hasNormals) {
                m_Keys[i].NormArray = new VxCompressedVector[m_VertexCount];
                memcpy(m_Keys[i].NormArray, buf, m_VertexCount * sizeof(VxCompressedVector));
                buf += m_VertexCount * sizeof(VxCompressedVector);
            } else {
                m_Keys[i].NormArray = nullptr;
            }
        }
    }

    return static_cast<int>(buf - static_cast<char *>(Buffer));
}

CKBOOL RCKMorphController::Compare(CKAnimController *control, float Threshold) {
    if (!control || control->GetType() != m_Type)
        return FALSE;

    RCKMorphController *other = static_cast<RCKMorphController *>(control);
    if (m_NbKeys != other->m_NbKeys)
        return FALSE;
    if (m_VertexCount != other->m_VertexCount)
        return FALSE;

    for (int i = 0; i < m_NbKeys; ++i) {
        if (!m_Keys[i].Compare(other->m_Keys[i], m_VertexCount, Threshold))
            return FALSE;
    }

    return TRUE;
}

CKBOOL RCKMorphController::Clone(CKAnimController *control) {
    if (!CKAnimController::Clone(control))
        return FALSE;

    RCKMorphController *other = static_cast<RCKMorphController *>(control);

    // Clean up existing data
    for (int i = 0; i < m_NbKeys; ++i) {
        delete[] m_Keys[i].PosArray;
        delete[] m_Keys[i].NormArray;
    }
    delete[] m_Keys;
    m_Keys = nullptr;

    m_VertexCount = other->m_VertexCount;

    if (other->m_NbKeys > 0) {
        m_Keys = new CKMorphKey[other->m_NbKeys];

        for (int i = 0; i < other->m_NbKeys; ++i) {
            m_Keys[i].TimeStep = other->m_Keys[i].TimeStep;

            if (other->m_Keys[i].PosArray && m_VertexCount > 0) {
                m_Keys[i].PosArray = new VxVector[m_VertexCount];
                memcpy(m_Keys[i].PosArray, other->m_Keys[i].PosArray, m_VertexCount * sizeof(VxVector));
            } else {
                m_Keys[i].PosArray = nullptr;
            }

            if (other->m_Keys[i].NormArray && m_VertexCount > 0) {
                m_Keys[i].NormArray = new VxCompressedVector[m_VertexCount];
                memcpy(m_Keys[i].NormArray, other->m_Keys[i].NormArray, m_VertexCount * sizeof(VxCompressedVector));
            } else {
                m_Keys[i].NormArray = nullptr;
            }
        }
    }

    return TRUE;
}

void RCKMorphController::SetMorphVertexCount(int count) {
    if (count == m_VertexCount)
        return;

    // Resize all existing keys
    for (int i = 0; i < m_NbKeys; ++i) {
        VxVector *newPosArray = nullptr;
        VxCompressedVector *newNormArray = nullptr;

        if (count > 0) {
            newPosArray = new VxVector[count];
            memset(newPosArray, 0, count * sizeof(VxVector));

            if (m_Keys[i].PosArray && m_VertexCount > 0) {
                int copyCount = (count < m_VertexCount) ? count : m_VertexCount;
                memcpy(newPosArray, m_Keys[i].PosArray, copyCount * sizeof(VxVector));
            }

            if (m_Keys[i].NormArray && m_VertexCount > 0) {
                newNormArray = new VxCompressedVector[count];
                memset(newNormArray, 0, count * sizeof(VxCompressedVector));

                int copyCount = (count < m_VertexCount) ? count : m_VertexCount;
                memcpy(newNormArray, m_Keys[i].NormArray, copyCount * sizeof(VxCompressedVector));
            }
        }

        delete[] m_Keys[i].PosArray;
        delete[] m_Keys[i].NormArray;

        m_Keys[i].PosArray = newPosArray;
        m_Keys[i].NormArray = newNormArray;
    }

    m_VertexCount = count;
}
