#ifndef RCKKEYFRAMEDATA_H
#define RCKKEYFRAMEDATA_H

#include "CKKeyframeData.h"

//===================================================================
// CKKeyframeData - Container for animation controllers
// Total size: 32 bytes (0x20) per IDA struct analysis
//===================================================================
struct CKKeyframeData
{
    CKAnimController *m_PositionController;   // 0x00
    CKAnimController *m_ScaleController;      // 0x04
    CKAnimController *m_RotationController;   // 0x08
    CKAnimController *m_ScaleAxisController;  // 0x0C
    CKMorphController *m_MorphController;     // 0x10
    float m_Length;                           // 0x14
    int m_RefCount;                           // 0x18 - Reference count for shared keyframe data (was mislabeled as m_Flags in IDA)
    class CKObjectAnimation *m_ObjectAnimation; // 0x1C - Owner animation

    CKKeyframeData();
    ~CKKeyframeData();

    // Factory method to create a controller
    CKAnimController *CreateController(CKANIMATION_CONTROLLER type);
};

//===================================================================
// Linear Position Controller
//===================================================================
class RCKLinearPositionController : public CKAnimController
{
public:
    RCKLinearPositionController();
    virtual ~RCKLinearPositionController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

protected:
    CKPositionKey *m_Keys;
};

//===================================================================
// Linear Rotation Controller
//===================================================================
class RCKLinearRotationController : public CKAnimController
{
public:
    RCKLinearRotationController();
    virtual ~RCKLinearRotationController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

protected:
    CKRotationKey *m_Keys;
};

//===================================================================
// Linear Scale Controller
//===================================================================
class RCKLinearScaleController : public CKAnimController
{
public:
    RCKLinearScaleController();
    virtual ~RCKLinearScaleController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

protected:
    CKScaleKey *m_Keys;
};

//===================================================================
// Linear Scale Axis Controller
//===================================================================
class RCKLinearScaleAxisController : public CKAnimController
{
public:
    RCKLinearScaleAxisController();
    virtual ~RCKLinearScaleAxisController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

protected:
    CKScaleAxisKey *m_Keys;
};

//===================================================================
// TCB Position Controller
//===================================================================
class RCKTCBPositionController : public CKAnimController
{
public:
    RCKTCBPositionController();
    virtual ~RCKTCBPositionController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

    // Computes tangents for spline interpolation
    void ComputeTangents();

protected:
    CKTCBPositionKey *m_Keys;
    // Precomputed tangent data (2 tangents per key)
    VxVector *m_Tangents;
};

//===================================================================
// TCB Rotation Controller
//===================================================================
class RCKTCBRotationController : public CKAnimController
{
public:
    RCKTCBRotationController();
    virtual ~RCKTCBRotationController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

    // Computes tangents for spline interpolation
    void ComputeTangents();

protected:
    CKTCBRotationKey *m_Keys;
    // Precomputed tangent data
    VxQuaternion *m_Tangents;
};

//===================================================================
// TCB Scale Controller
//===================================================================
class RCKTCBScaleController : public CKAnimController
{
public:
    RCKTCBScaleController();
    virtual ~RCKTCBScaleController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

    void ComputeTangents();

protected:
    CKTCBScaleKey *m_Keys;
    VxVector *m_Tangents;
};

//===================================================================
// TCB Scale Axis Controller
//===================================================================
class RCKTCBScaleAxisController : public CKAnimController
{
public:
    RCKTCBScaleAxisController();
    virtual ~RCKTCBScaleAxisController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

    void ComputeTangents();

protected:
    CKTCBScaleAxisKey *m_Keys;
    VxQuaternion *m_Tangents;
};

//===================================================================
// Bezier Position Controller
//===================================================================
class RCKBezierPositionController : public CKAnimController
{
public:
    RCKBezierPositionController();
    virtual ~RCKBezierPositionController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

    // Compute tangent points for Bezier interpolation
    void ComputeBezierPts();
    void ComputeBezierPts(int index);

protected:
    // Helper to compute distance between two keys
    float ComputeKeyDistance(int key1, int key2);

    CKBezierPositionKey *m_Keys;
    CKBOOL m_TangentsComputed;
};

//===================================================================
// Bezier Scale Controller
//===================================================================
class RCKBezierScaleController : public CKAnimController
{
public:
    RCKBezierScaleController();
    virtual ~RCKBezierScaleController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual int AddKey(CKKey *key);
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);

    void ComputeBezierPts();
    void ComputeBezierPts(int index);

protected:
    float ComputeKeyDistance(int key1, int key2);

    CKBezierScaleKey *m_Keys;
    CKBOOL m_TangentsComputed;
};

//===================================================================
// Morph Controller
//===================================================================
class RCKMorphController : public CKMorphController
{
public:
    RCKMorphController();
    virtual ~RCKMorphController();

    virtual CKBOOL Evaluate(float TimeStep, void *res);
    virtual CKBOOL Evaluate(float TimeStep, int VertexCount, void *VertexPtr, CKDWORD VertexStride, VxCompressedVector *NormalPtr);
    virtual int AddKey(float TimeStep, CKBOOL AllocateNormals);
    virtual int AddKey(CKKey *key, CKBOOL AllocateNormals);
    virtual int AddKey(CKKey *key) { return AddKey(key, TRUE); }
    virtual CKKey *GetKey(int index);
    virtual void RemoveKey(int index);
    virtual int DumpKeysTo(void *Buffer);
    virtual int ReadKeysFrom(void *Buffer);
    virtual CKBOOL Compare(CKAnimController *control, float Threshold = 0.0f);
    virtual CKBOOL Clone(CKAnimController *control);
    virtual void SetMorphVertexCount(int count);
    
    // Get the vertex count for morph data
    int GetMorphVertexCount() const { return m_VertexCount; }

protected:
    CKMorphKey *m_Keys;
    int m_VertexCount;
};

#endif // RCKKEYFRAMEDATA_H
