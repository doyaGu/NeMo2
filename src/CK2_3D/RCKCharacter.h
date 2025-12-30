#ifndef RCKCHARACTER_H
#define RCKCHARACTER_H

#include "RCK3dEntity.h"
#include "XObjectArray.h"

typedef enum CK_SECONDARYANIMATION_RUNTIME_MODE {
    CKSECONDARYANIMATIONRUNTIME_STARTINGWARP = 1,
    CKSECONDARYANIMATIONRUNTIME_PLAYING = 2,
    CKSECONDARYANIMATIONRUNTIME_STOPPINGWARP = 3,
} CK_SECONDARYANIMATION_RUNTIME_MODE;

// Layout verified against IDA decompilation (28 bytes total)
// PlaySecondaryAnimation allocates: 28 * (count + 2) bytes
struct CKSecondaryAnimation {
    CK_ID AnimID;                              // +0x00: CK_ID of the secondary animation
    RCKKeyedAnimation *Animation;              // +0x04: Transition animation (warper) when warping
    CKDWORD Flags;                             // +0x08: CK_SECONDARYANIMATION_FLAGS bitmask
    float WarpLength;                          // +0x0C: Warp duration in frames
    CKDWORD Padding;                           // +0x10: Unused padding
    CKDWORD StartingFrameBits;                 // +0x14: Raw float bits of starting frame (field_14)
    CK_SECONDARYANIMATION_RUNTIME_MODE Mode;   // +0x18: Runtime mode
    CKDWORD LoopCountRemaining;                // +0x1C: Loop count when LOOPNTIMES is set

    float GetStartingFrame() const {
        float frame;
        memcpy(&frame, &StartingFrameBits, sizeof(frame));
        return frame;
    }

    void SetStartingFrame(float frame) {
        memcpy(&StartingFrameBits, &frame, sizeof(frame));
    }
};

class RCKCharacter : public RCK3dEntity {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKCharacter.h"
#undef CK_3DIMPLEMENTATION

    //--------------------------------------------------------
    ////               Private Part

    //----------------------------------------------------------
    // Internal functions

    explicit RCKCharacter(CKContext *Context, CKSTRING name = nullptr);
    ~RCKCharacter() override;
    CK_CLASSID GetClassID() override;

    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    void CheckPreDeletion() override;

    int GetMemoryOccupation() override;
    int IsObjectUsed(CKObject *o, CK_CLASSID cid) override;

    void Show(CK_OBJECT_SHOWOPTION show) override;

    //--------------------------------------------
    // Dependencies Functions	{Secret}
    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    void AddToScene(CKScene *scene, CKBOOL dependencies = TRUE) override;
    void RemoveFromScene(CKScene *scene, CKBOOL dependencies = TRUE) override;

    const VxBbox &GetBoundingBox(CKBOOL Local = FALSE) override;
    CKBOOL GetBaryCenter(VxVector *Pos) override;
    float GetRadius() override;

    //--------------------------------------------
    // Class Registering	{Secret}
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKCharacter *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    void PreDeleteBodyPartsForAnimation(CKAnimation *anim);
    void FindFloorReference();
    void RemoveSecondaryAnimationAt(int index);

    XSObjectPointerArray m_BodyParts;      // Stores CKObject* (body parts)
    XSObjectPointerArray m_Animations;     // Stores CKObject* (animations)
    CKSecondaryAnimation *m_SecondaryAnimations;
    CKWORD m_SecondaryAnimationsCount;
    CKWORD m_SecondaryAnimationsAllocated;
    RCKBodyPart *m_RootBodyPart;
    RCKKeyedAnimation *m_ActiveAnimation;
    RCKAnimation *m_AnimDest;
    RCKKeyedAnimation *m_Warper;
    float m_FrameDest;
    CKDWORD field_1D4;
    RCK3dEntity *m_FloorRef;
    float m_AnimationLevelOfDetail;
    float m_FrameSrc;
    RCKAnimation *m_AnimSrc;
    CKDWORD m_TransitionMode;
};

#endif // RCKCHARACTER_H
