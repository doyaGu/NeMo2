#ifndef RCKPLACE_H
#define RCKPLACE_H

#include "RCK3dEntity.h"

// Portal entry - pair of destination place and portal geometry
struct CKPortalEntry {
    CKPlace *place;
    CK3dEntity *portal;

    bool operator==(const CKPortalEntry &other) const {
        return (place == other.place) && (portal == other.portal);
    }
};

class RCKPlace : public RCK3dEntity {
public:

#undef CK_PURE
#define CK_3DIMPLEMENTATION
#include "CKPlace.h"
#undef CK_3DIMPLEMENTATION

    explicit RCKPlace(CKContext *Context, CKSTRING name = nullptr);
    ~RCKPlace() override;
    CK_CLASSID GetClassID() override;

    // CKObject overrides
    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;
    void CheckPreDeletion() override;
    void CheckPostDeletion() override;
    int GetMemoryOccupation() override;
    CKBOOL IsObjectUsed(CKObject *o, CK_CLASSID cid) override;

    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    // CKBeObject overrides
    void AddToScene(CKScene *scene, CKBOOL dependencies) override;
    void RemoveFromScene(CKScene *scene, CKBOOL dependencies) override;

    // CK3dEntity overrides
    const VxBbox &GetBoundingBox(CKBOOL local = FALSE) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKPlace *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    XSArray<CKPortalEntry> m_Portals;
    CK_ID m_Level;
    CK_ID m_Camera;
    VxRect m_ClippingRect;
};

#endif // RCKPLACE_H
