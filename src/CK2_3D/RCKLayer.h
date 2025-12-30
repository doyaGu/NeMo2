#ifndef RCKLAYER_H
#define RCKLAYER_H

#include "CKRenderEngineTypes.h"

#include "CKLayer.h"

class CKGrid;

class RCKLayer : public CKLayer {
public:
    // CKLayer implementation
    void SetType(int Type) override;
    int GetType() override;

    void SetFormat(int Format) override;
    int GetFormat() override;

    void SetValue(int x, int y, void *val) override;
    void GetValue(int x, int y, void *val) override;
    CKBOOL SetValue2(int x, int y, void *val) override;
    CKBOOL GetValue2(int x, int y, void *val) override;

    CKSquare *GetSquareArray() override;
    void SetSquareArray(CKSquare *sqarray) override;

    void SetVisible(CKBOOL vis = TRUE) override;
    CKBOOL IsVisible() override;

    void InitOwner(CK_ID owner) override;
    void SetOwner(CK_ID owner) override;
    CK_ID GetOwner() override;

    RCKLayer(CKContext *Context, CKSTRING name, CK_ID owner);
    ~RCKLayer() override;
    CK_CLASSID GetClassID() override;

    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    int GetMemoryOccupation() override;

    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;

    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKLayer *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

protected:
    CKGrid *m_Grid;
    int m_Type;
    int m_Format;
    CKDWORD m_Flags;
    CKSquare *m_SquareArray;
};

#endif // RCKLAYER_H
