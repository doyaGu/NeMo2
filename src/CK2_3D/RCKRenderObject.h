#ifndef RCKRENDEROBJECT_H
#define RCKRENDEROBJECT_H

#include "CKRenderEngineTypes.h"

#include "CKRenderObject.h"

class RCKRenderObject : public CKRenderObject {
    friend struct CKSceneGraphNode;
public:
    void AddToRenderContext(CKRenderContext *context);
    void RemoveFromRenderContext(CKRenderContext *context);

    int CanBeHide() override;

    CKBOOL IsInRenderContext(CKRenderContext *context) override;
    CKBOOL IsRootObject() override;
    CKBOOL IsToBeRendered() override;
    void SetZOrder(int Z) override;
    int GetZOrder() override;

    CKBOOL IsToBeRenderedLast() override;

    CKBOOL AddPreRenderCallBack(CK_RENDEROBJECT_CALLBACK Function, void *Argument, CKBOOL Temp) override;
    CKBOOL RemovePreRenderCallBack(CK_RENDEROBJECT_CALLBACK Function, void *Argument) override;

    CKBOOL SetRenderCallBack(CK_RENDEROBJECT_CALLBACK Function, void *Argument) override;
    CKBOOL RemoveRenderCallBack() override;

    CKBOOL AddPostRenderCallBack(CK_RENDEROBJECT_CALLBACK Function, void *Argument, CKBOOL Temp) override;
    CKBOOL RemovePostRenderCallBack(CK_RENDEROBJECT_CALLBACK Function, void *Argument) override;

    void RemoveAllCallbacks() override;

    explicit RCKRenderObject(CKContext *Context, CKSTRING name = nullptr);
    ~RCKRenderObject() override;
    CK_CLASSID GetClassID() override;

    int GetMemoryOccupation() override;

    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static CKRenderObject *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    CKDWORD GetInRenderContextMask() const { return m_InRenderContext; }

protected:
    CKDWORD m_InRenderContext;
    CKCallbacksContainer *m_Callbacks;
};

#endif // RCKRENDEROBJECT_H
