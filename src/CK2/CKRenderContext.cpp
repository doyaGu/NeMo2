#include "CKRenderContext.h"

CKRenderContext::CKRenderContext(CKContext *Context, CKSTRING name) : CKObject(Context, name) {
	if (!Context->m_BehaviorContext.CurrentRenderContext)
		Context->m_BehaviorContext.CurrentRenderContext = this;
}

CKRenderContext::~CKRenderContext() {
    if (m_Context->m_BehaviorContext.CurrentRenderContext == this)
        m_Context->m_BehaviorContext.CurrentRenderContext = nullptr;
}
