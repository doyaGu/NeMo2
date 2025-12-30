#include "RCKVertexBuffer.h"

#include "CKRasterizer.h"
#include "RCKRenderManager.h"
#include "RCKRenderContext.h"

RCKVertexBuffer::RCKVertexBuffer(CKContext *context) : CKVertexBuffer(), m_Desc(), m_MemoryPool() {
    m_CKContext = context;
    RCKRenderManager *rm = (RCKRenderManager *) m_CKContext->GetRenderManager();
    m_ObjectIndex = rm->CreateObjectIndex(CKRST_OBJ_VERTEXBUFFER);
    m_DpData.Flags = 0;
    m_Valid = FALSE;
}

RCKVertexBuffer::~RCKVertexBuffer() {
    RCKRenderManager *rm = (RCKRenderManager *) m_CKContext->GetRenderManager();
    rm->ReleaseObjectIndex(m_ObjectIndex, CKRST_OBJ_VERTEXBUFFER);
}

void RCKVertexBuffer::Destroy() {
    RCKRenderManager *rm = (RCKRenderManager *) m_CKContext->GetRenderManager();
    rm->DestroyVertexBuffer(this);
}

CKVB_STATE RCKVertexBuffer::Check(CKRenderContext *Ctx, CKDWORD MaxVertexCount, CKRST_DPFLAGS Format, CKBOOL Dynamic) {
    CKRasterizerContext *rstCtx = Ctx->GetRasterizerContext();
    CKVertexBufferDesc *vertexBuffer = rstCtx->GetVertexBufferData(m_ObjectIndex);
    CKDWORD vertexSize = 0;
    CKDWORD vertexFormat = CKRSTGetVertexFormat(Format, vertexSize);

    bool incompatible = (Format != m_DpData.Flags || vertexFormat != m_Desc.m_VertexFormat || MaxVertexCount > m_Desc.m_MaxVertexCount);
    CKVB_STATE state = (incompatible) ? CK_VB_LOST : CK_VB_OK;
    if (vertexBuffer && incompatible) {
        rstCtx->DeleteObject(m_ObjectIndex, CKRST_OBJ_VERTEXBUFFER);
        vertexBuffer = nullptr;
    }

    m_Desc.m_VertexFormat = vertexFormat;
    m_Desc.m_VertexSize = vertexSize;
    m_Desc.m_MaxVertexCount = MaxVertexCount;
    m_DpData.Flags = Format;

    if (!vertexBuffer) {
        m_Desc.m_Flags = CKRST_DP_DOCLIP;
        if ((Format & CKRST_DP_DOCLIP) != 0) {
            m_Desc.m_Flags |= CKRST_DP_DIFFUSE;
        }
        if (Dynamic) {
            m_Desc.m_Flags |= 0x8;
        }
        if (rstCtx->CreateObject(m_ObjectIndex, CKRST_OBJ_VERTEXBUFFER, &m_Desc)) {
            m_Valid = TRUE;
            return CK_VB_LOST;
        }
        if (m_Valid)
            state = CK_VB_LOST;
        m_Valid = FALSE;
        m_MemoryPool.Allocate((int) ((m_Desc.m_VertexSize * m_Desc.m_MaxVertexCount) >> 2) + 1);
    }
    return state;
}

VxDrawPrimitiveData *RCKVertexBuffer::Lock(CKRenderContext *Ctx, CKDWORD StartVertex, CKDWORD VertexCount, CKLOCKFLAGS LockFlags) {
    CKBYTE *mem;
    if (m_Valid) {
        CKRasterizerContext *rstCtx = Ctx->GetRasterizerContext();
        mem = (CKBYTE *) rstCtx->LockVertexBuffer(m_ObjectIndex, VertexCount, LockFlags);
    } else {
        mem = (CKBYTE *) m_MemoryPool.Buffer() + (StartVertex * m_Desc.m_VertexSize);
    }

    m_DpData.VertexCount = (int) VertexCount;
    CKRSTSetupDPFromVertexBuffer(mem, &m_Desc, m_DpData);
    return &m_DpData;
}

void RCKVertexBuffer::Unlock(CKRenderContext *Ctx) {
    if (m_Valid) {
        CKRasterizerContext *rstCtx = Ctx->GetRasterizerContext();
        rstCtx->UnlockVertexBuffer(m_ObjectIndex);
    }
}

CKBOOL RCKVertexBuffer::Draw(CKRenderContext *Ctx, VXPRIMITIVETYPE pType, CKWORD *Indices, int IndexCount, CKDWORD StartVertex, CKDWORD VertexCount) {
    CKRasterizerContext *rstCtx = Ctx->GetRasterizerContext();

    if (VertexCount == 0)
        return FALSE;

    if ((m_DpData.Flags & CKRST_DP_LIGHT) != 0) {
        rstCtx->SetRenderState(VXRENDERSTATE_LIGHTING, 1);
    } else {
        rstCtx->SetRenderState(VXRENDERSTATE_LIGHTING, 0);
    }

    if (!Indices)
        IndexCount = (int) VertexCount;

    VxStats &stats = ((RCKRenderContext *) Ctx)->GetStats();
    switch (pType) {
    case VX_POINTLIST:
        stats.NbPointsDrawn += (int) VertexCount;
        break;
    case VX_LINELIST:
        stats.NbLinesDrawn += IndexCount >> 1;
        break;
    case VX_LINESTRIP:
        stats.NbLinesDrawn = IndexCount + stats.NbLinesDrawn - 1;
        break;
    case VX_TRIANGLELIST:
        stats.NbTrianglesDrawn += IndexCount / 3;
        break;
    case VX_TRIANGLESTRIP:
    case VX_TRIANGLEFAN:
        stats.NbTrianglesDrawn = IndexCount + stats.NbTrianglesDrawn - 2;
        break;
    }
    stats.NbVerticesProcessed += (int) VertexCount;

    if (m_Valid) {
        return rstCtx->DrawPrimitiveVB(pType, m_ObjectIndex, StartVertex, VertexCount, Indices, IndexCount);
    } else {
        CKBYTE *mem = (CKBYTE *) m_MemoryPool.Buffer() + (StartVertex * m_Desc.m_VertexSize);
        m_DpData.VertexCount = (int) VertexCount;
        CKRSTSetupDPFromVertexBuffer(mem, &m_Desc, m_DpData);
        return rstCtx->DrawPrimitive(pType, Indices, IndexCount, &m_DpData);
    }
}
