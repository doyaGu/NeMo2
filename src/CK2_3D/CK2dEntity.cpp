#include <CKTexture.h>

#include "RCK2dEntity.h"
#include "RCKRenderContext.h"
#include "CKRasterizer.h"
#include "CKSprite.h"

// External function from CKMeshUtils.cpp
extern CKBOOL PreciseTexturePick(CKMaterial *mat, float u, float v);

// IDA: sub_1005C750 - Comparison function for sorting children by ZOrder
// Compares two CK2dEntity* pointers by their m_ZOrder (at offset 0xB4 = 180)
static int CompareByZOrder(const void *a, const void *b) {
    CK2dEntity *ent1 = *(CK2dEntity **) a;
    CK2dEntity *ent2 = *(CK2dEntity **) b;
    return ent1->GetZOrder() - ent2->GetZOrder();
}

CK_CLASSID RCK2dEntity::m_ClassID = CKCID_2DENTITY;

CK_CLASSID RCK2dEntity::GetClassID() {
    return m_ClassID;
}

RCK2dEntity::RCK2dEntity(CKContext *Context, CKSTRING name) : RCKRenderObject(Context, name) {
    m_Flags = CK_2DENTITY_RESERVED3 | CK_2DENTITY_RATIOOFFSET | CK_2DENTITY_CLIPTOCAMERAVIEW | CK_2DENTITY_STICKLEFT | CK_2DENTITY_STICKTOP;
    m_Parent = nullptr;
    m_SourceRect = VxRect(0.0f, 0.0f, 1.0f, 1.0f);
    m_HomogeneousRect = nullptr;
    m_ZOrder = 0;
    m_Material = nullptr;
}

RCK2dEntity::~RCK2dEntity() {
    if (m_HomogeneousRect)
        delete m_HomogeneousRect;
    m_Children.Clear();
}

CKERROR RCK2dEntity::GetPosition(Vx2DVector &vect, CKBOOL hom, CK2dEntity *ref) {
    if (hom) {
        if (!(m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) || !m_HomogeneousRect)
            return CKERR_INVALIDPARAMETER;
        vect.x = m_HomogeneousRect->left;
        vect.y = m_HomogeneousRect->top;
    } else {
        vect.x = m_Rect.left;
        vect.y = m_Rect.top;
        if (ref) {
            VxRect rect;
            ref->GetRect(rect);
            vect -= rect.GetTopLeft();
        }
    }
    return CK_OK;
}

void RCK2dEntity::SetPosition(const Vx2DVector &vect, CKBOOL hom, CKBOOL KeepChildren, CK2dEntity *ref) {
    if (hom) {
        if ((m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) && m_HomogeneousRect) {
            m_HomogeneousRect->Move(vect);
            SetHomogeneousRect(*m_HomogeneousRect, KeepChildren);
        }
    } else {
        if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
            m_Flags |= CK_2DENTITY_UPDATEHOMOGENEOUSCOORD;
        }

        Vx2DVector pos = vect;
        if (ref) {
            VxRect rect;
            ref->GetRect(rect);
            pos += rect.GetTopLeft();
        }

        VxRect newRect = m_Rect;
        newRect.Move(pos);
        SetRect(newRect, KeepChildren);
    }
}

CKERROR RCK2dEntity::GetSize(Vx2DVector &vect, CKBOOL hom) {
    if (hom) {
        if (!(m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) || !m_HomogeneousRect)
            return CKERR_INVALIDPARAMETER;
        vect.x = m_HomogeneousRect->GetWidth();
        vect.y = m_HomogeneousRect->GetHeight();
    } else {
        vect.x = m_Rect.GetWidth();
        vect.y = m_Rect.GetHeight();
    }
    return CK_OK;
}

void RCK2dEntity::SetSize(const Vx2DVector &vect, CKBOOL hom, CKBOOL KeepChildren) {
    if (hom) {
        if ((m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) && m_HomogeneousRect) {
            m_HomogeneousRect->SetSize(vect);
            SetHomogeneousRect(*m_HomogeneousRect, KeepChildren);
        }
    } else {
        if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
            m_Flags |= CK_2DENTITY_UPDATEHOMOGENEOUSCOORD;
        }
        VxRect newRect = m_Rect;
        newRect.SetSize(vect);
        SetRect(newRect, KeepChildren);
    }
}

void RCK2dEntity::SetRect(const VxRect &rect, CKBOOL KeepChildren) {
    // Handle homogeneous coordinate system
    if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
        VxRect relRect;
        GetHomogeneousRelativeRect(relRect);
        *m_HomogeneousRect = rect;
        m_HomogeneousRect->TransformToHomogeneous(relRect);
    }

    if (!KeepChildren) {
        for (auto it = m_Children.Begin(); it != m_Children.End(); ++it) {
            RCK2dEntity *child = (RCK2dEntity *) *it;
            VxRect childRect;

            if (child->m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
                // Handle child with homogeneous coords
                childRect = *child->m_HomogeneousRect;
                VxRect childRelRect;
                child->GetHomogeneousRelativeRect(childRelRect);
                childRect.TransformFromHomogeneous(childRelRect);
            } else {
                float childWidth = child->m_Rect.GetWidth();
                float childHeight = child->m_Rect.GetHeight();

                if (child->m_Flags & CK_2DENTITY_STICKLEFT) {
                    childRect.left = child->m_Rect.left - m_Rect.left + rect.left;
                    if (child->m_Flags & CK_2DENTITY_STICKRIGHT) {
                        childRect.right = rect.right - (m_Rect.right - child->m_Rect.right);
                    } else {
                        childRect.right = childRect.left + childWidth;
                    }
                } else if (child->m_Flags & CK_2DENTITY_STICKRIGHT) {
                    childRect.right = rect.right - (m_Rect.right - child->m_Rect.right);
                    childRect.left = childRect.right - childWidth;
                } else {
                    float centerX = (child->m_Rect.left + child->m_Rect.right) * 0.5f;
                    float ratio = (centerX - m_Rect.left) / m_Rect.GetWidth();
                    float newCenter = rect.GetWidth() * ratio + rect.left;
                    childRect.left = newCenter - childWidth * 0.5f;
                    childRect.right = childRect.left + childWidth;
                }

                if (child->m_Flags & CK_2DENTITY_STICKTOP) {
                    childRect.top = child->m_Rect.top - m_Rect.top + rect.top;
                    if (child->m_Flags & CK_2DENTITY_STICKBOTTOM) {
                        childRect.bottom = rect.bottom - (m_Rect.bottom - child->m_Rect.bottom);
                    } else {
                        childRect.bottom = childRect.top + childHeight;
                    }
                } else if (child->m_Flags & CK_2DENTITY_STICKBOTTOM) {
                    childRect.bottom = rect.bottom - (m_Rect.bottom - child->m_Rect.bottom);
                    childRect.top = childRect.bottom - childHeight;
                } else {
                    float centerY = (child->m_Rect.top + child->m_Rect.bottom) * 0.5f;
                    float ratio = (centerY - m_Rect.top) / m_Rect.GetHeight();
                    float newCenter = rect.GetHeight() * ratio + rect.top;
                    childRect.top = newCenter - childHeight * 0.5f;
                    childRect.bottom = childRect.top + childHeight;
                }
            }

            child->SetRect(childRect, FALSE);
        }
    }
    m_Rect = rect;
}

void RCK2dEntity::GetRect(VxRect &rect) {
    rect = m_Rect;
}

CKERROR RCK2dEntity::SetHomogeneousRect(const VxRect &rect, CKBOOL KeepChildren) {
    if (!(m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) || !m_HomogeneousRect)
        return CKERR_INVALIDPARAMETER;

    // Homogeneous rect becomes authoritative; keep screen rect in sync.
    *m_HomogeneousRect = rect;

    VxRect relRect;
    GetHomogeneousRelativeRect(relRect);

    VxRect screenRect = *m_HomogeneousRect;
    screenRect.TransformFromHomogeneous(relRect);

    // Use SetRect to preserve child-sticking behavior.
    SetRect(screenRect, KeepChildren);

    return CK_OK;
}

CKERROR RCK2dEntity::GetHomogeneousRect(VxRect &rect) {
    if ((m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) && m_HomogeneousRect) {
        rect = *m_HomogeneousRect;
        return CK_OK;
    }
    return CKERR_INVALIDPARAMETER;
}

void RCK2dEntity::GetHomogeneousRelativeRect(VxRect &rect) {
    CKRenderContext *ctx = m_Context->GetPlayerRenderContext();
    RCK2dEntity *parent = (RCK2dEntity *) GetParent();

    if (parent) {
        parent->GetRect(rect);
        return;
    }

    if (IsRatioOffset()) {
        if (ctx) {
            ctx->GetViewRect(rect);
        }
    } else {
        if (ctx) {
            ctx->GetWindowRect(rect, FALSE);
        }
    }

    // Translate rect so that its top-left is at origin
    Vx2DVector offset(-rect.left, -rect.top);
    rect.Translate(offset);
}

void RCK2dEntity::SetSourceRect(VxRect &rect) {
    m_SourceRect = rect;
}

void RCK2dEntity::GetSourceRect(VxRect &rect) {
    rect = m_SourceRect;
}

void RCK2dEntity::UseSourceRect(CKBOOL Use) {
    if (Use)
        m_Flags |= CK_2DENTITY_USESRCRECT;
    else
        m_Flags &= ~CK_2DENTITY_USESRCRECT;
}

CKBOOL RCK2dEntity::IsUsingSourceRect() {
    return (m_Flags & CK_2DENTITY_USESRCRECT) != 0;
}

void RCK2dEntity::SetPickable(CKBOOL Pick) {
    if (Pick)
        m_Flags &= ~CK_2DENTITY_NOTPICKABLE;
    else
        m_Flags |= CK_2DENTITY_NOTPICKABLE;
}

CKBOOL RCK2dEntity::IsPickable() {
    return (m_Flags & CK_2DENTITY_NOTPICKABLE) == 0;
}

// IDA: sub_1005E1EE - recursively set/clear background flag
void RCK2dEntity::HierarchySetBackground(CKBOOL back) {
    if (back)
        m_Flags |= CK_2DENTITY_BACKGROUND;
    else
        m_Flags &= ~CK_2DENTITY_BACKGROUND;

    // Recursively apply to all children
    for (CK2dEntity **it = m_Children.Begin(); it != m_Children.End(); ++it) {
        ((RCK2dEntity *) *it)->HierarchySetBackground(back);
    }
}

void RCK2dEntity::SetBackground(CKBOOL back) {
    CKRenderContext *rc = m_Context->GetPlayerRenderContext();
    CK2dEntity *foregroundRoot = nullptr;
    CK2dEntity *backgroundRoot = nullptr;

    if (rc) {
        foregroundRoot = rc->Get2dRoot(FALSE);
        backgroundRoot = rc->Get2dRoot(TRUE);
    }

    if (!back) {
        if (!GetParent() && (m_Flags & CK_2DENTITY_BACKGROUND)) {
            SetParent(foregroundRoot);
            HierarchySetBackground(FALSE);
        }
    } else {
        if (!GetParent() && !(m_Flags & CK_2DENTITY_BACKGROUND)) {
            SetParent(backgroundRoot);
            HierarchySetBackground(TRUE);
        }
    }
}

CKBOOL RCK2dEntity::IsBackground() {
    return (m_Flags & CK_2DENTITY_BACKGROUND) != 0;
}

void RCK2dEntity::SetClipToParent(CKBOOL clip) {
    if (clip)
        m_Flags |= CK_2DENTITY_CLIPTOPARENT;
    else
        m_Flags &= ~CK_2DENTITY_CLIPTOPARENT;
}

CKBOOL RCK2dEntity::IsClipToParent() {
    return (m_Flags & CK_2DENTITY_CLIPTOPARENT) != 0;
}

// IDA: 0x1005cabb - Internal Pick method used by RCKRenderContext::_Pick2D
CK2dEntity *RCK2dEntity::Pick(const Vx2DVector &pt, CKBOOL ignoreUnpickable) {
    // IDA line 14-15: Check if hidden by parent hierarchy (ObjectFlags & CK_OBJECT_HIERACHICALHIDE)
    if (IsHiddenByParent())
        return nullptr;

    // IDA line 16-24: Recursively check children in reverse order (back to front)
    CK2dEntity *picked = nullptr;
    for (int i = m_Children.Size() - 1; i >= 0; --i) {
        RCK2dEntity *entity = (RCK2dEntity *) m_Children[i];
        picked = entity->Pick(pt, ignoreUnpickable);
        if (picked)
            return picked;
    }

    // IDA line 25-26: Check pickability
    if (!ignoreUnpickable && (m_Flags & CK_2DENTITY_NOTPICKABLE) != 0)
        return nullptr;

    // IDA line 27-28: Check visibility and interface object flag
    if (!IsVisible())
        return nullptr;
    if ((m_ObjectFlags & CK_OBJECT_INTERFACEOBJ) != 0)
        return nullptr;

    // IDA line 29-30: Check if point is inside m_VtxPos (sub_100600B0 = point-in-rect test)
    if (pt.x < m_VtxPos.left || pt.x > m_VtxPos.right ||
        pt.y < m_VtxPos.top || pt.y > m_VtxPos.bottom)
        return nullptr;

    // IDA line 31-32: If no material, consider hit
    if (!m_Material)
        return (CK2dEntity *) this;

    // IDA line 33: Initialize UV
    Vx2DVector uv;

    // IDA line 34-58: Calculate normalized UV coordinates within the entity
    if ((m_Flags & CK_2DENTITY_RATIOOFFSET) != 0) {
        // IDA line 36-42: With ratio offset, subtract viewport position
        RCKRenderContext *rc = (RCKRenderContext *) m_Context->GetPlayerRenderContext();
        if (rc) {
            uv.x = (pt.x - m_Rect.left - (float) rc->m_ViewportData.ViewX) / m_Rect.GetWidth();
            uv.y = (pt.y - m_Rect.top - (float) rc->m_ViewportData.ViewY) / m_Rect.GetHeight();
        } else {
            uv.x = (pt.x - m_Rect.left) / m_Rect.GetWidth();
            uv.y = (pt.y - m_Rect.top) / m_Rect.GetHeight();
        }
    } else {
        // IDA line 54-57: Without ratio offset
        uv.x = (pt.x - m_Rect.left) / m_Rect.GetWidth();
        uv.y = (pt.y - m_Rect.top) / m_Rect.GetHeight();
    }

    // IDA line 59: Map UV to source rect coordinates (sub_10060070)
    // NOTE: IDA shows: srcU = width * uv.x + left, srcV = height * uv.y + left (BUG in original: uses left instead of top)
    // We reproduce the exact behavior for faithfulness
    float srcU = m_SourceRect.GetWidth() * uv.x + m_SourceRect.left;
    float srcV = m_SourceRect.GetHeight() * uv.y + m_SourceRect.left; // IDA BUG: uses left instead of top

    // IDA line 60-63: Use PreciseTexturePick for alpha test
    if (PreciseTexturePick(m_Material, srcU, srcV))
        return (CK2dEntity *) this;
    
    return nullptr;
}

void RCK2dEntity::SetFlags(CKDWORD Flags) {
    m_Flags = Flags;

    // Keep homogeneous-rect storage consistent with the flag.
    if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
        if (!m_HomogeneousRect)
            m_HomogeneousRect = new VxRect();
    } else {
        if (m_HomogeneousRect) {
            delete m_HomogeneousRect;
            m_HomogeneousRect = nullptr;
        }
        m_Flags &= ~CK_2DENTITY_UPDATEHOMOGENEOUSCOORD;
    }
}

void RCK2dEntity::ModifyFlags(CKDWORD add, CKDWORD remove) {
    m_Flags |= add;
    m_Flags &= ~remove;

    // Keep homogeneous-rect storage consistent with the flag.
    if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
        if (!m_HomogeneousRect)
            m_HomogeneousRect = new VxRect();
    } else {
        if (m_HomogeneousRect) {
            delete m_HomogeneousRect;
            m_HomogeneousRect = nullptr;
        }
        m_Flags &= ~CK_2DENTITY_UPDATEHOMOGENEOUSCOORD;
    }
}

CKDWORD RCK2dEntity::GetFlags() {
    return m_Flags;
}

void RCK2dEntity::EnableRatioOffset(CKBOOL Ratio) {
    if (Ratio)
        m_Flags |= CK_2DENTITY_RATIOOFFSET;
    else
        m_Flags &= ~CK_2DENTITY_RATIOOFFSET;
}

CKBOOL RCK2dEntity::IsRatioOffset() {
    return (m_Flags & CK_2DENTITY_RATIOOFFSET) != 0;
}

CKBOOL RCK2dEntity::SetParent(CK2dEntity *parent) {
    // IDA: 0x1005dd22
    RCK2dEntity *newParent = (RCK2dEntity *) parent;

    // Prevent circular hierarchy
    for (RCK2dEntity *p = newParent; p; p = (RCK2dEntity *) p->m_Parent) {
        if (p == this)
            return FALSE;
    }

    CKRenderContext *rc = m_Context->GetPlayerRenderContext();

    CK2dEntity *foreground = rc ? rc->Get2dRoot(FALSE) : nullptr; // Foreground root
    CK2dEntity *background = rc ? rc->Get2dRoot(TRUE) : nullptr;  // Background root

    // Remove from current parent
    if (m_Parent) {
        ((RCK2dEntity *) m_Parent)->m_Children.Remove((CK2dEntity *) this);
    } else {
        // Remove from roots if no parent
        if (foreground) {
            ((RCK2dEntity *) foreground)->m_Children.Remove((CK2dEntity *) this);
        }
        if (background) {
            ((RCK2dEntity *) background)->m_Children.Remove((CK2dEntity *) this);
        }
    }

    // Check if new parent is null or a root
    if (!newParent || newParent == (RCK2dEntity *) foreground || newParent == (RCK2dEntity *) background) {
        // No real parent, add to appropriate root
        if (!newParent) {
            if (IsBackground()) {
                newParent = (RCK2dEntity *) background;
            } else {
                newParent = (RCK2dEntity *) foreground;
            }
        }
        if (newParent) {
            newParent->m_Children.PushBack((CK2dEntity *) this);
            newParent->m_Children.Sort(CompareByZOrder);
        }
        m_Parent = nullptr;
    } else {
        // Real parent
        newParent->m_Children.PushBack((CK2dEntity *) this);
        newParent->m_Children.Sort(CompareByZOrder);
        if (newParent->IsBackground()) {
            m_Flags |= CK_2DENTITY_BACKGROUND;
        } else {
            m_Flags &= ~CK_2DENTITY_BACKGROUND;
        }
        m_Parent = (CK2dEntity *) newParent;
    }

    return TRUE;
}

CK2dEntity *RCK2dEntity::GetParent() const {
    return m_Parent;
}

int RCK2dEntity::GetChildrenCount() const {
    return m_Children.Size();
}

CK2dEntity *RCK2dEntity::GetChild(int i) const {
    if (i >= 0 && i < m_Children.Size())
        return (CK2dEntity *) m_Children[i];
    return nullptr;
}

CK2dEntity *RCK2dEntity::HierarchyParser(CK2dEntity *current) const {
    // IDA: 0x1005df68
    CKRenderContext *rc = m_Context->GetPlayerRenderContext();
    CK2dEntity *foregroundRoot = rc ? rc->Get2dRoot(FALSE) : nullptr;
    CK2dEntity *backgroundRoot = rc ? rc->Get2dRoot(TRUE) : nullptr;

    if (current) {
        // If current has children, return first child
        if (current->GetChildrenCount() > 0)
            return current->GetChild(0);

        // Walk up the hierarchy
        CK2dEntity *node = current;
        while (true) {
            CK2dEntity *parent;
            if (node->GetParent()) {
                parent = node->GetParent();
            } else {
                // If no parent, use root based on background flag
                parent = ((RCK2dEntity *) node)->IsBackground() ? backgroundRoot : foregroundRoot;
            }

            if (!parent)
                return nullptr;

            // Find index of node in parent's children
            int childCount = parent->GetChildrenCount();
            int index = 0;
            while (index < childCount && parent->GetChild(index) != node) {
                ++index;
            }
            ++index; // Move to next sibling

            // If there's a next sibling, return it
            if (index < childCount)
                return parent->GetChild(index);

            // If we've reached the root of hierarchy, stop
            if (parent == (CK2dEntity *) this)
                return nullptr;

            // Otherwise, continue up
            node = parent;
        }
    } else {
        // If no current, return first child of this
        if (GetChildrenCount() > 0)
            return GetChild(0);
    }
    return nullptr;
}

void RCK2dEntity::SetMaterial(CKMaterial *mat) {
    m_Material = mat;
}

CKMaterial *RCK2dEntity::GetMaterial() {
    return m_Material;
}

void RCK2dEntity::SetHomogeneousCoordinates(CKBOOL Coord) {
    if (Coord) {
        if (!(m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD)) {
            m_Flags |= CK_2DENTITY_USEHOMOGENEOUSCOORD;
            if (!m_HomogeneousRect)
                m_HomogeneousRect = new VxRect();
        }
    } else {
        if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
            m_Flags &= ~CK_2DENTITY_USEHOMOGENEOUSCOORD;
            if (m_HomogeneousRect) {
                delete m_HomogeneousRect;
                m_HomogeneousRect = nullptr;
            }
        }
    }
}

CKBOOL RCK2dEntity::IsHomogeneousCoordinates() {
    return (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) != 0;
}

void RCK2dEntity::EnableClipToCamera(CKBOOL Clip) {
    if (Clip)
        m_Flags |= CK_2DENTITY_CLIPTOCAMERAVIEW;
    else
        m_Flags &= ~CK_2DENTITY_CLIPTOCAMERAVIEW;
}

CKBOOL RCK2dEntity::IsClippedToCamera() {
    return (m_Flags & CK_2DENTITY_CLIPTOCAMERAVIEW) != 0;
}

// IDA: 0x1005cce5 - Updates clipped extents (m_VtxPos and m_SrcRect) for rendering
CKBOOL RCK2dEntity::UpdateExtents(CKRenderContext *dev) {
    RCKRenderContext *rc = (RCKRenderContext *) dev;

    // Determine source rect to use
    VxRect srcRect;
    if (m_Flags & CK_2DENTITY_USESRCRECT) {
        srcRect = m_SourceRect;
    } else {
        // DLL behavior:
        // - Sprites default to pixel-space source rect (0..width/height)
        // - Other 2D entities default to normalized (0..1)
        if (CKIsChildClassOf(this, CKCID_SPRITE)) {
            CKSprite *sprite = (CKSprite *) this;
            srcRect = VxRect(0.0f, 0.0f, (float) sprite->GetWidth(), (float) sprite->GetHeight());
        } else {
            srcRect = VxRect(0.0f, 0.0f, 1.0f, 1.0f);
        }
    }

    // Initialize extents rectangles
    m_VtxPos = VxRect(0.0f, 0.0f, 0.0f, 0.0f);
    m_SrcRect = VxRect(0.0f, 0.0f, 0.0f, 0.0f);

    // Calculate clip bounds
    VxRect clipRect;
    if (m_Flags & CK_2DENTITY_CLIPTOCAMERAVIEW) {
        // Clip to camera viewport
        clipRect = VxRect((float) rc->m_ViewportData.ViewX,
                          (float) rc->m_ViewportData.ViewY,
                          (float) (rc->m_ViewportData.ViewX + rc->m_ViewportData.ViewWidth),
                          (float) (rc->m_ViewportData.ViewY + rc->m_ViewportData.ViewHeight));
    } else {
        // Clip to render context rect
        clipRect = VxRect(0.0f, 0.0f,
                          (float) rc->m_Settings.m_Rect.right,
                          (float) rc->m_Settings.m_Rect.bottom);
    }

    // Clip to parent if needed
    if ((m_Flags & CK_2DENTITY_CLIPTOPARENT) && m_Parent) {
        RCK2dEntity *parent = (RCK2dEntity *) m_Parent;
        // Intersect with parent's VtxPos
        if (parent->m_VtxPos.left > clipRect.left) clipRect.left = parent->m_VtxPos.left;
        if (parent->m_VtxPos.top > clipRect.top) clipRect.top = parent->m_VtxPos.top;
        if (parent->m_VtxPos.right < clipRect.right) clipRect.right = parent->m_VtxPos.right;
        if (parent->m_VtxPos.bottom < clipRect.bottom) clipRect.bottom = parent->m_VtxPos.bottom;
    }

    // Calculate source dimensions
    float srcWidth = srcRect.right - srcRect.left;
    float srcHeight = srcRect.bottom - srcRect.top;

    // Calculate actual rect (apply homogeneous coords if needed)
    VxRect rect;
    float rectWidth, rectHeight;

    if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
        if (m_Flags & CK_2DENTITY_UPDATEHOMOGENEOUSCOORD) {
            // Update homogeneous rect from screen rect
            m_Flags &= ~CK_2DENTITY_UPDATEHOMOGENEOUSCOORD;
            VxRect relRect;
            GetHomogeneousRelativeRect(relRect);
            if (m_HomogeneousRect) {
                *m_HomogeneousRect = m_Rect;
                m_HomogeneousRect->TransformToHomogeneous(relRect);
            }
            rect = m_Rect;
        } else if (m_HomogeneousRect) {
            // Convert homogeneous to screen coordinates
            rect = *m_HomogeneousRect;
            VxRect relRect;
            GetHomogeneousRelativeRect(relRect);
            rect.TransformFromHomogeneous(relRect);
            m_Rect = rect;
        } else {
            rect = m_Rect;
        }
        rectWidth = rect.GetWidth();
        rectHeight = rect.GetHeight();
    } else {
        rect = m_Rect;
        rectWidth = m_Rect.GetWidth();
        rectHeight = m_Rect.GetHeight();
    }

    // Ensure rect is valid
    if (rect.right < rect.left) rect.right = rect.left;
    if (rect.bottom < rect.top) rect.bottom = rect.top;

    // Calculate inverse dimensions for UV mapping
    float invWidth = (rectWidth > 0.0f) ? (1.0f / rectWidth) : 0.0f;
    float invHeight = (rectHeight > 0.0f) ? (1.0f / rectHeight) : 0.0f;

    // Apply ratio offset if needed
    if (m_Flags & CK_2DENTITY_RATIOOFFSET) {
        rect.left += (float) rc->m_ViewportData.ViewX;
        rect.right += (float) rc->m_ViewportData.ViewX;
        rect.top += (float) rc->m_ViewportData.ViewY;
        rect.bottom += (float) rc->m_ViewportData.ViewY;
    }

    // Visibility test
    if (rect.right < clipRect.left || rect.left > clipRect.right ||
        rect.bottom < clipRect.top || rect.top > clipRect.bottom) {
        return FALSE; // Completely outside clip region
    }

    // Clip to bounds and adjust source rect
    float srcLeft = srcRect.left;
    float srcTop = srcRect.top;

    // Left edge clip
    if (rect.left < clipRect.left) {
        float delta = clipRect.left - rect.left;
        srcLeft += delta * srcWidth * invWidth;
        rect.left = clipRect.left;
    }

    // Top edge clip
    if (rect.top < clipRect.top) {
        float delta = clipRect.top - rect.top;
        srcTop += delta * srcHeight * invHeight;
        rect.top = clipRect.top;
    }

    // Right edge clip
    if (rect.right > clipRect.right) {
        float delta = rect.right - clipRect.right;
        srcRect.right = srcLeft + (rectWidth - delta) * srcWidth * invWidth;
        rect.right = clipRect.right;
    } else {
        srcRect.right = srcRect.right; // Keep original
    }

    // Bottom edge clip
    if (rect.bottom > clipRect.bottom) {
        float delta = rect.bottom - clipRect.bottom;
        srcRect.bottom = srcTop + (rectHeight - delta) * srcHeight * invHeight;
        rect.bottom = clipRect.bottom;
    } else {
        srcRect.bottom = srcRect.bottom; // Keep original
    }

    // Store clipped extents
    m_VtxPos = rect;
    m_SrcRect.left = srcLeft;
    m_SrcRect.top = srcTop;
    m_SrcRect.right = srcRect.right;
    m_SrcRect.bottom = srcRect.bottom;

    return TRUE;
}

CKERROR RCK2dEntity::Render(CKRenderContext *context) {
    // IDA: 0x1005ed00
    RCKRenderContext *dev = (RCKRenderContext *) context;

    // Check if hidden by parent hierarchy
    if (m_ObjectFlags & CK_OBJECT_HIERACHICALHIDE)
        return CK_OK;

    // Determine visibility
    CKBOOL visible = FALSE;
    if (IsVisible()) {
        CKScene *scene = m_Context->GetCurrentScene();
        // IDA: CKSceneObject::IsInScene(this, CurrentScene) || sub_100604B0(this)
        if (IsInScene(scene) || (m_ObjectFlags & CK_OBJECT_INTERFACEOBJ))
            visible = TRUE;
    }

    // If not visible and no children, skip
    if (!visible && m_Children.Size() == 0)
        return CK_OK;

    // Update extents (returns FALSE if completely clipped)
    CKBOOL clipped = !UpdateExtents(context);

    // Execute pre-render callbacks if visible and has callbacks
    if (visible && m_Callbacks && m_Callbacks->m_PreCallBacks.Size() > 0) {
        dev->m_SpriteCallbacksTimeProfiler.Reset();
        dev->m_RasterizerContext->SetVertexShader(0);
        for (auto it = m_Callbacks->m_PreCallBacks.Begin(); it != m_Callbacks->m_PreCallBacks.End(); ++it) {
            ((CK_RENDEROBJECT_CALLBACK) it->callback)(dev, (CKRenderObject *) this, it->argument);
        }
        dev->m_Stats.SpriteCallbacksTime += dev->m_SpriteCallbacksTimeProfiler.Current();
    }

    // Draw if visible and not completely clipped
    if (!clipped && visible)
        Draw(context);

    // Render children
    for (auto it = m_Children.Begin(); it != m_Children.End(); ++it) {
        RCK2dEntity *child = (RCK2dEntity *) *it;
        // Skip children that clip to parent if we're completely clipped
        if (!clipped || !(child->m_Flags & CK_2DENTITY_CLIPTOPARENT))
            child->Render(context);
    }

    // Execute post-render callbacks if visible and has callbacks
    if (visible && m_Callbacks && m_Callbacks->m_PostCallBacks.Size() > 0) {
        dev->m_SpriteCallbacksTimeProfiler.Reset();
        dev->m_RasterizerContext->SetVertexShader(0);
        for (auto it = m_Callbacks->m_PostCallBacks.Begin(); it != m_Callbacks->m_PostCallBacks.End(); ++it) {
            ((CK_RENDEROBJECT_CALLBACK) it->callback)(dev, (CKRenderObject *) this, it->argument);
        }
        dev->m_Stats.SpriteCallbacksTime += dev->m_SpriteCallbacksTimeProfiler.Current();
    }

    return CK_OK;
}

CKERROR RCK2dEntity::Draw(CKRenderContext *context) {
    // IDA: 0x1005e430
    RCKRenderContext *dev = (RCKRenderContext *) context;

    if (m_Material) {
        // Save viewport if not clip-to-camera
        VxRect savedViewRect;
        if (!(m_Flags & CK_2DENTITY_CLIPTOCAMERAVIEW)) {
            VxRect windowRect;
            dev->GetViewRect(savedViewRect);
            dev->GetWindowRect(windowRect, FALSE);

            float width = windowRect.GetWidth();
            float height = windowRect.GetHeight();

            // Set full viewport
            dev->SetFullViewport(&dev->m_RasterizerContext->m_ViewportData, (int) width, (int) height);
            dev->m_RasterizerContext->SetViewport(&dev->m_RasterizerContext->m_ViewportData);
        }

        // Set material
        m_Material->SetAsCurrent(dev, TRUE, FALSE);

        // Set render states
        dev->SetState(VXRENDERSTATE_CULLMODE, VXCULL_NONE);
        dev->SetState(VXRENDERSTATE_FOGENABLE, FALSE);

        // Background entities: disable Z test and write
        if (IsBackground()) {
            dev->SetState(VXRENDERSTATE_ZFUNC, VXCMP_ALWAYS);
            dev->SetState(VXRENDERSTATE_ZWRITEENABLE, FALSE);
        }

        // Get draw primitive structure for 4 vertices (quad)
        VxDrawPrimitiveData *data = dev->GetDrawPrimitiveStructure(CKRST_DP_CL_VCT, 4);
        CKDWORD *texCoordPtr = (CKDWORD *) data->TexCoordPtr;
        float *positionPtr = (float *) data->PositionPtr;
        void *colorPtr = data->ColorPtr;

        // Get diffuse color from material and fill all vertices
        const VxColor &diffuse = m_Material->GetDiffuse();
        CKDWORD colorValue = RGBAFTOCOLOR(&diffuse);
        VxFillStructure(4, colorPtr, data->ColorStride, 4, &colorValue);

        // Set texture coordinates if texture present
        if (m_Material->GetTexture(0)) {
            // Vertex 0: top-left (srcRect left, top)
            texCoordPtr[0] = *(CKDWORD *) &m_SrcRect.left;
            texCoordPtr[1] = *(CKDWORD *) &m_SrcRect.top;
            // Vertex 1: top-right (srcRect right, top)
            texCoordPtr[2] = *(CKDWORD *) &m_SrcRect.right;
            texCoordPtr[3] = *(CKDWORD *) &m_SrcRect.top;
            // Vertex 2: bottom-right (srcRect right, bottom)
            texCoordPtr[4] = *(CKDWORD *) &m_SrcRect.right;
            texCoordPtr[5] = *(CKDWORD *) &m_SrcRect.bottom;
            // Vertex 3: bottom-left (srcRect left, bottom)
            texCoordPtr[6] = *(CKDWORD *) &m_SrcRect.left;
            texCoordPtr[7] = *(CKDWORD *) &m_SrcRect.bottom;
        }

        // Set vertex positions (round to nearest pixel)
        int posStride = data->PositionStride;

        // Vertex 0: top-left
        positionPtr[0] = (float) (int) (m_VtxPos.left + 0.5f);
        positionPtr[1] = (float) (int) (m_VtxPos.top + 0.5f);
        positionPtr[2] = 0.0f;
        positionPtr[3] = 1.0f;
        positionPtr = (float *) ((char *) positionPtr + posStride);

        // Vertex 1: top-right
        positionPtr[0] = (float) (int) (m_VtxPos.right + 0.5f);
        positionPtr[1] = (float) (int) (m_VtxPos.top + 0.5f);
        positionPtr[2] = 0.0f;
        positionPtr[3] = 1.0f;
        positionPtr = (float *) ((char *) positionPtr + posStride);

        // Vertex 2: bottom-right
        positionPtr[0] = (float) (int) (m_VtxPos.right + 0.5f);
        positionPtr[1] = (float) (int) (m_VtxPos.bottom + 0.5f);
        positionPtr[2] = 0.0f;
        positionPtr[3] = 1.0f;
        positionPtr = (float *) ((char *) positionPtr + posStride);

        // Vertex 3: bottom-left
        positionPtr[0] = (float) (int) (m_VtxPos.left + 0.5f);
        positionPtr[1] = (float) (int) (m_VtxPos.bottom + 0.5f);
        positionPtr[2] = 0.0f;
        positionPtr[3] = 1.0f;

        // Draw quad as triangle fan
        dev->DrawPrimitive(VX_TRIANGLEFAN, NULL, 4, data);

        // Restore fog state
        dev->SetState(VXRENDERSTATE_FOGENABLE, dev->m_RenderedScene->m_FogMode != 0);

        // Restore viewport if changed
        if (!(m_Flags & CK_2DENTITY_CLIPTOCAMERAVIEW)) {
            dev->m_RasterizerContext->m_ViewportData.ViewX = (int) savedViewRect.left;
            dev->m_RasterizerContext->m_ViewportData.ViewY = (int) savedViewRect.top;
            dev->m_RasterizerContext->m_ViewportData.ViewWidth = (int) (savedViewRect.right - savedViewRect.left);
            dev->m_RasterizerContext->m_ViewportData.ViewHeight = (int) (savedViewRect.bottom - savedViewRect.top);
            dev->m_RasterizerContext->SetViewport(&dev->m_RasterizerContext->m_ViewportData);
        }
    } else {
        // No material - draw placeholder (editor mode only)
        if (m_Context->IsPlaying())
            return CK_OK;

        // Set blend states for transparent black fill
        dev->SetState(VXRENDERSTATE_ALPHABLENDENABLE, TRUE);
        dev->SetState(VXRENDERSTATE_ALPHATESTENABLE, FALSE);
        dev->SetState(VXRENDERSTATE_SRCBLEND, VXBLEND_SRCALPHA);
        dev->SetState(VXRENDERSTATE_DESTBLEND, VXBLEND_INVSRCALPHA);
        dev->SetState(VXRENDERSTATE_CULLMODE, VXCULL_NONE);
        dev->SetState(VXRENDERSTATE_ZWRITEENABLE, FALSE);
        dev->SetState(VXRENDERSTATE_ZFUNC, VXCMP_ALWAYS);
        dev->SetState(VXRENDERSTATE_FOGENABLE, FALSE);
        dev->SetTexture(NULL, FALSE, 0);

        // Get draw primitive structure
        VxDrawPrimitiveData *data = dev->GetDrawPrimitiveStructure(CKRST_DP_CL_VCT, 4);
        float *positionPtr = (float *) data->PositionPtr;
        void *colorPtr = data->ColorPtr;
        int posStride = data->PositionStride;

        // Black color with 40% alpha
        VxColor black(0.0f, 0.0f, 0.0f, 0.4f);
        CKDWORD blackColor = RGBAFTOCOLOR(&black);
        VxFillStructure(4, colorPtr, data->ColorStride, 4, &blackColor);

        // Vertex 0: top-left
        positionPtr[0] = m_VtxPos.left;
        positionPtr[1] = m_VtxPos.top;
        positionPtr[2] = 0.0f;
        positionPtr[3] = 1.0f;
        positionPtr = (float *) ((char *) positionPtr + posStride);

        // Vertex 1: top-right
        positionPtr[0] = m_VtxPos.right;
        positionPtr[1] = m_VtxPos.top;
        positionPtr[2] = 0.0f;
        positionPtr[3] = 1.0f;
        positionPtr = (float *) ((char *) positionPtr + posStride);

        // Vertex 2: bottom-right
        positionPtr[0] = m_VtxPos.right;
        positionPtr[1] = m_VtxPos.bottom;
        positionPtr[2] = 0.0f;
        positionPtr[3] = 1.0f;
        positionPtr = (float *) ((char *) positionPtr + posStride);

        // Vertex 3: bottom-left
        positionPtr[0] = m_VtxPos.left;
        positionPtr[1] = m_VtxPos.bottom;
        positionPtr[2] = 0.0f;
        positionPtr[3] = 1.0f;

        // Draw filled quad
        dev->DrawPrimitive(VX_TRIANGLEFAN, NULL, 4, data);

        // Draw white outline
        CKWORD *indices = dev->GetDrawPrimitiveIndices(5);
        indices[0] = 0;
        indices[1] = 1;
        indices[2] = 2;
        indices[3] = 3;
        indices[4] = 0;

        // White color
        CKDWORD whiteColor = 0xFFFFFFFF;

        // Adjust positions for outline (shrink by 1 pixel)
        positionPtr = (float *) data->PositionPtr;
        positionPtr = (float *) ((char *) positionPtr + posStride); // Skip to vertex 1
        positionPtr[0] -= 1.0f;                                     // Move left
        positionPtr = (float *) ((char *) positionPtr + posStride); // To vertex 2
        positionPtr[0] -= 1.0f;                                     // Move left
        positionPtr[1] -= 1.0f;                                     // Move up
        positionPtr = (float *) ((char *) positionPtr + posStride); // To vertex 3
        positionPtr[1] -= 1.0f;                                     // Move up

        VxFillStructure(4, colorPtr, data->ColorStride, 4, &whiteColor);

        // Draw outline as line strip
        dev->DrawPrimitive(VX_LINESTRIP, indices, 5, data);

        // Restore fog state
        dev->SetState(VXRENDERSTATE_FOGENABLE, dev->m_RenderedScene->m_FogMode != 0);
    }

    return CK_OK;
}

void RCK2dEntity::GetExtents(VxRect &srcrect, VxRect &rect) {
    srcrect = m_SourceRect;
    rect = m_Rect;
}

void RCK2dEntity::SetExtents(const VxRect &srcrect, const VxRect &rect) {
    m_SourceRect = srcrect;
    m_Rect = rect;
}

void RCK2dEntity::RestoreInitialSize() {}

CKBOOL RCK2dEntity::IsHiddenByParent() {
    CK2dEntity *p = m_Parent;
    while (p) {
        if (!p->IsVisible()) return TRUE;
        p = p->GetParent();
    }
    return FALSE;
}

CKBOOL RCK2dEntity::IsInRenderContext(CKRenderContext *context) {
    RCKRenderContext *dev = (RCKRenderContext *) context;
    return (dev->m_MaskFree & m_InRenderContext) != 0;
}

CKBOOL RCK2dEntity::IsRootObject() {
    return m_Parent == nullptr;
}

CKBOOL RCK2dEntity::IsToBeRendered() {
    return IsVisible();
}

void RCK2dEntity::SetZOrder(int Z) {
    m_ZOrder = Z;

    // IDA: sub_1005d3ed - sort parent's children or root's children after Z change
    if (m_Parent) {
        ((RCK2dEntity *) m_Parent)->m_Children.Sort(CompareByZOrder);
    } else {
        CKRenderContext *rc = m_Context->GetPlayerRenderContext();
        if (rc) {
            RCK2dEntity *root = (RCK2dEntity *) rc->Get2dRoot(IsBackground());
            if (root) {
                root->m_Children.Sort(CompareByZOrder);
            }
        }
    }
}

int RCK2dEntity::GetZOrder() {
    return m_ZOrder;
}

CKERROR RCK2dEntity::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) return CKERR_INVALIDPARAMETER;

    CKERROR err = CKBeObject::Load(chunk, file);
    if (err != CK_OK) return err;

    if (chunk->GetDataVersion() >= 5) {
        // New format (version 5+)
        if (chunk->SeekIdentifier(0x10F000)) {
            CKDWORD saveFlags = chunk->ReadDword();
            m_Flags = saveFlags & ~(CK_2DENTITY_UPDATEHOMOGENEOUSCOORD | CK_2DENTITY_RESERVED0 | CK_2DENTITY_RESERVED1 | CK_2DENTITY_RESERVED2);

            // Allocate homogeneous rect if needed
            if ((m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) && !m_HomogeneousRect) {
                m_HomogeneousRect = new VxRect();
            }

            // Read rect
            if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
                chunk->ReadAndFillBuffer_LEndian(sizeof(VxRect), m_HomogeneousRect);
            } else {
                chunk->ReadAndFillBuffer_LEndian(sizeof(VxRect), &m_Rect);
            }

            // Read source rect conditionally
            if (saveFlags & CK_2DENTITY_RESERVED0) {
                chunk->ReadAndFillBuffer_LEndian(sizeof(VxRect), &m_SourceRect);
            } else if (GetClassID() == CKCID_2DENTITY) {
                m_SourceRect = VxRect(0.0f, 0.0f, 1.0f, 1.0f);
            } else {
                m_SourceRect = VxRect(0.0f, 0.0f, 0.0f, 0.0f);
            }

            // Read ZOrder conditionally
            if (saveFlags & CK_2DENTITY_RESERVED1) {
                m_ZOrder = chunk->ReadInt();
            } else {
                m_ZOrder = 0;
            }

            // Read parent conditionally
            if (saveFlags & CK_2DENTITY_RESERVED2) {
                CK2dEntity *parent = (CK2dEntity *) chunk->ReadObject(m_Context);
                if (file) {
                    m_Parent = parent;
                } else {
                    SetParent(parent);
                }
            } else {
                SetParent(nullptr);
            }
        }

        // Read material for CK2dEntity only
        if (GetClassID() == CKCID_2DENTITY && chunk->SeekIdentifier(0x200000)) {
            m_Material = (CKMaterial *) chunk->ReadObject(m_Context);
        } else {
            m_Material = nullptr;
        }
    } else {
        // Old format (version < 5)
        if (chunk->SeekIdentifier(0x4000)) {
            m_Flags = chunk->ReadDword();
            if (!(m_Flags & 1)) {
                m_Flags |= 1;
                Show(CKHIDE);
            }
            m_Flags &= ~CK_2DENTITY_UPDATEHOMOGENEOUSCOORD;
            m_Flags |= CK_2DENTITY_STICKTOP | CK_2DENTITY_STICKLEFT;
        }

        if ((m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) && !m_HomogeneousRect) {
            m_HomogeneousRect = new VxRect();
        }

        if (chunk->SeekIdentifier(0x8000)) {
            if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
                m_HomogeneousRect->left = chunk->ReadFloat();
                m_HomogeneousRect->top = chunk->ReadFloat();
            } else {
                int x = chunk->ReadInt();
                int y = chunk->ReadInt();
                m_Rect.left = (float) x;
                m_Rect.top = (float) y;
            }
        }

        if (chunk->SeekIdentifier(0x2000)) {
            if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
                float w = chunk->ReadFloat();
                float h = chunk->ReadFloat();
                m_HomogeneousRect->right = m_HomogeneousRect->left + w;
                m_HomogeneousRect->bottom = m_HomogeneousRect->top + h;
            } else {
                int w = chunk->ReadInt();
                int h = chunk->ReadInt();
                m_Rect.right = m_Rect.left + (float) w;
                m_Rect.bottom = m_Rect.top + (float) h;
            }
        }

        if (chunk->SeekIdentifier(0x1000)) {
            m_SourceRect.right = (float) chunk->ReadInt();
            m_SourceRect.left = (float) chunk->ReadInt();
            m_SourceRect.top = (float) chunk->ReadInt();
            m_SourceRect.bottom = (float) chunk->ReadInt();
        }

        if (chunk->SeekIdentifier(0x100000)) {
            m_ZOrder = chunk->ReadInt();
        }
    }

    // Transform from homogeneous to screen coords if needed
    if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
        VxRect tempRect = *m_HomogeneousRect;
        VxRect relRect;
        GetHomogeneousRelativeRect(relRect);
        tempRect.TransformFromHomogeneous(relRect);
        m_Rect = tempRect;
    }

    return CK_OK;
}

void RCK2dEntity::PostLoad(void) {
    // IDA: 0x1005f7b1
    if (m_Parent) {
        ((RCK2dEntity *) m_Parent)->m_Children.PushBack((CK2dEntity *) this);
        // IDA: Sort children by ZOrder after adding
        // sub_10041C90 sorts the array using comparison function sub_1005C750
        ((RCK2dEntity *) m_Parent)->m_Children.Sort(CompareByZOrder);

        // IDA: Remove from all render contexts when parented
        // This ensures the 2D entity is not in both root list and child list
        CKRenderManager *rm = m_Context->GetRenderManager();
        int count = rm->GetRenderContextCount();
        for (int i = 0; i < count; ++i) {
            CKRenderContext *rc = rm->GetRenderContext(i);
            rc->RemoveObject((CKRenderObject *) this);
        }
    }
    CKObject::PostLoad();
}

void RCK2dEntity::PreSave(CKFile *file, CKDWORD flags) {
    // IDA: 0x1005efa8
    CKBeObject::PreSave(file, flags);
    if ((flags & CK_STATESAVE_2DENTITYMATERIAL) && m_Material) {
        file->SaveObject(m_Material, flags);
    }
    if (flags & CK_STATESAVE_2DENTITYHIERARCHY) {
        // IDA uses SaveObjects with array pointer and count
        int count = m_Children.Size();
        if (count > 0) {
            file->SaveObjects((CKObject **) m_Children.Begin(), count, 0xFFFFFFFF);
        }
    }
}

CKStateChunk *RCK2dEntity::Save(CKFile *file, CKDWORD flags) {
    CKStateChunk *chunk = CreateCKStateChunk(CKCID_2DENTITY, file);
    CKStateChunk *baseChunk = CKBeObject::Save(file, flags);
    if (!chunk) return baseChunk;

    chunk->StartWrite();
    chunk->AddChunkAndDelete(baseChunk);

    chunk->WriteIdentifier(0x10F000);

    // Build save flags
    CKDWORD saveFlags = m_Flags;
    if (m_SourceRect.top != 0.0f || m_SourceRect.left != 0.0f ||
        m_SourceRect.bottom != 0.0f || m_SourceRect.right != 0.0f) {
        saveFlags |= CK_2DENTITY_RESERVED0;
    }
    if (m_ZOrder) {
        saveFlags |= CK_2DENTITY_RESERVED1;
    }
    if (m_Parent) {
        saveFlags |= CK_2DENTITY_RESERVED2;
    }

    chunk->WriteDword(saveFlags);

    // Write rect - homogeneous or screen coords
    if (m_Flags & CK_2DENTITY_USEHOMOGENEOUSCOORD) {
        chunk->WriteBufferNoSize_LEndian(sizeof(VxRect), m_HomogeneousRect);
    } else {
        chunk->WriteBufferNoSize_LEndian(sizeof(VxRect), &m_Rect);
    }

    // Conditionally write source rect
    if (saveFlags & CK_2DENTITY_RESERVED0) {
        chunk->WriteBufferNoSize_LEndian(sizeof(VxRect), &m_SourceRect);
    }

    // Conditionally write ZOrder
    if (saveFlags & CK_2DENTITY_RESERVED1) {
        chunk->WriteInt(m_ZOrder);
    }

    // Conditionally write parent
    if (saveFlags & CK_2DENTITY_RESERVED2) {
        chunk->WriteObject(m_Parent);
    }

    // Write material if present
    if (m_Material) {
        chunk->WriteIdentifier(0x200000);
        chunk->WriteObject(m_Material);
    }

    if (GetClassID() == CKCID_2DENTITY) {
        chunk->CloseChunk();
    } else {
        chunk->UpdateDataSize();
    }
    return chunk;
}

void RCK2dEntity::PreDelete() {
    // IDA: 0x1005c8e1
    CKBeObject::PreDelete();

    CKRenderContext *rc = m_Context->GetPlayerRenderContext();

    // Don't process if this is a 2D root entity
    if (rc) {
        CK2dEntity *root0 = rc->Get2dRoot(0);
        CK2dEntity *root1 = rc->Get2dRoot(1);
        if (this == (RCK2dEntity *) root1 || this == (RCK2dEntity *) root0) {
            return;
        }
    }

    // Find first valid parent (that is not being deleted)
    RCK2dEntity *validParent = (RCK2dEntity *) m_Parent;
    while (validParent && validParent->IsToBeDeleted()) {
        validParent = (RCK2dEntity *) validParent->m_Parent;
    }

    // Copy children array to avoid modification during iteration
    XArray<CK2dEntity *> childrenCopy;
    int count = m_Children.Size();
    for (int i = 0; i < count; ++i) {
        childrenCopy.PushBack(m_Children[i]);
    }

    // Reparent children that are not being deleted
    for (auto it = childrenCopy.Begin(); it != childrenCopy.End(); ++it) {
        RCK2dEntity *child = (RCK2dEntity *) *it;
        if (!child->IsToBeDeleted()) {
            child->SetParent((CK2dEntity *) validParent);
        }
    }

    // Remove this from parent's children list
    if (m_Parent && !m_Parent->IsToBeDeleted()) {
        ((RCK2dEntity *) m_Parent)->m_Children.Remove((CK2dEntity *) this);
    } else {
        // Remove from root if no valid parent
        if (rc) {
            CK2dEntity *root = rc->Get2dRoot(IsBackground() ? 1 : 0);
            if (root) {
                ((RCK2dEntity *) root)->m_Children.Remove((CK2dEntity *) this);
            }
        }
    }
}

void RCK2dEntity::CheckPreDeletion() {
    // IDA: 0x1005f755
    CKObject::CheckPreDeletion();
    if (m_Material) {
        if (m_Material->IsToBeDeleted()) {
            m_Material = nullptr;
        }
    }
}

int RCK2dEntity::GetMemoryOccupation() {
    // IDA: 0x1005f793 - RCKRenderObject::GetMemoryOccupation(this) + 96
    return RCKRenderObject::GetMemoryOccupation() + (sizeof(RCK2dEntity) - sizeof(RCKRenderObject));
}

CKERROR RCK2dEntity::PrepareDependencies(CKDependenciesContext &context) {
    // IDA: 0x1005fa5b
    CKERROR err = CKBeObject::PrepareDependencies(context);
    if (err != CK_OK)
        return err;

    CKDWORD classDeps = context.GetClassDependencies(CKCID_2DENTITY);

    // Bit 0: Material dependency
    if ((classDeps & 1) && m_Material) {
        m_Material->PrepareDependencies(context);
    }

    // Bit 1: Children dependency
    if (classDeps & 2) {
        int count = GetChildrenCount();
        for (int i = 0; i < count; ++i) {
            CK2dEntity *child = GetChild(i);
            if (child) {
                child->PrepareDependencies(context);
            }
        }
    }

    return context.FinishPrepareDependencies(this, m_ClassID);
}

CKERROR RCK2dEntity::RemapDependencies(CKDependenciesContext &context) {
    // IDA: 0x1005fb2d
    CKERROR err = CKBeObject::RemapDependencies(context);
    if (err != CK_OK)
        return err;

    CKDWORD classDeps = context.GetClassDependencies(CKCID_2DENTITY);

    // Bit 0: Remap material
    if (classDeps & 1) {
        m_Material = (CKMaterial *) context.Remap(m_Material);
    }

    // Bit 1: Remap parent
    if (classDeps & 2) {
        CK2dEntity *parent = GetParent();
        CK2dEntity *newParent = (CK2dEntity *) context.Remap(parent);
        if (newParent) {
            SetParent(newParent);
        }
    }

    return CK_OK;
}

CKERROR RCK2dEntity::Copy(CKObject &o, CKDependenciesContext &context) {
    // IDA: 0x1005fbc6
    CKERROR err = CKBeObject::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCK2dEntity *src = (RCK2dEntity *) &o;

    m_Flags = src->m_Flags;
    m_SourceRect = src->m_SourceRect;
    m_ZOrder = src->m_ZOrder;
    m_Rect = src->m_Rect;
    m_Material = src->m_Material;
    SetParent(src->m_Parent);

    // Copy homogeneous rect if source has one
    if (src->m_HomogeneousRect) {
        if (m_HomogeneousRect) {
            delete m_HomogeneousRect;
        }
        m_HomogeneousRect = new VxRect(*src->m_HomogeneousRect);
    }

    return CK_OK;
}

CKSTRING RCK2dEntity::GetClassName() {
    return (CKSTRING) "2dEntity";
}

int RCK2dEntity::GetDependenciesCount(int mode) {
    // IDA: 0x1005f906
    switch (mode) {
    case CK_DEPENDENCIES_COPY: return 2;
    case CK_DEPENDENCIES_DELETE: return 2;
    case CK_DEPENDENCIES_REPLACE: return 0;
    case CK_DEPENDENCIES_SAVE: return 2;
    default: return 0;
    }
}

CKSTRING RCK2dEntity::GetDependencies(int i, int mode) {
    // IDA: 0x1005f958
    if (i == 0)
        return (CKSTRING) "Material";
    if (i == 1)
        return (CKSTRING) "Children";
    return nullptr;
}

void RCK2dEntity::Register() {
    CKCLASSNOTIFYFROMCID(RCK2dEntity, CKCID_MATERIAL);
    CKPARAMETERFROMCLASS(RCK2dEntity, CKPGUID_2DENTITY);
    CKCLASSDEFAULTCOPYDEPENDENCIES(RCK2dEntity, CK_DEPENDENCIES_REPLACE);
    CKCLASSDEFAULTSAVEDEPENDENCIES(RCK2dEntity, CK_DEPENDENCIES_REPLACE);
}

RCK2dEntity *RCK2dEntity::CreateInstance(CKContext *Context) {
    return new RCK2dEntity(Context);
}
