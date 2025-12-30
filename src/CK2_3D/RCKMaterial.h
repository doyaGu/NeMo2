#ifndef RCKMATERIAL_H
#define RCKMATERIAL_H

#include "CKRenderEngineTypes.h"
#include "CKMaterial.h"

struct CKSprite3DBatch;
class RCKRenderContext;
class RCK3dEntity;

/**
 * @class RCKMaterial
 * @brief Custom implementation of CKMaterial for the CKRenderEngine.
 *
 * This class implements the material system for 3D rendering, including
 * color properties, textures, blend modes, and various rendering states.
 * Materials define how surfaces appear when rendered and interact with lights.
 *
 * Member layout based on decompilation (total size: 0xF0 = 240 bytes):
 * - Base CKBeObject data
 * - m_Textures[4]:       4 texture slots for multi-texturing
 * - m_MaterialData:      Diffuse, Ambient, Specular, Emissive colors and power
 * - m_SpecularColor:     Cached specular color (separate from MaterialData.Specular)
 * - m_TextureBlendMode:  How texture and vertex colors are combined
 * - m_TextureMinMode:    Texture minification filter
 * - m_TextureMagMode:    Texture magnification filter
 * - m_SourceBlend:       Source blend factor
 * - m_DestBlend:         Destination blend factor
 * - m_ShadeMode:         Flat or Gouraud shading
 * - m_FillMode:          Solid, wireframe, or point
 * - m_TextureAddressMode: Wrap, clamp, mirror, border
 * - m_TextureBorderColor: Color for border address mode
 * - m_Flags:             Packed flags (TwoSided, ZWrite, AlphaBlend, AlphaTest, ZFunc, AlphaFunc, Effect)
 * - m_AlphaRef:          Alpha reference value for alpha testing
 * - m_Sprite3DBatch:     Batch for sprite rendering
 * - m_Callback:          User callback for material setup
 * - m_CallbackArgument:  Argument passed to callback
 * - m_EffectParameter:   Parameter for special effects
 */
class RCKMaterial : public CKMaterial {
    // Allow RCKRenderContext to access protected members for batch rendering
    friend class RCKRenderContext;

public:
    //=========================================================================
    // Construction/Destruction
    //=========================================================================
    explicit RCKMaterial(CKContext *Context, CKSTRING name = nullptr);
    ~RCKMaterial() override;

    //=========================================================================
    // CKObject Virtual Methods
    //=========================================================================
    CK_CLASSID GetClassID() override;
    void CheckPreDeletion() override;
    int GetMemoryOccupation() override;
    CKBOOL IsObjectUsed(CKObject *obj, CK_CLASSID cid) override;

    //=========================================================================
    // Serialization
    //=========================================================================
    void PreSave(CKFile *file, CKDWORD flags) override;
    CKStateChunk *Save(CKFile *file, CKDWORD flags) override;
    CKERROR Load(CKStateChunk *chunk, CKFile *file) override;

    //=========================================================================
    // Dependencies
    //=========================================================================
    CKERROR PrepareDependencies(CKDependenciesContext &context) override;
    CKERROR RemapDependencies(CKDependenciesContext &context) override;
    CKERROR Copy(CKObject &o, CKDependenciesContext &context) override;

    //=========================================================================
    // Static Class Methods
    //=========================================================================
    static CKSTRING GetClassName();
    static int GetDependenciesCount(int mode);
    static CKSTRING GetDependencies(int i, int mode);
    static void Register();
    static RCKMaterial *CreateInstance(CKContext *Context);
    static CK_CLASSID m_ClassID;

    //=========================================================================
    // CKMaterial Virtual Methods - Color Properties
    //=========================================================================
    float GetPower() override;
    void SetPower(float Value) override;
    const VxColor &GetAmbient() override;
    void SetAmbient(const VxColor &Color) override;
    const VxColor &GetDiffuse() override;
    void SetDiffuse(const VxColor &Color) override;
    const VxColor &GetSpecular() override;
    void SetSpecular(const VxColor &Color) override;
    const VxColor &GetEmissive() override;
    void SetEmissive(const VxColor &Color) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Texture Management
    //=========================================================================
    CKTexture *GetTexture(int TexIndex = 0) override;
    void SetTexture(int TexIndex, CKTexture *Tex) override;
    void SetTexture0(CKTexture *Tex) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Texture Blend Mode
    //=========================================================================
    void SetTextureBlendMode(VXTEXTURE_BLENDMODE BlendMode) override;
    VXTEXTURE_BLENDMODE GetTextureBlendMode() override;

    //=========================================================================
    // CKMaterial Virtual Methods - Texture Filter Mode
    //=========================================================================
    void SetTextureMinMode(VXTEXTURE_FILTERMODE FilterMode) override;
    VXTEXTURE_FILTERMODE GetTextureMinMode() override;
    void SetTextureMagMode(VXTEXTURE_FILTERMODE FilterMode) override;
    VXTEXTURE_FILTERMODE GetTextureMagMode() override;

    //=========================================================================
    // CKMaterial Virtual Methods - Texture Address Mode
    //=========================================================================
    void SetTextureAddressMode(VXTEXTURE_ADDRESSMODE Mode) override;
    VXTEXTURE_ADDRESSMODE GetTextureAddressMode() override;

    //=========================================================================
    // CKMaterial Virtual Methods - Texture Border Color
    //=========================================================================
    void SetTextureBorderColor(CKDWORD Color) override;
    CKDWORD GetTextureBorderColor() override;

    //=========================================================================
    // CKMaterial Virtual Methods - Blend Factors
    //=========================================================================
    void SetSourceBlend(VXBLEND_MODE BlendMode) override;
    void SetDestBlend(VXBLEND_MODE BlendMode) override;
    VXBLEND_MODE GetSourceBlend() override;
    VXBLEND_MODE GetDestBlend() override;

    //=========================================================================
    // CKMaterial Virtual Methods - Two-Sided
    //=========================================================================
    CKBOOL IsTwoSided() override;
    void SetTwoSided(CKBOOL TwoSided) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Z Buffer
    //=========================================================================
    CKBOOL ZWriteEnabled() override;
    void EnableZWrite(CKBOOL ZWrite = TRUE) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Alpha Blending
    //=========================================================================
    CKBOOL AlphaBlendEnabled() override;
    void EnableAlphaBlend(CKBOOL Blend = TRUE) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Z Comparison
    //=========================================================================
    VXCMPFUNC GetZFunc() override;
    void SetZFunc(VXCMPFUNC ZFunc = VXCMP_LESSEQUAL) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Perspective Correction
    //=========================================================================
    CKBOOL PerspectiveCorrectionEnabled() override;
    void EnablePerspectiveCorrection(CKBOOL Perspective = TRUE) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Fill Mode
    //=========================================================================
    void SetFillMode(VXFILL_MODE FillMode) override;
    VXFILL_MODE GetFillMode() override;

    //=========================================================================
    // CKMaterial Virtual Methods - Shade Mode
    //=========================================================================
    void SetShadeMode(VXSHADE_MODE ShadeMode) override;
    VXSHADE_MODE GetShadeMode() override;

    //=========================================================================
    // CKMaterial Virtual Methods - Set As Current
    //=========================================================================
    CKBOOL SetAsCurrent(CKRenderContext *context, CKBOOL Lit = TRUE, int TextureStage = 0) override;
    CKBOOL IsAlphaTransparent() override;

    //=========================================================================
    // CKMaterial Virtual Methods - Alpha Testing
    //=========================================================================
    CKBOOL AlphaTestEnabled() override;
    void EnableAlphaTest(CKBOOL Enable = TRUE) override;
    VXCMPFUNC GetAlphaFunc() override;
    void SetAlphaFunc(VXCMPFUNC AlphaFunc = VXCMP_ALWAYS) override;
    CKBYTE GetAlphaRef() override;
    void SetAlphaRef(CKBYTE AlphaRef = 0) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Callback
    //=========================================================================
    void SetCallback(CK_MATERIALCALLBACK Fct, void *Argument) override;
    CK_MATERIALCALLBACK GetCallback(void **Argument = nullptr) override;

    //=========================================================================
    // CKMaterial Virtual Methods - Effects
    //=========================================================================
    void SetEffect(VX_EFFECT Effect) override;
    VX_EFFECT GetEffect() override;
    CKParameter *GetEffectParameter() override;

    //=========================================================================
    // Effect Helper Methods (non-virtual)
    //=========================================================================
    CKDWORD TexGenEffect(RCKRenderContext *dev, VX_EFFECTTEXGEN texGen, RCK3dEntity *refEntity, int stage);
    CKDWORD BumpMapEnvEffect(RCKRenderContext *dev);
    CKDWORD DP3Effect(RCKRenderContext *dev, int stage);
    CKDWORD BlendTexturesEffect(RCKRenderContext *dev, int stage);

    //=========================================================================
    // Sprite3D Batch Methods (non-virtual)
    //=========================================================================
    CKBOOL AddSprite3DBatch(RCKSprite3D *sprite);
    void FlushSprite3DBatch();
    CKSprite3DBatch *GetSprite3DBatch() { return m_Sprite3DBatch; }

    //=====================================================================
    // Internal helpers (non-virtual)
    //=====================================================================
    // These helpers exist to mirror the original engine's behavior where
    // channel rendering temporarily patches blend/flags directly without
    // triggering side effects (e.g., transparency re-evaluation).
    void PatchForChannelRender(VXBLEND_MODE sourceBlend, VXBLEND_MODE destBlend,
                               VXBLEND_MODE &savedSourceBlend, VXBLEND_MODE &savedDestBlend, CKDWORD &savedFlags);
    void RestoreAfterChannelRender(VXBLEND_MODE savedSourceBlend, VXBLEND_MODE savedDestBlend, CKDWORD savedFlags);

protected:
    // Texture slots (4 for multi-texturing support)
    CKTexture *m_Textures[4];

    // Core material data (Diffuse, Ambient, Specular, Emissive, SpecularPower)
    CKMaterialData m_MaterialData;

    // Cached specular color (MaterialData.Specular is computed from this)
    VxColor m_SpecularColor;

    // Texture and rendering modes
    CKDWORD m_TextureBlendMode;
    CKDWORD m_TextureMinMode;
    CKDWORD m_TextureMagMode;
    VXBLEND_MODE m_SourceBlend;
    VXBLEND_MODE m_DestBlend;
    CKDWORD m_ShadeMode;
    CKDWORD m_FillMode;
    CKDWORD m_TextureAddressMode;
    CKDWORD m_TextureBorderColor;

    /**
     * Packed flags field layout:
     * Bit 0:      TwoSided (1 = two-sided)
     * Bit 1:      ZWrite (1 = enabled)
     * Bit 2:      PerspectiveCorrection (1 = enabled)
     * Bit 3:      AlphaBlend (8 = enabled)
     * Bit 4:      AlphaTest (16 = enabled)
     * Bit 5:      Sprite3DBatch flag
     * Bits 8-13:  Effect (VX_EFFECT, 6 bits)
     * Bits 14-18: ZFunc (VXCMPFUNC, 5 bits)
     * Bits 19-23: AlphaFunc (VXCMPFUNC, 5 bits)
     */
    CKDWORD m_Flags;

    // Alpha reference value for alpha testing (0-255)
    CKDWORD m_AlphaRef;

    // Batch for optimized sprite rendering
    CKSprite3DBatch *m_Sprite3DBatch;

    // User callback and argument
    CK_MATERIALCALLBACK m_Callback;
    void *m_CallbackArgument;

    // Effect parameter for advanced effects
    CKParameter *m_EffectParameter;
};

#endif // RCKMATERIAL_H
