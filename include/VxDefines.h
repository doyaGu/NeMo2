#ifndef VXDEFINES_H
#define VXDEFINES_H

#include "VxMathDefines.h"

#if VX_MSVC > 1000
#pragma warning(disable : 4251)
#endif

class CKRasterizerContext;
class CKRasterizer;
class CKRasterizerDriver;

/**
 * @brief Texture Coordinates structure.
 *
 * @remarks Textures coordinates determine which texel (Texture pixel) is assigned to a specific vertex.
 * @see CKPatchMesh::GetTVs
 */
class VxUV {
public:
    float u; ///< The U-coordinate (horizontal).
    float v; ///< The V-coordinate (vertical).

    /**
     * @brief Default constructor.
     * @param _u Initial U value.
     * @param _v Initial V value.
     */
    VxUV(float _u = 0, float _v = 0) : u(_u), v(_v) {}

    /**
     * @brief Addition assignment operator.
     * @param uv The VxUV to add.
     * @return A reference to this VxUV.
     */
    VxUV &operator+=(const VxUV &uv) {
        u += uv.u;
        v += uv.v;
        return *this;
    }

    /**
     * @brief Subtraction assignment operator.
     * @param uv The VxUV to subtract.
     * @return A reference to this VxUV.
     */
    VxUV &operator-=(const VxUV &uv) {
        u -= uv.u;
        v -= uv.v;
        return *this;
    }

    /**
     * @brief Scalar multiplication assignment operator.
     * @param s The scalar value.
     * @return A reference to this VxUV.
     */
    VxUV &operator*=(float s) {
        u *= s;
        v *= s;
        return *this;
    }

    /**
     * @brief Scalar division assignment operator.
     * @param s The scalar value.
     * @return A reference to this VxUV.
     */
    VxUV &operator/=(float s) {
        u /= s;
        v /= s;
        return *this;
    }

    // =====================================
    // Unary operators
    /**
     * @brief Unary plus operator.
     * @param uv The input VxUV.
     * @return A copy of the input VxUV.
     */
    friend VxUV operator+(const VxUV &uv) { return uv; }
    /**
     * @brief Unary minus operator (negation).
     * @param uv The input VxUV.
     * @return A new VxUV with negated components.
     */
    friend VxUV operator-(const VxUV &uv) { return VxUV(-uv.u, -uv.v); }

    // =====================================
    // Binary operators
    /**
     * @brief Addition operator.
     * @param v1 The first VxUV.
     * @param v2 The second VxUV.
     * @return The sum of the two VxUVs.
     */
    friend VxUV operator+(const VxUV &v1, const VxUV &v2) { return VxUV(v1.u + v2.u, v1.v + v2.v); }
    /**
     * @brief Subtraction operator.
     * @param v1 The first VxUV.
     * @param v2 The second VxUV.
     * @return The difference of the two VxUVs.
     */
    friend VxUV operator-(const VxUV &v1, const VxUV &v2) { return VxUV(v1.u - v2.u, v1.v - v2.v); }
    /**
     * @brief Scalar multiplication operator.
     * @param uv The VxUV.
     * @param s The scalar.
     * @return The scaled VxUV.
     */
    friend VxUV operator*(const VxUV &uv, float s) { return VxUV(uv.u * s, uv.v * s); }
    /**
     * @brief Scalar multiplication operator.
     * @param s The scalar.
     * @param uv The VxUV.
     * @return The scaled VxUV.
     */
    friend VxUV operator*(float s, const VxUV &uv) { return VxUV(uv.u * s, uv.v * s); }
    /**
     * @brief Scalar division operator.
     * @param uv The VxUV.
     * @param s The scalar.
     * @return The scaled VxUV.
     */
    friend VxUV operator/(const VxUV &uv, float s) { return VxUV(uv.u / s, uv.v / s); }
};

/**
 * @brief Vertex format and draw primitive options.
 *
 * @remarks
 * This enumeration is used to specify the format of vertex used when drawing a
 * a primitive and the actions to perform.
 *
 * It can also be used by the CKRenderContext::GetDrawPrimitiveStructure
 * when asking to retrieve a pre-allocated VxDrawPrimitiveData structure. For example
 * calling RenderContext->GetDrawPrimitiveStructure(CKRST_DP_TR_CL_VNT,20);
 * returns a VxDrawPrimitiveData structure with a vertex count of 20. Normals and texture coordinates
 * will be present and when drawing the vertices will be transformed,lit and clipped.
 *
 * If the flags CKRST_DP_VBUFFER is set and the current context supports Vertex Buffers, the
 * returned structure will point to a vertex buffer. It must be released before any rendering through
 * the CKRenderContext::ReleaseCurrentVB method.
 *
 * @see VxDrawPrimitiveData, CKRenderContext::DrawPrimitive, CKRenderContext::GetDrawPrimitiveStructure
 */
typedef enum CKRST_DPFLAGS {
    CKRST_DP_TRANSFORM = 0x00000001UL,
    ///< Transform vertices using the current transformations matrices (see CKRenderContext::SetWorldTransformationMatrix,..)
    CKRST_DP_LIGHT      = 0x00000002UL, ///< Enable lighting and lit vertices if normals information is present.
    CKRST_DP_DOCLIP     = 0x00000004UL, ///< Perform frustum clipping
    CKRST_DP_DIFFUSE    = 0x00000010UL, ///< Diffuse Color
    CKRST_DP_SPECULAR   = 0x00000020UL, ///< Specular Color
    CKRST_DP_STAGESMASK = 0x0001FE00UL, ///< Mask for texture coord sets
    CKRST_DP_STAGES0    = 0x00000200UL, ///< Texture coords up to Stage 0
    CKRST_DP_STAGES1    = 0x00000400UL, ///< Texture coords up to Stage 1
    CKRST_DP_STAGES2    = 0x00000800UL, ///< ...
    CKRST_DP_STAGES3    = 0x00001000UL, ///< ...
    CKRST_DP_STAGES4    = 0x00002000UL, ///< ...
    CKRST_DP_STAGES5    = 0x00004000UL, ///< ...
    CKRST_DP_STAGES6    = 0x00008000UL, ///< ...
    CKRST_DP_STAGES7    = 0x00010000UL, ///< Texture coords up to Stage 7

    CKRST_DP_WEIGHTMASK = 0x01F00000UL, ///< Mask for vertex weight sets
    CKRST_DP_WEIGHTS1   = 0x00100000UL, ///< 1 Weight data is added to each vertex for HW vertex skinning or blending
    CKRST_DP_WEIGHTS2   = 0x00200000UL, ///< 2 Weights data are added to each vertex for HW vertex skinning or blending
    CKRST_DP_WEIGHTS3   = 0x00400000UL, ///< 3 Weights data are added to each vertex for HW vertex skinning or blending
    CKRST_DP_WEIGHTS4   = 0x00800000UL, ///< 4 Weights data are added to each vertex for HW vertex skinning or blending
    CKRST_DP_WEIGHTS5   = 0x01000000UL, ///< 5 Weights data are added to each vertex for HW vertex skinning or blending
    CKRST_DP_MATRIXPAL  = 0x02000000UL,
    ///< The last weight is a DWORD which contains the indices of matrix to which the weights are associated.

    CKRST_DP_VBUFFER = 0x10000000UL,
    ///< If a Vertex Buffer can be created, the returned structure should directly point to it.

    // Common combinations
    CKRST_DP_TR_CL_VNT  = 0x00000207UL, ///< non-transformed vertices with normal and texture coords
    CKRST_DP_TR_CL_VCST = 0x00000235UL, ///< non-transformed vertices with colors (diffuse+specular) and texture coords
    CKRST_DP_TR_CL_VCT  = 0x00000215UL, ///< non-transformed vertices with diffuse color only and texture coords
    CKRST_DP_TR_CL_VCS  = 0x00000035UL,
    ///< non-transformed vertices with colors (diffuse+specular) and no-texture coords
    CKRST_DP_TR_CL_VC = 0x00000015UL, ///< non-transformed vertices with diffuse color only.
    CKRST_DP_TR_CL_V  = 0x00000005U,  ///< non-transformed vertices (white).

    CKRST_DP_CL_VCST = 0x00000234UL, ///< pre-transformed vertices with colors (diffuse+specular) and texture coords
    CKRST_DP_CL_VCT  = 0x00000214UL, ///< pre-transformed vertices with diffuse color and texture coords
    CKRST_DP_CL_VC   = 0x00000014UL, ///< pre-transformed vertices with diffuse color only.
    CKRST_DP_CL_V    = 0x00000004UL, ///< pre-transformed vertices (white).

    CKRST_DP_TR_VNT  = 0x00000203UL, ///< non-transformed vertices with normal and texture coords (No Clipping).
    CKRST_DP_TR_VCST = 0x00000231UL,
    ///< non-transformed vertices with colors (diffuse+specular) and texture coords (No Clipping).
    CKRST_DP_TR_VCT = 0x00000211UL,
    ///< non-transformed vertices with diffuse color only and texture coords (No Clipping).
    CKRST_DP_TR_VCS = 0x00000031UL,
    ///< non-transformed vertices with colors (diffuse+specular) and no-texture coords (No Clipping).
    CKRST_DP_TR_VC = 0x00000011UL, ///< non-transformed vertices with colors (diffuse only) (No Clipping).
    CKRST_DP_TR_V  = 0x00000001UL, ///< non-transformed vertices (white,No Clipping).

    CKRST_DP_V = 0x00000000UL, ///< pre-transformed vertices (White,No Clipping).
    CKRST_DP_VC = 0x00000010UL, ///< pre-transformed vertices with diffuse color only (No Clipping).
    CKRST_DP_VCT = 0x00000210UL, ///< pre-transformed vertices with diffuse color and texture coords only (No Clipping).
    CKRST_DP_VCST = 0x00000230UL
    ///< pre-transformed vertices with diffuse and specular color and texture coords only (No Clipping).
} CKRST_DPFLAGS;

/// @brief Macro to generate the flag for a specific number of vertex weights.
#define CKRST_DP_WEIGHT(x) (x ? (CKRST_DP_WEIGHTS1 << (x - 1)) : 0)
/// @brief Macro to generate the flag for a specific number of vertex weights with indexed matrices.
#define CKRST_DP_IWEIGHT(x) (x ? (CKRST_DP_MATRIXPAL | (CKRST_DP_WEIGHTS1 << (x - 1))) : 0)
/// @brief Macro to generate the flag for a specific texture stage.
#define CKRST_DP_STAGE(i) (CKRST_DP_STAGES0 << i)
/// @brief Macro to extract the stage flags from the combined DPFLAGS.
#define CKRST_DP_STAGEFLAGS(f) ((f & CKRST_DP_STAGESMASK) >> 9)
/// @brief The maximum number of texture stages supported.
#define CKRST_MAX_STAGES 8

/**
 * @brief Simple data structure for drawing primitives.
 * @details Used by CKRenderContext::DrawPrimitive to describe the vertices to render.
 *
 * @remarks
 * If the vertices are already transformed, they should be provided in a VxVector4 form
 * in order the rasterizer to have all information about the vertices(x,y,z,rhw) to draw
 * (rhw = Reciprocal homogenous w used for texture correction).
 *
 * The flags member specifies which vertex data is available and how vertices should be processed.
 * This structure can also point to a vertex buffer (see CKRST_DP_VBUFFER in CKRST_DPFLAGS).
 *
 * @see CKRST_DPFLAGS, CKRenderContext::DrawPrimitive, CKRenderContext::GetDrawPrimitiveStructure
 */
struct VxDrawPrimitiveDataSimple {
    int VertexCount;    ///< Number of vertices to draw
    unsigned int Flags; ///< A combination of CKRST_DPFLAGS
    void *PositionPtr;
    ///< A pointer to position data (VxVector for model coordinates, VxVector4 for screen coordinates)
    unsigned int PositionStride;      ///< Amount in bytes between two positions in the PositionPtr buffer
    void *NormalPtr;                  ///< A pointer to normals data
    unsigned int NormalStride;        ///< Amount in bytes between two normals in the NormalPtr buffer
    void *ColorPtr;                   ///< A pointer to diffuse colors data
    unsigned int ColorStride;         ///< Amount in bytes between two colors in the ColorPtr buffer
    void *SpecularColorPtr;           ///< A pointer to specular colors data
    unsigned int SpecularColorStride; ///< Amount in bytes between two specular colors in the SpecularColorPtr buffer
    void *TexCoordPtr;                ///< A pointer to texture coordinates data for the first stage
    unsigned int TexCoordStride;      ///< Amount in bytes between two texture coordinates in the TexCoordPtr buffer
};

/**
 * @brief Extended data structure for drawing primitives with multiple texture stages.
 */
struct VxDrawPrimitiveData : public VxDrawPrimitiveDataSimple {
    void *TexCoordPtrs[CKRST_MAX_STAGES - 1]; ///< Pointers to texture coordinates data for subsequent texture stages.
    unsigned int TexCoordStrides[CKRST_MAX_STAGES - 1]; ///< Strides for subsequent texture coordinate buffers.
};

/**
 * @brief Display Mode Description.
 *
 * @remarks The VxDisplayMode contains the description of a fullscreen display mode.
 * @see CKRenderManager::GetRenderDriverDescription, CKRenderManager::GetRenderDriverCount, VxDriverDesc
 */
typedef struct VxDisplayMode {
    int Width;       ///< Width in pixel of the display mode.
    int Height;      ///< Height in pixel of the display mode.
    int Bpp;         ///< Number of bits per pixel.
    int RefreshRate; ///< Refresh rate in Hertz.
} VxDisplayMode;

/**
 * @brief Equality operator for VxDisplayMode.
 * @param lhs The left-hand side display mode.
 * @param rhs The right-hand side display mode.
 * @return True if all members are equal, false otherwise.
 */
inline bool operator==(const VxDisplayMode &lhs, const VxDisplayMode &rhs) {
    return lhs.Width == rhs.Width &&
           lhs.Height == rhs.Height &&
           lhs.Bpp == rhs.Bpp &&
           lhs.RefreshRate == rhs.RefreshRate;
}

/**
 * @brief Inequality operator for VxDisplayMode.
 * @param lhs The left-hand side display mode.
 * @param rhs The right-hand side display mode.
 * @return True if any member is not equal, false otherwise.
 */
inline bool operator!=(const VxDisplayMode &lhs, const VxDisplayMode &rhs) {
    return !(rhs == lhs);
}

/**
 * @brief Structure for transforming vertices to screen coordinates.
 *
 * @remarks
 * The VxTransformData structure is used by the CKRenderContext::TransformVertices function
 * to specify vertices to transform and up to two arrays that will receive the transformed
 * homogeneous vertices and the screen coordinates of these vertices.
 *
 * All members except InVertices are optional.
 *
 * The ClipFlags member is an array of DWORD (VXCLIP_FLAGS) containing the clipping flags for each
 * vertex: if one of the flags is set, the vertex is outside the viewing frustum.
 *
 * @see CKRenderContext::TransformVertices, VXCLIP_FLAGS
 */
typedef struct VxTransformData {
    void *InVertices;          ///< VxVector (x,y,z) in model coordinate space
    unsigned int InStride;     ///< Amount in bytes separating two vertices in InVertices array
    void *OutVertices;         ///< VxVector4 Homogenous coordinates (xh,yh,zh,wh)
    unsigned int OutStride;    ///< Amount in bytes separating two vertices in OutVertices array
    void *ScreenVertices;      ///< VxVector4 Screen coordinates (xs,ys,zs,rhw)
    unsigned int ScreenStride; ///< Amount in bytes separating two vertices in ScreenVertices array
    unsigned int *ClipFlags;   ///< Array of DWORD containing a combination of clipping flags (see VXCLIP_FLAGS)
    CKRECT m_2dExtents;        ///< Obsolete: 2D extents of the transformed vertices.
    unsigned int m_Offscreen;
    ///< A combination of all the clipping flags: if !=0, all vertices are clipped by at least one clipping plane.
} VxTransformData;

/**
 * @brief DirectX specific data.
 *
 * @remarks
 * The data stored in this structure is only available when using a DirectX based rasterizer
 * (CKDX8Rasterizer, CKDX7Rasterizer, or CKDX5Rasterizer).
 * The type and version of objects depends on the version of the rasterizer used.
 * - **CKDX8Rasterizer**:
 *   - `DDBackBuffer`: `LPDIRECT3DSURFACE8`
 *   - `DDPrimaryBuffer`: `NULL`
 *   - `DDZBuffer`: `LPDIRECT3DSURFACE8`
 *   - `DirectDraw`: `NULL`
 *   - `Direct3D`: `LPDIRECT3D8`
 *   - `DDClipper`: `NULL`
 *   - `D3DDevice`: `LPDIRECT3DDEVICE8`
 *   - `D3DViewport`: `NULL`
 *   - `DxVersion`: `0x0801`
 *
 * - **CKDX7Rasterizer**:
 *   - `DDBackBuffer`: `LPDIRECTDRAWSURFACE7`
 *   - `DDPrimaryBuffer`: `LPDIRECTDRAWSURFACE7`
 *   - `DDZBuffer`: `LPDIRECTDRAWSURFACE7`
 *   - `DirectDraw`: `LPDIRECTDRAW7`
 *   - `Direct3D`: `LPDIRECT3D7`
 *   - `DDClipper`: `LPDIRECTDRAWCLIPPER`
 *   - `D3DDevice`: `LPDIRECT3DDEVICE7`
 *   - `D3DViewport`: `NULL`
 *   - `DxVersion`: `0x0700`
 *
 * - **CKDX5Rasterizer**:
 *   - `DDBackBuffer`: `LPDIRECTDRAWSURFACE3`
 *   - `DDPrimaryBuffer`: `LPDIRECTDRAWSURFACE3`
 *   - `DDZBuffer`: `LPDIRECTDRAWSURFACE3`
 *   - `DirectDraw`: `LPDIRECTDRAW2`
 *   - `Direct3D`: `LPDIRECT3D2`
 *   - `DDClipper`: `LPDIRECTDRAWCLIPPER`
 *   - `D3DDevice`: `LPDIRECT3DDEVICE2`
 *   - `D3DViewport`: `LPDIRECT3DVIEWPORT2`
 *   - `DxVersion`: `0x0500`
 *
 * @see CKRenderContext::GetDirectXInfo
 */
typedef struct VxDirectXData {
    void *DDBackBuffer;    ///< Pointer to the back buffer surface
    void *DDPrimaryBuffer; ///< Pointer to the front buffer (primary) surface
    void *DDZBuffer;       ///< Pointer to the depth buffer surface
    void *DirectDraw;      ///< Pointer to the IDirectDraw object
    void *Direct3D;        ///< Pointer to the IDirect3D object
    void *DDClipper;       ///< Pointer to the clipper object
    void *D3DDevice;       ///< Pointer to the IDirect3DDevice object
    void *D3DViewport;     ///< Pointer to the IDirect3DViewport object
    XULONG DxVersion;      ///< DirectX Version (e.g., 0x0500 for DirectX 5.0)
} VxDirectXData;

/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/************************ RENDER STATES *************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

/**
 * @brief Flags for locking video memory surfaces.
 *
 * @remarks When accessing a video memory surface, these flags should be
 * used to inform the driver of the intended operation (Write or Read Only).
 * @see CKTexture::LockVideoMemory, CKSprite::LockVideoMemory
 */
typedef enum VX_LOCKFLAGS {
    VX_LOCK_DEFAULT   = 0x00000000, ///< No assumption about the operation.
    VX_LOCK_WRITEONLY = 0x00000001, ///< The surface will only be written to.
    VX_LOCK_READONLY  = 0x00000002, ///< The surface will only be read from.
    VX_LOCK_DISCARD   = 0x00000004  ///< Write operation only; all current content can be discarded.
} VX_LOCKFLAGS;

/**
 * @brief Flags for window resize operations.
 */
typedef enum VX_RESIZE_FLAGS {
    VX_RESIZE_NOMOVE = 0x0001UL, ///< Do not move the window.
    VX_RESIZE_NOSIZE = 0x0002UL  ///< Do not resize the window.
} VX_RESIZE_FLAGS;

/**
 * @brief Light type enumeration.
 *
 * @remarks Used by CKLight::SetType to specify the type of a light.
 * @see CKLight::SetType, CKLight::GetType
 */
typedef enum VXLIGHT_TYPE {
    VX_LIGHTPOINT = 1UL, ///< The Light is a point of light.
    VX_LIGHTSPOT  = 2UL, ///< The light is a spotlight.
    VX_LIGHTDIREC = 3UL, ///< The light is a directional light: light comes from an infinite point.
    VX_LIGHTPARA  = 4UL  ///< Obsolete, do not use.
} VXLIGHT_TYPE;

/**
 * @brief Type of primitive (Triangle, strips, line, points, etc.) to draw.
 *
 * @remarks Used by CKRenderContext::DrawPrimitive to specify the type of primitive to be drawn.
 * @see CKRenderContext::DrawPrimitive
 */
typedef enum VXPRIMITIVETYPE {
    VX_POINTLIST     = 1UL, ///< Draw a list of points.
    VX_LINELIST      = 2UL, ///< Draw a list of lines.
    VX_LINESTRIP     = 3UL, ///< Draw a strip of connected lines.
    VX_TRIANGLELIST  = 4UL, ///< Draw a list of triangles.
    VX_TRIANGLESTRIP = 5UL, ///< Draw a strip of connected triangles.
    VX_TRIANGLEFAN   = 6UL  ///< Draw a fan of connected triangles.
} VXPRIMITIVETYPE;

/**
 * @brief Video Buffer types.
 *
 * @remarks The VXBUFFER_TYPE is used by CKRenderContext::DumpToMemory() method to specify which
 * video buffer to copy.
 * @see CKRenderContext::CopyToVideo, CKRenderContext::DumpToMemory
 */
typedef enum VXBUFFER_TYPE {
    VXBUFFER_BACKBUFFER    = 0x00000001UL, ///< The back buffer.
    VXBUFFER_ZBUFFER       = 0x00000002UL, ///< The depth buffer (Z-buffer).
    VXBUFFER_STENCILBUFFER = 0x00000004UL  ///< The stencil buffer.
} VXBUFFER_TYPE;

/**
 * @brief Texture blend mode flags.
 *
 * @remarks The VXTEXTURE_BLENDMODE is used by CKMaterial::SetTextureBlendMode() to specify how
 * a texture is applied to primitives. It is also used as a value for the CKRST_TSS_TEXTUREMAPBLEND
 * texture stage state.
 * @see Using Materials, CKMaterial, CKTexture, CKMaterial::SetTextureBlendMode, CKRST_TSS_TEXTUREMAPBLEND
 */
typedef enum VXTEXTURE_BLENDMODE {
    VXTEXTUREBLEND_DECAL    = 1UL, ///< Texture color replaces the material color.
    VXTEXTUREBLEND_MODULATE = 2UL,
    ///< Texture and material colors are multiplied. Texture alpha replaces material alpha.
    VXTEXTUREBLEND_DECALALPHA    = 3UL, ///< Texture alpha specifies how material and texture are blended.
    VXTEXTUREBLEND_MODULATEALPHA = 4UL, ///< Texture alpha specifies how material and texture colors are blended.
    VXTEXTUREBLEND_DECALMASK     = 5UL, ///< Decal using a mask.
    VXTEXTUREBLEND_MODULATEMASK  = 6UL, ///< Modulate using a mask.
    VXTEXTUREBLEND_COPY          = 7UL, ///< Equivalent to DECAL.
    VXTEXTUREBLEND_ADD           = 8UL, ///< Texture and material colors are added.
    VXTEXTUREBLEND_DOTPRODUCT3   = 9UL,
    ///< Perform a Dot Product 3 between texture (normal map) and a vector from VXRENDERSTATE_TEXTUREFACTOR.
    VXTEXTUREBLEND_MAX  = 10UL, ///< The maximum of texture and material color.
    VXTEXTUREBLEND_MASK = 0xFUL ///< Mask for all blend modes.
} VXTEXTURE_BLENDMODE;

/**
 * @brief Texture filter mode options.
 *
 * @remarks The VXTEXTURE_FILTERMODE is used by CKMaterial::SetTextureMagMode and CKMaterial::SetTextureMinMode
 * to specify how a texture is filtered. It is also used as a value for CKRST_TSS_MAGFILTER and CKRST_TSS_MINFILTER.
 * @see Using Materials, CKMaterial, CKTexture, CKMaterial::SetTextureMagMode, CKMaterial::SetTextureMinMode, CKRenderContext::SetTextureStageState
 */
typedef enum VXTEXTURE_FILTERMODE {
    VXTEXTUREFILTER_NEAREST          = 1UL,  ///< No filter (point sampling).
    VXTEXTUREFILTER_LINEAR           = 2UL,  ///< Bilinear interpolation.
    VXTEXTUREFILTER_MIPNEAREST       = 3UL,  ///< Mipmapping with point sampling.
    VXTEXTUREFILTER_MIPLINEAR        = 4UL,  ///< Mipmapping with bilinear interpolation (bilinear filtering).
    VXTEXTUREFILTER_LINEARMIPNEAREST = 5UL,  ///< Mipmapping with bilinear interpolation between mipmap levels.
    VXTEXTUREFILTER_LINEARMIPLINEAR  = 6UL,  ///< Trilinear filtering.
    VXTEXTUREFILTER_ANISOTROPIC      = 7UL,  ///< Anisotropic filtering.
    VXTEXTUREFILTER_MASK             = 0xFUL ///< Mask for all filter modes.
} VXTEXTURE_FILTERMODE;

/**
 * @brief Blending mode options.
 *
 * @remarks
 * The VXBLEND_MODE is used by CKMaterial::SetSourceBlend() and SetDestBlend() to specify the blend
 * factors that are used when blending is enabled. (Rs,Gs,Bs,As) are color components of the source pixel
 * (being drawn) and (Rd,Gd,Bd,Ad) are color components of the destination pixel (current pixel on screen).
 *
 * When blending is enabled the final pixel will be equal to:
 * `SrcBlendFactor * SrcPixel + DstBlendFactor * CurrentPixelOnScreen`
 *
 * It is also used as a value for VXRENDERSTATE_SRCBLEND and VXRENDERSTATE_DESTBLEND render states.
 *
 * @see CKMaterial, CKTexture, CKMaterial::SetSourceBlend, CKMaterial::SetDestBlend, CKRenderContext::SetState, CKSprite::SetBlending, VXRENDERSTATE_SRCBLEND, VXRENDERSTATE_DESTBLEND
 */
typedef enum VXBLEND_MODE {
    VXBLEND_ZERO            = 1UL,  ///< Blend factor is (0, 0, 0, 0).
    VXBLEND_ONE             = 2UL,  ///< Blend factor is (1, 1, 1, 1).
    VXBLEND_SRCCOLOR        = 3UL,  ///< Blend factor is (Rs, Gs, Bs, As).
    VXBLEND_INVSRCCOLOR     = 4UL,  ///< Blend factor is (1-Rs, 1-Gs, 1-Bs, 1-As).
    VXBLEND_SRCALPHA        = 5UL,  ///< Blend factor is (As, As, As, As).
    VXBLEND_INVSRCALPHA     = 6UL,  ///< Blend factor is (1-As, 1-As, 1-As, 1-As).
    VXBLEND_DESTALPHA       = 7UL,  ///< Blend factor is (Ad, Ad, Ad, Ad).
    VXBLEND_INVDESTALPHA    = 8UL,  ///< Blend factor is (1-Ad, 1-Ad, 1-Ad, 1-Ad).
    VXBLEND_DESTCOLOR       = 9UL,  ///< Blend factor is (Rd, Gd, Bd, Ad).
    VXBLEND_INVDESTCOLOR    = 10UL, ///< Blend factor is (1-Rd, 1-Gd, 1-Bd, 1-Ad).
    VXBLEND_SRCALPHASAT     = 11UL, ///< Blend factor is (f, f, f, 1); where f = min(As, 1-Ad).
    VXBLEND_BOTHSRCALPHA    = 12UL, ///< Source: (As, As, As, As), Destination: (1-As, 1-As, 1-As, 1-As).
    VXBLEND_BOTHINVSRCALPHA = 13UL, ///< Source: (1-As, 1-As, 1-As, 1-As), Destination: (As, As, As, As).
    VXBLEND_MASK            = 0xFUL ///< Mask for all blend modes.
} VXBLEND_MODE;

/**
 * @brief Texture addressing modes.
 *
 * @remarks The VXTEXTURE_ADDRESSMODE is used by CKMaterial::SetTextureAddressMode to specify how
 * texture coordinates are handled when they are outside the range [0.0, 1.0].
 * It is also used as a value for the CKRST_TSS_ADDRESS texture stage state.
 * @see CKMaterial, CKTexture, CKRST_TSS_ADDRESS, CKRenderContext::SetTextureStageState
 */
typedef enum VXTEXTURE_ADDRESSMODE {
    VXTEXTURE_ADDRESSWRAP       = 1UL,  ///< Repeats the texture (tiling).
    VXTEXTURE_ADDRESSMIRROR     = 2UL,  ///< Texture coordinates are flipped at every integer boundary.
    VXTEXTURE_ADDRESSCLAMP      = 3UL,  ///< Texture coordinates outside [0,1] are clamped to the nearest edge.
    VXTEXTURE_ADDRESSBORDER     = 4UL,  ///< Coordinates outside [0,1] use a specified border color.
    VXTEXTURE_ADDRESSMIRRORONCE = 5UL,  ///< Texture coordinates are mirrored once and then clamped.
    VXTEXTURE_ADDRESSMASK       = 0x7UL ///< Mask for all address modes.
} VXTEXTURE_ADDRESSMODE;

/**
 * @brief Fill Mode Options.
 *
 * @remarks The VXFILL_MODE is used by CKMaterial::SetFillMode to specify how faces are drawn.
 * It is also used as a value for the VXRENDERSTATE_FILLMODE render state.
 * @see CKMaterial::SetFillMode, VXRENDERSTATE_FILLMODE
 */
typedef enum VXFILL_MODE {
    VXFILL_POINT     = 1UL, ///< Render vertices as points.
    VXFILL_WIREFRAME = 2UL, ///< Render edges as lines.
    VXFILL_SOLID     = 3UL, ///< Render faces as solid polygons.
    VXFILL_MASK      = 3UL  ///< Mask for all fill modes.
} VXFILL_MODE;

/**
 * @brief Shade Mode Options.
 *
 * @remarks The VXSHADE_MODE is used by CKMaterial::SetShadeMode to specify how color
 * interpolation is performed on faces. It is also used as a value for the VXRENDERSTATE_SHADEMODE render state.
 * @see CKMaterial::SetShadeMode, VXRENDERSTATE_SHADEMODE
 */
typedef enum VXSHADE_MODE {
    VXSHADE_FLAT    = 1UL, ///< Flat shading (one color per face).
    VXSHADE_GOURAUD = 2UL, ///< Gouraud shading (color interpolated between vertices).
    VXSHADE_PHONG   = 3UL, ///< Phong shading (Not yet supported by most implementations).
    VXSHADE_MASK    = 3UL  ///< Mask for all shade modes.
} VXSHADE_MODE;

/**
 * @brief Backface culling options.
 *
 * @remarks Used by CKRenderContext::SetState(VXRENDERSTATE_CULLMODE) to specify the type of backface culling.
 * @see CKRenderContext::SetState, VXRENDERSTATE_CULLMODE
 */
typedef enum VXCULL {
    VXCULL_NONE = 1UL, ///< No backface culling.
    VXCULL_CW   = 2UL, ///< Cull faces with clockwise-ordered vertices.
    VXCULL_CCW  = 3UL, ///< Cull faces with counter-clockwise-ordered vertices.
    VXCULL_MASK = 3UL  ///< Mask for all culling modes.
} VXCULL;

/**
 * @brief Comparison Functions.
 *
 * @remarks Used by CKRenderContext::SetState with VXRENDERSTATE_ZFUNC, VXRENDERSTATE_ALPHAFUNC, or VXRENDERSTATE_STENCILFUNC
 * to specify the type of Z, Alpha, or Stencil comparison function. The function compares a reference value to an entry in the respective buffer.
 * @see CKRenderContext::SetState, VXRENDERSTATETYPE, VXRENDERSTATE_ZFUNC, VXRENDERSTATE_ALPHAFUNC, VXRENDERSTATE_STENCILFUNC
 */
typedef enum VXCMPFUNC {
    VXCMP_NEVER        = 1UL,  ///< Always fail the test.
    VXCMP_LESS         = 2UL,  ///< Accept if the new value is less than the existing value.
    VXCMP_EQUAL        = 3UL,  ///< Accept if the new value is equal to the existing value.
    VXCMP_LESSEQUAL    = 4UL,  ///< Accept if the new value is less than or equal to the existing value.
    VXCMP_GREATER      = 5UL,  ///< Accept if the new value is greater than the existing value.
    VXCMP_NOTEQUAL     = 6UL,  ///< Accept if the new value is not equal to the existing value.
    VXCMP_GREATEREQUAL = 7UL,  ///< Accept if the new value is greater than or equal to the existing value.
    VXCMP_ALWAYS       = 8UL,  ///< Always pass the test.
    VXCMP_MASK         = 0xFUL ///< Mask for all comparison functions.
} VXCMPFUNC;

/**
 * @brief Sprite rendering options.
 * @see VxSpriteRenderOptions, CKSprite::SetRenderOptions, CKSprite::GetRenderOptions
 */
typedef enum VXSPRITE_RENDEROPTIONS {
    VXSPRITE_NONE      = 0x00000000UL, ///< Default sprite rendering: no blending, filtering, or modulation.
    VXSPRITE_ALPHATEST = 0x00000001UL, ///< Alpha test is enabled.
    VXSPRITE_BLEND     = 0x00000002UL, ///< Blending is enabled.
    VXSPRITE_MODULATE  = 0x00000004UL, ///< Color modulation is enabled.
    VXSPRITE_FILTER    = 0x00000008UL  ///< Bi-linear filtering is enabled.
} VXSPRITE_RENDEROPTIONS;

/**
 * @brief Additional sprite rendering options.
 * @see VxSpriteRenderOptions, CKSprite::SetRenderOptions, CKSprite::GetRenderOptions
 */
typedef enum VXSPRITE_RENDEROPTIONS2 {
    VXSPRITE2_NONE                  = 0x00000000UL, ///< Default sprite rendering.
    VXSPRITE2_DISABLE_AA_CORRECTION = 0x00000001UL
    ///< Disable anti-aliasing special processing on the sprite (e.g., UV offset).
} VXSPRITE_RENDEROPTIONS2;

/**
 * @brief Sprite rendering options structure.
 *
 * @remarks The default rendering of a sprite does not perform any blending, filtering, or color modulation.
 * This structure allows customizing these settings via CKSprite::SetRenderOptions.
 * @see VXSPRITE_RENDEROPTIONS, CKSprite::SetRenderOptions, CKSprite::GetRenderOptions
 */
struct VxSpriteRenderOptions {
    XULONG ModulateColor;          ///< A DWORD ARGB Color to use for modulation if VXSPRITE_MODULATE is enabled.
    XULONG Options            : 4; ///< A combination of VXSPRITE_RENDEROPTIONS.
    VXCMPFUNC AlphaTestFunc   : 4; ///< The alpha test comparison function if VXSPRITE_ALPHATEST is enabled.
    VXBLEND_MODE SrcBlendMode : 4; ///< The source blend mode if VXSPRITE_BLEND is enabled.
    XULONG Options2           : 4; ///< A combination of VXSPRITE_RENDEROPTIONS2.
    VXBLEND_MODE DstBlendMode : 8; ///< The destination blend mode if VXSPRITE_BLEND is enabled.
    XBYTE AlphaRefValue       : 8; ///< The reference value for the alpha test if VXSPRITE_ALPHATEST is enabled.
};

/**
 * @brief Stencil buffer operations.
 *
 * @remarks Describes the operations for the VXRENDERSTATE_STENCILFAIL, VXRENDERSTATE_STENCILZFAIL,
 * and VXRENDERSTATE_STENCILPASS render states.
 * @see VXRENDERSTATETYPE, CKRenderContext::SetState
 */
typedef enum VXSTENCILOP {
    VXSTENCILOP_KEEP    = 1UL,  ///< Keep the current stencil value.
    VXSTENCILOP_ZERO    = 2UL,  ///< Set the stencil value to zero.
    VXSTENCILOP_REPLACE = 3UL,  ///< Set the stencil value to the reference value (VXRENDERSTATE_STENCILREF).
    VXSTENCILOP_INCRSAT = 4UL,  ///< Increment the stencil value, clamping at the maximum value.
    VXSTENCILOP_DECRSAT = 5UL,  ///< Decrement the stencil value, clamping at zero.
    VXSTENCILOP_INVERT  = 6UL,  ///< Invert the bits of the stencil value.
    VXSTENCILOP_INCR    = 7UL,  ///< Increment the stencil value, wrapping to zero on overflow.
    VXSTENCILOP_DECR    = 8UL,  ///< Decrement the stencil value, wrapping on underflow.
    VXSTENCILOP_MASK    = 0xFUL ///< Mask for all stencil operations.
} VXSTENCILOP;

/**
 * @brief Fog calculation modes.
 *
 * @remarks Used by CKRenderContext::SetFogMode to specify the type of fog to apply to the scene.
 * Also used as a value for the VXRENDERSTATE_FOGMODE render state.
 * @see CKRenderContext::SetFogMode, VXRENDERSTATE_FOGMODE, CKRenderContext::SetState
 */
typedef enum VXFOG_MODE {
    VXFOG_NONE   = 0UL, ///< No fog (Default).
    VXFOG_EXP    = 1UL, ///< Exponential fog.
    VXFOG_EXP2   = 2UL, ///< Square exponential fog.
    VXFOG_LINEAR = 3UL  ///< Linear fog.
} VXFOG_MODE;

/**
 * @brief Texture stage operations (for DX7/DX8 level hardware).
 * @details Defines how texture color and alpha are combined.
 */
typedef enum CKRST_TEXTUREOP {
    CKRST_TOP_DISABLE                   = 1UL,  ///< Disable texture stage.
    CKRST_TOP_SELECTARG1                = 2UL,  ///< Result is Arg1.
    CKRST_TOP_SELECTARG2                = 3UL,  ///< Result is Arg2.
    CKRST_TOP_MODULATE                  = 4UL,  ///< Result is Arg1 * Arg2.
    CKRST_TOP_MODULATE2X                = 5UL,  ///< Result is Arg1 * Arg2 * 2.
    CKRST_TOP_MODULATE4X                = 6UL,  ///< Result is Arg1 * Arg2 * 4.
    CKRST_TOP_ADD                       = 7UL,  ///< Result is Arg1 + Arg2.
    CKRST_TOP_ADDSIGNED                 = 8UL,  ///< Result is Arg1 + Arg2 - 0.5.
    CKRST_TOP_ADDSIGNED2X               = 9UL,  ///< Result is (Arg1 + Arg2 - 0.5) * 2.
    CKRST_TOP_SUBTRACT                  = 10UL, ///< Result is Arg1 - Arg2.
    CKRST_TOP_ADDSMOOTH                 = 11UL, ///< Result is Arg1 + Arg2 - Arg1 * Arg2.
    CKRST_TOP_BLENDDIFFUSEALPHA         = 12UL, ///< Blend using diffuse alpha.
    CKRST_TOP_BLENDTEXTUREALPHA         = 13UL, ///< Blend using texture alpha.
    CKRST_TOP_BLENDFACTORALPHA          = 14UL, ///< Blend using texture factor alpha.
    CKRST_TOP_BLENDTEXTUREALPHAPM       = 15UL, ///< Blend using pre-multiplied texture alpha.
    CKRST_TOP_BLENDCURRENTALPHA         = 16UL, ///< Blend using current alpha.
    CKRST_TOP_PREMODULATE               = 17UL, ///< Pre-modulate color.
    CKRST_TOP_MODULATEALPHA_ADDCOLOR    = 18UL, ///< Modulate alpha, add color.
    CKRST_TOP_MODULATECOLOR_ADDALPHA    = 19UL, ///< Modulate color, add alpha.
    CKRST_TOP_MODULATEINVALPHA_ADDCOLOR = 20UL, ///< Modulate inverse alpha, add color.
    CKRST_TOP_MODULATEINVCOLOR_ADDALPHA = 21UL, ///< Modulate inverse color, add alpha.
    CKRST_TOP_BUMPENVMAP                = 22UL, ///< Bump environment mapping.
    CKRST_TOP_BUMPENVMAPLUMINANCE       = 23UL, ///< Bump environment mapping with luminance.
    CKRST_TOP_DOTPRODUCT3               = 24UL, ///< 3D dot product.
    CKRST_TOP_MULTIPLYADD               = 25UL, ///< Result = Arg1 * Arg2 + Arg3
    CKRST_TOP_LERP                      = 26UL  ///< Result = (Arg1) * Arg3 + (1-Arg1) * Arg2
} CKRST_TEXTUREOP;

/**
 * @brief Texture stage arguments (for DX7/DX8 level hardware).
 * @details Defines the source for arguments in texture stage operations.
 */
typedef enum CKRST_TEXTUREARG {
    CKRST_TA_DIFFUSE        = 0x00000000UL, ///< Use diffuse color from vertex/material.
    CKRST_TA_CURRENT        = 0x00000001UL, ///< Use the result of the previous texture stage.
    CKRST_TA_TEXTURE        = 0x00000002UL, ///< Use the color from the texture of this stage.
    CKRST_TA_TFACTOR        = 0x00000003UL, ///< Use the color from the RENDERSTATE_TEXTUREFACTOR.
    CKRST_TA_SPECULAR       = 0x00000004UL, ///< Use specular color from vertex/material.
    CKRST_TA_TEMP           = 0x00000005UL, ///< Use a temporary register.
    CKRST_TA_COMPLEMENT     = 0x00000010UL, ///< Take 1.0 - x of the argument.
    CKRST_TA_ALPHAREPLICATE = 0x00000020UL  ///< Replicate alpha to all color components (A,A,A).
} CKRST_TEXTUREARG;

/**
 * @brief Texture transform flags (for DX7/DX8 level hardware).
 * @details Controls how texture coordinates are transformed before being used.
 */
typedef enum CKRST_TEXTURETRANSFORMFLAGS {
    CKRST_TTF_NONE      = 0x00000000UL, ///< Texture coordinates are passed directly.
    CKRST_TTF_COUNT1    = 0x00000001UL, ///< The rasterizer should expect 1-D texture coordinates.
    CKRST_TTF_COUNT2    = 0x00000002UL, ///< The rasterizer should expect 2-D texture coordinates.
    CKRST_TTF_COUNT3    = 0x00000003UL, ///< The rasterizer should expect 3-D texture coordinates.
    CKRST_TTF_COUNT4    = 0x00000004UL, ///< The rasterizer should expect 4-D texture coordinates.
    CKRST_TTF_PROJECTED = 0x00000100UL, ///< Divide all coordinates by the last one (projective texturing).
} CKRST_TEXTURETRANSFORMFLAGS;

/// @brief Macro to pack source and destination blend modes for CKRST_TSS_STAGEBLEND.
#define STAGEBLEND(Src, Dst) ((Src << 4) | Dst)

/**
 * @brief Texture stage render states.
 *
 * @remarks These states define the texture filtering, blending, or addressing modes
 * for a specific texture stage. Each time a mesh, sprite3d, or 2dEntity is drawn, it automatically
 * sets the appropriate render states for its rendering.
 * @see CKRenderContext::SetTextureStageState, CKRenderContext::SetState
 */
typedef enum CKRST_TEXTURESTAGESTATETYPE {
    CKRST_TSS_OP            = 1UL,  ///< CKRST_TEXTUREOP (DirectX/OpenGL Usage Only)
    CKRST_TSS_ARG1          = 2UL,  ///< CKRST_TEXTUREARG (DirectX Usage Only)
    CKRST_TSS_ARG2          = 3UL,  ///< CKRST_TEXTUREARG (DirectX Usage Only)
    CKRST_TSS_AOP           = 4UL,  ///< CKRST_TEXTUREOP for alpha channel (DirectX Usage Only)
    CKRST_TSS_AARG1         = 5UL,  ///< CKRST_TEXTUREARG for alpha channel (DirectX Usage Only)
    CKRST_TSS_AARG2         = 6UL,  ///< CKRST_TEXTUREARG for alpha channel (DirectX Usage Only)
    CKRST_TSS_BUMPENVMAT00  = 7UL,  ///< float (DirectX Usage Only)
    CKRST_TSS_BUMPENVMAT01  = 8UL,  ///< float (DirectX Usage Only)
    CKRST_TSS_BUMPENVMAT10  = 9UL,  ///< float (DirectX Usage Only)
    CKRST_TSS_BUMPENVMAT11  = 10UL, ///< float (DirectX Usage Only)
    CKRST_TSS_TEXCOORDINDEX = 11UL, ///< int, texture coordinate set index (DirectX Usage Only)
    CKRST_TSS_ADDRESS       = 12UL, ///< VXTEXTURE_ADDRESSMODE for U, V, and W
    CKRST_TSS_ADDRESSU      = 13UL, ///< VXTEXTURE_ADDRESSMODE for U
    CKRST_TSS_ADDRESSV      = 14UL, ///< VXTEXTURE_ADDRESSMODE for V
    CKRST_TSS_BORDERCOLOR   = 15UL, ///< DWORD RGBA border color
    CKRST_TSS_MAGFILTER     = 16UL, ///< VXTEXTURE_FILTERMODE for magnification
    CKRST_TSS_MINFILTER     = 17UL, ///< VXTEXTURE_FILTERMODE for minification

    CKRST_TSS_MIPMAPLODBIAS  = 19UL, ///< float, Mip-map level of detail bias
    CKRST_TSS_MAXMIPMLEVEL   = 20UL, ///< int, Maximum mip-map level
    CKRST_TSS_MAXANISOTROPY  = 21UL, ///< int, Maximum anisotropy
    CKRST_TSS_BUMPENVLSCALE  = 22UL, ///< float, Bump map luminance scale
    CKRST_TSS_BUMPENVLOFFSET = 23UL, ///< float, Bump map luminance offset

    CKRST_TSS_TEXTURETRANSFORMFLAGS = 24UL, ///< CKRST_TEXTURETRANSFORMFLAGS

    CKRST_TSS_ADDRESW    = 25UL, ///< VXTEXTURE_ADDRESSMODE for W
    CKRST_TSS_COLORARG0  = 26UL, ///< CKRST_TEXTUREARG (DirectX Usage Only)
    CKRST_TSS_ALPHAARG0  = 27UL, ///< CKRST_TEXTUREARG (DirectX Usage Only)
    CKRST_TSS_RESULTARG0 = 28UL, ///< CKRST_TEXTUREARG (DirectX Usage Only)

    CKRST_TSS_TEXTUREMAPBLEND = 39UL, ///< VXTEXTURE_BLENDMODE, legacy texture blend mode
    CKRST_TSS_STAGEBLEND      = 40UL, ///< Use STAGEBLEND macro for mono-pass multitexturing test.
    CKRST_TSS_MAXSTATE        = 41UL  ///< Marks the end of the texture stage states.
} CKRST_TEXTURESTAGESTATETYPE;

/**
 * @brief Automatic texture coordinate generation modes.
 * @details Can be combined with the texture coordinate index in CKRST_TSS_TEXCOORDINDEX.
 */
enum VXTEXCOORD_GEN {
    VXTEXCOORD_SKIP         = 0,   ///< Use the specified texture coordinate set.
    VXTEXCOORD_PROJNORMAL   = 0x1, ///< Texcoords = vertex normal, transformed to camera space.
    VXTEXCOORD_PROJPOSITION = 0x2, ///< Texcoords = vertex position, transformed to camera space.
    VXTEXCOORD_PROJREFLECT  = 0x3, ///< Texcoords = reflection vector, transformed to camera space.
    VXTEXCOORD_MASK         = 0x3  ///< Mask for all generation modes.
};

/**
 * @brief Texture coordinates wrapping mode.
 */
typedef enum VXWRAP_MODE {
    VXWRAP_U    = 0x00000001UL, ///<	Wrap among U texture coordinates.
    VXWRAP_V    = 0x00000002UL, ///<	Wrap among V texture coordinates.
    VXWRAP_S    = 0x00000004UL, ///<	Wrap among S texture coordinates (3D textures).
    VXWRAP_T    = 0x00000008UL, ///<	Wrap among T texture coordinates (3D textures).
    VXWRAP_MASK = 0x000FUL      ///< Mask for all wrap modes.
} VXWRAP_MODE;

/**
 * @brief Blend operations for combining source and destination pixels.
 */
typedef enum VXBLENDOP {
    VXBLENDOP_ADD         = 0x00000001L, ///< Result = Source + Destination
    VXBLENDOP_SUBTRACT    = 0x00000002L, ///< Result = Source - Destination
    VXBLENDOP_REVSUBTRACT = 0x00000003L, ///< Result = Destination - Source
    VXBLENDOP_MIN         = 0x00000004L, ///< Result = min(Source, Destination)
    VXBLENDOP_MAX         = 0x00000005L, ///< Result = max(Source, Destination)
    VXBLENDOP_MASK        = 0x00000007UL ///< Mask for all blend operations.
} VXBLENDOP;

/**
 * @brief Flags for hardware vertex blending (skinning).
 */
typedef enum VXVERTEXBLENDFLAGS {
    VXVBLEND_DISABLE  = 0x00000000UL, ///< Disable vertex blending.
    VXVBLEND_1WEIGHTS = 0x00000001UL, ///< Blend using 1 weight per vertex.
    VXVBLEND_2WEIGHTS = 0x00000002UL, ///< Blend using 2 weights per vertex.
    VXVBLEND_3WEIGHTS = 0x00000003UL, ///< Blend using 3 weights per vertex.
    VXVBLEND_TWEENING = 0x000000FFUL, ///< Use tweening between two vertex sets.
    VXVBLEND_0WEIGHTS = 0x00000100UL  ///< Same as 1 weight, but with weight 0 being implicit.
} VXVERTEXBLENDFLAGS;

/**
 * @brief Global rendering states.
 *
 * @remarks Through VXRENDERSTATETYPE, one can specify various modes for rendering.
 * Most of the time these settings are automatically set by the render engine according to textures,
 * objects and materials properties. Using SetState enables one to override the default behavior,
 * especially in render callbacks.
 *
 * @see CKRenderContext, CKRenderContext::SetState
 */
typedef enum VXRENDERSTATETYPE {
    VXRENDERSTATE_ANTIALIAS          = 2,   ///< Antialiasing mode (TRUE/FALSE)
    VXRENDERSTATE_TEXTUREPERSPECTIVE = 4,   ///< Enable Perspective correction (TRUE/FALSE)
    VXRENDERSTATE_ZENABLE            = 7,   ///< Enable z-buffer test (TRUE/FALSE)
    VXRENDERSTATE_FILLMODE           = 8,   ///< Fill mode (VXFILL_MODE)
    VXRENDERSTATE_SHADEMODE          = 9,   ///< Shade mode (VXSHADE_MODE)
    VXRENDERSTATE_LINEPATTERN        = 10,  ///< Line pattern (bit pattern in a DWORD)
    VXRENDERSTATE_ZWRITEENABLE       = 14,  ///< Enable z-buffer writes (TRUE/FALSE)
    VXRENDERSTATE_ALPHATESTENABLE    = 15,  ///< Enable alpha tests (TRUE/FALSE)
    VXRENDERSTATE_SRCBLEND           = 19,  ///< Blend factor for source (VXBLEND_MODE)
    VXRENDERSTATE_DESTBLEND          = 20,  ///< Blend factor for destination (VXBLEND_MODE)
    VXRENDERSTATE_CULLMODE           = 22,  ///< Back-face culling mode (VXCULL)
    VXRENDERSTATE_ZFUNC              = 23,  ///< Z-comparison function (VXCMPFUNC)
    VXRENDERSTATE_ALPHAREF           = 24,  ///< Reference alpha value (DWORD 0-255)
    VXRENDERSTATE_ALPHAFUNC          = 25,  ///< Alpha-comparison function (VXCMPFUNC)
    VXRENDERSTATE_DITHERENABLE       = 26,  ///< Enable dithering (TRUE/FALSE)
    VXRENDERSTATE_ALPHABLENDENABLE   = 27,  ///< Enable alpha blending (TRUE/FALSE)
    VXRENDERSTATE_FOGENABLE          = 28,  ///< Enable fog (TRUE/FALSE)
    VXRENDERSTATE_SPECULARENABLE     = 29,  ///< Enable specular highlights (TRUE/FALSE)
    VXRENDERSTATE_FOGCOLOR           = 34,  ///< Fog color (DWORD ARGB)
    VXRENDERSTATE_FOGPIXELMODE       = 35,  ///< Fog mode for per-pixel fog (VXFOG_MODE)
    VXRENDERSTATE_FOGSTART           = 36,  ///< Fog start distance (float)
    VXRENDERSTATE_FOGEND             = 37,  ///< Fog end distance (float)
    VXRENDERSTATE_FOGDENSITY         = 38,  ///< Fog density (float)
    VXRENDERSTATE_EDGEANTIALIAS      = 40,  ///< Antialias edges (TRUE/FALSE)
    VXRENDERSTATE_ZBIAS              = 47,  ///< Z-bias for co-planar polygons (DWORD 0-16)
    VXRENDERSTATE_RANGEFOGENABLE     = 48,  ///< Enables range-based fog (TRUE/FALSE)
    VXRENDERSTATE_STENCILENABLE      = 52,  ///< Enable or disable stenciling (TRUE/FALSE)
    VXRENDERSTATE_STENCILFAIL        = 53,  ///< Stencil operation on stencil fail (VXSTENCILOP)
    VXRENDERSTATE_STENCILZFAIL       = 54,  ///< Stencil operation on Z-fail (VXSTENCILOP)
    VXRENDERSTATE_STENCILPASS        = 55,  ///< Stencil operation on stencil and Z pass (VXSTENCILOP)
    VXRENDERSTATE_STENCILFUNC        = 56,  ///< Stencil comparison function (VXCMPFUNC)
    VXRENDERSTATE_STENCILREF         = 57,  ///< Reference value for stencil test (DWORD 0-255)
    VXRENDERSTATE_STENCILMASK        = 58,  ///< Mask for stencil test (DWORD)
    VXRENDERSTATE_STENCILWRITEMASK   = 59,  ///< Stencil buffer write mask (DWORD)
    VXRENDERSTATE_TEXTUREFACTOR      = 60,  ///< Texture factor color (DWORD ARGB)
    VXRENDERSTATE_WRAP0              = 128, ///< Wrap flags for 1st texture coord set (VXWRAP_MODE)
    VXRENDERSTATE_WRAP1              = 129, ///< Wrap flags for 2nd texture coord set (VXWRAP_MODE)
    VXRENDERSTATE_WRAP2              = 130, ///< Wrap flags for 3rd texture coord set (VXWRAP_MODE)
    VXRENDERSTATE_WRAP3              = 131, ///< Wrap flags for 4th texture coord set (VXWRAP_MODE)
    VXRENDERSTATE_WRAP4              = 132, ///< Wrap flags for 5th texture coord set (VXWRAP_MODE)
    VXRENDERSTATE_WRAP5              = 133, ///< Wrap flags for 6th texture coord set (VXWRAP_MODE)
    VXRENDERSTATE_WRAP6              = 134, ///< Wrap flags for 7th texture coord set (VXWRAP_MODE)
    VXRENDERSTATE_WRAP7              = 135, ///< Wrap flags for last texture coord set
    VXRENDERSTATE_CLIPPING           = 136, ///< Enable or disable primitive clipping (TRUE/FALSE)
    VXRENDERSTATE_LIGHTING           = 137, ///< Enable or disable lighting (TRUE/FALSE)
    VXRENDERSTATE_AMBIENT            = 139, ///< Ambient color for scene (DWORD ARGB)
    VXRENDERSTATE_FOGVERTEXMODE      = 140, ///< Fog mode for per-vertex fog (VXFOG_MODE)
    VXRENDERSTATE_COLORVERTEX        = 141, ///< Enable or disable per-vertex color
    VXRENDERSTATE_LOCALVIEWER        = 142, ///< Camera relative specular highlights (TRUE/FALSE)
    VXRENDERSTATE_NORMALIZENORMALS   = 143, ///< Enable automatic normalization of vertex normals
    VXRENDERSTATE_DIFFUSEFROMVERTEX  = 145, ///< If TRUE, diffuse color is from vertex; otherwise from material.
    VXRENDERSTATE_SPECULARFROMVERTEX = 146, ///< If TRUE, specular color is from vertex; otherwise from material.
    VXRENDERSTATE_AMBIENTFROMVERTEX  = 147, ///< If TRUE, ambient color is from vertex; otherwise from material.
    VXRENDERSTATE_EMISSIVEFROMVERTEX = 148, ///< If TRUE, emissive color is from vertex; otherwise from material.

    VXRENDERSTATE_VERTEXBLEND         = 151, ///< Set vertex blending mode (VXVERTEXBLENDFLAGS)
    VXRENDERSTATE_SOFTWAREVPROCESSING = 153, ///< Force software vertex processing in mixed mode T&L driver (TRUE/FALSE)

    VXRENDERSTATE_POINTSIZE         = 154, ///< Size of point sprites (float)
    VXRENDERSTATE_POINTSIZE_MIN     = 155, ///< Minimum size of point sprites (float)
    VXRENDERSTATE_POINTSIZE_MAX     = 166, ///< Maximum size of point sprites (float)
    VXRENDERSTATE_POINTSPRITEENABLE = 156, ///< Enable point sprites (TRUE/FALSE)

    VXRENDERSTATE_POINTSCALEENABLE = 157, ///< If TRUE, point size is attenuated by distance.
    VXRENDERSTATE_POINTSCALE_A     = 158, ///< Constant attenuation factor for point size (float)
    VXRENDERSTATE_POINTSCALE_B     = 159, ///< Linear attenuation factor for point size (float)
    VXRENDERSTATE_POINTSCALE_C     = 160, ///< Quadratic attenuation factor for point size (float)

    VXRENDERSTATE_CLIPPLANEENABLE   = 152, ///< Enable user-defined clipping planes (DWORD mask)
    VXRENDERSTATE_INDEXVBLENDENABLE = 167, ///< Enable indexed vertex blending (TRUE/FALSE)
    VXRENDERSTATE_BLENDOP           = 171, ///< Set blending operation (VXBLENDOP)

    // Virtools Specific Render States
    VXRENDERSTATE_TEXTURETARGET  = 253, ///< Hint: context is used to render to a texture (TRUE/FALSE)
    VXRENDERSTATE_INVERSEWINDING = 254, ///< Invert Cull CW and Cull CCW (TRUE/FALSE)
    VXRENDERSTATE_MAXSTATE       = 256, ///< Marks the end of render states.
    VXRENDERSTATE_FORCE_DWORD    = 0x7fffffff
} VXRENDERSTATETYPE;

/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/***************** DRIVER INFO AND CAPS *************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

/**
 * @brief Bits per pixel constants.
 *
 * @remarks Used to specify available bit-per-pixel modes for a buffer (Depth, Color, or Stencil).
 * @see Vx3DCapsDesc
 */
typedef enum VxBpps {
    VX_BPP1  = 0x00004000UL, ///< 1 bit per pixel
    VX_BPP2  = 0x00002000UL, ///< 2 bits per pixel
    VX_BPP4  = 0x00001000UL, ///< 4 bits per pixel
    VX_BPP8  = 0x00000800UL, ///< 8 bits per pixel
    VX_BPP16 = 0x00000400UL, ///< 16 bits per pixel
    VX_BPP24 = 0x00000200UL, ///< 24 bits per pixel
    VX_BPP32 = 0x00000100UL  ///< 32 bits per pixel
} VxBpps;

/**
 * @brief Rasterizer implementation family.
 */
typedef enum CKRST_RSTFAMILY {
    CKRST_DIRECTX = 0UL, ///< DirectX-based rasterizer.
    CKRST_OPENGL  = 1UL, ///< OpenGL-based rasterizer.
    CKRST_SOFT    = 3UL, ///< Software rasterizer.
    CKRST_ALCHEMY = 5UL, ///< Alchemy-based rasterizer (specific hardware).
    CKRST_UNKNOWN = 4UL  ///< Unknown rasterizer family.
} CKRST_RSTFAMILY;

/**
 * @brief 2D capabilities of a render driver.
 * @see VxDriverDesc, CKRenderManager::GetRenderDriverDescription
 */
typedef struct Vx2DCapsDesc {
    CKRST_RSTFAMILY Family;      ///< The type of driver implementation (DirectX, OpenGL, etc.).
    XULONG MaxVideoMemory;       ///< Maximum video memory (in bytes).
    XULONG AvailableVideoMemory; ///< Available video memory (in bytes).
    XULONG Caps;                 ///< General 2D capabilities (CKRST_2DCAPS flags).
} Vx2DCapsDesc;

/**
 * @brief 3D capabilities of a render driver.
 *
 * @remarks The Vx3DCapsDesc contains the 3D capabilities of a render driver,
 * such as texture size limitations, filtering capabilities, blending capabilities,
 * multi-texturing capabilities, etc.
 * @see VxDriverDesc, CKRenderManager::GetRenderDriverDescription
 */
typedef struct Vx3DCapsDesc {
    XULONG DevCaps;               ///< Unused device capabilities.
    XULONG RenderBpps;            ///< Supported pixel formats for a 3D device (combination of VxBpps).
    XULONG ZBufferBpps;           ///< Supported pixel formats for Z-buffer (combination of VxBpps).
    XULONG StencilBpps;           ///< Supported pixel formats for Stencil buffer (combination of VxBpps).
    XULONG StencilCaps;           ///< Stencil capabilities (CKRST_STENCILCAPS flags).
    XULONG MinTextureWidth;       ///< Minimum allowed width for a texture.
    XULONG MinTextureHeight;      ///< Minimum allowed height for a texture.
    XULONG MaxTextureWidth;       ///< Maximum allowed width for a texture.
    XULONG MaxTextureHeight;      ///< Maximum allowed height for a texture.
    XULONG MaxClipPlanes;         ///< Maximum number of user-defined clip planes.
    XULONG VertexCaps;            ///< Vertex Processing capabilities (CKRST_VTXCAPS flags).
    XULONG MaxActiveLights;       ///< Maximum number of simultaneous active lights.
    XULONG MaxNumberBlendStage;   ///< Maximum number of blend stages.
    XULONG MaxNumberTextureStage; ///< Maximum number of simultaneous textures (multi-texturing).
    XULONG MaxTextureRatio;       ///< Maximum texture aspect ratio (Width/Height).

    XULONG TextureFilterCaps;        ///< Texture filtering capabilities (CKRST_TFILTERCAPS flags).
    XULONG TextureAddressCaps;       ///< Texture addressing capabilities (CKRST_TADDRESSCAPS flags).
    XULONG TextureCaps;              ///< General texture capabilities (CKRST_TEXTURECAPS flags).
    XULONG MiscCaps;                 ///< Miscellaneous capabilities (CKRST_MISCCAPS flags).
    XULONG AlphaCmpCaps;             ///< Alpha compare function capabilities (CKRST_CMPCAPS flags).
    XULONG ZCmpCaps;                 ///< Z compare function capabilities (CKRST_CMPCAPS flags).
    XULONG RasterCaps;               ///< Rasterization capabilities (CKRST_RASTERCAPS flags).
    XULONG SrcBlendCaps;             ///< Source blend capabilities (CKRST_BLENDCAPS flags).
    XULONG DestBlendCaps;            ///< Destination blend capabilities (CKRST_BLENDCAPS flags).
    XULONG CKRasterizerSpecificCaps; ///< Rasterizer-specific capabilities (CKRST_SPECIFICCAPS flags).
} Vx3DCapsDesc;

/**
 * @brief CKRasterizer specific capabilities.
 *
 * @remarks This enumeration is used to know special implementation
 * details of a given rasterizer.
 * @see VxDriverDesc, Vx3DCapsDesc, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_SPECIFICCAPS {
    CKRST_SPECIFICCAPS_SPRITEASTEXTURES = 0x00000001UL, ///< Sprites are rendered using textured primitives.
    CKRST_SPECIFICCAPS_CLAMPEDGEALPHA   = 0x00000002UL,
    ///< Texture border alpha needs to be zero for clamping to work correctly (e.g., OpenGL).
    CKRST_SPECIFICCAPS_CANDOVERTEXBUFFER  = 0x00000004UL, ///< Vertex buffers can be locked for read/write access.
    CKRST_SPECIFICCAPS_GLATTENUATIONMODEL = 0x00000008UL, ///< Supports GL-style (or DX7) light attenuation model.
    CKRST_SPECIFICCAPS_SOFTWARE           = 0x00000010UL, ///< Software transformations, lighting, and rasterization.
    CKRST_SPECIFICCAPS_HARDWARE           = 0x00000020UL, ///< Hardware rasterization with software T&L.
    CKRST_SPECIFICCAPS_HARDWARETL         = 0x00000040UL, ///< Hardware transform, lighting, and rasterization.
    CKRST_SPECIFICCAPS_COPYTEXTURE        = 0x00000080UL, ///< Can copy from the back buffer into a texture.

    CKRST_SPECIFICCAPS_DX5 = 0x00000100UL, ///< DirectX 5 implementation.
    CKRST_SPECIFICCAPS_DX7 = 0x00000200UL, ///< DirectX 7 implementation.
    CKRST_SPECIFICCAPS_DX8 = 0x00000400UL, ///< DirectX 8.1 implementation.
    CKRST_SPECIFICCAPS_DX9 = 0x00000800UL, ///< DirectX 9 implementation.

    CKRST_SPECIFICCAPS_SUPPORTSHADERS = 0x00001000UL, ///< Supports programmable vertex and pixel shaders.
    CKRST_SPECIFICCAPS_POINTSPRITES   = 0x00002000UL, ///< Supports point sprites.

    CKRST_SPECIFICCAPS_VERTEXCOLORABGR = 0x00004000UL, ///< Vertex colors are expected in ABGR format (OpenGL).
    CKRST_SPECIFICCAPS_BLENDTEXTEFFECT = 0x00008000UL,
    ///< Supports texture combine effects (e.g., GL_TEXTURE_ENV_COMBINE).

    CKRST_SPECIFICCAPS_CANDOINDEXBUFFER = 0x00010000UL, ///< Index buffers can be locked for read/write access.
    CKRST_SPECIFICCAPS_HW_SKINNING      = 0x00020000UL, ///< Supports hardware-accelerated skinning.

    CKRST_SPECIFICCAPS_AUTGENMIPMAP = 0x00040000UL, ///< Supports automatic generation of mipmaps.
} CKRST_SPECIFICCAPS;

/**
 * @brief Texture filtering capabilities.
 *
 * @remarks This enumeration gives the filter modes supported by a render driver.
 * @see VxDriverDesc, Vx3DCapsDesc, CKMaterial::SetTextureMinMode, CKMaterial::SetTextureMagMode, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_TFILTERCAPS {
    CKRST_TFILTERCAPS_NEAREST          = 0x00000001UL, ///< Point sampling supported.
    CKRST_TFILTERCAPS_LINEAR           = 0x00000002UL, ///< Bilinear filtering supported.
    CKRST_TFILTERCAPS_MIPNEAREST       = 0x00000004UL, ///< Mipmapping with point sampling supported.
    CKRST_TFILTERCAPS_MIPLINEAR        = 0x00000008UL, ///< Mipmapping with bilinear filtering supported.
    CKRST_TFILTERCAPS_LINEARMIPNEAREST = 0x00000010UL, ///< Bilinear interpolation between mipmaps with point sampling.
    CKRST_TFILTERCAPS_LINEARMIPLINEAR  = 0x00000020UL, ///< Trilinear filtering supported.
    CKRST_TFILTERCAPS_ANISOTROPIC      = 0x00000400UL  ///< Anisotropic filtering supported.
} CKRST_TFILTERCAPS;

/**
 * @brief Texture addressing capabilities.
 *
 * @remarks This enumeration gives the texture addressing modes supported by a render driver.
 * @see VxDriverDesc, Vx3DCapsDesc, CKMaterial::SetTextureAddressMode, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_TADDRESSCAPS {
    CKRST_TADDRESSCAPS_WRAP          = 0x00000001UL, ///< Texture wrapping supported.
    CKRST_TADDRESSCAPS_MIRROR        = 0x00000002UL, ///< Texture mirroring supported.
    CKRST_TADDRESSCAPS_CLAMP         = 0x00000004UL, ///< Texture clamping supported.
    CKRST_TADDRESSCAPS_BORDER        = 0x00000008UL, ///< Border color addressing supported.
    CKRST_TADDRESSCAPS_INDEPENDENTUV = 0x00000010UL  ///< Address modes can be set independently for U and V.
} CKRST_TADDRESSCAPS;

/**
 * @brief General texture capabilities.
 *
 * @remarks This enumeration gives the texture operations supported by a render driver.
 * @see VxDriverDesc, Vx3DCapsDesc, CKMaterial::PerspectiveCorrectionEnabled, CKMaterial::SetTextureBlendMode, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_TEXTURECAPS {
    CKRST_TEXTURECAPS_PERSPECTIVE = 0x00000001UL, ///< Perspective correction is supported.
    CKRST_TEXTURECAPS_POW2 = 0x00000002UL, ///< Texture dimensions must be powers of 2.
    CKRST_TEXTURECAPS_ALPHA = 0x00000004UL, ///< Supports textures with alpha for decal and modulate blend modes.
    CKRST_TEXTURECAPS_SQUAREONLY = 0x00000020UL, ///< Textures must have equal width and height.
    CKRST_TEXTURECAPS_CONDITIONALNONPOW2 = 0x00000100UL, ///< Supports non-power-of-2 textures under certain conditions.
    CKRST_TEXTURECAPS_PROJECTED = 0x00000400UL, ///< Supports projective texturing (per-pixel divide).
    CKRST_TEXTURECAPS_CUBEMAP = 0x00000800UL, ///< Supports cube maps.
    CKRST_TEXTURECAPS_VOLUMEMAP = 0x00002000UL, ///< Supports volume (3D) maps.
} CKRST_TEXTURECAPS;

/**
 * @brief Stencil buffer operation capabilities.
 *
 * @remarks This enumeration gives the stencil operations supported by a render driver.
 * @see VxDriverDesc, Vx3DCapsDesc, VXSTENCILOP, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_STENCILCAPS {
    CKRST_STENCILCAPS_KEEP    = 0x00000001UL, ///< VXSTENCILOP_KEEP is supported.
    CKRST_STENCILCAPS_ZERO    = 0x00000002UL, ///< VXSTENCILOP_ZERO is supported.
    CKRST_STENCILCAPS_REPLACE = 0x00000004UL, ///< VXSTENCILOP_REPLACE is supported.
    CKRST_STENCILCAPS_INCRSAT = 0x00000008UL, ///< VXSTENCILOP_INCRSAT is supported.
    CKRST_STENCILCAPS_DECRSAT = 0x00000010UL, ///< VXSTENCILOP_DECRSAT is supported.
    CKRST_STENCILCAPS_INVERT  = 0x00000020UL, ///< VXSTENCILOP_INVERT is supported.
    CKRST_STENCILCAPS_INCR    = 0x00000040UL, ///< VXSTENCILOP_INCR is supported.
    CKRST_STENCILCAPS_DECR    = 0x00000080UL  ///< VXSTENCILOP_DECR is supported.
} CKRST_STENCILCAPS;

/**
 * @brief Miscellaneous rendering capabilities.
 * @see VxDriverDesc, Vx3DCapsDesc, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_MISCCAPS {
    CKRST_MISCCAPS_MASKZ      = 0x00000002UL, ///< Driver can mask Z-buffer writes (ZWriteEnable).
    CKRST_MISCCAPS_CONFORMANT = 0x00000008UL, ///< Driver is conformant to the OpenGL standard.
    CKRST_MISCCAPS_CULLNONE   = 0x00000010UL, ///< Supports no culling.
    CKRST_MISCCAPS_CULLCW     = 0x00000020UL, ///< Supports clockwise culling.
    CKRST_MISCCAPS_CULLCCW    = 0x00000040UL  ///< Supports counter-clockwise culling.
} CKRST_MISCCAPS;

/**
 * @brief Rasterization capabilities.
 * @see VxDriverDesc, Vx3DCapsDesc, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_RASTERCAPS {
    CKRST_RASTERCAPS_DITHER         = 0x00000001UL, ///< Driver can perform dithering.
    CKRST_RASTERCAPS_ZTEST          = 0x00000010UL, ///< Z-test operations are supported.
    CKRST_RASTERCAPS_SUBPIXEL       = 0x00000060UL, ///< Subpixel placement of primitives is supported.
    CKRST_RASTERCAPS_FOGVERTEX      = 0x00000080UL, ///< Per-vertex fog is supported.
    CKRST_RASTERCAPS_FOGPIXEL       = 0x00000100UL, ///< Per-pixel fog (table fog) is supported.
    CKRST_RASTERCAPS_ZBIAS          = 0x00004000UL, ///< Z-bias is supported to avoid z-fighting.
    CKRST_RASTERCAPS_ZBUFFERLESSHSR = 0x00008000UL, ///< Hidden surface removal can be performed without a Z-buffer.
    CKRST_RASTERCAPS_FOGRANGE       = 0x00010000UL, ///< Supports range-based fog.
    CKRST_RASTERCAPS_ANISOTROPY     = 0x00020000UL, ///< Anisotropic filtering is supported.
    CKRST_RASTERCAPS_WBUFFER        = 0x00040000UL, ///< Depth buffering can be performed using W-values.
    CKRST_RASTERCAPS_WFOG           = 0x00100000UL, ///< Supports W-based pixel fog.
    CKRST_RASTERCAPS_ZFOG           = 0x00200000UL  ///< Supports Z-based pixel fog.
} CKRST_RASTERCAPS;

/**
 * @brief Supported blending factor capabilities.
 * @remarks This enumeration gives the blending factors (VXBLEND_MODE) supported by a render driver.
 * @see VxDriverDesc, Vx3DCapsDesc, CKRenderManager::GetRenderDriverDescription, VXBLEND_MODE
 */
typedef enum CKRST_BLENDCAPS {
    CKRST_BLENDCAPS_ZERO            = 0x00000001UL, ///< Driver supports VXBLEND_ZERO.
    CKRST_BLENDCAPS_ONE             = 0x00000002UL, ///< Driver supports VXBLEND_ONE.
    CKRST_BLENDCAPS_SRCCOLOR        = 0x00000004UL, ///< Driver supports VXBLEND_SRCCOLOR.
    CKRST_BLENDCAPS_INVSRCCOLOR     = 0x00000008UL, ///< Driver supports VXBLEND_INVSRCCOLOR.
    CKRST_BLENDCAPS_SRCALPHA        = 0x00000010UL, ///< Driver supports VXBLEND_SRCALPHA.
    CKRST_BLENDCAPS_INVSRCALPHA     = 0x00000020UL, ///< Driver supports VXBLEND_INVSRCALPHA.
    CKRST_BLENDCAPS_DESTALPHA       = 0x00000040UL, ///< Driver supports VXBLEND_DESTALPHA.
    CKRST_BLENDCAPS_INVDESTALPHA    = 0x00000080UL, ///< Driver supports VXBLEND_INVDESTALPHA.
    CKRST_BLENDCAPS_DESTCOLOR       = 0x00000100UL, ///< Driver supports VXBLEND_DESTCOLOR.
    CKRST_BLENDCAPS_INVDESTCOLOR    = 0x00000200UL, ///< Driver supports VXBLEND_INVDESTCOLOR.
    CKRST_BLENDCAPS_SRCALPHASAT     = 0x00000400UL, ///< Driver supports VXBLEND_SRCALPHASAT.
    CKRST_BLENDCAPS_BOTHSRCALPHA    = 0x00000800UL, ///< Driver supports VXBLEND_BOTHSRCALPHA.
    CKRST_BLENDCAPS_BOTHINVSRCALPHA = 0x00001000UL  ///< Driver supports VXBLEND_BOTHINVSRCALPHA.
} CKRST_BLENDCAPS;

/**
 * @brief Supported comparison function capabilities.
 * @remarks This enumeration gives the comparison functions (VXCMPFUNC) supported by a render driver.
 * @see VxDriverDesc, Vx3DCapsDesc, CKRenderManager::GetRenderDriverDescription, VXCMPFUNC
 */
typedef enum CKRST_CMPCAPS {
    CKRST_CMPCAPS_NEVER        = 0x00000001UL, ///< Driver supports VXCMP_NEVER.
    CKRST_CMPCAPS_LESS         = 0x00000002UL, ///< Driver supports VXCMP_LESS.
    CKRST_CMPCAPS_EQUAL        = 0x00000004UL, ///< Driver supports VXCMP_EQUAL.
    CKRST_CMPCAPS_LESSEQUAL    = 0x00000008UL, ///< Driver supports VXCMP_LESSEQUAL.
    CKRST_CMPCAPS_GREATER      = 0x00000010UL, ///< Driver supports VXCMP_GREATER.
    CKRST_CMPCAPS_NOTEQUAL     = 0x00000020UL, ///< Driver supports VXCMP_NOTEQUAL.
    CKRST_CMPCAPS_GREATEREQUAL = 0x00000040UL, ///< Driver supports VXCMP_GREATEREQUAL.
    CKRST_CMPCAPS_ALWAYS       = 0x00000080UL  ///< Driver supports VXCMP_ALWAYS.
} CKRST_CMPCAPS;

/**
 * @brief Supported vertex processing capabilities.
 * @see VxDriverDesc, Vx3DCapsDesc, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_VTXCAPS {
    CKRST_VTXCAPS_TEXGEN         = 0x00000001UL, ///< Device can generate texture coordinates automatically.
    CKRST_VTXCAPS_MATERIALSOURCE = 0x00000002UL,
    ///< Device supports selectable vertex color sources (material or vertex).
    CKRST_VTXCAPS_VERTEXFOG         = 0x00000004UL, ///< Device supports vertex fog.
    CKRST_VTXCAPS_DIRECTIONALLIGHTS = 0x00000008UL, ///< Device supports directional lights.
    CKRST_VTXCAPS_POSITIONALLIGHTS  = 0x00000010UL, ///< Device supports positional lights (point and spotlights).
    CKRST_VTXCAPS_LOCALVIEWER       = 0x00000020UL  ///< Device supports local viewer for specular highlights.
} CKRST_VTXCAPS;

/**
 * @brief Supported 2D capabilities.
 * @see VxDriverDesc, Vx2DCapsDesc, CKRenderManager::GetRenderDriverDescription
 */
typedef enum CKRST_2DCAPS {
    CKRST_2DCAPS_WINDOWED = 0x00000001UL, ///< Driver supports windowed rendering.
    CKRST_2DCAPS_3D       = 0x00000002UL, ///< Driver supports 3D acceleration.
    CKRST_2DCAPS_GDI      = 0x00000004UL  ///< Driver is shared with GDI.
} CKRST_2DCAPS;

#endif // VXDEFINES_H
