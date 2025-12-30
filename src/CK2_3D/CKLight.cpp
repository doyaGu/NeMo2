#include "RCKLight.h"

#include "VxMath.h"
#include "CKStateChunk.h"
#include "CKFile.h"
#include "CKRasterizer.h"
#include "RCK3dEntity.h"

// Static class ID definition
CK_CLASSID RCKLight::m_ClassID = CKCID_LIGHT;

// Flag definitions from IDA analysis
// m_Flags bit layout:
// - Bit 8 (0x100): Active flag
// - Bit 9 (0x200): Specular flag

/*************************************************
Summary: Constructor for RCKLight.
Purpose: Initializes light with default values.
Remarks:
- Initializes m_Flags to 0x100 (active by default)
- Sets default light type to VX_LIGHTPOINT
- Sets diffuse color to white (1,1,1,1)
- Sets specular/ambient to black
- Sets default attenuation (constant=1.0, linear=0, quadratic=0)
- Sets default range to 5000.0
- Sets falloff shape to 1.0
- Sets default spotlight angles (hotspot=0.69813, falloff=0.78540)
- Sets light power to 1.0

Implementation based on decompilation at 0x1001ac80.
*************************************************/
RCKLight::RCKLight(CKContext *Context, CKSTRING name)
    : RCK3dEntity(Context, name), m_LightData(), m_Flags(0x100), m_LightPower(1.0f) {
    // Initialize light type
    m_LightData.Type = VX_LIGHTPOINT;

    // Initialize colors
    m_LightData.Diffuse = VxColor(1.0f, 1.0f, 1.0f, 1.0f);
    m_LightData.Specular = VxColor(0.0f, 0.0f, 0.0f, 0.0f);
    m_LightData.Ambient = VxColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Initialize attenuation
    m_LightData.Range = 5000.0f;
    m_LightData.Falloff = 1.0f;
    m_LightData.Attenuation0 = 1.0f;
    m_LightData.Attenuation1 = 0.0f;
    m_LightData.Attenuation2 = 0.0f;

    // Initialize spotlight angles (approximately 40 and 45 degrees in radians)
    m_LightData.InnerSpotCone = 0.69813174f;
    m_LightData.OuterSpotCone = 0.78539819f;
}

/*************************************************
Summary: Destructor for RCKLight.
Purpose: Cleans up light resources.
Remarks:
- Base class destructor handles entity cleanup

Implementation based on decompilation at 0x1001adf9.
*************************************************/
RCKLight::~RCKLight() {
    // Base class destructor handles cleanup
}

//=============================================================================
// Color Methods
//=============================================================================

/*************************************************
Summary: Sets the light diffuse color.
Purpose: Sets the diffuse color component of the light.
Remarks:
- Directly copies the color to m_LightData.Diffuse

Implementation based on decompilation at 0x1001ae15.
*************************************************/
void RCKLight::SetColor(const VxColor &c) {
    m_LightData.Diffuse = c;
}

/*************************************************
Summary: Returns the light diffuse color.
Purpose: Gets a reference to the diffuse color component.
Return Value: Reference to the diffuse color.

Implementation based on decompilation at 0x1001af2f.
*************************************************/
const VxColor &RCKLight::GetColor() {
    return m_LightData.Diffuse;
}

//=============================================================================
// Attenuation Methods
//=============================================================================

/*************************************************
Summary: Sets the constant attenuation factor.
Purpose: Sets Attenuation0 in the light data.
Remarks:
- Attenuation0 is the constant term in the attenuation formula

Implementation based on decompilation at 0x1001ae44.
*************************************************/
void RCKLight::SetConstantAttenuation(float Value) {
    m_LightData.Attenuation0 = Value;
}

/*************************************************
Summary: Sets the linear attenuation factor.
Purpose: Sets Attenuation1 in the light data.
Remarks:
- Attenuation1 is the linear term in the attenuation formula

Implementation based on decompilation at 0x1001ae5d.
*************************************************/
void RCKLight::SetLinearAttenuation(float Value) {
    m_LightData.Attenuation1 = Value;
}

/*************************************************
Summary: Sets the quadratic attenuation factor.
Purpose: Sets Attenuation2 in the light data.
Remarks:
- Attenuation2 is the quadratic term in the attenuation formula

Implementation based on decompilation at 0x1001ae76.
*************************************************/
void RCKLight::SetQuadraticAttenuation(float Value) {
    m_LightData.Attenuation2 = Value;
}

/*************************************************
Summary: Gets the constant attenuation factor.
Return Value: The constant attenuation factor.

Implementation based on decompilation at 0x1001aef3.
*************************************************/
float RCKLight::GetConstantAttenuation() {
    return m_LightData.Attenuation0;
}

/*************************************************
Summary: Gets the linear attenuation factor.
Return Value: The linear attenuation factor.

Implementation based on decompilation at 0x1001af07.
*************************************************/
float RCKLight::GetLinearAttenuation() {
    return m_LightData.Attenuation1;
}

/*************************************************
Summary: Gets the quadratic attenuation factor.
Return Value: The quadratic attenuation factor.

Implementation based on decompilation at 0x1001af1b.
*************************************************/
float RCKLight::GetQuadraticAttenuation() {
    return m_LightData.Attenuation2;
}

//=============================================================================
// Type Methods
//=============================================================================

/*************************************************
Summary: Gets the light type.
Return Value: The light type (point, spot, or directional).

Implementation based on decompilation at 0x1001af42.
*************************************************/
VXLIGHT_TYPE RCKLight::GetType() {
    return m_LightData.Type;
}

/*************************************************
Summary: Sets the light type.
Purpose: Sets the type to point, spot, or directional.

Implementation based on decompilation at 0x1001ae8f.
*************************************************/
void RCKLight::SetType(VXLIGHT_TYPE Type) {
    m_LightData.Type = Type;
}

//=============================================================================
// Range Methods
//=============================================================================

/*************************************************
Summary: Gets the light range.
Return Value: The distance beyond which the light has no effect.

Implementation based on decompilation at 0x1001af56.
*************************************************/
float RCKLight::GetRange() {
    return m_LightData.Range;
}

/*************************************************
Summary: Sets the light range.
Purpose: Sets the distance beyond which the light has no effect.

Implementation based on decompilation at 0x1001af6a.
*************************************************/
void RCKLight::SetRange(float Value) {
    m_LightData.Range = Value;
}

//=============================================================================
// Spotlight Methods
//=============================================================================

/*************************************************
Summary: Gets the inner cone angle for spotlights.
Return Value: The inner cone angle in radians.

Implementation based on decompilation at 0x1001af83.
*************************************************/
float RCKLight::GetHotSpot() {
    return m_LightData.InnerSpotCone;
}

/*************************************************
Summary: Gets the outer cone angle for spotlights.
Return Value: The outer cone angle in radians.

Implementation based on decompilation at 0x1001af97.
*************************************************/
float RCKLight::GetFallOff() {
    return m_LightData.OuterSpotCone;
}

/*************************************************
Summary: Sets the inner cone angle for spotlights.
Purpose: Sets the angle of the inner cone in radians.

Implementation based on decompilation at 0x1001aea8.
*************************************************/
void RCKLight::SetHotSpot(float Value) {
    m_LightData.InnerSpotCone = Value;
}

/*************************************************
Summary: Sets the outer cone angle for spotlights.
Purpose: Sets the angle of the outer cone in radians.

Implementation based on decompilation at 0x1001aec1.
*************************************************/
void RCKLight::SetFallOff(float Value) {
    m_LightData.OuterSpotCone = Value;
}

/*************************************************
Summary: Gets the falloff shape between inner and outer cone.
Return Value: The interpolation factor between cones.

Implementation based on decompilation at 0x1001afab.
*************************************************/
float RCKLight::GetFallOffShape() {
    return m_LightData.Falloff;
}

/*************************************************
Summary: Sets the falloff shape between inner and outer cone.
Purpose: Sets the interpolation factor between cones.

Implementation based on decompilation at 0x1001aeda.
*************************************************/
void RCKLight::SetFallOffShape(float Value) {
    m_LightData.Falloff = Value;
}

//=============================================================================
// Activity Methods
//=============================================================================

/*************************************************
Summary: Activates or deactivates the light.
Purpose: Sets the active bit in m_Flags.
Remarks:
- Uses bit 8 (0x100) for the active flag
- Active is TRUE to enable the light, FALSE to disable

Implementation based on decompilation at 0x1001afd9.
*************************************************/
void RCKLight::Active(CKBOOL Active) {
    if (Active)
        m_Flags |= 0x100;
    else
        m_Flags &= ~0x100;
}

/*************************************************
Summary: Gets the light activity status.
Return Value: TRUE if the light is active, FALSE otherwise.

Implementation based on decompilation at 0x1001b018.
*************************************************/
CKBOOL RCKLight::GetActivity() {
    return (m_Flags & 0x100) != 0;
}

/*************************************************
Summary: Enables or disables specular highlights.
Purpose: Sets the specular bit in m_Flags.
Remarks:
- Uses bit 9 (0x200) for the specular flag

Implementation based on decompilation at 0x1001b037.
*************************************************/
void RCKLight::SetSpecularFlag(CKBOOL Specular) {
    if (Specular)
        m_Flags |= 0x200;
    else
        m_Flags &= ~0x200;
}

/*************************************************
Summary: Gets the specular highlight status.
Return Value: TRUE if specular highlights are enabled, FALSE otherwise.

Implementation based on decompilation at 0x1001b076.
*************************************************/
CKBOOL RCKLight::GetSpecularFlag() {
    return (m_Flags & 0x200) != 0;
}

//=============================================================================
// Target Methods (for CKLight base class - no target support)
//=============================================================================

/*************************************************
Summary: Gets the target of the light.
Return Value: Always NULL for CKLight (targets only supported in CKTargetLight).
*************************************************/
CK3dEntity *RCKLight::GetTarget() {
    return nullptr;
}

/*************************************************
Summary: Sets the target of the light.
Remarks: Does nothing for CKLight (targets only supported in CKTargetLight).
*************************************************/
void RCKLight::SetTarget(CK3dEntity *target) {
    // CKLight does not support targets - this is for CKTargetLight
}

//=============================================================================
// Light Power Methods
//=============================================================================

/*************************************************
Summary: Gets the light power multiplier.
Return Value: The light power factor.

Implementation based on decompilation at 0x1001b095.
*************************************************/
float RCKLight::GetLightPower() {
    return m_LightPower;
}

/*************************************************
Summary: Sets the light power multiplier.
Purpose: Sets a multiplier for the light color during rendering.
Remarks:
- Default value is 1.0
- Can be any value including negative for special effects

Implementation based on decompilation at 0x1001b0a9.
*************************************************/
void RCKLight::SetLightPower(float power) {
    m_LightPower = power;
}

/*************************************************
Summary: Save method for RCKLight.
Purpose: Saves light data to a state chunk including light type, color, attenuation, and spot light properties.
Remarks:
- Calls base class RCK3dEntity::Save() first to handle entity data
- Creates light-specific state chunk with identifier 0x400000 for main light data
- Saves light type combined with flags for efficient storage
- Converts diffuse color to packed DWORD format
- Saves attenuation coefficients and range
- For spot lights, saves cone angles and falloff
- Optionally saves light power if different from default (1.0)

Implementation based on decompilation at 0x1001B389:
- Uses chunk identifier 0x400000 for main light data
- Uses chunk identifier 0x800000 for optional light power
- Combines light type and flags in single DWORD
- Converts VxColor to packed DWORD using helper function
- Handles spot light-specific properties only for VX_LIGHTSPOT type

Arguments:
- file: The file context for saving (may be NULL for memory operations)
- flags: Save flags controlling behavior and data inclusion
Return Value:
- CKStateChunk*: The created state chunk containing light data
*************************************************/
CKStateChunk *RCKLight::Save(CKFile *file, CKDWORD flags) {
    // Call base class save first to handle entity data
    CKStateChunk *baseChunk = RCK3dEntity::Save(file, flags);

    // Return early if no file context and not in specific save modes
    if (!file && !(flags & CK_STATESAVE_LIGHTONLY)) {
        return baseChunk;
    }

    // Create light-specific state chunk
    CKStateChunk *lightChunk = CreateCKStateChunk(CKCID_LIGHT, file);
    if (!lightChunk) {
        return baseChunk;
    }

    lightChunk->StartWrite();
    lightChunk->AddChunkAndDelete(baseChunk);

    // Write main light data with identifier 0x400000
    lightChunk->WriteIdentifier(CK_STATESAVE_LIGHTDATA);

    // Combine light type and flags for efficient storage
    CKDWORD typeAndFlags = m_LightData.Type | m_Flags;
    lightChunk->WriteDword(typeAndFlags);

    // Convert diffuse color to packed DWORD format (ARGB with alpha forced to 0xFF)
    // IDA: sub_1001BA90 returns RGBAFTOCOLOR(this) | 0xFF000000
    // RGBAFTOCOLOR uses: (A << 24) | (R << 16) | (G << 8) | (B << 0)
    lightChunk->WriteDword(m_LightData.Diffuse.GetRGB());

    // Write attenuation and range
    lightChunk->WriteFloat(m_LightData.Attenuation0);
    lightChunk->WriteFloat(m_LightData.Attenuation1);
    lightChunk->WriteFloat(m_LightData.Attenuation2);
    lightChunk->WriteFloat(m_LightData.Range);

    // For spot lights, save additional properties
    if (m_LightData.Type == VX_LIGHTSPOT) {
        lightChunk->WriteFloat(m_LightData.OuterSpotCone);
        lightChunk->WriteFloat(m_LightData.InnerSpotCone);
        lightChunk->WriteFloat(m_LightData.Falloff);
    }

    // Save light power if different from default
    if (m_LightPower != 1.0f) {
        lightChunk->WriteIdentifier(CK_STATESAVE_LIGHTDATA2);
        lightChunk->WriteFloat(m_LightPower);
    }

    // Close chunk appropriately based on class type
    if (GetClassID() == CKCID_LIGHT) {
        lightChunk->CloseChunk();
    } else {
        lightChunk->UpdateDataSize();
    }

    return lightChunk;
}

/*************************************************
Summary: Load method for RCKLight.
Purpose: Loads light data from a state chunk including light properties, color, and attenuation settings.
Remarks:
- Calls base class RCK3dEntity::Load() first to handle entity data
- Supports both legacy format (data version < 5) and current format
- Legacy format reads individual properties with separate identifiers
- Current format uses identifier 0x400000 with packed data
- Unpacks light type and flags from combined DWORD
- Converts packed color DWORD back to VxColor
- Handles spot light properties only for VX_LIGHTSPOT type
- Validates light type and sets default if invalid

Implementation based on decompilation at 0x1001B50E:
- Legacy format: single chunk 0x400000 with all light properties
- Current format: main chunk 0x400000 and optional power chunk 0x800000
- Unpacks type/flags and color from packed DWORDs
- Sets default light power to 1.0 if not specified
- Validates light type range and defaults to VX_LIGHTPOINT if invalid

Arguments:
- chunk: The state chunk containing light data
- file: The file context for loading (may be NULL)
Return Value:
- CKERROR: 0 for success, -1 for invalid chunk
*************************************************/
CKERROR RCKLight::Load(CKStateChunk *chunk, CKFile *file) {
    if (!chunk) {
        return CKERR_INVALIDPARAMETER;
    }

    // Call base class load first to handle entity data
    RCK3dEntity::Load(chunk, file);

    // Handle legacy format (data version < 5)
    if (chunk->GetDataVersion() < 5) {
        if (chunk->SeekIdentifier(CK_STATESAVE_LIGHTDATA)) {
            m_LightData.Type = (VXLIGHT_TYPE) chunk->ReadDword();

            // Read diffuse color components
            m_LightData.Diffuse.r = chunk->ReadFloat();
            m_LightData.Diffuse.g = chunk->ReadFloat();
            m_LightData.Diffuse.b = chunk->ReadFloat();
            chunk->ReadFloat(); // Skip alpha component

            // Read activation and specular flags
            CKBOOL active = chunk->ReadInt();
            Active(active);

            CKBOOL specular = chunk->ReadInt();
            SetSpecularFlag(specular);

            // Read attenuation and range
            m_LightData.Attenuation0 = chunk->ReadFloat();
            m_LightData.Attenuation1 = chunk->ReadFloat();
            m_LightData.Attenuation2 = chunk->ReadFloat();
            m_LightData.Range = chunk->ReadFloat();

            // Read spot light properties
            m_LightData.OuterSpotCone = chunk->ReadFloat();
            m_LightData.InnerSpotCone = chunk->ReadFloat();
            m_LightData.Falloff = chunk->ReadFloat();

            // Set default light power
            m_LightPower = 1.0f;
        }
    }
    // Current format (data version >= 5)
    else {
        if (chunk->SeekIdentifier(CK_STATESAVE_LIGHTDATA)) {
            // Unpack light type and flags from combined DWORD
            CKDWORD typeAndFlags = chunk->ReadDword();
            m_LightData.Type = (VXLIGHT_TYPE) (typeAndFlags & 0xFF);
            m_Flags = typeAndFlags & ~0xFF;

            // Convert packed color DWORD back to VxColor
            // IDA: sub_10016990 unpacks ARGB format:
            // r = BYTE2(color) / 255 (bits 16-23), g = BYTE1(color) / 255 (bits 8-15)
            // b = BYTE0(color) / 255 (bits 0-7), a = HIBYTE(color) / 255 (bits 24-31)
            CKDWORD packedColor = chunk->ReadDword();
            m_LightData.Diffuse.Set(packedColor);

            // Read attenuation and range
            m_LightData.Attenuation0 = chunk->ReadFloat();
            m_LightData.Attenuation1 = chunk->ReadFloat();
            m_LightData.Attenuation2 = chunk->ReadFloat();
            m_LightData.Range = chunk->ReadFloat();

            // For spot lights, read additional properties
            if (m_LightData.Type == VX_LIGHTSPOT) {
                m_LightData.OuterSpotCone = chunk->ReadFloat();
                m_LightData.InnerSpotCone = chunk->ReadFloat();
                m_LightData.Falloff = chunk->ReadFloat();
            }
        }

        // Read light power if present
        if (chunk->SeekIdentifier(CK_STATESAVE_LIGHTDATA2)) {
            m_LightPower = chunk->ReadFloat();
        } else {
            m_LightPower = 1.0f;
        }
    }

    // Validate light type and set default if invalid
    if (m_LightData.Type < VX_LIGHTPOINT || m_LightData.Type > VX_LIGHTDIREC) {
        m_LightData.Type = VX_LIGHTPOINT;
    }

    return CK_OK;
}

//=============================================================================
// Setup Method
//=============================================================================

/*************************************************
Summary: Sets up the light in the rasterizer.
Purpose: Configures the light at the specified index in the rasterizer.
Remarks:
- Checks visibility first
- For non-directional lights, checks if attenuation sum is sufficient
- Checks if light is active (bit 8 / 0x100 in m_Flags)
- Extracts position from world matrix row 3
- Extracts direction from world matrix row 2
- Handles specular flag (0x200) - scales diffuse by light power
- Applies light power scaling to diffuse color if != 1.0

Implementation based on decompilation at 0x1001b0c2.
*************************************************/
CKBOOL RCKLight::Setup(CKRasterizerContext *rst, CKDWORD lightIndex) {
    // Check visibility
    if (!IsVisible())
        return FALSE;

    // For non-directional lights, check attenuation sum
    if (m_LightData.Type != VX_LIGHTDIREC) {
        float attenuationSum = m_LightData.Attenuation0 + m_LightData.Attenuation1 + m_LightData.Attenuation2;
        if (attenuationSum < 0.00001f)
            return FALSE;
    }

    // Check if light is active (0x100 flag)
    if (!(m_Flags & 0x100))
        return FALSE;

    // Extract position from world matrix row 3
    const VxMatrix &worldMat = GetWorldMatrix();
    m_LightData.Position.x = worldMat[3][0];
    m_LightData.Position.y = worldMat[3][1];
    m_LightData.Position.z = worldMat[3][2];

    // Extract direction from world matrix row 2
    m_LightData.Direction.x = worldMat[2][0];
    m_LightData.Direction.y = worldMat[2][1];
    m_LightData.Direction.z = worldMat[2][2];

    // Handle specular flag (0x200) - set specular to scaled diffuse or black
    // IDA: sub_1001BA40 scales r,g,b by power; sub_1001B9B0 creates (val,val,val,1.0)
    if (m_Flags & 0x200) {
        // Scale diffuse color RGB by light power for specular, alpha = 1.0
        m_LightData.Specular.r = m_LightData.Diffuse.r * m_LightPower;
        m_LightData.Specular.g = m_LightData.Diffuse.g * m_LightPower;
        m_LightData.Specular.b = m_LightData.Diffuse.b * m_LightPower;
        m_LightData.Specular.a = 1.0f;
    } else {
        // Set specular to black with alpha = 1.0
        m_LightData.Specular.r = 0.0f;
        m_LightData.Specular.g = 0.0f;
        m_LightData.Specular.b = 0.0f;
        m_LightData.Specular.a = 1.0f;
    }

    // Apply light power scaling
    if (m_LightPower == 1.0f) {
        // No scaling needed
        rst->SetLight(lightIndex, &m_LightData);
        rst->EnableLight(lightIndex, TRUE);
    } else {
        // Save original diffuse, scale it, set light, restore
        VxColor originalDiffuse = m_LightData.Diffuse;
        m_LightData.Diffuse.r *= m_LightPower;
        m_LightData.Diffuse.g *= m_LightPower;
        m_LightData.Diffuse.b *= m_LightPower;
        m_LightData.Diffuse.a *= m_LightPower;

        rst->SetLight(lightIndex, &m_LightData);
        rst->EnableLight(lightIndex, TRUE);

        // Restore original diffuse
        m_LightData.Diffuse = originalDiffuse;
    }

    return TRUE;
}

//=============================================================================
// CKObject Overrides
//=============================================================================

/*************************************************
Summary: Returns the class ID of this object.
Return Value: CKCID_LIGHT class identifier.

Implementation based on decompilation at 0x1001b35b.
*************************************************/
CK_CLASSID RCKLight::GetClassID() {
    return m_ClassID;
}

/*************************************************
Summary: Returns the memory footprint of this object.
Return Value: Memory size in bytes.
Remarks:
- Adds 112 bytes for light-specific data to base class size
- 112 bytes = sizeof(CKLightData) + sizeof(m_Flags) + sizeof(m_LightPower) + padding

Implementation based on decompilation at 0x1001b36b.
*************************************************/
int RCKLight::GetMemoryOccupation() {
    return RCK3dEntity::GetMemoryOccupation() + (sizeof(RCKLight) - sizeof(RCK3dEntity));
}

/*************************************************
Summary: Copies light data from another light object.
Purpose: Deep copy of all light properties including CKLightData.
Remarks:
- Calls base class Copy first
- Copies entire light data block using memcpy (0x70 = 112 bytes)
- This includes CKLightData (104 bytes), m_Flags (4 bytes), m_LightPower (4 bytes)

Implementation based on decompilation at 0x1001b873.
*************************************************/
CKERROR RCKLight::Copy(CKObject &o, CKDependenciesContext &context) {
    CKERROR err = RCK3dEntity::Copy(o, context);
    if (err != CK_OK)
        return err;

    RCKLight *srcLight = static_cast<RCKLight *>(&o);

    // Copy entire light data block (0x70 = 112 bytes)
    // This copies m_LightData (104), m_Flags (4), and m_LightPower (4)
    // Must match original qmemcpy(&this->m_LightData, &a2->m_LightData, 0x70u)
    memcpy(&m_LightData, &srcLight->m_LightData, sizeof(CKLightData));

    return CK_OK;
}

//=============================================================================
// Static Class Methods (for class registration)
// Based on IDA Pro analysis of original CK2_3D.dll
//=============================================================================

CKSTRING RCKLight::GetClassName() {
    return (CKSTRING) "Light";
}

int RCKLight::GetDependenciesCount(int mode) {
    return 0;
}

CKSTRING RCKLight::GetDependencies(int i, int mode) {
    return nullptr;
}

void RCKLight::Register() {
    // Register associated parameter GUID
    CKPARAMETERFROMCLASS(RCKLight, CKPGUID_LIGHT);
}

CKLight *RCKLight::CreateInstance(CKContext *Context) {
    RCKLight *light = new RCKLight(Context, nullptr);
    return reinterpret_cast<CKLight *>(light);
}
