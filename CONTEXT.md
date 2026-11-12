# NeMo2 engine

NeMo2 implements the Virtools runtime and assembles its modules through CMake.

## Language

**Engine runtime**:
The Virtools libraries, rasterizer, manager plugins, file readers, and building
blocks.

**Runtime component manifest**:
The configured engine module declarations published for an application to
assemble and check its runtime.

**Virtools-Style Rasterizer Interface**:
The private C++ surface around `CKRasterizer`, `CKRasterizerDriver`, and
`CKRasterizerContext`, retaining their original resource and state vocabulary.
_Avoid_: public SDK promise

**Concrete Rasterizer Context**:
The SDL_GPU `CKRasterizerContext` implementation owning native graphics
resources, submission, and presentation.
_Avoid_: backend adapter, provider

**CKFFPLib component**:
The fixed-function state, validation, and draw-data implementation used by the
Concrete Rasterizer Context, without native device ownership or submission.
_Avoid_: second rasterizer interface

**Rasterizer Handle**:
The opaque `CKDWORD` resource identity exposed to callers; zero is invalid.
_Avoid_: raw native handle

**Original Feature Baseline**:
The observable behavior represented by original rasterizer methods, resource
types, state operations, and reported capabilities.

See [rendering implementation guidance](docs/RENDERING.md).
