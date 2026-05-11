// SPDX-License-Identifier: GPL-3.0-only
//
// Path B chunk 2 of the cosmic-comp HDR experiment — per-surface
// decode-to-linear wrapper around `WaylandSurfaceRenderElement`.
//
// When HDR is enabled on an output, every wayland surface contributing pixels
// to the composite gets wrapped in `LinearizedSurfaceRenderElement`. At draw
// time, this element overrides the GlesFrame's default texture shader with the
// linearize program (see `shaders/linearize.frag`), so the surface's pixels are
// decoded from their declared color encoding into a unified linear composite
// space (PQ-aligned, [0, 1] = [0, 10000 cd/m²]) as they're written to the
// offscreen framebuffer.
//
// The offscreen framebuffer must be a floating-point format (Path B chunk 1
// switches it to Abgr16161616f / RGBA16F) to hold linear values without
// clamping at the HDR highlight range.
//
// Companion: the postprocess shader (Path B chunk 3) is updated to take linear
// input — its old sRGB→linear stage is removed; it just applies the output's
// encoding (PQ for HDR).
//
// Structurally mirrors `ClippedSurfaceRenderElement` in `clipped_surface.rs`.

use std::borrow::BorrowMut;
use std::sync::Arc;

use smithay::{
    backend::renderer::{
        ImportAll, ImportMem, Renderer,
        element::{
            Element, Id, Kind, RenderElement, UnderlyingStorage,
            surface::WaylandSurfaceRenderElement,
        },
        gles::{GlesFrame, GlesRenderer, GlesTexProgram, Uniform, UniformValue},
        utils::{CommitCounter, DamageSet, OpaqueRegions},
    },
    reexports::wayland_protocols::wp::color_management::v1::server::wp_color_manager_v1::{
        Primaries as ProtoPrimaries, TransferFunction as ProtoTransferFunction,
    },
    utils::{Buffer, Physical, Rectangle, Scale, Transform, user_data::UserDataMap},
    wayland::color_management::{ImageDescription, PrimariesDef, TransferFunctionDef},
};

use crate::backend::render::element::AsGlowRenderer;

/// GLSL source — `#include_str!`'d into the binary so we don't need a file
/// next to the binary at runtime.
pub static LINEARIZE_SHADER: &str = include_str!("./shaders/linearize.frag");

/// Wrapper around `GlesTexProgram` so it can be stored in the renderer's
/// `UserDataMap` for retrieval at draw time. Mirrors `ClippingShader`.
pub struct LinearizeShader(pub GlesTexProgram);

impl LinearizeShader {
    pub fn get<R: AsGlowRenderer>(renderer: &R) -> GlesTexProgram {
        std::borrow::Borrow::<GlesRenderer>::borrow(renderer.glow_renderer())
            .egl_context()
            .user_data()
            .get::<LinearizeShader>()
            .expect("LinearizeShader not initialized — call init_shaders first")
            .0
            .clone()
    }
}

// ---------------------------------------------------------------------------
// Transfer-function and primaries → shader uniform mapping
// ---------------------------------------------------------------------------

/// `tf_id` values matching the linearize.frag shader. See top of that file for
/// the canonical list. Some values are not yet referenced in cosmic-comp's
/// wrapping code but are listed here so the shader's enum stays
/// human-readable.
#[allow(dead_code)]
mod tf {
    pub const PASSTHROUGH: i32 = 0;
    pub const SRGB: i32 = 1;
    pub const BT1886: i32 = 2;
    pub const GAMMA22: i32 = 3;
    pub const ST2084_PQ: i32 = 4;
    pub const HLG: i32 = 5;
    pub const EXT_LINEAR: i32 = 6;
}

/// Resolve a surface's transfer function (if declared) into the shader's
/// `tf_id` constant. Surfaces without a declared color description default to
/// sRGB — that matches every current SDR client's actual content.
fn tf_id_for_description(description: Option<&ImageDescription>) -> i32 {
    let Some(desc) = description else {
        return tf::SRGB;
    };
    match desc.transfer_function {
        Some(TransferFunctionDef::Named(ProtoTransferFunction::Srgb)) => tf::SRGB,
        Some(TransferFunctionDef::Named(ProtoTransferFunction::Bt1886)) => tf::BT1886,
        Some(TransferFunctionDef::Named(ProtoTransferFunction::Gamma22)) => tf::GAMMA22,
        Some(TransferFunctionDef::Named(ProtoTransferFunction::St2084Pq)) => tf::ST2084_PQ,
        Some(TransferFunctionDef::Named(ProtoTransferFunction::Hlg)) => tf::HLG,
        Some(TransferFunctionDef::Named(ProtoTransferFunction::ExtLinear)) => tf::EXT_LINEAR,
        // For named-but-not-handled TFs and explicit power curves, fall back to
        // sRGB. Real PQ/HLG content always uses the matching named enum, so
        // this branch only fires for esoteric configurations.
        _ => tf::SRGB,
    }
}

/// Build the source-primaries → composite-primaries matrix.
///
/// The composite space is **linear BT.2020** when the output is HDR (matches
/// the output's encoded BT.2020 primaries) and **linear BT.709** when the
/// output is SDR. Surface-side primaries are mapped to that target.
///
/// Matrices are computed via Color Spaces & Transformations math —
/// `M_composite = M_RGB_target_to_XYZ_inv * M_RGB_source_to_XYZ` — encoded
/// here as pre-computed 3×3 constants for the four (source, target) pairs we
/// support. Identity in the common case.
fn primaries_matrix_for_description(
    description: Option<&ImageDescription>,
    composite_is_bt2020: bool,
) -> [f32; 9] {
    let source = match description.and_then(|d| d.primaries) {
        Some(PrimariesDef::Named(p)) => p,
        // Surfaces without declared primaries → assume BT.709/sRGB. That's
        // what every current SDR client actually emits.
        _ => ProtoPrimaries::Srgb,
    };

    // Column-major 3×3 matrices, suitable for GLSL `mat3` uniforms with
    // `transpose: false`. Identity is the no-op default.
    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    // BT.709 → BT.2020 (Rec.2020 spec, table 4). Source RGB into Rec.2020 RGB.
    // Computed from the standard primary chromaticities via D65 white point.
    // Stored column-major.
    const BT709_TO_BT2020: [f32; 9] = [
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362315, 0.895595253,
    ];

    // DCI-P3 (D65) → BT.2020.
    const DCIP3_TO_BT2020: [f32; 9] = [
        0.753833151, 0.045744140, 0.001210226,
        0.198597369, 0.941777862, 0.017601327,
        0.047569481, 0.012477999, 0.981188447,
    ];

    // Display-P3 (D65) → BT.2020. Display-P3 uses same primaries as DCI-P3,
    // different white point and OETF; here we treat it identically to DCI-P3
    // for the matrix (the OETF was handled by the transfer-function decode).
    const DISPLAYP3_TO_BT2020: [f32; 9] = DCIP3_TO_BT2020;

    // BT.2020 → BT.709 (inverse of BT709_TO_BT2020).
    const BT2020_TO_BT709: [f32; 9] = [
        1.660491386, -0.124550414, -0.018150763,
        -0.587641138, 1.132899743, -0.100578992,
        -0.072850248, -0.008349329, 1.118729755,
    ];

    // DCI-P3 → BT.709.
    const DCIP3_TO_BT709: [f32; 9] = [
        1.224940180, -0.042056955, -0.019637555,
        -0.224940180, 1.042056955, -0.078636045,
        0.000000000, 0.000000000, 1.098273600,
    ];

    match (source, composite_is_bt2020) {
        // Source matches composite — identity.
        (ProtoPrimaries::Srgb, false) => IDENTITY, // BT.709 → BT.709
        (ProtoPrimaries::Bt2020, true) => IDENTITY,
        (ProtoPrimaries::DciP3, _) if composite_is_bt2020 => DCIP3_TO_BT2020,
        (ProtoPrimaries::DisplayP3, _) if composite_is_bt2020 => DISPLAYP3_TO_BT2020,
        // Source ≠ composite.
        (ProtoPrimaries::Srgb, true) => BT709_TO_BT2020,
        (ProtoPrimaries::Bt2020, false) => BT2020_TO_BT709,
        (ProtoPrimaries::DciP3, false) => DCIP3_TO_BT709,
        (ProtoPrimaries::DisplayP3, false) => DCIP3_TO_BT709,
        // Anything else: identity. Esoteric primaries (CIE1931 XYZ, NTSC,
        // PAL-M, Adobe RGB, etc.) are not yet handled here — they'd fall
        // through to identity, which means slightly-off colors for those
        // surfaces. Real content using them is rare; we'll add matrices when
        // we hit a real case.
        _ => IDENTITY,
    }
}

// ---------------------------------------------------------------------------
// Element wrapper
// ---------------------------------------------------------------------------

/// Wraps a [`WaylandSurfaceRenderElement`] so its texture render is decoded
/// from the surface's declared color encoding into the compositor's linear
/// composite space.
///
/// Constructed via [`Self::new`] at the point a `WaylandSurfaceRenderElement`
/// is built (typically inside cosmic-comp shell element render paths). The
/// `output_description` parameter is the output's image description for the
/// frame being rendered — used to determine the composite color space (BT.2020
/// linear when the output is HDR, BT.709 linear when SDR).
///
/// All Element trait methods forward to the inner element. Only `draw` is
/// overridden to install the linearize shader override on the GlesFrame for
/// the duration of the inner element's draw.
#[derive(Debug)]
pub struct LinearizedSurfaceRenderElement<R>
where
    R: Renderer + ImportAll + ImportMem,
{
    inner: WaylandSurfaceRenderElement<R>,
    program: GlesTexProgram,
    uniforms: Vec<Uniform<'static>>,
}

impl<R> LinearizedSurfaceRenderElement<R>
where
    R: Renderer + ImportAll + ImportMem,
{
    /// Build a linearized wrapper around `elem`.
    ///
    /// - `surface_description`: the wp_color_management_v1 description the
    ///   client has set on the surface (read via
    ///   `smithay::wayland::color_management::with_surface_image_description`).
    ///   `None` means the client hasn't declared one — defaults to sRGB.
    /// - `output_is_hdr`: whether the output being rendered to has HDR
    ///   enabled. Drives the composite-space choice (BT.2020 vs BT.709) and
    ///   whether `ref_white_scale` applies.
    /// - `ref_white_nits`: the SDR reference-white luminance configured for
    ///   the output (typically 203 cd/m²). SDR surfaces get scaled by
    ///   `ref_white_nits / 10000` so a white sRGB pixel lands at the right
    ///   luminance in the PQ composite scale.
    pub fn new<R2: AsGlowRenderer>(
        renderer: &R2,
        elem: WaylandSurfaceRenderElement<R>,
        surface_description: Option<&Arc<ImageDescription>>,
        output_is_hdr: bool,
        ref_white_nits: u32,
    ) -> Self {
        let desc = surface_description.map(|a| a.as_ref());
        let tf_id = tf_id_for_description(desc);
        let primaries = primaries_matrix_for_description(desc, output_is_hdr);

        // SDR reference white scale. In HDR composite, sRGB white (1.0) should
        // map to ref_white_nits / 10000. In SDR composite, there's no scaling
        // (1.0 = max SDR luminance, whatever the panel produces).
        let ref_white_scale = if output_is_hdr {
            (ref_white_nits as f32) / 10000.0
        } else {
            1.0
        };

        let uniforms = vec![
            Uniform::new("tf_id", tf_id),
            Uniform::new("ref_white_scale", ref_white_scale),
            Uniform::new(
                "primaries_matrix",
                UniformValue::Matrix3x3 {
                    matrices: vec![primaries],
                    transpose: false,
                },
            ),
        ];

        Self {
            inner: elem,
            program: LinearizeShader::get(renderer),
            uniforms,
        }
    }
}

// ---------------------------------------------------------------------------
// Element + RenderElement forwarding
// ---------------------------------------------------------------------------

impl<R> Element for LinearizedSurfaceRenderElement<R>
where
    R: Renderer + ImportAll + ImportMem,
{
    fn id(&self) -> &Id {
        self.inner.id()
    }

    fn current_commit(&self) -> CommitCounter {
        self.inner.current_commit()
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.inner.geometry(scale)
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.inner.src()
    }

    fn transform(&self) -> Transform {
        self.inner.transform()
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        self.inner.damage_since(scale, commit)
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        self.inner.opaque_regions(scale)
    }

    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }

    fn kind(&self) -> Kind {
        self.inner.kind()
    }
}

impl<R> RenderElement<R> for LinearizedSurfaceRenderElement<R>
where
    R: AsGlowRenderer + Renderer + ImportAll + ImportMem,
    R::TextureId: 'static,
{
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        // Install the linearize shader as the override texture program for the
        // duration of the inner element's draw. Identical pattern to
        // `ClippedSurfaceRenderElement::draw`.
        BorrowMut::<GlesFrame>::borrow_mut(<R as AsGlowRenderer>::glow_frame_mut(frame))
            .override_default_tex_program(self.program.clone(), self.uniforms.clone());
        let res = self
            .inner
            .draw(frame, src, dst, damage, opaque_regions, cache);
        BorrowMut::<GlesFrame>::borrow_mut(<R as AsGlowRenderer>::glow_frame_mut(frame))
            .clear_tex_program_override();
        res
    }

    fn underlying_storage(&self, _renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        None
    }
}
