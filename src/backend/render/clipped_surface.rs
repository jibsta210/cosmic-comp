// Taken and modified from niri, licensed GPL-3.
//
// Extended for Path B of the cosmic-comp HDR experiment to subsume the
// per-surface linearize stage (inverse-EOTF + primaries matrix + SDR ref_white
// scaling). The shader (clipped_surface.frag) gates this behind `tf_id != 0`
// so existing clipping-only callers stay passthrough.

use std::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use cgmath::{Matrix3, Vector2};
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform};
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
    utils::user_data::UserDataMap,
    wayland::color_management::{ImageDescription, PrimariesDef, TransferFunctionDef},
};

use crate::backend::render::element::AsGlowRenderer;

pub static CLIPPING_SHADER: &str = include_str!("./shaders/clipped_surface.frag");
pub struct ClippingShader(pub GlesTexProgram);

// ---------------------------------------------------------------------------
// Path B — ColorTransform (linearize params)
// ---------------------------------------------------------------------------

/// Per-surface color-transform inputs for the linearize stage in
/// `clipped_surface.frag`. The shader's `tf_id != 0` branch applies an
/// inverse-EOTF + primaries matrix + SDR ref-white scaling, mapping the
/// surface's pixels into the compositor's linear composite color space
/// (PQ-aligned in HDR mode).
///
/// Use [`ColorTransform::passthrough`] for existing clipping-only callers —
/// the shader skips the linearize block entirely. Use
/// [`ColorTransform::for_surface`] to derive the params from a surface's
/// wp_color_management_v1 description and the output's HDR state.
#[derive(Debug, Clone, Copy)]
pub struct ColorTransform {
    pub tf_id: i32,
    pub ref_white_scale: f32,
    pub primaries_matrix: [f32; 9],
}

/// tf_id constants matching clipped_surface.frag's enum.
#[allow(dead_code)]
pub mod tf {
    pub const PASSTHROUGH: i32 = 0;
    pub const SRGB: i32 = 1;
    pub const BT1886: i32 = 2;
    pub const GAMMA22: i32 = 3;
    pub const ST2084_PQ: i32 = 4;
    pub const HLG: i32 = 5;
    pub const EXT_LINEAR: i32 = 6;
}

// ---------------------------------------------------------------------------
// Path B — per-render-frame HDR context (thread-local)
// ---------------------------------------------------------------------------

use std::cell::Cell;
use std::sync::OnceLock;

/// Read once at process startup: `COSMIC_HDR_PATH_B=1` enables Path B (scene-
/// linear compositing + per-surface linearize + linear-input postprocess
/// shader). Default off, falls back to Phase 2A.2 hardware-CRTC behavior.
///
/// Opt-in so we can A/B test Path B against the existing hardware path without
/// committing the system to one or the other.
pub fn path_b_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("COSMIC_HDR_PATH_B")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
    })
}

thread_local! {
    /// `Some((true, ref_white_nits))` when the current rendering thread is in
    /// the middle of a render frame on an HDR-enabled output, `None`
    /// otherwise. Set by surface/mod.rs::render_frame at the start of each
    /// frame and cleared at the end. Read by `ClippedSurfaceRenderElement::new`
    /// to decide whether to apply linearize.
    ///
    /// Using thread-local state here so render-element construction sites
    /// scattered throughout cosmic-comp's shell module don't have to thread
    /// HDR config through every function signature. Each output's render
    /// frame runs in its own thread (per cosmic-comp's surface render thread
    /// model), so the thread-local maps cleanly to per-output state.
    static RENDER_OUTPUT_HDR_CONTEXT: Cell<Option<(bool, u32)>> = const { Cell::new(None) };
}

/// Mark the start of an HDR render frame. Pair with [`clear_render_hdr_context`]
/// at the end. Called from surface/mod.rs::render_frame.
pub fn set_render_hdr_context(enabled: bool, ref_white_nits: u32) {
    RENDER_OUTPUT_HDR_CONTEXT.with(|c| c.set(Some((enabled, ref_white_nits))));
}

/// Clear the HDR context — render frame done. Called from
/// surface/mod.rs::render_frame.
pub fn clear_render_hdr_context() {
    RENDER_OUTPUT_HDR_CONTEXT.with(|c| c.set(None));
}

/// True if the current render frame is on an HDR-enabled output. Used by
/// shell element render paths to decide whether to wrap non-clipped surfaces
/// in `ClippedSurfaceRenderElement` (with no-op corner_radius) for the
/// linearize stage.
pub fn render_hdr_active() -> bool {
    current_hdr_context().is_some_and(|(enabled, _)| enabled)
}

/// Convert an sRGB-encoded RGB color into the Path B linear composite space.
///
/// When Path B is active in an HDR frame, non-surface shader elements
/// (IndicatorShader, BackdropShader, etc.) write their colors directly to the
/// linear RGBA16F offscreen. If those colors stay sRGB-encoded, the postprocess
/// shader (which skips its sRGB-decode step in Path B mode) interprets them as
/// linear and the panel sees catastrophically wrong values (e.g. 0.5 sRGB
/// gray becoming 5,000 cd/m² HDR).
///
/// Call this helper on every color before passing as a shader uniform so the
/// values match the surface-side linearize output. Outside Path B / HDR, it's
/// a passthrough.
pub fn linearize_srgb_color_for_path_b(color: [f32; 3]) -> [f32; 3] {
    if !path_b_enabled() {
        return color;
    }
    let Some((true, ref_white_nits)) = current_hdr_context() else {
        return color;
    };

    // sRGB inverse EOTF (piecewise: linear segment + gamma 2.4 curve). Same
    // math as clipped_surface.frag's `decode_srgb`.
    let srgb_decode = |c: f32| -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let lin = [srgb_decode(color[0]), srgb_decode(color[1]), srgb_decode(color[2])];

    // BT.709 → BT.2020 (BT.2087 Annex 1). Same matrix as in offscreen.frag
    // M709to2020 and in clipped_surface.rs's BT709_TO_BT2020 (transposed for
    // row-major matrix * column-vector application here).
    let bt2020 = [
        0.6274 * lin[0] + 0.3293 * lin[1] + 0.0433 * lin[2],
        0.0691 * lin[0] + 0.9195 * lin[1] + 0.0114 * lin[2],
        0.0164 * lin[0] + 0.0880 * lin[1] + 0.8956 * lin[2],
    ];

    let ref_white_scale = (ref_white_nits as f32) / 10000.0;
    [
        bt2020[0] * ref_white_scale,
        bt2020[1] * ref_white_scale,
        bt2020[2] * ref_white_scale,
    ]
}

/// Read the current HDR render context. Returns `None` if not in an HDR frame
/// (or if no context has been set — i.e. existing non-HDR render paths).
fn current_hdr_context() -> Option<(bool, u32)> {
    RENDER_OUTPUT_HDR_CONTEXT.with(|c| c.get())
}

impl ColorTransform {
    /// Pick the appropriate color transform for the current render frame.
    ///
    /// HDR frame + Path B enabled → real linearize transform (sRGB→linear→
    /// BT.709→BT.2020 matrix→ref_white scale). Default surface description
    /// is sRGB; future enhancement reads the surface's actual
    /// `wp_color_management_v1` description via `with_surface_image_description`
    /// (would need the WlSurface handle threaded to here, which requires
    /// modifying WaylandSurfaceRenderElement upstream).
    ///
    /// Otherwise → passthrough (existing pre-Path-B behavior preserved).
    pub fn for_current_frame() -> Self {
        if !path_b_enabled() {
            return Self::passthrough();
        }
        match current_hdr_context() {
            Some((true, ref_white_nits)) => Self::for_surface(None, true, ref_white_nits),
            _ => Self::passthrough(),
        }
    }

    /// Identity transform — shader skips the linearize block, behavior matches
    /// niri's original clipping-only path.
    pub fn passthrough() -> Self {
        Self {
            tf_id: tf::PASSTHROUGH,
            ref_white_scale: 1.0,
            // Column-major identity 3×3.
            primaries_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Build a transform appropriate for the given surface + output. None for
    /// `surface_description` means the client never declared one — defaults
    /// to sRGB / BT.709, which matches every current SDR client's content.
    pub fn for_surface(
        surface_description: Option<&Arc<ImageDescription>>,
        output_is_hdr: bool,
        ref_white_nits: u32,
    ) -> Self {
        let desc = surface_description.map(|a| a.as_ref());
        let tf_id = tf_id_for_description(desc);
        let primaries_matrix = primaries_matrix_for_description(desc, output_is_hdr);
        let ref_white_scale = if output_is_hdr {
            (ref_white_nits as f32) / 10000.0
        } else {
            1.0
        };
        Self {
            tf_id,
            ref_white_scale,
            primaries_matrix,
        }
    }
}

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
        _ => tf::SRGB,
    }
}

fn primaries_matrix_for_description(
    description: Option<&ImageDescription>,
    composite_is_bt2020: bool,
) -> [f32; 9] {
    let source = match description.and_then(|d| d.primaries) {
        Some(PrimariesDef::Named(p)) => p,
        _ => ProtoPrimaries::Srgb,
    };

    // Column-major 3×3 for GLSL mat3 uniforms with `transpose: false`.
    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    const BT709_TO_BT2020: [f32; 9] = [
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362315, 0.895595253,
    ];

    const DCIP3_TO_BT2020: [f32; 9] = [
        0.753833151, 0.045744140, 0.001210226,
        0.198597369, 0.941777862, 0.017601327,
        0.047569481, 0.012477999, 0.981188447,
    ];

    const DISPLAYP3_TO_BT2020: [f32; 9] = DCIP3_TO_BT2020;

    const BT2020_TO_BT709: [f32; 9] = [
        1.660491386, -0.124550414, -0.018150763,
        -0.587641138, 1.132899743, -0.100578992,
        -0.072850248, -0.008349329, 1.118729755,
    ];

    const DCIP3_TO_BT709: [f32; 9] = [
        1.224940180, -0.042056955, -0.019637555,
        -0.224940180, 1.042056955, -0.078636045,
        0.000000000, 0.000000000, 1.098273600,
    ];

    match (source, composite_is_bt2020) {
        (ProtoPrimaries::Srgb, false) => IDENTITY,
        (ProtoPrimaries::Bt2020, true) => IDENTITY,
        (ProtoPrimaries::DciP3, _) if composite_is_bt2020 => DCIP3_TO_BT2020,
        (ProtoPrimaries::DisplayP3, _) if composite_is_bt2020 => DISPLAYP3_TO_BT2020,
        (ProtoPrimaries::Srgb, true) => BT709_TO_BT2020,
        (ProtoPrimaries::Bt2020, false) => BT2020_TO_BT709,
        (ProtoPrimaries::DciP3, false) => DCIP3_TO_BT709,
        (ProtoPrimaries::DisplayP3, false) => DCIP3_TO_BT709,
        _ => IDENTITY,
    }
}

impl ClippingShader {
    pub fn get<R: AsGlowRenderer>(renderer: &R) -> GlesTexProgram {
        Borrow::<GlesRenderer>::borrow(renderer.glow_renderer())
            .egl_context()
            .user_data()
            .get::<ClippingShader>()
            .expect("Custom Shaders not initialized")
            .0
            .clone()
    }
}

#[derive(Debug)]
pub struct ClippedSurfaceRenderElement<R>
where
    R: Renderer + ImportAll + ImportMem,
{
    inner: WaylandSurfaceRenderElement<R>,
    program: GlesTexProgram,
    radius: [u8; 4],
    geometry: Rectangle<f64, Logical>,
    uniforms: Vec<Uniform<'static>>,
}

impl<R> ClippedSurfaceRenderElement<R>
where
    R: Renderer + ImportAll + ImportMem,
{
    pub fn new(
        renderer: &mut R,
        elem: WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
        geometry: Rectangle<f64, Logical>,
        radius: [u8; 4],
    ) -> Self
    where
        R: AsGlowRenderer,
    {
        // Path B — automatically pick up the HDR linearize transform from the
        // current render frame's thread-local context. Outside an HDR frame
        // this returns passthrough, matching pre-HDR behavior exactly.
        Self::new_with_color(
            renderer,
            elem,
            scale,
            geometry,
            radius,
            ColorTransform::for_current_frame(),
        )
    }

    /// Variant of [`Self::new`] that also runs the per-surface linearize stage
    /// in the shader. Use this from HDR-aware render paths (Path B); pass
    /// `corner_radius=[0; 4]` for non-clipped surfaces if you just want
    /// linearization.
    pub fn new_with_color(
        renderer: &mut R,
        elem: WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
        geometry: Rectangle<f64, Logical>,
        radius: [u8; 4],
        color: ColorTransform,
    ) -> Self
    where
        R: AsGlowRenderer,
    {
        let elem_geo = elem.geometry(scale);
        let geo: Rectangle<i32, Physical> = geometry.to_physical_precise_round(scale);
        let buf_size = elem.buffer_size();
        let view = elem.view();

        let transform = elem.transform();
        let transform_matrix = Matrix3::<f32>::from_translation(Vector2::new(0.5, 0.5))
            * transform.matrix()
            * Matrix3::<f32>::from_translation(-Vector2::new(0.5, 0.5));

        let geo_scale = {
            let Scale { x, y } = elem_geo.size.to_f64() / geo.size.to_f64();
            Matrix3::from_nonuniform_scale(x as f32, y as f32)
        };

        let geo_translation = {
            let offset = (elem_geo.loc - geo.loc).to_f64();
            Matrix3::from_translation(Vector2::new(
                (offset.x / elem_geo.size.w as f64) as f32,
                (offset.y / elem_geo.size.h as f64) as f32,
            ))
        };

        let buf_scale = {
            let Scale { x, y } = buf_size.to_f64() / view.src.size.to_f64();
            Matrix3::from_nonuniform_scale(x as f32, y as f32)
        };

        let buf_translation = Matrix3::from_translation(Vector2::new(
            (view.src.loc.x / buf_size.w as f64) as f32,
            (view.src.loc.y / buf_size.h as f64) as f32,
        ));

        let input_to_geo =
            transform_matrix * geo_scale * geo_translation * buf_scale * buf_translation;

        let uniforms = vec![
            Uniform::new("geo_size", (geometry.size.w as f32, geometry.size.h as f32)),
            Uniform::new(
                "corner_radius",
                [
                    radius[3] as f32,
                    radius[1] as f32,
                    radius[0] as f32,
                    radius[2] as f32,
                ],
            ),
            Uniform::new(
                "input_to_geo",
                UniformValue::Matrix3x3 {
                    matrices: vec![*AsRef::<[f32; 9]>::as_ref(&input_to_geo)],
                    transpose: false,
                },
            ),
            // Path B linearize uniforms — passthrough by default; HDR paths
            // pass meaningful values via [`Self::new_with_color`].
            Uniform::new("tf_id", color.tf_id),
            Uniform::new("ref_white_scale", color.ref_white_scale),
            Uniform::new(
                "primaries_matrix",
                UniformValue::Matrix3x3 {
                    matrices: vec![color.primaries_matrix],
                    transpose: false,
                },
            ),
        ];

        Self {
            inner: elem,
            program: ClippingShader::get(renderer),
            radius,
            geometry,
            uniforms,
        }
    }

    pub fn will_clip(
        elem: &WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
        geometry: Rectangle<f64, Logical>,
        radius: [u8; 4],
    ) -> bool {
        let elem_geo = elem.geometry(scale);
        let geo = geometry.to_physical_precise_round(scale);

        let corners = Self::rounded_corners(geometry, radius);
        let corners = corners
            .into_iter()
            .map(|rect| rect.to_physical_precise_up(scale));
        let geo = Rectangle::subtract_rects_many([geo], corners);
        !Rectangle::subtract_rects_many([elem_geo], geo).is_empty()
    }

    fn rounded_corners(
        geo: Rectangle<f64, Logical>,
        radius: [u8; 4],
    ) -> [Rectangle<f64, Logical>; 4] {
        let top_left = radius[3] as f64;
        let top_right = radius[1] as f64;
        let bottom_right = radius[0] as f64;
        let bottom_left = radius[2] as f64;

        [
            Rectangle::new(geo.loc, Size::from((top_left, top_left))),
            Rectangle::new(
                Point::from((geo.loc.x + geo.size.w - top_right, geo.loc.y)),
                Size::from((top_right, top_right)),
            ),
            Rectangle::new(
                Point::from((
                    geo.loc.x + geo.size.w - bottom_right,
                    geo.loc.y + geo.size.h - bottom_right,
                )),
                Size::from((bottom_right, bottom_right)),
            ),
            Rectangle::new(
                Point::from((geo.loc.x, geo.loc.y + geo.size.h - bottom_left)),
                Size::from((bottom_left, bottom_left)),
            ),
        ]
    }
}

impl<R> Element for ClippedSurfaceRenderElement<R>
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
        // FIXME: radius changes need to cause damage.
        let damage = self.inner.damage_since(scale, commit);

        // Intersect with geometry, since we're clipping by it.
        let mut geo = self.geometry.to_physical_precise_round(scale);
        geo.loc -= self.geometry(scale).loc;
        damage
            .into_iter()
            .filter_map(|rect| rect.intersection(geo))
            .collect()
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        let regions = self.inner.opaque_regions(scale);

        // Intersect with geometry, since we're clipping by it.
        let mut geo = self.geometry.to_physical_precise_round(scale);
        geo.loc -= self.geometry(scale).loc;
        let regions = regions
            .into_iter()
            .filter_map(|rect| rect.intersection(geo));

        // Subtract the rounded corners.
        let corners = Self::rounded_corners(self.geometry, self.radius);

        let elem_loc = self.geometry(scale).loc;
        let corners = corners.into_iter().map(|rect| {
            let mut rect = rect.to_physical_precise_up(scale);
            rect.loc -= elem_loc;
            rect
        });

        OpaqueRegions::from_slice(&Rectangle::subtract_rects_many(regions, corners))
    }

    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }

    fn kind(&self) -> Kind {
        self.inner.kind()
    }
}

impl<R> RenderElement<R> for ClippedSurfaceRenderElement<R>
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
        BorrowMut::<GlesFrame>::borrow_mut(<R as AsGlowRenderer>::glow_frame_mut(frame))
            .override_default_tex_program(self.program.clone(), self.uniforms.clone());
        self.inner
            .draw(frame, src, dst, damage, opaque_regions, cache)?;
        BorrowMut::<GlesFrame>::borrow_mut(<R as AsGlowRenderer>::glow_frame_mut(frame))
            .clear_tex_program_override();
        Ok(())
    }

    fn underlying_storage(&self, _renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        None
    }
}
