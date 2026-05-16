// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    backend::render::{
        clipped_surface::{ClippedSurfaceRenderElement, LinearizedElement},
        element::AsGlowRenderer,
    },
    utils::prelude::*,
    wayland::handlers::compositor::FRAME_TIME_FILTER,
};
use smithay::{
    backend::{
        allocator::Fourcc,
        renderer::{
            ImportAll, ImportMem, Renderer,
            element::{
                Element, Id, Kind, RenderElement, UnderlyingStorage,
                memory::{MemoryRenderBuffer, MemoryRenderBufferRenderElement},
                surface::{WaylandSurfaceRenderElement, render_elements_from_surface_tree},
            },
            utils::{CommitCounter, DamageSet, OpaqueRegions},
        },
    },
    input::{
        Seat,
        pointer::{CursorIcon, CursorImageAttributes, CursorImageStatus},
    },
    reexports::wayland_server::protocol::wl_surface,
    utils::{
        Buffer as BufferCoords, Logical, Monotonic, Physical, Point, Rectangle, Scale, Size, Time,
        Transform, user_data::UserDataMap,
    },
    wayland::compositor::{get_role, with_states},
};
use std::{collections::HashMap, io::Read, sync::Mutex};
use tracing::warn;
use xcursor::{
    CursorTheme,
    parser::{Image, parse_xcursor},
};

static FALLBACK_CURSOR_DATA: &[u8] = include_bytes!("../../../resources/cursor.rgba");

#[derive(Debug, Clone)]
pub struct Cursor {
    icons: Vec<Image>,
    size: u32,
}

impl Cursor {
    pub fn load(theme: &CursorTheme, shape: CursorIcon, size: u32) -> Cursor {
        let icons = load_icon(theme, shape)
            .map_err(|err| warn!(?err, "Unable to load xcursor, using fallback cursor"))
            .or_else(|_| load_icon(theme, CursorIcon::Default))
            .unwrap_or_else(|_| {
                vec![Image {
                    size: 32,
                    width: 64,
                    height: 64,
                    xhot: 1,
                    yhot: 1,
                    delay: 1,
                    pixels_rgba: Vec::from(FALLBACK_CURSOR_DATA),
                    pixels_argb: vec![], //unused
                }]
            });

        Cursor { icons, size }
    }

    pub fn get_image(&self, scale: u32, millis: u32) -> Image {
        let size = self.size * scale;
        frame(millis, size, &self.icons)
    }
}

fn nearest_images(size: u32, images: &[Image]) -> impl Iterator<Item = &Image> {
    // Follow the nominal size of the cursor to choose the nearest
    let nearest_image = images
        .iter()
        .min_by_key(|image| u32::abs_diff(size, image.size))
        .unwrap();

    images.iter().filter(move |image| {
        image.width == nearest_image.width && image.height == nearest_image.height
    })
}

fn frame(mut millis: u32, size: u32, images: &[Image]) -> Image {
    let total = nearest_images(size, images).fold(0, |acc, image| acc + image.delay);

    if total == 0 {
        millis = 0;
    } else {
        millis %= total;
    }

    for img in nearest_images(size, images) {
        if millis <= img.delay {
            return img.clone();
        }
        millis -= img.delay;
    }

    unreachable!()
}

#[derive(thiserror::Error, Debug)]
enum Error {
    #[error("Theme has no default cursor")]
    NoDefaultCursor,
    #[error("Error opening xcursor file: {0}")]
    File(#[from] std::io::Error),
    #[error("Failed to parse XCursor file")]
    Parse,
}

fn load_icon(theme: &CursorTheme, shape: CursorIcon) -> Result<Vec<Image>, Error> {
    let icon_path = theme
        .load_icon(&shape.to_string())
        .ok_or(Error::NoDefaultCursor)?;
    let mut cursor_file = std::fs::File::open(&icon_path)?;
    let mut cursor_data = Vec::new();
    cursor_file.read_to_end(&mut cursor_data)?;
    parse_xcursor(&cursor_data).ok_or(Error::Parse)
}

/// The cursor is composited into the same offscreen as everything else, so on
/// an HDR output (Path B) it must be linearized too — otherwise its sRGB
/// pixels are written verbatim into the linear RGBA16F buffer and the
/// postprocess PQ-encode blows them up to peak luminance.
///
/// The named/themed cursor is a CPU memory buffer, wrapped in
/// [`LinearizedElement`]; a client-provided cursor surface is a
/// `WaylandSurfaceRenderElement`, wrapped in [`ClippedSurfaceRenderElement`]
/// with no corner radius (linearize-only). Both wrappers are transparent
/// passthroughs outside a Path B HDR frame.
///
/// Hand-written rather than via `render_elements!` because that macro emits
/// the enum with only a `R: Renderer` bound, which can't hold the wrapper
/// types (they require `R: ImportAll + ImportMem`).
pub enum CursorRenderElement<R>
where
    R: Renderer,
{
    Static(LinearizedElement<R>),
    Surface(ClippedSurfaceRenderElement<R>),
}

impl<R> Element for CursorRenderElement<R>
where
    R: Renderer + ImportAll + ImportMem,
{
    fn id(&self) -> &Id {
        match self {
            CursorRenderElement::Static(elem) => elem.id(),
            CursorRenderElement::Surface(elem) => elem.id(),
        }
    }

    fn current_commit(&self) -> CommitCounter {
        match self {
            CursorRenderElement::Static(elem) => elem.current_commit(),
            CursorRenderElement::Surface(elem) => elem.current_commit(),
        }
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        match self {
            CursorRenderElement::Static(elem) => elem.src(),
            CursorRenderElement::Surface(elem) => elem.src(),
        }
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        match self {
            CursorRenderElement::Static(elem) => elem.geometry(scale),
            CursorRenderElement::Surface(elem) => elem.geometry(scale),
        }
    }

    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        match self {
            CursorRenderElement::Static(elem) => elem.location(scale),
            CursorRenderElement::Surface(elem) => elem.location(scale),
        }
    }

    fn transform(&self) -> Transform {
        match self {
            CursorRenderElement::Static(elem) => elem.transform(),
            CursorRenderElement::Surface(elem) => elem.transform(),
        }
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        match self {
            CursorRenderElement::Static(elem) => elem.damage_since(scale, commit),
            CursorRenderElement::Surface(elem) => elem.damage_since(scale, commit),
        }
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        match self {
            CursorRenderElement::Static(elem) => elem.opaque_regions(scale),
            CursorRenderElement::Surface(elem) => elem.opaque_regions(scale),
        }
    }

    fn alpha(&self) -> f32 {
        match self {
            CursorRenderElement::Static(elem) => elem.alpha(),
            CursorRenderElement::Surface(elem) => elem.alpha(),
        }
    }

    fn kind(&self) -> Kind {
        match self {
            CursorRenderElement::Static(elem) => elem.kind(),
            CursorRenderElement::Surface(elem) => elem.kind(),
        }
    }

    fn is_framebuffer_effect(&self) -> bool {
        match self {
            CursorRenderElement::Static(elem) => elem.is_framebuffer_effect(),
            CursorRenderElement::Surface(elem) => elem.is_framebuffer_effect(),
        }
    }

    fn allow_direct_scanout(&self) -> bool {
        match self {
            CursorRenderElement::Static(elem) => elem.allow_direct_scanout(),
            CursorRenderElement::Surface(elem) => elem.allow_direct_scanout(),
        }
    }
}

impl<R> RenderElement<R> for CursorRenderElement<R>
where
    R: AsGlowRenderer + Renderer + ImportAll + ImportMem,
    R::TextureId: 'static,
{
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        match self {
            CursorRenderElement::Static(elem) => {
                elem.draw(frame, src, dst, damage, opaque_regions, cache)
            }
            CursorRenderElement::Surface(elem) => {
                elem.draw(frame, src, dst, damage, opaque_regions, cache)
            }
        }
    }

    fn underlying_storage(&self, renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        match self {
            CursorRenderElement::Static(elem) => elem.underlying_storage(renderer),
            CursorRenderElement::Surface(elem) => elem.underlying_storage(renderer),
        }
    }

    fn capture_framebuffer(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        cache: &UserDataMap,
    ) -> Result<(), R::Error> {
        match self {
            CursorRenderElement::Static(elem) => elem.capture_framebuffer(frame, src, dst, cache),
            CursorRenderElement::Surface(elem) => {
                elem.capture_framebuffer(frame, src, dst, cache)
            }
        }
    }
}

pub fn draw_surface_cursor<R>(
    renderer: &mut R,
    surface: &wl_surface::WlSurface,
    location: Point<f64, Logical>,
    scale: impl Into<Scale<f64>>,
) -> Vec<(CursorRenderElement<R>, Point<i32, Physical>)>
where
    R: Renderer + ImportAll + ImportMem + AsGlowRenderer,
    R::TextureId: Clone + 'static,
{
    let scale = scale.into();
    let h = with_states(surface, |states| {
        states
            .data_map
            .get::<Mutex<CursorImageAttributes>>()
            .unwrap()
            .lock()
            .unwrap()
            .hotspot
            .to_physical_precise_round(scale)
    });

    let surface_elements: Vec<WaylandSurfaceRenderElement<R>> = render_elements_from_surface_tree(
        renderer,
        surface,
        location.to_physical(scale).to_i32_round(),
        scale,
        1.0,
        Kind::Cursor,
    );
    surface_elements
        .into_iter()
        .map(|elem| {
            // Linearize-only wrap (corner_radius = [0; 4] makes the clip/round
            // stage inert). `new` resolves the current frame's color
            // transform — passthrough on SDR outputs, sRGB→linear on HDR.
            let geo = elem.geometry(scale).to_f64().to_logical(scale);
            let clipped =
                ClippedSurfaceRenderElement::new(renderer, elem, scale, geo, [0; 4]);
            (CursorRenderElement::Surface(clipped), h)
        })
        .collect()
}

#[profiling::function]
pub fn draw_dnd_icon<R>(
    renderer: &mut R,
    surface: &wl_surface::WlSurface,
    location: Point<f64, Logical>,
    scale: impl Into<Scale<f64>>,
) -> Vec<WaylandSurfaceRenderElement<R>>
where
    R: Renderer + ImportAll,
    R::TextureId: Clone + 'static,
{
    if get_role(surface) != Some("dnd_icon") {
        warn!(
            ?surface,
            "Trying to display as a dnd icon a surface that does not have the DndIcon role."
        );
    }
    let scale = scale.into();
    render_elements_from_surface_tree(
        renderer,
        surface,
        location.to_physical(scale).to_i32_round(),
        scale,
        1.0,
        FRAME_TIME_FILTER,
    )
}

pub type CursorState = Mutex<CursorStateInner>;
pub struct CursorStateInner {
    current_cursor: Option<CursorIcon>,

    cursor_theme: CursorTheme,
    cursor_size: u32,

    cursors: HashMap<CursorIcon, Cursor>,
    current_image: Option<Image>,
    image_cache: Vec<(Image, MemoryRenderBuffer)>,
}

impl CursorStateInner {
    pub fn set_shape(&mut self, shape: CursorIcon) {
        self.current_cursor = Some(shape);
    }

    pub fn unset_shape(&mut self) {
        self.current_cursor = None;
    }

    pub fn get_named_cursor(&mut self, shape: CursorIcon) -> &Cursor {
        self.cursors
            .entry(shape)
            .or_insert_with(|| Cursor::load(&self.cursor_theme, shape, self.cursor_size))
    }

    pub fn size(&self) -> u32 {
        self.cursor_size
    }
}

pub fn load_cursor_env() -> (String, u32) {
    let name = std::env::var("XCURSOR_THEME")
        .ok()
        .unwrap_or_else(|| "default".into());
    let size = std::env::var("XCURSOR_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    (name, size)
}

pub fn load_cursor_theme() -> (CursorTheme, u32) {
    let (name, size) = load_cursor_env();
    (CursorTheme::load(&name), size)
}

impl Default for CursorStateInner {
    fn default() -> CursorStateInner {
        let (theme, size) = load_cursor_theme();
        CursorStateInner {
            current_cursor: None,

            cursor_size: size,
            cursor_theme: theme,

            cursors: HashMap::new(),
            current_image: None,
            image_cache: Vec::new(),
        }
    }
}

#[profiling::function]
pub fn draw_cursor<R>(
    renderer: &mut R,
    seat: &Seat<State>,
    location: Point<f64, Logical>,
    scale: Scale<f64>,
    buffer_scale: f64,
    time: Time<Monotonic>,
    draw_default: bool,
) -> Vec<(CursorRenderElement<R>, Point<i32, Physical>)>
where
    R: Renderer + ImportMem + ImportAll + AsGlowRenderer,
    R::TextureId: Send + Clone + 'static,
{
    // draw the cursor as relevant
    let cursor_status = seat.cursor_image_status();

    let seat_userdata = seat.user_data();
    let mut state_ref = seat_userdata.get::<CursorState>().unwrap().lock().unwrap();
    let state = &mut *state_ref;

    let named_cursor = state.current_cursor.or(match cursor_status {
        CursorImageStatus::Named(named_cursor) => Some(named_cursor),
        _ => None,
    });
    if let Some(current_cursor) = named_cursor {
        if !draw_default && current_cursor == CursorIcon::Default {
            return Vec::new();
        }

        let integer_scale = (scale.x.max(scale.y) * buffer_scale).ceil() as u32;
        let frame = state
            .get_named_cursor(current_cursor)
            .get_image(integer_scale, time.as_millis());
        let actual_scale = (frame.size / state.size()).max(1);

        let pointer_images = &mut state.image_cache;
        let maybe_image = pointer_images
            .iter()
            .find_map(|(image, texture)| if image == &frame { Some(texture) } else { None });
        let pointer_image = match maybe_image {
            Some(image) => image,
            None => {
                let buffer = MemoryRenderBuffer::from_slice(
                    &frame.pixels_rgba,
                    Fourcc::Argb8888,
                    (frame.width as i32, frame.height as i32),
                    actual_scale as i32,
                    Transform::Normal,
                    None,
                );
                pointer_images.push((frame.clone(), buffer));
                pointer_images.last().map(|(_, i)| i).unwrap()
            }
        };

        let hotspot = Point::<i32, BufferCoords>::from((frame.xhot as i32, frame.yhot as i32))
            .to_logical(
                actual_scale as i32,
                Transform::Normal,
                &Size::from((frame.width as i32, frame.height as i32)),
            );
        state.current_image = Some(frame);

        let memory_element = MemoryRenderBufferRenderElement::from_buffer(
            renderer,
            location.to_physical(scale),
            pointer_image,
            None,
            None,
            None,
            Kind::Cursor,
        )
        .expect("Failed to import cursor bitmap");
        return vec![(
            // Passthrough on SDR; `into_path_b_linearized` attaches the
            // linearize shader when an HDR render frame is active.
            CursorRenderElement::Static(
                LinearizedElement::passthrough(memory_element).into_path_b_linearized(renderer),
            ),
            hotspot.to_physical_precise_round(scale),
        )];
    } else if let CursorImageStatus::Surface(ref wl_surface) = cursor_status {
        return draw_surface_cursor(renderer, wl_surface, location, scale);
    } else {
        Vec::new()
    }
}
