// SPDX-License-Identifier: GPL-3.0-only
//
// Phase 3.2 — synthesize truthful per-output `wp_image_description_v1` from
// the existing HDR output config.
//
// Behavior:
//
// - SDR output (hdr_enabled = None | Some(false)): returns the interner's
//   pre-populated sRGB description. Identical to the Phase 3.1.5 default.
// - HDR output (hdr_enabled = Some(true)): synthesizes a real description
//   from `hdr_colorspace` (BT.2020 vs DCI-P3), `hdr_reference_white` (default
//   203 cd/m²), and the panel's mastering luminance characteristics
//   (525 nit peak, 0.0005 nit blacks — see `HdrMasteringLuminance`).
//
// `preferred_image_description` returns the surface's primary scanout output's
// description (or sRGB if the surface isn't on any output yet).
//
// **Render path is not yet wired** — surfaces that call `set_image_description`
// have their description double-buffered into smithay's surface state, but the
// existing `postprocess_elements` shader path doesn't yet consult it. SDR
// clients keep working; HDR-aware clients that submit PQ buffers based on our
// truthful output advertisement would be re-encoded incorrectly. Phase 3.2.5
// (or 3.3) wires the render path.

use crate::state::State;
use cosmic_comp_config::output::comp::{HdrColorspace, OutputConfig};
use smithay::{
    delegate_color_management,
    output::Output,
    reexports::{
        wayland_protocols::wp::color_management::v1::server::wp_color_manager_v1::{
            Primaries as ProtoPrimaries, TransferFunction as ProtoTransferFunction,
        },
        wayland_server::protocol::{wl_output::WlOutput, wl_surface::WlSurface},
    },
    wayland::color_management::{
        ColorManagementHandler, ColorManagementState, ImageDescription, Luminances,
        MasteringLuminance, PrimariesDef, TransferFunctionDef,
    },
};
use std::{cell::RefCell, sync::Arc};

/// Panel-level peak luminance, cd/m². Matches `HdrMasteringLuminance::fallback_oled`
/// (Phase 1) — the Tandem OLED's mastering capability.
const PANEL_PEAK_NITS: u32 = 525;
/// Panel-level minimum luminance, scaled by 10,000 per the protocol's
/// `min_lum` units. 0.0005 cd/m² → 5.
const PANEL_MIN_LUM_X10000: u32 = 5;
/// Default SDR reference white when the user hasn't set `hdr_reference_white`.
/// BT.2408 spec value for graded SDR-in-HDR content.
const DEFAULT_REFERENCE_WHITE_NITS: u32 = 203;

impl ColorManagementHandler for State {
    fn color_management_state(&mut self) -> &mut ColorManagementState {
        &mut self.common.color_management_state
    }

    fn output_image_description(&mut self, wl_output: &WlOutput) -> Arc<ImageDescription> {
        let Some(output) = Output::from_resource(wl_output) else {
            // wl_output not associated with a smithay Output (shouldn't happen
            // for our outputs, but be defensive). Default sRGB.
            return self
                .common
                .color_management_state
                .interner()
                .srgb_default();
        };

        let config = read_output_config(&output);
        synthesize_output_description(self, &config)
    }

    fn preferred_image_description(&mut self, surface: &WlSurface) -> Arc<ImageDescription> {
        // Find the surface's primary scanout output via smithay's per-surface
        // state. Returns None if the surface isn't on any output yet (newly
        // mapped, layer-shell during init) — fall back to sRGB; the client
        // will re-query when `preferred_changed[2]` fires.
        let primary_output = smithay::wayland::compositor::with_states(surface, |states| {
            smithay::desktop::utils::surface_primary_scanout_output(surface, states)
        });

        let Some(output) = primary_output else {
            return self
                .common
                .color_management_state
                .interner()
                .srgb_default();
        };

        let config = read_output_config(&output);
        synthesize_output_description(self, &config)
    }
}

/// Read the cached `OutputConfig` from a smithay `Output`'s user-data, or fall
/// back to defaults if it hasn't been attached yet (e.g. very early boot).
fn read_output_config(output: &Output) -> OutputConfig {
    output
        .user_data()
        .get::<RefCell<OutputConfig>>()
        .map(|cell| cell.borrow().clone())
        .unwrap_or_default()
}

/// Build (and intern) an `ImageDescription` matching the given output config.
/// SDR config returns the interner's pre-populated sRGB. HDR config builds a
/// fresh description with the panel's mastering luminance + the user's chosen
/// colorspace + reference white.
fn synthesize_output_description(
    state: &mut State,
    config: &OutputConfig,
) -> Arc<ImageDescription> {
    let hdr_on = config.hdr_enabled.unwrap_or(false);
    if !hdr_on {
        return state
            .common
            .color_management_state
            .interner()
            .srgb_default();
    }

    let colorspace = config.hdr_colorspace.unwrap_or_default(); // Bt2020
    let primaries = match colorspace {
        HdrColorspace::Bt2020 => ProtoPrimaries::Bt2020,
        HdrColorspace::DciP3 => ProtoPrimaries::DciP3,
    };
    let reference_white = config
        .hdr_reference_white
        .unwrap_or(DEFAULT_REFERENCE_WHITE_NITS);

    // Content-aware mastering luminance: max_cll matches reference_white (our
    // SDR-in-HDR content peaks at ref_white, not panel max), max_fall ~40% of
    // peak (typical desktop). max_lum is the panel's actual capability.
    let max_fall = reference_white.saturating_mul(2) / 5;

    let description = ImageDescription {
        primaries: Some(PrimariesDef::Named(primaries)),
        transfer_function: Some(TransferFunctionDef::Named(ProtoTransferFunction::St2084Pq)),
        luminances: Some(Luminances {
            min_lum: PANEL_MIN_LUM_X10000,
            max_lum: PANEL_PEAK_NITS,
            reference_lum: reference_white,
        }),
        mastering_primaries: None, // Optional; matches `primaries` by convention.
        mastering_luminance: Some(MasteringLuminance {
            min_lum: PANEL_MIN_LUM_X10000,
            max_lum: PANEL_PEAK_NITS,
        }),
        max_cll: Some(reference_white),
        max_fall: Some(max_fall),
        icc: None,
        windows_scrgb: false,
        identity: 0,
    };

    state
        .common
        .color_management_state
        .interner_mut()
        .intern(description)
}

delegate_color_management!(State);
