// SPDX-License-Identifier: GPL-3.0-only
//
// Phase 3.1.5 — minimal cosmic-comp wiring for `wp_color_management_v1`.
//
// This file provides the smallest viable hookup so the protocol's manager global
// is advertised on the cosmic-comp-hdr session. Behavior is deliberately
// minimal:
//
// - `output_image_description` and `preferred_image_description` use the
//   default trait impls (return interner sRGB). That means clients can bind,
//   build descriptions, query outputs / get_preferred — all without crashing —
//   but the descriptions don't yet drive any rendering decisions.
// - Phase 3.2 wires per-output description synthesis from the existing
//   `hdr_enabled` / `hdr_colorspace` config and routes per-surface descriptions
//   into the render path.
// - Phase 3.3 will read `with_surface_image_description` in smithay's
//   `DrmCompositor` to gate overlay-plane scanout in HDR mode.

use crate::state::State;
use smithay::{
    delegate_color_management,
    wayland::color_management::{ColorManagementHandler, ColorManagementState},
};

impl ColorManagementHandler for State {
    fn color_management_state(&mut self) -> &mut ColorManagementState {
        &mut self.common.color_management_state
    }

    // output_image_description, preferred_image_description, new_image_description,
    // and image_description_failed all use trait defaults for now (sRGB). They'll
    // be overridden in Phase 3.2 to reflect real per-output / per-surface state.
}

delegate_color_management!(State);
