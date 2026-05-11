// SPDX-License-Identifier: GPL-3.0-only
//
// Phase 3.5 — wp_color_representation_v1 wiring.
//
// Minimal: ColorRepresentationState advertises alpha modes + (coefficients,
// range) pairs per the conservative defaults (premultiplied + straight; BT.709
// and BT.2020 in both full + limited ranges). Per-surface metadata is
// double-buffered by smithay; the render path will consume it once Phase 3.3
// wires `with_surface_color_representation` into plane assignment and the
// shader blend path.

use crate::state::State;
use smithay::{
    delegate_color_representation,
    wayland::color_representation::{ColorRepresentationHandler, ColorRepresentationState},
};

impl ColorRepresentationHandler for State {
    fn color_representation_state(&mut self) -> &mut ColorRepresentationState {
        &mut self.common.color_representation_state
    }
}

delegate_color_representation!(State);
