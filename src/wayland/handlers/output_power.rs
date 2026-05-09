// SPDX-License-Identifier: GPL-3.0-only

use smithay::output::Output;

use crate::{
    backend::kms::Surface,
    state::{BackendData, State},
    utils::prelude::OutputExt,
    wayland::protocols::output_power::{
        OutputPowerHandler, OutputPowerState, delegate_output_power,
    },
};

pub fn set_all_surfaces_dpms_on(state: &mut State) {
    let mut changed = false;
    for surface in kms_surfaces(state) {
        if !surface.get_dpms() {
            surface.set_dpms(true);
            changed = true;
        }
    }

    if changed {
        OutputPowerState::refresh(state);
        // After waking from DPMS off, re-apply output config so the HDR
        // state (CRTC color pipeline blobs: DEGAMMA_LUT + CTM + GAMMA_LUT,
        // plus connector Colorspace + HDR_OUTPUT_METADATA) gets re-staged.
        // The kernel can drop CRTC color-block references during the off
        // period, and smithay's pending state might still reference blob
        // IDs that are no longer live on the CRTC — leading to commits
        // that reference stale blobs being rejected by atomic_check, which
        // manifested as a "screen stays black after timeout, can't recover"
        // bug. Re-running refresh_output_config takes the apply path
        // through the HDR-enable code, which creates fresh blobs and
        // pushes them via smithay set_hdr_state. SDR-only setups pay a
        // cheap config-reread; HDR setups get their hardware color
        // pipeline back online cleanly.
        if let Err(err) = state.refresh_output_config() {
            tracing::warn!(?err, "[HDR] Failed to refresh output config on DPMS resume");
        }
    }
}

fn kms_surfaces(state: &mut State) -> impl Iterator<Item = &mut Surface> {
    if let BackendData::Kms(kms_state) = &mut state.backend {
        Some(
            kms_state
                .drm_devices
                .values_mut()
                .flat_map(|device| device.inner.surfaces.values_mut()),
        )
    } else {
        None
    }
    .into_iter()
    .flatten()
}

// Get KMS `Surface` for output, and for all outputs mirroring it
fn kms_surfaces_for_output<'a>(
    state: &'a mut State,
    output: &'a Output,
) -> impl Iterator<Item = &'a mut Surface> + 'a {
    kms_surfaces(state).filter(move |surface| {
        surface.output == *output || surface.output.mirroring().as_ref() == Some(output)
    })
}

// Get KMS `Surface` for output
fn primary_kms_surface_for_output<'a>(
    state: &'a mut State,
    output: &Output,
) -> Option<&'a mut Surface> {
    kms_surfaces(state).find(|surface| surface.output == *output)
}

impl OutputPowerHandler for State {
    fn output_power_state(&mut self) -> &mut OutputPowerState {
        &mut self.common.output_power_state
    }

    fn get_dpms(&mut self, output: &Output) -> Option<bool> {
        let surface = primary_kms_surface_for_output(self, output)?;
        Some(surface.get_dpms())
    }

    fn set_dpms(&mut self, output: &Output, on: bool) {
        for surface in kms_surfaces_for_output(self, output) {
            surface.set_dpms(on);
        }
    }
}

delegate_output_power!(State);
