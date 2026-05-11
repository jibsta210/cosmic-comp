#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::len_without_is_empty,
    clippy::collapsible_match
)]
// SPDX-License-Identifier: GPL-3.0-only

use calloop::timer::{TimeoutAction, Timer};
use smithay::{
    reexports::{
        calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic},
        wayland_server::{Display, DisplayHandle},
    },
    wayland::socket::ListeningSocketSource,
};

use anyhow::{Context, Result};
use state::{LastRefresh, State};
use std::{
    env,
    ffi::OsString,
    os::unix::process::CommandExt,
    process,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{error, info, warn};
use wayland::protocols::overlap_notify::OverlapNotifyState;

use crate::wayland::handlers::compositor::client_compositor_state;

use clap_lex::RawArgs;

use std::error::Error;

pub mod backend;
pub mod config;
pub mod dbus;
#[cfg(feature = "debug")]
pub mod debug;
pub mod hooks;
pub mod input;
mod logger;
pub mod session;
pub mod shell;
pub mod state;
#[cfg(feature = "systemd")]
pub mod systemd;
pub mod theme;
pub mod utils;
pub mod wayland;
pub mod xwayland;

#[cfg(feature = "profile-with-tracy")]
#[global_allocator]
static GLOBAL: profiling::tracy_client::ProfiledAllocator<std::alloc::System> =
    profiling::tracy_client::ProfiledAllocator::new(std::alloc::System, 10);

// called by the Xwayland source, either after starting or failing
impl State {
    fn notify_ready(&mut self) {
        // TODO: Don't notify again, but potentially import updated env-variables
        // into systemd and the session?
        self.ready.call_once(|| {
            // potentially tell systemd we are setup now
            if let state::BackendData::Kms(_) = &self.backend {
                #[cfg(feature = "systemd")]
                systemd::ready(&self.common);
                if let Err(err) = dbus::ready(&self.common) {
                    error!(?err, "Failed to update the D-Bus activation environment");
                }
            }

            // potentially tell the session we are setup now
            if let Err(err) =
                session::run_socket(self.common.event_loop_handle.clone(), &self.common)
            {
                warn!(?err, "Failed to setup cosmic-session communication");
            }

            let mut args = env::args().skip(1);
            self.common.kiosk_child = if let Some(exec) = args.next() {
                // Run command in kiosk mode
                let mut command = process::Command::new(&exec);
                command.args(args);
                command.envs(
                    session::get_env(&self.common).expect("WAYLAND_DISPLAY should be valid UTF-8"),
                );
                unsafe {
                    command.pre_exec(|| {
                        utils::rlimit::restore_nofile_limit();
                        Ok(())
                    })
                };

                info!("Running {:?}", exec);
                command
                    .spawn()
                    .map_err(|err| {
                        // TODO: replace with `inspect_err` once stable
                        error!(?err, "Error running kiosk child.");
                        err
                    })
                    .ok()
            } else {
                None
            };
        });
    }
}

pub fn run(hooks: crate::hooks::Hooks) -> Result<(), Box<dyn Error>> {
    let raw_args = RawArgs::from_args();
    let mut cursor = raw_args.cursor();
    let git_hash = option_env!("GIT_HASH").unwrap_or("unknown");

    let mut with_xwayland = true;
    // Parse the arguments
    while let Some(arg) = raw_args.next_os(&mut cursor) {
        match arg.to_str() {
            Some("--help") | Some("-h") => {
                print_help(env!("CARGO_PKG_VERSION"), git_hash);
                return Ok(());
            }
            Some("--no-xwayland") => {
                tracing::info!("Running without Xwayland");
                with_xwayland = false;
            }
            Some("--version") | Some("-V") => {
                println!(
                    "cosmic-comp {} (git commit {})",
                    env!("CARGO_PKG_VERSION"),
                    git_hash
                );
                return Ok(());
            }
            _ => {}
        }
    }

    // setup logger
    logger::init_logger()?;
    info!("Cosmic starting up!");

    profiling::register_thread!("Main Thread");
    #[cfg(feature = "profile-with-tracy")]
    tracy_client::Client::start();

    utils::rlimit::increase_nofile_limit();
    // This needs to be done before any potential program launches
    // (e.g. Xwayland) as it handles passed file descriptors.
    if let Err(err) = session::setup_socket() {
        warn!("Session error: {:?}", err);
    };

    // init hook globals
    hooks::HOOKS.set(hooks)
        .expect("Hooks global has already been initialized. Running multiple instances of COSMIC in one process is not supported.");

    // init event loop
    let mut event_loop = EventLoop::try_new().with_context(|| "Failed to initialize event loop")?;
    // init wayland
    let (display, socket) = init_wayland_display(&mut event_loop)?;
    // init state
    let mut state = state::State::new(
        &display,
        socket,
        event_loop.handle(),
        event_loop.get_signal(),
        with_xwayland,
    );
    // init backend
    backend::init_backend_auto(&display, &mut event_loop, &mut state)?;

    if let Err(err) = theme::watch_theme(event_loop.handle()) {
        warn!(?err, "Failed to watch theme");
    }

    // SIGUSR1 = "HDR tuning changed in outputs.ron, push it to surfaces".
    // SURGICAL path — re-reads the file, then walks each KMS surface and pushes
    // only the HDR shader uniforms via Surface::set_hdr_tuning. Skips the full
    // refresh_output_config apply (which re-runs modes / scales / surface init
    // and feels like a relogin). Used by cosmic-hdr-tuner.
    //
    // We use signal_hook::flag (atomic bool flipped from inside the signal
    // handler) rather than calloop's signalfd because cosmic-comp spawns
    // surface threads BEFORE we'd register the source — those threads don't
    // inherit the signal mask, so signalfd sees nothing while the kernel
    // delivers SIGUSR1 to a worker thread (where it's silently swallowed by
    // smithay/wgpu). The atomic-flag pattern works regardless of which thread
    // the kernel picks. We poll the flag once per event-loop iteration below.
    let hdr_reload_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    if let Err(err) = signal_hook::flag::register(
        signal_hook::consts::SIGUSR1,
        std::sync::Arc::clone(&hdr_reload_flag),
    ) {
        warn!(?err, "Failed to register SIGUSR1 handler for HDR live-reload");
    } else {
        warn!("[HDR] SIGUSR1 hot-reload registered (cosmic-hdr-tuner)");
    }

    // run the event loop
    event_loop.run(None, &mut state, |state| {
        // HDR live-reload: pick up sliders saved by cosmic-hdr-tuner.
        if hdr_reload_flag.swap(false, std::sync::atomic::Ordering::Relaxed) {
            warn!("[HDR] SIGUSR1 received — reloading outputs.ron + pushing HDR tuning");
            state.common.config.dynamic_conf.reload_outputs_from_disk();
            push_hdr_tuning_to_surfaces(state);
        }

        // shall we shut down?
        if state.common.should_stop {
            info!("Shutting down");
            state.common.event_loop_signal.stop();
            state.common.event_loop_signal.wakeup();
            return;
        }

        // trigger routines
        let clients = state.common.shell.write().update_animations();
        {
            let dh = state.common.display_handle.clone();
            for client in clients.values() {
                client_compositor_state(client).blocker_cleared(state, &dh);
            }
        }

        refresh(state);

        {
            let shell = state.common.shell.read();
            if shell.animations_going() {
                for output in shell.outputs().cloned().collect::<Vec<_>>().into_iter() {
                    state.backend.schedule_render(&output);
                }
            }
        }

        // Flush queued color-management destructor events (info `done()` /
        // description `failed(...)`) outside any dispatch callback. Sending
        // these inside dispatch panics wayland-backend (common_poll.rs:284).
        // Must happen before flush_clients so events go out in this round.
        state.common.color_management_state.flush_pending();

        // send out events
        let _ = state.common.display_handle.flush_clients();

        // check if kiosk child is running
        if let Some(child) = state.common.kiosk_child.as_mut() {
            match child.try_wait() {
                // Kiosk child exited with status
                Ok(Some(exit_status)) => {
                    info!("Command exited with status {:?}", exit_status);
                    match exit_status.code() {
                        // Exiting with the same status as the kiosk child
                        Some(code) => process::exit(code),
                        // The kiosk child exited with signal, exiting with error
                        None => process::exit(1),
                    }
                }
                // Command still running
                Ok(None) => {}
                // Kiosk child disappeared, exiting with error
                Err(err) => {
                    warn!(?err, "Failed to wait for command");
                    process::exit(1);
                }
            }
        }
    })?;

    // kill kiosk child if loop exited
    if let Some(mut child) = state.common.kiosk_child.take() {
        let _ = child.kill();
    }

    // drop eventloop & state before logger
    std::mem::drop(event_loop);
    std::mem::drop(state);

    Ok(())
}

/// Surgical SIGUSR1 path: read the freshly-loaded outputs.ron from
/// `state.common.config.dynamic_conf`, walk every KMS surface, and push the
/// new HDR tuning values to its render thread. Also updates the per-Output
/// cached `OutputConfig` so future reads see the new HDR fields. Does NOT
/// re-run modes / scales / surface init — that's the heavy `refresh_output_config`
/// path which feels like a relogin.
fn push_hdr_tuning_to_surfaces(state: &mut state::State) {
    use crate::state::BackendData;
    use crate::utils::prelude::OutputExt;

    // Snapshot the outputs config first (immutable borrow), so we can release
    // it before calling mutable methods on the backend.
    let snapshot: Vec<(String, cosmic_comp_config::output::comp::OutputConfig)> = {
        let outputs = state.common.config.dynamic_conf.outputs();
        let mut v = Vec::new();
        for (infos, configs) in outputs.config.iter() {
            for (info, cfg) in infos.iter().zip(configs.iter()) {
                v.push((info.connector.clone(), cfg.clone()));
            }
        }
        v
    };

    // Track which outputs had HDR config changes so we can fire
    // wp_color_management_v1's image_description_changed events after the
    // backend borrow drops.
    let mut changed_outputs: Vec<smithay::output::Output> = Vec::new();

    let kms = match &mut state.backend {
        BackendData::Kms(k) => k,
        _ => {
            warn!("[HDR] SIGUSR1 received but backend is not KMS — ignored");
            return;
        }
    };

    let mut pushed = 0usize;
    for device in kms.drm_devices.values_mut() {
        // (Live CTM/GAMMA_LUT regen on SIGUSR1 was tried, but couldn't
        // be made stable on Intel xe under motion — re-committing the
        // CRTC color pipeline blobs requires an atomic commit() per
        // slider tick, which xe rejects intermittently and causes
        // moving-window glitches. Reverted; sat/gamma live in shader
        // uniforms instead, ref_white only updates on the next natural
        // commit (mode change, vrr toggle, display reconfig). Phase 3
        // protocols give us a cleaner per-surface live-update path.)

        for surface in device.inner.surfaces.values_mut() {
            let connector_name = surface.output.name();
            let Some(cfg) = snapshot
                .iter()
                .find(|(c, _)| *c == connector_name)
                .map(|(_, c)| c.clone())
            else {
                continue;
            };

            // Update the per-Output cached config so other code sees the new
            // HDR fields without going through the full apply path. Track if
            // the protocol-relevant fields actually shifted; if so, queue this
            // Output for a wp_color_management_v1 change broadcast after the
            // backend borrow ends.
            let protocol_relevant_changed = {
                let mut out_cfg = surface.output.config_mut();
                let changed = out_cfg.hdr_enabled != cfg.hdr_enabled
                    || out_cfg.hdr_colorspace != cfg.hdr_colorspace
                    || out_cfg.hdr_reference_white != cfg.hdr_reference_white;
                out_cfg.hdr_enabled = cfg.hdr_enabled;
                out_cfg.hdr_colorspace = cfg.hdr_colorspace;
                out_cfg.hdr_reference_white = cfg.hdr_reference_white;
                out_cfg.hdr_gamut_strength = cfg.hdr_gamut_strength;
                out_cfg.hdr_saturation = cfg.hdr_saturation;
                out_cfg.hdr_midtone_gamma = cfg.hdr_midtone_gamma;
                out_cfg.hdr_test_pattern = cfg.hdr_test_pattern;
                changed
            };
            if protocol_relevant_changed {
                changed_outputs.push(surface.output.clone());
            }

            // Push to surface render thread.
            let hdr_on = cfg.hdr_enabled.unwrap_or(false);
            surface.set_hdr_enabled(hdr_on);
            if hdr_on {
                let cs_for_shader = match cfg.hdr_colorspace {
                    Some(cosmic_comp_config::output::comp::HdrColorspace::DciP3) => 1.0,
                    _ => 0.0,
                };
                let ref_white = cfg.hdr_reference_white.map(|n| n as f32).unwrap_or(250.0);
                let gamut_mix = cfg.hdr_gamut_strength.map(|p| (p as f32) / 100.0).unwrap_or(1.0);
                let saturation = cfg.hdr_saturation.map(|p| (p as f32) / 100.0).unwrap_or(1.2);
                // Path B's per-surface linearize produces colorimetrically
                // correct SDR luminance (~ref_white cd/m²) which is dim
                // relative to the panel's HDR peak. The CRTC path's default
                // 0.7 midtone exponent makes SDR pop; Path B needs a higher
                // value (~1.4-1.5) to compensate for the different math
                // composition order. Default only kicks in if the user hasn't
                // explicitly set hdr_midtone_gamma in outputs.ron.
                let default_midtone = if crate::backend::render::clipped_surface::path_b_enabled() {
                    1.5
                } else {
                    0.7
                };
                let midtone_gamma = cfg
                    .hdr_midtone_gamma
                    .map(|p| (p as f32) / 100.0)
                    .unwrap_or(default_midtone);
                let test_pattern = cfg.hdr_test_pattern.unwrap_or(false);
                warn!(
                    "[HDR] surgical push to {}: cs={:.1} ref_w={:.1} mix={:.2} sat={:.2} gamma={:.2} test={}",
                    connector_name, cs_for_shader, ref_white, gamut_mix, saturation, midtone_gamma, test_pattern
                );
                surface.set_hdr_tuning(cs_for_shader, ref_white, gamut_mix, saturation, midtone_gamma, test_pattern);

            }
            pushed += 1;
        }
    }

    if pushed == 0 {
        warn!("[HDR] SIGUSR1 surgical reload: no surfaces matched any outputs.ron entry");
    }

    // Backend borrow has dropped. Broadcast wp_color_management_v1's
    // image_description_changed for any output whose protocol-relevant fields
    // shifted. Clients then re-call get_image_description and observe the new
    // BT.2020/PQ description (or sRGB if HDR was just turned off).
    for output in changed_outputs {
        smithay::wayland::color_management::notify_output_image_description_changed(
            state, &output,
        );
    }
}

fn print_help(version: &str, git_rev: &str) {
    println!(
        r#"cosmic-comp {version} (git commit {git_rev})
System76 <info@system76.com>

Designed for the COSMIC™ desktop environment, cosmic-comp is a Wayland Compositor.

Project home page: https://github.com/pop-os/cosmic-comp

Options:
  -h, --help          Show this message
  --no-xwayland       Run without Xwayland
  -v, --version       Show the version of cosmic-comp"#
    );
}

fn init_wayland_display(
    event_loop: &mut EventLoop<state::State>,
) -> Result<(DisplayHandle, OsString)> {
    let display = Display::new().unwrap();
    let handle = display.handle();

    let source = ListeningSocketSource::new_auto().unwrap();
    let socket_name = source.socket_name().to_os_string();
    info!("Listening on {:?}", socket_name);

    event_loop
        .handle()
        .insert_source(source, |client_stream, _, state| {
            let client_state = state.new_client_state();
            if let Err(err) = state
                .common
                .display_handle
                .insert_client(client_stream, Arc::new(client_state))
            {
                warn!(?err, "Error adding wayland client")
            };
        })
        .with_context(|| "Failed to init the wayland socket source.")?;
    event_loop
        .handle()
        .insert_source(
            Generic::new(display, Interest::READ, Mode::Level),
            move |_, display, state| {
                // SAFETY: We don't drop the display
                match unsafe { display.get_mut().dispatch_clients(state) } {
                    Ok(_) => Ok(PostAction::Continue),
                    Err(err) => {
                        error!(?err, "I/O error on the Wayland display");
                        state.common.should_stop = true;
                        Err(err)
                    }
                }
            },
        )
        .with_context(|| "Failed to init the wayland event source.")?;

    Ok((handle, socket_name))
}

fn refresh(state: &mut State) {
    if matches!(state.last_refresh, LastRefresh::Scheduled(_)) {
        return;
    }

    if matches!(state.last_refresh, LastRefresh::At(instant) if Instant::now().duration_since(instant) < Duration::from_millis(150))
    {
        if let Ok(token) = state.common.event_loop_handle.insert_source(
            Timer::from_duration(Duration::from_millis(150)),
            |_, _, state| {
                state.last_refresh = LastRefresh::None;
                TimeoutAction::Drop
            },
        ) {
            state.last_refresh = LastRefresh::Scheduled(token);
            return;
        } else {
            warn!("Failed to schedule refresh");
        }
    }

    state.common.refresh();
    state::Common::refresh_focus(state);
    OverlapNotifyState::refresh(state);
    state.common.update_x11_stacking_order();
    state.last_refresh = LastRefresh::At(Instant::now());
}
