// SPDX-License-Identifier: GPL-3.0-only

use crate::{
    backend::render::{
        CLEAR_COLOR, CursorMode, GlMultiError, GlMultiRenderer, PostprocessOutputConfig,
        PostprocessShader, PostprocessState, ScreencopySdrShader,
        element::{AsGlowRenderer, CosmicElement, DamageElement},
        init_shaders, output_elements,
    },
    config::ScreenFilter,
    shell::Shell,
    state::SurfaceDmabufFeedback,
    utils::prelude::*,
    wayland::handlers::{
        compositor::recursive_frame_time_estimation,
        image_copy_capture::{FrameHolder, PendingImageCopyData, SessionData, submit_buffer},
    },
};

use anyhow::{Context, Result};
use calloop::channel::Channel;
use cosmic_comp_config::output::comp::AdaptiveSync;
use smithay::{
    backend::{
        allocator::{
            Fourcc,
            format::FormatSet,
            gbm::{GbmAllocator, GbmBuffer},
        },
        drm::{
            DrmDeviceFd, DrmEventMetadata, DrmEventTime, DrmNode, VrrSupport,
            compositor::{
                BlitFrameResultError, FrameError, FrameFlags, PrimaryPlaneElement,
                RenderFrameResult,
            },
            exporter::gbm::GbmFramebufferExporter,
            gbm::GbmFramebuffer,
            output::DrmOutput,
        },
        egl::EGLContext,
        renderer::{
            Bind, Blit, BufferType, Frame, ImportDma, Offscreen, Renderer, RendererSuper, Texture,
            TextureFilter, buffer_dimensions, buffer_type,
            damage::Error as RenderError,
            element::{
                Element, Kind, RenderElementStates,
                texture::TextureRenderElement,
                utils::{
                    ConstrainAlign, ConstrainScaleBehavior, Relocate, RelocateRenderElement,
                    constrain_render_elements,
                },
            },
            gles::{
                GlesRenderbuffer, GlesRenderer, GlesTexture, Uniform, element::TextureShaderElement,
            },
            glow::GlowRenderer,
            multigpu::{ApiDevice, Error as MultiError, GpuManager},
            sync::SyncPoint,
            utils::with_renderer_surface_state,
        },
    },
    desktop::utils::OutputPresentationFeedback,
    output::{Output, OutputNoMode},
    reexports::{
        calloop::{
            EventLoop, LoopHandle, RegistrationToken,
            channel::{Event, Sender, channel},
            timer::{TimeoutAction, Timer},
        },
        drm::control::{connector, crtc},
        wayland_protocols::wp::{
            linux_dmabuf::zv1::server::zwp_linux_dmabuf_feedback_v1,
            presentation_time::server::wp_presentation_feedback,
        },
        wayland_server::protocol::wl_surface::WlSurface,
    },
    utils::{Clock, Monotonic, Physical, Point, Rectangle, Transform},
    wayland::{
        dmabuf::{DmabufFeedbackBuilder, get_dmabuf},
        image_copy_capture::{
            CaptureFailureReason, Frame as ScreencopyFrame, SessionRef as ScreencopySessionRef,
        },
        presentation::Refresh,
        seat::WaylandFocus,
        shm::{shm_format_to_fourcc, with_buffer_contents},
    },
};
use tracing::{error, info, trace, warn};

use std::{
    borrow::{Borrow, BorrowMut},
    collections::{HashMap, HashSet, hash_map},
    mem,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender},
    },
    thread::JoinHandle,
    time::Duration,
};

mod timings;
pub use self::timings::Timings;

use super::{drm_helpers, render::gles::GbmGlowBackend};

#[cfg(feature = "debug")]
use smithay_egui::EguiState;

#[derive(Debug)]
pub struct Surface {
    pub(crate) connector: connector::Handle,
    pub(crate) crtc: crtc::Handle,
    pub(crate) output: Output,
    known_nodes: HashSet<DrmNode>,

    active: Arc<AtomicBool>,
    pub feedback: HashMap<DrmNode, SurfaceDmabufFeedback>,
    pub(super) primary_plane_formats: FormatSet,
    overlay_plane_formats: Option<FormatSet>,

    loop_handle: LoopHandle<'static, State>,
    thread_command: Sender<ThreadCommand>,
    thread_token: RegistrationToken,
    thread: Option<JoinHandle<()>>,

    dpms: bool,
}

pub struct SurfaceThreadState {
    // rendering
    api: GpuManager<GbmGlowBackend<DrmDeviceFd>>,
    primary_node: Arc<RwLock<Option<DrmNode>>>,
    target_node: DrmNode,
    active: Arc<AtomicBool>,
    vrr_mode: AdaptiveSync,
    frame_flags: FrameFlags,
    compositor: Option<GbmDrmOutput>,

    state: QueueState,
    timings: Timings,
    frame_callback_seq: usize,
    thread_sender: Sender<SurfaceCommand>,

    output: Output,
    mirroring: Option<Output>,
    screen_filter: ScreenFilter,
    /// HDR signaling state — driven by `output_config.hdr_enabled`. When true,
    /// the surface uses the offscreen postprocess pipeline with the PQ-encode
    /// path (color_mode=5.0) and forces the swapchain to a 10-bit FB format.
    /// See `src/backend/render/shaders/offscreen.frag` for the encode math
    /// and `src/backend/kms/mod.rs` for where this gets toggled.
    hdr_enabled: bool,
    /// Live-tunable HDR shader knobs. Pushed in via `UpdateHdrConfig`. Defaults
    /// match BT.2408 / standards-compliant playback so out-of-the-box HDR is
    /// reasonable; the hdr-tuner GUI overrides these for live experimentation.
    hdr_colorspace_for_shader: f32, // 0.0 = BT.2020, 1.0 = DCI-P3
    hdr_ref_white: f32,             // cd/m^2
    hdr_gamut_mix: f32,             // 0.0..=1.0
    hdr_saturation: f32,            // 1.0 = neutral, >1.0 = more vivid
    hdr_midtone_gamma: f32,         // 1.0 = neutral, <1.0 lifts SDR midtones into HDR range
    hdr_test_pattern: bool,
    /// True when the kernel CRTC color pipeline (DEGAMMA + CTM + GAMMA)
    /// is staged for this surface — the postprocess shader's PQ-encode
    /// path becomes redundant (shader uses color_mode=7.0 = passthrough,
    /// hardware does the encode). When false, shader runs the full
    /// software encoding path (color_mode=5.0).
    hdr_hardware_path_active: bool,
    postprocess_textures: HashMap<DrmNode, PostprocessState>,

    shell: Arc<parking_lot::RwLock<Shell>>,

    loop_handle: LoopHandle<'static, Self>,
    clock: Clock<Monotonic>,

    #[cfg(feature = "debug")]
    egui: EguiState,

    last_sequence: Option<u32>,
    /// Tracy frame that goes from vblank to vblank.
    vblank_frame: Option<tracy_client::Frame>,
    /// Frame name for the VBlank frame.
    vblank_frame_name: tracy_client::FrameName,
    /// Plot name for the time since presentation plot.
    time_since_presentation_plot_name: tracy_client::PlotName,
    /// Plot name for the presentation misprediction plot.
    presentation_misprediction_plot_name: tracy_client::PlotName,
    sequence_delta_plot_name: tracy_client::PlotName,
}

pub type GbmDrmOutput = DrmOutput<
    GbmAllocator<DrmDeviceFd>,
    GbmFramebufferExporter<DrmDeviceFd>,
    Option<(
        OutputPresentationFeedback,
        Receiver<PendingImageCopyData>,
        Duration,
    )>,
    DrmDeviceFd,
>;

#[derive(Debug, Default)]
pub enum QueueState {
    #[default]
    Idle,
    /// A redraw is queued.
    Queued(RegistrationToken),
    /// We submitted a frame to the KMS and waiting for it to be presented.
    WaitingForVBlank { redraw_needed: bool },
    /// We did not submit anything to KMS and made a timer to fire at the estimated VBlank.
    WaitingForEstimatedVBlank(RegistrationToken),
    /// A redraw is queued on top of the above.
    WaitingForEstimatedVBlankAndQueued {
        estimated_vblank: RegistrationToken,
        queued_render: RegistrationToken,
    },
}

#[derive(Debug)]
pub enum ThreadCommand {
    Suspend(SyncSender<()>),
    Resume {
        compositor: GbmDrmOutput,
    },
    NodeAdded {
        node: DrmNode,
        gbm: GbmAllocator<DrmDeviceFd>,
        egl: EGLContext,
        sync: SyncSender<()>,
    },
    NodeRemoved {
        node: DrmNode,
        sync: SyncSender<()>,
    },
    UpdateMirroring(Option<Output>),
    UpdateScreenFilter(ScreenFilter),
    UpdateHdrEnabled(bool),
    UpdateHdrHardwarePath(bool),
    UpdateHdrTuning {
        colorspace_for_shader: f32,
        ref_white: f32,
        gamut_mix: f32,
        saturation: f32,
        midtone_gamma: f32,
        test_pattern: bool,
    },
    VBlank(Option<DrmEventMetadata>),
    ScheduleRender,
    AdaptiveSyncAvailable(SyncSender<Result<VrrSupport>>),
    UseAdaptiveSync(AdaptiveSync),
    AllowFrameFlags(bool, FrameFlags),
    End,
    DpmsOff,
}

#[derive(Debug)]
pub enum SurfaceCommand {
    SendFrames(usize),
    RenderStates(RenderElementStates),
}

#[derive(Debug, Default)]
struct PrePostprocessData {
    states: Option<RenderElementStates>,
    texture: Option<GlesTexture>,
    cursor_texture: Option<GlesTexture>,
    cursor_geometry: Option<Rectangle<i32, Physical>>,
}

impl Surface {
    pub fn new(
        output: &Output,
        crtc: crtc::Handle,
        connector: connector::Handle,
        primary_node: Arc<RwLock<Option<DrmNode>>>,
        dev_node: DrmNode,
        target_node: DrmNode,
        evlh: &LoopHandle<'static, State>,
        screen_filter: ScreenFilter,
        shell: Arc<parking_lot::RwLock<Shell>>,
        startup_done: Arc<AtomicBool>,
    ) -> Result<Self> {
        let (tx, rx) = channel::<ThreadCommand>();
        let (tx2, rx2) = channel::<SurfaceCommand>();
        let active = Arc::new(AtomicBool::new(false));

        let active_clone = active.clone();
        let output_clone = output.clone();

        let thread = std::thread::Builder::new()
            .name(format!("surface-{}", output.name()))
            .spawn(move || {
                if let Err(err) = surface_thread(
                    output_clone,
                    primary_node,
                    target_node,
                    shell,
                    active_clone,
                    screen_filter,
                    tx2,
                    rx,
                    startup_done,
                ) {
                    error!("Surface thread crashed: {}", err);
                }
            })
            .context("Failed to spawn surface thread")?;

        let output_clone = output.clone();
        let thread_token = evlh
            .insert_source(rx2, move |command, _, state| match command {
                Event::Msg(SurfaceCommand::SendFrames(sequence)) => {
                    if output_clone.mirroring().is_some() {
                        return;
                    }
                    state.common.send_frames(&output_clone, Some(sequence));
                }
                Event::Msg(SurfaceCommand::RenderStates(states)) => {
                    if output_clone.mirroring().is_some() {
                        return;
                    }
                    state.common.update_primary_output(&output_clone, &states);
                    let kms = state.backend.kms();
                    let surface = &mut kms
                        .drm_devices
                        .get_mut(&dev_node)
                        .unwrap()
                        .inner
                        .surfaces
                        .get_mut(&crtc)
                        .unwrap();

                    state
                        .common
                        .send_dmabuf_feedback(&output_clone, &states, |source_node| {
                            if let Some(cached_feedback) = surface.feedback.get(&source_node) {
                                Some(cached_feedback.clone())
                            } else {
                                // If we have freed the node, because it didn't have any active buffers/surfaces,
                                // we might not be able to evaluate surface feedback yet.
                                let render_formats =
                                    kms.api.single_renderer(&source_node).ok()?.dmabuf_formats();
                                // In contrast we must have the target node, if we have an active surface
                                let target_formats = kms
                                    .api
                                    .single_renderer(&target_node)
                                    .unwrap()
                                    .dmabuf_formats();
                                let feedback = get_surface_dmabuf_feedback(
                                    source_node,
                                    target_node,
                                    render_formats,
                                    target_formats,
                                    surface.primary_plane_formats.clone(),
                                    surface.overlay_plane_formats.clone(),
                                );
                                surface.feedback.insert(source_node, feedback.clone());
                                Some(feedback)
                            }
                        });
                }
                Event::Closed => {}
            })
            .map_err(|_| anyhow::anyhow!("Failed to establish channel to surface thread"))?;

        Ok(Surface {
            connector,
            crtc,
            output: output.clone(),
            known_nodes: HashSet::new(),
            active,
            feedback: HashMap::new(),
            primary_plane_formats: FormatSet::default(),
            overlay_plane_formats: None,
            loop_handle: evlh.clone(),
            thread_command: tx,
            thread_token,
            thread: Some(thread),
            dpms: true,
        })
    }

    pub fn known_nodes(&self) -> &HashSet<DrmNode> {
        &self.known_nodes
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    pub fn add_node(&mut self, node: DrmNode, gbm: GbmAllocator<DrmDeviceFd>, egl: EGLContext) {
        self.known_nodes.insert(node);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self.thread_command.send(ThreadCommand::NodeAdded {
            node,
            gbm,
            egl,
            sync: tx,
        });
        let _ = rx.recv();
    }

    pub fn remove_node(&mut self, node: DrmNode) {
        self.known_nodes.remove(&node);
        self.feedback.remove(&node);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self
            .thread_command
            .send(ThreadCommand::NodeRemoved { node, sync: tx });
        // Block so we can be sure the file descriptor is closed
        // (which is relevant for the udev device_removed callback).
        let _ = rx.recv();
    }

    pub fn on_vblank(&self, metadata: Option<DrmEventMetadata>) {
        let _ = self.thread_command.send(ThreadCommand::VBlank(metadata));
    }

    pub fn schedule_render(&self) {
        if self.dpms {
            let _ = self.thread_command.send(ThreadCommand::ScheduleRender);
        }
    }

    pub fn set_mirroring(&mut self, output: Option<Output>) {
        let _ = self
            .thread_command
            .send(ThreadCommand::UpdateMirroring(output));
    }

    pub fn set_screen_filter(&mut self, config: ScreenFilter) {
        let _ = self
            .thread_command
            .send(ThreadCommand::UpdateScreenFilter(config));
    }

    /// Toggle HDR PQ post-process for this surface. When `true` the surface
    /// runs the offscreen postprocess pipeline with `color_mode=5.0` (PQ
    /// encode) regardless of `screen_filter` state, so output is BT.2020/PQ
    /// for the panel that's been signaled into HDR mode.
    /// Tell the surface whether the kernel CRTC color pipeline is doing
    /// the HDR encode (DEGAMMA+CTM+GAMMA staged via HdrState). When true
    /// the shader switches to color_mode=7.0 (passthrough) so the encode
    /// only happens once, in hardware. When false the shader runs the
    /// full software path (color_mode=5.0).
    pub fn set_hdr_hardware_path(&mut self, active: bool) {
        let _ = self
            .thread_command
            .send(ThreadCommand::UpdateHdrHardwarePath(active));
    }

    pub fn set_hdr_enabled(&mut self, enabled: bool) {
        warn!("[HDR] Surface::set_hdr_enabled({enabled}) -> dispatching ThreadCommand");
        let send_result = self
            .thread_command
            .send(ThreadCommand::UpdateHdrEnabled(enabled));
        if let Err(err) = send_result {
            warn!(
                ?err,
                "[HDR] Surface::set_hdr_enabled — thread_command channel send failed"
            );
        }
    }

    /// Push live HDR shader tuning to the surface's render thread. Cheap —
    /// just stores values that the next frame's `postprocess_elements` will
    /// pass as GLES uniforms. Driven by the hdr-tuner GUI watcher.
    pub fn set_hdr_tuning(
        &mut self,
        colorspace_for_shader: f32,
        ref_white: f32,
        gamut_mix: f32,
        saturation: f32,
        midtone_gamma: f32,
        test_pattern: bool,
    ) {
        let _ = self.thread_command.send(ThreadCommand::UpdateHdrTuning {
            colorspace_for_shader,
            ref_white,
            gamut_mix,
            saturation,
            midtone_gamma,
            test_pattern,
        });
    }

    pub fn adaptive_sync_support(&self) -> Result<VrrSupport> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self
            .thread_command
            .send(ThreadCommand::AdaptiveSyncAvailable(tx));
        rx.recv().context("Surface thread died")?
    }

    pub fn use_adaptive_sync(&mut self, vrr: AdaptiveSync) {
        let _ = self
            .thread_command
            .send(ThreadCommand::UseAdaptiveSync(vrr));
    }

    pub fn allow_frame_flags(&mut self, flag: bool, flags: FrameFlags) {
        let _ = self
            .thread_command
            .send(ThreadCommand::AllowFrameFlags(flag, flags));
    }

    pub fn suspend(&mut self) {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self.thread_command.send(ThreadCommand::Suspend(tx));
        let _ = rx.recv();
    }

    pub fn resume(
        &mut self,
        compositor: GbmDrmOutput,
        primary_plane_formats: FormatSet,
        overlay_plane_formats: Option<FormatSet>,
    ) {
        self.primary_plane_formats = primary_plane_formats;
        self.overlay_plane_formats = overlay_plane_formats;
        self.feedback.clear();
        self.active.store(true, Ordering::SeqCst);
        self.dpms = true;

        let _ = self
            .thread_command
            .send(ThreadCommand::Resume { compositor });
    }

    pub fn get_dpms(&mut self) -> bool {
        self.dpms
    }

    pub fn set_dpms(&mut self, on: bool) {
        if self.dpms != on {
            self.dpms = on;
            if on {
                self.schedule_render();
            } else {
                let _ = self.thread_command.send(ThreadCommand::DpmsOff);
            }
        }
    }

    pub fn drop_and_join(mut self) {
        let thread = self.thread.take();
        let _ = self;
        if let Some(thread) = thread {
            let name = thread.thread().name().unwrap().to_string();
            let _ = thread.join();
            info!("Thread {} terminated.", name)
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let _ = self.thread_command.send(ThreadCommand::End);
        self.loop_handle.remove(self.thread_token);
        if let Some(thread) = self.thread.take() {
            let _ = thread;
            // We want to do this, but this currently deadlocks on `apply_config_for_outputs`.
            /*
                let name = thread.thread().name().unwrap().to_string();
                let _ = thread.join();
                info!("Thread {} terminated.", name)
            */
        }
    }
}

fn surface_thread(
    output: Output,
    primary_node: Arc<RwLock<Option<DrmNode>>>,
    target_node: DrmNode,
    shell: Arc<parking_lot::RwLock<Shell>>,
    active: Arc<AtomicBool>,
    screen_filter: ScreenFilter,
    thread_sender: Sender<SurfaceCommand>,
    thread_receiver: Channel<ThreadCommand>,
    startup_done: Arc<AtomicBool>,
) -> Result<()> {
    let name = output.name();
    profiling::register_thread!(&format!("Surface Thread {}", name));

    let mut event_loop = EventLoop::try_new().unwrap();

    let api = GpuManager::new(GbmGlowBackend::<DrmDeviceFd>::default())
        .context("Failed to initialize rendering api")?;

    #[cfg(feature = "debug")]
    let egui = {
        let state =
            smithay_egui::EguiState::new(smithay::utils::Rectangle::from_size((400, 800).into()));
        let visuals = egui::style::Visuals {
            window_shadow: egui::Shadow::NONE,
            ..Default::default()
        };
        state.context().set_visuals(visuals);
        state
    };

    let vblank_frame_name = tracy_client::FrameName::new_leak(format!("vblank on {name}"));
    let time_since_presentation_plot_name =
        tracy_client::PlotName::new_leak(format!("{name} time since presentation, ms"));
    let presentation_misprediction_plot_name =
        tracy_client::PlotName::new_leak(format!("{name} presentation misprediction, ms"));
    let sequence_delta_plot_name =
        tracy_client::PlotName::new_leak(format!("{name} sequence delta"));

    let mut state = SurfaceThreadState {
        api,
        primary_node,
        target_node,
        active,
        compositor: None,
        frame_flags: FrameFlags::DEFAULT,
        vrr_mode: AdaptiveSync::Disabled,

        state: QueueState::Idle,
        timings: Timings::new(None, None, false, target_node),
        frame_callback_seq: 0,
        thread_sender,

        output,
        mirroring: None,
        screen_filter,
        hdr_enabled: false,
        hdr_colorspace_for_shader: 0.0, // BT.2020 default
        hdr_ref_white: 250.0,           // close to KWin / BT.2408 default of 200-203 nits
        hdr_gamut_mix: 1.0,             // full conversion default — partial mix is "washed out"
        hdr_saturation: 1.2,            // mild vibrance boost to compensate for loss of vendor SDR enhancement
        hdr_midtone_gamma: 0.7,         // lift SDR midtones into HDR luminance range (Windows AutoHDR-like)
        hdr_test_pattern: false,
        hdr_hardware_path_active: false,
        postprocess_textures: HashMap::new(),

        shell,
        loop_handle: event_loop.handle(),
        clock: Clock::new(),
        #[cfg(feature = "debug")]
        egui,

        last_sequence: None,
        vblank_frame: None,
        vblank_frame_name,
        time_since_presentation_plot_name,
        presentation_misprediction_plot_name,
        sequence_delta_plot_name,
    };

    let signal = event_loop.get_signal();
    event_loop
        .handle()
        .insert_source(thread_receiver, move |command, _, state| match command {
            Event::Msg(ThreadCommand::Suspend(tx)) => state.suspend(tx),
            Event::Msg(ThreadCommand::Resume { compositor }) => {
                state.resume(compositor);
            }
            Event::Msg(ThreadCommand::NodeAdded {
                node,
                gbm,
                egl,
                sync,
            }) => {
                if let Err(err) = state.node_added(node, gbm, egl) {
                    warn!(?err, ?node, "Failed to add node to surface-thread");
                }
                let _ = sync.send(());
            }
            Event::Msg(ThreadCommand::NodeRemoved { node, sync }) => {
                state.node_removed(node);
                let _ = sync.send(());
            }
            Event::Msg(ThreadCommand::VBlank(metadata)) => {
                state.on_vblank(metadata);
            }
            Event::Msg(ThreadCommand::ScheduleRender) => {
                if !startup_done.load(Ordering::SeqCst) {
                    return;
                }

                state.queue_redraw(false);
            }
            Event::Msg(ThreadCommand::UpdateMirroring(mirroring_output)) => {
                state.update_mirroring(mirroring_output);
            }
            Event::Msg(ThreadCommand::UpdateScreenFilter(filter_config)) => {
                state.update_screen_filter(filter_config);
            }
            Event::Msg(ThreadCommand::UpdateHdrHardwarePath(active)) => {
                warn!(
                    "[HDR-HW] surface thread: hardware color pipeline active = {} (was {})",
                    active, state.hdr_hardware_path_active
                );
                state.hdr_hardware_path_active = active;
                state.queue_redraw(false);
            }
            Event::Msg(ThreadCommand::UpdateHdrEnabled(enabled)) => {
                warn!(
                    "[HDR] thread received UpdateHdrEnabled({enabled}) — was {} now {}",
                    state.hdr_enabled, enabled
                );
                state.hdr_enabled = enabled;
            }
            Event::Msg(ThreadCommand::UpdateHdrTuning {
                colorspace_for_shader,
                ref_white,
                gamut_mix,
                saturation,
                midtone_gamma,
                test_pattern,
            }) => {
                warn!(
                    "[HDR] surface thread received UpdateHdrTuning: cs={:.1} ref_w={:.1} mix={:.2} sat={:.2} gamma={:.2} test={}",
                    colorspace_for_shader, ref_white, gamut_mix, saturation, midtone_gamma, test_pattern
                );
                state.hdr_colorspace_for_shader = colorspace_for_shader;
                state.hdr_ref_white = ref_white;
                state.hdr_gamut_mix = gamut_mix;
                state.hdr_saturation = saturation;
                state.hdr_midtone_gamma = midtone_gamma;
                state.hdr_test_pattern = test_pattern;
                state.queue_redraw(false);
            }
            Event::Msg(ThreadCommand::AdaptiveSyncAvailable(result)) => {
                if let Some(compositor) = state.compositor.as_mut() {
                    let _ = result.send(
                        compositor
                            .with_compositor(|c| {
                                c.vrr_supported(c.pending_connectors().into_iter().next().unwrap())
                            })
                            .map_err(Into::into),
                    );
                } else {
                    let _ = result.send(Err(anyhow::anyhow!("Set vrr with inactive surface")));
                }
            }
            Event::Msg(ThreadCommand::UseAdaptiveSync(vrr)) => {
                state.vrr_mode = vrr;
            }
            Event::Msg(ThreadCommand::DpmsOff) => {
                if let Some(compositor) = state.compositor.as_mut() {
                    if let Err(err) = compositor.with_compositor(|c| c.clear()) {
                        error!("Failed to set DPMS off: {:?}", err);
                    }
                    match std::mem::replace(&mut state.state, QueueState::Idle) {
                        QueueState::Idle => {}
                        QueueState::Queued(token)
                        | QueueState::WaitingForEstimatedVBlank(token) => {
                            state.loop_handle.remove(token);
                        }
                        QueueState::WaitingForVBlank { .. } => {
                            state.timings.discard_current_frame()
                        }
                        QueueState::WaitingForEstimatedVBlankAndQueued {
                            estimated_vblank,
                            queued_render,
                        } => {
                            state.loop_handle.remove(estimated_vblank);
                            state.loop_handle.remove(queued_render);
                        }
                    };
                }
            }
            Event::Msg(ThreadCommand::AllowFrameFlags(flag, mut flags)) => {
                if crate::utils::env::bool_var("COSMIC_DISABLE_DIRECT_SCANOUT").unwrap_or(false) {
                    flags.remove(FrameFlags::ALLOW_SCANOUT);
                }
                if crate::utils::env::bool_var("COSMIC_DISABLE_OVERLAY_SCANOUT").unwrap_or(false) {
                    flags.remove(FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT);
                }

                if flag {
                    state.frame_flags.insert(flags);
                } else {
                    state.frame_flags.remove(flags);
                }
            }
            Event::Closed | Event::Msg(ThreadCommand::End) => {
                signal.stop();
                signal.wakeup();
            }
        })
        .map_err(|insert_error| insert_error.error)
        .context("Failed to listen for events")?;

    event_loop.run(None, &mut state, |_| {}).map_err(Into::into)
}

impl SurfaceThreadState {
    fn suspend(&mut self, tx: SyncSender<()>) {
        self.active.store(false, Ordering::SeqCst);
        let _ = self.compositor.take();

        match std::mem::replace(&mut self.state, QueueState::Idle) {
            QueueState::Idle => {}
            QueueState::Queued(token) | QueueState::WaitingForEstimatedVBlank(token) => {
                self.loop_handle.remove(token);
            }
            QueueState::WaitingForVBlank { .. } => self.timings.discard_current_frame(),
            QueueState::WaitingForEstimatedVBlankAndQueued {
                estimated_vblank,
                queued_render,
            } => {
                self.loop_handle.remove(estimated_vblank);
                self.loop_handle.remove(queued_render);
            }
        };

        let _ = tx.send(());
    }

    fn resume(&mut self, compositor: GbmDrmOutput) {
        let (mode, min_hz) = compositor.with_compositor(|c| {
            (
                c.surface().pending_mode(),
                drm_helpers::get_minimum_refresh_rate(
                    c.surface(),
                    c.pending_connectors().into_iter().next().unwrap(),
                )
                .ok()
                .flatten(),
            )
        });
        let interval =
            Duration::from_secs_f64(1_000. / drm_helpers::calculate_refresh_rate(mode) as f64);
        self.timings.set_refresh_interval(Some(interval));

        const SAFETY_MARGIN: u32 = 2; // Magic two frames margin taken from kwin to not trigger low-framerate-compensation
        let min_min_refresh_interval = Duration::from_secs_f64(1. / 30.); // 30Hz
        self.timings.set_min_refresh_interval(Some(
            min_hz
                .map(|min| Duration::from_secs_f64(1. / (min + SAFETY_MARGIN) as f64))
                .unwrap_or(min_min_refresh_interval) // alternatively use 30Hz
                .max(min_min_refresh_interval),
        ));

        if crate::utils::env::bool_var("COSMIC_DISABLE_DIRECT_SCANOUT").unwrap_or(false) {
            self.frame_flags.remove(FrameFlags::ALLOW_SCANOUT);
        } else if crate::utils::env::bool_var("COSMIC_DISABLE_OVERLAY_SCANOUT").unwrap_or(false) {
            self.frame_flags
                .remove(FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT);
        }
        self.compositor = Some(compositor);
    }

    fn node_added(
        &mut self,
        node: DrmNode,
        gbm: GbmAllocator<DrmDeviceFd>,
        egl: EGLContext,
    ) -> Result<()> {
        let mut renderer =
            unsafe { GlowRenderer::new(egl) }.context("Failed to create renderer")?;
        init_shaders(renderer.borrow_mut()).context("Failed to initialize shaders")?;

        self.api.as_mut().add_node(node, gbm, renderer);

        Ok(())
    }

    fn node_removed(&mut self, node: DrmNode) {
        self.api.as_mut().remove_node(&node);
        // force enumeration
        let _ = self.api.devices();
    }

    #[profiling::function]
    fn on_vblank(&mut self, metadata: Option<DrmEventMetadata>) {
        let Some(compositor) = self.compositor.as_mut() else {
            return;
        };

        // handle edge-cases right after resume
        if !matches!(
            self.state,
            QueueState::WaitingForVBlank { .. } | QueueState::Idle
        ) {
            match mem::replace(&mut self.state, QueueState::Idle) {
                QueueState::WaitingForVBlank { .. } | QueueState::Idle => unreachable!(),
                QueueState::Queued(token) | QueueState::WaitingForEstimatedVBlank(token) => {
                    self.loop_handle.remove(token);
                }
                QueueState::WaitingForEstimatedVBlankAndQueued {
                    estimated_vblank,
                    queued_render,
                } => {
                    self.loop_handle.remove(estimated_vblank);
                    self.loop_handle.remove(queued_render);
                }
            }
        }
        if matches!(self.state, QueueState::Idle) {
            return;
        }

        let now = self.clock.now();
        let presentation_time = match metadata.as_ref().map(|data| &data.time) {
            Some(DrmEventTime::Monotonic(tp)) => Some(*tp),
            _ => None,
        };
        let sequence = metadata.as_ref().map(|data| data.sequence).unwrap_or(0);

        // finish tracy frame
        let _ = self.vblank_frame.take();

        // mark last frame completed
        if let Ok(Some(Some((mut feedback, frames, estimated_presentation_time)))) =
            compositor.frame_submitted()
            && self.mirroring.is_none()
        {
            let name = self.output.name();
            let message = if let Some(presentation_time) = presentation_time {
                let misprediction_s =
                    presentation_time.as_secs_f64() - estimated_presentation_time.as_secs_f64();
                tracy_client::Client::running().unwrap().plot(
                    self.presentation_misprediction_plot_name,
                    misprediction_s * 1000.,
                );

                let now = Duration::from(now);
                if presentation_time > now {
                    let diff = presentation_time - now;
                    tracy_client::Client::running().unwrap().plot(
                        self.time_since_presentation_plot_name,
                        -diff.as_secs_f64() * 1000.,
                    );
                    format!("vblank on {name}, presentation is {diff:?} later")
                } else {
                    let diff = now - presentation_time;
                    tracy_client::Client::running().unwrap().plot(
                        self.time_since_presentation_plot_name,
                        diff.as_secs_f64() * 1000.,
                    );
                    format!("vblank on {name}, presentation was {diff:?} ago")
                }
            } else {
                format!("vblank on {name}, presentation time unknown")
            };
            tracy_client::Client::running()
                .unwrap()
                .message(&message, 0);

            let (clock, flags) = if let Some(tp) = presentation_time {
                (
                    tp.into(),
                    wp_presentation_feedback::Kind::Vsync
                        | wp_presentation_feedback::Kind::HwClock
                        | wp_presentation_feedback::Kind::HwCompletion,
                )
            } else {
                (
                    now,
                    wp_presentation_feedback::Kind::Vsync
                        | wp_presentation_feedback::Kind::HwCompletion,
                )
            };

            let rate = self
                .output
                .current_mode()
                .map(|mode| Duration::from_secs_f64(1_000.0 / mode.refresh as f64));
            let refresh = match rate {
                Some(rate)
                    if self
                        .compositor
                        .as_ref()
                        .is_some_and(|comp| comp.with_compositor(|c| c.vrr_enabled())) =>
                {
                    Refresh::Variable(rate)
                }
                Some(rate) => Refresh::Fixed(rate),
                None => Refresh::Unknown,
            };

            if let Some(last_sequence) = self.last_sequence {
                let delta = sequence as f64 - last_sequence as f64;
                tracy_client::Client::running()
                    .unwrap()
                    .plot(self.sequence_delta_plot_name, delta);
            }
            self.last_sequence = Some(sequence);

            feedback.presented(clock, refresh, sequence as u64, flags);

            self.timings.presented(clock);

            while let Ok(pending_image_copy_data) = frames.recv() {
                pending_image_copy_data.send_success_when_ready(
                    self.output.current_transform(),
                    &self.loop_handle,
                    clock,
                );
            }
        }

        let redraw_needed = match mem::replace(&mut self.state, QueueState::Idle) {
            QueueState::Idle => unreachable!(),
            QueueState::Queued(_) => unreachable!(),
            QueueState::WaitingForVBlank { redraw_needed } => redraw_needed,
            QueueState::WaitingForEstimatedVBlank(_) => unreachable!(),
            QueueState::WaitingForEstimatedVBlankAndQueued { .. } => unreachable!(),
        };

        if redraw_needed || self.shell.read().animations_going() {
            let vblank_frame = tracy_client::Client::running()
                .unwrap()
                .non_continuous_frame(self.vblank_frame_name);
            self.vblank_frame = Some(vblank_frame);

            self.queue_redraw(false);
        }
        self.send_frame_callbacks();
    }

    #[profiling::function]
    fn on_estimated_vblank(&mut self, force: bool) {
        match mem::replace(&mut self.state, QueueState::Idle) {
            QueueState::Idle => unreachable!(),
            QueueState::Queued(_) => unreachable!(),
            QueueState::WaitingForVBlank { .. } => unreachable!(),
            QueueState::WaitingForEstimatedVBlank(_) => (),
            // The timer fired just in front of a redraw.
            QueueState::WaitingForEstimatedVBlankAndQueued { queued_render, .. } => {
                self.state = QueueState::Queued(queued_render);
                return;
            }
        }

        self.frame_callback_seq = self.frame_callback_seq.wrapping_add(1);

        if force || self.shell.read().animations_going() {
            self.queue_redraw(false);
        }
        self.send_frame_callbacks();
    }

    fn queue_redraw(&mut self, force: bool) {
        let Some(_compositor) = self.compositor.as_mut() else {
            return;
        };

        if let QueueState::WaitingForVBlank { .. } = &self.state {
            // We're waiting for VBlank, request a redraw afterwards.
            self.state = QueueState::WaitingForVBlank {
                redraw_needed: true,
            };
            return;
        }

        if !force {
            match &self.state {
                QueueState::Idle | QueueState::WaitingForEstimatedVBlank(_) => {}

                // A redraw is already queued.
                QueueState::Queued(_) | QueueState::WaitingForEstimatedVBlankAndQueued { .. } => {
                    return;
                }
                _ => unreachable!(),
            };
        }

        let estimated_presentation = self.timings.next_presentation_time(&self.clock);
        let render_start = self.timings.next_render_time(&self.clock);

        let timer = if render_start.is_zero() {
            trace!("Running late for frame.");
            // TODO triple buffering
            Timer::immediate()
        } else {
            Timer::from_duration(render_start)
        };

        let token = self
            .loop_handle
            .insert_source(timer, move |_time, _, state| {
                if let Err(err) = state.redraw(estimated_presentation) {
                    let name = state.output.name();
                    warn!(?name, "Failed to submit rendering: {:?}", err);
                    state.queue_redraw(true);
                }
                TimeoutAction::Drop
            })
            .expect("Failed to schedule render");

        match &self.state {
            QueueState::Idle => {
                self.state = QueueState::Queued(token);
            }
            QueueState::WaitingForEstimatedVBlank(estimated_vblank) => {
                self.state = QueueState::WaitingForEstimatedVBlankAndQueued {
                    estimated_vblank: *estimated_vblank,
                    queued_render: token,
                };
            }
            QueueState::Queued(old_token) if force => {
                self.loop_handle.remove(*old_token);
                self.state = QueueState::Queued(token);
            }
            QueueState::WaitingForEstimatedVBlankAndQueued {
                estimated_vblank,
                queued_render,
            } if force => {
                self.loop_handle.remove(*queued_render);
                self.state = QueueState::WaitingForEstimatedVBlankAndQueued {
                    estimated_vblank: *estimated_vblank,
                    queued_render: token,
                };
            }
            _ => unreachable!(),
        }
    }

    #[profiling::function]
    fn redraw(&mut self, estimated_presentation: Duration) -> Result<()> {
        let Some(compositor) = self.compositor.as_mut() else {
            return Ok(());
        };

        let render_node = render_node_for_output(
            self.mirroring.as_ref().unwrap_or(&self.output),
            self.primary_node
                .read()
                .unwrap()
                .as_ref()
                .unwrap_or(&self.target_node),
            &self.target_node,
            &self.shell.read(),
        );

        let mut renderer = if render_node != self.target_node {
            self.api
                .renderer(&render_node, &self.target_node, compositor.format())
                .unwrap()
        } else {
            self.api.single_renderer(&self.target_node).unwrap()
        };

        self.timings.start_render(&self.clock);

        let mut additional_frame_flags = FrameFlags::empty();
        let mut remove_frame_flags = FrameFlags::empty();

        let (has_active_fullscreen, fullscreen_drives_refresh_rate, animations_going) = {
            let shell = self.shell.read();
            let animations_going = shell.animations_going();
            let output = self.mirroring.as_ref().unwrap_or(&self.output);
            if let Some((_, workspace)) = shell.workspaces.active(output) {
                if let Some(fullscreen_surface) = workspace.get_fullscreen() {
                    const _30_FPS: Duration = Duration::from_nanos(1_000_000_000 / 30);
                    (
                        true,
                        fullscreen_surface.wl_surface().is_some_and(|surface| {
                            recursive_frame_time_estimation(&self.clock, &surface)
                                .is_some_and(|dur| dur <= _30_FPS)
                        }),
                        animations_going,
                    )
                } else {
                    (false, false, animations_going)
                }
            } else {
                (false, false, animations_going)
            }
        };

        if has_active_fullscreen || animations_going {
            // skip overlay plane assign if we have a fullscreen surface or dynamic contents to save on tests
            remove_frame_flags |= FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT;
        }

        // Direct scanout (both primary and overlay planes) is disabled
        // entirely when HDR is on, until per-surface color encoding
        // negotiation lands via wp_color_management_v1 +
        // wp_color_representation_v1 (Phase 3).
        //
        // Two distinct issues each break direct scanout in HDR today:
        //
        //   1. Overlay-plane direct scanout — kernel composites planes in
        //      pre-DEGAMMA space with mixed per-plane encodings; result fed
        //      into DEGAMMA isn't uniformly sRGB, so DEGAMMA decodes garbage.
        //      Smithay's per-frame overlay-plane assignment heuristic flips
        //      surfaces between overlay and primary frame-to-frame, causing
        //      region-localized flicker on multi-window desktop.
        //      (project_hdr_overlay_plane_bug.md)
        //
        //   2. Primary-plane direct scanout — the no-offscreen render path
        //      has alpha/blend semantics that disagree with the offscreen
        //      path. Opaque clients (Firefox) render correctly, but clients
        //      with alpha < 1.0 (panels, popups, drop-shadows) appear
        //      translucent / z-fight against windows beneath during motion.
        //      Toggling VRR forces a full atomic commit() that re-stages
        //      plane properties and "fixes" it briefly, confirming plane-
        //      level state drift between commits in the no-offscreen path.
        //      Tried Xbgr2101010/Xrgb2101010 alpha-less primary fb formats —
        //      did not fix.
        //
        // Both share a root cause: cosmic-comp/smithay don't know each
        // plane's color encoding contract. Phase 3 protocols give us per-
        // surface encoding metadata, which lets us reject non-matching
        // surfaces from scanout or normalize them. Until then, force the
        // offscreen pass for any HDR rendering — costs us one full-screen
        // blit per frame, but the visual is correct on every kind of
        // window.
        let hdr_shader_needed = self.hdr_enabled;
        let shader_load_bearing = !self.screen_filter.is_noop() || hdr_shader_needed;
        if hdr_shader_needed {
            remove_frame_flags |= FrameFlags::ALLOW_SCANOUT;
            remove_frame_flags |= FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT;
        }

        let mut vrr = matches!(self.vrr_mode, AdaptiveSync::Force);

        if self.vrr_mode == AdaptiveSync::Enabled {
            vrr = has_active_fullscreen;
        }

        // Path B — install the HDR render context so every
        // ClippedSurfaceRenderElement constructed during output_elements
        // automatically picks up the linearize transform (sRGB→linear→BT.2020
        // matrix→ref_white scale). Cleared at the end of render_frame.
        if self.hdr_enabled {
            crate::backend::render::clipped_surface::set_render_hdr_context(
                true,
                self.hdr_ref_white as u32,
            );
        }

        let mut elements = output_elements(
            Some(&render_node),
            &mut renderer,
            &self.shell,
            self.clock.now(),
            self.mirroring.as_ref().unwrap_or(&self.output),
            CursorMode::All,
            #[cfg(not(feature = "debug"))]
            None,
            #[cfg(feature = "debug")]
            Some((&self.egui, &self.timings)),
        )
        .map_err(|err| {
            crate::backend::render::clipped_surface::clear_render_hdr_context();
            anyhow::format_err!("Failed to accumulate elements for rendering: {:?}", err)
        })?;

        if vrr && fullscreen_drives_refresh_rate && !self.timings.past_min_render_time(&self.clock)
        {
            additional_frame_flags |= FrameFlags::SKIP_CURSOR_ONLY_UPDATES;
        };
        self.timings.set_vrr(vrr);
        self.timings.elements_done(&self.clock);

        // we can't use the elements after `compositor.render_frame`,
        // so let's collect everything we need for screencopy now
        let mut has_cursor_mode_none = false;
        let frames = if self.mirroring.is_none() {
            take_screencopy_frames(&self.output, &elements, &mut has_cursor_mode_none)
        } else {
            Default::default()
        };

        // actual rendering
        // We need the offscreen postprocess pipeline whenever the shader is
        // doing real work — screen filter, software HDR PQ encode (color_mode=5),
        // HDR test pattern (color_mode=6), or HDR tuner sat/gamma on top of the
        // hardware path (color_mode=8). When the hardware color pipeline is
        // active and the tuner is at neutral with no test pattern, the shader
        // is pure passthrough (color_mode=7) so we can skip offscreen entirely
        // and let smithay scan the primary plane out directly — the CRTC
        // degamma/CTM/gamma LUTs do all the HDR work on the panel side.
        let needs_offscreen = shader_load_bearing;
        // Log transitions only (once per state change) — helps confirm both
        // that self.hdr_enabled is actually `true` here when output_config
        // says so AND whether we're scanning out directly vs going through
        // the postprocess offscreen pass.
        {
            use std::sync::atomic::{AtomicU8, Ordering};
            static LAST_STATE: AtomicU8 = AtomicU8::new(0xFF);
            let bits = (self.hdr_enabled as u8)
                | ((needs_offscreen as u8) << 1)
                | ((!self.screen_filter.is_noop() as u8) << 2)
                | ((self.hdr_hardware_path_active as u8) << 3)
                | ((self.hdr_test_pattern as u8) << 4);
            if LAST_STATE.swap(bits, Ordering::Relaxed) != bits {
                trace!(
                    hdr_enabled = self.hdr_enabled,
                    hw_path = self.hdr_hardware_path_active,
                    test_pattern = self.hdr_test_pattern,
                    screen_filter_active = !self.screen_filter.is_noop(),
                    needs_offscreen,
                    scanout_allowed = !needs_offscreen,
                    "[HDR] render-loop state transition"
                );
            }
        }
        let source_output = self
            .mirroring
            .as_ref()
            .or(needs_offscreen.then_some(&self.output))
            .filter(|output| {
                PostprocessOutputConfig::for_output_untransformed(output)
                    != PostprocessOutputConfig::for_output(&self.output)
                    || needs_offscreen
            });

        let mut pre_postprocess_data = PrePostprocessData::default();

        let res = if let Some(source_output) = source_output {
            let offscreen_output_config =
                PostprocessOutputConfig::for_output_untransformed(source_output);
            // Path B (COSMIC_HDR_PATH_B=1) composites in linear floating-point.
            // The offscreen FB needs RGBA16F to hold linear HDR values without
            // crushing the highlight headroom. SDR-only paths and the
            // hardware-CRTC HDR path (Phase 2A.1) keep the output FB format
            // (Abgr2101010 in HDR, Argb8888 in SDR) which is what they
            // semantically expect.
            let offscreen_format = if self.hdr_enabled
                && crate::backend::render::clipped_surface::path_b_enabled()
            {
                Fourcc::Abgr16161616f
            } else {
                compositor.format()
            };
            let postprocess_state = match self.postprocess_textures.entry(self.target_node) {
                hash_map::Entry::Occupied(occupied) => {
                    let postprocess_state = occupied.into_mut();
                    // If output config OR format differs, re-create.
                    if postprocess_state.output_config != offscreen_output_config
                        || postprocess_state
                            .texture
                            .format()
                            .is_some_and(|f| f != offscreen_format)
                    {
                        *postprocess_state = PostprocessState::new_with_renderer(
                            &mut renderer,
                            offscreen_format,
                            offscreen_output_config,
                        )?
                    }
                    postprocess_state
                }
                hash_map::Entry::Vacant(vacant) => {
                    vacant.insert(PostprocessState::new_with_renderer(
                        &mut renderer,
                        offscreen_format,
                        offscreen_output_config,
                    )?)
                }
            };

            if has_cursor_mode_none && self.mirroring.is_none() {
                // TODO: use `extract_if` once stablized
                let cursor_element_count = elements
                    .iter()
                    .take_while(|elem| elem.kind() == Kind::Cursor)
                    .count();
                let cursor_elements = elements.drain(..cursor_element_count).collect::<Vec<_>>();
                let scale = source_output.current_scale().fractional_scale().into();

                let geometry: Option<Rectangle<i32, Physical>> =
                    cursor_elements.iter().fold(None, |acc, elem| {
                        let geometry = elem.geometry(scale);
                        if let Some(acc) = acc {
                            Some(acc.merge(geometry))
                        } else {
                            Some(geometry)
                        }
                    });

                if let Some(geometry) = geometry {
                    let cursor_elements = cursor_elements
                        .into_iter()
                        .map(|elem| {
                            RelocateRenderElement::from_element(
                                elem,
                                Point::from((-geometry.loc.x, -geometry.loc.y)),
                                Relocate::Relative,
                            )
                        })
                        .collect::<Vec<_>>();

                    postprocess_state.track_cursor(
                        &mut renderer,
                        Fourcc::Abgr8888,
                        geometry.size,
                        scale,
                    )?;

                    postprocess_state
                        .cursor_texture
                        .as_mut()
                        .unwrap()
                        .render()
                        .draw::<_, <GlMultiRenderer as RendererSuper>::Error>(|tex| {
                            if self.mirroring.is_none() {
                                pre_postprocess_data.cursor_geometry = Some(geometry);
                                pre_postprocess_data.cursor_texture = Some(tex.clone());
                            }

                            let mut fb = renderer.bind(tex)?;
                            let res = match postprocess_state
                                .cursor_damage_tracker
                                .as_mut()
                                .unwrap()
                                .render_output(
                                    &mut renderer,
                                    &mut fb,
                                    1,
                                    &cursor_elements,
                                    [0.0, 0.0, 0.0, 0.0],
                                ) {
                                Ok(res) => res,
                                Err(RenderError::Rendering(err)) => return Err(err),
                                Err(RenderError::OutputNoMode(_)) => unreachable!(),
                            };

                            if self.mirroring.is_none() {
                                pre_postprocess_data.states = Some(res.states);
                            }

                            renderer.wait(&res.sync)?;
                            std::mem::drop(fb);

                            let transform = source_output.current_transform();
                            let area = tex.size().to_logical(1, transform);

                            Ok(res
                                .damage
                                .cloned()
                                .map(|v| {
                                    v.into_iter()
                                        .map(|r| r.to_logical(1).to_buffer(1, transform, &area))
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default())
                        })
                        .context("Failed to draw to offscreen render target")?;
                }
            } else {
                postprocess_state.remove_cursor();
            }

            postprocess_state
                .texture
                .render()
                .draw::<_, <GlMultiRenderer as RendererSuper>::Error>(|tex| {
                    if self.mirroring.is_none() {
                        pre_postprocess_data.texture = Some(tex.clone());
                    }

                    let mut fb = renderer.bind(tex)?;
                    let res = match postprocess_state.damage_tracker.render_output(
                        &mut renderer,
                        &mut fb,
                        1,
                        &elements,
                        CLEAR_COLOR,
                    ) {
                        Ok(res) => res,
                        Err(RenderError::Rendering(err)) => return Err(err),
                        Err(RenderError::OutputNoMode(_)) => unreachable!(),
                    };

                    if self.mirroring.is_none() {
                        if let Some(states) = pre_postprocess_data.states.as_mut() {
                            states.states.extend(res.states.states);
                        } else {
                            pre_postprocess_data.states = Some(res.states);
                        }
                    }

                    renderer.wait(&res.sync)?;
                    std::mem::drop(fb);

                    let transform = source_output.current_transform();
                    let area = tex.size().to_logical(1, transform);

                    Ok(res
                        .damage
                        .cloned()
                        .map(|v| {
                            v.into_iter()
                                .map(|r| r.to_logical(1).to_buffer(1, transform, &area))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default())
                })
                .context("Failed to draw to offscreen render target")?;

            renderer = self.api.single_renderer(&self.target_node).unwrap();

            elements = postprocess_elements(
                &mut renderer,
                &self.output,
                &pre_postprocess_data,
                postprocess_state,
                &self.screen_filter,
                self.hdr_enabled,
                self.hdr_colorspace_for_shader,
                self.hdr_ref_white,
                self.hdr_gamut_mix,
                self.hdr_saturation,
                self.hdr_midtone_gamma,
                self.hdr_test_pattern,
                self.hdr_hardware_path_active,
            );

            if let Err(err) = compositor.with_compositor(|c| c.use_vrr(vrr)) {
                warn!("Unable to set adaptive VRR state: {}", err);
            }
            // [HDR-FRAME-CALL] Log what we're handing to compositor.render_frame
            // — element count + flags. Throttled 1/sec.
            {
                use std::sync::atomic::{AtomicU64, Ordering};
                static LAST: AtomicU64 = AtomicU64::new(0);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                if LAST.load(Ordering::Relaxed) != now {
                    let final_flags = self.frame_flags
                        .union(additional_frame_flags)
                        .difference(remove_frame_flags);
                    trace!(
                        "[HDR-FRAME-CALL] render_frame: hdr_enabled={} elements.len()={} flags={:?} (1/sec)",
                        self.hdr_enabled, elements.len(), final_flags
                    );
                    LAST.store(now, Ordering::Relaxed);
                }
            }
            compositor.render_frame(
                &mut renderer,
                &elements,
                [0.0, 0.0, 0.0, 0.0],
                self.frame_flags
                    .union(additional_frame_flags)
                    .difference(remove_frame_flags),
            )
        } else {
            if let Err(err) = compositor.with_compositor(|c| c.use_vrr(vrr)) {
                warn!("Unable to set adaptive VRR state: {}", err);
            }
            compositor.render_frame(
                &mut renderer,
                &elements,
                CLEAR_COLOR, // TODO use a theme neutral color
                self.frame_flags
                    .union(additional_frame_flags)
                    .difference(remove_frame_flags),
            )
        };
        // Path B — clear the HDR render context now that rendering is done.
        crate::backend::render::clipped_surface::clear_render_hdr_context();
        self.timings.draw_done(&self.clock);

        match res {
            Ok(frame_result) => {
                let (tx, rx) = std::sync::mpsc::channel();

                let feedback = if !frame_result.is_empty && self.mirroring.is_none() {
                    Some((
                        self.shell
                            .read()
                            .take_presentation_feedback(&self.output, &frame_result.states),
                        rx,
                        estimated_presentation,
                    ))
                } else {
                    None
                };

                if frame_result.needs_sync()
                    && let PrimaryPlaneElement::Swapchain(elem) = &frame_result.primary_element
                {
                    elem.sync.wait()?;
                }

                match compositor.queue_frame(feedback) {
                    x @ Ok(()) | x @ Err(FrameError::EmptyFrame) => {
                        self.timings.submitted_for_presentation(&self.clock);

                        // Update `state` after `queue_frame`, before any early return from errors
                        if x.is_ok() {
                            let new_state = QueueState::WaitingForVBlank {
                                redraw_needed: false,
                            };
                            match mem::replace(&mut self.state, new_state) {
                                QueueState::Idle => unreachable!(),
                                QueueState::Queued(_) => (),
                                QueueState::WaitingForVBlank { .. } => unreachable!(),
                                QueueState::WaitingForEstimatedVBlank(estimated_vblank)
                                | QueueState::WaitingForEstimatedVBlankAndQueued {
                                    estimated_vblank,
                                    ..
                                } => {
                                    self.loop_handle.remove(estimated_vblank);
                                }
                            };
                        }

                        let now = self.clock.now();
                        for (session, frame, res) in frames {
                            if let Err(err) = send_screencopy_result(
                                &mut renderer,
                                &self.output,
                                &mut pre_postprocess_data,
                                &tx,
                                &frame_result,
                                &elements,
                                (&session, frame, res),
                                now.into(),
                                // Path B tone-down needs ref_white to undo
                                // the per-surface scaling. 0 disables the
                                // shader path (for hardware HDR + SDR).
                                if self.hdr_enabled
                                    && crate::backend::render::clipped_surface::path_b_enabled()
                                {
                                    self.hdr_ref_white
                                } else {
                                    0.0
                                },
                            ) {
                                tracing::warn!(?err, "Failed to screencopy");
                            }
                        }

                        if self.mirroring.is_none() {
                            // If postprocessing, use states from first render
                            let states = pre_postprocess_data.states.unwrap_or(frame_result.states);
                            self.send_dmabuf_feedback(states);
                        }

                        if x.is_ok() {
                            if self.mirroring.is_none() {
                                self.frame_callback_seq = self.frame_callback_seq.wrapping_add(1);
                                self.send_frame_callbacks();
                            }
                        } else {
                            // Atomic commit failed — kernel rejected the new state
                            // (most often: HDR connector props + framebuffer-format
                            // combination it doesn't accept, or VRR feature mismatch).
                            //
                            // Critical: still dispatch frame callbacks even on
                            // commit failure. Wayland clients block their next
                            // submit waiting for the previous frame's "presented"
                            // signal; if commits keep failing and we never fire
                            // callbacks, every non-overlay client (i.e. everything
                            // except direct-scanout apps like Firefox) freezes
                            // permanently. We rely on the queued estimated vblank
                            // below as the primary callback path, but in case
                            // that timer never fires (timer races, callbacks
                            // gated downstream), also dispatch synchronously here
                            // as a backstop so apps can keep submitting frames
                            // — they might display sometimes, that's fine, the
                            // alternative is everything looking frozen.
                            if self.mirroring.is_none() {
                                self.frame_callback_seq =
                                    self.frame_callback_seq.wrapping_add(1);
                                self.send_frame_callbacks();
                            }

                            // we don't expect a vblank
                            let _ = self.vblank_frame.take();

                            self.queue_estimated_vblank(
                                estimated_presentation,
                                // Make sure we redraw to reevaluate, if we intentionally missed content
                                additional_frame_flags
                                    .contains(FrameFlags::SKIP_CURSOR_ONLY_UPDATES),
                            );
                        }
                    }
                    Err(err) => {
                        for (_session, frame, _) in frames {
                            frame.fail(CaptureFailureReason::Unknown);
                        }
                        return Err(err).with_context(|| "Failed to submit result for display");
                    }
                };
            }
            Err(err) => {
                compositor.reset_buffers();
                anyhow::bail!("Rendering failed: {}", err);
            }
        }

        for device in self.api.devices_mut()? {
            device.renderer_mut().cleanup_texture_cache()?;
        }

        Ok(())
    }

    fn queue_estimated_vblank(&mut self, target_presentation_time: Duration, force: bool) {
        match mem::take(&mut self.state) {
            QueueState::Idle => unreachable!(),
            QueueState::Queued(_) => (),
            QueueState::WaitingForVBlank { .. } => unreachable!(),
            QueueState::WaitingForEstimatedVBlank(token)
            | QueueState::WaitingForEstimatedVBlankAndQueued {
                estimated_vblank: token,
                ..
            } => {
                self.state = QueueState::WaitingForEstimatedVBlank(token);
                return;
            }
        }

        let now = self.clock.now();
        let mut duration = target_presentation_time.saturating_sub(now.into());

        // No use setting a zero timer, since we'll send frame callbacks anyway right after the call to
        // render(). This can happen for example with unknown presentation time from DRM.
        if duration.is_zero() {
            duration += self.timings.refresh_interval();
        }

        trace!("queueing estimated vblank timer to fire in {duration:?}");

        let timer = Timer::from_duration(duration);
        let token = self
            .loop_handle
            .insert_source(timer, move |_, _, data| {
                data.on_estimated_vblank(force);
                TimeoutAction::Drop
            })
            .unwrap();
        self.state = QueueState::WaitingForEstimatedVBlank(token);
    }

    fn update_mirroring(&mut self, mirroring_output: Option<Output>) {
        self.mirroring = mirroring_output;
        self.postprocess_textures.clear();
    }

    fn update_screen_filter(&mut self, filter_config: ScreenFilter) {
        self.screen_filter = filter_config;
        self.postprocess_textures.clear();
    }

    fn send_frame_callbacks(&mut self) {
        if self.mirroring.is_none() {
            let _ = self
                .thread_sender
                .send(SurfaceCommand::SendFrames(self.frame_callback_seq));
        }
    }

    fn send_dmabuf_feedback(&mut self, states: RenderElementStates) {
        let _ = self
            .thread_sender
            .send(SurfaceCommand::RenderStates(states));
    }
}

fn source_node_for_surface(w: &WlSurface) -> Option<DrmNode> {
    with_renderer_surface_state(w, |state| {
        state
            .buffer()
            .and_then(|buffer| get_dmabuf(buffer).ok().and_then(|dmabuf| dmabuf.node()))
    })
    .flatten()
}

// TODO: Introduce can_shared_dmabuf_framebuffer for cases where we might select another gpu
//  and composite on target if not possible to finally get rid of "primary"
#[profiling::function]
fn render_node_for_output(
    output: &Output,
    primary_node: &DrmNode,
    target_node: &DrmNode,
    shell: &Shell,
) -> DrmNode {
    if target_node == primary_node {
        return *target_node;
    }

    let Some(workspace) = shell.active_space(output) else {
        return *target_node;
    };
    let nodes = workspace
        .get_fullscreen()
        .map(|w| vec![w.clone()])
        .unwrap_or_else(|| {
            workspace
                .mapped()
                .map(|mapped| mapped.active_window())
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flat_map(|w| w.wl_surface().and_then(|s| source_node_for_surface(&s)))
        .collect::<Vec<_>>();

    if nodes.contains(target_node) || nodes.is_empty() {
        *target_node
    } else {
        *primary_node
    }
}

fn get_surface_dmabuf_feedback(
    render_node: DrmNode,
    target_node: DrmNode,
    render_formats: FormatSet,
    _target_formats: FormatSet,
    primary_plane_formats: FormatSet,
    overlay_plane_formats: Option<FormatSet>,
) -> SurfaceDmabufFeedback {
    // We limit the scan-out trache to formats we can also render from
    // so that there is always a fallback render path available in case
    // the supplied buffer can not be scanned out directly

    let primary_plane_formats = primary_plane_formats
        .intersection(&render_formats)
        .cloned()
        .collect::<FormatSet>();
    let overlay_plane_formats = overlay_plane_formats.map(|formats| {
        formats
            .intersection(&render_formats)
            .cloned()
            .collect::<FormatSet>()
    });
    let builder = DmabufFeedbackBuilder::new(render_node.dev_id(), render_formats);

    /*
    // Sadly no implementation would pick this up as a preferred render tranche,
    // where the combined formats would increase our chances of doing a dmabuf copy.
    // .. So we should probably not advertise this on the off-chance it actually triggers bugs.
    //

    let combined_formats = render_formats.intersection(&target_formats).cloned().collect::<FormatSet>();
    if target_node != render_node.dev_id() && !combined_formats.is_empty() {
        builder = builder.add_preference_tranche(
            render_node.dev_id(),
            None,
            combined_formats,
        );
    };

    // We also can't advertise scan out tranches for the actual display device,
    // as e.g. the nvidia driver might then send us dmabufs, that makes e.g. the iris hangs on import...
    if target_node != render_node.dev_id() && !combined_formats.is_empty() {
        builder = builder.add_preference_tranche(
            target_node.dev_id(),
            Some(zwp_linux_dmabuf_feedback_v1::TrancheFlags::Scanout),
            combined_formats,
        );
    };

    // So no fun combinations, we gotta wait for dmabuf-v6
    */

    let render_feedback = builder.clone().build().unwrap();
    let primary_scanout_feedback = (target_node == render_node).then(|| {
        builder
            .clone()
            .add_preference_tranche(
                render_node.dev_id(),
                Some(zwp_linux_dmabuf_feedback_v1::TrancheFlags::Scanout),
                primary_plane_formats,
            )
            .build()
            .unwrap()
    });
    let overlay_scanout_feedback = overlay_plane_formats
        .filter(|_| target_node == render_node)
        .map(|formats| {
            builder
                .add_preference_tranche(
                    render_node.dev_id(),
                    Some(zwp_linux_dmabuf_feedback_v1::TrancheFlags::Scanout),
                    formats,
                )
                .build()
                .unwrap()
        });

    SurfaceDmabufFeedback {
        render_feedback,
        overlay_scanout_feedback,
        primary_scanout_feedback,
    }
}

fn take_screencopy_frames(
    output: &Output,
    elements: &[CosmicElement<GlMultiRenderer>],
    has_cursor_mode_none: &mut bool,
) -> Vec<(
    ScreencopySessionRef,
    ScreencopyFrame,
    Result<(Option<Vec<Rectangle<i32, Physical>>>, RenderElementStates), OutputNoMode>,
)> {
    output
        .take_pending_frames()
        .into_iter()
        .map(|(session, frame)| {
            let additional_damage = frame.damage();
            let session_data = session.user_data().get::<SessionData>().unwrap();
            let mut damage_tracking = session_data.lock().unwrap();

            let buffer = frame.buffer();
            let age = if matches!(buffer_type(&buffer), Some(BufferType::Shm)) {
                // TODO re-use offscreen buffer to damage track screencopy to shm
                0
            } else {
                1
            };

            if !additional_damage.is_empty() {
                let area = output
                    .current_mode()
                    .unwrap()
                    /* TODO: Mode is Buffer..., why is this Physical in the first place */
                    .size
                    .to_logical(1)
                    .to_buffer(1, Transform::Normal)
                    .to_f64();

                let additional_damage_elements: Vec<_> = additional_damage
                    .into_iter()
                    .map(|rect| {
                        rect.to_f64()
                            .to_logical(
                                output.current_scale().fractional_scale(),
                                output.current_transform(),
                                &area,
                            )
                            .to_i32_round()
                    })
                    .map(DamageElement::new)
                    .collect();
                let _ = damage_tracking
                    .dt
                    .damage_output(age, &additional_damage_elements);
            };

            let res = damage_tracking.dt.damage_output(age, elements);

            if !session.draw_cursor() {
                *has_cursor_mode_none = true;
            }

            let res = res.map(|(a, b)| (a.cloned(), b));
            std::mem::drop(damage_tracking);
            (session, frame, res)
        })
        .collect()
}

fn send_screencopy_result<'a>(
    renderer: &mut GlMultiRenderer<'a>,
    output: &Output,
    pre_postprocess_data: &mut PrePostprocessData,
    tx: &std::sync::mpsc::Sender<PendingImageCopyData>,
    frame_result: &RenderFrameResult<GbmBuffer, GbmFramebuffer, CosmicElement<GlMultiRenderer<'a>>>,
    elements: &[CosmicElement<GlMultiRenderer>],
    (session, frame, res): (
        &ScreencopySessionRef,
        ScreencopyFrame,
        Result<(Option<Vec<Rectangle<i32, Physical>>>, RenderElementStates), OutputNoMode>,
    ),
    presentation_time: Duration,
    // Path B: the per-surface linearize stage in clipped_surface.frag bakes
    // `ref_white_nits / 10000.0` into the offscreen RGBA16F texture. We need
    // the inverse of this scale to tone-down to sRGB for screencopy clients.
    // Pass the live ref_white_nits value — 0 if not in Path B.
    hdr_ref_white_for_path_b: f32,
) -> Result<()> {
    let (damage, _) = res?;

    let mut sync = SyncPoint::default();
    let mut dmabuf_clone;
    let mut render_buffer;
    let buffer = frame.buffer();
    let mut shm_buffer = false;
    let buffer_size = buffer_dimensions(&buffer).ok_or(RenderError::<
        <GlMultiRenderer as RendererSuper>::Error,
    >::Rendering(
        MultiError::ImportFailed
    ))?;
    let mut fb = if let Ok(dmabuf) = get_dmabuf(&buffer) {
        dmabuf_clone = dmabuf.clone();
        Some(
            renderer
                .bind(&mut dmabuf_clone)
                .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?,
        )
    } else {
        shm_buffer = true;
        let format = with_buffer_contents(&buffer, |_, _, data| shm_format_to_fourcc(data.format))
            .map_err(|_| OutputNoMode)? // eh, we have to do some error
            .expect("We should be able to convert all hardcoded shm screencopy formats");

        if pre_postprocess_data
            .texture
            .as_ref()
            .is_some_and(|tex| tex.format() == Some(format))
            && (!session.draw_cursor() || pre_postprocess_data.cursor_texture.is_none())
        {
            None
        } else {
            render_buffer =
                Offscreen::<GlesRenderbuffer>::create_buffer(renderer, format, buffer_size)
                    .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?;
            Some(
                renderer
                    .bind(&mut render_buffer)
                    .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?,
            )
        }
    };

    if let Some(ref damage) = damage {
        let (output_size, output_scale, output_transform) = (
            output.current_mode().ok_or(OutputNoMode)?.size,
            output.current_scale().fractional_scale(),
            output.current_transform(),
        );

        let filter = (!session.draw_cursor())
            .then(|| {
                elements.iter().filter_map(|elem| {
                    if let CosmicElement::Cursor(_) = elem {
                        Some(elem.id().clone())
                    } else {
                        None
                    }
                })
            })
            .into_iter()
            .flatten();

        // If the screen is rotated, we must convert damage to match output.
        let adjusted = damage
            .iter()
            .copied()
            .map(|rect| {
                let logical = rect.to_logical(1);
                logical
                    .to_buffer(
                        1,
                        output_transform.invert(),
                        &buffer_size.to_logical(1, output_transform),
                    )
                    .to_logical(1, Transform::Normal, &buffer_size)
                    .to_physical(1)
            })
            .collect::<Vec<_>>();

        if let Some(tex) = pre_postprocess_data.texture.as_mut() {
            // Path B: the pre-postprocess texture is RGBA16F linear-BT.2020
            // with `ref_white_nits / 10000.0` scaling baked in. Raw blit to
            // sRGB ARGB8888 produces a washed/broken screenshot — the bytes
            // would be reinterpreted as sRGB. Instead, render the texture
            // through screencopy_sdr_shader which inverts the linearize
            // pipeline: undo ref_white scale → BT.2020→BT.709 matrix →
            // linear→sRGB encode.
            // GATE OFF: the path_b_tonedown render pass causes the live
            // desktop to render washed, even though it only writes to the
            // screencopy client's framebuffer and not to the main offscreen.
            // Hypothesis: the per-frame screencopy from cosmic-shell's
            // workspace-overview / app-library thumbnails triggers this
            // path constantly, and the renderer state (sampler bindings,
            // texture program, sync fences) leaks back into the next
            // postprocess pass. Reverting to the raw blit (which produces
            // broken screenshots, the pre-0dc0cb50 behavior) is the lesser
            // evil until the leak is identified. Set
            // COSMIC_HDR_PATH_B_TONEDOWN=1 to re-enable for testing.
            let path_b_tonedown = hdr_ref_white_for_path_b > 0.0
                && matches!(tex.format(), Some(Fourcc::Abgr16161616f))
                && std::env::var("COSMIC_HDR_PATH_B_TONEDOWN").is_ok();

            if path_b_tonedown {
                if let Some(fb) = fb.as_mut() {
                    let shader = renderer
                        .glow_renderer_mut()
                        .egl_context()
                        .user_data()
                        .get::<ScreencopySdrShader>()
                        .expect(
                            "ScreencopySdrShader should be installed by init_shaders",
                        )
                        .0
                        .clone();
                    let ref_white_scale = hdr_ref_white_for_path_b / 10000.0;
                    let tex_size_buffer = tex
                        .size()
                        .to_logical(1, Transform::Normal)
                        .to_buffer(1, Transform::Normal)
                        .to_f64();
                    let src_rect = Rectangle::new(Point::from((0., 0.)), tex_size_buffer);
                    let dst_rect = Rectangle::from_size(output_size);
                    let tex_clone = tex.clone();
                    let mut frame =
                        renderer.render(fb, output_size, output_transform).map_err(
                            RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                        )?;
                    BorrowMut::<smithay::backend::renderer::gles::GlesFrame>::borrow_mut(
                        <GlMultiRenderer as AsGlowRenderer>::glow_frame_mut(&mut frame),
                    )
                    .override_default_tex_program(
                        shader,
                        vec![Uniform::new("ref_white_scale", ref_white_scale)],
                    );
                    frame
                        .as_mut()
                        .render_texture_from_to(
                            &tex_clone,
                            src_rect,
                            dst_rect,
                            &adjusted,
                            &[dst_rect],
                            Transform::Normal,
                            1.0,
                        )
                        .map_err(GlMultiError::Render)
                        .map_err(
                            RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                        )?;
                    BorrowMut::<smithay::backend::renderer::gles::GlesFrame>::borrow_mut(
                        <GlMultiRenderer as AsGlowRenderer>::glow_frame_mut(&mut frame),
                    )
                    .clear_tex_program_override();
                    sync = frame.finish().map_err(
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                    )?;
                    renderer.wait(&sync).map_err(
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                    )?;
                }
                // Cursor compositing on top of tone-downed buffer.
                if let Some(fb) = fb.as_mut() {
                    if let Some(cursor_geometry) = pre_postprocess_data
                        .cursor_geometry
                        .as_ref()
                        .filter(|_| session.draw_cursor())
                    {
                        let cursor_damage = adjusted
                            .iter()
                            .filter_map(|rect| cursor_geometry.intersection(*rect))
                            .map(|rect| {
                                Rectangle::new(rect.loc - cursor_geometry.loc, rect.size)
                            })
                            .collect::<Vec<_>>();
                        let mut frame = renderer
                            .render(fb, output_size, output_transform)
                            .map_err(
                                RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                            )?;
                        frame
                            .as_mut()
                            .render_texture_from_to(
                                pre_postprocess_data.cursor_texture.as_ref().unwrap(),
                                Rectangle::new(
                                    Point::from((0., 0.)),
                                    cursor_geometry
                                        .size
                                        .to_logical(1)
                                        .to_buffer(1, Transform::Normal)
                                        .to_f64(),
                                ),
                                *cursor_geometry,
                                &cursor_damage,
                                &[*cursor_geometry],
                                Transform::Normal,
                                1.0,
                            )
                            .map_err(GlMultiError::Render)
                            .map_err(
                                RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                            )?;
                        let sync2 = frame.finish().map_err(
                            RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                        )?;
                        renderer.wait(&sync2).map_err(
                            RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                        )?;
                    }
                }
            } else {
            let tex_fb = renderer
                .bind(tex)
                .map_err(RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering)?;

            if let Some(fb) = fb.as_mut() {
                for rect in adjusted.iter().copied() {
                    // TODO: On Vulkan, may need to combine sync points instead of just using latest?
                    sync = renderer
                        .blit(&tex_fb, fb, rect, rect, TextureFilter::Linear)
                        .map_err(
                            RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                        )?;
                }
                if let Some(cursor_geometry) = pre_postprocess_data
                    .cursor_geometry
                    .as_ref()
                    .filter(|_| session.draw_cursor())
                {
                    let cursor_damage = adjusted
                        .iter()
                        .filter_map(|rect| cursor_geometry.intersection(*rect))
                        .map(|rect| Rectangle::new(rect.loc - cursor_geometry.loc, rect.size))
                        .collect::<Vec<_>>();
                    let mut frame = renderer.render(fb, output_size, output_transform).map_err(
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                    )?;
                    frame
                        .as_mut()
                        .render_texture_from_to(
                            pre_postprocess_data.cursor_texture.as_ref().unwrap(),
                            Rectangle::new(
                                Point::from((0., 0.)),
                                cursor_geometry
                                    .size
                                    .to_logical(1)
                                    .to_buffer(1, Transform::Normal)
                                    .to_f64(),
                            ),
                            *cursor_geometry,
                            &cursor_damage,
                            &[*cursor_geometry],
                            Transform::Normal,
                            1.0,
                        )
                        .map_err(GlMultiError::Render)
                        .map_err(
                            RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                        )?;
                    let sync = frame.finish().map_err(
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                    )?;
                    renderer.wait(&sync).map_err(
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering,
                    )?;
                }
            } else {
                fb = Some(tex_fb);
            }
            } // end of `if path_b_tonedown { ... } else { ... }`
        } else {
            sync = frame_result
                .blit_frame_result(
                    output_size,
                    output_transform,
                    output_scale,
                    renderer,
                    fb.as_mut().unwrap(),
                    adjusted,
                    filter,
                )
                .map_err(|err| match err {
                    BlitFrameResultError::Rendering(err) => {
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering(err)
                    }
                    BlitFrameResultError::Export(_) => {
                        RenderError::<<GlMultiRenderer as RendererSuper>::Error>::Rendering(
                            MultiError::DeviceMissing,
                        )
                    }
                })?;
        };
    }

    let transform = output.current_transform();

    if let Some(data) = submit_buffer(
        frame,
        renderer,
        shm_buffer.then_some(fb.as_mut().unwrap()),
        transform,
        damage.as_deref(),
        sync,
    )? {
        if frame_result.is_empty {
            data.frame
                .success(transform, data.damage, presentation_time);
        } else {
            let _ = tx.send(data);
        }
    }

    Ok(())
}

fn postprocess_elements<'a>(
    renderer: &mut GlMultiRenderer<'a>,
    output: &Output,
    pre_postprocess_data: &PrePostprocessData,
    postprocess_state: &PostprocessState,
    screen_filter: &ScreenFilter,
    hdr_enabled: bool,
    hdr_colorspace_for_shader: f32,
    hdr_ref_white: f32,
    hdr_gamut_mix: f32,
    hdr_saturation: f32,
    hdr_midtone_gamma: f32,
    hdr_test_pattern: bool,
    hdr_hardware_path_active: bool,
) -> Vec<CosmicElement<GlMultiRenderer<'a>>> {
    // color_mode selection:
    //   0   = SDR no filter (existing)
    //   1-4 = SDR screen filter (greyscale + daltonization, existing)
    //   5   = software HDR fallback (no hardware color pipeline available;
    //         shader does sRGB→linear→matrix→ref_white→PQ end-to-end)
    //   6   = HDR test pattern (calibration grid generated in shader)
    //   7   = hardware HDR with neutral tuner (CRTC pipeline does encode;
    //         shader is identity passthrough)
    //   8   = hardware HDR with non-neutral sat/gamma slider (shader applies
    //         the sat/gamma curves in linear space then sRGB-re-encodes;
    //         hardware then does DEGAMMA → CTM → GAMMA_LUT)
    //
    // Sat / gamma intentionally live in the shader (uniform updates apply
    // on next render frame — free) rather than in the CTM / GAMMA_LUT
    // blobs (which would need an atomic commit() per slider tick to push
    // to the kernel; Intel xe rejects per-frame full commits under motion
    // → moving-window glitches).
    //
    // Hardware path's CTM is `gamut + ref_white` only (saturation in the
    // CTM math is set to 1.0 / identity at staging time). GAMMA_LUT is
    // plain PQ encode (no gamma curve baked in). Shader picks up where
    // those leave off via color_mode=8.
    let tuner_at_neutral = (hdr_saturation - 1.0).abs() < 0.01
        && (hdr_midtone_gamma - 1.0).abs() < 0.01;
    let color_mode_value: f32 = if hdr_enabled {
        if hdr_test_pattern {
            6.0
        } else if hdr_hardware_path_active {
            if tuner_at_neutral { 7.0 } else { 8.0 }
        } else {
            5.0
        }
    } else {
        screen_filter
            .color_filter
            .map(|val| val as u8 as f32)
            .unwrap_or(0.)
    };
    // Path B activation flag for the MAIN offscreen element. When Path B is on
    // (`hdr_enabled && path_b_enabled()`), the offscreen FB is RGBA16F holding
    // linear BT.2020 ref_white-scaled values — so this uniform MUST be 1.0 so
    // the postprocess shader skips the redundant sRGB→linear→matrix→ref_white
    // pre-stages (already applied per-surface in clipped_surface.frag) and
    // goes straight to sat/gamma + PQ encode.
    //
    // ROOT-CAUSE FIX for the "PrintScreen permanently washes the desktop" bug:
    // previously the main element omitted the path_b_active uniform entirely.
    // GL uniforms are program state, not per-draw — they retain whatever was
    // last set. The cursor element below pushes path_b_active=0.0 when present,
    // so once the cursor screencopy path triggered (the moment a screen-capture
    // session arrived with draw_cursor=false, i.e. PrintScreen), the main
    // element inherited 0.0 and the shader re-decoded the already-linear pixels
    // as sRGB → washed live output. The 0.0 stuck for every subsequent frame
    // because nothing reset it, so the desktop stayed washed until reboot.
    let path_b_main_active: f32 =
        if hdr_enabled && crate::backend::render::clipped_surface::path_b_enabled() {
            1.0
        } else {
            0.0
        };
    // Log color_mode transitions only — once per change rather than per frame —
    // so we can confirm in journal whether the PQ shader is actually engaging.
    {
        use std::sync::atomic::{AtomicU32, Ordering};
        static LAST_LOGGED: AtomicU32 = AtomicU32::new(u32::MAX);
        // Hash inputs into u32 so we log only on change (color_mode + ref_white + mix + cs).
        let key: u32 = ((color_mode_value * 10.0) as u32)
            ^ ((hdr_ref_white as u32).wrapping_mul(7919))
            ^ ((hdr_gamut_mix * 1000.0) as u32).wrapping_mul(31)
            ^ ((hdr_colorspace_for_shader * 10.0) as u32).wrapping_mul(127)
            ^ ((hdr_test_pattern as u32).wrapping_mul(2003));
        if LAST_LOGGED.swap(key, Ordering::Relaxed) != key {
            trace!(
                "[HDR] postprocess_elements building element list: color_mode={:.1} hdr_enabled={} cs={:.1} ref_w={:.1} mix={:.2} test={}",
                color_mode_value,
                hdr_enabled,
                hdr_colorspace_for_shader,
                hdr_ref_white,
                hdr_gamut_mix,
                hdr_test_pattern,
            );
        }
    }
    let postprocess_texture_shader = Borrow::<GlesRenderer>::borrow(renderer.as_ref())
        .egl_context()
        .user_data()
        .get::<PostprocessShader>()
        .expect("OffscreenShader should be available through `init_shaders`");

    let mut elements: [Option<TextureShaderElement>; 2] = [None, None];
    if let Some(cursor_texture) = postprocess_state.cursor_texture.as_ref() {
        let cursor_geometry = pre_postprocess_data.cursor_geometry.unwrap();
        let texture_elem = TextureRenderElement::from_texture_render_buffer(
            cursor_geometry.loc.to_f64(),
            cursor_texture,
            None,
            Some(Rectangle::new(
                Point::from((0., 0.)),
                cursor_geometry.size.to_logical(1).to_f64(),
            )),
            Some(
                cursor_geometry
                    .size
                    .to_f64()
                    .to_logical(output.current_scale().fractional_scale())
                    .to_i32_round(),
            ),
            Kind::Cursor,
        );

        elements[0] = Some(TextureShaderElement::new(
            texture_elem,
            postprocess_texture_shader.0.clone(),
            vec![
                Uniform::new("invert", if screen_filter.inverted { 1. } else { 0. }),
                Uniform::new("color_mode", color_mode_value),
                Uniform::new("hdr_colorspace", hdr_colorspace_for_shader),
                Uniform::new("hdr_ref_white", hdr_ref_white),
                Uniform::new("hdr_gamut_mix", hdr_gamut_mix),
                Uniform::new("hdr_saturation", hdr_saturation),
                Uniform::new("hdr_midtone_gamma", hdr_midtone_gamma),
                // Cursor texture is rendered as sRGB Argb8888 (it doesn't go
                // through the per-surface Path B linearize stage). Force
                // path_b_active=0.0 here so the postprocess shader applies
                // its full sRGB→linear→matrix→ref_white→PQ pipeline to the
                // cursor pixels, matching what they were before Path B.
                Uniform::new("path_b_active", 0.0_f32),
            ],
        ));
    }

    let texture_elem = TextureRenderElement::from_texture_render_buffer(
        (0., 0.),
        &postprocess_state.texture,
        None,
        Some(Rectangle::new(
            Point::from((0., 0.)),
            postprocess_state.output_config.size.to_logical(1).to_f64(),
        )),
        Some(
            postprocess_state
                .output_config
                .size
                .to_f64()
                .to_logical(postprocess_state.output_config.fractional_scale)
                .to_i32_round(),
        ),
        Kind::Unspecified,
    );
    elements[1] = Some(TextureShaderElement::new(
        texture_elem,
        postprocess_texture_shader.0.clone(),
        vec![
            Uniform::new("invert", if screen_filter.inverted { 1. } else { 0. }),
            Uniform::new("color_mode", color_mode_value),
            Uniform::new("hdr_colorspace", hdr_colorspace_for_shader),
            Uniform::new("hdr_ref_white", hdr_ref_white),
            Uniform::new("hdr_gamut_mix", hdr_gamut_mix),
            Uniform::new("hdr_saturation", hdr_saturation),
            Uniform::new("hdr_midtone_gamma", hdr_midtone_gamma),
            // Path B: 1.0 → main offscreen is linear RGBA16F, skip the shader's
            // sRGB-decode + matrix + ref_white pre-stages. 0.0 → offscreen is
            // sRGB, do the full pipeline (matches old behavior in non-Path-B).
            // MUST be set explicitly here even when 0.0 — otherwise the program
            // inherits whatever the cursor element last bound (also 0.0 but
            // unrelated reason), and worse, in Path B mode it'd inherit 0.0 too
            // which silently double-decodes the offscreen → washed output.
            Uniform::new("path_b_active", path_b_main_active),
        ],
    ));

    constrain_render_elements(
        elements.into_iter().flatten(),
        (0, 0),
        Rectangle::from_size(
            output
                .geometry()
                .size
                .as_logical()
                .to_physical_precise_round(output.current_scale().fractional_scale()),
        ),
        Rectangle::new(Point::from((0, 0)), postprocess_state.output_config.size),
        ConstrainScaleBehavior::Fit,
        ConstrainAlign::CENTER,
        postprocess_state.output_config.fractional_scale,
    )
    .map(CosmicElement::<GlMultiRenderer>::Postprocess)
    .collect::<Vec<_>>()
}
