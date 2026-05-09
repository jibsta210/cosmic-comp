// SPDX-License-Identifier: GPL-3.0-only

use anyhow::{Context, Result, anyhow};
use libdisplay_info::{edid::DisplayDescriptorTag, info::Info};
use smithay::{
    reexports::drm::control::{
        AtomicCommitFlags, Device as ControlDevice, Mode, ModeFlags, PlaneType, ResourceHandle,
        atomic::AtomicModeReq,
        connector::{self, State as ConnectorState},
        crtc,
        dumbbuffer::DumbBuffer,
        property,
    },
    utils::Transform,
};
use std::{collections::HashMap, ops::Range};

pub fn display_configuration(
    device: &mut impl ControlDevice,
    supports_atomic: bool,
) -> Result<HashMap<connector::Handle, Option<crtc::Handle>>> {
    let res_handles = device.resource_handles()?;
    let connectors = res_handles.connectors();

    let mut map = HashMap::new();
    let mut cleanup = Vec::new();

    // We expect the previous running drm master (likely the login mananger)
    // to leave the drm device in a sensible state.
    // That means, to reduce flickering, we try to keep an established mapping.
    for conn in connectors
        .iter()
        .flat_map(|conn| device.get_connector(*conn, true).ok())
    {
        if let Some(enc) = conn.current_encoder()
            && let Some(crtc) = device.get_encoder(enc)?.crtc()
        {
            // If is is connected we found a mapping
            if conn.state() == ConnectorState::Connected {
                map.insert(conn.handle(), Some(crtc));
            // If not, the user just unplugged something,
            // or the drm master did not cleanup?
            // Well, I guess we cleanup after them.
            } else {
                cleanup.push(crtc);
            }
        }
    }

    // But just in case we try to match all remaining connectors.
    for conn in connectors
        .iter()
        .flat_map(|conn| device.get_connector(*conn, false).ok())
        .filter(|conn| conn.state() == ConnectorState::Connected)
        .filter(|conn| !map.contains_key(&conn.handle()))
        .collect::<Vec<_>>()
        .iter()
    {
        'outer: for encoder_info in conn
            .encoders()
            .iter()
            .flat_map(|encoder_handle| device.get_encoder(*encoder_handle))
        {
            for crtc in res_handles.filter_crtcs(encoder_info.possible_crtcs()) {
                if !map.values().any(|v| *v == Some(crtc)) {
                    map.insert(conn.handle(), Some(crtc));
                    break 'outer;
                }
            }
        }

        map.entry(conn.handle()).or_insert(None);
    }

    // And then cleanup
    if supports_atomic {
        let mut req = AtomicModeReq::new();
        let plane_handles = device.plane_handles()?;

        for conn in connectors
            .iter()
            .flat_map(|conn| device.get_connector(*conn, false).ok())
            .filter(|conn| {
                if let Some(enc) = conn.current_encoder()
                    && let Ok(enc) = device.get_encoder(enc)
                    && let Some(crtc) = enc.crtc()
                {
                    return cleanup.contains(&crtc);
                }
                false
            })
            .map(|info| info.handle())
        {
            let crtc_id = get_prop(device, conn, "CRTC_ID")?;
            req.add_property(conn, crtc_id, property::Value::CRTC(None));
        }

        // We cannot just shortcut and use the legacy api for all cleanups because of this.
        // (Technically a device does not need to be atomic for planes to be used, but nobody does this otherwise.)
        for plane in plane_handles {
            let info = device.get_plane(plane)?;
            if let Some(crtc) = info.crtc() {
                let is_primary = get_property_val(device, plane, "type").map(
                    |(val_type, val)| match val_type.convert_value(val) {
                        property::Value::Enum(Some(val)) => {
                            val.value() == PlaneType::Primary as u64
                        }
                        _ => false,
                    },
                )?;
                if cleanup.contains(&crtc) || !is_primary {
                    let crtc_id = get_prop(device, plane, "CRTC_ID")?;
                    let fb_id = get_prop(device, plane, "FB_ID")?;
                    req.add_property(plane, crtc_id, property::Value::CRTC(None));
                    req.add_property(plane, fb_id, property::Value::Framebuffer(None));
                }
            }
        }

        for crtc in cleanup {
            let mode_id = get_prop(device, crtc, "MODE_ID")?;
            let active = get_prop(device, crtc, "ACTIVE")?;
            req.add_property(crtc, active, property::Value::Boolean(false));
            req.add_property(crtc, mode_id, property::Value::Unknown(0));
        }

        device.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, req)?;
    } else {
        for crtc in res_handles.crtcs() {
            #[allow(deprecated)]
            let _ = device.set_cursor(*crtc, Option::<&DumbBuffer>::None);
        }
        for crtc in cleanup {
            // null commit (necessary to trigger removal on the kernel side with the legacy api.)
            let _ = device.set_crtc(crtc, None, (0, 0), &[], None);
        }
    }

    Ok(map)
}

pub fn interface_name(device: &impl ControlDevice, connector: connector::Handle) -> Result<String> {
    let conn_info = device.get_connector(connector, false)?;

    let other_short_name;
    let interface_short_name = match conn_info.interface() {
        connector::Interface::DVII => "DVI-I",
        connector::Interface::DVID => "DVI-D",
        connector::Interface::DVIA => "DVI-A",
        connector::Interface::SVideo => "S-VIDEO",
        connector::Interface::DisplayPort => "DP",
        connector::Interface::HDMIA => "HDMI-A",
        connector::Interface::HDMIB => "HDMI-B",
        connector::Interface::EmbeddedDisplayPort => "eDP",
        other => {
            other_short_name = format!("{:?}", other);
            &other_short_name
        }
    };

    Ok(format!(
        "{}-{}",
        interface_short_name,
        conn_info.interface_id()
    ))
}

pub fn edid_info(device: &impl ControlDevice, connector: connector::Handle) -> Result<Info> {
    let edid_prop = get_prop(device, connector, "EDID")?;
    let edid_info = device.get_property(edid_prop)?;

    let mut edid = None;
    let props = device.get_properties(connector)?;
    let (ids, vals) = props.as_props_and_values();
    for (&id, &val) in ids.iter().zip(vals.iter()) {
        if id == edid_prop {
            if let property::Value::Blob(edid_blob) = edid_info.value_type().convert_value(val) {
                let blob = device.get_property_blob(edid_blob)?;
                edid = Some(Info::parse_edid(&blob).context("Unable to parse edid")?);
            }
            break;
        }
    }

    edid.ok_or(anyhow!("No EDID found"))
}

pub fn get_prop(
    device: &impl ControlDevice,
    handle: impl ResourceHandle,
    name: &str,
) -> Result<property::Handle> {
    let props = device.get_properties(handle)?;
    let (prop_handles, _) = props.as_props_and_values();
    for prop in prop_handles {
        let info = device.get_property(*prop)?;
        if Some(name) == info.name().to_str().ok() {
            return Ok(*prop);
        }
    }
    anyhow::bail!("No prop found for {}", name)
}

pub fn get_property_val(
    device: &impl ControlDevice,
    handle: impl ResourceHandle,
    name: &str,
) -> Result<(property::ValueType, property::RawValue)> {
    let props = device.get_properties(handle)?;
    let (prop_handles, values) = props.as_props_and_values();
    for (&prop, &val) in prop_handles.iter().zip(values.iter()) {
        let info = device.get_property(prop)?;
        if Some(name) == info.name().to_str().ok() {
            let val_type = info.value_type();
            return Ok((val_type, val));
        }
    }
    anyhow::bail!("No prop found for {}", name)
}

// Returns refresh rate in milliherz
pub fn calculate_refresh_rate(mode: Mode) -> u32 {
    let htotal = mode.hsync().2 as u32;
    let vtotal = mode.vsync().2 as u32;
    let mut refresh =
        (mode.clock() as u64 * 1000000_u64 / htotal as u64 + vtotal as u64 / 2) / vtotal as u64;

    if mode.flags().contains(ModeFlags::INTERLACE) {
        refresh *= 2;
    }
    if mode.flags().contains(ModeFlags::DBLSCAN) {
        refresh /= 2;
    }
    if mode.vscan() > 1 {
        refresh /= mode.vscan() as u64;
    }

    refresh as u32
}

pub fn get_minimum_refresh_rate(
    device: &impl ControlDevice,
    connector: connector::Handle,
) -> Result<Option<u32>> {
    let info = edid_info(device, connector)?;
    let edid = info.edid().context("EDID lacking into")?;
    for descriptor in edid.display_descriptors() {
        if descriptor.tag() == DisplayDescriptorTag::RangeLimits {
            return Ok(Some(
                descriptor
                    .range_limits()
                    .context("Invalid range limits descriptor")?
                    .min_vert_rate_hz as u32,
            ));
        }
    }

    Ok(None)
}

pub fn get_max_bpc(
    dev: &impl ControlDevice,
    conn: connector::Handle,
) -> Result<Option<(u32, Range<u32>)>> {
    let Some(handle) = get_prop(dev, conn, "max bpc").ok() else {
        return Ok(None);
    };

    let info = dev.get_property(handle)?;
    let range = match info.value_type() {
        property::ValueType::UnsignedRange(x, y) => (x as u32)..(y as u32),
        _ => return Err(anyhow!("max bpc has wrong value type")),
    };

    let value = get_property_val(dev, conn, "max bpc").map(|(val_type, val)| {
        match val_type.convert_value(val) {
            property::Value::UnsignedRange(res) => res as u32,
            _ => unreachable!(),
        }
    })?;

    Ok(Some((value, range)))
}

/// Set the connector's `Broadcast RGB` enum property to **Full** (PC range,
/// 0-255). Critical for HDR signaling on panels where the kernel default
/// (`Automatic`) resolves to `Limited 16:235` (TV range), which compresses
/// PC-range PQ-encoded content into 16-235 of the panel's 0-255 — visibly
/// "washed out and lower contrast". Mutter sets this property unconditionally
/// during its HDR path; we should too.
///
/// On this Dell XPS 16 Tandem OLED via xe driver, the panel's reported
/// "Minimum SDR Luminance Full Coverage = 20 cd/m^2" is consistent with
/// 16/255 of the panel's max — i.e. it's currently in Limited range.
///
/// Returns Ok(()) silently if the connector doesn't expose this property
/// (some embedded panels don't). Errors only on a real DRM failure.
pub fn set_broadcast_rgb_full(dev: &impl ControlDevice, conn: connector::Handle) -> Result<()> {
    let prop_handle = match get_prop(dev, conn, "Broadcast RGB") {
        Ok(h) => h,
        Err(_) => return Ok(()), // property absent on this connector — no-op
    };
    let info = dev.get_property(prop_handle)?;
    let variants = match info.value_type() {
        property::ValueType::Enum(values) => values,
        _ => return Err(anyhow!("Broadcast RGB has wrong value type")),
    };
    // Look for the "Full" variant explicitly (value=1 on Intel xe, but we
    // resolve by name to be portable).
    let full_value = variants
        .values()
        .1
        .iter()
        .find(|v| v.name().to_str().ok() == Some("Full"))
        .map(|v| v.value())
        .ok_or_else(|| anyhow!("Broadcast RGB enum has no `Full` variant"))?;
    // set_property takes RawValue (u64). Enum prop values ARE u64 raw values
    // — we can pass directly without going through `property::Value::Enum`
    // (which borrows a `&'a EnumValue` and complicates lifetimes here).
    dev.set_property(conn, prop_handle, full_value)
        .map_err(Into::<anyhow::Error>::into)?;
    Ok(())
}

/// Diagnostic / belt-and-braces: also set the `Colorspace` enum property via
/// the legacy non-atomic `set_property` IOCTL. The atomic commit path SHOULD
/// be carrying this through smithay's `set_hdr_state`, but on this xe + Tandem
/// OLED combo modetest reads `Colorspace: value: 0 (Default)` after a successful
/// HDR setup, which means panel is in SDR mode and PQ-encoded content gets
/// interpreted as sRGB — exactly the washed-out symptom we're chasing. Writing
/// the property here doubles up: if atomic commits the prop fine, this is a
/// no-op; if atomic silently drops it, this puts the panel into the right
/// signaling mode anyway.
pub fn set_colorspace_legacy(
    dev: &impl ControlDevice,
    conn: connector::Handle,
    variant_name: &str,
) -> Result<()> {
    let prop_handle = match get_prop(dev, conn, "Colorspace") {
        Ok(h) => h,
        Err(_) => return Ok(()),
    };
    let info = dev.get_property(prop_handle)?;
    let variants = match info.value_type() {
        property::ValueType::Enum(values) => values,
        _ => return Err(anyhow!("Colorspace has wrong value type")),
    };
    let target_value = variants
        .values()
        .1
        .iter()
        .find(|v| v.name().to_str().ok() == Some(variant_name))
        .map(|v| v.value())
        .ok_or_else(|| anyhow!("Colorspace enum has no variant {variant_name:?}"))?;
    dev.set_property(conn, prop_handle, target_value)
        .map_err(Into::<anyhow::Error>::into)?;
    Ok(())
}

pub fn set_max_bpc(dev: &impl ControlDevice, conn: connector::Handle, bpc: u32) -> Result<u32> {
    let (_, range) =
        get_max_bpc(dev, conn)?.ok_or(anyhow!("max bpc does not exist for connector"))?;
    dev.set_property(
        conn,
        get_prop(dev, conn, "max bpc")?,
        property::Value::UnsignedRange(bpc.clamp(range.start, range.end) as u64).into(),
    )
    .map_err(Into::<anyhow::Error>::into)
    .and_then(|_| get_property_val(dev, conn, "max bpc"))
    .map(|(val_type, val)| match val_type.convert_value(val) {
        property::Value::UnsignedRange(val) => val as u32,
        _ => unreachable!(),
    })
}

// =====================================================================
// HDR signaling: Colorspace + HDR_OUTPUT_METADATA
// =====================================================================
//
// To put an HDR-capable display into HDR mode, the compositor needs to:
//   1. Set the connector's `Colorspace` enum property to a wide-gamut option
//      (we use BT2020_RGB; DCI-P3_RGB_D65 is also valid for some panels).
//   2. Write an `HDR_OUTPUT_METADATA` blob containing an HDMI HDR Static
//      Metadata Type 1 InfoFrame (CTA-861.3 / BT.2100 mastering metadata)
//      describing the source content's EOTF (PQ/HLG) and luminance range.
//   3. Ensure max bpc >= 10 so the PQ curve doesn't band visibly.
//
// Without (2), most panels stay in SDR mode regardless of (1). Without (1),
// the panel ignores (2). Both must be set in the same atomic commit (or a
// quick succession) for the panel to switch into HDR mode.
//
// Reference: include/uapi/linux/drm/drm_mode.h (struct hdr_output_metadata,
// struct hdr_metadata_infoframe), CTA-861.3, BT.2100.

/// Kernel UAPI struct for the `HDR_OUTPUT_METADATA` connector blob property.
/// Layout matches `struct hdr_output_metadata` in `drm_mode.h` byte-for-byte.
///
/// **Critical**: the kernel struct interleaves primaries as
/// `struct { u16 x, y; } display_primaries[3]` — in memory `r.x, r.y, g.x,
/// g.y, b.x, b.y`. Not separate `x[3]` / `y[3]` arrays! Earlier versions of
/// this code had separate arrays which produced a garbled layout and caused
/// atomic commits to fail (panel firmware rejected the bad InfoFrame and
/// rendering froze).
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct HdrOutputMetadata {
    /// 0 = HDR_OUTPUT_METADATA_TYPE1 (the only kind defined as of 6.x)
    metadata_type: u32,
    // ↓ struct hdr_metadata_infoframe begins here
    /// 0 = traditional SDR gamma, 1 = traditional HDR gamma,
    /// 2 = SMPTE ST 2084 (PQ), 3 = HLG.
    eotf: u8,
    /// Always 0 for "static metadata".
    static_metadata_type: u8,
    /// Display primaries in 0.00002 chromaticity units (50000 = 1.0).
    /// Layout: `[r_x, r_y, g_x, g_y, b_x, b_y]` — kernel reads as
    /// `struct { u16 x, y; } display_primaries[3]`.
    display_primaries: [u16; 6],
    /// Reference white point chromaticity in same units, layout `[x, y]`.
    white_point: [u16; 2],
    /// Mastering display peak luminance in cd/m^2 (nits).
    max_display_mastering_luminance: u16,
    /// Mastering display min luminance in 0.0001 cd/m^2 units
    /// (i.e. 1 = 0.0001 nits = a real OLED black; 5 = 0.0005 nits).
    min_display_mastering_luminance: u16,
    /// Maximum Content Light Level (peak brightest pixel) in cd/m^2.
    max_cll: u16,
    /// Maximum Frame Average Light Level (frame avg) in cd/m^2.
    max_fall: u16,
}

/// EOTF: SMPTE ST 2084 (PQ) — the modern HDR standard used by HDR10 etc.
const EOTF_PQ: u8 = 2;

// DCI-P3 D65 mastering display primaries in 0.00002 chromaticity units
// (50000 = 1.0). Layout: r_x, r_y, g_x, g_y, b_x, b_y — interleaved per the
// kernel's `display_primaries[3]` struct array.
//
// NOTE: signaling DCI-P3 (not BT.2020) because the actual panel hardware on
// Jake's laptop is a P3-gamut Tandem OLED. Tagging BT.2020 caused the panel
// to map values from BT.2020 → its native P3 hardware, crushing saturated
// colors as a side effect. Tagging DCI-P3 lets the panel decode directly
// without an extra gamut compression step. The kernel `Colorspace` property
// is set to `DCI-P3_RGB_D65` to match.
const DCI_P3_PRIMARIES: [u16; 6] = [
    34000, 16000, // R: 0.680, 0.320
    13250, 34500, // G: 0.265, 0.690
    7500, 3000, // B: 0.150, 0.060
];
// BT.2020 (ITU-R Rec.2020) primaries, same 0.00002 chroma units.
const BT2020_PRIMARIES: [u16; 6] = [
    35400, 14600, // R: 0.708, 0.292
    8500, 39850, // G: 0.170, 0.797
    6550, 2300, // B: 0.131, 0.046
];
// D65 reference white in same units, [x, y].
const D65_WHITE: [u16; 2] = [15635, 16450]; // 0.3127, 0.3290

/// Which wide-gamut container we're tagging the InfoFrame as. Drives both
/// `display_primaries` in the metadata blob and the `Colorspace` enum
/// variant we set on the connector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrColorContainer {
    Bt2020,
    DciP3,
}

impl HdrColorContainer {
    /// `Colorspace` enum-property variant name for this container.
    pub fn colorspace_variant(self) -> &'static str {
        match self {
            HdrColorContainer::Bt2020 => "BT2020_RGB",
            HdrColorContainer::DciP3 => "DCI-P3_RGB_D65",
        }
    }

    /// Display primaries (interleaved [r_x, r_y, g_x, g_y, b_x, b_y]).
    pub fn primaries(self) -> [u16; 6] {
        match self {
            HdrColorContainer::Bt2020 => BT2020_PRIMARIES,
            HdrColorContainer::DciP3 => DCI_P3_PRIMARIES,
        }
    }
}

/// Per-output mastering luminance bounds, in the units the kernel expects.
/// Derived from the panel's EDID HDR static metadata block (Phase 1.5);
/// for now callers can hand-pass values gleaned from `edid-decode`.
#[derive(Debug, Clone, Copy)]
pub struct HdrMasteringLuminance {
    /// Peak luminance in cd/m^2 (e.g. 525 for an OLED with 525 nit peak).
    pub max_lum_nits: u16,
    /// Min luminance in 0.0001 cd/m^2 units (e.g. 5 for OLED with 0.0005 nit blacks).
    pub min_lum_units: u16,
    /// MaxCLL — peak pixel cd/m^2. Usually equal to or less than max_lum_nits.
    pub max_cll_nits: u16,
    /// MaxFALL — frame-average peak cd/m^2. Usually less than max_cll.
    pub max_fall_nits: u16,
}

impl HdrMasteringLuminance {
    /// Conservative defaults for an HDR-capable OLED panel where the EDID
    /// block hasn't been parsed yet. Tuned for ~500 nit OLEDs (close to
    /// jake's panel; safe-ish for most laptop OLEDs).
    ///
    /// **Important distinction (CTA-861.3 / SMPTE 2086):**
    /// - `max_lum_nits` / `min_lum_units` describe the **mastering display**
    ///   used to grade the content (= the panel, since we're "mastering live").
    /// - `max_cll_nits` describes the **content's peak pixel** — for SDR
    ///   content scaled to ref_white, this should be ref_white (not panel max).
    /// - `max_fall_nits` describes the **content's frame-average** — desktop
    ///   content typically averages ~30% of peak.
    ///
    /// Telling the panel "content peaks at 500 nits" when our content actually
    /// peaks at ref_white=200 makes the panel firmware tone-map for "bright
    /// HDR content" — which on this Tandem OLED visibly suppresses the low-PQ
    /// desktop pixels (= the "dull desktop" symptom). Mutter sets target_max_cll
    /// dynamically per-output, matching content not panel.
    pub fn fallback_oled() -> Self {
        Self {
            max_lum_nits: 525,
            min_lum_units: 5, // 0.0005 nits
            max_cll_nits: 200,
            max_fall_nits: 80,
        }
    }

    /// Build content-aware metadata from the current SDR reference white. The
    /// panel firmware uses these fields to drive its internal tone-mapping; if
    /// they describe the actual content (rather than the panel) it can scale
    /// our 0..ref_white range across more of its display volume.
    pub fn for_sdr_content(ref_white_nits: u16) -> Self {
        Self {
            max_lum_nits: 525, // panel mastering capability
            min_lum_units: 5,
            max_cll_nits: ref_white_nits,
            max_fall_nits: ref_white_nits.saturating_mul(2) / 5, // ~40% avg, conservative for desktop
        }
    }
}

/// Build the raw bytes of an HDR_OUTPUT_METADATA blob describing PQ-encoded
/// HDR content with BT.2020 primaries and D65 white, mastered to the given
/// luminance bounds. Resulting buffer goes into `Device::create_property_blob`.
fn build_hdr_metadata_blob(
    lum: HdrMasteringLuminance,
    container: HdrColorContainer,
) -> Vec<u8> {
    let m = HdrOutputMetadata {
        metadata_type: 0, // HDR_OUTPUT_METADATA_TYPE1
        eotf: EOTF_PQ,
        static_metadata_type: 0,
        display_primaries: container.primaries(),
        white_point: D65_WHITE,
        max_display_mastering_luminance: lum.max_lum_nits,
        min_display_mastering_luminance: lum.min_lum_units,
        max_cll: lum.max_cll_nits,
        max_fall: lum.max_fall_nits,
    };
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            (&m as *const HdrOutputMetadata) as *const u8,
            std::mem::size_of::<HdrOutputMetadata>(),
        )
    };
    bytes.to_vec()
}

/// Look up the raw u64 value of a `Colorspace` enum variant by name on the
/// given connector. Used by the HDR setup path which hands the value to
/// `smithay::backend::drm::surface::HdrState::colorspace_value`.
pub fn colorspace_enum_value(
    dev: &impl ControlDevice,
    conn: connector::Handle,
    variant_name: &str,
) -> Result<u64> {
    let prop_handle = get_prop(dev, conn, "Colorspace")?;
    let info = dev.get_property(prop_handle)?;
    let variants = match info.value_type() {
        property::ValueType::Enum(values) => values,
        _ => return Err(anyhow!("Colorspace has wrong value type")),
    };
    // `EnumValues::values()` returns `(&[u64], &[EnumValue])`.
    variants
        .values()
        .1
        .iter()
        .find(|v| v.name().to_str().ok() == Some(variant_name))
        .map(|v| v.value())
        .ok_or_else(|| anyhow!("Colorspace enum has no variant {variant_name:?}"))
}

/// Returns true if the connector advertises the property surface required to
/// drive HDR signaling. We probe for `DCI-P3_RGB_D65` (the colorspace we
/// actually use for the Tandem OLED panels in scope here) — most HDR-capable
/// connectors expose this enum variant alongside BT2020_RGB.
pub fn connector_supports_hdr(dev: &impl ControlDevice, conn: connector::Handle) -> bool {
    if get_prop(dev, conn, "HDR_OUTPUT_METADATA").is_err() {
        return false;
    }
    colorspace_enum_value(dev, conn, "DCI-P3_RGB_D65").is_ok()
        || colorspace_enum_value(dev, conn, "BT2020_RGB").is_ok()
}

/// Build an `HDR_OUTPUT_METADATA` blob (PQ + BT.2020 + supplied luminance) and
/// return its raw u64 ID for use with smithay's `HdrState::metadata_blob_id`.
///
/// The kernel cleans the blob up automatically when the DRM master is dropped.
/// If callers want to free earlier (e.g. when toggling HDR off mid-session),
/// they can call `device.destroy_property_blob(id)`.
pub fn create_hdr_metadata_blob(
    dev: &impl ControlDevice,
    lum: HdrMasteringLuminance,
    container: HdrColorContainer,
) -> Result<u64> {
    let bytes = build_hdr_metadata_blob(lum, container);
    let blob = dev
        .create_property_blob(&bytes)
        .context("create HDR_OUTPUT_METADATA blob")?;
    Ok(blob.into())
}

// =====================================================================
// CRTC color pipeline (DEGAMMA_LUT, CTM, GAMMA_LUT) helpers
// =====================================================================
//
// These wrap the kernel KMS color-pipeline blob creation so the higher-
// level HDR setup can stage hardware encode (sRGB decode + gamut + ref-
// white scale + PQ encode) entirely in the display engine's fixed-
// function color blocks instead of doing it in the GLES postprocess
// shader on every frame.
//
// LUT generation lives in `crate::backend::render::hw_color_pipeline`
// (pure math, unit-tested). This file owns the kernel ABI side: probe
// LUT sizes from the CRTC, build blobs, return blob IDs for smithay's
// `HdrState`.

/// Read a CRTC's `DEGAMMA_LUT_SIZE` or `GAMMA_LUT_SIZE` (or any range
/// property by name). Returns `Some(size)` on success, `None` if the
/// property doesn't exist on this CRTC (older driver / hardware without
/// hardware color pipeline support — caller should fall back to shader).
pub fn get_crtc_range_prop(
    dev: &impl ControlDevice,
    crtc: crtc::Handle,
    name: &str,
) -> Option<u64> {
    let props = dev.get_properties(crtc).ok()?;
    let (handles, values) = props.as_props_and_values();
    for (handle, value) in handles.iter().zip(values.iter()) {
        let info = dev.get_property(*handle).ok()?;
        if info.name().to_str().ok() == Some(name) {
            return Some(*value);
        }
    }
    None
}

/// Convenience: returns the CRTC's `DEGAMMA_LUT_SIZE` if exposed.
/// On Intel xe + Tandem OLED this is 129; on AMD it varies; on hardware
/// without the property at all this is `None`.
pub fn crtc_degamma_lut_size(
    dev: &impl ControlDevice,
    crtc: crtc::Handle,
) -> Option<u32> {
    get_crtc_range_prop(dev, crtc, "DEGAMMA_LUT_SIZE").map(|v| v as u32)
}

/// Convenience: returns the CRTC's `GAMMA_LUT_SIZE`. Intel xe reports
/// 1024 here; older Intel typically 256.
pub fn crtc_gamma_lut_size(
    dev: &impl ControlDevice,
    crtc: crtc::Handle,
) -> Option<u32> {
    get_crtc_range_prop(dev, crtc, "GAMMA_LUT_SIZE").map(|v| v as u32)
}

/// Returns true if the CRTC has all three properties needed for the
/// hardware-accelerated HDR encode path: `DEGAMMA_LUT`, `CTM`, `GAMMA_LUT`.
/// The legacy CTM API (stable since ~Linux 4.6 on Intel/AMD) — newer
/// drivers may also expose a colorop API but we don't need it for our
/// encode pipeline.
pub fn crtc_has_color_pipeline(
    dev: &impl ControlDevice,
    crtc: crtc::Handle,
) -> bool {
    crtc_degamma_lut_size(dev, crtc).is_some_and(|s| s >= 2)
        && get_crtc_range_prop(dev, crtc, "GAMMA_LUT_SIZE").is_some_and(|s| s >= 2)
        && {
            // CTM is a blob, not a range — probe via property handle existence.
            let Ok(props) = dev.get_properties(crtc) else { return false };
            let (handles, _) = props.as_props_and_values();
            handles.iter().any(|h| {
                dev.get_property(*h)
                    .ok()
                    .and_then(|info| info.name().to_str().ok().map(|n| n == "CTM"))
                    .unwrap_or(false)
            })
        }
}

/// Create a kernel property blob from a `Vec<DrmColorLutEntry>`. The kernel
/// expects the bytes laid out as `struct drm_color_lut[]` — RGB+reserved
/// u16 quads, exactly what `DrmColorLutEntry` already is via `#[repr(C, packed)]`.
///
/// `create_property_blob` requires `T: Sized` so we copy into a heap-owned
/// `Vec<u8>` first (already-Sized via the Vec's pointer/length/capacity).
pub fn create_color_lut_blob(
    dev: &impl ControlDevice,
    entries: &[crate::backend::render::hw_color_pipeline::DrmColorLutEntry],
) -> Result<u64> {
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            entries.as_ptr() as *const u8,
            std::mem::size_of_val(entries),
        )
    }
    .to_vec();
    let blob = dev
        .create_property_blob(&bytes)
        .context("create color LUT blob")?;
    Ok(blob.into())
}

/// Create a kernel property blob from a CTM matrix encoded as 9 u64
/// sign-magnitude S31.32 values (kernel `struct drm_color_ctm`).
pub fn create_ctm_blob(dev: &impl ControlDevice, matrix: &[u64; 9]) -> Result<u64> {
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            matrix.as_ptr() as *const u8,
            std::mem::size_of::<[u64; 9]>(),
        )
    }
    .to_vec();
    let blob = dev
        .create_property_blob(&bytes)
        .context("create CTM blob")?;
    Ok(blob.into())
}

pub fn panel_orientation(dev: &impl ControlDevice, conn: connector::Handle) -> Result<Transform> {
    let (val_type, val) = get_property_val(dev, conn, "panel orientation")?;
    match val_type.convert_value(val) {
        property::Value::Enum(Some(val)) => match val.value() {
            // "Normal"
            0 => Ok(Transform::Normal),
            // "Upside Down"
            1 => Ok(Transform::_180),
            // "Left Side Up"
            2 => Ok(Transform::_90),
            // "Right Side Up"
            3 => Ok(Transform::_270),
            _ => Err(anyhow!("panel orientation has invalid value '{:?}'", val)),
        },
        _ => Err(anyhow!("panel orientation has wrong value type")),
    }
}
