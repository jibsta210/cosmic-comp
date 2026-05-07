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

// BT.2020 / BT.2100 mastering display primaries in 0.00002 chromaticity units
// (50000 = 1.0). Layout: r_x, r_y, g_x, g_y, b_x, b_y — interleaved per the
// kernel's `display_primaries[3]` struct array.
const BT2020_PRIMARIES: [u16; 6] = [
    35400, 14600, // R: 0.708, 0.292
    8500, 39850, // G: 0.170, 0.797
    6550, 2300, // B: 0.131, 0.046
];
// D65 reference white in same units, [x, y].
const D65_WHITE: [u16; 2] = [15635, 16450]; // 0.3127, 0.3290

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
    pub fn fallback_oled() -> Self {
        Self {
            max_lum_nits: 500,
            min_lum_units: 5, // 0.0005 nits
            max_cll_nits: 500,
            max_fall_nits: 400,
        }
    }
}

/// Build the raw bytes of an HDR_OUTPUT_METADATA blob describing PQ-encoded
/// HDR content with BT.2020 primaries and D65 white, mastered to the given
/// luminance bounds. Resulting buffer goes into `Device::create_property_blob`.
fn build_hdr_metadata_blob(lum: HdrMasteringLuminance) -> Vec<u8> {
    let m = HdrOutputMetadata {
        metadata_type: 0, // HDR_OUTPUT_METADATA_TYPE1
        eotf: EOTF_PQ,
        static_metadata_type: 0,
        display_primaries: BT2020_PRIMARIES,
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
/// drive HDR signaling (BT2020_RGB Colorspace variant + HDR_OUTPUT_METADATA).
pub fn connector_supports_hdr(dev: &impl ControlDevice, conn: connector::Handle) -> bool {
    if get_prop(dev, conn, "HDR_OUTPUT_METADATA").is_err() {
        return false;
    }
    colorspace_enum_value(dev, conn, "BT2020_RGB").is_ok()
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
) -> Result<u64> {
    let bytes = build_hdr_metadata_blob(lum);
    let blob = dev
        .create_property_blob(&bytes)
        .context("create HDR_OUTPUT_METADATA blob")?;
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
