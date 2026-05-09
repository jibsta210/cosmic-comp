// SPDX-License-Identifier: GPL-3.0-only

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::OpenOptions, path::Path};
use tracing::{error, warn};

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OutputState {
    #[serde(rename = "true")]
    Enabled,
    #[serde(rename = "false")]
    Disabled,
    Mirroring(String),
}

fn default_state() -> OutputState {
    OutputState::Enabled
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AdaptiveSync {
    #[serde(rename = "true")]
    Enabled,
    #[serde(rename = "false")]
    Disabled,
    Force,
}

fn default_sync() -> AdaptiveSync {
    AdaptiveSync::Enabled
}

/// Wide-gamut colorspace tag the compositor signals to the panel via the
/// connector `Colorspace` property and `HDR_OUTPUT_METADATA` primaries.
/// The shader-side gamut matrix (Rec.709 → target) is selected to match.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HdrColorspace {
    /// BT.2020 — broadest container, the ITU-R HDR standard. Panel firmware
    /// remaps internally to its native gamut. Most "compatible" choice.
    Bt2020,
    /// DCI-P3 D65 — closer to the Tandem OLED native gamut. Avoids one round
    /// of internal mapping at the cost of being a less-universal tag.
    DciP3,
}

impl Default for HdrColorspace {
    fn default() -> Self {
        HdrColorspace::Bt2020
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputsConfig {
    pub config: HashMap<Vec<OutputInfo>, Vec<OutputConfig>>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct OutputConfig {
    pub mode: ((i32, i32), Option<u32>),
    #[serde(default = "default_sync")]
    pub vrr: AdaptiveSync,
    pub scale: f64,
    pub transform: TransformDef,
    pub position: (u32, u32),
    #[serde(default = "default_state")]
    pub enabled: OutputState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bpc: Option<u32>,
    #[serde(default)]
    pub xwayland_primary: bool,
    /// HDR output mode for this connector. When `true`, on connectors that
    /// advertise HDR capabilities (`Colorspace` enum supporting `BT2020_RGB`
    /// + `HDR_OUTPUT_METADATA` blob property + EDID HDR static metadata block),
    /// cosmic-comp will:
    /// 1. Set the connector's `Colorspace` property to `BT2020_RGB`
    /// 2. Build and write an `HDR_OUTPUT_METADATA` blob (BT.2100 InfoFrame)
    ///    derived from the panel's EDID-reported peak/min luminance
    /// 3. Force `max_bpc >= 10` (HDR with 8-bit color is unwatchably banded)
    ///
    /// This signals the panel into HDR mode. SDR client content rendered in
    /// this mode will look dim/washed until SDR-to-HDR tone mapping (Phase 2)
    /// lands, but the panel itself will be in genuine HDR pipeline.
    ///
    /// Defaults to disabled (`None` → SDR Rec.709, identical to upstream).
    /// Only set on connectors known to be HDR-capable; the apply path bails
    /// safely on connectors that don't advertise the required properties.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_enabled: Option<bool>,
    /// Wide-gamut tag. `None` defaults to BT.2020 (standards-compliant).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_colorspace: Option<HdrColorspace>,
    /// SDR reference-white luminance in cd/m^2 (nits). BT.2408 says 203 for
    /// graded content; raised for desktop content on bright OLEDs (~300).
    /// Range we'll surface in UI: 80–500.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_reference_white: Option<u32>,
    /// Strength of the Rec.709 → target-gamut matrix in the shader,
    /// expressed as a percentage 0..=100. 0 = pass sRGB primaries through
    /// unchanged (trust panel firmware to remap). 100 = full conversion.
    /// Useful for exploring "washed out" failure modes by mixing the two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_gamut_strength: Option<u8>,
    /// Luminance-preserving saturation boost applied after the gamut matrix
    /// in the HDR shader, expressed as a percentage 50..=200 (100 = neutral
    /// / colorimetrically truthful, >100 = more vivid, <100 = more washed).
    /// Compensates for loss of vendor-applied SDR vibrance enhancements when
    /// switching to colorimetrically-strict HDR mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_saturation: Option<u8>,
    /// Midtone gamma applied in luminance space before PQ encode, as
    /// percentage 30..=150 (100 = neutral, <100 lifts midtones into HDR
    /// range, equivalent to Windows' AutoHDR brightness-lift). 70 ≈ default
    /// soft lift, 50 = aggressive. Solves the "SDR pixels look dim in HDR"
    /// problem by mapping cosmic UI midtones from ~50 nits to ~150-200 nits
    /// without desaturating (chroma-preserving via Y/Y_orig scaling).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_midtone_gamma: Option<u8>,
    /// When true, HDR output replaces normal content with a calibration
    /// test pattern (`color_mode=6.0` in the offscreen shader). Quadrants
    /// at known nits values + saturated primaries for eyeballing math.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hdr_test_pattern: Option<bool>,
}

impl Default for OutputConfig {
    fn default() -> OutputConfig {
        OutputConfig {
            mode: ((0, 0), None),
            vrr: AdaptiveSync::Enabled,
            scale: 1.0,
            transform: TransformDef::Normal,
            position: (0, 0),
            enabled: OutputState::Enabled,
            max_bpc: None,
            xwayland_primary: false,
            hdr_enabled: None,
            hdr_colorspace: None,
            hdr_reference_white: None,
            hdr_gamut_strength: None,
            hdr_saturation: None,
            hdr_midtone_gamma: None,
            hdr_test_pattern: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputInfo {
    pub connector: String,
    pub make: String,
    pub model: String,
}

pub fn load_outputs(path: Option<impl AsRef<Path>>) -> OutputsConfig {
    if let Some(path) = path.as_ref() {
        let path: &Path = path.as_ref();
        if path.exists() {
            match ron::de::from_reader::<_, OutputsConfig>(
                OpenOptions::new().read(true).open(path).unwrap(),
            ) {
                Ok(mut config) => {
                    for (info, config) in config.config.iter_mut() {
                        let config_clone = config.clone();
                        for conf in config.iter_mut() {
                            if let OutputState::Mirroring(conn) = &conf.enabled {
                                if let Some((j, _)) = info
                                    .iter()
                                    .enumerate()
                                    .find(|(_, info)| &info.connector == conn)
                                {
                                    if config_clone[j].enabled != OutputState::Enabled {
                                        warn!(
                                            "Invalid Mirroring tag, overriding with `Enabled` instead"
                                        );
                                        conf.enabled = OutputState::Enabled;
                                    }
                                } else {
                                    warn!(
                                        "Invalid Mirroring tag, overriding with `Enabled` instead"
                                    );
                                    conf.enabled = OutputState::Enabled;
                                }
                            }
                        }
                    }
                    return config;
                }
                Err(err) => {
                    warn!(?err, "Failed to read output_config, resetting..");
                    if let Err(err) = std::fs::remove_file(path) {
                        error!(?err, "Failed to remove output_config.");
                    }
                }
            };
        }
    }

    OutputsConfig {
        config: HashMap::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformDef {
    Normal,
    _90,
    _180,
    _270,
    Flipped,
    Flipped90,
    Flipped180,
    Flipped270,
}
