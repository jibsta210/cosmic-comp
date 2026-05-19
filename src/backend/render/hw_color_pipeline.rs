//! Hardware CRTC color pipeline LUT + CTM generation.
//!
//! Generates the three pieces of data the kernel KMS color pipeline needs to
//! do the static portion of our HDR encode (sRGB decode + Rec.709→BT.2020 or
//! DCI-P3 + ref-white scale + ST 2084 PQ encode) entirely in the display
//! engine's fixed-function color blocks, freeing the GPU shader from doing
//! it per frame.
//!
//! The pipeline order in hardware is:
//!
//! ```text
//! framebuffer (sRGB-encoded RGB) → DEGAMMA_LUT → CTM → GAMMA_LUT → scanout
//! ```
//!
//! Mapping that to our HDR encode:
//!
//! - **DEGAMMA_LUT**: piecewise sRGB → linear (replaces the shader's per-pixel
//!   sRGB-decode branch)
//! - **CTM**: 3×3 matrix that combines `Rec.709 → target_gamut` AND the
//!   `× (ref_white / 10000)` scale needed by PQ in one operation
//! - **GAMMA_LUT**: linear → PQ (inverse ST 2084 EOTF)
//!
//! All four user tuners now fold into the hardware CTM + GAMMA_LUT pair so
//! the shader can stay a pure passthrough whenever the hardware path is
//! active. That lets smithay scan out the primary plane directly even with
//! tuner sliders away from neutral, skipping the offscreen postprocess pass:
//!
//! - **Reference white**: scalar `ref_white / 10000` folded into the CTM.
//! - **Gamut strength**: linear interpolation between identity and the full
//!   `Rec.709 → target_gamut` matrix, folded into the CTM.
//! - **Saturation**: chroma-preserving 3×3 luma-mix matrix
//!   `S = sat·I + (1-sat)·1·wᵀ` (where `w` are the container's luma weights),
//!   composed into the CTM as `CTM = (ref_w/10000) · S · M_gamut_lerp` so
//!   gamut remap runs first, then saturation, then luminance scaling.
//! - **Midtone gamma**: per-channel power curve baked into the GAMMA_LUT
//!   entries: `LUT[i] = pq_encode((i/(N-1))^gamma_eff)`. Per-channel rather
//!   than per-luma (which is what the legacy shader path did) — the legacy
//!   GAMMA_LUT API only supports per-channel curves, but for desktop content
//!   the difference is perceptually small (slight hue stretch on saturated
//!   colors). The slider is dampened 0.5× from raw before baking in to match
//!   the dampening the shader path already applied for color_mode=8.
//!
//! Net effect: with HDR + hardware path, the GPU draws an identity-shaded
//! frame and the kernel KMS color blocks do every step of the HDR encode
//! including all four user-tunable knobs. SIGUSR1 live updates regenerate
//! both blobs (CTM for ref-white / gamut / saturation, GAMMA_LUT for
//! midtone gamma) and push them via smithay `set_hdr_state`.
//!
//! Everything here is `no_std`-friendly pure math. No smithay, no DRM, no
//! cosmic-comp surface state. Tested via unit tests; staging to the kernel
//! is the caller's job.

use crate::backend::kms::drm_helpers::HdrColorContainer;
// Re-exports to keep the LUT/CTM math testable without a full session.
//
// HdrColorContainer is in the kms backend module which is `pub(crate)` not
// `pub`; we live in the same crate so direct access works at compile time.

/// `drm_color_lut` entry in the kernel UAPI:
///
/// ```c
/// struct drm_color_lut {
///     __u16 red;
///     __u16 green;
///     __u16 blue;
///     __u16 reserved;  /* must be zero */
/// };
/// ```
///
/// Caller serializes a `Vec<DrmColorLutEntry>` via `bytemuck`/raw bytes
/// and passes to `drmModeCreatePropertyBlob`.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DrmColorLutEntry {
    pub red: u16,
    pub green: u16,
    pub blue: u16,
    pub reserved: u16,
}

/// Build a DEGAMMA_LUT mapping framebuffer sRGB-encoded values back to linear.
///
/// `size` is the value the kernel reports via `DEGAMMA_LUT_SIZE` (this CRTC
/// reports 129 on Intel xe + Tandem OLED, but the function is size-agnostic).
///
/// Entries are linearly spaced over the 0..1 normalized input domain — for
/// each entry `i`, `entry[i].rgb = sRGB_to_linear(i / (size - 1))` mapped to
/// the kernel's u16 fixed-point range `0..=0xFFFF`. The kernel hardware
/// interpolates linearly between adjacent entries based on the actual
/// framebuffer pixel value, so a curve as steep as sRGB's shadow toe still
/// resolves OK with 129 entries (worst-case error <0.5% of full-scale).
///
/// Uses the standard piecewise sRGB EOTF (IEC 61966-2-1):
///
/// ```text
/// linear(v) = v / 12.92                          if v ≤ 0.04045
///           = ((v + 0.055) / 1.055) ^ 2.4        otherwise
/// ```
pub fn srgb_decode_lut(size: u32) -> Vec<DrmColorLutEntry> {
    assert!(size >= 2, "LUT size must be at least 2");
    let n = size as usize;
    let mut lut = Vec::with_capacity(n);
    for i in 0..n {
        let v = i as f64 / (n - 1) as f64;
        let linear = if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        };
        let q = (linear.clamp(0.0, 1.0) * 65535.0).round() as u16;
        lut.push(DrmColorLutEntry {
            red: q,
            green: q,
            blue: q,
            reserved: 0,
        });
    }
    lut
}

/// Build a GAMMA_LUT that takes linear-light values to PQ (ST 2084) encoded
/// values, scaled so that the input range `0..=1.0` represents
/// `0..=10000 cd/m²` (the PQ standard reference range).
///
/// The CTM (built separately by `gamut_ctm_with_ref_white`) is responsible
/// for scaling `linear sRGB 1.0` down to `ref_white / 10000` so that the
/// LUT input never exceeds the panel's actual luminance capability.
///
/// `size` is what the kernel reports via `GAMMA_LUT_SIZE` (1024 on this
/// CRTC). Entries are linearly spaced; for SDR-derived content all the
/// useful range sits in the first ~5% of the LUT (since SDR white at
/// `ref_white = 525` only reaches `linear = 0.0525`), but the rest of the
/// LUT is still required because future HDR-aware content (HDR videos,
/// games delivering real PQ pixels) will use the upper portion.
///
/// Inverse ST 2084 PQ EOTF, from the spec:
///
/// ```text
/// V_PQ = ((c1 + c2 * Y^m1) / (1 + c3 * Y^m1)) ^ m2
///
/// m1 = 2610 / 16384         = 0.1593017578125
/// m2 = 2523 / 4096 * 128    = 78.84375
/// c1 = 3424 / 4096          = 0.8359375
/// c2 = 2413 / 4096 * 32     = 18.8515625
/// c3 = 2392 / 4096 * 32     = 18.6875
/// ```
pub fn pq_encode_lut(size: u32) -> Vec<DrmColorLutEntry> {
    pq_encode_lut_with_gamma(size, 1.0)
}

/// Build a GAMMA_LUT that takes linear-light values to PQ values, with an
/// extra per-channel midtone gamma curve baked in BEFORE the PQ encode:
/// `LUT[i] = pq_encode((i/(N-1))^gamma_eff)`.
///
/// `gamma_raw` is the raw user slider value (`1.0` = neutral). We apply the
/// same 0.5× dampening the legacy shader path used for `color_mode = 8`,
/// because the linear-domain gamma applied before PQ encoding is very
/// perceptually punchy and the slider felt twice as aggressive at full
/// strength. Slider 1.10 → effective gamma 1.05.
///
/// Per-channel rather than per-luma (the legacy shader path operated on
/// luminance and scaled RGB by the lift factor); GAMMA_LUT only supports
/// per-channel curves, but for desktop content the difference is small —
/// pure colors get the same effect, mixed colors get a slight hue stretch.
///
/// `gamma_raw = 1.0` returns a LUT identical to `pq_encode_lut(size)`.
pub fn pq_encode_lut_with_gamma(size: u32, gamma_raw: f32) -> Vec<DrmColorLutEntry> {
    assert!(size >= 2, "LUT size must be at least 2");
    let n = size as usize;

    const M1: f64 = 0.1593017578125;
    const M2: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;

    let pq = |y: f64| -> f64 {
        let y = y.max(0.0);
        let ym = y.powf(M1);
        let num = C1 + C2 * ym;
        let den = 1.0 + C3 * ym;
        (num / den).max(0.0).powf(M2)
    };

    let g = gamma_raw.max(0.1) as f64;
    let g_eff = 1.0 + (g - 1.0) * 0.5;

    let mut lut = Vec::with_capacity(n);
    for i in 0..n {
        let y_lin = i as f64 / (n - 1) as f64;
        let y_lifted = y_lin.max(0.0).powf(g_eff);
        let v = pq(y_lifted);
        let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
        lut.push(DrmColorLutEntry {
            red: q,
            green: q,
            blue: q,
            reserved: 0,
        });
    }
    lut
}

/// Build a GAMMA_LUT with a KWin-style modified-Reinhard tone-map curve
/// baked in, plus the optional midtone gamma, plus PQ encoding.
///
/// Maps absolute luminance through a shoulder-controlled Reinhard curve
/// before PQ encoding so SDR content sits at the user's reference white
/// and content above ref_white rolls smoothly toward the panel's peak
/// instead of clipping hard. Source: KDE KWin `src/opengl/colormanagement.glsl`
/// `doTonemapping` (modified Reinhard). KWin operates per-luma in ICtCp
/// space for chroma preservation; the GAMMA_LUT can only do per-channel,
/// so we approximate. The chroma shift for desktop content is small (the
/// curve is mostly linear in the SDR-and-below range), trades correctness
/// for a 1D LUT we can update in real time.
///
/// Curve math (from KWin):
///   inputRange  = source_peak / dest_ref       (sRGB content → 1.0)
///   outputRange = dest_peak   / dest_ref
///   v           = (outputRange * (1 + inputRange) - inputRange) / inputRange²
///   x' = x * (1 + x * v) / (1 + x)
///
/// Net behavior:
/// - At `x = 0`: slope is 1.0 (SDR linearity preserved through reference white)
/// - At `x = inputRange`: maps exactly to `outputRange` (source peak →
///   panel peak, no clipping)
/// - panel_peak < ref_white: curve COMPRESSES (HDR-into-SDR-headroom case;
///   e.g. user has ref_white=10000 and a 1500-nit panel → smooth roll-off)
/// - panel_peak > ref_white: curve EXPANDS (the usual case; SDR content
///   pushed up to panel peak)
/// - panel_peak == ref_white: identity (no tone-mapping needed)
///
/// `gamma_raw` is the same midtone-punch knob the gamma-only variant uses,
/// applied BEFORE the tone-map in linear space. With both at neutral
/// (gamma=1.0, ref_white==panel_peak), this LUT reduces to `pq_encode_lut`.
///
/// `panel_peak_nits = 0` skips tone-mapping entirely and behaves identically
/// to `pq_encode_lut_with_gamma`. Use that as a fallback when EDID mastering
/// metadata isn't available.
pub fn pq_encode_lut_with_tonemap(
    size: u32,
    gamma_raw: f32,
    ref_white_nits: u16,
    panel_peak_nits: u16,
) -> Vec<DrmColorLutEntry> {
    assert!(size >= 2, "LUT size must be at least 2");

    // Fall through to the no-tone-map LUT if we have nothing to map against
    // or the curve degenerates (panel can't be assumed brighter than 0).
    if panel_peak_nits == 0 || ref_white_nits == 0 {
        return pq_encode_lut_with_gamma(size, gamma_raw);
    }

    let n = size as usize;

    const M1: f64 = 0.1593017578125;
    const M2: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;

    let pq = |y: f64| -> f64 {
        let y = y.max(0.0);
        let ym = y.powf(M1);
        let num = C1 + C2 * ym;
        let den = 1.0 + C3 * ym;
        (num / den).max(0.0).powf(M2)
    };

    // Midtone gamma: same 0.5× dampening as the gamma-only variant so the
    // slider feels consistent regardless of which LUT builder is active.
    let g = gamma_raw.max(0.1) as f64;
    let g_eff = 1.0 + (g - 1.0) * 0.5;

    // KWin curve params. inputRange == 1 in our usage: we assume source
    // peak == ref_white (which is true for the desktop SDR case; HDR
    // clients with their own image_description will bypass this path
    // entirely once wp_color_management_v1 lands).
    let dest_ref = ref_white_nits as f64;
    let dest_peak = panel_peak_nits as f64;
    let input_range = 1.0_f64;
    let output_range = dest_peak / dest_ref;
    let v_curve = (output_range * (1.0 + input_range) - input_range) / (input_range * input_range);

    let mut lut = Vec::with_capacity(n);
    for i in 0..n {
        // LUT input x in [0,1] represents linear-light intensity normalized to
        // the user's reference white: x = 1.0 means "the brightness an sRGB
        // surface labeled `#ffffff` should reach on the panel" = ref_white nits.
        //
        // This requires the CTM to be IDENTITY in the luminance dimension
        // (i.e. CTM scale = 1.0, not ref_white/10000). The earlier design that
        // scaled the CTM by ref_white/10000 squeezed the LUT input range to
        // 0..(ref_white/10000) — for ref_white=200 that meant only ~21 out of
        // 1024 LUT entries carried the entire SDR range, and the other 1000+
        // mapped to the post-tone-map clamp. Result: heavy banding in the
        // bright midtones where the curve has its steepest slope. With CTM
        // scale=1.0 and the ref_white→nits map living entirely in this LUT,
        // all 1024 entries cover the SDR range and the curve resolution is
        // ~9× finer per nit even at ref_white=200.
        let x = i as f64 / (n - 1) as f64;

        // 1. midtone gamma (linear-domain power curve).
        let y_lifted = x.max(0.0).powf(g_eff);

        // 2. tone-map in relative-luminance domain (rel = abs_nits/ref).
        // y_lifted=1.0 maps to ref_white nits absolute, exactly the
        // semantic the CTM-scale-1.0 design promises.
        let abs_nits = y_lifted * dest_ref;
        let rel = abs_nits / dest_ref;
        let rel_out = if rel <= 0.0 {
            0.0
        } else {
            rel * (1.0 + rel * v_curve) / (1.0 + rel)
        };
        let abs_nits_out = (rel_out * dest_ref).min(dest_peak);
        let y_mapped = (abs_nits_out / 10000.0).clamp(0.0, 1.0);

        // 3. PQ encode for the wire.
        let pq_val = pq(y_mapped);
        let q = (pq_val.clamp(0.0, 1.0) * 65535.0).round() as u16;
        lut.push(DrmColorLutEntry {
            red: q,
            green: q,
            blue: q,
            reserved: 0,
        });
    }
    lut
}

/// Build the CTM matrix combining gamut conversion with ref-white scaling.
///
/// Output format matches the kernel's `struct drm_color_ctm`:
///
/// ```c
/// struct drm_color_ctm {
///     /* Sign-magnitude S31.32: bit 63 = sign, bits 0..63 = magnitude */
///     __u64 matrix[9];
/// };
/// ```
///
/// The matrix is row-major when applied as `out = M * in` (i.e., index 0..2
/// is row 0, 3..5 is row 1, 6..8 is row 2). Each `f64` value is encoded in
/// sign-magnitude S31.32: sign bit set if the value is negative, magnitude
/// in the lower 63 bits as a Q31.32 fixed-point.
///
/// We fold four transforms into one matrix, applied in this order to the
/// linear Rec.709 input vector:
/// 1. `Rec.709 → target_gamut` (BT.2020 or DCI-P3 D65), linearly interpolated
///    between identity (`gamut_mix = 0.0`) and the full remap matrix
///    (`gamut_mix = 1.0`) so the user's gamut slider actually does something.
/// 2. Saturation: chroma-preserving luma-mix matrix
///    `S = sat·I + (1-sat)·1·wᵀ` where `w` are the target gamut's luma
///    weights. `sat = 1.0` is identity (no change), `sat > 1.0` boosts chroma,
///    `sat < 1.0` desaturates toward grey, `sat = 0.0` collapses to luminance.
///    Saturation is applied AFTER the gamut remap so the luma weights match
///    the output gamut.
/// 3. Scalar multiplication by `ref_white_nits / 10000` so that linear sRGB
///    `1.0` becomes `ref_white_nits / 10000` at the GAMMA_LUT input — i.e.
///    sRGB white encodes to PQ value corresponding to `ref_white_nits` at
///    the panel.
///
/// Net: input `(R, G, B)` in linear Rec.709 → output `(R', G', B')` in
/// linear target gamut at the right luminance for PQ encoding, with
/// chroma-preserving saturation applied. The shader can stay an identity
/// passthrough — direct primary-plane scanout works even when the saturation
/// slider is non-default.
pub fn gamut_ctm_with_ref_white(
    container: HdrColorContainer,
    ref_white_nits: u16,
    gamut_mix: f32,
    saturation: f32,
) -> [u64; 9] {
    // Linear-light Rec.709 → BT.2020 matrix (BT.2087-0 Annex 1, equivalent
    // to converting via XYZ pivot using D65 chromaticities).
    const M_REC709_TO_BT2020: [[f64; 3]; 3] = [
        [0.62740389896, 0.32928303525, 0.04331306579],
        [0.06909728935, 0.91954039517, 0.01136231548],
        [0.01639143887, 0.08801330593, 0.89559525520],
    ];

    // Linear-light Rec.709 → DCI-P3 D65 (computed via XYZ pivot, D65 white).
    const M_REC709_TO_P3: [[f64; 3]; 3] = [
        [0.82246197, 0.17753803, 0.00000000],
        [0.03319420, 0.96680580, 0.00000000],
        [0.01708263, 0.07239738, 0.91051999],
    ];

    let m_remap = match container {
        HdrColorContainer::Bt2020 => M_REC709_TO_BT2020,
        HdrColorContainer::DciP3 => M_REC709_TO_P3,
    };

    // Identity matrix (709 in, 709 out — i.e. no gamut remap).
    const M_IDENTITY: [[f64; 3]; 3] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    // Linearly interpolate between identity and the full 709→target remap.
    // 0.0 = pure 709 passthrough, 1.0 = full BT.2020/P3 remap. (Note: the
    // signal still gets tagged BT.2020/PQ on the wire; the gamut slider
    // controls only how aggressively colors are *remapped* into the wider
    // container. At mix=0 the panel sees 709 primaries inside a BT.2020
    // container — duller but stable; at mix=1 the panel sees the full
    // wide-gamut remap.)
    let mix = gamut_mix.clamp(0.0, 1.0) as f64;
    let mut m_gamut = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m_gamut[r][c] = M_IDENTITY[r][c] * (1.0 - mix) + m_remap[r][c] * mix;
        }
    }

    // Saturation matrix S = sat·I + (1-sat)·1·wᵀ in the target gamut, so
    // luma weights match where saturation is being applied (after the
    // gamut remap). BT.2020 weights for both BT.2020 and P3 D65 — the
    // P3 weights differ slightly but on a wide-gamut OLED the difference
    // is perceptually negligible and using one set keeps the math simple.
    let sat = saturation.max(0.0) as f64;
    const W: [f64; 3] = [0.2627, 0.6780, 0.0593]; // BT.2020 luma weights
    let mut s_sat = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let identity = if r == c { 1.0 } else { 0.0 };
            s_sat[r][c] = sat * identity + (1.0 - sat) * W[c];
        }
    }

    // Compose: M_combined = S_sat · M_gamut (matrix-mul; M_gamut runs first
    // on the input vector, then S_sat). Row-major naive 3×3 multiply.
    let mut m_combined = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m_combined[r][c] =
                s_sat[r][0] * m_gamut[0][c] + s_sat[r][1] * m_gamut[1][c] + s_sat[r][2] * m_gamut[2][c];
        }
    }

    // CTM scale is 1.0 — the ref_white → absolute nits mapping lives entirely
    // in the GAMMA_LUT now (see pq_encode_lut_with_tonemap). This gives the
    // LUT its full 1024-entry input range to represent the SDR luminance
    // curve at high precision; the previous design that baked ref_white/10000
    // into the CTM left the LUT with only ~21 entries covering the SDR range
    // for typical ref_white values (200-300 nits), which caused severe
    // posterization / banding in midtone luminances after the tone-mapping
    // landed.
    //
    // `ref_white_nits` stays in the function signature for API stability —
    // existing callers don't need to change — but the value is unused here.
    // The LUT builder reads it (and panel_peak) directly to parameterize the
    // tone-map curve.
    let _ = ref_white_nits; // intentionally unused; LUT owns ref-white mapping
    let mut out = [0u64; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] = encode_s31_32_sign_magnitude(m_combined[row][col]);
        }
    }
    out
}

/// Encode an `f64` as the kernel's sign-magnitude S31.32 fixed-point.
///
/// Bit 63 is the sign (1 = negative, 0 = positive); bits 0..63 are the
/// magnitude as a 31-integer-bit / 32-fractional-bit unsigned fixed-point.
/// Values that overflow the 31-bit integer range saturate to max magnitude.
fn encode_s31_32_sign_magnitude(value: f64) -> u64 {
    let neg = value < 0.0;
    let mag = value.abs();
    // 2^32 = 4_294_967_296. Multiplying by this puts 32 fractional bits
    // into the low half. Saturate above 2^63 - 1 to avoid u64 overflow.
    let scaled = (mag * 4_294_967_296.0).min((u64::MAX >> 1) as f64) as u64;
    if neg { scaled | (1u64 << 63) } else { scaled }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to extract the `red` field from a `#[repr(C, packed)]` entry
    /// safely (test code accessing packed fields directly is UB).
    fn red_of(e: DrmColorLutEntry) -> u16 {
        let copied = e;
        copied.red
    }

    #[test]
    fn srgb_decode_lut_endpoints() {
        let lut = srgb_decode_lut(129);
        assert_eq!(lut.len(), 129);
        assert_eq!(red_of(lut[0]), 0);
        assert_eq!(red_of(lut[128]), 65535);
        // mid-gray (0.5 sRGB) decodes to ~0.214 linear
        let mid_idx = 64;
        let mid_val = red_of(lut[mid_idx]) as f64 / 65535.0;
        let expected_input: f64 = mid_idx as f64 / 128.0;
        let expected_linear = if expected_input <= 0.04045 {
            expected_input / 12.92
        } else {
            ((expected_input + 0.055) / 1.055).powf(2.4)
        };
        assert!(
            (mid_val - expected_linear).abs() < 1e-3,
            "mid-gray decode mismatch: {} vs {}",
            mid_val,
            expected_linear
        );
    }

    #[test]
    fn pq_encode_lut_endpoints() {
        let lut = pq_encode_lut(1024);
        assert_eq!(lut.len(), 1024);
        assert!(red_of(lut[0]) < 100);
        assert!(red_of(lut[1023]) > 65500);
    }

    #[test]
    fn pq_encode_lut_known_values() {
        // PQ encode of 100 nits = linear 0.01 should give ~0.502.
        let lut = pq_encode_lut(1024);
        let idx: usize = (0.01_f64 * 1023.0).round() as usize;
        let v = red_of(lut[idx]) as f64 / 65535.0;
        assert!(
            (v - 0.502).abs() < 0.02,
            "PQ at 100 nits should be ~0.502, got {}",
            v
        );
    }

    #[test]
    fn ctm_identity_at_full_scale() {
        // Test pure passthrough — Rec.709 to itself with ref_white = 10000
        // would be identity scaled by 1.0. We don't expose Rec.709 as a
        // target, but we can test that BT.2020 identity-element diagonal
        // is reasonable.
        let ctm = gamut_ctm_with_ref_white(HdrColorContainer::Bt2020, 10000, 1.0, 1.0);
        // Diagonal entries should be positive
        assert_eq!(ctm[0] >> 63, 0); // [0][0] is positive
        assert_eq!(ctm[4] >> 63, 0); // [1][1] is positive
        assert_eq!(ctm[8] >> 63, 0); // [2][2] is positive
        // Magnitude of [0][0] should be ~0.6274 * 2^32 (sat=1 → S=I, so the
        // composed matrix is just the gamut remap)
        let expected = (0.62740389896 * 4_294_967_296.0) as u64;
        let actual = ctm[0] & !(1u64 << 63);
        let diff = if actual > expected {
            actual - expected
        } else {
            expected - actual
        };
        assert!(diff < 1024, "[0][0] off by more than 1024 ulps: {}", diff);
    }

    #[test]
    fn ctm_ignores_ref_white_param() {
        // ref_white was previously baked into the CTM as a scale factor;
        // it now lives in the GAMMA_LUT (see pq_encode_lut_with_tonemap).
        // CTMs for any two ref_white values with the same gamut_mix + sat
        // must be byte-identical.
        let ctm_a = gamut_ctm_with_ref_white(HdrColorContainer::Bt2020, 200, 1.0, 1.0);
        let ctm_b = gamut_ctm_with_ref_white(HdrColorContainer::Bt2020, 10000, 1.0, 1.0);
        assert_eq!(ctm_a, ctm_b, "CTM must not depend on ref_white anymore");
    }

    #[test]
    fn ctm_gamut_mix_zero_is_identity() {
        // gamut_mix = 0 → 709 passthrough, no scale.
        // [0][0] should be exactly 1.0, [0][1] should be 0.
        let ctm = gamut_ctm_with_ref_white(HdrColorContainer::Bt2020, 1000, 0.0, 1.0);
        let m00 = (ctm[0] & !(1u64 << 63)) as f64 / 4_294_967_296.0;
        let m01 = (ctm[1] & !(1u64 << 63)) as f64 / 4_294_967_296.0;
        assert!((m00 - 1.0).abs() < 1e-5, "[0][0] should be 1.0, got {}", m00);
        assert!(m01 < 1e-5, "[0][1] should be ~0, got {}", m01);
    }

    #[test]
    fn ctm_gamut_mix_half() {
        // gamut_mix = 0.5 → halfway between identity and full BT.2020 remap.
        // [0][0] should be (1.0 + 0.6274) / 2 = 0.8137, no scaling now.
        let ctm = gamut_ctm_with_ref_white(HdrColorContainer::Bt2020, 1000, 0.5, 1.0);
        let m00 = (ctm[0] & !(1u64 << 63)) as f64 / 4_294_967_296.0;
        let expected = (1.0 + 0.62740389896) / 2.0;
        assert!(
            (m00 - expected).abs() < 1e-5,
            "[0][0] half-mix should be {}, got {}",
            expected,
            m00
        );
    }

    /// Helper — read the magnitude of an S31.32 sign-magnitude entry as f64.
    fn ctm_entry(ctm: &[u64; 9], i: usize) -> f64 {
        let mag = (ctm[i] & !(1u64 << 63)) as f64 / 4_294_967_296.0;
        if ctm[i] >> 63 != 0 { -mag } else { mag }
    }

    #[test]
    fn ctm_saturation_one_is_identity() {
        // sat=1 with gamut_mix=0 (709 passthrough) and ref_white=10000 should
        // give a pure identity matrix. Diagonal=1.0, off-diagonal=0.
        let ctm = gamut_ctm_with_ref_white(HdrColorContainer::Bt2020, 10000, 0.0, 1.0);
        for i in 0..3 {
            for j in 0..3 {
                let v = ctm_entry(&ctm, i * 3 + j);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - expected).abs() < 1e-5,
                    "sat=1 ctm[{},{}] = {} (want {})",
                    i, j, v, expected
                );
            }
        }
    }

    #[test]
    fn ctm_saturation_zero_collapses_to_luma() {
        // sat=0 with gamut_mix=0 means every output channel is the BT.2020
        // luma of the input — every row should be the luma weight vector.
        let ctm = gamut_ctm_with_ref_white(HdrColorContainer::Bt2020, 10000, 0.0, 0.0);
        const W: [f64; 3] = [0.2627, 0.6780, 0.0593];
        for i in 0..3 {
            for j in 0..3 {
                let v = ctm_entry(&ctm, i * 3 + j);
                assert!(
                    (v - W[j]).abs() < 1e-5,
                    "sat=0 ctm[{},{}] = {} (want {})",
                    i, j, v, W[j]
                );
            }
        }
    }

    #[test]
    fn ctm_saturation_boost_off_diagonal_negative() {
        // sat>1 amplifies chroma; off-diagonal entries become negative
        // (the formula gives (1-sat)·w_j which is negative when sat>1).
        // sat=2 with gamut_mix=0 ref_white=10000: off-diag[0][1] = -1·0.678
        let ctm = gamut_ctm_with_ref_white(HdrColorContainer::Bt2020, 10000, 0.0, 2.0);
        let m01 = ctm_entry(&ctm, 1);
        assert!(m01 < 0.0, "sat=2 ctm[0][1] should be negative, got {}", m01);
        assert!(
            (m01 - (-0.678)).abs() < 1e-5,
            "sat=2 ctm[0][1] should be ~-0.678, got {}",
            m01
        );
    }

    #[test]
    fn pq_lut_with_gamma_one_matches_plain() {
        // gamma=1.0 should produce the exact same LUT as pq_encode_lut.
        let plain = pq_encode_lut(1024);
        let baked = pq_encode_lut_with_gamma(1024, 1.0);
        assert_eq!(plain.len(), baked.len());
        for i in 0..plain.len() {
            assert_eq!(red_of(plain[i]), red_of(baked[i]),
                "diverged at entry {}", i);
        }
    }

    #[test]
    fn pq_lut_with_gamma_punch_darkens_midtones() {
        // gamma > 1 (more punch / contrast) should DARKEN linear-domain
        // midtones before PQ-encoding them, so the PQ output for a given
        // input bin is lower than the plain LUT.
        let plain = pq_encode_lut(1024);
        let punched = pq_encode_lut_with_gamma(1024, 1.5);
        // pick a midtone bin (index ~10 = linear 0.01 = 100 nits with
        // ref_white scaling — solidly in the SDR midtone region).
        let mid_idx: usize = 10;
        assert!(red_of(punched[mid_idx]) < red_of(plain[mid_idx]),
            "punch (gamma>1) should lower midtone PQ output: plain={} punched={}",
            red_of(plain[mid_idx]), red_of(punched[mid_idx]));
        // Endpoints stay pinned: 0→0, 1→max.
        assert_eq!(red_of(punched[0]), red_of(plain[0]));
        assert_eq!(red_of(punched[1023]), red_of(plain[1023]));
    }
}
