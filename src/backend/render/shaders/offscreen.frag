#version 100

//_DEFINES_

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

precision highp float;
#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

uniform float invert;
uniform float color_mode;

// Path B (COSMIC_HDR_PATH_B=1) — when 1.0, the offscreen FB is RGBA16F linear
// in the composite color space with SDR ref_white already scaled in (the
// per-surface linearize stage in clipped_surface.frag did sRGB→linear→
// BT.709→BT.2020 matrix→ref_white). Postprocess's color_mode=5 branch skips
// those stages and only applies sat/gamma + PQ encode.
//
// When 0.0 (default), color_mode=5 does the full sRGB→linear→matrix→
// ref_white→PQ pipeline as before.
uniform float path_b_active;

// HDR tuning uniforms (only meaningful when color_mode >= 5.0).
//   hdr_colorspace: 0.0 = BT.2020 target, 1.0 = DCI-P3 target.
//   hdr_ref_white:  SDR white anchor in cd/m^2 (e.g. 100..500). Defaults
//                   200 if uniform isn't bound (matches ~BT.2408 cinema).
//   hdr_gamut_mix:  0.0 = no Rec.709→target matrix (raw sRGB through),
//                   1.0 = full conversion. Lerp between identity and the
//                   selected matrix. Useful for triaging "washed out" vs
//                   "oversaturated" without rebuilding.
uniform float hdr_colorspace;
uniform float hdr_ref_white;
uniform float hdr_gamut_mix;
// Saturation boost applied in linear-light luminance space. 1.0 = neutral
// (colorimetrically correct), >1.0 = more vivid, <1.0 = washed. Compensates
// for the perceived loss of "punch" when going from vendor-saturated SDR
// mode to colorimetrically-truthful HDR mode. Range typically 0.8 - 1.5.
uniform float hdr_saturation;
// Midtone gamma applied to LUMINANCE (Y) before saturation + PQ encoding.
// Solves the "SDR content rendered in HDR mode looks dim/washed" problem
// — desktop UI pixels have low absolute luminance after sRGB decode (mid-gray
// linear ~0.21), so when scaled by ref_white/10000 they end up at ~50 nits
// while the panel can do 525. A gamma < 1.0 lifts midtones into the HDR
// luminance range without touching chroma (RGB scale uniformly by the new
// Y / old Y ratio). 1.0 = neutral, 0.7 = soft lift, 0.5 = aggressive.
// Equivalent to Windows AutoHDR's brightness-lift curve concept.
uniform float hdr_midtone_gamma;

void main() {
    vec4 color = texture2D(tex, v_coords);

#if defined(NO_ALPHA)
    color = vec4(color.rgb, 1.0) * alpha;
#else
    color = color * alpha;
#endif

    // un-multiply
    color.rgb /= color.a;

    // First invert then filter

    if (invert == 1.0) {
        color.rgb = 1.0 - color.rgb;
    }

    if (color_mode == 1.0) {        // greyscale
        float value = (color.r + color.g + color.b) / 3.0;
        color = vec4(value, value, value, color.a);
    } else if (color_mode >= 2.0 && color_mode <= 4.0) {
        float L = (17.8824 * color.r) + (43.5161 * color.g) + (4.11935 * color.b);
	    float M = (3.45565 * color.r) + (27.1554 * color.g) + (3.86714 * color.b);
    	float S = (0.0299566 * color.r) + (0.184309 * color.g) + (1.46709 * color.b);

        float l, m, s;
        if (color_mode == 2.0) { // Protanopia
            l = 0.0 * L + 2.02344 * M + -2.52581 * S;
		    m = 0.0 * L + 1.0 * M + 0.0 * S;
		    s = 0.0 * L + 0.0 * M + 1.0 * S;
        } else if (color_mode == 3.0) { // Deuteranopia
            l = 1.0 * L + 0.0 * M + 0.0 * S;
            m = 0.494207 * L + 0.0 * M + 1.24827 * S;
            s = 0.0 * L + 0.0 * M + 1.0 * S; 
        } else if (color_mode == 4.0) { // Tritanopia
            l = 1.0 * L + 0.0 * M + 0.0 * S;
            m = 0.0 * L + 1.0 * M + 0.0 * S;
            s = -0.395913 * L + 0.801109 * M + 0.0 * S; 
        } else {
            // unknown
            l = L;
            m = M;
            s = S;
        }

        vec3 error;
        error.r = (0.0809444479 * l) + (-0.130504409 * m) + (0.116721066 * s);
        error.g = (-0.0102485335 * l) + (0.0540193266 * m) + (-0.113614708 * s);
        error.b = (-0.000365296938 * l) + (-0.00412161469 * m) + (0.693511405 * s);

        vec3 diff = color.rgb - error;
        vec3 correction;
        correction.r = 0.0;
        correction.g = (diff.r * 0.7) + (diff.g * 1.0);
        correction.b =  (diff.r * 0.7) + (diff.b * 1.0);

        color.rgb += correction;
    }

    // ---- HDR PQ encode (color_mode == 5.0 normal, 6.0 test pattern) ------
    // Pipeline: sRGB-encoded → linearize → Rec.709 → target-gamut primary
    // remap (mixed by hdr_gamut_mix) → scale to hdr_ref_white cd/m² → inverse
    // PQ EOTF → write 10-bit PQ-encoded values to the FB. The output
    // framebuffer is 10-bit Abgr2101010 wired to a connector with the
    // matching Colorspace + HDR_OUTPUT_METADATA (PQ EOTF). All math from
    // public standards (BT.2100, BT.2087, ST 2084, BT.2408).
    // ---- Hardware HDR path (color_mode == 7.0) --------------------------
    // CRTC color pipeline (DEGAMMA_LUT + CTM + GAMMA_LUT) is staged via
    // smithay's `HdrState`; the kernel display engine handles sRGB decode
    // → gamut + ref-white scale → PQ encode entirely in fixed-function
    // hardware on output. Shader's only job here is "stay out of the way":
    // pass framebuffer-tagged sRGB content through unchanged so the hardware
    // pipeline gets canonical sRGB at its DEGAMMA input.
    //
    // Tunable saturation / midtone-gamma in this mode would require a
    // sRGB-decode → adjust-in-linear → sRGB-re-encode shader pass that
    // stays "below" the hardware encoding. Deferred — Phase 2A.2 work.
    // For now, color_mode=7 means: shader is a no-op; user's saturation
    // and midtone gamma sliders affect nothing while in hardware path.
    if (color_mode > 6.5 && color_mode < 7.5) {
        // Skip the PQ encode block below entirely. color stays sRGB-ish
        // (whatever cosmic rendered), hardware handles HDR encoding.
    } else if (color_mode > 7.5 && color_mode < 8.5) {
        // ---- Hardware HDR with shader saturation+midtone tuning -------
        // Hardware CRTC pipeline does sRGB-decode → CTM (gamut+ref_white)
        // → PQ-encode on output. Shader's job here is to pre-modify the
        // sRGB framebuffer values for tuner saturation + midtone gamma
        // BEFORE the hardware DEGAMMA_LUT picks them up. To do that we
        // sRGB-decode in shader, apply the sat/gamma curves in linear
        // luminance space (chroma-preserving), then sRGB-re-encode so
        // the hardware's DEGAMMA gets canonical sRGB input again.
        // No PQ encode in shader — hardware does it.
        vec3 lin8;
        lin8.r = (color.r <= 0.04045) ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4);
        lin8.g = (color.g <= 0.04045) ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4);
        lin8.b = (color.b <= 0.04045) ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4);

        // Midtone gamma in luminance space (preserves chroma).
        // Sensitivity dampening: in the hardware path, the shader applies
        // gamma in 709 linear space and the hardware CTM then maps
        // 709→BT.2020 + ref_white scale before the GAMMA_LUT does inverse-PQ
        // encode. Because PQ is very steep through low/midtones, a small
        // linear-domain gamma change becomes a large PQ-domain perceptual
        // change. Empirically the slider at 1.10 here ≈ 2.00 slider in the
        // old software path — halve the raw slider deviation from 1.0 so the
        // tuner has more granularity / breathing room.
        float gamma8_raw = (hdr_midtone_gamma < 0.1) ? 1.0 : hdr_midtone_gamma;
        float gamma8 = 1.0 + (gamma8_raw - 1.0) * 0.5;
        float Y_orig8 = dot(lin8, vec3(0.2627, 0.6780, 0.0593));
        float Y_new8 = pow(max(Y_orig8, 0.0), gamma8);
        float lift8 = (Y_orig8 > 0.0001) ? (Y_new8 / Y_orig8) : 1.0;
        vec3 lin8_lifted = lin8 * lift8;

        // Saturation (mix between luma-only and full chroma).
        float sat8 = (hdr_saturation < 0.5) ? 1.0 : hdr_saturation;
        float Y_lifted8 = dot(lin8_lifted, vec3(0.2627, 0.6780, 0.0593));
        vec3 lin8_final = mix(vec3(Y_lifted8), lin8_lifted, sat8);
        lin8_final = clamp(lin8_final, vec3(0.0), vec3(1.0));

        // sRGB re-encode (piecewise inverse of decode above).
        vec3 srgb8;
        srgb8.r = (lin8_final.r <= 0.0031308) ? 12.92 * lin8_final.r
                : 1.055 * pow(lin8_final.r, 1.0 / 2.4) - 0.055;
        srgb8.g = (lin8_final.g <= 0.0031308) ? 12.92 * lin8_final.g
                : 1.055 * pow(lin8_final.g, 1.0 / 2.4) - 0.055;
        srgb8.b = (lin8_final.b <= 0.0031308) ? 12.92 * lin8_final.b
                : 1.055 * pow(lin8_final.b, 1.0 / 2.4) - 0.055;
        color.rgb = clamp(srgb8, vec3(0.0), vec3(1.0));
    } else if (color_mode >= 5.0) {
        // ---- Source pixel: either real content (5.0) or test pattern (6.0).
        vec3 src = color.rgb;
        if (color_mode > 5.5) {
            // Test pattern in v_coords (0..1 across the offscreen target):
            //   Top-left  : 100 nits gray
            //   Top-right : 300 nits gray
            //   Bot-left  : 500 nits gray
            //   Bot-right : 200 nits saturated R / G / B vertical bars
            // Each region writes the linear-light value the rest of the
            // pipeline expects (sRGB-encoded values that linearize to the
            // matching nits/hdr_ref_white). Bars give a saturation reference;
            // grays give a luminance ramp reference.
            float u = v_coords.x;
            float v = v_coords.y;
            float ref_w = max(hdr_ref_white, 1.0);
            if (u < 0.5 && v < 0.5) {
                float frac = 100.0 / ref_w;
                src = vec3(frac, frac, frac);
            } else if (u >= 0.5 && v < 0.5) {
                float frac = 300.0 / ref_w;
                src = vec3(frac, frac, frac);
            } else if (u < 0.5 && v >= 0.5) {
                float frac = 500.0 / ref_w;
                src = vec3(frac, frac, frac);
            } else {
                float frac = 200.0 / ref_w;
                if (u < 0.6667) {
                    src = vec3(frac, 0.0, 0.0);
                } else if (u < 0.8333) {
                    src = vec3(0.0, frac, 0.0);
                } else {
                    src = vec3(0.0, 0.0, frac);
                }
            }
            // Test pattern is already linear-light, skip sRGB decode.
        }

        // 1. sRGB → linear (piecewise sRGB EOTF). For test pattern we put
        //    linear values straight through by gating; checked above.
        //    PATH B: input is already linear from per-surface linearize stage,
        //    so this is also a passthrough.
        vec3 lin;
        if (color_mode > 5.5 || path_b_active > 0.5) {
            lin = src;
        } else {
            lin.r = (src.r <= 0.04045) ? src.r / 12.92 : pow((src.r + 0.055) / 1.055, 2.4);
            lin.g = (src.g <= 0.04045) ? src.g / 12.92 : pow((src.g + 0.055) / 1.055, 2.4);
            lin.b = (src.b <= 0.04045) ? src.b / 12.92 : pow((src.b + 0.055) / 1.055, 2.4);
        }

        // 2. Rec.709 → target-gamut matrix, mixed by hdr_gamut_mix in [0,1].
        //    target = BT.2020 if hdr_colorspace < 0.5, else DCI-P3 D65.
        //    Both matrices computed analytically from chromaticity coords
        //    (BT.2087 method; values published in BT.2087-0 §4 / ITU docs).
        vec3 lin_target;
        if (path_b_active > 0.5) {
            // PATH B: per-surface linearize already applied source→composite
            // primaries matrix. Skip the matrix here.
            lin_target = lin;
        } else {
            vec3 lin_remap;
            if (hdr_colorspace < 0.5) {
                // Rec.709 → BT.2020 (BT.2087 Annex 1)
                mat3 M709to2020 = mat3(
                    0.6274,  0.0691,  0.0164,   // col 0 (R'-from)
                    0.3293,  0.9195,  0.0880,   // col 1 (G'-from)
                    0.0433,  0.0114,  0.8956    // col 2 (B'-from)
                );
                lin_remap = M709to2020 * lin;
            } else {
                // Rec.709 → DCI-P3 D65 (computed via XYZ pivot)
                mat3 M709toP3 = mat3(
                    0.8225,  0.0331,  0.0171,
                    0.1774,  0.9669,  0.0724,
                    0.0000,  0.0000,  0.9108
                );
                lin_remap = M709toP3 * lin;
            }
            lin_target = mix(lin, lin_remap, clamp(hdr_gamut_mix, 0.0, 1.0));
        }

        // 3. Optional saturation boost in linear-light luminance space.
        //    KWin and other reference HDR pipelines do NOT include a
        //    saturation lift in their HDR encode path — they aim for
        //    colorimetric truth. But on this Tandem OLED, SDR mode applies
        //    vendor color enhancements that go away in HDR mode, making the
        //    HDR desktop look "washed" in user perception even though it's
        //    mathematically correct. A modest saturation boost (1.1 - 1.3)
        //    closes the perceived gap. Implemented as luminance-preserving
        //    `mix(vec3(Y), color, saturation)` so chroma scales but
        //    luminance is preserved (BT.2020 luma weights for our target).
        // 3a. Midtone gamma in LUMINANCE space — lifts dim SDR-derived pixels
        //     into the HDR luminance range without desaturating. Default 1.0
        //     = no lift (colorimetric). 0.6-0.8 makes desktop content look
        //     "punchy HDR-like." The trick: compute Y, gamma-curve it, scale
        //     RGB by the ratio so chroma is preserved.
        float gamma = (hdr_midtone_gamma < 0.1) ? 1.0 : hdr_midtone_gamma;
        float Y_orig = dot(lin_target, vec3(0.2627, 0.6780, 0.0593));
        float Y_new = pow(max(Y_orig, 0.0), gamma);
        float lift_scale = (Y_orig > 0.0001) ? (Y_new / Y_orig) : 1.0;
        vec3 lin_lifted = lin_target * lift_scale;

        // 3b. Saturation: same fallback logic as midtone. <0.5 means uniform
        //     binding failed; use 1.0 (no boost) so we don't blow into pure
        //     grayscale (mix at 0 would do that).
        float sat = (hdr_saturation < 0.5) ? 1.0 : hdr_saturation;
        float Y_lifted = dot(lin_lifted, vec3(0.2627, 0.6780, 0.0593));
        vec3 lin_final = mix(vec3(Y_lifted), lin_lifted, sat);

        // 4. Linear SDR 1.0 → hdr_ref_white cd/m², normalized to PQ's 10000-nit
        //    peak. (PQ encodes absolute luminance; 1.0 in == 10000 nits out.)
        //    Floor at 250 nits — if the uniform fails to bind for any reason
        //    we'd rather surface a usable brightness instead of pitch black.
        //
        //    PATH B: per-surface linearize already scaled to ref_white. Skip.
        vec3 hdr_lin;
        if (path_b_active > 0.5) {
            hdr_lin = lin_final;
        } else {
            float ref_w = max(hdr_ref_white, 250.0);
            hdr_lin = lin_final * (ref_w / 10000.0);
        }

        // 4. Inverse PQ EOTF (ST 2084) — encode linear (nits/10000) to FB val.
        const float m1 = 0.1593017578125;     // 2610/16384
        const float m2 = 78.84375;            // 2523/4096 * 128
        const float c1 = 0.8359375;           // 3424/4096
        const float c2 = 18.8515625;          // 2413/4096 * 32
        const float c3 = 18.6875;             // 2392/4096 * 32
        vec3 ym = pow(max(hdr_lin, 0.0), vec3(m1));
        vec3 pq;
        pq.r = pow((c1 + c2 * ym.r) / (1.0 + c3 * ym.r), m2);
        pq.g = pow((c1 + c2 * ym.g) / (1.0 + c3 * ym.g), m2);
        pq.b = pow((c1 + c2 * ym.b) / (1.0 + c3 * ym.b), m2);

        color.rgb = pq;
    }
    // ----------------------------------------------------------------------

    // re-multiply
    color.rgb *= color.a;


    gl_FragColor = color;
}