// Linearize fragment shader — Path B chunk 2 of the cosmic-comp HDR experiment.
//
// Per-surface decode-to-linear shader. Each surface declares its color encoding
// via wp_color_management_v1; this shader applies the inverse transfer function
// + primaries-to-composite-space matrix + reference white scaling so the
// offscreen framebuffer receives uniformly linear values in the composite
// color volume.
//
// Composite scale convention: linear normalized [0, 1] = [0, 10000 cd/m²]
// (PQ peak). SDR clients' sRGB-decoded values are scaled by `ref_white / 10000`
// so a sRGB white pixel lands at ~ref_white / 10000 in the composite.
//
// Uniforms:
//   tf_id:                int   selector for decode function (see below)
//   ref_white_scale:      float reference-white luminance / 10000 (SDR clients)
//   primaries_matrix:     mat3  source primaries → composite-space primaries
//
// tf_id values mirror the order of the enum in our protocol bindings, with 0
// reserved for "passthrough" (treat texture sample as already linear in the
// composite scale).
//
//   0 = passthrough (already linear, no decode)
//   1 = sRGB (gamma 2.4 + linear segment)
//   2 = BT.1886 (pure power 2.4)
//   3 = Gamma 2.2 (pure power 2.2)
//   4 = ST.2084 PQ (10,000 cd/m² absolute)
//   5 = HLG (Hybrid Log-Gamma, scene-referred inverse OETF)
//   6 = Extended linear (identity, but no ref_white scaling — for scRGB)

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

uniform int tf_id;
uniform float ref_white_scale;
uniform mat3 primaries_matrix;

// sRGB inverse EOTF — gamma 2.4 segment + linear toe.
vec3 decode_srgb(vec3 c) {
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    vec3 lo = c / 12.92;
    return mix(lo, hi, step(0.04045, c));
}

// BT.1886 — pure power-curve approximation of CRT response. Common for SDR
// video content. Ignores L_b (black-level adjustment) since our blacks are
// effectively zero on OLED.
vec3 decode_bt1886(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.4));
}

// Pure 2.2 power curve. Used as a "display-referred SDR" decode.
vec3 decode_gamma22(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

// ST.2084 PQ inverse — SMPTE 2084 absolute luminance encoding. Input is
// PQ-encoded [0, 1]; output is linear [0, 1] = [0, 10000 cd/m²].
vec3 decode_pq(vec3 c) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 v_pow = pow(max(c, vec3(0.0)), vec3(1.0 / m2));
    vec3 num = max(v_pow - c1, vec3(0.0));
    vec3 den = c2 - c3 * v_pow;
    return pow(num / den, vec3(1.0 / m1));
}

// HLG inverse OETF (without OOTF). Scene-referred: assumes the downstream
// pipeline will apply the display-side OOTF appropriate to the panel. For
// our compositor, that means HLG content composites at scene luminance
// (peak = 12 in scene-light units) which we map to [0, 1] = [0, 10000 cd/m²]
// via a 1/12 scaling.
vec3 decode_hlg(vec3 c) {
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float cc = 0.55991073;
    vec3 lo = (c * c) / 3.0;
    vec3 hi = (exp((c - cc) / a) + b) / 12.0;
    return mix(lo, hi, step(0.5, c));
}

void main() {
    // Sample texture.
    vec4 color = texture2D(tex, v_coords);
    #if defined(NO_ALPHA)
    color = vec4(color.rgb, 1.0);
    #endif

    // Save alpha — we operate on RGB only, alpha is linear-already.
    float a = color.a;
    vec3 rgb = color.rgb;

    // Decode transfer function to linear.
    if (tf_id == 1) {
        rgb = decode_srgb(rgb);
        // Scale SDR reference white into composite (PQ-aligned) scale.
        rgb *= ref_white_scale;
    } else if (tf_id == 2) {
        rgb = decode_bt1886(rgb);
        rgb *= ref_white_scale;
    } else if (tf_id == 3) {
        rgb = decode_gamma22(rgb);
        rgb *= ref_white_scale;
    } else if (tf_id == 4) {
        // PQ already absolute-luminance encoded; no ref_white scaling.
        rgb = decode_pq(rgb);
    } else if (tf_id == 5) {
        // HLG scene-referred decode; 1/12 maps scene-peak to PQ-scale.
        rgb = decode_hlg(rgb) / 12.0;
    } else if (tf_id == 6) {
        // Extended linear (scRGB-style). Already linear, but units assumed
        // to be sRGB-relative — scale by ref_white_scale to enter PQ-scale.
        rgb *= ref_white_scale;
    }
    // tf_id == 0 → passthrough (e.g. for cursor / non-color content). Skipped.

    // Apply source-primaries → composite-primaries matrix. Identity matrix
    // is a valid passthrough when source and composite primaries already
    // match (e.g. both BT.709).
    rgb = primaries_matrix * rgb;

    // Premultiply by alpha — output is premultiplied linear.
    rgb *= a;

    // Apply uniform alpha (e.g. window opacity slider).
    vec4 result = vec4(rgb, a) * alpha;

    #if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        result = vec4(0.0, 0.2, 0.0, 0.2) + result * 0.8;
    #endif

    gl_FragColor = result;
}
