// Taken from niri and modified, licensed GPL-3.0
//
// Extended for Path B of the cosmic-comp HDR experiment to subsume the
// linearize stage. When tf_id != 0 (passthrough), the shader:
//   1. samples the texture
//   2. decodes via the inverse-EOTF selected by tf_id
//   3. applies the source→composite primaries matrix
//   4. scales SDR content into PQ-aligned composite luminance
// before applying clipping + corner rounding. Output is linear in the
// composite color space, with alpha pre-multiplied.
//
// When tf_id == 0, the linearize stage is a no-op — original niri behavior,
// surfaces with rounded corners but no HDR linearization needed.
//
// tf_id values:
//   0 = passthrough (no decode — used by existing clipping-only call sites)
//   1 = sRGB
//   2 = BT.1886
//   3 = Gamma 2.2
//   4 = ST.2084 PQ
//   5 = HLG
//   6 = extended linear (scRGB-style)

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

uniform vec2 geo_size;
uniform vec4 corner_radius;
uniform mat3 input_to_geo;

// Path B linearize uniforms — default to passthrough (tf_id=0, identity matrix,
// scale=1.0) for existing clipping-only callers.
uniform int tf_id;
uniform float ref_white_scale;
uniform mat3 primaries_matrix;

// ---------------------------------------------------------------------------
// Linearize stage — inverse EOTFs (Path B).
// ---------------------------------------------------------------------------

vec3 decode_srgb(vec3 c) {
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    vec3 lo = c / 12.92;
    return mix(lo, hi, step(0.04045, c));
}

vec3 decode_bt1886(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.4));
}

vec3 decode_gamma22(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

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

vec3 decode_hlg(vec3 c) {
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float cc = 0.55991073;
    vec3 lo = (c * c) / 3.0;
    vec3 hi = (exp((c - cc) / a) + b) / 12.0;
    return mix(lo, hi, step(0.5, c));
}

// ---------------------------------------------------------------------------
// Clipping / rounding (original).
// ---------------------------------------------------------------------------

float rounding_alpha(vec2 coords, vec2 size) {
    vec2 center;
    float radius;

    if (coords.x < corner_radius.x && coords.y < corner_radius.x) {
        radius = corner_radius.x;
        center = vec2(radius, radius);
    } else if (size.x - corner_radius.y < coords.x && coords.y < corner_radius.y) {
        radius = corner_radius.y;
        center = vec2(size.x - radius, radius);
    } else if (size.x - corner_radius.z < coords.x && size.y - corner_radius.z < coords.y) {
        radius = corner_radius.z;
        center = vec2(size.x - radius, size.y - radius);
    } else if (coords.x < corner_radius.w && size.y - corner_radius.w < coords.y) {
        radius = corner_radius.w;
        center = vec2(radius, size.y - radius);
    } else {
        return 1.0;
    }

    float dist = distance(coords, center);
    float half_px = 0.5;
    return 1.0 - smoothstep(radius - half_px, radius + half_px, dist);
}

void main() {
    vec3 coords_geo = input_to_geo * vec3(v_coords, 1.0);

    // 1. Sample the texture.
    vec4 color = texture2D(tex, v_coords);
    #if defined(NO_ALPHA)
    color = vec4(color.rgb, 1.0);
    #endif

    // 2. Linearize (Path B). Bypassed when tf_id == 0.
    if (tf_id != 0) {
        // Source-encoded RGB is in premultiplied-alpha form. Un-premultiply,
        // decode, re-premultiply at the end.
        float a = color.a;
        vec3 rgb = (a > 0.0) ? (color.rgb / a) : color.rgb;

        if (tf_id == 1) {
            rgb = decode_srgb(rgb) * ref_white_scale;
        } else if (tf_id == 2) {
            rgb = decode_bt1886(rgb) * ref_white_scale;
        } else if (tf_id == 3) {
            rgb = decode_gamma22(rgb) * ref_white_scale;
        } else if (tf_id == 4) {
            // PQ — already absolute luminance, no ref_white scaling.
            rgb = decode_pq(rgb);
        } else if (tf_id == 5) {
            // HLG scene-referred decode; 1/12 maps scene peak to PQ scale.
            rgb = decode_hlg(rgb) / 12.0;
        } else if (tf_id == 6) {
            rgb = rgb * ref_white_scale;
        }

        // Source primaries → composite primaries.
        rgb = primaries_matrix * rgb;

        // Re-premultiply.
        color = vec4(rgb * a, a);
    }

    // 3. Clip / round.
    if (coords_geo.x < 0.0 || 1.0 < coords_geo.x || coords_geo.y < 0.0 || 1.0 < coords_geo.y) {
        color = vec4(0.0);
    } else {
        color = color * rounding_alpha(coords_geo.xy * geo_size, geo_size);
    }

    // 4. Apply final alpha + tint.
    color = color * alpha;

    #if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
    #endif

    gl_FragColor = color;
}
