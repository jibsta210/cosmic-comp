#version 100

//_DEFINES_

// Path B screencopy tone-down shader.
//
// In Path B mode (COSMIC_HDR_PATH_B=1 + HDR enabled), cosmic-comp composites
// into an RGBA16F offscreen FB in *linear, BT.2020 primaries, with absolute
// luminance scaling baked in* — specifically each sRGB-1.0 surface pixel ends
// up at `ref_white_nits / 10000.0` in the texture (per
// clipped_surface.frag's path-B branch).
//
// Screencopy / PrintScreen clients (gnome-screenshot, grim, cosmic-screenshot,
// xdg-desktop-portal-cosmic) request sRGB Argb8888. The naive blit from our
// RGBA16F to their 8-bit sRGB target leaves them with garbage bytes (looks
// washed out / overbright). This shader is the inverse of the per-surface
// linearize stage: undo ref_white scaling → BT.2020→BT.709 matrix → linear→
// sRGB encode → clamp.
//
// Used by `send_screencopy_result` (kms/surface/mod.rs) as a single-pass
// render into a sRGB Argb8888 intermediate texture, which is then handed to
// the existing blit/submit path unchanged.

precision highp float;

uniform sampler2D tex;
uniform float alpha;
varying vec2 v_coords;

// SDR ref_white scaling factor used by clipped_surface.frag's path-B branch.
// `ref_white_nits / 10000.0`. Default 0.025 (= 250 nits) if uniform binding
// fails — matches default ref_white in lib.rs's push_hdr_tuning_to_surfaces.
uniform float ref_white_scale;

void main() {
    vec4 color = texture2D(tex, v_coords);

    // Un-premultiply so we can manipulate RGB without alpha skew.
    if (color.a > 0.0001) {
        color.rgb /= color.a;
    }

    // 1. Undo the per-surface ref_white scale to bring 1.0 back to "sRGB
    //    diffuse white" in linear light. Floor at 0.001 to avoid div-by-zero
    //    if the uniform somehow binds to 0.
    float scale = max(ref_white_scale, 0.001);
    color.rgb /= scale;

    // 2. BT.2020 → BT.709 inverse primaries matrix (BT.2087 Annex 2). This
    //    is the analytic inverse of the M709to2020 matrix in offscreen.frag's
    //    color_mode=5 branch.
    mat3 M2020to709 = mat3(
         1.6605, -0.1246, -0.0182,   // col 0
        -0.5876,  1.1329, -0.1006,   // col 1
        -0.0728, -0.0083,  1.1187    // col 2
    );
    color.rgb = M2020to709 * color.rgb;

    // 3. Clamp to [0,1]. HDR content that exceeded ref_white in linear or
    //    that was out-of-Rec.709-gamut will be hard-clipped here. A real
    //    tone-mapper (Reinhard / Hable / etc.) would compress instead, but
    //    for desktop SDR screencopy of UI content this is the correct
    //    behavior — display-referred SDR maxes at 1.0.
    color.rgb = clamp(color.rgb, 0.0, 1.0);

    // 4. Linear → sRGB OETF (piecewise IEC 61966-2-1).
    vec3 srgb;
    srgb.r = (color.r <= 0.0031308)
        ? 12.92 * color.r
        : 1.055 * pow(color.r, 1.0 / 2.4) - 0.055;
    srgb.g = (color.g <= 0.0031308)
        ? 12.92 * color.g
        : 1.055 * pow(color.g, 1.0 / 2.4) - 0.055;
    srgb.b = (color.b <= 0.0031308)
        ? 12.92 * color.b
        : 1.055 * pow(color.b, 1.0 / 2.4) - 0.055;
    color.rgb = srgb;

    // Re-premultiply.
    color.rgb *= color.a;

    // Apply caller-provided alpha for fade-in / fade-out (compositor never
    // does this for screencopy in practice; included for completeness).
    color *= alpha;

    gl_FragColor = color;
}
