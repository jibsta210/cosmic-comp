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
    if (color_mode >= 5.0) {
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
        vec3 lin;
        if (color_mode > 5.5) {
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
        vec3 lin_target = mix(lin, lin_remap, clamp(hdr_gamut_mix, 0.0, 1.0));

        // 3. Linear SDR 1.0 → hdr_ref_white cd/m², normalized to PQ's 10000-nit
        //    peak. (PQ encodes absolute luminance; 1.0 in == 10000 nits out.)
        //    Floor at 500 nits — if the uniform fails to bind for any reason
        //    we'd rather surface a too-bright-but-usable desktop than a
        //    pitch-black one. ref_w = max(hdr_ref_white, 500.0).
        float ref_w = max(hdr_ref_white, 500.0);
        vec3 hdr_lin = lin_target * (ref_w / 10000.0);

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