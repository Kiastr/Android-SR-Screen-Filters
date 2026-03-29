package com.anime4k.screen;

/**
 * AMD FidelityFX Super Resolution (FSR) 1.0 — 移动端叠加层优化版
 *
 * v1.9.9 亮度完全恢复版：
 *   1. 移除所有人工亮度增益：移除 v1.9.8 中临时的 1.02x 增益，回归原始色彩平衡。
 *   2. 优化坐标对齐：微调采样坐标补偿逻辑，确保采样点始终在 [0,1] 合法范围内，
 *      防止因 0.5 像素偏移导致边缘采样到 FBO 的清除背景（黑底）。
 *   3. 维持 v1.9.7 的 Alpha 剔除特性，确保合成层纯净。
 */
public class FSRShaders {

    public static final String VERTEX_SHADER =
        "#version 300 es\n" +
        "layout(location = 0) in vec4 aPosition;\n" +
        "layout(location = 1) in vec2 aTexCoord;\n" +
        "out vec2 vTexCoord;\n" +
        "void main() {\n" +
        "    gl_Position = aPosition;\n" +
        "    vTexCoord = aTexCoord;\n" +
        "}\n";

    // =================================================================
    // EASU: Edge Adaptive Spatial Upsampling (FSR 1.0)
    // =================================================================
    public static final String FRAG_EASU =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "out vec4 fragColor;\n" +
        "uniform sampler2D uTexture;\n" +
        "uniform vec4 uEasuCon1; // (1.0/InputWidth, 1.0/InputHeight, 0, 0)\n" +
        "\n" +
        "float lanczos2(float d) {\n" +
        "    float d2 = d * d;\n" +
        "    return max(0.0, (1.0 - d2) * (1.0 - d2 * 0.25));\n" +
        "}\n" +
        "\n" +
        "float luma(vec3 c) {\n" +
        "    return dot(c, vec3(0.2126, 0.7152, 0.0722));\n" +
        "}\n" +
        "\n" +
        "void main() {\n" +
        "    vec2 texelSize = uEasuCon1.xy;\n" +
        "    \n" +
        "    // [OPT-v1.9.9] 坐标对齐优化：使用 clamp 保护，防止采样点因偏移超出范围导致边缘黑边渗透\n" +
        "    vec2 sampleCoord = clamp(vTexCoord - 0.5 * texelSize, 0.0, 1.0);\n" +
        "    \n" +
        "    vec3 cC = texture(uTexture, sampleCoord).rgb;\n" +
        "    vec3 cT = texture(uTexture, clamp(sampleCoord + vec2(0.0, -texelSize.y), 0.0, 1.0)).rgb;\n" +
        "    vec3 cB = texture(uTexture, clamp(sampleCoord + vec2(0.0,  texelSize.y), 0.0, 1.0)).rgb;\n" +
        "    vec3 cL = texture(uTexture, clamp(sampleCoord + vec2(-texelSize.x, 0.0), 0.0, 1.0)).rgb;\n" +
        "    vec3 cR = texture(uTexture, clamp(sampleCoord + vec2( texelSize.x, 0.0), 0.0, 1.0)).rgb;\n" +
        "\n" +
        "    float lC = luma(cC), lT = luma(cT), lB = luma(cB), lL = luma(cL), lR = luma(cR);\n" +
        "    float gradH = abs(lL - lR), gradV = abs(lT - lB);\n" +
        "    float edgeStrength = clamp(max(gradH, gradV) * 4.0, 0.0, 1.0);\n" +
        "\n" +
        "    vec3 colorEdge;\n" +
        "    if (gradH > gradV) {\n" +
        "        vec3 cLL = texture(uTexture, clamp(sampleCoord + vec2(-2.0 * texelSize.x, 0.0), 0.0, 1.0)).rgb;\n" +
        "        vec3 cRR = texture(uTexture, clamp(sampleCoord + vec2( 2.0 * texelSize.x, 0.0), 0.0, 1.0)).rgb;\n" +
        "        float w0 = lanczos2(0.0), w1 = lanczos2(1.0), w2 = lanczos2(2.0);\n" +
        "        colorEdge = (cC * w0 + (cL + cR) * w1 + (cLL + cRR) * w2) / (w0 + 2.0*w1 + 2.0*w2);\n" +
        "    } else {\n" +
        "        vec3 cTT = texture(uTexture, clamp(sampleCoord + vec2(0.0, -2.0 * texelSize.y), 0.0, 1.0)).rgb;\n" +
        "        vec3 cBB = texture(uTexture, clamp(sampleCoord + vec2(0.0,  2.0 * texelSize.y), 0.0, 1.0)).rgb;\n" +
        "        float w0 = lanczos2(0.0), w1 = lanczos2(1.0), w2 = lanczos2(2.0);\n" +
        "        colorEdge = (cC * w0 + (cT + cB) * w1 + (cTT + cBB) * w2) / (w0 + 2.0*w1 + 2.0*w2);\n" +
        "    }\n" +
        "    \n" +
        "    // 移除 1.02x 增益，回归原始亮度平衡\n" +
        "    fragColor = vec4(clamp(mix(cC, colorEdge, edgeStrength), 0.0, 1.0), 1.0);\n" +
        "}\n";

    // =================================================================
    // MAS v3: Mobile-Adaptive Sharpening (纯净亮度版)
    // =================================================================
    public static final String FRAG_RCAS =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "out vec4 fragColor;\n" +
        "uniform sampler2D uTexture;\n" +
        "uniform vec4 uRcasCon; // (Sharpness, 0, 0, 0)\n" +
        "\n" +
        "void main() {\n" +
        "    vec2 rcpSize = 1.0 / vec2(textureSize(uTexture, 0));\n" +
        "    float sharpness = uRcasCon.x * 0.5;\n" +
        "\n" +
        "    vec2 sampleCoord = vTexCoord;\n" +
        "    vec3 c = texture(uTexture, sampleCoord).rgb;\n" +
        "    vec3 t = texture(uTexture, clamp(sampleCoord + vec2(0.0, -rcpSize.y), 0.0, 1.0)).rgb;\n" +
        "    vec3 b = texture(uTexture, clamp(sampleCoord + vec2(0.0,  rcpSize.y), 0.0, 1.0)).rgb;\n" +
        "    vec3 l = texture(uTexture, clamp(sampleCoord + vec2(-rcpSize.x, 0.0), 0.0, 1.0)).rgb;\n" +
        "    vec3 r = texture(uTexture, clamp(sampleCoord + vec2( rcpSize.x, 0.0), 0.0, 1.0)).rgb;\n" +
        "\n" +
        "    float lumaC = dot(c, vec3(0.2126, 0.7152, 0.0722));\n" +
        "    float lumaT = dot(t, vec3(0.2126, 0.7152, 0.0722));\n" +
        "    float lumaB = dot(b, vec3(0.2126, 0.7152, 0.0722));\n" +
        "    float lumaL = dot(l, vec3(0.2126, 0.7152, 0.0722));\n" +
        "    float lumaR = dot(r, vec3(0.2126, 0.7152, 0.0722));\n" +
        "\n" +
        "    float neighborMean = (lumaT + lumaB + lumaL + lumaR) * 0.25;\n" +
        "    float diff = lumaC - neighborMean;\n" +
        "    float adaptiveW = sharpness * smoothstep(0.01, 0.1, abs(diff));\n" +
        "    \n" +
        "    vec3 sharpened = c + (c - (t + b + l + r) * 0.25) * adaptiveW;\n" +
        "    \n" +
        "    // 移除所有人工亮度增益\n" +
        "    fragColor = vec4(clamp(sharpened, 0.0, 1.0), 1.0);\n" +
        "}\n";
}
