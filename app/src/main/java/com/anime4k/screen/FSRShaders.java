package com.anime4k.screen;

/**
 * AMD FidelityFX Super Resolution (FSR) 1.0 — 移动端叠加层优化版
 *
 * v1.10.1 终极稳定版：
 *   1. 彻底切断残影：配合 Renderer 的 glClear(0,0,0,0) 策略，阻断跨帧反馈。
 *   2. 维持像素级清晰度：保持 v1.10.0 的 1:1 采样对齐，无模糊偏移。
 *   3. 强制 Alpha 隔离：在 EASU 和 MAS 的最后一步强制输出 Alpha=1.0，
 *      防止 FBO 内部残留的透明度数据污染系统合成器。
 *   4. MAS v5：微调边缘增强算子，在保证亮度的前提下提供极致的清晰度。
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
        "    vec2 sampleCoord = vTexCoord;\n" +
        "    \n" +
        "    // 初始采样\n" +
        "    vec3 cC = texture(uTexture, sampleCoord).rgb;\n" +
        "    \n" +
        "    // 梯度探测采样\n" +
        "    vec3 cT = texture(uTexture, clamp(sampleCoord + vec2(0.0, -texelSize.y), 0.0, 1.0)).rgb;\n" +
        "    vec3 cB = texture(uTexture, clamp(sampleCoord + vec2(0.0,  texelSize.y), 0.0, 1.0)).rgb;\n" +
        "    vec3 cL = texture(uTexture, clamp(sampleCoord + vec2(-texelSize.x, 0.0), 0.0, 1.0)).rgb;\n" +
        "    vec3 cR = texture(uTexture, clamp(sampleCoord + vec2( texelSize.x, 0.0), 0.0, 1.0)).rgb;\n" +
        "\n" +
        "    float lC = luma(cC), lT = luma(cT), lB = luma(cB), lL = luma(cL), lR = luma(cR);\n" +
        "    float gradH = abs(lL - lR), gradV = abs(lT - lB);\n" +
        "    float edgeStrength = clamp(max(gradH, gradV) * 5.0, 0.0, 1.0);\n" +
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
        "    // [FIX-v1.10.1] 强制 Alpha=1.0 隔离\n" +
        "    fragColor = vec4(clamp(mix(cC, colorEdge, edgeStrength), 0.0, 1.0), 1.0);\n" +
        "}\n";

    // =================================================================
    // MAS v5: Mobile-Adaptive Sharpening (Ultra-Clear Stable)
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
        "    float sharpness = uRcasCon.x * 0.7;\n" +
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
        "    // [FIX-v1.10.1] 强制 Alpha=1.0 隔离\n" +
        "    fragColor = vec4(clamp(sharpened, 0.0, 1.0), 1.0);\n" +
        "}\n";
}
