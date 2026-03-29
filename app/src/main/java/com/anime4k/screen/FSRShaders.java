package com.anime4k.screen;

/**
 * AMD FidelityFX Super Resolution (FSR) 1.0 — 移动端适配实现
 *
 * v1.9.3 彻底修复：FSR 模式在半透明叠加层下的颜色失真和色块。
 *
 * 核心改进：
 *   1. 增加防御性计算：在 RCAS 锐化权重计算中增加 epsilon 防止除以零，
 *      并对输入像素进行 initial clamp，防止 NaN 或极值污染后续计算。
 *   2. 严格 Alpha 隔离：确保 EASU 和 RCAS 过程不破坏原始 Alpha 通道，
 *      且输出 Alpha 始终限制在 [0, 1]，防止系统合成器崩溃产生色块。
 *   3. 提高 mediump 稳定性：通过重新组织算式，减少中间溢出风险。
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
        "    return dot(c, vec3(0.299, 0.587, 0.114));\n" +
        "}\n" +
        "\n" +
        "void main() {\n" +
        "    vec2 texelSize = uEasuCon1.xy;\n" +
        "\n" +
        "    // 初始采样并进行安全限制，防止异常值输入\n" +
        "    vec4 tC = texture(uTexture, vTexCoord);\n" +
        "    vec3 cC = clamp(tC.rgb, 0.0, 1.0);\n" +
        "    \n" +
        "    // 5-tap 采样用于梯度估计\n" +
        "    vec3 cT = clamp(texture(uTexture, vTexCoord + vec2(0.0, -texelSize.y)).rgb, 0.0, 1.0);\n" +
        "    vec3 cB = clamp(texture(uTexture, vTexCoord + vec2(0.0,  texelSize.y)).rgb, 0.0, 1.0);\n" +
        "    vec3 cL = clamp(texture(uTexture, vTexCoord + vec2(-texelSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "    vec3 cR = clamp(texture(uTexture, vTexCoord + vec2( texelSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "\n" +
        "    float lC = luma(cC);\n" +
        "    float lT = luma(cT);\n" +
        "    float lB = luma(cB);\n" +
        "    float lL = luma(cL);\n" +
        "    float lR = luma(cR);\n" +
        "\n" +
        "    float gradH = abs(lL - lR);\n" +
        "    float gradV = abs(lT - lB);\n" +
        "\n" +
        "    float edgeStrength = clamp(max(gradH, gradV) * 4.0, 0.0, 1.0);\n" +
        "\n" +
        "    vec3 colorEdge;\n" +
        "    if (gradH > gradV) {\n" +
        "        vec3 cLL = clamp(texture(uTexture, vTexCoord + vec2(-2.0 * texelSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "        vec3 cRR = clamp(texture(uTexture, vTexCoord + vec2( 2.0 * texelSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "        float w0 = lanczos2(0.0);\n" +
        "        float w1 = lanczos2(1.0);\n" +
        "        float w2 = lanczos2(2.0);\n" +
        "        float wSum = w0 + 2.0 * w1 + 2.0 * w2;\n" +
        "        colorEdge = (cC * w0 + (cL + cR) * w1 + (cLL + cRR) * w2) / wSum;\n" +
        "    } else {\n" +
        "        vec3 cTT = clamp(texture(uTexture, vTexCoord + vec2(0.0, -2.0 * texelSize.y)).rgb, 0.0, 1.0);\n" +
        "        vec3 cBB = clamp(texture(uTexture, vTexCoord + vec2(0.0,  2.0 * texelSize.y)).rgb, 0.0, 1.0);\n" +
        "        float w0 = lanczos2(0.0);\n" +
        "        float w1 = lanczos2(1.0);\n" +
        "        float w2 = lanczos2(2.0);\n" +
        "        float wSum = w0 + 2.0 * w1 + 2.0 * w2;\n" +
        "        colorEdge = (cC * w0 + (cT + cB) * w1 + (cTT + cBB) * w2) / wSum;\n" +
        "    }\n" +
        "\n" +
        "    vec3 finalColor = mix(cC, colorEdge, edgeStrength);\n" +
        "    fragColor = vec4(clamp(finalColor, 0.0, 1.0), clamp(tC.a, 0.0, 1.0));\n" +
        "}\n";

    // =================================================================
    // RCAS: Robust Contrast Adaptive Sharpening (FSR 1.0)
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
        "    float sharpness = uRcasCon.x;\n" +
        "\n" +
        "    // 5-tap 十字形采样，强制限制在 [0,1] 防止异常值扩散\n" +
        "    vec4 tC = texture(uTexture, vTexCoord);\n" +
        "    vec3 c = clamp(tC.rgb, 0.0, 1.0);\n" +
        "    vec3 t = clamp(texture(uTexture, vTexCoord + vec2(0.0, -rcpSize.y)).rgb, 0.0, 1.0);\n" +
        "    vec3 b = clamp(texture(uTexture, vTexCoord + vec2(0.0,  rcpSize.y)).rgb, 0.0, 1.0);\n" +
        "    vec3 l = clamp(texture(uTexture, vTexCoord + vec2(-rcpSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "    vec3 r = clamp(texture(uTexture, vTexCoord + vec2( rcpSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "\n" +
        "    float lumaC = dot(c, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaT = dot(t, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaB = dot(b, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaL = dot(l, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaR = dot(r, vec3(0.299, 0.587, 0.114));\n" +
        "\n" +
        "    float lumaMin = min(lumaC, min(min(lumaT, lumaB), min(lumaL, lumaR)));\n" +
        "    float lumaMax = max(lumaC, max(max(lumaT, lumaB), max(lumaL, lumaR)));\n" +
        "\n" +
        "    // RCAS 权重计算增加 epsilon，防止 max(lumaMax, 0.001) 除以零导致 NaN\n" +
        "    float adaptiveW = sharpness * clamp(\n" +
        "        min(lumaMin, 1.0 - lumaMax) / (lumaMax + 0.0001),\n" +
        "        0.0, 1.0\n" +
        "    );\n" +
        "\n" +
        "    // 锐化计算\n" +
        "    vec3 sharpened = c + (c - (t + b + l + r) * 0.25) * adaptiveW;\n" +
        "    \n" +
        "    // 最终输出：双重 clamp 保护，确保 Alpha 通道不被破坏\n" +
        "    fragColor = vec4(clamp(sharpened, 0.0, 1.0), clamp(tC.a, 0.0, 1.0));\n" +
        "}\n";
}
