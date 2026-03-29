package com.anime4k.screen;

/**
 * AMD FidelityFX Super Resolution (FSR) 1.0 — 移动端适配实现
 *
 * v1.9.4 调试修复：彻底移除 RCAS 锐化 Pass。
 *
 * 核心发现：
 *   在 FSR 模式下，即便增加了 clamp，只要 RCAS 开启，在半透明叠加层下依然会出现颜色失真。
 *   这说明 RCAS 的 5-tap 十字采样权重计算（中心像素增强，周围像素负权重）在某些
 *   移动端 GPU 驱动上，处理 PixelFormat.TRANSLUCENT 表面时会触发不可控的合成异常。
 *
 * 解决方案：
 *   1. 彻底移除 RCAS Pass，仅保留 EASU。
 *   2. 在 EASU 中强制输出 Alpha = 1.0，完全依靠 WindowManager 的 alpha 混合。
 *   3. 进一步简化 EASU，确保其输出是绝对干净的 RGBA8。
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
        "    // 初始采样\n" +
        "    vec3 cC = clamp(texture(uTexture, vTexCoord).rgb, 0.0, 1.0);\n" +
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
        "    // 强制输出 Alpha=1.0，完全依靠系统层的透明度混合，消除 FBO 内部 Alpha 残留\n" +
        "    fragColor = vec4(clamp(finalColor, 0.0, 1.0), 1.0);\n" +
        "}\n";

    // v1.9.4 彻底移除 RCAS 实现，Passthrough 代替
    public static final String FRAG_RCAS =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "out vec4 fragColor;\n" +
        "uniform sampler2D uTexture;\n" +
        "void main() {\n" +
        "    fragColor = texture(uTexture, vTexCoord);\n" +
        "}\n";
}
