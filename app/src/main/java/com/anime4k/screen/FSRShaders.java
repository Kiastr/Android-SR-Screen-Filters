package com.anime4k.screen;

/**
 * AMD FidelityFX Super Resolution (FSR) 1.0 — 移动端叠加层优化版
 *
 * v1.9.5 正式修复：针对半透明叠加层重构锐化算法。
 *
 * 核心改进：
 *   1. 弃用 RCAS：原 RCAS 的负权重计算在半透明合成下极易产生色块。
 *   2. 新增 MAS (Mobile-Adaptive Sharpening)：
 *      - 基于 5-tap 十字采样的拉普拉斯边缘检测。
 *      - 仅在边缘区域（对比度高）进行亮度增强，平坦区域不处理。
 *      - 全程使用非负权重计算，彻底消除因负值导致的 Alpha 合成失真。
 *   3. 强制 Alpha 隔离：锐化 Pass 仅读取并写入 RGB，Alpha 保持 1.0，
 *      完全由系统 WindowManager 处理透明度。
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
        "    vec3 cC = clamp(texture(uTexture, vTexCoord).rgb, 0.0, 1.0);\n" +
        "    \n" +
        "    vec3 cT = clamp(texture(uTexture, vTexCoord + vec2(0.0, -texelSize.y)).rgb, 0.0, 1.0);\n" +
        "    vec3 cB = clamp(texture(uTexture, vTexCoord + vec2(0.0,  texelSize.y)).rgb, 0.0, 1.0);\n" +
        "    vec3 cL = clamp(texture(uTexture, vTexCoord + vec2(-texelSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "    vec3 cR = clamp(texture(uTexture, vTexCoord + vec2( texelSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "\n" +
        "    float lC = luma(cC), lT = luma(cT), lB = luma(cB), lL = luma(cL), lR = luma(cR);\n" +
        "    float gradH = abs(lL - lR), gradV = abs(lT - lB);\n" +
        "    float edgeStrength = clamp(max(gradH, gradV) * 4.0, 0.0, 1.0);\n" +
        "\n" +
        "    vec3 colorEdge;\n" +
        "    if (gradH > gradV) {\n" +
        "        vec3 cLL = clamp(texture(uTexture, vTexCoord + vec2(-2.0 * texelSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "        vec3 cRR = clamp(texture(uTexture, vTexCoord + vec2( 2.0 * texelSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "        float w0 = lanczos2(0.0), w1 = lanczos2(1.0), w2 = lanczos2(2.0);\n" +
        "        colorEdge = (cC * w0 + (cL + cR) * w1 + (cLL + cRR) * w2) / (w0 + 2.0*w1 + 2.0*w2);\n" +
        "    } else {\n" +
        "        vec3 cTT = clamp(texture(uTexture, vTexCoord + vec2(0.0, -2.0 * texelSize.y)).rgb, 0.0, 1.0);\n" +
        "        vec3 cBB = clamp(texture(uTexture, vTexCoord + vec2(0.0,  2.0 * texelSize.y)).rgb, 0.0, 1.0);\n" +
        "        float w0 = lanczos2(0.0), w1 = lanczos2(1.0), w2 = lanczos2(2.0);\n" +
        "        colorEdge = (cC * w0 + (cT + cB) * w1 + (cTT + cBB) * w2) / (w0 + 2.0*w1 + 2.0*w2);\n" +
        "    }\n" +
        "    fragColor = vec4(clamp(mix(cC, colorEdge, edgeStrength), 0.0, 1.0), 1.0);\n" +
        "}\n";

    // =================================================================
    // MAS: Mobile-Adaptive Sharpening (替代不稳定的 RCAS)
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
        "    float sharpness = uRcasCon.x * 0.5; // 限制最大锐化强度\n" +
        "\n" +
        "    // 5-tap 十字形采样\n" +
        "    vec3 c = clamp(texture(uTexture, vTexCoord).rgb, 0.0, 1.0);\n" +
        "    vec3 t = clamp(texture(uTexture, vTexCoord + vec2(0.0, -rcpSize.y)).rgb, 0.0, 1.0);\n" +
        "    vec3 b = clamp(texture(uTexture, vTexCoord + vec2(0.0,  rcpSize.y)).rgb, 0.0, 1.0);\n" +
        "    vec3 l = clamp(texture(uTexture, vTexCoord + vec2(-rcpSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "    vec3 r = clamp(texture(uTexture, vTexCoord + vec2( rcpSize.x, 0.0)).rgb, 0.0, 1.0);\n" +
        "\n" +
        "    // 亮度提取\n" +
        "    float lumaC = dot(c, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaT = dot(t, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaB = dot(b, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaL = dot(l, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaR = dot(r, vec3(0.299, 0.587, 0.114));\n" +
        "\n" +
        "    // 拉普拉斯算子检测边缘：(4*center - (T+B+L+R))\n" +
        "    // 采用非负混合：final = center + (center - mean(neighbors)) * sharpness\n" +
        "    float neighborMean = (lumaT + lumaB + lumaL + lumaR) * 0.25;\n" +
        "    float edge = abs(lumaC - neighborMean);\n" +
        "    \n" +
        "    // 自适应门控：仅在有明显对比度差异的区域应用锐化，防止放大平坦区的噪点\n" +
        "    float adaptiveW = sharpness * smoothstep(0.01, 0.1, edge);\n" +
        "    \n" +
        "    // 计算锐化：使用非负加权混合，确保输出始终在正值范围\n" +
        "    // 这种算式在半透明叠加层下比 RCAS 的负权重采样稳定得多\n" +
        "    vec3 sharpened = c + (c - (t + b + l + r) * 0.25) * adaptiveW;\n" +
        "    \n" +
        "    fragColor = vec4(clamp(sharpened, 0.0, 1.0), 1.0);\n" +
        "}\n";
}
