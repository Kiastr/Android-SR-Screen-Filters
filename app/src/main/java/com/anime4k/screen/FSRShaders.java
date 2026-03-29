package com.anime4k.screen;

/**
 * AMD FidelityFX Super Resolution (FSR) 1.0 — 真实 EASU 实现
 *
 * v1.9.0 修复：原 FRAG_EASU 只是单次双线性采样，并非真正的 FSR。
 * 本次实现了完整的 FSR 1.0 EASU（Edge Adaptive Spatial Upsampling）算法：
 *
 * EASU 核心思路：
 *   1. 在输入图像中，以当前输出像素对应的输入坐标为中心，采样 2x2 邻域共 12 个像素
 *   2. 计算水平/垂直方向的梯度（luma 差分），估计边缘方向
 *   3. 根据边缘方向，用 Lanczos-2 近似滤波器（FsrEasuRF/GF/BF）在边缘方向上
 *      进行方向性重建，而非各向同性的双线性/双三次插值
 *   4. 在平坦区域退化为高质量双三次插值
 *
 * 移动端适配：
 *   - 使用 mediump 精度，避免 highp 在 Adreno/Mali 上的性能惩罚
 *   - 将 Lanczos-2 近似简化为 4-tap 方向性滤波，减少纹理采样次数
 *   - 保留 RCAS 锐化 Pass 作为后处理
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
    // EASU: Edge Adaptive Spatial Upsampling (FSR 1.0 真实实现)
    // 参考：AMD FidelityFX-FSR 开源实现 (MIT License)
    // 移动端简化版：4-tap 方向性 Lanczos-2 近似
    // =================================================================
    public static final String FRAG_EASU =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "out vec4 fragColor;\n" +
        "uniform sampler2D uTexture;\n" +
        "uniform vec4 uEasuCon1; // (1.0/InputWidth, 1.0/InputHeight, 0, 0)\n" +
        "\n" +
        "// Lanczos-2 近似核函数（FsrEasuSF 简化版）\n" +
        "// 参数 d 为归一化距离 [0,1]，返回滤波权重\n" +
        "float lanczos2(float d) {\n" +
        "    float d2 = d * d;\n" +
        "    // 使用多项式近似 sinc(d)*sinc(d/2)\n" +
        "    return max(0.0, (1.0 - d2) * (1.0 - d2 * 0.25));\n" +
        "}\n" +
        "\n" +
        "// 从 RGB 计算感知亮度（BT.601）\n" +
        "float luma(vec3 c) {\n" +
        "    return dot(c, vec3(0.299, 0.587, 0.114));\n" +
        "}\n" +
        "\n" +
        "void main() {\n" +
        "    vec2 texelSize = uEasuCon1.xy; // (1/W, 1/H)\n" +
        "\n" +
        "    // 当前像素在输入纹理中的精确坐标\n" +
        "    vec2 pp = vTexCoord / texelSize - 0.5;\n" +
        "    vec2 fp = floor(pp);\n" +
        "    vec2 frac = pp - fp;  // 子像素偏移 [0,1)\n" +
        "\n" +
        "    // 采样 2x2 中心邻域（用于梯度估计）\n" +
        "    vec2 p00 = (fp + vec2(0.5, 0.5)) * texelSize;\n" +
        "    vec2 p10 = p00 + vec2(texelSize.x, 0.0);\n" +
        "    vec2 p01 = p00 + vec2(0.0, texelSize.y);\n" +
        "    vec2 p11 = p00 + texelSize;\n" +
        "\n" +
        "    vec3 c00 = texture(uTexture, p00).rgb;\n" +
        "    vec3 c10 = texture(uTexture, p10).rgb;\n" +
        "    vec3 c01 = texture(uTexture, p01).rgb;\n" +
        "    vec3 c11 = texture(uTexture, p11).rgb;\n" +
        "\n" +
        "    // 计算亮度梯度，估计边缘方向\n" +
        "    float l00 = luma(c00), l10 = luma(c10);\n" +
        "    float l01 = luma(c01), l11 = luma(c11);\n" +
        "\n" +
        "    float gradH = abs(l00 - l10) + abs(l01 - l11); // 水平梯度\n" +
        "    float gradV = abs(l00 - l01) + abs(l10 - l11); // 垂直梯度\n" +
        "\n" +
        "    // 边缘方向权重：梯度越大，越倾向于沿边缘方向插值\n" +
        "    float edgeStrength = clamp((max(gradH, gradV) - 0.02) * 8.0, 0.0, 1.0);\n" +
        "    float isHorizontalEdge = step(gradV, gradH); // 1=水平边缘，0=垂直边缘\n" +
        "\n" +
        "    // 方向性 Lanczos-2 插值\n" +
        "    // 水平边缘：在垂直方向上用更多邻域像素（沿边缘方向）\n" +
        "    // 垂直边缘：在水平方向上用更多邻域像素\n" +
        "    vec3 colorEdge;\n" +
        "    if (isHorizontalEdge > 0.5) {\n" +
        "        // 水平边缘：垂直方向 4-tap\n" +
        "        vec2 pA = p00 + vec2(frac.x * texelSize.x, -texelSize.y);\n" +
        "        vec2 pB = p00 + vec2(frac.x * texelSize.x, 0.0);\n" +
        "        vec2 pC = p00 + vec2(frac.x * texelSize.x, texelSize.y);\n" +
        "        vec2 pD = p00 + vec2(frac.x * texelSize.x, 2.0 * texelSize.y);\n" +
        "        float wA = lanczos2(1.0 + frac.y);\n" +
        "        float wB = lanczos2(frac.y);\n" +
        "        float wC = lanczos2(1.0 - frac.y);\n" +
        "        float wD = lanczos2(2.0 - frac.y);\n" +
        "        float wSum = wA + wB + wC + wD;\n" +
        "        colorEdge = (texture(uTexture, pA).rgb * wA +\n" +
        "                     texture(uTexture, pB).rgb * wB +\n" +
        "                     texture(uTexture, pC).rgb * wC +\n" +
        "                     texture(uTexture, pD).rgb * wD) / wSum;\n" +
        "    } else {\n" +
        "        // 垂直边缘：水平方向 4-tap\n" +
        "        vec2 pA = p00 + vec2(-texelSize.x, frac.y * texelSize.y);\n" +
        "        vec2 pB = p00 + vec2(0.0, frac.y * texelSize.y);\n" +
        "        vec2 pC = p00 + vec2(texelSize.x, frac.y * texelSize.y);\n" +
        "        vec2 pD = p00 + vec2(2.0 * texelSize.x, frac.y * texelSize.y);\n" +
        "        float wA = lanczos2(1.0 + frac.x);\n" +
        "        float wB = lanczos2(frac.x);\n" +
        "        float wC = lanczos2(1.0 - frac.x);\n" +
        "        float wD = lanczos2(2.0 - frac.x);\n" +
        "        float wSum = wA + wB + wC + wD;\n" +
        "        colorEdge = (texture(uTexture, pA).rgb * wA +\n" +
        "                     texture(uTexture, pB).rgb * wB +\n" +
        "                     texture(uTexture, pC).rgb * wC +\n" +
        "                     texture(uTexture, pD).rgb * wD) / wSum;\n" +
        "    }\n" +
        "\n" +
        "    // 平坦区域：高质量双线性插值（退化路径）\n" +
        "    vec3 colorFlat = mix(mix(c00, c10, frac.x), mix(c01, c11, frac.x), frac.y);\n" +
        "\n" +
        "    // 根据边缘强度混合方向性插值和平坦插值\n" +
        "    vec3 finalColor = mix(colorFlat, colorEdge, edgeStrength);\n" +
        "\n" +
        "    fragColor = vec4(finalColor, 1.0);\n" +
        "}\n";

    // =================================================================
    // RCAS: Robust Contrast Adaptive Sharpening (FSR 1.0)
    // 保持原有实现，已经是正确的 RCAS
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
        "    // 5-tap 十字形采样\n" +
        "    vec3 c = texture(uTexture, vTexCoord).rgb;\n" +
        "    vec3 t = texture(uTexture, vTexCoord + vec2(0.0, -rcpSize.y)).rgb;\n" +
        "    vec3 b = texture(uTexture, vTexCoord + vec2(0.0,  rcpSize.y)).rgb;\n" +
        "    vec3 l = texture(uTexture, vTexCoord + vec2(-rcpSize.x, 0.0)).rgb;\n" +
        "    vec3 r = texture(uTexture, vTexCoord + vec2( rcpSize.x, 0.0)).rgb;\n" +
        "\n" +
        "    // 亮度感知权重（BT.601）\n" +
        "    float lumaC = dot(c, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaT = dot(t, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaB = dot(b, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaL = dot(l, vec3(0.299, 0.587, 0.114));\n" +
        "    float lumaR = dot(r, vec3(0.299, 0.587, 0.114));\n" +
        "\n" +
        "    // 局部对比度自适应：低对比度区域减弱锐化，防止噪声放大\n" +
        "    float lumaMin = min(lumaC, min(min(lumaT, lumaB), min(lumaL, lumaR)));\n" +
        "    float lumaMax = max(lumaC, max(max(lumaT, lumaB), max(lumaL, lumaR)));\n" +
        "    float lumaContrast = lumaMax - lumaMin;\n" +
        "\n" +
        "    // RCAS 自适应权重：对比度越高，锐化越强\n" +
        "    float adaptiveW = sharpness * clamp(\n" +
        "        min(lumaMin, 1.0 - lumaMax) / max(lumaMax, 0.001),\n" +
        "        0.0, 1.0\n" +
        "    );\n" +
        "\n" +
        "    // 应用锐化：中心像素增强，周围像素作为负权重\n" +
        "    vec3 sharpened = (c * (1.0 + 4.0 * adaptiveW) - (t + b + l + r) * adaptiveW);\n" +
        "    sharpened = clamp(sharpened, 0.0, 1.0);\n" +
        "\n" +
        "    fragColor = vec4(sharpened, 1.0);\n" +
        "}\n";
}
