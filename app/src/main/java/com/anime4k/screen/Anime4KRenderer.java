package com.anime4k.screen;

import android.graphics.Bitmap;
import android.opengl.GLES30;
import android.opengl.GLUtils;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Anime4K renderer using OpenGL ES 3.0.
 * Supports Anime4K-v3.2 and AMD FSR 1.0.
 *
 * =====================================================================
 * Pseudo-MV Frame Interpolation v1.5.0
 * =====================================================================
 * 将 Anime4K 模式下的插帧算法从 ATW（异步时间扭曲）升级为
 * 基于 Anime4K 边缘梯度信息的伪运动矢量插值（Pseudo-MV）。
 *
 * 算法原理：
 * Anime4K 管线在处理每一帧时，会产出两张关键的中间纹理：
 *   - lumadTexture  (R=Sobel边缘强度, G=细化权重 dval)
 *   - edgeDirTex    (R=归一化边缘方向X, G=归一化边缘方向Y)  [即原 tempTex2]
 *
 * 边缘方向场（Edge Direction Field）的法线方向，在物理上近似于
 * 图像中运动物体的局部运动方向。Pseudo-MV 算法利用这一特性，
 * 通过三个 Shader Pass 实现运动感知插帧：
 *
 * Pass A — 运动矢量估计 (Motion Vector Estimation):
 *   对每个像素，从 lumadTexture 读取边缘强度，从 edgeDirTex 读取边缘方向。
 *   将边缘法线方向（垂直于边缘切线）作为该像素的伪运动矢量（PMV）。
 *   对非边缘区域（dval < threshold），使用邻域边缘矢量的加权扩散填充，
 *   确保运动矢量场的连续性（避免运动空洞）。
 *   输出：mvTexture (RG=运动矢量, B=置信度)
 *
 * Pass B — 前向翘曲 (Forward Warp):
 *   使用 t=0.5 的插值时刻，将当前帧（T）沿 PMV 方向向前翘曲 0.5 步，
 *   得到 T+0.5 时刻的前向预测帧。
 *   输出：warpFwdTex
 *
 * Pass C — 后向翘曲 + 双向混合 (Backward Warp + Bidirectional Blend):
 *   将上一帧（T-1）沿 PMV 反方向向后翘曲 0.5 步，得到后向预测帧。
 *   对两个预测帧按置信度加权混合，并用边缘强度在混合帧与原帧之间
 *   进行自适应融合（边缘区域更信任插值，平坦区域保留原始颜色）。
 *   输出：最终插帧结果
 *
 * 优势对比 ATW：
 *   ATW 使用固定的全局偏移量对上一帧进行简单平移，无法感知局部运动，
 *   在运动方向多变的场景中会产生模糊和鬼影。
 *   Pseudo-MV 利用 Anime4K 已计算好的边缘方向场，为每个像素生成
 *   独立的运动矢量，实现逐像素的精确运动补偿，显著减少运动模糊。
 *
 * 性能开销：
 *   相比 ATW（1 个 Pass），Pseudo-MV 增加了 3 个 Pass（A/B/C），
 *   但 Pass A/B 分辨率可降至输出分辨率的 1/2（MV 场不需要全分辨率），
 *   整体开销可控，在中端设备上仍能维持 30fps 以上。
 *
 * 插帧策略：
 *   - Anime4K 模式：使用 Pseudo-MV（边缘引导伪运动矢量插值）
 *   - FSR 模式：降级使用 ATW（异步时间扭曲），因 FSR 不产出边缘方向场
 *
 * 性能优化继承自 v1.4.0：
 * [OPT-1] VBO/VAO  [OPT-3] Uniform 预缓存  [OPT-4] 预分配纹理
 * [OPT-5] FBO 固定绑定  [OPT-6] Shader 精度分级
 */
public class Anime4KRenderer {

    private static final String TAG = "Anime4KRenderer";

    private static final float[] QUAD_VERTICES = {
        -1f, -1f, 0f, 0f,
         1f, -1f, 1f, 0f,
        -1f,  1f, 0f, 1f,
         1f,  1f, 1f, 1f,
    };
    private static final float[] QUAD_VERTICES_FLIPPED = {
        -1f, -1f, 0f, 1f,
         1f, -1f, 1f, 1f,
        -1f,  1f, 0f, 0f,
         1f,  1f, 1f, 0f,
    };

    // VBO / VAO
    private int[] vbo = new int[2];
    private int[] vao = new int[2];

    // ---- Shader 程序 ----
    // Anime4K passes
    private int programLuma, programGradX1, programGradY1;
    private int programGradX2, programGradY2, programApply;
    // Pseudo-MV passes（Anime4K 模式专用）
    private int programPMV_Estimate;  // Pass A: 运动矢量估计
    private int programPMV_WarpFwd;   // Pass B: 前向翘曲
    private int programPMV_Blend;     // Pass C: 后向翘曲 + 双向混合
    // ATW（FSR 模式专用）
    private int programATW;
    // 辅助
    private int programPassthrough;
    // FSR
    private int programFsrEasu, programFsrRcas;

    // ---- FBO 索引说明 ----
    // [0]=luma  [1]=tempTex(gradX1)  [2]=lumad  [3]=lumamm(gradX2)
    // [4]=output  [5]=lastOutput  [6]=fsrTemp  [7]=edgeDir(gradY2)
    // [8]=mvTex  [9]=warpFwd
    private int[] fbo = new int[10];

    // ---- 纹理 ----
    private int inputTexture;
    private int lumaTexture;
    private int lumadTexture;   // R=Sobel强度, G=细化权重
    private int lumammTexture;  // gradX2 中间纹理
    private int outputTexture;
    private int lastOutputTexture;
    private int fsrTempTexture;
    private int tempTex;        // gradX1 临时
    private int edgeDirTex;     // R=边缘方向X, G=边缘方向Y（原 tempTex2）
    // Pseudo-MV 专用纹理
    private int mvTexture;      // RG=运动矢量, B=置信度
    private int warpFwdTex;     // 前向翘曲结果

    private int inputWidth, inputHeight;
    private int outputWidth, outputHeight;
    // MV 场使用半分辨率，节省 GPU 带宽
    private int mvWidth, mvHeight;

    private float refineStrength  = 0.5f;
    private float pmvStrength     = 0.5f;  // Pseudo-MV 插帧强度 (0=关闭, 1=最强)
    private boolean fsrEnabled    = false;
    private float fsrSharpness    = 0.2f;

    private boolean initialized = false;

    // ---- 预缓存 Uniform Location ----
    private int uLuma_texture;
    private int uGradX1_luma, uGradX1_texelSize;
    private int uGradY1_grad, uGradY1_texelSize, uGradY1_refineStrength;
    private int uGradX2_lumad, uGradX2_texelSize;
    private int uGradY2_lumad, uGradY2_lumamm, uGradY2_texelSize;
    private int uApply_texture, uApply_lumad, uApply_lumamm, uApply_texelSize;
    private int uPassthrough_texture;
    private int uEasu_texture, uEasu_con1;
    private int uRcas_texture, uRcas_con;
    // Pseudo-MV uniforms（Anime4K 模式）
    private int uPMV_Est_lumad, uPMV_Est_edgeDir, uPMV_Est_texelSize, uPMV_Est_strength;
    private int uPMV_Fwd_current, uPMV_Fwd_mvTex, uPMV_Fwd_texelSize;
    private int uPMV_Blend_current, uPMV_Blend_last, uPMV_Blend_mvTex;
    private int uPMV_Blend_lumad, uPMV_Blend_texelSize, uPMV_Blend_strength;
    // ATW uniforms（FSR 模式）
    private int uATW_texture, uATW_lastTexture, uATW_strength, uATW_offset;

    // =================================================================
    // Shaders
    // =================================================================

    private static final String VERTEX_SHADER =
        "#version 300 es\n" +
        "in vec4 aPosition;\n" +
        "in vec2 aTexCoord;\n" +
        "out vec2 vTexCoord;\n" +
        "void main() {\n" +
        "    gl_Position = aPosition;\n" +
        "    vTexCoord = aTexCoord;\n" +
        "}\n";

    // Pass 1: Luma
    private static final String FRAG_LUMA =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uTexture;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    vec3 c = texture(uTexture, vTexCoord).rgb;\n" +
        "    float luma = dot(vec3(0.299, 0.587, 0.114), c);\n" +
        "    fragColor = vec4(luma, 0.0, 0.0, 1.0);\n" +
        "}\n";

    // Pass 2: GradX1
    private static final String FRAG_GRAD_X1 =
        "#version 300 es\n" +
        "precision lowp float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uLuma;\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    float l = texture(uLuma, vTexCoord + vec2(-uTexelSize.x, 0.0)).r;\n" +
        "    float c = texture(uLuma, vTexCoord).r;\n" +
        "    float r = texture(uLuma, vTexCoord + vec2(uTexelSize.x, 0.0)).r;\n" +
        "    fragColor = vec4((-l+r)*0.5+0.5, (l+c+c+r)*0.25+0.5, 0.0, 1.0);\n" +
        "}\n";

    // Pass 3: GradY1
    private static final String FRAG_GRAD_Y1 =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uGrad;\n" +
        "uniform vec2 uTexelSize;\n" +
        "uniform float uRefineStrength;\n" +
        "float power_function(float x) {\n" +
        "    float x2=x*x; float x3=x2*x; float x4=x2*x2; float x5=x2*x3;\n" +
        "    return 11.68129591*x5-42.46906057*x4+60.28286266*x3\n" +
        "          -41.84451327*x2+14.05517353*x-1.081521930;\n" +
        "}\n" +
        "void main() {\n" +
        "    float tx=texture(uGrad,vTexCoord+vec2(0.,-uTexelSize.y)).r*2.-1.;\n" +
        "    float cx=texture(uGrad,vTexCoord).r*2.-1.;\n" +
        "    float bx=texture(uGrad,vTexCoord+vec2(0.,uTexelSize.y)).r*2.-1.;\n" +
        "    float ty=texture(uGrad,vTexCoord+vec2(0.,-uTexelSize.y)).g*4.-2.;\n" +
        "    float by=texture(uGrad,vTexCoord+vec2(0.,uTexelSize.y)).g*4.-2.;\n" +
        "    float xg=(tx+cx+cx+bx); float yg=(-ty+by);\n" +
        "    float sobel=clamp(sqrt(xg*xg+yg*yg),0.,1.);\n" +
        "    float dval=clamp(power_function(sobel)*uRefineStrength,0.,1.);\n" +
        "    gl_FragColor=vec4(sobel*0.5+0.5,dval,0.,1.);\n" +
        "}\n";

    // Pass 4: GradX2
    private static final String FRAG_GRAD_X2 =
        "#version 300 es\n" +
        "precision lowp float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uLumad;\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    float dval=texture(uLumad,vTexCoord).g;\n" +
        "    if(dval<0.1){fragColor=vec4(0.5,0.5,0.,1.);return;}\n" +
        "    float s=texture(uLumad,vTexCoord).r*2.-1.;\n" +
        "    float l=texture(uLumad,vTexCoord+vec2(-uTexelSize.x,0.)).r*2.-1.;\n" +
        "    float r=texture(uLumad,vTexCoord+vec2(uTexelSize.x,0.)).r*2.-1.;\n" +
        "    fragColor=vec4((-l+r)*0.5+0.5,(l+s+s+r)*0.25+0.5,0.,1.);\n" +
        "}\n";

    // Pass 5: GradY2 — 输出归一化边缘方向场（edgeDirTex）
    private static final String FRAG_GRAD_Y2 =
        "#version 300 es\n" +
        "precision lowp float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uLumad;\n" +
        "uniform sampler2D uLumamm;\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    float dval=texture(uLumad,vTexCoord).g;\n" +
        "    if(dval<0.1){fragColor=vec4(0.5,0.5,0.,1.);return;}\n" +
        "    float tx=texture(uLumamm,vTexCoord+vec2(0.,-uTexelSize.y)).r*2.-1.;\n" +
        "    float cx=texture(uLumamm,vTexCoord).r*2.-1.;\n" +
        "    float bx=texture(uLumamm,vTexCoord+vec2(0.,uTexelSize.y)).r*2.-1.;\n" +
        "    float ty=texture(uLumamm,vTexCoord+vec2(0.,-uTexelSize.y)).g*4.-2.;\n" +
        "    float by=texture(uLumamm,vTexCoord+vec2(0.,uTexelSize.y)).g*4.-2.;\n" +
        "    float xg=(tx+cx+cx+bx); float yg=(-ty+by);\n" +
        "    float norm=sqrt(xg*xg+yg*yg);\n" +
        "    if(norm<=0.001){fragColor=vec4(0.5,0.5,0.,1.);return;}\n" +
        "    fragColor=vec4(xg/norm*0.5+0.5,yg/norm*0.5+0.5,0.,1.);\n" +
        "}\n";

    // Pass 6: Apply
    private static final String FRAG_APPLY =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uTexture;\n" +
        "uniform sampler2D uLumad;\n" +
        "uniform sampler2D uLumamm;\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    float dval=texture(uLumad,vTexCoord).g;\n" +
        "    vec4 orig=texture(uTexture,vTexCoord);\n" +
        "    if(dval<0.1){fragColor=orig;return;}\n" +
        "    vec2 dir=texture(uLumamm,vTexCoord).rg*2.-1.;\n" +
        "    if(abs(dir.x)+abs(dir.y)<=0.0001){fragColor=orig;return;}\n" +
        "    float xp=-sign(dir.x); float yp=-sign(dir.y);\n" +
        "    vec4 xv=texture(uTexture,vTexCoord+vec2(uTexelSize.x*xp,0.));\n" +
        "    vec4 yv=texture(uTexture,vTexCoord+vec2(0.,uTexelSize.y*yp));\n" +
        "    float r=abs(dir.x)/(abs(dir.x)+abs(dir.y));\n" +
        "    fragColor=mix(yv,xv,r)*dval+orig*(1.-dval);\n" +
        "}\n";

    // ATW（FSR 模式专用）— mediump
    private static final String FRAG_ATW =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uTexture;\n" +
        "uniform sampler2D uLastTexture;\n" +
        "uniform float uATWStrength;\n" +
        "uniform vec2 uOffset;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    vec4 current = texture(uTexture, vTexCoord);\n" +
        "    if (uATWStrength <= 0.0) { fragColor = current; return; }\n" +
        "    vec4 last = texture(uLastTexture, vTexCoord + uOffset * uATWStrength);\n" +
        "    fragColor = mix(current, last, 0.5 * uATWStrength);\n" +
        "}\n";

    // Passthrough
    private static final String FRAG_PASSTHROUGH =
        "#version 300 es\n" +
        "precision lowp float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uTexture;\n" +
        "out vec4 fragColor;\n" +
        "void main() { fragColor = texture(uTexture, vTexCoord); }\n";

    // =================================================================
    // Pseudo-MV Shaders
    // =================================================================

    /**
     * Pass A — 运动矢量估计 (Motion Vector Estimation)
     *
     * 核心思路：
     *   边缘方向场（edgeDirTex）存储的是归一化的边缘切线方向。
     *   运动方向（法线方向）= 旋转 90° = (-edgeDir.y, edgeDir.x)。
     *   对于强边缘像素（dval >= threshold），直接使用法线方向作为 PMV。
     *   对于弱边缘/平坦区域，通过 3x3 邻域加权扩散填充 PMV，
     *   权重由邻域像素的边缘强度（sobel）决定，确保 MV 场连续。
     *
     * 输出编码：
     *   R = mv.x * 0.5 + 0.5  ([-1,1] → [0,1])
     *   G = mv.y * 0.5 + 0.5
     *   B = confidence (边缘置信度，用于后续混合权重)
     *   A = 1.0
     */
    private static final String FRAG_PMV_ESTIMATE =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uLumad;    // R=sobel, G=dval\n" +
        "uniform sampler2D uEdgeDir;  // R=edgeX, G=edgeY (归一化，编码为[0,1])\n" +
        "uniform vec2 uTexelSize;\n" +
        "uniform float uStrength;     // PMV 强度缩放 [0,1]\n" +
        "out vec4 fragColor;\n" +
        "\n" +
        "// 从编码纹理解码运动矢量（切线→法线旋转90°）\n" +
        "vec2 edgeToMotion(vec2 uv) {\n" +
        "    vec2 edgeDir = texture(uEdgeDir, uv).rg * 2.0 - 1.0;\n" +
        "    // 法线方向 = 旋转 90°\n" +
        "    return vec2(-edgeDir.y, edgeDir.x);\n" +
        "}\n" +
        "\n" +
        "void main() {\n" +
        "    vec2 lumadVal = texture(uLumad, vTexCoord).rg;\n" +
        "    float sobel = lumadVal.r * 2.0 - 1.0;  // [-1,1]\n" +
        "    float dval  = lumadVal.g;               // [0,1]\n" +
        "\n" +
        "    vec2 mv;\n" +
        "    float confidence;\n" +
        "\n" +
        "    if (dval >= 0.15) {\n" +
        "        // 强边缘区域：直接使用边缘法线作为运动矢量\n" +
        "        mv = edgeToMotion(vTexCoord);\n" +
        "        confidence = dval;\n" +
        "    } else {\n" +
        "        // 弱边缘/平坦区域：3x3 邻域加权扩散\n" +
        "        vec2 mvSum = vec2(0.0);\n" +
        "        float wSum = 0.0;\n" +
        "        for (int dy = -1; dy <= 1; dy++) {\n" +
        "            for (int dx = -1; dx <= 1; dx++) {\n" +
        "                if (dx == 0 && dy == 0) continue;\n" +
        "                vec2 offset = vec2(float(dx), float(dy)) * uTexelSize;\n" +
        "                vec2 nUV = vTexCoord + offset;\n" +
        "                float nDval = texture(uLumad, nUV).g;\n" +
        "                if (nDval >= 0.15) {\n" +
        "                    // 距离权重：中心邻居权重更高\n" +
        "                    float distW = (abs(float(dx)) + abs(float(dy)) == 1.0) ? 1.0 : 0.707;\n" +
        "                    float w = nDval * distW;\n" +
        "                    mvSum += edgeToMotion(nUV) * w;\n" +
        "                    wSum += w;\n" +
        "                }\n" +
        "            }\n" +
        "        }\n" +
        "        if (wSum > 0.001) {\n" +
        "            mv = mvSum / wSum;\n" +
        "            confidence = wSum / 8.0 * 0.5; // 扩散区域置信度较低\n" +
        "        } else {\n" +
        "            mv = vec2(0.0);\n" +
        "            confidence = 0.0;\n" +
        "        }\n" +
        "    }\n" +
        "\n" +
        "    // 按强度缩放 MV（控制插帧幅度）\n" +
        "    mv *= uStrength;\n" +
        "    confidence *= uStrength;\n" +
        "\n" +
        "    // 编码输出：MV 映射到 [0,1]，置信度直接存 B 通道\n" +
        "    fragColor = vec4(mv * 0.5 + 0.5, confidence, 1.0);\n" +
        "}\n";

    /**
     * Pass B — 前向翘曲 (Forward Warp, t=0.5)
     *
     * 对当前帧（T）的每个像素，沿 PMV 方向向前移动 0.5 步，
     * 采样当前帧在 (uv + mv * 0.5) 处的颜色，生成前向预测帧。
     * 这等价于：假设运动在 [T, T+1] 之间匀速，在 T+0.5 时刻
     * 物体已移动了半个 MV 的距离。
     */
    private static final String FRAG_PMV_WARP_FWD =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uCurrent;  // 当前增强帧 T\n" +
        "uniform sampler2D uMvTex;    // 运动矢量场\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    // 解码运动矢量（[0,1] → [-1,1]，再转为纹理坐标偏移）\n" +
        "    vec3 mvData = texture(uMvTex, vTexCoord).rgb;\n" +
        "    vec2 mv = (mvData.rg * 2.0 - 1.0) * uTexelSize * 8.0; // 8像素最大位移\n" +
        "    float conf = mvData.b;\n" +
        "\n" +
        "    // 前向翘曲：沿 MV 方向采样当前帧\n" +
        "    vec2 fwdUV = vTexCoord + mv * 0.5;\n" +
        "    fwdUV = clamp(fwdUV, vec2(0.0), vec2(1.0));\n" +
        "    vec4 fwdColor = texture(uCurrent, fwdUV);\n" +
        "\n" +
        "    // 将置信度存入 Alpha，供 Pass C 使用\n" +
        "    fragColor = vec4(fwdColor.rgb, conf);\n" +
        "}\n";

    /**
     * Pass C — 后向翘曲 + 双向自适应混合 + 时域差分抗拖影 (Anti-Ghosting)
     *
     * LSFG 核心思想移植：
     * LSFG 通过光流算法计算真实运动矢量，对剧烈运动区域不进行帧混合而是直接输出当前帧。
     * 本 Shader 实现类似的“运动感知抗拖影”机制：
     *
     * 1. 后向翘曲：对上一帧（T-1）在 (uv - mv * 0.5) 处采样。
     * 2. 双向混合：前向预测帧和后向预测帧按置信度加权平均。
     * 3. 时域差分抗拖影 (Temporal Diff Anti-Ghosting) [LSFG 核心]:
     *    计算当前帧与上一帧的像素亮度差异（时域差分）。
     *    - 差异大（剧烈运动）：运动矢量不可靠，拒绝混合，直接输出当前帧。彻底消除拖影。
     *    - 差异小（平缓运动或静止）：插値平滑有效，进行混合提升流畅感。
     * 4. 自适应融合：边缘强度 + 置信度 + 时域差分共同决定最终混合权重。
     */
    private static final String FRAG_PMV_BLEND =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uCurrent;   // 当前增强帧 T\n" +
        "uniform sampler2D uLast;      // 上一增强帧 T-1\n" +
        "uniform sampler2D uMvTex;     // 运动矢量场\n" +
        "uniform sampler2D uLumad;     // 边缘强度\n" +
        "uniform vec2 uTexelSize;\n" +
        "uniform float uStrength;      // 插帧强度 [0,1]\n" +
        "out vec4 fragColor;\n" +
        "\n" +
        "// 计算两个颜色的感知亮度差异\n" +
        "float lumaDiff(vec4 a, vec4 b) {\n" +
        "    float la = dot(a.rgb, vec3(0.299, 0.587, 0.114));\n" +
        "    float lb = dot(b.rgb, vec3(0.299, 0.587, 0.114));\n" +
        "    return abs(la - lb);\n" +
        "}\n" +
        "\n" +
        "void main() {\n" +
        "    vec3 mvData = texture(uMvTex, vTexCoord).rgb;\n" +
        "    vec2 mv = (mvData.rg * 2.0 - 1.0) * uTexelSize * 8.0;\n" +
        "    float conf = mvData.b;\n" +
        "\n" +
        "    vec4 current = texture(uCurrent, vTexCoord);\n" +
        "\n" +
        "    if (conf < 0.05 || uStrength < 0.01) {\n" +
        "        fragColor = current;\n" +
        "        return;\n" +
        "    }\n" +
        "\n" +
        "    // 前向翘曲预测\n" +
        "    vec2 fwdUV = clamp(vTexCoord + mv * 0.5, vec2(0.0), vec2(1.0));\n" +
        "    vec4 fwdColor = texture(uCurrent, fwdUV);\n" +
        "\n" +
        "    // 后向翘曲预测\n" +
        "    vec2 bwdUV = clamp(vTexCoord - mv * 0.5, vec2(0.0), vec2(1.0));\n" +
        "    vec4 bwdColor = texture(uLast, bwdUV);\n" +
        "\n" +
        "    // 双向混合\n" +
        "    vec4 blended = mix(bwdColor, fwdColor, 0.5);\n" +
        "\n" +
        "    // ===== 时域差分抗拖影 (LSFG 核心思想) =====\n" +
        "    // 计算当前帧与上一帧在当前位置的亮度差异\n" +
        "    vec4 lastAtCurrent = texture(uLast, vTexCoord);\n" +
        "    float temporalDiff = lumaDiff(current, lastAtCurrent);\n" +
        "\n" +
        "    // 时域差异越大，运动越剧烈，插値可靠性越低\n" +
        "    // threshold=0.12: 超过此差异的区域被认为剧烈运动，直接输出当前帧\n" +
        "    float antiGhostFactor = 1.0 - smoothstep(0.08, 0.20, temporalDiff);\n" +
        "\n" +
        "    // 边缘权重\n" +
        "    float edgeW = clamp(texture(uLumad, vTexCoord).r * 2.0 - 1.0, 0.0, 1.0);\n" +
        "\n" +
        "    // 最终混合权重 = 置信度 * 边缘权重 * 抗拖影因子 * 用户强度\n" +
        "    // antiGhostFactor 在剧烈运动时趋近 0，强制使用当前帧，彻底消除拖影\n" +
        "    float blendW = conf * (0.4 + 0.6 * edgeW) * antiGhostFactor * uStrength;\n" +
        "    blendW = clamp(blendW, 0.0, 0.85);\n" +
        "\n" +
        "    fragColor = mix(current, blended, blendW);\n" +
        "}\n";

    // =================================================================
    // 构造 & 配置
    // =================================================================

    public Anime4KRenderer() {}

    public void setRefineStrength(float s) { refineStrength = s; }
    /** 设置 Pseudo-MV 插帧强度（0=关闭, 1=最强）*/
    public void setPMVStrength(float s)    { pmvStrength = Math.max(0f, Math.min(1f, s)); }
    /** 兼容旧接口：setATWStrength 映射到 setPMVStrength */
    public void setATWStrength(float s)    { setPMVStrength(s); }
    public void setFsrEnabled(boolean e)   { fsrEnabled = e; }
    public void setFsrSharpness(float s)   { fsrSharpness = s; }

    // =================================================================
    // 初始化
    // =================================================================

    public void init(int inW, int inH, int outW, int outH) {
        if (initialized) {
            GLES30.glDeleteTextures(12, new int[]{
                inputTexture, lumaTexture, lumadTexture, lumammTexture,
                outputTexture, lastOutputTexture, fsrTempTexture,
                tempTex, edgeDirTex, mvTexture, warpFwdTex, 0 /*padding*/
            }, 0);
            GLES30.glDeleteFramebuffers(10, fbo, 0);
            GLES30.glDeleteBuffers(2, vbo, 0);
            GLES30.glDeleteVertexArrays(2, vao, 0);
        }

        inputWidth  = inW;  inputHeight  = inH;
        outputWidth = outW; outputHeight = outH;
        // MV 场使用半分辨率（节省带宽，MV 场不需要全精度）
        mvWidth  = outW / 2;
        mvHeight = outH / 2;

        if (!initialized) {
            programLuma         = createProgram(VERTEX_SHADER, FRAG_LUMA);
            programGradX1       = createProgram(VERTEX_SHADER, FRAG_GRAD_X1);
            programGradY1       = createProgram(VERTEX_SHADER, FRAG_GRAD_Y1);
            programGradX2       = createProgram(VERTEX_SHADER, FRAG_GRAD_X2);
            programGradY2       = createProgram(VERTEX_SHADER, FRAG_GRAD_Y2);
            programApply        = createProgram(VERTEX_SHADER, FRAG_APPLY);
            programPMV_Estimate = createProgram(VERTEX_SHADER, FRAG_PMV_ESTIMATE);
            programPMV_WarpFwd  = createProgram(VERTEX_SHADER, FRAG_PMV_WARP_FWD);
            programPMV_Blend    = createProgram(VERTEX_SHADER, FRAG_PMV_BLEND);
            programATW          = createProgram(VERTEX_SHADER, FRAG_ATW);
            programPassthrough  = createProgram(VERTEX_SHADER, FRAG_PASSTHROUGH);
            programFsrEasu      = createProgram(FSRShaders.VERTEX_SHADER, FSRShaders.FRAG_EASU);
            programFsrRcas      = createProgram(FSRShaders.VERTEX_SHADER, FSRShaders.FRAG_RCAS);
        }

        cacheUniformLocations();

        // 创建纹理
        inputTexture      = createTexture(inputWidth,  inputHeight);
        lumaTexture       = createTexture(inputWidth,  inputHeight);
        lumadTexture      = createTexture(outputWidth, outputHeight);
        lumammTexture     = createTexture(outputWidth, outputHeight);
        outputTexture     = createTexture(outputWidth, outputHeight);
        lastOutputTexture = createTexture(outputWidth, outputHeight);
        fsrTempTexture    = createTexture(outputWidth, outputHeight);
        tempTex           = createTexture(inputWidth,  inputHeight);
        edgeDirTex        = createTexture(outputWidth, outputHeight);
        // Pseudo-MV 专用（半分辨率 MV 场）
        mvTexture         = createTexture(mvWidth, mvHeight);
        warpFwdTex        = createTexture(outputWidth, outputHeight);

        // 创建并固定绑定 FBO
        GLES30.glGenFramebuffers(10, fbo, 0);
        bindFboTexture(fbo[0], lumaTexture);
        bindFboTexture(fbo[1], tempTex);
        bindFboTexture(fbo[2], lumadTexture);
        bindFboTexture(fbo[3], lumammTexture);
        bindFboTexture(fbo[4], outputTexture);
        bindFboTexture(fbo[5], lastOutputTexture);
        bindFboTexture(fbo[6], fsrTempTexture);
        bindFboTexture(fbo[7], edgeDirTex);
        bindFboTexture(fbo[8], mvTexture);
        bindFboTexture(fbo[9], warpFwdTex);

        setupVboVao();
        initialized = true;
    }

    private void cacheUniformLocations() {
        uLuma_texture          = GLES30.glGetUniformLocation(programLuma,         "uTexture");
        uGradX1_luma           = GLES30.glGetUniformLocation(programGradX1,       "uLuma");
        uGradX1_texelSize      = GLES30.glGetUniformLocation(programGradX1,       "uTexelSize");
        uGradY1_grad           = GLES30.glGetUniformLocation(programGradY1,       "uGrad");
        uGradY1_texelSize      = GLES30.glGetUniformLocation(programGradY1,       "uTexelSize");
        uGradY1_refineStrength = GLES30.glGetUniformLocation(programGradY1,       "uRefineStrength");
        uGradX2_lumad          = GLES30.glGetUniformLocation(programGradX2,       "uLumad");
        uGradX2_texelSize      = GLES30.glGetUniformLocation(programGradX2,       "uTexelSize");
        uGradY2_lumad          = GLES30.glGetUniformLocation(programGradY2,       "uLumad");
        uGradY2_lumamm         = GLES30.glGetUniformLocation(programGradY2,       "uLumamm");
        uGradY2_texelSize      = GLES30.glGetUniformLocation(programGradY2,       "uTexelSize");
        uApply_texture         = GLES30.glGetUniformLocation(programApply,        "uTexture");
        uApply_lumad           = GLES30.glGetUniformLocation(programApply,        "uLumad");
        uApply_lumamm          = GLES30.glGetUniformLocation(programApply,        "uLumamm");
        uApply_texelSize       = GLES30.glGetUniformLocation(programApply,        "uTexelSize");
        uPassthrough_texture   = GLES30.glGetUniformLocation(programPassthrough,  "uTexture");
        uEasu_texture          = GLES30.glGetUniformLocation(programFsrEasu,      "uTexture");
        uEasu_con1             = GLES30.glGetUniformLocation(programFsrEasu,      "uEasuCon1");
        uRcas_texture          = GLES30.glGetUniformLocation(programFsrRcas,      "uTexture");
        uRcas_con              = GLES30.glGetUniformLocation(programFsrRcas,      "uRcasCon");
        // Pseudo-MV
        uPMV_Est_lumad         = GLES30.glGetUniformLocation(programPMV_Estimate, "uLumad");
        uPMV_Est_edgeDir       = GLES30.glGetUniformLocation(programPMV_Estimate, "uEdgeDir");
        uPMV_Est_texelSize     = GLES30.glGetUniformLocation(programPMV_Estimate, "uTexelSize");
        uPMV_Est_strength      = GLES30.glGetUniformLocation(programPMV_Estimate, "uStrength");
        uPMV_Fwd_current       = GLES30.glGetUniformLocation(programPMV_WarpFwd,  "uCurrent");
        uPMV_Fwd_mvTex         = GLES30.glGetUniformLocation(programPMV_WarpFwd,  "uMvTex");
        uPMV_Fwd_texelSize     = GLES30.glGetUniformLocation(programPMV_WarpFwd,  "uTexelSize");
        uPMV_Blend_current     = GLES30.glGetUniformLocation(programPMV_Blend,    "uCurrent");
        uPMV_Blend_last        = GLES30.glGetUniformLocation(programPMV_Blend,    "uLast");
        uPMV_Blend_mvTex       = GLES30.glGetUniformLocation(programPMV_Blend,    "uMvTex");
        uPMV_Blend_lumad       = GLES30.glGetUniformLocation(programPMV_Blend,    "uLumad");
        uPMV_Blend_texelSize   = GLES30.glGetUniformLocation(programPMV_Blend,    "uTexelSize");
        uPMV_Blend_strength    = GLES30.glGetUniformLocation(programPMV_Blend,    "uStrength");
        // ATW（FSR 模式）
        uATW_texture           = GLES30.glGetUniformLocation(programATW,           "uTexture");
        uATW_lastTexture       = GLES30.glGetUniformLocation(programATW,           "uLastTexture");
        uATW_strength          = GLES30.glGetUniformLocation(programATW,           "uATWStrength");
        uATW_offset            = GLES30.glGetUniformLocation(programATW,           "uOffset");
    }

    private void bindFboTexture(int fboId, int texId) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId);
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER,
                GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, texId, 0);
    }

    private void setupVboVao() {
        GLES30.glGenBuffers(2, vbo, 0);
        GLES30.glGenVertexArrays(2, vao, 0);
        int posLoc = GLES30.glGetAttribLocation(programPassthrough, "aPosition");
        int texLoc = GLES30.glGetAttribLocation(programPassthrough, "aTexCoord");
        setupSingleVboVao(vbo[0], vao[0], QUAD_VERTICES, posLoc, texLoc);
        setupSingleVboVao(vbo[1], vao[1], QUAD_VERTICES_FLIPPED, posLoc, texLoc);
    }

    private void setupSingleVboVao(int vboId, int vaoId, float[] verts, int posLoc, int texLoc) {
        FloatBuffer buf = ByteBuffer.allocateDirect(verts.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        buf.put(verts).position(0);
        GLES30.glBindVertexArray(vaoId);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vboId);
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, verts.length * 4, buf, GLES30.GL_STATIC_DRAW);
        GLES30.glEnableVertexAttribArray(posLoc);
        GLES30.glVertexAttribPointer(posLoc, 2, GLES30.GL_FLOAT, false, 16, 0);
        GLES30.glEnableVertexAttribArray(texLoc);
        GLES30.glVertexAttribPointer(texLoc, 2, GLES30.GL_FLOAT, false, 16, 8);
        GLES30.glBindVertexArray(0);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);
    }

    // =================================================================
    // 公开入口
    // =================================================================

    public int processFromBuffer(ByteBuffer buffer, int width, int height, int rowPixels) {
        if (!initialized) return 0;
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture);
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1);
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, rowPixels);
        buffer.position(0);
        GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0,
                width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer);
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0);
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 4);
        return runPipeline();
    }

    public int process(Bitmap bitmap) {
        if (!initialized) return 0;
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture);
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0);
        return runPipeline();
    }

    // =================================================================
    // 核心渲染管线
    // =================================================================

    private int runPipeline() {
        int currentTexture;

        if (!fsrEnabled) {
            // ---- Anime4K 管线（Pass 1-6）----

            // Pass 1: Luma
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[0]);
            GLES30.glViewport(0, 0, inputWidth, inputHeight);
            GLES30.glUseProgram(programLuma);
            bindTex(0, inputTexture); GLES30.glUniform1i(uLuma_texture, 0);
            drawVao(vao[0]);

            // Pass 2: GradX1 → tempTex
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[1]);
            GLES30.glViewport(0, 0, inputWidth, inputHeight);
            GLES30.glUseProgram(programGradX1);
            bindTex(0, lumaTexture); GLES30.glUniform1i(uGradX1_luma, 0);
            GLES30.glUniform2f(uGradX1_texelSize, 1f/inputWidth, 1f/inputHeight);
            drawVao(vao[0]);

            // Pass 3: GradY1 → lumadTexture
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[2]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programGradY1);
            bindTex(0, tempTex); GLES30.glUniform1i(uGradY1_grad, 0);
            GLES30.glUniform2f(uGradY1_texelSize, 1f/inputWidth, 1f/inputHeight);
            GLES30.glUniform1f(uGradY1_refineStrength, refineStrength);
            drawVao(vao[0]);

            // Pass 4: GradX2 → lumammTexture
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[3]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programGradX2);
            bindTex(0, lumadTexture); GLES30.glUniform1i(uGradX2_lumad, 0);
            GLES30.glUniform2f(uGradX2_texelSize, 1f/outputWidth, 1f/outputHeight);
            drawVao(vao[0]);

            // Pass 5: GradY2 → edgeDirTex（归一化边缘方向场）
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[7]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programGradY2);
            bindTex(0, lumadTexture);  GLES30.glUniform1i(uGradY2_lumad,  0);
            bindTex(1, lumammTexture); GLES30.glUniform1i(uGradY2_lumamm, 1);
            GLES30.glUniform2f(uGradY2_texelSize, 1f/outputWidth, 1f/outputHeight);
            drawVao(vao[0]);

            // Pass 6: Apply → outputTexture
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[4]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programApply);
            bindTex(0, inputTexture); GLES30.glUniform1i(uApply_texture, 0);
            bindTex(1, lumadTexture); GLES30.glUniform1i(uApply_lumad,   1);
            bindTex(2, edgeDirTex);   GLES30.glUniform1i(uApply_lumamm,  2);
            GLES30.glUniform2f(uApply_texelSize, 1f/inputWidth, 1f/inputHeight);
            drawVao(vao[0]);
            currentTexture = outputTexture;

            // ---- Pseudo-MV 插帧（Pass A/B/C，仅 Anime4K 模式）----
            if (pmvStrength > 0.01f) {
                currentTexture = runPseudoMV(currentTexture);
            } else {
                // 无插帧：将当前帧存入 lastOutput 供下帧备用
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
                GLES30.glViewport(0, 0, outputWidth, outputHeight);
                GLES30.glUseProgram(programPassthrough);
                bindTex(0, currentTexture); GLES30.glUniform1i(uPassthrough_texture, 0);
                drawVao(vao[0]);
            }

        } else {
            // ---- FSR 管线 ----
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[6]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programFsrEasu);
            bindTex(0, inputTexture); GLES30.glUniform1i(uEasu_texture, 0);
            float[] easuCon1 = {1f/inputWidth, 1f/inputHeight, 0f, 0f};
            GLES30.glUniform4fv(uEasu_con1, 1, easuCon1, 0);
            drawVao(vao[0]);

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[4]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programFsrRcas);
            bindTex(0, fsrTempTexture); GLES30.glUniform1i(uRcas_texture, 0);
            GLES30.glUniform4f(uRcas_con, fsrSharpness, 0, 0, 0);
            drawVao(vao[0]);
            currentTexture = outputTexture;

            // FSR 模式：使用 ATW 插帧（因无边缘方向场，降级为异步时间扭曲）
            if (pmvStrength > 0.01f) {
                // ATW：将结果写入 lastOutputTexture，再 ping-pong 交换
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
                GLES30.glViewport(0, 0, outputWidth, outputHeight);
                GLES30.glUseProgram(programATW);
                bindTex(0, currentTexture);    GLES30.glUniform1i(uATW_texture,     0);
                bindTex(1, lastOutputTexture); GLES30.glUniform1i(uATW_lastTexture, 1);
                GLES30.glUniform1f(uATW_strength, pmvStrength);
                GLES30.glUniform2f(uATW_offset, 0.001f, 0.001f);
                drawVao(vao[0]);
                // ping-pong 交换：outputTexture ↔ lastOutputTexture
                int tmp = outputTexture;
                outputTexture     = lastOutputTexture;
                lastOutputTexture = tmp;
                bindFboTexture(fbo[4], outputTexture);
                bindFboTexture(fbo[5], lastOutputTexture);
                currentTexture = outputTexture;
            } else {
                // 无插帧：将当前帧存入 lastOutput 供下帧备用
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
                GLES30.glViewport(0, 0, outputWidth, outputHeight);
                GLES30.glUseProgram(programPassthrough);
                bindTex(0, currentTexture); GLES30.glUniform1i(uPassthrough_texture, 0);
                drawVao(vao[0]);
            }
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        return outputTexture;
    }

    /**
     * Pseudo-MV 插帧管线（Pass A → B → C）
     * 输入：当前 Anime4K 增强帧（currentTex）、lumadTexture、edgeDirTex、lastOutputTexture
     * 输出：插帧后的结果（写回 outputTexture，并更新 lastOutputTexture）
     */
    private int runPseudoMV(int currentTex) {
        // ---- Pass A: 运动矢量估计 → mvTexture（半分辨率）----
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[8]);
        GLES30.glViewport(0, 0, mvWidth, mvHeight);
        GLES30.glUseProgram(programPMV_Estimate);
        bindTex(0, lumadTexture); GLES30.glUniform1i(uPMV_Est_lumad,   0);
        bindTex(1, edgeDirTex);   GLES30.glUniform1i(uPMV_Est_edgeDir, 1);
        GLES30.glUniform2f(uPMV_Est_texelSize, 1f/outputWidth, 1f/outputHeight);
        GLES30.glUniform1f(uPMV_Est_strength, pmvStrength);
        drawVao(vao[0]);

        // ---- Pass B: 前向翘曲 → warpFwdTex（全分辨率）----
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[9]);
        GLES30.glViewport(0, 0, outputWidth, outputHeight);
        GLES30.glUseProgram(programPMV_WarpFwd);
        bindTex(0, currentTex);  GLES30.glUniform1i(uPMV_Fwd_current, 0);
        bindTex(1, mvTexture);   GLES30.glUniform1i(uPMV_Fwd_mvTex,   1);
        GLES30.glUniform2f(uPMV_Fwd_texelSize, 1f/outputWidth, 1f/outputHeight);
        drawVao(vao[0]);

        // ---- Pass C: 后向翘曲 + 双向混合 → lastOutputTexture（ping-pong）----
        // 将结果写入 lastOutputTexture，然后与 outputTexture 交换引用
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
        GLES30.glViewport(0, 0, outputWidth, outputHeight);
        GLES30.glUseProgram(programPMV_Blend);
        bindTex(0, currentTex);        GLES30.glUniform1i(uPMV_Blend_current,  0);
        bindTex(1, lastOutputTexture); GLES30.glUniform1i(uPMV_Blend_last,     1);
        bindTex(2, mvTexture);         GLES30.glUniform1i(uPMV_Blend_mvTex,    2);
        bindTex(3, lumadTexture);      GLES30.glUniform1i(uPMV_Blend_lumad,    3);
        GLES30.glUniform2f(uPMV_Blend_texelSize, 1f/outputWidth, 1f/outputHeight);
        GLES30.glUniform1f(uPMV_Blend_strength, pmvStrength);
        drawVao(vao[0]);

        // Ping-pong：交换 outputTexture 与 lastOutputTexture
        // lastOutputTexture 现在存的是插帧结果，outputTexture 存的是旧的 lastOutput
        // 我们希望：outputTexture = 插帧结果，lastOutputTexture = 当前帧（供下帧使用）
        // 策略：先把当前帧（currentTex = outputTexture）拷贝到 lastOutputTexture 的旧位置
        //       然后交换引用，使 outputTexture 指向插帧结果
        int tmp = outputTexture;
        outputTexture     = lastOutputTexture; // 插帧结果
        lastOutputTexture = tmp;               // 旧的 outputTexture（将存当前帧）

        // 更新 FBO 绑定
        bindFboTexture(fbo[4], outputTexture);
        bindFboTexture(fbo[5], lastOutputTexture);

        // 将当前帧（currentTex，即交换前的 outputTexture）存入新的 lastOutputTexture
        // 供下一帧的 Pass C 作为 uLast 输入
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
        GLES30.glViewport(0, 0, outputWidth, outputHeight);
        GLES30.glUseProgram(programPassthrough);
        bindTex(0, currentTex); GLES30.glUniform1i(uPassthrough_texture, 0);
        drawVao(vao[0]);

        return outputTexture;
    }

    public void renderToScreen(int texture, int screenWidth, int screenHeight) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        GLES30.glViewport(0, 0, screenWidth, screenHeight);
        GLES30.glClearColor(0f, 0f, 0f, 0f);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
        GLES30.glUseProgram(programPassthrough);
        bindTex(0, texture); GLES30.glUniform1i(uPassthrough_texture, 0);
        drawVao(vao[1]);
    }

    // =================================================================
    // 辅助方法
    // =================================================================

    private void bindTex(int unit, int texId) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texId);
    }

    private void drawVao(int vaoId) {
        GLES30.glBindVertexArray(vaoId);
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4);
        GLES30.glBindVertexArray(0);
    }

    private int createTexture(int w, int h) {
        int[] tex = new int[1];
        GLES30.glGenTextures(1, tex, 0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex[0]);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
                w, h, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null);
        return tex[0];
    }

    private int createProgram(String vert, String frag) {
        int vs = loadShader(GLES30.GL_VERTEX_SHADER, vert);
        int fs = loadShader(GLES30.GL_FRAGMENT_SHADER, frag);
        int prog = GLES30.glCreateProgram();
        GLES30.glAttachShader(prog, vs);
        GLES30.glAttachShader(prog, fs);
        GLES30.glLinkProgram(prog);
        int[] st = new int[1];
        GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, st, 0);
        if (st[0] != GLES30.GL_TRUE) {
            Log.e(TAG, "Link failed: " + GLES30.glGetProgramInfoLog(prog));
            GLES30.glDeleteProgram(prog);
            return 0;
        }
        GLES30.glDeleteShader(vs);
        GLES30.glDeleteShader(fs);
        return prog;
    }

    private int loadShader(int type, String src) {
        int shader = GLES30.glCreateShader(type);
        GLES30.glShaderSource(shader, src);
        GLES30.glCompileShader(shader);
        int[] st = new int[1];
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, st, 0);
        if (st[0] == 0) {
            Log.e(TAG, "Compile failed type=" + type + ": " + GLES30.glGetShaderInfoLog(shader));
            GLES30.glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    public void release() {
        if (!initialized) return;
        int[] programs = {programLuma, programGradX1, programGradY1, programGradX2,
                          programGradY2, programApply, programPMV_Estimate,
                          programPMV_WarpFwd, programPMV_Blend, programATW,
                          programPassthrough, programFsrEasu, programFsrRcas};
        for (int p : programs) if (p != 0) GLES30.glDeleteProgram(p);
        GLES30.glDeleteTextures(11, new int[]{
            inputTexture, lumaTexture, lumadTexture, lumammTexture,
            outputTexture, lastOutputTexture, fsrTempTexture,
            tempTex, edgeDirTex, mvTexture, warpFwdTex
        }, 0);
        GLES30.glDeleteFramebuffers(10, fbo, 0);
        GLES30.glDeleteBuffers(2, vbo, 0);
        GLES30.glDeleteVertexArrays(2, vao, 0);
        initialized = false;
    }
}
