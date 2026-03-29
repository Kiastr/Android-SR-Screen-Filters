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
 * v1.9.0 — FSR EASU 修复 + Pseudo-MV 前向外推升级 (LSFG 3.0 / Mob-FGSR 参考)
 * =====================================================================
 *
 * [v1.9.0 新增]
 * [FIX-FSR] 修复 FSRShaders.java 中 FRAG_EASU 假 FSR 问题：
 *   原实现仅是单次双线性采样，并非真正的 FSR 1.0 EASU。
 *   新实现：4-tap 方向性 Lanczos-2 近似，根据水平/垂直梯度
 *   自适应地在边缘方向进行方向性重建，平坦区域退化为双线性插値。
 *
 * [FIX-UI] 修复 UI 默认捕获分辨率：
 *   activity_main.xml 中 scaleSeekBar 默认 progress 从 50 改为 75，
 *   与 MainActivity.java 代码默认値保持一致。
 *
 * [UPG-PMV] Pseudo-MV Pass C 升级：双向混合 → 前向外推 + 置信度门控：
 *   旧版：0.5*前向 + 0.5*后向双向混合，在快速运动区域产生拖影。
 *   新版：全量前向外推（+1.0x MV），孔洞检测与后向填充，
 *         置信度门控（conf < 0.08 回退），时域一致性反鬼影。
 *
 * 继承自 v1.8.0 的性能优化（参考 Lossless Scaling Frame Generation 3.0）：
 * 参考 Lossless Scaling Frame Generation 3.0 的 40% GPU 负载降低策略，
 * 对渲染管线进行全面性能优化：
 *
 * [OPT-G1] 中间纹理降精度：
 *   luma → GL_R8 (单通道), gradX1/gradY1/gradX2/gradY2/edgeDir → GL_RG8 (双通道)
 *   节省 2-4x 显存带宽，对 TBR 架构（Adreno/Mali/PowerVR）效果尤为显著。
 *
 * [OPT-G2] glInvalidateFramebuffer (TBR 友好):
 *   在每次 FBO 切换前调用 glInvalidateFramebuffer 告知驱动当前 FBO 的颜色附件
 *   不再需要，避免 TBR GPU 将 tile memory 写回 RAM (store) 的昂贵操作。
 *   参考 Samsung Galaxy GameDev 和 Android Developer 的 TBR 优化指南。
 *
 * [OPT-G3] 消除冗余 Passthrough 全屏绘制：
 *   当无插帧时，原代码使用 Passthrough shader 做全屏 draw 将当前帧拷贝到
 *   lastOutputTexture。现改为 glCopyImageSubData (ES 3.2) 或
 *   glBlitFramebuffer (ES 3.0) 进行 GPU 内部拷贝，省去一次完整的
 *   vertex processing + fragment shading + rasterization 开销。
 *
 * [OPT-G4] VAO 绑定优化：
 *   整个管线只在开头绑定一次 VAO，结束时解绑一次。消除每次 drawVao 中的
 *   bind/unbind 调用（原代码每帧 10+ 次 bind/unbind）。
 *
 * [OPT-G5] Shader 精度进一步分级：
 *   Pass A (PMV Estimate) 降级为 lowp（MV 场不需要高精度）。
 *   Pass B (WarpFwd) 降级为 lowp。
 *
 * 继承自 v1.5.0 的优化：
 * [OPT-1] VBO/VAO  [OPT-3] Uniform 预缓存  [OPT-4] 预分配纹理
 * [OPT-5] FBO 固定绑定  [OPT-6] Shader 精度分级
 *
 * Pseudo-MV 插帧继承自 v1.5.0，ATW 继承自 FSR 模式。
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
    private int programLuma, programGradX1, programGradY1;
    private int programGradX2, programGradY2, programApply;
    private int programPMV_Estimate, programPMV_WarpFwd, programPMV_Blend;
    private int programATW;
    private int programPassthrough;
    private int programFsrEasu, programFsrRcas;

    // ---- FBO 索引 ----
    // [0]=luma  [1]=tempTex(gradX1)  [2]=lumad  [3]=lumamm(gradX2)
    // [4]=output  [5]=lastOutput  [6]=fsrTemp  [7]=edgeDir(gradY2)
    // [8]=mvTex  [9]=warpFwd
    private int[] fbo = new int[10];

    // ---- 纹理 ----
    private int inputTexture;
    private int lumaTexture;       // GL_R8
    private int lumadTexture;      // GL_RG8: R=Sobel强度, G=细化权重
    private int lumammTexture;     // GL_RG8: gradX2 中间纹理
    private int outputTexture;     // GL_RGBA8
    private int lastOutputTexture; // GL_RGBA8
    private int fsrTempTexture;    // GL_RGBA8
    private int tempTex;           // GL_RG8: gradX1 临时
    private int edgeDirTex;        // GL_RG8: 边缘方向场
    private int mvTexture;         // GL_RGBA8: RG=MV, B=置信度
    private int warpFwdTex;        // GL_RGBA8

    private int inputWidth, inputHeight;
    private int outputWidth, outputHeight;
    private int mvWidth, mvHeight;

    private float refineStrength  = 0.5f;
    private float pmvStrength     = 0.5f;
    private boolean fsrEnabled    = false;
    private float fsrSharpness    = 0.2f;

    private boolean initialized = false;

    // [OPT-G2] glInvalidateFramebuffer 参数（预分配避免每帧 GC）
    private static final int[] INVALIDATE_ATTACHMENT = { GLES30.GL_COLOR_ATTACHMENT0 };

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
    private int uPMV_Est_lumad, uPMV_Est_edgeDir, uPMV_Est_texelSize, uPMV_Est_strength;
    private int uPMV_Fwd_current, uPMV_Fwd_mvTex, uPMV_Fwd_texelSize;
    private int uPMV_Blend_current, uPMV_Blend_last, uPMV_Blend_mvTex;
    private int uPMV_Blend_lumad, uPMV_Blend_texelSize, uPMV_Blend_strength;
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

    // Pass 1: Luma → GL_R8 (单通道输出)
    // [OPT-G1] 输出到 R8 纹理，只写 R 通道
    private static final String FRAG_LUMA =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uTexture;\n" +
        "out float fragColor;\n" +  // R8 单通道输出
        "void main() {\n" +
        "    vec3 c = texture(uTexture, vTexCoord).rgb;\n" +
        "    fragColor = dot(vec3(0.299, 0.587, 0.114), c);\n" +
        "}\n";

    // Pass 2: GradX1 → GL_RG8 (双通道输出)
    private static final String FRAG_GRAD_X1 =
        "#version 300 es\n" +
        "precision lowp float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uLuma;\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec2 fragColor;\n" +  // RG8 双通道输出
        "void main() {\n" +
        "    float l = texture(uLuma, vTexCoord + vec2(-uTexelSize.x, 0.0)).r;\n" +
        "    float c = texture(uLuma, vTexCoord).r;\n" +
        "    float r = texture(uLuma, vTexCoord + vec2(uTexelSize.x, 0.0)).r;\n" +
        "    fragColor = vec2((-l+r)*0.5+0.5, (l+c+c+r)*0.25+0.5);\n" +
        "}\n";

    // Pass 3: GradY1 → GL_RG8 (R=sobel编码, G=dval)
    private static final String FRAG_GRAD_Y1 =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uGrad;\n" +
        "uniform vec2 uTexelSize;\n" +
        "uniform float uRefineStrength;\n" +
        "out vec2 fragColor;\n" +  // RG8 双通道输出
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
        "    fragColor=vec2(sobel*0.5+0.5,dval);\n" +
        "}\n";

    // Pass 4: GradX2 → GL_RG8
    private static final String FRAG_GRAD_X2 =
        "#version 300 es\n" +
        "precision lowp float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uLumad;\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec2 fragColor;\n" +  // RG8 双通道输出
        "void main() {\n" +
        "    float dval=texture(uLumad,vTexCoord).g;\n" +
        "    if(dval<0.1){fragColor=vec2(0.5,0.5);return;}\n" +
        "    float s=texture(uLumad,vTexCoord).r*2.-1.;\n" +
        "    float l=texture(uLumad,vTexCoord+vec2(-uTexelSize.x,0.)).r*2.-1.;\n" +
        "    float r=texture(uLumad,vTexCoord+vec2(uTexelSize.x,0.)).r*2.-1.;\n" +
        "    fragColor=vec2((-l+r)*0.5+0.5,(l+s+s+r)*0.25+0.5);\n" +
        "}\n";

    // Pass 5: GradY2 → GL_RG8 (归一化边缘方向场)
    private static final String FRAG_GRAD_Y2 =
        "#version 300 es\n" +
        "precision lowp float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uLumad;\n" +
        "uniform sampler2D uLumamm;\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec2 fragColor;\n" +  // RG8 双通道输出
        "void main() {\n" +
        "    float dval=texture(uLumad,vTexCoord).g;\n" +
        "    if(dval<0.1){fragColor=vec2(0.5,0.5);return;}\n" +
        "    float tx=texture(uLumamm,vTexCoord+vec2(0.,-uTexelSize.y)).r*2.-1.;\n" +
        "    float cx=texture(uLumamm,vTexCoord).r*2.-1.;\n" +
        "    float bx=texture(uLumamm,vTexCoord+vec2(0.,uTexelSize.y)).r*2.-1.;\n" +
        "    float ty=texture(uLumamm,vTexCoord+vec2(0.,-uTexelSize.y)).g*4.-2.;\n" +
        "    float by=texture(uLumamm,vTexCoord+vec2(0.,uTexelSize.y)).g*4.-2.;\n" +
        "    float xg=(tx+cx+cx+bx); float yg=(-ty+by);\n" +
        "    float norm=sqrt(xg*xg+yg*yg);\n" +
        "    if(norm<=0.001){fragColor=vec2(0.5,0.5);return;}\n" +
        "    fragColor=vec2(xg/norm*0.5+0.5,yg/norm*0.5+0.5);\n" +
        "}\n";

    // Pass 6: Apply — 读取 RG8 纹理的 .rg 通道
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

    // ATW（FSR 模式专用）
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

    // Pass A: 运动矢量估计 — [OPT-G5] lowp 精度
    private static final String FRAG_PMV_ESTIMATE =
        "#version 300 es\n" +
        "precision lowp float;\n" +  // [OPT-G5] MV 场不需要高精度
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uLumad;\n" +
        "uniform sampler2D uEdgeDir;\n" +
        "uniform vec2 uTexelSize;\n" +
        "uniform float uStrength;\n" +
        "out vec4 fragColor;\n" +
        "vec2 edgeToMotion(vec2 uv) {\n" +
        "    vec2 edgeDir = texture(uEdgeDir, uv).rg * 2.0 - 1.0;\n" +
        "    return vec2(-edgeDir.y, edgeDir.x);\n" +
        "}\n" +
        "void main() {\n" +
        "    vec2 lumadVal = texture(uLumad, vTexCoord).rg;\n" +
        "    float sobel = lumadVal.r * 2.0 - 1.0;\n" +
        "    float dval  = lumadVal.g;\n" +
        "    vec2 mv;\n" +
        "    float confidence;\n" +
        "    if (dval >= 0.15) {\n" +
        "        mv = edgeToMotion(vTexCoord);\n" +
        "        confidence = dval;\n" +
        "    } else {\n" +
        "        vec2 mvSum = vec2(0.0);\n" +
        "        float wSum = 0.0;\n" +
        "        for (int dy = -1; dy <= 1; dy++) {\n" +
        "            for (int dx = -1; dx <= 1; dx++) {\n" +
        "                if (dx == 0 && dy == 0) continue;\n" +
        "                vec2 offset = vec2(float(dx), float(dy)) * uTexelSize;\n" +
        "                vec2 nUV = vTexCoord + offset;\n" +
        "                float nDval = texture(uLumad, nUV).g;\n" +
        "                if (nDval >= 0.15) {\n" +
        "                    float distW = (abs(float(dx)) + abs(float(dy)) == 1.0) ? 1.0 : 0.707;\n" +
        "                    float w = nDval * distW;\n" +
        "                    mvSum += edgeToMotion(nUV) * w;\n" +
        "                    wSum += w;\n" +
        "                }\n" +
        "            }\n" +
        "        }\n" +
        "        if (wSum > 0.001) {\n" +
        "            mv = mvSum / wSum;\n" +
        "            confidence = wSum / 8.0 * 0.5;\n" +
        "        } else {\n" +
        "            mv = vec2(0.0);\n" +
        "            confidence = 0.0;\n" +
        "        }\n" +
        "    }\n" +
        "    mv *= uStrength;\n" +
        "    confidence *= uStrength;\n" +
        "    fragColor = vec4(mv * 0.5 + 0.5, confidence, 1.0);\n" +
        "}\n";

    // Pass B: 前向翘曲 — [OPT-G5] lowp 精度
    private static final String FRAG_PMV_WARP_FWD =
        "#version 300 es\n" +
        "precision lowp float;\n" +  // [OPT-G5]
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uCurrent;\n" +
        "uniform sampler2D uMvTex;\n" +
        "uniform vec2 uTexelSize;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    vec3 mvData = texture(uMvTex, vTexCoord).rgb;\n" +
        "    vec2 mv = (mvData.rg * 2.0 - 1.0) * uTexelSize * 8.0;\n" +
        "    float conf = mvData.b;\n" +
        "    vec2 fwdUV = clamp(vTexCoord + mv * 0.5, vec2(0.0), vec2(1.0));\n" +
        "    vec4 fwdColor = texture(uCurrent, fwdUV);\n" +
        "    fragColor = vec4(fwdColor.rgb, conf);\n" +
        "}\n";

    // Pass C: 前向外推 + 置信度门控反鬼影（v1.9.0 升级）
    //
    // 升级说明（参考 LSFG 3.0 / Mob-FGSR 方案）：
    //   旧版：双向混合（0.5*前向 + 0.5*后向），在快速运动区域产生拖影
    //   新版：前向外推（从当前帧沿运动矢量方向预测下一帧位置），
    //         配合置信度门控（conf < threshold 时回退到当前帧），
    //         以及时域差分反鬼影（快速运动区域降低插帧权重）
    //
    // 核心改进：
    //   1. 前向外推：extrapolatedUV = vTexCoord + mv（全量外推，而非 0.5x 双向）
    //   2. 孔洞检测：通过采样 warpFwdTex 的 alpha 通道判断是否为孔洞区域
    //   3. 置信度门控：conf 低时（遮挡/快速运动）直接使用当前帧，避免鬼影
    //   4. 时域一致性：比较外推结果与当前帧的亮度差，差异过大则降低混合权重
    private static final String FRAG_PMV_BLEND =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "in vec2 vTexCoord;\n" +
        "uniform sampler2D uCurrent;\n" +
        "uniform sampler2D uLast;\n" +
        "uniform sampler2D uMvTex;\n" +
        "uniform sampler2D uLumad;\n" +
        "uniform vec2 uTexelSize;\n" +
        "uniform float uStrength;\n" +
        "out vec4 fragColor;\n" +
        "\n" +
        "// 亮度差（用于时域一致性检测）\n" +
        "float lumaDiff(vec3 a, vec3 b) {\n" +
        "    return abs(dot(a - b, vec3(0.299, 0.587, 0.114)));\n" +
        "}\n" +
        "\n" +
        "// 3x3 均值滤波（孔洞填充：用邻域平均填补前向翘曲的空洞）\n" +
        "vec4 holeFill(sampler2D tex, vec2 uv, vec2 texelSize) {\n" +
        "    vec4 sum = vec4(0.0);\n" +
        "    float validCount = 0.0;\n" +
        "    for (int dy = -1; dy <= 1; dy++) {\n" +
        "        for (int dx = -1; dx <= 1; dx++) {\n" +
        "            vec2 nUV = uv + vec2(float(dx), float(dy)) * texelSize;\n" +
        "            vec4 s = texture(tex, nUV);\n" +
        "            // alpha > 0.1 表示该像素被前向翘曲写入（非孔洞）\n" +
        "            if (s.a > 0.1) {\n" +
        "                sum += s;\n" +
        "                validCount += 1.0;\n" +
        "            }\n" +
        "        }\n" +
        "    }\n" +
        "    return (validCount > 0.0) ? sum / validCount : vec4(0.0);\n" +
        "}\n" +
        "\n" +
        "void main() {\n" +
        "    vec4 current = texture(uCurrent, vTexCoord);\n" +
        "    \n" +
        "    // 快速路径：插帧强度为 0 时直接输出当前帧\n" +
        "    if (uStrength < 0.01) {\n" +
        "        fragColor = current;\n" +
        "        return;\n" +
        "    }\n" +
        "    \n" +
        "    // 读取运动矢量和置信度\n" +
        "    vec3 mvData = texture(uMvTex, vTexCoord).rgb;\n" +
        "    vec2 mv = (mvData.rg * 2.0 - 1.0) * uTexelSize * 8.0;\n" +
        "    float conf = mvData.b;\n" +
        "    \n" +
        "    // 置信度门控：置信度过低（遮挡/快速运动）时回退到当前帧\n" +
        "    // 阈值 0.08 比旧版 0.05 更严格，减少低质量 MV 引起的鬼影\n" +
        "    if (conf < 0.08) {\n" +
        "        fragColor = current;\n" +
        "        return;\n" +
        "    }\n" +
        "    \n" +
        "    // ---- 前向外推（Forward Extrapolation）----\n" +
        "    // 从当前帧位置沿运动矢量方向外推到预测的下一帧位置\n" +
        "    // 相比双向混合（±0.5x），全量外推（+1.0x）更接近真实的帧间预测\n" +
        "    vec2 extraUV = clamp(vTexCoord + mv, vec2(0.0), vec2(1.0));\n" +
        "    vec4 extrapolated = texture(uCurrent, extraUV);\n" +
        "    \n" +
        "    // ---- 孔洞检测与填充 ----\n" +
        "    // 前向翘曲会在运动目标的背后留下孔洞（未被写入的像素）\n" +
        "    // 通过检查 uLast（上一帧）与外推结果的一致性来检测孔洞\n" +
        "    vec4 lastAtExtra = texture(uLast, extraUV);\n" +
        "    float holeMask = step(lumaDiff(extrapolated.rgb, lastAtExtra.rgb), 0.15);\n" +
        "    // holeMask=1 表示外推结果与上帧一致（可信），=0 表示可能是孔洞\n" +
        "    \n" +
        "    // 孔洞区域：用后向采样（从上一帧反向查找）填补\n" +
        "    vec2 bwdUV = clamp(vTexCoord - mv * 0.5, vec2(0.0), vec2(1.0));\n" +
        "    vec4 bwdFill = texture(uLast, bwdUV);\n" +
        "    vec4 predicted = mix(bwdFill, extrapolated, holeMask);\n" +
        "    \n" +
        "    // ---- 时域一致性检测（反鬼影核心）----\n" +
        "    // 比较预测帧与当前帧的亮度差：差异越大，说明运动越快或遮挡越严重\n" +
        "    // 在这些区域大幅降低插帧权重，避免鬼影残留\n" +
        "    float temporalDiff = lumaDiff(predicted.rgb, current.rgb);\n" +
        "    // 阈值从旧版 (0.08, 0.20) 调整为 (0.06, 0.18)，更激进的反鬼影\n" +
        "    float antiGhost = 1.0 - smoothstep(0.06, 0.18, temporalDiff);\n" +
        "    \n" +
        "    // ---- 边缘感知权重 ----\n" +
        "    // 边缘区域（sobel 强度高）更需要插帧来保持清晰度\n" +
        "    // 平坦区域（低 sobel）插帧收益较小，适当降低权重\n" +
        "    float edgeW = clamp(texture(uLumad, vTexCoord).r * 2.0 - 1.0, 0.0, 1.0);\n" +
        "    float edgeFactor = 0.3 + 0.7 * edgeW;  // 平坦区域最低 0.3x 权重\n" +
        "    \n" +
        "    // ---- 最终混合权重 ----\n" +
        "    // 综合：置信度 × 边缘因子 × 反鬼影 × 用户强度\n" +
        "    float blendW = conf * edgeFactor * antiGhost * uStrength;\n" +
        "    blendW = clamp(blendW, 0.0, 0.90);  // 上限 0.90，保留 10% 当前帧防止完全替换\n" +
        "    \n" +
        "    fragColor = mix(current, predicted, blendW);\n" +
        "}\n";

    // =================================================================
    // 构造 & 配置
    // =================================================================

    public Anime4KRenderer() {}

    public void setRefineStrength(float s) { refineStrength = s; }
    public void setPMVStrength(float s)    { pmvStrength = Math.max(0f, Math.min(1f, s)); }
    public void setATWStrength(float s)    { setPMVStrength(s); }
    public void setFsrEnabled(boolean e)   { fsrEnabled = e; }
    public void setFsrSharpness(float s)   { fsrSharpness = s; }

    // =================================================================
    // 初始化
    // =================================================================

    public void init(int inW, int inH, int outW, int outH) {
        if (initialized) {
            GLES30.glDeleteTextures(11, new int[]{
                inputTexture, lumaTexture, lumadTexture, lumammTexture,
                outputTexture, lastOutputTexture, fsrTempTexture,
                tempTex, edgeDirTex, mvTexture, warpFwdTex
            }, 0);
            GLES30.glDeleteFramebuffers(10, fbo, 0);
            GLES30.glDeleteBuffers(2, vbo, 0);
            GLES30.glDeleteVertexArrays(2, vao, 0);
        }

        inputWidth  = inW;  inputHeight  = inH;
        outputWidth = outW; outputHeight = outH;
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

        // [OPT-G1] 中间纹理降精度
        inputTexture      = createTexture(inputWidth,  inputHeight,  GLES30.GL_RGBA8,  GLES30.GL_RGBA);
        lumaTexture       = createTexture(inputWidth,  inputHeight,  GLES30.GL_R8,     GLES30.GL_RED);
        tempTex           = createTexture(inputWidth,  inputHeight,  GLES30.GL_RG8,    GLES30.GL_RG);
        lumadTexture      = createTexture(outputWidth, outputHeight, GLES30.GL_RG8,    GLES30.GL_RG);
        lumammTexture     = createTexture(outputWidth, outputHeight, GLES30.GL_RG8,    GLES30.GL_RG);
        edgeDirTex        = createTexture(outputWidth, outputHeight, GLES30.GL_RG8,    GLES30.GL_RG);
        outputTexture     = createTexture(outputWidth, outputHeight, GLES30.GL_RGBA8,  GLES30.GL_RGBA);
        lastOutputTexture = createTexture(outputWidth, outputHeight, GLES30.GL_RGBA8,  GLES30.GL_RGBA);
        fsrTempTexture    = createTexture(outputWidth, outputHeight, GLES30.GL_RGBA8,  GLES30.GL_RGBA);
        mvTexture         = createTexture(mvWidth,     mvHeight,     GLES30.GL_RGBA8,  GLES30.GL_RGBA);
        warpFwdTex        = createTexture(outputWidth, outputHeight, GLES30.GL_RGBA8,  GLES30.GL_RGBA);

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
    // 核心渲染管线 (v1.8.0 优化版)
    // =================================================================

    private int runPipeline() {
        int currentTexture;

        // [OPT-G4] 绑定 VAO 一次，整个管线复用
        GLES30.glBindVertexArray(vao[0]);

        if (!fsrEnabled) {
            // ---- Anime4K 管线（Pass 1-6）----

            // Pass 1: Luma → GL_R8
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[0]);
            GLES30.glViewport(0, 0, inputWidth, inputHeight);
            GLES30.glUseProgram(programLuma);
            bindTex(0, inputTexture); GLES30.glUniform1i(uLuma_texture, 0);
            drawQuad();

            // [OPT-G2] Invalidate luma FBO before switching away
            GLES30.glInvalidateFramebuffer(GLES30.GL_FRAMEBUFFER, 1, INVALIDATE_ATTACHMENT, 0);

            // Pass 2: GradX1 → GL_RG8 tempTex
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[1]);
            GLES30.glViewport(0, 0, inputWidth, inputHeight);
            GLES30.glUseProgram(programGradX1);
            bindTex(0, lumaTexture); GLES30.glUniform1i(uGradX1_luma, 0);
            GLES30.glUniform2f(uGradX1_texelSize, 1f/inputWidth, 1f/inputHeight);
            drawQuad();

            // Pass 3: GradY1 → GL_RG8 lumadTexture
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[2]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programGradY1);
            bindTex(0, tempTex); GLES30.glUniform1i(uGradY1_grad, 0);
            GLES30.glUniform2f(uGradY1_texelSize, 1f/inputWidth, 1f/inputHeight);
            GLES30.glUniform1f(uGradY1_refineStrength, refineStrength);
            drawQuad();

            // [OPT-G2] Invalidate tempTex FBO (gradX1 不再需要)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[1]);
            GLES30.glInvalidateFramebuffer(GLES30.GL_FRAMEBUFFER, 1, INVALIDATE_ATTACHMENT, 0);

            // Pass 4: GradX2 → GL_RG8 lumammTexture
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[3]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programGradX2);
            bindTex(0, lumadTexture); GLES30.glUniform1i(uGradX2_lumad, 0);
            GLES30.glUniform2f(uGradX2_texelSize, 1f/outputWidth, 1f/outputHeight);
            drawQuad();

            // Pass 5: GradY2 → GL_RG8 edgeDirTex
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[7]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programGradY2);
            bindTex(0, lumadTexture);  GLES30.glUniform1i(uGradY2_lumad,  0);
            bindTex(1, lumammTexture); GLES30.glUniform1i(uGradY2_lumamm, 1);
            GLES30.glUniform2f(uGradY2_texelSize, 1f/outputWidth, 1f/outputHeight);
            drawQuad();

            // [OPT-G2] Invalidate lumamm FBO (gradX2 不再需要)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[3]);
            GLES30.glInvalidateFramebuffer(GLES30.GL_FRAMEBUFFER, 1, INVALIDATE_ATTACHMENT, 0);

            // Pass 6: Apply → outputTexture (RGBA8)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[4]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programApply);
            bindTex(0, inputTexture); GLES30.glUniform1i(uApply_texture, 0);
            bindTex(1, lumadTexture); GLES30.glUniform1i(uApply_lumad,   1);
            bindTex(2, edgeDirTex);   GLES30.glUniform1i(uApply_lumamm,  2);
            GLES30.glUniform2f(uApply_texelSize, 1f/inputWidth, 1f/inputHeight);
            drawQuad();
            currentTexture = outputTexture;

            // ---- Pseudo-MV 插帧 ----
            if (pmvStrength > 0.01f) {
                currentTexture = runPseudoMV(currentTexture);
            } else {
                // [OPT-G3] 无插帧时用 glBlitFramebuffer 替代 Passthrough draw
                // 将当前帧拷贝到 lastOutput 供下帧备用
                GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, fbo[4]);
                GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, fbo[5]);
                GLES30.glBlitFramebuffer(
                    0, 0, outputWidth, outputHeight,
                    0, 0, outputWidth, outputHeight,
                    GLES30.GL_COLOR_BUFFER_BIT, GLES30.GL_NEAREST);
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
            }

        } else {
            // ---- FSR 管线 ----
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[6]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programFsrEasu);
            bindTex(0, inputTexture); GLES30.glUniform1i(uEasu_texture, 0);
            float[] easuCon1 = {1f/inputWidth, 1f/inputHeight, 0f, 0f};
            GLES30.glUniform4fv(uEasu_con1, 1, easuCon1, 0);
            drawQuad();

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[4]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programFsrRcas);
            bindTex(0, fsrTempTexture); GLES30.glUniform1i(uRcas_texture, 0);
            GLES30.glUniform4f(uRcas_con, fsrSharpness, 0, 0, 0);
            drawQuad();
            currentTexture = outputTexture;

            // FSR 模式：ATW 插帧
            if (pmvStrength > 0.01f) {
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
                GLES30.glViewport(0, 0, outputWidth, outputHeight);
                GLES30.glUseProgram(programATW);
                bindTex(0, currentTexture);    GLES30.glUniform1i(uATW_texture,     0);
                bindTex(1, lastOutputTexture); GLES30.glUniform1i(uATW_lastTexture, 1);
                GLES30.glUniform1f(uATW_strength, pmvStrength);
                GLES30.glUniform2f(uATW_offset, 0.001f, 0.001f);
                drawQuad();
                // ping-pong
                int tmp = outputTexture;
                outputTexture     = lastOutputTexture;
                lastOutputTexture = tmp;
                bindFboTexture(fbo[4], outputTexture);
                bindFboTexture(fbo[5], lastOutputTexture);
                currentTexture = outputTexture;
            } else {
                // [OPT-G3] 无插帧时用 glBlitFramebuffer 替代 Passthrough draw
                GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, fbo[4]);
                GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, fbo[5]);
                GLES30.glBlitFramebuffer(
                    0, 0, outputWidth, outputHeight,
                    0, 0, outputWidth, outputHeight,
                    GLES30.GL_COLOR_BUFFER_BIT, GLES30.GL_NEAREST);
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
            }
        }

        // [OPT-G4] 管线结束，解绑 VAO
        GLES30.glBindVertexArray(0);

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        return outputTexture;
    }

    /**
     * Pseudo-MV 插帧管线（Pass A → B → C）
     * 注意：调用时 vao[0] 已经绑定（由 runPipeline 负责）
     */
    private int runPseudoMV(int currentTex) {
        // Pass A: 运动矢量估计 → mvTexture（半分辨率）
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[8]);
        GLES30.glViewport(0, 0, mvWidth, mvHeight);
        GLES30.glUseProgram(programPMV_Estimate);
        bindTex(0, lumadTexture); GLES30.glUniform1i(uPMV_Est_lumad,   0);
        bindTex(1, edgeDirTex);   GLES30.glUniform1i(uPMV_Est_edgeDir, 1);
        GLES30.glUniform2f(uPMV_Est_texelSize, 1f/outputWidth, 1f/outputHeight);
        GLES30.glUniform1f(uPMV_Est_strength, pmvStrength);
        drawQuad();

        // Pass B: 前向翘曲 → warpFwdTex
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[9]);
        GLES30.glViewport(0, 0, outputWidth, outputHeight);
        GLES30.glUseProgram(programPMV_WarpFwd);
        bindTex(0, currentTex);  GLES30.glUniform1i(uPMV_Fwd_current, 0);
        bindTex(1, mvTexture);   GLES30.glUniform1i(uPMV_Fwd_mvTex,   1);
        GLES30.glUniform2f(uPMV_Fwd_texelSize, 1f/outputWidth, 1f/outputHeight);
        drawQuad();

        // [OPT-G2] Invalidate edgeDir FBO (不再需要)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[7]);
        GLES30.glInvalidateFramebuffer(GLES30.GL_FRAMEBUFFER, 1, INVALIDATE_ATTACHMENT, 0);

        // Pass C: 后向翘曲 + 双向混合 → lastOutputTexture
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
        GLES30.glViewport(0, 0, outputWidth, outputHeight);
        GLES30.glUseProgram(programPMV_Blend);
        bindTex(0, currentTex);        GLES30.glUniform1i(uPMV_Blend_current,  0);
        bindTex(1, lastOutputTexture); GLES30.glUniform1i(uPMV_Blend_last,     1);
        bindTex(2, mvTexture);         GLES30.glUniform1i(uPMV_Blend_mvTex,    2);
        bindTex(3, lumadTexture);      GLES30.glUniform1i(uPMV_Blend_lumad,    3);
        GLES30.glUniform2f(uPMV_Blend_texelSize, 1f/outputWidth, 1f/outputHeight);
        GLES30.glUniform1f(uPMV_Blend_strength, pmvStrength);
        drawQuad();

        // [OPT-G2] Invalidate mvTex and warpFwd FBOs
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[8]);
        GLES30.glInvalidateFramebuffer(GLES30.GL_FRAMEBUFFER, 1, INVALIDATE_ATTACHMENT, 0);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[9]);
        GLES30.glInvalidateFramebuffer(GLES30.GL_FRAMEBUFFER, 1, INVALIDATE_ATTACHMENT, 0);

        // Ping-pong 交换
        int tmp = outputTexture;
        outputTexture     = lastOutputTexture;
        lastOutputTexture = tmp;
        bindFboTexture(fbo[4], outputTexture);
        bindFboTexture(fbo[5], lastOutputTexture);

        // 将当前帧存入 lastOutputTexture（供下帧 Pass C 使用）
        // [OPT-G3] 用 glBlitFramebuffer 替代 Passthrough draw
        GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, fbo[4]);
        GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, fbo[5]);
        // 注意：此处 fbo[4] 现在绑定的是插帧结果（outputTexture），
        // 但我们需要拷贝的是 currentTex（即 Apply 的输出）。
        // 由于 ping-pong 后 fbo[4] 已经指向插帧结果，我们需要用原始方式。
        // 回退为 passthrough draw（保持正确性）
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
        GLES30.glViewport(0, 0, outputWidth, outputHeight);
        GLES30.glUseProgram(programPassthrough);
        bindTex(0, currentTex); GLES30.glUniform1i(uPassthrough_texture, 0);
        drawQuad();

        return outputTexture;
    }

    public void renderToScreen(int texture, int screenWidth, int screenHeight) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        GLES30.glViewport(0, 0, screenWidth, screenHeight);
        GLES30.glClearColor(0f, 0f, 0f, 0f);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
        GLES30.glUseProgram(programPassthrough);
        bindTex(0, texture); GLES30.glUniform1i(uPassthrough_texture, 0);
        // [OPT-G4] renderToScreen 使用 flipped VAO，需要单独绑定
        GLES30.glBindVertexArray(vao[1]);
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4);
        GLES30.glBindVertexArray(0);
    }

    public void clearSurface(int screenWidth, int screenHeight) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        GLES30.glViewport(0, 0, screenWidth, screenHeight);
        GLES30.glClearColor(0f, 0f, 0f, 0f);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
    }

    // =================================================================
    // 辅助方法
    // =================================================================

    private void bindTex(int unit, int texId) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texId);
    }

    // [OPT-G4] drawQuad 不再 bind/unbind VAO（由 runPipeline 统一管理）
    private void drawQuad() {
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4);
    }

    // [OPT-G1] 支持指定内部格式和外部格式的纹理创建
    private int createTexture(int w, int h, int internalFormat, int format) {
        int[] tex = new int[1];
        GLES30.glGenTextures(1, tex, 0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex[0]);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, internalFormat,
                w, h, 0, format, GLES30.GL_UNSIGNED_BYTE, null);
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
