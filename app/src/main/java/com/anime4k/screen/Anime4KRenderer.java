package com.anime4k.screen;

import android.graphics.Bitmap;
import android.opengl.GLES30;
import android.opengl.GLUtils;
import android.util.Log;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class Anime4KRenderer {
    private static final String TAG = "Anime4KRenderer";
    
    private static final String VERTEX_SHADER = "#version 300 es\nin vec4 aPosition;\nin vec2 aTexCoord;\nout vec2 vTexCoord;\nvoid main() {\n    gl_Position = aPosition;\n    vTexCoord = aTexCoord;\n}\n";
    private static final String FRAG_PASSTHROUGH = "#version 300 es\nprecision lowp float;\nin vec2 vTexCoord;\nuniform sampler2D uTexture;\nout vec4 fragColor;\nvoid main() { fragColor = texture(uTexture, vTexCoord); }\n";
    private static final String FRAG_ATW = "#version 300 es\nprecision mediump float;\nin vec2 vTexCoord;\nuniform sampler2D uTexture;\nuniform sampler2D uLastTexture;\nuniform float uATWStrength;\nuniform vec2 uOffset;\nout vec4 fragColor;\nvoid main() {\n    vec4 current = texture(uTexture, vTexCoord);\n    if (uATWStrength <= 0.0) { fragColor = current; return; }\n    vec4 last = texture(uLastTexture, vTexCoord + uOffset * uATWStrength);\n    fragColor = mix(current, last, 0.5 * uATWStrength);\n}\n";

    private int[] fbo = new int[10];
    private int[] vao = new int[2];
    private int[] vbo = new int[2];
    private boolean initialized = false;
    private int inputWidth, inputHeight, outputWidth, outputHeight;
    
    private float refineStrength = 0.0f;
    private float atwStrength = 0.0f;
    private boolean fsrEnabled = false;
    private float fsrSharpness = 0.0f;

    private int inputTexture, outputTexture, lastOutputTexture, fsrTempTexture;
    private int programFsrEasu, programFsrRcas, programATW, programPassthrough;
    private int uEasu_texture, uEasu_con1, uRcas_texture, uRcas_con, uATW_texture, uATW_lastTexture, uATW_strength, uATW_offset, uPassthrough_texture;

    public void init(int inW, int inH, int outW, int outH) {
        if (initialized) release();
        this.inputWidth = inW; this.inputHeight = inH;
        this.outputWidth = outW; this.outputHeight = outH;

        inputTexture = createTexture(inW, inH, GLES30.GL_RGBA8, GLES30.GL_RGBA);
        fsrTempTexture = createTexture(outW, outH, GLES30.GL_RGBA8, GLES30.GL_RGBA);
        outputTexture = createTexture(outW, outH, GLES30.GL_RGBA8, GLES30.GL_RGBA);
        lastOutputTexture = createTexture(outW, outH, GLES30.GL_RGBA8, GLES30.GL_RGBA);

        GLES30.glGenFramebuffers(10, fbo, 0);
        bindFboTexture(fbo[6], fsrTempTexture);
        bindFboTexture(fbo[4], outputTexture);
        bindFboTexture(fbo[5], lastOutputTexture);

        programFsrEasu = createProgram(FSRShaders.VERTEX_SHADER, FSRShaders.FRAG_EASU);
        uEasu_texture = GLES30.glGetUniformLocation(programFsrEasu, "uTexture");
        uEasu_con1 = GLES30.glGetUniformLocation(programFsrEasu, "uEasuCon1");

        programFsrRcas = createProgram(FSRShaders.VERTEX_SHADER, FSRShaders.FRAG_RCAS);
        uRcas_texture = GLES30.glGetUniformLocation(programFsrRcas, "uTexture");
        uRcas_con = GLES30.glGetUniformLocation(programFsrRcas, "uRcasCon");

        programATW = createProgram(VERTEX_SHADER, FRAG_ATW);
        uATW_texture = GLES30.glGetUniformLocation(programATW, "uTexture");
        uATW_lastTexture = GLES30.glGetUniformLocation(programATW, "uLastTexture");
        uATW_strength = GLES30.glGetUniformLocation(programATW, "uATWStrength");
        uATW_offset = GLES30.glGetUniformLocation(programATW, "uOffset");

        programPassthrough = createProgram(VERTEX_SHADER, FRAG_PASSTHROUGH);
        uPassthrough_texture = GLES30.glGetUniformLocation(programPassthrough, "uTexture");

        initVao();
        initialized = true;
    }

    private void initVao() {
        float[] quadPos = {-1f, 1f, -1f, -1f, 1f, 1f, 1f, -1f};
        float[] quadTex = {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        GLES30.glGenVertexArrays(2, vao, 0);
        GLES30.glGenBuffers(2, vbo, 0);

        GLES30.glBindVertexArray(vao[0]);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo[0]);
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, quadPos.length * 4, FloatBuffer.wrap(quadPos), GLES30.GL_STATIC_DRAW);
        GLES30.glEnableVertexAttribArray(0);
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, 0);

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo[1]);
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, quadTex.length * 4, FloatBuffer.wrap(quadTex), GLES30.GL_STATIC_DRAW);
        GLES30.glEnableVertexAttribArray(1);
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 0, 0);
        GLES30.glBindVertexArray(0);
    }

    // --- OverlayService.java 调用的 Setter 方法 ---
    public void setRefineStrength(float strength) { this.refineStrength = strength / 100.0f; }
    public void setATWStrength(float strength) { this.atwStrength = strength / 100.0f; }
    public void setFsrEnabled(boolean enabled) { this.fsrEnabled = enabled; }
    public void setFsrSharpness(float sharpness) { this.fsrSharpness = sharpness / 10.0f; }

    public int processFromBuffer(ByteBuffer buffer, int width, int height, int rowPixels) {
        if (!initialized) return 0;
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, inputTexture);
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1);
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, rowPixels);
        buffer.position(0);
        GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer);
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ROW_LENGTH, 0);
        return runPipeline();
    }

    private int runPipeline() {
        GLES30.glBindVertexArray(vao[0]);
        if (!fsrEnabled) {
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[4]);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programPassthrough);
            bindTex(0, inputTexture); GLES30.glUniform1i(uPassthrough_texture, 0);
            drawQuad();
        } else {
            // FSR 路径 (v1.12.0 绝对基准修复版)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[6]);
            // [FIX-v1.12.0] 激进清空策略：牺牲亮度保无残影
            GLES30.glClearColor(0f, 0f, 0f, 1f); 
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programFsrEasu);
            bindTex(0, inputTexture); GLES30.glUniform1i(uEasu_texture, 0);
            float[] easuCon1 = {1f/inputWidth, 1f/inputHeight, 0f, 0f};
            GLES30.glUniform4fv(uEasu_con1, 1, easuCon1, 0);
            drawQuad();

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[4]);
            GLES30.glClearColor(0f, 0f, 0f, 1f);
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
            GLES30.glViewport(0, 0, outputWidth, outputHeight);
            GLES30.glUseProgram(programFsrRcas);
            bindTex(0, fsrTempTexture); GLES30.glUniform1i(uRcas_texture, 0);
            GLES30.glUniform4f(uRcas_con, fsrSharpness, 0, 0, 0);
            drawQuad();

            if (atwStrength > 0.01f) {
                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[5]);
                GLES30.glClearColor(0f, 0f, 0f, 1f);
                GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
                GLES30.glViewport(0, 0, outputWidth, outputHeight);
                GLES30.glUseProgram(programATW);
                bindTex(0, outputTexture); GLES30.glUniform1i(uATW_texture, 0);
                bindTex(1, lastOutputTexture); GLES30.glUniform1i(uATW_lastTexture, 1);
                GLES30.glUniform1f(uATW_strength, atwStrength);
                GLES30.glUniform2f(uATW_offset, 0.001f, 0.001f);
                drawQuad();
                
                int tmp = outputTexture;
                outputTexture = lastOutputTexture;
                lastOutputTexture = tmp;
                bindFboTexture(fbo[4], outputTexture);
                bindFboTexture(fbo[5], lastOutputTexture);
            }
        }
        GLES30.glBindVertexArray(0);
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        return outputTexture;
    }

    public void renderToScreen(int texture, int screenWidth, int screenHeight) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        GLES30.glViewport(0, 0, screenWidth, screenHeight);
        GLES30.glClearColor(0f, 0f, 0f, 0f);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
        GLES30.glUseProgram(programPassthrough);
        bindTex(0, texture); GLES30.glUniform1i(uPassthrough_texture, 0);
        drawQuad();
    }

    public void clearSurface(int screenWidth, int screenHeight) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0);
        GLES30.glViewport(0, 0, screenWidth, screenHeight);
        GLES30.glClearColor(0f, 0f, 0f, 0f);
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
    }

    private void bindFboTexture(int fboId, int texId) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId);
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, texId, 0);
    }

    private int createTexture(int w, int h, int internalFormat, int format) {
        int[] tex = new int[1];
        GLES30.glGenTextures(1, tex, 0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex[0]);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, internalFormat, w, h, 0, format, GLES30.GL_UNSIGNED_BYTE, null);
        return tex[0];
    }

    private int createProgram(String vert, String frag) {
        int vs = loadShader(GLES30.GL_VERTEX_SHADER, vert);
        int fs = loadShader(GLES30.GL_FRAGMENT_SHADER, frag);
        int prog = GLES30.glCreateProgram();
        GLES30.glAttachShader(prog, vs);
        GLES30.glAttachShader(prog, fs);
        GLES30.glLinkProgram(prog);
        return prog;
    }

    private int loadShader(int type, String src) {
        int shader = GLES30.glCreateShader(type);
        GLES30.glShaderSource(shader, src);
        GLES30.glCompileShader(shader);
        return shader;
    }

    private void bindTex(int unit, int texId) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texId);
    }

    private void drawQuad() {
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4);
    }

    public void release() {
        if (!initialized) return;
        GLES30.glDeleteFramebuffers(10, fbo, 0);
        GLES30.glDeleteTextures(4, new int[]{inputTexture, fsrTempTexture, outputTexture, lastOutputTexture}, 0);
        initialized = false;
    }
}
