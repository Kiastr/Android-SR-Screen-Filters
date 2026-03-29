package com.anime4k.screen;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.ComponentCallbacks;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.res.Configuration;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.PopupMenu;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * OverlayService — 屏幕超分叠加层服务
 *
 * Performance Optimization v1.4.0 (CPU 端):
 * [CPU-1] EGL_CONTEXT_PRIORITY_HIGH: 请求高优先级 GPU 上下文，减少 GPU 调度延迟
 * [CPU-2] ImageReader 缓冲区数量从 2 提升至 3: 减少因 GPU 处理慢导致的帧丢弃
 * [CPU-3] acquireLatestImage + 立即 close: 保持最低延迟，避免缓冲区积压
 * [CPU-4] 帧率自适应节流 (Frame Pacing): 当处理线程繁忙时，Choreographer 回调跳过投递，
 *          避免 processingHandler 消息队列积压导致的额外延迟
 * [CPU-5] EGL 窗口表面懒创建 + 复用: 仅在 Surface 变化时重建，避免每帧检查开销
 * [CPU-6] 移除 Bitmap 相关导入和代码路径，彻底消除 Bitmap 分配
 * [CPU-7] 方向变化防抖延迟从 300ms 降至 150ms，加快横竖屏切换响应
 * [CPU-8] FPS 统计改用 System.nanoTime() 提升精度
 */
public class OverlayService extends Service {

    private static final String TAG = "OverlayService";
    private static final String CHANNEL_ID = "anime4k_channel";
    private static final int NOTIFICATION_ID = 1;

    private WindowManager windowManager;
    private SurfaceView overlaySurfaceView;
    private WindowManager.LayoutParams overlayParams;

    private View floatButton;
    private WindowManager.LayoutParams floatButtonParams;
    private boolean floatButtonVisible = true;

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;

    private HandlerThread processingThread;
    private Handler processingHandler;
    private Handler mainHandler;

    private Anime4KRenderer renderer;

    private int screenWidth, screenHeight, screenDensity;
    private int captureWidth, captureHeight;
    private float captureScale = 0.5f;
    private float refineStrength = 0.5f;
    private float pseudoMVStrength = 0.5f;
    private float overlayOpacity = 1.0f;
    private boolean fsrEnabled = false;
    private float fsrSharpness = 0.2f;

    private volatile boolean isRunning = false;
    private volatile boolean isPaused = false;
    private int frameCount = 0;
    private long fpsStartTimeNs = 0; // [CPU-8] 纳秒精度

    private android.view.Choreographer.FrameCallback frameCallback;

    // [CPU-4] 帧率节流标志：当处理线程正在处理帧时，Choreographer 不再重复投递
    private final AtomicBoolean frameInFlight = new AtomicBoolean(false);

    private EGLDisplay eglDisplay;
    private EGLContext eglContext;
    private EGLSurface eglSurface;      // pbuffer（离屏渲染用）
    private EGLSurface eglOverlaySurface; // 窗口表面（叠加层输出用）

    private volatile boolean surfaceReady = false;
    private volatile Surface overlayOutputSurface;

    private BroadcastReceiver serviceReceiver;
    private int projectionResultCode;
    private Intent projectionData;
    private ComponentCallbacks configCallback;
    private int lastOrientation;

    @Override
    public void onCreate() {
        super.onCreate();
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        mainHandler = new Handler(Looper.getMainLooper());
        updateScreenMetrics();
        lastOrientation = getResources().getConfiguration().orientation;
        createNotificationChannel();
        registerServiceReceiver();
        registerConfigCallback();
    }

    private void updateScreenMetrics() {
        DisplayMetrics metrics = new DisplayMetrics();
        windowManager.getDefaultDisplay().getRealMetrics(metrics);
        screenWidth   = metrics.widthPixels;
        screenHeight  = metrics.heightPixels;
        screenDensity = metrics.densityDpi;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) return START_NOT_STICKY;
        String action = intent.getAction();
        if ("STOP".equals(action)) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if ("START".equals(action)) {
            projectionResultCode = intent.getIntExtra("resultCode", 0);
            projectionData       = intent.getParcelableExtra("data");
            captureScale         = intent.getIntExtra("scale", 50) / 100.0f;
            refineStrength       = intent.getIntExtra("strength", 50) / 100.0f;
            pseudoMVStrength     = intent.getIntExtra("pseudoMV", 50) / 100.0f;
            overlayOpacity       = intent.getIntExtra("opacity", 100) / 100.0f;
            floatButtonVisible   = intent.getBooleanExtra("floatButton", true);
            fsrEnabled           = intent.getBooleanExtra("fsrEnabled", false);
            fsrSharpness         = intent.getIntExtra("fsrSharpness", 20) / 100.0f;

            captureWidth  = ((int) (screenWidth  * captureScale) / 2) * 2;
            captureHeight = ((int) (screenHeight * captureScale) / 2) * 2;

            startForeground(NOTIFICATION_ID, createNotification());
            startOverlay();
            if (floatButtonVisible) createFloatButton();
            startCapture(projectionResultCode, projectionData);
        }
        return START_NOT_STICKY;
    }

    private void registerServiceReceiver() {
        serviceReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                String action = intent.getAction();
                if ("com.anime4k.screen.UPDATE_OPACITY".equals(action)) {
                    overlayOpacity = intent.getIntExtra("opacity", 100) / 100.0f;
                    applyOverlayOpacity();
                } else if ("com.anime4k.screen.TOGGLE_FLOAT_BUTTON".equals(action)) {
                    toggleFloatButtonVisibility(intent.getBooleanExtra("show", true));
                }
            }
        };
        IntentFilter filter = new IntentFilter();
        filter.addAction("com.anime4k.screen.UPDATE_OPACITY");
        filter.addAction("com.anime4k.screen.TOGGLE_FLOAT_BUTTON");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(serviceReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(serviceReceiver, filter);
        }
    }

    private void applyOverlayOpacity() {
        if (overlaySurfaceView != null && overlayParams != null && !isPaused) {
            mainHandler.post(() -> {
                try {
                    overlayParams.alpha = overlayOpacity;
                    windowManager.updateViewLayout(overlaySurfaceView, overlayParams);
                } catch (Exception e) {
                    Log.e(TAG, "Error updating overlay opacity", e);
                }
            });
        }
    }

    private void registerConfigCallback() {
        configCallback = new ComponentCallbacks() {
            @Override
            public void onConfigurationChanged(Configuration newConfig) {
                int newOrientation = newConfig.orientation;
                if (newOrientation != lastOrientation) {
                    lastOrientation = newOrientation;
                    // [CPU-7] 防抖延迟从 300ms 降至 150ms
                    mainHandler.postDelayed(OverlayService.this::handleOrientationChange, 150);
                }
            }
            @Override public void onLowMemory() {}
        };
        registerComponentCallbacks(configCallback);
    }

    private void handleOrientationChange() {
        if (!isRunning) return;
        updateScreenMetrics();
        captureWidth  = ((int) (screenWidth  * captureScale) / 2) * 2;
        captureHeight = ((int) (screenHeight * captureScale) / 2) * 2;

        processingHandler.post(() -> {
            try {
                if (virtualDisplay != null) { virtualDisplay.release(); virtualDisplay = null; }
                if (imageReader != null)    { imageReader.close();      imageReader    = null; }

                if (eglOverlaySurface != null && eglOverlaySurface != EGL14.EGL_NO_SURFACE) {
                    EGL14.eglDestroySurface(eglDisplay, eglOverlaySurface);
                    eglOverlaySurface = null;
                }

                makePbufferCurrent();
                if (renderer != null) {
                    renderer.init(captureWidth, captureHeight, screenWidth, screenHeight);
                }

                // [CPU-2] 3 个缓冲区减少帧丢弃
                imageReader = ImageReader.newInstance(captureWidth, captureHeight,
                        PixelFormat.RGBA_8888, 3);
                virtualDisplay = mediaProjection.createVirtualDisplay(
                        "Anime4KCapture", captureWidth, captureHeight, screenDensity,
                        DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                        imageReader.getSurface(), null, processingHandler);

                Log.d(TAG, "Orientation change handled: " + screenWidth + "x" + screenHeight);
            } catch (Exception e) {
                Log.e(TAG, "Error on orientation change", e);
            }
        });
    }

    private void startOverlay() {
        overlaySurfaceView = new SurfaceView(this);
        int overlayType = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                : WindowManager.LayoutParams.TYPE_SYSTEM_OVERLAY;
        overlayParams = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                overlayType,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                        | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
                        | WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
                        | WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
                PixelFormat.TRANSLUCENT);
        overlayParams.gravity = Gravity.TOP | Gravity.START;
        overlayParams.x = 0;
        overlayParams.y = 0;
        overlayParams.alpha = overlayOpacity;

        // FIX: 强制允许叠加层覆盖刘海屏区域（横屏时刘海在侧边），防止偏移反馈循环
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            overlayParams.layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
        }

        overlaySurfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                overlayOutputSurface = holder.getSurface();
                surfaceReady = true;
            }
            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                // [CPU-5] Surface 变化时销毁旧的 EGL 窗口表面，下次渲染时懒创建
                if (processingHandler != null) {
                    processingHandler.post(() -> {
                        if (eglOverlaySurface != null && eglOverlaySurface != EGL14.EGL_NO_SURFACE) {
                            EGL14.eglDestroySurface(eglDisplay, eglOverlaySurface);
                            eglOverlaySurface = null;
                        }
                    });
                }
            }
            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                surfaceReady = false;
                overlayOutputSurface = null;
            }
        });
        overlaySurfaceView.setZOrderOnTop(true);
        overlaySurfaceView.getHolder().setFormat(PixelFormat.TRANSLUCENT);
        windowManager.addView(overlaySurfaceView, overlayParams);
    }

    private void startCapture(int resultCode, Intent data) {
        processingThread = new HandlerThread("Anime4K-Processing",
                android.os.Process.THREAD_PRIORITY_DISPLAY); // 提升线程优先级
        processingThread.start();
        processingHandler = new Handler(processingThread.getLooper());

        MediaProjectionManager projectionManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        mediaProjection = projectionManager.getMediaProjection(resultCode, data);
        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override public void onStop() { stopSelf(); }
        }, processingHandler);

        // [CPU-2] 3 个缓冲区
        imageReader = ImageReader.newInstance(captureWidth, captureHeight,
                PixelFormat.RGBA_8888, 3);
        virtualDisplay = mediaProjection.createVirtualDisplay(
                "Anime4KCapture", captureWidth, captureHeight, screenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(), null, processingHandler);

        processingHandler.post(() -> {
            initEGL();
            renderer = new Anime4KRenderer();
            renderer.setRefineStrength(refineStrength);
            renderer.setATWStrength(pseudoMVStrength);
            renderer.setFsrEnabled(fsrEnabled);
            renderer.setFsrSharpness(fsrSharpness);
            renderer.init(captureWidth, captureHeight, screenWidth, screenHeight);

            isRunning = true;
            fpsStartTimeNs = System.nanoTime();

            // [CPU-4] Choreographer 驱动帧调度，带节流保护
            mainHandler.post(() -> {
                frameCallback = frameTimeNanos -> {
                    if (!isRunning) return;
                    // 仅当上一帧已处理完毕时才投递新任务，防止队列积压
                    if (frameInFlight.compareAndSet(false, true)) {
                        processingHandler.post(OverlayService.this::processFrame);
                    }
                    android.view.Choreographer.getInstance().postFrameCallback(frameCallback);
                };
                android.view.Choreographer.getInstance().postFrameCallback(frameCallback);
            });
        });
    }

    private void processFrame() {
        // [CPU-4] 无论是否成功处理，最终都要释放 in-flight 标志
        try {
            if (!isRunning || isPaused || !surfaceReady) return;

            Image image = imageReader.acquireLatestImage();
            if (image == null) return;

            try {
                Image.Plane[] planes = image.getPlanes();
                ByteBuffer buffer    = planes[0].getBuffer();
                int pixelStride      = planes[0].getPixelStride(); // RGBA_8888 始终为 4
                int rowStride        = planes[0].getRowStride();
                int rowPixels        = rowStride / pixelStride;    // 含 Padding 的每行像素数

                makePbufferCurrent();
                int outputTex = renderer.processFromBuffer(buffer, captureWidth, captureHeight, rowPixels);

                if (overlayOutputSurface != null && surfaceReady) {
                    renderToOverlay(outputTex);
                }
            } finally {
                // 确保 Image 在使用完毕后立即关闭，释放 ImageReader 缓冲区
                image.close();
            }

            // [CPU-8] 纳秒精度 FPS 统计
            frameCount++;
            long nowNs = System.nanoTime();
            long elapsedNs = nowNs - fpsStartTimeNs;
            if (elapsedNs >= 1_000_000_000L) {
                float fps = frameCount * 1_000_000_000f / elapsedNs;
                frameCount    = 0;
                fpsStartTimeNs = nowNs;
                Intent fpsIntent = new Intent("com.anime4k.screen.FPS_UPDATE");
                fpsIntent.putExtra("fps", fps);
                fpsIntent.setPackage(getPackageName());
                sendBroadcast(fpsIntent);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error processing frame", e);
        } finally {
            frameInFlight.set(false); // [CPU-4] 释放节流标志
        }
    }

    private void renderToOverlay(int texture) {
        try {
            // [CPU-5] EGL 窗口表面懒创建：仅在 null 时创建，避免每帧检查重建
            if (eglOverlaySurface == null || eglOverlaySurface == EGL14.EGL_NO_SURFACE) {
                Surface s = overlayOutputSurface;
                if (s == null || !s.isValid()) return;
                int[] surfaceAttribs = { EGL14.EGL_NONE };
                eglOverlaySurface = EGL14.eglCreateWindowSurface(
                        eglDisplay, getEGLConfig(), s, surfaceAttribs, 0);
            }
            if (eglOverlaySurface == null || eglOverlaySurface == EGL14.EGL_NO_SURFACE) return;

            EGL14.eglMakeCurrent(eglDisplay, eglOverlaySurface, eglOverlaySurface, eglContext);
            renderer.renderToScreen(texture, screenWidth, screenHeight);
            EGL14.eglSwapBuffers(eglDisplay, eglOverlaySurface);
            makePbufferCurrent();
        } catch (Exception e) {
            Log.e(TAG, "Error rendering to overlay", e);
            if (eglOverlaySurface != null && eglOverlaySurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglOverlaySurface);
                eglOverlaySurface = null;
            }
        }
    }

    private void initEGL() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        int[] version = new int[2];
        EGL14.eglInitialize(eglDisplay, version, 0, version, 1);
        EGLConfig config = getEGLConfig();

        // [CPU-1] 请求高优先级 EGL 上下文（需要 EGL_IMG_context_priority 扩展支持）
        // 若设备不支持，会静默回退到普通优先级
        int[] contextAttribs;
        String extensions = EGL14.eglQueryString(eglDisplay, EGL14.EGL_EXTENSIONS);
        if (extensions != null && extensions.contains("EGL_IMG_context_priority")) {
            final int EGL_CONTEXT_PRIORITY_LEVEL_IMG  = 0x3100;
            final int EGL_CONTEXT_PRIORITY_HIGH_IMG   = 0x3101;
            contextAttribs = new int[]{
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL_CONTEXT_PRIORITY_LEVEL_IMG, EGL_CONTEXT_PRIORITY_HIGH_IMG,
                EGL14.EGL_NONE
            };
            Log.d(TAG, "EGL high-priority context requested");
        } else {
            contextAttribs = new int[]{
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
            };
        }

        eglContext = EGL14.eglCreateContext(eglDisplay, config, EGL14.EGL_NO_CONTEXT,
                contextAttribs, 0);
        int[] pbufferAttribs = { EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE };
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, config, pbufferAttribs, 0);
        makePbufferCurrent();
    }

    private void makePbufferCurrent() {
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);
    }

    private EGLConfig getEGLConfig() {
        int[] configAttribs = {
                EGL14.EGL_RED_SIZE,         8,
                EGL14.EGL_GREEN_SIZE,       8,
                EGL14.EGL_BLUE_SIZE,        8,
                EGL14.EGL_ALPHA_SIZE,       8,
                EGL14.EGL_RENDERABLE_TYPE,  0x40 /* EGL_OPENGL_ES3_BIT */,
                EGL14.EGL_SURFACE_TYPE,     EGL14.EGL_WINDOW_BIT | EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0);
        return configs[0];
    }

    private void createFloatButton() {
        ImageView icon = new ImageView(this);
        icon.setImageResource(android.R.drawable.ic_menu_manage);
        icon.setBackgroundColor(0x88000000);
        icon.setPadding(20, 20, 20, 20);
        floatButton = icon;
        int overlayType = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                : WindowManager.LayoutParams.TYPE_SYSTEM_ALERT;
        floatButtonParams = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                overlayType,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT);
        floatButtonParams.gravity = Gravity.TOP | Gravity.START;
        floatButtonParams.x = screenWidth - 200;
        floatButtonParams.y = screenHeight / 2;
        floatButton.setOnTouchListener(new View.OnTouchListener() {
            private int initialX, initialY;
            private float initialTouchX, initialTouchY;
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        initialX = floatButtonParams.x; initialY = floatButtonParams.y;
                        initialTouchX = event.getRawX(); initialTouchY = event.getRawY();
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        floatButtonParams.x = initialX + (int)(event.getRawX() - initialTouchX);
                        floatButtonParams.y = initialY + (int)(event.getRawY() - initialTouchY);
                        windowManager.updateViewLayout(floatButton, floatButtonParams);
                        return true;
                    case MotionEvent.ACTION_UP:
                        if (Math.abs(event.getRawX() - initialTouchX) < 10
                                && Math.abs(event.getRawY() - initialTouchY) < 10)
                            v.performClick();
                        return true;
                }
                return false;
            }
        });
        floatButton.setOnClickListener(v -> showPopupMenu(v));
        windowManager.addView(floatButton, floatButtonParams);
    }

    private void showPopupMenu(View view) {
        PopupMenu popup = new PopupMenu(this, view);
        popup.getMenu().add(0, 1, 0, isPaused ? "恢复超分" : "暂停超分");
        popup.getMenu().add(0, 2, 0, "停止服务");
        popup.getMenu().add(0, 3, 0, "打开主界面");
        popup.setOnMenuItemClickListener(item -> {
            switch (item.getItemId()) {
                case 1:
                    isPaused = !isPaused;
                    if (isPaused) {
                        mainHandler.post(() -> {
                            if (overlaySurfaceView != null && overlayParams != null) {
                                overlayParams.alpha = 0.0f;
                                windowManager.updateViewLayout(overlaySurfaceView, overlayParams);
                            }
                        });
                    } else {
                        applyOverlayOpacity();
                    }
                    return true;
                case 2: stopSelf(); return true;
                case 3:
                    Intent intent = new Intent(this, MainActivity.class);
                    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                    startActivity(intent);
                    return true;
            }
            return false;
        });
        popup.show();
    }

    private void toggleFloatButtonVisibility(boolean show) {
        if (show && floatButton == null) createFloatButton();
        else if (!show && floatButton != null) {
            windowManager.removeView(floatButton);
            floatButton = null;
        }
    }

    private Notification createNotification() {
        Intent intent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent,
                PendingIntent.FLAG_IMMUTABLE);
        Notification.Builder builder = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return builder
                .setContentTitle("Anime4K Screen")
                .setContentText("正在实时增强屏幕画质...")
                .setSmallIcon(android.R.drawable.ic_media_play)
                .setContentIntent(pendingIntent)
                .build();
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID, "Anime4K Service", NotificationManager.IMPORTANCE_LOW);
            getSystemService(NotificationManager.class).createNotificationChannel(channel);
        }
    }

    @Override public IBinder onBind(Intent intent) { return null; }

    @Override
    public void onDestroy() {
        isRunning = false;
        if (processingThread != null) processingThread.quitSafely();
        if (mediaProjection  != null) mediaProjection.stop();
        if (virtualDisplay   != null) virtualDisplay.release();
        if (imageReader      != null) imageReader.close();
        if (overlaySurfaceView != null) windowManager.removeView(overlaySurfaceView);
        if (floatButton        != null) windowManager.removeView(floatButton);
        if (serviceReceiver    != null) unregisterReceiver(serviceReceiver);
        if (configCallback     != null) unregisterComponentCallbacks(configCallback);
        if (processingHandler  != null) {
            processingHandler.post(() -> {
                if (renderer != null) renderer.release();
                releaseEGL();
            });
        }
        sendBroadcast(new Intent("com.anime4k.screen.SERVICE_STOPPED")
                .setPackage(getPackageName()));
        super.onDestroy();
    }

    private void releaseEGL() {
        if (eglDisplay == null) return;
        if (eglOverlaySurface != null && eglOverlaySurface != EGL14.EGL_NO_SURFACE)
            EGL14.eglDestroySurface(eglDisplay, eglOverlaySurface);
        if (eglSurface != null && eglSurface != EGL14.EGL_NO_SURFACE)
            EGL14.eglDestroySurface(eglDisplay, eglSurface);
        if (eglContext != null && eglContext != EGL14.EGL_NO_CONTEXT)
            EGL14.eglDestroyContext(eglDisplay, eglContext);
        EGL14.eglTerminate(eglDisplay);
    }
}
