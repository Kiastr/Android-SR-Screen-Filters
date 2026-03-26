package com.anime4k.screen;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.View;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;

import com.google.android.material.button.MaterialButton;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_MEDIA_PROJECTION = 1001;
    private static final int REQUEST_OVERLAY_PERMISSION = 1002;

    private MaterialButton toggleButton;
    private TextView statusText, fpsText, scaleValue, strengthValue, pseudoMVValue, opacityValue, fsrSharpnessValue, strengthLabel, fsrSharpnessLabel, githubLink;
    private SeekBar scaleSeekBar, strengthSeekBar, pseudoMVSeekBar, opacitySeekBar, fsrSharpnessSeekBar;
    private SwitchCompat floatButtonSwitch, fsrSwitch;
    private boolean isRunning = false;

    private SharedPreferences prefs;

    private BroadcastReceiver fpsReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if ("com.anime4k.screen.FPS_UPDATE".equals(intent.getAction())) {
                float fps = intent.getFloatExtra("fps", 0);
                fpsText.setText(String.format("FPS: %.1f", fps));
            } else if ("com.anime4k.screen.SERVICE_STOPPED".equals(intent.getAction())) {
                isRunning = false;
                updateUI();
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        prefs = getSharedPreferences("anime4k_prefs", MODE_PRIVATE);

        toggleButton = findViewById(R.id.toggleButton);
        statusText = findViewById(R.id.statusText);
        fpsText = findViewById(R.id.fpsText);
        scaleValue = findViewById(R.id.scaleValue);
        strengthValue = findViewById(R.id.strengthValue);
        pseudoMVValue = findViewById(R.id.pseudoMVValue);
        opacityValue = findViewById(R.id.opacityValue);
        fsrSharpnessValue = findViewById(R.id.fsrSharpnessValue);
        
        strengthLabel = findViewById(R.id.strengthLabel);
        fsrSharpnessLabel = findViewById(R.id.fsrSharpnessLabel);
        githubLink = findViewById(R.id.githubLink);

        scaleSeekBar = findViewById(R.id.scaleSeekBar);
        strengthSeekBar = findViewById(R.id.strengthSeekBar);
        pseudoMVSeekBar = findViewById(R.id.pseudoMVSeekBar);
        opacitySeekBar = findViewById(R.id.opacitySeekBar);
        fsrSharpnessSeekBar = findViewById(R.id.fsrSharpnessSeekBar);
        
        floatButtonSwitch = findViewById(R.id.floatButtonSwitch);
        fsrSwitch = findViewById(R.id.fsrSwitch);

        // Restore preferences
        scaleSeekBar.setProgress(prefs.getInt("scale", 50));
        strengthSeekBar.setProgress(prefs.getInt("strength", 50));
        pseudoMVSeekBar.setProgress(prefs.getInt("pseudoMV", 50));
        opacitySeekBar.setProgress(prefs.getInt("opacity", 100));
        fsrSharpnessSeekBar.setProgress(prefs.getInt("fsrSharpness", 20));
        floatButtonSwitch.setChecked(prefs.getBoolean("floatButton", true));
        fsrSwitch.setChecked(prefs.getBoolean("fsrEnabled", false));

        updateScaleLabel(scaleSeekBar.getProgress());
        updateStrengthLabel(strengthSeekBar.getProgress());
        updatePseudoMVLabel(pseudoMVSeekBar.getProgress());
        updateOpacityLabel(opacitySeekBar.getProgress());
        updateFsrSharpnessLabel(fsrSharpnessSeekBar.getProgress());
        updateFsrVisibility(fsrSwitch.isChecked());

        scaleSeekBar.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                updateScaleLabel(progress);
                prefs.edit().putInt("scale", progress).apply();
            }
        });

        strengthSeekBar.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                updateStrengthLabel(progress);
                prefs.edit().putInt("strength", progress).apply();
            }
        });

        fsrSharpnessSeekBar.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                updateFsrSharpnessLabel(progress);
                prefs.edit().putInt("fsrSharpness", progress).apply();
            }
        });

        pseudoMVSeekBar.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                updatePseudoMVLabel(progress);
                prefs.edit().putInt("pseudoMV", progress).apply();
            }
        });

        opacitySeekBar.setOnSeekBarChangeListener(new SimpleSeekBarListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                updateOpacityLabel(progress);
                prefs.edit().putInt("opacity", progress).apply();
                if (isRunning) {
                    Intent opacityIntent = new Intent("com.anime4k.screen.UPDATE_OPACITY");
                    opacityIntent.putExtra("opacity", progress);
                    opacityIntent.setPackage(getPackageName());
                    sendBroadcast(opacityIntent);
                }
            }
        });

        floatButtonSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            prefs.edit().putBoolean("floatButton", isChecked).apply();
            if (isRunning) {
                Intent toggleIntent = new Intent("com.anime4k.screen.TOGGLE_FLOAT_BUTTON");
                toggleIntent.putExtra("show", isChecked);
                toggleIntent.setPackage(getPackageName());
                sendBroadcast(toggleIntent);
            }
        });

        fsrSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            prefs.edit().putBoolean("fsrEnabled", isChecked).apply();
            updateFsrVisibility(isChecked);
        });

        toggleButton.setOnClickListener(v -> {
            if (isRunning) {
                stopUpscaling();
            } else {
                startUpscaling();
            }
        });

        githubLink.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(getString(R.string.github_url)));
            startActivity(intent);
        });

        IntentFilter filter = new IntentFilter();
        filter.addAction("com.anime4k.screen.FPS_UPDATE");
        filter.addAction("com.anime4k.screen.SERVICE_STOPPED");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(fpsReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(fpsReceiver, filter);
        }
    }

    private void updateFsrVisibility(boolean fsrEnabled) {
        if (fsrEnabled) {
            fsrSharpnessLabel.setVisibility(View.VISIBLE);
            fsrSharpnessSeekBar.setVisibility(View.VISIBLE);
            fsrSharpnessValue.setVisibility(View.VISIBLE);
            strengthLabel.setVisibility(View.GONE);
            strengthSeekBar.setVisibility(View.GONE);
            strengthValue.setVisibility(View.GONE);
        } else {
            fsrSharpnessLabel.setVisibility(View.GONE);
            fsrSharpnessSeekBar.setVisibility(View.GONE);
            fsrSharpnessValue.setVisibility(View.GONE);
            strengthLabel.setVisibility(View.VISIBLE);
            strengthSeekBar.setVisibility(View.VISIBLE);
            strengthValue.setVisibility(View.VISIBLE);
        }
    }

    private void updateScaleLabel(int progress) {
        scaleValue.setText(progress + "%");
    }

    private void updateStrengthLabel(int progress) {
        strengthValue.setText(String.format("%.2f", progress / 100.0f));
    }

    private void updateFsrSharpnessLabel(int progress) {
        fsrSharpnessValue.setText(String.format("%.2f", progress / 100.0f));
    }

    private void updatePseudoMVLabel(int progress) {
        pseudoMVValue.setText(String.format("%.2f", progress / 100.0f));
    }

    private void updateOpacityLabel(int progress) {
        opacityValue.setText(progress + "%");
    }

    private void startUpscaling() {
        if (!Settings.canDrawOverlays(this)) {
            Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName()));
            startActivityForResult(intent, REQUEST_OVERLAY_PERMISSION);
            Toast.makeText(this, "请授予悬浮窗权限", Toast.LENGTH_LONG).show();
            return;
        }

        MediaProjectionManager projectionManager =
                (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        startActivityForResult(projectionManager.createScreenCaptureIntent(),
                REQUEST_MEDIA_PROJECTION);
    }

    private void stopUpscaling() {
        Intent intent = new Intent(this, OverlayService.class);
        intent.setAction("STOP");
        startService(intent);
        isRunning = false;
        updateUI();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_OVERLAY_PERMISSION) {
            if (Settings.canDrawOverlays(this)) {
                startUpscaling();
            } else {
                Toast.makeText(this, "需要悬浮窗权限才能运行", Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == REQUEST_MEDIA_PROJECTION) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                Intent serviceIntent = new Intent(this, OverlayService.class);
                serviceIntent.setAction("START");
                serviceIntent.putExtra("resultCode", resultCode);
                serviceIntent.putExtra("data", data);
                serviceIntent.putExtra("scale", scaleSeekBar.getProgress());
                serviceIntent.putExtra("strength", strengthSeekBar.getProgress());
                serviceIntent.putExtra("pseudoMV", pseudoMVSeekBar.getProgress());
                serviceIntent.putExtra("opacity", opacitySeekBar.getProgress());
                serviceIntent.putExtra("floatButton", floatButtonSwitch.isChecked());
                serviceIntent.putExtra("fsrEnabled", fsrSwitch.isChecked());
                serviceIntent.putExtra("fsrSharpness", fsrSharpnessSeekBar.getProgress());

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent);
                } else {
                    startService(serviceIntent);
                }

                isRunning = true;
                updateUI();
                moveTaskToBack(true);
            } else {
                Toast.makeText(this, "需要屏幕录制权限", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void updateUI() {
        if (isRunning) {
            toggleButton.setText("停止超分");
            toggleButton.setBackgroundTintList(
                    android.content.res.ColorStateList.valueOf(0xFFCF6679));
            statusText.setText("运行中");
            statusText.setTextColor(0xFF03DAC5);
        } else {
            toggleButton.setText("启动超分");
            toggleButton.setBackgroundTintList(
                    android.content.res.ColorStateList.valueOf(0xFF6200EE));
            statusText.setText("已停止");
            statusText.setTextColor(0xFFFFFFFF);
            fpsText.setText("FPS: --");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            unregisterReceiver(fpsReceiver);
        } catch (Exception e) {
            // ignore
        }
    }

    static abstract class SimpleSeekBarListener implements SeekBar.OnSeekBarChangeListener {
        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {}
        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {}
    }
}
