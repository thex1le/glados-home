# GLaDOS Animatronic Robot

An animatronic GLaDOS (Portal) robot head with distributed computing across three systems communicating via MQTT.

## Systems

| System | Entry Point | Hardware | Role |
|--------|-------------|----------|------|
| Pi4 B+ | `BodyServer.py` | Servos, LEDs, LCDs, sensors, head camera | Body control |
| Pi5 | `GLaDOS.py` | 2x fisheye cameras, voice bonnet, IMU | Main controller, voice |
| GPU Server (Ubuntu, 4090) | `AiServer.py` | NVIDIA GPU | YOLO, pose estimation, WhisperX, TTS |

Config file passed to each: `python3 <entry>.py -config glog.conf`

There is one config file **schema** (`glog.conf`) but each system gets its own copy filled out with the correct IPs, paths, and hardware settings for that machine. The section names and key names are identical across all three -- only the values differ. The `glog.conf` in the repo is a **template** with generic placeholder IPs (e.g. `192.168.1.2`) -- the actual deployed configs with real IPs are not committed. Never put real/private IP addresses in committed files. When adding a new config section or key:
- Add it to `GladosEnums.py` as an enum
- Add it to the `glog.conf` template in the repo
- Document which systems use it and what value each system should have
- Never assume a section/key exists only on one system -- all three parse the same schema, they just ignore sections they don't use

Start order: BodyServer first (starts MQTT-dependent services), then AiServer (loads ML models), then GLaDOS (connects to everything).

### External module: gladosTTS
`AiServer.py` imports `from gladosTTS import engine as glados_voice`. This TTS engine is NOT in this repo -- it is a separate project that must be installed on the GPU server. `glados_voice.main()` runs the TTS HTTP server that Pi5 calls for speech synthesis.

### Required asset directories
These directories must exist with their files or the system will crash at startup:
- `wav/` -- `ding_on.wav`, `ding_off.wav`, `portal_still_alive.wav`, `portal2_want_you_gone.wav`
- `aperture_logo/` -- numbered BMP frames (`apature_logo_0001.bmp` etc.) for LCD startup animation
- `txt_responses/` -- one response per line: `greetings.txt`, `processing.txt`, `insults.txt`, `questions.txt`, `question_response.txt`, `fuck.txt`, `cancel_response.txt`

### No graceful shutdown
There is no signal handling. On Ctrl+C or crash:
- Daemon threads die immediately
- Servos hold their last commanded position (may stress mechanics)
- MQTT connections are not cleanly disconnected
- No servo-centering on shutdown
This is a known gap. If adding shutdown handling, center all servos and disconnect MQTT before exit.

## Coding Style

Follow these conventions to match the existing codebase. When modifying existing code, adopt the style of the surrounding code. When writing new code, follow these rules.

### Naming
- **snake_case** for variables, functions, methods, modules
- **PascalCase** for classes
- **UPPER_SNAKE_CASE** for enum members and class-level constants
- Spell it **Glados** in class/variable names (e.g. `GladosLocal`, `GladosLCD`, `GladosEnums`) not `GLADOS` or `glados`. The exception is `GLaDOS.py` which is the main entry point filename.

### Imports
Group imports in this order with comment headers:
```python
# builtin
from threading import Thread
from time import sleep

# 3rd party
import cv2
import paho.mqtt.client as mqtt

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import ServoEnum, SystemEnums
```

### Type Hints
- Use type hints on all function signatures and return types
- Import from `typing` module: `Dict`, `Callable`, `Tuple`, `Any`, `Optional`, `List`, `NamedTuple`
- Always annotate `__init__` return as `-> None`

### Docstrings
- Use Google-style docstrings on all public classes and methods
- Include `Args:` and `Returns:` sections where applicable
- Class docstrings should describe purpose and key behavior
- Private methods (`__method`) get docstrings too if the logic is non-obvious
```python
def send_status(self, location: str, results: dict) -> dict:
    """Send current servo status to MQTT.

    Args:
        location: Servo location identifier.
        results: Dict of current servo state.

    Returns:
        The status message dict that was sent.
    """
```

### String Formatting
- Use **f-strings** everywhere: `f"Moving to {angle}"`
- Do not use `.format()` or `%` formatting in new code
- When fixing existing `.format()` calls, convert to f-strings

### Comparisons
- Use bare truthiness: `if value:` not `if value is True:`
- Use `is None` / `is not None` for None checks (this is correct)
- Use `not` for negation: `if not connected:` not `if connected is False:`
- Note: the existing codebase has many `is True`/`is False` comparisons. When touching a line that uses this pattern, fix it. Do not go out of your way to fix unrelated lines.

### Classes
- Most classes inherit from `Thread` and/or `MQTTClient`
- Use explicit parent init calls (this is the established pattern):
  ```python
  Thread.__init__(self)
  self.daemon = True
  MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
  ```
- Initialize `self.logger` early using `setup_logger(name=self.__name__)`
- Set `self.daemon = True` on all daemon threads (not `Thread.daemon = True`)
- **Process subclasses**: Only store config in `__init__`. All hardware init (I2C, SPI, GPIO, cameras, GStreamer) goes in `run()` which executes in the child process after fork. See `CameraModule.Camera` for the pattern.

### Enums
- **All string constants that appear in MQTT messages, config keys, or dict keys must use enums** from `GladosEnums.py`
- Access with `.value`: `ServoEnum.MSG_COMMAND_MOVE.value`
- When adding a new message key or MQTT topic, add the enum first, then use it
- Never hardcode MQTT topics, servo locations, camera names, or message keys as raw strings
- Tuning parameters (thresholds, timeouts, speeds) belong in enums or config, not inline

### Error Handling
- Catch specific exceptions where possible (`JSONDecodeError`, `KeyboardInterrupt`)
- Always log the full traceback: `self.logger.error(f"Error: {traceback.format_exc()}")`
- Add exponential backoff in retry loops (see `MachineVision.py` tracker loop pattern)
- Never silently swallow exceptions

### Logging
- Every class gets a logger: `self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)`
- Use `self.logger.debug()` for high-frequency operational messages
- Use `self.logger.info()` for startup/shutdown and state transitions
- Use `self.logger.error()` for failures, always include context
- Do not use `print()` for operational output -- use the logger

### Comments
- Explain **why**, not **what**
- Use `# TODO` for known incomplete work
- Do not add section separator comments like `# ========` or `# --- Methods ---`
- Inline comments on the same line as code are fine for non-obvious logic

### MQTT Message Patterns
- All messages include a `uuid` field (added automatically by `MQTTClient.send_command`)
- Use `ServoMessageBuilder`, `CameraMessageBuilder`, etc. for constructing messages
- Use `move_all` for coordinated multi-servo commands
- Include `trace_id` and `ts_vision` on vision pipeline messages for debugging

## Gotchas

- **Hardware init must happen AFTER process fork**: `CameraModule.Camera` extends `multiprocessing.Process`. I2C bus handles, SPI connections, Picamera2 instances, and GStreamer pipelines hold file descriptors and kernel-side state that do not survive `fork()`. All hardware initialization MUST happen inside `run()` (child process), never in `__init__` (parent process). If you create a new Process subclass that touches hardware, follow the same pattern: config only in `__init__`, hardware init in `run()`. Moving hardware init to `__init__` will cause silent failures, bus lockups, or segfaults after fork.
- **torch.load monkey patch** (`MachineVision.py:19-31`): Forces `weights_only=False` to load YOLOv8 weights. Do not remove this -- model loading will fail without it.
- **LCD startup deadlock**: `GladosLCD` startup via MQTT is intentionally commented out. Calling startup through MQTT causes a deadlock that freezes the system. Only call `__startup()` directly.
- **Camera is a Process, not a Thread**: `CameraModule.Camera` extends `multiprocessing.Process`. Its frame buffer is NOT accessible from the main process. The web dashboard uses RTSP consumers on Pi4/Pi5 for this reason.
- **5-second camera delay** (`GLaDOS.py:59`): Left camera must fully initialize before right camera starts, or both fail. This sleep is required.
- **MQTT initial connection has no retry**: If the MQTT broker is unreachable at startup, `client.connect()` in `MqttConnector.__init__` will crash the program. The broker must be running before any GLaDOS system starts.
- **Working directory matters**: WAV files in `GLaDOSLocal.py` resolve relative to `getcwd()`. Always run from the glados-home root: `cd /path/to/glados-home && python3 GLaDOS.py -config glog.conf`
- **OpenAI wait loop**: `GLaDOS.py:78` polls `gladosgpt.real_response` in a while loop with `sleep(0.3)`. If the OpenAI API hangs, the main loop blocks indefinitely with no timeout.
- **I2C bus sharing**: All sensors (IMU, SHT40, TOF, ENS160) share the same I2C bus. Address conflicts at init will hang the bus. If adding a new sensor, verify its I2C address doesn't collide.
- **MQTT UUID deduplication**: Every message gets a UUID with a 60-second TTL cache. If the same message is received twice within 60 seconds (QoS 1 redelivery), the duplicate is silently dropped. This is intentional.
- **`handler` must be initialized before the lock block** in `MQTTClient.on_message()`: The `handler` variable is only assigned inside a conditional (`if uuid not in cache`). It must be initialized to `None` before the `with self._lock:` block, otherwise duplicate UUIDs cause `UnboundLocalError` that silently kills the MQTT callback thread.

## Hardware Pin Map (Pi4)

```
PCA9685 Servo Hat (16-channel, I2C):
  Channel 0: body_left_right  (GS3508MG, pulse 635-2665us)
  Channel 1: body_up_down     (MG92B, pulse 605-2550us)
  Channel 2: head_left_right  (MG92B, pulse 605-2550us)
  Channel 3: head_up_down     (MG90D, pulse 610-2665us)
  Channel 4: PWM LED (head backlight, duty cycle)

NeoPixel LEDs:
  GPIO D12: Shoulder LEDs (64 pixels, RGB)
  GPIO D18: Head eye LED (1 pixel, RGB)
  GPIO D21: Lamp strip (8 pixels, RGBW)

SPI LCD (ST7789 240x198):
  CS=CE0, DC=D25, RST=D24, SCK/MOSI=default SPI
  baudrate=25MHz, rotation=0, x_offset=0, y_offset=122

I2C Sensors (board.SCL/SDA, STEMMA):
  BNO055 IMU, SHT40 Temp/Humidity, VL53L4CX TOF, ENS160 MOX Gas
```

## Config File (glog.conf)

INI format parsed with `ConfigParser`. All systems read the same file.

Required sections: `[DEFAULT]`, `[CAMERAS]`, `[STT]`, `[SERVOS]`, `[RTSP]`, `[MQTT]`, `[YOLO]`, `[LOCALSPEAK]`, `[OPENAI]`

Optional sections: `[HOMEASSISTANT]` (weather data), `[DASHBOARD]` (web dashboard port, default 8080)

Missing required sections cause `KeyError` at startup. Missing optional keys use `.get()` fallbacks where implemented.

Key servo config values (angles are degrees, pulse widths in microseconds):
```ini
[SERVOS]
default_max_min_center = 180,0,90
head_min_max_center = 125,6,83
neck_min_max_center = 120,52,92
mg90d_pulse = 2665,610
mg92b_pulse = 2550,605
gs3508mg_pulse = 2665,635
```

## MQTT Topics

| Topic | Direction | System | Purpose |
|-------|-----------|--------|---------|
| `body/servo` | cmd in | Pi4 | Servo move/move_all/status commands |
| `body/servo/status` | status out | Pi4 | Servo position, velocity, moving flag |
| `body/led` | cmd in | Pi4 | LED animation commands |
| `body/lcd` | cmd in | Pi4 | LCD display commands |
| `body/imu/status` | status out | Pi5 | IMU sensor data (100ms interval) |
| `body/tof/status` | status out | Pi4 | TOF distance readings (100ms) |
| `body/th/status` | status out | Pi4 | Temperature/humidity (100ms) |
| `body/mox/status` | status out | Pi4 | Air quality AQI/TVOC/eCO2 (100ms) |
| `vision/camera/results` | data out | GPU | YOLO detection + pose results |
| `sst/results` | data out | GPU | WhisperX transcription results |
| `system/track` | cmd in | GPU | Start tracking command |
| `system/health/<host>` | status out | All | Thread health heartbeat (5s) |
| `system/log_level` | cmd in | All | Remote log level changes |
| `intensity` | cmd in | Pi4 | System-wide intensity multiplier |

## Thread Model

All worker threads are daemon threads. If the main thread dies, everything dies.

**Pi4 (BodyServer.py):** 4x Gservo (50Hz physics loop each), GladosLCD, Camera (Process), HealthMonitor, WebDashboard

**Pi5 (GLaDOS.py):** IMU, GladosLocal, GladosSTT, 2x Camera (Process), HealthMonitor, WebDashboard. Main thread runs the voice interaction loop.

**GPU (AiServer.py):** MLDetect (spawns 3 camera tracker threads internally), AudioServerRX, HealthMonitor, WebDashboard. TTS engine runs in foreground after all threads start.

If a thread dies: HealthMonitor reports it as dead on the dashboard, but there is no automatic restart. The system continues running with that component non-functional.

## External Services

| Service | Required By | Failure Behavior |
|---------|-------------|-----------------|
| MQTT broker | All systems | Crash at startup (no retry on initial connect) |
| TTS engine | Pi5 (GLaDOSLocal) | Cannot speak, error logged, continues |
| OpenAI API | Pi5 (GLaDOS.py) | Main loop blocks indefinitely waiting for response |
| Home Assistant | Pi5 (GLaDOSLocal) | Weather unavailable, fails silently |
| WhisperX | GPU (AiServer) | Audio transcription unavailable |

## Cross-System Contracts (change one side, break the other)

These are the interfaces between systems where a change on one side silently breaks the other. If you modify any of these, update BOTH sides.

### Servo status message format
`Gservo.get_angles()` (Pi4) produces a dict that `ServoLocation.servo_handle_cmd()` (GPU) parses into a `ServoTuple` namedtuple. The fields and their order must match:
```
current, max, min, middle, axis, moving, location, last, velocity
```
If you add/remove/rename a field in `get_angles()`, you must also update:
- `ServoTuple` definition in `MqttConsumerModules.py`
- `servo_handle_cmd()` parsing in `MqttConsumerModules.py`
- `SpringDamperEstimator.sync()` in `VisionTracker.py` (reads `current` and `velocity`)

### move_all message format
`ServoMessageBuilder.move_all()` (GPU) builds a message that `Gservo.handle_cmd()` (Pi4) parses. The format is:
```json
{"cmd": "move_all", "targets": {"head_left_right": {"angle": 90, "speed": 3}, ...}}
```
If you change key names, update both `MqttConnector.py` (builder) and `BodyControlModules.py` (parser). The trace fields `trace_id` and `ts_vision` are also attached to this message by `VisionTracker._update_targets()` and read by `Gservo.handle_cmd()`.

### Vision results message format
`MachineVision.__process_image()` (GPU) builds results that `VisionTracker` (GPU) and `MqttConsumerModules.VisionTracker` (GPU) parse. If you change the detection result structure (box keys, pose keys, confidence key), update:
- `__translate_results()` and `assign_key_points_to_response()` in `MachineVision.py`
- `VisionTracker.parse_camera()` in `MqttConsumerModules.py`
- `MotionTrack.track_loop()` in `VisionTracker.py`
- `MotionRecorder` replay functions (which recompute from recorded inputs)

### Spring-damper physics must match on both sides
`Gservo.run()` (Pi4) and `SpringDamperEstimator` (GPU) run the same physics equation:
```
accel = omega^2 * (target - position) - 2 * zeta * omega * velocity
```
If you change the physics model on one side, the GPU's position estimate will diverge from the Pi4's actual servo position, causing tracking errors. Both must use identical math. The `MotionProfile` omega/zeta tables must also stay in sync.

### UD interpolation table
`MotionProfile.HEAD_UD_TO_BODY_UD_TABLE` is used by both `MotionTrack._interpolate_body_ud()` (GPU) and was previously in `__level_servos()`. This table is empirically calibrated from the physical robot. Do not change these values without physical testing. If the mechanical geometry changes (different servo, different mounting), re-calibrate and update the table.

### Camera mounting offsets
`MotionProfile.CAMERA_LEFT_MOUNTING_OFFSET` (-55.0) and `CAMERA_RIGHT_MOUNTING_OFFSET` (55.0) define the angle of the side cameras relative to the body center. These are used in `_pixel_to_world_angle()` to convert side camera detections to world-space angles. If the physical camera mounting changes, these must be re-measured.

### Motion recording replay compatibility
`MotionRecorder.build_frame_record()` records inputs and `MotionReplay.replay()` re-runs the math. If you change `_pixel_to_world_angle()` or `_update_targets()` logic, old recordings will produce different outputs when replayed (which is expected -- that's the point of regression testing). But if you change the **recording format** (field names, structure), old recordings become unreadable. Bump a version number or write a migration.

### HealthMonitor thread registration
Any new thread or process that should be monitored must be registered with `health.register("name", thread)` in the entry point. If you add a new daemon thread and don't register it, it can die silently with no indication on the dashboard.

### LCD displays span two systems
There are two ST7789 round LCDs -- one on Pi4 SPI and one on Pi5 SPI. Currently only the Pi4 LCD is implemented (`GladosLCD` in `BodyControlModules.py`, `RIGHT_LCD`). The Pi5 LCD (`LEFT_LCD`) is a TODO (`GLaDOSLocal.py:172`).

When the Pi5 LCD is added:
- `GladosLCD` will need to work on both Pi4 and Pi5 (same class, different SPI pins)
- Most of the time both LCDs show the same animation (breathing, aperture logo)
- Sometimes they show different content (e.g., one eye angry, one normal)
- LCD commands via MQTT must support: "both" (default), "left only", "right only"
- The `LCDEnums.MSG_LOCATION_KEY` (`"lcd"`) already routes by location string (`RIGHT_LCD`/`LEFT_LCD`), so per-LCD targeting works. A "broadcast to both" mode will need a wildcard or separate "both" command.
- The LCD startup deadlock gotcha applies to both -- never trigger startup via MQTT, only via direct `__startup()` call
- Pi5 LCD init must happen in the main process (not inside a Camera Process fork) since it uses SPI hardware

### Config file schema (glog.conf)
One config schema, three fills. Every system reads the same section and key names -- the values differ per machine (IPs, camera numbers, paths). If you:
- **Add a new config key**: add the enum in `GladosEnums.py`, add the key to `glog.conf` in the repo, and use `.get()` with a sensible fallback so systems that don't need it don't crash.
- **Rename a config key**: update the enum, the config file, AND every system's deployed copy. There's no migration -- the old key just stops being read.
- **Add a new section**: all three systems will parse it (ConfigParser doesn't error on extra sections). Only the systems that import the corresponding enum will actually read it.
- **Remove a section/key**: make sure no system still references the enum. Search all entry points and their import chains.

The `[DEFAULT]` section in ConfigParser is special -- its values are inherited as fallbacks by all other sections. Don't put system-specific values there.

### WebDashboard feed access patterns
- GPU server: uses `rtsp_server=mv.rtsp` for direct frame buffer access (in-process)
- Pi4/Pi5: uses `feed_uris={"label": "rtsp://..."}` for RTSP consumer (cross-process)

Do not mix these. If you try to pass a Camera's RTSPServer as `rtsp_server` on Pi4/Pi5, it will be `None` because the RTSPServer lives in the Camera's child process, not the main process.

## Architecture

### Communication
- **MQTT** for all inter-system commands and status (broker on GPU server)
- **RTSP** for camera streams (GStreamer)
- **TCP** for audio streaming (Pi5 -> GPU)
- **HTTP** for TTS and OpenAI API calls

### Motion System
- Spring-damper physics loop at 50Hz on Pi4 (`Gservo.run()`)
- World-space angle tracking on GPU (`VisionTracker.py`)
- Head locks on fast (high omega), body follows slow (low omega)
- Parameters in `MotionProfile` enum, tunable per emotion speed level 1-5
- No servo encoder feedback -- position is estimated from spring-damper math

### Debugging
- Pipeline trace IDs flow from vision through tracking to servo (`TraceLog.py`)
- Health heartbeats on `system/health/<hostname>` every 5s (`HealthMonitor.py`)
- RTSP debug overlay shows tracking state on annotated video
- Remote log level control via `system/log_level` MQTT topic
- Motion recording/replay for regression testing (`MotionRecorder.py`)
- `python collect_debug.py` bundles logs/recordings into a zip for offline analysis

### Web Dashboard
- Each system serves status + video at `http://<ip>:<port>` (port from `[DASHBOARD]` in config)
- GPU shows annotated feeds (direct frame buffer), Pi4/Pi5 show raw feeds (RTSP consumer)

## Key Files

| File | Purpose |
|------|---------|
| `glados_modules/GladosEnums.py` | All enums, constants, config keys |
| `glados_modules/BodyControlModules.py` | Servo spring-damper, sensors, LEDs, LCD |
| `glados_modules/VisionTracker.py` | World-space tracking, spring-damper estimator, idle drift |
| `glados_modules/MachineVision.py` | YOLO + pose detection, RTSP annotation, trace stamping |
| `glados_modules/MqttConnector.py` | MQTT base client, message builders, log level control |
| `glados_modules/MqttConsumerModules.py` | Servo/sensor/vision status consumers |
| `glados_modules/MotionRecorder.py` | Frame recording and replay for regression testing |
| `glados_modules/TraceLog.py` | Pipeline trace ID generation and JSONL logging |
| `glados_modules/HealthMonitor.py` | Thread health monitoring and MQTT heartbeat |
| `glados_modules/WebDashboard.py` | Flask web dashboard with MJPEG video feeds |
| `glog.conf` | Shared config file (IPs, ports, servo params, etc.) |

## Testing

Run tests locally before deploying to the remote systems:
```bash
python -m pytest Tests/ -v
```

Tests run on Windows/Linux/Mac with no hardware. `Tests/conftest.py` mocks all hardware dependencies (adafruit, GPIO, picamera2, torch, GStreamer) at the module level so glados_modules can be imported anywhere.

### Test files

**Unit tests (individual methods):**
| File | Tests | What it validates |
|------|-------|-------------------|
| `test_enums.py` | 18 | Enum completeness, MQTT topics, servo locations, MotionProfile tables |
| `test_spring_damper.py` | 10 | Physics convergence, overshoot, limits, head-faster-than-body, estimator sync |
| `test_mqtt_messages.py` | 8 | Message builder output matches parser expectations |
| `test_mqtt_client.py` | 13 | UUID deduplication, send_command UUID injection, remote log level control |
| `test_world_space_tracking.py` | 11 | Pixel-to-world math, head/body target split, UD interpolation, EMA smoothing |
| `test_gservo.py` | 21 | Command parsing (move/move_all/status), speed/angle clamping, spring param selection |
| `test_glados_local.py` | 26 | Command regex matching, time parsing, timer regex, temperature regex, sight processing |
| `test_health_monitor.py` | 14 | Thread registration, error counting, status format, peer tracking, timeout detection |
| `test_egg_timer.py` | 7 | Timer start/stop, completion callback, remaining time |
| `test_lcd_led.py` | 8 | LCD command parsing (set/get breath), breath options format, location filtering |
| `test_motion_recorder.py` | 7 | Recording/replay format, smoothness metrics, comparison logic |
| `test_trace_log.py` | 6 | Trace ID generation, JSONL writing, incomplete trace handling |

**Integration tests (multiple components):**
| File | Tests | What it validates |
|------|-------|-------------------|
| `test_tracking_pipeline.py` | 9 | Full detection -> world angle -> target split -> MQTT message, trace propagation |
| `test_gservo_loop.py` | 5 | Threaded 50Hz physics loop convergence, mid-motion target change, limits |
| `test_config_parsing.py` | 14 | glog.conf sections/keys parse correctly, servo/camera values valid |
| `test_message_roundtrip.py` | 12 | Serialize -> deserialize for servo, sensor, and vision messages |
| `test_web_dashboard.py` | 5 | Flask routes return correct HTML/JSON/MJPEG content types |
| `test_recorder_roundtrip.py` | 3 | Record -> replay -> compare produces consistent results |

### When writing new code
- **Run `python -m pytest Tests/ -v` after every change.** All 200 tests must pass before deploying.
- Add tests for any new enum, message format, or calculation
- If you change a cross-system contract (servo status format, move_all format, vision results), add a test that validates the format matches on both sides
- Tests must pass with `conftest.py` hardware mocks -- do not import hardware libraries directly in tests
- Dev machine dependencies: `pip install pytest flexmock cachetools regex flask`

## Dependencies

Separate requirements files per system:
- `requirements_pi4.txt` -- adafruit hardware + MQTT + Flask
- `requirements_pi5.txt` -- voice/audio + cameras + MQTT + Flask
- `requirements_gpu.txt` -- torch + ultralytics + rtmlib + whisperx + Flask
