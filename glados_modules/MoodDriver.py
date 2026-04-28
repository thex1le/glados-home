"""MoodDriver — translates the brain's PAD vector into hardware commands.

Step 7c-3 (Option C): replaces GladosMood as the source of truth for eye
animation + LCD breathing speed. Reads PAD from MoodConsumer, computes the
desired hardware state with hysteresis to avoid flickering, and publishes
the existing body/led + body/lcd commands so LedHead and GladosLCD need
no changes.

Only runs when [FEATURES] mood_source = pad. When mood_source = legacy
(the default), GladosMood drives the hardware exactly as today.
"""

# native imports
from threading import Thread
from time import sleep
from typing import Any, Dict, Optional

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, MQTTEnums, LCDEnums,
                                        LEDHead, SystemEnums)
from glados_modules.MoodConsumer import MoodConsumer, PADState
from glados_modules.MqttConnector import MQTTClient, LEDMessageBuilder


# Hysteresis thresholds. Switch in is the trigger; switch out has more slack
# so PAD oscillating around the boundary doesn't flap the hardware.
ANGRY_TRIGGER_AROUSAL = 0.7
ANGRY_TRIGGER_PLEASURE = -0.3
ANGRY_RELEASE_AROUSAL = 0.5
ANGRY_RELEASE_PLEASURE = -0.1

FAST_BREATH_TRIGGER = 0.5
FAST_BREATH_RELEASE = 0.3


class MoodDriver(Thread, MQTTClient):
    """Drives LedHead + GladosLCD from PAD via existing MQTT commands.

    Attributes:
        logger: Logger instance for logging.
        consumer: MoodConsumer instance providing the latest PAD vector.
        configFile: Configuration parser instance loaded from glog.conf.
        topic_handler: Empty — driver only publishes, doesn't subscribe.
    """

    TICK_INTERVAL: float = 1.0  # how often to re-evaluate target state

    def __init__(self, config_file: Any, consumer: MoodConsumer) -> None:
        """Initialize the driver.

        Args:
            config_file: configparser instance loaded from glog.conf.
            consumer: A started MoodConsumer to read PAD from.
        """
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.configFile = config_file
        self.consumer = consumer
        self.stop: bool = False

        # Hysteresis state — the target we're holding right now
        self._current_eye_animation: Optional[str] = None
        self._current_breath_fast: Optional[bool] = None

        # MQTTClient needs a topic_handler attribute even when we don't subscribe
        self.topic_handler: Dict = {}
        mqtt_c = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        MQTTClient.__init__(self,
                            ip=mqtt_c[SystemEnums.MQTT_SERVER_IP.value],
                            port=int(mqtt_c[SystemEnums.MQTT_PORT.value]))

    # ------------------------------------------------------------------
    # PAD → hardware mapping (with hysteresis)
    # ------------------------------------------------------------------

    def _target_eye_animation(self, pad: PADState) -> str:
        """Decide which eye animation should be active given current PAD.

        Hysteresis: once 'angry_eye' is the current animation, only release
        back to 'normal_eye' when arousal AND pleasure both move into the
        release band. Prevents flickering when PAD hovers near the boundary.
        """
        is_angry_now = self._current_eye_animation == LEDHead.ANIMATION_ANGRY_EYE_KEY.value
        if is_angry_now:
            # Stay angry until clearly released
            if (pad.arousal < ANGRY_RELEASE_AROUSAL
                    and pad.pleasure > ANGRY_RELEASE_PLEASURE):
                return LEDHead.ANIMATION_NORMAL_EYE_KEY.value
            return LEDHead.ANIMATION_ANGRY_EYE_KEY.value
        # Currently normal — only flip to angry when the trigger conditions hit
        if (pad.arousal > ANGRY_TRIGGER_AROUSAL
                and pad.pleasure < ANGRY_TRIGGER_PLEASURE):
            return LEDHead.ANIMATION_ANGRY_EYE_KEY.value
        return LEDHead.ANIMATION_NORMAL_EYE_KEY.value

    def _target_breath_fast(self, pad: PADState) -> bool:
        """Decide whether breath should be fast (high arousal) or slow.

        Hysteresis on arousal: trigger fast at 0.5, release back to slow at 0.3.
        """
        if self._current_breath_fast is True:
            return pad.arousal >= FAST_BREATH_RELEASE
        return pad.arousal > FAST_BREATH_TRIGGER

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    def _publish_eye_animation(self, animation_key: str) -> None:
        """Publish a head LED animation command in LedHead's expected shape."""
        payload = {LEDHead.MSG_COMMAND_LOCATION_KEY.value: LEDHead.EYE_LED_LOCATION.value,
                   LEDHead.MSG_COMMAND_KEY.value: animation_key}
        msg = LEDMessageBuilder.send_led_animation(payload)
        try:
            self.send_command(msg, MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value)
        except Exception as e:
            self.logger.error(f"Failed to publish eye animation: {e}")

    def _publish_breath(self, fast: bool) -> None:
        """Publish a set_breath command to both LCD eyes."""
        for location in (SystemEnums.LEFT_LCD.value, SystemEnums.RIGHT_LCD.value):
            msg = {
                LCDEnums.MSG_LOCATION_KEY.value: location,
                LCDEnums.MSG_COMMAND_KEY.value: LCDEnums.COMMAND_SET_BREATH.value,
                # GladosLCD.set_breath_options requires all three keys
                LCDEnums.OPTIONS_KEY.value: {
                    "fast": bool(fast),
                    "animation": True,
                    "rainbow": False,
                },
            }
            try:
                self.send_command(msg, MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value)
            except Exception as e:
                self.logger.error(f"Failed to publish breath ({location}): {e}")

    # ------------------------------------------------------------------
    # Tick — evaluate, transition, publish
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """One evaluation cycle: read PAD, compute targets, publish on change.

        When PAD is stale (consumer hasn't seen an update in the staleness
        window), hold whatever state we last published — don't react to a
        possibly-stale baseline. If we've never published, do nothing yet.
        """
        if self.consumer.is_stale():
            self.logger.debug("PAD stale; holding current hardware state.")
            return

        pad = self.consumer.get_pad()
        target_anim = self._target_eye_animation(pad)
        target_fast = self._target_breath_fast(pad)

        if target_anim != self._current_eye_animation:
            self.logger.info(
                f"Eye animation: {self._current_eye_animation} → {target_anim} "
                f"(P={pad.pleasure:+.2f} A={pad.arousal:+.2f})")
            self._publish_eye_animation(target_anim)
            self._current_eye_animation = target_anim

        if target_fast != self._current_breath_fast:
            self.logger.info(
                f"Breath fast: {self._current_breath_fast} → {target_fast} "
                f"(A={pad.arousal:+.2f})")
            self._publish_breath(target_fast)
            self._current_breath_fast = target_fast

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.logger.info("MoodDriver started; PAD now drives hardware.")
        while self.stop is False:
            try:
                self._tick()
            except Exception as e:
                self.logger.error(f"MoodDriver tick failed: {e}")
            sleep(self.TICK_INTERVAL)
