"""Tests for GladosLCD command parsing and LED handler dispatch."""

import pytest
from unittest.mock import MagicMock, patch
from json import dumps

from glados_modules.GladosEnums import LCDEnums, MQTTEnums, LEDHead, LEDShoulders


class TestGladosLCDHandleCmd:
    """Test LCD MQTT command parsing."""

    def _make_lcd(self, location="right_lcd"):
        from glados_modules.BodyControlModules import GladosLCD
        with patch.object(GladosLCD, '__init__', lambda self, *a, **kw: None):
            lcd = GladosLCD.__new__(GladosLCD)
        lcd.location = location
        lcd.logger = MagicMock()
        lcd.send_command = MagicMock()
        lcd.breath_fast = False
        lcd.breath_animation = True
        lcd.rainbow = False
        lcd._lock = __import__('threading').Lock()
        return lcd

    def _make_msg(self, payload):
        msg = MagicMock()
        msg.payload.decode.return_value = dumps(payload)
        msg.topic = MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value
        return msg

    def test_set_breath_command(self):
        lcd = self._make_lcd("right_lcd")
        msg = self._make_msg({
            LCDEnums.MSG_LOCATION_KEY.value: "right_lcd",
            LCDEnums.MSG_COMMAND_KEY.value: LCDEnums.COMMAND_SET_BREATH.value,
            LCDEnums.OPTIONS_KEY.value: {"fast": True, "animation": False, "rainbow": True},
        })
        lcd.handle_cmd(msg)
        assert lcd.breath_fast is True
        assert lcd.breath_animation is False
        assert lcd.rainbow is True
        assert lcd.send_command.called

    def test_get_breath_command(self):
        lcd = self._make_lcd("right_lcd")
        lcd.breath_fast = True
        lcd.rainbow = True
        msg = self._make_msg({
            LCDEnums.MSG_LOCATION_KEY.value: "right_lcd",
            LCDEnums.MSG_COMMAND_KEY.value: LCDEnums.COMMAND_GET_BREATH.value,
        })
        lcd.handle_cmd(msg)
        assert lcd.send_command.called
        sent = lcd.send_command.call_args[0][0]
        assert "right_lcd" in sent

    def test_wrong_location_ignored(self):
        lcd = self._make_lcd("right_lcd")
        msg = self._make_msg({
            LCDEnums.MSG_LOCATION_KEY.value: "left_lcd",
            LCDEnums.MSG_COMMAND_KEY.value: LCDEnums.COMMAND_SET_BREATH.value,
            LCDEnums.OPTIONS_KEY.value: {"fast": True, "animation": True, "rainbow": True},
        })
        lcd.handle_cmd(msg)
        assert lcd.breath_fast is False  # unchanged

    def test_unknown_command_ignored(self):
        lcd = self._make_lcd("right_lcd")
        msg = self._make_msg({
            LCDEnums.MSG_LOCATION_KEY.value: "right_lcd",
            LCDEnums.MSG_COMMAND_KEY.value: "nonexistent",
        })
        lcd.handle_cmd(msg)
        assert not lcd.send_command.called


class TestGladosLCDBreathOptions:
    def _make_lcd(self):
        from glados_modules.BodyControlModules import GladosLCD
        with patch.object(GladosLCD, '__init__', lambda self, *a, **kw: None):
            lcd = GladosLCD.__new__(GladosLCD)
        lcd.location = "right_lcd"
        lcd.breath_fast = False
        lcd.breath_animation = True
        lcd.rainbow = False
        return lcd

    def test_set_options(self):
        lcd = self._make_lcd()
        lcd.set_breath_options({"fast": True, "animation": False, "rainbow": True})
        assert lcd.breath_fast is True
        assert lcd.breath_animation is False
        assert lcd.rainbow is True

    def test_get_options_format(self):
        lcd = self._make_lcd()
        lcd.breath_fast = True
        lcd.rainbow = True
        result = lcd.get_breath_options()
        assert "response" in result
        resp = result["response"]
        assert resp["fast"] is True
        assert resp["rainbow"] is True
        assert resp["animation"] is True
        assert resp["location"] == "right_lcd"
