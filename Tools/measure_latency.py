#!/usr/bin/env python3
"""measure_latency.py — calibrate the speech-eye sync.

Run on the Pi 5 brain side. Measures:
    1. MQTT one-way latency from brain → body via probe/echo.
    2. Audio output latency via tone playback + mic capture.

Then prints the recommended value for [BRAIN] speech_led_delay_s. Drop
the recommended value into glog.conf and the speech-eye sync is calibrated
to your hardware.

The LatencyResponder must be running on the Pi 4 (BodyServer.py wires it
in unconditionally as of step 8a-0). The audio test requires sounddevice +
a microphone in earshot of the speaker.

Usage:
    python3 Tools/measure_latency.py -config glog.conf
    python3 Tools/measure_latency.py -config glog.conf --skip-audio
    python3 Tools/measure_latency.py -config glog.conf --pings 200
"""

# native imports
from argparse import ArgumentParser
from configparser import ConfigParser
from os import path
import sys
from time import sleep

# glados imports
from glados_modules.GladosEnums import LatencyEnums
from glados_modules.LatencyProbe import LatencyProbe, LatencyProbeException


def _print_stats(label: str, stats: dict) -> None:
    print(f"  {label}:")
    print(f"    samples: {stats[LatencyEnums.SAMPLE_COUNT_KEY.value]}")
    print(f"    mean:    {stats[LatencyEnums.MEAN_MS_KEY.value]:7.2f} ms")
    print(f"    median:  {stats[LatencyEnums.MEDIAN_MS_KEY.value]:7.2f} ms")
    print(f"    p95:     {stats[LatencyEnums.P95_MS_KEY.value]:7.2f} ms")
    print(f"    p99:     {stats[LatencyEnums.P99_MS_KEY.value]:7.2f} ms")


def main() -> int:
    parser = ArgumentParser(description="GLaDOS speech-eye sync calibration")
    parser.add_argument('-config', type=str, default=None, dest='conf', nargs=1,
                        required=True, help='Config File (glog.conf)')
    parser.add_argument('--pings', type=int, default=100,
                        help='Number of MQTT probes (default 100)')
    parser.add_argument('--ping-interval', type=float, default=0.05,
                        help='Seconds between probes (default 0.05)')
    parser.add_argument('--ping-timeout', type=float, default=2.0,
                        help='Seconds to wait per probe (default 2.0)')
    parser.add_argument('--no-ntp', action='store_true',
                        help='Use round-trip / 2 instead of responder timestamp')
    parser.add_argument('--skip-audio', action='store_true',
                        help='Skip the audio output measurement')
    parser.add_argument('--audio-reps', type=int, default=5,
                        help='Audio measurement repetitions (default 5)')
    parser.add_argument('--mic-delay-ms', type=float,
                        default=LatencyEnums.DEFAULT_MIC_CAPTURE_DELAY_MS.value,
                        help='Estimated mic capture delay to subtract (ms)')
    args = parser.parse_args()

    config_path = args.conf[0]
    if not path.isfile(config_path):
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    cp = ConfigParser()
    cp.read(config_path)

    print("=" * 60)
    print("  GLaDOS speech-eye sync latency probe")
    print("=" * 60)
    print()

    probe = LatencyProbe(cp)
    probe.start()
    sleep(0.5)  # let MQTT subscribe settle

    # ------------------------------------------------------------------
    # MQTT pipeline
    # ------------------------------------------------------------------
    print(f"[1/2] Measuring MQTT pipeline ({args.pings} probes)...")
    mqtt_results = probe.measure_mqtt(
        n_pings=args.pings,
        interval_s=args.ping_interval,
        timeout_s=args.ping_timeout,
        assume_ntp_synced=not args.no_ntp,
    )
    print()
    print(f"  sent={mqtt_results['sent']} lost={mqtt_results['lost']}")
    if mqtt_results['lost'] >= mqtt_results['sent']:
        print("\033[91m  ERROR: every probe was lost. Is LatencyResponder "
              "running on the body machine?\033[0m")
        return 2
    _print_stats("round-trip", mqtt_results['round_trip'])
    _print_stats("one-way   ", mqtt_results['one_way'])
    mqtt_one_way_ms = mqtt_results['one_way'][LatencyEnums.MEAN_MS_KEY.value]
    print()

    # ------------------------------------------------------------------
    # Audio pipeline
    # ------------------------------------------------------------------
    audio_output_latency_ms = None
    if args.skip_audio:
        print("[2/2] Audio measurement skipped (--skip-audio)")
        print()
    else:
        print(f"[2/2] Measuring audio pipeline ({args.audio_reps} tone bursts)...")
        print("      You should hear a series of short beeps.")
        try:
            audio_results = probe.measure_audio(
                repetitions=args.audio_reps,
                mic_capture_delay_ms=args.mic_delay_ms,
            )
            _print_stats("output latency", audio_results['output_latency_ms'])
            audio_output_latency_ms = audio_results['output_latency_ms'][
                LatencyEnums.MEAN_MS_KEY.value]
            samples = audio_results['raw_samples_ms']
            if samples:
                print(f"  raw samples (ms): "
                      f"{[f'{s:.1f}' for s in samples]}")
            print(f"  mic capture delay subtracted: "
                  f"{audio_results['mic_capture_delay_ms_assumed']:.1f} ms")
        except LatencyProbeException as e:
            print(f"\033[93m  Audio measurement skipped: {e}\033[0m")
            print("  Re-run with --skip-audio or install sounddevice.")
        print()

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Calibration recommendation")
    print("=" * 60)
    if audio_output_latency_ms is None:
        print(f"  MQTT one-way latency: {mqtt_one_way_ms:.2f} ms")
        print("  Audio latency: not measured.")
        print()
        print("  Cannot recommend speech_led_delay_s without audio data.")
        print("  Re-run without --skip-audio, or measure manually:")
        print("    1. Set [BRAIN] speech_led_delay_s = 0.5")
        print("    2. Trigger a TTS phrase, watch the eye + speaker.")
        print("    3. If LED is early, decrease by 50ms; if late, increase.")
        return 0

    delay_s = LatencyProbe.recommend_speech_led_delay_s(
        mqtt_one_way_ms=mqtt_one_way_ms,
        audio_output_latency_ms=audio_output_latency_ms,
    )
    print(f"  MQTT one-way latency:    {mqtt_one_way_ms:7.2f} ms")
    print(f"  Audio output latency:    {audio_output_latency_ms:7.2f} ms")
    print(f"  Recommended delay:       {delay_s:7.3f} s "
          f"(= audio - mqtt_one_way)")
    print()
    print("  Set in glog.conf:")
    print(f"    [BRAIN]")
    print(f"    speech_led_delay_s = {delay_s:.3f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
