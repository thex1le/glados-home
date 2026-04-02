#!/usr/bin/env python3
import sys
import wave
import pyaudio

CHUNK = 1024


def open_audio_stream(wave_file_path):
    # Open the wave file
    wf = wave.open(wave_file_path, 'rb')
    p = pyaudio.PyAudio()
    # Open an output stream using the file’s parameters
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )
    return p, stream, wf


def play_audio(stream, wf):
    # Rewind the file for a fresh playback
    wf.rewind()
    data = wf.readframes(CHUNK)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)


def main():
    if len(sys.argv) < 2:
        print("Usage: python play_wave_stream.py <path_to_wave_file>")
        sys.exit(1)

    wave_file_path = sys.argv[1]
    print(f"Opening audio device and loading {wave_file_path}...")

    # Open the stream once and keep it open
    p, stream, wf = open_audio_stream(wave_file_path)

    try:
        # You can call play_audio as many times as needed using the same stream
        print("Playing audio...")
        play_audio(stream, wf)
        # For example, play it again:
        print("Playing audio again...")
        play_audio(stream, wf)
    finally:
        # When done, close the stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()
        print("Audio device closed.")


if __name__ == "__main__":
    main()
