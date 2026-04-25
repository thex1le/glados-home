#!/usr/bin/env python3
"""Quick LCD debug tool for the Pi5 ST7789 (spidev) display.

Pushes a single static image to the LCD so you can verify wiring,
SPI access, and orientation without bringing up the full BodyServer
or GLaDOS stacks.

Usage:
    python Tools/test_lcd.py                       # solid color test pattern
    python Tools/test_lcd.py --image path/to.png   # display a file
    python Tools/test_lcd.py --color 255,0,0       # solid red
    python Tools/test_lcd.py --bars                # color bars
    python Tools/test_lcd.py --flip                # rotate 180
"""
import argparse
import sys
from os import path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 240
HEIGHT = 198
DC_BCM = 25
RST_BCM = 24
SPI_PORT = 0
SPI_CS = 0
SPI_HZ = 25_000_000
OFFSET_TOP = 122


def open_display():
    import st7789
    return st7789.ST7789(
        port=SPI_PORT, cs=SPI_CS, dc=DC_BCM, rst=RST_BCM,
        width=WIDTH, height=HEIGHT, rotation=0,
        spi_speed_hz=SPI_HZ,
        offset_left=0, offset_top=OFFSET_TOP,
    )


def parse_color(s):
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--color expects R,G,B, got {s!r}")
    return tuple(parts)


def make_test_pattern():
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Border so you can confirm full extent is reaching the panel
    draw.rectangle([0, 0, WIDTH - 1, HEIGHT - 1], outline=(255, 255, 255), width=2)
    # Corner markers (R top-left, G top-right, B bottom-left, W bottom-right)
    s = 20
    draw.rectangle([0, 0, s, s], fill=(255, 0, 0))
    draw.rectangle([WIDTH - s, 0, WIDTH, s], fill=(0, 255, 0))
    draw.rectangle([0, HEIGHT - s, s, HEIGHT], fill=(0, 0, 255))
    draw.rectangle([WIDTH - s, HEIGHT - s, WIDTH, HEIGHT], fill=(255, 255, 255))
    # Cross hair
    draw.line([0, HEIGHT // 2, WIDTH, HEIGHT // 2], fill=(80, 80, 80))
    draw.line([WIDTH // 2, 0, WIDTH // 2, HEIGHT], fill=(80, 80, 80))
    # Text
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((WIDTH // 2 - 40, HEIGHT // 2 - 12), "LCD OK", fill=(255, 154, 0), font=font)
    return img


def make_color_bars():
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    bars = [
        (255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
        (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0),
    ]
    bw = WIDTH // len(bars)
    for i, c in enumerate(bars):
        draw.rectangle([i * bw, 0, (i + 1) * bw, HEIGHT], fill=c)
    return img


def load_image(p):
    img = Image.open(p).convert("RGB")
    ratio = img.width / img.height
    new_w = WIDTH
    new_h = int(WIDTH / ratio)
    if new_h > HEIGHT:
        new_h = HEIGHT
        new_w = int(HEIGHT * ratio)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    canvas.paste(img, ((WIDTH - new_w) // 2, (HEIGHT - new_h) // 2))
    return canvas


def main():
    parser = argparse.ArgumentParser(description="ST7789 SPI LCD debug tool")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--image", help="path to image file to display")
    g.add_argument("--color", help="solid R,G,B color (e.g. 255,0,0)")
    g.add_argument("--bars", action="store_true", help="show color bars")
    parser.add_argument("--flip", action="store_true", help="rotate 180")
    args = parser.parse_args()

    if args.image:
        if not path.isfile(args.image):
            print(f"No such file: {args.image}", file=sys.stderr)
            sys.exit(1)
        img = load_image(args.image)
    elif args.color:
        img = Image.new("RGB", (WIDTH, HEIGHT), parse_color(args.color))
    elif args.bars:
        img = make_color_bars()
    else:
        img = make_test_pattern()

    if args.flip:
        img = img.rotate(180)

    disp = open_display()
    disp.display(img)
    print(f"Pushed {img.size} image to LCD (port={SPI_PORT} cs={SPI_CS} "
          f"dc={DC_BCM} rst={RST_BCM})")


if __name__ == "__main__":
    main()
