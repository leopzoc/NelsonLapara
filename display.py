"""
Display Backend — fullscreen color output for the RPi 5.

Supports three backends:
  • framebuffer  — direct /dev/fb0 write (no X required, fastest)
  • pygame       — SDL2 window (good for HDMI with desktop)
  • tkinter      — fallback for dev machines
"""

from __future__ import annotations

import logging
import struct
from abc import ABC, abstractmethod

import config as cfg

log = logging.getLogger(__name__)


class DisplayBackend(ABC):
    @abstractmethod
    def show_color(self, hex_color: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


# ── Framebuffer (headless RPi) ──────────────────────────────────────

class FramebufferDisplay(DisplayBackend):
    """Write raw BGRA pixels to /dev/fb0."""

    def __init__(
        self,
        device: str = cfg.FRAMEBUFFER_DEVICE,
        width: int = cfg.SCREEN_WIDTH,
        height: int = cfg.SCREEN_HEIGHT,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.bpp = 4   # 32-bit BGRA

    def show_color(self, hex_color: str) -> None:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        pixel = struct.pack("BBBB", b, g, r, 255)  # BGRA
        row = pixel * self.width
        screen = row * self.height
        try:
            with open(self.device, "wb") as fb:
                fb.write(screen)
        except PermissionError:
            log.error("Cannot write to %s — run with sudo or add user to 'video' group", self.device)

    def close(self) -> None:
        pass


# ── Pygame (desktop / HDMI) ─────────────────────────────────────────

class PygameDisplay(DisplayBackend):
    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)

    def show_color(self, hex_color: str) -> None:
        h = hex_color.lstrip("#")
        color = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
        self.screen.fill(color)
        self.pg.display.flip()

    def close(self) -> None:
        self.pg.quit()


# ── Tkinter (dev fallback) ──────────────────────────────────────────

class TkinterDisplay(DisplayBackend):
    def __init__(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.configure(cursor="none")
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()

    def show_color(self, hex_color: str) -> None:
        self.canvas.configure(bg=hex_color)
        self.root.update()

    def close(self) -> None:
        self.root.destroy()


# ── Factory ─────────────────────────────────────────────────────────

def create_display(backend: str = cfg.DISPLAY_BACKEND) -> DisplayBackend:
    match backend:
        case "framebuffer":
            return FramebufferDisplay()
        case "pygame":
            return PygameDisplay()
        case "tkinter":
            return TkinterDisplay()
        case _:
            raise ValueError(f"Unknown display backend: {backend}")
