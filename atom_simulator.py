"""
3D Atom Simulator
=================
A simple 3D atom visualizer using PyOpenGL, GLFW, and NumPy.
No GLUT dependency — spheres are drawn manually with OpenGL.

Install dependencies:
    pip install PyOpenGL glfw numpy

Run:
    python atom_simulator.py

Controls:
    Left-click + drag : Rotate the scene
    Scroll wheel      : Zoom in / out
    ESC               : Quit
"""

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import math
import random
import ctypes

# ─── Configuration ───────────────────────────────────────────────────────────

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
WINDOW_TITLE = "3D Atom Simulator"

# ─── Global State ────────────────────────────────────────────────────────────

rotation_x = 20.0
rotation_y = 0.0
zoom = -18.0
mouse_pressed = False
last_mouse_x = 0.0
last_mouse_y = 0.0
mouse_x = 0.0
mouse_y = 0.0
atomic_number = 6

proton_positions = []
neutron_positions = []

# Sphere display list ID (created once)
sphere_dl = None


# ─── GUI Input Dialog ────────────────────────────────────────────────────────

def ask_atomic_number_gui():
    """Open a small tkinter dialog to ask for atomic number."""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Style the dialog
        result = simpledialog.askinteger(
            "Atom Simulator",
            "Enter atomic number (Z):\n\n"
            "Examples:\n"
            "  1 = Hydrogen\n"
            "  6 = Carbon\n"
            "  7 = Nitrogen\n"
            "  8 = Oxygen\n"
            "  26 = Iron\n"
            "  79 = Gold",
            initialvalue=6,
            minvalue=1,
            maxvalue=118,
            parent=root,
        )
        root.destroy()
        return result if result is not None else 6
    except Exception:
        # Fallback to console if tkinter unavailable
        try:
            val = input("Enter atomic number Z (default 6): ").strip()
            return int(val) if val else 6
        except (ValueError, EOFError):
            return 6


# ─── Atom Setup ──────────────────────────────────────────────────────────────

def setup_atom(z):
    """Calculate particle counts and generate nucleus positions."""
    global proton_positions, neutron_positions

    protons = z
    neutrons = z  # simplified assumption
    electrons = z

    nucleus_radius = 0.6 + 0.1 * (protons + neutrons) ** (1 / 3)

    proton_positions = _random_sphere_points(protons, nucleus_radius)
    neutron_positions = _random_sphere_points(neutrons, nucleus_radius)

    return protons, neutrons, electrons


def _random_sphere_points(count, radius):
    """Generate `count` random 3D points inside a sphere of given radius."""
    points = []
    for _ in range(count):
        u = random.random()
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(2 * random.random() - 1)
        r = radius * u ** (1 / 3)
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        points.append((x, y, z))
    return points


# ─── Custom Sphere (no GLUT needed) ─────────────────────────────────────────

def create_sphere_display_list(radius=1.0, slices=16, stacks=16):
    """Create an OpenGL display list for a unit sphere."""
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + i / stacks)
        z0 = radius * math.sin(lat0)
        zr0 = radius * math.cos(lat0)

        lat1 = math.pi * (-0.5 + (i + 1) / stacks)
        z1 = radius * math.sin(lat1)
        zr1 = radius * math.cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * j / slices
            x = math.cos(lng)
            y = math.sin(lng)

            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0, y * zr0, z0)

            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1, y * zr1, z1)
        glEnd()
    glEndList()
    return dl


def draw_sphere(x, y, z, radius, color, dl):
    """Draw a sphere at (x, y, z) using the precompiled display list."""
    glPushMatrix()
    glTranslatef(x, y, z)
    glScalef(radius, radius, radius)
    glColor3f(*color)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [*color, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 30.0)
    glCallList(dl)
    glPopMatrix()


# ─── Simple Bitmap Text (no GLUT) ───────────────────────────────────────────

# Tiny 5x7 bitmap font for A-Z, a-z, 0-9, and a few symbols
_FONT_DATA = {}


def _init_font():
    """Pre-build a tiny bitmap font for on-screen text."""
    # We'll use OpenGL's own raster capabilities with raw bitmaps
    # Each char is 6 wide x 10 tall (padded 5x7 glyphs)
    glyphs = {
        'A': [0x04, 0x0A, 0x11, 0x11, 0x1F, 0x11, 0x11],
        'B': [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C': [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D': [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E': [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F': [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G': [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
        'H': [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I': [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J': [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K': [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L': [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M': [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N': [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O': [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P': [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q': [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R': [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S': [0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E],
        'T': [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U': [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V': [0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04],
        'W': [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X': [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y': [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z': [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '0': [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1': [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2': [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
        '3': [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
        '4': [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5': [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6': [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7': [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8': [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9': [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        ' ': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        '=': [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        ':': [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00],
        '|': [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        '-': [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '.': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
        '(': [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')': [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
    }
    # Add lowercase as copies of uppercase
    for ch in list(glyphs.keys()):
        if ch.isalpha():
            glyphs[ch.lower()] = glyphs[ch]

    for ch, rows in glyphs.items():
        # Convert to 8-bit wide bitmap (OpenGL expects byte-aligned rows)
        bitmap = bytes([(row << 3) & 0xFF for row in reversed(rows)])
        _FONT_DATA[ch] = bitmap


def draw_text(x, y, text, color=(1.0, 1.0, 1.0)):
    """Draw text at screen position (x, y) using custom bitmap font."""
    glColor3f(*color)
    glRasterPos2f(x, y)
    for ch in text:
        bmp = _FONT_DATA.get(ch)
        if bmp:
            glBitmap(8, 7, 0, 0, 7, 0, bmp)
        else:
            glBitmap(0, 0, 0, 0, 7, 0, None)  # skip unknown chars


# ─── Drawing Helpers ─────────────────────────────────────────────────────────

def draw_orbit_ring(radius, tilt_x, tilt_z):
    """Draw a faint circle representing an electron orbit."""
    glPushMatrix()
    glRotatef(tilt_x, 1, 0, 0)
    glRotatef(tilt_z, 0, 0, 1)

    glDisable(GL_LIGHTING)
    glColor4f(0.4, 0.4, 0.4, 0.35)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    for i in range(100):
        angle = 2.0 * math.pi * i / 100
        glVertex3f(radius * math.cos(angle), radius * math.sin(angle), 0.0)
    glEnd()
    glEnable(GL_LIGHTING)

    glPopMatrix()


# ─── Core Drawing ────────────────────────────────────────────────────────────

def draw_atom(num_protons, num_neutrons, num_electrons, time, dl):
    """Draw the complete atom: nucleus + electron orbits + electrons."""

    # --- Nucleus ---
    for pos in proton_positions:
        draw_sphere(*pos, radius=0.35, color=(0.9, 0.15, 0.15), dl=dl)

    for pos in neutron_positions:
        draw_sphere(*pos, radius=0.35, color=(0.15, 0.25, 0.9), dl=dl)

    # --- Electrons ---
    base_orbit_radius = 3.0
    shells = _build_shells(num_electrons)

    for shell_num, shell_count in enumerate(shells):
        orbit_r = base_orbit_radius + shell_num * 2.0

        for i in range(shell_count):
            angle_offset = (2.0 * math.pi * i) / shell_count
            tilt_x = 30.0 * (i % 3) + 15.0 * shell_num
            tilt_z = 60.0 * i + 20.0 * shell_num

            draw_orbit_ring(orbit_r, tilt_x, tilt_z)

            speed = 1.5 - 0.2 * shell_num
            angle = speed * time + angle_offset
            ex, ey, ez = _orbit_position(orbit_r, angle, tilt_x, tilt_z)

            draw_sphere(ex, ey, ez, radius=0.22, color=(0.1, 0.85, 0.2), dl=dl)


def _build_shells(num_electrons):
    """Distribute electrons into shells: 2, 8, 8, ... (simplified)."""
    shells = []
    remaining = num_electrons
    capacity = [2, 8, 8, 18, 18, 32, 32]
    for cap in capacity:
        if remaining <= 0:
            break
        n = min(remaining, cap)
        shells.append(n)
        remaining -= n
    if remaining > 0:
        shells.append(remaining)
    return shells


def _orbit_position(radius, angle, tilt_x_deg, tilt_z_deg):
    """Compute 3D position on a tilted circular orbit."""
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    z = 0.0

    tx = math.radians(tilt_x_deg)
    y2 = y * math.cos(tx) - z * math.sin(tx)
    z2 = y * math.sin(tx) + z * math.cos(tx)
    y, z = y2, z2

    tz = math.radians(tilt_z_deg)
    x2 = x * math.cos(tz) - y * math.sin(tz)
    y2 = x * math.sin(tz) + y * math.cos(tz)
    x, y = x2, y2

    return x, y, z


# ─── Hover Info HUD ─────────────────────────────────────────────────────────

def display_hover_info(window):
    """Draw on-screen hover info and atom stats."""
    global mouse_x, mouse_y

    w, h = glfw.get_window_size(window)
    cx, cy = w / 2.0, h / 2.0
    dist = math.sqrt((mouse_x - cx) ** 2 + (mouse_y - cy) ** 2)

    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, w, 0, h, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    if dist < 100:
        label = "Nucleus"
        color = (1.0, 0.4, 0.4)
    else:
        label = "Electron Region"
        color = (0.3, 1.0, 0.4)

    # Background bar
    text_x = 10.0
    text_y = h - 28.0
    glColor4f(0.0, 0.0, 0.0, 0.55)
    glBegin(GL_QUADS)
    glVertex2f(0, text_y - 20)
    glVertex2f(260, text_y - 20)
    glVertex2f(260, text_y + 14)
    glVertex2f(0, text_y + 14)
    glEnd()

    draw_text(text_x, text_y, label, color)

    z = atomic_number
    info = f"Z={z}  P:{z}  N:{z}  E:{z}"
    draw_text(text_x, text_y - 14, info, (0.75, 0.75, 0.75))

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


# ─── Callbacks ───────────────────────────────────────────────────────────────

def mouse_button_callback(window, button, action, mods):
    global mouse_pressed, last_mouse_x, last_mouse_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            mouse_pressed = True
            last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            mouse_pressed = False


def cursor_pos_callback(window, xpos, ypos):
    global rotation_x, rotation_y, last_mouse_x, last_mouse_y, mouse_x, mouse_y
    mouse_x = xpos
    mouse_y = ypos
    if mouse_pressed:
        dx = xpos - last_mouse_x
        dy = ypos - last_mouse_y
        rotation_y += dx * 0.4
        rotation_x += dy * 0.4
        last_mouse_x = xpos
        last_mouse_y = ypos


def scroll_callback(window, xoffset, yoffset):
    global zoom
    zoom += yoffset * 1.2
    zoom = max(-60.0, min(-5.0, zoom))


# ─── OpenGL Init ─────────────────────────────────────────────────────────────

def init_opengl():
    """Set up OpenGL state: lights, depth, blending."""
    glClearColor(0.05, 0.05, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)

    glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 10.0, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])


def set_projection(width, height):
    """Update the perspective projection matrix."""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = width / max(height, 1)
    gluPerspective(45.0, aspect, 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)
    set_projection(width, height)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    global atomic_number, sphere_dl

    # --- GUI input for atomic number ---
    if len(sys.argv) > 1:
        try:
            atomic_number = int(sys.argv[1])
        except ValueError:
            atomic_number = 6
    else:
        atomic_number = ask_atomic_number_gui()

    atomic_number = max(1, min(atomic_number, 118))

    print(f"\n  Atom: Z = {atomic_number}")
    print(f"  Protons:   {atomic_number}")
    print(f"  Neutrons:  {atomic_number}")
    print(f"  Electrons: {atomic_number}\n")

    # ── Init GLFW ──
    if not glfw.init():
        print("ERROR: Could not initialize GLFW.")
        sys.exit(1)

    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, None, None)
    if not window:
        glfw.terminate()
        print("ERROR: Could not create GLFW window.")
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Register callbacks
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    # OpenGL setup
    init_opengl()
    _init_font()
    w, h = glfw.get_framebuffer_size(window)
    glViewport(0, 0, w, h)
    set_projection(w, h)

    # Create reusable sphere display list
    sphere_dl = create_sphere_display_list(radius=1.0, slices=20, stacks=20)

    # Build atom data
    num_p, num_n, num_e = setup_atom(atomic_number)

    print("  Controls:")
    print("    Left-click + drag  ->  Rotate")
    print("    Scroll             ->  Zoom")
    print("    ESC                ->  Quit\n")

    # ── Main Loop ──
    while not glfw.window_should_close(window):
        glfw.poll_events()

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        t = glfw.get_time()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        # Draw
        draw_atom(num_p, num_n, num_e, t, sphere_dl)

        # HUD
        display_hover_info(window)

        glfw.swap_buffers(window)

    glfw.terminate()
    print("Goodbye!")


if __name__ == "__main__":
    main()
