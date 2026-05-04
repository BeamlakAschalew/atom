"""
3D Atom Simulator
=================
A simple 3D atom visualizer using PyOpenGL, GLFW, and NumPy.
No GLUT dependency — spheres are drawn manually with OpenGL.

Install dependencies:
    pip install PyOpenGL glfw numpy

Run:
    python atom_simulator.py [atomic_number]

Controls:
    Left-click + drag : Rotate the scene
    Scroll wheel      : Zoom in / out
    ESC               : Quit
"""

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import math
import random
import subprocess

# ─── Element Data ─────────────────────────────────────────────────────────────

ELEMENTS = {
    1:  ("Hydrogen",      "H"),  2:  ("Helium",        "He"), 3:  ("Lithium",       "Li"),
    4:  ("Beryllium",     "Be"), 5:  ("Boron",          "B"),  6:  ("Carbon",        "C"),
    7:  ("Nitrogen",      "N"),  8:  ("Oxygen",         "O"),  9:  ("Fluorine",      "F"),
    10: ("Neon",          "Ne"), 11: ("Sodium",         "Na"), 12: ("Magnesium",     "Mg"),
    13: ("Aluminum",      "Al"), 14: ("Silicon",        "Si"), 15: ("Phosphorus",    "P"),
    16: ("Sulfur",        "S"),  17: ("Chlorine",       "Cl"), 18: ("Argon",         "Ar"),
    19: ("Potassium",     "K"),  20: ("Calcium",        "Ca"), 21: ("Scandium",      "Sc"),
    22: ("Titanium",      "Ti"), 23: ("Vanadium",       "V"),  24: ("Chromium",      "Cr"),
    25: ("Manganese",     "Mn"), 26: ("Iron",           "Fe"), 27: ("Cobalt",        "Co"),
    28: ("Nickel",        "Ni"), 29: ("Copper",         "Cu"), 30: ("Zinc",          "Zn"),
    31: ("Gallium",       "Ga"), 32: ("Germanium",      "Ge"), 33: ("Arsenic",       "As"),
    34: ("Selenium",      "Se"), 35: ("Bromine",        "Br"), 36: ("Krypton",       "Kr"),
    37: ("Rubidium",      "Rb"), 38: ("Strontium",      "Sr"), 39: ("Yttrium",       "Y"),
    40: ("Zirconium",     "Zr"), 41: ("Niobium",        "Nb"), 42: ("Molybdenum",    "Mo"),
    43: ("Technetium",    "Tc"), 44: ("Ruthenium",      "Ru"), 45: ("Rhodium",       "Rh"),
    46: ("Palladium",     "Pd"), 47: ("Silver",         "Ag"), 48: ("Cadmium",       "Cd"),
    49: ("Indium",        "In"), 50: ("Tin",            "Sn"), 51: ("Antimony",      "Sb"),
    52: ("Tellurium",     "Te"), 53: ("Iodine",         "I"),  54: ("Xenon",         "Xe"),
    55: ("Cesium",        "Cs"), 56: ("Barium",         "Ba"), 57: ("Lanthanum",     "La"),
    58: ("Cerium",        "Ce"), 59: ("Praseodymium",   "Pr"), 60: ("Neodymium",     "Nd"),
    61: ("Promethium",    "Pm"), 62: ("Samarium",       "Sm"), 63: ("Europium",      "Eu"),
    64: ("Gadolinium",    "Gd"), 65: ("Terbium",        "Tb"), 66: ("Dysprosium",    "Dy"),
    67: ("Holmium",       "Ho"), 68: ("Erbium",         "Er"), 69: ("Thulium",       "Tm"),
    70: ("Ytterbium",     "Yb"), 71: ("Lutetium",       "Lu"), 72: ("Hafnium",       "Hf"),
    73: ("Tantalum",      "Ta"), 74: ("Tungsten",       "W"),  75: ("Rhenium",       "Re"),
    76: ("Osmium",        "Os"), 77: ("Iridium",        "Ir"), 78: ("Platinum",      "Pt"),
    79: ("Gold",          "Au"), 80: ("Mercury",        "Hg"), 81: ("Thallium",      "Tl"),
    82: ("Lead",          "Pb"), 83: ("Bismuth",        "Bi"), 84: ("Polonium",      "Po"),
    85: ("Astatine",      "At"), 86: ("Radon",          "Rn"), 87: ("Francium",      "Fr"),
    88: ("Radium",        "Ra"), 89: ("Actinium",       "Ac"), 90: ("Thorium",       "Th"),
    91: ("Protactinium",  "Pa"), 92: ("Uranium",        "U"),  93: ("Neptunium",     "Np"),
    94: ("Plutonium",     "Pu"), 95: ("Americium",      "Am"), 96: ("Curium",        "Cm"),
    97: ("Berkelium",     "Bk"), 98: ("Californium",    "Cf"), 99: ("Einsteinium",   "Es"),
    100:("Fermium",       "Fm"),101:("Mendelevium",     "Md"),102:("Nobelium",       "No"),
    103:("Lawrencium",    "Lr"),104:("Rutherfordium",   "Rf"),105:("Dubnium",        "Db"),
    106:("Seaborgium",    "Sg"),107:("Bohrium",         "Bh"),108:("Hassium",        "Hs"),
    109:("Meitnerium",    "Mt"),110:("Darmstadtium",   "Ds"), 111:("Roentgenium",   "Rg"),
    112:("Copernicium",   "Cn"),113:("Nihonium",        "Nh"),114:("Flerovium",      "Fl"),
    115:("Moscovium",     "Mc"),116:("Livermorium",     "Lv"),117:("Tennessine",     "Ts"),
    118:("Oganesson",     "Og"),
}

# Distinct colours for each particle type
PROTON_COLOR   = (0.95, 0.20, 0.15)   # vivid red
NEUTRON_COLOR  = (0.65, 0.65, 0.70)   # light grey
ELECTRON_COLOR = (0.10, 0.85, 0.95)   # bright cyan

# ─── Global State ─────────────────────────────────────────────────────────────

WINDOW_WIDTH  = 900
WINDOW_HEIGHT = 700

rotation_x    = 20.0
rotation_y    = 0.0
zoom          = -18.0
mouse_pressed = False
last_mouse_x  = 0.0
last_mouse_y  = 0.0
atomic_number = 6

proton_positions  = []
neutron_positions = []
sphere_dl         = None

# ─── GUI Input (zenity / kdialog / CLI) ───────────────────────────────────────

def _try_zenity(default=6):
    """Use zenity --entry on GNOME desktops."""
    try:
        r = subprocess.run(
            ["zenity", "--entry",
             "--title=Atom Simulator",
             "--text=Enter atomic number (1-118):",
             f"--entry-text={default}"],
            capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return int(r.stdout.strip())
    except Exception:
        pass
    return None


def _try_kdialog(default=6):
    """Use kdialog --inputbox on KDE desktops."""
    try:
        r = subprocess.run(
            ["kdialog", "--inputbox",
             "Enter atomic number (1-118):",
             str(default),
             "--title", "Atom Simulator"],
            capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return int(r.stdout.strip())
    except Exception:
        pass
    return None


def _try_tkinter(default=6):
    """Custom tkinter dialog with live element name update."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.title("Atom Simulator")
        root.resizable(False, False)
        BG, FG, ACCENT = "#0d1117", "#e6edf3", "#58a6ff"
        root.configure(bg=BG)

        tk.Label(root, text="⚛  3D Atom Simulator", bg=BG, fg=ACCENT,
                 font=("Helvetica", 14, "bold")).pack(pady=(18, 4))
        tk.Label(root, text="Enter an atomic number (1 – 118):",
                 bg=BG, fg=FG, font=("Helvetica", 10)).pack(padx=14, pady=4)

        elem_var = tk.StringVar(value="Carbon  (C)")
        tk.Label(root, textvariable=elem_var, bg=BG, fg=ACCENT,
                 font=("Helvetica", 13, "bold")).pack(pady=(0, 6))

        z_var = tk.IntVar(value=default)

        def on_change(*_):
            try:
                z = max(1, min(118, int(z_var.get())))
                n, s = ELEMENTS.get(z, ("Unknown", "?"))
                elem_var.set(f"{n}  ({s})")
            except Exception:
                elem_var.set("—")

        tk.Spinbox(root, from_=1, to=118, textvariable=z_var, width=6,
                   font=("Helvetica", 13), bg="#161b22", fg=FG,
                   buttonbackground="#21262d", relief="flat",
                   command=on_change).pack(padx=14, pady=4)
        z_var.trace_add("write", on_change)

        tk.Label(root, text="Examples: 1=H  6=C  8=O  26=Fe  79=Au",
                 bg=BG, fg="#8b949e", font=("Helvetica", 8)).pack(pady=(2, 10))

        result = [default]

        def on_ok():
            try:
                result[0] = max(1, min(118, int(z_var.get())))
            except Exception:
                result[0] = default
            root.destroy()

        def on_cancel():
            root.destroy()

        bf = tk.Frame(root, bg=BG); bf.pack(pady=(0, 16))
        tk.Button(bf, text="  Visualize  ", command=on_ok,
                  bg=ACCENT, fg="#0d1117", font=("Helvetica", 10, "bold"),
                  relief="flat", cursor="hand2").pack(side="left", padx=6)
        tk.Button(bf, text="Cancel", command=on_cancel,
                  bg="#21262d", fg=FG, font=("Helvetica", 10),
                  relief="flat", cursor="hand2").pack(side="left", padx=6)

        root.bind("<Return>", lambda e: on_ok())
        root.bind("<Escape>", lambda e: on_cancel())
        root.update_idletasks()
        x = (root.winfo_screenwidth()  - root.winfo_reqwidth())  // 2
        y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
        root.geometry(f"+{x}+{y}")
        root.mainloop()
        return result[0]
    except Exception:
        return None


def ask_atomic_number_gui(default=6):
    """Try GUI dialogs in order, fall back to CLI."""
    for fn in (_try_tkinter, _try_zenity, _try_kdialog):
        val = fn(default)
        if val is not None:
            return max(1, min(118, val))
    # CLI fallback
    try:
        raw = input(f"Enter atomic number Z (1-118, default {default}): ").strip()
        return int(raw) if raw else default
    except Exception:
        return default


# ─── Atom Setup ───────────────────────────────────────────────────────────────

def setup_atom(z):
    global proton_positions, neutron_positions
    nucleus_radius = 0.6 + 0.1 * (2 * z) ** (1 / 3)
    proton_positions  = _random_sphere_points(z, nucleus_radius)
    neutron_positions = _random_sphere_points(z, nucleus_radius)
    return z, z, z   # protons, neutrons, electrons


def _random_sphere_points(count, radius):
    pts = []
    for _ in range(count):
        theta = random.uniform(0, 2 * math.pi)
        phi   = math.acos(2 * random.random() - 1)
        r     = radius * random.random() ** (1 / 3)
        pts.append((r * math.sin(phi) * math.cos(theta),
                    r * math.sin(phi) * math.sin(theta),
                    r * math.cos(phi)))
    return pts


# ─── Sphere ───────────────────────────────────────────────────────────────────

def create_sphere_dl(radius=1.0, slices=20, stacks=20):
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    for i in range(stacks):
        lat0  = math.pi * (-0.5 + i / stacks)
        z0    = radius * math.sin(lat0);  zr0 = radius * math.cos(lat0)
        lat1  = math.pi * (-0.5 + (i + 1) / stacks)
        z1    = radius * math.sin(lat1);  zr1 = radius * math.cos(lat1)
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * j / slices
            cx, cy = math.cos(lng), math.sin(lng)
            glNormal3f(cx * zr0, cy * zr0, z0);  glVertex3f(cx * zr0, cy * zr0, z0)
            glNormal3f(cx * zr1, cy * zr1, z1);  glVertex3f(cx * zr1, cy * zr1, z1)
        glEnd()
    glEndList()
    return dl


def draw_sphere(x, y, z, radius, color, dl):
    r, g, b = color
    glPushMatrix()
    glTranslatef(x, y, z)
    glScalef(radius, radius, radius)
    # Emissive tint so colour is vivid even in shadow
    glMaterialfv(GL_FRONT, GL_AMBIENT,   [r * 0.5, g * 0.5, b * 0.5, 1.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   [r * 0.9, g * 0.9, b * 0.9, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR,  [0.15, 0.15, 0.15, 1.0])
    glMaterialfv(GL_FRONT, GL_EMISSION,  [r * 0.25, g * 0.25, b * 0.25, 1.0])
    glMaterialf (GL_FRONT, GL_SHININESS, 20.0)
    glColor3f(r, g, b)
    glCallList(dl)
    glPopMatrix()


# ─── Pixel-quad text renderer (reliable on all OpenGL drivers) ────────────────
#
#  Each glyph is a 5-wide × 7-tall bitmap stored as 7 integers (row 0 = top).
#  Bit 4 (0x10) = leftmost column, bit 0 (0x01) = rightmost column.

_GLYPHS = {
    'A': [0x04,0x0A,0x11,0x1F,0x11,0x11,0x11],
    'B': [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E],
    'C': [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E],
    'D': [0x1E,0x11,0x11,0x11,0x11,0x11,0x1E],
    'E': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F],
    'F': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10],
    'G': [0x0E,0x11,0x10,0x17,0x11,0x11,0x0E],
    'H': [0x11,0x11,0x11,0x1F,0x11,0x11,0x11],
    'I': [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E],
    'J': [0x07,0x02,0x02,0x02,0x02,0x12,0x0C],
    'K': [0x11,0x12,0x14,0x18,0x14,0x12,0x11],
    'L': [0x10,0x10,0x10,0x10,0x10,0x10,0x1F],
    'M': [0x11,0x1B,0x15,0x15,0x11,0x11,0x11],
    'N': [0x11,0x19,0x1D,0x15,0x17,0x13,0x11],
    'O': [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E],
    'P': [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10],
    'Q': [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D],
    'R': [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11],
    'S': [0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E],
    'T': [0x1F,0x04,0x04,0x04,0x04,0x04,0x04],
    'U': [0x11,0x11,0x11,0x11,0x11,0x11,0x0E],
    'V': [0x11,0x11,0x11,0x11,0x0A,0x0A,0x04],
    'W': [0x11,0x11,0x15,0x15,0x15,0x1B,0x11],
    'X': [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11],
    'Y': [0x11,0x11,0x0A,0x04,0x04,0x04,0x04],
    'Z': [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F],
    '0': [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E],
    '1': [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E],
    '2': [0x0E,0x11,0x01,0x06,0x08,0x10,0x1F],
    '3': [0x0E,0x11,0x01,0x06,0x01,0x11,0x0E],
    '4': [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02],
    '5': [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E],
    '6': [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E],
    '7': [0x1F,0x01,0x02,0x04,0x08,0x08,0x08],
    '8': [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E],
    '9': [0x0E,0x11,0x11,0x0F,0x01,0x11,0x0E],
    ' ': [0x00]*7,
    ':': [0x00,0x04,0x04,0x00,0x04,0x04,0x00],
    '=': [0x00,0x1F,0x00,0x00,0x1F,0x00,0x00],
    '-': [0x00,0x00,0x00,0x1F,0x00,0x00,0x00],
    '.': [0x00,0x00,0x00,0x00,0x00,0x00,0x04],
    '/': [0x01,0x02,0x02,0x04,0x08,0x08,0x10],
    '(': [0x02,0x04,0x08,0x08,0x08,0x04,0x02],
    ')': [0x08,0x04,0x02,0x02,0x02,0x04,0x08],
}
# Lower-case → same as upper
for _c in list(_GLYPHS.keys()):
    if _c.isalpha():
        _GLYPHS[_c.lower()] = _GLYPHS[_c]

_CHAR_W = 5   # pixel columns per glyph
_CHAR_H = 7   # pixel rows per glyph
_PX     = 2   # rendered size of each pixel square (pixels on screen)
_GAP    = 1   # gap between characters


def draw_text(ox, oy, text, color=(1.0, 1.0, 1.0)):
    """
    Draw text starting at screen position (ox, oy) — bottom-left of first char.
    Uses filled quads instead of glBitmap, so it works on all GL drivers.
    Y axis: 0 = bottom of window (standard OpenGL ortho).
    """
    glColor3f(*color)
    glBegin(GL_QUADS)
    cx = ox
    for ch in text:
        rows = _GLYPHS.get(ch.upper() if ch.upper() in _GLYPHS else ch)
        if rows is None:
            cx += (_CHAR_W + _GAP) * _PX
            continue
        for row_idx, row_bits in enumerate(rows):
            # row_idx 0 = top row of glyph → highest y on screen
            py = oy + (_CHAR_H - 1 - row_idx) * _PX
            for col in range(_CHAR_W):
                # bit 4 = leftmost column
                if row_bits & (1 << (_CHAR_W - 1 - col)):
                    px = cx + col * _PX
                    glVertex2f(px,        py)
                    glVertex2f(px + _PX,  py)
                    glVertex2f(px + _PX,  py + _PX)
                    glVertex2f(px,        py + _PX)
        cx += (_CHAR_W + _GAP) * _PX
    glEnd()


def text_width(text):
    """Return pixel width of rendered text string."""
    return len(text) * (_CHAR_W + _GAP) * _PX


def draw_filled_rect(x0, y0, x1, y1, color=(0.0, 0.0, 0.0, 0.55)):
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex2f(x0, y0); glVertex2f(x1, y0)
    glVertex2f(x1, y1); glVertex2f(x0, y1)
    glEnd()


# ─── Orbit ring ───────────────────────────────────────────────────────────────

def draw_orbit_ring(orbit_r, tilt_x, tilt_z):
    """Draw a white ring in the XY plane, tilted to match _orbit_position."""
    glPushMatrix()
    # Same rotation order used by OpenGL stack that electrons are computed for:
    # final transform = Rot_Z * Rot_X (because we apply tilt_x first, tilt_z second)
    glRotatef(tilt_z, 0, 0, 1)
    glRotatef(tilt_x, 1, 0, 0)

    glDisable(GL_LIGHTING)
    glColor4f(1.0, 1.0, 1.0, 0.60)
    glLineWidth(1.3)
    STEPS = 120
    glBegin(GL_LINE_LOOP)
    for i in range(STEPS):
        a = 2.0 * math.pi * i / STEPS
        glVertex3f(orbit_r * math.cos(a), orbit_r * math.sin(a), 0.0)
    glEnd()
    glEnable(GL_LIGHTING)
    glPopMatrix()


# ─── Atom drawing ─────────────────────────────────────────────────────────────

def _build_shells(num_electrons):
    shells, remaining = [], num_electrons
    for cap in [2, 8, 8, 18, 18, 32, 32]:
        if remaining <= 0:
            break
        n = min(remaining, cap)
        shells.append(n)
        remaining -= n
    if remaining > 0:
        shells.append(remaining)
    return shells


def _orbit_position(radius, angle, tilt_x_deg, tilt_z_deg):
    """
    Return 3D world position for an electron.
    Rotation order MUST match the glRotatef order in draw_orbit_ring:
      first Rot_Z (tilt_z), then Rot_X (tilt_x).
    In matrix terms: final = Rot_Z * Rot_X * v
    """
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    z = 0.0

    # 1. Apply Rot_Z (around Z axis)
    tz  = math.radians(tilt_z_deg)
    x2  = x * math.cos(tz) - y * math.sin(tz)
    y2  = x * math.sin(tz) + y * math.cos(tz)
    x, y = x2, y2

    # 2. Apply Rot_X (around X axis)
    tx  = math.radians(tilt_x_deg)
    y3  = y * math.cos(tx) - z * math.sin(tx)
    z3  = y * math.sin(tx) + z * math.cos(tx)
    y, z = y3, z3

    return x, y, z


def draw_atom(num_protons, num_neutrons, num_electrons, t, dl):
    # Nucleus
    for pos in proton_positions:
        draw_sphere(*pos, radius=0.35, color=PROTON_COLOR, dl=dl)
    for pos in neutron_positions:
        draw_sphere(*pos, radius=0.35, color=NEUTRON_COLOR, dl=dl)

    # Electrons + orbit rings
    base_r = 3.0
    shells  = _build_shells(num_electrons)
    for shell_num, shell_count in enumerate(shells):
        orbit_r = base_r + shell_num * 2.0
        for i in range(shell_count):
            tilt_x = 30.0 * (i % 3) + 15.0 * shell_num
            tilt_z = 60.0 * i        + 20.0 * shell_num
            draw_orbit_ring(orbit_r, tilt_x, tilt_z)

            speed        = 1.5 - 0.2 * shell_num
            angle_offset = (2.0 * math.pi * i) / shell_count
            angle        = speed * t + angle_offset
            ex, ey, ez   = _orbit_position(orbit_r, angle, tilt_x, tilt_z)
            draw_sphere(ex, ey, ez, radius=0.22, color=ELECTRON_COLOR, dl=dl)


# ─── HUD ──────────────────────────────────────────────────────────────────────

def display_hud(window, num_p, num_n, num_e):
    w, h = glfw.get_window_size(window)
    z    = atomic_number
    name, sym = ELEMENTS.get(z, ("Unknown", "?"))

    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, w, 0, h, -1, 1)
    glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity()

    LINE  = (_CHAR_H + 3) * _PX      # vertical line spacing
    PAD   = 8

    # ── Top-left info panel ──────────────────────────────────────────────────
    panel_h = 5 * LINE + PAD * 2
    draw_filled_rect(0, h - panel_h, 320, h, (0.0, 0.0, 0.08, 0.72))

    ty = h - PAD - _CHAR_H * _PX   # top text baseline
    draw_text(PAD, ty,            f"{name} ({sym})", color=(0.35, 0.80, 1.00))
    draw_text(PAD, ty - LINE,     f"Z = {z}",        color=(0.80, 0.80, 0.80))
    draw_text(PAD, ty - 2 * LINE, f"Protons:   {num_p}", color=PROTON_COLOR)
    draw_text(PAD, ty - 3 * LINE, f"Neutrons:  {num_n}", color=NEUTRON_COLOR)
    draw_text(PAD, ty - 4 * LINE, f"Electrons: {num_e}", color=ELECTRON_COLOR)

    # ── Bottom-right legend ──────────────────────────────────────────────────
    legend_lines = [
        ("Proton  (red)",   PROTON_COLOR),
        ("Neutron (grey)",  NEUTRON_COLOR),
        ("Electron (cyan)", ELECTRON_COLOR),
    ]
    lw = max(text_width(l) for l, _ in legend_lines) + PAD * 2
    lh = len(legend_lines) * LINE + PAD * 2
    draw_filled_rect(w - lw, 0, w, lh, (0.0, 0.0, 0.08, 0.72))
    for idx, (lbl, col) in enumerate(legend_lines):
        ly = PAD + (len(legend_lines) - 1 - idx) * LINE
        draw_text(w - lw + PAD, ly, lbl, color=col)

    # ── Bottom-centre controls hint ──────────────────────────────────────────
    hint  = "Drag:Rotate  Scroll:Zoom  ESC:Quit"
    hw    = text_width(hint)
    hx    = (w - hw) // 2
    draw_filled_rect(hx - PAD, 0, hx + hw + PAD, LINE + PAD, (0.0, 0.0, 0.0, 0.50))
    draw_text(hx, PAD // 2, hint, color=(0.55, 0.55, 0.60))

    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


# ─── Callbacks ────────────────────────────────────────────────────────────────

def mouse_button_callback(window, button, action, mods):
    global mouse_pressed, last_mouse_x, last_mouse_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            mouse_pressed = True
            last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)
        else:
            mouse_pressed = False


def cursor_pos_callback(window, xpos, ypos):
    global rotation_x, rotation_y, last_mouse_x, last_mouse_y
    if mouse_pressed:
        rotation_y += (xpos - last_mouse_x) * 0.4
        rotation_x += (ypos - last_mouse_y) * 0.4
        last_mouse_x, last_mouse_y = xpos, ypos


def scroll_callback(window, xoffset, yoffset):
    global zoom
    zoom = max(-60.0, min(-5.0, zoom + yoffset * 1.2))


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)
    _set_projection(width, height)


# ─── OpenGL Initialisation ────────────────────────────────────────────────────

def _init_opengl():
    glClearColor(0.05, 0.05, 0.10, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)
    glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 8.0, 10.0, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0,  1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.4, 0.4, 0.4,  1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.3, 0.3, 0.3,  1.0])


def _set_projection(width, height):
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45.0, width / max(height, 1), 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global atomic_number, sphere_dl

    if len(sys.argv) > 1:
        try:
            atomic_number = int(sys.argv[1])
        except ValueError:
            atomic_number = 6
    else:
        atomic_number = ask_atomic_number_gui(default=6)

    atomic_number = max(1, min(118, atomic_number))
    name, sym = ELEMENTS.get(atomic_number, ("Unknown", "?"))

    print(f"\n  Atom: {name} ({sym})  Z={atomic_number}")
    print(f"  Protons / Neutrons / Electrons: {atomic_number}\n")

    if not glfw.init():
        print("ERROR: Could not initialise GLFW."); sys.exit(1)

    glfw.window_hint(glfw.SAMPLES, 4)
    title  = f"3D Atom Simulator — {name} ({sym})  Z={atomic_number}"
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, title, None, None)
    if not window:
        glfw.terminate(); print("ERROR: Could not create window."); sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_mouse_button_callback(window,    mouse_button_callback)
    glfw.set_cursor_pos_callback(window,      cursor_pos_callback)
    glfw.set_scroll_callback(window,          scroll_callback)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    _init_opengl()
    w, h = glfw.get_framebuffer_size(window)
    glViewport(0, 0, w, h)
    _set_projection(w, h)

    sphere_dl       = create_sphere_dl()
    num_p, num_n, num_e = setup_atom(atomic_number)

    print("  Controls:  Left-drag=Rotate  Scroll=Zoom  ESC=Quit\n")

    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        t = glfw.get_time()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        draw_atom(num_p, num_n, num_e, t, sphere_dl)
        display_hud(window, num_p, num_n, num_e)

        glfw.swap_buffers(window)

    glfw.terminate()
    print("Goodbye!")


if __name__ == "__main__":
    main()
