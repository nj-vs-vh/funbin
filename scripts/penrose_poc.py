"""
Penrose tiling (P2/P3) generation prototype, modeled after:
- https://preshing.com/20110831/penrose-tiling-explained/
Implementation is based on:
- https://github.com/samm00/penrose/tree/main
"""

import cmath
import math
import random
import re
from typing import Literal

import cairo

divisions = 5

scale = 2

r1, r2 = 1080, 1080

c1, c2, c3 = "red", "blue", "grey"

filename = "test.png"

colors = []
for c in c1, c2, c3:
    try:
        colors.append(
            {
                "random": [random.randint(0, 256) / 256, random.randint(0, 256) / 256, random.randint(0, 256) / 256],
                "red": [0.8, 0.3, 0.3],
                "orange": [0.9, 0.6, 0.3],
                "yellow": [0.6, 0.9, 0.3],
                "green": [0.3, 0.9, 0.6],
                "blue": [0.3, 0.6, 0.9],
                "purple": [0.8, 0.3, 0.6],
                "grey": [0.2, 0.2, 0.2],
                "brown": [0.6, 0.3, 0.1],
                "black": [0, 0, 0],
                "white": [1, 1, 1],
            }[c]
        )
    except KeyError:
        shape = [int(x, 16) / 256 for x in re.compile("[0-9a-fA-F]{2}").findall(c)]

        if len(shape) != 3:
            print("\nColor not supported")
            raise SystemExit(0)

        colors.append(shape)

phi = (5**0.5 + 1) / 2  # Golden ratio

base = 5
Penrose2Tiling = Literal["P2", "P3"]
kind: Penrose2Tiling = "P2"

# Canvas setup
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, r1, r2)
ctx = cairo.Context(surface)
ctx.scale(max(r1, r2) / scale, max(r1, r2) / scale)
ctx.translate(0.5 * scale, 0.5 * scale)  # Center the drawing

# Create first layer of triangles
TriangleKind = Literal[0, 1]
Triangle = tuple[TriangleKind, complex, complex, complex]

triangles: list[Triangle] = []
for i in range(base * 2):
    v2 = cmath.rect(1, (2 * i - 1) * math.pi / (base * 2))
    v3 = cmath.rect(1, (2 * i + 1) * math.pi / (base * 2))
    if i % 2 == 0:
        v2, v3 = v3, v2  # Mirror every other triangle

    if kind == "P2":
        triangles.append((0, v2, 0j, v3))
    else:
        triangles.append((0, 0, v2, v3))


for i in range(divisions):
    new_triangles: list[Triangle] = []
    for shape, v1, v2, v3 in triangles:
        if kind == "P2":
            if shape == 0:
                # Subdivide red (sharp isosceles) (half kite) triangle
                p1 = v1 + (v2 - v1) / phi
                p2 = v2 + (v3 - v2) / phi
                new_triangles.extend(((1, p2, p1, v2), (0, p1, v1, p2), (0, v3, v1, p2)))
            else:
                # Subdivide blue (fat isosceles) (half dart) triangle
                p3 = v3 + (v1 - v3) / phi
                new_triangles.extend(((1, v2, p3, v1), (0, p3, v3, v2)))
        else:
            if shape == 0:
                # Divide thin rhombus
                p1 = v1 + (v2 - v1) / phi
                new_triangles.extend(((1, p1, v3, v1), (0, v3, p1, v2)))
            else:
                # Divide thicc rhombus
                p2 = v2 + (v1 - v2) / phi
                p3 = v2 + (v3 - v2) / phi
                new_triangles.extend(((1, p3, v3, v1), (1, p2, p3, v2), (0, p3, p2, v1)))

    triangles = new_triangles

# Draw kind 0
for shape, v1, v2, v3 in triangles:
    if shape == 0:
        ctx.move_to(v1.real, v1.imag)
        ctx.line_to(v2.real, v2.imag)
        ctx.line_to(v3.real, v3.imag)
        ctx.close_path()
ctx.set_source_rgb(colors[0][0], colors[0][1], colors[0][2])
ctx.fill()

# # Draw kind 1
for shape, v1, v2, v3 in triangles:
    if shape == 1:
        ctx.move_to(v1.real, v1.imag)
        ctx.line_to(v2.real, v2.imag)
        ctx.line_to(v3.real, v3.imag)
        ctx.close_path()
ctx.set_source_rgb(colors[1][0], colors[1][1], colors[1][2])
ctx.fill()

# Determine line width
shape, v1, v2, v3 = triangles[0]
ctx.set_line_width(abs(v2 - v1) / (base * 2))
ctx.set_line_join(cairo.LINE_JOIN_ROUND)

# Draw outlines
for shape, v1, v2, v3 in triangles:
    ctx.move_to(v2.real, v2.imag)
    ctx.line_to(v1.real, v1.imag)
    ctx.line_to(v3.real, v3.imag)
    # ctx.close_path()

ctx.set_source_rgb(colors[2][0], colors[2][1], colors[2][2])
ctx.set_line_width(divisions**-3) if divisions > 3 else ctx.set_line_width(divisions**-5)
ctx.stroke()

surface.write_to_png(filename)
