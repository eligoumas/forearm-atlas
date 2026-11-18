# Forearm Atlas: builds the 3D forearm and hand in Blender and exports models/forearm.glb (next to index.html).
#
# Run it from Blender's Python console or the MCP add-on, with p set to this file's full path:
#   p = r"C:\Users\elgou\OneDrive\Documents\Cool Project\forearm-atlas\blender\build_forearm.py"
#   exec(compile(open(p, encoding='utf-8').read(), p, 'exec'), {'__file__': p, '__name__': '__main__'})
#
# Everything lands in a "ForearmAtlas" collection; nothing else in the open file is touched.
# Right arm, lying palm up. Blender axes: +X ulnar (little-finger side), +Y from elbow to fingertips,
# +Z palmar. Units are metres. glTF export turns this into three.js axes (x, z, -y).
#
# Object names are what the app reads:
#   Skin, B_<bone>, M_<muscle key>, T_<tendon key>, N_/A_/V_<nerve/artery/vein key>, X_<other>, Z_nail
#   A suffix after "__" only keeps names unique (M_ADP__2 is the second piece of ADP).
# Muscle keys match the app's activity keys (FDS_i, ED_p, DI1 ...), so the app can light them up.

import bpy, bmesh, math, json, os, random
import numpy as np
from mathutils import Vector as V, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', 'models', 'forearm.glb'))
DO_EXPORT = globals().get('DO_EXPORT', True)

# ---------------------------------------------------------------- helpers
def lerp(a, b, t): return a + (b - a) * t

def interp(table, y):
    if y <= table[0][0]: return table[0][1]
    for (y0, v0), (y1, v1) in zip(table, table[1:]):
        if y <= y1: return v0 + (v1 - v0) * (y - y0) / (y1 - y0)
    return table[-1][1]

# Skin half-width along the forearm, and depth as a fraction of width. These are the same
# "skin half-width = 1" units the app's old cross-section plates used, so their muscle
# positions carry straight over.
HW = [(-0.08, 0.042), (-0.03, 0.044), (0.0, 0.045), (0.04, 0.042), (0.09, 0.0375), (0.13, 0.034),
      (0.18, 0.030), (0.23, 0.027), (0.255, 0.0266), (0.275, 0.0285)]
DR = [(-0.08, 0.90), (0.0, 0.86), (0.13, 0.80), (0.20, 0.72), (0.23, 0.66), (0.255, 0.62), (0.28, 0.58)]
def hw(y): return interp(HW, y)
def dr(y): return interp(DR, y)
def SP(y, sx, sz): h = hw(y); return V((sx * h, y, sz * h))

def cr1(a, b, c, d, t):
    t2 = t * t; t3 = t2 * t
    return 0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3)

def cr_sample(ctrl, step=0.004):
    """Catmull-Rom through control tuples (pos, rx, rz); returns dense samples."""
    P = [(V(c[0]), float(c[1]), float(c[2])) for c in ctrl]
    if len(P) == 1: return P
    ext = lambda a, b: (a[0] * 2 - b[0], a[1] * 2 - b[1], a[2] * 2 - b[2])
    P = [ext(P[0], P[1])] + P + [ext(P[-1], P[-2])]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        m = max(2, int(math.ceil((p2[0] - p1[0]).length / step)))
        for s in range(m):
            t = s / m
            out.append((cr1(p0[0], p1[0], p2[0], p3[0], t),
                        max(0.0002, cr1(p0[1], p1[1], p2[1], p3[1], t)),
                        max(0.0002, cr1(p0[2], p1[2], p2[2], p3[2], t))))
    out.append(P[-2])
    return out

def loft_data(samples, segs=12, cap0=0.0, cap1=0.0, sq=2.0, ncap=4, uscale=25.0):
    """Tube through samples (pos, rx, rz) with elliptical (or squarer, sq>2) sections and rounded caps.
    rx runs sideways (≈ X for a tube along Y), rz along the palmar direction."""
    n = len(samples)
    pos = [s[0] for s in samples]
    T = []
    for i in range(n):
        t = pos[min(n - 1, i + 1)] - pos[max(0, i - 1)]
        T.append(t.normalized() if t.length > 1e-9 else V((0, 1, 0)))
    up = V((0, 0, 1)); side = V((1, 0, 0))
    N0 = up - T[0] * up.dot(T[0])
    if N0.length < 0.2: N0 = side - T[0] * side.dot(T[0])
    N = [N0.normalized()]
    for i in range(1, n):  # parallel transport keeps the section from twisting
        v = T[i - 1].rotation_difference(T[i]) @ N[-1]
        N.append((v - T[i] * v.dot(T[i])).normalized())
    arc = [0.0]
    for i in range(1, n): arc.append(arc[-1] + (pos[i] - pos[i - 1]).length)
    rings = [(pos[i], T[i], N[i], samples[i][1], samples[i][2], arc[i] * uscale) for i in range(n)]
    pre, post = [], []
    for k in range(ncap - 1, 0, -1):
        f = (k / ncap) * math.pi / 2
        if cap0 > 0:
            p, t, nn, rx, rz, u = rings[0]
            pre.append((p - t * cap0 * math.sin(f), t, nn, rx * math.cos(f), rz * math.cos(f), u - cap0 * math.sin(f) * uscale))
    for k in range(1, ncap):
        f = (k / ncap) * math.pi / 2
        if cap1 > 0:
            p, t, nn, rx, rz, u = rings[-1]
            post.append((p + t * cap1 * math.sin(f), t, nn, rx * math.cos(f), rz * math.cos(f), u + cap1 * math.sin(f) * uscale))
    rings = pre + rings + post
    verts, faces, fuv = [], [], []
    for (p, t, nn, rx, rz, u) in rings:
        b = t.cross(nn)
        for j in range(segs):
            a = 2 * math.pi * j / segs
            ca, sa = math.cos(a), math.sin(a)
            k = (abs(ca) ** sq + abs(sa) ** sq) ** (-1.0 / sq) if sq != 2.0 else 1.0
            verts.append(p + b * (ca * k * rx) + nn * (sa * k * rz))
    R = len(rings)
    for i in range(R - 1):
        for j in range(segs):
            j1 = (j + 1) % segs
            faces.append((i * segs + j, i * segs + j1, (i + 1) * segs + j1, (i + 1) * segs + j))
            u0, u1 = rings[i][5], rings[i + 1][5]
            fuv.append([(u0, j / segs), (u0, (j + 1) / segs), (u1, (j + 1) / segs), (u1, j / segs)])
    p0 = len(verts); verts.append((pos[0] - T[0] * cap0) if cap0 > 0 else pos[0].copy())
    p1 = len(verts); verts.append((pos[-1] + T[-1] * cap1) if cap1 > 0 else pos[-1].copy())
    last = (R - 1) * segs
    for j in range(segs):
        j1 = (j + 1) % segs
        faces.append((p0, j1, j)); fuv.append([(rings[0][5], 0.5), (rings[0][5], (j + 1) / segs), (rings[0][5], j / segs)])
        faces.append((p1, last + j, last + j1)); fuv.append([(rings[-1][5], 0.5), (rings[-1][5], j / segs), (rings[-1][5], (j + 1) / segs)])
    return verts, faces, fuv

def sphere_data(c, r, rot_z=0.0, seg=20, rings=12):
    rx, ry, rz = r
    M = Matrix.Translation(V(c)) @ Matrix.Rotation(rot_z, 4, 'Z') @ Matrix.Diagonal((rx, ry, rz, 1.0))
    verts, faces, fuv = [], [], []
    verts.append(M @ V((0, 0, -1)))
    for i in range(1, rings):
        th = math.pi * i / rings
        for j in range(seg):
            ph = 2 * math.pi * j / seg
            verts.append(M @ V((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), -math.cos(th))))
    verts.append(M @ V((0, 0, 1)))
    top = len(verts) - 1
    idx = lambda i, j: 1 + (i - 1) * seg + (j % seg)
    for j in range(seg):
        faces.append((0, idx(1, j + 1), idx(1, j))); fuv.append([(0, 0), (1, 0), (0, 0)])
        faces.append((top, idx(rings - 1, j), idx(rings - 1, j + 1))); fuv.append([(0, 1), (0, 1), (1, 1)])
    for i in range(1, rings - 1):
        for j in range(seg):
            faces.append((idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)))
            fuv.append([(j / seg, i / rings), ((j + 1) / seg, i / rings), ((j + 1) / seg, (i + 1) / rings), (j / seg, (i + 1) / rings)])
    return verts, faces, fuv

# ---------------------------------------------------------------- scene, materials, naming
COLL = 'ForearmAtlas'
def collection():
    c = bpy.data.collections.get(COLL)
    if c is None:
        c = bpy.data.collections.new(COLL); bpy.context.scene.collection.children.link(c)
    for o in list(c.objects):
        d = o.data; bpy.data.objects.remove(o, do_unlink=True)
        if d is not None and d.users == 0:
            (bpy.data.meshes if isinstance(d, bpy.types.Mesh) else bpy.data.armatures).remove(d)
    return c
C = collection()

COLORS = {'skin': ((0.78, 0.55, 0.46), 0.55), 'bone': ((0.90, 0.84, 0.72), 0.65), 'muscle': ((0.50, 0.14, 0.13), 0.5),
          'tendon': ((0.88, 0.90, 0.90), 0.35), 'nerve': ((0.88, 0.68, 0.22), 0.45), 'artery': ((0.78, 0.17, 0.19), 0.4),
          'vein': ((0.27, 0.42, 0.72), 0.4), 'fascia': ((0.82, 0.85, 0.86), 0.5), 'nail': ((0.90, 0.74, 0.69), 0.3)}
def material(cat):
    name = 'FA_' + cat
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        try: m.use_nodes = True
        except Exception: pass
        col, rough = COLORS[cat]
        bsdf = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        bsdf.inputs['Base Color'].default_value = (*col, 1.0)
        bsdf.inputs['Roughness'].default_value = rough
        m.diffuse_color = (*col, 1.0)
    return m

USED = {}
def uname(base):
    n = USED.get(base, 0) + 1; USED[base] = n
    return base if n == 1 else f'{base}__{n}'

CAT = {}  # object name -> category
def make_obj(name, parts, cat):
    bm = bmesh.new(); uvl = bm.loops.layers.uv.new('UV')
    for verts, faces, fuv in parts:
        bv = [bm.verts.new(v) for v in verts]
        for f, uvs in zip(faces, fuv):
            try: face = bm.faces.new([bv[i] for i in f])
            except ValueError: continue
            face.smooth = True
            for loop, uv in zip(face.loops, uvs): loop[uvl].uv = uv
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    me.materials.append(material(cat))
    ob = bpy.data.objects.new(name, me); C.objects.link(ob)
    CAT[name] = cat
    return ob

# ---------------------------------------------------------------- skeleton landmarks
Zv = V((0, 0, 1))
W = V((0.0, 0.255, -0.002))  # radiocarpal joint
FK = 'imrp'
FNAME = {'i': 'index', 'm': 'middle', 'r': 'ring', 'p': 'little'}
BASE = {'i': V((-0.013, 0.288, -0.006)), 'm': V((-0.002, 0.290, -0.007)), 'r': V((0.009, 0.293, -0.006)), 'p': V((0.019, 0.290, -0.004))}
MCP = {'i': V((-0.027, 0.355, -0.004)), 'm': V((-0.008, 0.360, -0.006)), 'r': V((0.010, 0.355, -0.005)), 'p': V((0.026, 0.343, -0.003))}
FDIR = {'i': V((-0.16, 1, 0)).normalized(), 'm': V((-0.03, 1, 0)).normalized(), 'r': V((0.08, 1, 0)).normalized(), 'p': V((0.20, 1, 0)).normalized()}
FLEN = {'i': (0.040, 0.023, 0.019), 'm': (0.045, 0.027, 0.020), 'r': (0.042, 0.026, 0.020), 'p': (0.032, 0.019, 0.018)}
FRAD = {'i': (0.0098, 0.0090, 0.0080, 0.0076), 'm': (0.0100, 0.0092, 0.0082, 0.0078),
        'r': (0.0094, 0.0087, 0.0078, 0.0074), 'p': (0.0084, 0.0078, 0.0071, 0.0068)}  # skin radius at MCP, PIP, DIP, tip
J = {}
for k in FK:
    pts = [MCP[k].copy()]
    for L in FLEN[k]: pts.append(pts[-1] + FDIR[k] * L)
    J[k] = pts  # MCP, PIP, DIP, tip

def perp(v, d): v = v - d * v.dot(d); return v.normalized()
T0 = V((-0.024, 0.282, 0.004))  # thumb CMC (trapezium saddle)
TD = [V((-0.55, 0.78, 0.30)).normalized(), V((-0.40, 0.88, 0.25)).normalized(), V((-0.32, 0.92, 0.22)).normalized()]
TL = (0.046, 0.032, 0.027)
TJ = [T0.copy()]
for d, L in zip(TD, TL): TJ.append(TJ[-1] + d * L)
TP = [perp(V((0.62, -0.05, 0.78)), d) for d in TD]  # direction the thumb pad faces, per segment

def fpt(k, seg, u, off=0.0, dx=0.0):
    """Point on finger k. seg 0 = metacarpal (base->MCP), 1..3 = phalanges. off = palmar offset (m)."""
    p = lerp(BASE[k], MCP[k], u) if seg == 0 else lerp(J[k][seg - 1], J[k][seg], u)
    return p + Zv * off + V((dx, 0, 0))
def tpt(seg, u, off=0.0):
    """Point on the thumb. seg 0 = metacarpal, 1 = proximal phalanx, 2 = distal. off = toward the pad (m)."""
    return lerp(TJ[seg], TJ[seg + 1], u) + TP[seg] * off

# Palm shape (half-width, half-thickness, centre height along y), shared by the skin and the surface veins.
PALM_RX = [(0.250, hw(0.25)), (0.265, 0.0270), (0.29, 0.030), (0.33, 0.037), (0.352, 0.0385)]
PALM_RZ = [(0.250, hw(0.25) * dr(0.25)), (0.265, 0.0150), (0.29, 0.0132), (0.33, 0.0120), (0.352, 0.0115)]
PALM_ZC = [(0.250, 0.0), (0.265, -0.0008), (0.29, -0.0020), (0.33, -0.0020), (0.352, -0.0012)]

def fsurf(y, a, inset=0.0018):
    """Point just under the forearm skin at angle a (0 = little-finger side, pi/2 = palm side, pi = thumb side, -pi/2 = back)."""
    h = hw(y); d = h * dr(y)
    return V((math.cos(a) * (h - inset), y, math.sin(a) * (d - inset)))
def dorsum(x, y, inset=0.0016):
    """Point just under the skin on the back of the hand."""
    rx, rz, zc = interp(PALM_RX, y), interp(PALM_RZ, y), interp(PALM_ZC, y)
    t = min(0.97, abs(x) / rx)
    return V((x, y, zc - rz * math.sqrt(1 - t * t) + inset))
def fdorsal(k, seg, u, dx=0.0, inset=0.0014):
    """Point just under the skin on the back of finger k."""
    r = FRAD[k][max(0, seg - 1)]
    return fpt(k, seg, u, 0.0012 - 0.9 * r * math.sqrt(max(0.0, 1 - (dx / r) ** 2)) + inset, dx)

# ---------------------------------------------------------------- skin
def build_skin():
    parts = []
    # Forearm tube; the palm tube starts with the same section where this one stops, so the wrist has no step.
    ys = [-0.065 + 0.005 * i for i in range(0, 65)]
    parts.append(loft_data([(V((0, y, 0)), hw(y), hw(y) * dr(y)) for y in ys], segs=28, cap0=0.004, cap1=0.003))
    pys = [0.250 + 0.005 * i for i in range(21)]
    prx, prz, pzc = PALM_RX, PALM_RZ, PALM_ZC
    parts.append(loft_data([(V((interp([(0.25, 0), (0.35, -0.0005)], y), y, interp(pzc, y))), interp(prx, y), interp(prz, y)) for y in pys],
                           segs=28, cap0=0.006, cap1=0.008, sq=2.0))
    # distal palm pad under the knuckles
    parts.append(loft_data(cr_sample([(fpt('i', 1, 0, 0.0035) - FDIR['i'] * 0.008, 0.007, 0.0085), (fpt('m', 1, 0, 0.004) - FDIR['m'] * 0.009, 0.008, 0.009),
                                      (fpt('r', 1, 0, 0.004) - FDIR['r'] * 0.009, 0.008, 0.009), (fpt('p', 1, 0, 0.0035) - FDIR['p'] * 0.008, 0.007, 0.008)], 0.003),
                           segs=18, cap0=0.006, cap1=0.006))
    for k in FK:
        d = FDIR[k]; r = FRAD[k]; j = J[k]; o = Zv * 0.0012
        ctrl = [(j[0] - d * 0.012 + o, r[0] * 1.05, r[0] * 0.95), (j[0] + o, r[0], r[0] * 0.9),
                (lerp(j[0], j[1], 0.5) + o + Zv * 0.0006, (r[0] + r[1]) / 2 * 0.97, (r[0] + r[1]) / 2 * 0.9),
                (j[1] + o, r[1], r[1] * 0.9), (lerp(j[1], j[2], 0.5) + o + Zv * 0.0006, (r[1] + r[2]) / 2 * 0.97, (r[1] + r[2]) / 2 * 0.9),
                (j[2] + o, r[2], r[2] * 0.9), (lerp(j[2], j[3], 0.6) + o + Zv * 0.0008, r[3], r[3] * 0.92), (j[3] + o, r[3] * 0.9, r[3] * 0.82)]
        parts.append(loft_data(cr_sample(ctrl, 0.003), segs=20, cap0=0.004, cap1=r[3] * 0.95))
    o = lambda s: TP[s] * 0.0012
    ctrl = [(TJ[0] - TD[0] * 0.005, 0.0120, 0.0115), (lerp(TJ[0], TJ[1], 0.5) + o(0), 0.0125, 0.0115), (TJ[1] + o(1), 0.0108, 0.0098),
            (lerp(TJ[1], TJ[2], 0.5) + o(1), 0.0103, 0.0094), (TJ[2] + o(2), 0.0098, 0.0088), (lerp(TJ[2], TJ[3], 0.55) + o(2) * 1.5, 0.0100, 0.0090),
            (TJ[3] + o(2), 0.0090, 0.0080)]
    parts.append(loft_data(cr_sample(ctrl, 0.003), segs=20, cap0=0.006, cap1=0.0088))
    # thenar and hypothenar pads
    parts.append(loft_data(cr_sample([(V((-0.010, 0.264, 0.006)), 0.010, 0.009), (tpt(0, 0.45, 0.004), 0.0125, 0.0105),
                                      (tpt(0, 0.95, 0.002), 0.0095, 0.0085)], 0.003), segs=20, cap0=0.009, cap1=0.007))
    parts.append(loft_data(cr_sample([(V((0.014, 0.266, 0.001)), 0.0095, 0.0095), (V((0.024, 0.300, 0.0015)), 0.0115, 0.0110),
                                      (V((0.028, 0.335, 0.000)), 0.009, 0.0088)], 0.003), segs=20, cap0=0.010, cap1=0.008))
    # webs between the fingers and the first web space
    for a, b in (('i', 'm'), ('m', 'r'), ('r', 'p')):
        c = lerp(J[a][0], J[b][0], 0.5) + Zv * 0.001; d = (FDIR[a] + FDIR[b]).normalized()
        half = (J[a][0] - J[b][0]).length / 2
        parts.append(loft_data(cr_sample([(c - d * 0.006, half * 0.9, 0.0060), (c + d * 0.010, half * 0.75, 0.0050)], 0.003),
                               segs=16, cap0=0.004, cap1=0.004))
    web_t = lerp(TJ[0], TJ[1], 0.62); web_i = lerp(BASE['i'], MCP['i'], 0.45) + V((-0.006, 0, 0.0))
    parts.append(loft_data(cr_sample([(web_t, 0.017, 0.0075), (lerp(web_t, web_i, 0.5) + V((0, -0.003, 0)), 0.018, 0.0068), (web_i, 0.017, 0.0075)], 0.003),
                           segs=20, cap0=0.006, cap1=0.006))
    raw = make_obj('SkinRaw', parts, 'skin')
    # Union the pieces into one surface, smooth the seams, then thin it out.
    m = raw.modifiers.new('remesh', 'REMESH'); m.mode = 'VOXEL'; m.voxel_size = 0.0011; m.adaptivity = 0.0
    m = raw.modifiers.new('smooth', 'SMOOTH'); m.factor = 1.0; m.iterations = 30
    m = raw.modifiers.new('decimate', 'DECIMATE'); m.ratio = 0.16
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(raw.evaluated_get(dg))
    me.name = 'Skin'
    for p in me.polygons: p.use_smooth = True
    old = raw.data; bpy.data.objects.remove(raw, do_unlink=True); bpy.data.meshes.remove(old)
    me.materials.clear(); me.materials.append(material('skin'))
    ob = bpy.data.objects.new('Skin', me); C.objects.link(ob); CAT['Skin'] = 'skin'
    return ob

# ---------------------------------------------------------------- bones
def long_bone(name, ctrl, cap0, cap1, extra=(), segs=16):
    parts = [loft_data(cr_sample(ctrl, 0.003), segs=segs, cap0=cap0, cap1=cap1)]
    for c, r in extra: parts.append(sphere_data(c, r))
    return make_obj(name, parts, 'bone')

def tube_bone(name, a, b, rb, rs, rh, flat=0.85, bow=0.0012, segs=14):
    """Metacarpal or phalanx from joint a to joint b: flared base, slim shaft, rounded head, slight dorsal bow."""
    d = (b - a).normalized(); L = (b - a).length
    a2 = a + d * (0.0008 + rb * 0.55); b2 = b - d * (0.0008 + rh * 0.75)
    ctrl = []
    for i in range(9):
        u = i / 8
        r = rs + (rb - rs) * math.exp(-(u / 0.2) ** 2) + (rh - rs) * math.exp(-((1 - u) / 0.17) ** 2)
        ctrl.append((lerp(a2, b2, u) - Zv * bow * math.sin(math.pi * u), r, r * flat))
    return make_obj(name, [loft_data(cr_sample(ctrl, 0.002), segs=segs, cap0=rb * 0.55, cap1=rh * 0.75)], 'bone')

def build_bones():
    long_bone('B_radius', [(V((-0.018, -0.006, -0.002)), 0.0105, 0.0105), (V((-0.018, 0.004, -0.002)), 0.0105, 0.0105),
                           (V((-0.0175, 0.012, -0.002)), 0.0068, 0.0068), (V((-0.0165, 0.026, 0.0005)), 0.0078, 0.0072),
                           (V((-0.0182, 0.060, -0.001)), 0.0064, 0.0058), (V((-0.0172, 0.130, -0.001)), 0.0068, 0.0056),
                           (V((-0.0150, 0.190, -0.002)), 0.0088, 0.0066), (V((-0.0115, 0.225, -0.003)), 0.0122, 0.0086),
                           (V((-0.0100, 0.246, -0.003)), 0.0145, 0.0095)], 0.002, 0.004,
              extra=[((-0.0215, 0.249, -0.001), (0.0035, 0.005, 0.004)), ((-0.008, 0.236, -0.0115), (0.0025, 0.005, 0.002))])
    long_bone('B_ulna', [(V((0.010, -0.030, -0.022)), 0.007, 0.007), (V((0.010, -0.020, -0.016)), 0.011, 0.012),
                         (V((0.010, -0.002, -0.010)), 0.010, 0.014), (V((0.011, 0.018, -0.007)), 0.0095, 0.0095),
                         (V((0.014, 0.055, -0.006)), 0.0080, 0.0080), (V((0.017, 0.130, -0.0055)), 0.0062, 0.0066),
                         (V((0.0165, 0.200, -0.004)), 0.0054, 0.0056), (V((0.0152, 0.232, -0.004)), 0.0068, 0.0068),
                         (V((0.0150, 0.247, -0.004)), 0.0072, 0.0070)], 0.006, 0.004,
              extra=[((0.009, 0.004, 0.003), (0.006, 0.005, 0.005)), ((0.0185, 0.252, -0.007), (0.0025, 0.004, 0.0025))])
    carpals = {'scaphoid': ((-0.014, 0.263, -0.002), (0.0085, 0.0065, 0.0065), 0.5), 'lunate': ((-0.001, 0.261, -0.003), (0.0065, 0.006, 0.0068), 0),
               'triquetrum': ((0.011, 0.264, -0.005), (0.006, 0.006, 0.0055), -0.3), 'pisiform': ((0.014, 0.266, 0.0055), (0.0042, 0.0045, 0.0042), 0),
               'trapezium': ((-0.020, 0.278, 0.003), (0.007, 0.0065, 0.006), 0.4), 'trapezoid': ((-0.010, 0.281, -0.004), (0.0055, 0.0055, 0.0055), 0),
               'capitate': ((0.000, 0.279, -0.004), (0.0065, 0.010, 0.007), 0), 'hamate': ((0.012, 0.278, -0.003), (0.0065, 0.0085, 0.007), -0.15)}
    for nm, (c, r, rot) in carpals.items():
        parts = [sphere_data(c, r, rot)]
        if nm == 'hamate': parts.append(sphere_data((0.0135, 0.281, 0.0055), (0.0025, 0.004, 0.0038)))
        make_obj('B_' + nm, parts, 'bone')
    for n, k in enumerate(FK, start=2):
        tube_bone(f'B_MC__{n}', BASE[k], MCP[k], 0.0058, 0.0037, 0.0056, flat=0.95)
        rads = [(0.0050, 0.0033, 0.0042), (0.0042, 0.0029, 0.0035), (0.0035, 0.0021, 0.0030)]
        for s in range(3):
            tube_bone(f'B_phalanx__{k}{s + 1}', J[k][s], J[k][s + 1], *rads[s], flat=0.8)
    tube_bone('B_MC__1', TJ[0], TJ[1], 0.0060, 0.0040, 0.0058, flat=0.9, bow=0.0)
    tube_bone('B_phalanx__t1', TJ[1], TJ[2], 0.0055, 0.0036, 0.0045, flat=0.8, bow=0.0)
    tube_bone('B_phalanx__t2', TJ[2], TJ[3], 0.0045, 0.0026, 0.0035, flat=0.8, bow=0.0)

# ---------------------------------------------------------------- muscles, tendons, nerves, vessels
def s(y, sx, sz, rx, rz, k=1.3):
    """Belly section in plate units. k > 1 so neighbouring bellies touch, as they do in a real forearm;
    the radius is then capped so the belly stays inside the skin."""
    h = hw(y); d = math.hypot(sx, sz / dr(y))
    room = max(0.04, 0.94 - d); big = max(rx, rz / dr(y)) * k
    k *= min(1.0, room / big) if big > 0 else 1.0
    return (V((sx * h, y, sz * h)), rx * h * k, rz * h * k)
def a(p, rx, rz): return (V(p), rx, rz)

def belly(key, ctrl, segs=14):
    c0 = min(ctrl[0][1], ctrl[0][2]) * 0.9; c1 = min(ctrl[-1][1], ctrl[-1][2]) * 0.9
    return make_obj(uname('M_' + key), [loft_data(cr_sample(ctrl, 0.004), segs=segs, cap0=c0, cap1=c1)], 'muscle')

def cord(prefix, key, pts, r, cat, flat=1.0, segs=8, step=0.004):
    ctrl = [(V(p), r, r * flat) for p in pts]
    return make_obj(uname(prefix + key), [loft_data(cr_sample(ctrl, step), segs=segs, cap0=r * 0.8, cap1=r * 0.8)], cat)

def tendon(key, pts, r, flat=1.0): return cord('T_', key, pts, r, 'tendon', flat)
def last(ctrl): return ctrl[-1][0]

def flexor_hand(k, deep):
    if deep:
        return [fpt(k, 0, 0.15, 0.0085), fpt(k, 0, 0.6, 0.0090), fpt(k, 1, 0, 0.0075), fpt(k, 1, 0.5, 0.0052),
                fpt(k, 2, 0, 0.0046), fpt(k, 2, 0.5, 0.0040), fpt(k, 3, 0.12, 0.0034)]
    return [fpt(k, 0, 0.15, 0.0105), fpt(k, 0, 0.6, 0.0112), fpt(k, 1, 0, 0.0095), fpt(k, 1, 0.5, 0.0064),
            fpt(k, 2, 0, 0.0050), fpt(k, 2, 0.12, 0.0040)]
def extensor_hand(k, dx=0.0):
    return [fpt(k, 0, 0.12, -0.0072, dx), fpt(k, 0, 0.55, -0.0068, dx), fpt(k, 1, 0, -0.0078, dx * 0.6), fpt(k, 1, 0.5, -0.0055),
            fpt(k, 2, 0, -0.0052), fpt(k, 2, 0.5, -0.0042), fpt(k, 3, 0.1, -0.0036)]

def build_forearm_muscles():
    # Finger flexors and extensors, one slip per finger. (mid-forearm x, z, rx, rz) and (wrist x, z) come from the app's old plates.
    FDS = {'m': (0.04, 0.47, .12, .09, -0.03, 0.30), 'r': (0.29, 0.47, .11, .09, 0.12, 0.30), 'i': (-0.10, 0.31, .11, .08, -0.12, 0.18), 'p': (0.33, 0.30, .09, .07, 0.25, 0.18)}
    FDP = {'i': (0.03, 0.08, .10, .08, -0.19, 0.05), 'm': (0.21, 0.08, .09, .08, -0.04, 0.03), 'r': (0.38, 0.07, .09, .08, 0.11, 0.03), 'p': (0.54, 0.05, .08, .08, 0.25, 0.06)}
    ED = {'i': (-0.42, -0.56, .09, .085, 0.04, -0.47), 'm': (-0.25, -0.61, .09, .085, 0.13, -0.48), 'r': (-0.08, -0.62, .09, .08, 0.22, -0.47), 'p': (0.08, -0.60, .08, .075, 0.30, -0.44)}
    for k, (mx, mz, rx, rz, wx, wz) in FDS.items():
        c = [s(0.01, 0.55, 0.32, .05, .05), s(0.06, lerp(.55, mx, .6), lerp(.32, mz, .6), rx * 1.1, rz * 1.1), s(0.13, mx, mz, rx, rz),
             s(0.19, mx * .7 + wx * .3, mz, rx * .55, rz * .55), s(0.215, mx * .5 + wx * .5, mz * .95, .035, .03)]
        belly('FDS_' + k, c)
        tendon('FDS_' + k, [last(c), SP(0.245, lerp(mx, wx, .7), lerp(mz, wz, .5)), SP(0.272, wx, wz)] + flexor_hand(k, False), 0.0016)
    for k, (mx, mz, rx, rz, wx, wz) in FDP.items():
        c = [s(0.02, 0.42, 0.0, .06, .05), s(0.07, lerp(.42, mx, .6), lerp(0, mz, .6), rx * 1.15, rz * 1.1), s(0.13, mx, mz, rx, rz),
             s(0.195, mx * .7 + wx * .3, mz, rx * .6, rz * .6), s(0.22, mx * .5 + wx * .5, mz, .035, .03)]
        belly('FDP_' + k, c)
        tendon('FDP_' + k, [last(c), SP(0.248, lerp(mx, wx, .75), lerp(mz, wz, .6)), SP(0.272, wx, wz)] + flexor_hand(k, True), 0.0016)
    for k, (mx, mz, rx, rz, wx, wz) in ED.items():
        c = [s(0.0, -0.70, -0.40, .05, .05), s(0.06, lerp(-.7, mx, .6), lerp(-.4, mz, .6), rx * 1.1, rz * 1.05), s(0.13, mx, mz, rx, rz),
             s(0.19, mx * .6 + wx * .4, mz * .95, rx * .55, rz * .5), s(0.212, mx * .45 + wx * .55, mz * .9, .03, .025)]
        belly('ED_' + k, c)
        tendon('ED_' + k, [last(c), SP(0.245, mx * .2 + wx * .8, -0.50), SP(0.268, wx, wz)] + extensor_hand(k), 0.0012, flat=0.6)

    c = [s(0.005, 0.62, 0.30, .05, .05), s(0.05, 0.05, 0.52, .16, .12), s(0.13, -0.40, 0.48, .14, .10), s(0.175, -0.47, 0.46, .06, .045)]
    belly('FCR', c)
    tendon('FCR', [last(c), SP(0.22, -0.52, 0.45), SP(0.255, -0.56, 0.32), SP(0.272, -0.55, 0.22), V((-0.015, 0.282, 0.006)), BASE['i'] + V((-0.001, 0.004, 0.006))], 0.0016)
    c = [s(0.005, 0.55, 0.42, .04, .04), s(0.06, 0.25, 0.62, .09, .06), s(0.12, -0.08, 0.65, .07, .05), s(0.155, -0.10, 0.66, .035, .03)]
    belly('PL', c)
    pl_end = V((-0.001, 0.305, 0.0128))
    tendon('PL', [last(c), SP(0.21, -0.05, 0.64), SP(0.255, -0.02, 0.56), SP(0.275, -0.02, 0.50), V((0, 0.292, 0.0125)), pl_end], 0.0011, flat=0.7)
    for k in FK:  # palmar aponeurosis slips
        tendon('PL', [pl_end, lerp(pl_end, fpt(k, 1, 0, 0.0115) - FDIR[k] * 0.006, 0.5) + Zv * 0.0003, fpt(k, 1, 0, 0.0112) - FDIR[k] * 0.006], 0.0006, flat=0.5)
    c = [s(-0.01, 0.80, -0.05, .08, .08), s(0.05, 0.72, 0.25, .21, .15), s(0.13, 0.62, 0.36, .19, .13), s(0.20, 0.60, 0.34, .09, .07), s(0.225, 0.6, 0.33, .05, .04)]
    belly('FCU', c)
    tendon('FCU', [last(c), SP(0.25, 0.58, 0.32), V((0.014, 0.266, 0.0095))], 0.0018)
    c = [s(-0.05, -0.80, -0.05, .08, .08), s(0.0, -0.88, 0.05, .17, .15), s(0.06, -0.84, 0.10, .17, .20), s(0.13, -0.76, 0.12, .13, .20), s(0.165, -0.78, 0.08, .06, .06)]
    belly('BR', c)
    tendon('BR', [last(c), SP(0.21, -0.86, 0.0), V((-0.0215, 0.247, -0.0005))], 0.0018)
    c = [s(-0.04, -0.86, -0.15, .08, .08), s(0.02, -0.88, -0.12, .13, .13), s(0.08, -0.84, -0.15, .12, .14), s(0.13, -0.80, -0.17, .10, .12), s(0.155, -0.78, -0.20, .05, .05)]
    belly('ECRL', c)
    tendon('ECRL', [last(c), SP(0.21, -0.68, -0.32), SP(0.255, -0.52, -0.40), SP(0.268, -0.50, -0.36), BASE['i'] + V((-0.002, 0.003, -0.0075))], 0.0015)
    c = [s(0.0, -0.80, -0.30, .07, .07), s(0.06, -0.75, -0.38, .14, .11), s(0.13, -0.67, -0.40, .13, .10), s(0.17, -0.62, -0.42, .05, .04)]
    belly('ECRB', c)
    tendon('ECRB', [last(c), SP(0.22, -0.45, -0.48), SP(0.255, -0.33, -0.48), SP(0.268, -0.32, -0.40), BASE['m'] + V((0, 0.003, -0.0085))], 0.0016)
    c = [s(0.0, -0.62, -0.48, .04, .04), s(0.07, -0.1, -0.6, .07, .07), s(0.13, 0.23, -0.56, .07, .075), s(0.20, 0.36, -0.5, .04, .04)]
    belly('EDM', c)
    tendon('EDM', [last(c), SP(0.255, 0.45, -0.42), SP(0.268, 0.45, -0.38)] + extensor_hand('p', 0.0018), 0.0010, flat=0.7)
    c = [s(0.0, -0.5, -0.55, .05, .05), s(0.06, 0.15, -0.55, .13, .09), s(0.13, 0.45, -0.45, .14, .10), s(0.20, 0.58, -0.32, .07, .05), s(0.22, 0.6, -0.3, .04, .035)]
    belly('ECU', c)
    tendon('ECU', [last(c), SP(0.25, 0.62, -0.28), SP(0.268, 0.62, -0.26), BASE['p'] + V((0.006, 0.002, -0.003))], 0.0016)
    c = [s(0.06, 0.25, -0.32, .04, .04), s(0.10, -0.10, -0.32, .10, .06), s(0.15, -0.45, -0.30, .10, .06), s(0.20, -0.70, -0.22, .05, .04)]
    belly('APL', c)
    tendon('APL', [last(c), SP(0.235, -0.86, -0.12), SP(0.262, -0.88, -0.08), TJ[0] + V((-0.006, 0.002, -0.001))], 0.0013)
    c = [s(0.10, 0.10, -0.36, .04, .035), s(0.14, -0.25, -0.40, .08, .055), s(0.19, -0.55, -0.32, .06, .04), s(0.21, -0.68, -0.26, .03, .025)]
    belly('EPB', c)
    tendon('EPB', [last(c), SP(0.24, -0.82, -0.18), SP(0.262, -0.80, -0.20), tpt(0, 0.15, -0.0070), tpt(0, 0.6, -0.0088), tpt(1, 0, -0.0086), tpt(1, 0.12, -0.0070)], 0.0010)
    c = [s(0.06, 0.35, -0.28, .04, .04), s(0.11, 0.05, -0.34, .10, .06), s(0.17, -0.05, -0.38, .07, .05), s(0.21, -0.12, -0.45, .03, .025)]
    belly('EPL', c)
    tendon('EPL', [last(c), SP(0.24, -0.22, -0.55), SP(0.262, -0.36, -0.50), lerp(BASE['i'], TJ[0], 0.5) + V((0, 0.004, -0.008)),
                   tpt(0, 0.5, -0.0085), tpt(1, 0, -0.0085), tpt(1, 0.5, -0.0062), tpt(2, 0, -0.0058), tpt(2, 0.2, -0.0048)], 0.0010)
    c = [s(0.11, 0.40, -0.30, .04, .035), s(0.15, 0.20, -0.36, .07, .05), s(0.20, 0.05, -0.40, .05, .035), s(0.225, 0.0, -0.42, .025, .02)]
    belly('EIP', c)
    tendon('EIP', [last(c), SP(0.26, -0.02, -0.40), SP(0.268, -0.03, -0.38), fpt('i', 0, 0.5, -0.0066, 0.0018), fpt('i', 1, 0, -0.0080, 0.0012), fpt('i', 1, 0.3, -0.0058)], 0.0009)
    c = [s(0.05, -0.40, 0.05, .05, .04), s(0.10, -0.38, 0.13, .13, .09), s(0.15, -0.36, 0.16, .14, .09), s(0.21, -0.36, 0.14, .06, .04), s(0.225, -0.36, 0.12, .035, .03)]
    belly('FPL', c)
    tendon('FPL', [last(c), SP(0.25, -0.36, 0.11), SP(0.272, -0.36, 0.10), tpt(0, 0.35, 0.009), tpt(0, 0.8, 0.0085), tpt(1, 0, 0.0078),
                   tpt(1, 0.5, 0.0058), tpt(2, 0, 0.0052), tpt(2, 0.15, 0.0045)], 0.0015)
    # No live activity for these three, but they belong in the picture.
    belly('PT', [s(-0.01, 0.70, 0.30, .08, .08), s(0.03, 0.30, 0.30, .13, .11), s(0.07, -0.40, 0.30, .12, .10), s(0.10, -0.62, 0.0, .07, .06)])
    belly('SUP', [s(0.0, 0.30, -0.25, .05, .05), s(0.02, -0.2, -0.35, .10, .06), s(0.05, -0.55, -0.15, .10, .07), s(0.07, -0.62, 0.05, .05, .05)])
    belly('PQ', [a((0.017, 0.225, 0.002), 0.008, 0.0022), a((0.0, 0.229, 0.0045), 0.0085, 0.0026), a((-0.016, 0.228, 0.002), 0.008, 0.0022)])

def build_hand_muscles():
    belly('THEN', [a((-0.011, 0.272, 0.0105), 0.0045, 0.004), (tpt(0, 0.45, 0.0085) + Zv * 0.002, 0.0095, 0.0072), (tpt(0, 0.95, 0.0065), 0.004, 0.0035)])
    belly('THEN', [a((-0.016, 0.278, 0.008), 0.0035, 0.003), (tpt(0, 0.5, 0.0055), 0.0065, 0.005), (tpt(0, 0.9, 0.0045), 0.003, 0.0028)])
    belly('HYPO', [a((0.014, 0.268, 0.011), 0.004, 0.004), a((0.024, 0.300, 0.0085), 0.0085, 0.0075), (fpt('p', 1, 0, 0.004) + V((0.0055, -0.002, 0)), 0.003, 0.003)])
    belly('HYPO', [a((0.012, 0.280, 0.008), 0.003, 0.003), (fpt('p', 0, 0.5, 0.0045, 0.002), 0.006, 0.0055), (fpt('p', 0, 0.95, 0.0035), 0.0025, 0.0025)])
    belly('ADP', [(fpt('m', 0, 0.7, 0.0068), 0.006, 0.0025), (lerp(fpt('m', 0, 0.6, 0.007), TJ[1], 0.5) + Zv * 0.003, 0.009, 0.0028), (TJ[1] + V((0.004, 0.002, 0.004)), 0.003, 0.0025)])
    o = V((-0.002, 0.285, 0.0055))
    belly('ADP', [(o, 0.004, 0.003), (lerp(o, TJ[1], 0.5) + Zv * 0.004, 0.006, 0.0035), (TJ[1] + V((0.004, 0.0, 0.004)), 0.003, 0.0028)])
    for k in FK:
        belly('LUM_' + k, [(fpt(k, 0, 0.35, 0.0098, -0.002), 0.0016, 0.0016), (fpt(k, 0, 0.75, 0.0095, -0.0045), 0.0028, 0.0026),
                           (fpt(k, 1, 0, 0.0045, -0.0068), 0.0022, 0.002), (fpt(k, 1, 0.25, -0.001, -0.0068), 0.0012, 0.0012)], segs=10)
    A_ = lerp(lerp(TJ[0], TJ[1], 0.3), BASE['i'], 0.5) - Zv * 0.002
    B_ = lerp(lerp(TJ[0], TJ[1], 0.75), fpt('i', 0, 0.5), 0.55)
    belly('DI1', [(A_, 0.004, 0.004), (B_, 0.008, 0.0055), (fpt('i', 0, 0.85, -0.001, -0.0065), 0.006, 0.0045), (fpt('i', 1, 0.08, -0.001, -0.0065), 0.0025, 0.0025)])
    for n, (p, q, t, sd) in enumerate((('i', 'm', 'm', -1), ('m', 'r', 'm', 1), ('r', 'p', 'r', 1)), start=2):
        belly(f'DI{n}', [(lerp(BASE[p], BASE[q], 0.5) + V((0, 0.004, -0.004)), 0.0025, 0.003),
                         (lerp(fpt(p, 0, 0.5), fpt(q, 0, 0.5), 0.5) - Zv * 0.0035, 0.0035, 0.0045),
                         (fpt(t, 0, 0.88, -0.001, sd * 0.006), 0.003, 0.0035), (fpt(t, 1, 0.08, 0, sd * 0.0062), 0.0018, 0.0018)], segs=12)
    for n, (p, q, t, sd) in enumerate((('i', 'm', 'i', 1), ('m', 'r', 'r', -1), ('r', 'p', 'p', -1)), start=1):
        near = 0.35 if t == p else 0.65
        belly(f'PI{n}', [(lerp(BASE[p], BASE[q], near) + V((0, 0.006, 0.002)), 0.0022, 0.0022),
                         (lerp(fpt(p, 0, 0.5), fpt(q, 0, 0.5), near) + Zv * 0.002, 0.0032, 0.0032),
                         (fpt(t, 0, 0.9, 0.001, sd * 0.006), 0.0025, 0.0025), (fpt(t, 1, 0.08, 0, sd * 0.006), 0.0015, 0.0015)], segs=12)

def digital(prefix, key, cat, start, a_, b_, r):
    """Common digital branch from start to the web between fingers a_ and b_, then one proper branch down each facing side."""
    web = lerp(fpt(a_, 0, 0.8, 0.0108), fpt(b_, 0, 0.8, 0.0108), 0.5)
    for k, sd in ((a_, 1), (b_, -1)):
        sx = sd * (FRAD[k][1] * 0.62)
        cord(prefix, key, [start, web, fpt(k, 1, 0.1, 0.0060, sx), fpt(k, 1, 0.6, 0.0050, sx), fpt(k, 2, 0.5, 0.0044, sx * 0.95),
                           fpt(k, 3, 0.6, 0.0036, sx * 0.9)], r, cat, segs=6)

def build_nerves_vessels():
    med = V((-0.006, 0.292, 0.0085))
    cord('N_', 'median', [SP(-0.06, .2, .45), SP(0.0, .15, .35), SP(.06, .12, .25), SP(.13, .12, .22), SP(.20, .02, .30), SP(.24, -.12, .40),
                          SP(.272, -.21, .31), med], 0.0019, 'nerve', flat=0.8)
    cord('N_', 'digitalN', [med, tpt(0, 0.6, 0.0095), tpt(1, 0.5, 0.0072), tpt(2, 0.6, 0.0062)], 0.0008, 'nerve', segs=6)
    cord('N_', 'digitalN', [med, fpt('i', 0, 0.6, 0.0110, -0.004), fpt('i', 1, 0.5, 0.0055, -0.0056), fpt('i', 3, 0.6, 0.0036, -0.0042)], 0.0008, 'nerve', segs=6)
    digital('N_', 'digitalN', 'nerve', med, 'i', 'm', 0.0008)
    digital('N_', 'digitalN', 'nerve', med, 'm', 'r', 0.0008)
    uln = V((0.012, 0.270, 0.0112))
    cord('N_', 'ulnarN', [SP(-0.06, .72, -.15), SP(-0.01, .80, 0.0), SP(.06, .72, .12), SP(.13, .66, .15), SP(.20, .60, .25), SP(.255, .55, .32), uln], 0.0016, 'nerve')
    cord('N_', 'digitalN', [uln, fpt('p', 0, 0.5, 0.0105, 0.006), fpt('p', 1, 0.5, 0.0048, 0.005), fpt('p', 3, 0.6, 0.0034, 0.0038)], 0.0008, 'nerve', segs=6)
    digital('N_', 'digitalN', 'nerve', uln, 'r', 'p', 0.0008)
    rs = V((-0.020, 0.285, -0.011))
    cord('N_', 'radialN', [SP(-0.06, -.80, .10), SP(0.0, -.75, .20), SP(.06, -.70, .25), SP(.13, -.66, .25), SP(.18, -.74, .08), SP(.22, -.85, -.25), SP(.25, -.70, -.38), rs], 0.0011, 'nerve')
    cord('N_', 'radialN', [rs, tpt(0, 0.6, -0.0085), tpt(1, 0.6, -0.0072)], 0.0007, 'nerve', segs=6)
    cord('N_', 'radialN', [rs, fpt('i', 0, 0.5, -0.0085, -0.004), fpt('i', 1, 0.3, -0.0065, -0.004)], 0.0007, 'nerve', segs=6)
    cord('N_', 'AIN', [SP(.03, 0.0, -.03), SP(.13, -.05, -.07), SP(.22, -.05, -.04)], 0.0007, 'nerve', segs=6)
    cord('N_', 'PIN', [SP(0.0, -.60, -.30), SP(.04, -.30, -.45), SP(.13, -.04, -.46), SP(.22, -.05, -.40)], 0.0008, 'nerve', segs=6)

    cord('A_', 'radialA', [SP(-0.06, .05, .45), SP(0.0, 0.0, .45), SP(.06, -.40, .36), SP(.13, -.58, .33), SP(.20, -.62, .33), SP(.245, -.60, .35),
                           SP(.262, -.80, .05), V((-0.023, 0.285, -0.006)), lerp(T0, BASE['i'], 0.5) + V((0, 0.010, -0.004))], 0.0013, 'artery')
    ua = V((0.010, 0.272, 0.0118))
    cord('A_', 'ulnarA', [SP(-0.06, .10, .45), SP(0.0, .15, .40), SP(.06, .40, .24), SP(.13, .57, .20), SP(.20, .52, .28), SP(.255, .45, .37), ua], 0.0014, 'artery')
    arch = [ua, V((0.012, 0.300, 0.0115)), V((0.004, 0.318, 0.0112)), V((-0.010, 0.318, 0.0112)), V((-0.020, 0.305, 0.010))]
    cord('A_', 'arch', arch, 0.0010, 'artery')
    for p, q, st in (('i', 'm', arch[3]), ('m', 'r', arch[2]), ('r', 'p', V((0.010, 0.31, 0.0113)))):
        digital('A_', 'arch', 'artery', st, p, q, 0.0006)
    cord('A_', 'AIA', [SP(.03, .05, -.06), SP(.13, .03, -.09), SP(.22, .03, -.06)], 0.0007, 'artery', segs=6)

    build_vessel_detail()

def vessel(prefix, key, pts, r0, r1, cat, segs=8):
    """Tapered tube from r0 to r1 through pts."""
    n = len(pts)
    ctrl = [(V(p), lerp(r0, r1, i / max(1, n - 1)), lerp(r0, r1, i / max(1, n - 1))) for i, p in enumerate(pts)]
    return make_obj(uname(prefix + key), [loft_data(cr_sample(ctrl, 0.003), segs=segs, cap0=r0 * 0.8, cap1=r1 * 0.8)], cat)

def fvein(key, ya, r0, r1, seed, inset=0.0018, wob=0.07, prefix='V_', cat='vein'):
    """Surface vessel on the forearm through (y, angle) waypoints, with a little natural meander."""
    rnd = random.Random(seed); pts = []
    for (y0, a0), (y1, a1) in zip(ya, ya[1:]):
        n = max(1, int(abs(y1 - y0) / 0.018))
        for s_ in range(n):
            t = s_ / n; pts.append((lerp(y0, y1, t), lerp(a0, a1, t)))
    pts.append(ya[-1])
    out = [fsurf(y, a + (0 if i in (0, len(pts) - 1) else rnd.uniform(-wob, wob)), inset) for i, (y, a) in enumerate(pts)]
    return vessel(prefix, key, out, r0, r1, cat)

def build_vessel_detail():
    # Superficial forearm veins, running from the back of the hand up to the elbow just under the skin.
    PI = math.pi
    # Angles run continuously past -pi (i.e. -3.6 is the same place as 2.68) so a path never swings the long way round.
    fvein('cephalic', [(0.262, -2.55), (0.24, -2.75), (0.20, -3.23), (0.15, -3.63), (0.10, -3.83), (0.04, -3.98), (-0.02, -4.08), (-0.066, -4.16)], 0.0013, 0.0021, 1)
    fvein('basilic', [(0.262, -0.55), (0.23, -0.35), (0.19, 0.0), (0.14, 0.35), (0.09, 0.6), (0.03, 0.78), (-0.02, 0.86), (-0.066, 0.92)], 0.0013, 0.0021, 2)
    fvein('mav', [(0.235, 1.5), (0.18, 1.62), (0.12, 1.55), (0.06, 1.68), (0.0, 1.72), (-0.03, 1.75)], 0.0008, 0.0015, 3)
    fvein('medianCubital', [(-0.018, 2.18), (-0.032, 1.85), (-0.045, 1.5), (-0.058, 1.12), (-0.066, 0.95)], 0.0018, 0.0019, 4, wob=0.03)
    fvein('accCephalic', [(0.215, -2.0), (0.16, -2.25), (0.10, -2.6), (0.05, -2.95), (0.0, -3.53), (-0.025, -3.93)], 0.0008, 0.0013, 5)
    # smaller tributaries that feed the main veins
    for i, ya in enumerate([[(0.225, -1.35), (0.17, -1.7), (0.12, -2.05), (0.085, -2.4)], [(0.215, -0.95), (0.16, -0.65), (0.11, -0.3), (0.075, 0.05), (0.05, 0.45)],
                            [(0.20, 1.95), (0.15, 1.82), (0.10, 1.68), (0.06, 1.66)], [(0.19, 1.12), (0.14, 1.28), (0.09, 1.48), (0.04, 1.62)],
                            [(0.13, -1.25), (0.09, -1.55), (0.05, -1.95), (0.02, -2.35), (-0.01, -2.7)], [(0.17, 2.45), (0.12, 2.15), (0.07, 1.92), (0.03, 1.8)]]):
        fvein('tributary', ya, 0.0005, 0.0009, 10 + i, wob=0.05)

    # Back of the hand: dorsal venous arch fed by metacarpal and finger veins, draining to cephalic and basilic.
    rnd = random.Random(7)
    arch = [dorsum(-0.026, 0.300), dorsum(-0.017, 0.312), dorsum(-0.005, 0.318), dorsum(0.007, 0.316), dorsum(0.017, 0.307), dorsum(0.025, 0.294)]
    vessel('V_', 'dorsalV', [fsurf(0.262, -2.55)] + arch[:1], 0.0013, 0.0013, 'vein')
    vessel('V_', 'dorsalV', arch, 0.0013, 0.0012, 'vein')
    vessel('V_', 'dorsalV', [arch[-1], fsurf(0.262, -0.55)], 0.0012, 0.0013, 'vein')
    webs = [('t', 'i'), ('i', 'm'), ('m', 'r'), ('r', 'p')]
    for n, (a_, b_) in enumerate(webs):
        if a_ == 't':
            web = lerp(TJ[1], MCP['i'], 0.45); web = dorsum(web.x, min(web.y, 0.345)); join = arch[0]
        else:
            web = dorsum(lerp(MCP[a_].x, MCP[b_].x, 0.5), lerp(MCP[a_].y, MCP[b_].y, 0.5) - 0.002); join = arch[n + 1]
        mid = lerp(web, join, 0.5); mid = dorsum(mid.x + rnd.uniform(-0.002, 0.002), mid.y)
        vessel('V_', 'dorsalV', [web, mid, join], 0.0007, 0.0011, 'vein')
        # dorsal digital veins down the facing sides of the two fingers
        for k, sd in ((a_, 1), (b_, -1)):
            if k == 't':
                pts = [tpt(1, 0.85, -0.0078), tpt(1, 0.45, -0.0085), tpt(0, 0.95, -0.0095), web]
            else:
                dx = sd * FRAD[k][1] * 0.45
                pts = [fdorsal(k, 2, 0.2, dx * 0.8), fdorsal(k, 1, 0.6, dx), fdorsal(k, 1, 0.15, dx), web]
            vessel('V_', 'dorsalV', pts, 0.0004, 0.0007, 'vein', segs=6)
    vessel('V_', 'dorsalV', [fdorsal('p', 2, 0.2, 0.0035), fdorsal('p', 1, 0.5, 0.004), dorsum(0.031, 0.335), arch[-1]], 0.0004, 0.0008, 'vein', segs=6)

    # Paired small veins that travel with the radial and ulnar arteries.
    for side in (-1, 1):
        o = V((side * 0.0019, 0, 0.0004))
        vessel('V_', 'comitantes', [SP(-0.06, .05, .45) + o, SP(0.0, 0.0, .45) + o, SP(.06, -.40, .36) + o, SP(.13, -.58, .33) + o, SP(.20, -.62, .33) + o, SP(.245, -.60, .35) + o], 0.0006, 0.0007, 'vein', segs=6)
        vessel('V_', 'comitantes', [SP(-0.06, .10, .45) + o, SP(0.0, .15, .40) + o, SP(.06, .40, .24) + o, SP(.13, .57, .20) + o, SP(.20, .52, .28) + o, SP(.255, .45, .37) + o], 0.0006, 0.0007, 'vein', segs=6)

    # Deep palmar arch (from the radial artery) with palmar metacarpal branches; dorsal carpal arch with dorsal metacarpal branches.
    deep = [lerp(T0, BASE['i'], 0.5) + V((0, 0.010, 0.0)), V((-0.014, 0.300, 0.0035)), V((-0.002, 0.303, 0.0040)), V((0.010, 0.301, 0.0038)), V((0.017, 0.296, 0.0032))]
    vessel('A_', 'deepArch', deep, 0.0011, 0.0007, 'artery')
    for a_, b_, st in (('i', 'm', deep[1]), ('m', 'r', deep[2]), ('r', 'p', deep[3])):
        vessel('A_', 'deepArch', [st, lerp(fpt(a_, 0, 0.6, 0.0035), fpt(b_, 0, 0.6, 0.0035), 0.5), lerp(fpt(a_, 0, 0.95, 0.009), fpt(b_, 0, 0.95, 0.009), 0.5)], 0.0006, 0.0004, 'artery', segs=6)
    dca = [V((-0.021, 0.276, -0.0085)), V((-0.010, 0.278, -0.0115)), V((0.002, 0.279, -0.0120)), V((0.013, 0.277, -0.0105)), V((0.020, 0.272, -0.0075))]
    vessel('A_', 'dorsalArch', dca, 0.0007, 0.0006, 'artery', segs=6)
    for a_, b_, st in (('i', 'm', dca[1]), ('m', 'r', dca[2]), ('r', 'p', dca[3])):
        web = lerp(MCP[a_], MCP[b_], 0.5)
        vessel('A_', 'dorsalArch', [st, lerp(BASE[a_], BASE[b_], 0.5) + V((0, 0.006, -0.0075)), lerp(fpt(a_, 0, 0.6), fpt(b_, 0, 0.6), 0.5) - Zv * 0.006, web - Zv * 0.0065], 0.0005, 0.0003, 'artery', segs=6)

    # Skin nerves of the forearm alongside the cephalic and basilic veins; dorsal branch of the ulnar nerve to the back of the hand.
    fvein('LABC', [(-0.066, 2.02), (-0.02, 2.08), (0.04, 2.18), (0.10, 2.33), (0.16, 2.55), (0.21, 2.85)], 0.0007, 0.0004, 21, inset=0.0026, prefix='N_', cat='nerve')
    fvein('MABC', [(-0.066, 1.02), (-0.02, 0.95), (0.04, 0.86), (0.10, 0.68), (0.16, 0.42), (0.21, 0.1)], 0.0007, 0.0004, 22, inset=0.0026, prefix='N_', cat='nerve')
    du = [SP(.19, .62, .22), SP(.21, .82, -.05), fsurf(0.235, -0.45, 0.003), dorsum(0.022, 0.285, 0.0028)]
    cord('N_', 'dorsalUlnarN', du, 0.0008, 'nerve', segs=6)
    for k in ('r', 'p'):
        cord('N_', 'dorsalUlnarN', [du[-1], fdorsal(k, 0, 0.6, -0.003 if k == 'p' else 0.004, 0.0028), fdorsal(k, 1, 0.3, -0.003 if k == 'p' else 0.003, 0.0022)], 0.0005, 'nerve', segs=6)
    # muscular twigs of the median nerve into the superficial flexors
    for y, tx, tz in ((0.03, 0.45, 0.38), (0.05, -0.15, 0.52), (0.08, 0.05, 0.42), (0.11, 0.35, 0.40)):
        p0 = SP(y, .13, .27)
        cord('N_', 'median', [p0, lerp(p0, SP(y + 0.012, tx, tz), 0.5) + V((0, 0, 0.0006)), SP(y + 0.012, tx, tz)], 0.0004, 'nerve', segs=6)

def build_other():
    # Interosseous membrane: a thin sheet strung between the radius and ulna shafts.
    verts, faces, fuv = [], [], []
    ys = [0.03 + 0.01 * i for i in range(20)]
    for y in ys:
        rx = interp([(0.03, -0.0165), (0.13, -0.0172), (0.22, -0.012)], y) + 0.005
        ux = interp([(0.03, 0.0125), (0.13, 0.017), (0.22, 0.0155)], y) - 0.004
        z = interp([(0.03, -0.004), (0.13, -0.003), (0.22, -0.0035)], y)
        verts += [V((rx, y, z + 0.0003)), V((ux, y, z + 0.0003)), V((rx, y, z - 0.0003)), V((ux, y, z - 0.0003))]
    for i in range(len(ys) - 1):
        b, n = i * 4, (i + 1) * 4
        for q in ((b, b + 1, n + 1, n), (b + 3, b + 2, n + 2, n + 3), (b + 1, b + 3, n + 3, n + 1), (b + 2, b, n, n + 2)):
            faces.append(q); fuv.append([(0, 0), (1, 0), (1, 1), (0, 1)])
    make_obj('X_membrane', [(verts, faces, fuv)], 'fascia')
    cord('X_', 'retinaculum', [V((-0.019, 0.278, 0.0085)), V((-0.010, 0.279, 0.0125)), V((0.002, 0.279, 0.0135)), V((0.012, 0.279, 0.0115)), V((0.0155, 0.279, 0.0080))],
         0.0065, 'fascia', flat=0.14, segs=12)
    cord('X_', 'extRetinaculum', [V((-0.026, 0.244, -0.004)), V((-0.020, 0.244, -0.0125)), V((0.0, 0.244, -0.0158)), V((0.017, 0.244, -0.0135)), V((0.024, 0.244, -0.006))],
         0.008, 'fascia', flat=0.09, segs=12)
    for k in FK:
        d = FDIR[k]; r = FRAD[k][3]; top = -r * 0.9 + 0.0012 - 0.0004
        pts = [J[k][2] + d * (FLEN[k][2] * 0.35) + Zv * top, J[k][3] + d * 0.0012 + Zv * (top - 0.0004)]
        make_obj(uname('Z_nail'), [loft_data(cr_sample([(p, r * 0.6, 0.0014) for p in pts], 0.002), segs=16, cap0=0.001, cap1=0.002)], 'nail')
    pts = [tpt(2, 0.45, -0.0072), TJ[3] + TD[2] * 0.001 - TP[2] * 0.0068]
    make_obj(uname('Z_nail'), [loft_data(cr_sample([(p, 0.0050, 0.0012) for p in pts], 0.002), segs=16, cap0=0.001, cap1=0.002)], 'nail')

def tuck_inside(skin, margin=0.0007):
    """Push any vertex of the inner anatomy that ends up outside (or right at) the skin back just under it."""
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromObject(skin, bpy.context.evaluated_depsgraph_get())
    moved = 0
    for ob in C.objects:
        if ob.type != 'MESH' or ob is skin or ob.name.startswith('Z_nail'): continue
        m = margin * (0.6 if ob.name.startswith(('V_', 'N_', 'A_')) else 1.0)
        for v in ob.data.vertices:
            loc, nrm, _, _ = bvh.find_nearest(v.co)
            if loc is None: continue
            if (v.co - loc).dot(nrm) > -m:
                v.co = loc - nrm * m; moved += 1
        ob.data.update()
    return moved

# ---------------------------------------------------------------- rig and weights
BONES = {}  # name -> (head, tail, parent, flex axis, spread axis)
def build_rig():
    arm_data = bpy.data.armatures.new('Rig'); arm = bpy.data.objects.new('Rig', arm_data); C.objects.link(arm)
    BONES['forearm'] = (V((0, -0.07, 0)), W.copy(), None, V((1, 0, 0)), Zv)
    BONES['hand'] = (W.copy(), MCP['m'].copy(), 'forearm', V((1, 0, 0)), Zv)
    for s_ in range(3):
        BONES[f'thumb_{s_ + 1}'] = (TJ[s_], TJ[s_ + 1], 'hand' if s_ == 0 else f'thumb_{s_}', TD[s_].cross(TP[s_]).normalized(), TP[s_])
    for k in FK:
        for s_ in range(3):
            BONES[f'{FNAME[k]}_{s_ + 1}'] = (J[k][s_], J[k][s_ + 1], 'hand' if s_ == 0 else f'{FNAME[k]}_{s_}', FDIR[k].cross(Zv).normalized(), Zv)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for nm, (h, t, par, flex, spread) in BONES.items():
        eb = arm_data.edit_bones.new(nm); eb.head = h; eb.tail = t; eb.align_roll(spread)
        if par: eb.parent = arm_data.edit_bones[par]; eb.use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    rig = {'bones': {nm: {'flex': list(b[3]), 'spread': list(b[4])} for nm, b in BONES.items()}, 'units': 'm', 'axes': 'blender'}
    arm['atlas_rig'] = json.dumps(rig)
    return arm

def seg_dist(P, a_, b_):
    a_ = np.array(a_); b_ = np.array(b_); ab = b_ - a_
    t = np.clip(((P - a_) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a_ + t[:, None] * ab), axis=1)

def weight_segments():
    seg = {'forearm': [(V((0, -0.12, 0)), W)], 'hand': [(W, T0)]}
    for k in FK: seg['hand'] += [(W, BASE[k]), (BASE[k], MCP[k])]
    for nm, (h, t, *_ ) in BONES.items():
        if nm in seg: continue
        tail = t + (t - h).normalized() * 0.01 if nm.endswith('_3') else t
        seg[nm] = [(h, tail)]
    return seg

RIGID = {'B_radius': 'forearm', 'B_ulna': 'forearm', 'B_MC__1': 'thumb_1', 'B_phalanx__t1': 'thumb_2', 'B_phalanx__t2': 'thumb_3', 'X_membrane': 'forearm'}
for nm in ('scaphoid', 'lunate', 'triquetrum', 'pisiform', 'trapezium', 'trapezoid', 'capitate', 'hamate'): RIGID['B_' + nm] = 'hand'
for n in range(2, 6): RIGID[f'B_MC__{n}'] = 'hand'
for k in FK:
    for s_ in range(3): RIGID[f'B_phalanx__{k}{s_ + 1}'] = f'{FNAME[k]}_{s_ + 1}'

def skin_all(arm):
    seg = weight_segments(); names = list(seg)
    nails = [o for o in C.objects if o.name.startswith('Z_nail')]
    for i, o in enumerate(sorted(nails, key=lambda o: o.name)):  # nails follow the fingertip bones (order: i, m, r, p, thumb)
        RIGID[o.name] = (['index_3', 'middle_3', 'ring_3', 'little_3', 'thumb_3'])[i]
    for ob in [o for o in C.objects if o.type == 'MESH']:
        me = ob.data; n = len(me.vertices)
        if ob.name in RIGID:
            ob.vertex_groups.new(name=RIGID[ob.name]).add(range(n), 1.0, 'REPLACE')
        else:
            P = np.empty(n * 3); me.vertices.foreach_get('co', P); P = P.reshape(-1, 3)
            D = np.stack([np.min([seg_dist(P, a_, b_) for a_, b_ in seg[nm]], axis=0) for nm in names], axis=1)
            Wt = (D + 0.0015) ** -6.0
            top = np.argsort(-Wt, axis=1)[:, :4]
            groups = {}
            for vi in range(n):
                idx = top[vi]; w = Wt[vi, idx]; w = w / w.sum()
                for bi, wi in zip(idx, w):
                    if wi > 0.01:
                        nm = names[bi]
                        if nm not in groups: groups[nm] = ob.vertex_groups.new(name=nm)
                        groups[nm].add([vi], float(wi), 'REPLACE')
        md = ob.modifiers.new('Armature', 'ARMATURE'); md.object = arm
        ob.parent = arm

# ---------------------------------------------------------------- build + export
for o in bpy.context.scene.objects:  # keep the default scene's cube out of the way (hidden, not deleted)
    if o.name not in C.objects and o.type == 'MESH': o.hide_set(True)
skin = build_skin()
build_bones()
build_forearm_muscles()
build_hand_muscles()
build_nerves_vessels()
build_other()
print('tucked under the skin:', tuck_inside(skin), 'vertices')
arm = build_rig()
skin_all(arm)
print('objects:', len(C.objects), 'verts:', sum(len(o.data.vertices) for o in C.objects if o.type == 'MESH'), 'skin verts:', len(skin.data.vertices))

if DO_EXPORT:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    vl = bpy.context.view_layer
    vl.active_layer_collection = vl.layer_collection.children[COLL]
    bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', use_active_collection=True, use_active_collection_with_nested=True,
                              export_extras=True, export_yup=True, export_apply=False, export_skins=True, export_animations=False,
                              export_morph=False, export_texcoords=True, export_normals=True, export_materials='EXPORT',
                              export_cameras=False, export_lights=False, export_def_bones=False, export_leaf_bone=False)
    print('exported', OUT, os.path.getsize(OUT) // 1024, 'KB')
