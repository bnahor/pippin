"""Procedural scene composition with per-environment variation.

A scene is one model shared by every environment (so batched simulation and
rendering stay fast), plus per-environment parameters that make each env
different: which object occupies each slot, its size, mass, friction, color,
and where it starts. Everything is drawn from a seeded generator, so a
(builder, seed) pair always reproduces the same set of scenes.

    from pippin.scene import SceneBuilder, Box, Cylinder, Sphere, Mesh

    sb = SceneBuilder()
    sb.floor()
    top = sb.table(size=(0.5, 0.4), height=0.4)
    sb.robot("assets/urdf/arm.urdf", pos=(-0.45, 0.0, top))
    sb.objects([Box(size=((0.02, 0.04),) * 3), Cylinder(radius=(0.02, 0.035), half_height=(0.03, 0.06)),
                Sphere(radius=(0.02, 0.04)), Mesh("assets/urdf/meshes/wedge.obj")],
               count=(2, 5), region=((-0.25, 0.35), (-0.3, 0.3)), surface=top)
    plan = sb.build(num_envs=1024, seed=0)
    env = pippin.AsyncEnv(plan.xml, 1024, render=...)
    plan.apply(env)

How variation works: each object slot is one free body holding one geom per
candidate shape. Per environment, the chosen candidate is enabled and sized,
the others are hidden and removed from collision, and the body's mass and
inertia are set to the chosen shape's. Slots unused in an environment are
parked out of view on the floor.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from pippin import _pippin

Range = "float | tuple[float, float]"

_PALETTE = [
    (0.85, 0.25, 0.20), (0.20, 0.55, 0.85), (0.95, 0.75, 0.15), (0.30, 0.70, 0.35),
    (0.60, 0.35, 0.75), (0.95, 0.50, 0.20), (0.20, 0.75, 0.75), (0.85, 0.85, 0.85),
]


def _sample(rng, r) -> float:
    if isinstance(r, (int, float)):
        return float(r)
    return float(rng.uniform(r[0], r[1]))


def _upper(r) -> float:
    return float(r) if isinstance(r, (int, float)) else float(max(r))


@dataclass
class _Sampled:
    size: tuple[float, float, float]  # geom size (MJCF layout)
    mass: float
    inertia: tuple[float, float, float]  # principal moments in the body frame
    ipos: tuple[float, float, float]  # center of mass in the body frame
    rest_z: float  # height of the body origin above the surface at rest
    radius: float  # footprint radius for non-overlapping placement


@dataclass(kw_only=True)
class ObjectSpec:
    """Shared options (keyword-only): density, friction, and colors."""

    density: object = (300.0, 1200.0)
    friction: object = (0.5, 1.0)
    colors: list | None = None  # RGB tuples; default: built-in palette

    def _common(self, rng):
        rgb = (self.colors or _PALETTE)[rng.integers(len(self.colors or _PALETTE))]
        return _sample(rng, self.density), _sample(rng, self.friction), (*rgb, 1.0)


@dataclass
class Box(ObjectSpec):
    """Box with half-extents; each entry a value or (low, high) range."""

    size: tuple = ((0.02, 0.04), (0.02, 0.04), (0.02, 0.04))

    def geom_attrs(self, meshes):
        a, b, c = (_upper(s) for s in self.size)
        return {"type": "box", "size": f"{a} {b} {c}"}

    def sample(self, rng, density):
        a, b, c = (_sample(rng, s) for s in self.size)
        m = density * 8 * a * b * c
        return _Sampled((a, b, c), m, (m * (b * b + c * c) / 3, m * (a * a + c * c) / 3, m * (a * a + b * b) / 3),
                        (0, 0, 0), c, math.hypot(a, b))


@dataclass
class Cylinder(ObjectSpec):
    """Upright cylinder: radius and half-height (values or ranges)."""

    radius: object = (0.02, 0.035)
    half_height: object = (0.03, 0.06)

    def geom_attrs(self, meshes):
        return {"type": "cylinder", "size": f"{_upper(self.radius)} {_upper(self.half_height)}"}

    def sample(self, rng, density):
        r, h = _sample(rng, self.radius), _sample(rng, self.half_height)
        m = density * math.pi * r * r * 2 * h
        ix = m * (3 * r * r + 4 * h * h) / 12
        return _Sampled((r, h, 0), m, (ix, ix, m * r * r / 2), (0, 0, 0), h, r)


@dataclass
class Sphere(ObjectSpec):
    radius: object = (0.02, 0.04)

    def geom_attrs(self, meshes):
        return {"type": "sphere", "size": f"{_upper(self.radius)}"}

    def sample(self, rng, density):
        r = _sample(rng, self.radius)
        m = density * 4 / 3 * math.pi * r**3
        i = 0.4 * m * r * r
        return _Sampled((r, 0, 0), m, (i, i, i), (0, 0, 0), r, r)


@dataclass
class Mesh(ObjectSpec):
    """Mesh object (STL/OBJ), colliding through its convex hull. Size is
    fixed by `scale`; the mesh rests on its lowest point."""

    path: str = ""
    scale: float = 1.0

    def __post_init__(self):
        self.path = os.path.abspath(self.path)
        self._props = _pippin.mesh_properties(self.path, (self.scale,) * 3)

    def geom_attrs(self, meshes):
        key = (self.path, self.scale)
        if key not in meshes:
            meshes[key] = f"scene_mesh{len(meshes)}"
        return {"type": "mesh", "mesh": meshes[key]}

    def sample(self, rng, density):
        p = self._props
        m = density * p["volume"]
        inertia = np.array(p["inertia"]).reshape(3, 3) * density
        lo, hi = p["min"], p["max"]
        radius = max(math.hypot(x, y) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]))
        # principal moments along the mesh axes (off-diagonal terms dropped)
        return _Sampled((0, 0, 0), m, tuple(np.diag(inertia)), tuple(p["com"]), -lo[2], radius)


@dataclass
class _Slots:
    pool: list
    count: object
    region: tuple
    surface: float
    gap: float
    name: str


@dataclass
class ScenePlan:
    """A built scene: shared MJCF plus per-environment parameters."""

    xml: str
    num_envs: int
    seed: int
    params: list = field(default_factory=list)  # (name, id, values[num_envs, width])
    qpos0: np.ndarray | None = None  # (num_envs, nq) initial state per env
    choices: np.ndarray | None = None  # (num_envs, slots) candidate index, -1 = parked

    def apply(self, env) -> None:
        """Push per-environment parameters and initial states into an env,
        then reset every environment to its own initial state."""
        for name, gid, values in self.params:
            env.set_param(name, gid, np.ascontiguousarray(values, dtype=np.float64))
        env.set_param("qpos0", 0, np.ascontiguousarray(self.qpos0, dtype=np.float64))
        env.reset()


class SceneBuilder:
    def __init__(self, timestep: float = 0.002):
        self.timestep = timestep
        self._world: list[ET.Element] = []
        self._assets: list[ET.Element] = []
        self._actuators: list[ET.Element] = []
        self._defaults: list[ET.Element] = []
        self._compiler = {"angle": "radian", "autolimits": "true"}
        self._slots: list[_Slots] = []
        self._has_floor = False

    # ---- static scenery ----
    def floor(self, size: float = 5.0, rgba=(0.55, 0.6, 0.62, 1.0)):
        self._world.append(ET.Element("geom", name="floor", type="plane", size=f"{size} {size} 0.1",
                                      rgba=" ".join(map(str, rgba))))
        self._has_floor = True

    def table(self, size=(0.5, 0.4), height: float = 0.4, thickness: float = 0.02, pos=(0.0, 0.0),
              rgba=(0.55, 0.42, 0.3, 1.0)) -> float:
        """Static table top plus legs; returns the height of the top surface."""
        body = ET.Element("body", name="table", pos=f"{pos[0]} {pos[1]} 0")
        z = height - thickness
        ET.SubElement(body, "geom", name="table_top", type="box", pos=f"0 0 {z}",
                      size=f"{size[0]} {size[1]} {thickness}", rgba=" ".join(map(str, rgba)))
        leg = (height - 2 * thickness) / 2
        for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            ET.SubElement(body, "geom", name=f"table_leg{i}", type="box",
                          pos=f"{sx * (size[0] - 0.03)} {sy * (size[1] - 0.03)} {leg}",
                          size=f"0.02 0.02 {leg}", rgba=" ".join(map(str, rgba)))
        self._world.append(body)
        return height

    # ---- robots ----
    def robot(self, path: str, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), floating: bool = False,
              name: str = "robot"):
        """Add a robot from URDF or MJCF. MJCF assets, defaults, and actuators
        are merged; mesh paths are made absolute."""
        path = os.path.abspath(path)
        text = open(path).read()
        root = ET.fromstring(text)
        if root.tag == "robot":
            root = ET.fromstring(_pippin.urdf_to_mjcf(path, floating))
        base = os.path.dirname(path)
        comp = root.find("compiler")
        meshdir = base
        if comp is not None:
            for k in ("angle", "eulerseq", "autolimits"):
                if k in comp.attrib:
                    self._compiler[k] = comp.attrib[k]
            if "angle" not in comp.attrib:
                self._compiler["angle"] = "degree"  # MJCF default
            meshdir = os.path.join(base, comp.attrib.get("meshdir", comp.attrib.get("assetdir", "")))
        else:
            self._compiler["angle"] = "degree"
        for asset in root.findall("asset"):
            for a in asset:
                if "file" in a.attrib and not os.path.isabs(a.attrib["file"]):
                    a.attrib["file"] = os.path.join(meshdir, a.attrib["file"])
                self._assets.append(a)
        for d in root.findall("default"):
            self._defaults.extend(list(d))
        for act in root.findall("actuator"):
            self._actuators.extend(list(act))
        mount = ET.Element("body", name=f"{name}_mount", pos=" ".join(map(str, pos)), quat=" ".join(map(str, quat)))
        for wb in root.findall("worldbody"):
            mount.extend(list(wb))
        self._world.append(mount)

    # ---- randomized objects ----
    def objects(self, pool: list, count=3, region=((-0.3, 0.3), (-0.3, 0.3)), surface: float = 0.0,
                gap: float = 0.01, name: str = "obj"):
        """Scatter `count` objects (int or (min, max)) drawn from `pool` over
        the rectangle `region` at height `surface`, without overlaps."""
        if not pool:
            raise ValueError("object pool is empty")
        self._slots.append(_Slots(list(pool), count, region, surface, gap, name))

    # ---- build ----
    def _xml(self) -> str:
        root = ET.Element("mujoco", model="pippin_scene")
        ET.SubElement(root, "compiler", **self._compiler)
        ET.SubElement(root, "option", timestep=str(self.timestep))
        if self._defaults:
            ET.SubElement(root, "default").extend(self._defaults)
        asset = ET.SubElement(root, "asset")
        asset.extend(self._assets)
        world = ET.SubElement(root, "worldbody")
        world.extend(self._world)
        meshes: dict = {}
        for g in self._slots:
            for s in range(_count_max(g.count)):
                body = ET.SubElement(world, "body", name=f"{g.name}{s}", pos=f"{100 + 2 * s} 0 1")
                ET.SubElement(body, "freejoint", name=f"{g.name}{s}_joint")
                # placeholder inertia; set per environment
                ET.SubElement(body, "inertial", pos="0 0 0", mass="0.1", diaginertia="1e-4 1e-4 1e-4")
                for c, spec in enumerate(g.pool):
                    ET.SubElement(body, "geom", name=f"{g.name}{s}_c{c}", **spec.geom_attrs(meshes))
        for (path, scale), mname in meshes.items():
            ET.SubElement(asset, "mesh", name=mname, file=path, scale=f"{scale} {scale} {scale}")
        if self._actuators:
            ET.SubElement(root, "actuator").extend(self._actuators)
        return ET.tostring(root, encoding="unicode")

    def build(self, num_envs: int, seed: int = 0) -> ScenePlan:
        xml = self._xml()
        info = _pippin.model_info(xml)
        geom_id = {n: i for i, n in enumerate(info["geom_names"])}
        body_id = {n: i for i, n in enumerate(info["body_names"])}
        jnt_q = dict(zip(info["joint_names"], info["joint_qposadr"]))
        rng = np.random.default_rng(seed)
        qpos0 = np.tile(np.asarray(info["qpos0"], dtype=np.float64), (num_envs, 1))

        params: dict = {}

        def put(name, idx, env, value):
            arr = params.get((name, idx))
            if arr is None:
                arr = params[(name, idx)] = np.zeros((num_envs, len(value)))
            arr[env] = value

        all_choices = []
        for g in self._slots:
            nslot = _count_max(g.count)
            choices = np.full((num_envs, nslot), -1)
            for e in range(num_envs):
                k = _count_sample(rng, g.count)
                placed: list[tuple[float, float, float]] = []
                for s in range(nslot):
                    b = body_id[f"{g.name}{s}"]
                    q = jnt_q[f"{g.name}{s}_joint"]
                    active = s < k
                    c = int(rng.integers(len(g.pool))) if active else 0
                    spec = g.pool[c]
                    density, friction, rgba = spec._common(rng)
                    sm = spec.sample(rng, density)
                    for ci in range(len(g.pool)):
                        gid = geom_id[f"{g.name}{s}_c{ci}"]
                        on = active and ci == c
                        put("geom_rgba", gid, e, rgba if on else (0, 0, 0, 0))
                        put("geom_contype", gid, e, (1,) if on or (not active and ci == 0) else (0,))
                        put("geom_conaffinity", gid, e, (1,) if on or (not active and ci == 0) else (0,))
                        if ci == c and not isinstance(spec, Mesh):
                            put("geom_size", gid, e, sm.size)
                        elif not isinstance(g.pool[ci], Mesh):
                            put("geom_size", gid, e, _nominal_size(g.pool[ci]))
                        put("geom_friction", gid, e, (friction,))
                    put("body_mass", b, e, (sm.mass,))
                    put("body_inertia", b, e, sm.inertia)
                    put("body_ipos", b, e, sm.ipos)
                    if active:
                        x, y = _place(rng, g, sm.radius, placed)
                        placed.append((x, y, sm.radius))
                        yaw = rng.uniform(-math.pi, math.pi)
                        pose = (x, y, g.surface + sm.rest_z + 1e-3, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
                        choices[e, s] = c
                    else:
                        if not self._has_floor:
                            raise ValueError("a variable object count needs a floor() to park unused objects")
                        pose = (100 + 2 * s + 0.01 * e, 0.0, sm.rest_z + 1e-3, 1, 0, 0, 0)
                    qpos0[e, q:q + 7] = pose
            all_choices.append(choices)

        plan = ScenePlan(xml, num_envs, seed)
        plan.params = [(name, idx, arr) for (name, idx), arr in params.items()]
        # mass/inertia/com before geometry so derived constants are computed once
        order = {"body_mass": 0, "body_ipos": 1, "body_inertia": 2}
        plan.params.sort(key=lambda p: order.get(p[0], 3))
        plan.qpos0 = qpos0
        plan.choices = np.concatenate(all_choices, axis=1) if all_choices else np.zeros((num_envs, 0), int)
        return plan


def _count_max(count) -> int:
    return int(count) if isinstance(count, int) else int(max(count))


def _count_sample(rng, count) -> int:
    return int(count) if isinstance(count, int) else int(rng.integers(count[0], count[1] + 1))


def _nominal_size(spec) -> tuple:
    return tuple(float(x) for x in spec.geom_attrs({})["size"].split()) + (0.0,) * (3 - len(spec.geom_attrs({})["size"].split()))


def _place(rng, g: _Slots, radius: float, placed) -> tuple[float, float]:
    (x0, x1), (y0, y1) = g.region
    for _ in range(200):
        x = rng.uniform(x0 + radius, x1 - radius) if x1 - x0 > 2 * radius else (x0 + x1) / 2
        y = rng.uniform(y0 + radius, y1 - radius) if y1 - y0 > 2 * radius else (y0 + y1) / 2
        if all(math.hypot(x - px, y - py) >= radius + pr + g.gap for px, py, pr in placed):
            return x, y
    raise ValueError(f"could not place object without overlap in region {g.region}; enlarge it or reduce count")
