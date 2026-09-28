"""Scene composer: reproducibility, physical consistency, and variation."""

import math

import numpy as np
import pytest

import pippin
from pippin.scene import Box, Cylinder, Mesh, SceneBuilder, Sphere

TOP = 0.4


def builder():
    sb = SceneBuilder()
    sb.floor()
    sb.table(size=(0.45, 0.35), height=TOP)
    pool = [
        Box(size=((0.015, 0.03),) * 3),
        Cylinder(radius=(0.015, 0.03), half_height=(0.02, 0.04)),
        Sphere(radius=(0.015, 0.03)),
        Mesh("assets/urdf/meshes/wedge.obj"),
    ]
    sb.objects(pool, count=(1, 4), region=((-0.3, 0.3), (-0.25, 0.25)), surface=TOP)
    return sb


def test_same_seed_reproduces_and_different_seed_varies():
    a, b, c = builder().build(16, seed=7), builder().build(16, seed=7), builder().build(16, seed=8)
    assert a.xml == b.xml
    np.testing.assert_array_equal(a.qpos0, b.qpos0)
    for (na, ia, va), (nb, ib, vb) in zip(a.params, b.params):
        assert (na, ia) == (nb, ib)
        np.testing.assert_array_equal(va, vb)
    assert not np.array_equal(a.qpos0, c.qpos0)


def test_objects_start_separated_and_settle_on_the_table():
    n = 16
    plan = builder().build(n, seed=3)
    info = pippin._pippin.model_info(plan.xml)
    qadr = dict(zip(info["joint_names"], info["joint_qposadr"]))
    slots = [j for j in info["joint_names"] if j.startswith("obj")]
    env = pippin.AsyncEnv(plan.xml, n, groups=1, substeps=250)
    plan.apply(env)
    start = env.recv()["qpos"]
    for e in range(n):
        active = [qadr[j] for s, j in enumerate(slots) if plan.choices[e, s] >= 0]
        xy = [start[e, q:q + 2] for q in active]
        for i in range(len(xy)):
            for k in range(i + 1, len(xy)):
                assert np.linalg.norm(xy[i] - xy[k]) > 0.02, f"env {e}: objects overlap at start"
    for _ in range(4):  # 2 s of settling
        env.send(0, np.zeros((n, env.nu)))
        obs = env.recv()
    q, v = obs["qpos"], obs["qvel"]
    assert np.abs(v).max() < 0.05, f"not settled: max |qvel| {np.abs(v).max():.3f}"
    for e in range(n):
        for s, j in enumerate(slots):
            z = q[e, qadr[j] + 2]
            if plan.choices[e, s] >= 0:
                assert TOP - 0.001 < z < TOP + 0.08, f"env {e} slot {s}: z {z:.3f} not on table"
            else:
                assert q[e, qadr[j]] > 50, "parked object should stay out of the scene"


def test_mass_matches_chosen_shape():
    plan = builder().build(32, seed=1)
    info = pippin._pippin.model_info(plan.xml)
    bid = {nm: i for i, nm in enumerate(info["body_names"])}
    gid = {nm: i for i, nm in enumerate(info["geom_names"])}
    mass = {i: v for name, i, v in plan.params if name == "body_mass"}
    size = {i: v for name, i, v in plan.params if name == "geom_size"}
    checked = 0
    for e in range(32):
        c = plan.choices[e, 0]
        if c in (0, 2):  # box or sphere: mass = density * volume, density in [300, 1200]
            s = size[gid[f"obj0_c{c}"]][e]
            vol = 8 * s[0] * s[1] * s[2] if c == 0 else 4 / 3 * math.pi * s[0] ** 3
            density = mass[bid["obj0"]][e, 0] / vol
            assert 300 - 1e-9 <= density <= 1200 + 1e-9, density
            checked += 1
    assert checked > 5


def test_bad_region_is_reported():
    sb = SceneBuilder()
    sb.floor()
    sb.objects([Sphere(radius=0.05)], count=6, region=((0, 0.1), (0, 0.1)), surface=0.0)
    with pytest.raises(ValueError, match="could not place"):
        sb.build(1)
