//! Asynchronous, pipelined vector environment.
//!
//! Environments are split into `groups`. A physics thread steps one group at a
//! time on all CPU cores, then hands its poses to a render thread that drives
//! the GPU. Meanwhile the caller consumes finished groups (for example running
//! policy inference on the GPU) and sends back actions. With two or more
//! groups, CPU physics, GPU rendering, and inference overlap:
//!
//! ```text
//! physics : [A1][B1]    [A2][B2]    ...
//! render  :     [A1][B1]    [A2][B2]
//! policy  :         [A1][B1]    [A2]
//! ```
//!
//! Usage follows EnvPool: `send(group, actions)` then `recv()` returns whichever
//! group finished next. A group's frames stay valid until that group is sent
//! again (they live in the renderer's slot for that group; no copies).

use std::ops::Range;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::Instant;

use crate::{Appearance, Dims, Field, Param, Physics, Pose, Renderer, Scene};

/// Frames of one group, borrowed from the renderer's output slot.
#[derive(Clone, Copy, Debug)]
pub struct FrameView {
    pub rgba: *const u8,
    pub depth: *const f32,
    pub segmentation: *const i32,
    /// Pixels in this group (envs x views x height x width).
    pub pixels: usize,
}

// The pointers reference shared GPU buffers that stay alive for the life of
// the pipeline and are not rewritten until the group is sent again.
unsafe impl Send for FrameView {}

impl FrameView {
    /// # Safety
    /// Valid only until this group is sent again or the pipeline is dropped.
    pub unsafe fn rgba(&self) -> &[u8] {
        std::slice::from_raw_parts(self.rgba, self.pixels * 4)
    }
    /// # Safety
    /// See [`FrameView::rgba`].
    pub unsafe fn depth(&self) -> &[f32] {
        std::slice::from_raw_parts(self.depth, self.pixels)
    }
    /// # Safety
    /// See [`FrameView::rgba`].
    pub unsafe fn segmentation(&self) -> &[i32] {
        std::slice::from_raw_parts(self.segmentation, self.pixels)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub physics_ms: f64,
    pub render_ms: f64,
}

pub struct Obs {
    pub group: usize,
    pub envs: Range<usize>,
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub frames: Option<FrameView>,
    pub timing: Timing,
}

enum PhysJob {
    Step { group: usize, ctrl: Vec<f64> },
    Reset { group: usize },
    SetParam { p: Param, id: usize, envs: Range<usize>, values: Vec<f64>, reply: Sender<Result<(), String>> },
    GetParam { p: Param, id: usize, envs: Range<usize>, width: usize, reply: Sender<Result<Vec<f64>, String>> },
    Stop,
}

struct RenderJob {
    obs: Obs,
    geoms: Vec<Pose>,
    cams: Vec<Pose>,
    appearance: Vec<Appearance>,
}

pub struct AsyncEnv {
    dims: Dims,
    scene: Scene,
    groups: Vec<Range<usize>>,
    phys_tx: Sender<PhysJob>,
    out_rx: Mutex<Receiver<Result<Obs, String>>>,
    threads: Vec<JoinHandle<()>>,
}

fn split(n: usize, groups: usize) -> Vec<Range<usize>> {
    let groups = groups.clamp(1, n.max(1));
    (0..groups).map(|g| (g * n / groups)..((g + 1) * n / groups)).collect()
}

impl AsyncEnv {
    /// `substeps` physics steps are taken per `send` (control held constant).
    pub fn new(mut physics: Box<dyn Physics>, renderer: Option<Box<dyn Renderer>>, groups: usize, substeps: usize) -> AsyncEnv {
        let dims = physics.dims();
        let scene = physics.scene();
        let ranges = split(physics.num_envs(), groups);
        let (phys_tx, phys_rx) = channel::<PhysJob>();
        let (out_tx, out_rx) = channel::<Result<Obs, String>>();
        let mut threads = vec![];

        // render thread (optional)
        let render_tx = renderer.map(|mut r| {
            let (tx, rx) = channel::<RenderJob>();
            let out = out_tx.clone();
            threads.push(
                std::thread::Builder::new()
                    .name("pippin-render".into())
                    .spawn(move || {
                        while let Ok(mut job) = rx.recv() {
                            let t = Instant::now();
                            let res = r.render(job.obs.group, job.obs.envs.len(), &job.geoms, &job.cams, Some(&job.appearance)).map(|f| FrameView {
                                rgba: f.rgba.as_ptr(),
                                depth: f.depth.as_ptr(),
                                segmentation: f.segmentation.as_ptr(),
                                pixels: f.depth.len(),
                            });
                            job.obs.timing.render_ms = t.elapsed().as_secs_f64() * 1e3;
                            let msg = res.map(|fv| {
                                job.obs.frames = Some(fv);
                                job.obs
                            });
                            if out.send(msg).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn render thread"),
            );
            tx
        });

        // physics thread
        let groups_c = ranges.clone();
        threads.push(
            std::thread::Builder::new()
                .name("pippin-physics".into())
                .spawn(move || {
                    while let Ok(job) = phys_rx.recv() {
                        let t = Instant::now();
                        let group = match job {
                            PhysJob::Stop => break,
                            PhysJob::SetParam { p, id, envs, values, reply } => {
                                let _ = reply.send(physics.set_param(p, id, envs, &values));
                                continue;
                            }
                            PhysJob::GetParam { p, id, envs, width, reply } => {
                                let mut out = vec![0.0; envs.len() * width];
                                let r = physics.get_param(p, id, envs, &mut out).map(|_| out);
                                let _ = reply.send(r);
                                continue;
                            }
                            PhysJob::Step { group, ctrl } => {
                                let r = groups_c[group].clone();
                                physics.set(Field::Ctrl, r.clone(), &ctrl);
                                physics.step(r, substeps);
                                group
                            }
                            PhysJob::Reset { group } => {
                                let ids: Vec<usize> = groups_c[group].clone().collect();
                                physics.reset(&ids);
                                group
                            }
                        };
                        let envs = groups_c[group].clone();
                        let n = envs.len();
                        let mut obs = Obs {
                            group,
                            envs: envs.clone(),
                            qpos: vec![0.0; n * dims.nq],
                            qvel: vec![0.0; n * dims.nv],
                            frames: None,
                            timing: Timing::default(),
                        };
                        physics.get(Field::Qpos, envs.clone(), &mut obs.qpos);
                        physics.get(Field::Qvel, envs.clone(), &mut obs.qvel);
                        obs.timing.physics_ms = t.elapsed().as_secs_f64() * 1e3;
                        let sent = match &render_tx {
                            Some(tx) => {
                                let mut geoms = vec![Pose::default(); n * dims.ngeom];
                                let mut cams = vec![Pose::default(); n * dims.ncam];
                                let mut appearance = vec![Appearance::default(); n * dims.ngeom];
                                physics.poses(envs.clone(), &mut geoms, &mut cams);
                                physics.appearance(envs, &mut appearance);
                                tx.send(RenderJob { obs, geoms, cams, appearance }).is_ok()
                            }
                            None => out_tx.send(Ok(obs)).is_ok(),
                        };
                        if !sent {
                            break;
                        }
                    }
                })
                .expect("spawn physics thread"),
        );

        AsyncEnv { dims, scene, groups: ranges, phys_tx, out_rx: Mutex::new(out_rx), threads }
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }
    pub fn scene(&self) -> &Scene {
        &self.scene
    }
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }
    pub fn group_envs(&self, group: usize) -> Range<usize> {
        self.groups[group].clone()
    }

    /// Queue `group` to step with `ctrl` shaped (group envs, nu).
    pub fn send(&self, group: usize, ctrl: Vec<f64>) {
        assert_eq!(ctrl.len(), self.groups[group].len() * self.dims.nu, "ctrl has the wrong size for group {group}");
        self.phys_tx.send(PhysJob::Step { group, ctrl }).expect("physics thread stopped");
    }

    /// Queue a reset of every environment in `group`; produces an observation.
    pub fn reset(&self, group: usize) {
        self.phys_tx.send(PhysJob::Reset { group }).expect("physics thread stopped");
    }

    /// Set a model parameter for `envs` (applied in order with queued steps;
    /// blocks until applied). `values` is shaped (envs.len(), width).
    pub fn set_param(&self, p: Param, id: usize, envs: Range<usize>, values: Vec<f64>) -> Result<(), String> {
        let (reply, rx) = channel();
        self.phys_tx.send(PhysJob::SetParam { p, id, envs, values, reply }).map_err(|_| "physics thread stopped")?;
        rx.recv().map_err(|_| "physics thread stopped".to_string())?
    }

    /// Read a model parameter for `envs`, shaped (envs.len(), width).
    pub fn get_param(&self, p: Param, id: usize, envs: Range<usize>, width: usize) -> Result<Vec<f64>, String> {
        let (reply, rx) = channel();
        self.phys_tx.send(PhysJob::GetParam { p, id, envs, width, reply }).map_err(|_| "physics thread stopped")?;
        rx.recv().map_err(|_| "physics thread stopped".to_string())?
    }

    /// Block until the next group finishes.
    pub fn recv(&self) -> Result<Obs, String> {
        let rx = self.out_rx.lock().map_err(|_| "pipeline receiver poisoned".to_string())?;
        rx.recv().map_err(|_| "pipeline stopped".to_string())?
    }
}

impl Drop for AsyncEnv {
    fn drop(&mut self) {
        let _ = self.phys_tx.send(PhysJob::Stop);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
