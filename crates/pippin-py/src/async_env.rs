//! Python binding for the pipelined AsyncEnv.

use numpy::ndarray::{ArrayView, IxDyn};
use numpy::prelude::*;
use numpy::{PyArray, PyArray1, PyReadonlyArray2};
use pippin_env::{AsyncEnv, Physics, PippinCpu, RenderConfig, Renderer, ViewSource};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass(name = "AsyncEnv")]
pub struct PyAsyncEnv {
    env: AsyncEnv,
    /// (views, height, width) when rendering.
    image: Option<(usize, usize, usize)>,
}

fn physics(model: &str, backend: &str, n: usize) -> PyResult<Box<dyn Physics>> {
    Ok(match backend {
        "pippin" => Box::new(PippinCpu::from_file(model, n).map_err(|e| PyValueError::new_err(e.to_string()))?),
        "mujoco" => Box::new(pippin_mujoco::MujocoPhysics::from_file(model, n).map_err(|e| PyValueError::new_err(e.to_string()))?),
        other => return Err(PyValueError::new_err(format!("unknown backend '{other}' (use 'pippin' or 'mujoco')"))),
    })
}

fn f3(v: &Bound<'_, PyAny>) -> PyResult<[f32; 3]> {
    let v: Vec<f32> = v.extract()?;
    v.try_into().map_err(|_| PyValueError::new_err("expected 3 numbers"))
}

fn parse_views(views: &Bound<'_, PyList>, cameras: &[String]) -> PyResult<Vec<ViewSource>> {
    views
        .iter()
        .map(|v| {
            let d = v.cast::<PyDict>()?;
            if let Some(name) = d.get_item("camera")? {
                let name: String = name.extract()?;
                let idx = cameras
                    .iter()
                    .position(|c| *c == name)
                    .ok_or_else(|| PyValueError::new_err(format!("no camera named '{name}' (have {cameras:?})")))?;
                return Ok(ViewSource::Scene(idx));
            }
            let eye = f3(&d.get_item("eye")?.ok_or_else(|| PyValueError::new_err("view needs 'camera' or 'eye'"))?)?;
            let target = match d.get_item("target")? {
                Some(t) => f3(&t)?,
                None => [0.0; 3],
            };
            let fovy: f32 = match d.get_item("fovy")? {
                Some(f) => f.extract()?,
                None => 45.0,
            };
            Ok(ViewSource::look_at(eye, target, fovy))
        })
        .collect()
}

/// numpy view of GPU-shared memory, kept alive by `owner`.
unsafe fn view<'py, T: numpy::Element>(ptr: *const T, shape: Vec<usize>, owner: Bound<'py, PyAny>) -> Bound<'py, PyArray<T, IxDyn>> {
    let a = ArrayView::from_shape_ptr(IxDyn(&shape), ptr);
    PyArray::borrow_from_array(&a, owner)
}

#[pymethods]
impl PyAsyncEnv {
    /// Pipelined vector env: physics on CPU cores, rendering on the GPU.
    ///
    /// render: None or dict(width=64, height=64, views=[dict(eye=[x,y,z],
    /// target=[x,y,z], fovy=45) | dict(camera="name")], near=0.01, far=20).
    #[new]
    #[pyo3(signature = (model, num_envs, backend = "pippin", groups = 2, substeps = 1, render = None))]
    fn new(model: &str, num_envs: usize, backend: &str, groups: usize, substeps: usize, render: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let phys = physics(model, backend, num_envs)?;
        let mut image = None;
        let renderer: Option<Box<dyn Renderer>> = match render {
            None => None,
            Some(r) => {
                let get = |k: &str, default: f64| -> PyResult<f64> { Ok(r.get_item(k)?.map(|v| v.extract()).transpose()?.unwrap_or(default)) };
                let width = get("width", 64.0)? as usize;
                let height = get("height", 64.0)? as usize;
                let cameras: Vec<String> = phys.scene().cameras.iter().map(|c| c.name.clone()).collect();
                let views = match r.get_item("views")? {
                    Some(v) => parse_views(v.cast::<PyList>()?, &cameras)?,
                    None => vec![ViewSource::look_at([1.5, -1.5, 1.2], [0.0, 0.0, 0.3], 45.0)],
                };
                image = Some((views.len(), height, width));
                let cfg = RenderConfig { width, height, views, near: get("near", 0.01)? as f32, far: get("far", 20.0)? as f32 };
                Some(Box::new(pippin_render::MetalRenderer::new(&phys.scene(), cfg).map_err(PyRuntimeError::new_err)?))
            }
        };
        Ok(PyAsyncEnv { env: AsyncEnv::new(phys, renderer, groups, substeps.max(1)), image })
    }

    #[getter]
    fn num_groups(&self) -> usize {
        self.env.num_groups()
    }
    #[getter]
    fn nq(&self) -> usize {
        self.env.dims().nq
    }
    #[getter]
    fn nv(&self) -> usize {
        self.env.dims().nv
    }
    #[getter]
    fn nu(&self) -> usize {
        self.env.dims().nu
    }

    /// (start, stop) env indices of a group.
    fn group_envs(&self, group: usize) -> (usize, usize) {
        let r = self.env.group_envs(group);
        (r.start, r.end)
    }

    /// Queue a reset of every environment in `group` (all groups if None).
    #[pyo3(signature = (group = None))]
    fn reset(&self, group: Option<usize>) {
        match group {
            Some(g) => self.env.reset(g),
            None => (0..self.env.num_groups()).for_each(|g| self.env.reset(g)),
        }
    }

    /// Queue a step of `group` with actions shaped (group size, nu).
    fn send(&self, group: usize, actions: PyReadonlyArray2<f64>) -> PyResult<()> {
        if group >= self.env.num_groups() {
            return Err(PyValueError::new_err("group out of range"));
        }
        let n = self.env.group_envs(group).len();
        let nu = self.env.dims().nu;
        if actions.shape() != [n, nu] {
            return Err(PyValueError::new_err(format!("expected actions shape ({n}, {nu}), got {:?}", actions.shape())));
        }
        self.env.send(group, actions.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?.to_vec());
        Ok(())
    }

    /// Wait for the next finished group. Images are zero-copy views that stay
    /// valid until this group is sent again.
    fn recv<'py>(slf: Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let obs = {
            let me = slf.borrow();
            let env = &me.env;
            py.detach(|| env.recv()).map_err(PyRuntimeError::new_err)?
        };
        let me = slf.borrow();
        let dims = me.env.dims();
        let n = obs.envs.len();
        let out = PyDict::new(py);
        out.set_item("group", obs.group)?;
        out.set_item("env_ids", (obs.envs.start, obs.envs.end))?;
        out.set_item("qpos", PyArray1::from_vec(py, obs.qpos).reshape([n, dims.nq])?)?;
        out.set_item("qvel", PyArray1::from_vec(py, obs.qvel).reshape([n, dims.nv])?)?;
        out.set_item("physics_ms", obs.timing.physics_ms)?;
        out.set_item("render_ms", obs.timing.render_ms)?;
        if let (Some(f), Some((v, h, w))) = (obs.frames, me.image) {
            let owner = slf.clone().into_any();
            unsafe {
                out.set_item("rgb", view(f.rgba, vec![n, v, h, w, 4], owner.clone()))?;
                out.set_item("depth", view(f.depth, vec![n, v, h, w], owner.clone()))?;
                out.set_item("segmentation", view(f.segmentation, vec![n, v, h, w], owner))?;
            }
        }
        Ok(out)
    }
}
