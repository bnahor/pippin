//! Python bindings: a vectorized simulator handle with numpy state access.

mod async_env;

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pippin::batch::Field;
use pippin::{mjcf, Batch};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "Sim")]
struct PySim {
    batch: Batch,
}

fn field(name: &str) -> PyResult<Field> {
    Ok(match name {
        "qpos" => Field::Qpos,
        "qvel" => Field::Qvel,
        "ctrl" => Field::Ctrl,
        "qfrc_applied" => Field::QfrcApplied,
        _ => return Err(PyKeyError::new_err(format!("unknown field '{name}'"))),
    })
}

#[pymethods]
impl PySim {
    /// Load an MJCF model from a file path or XML string and create `num_envs` copies.
    #[new]
    #[pyo3(signature = (model, num_envs = 1))]
    fn new(model: &str, num_envs: usize) -> PyResult<Self> {
        let m = if model.trim_start().starts_with('<') { mjcf::load_str(model) } else { mjcf::load_file(model) }
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySim { batch: Batch::new(m, num_envs) })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.batch.len()
    }
    #[getter]
    fn nq(&self) -> usize {
        self.batch.model.nq
    }
    #[getter]
    fn nv(&self) -> usize {
        self.batch.model.nv
    }
    #[getter]
    fn nu(&self) -> usize {
        self.batch.model.nu
    }
    #[getter]
    fn timestep(&self) -> f64 {
        self.batch.model.timestep
    }
    #[getter]
    fn joint_names(&self) -> Vec<String> {
        self.batch.model.jnt_names.clone()
    }
    #[getter]
    fn joint_qposadr(&self) -> Vec<usize> {
        self.batch.model.jnt_qposadr.clone()
    }
    #[getter]
    fn joint_dofadr(&self) -> Vec<usize> {
        self.batch.model.jnt_dofadr.clone()
    }
    #[getter]
    fn body_names(&self) -> Vec<String> {
        self.batch.model.body_names.clone()
    }

    /// Advance all environments `nstep` steps (releases the GIL).
    #[pyo3(signature = (nstep = 1))]
    fn step(&mut self, py: Python<'_>, nstep: usize) {
        let b = &mut self.batch;
        py.detach(|| b.step(nstep));
    }

    /// Open-loop rollout of `ctrl` shaped (num_envs, nstep, nu), starting from the
    /// current state. Returns qpos after every step, shaped (num_envs, nstep, nq).
    fn rollout<'py>(&mut self, py: Python<'py>, ctrl: numpy::PyReadonlyArray3<f64>) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        let shape = ctrl.shape().to_vec();
        let (n, nu, nq) = (self.batch.len(), self.batch.model.nu, self.batch.model.nq);
        if shape[0] != n || shape[2] != nu {
            return Err(PyValueError::new_err(format!("expected ctrl shape ({n}, T, {nu}), got {shape:?}")));
        }
        let nstep = shape[1];
        let c = ctrl.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?.to_vec();
        let mut out = vec![0.0; n * nstep * nq];
        let b = &mut self.batch;
        py.detach(|| b.rollout(&c, nstep, &mut out));
        PyArray1::from_vec(py, out).reshape([n, nstep, nq])
    }

    fn forward(&mut self, py: Python<'_>) {
        let b = &mut self.batch;
        py.detach(|| b.forward());
    }

    /// Reset environments (all if `ids` is None).
    #[pyo3(signature = (ids = None))]
    fn reset(&mut self, ids: Option<Vec<usize>>) {
        let ids = ids.unwrap_or_else(|| (0..self.batch.len()).collect());
        self.batch.reset(&ids);
    }

    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let f = field(name)?;
        let w = f.width(&self.batch.model);
        let mut out = vec![0.0; w * self.batch.len()];
        self.batch.get(f, &mut out);
        PyArray1::from_vec(py, out).reshape([self.batch.len(), w])
    }

    fn set(&mut self, name: &str, value: PyReadonlyArray2<f64>) -> PyResult<()> {
        let f = field(name)?;
        let w = f.width(&self.batch.model);
        let shape = value.shape();
        if shape != [self.batch.len(), w] {
            return Err(PyValueError::new_err(format!("expected shape ({}, {w}), got {shape:?}", self.batch.len())));
        }
        let s = value.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.batch.set(f, s);
        Ok(())
    }

    /// World positions of all bodies, shape (num_envs, nbody, 3).
    fn body_pos<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        let nb = self.batch.model.nbody();
        let v: Vec<f64> = self.batch.envs.iter().flat_map(|d| d.xpos.iter().flat_map(|p| p.0)).collect();
        PyArray1::from_vec(py, v).reshape([self.batch.len(), nb, 3])
    }

    /// World orientations (w, x, y, z) of all bodies, shape (num_envs, nbody, 4).
    fn body_quat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        let nb = self.batch.model.nbody();
        let v: Vec<f64> = self.batch.envs.iter().flat_map(|d| d.xquat.iter().flat_map(|q| q.0)).collect();
        PyArray1::from_vec(py, v).reshape([self.batch.len(), nb, 4])
    }

    /// Model-level mass properties (body_mass, body_inertia_com) for validation.
    fn body_mass(&self) -> Vec<f64> {
        self.batch.model.body_mass.clone()
    }

    /// Per-body (translational, rotational) inverse inertia at qpos0.
    fn body_invweight(&self) -> Vec<[f64; 2]> {
        self.batch.model.body_invweight.clone()
    }

    /// Per-dof inverse inertia at qpos0.
    fn dof_invweight(&self) -> Vec<f64> {
        self.batch.model.dof_invweight.clone()
    }

    /// Joint-space inertia of env 0 after forward().
    fn mass_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let nv = self.batch.model.nv;
        PyArray1::from_vec(py, self.batch.envs[0].qm.clone()).reshape([nv, nv])
    }

    /// Bias forces (Coriolis + gravity) of env 0 after forward().
    fn qfrc_bias<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.batch.envs[0].qfrc_bias.clone())
    }

    /// Number of active contacts per environment.
    fn ncon(&self) -> Vec<usize> {
        self.batch.envs.iter().map(|d| d.contacts.len()).collect()
    }
}

fn load_model(model: &str) -> PyResult<pippin::Model> {
    if model.trim_start().starts_with('<') { mjcf::load_str(model) } else { mjcf::load_file(model) }
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert a URDF file to an MJCF string (mesh paths made absolute).
#[pyfunction]
#[pyo3(signature = (path, floating_base = false))]
fn urdf_to_mjcf(path: &str, floating_base: bool) -> PyResult<String> {
    let text = std::fs::read_to_string(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let base = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
    let base = std::fs::canonicalize(base).unwrap_or(base.to_path_buf());
    let opts = pippin::urdf::UrdfOptions { floating_base, ..Default::default() };
    let (xml, warnings) = pippin::urdf::to_mjcf(&text, &base, &opts).map_err(|e| PyValueError::new_err(e.to_string()))?;
    for w in warnings {
        eprintln!("pippin urdf warning: {w}");
    }
    Ok(xml)
}

/// Layout of a model: dimensions and name -> index/address tables.
#[pyfunction]
fn model_info<'py>(py: Python<'py>, model: &str) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let m = load_model(model)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("nq", m.nq)?;
    d.set_item("nv", m.nv)?;
    d.set_item("nu", m.nu)?;
    d.set_item("qpos0", m.qpos0.clone())?;
    d.set_item("joint_names", m.jnt_names.clone())?;
    d.set_item("joint_qposadr", m.jnt_qposadr.clone())?;
    d.set_item("body_names", m.body_names.clone())?;
    d.set_item("geom_names", m.geom_names.clone())?;
    d.set_item("actuator_names", m.actuator_names.clone())?;
    Ok(d)
}

/// Mass properties of a mesh file's convex hull (per unit density):
/// volume, center of mass, inertia about the com, and bounding box.
#[pyfunction]
#[pyo3(signature = (path, scale = (1.0, 1.0, 1.0)))]
fn mesh_properties<'py>(py: Python<'py>, path: &str, scale: (f64, f64, f64)) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut mesh = pippin::mesh::load(std::path::Path::new(path)).map_err(|e| PyValueError::new_err(e.to_string()))?;
    for v in &mut mesh.vertices {
        *v = v.mul_elem(pippin::math::Vec3::new(scale.0, scale.1, scale.2));
    }
    let hull = pippin::mesh::convex_hull(&mesh.vertices);
    let (vol, com, inertia) = pippin::mesh::mass_properties(&hull);
    let lo = hull.vertices.iter().fold(pippin::math::Vec3::new(f64::MAX, f64::MAX, f64::MAX), |a, v| a.min(*v));
    let hi = hull.vertices.iter().fold(pippin::math::Vec3::new(f64::MIN, f64::MIN, f64::MIN), |a, v| a.max(*v));
    let d = pyo3::types::PyDict::new(py);
    d.set_item("volume", vol)?;
    d.set_item("com", com.0.to_vec())?;
    d.set_item("inertia", inertia.0.to_vec())?;
    d.set_item("min", lo.0.to_vec())?;
    d.set_item("max", hi.0.to_vec())?;
    Ok(d)
}

#[pymodule]
fn _pippin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(urdf_to_mjcf, m)?)?;
    m.add_function(wrap_pyfunction!(model_info, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_properties, m)?)?;
    m.add_class::<PySim>()?;
    m.add_class::<async_env::PyAsyncEnv>()?;
    Ok(())
}
