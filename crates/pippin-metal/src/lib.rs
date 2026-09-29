//! Metal backend: steps thousands of environments on the Apple GPU.
//!
//! One GPU thread simulates one environment with the same algorithms as the
//! CPU reference engine (Newton constraint solver), in f32. State lives in
//! shared (unified) memory, so the CPU reads and writes it without copies.

pub mod codegen;

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLMathMode, MTLResourceOptions, MTLSize,
};
use pippin::model::Model;

pub use codegen::Limits;

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device available")]
    NoDevice,
    #[error("Metal shader compilation failed: {0}")]
    Compile(String),
    #[error("Metal error: {0}")]
    Runtime(String),
    #[error("model not supported on the GPU backend: {0}")]
    Unsupported(String),
}

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct MetalSim {
    pub model: Model,
    n: usize,
    limits: Limits,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    qpos: Buffer,
    qvel: Buffer,
    ctrl: Buffer,
    qacc: Buffer,
    group: usize,
    device_name: String,
}

// Metal objects are thread-safe to use from any thread; the handle is only
// ever used through &mut self.
unsafe impl Send for MetalSim {}
unsafe impl Sync for MetalSim {}

fn buffer(device: &ProtocolObject<dyn MTLDevice>, floats: usize) -> Result<Buffer, MetalError> {
    device
        .newBufferWithLength_options(4 * floats.max(1), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::Runtime("buffer allocation failed".into()))
}

fn slice(buf: &Buffer, len: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len) }
}

#[allow(clippy::mut_from_ref)]
fn slice_mut(buf: &Buffer, len: usize) -> &mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut f32, len) }
}

impl MetalSim {
    pub fn new(model: Model, n: usize) -> Result<MetalSim, MetalError> {
        let limits = Limits::for_model(&model);
        Self::with_limits(model, n, limits)
    }

    pub fn with_limits(model: Model, n: usize, limits: Limits) -> Result<MetalSim, MetalError> {
        let src = codegen::source(&model, limits);
        Self::with_source(model, n, limits, &src)
    }

    /// Build from explicit Metal source (for profiling experiments).
    #[doc(hidden)]
    pub fn with_source(model: Model, n: usize, limits: Limits, src: &str) -> Result<MetalSim, MetalError> {
        if model.nv == 0 {
            return Err(MetalError::Unsupported("model has no degrees of freedom".into()));
        }
        let unsupported = [
            (model.actuator_moment.iter().any(|mo| mo.len() != 1), "tendon actuator transmissions"),
            (!model.eq_joint1.is_empty(), "equality constraints"),
            (model.integrator != pippin::model::Integrator::Euler, "the implicitfast integrator"),
        ];
        if let Some((_, what)) = unsupported.iter().find(|(bad, _)| *bad) {
            return Err(MetalError::Unsupported(format!("the Metal backend does not support {what} yet; use a CPU backend")));
        }
        if let Some(&(a, b)) =
            model.collision_pairs.iter().find(|&&(a, b)| pippin::collision::needs_general(model.geom_type[a], model.geom_type[b]))
        {
            return Err(MetalError::Unsupported(format!(
                "geoms '{}' and '{}' need the general convex narrow phase (meshes, cylinders); use a CPU backend",
                model.geom_names[a], model.geom_names[b]
            )));
        }
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
            .map_err(|e| MetalError::Compile(e.localizedDescription().to_string()))?;
        let func = lib
            .newFunctionWithName(&NSString::from_str("step_kernel"))
            .ok_or_else(|| MetalError::Compile("step_kernel not found".into()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| MetalError::Compile(e.localizedDescription().to_string()))?;
        let queue = device.newCommandQueue().ok_or_else(|| MetalError::Runtime("no command queue".into()))?;
        let group = pipeline.threadExecutionWidth().min(pipeline.maxTotalThreadsPerThreadgroup()).max(1);

        let sim = MetalSim {
            qpos: buffer(&device, n * model.nq)?,
            qvel: buffer(&device, n * model.nv)?,
            ctrl: buffer(&device, n * model.nu)?,
            qacc: buffer(&device, n * model.nv)?,
            device_name: device.name().to_string(),
            model,
            n,
            limits,
            queue,
            pipeline,
            group,
        };
        let mut sim = sim;
        sim.reset_all();
        Ok(sim)
    }

    pub fn num_envs(&self) -> usize {
        self.n
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Advance every environment `nstep` steps in a single GPU dispatch
    /// (controls are held constant across those steps).
    pub fn step(&mut self, nstep: usize) -> Result<(), MetalError> {
        if nstep == 0 || self.n == 0 {
            return Ok(());
        }
        let cb = self.queue.commandBuffer().ok_or_else(|| MetalError::Runtime("no command buffer".into()))?;
        let enc = cb.computeCommandEncoder().ok_or_else(|| MetalError::Runtime("no encoder".into()))?;
        enc.setComputePipelineState(&self.pipeline);
        let nenv = self.n as u32;
        let steps = nstep as u32;
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.qpos), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.qvel), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.ctrl), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.qacc), 0, 3);
            enc.setBytes_length_atIndex(NonNull::from(&nenv).cast::<c_void>(), 4, 4);
            enc.setBytes_length_atIndex(NonNull::from(&steps).cast::<c_void>(), 4, 5);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: self.n, height: 1, depth: 1 },
            MTLSize { width: self.group, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            let msg = cb.error().map(|e| e.localizedDescription().to_string()).unwrap_or_default();
            return Err(MetalError::Runtime(msg));
        }
        Ok(())
    }

    pub fn qpos(&self) -> &[f32] {
        slice(&self.qpos, self.n * self.model.nq)
    }
    pub fn qpos_mut(&mut self) -> &mut [f32] {
        slice_mut(&self.qpos, self.n * self.model.nq)
    }
    pub fn qvel(&self) -> &[f32] {
        slice(&self.qvel, self.n * self.model.nv)
    }
    pub fn qvel_mut(&mut self) -> &mut [f32] {
        slice_mut(&self.qvel, self.n * self.model.nv)
    }
    pub fn ctrl(&self) -> &[f32] {
        slice(&self.ctrl, self.n * self.model.nu)
    }
    pub fn ctrl_mut(&mut self) -> &mut [f32] {
        slice_mut(&self.ctrl, self.n * self.model.nu)
    }

    /// Reset the given environments to the reference configuration.
    pub fn reset(&mut self, ids: &[usize]) {
        let (nq, nv, nu) = (self.model.nq, self.model.nv, self.model.nu);
        let q0: Vec<f32> = self.model.qpos0.iter().map(|&x| x as f32).collect();
        let qpos = slice_mut(&self.qpos, self.n * nq);
        let qvel = slice_mut(&self.qvel, self.n * nv);
        let qacc = slice_mut(&self.qacc, self.n * nv);
        let ctrl = slice_mut(&self.ctrl, self.n * nu);
        for &i in ids {
            qpos[i * nq..(i + 1) * nq].copy_from_slice(&q0);
            qvel[i * nv..(i + 1) * nv].fill(0.0);
            qacc[i * nv..(i + 1) * nv].fill(0.0);
            ctrl[i * nu..(i + 1) * nu].fill(0.0);
        }
    }

    pub fn reset_all(&mut self) {
        let ids: Vec<usize> = (0..self.n).collect();
        self.reset(&ids);
    }
}
