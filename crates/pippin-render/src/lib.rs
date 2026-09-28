//! Batched Metal rasterizer.
//!
//! Every geom of the scene lives in one vertex buffer; a frame is a single
//! instanced draw with one instance per (environment, view), each writing its
//! own slice of layered render targets. Images are then copied into shared
//! (unified-memory) buffers: RGBA8, linear depth, and geom-id segmentation.

pub mod mesh;

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use pippin_env::{Appearance, Frames, Pose, RenderConfig, Renderer, Scene, ViewSource};

/// Metal's limit on render target array slices.
const MAX_LAYERS: usize = 2048;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Texture = Retained<ProtocolObject<dyn MTLTexture>>;

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuView {
    scene_cam: i32,
    fovy: f32,
    pad: [f32; 2],
    fixed: Pose,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Uniforms {
    ngeom: u32,
    ncam: u32,
    nview: u32,
    first_image: u32,
    near: f32,
    far: f32,
    aspect: f32,
    pad: f32,
}

pub struct MetalRenderer {
    config: RenderConfig,
    ngeom: usize,
    ncam: usize,
    nindex: usize,
    indices: Buffer,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    readback: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    depth_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    verts: Buffer,
    /// Scene default appearance per geom, used when a frame supplies none.
    base_appearance: Vec<Appearance>,
    views: Buffer,
    cam_fovy: Buffer,
    // grow on demand
    geom_poses: Option<Buffer>,
    cam_poses: Option<Buffer>,
    appearance: Option<Buffer>,
    targets: Option<(usize, [Texture; 4])>,
    /// Output buffers per slot: (images capacity, [rgba, depth, seg]).
    out: Vec<Option<(usize, [Buffer; 3])>>,
}

unsafe impl Send for MetalRenderer {}

fn err(e: Retained<objc2_foundation::NSError>) -> String {
    e.localizedDescription().to_string()
}

fn new_buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Result<Buffer, String> {
    device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| "buffer allocation failed".to_string())
}

fn buffer_from<T: Copy>(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> Result<Buffer, String> {
    let bytes = std::mem::size_of_val(data);
    let buf = new_buffer(device, bytes)?;
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents().as_ptr() as *mut u8, bytes) };
    Ok(buf)
}

fn write<T: Copy>(buf: &Buffer, data: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents().as_ptr() as *mut u8, std::mem::size_of_val(data))
    };
}

impl MetalRenderer {
    pub fn new(scene: &Scene, config: RenderConfig) -> Result<MetalRenderer, String> {
        if config.views.is_empty() {
            return Err("RenderConfig needs at least one view".into());
        }
        for v in &config.views {
            if let ViewSource::Scene(c) = v {
                if *c >= scene.cameras.len() {
                    return Err(format!("view refers to scene camera {c}, scene has {}", scene.cameras.len()));
                }
            }
        }
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let opts = MTLCompileOptions::new();
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(include_str!("render.metal")), Some(&opts))
            .map_err(err)?;
        let func = |name: &str| lib.newFunctionWithName(&NSString::from_str(name)).ok_or(format!("missing {name}"));

        let desc = MTLRenderPipelineDescriptor::new();
        let (vs, fs) = (func("vs")?, func("fs")?);
        desc.setVertexFunction(Some(&vs));
        desc.setFragmentFunction(Some(&fs));
        unsafe {
            let ca = desc.colorAttachments();
            ca.objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::RGBA8Unorm);
            ca.objectAtIndexedSubscript(1).setPixelFormat(MTLPixelFormat::R32Float);
            ca.objectAtIndexedSubscript(2).setPixelFormat(MTLPixelFormat::R32Sint);
            desc.setInputPrimitiveTopology(MTLPrimitiveTopologyClass::Triangle);
        }
        desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
        let pipeline = device.newRenderPipelineStateWithDescriptor_error(&desc).map_err(err)?;
        let rb = func("readback")?;
        let readback = device.newComputePipelineStateWithFunction_error(&rb).map_err(err)?;

        let ds = MTLDepthStencilDescriptor::new();
        ds.setDepthCompareFunction(MTLCompareFunction::Less);
        ds.setDepthWriteEnabled(true);
        let depth_state = device.newDepthStencilStateWithDescriptor(&ds).ok_or("depth state")?;

        let mesh = mesh::build_indexed(scene);
        let base_appearance: Vec<Appearance> =
            scene.geoms.iter().map(|g| Appearance { rgba: g.rgba, scale: [1.0; 3], pad: 0.0 }).collect();
        let views: Vec<GpuView> = config
            .views
            .iter()
            .map(|v| match *v {
                ViewSource::Scene(c) => GpuView { scene_cam: c as i32, fovy: 0.0, pad: [0.0; 2], fixed: Pose::default() },
                ViewSource::Fixed { pose, fovy } => GpuView { scene_cam: -1, fovy, pad: [0.0; 2], fixed: pose },
            })
            .collect();
        let fovy: Vec<f32> = scene.cameras.iter().map(|c| c.fovy).collect();

        Ok(MetalRenderer {
            ngeom: scene.geoms.len(),
            ncam: scene.cameras.len(),
            nindex: mesh.indices.len(),
            indices: buffer_from(&device, &mesh.indices)?,
            verts: buffer_from(&device, &mesh.vertices)?,
            base_appearance,
            views: buffer_from(&device, &views)?,
            cam_fovy: buffer_from(&device, if fovy.is_empty() { &[45.0f32][..] } else { &fovy })?,
            queue: device.newCommandQueue().ok_or("no queue")?,
            pipeline,
            readback,
            depth_state,
            geom_poses: None,
            cam_poses: None,
            appearance: None,
            targets: None,
            out: vec![],
            config,
            device,
        })
    }

    fn ensure_capacity(&mut self, slot: usize, nenv: usize) -> Result<(), String> {
        let nimg = nenv * self.config.views.len();
        let pose = std::mem::size_of::<Pose>();
        if self.geom_poses.as_ref().is_none_or(|b| b.length() < nenv * self.ngeom * pose) {
            self.geom_poses = Some(new_buffer(&self.device, nenv * self.ngeom * pose)?);
            self.cam_poses = Some(new_buffer(&self.device, nenv * self.ncam.max(1) * pose)?);
            self.appearance = Some(new_buffer(&self.device, nenv * self.ngeom * std::mem::size_of::<Appearance>())?);
        }
        let layers = nimg.min(MAX_LAYERS);
        if self.targets.as_ref().is_none_or(|(n, _)| *n < layers) {
            let tex = |fmt: MTLPixelFormat, readable: bool| -> Result<Texture, String> {
                let d = MTLTextureDescriptor::new();
                d.setTextureType(MTLTextureType::Type2DArray);
                d.setPixelFormat(fmt);
                unsafe {
                    d.setWidth(self.config.width);
                    d.setHeight(self.config.height);
                    d.setArrayLength(layers);
                }
                d.setStorageMode(MTLStorageMode::Private);
                d.setUsage(if readable {
                    MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead
                } else {
                    MTLTextureUsage::RenderTarget
                });
                self.device.newTextureWithDescriptor(&d).ok_or_else(|| "texture allocation failed".to_string())
            };
            self.targets = Some((
                layers,
                [
                    tex(MTLPixelFormat::RGBA8Unorm, true)?,
                    tex(MTLPixelFormat::R32Float, true)?,
                    tex(MTLPixelFormat::R32Sint, true)?,
                    tex(MTLPixelFormat::Depth32Float, false)?,
                ],
            ));
        }
        if self.out.len() <= slot {
            self.out.resize_with(slot + 1, || None);
        }
        if self.out[slot].as_ref().is_none_or(|(n, _)| *n < nimg) {
            let px = nimg * self.config.width * self.config.height;
            let b = || new_buffer(&self.device, px * 4);
            self.out[slot] = Some((nimg, [b()?, b()?, b()?]));
        }
        Ok(())
    }
}

impl Renderer for MetalRenderer {
    fn name(&self) -> &str {
        "metal-raster"
    }

    fn config(&self) -> &RenderConfig {
        &self.config
    }

    fn render(
        &mut self,
        slot: usize,
        nenv: usize,
        geoms: &[Pose],
        cams: &[Pose],
        appearance: Option<&[Appearance]>,
    ) -> Result<Frames<'_>, String> {
        if geoms.len() != nenv * self.ngeom || cams.len() != nenv * self.ncam {
            return Err(format!("expected {} geom and {} camera poses", nenv * self.ngeom, nenv * self.ncam));
        }
        if appearance.is_some_and(|a| a.len() != nenv * self.ngeom) {
            return Err(format!("expected {} appearance entries", nenv * self.ngeom));
        }
        self.ensure_capacity(slot, nenv)?;
        let gp = self.geom_poses.as_ref().unwrap();
        let cp = self.cam_poses.as_ref().unwrap();
        write(gp, geoms);
        write(cp, cams);
        let ap = self.appearance.as_ref().unwrap();
        match appearance {
            Some(a) => write(ap, a),
            None => {
                let defaults: Vec<Appearance> = (0..nenv).flat_map(|_| self.base_appearance.iter().copied()).collect();
                write(ap, &defaults);
            }
        }

        let (w, h) = (self.config.width, self.config.height);
        let nview = self.config.views.len();
        let nimg = nenv * nview;
        let (layers, targets) = self.targets.as_ref().unwrap();
        let (_, outs) = self.out[slot].as_ref().unwrap();
        let cb = self.queue.commandBuffer().ok_or("no command buffer")?;

        let mut first = 0;
        while first < nimg {
            let count = (nimg - first).min(*layers);
            let rp = MTLRenderPassDescriptor::new();
            unsafe {
                let ca = rp.colorAttachments();
                let clears = [
                    MTLClearColor { red: 0.72, green: 0.78, blue: 0.86, alpha: 1.0 },
                    MTLClearColor { red: self.config.far as f64, green: 0.0, blue: 0.0, alpha: 0.0 },
                    MTLClearColor { red: -1.0, green: 0.0, blue: 0.0, alpha: 0.0 },
                ];
                for (i, clear) in clears.into_iter().enumerate() {
                    let a = ca.objectAtIndexedSubscript(i);
                    a.setTexture(Some(&targets[i]));
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearColor(clear);
                    a.setStoreAction(MTLStoreAction::Store);
                }
            }
            let da = rp.depthAttachment();
            da.setTexture(Some(&targets[3]));
            da.setLoadAction(MTLLoadAction::Clear);
            da.setClearDepth(1.0);
            da.setStoreAction(MTLStoreAction::DontCare);
            rp.setRenderTargetArrayLength(count);

            let enc = cb.renderCommandEncoderWithDescriptor(&rp).ok_or("no render encoder")?;
            enc.setRenderPipelineState(&self.pipeline);
            enc.setDepthStencilState(Some(&self.depth_state));
            enc.setCullMode(MTLCullMode::None);
            let u = Uniforms {
                ngeom: self.ngeom as u32,
                ncam: self.ncam as u32,
                nview: nview as u32,
                first_image: first as u32,
                near: self.config.near,
                far: self.config.far,
                aspect: w as f32 / h as f32,
                pad: 0.0,
            };
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(&self.verts), 0, 0);
                enc.setVertexBuffer_offset_atIndex(Some(gp), 0, 1);
                enc.setVertexBuffer_offset_atIndex(Some(cp), 0, 2);
                enc.setVertexBuffer_offset_atIndex(Some(&self.views), 0, 3);
                enc.setVertexBytes_length_atIndex(NonNull::from(&u).cast::<c_void>(), std::mem::size_of::<Uniforms>(), 4);
                enc.setVertexBuffer_offset_atIndex(Some(ap), 0, 5);
                enc.setVertexBuffer_offset_atIndex(Some(&self.cam_fovy), 0, 6);
                if self.nindex > 0 {
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(
                        MTLPrimitiveType::Triangle,
                        self.nindex,
                        MTLIndexType::UInt32,
                        &self.indices,
                        0,
                        count,
                    );
                }
            }
            enc.endEncoding();

            let ce = cb.computeCommandEncoder().ok_or("no compute encoder")?;
            ce.setComputePipelineState(&self.readback);
            let first_u = first as u32;
            unsafe {
                for i in 0..3 {
                    ce.setTexture_atIndex(Some(&targets[i]), i);
                    ce.setBuffer_offset_atIndex(Some(&outs[i]), 0, i);
                }
                ce.setBytes_length_atIndex(NonNull::from(&first_u).cast::<c_void>(), 4, 3);
            }
            ce.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: w, height: h, depth: count },
                MTLSize { width: 16, height: 16, depth: 1 },
            );
            ce.endEncoding();
            first += count;
        }
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() == MTLCommandBufferStatus::Error {
            return Err(cb.error().map(err).unwrap_or_default());
        }
        let px = nimg * w * h;
        unsafe {
            Ok(Frames {
                rgba: std::slice::from_raw_parts(outs[0].contents().as_ptr() as *const u8, px * 4),
                depth: std::slice::from_raw_parts(outs[1].contents().as_ptr() as *const f32, px),
                segmentation: std::slice::from_raw_parts(outs[2].contents().as_ptr() as *const i32, px),
            })
        }
    }
}
