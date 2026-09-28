// Batched rasterizer: one instance per (environment, view), each rendered
// into its own slice of layered render targets.

#include <metal_stdlib>
using namespace metal;

struct Vtx {
    float pos[3];
    uint geom;
    float normal[3];
    uint flags; // bit 0: checker pattern
};

struct Pose {
    float pos[3];
    float mat[9]; // row-major
};

struct View {
    int scene_cam; // >= 0: index into scene cameras; -1: fixed pose below
    float fovy;    // degrees; <= 0 means "use the scene camera's fovy"
    float pad[2];
    Pose fixed;
};

struct Uniforms {
    uint ngeom;
    uint ncam;
    uint nview;
    uint first_image; // image index of instance 0 in this pass
    float near;
    float far;
    float aspect;
    float pad;
};

struct VOut {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    float3 normal_cam;
    float depth;
    float4 color [[flat]];
    int seg [[flat]];
    uint flags [[flat]];
    float2 local_xy;
};

struct FOut {
    float4 color [[color(0)]];
    float depth [[color(1)]];
    int seg [[color(2)]];
};

inline float3x3 rot(Pose p) {
    // rows -> Metal column-major constructor takes columns
    return float3x3(float3(p.mat[0], p.mat[3], p.mat[6]),
                    float3(p.mat[1], p.mat[4], p.mat[7]),
                    float3(p.mat[2], p.mat[5], p.mat[8]));
}

vertex VOut vs(uint vid [[vertex_id]],
               uint iid [[instance_id]],
               device const Vtx* verts [[buffer(0)]],
               device const Pose* geoms [[buffer(1)]],
               device const Pose* cams [[buffer(2)]],
               constant View* views [[buffer(3)]],
               constant Uniforms& u [[buffer(4)]],
               device const float4* colors [[buffer(5)]],
               constant float* cam_fovy [[buffer(6)]]) {
    uint image = u.first_image + iid;
    uint env = image / u.nview;
    uint view = image % u.nview;
    Vtx v = verts[vid];
    Pose g = geoms[env * u.ngeom + v.geom];
    float3x3 R = rot(g);
    float3 world = R * float3(v.pos[0], v.pos[1], v.pos[2]) + float3(g.pos[0], g.pos[1], g.pos[2]);
    float3 nworld = R * float3(v.normal[0], v.normal[1], v.normal[2]);

    View vw = views[view];
    Pose c = vw.scene_cam >= 0 ? cams[env * u.ncam + vw.scene_cam] : vw.fixed;
    float fovy = vw.fovy > 0 ? vw.fovy : cam_fovy[max(vw.scene_cam, 0)];
    float3x3 C = rot(c);
    float3 pc = transpose(C) * (world - float3(c.pos[0], c.pos[1], c.pos[2]));

    float f = 1.0f / tan(0.5f * fovy * M_PI_F / 180.0f);
    float A = u.far / (u.near - u.far);
    VOut o;
    o.position = float4(f / u.aspect * pc.x, f * pc.y, A * pc.z + A * u.near, -pc.z);
    o.layer = iid;
    o.normal_cam = transpose(C) * nworld;
    o.depth = -pc.z;
    o.color = colors[v.geom];
    o.seg = int(v.geom);
    o.flags = v.flags;
    o.local_xy = float2(v.pos[0], v.pos[1]);
    return o;
}

fragment FOut fs(VOut in [[stage_in]], bool front [[front_facing]]) {
    float3 n = normalize(in.normal_cam);
    // headlight plus a fixed key light from above-left, in camera space
    float head = abs(n.z);
    float key = max(dot(n, normalize(float3(-0.4f, 0.8f, 0.5f))), 0.0f);
    float shade = 0.25f + 0.5f * head + 0.35f * key;
    if (in.flags & 1u) {
        // 0.25 m checkerboard, in the plane's local frame
        int2 cell = int2(floor(in.local_xy / 0.25f));
        shade *= ((cell.x + cell.y) & 1) ? 0.8f : 1.05f;
    }
    FOut o;
    o.color = float4(saturate(in.color.rgb * shade), 1.0f);
    o.depth = in.depth;
    o.seg = in.seg;
    return o;
}

// Copy layered textures into linear buffers laid out (image, y, x).
kernel void readback(texture2d_array<float, access::read> color [[texture(0)]],
                     texture2d_array<float, access::read> depth [[texture(1)]],
                     texture2d_array<int, access::read> seg [[texture(2)]],
                     device uchar4* out_color [[buffer(0)]],
                     device float* out_depth [[buffer(1)]],
                     device int* out_seg [[buffer(2)]],
                     constant uint& first_image [[buffer(3)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint w = color.get_width(), h = color.get_height();
    if (gid.x >= w || gid.y >= h) return;
    uint idx = ((first_image + gid.z) * h + gid.y) * w + gid.x;
    float4 c = color.read(gid.xy, gid.z);
    out_color[idx] = uchar4(uchar(c.r * 255.0f + 0.5f), uchar(c.g * 255.0f + 0.5f), uchar(c.b * 255.0f + 0.5f), 255);
    out_depth[idx] = depth.read(gid.xy, gid.z).r;
    out_seg[idx] = seg.read(gid.xy, gid.z).r;
}
