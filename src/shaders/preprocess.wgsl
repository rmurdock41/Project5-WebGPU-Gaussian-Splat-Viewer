const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    uv_size: array<u32, 2>,
    color_opacity: array<u32, 2>,
    conic_radius: array<u32, 2>
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<uniform> render_settings: RenderSettings;
@group(1) @binding(2)
var<storage, read> sh_coeffs: array<u32>;
@group(1) @binding(3)
var<storage, read_write> splats: array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let idx = splat_idx * 24 + (c_idx / 2) * 3 + (c_idx % 2);
    let sha = unpack2x16float(sh_coeffs[idx]);
    let shb = unpack2x16float(sh_coeffs[idx + 1]);
    if ((c_idx % 2) == 0) {
        return vec3<f32>(sha.x, sha.y, shb.x);
    } else {
        return vec3<f32>(sha.y, shb.x, shb.y);
    }
}

fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {
            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return max(vec3<f32>(0.), result);
}

fn quat_to_mat3(q_in: vec4<f32>) -> mat3x3<f32> {
    let q_norm = normalize(q_in);
    let r = q_norm.x;
    let x = q_norm.y;
    let y = q_norm.z;
    let z = q_norm.w;
    return mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    );
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= arrayLength(&gaussians)) {
        return;
    }
    
    let gaussian = gaussians[idx];
    let pa = unpack2x16float(gaussian.pos_opacity[0]);
    let pb = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4<f32>(pa.x, pa.y, pb.x, 1.0);
    let opacity = 1.0 / (1.0 + exp(-pb.y));
    
    let view_pos = camera.view * pos;
    var ndc_pos = camera.proj * view_pos;
    ndc_pos = ndc_pos / ndc_pos.w;

    if (view_pos.z <= 0.0 || abs(ndc_pos.x) > 1.2 || abs(ndc_pos.y) > 1.2) {
        return;
    }
    
    let ra = unpack2x16float(gaussian.rot[0]);
    let rb = unpack2x16float(gaussian.rot[1]);
    let rot_quat = vec4<f32>(ra.x, ra.y, rb.x, rb.y);
    let R = quat_to_mat3(rot_quat);
    
    let sa = unpack2x16float(gaussian.scale[0]);
    let sb = unpack2x16float(gaussian.scale[1]);
    let scale = vec3<f32>(exp(sa.x), exp(sa.y), exp(sb.x)) * render_settings.gaussian_scaling;
    let S = mat3x3<f32>(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    
    let cov3d = transpose(S * R) * (S * R);
    
    let t = view_pos.xyz;
    let J = mat3x3<f32>(
        camera.focal.x / t.z, 0.0, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );
    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;
    
    let Vrk = mat3x3<f32>(
        cov3d[0][0], cov3d[0][1], cov3d[0][2],
        cov3d[0][1], cov3d[1][1], cov3d[1][2],
        cov3d[0][2], cov3d[1][2], cov3d[2][2]
    );
    
    var cov2d_mat = transpose(T) * Vrk * T;
    cov2d_mat[0][0] += 0.3;
    cov2d_mat[1][1] += 0.3;
    let a = cov2d_mat[0][0];
    let b = cov2d_mat[0][1];
    let c = cov2d_mat[1][1];
    
    let det = (a * c - b * b);
    if (det <= 0.0) {
        return;
    }
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(c * det_inv, -b * det_inv, a * det_inv);
    
    let mid = 0.5 * (a + c);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    let quad_size = vec2<f32>(radius, radius) / camera.viewport;
    
    let splat_idx = atomicAdd(&sort_infos.keys_size, 1u);
    let dir = normalize(pos.xyz - camera.view_inv[3].xyz);
    let splat_color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));
    
    splats[splat_idx].uv_size[0] = pack2x16float(ndc_pos.xy);
    splats[splat_idx].uv_size[1] = pack2x16float(quad_size);
    splats[splat_idx].color_opacity[0] = pack2x16float(vec2<f32>(splat_color.x, splat_color.y));
    splats[splat_idx].color_opacity[1] = pack2x16float(vec2<f32>(splat_color.z, opacity));
    splats[splat_idx].conic_radius[0] = pack2x16float(vec2<f32>(conic.x, conic.y));
    splats[splat_idx].conic_radius[1] = pack2x16float(vec2<f32>(conic.z, radius));
    
    let sort_key = 0xFFFFFFFFu - bitcast<u32>(-view_pos.z);
    sort_depths[splat_idx] = sort_key;
    sort_indices[splat_idx] = splat_idx;
    
    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    if (splat_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}