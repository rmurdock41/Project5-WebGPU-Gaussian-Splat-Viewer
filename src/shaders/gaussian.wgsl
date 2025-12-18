struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    uv_size: array<u32, 2>,
    color_opacity: array<u32, 2>,
    conic_radius: array<u32, 2>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) center_ndc: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) conic: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;

@group(1) @binding(1)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    let index = sort_indices[instance_index];
    let quad_positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );
    let quad_coord = quad_positions[vertex_index];
    
    let splat = splats[index];
    
    let ndc_pos = unpack2x16float(splat.uv_size[0]);
    let quad_size = unpack2x16float(splat.uv_size[1]);
    let color_rg = unpack2x16float(splat.color_opacity[0]);
    let color_b_opacity = unpack2x16float(splat.color_opacity[1]);
    let color = vec3<f32>(color_rg.x, color_rg.y, color_b_opacity.x);
    let opacity = color_b_opacity.y;
    
    let conic_ab = unpack2x16float(splat.conic_radius[0]);
    let conic_c_radius = unpack2x16float(splat.conic_radius[1]);
    let conic = vec3<f32>(conic_ab.x, conic_ab.y, conic_c_radius.x);
    
    let vertex_ndc = ndc_pos + quad_coord * quad_size;
    
    var out: VertexOutput;
    out.position = vec4<f32>(vertex_ndc, 0.0, 1.0);
    out.center_ndc = ndc_pos.xy;
    out.color = color;
    out.opacity = opacity;
    out.conic = conic;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var ndc = (in.position.xy / camera.viewport) * 2.0 - 1.0;
    ndc.y *= -1.0;
    var d = (ndc - in.center_ndc) * camera.viewport * 0.5;
    d.x = -d.x;
    let conic = in.conic;
    
    let power = -0.5 * (conic.x * d.x * d.x +
                        2.0 * conic.y * d.x * d.y +
                        conic.z * d.y * d.y);
    if (power > 0.0) {
        discard;
    }
    
    let alpha = clamp(in.opacity * exp(power), 0.0, 0.99);
    
    return vec4<f32>(in.color * alpha, alpha);
}