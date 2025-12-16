struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    xy: vec2<f32>,        // 2D center in NDC
    conic: vec3<f32>,     // conic matrix (a, b, c)
    color: vec3<f32>,     // RGB color
    opacity: f32,         // opacity
};

@vertex
fn vs_main(
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 

    var out: VertexOutput;
    out.position = vec4<f32>(1. ,1. , 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.);
}