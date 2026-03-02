struct Params {
	anti : u32, // 0 | 1; booleans not supported
	resolution : vec2<u32>,
	samples_per_thread : u32,
	max_iterations : u32,
	max_iterations_2 : u32,
	max_iterations_3 : u32,
	seed : u32,
	sample_center : vec2<f32>,
	sample_radius : f32,
	sample_uniform_distribution : u32, // 0 | 1
	view_center : vec2<f32>,
	initial_z : vec2<f32>,
	exponent : vec2<f32>,
	rotation : f32,
	view_y_span : f32,
	view_aspect_ratio : f32,
	escape_radius_sq : f32,
	gamma : f32,
	histogram_lerp : f32,
	z_indicator_size : f32,
	e_indicator_size : f32,
};

@group(0) @binding(0) var<storage, read> histogram : array<u32>;
@group(0) @binding(1) var<storage, read_write> lerped : array<f32>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
	let pixels = params.resolution.x * params.resolution.y;
	let count = pixels * 3u + 3u;
	let index = gid.x;
	if (index >= count) {
		return;
	}

	let t = clamp(params.histogram_lerp, 0.0, 1.0);
	let curr = f32(histogram[index]);
	let prev = lerped[index];
	lerped[index] = mix(prev, curr, t);
}
