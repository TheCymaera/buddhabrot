
struct Params {
	anti : u32, // 0 | 1; booleans not supported
	resolution : vec2<u32>,
	samples_per_thread : u32,
	max_iterations_1 : u32,
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
	frame_lerp : f32,
	z_indicator_size : f32,
	e_indicator_size : f32,
	normalization_floor : f32,
};

@group(0) @binding(0) var<storage, read> histogram : array<f32>;
@group(0) @binding(1) var<uniform> params : Params;



const z_color = vec4<f32>(0.0, 0.8, 0.8, 1.0);
const e_color = vec4<f32>(1.0, 0.0, 0.0, 1.0);

struct VSOut {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
};

fn rotate_point(p: vec2<f32>, angle: f32) -> vec2<f32> {
	return vec2<f32>(
		cos(angle) * p.x - sin(angle) * p.y,
		sin(angle) * p.x + cos(angle) * p.y
	);
}

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> VSOut {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>( 3.0, -1.0),
		vec2<f32>(-1.0,  3.0),
	);
	var uvs = array<vec2<f32>, 3>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(2.0, 0.0),
		vec2<f32>(0.0, 2.0),
	);

	var out : VSOut;
	out.position = vec4<f32>(positions[vid], 0.0, 1.0);
	out.uv = uvs[vid];
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let res = params.resolution;
	
	let uv = in.uv;

	let x = u32(clamp(floor(uv.x * f32(res.x)), 0.0, f32(max(res.x, 1u) - 1u)));
	let y = u32(clamp(floor(uv.y * f32(res.y)), 0.0, f32(max(res.y, 1u) - 1u)));

	// render indicators
	{
		let uv_clamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
		let half_h = params.view_y_span * 0.5;
		let half_w = half_h * params.view_aspect_ratio;
		let offset = vec2<f32>(
			mix(-half_w, half_w, uv_clamped.x),
			mix(-half_h, half_h, uv_clamped.y)
		);
		let world = rotate_point(offset, params.rotation) + params.view_center;

		if (distance(world, params.initial_z) < params.z_indicator_size) {
			return z_color;
		}
		if (distance(world, params.exponent) < params.e_indicator_size) {
			return e_color;
		}
	}

	let pixels = res.x * res.y;
	let pixel_index = (y * res.x + x) * 3u;

	let r_iterations = histogram[pixel_index + 0u];
	let b_iterations = histogram[pixel_index + 2u];
	let g_iterations = histogram[pixel_index + 1u];

	let max_base = pixels * 3u;
	let r_max = max(histogram[max_base + 0u], params.normalization_floor);
	let b_max = max(histogram[max_base + 2u], params.normalization_floor);
	let g_max = max(histogram[max_base + 1u], params.normalization_floor);

	let r_t = clamp(r_iterations / r_max, 0.0, 1.0);
	let g_t = clamp(g_iterations / g_max, 0.0, 1.0);
	let b_t = clamp(b_iterations / b_max, 0.0, 1.0);

	let r = 1.0 - pow(1.0 - r_t, params.gamma);
	let g = 1.0 - pow(1.0 - g_t, params.gamma);
	let b = 1.0 - pow(1.0 - b_t, params.gamma);

	return vec4<f32>(r, g, b, 1.0);
	//return vec4<f32>(1.0, .0, .0, 1.0);
}