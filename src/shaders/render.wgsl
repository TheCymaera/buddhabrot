
struct Params {
	resolution : vec2<u32>,
	samples_per_thread : u32,
	min_iterations : u32,
	max_iterations : u32,
	seed : u32,
	sample_min : vec2<f32>,
	sample_max : vec2<f32>,
	view_center : vec2<f32>,
	initial_z : vec2<f32>,
	exponent : vec2<f32>,
	rotation : f32,
	view_y_span : f32,
	view_aspect_ratio : f32,
	escape_radius_sq : f32,
	gamma : f32,
	z_indicator_size : f32,
	e_indicator_size : f32,
	base_color : vec4<f32>,
};

@group(0) @binding(0) var<storage, read> histogram : array<u32>;
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
		let uv01 = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
		let half_h = params.view_y_span * 0.5;
		let half_w = half_h * params.view_aspect_ratio;
		let offset = vec2<f32>(
			mix(-half_w, half_w, uv01.x),
			mix(-half_h, half_h, uv01.y)
		);
		let world = rotate_point(offset, params.rotation) + params.view_center;

		if (distance(world, params.initial_z) < params.z_indicator_size) {
			return z_color;
		}
		if (distance(world, params.exponent) < params.e_indicator_size) {
			return e_color;
		}
	}

	let index = y * res.x + x;

	let value = f32(histogram[index]);
	let max_index = res.x * res.y;
	let max_value = max(f32(histogram[max_index]), 1.0);

	let t = clamp(value / max_value, 0.0, 1.0);

	let eased = 1 - pow(1.0 - t, params.gamma);

	let color = params.base_color * eased;
	return color;
}