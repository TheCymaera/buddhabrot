@group(0) @binding(0) var<storage, read_write> histogram : array<atomic<u32>>;
@group(0) @binding(1) var<uniform> params : Params;

struct Params {
	resolution : vec2<u32>,
	samples_per_thread : u32,
	min_iterations : u32,
	max_iterations : u32,
	seed : u32,
	sample_min : vec2<f32>,
	sample_max : vec2<f32>,
	view_center : vec2<f32>,
	rotation : f32,
	view_y_span : f32,
	view_aspect_ratio : f32,
	escape_radius_sq : f32,
	gamma : f32,
	workgroup_count : u32,
	base_color : vec4<f32>,
};

fn lcg(state: ptr<function, u32>) -> f32 {
	(*state) = (*state) * 1664525u + 1013904223u;
	return f32((*state & 0x00FFFFFFu)) / 16777216.0;
}

fn random_in_range(state: ptr<function, u32>, min: f32, max: f32) -> f32 {
	return lcg(state) * (max - min) + min;
}

fn complex_random(state: ptr<function, u32>, min: vec2<f32>, max: vec2<f32>) -> vec2<f32> {
	let x = random_in_range(state, min.x, max.x);
	let y = random_in_range(state, min.y, max.y);
	return vec2<f32>(x, y);
}

fn complex_pow(z: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
	let r = length(z);
	if (r == 0.0) {
		return vec2<f32>(0.0, 0.0);
	}

	let theta = atan2(z.y, z.x);
	let log_r = log(r);

	let new_r = pow(r, e.x) * exp(-e.y * theta);
	let new_theta = e.x * theta + e.y * log_r;

	return vec2<f32>(new_r * cos(new_theta), new_r * sin(new_theta));
}

fn normalize(v: f32, min: f32, max: f32) -> f32 {
	return (v - min) / (max - min);
}

fn rotate_point(p: vec2<f32>, angle: f32) -> vec2<f32> {
	return vec2<f32>(
		cos(angle) * p.x - sin(angle) * p.y,
		sin(angle) * p.x + cos(angle) * p.y
	);
}

const PI = 3.141592653589793;
fn world_to_pixel(p: vec2<f32>, resolution: vec2<u32>) -> vec2<i32> {
	let span = params.view_y_span;
	let aspect_ratio = params.view_aspect_ratio;
	let center = params.view_center;

	let offset = rotate_point(p - center, params.rotation);

	let half_w = span / 2.0 * aspect_ratio;
	let half_h = span / 2.0;
	let norm_x = normalize(offset.x, -half_w, half_w);
	let norm_y = normalize(offset.y, -half_h, half_h);

	let px = i32(floor(norm_x * f32(resolution.x)));
	let py = i32(floor(norm_y * f32(resolution.y)));
	return vec2<i32>(px, py);
}

fn count_iterations(z0: vec2<f32>, e: vec2<f32>, c: vec2<f32>) -> u32 {
	var iterations = 0u;
	
	var z = z0;
	while (iterations < params.max_iterations) {
		z = complex_pow(z, e) + c;
		iterations++;

		if (dot(z, z) > params.escape_radius_sq) {
			break;
		}
	}

	return iterations;
}

fn increment_pixel(pixel: vec2<i32>) {
	let resolution = params.resolution;
	if (pixel.x < 0 || pixel.y < 0 || pixel.x >= i32(resolution.x) || pixel.y >= i32(resolution.y)) {
		return;
	}

	let index = u32(pixel.y) * resolution.x + u32(pixel.x);
	let new_value = atomicAdd(&histogram[index], 1u) + 1u;

	let max_index = resolution.x * resolution.y;
	atomicMax(&histogram[max_index], new_value);
}

fn accumulate_orbit(
	z0: vec2<f32>, e: vec2<f32>, c: vec2<f32>, 
	iterations: u32
) {
	let resolution = params.resolution;
	var z = z0;
	for (var i = 0u; i < iterations; i = i + 1u) {
		z = complex_pow(z, e) + c;
		let pixel = world_to_pixel(z, resolution);
		increment_pixel(pixel);
	}
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
	let resolution = params.resolution;
	var seed = params.seed + gid.x * 747796405u + gid.y * 2891336453u + gid.z * 805459861u + 1u;
	let sample_count = params.samples_per_thread;

	for (var s = 0u; s < sample_count; s = s + 1u) {
		let z0 = vec2<f32>(0.0, 0.0);
		let e = vec2<f32>(2.0, 0.0);
		let c = complex_random(&seed, params.sample_min, params.sample_max);
		
		let i = count_iterations(z0, e, c);

		let outside = i >= params.max_iterations || i < params.min_iterations;
		if (outside) { continue; }

		accumulate_orbit(z0, e, c, i);
	}
}