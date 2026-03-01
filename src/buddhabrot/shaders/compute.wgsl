@group(0) @binding(0) var<storage, read_write> histogram : array<atomic<u32>>;
@group(0) @binding(1) var<uniform> params : Params;

struct Params {
	resolution : vec2<u32>,
	samples_per_thread : u32,
	max_iterations : u32,
	max_iterations_2 : u32,
	max_iterations_3 : u32,
	seed : u32,
	sample_center : vec2<f32>,
	sample_radius : f32,
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

fn pcg_hash(input: u32) -> u32 {
	var state = input * 747796405u + 2891336453u;
	let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

fn random_u32(state: ptr<function, u32>) -> u32 {
	var x = *state;
	x = x ^ (x << 13u);
	x = x ^ (x >> 17u);
	x = x ^ (x << 5u);
	*state = x;
	return x;
}

fn random_f32_unit(state: ptr<function, u32>) -> f32 {
	let x = random_u32(state);
	return f32(x) * (1.0 / 4294967296.0);
}

fn random_f32(state: ptr<function, u32>, min: f32, max: f32) -> f32 {
	let range = max - min;
	let x = random_u32(state);
	return f32(x) * (range / 4294967296.0) + min;
}

fn random_complex_delta(state: ptr<function, u32>, original: vec2<f32>, max_delta: f32) -> vec2<f32> {
	let angle = random_f32(state, 0.0, 6.28318530718);
	let radius = random_f32(state, 0.0, max_delta);
	return original + vec2<f32>(cos(angle), sin(angle)) * radius;
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

fn world_to_pixel(p: vec2<f32>) -> vec2<i32> {
	let span = params.view_y_span;
	let resolution = params.resolution;
	let aspect_ratio = params.view_aspect_ratio;
	let center = params.view_center;

	let offset = rotate_point(p - center, -params.rotation);

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

fn increment_pixel_channel(pixel: vec2<i32>, channel: u32) {
	let resolution = params.resolution;
	if (pixel.x < 0 || pixel.y < 0 || pixel.x >= i32(resolution.x) || pixel.y >= i32(resolution.y)) {
		return;
	}

	let pixels = resolution.x * resolution.y;
	let index = (u32(pixel.y) * resolution.x + u32(pixel.x)) * 3u + channel;
	let new_value = atomicAdd(&histogram[index], 1u) + 1u;

	let max_index = pixels * 3u + channel;
	atomicMax(&histogram[max_index], new_value);
}

fn accumulate_orbit(
	z0: vec2<f32>, e: vec2<f32>, c: vec2<f32>, 
	iterations: u32
) {
	var z = z0;
	for (var i = 0u; i < iterations; i++) {
		z = complex_pow(z, e) + c;
		let pixel = world_to_pixel(z);

		if (i <= params.max_iterations) { increment_pixel_channel(pixel, 0u); }
		if (i <= params.max_iterations_2) { increment_pixel_channel(pixel, 1u); }
		if (i <= params.max_iterations_3) { increment_pixel_channel(pixel, 2u); }
	}
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
	let resolution = params.resolution;

	var seed = params.seed + gid.x * 747796405u + gid.y * 2891336453u + gid.z * 805459861u + 1u;
	seed = pcg_hash(seed);
	if (seed == 0u) { seed = 1u; }

	let sample_count = params.samples_per_thread;

	for (var s = 0u; s < sample_count; s++) {
		let z0 = params.initial_z;
		let e = params.exponent;
		let c = random_complex_delta(&seed, params.sample_center, params.sample_radius);
		
		let i = count_iterations(z0, e, c);

		let inside = i >= params.max_iterations;
		if (inside) { continue; }

		accumulate_orbit(z0, e, c, i);
	}
}