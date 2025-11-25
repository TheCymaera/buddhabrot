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
	view_y_width : f32,
	view_aspect_ratio : f32,
	escape_radius_sq : f32,
	gamma : f32,
	workgroup_count : u32,
};

fn lcg(state: ptr<function, u32>) -> f32 {
	(*state) = (*state) * 1664525u + 1013904223u;
	return f32((*state & 0x00FFFFFFu)) / 16777216.0;
}

fn random_in_range(state: ptr<function, u32>, min: f32, max: f32) -> f32 {
	return lcg(state) * (max - min) + min;
}

fn random_c(state: ptr<function, u32>) -> vec2<f32> {
	let min = params.sample_min;
	let max = params.sample_max;
	let x = random_in_range(state, min.x, max.x);
	let y = random_in_range(state, min.y, max.y);
	return vec2<f32>(x, y);
}

fn complex_sq(z: vec2<f32>) -> vec2<f32> {
	return vec2<f32>(
		z.x * z.x - z.y * z.y,
		2.0 * z.x * z.y
	);
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
fn world_to_pixel(z: vec2<f32>, resolution: vec2<u32>) -> vec2<i32> {
	let span = params.view_y_width;
	let aspect_ratio = params.view_aspect_ratio;
	let center = params.view_center;

	let rotation = -PI / 2.0;
	let offset = rotate_point(z - center, rotation);

	let half_w = span / 2.0 * aspect_ratio;
	let half_h = span / 2.0;
	let norm_x = normalize(offset.x, -half_w, half_w);
	let norm_y = normalize(offset.y, -half_h, half_h);

	let px = i32(floor(norm_x * f32(resolution.x)));
	let py = i32(floor(norm_y * f32(resolution.y)));
	return vec2<i32>(px, py);
}

fn count_iterations(c: vec2<f32>, max_iter: u32, escape_sq: f32) -> u32 {
	var z = vec2<f32>(0.0, 0.0);
	var iter = 0u;
	loop {
		if (iter >= max_iter) {
			return iter;
		}
		z = complex_sq(z) + c;
		iter = iter + 1u;
		if (dot(z, z) > escape_sq) {
			return iter;
		}
	}
}

fn accumulate_orbit(c: vec2<f32>, steps: u32, resolution: vec2<u32>, max_index: u32) {
	var z = vec2<f32>(0.0, 0.0);
	for (var i = 0u; i < steps; i = i + 1u) {
		z = complex_sq(z) + c;
		let pixel = world_to_pixel(z, resolution);
		if (pixel.x < 0 || pixel.y < 0 || pixel.x >= i32(resolution.x) || pixel.y >= i32(resolution.y)) {
			continue;
		}

		let index = u32(pixel.y) * resolution.x + u32(pixel.x);
		let new_value = atomicAdd(&histogram[index], 1u) + 1u;
		atomicMax(&histogram[max_index], new_value);
	}
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
	let resolution = params.resolution;
	var seed = params.seed + gid.x * 747796405u + gid.y * 2891336453u + gid.z * 805459861u + 1u;
	let min_iterations = params.min_iterations;
	let max_iterations = params.max_iterations;
	let escape_square = params.escape_radius_sq;
	let sample_count = params.samples_per_thread;
    let pixel_count = resolution.x * resolution.y;
    let max_slot = pixel_count;

	for (var s = 0u; s < sample_count; s = s + 1u) {
		let c = random_c(&seed);
		
		let i = count_iterations(c, max_iterations, escape_square);

		// discard points that do not escape (in the Mandelbrot set),
		// and points that escape too quickly
		if (i >= max_iterations || i < min_iterations) { continue; }

		accumulate_orbit(c, i, resolution, max_slot);
	}
}