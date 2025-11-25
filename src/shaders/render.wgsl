
struct Params {
	v0 : vec4<f32>,
	v1 : vec4<f32>,
	v2 : vec4<f32>,
	v3 : vec4<f32>,
};

@group(0) @binding(0) var<storage, read> histogram : array<u32>;
@group(0) @binding(1) var<uniform> params : Params;

fn resolution() -> vec2<u32> {
	return vec2<u32>(u32(params.v0.x), u32(params.v0.y));
}

fn gamma() -> f32 {
	return params.v3.w;
}

fn max_value() -> u32 {
	let res = resolution();
	var max_val = 0u;
	let max_index = res.x * res.y;
	for (var i = 0u; i < max_index; i++) {
		if (histogram[i] > max_val) {
			max_val = histogram[i];
		}
	}
	return max_val;
}

struct VSOut {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
};

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
	let res = resolution();
	
	let uv = in.uv;

	let x = u32(clamp(floor(uv.x * f32(res.x)), 0.0, f32(max(res.x, 1u) - 1u)));
	let y = u32(clamp(floor(uv.y * f32(res.y)), 0.0, f32(max(res.y, 1u) - 1u)));

	let index = y * res.x + x;

	let value = f32(histogram[index]);
	//let max_value = 1000.0;
	let max_value = f32(max_value());



	let t = clamp(value / max_value, 0.0, 1.0);

	let eased = 1 - pow(1.0 - t, gamma());

	let color_base = vec4<f32>(166, 222, 255, 255) / 255.0;

	let color = color_base * eased;
	return color;
}