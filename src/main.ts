import './main.css';
import COMPUTE_SHADER from './shaders/compute.wgsl?raw';
import RENDER_SHADER from './shaders/render.wgsl?raw';
import SMOOTH_SHADER from './shaders/smooth.wgsl?raw';
import { Struct } from './struct.js';
import { Vec2 } from './Vec2.js';

// config
const SAMPLES = 2 ** 12;
const MAX_ITERATIONS = 1000 * 3;
const MIN_ITERATIONS = 0;
const ESCAPE_RADIUS = 4;
//const SAMPLE_MIN = Vec2.new(-2, -2);
//const SAMPLE_MAX = Vec2.new(2, 2);
const SAMPLE_CENTER = Vec2.new(0, 0);
const SAMPLE_RADIUS = 2.5;
let VIEW_Y_SPAN = 4.0;
const VIEW_CENTER = Vec2.new(0, 0);
const INITIAL_Z = Vec2.new(0, 0);
const EXPONENT = Vec2.new(2, 0);
const ROTATION = Math.PI / 2;
const Z_INDICATOR_SIZE = 0.025;
const E_INDICATOR_SIZE = 0.025;
const SEED = () => performance.now();
const BASE_COLOR = [166, 222, 255, 255].map(c => c / 255) as [number, number, number, number];
const GAMMA = 4.0;
const HISTOGRAM_LERP = 0.2;

let inputMode: "c" | "z" | "e" = "c";

const WORKGROUP_SIZE = parseInt(COMPUTE_SHADER.match(/@workgroup_size\((\d+)\)/)?.[1]!);
if (!isFinite(WORKGROUP_SIZE)) throw new Error('Failed to parse workgroup size from compute shader.');
const WORKGROUP_COUNT = Math.ceil(SAMPLES / WORKGROUP_SIZE);
const SAMPLES_PER_THREAD = Math.ceil(SAMPLES / WORKGROUP_COUNT);

const SMOOTH_WORKGROUP_SIZE = parseInt(SMOOTH_SHADER.match(/@workgroup_size\((\d+)\)/)?.[1]!);
if (!isFinite(SMOOTH_WORKGROUP_SIZE)) throw new Error('Failed to parse workgroup size from smooth shader.');

if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');

const canvas = document.querySelector('canvas')!;

const context = canvas.getContext('webgpu')!;
if (!context) throw new Error('Failed to create WebGPU context');

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('Failed to get GPU adapter');

const device = await adapter.requestDevice();
const format = navigator.gpu.getPreferredCanvasFormat();

const uniformBuffer = device.createBuffer({
	size: getUniformData().byteLength,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const computeModule = device.createShaderModule({ code: COMPUTE_SHADER });
const renderModule = device.createShaderModule({ code: RENDER_SHADER });
const smoothModule = device.createShaderModule({ code: SMOOTH_SHADER });

const computePipeline = device.createComputePipeline({
	layout: 'auto',
	compute: {
		module: computeModule,
		entryPoint: 'main',
	},
});

const smoothPipeline = device.createComputePipeline({
	layout: 'auto',
	compute: {
		module: smoothModule,
		entryPoint: 'main',
	},
});

const renderPipeline = device.createRenderPipeline({
	layout: 'auto',
	vertex: {
		module: renderModule,
		entryPoint: 'vs_main',
	},
	fragment: {
		module: renderModule,
		entryPoint: 'fs_main',
		targets: [{ format }],
	},
});

let histogramBuffer: GPUBuffer;
let smoothedHistogramBuffer: GPUBuffer;
let computeBindGroup: GPUBindGroup;
let smoothBindGroup: GPUBindGroup;
let renderBindGroup: GPUBindGroup;

function getHistogramElementCount(width: number, height: number) {
	return (width * height * 3) + 3; // 3 channels + max slots
}

function createHistogramBuffer(width: number, height: number) {
	const elementCount = getHistogramElementCount(width, height);
	const buffer = device.createBuffer({
		size: elementCount * Uint32Array.BYTES_PER_ELEMENT,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		mappedAtCreation: true,
	});
	new Uint32Array(buffer.getMappedRange()).fill(0);
	buffer.unmap();
	return buffer;
}

function createSmoothedHistogramBuffer(width: number, height: number) {
	const elementCount = getHistogramElementCount(width, height);
	const buffer = device.createBuffer({
		size: elementCount * Float32Array.BYTES_PER_ELEMENT,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		mappedAtCreation: true,
	});
	new Float32Array(buffer.getMappedRange()).fill(0);
	buffer.unmap();
	return buffer;
}

function createComputeBindGroup() {
	return device.createBindGroup({
		label: 'compute bind group',
		layout: computePipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: histogramBuffer } },
			{ binding: 1, resource: { buffer: uniformBuffer } },
		],
	});
}

function createSmoothBindGroup() {
	return device.createBindGroup({
		label: 'smooth bind group',
		layout: smoothPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: histogramBuffer } },
			{ binding: 1, resource: { buffer: smoothedHistogramBuffer } },
			{ binding: 2, resource: { buffer: uniformBuffer } },
		],
	});
}

function createRenderBindGroup() {
	return device.createBindGroup({
		label: 'render bind group',
		layout: renderPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: smoothedHistogramBuffer } },
			{ binding: 1, resource: { buffer: uniformBuffer } },
		],
	});
}

function updateCanvasSize() {
	const dpr = window.devicePixelRatio || 1;
	canvas.width = Math.floor(canvas.clientWidth * dpr) || 1;
	canvas.height = Math.floor(canvas.clientHeight * dpr) || 1;

	histogramBuffer?.destroy();
	smoothedHistogramBuffer?.destroy();
	histogramBuffer = createHistogramBuffer(canvas.width, canvas.height);
	smoothedHistogramBuffer = createSmoothedHistogramBuffer(canvas.width, canvas.height);
	computeBindGroup = createComputeBindGroup();
	smoothBindGroup = createSmoothBindGroup();
	renderBindGroup = createRenderBindGroup();
};

function render() {
	// write uniforms
	device.queue.writeBuffer(uniformBuffer, 0, getUniformData());

	const encoder = device.createCommandEncoder();
	
	// compute pass
	const computePass = encoder.beginComputePass();
	computePass.setPipeline(computePipeline);
	computePass.setBindGroup(0, computeBindGroup);
	computePass.dispatchWorkgroups(WORKGROUP_COUNT);
	computePass.end();

	// smooth histogram
	const smoothPass = encoder.beginComputePass();
	smoothPass.setPipeline(smoothPipeline);
	smoothPass.setBindGroup(0, smoothBindGroup);
	const smoothCount = Math.ceil(getHistogramElementCount(canvas.width, canvas.height) / SMOOTH_WORKGROUP_SIZE);
	smoothPass.dispatchWorkgroups(smoothCount);
	smoothPass.end();

	// render pass (into textureView)
	const textureView = context.getCurrentTexture().createView();
	const renderPass = encoder.beginRenderPass({
		colorAttachments: [
			{
				view: textureView,
				loadOp: 'clear',
				storeOp: 'store',
				clearValue: { r: 0, g: 0, b: 0, a: 0 },
			},
		],
	});
	renderPass.setPipeline(renderPipeline);
	renderPass.setBindGroup(0, renderBindGroup);
	renderPass.draw(3);
	renderPass.end();

	// submit
	device.queue.submit([encoder.finish()]);
};


// configure
context.configure({ device, format, alphaMode: 'opaque' });

// handle resize
{
	let init = false;
	new ResizeObserver(() => {
		if (!init) {
			init = true;
			return;
		}

		updateCanvasSize();
		render();
	}).observe(canvas);
}

// main loop
function loop() {
	handleControls();
	render();
	requestAnimationFrame(loop);
}
updateCanvasSize();
requestAnimationFrame(loop);

function getUniformData() {
	return new Struct()
		.vec2_u32([canvas.width, canvas.height])
		.u32(SAMPLES_PER_THREAD)
		.u32(MIN_ITERATIONS)
		.u32(MAX_ITERATIONS)
		.u32(SEED())
		//.vec2_f32([SAMPLE_MIN.x, SAMPLE_MIN.y])
		//.vec2_f32([SAMPLE_MAX.x, SAMPLE_MAX.y])
		.vec2_f32([SAMPLE_CENTER.x, SAMPLE_CENTER.y])
		.f32(SAMPLE_RADIUS)
		.vec2_f32([VIEW_CENTER.x, VIEW_CENTER.y])
		.vec2_f32([INITIAL_Z.x, INITIAL_Z.y])
		.vec2_f32([EXPONENT.x, EXPONENT.y])
		.f32(ROTATION)
		.f32(VIEW_Y_SPAN)
		.f32(canvas.width / Math.max(canvas.height, 1))
		.f32(ESCAPE_RADIUS ** 2)
		.f32(GAMMA)
		.f32(HISTOGRAM_LERP)
		.f32(inputMode === 'z' ? Z_INDICATOR_SIZE : 0)
		.f32(inputMode === 'e' ? E_INDICATOR_SIZE : 0)
		.vec4_f32(BASE_COLOR)
		.pack();
}

const keysPressed = new Set<string>();

let moveSpeed = 1;

addEventListener('keydown', (e) => {
	keysPressed.add(e.code);

	if (e.code === 'Digit1') {
		inputMode = 'c';
	}
	
	if (e.code === 'Digit2') {
		inputMode = 'z';
	}
	
	if (e.code === 'Digit3') {
		inputMode = 'e';
	}

	if (e.code === 'BracketLeft') {
		moveSpeed /= 2;
	}

	if (e.code === 'BracketRight') {
		moveSpeed *= 2;
	}

	console.log(e.code);
});

addEventListener('keyup', (e) => {
	keysPressed.delete(e.code);
});

function handleControls() {
	const panAmount = 0.01 * VIEW_Y_SPAN * moveSpeed;
	const zoomFactor = 0.95;

	let didChange = false;
	const consumeKey = (key: string) => {
		const result = keysPressed.has(key);
		if (result) didChange = true;
		return result;
	}

	const velocity = Vec2.new(0, 0);
	if (consumeKey('KeyW')) {
		velocity.y += panAmount;
	}
	if (consumeKey('KeyS')) {
		velocity.y -= panAmount;
	}
	if (consumeKey('KeyA')) {
		velocity.x -= panAmount;
	}
	if (consumeKey('KeyD')) {
		velocity.x += panAmount;
	}
	if (consumeKey('ShiftLeft') || consumeKey('ShiftRight')) {
		VIEW_Y_SPAN *= zoomFactor;
	}
	if (consumeKey('Space')) {
		VIEW_Y_SPAN /= zoomFactor;
	}
	
	velocity.rotate(ROTATION);

	const targetVector =
		inputMode === 'c' ? VIEW_CENTER :
		inputMode === 'z' ? INITIAL_Z :
		EXPONENT;

	targetVector.add(velocity);

	if (didChange) {
		resetHistogram();
	}
}

function resetHistogram() {
	const elementCount = getHistogramElementCount(canvas.width, canvas.height);
	device.queue.writeBuffer(
		histogramBuffer,
		0,
		new Uint32Array(elementCount).fill(0)
	);
	//device.queue.writeBuffer(
	//	smoothedHistogramBuffer,
	//	0,
	//	new Float32Array(elementCount).fill(0)
	//);
}