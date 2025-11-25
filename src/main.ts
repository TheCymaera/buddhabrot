import './main.css';
import COMPUTE_SHADER from './shaders/compute.wgsl?raw';
import RENDER_SHADER from './shaders/render.wgsl?raw';

// config
const WORKGROUP_COUNT = 256;
const SAMPLES_PER_THREAD = 2 ** 10;
const MAX_ITERATIONS = 1000 * 5;
const MIN_ITERATIONS = 20;
const ESCAPE_RADIUS = 4;
const SAMPLE_MIN = { x: -2, y: -1.5 };
const SAMPLE_MAX = { x:  1, y:  1.5 };
const VIEW_Y_WIDTH = SAMPLE_MAX.y - SAMPLE_MIN.y;
const VIEW_CENTER = { x: -.5, y: 0 };
const SEED = 123456;
const GAMMA = 4.0;

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

const computePipeline = device.createComputePipeline({
	layout: 'auto',
	compute: {
		module: computeModule,
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
let computeBindGroup: GPUBindGroup;
let renderBindGroup: GPUBindGroup;

function createHistogramBuffer(width: number, height: number) {
	const elementCount = width * height || 1;
	const buffer = device.createBuffer({
		size: elementCount * Uint32Array.BYTES_PER_ELEMENT,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		mappedAtCreation: true,
	});
	new Uint32Array(buffer.getMappedRange()).fill(0);
	buffer.unmap();
	return buffer;
}

function createBindGroup(pipeline: GPUComputePipeline | GPURenderPipeline) {
	return device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: histogramBuffer } },
			{ binding: 1, resource: { buffer: uniformBuffer } },
		],
	});
}

function updateCanvasSize() {
	const dpr = window.devicePixelRatio || 1;
	canvas.width = Math.floor(window.innerWidth * dpr) || 1;
	canvas.height = Math.floor(window.innerHeight * dpr) || 1;
	histogramBuffer = createHistogramBuffer(canvas.width, canvas.height);
	computeBindGroup = createBindGroup(computePipeline);
	renderBindGroup = createBindGroup(renderPipeline);
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


// configure and render
context.configure({ device, format, alphaMode: 'opaque' });

new ResizeObserver(debounce(() => {
	updateCanvasSize();
	render();
}, 200)).observe(document.body);

function getUniformData() {
	const uniformData = new Float32Array(16);
	
	uniformData[0] = canvas.width;
	uniformData[1] = canvas.height;
	uniformData[2] = SAMPLES_PER_THREAD;
	uniformData[3] = MIN_ITERATIONS

	uniformData[4] = MAX_ITERATIONS;
	uniformData[5] = SEED;
	uniformData[6] = SAMPLE_MIN.x;
	uniformData[7] = SAMPLE_MIN.y;

	uniformData[8] = SAMPLE_MAX.x;
	uniformData[9] = SAMPLE_MAX.y;
	uniformData[10] = VIEW_CENTER.x;
	uniformData[11] =VIEW_CENTER.y ;

	uniformData[12] = VIEW_Y_WIDTH;
	uniformData[13] = (canvas.width / Math.max(canvas.height, 1));
	uniformData[14] = ESCAPE_RADIUS ** 2;
	uniformData[15] = GAMMA;

	return uniformData;
}

function debounce<F extends (...args: any[]) => void>(fn: F, delay: number): F {
	let timeoutId: number | undefined;
	return function(this: any, ...args: any[]) {
		if (timeoutId !== undefined) {
			clearTimeout(timeoutId);
		}
		timeoutId = window.setTimeout(() => {
			fn.apply(this, args);
		}, delay);
	} as F;
}