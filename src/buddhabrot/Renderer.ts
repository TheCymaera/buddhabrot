import { Buddhabrot } from './Buddhabrot.js';
import COMPUTE_SHADER from './shaders/compute.wgsl?raw';
import RENDER_SHADER from './shaders/render.wgsl?raw';
import LERP_SHADER from './shaders/lerp.wgsl?raw';
import { Struct } from './Struct.js';

const WORKGROUP_SIZE = parseInt(COMPUTE_SHADER.match(/@workgroup_size\((\d+)\)/)?.[1]!);
if (!isFinite(WORKGROUP_SIZE)) throw new Error('Failed to parse workgroup size from compute shader.');

const LERP_WORKGROUP_SIZE = parseInt(LERP_SHADER.match(/@workgroup_size\((\d+)\)/)?.[1]!);
if (!isFinite(LERP_WORKGROUP_SIZE)) throw new Error('Failed to parse workgroup size from lerp shader.');

async function createPipelines(device: GPUDevice, format: GPUTextureFormat) {
	const computeModule = device.createShaderModule({ code: COMPUTE_SHADER });
	const renderModule = device.createShaderModule({ code: RENDER_SHADER });
	const lerpModule = device.createShaderModule({ code: LERP_SHADER });

	const computePipeline = device.createComputePipelineAsync({
		layout: 'auto',
		compute: {
			module: computeModule,
			entryPoint: 'main',
		},
	});

	const lerpPipeline = device.createComputePipelineAsync({
		layout: 'auto',
		compute: {
			module: lerpModule,
			entryPoint: 'main',
		},
	});

	const renderPipeline = device.createRenderPipelineAsync({
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

	return {
		compute: await computePipeline,
		lerp: await lerpPipeline,
		render: await renderPipeline,
	}
}


export class Renderer {
	autoClearHistogram = true;
	alwaysClearHistogram = false;

	/**
	 * Re-created on resolution change.
	 */
	#transient?: {
		readonly buffers: {
			readonly histogram: GPUBuffer,
			readonly lerpedHistogram: GPUBuffer,
		}
		readonly bindGroups: {
			readonly compute: GPUBindGroup,
			readonly lerp: GPUBindGroup,
			readonly render: GPUBindGroup,
		}
	}

	private constructor(
		private readonly device: GPUDevice,
		private readonly canvas: HTMLCanvasElement,
		private readonly context: GPUCanvasContext,
		private readonly uniformBuffer: GPUBuffer,
		private readonly pipelines: {
			readonly compute: GPUComputePipeline,
			readonly lerp: GPUComputePipeline,
			readonly render: GPURenderPipeline,
		},
	) { }

	static async create(canvas: HTMLCanvasElement) {
		const context = canvas.getContext('webgpu')!;
		if (!context) throw new Error('Failed to create WebGPU context');

		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) throw new Error('Failed to get GPU adapter');

		const device = await adapter.requestDevice();
		const format = navigator.gpu.getPreferredCanvasFormat();

		// configure
		context.configure({ device, format, alphaMode: 'opaque' });

		// create uniform buffer
		const uniformBuffer = device.createBuffer({
			size: Renderer.#createUniformBuffer({ width: canvas.width, height: canvas.height }, new Buddhabrot()).byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		// create renderer
		const out = new Renderer(
			device,
			canvas,
			context,
			uniformBuffer,
			await createPipelines(device, format),
		);

		return out;
	}

	#previousBuddhabrot?: Buddhabrot;
	render(buddhabrot: Buddhabrot) {
		if (this.alwaysClearHistogram ||
			this.autoClearHistogram && 
			this.#previousBuddhabrot && 
			!this.#previousBuddhabrot?.canReuseHistogram(buddhabrot)) {
			this.clearHistogram();
		}

		this.#previousBuddhabrot = buddhabrot.clone();


		if (!this.#transient) this.#transient = this.#createTransientResources();

		const workgroupCount = Math.ceil(buddhabrot.samples / WORKGROUP_SIZE);

		const encoder = this.device.createCommandEncoder();

		// set uniforms
		const buffer = Renderer.#createUniformBuffer({ width: this.canvas.width, height: this.canvas.height }, buddhabrot);
		this.device.queue.writeBuffer(this.uniformBuffer, 0, buffer);
		
		// compute pass
		const computePass = encoder.beginComputePass();
		computePass.setPipeline(this.pipelines.compute);
		computePass.setBindGroup(0, this.#transient!.bindGroups.compute);
		computePass.dispatchWorkgroups(workgroupCount);
		computePass.end();

		// lerp pass
		const lerpPass = encoder.beginComputePass();
		lerpPass.setPipeline(this.pipelines.lerp);
		lerpPass.setBindGroup(0, this.#transient.bindGroups.lerp);
		const lerpCount = Math.ceil(this.#getHistogramElementCount(this.canvas.width, this.canvas.height) / LERP_WORKGROUP_SIZE);
		lerpPass.dispatchWorkgroups(lerpCount);
		lerpPass.end();

		// render pass (into textureView)
		const textureView = this.context.getCurrentTexture().createView();
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
		renderPass.setPipeline(this.pipelines.render);
		renderPass.setBindGroup(0, this.#transient.bindGroups.render);
		renderPass.draw(3);
		renderPass.end();

		// submit
		this.device.queue.submit([encoder.finish()]);
	}

	setResolution({ width, height }: { width: number, height: number }) {
		if (this.canvas.width === width && this.canvas.height === height) {
			return;
		}

		this.canvas.width = width;
		this.canvas.height = height;

		this.#transient?.buffers.histogram.destroy();
		this.#transient?.buffers.lerpedHistogram.destroy();

		this.#transient = this.#createTransientResources();
	}

	clearHistogram() {
		if (!this.#transient) return;

		const elementCount = this.#getHistogramElementCount(this.canvas.width, this.canvas.height);
		this.device.queue.writeBuffer(
			this.#transient.buffers.histogram,
			0,
			new Uint32Array(elementCount).fill(0)
		);
	}

	downloadImage(fileName: string) {
		this.canvas.toBlob((blob) => {
			if (!blob) {
				console.error('Failed to create blob from canvas');
				return;
			}

			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = fileName;
			a.click();
			URL.revokeObjectURL(url);
		});
	}

	getImageBlob() {
		return new Promise<Blob>((resolve, reject) => {
			this.canvas.toBlob((blob) => {
				if (!blob) {
					reject(new Error('Failed to create blob from canvas'));
					return;
				}
				resolve(blob);
			});
		});
	}

	onFinish() {
		return this.device.queue.onSubmittedWorkDone();
	}

	#getHistogramElementCount(width: number, height: number) {
		return (width * height * 3) + 3; // 3 channels + 3 max slots
	}

	#createTransientResources() {
		const elementCount = this.#getHistogramElementCount(this.canvas.width, this.canvas.height);

		// histogram
		const histogram = this.device.createBuffer({
			size: elementCount * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true,
		});
		new Uint32Array(histogram.getMappedRange()).fill(0);
		histogram.unmap();
		
		// lerped histogram
		const lerpedHistogram = this.device.createBuffer({
			size: elementCount * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true,
		});
		new Float32Array(lerpedHistogram.getMappedRange()).fill(0);
		lerpedHistogram.unmap();
		
		// compute bind group
		const computeBindGroup = this.device.createBindGroup({
			label: 'compute bind group',
			layout: this.pipelines.compute.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: histogram } },
				{ binding: 1, resource: { buffer: this.uniformBuffer } },
			],
		});
		
		// lerp bind group
		const lerpBindGroup = this.device.createBindGroup({
			label: 'lerp bind group',
			layout: this.pipelines.lerp.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: histogram } },
				{ binding: 1, resource: { buffer: lerpedHistogram } },
				{ binding: 2, resource: { buffer: this.uniformBuffer } },
			],
		});
		
		// render bind group
		const renderBindGroup = this.device.createBindGroup({
			label: 'render bind group',
			layout: this.pipelines.render.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: lerpedHistogram } },
				{ binding: 1, resource: { buffer: this.uniformBuffer } },
			],
		});

		return {
			buffers: {
				histogram,
				lerpedHistogram,
			},
			bindGroups: {
				compute: computeBindGroup,
				lerp: lerpBindGroup,
				render: renderBindGroup,
			}
		}
	}

	static #createUniformBuffer(resolution: { width: number, height: number }, buddhabrot: Buddhabrot) {
		const workgroupCount = Math.ceil(buddhabrot.samples / WORKGROUP_SIZE);
		const samplesPerThread = Math.ceil(buddhabrot.samples / workgroupCount);
		
		return new Struct()
			.u32(buddhabrot.anti ? 1 : 0)
			.vec2_u32([resolution.width, resolution.height])
			.u32(samplesPerThread)
			.u32(buddhabrot.maxIterations1)
			.u32(buddhabrot.maxIterations2)
			.u32(buddhabrot.maxIterations3)
			.u32(buddhabrot.seed())
			.vec2_f32([buddhabrot.sampleCenter.x, buddhabrot.sampleCenter.y])
			.f32(buddhabrot.sampleRadius)
			.u32(buddhabrot.uniformSampleDistribution ? 1 : 0)
			.vec2_f32([buddhabrot.viewCenter.x, buddhabrot.viewCenter.y])
			.vec2_f32([buddhabrot.initialZ.x, buddhabrot.initialZ.y])
			.vec2_f32([buddhabrot.exponent.x, buddhabrot.exponent.y])
			.f32(buddhabrot.rotation)
			.f32(buddhabrot.viewYSpan)
			.f32(resolution.width / Math.max(resolution.height, 1))
			.f32(buddhabrot.bailoutRadius ** 2)
			.f32(buddhabrot.gamma)
			.f32(1 - buddhabrot.frameLerp)
			.f32(buddhabrot.effectiveZIndicatorSize / buddhabrot.zoomLevel)
			.f32(buddhabrot.effectiveEIndicatorSize / buddhabrot.zoomLevel)
			.pack();
	}
}