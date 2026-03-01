<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { Renderer } from '../buddhabrot/Renderer.js';
	import NumberField from '../ui-components/NumberField.svelte';
	import Button from '../ui-components/Button.svelte';
	import CircleButton from '../ui-components/CircleButton.svelte';
	import { fa5_brands_github, fa5_solid_bars, fa5_solid_book, fa5_solid_code, fa5_solid_info, fa5_solid_paintBrush, fa5_solid_play, fa5_solid_times, fa6_solid_upDownLeftRight } from 'fontawesome-svgs';
	import SelectField from '../ui-components/SelectField.svelte';
	import { githubRepositoryLink } from './links.js';
	import NavRailButton from '../ui-components/NavRailButton.svelte';
	import NavRail from '../ui-components/NavRail.svelte';
	import NavRailSpacer from '../ui-components/NavRailSpacer.svelte';
	import { MediaQuery } from 'svelte/reactivity';
	import { Buddhabrot, IndicatorSetting, InputMode } from '../buddhabrot/Buddhabrot.js';
	import { Vec2 } from '../open-utilities/Vec2.js';
	import { Timeline } from '../buddhabrot/Timeline.js';
	import { AnimationFrameScheduler } from '../open-utilities/AnimationFrameScheduler.js';
	import { Duration } from '../open-utilities/Duration.js';
	import { inputMap } from '../buddhabrot/InputMap.js';
	import { degToRad, radToDeg } from '../open-utilities/numbers.js';
	import TextField from '../ui-components/TextField.svelte';

	let canvas: HTMLCanvasElement;
	const clientDimensions = $state({ width: 0, height: 0 });
	
	const buddhabrot = new Buddhabrot();
	let renderer: Renderer | undefined = undefined;
	let rendererInitError: Error | undefined = $state(undefined);

	const resolution = $state({
		height: "Auto",
		width: "Auto",
	});

	const recordingIsEnabled = window.location.search.includes("record");

	// svelte-ignore perf_avoid_inline_class
	const app = new class App {
		moveSpeed = 1;
		runMainLoop = true;

		readonly timeline = new Timeline();
		isRecording = false;
		lastFrameTime = performance.now();
	}

	let reactive = $state({
		app,
		buddhabrot
	})

	tick().then(async () => {
		// create renderer
		const result = await Renderer.create(canvas).catch(e => e);

		if (result instanceof Error) {
			rendererInitError = result;
		} else {
			renderer = result;
		}

		// init resolution
		const dpr = window.devicePixelRatio || 1;
		const height = (Math.min(1024, clientDimensions.height * dpr) | 0) || 1;

		resolution.height = height.toString();
		resolution.width = "Auto";

		// add to global for scripting
		Object.assign(globalThis, {
			buddhabrot,
			renderer,
			Vec2,
			Buddhabrot,
			AnimationFrameScheduler,
			Duration,
			Timeline,
			inputMap,
			app,
		});
	});


	function calculateResolution() {
		const aspectRatio = (clientDimensions.width / Math.max(clientDimensions.height, 1)) || 1;
		const dpr = window.devicePixelRatio || 1;
		
		const widthSetting = parseInt(resolution.width) || undefined;
		const heightSetting = parseInt(resolution.height) || undefined;

		if (!widthSetting && !heightSetting) return {
			width: Math.floor(clientDimensions.width * dpr),
			height: Math.floor(clientDimensions.height * dpr),
		}

		if (heightSetting && !widthSetting) {
			const height = heightSetting;
			const width = Math.round(height * aspectRatio) || 1;
			return { height, width }
		}

		if (widthSetting && !heightSetting) {
			const width = widthSetting;
			const height = Math.round(width / aspectRatio) || 1;
			return { width, height };
		}

		return {
			width: widthSetting!,
			height: heightSetting!,
			useContainFit: true,
		};
	};

	$effect(()=>{
		const { width, height, useContainFit } = calculateResolution();
		
		if (clientDimensions.width === 0 || clientDimensions.height === 0) return;
		if (!app.runMainLoop) return;

		renderer?.setResolution({ width, height });
		renderer?.render(buddhabrot);

		canvas.style.objectFit = useContainFit ? 'contain' : '';
	});

	// main loop
	AnimationFrameScheduler.loop((deltaTime) => {
		reactive = {...reactive};

		if (!app.runMainLoop) return;

		update(deltaTime);
		renderer?.render(buddhabrot);

		if (app.isRecording) {
			app.timeline.addKeyframe({
				state: buddhabrot.clone(),
				duration: deltaTime,
			});
		}
	});

	// handle inputs
	inputMap.listeners.onInputModeChange = (mode) => buddhabrot.inputMode = mode;
	inputMap.listeners.onHalfSpeed = () => app.moveSpeed /= 2;
	inputMap.listeners.onDoubleSpeed = () => app.moveSpeed *= 2;
	inputMap.listeners.onToggleRecording = () => {
		if (!recordingIsEnabled) return;

		app.isRecording = !app.isRecording;
		if (app.isRecording) {
			app.timeline.frames.length = 0;
		}
	}

	function snapToCardinalDirection(vec: Vec2, threshold = 0.00001): Vec2 {
		if (vec.x > 1 - threshold) return Vec2.X();
		if (vec.x < -1 + threshold) return Vec2.NEG_X();
		if (vec.y > 1 - threshold) return Vec2.Y();
		if (vec.y < -1 + threshold) return Vec2.NEG_Y();
		return vec;
	}

	function update(deltaTime: Duration) {
		const panAmount = app.moveSpeed / buddhabrot.zoomLevel * deltaTime.seconds;
		const zoomAmount = app.moveSpeed * deltaTime.seconds;

		let velocity = Vec2.new(0, 0);
		if (inputMap.up) velocity.y += 1;
		if (inputMap.down) velocity.y -= 1;
		if (inputMap.right) velocity.x += 1;
		if (inputMap.left) velocity.x -= 1;

		if (inputMap.zoomOut) buddhabrot.zoom += zoomAmount;
		if (inputMap.zoomIn) buddhabrot.zoom -= zoomAmount;
		
		velocity.rotate(buddhabrot.rotation);
		velocity.normalize();
		velocity = snapToCardinalDirection(velocity);
		velocity.scale(panAmount);

		const targetVector = ({
			[InputMode.Mandelbrot]: buddhabrot.viewCenter,
			[InputMode.Julia]: buddhabrot.initialZ,
			[InputMode.Exponent]: buddhabrot.exponent,
		})[buddhabrot.inputMode];

		targetVector.add(velocity);
	}

	let sidebarOpen = $state(true);

	let sidebarSection: "controls" | "rendering" = $state("controls");

	const deviceSupportsHover = new MediaQuery("(hover: hover)");
</script>
<main 
	style:--sidebar-width="450px"
	style:--sidebar-height="50%"
	class="fixed inset-0 bg-background overflow-hidden"
>
	<div class="
		absolute transition-all duration-300
		inset-0
		{sidebarOpen ? `
			bottom-(--sidebar-height) md:bottom-0
			md:left-(--sidebar-width)
		` : ""}
	">
		<!--<div
			class="absolute inset-0"
			bind:clientHeight={clientHeight}
			bind:clientWidth={clientWidth}
			{@attach (node)=>{
				node.appendChild(canvas);
				canvas.className = "absolute inset-0 w-full h-full bg-black object-fill";
			}}
		></div>-->
		<canvas 
			class="absolute inset-0 w-full h-full bg-black object-fill"
			bind:this={canvas}
			bind:clientWidth={clientDimensions.width}
			bind:clientHeight={clientDimensions.height}
		></canvas>

		{#if rendererInitError}
			<div class="
				absolute inset-0 p-3
				bg-black/50
				overflow-auto
				flex items-center justify-center
				text-white text-center
			">
				<span>
					Failed to initialize renderer.<br>
					This application requires a browser that supports WebGPU.<br>
					<br>
					See <a 
						href="https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API#Browser_compatibility" 
						class="underline text-primary-500"
						target="_blank"
					>here</a> for a list of compatible browsers.

					<hr class="my-4" >

					<span class="whitespace-pre-wrap">{rendererInitError.message}</span>
				</span>
			</div>
		{/if}

		<div
			class="absolute top-2 left-2 text-white gap-3 flex items-center"
			hidden={!reactive.app.isRecording}
		>
			<div class="bg-red-500 w-4 h-4 rounded-full animate-pulse"></div>
			<span>
				{reactive.app.timeline.frames.length} keyframes
			</span>
		</div>

		<div class="
			absolute top-0 left-0 flex flex-col gap-4 p-4
			w-min h-full
			hover:opacity-100 transition-opacity delay-50 duration-500
			{deviceSupportsHover.current ? "opacity-0" : ""}
		">
			<CircleButton 
				onPress={()=>(sidebarOpen = !sidebarOpen)}
				label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
			>
				{@html sidebarOpen ? fa5_solid_times : fa5_solid_bars}
			</CircleButton>
		</div>
	</div>
	
	<!-- Collapsible sidebar -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div 
		class="
			absolute bottom-0 left-0 bg-surface transition-transform duration-300
			w-full h-(--sidebar-height)
			md:w-(--sidebar-width) md:h-full
			grid grid-cols-[min-content_1fr]
			{sidebarOpen ? 
				`translate-x-0 translate-y-0` : 
				'translate-y-full md:translate-y-0 md:-translate-x-full'
			}
		"
		onkeydown={(e)=>{
			if (e.code === "Space" || e.code === "Shift") {
				e.preventDefault();
			}
		}}
	>
		<NavRail placement="left">
			<NavRailButton
				selected={sidebarSection === "controls"}
				onPress={() => sidebarSection = "controls"}
				label="Position"
				displayLabel={true}
			>
				{@html fa6_solid_upDownLeftRight}
			</NavRailButton>

			<NavRailButton
				selected={sidebarSection === "rendering"}
				onPress={() => sidebarSection = "rendering"}
				label="Display"
				displayLabel={true}
			>
				{@html fa5_solid_paintBrush}
			</NavRailButton>

			<NavRailSpacer />

			<NavRailButton
				label="Info"
				onPress={()=>location.hash = "#info"}
			>
				{@html fa5_solid_info}
			</NavRailButton>
			<a tabindex="-1" href="{githubRepositoryLink}" target="_blank">
				<NavRailButton label="GitHub" onPress={()=>{}}>
					{@html fa5_brands_github}
				</NavRailButton>
			</a>
		</NavRail>
		<div class="p-4 overflow-y-auto">
			{#if sidebarSection === "controls"}
				{@render controlSettings()}
			{:else if sidebarSection === "rendering"}
				{@render renderSettings()}
			{/if}
		</div>
	</div>
</main>

{#snippet controlSettings()}
	<!-- Input Mode -->
	<div class="mb-6">
		<h3 class="text-lg font-semibold mb-2">Input Mode</h3>
		
		<div class="grid grid-cols-3 gap-2 text-sm mb-4">
			{#each [
				{ name: 'Mandelbrot', mode: InputMode.Mandelbrot },
				{ name: 'Julia', mode: InputMode.Julia },
				{ name: 'X', mode: InputMode.Exponent },
			] as const as { name, mode } }
				<Button 
					onPress={() => reactive.buddhabrot.inputMode = mode}
					className="w-full p-2! rounded! transition-[background-color,color,outline-offset]!"
					variant={reactive.buddhabrot.inputMode === mode ? 'filled' : 'outlined'}
				>
					{name}
				</Button>
			{/each}
		</div>

		{#snippet kbd(text: string)}
			<kbd class="
				bg-surfaceContainer text-onSurfaceContainer rounded px-3 ml-1 font-mono
			">{text}</kbd>
		{/snippet}

		<div class="text-sm mb-3">
			<div class="flex items-center mb-1">
				Press {@render kbd("1")}, {@render kbd("2")}, or {@render kbd("3")} to switch modes
			</div>

			<div class="flex items-center mb-1">
				<div>
					Move {{
						[InputMode.Mandelbrot]: "Mandelbrot",
						[InputMode.Julia]: "Julia",
						[InputMode.Exponent]: "X"
					}[reactive.buddhabrot.inputMode]}
				</div>

				{@render kbd("W")}
				{@render kbd("D")}
				{@render kbd("A")}
				{@render kbd("S")}
			</div>

			<div class="flex items-center mb-1">
				Zoom In / Out

				{@render kbd("Shift")}
				{@render kbd("Space")}
			</div>

			<div class="flex items-center mb-1">
				<div>
					Adjust Speed
				</div>
				{@render kbd("]")}
				{@render kbd("[")}
			</div>
		</div>

		<div class="text-sm mb-3 font-mono bg-surfaceContainer p-2 rounded">
			z = p.z + p.w * i <span class="opacity-30">// Julia</span><br>
			c = p.x + p.y * i <span class="opacity-30">// Mandelbrot</span><br>
			e = p.v + p.u * i <span class="opacity-30">// X</span>
		</div>

		<div class="text-sm mb-3 font-mono bg-surfaceContainer p-2 rounded">
			z = z ^ e + c
		</div>

		<a class="text-primary-500 underline text-sm" href="#info">Mathematical Background</a>
	</div>

	<!-- Controls -->
	<div class="mb-6">
		<h3 class="text-lg font-semibold mb-2">Controls</h3>
		<div class="grid gap-2 mb-2">
			<NumberField 
				label="Speed"
				value={reactive.app.moveSpeed}
				onInput={e => reactive.app.moveSpeed = e.value}
			/>
		</div>
	</div>


	<!-- Camera -->
	<div class="mb-6">
		<h3 class="text-lg font-semibold mb-2">Orientation</h3>
		<div class="pl-(--radius-md)">
			Position
		</div>

		{@render vector({ vector: reactive.buddhabrot.viewCenter, readonly: false })}
		<div class="mb-2"></div>
		{@render vector({ vector: reactive.buddhabrot.initialZ, readonly: false })}
		<div class="mb-2"></div>
		{@render vector({ vector: reactive.buddhabrot.exponent, readonly: false })}
		<div class="mb-4"></div>

		<div class="grid grid-cols-2 gap-2">
			<NumberField 
				label="Zoom" 
				value={reactive.buddhabrot.zoom}
				onInput={e => reactive.buddhabrot.zoom = e.value}
				className="w-full"
			/>

			<NumberField 
				label="Rotation" 
				bind:value={
					()=>radToDeg(reactive.buddhabrot.rotation),
					v=>reactive.buddhabrot.rotation = degToRad(v)
				}
				className="w-full"
			/>
		</div>
	</div>
{/snippet}

{#snippet renderSettings()}
	<!-- Render settings -->
	<div class="mb-6">
		<h3 class="text-lg font-semibold mb-2">Indicators</h3>
		<div class="grid grid-cols-2 gap-2 mb-2">
			<NumberField 
				label="Z Indicator Size"
				value={buddhabrot.zIndicatorSize}
				onInput={e => buddhabrot.zIndicatorSize = e.value}
			/>
			<NumberField 
				label="E Indicator Size"
				value={buddhabrot.eIndicatorSize}
				onInput={e => buddhabrot.eIndicatorSize = e.value}
			/>
		</div>
		<SelectField 
			label="Show Indicator"
			className="mb-2"
			bind:value={
				()=>buddhabrot.zIndicatorSetting,
				(value)=> {
					buddhabrot.zIndicatorSetting = value;
					buddhabrot.eIndicatorSetting = value;
				}
			}
			options={[
				{ value: IndicatorSetting.Never, label: "Never" },
				{ value: IndicatorSetting.Always, label: "Always" },
				{ value: IndicatorSetting.WhenToolSelected, label: "When Plane Selected" },
			]}
		/>
	</div>

	<div class="mb-6">
		<h3 class="text-lg font-semibold mb-2">Resolution</h3>
		<div class="grid grid-cols-2 gap-2 mb-2">
			<TextField
				label="Width"
				bind:value={resolution.width}
			/>

			<TextField
				label="Height"
				bind:value={resolution.height}
			/>
		</div>
	</div>

	<div class="mb-6">
		<h3 class="text-lg font-semibold mb-2">Iteration Settings</h3>

		<div class="grid gap-2 grid-cols-3 mb-2">
			<NumberField 
				label="Red"
				value={buddhabrot.maxIterations1}
				onInput={e => buddhabrot.maxIterations1 = e.value}
			/>

			<NumberField 
				label="Green ~"
				bind:value={
					()=>buddhabrot.maxIterations2 / buddhabrot.maxIterations1,
					v=>buddhabrot.maxIterations2 = v * buddhabrot.maxIterations1
				}
			/>

			<NumberField 
				label="Blue ~"
				bind:value={
					()=>buddhabrot.maxIterations3 / buddhabrot.maxIterations1,
					v=>buddhabrot.maxIterations3 = v * buddhabrot.maxIterations1
				}
			/>
		</div>

		<div class="grid gap-2 mb-2">
			<NumberField 
				label="Bailout Radius"
				value={buddhabrot.bailoutRadius}
				onInput={e => buddhabrot.bailoutRadius = e.value}
			/>

			<NumberField 
				label="Samples Per Frame"
				value={buddhabrot.samples}
				onInput={e => buddhabrot.samples = e.value}
			/>

			<NumberField 
				label="Sample Radius"
				value={buddhabrot.sampleRadius}
				onInput={e => buddhabrot.sampleRadius = e.value}
			/>
		</div>
	</div>


	<div class="mb-6">
		<h3 class="text-lg font-semibold mb-2">Render Settings</h3>
		<div class="grid gap-2 mb-2">
			<NumberField
				label="Gamma"
				value={buddhabrot.gamma}
				onInput={e => buddhabrot.gamma = e.value}
			/>
			<NumberField 
				label="Frame Interpolation"
				value={buddhabrot.frameLerp}
				onInput={e => buddhabrot.frameLerp = e.value}
			/>
		</div>
	</div>
{/snippet}


{#snippet vector(options: { vector: Vec2, readonly: boolean})}
	<div class="grid grid-cols-2 gap-2">
		<NumberField
			label="X"
			hideLabel={true}
			readonly={options.readonly}
			value={options.vector.x}
			onInput={e => options.vector.x = e.value}
		/>
		<NumberField
			label="Y"
			hideLabel={true}
			readonly={options.readonly}
			value={options.vector.y}
			onInput={e => options.vector.y = e.value}
		/>
	</div>
{/snippet}

