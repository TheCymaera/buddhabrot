<script lang="ts" module>
	import { fa5_solid_times } from 'fontawesome-svgs';
	import Button from '../ui-components/Button.svelte';
	import { githubRepositoryLink, homeLink, minecraftYouTubeCode } from './links.js';
	import IconButton from '../ui-components/IconButton.svelte';

	export function infoWindowOpened() {
		return window.location.hash === "#info";
	}
</script>

<div class="bg-surface text-onSurface/80 p-6 rounded-lg text-pretty">
	<h1 class="text-2xl text-onSurface font-bold mb-4 pr-7">Buddhabrot Explorer</h1>


	<IconButton 
		className="absolute! top-6 right-6 text-xl"
		label="Close"
		onPress={() => window.location.hash = ""}
	>
		{@html fa5_solid_times}
	</IconButton>


	<div class="mb-4">
		The Buddhabrot (Nebulabrot) displays the trajectory of points that escape the Mandelbrot set. It resembles the Seated Buddha when rotated 90 degrees.
	</div>


	<h2 class="text-xl text-onSurface font-bold mb-2">How to render a Buddhabrot</h2>
	<div class="mb-4">
		<ol class="list-decimal ml-6 mb-4">
			<li>
				Randomly select points on the complex plane.
			</li>
			<li>
				Iterate each sample point <code>c</code> by the Mandelbrot recurrence to see whether it escapes to infinity:
				<code class="block p-3 mb-1">
					z = z ^ e + c
				</code>
			</li>
			<li>
				If a point escapes (i.e. it is not in the Mandelbrot set), iterate a second time and increment the brightness of each pixel visited by <code>z</code>.
			</li>
		</ol>
	</div>


	<h2 class="text-xl text-onSurface font-bold mb-2">Customization</h2>
	<div class="mb-4">
		To create multicolored Buddhabrots, we can use a different max iteration value for the red, green, and blue channels.
	</div>

	<div class="mb-4">
		An Anti-Buddhabrot is rendered by tracing the trajectories of points that do not escape, instead of those that do.
	</div>

	<div class="mb-4">
		The parameters Julia (<code>z₁</code>) and X (<code>e</code>) were inspired by <a href="https://www.youtube.com/watch?v=Ed1gsyxxwM0" target="_blank" class="text-primary-500 hover:underline">2swap's</a> 6D generalization of the Mandelbrot Set.
	</div>


	<h2 class="text-xl text-onSurface font-bold mb-2">Technical</h2>
	<div class="mb-4">
		This app uses the experimental WebGPU API to compute thousands of trajectories in parallel via a compute shader.
	</div>

	<h2 class="text-xl text-onSurface font-bold mb-2">Similar Projects</h2>
	<div class="mb-6">
		<a
			href="https://heledron.com/misc/mandelbrot/"
			target="_blank" 
			class="text-primary-500 hover:underline block mb-2"
		>
			6D Mandelbrot Explorer
		</a>

		<a 
			href="https://youtube.com/watch?v={minecraftYouTubeCode}"
			target="_blank" 
			class="text-primary-500 hover:underline block mb-2"
		>
			6D Mandelbrot in Vanilla Minecraft
		</a>
	</div>


	<h2 class="text-xl text-onSurface font-bold mb-2">Source Code</h2>
	
	<div class="mb-6">
		<a 
			href="{githubRepositoryLink}" 
			target="_blank" 
			class="text-primary-500 hover:underline"
		>
			{githubRepositoryLink}
		</a>
	</div>

	
	<div class="mb-6">
		<h2 class="text-xl text-onSurface font-bold mb-2">Other Tools</h2>
		<a 
			href="{homeLink}" 
			target="_blank" 
			class="text-primary-500 hover:underline"
		>
			{homeLink}
		</a>
	</div>
	
	<div class="mt-6 text-right">
		<Button 
			variant="filled" 
			onPress={() => window.location.hash = ""}
		>
			Close
		</Button>
	</div>
</div>

<style>
@layer base {
	code {
		background-color: var(--color-surfaceContainer);
		padding: 0em .25em;
	}
}
</style>