
<script lang="ts">
	import { generateElementId } from "./ui-component-utilities.js";

	let {
		value = $bindable(),
		min = 0,
		max = 1,
		step = 0.01,
		disabled = false,
		label = "",
		hideLabel = false,
		hint = "",
		error = "",
		className = ""
	}: {
		value: number,
		min?: number,
		max?: number,
		step?: number,
		disabled?: boolean,
		label?: string,
		hideLabel?: boolean,
		hint?: string,
		error?: string,
		className?: string
	} = $props();

	const id = generateElementId('range-slider');
</script>

<div class={className}>
	<label class="pl-(--radius-md)" style:display={hideLabel ? 'none' : ''} for={id}>
		{label}
	</label>
	<div class="inputContainer">
		<input
			id={id}
			type="range"
			class="helion-slider"
			bind:value
			min={min}
			max={max}
			step={step}
			disabled={disabled}
		/>
	</div>
	<output class="pl-(--radius-md)" style:display={(error || hint) ? '' : 'none'}>
		<small class={error ? 'text-red-500' : 'opacity-80'}>{error || hint}</small>
	</output>
</div>
<style>
.inputContainer {
  display: grid;
}
.inputContainer > * {
  grid-area: 1 / 1;
}

input {
	margin: 7px 0;
	border: none;
	background-color: transparent;
	-webkit-appearance: none;
	-moz-appearance: none;
	appearance: none;
	outline: none;
	width: 100%;

	&:disabled {
		opacity: .7;
	}
	&:not(:disabled) {
		cursor: pointer;
	}
	&::-webkit-slider-runnable-track {
		width: 100%;
		height: 7px;
		border-radius: 7px;
		background: var(--color-surfaceContainer);
		border: none;
	}
	&::-moz-range-track {
		width: 100%;
		height: 7px;
		border-radius: 7px;
		background: var(--color-surfaceContainer);
		border: none;
	}
	&:hover::-moz-range-track {
		background: var(--color-surfaceContainer);
	}
	&:hover::-webkit-slider-runnable-track {
		background: var(--color-surfaceContainer);
	}
	&::-webkit-slider-thumb {
		margin-top: -7.5px;
		-webkit-appearance: none;
		width: 22px;
		height: 22px;
		border-radius: 50%;
		background: var(--color-onSurface);
		border: .2em solid var(--color-surface);
		box-sizing: border-box;
		transition: background-color .1s ease,box-shadow .1s ease;
	}
	&::-moz-range-thumb {
		width: 22px;
		height: 22px;
		border-radius: 50%;
		background: var(--color-onSurface);
		border: .2em solid var(--color-surface);
		box-sizing: border-box;
		transition: background-color .1s ease,box-shadow .1s ease;
	}
	&:hover:not(:disabled)::-webkit-slider-thumb {
		background: var(--color-onSurface);
	}
	&:hover:not(:disabled)::-moz-range-thumb {
		background: var(--color-onSurface);
	}
	&:is(:active,:focus-visible):not(:disabled)::-webkit-slider-thumb {
		background: var(--color-primary-500);
		box-shadow: 0 0 0 .15em var(--color-primary-500);
	}
	&:is(:active,:focus-visible):not(:disabled)::-moz-range-thumb {
		background: var(--color-primary-500);
		box-shadow: 0 0 0 .15em var(--color-primary-500);
	}
}
</style>