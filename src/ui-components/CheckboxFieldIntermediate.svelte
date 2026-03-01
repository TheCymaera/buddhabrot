<script lang="ts">
	import { fa5_solid_check, fa5_solid_minus } from "fontawesome-svgs";

	let {
		checked = $bindable(),
		label,
		readonly = false,
		level = 0,
		onChange = () => {},
		hint = "",
		className = "",
	}: {
		checked: boolean | "indeterminate",
		label: string,
		readonly?: boolean,
		level?: number,
		onChange?: (event: { checked: boolean }) => void,
		hint?: string,
		className?: string,
	} = $props();

	let active = $derived(checked === true || checked === "indeterminate");

	let pressed = $state(false);
	let pressTime = 0;
	const minPressTime = 200;
	function press() {
		pressed = true;
		pressTime = performance.now();
	}

	function unPress() {
		const elapsed = performance.now() - pressTime;
		if (elapsed > minPressTime) return void (pressed = false);
		setTimeout(() => pressed = false, minPressTime - elapsed);
	}
</script>

<svelte:window onpointerup={unPress} />

<div 
	style:padding=".5em"
	style:padding-left={(level) + 1.0 + "rem"}
	class="{className}"
>
	<label class="flex items-center gap-2 cursor-pointer"
		style:pointer-events={readonly ? "none" : ""}
		class:Checked={active} 
		class:Pressed={pressed}
		onpointerdown={press}
	>
		<input type="checkbox" class="absolute opacity-0" disabled={readonly} bind:checked={checked as boolean} indeterminate={checked === "indeterminate"} onchange={()=>{
			if (checked === "indeterminate") return;
			onChange({ checked });
		}} />
	
		<helion-checkbox-box 
			class="
				flex-none
				relative grid place-items-center h-5 w-5 rounded text-bg border-containerBorder
				{active && !readonly ? "bg-primary-500 border-transparent!" : ""}
				{active && readonly ? "bg-containerBorder border-transparent!" : ""}
			"
		>
			<helion-checkbox-tick class="grid place-items-center text-xs text-onPrimary" class:Tick_checked={active}>
				{#if checked !== "indeterminate"}
					<span class="grid place-items-center mt-px">{@html fa5_solid_check}</span>
				{:else}
					{@html fa5_solid_minus}
				{/if}
			</helion-checkbox-tick>
	
			<helion-checkbox-splash class:Splash_shown={pressed} class={active ? "bg-primary-500" : "bg-inkWell"}></helion-checkbox-splash>
		</helion-checkbox-box>
	
		<span>{label}</span>
	</label>
	{#if hint}
	<small style:padding-left={1.8 + "rem"}>
		{hint}
	</small>
	{/if}
</div>

<style>
/* Box */
helion-checkbox-box {
	transition: background-color .2s, border-color .2s;
	border-width: .08em;
	z-index: 0;
}

/* Tick */
helion-checkbox-tick {
	opacity: 0;
	transition: opacity .2s;
}

.Tick_checked {
	opacity: 1;
}

/* Splash */
helion-checkbox-splash {
	position: absolute;
	left: 50%; 
	top: 50%;
	translate: -50% -50%;

	opacity: .1;
	width: 3em; 
	height: 3em;
	border-radius: 50%;

	scale: 0;
	transition: scale .2s;

	z-index: -1;
}

.Splash_shown,
input:active + * helion-checkbox-splash {
	scale: 1;
}


input:focus-visible + helion-checkbox-box {
	outline: var(--outline-width) solid var(--color-primary-500);
	outline-offset: var(--outline-width);
}
</style>