<script lang="ts">
	import { fa5_solid_check, fa5_solid_minus } from "fontawesome-svgs";
    import { prolongActive } from "./prolongedActive.js";

	let {
		checked = $bindable(),
		intermediate = false,
		label,
		readonly = false,
		level = 0,
		onChange = () => {},
		hint = "",
		className = "",
	}: {
		checked: boolean,
		intermediate?: boolean,
		label: string,
		readonly?: boolean,
		level?: number,
		onChange?: (event: { checked: boolean }) => void,
		hint?: string,
		className?: string,
	} = $props();

	let active = $derived(checked || intermediate);

	let showSplash = $state(false);
</script>
<helion-checkbox-field 
	style:padding=".5em"
	style:padding-left={(level) + 1.0 + "rem"}
	class={className}
>
	<label class="flex items-center gap-2 cursor-pointer"
		style:pointer-events={readonly ? "none" : ""}
		{@attach prolongActive({
			callback: (value) => showSplash = value,
		})}
	>
		<input
			type="checkbox"
			class="helion-checkbox-input absolute opacity-0"
			disabled={readonly}
			bind:checked={checked}
			indeterminate={intermediate}
			onchange={()=>onChange({ checked })}
			{@attach prolongActive({
				callback: (value) => showSplash = value,
			})}
		/>
	
		<helion-checkbox-box 
			class="
				flex-none
				relative grid place-items-center h-5 w-5 rounded text-bg border-containerBorder
				{active && !readonly ? "bg-primary-500 border-transparent!" : ""}
				{active && readonly ? "bg-containerBorder border-transparent!" : ""}
			"
		>
			<helion-checkbox-tick class="grid place-items-center text-xs text-onPrimary" class:__checked={active}>
				{#if !intermediate}
					<span class="grid place-items-center mt-px">{@html fa5_solid_check}</span>
				{:else}
					{@html fa5_solid_minus}
				{/if}
			</helion-checkbox-tick>
	
			<helion-checkbox-splash class={active ? "bg-primary-500" : "bg-inkWell"} class:__shown={showSplash}></helion-checkbox-splash>
		</helion-checkbox-box>
	
		<span class="whitespace-pre">{label}</span>
	</label>
	{#if hint}
	<small style:padding-left={1.8 + "rem"}>
		{hint}
	</small>
	{/if}
</helion-checkbox-field>

<style>
:global {
@layer base {
	helion-checkbox-field {
		display: block;
	}

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

	helion-checkbox-tick.__checked {
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

	helion-checkbox-splash.__shown,
	.helion-checkbox-input:active + * helion-checkbox-splash {
		scale: 1;
	}


	.helion-checkbox-input:focus-visible + helion-checkbox-box {
		outline: var(--outline-width) solid var(--color-primary-500);
		outline-offset: var(--outline-width);
	}
}
}
</style>