<script lang="ts" generics="T">
	import { fa5_solid_angleDown } from "fontawesome-svgs";
	import { generateId } from "./generateId.js";

	let {
		value = $bindable(),
		label = "",
		readonly = false,
		options = [],
		hint = "",
		error = "",
		boxClassName = "",
		onChange = ()=>{},
		className = ""
	}: {
		value: T,
		label?: string,
		readonly?: boolean,
		options?: readonly { value: T, label: string, disabled?: boolean }[],
		hint?: string,
		error?: string,
		boxClassName?: string,
		onChange?: (event: { value: T })=>void,
		className?: string
	} = $props();

	const id = generateId();
</script>

<helion-select-field class={className}>
	<label for={id} class="block pl-1">{label}</label>
	<div class="relative">
		<select 
			id={id}
			class="helion-box-field pr-7 appearance-none {boxClassName}"
			bind:value={value}
			disabled={readonly}
			onchange={() => onChange({ value })}
		>
			{#each options as {value, label: text, disabled}}
				<option value={value} class="bg-paper text-onPaper" disabled={disabled}>
					{text}
				</option>
			{/each}
		</select>

		<div class="absolute right-0 top-0 h-full w-7 grid place-items-center pointer-events-none">
			{@html fa5_solid_angleDown}
		</div>
	</div>
	<output class="pl-1" style:display={(error || hint) ? "" : "none"}>
		<small class="whitespace-pre {error ? "text-red-500" : ""}">{error || hint}</small>
	</output>
</helion-select-field>

<style lang="postcss">
	@layer base {
		helion-select-field {
			display: block;
		}
	}
</style>