<script lang="ts" generics="T">
	import { fa5_solid_angleDown } from "fontawesome-svgs";

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
		options?: readonly { value: T, label: string }[],
		hint?: string,
		error?: string,
		boxClassName?: string,
		onChange?: (event: { value: T })=>void,
		className?: string
	} = $props();
</script>

<label class="flex flex-col {className}">
	<div class="pl-1">{label}</div>
	<div class="relative">
		<select 
			class="
				flex-1 w-full px-3 p-2 pr-6! border-[.08rem] border-containerBorder rounded-md bg-transparent
				outline-offset-[calc(var(--outline-width)*-1)]
				disabled:opacity-50
				appearance-none
				{boxClassName}
			" 
			bind:value={value}
			disabled={readonly}
			onchange={() => onChange({ value })}
		>
			{#each options as {value, label: text}}
				<option value={value} class="bg-surfaceContainer text-onSurfaceContainer">
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
</label>