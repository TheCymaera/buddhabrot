<script lang="ts">
    import type { FullAutoFill } from "svelte/elements";
	import { generateId } from "./generateId.js";

	interface Props {
		value: string,
		type?: string,
		label?: string,
		placeholder?: string,
		readonly?: boolean,
		hint?: string,
		error?: string,
		className?: string,
		boxClassName?: string,
		trailingIcon?: import('svelte').Snippet,
		autocomplete?: FullAutoFill,
		onInput?: (value: string) => void
	}

	let {
		value = $bindable(),
		type = "text",
		label = "",
		placeholder = "",
		readonly = false,
		hint = "",
		error = "",
		className,
		boxClassName = "",
		trailingIcon,
		autocomplete,
		onInput = () => {}
	}: Props = $props();

	const id = generateId();
</script>

<helion-text-field class={className}>
	<label for={id} class="block pl-1">{label}</label>
	<div class="relative">
		<input 
			id={id}
			type={type} 
			value={value} 
			placeholder={placeholder} 
			disabled={readonly} 
			class="helion-box-field {boxClassName}"
			oninput={function () { value = this.value; onInput(value) }}
			autocomplete={autocomplete}
		/>
		<div class="absolute right-0 top-0 h-full grid">
			{@render trailingIcon?.()}
		</div>
	</div>
	<output class="pl-1" style:display={(error || hint) ? "" : "none"}>
		<small class="whitespace-pre {error ? "text-red-500" : ""}">{error || hint}</small>
	</output>
</helion-text-field>

<style>
	@layer base {
		helion-text-field {
			display: block;
		}
	}
</style>