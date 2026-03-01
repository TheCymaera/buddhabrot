<script lang="ts">
	import { generateElementId } from './ui-component-utilities.js';

	interface Props {
		value: string,
		type?: string,
		label?: string,
		placeholder?: string,
		readonly?: boolean,
		hint?: string,
		error?: string,
		className?: string,
		trailingIcon?: import('svelte').Snippet,
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
		trailingIcon,
		onInput = () => {}
	}: Props = $props();

	const id = generateElementId("text-field");
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
			class="
				w-full p-3 border-[.08rem] border-containerBorder rounded-md bg-transparent
				inset-outline
				outline-offset-[calc(var(--outline-width)*-1)]
				disabled:opacity-50
			"
			oninput={function () { value = this.value; onInput(value) }}
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