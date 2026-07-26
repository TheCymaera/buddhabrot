<script lang="ts">
  import { generateId } from "./generateId.js";

	interface Props {
		type?: string,
		value: number,
		label?: string,
		hideLabel?: boolean,
		placeholder?: string,
		readonly?: boolean,
		hint?: string,
		error?: string,
		nullValue?: number,
		emptyIfNull?: boolean,
		className?: string,
		boxClassName?: string,
		onInput?: (event: { value: number}) => void,
		leadingIcon?: import('svelte').Snippet
		trailingIcon?: import('svelte').Snippet
	}

	let {
		type = "text",
		value = $bindable(),
		label = "",
		hideLabel = false,
		placeholder = "",
		readonly = false,
		hint = "",
		error = "",
		nullValue = 0,
		emptyIfNull = false,
		onInput = () => {},
		className = "",
		boxClassName = "",
		leadingIcon,
		trailingIcon
	}: Props = $props();

	let input: HTMLInputElement;
	$effect(() => {
		emptyIfNull; nullValue; value;
		
		if (!input) return
		if (document.activeElement === input) return
		
		input.value = (emptyIfNull && value === nullValue) ? "" : `${value}`
	});

	const id = generateId();
</script>

<helion-number-field class={className}>
	<label for={id} class="block {hideLabel ? "hidden" : ""}">
		{label}
	</label>
	<div class="grid *:[grid-area:1/1]">
		<input
			id={id}
			type={type}
			inputmode="decimal"
			class="
				helion-box-field
				{boxClassName}
				{className.includes("absolute") ? "absolute inset-0" : ""}
			"
			disabled={readonly} 
			placeholder="{placeholder || nullValue.toString()}"
			bind:this={input} 
			oninput={()=>{
				const parsed = parseFloat(input.value)
				value = isNaN(parsed) ? nullValue : parsed;
				onInput({ value });
			}}
		/>
		<div class="flex justify-end items-center pointer-events-none">
			{@render trailingIcon?.()}
		</div>
		<div class="flex justify-start items-center pointer-events-none">
			{@render leadingIcon?.()}
		</div>
	</div>
	<output class="block mt-0.5 pl-(--radius-md)" style:display={(error || hint) ? "" : "none"}>
		<small class="block whitespace-pre-wrap {error ? "text-red-500" : "opacity-80"}">{error || hint}</small>
	</output>
</helion-number-field>
<style>
	@layer base {
		helion-number-field {
			display: block;
		}
	}
</style>