<script lang="ts">
	import CheckboxFieldIntermediate from "./CheckboxFieldIntermediate.svelte";

	let {
		checked = $bindable(),
		label,
		readonly = false,
		level = 0,
		hint = "",
		className = "",
		onChange = () => {}
	}: {
		checked: boolean,
		label: string,
		readonly?: boolean,
		level?: number,
		hint?: string,
		className?: string,
		onChange?: (event: { checked: boolean }) => void
	} = $props();
</script>
<CheckboxFieldIntermediate bind:checked={checked} label={label} readonly={readonly} level={level} hint={hint} className={className} onChange={onChange} />