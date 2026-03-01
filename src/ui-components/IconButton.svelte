<script lang="ts">
	interface Props {
		disabled?: boolean,
		onPress: () => void,
		label: string,
		children?: import('svelte').Snippet,
		className?: string
	}

	let {
		disabled = false,
		onPress,
		label,
		children,
		className = ""
	}: Props = $props();
</script>
<button 
	onclick={onPress} 
	disabled={disabled}
	title={label}
	class="
	relative
	bg-transparent rounded-full
	flex flex-col items-center justify-center transition-color duration-200

	outline-offset-[calc(var(--outline-width)*-1)]
	disabled:opacity-30

	before:bg-inkWell 
	before:opacity-0
	disabled:before:opacity-0
	before:transition-opacity
	hover:before:opacity-10
	active:before:opacity-20

	cursor-pointer
	{className}
"
>
	{@render children?.()}
</button>
<style>
:root {
	--helion-IconButton-iconSize: 1em;
	--helion-IconButton-splashSize: 2.5em;
}

button {
	width: var(--helion-IconButton-iconSize);
	height: var(--helion-IconButton-iconSize);
}

button::before {
	content: "";

	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);

	width: var(--helion-IconButton-splashSize);
	height: var(--helion-IconButton-splashSize);
	border-radius: 50%;
}
</style>