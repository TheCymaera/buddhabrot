<script lang="ts">
	interface Props { 
		stretched?: boolean,
		placement?: "bottom" | "top" | "left" | "right",
		children?: import('svelte').Snippet
	}

	let { 
		stretched = false,
		placement = "left",
		children
	}: Props = $props();
	
	let axis = $derived(placement === "left" || placement === "right" ? "vertical" : "horizontal");
</script>

<helion-nav-rail
	data-placement={placement}
	data-axis={axis}
	class:stretched={stretched}
>
	{@render children?.()}
</helion-nav-rail>
<style>
helion-nav-rail {
	--helion-NavRail-width: 65px;

	font: inherit;
	line-height: 100%;
	
	display: flex;
	overflow: hidden;
	
	user-select: none;
	-webkit-user-select: none;

	background: var(--color-navRail);
	color: var(--color-onNavRail);
}

helion-nav-rail.stretched > :global(*) {
	flex: 1;
}

helion-nav-rail.stretched > :global(.HelionNavRailSpacer) {
	display: none;
}

helion-nav-rail[data-axis=vertical] {
	flex-direction: column;
	height: 100%;
	width: calc(var(--helion-NavRail-width) + var(--helion-NavRail-padLeft) + var(--helion-NavRail-padRight));
	padding-left: var(--helion-NavRail-padLeft);
	padding-right: var(--helion-NavRail-padRight);
}

helion-nav-rail[data-axis=horizontal] {
	flex-direction: row;
	height: calc(var(--helion-NavRail-width) + var(--helion-NavRail-padBottom) + var(--helion-NavRail-padTop));
	width: 100%;
	padding-top: var(--helion-NavRail-padTop);
	padding-bottom: var(--helion-NavRail-padBottom);
}

helion-nav-rail[data-placement=left] {
	--helion-NavRail-padLeft: env(safe-area-inset-left);
}

helion-nav-rail[data-placement=right] {
	--helion-NavRail-padRight: env(safe-area-inset-right);
}

helion-nav-rail[data-placement=top] {
	--helion-NavRail-padTop: env(safe-area-inset-top);
}

helion-nav-rail[data-placement=bottom] {
	--helion-NavRail-padBottom: env(safe-area-inset-bottom);
}
</style>