<script lang="ts">
import MainView from "./MainView.svelte";
import AppInfo from "./AppInfo.svelte";

let hash = $state(window.location.hash);
</script>
<svelte:window 
	onhashchange={()=> hash = window.location.hash}
/>

<svelte:boundary>
	<MainView />

	<div 
		class="
			fixed inset-0 bg-black/60 flex items-start justify-center p-4 z-50 overflow-y-auto
			transition-[opacity,visibility] transition-discrete
			{hash === "#info" ? "opacity-100 visible" : "opacity-0 invisible"}
		"
	>
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions --> 
		<div class="fixed z-0 inset-0" onclick={() => window.location.hash = ""}></div>

		<div class="relative z-10 w-full max-w-3xl">
			<AppInfo />
		</div>
	</div>

	{#snippet failed(error)}
		<div class="
			absolute inset-0 p-3
			bg-black/50
			overflow-auto
		">
			<div class="
				bg-surfaceContainer text-onSurfaceContainer 
				p-4 rounded-md shadow-md
				w-full max-w-min m-auto
				overflow-auto
			">
				<h1 class="text-xl font-bold">Render Error</h1>
				<p class="whitespace-pre">{error instanceof Error ? error.stack : `${error}`}</p>
			</div>
		</div>
	{/snippet}
</svelte:boundary>