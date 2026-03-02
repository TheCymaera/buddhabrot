<script lang="ts">
	interface Props {
		checked: boolean
		onChange: (event: { checked: boolean }) => void
		label: string
		readonly?: boolean
		className?: string
	}

	let { checked = $bindable(), onChange, label, readonly = false, className }: Props = $props();

	let pressed = $state(false);
	let pressTime = 0;
	const minPressTime = 200;
	function press() {
		pressed = true;
		pressTime = performance.now();
	}

	function unPress() {
		const elapsed = performance.now() - pressTime;
		if (elapsed > minPressTime) return void (pressed = false);
		setTimeout(() => pressed = false, minPressTime - elapsed);
	}
</script>

<svelte:window onpointerup={unPress} />

<helion-toggle-switch-field class={className}>
	<label 
		class="flex items-center gap-2 cursor-pointer mx-2 {readonly ? "pointer-events-none" : ""}"
		onpointerdown={press}
	>
		<input
			type="checkbox"
			class="absolute opacity-0"
			bind:checked={checked}
			onchange={() => onChange({ checked })}
			disabled={readonly}
		/>

		<helion-toggle-switch-track
			class:Checked={checked}
			class:Pressed={pressed}
			class:Disabled={readonly}
		>
			<helion-toggle-switch-splash></helion-toggle-switch-splash>
			<helion-toggle-switch-thumb></helion-toggle-switch-thumb>
		</helion-toggle-switch-track>

		<span>{label}</span>
	</label>
</helion-toggle-switch-field>


<style>
@layer base {
:root {
	--helion-ToggleSwitch-trackHeight: 1.5em;
	--helion-ToggleSwitch-trackWidth: 2.5em;
	
	--helion-ToggleSwitch-thumbSize: calc(var(--helion-ToggleSwitch-trackHeight) * .78);

	--helion-ToggleSwitch-splashSize: 3em;
	--helion-ToggleSwitch-splashOpacity: .15;

	--helion-ToggleSwitch-transition: .2s ease;

	--helion-ToggleSwitch-trackColor: var(--color-toggleSwitchTrackColor);
	--helion-ToggleSwitch-thumbColor: var(--color-toggleSwitchThumbColor);
}

helion-toggle-switch-field {
	display: block;
}

helion-toggle-switch-track {
	position: relative;
	height: var(--helion-ToggleSwitch-trackHeight);
	width: var(--helion-ToggleSwitch-trackWidth);
	box-sizing: border-box;

	border-radius: var(--helion-ToggleSwitch-trackHeight);

	background-color: var(--helion-ToggleSwitch-trackColor);
	transition: background-color var(--helion-ToggleSwitch-transition), opacity var(--helion-ToggleSwitch-transition);

	&.Disabled {
		opacity: 0.5;
	}

	&.Checked {
		background-color: var(--color-primary-500);
	}

	/* Translate children */
	> * {
		position: absolute;
		top: 50%;
		left: 0;
		translate: calc(var(--helion-ToggleSwitch-trackHeight) / 2 - 50%) -50%;
	}

	&.Checked > * {
		translate: calc(var(--helion-ToggleSwitch-trackWidth) - var(--helion-ToggleSwitch-trackHeight) / 2  - 50%) -50%;
	}

	/* Thumb */
	> helion-toggle-switch-thumb {
		width:	var(--helion-ToggleSwitch-thumbSize);
		height:	var(--helion-ToggleSwitch-thumbSize);
		transition: translate var(--helion-ToggleSwitch-transition);

		display: block;
		border-radius: 50%;

		background-color: var(--helion-ToggleSwitch-thumbColor);
	}

	/* Splash */
	> helion-toggle-switch-splash {
		width:	var(--helion-ToggleSwitch-splashSize);
		height:	var(--helion-ToggleSwitch-splashSize);
		opacity: 0;

		display: block;
		border-radius: 50%;

		scale: 0;

		--helion-ToggleSwitch-splashTransition: .5s ease-out;

		transition: 
			translate var(--helion-ToggleSwitch-transition), 
			scale var(--helion-ToggleSwitch-splashTransition),
			background-color var(--helion-ToggleSwitch-splashTransition), 
			opacity var(--helion-ToggleSwitch-splashTransition);

		background-color: var(--color-inkWell);
	}

	&.Checked > helion-toggle-switch-splash {
		background-color: var(--color-primary-500);
	}
}

input:active + * helion-toggle-switch-splash, .Pressed helion-toggle-switch-splash {
	--helion-ToggleSwitch-splashTransition: .2s ease-in-out;

	scale: 1;
	opacity: var(--helion-ToggleSwitch-splashOpacity);
}


input:focus-visible ~ helion-toggle-switch-track {
	outline: var(--outline-width) solid var(--color-primary-500);
	outline-offset: var(--outline-width);
}
}
</style>