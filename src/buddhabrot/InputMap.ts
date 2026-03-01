import { infoWindowOpened } from "../app-ui/AppInfo.svelte";
import { InputMode } from "./Buddhabrot.js";

const keysPressed = new Set<string>();

export const inputMap = new class InputMap {
	get left() {
		return keysPressed.has('KeyA');
	}

	get right() {
		return keysPressed.has('KeyD');
	}

	get up() {
		return keysPressed.has('KeyW');
	}

	get down() {
		return keysPressed.has('KeyS');
	}

	get zoomIn() {
		return keysPressed.has('Space');
	}

	get zoomOut() {
		return keysPressed.has('ShiftLeft') || keysPressed.has('ShiftRight');
	}

	readonly listeners = {
		onInputModeChange: (mode: InputMode) => {},
		onHalfSpeed: () => {},
		onDoubleSpeed: () => {},
		onToggleRecording: () => {},
	}
}


addEventListener('keydown', (e) => {
	// ignore if input is focused
	if (document.activeElement instanceof HTMLInputElement ||
		document.activeElement instanceof HTMLTextAreaElement) {
		return;
	}

	// ignore if info page is open
	if (infoWindowOpened()) {
		return;
	}

	keysPressed.add(e.code);

	if (e.code === 'Digit1') {
		inputMap.listeners.onInputModeChange(InputMode.Mandelbrot);
	}
	
	if (e.code === 'Digit2') {
		inputMap.listeners.onInputModeChange(InputMode.Julia);
	}
	
	if (e.code === 'Digit3') {
		inputMap.listeners.onInputModeChange(InputMode.Exponent);
	}

	if (e.code === 'BracketLeft') {
		inputMap.listeners.onHalfSpeed();
	}

	if (e.code === 'BracketRight') {
		inputMap.listeners.onDoubleSpeed();
	}

	if (e.code === 'KeyR') {
		inputMap.listeners.onToggleRecording();
	}
});

addEventListener('keyup', (e) => {
	keysPressed.delete(e.code);
});