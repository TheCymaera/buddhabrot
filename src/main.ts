import { AnimationFrameScheduler } from './open-utilities/AnimationFrameScheduler.js';
import { Buddhabrot } from './buddhabrot.js';
import { Duration } from './open-utilities/Duration.js';
import { inputMap } from './InputMap.js';
import './main.css';
import { Renderer } from './Renderer.js';
import { Timeline } from './Timeline.js';
import { Vec2 } from './open-utilities/Vec2.js';

const canvas = document.querySelector('canvas')!;
const buddhabrot = new Buddhabrot();
const renderer = await Renderer.create(canvas);

const app = new class App {
	moveSpeed = 1;
	doUpdates = true;

	readonly timeline = new Timeline();
	isRecording = false;

	lastFrameTime = performance.now();
}

Object.assign(globalThis, {
	buddhabrot,
	renderer,
	Vec2,
	Buddhabrot,
	AnimationFrameScheduler,
	Duration,
	Timeline,
	inputMap,
	app,
})

// handle canvas resize
new ResizeObserver(() => {
	if (!app.doUpdates) return;

	const dpr = window.devicePixelRatio || 1;
	renderer.setResolution({
		width: Math.floor(canvas.clientWidth * dpr),
		height: Math.floor(canvas.clientHeight * dpr),
	});
	renderer.render(buddhabrot);
}).observe(canvas);

// main loop
AnimationFrameScheduler.loop((deltaTime) => {
	if (!app.doUpdates) return;

	update(deltaTime);
	renderer.render(buddhabrot);

	if (app.isRecording) {
		app.timeline.addKeyframe({
			state: buddhabrot.clone(),
			duration: deltaTime,
		});
	}

	document.querySelector('#recordingIndicator')!.toggleAttribute('hidden', !app.isRecording);
	document.querySelector('#recordingIndicator_frameCount')!.textContent = String(app.timeline.frames.length);
});

// handle inputs
inputMap.listeners.onInputModeChange = (mode) => buddhabrot.inputMode = mode;
inputMap.listeners.onHalfSpeed = () => app.moveSpeed /= 2;
inputMap.listeners.onDoubleSpeed = () => app.moveSpeed *= 2;
inputMap.listeners.onToggleRecording = () => {
	app.isRecording = !app.isRecording;
	if (app.isRecording) {
		app.timeline.frames.length = 0;
	}
}

function update(deltaTime: Duration) {
	const panAmount = app.moveSpeed / buddhabrot.zoomLevel * deltaTime.seconds;
	const zoomAmount = app.moveSpeed * deltaTime.seconds;

	let didChange = false;
	const velocity = Vec2.new(0, 0);
	if (inputMap.up) {
		velocity.y += panAmount;
		didChange = true
	}
	if (inputMap.down) {
		velocity.y -= panAmount;
		didChange = true
	}
	if (inputMap.right) {
		velocity.x += panAmount;
		didChange = true
	}
	if (inputMap.left) {
		velocity.x -= panAmount;
		didChange = true
	}
	if (inputMap.zoomOut) {
		buddhabrot.zoom += zoomAmount;
		didChange = true
	}
	if (inputMap.zoomIn) {
		buddhabrot.zoom -= zoomAmount;
		didChange = true
	}
	
	velocity.rotate(buddhabrot.rotation);

	const targetVector =
		buddhabrot.inputMode === 'c' ? buddhabrot.viewCenter :
		buddhabrot.inputMode === 'z' ? buddhabrot.initialZ :
		buddhabrot.exponent;

	targetVector.add(velocity);

	if (didChange) {
		renderer.clearHistogram();
	}
}