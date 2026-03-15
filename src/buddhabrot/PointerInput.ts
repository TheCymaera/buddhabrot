import { Vec2 } from "../open-utilities/Vec2"

export interface PinchGestureEvent {
	previous: [Vec2, Vec2]
	current: [Vec2, Vec2]
	zoomDelta: number
	angleDelta: number
	previousMidpoint: Vec2
	currentMidpoint: Vec2
}

export interface DragGestureEvent {
	current: Vec2
	previous: Vec2
}

export interface MouseWheelEvent {
	position: Vec2
	delta: Vec2
}

export class PointerInput {
	readonly #activePointers = new Map<number, Vec2>();
	readonly #abortController = new AbortController();

	pointerFilter = (event: PointerEvent) => {
		if (event.pointerType === 'mouse') return event.button === 0;
		return true;
	};

	onPinchGesture: (event: PinchGestureEvent) => void = () => { };

	onDragGesture: (event: DragGestureEvent) => void = () => { };

	onPointerCapture: (pointerId: number) => void = () => { };
	onPointerRelease: (pointerId: number) => void = () => { };
	onMouseWheelGesture: (event: MouseWheelEvent) => void = () => { };
	
	minPinchDistance = 8;

	constructor(canvas: HTMLCanvasElement) {
		canvas.style.touchAction = 'none';

		const signal = this.#abortController.signal;
		canvas.addEventListener('pointerdown', (event) => this.#onPointerDown(event), { signal });
		canvas.addEventListener('pointerup', (event) => this.#onPointerEnd(event), { signal });
		canvas.addEventListener('pointercancel', (event) => this.#onPointerEnd(event), { signal });
		canvas.addEventListener('pointermove', (event) => this.#onPointerMove(event), { signal });
		canvas.addEventListener('lostpointercapture', (event) => this.#onPointerEnd(event), { signal });
		canvas.addEventListener('wheel', (event) => this.#onWheel(event), { passive: false, signal });
	}

	destroy() {
		this.#abortController.abort();
	}

	#onWheel(event: WheelEvent) {
		event.preventDefault();
		const position = new Vec2(event.clientX, event.clientY);
		const delta = new Vec2(event.deltaX, event.deltaY);
		this.onMouseWheelGesture({ position, delta });
	}

	#onPointerDown(event: PointerEvent) {
		if (!this.pointerFilter(event)) return;

		const element = event.currentTarget as HTMLCanvasElement;

		event.preventDefault();
		element.setPointerCapture(event.pointerId);
		this.onPointerCapture(event.pointerId);
		this.#activePointers.set(event.pointerId, new Vec2(event.clientX, event.clientY));
	}

	#onPointerEnd(event: PointerEvent) {
		const element = event.currentTarget as HTMLCanvasElement;

		this.#activePointers.delete(event.pointerId);

		if (element.hasPointerCapture(event.pointerId)) {
			element.releasePointerCapture(event.pointerId);
		}
		
		this.onPointerRelease(event.pointerId);
	}

	#onPointerMove(event: PointerEvent) {
		// get previous pointers
		const previous = this.#activePointers.get(event.pointerId);
		if (!previous) return;
		const previousPinch = coerceTuple2(this.#activePointers.values());

		event.preventDefault();

		// get current pointers
		const current = new Vec2(event.clientX, event.clientY);
		this.#activePointers.set(event.pointerId, current);
		const currentPinch = coerceTuple2(this.#activePointers.values());

		if (previousPinch && currentPinch) {
			// pinch gesture
			const previousDistance = previousPinch[0].distanceTo(previousPinch[1]);
			const currentDistance = currentPinch[0].distanceTo(currentPinch[1]);

			if (previousDistance < this.minPinchDistance) return;
			if (currentDistance < this.minPinchDistance) return;

			const zoomDelta = Math.log2(currentDistance / previousDistance);
			const angleDelta = currentPinch[0].angleTo(currentPinch[1]) - 
				previousPinch[0].angleTo(previousPinch[1]);

			const previousMidpoint = previousPinch[0].clone().lerp(previousPinch[1], .5);
			const currentMidpoint = currentPinch[0].clone().lerp(currentPinch[1], .5);

			this.onPinchGesture({ previous: previousPinch, current: currentPinch, zoomDelta, angleDelta, previousMidpoint, currentMidpoint });
		} else {
			// drag gesture
			this.onDragGesture({ current, previous });
		}
	}
}

function coerceTuple2<T>(items: Iterable<T>): [T, T] | undefined {
	const [first, second] = [...items];
	if (!first || !second) return undefined;
	return [first, second];
}