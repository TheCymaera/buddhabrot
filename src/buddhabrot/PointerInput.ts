import { Vec2 } from "../open-utilities/Vec2.js";

export interface PinchGestureEvent {
	previous: [Vec2, Vec2],
	current: [Vec2, Vec2],
	zoomDelta: number,
	angleDelta: number,
	previousMidpoint: Vec2,
	currentMidpoint: Vec2,
}

export interface DragGestureEvent {
	current: Vec2,
	previous: Vec2,
	delta: Vec2
}

export class PointerInput {
	private readonly activePointers = new Map<number, Vec2>();

	pointerFilter = (event: PointerEvent) => {
		if (event.pointerType === 'mouse') return event.button === 0;
		return true;
	};

	onPinchGesture: (event: PinchGestureEvent) => void = () => { };

	onDragGesture: (event: DragGestureEvent) => void = () => { };

	constructor(canvas: HTMLCanvasElement) {
		canvas.addEventListener('pointerdown', (event) => this.onPointerDown(event));
		canvas.addEventListener('pointerup', (event) => this.onPointerEnd(event));
		canvas.addEventListener('pointercancel', (event) => this.onPointerEnd(event));
		canvas.addEventListener('pointermove', (event) => this.onPointerMove(event));
	}

	private onPointerDown(event: PointerEvent) {
		if (!this.pointerFilter(event)) return;

		const element = event.currentTarget as HTMLCanvasElement;

		event.preventDefault();
		element.setPointerCapture(event.pointerId);
		this.activePointers.set(event.pointerId, Vec2.new(event.clientX, event.clientY));
	}

	private onPointerEnd(event: PointerEvent) {
		const element = event.currentTarget as HTMLCanvasElement;

		this.activePointers.delete(event.pointerId);

		if (element.hasPointerCapture(event.pointerId)) {
			element.releasePointerCapture(event.pointerId);
		}
	}

	minPinchDistance = 8;
	private onPointerMove(event: PointerEvent) {
		// get previous pointers
		const previous = this.activePointers.get(event.pointerId);
		if (!previous) return;
		const previousPinch = coerceTuple2(this.activePointers.values());

		event.preventDefault();

		// get current pointers
		const current = Vec2.new(event.clientX, event.clientY);
		this.activePointers.set(event.pointerId, current);
		const currentPinch = coerceTuple2(this.activePointers.values());

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
			const delta = current.clone().subtract(previous);
			this.onDragGesture({ current, previous, delta });
		}
	}
}

function coerceTuple2<T>(items: Iterable<T>): [T, T] | undefined {
	const [first, second] = [...items];
	if (!first || !second) return undefined;
	return [first, second];
}