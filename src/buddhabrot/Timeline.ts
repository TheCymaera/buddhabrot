import type { Buddhabrot } from "./Buddhabrot.js";
import { Duration } from "../open-utilities/Duration.js";

export interface Keyframe {
	time: number;
	state: Buddhabrot;
}


export class Timeline {
	constructor(
		readonly frames: Keyframe[] = []
	) { }

	duration() {
		if (this.frames.length === 0) return Duration.milliseconds(0);
		return Duration.milliseconds(this.frames[this.frames.length - 1]!.time);
	}

	addKeyframe({state, duration}: {state: Buddhabrot, duration: Duration}) {
		const previous = this.frames[this.frames.length - 1]!;
		if (!previous) {
			this.frames.push({ time: 0, state });
			return;
		}
		
		const currentTime = previous.time + duration.milliseconds;
		this.frames.push({ time: currentTime, state });

		// flatten timeline if possible
		if (this.frames.length >= 3) {
			const a = this.frames[this.frames.length - 3]!;
			const b = this.frames[this.frames.length - 2]!;
			const c = this.frames[this.frames.length - 1]!;

			if (a.state.equals(c.state) && b.state.equals(c.state)) {
				this.frames.splice(this.frames.length - 2, 1);
			}
		}
	}

	get(time: number | Duration) {
		const current = typeof time === "number" ? time * this.duration().milliseconds : time.milliseconds;

		if (this.frames.length === 0) return undefined;

		const first = this.frames[0]!;
		if (current <= first.time) return first.state.clone();

		const last = this.frames[this.frames.length - 1]!;
		if (current >= last.time) return last.state.clone();

		for (let i = 0; i < this.frames.length - 1; i++) {
			const start = this.frames[i]!;
			const end = this.frames[i + 1]!;

			if (current <= end.time) {
				const span = end.time - start.time;
				if (span <= 0) return end.state.clone();

				const t = (current - start.time) / span;
				return start.state.clone().lerp(end.state, t);
			}
		}

		return last.state.clone();
	}
}