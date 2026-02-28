import { lerpNumber } from "./open-utilities/numbers.js";
import { Vec2 } from "./open-utilities/Vec2.js";

export class Buddhabrot {
	static readonly initialViewYSpan = 4.0;

	static seedGenerator(seed = 0) {
		return function next() {
			return ++seed;
		}
	}

	inputMode: "c" | "z" | "e" = "c";

	samples = 2 ** 12;
	maxIterations = 1000;
	minIterations = 0;
	escapeRadius = 4;

	sampleCenter = Vec2.new(0, 0);
	sampleRadius = 2.5;

	//viewYSpan = Buddhabrot.initialViewYSpan;
	zoom = 0;
	viewCenter = Vec2.new(0, 0);
	initialZ = Vec2.new(0, 0);
	exponent = Vec2.new(2, 0);
	
	rotation = Math.PI / 2;

	zIndicatorSize = 0.025;
	eIndicatorSize = 0.025;

	seed = Buddhabrot.seedGenerator();

	gamma = 4.0;
	histogramLerp = 1;

	get zoomLevel() {
		return Math.pow(2, this.zoom);
	}

	viewYSpan() {
		return Buddhabrot.initialViewYSpan / this.zoomLevel;
	}

	clone() {
		return new Buddhabrot().copy(this);
	}

	lerp(other: Buddhabrot, t: number) {
		this.zoom = lerpNumber(this.zoom, other.zoom, t);
		this.viewCenter.lerp(other.viewCenter, t);
		this.rotation = lerpNumber(this.rotation, other.rotation, t);
		this.initialZ.lerp(other.initialZ, t);
		this.exponent.lerp(other.exponent, t);
		return this;
	}

	equals(other: Buddhabrot) {
		return JSON.stringify(this) === JSON.stringify(other);
	}

	copy(other: Buddhabrot) {
		Object.assign(this, other);
		for (const key of Object.keys(this) as (keyof Buddhabrot)[]) {
			const value = this[key];
			if (value instanceof Vec2) {
				// @ts-expect-error - TS is wrong
				this[key] = value.clone();
			}
		}
		return this;
	}
}