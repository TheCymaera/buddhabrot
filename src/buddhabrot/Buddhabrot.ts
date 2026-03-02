import { lerpNumber } from "../open-utilities/numbers.js";
import { Vec2 } from "../open-utilities/Vec2.js";

export enum InputMode {
	Mandelbrot = "Mandelbrot",
	Julia = "Julia",
	Exponent = "Exponent",
}

export enum IndicatorSetting {
	Never = "Never",
	Always = "Always",
	WhenToolSelected = "WhenToolSelected",
}

export class Buddhabrot {
	static readonly initialViewYSpan = 4.0;

	static createSeedGenerator(seed = 0) {
		return function next() {
			return ++seed;
		}
	}

	anti = false;

	inputMode: InputMode = InputMode.Mandelbrot;

	samples = 2 ** 12;
	uniformSampleDistribution = true;
	maxIterations1 = 1000;
	maxIterations2 = this.maxIterations1 / 10;
	maxIterations3 = this.maxIterations1 / 100;
	minIterations = 0;
	bailoutRadius = 4;

	sampleCenter = Vec2.new(0, 0);
	sampleRadius = 2.5;

	zoom = 0;
	viewCenter = Vec2.new(0, 0);
	initialZ = Vec2.new(0, 0);
	exponent = Vec2.new(2, 0);
	
	rotation = Math.PI / 2;

	zIndicatorSize = 0.025;
	eIndicatorSize = 0.025;
	zIndicatorSetting = IndicatorSetting.WhenToolSelected;
	eIndicatorSetting = IndicatorSetting.WhenToolSelected;

	seed = Buddhabrot.createSeedGenerator();

	gamma = 4.0;
	frameLerp = 0;

	normalizationFloor = 15;

	get zoomLevel() {
		return Math.pow(2, this.zoom);
	}

	get viewYSpan() {
		return Buddhabrot.initialViewYSpan / this.zoomLevel;
	}

	get effectiveZIndicatorSize() {
		if (this.zIndicatorSetting === IndicatorSetting.Never) return 0;
		if (this.zIndicatorSetting === IndicatorSetting.Always) return this.zIndicatorSize;
		return this.inputMode === InputMode.Julia ? this.zIndicatorSize : 0;
	}

	get effectiveEIndicatorSize() {
		if (this.eIndicatorSetting === IndicatorSetting.Never) return 0;
		if (this.eIndicatorSetting === IndicatorSetting.Always) return this.eIndicatorSize;
		return this.inputMode === InputMode.Exponent ? this.eIndicatorSize : 0;
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

	equals(other: unknown) {
		if (!(other instanceof Buddhabrot)) return false;

		return JSON.stringify(this) === JSON.stringify(other);
	}

	canReuseHistogram(other: Buddhabrot) {
		return (
			this.anti === other.anti &&
			this.uniformSampleDistribution === other.uniformSampleDistribution &&
			this.zoom === other.zoom &&
			this.viewCenter.equals(other.viewCenter) &&
			this.rotation === other.rotation &&
			this.initialZ.equals(other.initialZ) &&
			this.exponent.equals(other.exponent) &&
			this.bailoutRadius === other.bailoutRadius &&
			this.maxIterations1 === other.maxIterations1 &&
			this.maxIterations2 === other.maxIterations2 &&
			this.maxIterations3 === other.maxIterations3
		);
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