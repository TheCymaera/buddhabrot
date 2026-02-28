import { lerpNumber } from "./numbers.js";

export class Vec2 {
	static new(x: number, y: number) {
		return new Vec2(x, y);
	}

	constructor(public x: number, public y: number) { }

	add(other: Vec2) {
		this.x += other.x;
		this.y += other.y;
		return this;
	}

	subtract(other: Vec2) {
		this.x -= other.x;
		this.y -= other.y;
		return this;
	}

	scale(scalar: number) {
		this.x *= scalar;
		this.y *= scalar;
		return this;
	}

	rotate(angle: number) {
		const cosR = Math.cos(angle);
		const sinR = Math.sin(angle);
		const x = this.x * cosR - this.y * sinR;
		const y = this.x * sinR + this.y * cosR;
		this.x = x;
		this.y = y;
		return this;
	}

	lerp(other: Vec2, t: number) {
		this.x = lerpNumber(this.x, other.x, t);
		this.y = lerpNumber(this.y, other.y, t);
		return this;
	}

	clone() {
		return new Vec2(this.x, this.y);
	}
}