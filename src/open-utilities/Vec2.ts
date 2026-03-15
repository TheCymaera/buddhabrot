import { lerpNumber } from "./numbers.js";

export class Vec2 {
	static new(x: number, y: number) {
		return new Vec2(x, y);
	}

	static X() {
		return new Vec2(1, 0);
	}

	static Y() {
		return new Vec2(0, 1);
	}

	static NEG_X() {
		return new Vec2(-1, 0);
	}

	static NEG_Y() {
		return new Vec2(0, -1);
	}

	constructor(public x: number, public y: number) { }

	copy(other: Vec2) {
		this.x = other.x;
		this.y = other.y;
		return this;
	}

	angleTo(other: Vec2) {
		return Math.atan2(other.y - this.y, other.x - this.x);
	}

	distanceTo(other: Vec2) {
		const dx = this.x - other.x;
		const dy = this.y - other.y;
		return Math.sqrt(dx * dx + dy * dy);
	}

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

	equals(other: unknown) {
		if (!(other instanceof Vec2)) return false;
		return this.x === other.x && this.y === other.y;
	}

	normalize() {
		const length = Math.sqrt(this.x * this.x + this.y * this.y);
		if (length === 0) return undefined;

		this.x /= length;
		this.y /= length;
		return this;
	}

	toString() {
		return `Vec2(${this.x}, ${this.y})`;
	}
}