export function lerpNumber(a: number, b: number, t: number) {
	return a * (1 - t) + b * t;
}

export function degToRad(degrees: number) {
	return degrees * Math.PI / 180;
}

export function radToDeg(radians: number) {
	return radians * 180 / Math.PI;
}