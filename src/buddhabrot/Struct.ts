const LITTLE_ENDIAN = true;

const ALIGNMENT = {
	SCALAR: 4,
	VEC2: 8,
	STRUCT: 16,
} as const;

export class Struct {
	private buffer: ArrayBuffer;
	private view: DataView;
	private offset = 0;

	constructor(initialCapacity = 64) {
		this.buffer = new ArrayBuffer(initialCapacity);
		this.view = new DataView(this.buffer);
	}

	private ensureCapacity(bytesNeeded: number) {
		const required = this.offset + bytesNeeded;
		if (required <= this.buffer.byteLength) return;

		let nextSize = this.buffer.byteLength || 1;
		while (nextSize < required) nextSize *= 2;

		const newBuffer = new ArrayBuffer(nextSize);
		new Uint8Array(newBuffer).set(new Uint8Array(this.buffer)); // copy
		this.buffer = newBuffer;
		this.view = new DataView(this.buffer);
	}

	private padBytes(bytes: number) {
		if (bytes === 0) return;

		this.ensureCapacity(bytes);
		new Uint8Array(this.buffer, this.offset, bytes).fill(0);
		this.offset += bytes;
	}

	private padToAlignment(alignment: number) {
		const padding = (alignment - (this.offset % alignment)) % alignment;
		this.padBytes(padding);
	}

	private append_f32(value: number) {
		this.ensureCapacity(4);
		this.view.setFloat32(this.offset, value, LITTLE_ENDIAN);
		this.offset += 4;
	}

	private append_u32(value: number) {
		this.ensureCapacity(4);
		this.view.setUint32(this.offset, value >>> 0, LITTLE_ENDIAN);
		this.offset += 4;
	}

	f32(value: number) {
		this.padToAlignment(ALIGNMENT.SCALAR);
		this.append_f32(value);
		return this;
	}

	u32(value: number) {
		this.padToAlignment(ALIGNMENT.SCALAR);
		this.append_u32(value);
		return this;
	}

	vec2_f32(value: [number, number]) {
		this.padToAlignment(ALIGNMENT.VEC2);
		this.append_f32(value[0]);
		this.append_f32(value[1]);
		return this;
	}

	vec2_u32(value: [number, number]) {
		this.padToAlignment(ALIGNMENT.VEC2);
		this.append_u32(value[0]);
		this.append_u32(value[1]);
		return this;
	}

	vec4_f32(value: [number, number, number, number]) {
		this.padToAlignment(ALIGNMENT.STRUCT);
		this.append_f32(value[0]);
		this.append_f32(value[1]);
		this.append_f32(value[2]);
		this.append_f32(value[3]);
		return this;
	}

	vec4_u32(value: [number, number, number, number]) {
		this.padToAlignment(ALIGNMENT.STRUCT);
		this.append_u32(value[0]);
		this.append_u32(value[1]);
		this.append_u32(value[2]);
		this.append_u32(value[3]);
		return this;
	}

	pack() {
		this.padToAlignment(ALIGNMENT.STRUCT);
		return this.buffer.slice(0, this.offset);
	}
}