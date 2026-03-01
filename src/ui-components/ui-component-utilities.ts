let i = 0;
export function generateElementId(name: string): string {
	return `${name}-${i++}`;
}
