//const prolongActiveMilliseconds = 200;


//if ("document" in globalThis) {
//	// Fix Safari :active
//	document.addEventListener("touchstart", () => {}, { passive: true });

//	// Prolonged active
//	document.addEventListener("pointerdown", (event) => {
//		const target = event.target as HTMLElement;
//		const button = target.closest("a, button, [role='button']");
//		if (!button) return;

//		button.toggleAttribute("data-prolonged-active", true);
//		setTimeout(() => button.toggleAttribute("data-prolonged-active", false), prolongActiveMilliseconds);
//	});

//	document.addEventListener("keydown", async (event) => {
//		if (event.repeat) return;

//		const button = document.activeElement as HTMLElement;
//		if (!button.matches("a, button, [role='button']")) return;
//		await new Promise(r => setTimeout(r, 0));
//		if (!button.matches(":active")) return;

//		button.toggleAttribute("data-prolonged-active", true);
//		setTimeout(() => button.toggleAttribute("data-prolonged-active", false), prolongActiveMilliseconds);
//	});
//}

export function prolongActive({
	milliseconds = 200,
	callback,
}: {
	milliseconds?: number,
	callback: (isActive: boolean) => void,
}) {
	return (element: HTMLElement) => {
		const abortController = new AbortController();

		element.addEventListener("pointerdown", ()=>{
			callback(true);
			setTimeout(() => callback(false), milliseconds);
		}, { signal: abortController.signal });

		element.addEventListener("keydown", async (event)=>{
			if (event.repeat) return;

			await new Promise(r => setTimeout(r, 0));
			if (!element.matches(":active")) return;

			callback(true);
			setTimeout(() => callback(false), milliseconds);
		}, { signal: abortController.signal });

		return ()=>abortController.abort();
	}
}