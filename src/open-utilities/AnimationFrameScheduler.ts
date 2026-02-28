import { Duration } from "./Duration.js";

export class AnimationFrameScheduler {
	private static lastTime = performance.now();
	
	private static promise: Promise<Duration> | undefined = undefined;
	static next() {
		if (this.promise) return this.promise;

		return this.promise = new Promise<Duration>(resolve => {
			requestAnimationFrame((currentTime) => {
				this.promise = undefined;
				
				const deltaTime = (currentTime - this.lastTime);
				this.lastTime = currentTime;
				resolve(Duration.milliseconds(deltaTime));
			});
		});
	}

	static loop(callback: (deltaTime: Duration) => void) {
		(async () => {
			while (true) {
				const deltaTime = await this.next();
				callback(deltaTime);
			}
		})();
	}
}