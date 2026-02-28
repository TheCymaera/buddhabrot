/**
 * Duration
 */
 export class Duration {
	//private constructor({milliseconds = 0, seconds = 0, minutes = 0, hours = 0, days = 0, weeks = 0}) {
	//	this.milliseconds = (
	//		milliseconds +
	//		seconds * Duration.millisecondsPerSecond +
	//		minutes * Duration.millisecondsPerMinute +
	//		hours * Duration.millisecondsPerHour +
	//		days * Duration.millisecondsPerDay +
	//		weeks * Duration.millisecondsPerWeek
	//	);
	//}
	private constructor(readonly milliseconds: number) { }

	get seconds() {
		return this.milliseconds / Duration.millisecondsPerSecond;
	}

	get minutes() {
		return this.milliseconds / Duration.millisecondsPerMinute;
	}

	get hours() {
		return this.milliseconds / Duration.millisecondsPerHour;
	}

	get days() {
		return this.milliseconds / Duration.millisecondsPerDay;
	}

	get weeks() {
		return this.milliseconds / Duration.millisecondsPerWeek;
	}

	get weeksPart() {
		return Math.floor(this.milliseconds / Duration.millisecondsPerWeek);
	}

	get daysPart() {
		return Math.floor(this.milliseconds / Duration.millisecondsPerDay);
	}

	get hoursPart() {
		return Math.floor((this.milliseconds % Duration.millisecondsPerDay) / Duration.millisecondsPerHour);
	}

	get minutesPart() {
		return Math.floor((this.milliseconds % Duration.millisecondsPerHour) / Duration.millisecondsPerMinute);
	}

	get secondsPart() {
		return Math.floor((this.milliseconds % Duration.millisecondsPerMinute) / Duration.millisecondsPerSecond);
	}

	get millisecondsPart() {
		return this.milliseconds % Duration.millisecondsPerSecond;
	}

	abs() {
		return Duration.milliseconds(Math.abs(this.milliseconds));
	}

	add(duration: Duration) {
		return Duration.milliseconds(this.milliseconds + duration.milliseconds);
	}

	subtract(duration: Duration) {
		return Duration.milliseconds(this.milliseconds - duration.milliseconds);
	}

	scale(scale: number) {
		return Duration.milliseconds(this.milliseconds * scale);
	}

	toString() {
		const abs = this.abs();
		return `${this.daysPart}:${abs.hoursPart}:${abs.minutesPart}:${abs.secondsPart}:${abs.millisecondsPart}`;
	}

	static readonly millisecondsPerSecond = 1000;
	static readonly millisecondsPerMinute = Duration.millisecondsPerSecond * 60;
	static readonly millisecondsPerHour   = Duration.millisecondsPerMinute * 60;
	static readonly millisecondsPerDay    = Duration.millisecondsPerHour * 24;
	static readonly millisecondsPerWeek   = Duration.millisecondsPerDay * 7;

	static milliseconds(milliseconds: number) {
		return new Duration(milliseconds);
	}

	static seconds(seconds: number) {
		return new Duration(seconds * Duration.millisecondsPerSecond);
	}

	static minutes(minutes: number) {
		return new Duration(minutes * Duration.millisecondsPerMinute);
	}

	static hours(hours: number) {
		return new Duration(hours * Duration.millisecondsPerHour);
	}

	static days(days: number) {
		return new Duration(days * Duration.millisecondsPerDay);
	}

	static weeks(weeks: number) {
		return new Duration(weeks * Duration.millisecondsPerWeek);
	}
}