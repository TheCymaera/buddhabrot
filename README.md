# API
Subject to change.

## Record
Press R to start/stop.

## Export Recording
```javascript
const directory = await showDirectoryPicker({ mode: "readwrite" });

// disable automatic updates
app.doUpdates = false;
await renderer.onFinish();

// configure
renderer.setResolution({
	width: 512,
	height: 512,
});

const FPS = 24;
const TOTAL_FRAMES = Math.ceil(app.timeline.duration().seconds * FPS);

let previous = app.timeline.get(0);
for (let i = 0; i < TOTAL_FRAMES; i++) {
	const current = app.timeline.get(i / TOTAL_FRAMES);
	current.samples = 2 ** 16;

	let didClear = false;
	if (!current.equals(previous)) {
		renderer.clearHistogram();
		didClear = true;
	}
	previous = current;
	
	renderer.render(current);
	await renderer.onFinish();

	// write to file
	const handle = await directory.getFileHandle(`frame ${i}.png`, { create: true });
	const writable = await handle.createWritable();
	await writable.write(await renderer.getImageBlob());
	await writable.close();

	console.log(`Exported frame ${i + 1} / ${TOTAL_FRAMES} ${didClear ? "(cleared histogram)" : ""}`);
}
```

# Custom Timeline
```javascript
const timeline = new Timeline();
timeline.addKeyframe({
	state: new Buddhabrot(),
	duration: Duration.seconds(1),
});

timeline.addKeyframe({
	state: new Buddhabrot(),
	duration: Duration.seconds(1),
});
```

# High Quality Render
```javascript
// disable automatic updates
app.doUpdates = false;
renderer.clearHistogram();
await renderer.onFinish();

// configure
renderer.setResolution({
	width: 2048,
	height: 2048,
});

const totalSamples = 2 ** 20;
buddhabrot.samples = 2 ** 17;
buddhabrot.maxIterations = 20_000;

buddhabrot.inputMode = "c"
buddhabrot.seed = Buddhabrot.seedGenerator();

// progressively render in stages to prevent
// the browser from freezing for too long.
const stages = Math.ceil(totalSamples / buddhabrot.samples);
for (let i = 0; i < stages; i++) {
	renderer.render(buddhabrot);
	await renderer.onFinish();

	console.log(`Rendered ${i + 1} / ${stages}`);
}

console.log(`Samples = ${totalSamples} (${stages} stages), Iterations = ${buddhabrot.maxIterations}`);
```