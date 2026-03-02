# API
Subject to change.

Run in the browser console.

## Record
1. Add `?record` to the URL to enable recording.
2. Press R to start/stop.

## Export Recording
```javascript
// settings
const directory = await showDirectoryPicker({ mode: "readwrite" });
const fps = 24;
const totalFrames = Math.ceil(app.timeline.duration().seconds * fps);
const samplesPerFrame = 2 ** 16;

// disable main loop
app.runMainLoop = false;
await renderer.onFinish();

// set resolution
renderer.setResolution({
	width: 512,
	height: 512,
});

for (let i = 0; i < totalFrames; i++) {
	const current = app.timeline.get(i / totalFrames);
	current.samples = samplesPerFrame;
	
	renderer.render(current);
	await renderer.onFinish();

	// write to file
	const handle = await directory.getFileHandle(`frame ${i}.png`, { create: true });
	const writable = await handle.createWritable();
	await writable.write(await renderer.getImageBlob());
	await writable.close();

	console.log(`Exported frame ${i + 1} / ${totalFrames}`);
}
```

## Custom Timeline
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

## High Quality Render
```javascript
// disable main loop
app.runMainLoop = false;
await renderer.onFinish();

// set resolution
renderer.setResolution({
	width: 2048,
	height: 2048,
});

const totalSamples = 2 ** 20;
buddhabrot.samples = 2 ** 17;
buddhabrot.maxIterations1 = 20_000;
buddhabrot.maxIterations2 = buddhabrot.maxIterations1 / 10;
buddhabrot.maxIterations3 = buddhabrot.maxIterations1 / 20;
buddhabrot.seed = Buddhabrot.createSeedGenerator();

// progressively render in stages to prevent
// the browser from freezing for too long.
const stages = Math.ceil(totalSamples / buddhabrot.samples);
for (let i = 0; i < stages; i++) {
	renderer.render(buddhabrot);
	await renderer.onFinish();

	console.log(`Rendered ${i + 1} / ${stages}`);
}

console.log(`Samples = ${totalSamples} (${stages} stages), Iterations = ${buddhabrot.maxIterations1}`);
```