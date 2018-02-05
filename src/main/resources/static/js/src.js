var simpleBoard = new DrawingBoard.Board('simple-board', {
	controls: [
		{ DrawingMode: { filler: false } },
		'Navigation',
		'Download'
	],
	webStorage: 'local',
	size: 10
});
