import './main.css';

import MyApp from './app-ui/MyApp.svelte';
import { mount } from 'svelte';
mount(MyApp, {
	target: document.querySelector('.SvelteOutlet')!,
});

if (import.meta.hot) {
	import.meta.hot.accept(() => {
		import.meta.hot!.invalidate();
	});
}