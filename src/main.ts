import './main.css';

import MyApp from './app-ui/MyApp.svelte';
import { mount } from 'svelte';

// Fix iOS Safari :active styles.
document.documentElement.addEventListener("touchstart",()=>undefined);

// mount app
const element = document.querySelector(".SvelteOutlet")!;
mount(MyApp, { target: element });

if (import.meta.hot) {
	import.meta.hot.accept(() => {
		import.meta.hot!.invalidate();
	});
}