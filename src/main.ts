import './main.css';

import MyApp from './app-ui/Router.svelte';
import { mount } from 'svelte';

// Fix iOS Safari :active styles.
document.documentElement.addEventListener("touchstart",()=>undefined);

// mount app
const element = document.querySelector(".svelte-outlet")!;
mount(MyApp, { target: element });

if (import.meta.hot) {
	import.meta.hot.accept(() => {
		import.meta.hot!.invalidate();
	});
}