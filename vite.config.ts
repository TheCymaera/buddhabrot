import * as vite from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

export default vite.defineConfig({
	base: "./",

	plugins: [
		tailwindcss(),
		svelte(),
	],
});