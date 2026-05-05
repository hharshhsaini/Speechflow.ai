import terser from "@rollup/plugin-terser";
import { nodeResolve } from "@rollup/plugin-node-resolve";

const plugins = (browser) => [nodeResolve({ browser }), terser({ format: { comments: false } })];

const OUTPUT_CONFIGS = [
  // Node versions
  {
    file: "./dist/speechflow.cjs",
    format: "cjs",
  },
  {
    file: "./dist/speechflow.js",
    format: "esm",
  },

  // Web version
  {
    file: "./dist/speechflow.web.js",
    format: "esm",
  },
];

const WEB_SPECIFIC_CONFIG = {
  onwarn: (warning, warn) => {
    if (!warning.message.includes("@huggingface/transformers")) warn(warning);
  },
};

const NODE_SPECIFIC_CONFIG = {
  external: ["@huggingface/transformers", "phonemizer"],
};

export default OUTPUT_CONFIGS.map((output) => {
  const web = output.file.endsWith(".web.js");
  return {
    input: "./src/speechflow.js",
    output,
    plugins: plugins(web),
    ...(web ? WEB_SPECIFIC_CONFIG : NODE_SPECIFIC_CONFIG),
  };
});
