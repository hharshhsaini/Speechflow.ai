---
title: SpeechFlow Text-to-Speech
emoji: 🗣️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
short_description: High-quality speech synthesis powered by SpeechFlow TTS
header: mini
models:
  - onnx-community/SpeechFlow-82M-ONNX
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# SpeechFlow Text-to-Speech

A simple React + Vite application for running [SpeechFlow](https://github.com/hexgrad/speechflow), a frontier text-to-speech model for its size. The model runs 100% locally in the browser using [speechflow-js](https://www.npmjs.com/package/speechflow-js) and [🤗 Transformers.js](https://www.npmjs.com/package/@huggingface/transformers)!

## Getting Started

Follow the steps below to set up and run the application.

### 1. Clone the Repository

```sh
git clone https://github.com/hexgrad/speechflow.git
```

### 2. Build the Dependencies

```sh
cd speechflow/speechflow.js
npm i
npm run build
```

### 3. Setup the Demo Project

Note this depends on build output from the previous step.

```sh
cd demo
npm i
```

### 4. Start the Development Server

```sh
npm run dev
```

The application should now be running locally. Open your browser and go to [http://localhost:5173](http://localhost:5173) to see it in action.
