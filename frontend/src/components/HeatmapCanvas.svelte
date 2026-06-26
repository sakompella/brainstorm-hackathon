<script lang="ts">
  import { onMount } from "svelte";
  import type { FeaturesMessage, FrameAnalysis, ValueRange } from "../lib/types";
  import { drawHeatmap } from "../lib/renderers/heatmap";
  import { analyzeFrame, updateValueRange } from "../lib/analysis";

  interface Props {
    features: FeaturesMessage | null;
    onAnalysis?: (analysis: FrameAnalysis) => void;
    onFps?: (fps: number) => void;
  }

  let { features, onAnalysis, onFps }: Props = $props();

  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  let cssWidth = 0;
  let cssHeight = 0;
  let valueRange: ValueRange = { vMin: 0, vMax: 0.01 };

  // FPS tracking
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let fps = $state(0);

  onMount(() => {
    ctx = canvasEl.getContext("2d");

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      const rect = entry.contentRect;
      cssWidth = Math.floor(rect.width);
      cssHeight = Math.floor(rect.height);
      const dpr = window.devicePixelRatio || 1;

      canvasEl.width = Math.floor(cssWidth * dpr);
      canvasEl.height = Math.floor(cssHeight * dpr);
      canvasEl.style.width = `${cssWidth}px`;
      canvasEl.style.height = `${cssHeight}px`;

      if (ctx) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = "#0a0a0f";
        ctx.fillRect(0, 0, cssWidth, cssHeight);
      }
    });

    observer.observe(canvasEl.parentElement!);
    return () => observer.disconnect();
  });

  // Draw on each new features frame
  $effect(() => {
    if (!features || !ctx || cssWidth === 0) return;

    const { heatmap, centroid, center_distance: coverage } = features;
    const rows = heatmap.length;
    const cols = heatmap[0]?.length ?? 0;
    if (rows === 0 || cols === 0) return;

    // Update EMA range
    valueRange = updateValueRange(valueRange, heatmap);

    // Analyze frame
    const analysis = analyzeFrame(centroid, rows, cols, coverage);
    onAnalysis?.(analysis);

    // Draw
    drawHeatmap({
      ctx,
      width: cssWidth,
      height: cssHeight,
      heatmap,
      centroid,
      range: valueRange,
      analysis,
      coverage,
    });

    // FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastFpsTime = now;
      onFps?.(fps);
    }
  });
</script>

<canvas bind:this={canvasEl}></canvas>

<style>
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    border-radius: 8px;
  }
</style>
