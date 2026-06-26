<script lang="ts">
  import { onMount } from "svelte";
  import type { FeaturesMessage } from "../lib/types";
  import { meanHeatmapPower, toDb } from "../lib/analysis";
  import { TimeSeriesBuffer } from "../lib/timeseries-buffer";
  import { drawTimeSeries } from "../lib/renderers/timeseries";

  interface Props {
    features: FeaturesMessage | null;
  }

  let { features }: Props = $props();

  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  let cssWidth = 0;
  let cssHeight = 0;

  const buffer = new TimeSeriesBuffer(500);
  let baselinePower: number | null = null;

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
      }
    });

    observer.observe(canvasEl);
    return () => observer.disconnect();
  });

  $effect(() => {
    if (!features || !ctx || cssWidth === 0) return;

    const meanPower = meanHeatmapPower(features.heatmap);

    // Latch baseline on first frame
    if (baselinePower === null) {
      baselinePower = Math.max(meanPower, 1e-12);
    }

    const dbPower = toDb(meanPower, baselinePower);
    buffer.push({ t: features.t, value: dbPower });

    drawTimeSeries({
      ctx,
      width: cssWidth,
      height: cssHeight,
      points: buffer.points,
    });
  });
</script>

<canvas bind:this={canvasEl}></canvas>

<style>
  canvas {
    flex: 0 0 auto;
    height: clamp(180px, 24vh, 260px);
    min-height: 180px;
    width: 100%;
    display: block;
  }
</style>
