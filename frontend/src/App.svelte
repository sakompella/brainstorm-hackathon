<script lang="ts">
  import { onMount } from "svelte";
  import { createWsStore } from "./lib/ws.svelte";
  import type { FrameAnalysis } from "./lib/types";
  import StatusBar from "./components/StatusBar.svelte";
  import HeatmapCanvas from "./components/HeatmapCanvas.svelte";
  import CenteringCard from "./components/CenteringCard.svelte";
  import MoveCard from "./components/MoveCard.svelte";
  import TimeSeriesCanvas from "./components/TimeSeriesCanvas.svelte";

  const store = createWsStore();

  let analysis: FrameAnalysis | null = $state(null);
  let heatmapFps = $state(0);

  // Derived values from latest features
  let time = $derived(store.features?.t ?? 0);
  let channelCount = $derived(store.features?.n_ch ?? 0);
  let gridSize = $derived(store.features?.heatmap?.length ?? 0);
  let centering = $derived(store.features?.center_distance ?? 0);
  let direction = $derived(analysis?.direction ?? "Center");

  onMount(() => {
    store.connect();
    return () => store.destroy();
  });

  function handleAnalysis(a: FrameAnalysis) {
    analysis = a;
  }

  function handleFps(fps: number) {
    heatmapFps = fps;
  }
</script>

<div class="container">
  <header>
    <h1>Neural Activity</h1>
    <StatusBar
      status={store.status}
      statusText={store.statusText}
      {time}
      fps={heatmapFps}
      {channelCount}
      {gridSize}
    />
  </header>

  <main>
    <div class="layout">
      <section class="panel-left panel">
        <HeatmapCanvas
          features={store.features}
          onAnalysis={handleAnalysis}
          onFps={handleFps}
        />
      </section>
      <aside class="sidebar">
        <CenteringCard {centering} />
        <MoveCard {direction} />
        <div class="info-card panel signal-monitor">
          <h2>High Gamma</h2>
          <TimeSeriesCanvas features={store.features} />
        </div>
      </aside>
    </div>
  </main>
</div>
