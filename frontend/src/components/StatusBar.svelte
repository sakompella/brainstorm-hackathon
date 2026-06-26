<script lang="ts">
  import type { ConnectionStatus } from "../lib/types";

  interface Props {
    status: ConnectionStatus;
    statusText: string;
    time: number;
    fps: number;
    channelCount: number;
    gridSize: number;
  }

  let { status, statusText, time, fps, channelCount, gridSize }: Props =
    $props();
</script>

<div class="status-bar">
  <span class="status-indicator {status}"></span>
  <span>{statusText}</span>
  <span class="separator">|</span>
  <span class="accent">t = {time.toFixed(2)}s</span>
  <span class="separator">|</span>
  <span class="accent">{fps} FPS</span>
  <span class="separator">|</span>
  <span class="accent"
    >{channelCount} channels ({gridSize}x{gridSize} data)</span
  >
</div>
