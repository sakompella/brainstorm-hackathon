<script lang="ts">
  import type { Direction } from "../lib/types";

  interface Props {
    direction: Direction;
  }

  let { direction }: Props = $props();
</script>

<div class="info-card card card-move card-centered">
  <h2>Move Instruction</h2>
  <h1><span class="card-body" id="direction-display">{direction}</span></h1>
</div>
