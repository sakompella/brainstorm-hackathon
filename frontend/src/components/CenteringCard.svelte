<script lang="ts">
  interface Props {
    /** Centering score from backend (center_distance): 1.0 = hotspot at array center, 0.0 = corner. */
    centering: number;
  }

  let { centering }: Props = $props();

  let percent = $derived(centering * 100);
  let isGood = $derived(percent > 70);
</script>

<div class="info-card card card-coverage card-centered" class:aligned={isGood}>
  <h2>Array Centering</h2>
  <h1>
    <span class="card-body" style:color={isGood ? "#4ade80" : undefined}>
      {percent.toFixed(1)}% centered
    </span>
  </h1>
</div>

<style>
  .aligned {
    border-color: rgba(76, 222, 128, 0.6);
  }
</style>
