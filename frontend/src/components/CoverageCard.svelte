<script lang="ts">
  interface Props {
    coverage: number;
  }

  let { coverage }: Props = $props();

  let percent = $derived(coverage * 100);
  let isGood = $derived(percent > 70);
</script>

<div class="info-card card card-coverage card-centered" class:aligned={isGood}>
  <h2>Surgical Target Coverage</h2>
  <h1>
    <span class="card-body" style:color={isGood ? "#4ade80" : undefined}>
      Coverage: {percent.toFixed(1)}%
    </span>
  </h1>
</div>

<style>
  .aligned {
    border-color: rgba(76, 222, 128, 0.6);
  }
</style>
