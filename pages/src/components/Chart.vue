<template>
    <div id="chart">
        <canvas ref="chart"></canvas>
    </div>
</template>

<script>

import Chart from 'chart.js/auto';

export default {
  name: 'chart',
  components: {
  },
  data() {
    return {
      
    }
  },
  props:["data"],
  mounted(){
    new Chart(this.$refs.chart, this.data);
  }
}
</script>