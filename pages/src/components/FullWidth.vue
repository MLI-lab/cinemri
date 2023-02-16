<template>
  <div id="full-width">
    <slot></slot>
  </div>
</template>

<script>
import bus from './bus.js'
export default {
  name: 'FullWidth'
}
</script>
<style scoped>
  #full-width{
    width: 100vw;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
  }
</style>
