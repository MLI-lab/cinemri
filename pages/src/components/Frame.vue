<template>
  <div class="frame" :class="{'narrow': width == 'narrow'}">
    <slot></slot>
  </div>
</template>

<script>

export default {
  name: 'Frame',
  props: ['width']
}
</script>
<style scoped>

  .frame{
    max-width: 80vw;
    margin: 0 auto;
  }
  @media (max-width: 500px){
    .frame{
      max-width: 100vw !important;
    }
  }
  @media (max-width: 770px){
    .narrow{
      max-width: 100vw !important;
    }
  }
  .narrow{
    max-width: 70vw;
  }
  @media print{
    .frame{
      max-width: 100% !important;
    }
  }
</style>
