<template>
  <div id="inline-image">
    <img :src="imgUrl">
  </div>
</template>

<script>
export default {
  name: 'InlineImage',
  props: ['src'],
  computed: {
    imgUrl: function(){
      return require('./../assets/'+this.src)
    }
  }
}
</script>
<style scoped>
  #inline-image img{
    width: 100%;
  }
  #inline-image{
    display: inline-block;
    float: right;
    padding: 7px;
    padding-right: 0px;
    width:50%;
  }
</style>
