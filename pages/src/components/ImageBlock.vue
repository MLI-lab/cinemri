<template>
  <div id="image-block" :class="{'full-width': size == 'fullWidth', 'image-center':align == 'center'}">
    <img :src="imgUrl" :class="{'full-width': size == 'fullWidth'}">
    <div id="image-description">{{description}}</div>
  </div>
</template>

<script>
export default {
  name: 'ImageBlock',
  props: ['src','description','size','align'],
  computed: {
    imgUrl: function(){
      return require('./../assets/'+this.src)
    }
  }
}
</script>
<style scoped>
  #image-block img{
    width: 100%;
    max-width: 600px;
    box-shadow: 0px 0px 0px 1px rgba(0,0,0,0.1);
  }
  #image-block > div{
    width: 100%;
  }
  #image-block {
    padding: 15px 0px;
    width: fit-content;
    /* margin: 0 auto; */
  }
  #image-description{
    font-style: italic;
    font-size: 13px;
    line-height: 16px;
  }
  .full-width{
    display: block !important;
    max-width: unset !important;
    width:100% !important;
  }
  .image-center{
    margin: 0 auto;
  }
</style>
