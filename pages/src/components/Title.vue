<template>
  <div id="title" :style="{'border-color': colorVal}">
    <span id="title-index">{{" " + index}}</span><span ref="title"><slot></slot></span>
  </div>
</template>

<script>
import bus from './bus.js'
export default {
  name: 'Title',
  props: ['color','hirarchy','nolist'],
  data: function(){
    return {
      index: ""
    }
  },
  computed: {
    colorVal: function(){
      if(this.color){
        return this.color;
      }
      return this.$store.state.color;
    }
  },
  mounted: function(){
    var hirarchy = 1;
    if(this.hirarchy){
      hirarchy = this.hirarchy;
    }
    this.$refs['title'].elem = this;
    if(!this.nolist){
      bus.$emit("header",{element: this.$refs['title'], hirarchy:hirarchy});
    }

  }
}
</script>
<style scoped>
  #title{
    margin-bottom: 40px;
    margin-top: 50px;
    margin-left: 15px;
    margin-right: 15px;
    font-size: 40px;
    padding: 5px;
    border-bottom: 4px solid black;

  }
  @media (max-width: 600px){
    #title{
      font-size: 30px;
    }
  }
  #title-index{
    display:none;
  }
  @media print{
    #title{
      page-break-after: avoid;
      margin-bottom: 20px;
      border: none;
      padding: 0;
    }
    #title-index{
      display:unset !important;
    }
  }
</style>
