<template>
  <div id="subtitle" :style="{'border-color': colorVal}" :class="{'no-print':noprint}">
    <span id="subtitle-index">{{" " + index}}</span><span ref="subtitle"><slot></slot></span>
  </div>
</template>

<script>
import bus from './bus.js'
export default {
  name: 'SubTitle',
  props: ['color','hirarchy','nolist','noprint'],
  data: function(){
    return {
      index: ""
    }
  },
  computed: {
    colorVal: function(){
      return null;
    },
    colorVal: function(){
      if(this.color){
        return this.color;
      }
      return this.$store.state.color;
    }
  },
  mounted: function(){
    var hirarchy = 2;
    if(this.hirarchy){
      hirarchy = parseInt(this.hirarchy);
    }
    this.$refs['subtitle'].elem = this;
    if(!this.nolist){
      bus.$emit("header",{element: this.$refs['subtitle'], hirarchy:hirarchy});
    }
  }
}
</script>
<style scoped>
  #subtitle{
    margin-bottom: 20px;
    margin-top: 60px;
    margin-left: 15px;
    margin-right: 15px;
    font-size: 20px;
    padding: 5px;
    border-bottom: 4px solid black;
    display: table;
  }
  #subtitle-index{
    display:none;
  }
  @media print{
    #subtitle{
      page-break-after: avoid;
      margin-top: 30px;
      margin-bottom: 5px;
      border: none;
      padding: 0;
    }
    #subtitle-index{
      display:unset;
    }
    .no-print{
      display:none !important;
    }
  }
</style>
