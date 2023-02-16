<template>
  <div id="subsubtitle" :style="{'border-color': colorVal}">
    <span id="subsubtitle-index">{{" " + index}}</span><span ref="subsubtitle"><slot></slot></span>
  </div>
</template>

<script>
import bus from './bus.js'
export default {
  name: 'SubSubTitle',
  props: ['color','hirarchy','list'],
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
    var hirarchy = 3;
    if(this.hirarchy){
      hirarchy = parseInt(this.hirarchy);
    }
    this.$refs['subsubtitle'].elem = this;
    if(this.list){
      bus.$emit("header",{element: this.$refs['subsubtitle'], hirarchy:hirarchy});
    }
  }
}
</script>
<style scoped>
  #subsubtitle-index{
    display:none;
  }
  #subsubtitle{
    margin-bottom: 15px;
    margin-right: 15px;
    font-size: 15px;
    padding: 0px;
    font-size: 17px;
    font-weight: bold;
    display: table;
  }
  @media print {
    #subsubtitle{
      page-break-after: avoid;
    }
    #subsubtitle-index{
      display:unset;
    }
  }
</style>
