<template>
  <div id="formula">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.css" integrity="sha384-9eLZqc9ds8eNjO3TmqPeYcDj8n+Qfa4nuSiGYa6DjLNcv9BtN69ZIulL9+8CqC9Y" crossorigin="anonymous">
    <div id="katex-formula" ref="formula"></div><div id="formula-number"><div>({{num}})</div></div>
  </div>
</template>

<script>
import katex from 'katex'
export default {
  name: 'Formula',
  props:['content','num'],
  mounted: function(){
    if(this.content){
      katex.render(this.content, this.$refs['formula'], {
        displayMode: true
      });
    }
  }
}
</script>
<style scoped>
#formula{
  display: flex;
  overflow-x: auto;
  overflow-y:hidden;
  padding: 5px 0px;
}
#formula-number{
  flex: 1;
  text-align: right;
}
#formula-number > div{
  height: 100%;
  transform: translateY(calc(50% - 11px));
  line-height: 20px;
  font-size: 15px;
}
#katex-formula{
  flex:20;
}
</style>
