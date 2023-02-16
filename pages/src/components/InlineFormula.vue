<template>
  <div id="inline-formula">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.0/dist/katex.min.css" integrity="sha384-9eLZqc9ds8eNjO3TmqPeYcDj8n+Qfa4nuSiGYa6DjLNcv9BtN69ZIulL9+8CqC9Y" crossorigin="anonymous">
    <div ref="formula">
    </div>
  </div>
</template>

<script>
import katex from 'katex'
export default {
  name: 'InlineFormula',
  props:['content','c'],
  mounted: function(){
    if(this.content){
      katex.render(this.content, this.$refs['formula'], {
        displayMode: false
      });
    }
    else if(this.c){
      katex.render(this.c, this.$refs['formula'], {
        displayMode: false
      });
    }
  }
}
</script>
<style scoped>
#inline-formula{
  display: inline;
}
#inline-formula > div{
  display: inline;
}
</style>
