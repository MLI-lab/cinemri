<template>
  <div id="paragraph" :style="{'border-color': colorVal}">
    <slot></slot>
  </div>
</template>

<script>

export default {
  name: 'Paragraph',
  props: ['color'],
  computed:{
    colorVal: function(){
      if(this.color){
        return this.color;
      }
      return this.$store.state.color;
    }
  }
}
</script>
<style scoped>
  @font-face {
    font-family: ubuntu;
    src: url('../assets/Ubuntu-Regular.ttf');
  }
  #paragraph{
    border-left: 15px solid black;
    padding: 15px;
    text-align: justify;
    margin: 5px 0px;
    line-height: 30px;
    font-size: 15px;
    font-family: ubuntu;
    background-color: #fff;

  }
  #paragraph{
    word-break: normal;

  }
  ol, ul{
    padding-left: 15px;
  }
  @media (max-width: 500px){
    ol, ul{
      padding-left: 10px;
    }

    #paragraph{
      border-left-width: 5px;
    }
  }
  @media print{
    #paragraph{
      border: none;
      padding-top: 0;
      padding-bottom: 0;
      color: #000;
      font-size: 15px;
      line-height: 23px;
    }
  }
</style>
