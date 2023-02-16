<template>
  <div>
    <div id="nav-back">
      <router-link to="/"  :style="{color:colorVal}" id="link"><Icon name="arrow-left"></Icon></router-link>
      <div id="title">{{title}}</div>
    </div>
    <div id="spacer"></div>
  </div>
</template>

<script>
import Icon from 'vue-awesome/components/Icon'
import 'vue-awesome/icons/'
export default {
  name: 'NavBack',
  components: {
    'Icon': Icon
  },
  props: ['color', 'title'],
  computed: {
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
  #nav-back{
    position: absolute;
    top: 0;
    left: 0;
    /* padding: 15px; */
    font-size: 30px;
    cursor: pointer;
    width: 100%;

  }
  .fa-icon {
    width: auto;
    height: 1em;
    max-width: 100%;
    max-height: 100%;
  }

  #link{
    padding: 15px;
    position: absolute;
  }
  #title{
    display: inline-block;
    top: 10px;
    position: absolute;
    left: 0;
    margin-left: 50%;
    transform: translateX(-50%);
  }
  #spacer{
    height: 60px;
  }

</style>
