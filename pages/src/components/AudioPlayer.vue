<template>
  <div id="audioplayer">
  <audio controls>
    <source :src="audioSrc" type="audio/ogg">
      Your browser does not support the audio element.
  </audio>
  </div>
</template>

<script>
export default {
  name: 'AudioPlayer',
  props:['src'],
  computed:{
    audioSrc: function(){
      return require('../assets/'+this.src);
    }
  }
}
</script>
<style scoped>

audio{
width: 100%;
}
</style>
