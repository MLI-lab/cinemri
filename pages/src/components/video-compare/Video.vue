<template>
  <div id="video">
    <video
      ref="video"
      @canplaythrough="oncanplaythrough()"
      @ended="onended()"
      @timeupdate="ontimeupdate()"
      :style="`filter: brightness(${brightness}%) contrast(${contrast}%)`"
      :class="{'origsize':origsize}"
    >
      <source :src="src" type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  </div>
</template>

<script>
export default {
  data: function () {
    return {
      canplaythrough: false,
      playing: false,
      time: 0,
      duration: 0,
      ended: false
    };
  },
  props: ["src", "inittime", "brightness", "contrast", "origsize"],
  methods:{
    oncanplaythrough(){
      if(!this.canplaythrough){
        this.canplaythrough = true
        this.duration = this.$refs.video.duration
        this.$emit("canplaythrough")
        this.$refs.video.currentTime = this.inittime
      }
    },
    onended(){
      this.ended = true
      this.$emit("ended")
    },
    ontimeupdate(){
      this.$emit("timeupdate", this.$refs.video.currentTime)
    },
    play(){
      this.$refs.video.play()
      this.ended = false
    },
    pause(){
      this.$refs.video.pause()
    },
    settime(time){
      this.$refs.video.currentTime = time
      this.ended = false
    },
  }
};
</script>

<style scoped>
video{
  width: 100%;
  image-rendering: pixelated;
}
#video{
  text-align: center;
}

.origsize{
  width:unset;
}
</style>