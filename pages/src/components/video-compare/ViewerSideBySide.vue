<template>
  <div id="viewer-side-by-side">
    <div id="video-grid">
      <div v-for="(video, index) in videos" :key="index" ref="video" style="display:flex">
        <div style="margin:auto; width: 100%">
          <Video
            :src="video.src"
            :inittime="time"
            ref="vid"
            :key="video.src"
            :brightness="brightness"
            :contrast="contrast"
            :origsize="origsize"
            @canplaythrough="oncanplaythrough()"
            @ended="onended()"
            @timeupdate="ontimeupdate($event, index)"
          ></Video>
          <div style="margin-top: -10px" class="title">{{ video.title }}</div>
        </div>
      </div>
    </div>

    <div style="display: flex; margin-bottom: 5px;">
      <PlayButton
        id="playbutton"
        v-model="playing"
        @pause="pause()"
        @play="play()"
      ></PlayButton>
      <input
        type="range"
        ref="timeslider"
        id="timeslider"
        :value="time"
        min="0"
        :max="duration"
        step="any"
        @mousedown="pause()"
        @input="
          (event) => {
            settime(event.target.value);
          }
        "
      />
    </div>
    <!-- <div v-if="!ready">Loading ...</div> -->
    <!-- <video :src="src1" autoplay loop></video>
        <video :src="src2" ></video> -->
    <!-- <div>TIme: {{time}} Playing: {{playing}} Ready:{{ready}}</div> -->
  </div>
</template>
    
<script>
import PlayButton from "./PlayButton.vue";
import Video from "./Video.vue"
export default {
  components: {
    PlayButton: PlayButton,
    Video: Video
  },
  props: ["videos", "brightness", "contrast", "origsize"],
  data: function () {
    return {
      playing: false,
      time: 0,
      ready: true,
      duration: 0,
      max_duration_index: 0
    };
  },
  mounted() {},
  methods: {
    play() {
      if (this.ready) {
        for(var i=0;i<this.videos.length;i++){
          if(this.time <= this.$refs.vid[i].duration){
            this.$refs.vid[i].play()
          }
        }
        this.playing = true;
      }
    },
    pause() {
      for(var i=0;i<this.videos.length;i++){
        this.$refs.vid[i].pause()
      }
      this.playing = false;
    },
    log(txt) {
      console.log(txt);
    },
    settime(time) {
      for(var i=0;i<this.videos.length;i++){
        this.$refs.vid[i].settime(time)
      }
      this.time = time;
    },
    onended() {
      if (this.playing) {
        var ended = true
        for(var i=0;i<this.videos.length;i++){
            if(!this.$refs.vid[i].ended){
              ended = false
              break
            }
        }
        if(ended){
          this.settime(0);
          this.play();
        }
      }
    },
    oncanplaythrough(){
      this.checkready()
    },
    ontimeupdate(time, index){
      if(index == this.max_duration_index && this.playing){
        this.time = time;
      }
    },
    checkready(){
      var ready = true
      for(var i=0;i<this.videos.length;i++){
          if(!this.$refs.vid[i].canplaythrough){
            ready = false
            break
          }
      }
      if(this.$refs.vid.hasOwnProperty(0)){
        var max_duration = 0
        var max_duration_index = 0
        for(var i=0;i<this.videos.length;i++){
          console.log(this.$refs.vid[i].duration)
          if(this.$refs.vid[i].duration > max_duration){
            max_duration = this.$refs.vid[i].duration
            max_duration_index = i
          }
        }
        this.duration = max_duration
        this.max_duration_index = max_duration_index
      }
      
      this.ready = ready
    }
  },
  computed: {
    
  },
  watch: {
    videos: {
      handler(new_videos, old) {
        this.$nextTick(function () {
          this.pause()
          this.settime(this.time)
          this.checkready()
        })
      },
      flush: 'post',
      deep: true,
    }
  },
};
</script>

<style scoped>
#timeslider {
  width: 100%;
}

.video {
  width: 100%;
}
.title {
  text-align: center;
}

#video-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr;
  grid-gap: 5px;
}

@media screen and (max-width: 992px) {
  #video-grid {
    grid-template-columns: 1fr 1fr 1fr;
  }
}

/* On screens that are 600px or less, set the background color to olive */
@media screen and (max-width: 600px) {
  #video-grid {
    grid-template-columns: 1fr 1fr;
  }
}

#playbutton {
  padding: 5px;
}
</style>
