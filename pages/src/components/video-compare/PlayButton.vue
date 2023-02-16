<template>
  <div id="play-button">
    <div class="icon" v-if="value" @click="emit(false)"><Icon name="pause"></Icon></div>
    <div class="icon" v-else @click="emit(true)"><Icon name="play"></Icon></div>
  </div>
</template>

<script>
import Icon from "vue-awesome/components/Icon";
import "vue-awesome/icons/";
export default {
  components: {
    Icon: Icon,
  },
  data: function () {
    return {
      
    };
  },
  props: ['value'],
  methods:{
    emit: function(val){
        this.$emit("input", val)
        if(val){
            this.$emit("play")
        } else{
            this.$emit("pause")
        }
        
    }
  }
  
};
</script>

<style scoped>
.icon{
    line-height: 0px;
    cursor: pointer;
}
</style>