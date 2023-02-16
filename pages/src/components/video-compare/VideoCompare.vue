<template>
    <div id="video-compare">
        <ViewerSideBySide :videos="selectedVideos" :brightness="brightness" :contrast="contrast" :origsize="origsize"></ViewerSideBySide>

        <Register>
            <template v-slot:header>Configurations</template>
            <template v-slot:body>
                <div v-for="(g, i) in groups" :key="i" >
                    <div v-if="groups.length > 1" class="table-title">{{g.title}}</div>
                    <Table :videos="g.videos" :columns="g.columns"></Table>
                </div>
                
            </template>
        </Register>
        <Register>
            <template v-slot:header>Contrast/Brightness</template>
            <template v-slot:body>
                <div style="display:grid; grid-template-columns: 90px 1fr;">
                    <div style="padding-right:10px">Brightness:</div>
                    <div style="display:flex"><input type="range" min="0" max="200" value="100" v-model="brightness"><div><input type="number" min="0" max="200" v-model="brightness">%</div></div>
                    <div style="padding-right:10px">Contrast:</div>
                    <div style="display:flex"><input type="range" min="0" max="200" value="100" v-model="contrast"><div><input type="number" min="0" max="200" v-model="contrast">%</div></div>
                </div>
                <div style="display:flex"><input type="checkbox" selected v-model="origsize"> show original size</div>
            </template>
        </Register>
    </div>
</template>

<script>
import ViewerSideBySide from './ViewerSideBySide.vue'
import Table from './Table.vue'

import Register from '../Register.vue'
import Icon from "vue-awesome/components/Icon";
import "vue-awesome/icons/";

export default {
  name: 'VideoCompare',
  components: {
    'ViewerSideBySide': ViewerSideBySide,
    'Register': Register,
    'Icon': Icon,
    'Table': Table
  },
  props:["data"],
  data: function(){
    return {
        brightness: 100,
        contrast: 100,
        origsize: false
    }
  },
  computed:{
    groups(){
        if(this.data.hasOwnProperty("groups")){
            return this.data.groups
        }
        else {
            return [
                this.data 
            ]
        }
    },
    selectedVideos(){
        var selected = []
        for(const g of this.groups){
            for(const v of g.videos){
                if(v.selected){
                    selected.push(v)
                }
            }
        }
        return selected
    }
  },
  methods:{
    
  }
}
</script>
<style scoped>
.table-title{
    font-weight: bold;
    padding: 5px;
}

</style>