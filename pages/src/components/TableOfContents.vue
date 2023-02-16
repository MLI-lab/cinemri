<template>
  <div id="table-of-contents">
    <TableOfContentsItem v-for="(item, i) in table" :key="i" class="item" :item="item"></TableOfContentsItem>
  </div>
</template>

<script>
import bus from './bus.js'
import TableOfContentsItem from './TableOfContentsItem.vue'
export default {
  name: 'TableOfContents',
  components: {
    'TableOfContentsItem':TableOfContentsItem
  },
  data: function(){
    return {
      plainTable: [],
      table: []
    }
  },
  computed: {
  },
  created: function(){
    bus.$on("header",(data)=>{
      if(data.element.innerHTML){
        this.plainTable.push({hirarchy: data.hirarchy,
          yPosition: data.element.offsetTop,
          label: data.element.innerHTML,
          element: data.element
        });
        this.buildTable();
      }
    });
  },
  methods: {
    buildTable: function(){
      this.plainTable.sort(function(a,b){
        return a.yPosition > b.yPosition;
      });
      this.table = this.buildTableRecursive(0,1,'').table;
    },
    buildTableRecursive: function(startIndex, hirarchy, prefix){
      var table = [];
      var i = startIndex;
      var index = 0;
      while(i < this.plainTable.length){
        if(this.plainTable[i].hirarchy < hirarchy){
          return {table: table, index: i-1};
        } else if(this.plainTable[i].hirarchy == hirarchy){
          index++;
          this.plainTable[i].prefix = prefix + index + ".";
          table.push({label:this.plainTable[i].label, prefix: prefix + index + ".", element:this.plainTable[i].element, subs: []});
          this.plainTable[i].element.elem.index = prefix + index + ".";
        } else{
          var res = this.buildTableRecursive(i,this.plainTable[i].hirarchy, prefix + index + ".");
          i = res.index;
          table[table.length-1].subs = res.table;
        }
        i++;

      }
      return {table: table, index: i};
    }
  }
}
</script>
<style scoped>
#table-of-contents{
  font-size: 16px;
  line-height: 24px;
  margin-left:-20px;
  
}

.item{
  padding-bottom:5px;
}

ul{
  list-style-type:none;
}
</style>
