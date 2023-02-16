<template>
  <div id="table-of-contents-item">
    <div class="toc-inner">
      <span class="table-of-contents-item" @click="item.element.scrollIntoView({block: 'center',behavior:'smooth'});"><span>{{item.prefix}}</span><span>{{' ' + item.label.replace(/\r?\n|\r/g, '')}}</span></span>
      <div>
        <TableOfContentsItem v-for="(sub, i) in item.subs" :key="i" :item="sub"></TableOfContentsItem>
      </div>
    </div>
  </div>
</template>

<script>
import TableOfContentsItem from './TableOfContentsItem.vue'
export default {
  name: 'TableOfContentsItem',
  components: {
    'TableOfContentsItem':TableOfContentsItem
  },
  props: ['item'],
  computed: {
  },
  mounted: function(){

  }
}
</script>
<style scoped>
.table-of-contents-item{
  cursor: pointer;
}
.table-of-contents-item:hover{
  color: #55c;
}
.toc-inner{
  padding-left: 20px;
}
</style>
