<template>
  <div id="page-not-found">
    <NavBack></NavBack>
    <Frame>
      <Title :color="color">
        404 - Page not Found
      </Title>
  </Frame>
  </div>
</template>

<script>
import Title from '../components/Title.vue'
import Frame from '../components/Frame.vue'
import Paragraph from '../components/Paragraph.vue'
import SubTitle from '../components/SubTitle.vue'
import NavBack from '../components/NavBack.vue'

export default {
  name: 'PageNotFound',
  data: function(){
    return {
      color: "#944"
    }
  },
  components: {
    'Frame': Frame,
    'Title': Title,
    'SubTitle': SubTitle,
    'Paragraph': Paragraph,
    'NavBack': NavBack,
  }
}
</script>
<style scoped>
  #page-not-found{
    min-height: 100vh;
  }
</style>
