<template>
  <div id="blog-vae">
    
    <Frame width="narrow">
      <Title>Reconstruction Methods for Real-time Cardiac MRI</Title>
      <SubTitle>Supplementary Material</SubTitle>
      <Paragraph>
        List of all Blogs:
        <ul>
          <li><router-link to="/demo">Demo</router-link></li>
        </ul>
      </Paragraph>
      
    </Frame>
  </div>
</template>

<script>
import Title from "../components/Title.vue";
import Frame from "../components/Frame.vue";
import Paragraph from "../components/Paragraph.vue";
import SubTitle from "../components/SubTitle.vue";
import Header from "../components/Header.vue";
import TableOfContents from "../components/TableOfContents.vue";
import VideoCompare from "../components/video-compare/VideoCompare.vue";
import Chart from "../components/Chart.vue";
import Register from "../components/Register.vue";
import InlineFormula from "../components/InlineFormula.vue";
import SubSubTitle from "../components/SubSubTitle.vue";

var videodata_vae_tdip_11 = require("./cinemri/videodata_vae_tdip_11");


export default {
  name: "BlogCineMRI",
  data: function () {
    return {
      videodata_vae_tdip_11: videodata_vae_tdip_11,
      color: "#2c3e50",
    };
  },
  components: {
    Frame: Frame,
    Title: Title,
    SubTitle: SubTitle,
    Paragraph: Paragraph,
    Header: Header,
    TableOfContents: TableOfContents,
    VideoCompare: VideoCompare,
    Chart: Chart,
    Register: Register,
    InlineFormula: InlineFormula,
    SubSubTitle: SubSubTitle,
  },
  created() {
    this.$store.state.color = this.color;
  },
};
</script>
<style scoped>
#blog-vae {
  min-height: 100vh;
}
</style>
