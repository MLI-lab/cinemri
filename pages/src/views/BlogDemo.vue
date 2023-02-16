<template>
  <div id="blog-fmlp">
    <Header
      title="FMLP for Cardiac Cine MRI (Demo)"
      backgroundUrl="cinemri.png"
      textColor="white"
      :color="color"
      :hasNavBack="1"
    ></Header>
    <Frame width="narrow">
      <Title nolist="true">Table of Contents </Title>
      <Paragraph><TableOfContents></TableOfContents></Paragraph>
      <Title>FMLP vs. t-DIP</Title>
      <SubTitle>Low-resolution, high SNR dataset</SubTitle>
      <Paragraph>
        <SubSubTitle>Short acquisition time (K=225)</SubSubTitle>
        <VideoCompare :data="videodata_fmlp_tdip_10"></VideoCompare>
        <Register>
          <template v-slot:header>Performance Metrics</template>
          <template v-slot:body>
            <Chart
              :data="chartdata_sxy"
              style="max-width: 600px; margin: 0 auto"
            ></Chart>
            <Chart
              :data="chartdata_st"
              style="max-width: 600px; margin: 0 auto"
            ></Chart>
          </template>
        </Register>
      </Paragraph>
      <Paragraph>
        <SubSubTitle>Changing the acquisition time</SubSubTitle>
        <VideoCompare
          :data="videodata_fmlp_tdip_10_450_900"
        ></VideoCompare>
        <Register>
          <template v-slot:header>Performance Metrics</template>
          <template v-slot:body>
            <Chart
              :data="chartdata_acquisition_length"
              style="max-width: 600px; margin: 0 auto"
            ></Chart>
          </template>
        </Register>
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


var videodata_fmlp_tdip_10 = require("./cinemri/videodata_fmlp_tdip_10");
var chartdata_sxy = require("./cinemri/chartdata_sxy");
var chartdata_st = require("./cinemri/chartdata_st");


export default {
  name: "BlogCineMRI",
  data: function () {
    return {
      chartdata_sxy: chartdata_sxy,
      chartdata_st: chartdata_st,
      videodata_fmlp_tdip_10: videodata_fmlp_tdip_10,
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
#blog-fmlp {
  min-height: 100vh;
}
</style>
