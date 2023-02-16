<template>
  <div id="header">
     <div id="header-box" :class="{'header-extended':headerExtended}" :style="{'border-color': colorVal}">
      <div :style="{height:'50vh'}"></div>
      <div id="header-back" :style="{'color': textColorVal}" v-if="hasNavBack"><router-link to="/" :style="{color:'inherit'}"><Icon name="arrow-left"></Icon></router-link></div>
      <div id="header-wrapper" :style="{'background-color': colorVal}">
        <div id="header">
          <div id="header-background" :style="{'background-image': 'url('+backgrUrl+')'}"></div>
          <div class="header-content">
            <div>
              <div id="header-title" @click="scrollTop()" :style="{'color': textColorVal}">{{title}}</div>
              <div id="header-current-chapter" @click="tocActive = !tocActive" :style="{'color': textColorVal}">{{currentChapter.prefix}} {{currentChapter.label}}</div>
            </div>
          </div>
        </div>
        <div id="header-toc" :class="{'header-toc-active':tocActive && !headerExtended}">
          <TableOfContents></TableOfContents>
          <div id="header-close-toc" @click="tocActive = false"><Icon name="times"></Icon></div>
        </div>
        <div class="scrollbar">
          <div :style="{'transform':'translateX('+(animationState-100)+'%)','background-color':colorVal}" ></div>
        </div>
      </div>
    </div>
    <div id="header-print">{{title}}</div>
  </div>
</template>

<script>
import Icon from 'vue-awesome/components/Icon'
import 'vue-awesome/icons/'
import TableOfContents from './TableOfContents'
import bus from './bus.js'

export default {
  name: 'Header',
  props: ['title','backgroundUrl','color', 'textColor', 'hasNavBack'],
  data: function(){
    return {
      plainTable: [],
      animationState: undefined,
      currentChapter: {},
      headerExtended: true,
      headerHeight: 60,
      tocActive: false
    }
  },
  components: {
    'Icon': Icon,
    'TableOfContents': TableOfContents
  },
  created: function(){
    bus.$on("header",(data)=>{
      if(data.element.innerHTML){
        this.plainTable.push({
          hirarchy: data.hirarchy,
          yPosition: data.element.offsetTop,
          label: data.element.innerHTML,
          element: data.element,
          prefix: ""
        });
        this.plainTable.sort(function(a,b){
          return a.yPosition > b.yPosition;
        });
        this.buildTable();
      }
      this.onscroll();
    });
  },
  mounted: function(){
    bus.$on(this.animationStateId,(animationState)=>{
      this.animationState = animationState;
    });
    window.addEventListener("scroll",this.onscroll);

  },
  destroyed () {
    window.removeEventListener('scroll',this.onscroll);
  },
  computed: {
    backgrUrl: function(){
      if(this.backgroundUrl != undefined){
        return require('./../assets/'+this.backgroundUrl);
      } else {
        return "";
      }
    },
    colorVal: function(){
      if(this.color){
        return this.color;
      }
      return "green";
    },
    textColorVal: function(){
      if(this.textColor){
        return this.textColor;
      }
      return "green";
    }
  },
  methods: {
    scrollTop: function(){
      window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
    },
    onscroll: function(){
      this.animationState = window.pageYOffset/(document.body.clientHeight-window.innerHeight) * 100;
      this.headerExtended = window.pageYOffset < 60;
      this.tocActive &= !this.headerExtended;
      if(window.pageYOffset < window.innerHeight/2-60){
        this.headerHeight = window.innerHeight/2 - window.pageYOffset;
      } else{
        this.headerHeight = 60;
      }
      for(var i=this.plainTable.length-1;i>=0;i--){
        if(this.plainTable[i].element.getBoundingClientRect().top - this.plainTable[i].element.clientHeight < window.innerHeight/2){
          this.currentChapter = {label: this.plainTable[i].label, element: this.plainTable[i].element, prefix: this.plainTable[i].prefix};
          break;
        }
      }
    },
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
  #header-wrapper{
    width: 100vw;
    position: fixed;
    top: 0;
    left: 0;
    
    z-index: 100;

    height: 50vh;
    transition: transform 1s;
    transform: translateY(calc(-50vh + 60px));


  }
  #header{
    height: 100%;
    position: relative;
    height: 100%;

  }
  #header-back{
    font-size: 30px;
    padding: 15px;
    position: fixed;
    top: 0;
    z-index: 200;
  }
  .fa-icon {
    width: auto;
    height: 1em;
    max-width: 100%;
    max-height: 100%;
  }
  .scrollbar{
    width: 100%;
    height: 4px;
    position: absolute;
    bottom: -4px;

  }
  .scrollbar > div{
    width: 100%;
    height: 100%;
  }
  .header-content{
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0px;
    transition: transform 1s;
    transform: translateY(calc(100% - 30px));
    text-align: center;
    transition: transform 1s;
    transform-origin: top;
    pointer-events: none;
  }
  #header-back{
    color: inherit;
  }
  #header-title{
    font-size: 22px;
    margin-top: -22px;
    transform: translateY(0px);
    cursor: pointer;
    transition: transform 1s;
  }

  #header-current-chapter{
    transition: opacity 0.5s;
    cursor: pointer;
    line-height: 21px;
  }
  #header-current-chapter:hover{
    color: #33a;
  }
  #header-background{
    position: absolute;
    top: 0;
    width: 100%;
    height: 100%;
    background-size: cover;
    opacity: 0;
    background-position: center;
    transition: opacity 1s;
  }
  .header-extended #header-background{
    opacity: 0.7;
  }
  .header-extended #header-title{
    transform: translateY(11px);

    text-shadow: 0px 0px 20px white;
  }
  .header-extended #header-current-chapter{
    opacity: 0;
  }
  .header-extended #header-wrapper{
    transform: translateY(0);
  }

  .header-extended .header-content{
    transform: translateY(50%);
  }
  .header-content > div{
    transition: transform 1s;
    pointer-events: all;
  }
  .header-extended .header-content > div{
    transform: scale(1.6);
  }
  .header-toc-active{
    border-color: #fff !important;
  }
  #header-toc{
    overflow: hidden;
    max-height: 0;
    border-bottom: transparent 4px solid;
    background-color: #eee;
  }
  #header-toc > #table-of-contents{
    padding-top: 30px;
    padding-bottom: 0px;
    width: fit-content;
    margin: 0 auto;
  }
  .header-toc-active{
    max-height: 100vh !important;
  }
  #header-close-toc{
    font-size: 30px;
    padding: 15px 30px;
    cursor: pointer;
    text-align: right;
  }
  #header-print{
    display: none;
    font-size: 45px;
    text-align: center;
    margin-top: 40px;
    margin-bottom: 40px;
  }
  @media(max-width:600px){
    #header-title{
      font-size: 16px;
      line-height: 24px;
    }
    .header-extended .header-content > div{
      transform: scale(1.4);
    }
  }
  @media(max-width:370px){
    #header-title{
      font-size: 14px;
      line-height: 24px;
    }
    .header-extended .header-content > div{
      transform: scale(1.3);
    }
  }
  @media print{
    #header-box{
      display: none;
    }
    #header-print{
      display: block;
    }
  }
</style>
