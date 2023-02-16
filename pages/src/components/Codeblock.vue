<template>
  <div id="codeblock">
    <pre v-highlightjs><code :class="type">
      <slot></slot>
    </code></pre>
  </div>
</template>

<script>
import Vue from 'vue'
import VueHighlightJS from 'vue-highlightjs'
Vue.use(VueHighlightJS)
export default {
  name: 'Codeblock',
  props:['type']
}
</script>
<style scoped>
  pre{
    margin: 0 !important;
  }
  .hljs{
    display:block;
    padding:0px 15px;
    background:#ddd;
    text-align: left;
    overflow-x: auto;
    margin: 15px 0px;
  }
  .hljs,.hljs-subst{color:#444}
  .hljs-comment{color:#888888}
  .hljs-keyword,.hljs-attribute,.hljs-selector-tag,.hljs-meta-keyword,.hljs-doctag,.hljs-name{font-weight:bold}
  .hljs-type,.hljs-string,.hljs-number,.hljs-selector-id,.hljs-selector-class,.hljs-quote,.hljs-template-tag,.hljs-deletion{color:#880000}
  .hljs-title,.hljs-section{color:#880000;font-weight:bold}
  .hljs-regexp,.hljs-symbol,.hljs-variable,.hljs-template-variable,.hljs-link,.hljs-selector-attr,.hljs-selector-pseudo{color:#BC6060}
  .hljs-literal{color:#78A960}
  .hljs-built_in,.hljs-bullet,.hljs-code,.hljs-addition{color:#397300}
  .hljs-meta{color:#1f7199}.hljs-meta-string{color:#4d99bf}
  .hljs-emphasis{font-style:italic}
  .hljs-strong{font-weight:bold}
</style>
