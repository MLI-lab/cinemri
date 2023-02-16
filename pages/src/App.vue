<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<style>
#app {
  color: #2c3e50;
}
</style>

<script>
    export default {
        name: 'app',
        created(){
           document.title = "Supplementary Material";
           var link = document.createElement('link')
           link.id = 'dynamic-favicon'
           link.rel = 'shortcut icon'
           link.href = "/favicon.ico"
        }
    }
</script>