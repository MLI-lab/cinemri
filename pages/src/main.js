import Vue from 'vue'
import App from './App.vue'
import router from './router'
import Vuex from 'vuex'
import SmartTable from 'vuejs-smart-table'

Vue.use(Vuex)
Vue.config.productionTip = false

Vue.use(SmartTable)

const store = new Vuex.Store({
  state: {
    color: "",
    tilesData: []
  },
  mutations: {
  	setColor: function(c){this.color = c;}
  }
})

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')
