import Vue from 'vue'
import VueRouter from 'vue-router'
import BlogDemo from '../views/BlogDemo.vue'
import PageNotFound from '../views/PageNotFound.vue'
import Home from '../views/Home.vue'

Vue.use(VueRouter)

const routes = [
  {
    path: '/demo',
    component: BlogDemo
  },
  {
    path: '/',
    component: Home
  },
  {
    path: '*',
    component: PageNotFound
  }
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router
