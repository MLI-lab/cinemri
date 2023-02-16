<template>
  <div id="table">
    <v-table
      :data="listed_videos"
      :currentPage.sync="currentPage"
      :pageSize="10"
      @totalPagesChanged="totalPages = $event"
    >
      <thead slot="head">
        <th></th>
        <v-th v-for="(c, i) in columns" :key="i" :sortKey="c.key">{{
          c.name
        }}</v-th>
      </thead>
      <tbody slot="body" slot-scope="{ displayData }">
        <tr v-for="row in displayData" :key="row.id">
          <td style="width: 45px">
            <input
              type="radio"
              @click="solo($event, row)"
              :checked="row.solo"
            />
            <input type="checkbox" v-model="row.selected" />
          </td>
          <td v-for="(attr, j) in columns" :key="j">
            {{ accesByKeyPath(row, attr.key) }}
          </td>
        </tr>
      </tbody>
    </v-table>
    <smart-pagination
      :currentPage.sync="currentPage"
      :totalPages="totalPages"
    />
  </div>
</template>

<script>
export default {
  data: function () {
    return {
      currentPage: 1,
      totalPages: 0
    }
  },
  props: ["videos", "columns"],
  created: function () {},
  computed: {
    listed_videos() {
      var listed_videos = this.videos
        .map((elem, index) => {
          elem.id = index;
          return elem;
        })
        .filter((item) => {
          return !item.hasOwnProperty("listed") || item.listed;
        });
      return listed_videos;
    },
  },
  methods: {
    solo(event, video) {
      for (var i = 0; i < this.listed_videos.length; i++) {
        this.listed_videos[i].selected = false;
        this.listed_videos[i].solo = false;
      }
      video.selected = true;
      video.solo = true;
    },
    accesByKeyPath(object, key_path) {
      var path = key_path.split(".");
      for (var i = 0; i < path.length; i++) {
        object = object[path[i]];
      }
      return object;
    },
  },
};
</script>

<style scoped>
td,
th {
  border: 1px solid #dfe2e5;
  padding: 4px 8px;
}

tr {
  border-top: 1px solid #dfe2e5;
}

table {
  border-collapse: collapse;
  text-align: right;
  width: 100%;
}

tr:hover {
  background-color: rgba(180, 180, 180, 0.1);
}

#table {
  margin-bottom: 5px;
  overflow: scroll;
}
</style>

<style>
.page-item{
}
.pagination{
    list-style-type: none;
    display: grid;
    grid-auto-flow: column;
    width: fit-content;
    grid-gap: -1px;
    padding-inline-start: 0px;
}

.page-item.active .page-link {
    z-index: 1;
    color: #fff;
    background-color: #0175FF;
    border-color: #0175FF;
}

.page-link {
    position: relative;
    display: block;
    padding: 0.5rem 0.75rem;
    margin-left: 0px;
    line-height: 21px;
    color: #0175FF;
    background-color: #fff;
    border: 1px solid #dee2e6;
}

.page-link:not(:disabled):not(.disabled) {
    cursor: pointer;
}

</style>