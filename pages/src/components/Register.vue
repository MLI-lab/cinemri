<template>
    <div id="register">
        <div id="header" @click="open=!open">
            <div id="openclose">
                <div class="icon" v-if="open"><Icon name="caret-up"></Icon></div>
                <div class="icon" v-else><Icon name="caret-down"></Icon></div>
            </div>
            <slot name="header"></slot>
        </div>
        <div id="body" :class="{'closed': !open}">
            <slot name="body"></slot>
        </div>
    </div>
</template>

<script>
import Icon from "vue-awesome/components/Icon";
import "vue-awesome/icons/";
export default {
    name: "register",
      components: {
        Icon: Icon,
    },
    data: function(){
        return {
            open: false
        }
    }
}
</script>

<style scoped>
#header{
    display: flex;
    font-weight: bold;
    user-select: none;
}
#body{
    transition: max-height 0.2s;
    overflow: hidden;
    max-height: 1000px;
}
.closed{
    max-height: 0px !important;
}

#openclose{
    padding: 5px;
    line-height: 25px;
    cursor: pointer;
}


</style>