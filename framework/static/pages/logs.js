var Logs = Vue.component("Logs", {
    template: `
    <div class="container justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom" id="status">

        <h1 class="h2">Logs</h1>

            <textarea v-bind:style="textareaStyle">{{ log }}</textarea>
    </div>
  `,
    data: function () {
      return {
        textareaStyle: {
          fontSize: '13px',
          fontFamily: 'Courier New',
          width: '100%',
          height: '500px',
        },
        log: "",
        timer: ""
      }
    },
    // define methods under the `methods` object
    methods: {
        fetch: function (event) {

            this.$http.get('/log').then(response => {

                console.log(response);

                // get the Log
                this.log = "";
                for (var i = response.body.log.length-1; i >= 0; i--) {
                    var line = response.body.log[i];
                    this.log += line + "\n";
                }

            }, response => {
                // error callback
                console.log(response.body);
            });
        },
        cancelAutoUpdate: function () {
            clearInterval(this.timer);
        }
    },
    beforeDestroy() {
        clearInterval(this.timer)
    },
    beforeMount() {
        this.fetch();
        this.timer = setInterval(this.fetch, 1000);
    }
});