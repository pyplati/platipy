var Logs = Vue.component("Logs", {
    template: `
    <div class="container justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">

        <h1 class="h2">Logs</h1>
        <div class="row">
            <div class="col-sm-3">
                <input type="checkbox" id="chkError" v-model="logLevel.error">
                <label for="chkError">Error</label>
            </div>
            <div class="col-sm-3">
                <input type="checkbox" id="chkWarn" v-model="logLevel.warning">
                <label for="chkWarn">Warning</label>
            </div>
            <div class="col-sm-3">
                <input type="checkbox" id="chkInfo" v-model="logLevel.info">
                <label for="chkInfo">Info</label>
            </div>
            <div class="col-sm-3">
                <input type="checkbox" id="chkDebug" v-model="logLevel.debug">
                <label for="chkDebug">Debug</label>
            </div>
        </div>
        <textarea v-bind:style="textareaStyle">{{ log | level(logLevel) }}</textarea>
    </div>
  `,
  filters: {
      level: function (log, logLevel) {

        var result = "";
        for (var i = log.length-1; i >= 0; i--) {
            var line = log[i];

            if(!logLevel.error && line.includes("| ERROR")) {
                continue;
            }
            if(!logLevel.warning && line.includes("| WARNING")) {
                continue;
            }
            if(!logLevel.info && line.includes("| INFO")) {
                continue;
            }
            if(!logLevel.debug && line.includes("| DEBUG")) {
                continue;
            }
            result += line + "\n";
        }

        return result
      }
    },
    data: function () {
      return {
        textareaStyle: {
          fontSize: '13px',
          fontFamily: 'Courier New',
          width: '100%',
          height: '500px',
        },
        log: "",
        timer: "",
        logLevel: {
            error: true, 
            warning: true,
            info: true,
            debug: false
        }
      }
    },
    // define methods under the `methods` object
    methods: {
        fetch: function (event) {

            this.$http.get('/log').then(response => {

                console.log(response);

                // get the Log
                this.log = response.body.log;

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