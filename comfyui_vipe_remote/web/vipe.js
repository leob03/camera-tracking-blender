import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ViPE.LocalReview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ViPEVideoInput") {
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                created?.apply(this, arguments);
                const path = this.widgets.find(w => w.name === "video");
                const name = this.widgets.find(w => w.name === "source_name");
                const button = this.addWidget("button", "Upload video…", null, () => {
                    const picker = document.createElement("input");
                    picker.type = "file";
                    picker.accept = ".mp4,.mov,.mkv,.avi,.m4v,.webm";
                    picker.onchange = async () => {
                        const file = picker.files?.[0];
                        if (!file) return;
                        button.name = "Uploading video…";
                        this.setDirtyCanvas(true, true);
                        try {
                            const response = await api.fetchApi("/vipe/upload?" + new URLSearchParams({ filename: file.name }), {
                                method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file
                            });
                            if (!response.ok) throw new Error(await response.text());
                            const result = await response.json();
                            path.value = result.video;
                            name.value = file.name;
                            app.graph.change();
                        } catch (error) {
                            alert("Video upload failed: " + error.message);
                        } finally {
                            button.name = "Upload video…";
                            this.setDirtyCanvas(true, true);
                        }
                    };
                    picker.click();
                }, { serialize: false });
            };
        }
        if (nodeData.name === "ViPETrackVideo") {
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                created?.apply(this, arguments);
                const panel = document.createElement("div");
                panel.style.cssText = "padding:10px;box-sizing:border-box;color:#e2e8f0;background:#171f2b;font:13px/1.5 sans-serif;white-space:pre-wrap;";
                const message = document.createElement("div");
                message.textContent = "Run this workflow to create a camera and native-cloud result.";
                panel.append(message);
                const links = document.createElement("div");
                links.style.cssText = "display:flex;gap:14px;margin-top:10px;";
                panel.append(links);
                const widget = this.addDOMWidget("solve_result", "div", panel, { serialize: false });
                widget.computeSize = () => [this.size[0], 125];
                const executed = this.onExecuted;
                this.onExecuted = function (output) {
                    executed?.apply(this, arguments);
                    message.textContent = output.vipe_summary?.[0] || "Camera and native points ready.";
                    links.replaceChildren();
                    const result = output.vipe_result?.[0];
                    if (!result) return;
                    const download = document.createElement("a");
                    download.textContent = "Download Blender bundle";
                    download.style.color = "#7dd3fc";
                    download.href = api.apiURL("/view?" + new URLSearchParams({ filename: result.filename, subfolder: result.subfolder, type: result.type }));
                    download.download = "vipe-" + result.job_id.slice(0, 8) + ".zip";
                    const log = document.createElement("a");
                    log.textContent = "Worker log";
                    log.style.color = "#7dd3fc";
                    log.href = api.apiURL("/vipe/jobs/" + encodeURIComponent(result.job_id) + "/log");
                    log.target = "_blank";
                    links.append(download, log);
                    this.setDirtyCanvas(true, true);
                };
            };
        }
    }
});
