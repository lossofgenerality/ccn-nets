/* The CCN Net Lab TA: the client shared by every "Ask the TA" panel and the lab-wide TA page.
   © Computational Classnotes · CC BY-NC-SA 4.0
   By Dara O Shayda and Hikari Noor Hoshino (星野ヌール), AI System Programming Laboratory

   The page sets window.CCN_NETLAB_CONFIG = {MODEL, KERNEL_URL} first in <head>, and calls
   CCNTA.mount(element, {knowledge: "<url of a knowledge JSON>", title, chooser}).
   The conversation goes to NetLabKernel (mode "chat"); when the TA asks to run Wolfram code,
   the page sends it to the same kernel (mode "run", sandboxed) and returns the result to the TA.
   Only the CCN Cloud pass key is typed on the page. Nothing is stored but, where the browser
   allows it, the pass key for this tab (sessionStorage). */
(function () {
  "use strict";
  const CFG = window.CCN_NETLAB_CONFIG || {};
  const KERNEL_URL = String(CFG.KERNEL_URL || "");
  const MODELS = [["claude-opus-5-5", "Opus 5.5 (deepest)"], ["claude-sonnet-5-5", "Sonnet 5.5 (faster)"], ["claude-haiku-5-5", "Haiku 5.5 (quickest)"]];
  const MAX_ROUNDS = 6;          // tool rounds in one answer
  const KEEP_TURNS = 12;         // exchanges kept in the conversation

  // ---------- small helpers ----------
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function b64enc(str) { const b = new TextEncoder().encode(str); let s = ""; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); return btoa(s); }
  function b64dec(b) { return new TextDecoder().decode(Uint8Array.from(atob(b), c => c.charCodeAt(0))); }
  const cleanKey = k => String(k || "").replace(/[\s"'“”‘’`]/g, "");
  function store(k, v) { try { if (v === undefined) return sessionStorage.getItem(k); sessionStorage.setItem(k, v); } catch (e) { return null; } }

  async function kernel(params, key, signal) {
    if (!KERNEL_URL) throw new Error("This page has no kernel address (CCN_NETLAB_CONFIG.KERNEL_URL).");
    const r = await fetch(KERNEL_URL, { method: "POST", body: new URLSearchParams({ key: cleanKey(key), ...params }), signal });
    const j = await r.json().catch(() => null);
    if (!j) throw new Error("The kernel's reply could not be read (HTTP " + r.status + ").");
    if (j.Success === false || j.Failure || j.Message === "bad key") throw new Error(j.Message === "bad key" ? "The pass key was not accepted." : (j.Message || "Kernel error"));
    return j;
  }

  // ---------- Markdown, enough for the TA's answers ----------
  function inline(s) {
    return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>").replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }
  function md(text) {
    const lines = String(text).replace(/\r/g, "").split("\n"); const out = []; let i = 0;
    while (i < lines.length) {
      const l = lines[i];
      if (/^```/.test(l)) { const lang = l.slice(3).trim(); const buf = []; i++; while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]); i++; out.push('<pre><code data-lang="' + esc(lang) + '">' + esc(buf.join("\n")) + "</code></pre>"); continue; }
      if (/^\$\$/.test(l.trim())) { const buf = [l]; if (!/^\$\$.*\$\$\s*$/.test(l.trim()) || l.trim() === "$$") { i++; while (i < lines.length && !/\$\$\s*$/.test(lines[i])) buf.push(lines[i++]); if (i < lines.length) buf.push(lines[i]); } i++; out.push('<div class="ta-math">' + esc(buf.join("\n")) + "</div>"); continue; }
      if (/^#{1,4}\s/.test(l)) { const n = l.match(/^#+/)[0].length; out.push("<h" + (n + 2) + ">" + inline(l.replace(/^#+\s*/, "")) + "</h" + (n + 2) + ">"); i++; continue; }
      if (/^\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        const row = r => r.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map(c => inline(c.trim().replace(/\\\|/g, "|")));
        let h = "<table><thead><tr>" + row(l).map(c => "<th>" + c + "</th>").join("") + "</tr></thead><tbody>"; i += 2;
        while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) h += "<tr>" + row(lines[i++]).map(c => "<td>" + c + "</td>").join("") + "</tr>";
        out.push(h + "</tbody></table>"); continue;
      }
      if (/^\s*(\d+\.|[a-z]\.|[-*])\s/.test(l)) {
        const buf = []; while (i < lines.length && /^\s*(\d+\.|[a-z]\.|[-*])\s/.test(lines[i])) { const m = lines[i].match(/^(\s*)(\d+\.|[a-z]\.|[-*])\s+(.*)$/); buf.push('<div class="ta-li" style="margin-left:' + (m[1].length * 0.6) + 'em"><span class="ta-mark">' + esc(/[-*]/.test(m[2]) ? "·" : m[2]) + "</span> " + inline(m[3]) + "</div>"); i++; }
        out.push(buf.join("")); continue;
      }
      if (l.trim() === "") { i++; continue; }
      const buf = [l]; i++; while (i < lines.length && lines[i].trim() !== "" && !/^(```|#|\||\$\$|\s*(\d+\.|[a-z]\.|[-*])\s)/.test(lines[i])) buf.push(lines[i++]);
      out.push("<p>" + inline(buf.join(" ")) + "</p>");
    }
    return out.join("\n");
  }
  function typeset(el) {
    if (window.renderMathInElement) { try { window.renderMathInElement(el, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false }); } catch (e) { } }
  }

  // ---------- the TA ----------
  const TOOL = {
    name: "run_wolfram",
    description: "Evaluate Wolfram Language code in the CCN Cloud kernel (NetLabKernel) and return its result, its messages and, for graphics, an image. " +
      "The kernel is a sandbox: anything that writes, deletes, reads files or cloud objects, reaches the network, starts processes, changes the kernel for later calls " +
      "or turns a string into code is refused before evaluation. Each call starts afresh unless the kernel happens to be reused; define what you need in the same call. " +
      "Set preload to \"TensorNets\" to load ff_tensornets_new.nb (tnParse, tnSimplify, tnEvaluate, tnSolve, tnLex and the rest; about 1 s) or to \"SubjectNet\" " +
      "to load MotionPhrases.nb and SubjectNet.nb (MotionFind, MotionParse, MotionTransform, MotionSubject, SubjectCheckA, SubjectCheckB; about 17 s, since it trains both versions). " +
      "Time limit 50 s; output cut at 20,000 characters.",
    input_schema: { type: "object", properties: { code: { type: "string", description: "Wolfram Language code; the value of the last expression is returned" }, preload: { type: "string", enum: ["", "TensorNets", "SubjectNet"], description: "project code to load first" } }, required: ["code"] }
  };

  function systemBlocks(K, here) {
    return [
      { type: "text", text: K.rules + "\n\nYou are answering on: " + here + "." },
      { type: "text", text: "THE KNOWLEDGE PACK (" + K.title + ", built " + K.built + "):\n\n" + K.documents.map(d => "=== " + d.name + " (" + d.source + ") ===\n" + d.text).join("\n\n"), cache_control: { type: "ephemeral" } }
    ];
  }

  // the last KEEP_TURNS exchanges, each cut only where a question of the owner's begins, so that a tool
  // request is never separated from its result
  function recent(h) {
    const starts = []; h.forEach((m, i) => { if (m.role === "user" && typeof m.content === "string") starts.push(i); });
    return starts.length > KEEP_TURNS ? h.slice(starts[starts.length - KEEP_TURNS]) : h.slice();
  }

  function mount(root, opts) {
    root.classList.add("ta-root");
    root.innerHTML =
      '<div class="ta-head"><strong>' + esc(opts.title || "Ask the TA") + '</strong>' +
      (opts.chooser ? ' <select class="ta-scope" title="What the TA reads"></select>' : "") +
      ' <select class="ta-model" title="Model">' + MODELS.map(m => '<option value="' + m[0] + '"' + (m[0] === (CFG.MODEL || MODELS[0][0]) ? " selected" : "") + ">" + m[1] + "</option>").join("") + "</select>" +
      ' <button class="ta-new" type="button">New chat</button></div>' +
      '<div class="ta-key"><label>CCN Cloud pass key <input type="password" autocomplete="off" spellcheck="false"></label> <span class="ta-status"></span></div>' +
      '<div class="ta-log" aria-live="polite"></div>' +
      '<div class="ta-starters"></div>' +
      '<form class="ta-form"><textarea rows="3" placeholder="Ask about the maths, the configuration, the code or a change you are considering. Ctrl+Enter sends."></textarea>' +
      '<div class="ta-actions"><button type="submit" class="ta-send">Send</button> <button type="button" class="ta-stop" disabled>Stop</button>' +
      ' <span class="ta-note">An examination measures how well the TA keeps to its sources; verify anything you will rely on.</span></div></form>';
    const $ = s => root.querySelector(s);
    const keyEl = $(".ta-key input"), log = $(".ta-log"), ta = $("textarea"), statusEl = $(".ta-status");
    keyEl.value = store("ccn-netlab-key") || "";
    keyEl.addEventListener("change", () => store("ccn-netlab-key", cleanKey(keyEl.value)));
    let K = null, history = [], busy = null, scopeUrl = opts.knowledge, packStatus = "";

    async function loadKnowledge(url) {
      statusEl.textContent = "loading the knowledge pack…";
      const r = await fetch(url); K = await r.json();
      packStatus = statusEl.textContent = K.documents.length + " documents, " + Math.round(K.documents.reduce((a, d) => a + d.text.length, 0) / 1000) + "k characters";
      $(".ta-starters").innerHTML = (K.starters || []).map(s => '<button type="button">' + esc(s) + "</button>").join("");
      $(".ta-starters").querySelectorAll("button").forEach(b => b.onclick = () => { ta.value = b.textContent; send(); });
    }
    if (opts.chooser) {
      const sel = $(".ta-scope");
      sel.innerHTML = opts.chooser.map(c => '<option value="' + esc(c[1]) + '">' + esc(c[0]) + "</option>").join("");
      sel.onchange = () => { scopeUrl = sel.value; history = []; log.innerHTML = ""; loadKnowledge(scopeUrl); };
    }
    loadKnowledge(scopeUrl).catch(e => statusEl.textContent = "The knowledge pack could not be loaded: " + e.message);

    function bubble(cls, html) { const d = document.createElement("div"); d.className = "ta-msg " + cls; d.innerHTML = html; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }

    async function runTool(input, signal) {
      const t0 = performance.now();
      const j = await kernel({ mode: "run", code_b64: b64enc(input.code || ""), preload: input.preload || "" }, keyEl.value, signal);
      const text = j.text_b64 ? b64dec(j.text_b64) : "", msgs = j.messages_b64 ? b64dec(j.messages_b64) : "";
      return { ok: j.ok !== false, text, msgs, image: j.image || "", seconds: j.seconds != null ? j.seconds : ((performance.now() - t0) / 1000).toFixed(1) };
    }

    async function send() {
      const q = ta.value.trim(); if (!q || busy) return;
      if (!cleanKey(keyEl.value)) { statusEl.textContent = "Type the CCN Cloud pass key first."; keyEl.focus(); return; }
      if (!K) { statusEl.textContent = "The knowledge pack is still loading."; return; }
      store("ccn-netlab-key", cleanKey(keyEl.value)); statusEl.textContent = packStatus;
      ta.value = ""; bubble("ta-you", md(q));
      history.push({ role: "user", content: q });
      const ctl = new AbortController(); busy = ctl; $(".ta-stop").disabled = false; $(".ta-send").disabled = true;
      const answer = bubble("ta-bot", '<span class="ta-wait">thinking…</span>');
      try {
        for (let round = 0; round <= MAX_ROUNDS; round++) {
          const payload = { model: $(".ta-model").value, max_tokens: 8000, system: systemBlocks(K, opts.here || K.title), messages: recent(history), tools: round < MAX_ROUNDS ? [K.tool || TOOL] : [] };
          const j = await kernel({ mode: "chat", payload_b64: b64enc(JSON.stringify(payload)) }, keyEl.value, ctl.signal);
          if (j.error) throw new Error(j.error === "no chat key stored" ? "The kernel has no chat key stored yet (see ff_netlabkernel.nb, step 1)." : j.error);
          const content = JSON.parse(b64dec(j.content_b64));
          history.push({ role: "assistant", content });
          const texts = content.filter(c => c.type === "text").map(c => c.text).join("\n\n");
          const uses = content.filter(c => c.type === "tool_use");
          if (texts) { const w = answer.querySelector(".ta-wait"); if (w) w.remove(); const d = document.createElement("div"); d.innerHTML = md(texts); answer.appendChild(d); typeset(d); }
          if (j.stop_reason !== "tool_use" || !uses.length) break;
          const results = [];
          for (const u of uses) {
            const box = document.createElement("details"); box.className = "ta-run";
            box.innerHTML = "<summary>Ran in Wolfram" + (u.input.preload ? " (with " + esc(u.input.preload) + ")" : "") + "…</summary><pre><code>" + esc(u.input.code || "") + "</code></pre>";
            answer.appendChild(box); log.scrollTop = log.scrollHeight;
            let r; try { r = await runTool(u.input, ctl.signal); } catch (e) { if (e.name === "AbortError") throw e; r = { ok: false, text: "The kernel call failed: " + e.message, msgs: "", image: "", seconds: "" }; }
            box.querySelector("summary").textContent = (r.ok ? "Ran in Wolfram" : "Not run") + (u.input.preload ? " (with " + u.input.preload + ")" : "") + (r.seconds !== "" ? ", " + r.seconds + " s" : "");
            box.insertAdjacentHTML("beforeend", '<pre class="ta-out">' + esc(r.text) + (r.msgs ? "\n\n" + esc(r.msgs) : "") + "</pre>" + (r.image ? '<img alt="Wolfram graphics" src="data:image/png;base64,' + r.image + '">' : ""));
            const block = [{ type: "text", text: (r.ok ? "" : "REFUSED OR FAILED: ") + r.text + (r.msgs ? "\nMessages:\n" + r.msgs : "") + (r.seconds !== "" ? "\n(" + r.seconds + " s)" : "") }];
            if (r.image) block.push({ type: "image", source: { type: "base64", media_type: "image/png", data: r.image } });
            results.push({ type: "tool_result", tool_use_id: u.id, content: block, is_error: !r.ok });
          }
          history.push({ role: "user", content: results });
        }
      } catch (e) {
        const w = answer.querySelector(".ta-wait"); if (w) w.remove();
        answer.insertAdjacentHTML("beforeend", '<p class="ta-err">' + esc(e.name === "AbortError" ? "Stopped." : e.message) + "</p>");
        // keep the conversation well formed: drop an unanswered tool request
        while (history.length && history[history.length - 1].role === "assistant" && (history[history.length - 1].content || []).some && history[history.length - 1].content.some(c => c.type === "tool_use")) history.pop();
        if (history.length && history[history.length - 1].role === "user" && typeof history[history.length - 1].content === "string") history.pop();
      } finally { busy = null; $(".ta-stop").disabled = true; $(".ta-send").disabled = false; }
    }
    $(".ta-form").addEventListener("submit", e => { e.preventDefault(); send(); });
    ta.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); } });
    $(".ta-stop").onclick = () => busy && busy.abort();
    $(".ta-new").onclick = () => { if (busy) busy.abort(); history = []; log.innerHTML = ""; };
    return { send: q => { ta.value = q; return send(); }, md };
  }
  window.CCNTA = { mount, md, kernel, b64enc, b64dec, TOOL, systemBlocks };
})();
