"""Shared UI components for Gradio interface."""

import gradio as gr


def audio_output_row(label: str = "Output"):
    """Create a standard audio output component."""
    return gr.Audio(label=label, type="filepath")


def model_status_display():
    """Create a model status textbox."""
    return gr.Textbox(label="Status", interactive=False, lines=1)


def progress_display():
    """Create a progress info box."""
    return gr.Textbox(label="Progress", interactive=False, lines=1)


# ---------------------------------------------------------------------------
# Global Audio Player Footer
# ---------------------------------------------------------------------------

AUDIO_PLAYER_FOOTER_HTML = (
    '<div id="ag-audio-footer" class="ag-audio-footer">'
    '  <div class="ag-footer-track-info">'
    '    <span class="ag-footer-track-name" id="ag-footer-track-name">No track selected</span>'
    '    <span class="ag-footer-track-time" id="ag-footer-track-time">0:00 / 0:00</span>'
    '  </div>'
    '  <div class="ag-footer-controls">'
    '    <button class="ag-footer-btn" id="ag-btn-stop" title="Stop">&#9632;</button>'
    '    <button class="ag-footer-btn ag-footer-btn-play" id="ag-btn-play" title="Play/Pause (Space)">&#9654;</button>'
    '  </div>'
    '  <div class="ag-footer-progress-wrap">'
    '    <div class="ag-footer-progress-bar" id="ag-footer-progress-bar">'
    '      <div class="ag-footer-progress-fill" id="ag-footer-progress-fill" style="width:0%"></div>'
    '    </div>'
    '  </div>'
    '  <div class="ag-footer-volume-wrap">'
    '    <span class="ag-footer-volume-icon" id="ag-volume-icon">&#128264;</span>'
    '    <input type="range" id="ag-volume-slider" class="ag-footer-volume-slider" min="0" max="100" value="80" />'
    '  </div>'
    '  <audio id="ag-footer-audio" preload="auto"></audio>'
    '</div>'
    '<script>'
    '(function() {'
    '  var audio   = document.getElementById("ag-footer-audio");'
    '  var playBtn = document.getElementById("ag-btn-play");'
    '  var stopBtn = document.getElementById("ag-btn-stop");'
    '  var nameEl  = document.getElementById("ag-footer-track-name");'
    '  var timeEl  = document.getElementById("ag-footer-track-time");'
    '  var fillEl  = document.getElementById("ag-footer-progress-fill");'
    '  var volIcon = document.getElementById("ag-volume-icon");'
    '  var volSli  = document.getElementById("ag-volume-slider");'
    ''
    '  function fmt(sec) {'
    '    if (!sec || isNaN(sec)) return "0:00";'
    '    var m = Math.floor(sec / 60);'
    '    var s = Math.floor(sec % 60);'
    '    return m + ":" + (s < 10 ? "0" : "") + s;'
    '  }'
    ''
    '  window.agPlayFile = function(url, name) {'
    '    if (!url) return;'
    '    audio.src = url;'
    '    nameEl.textContent = name || url.split("/").pop() || "Unknown";'
    '    audio.play();'
    '  };'
    ''
    '  playBtn.addEventListener("click", function() {'
    '    if (!audio.src) return;'
    '    if (audio.paused) { audio.play(); } else { audio.pause(); }'
    '  });'
    '  stopBtn.addEventListener("click", function() {'
    '    if (!audio.src) return;'
    '    audio.pause();'
    '    audio.currentTime = 0;'
    '  });'
    ''
    '  audio.volume = 0.8;'
    '  volSli.addEventListener("input", function() {'
    '    audio.volume = this.value / 100;'
    '    if (this.value == 0) volIcon.innerHTML = "&#128263;";'
    '    else if (this.value < 50) volIcon.innerHTML = "&#128265;";'
    '    else volIcon.innerHTML = "&#128266;";'
    '  });'
    ''
    '  audio.addEventListener("timeupdate", function() {'
    '    timeEl.textContent = fmt(audio.currentTime) + " / " + fmt(audio.duration);'
    '    if (audio.duration) {'
    '      fillEl.style.width = (audio.currentTime / audio.duration * 100) + "%";'
    '    }'
    '  });'
    '  audio.addEventListener("ended", function() { playBtn.innerHTML = "&#9654;"; });'
    '  audio.addEventListener("play",  function() { playBtn.innerHTML = "&#10074;&#10074;"; });'
    '  audio.addEventListener("pause", function() { playBtn.innerHTML = "&#9654;"; });'
    ''
    '  document.getElementById("ag-footer-progress-bar").addEventListener("click", function(e) {'
    '    if (!audio.duration) return;'
    '    var rect = this.getBoundingClientRect();'
    '    var pct = (e.clientX - rect.left) / rect.width;'
    '    audio.currentTime = pct * audio.duration;'
    '  });'
    ''
    '  function hookAudioComponents() {'
    '    document.querySelectorAll("audio source").forEach(function(src) {'
    '      var parentAudio = src.closest("audio");'
    '      if (parentAudio && !parentAudio._agHooked) {'
    '        parentAudio._agHooked = true;'
    '        var container = parentAudio.parentElement;'
    '        if (container && !container.querySelector(".ag-play-in-footer")) {'
    '          var btn = document.createElement("button");'
    '          btn.className = "ag-play-in-footer";'
    '          btn.title = "Play in footer bar";'
    '          btn.innerHTML = "&#127925;";'
    '          btn.addEventListener("click", function(ev) {'
    '            ev.preventDefault();'
    '            var srcEl = parentAudio.querySelector("source");'
    '            var url = srcEl ? srcEl.src : parentAudio.src;'
    '            var name = "Audio";'
    '            var lbl = container.closest("[data-testid]");'
    '            if (lbl) { var lo = lbl.querySelector(".sr-only, label"); if (lo) name = lo.textContent.trim(); }'
    '            window.agPlayFile(url, name);'
    '          });'
    '          container.style.position = "relative";'
    '          container.appendChild(btn);'
    '        }'
    '      }'
    '    });'
    '  }'
    ''
    '  setInterval(hookAudioComponents, 2000);'
    '  setTimeout(hookAudioComponents, 1000);'
    '})();'
    '<\/script>'
)


def create_audio_player_footer() -> gr.HTML:
    """Render the global audio-player footer bar (call once at the bottom of the app)."""
    return gr.HTML(AUDIO_PLAYER_FOOTER_HTML)


# ---------------------------------------------------------------------------
# Keyboard Shortcuts Handler
# ---------------------------------------------------------------------------

KEYBOARD_SHORTCUTS_HTML = (
    '<script>'
    '(function() {'
    '  window.agToast = function(msg, duration) {'
    '    duration = duration || 2000;'
    '    var t = document.getElementById("ag-toast");'
    '    if (!t) { t = document.createElement("div"); t.id = "ag-toast"; document.body.appendChild(t); }'
    '    t.textContent = msg;'
    '    t.classList.add("ag-toast-visible");'
    '    clearTimeout(t._timer);'
    '    t._timer = setTimeout(function() { t.classList.remove("ag-toast-visible"); }, duration);'
    '  };'
    ''
    '  document.addEventListener("keydown", function(e) {'
    '    var tag = (e.target.tagName || "").toLowerCase();'
    '    var isInput = (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable);'
    ''
    '    /* Ctrl+S */'
    '    if ((e.ctrlKey || e.metaKey) && e.key === "s") {'
    '      e.preventDefault();'
    '      var saveBtn = document.querySelector("button[id*=\\"save_project\\" i]");'
    '      if (saveBtn) { saveBtn.click(); window.agToast("Project saved!"); }'
    '      else { window.agToast("No save action available"); }'
    '      return;'
    '    }'
    ''
    '    /* Escape */'
    '    if (e.key === "Escape") {'
    '      var closeBtn = document.querySelector(".modal button.close, [aria-label=\\"Close\\"]");'
    '      if (closeBtn) closeBtn.click();'
    '      return;'
    '    }'
    ''
    '    if (isInput) return;'
    ''
    '    /* Space */'
    '    if (e.key === " " || e.code === "Space") {'
    '      e.preventDefault();'
    '      var footer = document.getElementById("ag-footer-audio");'
    '      if (footer && footer.src) {'
    '        var playBtn = document.getElementById("ag-btn-play");'
    '        if (playBtn) playBtn.click();'
    '      }'
    '      return;'
    '    }'
    ''
    '    /* Ctrl+Enter */'
    '    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {'
    '      e.preventDefault();'
    '      var activePanel = document.querySelector(".tabitem:not([style*=\\"display: none\\"])")'
    '        || document.querySelector(".tabitem:not([hidden])");'
    '      if (activePanel) {'
    '        var btn = activePanel.querySelector("button.primary, button[id*=\\"generate\\" i], button[id*=\\"process\\" i], button[id*=\\"run\\" i]");'
    '        if (btn) { btn.click(); window.agToast("Submitted!"); }'
    '      }'
    '      return;'
    '    }'
    ''
    '    /* 1-9 tab switch */'
    '    if (e.key >= "1" && e.key <= "9") {'
    '      var idx = parseInt(e.key, 10) - 1;'
    '      var tabBtns = document.querySelectorAll(".tab-nav button");'
    '      if (tabBtns[idx]) {'
    '        tabBtns[idx].click();'
    '        window.agToast("Tab " + e.key + ": " + tabBtns[idx].textContent.trim().substring(0, 30));'
    '      }'
    '      return;'
    '    }'
    '  });'
    '})();'
    '<\\/script>'
)


def create_keyboard_shortcuts() -> gr.HTML:
    """Render the keyboard-shortcut handler JS (call once at the top of the app)."""
    return gr.HTML(KEYBOARD_SHORTCUTS_HTML)


# ---------------------------------------------------------------------------
# Toast notification placeholder (a hidden div the JS targets)
# ---------------------------------------------------------------------------

TOAST_HTML = """<div id="ag-toast" class="ag-toast"></div>"""


def create_toast_placeholder() -> gr.HTML:
    """Render the toast notification container (call once at top of the app)."""
    return gr.HTML(TOAST_HTML)
