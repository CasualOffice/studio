/* Model Studio site: the hero's story loop, the Board walkthrough, the real
   narration player and the prompt demo. No libraries. Everything here stands
   down when the reader asks for less motion, and the page reads correctly
   with none of it running. */
(function () {
  "use strict";
  var still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  // Real numbers, taken from the narration that ships with this page.
  var PEAKS = [0.06,0.83,0.7,0.63,0.84,0.89,0.47,0.56,0.54,0.42,0.1,0.71,0.86,0.49,0.26,0.49,0.69,0.7,0.81,0.76,0.63,0.65,0.55,0.53,0.34,0.03,0.76,0.8,0.7,0.66,0.56,0.59,0.3,0.74,1,0.5,0.74,0.54,0.45,0.25,0,0,0,0.57,0.81,0.77,0.56,0.39,0.63,0.72,0.85,0.53,0.56,0.29,0,0,0.01,0.86,0.49,0.06,0,0.13,0.88,0.81,0.32,0.68,0.59,0.76,0.71,0.57,0.53,0.15,0,0,0,0.71,0.81,0.84,0.68,0.56,0.52,0.03,0.63,0.64,0.68,0.52,0.18,0];
  var SPANS = [
    { t: "She sat with her head thrown back upon the cushion of the chair, quite motionless, except when a sob came up into her throat and shook her, as a child who has cried itself to sleep continues to sob in its dreams.", s: 0, e: 13.3 },
    { t: "There was something coming to her and she was waiting for it, fearfully.", s: 13.8, e: 17.95 },
    { t: "What was it?", s: 18.07, e: 19.62 },
    { t: "She did not know; it was too subtle and elusive to name.", s: 19.74, e: 23.59 },
    { t: "She said it over and over under her breath: “Free, free, free!”", s: 24.09, e: 28.49 }
  ];

  function bars(el, peaks) {
    el.innerHTML = peaks.map(function (p) { return '<i style="--h:' + Math.max(p, 0.04) + '"></i>'; }).join("");
    return $$("i", el);
  }

  /* ---------------------------------------------------------- hero loop -- */
  var stage = $("#stage");
  if (stage) {
    var msText = $("#msText");
    var SEGMENTS = [
      { t: "Knowing that " }, { m: "Mrs. Mallard" }, { t: " was afflicted with a heart trouble, great care was taken to break to her as gently as possible the news of her husband’s death. It was her sister " },
      { m: "Josephine" }, { t: " who told her, in broken sentences. Her husband’s friend " }, { m: "Richards" }, { t: " was there, too, near her." }
    ];
    var total = SEGMENTS.reduce(function (n, s) { return n + (s.t || s.m).length; }, 0);
    var render = function (n, caret) {
      var out = "", left = n;
      SEGMENTS.forEach(function (s) {
        if (left <= 0) return;
        var text = s.t || s.m, piece = text.slice(0, left);
        left -= text.length;
        out += s.m ? "<mark>" + piece + "</mark>" : piece;
      });
      msText.innerHTML = out + (caret ? '<span class="caret"></span>' : "");
    };
    var heroBars = bars($("#heroBars"), PEAKS.filter(function (_, i) { return i % 2 === 0; }));
    var buttons = $$(".steps-mini button");
    var step = 1, timers = [], paused = false, visible = true, held = 0;
    var clear = function () { timers.forEach(clearTimeout); timers.forEach(clearInterval); timers = []; };
    var show = function (n) {
      clear(); step = n;
      stage.className = "stage s" + n;
      buttons.forEach(function (b) { b.toggleAttribute("aria-current", +b.dataset.step === n); if (+b.dataset.step === n) b.setAttribute("aria-current", "step"); });
      heroBars.forEach(function (b) { b.classList.remove("on"); });
      if (n === 1 && !still) {
        var typed = 0;
        timers.push(setInterval(function () {
          typed = Math.min(total, typed + 3);
          render(typed, true);
          if (typed >= total) { clear(); render(total, false); next(1400); }
        }, 28));
        return;
      }
      render(total, false);
      if (n === 4 && !still) {
        var lit = 0;
        timers.push(setInterval(function () {
          if (lit < heroBars.length) heroBars[lit++].classList.add("on");
        }, 70));
      }
      next([0, 0, 3000, 3400, 4600][n]);
    };
    var next = function (ms) {
      if (still) return;
      timers.push(setTimeout(function () {
        if (paused || !visible) { held = step; return; }
        show(step === 4 ? 1 : step + 1);
      }, ms));
    };
    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        paused = true; show(+b.dataset.step);
        clearTimeout(b._resume);
        b._resume = setTimeout(function () { paused = false; next(800); }, 9000);
      });
    });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (es) {
        visible = es[0].isIntersecting;
        if (visible && held) { held = 0; show(step === 4 ? 1 : step + 1); }
      }).observe(stage);
    }
    show(still ? 4 : 1);
  }

  /* ---------------------------------------------- Board walkthrough ----- */
  var frames = $$("#storyVisual .frame");
  $$(".mobile-visual").forEach(function (slot) {
    var source = $('#storyVisual [data-frame="' + slot.dataset.clone + '"]');
    if (source) { var copy = source.cloneNode(true); copy.classList.add("on"); slot.appendChild(copy); }
  });
  if (frames.length && "IntersectionObserver" in window) {
    var steps = $$(".story-steps .step");
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        var n = e.target.dataset.step;
        steps.forEach(function (s) { s.classList.toggle("on", s === e.target); });
        frames.forEach(function (f) { f.classList.toggle("on", f.dataset.frame === n); });
      });
    }, { rootMargin: "-45% 0px -45% 0px" });
    steps.forEach(function (s) { io.observe(s); });
  }

  /* ---------------------------------------------------- narration ------- */
  var audio = $("#narration");
  if (audio) {
    var wave = bars($("#deckWave"), PEAKS);
    var tr = $("#transcript");
    tr.innerHTML = SPANS.map(function (s, i) { return '<button type="button" data-i="' + i + '">' + s.t + "</button>"; }).join("");
    var lines = $$("button", tr);
    var play = $("#deckPlay"), icon = $("#deckIcon"), time = $("#deckTime"), wv = $("#deckWave");
    var dur = SPANS[SPANS.length - 1].e;
    var fmt = function (s) { s = Math.max(0, Math.floor(s)); return Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2); };
    var paint = function () {
      var t = audio.currentTime, d = audio.duration || dur, k = Math.floor(t / d * wave.length);
      wave.forEach(function (b, i) { b.classList.toggle("on", i < k); });
      lines.forEach(function (l, i) { l.classList.toggle("on", t >= SPANS[i].s - 0.05 && t < (SPANS[i + 1] ? SPANS[i + 1].s : d + 1) && (t > 0 || !audio.paused)); });
      time.textContent = fmt(t) + " / " + fmt(d);
      wv.setAttribute("aria-valuenow", Math.round(t));
    };
    var setIcon = function () {
      icon.innerHTML = audio.paused ? '<path d="M5 3l12 7-12 7z"/>' : '<path d="M5 3h3.5v14H5zM11.5 3H15v14h-3.5z"/>';
      play.setAttribute("aria-label", audio.paused ? "Play the narration" : "Pause the narration");
    };
    var tick = function () { paint(); if (!audio.paused) requestAnimationFrame(tick); };
    play.addEventListener("click", function () { audio.paused ? audio.play() : audio.pause(); });
    audio.addEventListener("play", function () { setIcon(); tick(); });
    audio.addEventListener("pause", setIcon);
    audio.addEventListener("ended", function () { setIcon(); paint(); });
    audio.addEventListener("loadedmetadata", paint);
    lines.forEach(function (l, i) {
      l.addEventListener("click", function () { audio.currentTime = SPANS[i].s; audio.play(); });
    });
    var seek = function (x) {
      var r = wv.getBoundingClientRect();
      audio.currentTime = Math.max(0, Math.min(1, (x - r.left) / r.width)) * (audio.duration || dur);
      paint();
    };
    wv.addEventListener("click", function (e) { seek(e.clientX); });
    wv.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        audio.currentTime = Math.max(0, audio.currentTime + (e.key === "ArrowRight" ? 3 : -3)); paint(); e.preventDefault();
      }
    });
    paint();
  }

  /* --------------------------------------------------- prompt demo ------ */
  // Verbatim replies from Qwen3.5 9B on an M4, as the app now shows them.
  var GOALS = {
    enhance: { label: "Enriched · for FLUX.2", time: "17.4 s",
      text: "Mira, [dressed in a vibrant red outfit], extends her hand to offer a [gleaming brass key] to Jonas, who [stands before her] wearing a [cool blue garment]. The interaction is intimate and direct, capturing the precise moment of the exchange. [Soft, natural light] highlights the metallic sheen of the key and the rich textures of their clothing, creating a warm and focused atmosphere. The composition centers on their faces and the key, framed with a [shallow depth of field] to emphasize their connection without distraction." },
    tighten: { label: "Clearer · nothing added", time: "6.3 s",
      text: "Mira gives Jonas a brass key. Mira wears red; Jonas wears blue.",
      hint: "Already clear, so it is left as you wrote it." },
    style: { label: "Vintage oil painting · one of three looks", time: "26.6 s",
      text: "Mira wearing a red outfit stands close to Jonas who is dressed in blue clothing. She extends her hand to give Jonas a shiny brass key. The scene is set in a softly lit room with neutral walls that highlight the vibrant colors of their clothes and the metallic gleam of the key. The camera captures a medium shot focusing on their interaction, framing both figures clearly as the key changes hands. [Vintage oil painting with visible brushstrokes.]",
      hint: "Also offered: cinematic realism, minimalist 3D." }
  };
  var phText = $("#phText");
  if (phText) {
    var esc = function (s) { return s.replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); };
    var setGoal = function (g, animate) {
      var d = GOALS[g];
      $$(".goals button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.goal === g)); });
      var fill = function () {
        phText.innerHTML = esc(d.text).replace(/\[([^\]]+)\]/g, "<ins>$1</ins>");
        $("#phLabel").textContent = d.label;
        $("#phTime").textContent = "Qwen3.5 9B · " + d.time;
        $("#phHint").textContent = d.hint || "";
        phText.classList.remove("fading");
      };
      if (animate && !still) { phText.classList.add("fading"); setTimeout(fill, 220); } else fill();
    };
    $$(".goals button").forEach(function (b) { b.addEventListener("click", function () { setGoal(b.dataset.goal, true); }); });
    setGoal("enhance", false);
  }

  /* -------------------------------------------- reveals and counters ---- */
  var count = function (el) {
    var to = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0), suf = el.dataset.suffix || "";
    if (still || !to) return;
    var t0 = performance.now();
    var f = function (now) {
      var k = Math.min(1, (now - t0) / 1100), v = to * (1 - Math.pow(1 - k, 3));
      el.textContent = v.toFixed(dec) + suf;
      if (k < 1) requestAnimationFrame(f);
    };
    requestAnimationFrame(f);
  };
  var targets = $$(".reveal, #fit, [data-count]");
  if ("IntersectionObserver" in window) {
    var seen = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add("seen");
        if (e.target.dataset.count !== undefined) count(e.target);
        seen.unobserve(e.target);
      });
    }, { threshold: 0.18 });
    targets.forEach(function (t) { seen.observe(t); });
  } else {
    targets.forEach(function (t) { t.classList.add("seen"); });
  }
})();
