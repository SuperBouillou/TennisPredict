// TennisPredict — ui.js (refonte v2)

document.addEventListener('DOMContentLoaded', function() {

  // ── Segmented controls — toggle .on class ──────────────────────────────────
  document.querySelectorAll('.tp-seg').forEach(function(seg) {
    seg.querySelectorAll('button[data-val]').forEach(function(btn) {
      btn.addEventListener('click', function() {
        seg.querySelectorAll('button').forEach(function(b) { b.classList.remove('on'); });
        btn.classList.add('on');
      });
    });
  });

  // ── Stats tabs ─────────────────────────────────────────────────────────────
  var tabBtns = document.querySelectorAll('.tp-tab-btn');
  var tabPanels = document.querySelectorAll('.tp-tab-panel');
  tabBtns.forEach(function(btn) {
    btn.addEventListener('click', function() {
      var target = btn.dataset.tab;
      tabBtns.forEach(function(b) { b.classList.remove('active'); });
      tabPanels.forEach(function(p) { p.classList.remove('active'); });
      btn.classList.add('active');
      var panel = document.querySelector('.tp-tab-panel[data-tab="' + target + '"]');
      if (panel) panel.classList.add('active');
      // Update URL param
      var url = new URL(window.location);
      url.searchParams.set('tab', target);
      history.replaceState({}, '', url);
    });
  });

  // ── Init active tab from URL ───────────────────────────────────────────────
  var urlTab = new URLSearchParams(window.location.search).get('tab');
  if (urlTab) {
    var activeBtn = document.querySelector('.tp-tab-btn[data-tab="' + urlTab + '"]');
    if (activeBtn) activeBtn.click();
  } else if (tabBtns.length > 0) {
    tabBtns[0].click();
  }

  // ── FAQ accordion ──────────────────────────────────────────────────────────
  document.querySelectorAll('.tp-faq-item').forEach(function(item) {
    var q = item.querySelector('.tp-faq-q');
    var a = item.querySelector('.tp-faq-a');
    if (q && a) {
      a.style.display = 'none';
      q.style.cursor = 'pointer';
      q.addEventListener('click', function() {
        var open = a.style.display !== 'none';
        a.style.display = open ? 'none' : 'block';
        q.classList.toggle('open', !open);
      });
    }
  });

  // ── Sortable table ─────────────────────────────────────────────────────────
  var _col = null, _asc = true;
  var NUMS = new Set(['odd','stake','pnl']);
  window.sortTable = function(th) {
    var col = th.dataset.col;
    _asc = (_col === col) ? !_asc : true;
    _col = col;
    document.querySelectorAll('.sort-ind').forEach(function(s) { s.textContent = ''; });
    var ind = th.querySelector('.sort-ind');
    if (ind) ind.textContent = _asc ? ' ▲' : ' ▼';
    var tbody = th.closest('table').querySelector('tbody');
    if (!tbody) return;
    Array.from(tbody.querySelectorAll('tr')).sort(function(a, b) {
      var av = a.dataset[col] || '', bv = b.dataset[col] || '';
      if (NUMS.has(col)) { av = parseFloat(av)||0; bv = parseFloat(bv)||0; return _asc ? av-bv : bv-av; }
      return _asc ? av.localeCompare(bv,'fr',{sensitivity:'base'}) : bv.localeCompare(av,'fr',{sensitivity:'base'});
    }).forEach(function(r) { tbody.appendChild(r); });
  };

  // ── Tooltip ────────────────────────────────────────────────────────────────
  // Simple tooltip from data-tip attribute
  var tip = document.createElement('div');
  tip.id = 'tp-tooltip';
  tip.style.cssText = [
    'position:fixed','z-index:9999','pointer-events:none',
    'background:var(--bg-elev)','border:1px solid var(--line-2)',
    'border-radius:4px','padding:5px 10px',
    'font-family:var(--font-mono)','font-size:11px','color:var(--fg-2)',
    'max-width:220px','white-space:pre-wrap','line-height:1.45',
    'box-shadow:0 4px 16px rgba(0,0,0,.35)','opacity:0','transition:opacity .12s',
  ].join(';');
  document.body.appendChild(tip);

  document.addEventListener('mouseover', function(e) {
    var el = e.target.closest('[data-tip]');
    if (!el) { tip.style.opacity = '0'; return; }
    tip.textContent = el.dataset.tip;
    tip.style.opacity = '1';
  });
  document.addEventListener('mousemove', function(e) {
    var x = e.clientX + 12, y = e.clientY - 28;
    var tw = tip.offsetWidth, th2 = tip.offsetHeight;
    if (x + tw > window.innerWidth - 8) x = e.clientX - tw - 12;
    if (y < 8) y = e.clientY + 16;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  });
  document.addEventListener('mouseout', function(e) {
    if (!e.relatedTarget || !e.relatedTarget.closest('[data-tip]')) {
      tip.style.opacity = '0';
    }
  });

  // ── Modal helpers ──────────────────────────────────────────────────────────
  window.closeModal = function() {
    var overlay = document.getElementById('modal-overlay');
    if (overlay) overlay.style.display = 'none';
  };
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
  });

  // ── Landing scroll-reveal ──────────────────────────────────────────────────
  var revEls = document.querySelectorAll('.reveal');
  if (revEls.length) {
    if (!('IntersectionObserver' in window)) {
      revEls.forEach(function(el) { el.classList.add('visible'); });
    } else {
      var revObs = new IntersectionObserver(function(entries) {
        entries.forEach(function(e) {
          if (e.isIntersecting) {
            e.target.classList.add('visible');
            revObs.unobserve(e.target);
          }
        });
      }, { threshold: 0.10, rootMargin: '0px 0px -24px 0px' });
      revEls.forEach(function(el) { revObs.observe(el); });
    }
  }

  // ── Counter animation (landing metrics) ───────────────────────────────────
  var counterEls = document.querySelectorAll('.landing-metric-val[data-target]');
  if (counterEls.length && 'IntersectionObserver' in window) {
    var cntObs = new IntersectionObserver(function(entries) {
      entries.forEach(function(e) {
        if (e.isIntersecting) {
          animateCounter(e.target);
          cntObs.unobserve(e.target);
        }
      });
    }, { threshold: 0.5 });
    counterEls.forEach(function(el) { cntObs.observe(el); });
  } else {
    counterEls.forEach(function(el) {
      el.textContent = (el.dataset.prefix || '') + parseFloat(el.dataset.target).toFixed(parseInt(el.dataset.decimals,10)||0) + (el.dataset.suffix || '');
    });
  }

  function animateCounter(el) {
    var target = parseFloat(el.dataset.target);
    var decimals = parseInt(el.dataset.decimals, 10) || 0;
    var suffix = el.dataset.suffix || '';
    var prefix = el.dataset.prefix || '';
    var duration = 1400;
    var start = null;
    function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
    function tick(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / duration, 1);
      el.textContent = prefix + (easeOut(p) * target).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

});
