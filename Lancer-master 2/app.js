import { GlobeEngine } from './globe.js';
import * as topojson from 'topojson-client';

const COUNTRY_DATA = [
  {name: "United States", code: "US", numId: "840", region: "Americas", lat: 38, lng: -97},
  {name: "China", code: "CN", numId: "156", region: "East Asia", lat: 35, lng: 105},
  {name: "India", code: "IN", numId: "356", region: "South Asia", lat: 20, lng: 77},
  {name: "Russia", code: "RU", numId: "643", region: "Eurasia", lat: 60, lng: 100},
  {name: "Brazil", code: "BR", numId: "076", region: "South America", lat: -10, lng: -55},
  {name: "Australia", code: "AU", numId: "036", region: "Oceania", lat: -25, lng: 135},
  {name: "Canada", code: "CA", numId: "124", region: "Americas", lat: 60, lng: -95},
  {name: "United Kingdom", code: "GB", numId: "826", region: "Europe", lat: 55, lng: -3},
  {name: "Germany", code: "DE", numId: "276", region: "Europe", lat: 51, lng: 9},
  {name: "France", code: "FR", numId: "250", region: "Europe", lat: 46, lng: 2},
  {name: "Japan", code: "JP", numId: "392", region: "East Asia", lat: 36, lng: 138},
  {name: "South Africa", code: "ZA", numId: "710", region: "Africa", lat: -30, lng: 22},
  {name: "Egypt", code: "EG", numId: "818", region: "North Africa", lat: 26, lng: 30},
  {name: "Saudi Arabia", code: "SA", numId: "682", region: "Middle East", lat: 23, lng: 45},
  {name: "Mexico", code: "MX", numId: "484", region: "Americas", lat: 23, lng: -102},
  {name: "Indonesia", code: "ID", numId: "360", region: "Southeast Asia", lat: -0.5, lng: 118},
  {name: "Nigeria", code: "NG", numId: "566", region: "Africa", lat: 10, lng: 8},
  {name: "Iran", code: "IR", numId: "364", region: "Middle East", lat: 32, lng: 53},
  {name: "Turkey", code: "TR", numId: "792", region: "Eurasia", lat: 39, lng: 35},
  {name: "Argentina", code: "AR", numId: "032", region: "South America", lat: -38, lng: -63},
  {name: "South Korea", code: "KR", numId: "410", region: "East Asia", lat: 36, lng: 128},
  {name: "Italy", code: "IT", numId: "380", region: "Europe", lat: 42, lng: 12},
  {name: "Spain", code: "ES", numId: "724", region: "Europe", lat: 40, lng: -4},
  {name: "Thailand", code: "TH", numId: "764", region: "Southeast Asia", lat: 15, lng: 101},
  {name: "Kenya", code: "KE", numId: "404", region: "Africa", lat: -1, lng: 37},
  {name: "Sweden", code: "SE", numId: "752", region: "Northern Europe", lat: 62, lng: 15},
  {name: "Poland", code: "PL", numId: "616", region: "Europe", lat: 52, lng: 20},
  {name: "Colombia", code: "CO", numId: "170", region: "South America", lat: 4, lng: -72},
  {name: "Pakistan", code: "PK", numId: "586", region: "South Asia", lat: 30, lng: 69},
  {name: "Ukraine", code: "UA", numId: "804", region: "Europe", lat: 49, lng: 32},
  {name: "Taiwan", code: "TW", numId: "158", region: "East Asia", lat: 25, lng: 121},
  {name: "Iraq", code: "IQ", numId: "368", region: "Middle East", lat: 33, lng: 44},
  {name: "Cuba", code: "CU", numId: "192", region: "Americas", lat: 22, lng: -80}
];

// Build lookups
const NUM_TO_COUNTRY = {};
const NAME_TO_NUMID = {};
COUNTRY_DATA.forEach(c => {
  NUM_TO_COUNTRY[c.numId] = c;
  NAME_TO_NUMID[c.name] = c.numId;
});
NAME_TO_NUMID['US'] = '840';
NAME_TO_NUMID['USA'] = '840';
NAME_TO_NUMID['United States of America'] = '840';

/* ═══════════════════════════════════════════════════════
   SimAPI — Backend data layer (Flask sim_api.py)
   ═══════════════════════════════════════════════════════ */
const API_BASE = 'http://localhost:5001';

const SimAPI = {
  async predict(actor, receiver, context) {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor, receiver, context })
    });
    if (!res.ok) throw new Error('API error ' + res.status);
    return res.json();
  },

  async predictRegion(actor, receiver, context) {
    const res = await fetch(`${API_BASE}/predict/region`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor, receiver, context })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || 'API error ' + res.status);
    }
    return res.json();
  },

  async elaborate(actor, receiver, context, y_plus, y_minus, region) {
    const res = await fetch(`${API_BASE}/elaborate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ actor, receiver, context, y_plus, y_minus, region })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || 'API error ' + res.status);
    }
    return res.json();
  }
};

/* ═══════════════════════════════════════════════════════
   APP CONTROLLER
   ═══════════════════════════════════════════════════════ */
const MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
const MONTH_FULL = ['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER'];

const state = {
  simId: null, theory: '', startYear: 100, endYear: 2025,
  currentYear: 100, currentMonth: 0,
  playing: false, speed: 1, interval: null,
  allEvents: [], globe: null,
  viewMode: '3d', // '3d' or '2d'
  impactedCountries: new Set()
};

const $ = id => document.getElementById(id);

/* ── CHIPS ── */
document.querySelectorAll('.chip').forEach(c => {
  c.addEventListener('click', () => {
    const promptEl = $('promptInput') || $('theoryInput');
    if (promptEl) promptEl.value = c.dataset.prompt || c.dataset.theory || '';
  });
});

function parsePrompt(prompt) {
  const parts = prompt.split(',').map(s => s.trim()).filter(Boolean);
  return {
    actor: parts[0] || 'China',
    receiver: parts[1] || 'Taiwan',
    context: parts[2] || 'Asia 2025'
  };
}

/* ── RUN SIMULATION ── */
$('runBtn').addEventListener('click', async () => {
  const promptEl = $('promptInput') || $('theoryInput');
  const prompt = (promptEl?.value || '').trim();
  if (!prompt) { promptEl?.focus(); return; }

  const { actor, receiver, context } = parsePrompt(prompt);
  $('runBtn').classList.add('loading');
  $('runBtn').disabled = true;

  let result;
  try {
    result = await SimAPI.predictRegion(actor, receiver, context);
  } catch (e) {
    alert('Backend error: ' + (e.message || 'Is sim_api.py running on port 5001?'));
    $('runBtn').classList.remove('loading');
    $('runBtn').disabled = false;
    return;
  }

  state.predictionResult = result;
  state.theory = `${actor} vs ${receiver}: ${context}`;
  state.impactByNumId = {};
  state.impactedCountries = new Set();
  const addImpact = (name, yPlus, isPrimary) => {
    const numId = NAME_TO_NUMID[name];
    if (numId) {
      state.impactByNumId[numId] = { y_plus: yPlus, primary: !!isPrimary };
      const c = NUM_TO_COUNTRY[numId];
      if (c) state.impactedCountries.add(c.code);
    }
  };
  addImpact(receiver, result.primary.y_plus, true);
  (result.region || []).forEach(r => addImpact(r.receiver, r.y_plus, false));
  const actorNumId = NAME_TO_NUMID[actor];
  if (actorNumId) {
    state.impactByNumId[actorNumId] = { y_plus: 0, primary: false, actor: true };
    const c = NUM_TO_COUNTRY[actorNumId];
    if (c) state.impactedCountries.add(c.code);
  }
  state.currentEvent = {
    actor, receiver, context,
    y_plus: result.primary.y_plus, y_minus: 1 - result.primary.y_plus,
    primary: result.primary, region: result.region || [],
    title: `${actor} → ${receiver}`,
  };

  $('theoryLabel').textContent = state.theory;
  $('intro').classList.add('hidden');
  if (!$('sim').classList.contains('active')) {
    $('sim').classList.add('active');
    initGlobe();
    init2DMap();
  }
  $('simDate').textContent = `${Math.round(result.primary.y_plus * 100)}% probability`;

  if (state.globe && result.actor_coords) {
    const allResults = [result.primary, ...(result.region || [])];
    state.globe.addPredictionLines(result.actor_coords, allResults);
  }
  if (state.viewMode === '2d') draw2DMap();

  $('intelPlaceholder').style.display = 'none';
  $('intelReport').classList.add('active');
  $('intelBadge').textContent = 'Prediction';
  $('intelTitle').textContent = `${actor} → ${receiver}`;
  $('intelDesc').textContent = `Primary: ${(result.primary.y_plus * 100).toFixed(1)}% probability. Context: ${context}`;
  $('intelActors').innerHTML = `<span class="intel-actor-chip">${actor}</span><span class="intel-actor-chip">${receiver}</span>`;
  $('intelProb').textContent = Math.round(result.primary.y_plus * 100) + '%';
  $('intelAnalogue').textContent = 'Aggregate from 100 perspectives';
  const effects = (result.region || []).map(r =>
    `<div class="intel-effect-item"><div class="intel-effect-bar orange"></div><span>${r.receiver}: ${(r.y_plus * 100).toFixed(1)}% (downstream)</span></div>`
  ).join('');
  $('intelEffects').innerHTML = effects || '<div class="intel-effect-item"><span>No other countries in region</span></div>';

  $('runBtn').classList.remove('loading');
  $('runBtn').disabled = false;
});

const promptInput = $('promptInput') || $('theoryInput');
if (promptInput) promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); $('runBtn').click(); }
});

/* ── INIT GLOBE ── */
function initGlobe() {
  if (state.globe) return;
  state.globe = new GlobeEngine($('globeCanvas'));
  state.globe.onMarkerClick = (evt) => showEventOverlay(evt);
  state.globe.loadCountryDataset(COUNTRY_DATA);
}

/* ══════════════════════════════════════════
   2D FLAT MAP (Canvas + real country shapes)
   ══════════════════════════════════════════ */
let flatCtx = null;
let flatW = 0, flatH = 0;
let worldGeoFeatures = null;
let map2dClickBound = false;

async function init2DMap() {
  // Only fetch the world data — canvas sizing happens in switchView
  // because the panel may be display:none at this point
  try {
    const resp = await fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json');
    const world = await resp.json();
    const countries = topojson.feature(world, world.objects.countries);
    worldGeoFeatures = countries.features;
  } catch (e) {
    console.warn('Failed to load world map data:', e);
    worldGeoFeatures = [];
  }
}

function resizeFlatCanvas() {
  const canvas = $('flatMapCanvas');
  const parent = canvas.parentElement;
  const pRect = parent.getBoundingClientRect();
  if (pRect.width === 0 || pRect.height === 0) return;
  const dpr = window.devicePixelRatio || 1;
  // Use proper 2:1 equirectangular aspect ratio
  const maxW = pRect.width;
  const maxH = pRect.height;
  let w = maxW;
  let h = w / 2;
  if (h > maxH) { h = maxH; w = h * 2; }
  flatW = w * dpr;
  flatH = h * dpr;
  canvas.width = flatW;
  canvas.height = flatH;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  flatCtx = canvas.getContext('2d');

  // Bind click handler once
  if (!map2dClickBound) {
    map2dClickBound = true;
    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      const mx = (e.clientX - r.left) / r.width;
      const my = (e.clientY - r.top) / r.height;

      for (const c of COUNTRY_DATA) {
        const fx = (c.lng + 180) / 360;
        const fy = (90 - c.lat) / 180;
        const dx = mx - fx;
        const dy = my - fy;
        if (Math.sqrt(dx*dx + dy*dy) < 0.03) {
          showEventOverlay({ type: 'country', ...c });
          break;
        }
      }
    });
  }
}

function projX(lng) { return ((lng + 180) / 360) * flatW; }
function projY(lat) { return ((90 - lat) / 180) * flatH; }

function drawGeoFeature(ctx, geometry, fillColor, strokeColor, lw) {
  const polys = geometry.type === 'Polygon'
    ? [geometry.coordinates]
    : geometry.coordinates; // MultiPolygon

  ctx.beginPath();
  polys.forEach(rings => {
    rings.forEach(ring => {
      ring.forEach(([lng, lat], i) => {
        const x = projX(lng), y = projY(lat);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
    });
  });
  ctx.fillStyle = fillColor;
  ctx.fill('evenodd');
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = lw;
  ctx.stroke();
}

function draw2DMap() {
  if (!flatCtx || !worldGeoFeatures || flatW === 0) return;
  const ctx = flatCtx;
  const dpr = window.devicePixelRatio || 1;

  ctx.clearRect(0, 0, flatW, flatH);

  // Ocean gradient background
  const oceanGrad = ctx.createLinearGradient(0, 0, 0, flatH);
  oceanGrad.addColorStop(0, '#0c1220');
  oceanGrad.addColorStop(0.5, '#0a1018');
  oceanGrad.addColorStop(1, '#080c14');
  ctx.fillStyle = oceanGrad;
  ctx.fillRect(0, 0, flatW, flatH);

  // Subtle graticule grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5 * dpr;
  for (let lat = -60; lat <= 80; lat += 30) {
    ctx.beginPath();
    ctx.moveTo(projX(-180), projY(lat));
    ctx.lineTo(projX(180), projY(lat));
    ctx.stroke();
  }
  for (let lng = -180; lng <= 180; lng += 30) {
    ctx.beginPath();
    ctx.moveTo(projX(lng), projY(-80));
    ctx.lineTo(projX(lng), projY(80));
    ctx.stroke();
  }

  const impactByNumId = state.impactByNumId || {};

  // Draw all country shapes — darker fill by probability
  ctx.lineJoin = 'round';
  worldGeoFeatures.forEach(feature => {
    const fid = String(feature.id);
    const impact = impactByNumId[fid];

    if (impact) {
      if (impact.actor) {
        ctx.shadowColor = 'rgba(59,130,246,0.6)';
        ctx.shadowBlur = 10 * dpr;
        drawGeoFeature(ctx, feature.geometry,
          'rgba(59,130,246,0.12)', 'rgba(96,165,250,0.7)', 0.6 * dpr);
        ctx.shadowBlur = 0;
      } else {
        const p = impact.y_plus;
        const intensity = Math.min(1, p * 1.2);
        const isPrimary = impact.primary;
        const r = isPrimary ? Math.floor(239 + (1 - intensity) * 16) : 233;
        const g = isPrimary ? Math.floor(68 + (1 - intensity) * 82) : Math.floor(115 + (1 - intensity) * 35);
        const b = 22;
        const fillAlpha = 0.08 + intensity * 0.35;
        const strokeAlpha = 0.5 + intensity * 0.5;
        ctx.shadowColor = `rgba(233,115,22,${0.3 + intensity * 0.5})`;
        ctx.shadowBlur = (isPrimary ? 16 : 8) * dpr;
        drawGeoFeature(ctx, feature.geometry,
          `rgba(${r},${g},${b},${fillAlpha})`,
          `rgba(255,${Math.floor(150 + intensity * 50)},50,${strokeAlpha})`,
          (isPrimary ? 0.9 : 0.5) * dpr);
        ctx.shadowBlur = 0;
      }
    } else {
      drawGeoFeature(ctx, feature.geometry,
        'rgba(255,255,255,0.02)', 'rgba(255,255,255,0.12)', 0.3 * dpr);
    }
  });

  // Connection arcs from actor to each impacted receiver
  const pred = state.predictionResult;
  if (pred?.actor_coords) {
    const actor = pred.actor_coords;
    const receivers = [pred.primary, ...(pred.region || [])];
    receivers.forEach(r => {
      const intensity = Math.min(1, r.y_plus * 1.2);
      const color = intensity > 0.65 ? '239,68,68' : intensity > 0.5 ? '245,158,11' : '34,197,94';
      ctx.strokeStyle = `rgba(${color},${0.4 + intensity * 0.4})`;
      ctx.lineWidth = (r === pred.primary ? 2.5 : 1.5) * dpr;
      ctx.setLineDash(r === pred.primary ? [] : [6 * dpr, 4 * dpr]);
      const ax = projX(actor.lng), ay = projY(actor.lat);
      const bx = projX(r.lng), by = projY(r.lat);
      const cx = (ax + bx) / 2, cy = Math.min(ay, by) - 20 * dpr;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.quadraticCurveTo(cx, cy, bx, by);
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }

  // Impact dots on receiver countries
  if (pred?.actor_coords) {
    [pred.primary, ...(pred.region || [])].forEach(r => {
      const ex = projX(r.lng), ey = projY(r.lat);
      const intensity = Math.min(1, r.y_plus * 1.2);
      const color = intensity > 0.65 ? '#EF4444' : intensity > 0.5 ? '#F59E0B' : '#22C55E';
      const grad = ctx.createRadialGradient(ex, ey, 0, ex, ey, 18 * dpr);
      grad.addColorStop(0, color + '50');
      grad.addColorStop(0.6, color + '15');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(ex, ey, 18 * dpr, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(ex, ey, 4 * dpr, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = color + '60';
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath(); ctx.arc(ex, ey, 7 * dpr, 0, Math.PI * 2); ctx.stroke();
    });
  }

  // Country labels
  COUNTRY_DATA.forEach(c => {
    const px = projX(c.lng), py = projY(c.lat);
    const isImpacted = state.impactedCountries?.has(c.code);

    // Small dot at centroid
    ctx.fillStyle = isImpacted ? '#E97316' : 'rgba(255,255,255,0.25)';
    ctx.beginPath();
    ctx.arc(px, py, (isImpacted ? 3.5 : 2) * dpr, 0, Math.PI * 2);
    ctx.fill();

    // Label text
    ctx.fillStyle = isImpacted ? '#fff' : 'rgba(255,255,255,0.5)';
    ctx.font = `${(isImpacted ? 9.5 : 8) * dpr}px IBM Plex Sans, sans-serif`;
    ctx.textAlign = 'left';
    ctx.fillText(c.name.toUpperCase(), px + 6 * dpr, py + 3 * dpr);
  });
}

/* ── VIEW TOGGLE ── */
$('view3D').addEventListener('click', () => switchView('3d'));
$('view2D').addEventListener('click', () => switchView('2d'));

function switchView(mode) {
  state.viewMode = mode;
  $('view3D').classList.toggle('active', mode === '3d');
  $('view2D').classList.toggle('active', mode === '2d');
  $('map2dPanel').classList.toggle('active', mode === '2d');
  if (mode === '2d') {
    // Resize canvas now that panel is visible
    resizeFlatCanvas();
    draw2DMap();
  }
}

/* ── PLAY / PAUSE ── */
$('playBtn').addEventListener('click', () => {
  state.playing = !state.playing;
  $('playBtn').textContent = state.playing ? '❚❚' : '▶';
  $('monthGrid').classList.remove('active');
  if (state.playing) startPlayback(); else stopPlayback();
});

function startPlayback() {
  stopPlayback();
  state.interval = setInterval(() => advanceMonth(), 1200 / state.speed);
}
function stopPlayback() { if (state.interval) { clearInterval(state.interval); state.interval = null; } }

async function advanceMonth() {
  state.currentMonth++;
  if (state.currentMonth >= 12) { state.currentMonth = 0; state.currentYear++; }
  if (state.currentYear > state.endYear) { state.playing = false; $('playBtn').textContent = '▶'; stopPlayback(); return; }
  $('scrubber').value = state.currentYear;
  updateDateDisplay();
  await loadCurrentData();
}

function updateDateDisplay() {
  $('simDate').textContent = `${MONTH_FULL[state.currentMonth]} ${String(state.currentYear).padStart(4, '0')}`;
}

/* ── LOAD DATA ── */
async function loadCurrentData() {
  const events = await SimAPI.getEvents(state.simId, state.currentYear, state.currentMonth);
  events.forEach(e => {
    state.allEvents.push(e);
    addEventChip(e);
    if (state.globe) {
      state.globe.addMarker(e);
      // Smart rotate globe toward latest event (only on start or if explicitly requested)
      if (!state.hasAutoRotated) {
        rotateGlobeToEvent(e);
        setTimeout(() => state.hasAutoRotated = true, 3000); // Give it 3s to pan, then give control to user
      }
    }
    
    // Track impacted countries by proximity
    COUNTRY_DATA.forEach(c => {
      const dist = Math.sqrt(Math.pow(c.lat - e.lat, 2) + Math.pow(c.lng - e.lng, 2));
      if (dist < 15) state.impactedCountries.add(c.code);
    });
  });
  if (state.viewMode === '2d') draw2DMap();
}

function rotateGlobeToEvent(e) {
  if (!state.globe || !state.globe.controls) return;
  const phi = (90 - e.lat) * Math.PI / 180;
  const theta = (e.lng + 180) * Math.PI / 180;

  // Keep camera at comfortable viewing distance (e.g., radius of 180)
  const currentDist = state.globe.camera.position.length();
  const targetDist = Math.max(currentDist, 180); 

  const targetX = targetDist * Math.sin(phi) * Math.cos(theta);
  const targetY = targetDist * Math.cos(phi);
  const targetZ = targetDist * Math.sin(phi) * Math.sin(theta);
  
  // Smooth lerp toward target
  const cam = state.globe.camera.position;
  const lerpFactor = 0.08;
  cam.x += (targetX - cam.x) * lerpFactor;
  cam.y += (targetY - cam.y) * lerpFactor;
  cam.z += (targetZ - cam.z) * lerpFactor;
  state.globe.camera.lookAt(0, 0, 0);
}

/* ── EVENT CHIPS ON MAP ── */
function addEventChip(e) {
  const chip = document.createElement('div');
  chip.className = 'map-event-chip';
  chip.innerHTML = `<div class="chip-title">${e.title}</div><div class="chip-sub">Click for intel report</div>`;
  chip.addEventListener('click', () => showIntelReport(e));
  const container = $('mapEventChips');
  container.prepend(chip);
  while (container.children.length > 5) container.removeChild(container.lastChild);
  addNotification(e);
}

/* ── HISTORICAL ANALOGUES (lookup table) ── */
const ANALOGUES = {
  'Political Realignment': { analogue: 'Fall of the Berlin Wall (1989)', prob: 72 },
  'Trade Embargo': { analogue: 'OPEC Oil Embargo (1973-1974)', prob: 68 },
  'Technological Breakthrough': { analogue: 'Space Race Acceleration (1957-1969)', prob: 81 },
  'Energy Grid Failure': { analogue: 'Northeast Blackout (2003)', prob: 55 },
  'Economic Expansion': { analogue: 'Post-WWII Marshall Plan (1948)', prob: 77 },
  'Cultural Movement': { analogue: 'Arab Spring (2010-2012)', prob: 63 },
  'Agricultural Expansion': { analogue: 'Green Revolution (1960s-1970s)', prob: 79 },
  'Military Escalation': { analogue: 'Third Taiwan Strait Crisis (1995-1996)', prob: 85 },
  'Maritime Trade Boom': { analogue: 'Suez Canal Opening (1869)', prob: 74 },
  'Resource Discovery': { analogue: 'North Sea Oil Discovery (1969)', prob: 70 },
  'Cyber Infrastructure Disruption': { analogue: 'Stuxnet Cyberattack (2010)', prob: 66 }
};

const DOWNSTREAM_FX = {
  critical: [
    { text: 'Humanitarian crisis requires international intervention', bar: 'red' },
    { text: 'Regional supply chain collapse imminent', bar: 'orange' },
    { text: 'Capital flight destabilizes local currency', bar: 'orange' },
  ],
  high: [
    { text: 'Trade route disruption increases shipping costs', bar: 'orange' },
    { text: 'Insurance premiums surge across the sector', bar: 'blue' },
    { text: 'Regional capital flight detected', bar: 'orange' },
  ],
  medium: [
    { text: 'Market volatility increases across region', bar: 'blue' },
    { text: 'Diplomatic channels under sustained pressure', bar: 'orange' },
    { text: 'Migration patterns begin shifting', bar: 'blue' },
  ],
  low: [
    { text: 'Local economic indicators show marginal improvement', bar: 'blue' },
    { text: 'International attention draws future investment', bar: 'blue' },
  ]
};

/* ── SHOW INTELLIGENCE REPORT ── */
function showIntelReport(e) {
  $('intelPlaceholder').style.display = 'none';
  $('intelReport').classList.add('active');

  if (e.type === 'country') {
    state.currentEvent = null;
    $('intelBadge').textContent = 'Sovereign Profile';
    $('intelTitle').textContent = e.name;
    $('intelDesc').textContent = `Geopolitical intelligence and resource metrics for ${e.name}. Analysis indicates stable macro-economic trends with isolated volatility in ${e.region}.`;
    $('intelActors').innerHTML = `<span class="intel-actor-chip">${e.name}</span>`;
    $('intelProb').textContent = '—';
    $('intelAnalogue').textContent = 'No active scenario';
    $('intelEffects').innerHTML = [
      { text: 'Industrial output at optimal capacity', bar: 'blue' },
      { text: 'Trade networks maintaining 98% efficiency', bar: 'blue' },
      { text: 'No major internal disruptions detected', bar: 'blue' },
    ].map(fx => `<div class="intel-effect-item"><div class="intel-effect-bar ${fx.bar}"></div><span>${fx.text}</span></div>`).join('');
  } else {
    state.currentEvent = e;
    $('intelBadge').textContent = 'Intelligence Report';
    $('intelTitle').textContent = e.title;
    $('intelDesc').textContent = e.reasoning || e.description;

    // Primary Actors: use actor/receiver from backend when available, else nearby countries
    const actors = (e.actor && e.receiver)
      ? [e.actor, e.receiver]
      : COUNTRY_DATA.filter(c => {
          const dist = Math.sqrt(Math.pow(c.lat - e.lat, 2) + Math.pow(c.lng - e.lng, 2));
          return dist < 20;
        }).slice(0, 4).map(c => c.name);
    $('intelActors').innerHTML = actors.map(a => `<span class="intel-actor-chip">${a}</span>`).join('') || '<span class="intel-actor-chip">Unknown</span>';

    // Probability + Analogue (use y_plus from backend when available)
    const probPct = e.y_plus != null ? Math.round(e.y_plus * 100) : (ANALOGUES[e.title]?.prob ?? Math.floor(40 + Math.random() * 50));
    $('intelProb').textContent = probPct + '%';
    $('intelAnalogue').textContent = e.y_plus != null ? `Aggregate probability from 100 perspectives` : (ANALOGUES[e.title]?.analogue ?? 'No prior analogue');

    // Downstream Effects
    const effects = DOWNSTREAM_FX[e.severity] || DOWNSTREAM_FX.medium;
    $('intelEffects').innerHTML = effects.map(fx =>
      `<div class="intel-effect-item"><div class="intel-effect-bar ${fx.bar}"></div><span>${fx.text}</span></div>`
    ).join('');
  }
}

// Globe marker clicks open intel report instead of overlay
function showEventOverlay(e) { showIntelReport(e); }

/* ── CLOSE TACTICAL OVERLAY IF STILL PRESENT ── */
const tacClose = $('tacCloseBtn');
if (tacClose) tacClose.addEventListener('click', () => $('tacticalOverlay').classList.remove('active'));
const tacOverlay = $('tacticalOverlay');
if (tacOverlay) tacOverlay.addEventListener('click', (ev) => { if (ev.target === tacOverlay) tacOverlay.classList.remove('active'); });

/* ── ELABORATE FINDINGS (Featherless AI) ── */
const btnElaborate = $('intelElaborate');
if (btnElaborate) {
  btnElaborate.addEventListener('click', async () => {
    const evt = state.currentEvent;
    if (!evt) return;

    const tacTitle = document.querySelector('.tactical-header h2');
    const tacMeta = document.querySelector('.tac-meta');
    const tacTitleMain = document.querySelector('.tac-title');
    const tacDesc = document.querySelector('.tac-desc');

    if (tacTitle) tacTitle.textContent = 'Strategic Assessment';
    if (tacMeta) tacMeta.textContent = 'Extrapolating with Featherless AI...';
    if (tacTitleMain) tacTitleMain.textContent = evt.title;
    if (tacDesc) tacDesc.textContent = 'Loading AI elaboration...';
    $('tacticalOverlay').classList.add('active');

    btnElaborate.disabled = true;
    try {
      const res = await SimAPI.elaborate(
        evt.actor,
        evt.receiver,
        evt.context || '',
        evt.y_plus ?? evt.primary?.y_plus ?? 0.5,
        evt.y_minus ?? (1 - (evt.primary?.y_plus ?? 0.5)),
        evt.region || []
      );
      if (tacDesc) tacDesc.textContent = res.elaboration || 'No elaboration returned.';
      if (tacMeta) tacMeta.textContent = `REF: AP-${Math.floor(Math.random()*9000)+1000}`;
    } catch (e) {
      if (tacDesc) tacDesc.textContent = `Error: ${e.message}. Ensure FEATHERLESS_API_KEY is set and sim_api.py is running.`;
      if (tacMeta) tacMeta.textContent = 'Elaboration failed';
    }
    btnElaborate.disabled = false;
  });
}

function formatBig(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

/* ── YEAR CHIPS ── */
function buildYearChips() {
  const el = $('yearChips'); el.innerHTML = '';
  const step = Math.ceil((state.endYear - state.startYear) / 12);
  for (let y = state.startYear; y <= state.endYear; y += step) {
    const btn = document.createElement('button');
    btn.className = 'year-chip'; btn.textContent = y;
    btn.addEventListener('click', () => jumpToYear(y));
    el.appendChild(btn);
  }
}

/* ── MONTH GRID ── */
function buildMonthGrid() {
  const el = $('monthGrid'); el.innerHTML = '';
  MONTHS.forEach((m, i) => {
    const btn = document.createElement('button');
    btn.className = 'month-btn'; btn.textContent = m;
    btn.addEventListener('click', () => { state.currentMonth = i; updateDateDisplay(); loadCurrentData(); el.classList.remove('active'); });
    el.appendChild(btn);
  });
}

$('scrubber').addEventListener('click', () => { if (!state.playing) $('monthGrid').classList.toggle('active'); });
$('scrubber').addEventListener('input', () => jumpToYear(parseInt($('scrubber').value)));

async function jumpToYear(y) {
  state.currentYear = y;
  $('scrubber').value = y;
  updateDateDisplay();
  if (state.globe) state.globe.clearMarkers();
  await loadCurrentData();
  document.querySelectorAll('.year-chip').forEach(c => c.classList.toggle('active', parseInt(c.textContent) === y));
}

/* ── SPEED ── */
document.querySelectorAll('.speed').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.speed').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.speed = parseFloat(btn.dataset.speed);
    if (state.playing) startPlayback();
  });
});

/* ── NEW THEORY ── */
$('newTheoryBtn')?.addEventListener('click', () => {
  if (state.interval) clearInterval(state.interval);
  state.interval = null;
  state.playing = false;
  $('playBtn').textContent = '▶';
  $('sim').classList.remove('active');
  if (state.globe) state.globe.clearPredictionLines();
  $('mapEventChips').innerHTML = '';
  $('notifTickerInner').innerHTML = '';
  $('intelReport').classList.remove('active');
  $('intelPlaceholder').style.display = '';
  if (typeof switchView === 'function') switchView('3d');
  setTimeout(() => $('intro').classList.remove('hidden'), 100);
});

/* ── NOTIFICATION TICKER ── */
function addNotification(e) {
  const container = $('notifTickerInner');
  const item = document.createElement('div');
  item.className = 'notif-item';
  item.innerHTML = `<span class="notif-dot ${e.severity}"></span><span>[${e.region}] ${e.title}</span>`;
  container.prepend(item);
  // Keep only 1 visible
  while (container.children.length > 5) container.removeChild(container.lastChild);
}

/* ── NAV ── */
function setActiveNav(id) {
  document.querySelectorAll('.nav-link').forEach(n => n.classList.remove('active'));
  $(id).classList.add('active');
}

$('navSim').addEventListener('click', () => {
  setActiveNav('navSim');
});

