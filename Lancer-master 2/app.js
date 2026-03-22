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

// Build a quick lookup: numericId -> COUNTRY_DATA entry
const NUM_TO_COUNTRY = {};
COUNTRY_DATA.forEach(c => { NUM_TO_COUNTRY[c.numId] = c; });

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

  $('intelPlaceholder').style.display = 'none';
  $('intelReport').classList.add('active');
  $('intelBadge').textContent = 'Prediction';
  $('intelTitle').textContent = `${actor} → ${receiver}`;
  $('intelDesc').textContent = `Primary: ${(result.primary.y_plus * 100).toFixed(1)}% probability. Context: ${context}`;
  $('intelActors').innerHTML = `<span class="intel-actor-chip">${actor}</span><span class="intel-actor-chip">${receiver}</span>`;
  $('intelProb').textContent = Math.round(result.primary.y_plus * 100) + '%';
  $('intelAnalogue').textContent = 'Aggregate from 100 perspectives';
  const effects = (result.region || []).map(r =>
    `<div class="intel-effect-item"><div class="intel-effect-bar orange"></div><span>${r.receiver}: ${(r.y_plus * 100).toFixed(1)}%</span></div>`
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

  // Build set of impacted numeric IDs
  const impactedNums = new Set();
  state.impactedCountries.forEach(code => {
    const c = COUNTRY_DATA.find(d => d.code === code);
    if (c) impactedNums.add(c.numId);
  });

  // Draw all country shapes with realistic tech styling
  ctx.lineJoin = 'round';
  worldGeoFeatures.forEach(feature => {
    const fid = String(feature.id);
    const isImpacted = impactedNums.has(fid);

    if (isImpacted) {
      // Glow and intense borders for impacted areas
      ctx.shadowColor = 'rgba(233,115,22,0.8)';
      ctx.shadowBlur = 12 * dpr;
      drawGeoFeature(ctx, feature.geometry,
        'rgba(233,115,22,0.15)', 'rgba(255,150,50,0.9)', 0.6 * dpr);
      ctx.shadowBlur = 0; // reset
    } else {
      // Very sharp, subtle blueprint/satellite tech lines for un-impacted
      drawGeoFeature(ctx, feature.geometry,
        'rgba(255,255,255,0.02)', 'rgba(255,255,255,0.15)', 0.3 * dpr);
    }
  });

  // Connection arcs between impacted countries
  const impactedList = COUNTRY_DATA.filter(c => state.impactedCountries.has(c.code));
  if (impactedList.length >= 2) {
    ctx.strokeStyle = 'rgba(233,115,22,0.5)';
    ctx.lineWidth = 2.5 * dpr;
    ctx.setLineDash([8 * dpr, 5 * dpr]);
    for (let i = 0; i < impactedList.length; i++) {
      for (let j = i + 1; j < impactedList.length; j++) {
        const ax = projX(impactedList[i].lng), ay = projY(impactedList[i].lat);
        const bx = projX(impactedList[j].lng), by = projY(impactedList[j].lat);
        const cx = (ax + bx) / 2;
        const cy = Math.min(ay, by) - 25 * dpr;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.quadraticCurveTo(cx, cy, bx, by);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }

  // Event dots
  state.allEvents.forEach(evt => {
    const ex = projX(evt.lng), ey = projY(evt.lat);
    const colors = { critical: '#EF4444', high: '#F59E0B', medium: '#3B82F6', low: '#22C55E' };
    const color = colors[evt.severity] || '#3B82F6';

    const grad = ctx.createRadialGradient(ex, ey, 0, ex, ey, 14 * dpr);
    grad.addColorStop(0, color + '60');
    grad.addColorStop(0.6, color + '20');
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(ex, ey, 14 * dpr, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(ex, ey, 3 * dpr, 0, Math.PI * 2); ctx.fill();

    // Outer ring
    ctx.strokeStyle = color + '40';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath(); ctx.arc(ex, ey, 6 * dpr, 0, Math.PI * 2); ctx.stroke();
  });

  // Country labels
  COUNTRY_DATA.forEach(c => {
    const px = projX(c.lng), py = projY(c.lat);
    const isImpacted = state.impactedCountries.has(c.code);

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

/* ── ELABORATE FINDINGS ── */
const btnElaborate = $('intelElaborate');
if (btnElaborate) {
  btnElaborate.addEventListener('click', () => {
    if (state.currentEvent) {
      // Find elements in tactical overlay
      const tacTitle = document.querySelector('.tactical-header h2');
      const tacMeta = document.querySelector('.tac-meta');
      const tacTitleMain = document.querySelector('.tac-title');
      const tacDesc = document.querySelector('.tac-desc');
      
      if (tacTitle) tacTitle.textContent = 'Strategic Assessment';
      if (tacMeta) tacMeta.textContent = `REF: AP-${Math.floor(Math.random()*9000)+1000}`;
      if (tacTitleMain) tacTitleMain.textContent = state.currentEvent.title;
      if (tacDesc) tacDesc.textContent = state.currentEvent.reasoning || state.currentEvent.description || 'Comprehensive analysis indicates escalating multi-domain consequences spanning regional supply chains, diplomatic networks, and emergent technological paradigms.';
      
      $('tacticalOverlay').classList.add('active');
    }
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
  $('analytics').classList.remove('active');
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
  $('analytics').classList.remove('active');
});

$('navAnalytics').addEventListener('click', () => {
  setActiveNav('navAnalytics');
  $('analytics').classList.add('active');
  renderAnalytics();
});

$('analyticsBackBtn')?.addEventListener('click', () => {
  setActiveNav('navSim');
  $('analytics').classList.remove('active');
});

/* ═══════════════════════════════════════════
   ANALYTICS — Simple interactive canvas charts
   ═══════════════════════════════════════════ */
function renderAnalytics() {
  updateTextAnalysis();
  renderTimelineChart();
  renderSeverityChart();
  renderRegionChart();
}

function updateTextAnalysis() {
  const el = $('aiSummaryText');
  if (!el) return;

  if (state.allEvents.length === 0) {
    el.innerHTML = 'Initializing causal projection synthesis... Please run a simulation to aggregate data.';
    return;
  }

  const critical = state.allEvents.filter(e => e.severity === 'critical').length;
  const high = state.allEvents.filter(e => e.severity === 'high').length;
  
  const regions = {};
  state.allEvents.forEach(e => { regions[e.region] = (regions[e.region] || 0) + 1; });
  const topRegion = Object.keys(regions).sort((a,b) => regions[b] - regions[a])[0] || 'Unknown';

  let text = `<span style="color: var(--orange)">STRATEGIC OVERVIEW:</span> The simulation has registered <b>${state.allEvents.length}</b> significant events. `;
  
  if (critical > 0) {
    text += `Escalation vectors are severe, with <b style="color: var(--red)">${critical} critical inflection points</b> threatening immediate structural stability. `;
  } else if (high > 0) {
    text += `Tension is elevated, recording <b>${high} high-severity events</b> that require close monitoring. `;
  } else {
    text += `Current conditions remain largely stable, with minimal high-impact destabilizers detected. `;
  }

  text += `The primary epicenter of disruption is currently polarized around <b>${topRegion.toUpperCase()}</b>. `;
  text += `<br><br><span style="color: var(--text-dim); font-size: 0.75rem;">RECOMMENDATION: Immediate diplomatic intervention advised focusing on ${topRegion} trade corridors to mitigate downstream cascading failure.</span>`;

  el.innerHTML = text;
}

function setupCanvas(id) {
  const canvas = $(id);
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const w = parent.clientWidth;
  const h = Math.max(180, parent.clientHeight - 30);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

function renderTimelineChart() {
  const { ctx, w, h } = setupCanvas('chartTimeline');
  ctx.clearRect(0, 0, w, h);

  // Group events by year
  const yearCounts = {};
  state.allEvents.forEach(e => {
    const y = e.year;
    yearCounts[y] = (yearCounts[y] || 0) + 1;
  });

  const years = Object.keys(yearCounts).sort((a,b) => a - b);
  if (years.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '13px IBM Plex Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Run a simulation to see data', w/2, h/2);
    return;
  }

  const maxVal = Math.max(...Object.values(yearCounts), 1);
  const pad = { top: 20, right: 20, bottom: 30, left: 40 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;
  const barW = Math.min(30, chartW / years.length - 4);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.03)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * chartH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
  }
  ctx.setLineDash([]); // reset

  // Bars
  years.forEach((yr, i) => {
    const x = pad.left + (i / years.length) * chartW + (chartW / years.length - barW) / 2;
    const barH = (yearCounts[yr] / maxVal) * chartH;
    const y = pad.top + chartH - barH;

    // Track Background
    ctx.fillStyle = 'rgba(255,255,255,0.02)';
    ctx.beginPath();
    ctx.roundRect(x, pad.top, barW, chartH, 4);
    ctx.fill();

    // Bar Fill
    const grad = ctx.createLinearGradient(x, y, x, y + barH);
    grad.addColorStop(0, 'rgba(233,115,22,0.9)');
    grad.addColorStop(1, 'rgba(233,115,22,0.2)');
    ctx.fillStyle = grad;
    
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
    ctx.fill();

    // Year label
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '9px IBM Plex Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(yr, x + barW/2, h - pad.bottom + 16);

    // Value Badge
    if (yearCounts[yr] > 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.font = '600 9px IBM Plex Mono, monospace';
      ctx.fillText(yearCounts[yr], x + barW/2, y - 6);
    }
  });

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '9px IBM Plex Sans, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = Math.round(maxVal * (1 - i/4));
    ctx.fillText(val, pad.left - 6, pad.top + (i/4) * chartH + 3);
  }

  // Hover tooltip
  const canvas = $('chartTimeline');
  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    years.forEach((yr, i) => {
      const x = pad.left + (i / years.length) * chartW + (chartW / years.length - barW) / 2;
      if (mx >= x && mx <= x + barW) {
        canvas.title = `Year ${yr}: ${yearCounts[yr]} events`;
      }
    });
  };
}

function renderSeverityChart() {
  const { ctx, w, h } = setupCanvas('chartSeverity');
  ctx.clearRect(0, 0, w, h);

  const counts = { low: 0, medium: 0, high: 0, critical: 0 };
  state.allEvents.forEach(e => { counts[e.severity] = (counts[e.severity] || 0) + 1; });
  const total = Object.values(counts).reduce((a, b) => a + b, 0);

  if (total === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '13px IBM Plex Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No data yet', w/2, h/2);
    return;
  }

  const colors = { low: '#22C55E', medium: '#3B82F6', high: '#F59E0B', critical: '#EF4444' };
  const labels = { low: 'Low', medium: 'Medium', high: 'High', critical: 'Critical' };
  const cx = w * 0.35, cy = h / 2, radius = Math.min(w * 0.28, h * 0.38);
  let angle = -Math.PI / 2;

  Object.entries(counts).forEach(([sev, count]) => {
    if (count === 0) return;
    const slice = (count / total) * Math.PI * 2;
    
    // Draw slice with gap
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = colors[sev];
    ctx.fill();
    
    // Gap stroke matching bg gradient
    ctx.strokeStyle = '#0e121a';
    ctx.lineWidth = 4;
    ctx.stroke();
    
    angle += slice;
  });

  // Inner punchout for donut
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.65, 0, Math.PI * 2);
  ctx.fillStyle = '#0e121a';
  ctx.fill();

  // Total in center
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 24px IBM Plex Sans, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(total, cx, cy + 6);
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '600 9px IBM Plex Sans, sans-serif';
  ctx.letterSpacing = '1px';
  ctx.fillText('EVENTS', cx, cy + 20);
  ctx.letterSpacing = '0px';

  // Elegant Legend
  let ly = (h / 2) - ((Object.values(counts).filter(c=>c>0).length * 28) / 2) + 10;
  Object.entries(counts).forEach(([sev, count]) => {
    if (count === 0) return;
    
    // Dot indicator
    ctx.beginPath();
    ctx.arc(w * 0.68, ly - 4, 4, 0, Math.PI * 2);
    ctx.fillStyle = colors[sev];
    ctx.fill();
    
    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = '11px IBM Plex Sans, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${labels[sev]}`, w * 0.68 + 14, ly);
    
    // Count Badge
    ctx.fillStyle = '#fff';
    ctx.font = '600 11px IBM Plex Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(count.toString().padStart(2, '0'), w - 24, ly);
    
    ly += 26;
  });
}

function renderRegionChart() {
  const { ctx, w, h } = setupCanvas('chartRegion');
  ctx.clearRect(0, 0, w, h);

  const regionCounts = {};
  state.allEvents.forEach(e => {
    regionCounts[e.region] = (regionCounts[e.region] || 0) + 1;
  });

  const sorted = Object.entries(regionCounts).sort((a,b) => b[1] - a[1]);
  if (sorted.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '13px IBM Plex Sans, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No data yet', w/2, h/2);
    return;
  }

  const maxVal = sorted[0][1] || 1;
  const pad = { top: 20, right: 30, bottom: 20, left: 110 };
  const chartW = w - pad.left - pad.right;
  const barH = Math.min(18, (h - pad.top - pad.bottom) / sorted.length - 8);

  sorted.forEach(([region, count], i) => {
    const y = pad.top + i * (barH + 12);
    const bw = (count / maxVal) * chartW;

    // Track Background (Subtle)
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.beginPath();
    ctx.roundRect(pad.left, y, chartW, barH, 4);
    ctx.fill();

    // Bar Fill
    const grad = ctx.createLinearGradient(pad.left, y, pad.left + bw, y);
    grad.addColorStop(0, 'rgba(59,130,246,0.8)');
    grad.addColorStop(1, 'rgba(59,130,246,0.3)');
    ctx.fillStyle = grad;
    
    // Draw with rounded right edge
    ctx.beginPath();
    if(bw > 0) ctx.roundRect(pad.left, y, bw, barH, 4);
    ctx.fill();

    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '11px IBM Plex Sans, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(region.toUpperCase(), pad.left - 12, y + barH / 2 + 4);

    // Value Badge
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath();
    ctx.roundRect(pad.left + bw + 10, y - 2, 28, barH + 4, 3);
    ctx.fill();
    
    ctx.fillStyle = '#fff';
    ctx.font = '600 11px IBM Plex Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(count.toString().padStart(2, '0'), pad.left + bw + 24, y + barH / 2 + 4);
  });
}

