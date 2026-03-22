import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* ═══════════════════════════════════════════════
   GLOBE ENGINE — 3D Earth with event effects
   ═══════════════════════════════════════════════ */
export class GlobeEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.markers = [];
    this.effects = [];
    this.autoRotate = true;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.hoveredMarker = null;
    this.onMarkerHover = null;
    this.onMarkerClick = null;
    this.labels = []; // { pos3D, element }
    this.init();
  }

  init() {
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x03050a, 1);

    // Scene
    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    this.camera.position.set(0, 0, 2.8);

    // Controls — fully free rotation
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 6;
    this.controls.enablePan = false;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.3;
    this.controls.rotateSpeed = 0.8;
    this.controls.minPolarAngle = 0;
    this.controls.maxPolarAngle = Math.PI;

    // Lighting
    const ambient = new THREE.AmbientLight(0x445577, 0.6);
    this.scene.add(ambient);
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(5, 3, 5);
    this.scene.add(sun);
    const backLight = new THREE.DirectionalLight(0x3B82F6, 0.3);
    backLight.position.set(-5, -2, -5);
    this.scene.add(backLight);

    // Earth
    this.createEarth();
    // Atmosphere
    this.createAtmosphere();
    // Stars
    this.createStarfield();

    // Events
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('click', (e) => this.onMouseClick(e));
    window.addEventListener('resize', () => this.onResize());

    // Animate
    this.animate();
  }

  createEarth() {
    const geo = new THREE.SphereGeometry(1, 64, 64);

    const loader = new THREE.TextureLoader();
    const PATH = 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/';
    
    const mat = new THREE.MeshPhongMaterial({
      map: loader.load(PATH + 'earth_atmos_2048.jpg'),
      specularMap: loader.load(PATH + 'earth_specular_2048.jpg'),
      normalMap: loader.load(PATH + 'earth_normal_2048.jpg'),
      specular: new THREE.Color(0x222222),
      shininess: 25
    });

    this.earth = new THREE.Mesh(geo, mat);
    this.scene.add(this.earth);

    // Clouds
    const cloudGeo = new THREE.SphereGeometry(1.006, 64, 64);
    const cloudMat = new THREE.MeshPhongMaterial({
      map: loader.load(PATH + 'earth_clouds_1024.png'),
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.clouds = new THREE.Mesh(cloudGeo, cloudMat);
    this.earth.add(this.clouds);

    // Marker group attached to earth
    this.markerGroup = new THREE.Group();
    this.earth.add(this.markerGroup);
  }

  createAtmosphere() {
    const geo = new THREE.SphereGeometry(1.025, 64, 64);
    const mat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * vec4(vPosition, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          float intensity = pow(0.72 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
          gl_FragColor = vec4(0.23, 0.51, 0.96, 1.0) * intensity * 0.6;
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true
    });
    this.atmosphere = new THREE.Mesh(geo, mat);
    this.scene.add(this.atmosphere);
  }

  createStarfield() {
    const geo = new THREE.BufferGeometry();
    const verts = [];
    for (let i = 0; i < 3000; i++) {
      const r = 50 + Math.random() * 100;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      verts.push(r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi));
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.15, sizeAttenuation: true });
    this.scene.add(new THREE.Points(geo, mat));
  }

  /* ── Lat/Lng → 3D position ── */
  latLngToVec3(lat, lng, radius = 1.01) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lng + 180) * (Math.PI / 180);
    return new THREE.Vector3(
      -radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta)
    );
  }

  /* ── LOAD COUNTRY DATASET FOR CLICKING & LABELS ── */
  loadCountryDataset(data) {
    this.countryData = data;
    this.countryNodes = [];
    const container = document.getElementById('globeLabels');
    if (container) container.innerHTML = '';
    
    // Only label the major players to avoid clutter
    const majorCountries = ['US', 'CN', 'RU', 'IN', 'GB', 'FR', 'DE', 'JP', 'BR', 'ZA'];

    data.forEach(c => {
      const pos = this.latLngToVec3(c.lat, c.lng);
      
      // Invisible node for arcs and clicking
      const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 8), mat);
      mesh.position.copy(pos);
      mesh.userData = { type: 'country', ...c };
      this.markerGroup.add(mesh);
      this.markers.push(mesh);
      this.countryNodes.push(mesh);

      // Subtle permanent dot
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.005, 4, 4),
        new THREE.MeshBasicMaterial({ color: 0x4a5568 })
      );
      dot.position.copy(pos);
      this.markerGroup.add(dot);
    });
  }

  /* ── ADD EVENT MARKER ── */
  addMarker(event) {
    const pos = this.latLngToVec3(event.lat, event.lng);
    const colors = { low: 0x22C55E, medium: 0xF59E0B, high: 0xEF4444, critical: 0xFF2020 };
    const color = colors[event.severity] || 0x3B82F6;

    // Invisible hit target for hover/click
    const hitGeo = new THREE.SphereGeometry(0.04, 8, 8);
    const hitMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
    const hitMesh = new THREE.Mesh(hitGeo, hitMat);
    hitMesh.position.copy(pos);
    hitMesh.userData = event;
    this.markerGroup.add(hitMesh);
    this.markers.push(hitMesh);

    // Spawn visual effect
    this.spawnEffect(event, pos, color);

    return hitMesh;
  }

  /* ── VISUAL EFFECTS ── */
  spawnEffect(event, pos, color) {
    // Draw a small glowing node at the event location
    this.spawnCityNode(pos, color);
    // Draw arc connections from event to nearest countries
    this.spawnArcToNearest(pos, color, event.severity);
  }

  spawnArcToNearest(targetPos, color, severity) {
    if (!this.countryNodes || this.countryNodes.length === 0) return;
    // Find 1-3 closest country nodes and draw arcs to them
    const sorted = this.countryNodes
      .map(n => ({ node: n, dist: n.position.distanceTo(targetPos) }))
      .filter(d => d.dist > 0.05) // skip self
      .sort((a, b) => a.dist - b.dist);
    const count = severity === 'critical' ? 3 : severity === 'high' ? 2 : 1;
    for (let i = 0; i < Math.min(count, sorted.length); i++) {
      this.drawArc(targetPos, sorted[i].node.position, color);
    }
  }

  drawArc(start, end, color) {
    const mid = start.clone().lerp(end, 0.5).normalize().multiplyScalar(1.15 + Math.random() * 0.1);
    const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
    const points = curve.getPoints(48);
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.5,
      linewidth: 1
    });
    const line = new THREE.Line(geo, mat);
    this.markerGroup.add(line);

    // Animated pulse dot traveling along the arc
    const dotGeo = new THREE.SphereGeometry(0.004, 6, 6);
    const dotMat = new THREE.MeshBasicMaterial({ color: color, transparent: true, blending: THREE.AdditiveBlending });
    const dot = new THREE.Mesh(dotGeo, dotMat);
    dot.position.copy(start);
    this.markerGroup.add(dot);

    this.effects.push({
      type: 'arc',
      line: line,
      lineMat: mat,
      dot: dot,
      curve: curve,
      progress: 0,
      speed: 0.003 + Math.random() * 0.002,
      loops: 0,
      maxLoops: 3 + Math.floor(Math.random() * 3)
    });
  }

  spawnForest(pos) {
    const group = new THREE.Group();
    group.position.copy(pos);
    group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pos.clone().normalize());

    const numTrees = 3 + Math.floor(Math.random() * 4);
    for (let i = 0; i < numTrees; i++) {
        const tree = new THREE.Group();
        
        const trunkGeo = new THREE.CylinderGeometry(0.001, 0.002, 0.008, 5);
        trunkGeo.translate(0, 0.004, 0);
        const trunkMat = new THREE.MeshPhongMaterial({ color: 0x4B382A });
        const trunk = new THREE.Mesh(trunkGeo, trunkMat);
        tree.add(trunk);

        const leavesGeo = new THREE.ConeGeometry(0.005, 0.015, 6);
        leavesGeo.translate(0, 0.015, 0);
        const leavesMat = new THREE.MeshPhongMaterial({ color: 0x22C55E });
        const leaves = new THREE.Mesh(leavesGeo, leavesMat);
        tree.add(leaves);

        tree.position.set((Math.random()-0.5)*0.03, 0, (Math.random()-0.5)*0.03);
        const scale = 0.6 + Math.random() * 0.8;
        tree.scale.set(scale, scale, scale);
        tree.rotation.x = (Math.random()-0.5)*0.4;
        tree.rotation.z = (Math.random()-0.5)*0.4;

        group.add(tree);
    }
    
    group.scale.set(0.01, 0.01, 0.01);
    this.markerGroup.add(group);

    this.effects.push({
      type: 'forest',
      group: group,
      life: 0,
      targetScale: 1.0,
      swayOff: Math.random() * Math.PI * 2,
      mesh: group // for cleanup
    });
  }

  spawnCityNode(pos, color) {
    const nodeGeo = new THREE.SphereGeometry(0.004, 8, 8);
    const nodeMat = new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.8 });
    const node = new THREE.Mesh(nodeGeo, nodeMat);
    node.position.copy(pos);
    this.markerGroup.add(node);
    
    // Timer for occasional scanner pulses
    this.effects.push({ type: 'city', mesh: node, color: color, life: Math.random() * 4.0 });
  }

  /* ── UPDATE EFFECTS ── */
  updateEffects() {
    for (let i = this.effects.length - 1; i >= 0; i--) {
      const fx = this.effects[i];
      
      if (fx.type === 'arc') {
        fx.progress += fx.speed;
        if (fx.progress >= 1) {
          fx.progress = 0;
          fx.loops++;
          if (fx.loops >= fx.maxLoops) {
            // Fade out the arc
            fx.lineMat.opacity -= 0.02;
            if (fx.lineMat.opacity <= 0) {
              this.markerGroup.remove(fx.line);
              this.markerGroup.remove(fx.dot);
              fx.line.geometry.dispose(); fx.lineMat.dispose();
              fx.dot.geometry.dispose(); fx.dot.material.dispose();
              this.effects.splice(i, 1);
            }
            continue;
          }
        }
        const pt = fx.curve.getPoint(fx.progress);
        fx.dot.position.copy(pt);
        fx.dot.material.opacity = 0.6 + Math.sin(fx.progress * Math.PI) * 0.4;
      }
      else if (fx.type === 'forest') {
        fx.life += 0.01;
        if (fx.life < 1.0) {
          const s = Math.sin((fx.life * Math.PI) / 2) * fx.targetScale;
          fx.group.scale.set(s, s, s);
        } else {
          if (fx.life > 1000) fx.life = 2.0;
          const time = Date.now() * 0.001;
          const sway = Math.sin(time * 2 + fx.swayOff) * 0.05;
          fx.group.children.forEach(tree => {
            tree.rotation.x += sway * 0.02;
            tree.rotation.z += Math.cos(time * 1.5 + fx.swayOff) * 0.01;
          });
        }
      }
      else if (fx.type === 'city') {
        fx.life -= 0.02;
        if (fx.life <= 0) {
          const ringGeo = new THREE.RingGeometry(0.004, 0.008, 16);
          const ringMat = new THREE.MeshBasicMaterial({ color: fx.color, transparent: true, opacity: 0.6, side: THREE.DoubleSide, blending: THREE.AdditiveBlending });
          const ring = new THREE.Mesh(ringGeo, ringMat);
          ring.position.copy(fx.mesh.position);
          ring.lookAt(new THREE.Vector3(0, 0, 0));
          this.markerGroup.add(ring);
          this.effects.push({ type: 'pulse', mesh: ring, life: 1.0, decay: 0.015 });
          fx.life = 2.0 + Math.random() * 3.0;
        }
      }
      else if (fx.type === 'pulse') {
        fx.life -= fx.decay;
        const scale = 1 + (1 - fx.life) * 4;
        fx.mesh.scale.set(scale, scale, scale);
        fx.mesh.material.opacity = Math.max(0, fx.life * 0.8);
        if (fx.life <= 0) {
          this.markerGroup.remove(fx.mesh);
          fx.mesh.geometry.dispose();
          fx.mesh.material.dispose();
          this.effects.splice(i, 1);
        }
      }
    }
  }

  clearMarkers() {
    while (this.markerGroup.children.length) {
      const child = this.markerGroup.children[0];
      this.markerGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }
    this.markers = [];
    this.effects = [];
    this.predictionLines = this.predictionLines || [];
    this.predictionLines.forEach(line => {
      if (line.geometry) line.geometry.dispose();
      if (line.material) line.material.dispose();
    });
    this.predictionLines = [];
  }

  clearPredictionLines() {
    this.predictionLines = this.predictionLines || [];
    this.predictionLines.forEach(line => {
      this.markerGroup.remove(line);
      if (line.geometry) line.geometry.dispose();
      if (line.material) line.material.dispose();
    });
    this.predictionLines = [];
  }

  addPredictionLines(actorCoords, results) {
    this.clearPredictionLines();
    if (!actorCoords || !results || results.length === 0) return;
    const from = this.latLngToVec3(actorCoords.lat, actorCoords.lng, 1.02);
    results.forEach(r => {
      const to = this.latLngToVec3(r.lat, r.lng, 1.02);
      const mid = from.clone().lerp(to, 0.5).normalize().multiplyScalar(1.08);
      const curve = new THREE.QuadraticBezierCurve3(from, mid, to);
      const points = curve.getPoints(32);
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      const intensity = Math.min(1, r.y_plus * 1.2);
      const color = intensity > 0.65 ? 0xEF4444 : intensity > 0.5 ? 0xF59E0B : 0x22C55E;
      const mat = new THREE.LineBasicMaterial({ color, linewidth: 2, transparent: true, opacity: 0.7 });
      const line = new THREE.Line(geo, mat);
      line.userData = r;
      this.markerGroup.add(line);
      this.predictionLines.push(line);
    });
  }

  /* ── MOUSE INTERACTION ── */
  onMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  onMouseClick() {
    if (this.hoveredMarker && this.onMarkerClick) {
      this.onMarkerClick(this.hoveredMarker.userData);
    }
  }

  checkHover() {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hits = this.raycaster.intersectObjects(this.markers);
    const tooltip = document.getElementById('globeTooltip');

    if (hits.length > 0) {
      const marker = hits[0].object;
      this.hoveredMarker = marker;
      this.canvas.style.cursor = 'pointer';
      if (this.onMarkerHover) this.onMarkerHover(marker.userData, hits[0]);

      // Position tooltip
      const projected = marker.getWorldPosition(new THREE.Vector3()).project(this.camera);
      const rect = this.canvas.getBoundingClientRect();
      const x = (projected.x * 0.5 + 0.5) * rect.width;
      const y = (-projected.y * 0.5 + 0.5) * rect.height;
      tooltip.style.left = (x + 15) + 'px';
      tooltip.style.top = (y - 10) + 'px';
      tooltip.style.display = 'block';
      
      if (marker.userData.type === 'country') {
        document.getElementById('ttTitle').textContent = marker.userData.name;
        document.getElementById('ttMeta').textContent = 'SOVEREIGN STATE / REGION';
        document.getElementById('ttDesc').textContent = 'Click to analyze geopolitical landscape, resources, and current tension levels.';
      } else {
        document.getElementById('ttTitle').textContent = marker.userData.title;
        document.getElementById('ttMeta').textContent = `${marker.userData.region} — ${marker.userData.severity.toUpperCase()}`;
        document.getElementById('ttDesc').textContent = marker.userData.description.slice(0, 120) + '...';
      }
    } else {
      this.hoveredMarker = null;
      this.canvas.style.cursor = 'grab';
      tooltip.style.display = 'none';
    }
  }

  updateLabels() {
    if (!this.labels || this.labels.length === 0) return;
    const rect = this.canvas.getBoundingClientRect();
    const halfW = rect.width / 2;
    const halfH = rect.height / 2;
    
    this.labels.forEach(lbl => {
      // Check if coordinate is on the far side of the globe
      const dir = lbl.pos3D.clone().sub(this.camera.position).normalize();
      const dot = dir.dot(lbl.pos3D.clone().normalize());
      
      if (dot > -0.1) {
        // Behind the globe
        lbl.element.style.opacity = '0';
      } else {
        // Project to 2D
        const p = lbl.pos3D.clone().project(this.camera);
        const x = (p.x * halfW) + halfW;
        const y = -(p.y * halfH) + halfH;
        
        lbl.element.style.transform = `translate(-50%, -50%) translate(${x}px, ${y}px)`;
        lbl.element.style.opacity = '1';
      }
    });
  }

  onResize() {
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.updateEffects();
    this.checkHover();
    this.updateLabels();
    if (this.clouds) this.clouds.rotation.y += 0.0003;
    this.renderer.render(this.scene, this.camera);
  }
}
