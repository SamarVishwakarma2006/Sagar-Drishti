import * as THREE from 'three';
import {
  SitePhysics,
  VariableKey,
  PaletteKey,
  Selection,
  FloatRecord,
  ColorbarSettings,
} from '../types/ocean';
import {
  Ocean,
  CMAPS,
  PALETTES,
  buildLUT,
  clamp,
  lerp,
  isMobile,
  hashStr,
  mulberry32,
  isOcean,
  getBathymetry,
} from '../services/syntheticOcean';

function softDotTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const x = c.getContext('2d')!;
  const g = x.createRadialGradient(32, 32, 0, 32, 32, 30);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.4, 'rgba(255,255,255,0.55)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g;
  x.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

export class UnderwaterEngine {
  private renderer: THREE.WebGLRenderer | null = null;
  private scene: THREE.Scene | null = null;
  private camera: THREE.PerspectiveCamera | null = null;
  private container: HTMLElement | null = null;
  private site: SitePhysics | null = null;
  private disposed = false;
  private ro: ResizeObserver | null = null;
  private raf = 0;

  private params = {
    depth: 60,
    variable: 'temp' as VariableKey,
    timeOffset: 0,
  };

  private colorbar: ColorbarSettings = {
    palette: 'thermal',
    customMin: null,
    customMax: null,
    isLogScale: false,
    opacity: 0.55,
    verticalExaggeration: 1.0,
  };

  private sDepth = 8;
  private yaw = 0;
  private pitch = -0.05;
  private tYaw = 0;
  private tPitch = -0.05;
  private T = 0;

  private N = 0;
  private TRAIL = 12; // Smooth high-resolution ribbon trails
  private HALF = 170;

  private pos!: Float32Array;
  private trail!: Float32Array;
  private jit!: Float32Array;
  private life!: Float32Array;
  private maxLife!: Float32Array;
  private speed!: Float32Array;

  private pp!: Float32Array;
  private pc!: Float32Array;
  private lp!: Float32Array;
  private lc!: Float32Array;

  private lineGeo: THREE.BufferGeometry | null = null;
  private ptGeo: THREE.BufferGeometry | null = null;
  private lineMat: THREE.LineBasicMaterial | null = null;
  private ptMat: THREE.PointsMaterial | null = null;

  private lut: Float32Array = buildLUT(CMAPS.temp.stops);
  private vrange = { min: 0, max: 30 };

  private snowN = 0;
  private snow!: Float32Array;
  private snowSp!: Float32Array;
  private snowGeo: THREE.BufferGeometry | null = null;

  private shafts: THREE.Mesh[] = [];
  private surface: THREE.Mesh | null = null;
  private surfBase!: Float32Array;
  private floorGrp: THREE.Group | null = null;
  private floats: Array<{
    sp: THREE.Sprite;
    line: THREE.Line;
    rec: FloatRecord;
    baseY: number;
    ph: number;
    matN: THREE.SpriteMaterial;
    matH: THREE.SpriteMaterial;
  }> = [];
  private marker: THREE.Sprite | null = null;

  private onSelect: (s: Selection) => void = () => {};
  private onDepthDelta: (d: number) => void = () => {};
  private clock: THREE.Clock | null = null;
  private drag = false;
  private moved = 0;
  private lx = 0;
  private ly = 0;
  private ray = new THREE.Raycaster();
  private col = new THREE.Color();

  mount(
    container: HTMLElement,
    site: SitePhysics,
    cb: {
      onSelect: (s: Selection) => void;
      onDepthDelta: (d: number) => void;
    }
  ) {
    this.container = container;
    this.site = site;
    this.onSelect = cb.onSelect;
    this.onDepthDelta = cb.onDepthDelta;

    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    const r = (this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
    }));
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.setSize(w, h);
    r.domElement.style.touchAction = 'none';
    container.appendChild(r.domElement);

    const sc = (this.scene = new THREE.Scene());
    sc.fog = new THREE.FogExp2(0x0d4257, 0.007);

    const cam = (this.camera = new THREE.PerspectiveCamera(66, w/h, 0.5, 9000));
    cam.rotation.order = 'YXZ';
    cam.position.set(0, -this.sDepth, 0);

    this.clock = new THREE.Clock();
    this.buildParticles();
    this.buildSnow();
    this.buildShafts();
    this.buildSurface();
    this.buildFloor();
    this.buildFloats();
    this.buildMarker();
    this.rebuildLUT();

    this.bindInput(r.domElement);

    this.ro = new ResizeObserver(() => {
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      if (!cw || !ch) return;
      r.setSize(cw, ch);
      cam.aspect = cw / ch;
      cam.updateProjectionMatrix();
    });
    this.ro.observe(container);

    this.loop();
  }

  private rebuildLUT() {
    if (!this.site) return;
    const key = this.params.variable;
    const autoRange = Ocean.rangeAt(this.site, key);

    this.vrange = {
      min: this.colorbar.customMin !== null ? this.colorbar.customMin : autoRange.min,
      max: this.colorbar.customMax !== null ? this.colorbar.customMax : autoRange.max,
    };

    const paletteKey = this.colorbar.palette || (CMAPS[key] ? key : 'thermal');
    const pal = PALETTES[paletteKey as PaletteKey] || CMAPS[key] || PALETTES.thermal;
    this.lut = buildLUT(pal.stops);

    if (this.lineMat) {
      this.lineMat.opacity = clamp(this.colorbar.opacity * 0.85, 0.05, 0.98);
    }
    if (this.ptMat) {
      this.ptMat.opacity = clamp(this.colorbar.opacity, 0.1, 1.0);
    }
  }

  /* ---- GPU-Accelerated Particle Flow Field with Continuous Lifecycle & Ribbons ---- */
  private buildParticles() {
    this.N = isMobile ? 3200 : 7500;
    const N = this.N;
    const TR = this.TRAIL;
    const H = this.HALF;
    const dmax = this.site ? this.site.maxDepth : 2600;

    this.pos = new Float32Array(N * 3);
    this.trail = new Float32Array(N * TR * 3);
    this.jit = new Float32Array(N);
    this.life = new Float32Array(N);
    this.maxLife = new Float32Array(N);
    this.speed = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const x = (Math.random() * 2 - 1) * H;
      const y = -clamp(Math.random() * (dmax * 0.8) + 8, 4, dmax - 15);
      const z = (Math.random() * 2 - 1) * H;
      this.pos[i * 3] = x;
      this.pos[i * 3 + 1] = y;
      this.pos[i * 3 + 2] = z;

      for (let s = 0; s < TR; s++) {
        this.trail[(i * TR + s) * 3] = x;
        this.trail[(i * TR + s) * 3 + 1] = y;
        this.trail[(i * TR + s) * 3 + 2] = z;
      }

      this.jit[i] = Math.random() - 0.5;
      this.life[i] = Math.random(); // Staggered initial lifecycle
      this.maxLife[i] = 2.8 + Math.random() * 3.6; // 2.8 to 6.4 seconds duration
      this.speed[i] = 0.85 + Math.random() * 0.4;
    }

    this.lp = new Float32Array(N * (TR - 1) * 2 * 3);
    this.lc = new Float32Array(N * (TR - 1) * 2 * 3);

    const g = (this.lineGeo = new THREE.BufferGeometry());
    g.setAttribute('position', new THREE.BufferAttribute(this.lp, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(this.lc, 3).setUsage(THREE.DynamicDrawUsage));

    this.lineMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.scene!.add(new THREE.LineSegments(g, this.lineMat));

    this.pp = new Float32Array(N * 3);
    this.pc = new Float32Array(N * 3);

    const pg = (this.ptGeo = new THREE.BufferGeometry());
    pg.setAttribute('position', new THREE.BufferAttribute(this.pp, 3).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute('color', new THREE.BufferAttribute(this.pc, 3).setUsage(THREE.DynamicDrawUsage));

    this.ptMat = new THREE.PointsMaterial({
      size: 2.2,
      map: softDotTexture(),
      transparent: true,
      opacity: 0.65,
      vertexColors: true,
      depthWrite: false,
      sizeAttenuation: true,
    });
    this.scene!.add(new THREE.Points(pg, this.ptMat));
  }

  private updateParticles(dt: number) {
    if (!this.site) return;
    const site = this.site;
    const TR = this.TRAIL;
    const H = this.HALF;
    const N = this.N;
    const t = this.params.timeOffset * 0.6 + this.T * 0.05;
    const camY = -this.sDepth;
    const VIS = 34;
    const dtc = Math.min(dt, 0.05);

    const { min, max } = this.vrange;
    const span = Math.max(1e-6, max - min);
    const key = this.params.variable;
    const lut = this.lut;
    const pp = this.pp;
    const pc = this.pc;
    const lp = this.lp;
    const lc = this.lc;
    const pos = this.pos;
    const trail = this.trail;
    const jit = this.jit;
    const life = this.life;
    const maxLife = this.maxLife;
    const speed = this.speed;
    const dmax = site.maxDepth;
    const isLog = this.colorbar.isLogScale;
    const vertExag = this.colorbar.verticalExaggeration || 1.0;
    const baseOpacity = this.colorbar.opacity;

    for (let i = 0; i < N; i++) {
      const i3 = i * 3;
      let x = pos[i3];
      let y = pos[i3 + 1];
      let z = pos[i3 + 2];
      const d = clamp(-y, 1, dmax - 2);
      const spd = speed[i];

      // RK2 Vector Advection for smooth fluid flow curves
      const v1 = Ocean.vectorAt(site, x, z, d, t);
      const midX = x + v1.u * VIS * dtc * 0.5 * spd;
      const midZ = z + v1.v * VIS * dtc * 0.5 * spd;
      const midD = clamp(-(y + v1.w * 3.2 * dtc * vertExag * 0.5 * spd), 1, dmax - 2);
      const v2 = Ocean.vectorAt(site, midX, midZ, midD, t + dtc * 0.5);

      let nx = x + v2.u * VIS * dtc * spd;
      let ny = y + v2.w * 3.2 * dtc * vertExag * spd;
      let nz = z + v2.v * VIS * dtc * spd;

      // Particle lifecycle advance
      life[i] += dtc / maxLife[i];
      let curLife = life[i];

      const top = camY + H;
      const bot = camY - H;
      let respawn = false;

      // Check boundary exit or lifecycle expiration
      if (
        curLife >= 1.0 ||
        nx > H || nx < -H ||
        nz > H || nz < -H ||
        ny > top || ny < bot ||
        ny < -(dmax - 12)
      ) {
        respawn = true;
      }

      const b = i * TR * 3;

      if (respawn) {
        // Continuous upstream / random volume respawn
        nx = (Math.random() * 2 - 1) * H;
        ny = -clamp(Math.random() * (dmax * 0.75) + 6, 4, dmax - 15);
        nz = (Math.random() * 2 - 1) * H;
        pos[i3] = nx;
        pos[i3 + 1] = ny;
        pos[i3 + 2] = nz;
        life[i] = 0.0;
        maxLife[i] = 2.8 + Math.random() * 3.6;
        curLife = 0.0;

        for (let s = 0; s < TR; s++) {
          trail[b + s * 3] = nx;
          trail[b + s * 3 + 1] = ny;
          trail[b + s * 3 + 2] = nz;
        }
      } else {
        pos[i3] = nx;
        pos[i3 + 1] = ny;
        pos[i3 + 2] = nz;

        // Shift trail points smoothly
        trail.copyWithin(b, b + 3, b + TR * 3);
        trail[b + (TR - 1) * 3] = nx;
        trail[b + (TR - 1) * 3 + 1] = ny;
        trail[b + (TR - 1) * 3 + 2] = nz;
      }

      // Smooth lifecycle envelope: fade in (0..0.2), peak (0.2..0.8), fade out (0.8..1.0)
      const lifeFade =
        curLife < 0.2
          ? curLife / 0.2
          : curLife > 0.8
          ? Math.max(0, (1.0 - curLife) / 0.2)
          : 1.0;

      // Color sampling from active variable at particle depth
      const dd = clamp(-ny, 1, dmax - 2);
      const j = jit[i];
      let val: number;
      if (key === 'cur') val = v2.speed;
      else if (key === 'temp') val = Ocean.tempAt(site, dd, t, j * 0.5);
      else if (key === 'sal') val = Ocean.salAt(site, dd, j * 0.06);
      else val = Ocean.oxyAt(site, dd, j * 4);

      let f = (val - min) / span;
      if (isLog && f > 0) {
        f = Math.log10(1 + 9 * clamp(f, 0, 1));
      }
      f = f < 0 ? 0 : f > 1 ? 1 : f;

      const idx = (f * 255) | 0;
      const cr = lut[idx * 3];
      const cg = lut[idx * 3 + 1];
      const cb = lut[idx * 3 + 2];
      const dim = 1 - clamp((this.sDepth - 2500) / 7000, 0, 1) * 0.35;

      // Render smooth fading ribbon trail segments
      for (let s = 0; s < TR - 1; s++) {
        const o = (i * (TR - 1) + s) * 6;
        const s3 = s * 3;
        const s31 = (s + 1) * 3;

        lp[o] = trail[b + s3];
        lp[o + 1] = trail[b + s3 + 1];
        lp[o + 2] = trail[b + s3 + 2];
        lp[o + 3] = trail[b + s31];
        lp[o + 4] = trail[b + s31 + 1];
        lp[o + 5] = trail[b + s31 + 2];

        // Exponential ribbon opacity falloff towards tail
        const alpha0 = Math.pow((s + 1) / TR, 1.7) * lifeFade * baseOpacity * dim;
        const alpha1 = Math.pow((s + 2) / TR, 1.7) * lifeFade * baseOpacity * dim;

        lc[o] = cr * alpha0;
        lc[o + 1] = cg * alpha0;
        lc[o + 2] = cb * alpha0;
        lc[o + 3] = cr * alpha1;
        lc[o + 4] = cg * alpha1;
        lc[o + 5] = cb * alpha1;
      }

      // Leading particle head
      pp[i3] = nx;
      pp[i3 + 1] = ny;
      pp[i3 + 2] = nz;
      const headAlpha = lifeFade * baseOpacity * 0.9 * dim;
      pc[i3] = cr * headAlpha;
      pc[i3 + 1] = cg * headAlpha;
      pc[i3 + 2] = cb * headAlpha;
    }

    this.lineGeo!.attributes.position.needsUpdate = true;
    this.lineGeo!.attributes.color.needsUpdate = true;
    this.ptGeo!.attributes.position.needsUpdate = true;
    this.ptGeo!.attributes.color.needsUpdate = true;
  }

  private buildSnow() {
    this.snowN = isMobile ? 200 : 550;
    this.snow = new Float32Array(this.snowN * 3);
    this.snowSp = new Float32Array(this.snowN);

    for (let i = 0; i < this.snowN; i++) {
      this.snow[i * 3] = (Math.random() * 2 - 1) * 140;
      this.snow[i * 3 + 1] = -Math.random() * 300;
      this.snow[i * 3 + 2] = (Math.random() * 2 - 1) * 140;
      this.snowSp[i] = 0.5 + Math.random() * 1.6;
    }

    const g = (this.snowGeo = new THREE.BufferGeometry());
    g.setAttribute('position', new THREE.BufferAttribute(this.snow, 3).setUsage(THREE.DynamicDrawUsage));
    this.scene!.add(
      new THREE.Points(
        g,
        new THREE.PointsMaterial({
          size: 1.15,
          map: softDotTexture(),
          color: 0xbcd8de,
          transparent: true,
          opacity: 0.22,
          depthWrite: false,
          sizeAttenuation: true,
        })
      )
    );
  }

  private updateSnow(dt: number) {
    const H = 150;
    const camY = -this.sDepth;
    const t = this.T;
    for (let i = 0; i < this.snowN; i++) {
      const i3 = i * 3;
      let x = this.snow[i3] + Math.sin(t * 0.4 + i * 1.7) * 0.6 * dt;
      let y = this.snow[i3 + 1] - this.snowSp[i] * dt;
      let z = this.snow[i3 + 2];
      if (y < camY - H) {
        y = camY + H;
        x = (Math.random() * 2 - 1) * H;
        z = (Math.random() * 2 - 1) * H;
      }
      this.snow[i3] = x;
      this.snow[i3 + 1] = y;
      this.snow[i3 + 2] = z;
    }
    this.snowGeo!.attributes.position.needsUpdate = true;
  }

  private buildShafts() {
    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 256;
    const x = c.getContext('2d')!;
    const gr = x.createLinearGradient(0, 0, 0, 256);
    gr.addColorStop(0, 'rgba(210,240,250,0.55)');
    gr.addColorStop(0.7, 'rgba(210,240,250,0.12)');
    gr.addColorStop(1, 'rgba(210,240,250,0)');
    x.fillStyle = gr;
    x.fillRect(0, 0, 128, 256);
    const tex = new THREE.CanvasTexture(c);

    for (let i = 0; i < 7; i++) {
      const w = 60 + Math.random() * 110;
      const h = 750 + Math.random() * 250;
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({
          map: tex,
          transparent: true,
          opacity: 0.08,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          side: THREE.DoubleSide,
          fog: false,
        })
      );
      m.position.set((Math.random() * 2 - 1) * 260, -h / 2 + 40, (Math.random() * 2 - 1) * 260);
      m.rotation.y = Math.random() * Math.PI;
      m.userData.ph = Math.random() * 6.28;
      this.shafts.push(m);
      this.scene!.add(m);
    }
  }

  private updateShafts() {
    const so = Math.max(0, 1 - this.sDepth / 240);
    this.shafts.forEach((m) => {
      m.visible = so > 0.015;
      if (!m.visible) return;
      (m.material as THREE.MeshBasicMaterial).opacity =
        0.12 * so * (0.65 + 0.35 * Math.sin(this.T * 0.35 + m.userData.ph));
      m.rotation.z = Math.sin(this.T * 0.1 + m.userData.ph) * 0.035;
    });
  }

  private buildSurface() {
    const g = new THREE.PlaneGeometry(2600, 2600, 56, 56);
    g.rotateX(-Math.PI / 2);
    this.surfBase = (g.attributes.position.array as Float32Array).slice();
    this.surface = new THREE.Mesh(
      g,
      new THREE.MeshBasicMaterial({
        color: 0xaadde8,
        transparent: true,
        opacity: 0.28,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    this.scene!.add(this.surface);
  }

  private updateSurface() {
    if (!this.surface) return;
    const v = this.surface.geometry.attributes.position.array as Float32Array;
    const t = this.T;
    for (let i = 0; i < v.length; i += 3) {
      const x = this.surfBase[i];
      const z = this.surfBase[i + 2];
      v[i + 1] =
        1.1 * Math.sin(x * 0.045 + t * 0.8) +
        0.8 * Math.cos(z * 0.037 + t * 0.56) +
        0.4 * Math.sin((x + z) * 0.02 - t * 0.4);
    }
    this.surface.geometry.attributes.position.needsUpdate = true;
  }

  private buildFloor() {
    if (!this.site) return;
    const g = new THREE.PlaneGeometry(3600, 3600, 72, 72);
    g.rotateX(-Math.PI / 2);
    const p = g.attributes.position.array as Float32Array;
    for (let i = 0; i < p.length; i += 3) {
      const x = p[i];
      const z = p[i + 2];
      p[i + 1] =
        16 * Math.sin(x * 0.011 + 1.7) * Math.cos(z * 0.009 + 0.6) +
        8 * Math.sin(x * 0.031 + 0.4) * Math.sin(z * 0.027 + 2.1) +
        3.5 * Math.sin(x * 0.071 + z * 0.063);
    }
    const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x04141c }));
    const wire = new THREE.Mesh(
      g,
      new THREE.MeshBasicMaterial({
        color: 0x2a7d8c,
        wireframe: true,
        transparent: true,
        opacity: 0.07,
      })
    );
    this.floorGrp = new THREE.Group();
    this.floorGrp.add(mesh);
    this.floorGrp.add(wire);
    this.floorGrp.position.y = -this.site.maxDepth;
    this.scene!.add(this.floorGrp);
  }

  private makeFloatTexture(f: FloatRecord, hi: boolean): THREE.CanvasTexture {
    const c = document.createElement('canvas');
    c.width = 320;
    c.height = 176;
    const x = c.getContext('2d')!;
    const ring = hi ? 'rgba(190,245,255,1)' : 'rgba(120,224,238,0.85)';
    x.strokeStyle = ring;
    x.lineWidth = 4;
    x.beginPath();
    x.arc(58, 60, 26, 0, 6.2832);
    x.stroke();
    x.fillStyle = ring;
    x.beginPath();
    x.arc(58, 60, 7, 0, 6.2832);
    x.fill();
    if (hi) {
      x.strokeStyle = 'rgba(190,245,255,0.4)';
      x.lineWidth = 2;
      x.beginPath();
      x.arc(58, 60, 36, 0, 6.2832);
      x.stroke();
    }
    x.fillStyle = hi ? '#eafcff' : '#c9eef5';
    x.font = '600 25px "IBM Plex Mono", monospace';
    x.fillText(f.id, 104, 52);
    x.fillStyle = '#8fb4bc';
    x.font = '500 19px "IBM Plex Mono", monospace';
    x.fillText(`${Math.round(f.parkingDepth)} m · cycle ${f.cycle}`, 104, 84);
    x.fillText('ARGO · OBSERVED', 104, 112);
    return new THREE.CanvasTexture(c);
  }

  private buildFloats() {
    if (!this.site) return;
    const site = this.site;
    Ocean.floatsAt(site).forEach((f) => {
      const rnd = mulberry32(hashStr(f.id));
      const ang = rnd() * 6.2832;
      const rad = 130 + rnd() * 250;
      const x = Math.cos(ang) * rad;
      const z = Math.sin(ang) * rad;
      const y = -clamp(f.parkingDepth, 15, site.maxDepth - 15);
      const matN = new THREE.SpriteMaterial({
        map: this.makeFloatTexture(f, false),
        transparent: true,
        depthWrite: false,
      });
      const matH = new THREE.SpriteMaterial({
        map: this.makeFloatTexture(f, true),
        transparent: true,
        depthWrite: false,
      });
      const sp = new THREE.Sprite(matN);
      sp.position.set(x, y, z);
      sp.scale.set(40, 22, 1);
      sp.userData.fid = f.id;
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -4, 0), new THREE.Vector3(0, -26, 0)]),
        new THREE.LineBasicMaterial({ color: 0x3f96a5, transparent: true, opacity: 0.55 })
      );
      line.position.set(x, y - 10, z);
      this.floats.push({ sp, line, rec: f, baseY: y, ph: rnd() * 6.28, matN, matH });
      this.scene!.add(sp);
      this.scene!.add(line);
    });
  }

  private updateFloats() {
    const camY = -this.sDepth;
    this.floats.forEach((f) => {
      const vis = Math.abs(f.baseY - camY) < 520;
      f.sp.visible = vis;
      f.line.visible = vis;
      if (!vis) return;
      f.sp.position.y = f.baseY + Math.sin(this.T * 0.5 + f.ph) * 1.4;
      f.sp.material.opacity = 0.72 + 0.28 * Math.sin(this.T * 1.6 + f.ph);
    });
  }

  private buildMarker() {
    const c = document.createElement('canvas');
    c.width = c.height = 96;
    const x = c.getContext('2d')!;
    x.strokeStyle = '#eafcff';
    x.lineWidth = 5;
    x.beginPath();
    x.arc(48, 48, 26, 0, 6.2832);
    x.stroke();
    x.lineWidth = 3;
    x.beginPath();
    x.moveTo(48, 10);
    x.lineTo(48, 26);
    x.moveTo(48, 70);
    x.lineTo(48, 86);
    x.moveTo(10, 48);
    x.lineTo(26, 48);
    x.moveTo(70, 48);
    x.lineTo(86, 48);
    x.stroke();
    this.marker = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: new THREE.CanvasTexture(c),
        transparent: true,
        depthWrite: false,
      })
    );
    this.marker.scale.set(26, 26, 1);
    this.marker.visible = false;
    this.scene!.add(this.marker);
  }

  private bindInput(el: HTMLElement) {
    el.addEventListener('pointerdown', (e: PointerEvent) => {
      this.drag = true;
      this.moved = 0;
      this.lx = e.clientX;
      this.ly = e.clientY;
      el.setPointerCapture(e.pointerId);
    });

    el.addEventListener('pointermove', (e: PointerEvent) => {
      if (this.drag) {
        const dx = e.clientX - this.lx;
        const dy = e.clientY - this.ly;
        this.lx = e.clientX;
        this.ly = e.clientY;
        this.moved += Math.abs(dx) + Math.abs(dy);
        this.tYaw -= dx * 0.0031;
        this.tPitch = clamp(this.tPitch - dy * 0.0031, -1.35, 1.35);
      } else if (this.renderer && this.camera && this.site) {
        // Bathymetric masking feedback on hover
        const r = this.renderer.domElement.getBoundingClientRect();
        const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
        const ny = -((e.clientY - r.top) / r.height) * 2 + 1;
        this.ray.setFromCamera(new THREE.Vector2(nx, ny), this.camera);
        const p = this.ray.ray.at(150, new THREE.Vector3());
        const ll = Ocean.offsetToLatLon(this.site, p.x, p.z);
        if (isOcean(ll.lat, ll.lon) && p.y <= 0) {
          el.style.cursor = 'crosshair';
        } else {
          el.style.cursor = 'not-allowed';
        }
      }
    });

    el.addEventListener('pointerup', (e: PointerEvent) => {
      if (!this.drag) return;
      this.drag = false;
      if (this.moved < 7) this.handleClick(e);
    });

    el.addEventListener('pointercancel', () => {
      this.drag = false;
    });

    el.addEventListener(
      'wheel',
      (e: WheelEvent) => {
        e.preventDefault();
        const step = this.sDepth < 150 ? 10 : this.sDepth * 0.07;
        this.onDepthDelta((e.deltaY > 0 ? 1 : -1) * step);
      },
      { passive: false }
    );
  }

  private handleClick(e: PointerEvent) {
    if (!this.renderer || !this.camera || !this.site) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
    const ny = -((e.clientY - r.top) / r.height) * 2 + 1;

    this.ray.setFromCamera(new THREE.Vector2(nx, ny), this.camera);
    const hits = this.ray.intersectObjects(
      this.floats.map((f) => f.sp),
      false
    );
    if (hits.length > 0) {
      this.onSelect({ kind: 'float', id: hits[0].object.userData.fid });
      return;
    }

    const p = this.ray.ray.at(150, new THREE.Vector3());
    const site = this.site;
    const depth = clamp(-p.y, 2, site.maxDepth - 2);
    const ll = Ocean.offsetToLatLon(site, p.x, p.z);

    // Ocean-Only Bathymetric Masking: strictly block land clicks & sample drops
    if (!isOcean(ll.lat, ll.lon) || p.y > 0) {
      return;
    }

    this.onSelect({ kind: 'sample', lat: ll.lat, lon: ll.lon, depth });
  }

  private fogAt(d: number, out: THREE.Color) {
    const stops: [number, string][] = [
      [0, '#0d4257'],
      [180, '#0a3448'],
      [700, '#07293a'],
      [2200, '#03121d'],
      [11000, '#010508'],
    ];
    for (let i = 1; i < stops.length; i++) {
      if (d <= stops[i][0]) {
        const f = (d - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
        out.set(stops[i - 1][1]);
        const b = new THREE.Color(stops[i][1]);
        out.lerp(b, f);
        return out;
      }
    }
    out.set(stops[stops.length - 1][1]);
    return out;
  }

  private loop = () => {
    if (this.disposed || !this.renderer || !this.scene || !this.camera || !this.clock) return;
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(this.clock.getDelta(), 0.1);
    this.T += dt;
    this.sDepth += (this.params.depth - this.sDepth) * Math.min(1, dt * 1.9);
    this.yaw += (this.tYaw - this.yaw) * Math.min(1, dt * 9);
    this.pitch += (this.tPitch - this.pitch) * Math.min(1, dt * 9);

    this.camera.position.set(0, -this.sDepth, 0);
    this.camera.rotation.set(this.pitch, this.yaw, 0);

    this.fogAt(this.sDepth, this.col);
    (this.scene.fog as THREE.FogExp2).color.copy(this.col);
    (this.scene.fog as THREE.FogExp2).density =
      0.0085 + clamp(this.sDepth / 9000, 0, 1) * 0.0035;
    this.renderer.setClearColor(this.col);

    this.updateShafts();
    const surfVisible = this.sDepth < 240;
    if (this.surface) {
      this.surface.visible = surfVisible;
      if (surfVisible) this.updateSurface();
    }
    if (this.floorGrp && this.site) {
      this.floorGrp.visible = this.sDepth > this.site.maxDepth - 520;
    }
    this.updateFloats();
    this.updateParticles(dt);
    this.updateSnow(dt);

    if (this.marker && this.marker.visible) {
      (this.marker.material as THREE.SpriteMaterial).opacity =
        0.75 + 0.25 * Math.sin(this.T * 3);
    }

    this.renderer.render(this.scene, this.camera);
  };

  setParams(p: { depth: number; variable: VariableKey; timeOffset: number }) {
    const changed = p.variable !== this.params.variable;
    Object.assign(this.params, p);
    if (changed) this.rebuildLUT();
  }

  setColorbarSettings(cb: ColorbarSettings) {
    this.colorbar = { ...this.colorbar, ...cb };
    this.rebuildLUT();
  }

  setSelection(sel: Selection | null) {
    this.floats.forEach((f) => {
      f.sp.material = sel && sel.kind === 'float' && sel.id === f.rec.id ? f.matH : f.matN;
    });
    if (sel && sel.kind === 'sample' && this.site && this.marker) {
      const o = Ocean.latLonToOffset(this.site, sel.lat, sel.lon);
      this.marker.position.set(
        clamp(o.x, -this.HALF + 5, this.HALF - 5),
        -sel.depth,
        clamp(o.y, -this.HALF + 5, this.HALF - 5)
      );
      this.marker.visible = true;
    } else if (this.marker) {
      this.marker.visible = false;
    }
  }

  resetView() {
    this.tYaw = 0;
    this.tPitch = -0.05;
  }

  getYaw(): number {
    return this.yaw;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.ro?.disconnect();
    try {
      this.scene?.traverse((o: any) => {
        o.geometry?.dispose?.();
        const m = o.material;
        if (m) {
          (Array.isArray(m) ? m : [m]).forEach((mm: any) => {
            mm.map?.dispose?.();
            mm.dispose?.();
          });
        }
      });
    } catch (e) {
      // Ignore cleanup error
    }
    this.renderer?.dispose();
    const el = this.renderer?.domElement;
    if (el?.parentNode) el.parentNode.removeChild(el);
  }
}
