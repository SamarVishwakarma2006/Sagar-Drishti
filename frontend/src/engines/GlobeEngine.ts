import * as Cesium from 'cesium';
import { SitePhysics, LatLon, BoundingBox } from '../types/ocean';
import { Ocean, isOcean } from '../services/syntheticOcean';

export class GlobeEngine {
  viewer: Cesium.Viewer | null = null;
  private flying = false;
  private lastAct = 0;
  private group: Cesium.Entity[] = [];
  private disasterGroup: Cesium.Entity[] = [];
  private hover: LatLon | null = null;
  private lastHover = 0;

  async init(
    el: HTMLElement,
    cb: {
      onReady: () => void;
      onError: (m: string) => void;
      onPickPoint: (lat: number, lon: number) => void;
      onPickFloat: (id: string) => void;
    }
  ) {
    try {
      // Ensure Cesium is defined
      const viewer = (this.viewer = new Cesium.Viewer(el, {
        animation: false,
        timeline: false,
        baseLayerPicker: false,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        navigationHelpButton: false,
        fullscreenButton: false,
        infoBox: false,
        selectionIndicator: false,
        creditContainer: document.getElementById('cesium-credit') || undefined,
        baseLayer: false,
        terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      }));

      const scene = viewer.scene;
      scene.globe.baseColor = Cesium.Color.fromCssColorString('#0b3a55');
      if (scene.skyAtmosphere) {
        scene.skyAtmosphere.show = true;
      }

      // Natural Earth II Imagery
      try {
        const tms = await Cesium.TileMapServiceImageryProvider.fromUrl(
          Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII')
        );
        const layer = viewer.imageryLayers.addImageryProvider(tms);
        layer.brightness = 0.92;
        layer.saturation = 0.8;
        layer.contrast = 1.12;
        layer.gamma = 1.0;
      } catch (e) {
        // Fallback gracefully on dark globe base color
      }

      // Lat/Lon Graticule overlay
      const grid = viewer.imageryLayers.addImageryProvider(
        new Cesium.GridImageryProvider({
          cells: 6,
          color: Cesium.Color.fromCssColorString('#7ee0ea').withAlpha(0.2),
          glowColor: Cesium.Color.TRANSPARENT,
        })
      );
      grid.alpha = 0.28;

      // Center initial view on the Indian Ocean
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(76, 6, 2.0e7),
      });

      // Remove default double click zoom
      viewer.screenSpaceEventHandler.removeInputAction(
        Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK
      );

      const handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);

      // Track cursor position with Ocean-Only Bathymetric Masking
      handler.setInputAction((m: any) => {
        const now = performance.now();
        if (now - this.lastHover < 70) return;
        this.lastHover = now;
        const cart = viewer.camera.pickEllipsoid(m.endPosition, scene.globe.ellipsoid);
        if (cart) {
          const cg = Cesium.Cartographic.fromCartesian(cart);
          const lat = Cesium.Math.toDegrees(cg.latitude);
          const lon = Cesium.Math.toDegrees(cg.longitude);
          const inOcean = isOcean(lat, lon);

          if (inOcean) {
            scene.canvas.style.cursor = 'crosshair';
            this.hover = { lat, lon };
          } else {
            scene.canvas.style.cursor = 'not-allowed';
            this.hover = null;
          }
        } else {
          scene.canvas.style.cursor = 'default';
          this.hover = null;
        }
      }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

      // Handle click interactions with strict bathymetric ocean filtering
      handler.setInputAction((m: any) => {
        const pk = scene.drillPick(m.position, 3);
        for (let i = 0; i < pk.length; i++) {
          const id = pk[i]?.id?.id;
          if (typeof id === 'string' && id.startsWith('float:')) {
            cb.onPickFloat(id.slice(6));
            return;
          }
        }
        const cart = viewer.camera.pickEllipsoid(m.position, scene.globe.ellipsoid);
        if (cart) {
          const cg = Cesium.Cartographic.fromCartesian(cart);
          const lat = Cesium.Math.toDegrees(cg.latitude);
          const lon = Cesium.Math.toDegrees(cg.longitude);

          // Strictly block pointer interaction and sample drop on landmasses
          if (isOcean(lat, lon)) {
            cb.onPickPoint(lat, lon);
          }
        }
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

      scene.canvas.addEventListener('pointerdown', () => {
        this.lastAct = performance.now();
      });
      scene.canvas.addEventListener('wheel', () => {
        this.lastAct = performance.now();
      }, { passive: true });

      scene.canvas.style.cursor = 'crosshair';

      // Subtle slow planet rotation when idle
      scene.preRender.addEventListener(() => {
        if (this.flying || performance.now() - this.lastAct < 4000) return;
        const h = viewer.camera.positionCartographic.height;
        if (h > 2.5e6) {
          viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z, -0.0004);
        }
      });

      this.lastAct = performance.now();
      cb.onReady();
    } catch (e: any) {
      cb.onError(String(e?.message || 'Cesium Globe initialization failed').toUpperCase());
    }
  }

  getHover(): LatLon | null {
    return this.hover;
  }

  private clearGroup() {
    if (!this.viewer) return;
    this.group.forEach((e) => this.viewer!.entities.remove(e));
    this.group = [];
  }

  clearSite() {
    this.clearGroup();
  }

  showSite(site: SitePhysics) {
    if (!this.viewer) return;
    this.clearGroup();

    // Render Bounding Box rectangle if available, else circle
    if (site.bbox) {
      this.group.push(
        this.viewer.entities.add({
          id: 'site_bbox',
          rectangle: {
            coordinates: Cesium.Rectangle.fromDegrees(
              site.bbox.min_lon,
              site.bbox.min_lat,
              site.bbox.max_lon,
              site.bbox.max_lat
            ),
            material: Cesium.Color.fromCssColorString('#56d4e2').withAlpha(0.08),
            outline: true,
            outlineColor: Cesium.Color.fromCssColorString('#56d4e2').withAlpha(0.6),
            outlineWidth: 2,
          },
        })
      );
    }

    // Site Center Point and Label
    this.group.push(
      this.viewer.entities.add({
        id: 'site',
        position: Cesium.Cartesian3.fromDegrees(site.lon, site.lat),
        ellipse: {
          semiMajorAxis: 90000,
          semiMinorAxis: 90000,
          height: 0,
          material: Cesium.Color.fromCssColorString('#56d4e2').withAlpha(0.06),
          outline: true,
          outlineColor: Cesium.Color.fromCssColorString('#56d4e2').withAlpha(0.4),
          outlineWidth: 1.5,
        },
        point: {
          pixelSize: 8,
          color: Cesium.Color.fromCssColorString('#041018'),
          outlineColor: Cesium.Color.fromCssColorString('#56d4e2'),
          outlineWidth: 2.5,
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        },
        label: {
          text: site.name,
          font: '600 13px "Space Grotesk", sans-serif',
          fillColor: Cesium.Color.fromCssColorString('#d9e7ec'),
          showBackground: true,
          backgroundColor: Cesium.Color.fromCssColorString('rgba(4,10,16,0.85)'),
          backgroundPadding: new Cesium.Cartesian2(8, 5),
          pixelOffset: new Cesium.Cartesian2(0, -20),
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 2.2e7),
        },
      })
    );

    // Float 3D Markers
    const floats = Ocean.floatsAt(site);
    floats.forEach((f) => {
      this.group.push(
        this.viewer!.entities.add({
          id: 'float:' + f.id,
          position: Cesium.Cartesian3.fromDegrees(f.lon, f.lat),
          point: {
            pixelSize: 6,
            color: Cesium.Color.fromCssColorString('#56d4e2').withAlpha(0.95),
            outlineColor: Cesium.Color.fromCssColorString('#041018'),
            outlineWidth: 1.5,
            heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          },
          label: {
            text: f.id,
            font: '500 11px "IBM Plex Mono", monospace',
            fillColor: Cesium.Color.fromCssColorString('#9fd4de'),
            pixelOffset: new Cesium.Cartesian2(0, -12),
            verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 4.5e6),
          },
        })
      );
    });
  }

  flyToSite(site: SitePhysics) {
    if (!this.viewer) return;
    this.flying = true;

    if (site.bbox) {
      const rect = Cesium.Rectangle.fromDegrees(
        site.bbox.min_lon,
        site.bbox.min_lat,
        site.bbox.max_lon,
        site.bbox.max_lat
      );
      this.viewer.camera.flyTo({
        destination: rect,
        duration: 2.0,
        complete: () => {
          this.flying = false;
        },
        cancel: () => {
          this.flying = false;
        },
      });
    } else {
      this.viewer.camera.flyToBoundingSphere(
        new Cesium.BoundingSphere(
          Cesium.Cartesian3.fromDegrees(site.lon, site.lat, 0),
          220000
        ),
        {
          offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-42), 1.25e6),
          duration: 2.0,
          complete: () => {
            this.flying = false;
          },
          cancel: () => {
            this.flying = false;
          },
        }
      );
    }
  }

  resetView() {
    if (!this.viewer) return;
    this.flying = true;
    this.viewer.camera.flyHome(1.6);
    setTimeout(() => {
      this.flying = false;
    }, 1700);
  }

  // Staged dive sequence: approach -> pitch-over -> surface contact
  diveTo(site: SitePhysics, onDone: () => void) {
    if (!this.viewer) return;
    const v = this.viewer;
    this.flying = true;
    const dest = (h: number, pitch: number) => ({
      destination: Cesium.Cartesian3.fromDegrees(site.lon, site.lat, h),
      orientation: { heading: 0, pitch: Cesium.Math.toRadians(pitch), roll: 0 },
    });

    v.camera.flyTo({
      ...dest(300000, -38),
      duration: 1.5,
      easingFunction: Cesium.EasingFunction.QUADRATIC_IN_OUT,
      complete: () =>
        v.camera.flyTo({
          ...dest(6000, -88),
          duration: 2.1,
          easingFunction: Cesium.EasingFunction.QUADRATIC_IN_OUT,
          complete: () =>
            v.camera.flyTo({
              ...dest(140, -89.9),
              duration: 1.6,
              easingFunction: Cesium.EasingFunction.QUADRATIC_IN_OUT,
              complete: () => {
                this.flying = false;
                onDone();
              },
              cancel: () => {
                this.flying = false;
                onDone();
              },
            }),
          cancel: () => {
            this.flying = false;
            onDone();
          },
        }),
      cancel: () => {
        this.flying = false;
        onDone();
      },
    });
  }

  ascendFrom(site: SitePhysics) {
    if (!this.viewer) return;
    const v = this.viewer;
    this.flying = true;
    v.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(site.lon, site.lat, 420),
      orientation: { heading: 0, pitch: Cesium.Math.toRadians(-72), roll: 0 },
    });
    v.camera.flyToBoundingSphere(
      new Cesium.BoundingSphere(Cesium.Cartesian3.fromDegrees(site.lon, site.lat, 0), 1.5e6),
      {
        offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-48), 6.5e6),
        duration: 2.6,
        easingFunction: Cesium.EasingFunction.QUADRATIC_OUT,
        complete: () => {
          this.flying = false;
        },
        cancel: () => {
          this.flying = false;
        },
      }
    );
  }

  markDisasterHazardArea(hazard: {
    name: string;
    type?: string;
    date?: string;
    severity?: string;
    bbox?: BoundingBox | null;
    centroid_lat?: number | null;
    centroid_lon?: number | null;
    track_coordinates?: Array<{ date?: string; lat: number; lon: number; intensity_kts?: number }>;
    analog_assessment?: string;
  }) {
    if (!this.viewer) return;
    this.clearDisasterHazardArea();

    const cLat =
      hazard.centroid_lat ??
      (hazard.bbox ? (hazard.bbox.min_lat + hazard.bbox.max_lat) / 2 : undefined);
    const cLon =
      hazard.centroid_lon ??
      (hazard.bbox ? (hazard.bbox.min_lon + hazard.bbox.max_lon) / 2 : undefined);

    if (cLat === undefined || cLon === undefined) return;

    // 1. Hazard Bounding Box (if available)
    if (hazard.bbox) {
      const b = hazard.bbox;
      this.disasterGroup.push(
        this.viewer.entities.add({
          name: `Hazard BBox: ${hazard.name}`,
          rectangle: {
            coordinates: Cesium.Rectangle.fromDegrees(b.min_lon, b.min_lat, b.max_lon, b.max_lat),
            material: Cesium.Color.fromCssColorString('#ef4444').withAlpha(0.12),
            outline: true,
            outlineColor: Cesium.Color.fromCssColorString('#ef4444').withAlpha(0.7),
            outlineWidth: 2,
          },
        })
      );
    }

    // 2. Centroid Hazard Footprint & Warning Beacon
    this.disasterGroup.push(
      this.viewer.entities.add({
        name: `Hazard Center: ${hazard.name}`,
        position: Cesium.Cartesian3.fromDegrees(cLon, cLat, 0),
        ellipse: {
          semiMajorAxis: 180000,
          semiMinorAxis: 180000,
          height: 0,
          material: Cesium.Color.fromCssColorString('#dc2626').withAlpha(0.22),
          outline: true,
          outlineColor: Cesium.Color.fromCssColorString('#f87171').withAlpha(0.85),
          outlineWidth: 2.5,
        },
        point: {
          pixelSize: 10,
          color: Cesium.Color.fromCssColorString('#ef4444'),
          outlineColor: Cesium.Color.fromCssColorString('#ffffff'),
          outlineWidth: 2.5,
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        },
        label: {
          text: `⚠️ HAZARD ZONE: ${hazard.name.toUpperCase()}\n${hazard.severity ? `[${hazard.severity}] ` : ''}${hazard.date || ''}`,
          font: 'bold 13px "Space Grotesk", sans-serif',
          fillColor: Cesium.Color.fromCssColorString('#fee2e2'),
          showBackground: true,
          backgroundColor: Cesium.Color.fromCssColorString('rgba(153, 27, 27, 0.88)'),
          backgroundPadding: new Cesium.Cartesian2(10, 6),
          pixelOffset: new Cesium.Cartesian2(0, -32),
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 2.5e7),
        },
      })
    );

    // 3. Cyclone Track Polyline (if track coordinates present)
    if (hazard.track_coordinates && hazard.track_coordinates.length > 1) {
      const coords: number[] = [];
      hazard.track_coordinates.forEach((pt) => {
        coords.push(pt.lon, pt.lat);
      });

      this.disasterGroup.push(
        this.viewer.entities.add({
          name: `Track: ${hazard.name}`,
          polyline: {
            positions: Cesium.Cartesian3.fromDegreesArray(coords),
            width: 3.5,
            material: new Cesium.PolylineGlowMaterialProperty({
              glowPower: 0.3,
              color: Cesium.Color.fromCssColorString('#f87171'),
            }),
            clampToGround: true,
          },
        })
      );

      // Track Waypoints
      hazard.track_coordinates.forEach((pt, idx) => {
        this.disasterGroup.push(
          this.viewer!.entities.add({
            name: `Track Pt ${idx + 1}`,
            position: Cesium.Cartesian3.fromDegrees(pt.lon, pt.lat, 0),
            point: {
              pixelSize: 6,
              color: Cesium.Color.fromCssColorString('#fca5a5'),
              outlineColor: Cesium.Color.fromCssColorString('#7f1d1d'),
              outlineWidth: 1.5,
              heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
            },
            label: pt.intensity_kts
              ? {
                  text: `${pt.intensity_kts}kt`,
                  font: '500 10px "IBM Plex Mono", monospace',
                  fillColor: Cesium.Color.fromCssColorString('#fca5a5'),
                  pixelOffset: new Cesium.Cartesian2(0, -12),
                  verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
                  distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 5e6),
                }
              : undefined,
          })
        );
      });
    }
  }

  flyToDisasterArea(
    bbox?: BoundingBox | null,
    centroid?: { lat: number; lon: number } | null
  ) {
    if (!this.viewer) return;
    this.flying = true;

    if (bbox) {
      const rect = Cesium.Rectangle.fromDegrees(
        bbox.min_lon,
        bbox.min_lat,
        bbox.max_lon,
        bbox.max_lat
      );
      this.viewer.camera.flyTo({
        destination: rect,
        duration: 2.0,
        easingFunction: Cesium.EasingFunction.QUADRATIC_OUT,
        complete: () => {
          this.flying = false;
        },
        cancel: () => {
          this.flying = false;
        },
      });
    } else if (centroid) {
      this.viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(centroid.lon, centroid.lat, 1.2e6),
        duration: 2.0,
        easingFunction: Cesium.EasingFunction.QUADRATIC_OUT,
        complete: () => {
          this.flying = false;
        },
        cancel: () => {
          this.flying = false;
        },
      });
    }
  }

  clearDisasterHazardArea() {
    if (!this.viewer) return;
    this.disasterGroup.forEach((e) => {
      try {
        this.viewer!.entities.remove(e);
      } catch (_) {
        // entity might have been removed
      }
    });
    this.disasterGroup = [];
  }

  setPaused(p: boolean) {
    if (this.viewer) {
      this.viewer.useDefaultRenderLoop = !p;
    }
  }

  destroy() {
    try {
      this.viewer?.destroy();
    } catch (e) {
      // Ignore
    }
    this.viewer = null;
  }
}
