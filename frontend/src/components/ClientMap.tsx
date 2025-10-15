'use client';

import { MapContainer, TileLayer, Marker, Popup, Polygon } from 'react-leaflet';
import L from 'leaflet';
import { useEffect, useMemo, useState } from 'react';
import type React from 'react';
import * as h3 from 'h3-js';
import 'leaflet/dist/leaflet.css';

export type MarkerData = {
  position: [number, number];
  popup: string;
  day: number;
  rowIndex: number;
};

export default function ClientMap({
  markers,
  rows,
  setRows,
  centers,
  h3Indices,
  h3Resolution = 7,
  mode = 'plan',
}: {
  markers: MarkerData[];
  rows: string[][];
  setRows: React.Dispatch<React.SetStateAction<string[][]>>;
  centers?: { lat: number; lng: number; name?: string }[];
  h3Indices?: string[];
  h3Resolution?: number;
  mode?: 'plan' | 'cluster' | 'route';
}) {
  // UI State
  const [showHexes, setShowHexes] = useState(false);
  const [displayRes, setDisplayRes] = useState<number>(h3Resolution);
  const [expandK, setExpandK] = useState<number>(0);
  const [showSettings, setShowSettings] = useState(false);
  const [showPolygons, setShowPolygons] = useState(false);
  const [themeIdx, setThemeIdx] = useState(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('leeway-map-theme');
      if (stored !== null) {
        const idx = parseInt(stored);
        if (!isNaN(idx)) return idx;
      }
    }
    return 0;
  });

  useEffect(() => {
    delete (L.Icon.Default.prototype as any)._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon-2x.png',
      iconUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon.png',
      shadowUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-shadow.png',
    });
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('leeway-map-theme', String(themeIdx));
    }
  }, [themeIdx]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const s = localStorage.getItem('leeway-hex-show');
    const r = localStorage.getItem('leeway-hex-res');
    const k = localStorage.getItem('leeway-hex-expand');
    const p = localStorage.getItem('leeway-terr-polys');
    if (s !== null) setShowHexes(s === '1');
    if (r !== null && !isNaN(parseInt(r))) setDisplayRes(parseInt(r));
    if (k !== null && !isNaN(parseInt(k))) setExpandK(parseInt(k));
    if (p !== null) setShowPolygons(p === '1');
  }, []);
  useEffect(() => { if (typeof window !== 'undefined') localStorage.setItem('leeway-hex-show', showHexes ? '1' : '0'); }, [showHexes]);
  useEffect(() => { if (typeof window !== 'undefined') localStorage.setItem('leeway-hex-res', String(displayRes)); }, [displayRes]);
  useEffect(() => { if (typeof window !== 'undefined') localStorage.setItem('leeway-hex-expand', String(expandK)); }, [expandK]);
  useEffect(() => { if (typeof window !== 'undefined') localStorage.setItem('leeway-terr-polys', showPolygons ? '1' : '0'); }, [showPolygons]);

  const center: [number, number] = markers.length > 0 ? markers[0].position : [55.1, -6.6];

  // Palettes
  const palette = [
    '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6',
    '#bcf60c','#fabebe','#008080','#e6beff','#9a6324','#fffac8','#800000','#aaffc3',
    '#808000','#ffd8b1','#000075','#808080',
  ];
  const territoryPalette = [
    '#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6',
    '#bcf60c','#fabebe','#008080','#e6beff','#9a6324','#fffac8','#800000','#aaffc3',
    '#808000','#ffd8b1','#000075','#808080'
  ];

  const tileThemes = [
    { name: 'OSM Simple', url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', attribution: '&copy; OpenStreetMap contributors' },
    { name: 'CartoDB Positron', url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', attribution: '&copy; OpenStreetMap &copy; CartoDB' },
    { name: 'CartoDB Dark Matter', url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', attribution: '&copy; OpenStreetMap &copy; CartoDB' },
    { name: 'OSM Bright', url: 'https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png', attribution: '&copy; OpenStreetMap &copy; Stadia Maps' },
    { name: 'Stamen Toner', url: 'https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png', attribution: '&copy; OpenStreetMap &copy; Stamen Design' },
  ];

  // Unique territories and polygons
  const uniqueTerritories = useMemo(() => {
    const idx = rows[0]?.findIndex(h => h.toLowerCase().includes('territory')) ?? -1;
    if (idx === -1) return [] as string[];
    return Array.from(new Set(rows.slice(1).map(r => r[idx]).filter(Boolean)));
  }, [rows]);

  const territoryColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    uniqueTerritories.forEach((terr, idx) => {
      if (!terr) return;
      map[String(terr)] = territoryPalette[idx % territoryPalette.length];
    });
    return map;
  }, [uniqueTerritories]);

  const territoryPolygons = useMemo(() => {
    if (!showPolygons) return null;
    const idx = rows[0]?.findIndex(h => h.toLowerCase().includes('territory')) ?? -1;
    if (idx === -1) return null;
    const terrToHexes: Record<string, string[]> = {};
    const latIdx = rows[0].findIndex(h => /lat/i.test(h));
    const lonIdx = rows[0].findIndex(h => /lon/i.test(h));
    if (latIdx === -1 || lonIdx === -1) return null;
    for (let i = 1; i < rows.length; i++) {
      const terr = rows[i][idx];
      if (!terr) continue;
      const lat = parseFloat(rows[i][latIdx]);
      const lon = parseFloat(rows[i][lonIdx]);
      if (isNaN(lat) || isNaN(lon)) continue;
      try {
        const hex = (h3 as any).latLngToCell ? (h3 as any).latLngToCell(lat, lon, displayRes) : null;
        if (!hex) continue;
        if (!terrToHexes[terr]) terrToHexes[terr] = [];
        terrToHexes[terr].push(hex);
      } catch {}
    }
  const out: React.ReactElement[] = [];
    let colorIdx = 0;
    for (const terr of Object.keys(terrToHexes)) {
      const cells = Array.from(new Set(terrToHexes[terr]));
      const color = territoryPalette[colorIdx % territoryPalette.length];
      colorIdx++;
      const layers: React.ReactElement[] = [];
      const drawCells: string[] = ((h3 as any).compactCells ? (h3 as any).compactCells(cells) : cells) as string[];
      for (const c of drawCells) {
        try {
          const boundary = (h3 as any).cellToBoundary ? (h3 as any).cellToBoundary(c, true) : [];
          if (!boundary || (boundary as any[]).length === 0) continue;
          const latlngs = (boundary as [number, number][]) .map(([lng, lat]) => [lat, lng]) as [number, number][];
          layers.push(
            <Polygon key={`poly-${terr}-${c}`} positions={latlngs} pathOptions={{
              fillColor: color, fillOpacity: 0.25, color, weight: 1, opacity: 0.9
            }} />
          );
        } catch {}
      }
      out.push(<>{layers}</>);
    }
    return out;
  }, [rows, displayRes, showPolygons]);

  // Hex layer
  const baseHexes = useMemo(() => {
    if (!h3Indices || !h3Indices.length) return [] as string[];
    return Array.from(new Set(h3Indices));
  }, [h3Indices]);
  const displayHexToTerr = useMemo(() => {
    const map: Record<string, string> = {};
    const latIdx = rows[0]?.findIndex(h => /lat/i.test(h)) ?? -1;
    const lonIdx = rows[0]?.findIndex(h => /lon/i.test(h)) ?? -1;
    const terrIdx = rows[0]?.findIndex(h => h.toLowerCase().includes('territory')) ?? -1;
    if (latIdx === -1 || lonIdx === -1 || terrIdx === -1) return map;
    for (let i = 1; i < rows.length; i++) {
      const terr = rows[i][terrIdx];
      if (!terr) continue;
      const lat = parseFloat(rows[i][latIdx]);
      const lon = parseFloat(rows[i][lonIdx]);
      if (isNaN(lat) || isNaN(lon)) continue;
      try {
        const hex = (h3 as any).latLngToCell ? (h3 as any).latLngToCell(lat, lon, displayRes) : null;
        if (hex) map[hex] = terr;
      } catch {}
    }
    if ((h3 as any).gridDisk && expandK > 0) {
      const additions: Record<string, string> = {};
      for (const [hex, terr] of Object.entries(map)) {
        try {
          const ring: string[] = (h3 as any).gridDisk(hex, expandK) || [];
          for (const h of ring) {
            if (!(h in map) && !(h in additions)) additions[h] = terr;
          }
        } catch {}
      }
      for (const [h, t] of Object.entries(additions)) map[h] = t;
    }
    return map;
  }, [rows, displayRes, expandK]);

  const hexPolygons = useMemo(() => {
    if (!showHexes) return null;
    const out: React.ReactElement[] = [];
    let shown = 0;
    for (const [hex, terr] of Object.entries(displayHexToTerr)) {
      if (shown > 1200) break; // safety cap
      shown++;
      const boundary = (h3 as any).cellToBoundary ? (h3 as any).cellToBoundary(hex, true) : [];
      if (!boundary || (boundary as any[]).length === 0) continue;
      const latlngs = (boundary as [number, number][]) .map(([lng, lat]) => [lat, lng]) as [number, number][];
      const colorIdx = Math.abs(terr.split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0)) % territoryPalette.length;
      const color = territoryPalette[colorIdx];
      out.push(<Polygon key={`hex-${hex}`} positions={latlngs} pathOptions={{ fillColor: color, color, weight: 1, opacity: 0.8, fillOpacity: 0.15 }} />);
    }
    (out as any).length;
    return out;
  }, [displayHexToTerr, showHexes]);

  const globalDayNumbers = useMemo(() => {
    if (!rows || rows.length <= 1) return [] as number[];
    const hdr = rows[0] || [];
    const dayNumberIdx = hdr.findIndex(h => /^(day\s*number|daynumber|day_num|daynum)$/i.test(String(h || '')));
    if (dayNumberIdx === -1) return new Array(rows.length).fill(NaN) as number[];
    const result = new Array(rows.length).fill(NaN) as number[];
    for (let i = 1; i < rows.length; i++) {
      const raw = rows[i][dayNumberIdx];
      const value = Number(raw);
      if (Number.isFinite(value) && value > 0) {
        result[i] = value;
      }
    }
    return result;
  }, [rows]);

  // Helpers
  const blank = (v: any) => !String(v ?? '').trim() || /^n\/?a$/i.test(String(v ?? ''));

  // Render
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Toolbar */}
      <div style={{ position: 'absolute', zIndex: 1000, top: 10, right: 10 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={() => setShowHexes(v => !v)} title={showHexes ? 'Hide Hex overlay' : 'Show Hex overlay'} style={{ width: 36, height: 36, borderRadius: 8, border: '1px solid #ccc', background: '#fff' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill={showHexes ? '#3cb44b' : 'none'} stroke={showHexes ? '#3cb44b' : '#333'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><polygon points="12 2 19 7 19 17 12 22 5 17 5 7"></polygon></svg>
          </button>
          <button onClick={() => setThemeIdx((themeIdx + 1) % tileThemes.length)} title={`Map Theme: ${tileThemes[themeIdx].name}`} style={{ width: 36, height: 36, borderRadius: 8, border: '1px solid #ccc', background: '#fff' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="#4363d8" stroke="#4363d8" strokeWidth="1.5" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="#e6f0ff"/><path d="M12 3a9 9 0 0 1 0 18" fill="#4363d8" opacity="0.15"/><circle cx="8" cy="10" r="1.2" fill="#111"/><circle cx="14" cy="8" r="0.9" fill="#111"/><circle cx="16" cy="13" r="0.7" fill="#111"/></svg>
          </button>
          <div style={{ position: 'relative' }}>
            <button onClick={() => setShowSettings(s => !s)} title="Settings" style={{ width: 36, height: 36, borderRadius: 8, border: '1px solid #ccc', background: '#fff' }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0  0 0-1.82-.33 1.65 1.65 0  0 0-1 1.51V21a2 2 0  1 1-4 0v-.09A1.65 1.65 0  0 0 8 19.4a1.65 1.65 0  0 0-1.82.33l-.06.06a2 2 0  1 1-2.83-2.83l.06-.06A1.65 1.65 0  0 0 3.6 15 1.65 1.65 0  0 0 2 14h0a2 2 0  1 1 0-4h0A1.65 1.65 0  0 0 3.6 9a1.65 1.65 0  0 0-.33-1.82l-.06-.06a2 2 0  1 1 2.83-2.83l.06.06A1.65 1.65 0  0 0 8 3.6a1.65 1.65 0  0 0 1-1.51V2a2 2 0  1 1 4 0v.09A1.65 1.65 0  0 0 15 3.6a1.65 1.65 0  0 0 1.82-.33l.06-.06a2 2 0  1 1 2.83 2.83l-.06.06A1.65 1.65 0  0 0 20.4 9 1.65 1.65 0  0 0 22 10h0a2 2 0  1 1 0 4h0a1.65 1.65 0  0 0-1.6 1z"></path></svg>
            </button>
            {showSettings && (
              <div style={{ position: 'absolute', right: 0, top: 44, background: '#fff', border: '1px solid #ddd', borderRadius: 10, boxShadow: '0 4px 12px rgba(0,0,0,0.15)', padding: 10, width: 280 }}>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6 }}>Map Theme</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6, marginBottom: 10 }}>
                  {tileThemes.map((t, i) => (
                    <button key={t.name} onClick={() => setThemeIdx(i)} title={t.name} style={{ height: 36, borderRadius: 8, border: i === themeIdx ? '2px solid #3cb44b' : '1px solid #ccc', background: '#f7f7f7' }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: '#333' }}>{t.name.split(' ')[0]}</span>
                    </button>
                  ))}
                </div>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6 }}>Hex Overlay</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <button onClick={() => setShowHexes(v => !v)} title={showHexes ? 'Hide hexes' : 'Show hexes'} style={{ width: 36, height: 36, borderRadius: 8, border: '1px solid #ccc', background: '#fff' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill={showHexes ? '#3cb44b' : 'none'} stroke={showHexes ? '#3cb44b' : '#333'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 19 7 19 17 12 22 5 17 5 7"></polygon></svg>
                  </button>
                  <span style={{ fontSize: 12, color: '#555' }}>Hexes {(Object.keys(displayHexToTerr).length) || baseHexes.length}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <button onClick={() => setShowPolygons(p => !p)} title={showPolygons ? 'Hide territory polygons' : 'Show territory polygons'} style={{ width: 36, height: 36, borderRadius: 8, border: '1px solid #ccc', background: '#fff' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill={showPolygons ? '#f58231' : 'none'} stroke={showPolygons ? '#f58231' : '#333'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="7" width="8" height="10" rx="2"/><rect x="13" y="5" width="8" height="12" rx="2"/></svg>
                  </button>
                  <span style={{ fontSize: 12, color: '#555' }}>Territory Polygons</span>
                </div>
                <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>Hex Size</div>
                <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
                  {[5, 6, 7, 8].map(r => (
                    <button key={r} onClick={() => setDisplayRes(r)} title={`Resolution ${r}`} style={{ minWidth: 36, height: 32, borderRadius: 8, border: displayRes === r ? '2px solid #3cb44b' : '1px solid #ccc', background: '#fff', fontWeight: 700 }}>{r}</button>
                  ))}
                </div>
                <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>Expand</div>
                <div style={{ display: 'flex', gap: 6 }}>
                  {[0, 1, 2].map(k => (
                    <button key={k} onClick={() => setExpandK(k)} title={`Expand k=${k}`} style={{ minWidth: 36, height: 32, borderRadius: 8, border: expandK === k ? '2px solid #3cb44b' : '1px solid #ccc', background: '#fff', fontWeight: 700 }}>{k}</button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <MapContainer center={center} zoom={8} style={{ width: '100%', height: '100%' }}>
        <TileLayer url={tileThemes[themeIdx].url} attribution={tileThemes[themeIdx].attribution} />
        {territoryPolygons}
        {hexPolygons}

        {centers && centers.map((c, idx) => {
          const icon = L.divIcon({
            html: `<div style="background:#1f77b4;color:#fff;border-radius:6px;padding:4px 6px;line-height:16px;text-align:center;font-weight:700;box-shadow:0 0 2px rgba(0,0,0,0.5);font-size:12px;">★ ${c.name ? String(c.name).replace(/</g, '&lt;') : 'Base ' + (idx + 1)}</div>`,
            className: ''
          });
          return (
            <Marker key={`center-${idx}`} position={[c.lat, c.lng]} icon={icon as any}>
              <Popup>
                <div><strong>{c.name || `Base ${idx + 1}`}</strong><br />Lat: {c.lat.toFixed(5)}, Lng: {c.lng.toFixed(5)}</div>
              </Popup>
            </Marker>
          );
        })}

        {markers.map((m, i) => {
          const hdr = rows[0] || [];
          const latI = hdr.findIndex((h: string) => /lat/i.test(h));
          const lonI = hdr.findIndex((h: string) => /lon/i.test(h));
          const candidateIdx = Number.isInteger(m.rowIndex) ? m.rowIndex : NaN;
          const rowIdx = Number.isInteger(candidateIdx) && candidateIdx >= 1 && candidateIdx < rows.length
            ? candidateIdx
            : ((latI !== -1 && lonI !== -1)
              ? rows.findIndex((row, idx) => idx > 0 && parseFloat(row[latI]) === m.position[0] && parseFloat(row[lonI]) === m.position[1])
              : -1);
          // Robust header detection (avoid picking 'Calls per day')
          const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
          const headNorm = rows[0].map(h => norm(String(h || '')));
          const groupColIdx = rows[0].findIndex(h => norm(h) === 'group');
          const dayNumberIdx = rows[0].findIndex(h => /^(day\s*number|day_num|dayno|daynum)$/i.test(String(h || '')));
          const dayColIdx = (() => {
            // Prefer explicit Day column
            let i = rows[0].findIndex(h => /^day$/i.test(String(h || '')));
            if (i !== -1) return i;
            // Accept "Day" variants but avoid CallsPerDay, DayOfWeek, Weekday
            i = rows[0].findIndex(h => /^(day\s*[a-z0-9_-]*)$/i.test(String(h || '')) && !/calls\s*per\s*day/i.test(String(h || '')) && !/dayofweek/i.test(String(h || '')) && !/weekday/i.test(String(h || '')) && !/days?\b/i.test(String(h || '')));
            if (i !== -1) return i;
            return -1;
          })();
          const territoryColIdx = rows[0].findIndex(h => h.toLowerCase().includes('territory'));
          const weekColIdx = rows[0].findIndex(h => h.toLowerCase() === 'week');
          const dayOfWeekIdx = rows[0].findIndex(h => h.toLowerCase() === 'dayofweek');
          const dayOfWeekNumIdx = rows[0].findIndex(h => h.toLowerCase() === 'dayofweeknum');
          // dayNumberIdx defined above

          const blank = (v: any) => !String(v ?? '').trim() || /^n\/?a$/i.test(String(v ?? ''));

          let markerDayLabel = '';
          if (rowIdx > 0) {
            if (dayColIdx !== -1 && !blank(rows[rowIdx][dayColIdx])) {
              markerDayLabel = rows[rowIdx][dayColIdx];
            } else if (dayNumberIdx !== -1 && !blank(rows[rowIdx][dayNumberIdx])) {
              markerDayLabel = `Day ${rows[rowIdx][dayNumberIdx]}`;
            } else if (groupColIdx !== -1 && !blank(rows[rowIdx][groupColIdx])) {
              // Cluster mode fallback: use Group (usually like "Day N")
              markerDayLabel = rows[rowIdx][groupColIdx];
            } else {
              const dow = dayOfWeekIdx !== -1 ? rows[rowIdx][dayOfWeekIdx] : '';
              if (!blank(dow)) markerDayLabel = String(dow);
            }
          }
          const territoryLabel = (territoryColIdx !== -1 && rowIdx > 0) ? (rows[rowIdx][territoryColIdx] || '') : '';
          const fallbackDayLabel = markerDayLabel || ((rowIdx > 0 && groupColIdx !== -1) ? (rows[rowIdx][groupColIdx] || '') : '');
          const colorKey = territoryLabel || fallbackDayLabel;
          const colorHashSource = colorKey || 'default';
          const territoryColorIdx = Math.abs(String(colorHashSource).split('').reduce((acc, ch) => acc + ch.charCodeAt(0), 0)) % territoryPalette.length;
          const territoryColor = colorKey
            ? (territoryColorMap[colorKey] ?? territoryPalette[territoryColorIdx])
            : '#888';
          const uniqueDays = (() => {
            const sortFn = (a: string, b: string) => {
              const na = parseInt(String(a).match(/\d+/)?.[0] || '');
              const nb = parseInt(String(b).match(/\d+/)?.[0] || '');
              if (!isNaN(na) && !isNaN(nb)) return na - nb;
              if (!isNaN(na)) return -1; if (!isNaN(nb)) return 1; return String(a).localeCompare(String(b));
            };
            if (dayColIdx !== -1) {
              return Array.from(new Set(rows.slice(1).map(r => r[dayColIdx])))
                .filter(d => !blank(d))
                .sort(sortFn);
            }
            if (groupColIdx !== -1) {
              // Cluster mode: collect unique Group labels that look like days
              return Array.from(new Set(rows.slice(1).map(r => r[groupColIdx])))
                .filter(d => !blank(d))
                .sort(sortFn);
            }
            return [] as string[];
          })();

          // Color and label
          const getDayNum = (val: string) => { const m = String(val ?? '').match(/(\d+)/); return m ? parseInt(m[1]) : NaN; };
          const globalDay = rowIdx > 0 ? globalDayNumbers[rowIdx] : NaN;
          let dayNum = Number(globalDay);
          if (!Number.isFinite(dayNum) && rowIdx > 0 && dayNumberIdx !== -1) {
            const v = Number(rows[rowIdx][dayNumberIdx]);
            if (Number.isFinite(v) && v > 0) dayNum = v;
          }
          if (!Number.isFinite(dayNum)) dayNum = getDayNum(markerDayLabel);
          if (!Number.isFinite(dayNum) && rowIdx > 0 && groupColIdx !== -1) {
            dayNum = getDayNum(rows[rowIdx][groupColIdx]);
          }
          if (!Number.isFinite(dayNum) && typeof m.day === 'number' && m.day > 0) {
            dayNum = m.day;
            markerDayLabel = `Day ${m.day}`;
          }

          const markerLabel = Number.isFinite(dayNum) && dayNum > 0 ? String(Math.trunc(dayNum)) : '';
          const isBlankDay = markerLabel === '';
          const markerColor = territoryColor;

          const territoryTag = '';

          const icon = L.divIcon({
            html: isBlankDay
              ? `<div style="display:flex;flex-direction:column;align-items:center;"><div style=\"background:#e6194b;color:#fff;border-radius:50%;width:30px;height:30px;line-height:30px;text-align:center;font-weight:bold;box-shadow:0 0 2px rgba(0,0,0,0.5);font-size:22px;text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;\">&#10006;</div></div>`
              : `<div style="display:flex;flex-direction:column;align-items:center;"><div style=\"background:${markerColor};color:#fff;border-radius:50%;width:30px;height:30px;line-height:30px;text-align:center;font-weight:bold;box-shadow:0 0 2px rgba(0,0,0,0.5);font-size:18px;text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;\">${markerLabel}</div></div>`,
            className: ''
          });

          const postcodeIdx = rows[0].findIndex(h => h.toLowerCase().includes('postcode'));
          const postcode = rowIdx > 0 && postcodeIdx !== -1 ? rows[rowIdx][postcodeIdx] : '';

          const handleDayChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
            const value = e.target.value;
            let newDayValue = value;
            if (value === '__new__') {
              let newDay = prompt('Enter new day label (e.g. Day 4a):');
              if (!newDay || !newDay.trim()) return;
              newDay = newDay.trim();
              if (!/^Day\s+/i.test(newDay)) newDay = `Day ${newDay}`;
              newDayValue = newDay;
            }
            if (rowIdx <= 0 || dayColIdx === -1) return;
            // previous day number
            let prevDayNum = NaN as number;
            if (dayNumberIdx !== -1) { const v = parseInt(rows[rowIdx][dayNumberIdx]); if (!isNaN(v) && v > 0) prevDayNum = v; }
            if (isNaN(prevDayNum)) { const m = String(rows[rowIdx][dayColIdx] || '').match(/(\d+)/); if (m) prevDayNum = parseInt(m[1]); }

            // optimistic local update
            const updated = rows.map((r, idx) => (idx === rowIdx ? [...r] : [...r]));
            const header = updated[0] || [];
            const dIdx = header.findIndex(h => String(h).toLowerCase() === 'day' || (String(h).toLowerCase().includes('day') && !String(h).toLowerCase().includes('dayofweek') && !String(h).toLowerCase().includes('dayofweeknum') && !String(h).toLowerCase().includes('weekday')));
            const dnIdx = header.findIndex(h => /(day[\s_]*number)/i.test(String(h)));
            const terrIdx = header.findIndex(h => String(h).toLowerCase().includes('territory'));
            const cpdIdx = header.findIndex(h => /(calls[\s_]*per[\s_]*day|callsperday|calls\/?.*day)/i.test(String(h)));
            const weekIdx = header.findIndex(h => /^week$/i.test(String(h)));
            const dowIdx = header.findIndex(h => /^dayofweek$/i.test(String(h)));
            const dwnIdx = header.findIndex(h => /^dayofweeknum$/i.test(String(h)));
            const resIdx = header.findIndex(h => /resource/i.test(String(h)));

            if (dIdx !== -1) updated[rowIdx][dIdx] = newDayValue;
            let newNum = NaN as number; if (newDayValue) { const m = String(newDayValue).match(/(\d+)/); if (m) newNum = parseInt(m[1]); }
            const dayNames = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
            const dayOfWeekNum = Number.isFinite(newNum) && newNum > 0 ? (((newNum - 1) % 7) + 1) : NaN;
            const dayOfWeekLabel = Number.isFinite(dayOfWeekNum) ? dayNames[dayOfWeekNum - 1] : '';
            const weekNum = Number.isFinite(newNum) && newNum > 0 ? Math.ceil(newNum / 7) : NaN;
            if (dnIdx !== -1) updated[rowIdx][dnIdx] = Number.isFinite(newNum) ? String(newNum) : '';
            if (weekIdx !== -1) updated[rowIdx][weekIdx] = Number.isFinite(weekNum) ? String(weekNum) : '';
            if (dwnIdx !== -1) updated[rowIdx][dwnIdx] = Number.isFinite(dayOfWeekNum) ? String(dayOfWeekNum) : '';
            if (dowIdx !== -1) updated[rowIdx][dowIdx] = dayOfWeekLabel || '';

            if (cpdIdx !== -1) {
              const counts: Record<string, number> = {};
              const getNum = (val: any): number => { const m = String(val ?? '').match(/(\d+)/); return m ? parseInt(m[1]) : NaN; };
              const hasVRP = (dowIdx !== -1) && (weekIdx !== -1) && (resIdx !== -1);
              if (hasVRP) {
                for (let i = 1; i < updated.length; i++) {
                  const r = updated[i];
                  const res = String(r[resIdx] || ''); const w = String(r[weekIdx] || ''); const dlabel = String(r[dowIdx] || '');
                  if (!res || !w || !dlabel) continue; const key = `${res}||${w}||${dlabel}`; counts[key] = (counts[key] || 0) + 1;
                }
                for (let i = 1; i < updated.length; i++) {
                  const r = updated[i];
                  const res = String(r[resIdx] || ''); const w = String(r[weekIdx] || ''); const dlabel = String(r[dowIdx] || '');
                  const key = (res && w && dlabel) ? `${res}||${w}||${dlabel}` : ''; r[cpdIdx] = key ? String(counts[key] || 0) : '';
                }
              } else {
                for (let i = 1; i < updated.length; i++) {
                  const r = updated[i]; const terr = terrIdx !== -1 ? String(r[terrIdx] || '') : '';
                  let n = NaN; if (dnIdx !== -1) { const v = parseInt(String(r[dnIdx] || '')); if (!isNaN(v) && v > 0) n = v; }
                  if (isNaN(n) && dIdx !== -1) n = getNum(r[dIdx]); if (!isNaN(n) && n > 0) { const key = `${terr}||${n}`; counts[key] = (counts[key] || 0) + 1; }
                }
                for (let i = 1; i < updated.length; i++) {
                  const r = updated[i]; const terr = terrIdx !== -1 ? String(r[terrIdx] || '') : '';
                  let n = NaN; if (dnIdx !== -1) { const v = parseInt(String(r[dnIdx] || '')); if (!isNaN(v) && v > 0) n = v; }
                  if (isNaN(n) && dIdx !== -1) n = getNum(r[dIdx]); const key = !isNaN(n) && n > 0 ? `${terr}||${n}` : ''; r[cpdIdx] = key ? String(counts[key] || 0) : '';
                }
              }
            }

            setRows(updated);

            // Backend orchestration: re-optimise old and new day then normalize counts in one call
            try {
              const envBase = (process.env.NEXT_PUBLIC_API_URL || '').trim();
              const isBrowser = typeof window !== 'undefined';
              const isLocalhost = isBrowser && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
              const origin = isBrowser ? `${window.location.protocol}//${window.location.host}` : '';
              const apiBase = envBase || (isLocalhost ? 'http://localhost:8000' : origin);
              const hdrNow = updated[0] || [];
              const groupCol = hdrNow.includes('Resource') ? 'Resource' : (hdrNow.includes('Territory') ? 'Territory' : (hdrNow.find(h => String(h).toLowerCase() === 'group') || null));
              if (!groupCol) return;
              const payload: any = { rows: updated, groupCol };
              if (Number.isFinite(prevDayNum)) payload.beforeDay = prevDayNum;
              if (Number.isFinite(newNum)) payload.afterDay = newNum;
              const res = await fetch(`${apiBase}/api/reopt/apply-day-change`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
              if (res.ok) {
                const js = await res.json(); if (js && js.rows) setRows(js.rows);
              } else {
                // Fallback for older backends: call /api/reopt/day for before and after
                const hdrNow2 = updated[0] || [];
                const groupCol2 = hdrNow2.includes('Resource') ? 'Resource' : (hdrNow2.includes('Territory') ? 'Territory' : (hdrNow2.find(h => String(h).toLowerCase() === 'group') || null));
                const reopt = async (dayNum: number) => {
                  if (!Number.isFinite(dayNum) || dayNum <= 0) return null as any;
                  const r = await fetch(`${apiBase}/api/reopt/day`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rows: updated, day: dayNum, groupCol: groupCol2 }) });
                  if (!r.ok) return null; return r.json();
                };
                if (Number.isFinite(prevDayNum) && (!Number.isFinite(newNum) || prevDayNum !== newNum)) {
                  const r1 = await reopt(prevDayNum as number); if (r1 && r1.rows) setRows(r1.rows);
                }
                if (Number.isFinite(newNum)) {
                  const r2 = await reopt(newNum as number); if (r2 && r2.rows) setRows(r2.rows);
                }
              }
            } catch (err) { console.error('Auto re-optimise failed', err); }
          };

          const handleTerritoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
            const value = e.target.value;
            let newTerritoryVal = value;
            if (value === '__new__') {
              let t = prompt('Enter new territory label (e.g. Territory 2):');
              if (!t || !t.trim()) return;
              t = t.trim();
              if (!/territory\s+/i.test(t)) t = `Territory ${t}`;
              newTerritoryVal = t;
            }
            if (rowIdx <= 0) return;
            setRows(prev => {
              const header = prev[0] || [];
              let tIdx = header.findIndex(h => h.toLowerCase().includes('territory'));
              if (tIdx === -1) {
                const newHeader = [...header, 'Territory'];
                const newRows = [newHeader, ...prev.slice(1).map(r => [...r, ''])];
                const targetIdx = newHeader.length - 1;
                newRows[rowIdx][targetIdx] = newTerritoryVal;
                return newRows;
              } else {
                const next = prev.map((row, idx) => (idx === rowIdx ? [...row] : row));
                next[rowIdx][tIdx] = newTerritoryVal;
                return next;
              }
            });
          };

          return (
            <Marker key={i} position={m.position} icon={icon as any}>
              <Popup>
                <div>
                  <div><strong>Postcode:</strong> {postcode}</div>
                  {territoryLabel && (<div><strong>Territory:</strong> {territoryLabel}</div>)}
                  {markerDayLabel && (<div><strong>Day:</strong> {markerDayLabel}</div>)}
                  <div style={{ marginTop: 6 }}>
                    Assign to day:
                    <select value={isBlankDay ? '' : markerDayLabel} onChange={handleDayChange} style={{ marginLeft: 6 }}>
                      <option value="">(Unassigned)</option>
                      {uniqueDays.map(day => (<option key={day} value={day}>{day}</option>))}
                      <option value="__new__">+ Create new day</option>
                    </select>
                  </div>
                  <div style={{ marginTop: 6 }}>
                    Assign to territory:
                    <select value={territoryLabel || ''} onChange={handleTerritoryChange} style={{ marginLeft: 6 }}>
                      <option value="">(Unassigned)</option>
                      {uniqueTerritories.map(t => (<option key={t} value={t}>{t}</option>))}
                      <option value="__new__">+ Create new territory</option>
                    </select>
                  </div>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
