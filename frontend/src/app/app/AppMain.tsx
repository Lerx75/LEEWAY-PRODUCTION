"use client";

import * as h3 from 'h3-js';
import dynamic from 'next/dynamic';
import { memo, useEffect, useMemo, useState, useTransition } from 'react';
import * as XLSX from 'xlsx';
import '../globals.css';

import type { MarkerData } from '../../components/ClientMap';
const ClientMap = dynamic(() => import('../../components/ClientMap'), { ssr: false });
// Memoized map wrapper to avoid re-rendering the heavy map on unrelated state changes
const MemoClientMap = memo((props: any) => <ClientMap {...props} />);
export default function AppMain() {
  type Project = { name: string; rows: string[][] };
  const Help = ({ text }: { text: string }) => (
    <span title={text} style={{ marginLeft: 6, cursor: 'help', opacity: 0.7 }}>?</span>
  );
  // --- State and helpers ---
  // Clustering form state
  const [name, setName] = useState('');
  const [file, setFile] = useState<File|null>(null);
  const [fileName, setFileName] = useState('No file selected');
  // Optional Resources Excel for territory planning
  const [resourcesFile, setResourcesFile] = useState<File|null>(null);
  const [minC, setMinC] = useState(5);
  const [maxC, setMaxC] = useState(6);
  // Route-mode: time-budget options
  const [workDayMin, setWorkDayMin] = useState<number>(480);

  // Results and map state
  const [rows, setRows] = useState<string[][]>([]);
  const [markers, setMarkers] = useState<MarkerData[]>([]);
  const [mapKey, setMapKey] = useState(0);
  // Suggested/base centers from backend territory planning
  const [centers, setCenters] = useState<{lat:number; lng:number; name?:string}[] | null>(null);
  // H3 indices for hex overlay (aligned 1:1 with data rows excluding header)
  const [h3Indices, setH3Indices] = useState<string[] | null>(null);

  // Progress/status
  const [progress, setProgress] = useState<number>(0);
  const [status, setStatus] = useState<string>('');

  // Failed postcodes
  const [failedCount, setFailedCount] = useState(0);
  const [failedMessage, setFailedMessage] = useState<string | null>(null);
  const [failedExcelUrl, setFailedExcelUrl] = useState<string | null>(null);

  // Sorting/filtering state
  const [sortCol, setSortCol] = useState<number | null>(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [filterDay, setFilterDay] = useState<number | null>(null);
  const [filterWeek, setFilterWeek] = useState<string | number | null>(null);
  const [filterCallsPerDay, setFilterCallsPerDay] = useState<number | null>(null);
  // Excel-like multi-column filters (AND). Keyed by column index, value is a case-insensitive substring match
  const [colFilters, setColFilters] = useState<Record<number, string>>({});

  // Backend stats
  const [totalMiles, setTotalMiles] = useState<number | null>(null);
  const [totalMinutes, setTotalMinutes] = useState<number | null>(null);

  // AI overlay state
  const [aiOpen, setAiOpen] = useState<boolean>(false);
  const [aiQuestion, setAiQuestion] = useState<string>('');
  const [aiAnswer, setAiAnswer] = useState<string>('');
  const [isTyping, startTransition] = useTransition();
  const [aiHistory, setAiHistory] = useState<{q:string;a:string;provider?:string;used_rows?:number}[]>([]);
  const [aiLoading, setAiLoading] = useState<boolean>(false);
  const AI_NAME = 'Ask LeeW-AI';
  // Undo for last transform
  const [tfUndoRows, setTfUndoRows] = useState<string[][] | null>(null);
  // UI toggles
  // Removed legacy re-optimise/normalize buttons
  const [tableOpen, setTableOpen] = useState(true);

  // Helper: extract day number from "Day 11" or similar
  const getDayNum = (val: string) => {
    if (typeof val !== "string") val = String(val ?? "");
    const m = val.match(/(\d+)/);
    return m ? parseInt(m[1]) : NaN;
  };

  // Helper: get calls per day/route bucket
  const getColumnIdx = (name: string) => rows.length ? rows[0].findIndex(h => h.toLowerCase().includes(name)) : -1;
  const getCallsPerDay = () => {
    if (!rows.length) return {} as Record<string, number>;
    const hdr = rows[0].map(h => String(h || ''));
    const hasVRP = hdr.some(h => /dayofweek/i.test(h)) && hdr.some(h => /^week$/i.test(h)) && hdr.some(h => /resource/i.test(h));
    const counts: Record<string, number> = {};
    if (hasVRP) {
      const dayI = hdr.findIndex(h => /dayofweek/i.test(h));
      const weekI = hdr.findIndex(h => /^week$/i.test(h));
      const resI = hdr.findIndex(h => /resource/i.test(h));
      rows.slice(1).forEach(r => {
        const d = r[dayI] || '';
        const w = r[weekI] || '';
        const res = r[resI] || '';
        if (!d || !w || !res) return;
        const key = `${res}||${w}||${d}`;
        counts[key] = (counts[key] || 0) + 1;
      });
      return counts;
    }
    const dayIdx = getColumnIdx('day');
    if (dayIdx === -1) return {} as Record<string, number>;
    const terrIdx = getColumnIdx('territory');
    rows.slice(1).forEach(r => {
      const dayLabel = r[dayIdx] || '';
      if (!dayLabel || typeof dayLabel !== 'string') return;
      const terrLabel = terrIdx !== -1 ? (r[terrIdx] || '') : '';
      const key = `${terrLabel}||${dayLabel}`;
      counts[key] = (counts[key] || 0) + 1;
    });
    return counts;
  };

  // Helper: get stats for sidebar summary table
  const getSummaryStats = () => {
    if (!rows.length) return null;
    let mapped = Math.max(0, rows.length - 1);
    let failed = failedCount;
  const perDayCounts = getCallsPerDay(); // key depends on mode
  const days = Object.keys(perDayCounts).length;
    const callsCountSummary: Record<number, number> = {};
    Object.values(perDayCounts).forEach(cnt => {
      callsCountSummary[cnt] = (callsCountSummary[cnt] || 0) + 1;
    });
    return { mapped, failed, days, callsCountSummary, totalMiles, totalMinutes };
  };
  const [sidebar, setSidebar] = useState<'none'|'new'|'open'|'settings'|'project'|'stats'>('none');
  const [originalRows, setOriginalRows] = useState<string[][]>([]); // for rerun
  const [projects, setProjects] = useState<Project[]>([]);
  const [current, setCurrent] = useState<Project|null>(null);
  // API base debug helpers
  const [apiBaseUI, setApiBaseUI] = useState<string>('');
  const [apiPing, setApiPing] = useState<string>('');
  const [apiOsrm, setApiOsrm] = useState<string>('');

  // load saved
  useEffect(() => {
    const s = localStorage.getItem('leeway-projects');
    if (s) setProjects(JSON.parse(s));
  }, []);
  // Resolve and store API base for UI/debug
  useEffect(() => {
    const envBase = (process.env.NEXT_PUBLIC_API_URL || '').trim();
    const isBrowser = typeof window !== 'undefined';
    const isLocalhost = isBrowser && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
    const origin = isBrowser ? `${window.location.protocol}//${window.location.host}` : '';
    const base = envBase || (isLocalhost ? 'http://localhost:8000' : origin);
    setApiBaseUI(base);
  }, []);
  
  // Markers with colour by day
  useEffect(() => {
    if (!rows.length) return;
    const hdr = rows[0];
    const postcodeHeaders = [
      'postcode', 'post_code', 'postalcode', 'postal_code', 'eircod', 'eircode','Post Code','post code','TargetPostCode'
    ];
    const latI = hdr.findIndex(h=>/lat/i.test(h));
    const lonI = hdr.findIndex(h=>/lon/i.test(h));
  const dayI = hdr.findIndex(h=>h.toLowerCase().includes('day'));
  const dowI = hdr.findIndex(h=>/dayofweek/i.test(h));
  const dayNumberI = hdr.findIndex(h=>/day[\s_]*number/i.test(String(h)));
  const dayOfWeekNumI = hdr.findIndex(h=>/dayofweeknum/i.test(String(h)));
  const dayNumI = dayNumberI !== -1 ? dayNumberI : dayOfWeekNumI;
    const terrI = hdr.findIndex(h => /resource/i.test(h) || /territory/i.test(h.toLowerCase()));
    const weekI = hdr.findIndex(h => /^week$/i.test(h));
    const popupI = hdr.findIndex(h =>
      postcodeHeaders.some(ph => h.toLowerCase().replace(/[\s_]/g,'').includes(ph))
    );

    // Precompute calls-per-day counts if that filter is active
    let countsMap: Record<string, number> | null = null;
    if (filterCallsPerDay !== null) {
      countsMap = {};
      rows.slice(1).forEach(r => {
        const d = (dowI !== -1 ? r[dowI] : (dayI !== -1 ? r[dayI] : '')) || '';
        const t = terrI !== -1 ? (r[terrI] || '') : '';
        const w = weekI !== -1 ? (r[weekI] || '') : '';
        if (!d) return;
        const key = (weekI !== -1 && terrI !== -1) ? `${t}||${w}||${d}` : `${t}||${d}`;
        countsMap![key] = (countsMap![key] || 0) + 1;
      });
    }

    const mk: MarkerData[] = [];
    for (let i = 1; i < rows.length; i++) {
      const r = rows[i];
      const lat = parseFloat(r[latI]);
      const lon = parseFloat(r[lonI]);
      if (!isFinite(lat) || !isFinite(lon)) continue;
      const popup = popupI !== -1 ? r[popupI] : r[0];

      const dayRaw = dowI !== -1 ? r[dowI] : (dayI !== -1 ? r[dayI] : '');
      let dayNum = Number.NaN;
      if (dayNumI !== -1) {
        dayNum = Number(r[dayNumI]);
      }
      if (!isFinite(dayNum)) {
        const m = String(dayRaw ?? '').match(/(\d+)/);
        if (m) dayNum = parseInt(m[1]);
        if (!isFinite(dayNum) && typeof dayRaw === 'string') {
          const key3 = dayRaw.trim().slice(0,3).toLowerCase();
          const map: Record<string, number> = {mon:1,tue:2,wed:3,thu:4,fri:5,sat:6,sun:7};
          if (map[key3]) dayNum = map[key3];
        }
      }

      // Apply interactive filters
      if (filterDay !== null && Number(dayNum) !== Number(filterDay)) continue;
      if (filterWeek !== null && weekI !== -1) {
        if (String(r[weekI] ?? '') !== String(filterWeek)) continue;
      }
      if (filterCallsPerDay !== null && countsMap) {
        const t = terrI !== -1 ? (r[terrI] || '') : '';
        const w = weekI !== -1 ? (r[weekI] || '') : '';
        const d = String(dayRaw || '');
        const key = (weekI !== -1 && terrI !== -1) ? `${t}||${w}||${d}` : `${t}||${d}`;
        if ((countsMap[key] || 0) !== Number(filterCallsPerDay)) continue;
      }

  mk.push({ position: [lat, lon], popup, day: isFinite(dayNum) ? dayNum : 0, rowIndex: i });
    }

    setMarkers(mk);
  }, [rows, filterDay, filterWeek, filterCallsPerDay]);

  // Compute H3 indices whenever rows change
  useEffect(() => {
    if (!rows || rows.length <= 1) { setH3Indices(null); return; }
    const header = rows[0] || [];
    const latI = header.findIndex(h => /lat/i.test(h));
    const lonI = header.findIndex(h => /lon/i.test(h));
    if (latI === -1 || lonI === -1) { setH3Indices(null); return; }
  const res = 8; // align with backend default unless overridden later
    const idxs: string[] = [];
    for (let i = 1; i < rows.length; i++) {
      const r = rows[i];
      const lat = parseFloat(r[latI]);
      const lon = parseFloat(r[lonI]);
      if (!isFinite(lat) || !isFinite(lon)) { idxs.push(''); continue; }
      try {
    const cell = (h3 as any).latLngToCell ? (h3 as any).latLngToCell(lat, lon, res) : '';
        if (typeof cell === 'string' && cell.length > 0) idxs.push(cell); else idxs.push('');
      } catch {
        idxs.push('');
      }
    }
    setH3Indices(idxs);
  }, [rows]);

  // Force map resize when sidebar changes
  useEffect(() => {
    setMapKey(k => k + 1);
  }, [sidebar]);

  const save = (ps:Project[]) => {
    localStorage.setItem('leeway-projects', JSON.stringify(ps));
    setProjects(ps);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setCallHeaders([]);
      setMapCallName('');
      setMapCallPC('');
      setMapCallDur('');
      setMapCallDays('');
      setMapGroupCol('');
      // Parse headers for mapping (for Vehicle Routing)
      (async () => {
        try {
          const ab = await selectedFile.arrayBuffer();
          const wb = XLSX.read(ab);
          const ws = wb.Sheets[wb.SheetNames[0]];
          const rows1 = XLSX.utils.sheet_to_json(ws, { header: 1 }) as any[][];
          const headers = ((rows1?.[0] || []) as any[]).map((h) => String(h || '').trim()).filter(Boolean);
          setCallHeaders(headers);
          // Auto-suggest mappings if empty
          const norm = (s:string) => s.toLowerCase().replace(/[ _-]/g,'');
          const headersNorm = headers.map(h => ({ raw:h, key:norm(h) }));
          if (!mapCallName) {
            const cand = headersNorm.find(h => /(name|id|call|customer|account|site)/.test(h.key));
            if (cand) setMapCallName(cand.raw);
          }
          if (!mapCallPC) {
            const cand = headersNorm.find(h => /(postcode|postalcode|zipcode|zip|eircode)/.test(h.key));
            if (cand) setMapCallPC(cand.raw);
          }
          if (!mapCallDur) {
            const cand = headersNorm.find(h => /(duration|mins|minutes|time|servicetime)/.test(h.key));
            if (cand) setMapCallDur(cand.raw);
          }
          if (!mapCallDays) {
            const cand = headersNorm.find(h => /(days|weekday|open|availability)/.test(h.key));
            if (cand) setMapCallDays(cand.raw);
          }
          // Deprecated: do not auto-detect group/territory column anymore
          setMapGroupCol('');
        } catch (err) {
          // ignore parse errors; mapping can be typed
        }
      })();
    }
    else {
      setFile(null);
      setFileName('');
      setCallHeaders([]);
      setMapCallName('');
      setMapCallPC('');
      setMapCallDur('');
      setMapCallDays('');
      setMapGroupCol('');
    }
  };
  const handleResourcesFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setResourcesFile(selectedFile);
      // Parse headers for mapping
      (async () => {
        try {
          const ab = await selectedFile.arrayBuffer();
          const wb = XLSX.read(ab);
          const ws = wb.Sheets[wb.SheetNames[0]];
          const rows1 = XLSX.utils.sheet_to_json(ws, { header: 1 }) as any[][];
          const headers = ((rows1?.[0] || []) as any[]).map((h) => String(h || '').trim()).filter(Boolean);
          setResHeaders(headers);
          // Auto-suggest
          const norm = (s:string) => s.toLowerCase().replace(/[ _-]/g,'');
          const headersNorm = headers.map(h => ({ raw:h, key:norm(h) }));
          if (!mapResName) {
            const cand = headersNorm.find(h => /(name|resource|rescource|depot|base)/.test(h.key));
            if (cand) setMapResName(cand.raw);
          }
          if (!mapResPC) {
            const cand = headersNorm.find(h => /(postcode|postalcode|zipcode|zip|eircode)/.test(h.key));
            if (cand) setMapResPC(cand.raw);
          }
          if (!mapResDays) {
            const cand = headersNorm.find(h => /(days|weekday|workdays|availability)/.test(h.key));
            if (cand) setMapResDays(cand.raw);
          }
          if (!mapResStart) {
            const cand = headersNorm.find(h => /(start|shiftstart|starttime)/.test(h.key));
            if (cand) setMapResStart(cand.raw);
          }
          if (!mapResEnd) {
            const cand = headersNorm.find(h => /(end|shiftend|endtime|finish)/.test(h.key));
            if (cand) setMapResEnd(cand.raw);
          }
        } catch (err) {
          // ignore parse errors
        }
      })();
    } else {
      setResourcesFile(null);
    }
  };

  // --- Planning vs Clustering mode and inputs ---
  const [mode, setMode] = useState<'cluster'|'plan'|'route'>('plan');
  const [numTerritories, setNumTerritories] = useState<number>(3);
  // Vehicle routing mappings
  const [mapCallName, setMapCallName] = useState('');
  const [mapCallPC, setMapCallPC] = useState('');
  const [mapCallDur, setMapCallDur] = useState('');
  const [mapCallDays, setMapCallDays] = useState('');
  const [mapResName, setMapResName] = useState(''); // kept for plan mode resources parsing
  const [mapResPC, setMapResPC] = useState('');
  const [mapResDays, setMapResDays] = useState('');
  const [mapResStart, setMapResStart] = useState('');
  const [mapResEnd, setMapResEnd] = useState('');
  const [mapGroupCol, setMapGroupCol] = useState('');
  const [callHeaders, setCallHeaders] = useState<string[]>([]);
  const [resHeaders, setResHeaders] = useState<string[]>([]);
  // Solver knobs
  
  // Resource locations: user can type postcode or lat,lng per line -> parse to array
  const [resourceText, setResourceText] = useState<string>('');
  const parseResources = () => {
    const lines = resourceText.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
    const items: any[] = [];
    lines.forEach((line, i) => {
      // Accept: "postcode" OR "lat,lng" OR "name,postcode" OR "name,lat,lng"
      const parts = line.split(',').map(s=>s.trim()).filter(Boolean);
      if (parts.length === 1) {
        items.push({ postcode: parts[0] });
      } else if (parts.length === 2) {
        const a = Number(parts[0]); const b = Number(parts[1]);
        if (!isNaN(a) && !isNaN(b)) items.push({ lat:a, lng:b });
        else items.push({ name: parts[0], postcode: parts[1] });
      } else if (parts.length >= 3) {
        const a = Number(parts[1]); const b = Number(parts[2]);
        if (!isNaN(a) && !isNaN(b)) items.push({ name: parts[0], lat:a, lng:b });
        else items.push({ name: parts[0], postcode: parts.slice(1).join(',') });
      }
    });
    return items;
  };

  const run = async () => {
    if (!name) { alert("Name required"); return }
    if (mode === 'route') {
      if (!file) { alert('Calls Excel required'); return }
    } else {
      if (!file) { alert('Calls Excel required'); return }
    }
    // Allow rerun for current project name
    if (projects.some(p => p.name === name) && (!current || current.name !== name)) {
      alert("Project name already exists. Please choose a different name.");
      return;
    }
    const fd = new FormData();
    if (mode === 'route') {
      // In route mode the primary file carries the grouping column used downstream.
      fd.append('file', file!);
      fd.append('callsFile', file!);
      if (mapCallName) fd.append('callsNameCol', mapCallName);
      if (mapCallDur) fd.append('callsDurationCol', mapCallDur);
      if (mapCallPC) fd.append('callsPostcodeCol', mapCallPC);
      if (mapCallDays) fd.append('callsDaysCol', mapCallDays);
      fd.append('workDayMinutes', String(workDayMin));
      if (resourcesFile) fd.append('resourcesFile', resourcesFile);
      if (mapResName) fd.append('resNameCol', mapResName);
      if (mapResPC) fd.append('resPostcodeCol', mapResPC);
      if (mapResDays) fd.append('resDaysCol', mapResDays);
      if (mapResStart) fd.append('resStartCol', mapResStart);
      if (mapResEnd) fd.append('resEndCol', mapResEnd);
      fd.append('maxWeeks', '8');
    } else {
      fd.append('file', file!);
      fd.append('minCalls', String(minC));
      fd.append('maxCalls', String(maxC));
    }
    if (mapGroupCol) fd.append('groupCol', mapGroupCol);
    if (mode === 'plan') {
      fd.append('numTerritories', String(numTerritories));
      if (resourcesFile) {
        fd.append('resourcesFile', resourcesFile);
      } else {
        const resList = parseResources();
        if (resList.length > 0) fd.append('resourceLocations', JSON.stringify(resList));
      }
    }

    fd.append('mode', mode);
    fd.append('projectName', name);

    setProgress(10);
    setStatus('Uploading and processing...');
    setFailedCount(0);
    setFailedMessage(null);
    setFailedExcelUrl(null);

    try {
      const response = await fetch('/api/projects/run', { method: 'POST', body: fd });

      setProgress(60);

      if(!response.ok) {
        let msg = response.statusText || `HTTP ${response.status}`;
        try {
          const err = await response.json();
          if (err.detail) msg = err.detail;
          if (err.error) msg = err.error;
        } catch {}
        throw new Error(msg);
      }

      let data = await response.json() as any;

      const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
      const awaitRoutingJob = async (initial: any) => {
        const jobId = String(initial?.job_id || initial?.id || '');
        if (!jobId) {
          throw new Error('Vehicle routing job did not return a job id.');
        }
        const statusUrl = `/api/vehicle-route/status/${encodeURIComponent(jobId)}`;
        const resultUrl = `/api/vehicle-route/result/${encodeURIComponent(jobId)}`;

        const pollIntervalMs = 5000;
        const resultIntervalMs = 2000;
        const maxWaitMs = 45 * 60 * 1000; // 45 minutes upper bound for large datasets
        const started = Date.now();
        let attempts = 0;

        while (Date.now() - started < maxWaitMs) {
          attempts += 1;
          const statusResp = await fetch(statusUrl, { cache: 'no-store' });
          if (!statusResp.ok) {
            let errMsg = `Failed to check job status (HTTP ${statusResp.status})`;
            try {
              const err = await statusResp.json();
              if (err?.error) errMsg = String(err.error);
            } catch {}
            throw new Error(errMsg);
          }
          const statusJson = await statusResp.json();
          const state = String(statusJson?.state || '').toUpperCase();
          if (state === 'FAILURE') {
            throw new Error(statusJson?.error || 'Vehicle routing job failed');
          }
          if (statusJson?.ready && state === 'SUCCESS') {
            break;
          }

          const elapsedMs = Date.now() - started;
          const elapsedMin = Math.floor(elapsedMs / 60000);
          const elapsedSec = Math.floor((elapsedMs % 60000) / 1000);
          const elapsedLabel = elapsedMin > 0 ? `${elapsedMin}m ${elapsedSec.toString().padStart(2, '0')}s` : `${elapsedSec}s`;

          setStatus(`Routing job ${state || 'PENDING'} (waiting ${elapsedLabel})...`);
          setProgress(prev => {
            const baseline = 45;
            const scaled = baseline + Math.min(40, Math.floor(elapsedMs / 1000));
            return Math.min(92, Math.max(prev, scaled));
          });
          await sleep(pollIntervalMs);
        }

        if (Date.now() - started >= maxWaitMs) {
          throw new Error(`Timed out after ${Math.floor(maxWaitMs / 60000)} minutes waiting for routing job.`);
        }

        const resultDeadline = Date.now() + 5 * 60 * 1000; // allow up to 5 minutes for result fetch
        while (Date.now() < resultDeadline) {
          const resultResp = await fetch(resultUrl, { cache: 'no-store' });
          if (resultResp.status === 202) {
            await sleep(resultIntervalMs);
            continue;
          }
          if (!resultResp.ok) {
            let errMsg = `Failed to fetch routing result (HTTP ${resultResp.status})`;
            try {
              const err = await resultResp.json();
              if (err?.error) errMsg = String(err.error);
            } catch {}
            throw new Error(errMsg);
          }
          const resultJson = await resultResp.json();
          return resultJson?.result ?? resultJson;
        }
        throw new Error('Timed out waiting for routing result.');
      };

      if (mode === 'route' && data && data.job_id && !Array.isArray(data.rows)) {
        data = await awaitRoutingJob(data);
      }

      setRows(data.rows);
      setOriginalRows(data.rows); // Save original for rerun
  if (data.suggested_locations) setCenters(data.suggested_locations);
      // Aggregate KPIs if provided (route mode)
      if (Array.isArray(data.route_kpis)) {
        const totalKm = data.route_kpis.reduce((s:number, r:any) => s + (Number(r.total_km) || 0), 0);
        const totalMin = data.route_kpis.reduce((s:number, r:any) => s + (Number(r.drive_minutes) || 0) + (Number(r.service_minutes) || 0), 0);
        setTotalMiles(Number.isFinite(totalKm) ? Math.round(totalKm * 0.621371) : null);
        setTotalMinutes(Number.isFinite(totalMin) ? Math.round(totalMin) : null);
      }
  // Clear filters/sorts so fresh results render
  setFilterDay(null);
  setSortCol(null);
  setSortAsc(true);

      // Handle failed postcodes
      if (data.failed_count && data.failed_count > 0) {
        setFailedCount(data.failed_count);
        setFailedMessage(data.message || `${data.failed_count} postcode(s) failed to locate.`);
        if (data.failed_excel_b64) {
          const byteChars = atob(data.failed_excel_b64);
          const byteNumbers = new Array(byteChars.length);
          for (let i = 0; i < byteChars.length; i++) {
            byteNumbers[i] = byteChars.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], {type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"});
          const url = URL.createObjectURL(blob);
          setFailedExcelUrl(url);
        }
      }

      // Set backend stats
  if (data.total_miles !== undefined) setTotalMiles(data.total_miles);
  if (data.total_minutes !== undefined) setTotalMinutes(data.total_minutes);

      setProgress(100);
      let finalMsg = data.message ? String(data.message) : (mode === 'route' ? 'Vehicle routing complete!' : 'Clustering complete!');
      if (typeof data.unscheduled_count === 'number' && data.unscheduled_count > 0) {
        finalMsg += ` (${data.unscheduled_count} unscheduled)`;
      }
      setStatus(finalMsg);
      // save project
      const p:Project = { name, rows:data.rows };
      save([...projects, p]);
      setCurrent(p);
      setSidebar('stats');
    } catch(err:any) {
      setProgress(0);
      setStatus('');
      console.error(err);
      alert((mode === 'route' ? 'Routing failed: ' : 'Clustering failed: ') + err.message);
    }
  };

  // Quick API ping for debugging
  const pingApi = async () => {
    try {
  const envBase = (process.env.NEXT_PUBLIC_API_URL || '').trim();
  const isBrowser = typeof window !== 'undefined';
  const isLocalhost = isBrowser && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
  const origin = isBrowser ? `${window.location.protocol}//${window.location.host}` : '';
  const apiBase = envBase || (isLocalhost ? 'http://localhost:8000' : origin);
  const res = await fetch(`${apiBase}/healthz`, { method: 'GET' });
      setApiPing(res.ok ? 'OK' : `HTTP ${res.status}`);
    } catch (e:any) {
      setApiPing(e?.message || 'Error');
    }
  };

  const pingOsrm = async () => {
    try {
  const envBase = (process.env.NEXT_PUBLIC_API_URL || '').trim();
  const isBrowser = typeof window !== 'undefined';
  const isLocalhost = isBrowser && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
  const origin = isBrowser ? `${window.location.protocol}//${window.location.host}` : '';
  const apiBase = envBase || (isLocalhost ? 'http://localhost:8000' : origin);
  const nextApiBase = origin || '';
      const res = await fetch(`${apiBase}/healthz/osrm`, { method: 'GET' });
      if (!res.ok) {
        setApiOsrm(`HTTP ${res.status}`);
        return;
      }
      const j = await res.json();
      setApiOsrm(j.ok ? `OK (${j.osrm_url}${j.distance_sample_m ? ", sample " + j.distance_sample_m.toFixed(0) + "m" : ''})` : `Error: ${j.error || 'unknown'}`);
    } catch (e:any) {
      setApiOsrm(e?.message || 'Error');
    }
  };

  const openProject = (p:Project) => {
    // Reset map and sidebar state when opening a project
    setCurrent(p);
    setRows(p.rows);
    setOriginalRows(p.rows);
    setSidebar('stats');
    setMarkers([]);
    setMapKey(k => k + 1); // force map resize
  };

  const deleteProject = (idx:number, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering openProject
    const ps = projects.slice();
    ps.splice(idx,1);
    save(ps);
    setProjects(ps);
    if (current && projects[idx].name === current.name) {
      setCurrent(null);
      setRows([]);
      setOriginalRows([]);
      setSidebar('none');
    }
  };

  // Ask AI over current results (auto: try transform, else answer)
  const askAI = async () => {
    const q = aiQuestion.trim();
    if (!q) return;
    setAiLoading(true);
    setAiAnswer('');
    try {
      const envBase = (process.env.NEXT_PUBLIC_API_URL || '').trim();
      const isBrowser = typeof window !== 'undefined';
      const isLocalhost = isBrowser && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
      const origin = isBrowser ? `${window.location.protocol}//${window.location.host}` : '';
      const apiBase = envBase || (isLocalhost ? 'http://localhost:8000' : origin);
  const nextApiBase = origin;
      // First try transform
      // Send filtered subset of rows if day/week/calls-per-day filters are active
      const rowsForAI = (() => {
        if (!rows.length) return rows;
        let data = rows.slice(1);
        const hdr = rows[0] || [];
        const dayNumIdx = hdr.findIndex(h => /(dayofweeknum|day[\s_]*number)/i.test(String(h)));
        const dayIdx = hdr.findIndex(h => /dayofweek/i.test(h) || h.toLowerCase().includes('day'));
        const weekIdx = hdr.findIndex(h => /^week$/i.test(String(h)));
        const resIdx = hdr.findIndex(h => /resource/i.test(String(h)));
        if (filterDay !== null) {
          data = data.filter(r => {
            if (dayNumIdx !== -1) return Number(r[dayNumIdx]) === Number(filterDay);
            const m = String(r[dayIdx] ?? '').match(/(\d+)/);
            const n = m ? parseInt(m[1]) : NaN;
            return Number(n) === Number(filterDay);
          });
        }
        if (filterWeek !== null && weekIdx !== -1) {
          data = data.filter(r => String(r[weekIdx] ?? '') === String(filterWeek));
        }
        if (filterCallsPerDay !== null) {
          const dIdx = dayIdx;
          const tIdx = resIdx !== -1 ? resIdx : hdr.findIndex(h => h.toLowerCase().includes('territory'));
          const wIdx = weekIdx;
          const counts: Record<string, number> = {};
          rows.slice(1).forEach(rr => {
            const d = dIdx !== -1 ? rr[dIdx] : '';
            const t = tIdx !== -1 ? (rr[tIdx] || '') : '';
            const w = wIdx !== -1 ? (rr[wIdx] || '') : '';
            if (!d) return;
            const key = (wIdx !== -1 && tIdx !== -1) ? `${t}||${w}||${d}` : `${t}||${d}`;
            counts[key] = (counts[key] || 0) + 1;
          });
          data = data.filter(rr => {
            const d = dIdx !== -1 ? rr[dIdx] : '';
            const t = tIdx !== -1 ? (rr[tIdx] || '') : '';
            const w = wIdx !== -1 ? (rr[wIdx] || '') : '';
            const key = (wIdx !== -1 && tIdx !== -1) ? `${t}||${w}||${d}` : `${t}||${d}`;
            return (counts[key] || 0) === Number(filterCallsPerDay);
          });
        }
        return [hdr, ...data];
      })();
  const tRes = await fetch(`${nextApiBase}/api/ask/transform`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction: q, rows: rowsForAI })
      });
      if (tRes.ok) {
        const tj = await tRes.json();
        const plan = tj?.plan;
        if (plan && (Array.isArray(plan.updates) && plan.updates.length > 0)) {
          // Validate and apply
          if (!rows.length) throw new Error('No data loaded.');
          const hdr = rows[0];
          const allCols = new Set(hdr);
          const bad = [
            ...((plan.filters||[]).filter((f:any)=>!allCols.has(f.column))),
            ...((plan.updates||[]).filter((u:any)=>!allCols.has(u.column)))
          ];
          if (bad.length === 0) {
            const idxMap: Record<string, number> = {};
            hdr.forEach((h, i) => { idxMap[h] = i; });
            const data = rows.slice(1).map(r => r.slice());
            const andFilters = (plan.filters||[]) as any[];
            const updates = (plan.updates||[]) as any[];
            const matchRow = (r: string[]) => {
              if (!andFilters.length) return true;
              for (const f of andFilters) {
                const j = idxMap[f.column];
                const cell = String(r[j] ?? '');
                const ci = (f.caseInsensitive !== false);
                const a = ci ? cell.toLowerCase() : cell;
                const val = String(f.value ?? '');
                const b = ci ? val.toLowerCase() : val;
                const op = String(f.op || 'equals').toLowerCase();
                let ok = false;
                if (op === 'equals') ok = (a === b);
                else if (op === 'startswith') ok = a.startsWith(b);
                else if (op === 'contains') ok = a.includes(b);
                else if (op === 'regex') { try { ok = new RegExp(String(f.value)).test(cell); } catch { ok = false; } }
                if (!ok) return false;
              }
              return true;
            };
            let changed = 0;
            for (let i=0;i<data.length;i++) {
              const r = data[i];
              if (!matchRow(r)) continue;
              for (const u of updates) {
                const j = idxMap[u.column];
                r[j] = String(u.setTo ?? '');
              }
              changed++;
            }
            if (changed > 0) {
              const newRows = [hdr.slice(), ...data];
              setTfUndoRows(rows);
              setRows(newRows);
              setAiHistory(h => [{ q, a: `Applied to ${changed} row(s).`, provider: tj?.provider }, ...h].slice(0, 10));
              setAiAnswer(`Applied to ${changed} row(s).`);
              setAiLoading(false);
              return;
            }
          }
        }
      }
      // Fallback to Q&A
  const res = await fetch(`${nextApiBase}/api/ask`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q, rows: rowsForAI }) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = await res.json();
      const ans = (j && (j.answer || j.text || j.message)) ? (j.answer || j.text || j.message) : 'No answer.';
      setAiAnswer(ans);
      setAiHistory(h => [{ q, a: ans, provider: j?.provider, used_rows: j?.used_rows }, ...h].slice(0, 10));
    } catch (e:any) {
      setAiAnswer(e?.message || 'Failed.');
    } finally {
      setAiLoading(false);
    }
  };

  // MAIN PAGE RETURN

  // Memoize the Results Table block so typing in AI doesn't rebuild the whole table
  const columnOrder = rows.length > 0 ? [...Array(rows[0].length).keys()] : [];
  const resultsTableEl = useMemo(() => {
    return (
      <div style={{position: 'fixed', left: aiOpen ? 680 : 340, right: 0, bottom: 0, zIndex: 1000, background: 'var(--dark-2)', boxShadow: '0 -2px 8px rgba(0,0,0,0.12)', borderTop: '1px solid #222', transition: 'height 0.2s', height: tableOpen ? '22vh' : '32px', minHeight: 0, display: 'flex', flexDirection: 'column'}}>
        <div style={{display: 'flex', alignItems: 'center', padding: '0 12px', height: 32, borderBottom: tableOpen ? '1px solid #444' : 'none'}}>
          <button
            style={{background: 'none', border: 'none', color: '#fff', fontSize: 18, cursor: 'pointer', marginRight: 8, transform: tableOpen ? 'rotate(0deg)' : 'rotate(-90deg)', transition: 'transform 0.2s'}}
            onClick={() => setTableOpen(o => !o)}
            aria-label={tableOpen ? 'Collapse Table' : 'Expand Table'}
          >{tableOpen ? '▼' : '▲'}</button>
          <span style={{fontWeight: 600, fontSize: 15}}>Results Table</span>
          {Object.values(colFilters).some(v => v && v.length > 0) && (
            <span style={{marginLeft: 12, color: '#bbb', fontSize: 12}}>Filters active</span>
          )}
          <button
            style={{marginLeft: 'auto', marginRight: 8, background: '#444', color: '#fff', border: 'none', borderRadius: 6, padding: '4px 10px', fontWeight: 600, fontSize: 12}}
            onClick={() => { setFilterDay(null); setFilterWeek(null); setFilterCallsPerDay(null); setColFilters({}); }}
            title="Clear all filters"
          >Clear filters</button>
          {/* Removed: legacy buttons for Re-optimise Day, Re-optimise Territories, Normalize Rows */}
          <button
            style={{background: '#3cb44b', color: '#fff', border: 'none', borderRadius: 6, padding: '4px 12px', fontWeight: 600, fontSize: 13, cursor: 'pointer'}}
            onClick={() => {
              let data = rows.slice(1);
              // Apply column filters (AND)
              if (Object.keys(colFilters).length) {
                const filters = colFilters;
                const hdr = rows[0] || [];
                data = data.filter(r => {
                  for (const [kStr, val] of Object.entries(filters)) {
                    if (!val) continue;
                    const k = Number(kStr);
                    const hLower = String(hdr[k] ?? '').toLowerCase();
                    const valStr = String(val);
                    // Exact numeric compare for DayNumber/Day columns
                    if (/(day[\s_]*number|dayofweeknum)/.test(hLower)) {
                      if (Number(r[k]) !== Number(valStr)) return false;
                    } else if (hLower === 'day' || (hLower.includes('day') && !/dayofweek|weekday|day[\s_]*number|callsperday/.test(hLower))) {
                      const m = String(r[k] ?? '').match(/(\d+)/);
                      const n = m ? parseInt(m[1]) : NaN;
                      if (n !== Number(valStr)) return false;
                    } else {
                      const cell = String(r[k] ?? '').toLowerCase();
                      if (!cell.includes(valStr.toLowerCase())) return false;
                    }
                  }
                  return true;
                });
              }
              if (sortCol !== null) {
                data = [...data].sort((a, b) => {
                  const av = a[sortCol] ?? '';
                  const bv = b[sortCol] ?? '';
                  if (!isNaN(Number(av)) && !isNaN(Number(bv))) {
                    return sortAsc ? Number(av) - Number(bv) : Number(bv) - Number(av);
                  }
                  return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
                });
              }
              // Keep legacy day filter for convenience (AND with other filters)
              const dayIdx = rows[0]?.findIndex(h => /dayofweek/i.test(h) || h.toLowerCase().includes('group') || h.toLowerCase().includes('day'));
              const dayNumIdx = rows[0]?.findIndex(h => /(dayofweeknum|day[\s_]*number)/i.test(String(h)));
              if (filterDay !== null) {
                data = data.filter(r => {
                  if (dayNumIdx !== -1) return Number(r[dayNumIdx]) === Number(filterDay);
                  if (dayIdx !== -1) {
                    const day = r[dayIdx];
                    return day && String(day).toLowerCase().includes(String(filterDay));
                  }
                  return true;
                });
              }
              const getDayNum = (val: string) => {
                if (typeof val !== 'string') val = String(val ?? '');
                const m = val.match(/(\d+)/);
                return m ? parseInt(m[1]) : NaN;
              };
              let exportHeader = rows.length > 0 ? columnOrder.map(i => rows[0][i]) : [];
              exportHeader = [...exportHeader, 'Sort'];
              let exportRows: (string | number)[][] = [];
              if (rows.length > 1) {
                exportRows = data.map(r => {
                  const dayIdx = rows[0]?.findIndex(h => /dayofweek/i.test(h) || h.toLowerCase().includes('group') || h.toLowerCase().includes('day'));
                  const dayNumIdx = rows[0]?.findIndex(h => /dayofweeknum/i.test(h));
                  const dayVal = dayIdx !== -1 ? r[dayIdx] : '';
                  const sortVal = dayNumIdx !== -1 ? Number(r[dayNumIdx]) : (!dayVal || dayVal === '' ? '' : getDayNum(dayVal));
                  return [...columnOrder.map(i => r[i]), sortVal];
                });
              }
              exportRows = [exportHeader, ...exportRows];
              const ws = XLSX.utils.aoa_to_sheet(exportRows);
              const wb = XLSX.utils.book_new();
              XLSX.utils.book_append_sheet(wb, ws, 'Results');
              XLSX.writeFile(wb, 'results.xlsx');
            }}
          >Export</button>
        </div>
        {tableOpen && (
          <div style={{overflowX: 'auto', overflowY: 'auto', flex: 1, padding: '0 12px', background: 'var(--dark-2)'}}>
            <table style={{width: '100%', borderCollapse: 'collapse', fontSize: 12, background: 'transparent'}}>
              <thead>
                <tr>
                  {rows.length > 0 && rows[0] ? (
                    columnOrder.map((idx) => {
                      const h = rows[0][idx];
                      return (
                        <th
                          key={idx}
                          style={{color: '#fff', fontWeight: 700, border: '1px solid #444', padding: '4px 6px', background: '#111', cursor: 'pointer', userSelect: 'none'}}
                          onClick={() => {
                            setSortCol(idx);
                            setSortAsc(sortCol === idx ? !sortAsc : true);
                          }}
                        >
                          {h}
                          {sortCol === idx && (sortAsc ? ' ▲' : ' ▼')}
                        </th>
                      );
                    })
                  ) : (
                    <th style={{color:'#fff', fontWeight:700, border:'1px solid #444', padding:'4px 6px', background:'#111'}}>No columns</th>
                  )}
                </tr>
                {/* Filter row */}
              </thead>
              <tbody>
                {rows.length > 1 ? (() => {
                  let data = rows.slice(1);
                  // Apply column filters (AND)
                  if (Object.keys(colFilters).length) {
                    const filters = colFilters;
                    const hdr = rows[0] || [];
                    data = data.filter(r => {
                      for (const [kStr, val] of Object.entries(filters)) {
                        if (!val) continue;
                        const k = Number(kStr);
                        const hLower = String(hdr[k] ?? '').toLowerCase();
                        const valStr = String(val);
                        if (/(day[\s_]*number|dayofweeknum)/.test(hLower)) {
                          if (Number(r[k]) !== Number(valStr)) return false;
                        } else if (hLower === 'day' || (hLower.includes('day') && !/dayofweek|weekday|day[\s_]*number|callsperday/.test(hLower))) {
                          const m = String(r[k] ?? '').match(/(\d+)/);
                          const n = m ? parseInt(m[1]) : NaN;
                          if (n !== Number(valStr)) return false;
                        } else {
                          const cell = String(r[k] ?? '').toLowerCase();
                          if (!cell.includes(valStr.toLowerCase())) return false;
                        }
                      }
                      return true;
                    });
                  }
                  if (sortCol !== null) {
                    data = [...data].sort((a, b) => {
                      const av = a[sortCol] ?? '';
                      const bv = b[sortCol] ?? '';
                      if (!isNaN(Number(av)) && !isNaN(Number(bv))) {
                        return sortAsc ? Number(av) - Number(bv) : Number(bv) - Number(av);
                      }
                      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
                    });
                  }
                  const dayIdx = rows[0]?.findIndex(h => /dayofweek/i.test(h) || h.toLowerCase().includes('group') || h.toLowerCase().includes('day'));
                  const dayNumberIdx = rows[0]?.findIndex(h => /day[\s_]*number/i.test(String(h)));
                  const dayOfWeekNumIdx = rows[0]?.findIndex(h => /dayofweeknum/i.test(String(h)));
                  const dayNumIdx = (dayNumberIdx !== -1 && dayNumberIdx !== undefined) ? dayNumberIdx : dayOfWeekNumIdx;
                  const weekIdxFilter = rows[0]?.findIndex(h => /^week$/i.test(String(h)));
                  if (filterDay !== null) {
                    data = data.filter(r => {
                      if (dayNumIdx !== -1) return Number(r[dayNumIdx]) === Number(filterDay);
                      if (dayIdx !== -1) {
                        const m = String(r[dayIdx] ?? '').match(/(\d+)/);
                        const n = m ? parseInt(m[1]) : NaN;
                        return Number(n) === Number(filterDay);
                      }
                      return true;
                    });
                  }
                  if (filterWeek !== null && weekIdxFilter !== -1) {
                    data = data.filter(r => String(r[weekIdxFilter] ?? '') === String(filterWeek));
                  }
                  if (filterCallsPerDay !== null) {
                    // Recompute counts for the filtered dataset context (by territory/week/day key)
                    const hdrLocal = rows[0];
                    const isVRPLocal = hdrLocal.some(h => /dayofweek/i.test(h)) && hdrLocal.some(h => /^week$/i.test(h)) && hdrLocal.some(h => /resource/i.test(h));
                    const dIdx = hdrLocal.findIndex(h => /dayofweek/i.test(h) || h.toLowerCase().includes('group') || h.toLowerCase().includes('day'));
                    const tIdx = isVRPLocal ? hdrLocal.findIndex(h => /resource/i.test(h)) : hdrLocal.findIndex(h => h.toLowerCase().includes('territory'));
                    const wIdx = isVRPLocal ? hdrLocal.findIndex(h => /^week$/i.test(h)) : -1;
                    const counts: Record<string, number> = {};
                    rows.slice(1).forEach(rr => {
                      const d = dIdx !== -1 ? rr[dIdx] : '';
                      const t = tIdx !== -1 ? (rr[tIdx] || '') : '';
                      const w = wIdx !== -1 ? (rr[wIdx] || '') : '';
                      if (!d) return;
                      const key = isVRPLocal ? `${t}||${w}||${d}` : `${t}||${d}`;
                      counts[key] = (counts[key] || 0) + 1;
                    });
                    data = data.filter(rr => {
                      const d = dIdx !== -1 ? rr[dIdx] : '';
                      const t = tIdx !== -1 ? (rr[tIdx] || '') : '';
                      const w = wIdx !== -1 ? (rr[wIdx] || '') : '';
                      if (!d) return false;
                      const key = isVRPLocal ? `${t}||${w}||${d}` : `${t}||${d}`;
                      return (counts[key] || 0) === Number(filterCallsPerDay);
                    });
                  }
                  if (data.length === 0) {
                    return (
                      <tr><td colSpan={columnOrder.length || 1} style={{color:'#fff', textAlign:'center', padding:'16px', background:'transparent'}}>No results</td></tr>
                    );
                  }
                  const hdr = rows[0];
                  const isVRP = hdr.some(h => /dayofweek/i.test(h)) && hdr.some(h => /^week$/i.test(h)) && hdr.some(h => /resource/i.test(h));
                  const dayIdx2 = hdr.findIndex(h => /dayofweek/i.test(h) || h.toLowerCase().includes('group') || h.toLowerCase().includes('day'));
                  const dayNumberIdx2 = hdr.findIndex(h => /day[\s_]*number/i.test(String(h)));
                  const dayOfWeekNumIdx2 = hdr.findIndex(h => /dayofweeknum/i.test(h));
                  const preferredDayNumIdx2 = dayNumberIdx2 !== -1 ? dayNumberIdx2 : dayOfWeekNumIdx2;
                  const terrIdx = isVRP ? hdr.findIndex(h => /resource/i.test(h)) : hdr.findIndex(h => h.toLowerCase().includes('territory'));
                  const weekIdx = isVRP ? hdr.findIndex(h => /^week$/i.test(h)) : -1;
                  const countsMap: Record<string, number> = {};
                  rows.slice(1).forEach(rr => {
                    const d = dayIdx2 !== -1 ? rr[dayIdx2] : '';
                    const t = terrIdx !== -1 ? (rr[terrIdx] || '') : '';
                    const w = weekIdx !== -1 ? (rr[weekIdx] || '') : '';
                    if (!d) return;
                    const key = isVRP ? `${t}||${w}||${d}` : `${t}||${d}`;
                    countsMap[key] = (countsMap[key] || 0) + 1;
                  });

                  return data.map((r, i) => (
                    <tr key={i} style={{background:'transparent'}}>
                      {columnOrder.map((j) => {
                        let cell = r[j];
                        if (cell === undefined || cell === null || cell === '' || (typeof cell === 'string' && cell.trim() === '') || (typeof cell === 'number' && isNaN(cell)) || (typeof cell === 'string' && cell.toLowerCase() === 'nan')) {
                          cell = 'N/A';
                        }
                        const hLower = String(rows[0][j] ?? '').toLowerCase();
                        if (/dayofweek/i.test(rows[0][j]) || hLower.includes('group') || (hLower.includes('day') && !/day[\s_]*number|callsperday/.test(hLower))) {
                          const dayLabel = cell || '';
                          if (dayLabel) {
                            const terrLabel = terrIdx !== -1 ? (r[terrIdx] || '') : '';
                            const weekLabel = weekIdx !== -1 ? (r[weekIdx] || '') : '';
                            const key = isVRP ? `${terrLabel}||${weekLabel}||${dayLabel}` : `${terrLabel}||${dayLabel}`;
                            const c = countsMap[key] ?? undefined;
                            if (typeof c === 'number') {
                              cell = `${dayLabel} (${c} calls)`;
                            }
                          }
                          return (
                            <td
                              key={j}
                              style={{color: filterDay !== null && cell && (String(cell).includes(String(filterDay))) ? '#3cb44b' : '#fff', border: '1px solid #444', padding: '4px 6px', background: 'transparent', cursor: 'pointer', userSelect: 'none', fontWeight: filterDay !== null && cell && (String(cell).includes(String(filterDay))) ? 700 : 400}}
                              onClick={() => {
                                // Determine day number from raw row value (not decorated label)
                                const labelRaw = (preferredDayNumIdx2 !== -1 ? String(r[preferredDayNumIdx2]) : String(r[dayIdx2] ?? ''));
                                let nextNum: number = Number.NaN;
                                if (preferredDayNumIdx2 !== -1 && r[preferredDayNumIdx2] !== undefined) {
                                  nextNum = Number(r[preferredDayNumIdx2]);
                                } else {
                                  const m = String(labelRaw).match(/(\d+)/);
                                  if (m) nextNum = parseInt(m[1]);
                                  if (!isFinite(nextNum)) {
                                    const k3 = labelRaw.trim().slice(0,3).toLowerCase();
                                    const nameToNum: Record<string, number> = {mon:1,tue:2,wed:3,thu:4,fri:5,sat:6,sun:7};
                                    if (nameToNum[k3]) nextNum = nameToNum[k3];
                                  }
                                }
                                const newVal = filterDay === nextNum ? null : nextNum;
                                setFilterDay(newVal);
                                // Also set a column filter on the displayed day column to enable multi-column AND filtering
                                setColFilters(prev => {
                                  const copy = { ...prev } as Record<number,string>;
                                  if (newVal === null) {
                                    delete copy[j];
                                  } else {
                                    copy[j] = String(newVal);
                                  }
                                  return copy;
                                });
                              }}
                            >
                              {cell}
                            </td>
                          );
                        }
                        if (/(dayofweeknum|day[\s_]*number)/i.test(rows[0][j])) {
                          return (
                            <td
                              key={j}
                              style={{color: filterDay !== null && Number(filterDay) === Number(cell) ? '#3cb44b' : '#fff', border: '1px solid #444', padding: '4px 6px', background: 'transparent', cursor: 'pointer'}}
                              onClick={() => {
                                const val = Number(r[j]);
                                setFilterDay(prev => (prev !== null && Number(prev) === val) ? null : val);
                                setColFilters(prev => {
                                  const copy = { ...prev } as Record<number,string>;
                                  if (Number(prev[j] ?? 'NaN') === val) delete copy[j]; else copy[j] = String(val);
                                  return copy;
                                });
                              }}
                            >{cell}</td>
                          );
                        }
                        if (/^week$/i.test(rows[0][j])) {
                          return (
                            <td
                              key={j}
                              style={{color: filterWeek !== null && String(filterWeek) === String(cell) ? '#3cb44b' : '#fff', border: '1px solid #444', padding: '4px 6px', background: 'transparent', cursor: 'pointer'}}
                              onClick={() => {
                                const wk = String(r[j] ?? '');
                                setFilterWeek(prev => (prev !== null && String(prev) === wk) ? null : wk);
                                setColFilters(prev => {
                                  const copy = { ...prev } as Record<number,string>;
                                  if (String(prev[j] ?? '') === wk) delete copy[j]; else copy[j] = wk;
                                  return copy;
                                });
                              }}
                            >{cell}</td>
                          );
                        }
                        if (/callsperday/i.test(hLower)) {
                          return (
                            <td
                              key={j}
                              style={{color: filterCallsPerDay !== null && Number(filterCallsPerDay) === Number(cell) ? '#3cb44b' : '#fff', border: '1px solid #444', padding: '4px 6px', background: 'transparent', cursor: 'pointer'}}
                              onClick={() => {
                                const val = Number(r[j]);
                                setFilterCallsPerDay(prev => (prev !== null && Number(prev) === val) ? null : val);
                              }}
                            >{cell}</td>
                          );
                        }
                        return (
                          <td key={j} style={{color: '#fff', border: '1px solid #444', padding: '4px 6px', background: 'transparent'}}>{cell}</td>
                        );
                      })}
                    </tr>
                  ));
                })() : (
                  <tr><td style={{color:'#fff', textAlign:'center', padding:'32px', fontSize:'18px', fontWeight:700, background:'transparent'}}>No results to display. Run clustering or open a project.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  }, [aiOpen, tableOpen, rows, sortCol, sortAsc, filterDay]);

  // Sidebar renderer (Stats/New/Open/Project/Default)
  const renderSidebarContent = () => {
    const stats = getSummaryStats?.();
    if (sidebar === 'stats') {
      return (
        <div className="sidebar-inner">
          <div className="sidebar-title">Results</div>
          <div className="card" style={{display:'flex', gap:10, flexWrap:'wrap'}}>
            <button className="btn-primary" onClick={async ()=>{ if (file) { await run(); } else { alert('No file available for rerun. Please upload again.'); } }}>{mode === 'route' ? 'Rerun Routing' : 'Rerun Clustering'}</button>
            <button className="btn-secondary" onClick={() => setSidebar('none')}>Back</button>
          </div>
          <div className="card">
            <div style={{height: 8, background: '#222', borderRadius: 4, overflow: 'hidden', marginBottom: 6}}>
              <div style={{width: `${progress}%`, height: '100%', background: '#3cb44b', transition: 'width 0.3s'}} />
            </div>
            <div style={{color: '#fff', fontSize: 13}}>{status}</div>
            {failedMessage && (
              <div style={{color: '#e6194b', fontSize: 13, marginTop: 6}}>
                {failedMessage}
                {failedExcelUrl && (
                  <a href={failedExcelUrl} download="failed_postcodes.xlsx" style={{color:'#fff',marginLeft:8}}>Download failed</a>
                )}
              </div>
            )}
          </div>
          <div className="card">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 15, background: 'transparent', marginTop: 8 }}>
              <tbody>
                <tr>
                  <td style={{color: '#fff', fontWeight: 600, border: '1px solid #fff', padding: '8px 12px', background: 'transparent'}}>Total calls</td>
                  <td style={{color: '#fff', border: '1px solid #fff', padding: '8px 12px', background: 'transparent', textAlign: 'right'}}>{(stats as any)?.mapped}</td>
                </tr>
                <tr>
                  <td style={{color: '#fff', fontWeight: 600, border: '1px solid #fff', padding: '8px 12px', background: 'transparent'}}>Failed calls</td>
                  <td style={{color: '#fff', border: '1px solid #fff', padding: '8px 12px', background: 'transparent', textAlign: 'right'}}>{(stats as any)?.failed}</td>
                </tr>
                <tr>
                  <td style={{color: '#fff', fontWeight: 600, border: '1px solid #fff', padding: '8px 12px', background: 'transparent'}}>Total days</td>
                  <td style={{color: '#fff', border: '1px solid #fff', padding: '8px 12px', background: 'transparent', textAlign: 'right'}}>{(stats as any)?.days}</td>
                </tr>
                {Object.entries(((stats as any)?.callsCountSummary || {})).sort((a, b) => Number(a[0]) - Number(b[0])).map(([calls, numDays]) => (
                  <tr key={calls}>
                    <td style={{color: '#fff', fontWeight: 600, border: '1px solid #fff', padding: '8px 12px', background: 'transparent'}}>{calls} calls/day</td>
                    <td style={{color: '#fff', border: '1px solid #fff', padding: '8px 12px', background: 'transparent', textAlign: 'right'}}>{numDays as any}</td>
                  </tr>
                ))}
                <tr>
                  <td style={{color: '#fff', fontWeight: 600, border: '1px solid #fff', padding: '8px 12px', background: 'transparent'}}>Total Miles</td>
                  <td style={{color: '#fff', border: '1px solid #fff', padding: '8px 12px', background: 'transparent', textAlign: 'right'}}>{(stats as any)?.totalMiles}</td>
                </tr>
                <tr>
                  <td style={{color: '#fff', fontWeight: 600, border: '1px solid #fff', padding: '8px 12px', background: 'transparent'}}>Total Minutes</td>
                  <td style={{color: '#fff', border: '1px solid #fff', padding: '8px 12px', background: 'transparent', textAlign: 'right'}}>{(stats as any)?.totalMinutes}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      );
    }
    if (sidebar === 'new') {
      return (
        <div className="sidebar-inner">
          <div className="card">
            <h2>New Project</h2>
            <label style={{display:'block', fontWeight:600, marginBottom:6}}>Project Name <Help text="A label for this run; used for saving and reopening later." /></label>
            <input type="text" placeholder="Project Name" value={name} onChange={e => setName(e.target.value)} style={{marginBottom: 12, padding: 8, borderRadius: 8, border: '1px solid rgba(255,255,255,0.25)', width: '100%', background:'#0e1a24', color:'#fff'}} />
          </div>
          <div className="card" style={{display:'flex', gap:12, alignItems:'center'}}>
            <label style={{fontWeight:600}}>Mode <Help text="Cluster Only: group into days within existing territories. Plan Territories: split calls into territories and then days. Vehicle Routing: route calls to resources by working days and shifts." /></label>
            <select value={mode} onChange={e=>setMode(e.target.value as any)}>
              <option value="cluster">Cluster Only</option>
              <option value="plan">Plan Territories</option>
              <option value="route">Vehicle Routing</option>
            </select>
          </div>
          {mode !== 'route' && (
            <div className="card">
              <label style={{display:'block', fontWeight:600, marginBottom:6}}>Calls Excel <Help text="Upload your calls (.xlsx/.xls). Must contain a postcode column (e.g., Postcode/Eircode)." /></label>
              <input type="file" onChange={handleFileChange} accept=".xlsx,.xls" style={{marginBottom: 12}} />
            </div>
          )}
          {/* Removed deprecated territory/group selector for non-route modes */}
          {mode !== 'route' && (
            <>
              <div className="card">
                <label>Min Calls <Help text="Minimum calls per day/route (soft target used during rebalancing)." /> </label>
                <input type="number" value={minC} onChange={e => setMinC(Number(e.target.value))} style={{width: 60}} />
              </div>
              <div className="card">
                <label>Max Calls <Help text="Maximum calls per day/route (capacity limit)." /> </label>
                <input type="number" value={maxC} onChange={e => setMaxC(Number(e.target.value))} style={{width: 60}} />
              </div>
            </>
          )}
          {mode === 'plan' && (
            <div className="card" style={{display:'flex', flexDirection:'column', gap:12}}>
              <div>
                <label>Number of Territories <Help text="How many territories to create. If a Resources Excel is provided, the number of rows there determines territories (this value is then ignored)." /> </label>
                <input type="number" value={numTerritories} onChange={e=>setNumTerritories(Number(e.target.value))} style={{width:80}} />
              </div>
              <div>
                <label>Resources Excel (optional) <Help text="Upload centers/depots (.xlsx/.xls). Supported columns: name, latitude/lat, longitude/lon/lng, postcode/eircode. If lat/lng missing, postcode will be geocoded. Territories will be named from the 'name' column." /> </label>
                <input type="file" accept=".xlsx,.xls" onChange={handleResourcesFileChange} />
              </div>
              <div>
                <label>Resource Locations (one per line) <Help text="Only used if Resources Excel is not provided. Accepted formats per line: 'postcode' OR 'lat,lng' OR 'Name,postcode' OR 'Name,lat,lng'." /></label>
                <textarea
                  placeholder="postcode\nlat,lng\nName,postcode\nName,lat,lng"
                  value={resourceText}
                  onChange={e=>setResourceText(e.target.value)}
                  style={{width:'100%', minHeight:100}}
                />
              </div>
            </div>
          )}
          {mode === 'route' && (
            <>
              <div className="card" style={{display:'flex', flexDirection:'column', gap:12}}>
                <div style={{fontWeight:600}}>Calls (required)</div>
                <div>
                  <label>Calls Excel <Help text="Columns required: Name/ID, Postcode, Duration." /></label>
                  <input type="file" onChange={handleFileChange} accept=".xlsx,.xls" />
                </div>
        <div style={{display:'grid', gridTemplateColumns:'1fr', gap:8}}>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>ID/Name *</label>
                    {callHeaders.length ? (
          <select value={mapCallName} onChange={e=>setMapCallName(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {callHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="ID/Name" value={mapCallName} onChange={e=>setMapCallName(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>Postcode *</label>
                    {callHeaders.length ? (
          <select value={mapCallPC} onChange={e=>setMapCallPC(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {callHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="Postcode" value={mapCallPC} onChange={e=>setMapCallPC(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>Duration *</label>
                    {callHeaders.length ? (
          <select value={mapCallDur} onChange={e=>setMapCallDur(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {callHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="Duration" value={mapCallDur} onChange={e=>setMapCallDur(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>Days (optional)</label>
                    {callHeaders.length ? (
          <select value={mapCallDays} onChange={e=>setMapCallDays(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {callHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="Days (optional)" value={mapCallDays} onChange={e=>setMapCallDays(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                  {/* Removed deprecated Group/Territory column mapping */}
                </div>
              </div>
              <div className="card" style={{display:'flex', flexDirection:'column', gap:12}}>
                <div style={{fontWeight:600}}>Resources (required)</div>
                <div>
                  <label>Resources Excel <Help text="Columns required: Name/ID, Postcode (or Lat/Lng), Days, Start, End." /></label>
                  <input type="file" accept=".xlsx,.xls" onChange={handleResourcesFileChange} />
                </div>
        <div style={{display:'grid', gridTemplateColumns:'1fr', gap:8}}>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>resNameCol *</label>
                    {resHeaders.length ? (
          <select value={mapResName} onChange={e=>setMapResName(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {resHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="resNameCol" value={mapResName} onChange={e=>setMapResName(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>resPostcodeCol *</label>
                    {resHeaders.length ? (
          <select value={mapResPC} onChange={e=>setMapResPC(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {resHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="resPostcodeCol" value={mapResPC} onChange={e=>setMapResPC(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>resDaysCol *</label>
                    {resHeaders.length ? (
          <select value={mapResDays} onChange={e=>setMapResDays(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {resHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="resDaysCol" value={mapResDays} onChange={e=>setMapResDays(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>resStartCol *</label>
                    {resHeaders.length ? (
          <select value={mapResStart} onChange={e=>setMapResStart(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {resHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="resStartCol" value={mapResStart} onChange={e=>setMapResStart(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                  <div>
                    <label style={{display:'block', fontSize:12, color:'#fff'}}>resEndCol *</label>
                    {resHeaders.length ? (
          <select value={mapResEnd} onChange={e=>setMapResEnd(e.target.value)} style={{width:'100%'}}>
                        <option value="">Auto</option>
                        {resHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    ) : (
          <input placeholder="resEndCol" value={mapResEnd} onChange={e=>setMapResEnd(e.target.value)} style={{width:'100%'}} />
                    )}
                  </div>
                </div>
              </div>
              <div className="card">
                <label>Work-day minutes <Help text="Time budget per route/day (e.g., 480)." /></label>
                <input type="number" value={workDayMin} onChange={e=>setWorkDayMin(Number(e.target.value))} style={{width:100}} />
              </div>
            </>
          )}
          <div className="card">
            <div style={{height: 8, background: '#222', borderRadius: 4, overflow: 'hidden', marginBottom: 6}}>
              <div style={{width: `${progress}%`, height: '100%', background: '#3cb44b', transition: 'width 0.3s'}} />
            </div>
            <div style={{color: '#fff', fontSize: 13}}>{status}</div>
            {failedMessage && (
              <div style={{color: '#e6194b', fontSize: 13, marginTop: 6}}>
                {failedMessage}
                {failedExcelUrl && (
                  <a href={failedExcelUrl} download="failed_postcodes.xlsx" style={{color:'#fff',marginLeft:8}}>Download failed</a>
                )}
              </div>
            )}
          </div>
          <div style={{display:'flex', gap:10}}>
            <button className="btn-primary" onClick={run}>{mode === 'route' ? 'Run Routing' : 'Run Clustering'}</button>
            <button className="btn-secondary" onClick={() => setSidebar('none')}>Cancel</button>
          </div>
        </div>
      );
    }
    if (sidebar === 'open') {
      return (
        <div className="sidebar-inner">
          <div className="sidebar-title">Open Project</div>
          <div className="card">
            {projects.length === 0 && (
              <div style={{marginBottom: 12, color:'#fff'}}>No saved projects.</div>
            )}
            <ul style={{listStyle: 'none', padding: 0, margin: 0, display:'flex', flexDirection:'column', gap:10}}>
              {projects.map((p, idx) => (
                <li key={idx} style={{display:'flex', gap:8, alignItems:'center'}}>
                  <button className="btn-primary" style={{flex:1, textAlign:'left'}} onClick={() => openProject(p)}>
                    {p.name}
                  </button>
                  <button className="btn-secondary" style={{background:'#e6194b', border:'none'}} onClick={e => deleteProject(idx, e)}>Delete</button>
                </li>
              ))}
            </ul>
          </div>
          <div style={{display:'flex', gap:10}}>
            <button className="btn-secondary" onClick={() => setSidebar('none')}>Close</button>
          </div>
        </div>
      );
    }
    if (sidebar === 'project' && current) {
      return (
        <div className="sidebar-inner">
          <div className="sidebar-title">Project</div>
          <h2>{current.name}</h2>
          <div style={{marginBottom: 12}}>Project loaded.</div>
          <button className="btn-secondary" onClick={() => setSidebar('none')}>Close</button>
        </div>
      );
    }
    return (
      <div className="sidebar-inner">
        <div className="sidebar-title">Main Menu</div>
        <div className="card" style={{display:'flex', flexDirection:'column', gap:10}}>
          <button className="btn-primary" onClick={() => {
            setName('');
            setFile(null);
            setFileName('No file selected');
            setResourcesFile(null);
            setMinC(5);
            setMaxC(6);
            setRows([]);
            setMarkers([]);
            setMapKey(k => k + 1);
            setProgress(0);
            setStatus('');
            setFailedCount(0);
            setFailedMessage(null);
            setFailedExcelUrl(null);
            setSortCol(null);
            setSortAsc(true);
            setFilterDay(null);
            setTotalMiles(null);
            setTotalMinutes(null);
            setResourceText('');
            setCurrent(null);
            setSidebar('new');
          }}>+ New Project</button>
          <button className="btn-secondary" onClick={() => {
            setSidebar('open');
            setRows([]);
            setMarkers([]);
            setMapKey(k => k + 1);
            setCurrent(null);
          }}>Open Project</button>
          <button className="btn-secondary" onClick={() => setSidebar('settings')}>Settings</button>
          <button className="btn-secondary" onClick={() => setAiOpen(true)}>{AI_NAME}</button>
        </div>
      </div>
    );
  };

  // Column order feature: change this array to reorder columns
  // columnOrder already defined above

  return (
    <div style={{display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--dark-1)', fontFamily: 'Inter, Arial, sans-serif'}}>
      <header style={{
          position: 'sticky',
          top: 0,
          zIndex: 1000,
          width: '100%',
          height: 56,
          backgroundColor: 'var(--black)',
          backgroundImage: "url('/logo.png')",
          backgroundRepeat: 'repeat-x',
          backgroundPosition: 'center',
          backgroundSize: 'auto 40px',
          borderBottom: '1px solid #000',
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
        }} />
      {/* API status banner */}
    {/* status bar removed; controls moved into Settings */}
    <div style={{display: 'flex', flex: 1, minHeight: 0}}>
  <aside className="sidebar sidebar--img" style={{width: 340, color: '#fff', display: 'flex', flexDirection: 'column', borderRight: '1px solid #222', position: 'relative', zIndex: 2, boxShadow: '2px 0 8px rgba(0,0,0,0.08)'}}>
          {renderSidebarContent()}
        </aside>
        {/* Slide-out AI overlay beside the main sidebar (flex sibling, not fixed) */}
        <div
          aria-hidden={!aiOpen}
          style={{
            width: aiOpen ? 340 : 0,
            transition: 'width 0.25s ease',
            overflow: 'hidden',
            background: 'var(--dark-2)',
            borderRight: '1px solid #222',
            boxShadow: aiOpen ? '2px 0 8px rgba(0,0,0,0.2)' : 'none',
            zIndex: 2,
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          <div className="sidebar-inner" style={{padding: 12, display:'flex', flexDirection:'column', height:'100%'}}>
            <div className="sidebar-title" style={{display:'flex', alignItems:'center', gap:8, paddingBottom:8, borderBottom:'1px solid #222'}}>
              <span>{AI_NAME}</span>
              <span style={{marginLeft:'auto'}}>
                <button className="btn-secondary" onClick={() => setAiOpen(false)}>Close</button>
              </span>
            </div>
            {/* Chat scroll area */}
            <div style={{flex:1, overflowY:'auto', paddingTop:8, display:'flex', flexDirection:'column', gap:8}}>
              {[...aiHistory].reverse().map((m, i) => (
                <div key={i} style={{display:'flex', flexDirection:'column', gap:6}}>
                  <div style={{alignSelf:'flex-end', maxWidth:'100%', background:'#0e1a24', color:'#fff', padding:'10px 12px', borderRadius: 12}}>
                    <div style={{opacity:0.7, fontSize:12, marginBottom:4}}>You</div>
                    <div style={{whiteSpace:'pre-wrap'}}>{m.q}</div>
                  </div>
                  <div style={{alignSelf:'flex-start', maxWidth:'100%', background:'#111', color:'#fff', padding:'10px 12px', borderRadius: 12, border:'1px solid #222'}}>
                    <div style={{opacity:0.7, fontSize:12, marginBottom:4}}>LeeW-AI</div>
                    <div style={{whiteSpace:'pre-wrap'}}>{m.a}</div>
                    {(m.provider || typeof m.used_rows === 'number') && (
                      <div style={{opacity:0.6, fontSize:11, marginTop:8}}>
                        {m.provider ? `Source: ${m.provider}` : ''}
                        {typeof m.used_rows === 'number' ? ` · Rows used: ${m.used_rows}` : ''}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {aiAnswer && (
                <div style={{display:'flex', flexDirection:'column', gap:6}}>
                  <div style={{alignSelf:'flex-end', maxWidth:'100%', background:'#0e1a24', color:'#fff', padding:'10px 12px', borderRadius: 12}}>
                    <div style={{opacity:0.7, fontSize:12, marginBottom:4}}>You</div>
                    <div style={{whiteSpace:'pre-wrap'}}>{aiQuestion}</div>
                  </div>
                  <div style={{alignSelf:'flex-start', maxWidth:'100%', background:'#111', color:'#fff', padding:'10px 12px', borderRadius: 12, border:'1px solid #222'}}>
                    <div style={{opacity:0.7, fontSize:12, marginBottom:4}}>LeeW-AI</div>
                    <div style={{whiteSpace:'pre-wrap'}}>{aiAnswer}</div>
                  </div>
                </div>
              )}
            </div>
            {/* Input at bottom */}
            <div style={{paddingTop:8, borderTop:'1px solid #222', display:'flex', flexDirection:'column', gap:8}}>
              <textarea
                placeholder="Ask or instruct..."
                value={aiQuestion}
                onChange={e=>startTransition(() => setAiQuestion(e.target.value))}
                style={{width:'100%', minHeight:64}}
              />
              <div style={{display:'flex', gap:8}}>
                <button className="btn-primary" onClick={askAI} disabled={aiLoading || !aiQuestion.trim()}>{aiLoading ? 'Working…' : 'Send'}</button>
                <button className="btn-secondary" onClick={()=>{ setAiQuestion(''); setAiAnswer(''); }}>Clear</button>
                <button className="btn-secondary" disabled={!tfUndoRows} onClick={()=>{ if (!tfUndoRows) return; setRows(tfUndoRows); setTfUndoRows(null); setAiAnswer('Reverted last change.'); }}>Undo</button>
              </div>
            </div>
          </div>
        </div>
        {/* Floating vertical banner tab to open/close AI drawer */}
        <div style={{position:'fixed', top: 120, left: aiOpen ? 680 : 340, zIndex: 1001}}>
          <button
            aria-label={aiOpen ? 'Close Ask LeeW-AI' : 'Open Ask LeeW-AI'}
            onClick={() => setAiOpen(v => !v)}
            style={{
              background: '#0e1a24',
              color: '#fff',
              border: '1px solid #222',
              borderRadius: 8,
              width: 28,
              height: 140,
              cursor: 'pointer',
              boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 0,
              writingMode: 'vertical-rl',
              textOrientation: 'mixed',
              fontSize: 12,
              fontWeight: 700
            }}
            title={AI_NAME}
          >
            {aiOpen ? '⟨ ' : '⟩ '}{AI_NAME}
          </button>
        </div>
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, position: 'relative'}}>
          <div style={{flex: 1, minHeight: 0, position: 'relative', zIndex: 10}}>
            <MemoClientMap
              key={mapKey}
              markers={markers}
              rows={rows}
              setRows={setRows}
              centers={centers ?? undefined}
              h3Indices={h3Indices ?? undefined}
              h3Resolution={8}
              mode={mode}
            />
          </div>
          {/* Results Table at bottom, fixed position outside map container */}
          {resultsTableEl}
        </div>
      </div>
    </div>
  );
}