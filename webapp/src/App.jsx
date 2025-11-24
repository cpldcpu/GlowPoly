import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RefreshCw, Layers, Box, Hexagon, Circle, Zap, Scale, Play, Pause } from 'lucide-react';

// --- GEOMETRY GENERATORS ---

const normalizeVertices = (verts, radius = 2) => {
    return verts.map(v => v.normalize().multiplyScalar(radius));
};

const getUniqueEdges = (vertices, threshold = 0.1) => {
    const edges = [];
    const adj = vertices.map(() => []);
    let minD = Infinity;
    for(let i=0; i<vertices.length; i++){
        for(let j=i+1; j<vertices.length; j++){
            const d = vertices[i].distanceTo(vertices[j]);
            if(d < minD && d > 0.01) minD = d;
        }
    }
    const epsilon = 0.1;
    for (let i = 0; i < vertices.length; i++) {
        for (let j = i + 1; j < vertices.length; j++) {
            const dist = vertices[i].distanceTo(vertices[j]);
            if (Math.abs(dist - minD) < epsilon) {
                edges.push([i, j]);
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    return { edges, adj };
};

const Generators = {
    octahedron: () => {
        const raw = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]].map(p => new THREE.Vector3(...p));
        const vertices = normalizeVertices(raw, 2);
        const { edges, adj } = getUniqueEdges(vertices);
        return { vertices, edges, adj, type: 'octahedron' };
    },
    cuboctahedron: () => {
        const raw = [];
        const perms = [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]];
        perms.forEach(p => raw.push(new THREE.Vector3(...p)));
        const vertices = normalizeVertices(raw, 2);
        const { edges, adj } = getUniqueEdges(vertices);
        return { vertices, edges, adj, type: 'cuboctahedron' };
    },
    icosidodecahedron: () => {
        const phi = (1 + Math.sqrt(5)) / 2;
        const icoVerts = [];
        const pushPerms = (x,y,z) => {
            icoVerts.push(new THREE.Vector3(x,y,z));
            icoVerts.push(new THREE.Vector3(y,z,x));
            icoVerts.push(new THREE.Vector3(z,x,y));
        };
        [1, -1].forEach(i => { [phi, -phi].forEach(p => pushPerms(0, i, p)); });
        const icosidodecaVerts = [];
        const written = new Set();
        for(let i=0; i<icoVerts.length; i++){
            for(let j=i+1; j<icoVerts.length; j++){
                if(Math.abs(icoVerts[i].distanceTo(icoVerts[j]) - 2) < 0.1) {
                    const mid = new THREE.Vector3().addVectors(icoVerts[i], icoVerts[j]).multiplyScalar(0.5);
                    const key = `${mid.x.toFixed(3)},${mid.y.toFixed(3)},${mid.z.toFixed(3)}`;
                    if(!written.has(key)){ written.add(key); icosidodecaVerts.push(mid); }
                }
            }
        }
        const vertices = normalizeVertices(icosidodecaVerts, 2.5);
        const { edges, adj } = getUniqueEdges(vertices);
        return { vertices, edges, adj, type: 'icosidodecahedron' };
    },
    rhombicuboctahedron: () => {
        const H = 1 + Math.sqrt(2);
        const baseVerts = [];
        const combs = [[1,1,H], [1,H,1], [H,1,1]];
        combs.forEach(([x,y,z]) => {
             for(let i=0; i<8; i++) {
                 const sx = (i & 1) ? 1 : -1;
                 const sy = (i & 2) ? 1 : -1;
                 const sz = (i & 4) ? 1 : -1;
                 baseVerts.push(new THREE.Vector3(x*sx, y*sy, z*sz));
             }
        });
        const vertices = normalizeVertices(baseVerts, 2.5);
        const { edges, adj } = getUniqueEdges(vertices);
        return { vertices, edges, adj, type: 'rhombicuboctahedron' };
    }
};

// --- GRAPH ALGORITHMS ---

const getShortestDistance = (adj, start, end) => {
    if (start === end) return 0;
    const q = [start];
    const dist = new Map([[start, 0]]);
    while (q.length > 0) {
        const u = q.shift();
        if (u === end) return dist.get(u);
        const d = dist.get(u);
        for (const v of adj[u]) {
            if (!dist.has(v)) { dist.set(v, d + 1); q.push(v); }
        }
    }
    return Infinity;
};

// --- PAIR SEARCH HELPERS ---

const getLoopOppositePairs = (activeLoops) => {
    const pairSet = new Set();
    const pairs = [];
    activeLoops.forEach(loop => {
        const path = loop.path.slice(0, -1); 
        const len = path.length;
        if (len % 2 === 0) {
            const half = len / 2;
            for (let i = 0; i < half; i++) {
                const u = path[i];
                const v = path[i + half];
                const key = `${Math.min(u, v)}-${Math.max(u, v)}`;
                if (!pairSet.has(key)) {
                    pairSet.add(key);
                    pairs.push([u, v]);
                }
            }
        }
    });
    return pairs;
};

const getAntipodalPairs = (vertices) => {
    const pairs = [];
    const used = new Set();
    let maxDist = 0;
    for(let i=0; i<vertices.length; i++){
        for(let j=i+1; j<vertices.length; j++){
            const d = vertices[i].distanceTo(vertices[j]);
            if(d > maxDist) maxDist = d;
        }
    }
    const tolerance = 0.1;
    for(let i=0; i<vertices.length; i++){
        if(used.has(i)) continue;
        for(let j=i+1; j<vertices.length; j++){
            if(used.has(j)) continue;
            if(Math.abs(vertices[i].distanceTo(vertices[j]) - maxDist) < tolerance) {
                pairs.push([i, j]);
                used.add(i); used.add(j);
                break; 
            }
        }
    }
    return pairs;
};

function* getCombinations(elements, k) {
    if (k === 0) { yield []; return; }
    for (let i = 0; i < elements.length; i++) {
        const elem = elements[i];
        for (const rest of getCombinations(elements.slice(i + 1), k - 1)) {
            yield [elem, ...rest];
        }
    }
}

// --- DYNAMIC BALANCE & PHASING LOGIC ---

// Calculates balance assuming we can switch specific pairs ON for specific loops
// Returns stats AND the generated phases (groups of loops + active pair)
const calculateDynamicBalanceStats = (activeLoops, subsetSet) => {
    if (activeLoops.length === 0) return { score: 100, min: 0, max: 0, phases: [] };
    
    let globalMin = Infinity;
    let globalMax = -Infinity;
    const phases = []; // Array of { pair: [u, v], loops: [loopIndex] }
    const phaseMap = new Map(); // "u-v" -> index in phases array

    activeLoops.forEach((loop, loopIdx) => {
        const uniquePath = loop.path.slice(0, -1);
        const possiblePoints = uniquePath.filter(n => subsetSet.has(n));
        
        if (possiblePoints.length < 2) {
            // Loop cannot be driven
            globalMax = Infinity; // Penalty
            return;
        }

        // Find the BEST pair for this loop from the available points
        let bestPair = null;
        let bestBalanceDiff = Infinity; // Minimize difference between segment lengths
        let bestMinLen = 0;
        let bestMaxLen = 0;

        // Check all combinations of 2 points
        for (let i = 0; i < possiblePoints.length; i++) {
            for (let j = i + 1; j < possiblePoints.length; j++) {
                const u = possiblePoints[i];
                const v = possiblePoints[j];
                
                const idxA = uniquePath.indexOf(u);
                const idxB = uniquePath.indexOf(v);
                
                let len1 = Math.abs(idxA - idxB);
                let len2 = uniquePath.length - len1;
                
                const minL = Math.min(len1, len2);
                const maxL = Math.max(len1, len2);
                
                const diff = maxL - minL;
                
                // Prefer more balanced
                if (diff < bestBalanceDiff) {
                    bestBalanceDiff = diff;
                    bestPair = [u, v];
                    bestMinLen = minL;
                    bestMaxLen = maxL;
                }
            }
        }

        if (bestPair) {
            if (bestMinLen < globalMin) globalMin = bestMinLen;
            if (bestMaxLen > globalMax) globalMax = bestMaxLen;

            // Add to Phase
            const key = `${Math.min(bestPair[0], bestPair[1])}-${Math.max(bestPair[0], bestPair[1])}`;
            if (!phaseMap.has(key)) {
                phaseMap.set(key, phases.length);
                phases.push({ pair: bestPair, loops: [] });
            }
            phases[phaseMap.get(key)].loops.push(loopIdx);
        }
    });

    if (globalMax === -Infinity || globalMax === Infinity) return { score: 0, min: 0, max: 0, phases: [] };
    return { 
        score: Math.round((globalMin / globalMax) * 100), 
        min: globalMin, 
        max: globalMax,
        phases: phases
    };
};

const validateConfiguration = (activeLoops, subsetSet, adj) => {
    // Simplified validation: Can every loop be driven by at least one valid pair in the set?
    for (const loop of activeLoops) {
        const uniquePath = loop.path.slice(0, -1);
        const loopPoints = uniquePath.filter(n => subsetSet.has(n));
        if (loopPoints.length < 2) return false;
        
        let driven = false;
        // Check if any pair prevents sneak paths
        for (let i = 0; i < loopPoints.length; i++) {
            for (let j = i + 1; j < loopPoints.length; j++) {
                const u = loopPoints[i];
                const v = loopPoints[j];
                
                const idxA = uniquePath.indexOf(u);
                const idxB = uniquePath.indexOf(v);
                let len1 = Math.abs(idxA - idxB);
                let len2 = uniquePath.length - len1;
                
                // Check both segments
                // Note: A pair drives a loop if BOTH segments are valid (no sneak path shorter than segment)
                const globalDist = getShortestDistance(adj, u, v);
                if (Math.max(len1, len2) <= globalDist) {
                    driven = true;
                    break;
                }
            }
            if(driven) break;
        }
        if (!driven) return false;
    }
    return true;
};

// --- DECOMPOSITION LOGIC (Unchanged from working version) ---
const findDecompositions = (graph) => {
  const { vertices, adj, edges, type } = graph;
  const decompositions = { faces: [], hexagons: [], hamiltonian: [], eulerian: [] };

  const triangles = [];
  for (let i = 0; i < vertices.length; i++) {
    for (const j of adj[i]) {
      for (const k of adj[j]) {
        if (k > i && adj[k].includes(i)) {
          const tri = [i, j, k].sort((a,b)=>a-b);
          const id = tri.join('-');
          if (!triangles.some(t => t.ids.join('-') === id)) {
             triangles.push({ path: [i, j, k, i], color: '#EF4444', name: 'Triangle', ids: tri });
          }
        }
      }
    }
  }
  decompositions.faces.push(...triangles);

  if (type === 'cuboctahedron' || type === 'rhombicuboctahedron') {
     const squares = [];
     const foundKeys = new Set();
     for(let i=0; i<vertices.length; i++){
        for(const n1 of adj[i]){
           for(const n2 of adj[n1]){
               if(n2 === i) continue;
               for(const n3 of adj[n2]){
                   if(n3 === n1 || n3 === i) continue;
                   if(adj[n3].includes(i)){
                       const cycle = [i, n1, n2, n3].sort((a,b)=>a-b);
                       const key = cycle.join('-');
                       if(!foundKeys.has(key)){
                           foundKeys.add(key);
                           squares.push({ path: [i, n1, n2, n3, i], color: '#3B82F6', name: 'Square' });
                       }
                   }
               }
           }
        }
     }
     decompositions.faces.push(...squares);
  }

  if (type === 'icosidodecahedron') {
      const phi = (1 + Math.sqrt(5)) / 2;
      const normals = [];
      const pushPerms = (x,y,z) => {
          normals.push(new THREE.Vector3(x,y,z).normalize());
          normals.push(new THREE.Vector3(y,z,x).normalize());
          normals.push(new THREE.Vector3(z,x,y).normalize());
      };
      [1, -1].forEach(i => { [phi, -phi].forEach(p => pushPerms(0, i, p)); });
      const uniqueNormals = [];
      normals.forEach(n => { if (!uniqueNormals.some(un => un.distanceTo(n) < 0.1)) uniqueNormals.push(n); });

      uniqueNormals.forEach((normal) => {
          const planeVerts = [];
          let maxDot = -Infinity;
          vertices.forEach(v => { const d = v.clone().normalize().dot(normal); if (d > maxDot) maxDot = d; });
          vertices.forEach((v, i) => { if (Math.abs(v.clone().normalize().dot(normal) - maxDot) < 0.05) planeVerts.push(i); });
          if (planeVerts.length === 5) {
             const cycle = [planeVerts[0]]; let curr = planeVerts[0]; const visited = new Set([curr]);
             for(let k=0; k<4; k++){ const n = adj[curr].find(x => planeVerts.includes(x) && !visited.has(x)); if(n!==undefined) { curr=n; cycle.push(curr); visited.add(curr); } }
             cycle.push(planeVerts[0]);
             decompositions.faces.push({ path: cycle, color: '#10B981', name: 'Pentagon' });
          }
      });
  }

  // PETRIE
  const loops = [];
  const pColors = ['#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#3B82F6', '#EF4444'];
  
  if (type === 'rhombicuboctahedron') {
      const axes = [new THREE.Vector3(1,0,0), new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,1)];
      let loopCount = 0;
      axes.forEach(axis => {
          const dots = vertices.map((v, i) => ({ id: i, val: v.dot(axis) }));
          const groups = {};
          dots.forEach(d => { const k = Math.round(d.val * 10); if(!groups[k]) groups[k] = []; groups[k].push(d.id); });
          Object.values(groups).forEach(group => {
              if (group.length === 8) {
                  const cycle = [group[0]]; let curr = group[0]; const visited = new Set([curr]);
                  for(let k=0; k<7; k++) { const n = adj[curr].find(x => group.includes(x) && !visited.has(x)); if(n !== undefined) { curr = n; cycle.push(curr); visited.add(curr); } }
                  cycle.push(group[0]); 
                  if(cycle.length === 9 && adj[curr].includes(group[0])) { loops.push({ path: cycle, color: pColors[loopCount % pColors.length], name: `Octagon ${loopCount+1}` }); loopCount++; }
              }
          });
      });
  } else {
      let pNormals = [];
      if (type === 'octahedron' || type === 'cuboctahedron') {
          pNormals = [[1,1,1], [1,-1,1], [1,1,-1], [1,-1,-1]].map(p => new THREE.Vector3(...p).normalize());
      } else if (type === 'icosidodecahedron') {
           const phi = (1 + Math.sqrt(5)) / 2;
           const tempN = [];
           [1, -1].forEach(i => { [phi, -phi].forEach(p => { tempN.push(new THREE.Vector3(0, i, p).normalize()); tempN.push(new THREE.Vector3(p, 0, i).normalize()); tempN.push(new THREE.Vector3(i, p, 0).normalize()); })});
           tempN.forEach(n => { if (!pNormals.some(ex => Math.abs(Math.abs(ex.dot(n)) - 1) < 0.01)) pNormals.push(n); });
      }
      pNormals.forEach((normal, idx) => {
          const planeVerts = [];
          vertices.forEach((v, i) => { if (Math.abs(v.dot(normal)) < 0.1) planeVerts.push(i); });
          let expectedLen = (type === 'icosidodecahedron') ? 10 : 6;
          if (planeVerts.length === expectedLen) {
             const cycle = [planeVerts[0]]; let current = planeVerts[0]; const visited = new Set([current]);
             for(let k=0; k < expectedLen - 1; k++){ const n = adj[current].find(x => planeVerts.includes(x) && !visited.has(x)); if(n!==undefined) { current=n; cycle.push(current); visited.add(current); } }
             cycle.push(planeVerts[0]);
             if (adj[current].includes(planeVerts[0])) {
                 loops.push({ path: cycle, color: pColors[idx % pColors.length], name: `Petrie ${idx+1} (${expectedLen}-gon)` });
             }
          }
      });
  }
  decompositions.hexagons = loops;

  // HAMILTONIAN
  const findHamiltonian = () => {
      const targetLen = vertices.length;
      for(let attempt=0; attempt<500; attempt++) {
          const path = [0]; const visited = new Set([0]); let curr = 0;
          while(path.length < targetLen) {
             const candidates = adj[curr].filter(n => !visited.has(n));
             if (candidates.length === 0) break;
             const next = candidates[Math.floor(Math.random() * candidates.length)];
             path.push(next); visited.add(next); curr = next;
          }
          if (path.length === targetLen && adj[path[targetLen-1]].includes(path[0])) { path.push(path[0]); return path; }
      }
      return null;
  };
  let ham1 = null; 
  let count=0; while(!ham1 && count++ < 50) ham1 = findHamiltonian();
  if (ham1) decompositions.hamiltonian = [{ path: ham1, color: '#F472B6', name: 'Hamiltonian A' }];

  // EULERIAN
  const findEulerian = () => {
      const stack = [0]; const localAdj = adj.map(n => [...n]); const circuit = [];
      while (stack.length > 0) {
          const v = stack[stack.length - 1];
          if (localAdj[v].length > 0) {
              const u = localAdj[v].pop();
              const uIdx = localAdj[u].indexOf(v);
              if (uIdx > -1) localAdj[u].splice(uIdx, 1);
              stack.push(u);
          } else { circuit.push(stack.pop()); }
      }
      return circuit;
  };
  decompositions.eulerian = [{ path: findEulerian(), color: '#FACC15', name: 'Full Eulerian Circuit' }];

  return { ...graph, decompositions };
};

// --- MAIN COMPONENT ---

export default function GeometricExplorer() {
  const mountRef = useRef(null);
  const [data, setData] = useState(null);
  const [shape, setShape] = useState('cuboctahedron');
  const [mode, setMode] = useState('hexagons');
  const [activeLoops, setActiveLoops] = useState({});
  const [showFeeding, setShowFeeding] = useState(true);
  const [feedingStats, setFeedingStats] = useState({ count: 0, balance: { score: 0, min: 0, max: 0 }, phases: [] });
  const [isCycling, setIsCycling] = useState(false);
  const [currentPhaseIdx, setCurrentPhaseIdx] = useState(-1); // -1 = all
  
  useEffect(() => {
    const raw = Generators[shape]();
    const processed = findDecompositions(raw);
    setData(processed);
    const initialActive = {};
    const defaultMode = (processed.decompositions['hexagons'].length > 0) ? 'hexagons' : 'faces';
    processed.decompositions[defaultMode].forEach(d => initialActive[d.name] = true);
    setActiveLoops(initialActive);
    setMode(defaultMode);
    setIsCycling(false);
    setCurrentPhaseIdx(-1);
  }, [shape]);

  const handleModeChange = (newMode) => {
      setMode(newMode);
      if (!data) return;
      const newActive = {};
      data.decompositions[newMode].forEach(d => newActive[d.name] = true);
      setActiveLoops(newActive);
      setIsCycling(false);
      setCurrentPhaseIdx(-1);
  };

  const toggleLoop = (name) => { setActiveLoops(prev => ({...prev, [name]: !prev[name]})); };

  // --- CYCLING TIMER ---
  useEffect(() => {
      let interval;
      if (isCycling && feedingStats.phases.length > 0) {
          interval = setInterval(() => {
              setCurrentPhaseIdx(prev => (prev + 1) % feedingStats.phases.length);
          }, 1000); // 1 second per phase
      } else {
          setCurrentPhaseIdx(-1);
      }
      return () => clearInterval(interval);
  }, [isCycling, feedingStats.phases]);

  useEffect(() => {
    if (!data || !mountRef.current) return;
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#111827');
    scene.fog = new THREE.Fog('#111827', 5, 25);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    const camDist = (shape === 'icosidodecahedron' || shape === 'rhombicuboctahedron') ? 8 : 6;
    camera.position.set(0, 0, camDist);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.autoRotate = !isCycling;
    controls.autoRotateSpeed = 0.8;
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const p1 = new THREE.PointLight(0xffffff, 1); p1.position.set(10, 10, 10); scene.add(p1);
    const p2 = new THREE.PointLight(0xffffff, 0.5); p2.position.set(-10, -10, -10); scene.add(p2);
    
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(2000 * 3);
    for(let i=0; i<6000; i++) starPos[i] = (Math.random() - 0.5) * 80;
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({color: 0x666666, size: 0.05})));

    const mainGroup = new THREE.Group();
    scene.add(mainGroup);

    const createEdge = (uIdx, vIdx, color, isActive) => {
        const u = data.vertices[uIdx]; const v = data.vertices[vIdx];
        const dir = new THREE.Vector3().subVectors(v, u);
        const len = dir.length();
        const mid = new THREE.Vector3().addVectors(u, v).multiplyScalar(0.5);
        const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0), dir.clone().normalize());
        const g = new THREE.Group();
        g.position.copy(mid); g.setRotationFromQuaternion(quat);
        const r = isActive ? 0.05 : 0.02;
        const op = isActive ? 0.9 : 0.15;
        const c = isActive ? color : "#444";
        const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 8), new THREE.MeshStandardMaterial({ color: c, transparent: true, opacity: op, emissive: isActive ? c : 0, emissiveIntensity: isActive?0.5:0 }));
        g.add(mesh);
        if(isActive) {
            const cone = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.25, 10), new THREE.MeshStandardMaterial({ color: c, emissive: c }));
            cone.position.y = 0.2; g.add(cone);
        }
        return g;
    };

    const nGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const nMat = new THREE.MeshStandardMaterial({ color: "#ccc", roughness: 0.4 });
    data.vertices.forEach(v => { const m = new THREE.Mesh(nGeo, nMat); m.position.copy(v); mainGroup.add(m); });

    // --- LOGIC: OPTIMIZATION & PHASING ---
    const activeLoopsList = (data.decompositions[mode] || []).filter(l => activeLoops[l.name]);
    const activeEdgeMap = new Map();
    
    // Determine what to show based on CYCLING mode
    let loopsToShow = activeLoopsList;
    let pointsToShow = new Set();
    
    // 1. Find Optimal Set (Static)
    let bestSet = new Set(data.vertices.map((_, i) => i));
    if (activeLoopsList.length > 0) {
        // Use Pair Search first (fast and accurate for symmetric)
        let pairs = getLoopOppositePairs(activeLoopsList);
        if (pairs.length === 0) pairs = getAntipodalPairs(data.vertices);
        
        let foundOptimal = false;
        for (let k = 1; k <= 12; k++) {
            if (k > 4 && pairs.length > 15) break;
            for (const combo of getCombinations(pairs, k)) {
                const subset = new Set();
                combo.forEach(p => { subset.add(p[0]); subset.add(p[1]); });
                if (validateConfiguration(activeLoopsList, subset, data.adj)) {
                    bestSet = subset;
                    foundOptimal = true;
                    break;
                }
            }
            if (foundOptimal) break;
        }
    } else { bestSet.clear(); }

    // 2. Calculate Dynamic Phases (based on bestSet)
    const stats = calculateDynamicBalanceStats(activeLoopsList, bestSet);
    setFeedingStats({ count: bestSet.size, balance: stats, phases: stats.phases });

    // 3. Filter visualization if cycling
    if (isCycling && currentPhaseIdx >= 0 && stats.phases.length > 0) {
        const phase = stats.phases[currentPhaseIdx];
        // Show only the pair
        pointsToShow.add(phase.pair[0]);
        pointsToShow.add(phase.pair[1]);
        // Show only the loops driven by this pair
        loopsToShow = activeLoopsList.filter((_, idx) => phase.loops.includes(idx));
    } else {
        // Show all
        pointsToShow = bestSet;
        loopsToShow = activeLoopsList;
    }

    loopsToShow.forEach(loop => { 
        for(let i=0; i<loop.path.length-1; i++) activeEdgeMap.set(`${loop.path[i]}-${loop.path[i+1]}`, loop.color); 
    });

    // RENDER
    data.edges.forEach(([u, v]) => {
        const fwd = activeEdgeMap.get(`${u}-${v}`);
        const bwd = activeEdgeMap.get(`${v}-${u}`);
        mainGroup.add(createEdge(u, v, null, false)); // ghost
        if (fwd) mainGroup.add(createEdge(u, v, fwd, true));
        if (bwd) mainGroup.add(createEdge(v, u, bwd, true));
    });

    if (showFeeding) {
        const fGeo = new THREE.SphereGeometry(0.22, 16, 16);
        pointsToShow.forEach(nodeIdx => {
            const mat = new THREE.MeshBasicMaterial({ color: '#FACC15' });
            const mesh = new THREE.Mesh(fGeo, mat);
            mesh.position.copy(data.vertices[nodeIdx]);
            mainGroup.add(mesh);
        });
    }

    let animId;
    const animate = () => { animId = requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); };
    animate();
    const handleResize = () => { if(mountRef.current){ const w=mountRef.current.clientWidth; const h=mountRef.current.clientHeight; camera.aspect=w/h; camera.updateProjectionMatrix(); renderer.setSize(w,h); } };
    window.addEventListener('resize', handleResize);
    return () => { window.removeEventListener('resize', handleResize); cancelAnimationFrame(animId); if(mountRef.current) mountRef.current.innerHTML=''; renderer.dispose(); };
  }, [data, mode, activeLoops, showFeeding, isCycling, currentPhaseIdx]);

  if (!data) return <div className="h-screen bg-gray-900 flex items-center justify-center text-white">Generating...</div>;
  const currentDecomps = data.decompositions[mode] || [];

  return (
    <div className="w-full h-screen bg-gray-900 flex flex-col md:flex-row font-sans text-gray-100 overflow-hidden">
      <div className="flex-1 relative h-2/3 md:h-full">
        <div ref={mountRef} className="w-full h-full cursor-move" />
        <div className="absolute top-4 left-4 pointer-events-none">
            <h1 className="text-2xl font-bold text-white drop-shadow-md">Eulerian Decomposition</h1>
            <p className="text-gray-400 text-sm capitalize">{shape.replace(/_/g, ' ')} | Diode Analysis</p>
        </div>
        {showFeeding && (
            <div className="absolute bottom-4 left-4 bg-gray-900 bg-opacity-80 p-3 rounded-lg border border-gray-700 w-64">
                 <div className="flex items-center justify-between mb-2 pb-2 border-b border-gray-700">
                    <span className="text-xs font-bold text-white flex items-center gap-2">
                        <Zap size={14} className="text-yellow-400" />
                        Points: {feedingStats.count}
                    </span>
                    <span className="text-xs font-bold text-white flex items-center gap-2">
                        <Scale size={14} className="text-green-400" />
                        Bal: {feedingStats.balance.score}%
                    </span>
                </div>
                
                {/* Sequence Controls */}
                <div className="flex items-center justify-between bg-gray-800 p-2 rounded pointer-events-auto">
                    <span className="text-xs font-semibold text-gray-300">Sequence: {isCycling ? `Phase ${currentPhaseIdx+1}` : 'OFF'}</span>
                    <button 
                        onClick={() => setIsCycling(!isCycling)}
                        className={`p-1 rounded hover:bg-gray-700 ${isCycling ? 'text-green-400' : 'text-gray-400'}`}
                    >
                        {isCycling ? <Pause size={16} /> : <Play size={16} />}
                    </button>
                </div>
            </div>
        )}
      </div>

      <div className="w-full md:w-80 bg-gray-800 p-6 flex flex-col gap-6 shadow-2xl z-10 overflow-y-auto border-l border-gray-700">
        <div className="space-y-2">
            <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500">Select Shape</h2>
            <div className="flex flex-wrap gap-2">
                {['octahedron', 'cuboctahedron', 'icosidodecahedron', 'rhombicuboctahedron'].map(s => (
                    <button key={s} onClick={() => setShape(s)} className={`p-2 rounded flex-1 min-w-[100px] text-xs font-bold transition-colors ${shape===s ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>{s.replace(/_/g, ' ').slice(0,10)}...</button>
                ))}
            </div>
        </div>
        <div className="space-y-3">
            <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500">Mode</h2>
            <div className="grid grid-cols-2 gap-2">
                <button onClick={() => handleModeChange('hexagons')} className={`p-3 rounded-lg flex flex-col items-center gap-2 ${mode === 'hexagons' ? 'bg-indigo-600' : 'bg-gray-700 hover:bg-gray-600'}`}>{shape==='icosidodecahedron'?<Circle size={20}/>:<Hexagon size={20}/>}<span className="text-xs font-semibold">Petrie</span></button>
                <button onClick={() => handleModeChange('hamiltonian')} className={`p-3 rounded-lg flex flex-col items-center gap-2 ${mode === 'hamiltonian' ? 'bg-pink-600' : 'bg-gray-700 hover:bg-gray-600'}`}><RefreshCw size={20} /><span className="text-xs font-semibold">Hamiltonian</span></button>
                <button onClick={() => handleModeChange('faces')} className={`p-3 rounded-lg flex flex-col items-center gap-2 ${mode === 'faces' ? 'bg-red-600' : 'bg-gray-700 hover:bg-gray-600'}`}><Box size={20} /><span className="text-xs font-semibold">Faces</span></button>
                <button onClick={() => handleModeChange('eulerian')} className={`p-3 rounded-lg flex flex-col items-center gap-2 ${mode === 'eulerian' ? 'bg-yellow-600' : 'bg-gray-700 hover:bg-gray-600'}`}><Layers size={20} /><span className="text-xs font-semibold">Eulerian</span></button>
            </div>
        </div>
        <div className="flex items-center justify-between bg-gray-700 p-3 rounded-lg">
            <span className="text-sm font-medium">Show Feeding Points</span>
            <button onClick={() => setShowFeeding(!showFeeding)} className={`w-10 h-6 rounded-full p-1 transition-colors ${showFeeding ? 'bg-green-500' : 'bg-gray-900'}`}><div className={`w-4 h-4 bg-white rounded-full transition-transform ${showFeeding ? 'translate-x-4' : 'translate-x-0'}`} /></button>
        </div>
        <div className="flex-1 space-y-3">
            <h2 className="text-xs font-bold uppercase tracking-widest text-gray-500">Active Loops</h2>
            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                {currentDecomps.map((loop, idx) => (
                    <div key={idx} onClick={() => toggleLoop(loop.name)} className={`flex items-center p-3 rounded-lg cursor-pointer border ${activeLoops[loop.name] ? 'bg-gray-700 border-gray-600' : 'bg-gray-900 border-transparent opacity-50'}`}>
                        <div className="w-3 h-3 rounded-full mr-3 shrink-0" style={{ backgroundColor: loop.color }} />
                        <div className="flex-1 min-w-0"><span className="text-sm font-medium block truncate">{loop.name}</span><span className="text-xs text-gray-400">{loop.path.length - 1} Edges</span></div>
                    </div>
                ))}
            </div>
        </div>
      </div>
    </div>
  );
}
