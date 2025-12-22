/**
 * Glowing Polyhedrons - Geodesic Cover Viewer
 * Three.js implementation with flow analysis and retro wireframe aesthetic
 */

import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ============================================
// Configuration
// ============================================

const CONFIG = {
    colors: {
        background: 0x0a0a12,
        wireframe: 0x5588cc,
        wireframeGlow: 0x00ffff,
        vertex: 0x00ffff,
        pathPalette: [
            0xff0066, 0x00ff88, 0xffaa00, 0x00aaff,
            0xff00ff, 0xaaff00, 0xff6600, 0x0066ff,
            0xff0000, 0x00ffff, 0xff00aa, 0x88ff00,
            0xaa00ff, 0x00ff00, 0xffff00, 0xff8800,
            0x8800ff, 0x00ffaa, 0xff0088, 0x00ff66
        ]
    },
    bloom: {
        strength: 1.2,
        radius: 0.5,
        threshold: 0.1
    },
    animation: {
        rotationSpeed: 0.002,
        flowSpeed: 0.8
    }
};

// ============================================
// State
// ============================================

let scene, camera, renderer, composer, controls, labelRenderer;
let polyhedronGroup, pathsGroup, verticesGroup, arrowsGroup, labelsGroup;
let edgeCylinders = {};  // "u-v" -> cylinder mesh reference
let currentModel = null;
let currentResult = null;
let currentFlowData = null;
let allResults = [];
let allCycleResults = [];  // Cycle decomposition results
let isCycleMode = false;   // True when viewing a cycle decomposition
let flowParticles = [];
let clock = new THREE.Clock();

// UI State
const state = {
    autoRotate: true,
    showVertices: false,
    showLabels: false,
    showPaths: true,
    flowBrightness: false,
    animateFlow: false
};

// ============================================
// Initialization
// ============================================

async function init() {
    setupThreeJS();
    setupPostProcessing();
    setupControls();
    await loadData();
    setupUI();
    hideLoading();
    animate();
}

function setupThreeJS() {
    const canvas = document.getElementById('webgl-canvas');
    const container = canvas.parentElement;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(CONFIG.colors.background);

    // Camera
    camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        100
    );
    camera.position.set(0, 0, 4);

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Groups
    polyhedronGroup = new THREE.Group();
    pathsGroup = new THREE.Group();
    verticesGroup = new THREE.Group();
    arrowsGroup = new THREE.Group();
    labelsGroup = new THREE.Group();
    scene.add(polyhedronGroup);
    scene.add(pathsGroup);
    scene.add(verticesGroup);
    scene.add(arrowsGroup);
    scene.add(labelsGroup);

    // Trackball Controls - allows unlimited rotation in all directions
    controls = new TrackballControls(camera, canvas);
    controls.rotateSpeed = 3.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.noZoom = false;
    controls.noPan = false;
    controls.staticMoving = false;
    controls.dynamicDampingFactor = 0.1;

    // CSS2D Renderer for vertex labels
    labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(container.clientWidth, container.clientHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    labelRenderer.domElement.style.pointerEvents = 'none';
    container.appendChild(labelRenderer.domElement);

    // Resize handler
    window.addEventListener('resize', onResize);
}

function setupPostProcessing() {
    const container = document.getElementById('webgl-canvas').parentElement;

    composer = new EffectComposer(renderer);

    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);

    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(container.clientWidth, container.clientHeight),
        CONFIG.bloom.strength,
        CONFIG.bloom.radius,
        CONFIG.bloom.threshold
    );
    composer.addPass(bloomPass);
}

function setupControls() {
    // Auto-rotate toggle (handled in animation loop)
    document.getElementById('auto-rotate').addEventListener('change', (e) => {
        state.autoRotate = e.target.checked;
    });

    // Show vertex labels toggle
    document.getElementById('show-labels').addEventListener('change', (e) => {
        state.showLabels = e.target.checked;
        // Toggle both the CSS2DObject visibility and the DOM element display
        for (const child of labelsGroup.children) {
            child.visible = state.showLabels;
            if (child.element) {
                child.element.style.display = state.showLabels ? '' : 'none';
            }
        }
    });

    // Show paths toggle - updates edge coloring
    document.getElementById('show-paths').addEventListener('change', (e) => {
        state.showPaths = e.target.checked;
        pathsGroup.visible = state.showPaths;
        arrowsGroup.visible = state.showPaths;
        updateEdgeColors();
    });

    // Flow brightness toggle - scales edge brightness by flow weight
    document.getElementById('flow-brightness').addEventListener('change', (e) => {
        state.flowBrightness = e.target.checked;
        updateEdgeColors();
    });

    // Animate flow toggle - re-render paths when changed
    document.getElementById('animate-flow').addEventListener('change', (e) => {
        state.animateFlow = e.target.checked;
        // Re-render to show/hide arrows vs particles (only if paths are shown)
        if (state.showPaths) {
            if (isCycleMode && currentResult && currentResult.cycles) {
                // Cycle mode: re-render cycle paths
                renderCyclePaths(currentResult.cycles);
            } else if (currentFlowData) {
                // Geodesic mode: re-render geodesic paths
                renderPaths(currentFlowData);
                updateEdgeColors();  // Restore proper edge coloring after re-render
            }
        }
    });

    // Help Modal controls
    const helpModal = document.getElementById('help-modal');
    const helpBtn = document.getElementById('help-btn');
    const helpClose = document.getElementById('help-close');

    if (helpBtn && helpModal && helpClose) {
        helpBtn.addEventListener('click', () => {
            helpModal.classList.toggle('active');
        });

        helpClose.addEventListener('click', () => {
            helpModal.classList.remove('active');
        });

        // Close when clicking outside content (handled by shared logic below)
    }

    // Flow Analysis Modal controls
    const flowModal = document.getElementById('flow-modal');
    const flowBtn = document.getElementById('flow-analysis-btn');
    const modalClose = document.getElementById('modal-close');

    flowBtn.addEventListener('click', () => {
        flowModal.classList.toggle('active');
    });

    modalClose.addEventListener('click', () => {
        flowModal.classList.remove('active');
    });

    // Shared modal close on outside click
    window.addEventListener('click', (e) => {
        if (e.target === flowModal) {
            flowModal.classList.remove('active');
        }
        if (e.target === helpModal) {
            helpModal.classList.remove('active');
        }
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && flowModal.classList.contains('active')) {
            flowModal.classList.remove('active');
        }
    });
}

function onResize() {
    const container = document.getElementById('webgl-canvas').parentElement;
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
    composer.setSize(width, height);
    labelRenderer.setSize(width, height);
}

// ============================================
// Data Loading
// ============================================

async function loadData() {
    try {
        // Load geodesic cover results
        const response = await fetch('data/geodesic_cover_results.json');
        allResults = await response.json();

        // Filter to those with solutions and sort by edge count
        allResults.sort((a, b) => a.nE - b.nE);

    } catch (error) {
        console.error('Failed to load geodesic results:', error);
        allResults = [];
    }

    try {
        // Load cycle decomposition results
        const cycleResponse = await fetch('data/cycle_decomp_results.json');
        allCycleResults = await cycleResponse.json();

        // Sort by edge count
        allCycleResults.sort((a, b) => (a.n_edges || 0) - (b.n_edges || 0));

    } catch (error) {
        console.error('Failed to load cycle results:', error);
        allCycleResults = [];
    }
}

async function loadModel(polyName) {
    // Sanitize name for filename matching
    const sanitized = polyName.toLowerCase().replace(/[^a-z0-9]/g, '-');

    try {
        const response = await fetch(`models/${sanitized}.json`);
        if (!response.ok) throw new Error('Model not found');
        return await response.json();
    } catch (error) {
        console.error(`Failed to load model ${polyName}:`, error);
        return null;
    }
}

// ============================================
// UI Setup
// ============================================

function setupUI() {
    const select = document.getElementById('poly-select');

    // Setup filter checkbox handlers
    const filterIds = ['filter-1', 'filter-2', 'filter-3', 'filter-4', 'filter-none', 'filter-c'];
    filterIds.forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            updateDropdown();
        });
    });

    // Event listener for selection
    select.addEventListener('change', async (e) => {
        await selectPolyhedron(e.target.value);
    });

    // Initial dropdown population
    updateDropdown();
}

function getFilterState() {
    return {
        show1: document.getElementById('filter-1').checked,
        show2: document.getElementById('filter-2').checked,
        show3: document.getElementById('filter-3').checked,
        show4: document.getElementById('filter-4').checked,
        showNone: document.getElementById('filter-none').checked,
        showCyclic: document.getElementById('filter-c').checked
    };
}

function getPairCount(result) {
    if (!result.solution) return 0;
    return result.solution.length;
}

function updateDropdown() {
    const select = document.getElementById('poly-select');
    const filters = getFilterState();

    // Filter geodesic results based on checkbox state
    const filtered = allResults.filter(r => {
        const pairCount = getPairCount(r);
        if (pairCount === 0) return filters.showNone;
        if (pairCount === 1) return filters.show1;
        if (pairCount === 2) return filters.show2;
        if (pairCount === 3) return filters.show3;
        return filters.show4;  // 4 or more
    });

    // Filter cycle results
    const filteredCycles = filters.showCyclic ? allCycleResults.filter(r => r.ok) : [];

    select.innerHTML = '';

    // Group by solution availability
    const withSolution = filtered.filter(r => r.solution);
    const withoutSolution = filtered.filter(r => !r.solution);

    // Add options with geodesic solutions first
    if (withSolution.length > 0) {
        const optGroup = document.createElement('optgroup');
        optGroup.label = '✓ Geodesic Solutions';
        withSolution.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.name;
            const k = r.solution ? r.solution.length : 0;
            opt.textContent = `${r.name} (E=${r.nE}, k=${k})`;
            optGroup.appendChild(opt);
        });
        select.appendChild(optGroup);
    }

    // Add cycle decomposition results
    if (filteredCycles.length > 0) {
        const optGroup = document.createElement('optgroup');
        optGroup.label = '⟲ Cycle Decomposition';
        filteredCycles.forEach(r => {
            const opt = document.createElement('option');
            // Use special prefix to identify cycle results
            const modelName = r.model.replace('.json', '');
            opt.value = 'cycle:' + modelName;
            opt.textContent = `${modelName} (E=${r.n_edges}, L=${r.cycle_length}, c=${r.num_cycles})`;
            optGroup.appendChild(opt);
        });
        select.appendChild(optGroup);
    }

    // Add geodesic options without solutions
    if (withoutSolution.length > 0) {
        const optGroup = document.createElement('optgroup');
        optGroup.label = '○ No Geodesic Solution';
        withoutSolution.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r.name;
            opt.textContent = `${r.name} (E=${r.nE})`;
            optGroup.appendChild(opt);
        });
        select.appendChild(optGroup);
    }

    // Select first with solution by default
    if (withSolution.length > 0) {
        select.value = withSolution[0].name;
        selectPolyhedron(withSolution[0].name);
    } else if (filteredCycles.length > 0) {
        const firstCycle = filteredCycles[0].model.replace('.json', '');
        select.value = 'cycle:' + firstCycle;
        selectPolyhedron('cycle:' + firstCycle);
    } else if (filtered.length > 0) {
        select.value = filtered[0].name;
        selectPolyhedron(filtered[0].name);
    }
}

async function selectPolyhedron(name) {
    // Check if this is a cycle decomposition result
    if (name.startsWith('cycle:')) {
        isCycleMode = true;
        const modelName = name.substring(6);  // Remove 'cycle:' prefix

        // Find cycle result
        currentResult = allCycleResults.find(r => r.model.replace('.json', '') === modelName);
        if (!currentResult) return;

        currentModel = await loadModel(modelName);
        if (!currentModel) return;

        updateStats();
        renderPolyhedron();

        // Render cycle paths
        if (currentResult.cycles && currentResult.cycles.length > 0) {
            renderCyclePaths(currentResult.cycles);
            updateLegend();
        } else {
            clearPaths();
            clearLegend();
        }
        // Clear just the flow modal (not pair-weights which shows cycle weights)
        document.getElementById('flow-stats').innerHTML = '<p class="no-data">Cycle mode - no flow analysis</p>';
        document.getElementById('flow-summary').textContent = '-';

        updateEdgeColors();
        return;
    }

    // Standard geodesic result
    isCycleMode = false;
    currentResult = allResults.find(r => r.name === name);
    if (!currentResult) return;

    currentModel = await loadModel(name);
    if (!currentModel) return;

    updateStats();
    renderPolyhedron();

    if (currentResult.solution) {
        currentFlowData = computeFlowAnalysis(currentResult.solution);
        renderPaths(currentFlowData);
        updateFlowStats(currentFlowData);
        updateLegend();
    } else {
        currentFlowData = null;
        clearPaths();
        clearFlowStats();
        clearLegend();
    }

    // Apply edge colors based on current show/brightness state
    updateEdgeColors();
}

function updateStats() {
    // Hide cycle-specific stats by default
    document.getElementById('stat-cycles-row').style.display = 'none';
    document.getElementById('stat-endpoints-row').style.display = 'none';
    document.getElementById('stat-uniformity-row').style.display = 'none';

    if (isCycleMode) {
        // Cycle decomposition stats
        document.getElementById('stat-vertices').textContent = currentResult.n_vertices || '-';
        document.getElementById('stat-edges').textContent = currentResult.n_edges || '-';
        document.getElementById('stat-k').textContent = '-';
        document.getElementById('stat-L').textContent = currentResult.cycle_length || '-';

        // Show cycle-specific stats
        document.getElementById('stat-cycles-row').style.display = '';
        document.getElementById('stat-cycles').textContent = currentResult.num_cycles || '-';
        document.getElementById('stat-endpoints-row').style.display = '';
        document.getElementById('stat-endpoints').textContent = currentResult.num_unique_endpoints || '-';

        // Update pair-weights container with cycle weights
        updateCycleWeights();
    } else {
        // Geodesic stats
        document.getElementById('stat-vertices').textContent = currentResult.nV || '-';
        document.getElementById('stat-edges').textContent = currentResult.nE || '-';
        document.getElementById('stat-k').textContent = currentResult.best_k || '-';
        document.getElementById('stat-L').textContent = currentResult.best_L || '-';
    }
}

function updateCycleWeights() {
    const weightsContainer = document.getElementById('pair-weights');

    if (!currentResult || !currentResult.cycle_weights || !currentResult.cycles) {
        weightsContainer.innerHTML = '<p class="no-data">-</p>';
        return;
    }

    let html = '';

    // Show flow min/max style summary
    const weights = currentResult.cycle_weights;
    const minWeight = Math.min(...weights).toFixed(2);
    const maxWeight = Math.max(...weights).toFixed(2);
    html += `<div class="flow-minmax-row"><span>Flow: ${minWeight} – ${maxWeight}</span></div>`;
    html += '<hr style="border-color: var(--cyan-dim); margin: 6px 0;">';

    // Show individual cycle weights
    currentResult.cycles.forEach((cycle, idx) => {
        const weight = currentResult.cycle_weights[idx] !== undefined ?
            currentResult.cycle_weights[idx].toFixed(3) : '-';
        html += `<div class="pair-weight-item">
            <span class="pair-label">C${idx + 1}: ${cycle.start}⟲${cycle.end}</span>
            <span class="pair-value">${weight}</span>
        </div>`;
    });

    weightsContainer.innerHTML = html || '<p class="no-data">-</p>';
}

// ============================================
// Flow Analysis (ported from Python)
// ============================================

function extractPaths(pair) {
    if (pair.paths && pair.paths.length > 0) return pair.paths;
    if (pair.path && pair.path.length > 0) return [pair.path];
    return [];
}

function canonicalEdge(u, v) {
    return u <= v ? `${u}-${v}` : `${v}-${u}`;
}

function flowForPair(paths, s, t) {
    if (!paths || paths.length === 0) return {};

    // Build directed adjacency
    const adj = {};
    const indeg = {};
    const nodes = new Set();

    for (const p of paths) {
        for (let i = 0; i < p.length - 1; i++) {
            const a = p[i], b = p[i + 1];
            nodes.add(a);
            nodes.add(b);

            if (!adj[a]) adj[a] = [];
            if (!adj[a].includes(b)) {
                adj[a].push(b);
                indeg[b] = (indeg[b] || 0) + 1;
            }
            if (!(a in indeg)) indeg[a] = indeg[a] || 0;
        }
    }

    // Topological sort
    const queue = [];
    for (const n of nodes) {
        if ((indeg[n] || 0) === 0) queue.push(n);
    }

    const topo = [];
    while (queue.length > 0) {
        const u = queue.shift();
        topo.push(u);
        for (const v of (adj[u] || [])) {
            indeg[v]--;
            if (indeg[v] === 0) queue.push(v);
        }
    }

    // If cycle detected, fallback to path-average
    if (topo.length !== nodes.size) {
        const flowEdge = {};
        const sharePerPath = 1.0 / paths.length;
        for (const p of paths) {
            for (let i = 0; i < p.length - 1; i++) {
                const key = canonicalEdge(p[i], p[i + 1]);
                flowEdge[key] = (flowEdge[key] || 0) + sharePerPath;
            }
        }
        return flowEdge;
    }

    // Flow computation
    const flowNode = { [s]: 1.0 };
    const flowEdge = {};

    for (const u of topo) {
        const out = adj[u] || [];
        if (out.length === 0) continue;

        const share = (flowNode[u] || 0) / out.length;
        for (const v of out) {
            const key = canonicalEdge(u, v);
            flowEdge[key] = (flowEdge[key] || 0) + share;
            flowNode[v] = (flowNode[v] || 0) + share;
        }
    }

    return flowEdge;
}

function aggregateFlow(solution, normalizeMax = true) {
    const total = {};

    for (const pair of solution) {
        const paths = extractPaths(pair);
        const pf = flowForPair(paths, pair.s, pair.t);

        for (const [e, val] of Object.entries(pf)) {
            total[e] = (total[e] || 0) + val;
        }
    }

    if (normalizeMax && Object.keys(total).length > 0) {
        const maxVal = Math.max(...Object.values(total));
        if (maxVal > 0) {
            for (const e of Object.keys(total)) {
                total[e] /= maxVal;
            }
        }
    }

    return total;
}

function computeFlowAnalysis(solution) {
    const rawFlows = aggregateFlow(solution, false);
    const normFlows = aggregateFlow(solution, true);

    // Build edge list
    const edges = Object.keys(rawFlows).sort((a, b) => {
        const [a1, a2] = a.split('-').map(Number);
        const [b1, b2] = b.split('-').map(Number);
        return a1 !== b1 ? a1 - b1 : a2 - b2;
    });

    const nEdges = edges.length;
    const nPairs = solution.length;

    // Build matrix A[edge][pair] = flow contribution
    const A = [];
    const pairFlows = [];

    for (let j = 0; j < nPairs; j++) {
        const pair = solution[j];
        const pf = flowForPair(extractPaths(pair), pair.s, pair.t);
        pairFlows.push(pf);
    }

    for (let i = 0; i < nEdges; i++) {
        A[i] = [];
        for (let j = 0; j < nPairs; j++) {
            A[i][j] = pairFlows[j][edges[i]] || 0;
        }
    }

    // Least squares: minimize ||A*w - 1||^2
    // Simple iterative solution (since we don't have numpy)
    let weights = new Array(nPairs).fill(1.0 / nPairs);

    if (nEdges > 0 && nPairs > 0) {
        // Gradient descent to find optimal weights
        const lr = 0.1;
        const iterations = 100;

        for (let iter = 0; iter < iterations; iter++) {
            // Compute Aw
            const Aw = edges.map((_, i) =>
                A[i].reduce((sum, aij, j) => sum + aij * weights[j], 0)
            );

            // Compute gradient: 2 * A^T * (Aw - 1)
            const gradient = weights.map((_, j) => {
                let grad = 0;
                for (let i = 0; i < nEdges; i++) {
                    grad += A[i][j] * (Aw[i] - 1);
                }
                return 2 * grad;
            });

            // Update weights
            for (let j = 0; j < nPairs; j++) {
                weights[j] = Math.max(0, weights[j] - lr * gradient[j]);
            }
        }
    }

    // Compute optimized edge flows
    const optFlows = {};
    for (let i = 0; i < nEdges; i++) {
        const edgeKey = edges[i];
        optFlows[edgeKey] = A[i].reduce((sum, aij, j) => sum + aij * weights[j], 0);
    }

    // Normalize optimized flows
    const maxOpt = Math.max(...Object.values(optFlows), 1e-9);
    for (const e of Object.keys(optFlows)) {
        optFlows[e] /= maxOpt;
    }

    return {
        edges,
        rawFlows,
        normFlows,
        optFlows,
        weights,
        pairFlows
    };
}

function updateFlowStats(flowData) {
    const container = document.getElementById('flow-stats');

    if (!flowData || flowData.edges.length === 0) {
        container.innerHTML = '<p class="no-data">No flow data</p>';
        return;
    }

    // Show all edges in a scrollable table
    let html = '<table><thead><tr><th>Edge</th><th>Raw</th><th>Opt</th></tr></thead><tbody>';

    for (const e of flowData.edges) {
        html += `<tr>
            <td class="edge-col">${e}</td>
            <td class="raw-col">${(flowData.normFlows[e] || 0).toFixed(2)}</td>
            <td class="opt-col">${(flowData.optFlows[e] || 0).toFixed(2)}</td>
        </tr>`;
    }

    html += '</tbody></table>';
    container.innerHTML = html;

    // Update pair weights with min/max flow stats
    const weightsContainer = document.getElementById('pair-weights');
    let weightsHtml = '';

    // Calculate min/max optimized edge flow
    const optFlowValues = Object.values(flowData.optFlows);
    const minOptFlow = optFlowValues.length > 0 ? Math.min(...optFlowValues) : 0;
    const maxOptFlow = optFlowValues.length > 0 ? Math.max(...optFlowValues) : 0;

    // Min/Max on one line
    weightsHtml += `<div class="flow-minmax-row"><span>Flow: ${minOptFlow.toFixed(2)} – ${maxOptFlow.toFixed(2)}</span></div>`;
    weightsHtml += '<hr style="border-color: var(--cyan-dim); margin: 6px 0;">';

    if (currentResult && currentResult.solution) {
        currentResult.solution.forEach((pair, idx) => {
            weightsHtml += `<div class="pair-weight-item">
                <span class="pair-label">${pair.s}→${pair.t}</span>
                <span class="pair-value">${flowData.weights[idx].toFixed(3)}</span>
            </div>`;
        });
    }

    weightsContainer.innerHTML = weightsHtml || '<p class="no-data">-</p>';
}

function clearFlowStats() {
    document.getElementById('flow-stats').innerHTML = '<p class="no-data">No solution available</p>';
    document.getElementById('pair-weights').innerHTML = '<p class="no-data">-</p>';
    document.getElementById('flow-summary').textContent = '-';
}

// ============================================
// Rendering
// ============================================

function renderPolyhedron() {
    // Clear existing
    edgeCylinders = {};  // Clear edge references

    while (polyhedronGroup.children.length > 0) {
        const child = polyhedronGroup.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
        polyhedronGroup.remove(child);
    }

    while (verticesGroup.children.length > 0) {
        const child = verticesGroup.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
        verticesGroup.remove(child);
    }

    if (!currentModel) return;

    const vertices = currentModel.vertices;
    const edges = currentModel.edges;

    // Scale vertices to fit nicely
    const scale = computeScale(vertices);

    // Create edge tubes for better depth visualization
    // Using cylinders instead of lines allows for proper depth-based rendering
    const tubeRadius = 0.012;
    const radialSegments = 6;

    for (const edge of edges) {
        const [u, v] = Array.isArray(edge) ? edge : [edge.source || edge.u, edge.target || edge.v];
        const p0 = new THREE.Vector3(
            vertices[u][0] * scale,
            vertices[u][1] * scale,
            vertices[u][2] * scale
        );
        const p1 = new THREE.Vector3(
            vertices[v][0] * scale,
            vertices[v][1] * scale,
            vertices[v][2] * scale
        );

        // Create cylinder between p0 and p1
        const direction = new THREE.Vector3().subVectors(p1, p0);
        const length = direction.length();
        const midpoint = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);

        const geometry = new THREE.CylinderGeometry(tubeRadius, tubeRadius, length, radialSegments);

        // Custom shader material for depth-based opacity
        const material = new THREE.MeshBasicMaterial({
            color: CONFIG.colors.wireframe,
            transparent: true,
            opacity: 0.7,
            depthWrite: true
        });

        const cylinder = new THREE.Mesh(geometry, material);
        cylinder.position.copy(midpoint);

        // Orient cylinder along the edge
        const up = new THREE.Vector3(0, 1, 0);
        const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction.normalize());
        cylinder.setRotationFromQuaternion(quaternion);

        // Store original opacity for depth-based rendering
        cylinder.userData.isWireframeEdge = true;
        cylinder.userData.edgeKey = u < v ? `${u}-${v}` : `${v}-${u}`;

        // Store reference for coloring by flow
        const edgeKey = u < v ? `${u}-${v}` : `${v}-${u}`;
        edgeCylinders[edgeKey] = cylinder;

        polyhedronGroup.add(cylinder);
    }

    // Also add thin line version for glow effect (bloom catches these better)
    const edgePositions = [];
    for (const edge of edges) {
        const [u, v] = Array.isArray(edge) ? edge : [edge.source || edge.u, edge.target || edge.v];
        const p0 = vertices[u];
        const p1 = vertices[v];
        edgePositions.push(
            p0[0] * scale, p0[1] * scale, p0[2] * scale,
            p1[0] * scale, p1[1] * scale, p1[2] * scale
        );
    }

    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));

    const edgeMaterial = new THREE.LineBasicMaterial({
        color: CONFIG.colors.wireframeGlow,
        linewidth: 1,
        transparent: true,
        opacity: 0.3
    });

    const lineSegments = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    polyhedronGroup.add(lineSegments);

    // Create vertex spheres
    const sphereGeometry = new THREE.SphereGeometry(0.04, 16, 16);
    const sphereMaterial = new THREE.MeshBasicMaterial({
        color: CONFIG.colors.vertex,
        transparent: true,
        opacity: 0.8
    });

    for (const v of vertices) {
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.set(v[0] * scale, v[1] * scale, v[2] * scale);
        verticesGroup.add(sphere);
    }

    verticesGroup.visible = state.showVertices;

    // Clear existing labels
    while (labelsGroup.children.length > 0) {
        const child = labelsGroup.children[0];
        if (child.element && child.element.parentNode) {
            child.element.parentNode.removeChild(child.element);
        }
        labelsGroup.remove(child);
    }

    // Create vertex number labels
    vertices.forEach((v, idx) => {
        const labelDiv = document.createElement('div');
        labelDiv.className = 'vertex-label';
        labelDiv.textContent = idx.toString();
        labelDiv.style.color = 'white';
        labelDiv.style.fontSize = '18px';
        labelDiv.style.fontFamily = 'Share Tech Mono, monospace';
        labelDiv.style.fontWeight = 'bold';
        labelDiv.style.textShadow = '0 0 4px black, 0 0 2px black';

        const label = new CSS2DObject(labelDiv);
        label.position.set(v[0] * scale, v[1] * scale, v[2] * scale);
        // Set initial visibility based on state
        label.visible = state.showLabels;
        labelDiv.style.display = state.showLabels ? '' : 'none';
        labelsGroup.add(label);
    });
}

function computeScale(vertices) {
    let maxR = 0;
    for (const v of vertices) {
        const r = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
        maxR = Math.max(maxR, r);
    }
    return maxR > 0 ? 1.5 / maxR : 1;
}

function renderPaths(flowData) {
    clearPaths();

    if (!currentModel || !currentResult || !currentResult.solution) return;

    const vertices = currentModel.vertices;
    const scale = computeScale(vertices);
    const solution = currentResult.solution;

    // First pass: collect which pairs use each edge
    const edgeUsage = {};  // "u-v" -> [pairIdx, pairIdx, ...]

    solution.forEach((pair, pairIdx) => {
        const paths = extractPaths(pair);
        for (const path of paths) {
            for (let i = 0; i < path.length - 1; i++) {
                const a = path[i], b = path[i + 1];
                const key = a < b ? `${a}-${b}` : `${b}-${a}`;
                if (!edgeUsage[key]) edgeUsage[key] = [];
                if (!edgeUsage[key].includes(pairIdx)) {
                    edgeUsage[key].push(pairIdx);
                }
            }
        }
    });

    // Color the existing wireframe cylinders based on flow
    for (const [edgeKey, pairIndices] of Object.entries(edgeUsage)) {
        const cylinder = edgeCylinders[edgeKey];
        if (!cylinder) continue;

        if (pairIndices.length === 1) {
            // Single pair uses this edge - use its color
            const color = new THREE.Color(CONFIG.colors.pathPalette[pairIndices[0] % CONFIG.colors.pathPalette.length]);
            cylinder.material.color = color;
            cylinder.material.opacity = 0.9;
        } else {
            // Multiple pairs share this edge - blend colors
            const blendedColor = new THREE.Color(0, 0, 0);
            for (const pairIdx of pairIndices) {
                const pairColor = new THREE.Color(CONFIG.colors.pathPalette[pairIdx % CONFIG.colors.pathPalette.length]);
                blendedColor.r += pairColor.r / pairIndices.length;
                blendedColor.g += pairColor.g / pairIndices.length;
                blendedColor.b += pairColor.b / pairIndices.length;
            }
            cylinder.material.color = blendedColor;
            cylinder.material.opacity = 0.9;
        }
    }

    // Render flow particles or arrowheads
    solution.forEach((pair, pairIdx) => {
        const color = new THREE.Color(CONFIG.colors.pathPalette[pairIdx % CONFIG.colors.pathPalette.length]);
        const paths = extractPaths(pair);

        for (const path of paths) {
            if (state.animateFlow) {
                createFlowParticles(path, vertices, scale, color, pairIdx);
            } else {
                createArrowheads(path, vertices, scale, color, pairIdx, edgeUsage);
            }
        }

        // Mark start/end vertices
        const sPos = vertices[pair.s];
        const tPos = vertices[pair.t];

        // Start marker (sphere)
        const startGeom = new THREE.SphereGeometry(0.06, 16, 16);
        const startMat = new THREE.MeshBasicMaterial({ color: color });
        const startMesh = new THREE.Mesh(startGeom, startMat);
        startMesh.position.set(sPos[0] * scale, sPos[1] * scale, sPos[2] * scale);
        pathsGroup.add(startMesh);

        // End marker (octahedron)
        const endGeom = new THREE.OctahedronGeometry(0.08);
        const endMat = new THREE.MeshBasicMaterial({ color: color });
        const endMesh = new THREE.Mesh(endGeom, endMat);
        endMesh.position.set(tPos[0] * scale, tPos[1] * scale, tPos[2] * scale);
        pathsGroup.add(endMesh);
    });

    pathsGroup.visible = state.showPaths;
}

/**
 * Render cycle decomposition paths with colored edges and endpoint markers
 */
function renderCyclePaths(cycles) {
    clearPaths();

    if (!currentModel || !cycles || cycles.length === 0) return;

    const vertices = currentModel.vertices;
    const scale = computeScale(vertices);

    // Collect which cycles use each edge
    const edgeUsage = {};  // "u-v" -> [cycleIdx, cycleIdx, ...]

    cycles.forEach((cycle, cycleIdx) => {
        const directedEdges = cycle.directed_edges || [];
        for (const edge of directedEdges) {
            const [a, b] = edge;
            const key = a < b ? `${a}-${b}` : `${b}-${a}`;
            if (!edgeUsage[key]) edgeUsage[key] = [];
            if (!edgeUsage[key].includes(cycleIdx)) {
                edgeUsage[key].push(cycleIdx);
            }
        }
    });

    // Color the existing wireframe cylinders based on cycle membership
    for (const [edgeKey, cycleIndices] of Object.entries(edgeUsage)) {
        const cylinder = edgeCylinders[edgeKey];
        if (!cylinder) continue;

        if (cycleIndices.length === 1) {
            // Single cycle uses this edge
            const color = new THREE.Color(CONFIG.colors.pathPalette[cycleIndices[0] % CONFIG.colors.pathPalette.length]);
            cylinder.material.color = color;
            cylinder.material.opacity = 0.9;
        } else {
            // Multiple cycles share this edge - blend colors
            const blendedColor = new THREE.Color(0, 0, 0);
            for (const cycleIdx of cycleIndices) {
                const cycleColor = new THREE.Color(CONFIG.colors.pathPalette[cycleIdx % CONFIG.colors.pathPalette.length]);
                blendedColor.r += cycleColor.r / cycleIndices.length;
                blendedColor.g += cycleColor.g / cycleIndices.length;
                blendedColor.b += cycleColor.b / cycleIndices.length;
            }
            cylinder.material.color = blendedColor;
            cylinder.material.opacity = 0.9;
        }
    }

    // Create arrowheads or flow particles for directed edges
    cycles.forEach((cycle, cycleIdx) => {
        const color = new THREE.Color(CONFIG.colors.pathPalette[cycleIdx % CONFIG.colors.pathPalette.length]);

        if (state.animateFlow) {
            // Animate flow: create particles along the cycle path
            // Use vertices array and close the loop
            const cyclePath = [...cycle.vertices, cycle.vertices[0]];
            createFlowParticles(cyclePath, vertices, scale, color, cycleIdx);
        } else {
            // Static arrows on each edge
            const directedEdges = cycle.directed_edges || [];
            for (const edge of directedEdges) {
                const [from, to] = edge;
                const fromPos = new THREE.Vector3(
                    vertices[from][0] * scale,
                    vertices[from][1] * scale,
                    vertices[from][2] * scale
                );
                const toPos = new THREE.Vector3(
                    vertices[to][0] * scale,
                    vertices[to][1] * scale,
                    vertices[to][2] * scale
                );

                // Arrow at midpoint pointing toward 'to'
                const mid = new THREE.Vector3().addVectors(fromPos, toPos).multiplyScalar(0.5);
                const direction = new THREE.Vector3().subVectors(toPos, fromPos).normalize();

                const coneGeom = new THREE.ConeGeometry(0.03, 0.08, 8);
                const coneMat = new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: 0.8 });
                const cone = new THREE.Mesh(coneGeom, coneMat);

                cone.position.copy(mid);
                const up = new THREE.Vector3(0, 1, 0);
                const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction);
                cone.setRotationFromQuaternion(quaternion);

                arrowsGroup.add(cone);
            }
        }

        // Mark start/end vertices
        const sPos = vertices[cycle.start];
        const tPos = vertices[cycle.end];

        // Start marker (sphere)
        const startGeom = new THREE.SphereGeometry(0.06, 16, 16);
        const startMat = new THREE.MeshBasicMaterial({ color: color });
        const startMesh = new THREE.Mesh(startGeom, startMat);
        startMesh.position.set(sPos[0] * scale, sPos[1] * scale, sPos[2] * scale);
        pathsGroup.add(startMesh);

        // End marker (octahedron)
        const endGeom = new THREE.OctahedronGeometry(0.08);
        const endMat = new THREE.MeshBasicMaterial({ color: color });
        const endMesh = new THREE.Mesh(endGeom, endMat);
        endMesh.position.set(tPos[0] * scale, tPos[1] * scale, tPos[2] * scale);
        pathsGroup.add(endMesh);
    });

    pathsGroup.visible = state.showPaths;
    arrowsGroup.visible = state.showPaths;
}

function createFlowParticles(path, vertices, scale, color, pairIdx) {
    // Create small particles that flow along the path
    const particleCount = Math.max(2, Math.floor(path.length / 2));

    for (let i = 0; i < particleCount; i++) {
        const geometry = new THREE.SphereGeometry(0.025, 8, 8);
        const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.9
        });

        const particle = new THREE.Mesh(geometry, material);

        // Store path data for animation
        particle.userData = {
            path: path.map(nodeIdx => {
                const v = vertices[nodeIdx];
                return new THREE.Vector3(v[0] * scale, v[1] * scale, v[2] * scale);
            }),
            progress: i / particleCount,
            speed: CONFIG.animation.flowSpeed * (0.8 + Math.random() * 0.4)
        };

        pathsGroup.add(particle);
        flowParticles.push(particle);
    }
}

function createArrowheads(path, vertices, scale, color, pairIdx, edgeUsage) {
    const stripeOffset = 0.025;  // Same offset as lines

    // Create small cone arrowheads at the midpoint of each edge segment
    for (let i = 0; i < path.length - 1; i++) {
        const a = path[i], b = path[i + 1];
        const v0 = vertices[a];
        const v1 = vertices[b];

        const p0 = new THREE.Vector3(v0[0] * scale, v0[1] * scale, v0[2] * scale);
        const p1 = new THREE.Vector3(v1[0] * scale, v1[1] * scale, v1[2] * scale);

        // Direction vector
        const dir = new THREE.Vector3().subVectors(p1, p0);
        const segLen = dir.length();
        dir.normalize();

        // Calculate offset for shared edges
        const key = a < b ? `${a}-${b}` : `${b}-${a}`;
        const pairsOnEdge = edgeUsage ? edgeUsage[key] || [pairIdx] : [pairIdx];
        const stripeIndex = pairsOnEdge.indexOf(pairIdx);
        const numStripes = pairsOnEdge.length;

        // Get offset direction (perpendicular to edge)
        const up = new THREE.Vector3(0, 1, 0);
        let offsetDir = new THREE.Vector3().crossVectors(dir, up);
        if (offsetDir.length() < 0.01) {
            offsetDir = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(1, 0, 0));
        }
        offsetDir.normalize();

        const offsetAmount = (stripeIndex - (numStripes - 1) / 2) * stripeOffset;
        const offset = offsetDir.clone().multiplyScalar(offsetAmount);

        // Midpoint of the segment with offset
        const midpoint = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
        midpoint.add(offset);

        // Create cone geometry for arrow (slightly larger: 0.08 max)
        const arrowLen = Math.min(0.08, segLen * 0.25);
        const arrowRadius = arrowLen * 0.35;
        const coneGeometry = new THREE.ConeGeometry(arrowRadius, arrowLen, 8);
        const coneMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.9
        });

        const arrow = new THREE.Mesh(coneGeometry, coneMaterial);

        // Position at midpoint
        arrow.position.copy(midpoint);

        // Orient cone to point in direction of flow
        // Default cone points up (+Y), we need to rotate it to point along dir
        const quaternion = new THREE.Quaternion().setFromUnitVectors(up, dir);
        arrow.setRotationFromQuaternion(quaternion);

        arrowsGroup.add(arrow);
    }
}

function updateFlowParticles(delta) {
    if (!state.animateFlow) return;

    for (const particle of flowParticles) {
        const { path, speed } = particle.userData;
        if (!path || path.length < 2) continue;

        // Update progress
        particle.userData.progress += speed * delta;
        if (particle.userData.progress >= 1) {
            particle.userData.progress = 0;
        }

        // Interpolate position along path
        const totalSegments = path.length - 1;
        const progressAlongPath = particle.userData.progress * totalSegments;
        const segmentIndex = Math.floor(progressAlongPath);
        const segmentProgress = progressAlongPath - segmentIndex;

        if (segmentIndex < path.length - 1) {
            const p0 = path[segmentIndex];
            const p1 = path[segmentIndex + 1];

            particle.position.lerpVectors(p0, p1, segmentProgress);
        }
    }
}

function clearPaths() {
    while (pathsGroup.children.length > 0) {
        const child = pathsGroup.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
        pathsGroup.remove(child);
    }
    while (arrowsGroup.children.length > 0) {
        const child = arrowsGroup.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
        arrowsGroup.remove(child);
    }
    flowParticles = [];

    // Reset wireframe edge colors to default
    for (const cylinder of Object.values(edgeCylinders)) {
        if (cylinder && cylinder.material) {
            cylinder.material.color = new THREE.Color(CONFIG.colors.wireframe);
            cylinder.material.opacity = 0.7;
        }
    }
}

// Warm white 2700K color (bright for glow effect)
const WARM_WHITE_2700K = new THREE.Color(0xffddaa);

function updateEdgeColors() {
    if (!currentResult) return;

    // Build edge usage map for coloring (needed for both modes)
    const edgeUsage = {};  // "u-v" -> [pairIdx, ...]

    if (currentResult.solution) {
        currentResult.solution.forEach((pair, pairIdx) => {
            const paths = extractPaths(pair);
            for (const path of paths) {
                for (let i = 0; i < path.length - 1; i++) {
                    const a = path[i], b = path[i + 1];
                    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
                    if (!edgeUsage[key]) edgeUsage[key] = [];
                    if (!edgeUsage[key].includes(pairIdx)) {
                        edgeUsage[key].push(pairIdx);
                    }
                }
            }
        });
    } else if (isCycleMode && currentResult.cycles) {
        // Cycle decomposition mode
        currentResult.cycles.forEach((cycle, cycleIdx) => {
            const directedEdges = cycle.directed_edges || [];
            for (const edge of directedEdges) {
                const [a, b] = edge;
                const key = a < b ? `${a}-${b}` : `${b}-${a}`;
                if (!edgeUsage[key]) edgeUsage[key] = [];
                if (!edgeUsage[key].includes(cycleIdx)) {
                    edgeUsage[key].push(cycleIdx);
                }
            }
        });
    }

    // Apply colors to all edges
    for (const [edgeKey, cylinder] of Object.entries(edgeCylinders)) {
        if (!cylinder || !cylinder.material) continue;

        const isUsedEdge = edgeUsage[edgeKey] && edgeUsage[edgeKey].length > 0;

        // Determine base color
        let finalColor;
        if (!state.showPaths) {
            // Paths hidden: use warm white for all edges
            finalColor = WARM_WHITE_2700K.clone();
        } else if (isUsedEdge) {
            // Paths shown and edge is used: use path color(s)
            const pairIndices = edgeUsage[edgeKey];
            if (pairIndices.length === 1) {
                finalColor = new THREE.Color(CONFIG.colors.pathPalette[pairIndices[0] % CONFIG.colors.pathPalette.length]);
            } else {
                finalColor = new THREE.Color(0, 0, 0);
                for (const pairIdx of pairIndices) {
                    const pairColor = new THREE.Color(CONFIG.colors.pathPalette[pairIdx % CONFIG.colors.pathPalette.length]);
                    finalColor.r += pairColor.r / pairIndices.length;
                    finalColor.g += pairColor.g / pairIndices.length;
                    finalColor.b += pairColor.b / pairIndices.length;
                }
            }
        } else {
            // Paths shown but edge not used: default wireframe
            finalColor = new THREE.Color(CONFIG.colors.wireframe);
        }

        // Apply brightness scaling based on optimized flow if enabled
        if (state.flowBrightness && currentFlowData && currentFlowData.optFlows) {
            const optFlow = currentFlowData.optFlows[edgeKey] || 0;
            // Scale brightness: minimum 0.15 for used edges, 0.1 for unused
            const minBrightness = isUsedEdge ? 0.15 : 0.1;
            const scaledBrightness = minBrightness + optFlow * (1.0 - minBrightness);
            finalColor.multiplyScalar(scaledBrightness);
        }

        cylinder.material.color = finalColor;
        cylinder.material.opacity = isUsedEdge ? 0.95 : 0.6;
    }
}

function updateLegend() {
    const legend = document.getElementById('legend');

    // Handle cycle mode
    if (isCycleMode && currentResult && currentResult.cycles) {
        let html = '';
        currentResult.cycles.forEach((cycle, idx) => {
            const color = CONFIG.colors.pathPalette[idx % CONFIG.colors.pathPalette.length];
            const colorHex = '#' + color.toString(16).padStart(6, '0');

            html += `<div class="legend-item">
                <div class="legend-color" style="background-color: ${colorHex}; color: ${colorHex};"></div>
                <span class="legend-label">C${idx + 1}: ${cycle.start} ⟲ ${cycle.end}</span>
            </div>`;
        });
        legend.innerHTML = html;
        return;
    }

    // Handle geodesic mode
    if (!currentResult || !currentResult.solution) {
        legend.innerHTML = '';
        return;
    }

    let html = '';
    currentResult.solution.forEach((pair, idx) => {
        const color = CONFIG.colors.pathPalette[idx % CONFIG.colors.pathPalette.length];
        const colorHex = '#' + color.toString(16).padStart(6, '0');

        html += `<div class="legend-item">
            <div class="legend-color" style="background-color: ${colorHex}; color: ${colorHex};"></div>
            <span class="legend-label">${pair.s} → ${pair.t}</span>
        </div>`;
    });

    legend.innerHTML = html;
}

function clearLegend() {
    document.getElementById('legend').innerHTML = '';
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// ============================================
// Animation Loop
// ============================================

function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    controls.update();

    // Manual auto-rotate (since TrackballControls doesn't have it built-in)
    if (state.autoRotate && polyhedronGroup) {
        polyhedronGroup.rotation.y += delta * 0.5;
        pathsGroup.rotation.y += delta * 0.5;
        verticesGroup.rotation.y += delta * 0.5;
        arrowsGroup.rotation.y += delta * 0.5;
        labelsGroup.rotation.y += delta * 0.5;
    }

    updateFlowParticles(delta);
    updateDepthEmphasis();

    composer.render();
    labelRenderer.render(scene, camera);
}

function updateDepthEmphasis() {
    // Update edge opacity based on distance to camera
    // Edges closer to camera appear more solid/bright
    if (!polyhedronGroup) return;

    const cameraPos = camera.position;

    for (const child of polyhedronGroup.children) {
        if (child.userData.isWireframeEdge && child.material) {
            // Get world position of edge center
            const worldPos = new THREE.Vector3();
            child.getWorldPosition(worldPos);

            // Calculate distance to camera
            const dist = worldPos.distanceTo(cameraPos);

            // Map distance to opacity (closer = higher opacity)
            // Typical view distance is 2-6 units
            const minDist = 1.5;
            const maxDist = 6;
            const normalizedDist = Math.max(0, Math.min(1, (dist - minDist) / (maxDist - minDist)));

            // Foreground edges: bright and thick, background: dim
            const opacity = 0.9 - normalizedDist * 0.6;  // Range: 0.3 to 0.9
            child.material.opacity = opacity;
        }
    }
}

// ============================================
// Start
// ============================================

init();
