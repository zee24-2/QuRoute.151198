/**
 * High-performance HTML5 Canvas Visualizer for Traffic Network & Routing.
 * Supports:
 * - Street-level physical road network route tracing (following actual streets and intersections)
 * - Animated directional pulsing along active vehicle paths
 * - Edge traffic heatmap rendering (green -> orange -> red -> closed)
 * - Depot & Customer nodes with demand badges and status
 * - Interactive node/edge click selection
 * - Vehicle route isolation/highlighting
 * - Interactive A* Point-to-Point pathfinder visualizer
 * - Pan and zoom capabilities
 */

class NetworkCanvas {
  constructor(canvasElement, onNodeSelect, onEdgeSelect) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.onNodeSelect = onNodeSelect;
    this.onEdgeSelect = onEdgeSelect;

    this.graph = null;
    this.routes = []; // High-level customer stops: [[0, 3, 5, 0], ...]
    this.detailedStreetRoutes = []; // Street-level contiguous paths from backend A* expander
    this.highlightVehicleIndex = null; // null = show all, 0..K = highlight vehicle
    this.selectedNodeId = null;
    this.selectedEdge = null; // [u, v]
    this.activeIncidents = [];

    // A* Interactive Pathfinder State
    this.astarMode = false;
    this.astarStartId = null;
    this.astarGoalId = null;
    this.astarResult = null;

    // Transform state
    this.scale = 1.0;
    this.offsetX = 0;
    this.offsetY = 0;
    this.isDragging = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;

    // Animation state
    this.dashOffset = 0;
    this.isAnimating = false;

    this.vehicleColors = [
      '#38bdf8', // Soft Sky Blue (Vehicle 1)
      '#818cf8', // Soft Indigo (Vehicle 2)
      '#34d399', // Soft Emerald (Vehicle 3)
      '#fbbf24', // Soft Amber (Vehicle 4)
      '#f472b6', // Soft Rose Pink (Vehicle 5)
      '#a78bfa', // Soft Lavender (Vehicle 6)
    ];

    this.initEvents();
    this.resize();
    this.startAnimationLoop();
  }

  startAnimationLoop() {
    if (this.isAnimating) return;
    this.isAnimating = true;

    const animate = () => {
      // Advance pulse dash
      this.dashOffset = (this.dashOffset + 0.35) % 32;

      // Redraw if routes or A* path exist
      if (
        (this.detailedStreetRoutes && this.detailedStreetRoutes.length > 0) ||
        (this.routes && this.routes.length > 0) ||
        this.astarResult
      ) {
        this.draw();
      }

      requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this.draw();
  }

  setData(graphData, incidents = []) {
    this.graph = graphData;
    this.activeIncidents = incidents;
    this.autoFit();
    this.draw();
  }

  setRoutes(routes, detailedStreetRoutes = null) {
    this.routes = routes || [];
    if (detailedStreetRoutes) {
      this.detailedStreetRoutes = detailedStreetRoutes;
    } else if (this.graph && this.routes.length > 0) {
      // Build basic point sequences from graph if detailed not provided yet
      this.detailedStreetRoutes = this.buildFallbackStreetRoutes(this.routes);
    } else {
      this.detailedStreetRoutes = [];
    }
    this.draw();
  }

  buildFallbackStreetRoutes(routes) {
    if (!this.graph) return [];
    return routes.map((r, idx) => ({
      vehicle_id: idx,
      high_level_stops: r,
      street_node_path: r,
      street_coordinates: r.map(nid => {
        const n = this.graph.nodes.find(node => node.id === nid);
        return n ? [n.x, n.y] : [0, 0];
      }),
      is_empty: r.length <= 2,
    }));
  }

  setHighlightedVehicle(vehicleIndex) {
    this.highlightVehicleIndex = vehicleIndex;
    this.draw();
  }

  setIncidents(incidents) {
    this.activeIncidents = incidents || [];
    this.draw();
  }

  setAStarResult(astarResult, startId, goalId) {
    this.astarResult = astarResult;
    this.astarStartId = startId;
    this.astarGoalId = goalId;
    this.draw();
  }

  clearAStar() {
    this.astarResult = null;
    this.astarStartId = null;
    this.astarGoalId = null;
    this.draw();
  }

  autoFit() {
    if (!this.graph || !this.graph.nodes || this.graph.nodes.length === 0) return;

    const xs = this.graph.nodes.map(n => n.x);
    const ys = this.graph.nodes.map(n => n.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;

    const rangeX = Math.max(0.1, maxX - minX);
    const rangeY = Math.max(0.1, maxY - minY);

    const padding = 70;
    const scaleX = (w - padding * 2) / rangeX;
    const scaleY = (h - padding * 2) / rangeY;

    this.scale = Math.min(scaleX, scaleY);
    this.offsetX = (w - rangeX * this.scale) / 2 - minX * this.scale;
    this.offsetY = (h - rangeY * this.scale) / 2 - minY * this.scale;
  }

  toScreen(x, y) {
    return {
      x: x * this.scale + this.offsetX,
      y: y * this.scale + this.offsetY,
    };
  }

  toWorld(screenX, screenY) {
    return {
      x: (screenX - this.offsetX) / this.scale,
      y: (screenY - this.offsetY) / this.scale,
    };
  }

  initEvents() {
    window.addEventListener('resize', () => this.resize());

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
      this.handleClick(e.offsetX, e.offsetY);
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        const dx = e.clientX - this.lastMouseX;
        const dy = e.clientY - this.lastMouseY;
        this.offsetX += dx;
        this.offsetY += dy;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.draw();
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      const mouseX = e.offsetX;
      const mouseY = e.offsetY;

      this.offsetX = mouseX - (mouseX - this.offsetX) * zoomFactor;
      this.offsetY = mouseY - (mouseY - this.offsetY) * zoomFactor;
      this.scale *= zoomFactor;
      this.draw();
    });
  }

  handleClick(screenX, screenY) {
    if (!this.graph) return;

    // 1. Check if clicked near a node
    const clickRadius = 18;
    let clickedNode = null;

    for (const node of this.graph.nodes) {
      const s = this.toScreen(node.x, node.y);
      const dist = Math.hypot(s.x - screenX, s.y - screenY);
      if (dist <= clickRadius) {
        clickedNode = node;
        break;
      }
    }

    if (clickedNode) {
      this.selectedNodeId = clickedNode.id;
      this.selectedEdge = null;
      if (this.onNodeSelect) this.onNodeSelect(clickedNode);
      this.draw();
      return;
    }

    // 2. Check if clicked near an edge
    const nodesById = {};
    this.graph.nodes.forEach(n => nodesById[n.id] = n);

    let clickedEdge = null;
    let minEdgeDist = 14;

    for (const edge of this.graph.edges) {
      const uNode = nodesById[edge.u];
      const vNode = nodesById[edge.v];
      if (!uNode || !vNode) continue;

      const p1 = this.toScreen(uNode.x, uNode.y);
      const p2 = this.toScreen(vNode.x, vNode.y);

      const dist = this.pointToSegmentDistance(screenX, screenY, p1.x, p1.y, p2.x, p2.y);
      if (dist < minEdgeDist) {
        minEdgeDist = dist;
        clickedEdge = edge;
      }
    }

    if (clickedEdge) {
      this.selectedEdge = [clickedEdge.u, clickedEdge.v];
      this.selectedNodeId = null;
      if (this.onEdgeSelect) this.onEdgeSelect(clickedEdge);
      this.draw();
    }
  }

  pointToSegmentDistance(px, py, x1, y1, x2, y2) {
    const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2;
    if (l2 === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
  }

  draw() {
    const ctx = this.ctx;
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, w, h);

    if (!this.graph || !this.graph.nodes) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No graph network loaded. Select a generator to begin.', w / 2, h / 2);
      return;
    }

    const nodesById = {};
    this.graph.nodes.forEach(n => nodesById[n.id] = n);

    // 1. Draw Physical Road Edges & Heatmap
    this.drawEdges(ctx, nodesById);

    // 2. Draw Active Detailed Street Routes (Following Actual Roads)
    this.drawStreetRoutes(ctx, nodesById);

    // 3. Draw A* Point-to-Point Path if active
    this.drawAStarPath(ctx);

    // 4. Draw Nodes (Depot, Customers, Intersections)
    this.drawNodes(ctx);
  }

  drawEdges(ctx, nodesById) {
    for (const edge of this.graph.edges) {
      const u = nodesById[edge.u];
      const v = nodesById[edge.v];
      if (!u || !v) continue;

      const p1 = this.toScreen(u.x, u.y);
      const p2 = this.toScreen(v.x, v.y);

      const isSelected = this.selectedEdge && (
        (this.selectedEdge[0] === edge.u && this.selectedEdge[1] === edge.v) ||
        (this.selectedEdge[0] === edge.v && this.selectedEdge[1] === edge.u)
      );

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);

      if (edge.is_closed) {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = isSelected ? 5 : 3;
        ctx.setLineDash([5, 5]);
      } else {
        ctx.setLineDash([]);
        const mult = edge.congestion_multiplier;
        ctx.strokeStyle = this.getCongestionColor(mult);
        ctx.lineWidth = isSelected ? 5 : (mult > 2.0 ? 3.2 : 2.0);
      }

      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Incident Icon on congested/closed edges
      if (edge.congestion_multiplier > 3.0 || edge.is_closed) {
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        ctx.fillStyle = edge.is_closed ? '#ef4444' : '#f59e0b';
        ctx.beginPath();
        ctx.arc(midX, midY, 6, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }

  getCongestionColor(multiplier) {
    if (multiplier <= 1.05) return 'rgba(161, 161, 170, 0.35)'; // Free flow neutral
    if (multiplier <= 1.6) return 'rgba(251, 191, 36, 0.55)';  // Moderate soft amber
    if (multiplier <= 2.8) return 'rgba(251, 146, 60, 0.70)';  // Heavy soft orange
    return 'rgba(248, 113, 113, 0.85)';                        // Severe soft red
  }

  drawStreetRoutes(ctx, nodesById) {
    const routesToDraw = (this.detailedStreetRoutes && this.detailedStreetRoutes.length > 0)
      ? this.detailedStreetRoutes
      : this.buildFallbackStreetRoutes(this.routes);

    if (!routesToDraw || routesToDraw.length === 0) return;

    routesToDraw.forEach((vRoute, vIdx) => {
      if (!vRoute || vRoute.is_empty) return;

      const isIsolated = this.highlightVehicleIndex !== null;
      const isHighlighted = this.highlightVehicleIndex === vIdx;

      // If user highlighted a specific vehicle, dim others
      if (isIsolated && !isHighlighted) return;

      const color = this.vehicleColors[vIdx % this.vehicleColors.length];
      const coords = vRoute.street_coordinates;
      if (!coords || coords.length < 2) return;

      // 1. Base clean street path (soft, no blinding neon glow)
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = isHighlighted ? 4.5 : 3.2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      const firstPt = this.toScreen(coords[0][0], coords[0][1]);
      ctx.moveTo(firstPt.x, firstPt.y);

      for (let i = 1; i < coords.length; i++) {
        const pt = this.toScreen(coords[i][0], coords[i][1]);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();

      // 2. Animated pulse overlay moving forward along streets
      ctx.beginPath();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.8;
      ctx.setLineDash([6, 10]);
      ctx.lineDashOffset = -this.dashOffset;

      ctx.moveTo(firstPt.x, firstPt.y);
      for (let i = 1; i < coords.length; i++) {
        const pt = this.toScreen(coords[i][0], coords[i][1]);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      // 3. Draw Vehicle ID badge on route midpoint
      if (coords.length >= 3) {
        const midIdx = Math.floor(coords.length / 2);
        const midPt = this.toScreen(coords[midIdx][0], coords[midIdx][1]);
        ctx.save();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(midPt.x - 13, midPt.y - 8, 26, 16, 4);
        ctx.fill();
        ctx.strokeStyle = '#18181b';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.fillStyle = '#09090b';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`V${vIdx + 1}`, midPt.x, midPt.y + 3.5);
        ctx.restore();
      }
    });
  }

  drawAStarPath(ctx) {
    if (!this.astarResult || !this.astarResult.coordinates || this.astarResult.coordinates.length < 2) return;

    const coords = this.astarResult.coordinates;
    const astarColor = '#fbbf24'; // Soft warm gold

    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = astarColor;
    ctx.lineWidth = 4.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const p0 = this.toScreen(coords[0][0], coords[0][1]);
    ctx.moveTo(p0.x, p0.y);

    for (let i = 1; i < coords.length; i++) {
      const pt = this.toScreen(coords[i][0], coords[i][1]);
      ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();

    // Animated dash
    ctx.beginPath();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 10]);
    ctx.lineDashOffset = -this.dashOffset * 1.5;

    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < coords.length; i++) {
      const pt = this.toScreen(coords[i][0], coords[i][1]);
      ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  drawNodes(ctx) {
    for (const node of this.graph.nodes) {
      const pos = this.toScreen(node.x, node.y);
      const isSelected = this.selectedNodeId === node.id;
      const isAStarStart = this.astarStartId === node.id;
      const isAStarGoal = this.astarGoalId === node.id;

      if (isAStarStart || isAStarGoal) {
        // Special A* markers
        ctx.save();
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 13, 0, Math.PI * 2);
        ctx.fillStyle = isAStarStart ? '#10b981' : '#f59e0b';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(isAStarStart ? 'START' : 'GOAL', pos.x, pos.y + 3);
        ctx.restore();
        continue;
      }

      if (node.type === 'depot') {
        // Soft golden amber diamond
        ctx.save();
        ctx.translate(pos.x, pos.y);
        ctx.rotate(Math.PI / 4);

        ctx.fillStyle = '#f59e0b';
        ctx.fillRect(-9, -9, 18, 18);

        if (isSelected) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2.5;
          ctx.strokeRect(-12, -12, 24, 24);
        }
        ctx.restore();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('DEPOT', pos.x, pos.y - 14);

      } else if (node.type === 'customer') {
        // Customer target with demand badge
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 8.5, 0, Math.PI * 2);
        ctx.fillStyle = '#2563eb';
        ctx.fill();

        ctx.strokeStyle = isSelected ? '#ffffff' : '#93c5fd';
        ctx.lineWidth = isSelected ? 3.0 : 1.5;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(node.demand)}`, pos.x, pos.y + 3);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText(`C${node.id}`, pos.x, pos.y + 18);

      } else {
        // Intersection waypoint
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? '#38bdf8' : '#334155';
        ctx.fill();
      }
    }
  }
}
