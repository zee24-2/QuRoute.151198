/**
 * API client module for REST endpoints and live WebSocket solver streaming.
 */

const API = {
  baseUrl: window.location.origin,
  wsUrl: (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws/solve',

  async getCurrentGraph() {
    const res = await fetch(`${this.baseUrl}/api/graph/current`);
    return await res.json();
  },

  async generateGrid(params) {
    const res = await fetch(`${this.baseUrl}/api/graph/generate/grid`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async generateCity(params) {
    const res = await fetch(`${this.baseUrl}/api/graph/generate/city`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async generateSolomon(params) {
    const res = await fetch(`${this.baseUrl}/api/graph/generate/solomon`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async setBprAndBlackout(params) {
    const res = await fetch(`${this.baseUrl}/api/traffic/bpr`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async setSimTime(minutes) {
    const res = await fetch(`${this.baseUrl}/api/traffic/time`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ minutes }),
    });
    return await res.json();
  },

  async triggerIncident(params) {
    const res = await fetch(`${this.baseUrl}/api/traffic/incident`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async setManualEdge(params) {
    const res = await fetch(`${this.baseUrl}/api/traffic/edge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async editNode(nodeData) {
    const res = await fetch(`${this.baseUrl}/api/graph/node/edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(nodeData),
    });
    return await res.json();
  },

  async updateConfig(config) {
    const res = await fetch(`${this.baseUrl}/api/formulation/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return await res.json();
  },

  async runStatisticalBenchmark(params) {
    const res = await fetch(`${this.baseUrl}/api/benchmark/statistical`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async runScalabilitySweep(params) {
    const res = await fetch(`${this.baseUrl}/api/benchmark/scalability`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async inspectAStar(start_id, goal_id) {
    const res = await fetch(`${this.baseUrl}/api/route/astar_inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start_id, goal_id }),
    });
    return await res.json();
  },

  async quickSolve(params = {}) {
    const res = await fetch(`${this.baseUrl}/api/solve/quick`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  async liveReroute(params = {}) {
    const res = await fetch(`${this.baseUrl}/api/traffic/reroute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  },

  createSolverWebSocket(onMessage, onError, onClose) {
    const ws = new WebSocket(this.wsUrl);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (err) {
        console.error("WS parse error:", err);
      }
    };
    ws.onerror = (err) => {
      if (onError) onError(err);
    };
    ws.onclose = () => {
      if (onClose) onClose();
    };
    return ws;
  }
};
