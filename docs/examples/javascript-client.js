/**
 * AgentCore API Gateway - JavaScript/TypeScript Client Examples
 *
 * Demonstrates authentication, API calls, error handling, and WebSocket communication.
 */

class AgentCoreClient {
  /**
   * Create AgentCore API client.
   *
   * @param {Object} options - Client configuration
   * @param {string} options.baseUrl - API base URL (default: http://localhost:8080)
   * @param {string} options.username - Username for authentication
   * @param {string} options.password - Password for authentication
   */
  constructor({ baseUrl = 'http://localhost:8080', username, password } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.username = username;
    this.password = password;

    this.accessToken = null;
    this.refreshToken = null;
    this.tokenExpiresAt = null;
  }

  /**
   * Authenticate with username/password and obtain JWT tokens.
   *
   * @param {string} username - Username (optional if provided in constructor)
   * @param {string} password - Password (optional if provided in constructor)
   * @returns {Promise<Object>} Token response
   *
   * @example
   * const client = new AgentCoreClient();
   * const tokens = await client.authenticate('user', 'user123');
   * console.log(tokens.access_token);
   */
  async authenticate(username = this.username, password = this.password) {
    if (!username || !password) {
      throw new Error('Username and password required');
    }

    const response = await fetch(`${this.baseUrl}/auth/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grant_type: 'password',
        username,
        password,
        scope: 'user:read user:write agent:read agent:execute',
      }),
    });

    if (!response.ok) {
      await this._handleError(response);
    }

    const tokenData = await response.json();

    // Store tokens
    this.accessToken = tokenData.access_token;
    this.refreshToken = tokenData.refresh_token;

    // Calculate token expiration
    const expiresIn = tokenData.expires_in || 3600;
    this.tokenExpiresAt = new Date(Date.now() + expiresIn * 1000);

    return tokenData;
  }

  /**
   * Refresh access token using refresh token.
   *
   * @returns {Promise<Object>} New token response
   *
   * @example
   * const tokens = await client.refreshAccessToken();
   */
  async refreshAccessToken() {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${this.baseUrl}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        refresh_token: this.refreshToken,
      }),
    });

    if (!response.ok) {
      await this._handleError(response);
    }

    const tokenData = await response.json();

    // Update access token
    this.accessToken = tokenData.access_token;

    // Update expiration
    const expiresIn = tokenData.expires_in || 3600;
    this.tokenExpiresAt = new Date(Date.now() + expiresIn * 1000);

    return tokenData;
  }

  /**
   * Ensure access token is valid, refresh if needed.
   *
   * @private
   */
  async _ensureAuthenticated() {
    if (!this.accessToken) {
      await this.authenticate();
      return;
    }

    // Check if token is about to expire (within 5 minutes)
    const fiveMinutesFromNow = new Date(Date.now() + 5 * 60 * 1000);
    if (this.tokenExpiresAt && this.tokenExpiresAt <= fiveMinutesFromNow) {
      try {
        await this.refreshAccessToken();
      } catch (error) {
        // If refresh fails, re-authenticate
        await this.authenticate();
      }
    }
  }

  /**
   * Get HTTP headers with authentication.
   *
   * @private
   * @returns {Promise<Object>} Headers object
   */
  async _getHeaders() {
    await this._ensureAuthenticated();
    return {
      'Authorization': `Bearer ${this.accessToken}`,
      'Content-Type': 'application/json',
    };
  }

  /**
   * Handle API error responses.
   *
   * @private
   * @param {Response} response - Fetch response object
   */
  async _handleError(response) {
    let errorMessage;

    try {
      const errorData = await response.json();
      const error = errorData.error || {};
      const code = error.code || 'UNKNOWN_ERROR';
      const message = error.message || 'Unknown error';
      const details = error.details || {};
      const requestId = error.request_id;

      errorMessage = `${code}: ${message}`;
      if (Object.keys(details).length > 0) {
        errorMessage += ` | Details: ${JSON.stringify(details)}`;
      }
      if (requestId) {
        errorMessage += ` | Request ID: ${requestId}`;
      }
    } catch (e) {
      // Response is not JSON
      errorMessage = `HTTP ${response.status}: ${await response.text()}`;
    }

    throw new Error(errorMessage);
  }

  /**
   * Make GET request.
   *
   * @param {string} path - API path (e.g., "/auth/me")
   * @param {Object} options - Fetch options
   * @returns {Promise<Object>} Response JSON data
   *
   * @example
   * const user = await client.get('/auth/me');
   * console.log(user.username);
   */
  async get(path, options = {}) {
    const headers = await this._getHeaders();

    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'GET',
      headers: { ...headers, ...options.headers },
      ...options,
    });

    if (!response.ok) {
      await this._handleError(response);
    }

    return response.json();
  }

  /**
   * Make POST request.
   *
   * @param {string} path - API path
   * @param {Object} data - Request body data
   * @param {Object} options - Fetch options
   * @returns {Promise<Object>} Response JSON data
   *
   * @example
   * const result = await client.post('/agents', { name: 'MyAgent' });
   */
  async post(path, data = null, options = {}) {
    const headers = await this._getHeaders();

    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: { ...headers, ...options.headers },
      body: data ? JSON.stringify(data) : undefined,
      ...options,
    });

    if (!response.ok && response.status !== 201) {
      await this._handleError(response);
    }

    return response.text().then(text => text ? JSON.parse(text) : null);
  }

  /**
   * Get current authenticated user information.
   *
   * @returns {Promise<Object>} User information
   *
   * @example
   * const user = await client.getCurrentUser();
   * console.log(`Logged in as: ${user.username}`);
   */
  async getCurrentUser() {
    return this.get('/auth/me');
  }

  /**
   * List active sessions for current user.
   *
   * @returns {Promise<Array>} List of active sessions
   *
   * @example
   * const sessions = await client.listSessions();
   * for (const session of sessions) {
   *   console.log(`Session: ${session.session_id}`);
   * }
   */
  async listSessions() {
    const response = await this.get('/auth/sessions');
    return response.sessions || [];
  }

  /**
   * Logout and invalidate current session.
   *
   * @example
   * await client.logout();
   */
  async logout() {
    await this.post('/auth/logout');
    this.accessToken = null;
    this.refreshToken = null;
    this.tokenExpiresAt = null;
  }

  /**
   * Create WebSocket connection for real-time communication.
   *
   * @param {Object} options - WebSocket options
   * @param {Function} options.onMessage - Message handler
   * @param {Function} options.onError - Error handler
   * @param {Function} options.onClose - Close handler
   * @returns {Promise<WebSocket>} WebSocket connection
   *
   * @example
   * const ws = await client.connectWebSocket({
   *   onMessage: (data) => console.log('Received:', data),
   *   onError: (error) => console.error('Error:', error),
   * });
   */
  async connectWebSocket({ onMessage, onError, onClose } = {}) {
    await this._ensureAuthenticated();

    const wsUrl = this.baseUrl.replace(/^http/, 'ws');
    const ws = new WebSocket(`${wsUrl}/realtime/ws?token=${this.accessToken}`);

    ws.onopen = () => {
      console.log('WebSocket connected');

      // Send heartbeat ping every 25 seconds
      this._heartbeatInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 25000);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // Handle server ping
      if (data.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
        return;
      }

      if (onMessage) {
        onMessage(data);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) {
        onError(error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      if (this._heartbeatInterval) {
        clearInterval(this._heartbeatInterval);
      }
      if (onClose) {
        onClose();
      }
    };

    return ws;
  }
}

// Usage Examples

async function exampleBasicAuthentication() {
  console.log('Example: Basic Authentication');
  console.log('-'.repeat(50));

  // Create client and authenticate
  const client = new AgentCoreClient({
    username: 'user',
    password: 'user123',
  });

  // Get user information
  const user = await client.getCurrentUser();
  console.log(`Logged in as: ${user.username}`);
  console.log(`Email: ${user.email}`);
  console.log(`Roles: ${user.roles.join(', ')}`);

  // List active sessions
  const sessions = await client.listSessions();
  console.log(`\nActive sessions: ${sessions.length}`);
  for (const session of sessions) {
    console.log(`  - ${session.session_id} from ${session.ip_address}`);
  }

  // Logout
  await client.logout();
  console.log('\nLogged out successfully');
}

async function exampleErrorHandling() {
  console.log('\nExample: Error Handling');
  console.log('-'.repeat(50));

  const client = new AgentCoreClient();

  try {
    // Invalid credentials
    await client.authenticate('invalid', 'invalid');
  } catch (error) {
    console.log(`Authentication failed: ${error.message}`);
  }

  try {
    // Access protected endpoint without authentication
    client.accessToken = null;
    await client.getCurrentUser();
  } catch (error) {
    console.log(`Unauthorized access: ${error.message}`);
  }
}

async function exampleWebSocket() {
  console.log('\nExample: WebSocket Communication');
  console.log('-'.repeat(50));

  const client = new AgentCoreClient({
    username: 'user',
    password: 'user123',
  });

  const ws = await client.connectWebSocket({
    onMessage: (data) => {
      console.log('Received event:', data);
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    },
    onClose: () => {
      console.log('WebSocket connection closed');
    },
  });

  // Subscribe to events
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['agent.status_changed', 'agent.task_completed'],
  }));

  // Keep connection open for 30 seconds
  await new Promise(resolve => setTimeout(resolve, 30000));

  // Close connection
  ws.close();
}

// Run examples (Node.js)
if (typeof window === 'undefined') {
  (async () => {
    await exampleBasicAuthentication();
    await exampleErrorHandling();
    // await exampleWebSocket(); // Uncomment to test WebSocket
  })();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AgentCoreClient;
}
