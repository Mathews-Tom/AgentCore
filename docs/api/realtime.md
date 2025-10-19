# Real-time Communication Guide

AgentCore API Gateway supports real-time communication through WebSocket and Server-Sent Events (SSE).

## Overview

- **WebSocket:** Bidirectional communication for interactive applications
- **SSE (Server-Sent Events):** Server-to-client streaming for real-time updates

## WebSocket

### Connection

Establish WebSocket connection with authentication:

```javascript
const ws = new WebSocket(
  'ws://localhost:8080/realtime/ws?token=YOUR_ACCESS_TOKEN'
);

ws.onopen = () => {
  console.log('Connected to AgentCore');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from AgentCore');
};
```

### Authentication

Include JWT token in connection:
- **Query Parameter:** `?token=YOUR_ACCESS_TOKEN`
- **Header:** `Authorization: Bearer YOUR_ACCESS_TOKEN` (if supported by client)

### Message Format

All messages use JSON format:

```json
{
  "type": "event",
  "event": "agent.status_changed",
  "data": {
    "agent_id": "agent-123",
    "status": "running",
    "timestamp": "2025-10-18T10:30:00Z"
  },
  "timestamp": "2025-10-18T10:30:00Z"
}
```

### Subscribing to Events

Subscribe to specific event types:

```javascript
// Subscribe to agent events
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['agent.status_changed', 'agent.task_completed']
}));
```

### Heartbeat

Connection heartbeat keeps connection alive:

```javascript
// Server sends ping every 30 seconds
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'ping') {
    // Respond with pong
    ws.send(JSON.stringify({ type: 'pong' }));
  }
};
```

### Event Types

| Event | Description |
|-------|-------------|
| `agent.status_changed` | Agent status update |
| `agent.task_completed` | Agent task completion |
| `agent.error` | Agent error occurred |
| `task.progress` | Task progress update |
| `system.notification` | System notification |

## Server-Sent Events (SSE)

### Connection

Connect to SSE endpoint:

```javascript
const eventSource = new EventSource(
  'http://localhost:8080/realtime/sse?token=YOUR_ACCESS_TOKEN'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

eventSource.onerror = (error) => {
  console.error('SSE error:', error);
  eventSource.close();
};
```

### Event Stream Format

SSE events follow standard format:

```
event: agent.status_changed
data: {"agent_id":"agent-123","status":"running"}

event: keepalive
data: {"timestamp":"2025-10-18T10:30:00Z"}
```

### Keepalive

Server sends keepalive events every 30 seconds to maintain connection.

### Python Client Example

```python
import sseclient
import requests

url = 'http://localhost:8080/realtime/sse'
headers = {'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}

response = requests.get(url, headers=headers, stream=True)
client = sseclient.SSEClient(response)

for event in client.events():
    print(f'Event: {event.event}')
    print(f'Data: {event.data}')
```

## Connection Limits

- **Maximum Concurrent Connections:** 10,000 per instance
- **Connection Timeout:** 5 minutes of inactivity
- **Heartbeat Interval:** 30 seconds
- **Reconnection:** Automatic with exponential backoff

## Best Practices

### 1. Implement Reconnection Logic

```javascript
class ReconnectingWebSocket {
  constructor(url) {
    this.url = url;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.connect();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('Connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onclose = () => {
      this.reconnect();
    };

    this.ws.onerror = (error) => {
      console.error('Error:', error);
    };
  }

  reconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    console.log(`Reconnecting in ${delay}ms...`);

    setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }
}
```

### 2. Handle Authentication Expiry

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'error' && data.code === 'TOKEN_EXPIRED') {
    // Refresh token and reconnect
    refreshToken().then(newToken => {
      ws.close();
      connectWithNewToken(newToken);
    });
  }
};
```

### 3. Filter Events Client-Side

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  // Only process events for specific agent
  if (data.event === 'agent.status_changed' && data.data.agent_id === 'my-agent') {
    handleAgentStatus(data.data);
  }
};
```

## Next Steps

- [Authentication Guide](authentication.md)
- [Rate Limiting Guide](rate-limiting.md)
- [Error Handling](errors.md)
- [Code Examples](../examples/)
