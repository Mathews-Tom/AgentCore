#!/bin/bash
#
# Manual API testing scripts for reasoning.bounded_context JSON-RPC method
#
# Usage:
#   chmod +x test_scripts/reasoning_api_tests.sh
#   ./test_scripts/reasoning_api_tests.sh
#
# Prerequisites:
#   - Server running on http://localhost:8001
#   - curl and jq installed

BASE_URL="${BASE_URL:-http://localhost:8001}"
JSONRPC_ENDPOINT="${BASE_URL}/api/v1/jsonrpc"

echo "=============================================="
echo "Reasoning API Test Scripts"
echo "=============================================="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Simple reasoning request
echo "Test 1: Simple reasoning request"
echo "-----------------------------------"
curl -s -X POST "$JSONRPC_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.bounded_context",
    "params": {
      "query": "What is 2+2?",
      "temperature": 0.7,
      "chunk_size": 8192,
      "carryover_size": 4096,
      "max_iterations": 5
    },
    "id": 1
  }' | jq '.'
echo ""

# Test 2: Minimal parameters
echo "Test 2: Minimal parameters (only required fields)"
echo "-----------------------------------"
curl -s -X POST "$JSONRPC_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.bounded_context",
    "params": {
      "query": "Explain quantum entanglement in simple terms"
    },
    "id": 2
  }' | jq '.'
echo ""

# Test 3: Custom system prompt
echo "Test 3: Custom system prompt"
echo "-----------------------------------"
curl -s -X POST "$JSONRPC_ENDPOINT" \
  -H "Content-Type": application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.bounded_context",
    "params": {
      "query": "Write a haiku about coding",
      "system_prompt": "You are a creative poet who loves programming.",
      "temperature": 0.9
    },
    "id": 3
  }' | jq '.'
echo ""

# Test 4: A2A context
echo "Test 4: Request with A2A context"
echo "-----------------------------------"
curl -s -X POST "$JSONRPC_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.bounded_context",
    "params": {
      "query": "What are the benefits of agent-based systems?"
    },
    "id": 4,
    "a2a_context": {
      "source_agent": "test-agent-1",
      "target_agent": "reasoning-agent",
      "trace_id": "test-trace-123",
      "timestamp": "2025-01-16T00:00:00Z"
    }
  }' | jq '.'
echo ""

# Test 5: Error - invalid parameters
echo "Test 5: Error handling - invalid parameters"
echo "-----------------------------------"
curl -s -X POST "$JSONRPC_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.bounded_context",
    "params": {
      "query": "Test",
      "chunk_size": 4096,
      "carryover_size": 4096
    },
    "id": 5
  }' | jq '.'
echo ""

# Test 6: Error - missing required parameter
echo "Test 6: Error handling - missing required parameter"
echo "-----------------------------------"
curl -s -X POST "$JSONRPC_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "reasoning.bounded_context",
    "params": {},
    "id": 6
  }' | jq '.'
echo ""

# Test 7: Batch request
echo "Test 7: Batch request"
echo "-----------------------------------"
curl -s -X POST "$JSONRPC_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "jsonrpc": "2.0",
      "method": "reasoning.bounded_context",
      "params": {"query": "What is AI?"},
      "id": 7
    },
    {
      "jsonrpc": "2.0",
      "method": "reasoning.bounded_context",
      "params": {"query": "What is ML?"},
      "id": 8
    }
  ]' | jq '.'
echo ""

# Test 8: Introspection - list methods
echo "Test 8: Introspection - verify method registration"
echo "-----------------------------------"
curl -s -X POST "$JSONRPC_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "rpc.methods",
    "id": 9
  }' | jq '.result.methods | map(select(. == "reasoning.bounded_context"))'
echo ""

echo "=============================================="
echo "All tests completed"
echo "=============================================="
