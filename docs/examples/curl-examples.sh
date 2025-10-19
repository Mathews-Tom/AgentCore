#!/bin/bash
#
# AgentCore API Gateway - cURL Examples
#
# Demonstrates common API operations using cURL.
# Usage: bash curl-examples.sh

set -e

BASE_URL="${GATEWAY_URL:-http://localhost:8080}"
USERNAME="${GATEWAY_USERNAME:-user}"
PASSWORD="${GATEWAY_PASSWORD:-user123}"

echo "==================================="
echo "AgentCore API Gateway - cURL Examples"
echo "==================================="
echo "Base URL: $BASE_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. Authentication - Password Grant
echo -e "${BLUE}1. Authenticate with username/password${NC}"
echo "-----------------------------------"

TOKEN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d "{
    \"grant_type\": \"password\",
    \"username\": \"$USERNAME\",
    \"password\": \"$PASSWORD\",
    \"scope\": \"user:read user:write agent:read agent:execute\"
  }")

echo "Response: $TOKEN_RESPONSE"
echo ""

# Extract access token and refresh token
ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
REFRESH_TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"refresh_token":"[^"]*' | cut -d'"' -f4)

if [ -z "$ACCESS_TOKEN" ]; then
  echo "Error: Failed to obtain access token"
  exit 1
fi

echo -e "${GREEN}Access token obtained successfully${NC}"
echo ""

# 2. Get Current User
echo -e "${BLUE}2. Get current user information${NC}"
echo "-----------------------------------"

curl -s -X GET "$BASE_URL/auth/me" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq '.'

echo ""

# 3. List Active Sessions
echo -e "${BLUE}3. List active sessions${NC}"
echo "-----------------------------------"

curl -s -X GET "$BASE_URL/auth/sessions" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq '.'

echo ""

# 4. OAuth - List Providers
echo -e "${BLUE}4. List OAuth providers${NC}"
echo "-----------------------------------"

curl -s -X GET "$BASE_URL/oauth/providers" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq '.'

echo ""

# 5. OAuth - List Scopes
echo -e "${BLUE}5. List OAuth scopes${NC}"
echo "-----------------------------------"

curl -s -X GET "$BASE_URL/oauth/scopes" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq '.'

echo ""

# 6. Health Check
echo -e "${BLUE}6. Health check (no auth required)${NC}"
echo "-----------------------------------"

curl -s -X GET "$BASE_URL/health" | jq '.'

echo ""

# 7. Readiness Check
echo -e "${BLUE}7. Readiness check (no auth required)${NC}"
echo "-----------------------------------"

curl -s -X GET "$BASE_URL/health/ready" | jq '.'

echo ""

# 8. Refresh Token
echo -e "${BLUE}8. Refresh access token${NC}"
echo "-----------------------------------"

NEW_TOKEN_RESPONSE=$(curl -s -X POST "$BASE_URL/auth/refresh" \
  -H "Content-Type: application/json" \
  -d "{
    \"refresh_token\": \"$REFRESH_TOKEN\"
  }")

echo "Response: $NEW_TOKEN_RESPONSE"
echo ""

NEW_ACCESS_TOKEN=$(echo "$NEW_TOKEN_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ -z "$NEW_ACCESS_TOKEN" ]; then
  echo "Warning: Failed to refresh token"
else
  echo -e "${GREEN}Token refreshed successfully${NC}"
  ACCESS_TOKEN="$NEW_ACCESS_TOKEN"
fi

echo ""

# 9. Logout
echo -e "${BLUE}9. Logout and invalidate session${NC}"
echo "-----------------------------------"

curl -s -X POST "$BASE_URL/auth/logout" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq '.'

echo ""
echo -e "${GREEN}Examples completed successfully!${NC}"

# Additional Examples (commented out)

# Service Account Authentication
: <<'COMMENT'
curl -X POST "$BASE_URL/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "client_credentials",
    "client_id": "service",
    "client_secret": "service123",
    "scope": "service:read service:write"
  }'
COMMENT

# OAuth Authorization Flow (browser-based)
: <<'COMMENT'
# Step 1: Get authorization URL
AUTH_URL="$BASE_URL/oauth/authorize/google?scope=user:read user:write&redirect_after_login=/dashboard"
echo "Open this URL in browser: $AUTH_URL"

# Step 2: After callback, exchange code for token
# This happens automatically in the gateway
COMMENT

# WebSocket Connection (using websocat)
: <<'COMMENT'
# Install websocat: brew install websocat
websocat "ws://localhost:8080/realtime/ws?token=$ACCESS_TOKEN"

# Send message
echo '{"type":"subscribe","channels":["agent.status_changed"]}' | websocat "ws://localhost:8080/realtime/ws?token=$ACCESS_TOKEN"
COMMENT

# Server-Sent Events (SSE)
: <<'COMMENT'
curl -N -H "Authorization: Bearer $ACCESS_TOKEN" \
  "$BASE_URL/realtime/sse"
COMMENT

# Rate Limit Testing
: <<'COMMENT'
# Send multiple requests to trigger rate limit
for i in {1..100}; do
  curl -s -X GET "$BASE_URL/auth/me" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -w "Status: %{http_code}\n"
  sleep 0.1
done
COMMENT

# Error Handling Examples
: <<'COMMENT'
# 401 Unauthorized
curl -X GET "$BASE_URL/auth/me" \
  -H "Authorization: Bearer invalid_token"

# 403 Forbidden
curl -X POST "$BASE_URL/admin/users" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username":"newuser"}'

# 404 Not Found
curl -X DELETE "$BASE_URL/auth/sessions/invalid-session-id" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# 429 Rate Limit Exceeded
# Run the rate limit testing example above
COMMENT
