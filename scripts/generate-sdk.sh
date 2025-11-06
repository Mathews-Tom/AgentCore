#!/usr/bin/env bash
#
# SDK Generation Script for AgentCore Gateway API
#
# Usage:
#   ./scripts/generate-sdk.sh <language> [openapi-spec-path]
#
# Examples:
#   ./scripts/generate-sdk.sh python
#   ./scripts/generate-sdk.sh typescript-axios
#   ./scripts/generate-sdk.sh java ./custom-openapi.json
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GENERATOR_VERSION="7.1.0"
OPENAPI_SPEC="${2:-openapi.json}"
OUTPUT_BASE_DIR="./generated-sdks"

# Supported languages
declare -A LANGUAGE_CONFIG=(
    ["python"]="packageName=agentcore_sdk,projectName=agentcore-sdk,packageVersion=1.0.0"
    ["typescript-axios"]="npmName=@agentcore/sdk,npmVersion=1.0.0,supportsES6=true"
    ["java"]="groupId=ai.agentcore,artifactId=agentcore-sdk,artifactVersion=1.0.0"
    ["go"]="packageName=agentcore,packageVersion=1.0.0"
    ["csharp"]="packageName=AgentCore.SDK,packageVersion=1.0.0"
    ["ruby"]="gemName=agentcore-sdk,gemVersion=1.0.0"
    ["php"]="packageName=AgentCore\\SDK,packageVersion=1.0.0"
    ["rust"]="packageName=agentcore-sdk,packageVersion=1.0.0"
)

function print_usage() {
    echo -e "${YELLOW}Usage:${NC} $0 <language> [openapi-spec-path]"
    echo ""
    echo "Supported languages:"
    for lang in "${!LANGUAGE_CONFIG[@]}"; do
        echo "  - $lang"
    done
    echo ""
    echo "Examples:"
    echo "  $0 python"
    echo "  $0 typescript-axios ./custom-openapi.json"
}

function error_exit() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit 1
}

function info() {
    echo -e "${GREEN}==>${NC} $1"
}

function warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# Check arguments
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

LANGUAGE="$1"

# Validate language
if [ -z "${LANGUAGE_CONFIG[$LANGUAGE]}" ]; then
    error_exit "Unsupported language: $LANGUAGE. Run with no arguments to see supported languages."
fi

# Check if OpenAPI spec exists
if [ ! -f "$OPENAPI_SPEC" ]; then
    warn "OpenAPI spec not found at $OPENAPI_SPEC"
    info "Attempting to download from running gateway..."

    # Try to download from localhost
    if curl -f -s http://localhost:8080/openapi.json > "$OPENAPI_SPEC"; then
        info "OpenAPI spec downloaded successfully"
    else
        error_exit "Could not find or download OpenAPI spec. Ensure gateway is running or provide spec path."
    fi
fi

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/$LANGUAGE"
mkdir -p "$OUTPUT_DIR"

info "Generating $LANGUAGE SDK..."
info "OpenAPI Spec: $OPENAPI_SPEC"
info "Output Directory: $OUTPUT_DIR"

# Check if Docker is available
if command -v docker &> /dev/null; then
    info "Using Docker to generate SDK"

    docker run --rm \
        -v "${PWD}:/local" \
        openapitools/openapi-generator-cli:v${GENERATOR_VERSION} generate \
        -i "/local/$OPENAPI_SPEC" \
        -g "$LANGUAGE" \
        -o "/local/$OUTPUT_DIR" \
        --additional-properties="${LANGUAGE_CONFIG[$LANGUAGE]}" \
        --skip-validate-spec

elif command -v openapi-generator &> /dev/null; then
    info "Using locally installed openapi-generator"

    openapi-generator generate \
        -i "$OPENAPI_SPEC" \
        -g "$LANGUAGE" \
        -o "$OUTPUT_DIR" \
        --additional-properties="${LANGUAGE_CONFIG[$LANGUAGE]}" \
        --skip-validate-spec
else
    error_exit "Neither Docker nor openapi-generator found. Please install one of them."
fi

# Post-generation steps
info "SDK generated successfully!"
echo ""
echo -e "${GREEN}Next steps:${NC}"

case "$LANGUAGE" in
    python)
        echo "  cd $OUTPUT_DIR"
        echo "  pip install -e ."
        echo "  python -c 'import agentcore_sdk; print(agentcore_sdk.__version__)'"
        ;;
    typescript-axios)
        echo "  cd $OUTPUT_DIR"
        echo "  npm install"
        echo "  npm run build"
        ;;
    java)
        echo "  cd $OUTPUT_DIR"
        echo "  mvn clean install"
        ;;
    go)
        echo "  cd $OUTPUT_DIR"
        echo "  go mod init agentcore-sdk"
        echo "  go mod tidy"
        echo "  go build ./..."
        ;;
    *)
        echo "  cd $OUTPUT_DIR"
        echo "  # Follow language-specific build instructions in README.md"
        ;;
esac

echo ""
echo -e "${GREEN}Documentation:${NC} $OUTPUT_DIR/README.md"
echo -e "${GREEN}Examples:${NC} $OUTPUT_DIR/docs/"
