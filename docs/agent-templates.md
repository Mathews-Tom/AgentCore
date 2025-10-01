# Agent Template Library

## Overview

Pre-built agent configurations for common development and operational patterns. This library provides ready-to-use agent templates that reduce time-to-value and demonstrate best practices for AgentCore integration.

**Purpose:** Accelerate agent development by providing tested, production-ready configurations for common use cases.

**Target Users:**

- Developers building multi-agent systems
- DevOps teams automating workflows
- Data engineers creating analysis pipelines
- Platform teams standardizing agent patterns

## Template Catalog

### 1. Code Analysis Agent

**Purpose:** Static code analysis, linting, complexity metrics, security scanning

**Capabilities:** `python`, `javascript`, `typescript`, `linting`, `metrics`, `security-scanning`

**Cost:** Low (no LLM usage, static analysis tools only)

**Example Use Cases:**

- Pre-commit code quality checks
- Technical debt analysis
- Security vulnerability scanning
- Code complexity monitoring

**Configuration:**

```json
{
  "agent_id": "code-analyzer-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Code Analysis Agent",
    "capabilities": ["python", "javascript", "typescript", "linting", "metrics", "security-scanning"],
    "supported_interactions": ["task_execution"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8100",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["pylint", "eslint", "radon", "bandit"],
    "memory": "512MB",
    "cpu": "0.5"
  },
  "cost_per_request": 0.01
}
```

**Example Workflow Integration:**

```yaml
tasks:
  - id: "analyze_code"
    agent: "code-analyzer"
    input:
      repository: "https://github.com/example/repo"
      files: ["src/**/*.py"]
    output_requirements:
      - "linting_report"
      - "complexity_metrics"
      - "security_issues"
```

---

### 2. Research Agent

**Purpose:** Web search, information gathering, summarization, fact-checking

**Capabilities:** `web-search`, `summarization`, `fact-checking`, `citation-extraction`

**Cost:** Medium (LLM for summarization, ~$0.02 per request)

**Example Use Cases:**

- Competitive intelligence gathering
- Technical documentation research
- Market trend analysis
- Academic literature review

**Configuration:**

```json
{
  "agent_id": "research-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Research Agent",
    "capabilities": ["web-search", "summarization", "fact-checking", "citation-extraction"],
    "supported_interactions": ["task_execution", "streaming"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8101",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["serpapi", "openai"],
    "memory": "1GB",
    "cpu": "1.0"
  },
  "cost_per_request": 0.02,
  "latency_p95_ms": 3000
}
```

**Example Workflow Integration:**

```yaml
tasks:
  - id: "research_topic"
    agent: "research-agent"
    input:
      query: "Latest trends in agentic AI frameworks"
      depth: "comprehensive"
      sources: ["arxiv", "github", "blogs"]
    output_requirements:
      - "summary"
      - "key_findings"
      - "citations"
```

---

### 3. Testing Agent

**Purpose:** Test generation, execution, coverage analysis, regression testing

**Capabilities:** `test-generation`, `test-execution`, `pytest`, `jest`, `coverage-analysis`

**Cost:** Medium (LLM for test generation, ~$0.03 per request)

**Example Use Cases:**

- Automated test suite generation
- Regression test execution
- Code coverage analysis
- Test case validation

**Configuration:**

```json
{
  "agent_id": "testing-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Testing Agent",
    "capabilities": ["test-generation", "test-execution", "pytest", "jest", "coverage-analysis"],
    "supported_interactions": ["task_execution", "streaming"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8102",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["pytest", "jest", "coverage", "openai"],
    "memory": "2GB",
    "cpu": "2.0"
  },
  "cost_per_request": 0.03
}
```

**Example Workflow Integration:**

```yaml
tasks:
  - id: "generate_tests"
    agent: "testing-agent"
    input:
      source_files: ["src/services/*.py"]
      test_framework: "pytest"
      coverage_target: 90
    output_requirements:
      - "test_files"
      - "coverage_report"
```

---

### 4. Documentation Agent

**Purpose:** README generation, API documentation, code comments, technical writing

**Capabilities:** `documentation`, `markdown`, `openapi`, `code-analysis`, `technical-writing`

**Cost:** Medium (LLM for content generation, ~$0.04 per request)

**Example Use Cases:**

- README and documentation generation
- OpenAPI spec creation
- Code comment enhancement
- User guide writing

**Configuration:**

```json
{
  "agent_id": "documentation-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Documentation Agent",
    "capabilities": ["documentation", "markdown", "openapi", "code-analysis", "technical-writing"],
    "supported_interactions": ["task_execution"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8103",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["openai", "pandoc"],
    "memory": "1GB",
    "cpu": "1.0"
  },
  "cost_per_request": 0.04
}
```

---

### 5. Deployment Agent

**Purpose:** CI/CD orchestration, infrastructure deployment, container management

**Capabilities:** `docker`, `kubernetes`, `terraform`, `ci-cd`, `github-actions`

**Cost:** Low (no LLM, infrastructure tools only)

**Example Use Cases:**

- Automated deployment pipelines
- Infrastructure as code management
- Container orchestration
- Release automation

**Configuration:**

```json
{
  "agent_id": "deployment-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Deployment Agent",
    "capabilities": ["docker", "kubernetes", "terraform", "ci-cd", "github-actions"],
    "supported_interactions": ["task_execution", "streaming"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8104",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["docker", "kubectl", "terraform"],
    "memory": "1GB",
    "cpu": "1.0"
  },
  "cost_per_request": 0.005
}
```

---

### 6. Code Review Agent

**Purpose:** Pull request review, best practices enforcement, security analysis

**Capabilities:** `code-review`, `git`, `best-practices`, `security`, `diff-analysis`

**Cost:** Medium (LLM for review comments, ~$0.05 per PR)

**Example Use Cases:**

- Automated PR review
- Security vulnerability detection
- Code style enforcement
- Best practices validation

**Configuration:**

```json
{
  "agent_id": "code-review-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Code Review Agent",
    "capabilities": ["code-review", "git", "best-practices", "security", "diff-analysis"],
    "supported_interactions": ["task_execution"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8105",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["git", "openai", "semgrep"],
    "memory": "2GB",
    "cpu": "1.5"
  },
  "cost_per_request": 0.05
}
```

---

### 7. Refactoring Agent

**Purpose:** Code improvement, design pattern application, technical debt reduction

**Capabilities:** `refactoring`, `design-patterns`, `code-optimization`, `ast-analysis`

**Cost:** High (LLM for code transformation, ~$0.10 per file)

**Example Use Cases:**

- Legacy code modernization
- Design pattern application
- Performance optimization
- Code smell elimination

**Configuration:**

```json
{
  "agent_id": "refactoring-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Refactoring Agent",
    "capabilities": ["refactoring", "design-patterns", "code-optimization", "ast-analysis"],
    "supported_interactions": ["task_execution", "streaming"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8106",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["openai", "ast", "rope"],
    "memory": "4GB",
    "cpu": "2.0"
  },
  "cost_per_request": 0.10
}
```

---

### 8. Data Analysis Agent

**Purpose:** Data processing, statistical analysis, visualization generation

**Capabilities:** `pandas`, `numpy`, `data-viz`, `statistics`, `data-cleaning`

**Cost:** Low-Medium (computational, optional LLM for insights, ~$0.02 per analysis)

**Example Use Cases:**

- Dataset analysis and profiling
- Statistical modeling
- Data visualization
- Insight generation

**Configuration:**

```json
{
  "agent_id": "data-analysis-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Data Analysis Agent",
    "capabilities": ["pandas", "numpy", "data-viz", "statistics", "data-cleaning"],
    "supported_interactions": ["task_execution"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8107",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["pandas", "numpy", "matplotlib", "scikit-learn"],
    "memory": "8GB",
    "cpu": "4.0"
  },
  "cost_per_request": 0.02
}
```

---

### 9. API Integration Agent

**Purpose:** External API integration, webhook handling, data synchronization

**Capabilities:** `rest-api`, `graphql`, `webhooks`, `authentication`, `rate-limiting`

**Cost:** Low (no LLM, API calls only)

**Example Use Cases:**

- Third-party service integration
- Webhook endpoint creation
- API orchestration
- Data pipeline automation

**Configuration:**

```json
{
  "agent_id": "api-integration-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "API Integration Agent",
    "capabilities": ["rest-api", "graphql", "webhooks", "authentication", "rate-limiting"],
    "supported_interactions": ["task_execution", "streaming"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8108",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["requests", "httpx", "fastapi"],
    "memory": "512MB",
    "cpu": "0.5"
  },
  "cost_per_request": 0.005
}
```

---

### 10. Monitoring Agent

**Purpose:** System health monitoring, alerting, log analysis, performance tracking

**Capabilities:** `monitoring`, `alerting`, `log-analysis`, `prometheus`, `grafana`

**Cost:** Low (infrastructure monitoring, ~$0.01 per check)

**Example Use Cases:**

- System health monitoring
- Log aggregation and analysis
- Alert management
- Performance tracking

**Configuration:**

```json
{
  "agent_id": "monitoring-001",
  "agent_card": {
    "schema_version": "0.2",
    "agent_name": "Monitoring Agent",
    "capabilities": ["monitoring", "alerting", "log-analysis", "prometheus", "grafana"],
    "supported_interactions": ["task_execution", "streaming"],
    "authentication": {
      "type": "jwt",
      "requirements": {}
    },
    "endpoints": [{
      "url": "http://localhost:8109",
      "type": "http"
    }]
  },
  "requirements": {
    "tools": ["prometheus", "grafana", "elasticsearch"],
    "memory": "1GB",
    "cpu": "1.0"
  },
  "cost_per_request": 0.01
}
```

---

## How to Use Templates

### 1. Register Template Agent

Using the AgentCore CLI:

```bash
# Register with default configuration
agentcore agent register \
  --template code-analyzer \
  --name "my-code-analyzer"

# Register with customization
agentcore agent register \
  --template code-analyzer \
  --name "my-code-analyzer" \
  --customize '{"languages": ["python", "typescript"], "tools": ["pylint", "mypy"]}'
```

Using the JSON-RPC API:

```bash
curl -X POST http://localhost:8001/api/v1/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "agent.register",
    "params": {
      "agent_id": "my-code-analyzer",
      "agent_card": { ... }
    },
    "id": 1
  }'
```

### 2. Customize Template

**Modify Capabilities:**

```json
{
  "capabilities": ["python", "typescript", "linting", "type-checking"]
}
```

**Adjust Resource Requirements:**

```json
{
  "requirements": {
    "memory": "1GB",
    "cpu": "1.0",
    "gpu": false
  }
}
```

**Configure Specific Tools:**

```json
{
  "tools": {
    "linter": "ruff",
    "type_checker": "mypy",
    "formatter": "black"
  }
}
```

**Set Cost Parameters:**

```json
{
  "cost_per_request": 0.02,
  "latency_p95_ms": 500,
  "quality_score": 0.95
}
```

### 3. Deploy Template

**Using Agent Runtime:**

```yaml
# docker-compose.yml
services:
  code-analyzer:
    image: agentcore/agent-runtime:latest
    environment:
      - AGENT_TEMPLATE=code-analyzer
      - AGENT_CONFIG=/config/agent.json
    volumes:
      - ./config:/config
    ports:
      - "8100:8000"
```

**Configure Security Policies:**

```yaml
security:
  authentication: jwt
  rate_limiting:
    max_requests_per_minute: 100
  allowed_operations:
    - "file_read"
    - "process_execute"
```

**Set Up Monitoring:**

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
  logging:
    level: INFO
    format: json
```

**Define Orchestration Patterns:**

```yaml
workflow:
  name: "code_quality_pipeline"
  pattern: "supervisor"
  agents:
    - code-analyzer
    - testing-agent
    - documentation-agent
```

---

## Template Development Guide

### Creating Custom Templates

**Step 1: Identify Common Use Case**

- Analyze recurring workflow patterns
- Identify frequently used capabilities
- Survey user requirements
- Evaluate cost vs. value

**Step 2: Define Required Capabilities**

```json
{
  "capabilities": [
    "primary_capability",
    "supporting_capability_1",
    "supporting_capability_2"
  ],
  "capability_metadata": {
    "primary_capability": {
      "description": "Core functionality",
      "version": "1.0",
      "providers": ["openai", "anthropic"]
    }
  }
}
```

**Step 3: Specify Tool Dependencies**

```json
{
  "tools": {
    "required": ["tool1", "tool2"],
    "optional": ["tool3"],
    "versions": {
      "tool1": ">=2.0.0",
      "tool2": "^1.5.0"
    }
  }
}
```

**Step 4: Document Configuration Options**

```markdown
## Configuration Options

### Required
- `capability_1`: Description and valid values
- `endpoint`: Agent HTTP endpoint

### Optional
- `timeout`: Request timeout (default: 30s)
- `retry_policy`: Retry configuration (default: 3 retries)
```

**Step 5: Provide Example Usage**

```yaml
# Example workflow using custom template
workflow:
  name: "custom_workflow"
  tasks:
    - id: "process"
      agent: "custom-template"
      input:
        param1: "value1"
      output_requirements:
        - "result"
```

**Step 6: Test with Real Workflows**

```bash
# Unit tests
pytest tests/templates/test_custom_template.py

# Integration tests
pytest tests/integration/test_custom_workflow.py

# Load tests
locust -f tests/load/test_custom_template.py
```

---

## Template Best Practices

### Design Principles

1. **Single Responsibility**: Each template should have one primary purpose
2. **Clear Capabilities**: Document all capabilities with examples
3. **Sensible Defaults**: Provide defaults that work for 80% of use cases
4. **Cost Transparency**: Include cost estimates per operation
5. **Error Handling**: Define failure modes and recovery strategies
6. **Environment Agnostic**: Work across dev, staging, and production

### Configuration Guidelines

- **Keep templates focused**: Avoid combining unrelated capabilities
- **Document all capabilities**: Include descriptions, versions, and providers
- **Provide sensible defaults**: Most users should not need customization
- **Include cost estimates**: Help users understand operational costs
- **Add error handling patterns**: Define retry policies and failure recovery
- **Test across environments**: Validate in dev, staging, and production

### Resource Management

```json
{
  "resources": {
    "memory": {
      "min": "512MB",
      "recommended": "1GB",
      "max": "2GB"
    },
    "cpu": {
      "min": 0.5,
      "recommended": 1.0,
      "max": 2.0
    },
    "scaling": {
      "min_instances": 1,
      "max_instances": 10,
      "target_cpu_utilization": 70
    }
  }
}
```

### Security Considerations

- Validate all inputs
- Implement authentication and authorization
- Use environment variables for secrets
- Enable audit logging
- Follow least privilege principle
- Implement rate limiting

### Testing Requirements

- **Unit Tests**: 90%+ code coverage
- **Integration Tests**: Full workflow validation
- **Performance Tests**: Latency and throughput benchmarks
- **Security Tests**: Vulnerability scanning
- **Load Tests**: 100+ concurrent requests

---

## Template Contribution Guidelines

### How to Contribute

1. **Fork Repository**: Fork AgentCore repository on GitHub
2. **Create Template**: Develop template following best practices
3. **Add Documentation**: Complete README with usage examples
4. **Write Tests**: Include unit, integration, and performance tests
5. **Submit PR**: Open pull request with template details

### Review Criteria

- ✅ Solves common use case
- ✅ Follows template structure
- ✅ Includes complete documentation
- ✅ Has 90%+ test coverage
- ✅ Demonstrates value with examples
- ✅ Passes security review

### Community Templates

Browse community-contributed templates at:
**<https://github.com/agentcore/agent-templates>**

---

## Template Roadmap

### Planned Templates (Q1 2026)

- **Database Migration Agent**: Schema migrations, data transformations
- **Content Generation Agent**: Blog posts, marketing copy, social media
- **Translation Agent**: Multi-language content translation
- **Video Processing Agent**: Video analysis, transcription, editing
- **Infrastructure Provisioning Agent**: Cloud resource management

### Community Requests

Submit template requests via:

- GitHub Issues: <https://github.com/agentcore/agentcore/issues>
- Discord: <https://discord.gg/agentcore>
- Email: <templates@agentcore.io>

---

## Support and Resources

### Documentation

- **AgentCore Docs**: <https://docs.agentcore.io>
- **Template API Reference**: <https://docs.agentcore.io/templates>
- **Tutorial Videos**: <https://youtube.com/@agentcore>

### Community

- **Discord**: <https://discord.gg/agentcore>
- **GitHub Discussions**: <https://github.com/agentcore/agentcore/discussions>
- **Stack Overflow**: Tag `agentcore`

### Enterprise Support

- **Email**: <support@agentcore.io>
- **Custom Templates**: Contact <sales@agentcore.io>
- **SLA Options**: <https://agentcore.io/enterprise>
