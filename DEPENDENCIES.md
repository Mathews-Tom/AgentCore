# Dependency Management

This document defines the version management strategy for AgentCore dependencies, with special focus on LLM provider SDKs.

## Version Pinning Strategy

AgentCore uses **exact version pinning** (`==`) for critical LLM provider SDKs to prevent unexpected breaking changes and ensure reproducible builds. All other dependencies use minimum version constraints (`>=`) to allow compatible updates.

## LLM Provider SDK Versions

### Current Pinned Versions

| Provider | Version | Pinned Date | Reason |
|----------|---------|-------------|---------|
| OpenAI | `1.54.0` | 2025-10-26 | Stable API, proven compatibility with streaming, function calling, and JSON mode |
| Anthropic | `0.40.0` | 2025-10-26 | Last stable version before 0.50.0 breaking changes, supports Claude 3 models |
| Google GenAI | `0.2.0` | 2025-10-26 | Stable API for Gemini models, minimal breaking changes expected |

### Version Selection Criteria

Pinned versions are chosen based on:

1. **API Stability**: No known breaking changes in minor/patch updates
2. **Feature Completeness**: Supports all required features (streaming, function calling, async operations)
3. **Bug Fixes**: Critical bugs resolved in this version
4. **Community Adoption**: Wide usage in production environments
5. **Security**: No known vulnerabilities

### Known Issues

#### OpenAI 1.54.0
- ✅ No known critical issues
- ✅ Supports all required features (streaming, async, function calling, JSON mode)
- ✅ Compatible with Python 3.12+

#### Anthropic 0.40.0
- ⚠️ Version 0.50.0+ introduces breaking changes to streaming API
- ✅ Supports Claude 3 family (Opus, Sonnet, Haiku)
- ✅ Compatible with async operations and tool use

#### Google GenAI 0.2.0
- ⚠️ Newer versions (0.8.0+) have different async API patterns
- ✅ Supports Gemini Pro and Gemini Pro Vision
- ⚠️ Limited function calling support compared to other providers

## Compatibility Matrix

### Python Version Compatibility

| SDK | Python 3.12 | Python 3.13 | Notes |
|-----|-------------|-------------|-------|
| OpenAI 1.54.0 | ✅ | ✅ | Full support |
| Anthropic 0.40.0 | ✅ | ✅ | Full support |
| Google GenAI 0.2.0 | ✅ | ⚠️ | Limited testing on 3.13 |

### Feature Compatibility

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Streaming | ✅ 1.54.0+ | ✅ 0.40.0+ | ✅ 0.2.0+ |
| Async Operations | ✅ 1.54.0+ | ✅ 0.40.0+ | ✅ 0.2.0+ |
| Function Calling | ✅ 1.54.0+ | ✅ 0.40.0+ | ⚠️ Limited |
| JSON Mode | ✅ 1.54.0+ | ❌ | ❌ |
| Vision Models | ✅ 1.54.0+ | ✅ 0.40.0+ | ✅ 0.2.0+ |

### Breaking Changes to Watch

**OpenAI SDK:**
- `2.0.0`: Major API redesign, switched to async-first patterns
- `1.0.0`: Breaking changes to response models

**Anthropic SDK:**
- `0.50.0`: Streaming API redesign (breaking)
- `0.40.0`: Tool use API changes (minor)

**Google GenAI SDK:**
- `0.8.0`: Async API changes (breaking)
- `0.2.0`: Initial stable release

## Upgrade Procedure

### Monthly Review Process

**Schedule:** First Monday of each month

**Review Checklist:**

1. **Check for new versions**
   ```bash
   uv pip list --outdated | grep -E "(openai|anthropic|google-generativeai)"
   ```

2. **Review changelogs**
   - OpenAI: https://github.com/openai/openai-python/releases
   - Anthropic: https://github.com/anthropics/anthropic-sdk-python/releases
   - Google: https://github.com/google/generative-ai-python/releases

3. **Assess breaking changes**
   - Review migration guides
   - Check for deprecated features we use
   - Identify API changes affecting our code

4. **Security check**
   ```bash
   uv pip audit
   ```

### Upgrade Testing Procedure

**Before upgrading any SDK version:**

1. **Create upgrade branch**
   ```bash
   git checkout -b upgrade/provider-sdk-YYYY-MM
   ```

2. **Update single provider**
   ```toml
   # pyproject.toml - Example: OpenAI upgrade
   "openai==1.55.0",  # New version
   ```

3. **Install and verify**
   ```bash
   uv sync
   uv pip list | grep openai
   ```

4. **Run integration tests**
   ```bash
   # Provider-specific tests
   uv run pytest tests/integration/llm_client/test_openai_provider.py -v

   # Full integration suite
   uv run pytest tests/integration/llm_client/ -v

   # Run with coverage requirement
   uv run pytest --cov=src/agentcore/llm_client --cov-fail-under=90
   ```

5. **Run benchmarks**
   ```bash
   uv run python scripts/benchmark_llm.py --provider openai
   ```

6. **Validate SLOs**
   - P50 latency < 500ms
   - P95 latency < 2000ms
   - P99 latency < 5000ms
   - Error rate < 1%
   - Rate limit handling functional

7. **Manual testing**
   - Test streaming responses
   - Test function calling
   - Test error handling
   - Test rate limiting
   - Test timeout behavior

8. **Update DEPENDENCIES.md**
   - Update version table
   - Document any new known issues
   - Update compatibility matrix if needed

9. **Create PR**
   - Title: `chore(deps): upgrade [provider] SDK to vX.Y.Z`
   - Include test results and benchmark comparison
   - Tag with `dependencies`, `llm-client`

### Rollback Procedure

**If issues are discovered after upgrade:**

1. **Immediate rollback**
   ```bash
   git revert <commit-hash>
   # OR edit pyproject.toml to restore previous version
   ```

2. **Reinstall dependencies**
   ```bash
   uv sync --force
   ```

3. **Verify rollback**
   ```bash
   uv pip list | grep -E "(openai|anthropic|google-generativeai)"
   uv run pytest tests/integration/llm_client/ -v
   ```

4. **Document issue**
   - Create GitHub issue with `dependencies` and `bug` labels
   - Document specific failure mode
   - Add to "Known Issues" section for attempted version

5. **Deploy hotfix**
   ```bash
   git tag v0.1.1-hotfix
   git push origin v0.1.1-hotfix
   ```

## CI/CD Integration

### Version Validation

The CI pipeline validates dependency versions on every push and pull request.

**Automated Checks:**

1. **Dependency consistency check** (`.github/workflows/ci.yml`)
   ```bash
   # Verify pinned versions match requirements
   uv pip check
   ```

2. **Security audit** (runs nightly)
   ```bash
   uv pip audit
   ```

3. **Integration tests** (runs on all PRs)
   ```bash
   uv run pytest tests/integration/llm_client/
   ```

4. **Benchmark validation** (runs weekly)
   ```bash
   uv run python scripts/benchmark_llm.py
   ```

### Dependency Update Automation

**Dependabot Configuration:**

Dependabot is configured to create PRs for security updates only. Major and minor version updates are handled manually through the monthly review process.

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    # Only auto-update for security patches
    allow:
      - dependency-type: "all"
        update-types: ["security"]
```

### Pre-commit Hooks

**Dependency validation hook:**

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: check-pinned-versions
      name: Check LLM SDK versions are pinned
      entry: python scripts/validate_pinned_versions.py
      language: python
      files: pyproject.toml
```

## Troubleshooting

### Common Issues

**Problem: `ImportError` after upgrade**

Solution:
```bash
# Clear cache and reinstall
uv cache clean
uv sync --force
```

**Problem: Tests fail with API errors**

Solution:
1. Check if API keys are valid
2. Verify provider API status pages
3. Check rate limits
4. Review API version compatibility

**Problem: Streaming breaks after upgrade**

Solution:
1. Check streaming API changes in changelog
2. Review our streaming implementation
3. Test with minimal example
4. Consider rolling back if breaking change

**Problem: Performance regression after upgrade**

Solution:
1. Run benchmarks before/after
2. Profile slow operations
3. Check for new async patterns
4. Review connection pooling settings

### Emergency Contacts

**For critical issues with provider SDKs:**

- OpenAI Support: https://platform.openai.com/support
- Anthropic Support: https://console.anthropic.com/support
- Google AI Support: https://ai.google.dev/support

**Internal Escalation:**

- #engineering-llm-client (Slack)
- @agentcore/llm-team (GitHub)

## Version History

| Date | Provider | Old Version | New Version | Reason |
|------|----------|-------------|-------------|--------|
| 2025-10-26 | OpenAI | >=2.6.1 | ==1.54.0 | Initial pinning for stability |
| 2025-10-26 | Anthropic | >=0.40.0 | ==0.40.0 | Initial pinning for stability |
| 2025-10-26 | Google GenAI | >=0.8.0 | ==0.2.0 | Initial pinning for stability |

## References

- [A2A Protocol Specification](https://github.com/google/a2a-protocol)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Google GenAI Python SDK](https://github.com/google/generative-ai-python)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

---

*Last Updated: 2025-10-26*
*Next Review: 2025-11-04*
