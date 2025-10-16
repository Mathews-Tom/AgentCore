# Post-Launch Review Template - Bounded Context Reasoning

**Review Date:** _[Date]_
**Reviewers:** _[Names]_
**Launch Date:** _[Date]_
**Review Period:** _[Start Date] - [End Date]_ (Typically 28 days post-launch)

## Executive Summary

_[Brief 2-3 sentence summary of launch success, key metrics, and major findings]_

## Success Criteria Evaluation

### Performance Metrics

| Metric | Target | Actual | Status | Notes |
|--------|--------|--------|--------|-------|
| P50 Latency | < 1s | _[value]_ | ✅/❌ | |
| P95 Latency | < 2s | _[value]_ | ✅/❌ | |
| P99 Latency | < 5s | _[value]_ | ✅/❌ | |
| Error Rate | < 0.2% | _[value]_ | ✅/❌ | |
| Availability | > 99.9% | _[value]_ | ✅/❌ | |

### Compute Savings Metrics

| Metric | Target | Actual | Status | Notes |
|--------|--------|--------|--------|-------|
| Token Savings | > 50% | _[value]_ | ✅/❌ | |
| Cost Reduction | 40-60% | _[value]_ | ✅/❌ | |
| Avg Iterations | 3-5 | _[value]_ | ✅/❌ | |
| Memory Usage | O(1) | _[value]_ | ✅/❌ | |

### Adoption Metrics

| Metric | Target | Actual | Status | Notes |
|--------|--------|--------|--------|-------|
| Adoption Rate | > 80% | _[value]_ | ✅/❌ | |
| Active Users | _[target]_ | _[value]_ | ✅/❌ | |
| Queries/Day | _[target]_ | _[value]_ | ✅/❌ | |
| User Satisfaction | > 4.5/5 | _[value]_ | ✅/❌ | |

### Reliability Metrics

| Metric | Target | Actual | Status | Notes |
|--------|--------|--------|--------|-------|
| Security Incidents | 0 | _[value]_ | ✅/❌ | |
| P0 Incidents | 0 | _[value]_ | ✅/❌ | |
| P1 Incidents | < 2 | _[value]_ | ✅/❌ | |
| Rollbacks | 0 | _[value]_ | ✅/❌ | |

## Detailed Analysis

### Performance Analysis

**Latency:**
- _[Analysis of latency trends, spikes, improvements]_
- _[Comparison with pre-launch baseline]_
- _[Query patterns that performed well/poorly]_

**Throughput:**
- _[Peak requests/second achieved]_
- _[Sustained load handling]_
- _[Bottlenecks identified]_

**Resource Usage:**
- _[CPU utilization trends]_
- _[Memory usage patterns]_
- _[Database connection pool usage]_

### Cost Analysis

**LLM Token Usage:**
- Pre-launch baseline: _[tokens/day]_
- Post-launch usage: _[tokens/day]_
- Actual savings: _[percentage]_
- Cost impact: _[$amount/month]_

**Infrastructure Costs:**
- Additional compute: _[$amount]_
- Additional storage: _[$amount]_
- Additional network: _[$amount]_
- Total new costs: _[$amount]_

**Net Savings:**
- Gross LLM savings: _[$amount]_
- New infrastructure costs: _[$amount]_
- Net savings: _[$amount]_
- ROI: _[percentage]_

### User Feedback

**Positive Feedback:**
- _[Summary of positive user comments]_
- _[Most appreciated features]_
- _[Use cases where feature excelled]_

**Negative Feedback:**
- _[Summary of user complaints]_
- _[Common pain points]_
- _[Feature requests]_

**Customer Quotes:**
> _[Notable customer quote 1]_

> _[Notable customer quote 2]_

### Incident Analysis

**Incidents Summary:**

| Date | Severity | Duration | Impact | Root Cause | Resolution |
|------|----------|----------|--------|------------|------------|
| _[date]_ | P0/P1/P2 | _[duration]_ | _[users affected]_ | _[cause]_ | _[fix]_ |

**Key Incidents:**

1. **Incident Title:** _[Title]_
   - **Date/Time:** _[timestamp]_
   - **Duration:** _[duration]_
   - **Impact:** _[description]_
   - **Root Cause:** _[detailed cause]_
   - **Resolution:** _[how it was fixed]_
   - **Lessons Learned:** _[takeaways]_
   - **Action Items:** _[prevention measures]_

### Technical Challenges

**Challenges Encountered:**

1. **Challenge:** _[Description]_
   - **Impact:** _[severity and scope]_
   - **Resolution:** _[how it was addressed]_
   - **Time to Resolve:** _[duration]_

2. **Challenge:** _[Description]_
   - **Impact:** _[severity and scope]_
   - **Resolution:** _[how it was addressed]_
   - **Time to Resolve:** _[duration]_

## Rollout Timeline Analysis

| Phase | Start Date | Duration | Traffic % | Issues | Status |
|-------|-----------|----------|-----------|--------|--------|
| Phase 1 (1%) | _[date]_ | 24h | 1% | _[count]_ | ✅ |
| Phase 2 (10%) | _[date]_ | 48h | 10% | _[count]_ | ✅ |
| Phase 3 (25%) | _[date]_ | 72h | 25% | _[count]_ | ✅ |
| Phase 4 (50%) | _[date]_ | 72h | 50% | _[count]_ | ✅ |
| Phase 5 (100%) | _[date]_ | Ongoing | 100% | _[count]_ | ✅ |

**Rollout Notes:**
- _[Any phase delays or accelerations]_
- _[Rollback events, if any]_
- _[Decision points and rationale]_

## Team Retrospective

### What Went Well

1. _[Success point 1]_
2. _[Success point 2]_
3. _[Success point 3]_

### What Could Be Improved

1. _[Improvement area 1]_
2. _[Improvement area 2]_
3. _[Improvement area 3]_

### Action Items

| Action | Owner | Priority | Due Date | Status |
|--------|-------|----------|----------|--------|
| _[action]_ | _[name]_ | P0/P1/P2 | _[date]_ | 🔴/🟡/🟢 |

## Lessons Learned

### Process

**Testing:**
- _[What worked in testing approach]_
- _[What was missed in testing]_
- _[Improvements for next launch]_

**Monitoring:**
- _[Effectiveness of monitoring setup]_
- _[Gaps in observability]_
- _[Additional metrics needed]_

**Communication:**
- _[Internal communication effectiveness]_
- _[Customer communication clarity]_
- _[Improvements needed]_

### Technical

**Architecture:**
- _[Design decisions that worked well]_
- _[Design decisions to reconsider]_
- _[Technical debt created]_

**Operations:**
- _[Deployment process effectiveness]_
- _[Incident response quality]_
- _[Runbook completeness]_

## Recommendations

### Immediate Actions (Next 30 Days)

1. **Action:** _[description]_
   - **Rationale:** _[why]_
   - **Owner:** _[name]_
   - **Priority:** P0/P1/P2

### Short-Term (Next Quarter)

1. **Action:** _[description]_
   - **Rationale:** _[why]_
   - **Owner:** _[name]_
   - **Priority:** P0/P1/P2

### Long-Term (Next Year)

1. **Action:** _[description]_
   - **Rationale:** _[why]_
   - **Owner:** _[name]_
   - **Priority:** P0/P1/P2

## Future Enhancements

### Planned Features

1. **Feature:** _[name]_
   - **Description:** _[what it does]_
   - **Business Value:** _[why it matters]_
   - **Effort:** _[estimate]_
   - **Target Date:** _[when]_

### Optimization Opportunities

1. **Optimization:** _[name]_
   - **Current State:** _[baseline]_
   - **Target State:** _[goal]_
   - **Expected Impact:** _[benefit]_
   - **Effort:** _[estimate]_

## Appendices

### A. Detailed Metrics

_[Attach Grafana dashboard screenshots]_
_[Include Prometheus query results]_
_[Add cost analysis spreadsheets]_

### B. User Feedback

_[Full survey results]_
_[Support ticket analysis]_
_[Customer interview notes]_

### C. Technical Documentation

_[Architecture diagrams]_
_[Performance benchmark results]_
_[Security audit reports]_

### D. Meeting Notes

_[Standup notes from rollout period]_
_[Incident review meeting notes]_
_[Stakeholder update meetings]_

## Sign-Off

This post-launch review has been completed and the findings have been reviewed by the following stakeholders:

| Stakeholder | Role | Signature | Date |
|-------------|------|-----------|------|
| _[name]_ | Engineering Lead | | |
| _[name]_ | Product Manager | | |
| _[name]_ | SRE Lead | | |
| _[name]_ | Security Lead | | |
| _[name]_ | Finance/Ops | | |

**Overall Launch Status:** ✅ Success / ⚠️ Partial Success / ❌ Requires Remediation

**Next Review Date:** _[date]_ (90 days post-launch)

---

**Template Version:** 1.0
**Last Updated:** 2025-10-16
