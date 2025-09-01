# Agent–ViewModel (AVM)

## Intent

Expose a narrow, policy-enforced capability surface to an agent so it can perceive and act on a complex system safely and effectively, without seeing the underlying domain model or infrastructure.

## Also Known As

Agent Facade, Capability Surface, Anti-Corruption Layer for Agents, Ports-and-Adapters for LLMs.

## Motivation

Agents reason better when their world is small, stable, and typed. Enterprise systems are the opposite: heterogeneous, noisy, and permissioned. AVM interposes a purposeful layer that turns messy backends into a compact set of intents with clear inputs, outputs, and errors. The agent binds to this surface (as function/tools), while AVM enforces validation, policy, idempotency, and translation. You get least-privilege execution, predictable retries, lower token usage, and testable behavior.

## Applicability

Use AVM whenever an agent must read or change state in one or more real systems; when you must prevent schema or policy drift from leaking into prompts; when long-running operations require job orchestration; or when safety, auditability, and deterministic retries matter.

## Structure

```
Agent  ⇄  Agent–ViewModel  ⇄  Services/Adapters  ⇄  Domain Model & External Systems
  |           |                     |                          |
  |     typed tools                 | mapping, retries         | data, side effects
  |     read models                 | policy, ACL, caching     | authoritative truth
```

## Participants

The Agent consumes only AVM tools and read models. The Agent–ViewModel defines intent-level commands and queries, applies validation and policy, normalizes data, and guards side effects with idempotency and audits. Services/Adapters hide protocols and vendors, handling auth, pagination, retries, and transformation. The Domain Model and external systems remain unexposed to the agent.

## Collaborations

The agent issues a query or command; AVM validates, authorizes, and either returns a summarized read model or executes an idempotent command via services. Long tasks return a job identifier; the agent polls status through AVM. Errors are mapped to a small, stable taxonomy so the agent can recover.

## Consequences

AVM improves safety, clarity, and testability by shrinking the agent’s world and enforcing policy at one choke point. It reduces token cost and prompt brittleness by providing compact read models. It also adds a layer to design and maintain, and you must version the surface as capabilities evolve.

## Implementation

Design the surface around intents rather than endpoints. Define strict schemas for every tool and read model, including explicit error codes and invariants. Enforce least privilege and data redaction at the AVM boundary. Make commands idempotent with keys and return correlation ids for audit. Represent long-running work as jobs with start and get\_status. Cache and cap results to predictable sizes. Version your AVM contracts and keep adapters thin and swappable. Unit-test AVM in isolation and run conversation-level tests with golden transcripts.

## Sample Code

```ts
// Tool contracts: the agent only sees these.
type Ticket = { id: string; title: string; status: "Open"|"InProgress"|"Done"; priority_score: number };
type SearchTickets = (args: { query: string; assignee?: string; limit?: number }) => Promise<{ items: Ticket[] }>;
type UpdateStatus = (args: { id: string; to: "Open"|"InProgress"|"Done"; idem_key: string }) => Promise<{ ok: true }>;
type StartExport = (args: { query: string }) => Promise<{ job_id: string }>;
type GetJobStatus = (args: { job_id: string }) => Promise<{ state: "Queued"|"Running"|"Succeeded"|"Failed"; url?: string; error_code?: string }>;

class AgentViewModel {
  constructor(private tickets: TicketService, private audit: Audit) {}
  async searchTickets(a): Promise<{items: Ticket[]}> {
    const q = sanitize(a.query); const n = Math.min(a.limit ?? 20, 50);
    const rows = await this.tickets.search({ q, assignee: a.assignee, limit: n });
    return { items: rows.map(r => ({ id: r.id, title: r.title, status: mapStatus(r), priority_score: rank(r) })) };
  }
  async updateStatus(a): Promise<{ok:true}> {
    requireOneOf(a, ["id","to","idem_key"]); authorize("ticket.update", a.id);
    await this.tickets.updateStatus({ id: a.id, to: a.to, idem_key: a.idem_key });
    this.audit.log("ticket.update_status", a);
    return { ok: true };
  }
}
```

## Known Uses

Internal copilots that operate over ticketing, CRM, knowledge bases, and DevOps commonly use an AVM surface to expose a handful of safe verbs and summarized views while insulating the agent from vendor APIs and raw schemas.

## Related Patterns

Facade simplifies a subsystem much like AVM simplifies backends for an agent. Hexagonal Architecture (ports and adapters) underpins the separation between AVM and services. Anti-Corruption Layer prevents foreign models from leaking into the agent surface. CQRS aligns with AVM’s split between compact read models and narrow commands. Saga supports multi-step, compensable workflows behind AVM.
