# Agent–ViewModel (AVM)

## Intent

Provide a single, simplified capability surface where agents and other clients safely interact with complex systems. The AVM boundary hides domain complexity, enforces policy and least privilege, and exposes only meaningful intents and compact read models. It also maintains a shared live projection of state so multiple clients—automation agents, supervisory agents, and human UIs—can coordinate in real time: one client can act, another can observe or intervene, and everyone stays synchronized automatically.

## Also Known As

Agent Facade; Capability Surface; Anti-Corruption Layer for Agents; Ports-and-Adapters for LLMs; Reactive Read-Model for Agents.

## Motivation

Real systems are heterogeneous, permissioned, and messy; agents reason best over small, typed, and stable worlds. AVM introduces a boundary that presents high-level intents with strict contracts and emits a live projection of state. Successful actions update that projection so all subscribers immediately see the change, enabling automation with oversight and human-in-the-loop approvals without leaking backend details.

### Example: Defining an Intent

```python
class UpdateStatusIn(BaseModel):
    id: str
    to: Literal["Open","InProgress","Done"]
    precondition_version: int
    idem_key: str

@app.post("/intents/ticket.update_status", response_model=CommandResult)
async def ticket_update_status(args: UpdateStatusIn):
    return await avm.update_status(args)
```

This snippet shows an **intent** exposed via the AVM boundary: strictly typed input, predictable output, clear semantics.

## Applicability

Use AVM when agents or human users must read or change state in one or more real systems; when safety, auditability, deterministic retries, or approvals matter; when multiple clients need to coordinate; or when changes must be visible in real time across participants.

## Structure

```
Clients (Automation Agent, Supervisor Agent, Human UI)
   ⇄  Subscriptions (Projections, Diffs, Events)
   ⇄  Agent–ViewModel (Intents, Policy, Validation, Error mapping)
   ⇄  Command/Job Coordinator (Idempotency, Two-Phase, Sagas)
   ⇄  Services/Adapters (Auth, Retries, Mapping)
   ⇄  Domain Model & External Systems (authoritative truth)
```

### Example: Subscribing to a Projection

```python
@app.websocket("/subscriptions/{projection_id}")
async def subscribe(ws: WebSocket, projection_id: str):
    await ws.accept()
    queue = await projection_store.subscribe(projection_id)
    try:
        while True:
            ev: ProjectionEvent = await queue.get()
            await ws.send_json(ev.model_dump())
    except WebSocketDisconnect:
        return
```

Here, multiple clients bind to the same **projection**; when state changes, all subscribers receive updates immediately.

## Participants

Clients subscribe to projections and invoke intents. The Agent–ViewModel defines contracts, validates input, enforces policy and scopes, normalizes data, and guards side effects with idempotency and audit. A Projection Store materializes reasoning-friendly views and publishes updates. A Command/Job Coordinator handles idempotency, long-running jobs, and two-phase approvals. Services/Adapters encapsulate protocol, retries, and mappings to the real systems, which remain hidden and authoritative.

## Collaborations

A client subscribes to a projection to receive an initial snapshot and subsequent diffs. It invokes an intent through the AVM with a precondition version and an idempotency key. The AVM validates and authorizes, then either executes synchronously, starts a job with progress updates, or opens a two-phase hold for approval. On success or meaningful progress, the AVM updates the projection and publishes an event so all subscribers immediately see consistent state. Conflicts and approvals surface as small, predictable error codes so clients can recover deterministically.

### Example: Two-Phase Command

```python
@app.post("/intents/order.propose_change", response_model=CommandResult)
async def propose_change(args: ProposeIn):
    res = await coordinator.propose_change(args.target_id, args.patch, args.idem_key)
    await projection_store.update(f"order:{args.target_id}", {"pending_patch": args.patch})
    return res
```

An automation agent can **propose** a change. A human or supervisory agent can later **commit** or **cancel** it; everyone sees pending state in the projection.

## Consequences

You gain safe automation with oversight, a stable minimal surface for agents, deterministic retries and auditing, and lower token and cognitive load via compact projections. You take on a lightweight eventing substrate, projection versioning, and explicit concurrency semantics at the boundary. Versioned contracts keep clients stable as capabilities evolve.

## Implementation

Design around intents rather than endpoints. Define strict schemas for commands, queries, projections, events, and errors. Enforce least privilege and redact sensitive data at the boundary. Make commands idempotent with keys and use versioned preconditions. Represent long work as jobs with start and get\_status and publish progress into projections. Use two-phase commands for human approvals. Provide per-client capability profiles without cloning ViewModels. Keep adapters thin and swappable. Test contracts in isolation and run golden conversation tests for happy paths, conflicts, jobs, and interventions.

### Example: Long-Running Job

```python
@app.post("/jobs/start", response_model=JobState)
async def start_job(args: StartJobIn):
    job = await coordinator.start_job(args.kind, args.target_id, args.idem_key)
    await projection_store.update(f"job:{args.idem_key}", job.model_dump())
    return job
```

This shows a **job intent** where clients can start long-running tasks and watch progress via projection updates.

## Related Patterns

Facade, Hexagonal Architecture (Ports-and-Adapters), Anti-Corruption Layer, CQRS, Saga, Observer/Publish-Subscribe, Optimistic Concurrency.
