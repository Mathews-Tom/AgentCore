// Neo4j Graph Schema for AgentCore Memory System
// Component: Hybrid Memory (COMPASS + Mem0 + Graph)
// Purpose: Knowledge graphs for entity relationships and context

// ============================================================================
// NODE LABELS
// ============================================================================

// Memory Node: Core memory record stored in graph
// Properties:
// - memory_id: UUID (unique identifier, indexed)
// - agent_id: UUID (agent that created the memory)
// - session_id: UUID (session context)
// - layer: enum (episodic, semantic, procedural)
// - stage: enum (planning, execution, reflection, verification)
// - content: text (memory content)
// - criticality: float (0-1, importance score)
// - created_at: datetime
// - accessed_count: int
CREATE CONSTRAINT memory_id_unique IF NOT EXISTS
FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE;

CREATE INDEX memory_agent_idx IF NOT EXISTS
FOR (m:Memory) ON (m.agent_id);

CREATE INDEX memory_session_idx IF NOT EXISTS
FOR (m:Memory) ON (m.session_id);

CREATE INDEX memory_layer_idx IF NOT EXISTS
FOR (m:Memory) ON (m.layer);

CREATE INDEX memory_stage_idx IF NOT EXISTS
FOR (m:Memory) ON (m.stage);

CREATE INDEX memory_created_idx IF NOT EXISTS
FOR (m:Memory) ON (m.created_at);

CREATE FULLTEXT INDEX memory_content_fulltext IF NOT EXISTS
FOR (m:Memory) ON EACH [m.content];

// Entity Node: Extracted entities from memory content
// Properties:
// - entity_id: UUID (unique identifier, indexed)
// - name: string (entity name, normalized)
// - entity_type: enum (person, concept, tool, constraint, other)
// - confidence: float (0-1, extraction confidence)
// - properties: map (additional metadata)
// - first_seen: datetime
// - last_seen: datetime
// - mention_count: int
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

CREATE INDEX entity_name_idx IF NOT EXISTS
FOR (e:Entity) ON (e.name);

CREATE INDEX entity_type_idx IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type);

CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name];

// Concept Node: High-level semantic concepts
// Properties:
// - concept_id: UUID (unique identifier, indexed)
// - name: string (concept name)
// - description: text (concept description)
// - category: string (concept category)
// - created_at: datetime
// - usage_count: int
CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE;

CREATE INDEX concept_name_idx IF NOT EXISTS
FOR (c:Concept) ON (c.name);

CREATE INDEX concept_category_idx IF NOT EXISTS
FOR (c:Concept) ON (c.category);

CREATE FULLTEXT INDEX concept_description_fulltext IF NOT EXISTS
FOR (c:Concept) ON EACH [c.description];

// ============================================================================
// RELATIONSHIP TYPES
// ============================================================================

// MENTIONS: Memory mentions an Entity
// Properties:
// - position: int (position in text)
// - context: text (surrounding context)
// - sentiment: float (-1 to 1, optional)
// - created_at: datetime
CREATE INDEX mentions_created_idx IF NOT EXISTS
FOR ()-[r:MENTIONS]-() ON (r.created_at);

// RELATES_TO: Entity relates to another Entity
// Properties:
// - relationship_type: enum (depends_on, part_of, similar_to, contradicts, etc.)
// - strength: float (0-1, relationship strength)
// - confidence: float (0-1, detection confidence)
// - created_at: datetime
// - last_reinforced: datetime
// - reinforcement_count: int
CREATE INDEX relates_to_type_idx IF NOT EXISTS
FOR ()-[r:RELATES_TO]-() ON (r.relationship_type);

CREATE INDEX relates_to_strength_idx IF NOT EXISTS
FOR ()-[r:RELATES_TO]-() ON (r.strength);

// PART_OF: Entity is part of a Concept
// Properties:
// - relevance: float (0-1)
// - created_at: datetime
CREATE INDEX part_of_relevance_idx IF NOT EXISTS
FOR ()-[r:PART_OF]-() ON (r.relevance);

// FOLLOWS: Memory follows another Memory (temporal)
// Properties:
// - time_delta: int (seconds between memories)
// - stage_transition: bool (crosses stage boundary)
// - created_at: datetime
CREATE INDEX follows_time_delta_idx IF NOT EXISTS
FOR ()-[r:FOLLOWS]-() ON (r.time_delta);

// PRECEDES: Memory precedes another Memory (temporal inverse)
// Properties:
// - time_delta: int (seconds until next memory)
// - stage_transition: bool (crosses stage boundary)
// - created_at: datetime
CREATE INDEX precedes_time_delta_idx IF NOT EXISTS
FOR ()-[r:PRECEDES]-() ON (r.time_delta);

// TRIGGERS: Memory triggers another Memory (causal)
// Properties:
// - trigger_type: enum (error, success, decision, etc.)
// - confidence: float (0-1)
// - created_at: datetime
CREATE INDEX triggers_type_idx IF NOT EXISTS
FOR ()-[r:TRIGGERS]-() ON (r.trigger_type);

// ============================================================================
// UTILITY PROCEDURES
// ============================================================================

// Create Memory node with automatic indexing
CALL apoc.custom.asProcedure(
  'createMemory',
  'CREATE (m:Memory {
    memory_id: $memory_id,
    agent_id: $agent_id,
    session_id: $session_id,
    layer: $layer,
    stage: $stage,
    content: $content,
    criticality: $criticality,
    created_at: datetime($created_at),
    accessed_count: 0
  })
  RETURN m',
  'write',
  [['m', 'NODE']],
  [['memory_id', 'STRING'], ['agent_id', 'STRING'], ['session_id', 'STRING'],
   ['layer', 'STRING'], ['stage', 'STRING'], ['content', 'STRING'],
   ['criticality', 'FLOAT'], ['created_at', 'STRING']]
);

// Create Entity node with deduplication
CALL apoc.custom.asProcedure(
  'createOrUpdateEntity',
  'MERGE (e:Entity {name: $name, entity_type: $entity_type})
  ON CREATE SET
    e.entity_id = $entity_id,
    e.confidence = $confidence,
    e.properties = $properties,
    e.first_seen = datetime($first_seen),
    e.last_seen = datetime($last_seen),
    e.mention_count = 1
  ON MATCH SET
    e.last_seen = datetime($last_seen),
    e.mention_count = e.mention_count + 1,
    e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END
  RETURN e',
  'write',
  [['e', 'NODE']],
  [['entity_id', 'STRING'], ['name', 'STRING'], ['entity_type', 'STRING'],
   ['confidence', 'FLOAT'], ['properties', 'MAP'],
   ['first_seen', 'STRING'], ['last_seen', 'STRING']]
);

// ============================================================================
// PERFORMANCE OPTIMIZATION QUERIES
// ============================================================================

// Optimize graph for 2-hop traversal performance
// Run periodically to maintain <200ms p95 latency
CALL gds.graph.project.cypher(
  'memory-graph',
  'MATCH (n) WHERE n:Memory OR n:Entity OR n:Concept RETURN id(n) AS id',
  'MATCH (n)-[r]->(m) RETURN id(n) AS source, id(m) AS target, type(r) AS type'
);

// ============================================================================
// SCHEMA VALIDATION
// ============================================================================

// Verify indexes exist
SHOW INDEXES;

// Verify constraints exist
SHOW CONSTRAINTS;

// ============================================================================
// INITIAL DATA (OPTIONAL)
// ============================================================================

// Example: Create system concept nodes
MERGE (c:Concept {
  concept_id: 'system-init',
  name: 'System Initialization',
  description: 'Core system initialization and setup',
  category: 'system',
  created_at: datetime(),
  usage_count: 0
});
