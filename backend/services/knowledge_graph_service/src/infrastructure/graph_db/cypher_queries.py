"""
Central repository for parameterized Cypher queries used by the Knowledge Graph Service.
"""

# 1. GET_FAILURE_CHAIN
# Traverses: Machine -> Component -> FailureMode -> propagated failures -> MaintenanceActions (depth 0..3)
GET_FAILURE_CHAIN = """
MATCH (m:Machine {id: $machine_id})-[:HAS_COMPONENT]->(c:Component {id: $component_id})
MATCH (c)-[cf:CAN_FAIL_WITH]->(fm:FailureMode)
OPTIONAL MATCH path = (fm)-[:PROPAGATES_TO*0..3]->(downstream:FailureMode)
OPTIONAL MATCH (downstream)-[rb:RESOLVED_BY]->(ma:MaintenanceAction)
RETURN 
    fm.id AS root_failure,
    fm.name AS root_name,
    cf.probability AS initial_probability,
    [node IN nodes(path) | node.name] AS propagation_chain,
    downstream.name AS ultimate_failure,
    ma.name AS recommended_action,
    rb.avg_hours AS resolution_time
ORDER BY cf.probability DESC
"""

# 2. GET_SIMILAR_INCIDENTS
# Finds incidents in date range sharing failure modes, with resolution times
GET_SIMILAR_INCIDENTS = """
MATCH (i:Incident)-[:CAUSED_BY]->(fm:FailureMode {id: $failure_mode_id})
WHERE i.timestamp >= $start_date AND i.timestamp <= $end_date
OPTIONAL MATCH (fm)-[rb:RESOLVED_BY]->(ma:MaintenanceAction)
RETURN 
    i.id AS incident_id,
    i.timestamp AS timestamp,
    i.description AS description,
    rb.avg_hours AS avg_resolution_hours
ORDER BY i.timestamp DESC
LIMIT $limit
"""

# 3. GET_MACHINE_HEALTH_SUBGRAPH
# Extracts machine + all components + sensors + active risks + recent incidents
GET_MACHINE_HEALTH_SUBGRAPH = """
MATCH (m:Machine {id: $machine_id})
OPTIONAL MATCH (m)-[:HAS_COMPONENT]->(c:Component)
OPTIONAL MATCH (c)-[:HAS_SENSOR]->(s:Sensor)
OPTIONAL MATCH (c)-[risk:CAN_FAIL_WITH]->(fm:FailureMode)
OPTIONAL MATCH (i:Incident)-[:AFFECTED]->(m)
WHERE i.timestamp >= $recent_threshold_date OR i IS NULL
RETURN 
    m AS machine,
    collect(DISTINCT c) AS components,
    collect(DISTINCT s) AS sensors,
    collect(DISTINCT {failure: fm.name, probability: risk.probability}) AS active_risks,
    collect(DISTINCT i) AS recent_incidents
"""

# 4. CAUSAL_PATH_ANALYSIS
# sensor -> failure -> propagation chain -> compound_risk
CAUSAL_PATH_ANALYSIS = """
MATCH (s:Sensor {id: $sensor_id})-[ind:INDICATES]->(root_fm:FailureMode)
WHERE ind.confidence >= $min_confidence
OPTIONAL MATCH path = (root_fm)-[prop:PROPAGATES_TO*1..5]->(end_fm:FailureMode)
OPTIONAL MATCH (end_fm)-[rb:RESOLVED_BY]->(ma:MaintenanceAction)
WITH s, ind, root_fm, path, end_fm, ma,
     reduce(p = 1.0, r IN relationships(path) | p * coalesce(r.probability, 1.0)) AS propagation_prob
RETURN 
    s.id AS sensor_id,
    root_fm.id AS root_cause_id,
    root_fm.name AS root_cause_name,
    ind.confidence AS detection_confidence,
    [node IN nodes(path) | node.name] AS propagation_path,
    (ind.confidence * coalesce(propagation_prob, 1.0)) AS compound_risk,
    ma.name AS recommended_action
ORDER BY compound_risk DESC
"""

# 5. UPSERT_MACHINE
# Merge on machine.id, set all properties
UPSERT_MACHINE = """
MERGE (m:Machine {id: $machine_id})
SET m.name = $name,
    m.factory_id = $factory_id,
    m.status = $status,
    m.updated_at = $timestamp
RETURN m
"""

# 6. UPSERT_INCIDENT
# Create incident node + link to machine + failure mode
UPSERT_INCIDENT = """
MERGE (i:Incident {id: $incident_id})
SET i.timestamp = $timestamp,
    i.description = $description,
    i.severity = $severity
WITH i
MATCH (m:Machine {id: $machine_id})
MERGE (i)-[:AFFECTED]->(m)
WITH i
MATCH (fm:FailureMode {id: $failure_mode_id})
MERGE (i)-[:CAUSED_BY]->(fm)
RETURN i
"""

# 7. GET_MAINTENANCE_HISTORY
# All MaintenanceActions on a machine, ordered by date, with parts used
GET_MAINTENANCE_HISTORY = """
MATCH (m:Machine {id: $machine_id})<-[:AFFECTED]-(i:Incident)-[:CAUSED_BY]->(fm:FailureMode)
MATCH (fm)-[:RESOLVED_BY]->(ma:MaintenanceAction)
OPTIONAL MATCH (ma)-[:REQUIRES_PART]->(sp:SparePartSKU)
RETURN 
    i.timestamp AS incident_date,
    fm.name AS failure_reason,
    ma.name AS action_taken,
    collect(DISTINCT sp.sku) AS parts_used
ORDER BY i.timestamp DESC
"""
