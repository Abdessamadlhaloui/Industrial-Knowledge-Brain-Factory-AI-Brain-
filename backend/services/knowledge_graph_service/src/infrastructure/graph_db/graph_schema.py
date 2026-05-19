import logging
from backend.services.knowledge_graph_service.src.infrastructure.graph_db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Idempotent Schema Definitions
INDUSTRIAL_SCHEMA = [
    # Machine Constraints
    "CREATE CONSTRAINT machine_id IF NOT EXISTS FOR (m:Machine) REQUIRE m.id IS UNIQUE",
    "CREATE INDEX machine_factory IF NOT EXISTS FOR (m:Machine) ON (m.factory_id)",
    
    # Component Constraints
    "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
    
    # Sensor Constraints
    "CREATE CONSTRAINT sensor_id IF NOT EXISTS FOR (s:Sensor) REQUIRE s.id IS UNIQUE",
    
    # Failure Mode Constraints
    "CREATE CONSTRAINT failure_mode_id IF NOT EXISTS FOR (f:FailureMode) REQUIRE f.id IS UNIQUE",
    "CREATE INDEX failure_category IF NOT EXISTS FOR (f:FailureMode) ON (f.category)",
    
    # Maintenance Action Constraints
    "CREATE CONSTRAINT maintenance_action_id IF NOT EXISTS FOR (ma:MaintenanceAction) REQUIRE ma.id IS UNIQUE",
    
    # Incident Constraints
    "CREATE CONSTRAINT incident_id IF NOT EXISTS FOR (i:Incident) REQUIRE i.id IS UNIQUE",
    "CREATE INDEX incident_timestamp IF NOT EXISTS FOR (i:Incident) ON (i.timestamp)",
    
    # Spare Part Constraints
    "CREATE CONSTRAINT spare_part_sku IF NOT EXISTS FOR (sp:SparePartSKU) REQUIRE sp.sku IS UNIQUE"
]

async def apply_schema(client: Neo4jClient) -> None:
    """
    Apply the database schema (constraints and indexes).
    These operations are idempotent.
    """
    logger.info("Applying Neo4j Industrial Schema...")
    for statement in INDUSTRIAL_SCHEMA:
        try:
            await client.execute_write(statement)
        except Exception as e:
            logger.warning("Schema statement warning/error (could be normal if already exists differently): %s", e)
    logger.info("Schema application complete.")
