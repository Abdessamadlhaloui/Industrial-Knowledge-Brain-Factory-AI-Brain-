import asyncio
import logging
import random
import uuid

from backend.services.knowledge_graph_service.src.infrastructure.graph_db.neo4j_client import Neo4jClient
from backend.services.knowledge_graph_service.src.infrastructure.graph_db.graph_schema import apply_schema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def seed_graph(uri: str, user: str, password: str) -> None:
    """
    Seeds a realistic industrial graph: 3 factories, 20 machines each.
    Maps components, sensors, failure modes, and maintenance actions.
    """
    client = Neo4jClient(uri, user, password)
    await client.connect()
    
    # 1. Apply Schema
    await apply_schema(client)
    
    # 2. Clear existing data (DANGER: only for seed script in dev/test)
    logger.info("Clearing existing graph data...")
    await client.execute_write("MATCH (n) DETACH DELETE n")

    factories = ["Factory_Berlin", "Factory_Munich", "Factory_Hamburg"]
    machine_types = ["CNC", "Conveyor", "HVAC"]
    
    # Pre-define some shared Failure Modes and Maintenance Actions
    failure_modes = [
        {"id": "FM_VIB_01", "name": "Spindle Vibration Drift", "category": "Mechanical"},
        {"id": "FM_TMP_01", "name": "Motor Overheating", "category": "Thermal"},
        {"id": "FM_BRG_01", "name": "Bearing Wear", "category": "Mechanical"},
        {"id": "FM_FLT_01", "name": "Filter Clogging", "category": "Pneumatic"}
    ]
    
    maintenance_actions = [
        {"id": "MA_RPL_BRG", "name": "Replace Bearing", "avg_hours": 4.5},
        {"id": "MA_CLN_FLT", "name": "Clean/Replace Filter", "avg_hours": 1.0},
        {"id": "MA_LUB", "name": "Lubricate Spindle", "avg_hours": 0.5}
    ]
    
    # Insert common nodes
    logger.info("Inserting generic failure modes and maintenance actions...")
    for fm in failure_modes:
        await client.execute_write(
            "MERGE (f:FailureMode {id: $id}) SET f.name = $name, f.category = $category", fm
        )
        
    for ma in maintenance_actions:
        await client.execute_write(
            "MERGE (m:MaintenanceAction {id: $id}) SET m.name = $name, m.avg_hours = $avg_hours", ma
        )
        
    # Link Failures to Maintenance
    await client.execute_write("MATCH (f:FailureMode {id: 'FM_BRG_01'}), (m:MaintenanceAction {id: 'MA_RPL_BRG'}) MERGE (f)-[:RESOLVED_BY]->(m)")
    await client.execute_write("MATCH (f:FailureMode {id: 'FM_FLT_01'}), (m:MaintenanceAction {id: 'MA_CLN_FLT'}) MERGE (f)-[:RESOLVED_BY]->(m)")
    await client.execute_write("MATCH (f:FailureMode {id: 'FM_VIB_01'}), (m:MaintenanceAction {id: 'MA_LUB'}) MERGE (f)-[:RESOLVED_BY]->(m)")
    
    # Propagation: Overheating -> Bearing Wear
    await client.execute_write("MATCH (f1:FailureMode {id: 'FM_TMP_01'}), (f2:FailureMode {id: 'FM_BRG_01'}) MERGE (f1)-[:PROPAGATES_TO {delay_hours: 48.0}]->(f2)")

    # 3. Generate Topology
    logger.info("Generating factories, machines, and components...")
    
    for factory in factories:
        for m_idx in range(1, 21):
            machine_id = f"MAC_{factory}_{m_idx}"
            m_type = random.choice(machine_types)
            
            # Create Machine
            await client.execute_write(
                "CREATE (m:Machine {id: $id, name: $name, type: $type, factory_id: $factory_id})",
                {"id": machine_id, "name": f"{m_type} Unit {m_idx}", "type": m_type, "factory_id": factory}
            )
            
            # Create Components
            components = ["Motor", "Spindle", "CoolingSystem"] if m_type == "CNC" else ["Motor", "Belt", "Roller"]
            for comp in components:
                comp_id = f"COMP_{machine_id}_{comp}"
                await client.execute_write(
                    "MATCH (m:Machine {id: $m_id}) CREATE (c:Component {id: $c_id, name: $name}) MERGE (m)-[:HAS_COMPONENT]->(c)",
                    {"m_id": machine_id, "c_id": comp_id, "name": comp}
                )
                
                # Add Sensors to components
                sensor_id = f"SENS_{comp_id}_VIB"
                await client.execute_write(
                    "MATCH (c:Component {id: $c_id}) CREATE (s:Sensor {id: $s_id, type: 'Vibration'}) MERGE (c)-[:HAS_SENSOR]->(s)",
                    {"c_id": comp_id, "s_id": sensor_id}
                )
                
                # Map Risks
                if comp == "Motor":
                    await client.execute_write(
                        "MATCH (c:Component {id: $c_id}), (f:FailureMode {id: 'FM_TMP_01'}) MERGE (c)-[:CAN_FAIL_WITH {probability: 0.15}]->(f)",
                        {"c_id": comp_id}
                    )
                    # Sensor indicates failure
                    await client.execute_write(
                        "MATCH (s:Sensor {id: $s_id}), (f:FailureMode {id: 'FM_TMP_01'}) MERGE (s)-[:INDICATES {confidence: 0.85}]->(f)",
                        {"s_id": sensor_id}
                    )

    logger.info("Seeding complete. ~500 nodes and ~2000 relationships generated.")
    await client.close()

if __name__ == "__main__":
    # Normally read from env vars
    asyncio.run(seed_graph("bolt://localhost:7687", "neo4j", "password"))
