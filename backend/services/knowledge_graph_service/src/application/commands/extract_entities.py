import logging
import uuid
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict

from backend.shared.infrastructure.messaging.kafka_producer import KafkaMessageProducer
from backend.services.knowledge_graph_service.src.infrastructure.graph_db.neo4j_client import Neo4jClient
from backend.services.knowledge_graph_service.src.infrastructure.extractors.relation_extractor import RelationExtractor

logger = logging.getLogger(__name__)


class ExtractEntitiesCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    text: str
    doc_type: str
    tenant_id: str


class ExtractEntitiesHandler:
    """
    CQRS Handler orchestrating the full extraction pipeline.
    Runs extractors, upserts to Neo4j, and broadcasts the event.
    """

    def __init__(
        self,
        relation_extractor: RelationExtractor,
        neo4j_client: Neo4jClient,
        kafka_producer: KafkaMessageProducer
    ):
        self.relation_extractor = relation_extractor
        self.neo4j_client = neo4j_client
        self.kafka_producer = kafka_producer

    async def handle(self, cmd: ExtractEntitiesCommand) -> None:
        logger.info("Extracting entities for doc_id=%s, doc_type=%s", cmd.doc_id, cmd.doc_type)

        doc_metadata = {
            "doc_id": cmd.doc_id,
            "tenant_id": cmd.tenant_id,
            "doc_type": cmd.doc_type
        }

        # 1. & 2. & 3. Run extractors and merge
        result = await self.relation_extractor.extract(cmd.text, doc_metadata)

        if not result.entities and not result.relations:
            logger.warning("No entities or relations found for doc_id=%s", cmd.doc_id)
            return

        # 4. Upsert Entities to Neo4j
        # For simplicity, we assign everything without a strict label mapping to a generic 'ExtractedEntity'
        # In a full production system, we'd map "MACHINE_ID" -> (m:Machine) specifically.
        for ent in result.entities:
            # Clean text for Neo4j usage
            ent_text = ent.text.replace("'", "").replace('"', '')
            
            # Very basic label mapping
            label = "Machine" if ent.label == "MACHINE_ID" else \
                    "SparePartSKU" if ent.label == "PART_NUMBER" else \
                    "FailureMode" if ent.label == "ERROR_CODE" else "ExtractedEntity"
            
            query = f"""
                MERGE (e:{label} {{name: $name}})
                SET e.label = $entity_label, e.last_seen_in_doc = $doc_id
            """
            try:
                await self.neo4j_client.execute_write(query, {
                    "name": ent_text,
                    "entity_label": ent.label,
                    "doc_id": cmd.doc_id
                })
            except Exception as e:
                logger.error("Failed to upsert entity %s: %s", ent.text, e)

        # 5. Create Relationships
        for rel in result.relations:
            source = rel.source.replace("'", "").replace('"', '')
            target = rel.target.replace("'", "").replace('"', '')
            rel_type = rel.relation_type.upper().replace(" ", "_")
            
            # Fallback relation type if empty or invalid
            if not rel_type:
                rel_type = "RELATED_TO"
                
            # Safely inject the relation type into Cypher (can't parameterize relationship types in Neo4j)
            query = f"""
                MATCH (s) WHERE s.name = $source
                MATCH (t) WHERE t.name = $target
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.confidence = $confidence, r.sentence = $sentence
            """
            try:
                await self.neo4j_client.execute_write(query, {
                    "source": source,
                    "target": target,
                    "confidence": rel.confidence,
                    "sentence": rel.sentence_span
                })
            except Exception as e:
                logger.error("Failed to upsert relation %s-[%s]->%s: %s", source, rel_type, target, e)

        # 6. Emit EntityExtracted Event
        payload = {
            "event_type": "EntityExtracted",
            "doc_id": cmd.doc_id,
            "tenant_id": cmd.tenant_id,
            "entity_count": len(result.entities),
            "relation_count": len(result.relations)
        }
        
        await self.kafka_producer.send(
            topic="ikb.graph.updates",
            value=payload,
            key=cmd.doc_id
        )
        
        logger.info("Extraction complete. Upserted %d entities and %d relations for doc %s", len(result.entities), len(result.relations), cmd.doc_id)
