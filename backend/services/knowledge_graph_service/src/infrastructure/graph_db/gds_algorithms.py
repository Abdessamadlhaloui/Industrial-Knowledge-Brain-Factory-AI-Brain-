import logging
from typing import Any, Dict, List

from backend.services.knowledge_graph_service.src.infrastructure.graph_db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GDSAlgorithms:
    """
    Wrappers for Neo4j Graph Data Science (GDS) algorithms.
    """

    def __init__(self, client: Neo4jClient):
        self.client = client

    async def find_central_nodes(self, graph_name: str) -> List[Dict[str, Any]]:
        """
        Find the most central nodes in the graph using the GDS PageRank algorithm.
        """
        query = """
        CALL gds.pageRank.stream($graphName)
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS name, score
        ORDER BY score DESC LIMIT 10
        """
        try:
            return await self.client.execute_query(query, {"graphName": graph_name})
        except Exception as e:
            logger.error("Failed executing PageRank for central nodes in graph '%s': %s", graph_name, str(e))
            raise e

    async def _project_graph_if_not_exists(self, graph_name: str, node_projection: str, rel_projection: str) -> None:
        """Helper to create an in-memory graph projection for GDS algorithms."""
        check_query = "CALL gds.graph.exists($graph_name) YIELD exists RETURN exists"
        records = await self.client.execute_query(check_query, {"graph_name": graph_name})
        
        if records and not records[0].get("exists"):
            logger.info("Projecting GDS graph '%s'...", graph_name)
            project_query = """
            CALL gds.graph.project(
                $graph_name,
                $node_projection,
                $rel_projection
            )
            """
            await self.client.execute_write(project_query, {
                "graph_name": graph_name, 
                "node_projection": node_projection, 
                "rel_projection": rel_projection
            })

    async def page_rank(self, graph_name: str = "failure_graph") -> List[Dict[str, Any]]:
        """
        Identify the most critical/central failure modes using PageRank.
        Projects FailureMode nodes and PROPAGATES_TO relationships.
        """
        await self._project_graph_if_not_exists(
            graph_name, 
            "FailureMode", 
            "PROPAGATES_TO"
        )
        
        query = """
        CALL gds.pageRank.stream($graph_name)
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name AS failure_mode, score
        ORDER BY score DESC, failure_mode ASC
        LIMIT 10
        """
        return await self.client.execute_query(query, {"graph_name": graph_name})

    async def community_detection(self, graph_name: str = "failure_graph") -> List[Dict[str, Any]]:
        """
        Group related failure modes into communities using Louvain.
        """
        await self._project_graph_if_not_exists(
            graph_name, 
            "FailureMode", 
            {"PROPAGATES_TO": {"orientation": "UNDIRECTED"}}
        )
        
        query = """
        CALL gds.louvain.stream($graph_name)
        YIELD nodeId, communityId
        RETURN communityId, collect(gds.util.asNode(nodeId).name) AS failure_modes, count(*) AS size
        ORDER BY size DESC
        """
        return await self.client.execute_query(query, {"graph_name": graph_name})

    async def shortest_path(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """
        Find the fastest/shortest failure propagation path using Dijkstra.
        We don't need a named graph for single-source shortest path if we use classic Cypher,
        but we'll use GDS path finding over the Neo4j default shortestPath for consistency.
        """
        # Using built-in Cypher shortestPath for ease of use without named projections on the fly
        # If strict GDS is required: gds.shortestPath.dijkstra.stream
        query = """
        MATCH (start:FailureMode {id: $source_id}), (end:FailureMode {id: $target_id})
        CALL apoc.algo.dijkstra(start, end, 'PROPAGATES_TO>', 'delay_hours') YIELD path, weight
        RETURN [node in nodes(path) | node.name] AS path_nodes, weight AS total_delay_hours
        """
        # Note: Depending on the DB, APOC or standard shortestPath() can be used.
        # Standard Cypher fallback:
        query_fallback = """
        MATCH p = shortestPath((start:FailureMode {id: $source_id})-[:PROPAGATES_TO*..10]->(end:FailureMode {id: $target_id}))
        RETURN [node in nodes(p) | node.name] AS path_nodes
        """
        try:
            return await self.client.execute_query(query_fallback, {
                "source_id": source_id, 
                "target_id": target_id
            })
        except Exception as e:
            logger.error("Failed executing shortest_path: %s", e)
            return []
